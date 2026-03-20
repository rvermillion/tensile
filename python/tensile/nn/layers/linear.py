#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from typing import overload

import math

from .normalization import NormedModule
from ..common import *
from ..module import CompiledModule, FunctionModule, Module, ModuleArgs
from ..init import Initializer, Initializers
from ..activations import Activation
from ..quantization import QuantizableModuleArgs


class LinearArgs(QuantizableModuleArgs):
    in_dim: int
    out_dim: int
    bias: bool = True
    num_experts: int = 1
    initialize: Initializer = None


class BaseLinear(CompiledModule):
    r"""Applies an affine transformation to the input.

    Concretely:

    .. math::

        y = x W^\top + b

    where:
    where :math:`W` has shape ``[out_dim, in_dim]`` and :math:`b` has shape ``[out_dim]``.

    The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
    where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``in_dim``.

    Args:
        in_dim (int): The dimensionality of the input features
        out_dim (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will
          not use a bias. Default is ``True``.
    """

    __slots__ = ('weight', 'bias')

    weight: Annotated[Array, field(
        doc="The learnable weights of the layer.",
        parameter=True,
    )]
    bias: Annotated[Optional[Array], field(
        doc="The learnable bias of the layer.",
        parameter=True,
    )]

    def init_from_args(self, args: LinearArgs) -> None:
        super().init_from_args(args)

        self.weight, self.bias = self.init_weight_and_bias(args)

    def init_weight_and_bias(self, args: LinearArgs) -> tuple[Array, Optional[Array]]:
        return self.init_weight(args), self.init_bias(args)

    def init_weight(self, args: LinearArgs) -> Array:
        init = args.initialize or Initializers.uniform
        in_dim = args.in_dim
        out_dim = args.out_dim
        scale = math.sqrt(1.0 / in_dim)

        return init((out_dim, in_dim), scale=scale)

    def init_bias(self, args: LinearArgs) -> Optional[Array]:
        if args.bias:
            init = args.initialize or Initializers.uniform
            in_dim = args.in_dim
            out_dim = args.out_dim
            scale = math.sqrt(1.0 / in_dim)
            return init((out_dim, ), scale=scale)
        return None

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        if mode.is_compiled():
            # In compiled mode, we can close over the parameters

            weight_t = self.weight.T

            if self.bias is None:
                def call(x: Array) -> Array:
                    return ten.matmul(x, weight_t)
            else:
                bias = self.bias

                def call(x: Array) -> Array:
                    return ten.addmm(bias, x, weight_t)
        else:
            if self.bias is None:
                def call(x: Array) -> Array:
                    return ten.matmul(x, self.weight.T)
            else:
                # noinspection PyTypeChecker
                def call(x: Array) -> Array:
                    return ten.addmm(self.bias, x, self.weight.T)

        return call

    @property
    def in_dim(self) -> int:
        return 0 if (w := self.weight) is None else w.shape[-1]

    @property
    def out_dim(self) -> int:
        return 0 if (w := self.weight) is None else w.shape[-2]

    def _extra_structure(self):
        return (
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"bias={self.bias is not None}"
        )

    Args = LinearArgs


@meta.provides(Module, 'linear')
class Linear(BaseLinear, FunctionModule):

    __slots__ = ()

    @classmethod
    def refine_implementation(cls, args: LinearArgs) -> type['Linear']:
        if args.quantization.group_size > 0:
            return QuantizedLinear
        return cls

    @overload
    @classmethod
    def fuse(cls, a: 'Linear', b: 'Linear') -> Callable[[Array], tuple[Array, Array]]: ...

    @overload
    @classmethod
    def fuse(cls, a: 'Linear', b: 'Linear', c: 'Linear') -> Callable[[Array], tuple[Array, Array, Array]]: ...

    @classmethod
    def fuse(cls, *linears: 'Linear') -> Callable[[Array], tuple[Array, ...]]:
        if len(linears) <= 1:
            raise ValueError("Cannot fuse less than two linear layers")

        in_dim = linears[0].in_dim
        if any(lin.in_dim != in_dim for lin in linears[1:]):
            raise ValueError(f"Cannot fuse linear layers with different in_dims: " +
                             ', '.join(str(lin.in_dim) for lin in linears))

        if any(lin.bias is not None for lin in linears):
            if not all(lin.bias is not None for lin in linears):
                raise ValueError("Cannot fuse linear layers with biases")
            # noinspection PyTypeChecker
            bias = ten.concatenate([lin.bias for lin in linears], axis=0)
        else:
            bias = None

        split_pos = 0
        splits = []
        for lin in linears[:-1]:
            split_pos += lin.weight.shape[0]
            splits.append(split_pos)

        weight = ten.concatenate([lin.weight for lin in linears], axis=0)

        if bias is None:
            def fused_linear(x: Array) -> tuple[Array, ...]:
                y = ten.matmul(x, weight.T)
                return ten.split(y, splits, axis=-1)
        else:
            def fused_linear(x: Array) -> tuple[Array, ...]:
                y = ten.addmm(bias, x, weight.T)
                return ten.split(y, splits, axis=-1)

        return fused_linear


@provides(Linear, 'quantized')
class QuantizedLinear(Linear):
    """Applies an affine transformation to the input using a quantized weight matrix.

    It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
    parameters are frozen and will not be included in any gradient computation
    but this will probably change in the future.

    :obj:`QuantizedLinear` also provides a classmethod :meth:`from_linear` to
    convert linear layers to :obj:`QuantizedLinear` layers.

    Args:
        in_dim (int): The dimensionality of the input features.
        out_dim (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. Default: ``True``.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
    """

    __slots__ = ('scales', 'biases', 'group_size', 'mode', 'bits')

    scales: Annotated[Array, field(
        doc="The learnable scales of the layer.",
        parameter=False,
    )]
    biases: Annotated[Optional[Array], field(
        doc="The learnable biases of the layer.",
        parameter=False,
    )]
    group_size: Annotated[int, field(
        doc="The group size to use for the quantized weight.",
        default=64,
    )]
    bits: Annotated[int, field(
        doc="The number of bits to use for quantization.",
        default=8,
    )]
    mode: Annotated[str, field(
        doc="The quantization mode to use. Currently only ``affine`` is supported.",
        default="affine",
    )]

    keep_frozen = True

    def init_from_args(self, args: LinearArgs) -> None:
        super().init_from_args(args)

        # init = args.initialize or Initializers.uniform
        # in_dim = args.in_dim
        # out_dim = args.out_dim
        # scale = math.sqrt(1.0 / in_dim)
        #
        # weight = init((out_dim, in_dim), scale=scale)
        # if args.bias:
        #     self.bias = init((out_dim, ), scale=scale)
        # else:
        #     self.bias = None

        quant = args.quantization

        if quant is None:
            raise ValueError()

        # Quantization config
        self.group_size = quant.get("group_size", 64)
        self.bits = quant.get("bits", 8)
        self.mode = quant.get("mode", "affine")

        # Initialize the quantized weight
        self.weight, self.scales, *biases = ten.quantize(self.weight, self.group_size, self.bits, mode=self.mode)
        self.biases = biases[0] if biases else None

    @property
    def in_dim(self) -> int:
        return self.weight.shape[1] * (32 // self.bits)

    @property
    def out_dim(self) -> int:
        return self.weight.shape[0]

    def _extra_structure(self):
        return super()._extra_structure() + f"group_size={self.group_size}, bits={self.bits}"

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        group_size = self.group_size
        bits = self.bits
        quant_mode = self.mode

        if self.bias is None:
            def call(x: Array) -> Array:
                # ten.debug_eval(x)
                return ten.quantized_matmul(
                    x,
                    self.weight,
                    scales=self.scales,
                    biases=self.biases,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode=quant_mode,
                )
        else:
            def call(x: Array) -> Array:
                # ten.debug_eval(x)
                return ten.quantized_matmul(
                    x,
                    self.weight,
                    scales=self.scales,
                    biases=self.biases,
                    transpose=True,
                    group_size=group_size,
                    bits=bits,
                    mode=quant_mode,
                ) + self.bias

        return call


# TODO: this is not a FunctionModule because it returns a tuple of Arrays....
@provides(Linear, 'prefused')
class PrefusedLinear(Linear):

    __slots__ = ('splits',)

    splits: Annotated[list[int], field(
        doc="Whether to split the output of the fused linear layers",
        default=True,
    )]

    def init_from_args(self, args: LinearArgs) -> None:
        super().init_from_args(args)

        out_dim = self.out_dim

        splits = args.get('splits', 2)

        if isinstance(splits, int):
            if out_dim % splits != 0:
                raise ValueError(f'Cannot divide {out_dim} into {splits} even sections')
            size = out_dim // splits
            splits = list(range(size, out_dim, size))
        elif isinstance(splits, list):
            if any(split >= out_dim for split in splits):
                raise ValueError(f'Split ends cannot be greater that out_dim {out_dim}: {splits}')
        else:
            raise ValueError(f'Unexpected value for splits: {splits!r}')

        self.splits = splits


    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable[[Array], tuple[Array, ...]]:
        splits = self.splits

        if self.bias is None:
            def call(x: Array) -> tuple[Array, ...]:
                y = ten.matmul(x, self.weight.T)
                return ten.split(y, splits, axis=-1)
        else:
            def call(x: Array) -> tuple[Array, ...]:
                y = ten.addmm(self.bias, x, self.weight.T)
                return ten.split(y, splits, axis=-1)

        return call


@provides(Linear, 'fused')
class FusedLinear(Linear):

    __slots__ = ('fused', 'split_output')

    fused: Annotated[tuple[Linear, ...], field(
        doc="Tuple of linear layers to fuse",
    )]
    split_output: Annotated[bool, field(
        doc="Whether to split the output of the fused linear layers",
        default=True,
    )]

    def init_weight_and_bias(self, args: LinearArgs) -> tuple[Array, Optional[Array]]:
        fused = self.fused
        if len(fused) <= 1:
            raise ValueError("Cannot fuse less than two linear layers")

        in_dim = fused[0].in_dim
        if any(lin.in_dim != in_dim for lin in fused[1:]):
            raise ValueError(f"Cannot fuse linear layers with different in_dims: " +
                             ', '.join(str(lin.in_dim) for lin in fused))

        if any(lin.bias is not None for lin in fused):
            if not all(lin.bias is not None for lin in fused):
                raise ValueError("Cannot fuse linear layers with biases")
            bias = ten.concatenate([lin.bias for lin in fused], axis=0)
        else:
            bias = None

        weight = ten.concatenate([lin.weight for lin in fused], axis=0)

        return weight, bias

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        fused = self.fused

        if self.split_output:
            split_pos = 0
            splits = []
            for lin in fused[:-1]:
                split_pos += lin.weight.shape[0]
                splits.append(split_pos)


            if self.bias is None:
                def fused_linear(x: Array) -> Sequence[Array]:
                    y = ten.matmul(x, self.weight.T)
                    return ten.split(y, splits, axis=-1)
            else:
                def fused_linear(x: Array) -> Sequence[Array]:
                    y = ten.addmm(self.bias, x, self.weight.T)
                    return ten.split(y, splits, axis=-1)
        else:
            if self.bias is None:
                def fused_linear(x: Array) -> Array:
                    return ten.matmul(x, self.weight.T)
            else:
                def fused_linear(x: Array) -> Array:
                    return ten.addmm(self.bias, x, self.weight.T)

        return fused_linear



class GatedLinearArgs(ModuleArgs):

    in_dim: int = None
    out_dim: int = None
    bias: Annotated[bool, field(
        doc='Whether the MLP should have a bias',
        inherit='glu_bias',
    )] = False
    activation: Annotated[str, field(
        doc='The activation function to use',
        inherit='glu_activation',
    )] = None



@provides(Module, 'glu')
class GatedLinear(CompiledModule):

    __slots__ = ('in_dim', 'out_dim', 'activation', 'gate_proj', 'up_proj', 'bias')

    args: Annotated[GatedLinearArgs, field(ignore=True)]
    in_dim: int
    out_dim: int
    activation: Activation

    gate_proj: Module
    up_proj: Module
    bias: bool

    def init_from_args(self, args: GatedLinearArgs):
        super().init_from_args(args)

        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.bias = args.bias

        self.gate_proj = self.build_gate_proj(bias=self.bias)
        self.up_proj = self.build_up_proj(bias=self.bias)
        self.activation = self.build_activation()

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        activation = self.activation
        gate_proj = self.gate_proj
        up_proj = self.up_proj

        def call(x: Array) -> Array:
            # ten.eval(x)
            return activation(gate_proj(x)) * up_proj(x)
        return call

    default_activation_spec: ClassVar[str] = 'silu'

    def build_activation(self) -> Activation:
        return coerce(Activation, kind=self.args.activation or self.default_activation_spec)

    def build_gate_proj(self, bias: bool = False, name: str = 'gate_proj') -> Module:
        return self.build_proj(self.in_dim, self.hidden_dim, bias=bias, name=name)

    def build_up_proj(self, bias: bool = False, name: str = 'up_proj') -> Module:
        return self.build_proj(self.in_dim, self.hidden_dim, bias=bias, name=name)

    Args = GatedLinearArgs


provides(Linear, 'normed')(NormedModule)
