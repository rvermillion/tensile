#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

import math

from ..common import *
from ..module import CompiledModule, Module, ModuleArgs
from ..init import Initializer, Initializers
from ..activations import Activation
from ..quantization import QuantizableModuleArgs


class LinearArgs(QuantizableModuleArgs):
    input_dims: int
    output_dims: int
    bias: bool = True
    initialize: Initializer = None


@meta.provides(Module, 'linear')
class Linear(CompiledModule):
    r"""Applies an affine transformation to the input.

    Concretely:

    .. math::

        y = x W^\top + b

    where:
    where :math:`W` has shape ``[output_dims, input_dims]`` and :math:`b` has shape ``[output_dims]``.

    The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
    where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``input_dims``.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
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

        init = args.initialize or Initializers.uniform
        input_dims = args.input_dims
        output_dims = args.output_dims
        scale = math.sqrt(1.0 / input_dims)

        self.weight = init((output_dims, input_dims), scale=scale)
        if args.bias:
            self.bias = init((output_dims, ), scale=scale)
        else:
            self.bias = None

    def build_call(self, train: bool = False, **options) -> Callable:
        if self.bias is None:
            def call(x: Array) -> Array:
                return ten.matmul(x, self.weight.T)
        else:
            def call(x: Array) -> Array:
                return ten.addmm(self.bias, x, self.weight.T)

        return call

    @property
    def input_dims(self) -> int:
        return 0 if (w := self.weight) is None else w.shape[1]

    @property
    def output_dims(self) -> int:
        return 0 if (w := self.weight) is None else w.shape[0]

    @property
    def input_features(self) -> int:
        return self.input_dims

    @property
    def output_features(self) -> int:
        return self.output_dims

    def _extra_structure(self):
        return (
            f"input_dims={self.input_dims}, "
            f"output_dims={self.output_dims}, "
            f"bias={self.bias is not None}"
        )

    Args = LinearArgs

    @classmethod
    def refine_implementation(cls, args: LinearArgs) -> type[Self]:
        return QuantizedLinear if args.quantization.group_size > 0 else cls


class QuantizedLinear(CompiledModule):
    """Applies an affine transformation to the input using a quantized weight matrix.

    It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
    parameters are frozen and will not be included in any gradient computation
    but this will probably change in the future.

    :obj:`QuantizedLinear` also provides a classmethod :meth:`from_linear` to
    convert linear layers to :obj:`QuantizedLinear` layers.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. Default: ``True``.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
    """

    __slots__ = ('weight', 'bias', 'scales', 'biases', 'group_size', 'mode', 'bits')

    weight: Annotated[Array, field(
        doc="The learnable weights of the layer.",
        parameter=False,
    )]
    bias: Annotated[Optional[Array], field(
        doc="The learnable bias of the layer.",
        parameter=False,
    )]
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

        init = args.initialize or Initializers.uniform
        input_dims = args.input_dims
        output_dims = args.output_dims
        scale = math.sqrt(1.0 / input_dims)

        weight = init((output_dims, input_dims), scale=scale)
        if args.bias:
            self.bias = init((output_dims, ), scale=scale)
        else:
            self.bias = None

        quant = args.quantization

        if quant is None:
            raise ValueError()

        # Quantization config
        self.group_size = quant.get("group_size", 64)
        self.bits = quant.get("bits", 8)
        self.mode = quant.get("mode", "affine")

        # Initialize the quantized weight
        self.weight, self.scales, *biases = ten.quantize(weight, self.group_size, self.bits, mode=self.mode)
        self.biases = biases[0] if biases else None

    @property
    def input_dims(self) -> int:
        return self.weight.shape[1] * (32 // self.bits)

    @property
    def output_dims(self) -> int:
        return self.weight.shape[0]

    @property
    def input_features(self) -> int:
        return self.input_dims

    @property
    def output_features(self) -> int:
        return self.output_dims

    def _extra_repr(self):
        return super()._extra_repr() + f"group_size={self.group_size}, bits={self.bits}"

    # def build_call(self, train: bool) -> Callable:
    #
    # def __call__(self, x: Array) -> Array:
    #     y = ten.quantized_matmul(
    #         x,
    #         self.weight,
    #         scales=self.scales,
    #         biases=self.biases,
    #         transpose=True,
    #         group_size=self.group_size,
    #         bits=self.bits,
    #         mode=self.mode,
    #     )
    #     if self.bias is not None:
    #         y = y + self.bias
    #     # if native := self.native:
    #     #     yn = native(x)
    #     #     ten.eval(y, yn)
    #     #     if not ten.allclose(y, yn):
    #     #         print(f"Quantized linear and native linear outputs differ: {ten.norm(y - yn):.4f}")
    #     #         print(f"weight: {ten.norm(self.weight - native.weight):.4f}")
    #     #         if self.bias is not None:
    #     #             print(f"bias: {ten.norm(self.bias - native.bias):.4f}")
    #     #         print(f"scales: {ten.norm(self.scales - native.scales):.4f}")
    #     #         print(f"biases: {ten.norm(self.biases - native.biases):.4f}")
    #     return y

    def build_call(self, train: bool = False, **options) -> Callable:
        group_size = self.group_size
        bits = self.bits
        mode = self.mode

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
                    mode=mode,
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
                    mode=mode,
                ) + self.bias

        return call


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

    def build_call(self, train: bool = False, **options) -> Callable:
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


