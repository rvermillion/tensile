#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import math

from ..common import *
from ..init import Initializers
from ..module import Module, CompiledModule
from .linear import BaseLinear, Linear, LinearArgs
from .mlp import BaseMLP, GatedLinear, MLP, MLPArgs


class SwitchFunction(Protocol):

    def __call__(self, x: Array, indices: Array, /, sorted_indices: Optional[bool] = None) -> Array: ...



@provides(Module, 'switch')
class SwitchModule(CompiledModule):

    __slots__ = ()

    num_experts: Annotated[int, field(
        doc="The number of experts to switch between",
        ignore=True
    )]

    if TYPE_CHECKING:
        def build_call(self, mode: CompiledModule.Mode, **options) -> SwitchFunction: ...

        # noinspection PyFinal
        def __call__(self, x: Array, indices: Array, /, sorted_indices: Optional[bool] = None) -> Array: ...


@provides(SwitchModule, 'linear')
class SwitchLinear(BaseLinear, SwitchModule):

    __slots__ = ('num_experts',)

    num_experts: Annotated[int, field(
        doc="The number of experts to switch between",
    )]

    def init_from_args(self, args: LinearArgs) -> None:
        self.num_experts = args.num_experts
        super().init_from_args(args)

    def init_weight(self, args: LinearArgs) -> Array:
        init = args.initialize or Initializers.uniform
        in_dim = args.in_dim
        scale = math.sqrt(1 / in_dim)
        return init((self.num_experts, args.out_dim, in_dim), scale=scale)

    def init_bias(self, args: LinearArgs) -> Optional[Array]:
        if args.bias:
            init = args.initialize or Initializers.uniform
            in_dim = args.in_dim
            scale = math.sqrt(1.0 / in_dim)
            return init((self.num_experts, args.out_dim, ), scale=scale)
        return None

    def build_call(self, mode: CompiledModule.Mode, **options) -> SwitchFunction:
        if self.bias is None:
            def call(x: Array, indices: Array, /, sorted_indices: Optional[bool] = None) -> Array:
                x = ten.gather_mm(
                    x,
                    ten.swapaxes(self.weight, -1, -2),
                    rhs_indices=indices,
                    sorted_indices=bool(sorted_indices),
                )
                return x
        else:
            def call(x: Array, indices: Array, /, sorted_indices: Optional[bool] = None) -> Array:
                x = ten.gather_mm(
                    x,
                    ten.swapaxes(self.weight, -1, -2),
                    rhs_indices=indices,
                    sorted_indices=bool(sorted_indices),
                )
                x = x + ten.expand_dims(self.bias[indices], -2)
                return x


        return call

    # def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
    #     num_experts, output_dims, input_dims = self.weight.shape
    #     ql = QuantizedSwitchLinear(
    #         input_dims,
    #         output_dims,
    #         num_experts,
    #         False,
    #         group_size,
    #         bits,
    #         mode=mode,
    #     )
    #     ql.weight, ql.scales, *biases = mx.quantize(
    #         self.weight, group_size, bits, mode=mode
    #     )
    #     ql.biases = biases[0] if biases else None
    #
    #     if "bias" in self:
    #         ql.bias = self.bias
    #     return ql


@provides(SwitchModule, 'glu')
class SwitchGLU(BaseMLP, SwitchModule):

    __slots__ = ('num_experts', )

    num_experts: Annotated[int, field(
        doc="The number of experts to switch between",
    )]
    gate_proj: Annotated[Module, field(
        doc='The gate projection module of the MLP',
    )]
    up_proj: Annotated[Module, field(
        doc='The up projection module of the MLP',
    )]

    def init_from_args(self, args: MLPArgs):
        self.num_experts = args.get('num_experts')
        super().init_from_args(args)

        self.gate_proj = self.init_gate_proj(bias=self.bias)
        self.up_proj = self.init_up_proj(bias=self.bias)

    def init_gate_proj(self, bias: bool = False, name: str = 'gate_proj') -> Module:
        args = self.args.gate_proj.set_defaults(
            in_dim=self.in_dim,
            out_dim=self.hidden_dim,
            num_experts=self.num_experts,
            bias=bias,
        )
        return SwitchLinear.from_args(args)

    def init_up_proj(self, bias: bool = False, name: str = 'up_proj') -> Module:
        args = self.args.up_proj.set_defaults(
            in_dim=self.in_dim,
            out_dim=self.hidden_dim,
            num_experts=self.num_experts,
            bias=bias,
        )
        return SwitchLinear.from_args(args)

    def init_down_proj(self, bias: bool = False, name: str = 'down_proj') -> Module:
        args = self.args.up_proj.set_defaults(
            in_dim=self.hidden_dim,
            out_dim=self.out_dim,
            num_experts=self.num_experts,
            bias=bias,
        )
        return SwitchLinear.from_args(args)

    def build_gate_up_proj(self, mode: CompiledModule.Mode) -> Callable[[Array], tuple[Array, ...]]:
        up_proj = self.up_proj
        gate_proj = self.gate_proj

        if mode.is_compiled():
            if isinstance(gate_proj, Linear) and isinstance(up_proj, Linear):
                return Linear.fuse(gate_proj, up_proj)

        def project_gate_up(x: Array) -> tuple[Array, Array]:
            return gate_proj(x), up_proj(x)

        return project_gate_up

    def build_hidden(self, mode: CompiledModule.Mode) -> Callable[[Array], Array]:
        gate_up_proj = self.build_gate_up_proj(mode)
        activation = self.build_activation(mode)

        def hidden(x: Array) -> Array:
            gate, up = gate_up_proj(x)
            return activation(gate) * up

        return hidden

    def build_call(self, mode: CompiledModule.Mode, **options) -> SwitchFunction:
        up_proj = self.up_proj
        gate_proj = self.gate_proj
        down_proj = self.down_proj
        activation = self.activation

        def call(x: Array, indices: Array, /, sorted_indices: Optional[bool] = None) -> Array:
            x = ten.expand_dims(x, (-2, -3))

            # When we have many tokens, then sort them to make sure that the access
            # of different experts is in order.
            if sorted_indices is None:
                do_sort = sorted_indices = indices.size >= 64
            else:
                do_sort = False
            idx = indices
            inv_order = None
            if do_sort:
                x, idx, inv_order = _gather_sort(x, indices)

            if mode.is_train():
                idx = ten.stop_gradient(idx)

            x_up = up_proj(x, idx, sorted_indices=sorted_indices)
            x_gate = gate_proj(x, idx, sorted_indices=sorted_indices)
            x = down_proj(
                x_up * activation(x_gate),
                idx,
                sorted_indices=sorted_indices,
            )

            if do_sort:
                x = _scatter_unsort(x, inv_order, indices.shape)

            return x.squeeze(-2)

        return call


# Adapted from mlx_lm.models.switch_layers
# Copyright © 2023-2024 Apple Inc.

def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = ten.argsort(indices)
    inv_order = ten.argsort(order)
    return ten.flatten(x, 0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = ten.unflatten(x, 0, shape)
    return x
