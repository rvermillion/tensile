#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..module import CompiledModule, ModuleArgs, Module
from ..util import nilpotent
from .linear import LinearArgs


class MLPArgs(ModuleArgs):

    in_dim: int = None
    hidden_dim: int = None
    out_dim: int = None
    up_proj: Annotated[LinearArgs, field(
        doc='The arguments for the up projection',
    )]
    down_proj: Annotated[LinearArgs, field(
        doc='The arguments for the up projection',
    )]
    gate_proj: Annotated[LinearArgs, field(
        doc='The arguments for the up projection',
    )]
    inner_proj: Annotated[LinearArgs, field(
        doc='The arguments for the projection in any inner layers',
    )]
    layers: Annotated[int, field(
        default=1,
    )]
    bias: Annotated[bool, field(
        doc='Whether the MLP should have a bias',
        inherit='mlp_bias',
    )] = False
    activation: Annotated[str, field(
        doc='The activation function to use',
        inherit='hidden_act',
    )] = None


@provides(Module, 'mlp')
class MLP(CompiledModule):

    __slots__ = ('in_dim', 'out_dim', 'hidden_dim', 'activation', 'up_proj', 'down_proj', 'bias')

    args: Annotated[MLPArgs, field(ignore=True)]
    in_dim: Annotated[int, field(
        doc='The input dimension of the MLP',
    )]
    out_dim: Annotated[int, field(
        doc='The output dimension of the MLP',
    )]
    hidden_dim: Annotated[int, field(
        doc='The hidden dimension of the MLP',
    )]
    activation: Annotated[Activation, field(
        doc='The activation function to use',
    )]
    bias: Annotated[bool, field(
        doc='Whether to use bias in the MLP projections',
    )]
    up_proj: Annotated[Module, field(
        doc='The up projection module of the MLP',
    )]
    down_proj: Annotated[Module, field(
        doc='The down projection module of the MLP',
    )]

    def init_from_args(self, args: MLPArgs):
        super().init_from_args(args)

        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.hidden_dim = args.hidden_dim
        self.bias = args.bias

        self.up_proj = self.build_up_proj(bias=self.bias)
        self.down_proj = self.build_down_proj(bias=self.bias)
        self.activation = self.build_activation()

    def build_call(self, train: bool = False, **options) -> Callable:
        up_proj = self.up_proj
        down_proj = self.down_proj
        activation = self.activation

        if down_proj is None:
            def call(x: Array) -> Array:
                return activation(up_proj(x))
        else:
            def call(x: Array) -> Array:
                return down_proj(activation(up_proj(x)))

        return call

    default_activation_spec: ClassVar[str] = 'silu'

    def build_activation(self) -> Activation:
        """
        Build the activation module of the MLP.
        """
        return coerce(Activation, kind=self.args.activation or self.default_activation_spec)

    def build_up_proj(self, bias: bool = False, name: str = 'up_proj') -> Module:
        """
        Build the up projection module of the MLP.
        """
        args = self.args.up_proj.set_defaults(
            input_dims=self.in_dim,
            output_dims=self.hidden_dim,
            bias=bias,
        )
        return self.build_proj_from_args(args)

    def build_down_proj(self, bias: bool = False, name: str = 'down_proj') -> Module:
        """
        Build the down projection module of the MLP.
        """
        args = self.args.down_proj.set_defaults(
            input_dims=self.hidden_dim,
            output_dims=self.out_dim,
            bias=bias,
        )
        return self.build_proj_from_args(args)

    def _extra_structure(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, hidden_dim={self.hidden_dim}, '

    def nilpotent(self, scale: float = None, precision: DType = None):
        nilpotent(self.down_proj)

    Args = MLPArgs


@provides(Module, 'mlp.glu')
@provides(MLP, 'glu')
class GatedLinear(MLP):

    __slots__ = ('gate_proj',)

    gate_proj: Annotated[Module, field(
        doc='The gate projection module of the MLP',
    )]

    def init_from_args(self, args: MLPArgs):
        super().init_from_args(args)

        self.gate_proj = self.build_gate_proj(bias=self.bias)

    def build_call(self, train: bool = False, **options) -> Callable:
        up_proj = self.up_proj
        down_proj = self.down_proj
        gate_proj = self.gate_proj
        activation = self.activation

        if down_proj is None:
            def call(x: Array) -> Array:
                # ten.eval(x)
                return activation(gate_proj(x)) * up_proj(x)
        else:
            def call(x: Array) -> Array:
                # ten.eval(x)
                return down_proj(activation(gate_proj(x)) * up_proj(x))

        return call

    default_activation_spec: ClassVar[str] = 'silu'

    def build_gate_proj(self, bias: bool = False, name: str = 'gate_proj') -> Module:
        """
        Build the gate projection module of the MLP.
        """
        args = self.args.gate_proj.set_defaults(
            input_dims=self.in_dim,
            output_dims=self.hidden_dim,
            bias=bias,
        )
        return self.build_proj_from_args(args)
        # return self.build_proj(self.in_dim, self.hidden_dim, bias=bias, name=name)

    def _extra_structure(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, hidden_dim={self.hidden_dim}, '



__all__ = [
    'MLP',
    'MLPArgs',
]