#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from .normalization import NormedModule
from ..common import *
from ..module import CompiledModule, FunctionModule, ModuleArgs, Module
from ..util import nilpotent
from .linear import Linear, LinearArgs
from ...infra.function import identity


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
    gate_up_proj: Annotated[LinearArgs, field(
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
    fused: Annotated[bool, field(
        doc='Whether the MLP should be fused, i.e. the "fused" version of the MLP',
    )]


class BaseMLP(CompiledModule):

    __slots__ = ('in_dim', 'out_dim', 'hidden_dim', 'activation', 'bias', 'down_proj')

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
    down_proj: Annotated[Module, field(
        doc='The down projection module of the MLP',
    )]

    def init_from_args(self, args: MLPArgs):
        super().init_from_args(args)

        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.hidden_dim = args.hidden_dim
        self.bias = args.bias

        self.activation = self.init_activation()
        self.down_proj = self.init_down_proj(self.bias)

    default_activation_spec: ClassVar[str] = 'silu'

    def init_activation(self) -> Activation:
        """
        Build the activation module of the MLP.
        """
        return coerce(Activation, kind=self.args.activation or self.default_activation_spec)

    def init_up_proj(self, bias: bool = False, name: str = 'up_proj') -> Module:
        """
        Build the up projection module of the MLP.
        """
        args = self.args.up_proj.set_defaults(
            in_dim=self.in_dim,
            out_dim=self.hidden_dim,
            bias=bias,
        )
        return self.init_proj(args)

    def init_gate_proj(self, bias: bool = False, name: str = 'gate_proj') -> Module:
        """
        Build the up projection module of the MLP.
        """
        args = self.args.gate_proj.set_defaults(
            in_dim=self.in_dim,
            out_dim=self.hidden_dim,
            bias=bias,
        )
        return self.init_proj(args)

    def init_gate_up_proj(self, bias: bool = False, name: str = 'gate_up_proj') -> Module:
        """
        Build the up projection module of the MLP.
        """
        args = self.args.gate_up_proj.set_defaults(
            in_dim=self.in_dim,
            out_dim=2*self.hidden_dim,
            bias=bias,
            splits=2,
            kind='linear.prefused',
        )
        return Module.from_args(args)

    def init_down_proj(self, bias: bool = False, name: str = 'down_proj') -> Module:
        """
        Build the down projection module of the MLP.
        """
        args = self.args.down_proj.set_defaults(
            in_dim=self.hidden_dim,
            out_dim=self.out_dim,
            bias=bias,
        )
        return self.init_proj(args)

    def build_down_proj(self, mode: CompiledModule.Mode) -> Optional[Callable[[Array], Array]]:
        return self.down_proj

    def build_activation(self, mode: CompiledModule.Mode) -> Callable[[Array], Array]:
        return self.subinstrument('activation', self.activation, mode)

    def build_hidden(self, mode: CompiledModule.Mode) -> Callable[[Array], Array]:
        raise NotImplementedError()

    def build_call(self, mode: Module.Mode, **options) -> Callable[[Array], Array]:
        hidden = self.build_hidden(mode)
        down_proj = self.build_down_proj(mode)

        if down_proj is None:
            return hidden

        def call(x: Array) -> Array:
            return down_proj(hidden(x))

        return call

    def _extra_structure(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}, hidden_dim={self.hidden_dim}, '

    Args = MLPArgs


@provides(Module, 'mlp')
class MLP(BaseMLP, FunctionModule):

    __slots__ = ()


class GatedMLP(MLP):

    __slots__ = ()

    def build_gate_up_proj(self, mode: CompiledModule.Mode) -> Callable[[Array], tuple[Array, ...]]:
        raise NotImplementedError()

    def build_hidden(self, mode: CompiledModule.Mode) -> Callable[[Array], Array]:
        gate_up_proj = self.build_gate_up_proj(mode)
        activation = self.build_activation(mode)

        def hidden(x: Array) -> Array:
            gate, up = gate_up_proj(x)
            return activation(gate) * up

        return hidden


@provides(Module, 'mlp.glu')
@provides(MLP, 'glu')
class GatedLinear(GatedMLP):

    __slots__ = ('gate_proj', 'up_proj')

    gate_proj: Annotated[Module, field(
        doc='The gate projection module of the MLP',
    )]
    up_proj: Annotated[Module, field(
        doc='The up projection module of the MLP',
    )]

    def init_from_args(self, args: MLPArgs):
        super().init_from_args(args)

        self.gate_proj = self.init_gate_proj(bias=self.bias)
        self.up_proj = self.init_up_proj(bias=self.bias)

    def build_gate_up_proj(self, mode: CompiledModule.Mode) -> Callable[[Array], tuple[Array, ...]]:
        up_proj = self.up_proj
        gate_proj = self.gate_proj

        if mode.is_compiled():
            if isinstance(gate_proj, Linear) and isinstance(up_proj, Linear):
                return Linear.fuse(gate_proj, up_proj)

        def project_gate_up(x: Array) -> tuple[Array, Array]:
            return gate_proj(x), up_proj(x)

        return project_gate_up


@provides(Module, 'mlp.fused-glu')
@provides(MLP, 'fused-glu')
class FusedGatedLinear(GatedMLP):

    __slots__ = ('gate_up_proj',)

    gate_up_proj: Annotated[Module, field(
        doc='The fused gate and up projection module of the MLP',
    )]

    def init_from_args(self, args: MLPArgs):
        super().init_from_args(args)

        self.gate_up_proj = self.init_gate_up_proj(bias=self.bias)

    def build_gate_up_proj(self, mode: CompiledModule.Mode) -> Callable[[Array], tuple[Array, ...]]:
        return self.gate_up_proj



provides(MLP, 'normed')(NormedModule)



__all__ = [
    'GatedLinear',
    'MLP',
    'MLPArgs',
]