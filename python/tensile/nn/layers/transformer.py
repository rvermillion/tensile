#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

from ..module import *
from ..attention import Attention
from .mlp import MLP


class DecoderLayerArgs(ModuleArgs):

    kind: Annotated[str, field(inherit='layer_kind')] = 'transformer'
    hidden_dim: int

    attention: Attention.Args
    # intermediate_size: int
    # rms_norm_eps: float
    # num_attention_heads: int
    # head_dim: Optional[int] = None
    # num_key_value_heads: Optional[int] = None
    # attention_bias: bool = False
    mlp: MLP.Args
    # layernorm: Normalization.Args
    # input_layernorm: Annotated[Normalization.Args, field(
    #     doc="The input normalization layer of the decoder layer",
    #     aliases=['layernorm'],
    # )]
    # post_attention_layernorm: Annotated[Normalization.Args, field(
    #     doc="The post-attention normalization layer of the decoder layer",
    #     aliases=['layernorm'],
    # )]
    # post_mlp_layernorm: Annotated[Normalization.Args, field(
    #     doc="The post-attention normalization layer of the decoder layer",
    #     aliases=['layernorm'],
    # )]
    # mlp_bias: bool = False
    # rope_theta: float = 10000
    # rope_traditional: bool = False
    # rope_scaling: Optional[dict[str, Union[float, str]]] = None


class DecoderLayer(CompiledModule):

    __slots__ = ('hidden_dim', 'attention', 'mlp',)

    # num_attention_heads: Annotated[int, field(
    #     doc="The number of attention heads in the layer",
    # )]
    hidden_dim: Annotated[int, field(
        doc="The hidden size of the layer",
    )]
    attention: Annotated[Attention, field(
        doc="The attention module of the decoder layer",
        changed=CompiledModule.recompile,
    )]
    mlp: Annotated[MLP, field(
        doc="The MLP module of the decoder layer",
        changed=CompiledModule.recompile,
    )]

    def init_from_args(self, args: DecoderLayerArgs):
        super().init_from_args(args)

        # num_attention_heads = args.num_attention_heads
        hidden_dim = args.hidden_dim

        # self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim

        self.attention = self.init_attention(args)
        self.mlp = self.init_mlp(args)
        # self.input_layernorm = self.build_input_layernorm(args)
        # self.post_attention_layernorm = self.build_post_attention_layernorm(args)
        # self.post_mlp_layernorm = self.build_post_mlp_layernorm(args)

    @property
    def in_dim(self) -> int:
        return self.hidden_dim

    @property
    def out_dim(self) -> int:
        return self.hidden_dim

    # noinspection PyMethodMayBeStatic
    def init_attention(self, args: DecoderLayerArgs) -> Attention:
        return Attention.from_args(args.attention)
        # return Attention.from_args(args.attention)

    def init_mlp(self, args: DecoderLayerArgs) -> MLP:
        mlp_args = args.mlp.set_defaults(
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim,
        )
        return MLP.from_args(mlp_args)
        # mlp_args = args.mlp.set_defaults(
        #     in_dim=self.hidden_size,
        #     out_dim=self.hidden_size,
        # )
        # return MLP.from_args(mlp_args)

    default_weight_aliases = {
        'self_attn': 'attention.body',
        'input_layernorm': 'attention.pre_norm',
        'post_attention_layernorm': 'mlp.pre_norm',
        'post_mlp_layernorm': 'mlp.post_norm',
    }


    Args = DecoderLayerArgs


meta.for_class(DecoderLayer).configure_registry(
    modules='tensile.models',
    append_kind=True,
    default_kind='transformer'
)


@meta.provides(Module, 'transformer-block')
@meta.provides(DecoderLayer, 'transformer')
class TransformerBlock(DecoderLayer, FunctionModule):

    __slots__ = ()

    def build_call(self, mode: CompiledModule.Mode, **options) -> Functional:
        attention = self.attention
        mlp = self.mlp
        add_residual = self.subinstrument('add_residual', ten.add, mode)

        def call(x: Array, /) -> Array:
            r = attention(x)
            h = add_residual(x, r)
            r = mlp(h)
            out = add_residual(h, r)
            return out

        return call
