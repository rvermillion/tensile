#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

import tensile.nn

from ..common import *
from ..layers.linear import LinearArgs
from ..position import PositionEncoder
from ..module import CompiledModule, Module, ModuleArgs
from ..util import nilpotent
from .attend import Attend
from .util import reshape_qkv


class AttentionArgs(ModuleArgs):

    kind: Annotated[str, meta.field(inherit='attention_kind')] = 'standard'

    hidden_size: int
    num_attention_heads: int
    head_dim: Optional[int] = None
    q_head_dim: Optional[int] = None
    k_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    q_proj: Annotated[LinearArgs, field(
        doc='The arguments for the query projection',
    )]
    k_proj: Annotated[LinearArgs, field(
        doc='The arguments for the key projection',
    )]
    v_proj: Annotated[LinearArgs, field(
        doc='The arguments for the value projection',
    )]
    o_proj: Annotated[LinearArgs, field(
        doc='The arguments for the output projection',
    )]
    position_encoder: Annotated[PositionEncoder.Args, field(
        doc='The arguments for the position encoder',
    )]
    attend: Annotated[ModuleArgs, field(
        doc='The arguments for the attention mechanism',
    )]
    num_key_value_heads: Optional[int] = None
    bias: Annotated[bool, field(inherit='attention_bias')] = False

    def __post_init__(self):
        super().__post_init__()

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(CompiledModule):

    __slots__ = ('dim', 'in_dim', 'out_dim', 'n_heads', 'n_kv_heads', 'kv_heads_per_head', 'q_head_dim',
                 'k_head_dim', 'v_head_dim', 'scale', 'attend', 'encode_position',
                 'q_proj', 'k_proj', 'v_proj', 'o_proj')

    args: Annotated[AttentionArgs, field(ignore=True)]

    dim: int
    in_dim: int
    out_dim: int
    n_heads: int
    n_kv_heads: int
    kv_heads_per_head: int
    q_head_dim: int
    k_head_dim: int
    v_head_dim: int
    scale: float
    attend: Attend
    encode_position: PositionEncoder

    q_proj: Module
    k_proj: Module
    v_proj: Module
    o_proj: Module

    default_attention_bias: ClassVar[bool] = False

    # noinspection PyAttributeOutsideInit
    def init_from_args(self, args: AttentionArgs):
        super().init_from_args(args)

        dim = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim or dim // n_heads
        scale = head_dim**-0.5
        attention_bias = args.get('bias', default=self.default_attention_bias)

        self.dim = dim
        self.in_dim = dim
        self.out_dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_heads_per_head = n_heads // n_kv_heads
        self.q_head_dim = args.q_head_dim or head_dim
        self.v_head_dim = args.v_head_dim or head_dim
        self.k_head_dim = args.k_head_dim or head_dim
        self.scale = scale

        self.q_proj = self.build_q_proj(dim, bias=attention_bias)
        self.k_proj = self.build_k_proj(dim, bias=attention_bias)
        self.v_proj = self.build_v_proj(dim, bias=attention_bias)
        self.o_proj = self.build_o_proj(dim, bias=attention_bias)

        self.encode_position = self.build_position_encoder(args)
        self.attend = self.build_attend(args)

    def build_attend(self, args: AttentionArgs) -> Attend:
        attend_args = args.attend.set_defaults(
            kind='default'
        )
        return coerce(Attend, args=attend_args, kind=attend_args.kind)

    def build_q_proj(self, in_size: int, bias: bool = False, name: str = 'q_proj') -> Module:
        args = self.args.q_proj.set_defaults(
            input_dims=in_size,
            output_dims=self.n_heads * self.q_head_dim,
            bias=bias,
        )
        return self.build_proj_from_args(args)

    def build_k_proj(self, in_size: int, bias: bool = False, name: str = 'k_proj') -> Module:
        args = self.args.q_proj.set_defaults(
            input_dims=in_size,
            output_dims=self.n_kv_heads * self.k_head_dim,
            bias=bias,
        )
        return self.build_proj_from_args(args)

    def build_v_proj(self, in_size: int, bias: bool = False, name: str = 'v_proj') -> Module:
        args = self.args.q_proj.set_defaults(
            input_dims=in_size,
            output_dims=self.n_kv_heads * self.v_head_dim,
            bias=bias,
        )
        return self.build_proj_from_args(args)

    def build_o_proj(self, out_size: int, bias: bool = False, name: str = 'o_proj') -> Module:
        args = self.args.o_proj.set_defaults(
            input_dims=self.n_heads * self.v_head_dim,
            output_dims=out_size,
            bias=bias,
        )
        return self.build_proj_from_args(args)

    def build_position_encoder(self, args: AttentionArgs) -> PositionEncoder:
        encoder_args = args.position_encoder.set_defaults(
            kind='rope',
            dims=min(self.q_head_dim, self.k_head_dim)
        )
        return PositionEncoder.from_args(encoder_args)

    def nilpotent(self, scale: float = None, precision: DType = None):
        nilpotent(self.o_proj)

    Args = AttentionArgs


meta.for_class(Attention).configure_registry(
    modules='patchlm.models',
    append_kind=True,
    default_kind='standard'
)



ScoreExtra = Callable[[Array], dict[str, Array]]


@provides(Attention, 'standard')
class StandardAttention(Attention):

    __slots__ = ('score_extra',)

    score_extra: ScoreExtra

    def _lazy_score_extra(self) -> ScoreExtra:
        extra = {}
        def score_extra(x: Array) -> dict[str, Array]:
            return extra
        return score_extra

    # noinspection PyPep8Naming
    def build_call(self, train: bool = False, **options) -> Callable:
        q_encode_position = k_encode_position = self.encode_position
        q_proj = self.q_proj
        k_proj = self.k_proj
        v_proj = self.v_proj
        o_proj = self.o_proj

        scale = self.scale
        attend = self.attend
        score_extra = self.score_extra

        H = self.n_heads
        H_kv = self.n_kv_heads

        # noinspection PyPep8Naming
        def call(x: Array) -> Array:
            """

            What's the best way to implement a chunked scaled dot product attention calculation
            in a Metal kernel?

            Assume I have a query tensor, q, with shape (B, N_q, T_q, D) where B is the batch size,
            T_q is the length of the query sequence, N_q is the number of query heads, and D is the
            head dimension.

            Then, I have a list of key/value segments, with each key segment, k[i], and value
            segment, v[i], is a tensor of shape (B, N_kv, T_kv, D) where T_kv is the number of
            keys and values in the segment, and N_kv is the number of kv heads.

            For each key/value segment, i,  I need to compute the attention score:
               a[i] = exp(q @ k[i].T)
            which will have shape (B, B, L_q, L_kv).

            And

            :param x:
            :param cache:
            :return:
            """
            B, L, D = x.shape

            # Project the input into queries, keys and values
            queries, keys, values = q_proj(x), k_proj(x), v_proj(x)

            # Prepare the queries, keys and values for the attention computation

            # Queries will be shape (B, H, L, D_q)
            queries = ten.swapaxes(queries.reshape(B, L, H, -1), 1, 2)

            # Keys will be shape (B, H_kv, L, D_k)
            keys = ten.swapaxes(keys.reshape(B, L, H_kv, -1), 1, 2)

            # Keys will be shape (B, H_KV, L, D_v)
            values = ten.swapaxes(values.reshape(B, L, H_kv, -1), 1, 2)
            # queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
            # keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            # values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

            # ten.eval(queries, keys, values)

            extra = score_extra(x)

            if ctx := tensile.nn.lm.LanguageModelContext.get_current():
                cache = ctx.layer_cache
                masker = ctx.get_masker(L, dtype=queries.dtype)
            else:
                cache = None
                masker = None

            # ten.debug_eval(queries, keys, values)
            if cache is not None:
                # Calculate position encoding using offset from cache
                queries = q_encode_position(queries, offset=cache.offset)
                keys = k_encode_position(keys, offset=cache.offset)

                queries, keys, values = reshape_qkv(queries, keys, values, force_five=True)

                output = cache.attention(
                    queries, keys, values,
                    scale=scale,
                    masker=masker,
                    attend=attend,
                    **extra,
                )
            else:
                # Calculate position encoding using zero offset
                queries = q_encode_position(queries)
                keys = k_encode_position(keys)

                queries, keys, values = reshape_qkv(queries, keys, values, force_five=True)
                ten.debug_eval(queries, keys, values)
                output = attend(
                    queries, keys, values,
                    scale=scale,
                    masker=masker,
                    **extra,
                )
                # output = standard_attention(
                #     queries, keys, values,
                #     mask=mask,
                #     score_attention=score_attention,
                # )

            # Output will have shape (B, H, L, D_v)
            # Reshape it to (B, L, D)
            if output.ndim == 5:
                output = output.reshape(B, H, L, -1)
            ten.debug_eval(output)
            output = ten.swapaxes(output, 1, 2).reshape(B, L, D)
            return o_proj(output)
        return call


__all__ = [
    'Attention',
    'AttentionArgs',
    'StandardAttention',
]