#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..layers.linear import Linear, LinearArgs
from ..layers.normalization import NormedModule
from ..position import PositionEncoder
from ..module import CompiledModule, Module, ModuleArgs
from .attend import Attend
from .context import AttentionContext
from .util import reshape_qkv



class AttentionArgs(ModuleArgs):

    kind: Annotated[str, meta.field(inherit='attention_kind')] = 'standard'

    hidden_dim: int
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
    qkv_proj: Annotated[LinearArgs, field(
        doc='The arguments for the fused qkv projection',
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

    def postinit(self, spec: Spec):
        super().postinit(spec)

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@provides(Module, 'attention')
class Attention(CompiledModule):

    __slots__ = ('dim', 'in_dim', 'out_dim', 'n_heads', 'n_kv_heads', 'kv_heads_per_head', 'q_head_dim',
                 'k_head_dim', 'v_head_dim', 'scale', 'attend', 'encode_position',
                 'o_proj')

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

    o_proj: Module

    default_attention_bias: ClassVar[bool] = False

    # noinspection PyAttributeOutsideInit
    def init_from_args(self, args: AttentionArgs):
        super().init_from_args(args)

        dim = args.hidden_dim
        n_heads = args.num_attention_heads
        n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim or dim // n_heads
        scale = head_dim**-0.5
        attention_bias = self.attention_bias

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

        self.o_proj = self.init_o_proj(dim, bias=attention_bias)

        self.encode_position = self.init_position_encoder(args)
        self.attend = self.init_attend(args)

    @property
    def attention_bias(self) -> bool:
        return self.args.get('bias', self.default_attention_bias)

    def init_attend(self, args: AttentionArgs) -> Attend:
        attend_args = args.attend.set_defaults(
            kind='default'
        )
        return coerce(Attend, args=attend_args, kind=attend_args.kind)

    def init_q_proj(self, in_size: int, bias: bool = False, name: str = 'q_proj') -> Module:
        args = self.args.q_proj.set_defaults(
            in_dim=in_size,
            out_dim=self.n_heads * self.q_head_dim,
            bias=bias,
        )
        return self.init_proj(args)

    def init_k_proj(self, in_size: int, bias: bool = False, name: str = 'k_proj') -> Module:
        args = self.args.k_proj.set_defaults(
            in_dim=in_size,
            out_dim=self.n_kv_heads * self.k_head_dim,
            bias=bias,
        )
        return self.init_proj(args)

    def init_v_proj(self, in_size: int, bias: bool = False, name: str = 'v_proj') -> Module:
        args = self.args.v_proj.set_defaults(
            in_dim=in_size,
            out_dim=self.n_kv_heads * self.v_head_dim,
            bias=bias,
        )
        return self.init_proj(args)

    def init_qkv_proj(self, in_size: int, bias: bool = False, name: str = 'qkv_proj') -> Module:
        args = self.args.qkv_proj.set_defaults(
            in_dim=in_size,
            out_dim=self.n_heads * self.q_head_dim +
                        self.n_kv_heads * self.k_head_dim +
                        self.n_kv_heads * self.v_head_dim,
            bias=bias,
        )
        return self.init_proj(args)

    def init_o_proj(self, out_size: int, bias: bool = False, name: str = 'o_proj') -> Module:
        args = self.args.o_proj.set_defaults(
            in_dim=self.n_heads * self.v_head_dim,
            out_dim=out_size,
            bias=bias,
        )
        return self.init_proj(args)

    def init_position_encoder(self, args: AttentionArgs) -> PositionEncoder:
        encoder_args = args.position_encoder.set_defaults(
            kind='rope',
            dims=min(self.q_head_dim, self.k_head_dim)
        )
        return PositionEncoder.from_args(encoder_args)

    Args = AttentionArgs


meta.for_class(Attention).configure_registry(
    append_kind=True,
    default_kind='standard'
)


class BaseAttention(Attention):

    __slots__ = ()

    def build_qkv_proj(self, mode: CompiledModule.Mode, **options) -> Callable[[Array], tuple[Array, Array, Array]]:
        raise NotImplementedError()

    def build_encode_position(self, mode: CompiledModule.Mode) -> tuple[PositionEncoder, PositionEncoder]:
        return self.encode_position, self.encode_position

    # noinspection PyPep8Naming
    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        q_encode_position, k_encode_position = self.build_encode_position(mode)
        o_proj = self.o_proj

        project_qkv = self.build_qkv_proj(mode)
        scale = self.scale
        attend = self.attend

        H = self.n_heads
        H_kv = self.n_kv_heads

        # noinspection PyPep8Naming
        def call(x: Array) -> Array:
            B, L, D = x.shape

            # Project the input into queries, keys and values
            queries, keys, values = project_qkv(x)

            # Prepare the queries, keys and values for the attention computation

            # Queries will be shape (B, H, L, D_q)
            queries = ten.swapaxes(queries.reshape(B, L, H, -1), 1, 2)

            # Keys will be shape (B, H_kv, L, D_k)
            keys = ten.swapaxes(keys.reshape(B, L, H_kv, -1), 1, 2)

            # Keys will be shape (B, H_KV, L, D_v)
            values = ten.swapaxes(values.reshape(B, L, H_kv, -1), 1, 2)

            if ctx := AttentionContext.get_current():
                cache = ctx.layer_cache
                offset = ctx.offset
                masker = ctx.get_masker(L, dtype=queries.dtype)
            else:
                cache = None
                offset = 0
                masker = None

            # Calculate position encoding using offset from cache
            queries = q_encode_position(queries, offset=offset)
            keys = k_encode_position(keys, offset=offset)

            if cache is None:
                # If we don't have a cache, then just call `attend` directly, passing in the scale and masker
                output = attend(
                    queries, keys, values,
                    scale=scale,
                    masker=masker,
                )
            else:
                # If we have a cache, let it handle attention, but pass in the scale, masker and attend module for
                # it to use.
                output = cache.attention(
                    queries, keys, values,
                    scale=scale,
                    masker=masker,
                    attend=attend,
                )

            # Reshape the output, ensuring that it's only 4 dimensions
            if output.ndim == 5:
                output = output.reshape(B, H, L, -1)
            output = ten.swapaxes(output, 1, 2).reshape(B, L, D)

            # Project using o_proj to get the final output
            return o_proj(output)
        return call


ScoreExtra = Callable[[Array], dict[str, Array]]


@provides(Attention, 'standard')
class StandardAttention(BaseAttention):

    __slots__ = ('q_proj', 'k_proj', 'v_proj', 'fuse_for_inference')

    q_proj: Module
    k_proj: Module
    v_proj: Module
    fuse_for_inference: Annotated[bool, field(
        doc="Whether to fuse the query, key, and value projections for inference",
        default=False,
    )]

    def init_from_args(self, args: AttentionArgs):
        super().init_from_args(args)

        dim = self.dim
        attention_bias = self.attention_bias

        self.q_proj = self.init_q_proj(dim, bias=attention_bias)
        self.k_proj = self.init_k_proj(dim, bias=attention_bias)
        self.v_proj = self.init_v_proj(dim, bias=attention_bias)

    def build_qkv_proj(self, mode: CompiledModule.Mode, **options) -> Callable[[Array], tuple[Array, Array, Array]]:
        q_proj = self.q_proj
        k_proj = self.k_proj
        v_proj = self.v_proj

        if mode.is_compiled() and self.fuse_for_inference:
            if isinstance(q_proj, Linear) and isinstance(k_proj, Linear) and isinstance(v_proj, Linear):
                return Linear.fuse(q_proj, k_proj, v_proj)
            else:
                self.warn('Could not fuse projections for inference: q, k, and v projections are not all Linear modules')

        def project_qkv(x: Array) -> tuple[Array, Array, Array]:
            return q_proj(x), k_proj(x), v_proj(x)

        return project_qkv


@provides(Attention, 'fused-standard')
class FusedAttention(BaseAttention):

    __slots__ = ('qkv_proj',)

    qkv_proj: Module

    def init_from_args(self, args: AttentionArgs):
        super().init_from_args(args)

        dim = self.dim
        attention_bias = self.attention_bias

        self.qkv_proj = self.init_qkv_proj(dim, bias=attention_bias)

    def build_qkv_proj(self, mode: CompiledModule.Mode, **options) -> Callable[[Array], tuple[Array, Array, Array]]:
        qkv_proj = self.qkv_proj
        query_pos = self.n_heads * self.q_head_dim
        key_pos = query_pos + self.n_kv_heads * self.k_head_dim

        def project_qkv(x: Array) -> tuple[Array, Array, Array]:
            qkv = qkv_proj(x)
            q, k, v = ten.split(qkv, [query_pos, key_pos], axis=-1)
            return q, k, v

        return project_qkv


provides(Attention, 'normed')(NormedModule)


__all__ = [
    'Attention',
    'AttentionArgs',
    'StandardAttention',
]