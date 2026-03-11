#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..module import Module
from ..position import PositionEncoder
from .attention import Attention, AttentionArgs, StandardAttention
from .attend import Attend


@provides(Attention, 'coordinate')
class CoordinateAttention(StandardAttention):

    __slots__ = ('coord_dim', 'q_coord_proj', 'k_coord_proj')

    coord_dim: Annotated[int, field()]
    q_coord_proj: Annotated[Module, field()]
    k_coord_proj: Annotated[Module, field()]

    def init_from_args(self, args: AttentionArgs):
        super().init_from_args(args)

        dim = self.dim
        attention_bias = args.get('bias', default=self.default_attention_bias)

        self.coord_dim = args.get('coord_dim', default=self.q_head_dim // 8)
        self.q_coord_proj = self.build_q_coord_proj(dim, bias=attention_bias)
        self.k_coord_proj = self.build_k_coord_proj(dim, bias=attention_bias)

    def build_attend(self, args: AttentionArgs) -> Attend:
        return coerce(Attend, kind='gated')  #, args=args)

    def build_position_encoder(self, args: AttentionArgs) -> PositionEncoder:
        encoder_args = args.position_encoder.set_defaults(kind='none')
        return PositionEncoder.from_args(encoder_args)

    def build_q_coord_proj(self, in_size: int, bias: bool = False) -> Module:
        return self.build_proj(in_size, self.n_heads * self.coord_dim, bias=bias, name='q_coord_proj')

    def build_k_coord_proj(self, in_size: int, bias: bool = False) -> Module:
        return self.build_proj(in_size, self.n_kv_heads * self.coord_dim, bias=bias, name='k_coord_proj')

    def _lazy_score_extra(self) -> Callable[[Array], dict[str, Array]]:
        q_coord_proj = self.q_coord_proj
        k_coord_proj = self.k_coord_proj
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads

        # noinspection PyPep8Naming
        def score_extra(x: Array) -> dict[str, Array]:
            B, L = x.shape[:2]
            q_gate = q_coord_proj(x)
            q_gate = ten.swapaxes(q_gate.reshape(B, L, n_heads, -1), 1, 2)
            k_gate = k_coord_proj(x)
            k_gate = ten.swapaxes(k_gate.reshape(B, L, n_kv_heads, -1), 1, 2)
            return {
                'q_gate': q_gate,
                'k_gate_t': ten.swapaxes(k_gate, -2, -1),
            }
        return score_extra


__all__ = [
    'CoordinateAttention'
]