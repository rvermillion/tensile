#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import math

from .util import shift_slice
from ..common import *
from ..module import CompiledModule, ModuleArgs
from .score import AttentionScores
from .types import AttendFunction, AttentionMasker, AttentionScorer, KVSlice, QKV


class Attend(CompiledModule):

    __slots__ = ('dtype', 'contiguous')

    dtype: Annotated[Optional[DType], field(
        doc='The data type to use for the attention scores. If None, the dtype of the inputs will be used.',
    )]
    contiguous: Annotated[bool, field(
        doc='Whether to use contiguous memory layout for the attention scores',
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        # self.dtype = args.get('dtype', None)
        self.contiguous = args.get('contiguous', False)

    def scores_factory(self, queries: Array, qs: slice, v_dim: int) -> AttentionScores:
        return AttentionScores(queries, qs, v_dim)

    def build_prepare(self) -> Callable[[Array, Array, Array], QKV]:
        dtype = self.dtype
        contiguous = self.contiguous
        if dtype is None:
            if contiguous:
                def prepare(queries: Array, keys_t: Array, values: Array) -> tuple[Array, Array, Array]:
                    keys_t = ten.contiguous(keys_t)
                    return queries, keys_t, values
            else:
                def prepare(queries: Array, keys_t: Array, values: Array) -> tuple[Array, Array, Array]:
                    return queries, keys_t, values

        else:
            if contiguous:
                def prepare(queries: Array, keys_t: Array, values: Array) -> tuple[Array, Array, Array]:
                    queries = ten.as_type(queries, dtype)
                    keys_t = ten.as_type(keys_t, dtype)
                    values = ten.as_type(values, dtype)
                    keys_t = ten.contiguous(keys_t)
                    return queries, keys_t, values
            else:
                def prepare(queries: Array, keys_t: Array, values: Array) -> tuple[Array, Array, Array]:
                    queries = ten.as_type(queries, dtype)
                    keys_t = ten.as_type(keys_t, dtype)
                    values = ten.as_type(values, dtype)
                    return queries, keys_t, values

        return prepare


@provides(Attend, 'default')
class DefaultAttend(Attend):

    __slots__ = ('score', 'tile_size')

    score: Annotated[AttentionScorer, field(
        doc='The attention scorer to use for the attention mechanism',
    )]
    tile_size: Annotated[Optional[tuple[int, int]], field(
        doc='The tile size to use for the attention mechanism. If None, no tiling will be used.',
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        self.tile_size = args.get('tile_size', None)
        self.score = self.build_score()

    def build_score(self) -> AttentionScorer:
        return coerce(AttentionScorer, kind='sdpa')

    def attend(self, queries: Array, qs: slice, kv_iter: Iterable[KVSlice], /, scale: float = None, masker: AttentionMasker = None,
               offset: int = 0, **extra) -> Array:
        scores_factory = self.scores_factory
        score = self.score

        scores = None

        for keys_t, values, kvs in kv_iter:
            D_v = values.shape[-1]

            if scores is None:
                scores = scores_factory(queries, qs, D_v)

            scores.add_kvs(keys_t, values, kvs, masker=masker, scorer=score, **extra)

        return scores.out

    def build_call(self, train: bool = False, **options) -> AttendFunction:
        tile_size = self.tile_size
        score = self.score
        scores_factory = self.scores_factory
        prepare = self.build_prepare()

        if tile_size is None:
            # noinspection PyPep8Naming
            def attend(queries: Array, keys: Array, values: Array, /,
                       scale: float = None,
                       masker: AttentionMasker = None,
                       offset: int = 0,
                       **extra,
                       ) -> Array:
                Q = queries.shape[-2]
                K = keys.shape[-2]
                D_v = values.shape[-1]

                if scale is None:
                    scale = 1. / math.sqrt(queries.shape[-1])

                keys_t = ten.swapaxes(keys, -1, -2)

                queries, keys_t, values = prepare(queries, keys_t, values)

                queries = scale * queries

                qs = slice(offset, Q+offset)
                kvs = slice(0, K)

                scores = scores_factory(queries, qs, D_v)

                scores.add_kvs(keys_t, values, kvs, masker=masker, scorer=score, **extra)

                return scores.out

        else:
            q_tile_size, k_tile_size = tile_size

            # noinspection PyPep8Naming
            def attend(queries: Array, keys: Array, values: Array, /,
                       scale: float = None,
                       masker: AttentionMasker = None,
                       offset: int = 0,
                       **extra,
                       ) -> Array:

                ten.debug_eval(queries, keys)

                Q = queries.shape[-2]
                K = keys.shape[-2]
                D_v = values.shape[-1]

                if scale is None:
                    scale = 1. / math.sqrt(queries.shape[-1])

                keys_t = ten.swapaxes(keys, -1, -2)

                queries, keys_t, values = prepare(queries, keys_t, values)

                queries = scale * queries

                if Q < q_tile_size and K < k_tile_size:
                    qs = slice(offset, Q+offset)
                    kvs = slice(0, K)

                    scores = scores_factory(queries, qs, D_v)

                    scores.add_kvs(keys_t, values, kvs, masker=masker, scorer=score, **extra)

                    return scores.out

                q_tiles = []
                for i in range(0, Q, q_tile_size):
                    Q_i = min(q_tile_size, Q-i)

                    qs = slice(i, i + Q_i)

                    q_i = queries[..., qs, :]

                    qs = shift_slice(qs, offset)
                    scores_i = scores_factory(q_i, qs, D_v)

                    K_i = min(K, i+Q_i+offset) if masker.causal else K

                    for j in range(0, K_i, k_tile_size):
                        K_j = min(k_tile_size, K_i-j)

                        kvs = slice(j, j + K_j)

                        kt_j = keys_t[..., kvs]
                        v_j = values[..., kvs, :]

                        scores_i.add_kvs(kt_j, v_j, kvs, masker=masker, scorer=score, **extra)

                    q_tiles.append(ten.as_type(scores_i.out, queries.dtype))

                return ten.concatenate(q_tiles, axis=-2)

        return attend


def fast_attend(queries: Array, keys: Array, values: Array, /,
                scale: float = None, masker: AttentionMasker = None, offset: int = 0, **extra) -> Array:
    return ten.fast.scaled_dot_product_attention(queries, keys, values, scale=scale,
                                                 mask='causal' if masker.causal else masker.mask)


@provides(Attend, 'fast')
class FastAttend(CompiledModule):

    __slots__ = ()

    def score(self, queries: Array, keys_t: Array, qs: Optional[slice], ks: Optional[slice], /, offset: int = 0, **extra) -> Array:
        return ten.matmul(queries, keys_t)

    def build_call(self, train: bool = False, **options) -> AttendFunction:
        return fast_attend


default_initial_beta = -20.


@provides(Attend, 'sink')
class AttendWithSink(DefaultAttend):

    __slots__ = ('betas', 'null_values')

    betas: Annotated[Array, field(
        doc='The beta value to use for the attention mechanism',
        parameter=True,
    )]
    null_values: Annotated[Array, field(
        doc='The null value to use for the attention mechanism',
        parameter=True,
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        n_q_heads = args.get('num_attention_heads')
        n_kv_heads = args.get('num_key_value_heads')
        hidden_size = args.get('hidden_size')
        initial_beta = args.get('initial_beta', default=default_initial_beta)
        dtype = ten.dtype(args.get('dtype', default=ten.float32))

        if args.get('nilpotent', False):
            null_values = ten.zeros((1, n_kv_heads, 1, 1, hidden_size // n_q_heads), dtype=dtype)
            betas = ten.full((1, n_kv_heads, 1, 1, 1), initial_beta, dtype=dtype)
        else:
            size = n_kv_heads * (hidden_size//n_q_heads)
            scale = size ** 0.5
            null_values = ten.as_type(ten.random.normal(scale=scale, shape=(1, n_kv_heads, 1, 1, hidden_size // n_q_heads)), dtype)
            betas = ten.full((1, n_kv_heads, 1, 1, 1), 0., dtype=dtype)

        ten.eval(null_values, betas)

        self.betas = betas
        self.null_values = null_values

    def scores_factory(self, queries: Array, qs: slice, v_dim: int) -> AttentionScores:
        scores = AttentionScores(queries, qs, v_dim)
        scores.add_unmasked(self.betas, self.null_values)
        return scores



