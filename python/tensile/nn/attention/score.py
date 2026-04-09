#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import math

from ...infra import RootObject
from ..common import *
from ..module import CompiledModule, ModuleArgs
from .types import AttentionMasker, AttentionScorer, KVSlice


class AttentionScores(RootObject):

    __slots__ = ('queries', 'qs', 'max', 'sumexp', 'values')

    queries: Array
    qs: slice
    max: Array        # (B, *H, Q, 1)
    sumexp: Array     # (B, *H, Q, 1)
    values: Array     # (B, *H, Q, D_v)

    def __init__(self, queries: Array, qs: slice, v_dim: int, initial_max: Array = None, dtype: DType = None):
        # We initialize the max, sum and values. Broadcasting will do its magic when the first logits and values
        # are added to the accumulator.
        if dtype is None: dtype = queries.dtype
        self.queries = queries
        self.qs = qs
        self.max = ten.array(-ten.inf, dtype=dtype) if initial_max is None else initial_max
        self.sumexp = ten.array(0., dtype=dtype)
        self.values = ten.array(0., dtype=dtype)
        # self.values = ten.zeros((v_dim, ), dtype=dtype)

    def add_masked(self, logits: Array, values: Array) -> None:
        """
        Add new logits and values to the score accumulator.

        :param logits: Logits tensor for the current tile (B, *H, Q, K).
        :param values: Values tensor for the current tile (B, *H, K, D_v).
        :return: None
        """

        # ten.eval(logits, values)
        current_max = ten.max(logits, axis=-1, keepdims=True)
        new_max = ten.maximum(current_max, self.max)

        finite = ten.isfinite(new_max)

        old_exp = ten.where(finite, ten.exp(self.max - new_max), 1.0)

        new_exp = ten.where(finite, ten.exp(logits - new_max), 0.0)

        new_sumexp = ten.sum(new_exp, axis=-1, keepdims=True) + self.sumexp * old_exp

        new_values = ten.matmul(new_exp, values) + self.values * old_exp

        self.sumexp = new_sumexp
        self.values = new_values
        self.max = new_max

    def add_unmasked(self, logits: Array, values: Array) -> None:
        """
        Add new logits and values to the score accumulator.

        :param logits: Logits tensor for the current tile (B, *H, Q, K).
        :param values: Values tensor for the current tile (B, *H, K, D_v).
        :return: None
        """
        current_max = ten.max(logits, axis=-1, keepdims=True)
        new_max = ten.maximum(current_max, self.max)

        old_exp = ten.exp(self.max - new_max)

        new_exp = ten.exp(logits - new_max)

        new_sumexp = self.sumexp * old_exp + ten.sum(new_exp, axis=-1, keepdims=True)

        new_values = self.values * old_exp + ten.matmul(new_exp, values)

        self.sumexp = new_sumexp
        self.values = new_values
        self.max = new_max

    def combine(self, other: 'AttentionScores') -> None:
        new_max = ten.maximum(other.max, self.max)

        exp_self = ten.exp(self.max - new_max)
        exp_other = ten.exp(other.max - new_max)

        self.sumexp *= exp_self
        self.sumexp += other.sumexp * exp_other

        self.values *= exp_self
        self.values += other.values * exp_other

        self.max = new_max

    def add_kvs(self, keys_t: Array, values: Array, kvs: slice, /,
                masker: AttentionMasker = None,
                scorer: AttentionScorer = None,
                **extra) -> None:
        queries = self.queries
        qs = self.qs

        if masker is None:
            logits = scorer(queries, keys_t, qs, kvs, **extra)
            self.add_unmasked(logits, values)

        else:
            kvs = masker.filter(qs, kvs)

            if kvs is not None:
                logits = scorer(queries, keys_t, qs, kvs, **extra)
                logits = masker(logits, qs, kvs)
                self.add_masked(logits, values)

    def add_all_kvs(self, kv_iter: Iterable[KVSlice], /,
                masker: AttentionMasker = None,
                scorer: AttentionScorer = None,
                **extra) -> None:
        queries = self.queries
        qs = self.qs

        if masker is None:
            for keys_t, values, kvs in kv_iter:
                logits = scorer(queries, keys_t, qs, kvs, **extra)
                self.add_unmasked(logits, values)

        else:
            for keys_t, values, kvs in kv_iter:
                kvs = masker.filter(qs, kvs)

                if kvs is not None:
                    logits = scorer(queries, keys_t, qs, kvs, **extra)
                    logits = masker(logits, qs, kvs)
                    self.add_masked(logits, values)

    def out(self) -> Array:
        """
        Compute the final output of the score accumulator.

        :return: Normalized values tensor (B, *H, Q, D_v).
        """
        return ten.where(self.sumexp > 0., self.values / self.sumexp, 0.0)

    def _repr_args(self, **options) -> str:
        return f'shape={self.values.shape}, dtype={self.values.dtype}'


AttentionScoresFactory = Callable[[Array, slice, int], AttentionScores]



class BaseAttentionScorer(Object):

    __slots__ = ()

    def score(self, queries: Array, keys: Array, call: 'patchlm.cache.ModelCall') -> Array:
        raise NotImplementedError()

    def __call__(self, queries: Array, keys: Array, call: 'patchlm.cache.ModelCall') -> Array:
        return self.score(queries, keys, call)


class FunctionAttentionScorer(BaseAttentionScorer):

    __slots__ = ('score',)

    score: Annotated[AttentionScorer, field(
        doc='The attention scorer function.',
        required=True,
    )]


# @provides(AttentionScorer, 'sdpa', 'default', spread=True)
# def provide_sdpa_scorer() -> AttentionScorer:
#     return sdpa_attention_scorer

@meta.provides_singleton(AttentionScorer, 'sdpa', 'default')
def sdpa_attention_scorer(queries: Array, keys_t: Array, qs: Optional[slice], kvs: Optional[slice], /,
                          offset: int = 0, **extra) -> Array:
    """
    Computes scaled dot-product attention scores by performing matrix multiplication
    between scaled queries and transposed keys within the specified slices.

    :param offset:
    :param queries: Input tensor representing the queries. (B, *H, Q, D)
    :param keys_t: Transposed keys tensor.                 (B, *H, D, K)
    :param qs: Slice defining the range for the queries.
    :param kvs: Slice defining the range for the keys and values.
    :param extra: Additional keyword arguments (optional; this can act as a placeholder
                  for any implementation-specific parameters).
    :return: A tensor containing the scaled dot-product attention scores. (B, *H, Q, K)

    """
    return ten.matmul(queries, keys_t)


__all__ = [
    'AttentionScores',
    'AttentionScoresFactory',
]