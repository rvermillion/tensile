#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import Array, DType, Optional, Protocol, meta


class AttentionMasker(Protocol):

    @property
    def causal(self) -> bool: ...

    @property
    def mask(self) -> Optional[Array]: ...

    def __call__(self, logits: Array, qs: slice = None, kvs: slice = None, offset: int = 0) -> Array: ...

    def filter(self, qs: slice, kvs: slice, offset: int = 0) -> Optional[slice]: ...


class AttendFunction(Protocol):

    @property
    def score(self) -> 'AttentionScorer': ...

    def __call__(self, queries: Array, keys: Array, values: Array, /,
                 scale: float = None, masker: AttentionMasker = None, **kwargs) -> Array: ...


class AttentionScorer(Protocol):

    def __call__(self, queries: Array, keys_t: Array, qs: Optional[slice], ks: Optional[slice], /,
                 offset: int = 0, **extra) -> Array: ...


meta.for_class(AttentionScorer).configure_registry(
    default_kind='sdpa'
)


class AttentionValueWeighter(Protocol):

    def __call__(self, scores: Array, values: Array, attn_dtype: DType = None) -> Array: ...


class MaskBuilder(Protocol):

    def __call__(self, size: int, offset: int = 0, attn_dtype: DType = None) -> Array: ...



KV = tuple[Array, Array]
KVSlice = tuple[Array, Array, slice]
QKV = tuple[Array, Array, Array]


__all__ = [
    'AttendFunction',
    'AttentionMasker',
    'AttentionScorer',
    'KV',
    'KVSlice',
    'QKV',
]
