#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ...infrastructure import RootObject
from ..common import *
from .types import AttentionMasker, MaskBuilder


mask_value = -ten.inf


@provides(MaskBuilder, 'causal', spread=True)
def causal(window_size: Optional[int] = None, lengths: Optional[Array] = None, default_dtype: DType = ten.float32) -> MaskBuilder:
    if window_size is None and lengths is None:
        def create_mask(size: int, offset: int = 0, dtype: DType = default_dtype) -> Array:
            rinds = ten.arange(offset + size)
            linds = ten.arange(offset, offset + size) if offset else rinds
            linds = linds[:, None]
            rinds = rinds[None]
            mask = linds < rinds
            return ten.as_type(mask * mask_value, dtype=dtype)
        return create_mask
    else:
        def create_mask(size: int, offset: int = 0, dtype: DType = default_dtype) -> Array:
            rinds = ten.arange(offset + size)
            linds = ten.arange(offset, offset + size) if offset else rinds
            linds = linds[:, None]
            rinds = rinds[None]
            mask = linds < rinds
            if window_size is not None:
                mask = mask | (linds > rinds + window_size)
            if lengths is not None:
                l = lengths[:, None, None, None]
                mask = mask | (rinds >= l)
            return ten.as_type(mask * mask_value, dtype=dtype)
        return create_mask


# noinspection PyPep8Naming
def create_causal_mask(
    size: int,
    offset: int = 0,
    dtype: DType = ten.float32,
) -> Array:
    rinds = ten.arange(offset + size)
    linds = ten.arange(offset, offset + size) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds < rinds
    return ten.as_type(ten.where(mask, mask_value, 0.), dtype=dtype)


class BaseAttentionMasker(RootObject):

    __slots__ = ()

    causal: bool = False
    mask: Optional[Array] = None

    def __call__(self, logits: Array, qs: slice = None, kvs: slice = None, offset: int = 0) -> Array:
        return logits

    def filter(self, qs: slice, kvs: slice, offset: int = 0) -> Optional[slice]:
        if self.causal:
            kvstart = kvs.start + offset
            kvstop = kvs.stop + offset
            assert 0 <= qs.start < qs.stop and (qs.step is None or qs.step == 1)
            assert 0 <= kvstart < kvstop and (kvs.step is None or kvs.step == 1)
            if kvstart >= qs.stop:
                return None
            if kvstop <= qs.start:
                return kvs
            stop = min(qs.stop, kvstop)
            nkvs = slice(kvstart-offset, stop-offset)
            return nkvs
        return kvs



no_mask: AttentionMasker = BaseAttentionMasker()


class ArrayMasker(BaseAttentionMasker):

    __slots__ = ('mask', 'causal')

    mask: Array
    causal: bool

    def __init__(self, mask: Array, causal: bool = True):
        if not ten.is_array(mask): raise ValueError(f'Mask must be an array, not {mask!r}')
        self.mask = mask
        self.causal = causal

    def __call__(self, logits: Array, qs: slice = None, kvs: slice = None, offset: int = 0) -> Array:
        if qs is None:
            if kvs is None:
                mask = self.mask
            else:
                mask = self.mask[..., kvs]
        else:
            if kvs is None:
                mask = self.mask[..., qs, :]
            else:
                mask = self.mask[..., qs, kvs]
        logits += mask
        return logits
        # return apply_additive_mask(logits, mask, offset=offset)

    def _repr_args(self, **options) -> str:
        return f'mask={self.mask.shape}, +causal'


def make_additive_masker(mask: Array|None) -> AttentionMasker:
    if mask is None: return no_mask
    return ArrayMasker(mask)


# noinspection PyPep8Naming
def apply_additive_mask(scores: Array, mask: Array, offset: int = 0, in_place: bool = True) -> Array:
    Q, K = scores.shape[-2:]
    M = mask.shape[-1]

    assert M == mask.shape[-2], f'Mask must be of shape (..., M, M) not {mask.shape}'

    if Q == K:
        if Q < M:
            mask = mask[..., -Q:, -Q:]

        if in_place:
            scores += mask
        else:
            scores = scores + mask
    elif Q < K:
        if not in_place:
            scores = 0. + scores
        if Q < M:
            mask = mask[..., -Q:, -Q:]
        scores[..., -Q:] += mask
    else:
        raise ValueError(f'Cannot apply mask to scores: scores.shape={scores.shape} mask.shape={mask.shape}')
    return scores





__all__ = [
    'AttentionMasker',
    'apply_additive_mask',
    'create_causal_mask',
    'make_additive_masker',
    'no_mask',
]