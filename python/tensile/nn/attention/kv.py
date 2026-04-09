#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
# from typing import Iterable, Optional, TypeAlias, TypeVar, TYPE_CHECKING

import numpy as np

from tensile import AxisSelector, Selector

from ...infra import RootObject, Storable
from ...util import select
from ...util.buffer import ArrayBuffer, align_size, full_slice
from ..common import *


if TYPE_CHECKING:
    import patchlm.cache


KV = tuple[Array, Array]
KVM = tuple[Array, Array, Optional[Array]]


def create_pad(values: Array, size: int, axis: int = -1) -> Array:
    if axis == -1:
        pad_shape = (*values.shape[:-1], size)
    else:
        pad_shape = (*values.shape[:axis], size, *values.shape[axis+1:])
    return ten.zeros(pad_shape, dtype=values.dtype)


def full_repeat(count: int) -> Iterable[slice]:
    return (full_slice for _ in range(count))


def full_after(axis: int) -> Iterable[slice]:
    return full_repeat(-1-axis)


class KVSequence(RootObject, Storable):

    __slots__ = ()

    pos_start: int
    pos_end: int
    pos_len: int

    batch_count: int

    readonly: bool = True

    state: tuple[Array, ...]

    # @property
    # def pos_start(self) -> int:
    #     raise NotImplementedError()
    #
    # @property
    # def pos_end(self) -> int:
    #     raise NotImplementedError()
    #
    # @property
    # def pos_len(self) -> int:
    #     raise NotImplementedError()

    @property
    def pos_range(self) -> tuple[int, int]:
        return self.pos_start, self.pos_end

    # @property
    # def batch_count(self) -> int:
    #     raise NotImplementedError()

    def select(self, *,
               pos: Optional[AxisSelector] = None,
               pos_start: Optional[int] = None,
               pos_end: Optional[int] = None,
               batch: Optional[AxisSelector] = None,
               batch_end: Optional[int] = None) -> Selector:
        raise NotImplementedError()

    def get_kvs(self, kvs: Selector = None) -> KV:
        raise NotImplementedError()

    def fetch_kv(self, *, batch: AxisSelector = None, n_batches: int = None) -> KV:
        return self.get_kvs(self.select(batch=batch, batch_end=n_batches))

    def branch(self, idx: Array):
        raise NotImplementedError()

    def swap_kv(self, b: int, i: int, j: int):
        raise NotImplementedError()

    def mask_scores(self, scores: Array) -> Array:
        return scores

    def score_attention(self, queries: Array, keys: Array) -> Array:
        """

        :param queries: (B, ..., Q, D)
        :param keys: (B, ..., K, D)
        :param call:
        :return:
        """
        q = ten.as_type(queries, ten.float32)
        kt = ten.as_type(ten.swapaxes(keys, -1, -2), ten.float32)
        return ten.matmul(q, kt)

    def partial_attentions(self, queries: Array, segment_queries: Array) -> Iterable['PartialAttentionScores']:
        return self.partial_attention(queries),

    def cut_segment(self, start: int = 0, stop: int = None, n_batches: int = None, **kwargs) -> 'KVSegment':
        keys, values = self.get_kvs(self.select(pos_start=start, pos_end=stop, batch_end=n_batches))
        segment = KVSegment(start, keys, values)
        return segment

    def eval(self) -> None:
        pass

    def _store_arrays(self, arrays: dict, prefix: str = ''):
        keys, values = self.fetch_kv()
        arrays[prefix + 'keys'] = keys
        arrays[prefix + 'values'] = values

    def _metadata_to_store(self) -> Optional[dict]:
        return None


zero_tuple = 0,


X = TypeVar('X')

def ensure_index(l: list[X], i: int, fill: X):
    need = 1 + i - len(l)
    if need > 0:
        l.extend(fill for _ in range(need))


class BatchMasker(RootObject):

    __slots__ = ['max_end']

    max_end: int
    min_end: int
    mask_value: float = -1.e9

    def __init__(self, max_end: int = None):
        self.max_end = 0 if max_end is None else max_end

    def get_pos_end(self, b: int) -> int:
        return self.max_end

    def mask_scores(self, scores: Array) -> Array:
        raise NotImplementedError()

    def evict(self, b: int, pos: int, keys: Array, values: Array, positions: Array = None):
        raise NotImplementedError()

    @classmethod
    def contiguous(cls, pos_end: int, n_batches: int = None) -> 'ContiguousBatchMasker':
        return ContiguousBatchMasker(pos_end, n_batches=n_batches)


def swap_kv(b: int, i: int, j: int, keys: Array, values: Array, positions: Array = None):
    keys[b, ..., i, :] = keys[b, ..., j, :]
    values[b, ..., i, :] = values[b, ..., j, :]
    if positions is not None:
        positions[b, i] = positions[b, j]


class ContiguousBatchMasker(BatchMasker):

    __slots__ = ('pos_ends',)

    pos_ends: list[int]

    def __init__(self, max_end: int = None, pos_ends: list[int] = None, n_batches: int = None):
        if max_end is None:
            if pos_ends is None:
                max_end = 0
                pos_ends = [max_end] * (n_batches or 0)
            else:
                max_end = max(*pos_ends)
        else:
            pos_ends = [max_end] * (n_batches or 0)
        super().__init__(max_end)
        self.pos_ends = pos_ends

    @property
    def min_end(self) -> int:
        return min(*self.pos_ends)

    def get_pos_end(self, b: int) -> int:
        if b >= len(self.pos_ends):
            return 0
        return self.pos_ends[b]

    def set_pos_ends(self, pos: int, n_batches: int = None):
        if n_batches is None:
            n_batches = len(self.pos_ends)
        else:
            ensure_index(self.pos_ends, n_batches - 1, pos)
        if pos > self.max_end or n_batches == len(self.pos_ends):
            self.max_end = pos

        for b in range(n_batches):
            self.pos_ends[b] = pos

    def _update_pos_end(self, b: int, new: int, old: int):
        end = self.max_end
        self.pos_ends[b] = new
        if new > end:
            self.max_end = new
        elif new < end:
            if old == end:
                self.max_end = max(*self.pos_ends)

    def set_pos_end(self, b: int, pos: int):
        ensure_index(self.pos_ends, b, 0)
        self._update_pos_end(b, pos, self.pos_ends[b])

    def increment_pos_end(self, b: int, pos: int = 1):
        ensure_index(self.pos_ends, b, 0)
        old = self.pos_ends[b]
        self._update_pos_end(b, old + pos, old)

    def decrement_pos_end(self, b: int, pos: int = 1):
        ensure_index(self.pos_ends, b, 0)
        old = self.pos_ends[b]
        self._update_pos_end(b, old - pos, old)

    def mask_scores(self, scores: Array) -> Array:
        if batch_pos_end := self.pos_ends:
            B = min(scores.shape[0], len(batch_pos_end))
            mask_value = self.mask_value
            for b in range(B):
                scores[b, ..., batch_pos_end[b]:] = mask_value
                print('-' * 10, b, '-' * 60)
                print(scores[b])
        return scores

    def evict(self, b: int, pos: int, keys: Array, values: Array, positions: Array = None):
        end = self.pos_ends[b]
        last = end - 1
        if pos < last:
            swap_kv(b, pos, last, keys, values, positions)
        self.pos_ends[b] = last
        if end == self.max_end:
            self.max_end = max(*self.pos_ends)

    def evict_all(self, b: int, poses: Sequence[int], keys: Array, values: Array, positions: Array = None):
        end = self.pos_ends[b]
        last = end - 1
        for pos in poses:
            if pos < last:
                swap_kv(b, pos, last, keys, values, positions)
            last -= 1
        self.pos_ends[b] = last + 1
        if end == self.max_end:
            self.max_end = max(*self.pos_ends)


class KVArray(KVSequence):

    __slots__ = ['keys', 'values', '_positions', 'pos_start', 'pos_end', 'pos_len', 'batch_count', 'batch_masker']

    keys: Annotated[Optional[Array], field(
        doc="The keys in this KVArray of shape (batch_capacity, ..., kv_capacity, head_key_dim)."
    )]
    values: Annotated[Optional[Array], field(
        doc="The keys in this KVArray of shape (batch_capacity, ..., kv_capacity, head_value_dim)."
    )]
    _positions: Annotated[Optional[Array], field(
        doc="The positions of the keys and values in this KVArray of shape (batch_count, pos_len, )."
    )]
    batch_masker: Annotated[Optional[BatchMasker], field(
        doc="A masker for when batches are different lengths"
    )]

    kv_capacity: int
    batch_capacity: int

    batch_align: int = 8
    kv_align: int = 256

    def __init__(self, pos_start: int, keys: Array = None, values: Array = None, length: int = None, batch_count: int = None):
        if pos_start < 0:
            raise ValueError('pos_start must be >= 0')
        self.keys = keys
        self.values = values
        self.batch_masker = None
        if keys is None:
            self.pos_start = pos_start
            self.pos_end = pos_start
            self.pos_len = 0
            self.batch_count = 0
        else:
            self.pos_start = pos_start
            if length is None:
                pos_len = keys.shape[-2]
            else:
                if length < 0:
                    raise ValueError('length must be >= 0')
                pos_len = length
            self.pos_end = pos_start + pos_len
            self.pos_len = pos_len
            self.batch_count = keys.shape[0] if batch_count is None else batch_count

    @property
    def kv_capacity(self) -> int:
        return 0 if (keys := self.keys) is None else keys.shape[-2]

    @kv_capacity.setter
    def kv_capacity(self, kv_capacity: int):
        pass

    @property
    def batch_capacity(self) -> int:
        return 0 if (keys := self.keys) is None else keys.shape[0]

    @batch_capacity.setter
    def batch_capacity(self, batch_capacity: int):
        pass

    @property
    def pos_range(self) -> tuple[int, int]:
        return self.pos_start, self.pos_end

    @pos_range.setter
    def pos_range(self, pos_range: tuple[int, int]):
        start, end = pos_range
        if start < 0:
            raise ValueError(f'pos_start ({start}) must be >= 0')
        if end < start:
            raise ValueError(f'pos_end ({end}) must be greater than pos_start ({start})')
        self.pos_start = start
        self.pos_end = end
        self.pos_len = end - start

    @property
    def positions(self) -> Optional[Array]:
        positions = self._positions
        if positions is None:
            positions = self._positions = self.range_positions()
        return positions

    def range_positions(self) -> Array:
        return ten.broadcast_to(ten.arange(self.pos_start, self.pos_end), (self.batch_count, self.pos_len))

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.pos_len = self.keys.shape[-2]
        self.pos_end = self.pos_start + self.pos_len

    @property
    def key_dim(self) -> int:
        return self.keys.shape[-1]

    @property
    def value_dim(self) -> int:
        return self.values.shape[-1]

    @property
    def batch_pos_end(self) -> Optional[Array]:
        return None

    def swap_kv(self, b: int, i: int, j: int):
        keys, values, positions = self.keys, self.values, self.positions
        keys[b, ..., i, :] = keys[b, ..., j, :]
        values[b, ..., i, :] = values[b, ..., j, :]
        if positions is not None:
            positions[b, i] = positions[b, j]

    def pos_offset(self, pos: int) -> int:
        return pos - self.pos_start

    def pos_all(self) -> slice:
        return full_slice

    def pos_select(self, start: Optional[int] = None, end: Optional[int] = None) -> AxisSelector:
        if start is None:
            start = 0
        else:
            start -= self.pos_start
            if start < 0:
                raise ValueError(f'start ({start}) must be >= start_pos {self.pos_start}')
        if end is None:
            if start == 0:
                return self.pos_all()
            end = self.pos_len
        else:
            end -= self.pos_start
            if end < start:
                raise ValueError(f'end ({end}) must be None or >= start {start}')
        if start == 0 and end == self.pos_len:
            return self.pos_all()
        return slice(start, end)

    def batch_select(self, start: Optional[int] = None, end: Optional[int] = None) -> AxisSelector:
        if start is None:
            start = 0
        else:
            if start < 0:
                raise ValueError(f'start ({start}) must be >= start_pos {self.pos_start}')
        if end is None:
            end = self.batch_count
        else:
            if end < start:
                raise ValueError(f'end ({end}) must be None or >= start {start}')
            if end > self.batch_count:  # > 1:
                raise ValueError(f'end ({end}) must be None or <= batch count {self.batch_count}')
        if start == 0 and end == self.batch_capacity:
            return full_slice
        return slice(start, end)

    def select(self, *,
               pos: Optional[AxisSelector] = None,
               pos_start: Optional[int] = None, pos_end: Optional[int] = None,
               batch: Optional[AxisSelector] = None,
               batch_start: Optional[int] = None, batch_end: Optional[int] = None) -> Selector:
        if batch is None:
            batch = self.batch_select(batch_start, batch_end)
        if pos is None:
            pos = self.pos_select(pos_start, pos_end)
        if batch is full_slice:
            if pos is full_slice:
                return ...
            return ..., pos, full_slice
        if pos is full_slice:
            return batch, ...
        return batch, ..., pos, full_slice

    def get_kvs(self, kvs: Selector = None) -> KV:
        if kvs is ...:
            return self.keys, self.values
        return self.keys[kvs], self.values[kvs]

    def fetch_kv(self, *, batch: AxisSelector = None, n_batches: int = None) -> KV:
        return self.get_kvs(self.select(batch=batch, batch_end=n_batches))
        # return self.get_kvs(self.select(batch_end=n_batches))

    # def update(self, keys: Array, values: Array, offset: int = None) -> None:
    #     B = keys.shape[0]
    #     n_keys = keys.shape[-2]
    #     if offset is None:
    #         offset = self.end_position
    #         grow_keys = n_keys
    #     else:
    #         trim = self.end_position - offset
    #         if trim < 0:
    #             raise ValueError(f'Offset must be None or less than or equal to {self.end_position}')
    #         grow_keys = n_keys - trim
    #
    #     self.grow(keys, values, grow_keys)
    #
    #     cached_keys, cached_values = self.keys, self.values
    #
    #     self.end_position = end = offset + n_keys
    #     self.length = end - self.position
    #     cached_keys[:B, ..., offset:end, :] = keys
    #     cached_values[:B, ..., offset:end, :] = values

    def build_batch_masker(self) -> BatchMasker:
        self.batch_masker = BatchMasker.contiguous(self.pos_end, self.batch_count)
        return self.batch_masker

    def swap_positions(self, b: int, i: int, j: int):
        if i != j:
            positions = self.positions
            positions[b, i] = positions[b, j]

    def evict(self, pos: int, batch: int = None):
        keys, values, positions = self.keys, self.values, self.positions
        batch_masker = self.batch_masker
        if batch_masker is None:
            if batch is None:
                batch = slice(None)
                last = self.pos_end - 1
                keys[batch, ..., pos, :] = keys[batch, ..., last, :]
                values[batch, ..., pos, :] = values[batch, ..., last, :]
                positions[batch, pos] = positions[batch, last]
                self.pos_end = last
                self.pos_len -= 1
                return
            batch_masker = self.build_batch_masker()
        if batch is None:
            for b in range(self.batch_count):
                batch_masker.evict(b, pos, keys, values, positions)
        else:
            batch_masker.evict(batch, pos, keys, values, positions)
        self.pos_end = batch_masker.max_end
        self.pos_len = self.pos_end - self.pos_start

    def branch(self, branches: Array):
        if self.batch_count == 1 and self.readonly:
            return

        cnt, = branches.shape
        if cnt != 0:
            keys, values = self.keys, self.values
            batch_count = self.batch_count
            new_batch_count = batch_count + cnt
            added_batch_count = new_batch_count - keys.shape[0]
            if added_batch_count > 0:
                batch_align = self.batch_align
                new_capacity = align_size(added_batch_count, batch_align)
                if added_batch_count == new_capacity:
                    if self.pos_start > 0:
                        raise NotImplementedError()
                    else:
                        self.keys = ten.concatenate([keys, keys[branches]], axis=0)
                        self.values = ten.concatenate([values, values[branches]], axis=0)
                        self.batch_count = new_batch_count
                    return
                else:
                    if self.pos_start > 0:
                        raise NotImplementedError()
                    else:
                        k_shape = (new_capacity, *keys.shape[1:])
                        v_shape = (new_capacity, *values.shape[1:])
                        new_k = ten.zeros(k_shape, keys.dtype)
                        new_v = ten.zeros(v_shape, values.dtype)
                        self.keys = keys = ten.concatenate([keys, new_k], axis=0)
                        self.values = values = ten.concatenate([values, new_v], axis=0)

            keys[batch_count:new_batch_count, ...] = keys[branches]
            values[batch_count:new_batch_count, ...] = values[branches]
            self.batch_count = new_batch_count

    def batch_mask_scores(self, scores: Array, call: 'patchlm.cache.ModelCall') -> Array:
        if batch_masker := self.batch_masker:
            scores = batch_masker.mask_scores(scores)
        return scores

    def mask_scores(self, scores: Array, call: 'patchlm.cache.ModelCall') -> Array:
        if (mask := call.mask) is not None:
            seg_pos = self.pos_start
            seg_end = self.pos_end
            call_pos = call.position
            if seg_end > call_pos:
                end = min(seg_end, call.end)
                start = max(call_pos, seg_pos)
                scores[..., start-seg_pos:end-seg_pos] += mask[..., start-call_pos:end-call_pos]

        scores = self.batch_mask_scores(scores, call)
        return scores

    def eval(self) -> None:
        ten.eval(self.keys, self.values)


class KVBuffer(KVArray):

    __slots__ = ()

    readonly: bool = False

    @property
    def state(self):
        end = self.pos_len
        if end == self.kv_capacity:
            return self.keys, self.values
        else:
            return (
                self.keys[..., :end, :],
                self.values[..., :end, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.set_pos_len(self.keys.shape[-2])

    def set_pos_len(self, pos_len: int):
        if pos_len < 0:
            raise ValueError(f'length ({pos_len}) must be >= 0')
        self.pos_len = pos_len
        self.pos_end = self.pos_start + pos_len

    def set_pos_start(self, pos_start: int):
        if pos_start < 0:
            raise ValueError(f'pos_start ({pos_start}) must be >= 0')
        pos_end = self.pos_end
        if pos_end < pos_start:
            raise ValueError(f'pos_start ({pos_start}) must be <= pos_end ({pos_end})')
        self.pos_start = pos_start
        self.pos_len = pos_end - pos_start

    def set_pos_end(self, pos_end: int):
        pos_start = self.pos_start
        if pos_end < pos_start:
            raise ValueError(f'pos_end ({pos_end}) must be >= pos_start ({pos_start})')
        self.pos_end = pos_end
        self.pos_len = pos_end - pos_start

    def pos_all(self) -> slice:
        pos_len = self.pos_len
        return full_slice if pos_len == self.kv_capacity else slice(None, pos_len)

    def fire_grow(self, old_capacity: int, new_capacity: int):
        pass

    def grow(self, keys: Array, values: Array, kv_count: int) -> None:
        cached_keys = self.keys
        if cached_keys is None:
            key_shape = keys.shape
            added_capacity = align_size(kv_count, self.kv_align)
            B, *head_shape = key_shape[:-2]
            batch_capacity = B
            k_head_dim = key_shape[-1]
            v_head_dim = values.shape[-1]
            k_shape = (batch_capacity, *head_shape, added_capacity, k_head_dim)
            v_shape = k_shape if k_head_dim == v_head_dim else (batch_capacity, *head_shape, added_capacity, v_head_dim)
            self.fire_grow(0, added_capacity)
            self.keys = ten.zeros(k_shape, dtype=keys.dtype)
            self.values = ten.zeros(v_shape, dtype=values.dtype)
            self.batch_count = B
            self.kv_capacity = added_capacity
        else:
            end = self.pos_len
            kv_capacity = self.kv_capacity
            new_size = end + kv_count
            if new_size > kv_capacity:
                key_shape = keys.shape
                B, *head_shape = key_shape[:-2]
                batch_capacity = cached_keys.shape[0]
                k_head_dim = key_shape[-1]
                v_head_dim = values.shape[-1]
                added_capacity = align_size(new_size, self.kv_align, extra=1) - kv_capacity
                k_shape = (batch_capacity, *head_shape, added_capacity, k_head_dim)
                v_shape = k_shape if k_head_dim == v_head_dim else (batch_capacity, *head_shape, added_capacity, v_head_dim)
                new_k = ten.zeros(k_shape, dtype=keys.dtype)
                new_v = ten.zeros(v_shape, dtype=values.dtype)

                if end != kv_capacity:
                    cached_keys = cached_keys[..., :end, :]
                    cached_values = self.values[..., :end, :]
                else:
                    cached_values = self.values

                new_capacity = kv_capacity + added_capacity
                self.fire_grow(kv_capacity, new_capacity)
                self.keys = ten.concatenate([cached_keys, new_k], axis=-2)
                self.values = ten.concatenate([cached_values, new_v], axis=-2)
                self.batch_count = B
                self.kv_capacity = new_capacity

    # def fetch(self, *, batch: AxisSelector = None, n_batches: int = None) -> KV:
    #     return self.get_kvs(self.select(batch=batch, batch_end=n_batches))

    def update(self, keys: Array, values: Array, *, offset: int = None, batch: AxisSelector = None) -> None:
        B = keys.shape[0]
        n_keys = keys.shape[-2]
        pos_end = self.pos_end
        if offset is None:
            offset = pos_end
            grow_keys = n_keys
        else:
            trim = pos_end - offset
            if trim < 0:
                raise ValueError(f'Offset must be None or less than or equal to {self.pos_end}')
            grow_keys = n_keys - trim

        self.grow(keys, values, grow_keys)

        self.set_pos_end(offset + n_keys)

        batch_masker = self.batch_masker
        if batch_masker is None:
            kvs = self.select(pos_start=offset, batch=batch, batch_end=B)

            self.set_kvs(kvs, keys, values)
        else:
            for b in range(B):
                pos_start = batch_masker.get_pos_end(b)
                kvs = self.select(pos_start=pos_start, pos_end=pos_start+n_keys, batch=b)
                self.set_kvs(kvs, keys[b], values[b])

    def set_kvs(self, kvs: Selector, keys: Array, values: Array) -> None:
        # B = keys.shape[0]
        # if B > self.batch_capacity and self.batch_capacity == 1:
        #     self.keys = ten.broadcast_to(self.keys, shape=(B,) + self.keys.shape[1:])
        #     self.values = ten.broadcast_to(self.values, shape=(B,) + self.values.shape[1:])

        self.keys[kvs] = keys
        self.values[kvs] = values

    def forget(self, from_start: int, from_end: int = 0):
        if from_end == 0:
            kvs = select[..., from_start:, :]
        elif 0 < from_end < self.pos_len:
            kvs = select[..., from_start:-from_end, :]
        else:
            raise ValueError(f'from_end ({from_end}) must be between 0 and {self.pos_len}')

        self.keys = self.keys[kvs]
        self.values = self.values[kvs]

        pos_start = self.pos_start + from_start
        pos_end = self.pos_end - from_end
        self.pos_range = pos_start, pos_end

    def cut_segment(self, start: int = 0, stop: int = None, *, n_batches: int = None, forget: bool = False) -> 'KVSegment':
        segment = super().cut_segment(start, stop, n_batches)
        if forget and start == self.pos_start:
            self.forget(stop)
        return segment

    def to_segment(self) -> 'KVSegment':
        keys, values = self.fetch_kv()
        return KVSegment(self.pos_start, keys, values)


class RollingKVBuffer(KVBuffer):

    __slots__ = ('size',)

    size: int

    def __init__(self, size: int):
        super().__init__(0)
        self.size = size

    def grow(self, keys: Array, values: Array, kv_count: int) -> None:
        cached_keys = self.keys
        if cached_keys is None:
            key_shape = keys.shape
            added_capacity = self.size
            B, *head_shape = key_shape[:-2]
            batch_capacity = B
            k_head_dim = key_shape[-1]
            v_head_dim = values.shape[-1]
            k_shape = (batch_capacity, *head_shape, added_capacity, k_head_dim)
            v_shape = k_shape if k_head_dim == v_head_dim else (batch_capacity, *head_shape, added_capacity, v_head_dim)
            self.fire_grow(0, added_capacity)
            self.keys = ten.zeros(k_shape, dtype=keys.dtype)
            self.values = ten.zeros(v_shape, dtype=values.dtype)
            self.batch_count = B
            self.kv_capacity = added_capacity

    def update(self, keys: Array, values: Array, *, offset: int = None, batch: AxisSelector = None) -> None:
        B = keys.shape[0]
        n_keys = keys.shape[-2]
        pos_end = self.pos_end
        if offset is None:
            offset = pos_end
            grow_keys = n_keys
        else:
            trim = pos_end - offset
            if trim < 0:
                raise ValueError(f'Offset must be None or less than or equal to {self.pos_end}')
            grow_keys = n_keys - trim

        self.grow(keys, values, grow_keys)

        self.set_pos_end(offset + n_keys)

        batch_masker = self.batch_masker
        if batch_masker is None:

            size = self.size

            if offset >= size:
                pos_start = offset % size
            else:
                pos_start = offset
            pos_end = pos_start + n_keys
            if pos_end > size:
                pos_end = pos_end % size
                partial = size - pos_start
                kvs = self.select(pos_start=pos_start, pos_end=self.size, batch=batch, batch_end=B)
                self.set_kvs(kvs, keys[..., :partial, :], values[..., :partial, :])

                kvs = self.select(pos_start=0, pos_end=pos_end, batch=batch, batch_end=B)
                self.set_kvs(kvs, keys[..., partial:, :], values[..., partial:, :])

            else:
                kvs = self.select(pos_start=pos_start, pos_end=pos_end, batch=batch, batch_end=B)

                self.set_kvs(kvs, keys, values)
        else:
            raise NotImplementedError("Batch masker is not supported for RollingKVBuffer")
            # for b in range(B):
            #     pos_start = batch_masker.get_pos_end(b)
            #     kvs = self.select(pos_start=pos_start, pos_end=pos_start+n_keys, batch=b)
            #     self.set_kvs(kvs, keys[b], values[b])


KVSegmentId = tuple[int, ...]


class KVSegment(KVArray):

    __slots__ = ['segment_id', 'segment_key', 'selected_queries', 'skipped_queries']

    keys: Array
    values: Array

    segment_id: KVSegmentId
    segment_key: Optional[Array]
    selected_queries: int
    skipped_queries: int

    def __init__(self, pos_start: int, keys: Array, values: Array, segment_id: KVSegmentId = None, segment_key: Array = None):
        super().__init__(pos_start, keys, values)
        self.segment_id = segment_id
        self.segment_key = segment_key
        self.selected_queries = 0
        self.skipped_queries = 0

    def branch(self, idx: Array):
        if self.batch_count == 1:
            return
        cnt, = idx.shape
        if cnt != 0:
            keys, values = self.keys, self.values
            segment_key = self.segment_key
            batch_count = self.batch_count
            new_batch_count = batch_count + cnt
            added_batch_count = new_batch_count - keys.shape[0]
            if added_batch_count > 0:
                batch_step = self.batch_align
                # n_steps = (batch_step + added_batch_count - 1) // batch_step
                # new_capacity = n_steps * batch_step
                new_capacity = align_size(added_batch_count, batch_step)
                if added_batch_count == new_capacity:
                    self.keys = ten.concatenate([keys, keys[idx]], axis=0)
                    self.values = ten.concatenate([values, values[idx]], axis=0)
                    if segment_key is not None:
                        self.segment_key = ten.concatenate([segment_key, segment_key[idx]], axis=0)
                    self.batch_count = new_batch_count
                    return
                else:
                    k_shape = (new_capacity, *keys.shape[1:])
                    v_shape = (new_capacity, *values.shape[1:])
                    new_k = ten.zeros(k_shape, keys.dtype)
                    new_v = ten.zeros(v_shape, values.dtype)
                    self.keys = keys = ten.concatenate([keys, new_k], axis=0)
                    self.values = values = ten.concatenate([values, new_v], axis=0)
                    if segment_key is not None:
                        sk_shape = (new_capacity, *segment_key.shape[1:])
                        new_sk = ten.zeros(sk_shape, segment_key.dtype)
                        self.segment_key = segment_key = ten.concatenate([segment_key, new_sk], axis=0)

            keys[batch_count:new_batch_count, ...] = keys[idx]
            values[batch_count:new_batch_count, ...] = values[idx]
            if segment_key is not None:
                segment_key[batch_count:new_batch_count, ...] = segment_key[idx]
            self.batch_count = new_batch_count

    def to_segment(self) -> 'KVSegment':
        return self

    def _store_arrays(self, arrays: dict, prefix: str = ''):
        super()._store_arrays(arrays, prefix)
        if (segment_key := self.segment_key) is not None:
            B = self.batch_count
            segment_key = segment_key[:B, ...]

            arrays[prefix + 'segment_key'] = segment_key

    @classmethod
    def join(cls, segments: Sequence['KVSegment'], start: int = None, contiguous: bool = True) -> 'KVSegment':
        last: Optional[KVSegment] = None
        keys_list = []
        vals_list = []
        for segment in segments:
            if last is None:
                if start is None: start = segment.pos_start
            elif contiguous and last.pos_end != segment.pos_start:
                raise ValueError(f'Segments are not contiguous: {last} and {segment}')
            keys_list.append(segment.keys)
            vals_list.append(segment.values)
            last = segment

        keys = ten.concatenate(keys_list, axis=-2)
        vals = ten.concatenate(vals_list, axis=-2)
        return KVSegment(start, keys, vals)

    def _repr_args(self) -> str:
        # return f'start={self.pos_start}, length={self.pos_len}'
        return ''

    def _repr_kwargs(self, verbose: int = 0) -> Optional[dict[Optional[str], Any]]:
        if verbose > 0:
            return {
                'start': self.pos_start,
                'length': self.pos_len,
            }
        else:
            return None


class KeyPooler(RootObject):

    def __call__(self, segment: KVSegment) -> Array:
        segment_length = segment.pos_len
        segment_keys = segment.keys / segment_length
        if ten.any(ten.isinf(segment_keys)).item():
            print('Got an infinite segment key!')
        # segment_key = ten.sum(segment.keys, axis=-2)/segment_length
        pooled_key = ten.concatenate([
            ten.sum(ten.maximum(segment_keys, 0.), axis=-2),
            ten.sum(ten.minimum(segment_keys, 0.), axis=-2)
        ], axis=-2)
        if ten.any(ten.isinf(pooled_key)).item():
            print('Got an infinite pooled key!')
        pooled_key = ten.expand_dims(pooled_key, axis=2)
        return pooled_key


class KVSegmentContainer(RootObject, Storable):

    __slots__ = ['container_id']

    container_id: KVSegmentId
    segment_selects: np.ndarray
    segment_attends: np.ndarray

    def __init__(self, container_id: KVSegmentId = None):
        self.container_id = container_id

    @property
    def segment_skips(self) -> np.ndarray:
        return self.segment_attends - self.segment_selects

    @property
    def segment_skip_percent(self) -> float:
        n_segments = len(self)
        if n_segments == 0:
            return 0.
        return (np.sum(self.segment_skips[:n_segments]) / np.sum(self.segment_attends[:n_segments])).item()

    def segment_attention(self, segment_queries: Array, segment_keys: Array) -> Array:
        # pos_queries = ten.maximum(segment_queries, 0.)
        # neg_queries = ten.minimum(segment_queries, 0.)
        # pos_keys = segment_keys[:, :, :, 0, ...]
        # neg_keys = segment_keys[:, :, :, 1, ...]
        # pos_pos_scores = ten.matmul(pos_queries, pos_keys)
        # neg_neg_scores = ten.matmul(neg_queries, neg_keys)
        # pos_neg_scores = ten.matmul(pos_queries, neg_keys)
        # neg_pos_scores = ten.matmul(neg_queries, pos_keys)
        # pos_scores = pos_pos_scores + neg_neg_scores
        # neg_scores = pos_neg_scores + neg_pos_scores
        # alt_segment_scores = pos_scores + neg_scores
        segment_queries = ten.expand_dims(segment_queries, axis=3)
        signed_segment_scores = ten.matmul(segment_queries, segment_keys)
        segment_scores = ten.sum(signed_segment_scores, axis=3)
        max_scores = ten.max(segment_scores, axis=(0, 1, 2, 3))
        max_score = ten.max(max_scores)
        max_scores -= max_score
        return max_scores

    def select_segments(self, segment_queries: Array, segment_keys: Array) -> Array:
        segment_scores = self.segment_attention(segment_queries, segment_keys)
        selected_segments = segment_scores > -2.5
        return selected_segments

    def get_segments(self, segment_queries: Array, call: 'patchlm.cache.ModelCall') -> Iterable[KVSegment]:
        return ()

    def partial_attentions(self, queries: Array, segment_queries: Array, call: 'patchlm.cache.ModelCall') -> Iterable['PartialAttentionScores']:
        for segment in self.get_segments(segment_queries, call):
            yield segment.partial_attention(queries, call)

    def branch(self, idx: Array):
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0


__all__ = [
    'ArrayBuffer',
    'KVBuffer',
    'KVArray',
]