#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from collections.abc import Callable
from typing import Iterable, Optional
from tensile import ten, Array, AxisSelector, Selector
from ..infra import RootObject

full_slice = slice(None)

def align_size(size: int, align: int, extra: int = 0) -> int:
    return size + extra if align == 1 else align * (1 + extra + ((size - 1) // align))


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


def make_indexer(axis: int) -> Callable[[AxisSelector], tuple[AxisSelector, ...]]:
    if axis == -1:
        def index(i) -> tuple[AxisSelector, ...]:
            return ..., i
    elif axis == 0:
        def index(i) -> tuple[AxisSelector, ...]:
            return i, ...
    elif axis > 0:
        before_axis = tuple(full_repeat(axis))
        def index(i) -> tuple[AxisSelector, ...]:
            return *before_axis, i, ...
    else:
        after_axis = tuple(full_after(axis))
        def index(i) -> tuple[AxisSelector, ...]:
            return ..., i, *after_axis
    return index


default_align = 256


class ArrayBuffer(RootObject):
    """
    Provides a dynamically resizable buffer for data storage.

    This class represents a resizable data structure designed to store and manage
    numerical arrays efficiently. It allows appending new data while maintaining
    alignment constraints to optimize memory usage and access. Additionally, this
    class provides functionality to fetch data up to a specific length and to
    create branched versions of the buffer by using indices.

    :ivar buffer: A numerical array holding the buffer's data. Optional at initialization.
    :type buffer: Optional[Array]
    :ivar length: The number of elements currently in the buffer.
    :type length: int
    :ivar align: The alignment size, for memory block alignment constraints.
    :type align: int
    """
    __slots__ = ['buffer', 'length', 'align', 'axis', 'index']

    buffer: Optional[Array]
    length: int
    align: int
    axis: int
    index: Callable[[AxisSelector], tuple[AxisSelector, ...]]

    def __init__(self, buffer: Array = None, length: int = None, align: int = default_align, axis: int = 0):
        # if axis >= 0:
        #     raise ValueError('axis should be less than zero')
        if length is None:
            length = 0 if buffer is None else buffer.shape[axis]
        if axis < 0 and buffer is not None:
            axis += buffer.ndim
        self.buffer = buffer
        self.length = length
        self.align = align
        self.axis = axis
        self.index = make_indexer(axis)

    @property
    def ndim(self) -> int:
        return 0 if self.buffer is None else self.buffer.ndim

    @property
    def shape(self) -> ten.Shape:
        buff = self.buffer
        if buff is None: return ()
        shape = buff.shape
        return *shape[:self.axis], self.length, *shape[self.axis+1:]

    @property
    def capacity(self) -> int:
        if self.buffer is None:
            return 0
        return self.buffer.shape[self.axis]

    def _fix_index(self, item) -> Selector:
        axis = self.axis
        buffer = self.buffer
        if buffer is None:
            raise IndexError(f"Index {item} is out of bounds for axis {axis} in {self.shape} of {self.ndim} dimension")
        if isinstance(item, tuple):
            if any(i is None for i in item):
                raise NotImplementedError('We cannot handle None/newaxis index for multi-dimensional arrays')
            if len(item) <= axis:
                return item
            axis_item = item[axis]
            if isinstance(axis_item, slice):
                return *item[:axis], slice(*axis_item.indices(self.length)), *item[axis + 1:]
            elif isinstance(axis_item, int):
                if axis_item < 0:
                    axis_item += self.length
                    if 0 <= axis_item < self.length:
                        return *item[:axis], axis_item, *item[axis+1:]
                elif 0 <= axis_item < self.length:
                    return item
                raise IndexError(f"Index {item} is out of bounds for axis {axis} in {self.shape} of {self.ndim} dimension")
            raise NotImplementedError(f"Index {item} is not supported for {self.ndim} dimension in {self.shape}")
        else:
            if item is None:
                raise NotImplementedError('We cannot handle None/newaxis index for multi-dimensional arrays')
            if axis == 0:
                if isinstance(item, slice):
                    return slice(*item.indices(self.length))
                elif isinstance(item, int):
                    if item < 0:
                        item += self.length
                    if 0 <= item < self.length:
                        return item
                    raise IndexError(f"Index {item} is out of bounds for axis {axis} in {self.shape} of {self.ndim} dimension")
                raise TypeError(f"Invalid index type: {type(item)}")
            else:
                return item

    def __getitem__(self, item):
        if isinstance(item, tuple):
            raise IndexError('Cannot handle a tuple index')
        if isinstance(item, slice):
            return self.buffer[self.index(slice(*item.indices(self.length)))]
        elif isinstance(item, int):
            if item < 0:
                item += self.length
            if 0 <= item < self.length:
                return self.buffer[self.index(item)]
            # if item == self.length:
            #     self.extend(self.length+1)
            #     return self.buffer[self.index(item)]
            raise IndexError(f"Index {item} is out of bounds for the buffer")
        raise IndexError(f"Invalid index type: {type(item)}")

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            raise IndexError('Cannot handle a tuple index')
        length = self.length
        if isinstance(item, slice):
            if item.start < 0 or item.stop < 0:
                item = slice(*item.indices(length))
            else:
                new_length = max(item.start+1, item.stop)
                self.extend(new_length)
            self.buffer[self.index(item)] = value
        elif isinstance(item, int):
            if item < 0:
                item += length
                if item < 0:
                    raise IndexError(f"Index {item-length} is out of bounds for the buffer")
            if item >= length:
                self.extend(item+1)
            self.buffer[self.index(item)] = value
        else:
            raise IndexError(f"Invalid index type: {type(item)}")

    def fetch(self) -> Array:
        return self.buffer[self.index(slice(0, self.length))]

    def update(self, values: Array):
        self.buffer[self.index(slice(0, self.length))] = values

    def select(self, start: int = None, stop: int = None) -> Selector:
        return self.select_slice(slice(start, stop))

    def select_slice(self, s: slice) -> Selector:
        return self.index(slice(*s.indices(self.length)))

    def append(self, values: Array):
        buffer = self.buffer
        axis = self.axis
        vshape = values.shape
        n_values = vshape[axis]
        if buffer is None:
            added_capacity = align_size(n_values, self.align) - n_values
            if added_capacity == 0:
                self.buffer = values
            else:
                pad = create_pad(values, added_capacity, axis)
                self.buffer = ten.concatenate([values, pad], axis=axis)
            self.length = n_values
            if axis < 0:
                self.axis = values.ndim + axis
                if self.axis < abs(axis):
                    self.index = make_indexer(self.axis)
        else:
            bshape = self.buffer.shape
            if vshape[:axis] != bshape[:axis] or vshape[axis+1:] != bshape[axis+1:]:
                raise ValueError(f"Shape mismatch for axis {axis}: {vshape[:axis]} != {self.buffer.shape[:axis]}")
            length = self.length
            new_length = length + n_values
            self.extend(new_length)
            self.buffer[self.index(slice(length, new_length))] = values

    def extend(self, new_length: int) -> int:
        buffer = self.buffer
        axis = self.axis
        capacity = buffer.shape[axis]
        if new_length > capacity:
            added_capacity = align_size(new_length, self.align) - self.length
            pad = create_pad(buffer, added_capacity, axis)
            self.buffer = ten.concatenate([buffer, pad], axis=axis)
        self.length = new_length
        return new_length


    def branch(self, idx: Array, axis: int = 0):
        """
        Branch and concatenate the buffer based on the provided indices.

        This function takes an buffer of indices and duplicates the elements at those
        indices along the specified axis. If the index buffer is not empty, it updates
        internal buffer by concatenating its elements with the duplicated ones.

        :param idx: A 1-dimensional buffer of integers representing indices of elements
            to duplicate.
        :param axis: An integer representing the axis along which the concatenation
            will happen. Default value is 0.
        :return: None
        """
        buffer = self.buffer
        if buffer is None:
            raise ValueError('Cannot branch when the buffer is None')
        if axis < 0:
            axis = self.ndim + axis
        if axis == self.axis:
            raise ValueError(f'You cannot branch along the same axis as the buffer grows')
        if idx.ndim != 1:
            raise ValueError(f'Index buffer must be 1-dimensional, got {idx.ndim} dimensions')
        self.buffer = ten.concatenate([buffer, ten.take(buffer, idx, axis=axis)], axis=axis)

