#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import Iterable, Optional
from tensile import ten, Array, AxisSelector, Selector
from ..infrastructure import RootObject

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


default_align = 256


class ArrayBuffer(RootObject):
    """
    Provides a dynamically resizable buffer for data storage.

    This class represents a resizable data structure designed to store and manage
    numerical arrays efficiently. It allows appending new data while maintaining
    alignment constraints to optimize memory usage and access. Additionally, this
    class provides functionality to fetch data up to a specific length and to
    create branched versions of the buffer by using indices.

    :ivar array: A numerical array holding the buffer's data. Optional at initialization.
    :type array: Optional[Array]
    :ivar length: The number of elements currently in the buffer.
    :type length: int
    :ivar align: The alignment size, for memory block alignment constraints.
    :type align: int
    """
    __slots__ = ['buffer', 'length', 'align', 'axis', 'before_axis', 'after_axis']

    buffer: Optional[Array]
    length: int
    align: int
    axis: int
    before_axis: tuple[AxisSelector, ...]
    after_axis: tuple[AxisSelector, ...]

    def __init__(self, array: Array = None, length: int = None, align: int = default_align, axis: int = 0):
        # if axis >= 0:
        #     raise ValueError('axis should be less than zero')
        if length is None:
            length = 0 if array is None else array.shape[axis]
        if axis < 0 and array is not None:
            axis += array.ndim
        self.buffer = array
        self.length = length
        self.align = align
        self.axis = axis
        if axis == -1:
            before_axis = ...,
            after_axis = ()
        elif axis == 0:
            before_axis = (),
            after_axis = ...,
        elif axis > 0:
            before_axis = tuple(full_repeat(axis)),
            after_axis = ...,
        else:
            before_axis = ...,
            after_axis = tuple(full_after(axis))
        self.before_axis = before_axis
        self.after_axis = after_axis

    @property
    def ndim(self) -> int:
        return 0 if self.buffer is None else self.buffer.ndim

    @property
    def shape(self) -> ten.Shape:
        array = self.buffer
        if array is None:
            return ()
        shape = list(array.shape)
        shape[self.axis] = self.length
        return tuple(shape)

    @property
    def capacity(self) -> int:
        if self.buffer is None:
            return 0
        return self.buffer.shape[self.axis]

    def _fix_index(self, item) -> Selector:
        axis = self.axis
        buffer = self.buffer
        if buffer is None:
            raise IndexError(f"Index {item} is out of bounds for axis {axis} in {self.shape} of {self.n_dim} dimension")
        if isinstance(item, tuple):
            if len(item) <= axis:
                return item
            axis_item = item[axis]
            if isinstance(axis_item, slice):
                start, stop, step = axis_item.indices(self.length)
                return item[:axis] + (slice(start, stop, step),) + item[axis+1:]
            elif isinstance(axis_item, int):
                if axis_item < self.length:
                    return item
                raise IndexError(f"Index {item} is out of bounds for axis {axis} in {self.shape} of {self.n_dim} dimension")
            raise NotImplementedError(f"Index {item} is not supported for {self.n_dim} dimension in {self.shape}")
        elif axis == 0:
            if isinstance(item, slice):
                start, stop, step = item.indices(self.length)
                slice(start, stop, step)
            elif isinstance(item, int):
                if item < self.length:
                    return item
                raise IndexError(f"Index {item} is out of bounds for axis {axis} in {self.shape} of {self.n_dim} dimension")
            raise TypeError(f"Invalid index type: {type(item)}")
        else:
            return item

    def __getitem__(self, item):
        item = self._fix_index(item)
        return self.buffer[item]

    def __setitem__(self, key, value):
        key = self._fix_index(key)
        self.buffer[key] = value

    def fetch(self) -> Array:
        return self.buffer[self.select()]

    def update(self, values: Array):
        self.buffer[self.select()] = values

    def select(self, start: int = None, stop: int = None) -> Selector:
        stop = self.length if stop is None else min(stop, self.length)
        return self.before_axis + (slice(start, stop),) + self.after_axis

    def select_index(self, index: Array) -> Selector:
        return self.before_axis + (index,) + self.after_axis

    def append(self, values: Array):
        buffer = self.buffer
        axis = self.axis
        n_values = values.shape[axis]
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
        else:
            length = self.length
            new_length = length + n_values
            capacity = buffer.shape[axis]
            if new_length > capacity:
                added_capacity = align_size(new_length, self.align) - length
                pad = create_pad(values, added_capacity, axis)
                self.buffer = ten.concatenate([buffer, pad], axis=axis)
            self.buffer[self.select(length, new_length)] = values
            self.length = new_length

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
        cnt, = idx.shape
        if cnt != 0:
            buffer = self.buffer
            if axis == 0:
                self.buffer = ten.concatenate([buffer, buffer[idx]], axis=axis)
            else:
                self.buffer = ten.concatenate([buffer, buffer[self.select_index(idx)]], axis=axis)

