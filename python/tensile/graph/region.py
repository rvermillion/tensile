#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED
from typing import Generic, Iterable, Optional, TypeVar
from .. import ten

from .common import Array, Base, Index, Indices, Shape, Slice


full_slice = slice(None)
empty_slice = slice(0, 0)
index_dtype = ten.int32


T = TypeVar('T', bound='SetLike')


class SetLike(Generic[T], Base):

    __slots__ = ()

    def overlaps(self, other: T) -> bool:
        raise NotImplementedError()

    def contains(self, other: T) -> bool:
        raise NotImplementedError()

    def equals(self, other: T) -> bool:
        raise NotImplementedError()

    def intersect(self, other: T) -> T:
        raise NotImplementedError()

    def union(self, other: T) -> T:
        raise NotImplementedError()

    def minus(self, other: T) -> T:
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        return self is other or (isinstance(other, self.Element) and self.equals(other))

    def __and__(self, other: T) -> T:
        return self.intersect(other)

    def __or__(self, other: T) -> T:
        return self.union(other)

    def __sub__(self, other: T) -> T:
        return self.minus(other)

    Element: type[T]


class Region(SetLike['Region']):

    __slots__ = ('base',)

    base: Shape

    def __init__(self, base: Shape):
        self.base = base

    @property
    def contiguous(self) -> bool:
        raise NotImplementedError()

    @property
    def full(self) -> bool:
        raise NotImplementedError()

    @property
    def empty(self) -> bool:
        raise NotImplementedError()

    @property
    def key(self) -> tuple[Index, ...]:
        return ()

    @property
    def data_shape(self) -> Shape:
        raise NotImplementedError()

    @property
    def indices(self) -> Optional[tuple['RegionIndex', ...]]:
        return None

    def iter_contiguous(self) -> Iterable['Region']:
        raise NotImplementedError()

    def _validate(self) -> None:
        if not self.base:
            raise ValueError('Cannot create region with empty shape')

    def select(self, axis: int, index: int) -> 'Region':
        size = self.base[axis]
        if index < 0: index += size
        if 0 <= index < size:
            return self._select(axis, index)
        raise IndexError(f'Invalid index {index} for axis {axis} of shape {self.base}')

    def _select(self, axis: int, index: int) -> 'Region':
        raise NotImplementedError()

    def overlaps(self, other: 'Region') -> bool:
        if self.base != other.base:
            raise ValueError(f'Cannot test overlap of regions with different shapes: {self.base} vs {other.base}')
        return self._overlaps(other)

    def _overlaps(self, other: 'Region') -> bool:
        if self is other:
            return not self.empty
        if other.empty or self.empty:
            return False
        if other.full or self.full:
            return True
        raise NotImplementedError()

    def contains(self, other: 'Region') -> bool:
        if self.base != other.base:
            raise ValueError(f'Cannot test contains of regions with different shapes: {self.base} vs {other.base}')
        return self._contains(other)

    def _contains(self, other: 'Region') -> bool:
        if self is other or other.empty or self.full:
            return True
        if other.full:
            return False
        raise NotImplementedError()

    def equals(self, other: 'Region') -> bool:
        return self is other or (self.contains(other) and other.contains(self))

    @property
    def bounded(self) -> 'Region':
        if self.contiguous:
            return self
        indices = tuple(ContiguousRangeIndex.build(*self.bounds(a)) for a in range(self.ndim))
        return IndexedRegion.build(self.base, indices)

    def bounds(self, axis: int) -> tuple[int, int]:
        raise NotImplementedError()

    def intersect(self, other: 'Region') -> 'Region':
        if self.base != other.base:
            raise ValueError(f'Cannot intersect regions with different shapes: {self.base} vs {other.base}')
        if other.empty or self.full:
            return other
        if self.empty or other.full:
            return self
        return self._intersect(other)

    def _intersect(self, other: 'Region') -> 'Region':
        return IntersectionRegion.build(self, other)

    # noinspection PyMethodMayBeStatic
    def _fast_intersect(self, other: 'Region') -> Optional['Region']:
        return None

    def union(self, other: 'Region') -> 'Region':
        if self.base != other.base:
            raise ValueError(f'Cannot union regions with different shapes: {self.base} vs {other.base}')
        if other.empty or self.full:
            return self
        if self.empty or other.full:
            return other
        return self._union(other)

    def _union(self, other: 'Region') -> 'Region':
        return UnionRegion.build(self, other)

    # noinspection PyMethodMayBeStatic
    def _fast_union(self, other: 'Region') -> Optional['Region']:
        return None

    def minus(self, other: 'Region') -> 'Region':
        if self.base != other.base:
            raise ValueError(f'Cannot minus regions with different shapes: {self.base} vs {other.base}')
        if other.empty:
            return self
        if other.contains(self):
            return EmptyRegion.build(self.base)
        return self._minus(other)

    def _minus(self, other: 'Region') -> 'Region':
        return MinusRegion.build(self, other)

    def broadcast(self, shape: Shape) -> 'Region':
        raise NotImplementedError()

    # def __eq__(self, other):
    #     return self is other or (isinstance(other, Region) and self.equals(other))
    #
    # def __ne__(self, other):
    #     if self is other:
    #         return False
    #     return not (isinstance(other, Region) and self.equals(other))

    def __ge__(self, other):
        return isinstance(other, Region) and self.contains(other)

    def _repr_arg(self, short: bool = False) -> str:
        arg = f'{self.base}'
        if indices := self.indices:
            arg += ', [' + ', '.join(ind.display(size) for ind, size in zip(indices, self.base)) + ']'
            if not short:
                arg += f', {self.data_shape}, {self.key}'
        return arg


    @property
    def ndim(self) -> int:
        return len(self.base)

    def map(self, shape: Shape, key: Indices) -> 'Region':
        return self.from_key(shape, key)

    @classmethod
    def from_key(cls, shape: Shape, key: Indices) -> 'Region':
        indices = RegionIndex.from_keys(key, shape)
        return IndexedRegion.build(shape, indices)


# noinspection PyTypeChecker
Region.Element = Region


class RegionIndex(SetLike['RegionIndex']):

    __slots__ = ()

    @property
    def start(self) -> int:
        raise NotImplementedError()

    @property
    def last(self) -> int:
        raise NotImplementedError()

    @property
    def stop(self) -> int:
        raise NotImplementedError()

    @property
    def contiguous(self) -> bool:
        raise NotImplementedError()

    @property
    def empty(self) -> bool:
        return self.count == 0

    def full(self, size: int) -> bool:
        return self.contiguous and self.start == 0 and self.stop == size

    @property
    def index(self) -> Index:
        raise NotImplementedError()

    @property
    def slice(self) -> Optional[Slice]:
        raise NotImplementedError()

    @property
    def count(self) -> int:
        raise NotImplementedError()

    @property
    def bounded(self) -> 'RegionIndex':
        return self if self.contiguous else ContiguousRangeIndex(self.start, self.stop)

    def bounds(self) -> tuple[int, int]:
        return self.start, self.stop

    def array(self) -> Array:
        raise NotImplementedError()

    def iter(self) -> Iterable[int]:
        raise NotImplementedError()

    def has(self, index: int) -> bool:
        raise NotImplementedError()

    def could_overlap(self, other: 'RegionIndex') -> bool:
        return self.start < other.stop and other.start < self.stop

    def overlaps(self, other: 'RegionIndex') -> bool:
        return self.could_overlap(other) and self.contiguous and other.contiguous

    def contains(self, other: 'RegionIndex') -> bool:
        if other.empty:
            return True
        if self.contiguous:
            return self.start <= other.start and other.stop <= self.stop
        if isinstance(other, IntIndex):
            return self.has(other.index)
        for i in other.iter():
            if not self.has(i):
                return False
        return True

    def intersect(self, other: 'RegionIndex') -> 'RegionIndex':
        sstart, sstop = self.bounds()
        ostart, ostop = other.bounds()
        if sstop <= ostart or ostop <= sstart or ostart == ostop or sstart == sstop:
            return EmptyIndex.singleton
        if self.contiguous:
            if sstart <= ostart and sstop >= ostop:
                return other
            if other.contiguous:
                if ostart <= sstart and ostop >= sstop:
                    return self
                if sstop >= ostart and sstart <= ostop:
                    start = max(sstart, ostart)
                    stop = min(sstop, ostop)
                    return ContiguousRangeIndex(start, stop)
        elif other.contiguous and ostart <= sstart and ostop >= sstop:
            return self
        index = []
        for i in self.iter():
            if i >= ostop:
                break
            if i >= ostart and other.has(i):
                index.append(i)
        return ArrayIndex.build(ten.array(index)) if index else EmptyIndex.singleton

    def union(self, other: 'RegionIndex') -> 'RegionIndex':
        sstart, sstop = self.bounds()
        ostart, ostop = other.bounds()
        if self.contiguous:
            if sstart <= ostart and sstop >= ostop:
                return self
            if other.contiguous:
                if ostart <= sstart and ostop >= sstop:
                    return other
                if sstop >= ostart and sstart <= ostop:
                    start = min(sstart, ostart)
                    stop = max(sstop, ostop)
                    return ContiguousRangeIndex(start, stop)
        elif other.contiguous and ostart <= sstart and ostop >= sstop:
            return other
        elif other.empty:
            return self
        index = ten.concatenate([self.array(), other.array()])
        return ArrayIndex.build(index)

    def minus(self, other: 'RegionIndex') -> 'RegionIndex':
        sstart, sstop = self.bounds()
        ostart, ostop = other.bounds()
        if ostart >= sstop or ostop <= sstart:
            return self
        if other.contiguous:
            if ostart <= sstart and ostop >= sstop:
                return EmptyIndex.singleton
            if self.contiguous:
                if ostart <= sstart:
                    return RangeIndex.build(ostop, sstop)
                if ostop <= sstop:
                    return RangeIndex.build(sstart, ostart)
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.count

    def _validate(self) -> None:
        if self.start < 0:
            raise ValueError(f'Negative start: {self.start}')
        if self.stop <= self.start:
            raise ValueError(f'Invalid range: {self.start} <= {self.stop}')
        # if size is not None and self.stop > size:
        #     raise ValueError(f'Index out of bounds: {self.stop} > {size}')

    def validate_size(self, size: int) -> None:
        self.validate()
        if self.stop > size:
            raise ValueError(f'Index out of bounds: {self.stop} > {size}')

    def display(self, size: int = None) -> str:
        raise NotImplementedError()

    def _repr_arg(self, short: bool = False) -> str:
        return self.display()

    # def __and__(self, other: 'RegionIndex') -> 'RegionIndex':
    #     return self.intersect(other)
    #
    # def __or__(self, other: 'RegionIndex') -> 'RegionIndex':
    #     return self.union(other)
    #
    # def __sub__(self, other: 'RegionIndex') -> 'RegionIndex':
    #     return self.minus(other)

    @classmethod
    def range(cls, start: int, stop: int, step: int = None) -> 'RegionIndex':
        return RangeIndex.build(start, stop, step)

    @classmethod
    def single(cls, index: int) -> 'RegionIndex':
        return IntIndex(index)

    @classmethod
    def from_keys(cls, keys: Indices, shape: Shape) -> tuple['RegionIndex', ...]:
        if not shape:
            raise ValueError('Cannot create indices from empty shape')
        if keys is ...:
            return tuple(RangeIndex.build(0, size, 1) for size in shape)
        if isinstance(keys, (int, slice)):
            index = cls.from_key(keys, shape[0])
            if len(shape) > 1:
                return index, *(RangeIndex.build(0, size, 1) for size in shape[1:])
            return index,
        if isinstance(keys, tuple):
            indices = []
            seen_ellipsis = False
            k = 0
            d = 0
            ndim = len(shape)
            nkeys = len(keys)
            # if nkeys > ndim:
            #     raise ValueError(f'Too many keys: {keys} for shape {shape}')
            while k < nkeys and d < ndim:
                key = keys[k]
                if key is ...:
                    if seen_ellipsis:
                        raise ValueError('Cannot have multiple ellipses in indices')
                    seen_ellipsis = True
                    for i in range(ndim - nkeys + 1):
                        indices.append(RangeIndex.build(0, shape[d], 1))
                        d += 1
                else:
                    indices.append(cls.from_key(key, shape[d]))
                    d += 1
                k += 1
            while d < ndim:
                indices.append(RangeIndex.build(0, shape[d], 1))

            if len(indices) != ndim:
                raise ValueError(f'Too few keys: {keys} for shape {shape}')

            return tuple(indices)
        raise ValueError(f'Invalid keys: {keys} for shape {shape}')

    @classmethod
    def from_key(cls, key: Index, size: int) -> 'RegionIndex':
        if isinstance(key, int):
            if key < 0:
                key += size
            if 0 <= key < size:
                return IntIndex(key)
            return EmptyIndex.singleton
            # raise IndexError(f'Index {key} out of bounds for shape {size}')
        elif isinstance(key, slice):
            return RangeIndex.build(*key.indices(size))
        elif ten.is_array(key):
            return ArrayIndex.build(key)
        else:
            raise ValueError(f'Invalid key: {key} for size {size}')


# noinspection PyTypeChecker
RegionIndex.Element = RegionIndex


class EmptyIndex(RegionIndex):

    __slots__ = ()

    start: int = 0
    stop: int = 0
    last: int = -1
    count: int = 0
    contiguous: bool = True
    empty: bool = True
    index: Index = empty_slice
    slice: Slice = empty_slice

    def _validate(self) -> None:
        pass

    def has(self, index: int) -> bool:
        return False

    def could_overlap(self, other: 'RegionIndex') -> bool:
        return False

    def overlaps(self, other: 'RegionIndex') -> bool:
        return False

    def contains(self, other: RegionIndex) -> bool:
        return self is other

    def intersect(self, other: RegionIndex) -> RegionIndex:
        return self

    def union(self, other: RegionIndex) -> RegionIndex:
        return other

    def minus(self, other: RegionIndex) -> RegionIndex:
        return self

    def display(self, size: int = None) -> str:
        return 'empty'

    def _repr_arg(self, short: bool = False) -> str:
        return ''

    singleton: 'EmptyIndex'


EmptyIndex.singleton = EmptyIndex.new()


class IntIndex(RegionIndex):

    __slots__ = ('index', )

    index: int
    count: int = 1
    contiguous: bool = True
    empty: bool = False

    def __init__(self, index: int):
        self.index = index

    @property
    def start(self) -> int:
        return self.index

    @property
    def last(self) -> int:
        return self.index

    @property
    def stop(self) -> int:
        return self.last + 1

    @property
    def key(self) -> int:
        return self.index

    @property
    def slice(self) -> Slice:
        ind = self.index
        return slice(ind, ind + 1, 1)

    @property
    def bounded(self) -> 'RegionIndex':
        return self

    def full(self, size: int) -> bool:
        return size == 1 and self.index == 0

    def _validate(self) -> None:
        if self.index < 0:
            raise ValueError(f'Negative index: {self.index}')
        # if self.index >= size:
        #     raise ValueError(f'Index out of bounds: {self.index} >= {size}')

    def validate_size(self, size: int) -> None:
        self.validate()
        if self.index >= size:
            raise ValueError(f'Index out of bounds: {self.index} >= {size}')

    def array(self) -> Array:
        return ten.array([self.index], dtype=index_dtype)

    def iter(self) -> Iterable[int]:
        return self.index,

    def has(self, index: int) -> bool:
        return self.index == index

    def contains(self, other: RegionIndex) -> bool:
        return other.start == self.index and other.last == self.index

    def could_overlap(self, other: 'RegionIndex') -> bool:
        return other.start <= self.index < other.stop

    def overlaps(self, other: RegionIndex) -> bool:
        return other.has(self.index)

    def intersect(self, other: RegionIndex) -> RegionIndex:
        if other.has(self.index):
            return self
        return EmptyIndex.singleton

    def union(self, other: RegionIndex) -> RegionIndex:
        if other.has(self.index):
            return other
        if isinstance(other, ContiguousRangeIndex):
            if self.index == other.stop:
                return ContiguousRangeIndex(other.start, other.stop + 1)
            if self.index == other.start - 1:
                return ContiguousRangeIndex(other.start - 1, other.stop)
        return super().union(other)

    def minus(self, other: RegionIndex) -> RegionIndex:
        if other.has(self.index):
            return EmptyIndex.singleton
        return self

    def display(self, size: int = None) -> str:
        return str(self.index)


class ArrayIndex(RegionIndex):

    __slots__ = ('index', )

    index: Array
    contiguous: bool = False

    def __init__(self, index: Array):
        self.index = index

    @property
    def start(self) -> int:
        return self.index[0].item()

    @property
    def last(self) -> int:
        return self.index[-1].item()

    @property
    def stop(self) -> int:
        return self.last + 1

    def _validate(self) -> None:
        super()._validate()
        if not ten.is_integer(self.index.dtype):
            raise ValueError(f'Index must be integer, got {self.index.dtype}')
        if self.index.ndim != 1:
            raise ValueError(f'Index must be 1D, got {self.index.ndim}D')
        if ten.any(self.index < 0).item():
            raise ValueError(f'Negative index: {self.index}')
        if ten.any(self.index[:-1] > self.index[1:]).item():
            raise ValueError(f'Index not sorted: {self.index}')
        # if size is not None:
        #     if ten.any(self.index >= size).item():
        #         raise ValueError(f'Index out of bounds: {self.index} >= {size}')

    def validate_size(self, size: int) -> None:
        self.validate()
        if ten.any(self.index >= size).item():
            raise ValueError(f'Index out of bounds: {self.index} >= {size}')

    @property
    def count(self) -> int:
        count = self.index.size
        steps = self.index[1:] - self.index[:-1]
        return count - ten.sum(steps == 0).item()

    def array(self) -> Array:
        return self.index

    def iter(self) -> Iterable[int]:
        for i in self.index:
            yield i.item()

    def has(self, index: int) -> bool:
        return ten.any(self.index == index).item()

    def display(self, size: int = None) -> str:
        return str(self.index)

    @classmethod
    def build(cls, index: Array, fast: bool = False) -> RegionIndex:
        if not ten.is_integer(index.dtype):
            raise ValueError(f'Index must be integer, got {index.dtype}')
        if index.size == 0:
            return EmptyIndex.singleton
        if index.ndim != 1:
            index = ten.reshape(index, (-1,))
        start = index[0].item()
        if index.size == 1:
            return IntIndex.new(start)
        index = ten.sort(index)
        last = index[-1].item()
        if start == last:
            return IntIndex.new(start)
        if not fast:
            stop = last + 1
            steps = index[1:] - index[:-1]
            max_step = ten.max(steps).item()
            if max_step == 1:
                # If the maximum step size is 1, we can use a ContiguousRangeIndex
                return ContiguousRangeIndex.new(start, stop)

            # We replace zero steps (i.e duplicate indices) with a large step to find the
            # minimum non-zero step size
            min_step = ten.min(ten.where(steps == 0, max_step, steps)).item()
            if min_step == max_step:
                # If the minimum step size is the same as the maximum step size, we can
                # use a StepRangeIndex
                return StepRangeIndex.new(start, stop, min_step)
        return cls.new(index)


class RangeIndex(RegionIndex):

    __slots__ = ('start', 'stop', )

    start: int
    stop: int
    step: int

    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop

    @property
    def count(self) -> int:
        return (self.stop - self.start) // self.step

    @property
    def index(self) -> Index:
        return slice(self.start, self.stop, self.step)

    @property
    def slice(self) -> Slice:
        return slice(self.start, self.stop, self.step)

    def array(self) -> Array:
        return ten.arange(self.start, self.stop, self.step)

    def iter(self) -> Iterable[int]:
        return range(self.start, self.stop, self.step)

    @classmethod
    def build(cls, start: int, stop: int, step: int = None) -> RegionIndex:
        if step is None:
            step = 1
        elif step == -1:
            start, stop = stop, start
            step = 1

        if start < stop:
            if start + step >= stop:
                return IntIndex.new(start)
            if step == 1:
                return ContiguousRangeIndex.new(start, stop)
            steps = (stop - start - 1) // step
            stop = start + steps * step + 1
            return StepRangeIndex.new(start, stop, step)

        return EmptyIndex.singleton
        # raise ValueError(f'Invalid range: {start} < {stop}')

    def display(self, size: int = None) -> str:
        if size is not None and self.stop == size:
            if self.start == 0:
                return ':'
            return f'{self.start}:'
        if self.start == 0:
            return f':{self.stop}'
        return f'{self.start}:{self.stop}'


class ContiguousRangeIndex(RangeIndex):

    __slots__ = ()

    step: int = 1
    contiguous: bool = True

    @property
    def count(self) -> int:
        return self.stop - self.start

    @property
    def last(self) -> int:
        return self.stop - 1

    @property
    def bounded(self) -> 'RegionIndex':
        return self

    def has(self, index: int) -> bool:
        return self.start <= index < self.stop

    def overlaps(self, other: RegionIndex) -> bool:
        if self.start < other.stop and other.start < self.stop:
            if other.contiguous:
                return True
        return False


class FullIndex(ContiguousRangeIndex):

    full: bool = True
    empty: bool = False
    index: Index = full_slice

    def __init__(self, size: int):
        super().__init__(0, size)

    def has(self, index: int) -> bool:
        return 0 <= index < self.stop

    def contains(self, other: RegionIndex) -> bool:
        return True

    def overlaps(self, other: RegionIndex) -> bool:
        return not other.empty

    def intersect(self, other: RegionIndex) -> RegionIndex:
        return other

    def union(self, other: RegionIndex) -> RegionIndex:
        return self


class StepRangeIndex(RangeIndex):

    __slots__ = ('step', )

    contiguous: bool = False

    def __init__(self, start: int, stop: int, step: int):
        if step == 1:
            raise ValueError('Step cannot be 1')
        super().__init__(start, stop)
        self.step = step

    @property
    def bounded(self) -> 'RegionIndex':
        return ContiguousRangeIndex(self.start, self.stop)

    def _validate(self) -> None:
        super()._validate()
        if self.step == 1:
            raise ValueError(f'Step cannot be 1: {self.start} < {self.stop} by {self.step}')

    def has(self, index: int) -> bool:
        if self.start <= index < self.stop:
            return (index - self.start) % self.step == 0
        return False

    def display(self, size: int = None) -> str:
        return super().display(size) + f':{self.step}'


class IndexedRegion(Region):

    __slots__ = ('indices',)

    indices: tuple[RegionIndex, ...]
    contiguous: bool = False

    def __init__(self, base: Shape, indices: tuple[RegionIndex, ...]):
        super().__init__(base)
        self.indices = indices

    def _select(self, axis: int, index: int) -> 'Region':
        ind: RegionIndex = self.indices[axis]
        if ind.has(index):
            if isinstance(ind, IntIndex):
                return self
            indices = tuple(IntIndex(index) if a == axis else ind for a, ind in enumerate(self.indices))
            return IndexedRegion(self.base, indices)
        return EmptyRegion(self.base)

    @property
    def key(self) -> tuple[Index, ...]:
        return tuple(ind.index for ind in self.indices)

    @property
    def data_shape(self) -> Shape:
        return tuple(ind.count for ind in self.indices)
        # shape = []
        # for ind in self.indices:
        #     if isinstance(ind, IntIndex):
        #         pass
        #     else:
        #         shape.append(ind.count)
        # return tuple(shape)

    @property
    def full(self) -> bool:
        for ind, size in zip(self.indices, self.base):
            if not ind.full(size):
                return False
        return True

    @property
    def empty(self) -> bool:
        return any(ind.empty for ind in self.indices)

    def _validate(self) -> None:
        super()._validate()
        if len(self.indices) != len(self.base):
            raise ValueError(f'Invalid region shape: {self.base} with indices {self.indices}')
        if any(ind.empty for ind in self.indices):
            raise ValueError('Empty indices')

        array_size = 0
        for ind, size in zip(self.indices, self.base):
            ind.validate_size(size)
            if isinstance(ind, ArrayIndex):
                if array_size == 0:
                    array_size = ind.count
                elif array_size != ind.count:
                    raise ValueError('All array indices must have the same length')

    def broadcast(self, shape: Shape) -> Region:
        if shape == self.base:
            return self
        indices = []
        for size in shape[:-len(self.indices)]:
            indices.append(Regions.range_index(0, size))

        indices += self.indices

        return Regions.indexed(shape, tuple(indices))

    def _overlaps(self, other: Region) -> bool:
        if other_inds := other.indices:
            return all(sind.overlaps(oind) for sind, oind in zip(self.indices, other_inds))
        return super()._overlaps(other)

    def _contains(self, other: Region) -> bool:
        if isinstance(other, IndexedRegion):
            return all(sind.contains(oind) for sind, oind in zip(self.indices, other.indices))
        return super()._contains(other)

    def _intersect(self, other: 'Region') -> 'Region':
        if isinstance(other, IndexedRegion):
            indices = tuple(ind.intersect(other_ind) for ind, other_ind in zip(self.indices, other.indices))
            return Regions.indexed(self.base, indices)
        return super()._intersect(other)

    @property
    def bounded(self) -> Region:
        indices = tuple(ind.bounded for ind in self.indices)
        return Regions.contiguous(self.base, indices)

    def bounds(self, axis: int) -> tuple[int, int]:
        return self.indices[axis].bounded

    @classmethod
    def build(cls, base: Shape, indices: tuple[RegionIndex, ...]) -> Region:
        if len(base) != len(indices):
            raise ValueError(f'Invalid region shape: {base} with indices {indices}')
        if any(ind.empty for ind in indices):
            return Regions.empty(base)
        if all(ind.contiguous for ind in indices):
            return ContiguousRegion.new(base, indices)
        return IndexedRegion.new(base, indices)


class ContiguousRegion(IndexedRegion):

    __slots__ = ()

    contiguous: bool = True

    @property
    def key(self) -> tuple[Index, ...]:
        return tuple(ind.slice for ind in self.indices)

    @property
    def bounded(self) -> Region:
        return self

    def iter_contiguous(self) -> Iterable['Region']:
        return self,

    def _validate(self) -> None:
        super()._validate()
        if any(not ind.contiguous for ind in self.indices):
            raise ValueError(f'Non-contiguous indices: {self}')


class FullRegion(Region):

    __slots__ = ()

    full: bool = True
    empty: bool = False
    contiguous: bool = True

    @property
    def bounded(self) -> 'Region':
        return self

    def bounds(self, axis: int) -> tuple[int, int]:
        return 0, self.base[axis]

    def iter_contiguous(self) -> Iterable['Region']:
        return self,

    @property
    def indices(self) -> tuple[RegionIndex, ...]:
        return tuple(RangeIndex.build(0, size, 1) for size in self.base)

    @property
    def key(self) -> tuple[Index, ...]:
        return (full_slice,) * self.ndim

    @property
    def data_shape(self) -> Shape:
        return self.base

    def _select(self, axis: int, index: int) -> 'Region':
        if self.base[axis] == 1:
            return self
        indices = tuple(IntIndex(index) if a == axis else ax for a, ax in enumerate(self.indices))
        return IndexedRegion(self.base, indices)

    def _contains(self, other: 'Region') -> bool:
        return True

    def _overlaps(self, other: 'Region') -> bool:
        return True

    def _intersect(self, other: 'Region') -> 'Region':
        return other

    def _union(self, other: 'Region') -> 'Region':
        return self

    def broadcast(self, shape: Shape) -> Region:
        return self if shape == self.base else FullRegion(shape)

    @classmethod
    def build(cls, base: Shape) -> Region:
        return cls.new(base)


class EmptyRegion(Region):

    __slots__ = ()

    full: bool = False
    empty: bool = True
    contiguous: bool = True

    @property
    def bounded(self) -> 'Region':
        return self

    @property
    def indices(self) -> tuple[RegionIndex, ...]:
        return (EmptyIndex.singleton, ) * self.ndim

    @property
    def key(self) -> tuple[Index, ...]:
        return (empty_slice,) * self.ndim

    @property
    def data_shape(self) -> Shape:
        return ()

    def iter_contiguous(self) -> Iterable['Region']:
        return self,

    def bounds(self, axis: int) -> tuple[int, int]:
        return 0, 0

    def _select(self, axis: int, index: int) -> 'Region':
        return self

    def _contains(self, other: 'Region') -> bool:
        return other.empty

    def _overlaps(self, other: 'Region') -> bool:
        return False

    def _intersect(self, other: 'Region') -> 'Region':
        return self

    def _union(self, other: 'Region') -> 'Region':
        return other

    def _minus(self, other: 'Region') -> 'Region':
        return self

    def broadcast(self, shape: Shape) -> Region:
        return self if shape == self.base else EmptyRegion(shape)

    @classmethod
    def build(cls, base: Shape) -> Region:
        return cls.new(base)


class MinusRegion(Region):

    __slots__ = ('source', 'excluded',)

    source: Region
    excluded: Region

    def __init__(self, source: Region, excluded: Region):
        super().__init__(source.base)
        self.source = source
        self.excluded = excluded

    @classmethod
    def build(cls, source: Region, excluded: Region) -> Region:
        if source.base != excluded.base:
            raise ValueError(f'Regions must have the same base shape: {source.base} != {excluded.base}')
        if source.overlaps(excluded):
            return cls.new(source, excluded)
        return source


class CompositeRegion(Region):

    __slots__ = ('regions',)

    regions: tuple[Region, ...]

    def __init__(self, regions: tuple[Region, ...]):
        base = regions[0].base
        super().__init__(base)
        self.regions = regions

    @classmethod
    def build(cls, *regions: Region) -> Region:
        return cls.create(regions)

    @classmethod
    def create(cls, regions: tuple[Region, ...]) -> Region:
        if regions:
            if len(regions) == 1:
                return regions[0]
            shape = regions[0].base
            if not all(reg.base == shape for reg in regions[1:]):
                raise ValueError(f'Regions must have the same base shape: {regions}')
        else:
            raise ValueError('No regions provided')
        return cls.new(regions)


class IntersectionRegion(CompositeRegion):

    __slots__ = ()

    def _intersect(self, other: Region) -> Region:
        return IntersectionRegion(self.regions + (other,))

    def _union(self, other: Region) -> Region:
        return UnionRegion.build(self, other)


class UnionRegion(CompositeRegion):

    __slots__ = ()

    def _intersect(self, other: Region) -> Region:
        return Regions.union(*(region & other for region in self.regions))

    def _union(self, other: Region) -> Region:
        return Regions.union(*self.regions, other)


class Regions:

    @staticmethod
    def empty(shape: Shape) -> Region:
        return FullRegion.build(shape)

    @staticmethod
    def full(shape: Shape) -> Region:
        return FullRegion.build(shape)

    @staticmethod
    def indexed(shape: Shape, indices: tuple[RegionIndex, ...]) -> Region:
        return IndexedRegion.build(shape, indices)

    @staticmethod
    def contiguous(shape: Shape, indices: tuple[RegionIndex, ...]) -> Region:
        return ContiguousRegion.build(shape, indices)

    @staticmethod
    def intersect(*regions: Region, shape: Shape = None) -> Region:
        if not regions:
            if shape is None:
                raise ValueError('No regions or shape provided')
            return Regions.full(shape)
        if len(regions) == 1:
            return regions[0]
        components = []
        for region in regions:
            if shape is None:
                shape = region.base
            elif shape != region.base:
                raise ValueError(f'Regions must have the same base shape: {region.base} != {shape}')
            if region.empty:
                return Regions.empty(shape)
            if isinstance(region, IntersectionRegion):
                components.extend(region.regions)
            elif not region.full:
                components.append(region)
        if len(components) == 1:
            return components[0]
        return IntersectionRegion.build(*components)

    @staticmethod
    def union(*regions: Region, shape: Shape = None) -> Region:
        if not regions:
            if shape is None:
                raise ValueError('No regions or shape provided')
            return Regions.empty(shape)
        if len(regions) == 1:
            return regions[0]
        components = []
        for region in regions:
            if shape is None:
                shape = region.base
            elif shape != region.base:
                raise ValueError(f'Regions must have the same base shape: {region.base} != {shape}')
            if region.full:
                return Regions.full(shape)
            if isinstance(region, UnionRegion):
                components.extend(region.regions)
            else:
                components.append(region)
            if isinstance(region, UnionRegion):
                components.extend(region.regions)
            elif not region.empty:
                components.append(region)
        if len(components) == 1:
            return components[0]
        return UnionRegion.build(*components)

    @staticmethod
    def minus(source: Region, excluded: Region) -> Region:
        return MinusRegion.build(source, excluded)

    @staticmethod
    def empty_index() -> RegionIndex:
        return EmptyIndex.singleton

    @staticmethod
    def full_index(size: int) -> RegionIndex:
        return FullIndex.new(size)

    @staticmethod
    def range_index(start: int, stop: int, step: int = None) -> RegionIndex:
        return RangeIndex.build(start, stop, step)

    @staticmethod
    def int_index(index: int) -> RegionIndex:
        return IntIndex.new(index)

    @staticmethod
    def array_index(index: Array) -> RegionIndex:
        return ArrayIndex.build(index)
