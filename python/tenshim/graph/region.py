#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED
from typing import Generic, Iterable, TypeVar
from .. import ten

from .common import Array, Base, DType, Index, Indices, Shape, TensorType


full = slice(None, None, None)


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

    def validate(self) -> None:
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
        raise NotImplementedError()

    def contains(self, other: 'Region') -> bool:
        raise NotImplementedError()

    def equals(self, other: 'Region') -> bool:
        raise NotImplementedError()

    @property
    def bounds(self) -> 'Region':
        if self.contiguous:
            return self
        indices = tuple(ContiguousRangeIndex.build(*self.bound(a)) for a in range(self.ndims))
        return IndexedRegion(self.base, indices)

    def bound(self, axis: int) -> tuple[int, int]:
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

    def minus(self, other: 'Region') -> 'Region':
        if self.base != other.base:
            raise ValueError(f'Cannot minus regions with different shapes: {self.base} vs {other.base}')
        if other.empty:
            return self
        if other.full:
            return EmptyRegion(self.base)
        return self._minus(other)

    def _minus(self, other: 'Region') -> 'Region':
        return MinusRegion.build(self, other)

    def broadcast(self, shape: Shape) -> 'Region':
        raise NotImplementedError()

    def __eq__(self, other):
        return self is other or (isinstance(other, Region) and self.equals(other))

    def __ne__(self, other):
        if self is other:
            return False
        return not (isinstance(other, Region) and self.equals(other))

    def __ge__(self, other):
        return isinstance(other, Region) and self.contains(other)

    # def __and__(self, other: 'Region') -> 'Region':
    #     return self.intersect(other)
    #
    # def __or__(self, other: 'Region') -> 'Region':
    #     return self.union(other)
    #
    # def __sub__(self, other: 'Region') -> 'Region':
    #     return self.minus(other)

    def _repr_args(self) -> Iterable:
        return self.base

    @property
    def ndims(self) -> int:
        return len(self.base)

    def map(self, shape: Shape, key: Indices) -> 'Region':
        return self.from_key(shape, key)

    @classmethod
    def from_key(cls, shape: Shape, key: Indices) -> 'Region':
        indices = RegionIndex.from_keys(key, shape)
        if any(index.empty for index in indices):
            return EmptyRegion(shape)
        return IndexedRegion(shape, indices)


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
    def count(self) -> int:
        raise NotImplementedError()

    @property
    def bounds(self) -> 'RegionIndex':
        return self if self.contiguous else ContiguousRangeIndex(self.start, self.stop)

    def array(self) -> Array:
        raise NotImplementedError()

    def has(self, index: int) -> bool:
        raise NotImplementedError()

    def could_overlap(self, other: 'RegionIndex') -> bool:
        return self.start < other.stop and other.start < self.stop

    def overlaps(self, other: 'RegionIndex') -> bool:
        return self.could_overlap(other) and self.contiguous and other.contiguous

    def contains(self, other: 'RegionIndex') -> bool:
        return self.contiguous and self.start <= other.start and other.stop <= self.stop

    def intersect(self, other: 'RegionIndex') -> 'RegionIndex':
        if self.contiguous and self.start <= other.start and self.stop >= other.stop:
            return other
        if other.contiguous and other.start <= self.start and other.stop >= self.stop:
            return self
        raise NotImplementedError()

    def union(self, other: 'RegionIndex') -> 'RegionIndex':
        if other.contiguous and other.start <= self.start and other.stop >= self.stop:
            return other
        if self.contiguous and self.start <= other.start and self.stop >= other.stop:
            return self
        raise NotImplementedError()

    def minus(self, other: 'RegionIndex') -> 'RegionIndex':
        if other.start >= self.stop or other.stop <= self.start:
            return self
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.count

    def validate(self, size: int) -> None:
        if self.start < 0:
            raise ValueError(f'Negative start: {self.start}')
        if self.stop <= self.start:
            raise ValueError(f'Invalid range: {self.start} <= {self.stop}')
        if self.stop > size:
            raise ValueError(f'Index out of bounds: {self.stop} > {size}')

    def display(self, size: int = None) -> str:
        raise NotImplementedError()

    def _repr_arg(self) -> str:
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
            ndims = len(shape)
            nkeys = len(keys)
            # if nkeys > ndims:
            #     raise ValueError(f'Too many keys: {keys} for shape {shape}')
            while k < nkeys and d < ndims:
                key = keys[k]
                if key is ...:
                    if seen_ellipsis:
                        raise ValueError('Cannot have multiple ellipses in indices')
                    seen_ellipsis = True
                    for i in range(ndims - nkeys + 1):
                        indices.append(RangeIndex.build(0, shape[d], 1))
                        d += 1
                else:
                    indices.append(cls.from_key(key, shape[d]))
                    d += 1
                k += 1
            while d < ndims:
                indices.append(RangeIndex.build(0, shape[d], 1))

            if len(indices) != ndims:
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

    def validate(self, size: int) -> None:
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

    singleton: 'EmptyIndex'


EmptyIndex.singleton = EmptyIndex()


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
    def bounds(self) -> 'RegionIndex':
        return self

    def full(self, size: int) -> bool:
        return size == 1 and self.index == 0

    def validate(self, size: int) -> None:
        if self.index < 0:
            raise ValueError(f'Negative index: {self.index}')
        if self.index >= size:
            raise ValueError(f'Index out of bounds: {self.index} >= {size}')

    def array(self) -> Array:
        return ten.array(self.index)

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

    def validate(self, size: int) -> None:
        super().validate(size)
        if not ten.is_integer(self.index.dtype):
            raise ValueError(f'Index must be integer, got {self.index.dtype}')
        if self.index.ndim != 1:
            raise ValueError(f'Index must be 1D, got {self.index.ndim}D')
        if ten.any(self.index < 0).item():
            raise ValueError(f'Negative index: {self.index}')
        if ten.any(self.index[:-1] > self.index[1:]).item():
            raise ValueError(f'Index not sorted: {self.index}')
        if ten.any(self.index >= size).item():
            raise ValueError(f'Index out of bounds: {self.index} >= {size}')

    def array(self) -> Array:
        return self.index

    def has(self, index: int) -> bool:
        return ten.any(self.index == index).item()

    def intersect(self, other: RegionIndex) -> RegionIndex:
        raise NotImplementedError()

    def display(self, size: int = None) -> str:
        return str(self.index)

    @classmethod
    def build(cls, index: Array) -> RegionIndex:
        if not ten.is_integer(index.dtype):
            raise ValueError(f'Index must be integer, got {index.dtype}')
        if index.size == 0:
            return EmptyIndex.singleton
        if index.ndim != 1:
            index = ten.reshape(index, (-1,))
        start = index[0].item()
        if index.size == 1:
            return IntIndex(start)
        index = ten.sort(index)
        last = index[-1].item()
        if start == last:
            return IntIndex(start)
        stop = last + 1
        if index.size >= stop - start:
            return ContiguousRangeIndex(start, stop)
        return cls(index)


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

    def array(self) -> Array:
        return ten.arange(self.start, self.stop, self.step)

    @classmethod
    def build(cls, start: int, stop: int, step: int = None) -> RegionIndex:
        if step is None:
            step = 1
        elif step == -1:
            start, stop = stop, start
            step = 1

        if start < stop:
            if start + step >= stop:
                return IntIndex(start)
            if step == 1:
                return ContiguousRangeIndex(start, stop)
            steps = (stop - start - 1) // step
            stop = start + steps * step + 1
            return StepRangeIndex(start, stop, step)

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
    def bounds(self) -> 'RegionIndex':
        return self

    def has(self, index: int) -> bool:
        return self.start <= index < self.stop

    def overlaps(self, other: RegionIndex) -> bool:
        if self.start < other.stop and other.start < self.stop:
            if other.contiguous:
                return True
        return False

    def intersect(self, other: RegionIndex) -> RegionIndex:
        if isinstance(other, ContiguousRangeIndex):
            if self.start >= other.stop:
                return EmptyIndex.singleton
            if self.stop <= other.start:
                return EmptyIndex.singleton
            start = max(self.start, other.start)
            stop = min(self.stop, other.stop)
            return ContiguousRangeIndex(start, stop)
        raise NotImplementedError()

    def union(self, other: RegionIndex) -> RegionIndex:
        if isinstance(other, ContiguousRangeIndex):
            if self.start > other.stop:
                raise NotImplementedError()
            if self.stop < other.start:
                raise NotImplementedError()
            start = min(self.start, other.start)
            stop = max(self.stop, other.stop)
            return ContiguousRangeIndex(start, stop)
        raise NotImplementedError()


class StepRangeIndex(RangeIndex):

    __slots__ = ('step', )

    contiguous: bool = False

    def __init__(self, start: int, stop: int, step: int):
        if step == 1:
            raise ValueError('Step cannot be 1')
        super().__init__(start, stop)
        self.step = step

    @property
    def bounds(self) -> 'RegionIndex':
        return ContiguousRangeIndex(self.start, self.stop)

    def validate(self, size: int) -> None:
        super().validate(size)
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

    def __init__(self, base: Shape, indices: tuple[RegionIndex, ...]):
        super().__init__(base)
        self.indices = indices

    def _select(self, axis: int, index: int) -> 'Region':
        ind: RegionIndex = self.indices[axis]
        if ind.has(index):
            if isinstance(ind, IntIndex):
                return self
            indices = tuple(IntIndex(index) if a == axis else ax for a, ax in enumerate(self.indices))
            return IndexedRegion(self.base, indices)
        return EmptyRegion(self.base)

    @property
    def full(self) -> bool:
        for ind, size in zip(self.indices, self.base):
            if not ind.full(size):
                return False
        return True

    @property
    def empty(self) -> bool:
        return any(ind.empty for ind in self.indices)

    def validate(self) -> None:
        super().validate()
        if len(self.indices) != len(self.base):
            raise ValueError(f'Invalid region shape: {self.base} with indices {self.indices}')
        if any(ind.empty for ind in self.indices):
            raise ValueError('Empty indices')
        for ind, size in zip(self.indices, self.base):
            ind.validate(size)

    def broadcast(self, shape: Shape) -> Region:
        if shape == self.base:
            return self
        indices = []
        for size in shape[:-len(self.indices)]:
            indices.append(RangeIndex.build(0, size, 1))

        indices += self.indices

        return IndexedRegion(shape, tuple(indices))

    def _intersect(self, other: 'Region') -> 'Region':
        if isinstance(other, IndexedRegion):
            indices = tuple(ind.intersect(other_ind) for ind, other_ind in zip(self.indices, other.indices))
            if any(ind.empty for ind in indices):
                return EmptyRegion(self.base)
            return IndexedRegion(self.base, indices)
        return super()._intersect(other)

    @property
    def contiguous(self) -> bool:
        return all(index.contiguous for index in self.indices)

    @property
    def bounds(self) -> 'Region':
        if self.contiguous:
            return self
        indices = tuple(ind.bounds for ind in self.indices)
        return IndexedRegion(self.base, indices)

    def bound(self, axis: int) -> tuple[int, int]:
        index_bounds = self.indices[axis].bounds
        return index_bounds.start, index_bounds.stop

    def _repr_arg(self) -> str:
        return f'{self.base}, [' + ', '.join(ind.display(size) for ind, size in zip(self.indices, self.base)) + ']'


class FullRegion(Region):

    __slots__ = ()

    full: bool = True
    empty: bool = False
    contiguous: bool = True

    @property
    def bounds(self) -> 'Region':
        return self

    def bound(self, axis: int) -> tuple[int, int]:
        return 0, self.base[axis]

    @property
    def indices(self) -> tuple[RegionIndex, ...]:
        return tuple(RangeIndex.build(0, size, 1) for size in self.base)

    def _select(self, axis: int, index: int) -> 'Region':
        if self.base[axis] == 1:
            return self
        indices = tuple(IntIndex(index) if a == axis else ax for a, ax in enumerate(self.indices))
        return IndexedRegion(self.base, indices)

    def broadcast(self, shape: Shape) -> Region:
        return self if shape == self.base else FullRegion(shape)


class EmptyRegion(Region):

    __slots__ = ()

    full: bool = False
    empty: bool = True
    contiguous: bool = True

    @property
    def bounds(self) -> 'Region':
        return self

    def bound(self, axis: int) -> tuple[int, int]:
        return 0, 0

    def _select(self, axis: int, index: int) -> 'Region':
        return self

    def broadcast(self, shape: Shape) -> Region:
        return self if shape == self.base else EmptyRegion(shape)


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
            return cls(source, excluded)
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
        return cls(regions)


class IntersectionRegion(CompositeRegion):

    __slots__ = ()

    def _intersect(self, other: Region) -> Region:
        return IntersectionRegion(self.regions + (other,))

    def _union(self, other: Region) -> Region:
        return UnionRegion.build(self, other)


class UnionRegion(CompositeRegion):

    __slots__ = ()

    def _intersect(self, other: Region) -> Region:
        return IntersectionRegion.build(self, other)

    def _union(self, other: Region) -> Region:
        return UnionRegion(self.regions + (other,))
