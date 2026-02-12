#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED
from typing import Generic, Iterable, TypeVar
from .. import ten

from .common import Array, Base, DType, Index, Indices, Shape, TensorType


full = slice(None, None, None)


def repr_index(index: Index) -> str:
    if isinstance(index, slice):
        start, stop, step = index.indices(10)
    return str(index) if index is not None else '...'


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
    # indices: tuple[Index, ...]

    def __init__(self, base: Shape):
        self.base = base

    @property
    def contiguous(self) -> bool:
        raise NotImplementedError()

    def overlaps(self, other: 'Region') -> bool:
        raise NotImplementedError

    def contains(self, other: 'Region') -> bool:
        raise NotImplementedError

    def equals(self, other: 'Region') -> bool:
        raise NotImplementedError

    def intersect(self, other: 'Region') -> 'Region':
        return IntersectionRegion.build(self, other)

    def union(self, other: 'Region') -> 'Region':
        return UnionRegion.build(self, other)

    def minus(self, other: 'Region') -> 'Region':
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

    @classmethod
    def from_key(cls, shape: Shape, key: Indices) -> 'Region':
        indices = RegionIndex.from_keys(key, shape)
        # indices: tuple[RegionIndex, ...]
        # if key is ...:
        #     indices = tuple(RangeIndex.build(0, size, 1) for size in shape)
        # elif isinstance(key, int):
        #     if key < 0 and shape:
        #         key = shape[0] + key
        #     indices = (IntIndex(key),)
        # elif isinstance(key, slice):
        #     indices = (key,)
        # elif isinstance(key, tuple):
        #     indices = key
        # else:
        #     raise ValueError(f'Invalid key: {key} for shape {shape}')
        # if len(indices) < len(shape):
        #     indices += tuple(slice(0, size, 1) for size in shape[len(indices):])
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

    @property
    def count(self) -> int:
        raise NotImplementedError()

    def array(self) -> Array:
        raise NotImplementedError()

    def has(self, index: int) -> bool:
        raise NotImplementedError()

    def could_overlap(self, other: 'RegionIndex') -> bool:
        if self.start < other.start:
            return other.start < self.stop
        return self.start < other.stop

    def overlaps(self, other: 'RegionIndex') -> bool:
        return self.could_overlap(other) and self.contiguous and other.contiguous

    # def intersect(self, other: 'RegionIndex') -> 'RegionIndex':
    #     raise NotImplementedError()
    #
    # def union(self, other: 'RegionIndex') -> 'RegionIndex':
    #     raise NotImplementedError()
    #
    # def minus(self, other: 'RegionIndex') -> 'RegionIndex':
    #     raise NotImplementedError()

    def __len__(self) -> int:
        return self.count

    # def __and__(self, other: 'RegionIndex') -> 'RegionIndex':
    #     return self.intersect(other)
    #
    # def __or__(self, other: 'RegionIndex') -> 'RegionIndex':
    #     return self.union(other)
    #
    # def __sub__(self, other: 'RegionIndex') -> 'RegionIndex':
    #     return self.minus(other)

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

    def _repr_args(self) -> Iterable:
        return self.index,


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

    def array(self) -> Array:
        return self.index

    def has(self, index: int) -> bool:
        return ten.any(self.index == index).item()

    def intersect(self, other: RegionIndex) -> RegionIndex:
        raise NotImplementedError()

    def _repr_args(self) -> Iterable:
        return self.index,

    @classmethod
    def build(cls, index: Array) -> RegionIndex:
        if index.dtype != ten.int64:
            raise ValueError(f'Index must be int64, got {index.dtype}')
        if index.size == 0:
            return EmptyIndex.singleton
        if index.ndim != 1:
            index = ten.reshape(index, (-1,))
        index = ten.sort(index)
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
            return StepRangeIndex(start, stop, step)

        return EmptyIndex.singleton
        # raise ValueError(f'Invalid range: {start} < {stop}')


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

    def has(self, index: int) -> bool:
        return self.start <= index < self.stop

    def overlaps(self, other: RegionIndex) -> bool:
        if isinstance(other, ContiguousRangeIndex):
            if self.start < other.start:
                return other.start < self.stop
            return self.start < other.stop
        raise NotImplementedError()

    def union(self, other: RegionIndex) -> RegionIndex:
        if isinstance(other, ContiguousRangeIndex):
            start = min(self.start, other.start)
            stop = max(self.stop, other.stop)
            return RangeIndex.build(min(self.start, other.start), max(self.stop, other.stop))
        raise NotImplementedError()

    def _repr_args(self) -> Iterable:
        return f'{self.start}:{self.stop}',


class StepRangeIndex(RangeIndex):

    __slots__ = ('step', )

    contiguous: bool = False

    def __init__(self, start: int, stop: int, step: int):
        if step == 1:
            raise ValueError('Step cannot be 1')
        super().__init__(start, stop)
        self.step = step

    def has(self, index: int) -> bool:
        if self.start <= index < self.stop:
            return (index - self.start) % self.step == 0
        return False

    def _repr_args(self) -> Iterable:
        return f'{self.start}:{self.stop}:{self.step}',


class IndexedRegion(Region):

    __slots__ = ('indices',)

    indices: tuple[RegionIndex, ...]

    def __init__(self, base: Shape, indices: tuple[RegionIndex, ...]):
        super().__init__(base)
        self.indices = indices

    def broadcast(self, shape: Shape) -> Region:
        if shape == self.base:
            return self
        indices = []
        for size in shape[:-len(self.indices)]:
            indices.append(RangeIndex.build(0, size, 1))

        indices += self.indices

        return IndexedRegion(shape, tuple(indices))

    @property
    def contiguous(self) -> bool:
        return all(index.contiguous for index in self.indices)

    def _repr_args(self) -> Iterable:
        return self.base, *self.indices


class EmptyRegion(Region):

    __slots__ = ()

    def __init__(self, base: Shape):
        super().__init__(base)

    def broadcast(self, shape: Shape) -> Region:
        return self


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

    def intersect(self, other: Region) -> Region:
        return IntersectionRegion(self.regions + (other,))

    def union(self, other: Region) -> Region:
        return UnionRegion.build(self, other)


class UnionRegion(CompositeRegion):

    __slots__ = ()

    def intersect(self, other: Region) -> Region:
        return IntersectionRegion.build(self, other)

    def union(self, other: Region) -> Region:
        return UnionRegion(self.regions + (other,))
