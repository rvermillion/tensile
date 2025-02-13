#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, Iterator, Protocol, Sequence, TypeAlias, TypeGuard, TypeVar, Union

T = TypeVar('T')

Scalar: TypeAlias = Union[int, float, bool]

S = TypeVar('S', bound=Scalar)

AxisSelector: TypeAlias = Union[int, slice, 'Array', Ellipsis, None]
Selector: TypeAlias = Union[AxisSelector, tuple[AxisSelector, ...]]
Shape: TypeAlias = tuple[int, ...]
ShapeLike: TypeAlias = Union[int, Shape]
Axis: TypeAlias = int
Axes: TypeAlias = Union[None, int, tuple[int, ...]]
ArrayLike: TypeAlias = Union[S, 'Array', Sequence['ArrayLike[S]']]
ArrayOrScalar: TypeAlias = Union['Array', Scalar]
ArrayOrT: TypeAlias = Union['Array', T, Sequence[T]]
ArrayOrFloat: TypeAlias = ArrayOrT[float]


class DType(Protocol):

    @property
    def name(self) -> str: ...


DTypeLike: TypeAlias = Union[DType, str, type]


class Array(Protocol):

    @property
    def ndim(self) -> int: ...

    @property
    def shape(self) -> Shape: ...

    @property
    def dtype(self) -> DType: ...

    def reshape(self, *shape: int) -> 'Array': ...

    def item(self) -> Scalar: ...

    def __getitem__(self, key: Selector) -> 'Array': ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[ArrayLike]: ...

    def __neg__(self) -> 'Array': ...

    def __pos__(self) -> 'Array': ...

    def __add__(self, other) -> 'Array': ...

    def __radd__(self, other) -> 'Array': ...

    def __sub__(self, other) -> 'Array': ...

    def __rsub__(self, other) -> 'Array': ...

    def __mul__(self, other) -> 'Array': ...

    def __rmul__(self, other) -> 'Array': ...

    def __truediv__(self, other) -> 'Array': ...

    def __rtruediv__(self, other) -> 'Array': ...

    def __rdiv__(self, other) -> 'Array': ...

    def __pow__(self, other) -> 'Array': ...

    def __rpow__(self, other) -> 'Array': ...


class RNG(Protocol):

    def normal(self, loc: ArrayLike[float] = ..., scale: ArrayLike[float] = ..., size: ShapeLike = ...) -> ArrayOrScalar: ...

    def uniform(self, low: ArrayLike[float] = ..., high: ArrayLike[float] = ..., size: ShapeLike = ...) -> ArrayOrScalar: ...

    def exponential(self, rate: ArrayLike[float] = ..., size: ShapeLike = ...) -> ArrayOrScalar: ...


class Random(Protocol):

    Generator: type[RNG]

    def default_rng(self, seed: int = ...) -> RNG: ...

    def normal(self, loc: ArrayLike[float] = ..., scale: ArrayLike[float] = ..., size: ShapeLike = ...) -> ArrayOrScalar: ...

    def uniform(self, low: ArrayLike[float] = ..., high: ArrayLike[float] = ..., size: ShapeLike = ...) -> ArrayOrScalar: ...


class Backend(Protocol):

    float64: DType
    float32: DType
    float16: DType
    int64: DType
    int32: DType
    int16: DType
    int8: DType

    def is_array(self, obj: Any) -> TypeGuard[Array]: ...

    def is_dtype(self, obj: Any) -> TypeGuard[DType]: ...

    def is_rng(self, obj: Any) -> TypeGuard[RNG]: ...

    def astype(self, a: Any, dtype: DType) -> Array: ...

    def dtype(self, dtype: DTypeLike) -> DType: ...

    def array(self, data, dtype: DTypeLike = ..., *args, **kwargs) -> Array: ...

    def zeros(self, shape: Shape, dtype: DType = ..., *args, **kwargs) -> Array: ...

    def ones(self, shape: Shape, dtype: DType = ..., *args, **kwargs) -> Array: ...

    def full(self, shape: Shape, fill_value: Scalar, dtype: DType = ..., *args, **kwargs) -> Array: ...

    def fromfunction(self, function, shape, *, dtype=float, like=None, **kwargs) -> Array: ...

    def arange(self, start: Scalar, stop: Scalar = ..., step: Scalar = ..., dtype: DType = ...) -> Array: ...

    def concatenate(self, arrays: list[Array], axis: int) -> Array: ...

    def floor(self, a: ArrayLike, /, dtype: DType = ..., **kwargs) -> Array: ...

    def square(self, a: ArrayLike, /, dtype: DType = ..., **kwargs) -> Array: ...

    def exp(self, a: ArrayLike, /, dtype: DType = ..., **kwargs) -> Array: ...

    def log(self, a: ArrayLike, /, dtype: DType = ..., **kwargs) -> Array: ...

    def min(self, a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array: ...

    def max(self, a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array: ...

    def sum(self, a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array: ...

    def mean(self, a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array: ...

    def std(self, a: ArrayLike, axis: Axes = ..., dtype: DType = ..., ddof: int = ..., keepdims: bool = ..., **kwargs) -> Array: ...

    def var(self, a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array: ...

    def median(self, a: ArrayLike, axis: Axes = ..., dtype: DType = ..., ddof: int = ..., keepdims: bool = ..., **kwargs) -> Array: ...

    def minimum(self, a: ArrayLike, b: ArrayLike, **kwargs) -> Array: ...

    def maximum(self, a: ArrayLike, b: ArrayLike, **kwargs) -> Array: ...

    def average(self, a: ArrayLike, axis: Axes = ..., weights: ArrayLike = ..., **kwargs) -> Array: ...

    def sort(self, a: ArrayLike, axis: Axis = ..., **kwargs) -> Array: ...

    def searchsorted(self, a: ArrayLike, x: Scalar, side: str = ...) -> Array: ...

    random: Random
    bool: DType


class Select:

    __slots__ = ()

    def __getitem__(self, item: Selector) -> Selector:
        return item


full: AxisSelector = slice(None)
select = Select()



__all__ = [
    'Array',
    'ArrayLike',
    'Axes',
    'Axis',
    'AxisSelector',
    'Backend',
    'DType',
    'DTypeLike',
    'RNG',
    'Scalar',
    'Selector',
    'Shape',
    'ShapeLike',
    'full',
    'select',
]