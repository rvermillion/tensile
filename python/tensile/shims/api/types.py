#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from typing import Iterator, Protocol, Sequence, TypeAlias, Union
from ..common.types import *

AxisSelector: TypeAlias = Union[int, slice, 'Array', Ellipsis, None]
Selector: TypeAlias = MaybeTuple[AxisSelector]
ArrayLike: TypeAlias = Union['Array', Scalar, Sequence['ArrayLike']]
ArrayOrScalar: TypeAlias = Union['Array', S]
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
    def size(self) -> int: ...

    @property
    def shape(self) -> Shape: ...

    @property
    def dtype(self) -> DType: ...

    def reshape(self, *shape: int) -> 'Array': ...

    def astype(self, dtype: DType) -> 'Array': ...

    def item(self) -> Scalar: ...

    def __getitem__(self, key: Selector) -> 'Array': ...

    def __setitem__(self, key: Selector, value: ArrayLike) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[ArrayLike]: ...

    def __eq__(self, other) -> 'Array': ...

    def __ne__(self, other) -> 'Array': ...

    def __gt__(self, other) -> 'Array': ...

    def __lt__(self, other) -> 'Array': ...

    def __ge__(self, other) -> 'Array': ...

    def __le__(self, other) -> 'Array': ...

    def __neg__(self) -> 'Array': ...

    def __pos__(self) -> 'Array': ...

    def __abs__(self) -> 'Array': ...

    def __floor__(self) -> 'Array': ...

    def __invert__(self) -> 'Array': ...

    def __add__(self, other) -> 'Array': ...

    def __radd__(self, other) -> 'Array': ...

    def __sub__(self, other) -> 'Array': ...

    def __rsub__(self, other) -> 'Array': ...

    def __mul__(self, other) -> 'Array': ...

    def __rmul__(self, other) -> 'Array': ...

    def __matmul__(self, other) -> 'Array': ...

    def __rmatmul__(self, other) -> 'Array': ...

    def __truediv__(self, other) -> 'Array': ...

    def __rtruediv__(self, other) -> 'Array': ...

    def __rdiv__(self, other) -> 'Array': ...

    def __divmod__(self, other) -> 'Array': ...

    def __rdivmod__(self, other) -> 'Array': ...

    def __floordiv__(self, other) -> 'Array': ...

    def __rfloordiv__(self, other) -> 'Array': ...

    def __pow__(self, other) -> 'Array': ...

    def __rpow__(self, other) -> 'Array': ...

    def __mod__(self, other) -> 'Array': ...

    def __rmod__(self, other) -> 'Array': ...

    def __and__(self, other) -> 'Array': ...

    def __or__(self, other) -> 'Array': ...

    def __xor__(self, other) -> 'Array': ...

    def __rand__(self, other) -> 'Array': ...

    def __ror__(self, other) -> 'Array': ...

    def __rxor__(self, other) -> 'Array': ...


float64: DType
float32: DType
float16: DType
int64: DType
int32: DType
int16: DType
int8: DType
uint64: DType
uint32: DType
uint16: DType
uint8: DType
bool_: DType


__all__ = [
    'Array',
    'ArrayLike',
    'ArrayOrScalar',
    'ArrayOrT',
    'ArrayOrFloat',
    'Axis',
    'Axes',
    'AxisSelector',
    'DType',
    'DTypeLike',
    'Scalar',
    'Selector',
    'Shape',
    'ShapeLike',
    'float64',
    'float32',
    'float16',
    'int64',
    'int32',
    'int16',
    'int8',
    'uint64',
    'uint32',
    'uint16',
    'uint8',
    'bool_',
]
