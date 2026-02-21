#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from typing import Sequence, TypeAlias, TypeVar, Union

import numpy as np
import numpy.random as npr
from ..common.core import *
T = TypeVar('T')

S = TypeVar('S', bound=Scalar)

Array = np.ndarray
DType = np.dtype
RNG = npr.Generator

DTypeLike: TypeAlias = Union[DType, str, type]
MaybeTuple: TypeAlias = Union[T, tuple[T, ...]]
AxisSelector: TypeAlias = Union[int, slice, 'Array', Ellipsis, None]
Selector: TypeAlias = MaybeTuple[AxisSelector]
ArrayLike: TypeAlias = Union['Array', Scalar, Sequence['ArrayLike']]
ArrayOrScalar: TypeAlias = Union['Array', S]
ArrayOrT: TypeAlias = Union['Array', T, Sequence[T]]
ArrayOrFloat: TypeAlias = ArrayOrT[float]

Stream = None

float64: DType = np.float64
float32: DType = np.float32
float16: DType = np.float16
int64: DType = np.int64
int32: DType = np.int32
int16: DType = np.int16
int8: DType = np.int8
uint64: DType = np.uint64
uint32: DType = np.uint32
uint16: DType = np.uint16
uint8: DType = np.uint8
bool_: DType = np.bool_

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
    'RNG',
    'Scalar',
    'Selector',
    'Shape',
    'ShapeLike',
    'Stream',
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
