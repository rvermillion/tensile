#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Sequence, TypeAlias, Union

import mlx.core as mx
import numpy as np

from ..common.types import *

Array: TypeAlias = mx.array
DType: TypeAlias = mx.Dtype

DTypeLike: TypeAlias = Union[DType, str, type]
MaybeTuple: TypeAlias = Union[T, tuple[T, ...]]
AxisSelector: TypeAlias = Union[int, slice, 'Array', Ellipsis, None]
Selector: TypeAlias = MaybeTuple[AxisSelector]
ArrayLike: TypeAlias = Union['Array', Scalar, Sequence['ArrayLike'], np.ndarray]
ArrayOrScalar: TypeAlias = Union['Array', S]
ArrayOrT: TypeAlias = Union['Array', T, Sequence[T]]
ArrayOrFloat: TypeAlias = ArrayOrT[float]

float64: DType = mx.float32
float32: DType = mx.float32
float16: DType = mx.float16
int64: DType = mx.int64
int32: DType = mx.int32
int16: DType = mx.int16
int8: DType = mx.int8
uint64: DType = mx.uint64
uint32: DType = mx.uint32
uint16: DType = mx.uint16
uint8: DType = mx.uint8
bool_: DType = mx.uint8


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
