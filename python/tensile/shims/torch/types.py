#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import Sequence, TypeAlias, Union

import torch
import numpy as np

from ..common.types import *

Array = torch.Tensor
DType = torch.dtype

DTypeLike = Union[DType, str, type]
AxisSelector = Union[int, slice, 'Array', Ellipsis, None]
Selector = MaybeTuple[AxisSelector]
ArrayLike = Union['Array', Scalar, Sequence['ArrayLike'], np.ndarray]
ArrayOrScalar = Union['Array', S]
ArrayOrT = Union['Array', T, Sequence[T]]
ArrayOrFloat = ArrayOrT[float]

float64: DType = torch.float32
float32: DType = torch.float32
float16: DType = torch.float16
int64: DType = torch.int64
int32: DType = torch.int32
int16: DType = torch.int16
int8: DType = torch.int8
uint64: DType = torch.uint64
uint32: DType = torch.uint32
uint16: DType = torch.uint16
uint8: DType = torch.uint8
bool_: DType = torch.bool

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
