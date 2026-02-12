#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, Sequence, TypeAlias, TypeGuard, TypeVar, Union

import numpy as np
import numpy.random as npr

# from ..array import ArrayOrScalar, Shim, DType, RNG, Random
from .common import Scalar, Shape, ShapeLike, Axis, Axes

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

class NumpyRandom():

    Generator: type[RNG] = RNG

    default_rng = staticmethod(npr.default_rng)

    normal = staticmethod(npr.normal)
    uniform = staticmethod(npr.uniform)


def is_monotonic_test(vals: Array, op: np.ufunc) -> bool:
    return np.all(op(vals[:-1], vals[1:]))


def is_array(obj: Any) -> TypeGuard[Array]:
    return isinstance(obj, Array)

def is_dtype(obj: Any) -> TypeGuard[DType]:
    return isinstance(obj, DType)

def is_rng(obj: Any) -> TypeGuard[RNG]:
    return isinstance(obj, npr.Generator)

def astype(a: Any, dtype: DTypeLike) -> Array:
    return np.asarray(a, dtype=dtype)

from numpy import (
    array,
    zeros,
    zeros_like,
    ones,
    ones_like,
    full,
    full_like,
    empty,
    empty_like, fromfunction, arange, concatenate, reshape,
    abs, square, sqrt, exp, log,
    sum, max, min, mean, median, std, var, quantile, percentile,
    minimum, maximum, clip, average, argmax, argmin,
    cumsum, cumprod,
    floor, floor_divide,
    sort, searchsorted,
    expand_dims,
    broadcast_to,
)

ten_kind: str = 'numpy'


def update(array: Array, where: Array, value: ArrayOrScalar) -> None:
    array[where] = value

def is_increasing(vals: Array, strict: bool = False) -> bool:
    return is_monotonic_test(vals, np.less if strict else np.less_equal)

def is_decreasing(vals: Array, strict: bool = False) -> bool:
    return is_monotonic_test(vals, np.greater if strict else np.greater_equal)

def is_monotonic(vals: Array, strict: bool = False) -> bool:
    return (is_monotonic_test(vals, np.less if strict else np.less_equal) or
            is_monotonic_test(vals, np.greater if strict else np.greater_equal))


random: type[NumpyRandom] = NumpyRandom

float64: DType = np.float64
float32: DType = np.float32
float16: DType = np.float16
int64: DType = np.int64
int32: DType = np.int32
int16: DType = np.int16
int8: DType = np.int8
bool_: DType = np.bool_


__all__ = [
    'Array',
    'DType',
    'random'
]