#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, Sequence, TypeAlias, TypeGuard, TypeVar, Union

import numpy as np
import numpy.random as npr

from .types import *


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
    abs, square, sqrt, exp, log, expm1, logaddexp as logsumexp,
    sum, prod, max, min, mean, median, std, var, quantile, percentile,
    minimum, maximum, clip, average, argmax, argmin,
    cumsum, cumprod,
    where,
    argsort, argpartition,
    any, all,
    floor, floor_divide,
    sort, searchsorted,
    expand_dims, squeeze, transpose,
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


from . import functional, random


__all__ = [
    'Array',
    'DType',
    'functional',
    'random'
]