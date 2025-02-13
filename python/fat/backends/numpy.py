#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, TypeAlias, TypeGuard

import numpy as np
import numpy.random as npr

from ..array import Backend, DType, RNG, Random


NumpyArray: TypeAlias = np.ndarray


class NumpyRandom:

    Generator: type[RNG] = npr.Generator

    default_rng = staticmethod(npr.default_rng)

    normal = staticmethod(npr.normal)


class NumpyBackend:

    @staticmethod
    def is_array(obj: Any) -> TypeGuard[NumpyArray]:
        return isinstance(obj, NumpyArray)

    @staticmethod
    def is_dtype(obj: Any) -> TypeGuard[np.dtype]:
        return isinstance(obj, np.dtype)

    @staticmethod
    def is_rng(obj: Any) -> TypeGuard[npr.Generator]:
        return isinstance(obj, npr.Generator)

    @staticmethod
    def astype(a: Any, dtype: np.dtype) -> np.array:
        return np.asarray(a, dtype=dtype)

    array = staticmethod(np.array)

    zeros = staticmethod(np.zeros)
    zeros_like = staticmethod(np.zeros_like)

    ones = staticmethod(np.ones)
    ones_like = staticmethod(np.ones_like)

    full = staticmethod(np.full)
    full_like = staticmethod(np.full_like)

    empty = staticmethod(np.empty)
    empty_like = staticmethod(np.empty_like)

    fromfunction = staticmethod(np.fromfunction)

    arange = staticmethod(np.arange)
    concatenate = staticmethod(np.concatenate)

    reshape = staticmethod(np.reshape)

    square = staticmethod(np.square)
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)

    sum = staticmethod(np.sum)
    max = staticmethod(np.max)
    min = staticmethod(np.min)
    mean = staticmethod(np.mean)
    median = staticmethod(np.median)
    std = staticmethod(np.std)
    var = staticmethod(np.var)
    quantile = staticmethod(np.quantile)
    percentile = staticmethod(np.percentile)

    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    average = staticmethod(np.average)

    argmax = staticmethod(np.argmax)

    sort = staticmethod(np.sort)
    searchsorted = staticmethod(np.searchsorted)

    floor = staticmethod(np.floor)

    random: Random = NumpyRandom

    float64: DType = np.float64
    float32: DType = np.float32
    float16: DType = np.float16
    int64: DType = np.int64
    int32: DType = np.int32
    int16: DType = np.int16
    int8: DType = np.int8
    bool: DType = np.bool_




backend: Backend = NumpyBackend
