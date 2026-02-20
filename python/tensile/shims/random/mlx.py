#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

import numpy as np
import mlx.core as mx
import mlx.core.random as mxr

from mlx.core.random import *
# from mlx.core.random import (
#     normal as mx_normal,
#     uniform as mx_uniform,
# )

from ..mlx import Shape, ShapeLike, ArrayLike, Array, DType, ArrayOrScalar


def to_shape(size: ShapeLike) -> Shape:
    return (size, ) if isinstance(size, int) else size


class RNG:

    __slots__ = 'key',

    def __init__(self, key = None):
        self.key = key

    def normal(self, loc: ArrayLike = ..., scale: ArrayLike = ..., size: ShapeLike = None, dtype: DType = None) -> Array:
        return mxr.normal(shape=to_shape(size), loc=loc, scale=scale, dtype=dtype, key=self.key)

    def uniform(self, low: ArrayLike = ..., high: ArrayLike = ..., size: ShapeLike = None, dtype: DType = None) -> Array:
        return mxr.uniform(low, high, shape=to_shape(size), key=self.key)

    def exponential(self, rate: ArrayLike = ..., size: ShapeLike = ...) -> ArrayOrScalar:
        raise NotImplementedError()


Generator: type[RNG] = None # mxr.Generator

def default_rng(seed: int = None) -> RNG:
    return RNG(seed)

def normal(loc: ArrayLike = 0., scale: ArrayLike = 1., shape: ShapeLike = ...) -> Array:
    return mxr.normal(shape=to_shape(shape), loc=loc, scale=scale)

def uniform(low: ArrayLike = 0., high: ArrayLike = 1., shape: ShapeLike = ...) -> Array:
    return mxr.uniform(low, high, shape=to_shape(shape))

def permutation(size: int, **kwargs) -> Array:
    return mx.array(np.random.permutation(size))
