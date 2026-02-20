#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

import torch

from .types import Shape, ShapeLike, ArrayLike, Array, DType, ArrayOrScalar


def to_shape(size: ShapeLike) -> Shape:
    return (size, ) if isinstance(size, int) else size


class RNG:

    __slots__ = 'key',

    def __init__(self, key = None):
        self.key = key

    def normal(self, loc: ArrayLike = ..., scale: ArrayLike = ..., size: ShapeLike = None, dtype: DType = None) -> Array:
        return torch.normal(mean=loc, std=scale, size=to_shape(size))

    def uniform(self, low: ArrayLike = ..., high: ArrayLike = ..., size: ShapeLike = None, dtype: DType = None) -> Array:
        return torch.uniform(low, high, size=to_shape(size))

    def exponential(self, rate: ArrayLike = ..., size: ShapeLike = ...) -> ArrayOrScalar:
        raise NotImplementedError()


Generator: type[RNG] = None # mxr.Generator

def default_rng(seed: int = None) -> RNG:
    return RNG(seed)

def normal(loc: ArrayLike = 0.0, scale: ArrayLike = 1.0, shape: ShapeLike = ...) -> Array:
    return torch.normal(mean=loc, std=scale, size=to_shape(shape))

def uniform(low: ArrayLike = 0.0, high: ArrayLike = 1.0, shape: ShapeLike = ...) -> Array:
    u = torch.rand(to_shape(shape))
    return low + (high - low) * u

def permutation(size: int, **kwargs) -> Array:
    return torch.randperm(size)
