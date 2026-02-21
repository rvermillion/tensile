#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import numpy.random as npr

from .types import *


def normal(loc: ArrayLike = 0.0, scale: ArrayLike = 1.0, shape: ShapeLike = None) -> Array:
    return npr.normal(loc=loc, scale=scale, size=shape)

def uniform(low: ArrayLike = 0.0, high: ArrayLike = 1.0, shape: ShapeLike = None) -> Array:
    return npr.uniform(low=low, high=high, size=shape)

def randint(low: ArrayLike = None, high: ArrayLike = None, shape: ShapeLike = None, dtype: DType | None = None) -> Array:
    return npr.randint(low=low, high=high, size=shape, dtype=dtype)

def permutation(size: int, **kwargs) -> Array:
    return npr.permutation(size)

def seed(seed: int) -> None:
    return npr.seed(seed)
