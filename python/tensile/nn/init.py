#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
import math

from ..infra import meta
from .common import Spec, Array, Optional, Protocol, ten


class Initializer(Protocol):

    def __call__(self, shape: ten.Shape, scale: Optional[float] = ...) -> Array: ...


def make_uniform(*, low: float = None, high: float = None, default_scale: float = None) -> Initializer:
    if low is None and high is None:
        def uniform(shape: ten.Shape, /, scale: Optional[float] = default_scale) -> Array:
            if scale is None:
                scale = 1.0 / math.sqrt(shape[-1])
            return ten.random.uniform(low=-scale, high=scale, shape=shape)
    else:
        def uniform(shape: ten.Shape, /, scale: Optional[float] = default_scale) -> Array:
            if scale is None:
                return ten.random.uniform(low=low, high=high, shape=shape)
            return ten.random.uniform(low=-scale, high=scale, shape=shape)

    return uniform


def make_normal(*, loc: float = None, default_scale: float = None) -> Initializer:

    def normal(shape: ten.Shape, /, scale: Optional[float] = default_scale) -> Array:
        if scale is None:
            scale = 1.0 / math.sqrt(shape[-1])
        return ten.random.normal(loc=loc, scale=scale, shape=shape)

    return normal


init_uniform: Initializer = make_uniform()

init_normal: Initializer = make_normal()

init_default: Initializer = init_uniform


class Initializers:

    uniform: Initializer = init_uniform

    normal: Initializer = init_normal

    default: Initializer = uniform




@meta.provides(Initializer, 'uniform')
def uniform_provider(spec: Spec, *, low: float = None, high: float = None, scale: float = None, **kwargs) -> Initializer:
    return make_uniform(low=low, high=high, default_scale=scale)


@meta.provides(Initializer, 'normal')
def normal_provider(spec: Spec, *, loc: float = None, scale: float = None, **kwargs) -> Initializer:
    return make_normal(loc=loc, default_scale=scale)
