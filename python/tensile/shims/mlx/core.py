#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from typing import Any, Literal, TypeGuard

import mlx.core as mx
import mlx.core.random as mxr
import numpy as np

from .types import *
from ..common.core import *


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


# class MLXRandom:
#
#     Generator: type[RNG] = None # mxr.Generator
#
#     @staticmethod
#     def default_rng(seed: int = None) -> RNG:
#         return RNG(seed)
#
#     @staticmethod
#     def normal(loc: ArrayLike = 0.0, scale: ArrayLike = 1.0, shape: ShapeLike = ...) -> Array:
#         return mxr.normal(shape=to_shape(shape), loc=loc, scale=scale)
#
#     @staticmethod
#     def uniform(low: ArrayLike = ..., high: ArrayLike = ..., shape: ShapeLike = ...) -> Array:
#         return mxr.uniform(low, high, shape=to_shape(shape))
#
#     randint = staticmethod(mxr.randint)
#
#     seed = staticmethod(mxr.seed)
#
#     @staticmethod
#     def permutation(size: int, **kwargs) -> Array:
#         return mx.array(np.random.permutation(size))


def is_monotonic_test(vals: Array, op: Any) -> bool:
    return bool(mx.all(op(vals[:-1], vals[1:])).item())


full_slice = slice(None)
full_slices = tuple(full_slice for _ in range(10))


def is_array(obj: Any) -> TypeGuard[Array]:
    return isinstance(obj, Array)

def is_dtype(obj: Any) -> TypeGuard[DType]:
    return isinstance(obj, DType)

def is_rng(obj: Any) -> TypeGuard[RNG]:
    return False

def astype(a: Any, dtype: DType) -> Array:
    if isinstance(a, Array):
        return a.astype(dtype)
    return mx.array(a, dtype=dtype)

from mlx.core import (
    array,
    zeros, zeros_like,
    ones, ones_like,
    full,
    eye, trace,
    arange, concatenate, reshape,
    abs, square, sqrt, exp, log, expm1,
    sum, max, min, mean, std, var, logsumexp, prod,
    cumsum, cumprod, cummax, cummin,
    clip,
    addmm,
    isinf, isnan,
    matmul,
    minimum, maximum, clip,
    argmin, argmax,
    floor, floor_divide,
    sort, where,
    eval, async_eval,
    expand_dims, squeeze,
    save_safetensors,
    swapaxes, transpose,
    broadcast_to,
    inf,
    all, any,
    allclose,
    argsort,
    stack, array_equal,
    equal,
    stream, new_stream, default_stream,
    Stream,
    load, save,
    get_peak_memory,
    set_wired_limit,
    device_info, default_device,
    synchronize,
    metal,
    quantize, quantized_matmul,
)

clear_cache = metal.clear_cache


def softmax(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    if dtype is not None and dtype != a.dtype:
        a = a.astype(dtype)
    out = mx.softmax(a, axis=axis)
    if keepdims:
        if axis is None:
            shape = [1] * a.ndim
        else:
            shape = list(a.shape)
            if isinstance(axis, int):
                shape[axis] = 1
            elif isinstance(axis, tuple):
                for ax in axis:
                    shape[ax] = 1
        out = out.reshape(*shape)
    return out


from . import fast, linalg, random, functional


ten_kind: str = 'mlx'

debug: bool = True


def debug_eval(*args: Any) -> None:
    if debug:
        eval(*args)


def size(a: Array) -> int:
    return a.size


def full_like(a: Array, fill_value: Scalar, dtype: DType = None, *args, **kwargs) -> Array:
    if dtype is None:
        dtype = a.dtype
    return mx.full(a.shape, fill_value, dtype=dtype)

empty = mx.zeros
empty_like = mx.zeros_like

def fromfunction(function, shape, *, dtype=float, like=None, **kwargs) -> Array:
    raise NotImplementedError()


def broadcast_shapes(a: ArrayLike, b: ArrayLike) -> Shape:
    return mx.broadcast_to(mx.array(a), b.shape).shape

def median(a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array:
    raise NotImplementedError()

def quantile(a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array:
    raise NotImplementedError()

def percentile(a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> Array:
    raise NotImplementedError()

def average(a: ArrayLike, axis: Axes = ..., weights: ArrayLike = ..., **kwargs) -> Array:
    raise NotImplementedError()


def searchsorted(a: Array, x: ArrayOrScalar, side: Literal["left", "right"] = ...) -> Array:
    # TODO: Figure out if there's a better way than roundtripping to numpy
    return mx.array(np.searchsorted(np.array(a), np.array(x), side=side))

def update(array: Array, where: Array, value: ArrayOrScalar) -> None:
    s = full_slices[:array.ndim]
    # s = tuple(slice(None) for _ in range(array.ndim))
    array[s] = mx.where(where, value, array)


def is_increasing(vals: Array, strict: bool = False) -> bool:
    return is_monotonic_test(vals, mx.less if strict else mx.less_equal)

def is_decreasing(vals: Array, strict: bool = False) -> bool:
    return is_monotonic_test(vals, mx.greater if strict else mx.greater_equal)

def is_monotonic(vals: Array, strict: bool = False) -> bool:
    return (is_monotonic_test(vals, mx.less if strict else mx.less_equal) or
            is_monotonic_test(vals, mx.greater if strict else mx.greater_equal))

def is_integer(dtype: DType) -> bool:
    return mx.issubdtype(dtype, mx.integer)


def is_floating(dtype: DType) -> bool:
    return mx.issubdtype(dtype, mx.floating)


# noinspection PyShadowingNames
def select(a: Array, where: Array) -> Array:
    if where is None:
        raise ValueError('where must be specified')
    if where.dtype != mx.bool_:
        raise ValueError('where must be a boolean array')
    a_flat = a.reshape(-1)
    w = mx.broadcast_to(where, a.shape).reshape(-1)
    idx = mx.cumsum(w.astype(mx.int32))
    count = idx[-1].item()
    print(f'selecting {count} out of {where.size}')
    idx = mx.where(where, idx-1, -1)
    print(f'idx: {idx}')
    selected = mx.zeros(count+1, dtype=a.dtype)
    selected[idx] = a_flat
    return selected[:count]


# noinspection PyShadowingNames
def pack_front_sort(a: Array, where: Array, *, fill_value=0, index_dtype: DType = None) -> tuple[Array, Array]:
    if index_dtype is None:
        index_dtype = mx.int32
    elif not mx.issubdtype(index_dtype, mx.integer):
        raise ValueError(f'index_dtype must be an integer type, got {index_dtype}')
    a_flat = a.reshape(-1)
    w = mx.broadcast_to(where, a.shape).reshape(-1).astype(index_dtype)
    n = a_flat.size
    idx = mx.arange(n, 2*n, dtype=index_dtype)
    # Build a single sortable key: key = (w_false * big) + idx0
    # where w_false=1 for false, 0 for true.
    key = idx - w * n # trues in [0..n-1], falses in [n..2n-1]
    perm = mx.argsort(key)    # should be stable given unique key
    a_sorted = a_flat[perm]
    # w_sorted = w[perm]
    # Count still on-device
    count = mx.sum(w).reshape(1)
    # Create padded output: take a_sorted but replace the tail with fill_value
    out = mx.full((n,), fill_value, dtype=a_flat.dtype)
    mask = mx.arange(n, dtype=index_dtype) < count
    out = mx.where(mask, a_sorted, out)  # broadcast count
    return out, count


