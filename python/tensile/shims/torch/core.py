#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from typing import Any, Sequence, TypeAlias, TypeGuard, Union

import torch
import numpy as np

from .types import *

TorchArray: TypeAlias = torch.Tensor
TorchDType: TypeAlias = torch.Type


def to_shape(size: ShapeLike) -> Shape:
    return (size, ) if isinstance(size, int) else size


def is_tensor(obj: Any) -> TypeGuard[TorchArray]:
    return isinstance(obj, TorchArray)


def tensor(a: ArrayLike) -> TorchArray:
    return a if isinstance(a, TorchArray) else torch.tensor(a)


class TorchGenerator:

    __slots__ = 'key',

    def __init__(self, key = None):
        self.key = key

    def normal(self, loc: ArrayLike = ..., scale: ArrayLike = ..., size: ShapeLike = None, dtype: DType = None) -> Array:
        return torch.normal(mean=loc, std=scale, size=to_shape(size))

    def uniform(self, low: ArrayLike = ..., high: ArrayLike = ..., size: ShapeLike = None, dtype: DType = None) -> Array:
        raise NotImplementedError()

    def exponential(self, rate: ArrayLike = ..., size: ShapeLike = ...) -> ArrayOrScalar:
        raise NotImplementedError()


class TorchRandom:

    Generator: type[TorchGenerator] = None # mxr.Generator

    @staticmethod
    def default_rng(seed: int = None) -> TorchGenerator:
        return TorchGenerator(seed)

    @staticmethod
    def normal(loc: ArrayLike = ..., scale: ArrayLike = ..., size: ShapeLike = ...) -> Array:
        return torch.normal(mean=loc, std=scale, size=to_shape(size))

    @staticmethod
    def uniform(low: ArrayLike = ..., high: ArrayLike = ..., size: ShapeLike = ...) -> Array:
        raise NotImplementedError()


def is_monotonic_test(vals: Array, op: Any) -> bool:
    return torch.all(op(vals[:-1], vals[1:])).item()


full_slice = slice(None)
full_slices = tuple(full_slice for _ in range(10))


def is_array(obj: Any) -> TypeGuard[Array]:
    return isinstance(obj, TorchArray)

def is_dtype(obj: Any) -> TypeGuard[DType]:
    return isinstance(obj, TorchDType)

def is_rng(obj: Any) -> TypeGuard[TorchGenerator]:
    return False

def astype(a: Any, dtype: DType) -> Array:
    return torch.tensor(a, dtype=dtype)

from torch import (
    tensor as array,
    zeros, zeros_like, ones, ones_like, full, full_like, empty, empty_like,
    arange, concatenate, reshape,
    abs, square, sqrt, exp, log, expm1,
    median, std, var, quantile,
    addmm,
    isinf, isnan,
    matmul,
    minimum, clip,
    argmin, argmax,
    floor, floor_divide,
    sort, where,
    swapaxes,
    broadcast_to,
    inf,
    all, any,
    allclose,
    argsort,
    stack,
    squeeze,
    equal, searchsorted,
)

ten_kind: str = 'torch'

Stream = None

def eval(*args) -> None:
    pass


def transpose(a: Array, axes: Axes = None) -> Array:
    return a.permute(axes)

# noinspection PyShadowingNames
def select(a: Array, *, where: Array = None) -> Array:
    if where is None:
        return a
    return a[where]

def maximum(a: ArrayLike, b: ArrayLike) -> Array:
    return torch.maximum(tensor(a), tensor(b))


# noinspection PyShadowingBuiltins
def sum(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    return torch.sum(tensor(a), dim=axis, keepdim=keepdims, dtype=dtype)


# noinspection PyShadowingBuiltins
def prod(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    return torch.prod(tensor(a), dim=axis, keepdim=keepdims, dtype=dtype)


# noinspection PyShadowingBuiltins
def min(a: ArrayLike, axis: Axes = None, keepdims: bool = False) -> Array:
    return torch.amin(tensor(a), dim=axis, keepdim=keepdims)


# noinspection PyShadowingBuiltins
def max(a: ArrayLike, axis: Axes = None, keepdims: bool = False) -> Array:
    return torch.amax(tensor(a), dim=axis, keepdim=keepdims)


def mean(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    return torch.mean(tensor(a), dim=axis, keepdim=keepdims, dtype=dtype)


def logsumexp(a: ArrayLike, axis: Axes = None, keepdims: bool = False) -> Array:
    return torch.logsumexp(tensor(a), dim=axis, keepdim=keepdims)


def softmax(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    if axis is None:
        raise ValueError('axis must be specified')
    if isinstance(axis, int):
        out = torch.softmax(tensor(a), dim=axis, dtype=dtype)
        if keepdims:
            shape = list(a.shape)
            if isinstance(axis, int):
                shape[axis] = 1
            out = out.reshape(shape)
        return out
    if isinstance(axis, tuple):
        raise ValueError('axis must be int, not tuple')
    raise ValueError(f'axis must be int or tuple, got {type(axis)}')


def expand_dims(a: Array, axis: Axes = None) -> Array:
    if axis is None:
        raise ValueError('axis must be specified')
    if isinstance(axis, int):
        return a.unsqueeze(axis)
    if isinstance(axis, tuple):
        for ax in axis:
            a = a.unsqueeze(ax)
        return a
    raise TypeError(f'axis must be int or tuple, got {type(axis)}')


def fromfunction(function, shape, *, dtype=float, like=None, **kwargs) -> TorchArray:
    raise NotImplementedError()


def percentile(a: ArrayLike, axis: Axes = ..., dtype: DType = ..., keepdims: bool = ..., **kwargs) -> TorchArray:
    raise NotImplementedError()


def average(a: ArrayLike, axis: Axes = ..., weights: ArrayLike = ..., **kwargs) -> TorchArray:
    raise NotImplementedError()


def update(array: Array, where: Array, value: ArrayOrScalar) -> None:
    s = full_slices[:array.ndim]
    # s = tuple(slice(None) for _ in range(array.ndim))
    array[s] = torch.where(tensor(where), tensor(value), tensor(array))


def is_increasing(vals: TorchArray, strict: bool = False) -> bool:
    return is_monotonic_test(vals, torch.less if strict else torch.less_equal)


def is_decreasing(vals: TorchArray, strict: bool = False) -> bool:
    return is_monotonic_test(vals, torch.greater if strict else torch.greater_equal)


def is_monotonic(vals: TorchArray, strict: bool = False) -> bool:
    return (is_monotonic_test(vals, torch.less if strict else torch.less_equal) or
            is_monotonic_test(vals, torch.greater if strict else torch.greater_equal))


def default_device():
    return None


def new_stream(device) -> Stream:
    return None

from . import functional, random

