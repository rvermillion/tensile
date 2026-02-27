#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from typing import Any, Sequence, TypeAlias, TypeGuard, Union

import torch
import numpy as np

from .types import *

TorchArray = torch.Tensor
TorchDType = torch.Type


def to_shape(size: ShapeLike) -> Shape:
    return (size, ) if isinstance(size, int) else size


def is_tensor(obj: Any) -> TypeGuard[TorchArray]:
    return isinstance(obj, TorchArray)


def ensure(a: ArrayLike, *args, **kwargs) -> TorchArray:
    return a if isinstance(a, TorchArray) else tensor(a, *args, **kwargs)


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

def as_type(a: Any, dtype: DType) -> Array:
    if is_array(a):
        return a.to(dtype)
    return tensor(a, dtype=dtype)

from torch import (
    tensor,
    zeros, zeros_like, ones, ones_like, full, full_like, empty, empty_like,
    arange, concatenate, reshape,
    abs, square, sqrt, exp, log, expm1, sin, cos, tan,
    median, std, var, quantile,
    pi,
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
    get_default_device as default_device, set_default_device,
)


from torch.nn.parameter import Parameter

ten_kind: str = 'torch'

Stream = None

def eval(*args) -> None:
    pass

debug_eval = eval

# noinspection PyShadowingNames
def array(data, *args, **kwargs) -> Array:
    x = ensure(data, *args, **kwargs)
    if x.device.type != 'mps':
        print('Non-MPS device detected. Consider using MPS for better performance.')
    return x

# noinspection PyShadowingNames
def parameter(x: Array) -> Array:
    return Parameter(x)


def detach(a: Array) -> Array:
    return a.detach()

# # noinspection PyShadowingNames
# def zeros(shape: Shape, dtype: DTypeLike = ..., *args, **kwargs) -> Array: ...
#
# # noinspection PyShadowingNames
# def ones(shape: Shape, dtype: DTypeLike = ..., *args, **kwargs) -> Array: ...
#
# # noinspection PyShadowingNames
# def full(shape: Shape, fill_value: Scalar, dtype: DTypeLike = ..., *args, **kwargs) -> Array: ...
#
# # noinspection PyShadowingNames
# def zeros_like(array: Array, dtype: DTypeLike = ..., *args, **kwargs) -> Array: ...
#
# def ones_like(array: Array, dtype: DTypeLike = ..., *args, **kwargs) -> Array: ...
#
# def full_like(array: Array, fill_value: Scalar, dtype: DTypeLike = ..., *args, **kwargs) -> Array: ...
#
# def fill(array: Array, fill_value: Scalar, /, **kwargs) -> None: ...
#
# def eye(n: int, m: int = ..., k: int = ..., dtype: DTypeLike = ..., **kwargs) -> Array: ...
#
# def trace(a: Array, /, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: DTypeLike | None = None, **kwargs) -> Array: ...
#
# def fromfunction(function, shape, *, dtype=float, like=None, **kwargs) -> Array: ...
#
# def arange(start: Scalar, stop: Scalar = ..., step: Scalar = ..., dtype: DType = ...) -> Array: ...


def size(a: Array) -> int:
    return a.numel()


def transpose(a: Array, axes: Axes = None) -> Array:
    return a.permute(axes)

# noinspection PyShadowingNames
def select(a: Array, *, where: Array = None) -> Array:
    if where is None:
        return a
    return a[where]

def maximum(a: ArrayLike, b: ArrayLike) -> Array:
    return torch.maximum(ensure(a), ensure(b))


# noinspection PyShadowingBuiltins
def sum(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    return torch.sum(ensure(a), dim=axis, keepdim=keepdims, dtype=dtype)


# noinspection PyShadowingBuiltins
def norm(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    return torch.norm(ensure(a), dim=axis, keepdim=keepdims, dtype=dtype)


# noinspection PyShadowingBuiltins
def prod(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    return torch.prod(ensure(a), dim=axis, keepdim=keepdims, dtype=dtype)


# noinspection PyShadowingBuiltins
def min(a: ArrayLike, axis: Axes = None, keepdims: bool = False) -> Array:
    return torch.amin(ensure(a), dim=axis, keepdim=keepdims)


# noinspection PyShadowingBuiltins
def max(a: ArrayLike, axis: Axes = None, keepdims: bool = False) -> Array:
    return torch.amax(ensure(a), dim=axis, keepdim=keepdims)


def mean(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    return torch.mean(ensure(a), dim=axis, keepdim=keepdims, dtype=dtype)


def logsumexp(a: ArrayLike, axis: Axes = None, keepdims: bool = False) -> Array:
    return torch.logsumexp(ensure(a), dim=axis, keepdim=keepdims)


def softmax(a: ArrayLike, axis: Axes = None, keepdims: bool = False, dtype: DType = None) -> Array:
    if axis is None:
        raise ValueError('axis must be specified')
    if isinstance(axis, int):
        out = torch.softmax(ensure(a), dim=axis, dtype=dtype)
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
    array[s] = torch.where(ensure(where), ensure(value), ensure(array))


def is_increasing(vals: TorchArray, strict: bool = False) -> bool:
    return is_monotonic_test(vals, torch.less if strict else torch.less_equal)


def is_decreasing(vals: TorchArray, strict: bool = False) -> bool:
    return is_monotonic_test(vals, torch.greater if strict else torch.greater_equal)


def is_monotonic(vals: TorchArray, strict: bool = False) -> bool:
    return (is_monotonic_test(vals, torch.less if strict else torch.less_equal) or
            is_monotonic_test(vals, torch.greater if strict else torch.greater_equal))



def new_stream(device) -> Stream:
    return None

peak_memory = 0

def get_peak_memory():
    get_active_memory()
    return peak_memory

def get_active_memory():
    global peak_memory
    mem = 0
    if torch.cuda.is_available():
        mem += torch.cuda.max_memory_allocated()
    if torch.mps.is_available():
        mem += torch.mps.current_allocated_memory()
    if mem > peak_memory:
        peak_memory = mem
    return mem


from safetensors.torch import save_file, load_file

def load_tensors(filename: str) -> dict[str, Array]:
    arrays = load_file(filename)

    for k, v in arrays.items():
        arrays[k] = v.to(device=default_device())

    return arrays

# noinspection PyShadowingBuiltins
def save_tensors(filename: str, arrays: dict[str, Array], format: str = None) -> None:
    if format is None:
        if filename.endswith('.npz'):
            format = 'npz'
        elif filename.endswith('.safetensors'):
            format = 'safetensors'

    for k, v in arrays.items():
        arrays[k] = v.detach().cpu()

    if format == 'npz':
        raise ValueError(f'Unknown format {format}')
    elif format == 'safetensors':
        save_file(arrays, filename)
    else:
        raise ValueError(f'Unknown format {format}')

from . import fast, functional, random

