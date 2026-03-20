#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
import contextlib
from pathlib import Path
from typing import Any, Callable, Sequence, TypeAlias, TypeGuard, TypeVar, Union

import torch

import torch.nn.functional as F

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
    arange, reshape,
    abs, square, sqrt, exp, log, expm1, sin, cos, tan, sigmoid,
    median, std, var, quantile,
    pi,
    addmm,
    isinf, isnan, isfinite,
    add, subtract, multiply, divide, pow,
    matmul,
    minimum, clip,
    argmin, argmax,
    floor, floor_divide,
    sort, where,
    take, take_along_dim,
    gather,
    conv1d, conv2d, conv3d,
    swapaxes, transpose,
    as_strided,
    broadcast_to,
    rsqrt,
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

def dtype(dt: str|DType) -> DType:
    if isinstance(dt, str):
        dt = dt.lower()
        dt = getattr(torch, dt)
    if isinstance(dt, DType):
        return dt
    raise TypeError(f"Invalid dtype: {dt}")

# noinspection PyShadowingNames
def parameter(x: Array) -> Array:
    return Parameter(x)


def detach(a: Array) -> Array:
    return a.detach()


stop_gradient = detach


def require_grad(a: Array, grad: bool = True) -> Array:
    a.requires_grad = grad
    return a


def pad(x: Array, padding: Sequence[tuple[int, int]] = None, mode: str = 'constant', constant_values: int|float|bool|Array = 0.0) -> Array:
    assert padding[0] == (0, 0), "torch backend does not support padding the batch dimension"
    assert padding[-1] == (0, 0), "torch backend does not support padding the channel dimension"
    flat = [v for lo, hi in reversed(padding[1:-1]) for v in (lo, hi)]
    return F.pad(x, flat, mode=mode, value=constant_values)


def quantize(a: Array, group_size: int, bits: int, mode: str = 'affine') -> tuple[Array, Array, Array]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be positive and <= 8")
    if mode != 'affine':
        raise ValueError("mode must be 'affine'")

    q = torch.zeros((*a.shape[:-1], a.shape[-1] // 4), dtype=a.dtype)
    scales = torch.ones((*a.shape[:-1], a.shape[-1] // group_size), dtype=a.dtype)
    biases = torch.zeros((*a.shape[:-1], a.shape[-1] // group_size), dtype=a.dtype)

    return q, scales, biases


# noinspection PyUnusedLocal,PyShadowingNames
def quantized_matmul(x: Array, weight: Array, scales: Array = None, biases: Array = None, transpose: bool = False,
                     group_size: int = None, bits: int = None, mode: str = None) -> Array:
    """
    Correctness-first Torch version of MLX-style quantized_matmul.

    Dequantizes the weight, then does matmul.

    Common linear case:
      x       : (..., Q, I)
      weight  : (O, I_packed)
      scales  : (O, I // group_size)
      result  : (..., Q, O)   when transpose=True
    """
    w = dequantize(
        weight,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    ).to(x.dtype)

    if transpose:
        return x @ w.transpose(-2, -1)
    else:
        return x @ w


def _unpack_uint32_last_dim(x: Array, bits: int) -> Array:
    """
    Unpack packed uint32 values along the last dimension.

    For bits=8:
      (..., N_packed) -> (..., N_packed * 4)

    Returns integer codes in torch.int32.
    """
    if x.dtype != torch.uint32:
        raise TypeError(f"Expected uint32 packed tensor, got {x.dtype}")
    if bits <= 0 or 32 % bits != 0:
        raise ValueError(f"bits must divide 32, got {bits}")

    values_per_word = 32 // bits
    mask = (1 << bits) - 1

    # Bit ops are more predictable on int64 than uint32 in torch.
    x64 = x.to(device="cpu").to(dtype=torch.int64)

    shifts = torch.arange(
        0, 32, bits, device=x64.device, dtype=torch.int32
    )  # [0, bits, 2*bits, ...]

    # (..., N_packed, values_per_word)
    unpacked = (x64.unsqueeze(-1) >> shifts) & mask

    # Flatten packed words into the logical last dimension
    return unpacked.reshape(*x.shape[:-1], x.shape[-1] * values_per_word).to(dtype=torch.int32, device=x.device)


# noinspection PyUnusedLocal,PyShadowingNames
def dequantize(x: Array, scales: Array = None, biases: Array = None,
               group_size: int = None, bits: int = None, mode: str = None) -> Array:
    """
    MLX-style affine dequantization in Torch.

    Assumes:
      - x is packed uint32
      - packing is along the last dimension
      - scales/biases correspond to groups along that same logical dimension

    Typical shapes for a quantized linear weight:
      x       : (..., O, I_packed)
      scales  : (..., O, I // group_size)
      biases  : (..., O, I // group_size)
      result  : (..., O, I)
    """
    mode = mode or "affine"
    bits = bits or 8
    group_size = group_size or 64

    if mode != "affine":
        raise NotImplementedError(f"Only affine mode is implemented, got {mode!r}")

    q = _unpack_uint32_last_dim(x, bits).to(scales.dtype)  # (..., logical_dim)

    logical_dim = q.shape[-1]
    if logical_dim % group_size != 0:
        raise ValueError(
            f"Unpacked last dim {logical_dim} is not divisible by group_size={group_size}"
        )

    num_groups = logical_dim // group_size

    # Group the logical last dimension
    q = q.reshape(*q.shape[:-1], num_groups, group_size)  # (..., groups, group_size)

    if scales.shape[-1] != num_groups:
        raise ValueError(
            f"scales last dim {scales.shape[-1]} != expected num_groups {num_groups}"
        )

    s = scales.unsqueeze(-1)  # (..., groups, 1)

    if biases is None:
        b = 0.0
    else:
        if biases.shape != scales.shape:
            raise ValueError(f"biases shape {biases.shape} must match scales shape {scales.shape}")
        b = biases.unsqueeze(-1)  # (..., groups, 1)

    w = q * s + b
    return w.reshape(*w.shape[:-2], logical_dim).to(scales.dtype)


def split(x: Array, indices_or_sections: int|Sequence[int], axis: int = 0) -> Sequence[Array]:
    return torch.split(x, indices_or_sections, dim=axis)


@contextlib.contextmanager
def stream(s: Stream):
    yield s


C = TypeVar('C', bound=Callable)


def compile(shapeless: bool = False, **kwargs) -> Callable[[C], C]:
    return torch.compile(dynamic=shapeless, **kwargs)


def concatenate(arrays: list[Array]|tuple[Array, ...], axis: int = 0) -> Array:
    return torch.cat(arrays, dim=axis)


concat = concatenate

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

def contiguous(a: Array) -> Array:
    return a.contiguous()


def take_along_axis(a: Array, indices: Array, axis: int = None) -> Array:
    return take_along_dim(a, indices, dim=axis)


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


# noinspection PyShadowingNames
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


def gather_mm(a: Array, b: Array, /, lhs_indices: Array = None, rhs_indices: Array = None, *, sorted_indices: bool = False, **kwargs) -> Array:
    """
    x:       (*I, J)   - input tokens
    weights: (X, J, K) - per-expert weight matrices
    indices: (|I|,) - which expert each token is routed to [0..X)
    returns: (*I, K)
    """
    if lhs_indices is not None:
        raise NotImplementedError("gather_mm does not support lhs_indices yet")
    if rhs_indices is None:
        raise ValueError("rhs_indices must be specified")

    orig_shape = a.shape
    bshape = b.shape

    assert orig_shape[-1] == bshape[-2], "Incompatible shapes"

    a = a.reshape(-1, orig_shape[-2], orig_shape[-1])

    new_shape = a.shape

    assert new_shape[0] == rhs_indices[0], "Right hand side indices do not match input tensor size"

    output = torch.zeros((new_shape[0], new_shape[1], bshape[-1]), dtype=a.dtype)

    for i in range(bshape[0]):
        mask = rhs_indices == i
        if not torch.any(mask):
            continue
        a_i = a[mask]                    # (n_i, D_in)
        output[mask] = a_i @ b[i]    # (n_i, D_out)

    output = output.reshape(*orig_shape[:-1], bshape[-1])

    return output

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

def _load_safetensors(path: Path, device=None, **kwargs) -> dict[str, Array]:
    if device is None: device = default_device()
    arrays = load_file(path)
    return {name: x.to(device) for name, x in arrays.items()}


tensor_loaders = {
    'safetensors': _load_safetensors
}

# noinspection PyShadowingBuiltins
def load_tensors(filename: str|Path, /, format: str = None, **kwargs) -> dict[str, Array]:
    filename = Path(filename)
    if format is None: format = filename.suffix[1:]
    if loader := tensor_loaders.get(format):
        return loader(filename, **kwargs)
    raise ValueError(f'Unsupported file format: {format}')


def _save_safetensors(path: Path, arrays: dict[str, Array], **kwargs):
    save_file(arrays, path, **kwargs)

tensor_savers = {
    'safetensors': _save_safetensors,
}

# noinspection PyShadowingBuiltins
def save_tensors(filename: str|Path, arrays: dict[str, Array], format: str = None) -> None:
    filename = Path(filename)
    if format is None: format = filename.suffix[1:]

    for k, v in arrays.items():
        arrays[k] = v.detach().cpu()

    if saver := tensor_savers.get(format):
        saver(filename, arrays)
    else:
        raise ValueError(f'Unknown format {format}')

from . import fast, functional, random

