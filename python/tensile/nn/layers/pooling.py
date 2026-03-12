#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import operator
from itertools import accumulate

from ..common import *
from ..module import CompiledModule, ModuleArgs


class PoolingArgs(ModuleArgs):
    d: int = 1
    pool: str = "max"
    kernel_size: int|list[int]
    stride: int|list[int] = None
    padding: int|list[int] = 0


def _to_tuple(x, n, msg) -> tuple:
    if isinstance(x, (list, tuple)):
        if len(x) != n: raise ValueError(msg)
        return tuple(x)

    if not isinstance(x, int): raise ValueError(msg)
    return (x, ) * n


# This function is copied almost verbatim from the MLX implementation
def _non_overlapping_sliding_windows(x: Array, shape: Shape, window_shape: Shape):
    # Compute the intermediate shape
    new_shape = [shape[0]]
    for s, w in zip(shape[1:], window_shape):
        new_shape.append(s // w)
        new_shape.append(w)
    new_shape.append(shape[-1])
    new_shape = tuple(new_shape)

    last_axis = len(new_shape) - 1
    axis_order = 0, *range(1, last_axis, 2), *range(2, last_axis, 2), last_axis

    x = ten.reshape(x, new_shape)
    x = ten.transpose(x, axis_order)
    return x


# This function is copied almost verbatim from the MLX implementation
def _sliding_windows(x: Array, window_shape: Shape, window_strides: Shape):
    if x.ndim < 3:
        raise ValueError(
            f"To extract sliding windows at least 1 spatial dimension "
            f"(3 total) is needed but the input only has {x.ndim} dimensions."
        )

    shape = x.shape
    spatial_dims = shape[1:-1]
    if not (len(spatial_dims) == len(window_shape) == len(window_strides)):
        raise ValueError(
            f"To extract sliding windows the window shapes and strides must have "
            f"the same number of spatial dimensions as the signal but the signal "
            f"has {len(spatial_dims)} dims and the window shape has {len(window_shape)} "
            f"and strides have {len(window_strides)}."
        )

    if all(
        window == stride and size % window == 0
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ):
        return _non_overlapping_sliding_windows(x, shape, window_shape)

    strides = list(reversed(list(accumulate(reversed(shape + (1,)), operator.mul))))[1:]

    # Compute the output shape
    final_shape = [shape[0]]
    final_shape += [
        (size - window) // stride + 1
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ]
    final_shape += window_shape
    final_shape.append(shape[-1])
    final_shape = tuple(final_shape)

    # Compute the output strides
    final_strides = [strides[0]]
    final_strides += [
        og_stride * stride for og_stride, stride in zip(strides[1:-1], window_strides)
    ]
    final_strides += strides[1:]
    final_strides = tuple(final_strides)

    return ten.as_strided(x, final_shape, final_strides)


avg_pooling = 'Avg'
max_pooling = 'Max'

pooling_names: dict[str, str] = {
    'avg': avg_pooling,
    'average': avg_pooling,
    'max': max_pooling,
    'mean': avg_pooling,
}


class Pool(CompiledModule):

    __slots__ = ("d", "pool", "kernel_size", "stride", "padding")

    d: int
    pool: str
    padding: list[tuple[int, int]]
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]

    pool: Callable

    def init_from_args(self, args: PoolingArgs):
        d = args.d

        pool = args.pool.lower()
        kernel_size = args.kernel_size
        stride = args.stride
        padding = args.padding

        if pool in pooling_names:
            self.pool = pooling_names[pool]
        else:
            raise ValueError(f"Invalid pool type: {pool}")

        class_name = self._repr_type()
        msg = "[{}] '{}' must be an integer or a tuple containing {} integer"
        kernel_size = _to_tuple(
            kernel_size, d, msg.format(class_name, "kernel_size", d)
        )
        if stride is not None:
            stride = _to_tuple(stride, d, msg.format(class_name, "stride", d))
        else:
            stride = kernel_size
        padding = _to_tuple(padding, d, msg.format(class_name, "padding", d))
        padding = [(p, p) for p in padding]

        self.d = d
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @property
    def in_dim(self) -> int:
        return -1

    @property
    def out_dim(self) -> int:
        return -1

    def build_call(self, train: bool = False, **options) -> Callable:
        if self.pool == avg_pooling:
            pool = ten.mean
            padding_value = 0.
        elif self.pool == max_pooling:
            pool = ten.max
            padding_value = -float("inf")
        else:
            raise ValueError(f"Invalid pool type: {self.pool}")

        axes = tuple(range(-len(self.kernel_size) - 1, -1, 1))
        stride = self.stride
        kernel_size = self.kernel_size
        padding = self.padding

        if any(p[0] > 0 for p in padding):
            full_padding = [(0, 0)] + padding + [(0, 0)]
            def call(x):
                x = ten.pad(x, full_padding, constant_values=padding_value)
                x = _sliding_windows(x, kernel_size, stride)
                return pool(x, axis=axes)
        else:
            def call(x):
                x = _sliding_windows(x, kernel_size, stride)
                return pool(x, axis=axes)
        return call

    def _repr_type(self, **options) -> str:
        return f'{self.pool}Pool{self.d}d'

    def _extra_structure(self):
        ks = tuple(self.kernel_size)
        st = tuple(self.stride)
        pd = tuple(p[0] for p in self.padding)

        return f"kernel_size={ks}, stride={st}, padding={pd}"
