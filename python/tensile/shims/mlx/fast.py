#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from ...infrastructure.types import Callable
from .types import Array

import mlx.core as mx

from mlx.core.fast import (
    layer_norm,
    rms_norm,
    rope as fast_rope,
    scaled_dot_product_attention,
)


_validate = False


rope_cache: dict[tuple[int, float], tuple[Array, Array]] = {}


def get_sincos(half: int, offset: int, length: int, base: float) -> tuple[Array, Array]:
    end = offset + length
    sin = None
    cos = None
    try:
        sin, cos = rope_cache[half, base]
        if sin.shape[-2] >= end:
            return sin[..., offset:end, :], cos[..., offset:end, :]
    except KeyError:
        pass

    start = 0 if sin is None else sin.shape[-2]
    align = end + 256 - end % 256
    seq_positions = mx.arange(start, align)

    freq_seq = mx.arange(half)
    inv_freq = 1.0 / (base ** (freq_seq / half))

    # (L, half)
    angles = mx.outer(seq_positions, inv_freq)  #.astype(dtype=mx.float32)  #x.dtype)   # [None, None, :, :]

    angles = mx.expand_dims(angles, axis=(0, 1))

    if start > 0:
        sin = mx.concatenate([sin, mx.sin(angles)], axis=-2)
        cos = mx.concatenate([cos, mx.cos(angles)], axis=-2)
    else:
        sin = mx.sin(angles)
        cos = mx.cos(angles)

    rope_cache[half, base] = sin, cos

    return sin[..., offset:end, :], cos[..., offset:end, :]


# noinspection PyPep8Naming
def slow_rope(x: Array, dims: int,  *, traditional: bool = None, base: float = None, offset: Array|int = 0,
            scale: float = 1.0,
            **kwargs) -> Array:
    """
    x: (B, H, L, D)
    seq_positions: (L,) or None (defaults to arange)
    """
    # x = mx.ones_like(x)
    L, D = x.shape[-2:]

    d = min(D, dims)
    assert d % 2 == 0

    half = dims // 2

    # These will be shape (1, 1, L, half)
    sin, cos = get_sincos(half, offset, L, base)

    # seq_positions = mx.arange(L)
    # if isinstance(offset, int):
    #     seq_positions += offset
    # else:
    #     seq_positions += offset[..., None]
    #
    # half = d // 2
    # freq_seq = mx.arange(half)
    # inv_freq = 1.0 / (base ** (freq_seq / half))
    #
    # # (L, half)
    # angles = mx.outer(seq_positions, inv_freq)  #.astype(dtype=mx.float32)  #x.dtype)   # [None, None, :, :]
    #
    # angles = mx.expand_dims(angles, axis=(0, 1))
    #
    # sin = mx.sin(angles)
    # cos = mx.cos(angles)

    x1 = x[..., :half]
    x2 = x[..., half:d]

    parts = [
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ]

    if d < D:
        parts.append(x[..., d:])

    # rotation
    x_rotated = mx.concatenate(parts, axis=-1).astype(x.dtype)

    if _validate:
        x_fast = fast_rope(x, dims, traditional=traditional, base=base, offset=offset, scale=scale, **kwargs)
        if not mx.allclose(x_rotated, x_fast):
            delta = x_rotated - x_fast
            max_abs = mx.max(mx.abs(delta), axis=-1, keepdims=True)
            max_max_abs = mx.max(max_abs)
            mx.eval(x_rotated, x_fast, delta, max_max_abs, max_abs)
            print(f'WARNING: rope mismatch max_max_abs={mx.max(max_abs)}')
            pass

    return x_rotated


rope = fast_rope
