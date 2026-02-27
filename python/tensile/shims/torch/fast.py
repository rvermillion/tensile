#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import torch
from torch.nn import functional as F


from typing import Optional

from .types import *


def rms_norm(x: Array, weight: Optional[Array] = None, eps: float = ...,  **kwargs) -> Array:
    shape = weight.shape if weight is not None else (x.shape[-1],)
    return F.rms_norm(x, shape, weight=weight, eps=eps)


def rope(x: Array, dims: int,  *, traditional: bool = None, base: float = None, offset: Array|int = 0, **kwargs) -> Array:
    """
    x: (B, H, L, D)
    seq_positions: (L,) or None (defaults to arange)
    """
    L, D = x.shape[-2:]
    if dims >= D: dims = D
    assert dims % 2 == 0

    seq_positions = torch.arange(L, device=x.device)
    if isinstance(offset, int):
        seq_positions += offset
    else:
        seq_positions += offset[..., None]

    half = D // 2
    freq_seq = torch.arange(half, device=x.device)
    inv_freq = 1.0 / (base ** (freq_seq / half))

    # (L, half)
    angles = torch.outer(seq_positions, inv_freq)

    sin = angles.sin()[None, None, :, :]
    cos = angles.cos()[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:]

    # rotation
    x_rotated = torch.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rotated