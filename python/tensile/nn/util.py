#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

from .common import Array, DType, ten, TYPE_CHECKING
from .layers.linear import Linear


if TYPE_CHECKING:
    import tensile.nn.module


if ten.ten_kind == 'mlx':
    fast_sdp_attention = ten.fast.scaled_dot_product_attention
else:
    fast_sdp_attention = None
    # def fast_sdp_attention(q: Array, k: Array, v: Array, *,
    #                        scale: float,
    #                        mask: None | str | Array = None,
    #                        sinks: Array | None = None,
    #                        stream: ten.Stream = None) -> Array:
    #     raise NotImplementedError()


def zero(array: Array, scale: float = None, precision: DType = None):
    if precision is None:
        precision = array.dtype
    if scale is None:
        return ten.zeros(array.shape, dtype=precision)
    return ten.random.uniform(
        low=-scale,
        high=scale,
        shape=array.shape,
        dtype=precision,
    )


def nilpotent(module: 'tensile.nn.module.Module', scale: float = None, precision: DType = None):
    if nilpot := getattr(module, 'nilpotent', None):
        if callable(nilpot):
            nilpot(scale=scale, precision=precision)
        else:
            raise TypeError('nilpotent must be callable')
    elif isinstance(module, Linear):
        module.weight = zero(module.weight, scale=scale, precision=precision)
        if 'bias' in module:
            module.bias = zero(module.bias, scale=scale, precision=precision)
    else:
        raise ValueError(f'Cannot make module nilpotent: {module}')

__all__ = [
    # 'BaseModelArgs',
    'nilpotent',
    'fast_sdp_attention',
]