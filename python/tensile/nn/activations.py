#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from typing import Sequence

from .common import Array, Activation, BinaryActivation, meta, ten
from .module import Module


activation_modules = {
    'mlx': ['mlx.nn.layers.activations'],
    'numpy': [],
    'torch': ['torch.nn.functional'],
}

meta.for_class(Activation).configure_registry(
    modules=activation_modules[ten.ten_kind],
    default_kind='silu'
)


silu: Activation = meta.coerce(Activation, kind='silu')


@meta.provides_singleton(BinaryActivation, 'swiglu')
@ten.compile(shapeless=True)
def swiglu(gate: Array, x: Array) -> Array:
    return silu(gate) * x



__all__ = [
    'Activation',
]

