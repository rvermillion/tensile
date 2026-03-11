#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
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
@ten.compile()
def swiglu(gate: Array, x: Array) -> Array:
    return silu(gate) * x


@meta.provides(Activation, 'weighted')
class WeightedActivation(Module):

    __slots__ = ('weights', 'activations')

    weights: Array
    activations: list[Activation]

    def __call__(self, x: Array) -> Array:
        return ten.matmul(ten.softmax(self.weights), ten.stack([act(x) for act in self.activations], axis=0))


__all__ = [
    'Activation',
    'WeightedActivation'
]

