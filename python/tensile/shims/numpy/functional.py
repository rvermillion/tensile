#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import numpy as np
from .types import *


def gelu(x: Array) -> Array:
    raise NotImplementedError()

def relu(x: Array) -> Array:
    return np.maximum(x, 0.0)

def sigmoid(x: Array) -> Array:
    return 1. / (1. + np.exp(-x))

def silu(x: Array) -> Array:
    raise NotImplementedError()


__all__ = [
    'gelu',
    'relu',
    'sigmoid',
    'silu'
]
