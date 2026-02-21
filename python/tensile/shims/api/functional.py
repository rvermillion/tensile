#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from .types import *

def gelu(x: Array) -> Array: ...

def relu(x: Array) -> Array: ...

def sigmoid(x: Array) -> Array: ...

def silu(x: Array) -> Array: ...

__all__ = [
    'gelu',
    'relu',
    'sigmoid',
    'silu'
]
