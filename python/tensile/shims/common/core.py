#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from .types import *

def size_of_shape(shape: Shape) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


__all__ = [
    'size_of_shape',
]