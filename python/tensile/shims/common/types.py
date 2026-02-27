#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import TypeAlias, TypeVar, Union

Scalar = Union[int, float, bool]
Shape = tuple[int, ...]
ShapeLike = Union[int, Shape]
Axis = int
Axes = Union[None, int, tuple[int, ...]]
T = TypeVar('T')

S = TypeVar('S', bound=Scalar)

MaybeTuple = Union[T, tuple[T, ...]]


__all__ = [
    'Axes',
    'Axis',
    'Scalar',
    'Shape',
    'ShapeLike',
    'T',
    'S',
    'MaybeTuple'
]