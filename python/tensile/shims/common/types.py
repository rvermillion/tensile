#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import TypeAlias, TypeVar, Union

Scalar: TypeAlias = Union[int, float, bool]
Shape: TypeAlias = tuple[int, ...]
ShapeLike: TypeAlias = Union[int, Shape]
Axis: TypeAlias = int
Axes: TypeAlias = Union[None, int, tuple[int, ...]]
T = TypeVar('T')

S = TypeVar('S', bound=Scalar)

MaybeTuple: TypeAlias = Union[T, tuple[T, ...]]


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