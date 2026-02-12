#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import TypeAlias, TypeVar, Union

Scalar: TypeAlias = Union[int, float, bool]
Shape: TypeAlias = tuple[int, ...]
ShapeLike: TypeAlias = Union[int, Shape]
Axis: TypeAlias = int
Axes: TypeAlias = Union[None, int, tuple[int, ...]]
T = TypeVar('T')

S = TypeVar('S', bound=Scalar)

MaybeTuple: TypeAlias = Union[T, tuple[T, ...]]


def size_of_shape(shape: Shape) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


__all__ = [
    'Axes',
    'Axis',
    'MaybeTuple',
    'Scalar',
    'Shape',
    'ShapeLike',
    'size_of_shape',
    'S',
    'T',
]