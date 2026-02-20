#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
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