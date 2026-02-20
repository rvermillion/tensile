#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from .types import *

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