#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from .common import DType, Shape

def promote_types(a: DType, b: DType) -> DType:
    # For now we ignore
    return a


def broadcast_shapes(a: Shape, b: Shape) -> Shape:
    if a == b:
        return a
    if not a:
        return b
    if not b:
        return a
    m = -min(len(a), len(b))
    i = -1
    shape = []
    while (i >= m):
        if a[i] == b[i] or b[i] == 1:
            shape.append(a[i])
        elif a[i] == 1:
            shape.append(b[i])
        else:
            raise ValueError(f'Cannot broadcast shapes {a} and {b}: {a[i]} != {b[i]}')
        i -= 1
    if -m < len(a):
        shape.extend(a[:i+1])
    elif -m < len(b):
        shape.extend(b[:i+1])
    n = tuple(reversed(shape))
    print(f'Broadcasting {a} and {b} and got: {n}')
    return n


__all__ = [
    'broadcast_shapes',
    'promote_types',
]