#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from .shims import api as ten
# else:
#     pass
#     from .shims import mlx as ten
#     from .shims import torch as ten
# from .shims.mlx import core as ten
# from .shims import torch as ten
from .shims import core as ten

Array = ten.Array
ArrayLike = ten.ArrayLike
AxisSelector = ten.AxisSelector
DType = ten.DType
DTypeLike = ten.DTypeLike
Scalar = ten.Scalar
Selector = ten.Selector
Shape = ten.Shape
Slice = slice

class Select:

    __slots__ = ()

    def __getitem__(self, item: Selector) -> Selector:
        return item


full_slice: AxisSelector = slice(None)
select = Select()


__all__ = [
    'Array',
    'ArrayLike',
    'AxisSelector',
    'DType',
    'DTypeLike',
    'Scalar',
    'Selector',
    'Shape',
    'Slice',
    'full_slice',
    'select',
    'ten',
]
