#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

# from typing import TYPE_CHECKING
#
# if TYPE_CHECKING:
#     from .shims import api as ten
# else:
#     pass
#     from .shims import mlx as ten
#     from .shims import torch as ten
# from .shims.mlx import core as ten
# from .shims import torch as ten

from . import infra, util

infrastructure = infra

from .shims import ten
from .util import select

Array = ten.Array
ArrayLike = ten.ArrayLike
AxisSelector = ten.AxisSelector
DType = ten.DType
DTypeLike = ten.DTypeLike
Scalar = ten.Scalar
Selector = ten.Selector
Shape = ten.Shape
Slice = slice

full_slice: AxisSelector = slice(None)


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
