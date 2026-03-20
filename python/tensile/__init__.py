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
# import sys
from . import infra, util, ten

infrastructure = infra

# from .shims import ten
# sys.modules['tensile.ten'] = ten
#
from .util import select

from .ten import (
    Array,
    ArrayLike,
    AxisSelector,
    DType,
    DTypeLike,
    Scalar,
    Selector,
    Shape,
)
from .common import (
    Slice,
    full_slice,
)


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
