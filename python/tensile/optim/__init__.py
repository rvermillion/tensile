#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..shims import ten
from . import optimizer, schedule, types

from .optimizer import Optimizer, OptimizerStep
from .types import TrainFunction, PredictFunction
from .schedule import OptimizerSchedule, LRSchedule


if ten.ten_kind == 'mlx':
    from ..shims.mlx import optim as mlx
elif ten.ten_kind == 'torch':
    from ..shims.torch import optim as torch
else:
    raise ValueError(f'Unsupported ten kind: {ten.ten_kind}')
