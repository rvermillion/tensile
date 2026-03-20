#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ...infra import RootObject
from ...common import *

from ..optimizer import Optimizer, OptimizerParamGroup, OptimizerParamInfo


class OptimizerAlgorithm(RootObject):

    __slots__ = ()

    def init_state(self, parameter: Array, info: OptimizerParamInfo):
        raise NotImplementedError()

    def apply_gradient(self, gradient: Array, parameter: Array, info: OptimizerParamInfo) -> Array:
        raise NotImplementedError()

