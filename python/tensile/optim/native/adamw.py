#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ...common import *

from ..optimizer import OptimizerConfig, OptimizerSchedule, OptimizerParamInfo
from .algorithm import OptimizerAlgorithm

from .adam import AdamConfig, AdamAlgorithm


@provides(OptimizerConfig, 'native.adamw')
class AdamWConfig(AdamConfig):

    __slots__ = ('weight_decay',)

    weight_decay: Annotated[Optional[OptimizerSchedule], field(
        doc='The weight decay coefficient for AdamW.'
    )]

    @staticmethod
    def _coerce_weight_decay(spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    algorithm: ClassVar[str] = 'adamw'
    schedule_names = (*AdamConfig.schedule_names, 'weight_decay')


@provides(OptimizerAlgorithm, 'adamw')
class AdamWAlgorithm(AdamAlgorithm):

    __slots__ = ()

    def apply_gradient(self, gradient: Array, parameter: Array, info: OptimizerParamInfo) -> Array:
        """Performs the AdamW parameter update by modifying the parameters
        passed into Adam.
        """

        lr = self.learning_rate.astype(gradient.dtype)
        return super().apply_gradient(
            gradient, parameter * (1 - lr * self.weight_decay), info
        )
