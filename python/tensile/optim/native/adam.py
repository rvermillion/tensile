#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ...common import *

from ..optimizer import OptimizerConfig, OptimizerParamInfo
from .algorithm import OptimizerAlgorithm


@provides(OptimizerConfig, 'native.adam')
class AdamConfig(OptimizerConfig):

    __slots__ = ('betas', 'eps', 'bias_correction')

    eps: Annotated[float, field(
        doc='The epsilon value for this optimizer.',
        default=1e-8,
    )]
    betas: Annotated[Optional[tuple[float, float]], field(
        doc='The betas for AdamW.',
        default=(0.9, 0.999),
    )]
    bias_correction: Annotated[bool, field(
        doc='Whether to use bias correction for Adam.',
        default=False,
    )]

    @staticmethod
    def _coerce_betas(betas: Any):
        if betas is None: return None
        if isinstance(betas, tuple): return betas
        if isinstance(betas, list): return tuple(betas[:2])
        raise ValueError(f'Invalid betas: {betas}')

    algorithm: ClassVar[str] = 'adam'
    hyperparameter_names = ('eps', 'betas', 'bias_correction')


@provides(OptimizerAlgorithm, 'adam')
class AdamAlgorithm(OptimizerAlgorithm):

    __slots__ = ()

    def init_state(self, parameter: Array, info: OptimizerParamInfo):
        """Initialize optimizer state"""
        state = info.state
        state["m"] = ten.zeros_like(parameter)
        state["v"] = ten.zeros_like(parameter)

    def apply_gradient(self, gradient: Array, parameter: Array, info: OptimizerParamInfo) -> Array:
        """Performs the Adam parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        state = info.state
        if not state: self.init_state(parameter, info)

        config = info.group.config
        lr = config.learning_rate.astype(gradient.dtype)
        b1, b2 = config.betas
        eps = config.eps
        bias_correction = config.bias_correction
        step = config.step

        m = state["m"]
        v = state["v"]
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * ten.square(gradient)
        state["m"] = m
        state["v"] = v

        if bias_correction:
            c1 = (lr / (1 - b1**step)).astype(gradient.dtype)
            c2 = ten.rsqrt(1 - b2**step).astype(gradient.dtype)
            numerator = c1 * m
            denominator = ten.sqrt(v) * c2 + eps
            return parameter - numerator / denominator
        else:
            return parameter - lr * m / (ten.sqrt(v) + eps)

