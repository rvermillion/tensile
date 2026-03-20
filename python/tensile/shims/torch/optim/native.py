#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.common import *
from tensile.optim.native.algorithm import OptimizerAlgorithm

from tensile.optim.types import *

from .optimizer import TorchOptimizer, backend_factories

class NativeBackend(Object):

    __slots__ = ('optimizer', 'algorithm')

    optimizer: Annotated[TorchOptimizer, field(
        doc='The optimizer this backend belongs to.',
    )]
    algorithm: Annotated[OptimizerAlgorithm, field(
        doc='The algorithm this backend uses.',
    )]

    def _lazy_algorithm(self):
        return coerce(OptimizerAlgorithm, kind='adamw')

    def preinit(self, spec: Spec):
        print('preinit:', spec)

    param_groups: list[dict[str, Any]]

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]): ...


backend_factories['native'] = NativeBackend
