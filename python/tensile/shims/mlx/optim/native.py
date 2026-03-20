#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile.common import *
from tensile.optim.native.algorithm import OptimizerAlgorithm

from tensile.optim.types import *

from .optimizer import MLXOptimizer, backend_factories

class NativeBackend(Object):

    __slots__ = ('optimizer', 'algorithm')

    optimizer: Annotated[MLXOptimizer, field(
        doc='The optimizer this backend belongs to.',
    )]
    algorithm: Annotated[OptimizerAlgorithm, field(
        doc='The algorithm this backend uses.',
    )]

    def _lazy_algorithm(self):
        from tensile.optim.native.adam import AdamAlgorithm
        return AdamAlgorithm()

    def preinit(self, spec: Spec):
        print('preinit:', spec)

    @property
    def state(self) -> Any:
        return {}

    @state.setter
    def state(self, state: dict) -> None:
        pass

    def update(self, model: Module, gradients: dict) -> None:
        ten.debug_eval(model.parameters(), gradients)
        optimizer = self.optimizer
        algorithm = self.algorithm

        updates = {}
        for param, grad in tree.join(model, gradients, include=tree.is_array_entry):
            info = optimizer.all_params[param.path]
            updates[param.path] = algorithm.apply_gradient(grad.value, param.value, info)
            print(f'{param.path}: {info}')
        model.update(tree.unflatten(updates))

    def apply_gradients(self, gradients: dict, parameters: dict) -> Any:
        optimizer = self.optimizer
        algorithm = self.algorithm

        updates = {}
        for param, grad in tree.join(parameters, gradients, include=tree.is_array_entry):
            info = optimizer.all_params[param.path]
            updates[param.path] = algorithm.apply_gradient(grad.value, param.value, info)
            print(f'{grad.path}: {grad.value.shape}')
        return tree.unflatten(updates)


backend_factories['native'] = NativeBackend
