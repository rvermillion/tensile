#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from tensile import Array
from tensile.infra.tree import TreeEntry
from tensile.nn import Module
from tensile.infra.types import Callable, Iterable, Protocol, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import tensile.optim.optimizer


Batch = TypeVar('Batch')
Inputs = TypeVar('Inputs')
Outputs = TypeVar('Outputs')


class TrainFunction(Protocol[Batch]):

    def __call__(self, batch: Batch) -> Array: ...


class PredictFunction(Protocol[Inputs, Outputs]):

    def __call__(self, inputs: Inputs) -> Outputs: ...


class PredictionHandler(Protocol[Batch, Outputs]):

    def __call__(self, model: Module, predictions: Outputs, batch: Batch) -> None: ...


class OptimizerStep(Protocol[Batch]):

    def __call__(self, batch: Batch) -> Array: ...


class GradientHandler(Protocol):

    def __call__(self, optimizer: 'tensile.optim.optimizer.Optimizer', grads: Iterable[TreeEntry[Array]]) -> None: ...


OptimizerStartStep = Callable[['tensile.optim.optimizer.Optimizer[Batch]', Batch], None]
OptimizerEndStep = Callable[['tensile.optim.optimizer.Optimizer[Batch]', Array, Batch], None]


class OptimizerStepHandler(Protocol[Batch]):

    def on_start(self, optimizer: 'tensile.optim.optimizer.Optimizer[Batch]', batch: Batch): ...

    def on_end(self, optimizer: 'tensile.optim.optimizer.Optimizer[Batch]', loss: Array, batch: Batch) -> None: ...
