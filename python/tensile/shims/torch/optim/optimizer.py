#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import torch.optim as optim

from tensile.nn.common import *

from tensile.optim.types import *
from tensile.optim.optimizer import Optimizer, OptimizerParamGroup, OptimizerStep


backend_classes = {
    'adadelta': optim.Adadelta,
    'adafactor': optim.Adafactor,
    'adamax': optim.Adamax,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD,
}


backend_aliases = {
    'learning_rate': 'lr',
}


@provides(Optimizer, 'torch')
class TorchOptimizer(Optimizer[Batch]):

    __slots__ = ('groups_per_backend',)

    backends: Annotated[list[optim.Optimizer], field(
        doc='The backends for this optimizer.'
    )]
    groups_per_backend: Annotated[list[list[OptimizerParamGroup]], field(
        doc='The parameter groups per backend.',
        default_factory=list,
    )]

    hyperparameter_aliases = {'learning_rate': 'lr'}

    def _lazy_backends(self) -> list[optim.Optimizer]:
        backends = {}
        for param_group in self.param_groups:
            algo = param_group.config.algorithm
            if groups := backends.get(algo):
                groups.append(param_group)
            else:
                backends[algo] = [param_group]
        return [self.build_backend(algo, param_groups) for algo, param_groups in backends.items()]

    def build_backend(self, algo: str, param_groups: list[OptimizerParamGroup]) -> optim.Optimizer:
        backend_groups = []
        cls = backend_classes[algo]
        step = self.current_step
        for param_group in param_groups:
            params = param_group.filter_params(self.model)
            backend_groups.append({
                'params': params,
                **param_group.schedules.get(step, aliases=self.hyperparameter_aliases)
            })
        self.groups_per_backend.append(param_groups)
        return cls(backend_groups, **self.backend_hyperparameters())

    def start_step(self, batch: Batch) -> None:
        super().start_step(batch)
        self.schedule_backend()

    def schedule_backend(self):
        step = self.current_step
        for backend, param_groups in zip(self.backends, self.groups_per_backend):
            for backend_group, param_group in zip(backend.param_groups, param_groups):
                if schedule := param_group.schedules.get(step, include_constant=False):
                    backend_group.update(schedule)

    @property
    def lr(self) -> float:
        lr = self.learning_rate
        if callable(lr):
            return lr(self.current_step).item()
        elif isinstance(lr, float):
            return lr
        return 0.

    def trainable_parameter_list(self, group: int = None) -> list[Array]:
        return list(v for p, v in tree.flatten(self.trainable_parameters(group=group)))

    def stepper(self, train_fn: TrainFunction[Batch],  *,
                grad_handlers: list[GradientHandler] = None,
                start_step: OptimizerStartStep = None,
                end_step: OptimizerEndStep = None,
                **kwargs,
                ) -> OptimizerStep[Batch]:

        optimizers = self.backends
        if len(optimizers) == 1:
            optimizer = optimizers[0]

            zero_grad = optimizer.zero_grad

            optimizer_step = optimizer.step
        else:
            def zero_grad():
                for opt in optimizers:
                    opt.zero_grad()

            def optimizer_step() -> None:
                for opt in optimizers:
                    opt.step()

        def step(batch: Batch) -> Array:
            # x, y = batch.data
            self.start_step(batch)
            if start_step: start_step(self, batch)

            zero_grad()

            loss = train_fn(batch)

            ten.get_active_memory()

            loss.backward()

            if grad_handlers:
                flat_grads = [e for e in tree.flatten(self.trainable_parameters()) if e.value.grad is not None]
                for grad_handler in grad_handlers:
                    grad_handler(self, flat_grads)

            ten.get_active_memory()

            optimizer_step()

            detached_loss = ten.detach(loss)
            if end_step: end_step(self, detached_loss, batch)
            self.finish_step(detached_loss, batch)
            return loss


        return step


def add_args(opt: dict[str, Any], **kwargs) -> dict[str, Any]:
    for k, v in kwargs.items():
        if v is not None:
            opt.setdefault(k, v)
    return opt
