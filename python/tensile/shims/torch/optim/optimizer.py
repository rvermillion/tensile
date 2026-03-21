#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from pathlib import Path

import torch
import torch.optim as optim

from tensile.common import *

from tensile.optim.types import *
from tensile.optim.optimizer import Optimizer, OptimizerParamGroup, OptimizerStep


class TorchBackend(Protocol):

    param_groups: list[dict[str, Any]]

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]): ...


TorchBackendFactory = Callable[..., TorchBackend]


@provides(Optimizer, 'torch')
class TorchOptimizer(Optimizer[Batch]):

    __slots__ = ('groups_per_backend',)

    backends: Annotated[list[TorchBackend], field(
        doc='The backends for this optimizer.'
    )]
    groups_per_backend: Annotated[list[list[OptimizerParamGroup]], field(
        doc='The parameter groups per backend.',
        default_factory=list,
    )]

    hyperparameter_aliases = {'learning_rate': 'lr'}

    _step_dtype = ten.int64

    def _lazy_backends(self) -> list[TorchBackend]:
        backends = {}
        for param_group in self.param_groups:
            algo = param_group.config.algorithm
            if groups := backends.get(algo):
                groups.append(param_group)
            else:
                backends[algo] = [param_group]
        return [self.build_backend(algo, param_groups) for algo, param_groups in backends.items()]

    def build_backend(self, algo: str, param_groups: list[OptimizerParamGroup]) -> TorchBackend:
        backend_groups = []
        cls = backend_factories[algo]
        step = self.current_step
        for param_group in param_groups:
            params = param_group.filter_params(self.model)
            backend_groups.append({
                'params': params,
                **param_group.schedules.get(step, aliases=self.hyperparameter_aliases)
            })
        self.groups_per_backend.append(param_groups)
        if algo == 'native':
            return cls(params=backend_groups, **self.backend_hyperparameters(), optimizer=self)
        else:
            return cls(params=backend_groups, **self.backend_hyperparameters())

    def start_step(self, batch: Batch) -> None:
        super().start_step(batch)
        self.schedule_backend()

    def schedule_backend(self):
        step = self.current_step
        for backend, param_groups in zip(self.backends, self.groups_per_backend):
            for backend_group, param_group in zip(backend.param_groups, param_groups):
                if schedule := param_group.schedules.get(step, include_constant=False):
                    backend_group.update(schedule)

    def trainable_parameter_list(self, group: int = None) -> list[Array]:
        return list(v for p, v in tree.flatten(self.trainable_parameters(group=group)))

    def auxloss_train_fn(self, train_fn: TrainFunction[Batch]) -> TrainFunction[Batch]:
        auxloss_instruments = self.get_auxloss_instruments()
        if auxloss_instruments:
            def auxloss_train_fn(batch: Batch) -> Array:
                loss = train_fn(batch)
                for aux_loss in auxloss_instruments:
                    loss += aux_loss.compute()
                return loss

            return auxloss_train_fn
        else:
            return train_fn

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

        train_fn = self.auxloss_train_fn(train_fn)

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

    @staticmethod
    def _backend_filename(path: Path, b: int) -> Path:
        return path.with_name(path.stem + f'-{b}').with_suffix('.pt')

    def _load(self, path: Path, **kwargs) -> None:
        for b, backend in enumerate(self.backends):
            p = self._backend_filename(path, b)
            state = torch.load(p, weights_only=False)
            backend.load_state_dict(state)

    def _save(self, path: Path, **kwargs) -> None:
        for b, backend in enumerate(self.backends):
            state = backend.state_dict()
            p = self._backend_filename(path, b)
            torch.save(state, p)


# noinspection PyTypeChecker
backend_factories: dict[str, TorchBackendFactory] = {
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


def add_args(opt: dict[str, Any], **kwargs) -> dict[str, Any]:
    for k, v in kwargs.items():
        if v is not None:
            opt.setdefault(k, v)
    return opt
