#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import torch.optim as optim

from tensile.nn import Module
from tensile.nn.common import *

from tensile.optim.types import GradientHandler, TrainFunction, Batch
from tensile.optim.optimizer import (Optimizer, OptimizerStep, BaseSGDOptimizer, BaseAdamWOptimizer)



class TorchOptimizer(Optimizer[Batch]):

    __slots__ = ()

    backend: Annotated[optim.Optimizer, field(
        doc='The backend for this optimizer.'
    )]

    aliases = {'learning_rate': 'lr'}

    def _lazy_backend(self) -> optim.Optimizer:
        raise TypeError(f'{self.__class__.__name__} must implement _lazy_backend')

    def start_step(self, batch: Batch) -> None:
        super().start_step(batch)
        self.schedule_backend()

    def schedule_backend(self):
        for g, param_group in enumerate(self.backend.param_groups):
            if schedule := self.current_schedule(g):
                param_group.update(schedule)

    @property
    def lr(self) -> float:
        lr = self.learning_rate
        if callable(lr):
            return lr(self.current_step).item()
        elif isinstance(lr, float):
            return lr
        return 0.

    def trainable_parameter_list(self, group: int = None) -> list[Array]:
        return list(e.value for e in tree.flatten(self.trainable_parameters(group=group)))


    def stepper(self, model: Module,
                train_fn: TrainFunction[Batch],
                grad_handlers: list[GradientHandler] = None) -> OptimizerStep[Batch]:

        optimizer = self.backend

        def step(batch: Batch) -> Array:
            # x, y = batch.data
            self.start_step(batch)

            optimizer.zero_grad()

            loss = train_fn(model, batch)

            ten.get_active_memory()

            loss.backward()

            if grad_handlers:
                flat_grads = [e for e in tree.flatten(self.trainable_parameters()) if e[1].grad is not None]
                for grad_handler in grad_handlers:
                    grad_handler(self, flat_grads)

            ten.get_active_memory()

            optimizer.step()

            self.finish_step(ten.detach(loss), batch)
            return loss

        return step


@provides(Optimizer, 'sgd')
class SGDOptimizer(BaseSGDOptimizer, TorchOptimizer):

    __slots__ = ()

    # def schedules(self, param_group: int = 0) -> dict[str, OptimizerSchedule]:
    #     return add_to_schedules(
    #         lr=self.learning_rate,
    #         # momentum=self.momentum,
    #         # weight_decay=self.weight_decay,
    #         # dampening=self.dampening,
    #     )

    def _lazy_backend(self) -> optim.Optimizer:
        param_groups = []
        step = self.current_step
        for param_group in self.param_groups:
            params = param_group.filter_params(self.model)
            param_groups.append({
                'params': params,
                **param_group.schedules.get(step, aliases=self.aliases)
            })
        return optim.SGD(param_groups, **self.backend_spec())

        # return optim.SGD(params, **add_args(
        #     self.current_schedule(),
        #     lr=self.lr,
        #     momentum=self.momentum,
        # ))


@provides(Optimizer, 'adamw')
class AdamWOptimizer(BaseAdamWOptimizer, TorchOptimizer):

    __slots__ = ()

    # def schedules(self, param_group: int = 0) -> dict[str, OptimizerSchedule]:
    #     return add_to_schedules(
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay,
    #     )

    def _lazy_backend(self) -> optim.Optimizer:
        param_groups = []
        step = self.current_step
        for param_group in self.param_groups:
            params = param_group.filter_params(self.model)
            param_groups.append({
                'params': params,
                **param_group.schedules.get(step, aliases=self.aliases)
            })
        return optim.AdamW(param_groups, **self.backend_spec())
        # params = self.param_groups[0].filter_params(self.model)
        # return optim.AdamW(params, **self.backend_spec())
        # return optim.AdamW(params, **add_args(
        #     self.current_schedule(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     betas=self.betas,
        #     eps=self.eps,
        # ))


def add_args(opt: dict[str, Any], **kwargs) -> dict[str, Any]:
    for k, v in kwargs.items():
        if v is not None:
            opt.setdefault(k, v)
    return opt
