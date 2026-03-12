#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import mlx.core as mx
import mlx.optimizers as optim

from tensile.nn import Module
from tensile.nn.common import *

from tensile.optim.optimizer import (
    Optimizer, OptimizerParamGroup, BaseSGDOptimizer,
    BaseAdamWOptimizer,
)
from tensile.optim.types import GradientHandler, OptimizerStep, TrainFunction, Batch


ModelUpdater = Callable[[Module, tree.Tree[Array]], Array]

EvalFunction = Callable[[Module, Array], Array]


LossAndGradFunction = Callable[[Batch], tuple[Array, Tree[Array]]]


class MLXOptimizer(Optimizer[Batch]):

    __slots__ = ()

    backend: Annotated[list[optim.Optimizer], field(
        doc='The backend for this optimizer.'
    )]

    def _lazy_backend(self) -> list[optim.Optimizer]:
        return [self.build_backend(param_group) for param_group in self.param_groups]

    def build_backend(self, param_group: OptimizerParamGroup) -> optim.Optimizer:
        raise TypeError(f'{self.__class__.__name__} must implement _lazy_backend')

    def loss_and_grad_fn(self, model: Module, train_fn: TrainFunction[Batch]) -> LossAndGradFunction[Batch]:

        def inner_fn(params, batch: Batch) -> Array:
            # print('inner model.embed_tokens.weight:', id(params['model']['embed_tokens']['weight']))
            model.update(params)
            return train_fn(model, batch)

        value_grad_fn = mx.value_and_grad(inner_fn)

        # @wraps(fn)
        def wrapped_value_grad_fn(batch: Batch) -> tuple[Array, Any]:
            params = self.trainable_parameters(model)
            # print('wrapped model.embed_tokens.weight:', id(params['model']['embed_tokens']['weight']))
            value, grad = value_grad_fn(params, batch)
            # ten.debug_eval(value, grad)
            return value, grad

        return wrapped_value_grad_fn


    def stepper(self, model: Module, train_fn: TrainFunction[Batch],
                grad_handlers: Sequence[GradientHandler] = None,
                update: ModelUpdater = None, microbatch_size: int = None) -> OptimizerStep[Batch]:

        loss_and_grad_fn: LossAndGradFunction[Batch] = self.loss_and_grad_fn(model, train_fn)

        if len(self.param_groups) > 1:
            raise NotImplementedError("Multi-parameter group optimizers are not supported yet")

        optimizers: Iterable[optim.Optimizer] = self.backend
        eval_every = 1

        def eval_params():
            states = [opt.state for opt in optimizers]
            ten.eval(model.parameters(), *states)
            return True

        # if eval_every > 1:
        #     eval_params = Trainer.call_every(eval_every, eval_params)

        def update(mod: Module, grads: tree.Tree[Array]):
            for optimizer, param_group in zip(optimizers, self.param_groups):
                # pick out just the right grads for each optimizer
                optimizer.update(mod, param_group.filter_tree(grads))
            return True

        def step(batch: Batch) -> Array:
            # x, y = batch.data
            # ten.debug_eval(batch.data)
            self.start_step(batch)

            loss, grads = loss_and_grad_fn(batch)

            # ten.debug_eval(loss, grads)

            if grad_handlers:
                flat_grads = tree.flatten(grads)
                for grad_handler in grad_handlers:
                    grad_handler(self, flat_grads)

            # Update the model with the gradients. So far no computation has happened.
            update(model, grads)

            eval_params()

            self.finish_step(loss, batch)
            return loss

        return step


@provides(Optimizer, 'sgd')
class SGDOptimizer(BaseSGDOptimizer, MLXOptimizer):

    __slots__ = ()

    def build_backend(self, param_group: OptimizerParamGroup) -> optim.Optimizer:
        schedules = param_group.schedules.alias(include_constant=True, aliases=self.aliases)
        return optim.SGD(**schedules, **self.backend_spec())


@provides(Optimizer, 'adamw')
class AdamWOptimizer(BaseAdamWOptimizer, MLXOptimizer):

    __slots__ = ()

    def build_backend(self, param_group: OptimizerParamGroup) -> optim.Optimizer:
        schedules = param_group.schedules.alias(include_constant=True, aliases=self.aliases)
        return optim.AdamW(**schedules, **self.backend_spec())
        #     learning_rate=self.learning_rate,
        #     weight_decay=self.weight_decay,
        #     betas=self.betas,
        #     eps=self.eps,
        # )
