#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import mlx.core as mx
import mlx.optimizers as optim

from tensile.nn.common import *

from tensile.optim.optimizer import Optimizer, OptimizerParamGroup
from tensile.optim.schedule import call_every
from tensile.optim.types import *


ModelUpdater = Callable[[Module, tree.Tree[Array]], Array]

EvalFunction = Callable[[Module, Array], Array]


LossAndGradFunction = Callable[[Batch], tuple[Array, Tree[Array]]]


backend_classes = {
    'adadelta': optim.AdaDelta,
    'adafactor': optim.Adafactor,
    'adamax': optim.Adamax,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'lion': optim.Lion,
    'muon': optim.Muon,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD,
}

backend_aliases = {}


@provides(Optimizer, 'mlx')
class MLXOptimizer(Optimizer[Batch]):

    __slots__ = ()

    backends: Annotated[list[optim.Optimizer], field(
        doc='The backend for this optimizer.'
    )]

    def _lazy_backends(self) -> list[optim.Optimizer]:
        return [self.build_backend(param_group) for param_group in self.param_groups]

    def build_backend(self, param_group: OptimizerParamGroup) -> optim.Optimizer:
        config = param_group.config
        algo = config.algorithm
        cls = backend_classes[algo]
        aliases = backend_aliases.get(algo, {})
        schedules = config.schedules.alias(include_constant=True, aliases=aliases)
        return cls(**schedules, **config.backend_hyperparameters(aliases))

    def loss_and_grad_fn(self, model: Module, train_fn: TrainFunction[Batch]) -> LossAndGradFunction[Batch]:

        def inner_fn(params, batch: Batch) -> Array:
            # print('inner model.embed_tokens.weight:', id(params['model']['embed_tokens']['weight']))
            model.update(params)
            return train_fn(batch)

        value_grad_fn = mx.value_and_grad(inner_fn)

        # @wraps(fn)
        def wrapped_value_grad_fn(batch: Batch) -> tuple[Array, Any]:
            params = self.trainable_parameters(model)
            # print('wrapped model.embed_tokens.weight:', id(params['model']['embed_tokens']['weight']))
            value, grad = value_grad_fn(params, batch)
            # ten.debug_eval(value, grad)
            return value, grad

        return wrapped_value_grad_fn


    def stepper(self, train_fn: TrainFunction[Batch], *,
                grad_handlers: Sequence[GradientHandler] = None,
                start_step: OptimizerStartStep = None,
                end_step: OptimizerEndStep = None,
                eval_every: int = 1,
                microbatch_size: int = None,
                **kwargs) -> OptimizerStep[Batch]:

        model = self.model
        loss_and_grad_fn: LossAndGradFunction[Batch] = self.loss_and_grad_fn(model, train_fn)

        optimizers: list[optim.Optimizer] = self.backends

        if len(optimizers) == 1:
            optimizer = optimizers[0]
            param_group = self.param_groups[0]

            def eval_params():
                ten.eval(model.parameters(), optimizer.state)
                return True

            def update(grads: tree.Tree[Array]):
                optimizer.update(model, param_group.filter_tree(grads))
                return True

        else:
            def eval_params():
                states = [opt.state for opt in optimizers]
                ten.eval(model.parameters(), *states)
                return True

            def update(grads: tree.Tree[Array]):
                for opt, pg in zip(optimizers, self.param_groups):
                    # pick out just the right grads for each optimizer
                    group_params = pg.filter_tree(model)
                    group_grads = pg.filter_tree(grads)
                    # ten.debug_eval(group_grads, group_params)
                    model.update(opt.apply_gradients(group_grads, group_params))
                return True

        if eval_every > 1:
            eval_params = call_every(eval_every, eval_params)

        def step(batch: Batch) -> Array:
            # x, y = batch.data
            # ten.debug_eval(batch.data)
            self.start_step(batch)
            if start_step: start_step(self, batch)

            loss, grads = loss_and_grad_fn(batch)

            # ten.debug_eval(loss, grads)

            if grad_handlers:
                flat_grads = tree.flatten(grads)
                for grad_handler in grad_handlers:
                    grad_handler(self, flat_grads)

            # Update the model with the gradients. So far no computation has happened.
            update(grads)

            # Evaluate the parameters and optimizer state
            eval_params()

            if end_step: end_step(self, loss, batch)
            self.finish_step(loss, batch)
            return loss

        return step

