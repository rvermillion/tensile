#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from tensile.common import *

from tensile.optim.optimizer import Optimizer, OptimizerParamGroup
from tensile.optim.schedule import call_every
from tensile.optim.types import *


ModelUpdater = Callable[[Module, tree.Tree[Array]], Array]

EvalFunction = Callable[[Module, Array], Array]


LossAndGradFunction = Callable[[Batch], tuple[Array, Tree[Array]]]


class MLXBackend(Protocol):

    @property
    def state(self) -> Any: ...

    @state.setter
    def state(self, state: dict) -> None: ...

    def update(self, model: Module, gradients: dict) -> None: ...

    def apply_gradients(self, gradients: dict, parameters: dict) -> Any: ...


MLXBackendFactory = Callable[..., MLXBackend]


backend_aliases = {}


@provides(Optimizer, 'mlx')
class MLXOptimizer(Optimizer[Batch]):

    __slots__ = ()

    backends: Annotated[list[MLXBackend], field(
        doc='The backend for this optimizer.'
    )]

    def _lazy_backends(self) -> list[MLXBackend]:
        return [self.build_backend(param_group) for param_group in self.param_groups]

    def build_backend(self, param_group: OptimizerParamGroup) -> MLXBackend:
        config = param_group.config
        algo = config.algorithm
        factory = backend_factories[algo]
        aliases = backend_aliases.get(algo, {})
        schedules = config.schedules.alias(include_constant=True, aliases=aliases)
        if algo == 'native':
            return factory(**schedules, **config.backend_hyperparameters(aliases), optimizer=self)
        else:
            return factory(**schedules, **config.backend_hyperparameters(aliases))

    def loss_and_grad_fn(self, model: Module, train_fn: TrainFunction[Batch]) -> LossAndGradFunction[Batch]:

        aux_loss_instruments = self.get_auxloss_instruments(model)
        if aux_loss_instruments:
            def inner_fn(params, batch: Batch) -> Array:
                model.update(params)
                loss = train_fn(batch)
                for aux_loss in aux_loss_instruments:
                    loss += aux_loss.compute()
                return loss
        else:
            def inner_fn(params, batch: Batch) -> Array:
                model.update(params)
                return train_fn(batch)

        value_grad_fn = mx.value_and_grad(inner_fn)

        def wrapped_value_grad_fn(batch: Batch) -> tuple[Array, Any]:
            params = self.trainable_parameters(model)
            value, grad = value_grad_fn(params, batch)
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

        optimizers: list[MLXBackend] = self.backends

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

            ten.eval(loss, grads)

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

    def set_current_step(self, step: Array) -> None:
        super().set_current_step(step)
        for backend in self.backends:
            backend.state['step'] = self.current_step

    def _load(self, path: Path, **kwargs) -> None:
        for b, backend in enumerate(self.backends):
            p = path.with_name(path.stem + f'-{b}').with_suffix('.safetensors')
            flat = ten.load_tensors(p)
            state = tree.unflatten(flat.items())
            backend.state = state

    def _save(self, path: Path, **kwargs) -> None:
        for b, backend in enumerate(self.backends):
            state = tree.flatdict(backend.state)
            p = path.with_name(path.stem + f'-{b}').with_suffix('.safetensors')
            ten.save_tensors(p, state)



backend_factories: dict[str, MLXBackendFactory] = {
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

