#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from tensile.common import *
from tensile.nn.losses import cross_entropy, LossFunction
from tensile.nn.module import Module, ForwardContext
from tensile.nn.instrument import Call, Instrument
from tensile.train import TrainingContext
from tensile.optim import Optimizer


@provides(Instrument, 'extra.head_precondition')
class HeadPreconditionInstrument(Instrument):

    __slots__ = ('loss_fn', 'optimizer')

    loss_fn: Annotated[LossFunction, field(
        doc="The loss function to use",
        default=coerce(LossFunction, kind='cross_entropy'),
    )]
    optimizer: Annotated[dict[str, Any], field(
        doc="The optimizer specs to use.",
        default_factory=dict,
    )]

    def build_optimizer(self, module: Module) -> Optimizer:
        return Optimizer.coerce(self.optimizer, model=module)

    def instrument(self, module: Module, call: Call, mode: Module.Mode) -> Call:

        if not mode.is_train():
            return call

        loss_fn = self.loss_fn

        def train_fn(batch):
            inputs, targets = batch
            logits = call(inputs)
            loss = loss_fn(logits, targets)
            return loss

        optim = self.build_optimizer(module)

        step = optim.stepper(train_fn)

        def two_phase_call(h):
            ctx = TrainingContext.get_current()
            if batch := ctx.current_batch_data:
                _, targets = batch

                # Phase 1: detached inner update
                h_detached = ten.stop_gradient(h)

                step((h_detached, targets))

            # Phase 2: forward attached hiddens through updated head
            return call(h)

        return two_phase_call
