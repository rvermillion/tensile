#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from tensile.nn.losses import cross_entropy, LossFunction
from tensile.nn.module import Module, ForwardContext
from tensile.nn.module.instrument import C, Instrument
from tensile.nn.common import *
from tensile.optim import Optimizer


@provides(Instrument, 'extra.head_precondition')
class HeadPreconditionInstrument(Instrument):

    __slots__ = ('loss_fn', 'optimizer_specs')

    loss_fn: Annotated[LossFunction, field(
        doc="The loss function to use",
        default=cross_entropy
    )]
    optimizer_specs: Annotated[dict[str, Any], field(
        doc="The optimizer specs to use.",
        default_factory=dict,
    )]

    def build_optimizer(self, module: Module) -> Optimizer:
        spec = self.optimizer_specs.copy()
        spec['model'] = module
        return Optimizer.coerce(spec)

    def wrap_call(self, module: Module, call: C, training: bool) -> C:

        if not training:
            return call

        loss_fn = self.loss_fn

        def train_fn(batch):
            inputs, targets = batch
            logits = call(inputs)
            return loss_fn(logits, targets)

        optim = self.build_optimizer(module)

        step = optim.stepper(train_fn)

        def two_phase_call(h):
            ctx = ForwardContext.get_current()
            if batch := ctx.get_param('batch'):
                _, targets = batch

                # Phase 1: detached inner update
                h_detached = h.detach()

                step((h_detached, targets))

            # Phase 2: forward attached hiddens through updated head
            return call(h)

        return two_phase_call
