#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


from ..common import *
from ..module import CallMode, Module
from ..instrument import Call, Instrument


@provides(Instrument, 'loss')
class AuxLossInstrument(Instrument):

    __slots__ = ()

    def pre_forward(self):
        pass

    def compute(self) -> Array:
        raise NotImplementedError(self)

    def post_backward(self):
        pass


@ten.compile(shapeless=True)
def mean_square(x: Array) -> Array:
    return ten.mean(ten.square(x))


@provides(Instrument, 'loss.activation')
@provides(AuxLossInstrument, 'activation')
class ActivationLossInstrument(AuxLossInstrument):

    __slots__ = ('is_pre', 'loss_fn', 'activation')

    loss_fn: Annotated[Callable[[Array], Array], field(
        default=mean_square,
    )]
    activation: Optional[Array]
    is_pre: Annotated[bool, field(
        default=False,
    )]

    def instrument(self, module: Module, call: Call, mode: CallMode) -> Call:
        if mode.is_train():
            if self.is_pre:
                def wrapped(x: Array, *args, **kwargs):
                    self.activation = x
                    return call(x, *args, **kwargs)
            else:
                def wrapped(*args, **kwargs):
                    y = call(*args, **kwargs)
                    self.activation = y
                    return y
            return wrapped
        return call

    def compute(self) -> Array:
        activation = self.activation
        self.activation = None
        return ten.array(0.) if activation is None else self.loss_fn(activation)


@provides(Instrument, 'loss.parameter')
@provides(AuxLossInstrument, 'parameter')
class ParameterLossInstrument(AuxLossInstrument):

    __slots__ = ('module', 'loss_fn', 'parameters')

    module: Annotated[Optional[Module], field()]
    loss_fn: Annotated[Callable[[Array], Array], field(
        default=mean_square,
    )]
    parameters: Annotated[list[str], field(
        default_factory=list,
    )]

    def instrument(self, module: Module, call: Call, mode: CallMode) -> Call:
        self.module = module
        return call

    def current_parameters(self) -> Iterable[Array]:
        module = self.module
        if module is not None:
            for path in self.parameters:
                param = module.get_child(path)
                if isinstance(param, Array):
                    yield param
                else:
                    self.warn('Parameter {} in {} is not a Array', path, module)

    def compute(self) -> Array:
        loss = ten.array(0.)
        for param in self.current_parameters():
            loss += self.loss_fn(param)
        return loss

