#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..module import Module
from ..instrument import Instrument, Call
from ..layers import Dropout


@provides(Instrument, 'dropout')
class DropoutInstrument(Instrument):

    __slots__ = ('dropout', 'on_input')

    dropout: Annotated[Dropout, field(
        doc='The dropout layer to apply during training',
        coerce=True,
    )]
    on_input: Annotated[bool, field(
        default=False,
    )]

    def instrument(self, module: Module, call: Call, mode: Module.Mode) -> Call:
        if mode.is_train():
            dropout = self.dropout

            if self.on_input:
                def call_with_dropout(x: Array, *args, **kwargs) -> Array:
                    x_dropped = dropout(x)
                    return call(x_dropped, *args, **kwargs)
            else:
                def call_with_dropout(*args, **kwargs) -> Array:
                    y = call(*args, **kwargs)
                    return dropout(y)

            return call_with_dropout
        return call
