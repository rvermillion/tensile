#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ...shims import ten

from ...infrastructure import Object, field
from ...infrastructure.types import Annotated, Callable, Optional, Protocol, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import tensile.nn.module


C = TypeVar('C', bound=Callable)


class ModuleCallWrapper(Protocol[C]):

    def __call__(self, module: 'tensile.nn.module.Module', call: C, training: bool) -> C: ...


# noinspection PyUnusedLocal
def identity_wrapper(module: 'tensile.nn.module.Module', call: C, training: bool) -> C:
    return call



class Instrument(Object):

    __slots__ = ()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def wrap_call(self, module: 'tensile.nn.module.Module', call: C, training: bool) -> C:
        return call

    @classmethod
    def compose(cls, a: Optional['Instrument'], b: Optional['Instrument']) -> 'Instrument':
        if a is None: return b
        if b is None: return a
        ai = a.instruments if isinstance(a, CompositeInstrument) else [a]
        bi = b.instruments if isinstance(b, CompositeInstrument) else [b]
        return CompositeInstrument(instruments=ai+bi)

    @classmethod
    def _coerce_from_callable(cls, spec: Callable, /, **kwargs):
        return cls.from_wrapper(spec)

    @classmethod
    def from_wrapper(cls, wrap_call: ModuleCallWrapper):
        return WrapperInstrument(wrap_call=wrap_call)


class WrapperInstrument(Instrument):

    __slots__ = ('wrap_call',)

    wrap_call: Annotated[ModuleCallWrapper, field(
        doc='Call wrapper function',
        required=True,
    )]

    def _repr_args(self, **options) -> str:
        return self.wrap_call.__name__


class CompositeInstrument(Instrument):

    __slots__ = ('instruments',)

    instruments: Annotated[list[Instrument], field(
        doc='List of instruments to apply',
        required=True,
    )]

    def wrap_call(self, module: 'tensile.nn.module.Module', call: C, training: bool) -> C:
        wrapped = call
        for instrument in self.instruments:
            wrapped = instrument.wrap_call(module, wrapped, training)
        return wrapped

