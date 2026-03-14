#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *

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

    def __eq__(self, other) -> bool:
        if isinstance(other, Instrument):
            return self._eq_tuple() == other._eq_tuple()
        return False

    def __hash__(self):
        return hash(self._eq_tuple())

    def _eq_tuple(self) -> tuple:
        return self.__class__, self.wrap_call

    def __contains__(self, item):
        return item == self

    def __add__(self, other) -> 'Instrument':
        if other is None: return self
        return Instrument.compose(self, Instrument.coerce(other))

    def __radd__(self, other) -> 'Instrument':
        if other is None: return self
        return Instrument.compose(Instrument.coerce(other), self)

    def __sub__(self, other: Union['Instrument', type]) -> Optional['Instrument']:
        if isinstance(other, Instrument):
            return self.remove(predicates.eq(other))
        if isinstance(other, type):
            return self.remove(predicates.is_instance(other))
        return self

    def remove(self, where: Predicate['Instrument'] = None) -> Optional['Instrument']:
        if where is None or where(self): return None
        return self

    @classmethod
    def compose(cls, a: Optional['Instrument'], b: Optional['Instrument']) -> 'Instrument':
        if a is None: return b
        if b is None: return a
        ai = a.instruments if isinstance(a, CompositeInstrument) else [a]
        bi = b.instruments if isinstance(b, CompositeInstrument) else [b]
        return CompositeInstrument(instruments=tuple(ai+bi))

    @classmethod
    def _coerce_from_callable(cls, spec: Callable, /, **kwargs):
        return cls.from_wrapper(spec)

    @classmethod
    def _coerce_from_sequence(cls, spec: Sequence, /, **kwargs):
        instruments = tuple(Instrument.coerce(i) for i in spec if i is not None)
        if instruments:
            if len(instruments) == 1:
                return instruments[0]
            return CompositeInstrument(instruments=instruments)
        raise ValueError(f"Expected a sequence of instruments: {spec}")

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

    def _eq_tuple(self) -> tuple:
        return self.wrap_call,


class CompositeInstrument(Instrument):

    __slots__ = ('instruments',)

    instruments: Annotated[tuple[Instrument, ...], field(
        doc='List of instruments to apply',
        required=True,
    )]

    def wrap_call(self, module: 'tensile.nn.module.Module', call: C, training: bool) -> C:
        wrapped = call
        for instrument in self.instruments:
            wrapped = instrument.wrap_call(module, wrapped, training)
        return wrapped

    def remove(self, where: Predicate['Instrument'] = None) -> Optional['Instrument']:
        if where is None or where(self): return None
        instruments = [inst.remove(where) for inst in self.instruments]
        if instruments == self.instruments:
            return self
        instruments = tuple(inst for inst in instruments if inst is not None)
        if instruments:
            if len(instruments) == 1: return instruments[0]
            return CompositeInstrument(instruments=instruments)
        return None

    def _eq_tuple(self) -> tuple:
        return self.instruments

    def __contains__(self, item):
        return item == self or any(item in inst for inst in self.instruments)

