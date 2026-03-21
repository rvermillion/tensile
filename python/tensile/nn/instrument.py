#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from typing import final

from .common import *

if TYPE_CHECKING:
    import tensile.nn.module


Call = TypeVar('Call', bound=Callable)


class InstrumentFunction(Protocol[Call]):

    def __call__(self, module: 'tensile.nn.module.Module', call: Call, mode: 'tensile.nn.module.CallMode') -> Call: ...


# noinspection PyUnusedLocal
def identity_wrapper(module: 'tensile.nn.module.Module', call: Call, mode: 'tensile.nn.module.CallMode') -> Call:
    return call


InstrumentLike = Union['Instrument', Sequence['InstrumentLike'], InstrumentFunction]


class Instrument(Object):

    __slots__ = ('path', 'subinstruments',)

    path: Annotated[str, field(
        default='',
    )]
    subinstruments: Annotated[dict[str, 'Instrument'], field(
        doc='A dictionary of subinstruments',
        default_factory=dict,
    )]

    def _coerce_subinstruments(self, spec: Any) -> dict[str, 'Instrument']:
        if spec is None: return {}
        if isinstance(spec, Mapping):
            return {name: Instrument.coerce(inst) for name, inst in spec.items()}
        raise TypeError('Cannot coerce to dict[str, Instrument]: {spec!r}')

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def instrument(self, module: 'tensile.nn.module.Module', call: Call, mode: 'tensile.nn.module.CallMode') -> Call:
        return call

    def subinstrument(self, name: str, module: 'tensile.nn.module.Module', call: Call, mode: 'tensile.nn.module.CallMode') -> Call:
        subinstrument = self.get_subinstrument(name)
        return call if subinstrument is None else subinstrument.instrument(module, call, mode)

    def get_subinstrument(self, name: str) -> Optional['Instrument']:
        return self.subinstruments.get(name)

    def add_subinstrument(self, name: str, subinstrument: InstrumentLike) -> None:
        subinstrument = Instrument.coerce(subinstrument)
        if name in self.subinstruments:
            self.subinstruments[name] += subinstrument
        else:
            self.subinstruments[name] = subinstrument

    def add_subinstruments(self, **subinstruments: InstrumentLike) -> None:
        for name, subinst in subinstruments.items():
            self.add_subinstrument(name, subinst)

    def remove_subinstrument(self, name: str, where: Predicate['Instrument'] = None) -> Optional['Instrument']:
        if name in self.subinstruments:
            removed = self.subinstruments[name].remove(where)
            if removed is None:
                del self.subinstruments[name]
            else:
                self.subinstruments[name] = removed
            return removed
        return None

    def __eq__(self, other) -> bool:
        if isinstance(other, Instrument):
            return self._eq_tuple() == other._eq_tuple()
        return False

    def __hash__(self):
        return hash(self._eq_tuple())

    @final
    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        return 1

    def _eq_tuple(self) -> tuple:
        return self.__class__, self.instrument

    def __contains__(self, item):
        return item == self

    def __add__(self, other: InstrumentLike | None) -> 'Instrument':
        if other is None: return self
        return Instrument.compose(self, Instrument.coerce(other))

    def __radd__(self, other: InstrumentLike | None) -> 'Instrument':
        if other is None: return self
        return Instrument.compose(Instrument.coerce(other), self)

    def __sub__(self, other: Union['Instrument', type, Predicate['Instrument']]) -> Optional['Instrument']:
        if isinstance(other, CompositeInstrument):
            return self.remove(predicates.is_in(set(other.instruments)))
        if isinstance(other, Instrument):
            return self.remove(predicates.eq(other))
        if isinstance(other, type):
            return self.remove(predicates.is_instance(other))
        if isinstance(other, Predicate):
            return self.remove(other)
        return self

    def remove(self, where: Predicate['Instrument'] = None) -> Optional['Instrument']:
        if where is None or where(self): return None
        return self

    def find(self, where: Predicate['Instrument'] = None) -> Iterable['Instrument']:
        if where is None or where(self): return self,
        return ()

    @classmethod
    def compose(cls, a: Optional[InstrumentLike], b: Optional[InstrumentLike]) -> 'Instrument':
        if a is None: return Instrument.coerce(b)
        if b is None: return Instrument.coerce(a)
        a = Instrument.coerce(a)
        b = Instrument.coerce(b)
        ai = a.instruments if isinstance(a, CompositeInstrument) else [a]
        bi = b.instruments if isinstance(b, CompositeInstrument) else [b]
        return CompositeInstrument(instruments=tuple(ai+bi))

    @classmethod
    def compose_all(cls, specs: Iterable[InstrumentLike]) -> 'Instrument':
        instruments = tuple(Instrument.coerce(i) for i in specs if i is not None)
        if instruments:
            if len(instruments) == 1:
                return instruments[0]
            return CompositeInstrument(instruments=instruments)
        raise ValueError(f"Expected a sequence of instruments: {list(specs)}")

    @classmethod
    def _coerce_from_callable(cls, spec: Callable, /, **kwargs):
        return cls.custom(spec)

    @classmethod
    def _coerce_from_sequence(cls, spec: Sequence, /, **kwargs):
        return cls.compose_all(spec)

    @classmethod
    def custom(cls, instrument: InstrumentFunction, **subinstruments: InstrumentLike) -> 'CustomInstrument':
        return CustomInstrument(instrument=instrument, subinstruments={
            name: Instrument.coerce(subinst) for name, subinst in subinstruments.items()
        })

    Function = InstrumentFunction
    Like = InstrumentLike


class CustomInstrument(Instrument):

    __slots__ = ('instrument',)

    instrument: Annotated[InstrumentFunction, field(
        doc='The instrument function to wrap the module call function',
        required=True,
    )]

    def _repr_args(self, **options) -> str:
        return self.instrument.__name__

    def _eq_tuple(self) -> tuple:
        return self.instrument,


class CompositeInstrument(Instrument):

    __slots__ = ('instruments',)

    instruments: Annotated[tuple[Instrument, ...], field(
        doc='List of instruments to apply',
        required=True,
    )]

    def instrument(self, module: 'tensile.nn.module.Module', call: Call, mode: 'tensile.nn.module.CallMode') -> Call:
        wrapped = call
        for instrument in self.instruments:
            wrapped = instrument.instrument(module, wrapped, mode)
        return wrapped

    def subinstrument(self, name: str, module: 'tensile.nn.module.Module', call: Call, mode: 'tensile.nn.module.CallMode') -> Call:
        wrapped = call
        for instrument in self.instruments:
            wrapped = instrument.subinstrument(name, module, wrapped, mode)
        wrapped = super().subinstrument(name, module, wrapped, mode)
        return wrapped

    def get_subinstrument(self, name: str) -> Optional['Instrument']:
        subs = [inst.get_subinstrument(name) for inst in self.instruments]
        subs.append(self.subinstruments.get(name))
        subs = tuple(inst for inst in subs if inst is not None)
        if subs:
            if len(subs) == 1: return subs[0]
            return CompositeInstrument(instruments=subs)
        return None

    def remove_subinstrument(self, name: str, where: Predicate['Instrument'] = None) -> Optional['Instrument']:
        removed = super().remove_subinstrument(name, where)
        for inst in self.instruments:
            inst.remove_subinstrument(name, where)
        return removed

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

    def find(self, where: Predicate['Instrument'] = None) -> Iterable['Instrument']:
        if where is None or where(self): yield self
        for inst in self.instruments:
            yield from inst.find(where)

    def _eq_tuple(self) -> tuple:
        return self.instruments

    def __len__(self) -> int:
        return len(self.instruments)

    def __contains__(self, item):
        return item == self or any(item in inst for inst in self.instruments)


__all__ = [
    'Instrument',
    'InstrumentFunction',
    'InstrumentLike',
    'Call',
    'CustomInstrument',
]