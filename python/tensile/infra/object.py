#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
import enum
from pathlib import Path
from typing import final

from .field import field
from .load import try_load
from .meta import ObjectMeta, Spec, UpdateableObject, private_slot
from .types import Annotated, Any, Callable, ClassVar, Keywords, Mapping, Optional, Self, Sequence, TypeVar
from .util import process_specs


# noinspection PyUnusedLocal
def noop_init(self, spec: Keywords = None, /, **kwargs):
    pass


T = TypeVar('T', bound='Object')


class Lifecycle(enum.Enum):
    unknown = 0
    preinit = 1
    init = 2
    postinit = 3
    ready = 4
    error = 100

    def is_ready(self) -> bool:
        return self is Lifecycle.ready

    def is_error(self) -> bool:
        return self is Lifecycle.error


class ObjectClass(type):

    # noinspection PyPep8Naming
    def __new__(metacls, name: str, bases: tuple[type, ...], spec: dict[str, Any], /, interface: bool = False, **kwds):

        if slots := spec.get('__slots__', ()):
            slots = tuple(private_slot(slot) for slot in slots)
            spec['__slots__'] = slots

        cls = super().__new__(metacls, name, bases, spec, **kwds)

        Meta = getattr(cls, 'Meta', ObjectMeta)

        Meta.engineer(cls, interface=interface, **kwds)

        return cls



class Object(UpdateableObject, metaclass=ObjectClass):

    __slots__ = ('_spec',)

    _spec: Annotated[Spec, field(
        doc='The original spec used to create this object',
        init=False,
    )]
    meta: ClassVar[Annotated[ObjectMeta, field(
        doc='The meta object for this class.'
    )]]

    @final
    def __init__(self, spec: Keywords = None, /, **kwargs):
        spec = Spec.combine(spec, kwargs)
        self._spec = spec
        try:
            self.set_lifecycle(Lifecycle.preinit)
            self.preinit(spec)
            self.set_lifecycle(Lifecycle.init)
            self.init(spec)
            self.set_lifecycle(Lifecycle.postinit)
            self.postinit(spec)
            self.set_lifecycle(Lifecycle.ready)
        except Exception as e:
            self.warn('Failed to initialize {} in {}: {!r}', type(self), self.get_lifecycle(), e)
            self.set_lifecycle(Lifecycle.error)
            raise e

    def set_lifecycle(self, lifecycle: Lifecycle):
        pass

    def get_lifecycle(self) -> Lifecycle:
        return Lifecycle.unknown

    def preinit(self, spec: Spec):
        pass

    def init(self, spec: Spec):
        self.meta.init(self, spec)

    def postinit(self, spec: Spec):
        pass

    def _update_from_spec(self, spec: Spec, update: bool = False):
        if update:
            self.meta.update_instance(self, spec)
        else:
            for key, val in spec.items():
                self.set_field(key, val, update)

    def peek_field(self, key: str, default: Any) -> Any:
        val = self.meta.fields[key].peek(self)
        return default if val is None else val

    def get_field(self, key: str, default: Any) -> Any:
        try:
            return getattr(self, key, default)
        except AttributeError:
            return default

    def poke_field(self, key: str, value: Any) -> None:
        self.meta.fields[key].poke(self, value)

    def set_field(self, key: str, value: Any, update: bool = False):
        setattr(self, key, value)

    def cast(self, cls: type[T]) -> Optional[T]:
        return self if isinstance(self, cls) else None

    @classmethod
    def from_dict(cls, spec: Keywords) -> Self:
        return cls.coerce(spec)

    @classmethod
    def create(cls, spec: Keywords = None, /, **kwargs) -> Self:
        return cls(spec, **kwargs)

    @classmethod
    def load_from(cls, path: Path|str, **kwargs) -> Self:
        if isinstance(path, str):
            path = Path(path)
        spec = try_load(path)
        if spec is None:
            raise ValueError(f'Could not load from: {path}')
        return cls.coerce(spec, **kwargs)

    @classmethod
    def coerce(cls, spec: Any = None, /, **kwargs) -> Self:
        if isinstance(spec, cls):
            return spec
        if spec is None:
            if not kwargs:
                return cls._coerce_from_none()

        if spec is None or isinstance(spec, Mapping):
            return cls._coerce_from_mapping(spec, **kwargs)
        elif isinstance(spec, str):
            return cls._coerce_from_str(spec, **kwargs)
        elif isinstance(spec, Sequence):
            return cls._coerce_from_sequence(spec, **kwargs)
        elif isinstance(spec, type):
            return cls._coerce_from_type(spec, **kwargs)
        elif callable(spec):
            return cls._coerce_from_callable(spec, **kwargs)
        raise ValueError(f'Cannot coerce {spec!r} to {cls}')

    auto_coerce: ClassVar[Annotated[bool, field(ignore=True)]] = False

    @classmethod
    def _coerce_from_none(cls):
        if factory := cls.meta.get_factory(from_type='none'):
            return factory()
        raise TypeError(f'Cannot coerce None to {cls}')

    @classmethod
    def _coerce_from_str(cls, spec: str, /, **kwargs):
        if factory := cls.meta.get_factory(from_type='str'):
            return factory(spec, **kwargs)
        raise TypeError(f'Cannot coerce string to {cls}')

    @classmethod
    def _coerce_from_mapping(cls, spec: Mapping[str, Any], /, **kwargs):
        kind, spec = process_specs(spec, kwargs)
        if kind is None:
            return cls.create(spec)
        if factory := cls.meta.get_factory(kind=kind):
            return factory(spec, **kwargs)
        raise ValueError(f'Cannot coerce {spec} to kind [{kind}] of {cls}')

    _single_key_kind: ClassVar[Annotated[bool, field(ignore=True)]] = False

    @classmethod
    def _coerce_from_sequence(cls, spec: Sequence, /, **kwargs):
        if factory := cls.meta.get_factory(from_type='sequence'):
            return factory(kwargs, sequence=spec)
        if factory := cls.meta.get_factory(kind='sequence'):
            return factory(kwargs, sequence=spec)
        raise TypeError(f'Cannot coerce a sequence {spec} to {cls}')

    @classmethod
    def _coerce_from_type(cls, spec: type, /, **kwargs):
        if factory := cls.meta.get_factory(from_type='type'):
            return factory(spec, **kwargs)
        raise TypeError(f'Cannot coerce a type {spec} to {cls}')

    @classmethod
    def _coerce_from_callable(cls, spec: Callable, /, **kwargs):
        if factory := cls.meta.get_factory(from_type='callable'):
            return factory(spec, **kwargs)
        raise TypeError(f'Cannot coerce a function {spec} to {cls}')

    Meta: ClassVar[type[ObjectMeta]] = ObjectMeta
    Lifecycle: ClassVar[type[Lifecycle]] = Lifecycle


__all__ = [
    'Object',
    'ObjectClass',
]