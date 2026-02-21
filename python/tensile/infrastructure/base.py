#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
import json

from .meta import ObjectMeta, Meta, RootObject, Spec, UpdateableObject, field
from .types import Annotated, Any, Callable, ClassVar, Keywords, Mapping, Optional, Self, Sequence
from .util import process_specs


# class KindAware:
#
#     __slots__ = ()
#
#     def __init_subclass__(cls, interface: bool = False, **kwargs):
#         super().__init_subclass__(**kwargs)
#
#         if interface:
#             # print(f'made {cls} into an interface')
#             cls.kinds = {}
#
#         if 'kind' in cls.__dict__ and cls.kinds is not None:
#             cls.kinds[cls.kind] = cls.create
#
#     @classmethod
#     def create(cls, spec: Keywords, *rest, **kwargs) -> Self:
#         raise NotImplementedError()
#
#     kind: Optional[str]
#     kinds: ClassVar[dict[str, Factory['KindAware']]] = {}


# noinspection PyUnusedLocal
def noop_init(self, spec: Keywords, **kwargs):
    pass


class Object(UpdateableObject):

    __slots__ = ['spec']

    spec: Annotated[Spec, field(
        doc='The original spec used to create this object',
        init=None,
    )]
    meta: ClassVar[Annotated[ObjectMeta, field(
        doc='The meta object for this class.'
    )]]
    # init: ClassVar[Annotated[Callable, field(
    #     doc='The init method for this class.'
    # )]] = noop_init
    # kind: ClassVar[Annotated[Optional[str], field(
    #
    # )]] = None

    def __init__(self, spec: Keywords = None, /, **kwargs):
        spec = Spec.combine(spec, kwargs)
        self.spec = spec
        self.preinit(spec)
        self.init(spec)
        self.postinit(spec)

    def preinit(self, spec: Spec):
        pass

    def init(self, spec: Spec):
        self.meta.init(self, spec)

    def postinit(self, spec: Spec):
        pass

    # def _init_spec(self, spec: Spec) -> Spec:
    #     # if factories := self.default_factories:
    #     #     for key, factory in factories.items():
    #     #         if key not in spec or spec[key] is None:
    #     #             spec[key] = factory(self, spec)
    #     return spec

    def _update_from_spec(self, spec: Spec, update: bool = False):
        if update:
            self.meta.update_instance(self, spec)
        else:
            for key, val in spec.items():
                self.set(key, val, update)

    def peek(self, key: str, default: Any) -> Any:
        val = self.meta.fields[key].peek(self)
        return default if val is None else val

    def get(self, key: str, default: Any) -> Any:
        try:
            return getattr(self, key, default)
        except AttributeError:
            return default

    def poke(self, key: str, value: Any) -> None:
        self.meta.fields[key].poke(self, value)

    def set(self, key: str, value: Any, update: bool = False):
        setattr(self, key, value)

    def __init_subclass__(cls, interface: bool = None, **kwargs):
        super().__init_subclass__(**kwargs)

        cls.Meta.engineer(cls, interface=interface, **kwargs)

    # kind: Optional[str]
    # defaults: ClassVar[Optional[Keywords]] = None
    # default_factories: ClassVar[Optional[Mapping[str, Callable[['BaseObject', Keywords], Any]]]] = None
    # kinds: ClassVar[dict[str, Factory['BaseObject']]] = {}

    @classmethod
    def from_dict(cls, spec: Keywords) -> Self:
        return cls.coerce(spec)

    @classmethod
    def create(cls, spec: Keywords = None, *rest, **kwargs) -> Self:
        return cls(spec, **kwargs)

    @classmethod
    def coerce(cls, spec: Any = None, /, **kwargs) -> Self:
        if isinstance(spec, cls):
            return spec
        if spec is None:
            if not kwargs:
                return cls._coerce_from_none()

        # factory = None
        if spec is None or isinstance(spec, Mapping):
            # kind, spec = process_specs(spec, **kwargs)
            # if kind is None:
            #     return cls.create(spec)
            # factory = cls.meta.get_factory(kind=kind)
            return cls._coerce_from_mapping(spec, **kwargs)
        elif isinstance(spec, str):
            # factory = cls.meta.get_factory(from_type='str')
            return cls._coerce_from_str(spec, **kwargs)
        elif isinstance(spec, Sequence):
            # factory = cls.meta.get_factory(from_type='sequence')
            # if factory is None:
            #     factory = cls.meta.get_factory(from_type='sequence')
            return cls._coerce_from_sequence(spec, **kwargs)
        elif isinstance(spec, type):
            # factory = cls.meta.get_factory(from_type='type')
            return cls._coerce_from_type(spec, **kwargs)
        elif callable(spec):
            factory = cls.meta.get_factory(from_type='callable')
            return cls._coerce_from_callable(spec, **kwargs)

        # if factory:
        #     # noinspection PyCallingNonCallable
        #     return factory(spec, **kwargs)
        raise ValueError(f'Cannot coerce {spec} to {cls}')

    @classmethod
    def xcoerce(cls, spec: Any = None, /, **kwargs) -> Self:
        if isinstance(spec, cls):
            return spec
        if spec is None:
            if not kwargs:
                return cls._coerce_from_none()
        if spec is None or isinstance(spec, Mapping):
            return cls._coerce_from_mapping(spec, **kwargs)
        if isinstance(spec, str):
            return cls._coerce_from_str(spec, **kwargs)
        if isinstance(spec, Sequence):
            return cls._coerce_from_sequence(spec, **kwargs)
        if isinstance(spec, type):
            return cls._coerce_from_type(spec, **kwargs)
        if callable(spec):
            return cls._coerce_from_callable(spec, **kwargs)
        raise ValueError(f'Cannot coerce {spec} to {cls}')

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
        kind, spec = process_specs(spec, **kwargs)
        if kind is None:
            return cls.create(spec)
        if factory := cls.meta.get_factory(kind=kind):
            return factory(spec, **kwargs)
        raise ValueError(f'Cannot coerce {spec} to kind [{kind}] of {cls}')

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
        raise TypeError(f'Cannot coerce a function {spec} to {cls}')

    Meta: ClassVar[type[ObjectMeta]] = ObjectMeta


ObjectMeta.engineer(Object)

BaseObject = Object
# BaseObject.meta = Meta(BaseObject)


class Storable:

    __slots__ = ()

    def store(self, arrays: dict, metadata: dict, prefix: str = ''):
        self._store_arrays(arrays, prefix=prefix)
        self._store_metadata(metadata, prefix=prefix)

    def _store_arrays(self, arrays: dict, prefix: str = ''):
        pass

    def _store_metadata(self, metadata: dict, prefix: str = ''):
        md = self._metadata_to_store()
        if md:
            metadata[prefix + 'metadata'] = json.dumps(md)

    def _metadata_to_store(self) -> Optional[dict]:
        return None


class Loadable:

    __slots__ = ()

    def load(self, arrays: dict, metadata: dict, prefix: str = ''):
        raise NotImplementedError()



__all__ = [
    'BaseObject',
    'Loadable',
    'Object',
    'RootObject',
    'Storable',
    'field',
]