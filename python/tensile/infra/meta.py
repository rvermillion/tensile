#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from typing import Literal, cast, overload

import sys

from .function import identity
from .types import *
from .behavior import *
from .root import *
from .field import Field, FieldType, field, private_slot
from .registry import Registry, Factory, factory_key, meta_by_qname, meta_by_type, meta_register
from .util import class_qname, process_specs


if TYPE_CHECKING:
    import tensile.infra


class Updateable(Protocol):

    def update_fields(self, spec: Keywords = None, /, **kwargs): ...


def update_from_spec(this: Any, spec: Spec):
    for key, val in spec.items():
        setattr(this, key, val)



class UpdateableObject(RootObject):

    __slots__ = ()

    def update_fields(self, spec: Keywords = None, /, **kwargs):
        self._update_from_spec(Spec.combine(spec, kwargs), True)

    def _update_from_spec(self, spec: Spec, update: bool = False):
        for key, val in spec.items():
            setattr(self, key, val)


class CoerceFunction(Protocol):

    def __call__(self, spec: Any = None, /, **kwargs) -> Any: ...


class Meta(RootObject):

    __slots__ = ['cls', 'bases', 'children', 'registry', 'coerce', 'get_state', 'update_state']

    cls: type
    name: str
    qname: str
    origin: Optional['Meta']
    bases: tuple['Meta', ...]
    children: list['Meta']
    fields: dict[str, Field]
    registry: Optional[Registry[Any]]
    coerce: CoerceFunction
    get_state: Callable[[Any], Any]
    update_state: Callable[[Any, Any], Any]

    # noinspection PyShadowingNames
    def __init__(self, cls: type, coerce: CoerceFunction = None, get_state: Callable[[Any], Any] = None,
                 update_state: Callable[[Any, Any], Any] = None, **kwargs):
        if cls in meta_by_type:
            raise MetaError(f'Meta object already defined for {cls}')

        self.cls = cls
        self.children = []
        self.bases = ()
        self.registry = None
        self.coerce = self.default_coerce if coerce is None else coerce
        self.get_state = self.default_get_state if get_state is None else get_state
        self.update_state = self.default_update_state if update_state is None else update_state

        meta_register(self, cls=cls)

    @property
    def name(self) -> str:
        return self.cls.__qualname__

    @property
    def qname(self) -> str:
        return class_qname(self.cls)

    @property
    def fields(self) -> dict[str, Field]:
        return {}

    @property
    def class_fields(self) -> dict[str, Field]:
        return {n: f for n, f in self.fields.items() if Scope.is_class(f.scope)}

    @property
    def instance_fields(self) -> dict[str, Field]:
        return {n: f for n, f in self.fields.items() if Scope.is_instance(f.scope)}

    def default_coerce(self, spec: Any = None, /, **kwargs) -> Any:
        if self.has_instance(spec):
            return spec

        cls = self.cls

        if spec is None:
            if not kwargs:
                if factory := self.get_factory(from_type='none'):
                    return factory()
                raise TypeError(f'{cls}: cannot coerce None with kwargs {kwargs} because no factory is registered for None')

        factory = None
        if spec is None or isinstance(spec, Mapping):
            kind, new_spec = process_specs(spec, kwargs)
            if kind is not None:
                if factory := self.get_factory(kind=kind):
                    return factory(new_spec)
            factory = self.get_factory(from_type='mapping')
            if factory is None:
                if spec is None or isinstance(spec, dict):
                    factory = self.get_factory(from_type='dict')
            if factory:
                new_spec = kwargs
                if spec is not None: new_spec.update(spec)
                return factory(new_spec)
            raise TypeError(f'{cls}: cannot coerce {spec} with kwargs {kwargs} because no factory is registered for kind [{kind}]')
        elif isinstance(spec, str):
            factory = self.get_factory(from_type='str')
        elif isinstance(spec, Sequence):
            factory = self.get_factory(from_type='sequence')
        elif isinstance(spec, type):
            factory = self.get_factory(from_type='type')
        elif callable(spec):
            factory = self.get_factory(from_type='callable')

        if factory:
            # noinspection PyCallingNonCallable
            return factory(spec, **kwargs)
        elif spec:
            spec_cls = type(spec)
            factory = self.get_factory(from_type=spec_cls)
            if factory is None:
                for sup_cls in spec_cls.mro()[1:]:
                    self.warn('looking for factory of {} from spec superclass {}', cls, sup_cls)
                    factory = self.get_factory(from_type=sup_cls)
                    if factory: break

            if factory: return factory(spec, **kwargs)
        raise ValueError(f'Cannot coerce {spec} with kwargs [{kwargs}] to {cls}')

    def get_class(self, cls: type[T]) -> Optional[type[T]]:
        if issubclass(self.cls, cls):
            return self.cls
        return None

    def add_child(self, child: 'Meta'):
        self.children.append(child)

    def has_instance(self, obj: Any) -> bool:
        return isinstance(obj, self.cls)

    def is_special_instance(self, obj: Any) -> bool:
        if registry := self.registry:
            return registry.has_kind_for_impl(obj.__class__)
        return False

    def all_parents(self) -> Iterable['Meta']:
        inherit = self.cls.mro()[:0:-1]
        for parent in inherit:
            if meta := Meta.for_class(parent):
                yield meta

    def all_children(self, dedupe: Union[set['Meta'], bool] = None) -> Iterable['Meta']:
        if children := self.children:
            if dedupe is None or dedupe is False:
                for child in children:
                    yield child
                    yield from child.all_children(dedupe)
            else:
                if dedupe is True:
                    dedupe = set()
                for child in children:
                    if child not in dedupe:
                        dedupe.add(child)
                        yield child
                        yield from child.all_children(dedupe)

    def get_impl_for_kind(self, kind: str) -> Optional[type]:
        if registry := self.registry:
            if impl := registry.get_impl_for_kind(kind=kind):
                return impl
            dot = kind.find('.')
            if dot > 0:
                if impl := registry.get_impl_for_kind(kind[:dot]):
                    meta = for_class(impl)
                    if meta:
                        impl = meta.get_impl_for_kind(kind=kind[dot+1:])
                        registry.add_kind(impl, kind)
                        return impl
        return None

    def get_registry(self) -> Registry[Any]:
        registry = self.registry
        if registry is None:
            cls: type = self.cls
            # noinspection PyTypeChecker
            registry = self.registry = Registry(cls, self)
        return registry

    def configure_registry(self, default_kind: str = None, modules: Union[str, Sequence[str]] = None, append_kind: bool = False, **kwargs) -> Registry[Any]:
        registry = self.get_registry()
        registry.configure(
            default_kind=default_kind,
            modules=modules,
            append_kind=append_kind,
            **kwargs)
        return registry

    def get_factory(self, *, key: str = None, kind: str = None, from_type: str|type = None) -> Optional[Factory]:
        registry = self.registry
        if registry is None:
            # noinspection PyTypeChecker
            return None
            # return Registry.method_factory(self.cls, factory_key(key=key, kind=kind, from_type=from_type))
        return registry.get_factory(key=key, kind=kind, from_type=from_type)

    @staticmethod
    def spread_spec(factory: Factory[T]) -> Factory[T]:

        def spread(spec: Any = None, /, **kwargs) -> T:
            if spec and isinstance(spec, Mapping):
                if not kwargs:
                    return factory(**spec)
                kwargs.update(spec)
            return factory(**kwargs)

        # noinspection PyTypeChecker
        return spread

    # noinspection PyShadowingNames
    def provide_from_type(self, *types: Union[type, str], spread: bool = False) -> Callable[[T], T]:
        reg = self.get_registry()

        types = tuple(class_qname(t) if isinstance(t, type) else t for t in types)

        def decorator(sub: T) -> T:
            if isinstance(sub, type):
                factory = getattr(sub, 'provide_from', sub)
            elif callable(sub):
                if spread:
                    factory = self.spread_spec(sub)
                else:
                    factory = sub
            else:
                raise ValueError('Ooops!')

            for t in types:
                reg.debug('register({}): register from type [{}] as {}',
                          class_qname(self.cls), t, class_qname(factory))
                reg.put_factory(factory, from_type=t)
            return sub
        return decorator

    def register_singletons(self, **kinds: T) -> None:
        reg = self.get_registry()

        for kind, obj in kinds.items():
            reg.debug('register({}): register singleton for kind [{}] as {!r}',
                      class_qname(self.cls), kind, obj)
            reg.register_object(reg.get_key(kind=kind), obj)

    def provide_singleton(self, *kinds: str) -> Callable[[T], T]:
        reg = self.get_registry()

        def decorator(obj: T) -> T:
            for kind in kinds:
                reg.debug('register({}): register singleton for kind [{}] as {!r}',
                          class_qname(self.cls), kind, obj)
                reg.register_object(reg.get_key(kind=kind), obj)
            return obj
        return decorator

    def provide(self, *kinds: str, spread: bool = False) -> Callable[[T], T]:
        reg = self.get_registry()

        if kinds:
            def decorator(sub: T) -> T:
                if isinstance(sub, type):
                    for kind in kinds:
                        reg.put_implementation(sub, kind=kind)
                elif callable(sub):
                    factory = self.spread_spec(sub) if spread else sub
                    for kind in kinds:
                        reg.put_factory(factory, kind=kind)
                else:
                    raise ValueError('Ooops!')

                return sub
        else:
            def decorator(sub: type[T]) -> type[T]:
                if kind := sub.__dict__.get('kind'):
                    if isinstance(sub, type):
                        reg.put_implementation(sub, kind=kind)
                    elif callable(sub):
                        factory = self.spread_spec(sub) if spread else sub
                        reg.put_factory(factory, kind=kind)
                    else:
                        raise ValueError('Ooops!')
                else:
                    raise ValueError(f'register({self.qname}): Must specify a kind or '
                                     f'have a class attribute for {class_qname(sub)}')
                return sub

        return decorator

    def default_get_state(self, obj: Any) -> Any:
        if obj is None: return None
        self.logger.warn('Using string for state: {}', obj)
        return repr(obj)

    def default_update_state(self, obj: Any, state: Any) -> Any:
        self.logger.warn('Cannot update state: {}', obj)

    def _repr_args(self) -> str:
        return class_qname(self.cls)

    @overload
    @classmethod
    def for_class(cls, impl: type, attr: str = 'meta', build: Literal[True] = True) -> Self: ...

    @overload
    @classmethod
    def for_class(cls, impl: type, attr: str = 'meta', build: bool = False) -> Self|None: ...

    @overload
    @classmethod
    def for_class(cls, impl: type, attr: str = 'meta') -> Self: ...

    @classmethod
    def for_class(cls, impl: type, attr: str = 'meta', build: bool = False) -> Optional[Self]:
        meta = meta_by_type.get(impl)
        if meta is None:
            if meta := getattr(impl, attr, None):
                if isinstance(meta, cls):
                    return meta
        if meta is None and build:
            return cls.build(impl)
        if meta is not None and not isinstance(meta, cls):
            raise MetaError(f'Meta object for {class_qname(impl)} is not a {class_qname(cls)}: {meta}')
        return meta

    @classmethod
    def for_qname(cls, qname: str) -> Optional[Self]:
        meta = meta_by_qname.get(qname)
        if meta is None:
            if qname[-1] == ']':
                raise MetaError(f'Cannot handle type params yet: {qname}')
        if meta is not None and not isinstance(meta, cls):
            raise MetaError(f'Meta object for {qname} is not a {class_qname(cls)}: {meta}')
        return meta

    @classmethod
    def build(cls, impl: type, **kwargs) -> 'Meta':
        meta_cls = getattr(impl, 'Meta', None)
        if meta_cls is None:
            meta_cls = ProtocolMeta if is_protocol(impl) else cls
        meta = meta_cls(impl, **kwargs)
        return meta

    by_type: ClassVar[Mapping[type, 'Meta']] = meta_by_type
    by_qname: ClassVar[Mapping[str, 'Meta']] = meta_by_qname

    Field: ClassVar[type[Field]] = Field


Meta.for_class(str, build=True).get_state = identity
Meta.for_class(int, build=True).get_state = identity
Meta.for_class(float, build=True).get_state = identity
Meta.for_class(bool, build=True).get_state = identity


class ProtocolMeta(Meta):

    __slots__ = ()

    def has_instance(self, obj: Any) -> bool:
        return isinstance(obj, self.cls) if is_runtime_protocol(self.cls) else False


class ObjectMeta(Meta):

    __slots__ = ('fields', 'own_fields', 'field_inits', 'slots')

    cls: type['tensile.infra.Object']
    fields: dict[str, Field]
    own_fields: dict[str, Field]
    field_inits: tuple[Initter, ...]
    slots: set[str]

    def __init__(self, cls: type['tensile.infra.Object'], **kwargs):
        Meta.__init__(self, cls, **kwargs)

        cls.meta = self

        self.slots = set(getattr(cls, '__slots__', ()))

        bases = []
        for base in cls.__bases__:
            if base_meta := Meta.for_class(base):
                base_meta.add_child(self)
                bases.append(base_meta)
        self.bases = tuple(bases)

        if issubclass(cls, RootObject):
            fields = {}
            if len(bases) == 1:
                fields.update(bases[0].fields)
            else:
                for parent in self.all_parents():
                    fields.update(parent.fields)

            self.fields = fields
            self.own_fields = {}
        else:
            self.fields = self.own_fields = {}

    def has_slot(self, slot: str) -> bool:
        return slot in self.slots

    def add_field(self, name: str, spec: Spec):
        f = self.fields.get(name)
        if f is None:
            f = self.build_field(name, spec)
        else:
            f = f.override(self, spec)

        if f is not None:
            self.fields[name] = f
            self.own_fields[name] = f

    def build_field(self, name: str, spec: Spec) -> Optional[Field]:
        return self.Field.build(self, name, spec)

    def process_annotations(self, cls: type = None):
        if cls is None:
            cls = self.cls

        # noinspection PyShadowingNames,PyPep8Naming
        Field = self.Field

        annos = cls.__annotations__

        if annos:
            # print('-' * 10, f'{self.qname} processing {len(annos)} annotations ', '-' * 100)

            for name, anno in annos.items():
                field_spec = Field.new_spec()
                anno_origin = get_origin(anno)

                if anno_origin is ClassVar:
                    field_spec['scope'] = Scope.class_scope
                    # If you don't want ClassVar's ignored, you have to set ignore=False explicitly
                    field_spec['ignore'] = True
                    anno, = get_args(anno)
                    anno_origin = get_origin(anno)

                if anno_origin is Annotated:
                    anno, *anno_args = get_args(anno)

                    for anno_arg in anno_args:
                        if isinstance(anno_arg, str):
                            field_spec['doc'] = anno_arg
                        elif isinstance(anno_arg, Mapping):
                            field_spec.update(anno_arg)
                        elif isinstance(anno_arg, Field):
                            anno_arg.to_spec(field_spec)

                if field_spec.get('ignore', False):
                    continue

                field_spec['type'] = FieldType.from_anno(anno, cls)

                if name[0] == '_':
                    field_spec['visibility'] = Visibility.protected

                # value = getattr(cls, name, ...)
                value = cls.__dict__.get(name, ...)
                if value is not ...:
                    field_spec['member'] = value
                else:
                    value = getattr(cls, name, ...)
                    if value is not ... and isinstance(value, MemberDescriptorType):
                        field_spec['member'] = value

                self.add_slot(cls, name, field_spec)

                if field_spec.get('visibility', Visibility.public) is Visibility.protected:
                    self.debug(f'meta: processing protected field [{name}]:', field_spec)


                # print(f'  {name:>20}:', field_spec)
                self.add_field(name, field_spec)

            # print('-' * 100)

    @staticmethod
    def slot_names(name: str) -> Iterable[str]:
        return private_slot(name),

    def add_slot(self, cls: type, name: str, field_spec: Spec) -> None:
        for slot in self.slot_names(name):
            if self.has_slot(slot):
                field_spec['slot'] = slot
                if 'member' not in field_spec:
                    value = cls.__dict__.get(slot, ...)
                    if value is not ...  and isinstance(value, MemberDescriptorType):
                        field_spec['member'] = value
                return

    def engineer_fields(self) -> None:
        own_fields = self.own_fields
        for f in own_fields.values():
            f.engineer(self)

        init_fields = [f for f in self.fields.values() if Scope.is_instance(f.scope) and f.init]

        init_fields.sort(key=lambda x: x.init_order)

        self.field_inits = tuple(f.init for f in init_fields if f.init)

    def engineer_state(self) -> None:
        pass

    def init(self, this: Any, spec: Spec):
        for init in self.field_inits:
            init(this, spec)

    def update_instance(self, this: Any, spec: Spec):
        updateable_fields = self.updateable_fields
        for name, val in spec.items():
            if f := updateable_fields.get(name):
                try:
                    f.update(this, val)
                except Exception:
                    raise AttributeError(f'{f}: error on update')

    @property
    def updateable_fields(self) -> dict[str, Field]:
        return {n: f for n, f in self.fields.items() if Scope.is_instance(f.scope) and not f.readonly}

    def print(self, own: bool = True):
        print('-' * 100)
        print(self)
        fields = self.instance_fields
        if own:
            fields = {n: f for n, f in fields.items() if f.owner is self}
        for name, f in fields.items():
            print('  ', f.describe())
        fields = self.class_fields
        if own:
            fields = {n: f for n, f in fields.items() if f.owner is self}
        if fields:
            print('---')
            for name, f in fields.items():
                print('  ', f.describe())

    def _repr_args(self) -> str:
        return self.qname

    def default_coerce(self, spec: Any = None, /, **kwargs) -> 'tensile.infra.Object':
        if self.has_instance(spec):
            return cast('tensile.infra.Object', spec)

        cls = self.cls

        return cls.coerce(spec, **kwargs)

    @classmethod
    def engineer(cls, impl: type, **kwargs) -> None:
        meta = meta_by_type.get(impl)
        if meta is None:
            meta = impl.__dict__.get('meta')
        if meta is None:
            meta = cls.build(impl, **kwargs)
            meta.process_annotations()
            meta.engineer_fields()
            meta.engineer_state()


        # meta.print()


@overload
def for_class(cls: type, build: Literal[True]) -> Meta: ...

@overload
def for_class(cls: type, build: bool) -> Meta|None: ...

@overload
def for_class(cls: type) -> Meta: ...

def for_class(cls: type, build: bool = True) -> Meta | None:
    return Meta.for_class(cls, build=build)


def for_qname(qname: str) -> Optional[Meta]:
    return Meta.for_qname(qname)


def get_class(qname: str, carp: bool = True) -> Optional[type]:
    meta = for_qname(qname)
    if meta is None:
        if carp:
            raise MetaError(f'Cannot find class for {qname}')
        return None
    return meta.cls


def for_spec(spec: Any, build: bool = False) -> Optional[Meta]:
    if isinstance(spec, type):
        return for_class(spec, build=build)
    elif isinstance(spec, str):
        meta = for_qname(spec)
        if meta is None and build:
            raise MetaError(f'Cannot build meta for {spec}')
        return meta
    return None


def coerce_class(cls: str|type) -> type:
    if isinstance(cls, str):
        if m := meta_for_qname(cls):
            return m.cls
    elif isinstance(cls, type):
        return cls
    raise ValueError(f'Invalid class: {cls}')


def register_singletons(cls: type, **kinds: T) -> None:
    meta = Meta.for_class(cls, build=True)
    return meta.register_singletons(**kinds)


def provides(cls: type, *kinds, spread: bool = False) -> Callable[[T], T]:
    meta = Meta.for_class(cls, build=True)
    return meta.provide(*kinds, spread=spread)


def provides_singleton(cls: type, *kinds) -> Callable[[T], T]:
    meta = Meta.for_class(cls, build=True)
    return meta.provide_singleton(*kinds)


# noinspection PyShadowingNames
def provides_from_type(cls: type, *types, spread: bool = False) -> Callable[[T], T]:
    meta = Meta.for_class(cls, build=True)
    return meta.provide_from_type(*types, spread=spread)


def coerce(cls: type[T], spec: Any = None, /, **kwargs) -> T:
    try:
        meta = Meta.for_class(cls, build=True)
        return meta.coerce(spec, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to coerce {class_qname(cls)} with spec {spec} and kwargs {kwargs}: {e}") from e


def alias(module: str, classes: Sequence[str|type]) -> None:
    if mod := sys.modules.get(module):
        for obj in classes:
            if isinstance(obj, str):
                name = obj
                qname = module + '.' + name
                cls = getattr(mod, name, None)
            else:
                qname = None
                cls = obj

            if isinstance(cls, type):
                meta = Meta.for_class(cls, build=True)
                qname = qname or module + '.' + meta.name
                if qname != meta.qname:
                    if qname in meta_by_qname:
                        raise MetaError(f'Duplicate qname: {qname}')
                    meta_by_qname[qname] = meta
    else:
        raise ImportError(f'Cannot find module {module}')


# noinspection PyShadowingNames
def configure_coerce(cls: type[T], coerce: CoerceFunction) -> Meta:
    meta = Meta.for_class(cls, build=True)
    meta.coerce = coerce
    return meta


meta_configure_coerce = configure_coerce
meta_for_class = for_class
meta_for_qname = for_qname
meta_for_spec = for_spec
meta_coerce_class = coerce_class


__all__ = [
    'Annotated',
    'Field',
    'MemberDescriptorType',
    'Meta',
    'ObjectMeta',
    'RootObject',
    'Scope',
    'Spec',
    'UpdateableObject',
    'coerce',
    'field',
    'meta_configure_coerce',
    'meta_for_class',
    'meta_for_qname',
    'meta_for_spec',
    'meta_coerce_class',
    'provides',
    'private_slot'
]