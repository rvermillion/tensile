#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
import sys

from .types import *
from .behavior import *
from .root import *
from . import field as fields
from .field import Field, FieldType, field, private_slot
from .registry import Registry, Factory, factory_key, meta_by_qname, meta_by_type, meta_register
from .util import class_qname, process_specs


if TYPE_CHECKING:
    import tensile.infrastructure


class DeferredProperty:

    name: str
    owner: type
    slot: str

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __set_name__(self, owner, name):
        self.name = name
        self.owner = owner
        prop = self.build()
        setattr(owner, name, prop)

    def __getattr__(self, item):
        if item[0] != '_':
            if lazy := getattr(self, f'_lazy_{item}', None):
                return lazy()
        return super().__getattribute__(item)

    def _lazy_slot(self):
        return f'_{self.name}'

    def build(self) -> property:
        raise NotImplementedError()


class DeferredDictProperty(DeferredProperty):

    adder: str

    def _lazy_adder(self):
        return f'add_{self.name}'

    def build(self) -> property:
        slot = self.slot
        adder = self.adder

        def fget(this):
            return getattr(this, slot)

        if adder:
            def fset(this, value):
                setattr(this, slot, {})
                if isinstance(value, Sequence):
                    if value:
                        add = getattr(this, adder)
                        for spec in value:
                            if isinstance(spec, Mapping):
                                add(**spec)
                            else:
                                add(spec)
                elif isinstance(value, Mapping):
                    if value:
                        add = getattr(this, adder)
                        for path, spec in value.items():
                            if isinstance(spec, Mapping):
                                add(path, **spec)
                            else:
                                add(path, spec)
                elif value is not None:
                    raise ValueError(f'{slot[1:]} must be either a sequence or a mapping')
        else:
            def fset(this, value):
                coll = {}
                setattr(this, slot, coll)
                if isinstance(value, Mapping):
                    for key, spec in value.items():
                        coll[key] = spec
                elif value is not None:
                    raise ValueError(f'{slot[1:]} must be either a sequence or a mapping')

        return property(fget, fset)


if TYPE_CHECKING:
    # noinspection PyUnusedLocal
    def dict_property(slot: str = None, adder: str = None) -> property: ...
else:
    def dict_property(**kwargs):
        return DeferredDictProperty(**kwargs)


class Updateable(Protocol):

    def update(self, spec: Keywords = None, /, **kwargs): ...


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



class Meta(UpdateableObject):

    __slots__ = ['cls', 'bases', 'children', 'registry']

    cls: type
    name: str
    qname: str
    origin: Optional['Meta']
    bases: tuple['Meta', ...]
    children: list['Meta']
    fields: dict[str, Field]
    registry: Optional[Registry[Any]]

    def __init__(self, cls: type, **kwargs):
        if cls in meta_by_type:
            raise MetaError(f'Meta object already defined for {cls}')

        self.cls = cls
        self.children = []
        self.bases = ()
        self.registry = None

        meta_register(self, cls=cls)
        # meta_by_type[cls] = self
        # meta_by_qname[self.qname] = self

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
    def instance_fields(self) -> dict[str, Field]:
        return {n: f for n, f in self.fields.items() if Scope.is_instance(f.scope)}

    def add_child(self, child: 'Meta'):
        self.children.append(child)

    def has_instance(self, obj: Any) -> bool:
        return isinstance(obj, self.cls)

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

    def get_registry(self) -> Registry[Any]:
        registry = self.registry
        if registry is None:
            cls: type[Any] = self.cls
            # noinspection PyTypeChecker
            registry = self.registry = Registry(cls)
        return registry

    def configure_registry(self, default_kind: str = None, modules: Union[str, Sequence[str]] = None, append_kind: bool = False, **kwargs) -> Registry[Any]:
        registry = self.get_registry()
        registry.configure(
            default_kind=default_kind,
            modules=modules,
            append_kind=append_kind,
            **kwargs)
        return registry

    # def put_factory(self, factory: Factory[Any], *, key: str = None, kind: str = None, from_type: str = None,
    #                 override: bool = False) -> None:
    #     registry = self.get_registry()
    #     registry.put_factory(factory, key=key, kind=kind, from_type=from_type, override=override)

    def get_factory(self, *, key: str = None, kind: str = None, from_type: str|type = None) -> Optional[Factory[Any]]:
        registry = self.registry
        if registry is None:
            # noinspection PyTypeChecker
            return Registry.method_factory(self.cls, factory_key(key=key, kind=kind, from_type=from_type))
        return registry.get_factory(key=key, kind=kind, from_type=from_type)

    def coerce(self, spec: Any = None, /, **kwargs) -> Any:

        if self.has_instance(spec):
            return spec

        cls = self.cls

        if spec is None:
            if not kwargs:
                factory = self.get_factory(from_type='none')
                return factory()
                # return cls._coerce_from_none()

        factory = None
        kind = None
        if spec is None or isinstance(spec, Mapping):
            kind, new_spec = process_specs(spec, **kwargs)
            # if kind is None:
            #     kind = 'default'
                # return cls.create(spec)
            if kind is not None:
                factory = self.get_factory(kind=kind)
                if factory:
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
            # if factory is None:
            #     factory = self.get_factory(from_type='sequence')
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
        raise ValueError(f'Cannot coerce {spec} of kind [{kind}] to {cls}')

    def spread_spec(self, factory: Factory[T]) -> Factory[T]:

        def spread(spec: Any = None, /, **kwargs) -> T:
            if spec and isinstance(spec, Mapping):
                if not kwargs:
                    return factory(**spec)
                kwargs.update(spec)
            return factory(**kwargs)

        # noinspection PyTypeChecker
        return spread

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
                    factory = getattr(sub, 'provide_from', sub)
                elif callable(sub):
                    if spread:
                        factory = self.spread_spec(sub)
                    else:
                        factory = sub
                else:
                    raise ValueError('Ooops!')

                for kind in kinds:
                    reg.debug('register({}): register kind [{}] as {}',
                              class_qname(self.cls), kind, class_qname(factory))
                    reg.put_factory(factory, kind=kind)
                return sub
        else:
            def decorator(sub: type[T]) -> type[T]:
                if kind := sub.__dict__.get('kind'):
                    if isinstance(sub, type):
                        factory = getattr(sub, 'provide_from', sub)
                    elif callable(sub):
                        if spread:
                            factory = self.spread_spec(sub)
                        else:
                            factory = sub
                    else:
                        raise ValueError('Ooops!')
                    reg.debug('register({}): register kind [{}] as {}',
                              class_qname(self.cls), kind, class_qname(factory))
                    reg.put_factory(factory, kind=kind)
                else:
                    raise ValueError(f'register({self.qname}): Must specify a kind or '
                                     f'have a class attribute for {class_qname(sub)}')
                return sub

        return decorator

    def _repr_args(self) -> str:
        return class_qname(self.cls)

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
    def build(cls, impl: type, **kwargs) -> Self:
        meta_cls = getattr(impl, 'Meta', None)
        if meta_cls is None:
            meta_cls = ProtocolMeta if is_protocol(impl) else cls
        meta = meta_cls(impl, **kwargs)
        return meta

    by_type: ClassVar[Mapping[type, 'Meta']] = meta_by_type
    by_qname: ClassVar[Mapping[str, 'Meta']] = meta_by_qname

    Field: ClassVar[type[Field]] = Field


class ProtocolMeta(Meta):

    __slots__ = ()

    def has_instance(self, obj: Any) -> bool:
        return isinstance(obj, self.cls) if is_runtime_protocol(self.cls) else False


class ObjectMeta(Meta):

    __slots__ = ['fields', 'own_fields', 'field_inits', 'slots']

    cls: type['tensile.infrastructure.Object']
    fields: dict[str, Field]
    own_fields: dict[str, Field]
    field_inits: tuple[Initter, ...]
    slots: set[str]

    def __init__(self, cls: type['tensile.infrastructure.Object'], **kwargs):
        Meta.__init__(self, cls, **kwargs)

        cls.meta = self

        self.slots = set(getattr(cls, '__slots__', ()))

        bases = []
        for base in cls.__bases__:
            if base_meta := Meta.for_class(base):
                base_meta.add_child(self)
                bases.append(base_meta)
        self.bases = tuple(bases)

        # if slots := cls.__dict__.get('__slots__', ()):
        #     slots = set(slots)
        # elif len(cls.__bases__) == 1:
        #     pass
        # else:
        #     pass

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

    def slot_names(self, name: str) -> Iterable[str]:
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

        init_fields.sort(key=lambda f: f.init_order)

        self.field_inits = tuple(f.init for f in init_fields)

    def init(self, this: Any, spec: Spec):
        for init in self.field_inits:
            init(this, spec)

    def update_instance(self, this: Any, spec: Spec):
        fields = self.updateable_fields
        for name, val in spec.items():
            if f := fields.get(name):
                try:
                    f.update(this, val)
                except Exception:
                    raise AttributeError(f'{f}: error on update')

    @property
    def class_fields(self) -> dict[str, Field]:
        return {n: f for n, f in self.fields.items() if Scope.is_class(f.scope)}

    @property
    def instance_fields(self) -> dict[str, Field]:
        return {n: f for n, f in self.fields.items() if Scope.is_instance(f.scope)}
        # return [f for n, f in self.fields.items() if f.scope == 'instance']

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

    def coerce(self, spec: Any = None, /, **kwargs) -> Self:
        if self.has_instance(spec):
            return spec

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

        # meta.print()


def for_class(cls: type, build: bool = True) -> Meta:
    return Meta.for_class(cls, build=build)


def for_qname(qname: str) -> Optional[Meta]:
    return Meta.for_qname(qname)


def provides(cls: type, *kinds, spread: bool = False) -> Callable[[T], T]:
    meta = Meta.for_class(cls, build=True)
    return meta.provide(*kinds, spread=spread)


def provides_singleton(cls: type, *kinds) -> Callable[[T], T]:
    meta = Meta.for_class(cls, build=True)
    return meta.provide_singleton(*kinds)


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
    'provides',
    'private_slot'
]