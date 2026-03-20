#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import importlib

from . import log
from .root import RootObject
from .types import *
from .util import class_qname


if TYPE_CHECKING:
    from .meta import Meta


meta_by_type: dict[type, 'Meta'] = {}
meta_by_qname: dict[str, 'Meta'] = {}
qname_listeners: dict[str, list[Callable[['Meta'], None]]] = {}


def append_item(d: dict[str, list], key: str, item: Any) -> None:
    if items := d.get(key):
        items.append(item)
    else:
        d[key] = [item]


def add_qname_listener(qname: str, listener: Callable[['Meta'], None]) -> None:
    append_item(qname_listeners, qname, listener)


def meta_register(meta: 'Meta', cls: type = None, qname: str = None, *, override: bool = False) -> None:
    if qname is None: qname = meta.qname
    if cls is None: cls = meta.cls
    if qname in meta_by_qname and not override:
        raise MetaError(f'register({cls}): cannot register {cls} with qname {qname} because it is already registered')
    if cls in meta_by_type and not override:
        raise MetaError(f'register({cls}): cannot register {cls} because it is already registered as {meta_by_type[cls]}')
    meta_by_type[cls] = meta
    meta_by_qname[qname] = meta
    if listeners := qname_listeners.get(qname):
        for listener in listeners:
            listener(meta)
        listeners.clear()
        del qname_listeners[qname]


class Factory(Protocol[T]):

    def __call__(self, spec: Any = None, /, **kwargs) -> T: ...


class ModuleFallback(RootObject):

    __slots__ = ('modules', 'modules_tried', 'append_kind', 'register',)

    modules: list[str]
    modules_tried: int
    append_kind: bool
    register: bool

    def __init__(self, modules: list[str], *, append_kind: bool = False, register: bool = None) -> None:
        self.modules = list(modules)
        self.modules_tried = 0
        self.append_kind = append_kind
        if register is None:
            register = not append_kind
        self.register = register

    def iter_modules(self, kind: str) -> Iterable[str]:
        if self.append_kind:
            for module_name in self.modules:
                yield f'{module_name}.{kind}'
        for module_name in self.modules[self.modules_tried:]:
            self.modules_tried += 1
            yield module_name

    def __call__(self, registry: 'Registry', key: str) -> None:
        if self.modules_tried < len(self.modules):
            kind = key[len(kind_prefix):]
            for mname in self.iter_modules(kind):
                try:
                    module = importlib.import_module(mname)
                    self.debug('{}: dynamically loaded module [{}] for kind [{}]', registry, mname, kind)
                    if registry.namespaces is None:
                        registry.namespaces = [module]
                    else:
                        registry.namespaces.append(module)
                    factory = registry.peek_factory(key)
                    if factory is not None:
                        return
                    if self.register:
                        obj = getattr(module, kind, None)
                        if obj is not None:
                            registry.register_object(key, obj)
                    return
                except ImportError:
                    self.warn('{}: dynamically loading module [{}] for kind [{}]', registry, mname, kind)

            raise ValueError(f'{registry}: Nothing registered with kind [{kind}]')

    def _repr_args(self, **options) -> str:
        rep = ', '.join(self.modules)
        if self.append_kind:
            rep += ', +append_kind'
        if self.register:
            rep += ', +register'
        return rep


kind_prefix = 'kind:'
from_prefix = 'from:'


def factory_key(*, key: str = None, kind: str = None, from_type: str|type = None, default_kind: str = None):
    if key is None:
        if from_type is None:
            if kind is None:
                if default_kind is None:
                    key = 'default'
                else:
                    key = kind_prefix + default_kind
            else:
                key = kind_prefix + kind
        else:
            if isinstance(from_type, type):
                from_type = class_qname(from_type)
                if from_type.startswith('builtins.'):
                    from_type = from_type[9:]
            key = from_prefix + from_type.replace('.', '_')
    return key


class Kinds(RootObject):

    __slots__ = ('kinds_for_impls', 'impl_for_kind')

    kinds_for_impls: dict[type, list[str]]
    impl_for_kind: dict[str, type]

    def __init__(self, kind: str, impl: type):
        self.kinds_for_impls = {impl: [kind]}
        self.impl_for_kind = {kind: impl}

    def add(self, impl: type, kind: str, primary: bool = False):
        if kinds := self.kinds_for_impls.get(impl):
            if primary:
                kinds.insert(0, kind)
            else:
                kinds.append(kind)
        else:
            self.kinds_for_impls[impl] = [kind]
        self.impl_for_kind[kind] = impl

    def get_kinds(self, impl: type) -> Optional[list[str]]:
        return self.kinds_for_impls.get(impl)

    def get_kind(self, impl: type) -> Optional[str]:
        if kinds := self.get_kinds(impl):
            return kinds[0]
        return None

    def get_impl(self, kind: str) -> Optional[type]:
        return self.impl_for_kind.get(kind)



class Registry(RootObject, Generic[T]):

    __slots__ = ['ifc', 'meta', 'factories', 'fallback', 'namespaces', 'default_kind',
                 'kinds']

    ifc: type[T]
    meta: 'Meta'
    factories: dict[str, Factory[T]]
    fallback: Optional[ModuleFallback]
    namespaces: Optional[list]
    default_kind: Optional[str]
    kinds: Optional[Kinds]

    def __init__(self, ifc: type[T], meta: 'Meta'):
        self.ifc = ifc
        self.meta = meta
        self.factories = {}
        self.fallback = None
        self.namespaces = None
        self.default_kind = None
        self.kinds = None

    def configure(self, default_kind: str = None, modules: Union[str, Sequence[str]] = None, append_kind: bool = False,  **kwargs) -> None:
        self.debug('configuring registry {}: {}', class_qname(self.ifc), kwargs)
        if default_kind is not None:
            self.default_kind = default_kind
        if modules:
            self.add_search_packages(modules, append_kind)

    def add_search_packages(self, modules: Union[str, Sequence[str]], append_kind: bool = False) -> None:
        if isinstance(modules, str):
            modules = [modules]
        if fallback := self.fallback:
            fallback.modules.extend(modules)
            if append_kind:
                fallback.append_kind = True
        else:
            self.fallback = ModuleFallback(modules, append_kind=append_kind)

    def add_kind(self, impl: type, kind: str, primary: bool = False):
        if kinds := self.kinds:
            kinds.add(impl, kind, primary)
        else:
            self.kinds = Kinds(kind, impl)

    def get_kinds(self, impl: type) -> Optional[list[str]]:
        if kinds := self.kinds:
            return kinds.get_kinds(impl)
        return None

    def get_kind(self, impl: type) -> Optional[str]:
        if kinds := self.kinds:
            return kinds.get_kind(impl)
        return None

    def has_kind_for_impl(self, impl: type) -> bool:
        if kinds := self.kinds:
            return impl in kinds.kinds_for_impls
        return False

    def get_impl_for_kind(self, kind: str) -> Optional[type]:
        if kinds := self.kinds:
            return kinds.get_impl(kind)
        return None

    def put_implementation(self, impl: type, *, key: str = None, kind: str = None, from_type: str = None,
                           override: bool = False) -> Optional[Factory[Any]]:
        factory = getattr(impl, 'provide_from', impl)
        self.add_kind(impl, kind)
        return self.put_factory(factory, key=key, kind=kind, from_type=from_type, override=override)

    def put_factory(self, factory: Factory[Any], *, key: str = None, kind: str = None, from_type: str = None,
                    override: bool = False) -> Optional[Factory[Any]]:
        key = factory_key(key=key, kind=kind, from_type=from_type)
        factories = self.factories
        if key in factories and not override:
            raise KeyError(f'Factory for {class_qname(self.ifc)} already exists for key {key}')
        factories[key] = factory

    def peek_factory(self, key: str) -> Optional[Factory[T]]:
        if factory := self.factories.get(key):
            return factory
        if namespaces := self.namespaces:
            if key.startswith(kind_prefix):
                kind = key[len(kind_prefix):]
                for ns in namespaces:
                    if obj := getattr(ns, kind, None):
                        self.register_object(key, obj)
                        # self.factories[key] = factory
                        return self.factories[key]
        return None

    def fallback_factory(self, key: str) -> Optional[Factory[T]]:
        if self.fallback:
            self.fallback(self, key)
            if factory := self.peek_factory(key):
                return factory
            # while fallback := self.pop_fallback():
            #     fallback(self, key)
            #     if factory := self.peek_factory(key):
            #         return factory
        return None

    def register_object(self, key: str, obj: T):
        if isinstance(obj, type) and issubclass(obj, self.ifc):
            self.put_factory(obj, key=key)
        elif callable(obj):

            def singleton_factory(*args, **kwargs) -> T:
                self.debug('{}: getting singleton [{}] -> {}', class_qname(self.ifc), key, obj)
                return obj

            self.put_factory(singleton_factory, key=key)

    def get_key(self, *, key: str = None, kind: str = None, from_type: str|type = None) -> str:
        return factory_key(key=key, kind=kind, from_type=from_type, default_kind=self.default_kind)

    def get_factory(self, *, key: str = None, kind: str = None, from_type: str|type = None) -> Optional[Factory[T]]:
        key = factory_key(key=key, kind=kind, from_type=from_type, default_kind=self.default_kind)
        factory = self.peek_factory(key)

        if factory is None:
            if kind and '.' in kind:
                print(f'Got a kind with a dot: {kind}')
            factory = Registry.method_factory(self.ifc, key, reg=self)
            # if key.startswith('from:'):
            #     fallback_method = f'_coerce_from_{key[5:]}'
            #     factory = getattr(self.cls, fallback_method, None)
            # elif key.startswith('kind:'):
            #     factory = self.fallback_factory(key)
            # elif key == 'default':
            #     factory = getattr(self.cls, 'create', None)

            if factory is not None:
                self.factories[key] = factory
            elif from_type and isinstance(from_type, type):
                log.debug('No factory for {} with key {} and from_type {}',
                          class_qname(self.ifc), key, from_type)
            else:
                log.debug('No factory for {} with key {}', class_qname(self.ifc), key)
        return factory

    def get_factories(self, *, key: str = None, kind: str = None, from_type: str|type = None) -> Iterable[Factory[T]]:
        return self.get_factory(key=key, kind=kind, from_type=from_type),

    def _repr_args(self) -> str:
        return class_qname(self.ifc)

    @staticmethod
    def method_factory(ifc: type[T], key: str, reg: 'Registry[T]' = None) -> Optional[Factory[T]]:
        if key.startswith('from:'):
            fallback_method = f'_coerce_from_{key[5:]}'
            return getattr(ifc, fallback_method, None)
        elif key.startswith('kind:') and reg:
            return reg.fallback_factory(key)
        elif key == 'kind:default' or key == 'default':
            return getattr(ifc, 'provide_from', ifc)
        return None

