#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from . import log
from .root import RootObject
from .types import *
from .util import class_qname

if TYPE_CHECKING:
    from .meta import Meta


meta_by_type: dict[type, 'Meta'] = {}
meta_by_qname: dict[str, 'Meta'] = {}
qname_listeners: dict[str, list[Callable[['Meta'], None]]] = {}


def append_item(d: dict, key: str, item: Any) -> None:
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


RegistryFallback = Callable[['Registry', str], None]

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


class Registry(RootObject, Generic[T]):

    __slots__ = ['ifc', 'factories', 'fallbacks', 'namespaces', 'default_kind']

    ifc: type[T]
    factories: dict[str, Factory[T]]
    fallbacks: tuple[RegistryFallback, ...]
    namespaces: Optional[list]
    default_kind: Optional[str]

    def __init__(self, ifc: type[T], fallbacks: Iterable[RegistryFallback] = None):
        self.ifc = ifc
        self.factories = {}
        self.fallbacks = () if fallbacks is None else tuple(fallbacks)
        self.namespaces = None
        self.default_kind = None

    def configure(self, default_kind: str = None, modules: Union[str, Sequence[str]] = None, append_kind: bool = False,  **kwargs) -> None:
        self.debug('configuring registry {}: {}', class_qname(self.ifc), kwargs)
        if default_kind is not None:
            self.default_kind = default_kind
        if modules:
            if isinstance(modules, str):
                modules = modules,
            self.push_fallback(self.make_module_fallback(modules, append_kind=append_kind))

    def push_fallback(self, fallback: RegistryFallback) -> None:
        self.fallbacks = self.fallbacks + (fallback, )

    def pop_fallback(self) -> Optional[RegistryFallback]:
        if fallbacks := self.fallbacks:
            fallback = fallbacks[0]
            self.fallbacks = fallbacks[1:]
            return fallback
        return None

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
        if self.fallbacks:
            while fallback := self.pop_fallback():
                fallback(self, key)
                if factory := self.peek_factory(key):
                    return factory
        return None

    def register_object(self, key: str, obj: T):
        if isinstance(obj, type) and issubclass(obj, self.ifc):
            self.put_factory(obj, key=key)
        elif callable(obj):

            def singleton_factory(*args, **kwargs) -> T:
                self.debug('{}: getting singleton [{}] -> {}', class_qname(self.ifc), key, obj)
                return obj

            self.put_factory(singleton_factory, key=key)

    def get_factory(self, *, key: str = None, kind: str = None, from_type: str|type = None) -> Optional[Factory[T]]:
        key = factory_key(key=key, kind=kind, from_type=from_type, default_kind=self.default_kind)
        factory = self.peek_factory(key)

        if kind == 'relu':
            pass
        if factory is None:
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

    def _repr_args(self) -> str:
        return class_qname(self.ifc)

    @classmethod
    def method_factory(cls, ifc: type[T], key: str, reg: 'Registry[T]' = None) -> Optional[Factory[T]]:
        if key.startswith('from:'):
            fallback_method = f'_coerce_from_{key[5:]}'
            return getattr(ifc, fallback_method, None)
        elif key.startswith('kind:') and reg:
            return reg.fallback_factory(key)
        elif key == 'kind:default' or key == 'default':
            return getattr(ifc, 'provide_from', ifc)
        return None

    def make_module_fallback(self, modules: Union[str, Sequence[str]],
                             append_kind: bool = False,
                             register: bool = None) -> RegistryFallback:
        if not modules:
            raise ValueError('Must specify one or more modules to search in')

        if isinstance(modules, str):
            modules = modules,
        elif not isinstance(modules, Sequence) and all(isinstance(m, str) for m in modules):
            raise ValueError('modules must be a string or a sequence of strings')

        import importlib

        if append_kind:
            if register is None:
                register = False

            def fallback(registry: Registry[T], key: str) -> T:
                kind = key[len(kind_prefix):]
                for module_name in modules:
                    for mname in (f'{module_name}.{kind}', module_name):
                        try:
                            registry.debug('{}: dynamically loading module [{}] for kind [{}]', registry, mname, kind)
                            module = importlib.import_module(mname)
                            if register:
                                obj = getattr(module, kind, None)
                                if obj is not None:
                                    registry.register_object(key, obj)
                            return
                        except ImportError:
                            pass

                raise ValueError(f'{registry}: Nothing registered with kind [{kind}]')
        else:
            if register is None:
                register = True

            def fallback(registry: Registry[T], key: str) -> T:
                kind = key[len(kind_prefix):]
                for mname in modules:
                    try:
                        registry.debug('{}: dynamically loading module [{}] for kind [{}]', registry, mname, kind)
                        module = importlib.import_module(mname)
                        if registry.namespaces is None:
                            registry.namespaces = [module]
                        else:
                            registry.namespaces.append(module)
                        obj = getattr(module, kind, None)
                        if obj is not None:
                            if register:
                                registry.register_object(key, obj)
                            return
                    except ImportError:
                        pass

                raise ValueError(f'{registry}: Nothing registered with kind [{kind}]')

        return fallback

