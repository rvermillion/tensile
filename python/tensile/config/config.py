#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

import re

from ..infra import Object, Representable, log, meta
from ..infra.field import method_builders
from ..infra.meta import Meta, ObjectMeta, Field, Scope
from ..infra.abc import ABCClass
from ..infra.util import name_function
from ..infra.types import (
    Annotated, Any, Callable, ClassVar, Iterable, Iterator, Optional, Sequence, TypeVar, Self, Union,
    Getter, Setter, Spec, missing
)
from .match import KeyMatcher, default_matchers, list_matchers


C = TypeVar('C', bound='Config')


class ConfigIterable(Iterable):

    config: 'ConfigData'
    keys: tuple[str, ...]

    def __init__(self, config: 'ConfigData', keys: tuple[str, ...]):
        self.config = config
        self.keys = keys

    def __iter__(self) -> Iterator:
        return self.config.iter(*self.keys)

    def __repr__(self) -> str:
        return f'ConfigIterable({self.config.path}.{", ".join(self.keys)})'


use_default = object()


class ConfigData(Representable):

    config: Annotated['Config', meta.field(
        doc='The config this data is for',
    )]

    step: Annotated[str, meta.field(
        doc='The step',
        default='config'
    )]
    parent: Annotated[Optional['ConfigData'], meta.field(
        doc='The parent config',
        default=None
    )]
    locals: Sequence[dict[str, Any]]
    inherit: Annotated[bool, meta.field(
        doc='Whether to inherit',
        default=True
    )]
    fallback: Iterable['ConfigData']
    matchers: Optional[Sequence[KeyMatcher]]
    defaults: Optional[dict[str, Any]]
    cache: dict[str, Any]
    skip_for_iter: bool = False
    config_types: ClassVar[dict[str, type['Config']]] = {}

    def __init__(self, *,
                 config: 'Config' = None,
                 values: Iterable[dict[str, Any]] = None,
                 fallback: Iterable['ConfigData'] = (),
                 step: str = None,
                 parent: 'ConfigData' = None,
                 inherit: bool = True,
                 skip_for_iter: bool = False,
                 ):
        self.config = config
        if not step:
            step = 'config'

        if values:
            value_sets = []
            for v in values:
                if v is not None:
                    if not isinstance(v, dict):
                        v = dict(v)
                    value_sets.append(v)
            self.locals = tuple(value_sets)
        else:
            self.locals = ()
        # value_sets = (v if isinstance(v, dict) else dict(v) for v in values) if values else ()
        # self.config = config
        # self.locals = tuple(v for v in value_sets if v)
        # print(f'Making config with step {step} parent {parent} fallback {fallback} and locals: {self._locals}')
        self.step = step
        self.parent = parent
        self.fallback = fallback
        self.inherit = inherit
        self.skip_for_iter = skip_for_iter
        self.matchers = None
        self.cache = {}
        self.defaults = None

    def get_matchers(self) -> Sequence[KeyMatcher]:
        if self.matchers is None:
            self.matchers = self.build_matchers()
        return self.matchers

    def build_matchers(self):
        template_matchers = self.config._config_template_matchers
        matchers = []
        for values in self.iter_locals():
            for key, val in values.items():
                if matcher := KeyMatcher.build_first(key, val, matchers=template_matchers):
                    matchers.append(matcher)

        return tuple(sorted(matchers)) if matchers else ()

    def copy(self, *args, **kwargs) -> Self:

        kwargs.setdefault('step', self.step)
        kwargs.setdefault('fallback', self.fallback)
        kwargs.setdefault('parent', self.parent)
        kwargs.setdefault('config', self.config)
        kwargs.setdefault('skip_for_iter', self.skip_for_iter)

        return self.construct(*args, **kwargs)

    def override(self, **kwargs) -> Self:
        return self.copy(kwargs, *self.locals)

    @property
    def parent_for_iter(self) -> Optional['ConfigData']:
        if (parent := self.parent) is not None:
            return parent.for_iter
        return None

    @property
    def for_iter(self) -> Optional['ConfigData']:
        return self.parent_for_iter if self.skip_for_iter else self

    @property
    def path(self) -> str:
        if (parent := self.parent) is not None:
            return f'{parent.path}.{self.step}'
        return self.step

    def iterable(self, *keys: str) -> Iterable['ConfigData']:
        return ConfigIterable(self, keys)

    def list(self, *keys: str) -> list[Any]:
        return list(self.iter(*keys))

    def local_keys(self) -> set[str]:
        keys = set()
        for values in self.iter_locals():
            keys.update(values)
        return keys

    def iter(self, *keys: str, inherit: str|bool = None) -> Iterator[Any]:

        # print(f'{self.path}: iterating over {keys} with local {self.local_keys()}')
        yield from self.local(*keys)

        for fallback in self.fallback:
            if isinstance(fallback, ConfigData):
                yield from fallback.iter(*keys)
            elif isinstance(fallback, dict):
                for key in keys:
                    if key in fallback:
                        yield fallback[key]

        if inherit is None or inherit is True:
            inherit = keys[0] if keys else None

        # We check for local defaults before we go up to the parents
        if defaults := self.defaults:
            for key in keys:
                if key in defaults:
                    yield defaults[key]

        step = self.step
        if (parent := self.parent_for_iter) is not None:
            yield from parent.iter(*(f'{step}.{key}' for key in keys), inherit=False)

        if inherit:
            if (parent := self.parent_for_iter) is not None:
                yield from parent.iter(inherit)

    def iter_locals(self) -> Iterator[dict[str, Any]]:
        return iter(self.locals)

    def local(self, *keys: str) -> Iterator[Any]:

        for key in keys:
            for values in self.iter_locals():
                val = values.get(key, ...)
                if val is not ...:
                    yield val

            if matchers := self.config._config_matchers_for_key(key, self.get_matchers()):
                for matcher in matchers:
                    if matcher.matches(key, config=self.config):
                        yield matcher.value

    def get_local_or_default(self, *keys: str, default: Any = None) -> Any:
        for val in self.local(*keys):
            return val
        if defaults := self.defaults:
            for key in keys:
                if key in defaults:
                    return defaults[key]
        return default

    # noinspection PyShadowingNames
    def get_value_of_field(self, field: 'ConfigField', key: str|None, default: Any) -> Any:
        if key is None: key = field.name
        if key in self.cache:
            val = self.cache[key]
            if val is not use_default:
                return val
        else:
            if field.is_config:
                val = self.get_local_or_default(key, *field.aliases, default=...)
                if val is None:
                    self.cache[key] = val
                    return val

                fallback = self.iterable(key, *field.aliases)
                values = tuple(self.local(key))
                if not values and field.type.optional:
                    fallback = list(fallback)
                    if not fallback:
                        return field.default

                log.debug('config {}: making {} for {}: {}', self.path, field.type.qname, key, fallback)
                val = field.make_config(*values, step=key, fallback=fallback, parent=self)
                self.cache[key] = val
                return val
            for val in self.iter(key, *field.aliases, inherit=field.inherit):
                self.cache[key] = val
                return val
            self.cache[key] = use_default
        if default is None:
            if self.defaults:
                return self.defaults.get(key, field.default)
            return field.default
        return default

    def get_field(self, key: str) -> Optional['ConfigField']:
        # noinspection PyProtectedMember
        return self.config._config_get_field(key)

    def get_config(self, key: str, config_class: type[C] = None) -> C:
        if config_class is None: config_class = Config
        if f := self.get_field(key):
            config = f.get_value(self, key, missing)
            if config is missing:
                return config_class.empty()
        else:
            config = config_class.empty()
            for config in self.iter(key):
                break

        if config is None:
            return config_class.empty()
        elif isinstance(config, Config):
            if config_class is Config or isinstance(config, config_class):
                return config
            raise TypeError(
                f'Config value {config} is not a {config_class.qname} (got {type(config).__name__})'
            )
        elif isinstance(config, dict):
            return config_class.construct(config)
        elif isinstance(config, ConfigData):
            return config.config
        raise TypeError(f'Config value {config} is not a Config')

    def get(self, *keys: str, default: Any = None) -> Any:
        for key in keys:
            dot = key.find('.')
            if dot >= 0:
                prefix = key[:dot]
                suffix = key[dot+1:]
                intermediate = self.get_config(prefix)
                val = intermediate.get(suffix, default=...)
                if val is not ...:
                    return val
            else:
                if f := self.get_field(key):
                    return f.get_value(self, key, default)
                for val in self.iter(key):
                    return val
        return default

    def put(self, key: str, value: Any) -> None:
        if values := self.locals:
            values[0][key] = value
        else:
            self.locals = ({key: value}, )

    def to_dict(self, *keys: str) -> dict[str, Any]:
        return {key: self.get(key) for key in keys}

    def set_defaults(self, defaults: dict[str, Any]) -> Self:
        self.defaults = defaults.copy()
        return self

    def add_defaults(self, defaults: dict[str, Any]) -> Self:
        if defaults:
            if self.defaults:
                self.defaults.update(defaults)
            else:
                self.defaults = defaults.copy()
        return self

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def _repr_args(self) -> str:
        return self.path

    def _repr_kwargs(self, verbose: int = 0) -> Optional[dict[Optional[str], Any]]:
        kwargs = {}
        if verbose > 0:
            for values in self.iter_locals():
                for key, val in values.items():
                    kwargs.setdefault(key, val)
        return kwargs

    @classmethod
    def construct(cls, *args: dict[str, Any], values: Sequence[dict[str, Any]] = None, **kwargs) -> Self:
        if values is None:
            values = args
        elif args:
            values = args + tuple(values)
        return cls(values=values, **kwargs)


class ConfigField(Field):

    __slots__ = ('inherit', 'aliases')

    inherit: Optional[str]
    aliases: Sequence[str]

    slots = meta.Field.slots | set(__slots__)

    def __init__(self, owner: 'Meta', name: str, spec: Spec):
        if 'default' not in spec:
            field_type = spec['type']
            member = spec.get('member')
            if member is not None:
                if field_type.has_instance(member):
                    spec['default'] = member
        super().__init__(owner, name, spec)
        self.inherit = spec.get('inherit')
        self.aliases = ()

    @property
    def is_config(self) -> bool:
        return self.type.is_subclass(Config)

    def build_inherit(self, spec: Spec) -> str:
        return self.name

    def build_set(self, spec: Spec) -> Setter:
        name = self.name
        if name.startswith('_'):
            return super().build_set(spec)
        def setter(this: 'Config', val: Any) -> None:
            this.put(name, val)
        return name_function(setter, f'poke_config[{self.name}]')

    def build_get(self, spec: Spec) -> Getter:
        def getter(this: 'Config'):
            return self.get_value(this._config_data)
        return name_function(getter, f'get_config[{self.name}]')

    def get_value(self, config: 'ConfigData', key: str = None, default: Any = None) -> Any:
        # if key is not None:
        #     print(f'Hmmmm: {self} != {key}')
        return config.get_value_of_field(self, key, default)
        # if key is None:
        #     key = self.name
        # if self.is_config:
        #     fallback = config.iterable(key, *self.aliases)
        #     log.debug('config {}: making {} for {}: {}', config.path, self.type.qname, key, fallback)
        #     return self.make_config(*config.local(key), step=key, fallback=fallback, parent=config)
        # for val in config.iter(key, *self.aliases, inherit=self.inherit):
        #     return val
        # return self.default if default is None else default

    def make_config(self, *values, **kwargs) -> 'Config':
        if cls := self.type.get_subclass(Config):
            return cls.construct(*values, **kwargs)
        raise TypeError(f'Field {self.name} is not a Config')

    @classmethod
    def build(cls, owner: 'Meta', name: str, spec: Spec) -> Optional[Field]:
        if name[0] == '_':
            return Field(owner, name, spec)
        impl = cls
        if field_type := spec.get('type'):
            if field_cls := field_type.cls:
                if getattr(field_cls, '_is_list', False):
                    impl = ListConfigField
                elif issubclass(field_cls, Sequence):
                    if field_type_args := field_type.args:
                        field_type_item = field_type_args[0]
                        if issubclass(field_type_item.cls, Config):
                            impl = ListConfigField
                        # print(f'Got a list field type: {field_type_item!r}')

        return impl(owner, name, spec)

    attr_builders: ClassVar[dict[str, Callable[['Field', Spec], Any]]] = {
        **Field.attr_builders,
        **method_builders('inherit'),
    }

class ConfigMeta(ObjectMeta):

    __slots__ = ()

    Field = ConfigField


class ListConfigField(ConfigField):

    config_cls: type['ListConfig']
    item_type: type
    item_field: ConfigField

    def __init__(self, owner: 'Meta', name: str, spec: Spec):
        super().__init__(owner, name, spec)
        if field_type := self.type:
            if issubclass(field_type.cls, Sequence):
                self.config_cls = ListConfig
            else:
                self.config_cls = field_type.get_subclass(ListConfig)
            item_type = field_type.args[0]
            self.item_type = item_type.cls
            self.item_field = ConfigField(self.owner, '_', Spec(type=item_type))
        else:
            raise ValueError('Must have a field type')
            # self.item_type = None
            # self.item_field = None

    @property
    def is_config(self) -> bool:
        return True

    # def get_value(self, config: 'ConfigData', key: str = None, default: Any = None) -> Any:
    #     return super().get_value(config, key, default)

    def make_config(self, *values, **kwargs) -> 'Config':
        # if cls := self.type.get_subclass(ListConfig):
        if cls := self.config_cls:
            return cls.construct(*values, **kwargs, item_field=self.item_field)
        raise TypeError(f'Field {self.name} is not a ListConfig')


C = TypeVar('C', bound='Config')


class Config(Object):

    __slots__ = ('_config_data',)

    _config_data: Annotated[ConfigData, meta.field(
        doc='The actual data for this configuration',
        required=True,
    )]
    _config_default_step: ClassVar[Annotated[str, meta.field(
        doc="The default step name for this class of configuration",
        ignore=True,
    )]] = 'config'
    _config_template_matchers: ClassVar[Annotated[Sequence[type[KeyMatcher]], meta.field(
        doc='The template matchers for this class of configuration',
        ignore=True,
    )]] = default_matchers

    def postinit(self, spec: Spec):
        super().postinit(spec)
        data = self._config_data
        if data.config is None:
            data.config = self

    def override(self, **kwargs) -> Self:
        if kwargs:
            return self.__class__(_config_data=ConfigData.override(**kwargs))
        return self

    def _config_get_field(self, name: str) -> Optional[ConfigField]:
        return self.meta.fields.get(name)

    @classmethod
    def _config_matchers_for_key(cls, key: str, matchers: Sequence[KeyMatcher]) -> Sequence[KeyMatcher]:
        return matchers

    def get_keys(self) -> Iterable[str]:
        for name, field in self.meta.fields.items():
            if field.scope is Scope.instance_scope and isinstance(field, ConfigField):
                yield name

    def get_first(self, *keys: str, default: Any = ...) -> Any:
        return self._config_data.get(*keys, default=default)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config_data.get(key, default=default)

    def get_config(self, key: str, config_class: type[C] = None) -> C:
        return self._config_data.get_config(key, config_class=config_class)

    def put(self, key: str, value: Any) -> None:
        self._config_data.put(key, value)

    def to_dict(self, *keys: str) -> dict[str, Any]:
        if not keys:
            keys = self.get_keys()
        return {key: self.get(key) for key in keys}

    def set_defaults(self, **defaults) -> Self:
        if defaults:
            self._config_data.set_defaults(defaults)
        return self

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def _repr_args(self) -> str:
        return self._config_data.path

    def _repr_kwargs(self, verbose: int = 0) -> Optional[dict[Optional[str], Any]]:
        return self._config_data._repr_kwargs(verbose)

    @classmethod
    def construct(cls, *values: dict[str, Any], step: str = None, parent: Union['Config', ConfigData] = None, **kwargs) -> Self:
        if step is None: step = cls._config_default_step
        if isinstance(parent, Config): parent = parent._config_data
        return cls(_config_data=ConfigData.construct(values=values, step=step, parent=parent, **kwargs))

    @classmethod
    def empty(cls) -> Self:
        if '_config_empty' not in cls.__dict__:
            setattr(cls, '_config_empty', cls(_config_data=ConfigData()))
        return cls._config_empty

    _config_empty: ClassVar['Config']

    Meta = ConfigMeta


digits_pat = re.compile(r'^\d+$')



class ListConfig(Sequence[C], Config, metaclass=ABCClass):

    __slots__ = ('count', '_config_item_field')

    count: Annotated[int, meta.field(
        doc='The number of items in the list',
        inherit=False)
    ]
    _config_item_field: Annotated[ConfigField, meta.field(
        doc='The field for the items of the list'
    )]
    _config_template_matchers = list_matchers

    def _config_get_field(self, name: str) -> Optional[ConfigField]:
        f = self.meta.fields.get(name)
        return self._config_item_field if f is None else f

    @classmethod
    def _config_matchers_for_key(cls, key: str, matchers: Sequence[KeyMatcher]) -> Sequence[KeyMatcher]:
        if digits_pat.match(key):
            return matchers
        return ()

    def __len__(self):
        return self.count or 0

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            key = str(key)
        return Config.__getitem__(self, key)

    def _repr_type_args(self) -> str:
        return self._config_item_field.type.qname

    @classmethod
    def construct(cls, *values: dict[str, Any], item_field: ConfigField = None, **kwargs) -> Self:
        if item_field is None:
            raise  ValueError('Must have an item field')
        kwargs.setdefault('skip_for_iter', True)
        return cls(_config_data=ConfigData.construct(values=values, **kwargs), _config_item_field=item_field)


# if True or __name__ == '__main__':
#
#     class HeadConfig(Config):
#
#         foo: Annotated[str, field(doc='The foo')] = 'queef'
#         bar: Annotated[str, field(doc='The bar')]
#
#
#     class TailConfig(Config):
#
#         grunt: str
#         groan: str
#
#
#     class LayerConfig(Config):
#
#         desc: Annotated[str, field(doc='The description')]
#         input_dim: int
#         output_dim: int
#         hidden_dim: int
#
#
#     class TestConfig(Config):
#
#         size: int
#         foo: str
#         head: HeadConfig
#         tail: TailConfig
#         layers: ListConfig[LayerConfig]
#
#     def do_test():
#
#         # x = Dict(bar=12, blue=7)
#         # foo(**x)
#         # exit(0)
#
#         # for name, cls in Config._config_types.items():
#         #     print(f'{name}: {cls}')
#         print('-' * 80)
#
#         test_spec = {
#             'kind': 'llm',
#             'size': 10,
#             # 'foo': 'boo',
#             'bar': 'skiz',
#             'groan': 'away',
#             'output_dim': 256,
#             'head': {
#                 # 'foo': 'bar',
#                 'bar': 'baz'
#             },
#             'tail': {
#                 'grunt': 'groan'
#             },
#             'layers': {
#                 'count': 6,
#                 '0': {
#                     'desc': 'layer 0',
#                     'hidden_dim': 150
#                 },
#                 '/[12]/': {
#                     'desc': 'layer 1 or 2',
#                     'output_dim': 128,
#                 },
#                 '3:-1': {
#                     'desc': 'layer 3...5',
#                     'input_dim': 64,
#                 },
#                 '%3=1': {
#                     'desc': 'layer mod 3 == 1',
#                 },
#                 '%3=1|%3=2&4:': {
#                     'desc': 'layer mod 3 == 1, 2 and 4:',
#                 },
#                 '*': {
#                     'kind': 'transformer',
#                     'desc': 'layer default',
#                     'input_dim': 100,
#                     'hidden_dim': 50
#                 }
#             }
#         }
#
#         test_config = TestConfig.construct(test_spec, step='test')
#
#         def print_item(config: Config, key: str = None):
#             if key is None:
#                 print(f'{config._config_data.path}: {config}')
#             else:
#                 print(f'{config._config_data.path}.{key}: {config[key]}')
#
#         def print_attr(config: Config, key: str = None):
#             if key is None:
#                 print(f'{config._config_data.path}: {config}')
#             else:
#                 print(f'{config._config_data.path}.{key}: {getattr(config, key)}')
#
#         print_item(test_config)
#
#         for key in ['size', 'foo', 'bar', 'head']:
#             print_item(test_config, key)
#
#         # print_attr(test_config, 'size')
#         print(f'test.size: {test_config.size}')
#
#         head_config = test_config.head
#         for key in ['foo', 'bar']:
#             print_item(head_config, key)
#
#         layers = test_config.layers
#         print_item(layers)
#
#         for key in ['count']:
#             print_item(layers, key)
#
#         layer_0 = layers[0]
#         print_item(layer_0)
#         print('-'*80)
#         for key in layer_0.keys():
#             print_item(layer_0, key)
#         print('-'*80)
#
#         layer_1 = layers[1]
#         print_item(layer_1)
#         print(f'*test.layers.1.desc: {layer_1._config_data.list("desc")}')
#
#         n_layers = layers.count
#         for i in range(n_layers):
#             layer = layers[i]
#             for key in ['input_dim', 'output_dim', 'hidden_dim', 'desc']:
#                 print_item(layer, key)
#                 # print(f'test.layer.0.{key}: {layer_0[key]}')
#
#         for key in ['size', 'foo', 'head.foo', 'head.bar', 'tail.grunt', 'tail.groan', 'layers.1.hidden_dim', 'layers.5.desc']:
#             print_item(test_config, key)
#             # print(f'test.{key}: {test_config[key]}')
#
#         print('scoop:', layers[5].desc)
#
#         exit(0)
#     do_test()