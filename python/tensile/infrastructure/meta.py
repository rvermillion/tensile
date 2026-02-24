#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from typing import Generic

from .types import *
from .functional import *
from .root import *
from .field import FieldType
from .util import class_qname, process_specs


if TYPE_CHECKING:
    import tensile.infrastructure


class MetaError(RuntimeError):

    pass


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


if TYPE_CHECKING:
    # noinspection PyUnusedLocal
    def field(
        required: bool = False,
        readonly: bool = False,
        default: Any = ...,
        default_factory: Callable[[], Any] = ...,
        init: Optional[bool] = ...,
        doc: str = ...,
        inherit: str|bool = ...,
        ignore: bool = ...,
        **kwargs
    ) -> Keywords: ...
else:
    def field(**kwargs) -> Keywords:
        return kwargs


# cached_field_types: dict[Any, 'FieldType'] = {}
#
#
# class FieldType(RootObject):
#
#     __slots__ = ['anno', 'cls', 'qname', 'args', 'optional']
#
#     anno: Any
#     cls: Optional[type]
#     qname: str
#     args: tuple['FieldType', ...]
#     optional: bool
#
#     def __init__(self, anno: Any, cls: type = None, qname: str = None, args: tuple['FieldType', ...] = (),
#                  optional: bool = False):
#         if cls is None:
#             if qname is None:
#                 if anno is not ...:
#                     raise ValueError(f'cls or qname must be specified for {anno!r}')
#             else:
#                 add_qname_listener(qname, self.qname_registered)
#         else:
#             if not isinstance(cls, type):
#                 raise ValueError(f'cls must be a class or None: {cls}')
#             if qname is None:
#                 qname = class_qname(cls)
#
#         self.anno = anno
#         self.cls = cls
#         self.qname = qname
#         self.args = args
#         self.optional = optional
#
#         if not isinstance(anno, str) or '.' in anno:
#             # print(f'// caching field type for {anno!r}')
#             cached_field_types[anno] = self
#
#     def qname_registered(self, meta: 'Meta'):
#         if self.cls is None:
#             self.cls = meta.cls
#         else:
#             if self.cls is not meta.cls:
#                 raise MetaError(f'cannot register {meta} with qname {self.qname} because it is already registered as {self.cls}')
#
#     def has_instance(self, obj: Any) -> bool:
#         if cls := self.cls:
#             return is_runtime_class(cls) and isinstance(obj, cls)
#         return False
#
#     def is_subclass(self, sup: type) -> bool:
#         if cls := self.cls:
#             return issubclass(cls, sup)
#         return False
#
#     def get_subclass(self, sup: type[T]) -> Optional[type[T]]:
#         cls = self.cls
#         return cls if cls is not None and issubclass(cls, sup) else None
#
#     @property
#     def coerce(self) -> Optional[Coercer]:
#         return coerce_type(self.cls, optional=self.optional)
#
#     @property
#     def equiv(self) -> Equiv:
#         return eq_equiv
#
#     @property
#     def is_unique(self) -> bool:
#         if cls := self.cls:
#             return issubclass(cls, AbstractSet)
#         return False
#
#     @property
#     def is_sequence(self) -> bool:
#         if cls := self.cls:
#             return issubclass(cls, Sequence) and not issubclass(cls, str)
#         return False
#
#     @property
#     def is_mapping(self) -> bool:
#         if cls := self.cls:
#             return issubclass(cls, Mapping)
#         return False
#
#     @property
#     def is_multivalued(self) -> bool:
#         if cls := self.cls:
#             return issubclass(cls, Collection) and not issubclass(cls, str)
#         return False
#
#     @property
#     def update(self) -> Optional[Callable[[Any, Any], Any]]:
#         return None
#
#     def is_subclass_of(self, sup: Union[type, 'FieldType']) -> bool:
#         if cls := self.cls:
#             if isinstance(sup, type):
#                 return issubclass(cls, sup)
#             if isinstance(sup, FieldType) and (sup_cls := sup.cls):
#                 return issubclass(cls, sup_cls)
#         return False
#
#     def to_spec(self, spec: Spec):
#         if self.optional:
#             spec['optional'] = True
#         if self.is_multivalued:
#             spec['multivalued'] = True
#         if self.is_unique:
#             spec['unique'] = True
#         if self.is_sequence:
#             spec['sequence'] = True
#
#     def _repr_type(self) -> str:
#         return 'FieldType'
#
#     def _repr_args(self) -> str:
#         s = str(self)
#         # if self.is_multivalued:
#         #     s += ', +multivalued'
#         # if self.is_unique:
#         #     s += ', +unique'
#         # if self.is_sequence:
#         #     s += ', +sequence'
#         return s
#
#     # noinspection PyMethodMayBeStatic
#     def _show_args(self, args: tuple['FieldType', ...]) -> str:
#         return '[' + ', '.join(str(arg) for arg in args) + ']'
#
#     def __str__(self) -> str:
#         t = self.qname or str(self.anno)
#         if args := self.args:
#             t += self._show_args(args)
#         if self.optional:
#             return t + '?'
#         return t
#
#     @classmethod
#     def from_anno(cls, anno: Any, owner: type) -> 'FieldType':
#         if isinstance(anno, cls):
#             return anno
#
#         try:
#             impl = FieldType
#
#             if isinstance(anno, str):
#                 if '.' in anno:
#                     if field_type := cached_field_types.get(anno):
#                         # print(f'// using cached field type for {anno!r}')
#                         return field_type
#                 return impl(anno, None, anno)
#
#             if field_type := cached_field_types.get(anno):
#                 # print(f'// using cached field type for {anno!r}')
#                 return field_type
#
#             full_anno = anno
#             origin = get_origin(anno)
#
#             qname = None
#             args = ()
#             optional = False
#
#             if origin is Union:
#                 unions = set(get_args(anno))
#                 if NoneType in unions:
#                     if len(unions) == 2:
#                         unions.discard(NoneType)
#                         anno, = unions
#                         origin = get_origin(anno)
#                         optional = True
#
#             if isinstance(anno, str):
#                 qname = anno
#                 origin = None
#             elif isinstance(anno, type):
#                 qname = class_qname(anno)
#                 origin = anno
#             # elif is_generic_alias(anno):
#             #     args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
#             #     qname = class_qname(origin)
#             elif is_forward_ref(anno):
#                 qname = get_forward_ref_name(anno)
#                 if '.' in qname:
#                     if qname == class_qname(owner):
#                         origin = owner
#                     elif meta := meta_by_qname.get(qname):
#                         origin = meta.cls
#                 else:
#                     if qname == owner.__qualname__:
#                         qname = class_qname(owner)
#                         origin = owner
#                     else:
#                         qname = owner.__module__ + '.' + qname
#                         if meta := meta_by_qname.get(qname):
#                             origin = meta.cls
#             elif anno is ...:
#                 pass
#             elif origin is type:
#                 args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
#                 qname = class_qname(origin)
#             elif isinstance(origin, type):
#                 if is_protocol(origin):
#                     # if getattr(origin, '_is_protocol', False):
#                     args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
#                     qname = class_qname(origin)
#                 elif issubclass(origin, Callable):
#                     if raw_args := get_args(anno):
#                         params, ret = raw_args
#                         arg_list = [FieldType.from_anno(params, owner)] if params is ... else [FieldType.from_anno(arg, owner) for arg in params]
#                         arg_list.append(FieldType.from_anno(ret, owner))
#                         args = tuple(arg_list)
#                     qname = class_qname(origin)
#                     impl = CallableFieldType
#                 elif issubclass(origin, Collection):
#                     args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
#                     impl = CollectionFieldType
#                 elif is_generic_alias(anno):
#                     args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
#                     qname = class_qname(origin)
#             elif origin is Union:
#                 impl = UnionFieldType
#                 args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
#                 qname = class_qname(origin)
#                 # print(f'Problem processing union annotation: {anno} with args {args}')
#                 origin = object
#             else:
#                 print(f'Problem processing annotation: {anno} with origin {origin}')
#                 raise MetaError(f'Problem processing annotation: {anno} with origin {origin}')
#
#             return impl(full_anno, origin, qname=qname, args=args, optional=optional)
#         except Exception:
#             raise MetaError(f'Error getting field type from annotation: {anno!r}')
#
#
# class CallableFieldType(FieldType):
#
#     __slots__ = ()
#
#     def __str__(self) -> str:
#         if args := self.args:
#             if len(args) > 2:
#                 return '(' + ', '.join(str(arg) for arg in args[:-1]) + f') => {args[-1]}'
#             return ' => '.join(str(arg) for arg in args)
#         return '... => Any'
#
#
# class UnionFieldType(FieldType):
#
#     __slots__ = ()
#
#     def __init__(
#         self, anno: Any, cls: type = None, qname: str = None, args: tuple['FieldType', ...] = (),
#         optional: bool = False):
#         if len(args) < 2:
#             raise ValueError(f'UnionFieldType requires at least two args: {args}')
#         qname = 'Union[' + ', '.join(arg.qname for arg in args) + ']'
#         super().__init__(anno, cls, qname, args, optional)
#
#     @property
#     def coerce(self) -> Optional[Coercer]:
#         return None
#
#     def __str__(self) -> str:
#         return self.qname
#
#
# class CollectionFieldType(FieldType):
#
#     __slots__ = ()
#
#     # noinspection PyMethodOverriding
#     def update(self, old: Any, new: Iterable[Any]) -> Any:
#         if old is None:
#             return new
#         if isinstance(old, MutableSequence):
#             old.clear()
#             old.extend(new)
#             return old
#         return self.cls(new)
#


class Scope(Enum):

    instance_scope = 'instance'
    class_scope = 'class'

    @staticmethod
    def is_instance(scope: 'Scope') -> bool:
        return scope is Scope.instance_scope

    @staticmethod
    def is_class(scope: 'Scope') -> bool:
        return scope is Scope.class_scope


class Visibility(Enum):

    public = 'public'
    protected = 'protected'
    private = 'private'


field_attr_defaults: dict[str, Any] = {
    'aliases': (),
    'doc': '',
    'default': None,
    'default_factory': None,
    'delegate': None,
    'member': missing,
    'required': False,
    'readonly': False,
    'scope': Scope.instance_scope,
    # 'slot': None,
    'visibility': Visibility.public,
}

def default_builder(default: Any) -> Callable[['Field', Spec], Any]:
    return lambda f, spec: default


field_attr_default_builders: dict[str, Callable[['Field', Spec], Any]] = {
    attr: default_builder(dflt) for attr, dflt in field_attr_defaults.items()
}


def build_peek(member: Any = None, slot: str = None, desc: str = '') -> Optional[Getter]:
    if isinstance(member, MemberDescriptorType):
        return member.__get__
    if slot:
        return attr_getter(slot)
    return None


def build_poke(member: Any = None, slot: str = None, desc: str = '') -> Optional[Setter]:
    if isinstance(member, MemberDescriptorType):
        return member.__set__
    if slot:
        return attr_setter(slot)
    return None

def build_delete(member: Any = None, slot: str = None, desc: str = '') -> Optional[Deleter]:
    if isinstance(member, MemberDescriptorType):
        return member.__delete__
    if slot:
        return attr_deleter(slot)
    return None

def build_lazy(name: str = None, check_cls: type = None, desc: str = '') -> Optional[Getter]:
    lazy_method = f'_lazy_{name}'
    if check_cls is None or callable(getattr(check_cls, lazy_method, None)):
        return method_getter(method=lazy_method)
    return None


# noinspection PyShadowingNames
def build_coerce(name: str = None, check_cls: type = None, field_type: FieldType = None, desc: str = '', required: bool = None) -> Optional[Coercer]:
    coerce_method = f'_coerce_{name}'
    if check_cls is None:
        return method_coercer(method=coerce_method)
    elif callable(getattr(check_cls, coerce_method, None)):
        return method_coercer(method=coerce_method)
    else:
        coerce_method = f'_{name}_coerce'
        if callable(getattr(check_cls, coerce_method, None)):
            return method_coercer(method=coerce_method)
    if field_type is not None:
        coerce = field_type.get_coerce()
        if coerce is None:
            # log.warn('no coerce method for field [{}] of type {!r}', name, field_type)
            return None
        elif required is False and not field_type.optional:
            def coerce_maybe(this, val):
                return None if val is None else coerce(this, val)
            return coerce_maybe
        return coerce
    return None

def build_getter(peek: Getter = None, lazy: Getter = None, poke: Setter = None, desc: str = '') -> Optional[Getter]:
    getter: Getter

    if lazy:

        if poke is None:
            raise MetaError(f'build_getter{desc}: cannot build getter with lazy but no poke')

        def getter(this: Any) -> Any:
            val = peek(this)
            if val is None:
                val = lazy(this)
                poke(this, val)
            return val

    else:
        getter = peek

    return getter


def readonly_setter(desc: str = '') -> Setter:
    if not desc:
        desc = 'readonly'

    # noinspection PyUnusedLocal
    def setter(this: Any, val: Any):
        raise TypeError(f'{desc}: cannot set readonly field in {this!r}')
    return setter


def build_setter(poke: Setter = None, coerce: Coercer = None, readonly: bool = False, desc: str = '') -> Optional[Setter]:
    if readonly:
        return readonly_setter(desc=desc)

    if poke is None:
        return None
    if coerce is None:
        return poke

    # if poke is None:
    #     raise MetaError(f'build_setter({desc}): cannot build setter with coerce but no poke')

    def setter(this: Any, val: Any):
        try:
            coerced = coerce(this, val)
        except Exception as e:
            raise TypeError(f'{desc}: cannot coerce {val!r} in {this!r}') from e
        try:
            poke(this, coerced)
        except Exception as e:
            raise TypeError(f'{desc}: cannot set to {coerced!r} in {this!r}') from e

    return setter

def build_property(peek: Getter = None, poke: Setter = None, lazy: Getter = None, coerce: Coercer = None,
                   delete: Deleter = None, doc: str = None, desc: str = '') -> property:
    getter = build_getter(peek=peek, lazy=lazy, poke=poke, desc=desc)
    setter = build_setter(poke=poke, coerce=coerce, desc=desc)

    return property(getter, setter, delete, doc)


def wrap_slot(cls: type, name: str, lazy: Getter = None, coerce: Coercer = None, doc: str = None, desc: str = '') -> None:
    if not desc:
        desc = f'({name})'
    member = getattr(cls, name)
    if isinstance(member, MemberDescriptorType):
        if lazy is None and coerce is None:
            print(f'wrap_slot{desc}: not wrapping slot {name} with no behaviors')
        else:

            prop = build_property(peek=member.__get__, poke=member.__set__, lazy=lazy, coerce=coerce,
                                  delete=member.__delete__, doc=doc, desc=f'({name})')

            setattr(cls, name, prop)
    else:
        raise MetaError(f'wrap_slot{desc}: no slot [{name}] in class {cls}')


def method_builder(name: str) -> Callable[['Field', Spec], Any]:
    method = 'build_' + name
    def build(this: 'Field', spec: Spec) -> Any:
        return getattr(this, method)(spec)
    return build


def method_builders(*names: str) -> dict[str, Callable[['Field', Spec], Any]]:
    return {name: method_builder(name) for name in names}


class Field(RootObject):

    __slots__ = ['owner', 'name', 'spec', 'type', 'aliases', 'slot',
                 'peek', 'lazy', 'get', 'poke', 'coerce', 'write',
                 'set', 'delete', 'init', 'is_set', 'update',
                 'default', 'default_factory', 'required', 'readonly',
                 'doc', 'scope', 'member', 'visibility', 'delegate']

    owner: 'Meta'
    name: str
    spec: Spec
    type: FieldType
    slot: Optional[str]
    aliases: tuple[str, ...]
    peek: Getter
    lazy: Optional[Getter]
    get: Getter
    poke: Setter
    is_set: IsSetter
    coerce: Optional[Coercer]
    write: Setter
    set: Setter
    update: Setter
    delete: Optional[Deleter]
    init: Optional[Initter]
    delegate: Optional[str]
    doc: str
    member: Any
    default: Any
    default_factory: Callable[[], Any]
    required: bool
    readonly: bool
    scope: Scope
    visibility: Visibility

    def __init__(self, owner: 'Meta', name: str, spec: Spec):
        self.owner = owner
        self.name = name
        self.spec = spec

        # if owner.name == 'NamedObject':
        #     print(f'-- creating field {owner}.{name} with spec: {spec}')
        # spec_with_defaults = Spec.combine(field_defaults, spec)

        for key, val in spec.items():
            setattr(self, key, val)

        for key, builder in self.attr_builders.items():
            if key in spec:
                pass
            else:
                val = builder(self, spec)
                setattr(self, key, val)

        spec.pop('type', None)

    @property
    def qname(self) -> str:
        return self.owner.qname + '.' + self.name

    @property
    def equiv(self) -> Equiv:
        return self.type.equiv

    def has_methods(self, *methods) -> bool:
        cls = self.owner.cls
        for pfx in methods:
            method = f'_{pfx}_{self.name}'
            if callable(getattr(cls, method, None)):
                return True
        return False

    def needs_property(self) -> bool:
        if isinstance(self.member, property):
            return False
        if self.coerce or self.lazy:
            return True
        if self.slot is not None:
            return True
        if self.has_methods('get', 'set'):
            return True
        return False

    def build_property(self, cls: builtins.type) -> Optional[property]:
        if self.needs_property():
            return property(self.get, self.set, self.delete, self.doc)
        return None

    def engineer(self, meta: 'Meta' = None) -> None:
        if meta is None:
            meta = self.owner
        if Scope.is_instance(self.scope):
            cls = meta.cls
            if prop := self.build_property(cls):
                self.debug('-- adding {} field with property: {} --> {}', meta, self, prop)
                setattr(cls, self.name, prop)
            else:
                self.debug('-- adding {} field: {}', meta, self)
        else:
            self.debug('-- skipping {} field: {}', meta, self)

    def override(self, meta: 'Meta', spec: Spec) -> 'Field':
        cls = spec.get('field_class', meta.Field)
        return cls(meta, self.name, spec)

    def to_spec(self, spec: Spec):
        pass

    @classmethod
    def new_spec(cls, **kwargs) -> Spec:
        return Spec(**kwargs) if kwargs else Spec()

    def describe(self) -> str:
        s = f'{self.name}: {self.type}'
        if Scope.is_class(self.scope):
            return '#' + s
        return s

    def build_peek(self, spec: Spec) -> Optional[Getter]:
        return build_peek(member=self.member, slot=self.slot)

    def build_lazy(self, spec: Spec) -> Optional[Getter]:
        return build_lazy(name=self.name, check_cls=self.owner.cls)

    def build_get(self, spec: Spec) -> Getter:
        if isinstance(self.member, property):
            return self.member.fget
        getter_method = f'_get_{self.name}'
        if getattr(self.owner.cls, getter_method, None):
            return method_getter(method=getter_method)
        if delegate := self.delegate:
            name = self.name
            dot = delegate.find('.')
            if dot >= 0:
                delegate, name = delegate[:dot], delegate[dot + 1:]
            def getter(this: Any):
                getattr(getattr(this, delegate), name)
            return getter
        lazy = self.lazy
        getter = build_getter(peek=self.peek, lazy=lazy, poke=self.poke)
        if getter is None:
            def getter(this: Any) -> Any:
                raise AttributeError(f'Field {self.qname} of {this} has no getter')
        # elif lazy:
        #     getter = safe_getter(getter, f'Error getting field [{self.qname}]')
        return getter

    def build_is_set(self, spec: Spec) -> IsSetter:
        return peek_is_setter(peek=self.peek, default=self.default, equiv=self.equiv)

    def build_poke(self, spec: Spec) -> Setter:
        return build_poke(member=self.member, slot=self.slot, desc=self.qname)

    def build_coerce(self, spec: Spec) -> Optional[Coercer]:
        return build_coerce(self.name, check_cls=self.owner.cls, field_type=self.type, desc=self.qname,
                            required=self.required)

    def build_update(self, spec: Spec) -> Setter:
        if Scope.is_instance(self.scope):
            if self.readonly:
                return readonly_setter(desc=self.qname)
            field_type = self.type
            if field_update := field_type.update:
                peek = self.peek
                poke = self.poke
                def update(this: Any, value: Any):
                    old = peek(this)
                    poke(this, field_update(old, value))

                return update
        return self.set

    def build_write(self, spec: Spec) -> Optional[Setter]:
        if isinstance(self.member, property):
            return self.member.fset

        setter_method = f'_set_{self.name}'
        if getattr(self.owner.cls, setter_method, None):
            return method_setter(method=setter_method)

        if delegate := self.delegate:
            name = self.name
            dot = delegate.find('.')
            if dot >= 0:
                delegate, name = delegate[:dot], delegate[dot + 1:]
            def setter(this: Any, value: Any):
                setattr(getattr(this, delegate), name, value)
            return setter
        if Scope.is_instance(self.scope):
            setter = build_setter(poke=self.poke, coerce=self.coerce, desc=self.qname)
            # if setter is None:
            #     def setter(this, value):
            #         raise AttributeError(f'Field {self.qname} of {this} has no setter')
            return setter
        return None

    def build_set(self, spec: Spec) -> Optional[Setter]:
        if self.readonly:
            return readonly_setter(desc=self.qname)
        writer = self.write
        if writer is None:
            def writer(this, value):
                raise AttributeError(f'Field {self.qname} of {this} has no setter')
        else:
            writer = safe_setter(writer, f'Error setting field [{self.qname}]')
        return writer

    def build_init(self, spec: Spec) -> Optional[Initter]:
        if self.readonly and isinstance(self.member, property):
            return None
        return name_initter(self.name, self.aliases, writer=self.write,
                            default=self.default,
                            default_factory=self.default_factory)

    def build_delete(self, spec: Spec) -> Optional[Deleter]:
        if isinstance(self.member, property):
            return self.member.fdel
        return build_delete(member=self.member, slot=self.slot, desc=self.qname)

    def build_slot(self, spec: Spec) -> Optional[str]:
        return None if isinstance(self.member, MemberDescriptorType) else '_' + self.name
        # return '_' + self.name

    attr_builders: ClassVar[dict[str, Callable[['Field', Spec], Any]]] = {
        **field_attr_default_builders,
        **method_builders(
            'slot', 'peek', 'poke', 'lazy', 'coerce',
            'write', 'get', 'set', 'delete', 'update', 'init'),
        # 'slot': build_slot,
        # 'peek': build_peek,
        # 'poke': build_poke,
        # 'lazy': build_lazy,
        # 'coerce': build_coerce,
        # 'write': build_write,
        # 'get': build_get,
        # 'set': build_set,
        # 'delete': build_delete,
        # 'update': build_update,
        # 'init': build_init,
    }


    def _repr_args(self) -> str:
        s = self.describe()
        if spec := self.spec:
            return s + ', ' + spec.show_keywords()
        return s

    @classmethod
    def build(cls, owner: 'Meta', name: str, spec: Spec) -> Optional[Self]:
        return cls(owner, name, spec)

    FieldType: ClassVar[builtins.type[FieldType]] = FieldType


wrap_slot(Field, 'scope', coerce=transform_coercer(Scope))
wrap_slot(Field, 'visibility', coerce=transform_coercer(Visibility))


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


RegistryFallback: TypeAlias = Callable[['Registry', str], None]

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

    def push_fallback(self, fallback: RegistryFallback):
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
                    if factory := getattr(ns, kind, None):
                        self.factories[key] = factory
                        return factory
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

    def coerce(self, spec: Any = None, /, **kwargs) -> Self:

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

    def provide(self, *kinds, spread: bool = False) -> Callable[[T], T]:
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

                slot = f'_{name}'
                if self.has_slot(slot):
                    field_spec['slot'] = slot

                if field_spec.get('visibility', Visibility.public) is Visibility.protected:
                    self.debug(f'meta: processing protected field [{name}]:', field_spec)


                # print(f'  {name:>20}:', field_spec)
                self.add_field(name, field_spec)

            # print('-' * 100)

    def engineer_fields(self) -> None:
        for f in self.own_fields.values():
            f.engineer(self)

        self.field_inits = tuple(f.init for f in self.fields.values() if Scope.is_instance(f.scope) and f.init)

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


def provides_from_type(cls: type, *types, spread: bool = False) -> Callable[[T], T]:
    meta = Meta.for_class(cls, build=True)
    return meta.provide_from_type(*types, spread=spread)


def coerce(cls: type[T], spec: Any = None, /, **kwargs) -> T:
    try:
        meta = Meta.for_class(cls, build=True)
        return meta.coerce(spec, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to coerce {class_qname(cls)} with spec {spec} and kwargs {kwargs}: {e}") from e


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
]