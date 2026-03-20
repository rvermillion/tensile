#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.


from .behavior import *
from .root import *
from .util import StringBuffer, class_qname
from .registry import add_qname_listener, meta_by_qname


if TYPE_CHECKING:
    from .meta import Meta


cached_field_types: dict[Any, 'FieldType'] = {}


def private_slot(name: str) -> str:
    return f'_pvt_{name}'


class FieldType(RootObject):

    __slots__ = ['anno', 'cls', 'meta', 'qname', 'args', 'optional', 'coerce']

    anno: Any
    cls: Optional[type]
    meta: Optional['Meta']
    qname: str
    args: tuple['FieldType', ...]
    optional: bool
    coerce: Optional[Coercer]

    def __init__(self, anno: Any, cls: type = None, qname: str = None, args: tuple['FieldType', ...] = (),
                 optional: bool = False):
        if cls is None:
            if qname is None:
                if anno is not ...:
                    raise ValueError(f'cls or qname must be specified for {anno!r}')
            else:
                add_qname_listener(qname, self.qname_registered)
        else:
            if not isinstance(cls, type):
                raise ValueError(f'cls must be a class or None: {cls}')
            if qname is None:
                qname = class_qname(cls)

        self.anno = anno
        self.cls = cls
        self.qname = qname or ('unknown' if cls is None else class_qname(cls))
        self.args = args
        self.optional = optional
        self.coerce = None

        if not isinstance(anno, str) or '.' in anno:
            # print(f'// caching field type for {anno!r}')
            cached_field_types[anno] = self

    def qname_registered(self, meta: 'Meta'):
        # self.info('Deferred field type for {!r}: {!r}', self, meta)
        self.meta = meta
        if self.cls is None:
            self.cls = meta.cls
        else:
            if self.cls is not meta.cls:
                raise MetaError(f'cannot register {meta} with qname {self.qname} because it is already registered as {self.cls}')

    def has_instance(self, obj: Any) -> bool:
        if cls := self.cls:
            return is_runtime_class(cls) and isinstance(obj, cls)
        return False

    def is_subclass(self, sup: type) -> bool:
        if cls := self.cls:
            return issubclass(cls, sup)
        return False

    def get_subclass(self, sup: type[T]) -> Optional[type[T]]:
        cls = self.cls
        return cls if cls is not None and issubclass(cls, sup) else None

    def get_coerce(self, auto: bool = None, save: bool = True) -> Optional[Coercer]:
        if self.coerce is None:
            if cls := self.cls:
                # noinspection PyTypeChecker
                coerce = coerce_type(cls, optional=self.optional, auto=auto)
                if not save: return coerce
            else:
                coerce = None
            if coerce is None:
                self.debug('no coerce method for {!r}', self)
            else:
                self.coerce = coerce
        return self.coerce

    @property
    def equiv(self) -> Equiv:
        return eq_equiv

    @property
    def is_unique(self) -> bool:
        if cls := self.cls:
            return issubclass(cls, AbstractSet)
        return False

    @property
    def is_sequence(self) -> bool:
        if cls := self.cls:
            return issubclass(cls, Sequence) and not issubclass(cls, str)
        return False

    @property
    def is_mapping(self) -> bool:
        if cls := self.cls:
            return issubclass(cls, Mapping)
        return False

    @property
    def is_multivalued(self) -> bool:
        if cls := self.cls:
            return issubclass(cls, Collection) and not issubclass(cls, str)
        return False

    @property
    def update(self) -> Optional[Callable[[Any, Any], Any]]:
        return None

    def is_subclass_of(self, sup: Union[type, 'FieldType']) -> bool:
        if cls := self.cls:
            if isinstance(sup, type):
                return issubclass(cls, sup)
            if isinstance(sup, FieldType) and (sup_cls := sup.cls):
                return issubclass(cls, sup_cls)
        return False

    def to_spec(self, spec: Spec):
        if self.optional:
            spec['optional'] = True
        if self.is_multivalued:
            spec['multivalued'] = True
        if self.is_unique:
            spec['unique'] = True
        if self.is_sequence:
            spec['sequence'] = True

    def _repr_type(self, **options) -> str:
        return 'FieldType'

    def _repr_args(self, **options) -> str:
        s = str(self)
        if self.cls is None:
            return s + ', +deferred'
        return s

    # noinspection PyMethodMayBeStatic
    def _show_args(self, args: tuple['FieldType', ...]) -> str:
        return '[' + ', '.join(str(arg) for arg in args) + ']'

    def __str__(self) -> str:
        t = self.qname or str(self.anno)
        if args := self.args:
            t += self._show_args(args)
        if self.optional:
            return t + '?'
        return t

    @classmethod
    def from_anno(cls, anno: Any, owner: type) -> 'FieldType':
        if isinstance(anno, cls):
            return anno

        try:
            impl = FieldType

            if anno is None:
                anno = NoneType

            if isinstance(anno, str):
                if '.' in anno:
                    if field_type := cached_field_types.get(anno):
                        # print(f'// using cached field type for {anno!r}')
                        return field_type
                return impl(anno, None, anno)

            if field_type := cached_field_types.get(anno):
                # print(f'// using cached field type for {anno!r}')
                return field_type

            full_anno = anno
            origin = get_origin(anno)

            qname = None
            args = ()
            optional = False

            if origin is Union:
                unions = set(get_args(anno))
                if NoneType in unions:
                    if len(unions) == 2:
                        unions.discard(NoneType)
                        anno, = unions
                        origin = get_origin(anno)
                        optional = True

            if isinstance(anno, str):
                qname = anno
                origin = None
            elif isinstance(anno, type):
                qname = class_qname(anno)
                origin = anno
            # elif is_generic_alias(anno):
            #     args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
            #     qname = class_qname(origin)
            elif is_forward_ref(anno):
                qname = get_forward_ref_name(anno)
                if '.' in qname:
                    if qname == class_qname(owner):
                        origin = owner
                    elif meta := meta_by_qname.get(qname):
                        origin = meta.cls
                else:
                    if qname == owner.__qualname__:
                        qname = class_qname(owner)
                        origin = owner
                    else:
                        qname = owner.__module__ + '.' + qname
                        if meta := meta_by_qname.get(qname):
                            origin = meta.cls
            elif anno is ...:
                pass
            elif origin is type:
                args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
                qname = class_qname(origin)
            elif isinstance(origin, type):
                if is_protocol(origin):
                    # if getattr(origin, '_is_protocol', False):
                    args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
                    qname = class_qname(origin)
                elif issubclass(origin, Callable):
                    if raw_args := get_args(anno):
                        if origin is Callable:
                            if len(raw_args) == 2:
                                params, ret = raw_args
                                arg_list = [FieldType.from_anno(params, owner)] if params is ... else [FieldType.from_anno(arg, owner) for arg in params]
                                arg_list.append(FieldType.from_anno(ret, owner))
                                args = tuple(arg_list)
                                impl = CallableFieldType
                            else:
                                raise MetaError(f'Callable annotation must have two args: {anno!r}')
                        else:
                            args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
                    qname = class_qname(origin)
                elif issubclass(origin, Collection):
                    args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
                    impl = CollectionFieldType
                elif is_generic_alias(anno):
                    args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
                    qname = class_qname(origin)
            elif origin is Union:
                impl = UnionFieldType
                args = tuple(FieldType.from_anno(arg, owner) for arg in get_args(anno))
                qname = class_qname(origin)
                # print(f'Problem processing union annotation: {anno} with args {args}')
                origin = object
            elif isinstance(anno, TypeVar):
                origin = object
            else:
                print(f'Problem processing annotation: {anno} with origin {origin}')
                raise MetaError(f'Problem processing annotation: {anno} with origin {origin}')

            return impl(full_anno, origin, qname=qname, args=args, optional=optional)
        except Exception:
            raise MetaError(f'Error getting field type from annotation: {anno!r}')


class CallableFieldType(FieldType):

    __slots__ = ()

    def __str__(self) -> str:
        if args := self.args:
            if len(args) > 2:
                return '(' + ', '.join(str(arg) for arg in args[:-1]) + f') => {args[-1]}'
            return ' => '.join(str(arg) for arg in args)
        return '... => Any'


class UnionFieldType(FieldType):

    __slots__ = ()

    def __init__(
        self, anno: Any, cls: type = None, qname: str = None, args: tuple['FieldType', ...] = (),
        optional: bool = False):
        if len(args) < 2:
            raise ValueError(f'UnionFieldType requires at least two args: {args}')
        qname = 'Union[' + ', '.join(arg.qname for arg in args) + ']'
        super().__init__(anno, cls, qname, args, optional)

    def get_coerce(self) -> Optional[Coercer]:
        return None

    def __str__(self) -> str:
        return self.qname


class CollectionFieldType(FieldType):

    __slots__ = ()

    # noinspection PyMethodOverriding
    def update(self, old: Any, new: Iterable[Any]) -> Any:
        if old is None:
            return new
        if isinstance(old, MutableSequence):
            old.clear()
            old.extend(new)
            return old
        return self.cls(new)


class FieldSpec(dict[str, Any]):

    __slots__ = ()

    def __repr__(self):
        return 'field(' + show_keywords(self) + ')'


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
        coerce: bool|str|Coercer|None = ...,
        lazy: bool|str|Getter|None = ...,
        init_order: int = ...,
        **kwargs
    ) -> Keywords: ...
else:
    def field(**kwargs) -> Keywords:
        return FieldSpec(kwargs)


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
    'init_order': 1000,
}

def default_builder(attr, default: Any) -> Callable[['Field', Spec], Any]:
    return lambda f, spec: spec.get(attr, default)


field_attr_default_builders: dict[str, Callable[['Field', Spec], Any]] = {
    attr: default_builder(attr, dflt) for attr, dflt in field_attr_defaults.items()
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


def build_changed(name: str = None, check_cls: type = None, desc: str = '') -> Optional[Setter]:
    changed_method = f'_{name}_changed'
    if check_cls is None or callable(getattr(check_cls, changed_method, None)):
        return method_changed(method=changed_method)
    return None


def build_changing(name: str = None, check_cls: type = None, desc: str = '') -> Optional[Setter]:
    changing_method = f'_{name}_changing'
    if check_cls is None or callable(getattr(check_cls, changing_method, None)):
        return method_setter(method=changing_method)
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



# noinspection PyShadowingNames
def build_setter(poke: Setter = None, coerce: Coercer = None, peek: Getter = None,
                 changed: Setter = None, changing: Setter = None, readonly: bool = False, desc: str = '') -> Optional[Setter]:
    if readonly:
        return readonly_setter(desc=desc)

    if poke is None:
        return None
    if peek is None: peek = none_getter
    if coerce is None:
        if changing is None:
            if changed is None:
                return poke

            def setter(this: Any, val: Any):
                try:
                    try:
                        old = peek(this)
                    except AttributeError:
                        old = None
                    poke(this, val)
                    changed(this, val, old)
                except Exception as e:
                    raise TypeError(f'{desc}: cannot set to {val!r} in {this!r}') from e
        else:
            if changed is None:
                def setter(this: Any, val: Any):
                    try:
                        changing(this, val)
                        poke(this, val)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot set to {val!r} in {this!r}') from e
            else:
                def setter(this: Any, val: Any):
                    try:
                        changing(this, val)
                        try:
                            old = peek(this)
                        except AttributeError:
                            old = None
                        poke(this, val)
                        changed(this, val, old)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot set to {val!r} in {this!r}') from e


    else:
        if changing is None:
            if changed is None:
                def setter(this: Any, val: Any):
                    try:
                        coerced = coerce(this, val)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot coerce {val!r} in {this!r}') from e
                    try:
                        poke(this, coerced)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot set to {coerced!r} in {this!r}') from e
            else:
                def setter(this: Any, val: Any):
                    try:
                        coerced = coerce(this, val)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot coerce {val!r} in {this!r}') from e
                    try:
                        try:
                            old = peek(this)
                        except AttributeError:
                            old = None
                        poke(this, coerced)
                        changed(this, coerced, old)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot set to {coerced!r} in {this!r}') from e
        else:
            if changed is None:
                def setter(this: Any, val: Any):
                    try:
                        coerced = coerce(this, val)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot coerce {val!r} in {this!r}') from e
                    try:
                        changing(this, coerced)
                        poke(this, coerced)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot set to {coerced!r} in {this!r}') from e
            else:
                def setter(this: Any, val: Any):
                    try:
                        coerced = coerce(this, val)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot coerce {val!r} in {this!r}') from e
                    try:
                        changing(this, coerced)
                        try:
                            old = peek(this)
                        except AttributeError:
                            old = None
                        poke(this, coerced)
                        changed(this, coerced, old)
                    except Exception as e:
                        raise TypeError(f'{desc}: cannot set to {coerced!r} in {this!r}') from e


    return setter

def build_property(peek: Getter = None, poke: Setter = None, lazy: Getter = None, coerce: Coercer = None,
                   delete: Deleter = None, changed: Setter = None, changing: Setter = None, readonly: bool = False,
                   doc: str = None, desc: str = '') -> property:
    getter = build_getter(peek=peek, lazy=lazy, poke=poke, desc=desc)
    setter = build_setter(poke=poke, coerce=coerce, peek=peek, changed=changed, changing=changing, readonly=readonly, desc=desc)

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


building: set[str] = set()


class FieldProperty(property):

    __slots__ = ('field',)

    field: 'Field'

    # noinspection PyShadowingNames
    def __init__(self, field: 'Field'):
        self.field = field
        super().__init__(field.get, field.set, field.delete, field.doc)

    def __repr__(self):
        return f'<field {self.field.describe(True)}>'


class Field(RootObject):

    __slots__ = ['owner', 'name', 'spec', 'type', 'aliases', 'slot',
                 'peek', 'lazy', 'get', 'poke', 'coerce', 'write',
                 'set', 'delete', 'init', 'init_order', 'is_set', 'update',
                 'default', 'default_factory', 'required', 'readonly',
                 'doc', 'scope', 'member', 'visibility', 'delegate',
                 'changing', 'changed', 'options']

    slots: ClassVar[set[str]] = set(__slots__)

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
    init_order: int
    changed: Optional[Setter]
    changing: Optional[Setter]
    delegate: Optional[str]
    doc: str
    member: Any
    default: Any
    default_factory: Callable[[], Any]
    required: bool
    readonly: bool
    scope: Scope
    visibility: Visibility
    options: Optional[dict[str, Any]]

    def __init__(self, owner: 'Meta', name: str, spec: Spec):
        self.owner = owner
        self.name = name
        self.spec = spec

        self.type = spec.pop('type', None)
        options = {}
        for k, v in spec.items():
            if k not in self.slots:
                options[k] = v
        self.options = options if options else None

    def get_option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default) if self.options else default

    @property
    def qname(self) -> str:
        return self.owner.qname + '.' + self.name

    @property
    def equiv(self) -> Equiv:
        return self.type.equiv

    def preprocess_coerce(self, coerce: Any) -> Any:
        if coerce is True:
            return self.type.get_coerce(auto=True, save=False)
        return coerce

    preprocessors: dict[str, Callable[['Field', Any], Any]] = {
        'coerce': preprocess_coerce,
    }

    def preprocess(self, key: str, val: Any) -> Any:
        if preprocessor := self.preprocessors.get(key):
            return preprocessor(self, val)
        return val

    def __getattr__(self, attr: str) -> Any:
        if builder := self.attr_builders.get(attr):
            if attr in building:
                raise MetaError(f'circular reference detected in {self.owner.cls.__name__}.{self.name}: {attr}')
            building.add(attr)
            try:
                val = builder(self, self.spec)
                setattr(self, attr, val)
            finally:
                building.discard(attr)
            return val
        raise AttributeError(f'Field {self!r} has no attribute {attr!r}')

    def get_method(self, *methods, prefix: bool = True, suffix: bool = False) -> Optional[Callable]:
        cls = self.owner.cls

        def get_method(mname: str) -> Optional[Callable]:
            meth = getattr(cls, mname, None)
            return meth if callable(meth) else None

        for m in methods:
            if prefix:
                if method := get_method(f'{m}_{self.name}'): return method
            if suffix:
                if method := get_method(f'{self.name}_{m}'): return method
            elif not prefix:
                if method := get_method(m): return method
        return None

    def has_methods(self, *methods, prefix: bool = True, suffix: bool = False) -> bool:
        return self.get_method(*methods, prefix=prefix, suffix=suffix) is not None

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
            return FieldProperty(self)
            # return property(self.get, self.set, self.delete, self.doc)
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

    def override(self, meta: 'Meta', spec: Spec) -> Optional['Field']:
        cls = spec.get('field_class', meta.Field)
        for key, val in self.spec.items():
            spec.setdefault(key, val)
        # print(f'{meta.qname} override: {self.name} with {spec}')
        # if len(spec) == 1:
        #     if 'type' in spec:
        #         return None
        return cls(meta, self.name, spec)

    def to_spec(self, spec: Spec):
        pass

    @classmethod
    def new_spec(cls, **kwargs) -> Spec:
        return Spec(**kwargs) if kwargs else Spec()

    def add_state(self, this: Any, state: dict[str, Any]):
        state[self.name] = self.get(this)

    def describe(self, qualified: bool = False) -> str:
        buff = StringBuffer(sep=' ')
        name = self.qname if qualified else self.name
        scope = '#' if Scope.is_class(self.scope) else ''
        if self.visibility is not Visibility.public:
            buff.append(self.visibility.name)
        buff.append(f'{scope}{name}: {self.type or "unknown"}')
        if self.default is not None:
            buff.append(f'= {self.default!r}')
        elif self.default_factory is not None:
            buff.append(f'= {self.default_factory.__name__}()')
        buff.flag('lazy', self.lazy, skip=False)
        if self.coerce:
            buff.append('+coerce')
        if self.changing:
            buff.append('+changing')
        if self.changed:
            buff.append('+changed')
        if self.delegate is not None:
            buff.append('+delegate')
        if self.required:
            buff.append('+required')
        if self.readonly:
            buff.append('+readonly')
        return str(buff)

    def build_peek(self, spec: Spec) -> Optional[Getter]:
        return build_peek(member=self.member, slot=self.slot)

    def build_lazy(self, spec: Spec) -> Optional[Getter]:
        lazy = spec.get('lazy')
        if lazy is None or lazy:
            if isinstance(lazy, str):
                lazy = method_getter(method=lazy)
            elif callable(lazy):
                pass
            else:
                lazy = build_lazy(name=self.name, check_cls=self.owner.cls)
        return lazy

    def build_changed(self, spec: Spec) -> Optional[Setter]:
        changed = spec.get('changed')
        if changed is None or changed:
            if isinstance(changed, str):
                changed = method_setter(method=changed)
            elif callable(changed):
                pass
            else:
                changed = build_changed(name=self.name, check_cls=self.owner.cls)
        return changed

    def build_changing(self, spec: Spec) -> Optional[Setter]:
        changing = spec.get('changing')
        if isinstance(changing, str):
            changing = method_setter(method=changing)
        else:
            changing = build_changing(name=self.name, check_cls=self.owner.cls)
        return changing

    def build_get(self, spec: Spec) -> Getter:
        if getter := spec.get('get'):
            return getter
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
        if delegate := self.delegate:
            name = self.name
            dot = delegate.find('.')
            if dot >= 0:
                delegate, name = delegate[:dot], delegate[dot + 1:]
            def poke(this: Any, value: Any):
                setattr(getattr(this, delegate), name, value)
            return poke
        return build_poke(member=self.member, slot=self.slot, desc=self.qname)

    def build_coerce(self, spec: Spec) -> Optional[Coercer]:
        coerce = spec.get('coerce')
        if isinstance(coerce, str):
            coerce = method_coercer(method=coerce)
        else:
            if coerce is True:
                coerce = self.type.get_coerce(auto=True, save=False)
            if coerce is None:
                coerce = build_coerce(self.name, check_cls=self.owner.cls, field_type=self.type, desc=self.qname,
                                      required=self.required)
        return coerce

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

        if self.delegate:
            return self.poke
        if Scope.is_instance(self.scope):
            setter = build_setter(poke=self.poke, coerce=self.coerce, peek=self.peek,
                                  changed=self.changed, changing=self.changing,
                                  desc=self.qname)
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
        if not spec.get('init', True):
            return None
        if self.readonly and isinstance(self.member, property):
            return None

        # If we are delegating, don't poke!
        poke = self.write if self.delegate else self.poke
        return name_initter(self.name, self.aliases,
                            writer=self.write,
                            poke=poke,
                            default=self.default,
                            default_factory=self.default_factory,
                            optional=self.type.optional)

    def build_delete(self, spec: Spec) -> Optional[Deleter]:
        if isinstance(self.member, property):
            return self.member.fdel
        return build_delete(member=self.member, slot=self.slot, desc=self.qname)

    def build_slot(self, spec: Spec) -> Optional[str]:
        if slot := spec.get('slot'):
            return slot
        return None if isinstance(self.member, MemberDescriptorType) else private_slot(self.name)

    attr_builders: ClassVar[dict[str, Callable[['Field', Spec], Any]]] = {
        **field_attr_default_builders,
        **method_builders(
            'slot', 'peek', 'poke', 'lazy', 'coerce',
            'write', 'get', 'set', 'delete', 'update', 'init',
            'changing', 'changed',
        ),
    }


    def _repr_args(self) -> str:
        return self.describe()

    @classmethod
    def build(cls, owner: 'Meta', name: str, spec: Spec) -> Optional[Self]:
        return cls(owner, name, spec)

    FieldType: ClassVar[builtins.type[FieldType]] = FieldType


wrap_slot(Field, 'scope', coerce=transform_coercer(Scope))
wrap_slot(Field, 'visibility', coerce=transform_coercer(Visibility))
