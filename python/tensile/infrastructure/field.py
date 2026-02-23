#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from .functional import *
from .root import *
from .util import class_qname, process_specs
from . import meta as metal


cached_field_types: dict[Any, 'FieldType'] = {}


class FieldType(RootObject):

    __slots__ = ['anno', 'cls', 'qname', 'args', 'optional']

    anno: Any
    cls: Optional[type]
    qname: str
    args: tuple['FieldType', ...]
    optional: bool

    def __init__(self, anno: Any, cls: type = None, qname: str = None, args: tuple['FieldType', ...] = (),
                 optional: bool = False):
        if cls is None:
            if qname is None:
                if anno is not ...:
                    raise ValueError(f'cls or qname must be specified for {anno!r}')
            else:
                metal.add_qname_listener(qname, self.qname_registered)
        else:
            if not isinstance(cls, type):
                raise ValueError(f'cls must be a class or None: {cls}')
            if qname is None:
                qname = class_qname(cls)

        self.anno = anno
        self.cls = cls
        self.qname = qname
        self.args = args
        self.optional = optional

        if not isinstance(anno, str) or '.' in anno:
            # print(f'// caching field type for {anno!r}')
            cached_field_types[anno] = self

    def qname_registered(self, meta: 'Meta'):
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

    @property
    def coerce(self) -> Optional[Coercer]:
        return coerce_type(self.cls, optional=self.optional)

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

    def _repr_type(self) -> str:
        return 'FieldType'

    def _repr_args(self) -> str:
        s = str(self)
        # if self.is_multivalued:
        #     s += ', +multivalued'
        # if self.is_unique:
        #     s += ', +unique'
        # if self.is_sequence:
        #     s += ', +sequence'
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
                    elif meta := metal.meta_by_qname.get(qname):
                        origin = meta.cls
                else:
                    if qname == owner.__qualname__:
                        qname = class_qname(owner)
                        origin = owner
                    else:
                        qname = owner.__module__ + '.' + qname
                        if meta := metal.meta_by_qname.get(qname):
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
                        params, ret = raw_args
                        arg_list = [FieldType.from_anno(params, owner)] if params is ... else [FieldType.from_anno(arg, owner) for arg in params]
                        arg_list.append(FieldType.from_anno(ret, owner))
                        args = tuple(arg_list)
                    qname = class_qname(origin)
                    impl = CallableFieldType
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

    @property
    def coerce(self) -> Optional[Coercer]:
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

