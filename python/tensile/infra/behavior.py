#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from pathlib import Path

import tensile.infra as infra
from . import log
from .types import *
from .util import class_qname, name_function

class MetaError(RuntimeError):

    pass


is_equiv: Equiv[Any]
eq_equiv: Equiv[Any]


none: TransformFunction[Any, None]


def relation_predicate(relation: Relation[T, X]) -> PredicateFunction[tuple[T, X]]:
    def predicate(pair: tuple[T, X]) -> bool:
        return relation(*pair)
    return predicate


def predicate_relation(predicate: PredicateFunction[tuple[T, X]]) -> Relation[T, X]:
    def relation(a: T, b: X) -> bool:
        return predicate((a, b))
    return relation


def left_relation(relation: Relation[T, X], right: X) -> PredicateFunction[T]:
    def predicate(a: T) -> bool:
        return relation(a, right)
    return predicate


def right_relation(relation: Relation[T, X], left: T) -> PredicateFunction[X]:
    def predicate(a: X) -> bool:
        return relation(left, a)
    return predicate




def is_equiv(a: Any, b: Any) -> bool:
    return a is b


def eq_equiv(a: Any, b: Any) -> bool:
    return a == b


# noinspection PyUnusedLocal
def none_getter(this: Any) -> Any:
    return None


def constant_getter(const: X, desc: str = '') -> Getter[Any, X]:
    # noinspection PyUnusedLocal
    def getter(this: Any) -> X:
        return const
    if not desc:
        desc = f'get_const[{const}]'
    return name_function(getter, desc)


def attr_getter(name: str, default: Any = missing, desc: str = '') -> Getter:
    if default is missing:
        def getter(this: Any) -> Any:
            return getattr(this, name)
    else:
        def getter(this: Any) -> Any:
            return getattr(this, name, default)
    if not desc:
        desc = f'dynamic_get[{name}]'
    return name_function(getter, desc)


def safe_getter(getter: Getter, error: str, desc: str = '') -> Getter:
    def safe(this: Any) -> Any:
        try:
            return getter(this)
        except AttributeError as e:
            log.error(error, e)
            raise e
            # raise MetaError(error) from e
    if not desc:
        desc = f'safe_get[{getter}]'
    return name_function(safe, desc)


def attr_setter(name: str, default: Any = missing, desc: str = '') -> Setter:
    if default is missing:
        def setter(this: Any, value: Any) -> None:
            setattr(this, name, value)
    else:
        def setter(this: Any, value: Any) -> None:
            setattr(this, name, value if value is not missing else default)
    if not desc:
        desc = f'dynamic_set[{name}]'
    return name_function(setter, desc)


def safe_setter(setter: Setter, error: str, desc: str = '') -> Setter:
    def safe(this: Any, value: Any) -> None:
        try:
            setter(this, value)
        except AttributeError as e:
            raise MetaError(error) from e
    if not desc:
        desc = f'safe_set[{setter}]'
    return name_function(safe, desc)


def attr_deleter(name: str, desc: str = '') -> Deleter:
    def deleter(this: Any) -> None:
        delattr(this, name)
    if not desc:
        desc = f'dynamic_delete[{name}]'
    return name_function(deleter, desc)


def attr_is_setter(name: str, default: Any = missing, equiv: Equiv = eq_equiv, desc: str = '') -> IsSetter:
    if default is missing:
        def is_setter(this: Any) -> bool:
            return hasattr(this, name)
    else:
        def is_setter(this: Any) -> bool:
            val = getattr(this, name, default)
            return val is not missing and not equiv(val, default)
    if not desc:
        desc = f'dynamic_is_set[{name}]'
    return name_function(is_setter, desc)


def peek_is_setter(peek: Getter, default: Any = missing, equiv: Equiv = eq_equiv, desc: str = '') -> IsSetter:
    if default is missing:
        def is_setter(this: Any) -> bool:
            return peek(this) is not missing
    elif equiv is is_equiv:
        def is_setter(this: Any) -> bool:
            val = peek(this)
            return val is not missing and val is not default
    elif equiv is eq_equiv:
        def is_setter(this: Any) -> bool:
            val = peek(this)
            return val is not missing and val != default
    else:
        def is_setter(this: Any) -> bool:
            val = peek(this)
            return val is not missing and not equiv(val, default)
    if not desc:
        desc = f'dynamic_is_set[peek={peek}]'
    return name_function(is_setter, desc)


def prefix_method(method: str, prefix: str, strip_underscore: bool = True) -> str:
    if strip_underscore:
        if method[0] == "_":
            method = method[1:]
    return f'{prefix}_{method}'


def method_zero_caller(method: str, *args, pass_instance: bool = False, desc: str = '') -> Callable[[Any], Any]:
    if args:
        if pass_instance:
            def call_method(this: Any) -> Any:
                return getattr(this, method)(this, *args)
        else:
            def call_method(this: Any) -> Any:
                return getattr(this, method)(*args)
    elif pass_instance:
        def call_method(this: Any) -> Any:
            return getattr(this, method)(this)
    else:
        def call_method(this: Any) -> Any:
            return getattr(this, method)()
    if not desc:
        desc = f'dynamic_call[{method}]'
    return name_function(call_method, desc)


def method_one_caller(method: str, *args, pass_instance: bool = False, desc: str = '') -> Callable[[Any, Any], Any]:
    if args:
        if pass_instance:
            def call_method(this: Any, value: Any) -> None:
                return getattr(this, method)(this, value, *args)
        else:
            def call_method(this: Any, value: Any) -> None:
                return getattr(this, method)(value, *args)
    elif pass_instance:
        def call_method(this: Any, value: Any) -> None:
            return getattr(this, method)(this, value)
    else:
        def call_method(this: Any, value: Any) -> None:
            return getattr(this, method)(value)
    if not desc:
        desc = f'dynamic_call[{method}]'
    return name_function(call_method, desc)


def method_many_caller(method: str, *add_args, pass_instance: bool = False, desc: str = '') -> Callable[[Any, Any], Any]:
    if add_args:
        if pass_instance:
            def call_method(this: Any, *args: Any) -> None:
                return getattr(this, method)(this, *args, *add_args)
        else:
            def call_method(this: Any, *args: Any) -> None:
                return getattr(this, method)(*args, *add_args)
    elif pass_instance:
        def call_method(this: Any, *args: Any) -> None:
            return getattr(this, method)(this, *args)
    else:
        def call_method(this: Any, *args: Any) -> None:
            return getattr(this, method)(*args)
    if not desc:
        desc = f'dynamic_call[{method}]'
    return name_function(call_method, desc)


def method_getter(method: str, *args, pass_instance: bool = False, desc: str = '') -> Getter:
    return method_zero_caller(method, *args, pass_instance=pass_instance, desc=desc)


def method_setter(method: str, *args, pass_instance: bool = False, desc: str = '') -> Setter:
    return method_one_caller(method, *args, pass_instance=pass_instance, desc=desc)


def method_changed(method: str, *args, pass_instance: bool = False, desc: str = '') -> Setter:
    return method_many_caller(method, *args, pass_instance=pass_instance, desc=desc)


def method_coercer(method: str, *args, pass_instance: bool = False, desc: str = '') -> Coercer:
    return method_one_caller(method, *args, pass_instance=pass_instance, desc=desc)


def transform_coercer(transform: TransformFunction[Any, Y], desc: str = '') -> Coercer[Any, Y]:
    # noinspection PyUnusedLocal
    def coerce(this: Any, value: Any) -> Y:
        return transform(value)
    if not desc:
        desc = f'dynamic_coerce[transform={transform}]'
    return name_function(coerce, desc)


def name_initter(name: str, aliases: tuple[str, ...] = None, /, writer: Setter = None,
                 default: Any = None,
                 default_factory: Callable[[], Any] = None,
                 required: bool = False) -> Initter[Any]:
    if writer is None:
        writer = attr_setter(name)

    if aliases:
        if required:
            def init(this: Any, spec: Spec):
                value = spec.get(name, missing)

                if value is missing:
                    for alias in aliases:
                        value = spec.get(alias, missing)
                        if value is not missing:
                            break
                    else:
                        raise ValueError(f'{name} (or {", ".join(aliases)}) is required!')

                writer(this, value)
        elif default_factory is not None:
            def init(this: Any, spec: Spec):
                value = spec.get(name, missing)

                if value is missing:
                    for alias in aliases:
                        value = spec.get(alias, missing)
                        if value is not missing:
                            break
                    else:
                        value = default_factory()

                writer(this, value)
        else:
            def init(this: Any, spec: Spec):
                value = spec.get(name, missing)

                if value is missing:
                    for alias in aliases:
                        value = spec.get(alias, missing)
                        if value is not missing:
                            break
                    else:
                        value = default

                writer(this, value)
    else:
        if required:
            def init(this: Any, spec: Spec):
                value = spec.get(name, missing)
                if value is missing:
                    raise ValueError(f'{name} is required!')
                writer(this, value)
        elif default_factory is not None:
            def init(this: Any, spec: Spec):
                value = spec.get(name, missing)
                try:
                    if value is missing: value = default_factory()
                    writer(this, value)
                except Exception as e:
                    raise AttributeError(f'Error initializing field [{name}] with: {value}') from e
        else:
            def init(this: Any, spec: Spec):
                value = spec.get(name, default)
                try:
                    writer(this, value)
                except Exception as e:
                    raise AttributeError(f'Error initializing field [{name}] with: {value}') from e

    return name_function(init, f'dynamic_init[{name}]')


coerce_str: Coercer[Any, str]
coerce_int: Coercer[Any, int]


# noinspection PyUnusedLocal
def coerce_str(this: Any, val: Any) -> str:
    return str(val)


# noinspection PyUnusedLocal
def coerce_int(this: Any, val: Any) -> int:
    return int(val)


# noinspection PyUnusedLocal
def coerce_float(this: Any, val: Any) -> float:
    return float(val)


# noinspection PyUnusedLocal
def coerce_path(this: Any, val: Any) -> Path:
    return Path(val)


type_coercers: dict[type, Coercer] = {}

qname_coercers: dict[str, Coercer] = {}

def register_type_coercer(coerce: Coercer, cls: type = None, qname: str = None) -> None:
    if cls is not None:
        type_coercers[cls] = coerce
        if qname is None: qname = class_qname(cls)
    if qname is not None:
        qname_coercers[qname] = coerce
        if qname.startswith('builtins.'):
            qname_coercers[qname[9:]] = coerce

def register_type_coercers(coercers: dict[str|type, Coercer]) -> None:
    for key, coerce in coercers.items():
        if isinstance(key, type):
            register_type_coercer(coerce, cls=key)
        elif isinstance(key, str):
            register_type_coercer(coerce, qname=key)
        else:
            raise TypeError(f'Invalid key type {type(key)} for coercer registration')

register_type_coercers({
    str: coerce_str,
    int: coerce_int,
    float: coerce_float,
    Path: coerce_path,
})

class CoerceError(TypeError):

    pass


def coerce_const(const: X) -> Coercer[Any, X]:
    # noinspection PyUnusedLocal
    def coerce(this: Any, val: Any) -> Optional[X]:
        return const
    return name_function(coerce, f'coerce_const[{const}]')


def coerce_optional(coerce: Optional[Coercer[X, Y]], default: Optional[Y] = None, optional: bool = True) -> Optional[Coercer[X, Optional[Y]]]:
    if coerce and optional:
        if default is None:
            def coerce_or_none(this: Any, val: Any) -> Optional[Y]:
                return None if val is None else coerce(this, val)
        else:
            def coerce_or_none(this: Any, val: Any) -> Optional[Y]:
                return default if val is None else coerce(this, val)
        return name_function(coerce_or_none, f'coerce_optional[{coerce}]')
    return coerce


def coerce_conditional(coerce: Coercer[X, Y], condition: Callable[[Any], bool], else_coerce: Coercer[X, Y] = None) -> Coercer[X, Y]:
    if else_coerce is None:
        def coerce_if(this: Any, val: Any) -> Y:
            if condition(this):
                return coerce(this, val)
            return val
    else:
        def coerce_if(this: Any, val: Any) -> Y:
            if condition(this):
                return coerce(this, val)
            return else_coerce(this, val)
    return name_function(coerce_if, f'coerce_conditional[{coerce}, condition={condition}]')


def meta_coercer(meta: 'infra.Meta', this_arg: str = None) -> Coercer:

    log.warn('Using meta {} coerce for automatic coercion', meta)

    if this_arg is None:
        # noinspection PyUnusedLocal
        def coerce(this: Any, val: Any) -> X:
            return meta.coerce(val)
    else:
        def coerce(this: Any, val: Any) -> X:
            kwargs = {this_arg: this}
            return meta.coerce(val, **kwargs)

    return coerce


def coerce_type(cls: Optional[type[X]], optional: bool = False, qname: str = None, auto: bool = None, generate: bool = False) -> Optional[Coercer[Any, X]]:
    if isinstance(cls, type):
        coerce = type_coercers.get(cls)
        if coerce is None:
            if qname is None: qname = class_qname(cls)
            coerce = qname_coercers.get(qname)
        if coerce is None:
            if is_runtime_class(cls):
                if getattr(cls, 'auto_coerce', False) if auto is None else auto:
                    meta = infra.meta.for_spec(cls or qname, build=True)
                    coerce = meta_coercer(meta)
                if coerce is None:
                    # noinspection PyUnusedLocal
                    def coerce(this: Any, val: Any) -> X:
                        if isinstance(val, cls):
                            return val
                        raise CoerceError(f'Cannot coerce {val!r} to {cls}')
                # else:
                #     def auto_coerce(this: Any, val: Any) -> X:
                #         # noinspection PyCallingNonCallable
                #         return coerce(val)
                #     return auto_coerce
            elif auto:
                meta = infra.meta.for_spec(cls or qname, build=True)
                coerce = meta_coercer(meta)
            else:
                log.debug('Skipping non-runtime class:', cls)
    else:
        coerce = None
    return None if coerce is None else coerce_optional(coerce) if optional else coerce


def pick_arg(fn: Callable[..., T], arg: int = 0) -> Callable[..., T]:
    def picked(*args, **kwargs) -> T:
        return fn(args[arg], **kwargs)
    return picked


def slice_args(fn: Callable[..., T], start: int = 0, stop: int = None) -> Callable[..., T]:
    s = slice(start, stop)
    def sliced(*args, **kwargs) -> T:
        return fn(*args[s], **kwargs)
    return sliced
