#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from types import MemberDescriptorType
import typing

from .types import Any, Callable, Iterable, Keywords, Mapping, Optional, Self, Sequence, NoneType, TypeVar, missing


special_class_names: dict[type, str] = {}

special_class_qnames: dict[type, str] = {}


def register_special_class_name(cls: type, name: str, qname: str = None, override: bool = False):
    if qname is None:
        qname = name
    if not override:
        if cls in special_class_names:
            raise ValueError(f'class {cls} already has a special name: {special_class_names[cls]}')
        if cls in special_class_qnames:
            raise ValueError(f'class {cls} already has a special qname: {special_class_qnames[cls]}')
    special_class_names[cls] = name
    special_class_qnames[cls] = qname


def register_special_class_names(names: dict[type, str], qnames: dict[type, str] = None, override: bool = False):
    for cls, name in names.items():
        register_special_class_name(cls, name, qname=None if qnames is None else qnames.get(cls), override=override)


register_special_class_names({
    str: 'str',
    bool: 'bool',
    int: 'int',
    float: 'float',
    dict: 'dict',
    list: 'list',
    tuple: 'tuple',
    set: 'set',
    Callable: 'Callable',
    typing.get_origin(typing.Callable): 'Callable',
    NoneType: 'None',
    Any: 'Any'
})


def class_name(cls: type) -> str:
    if name := special_class_names.get(cls):
        return name
    return cls.__qualname__


def class_qname(cls: type) -> str:
    if qname := special_class_qnames.get(cls):
        return qname
    mod = cls.__module__
    if mod == '__main__':
        mod = 'main'
    return mod + '.' + cls.__qualname__


F = TypeVar('F', bound=Callable)


def name_function(func: F, name: str, qualname: str = None, cls: type = None, namespace: str = None) -> F:
    func._orig_names = (func.__name__, func.__qualname__)
    func.__name__ = name
    if qualname is None:
        if namespace is None and cls is not None:
            namespace = class_qname(cls)
        if namespace is None:
            func.__qualname__ = name
        else:
            func.__qualname__ = namespace + '.' + name
    else:
        func.__qualname__ = qualname
    return func


# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass


def process_specs(spec: Optional[Mapping[str, Any]], kwargs: dict[str, Any], single_key: bool = False) -> tuple[Optional[str], Mapping[str, Any]]:
    if spec is None:
        kind = kwargs.pop('kind', None)
        if kind is None and single_key and len(kwargs) == 1:
            return kwargs.popitem()
        return kind, kwargs
    elif kwargs:
        new_spec = dict(spec)
        new_spec.update(kwargs)
        kind = new_spec.pop('kind', None)
        if kind is None and single_key:
            if len(new_spec) == 1:
                return new_spec.popitem()
            if len(spec) == 1:
                kind, = spec.keys()
                subspec = spec[kind]
                if isinstance(subspec, Mapping):
                    for k, v in subspec.items():
                        kwargs.setdefault(k, v)
                    return kind, kwargs
                return kind, subspec
        return kind, new_spec
    else:
        new_spec = dict(spec)
        kind = new_spec.pop('kind', None)
        if kind is None and single_key:
            if len(new_spec) == 1:
                return new_spec.popitem()
        return kind, new_spec


def spread(fn: Callable, /, map_fn: Callable = None, seq_fn: Callable = None, scalar_fn: Callable = None) -> Callable[[Mapping[str, Any]], Any]:
    if map_fn is None: map_fn = fn
    if seq_fn is None: seq_fn = fn
    if scalar_fn is None: scalar_fn = fn
    def spreader(spec: Any) -> Any:
        if isinstance(spec, Mapping):
            return map_fn(**spec)
        if isinstance(spec, Sequence):
            return seq_fn(*spec)
        return scalar_fn(spec)
    return spreader


def spread_mapping(fn: Callable) -> Callable[[Mapping[str, Any]], Any]:
    def spreader(spec: Mapping[str, Any]) -> Any:
        return fn(**spec)
    return spreader


def spread_sequence(fn: Callable) -> Callable[[Sequence], Any]:
    def spreader(spec: Sequence) -> Any:
        return fn(*spec)
    return spreader



Format = Callable[[Any], str]


class StringBuffer:
    __slots__ = ('buff', 'sep', 'fmt')

    buff: list[str]
    sep: str
    fmt: Format

    def __init__(self, buff: Iterable[str] = None, sep: str = '', fmt: Format = str):
        if buff is None:
            buff = []
        else:
            buff = [str(part) for part in buff]
        self.buff = buff
        self.sep = str(sep)
        self.fmt = fmt

    def append(self, *values, skip_none: bool = True, fmt: Format = None) -> Self:
        if fmt is None: fmt = self.fmt
        if skip_none:
            append = self.buff.append
            for value in values:
                if value is not None:
                    append(fmt(value))
        else:
            self.buff.extend(map(fmt, values))
        return self

    def keyword(self, name: str, value: Any, /, skip_none: bool = True, fmt: Format = None, flag: bool = False) -> Self:
        if value is None and skip_none:
            return self
        if flag and isinstance(value, bool): return self.flag(name, value)
        if fmt is None: fmt = self.fmt
        self.buff.append(f'{name}={fmt(value)}')
        return self

    def keywords(self, keywords: Mapping[str, Any], /, skip_none: bool = True, fmt: Format = None, flag: bool = False) -> Self:
        if keywords:
            if fmt is None: fmt = self.fmt
            buff = self.buff
            for name, value in keywords.items():
                if value is None and skip_none: continue
                if flag and isinstance(value, bool):
                    self.flag(name, value)
                else:
                    buff.append(f'{name}={fmt(value)}')
        return self

    def attr(self, obj: Any, name: str, default: Any = missing, /, skip_none: bool = True, fmt: Format = None, flag: bool = False) -> Self:
        value = getattr(obj, name, default)
        if value is missing: return self
        return self.keyword(name, value, skip_none=skip_none, fmt=fmt, flag=flag)

    def flag(self, name: str, value: Any, /, skip: bool = None) -> Self:
        if value is skip: return self
        value = bool(value)
        if value is skip: return self
        return self.append(f'{"+" if value else "-"}{name}')

    def __str__(self) -> str:
        return self.sep.join(self.buff)


def tie_call(cls: type, slot: str):
    member = getattr(cls, slot, None)
    if isinstance(member, MemberDescriptorType):
        cls.__call__ = property(member.__get__)
    elif isinstance(member, property):
        cls.__call__ = property(member.fget)
    else:
        raise TypeError(f'Cannot tie __call__ to member [{slot}]: {member!r}')


def join_str(*args: str, sep: str = ', ') -> str:
    return sep.join(arg for arg in args if arg)


_get = dict.get
_getitem = dict.__getitem__
_setitem = dict.__setitem__
_delitem = dict.__delitem__
_iter = dict.__iter__
_items = dict.items
_contains = dict.__contains__


deleted = object()


class NewSpec(dict[str, Any]):

    __slots__ = ('origin',)

    origin: Mapping[str, Any]

    def __init__(self, origin: Mapping[str, Any]):
        dict.__init__(self)
        self.origin = origin

    def __len__(self) -> int:
        if dict.__len__(self) == 0:
            return len(self.origin)
        i = 0
        for _ in self:
            i += 1
        return i

    def __iter__(self) -> Iterable[str]:
        origin = self.origin
        for key in origin:
            val = _get(self, key, missing)
            if val is not deleted:
                yield key
        for key, value in _items(self):
            if value is not deleted and key not in origin:
                yield key

    def __contains__(self, item: str) -> bool:
        return _get(self, item, missing) is not deleted or item in self.origin

    def __getitem__(self, item: str) -> Any:
        val = _get(self, item, missing)
        if val is missing:
            return self.origin[item]
        if val is deleted:
            raise KeyError(item)
        return val

    def __setitem__(self, key: str, value):
        _setitem(self, key, value)

    def __delitem__(self, key: str):
        val = _get(self, key, missing)
        if val is deleted:
            raise KeyError(key)
        if val is missing:
            if key in self.origin:
                _setitem(self, key, deleted)
            else:
                raise KeyError(key)
        else:
            _setitem(self, key, deleted)

    def keys(self) -> set[str]:
        return set(self)

    def items(self) -> Iterable[tuple[str, Any]]:
        origin = self.origin
        for key, value in origin:
            nv = _get(self, key, missing)
            if nv is missing:
                yield key, value
            elif nv is not deleted:
                yield key, nv
        for key, value in _items(self):
            if value is not deleted and key not in origin:
                yield key, value

    def values(self) -> Iterable[Any]:
        origin = self.origin
        for key, value in origin:
            nv = _get(self, key, missing)
            if nv is missing:
                yield value
            elif nv is not deleted:
                yield nv
        for key, value in _items(self):
            if value is not deleted and key not in origin:
                yield value

    def pop(self, item: str, default=missing) -> None:
        val = _get(self, item, missing)
        if val is deleted:
            if default is missing:
                raise KeyError(item)
            return default
        if val is missing:
            if item in self.origin:
                _setitem(self, item, deleted)
                return self.origin[item]
            if default is missing:
                raise KeyError(item)
            return default
        _setitem(self, item, deleted)
        return val

    def copy(self) -> Self:
        copy = NewSpec(self.origin)
        copy.update(_items(self))
        return copy

    def collapse(self) -> dict[str, Any]:
        copy = dict(self.origin)
        for key, value in _items(self):
            if value is deleted:
                copy.pop(key, None)
            else:
                copy[key] = value
        return copy

    def get(self, key, default=None, /):
        val = _get(self, key, missing)
        if val is missing:
            return self.origin.get(key, default)
        if val is deleted:
            return default
        return val

    def clear(self):
        dict.clear(self)
        for key in self.origin:
            _setitem(self, key, deleted)

    def setdefault(self, key, default=None, /):
        val = _get(self, key, missing)
        if val is deleted:
            _setitem(self, key, default)
            return default
        if val is missing:
            val = self.origin.get(key, missing)
            if val is missing:
                _setitem(self, key, default)
                return default
            return val
        return val

    def expand(self, *keys: str, **defaults) -> dict[str, Any]:
        expanded = {}
        if keys:
            for key in keys:
                val = self.get(key, missing)
                if val is not missing:
                    expanded[key] = val
        if defaults:
            for key, default in defaults.items():
                val = self.get(key, missing)
                expanded[key] = default if val is missing else val
        return expanded

    def defaults(self, *specs: Optional[Keywords]) -> Self:
        if specs:
            setdefault = self.setdefault
            for spec in specs:
                if spec:
                    for key, val in spec.items():
                        setdefault(key, val)
        return self

    def merge(self, *specs: Optional[Keywords]) -> Self:
        if specs:
            update = self.update
            for spec in specs:
                if spec:
                    update(spec)
        return self

    def show_keywords(self):
        return ', '.join(f'{k}={v!r}' for k, v in self.items())

    def __repr__(self):
        return 'Spec(' + self.show_keywords() + ')'
