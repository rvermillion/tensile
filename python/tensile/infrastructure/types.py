#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

import builtins
import sys
from enum import Enum
import types, typing
from types import (NoneType, GenericAlias, MemberDescriptorType, UnionType)
from typing import (AbstractSet, Annotated, Annotated, Any, Callable, ClassVar, Collection, ForwardRef, Generic,
                    Iterable, Iterator,
                    Mapping, MutableSequence, Optional, Protocol, Self, Sequence, TYPE_CHECKING, TypeVar,
                    Union, get_args, get_origin)

_sysver = sys.version_info[:2]

C = TypeVar('C', bound=Callable)
T = TypeVar('T')
U = TypeVar('U', contravariant=True)
X = TypeVar('X')
Y = TypeVar('Y', covariant=True)


Keywords = dict[str, Any]

Getter = Callable[[T], Y]
Setter = Callable[[T, X], None]
Deleter = Callable[[T], None]
Coercer = Callable[[T, Any], Y]
Initter = Callable[[T, 'Spec'], None]

Predicate = Callable[[T], bool]
Transform = Callable[[U], X]
IsSetter = Callable[[T], bool]

Equiv = Callable[[X, X], bool]


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


class Missing:

    def __repr__(self) -> str:
        return 'missing'


missing = Missing()


class MetaError(RuntimeError):

    pass


class Spec(dict[str, Any]):

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

    @classmethod
    def combine(cls, *specs: Optional[Keywords]) -> Self:
        return cls().merge(*specs)

    @classmethod
    def coerce(cls, arg: Any) -> Self:
        if arg is None: return cls()
        if isinstance(arg, cls): return arg
        if isinstance(arg, Mapping): return cls(arg)
        raise TypeError(f'Cannot coerce {arg!r} to Spec')



def is_protocol(cls: type) -> bool:
    return getattr(cls, '_is_protocol', False)


def is_runtime_protocol(cls: type) -> bool:
    return getattr(cls, '_is_runtime_protocol', False)


def is_runtime_class(cls: Any) -> bool:
    return isinstance(cls, type) and (not is_protocol(cls) or is_runtime_protocol(cls))


if _sysver >= (3, 10):
    from typing import _GenericAlias

    def is_generic_alias(obj: Any) -> bool:
        return isinstance(obj, _GenericAlias)

else:
    def is_generic_alias(obj: Any) -> bool:
        return False


def is_union(obj: Any) -> bool:
    return isinstance(obj, UnionType)


def is_forward_ref(obj: Any) -> bool:
    return isinstance(obj, ForwardRef)


def get_forward_ref_name(obj: Any) -> str:
    return obj.__forward_arg__

# __all__ = [
#     'NoneType',
#     'MemberDescriptorType',
#     'Mapping',
#     'Sequence'
#     'Spec',
#     'is_protocol'
# ]