#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
import builtins
from enum import Enum
from types import (NoneType, GenericAlias, MemberDescriptorType)
from typing import (AbstractSet, Annotated, Annotated, Any, Callable, ClassVar, Collection, ForwardRef, Iterable,
                    Mapping, MutableSequence, Optional, Protocol, Self, Sequence, TYPE_CHECKING, TypeAlias, TypeVar,
                    Union, get_args, get_origin)
from typing import _GenericAlias

C = TypeVar('C', bound=Callable)
T = TypeVar('T')
U = TypeVar('U', contravariant=True)
X = TypeVar('X')
Y = TypeVar('Y', covariant=True)


Keywords: TypeAlias = dict[str, Any]

Getter: TypeAlias = Callable[[T], Y]
Setter: TypeAlias = Callable[[T, X], None]
Deleter: TypeAlias = Callable[[T], None]
Coercer: TypeAlias = Callable[[T, Any], Y]
Initter: TypeAlias = Callable[[T, 'Spec'], None]

Predicate: TypeAlias = Callable[[T], bool]
Transform: TypeAlias = Callable[[U], X]
IsSetter: TypeAlias = Callable[[T], bool]

Equiv: TypeAlias = Callable[[X, X], bool]


class Missing:

    def __repr__(self) -> str:
        return 'missing'


missing = Missing()


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
    def combine(cls, *specs: Optional[Keywords]) -> 'Spec':
        return cls().merge(*specs)


def is_protocol(cls: type) -> bool:
    return getattr(cls, '_is_protocol', False)


def is_runtime_protocol(cls: type) -> bool:
    return getattr(cls, '_is_runtime_protocol', False)


def is_runtime_class(cls: Any) -> bool:
    return isinstance(cls, type) and (not is_protocol(cls) or is_runtime_protocol(cls))


def is_generic_alias(obj: Any) -> bool:
    return isinstance(obj, _GenericAlias)


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