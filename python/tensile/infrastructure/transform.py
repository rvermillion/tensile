#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from types import EllipsisType, NoneType
from typing import Any, Generic, TYPE_CHECKING, TypeVar, final

import tensile.infrastructure as infra

from .function import identity, none, compose_all
from .meta import meta_configure_coerce
from .types import (Callable, Mapping, Missing, PredicateFunction, PredicateLike, Sequence, SupportsGetItem,
                    TransformFunction, TransformLike, missing)
from .util import name_function, spread_mapping, tie_call

if TYPE_CHECKING:
    from .predicate import Predicate, Predicates

T = TypeVar('T')
U = TypeVar('U', contravariant=True)
X = TypeVar('X', covariant=True)


class Transform(Generic[U, X]):

    __slots__ = ('transform',)

    transform: TransformFunction[U, X]

    def __init__(self, transform: TransformFunction[U, X]) -> None:
        self.transform = transform

    @property
    def __name__(self) -> str:
        return self.transform.__name__

    @final
    def __call__(self, value: U) -> X:
        return self.transform(value)

    def _eq_tuple(self) -> tuple:
        return (self.transform,)

    def __eq__(self, other: Any) -> bool:
        return self is other or (
            isinstance(other, self.__class__) and
            self._eq_tuple() == other._eq_tuple()
        )

    def __hash__(self) -> int:
        return hash(self._eq_tuple())

    def __add__(self, other: TransformLike[X, T]) -> 'Transform[U, T]':
        # noinspection PyTypeChecker
        return Transforms.chain(self, other)

    def describe(self, arg: str) -> str:
        return f'{self.__name__}({arg})'

    is_constant: bool = False


tie_call(Transform, 'transform')


def full_coerce(self, spec: Any = None, /, **kwargs) -> Transform:
    if spec is None:
        spec = kwargs
    elif kwargs and isinstance(spec, Mapping):
        kwargs.update(spec)
        spec = kwargs
    return coerce(spec)


meta_configure_coerce(Transform, full_coerce)
# meta_configure_coerce(TransformFunction, full_coerce)


def constant(value: T) -> TransformFunction[Any, T]:
    if value is None:
        return none

    # noinspection PyUnusedLocal
    def transform(arg: Any) -> X:
        return value
    return name_function(transform, f'constant[{value!r}]')


class ConstantTransform(Transform[Any, X]):

    __slots__ = ('value', )

    value: X

    def __init__(self, value: X):
        super().__init__(constant(value))
        self.value = value

    def describe(self, arg: str) -> str:
        return repr(self.value)

    is_constant = True


class IdentityTransform(Transform[T, T]):

    __slots__ = ()

    def __init__(self):
        super().__init__(identity)

    def describe(self, arg: str) -> str:
        return arg

    is_constant = True


def get_attr(name: str, default: Any = missing, desc: str = '') -> TransformFunction:
    if default is missing:
        def getter(this: Any) -> Any:
            return getattr(this, name)
        if not desc:
            desc = f'get_attr[{name}]'
    else:
        def getter(this: Any) -> Any:
            return getattr(this, name, default)
        if not desc:
            desc = f'get_attr[{name}, {default!r}]'
    return name_function(getter, desc)


def get_item(item: Any, default: Any = missing, desc: str = '') -> TransformFunction[SupportsGetItem, Any]:
    if default is missing:
        def getter(this: Any) -> Any:
            return this[item]
        if not desc:
            desc = f'get_item[{item!r}]'
    else:
        def getter(this: Any) -> Any:
            try:
                return this[item]
            except (KeyError, IndexError):
                return default
        if not desc:
            desc = f'get_item[{item!r}, {default!r}]'
    return name_function(getter, desc)



class AttrTransform(Transform[Any, T]):

    __slots__ = ('name', 'default')

    name: str
    default: Missing|T

    def __init__(self, name: X, default: Missing|T = missing) -> None:
        super().__init__(get_attr(name, default))
        self.name = name
        self.default = default

    def describe(self, arg: str) -> str:
        return f'{arg}.{self.name}' if self.default is missing else f'getattr({arg}, {self.name!r}, {self.default!r})'


class ItemTransform(Transform[T, T]):

    __slots__ = ('item', 'default')

    item: str
    default: Missing|T

    def __init__(self, key: X, default: Missing|T = missing) -> None:
        super().__init__(get_item(key, default))
        self.item = key
        self.default = default

    def describe(self, arg: str) -> str:
        return f'{arg}[{self.item!r}]' if self.default is missing else f'{arg}.get({self.item!r}, {self.default!r})'


def _combine_chain(txfs: Sequence[Transform[U, X]]) -> TransformFunction[U, X]:
    transforms = []
    for i, this in enumerate(txfs):
        if this.is_constant:
            del transforms[:i]
        transforms.append(this)
    return compose_all(transforms)


class ChainedTransform(Transform[Any, Any]):

    __slots__ = ('transforms', )

    transforms: tuple[Transform[Any, Any], ...]

    def __init__(self, transforms: tuple[Transform[Any, Any], ...]) -> None:
        super().__init__(_combine_chain(transforms))
        self.transforms = transforms

    def describe(self, arg: str) -> str:
        for t in self.transforms:
            arg = t.describe(arg)
        return arg


def where(condition: PredicateFunction[T], then: TransformFunction[T, X], otherwise: TransformFunction[T, X] = None) -> TransformFunction[T, X]:
    from .predicate import always, never
    if otherwise is None: otherwise = identity
    if condition is always: return then
    if condition is never: return otherwise
    if then is otherwise: return then

    def transform(arg: T) -> Any:
        return then(arg) if condition(arg) else otherwise(arg)

    return name_function(transform, f'where[{condition.__name__} ? {then.__name__} : {otherwise.__name__}]')


class ConditionalTransform(Transform[U, X]):

    __slots__ = ('condition', 'then', 'otherwise')

    condition: 'Predicate[U]'
    then: Transform[U, X]
    otherwise: Transform[U, X]

    def __init__(self, condition: 'Predicate[U]', then: Transform[U, X], otherwise: Transform[U, X]):
        super().__init__(where(condition.evaluate, then.transform, otherwise.transform))
        self.condition = condition
        self.then = then
        self.otherwise = otherwise

    def describe(self, arg: str) -> str:
        return self.condition.describe(arg) + ' ? ' + self.then.describe(arg) + ' : ' + self.otherwise.describe(arg)


def coerce(spec: TransformLike[U, X]) -> Transform[Any, Any]:
    if isinstance(spec, Transform): return spec
    if spec is None or spec is identity: return Transforms.identity
    if callable(spec): return Transform(spec)
    if isinstance(spec, Mapping):
        if len(spec) == 1:
            k, = spec.keys()
            if k != 'kind':
                value = spec[k]
                if factory := named_factories.get(k):
                    return factory(value)
                raise ValueError(f"Invalid key: {k!r}")
    if isinstance(spec, str):
        if singleton := named_singletons.get(spec):
            return singleton
        if factory := named_factories.get(spec):
            return factory(None)
        raise ValueError(f"Invalid string spec: {spec!r}")
    if isinstance(spec, Sequence):
        return Transforms.chain(*spec)
    raise ValueError(f"Cannot convert {spec!r} to a valid type or spec: {type(spec)}")


class Transforms:

    __slots__ = ()

    @staticmethod
    def constant(value: T) -> Transform[Any, T]:
        return Transforms.none if value is None else ConstantTransform(value)

    @staticmethod
    def get_attr(name: str, default: Missing|T = missing) -> Transform[Any, T]:
        return AttrTransform(name, default)

    @staticmethod
    def get_path(path: str, default: Missing|T = missing) -> Transform[Any, T]:
        return Transforms.chain(*(Transforms.get_attr(name, default=default) for name in path.split('.')))

    @staticmethod
    def get_item(item: T, default: Missing | X = missing) -> Transform[SupportsGetItem[T, X], X]:
        return ItemTransform(item, default)

    @staticmethod
    def chain(*transforms: TransformLike[Any, Any]) -> Transform[Any, Any]:
        return ChainedTransform(tuple(coerce(t) for t in transforms))

    coerce = staticmethod(coerce)

    @staticmethod
    def append(suffix: str) -> Transform[str, str]:
        return Transform(lambda x: x + suffix)

    @staticmethod
    def prepend(prefix: str) -> Transform[str, str]:
        return Transform(lambda x: prefix + x)

    @staticmethod
    def where(condition: PredicateLike[T], then: TransformLike[T, X], otherwise: TransformLike[T, X] = None) -> Transform[T, X]:
        return ConditionalTransform(infra.Predicates.coerce(condition), coerce(then), coerce(otherwise))

    @staticmethod
    def transform(transform: TransformLike[U, X]) -> Transform[U, X]:
        return coerce(transform)

    @staticmethod
    def named(name: str) -> Transform:
        t = getattr(Transforms, name)
        if isinstance(t, Transform):
            return t
        raise ValueError(f"Invalid transform name: {name}")

    none: Transform[Any, NoneType] = ConstantTransform(None)
    ellipsis: Transform[Any, EllipsisType] = ConstantTransform(Ellipsis)
    true: Transform[Any, bool] = ConstantTransform(True)
    false: Transform[Any, bool] = ConstantTransform(False)
    identity: Transform[Any, Any] = IdentityTransform()
    repr: Transform[Any, str] = Transform(repr)
    str: Transform[Any, str] = Transform(str)
    int: Transform[Any, int] = Transform(int)
    float: Transform[Any, float] = Transform(float)


named_factories: dict[str, Callable[[Any], Transform]] = {
    'none': Transforms.none,
    'attr': Transforms.get_attr,
    'key': Transforms.get_item,
}

named_singletons: dict[str, Transform] = {
    name: txf for name, txf in Transforms.__dict__.items() if isinstance(txf, Transform)
}


__all__ = [
    'Transform',
    'Transforms',
]