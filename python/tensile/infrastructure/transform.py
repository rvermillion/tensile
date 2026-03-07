#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from types import EllipsisType, NoneType
from typing import Any, Generic, Iterable, Sequence, TypeVar
from .function import identity, none, compose_all
from .types import Missing, SupportsGetItem, Transform, missing
from .util import name_function

T = TypeVar('T')
U = TypeVar('U', contravariant=True)
X = TypeVar('X', covariant=True)


class TransformObject(Generic[U, X]):

    __slots__ = ('transform',)

    transform: Transform[U, X]

    def __init__(self, transform: Transform[U, X]) -> None:
        self.transform = transform

    @property
    def __name__(self) -> str:
        return self.transform.__name__

    def __call__(self, value: U) -> X:
        return self.transform(value)

    def describe(self, arg: str) -> str:
        return f'{self.__name__}({arg})'

    is_constant: bool = False


class ConstantTransform(TransformObject[Any, X]):

    __slots__ = ('value', )

    value: X

    def __init__(self, value: X):
        if value is None:
            transform = none
        else:
            def transform(arg: Any) -> X:
                return value
            transform = name_function(transform, f'constant[{value!r}]')
        super().__init__(transform)
        self.value = value

    def describe(self, arg: str) -> str:
        return repr(self.value)

    is_constant = True


class IdentityTransform(TransformObject[T, T]):

    __slots__ = ()

    def __init__(self):
        super().__init__(identity)

    def describe(self, arg: str) -> str:
        return arg

    is_constant = True


def get_attr(name: str, default: Any = missing, desc: str = '') -> Transform:
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


def get_item(item: Any, default: Any = missing, desc: str = '') -> Transform[SupportsGetItem, Any]:
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



class AttrTransform(TransformObject[T, T]):

    __slots__ = ('name', 'default')

    name: str
    default: Missing|T

    def __init__(self, name: X, default: Missing|T = missing) -> None:
        super().__init__(get_attr(name, default))
        self.name = name
        self.default = default

    def describe(self, arg: str) -> str:
        return f'{arg}.{self.name}' if self.default is missing else f'getattr({arg}, {self.name!r}, {self.default!r})'


class ItemTransform(TransformObject[T, T]):

    __slots__ = ('item', 'default')

    item: str
    default: Missing|T

    def __init__(self, key: X, default: Missing|T = missing) -> None:
        super().__init__(get_item(key, default))
        self.item = key
        self.default = default

    def describe(self, arg: str) -> str:
        return f'{arg}[{self.item!r}]' if self.default is missing else f'{arg}.get({self.item!r}, {self.default!r})'


def _combine_compose(txfs: Sequence[TransformObject[U, X]]) -> Transform[U, X]:
    transforms = []
    for i, this in enumerate(txfs):
        if this.is_constant:
            del transforms[:i]
        transforms.append(this)
    return compose_all(transforms)


class ComposedTransform(TransformObject[Any, Any]):

    __slots__ = ('transforms', )

    transforms: tuple[TransformObject[Any, Any], ...]

    def __init__(self, transforms: tuple[TransformObject[Any, Any], ...]) -> None:
        super().__init__(_combine_compose(transforms))
        self.transforms = transforms

    def describe(self, arg: str) -> str:
        for t in self.transforms:
            arg = t.describe(arg)
        return arg


class Transforms:

    __slots__ = ()

    @staticmethod
    def constant(value: T) -> TransformObject[Any, T]:
        return Transforms.none if value is None else ConstantTransform(value)

    @staticmethod
    def get_attr(name: str, default: Missing|T = missing) -> TransformObject[Any, T]:
        return AttrTransform(name, default)

    @staticmethod
    def get_item(item: T, default: Missing | X = missing) -> TransformObject[SupportsGetItem[T, X], X]:
        return ItemTransform(item, default)

    @staticmethod
    def compose(*transforms: TransformObject[Any, Any]) -> Transform[Any, Any]:
        return ComposedTransform(tuple(Transforms.coerce(t) for t in transforms))

    @staticmethod
    def coerce(spec: Any) -> TransformObject[Any, Any]:
        if spec is None: return Transforms.identity
        if isinstance(spec, TransformObject): return spec
        if callable(spec): return TransformObject(spec)
        raise ValueError(f"Cannot convert {spec!r} to a valid type or spec: {type(spec)}")

    @staticmethod
    def append(suffix: str) -> TransformObject[str, str]:
        return TransformObject(lambda x: x + suffix)

    @staticmethod
    def prepend(prefix: str) -> TransformObject[str, str]:
        return TransformObject(lambda x: prefix + x)

    none: TransformObject[Any, NoneType] = ConstantTransform(None)
    ellipsis: TransformObject[Any, EllipsisType] = ConstantTransform(Ellipsis)
    true: TransformObject[Any, bool] = ConstantTransform(True)
    false: TransformObject[Any, bool] = ConstantTransform(False)
    identity: TransformObject[Any, Any] = IdentityTransform()
    repr: TransformObject[Any, str] = TransformObject(repr)
    str: TransformObject[Any, str] = TransformObject(str)
    int: TransformObject[Any, int] = TransformObject(int)
    float: TransformObject[Any, float] = TransformObject(float)

    __all__ = [
    'Transforms',
]