#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from typing import (Any, Callable, Iterable, Optional, Self, Sequence,
                    TypeAlias, Union, TYPE_CHECKING)

from ..shims import core as ten
from ..infrastructure.log import Logging
from ..infrastructure.represent import Representable



if TYPE_CHECKING:
    import tensile.graph.tensor
    import tensile.graph.region
    import tensile.graph.patch

TensorType: TypeAlias = 'tensile.graph.tensor.Tensor'
RegionType: TypeAlias = 'tensile.graph.region.Region'
PatchType: TypeAlias = 'tensile.graph.patch.Patch'

Array: TypeAlias = ten.Array
DType: TypeAlias = ten.DType
Shape: TypeAlias = ten.Shape
Slice: TypeAlias = slice

Index: TypeAlias = Union[int, Array, slice, Ellipsis, None]
Indices: TypeAlias = Union[Index, tuple[Index, ...]]

Axes: TypeAlias = tuple[int, ...]
AxisChoice: TypeAlias = Union[None, int, Sequence[int]]


Functional: TypeAlias = Callable[[Array], Array]


def repr_arg(arg: Any) -> str:
    return repr(arg)

def repr_item(item: tuple[str, Any]) -> str:
    return f'{item[0]}={item[1]!r}'


auto_validate: bool = True


class Base(Logging, Representable):

    __slots__ = ()

    _auto_validate: bool = auto_validate

    def __init_subclass__(cls, validate: bool = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if validate is not None:
            cls._auto_validate = validate

    def _postinit(self) -> None:
        if self._auto_validate:
            self.validate()
        self.debug(f'created {self:s}')

    def validate(self, warn: bool = False) -> None:
        if warn:
            try:
                self._validate()
            except Exception as e:
                self.warn(f'Validation failed for {self:s}: {e!r}')
                return
        self._validate()
        self.debug(f'validated {self:s}')

    def _validate(self) -> None:
        pass

    def _repr_type(self, short: bool = False, **options) -> str:
        return self.__class__.__name__

    def _repr_arg(self, short: bool = False) -> str:
        if short: return ''
        return ', '.join(map(repr_arg, self._repr_args(short=short)))

    def _repr_args(self, short: bool = False) -> Iterable:
        return ()

    def _repr_items(self, short: bool = False) -> Iterable[tuple[str, Any]]:
        if items := self._repr_item_dict(short=short):
            return items.items()
        return ()

    def _repr_item_dict(self, short: bool = False) -> Optional[dict[str, Any]]:
        return None

    # noinspection PyMethodMayBeStatic
    def _repr_short_arg(self) -> str:
        return ''

    def _repr_full(self, short: bool = None, maxlen: int = None) -> str:
        args = self._repr_arg(short=short)
        items = ', '.join(map(repr_item, self._repr_items(short=short)))
        if args:
            if items:
                args += ', ' + items
        else:
            args = items
        return f'{self._repr_type(short=short)}({args})'

    def _repr_shorten(self, r: str, maxlen: int = None) -> str:
        if maxlen is None: maxlen = self._repr_maxlen
        return r[:maxlen - 3] + '...' if len(r) > maxlen else r

    def _repr(self, short: bool = None, maxlen: int = None) -> str:
        if maxlen is None: maxlen = self._repr_maxlen
        if short is None: short = maxlen <= 20
        r = self._repr_full(short=short, maxlen=maxlen)
        return self._repr_shorten(r, maxlen)

    _repr_maxlen: int = 100

    def __format__(self, format_spec):
        if format_spec == '':
            return self.__repr__()
        if format_spec == 's':
            return self._repr(short=True)
        if method := getattr(self, f'_format_{format_spec}', None):
            return method()
        raise ValueError(f'unsupported format: {format_spec!r}')

    def __repr__(self):
        return self._repr()

    @classmethod
    def new(cls, *args, **kwargs) -> Self:
        cls._validate_new_args(*args, **kwargs)
        # noinspection PyArgumentList
        obj = cls(*args, **kwargs)
        obj._postinit()
        return obj

    @classmethod
    def _validate_new_args(cls, *args, **kwargs) -> None:
        pass

