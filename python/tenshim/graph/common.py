#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, Callable, Iterable, Optional, Sequence, TypeAlias, Union, TYPE_CHECKING

from .. import ten

if TYPE_CHECKING:
    import tenshim.graph.tensor

TensorType: TypeAlias = 'tenshim.graph.tensor.Tensor'

Array: TypeAlias = ten.Array
DType: TypeAlias = ten.DType
Shape: TypeAlias = ten.Shape

Index: TypeAlias = Union[int, Array, slice, Ellipsis, None]
Indices: TypeAlias = Union[Index, tuple[Index, ...]]

Axes: TypeAlias = tuple[int, ...]
AxisChoice: TypeAlias = Union[None, int, Sequence[int]]


Functional: TypeAlias = Callable[[Array], Array]


def repr_arg(arg: Any) -> str:
    return repr(arg)

def repr_item(item: tuple[str, Any]) -> str:
    return f'{item[0]}={item[1]!r}'


class Base:

    __slots__ = ()

    def _repr_type(self) -> str:
        return self.__class__.__name__

    def _repr_arg(self) -> str:
        return ', '.join(map(repr_arg, self._repr_args()))

    def _repr_args(self) -> Iterable:
        return ()

    def _repr_items(self) -> Iterable[tuple[str, Any]]:
        if items := self._repr_item_dict():
            return items.items()
        return ()

    def _repr_item_dict(self) -> Optional[dict[str, Any]]:
        return None

    def __repr__(self):
        args = self._repr_arg()
        items = ', '.join(map(repr_item, self._repr_items()))
        if args:
            if items:
                args += ', ' + items
        else:
            args = items
        return f'{self._repr_type()}({args})'
