#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import ParamSpec, final

from .root import RootObject
from .util import name_function, tie_call
from .types import Annotated, Any, Callable, Iterable, Sequence, TypeVar, Generic

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')


def identity(x: X) -> X:
    return x


def none(x: Any) -> None:
    return None


def chunk(seq: Sequence[X], chunk_size: int) -> Iterable[Sequence[X]]:
    cnt = len(seq)
    s = 0
    e = chunk_size
    while e < cnt:
        yield seq[s:e]
        s = e
        e += chunk_size
    yield seq[s:]


def compose(first: Callable[[X], Y], second: Callable[[Y], Z]) -> Callable[[X], Z]:
    if first is None or first is identity:
        return identity if second is None else second
    if second is None or second is identity:
        return first

    def composed(arg: X) -> Z:
        return second(first(arg))

    return name_function(composed, f'composed[{first.__name__}, {second.__name__}]')


def compose_all(fns: Sequence[Callable[[X], X]]) -> Callable[[X], X]:
    if fns:
        fns = [fn for fn in fns if fn is not identity and fn is not None]
        if fns:
            cnt = len(fns)
            if cnt == 1:
                return fns[0]
            elif cnt == 2:
                a, b = fns
                def composed(arg: X) -> X:
                    return b(a(arg))
            elif cnt == 3:
                a, b, c = fns
                def composed(arg: X) -> X:
                    return c(b(a(arg)))
            elif cnt == 4:
                a, b, c, d = fns
                def composed(arg: X) -> X:
                    return d(c(b(a(arg))))
            else:
                return compose_all([compose_all(f) for f in chunk(fns, min(4, (cnt + 3) // 4))])
            return name_function(composed, name=f'compose[{", ".join(fn.__name__ for fn in fns)}]')
    return identity


P = ParamSpec('P')
X = TypeVar("X")


class Function(RootObject, Generic[P, X]):

    __slots__ = ('call', '_name')

    call: Callable[P, X]
    _name: str

    def __init__(self, call: Callable[P, X], name: str = None):
        self.call = call
        self._name = name or call.__qualname__

    @final
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> X:
        return self.call(*args, **kwargs)

    def _repr_args(self, **options) -> str:
        return self._name


tie_call(Function, 'call')

