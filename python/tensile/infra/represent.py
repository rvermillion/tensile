#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import Any, Callable, Iterable, Optional


def repr_kw(key: Optional[str], val: Any, fmt: Callable[[Any], str] = repr) -> str:
    if key is None:
        return fmt(val)
    elif val is True:
        return f'+{key}'
    elif val is False:
        return f'-{key}'
    return f'{key}={fmt(val)}'


def join_kwargs(kwargs: Optional[dict[Optional[str], Any]]) -> str:
    return ', '.join(repr_kw(key, val) for key, val in kwargs.items()) if kwargs else ''


def join_args(args: Optional[str], kwargs: Optional[dict[Optional[str], Any]]) -> str:
    if args:
        if kwargs:
            return args + ', ' + join_kwargs(kwargs)
        return args
    elif kwargs:
        return join_kwargs(kwargs)
    else:
        return ''


class Representable:

    __slots__ = ()

    def __repr__(self) -> str:
        # noinspection PyBroadException
        try:
            return self._repr()
        except:
            return object.__repr__(self)

    def _repr(self, **options) -> str:
        args = self._repr_args(**options)
        kwargs = self._repr_kwargs(**options)
        r = self._repr_type(**options) + '(' + join_args(args, kwargs) + ')'
        return self._repr_shorten(r, **options)

    def _repr_type(self, **options) -> str:
        args = self._repr_type_args(**options)
        if args:
            return self.__class__.__qualname__ + '[' + args + ']'
        return self.__class__.__qualname__

    def _repr_type_args(self, **options) -> str:
        return ''

    def _repr_args(self, **options) -> str:
        return ', '.join(map(repr, self._repr_arg_items(**options)))

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _repr_arg_items(self, **options) -> Iterable[Any]:
        return ()

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        return None

    def _repr_attrs(self, *attrs: str) -> dict[str, Any]:
        return {a : getattr(self, a, None) for a in attrs}

    def _repr_shorten(self, representation: str, maxlen: int = None, **options) -> str:
        if maxlen is None: maxlen = self._repr_maxlen
        return representation[:maxlen - 3] + '...' if len(representation) > maxlen else representation

    _repr_maxlen: int = 100

    def __format__(self, format_spec):
        if format_spec == '':
            return self.__str__()
        if format_spec == 's':
            return self._repr(short=True)
        if format_spec[0] in '0123456789':
            return self._repr(maxlen=int(format_spec))
        if method := getattr(self, f'_format_{format_spec}', None):
            return method()
        raise ValueError(f'unsupported format: {format_spec!r}')


