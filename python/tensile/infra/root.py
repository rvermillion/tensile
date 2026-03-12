#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from .log import Logging
from .represent import Representable



class RootObject(Representable, Logging):

    __slots__ = ()

    @property
    def _repr_verbose(self) -> int:
        return self.verbose


__all__ = [
    'Logging',
    'Representable',
    'RootObject',
]