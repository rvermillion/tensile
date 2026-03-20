#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *


class Logging(Object):

    __slots__ = ()


    def configure(self, project: str, config: dict[str, Any]) -> None:
        raise NotImplementedError()

    def log(self, info: dict[str, Any], **kwargs) -> None:
        raise NotImplementedError()
