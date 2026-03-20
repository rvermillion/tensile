#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *


class Training(Protocol):

    @property
    def current_batch_data(self) -> Optional[tuple[Array, ...]]: ...
