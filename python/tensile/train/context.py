#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..nn.context import ForwardContext
from .training import Training


class TrainingContext(ForwardContext):

    __slots__ = ('training', )

    training: Annotated[Optional[Training], field()]

    @property
    def current_batch_data(self) -> Optional[tuple[Array, ...]]:
        if training := self.training:
            return training.current_batch_data
        return None
