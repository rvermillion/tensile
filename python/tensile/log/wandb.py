#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import wandb

from ..common import *
from .base import Logging


@provides(Logging, 'wandb')
class WandBLogging(Logging):

    __slots__ = ()

    def configure(self, project: str, config: dict[str, Any]) -> None:
        wandb.init(project=project, config=config)

    def log(self, info: dict[str, Any], **kwargs) -> None:
        pass


