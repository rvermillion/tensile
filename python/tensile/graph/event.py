#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from typing import Any, Iterable, Optional, TypeVar
from .. import ten

from typing import Any, Optional
from .common import Array, Base, DType, Index, Indices, Shape, TensorType, RegionType
from .patch import Patch


class TensorEvent(Base):

    __slots__ = ('source', 'patch')

    source: TensorType
    patch: Patch
    region: RegionType

    def __init__(self, source: TensorType, patch: Patch):
        self.source = source
        self.patch = patch

    @property
    def region(self) -> RegionType:
        return self.patch.region

    def _repr_item_dict(self, short: bool = False) -> Optional[dict[str, Any]]:
        items = {'source': self.source, 'region': self.region}
        return items

    @classmethod
    def create(cls, source: TensorType, region: RegionType, data: Array = None) -> 'TensorEvent':
        patch = Patch.create(region, data=data)
        return TensorEvent(source, patch)


