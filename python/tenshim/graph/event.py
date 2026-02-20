#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED
from typing import Any, Iterable, Optional, TypeVar
from .. import ten

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


