#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED
from typing import Generic, Iterable, Optional, TypeVar
from .. import ten

from .common import Array, Base, DType, Index, Indices, Shape, TensorType, RegionType


class Patch(Base):

    __slots__ = ()

    region: RegionType

    @property
    def target_shape(self) -> Shape:
        return self.region.base

    def apply(self, target: Array) -> None:
        if self.target_shape != target.shape:
            raise ValueError(f'Target shape {target.shape} does not match patch shape {self.target_shape}')
        self._apply(target)

    def _apply(self, target: Array) -> None:
        raise NotImplementedError()

    @classmethod
    def create(cls, region: RegionType, data: Array = None) -> 'Patch':
        return DataPatch(region, data=data)


class DataPatch(Patch):

    __slots__ = ('region', 'data',)

    data: Array

    def __init__(self, region: RegionType, data: Array = None):
        self.region = region
        self.data = data

    def _apply(self, target: Array) -> None:
        target[self.region.key] = self.data
