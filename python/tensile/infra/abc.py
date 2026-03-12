#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


from collections.abc import Collection
from .object import ObjectClass

ABCMeta = type(Collection)


class ABCClass(ObjectClass, ABCMeta):

    pass


