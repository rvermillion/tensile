#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import Optional
from ..module import Module
from ..transform import ModuleTransform, ModuleTransforms

FuseTransforms = ModuleTransforms.get_category('fuse', create=True)

def default_fuse_transform(module: Module, /, **options) -> Optional[Module]:
    if to_fused := getattr(module, 'to_fused', None):
        return to_fused(**options)
    return None


default_fuse_transform: ModuleTransform


FuseTransforms.set_default(default_fuse_transform)

