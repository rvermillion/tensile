#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from .common import Optional
from .module import Module, ModuleArgs
from .transform import ModuleTransforms


class QuantizationArgs(ModuleArgs):
    group_size: int = 0
    bits: int = 8
    mode: str = 'affine'


class QuantizableModuleArgs(ModuleArgs):
    quantization: QuantizationArgs = None



__all__ = [
    'QuantizationArgs',
    'QuantizableModuleArgs',
]
