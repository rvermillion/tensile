#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..module import Module
from ..quantization import QuantizationArgs
from ..transform import ModuleTransforms


QuantizeTransforms = ModuleTransforms.get_category('quantize', create=True)

def default_quantize_transform(module: Module, /, args: QuantizationArgs = None, **options) -> Optional[Module]:
    if to_quantized := getattr(module, 'to_quantized', None):
        return to_quantized(bits=args.bits, group_size=args.group_size, mode=args.mode)
    return None


QuantizeTransforms.set_default(default_quantize_transform)


__all__ = [
    'QuantizeTransforms'
]