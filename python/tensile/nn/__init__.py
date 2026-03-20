#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import TYPE_CHECKING

from .. import ten

if TYPE_CHECKING:
    # from ..shims.numpy.nn import *
    from ..shims.mlx.nn import *
    # from ..shims.torch.nn import *

else:
    ten_kind: str = ten.ten_kind

    if ten_kind == 'numpy':
        raise NotImplementedError('Numpy backend is not yet supported!')

    elif ten_kind == 'mlx':
        from ..shims.mlx.nn import *

    elif ten_kind == 'torch':
        from ..shims.torch.nn import *

    else:
        raise NotImplementedError(f'No tensor shim named {ten_kind} found!')

from . import module, activations, layers, losses, quantization

from .args import ModuleArgs
from .module import CompiledModule, Module
from .context import ForwardContext
from .instrument import Instrument
from .patch import Patch
