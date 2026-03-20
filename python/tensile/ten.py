#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


from os import environ
from typing import TYPE_CHECKING

ten_kind: str = environ.get('TENSILE', 'mlx').lower().strip()


if TYPE_CHECKING:
    from .shims.mlx.core import *
    from .shims.mlx.types import *

else:

    if ten_kind == 'numpy':
        from .shims.numpy.core import *
        from .shims.numpy.types import *

    elif ten_kind == 'mlx':
        from .shims.mlx.core import *
        from .shims.mlx.types import *

    elif ten_kind == 'torch':
        from .shims.torch.core import *
        from .shims.torch.types import *

    else:
        raise NotImplementedError(f'No tensor shim named {ten_kind} found!')

