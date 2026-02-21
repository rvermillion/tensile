#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from os import environ
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mlx import core, types

else:
    default_shim_name: str = environ.get('TENSILE', 'mlx')

    if default_shim_name == 'numpy':
        from .numpy import core, types

    elif default_shim_name == 'mlx':
        from .mlx import core, types

    elif default_shim_name == 'torch':
        from .torch import core, types

    else:
        raise NotImplementedError(f'No tensor shim named {default_shim_name} found!')
