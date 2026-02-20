#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from os import environ

default_shim_name: str = environ.get('TENSILE', 'mlx')

if default_shim_name == 'numpy':
    from .numpy import core

elif default_shim_name == 'mlx':
    from .mlx import core

elif default_shim_name == 'torch':
    from .torch import core

else:
    raise NotImplementedError(f'No tensor shim named {default_shim_name} found!')
