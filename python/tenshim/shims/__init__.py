#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from os import environ

default_shim_name: str = environ.get('TENSHIM', 'numpy')

if default_shim_name == 'numpy':
    from . import numpy as default

elif default_shim_name == 'mlx':
    from . import mlx as default

elif default_shim_name == 'torch':
    from . import torch as default

else:
    raise NotImplementedError(f'No tensor shim named {default_shim_name} found!')
