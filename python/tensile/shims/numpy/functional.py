#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

import numpy as np
from .types import *


def gelu(x: Array) -> Array:
    raise NotImplementedError()

def relu(x: Array) -> Array:
    return np.maximum(x, 0.0)

def sigmoid(x: Array) -> Array:
    return 1. / (1. + np.exp(-x))

def silu(x: Array) -> Array:
    raise NotImplementedError()


__all__ = [
    'gelu',
    'relu',
    'sigmoid',
    'silu'
]
