#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from .types import *

def gelu(x: Array) -> Array: ...

def relu(x: Array) -> Array: ...

def sigmoid(x: Array) -> Array: ...

def silu(x: Array) -> Array: ...

__all__ = [
    'gelu',
    'relu',
    'sigmoid',
    'silu'
]
