#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from ..api import Array

def cholesky(a: Array, upper: bool = False, **kwargs) -> Array: ...


def solve_triangular(a: Array, b: Array, *, upper: bool = False, **kwargs) -> Array: ...

