#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from .types import Array

def cholesky(a: Array, upper: bool = False, **kwargs) -> Array: ...


def solve_triangular(a: Array, b: Array, *, upper: bool = False, **kwargs) -> Array: ...

