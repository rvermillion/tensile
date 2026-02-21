#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
import mlx.core as mx

from .types import Array

cpu_device = mx.Device(mx.cpu)
cpu_stream = mx.new_stream(cpu_device)

from mlx.core.linalg import (
    cholesky as builtin_cholesky,
    cholesky_inv,
    cross,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    inv,
    lu,
    lu_factor,
    norm,
    pinv,
    qr,
    solve as builtin_solve,
    solve_triangular as builtin_solve_triangular,
    svd as builtin_svd,
    tri_inv,
)


def svd(a: Array, compute_uv: bool = True, *, stream: None | mx.Stream | mx.Device = None) -> tuple[Array, Array, Array]:
    return builtin_svd(a, compute_uv=compute_uv, stream=cpu_stream if stream is None else stream)


def cholesky(a: Array, upper: bool = False, *, stream: None | mx.Stream | mx.Device = None) -> Array:
    return builtin_cholesky(a, upper=upper, stream=cpu_stream if stream is None else stream)


def solve_triangular(a: Array, b: Array, *, upper: bool = False, stream: None | mx.Stream | mx.Device = None) -> Array:
    return builtin_solve_triangular(a, b, upper=upper, stream=cpu_stream if stream is None else stream)


def solve(a: Array, b: Array, *, stream: None | mx.Stream | mx.Device = None) -> Array:
    return builtin_solve(a, b, stream=cpu_stream if stream is None else stream)
