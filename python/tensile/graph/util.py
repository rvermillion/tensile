#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import numpy as np
from typing import Callable
from .common import Array, DType, Shape, ten

def promote_types(a: DType, b: DType) -> DType:
    # For now we ignore
    return a


def broadcast_to(a: Array, *arrays: Array, shape: Shape = None) -> tuple[Array, ...]:
    if shape is None:
        raise ValueError('shape must be specified')
    return ten.broadcast_to(a, shape=shape), *(ten.broadcast_to(x, shape=shape) for x in arrays)


def broadcast(a: Array, b: Array, *arrays: Array) -> tuple[Array, ...]:
    shape = broadcast_shapes(a.shape, b.shape, *(x.shape for x in arrays))
    return broadcast_to(a, b, *arrays, shape=shape)


def broadcast_shapes(a: Shape, b: Shape, *others: Shape) -> Shape:
    def finish(r: Shape) -> Shape:
        return broadcast_shapes(r, *others) if others else r

    if not a:
        return finish(b)
    if not b:
        return finish(a)
    if a == b:
        return finish(a)
    m = -min(len(a), len(b))
    i = -1
    shape = []
    while i >= m:
        if a[i] == b[i] or b[i] == 1:
            shape.append(a[i])
        elif a[i] == 1:
            shape.append(b[i])
        else:
            raise ValueError(f'Cannot broadcast shapes {a} and {b}: {a[i]} != {b[i]}')
        i -= 1
    if -m < len(a):
        shape.extend(a[:i+1])
    elif -m < len(b):
        shape.extend(b[:i+1])
    n = finish(tuple(reversed(shape)))
    print(f'Broadcasting {a} and {b} and got: {n}')
    return n


def int_log2(x: Array, max_bits: int = None, check: bool = False) -> Array:
    dt = x.dtype
    if not ten.is_integer(dt):
        raise ValueError(f'Expected integer dtype, got {dt}')
    if check and ten.any(x < 1).item():
        raise ValueError(f'Expected positive values, got {ten.min(x).item()}')
    if max_bits is None:
        if dt == ten.int8 or dt == ten.uint8: max_bits = 8
        elif dt == ten.int16 or dt == ten.uint16: max_bits = 16
        elif dt == ten.int32 or dt == ten.uint32: max_bits = 32
        elif dt == ten.int64 or dt == ten.uint64: max_bits = 64
        else: raise ValueError(f'Unsupported dtype: {dt}')
    elif check and ten.any(x >= 1 << max_bits).item():
        raise ValueError(
            f'Expected values less than {1 << max_bits}, got {ten.max(x).item()} for dtype {dt}'
        )
    r = ten.zeros_like(x)
    zero = ten.array(0, dtype=dt)
    one = ten.array(1, dtype=dt)

    bits = 1 << (max_bits.bit_length() - 1)  # highest power of 2 <= max_bits
    if bits == max_bits: bits >>= 1
    # print(f'Initial bits: {bits} from {max_bits}')

    while True:
        t = ten.where(x >= (one << bits), ten.array(bits, dtype=dt), zero)
        r = r + t
        bits = bits >> 1
        if bits == 0: return r
        x = x >> t


def indent_str(x, indent: int | str = '') -> str:
    if not isinstance(indent, str): indent = ' ' * indent
    sep = '\n' + indent
    return sep.join(str(x).split('\n'))


def show_array(x: Array, prefix: str = None, indent: bool | int | str = True, out: Callable = print, **printoptions):
    if indent is True:
        indent = len(prefix) if prefix else ''
    out(prefix, end='')
    if printoptions:
        out('array(', end='')
        indent = indent + 6 if isinstance(indent, int) else indent + ' ' * 6
        with np.printoptions(**printoptions):
            out(indent_str(np.array(x), indent=indent), end='')
        out(')')
    else:
        out(indent_str(x, indent=indent))



__all__ = [
    'broadcast_shapes',
    'indent_str',
    'int_log2',
    'promote_types',
    'show_array',
]