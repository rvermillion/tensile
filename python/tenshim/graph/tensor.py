#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Optional, Sequence, TypeAlias, Union
from ..shims.common import Shape
from .. import ten

from .op import TensorOp, TensorOps, AxisChoice, broadcast_shapes

Array: TypeAlias = ten.Array
DType: TypeAlias = ten.DType

Index: TypeAlias = Union[int, Array, slice, Ellipsis, None]
Indices: TypeAlias = Union[Index, tuple[Index, ...]]


def sum(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.sum(axis=axis, keepdims=keepdims), args=(a,), name=name)


def broadcast(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.broadcast(shape=shape), args=(a,), name=name)


def reshape(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.reshape(shape=shape), args=(a,), name=name)


def slice(a: 'Tensor', indices: Indices, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.slice(indices=indices), args=(a,), name=name)


def elementwise(op: TensorOp, a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    if not isinstance(b, Tensor):
        return NotImplemented
    if a.shape != b.shape:
        shape = broadcast_shapes(a.shape, b.shape)
        if shape != a.shape:
            a = broadcast(a, shape)
        if shape != b.shape:
            b = broadcast(b, shape)
    return Tensor(op=op, args=(a, b), name=name)


def add(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return elementwise(TensorOps.add(), a, b, name)


def mul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return elementwise(TensorOps.mul(), a, b, name)


def matmul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return elementwise(TensorOps.matmul(), a, b, name)


def data(data: Array, shape: Shape = None, dtype: DType = None, name: str = None) -> 'Tensor':
    return Tensor(data=data, shape=shape, dtype=dtype, name=name)


Dependent: TypeAlias = tuple[int, 'Tensor']


class Tensor:

    __slots__ = ('name', 'data', 'shape', 'dtype', 'op', 'args', 'dependents')

    name: Optional[str]
    data: Optional[Array]
    shape: Shape
    dtype: DType
    op: TensorOp
    args: tuple['Tensor', ...]
    dependents: list[Dependent]

    def __init__(self, op: TensorOp = None, args: Sequence['Tensor'] = (), data: Array = None, shape: Shape = None, dtype: DType = None, name: str = None):
        if data is None:
            if shape is None:
                if dtype is None:
                    shape, dtype = op.structure(args)
                else:
                    shape, _ = op.structure(args)
        else:
            if shape is None:
                shape = data.shape
            elif shape != data.shape:
                raise ValueError(f'Shape mismatch: {shape} != {data.shape}')
            if dtype is None:
                dtype = data.dtype
            elif dtype != data.dtype:
                raise ValueError(f'DType mismatch: {dtype} != {data.dtype}')
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.op = op
        self.args = tuple(args)
        for a, arg in enumerate(args):
            arg.add_dependent(self, a)
        self.dependents = []

    def add_dependent(self, dependent: 'Tensor', arg_index: int):
        self.dependents.append((arg_index, dependent))

    def get(self, index: Array = None) -> Array:
        cache = self.data
        if index is None:
            if cache is None:
                cache = self.data = self.op.evaluate(self.args)
            return cache if index is None else cache[index]
        else:
            if cache is None:
                cache = self.data = self.op.evaluate(self.args)
            return cache[index]

    def debug(self, *args, **kwargs) -> None:
        print(*args, **kwargs)

    # def handle(self, event: TensorEvent, arg_index: int = -1) -> None:
    #     self.debug(f'Handling {event} for argument {arg_index} of {self}')
    #     update = self.op.update(self, event, arg_index)
    #     if update is not None:
    #         for a, dep in self.dependents:
    #             dep.handle(update, a)
    #
    # def event(self, index: Array = None, data: Array = None, delta: Array = None) -> TensorEvent:
    #     if index is None:
    #         return FullTensorEvent(self, data)
    #     return IndexedTensorEvent(self, index, data=data, delta=delta)

    sum = sum

    def __getitem__(self, key):
        data = self.data
        if data is None:
            return slice(self, key)
        return data[key]

    def __setitem__(self, key, value):
        if self.op:
            raise ValueError(f'Cannot set value for {self}')
        if isinstance(key, int):
            if not ten.is_array(value):
                value = ten.array([value], dtype=self.dtype)
            key = ten.array([key], dtype=ten.int64)
        delta = value - self.data[key]
        self.data[key] = value
        update = self.event(index=key, data=value)
        for a, dep in self.dependents:
            dep.handle(update, a)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return add(self, other)

    def __radd__(self, other: 'Tensor') -> 'Tensor':
        return add(other, self)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return mul(self, other)

    def __rmul__(self, other: 'Tensor') -> 'Tensor':
        return mul(other, self)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return matmul(self, other)

    def __rmatmul__(self, other: 'Tensor') -> 'Tensor':
        return matmul(other, self)

    def __repr__(self):
        nm = f'{self.name}=' if self.name else ''
        if self.data is None:
            if self.op:
                deriv = f'{self.op}({", ".join(str(arg) for arg in self.args)}), '
            else:
                deriv = ''
            return f'Tensor({nm}{deriv}shape={self.shape}, dtype={self.dtype})'
        return f'Tensor({nm}{self.data})'


def test():
    a = reshape(data(ten.arange(9)), (3, 3),  name='a')
    b = data(ten.array([4, 5, 6]), name='b')
    c = a + b
    c.name = 'c'
    d = sum(c, axis=-1, name='d')
    t = (a, b, c, d)
    for x in t:
        print(x)
    print('-' * 80)
    for x in t:
        print(x.name, '->', x.dependents)
    print('-' * 80)
    for x in t:
        print(x.name, '=', x.get())
    exit(0)

test()