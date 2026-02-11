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


# noinspection PyShadowingBuiltins
def min(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.min(a, axis=axis, keepdims=keepdims), name=name)


# noinspection PyShadowingBuiltins
def max(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.max(a, axis=axis, keepdims=keepdims), name=name)


def mean(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.mean(a, axis=axis, keepdims=keepdims), name=name)


# noinspection PyShadowingBuiltins
def sum(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.sum(a, axis=axis, keepdims=keepdims), name=name)


def broadcast(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.broadcast(a, shape=shape), name=name)


def reshape(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.reshape(a, shape=shape), name=name)


def subscript(a: 'Tensor', indices: Indices, name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.subscript(a, indices=indices), name=name)


def greater(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.greater(a, b), name=name)


def greater_equal(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.greater_equal(a, b), name=name)


def add(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.add(a, b), name=name)


def sub(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.sub(a, b), name=name)


def mul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.mul(a, b), name=name)


def matmul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return Tensor(op=TensorOps.matmul(a, b), name=name)


def data(data: Array, shape: Shape = None, dtype: DType = None, name: str = None) -> 'Tensor':
    return Tensor(data=data, shape=shape, dtype=dtype, name=name)


def constant(value: Union[bool, int, float], dtype: DType = None, name: str = None) -> 'Tensor':
    return data(ten.array(value, dtype=dtype), dtype=dtype, name=name)


Dependent: TypeAlias = tuple[int, 'Tensor']


full = slice(None, None, None)


def repr_index(index: Index) -> str:
    if isinstance(index, slice):
        start, stop, step = index.indices(10)
    return str(index) if index is not None else '...'


class Region:

    __slots__ = ('indices',)

    indices: tuple[Index, ...]

    def __init__(self, indices: tuple[Index, ...]):
        self.indices = indices

    def __repr__(self):
        return f'{self.__class__.__name__}({self.indices})'

    @property
    def ndims(self) -> int:
        return len(self.indices)

    @classmethod
    def from_key(cls, tensor: 'Tensor',key: Indices) -> 'Region':
        shape = tensor.shape
        if key is ...:
            indices = tuple(slice(0, size, 1) for size in shape)
        elif isinstance(key, int):
            if key < 0 and shape:
                key = shape[0] + key
            indices = (key,)
        elif isinstance(key, slice):
            indices = (key,)
        elif isinstance(key, tuple):
            indices = key
        else:
            raise ValueError(f'Invalid key: {key} for shape {shape}')
        if len(indices) < len(shape):
            indices += tuple(slice(0, size, 1) for size in shape[len(indices):])
        return Region(indices)


class TensorEvent:

    __slots__ = ('source', 'region', 'data')

    source: 'Tensor'
    region: Region
    data: Optional[Array]

    def __init__(self, source: 'Tensor', region: Region, data: Optional[Array]):
        self.source = source
        self.region = region
        self.data = data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.source}, {self.region}, {self.data})'

    @classmethod
    def create(cls, source: 'Tensor', region: Region, data: Array = None) -> 'TensorEvent':
        return TensorEvent(source, region, data)


class Tensor:

    __slots__ = ('name', 'data', 'shape', 'dtype', 'op', 'dependents')

    name: Optional[str]
    data: Optional[Array]
    shape: Shape
    dtype: DType
    op: TensorOp
    dependents: list[Dependent]

    def __init__(self, op: TensorOp = None, data: Array = None, shape: Shape = None, dtype: DType = None, name: str = None):
        if data is None:
            if op is None:
                raise ValueError('Must provide either op or data')
            if shape is None:
                shape = op.shape
            if dtype is None:
                dtype = op.dtype
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
        if op is not None:
            for a, arg in enumerate(op.args):
                arg.add_dependent(self, a)
        self.dependents = []

    def add_dependent(self, dependent: 'Tensor', arg_index: int):
        self.dependents.append((arg_index, dependent))

    def remove_dependent(self, dependent: 'Tensor', arg_index: int = None):
        if arg_index is None:
            self.dependents = [dep for dep in self.dependents if dep[1] is not dependent]
        else:
            self.dependents = [dep for dep in self.dependents if dep[1] is not dependent or dep[0] != arg_index]

    def get(self, index: Array = None) -> Array:
        cache = self.data
        if index is None:
            if cache is None:
                cache = self.data = self.op.evaluate()
            return cache if index is None else cache[index]
        else:
            if cache is None:
                cache = self.data = self.op.evaluate(index=index)
            return cache[index]

    def debug(self, *args, **kwargs) -> None:
        print(*args, **kwargs)

    def handle(self, event: TensorEvent, arg_index: int = -1) -> None:
        self.debug(f'Handling {event} for argument {arg_index} of {self}')
        # For now, we recalculate the entire tensor every time.
        self.data = None
        update = TensorEvent.create(self, Region.from_key(self, ...))
        # update = self.op.update(self, event, arg_index)
        if update is not None:
            for a, dep in self.dependents:
                dep.handle(update, a)

    # def event(self, index: Array = None, data: Array = None, delta: Array = None) -> TensorEvent:
    #     if index is None:
    #         return FullTensorEvent(self, data)
    #     return IndexedTensorEvent(self, index, data=data, delta=delta)

    sum = sum

    def broadcast(self, shape: Shape) -> 'Tensor':
        return self if shape == self.shape else broadcast(self, shape)

    def __getitem__(self, key):
        data = self.data
        if data is None:
            return slice(self, key)
        return data[key]

    def __setitem__(self, key, value):
        if self.op:
            raise ValueError(f'Cannot set value for {self}')
        region = Region.from_key(self, key)
        # if isinstance(key, int):
        #     if not ten.is_array(value):
        #         value = ten.array([value], dtype=self.dtype)
        #     key = ten.array([key], dtype=ten.int64)
        # delta = value - self.data[key]
        self.data[key] = value
        update = TensorEvent.create(self, region, value)
        for a, dep in self.dependents:
            dep.handle(update, a)

    def __gt__(self, other: 'Tensor') -> 'Tensor':
        return greater(self, other)

    def __ge__(self, other: 'Tensor') -> 'Tensor':
        return greater_equal(self, other)

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
                deriv = f'{self.op}({", ".join(str(arg) for arg in self.op.args)}), '
            else:
                deriv = ''
            return f'Tensor({nm}{deriv}shape={self.shape}, dtype={self.dtype})'
        return f'Tensor({nm}{self.data})'


class DataTensor(Tensor):

    __slots__ = ()

    data: Array

    def __init__(self, data: Array, shape: Shape = None, dtype: DType = None, name: str = None):
        if shape is None:
            shape = data.shape
        elif shape != data.shape:
            raise ValueError(f'Shape mismatch: {shape} != {data.shape}')
        if dtype is None:
            dtype = data.dtype
        elif dtype != data.dtype:
            raise ValueError(f'DType mismatch: {dtype} != {data.dtype}')
        super().__init__(data=data, shape=shape, dtype=dtype, name=name)

    def __repr__(self):
        nm = f'{self.name}=' if self.name else ''
        return f'Tensor({nm}{self.data})'


class DerivedTensor(Tensor):

    __slots__ = ()

    op: TensorOp

    def __init__(self, op: TensorOp, shape: Shape = None, dtype: DType = None, name: str = None):
        if op is None:
            raise ValueError('Must provide either op or data')
        if shape is None:
            shape = op.shape
        if dtype is None:
            dtype = op.dtype
        super().__init__(op=op, shape=shape, dtype=dtype, name=name)

    def get(self, index: Array = None) -> Array:
        cache = self.data
        if index is None:
            if cache is None:
                cache = self.data = self.op.evaluate()
            return cache if index is None else cache[index]
        else:
            if cache is None:
                cache = self.data = self.op.evaluate(index=index)
            return cache[index]

    def __repr__(self):
        nm = f'{self.name}=' if self.name else ''
        if self.data is None:
            deriv = f'{self.op}({", ".join(str(arg) for arg in self.op.args)}), '
            return f'Tensor({nm}{deriv}shape={self.shape}, dtype={self.dtype})'
        return f'Tensor({nm}{self.data})'


def test():
    a = reshape(data(ten.arange(9.)), (3, 3),  name='a')
    b = data(ten.array([4, 5, 6]), name='b')
    c = a + b
    c.name = 'c'
    d = sum(c, axis=-1, name='d')
    e = constant(10, name='e')
    f = d * e
    f.name = 'f'
    g = matmul(a, b, name='g')
    t = (a, b, c, d, e, f, g)
    for x in t:
        print(x)
    print('-' * 80)
    for x in t:
        print(x.name, '->', len(x.dependents),  x.dependents)
    print('-' * 80)
    for x in t:
        print(x.name, '=', x.get())

    b[1] = 100
    print('-' * 80)
    for x in t:
        print(x.name, '=', x.get())

    exit(0)

test()