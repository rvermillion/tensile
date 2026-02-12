#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, Optional, Sequence, TypeAlias, Union
from .. import ten

from .common import Array, Axes, Base, DType, Functional, Indices, Shape
from .op import TensorOp, TensorOps, AxisChoice, broadcast_shapes
from .region import Region


# noinspection PyShadowingBuiltins
def min(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.min(a, axis=axis, keepdims=keepdims), name=name)


# noinspection PyShadowingBuiltins
def max(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.max(a, axis=axis, keepdims=keepdims), name=name)


def mean(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.mean(a, axis=axis, keepdims=keepdims), name=name)


# noinspection PyShadowingBuiltins
def sum(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.sum(a, axis=axis, keepdims=keepdims), name=name)


def expand_dims(a: 'Tensor', axis: AxisChoice = None, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.expand_dims(a, axis=axis), name=name)


def transpose(a: 'Tensor', axes: Axes = None, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.transpose(a, axes=axes), name=name)


def swapaxes(a: 'Tensor', axis1: int, axis2: int, *, name: str = None) -> 'Tensor':
    if axis1 < 0: axis1 += a.ndim
    if axis2 < 0: axis2 += a.ndim
    if axis1 == axis2: return a
    axes = list(range(a.ndim))
    axes[axis1] = axis2
    axes[axis2] = axis1
    return DerivedTensor(op=TensorOps.transpose(a, axes=tuple(axes)), name=name)


def moveaxis(a: 'Tensor', source: int, destination: int, *, name: str = None) -> 'Tensor':
    if source < 0: source += a.ndim
    if destination < 0: destination += a.ndim
    if source == destination: return a
    axes = list(range(a.ndim))
    del axes[source]
    # if source < destination: destination -= 1
    axes.insert(destination, source)
    return DerivedTensor(op=TensorOps.transpose(a, axes=tuple(axes)), name=name)


def logsumexp(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.logsumexp(a, axis=axis, keepdims=keepdims), name=name)


def broadcast(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.broadcast(a, shape=shape), name=name)


def reshape(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.reshape(a, shape=shape), name=name)


def flatten(a: 'Tensor', start_index: int = 0, end_index: int = -1, name: str = None) -> 'Tensor':
    orig = a.shape
    if start_index < 0: start_index += len(orig)
    if end_index < 0: end_index += len(orig)
    shape = *orig[:start_index], -1, *orig[end_index+1:]
    return DerivedTensor(op=TensorOps.reshape(a, shape=shape), name=name)


def functional(a: 'Tensor', func: Union[str, Functional], name: str = None) -> 'Tensor':
    if isinstance(func, str):
        func = getattr(ten.functional, func)
    return DerivedTensor(op=TensorOps.functional(a, func=func), name=name)


def subscript(a: 'Tensor', indices: Indices, name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.subscript(a, indices=indices), name=name)


def greater(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.greater(a, b), name=name)


def greater_equal(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.greater_equal(a, b), name=name)


def add(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.add(a, b), name=name)


def sub(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.sub(a, b), name=name)


def mul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.mul(a, b), name=name)


def matmul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor(op=TensorOps.matmul(a, b), name=name)


def data(data: Array, shape: Shape = None, dtype: DType = None, name: str = None) -> 'Tensor':
    return DataTensor(data=data, shape=shape, dtype=dtype, name=name)


def arange(start: ten.Scalar, stop: ten.Scalar = None, step: ten.Scalar = None, dtype: DType = None) -> 'Tensor':
    if step is None:
        if stop is None:
            stop = start
            start = 0
        step = 1
    elif stop is None:
        stop = start
        start = 0
    return data(ten.arange(start, stop, step, dtype=dtype))


def array(a: Union[list[ten.Scalar], ten.Scalar], dtype: DType = None, name: str = None) -> 'Tensor':
    return data(ten.array(a, dtype=dtype), dtype=dtype, name=name)


def constant(value: ten.Scalar, dtype: DType = None, name: str = None) -> 'Tensor':
    return data(ten.array(value, dtype=dtype), dtype=dtype, name=name)


# noinspection PyShadowingBuiltins
def eval(*tensors: 'Tensor') -> None:
    for t in tensors: t.get()


Dependent: TypeAlias = tuple[int, 'Tensor']


class TensorEvent(Base):

    __slots__ = ('source', 'region', 'data')

    source: 'Tensor'
    region: Region
    data: Optional[Array]

    def __init__(self, source: 'Tensor', region: Region, data: Optional[Array]):
        self.source = source
        self.region = region
        self.data = data

    def _repr_item_dict(self) -> Optional[dict[str, Any]]:
        return {'source': self.source, 'region': self.region, 'data': self.data}

    @classmethod
    def create(cls, source: 'Tensor', region: Region, data: Array = None) -> 'TensorEvent':
        return TensorEvent(source, region, data)


class Tensor:

    __slots__ = ('name', 'data', 'shape', 'dtype', 'dependents')

    name: Optional[str]
    data: Optional[Array]
    shape: Shape
    dtype: DType
    dependents: list[Dependent]

    def __init__(self, data: Array = None, shape: Shape = None, dtype: DType = None, name: str = None):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.dependents = []

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def add_dependent(self, dependent: 'Tensor', arg_index: int):
        self.dependents.append((arg_index, dependent))

    def remove_dependent(self, dependent: 'Tensor', arg_index: int = None):
        if arg_index is None:
            self.dependents = [dep for dep in self.dependents if dep[1] is not dependent]
        else:
            self.dependents = [dep for dep in self.dependents if dep[1] is not dependent or dep[0] != arg_index]

    def get(self, index: Array = None) -> Array:
        raise NotImplementedError()

    def debug(self, *args, **kwargs) -> None:
        print(*args, **kwargs)

    def handle(self, event: TensorEvent, arg_index: int = -1) -> None:
        raise TypeError(f'Cannot handle event for {self} with argument index {arg_index}')

    # def event(self, index: Array = None, data: Array = None, delta: Array = None) -> TensorEvent:
    #     if index is None:
    #         return FullTensorEvent(self, data)
    #     return IndexedTensorEvent(self, index, data=data, delta=delta)

    min = min
    max = max
    mean = mean
    sum = sum

    def reshape(self, *shape: int, name: str = None) -> 'Tensor':
        if name is None: name = f'reshape({self.name}, {shape})'
        return reshape(self, shape, name=name)

    def broadcast(self, shape: Shape, name: str = None) -> 'Tensor':
        if name is None: name = f'broadcast({self.name}, {shape})'
        return self if shape == self.shape else broadcast(self, shape, name=name)

    def __getitem__(self, key):
        data = self.data
        if data is None:
            return subscript(self, key)
        return data[key]

    def __setitem__(self, key, value):
        raise ValueError(f'Cannot set value for {self}')

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

    def get(self, index: Array = None) -> Array:
        if index is None:
            return self.data
        return self.data[index]

    def __setitem__(self, key, value):
        region = Region.from_key(self.shape, key)
        self.data[key] = value
        update = TensorEvent.create(self, region, value)
        for a, dep in self.dependents:
            dep.handle(update, a)


class DerivedTensor(Tensor):

    __slots__ = ('op', )

    op: TensorOp

    def __init__(self, op: TensorOp, shape: Shape = None, dtype: DType = None, name: str = None):
        if op is None:
            raise ValueError('Must provide either op or data')
        if shape is None:
            shape = op.shape
        if dtype is None:
            dtype = op.dtype
        super().__init__(shape=shape, dtype=dtype, name=name)
        self.op = op
        if op is not None:
            for a, arg in enumerate(op.args):
                arg.add_dependent(self, a)

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

    def handle(self, event: TensorEvent, arg_index: int = -1) -> None:
        self.debug(f'Tensor {self.name}: Handling {event} for argument {arg_index} of {self}')
        # For now, we recalculate the entire tensor every time.
        self.data = None
        region = self.op.map_region(arg_index, event.region)
        update = TensorEvent.create(self, region)
        # update = self.op.update(self, event, arg_index)
        if update is not None:
            for a, dep in self.dependents:
                dep.handle(update, a)

    def __repr__(self):
        nm = f'{self.name}=' if self.name else ''
        if self.data is None:
            deriv = f'{self.op}({", ".join(str(arg) for arg in self.op.args)}), '
            return f'Tensor({nm}{deriv}shape={self.shape}, dtype={self.dtype})'
        return f'Tensor({nm}{self.data})'


def test():
    # a = reshape(arange(-2., 10.), (4, 3),  name='a')
    a = arange(-2., 10.).reshape(4, 3,  name='a')
    b = array([4., 5., 6.], name='b')
    # c = a + b
    # c.name = 'c'
    c = add(a, b, name='c')
    d = sum(c, axis=0, name='d')
    e = constant(10., name='e')
    f = d * e
    f.name = 'f'
    g = matmul(a, b, name='g')

    h = expand_dims(a, axis=(-5, -3, -1), name='h')
    i = flatten(h, start_index=1, end_index=3, name='i')
    # j = functional(a, ten.functional.relu, name='j')
    j = functional(a, 'relu', name='j')
    k = transpose(c, name='k')
    # k = swapaxes(h, 1, -1, name='k')
    # k = moveaxis(g, 1, 2, name='k')
    l = add(k, e, name='l')

    t = (a, b, c, d, e, f, g, h, i, j, k, l)
    for x in t:
        print(x)
    print('-' * 80)
    for x in t:
        print(x.name, '->', len(x.dependents),  x.dependents)
    print('-' * 80)
    for x in t:
        print(x.name, '=', x.get())

    print('-' * 80)
    b[1] = 100
    print('-' * 80)
    for x in t:
        print(x.name, '=', x.shape, '=', x.get())

    print('ops:', ', '.join(TensorOp.names.keys()))

    exit(0)

test()