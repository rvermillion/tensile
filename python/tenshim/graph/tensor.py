#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Optional, Union
from .. import ten

from .common import Array, Axes, AxisChoice, Base, DType, Functional, Indices, Shape
from .op import TensorOp, TensorOps
from .region import Region, Regions
from .event import TensorEvent
from .util import int_log2


# noinspection PyShadowingBuiltins
def min(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.min(a, axis=axis, keepdims=keepdims), name=name)


# noinspection PyShadowingBuiltins
def max(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.max(a, axis=axis, keepdims=keepdims), name=name)


def mean(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.mean(a, axis=axis, keepdims=keepdims), name=name)


# noinspection PyShadowingBuiltins
def sum(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.sum(a, axis=axis, keepdims=keepdims), name=name)


def expand_dims(a: 'Tensor', axis: AxisChoice, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.expand_dims(a, axis=axis), name=name)


def squeeze(a: 'Tensor', axis: AxisChoice = None, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.squeeze(a, axis=axis), name=name)


def unsqueeze(a: 'Tensor', axis: AxisChoice, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.expand_dims(a, axis=axis), name=name)


def transpose(a: 'Tensor', axes: Axes = None, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.transpose(a, axes=axes), name=name)


def swapaxes(a: 'Tensor', axis1: int, axis2: int, *, name: str = None) -> 'Tensor':
    if axis1 < 0: axis1 += a.ndim
    if axis2 < 0: axis2 += a.ndim
    if axis1 == axis2: return a
    axes = list(range(a.ndim))
    axes[axis1] = axis2
    axes[axis2] = axis1
    return DerivedTensor.new(op=TensorOps.transpose(a, axes=tuple(axes)), name=name)


def moveaxis(a: 'Tensor', source: int, destination: int, *, name: str = None) -> 'Tensor':
    if source < 0: source += a.ndim
    if destination < 0: destination += a.ndim
    if source == destination: return a
    axes = list(range(a.ndim))
    del axes[source]
    # if source < destination: destination -= 1
    axes.insert(destination, source)
    return DerivedTensor.new(op=TensorOps.transpose(a, axes=tuple(axes)), name=name)


def logsumexp(a: 'Tensor', axis: AxisChoice = None, keepdims: bool = False, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.logsumexp(a, axis=axis, keepdims=keepdims), name=name)


def broadcast(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.broadcast(a, shape=shape), name=name)


def reshape(a: 'Tensor', shape: Shape, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.reshape(a, shape=shape), name=name)


def flatten(a: 'Tensor', start_index: int = 0, end_index: int = -1, name: str = None) -> 'Tensor':
    orig = a.shape
    if start_index < 0: start_index += len(orig)
    if end_index < 0: end_index += len(orig)
    shape = *orig[:start_index], -1, *orig[end_index+1:]
    return DerivedTensor.new(op=TensorOps.reshape(a, shape=shape), name=name)


def functional(a: 'Tensor', func: Union[str, Functional], name: str = None) -> 'Tensor':
    if isinstance(func, str):
        func = getattr(ten.functional, func)
    return DerivedTensor.new(op=TensorOps.functional(a, func=func), name=name)


def subscript(a: 'Tensor', indices: Indices, name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.subscript(a, indices=indices), name=name)


def greater(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.greater(a, b), name=name)


def greater_equal(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.greater_equal(a, b), name=name)


def add(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.add(a, b), name=name)


def sub(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.sub(a, b), name=name)


def mul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.mul(a, b), name=name)


def matmul(a: 'Tensor', b: 'Tensor', name: str = None) -> 'Tensor':
    return DerivedTensor.new(op=TensorOps.matmul(a, b), name=name)


def from_array(a: Array, shape: Shape = None, dtype: DType = None, name: str = None) -> 'Tensor':
    return DataTensor.new(data=a, shape=shape, dtype=dtype, name=name)


def arange(start: ten.Scalar, stop: ten.Scalar = None, step: ten.Scalar = None, dtype: DType = None, name: str = None) -> 'Tensor':
    if step is None:
        if stop is None:
            stop = start
            start = 0
        step = 1
    elif stop is None:
        stop = start
        start = 0
    return from_array(ten.arange(start, stop, step, dtype=dtype), name=name)


def array(a: Union[list[ten.Scalar], ten.Scalar], dtype: DType = None, name: str = None) -> 'Tensor':
    return from_array(ten.array(a, dtype=dtype), dtype=dtype, name=name)


def constant(value: ten.Scalar, dtype: DType = None, name: str = None) -> 'Tensor':
    return from_array(ten.array(value, dtype=dtype), dtype=dtype, name=name)


tensors_by_name: dict[str, 'Tensor'] = {}


# noinspection PyShadowingBuiltins
def eval(*tensors: 'Tensor') -> None:
    for t in tensors: t.get()


class Dependent(Base):
    __slots__ = ('arg_index', 'tensor')

    arg_index: int
    tensor: 'Tensor'

    def __init__(self, arg_index: int, tensor: 'Tensor'):
        if arg_index < 0: raise ValueError(f'Argument index must be non-negative: {arg_index}')
        self.arg_index = arg_index
        self.tensor = tensor

    def update(self, update: TensorEvent):
        self.tensor.handle(update, self.arg_index)

    def _repr_arg(self, short: bool = False) -> str:
        return f'{self.arg_index}={self.tensor.display(short=short)}'


class Tensor(Base):

    __slots__ = ('name', 'data', 'shape', 'dtype', 'dirty', 'dependents')

    name: Optional[str]
    data: Optional[Array]
    dirty: Optional[Region]
    shape: Shape
    dtype: DType
    dependents: list[Dependent]

    def __init__(self, data: Array = None, shape: Shape = None, dtype: DType = None, name: str = None):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.dirty = Regions.full(shape) if data is None else None
        self.dependents = []
        if name is not None:
            if name in tensors_by_name:
                self.warn('overwriting tensor with name ' + name + '')
            tensors_by_name[name] = self

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def add_dependent(self, dependent: 'Tensor', arg_index: int):
        self.dependents.append(Dependent(arg_index, dependent))

    def remove_dependent(self, dependent: 'Tensor', arg_index: int = None):
        if self.dependents:
            if arg_index is None:
                self.dependents = [dep for dep in self.dependents if dep.tensor is not dependent]
            else:
                self.dependents = [dep for dep in self.dependents if dep.tensor is not dependent or dep.arg_index != arg_index]

    def _validate(self) -> None:
        super()._validate()

    def get(self, index: Array = None) -> Array:
        raise NotImplementedError()

    # @staticmethod
    # def debug(*args, **kwargs) -> None:
    #     print(*args, **kwargs)

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

    def display(self, short: bool = False) -> str:
        return (short and self.name) or self._repr(short=short)

    def _repr_arg(self, short: bool = False) -> str:
        nm = f'{self.name}=' if self.name else ''
        if short: return f'{nm}{self.shape}'
        return f'{nm}{self.data}, dtype={self.dtype}'


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
        for dep in self.dependents:
            dep.update(update)


class DerivedTensor(Tensor):

    __slots__ = ('op', 'prev')

    op: TensorOp
    prev: Optional[Array]

    def __init__(self, op: TensorOp, shape: Shape = None, dtype: DType = None, name: str = None):
        if op is None:
            raise ValueError('Must provide either op or data')
        if shape is None:
            shape = op.shape
        if dtype is None:
            dtype = op.dtype
        super().__init__(shape=shape, dtype=dtype, name=name)
        self.op = op
        self.prev = None
        if op is not None:
            for a, arg in enumerate(op.args):
                arg.add_dependent(self, a)

    def get(self, region: Region = None) -> Array:
        cache = self.data
        if region is None:
            if cache is None:
                cache = self.data = self.op.evaluate()
            return cache # if index is None else cache[index]
        else:
            if cache is None:
                cache = self.data = self.op.evaluate()
            return cache   #[index]

    def handle(self, event: TensorEvent, arg_index: int = -1) -> None:
        self.prev = self.data
        self.data = None
        # self.debug(f'Tensor {self.name}: Handling {event.region} for argument {arg_index} of {self}')
        # For now, we recalculate the entire tensor every time.
        region = self.op.map_region(arg_index, event.region)
        self.debug(f'Tensor {self.name}: Operation {self.op.name} mapped {event.region:s} to {region:s} for argument {arg_index}')
        update = TensorEvent.create(self, region)
        if update is not None:
            for dep in self.dependents:
                dep.update(update)

    def _repr_arg(self, short: bool = False) -> str:
        if short: return f'{self.name}={self.shape}'
        nm = f'{self.name}=' if self.name else ''
        if self.data is None:
            return f'{nm}{self.op.derivation(short=short)}, shape={self.shape}, dtype={self.dtype}'
        return f'{nm}{self.data}'

    # def __repr__(self):
    #     nm = f'{self.name}=' if self.name else ''
    #     if self.data is None:
    #         deriv = f'{self.op}({", ".join(str(arg) for arg in self.op.args)}), '
    #         return f'Tensor({nm}{deriv}shape={self.shape}, dtype={self.dtype})'
    #     return f'Tensor({nm}{self.data})'


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
    # at = transpose(a, name='at')

    h = expand_dims(a, axis=(-5, -3, -1), name='h')
    i = flatten(h, start_index=1, end_index=3, name='i')
    # j = functional(a, ten.functional.relu, name='j')
    j = functional(a, 'relu', name='j')
    k = transpose(c, name='k')
    # k = swapaxes(h, 1, -1, name='k')
    # k = moveaxis(g, 1, 2, name='k')
    l = add(k, e, name='l')

    m = from_array(ten.arange(0., 9.).reshape(3, 3), name='m')
    n = from_array(ten.arange(0., -9., -1.).reshape(3, 3), name='n')
    o = matmul(m, n, name='o')
    p = expand_dims(o, axis=(0, -1), name='p')
    q = add(p, e, name='q')

    t = (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
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
    m[1, 1] = 1000
    print('-' * 80)
    for x in t:
        print(x.name, '=', x.shape, '=', x.get())

    print('ops:', ', '.join(TensorOp.names.keys()))

    exit(0)

def test_tree():

    def indent(x, ind: str = None, n=4):
        if ind is None: ind = ' ' * n
        sep = '\n' + ind
        return sep.join(str(x).split('\n'))

    def show(name: str, _a: Array):
        msg = f'{name}: {_a.shape} '
        print(f'\n{msg}{indent(_a, n=len(msg))}')

    depth = 4
    size = 1 << depth
    # offset = 0
    shift = 1
    node_count = size - shift
    leaf_count = size >> 1
    parent_count = node_count - leaf_count
    top_k = depth << 1

    print('depth:', depth, 'size:', size, 'node count:', node_count,
          'leaf count:', leaf_count, 'parent count:', parent_count,
          'top k:', top_k)

    nodes = ten.arange(shift, size, dtype=ten.int32)
    show('nodes', nodes)

    leaf_nodes = nodes[-leaf_count:]
    show('leaf nodes', leaf_nodes)

    paths = leaf_nodes.reshape(leaf_count, 1) >> ten.arange(depth, -1, -1)
    # if shift != 0:
    #     paths = paths - shift
    show('paths', paths)

    # parents = paths >> 1
    parents = paths[:, :-1]
    show('parents', parents)

    if ten.any(parents > parent_count).item():
        raise ValueError(
            f'Parent indices {parents} are out of bounds for a tree of depth {depth} and size {size}.'
        )

    children = paths[:, 1:]
    show('children', children)

    left = ten.random.uniform(shape=(parent_count,))
    right = 1. - left
    # if offset > 0:
    #     left[:offset] = 1.
    #     right[:offset] = 1.
    show('left', left)
    show('right', right)

    p = parents if shift == 0 else parents - shift
    splits = ten.where(children % 2, left[p], right[p])
    show('splits', splits)

    path_weights = ten.cumprod(splits, axis=-1)
    show('path weights', path_weights)

    # leaf_weights = weights[:, -1]
    # p('leaf weights:', leaf_weights.shape, leaf_weights)

    # layers = ten.array(1) << ten.arange(depth)
    # show('layers', layers)
    #
    # rev_layers = layers[-2::-1]
    # show('reversed layers', rev_layers)
    #
    # weight_vector = weights[]

    show('children', children)

    child_nodes = nodes[1:]
    show('child nodes', child_nodes)

    # log2 = ten.log(ten.array(2))
    # i = (ten.log(child_nodes)//log2).astype(ten.int32)

    i = int_log2(child_nodes, max_bits=depth)
    col = i - 1
    show('col', col)

    j = (depth-1) - i
    # show('j', j)

    k = child_nodes - (ten.array(1) << i)
    # show('k', k)

    row = k << j
    show('row', row)

    pluck = children[row, col]
    show('pluck', pluck)
    if not ten.all(pluck[1:] == pluck[:-1] + 1).item():
        raise ValueError('Expected pluck to be monotonic')

    weights = ten.ones((node_count, ), dtype=ten.float32)
    weights[1:] = path_weights[row, col]
    show('weights', weights)
    if top_k > 0:
        sorted_ind = ten.argsort(weights)
        show('sorted indices', sorted_ind)
        top_ind = sorted_ind[-top_k:]
        show('top indices', top_ind)
        weights = weights[top_ind]
        show('top weights', weights)
    total_weight = ten.sum(weights, axis=-1, keepdims=True) + 1.
    show('total weight', total_weight)
    weights = weights / total_weight
    show('normalized weights', weights)

    root_weight = 1. / total_weight

    show('new total weight', ten.sum(weights) + root_weight)
    value_dim = 1

    # values = ten.random.normal(shape=(node_count,))
    step = 100
    values = ten.arange(100, 100+(value_dim * node_count * step), step, dtype=ten.float32).reshape(node_count, value_dim)

    show('values', values)

    if top_k > 0:
        top_values = values[..., top_ind+1, :]
    else:
        top_values = values[..., 1:, :]
    # if offset > 0:
    #     values[:offset] = 0.
    show('top values', top_values)

    weighted_values = top_values[..., :, :] * weights[..., :, None]
    show('weighted values', weighted_values)

    root_value = values[..., 0, :]
    show('root value', root_value)

    summed = root_value/total_weight + ten.sum(weighted_values)
    show('summed', summed)

    exit(0)

test_tree()
test()