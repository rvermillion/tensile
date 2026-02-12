#  Copyright (c) 2025. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Optional, Sequence, TypeAlias, Union
from ..shims.common import Shape
from .. import ten

Array: TypeAlias = ten.Array
DType: TypeAlias = ten.DType

Index: TypeAlias = Union[int, Array, slice, Ellipsis, None]
Indices: TypeAlias = Union[Index, tuple[Index, ...]]


class RegionIndex:

    __slots__ = ()

    index: Index

    @property
    def first(self) -> int:
        raise NotImplementedError()

    @property
    def last(self) -> int:
        raise NotImplementedError()

    @property
    def stop(self) -> int:
        return self.last + 1

    @property
    def count(self) -> int:
        raise NotImplementedError()


class IntIndex(RegionIndex):

    __slots__ = ('index', )

    index: int

    def __init__(self, index: int):
        self.index = index

    @property
    def first(self) -> int:
        return self.index

    @property
    def last(self) -> int:
        return self.index

    @property
    def stop(self) -> int:
        return self.index + 1

    @property
    def count(self) -> int:
        return 1


class ArrayIndex(RegionIndex):

    __slots__ = ('index', )

    index: Array

    def __init__(self, index: Array):
        self.index = index

    @property
    def first(self) -> int:
        return ten.min(self.index).item()

    @property
    def last(self) -> int:
        return ten.max(self.index).item()

    @property
    def count(self) -> int:
        return self.index.size


class RangeIndex(RegionIndex):

    __slots__ = ('start', 'stop', 'step')

    start: int
    stop: int
    step: int

    def __init__(self, start: int, stop: int, step: int = 1):
        if start < 0 or stop < 0:
            raise ValueError(f'Range indices must be non-negative, not {start}, {stop}')
        if step == 0:
            raise ValueError(f'Range step cannot be zero')
        if start > stop if step > 0 else stop > start:
            raise ValueError(f'Range indices must be in {"in" if step > 0 else "de"}creasing order, not {start}, {stop}')
        self.start = start
        self.stop = stop
        self.step = step

    @property
    def index(self) -> slice:
        return slice(self.start, self.stop, self.step)

    @property
    def first(self) -> int:
        return self.start

    @property
    def last(self) -> int:
        return self.stop - 1

    @property
    def count(self) -> int:
        return (self.stop - self.start) // self.step


RegionIndices: TypeAlias = tuple[RegionIndex, ...]

Structure: TypeAlias = tuple[Shape, DType]


class Region:

    __slots__ = ()

    @property
    def shape(self) -> Shape:
        return tuple(ind.count for ind in self.indices)

    @property
    def indices(self) -> RegionIndices:
        raise NotImplementedError()

    def get(self, array: Array) -> Array:
        return array[self.indices]

    def set(self, array: Array, values: Array) -> None:
        array[self.indices] = values

    def intersect(self, other: 'Region') -> 'Region':
        raise NotImplementedError()

    def union(self, other: 'Region') -> 'Region':
        raise NotImplementedError()

    def minus(self, other: 'Region') -> 'Region':
        raise NotImplementedError()


class TensorEvent:

    __slots__ = ('origin', )

    type: int
    origin: 'EventTensor'

    index: Optional[Array] = None
    data: Optional[Array] = None
    delta: Optional[Array] = None

    def __init__(self, origin: 'EventTensor'):
        self.origin = origin

    def __repr__(self):
        index = self.index
        delta = self.delta
        if index is None:
            if delta is None:
                return f'TensorEvent({self.data})'
            return f'TensorEvent(delta={self.delta})'
        if delta is None:
            return f'TensorEvent(index={self.index}, data={self.data})'
        return f'TensorEvent(index={self.index}, delta={self.delta})'

    Full: int = 0
    Indexed: int = 1


class FullTensorEvent(TensorEvent):

    __slots__ = ('data', )

    type = TensorEvent.Full

    data: Array

    def __init__(self, origin: 'EventTensor', data: Array):
        TensorEvent.__init__(self, origin)
        self.data = data


class IndexedTensorEvent(TensorEvent):

    __slots__ = ('index', 'data', 'delta')

    type = TensorEvent.Indexed

    index: Array
    data: Array
    delta: Array

    def __init__(self, origin: 'EventTensor', index: Array, data: Array = None, delta: Array = None):
        if data is None:
            if index.shape != delta.shape:
                raise ValueError(f'Index shape {index.shape} does not match delta shape {delta.shape}')
        else:
            if index.shape != data.shape:
                raise ValueError(f'Index shape {index.shape} does not match data shape {data.shape}')
        TensorEvent.__init__(self, origin)
        self.index = index
        self.data = data
        self.delta = delta


class TensorOp:

    __slots__ = ()

    name: str

    def structure(self, args: Sequence['EventTensor']) -> Structure:
        raise NotImplementedError()

    def evaluate(self, args: Sequence['EventTensor'], index: Array = None) -> Array:
        raise NotImplementedError()

    def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> TensorEvent:
        raise NotImplementedError()

    def print_evaluating(self, *args: 'EventTensor', index: Array = None) -> None:
        if index is None:
            print(f'Evaluating {self} with ({args})')
        else:
            print(f'Evaluating {self} with ({args}) for index {index}')

    def __repr__(self):
        return self.name


AxisChoice: TypeAlias = Union[None, int, Sequence[int]]


class ReduceOp(TensorOp):

    __slots__ = ('axis', 'keepdims')

    axis: tuple[int, ...]
    keepdims: bool

    def __init__(self, axis: AxisChoice = None, keepdims: bool = False):
        self.axis = tuple(axis) if axis is not None else ()
        self.keepdims = keepdims

    def structure(self, args: Sequence['EventTensor']) -> Structure:
        if len(args) == 1:
            return self._structure(*args)
        raise ValueError(f'Expected one arguments for {self}')

    def _reduce_shape(self, shape: Shape) -> Shape:
        if axes := self.axis:
            if self.keepdims:
                reduced = list(shape)
                for a in axes:
                    reduced[a] = 1
                return tuple(reduced)
            else:
                reduced: list[Optional[int]] = list(shape)
                for a in axes:
                    reduced[a] = None
                return tuple(a for a in reduced if a is not None)
        return 1,

    def _structure(self, a: 'EventTensor') -> Structure:
        return self._reduce_shape(a.shape), a.dtype

    def evaluate(self, args: Sequence['EventTensor'], index: Array = None) -> Array:
        if len(args) == 1:
            return self._evaluate(*args, index=index)
        raise ValueError(f'Expected two arguments for {self}')

    def _evaluate(self, a: 'EventTensor', index: Array = None) -> Array:
        raise NotImplementedError()


class SumOp(ReduceOp):

    __slots__ = ()

    name = 'sum'

    def _evaluate(self, a: 'EventTensor', index: Array = None) -> Array:
        self.print_evaluating(a, index=index)
        return ten.sum(a.get(), keepdims=True)

    def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
        print(f'Updating {tensor} with {self} from event {event} for arg {arg_index}')
        data = tensor.data
        delta = event.delta
        if data is None or delta is None:
            tensor.data = self._evaluate(*tensor.args)
        else:
            tensor.data += ten.sum(delta, keepdims=True)
        return None


class BinaryOp(TensorOp):

    __slots__ = ()

    def evaluate(self, args: Sequence['EventTensor'], index: Array = None) -> Array:
        if len(args) == 2:
            return self._evaluate(*args, index=index)
        raise ValueError(f'Expected two arguments for {self}')

    def _evaluate(self, a: 'EventTensor', b: 'EventTensor', index: Array = None) -> Array:
        raise NotImplementedError()

    def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
        tensor.data = None
        return None


class ElementWiseBinaryOp(BinaryOp):

    __slots__ = ()

    def structure(self, args: Sequence['EventTensor']) -> Structure:
        if len(args) == 2:
            return self._structure(*args)
        raise ValueError(f'Expected two arguments for {self}')

    def _structure(self, a: 'EventTensor', b: 'EventTensor') -> Structure:
        shape = a.shape
        if shape != b.shape:
            raise ValueError(f'Shape mismatch for {self}: {a.shape} != {b.shape}')
        # return shape, ten.promote_types(a.dtype, b.dtype)
        # shape = ten.broadcast_shapes(a.shape, b.shape)
        return shape, a.dtype  # ten.promote_types(a.dtype, b.dtype)

    def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
        tensor.data = None


class AddOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'add'

    def _evaluate(self, a: 'EventTensor', b: 'EventTensor', index: Array = None) -> Array:
        self.print_evaluating(a, b, index=index)
        return a.get(index) + b.get(index)

    def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
        print(f'Updating {tensor} with {self} from event {event} for arg {arg_index}')
        idx = event.index
        if idx is None:
            tensor.data = self._evaluate(*tensor.args)
        else:
            data = tensor.data
            if data is None:
                tensor.data = self._evaluate(*tensor.args)
            else:
                delta = event.delta
                if delta is None:
                    data[idx] = self._evaluate(*tensor.args, index=idx)
                else:
                    data[idx] += delta

        return event


class MulOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'mul'

    def _evaluate(self, a: 'EventTensor', b: 'EventTensor', index: Array = None) -> Array:
        self.print_evaluating(a, b, index=index)
        return a.get(index) * b.get(index)

    def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
        print(f'Updating {tensor} with {self} from event {event} for arg {arg_index}')
        idx = event.index
        if idx is None:
            tensor.data = self._evaluate(*tensor.args)
        else:
            data = tensor.data
            if data is None:
                tensor.data = self._evaluate(*tensor.args)
            else:
                delta = event.delta
                if delta is None:
                    data[idx] = self._evaluate(*tensor.args, index=idx)
                else:
                    a = 1 - arg_index
                    data[idx] += tensor.args[a].data[idx] * delta

        return None


class TensorOps:

    add = AddOp()
    mul = MulOp()

    sum = SumOp()


class EventTarget:

    __slots__ = ()

    def handle(self, event: TensorEvent, arg_index: int = -1) -> None:
        raise NotImplementedError()


# class EventDependent:
#
#     __slots__ = ('target', 'arg_index')
#
#     target: EventTarget
#     arg_index: int
#
#     def __init__(self, target: EventTarget, arg_index: int):
#         self.target = target
#         self.arg_index = arg_index
#

class TensorSlice:

    __slots__ = ()




class EventTensor(EventTarget):

    __slots__ = ('name', 'data', 'shape', 'dtype', 'op', 'args', 'dependents')

    name: Optional[str]
    data: Optional[Array]
    shape: Shape
    dtype: DType
    op: TensorOp
    args: tuple['EventTensor', ...]
    dependents: list[tuple[int, EventTarget]]

    def __init__(self, op: TensorOp = None, args: Sequence['EventTensor'] = (), data: Array = None, shape: Shape = None, dtype: DType = None, name: str = None):
        if data is None:
            if shape is None:
                if dtype is None:
                    shape, dtype = op.structure(args)
                else:
                    shape, _ = op.structure(args)
        else:
            if shape is None:
                shape = data.shape
            if dtype is None:
                dtype = data.dtype
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = data
        self.op = op
        self.args = tuple(args)
        for a, arg in enumerate(args):
            arg.add_dependent(self, a)
        self.dependents = []

    def add_dependent(self, dependent: 'EventTensor', arg_index: int):
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

    def handle(self, event: TensorEvent, arg_index: int = -1) -> None:
        self.debug(f'Handling {event} for argument {arg_index} of {self}')
        update = self.op.update(self, event, arg_index)
        if update is not None:
            for a, dep in self.dependents:
                dep.handle(update, a)

    def event(self, index: Array = None, data: Array = None, delta: Array = None) -> TensorEvent:
        if index is None:
            return FullTensorEvent(self, data)
        return IndexedTensorEvent(self, index, data=data, delta=delta)

    def sum(self) -> 'EventTensor':
        return EventTensor(TensorOps.sum, (self,))

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

    def __add__(self, other: 'EventTensor') -> 'EventTensor':
        if not isinstance(other, EventTensor):
            return NotImplemented
        return EventTensor(TensorOps.add, (self, other))

    def __radd__(self, other: 'EventTensor') -> 'EventTensor':
        if not isinstance(other, EventTensor):
            return NotImplemented
        return EventTensor(TensorOps.add, (other, self))

    def __mul__(self, other: 'EventTensor') -> 'EventTensor':
        if not isinstance(other, EventTensor):
            return NotImplemented
        return EventTensor(TensorOps.mul, (self, other))

    def __rmul__(self, other: 'EventTensor') -> 'EventTensor':
        if not isinstance(other, EventTensor):
            return NotImplemented
        return EventTensor(TensorOps.mul, (other, self))

    def __repr__(self):
        nm = f'{self.name}=' if self.name else ''
        if self.data is None:
            if self.op:
                deriv = f'{self.op}({self.args}), '
            else:
                deriv = ''
            return f'EventTensor({nm}{deriv}shape={self.shape}, dtype={self.dtype})'
        return f'EventTensor({nm}{self.data})'