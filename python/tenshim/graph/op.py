#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, Optional, Self, Sequence, TypeAlias, Union, TYPE_CHECKING
from ..shims.common import Shape
from .. import ten

if TYPE_CHECKING:
    import tenshim.graph.tensor

TensorType: TypeAlias = 'tenshim.graph.tensor.Tensor'

Array: TypeAlias = ten.Array
DType: TypeAlias = ten.DType

Index: TypeAlias = Union[int, Array, slice, Ellipsis, None]
Indices: TypeAlias = Union[Index, tuple[Index, ...]]


Structure: TypeAlias = tuple[Shape, DType]


def promote_types(a: DType, b: DType) -> DType:
    # For now we ignore
    return a


def broadcast_shapes(a: Shape, b: Shape) -> Shape:
    if a == b:
        return a
    m = -min(len(a), len(b))
    i = -1
    shape = []
    while (i >= m):
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
    n = tuple(reversed(shape))
    print(f'Broadcasting {a} and {b} and got: {n}')
    return n


# class TensorDerivation:
#
#     __slots__ = ('op', 'args')
#
#     op: 'TensorOp'
#     args: tuple[TensorType, ...]
#
#     def __init__(self, op: 'TensorOp', args: Sequence[TensorType]):
#         self.op = op
#         self.args = tuple(args)


class TensorOp:

    __slots__ = ()

    name: str
    arity: int

    def gen_name(self, args: Sequence[TensorType]) -> str:
        name = self.name + '(' + ', '.join(arg.name for arg in args)
        if opts := self.options():
            return name + ',' + ', '.join(f"{k}={v!r}" for k, v in opts.items()) + ')'
        return name + ')'

    def options(self) -> dict[str, Any]:
        opts = {}
        self._options(opts)
        return opts

    def _options(self, opts: dict[str, Any]) -> None:
        pass

    def structure(self, args: Sequence[TensorType]) -> Structure:
        raise NotImplementedError()

    def evaluate(self, args: Sequence[TensorType], index: Array = None) -> Array:
        raise NotImplementedError()

    # def derive(self, args: Sequence[TensorType]) -> TensorDerivation:
    #     return self.Derivation(self, args)
    #
    # Derivation = TensorDerivation

    @classmethod
    def compile(cls, *args: TensorType, **kwargs) -> Self:
        if cls.arity != len(args):
            raise ValueError(f'Expected {cls.arity} arguments for {cls}, got {len(args)}')
        return cls._compile(*args, **kwargs)

    @classmethod
    def _compile(cls, *args: TensorType, **kwargs) -> Self:
        raise NotImplementedError()

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> TensorEvent:
    #     raise NotImplementedError()

    def print_evaluating(self, *args: Array, index: Array = None) -> None:
        disp = ', '.join(str(arg) for arg in args)
        if index is None:
            print(f'Evaluating {self} with ({disp})')
        else:
            print(f'Evaluating {self} with ({disp}) for index {index}')

    def __repr__(self):
        if opts := self.options():
            return self.name + '[' + ', '.join(f'{k}={v!r}' for k, v in opts.items()) + ']'
        return self.name


AxisChoice: TypeAlias = Union[None, int, Sequence[int]]


class UnaryOp(TensorOp):

    __slots__ = ()

    arity = 1


    def structure(self, args: Sequence[TensorType]) -> Structure:
        if len(args) == 1:
            return self._structure(*args)
        raise ValueError(f'Expected one arguments for {self}')

    def _structure(self, a: TensorType) -> Structure:
        raise NotImplementedError()

    def evaluate(self, args: Sequence[TensorType], index: Array = None) -> Array:
        if len(args) == 1:
            a = args[0].get()
            self.print_evaluating(a, index=index)
            return self._evaluate(a, index=index)
        raise ValueError(f'Expected one arguments for {self}')

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        raise NotImplementedError()


def can_broadcast(from_shape: Shape, to_shape: Shape) -> bool:
    return True


class BroadcastOp(UnaryOp):

    __slots__ = ('shape', )

    shape: Shape

    name = 'broadcast'

    def __init__(self, shape: Shape):
        self.shape = shape

    def _options(self, opts: dict[str, Any]) -> None:
        opts['shape'] = self.shape

    def _structure(self, a: TensorType) -> Structure:
        shape = self.shape
        if can_broadcast(a.shape, shape):
            return shape, a.dtype
        raise ValueError(f'Cannot broadcast: {a.shape} -> {shape}')

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.broadcast_to(a, self.shape)


def can_reshape(from_shape: Shape, to_shape: Shape) -> bool:
    return True


class ReshapeOp(UnaryOp):

    __slots__ = ('shape', )

    shape: Shape

    name = 'reshape'

    def __init__(self, shape: Shape):
        self.shape = shape

    def _options(self, opts: dict[str, Any]) -> None:
        opts['shape'] = self.shape

    def _structure(self, a: TensorType) -> Structure:
        shape = self.shape
        if can_reshape(a.shape, shape):
            return shape, a.dtype
        raise ValueError(f'Cannot reshape: {a.shape} -> {shape}')

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.reshape(a, self.shape)


class SliceOp(UnaryOp):

    __slots__ = ('indices', )

    indices: Indices

    def __init__(self, indices: Indices):
        self.indices = indices

    def _options(self, opts: dict[str, Any]) -> None:
        opts['indices'] = self.indices

    def _structure(self, a: TensorType) -> Structure:
        raise NotImplementedError()
        # return self.shape, a.dtype

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return a[self.indices]


class ReduceOp(UnaryOp):

    __slots__ = ('axis', 'keepdims')

    axis: Optional[tuple[int, ...]]
    keepdims: bool

    def __init__(self, axis: AxisChoice = None, keepdims: bool = False):
        self.axis = (axis, ) if isinstance(axis, int) else tuple(axis) if axis else None
        self.keepdims = keepdims

    def _options(self, opts: dict[str, Any]) -> None:
        if self.axis is not None:
            opts['axis'] = self.axis
        if self.keepdims:
            opts['keepdims'] = self.keepdims

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

    def _structure(self, a: TensorType) -> Structure:
        return self._reduce_shape(a.shape), a.dtype

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        raise NotImplementedError()


class MinOp(ReduceOp):

    __slots__ = ()

    name = 'min'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.min(a, axis=self.axis, keepdims=self.keepdims)


class MaxOp(ReduceOp):

    __slots__ = ()

    name = 'max'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.max(a, axis=self.axis, keepdims=self.keepdims)


class MeanOp(ReduceOp):

    __slots__ = ()

    name = 'mean'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.mean(a, axis=self.axis, keepdims=self.keepdims)


class SumOp(ReduceOp):

    __slots__ = ()

    name = 'sum'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.sum(a, axis=self.axis, keepdims=self.keepdims)

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     print(f'Updating {tensor} with {self} from event {event} for arg {arg_index}')
    #     data = tensor.data
    #     delta = event.delta
    #     if data is None or delta is None:
    #         tensor.data = self._evaluate(*tensor.args)
    #     else:
    #         tensor.data += ten.sum(delta, keepdims=True)
    #     return None


class BinaryOp(TensorOp):

    __slots__ = ()

    arity = 2

    def structure(self, args: Sequence[TensorType]) -> Structure:
        if len(args) == 2:
            return self._structure(*args)
        raise ValueError(f'Expected two arguments for {self}')

    def _structure(self, a: TensorType, b: TensorType) -> Structure:
        raise NotImplementedError()

    def evaluate(self, args: Sequence[TensorType], index: Array = None) -> Array:
        if len(args) == 2:
            a = args[0].get()
            b = args[1].get()
            self.print_evaluating(a, b, index=index)
            return self._evaluate(a, b, index=index)
        raise ValueError(f'Expected two arguments for {self}')

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        raise NotImplementedError()

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     tensor.data = None
    #     return None


class MatMulOp(BinaryOp):

    __slots__ = ()

    def _structure(self, a: TensorType, b: TensorType) -> Structure:
        ashape = a.shape
        if ashape[-1] != b.shape[-2]:
            raise ValueError(f'Shape mismatch for {self}: {a.shape} != {b.shape}')
        # return ashape, ten.promote_types(a.dtype, b.dtype)
        # ashape = ten.broadcast_shapes(a.ashape, b.ashape)
        shape = ashape[:-1] + (b.shape[-1],)
        return shape, promote_types(a.dtype, b.dtype)  # ten.promote_types(a.dtype, b.dtype)

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        if index is None:
            return a @ b
        return a[index] @ b[index]


class ElementWiseBinaryOp(BinaryOp):

    __slots__ = ()

    def _structure(self, a: TensorType, b: TensorType) -> Structure:
        shape = a.shape
        if shape != b.shape:
            shape = broadcast_shapes(shape, b.shape)
            # raise ValueError(f'Shape mismatch for {self}: {a.shape} != {b.shape}')
        # return shape, ten.promote_types(a.dtype, b.dtype)
        # shape = ten.broadcast_shapes(a.shape, b.shape)
        return shape, a.dtype  # ten.promote_types(a.dtype, b.dtype)

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     tensor.data = None


class AddOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'add'

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        if index is None:
            return a + b
        return a[index] + b[index]

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     print(f'Updating {tensor} with {self} from event {event} for arg {arg_index}')
    #     idx = event.index
    #     if idx is None:
    #         tensor.data = self._evaluate(*tensor.args)
    #     else:
    #         data = tensor.data
    #         if data is None:
    #             tensor.data = self._evaluate(*tensor.args)
    #         else:
    #             delta = event.delta
    #             if delta is None:
    #                 data[idx] = self._evaluate(*tensor.args, index=idx)
    #             else:
    #                 data[idx] += delta
    #
    #     return event


class MulOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'mul'

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        if index is None:
            return a * b
        return a[index] * b[index]

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     print(f'Updating {tensor} with {self} from event {event} for arg {arg_index}')
    #     idx = event.index
    #     if idx is None:
    #         tensor.data = self._evaluate(*tensor.args)
    #     else:
    #         data = tensor.data
    #         if data is None:
    #             tensor.data = self._evaluate(*tensor.args)
    #         else:
    #             delta = event.delta
    #             if delta is None:
    #                 data[idx] = self._evaluate(*tensor.args, index=idx)
    #             else:
    #                 a = 1 - arg_index
    #                 data[idx] += tensor.args[a].data[idx] * delta
    #
    #     return None


add = AddOp()
mul = MulOp()
matmul = MatMulOp()
default_sum = SumOp()
default_min = MinOp()
default_max = MaxOp()
default_mean = MeanOp()


class TensorOps:

    @staticmethod
    def add() -> AddOp:
        return add

    @staticmethod
    def mul() -> MulOp:
        return mul

    @staticmethod
    def matmul() -> MatMulOp:
        return matmul

    @staticmethod
    def broadcast(*, shape: Shape) -> BroadcastOp:
        return BroadcastOp(shape)

    @staticmethod
    def reshape(*, shape: Shape) -> ReshapeOp:
        return ReshapeOp(shape)

    @staticmethod
    def slice(*, indices: Indices) -> SliceOp:
        return SliceOp(indices)

    @staticmethod
    def min(*, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        if axis is None and not keepdims:
            return default_min
        return MinOp(axis, keepdims)

    @staticmethod
    def max(*, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        if axis is None and not keepdims:
            return default_max
        return MaxOp(axis, keepdims)

    @staticmethod
    def mean(*, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        if axis is None and not keepdims:
            return default_mean
        return MeanOp(axis, keepdims)

    @staticmethod
    def sum(*, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        if axis is None and not keepdims:
            return default_sum
        return SumOp(axis, keepdims)

__all__ = [
    'TensorOp', 'TensorOps',
    'broadcast_shapes', 'promote_types',
]
