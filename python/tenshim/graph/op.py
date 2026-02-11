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
    if not a:
        return b
    if not b:
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

    __slots__ = ('shape', 'dtype')

    name: str
    arity: int
    args: tuple[TensorType, ...]
    shape: Shape
    dtype: DType

    def __init__(self, shape: Shape, dtype: DType):
        self.shape = shape
        self.dtype = dtype

    @property
    def args(self) -> tuple[TensorType, ...]:
        raise NotImplementedError()

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

    def structure(self) -> Structure:
        return self.shape, self.dtype

    def evaluate(self, index: Array = None) -> Array:
        raise NotImplementedError()

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

    __slots__ = ('arg',)

    arity = 1

    arg: TensorType

    def __init__(self, arg: TensorType, shape: Shape, dtype: DType):
        super().__init__(shape=shape, dtype=dtype)
        self.arg = arg

    @property
    def args(self) -> tuple[TensorType, ...]:
        return self.arg,

    def evaluate(self, index: Array = None) -> Array:
        a = self.arg.get()
        self.print_evaluating(a, index=index)
        return self._evaluate(a, index=index)

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        raise NotImplementedError()


def can_broadcast(from_shape: Shape, to_shape: Shape) -> bool:
    return True


class BroadcastOp(UnaryOp):

    __slots__ = ()

    name = 'broadcast'

    def _options(self, opts: dict[str, Any]) -> None:
        opts['shape'] = self.shape

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.broadcast_to(a, self.shape)

    @classmethod
    def _compile(cls, arg: TensorType, *, shape: Shape = None, **kwargs) -> Self:
        if can_broadcast(arg.shape, shape):
            return cls(arg, shape=shape, dtype=arg.dtype)
        raise ValueError(f'Cannot broadcast: {arg.shape} -> {shape}')


def can_reshape(from_shape: Shape, to_shape: Shape) -> bool:
    return True


class ReshapeOp(UnaryOp):

    __slots__ = ()

    name = 'reshape'

    def _options(self, opts: dict[str, Any]) -> None:
        opts['shape'] = self.shape

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.reshape(a, self.shape)

    @classmethod
    def _compile(cls, arg: TensorType, *, shape: Shape = None, **kwargs) -> Self:
        if can_reshape(arg.shape, shape):
            return cls(arg, shape=shape, dtype=arg.dtype)
        raise ValueError(f'Cannot reshape: {arg.shape} -> {shape}')


class SubscriptOp(UnaryOp):

    __slots__ = ('indices', )

    indices: Indices

    def __init__(self, arg: TensorType, indices: Indices, shape: Shape, dtype: DType):
        super().__init__(arg, shape, dtype)
        self.indices = indices

    def _options(self, opts: dict[str, Any]) -> None:
        opts['indices'] = self.indices

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return a[self.indices]

    @classmethod
    def _compile(cls, arg: TensorType, *, indices: Indices = None, **kwargs) -> Self:
        shape = arg.shape
        return cls(arg, indices, shape=shape, dtype=arg.dtype)


class ElementWiseUnaryOp(UnaryOp):

    __slots__ = ()

    @classmethod
    def _compile(cls, arg: TensorType, **kwargs) -> Self:
        return cls(arg, shape=arg.shape, dtype=cls._compile_dtype(arg))

    @classmethod
    def _compile_dtype(cls, arg: TensorType) -> DType:
        return arg.dtype


class ExpOp(ElementWiseUnaryOp):

    __slots__ = ()

    name = 'exp'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.exp(a if index is None else a[index])


class ExpM1Op(ElementWiseUnaryOp):

    __slots__ = ()

    name = 'expm1'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.expm1(a if index is None else a[index])


class LogOp(ElementWiseUnaryOp):

    __slots__ = ()

    name = 'log'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.log(a if index is None else a[index])


class SqrtOp(ElementWiseUnaryOp):

    __slots__ = ()

    name = 'sqrt'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.sqrt(a if index is None else a[index])


class SquareOp(ElementWiseUnaryOp):

    __slots__ = ()

    name = 'square'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.square(a if index is None else a[index])


class FloorOp(ElementWiseUnaryOp):

    __slots__ = ()

    name = 'floor'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.floor(a if index is None else a[index])




def reduce_shape(shape: Shape, axis: AxisChoice, keepdims: bool) -> Shape:
    if axes := axis:
        if keepdims:
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


class ReduceOp(UnaryOp):

    __slots__ = ('axis', 'keepdims')

    axis: Optional[tuple[int, ...]]
    keepdims: bool

    def __init__(self, arg: TensorType, shape: Shape, dtype: DType, axis: Optional[tuple[int, ...]], keepdims: bool):
        super().__init__(arg, shape, dtype)
        self.axis = axis
        self.keepdims = keepdims

    def _options(self, opts: dict[str, Any]) -> None:
        if self.axis is not None:
            opts['axis'] = self.axis
        if self.keepdims:
            opts['keepdims'] = self.keepdims

    @classmethod
    def _compile(cls, arg: TensorType, *, axis: AxisChoice = None, keepdims: bool = False, **kwargs) -> Self:
        axis = (axis, ) if isinstance(axis, int) else tuple(axis) if axis else None
        if axis:
            ndims = len(arg.shape)
            axis = tuple(ndims + a if a < 0 else a for a in axis)
            if not all(0 <= a < ndims for a in axis):
                raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")

        shape = reduce_shape(arg.shape, axis, keepdims)
        return cls(arg, axis=axis, shape=shape, dtype=arg.dtype, keepdims=keepdims)


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

    __slots__ = ('left', 'right')

    arity = 2

    left: TensorType
    right: TensorType

    def __init__(self, left: TensorType, right: TensorType, shape: Shape, dtype: DType):
        super().__init__(shape, dtype)
        self.left = left
        self.right = right

    @property
    def args(self) -> tuple[TensorType, ...]:
        return self.left, self.right

    def evaluate(self, index: Array = None) -> Array:
        left = self.left.get()
        right = self.right.get()
        self.print_evaluating(left, right, index=index)
        return self._evaluate(left, right, index=index)

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        raise NotImplementedError()

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     tensor.data = None
    #     return None


class MatMulOp(BinaryOp):

    __slots__ = ()

    name = 'matmul'

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        if index is None:
            return a @ b
        return a[index] @ b[index]

    @classmethod
    def _compile(cls, left: TensorType, right: TensorType, **kwargs) -> Self:
        ls = left.shape
        rs = right.shape
        if len(ls) == 1:
            ls = 1, ls[0]
        if len(rs) == 1:
            rs = rs[0], 1
        if ls[-1] != rs[-2]:
            raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')
        shape = broadcast_shapes(ls[:-2], rs[:-2]) + (ls[-2], rs[-1])
        dtype = promote_types(left.dtype, right.dtype)  # ten.promote_types(a.dtype, b.dtype)
        return cls(left, right, shape=shape, dtype=dtype)


class ElementWiseBinaryOp(BinaryOp):

    __slots__ = ()

    @classmethod
    def _compile(cls, left: TensorType, right: TensorType, **kwargs) -> Self:
        ls = left.shape
        rs = right.shape
        if ls != rs:
            shape = broadcast_shapes(ls, rs)
            if shape != ls:
                left = left.broadcast(shape)
            if shape != rs:
                right = right.broadcast(shape)
        else:
            shape = ls
        dtype = cls._promote_types(left.dtype, right.dtype)
        return cls(left, right, shape=shape, dtype=dtype)

    @classmethod
    def _promote_types(cls, left: DType, right: DType) -> DType:
        return promote_types(left, right)

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     tensor.data = None


class CompareOp(ElementWiseBinaryOp):

    __slots__ = ()

    @classmethod
    def _promote_types(cls, left: DType, right: DType) -> DType:
        return ten.bool_


class GreaterOp(CompareOp):

    __slots__ = ()

    name = 'greater'

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        if index is None:
            return a > b
        return a[index] > b[index]


class GreaterEqualOp(CompareOp):

    __slots__ = ()

    name = 'greater_equal'

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        if index is None:
            return a >= b
        return a[index] >= b[index]


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

class SubOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'sub'

    def _evaluate(self, a: Array, b: Array, index: Array = None) -> Array:
        if index is None:
            return a - b
        return a[index] - b[index]


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


class TensorOps:

    @staticmethod
    def greater(left: TensorType, right: TensorType) -> GreaterOp:
        return GreaterOp.compile(left, right)

    @staticmethod
    def greater_equal(left: TensorType, right: TensorType) -> GreaterEqualOp:
        return GreaterEqualOp.compile(left, right)

    @staticmethod
    def add(left: TensorType, right: TensorType) -> AddOp:
        return AddOp.compile(left, right)

    @staticmethod
    def sub(left: TensorType, right: TensorType) -> SubOp:
        return SubOp.compile(left, right)

    @staticmethod
    def mul(left: TensorType, right: TensorType) -> MulOp:
        return MulOp.compile(left, right)

    @staticmethod
    def matmul(left: TensorType, right: TensorType) -> MatMulOp:
        return MatMulOp.compile(left, right)

    @staticmethod
    def broadcast(arg: TensorType, *, shape: Shape) -> BroadcastOp:
        return BroadcastOp.compile(arg, shape=shape)

    @staticmethod
    def reshape(arg: TensorType, *, shape: Shape) -> ReshapeOp:
        return ReshapeOp.compile(arg, shape=shape)

    @staticmethod
    def subscript(arg: TensorType, *, indices: Indices) -> SubscriptOp:
        return SubscriptOp.compile(arg, indices=indices)

    @staticmethod
    def min(arg: TensorType, *, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        return MinOp.compile(arg, axis=axis, keepdims=keepdims)

    @staticmethod
    def max(arg: TensorType, *, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        return MaxOp.compile(arg, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(arg: TensorType, *, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        return MeanOp.compile(arg, axis=axis, keepdims=keepdims)

    @staticmethod
    def sum(arg: TensorType, *, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        return SumOp.compile(arg, axis=axis, keepdims=keepdims)

__all__ = [
    'TensorOp', 'TensorOps',
    'broadcast_shapes', 'promote_types',
]
