#  Copyright (c) 2026. Fulcrum Analytics, Inc. All Rights Reserved.
#  This file is part of the Fulcrum Impact product and the property of Fulcrum Analytics, Inc.
#  WARNING: CONFIDENTIAL TRADE SECRETS OF FULCRUM ANALYTICS, INC.
#  UNAUTHORIZED COPYING, DISTRIBUTION, OR DISCLOSURE IS STRICTLY PROHIBITED

from typing import Any, Callable, Optional, Self, Sequence, TypeAlias, Union, TYPE_CHECKING
from .. import ten
from .common import Array, AxisChoice, Axes, Base, DType, Functional, Index, Indices, Shape, TensorType
from .util import broadcast_shapes, promote_types
from .region import IndexedRegion, Region, RegionIndex


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


class TensorOp(Base):

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

    def evaluate(self, index: Array = None) -> Array:
        raise NotImplementedError()

    def map_region(self, arg_index: int, region: Region) -> Region:
        return Region.from_key(self.shape, ...)

    @classmethod
    def build(cls, op: str, *args: TensorType, **options) -> Self:
        subcls = cls.names[op]
        return subcls.compile(*args, **options)

    @classmethod
    def compile(cls, *args: TensorType, **options) -> Self:
        if cls.arity != len(args):
            raise ValueError(f'Expected {cls.arity} arguments for {cls}, got {len(args)}')
        return cls._compile(*args, **options)

    @classmethod
    def _compile(cls, *args: TensorType, **options) -> Self:
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

    names: dict[str, type['TensorOp']] = {}

    def __init_subclass__(cls, **kwargs):
        if name := cls.__dict__.get('name'):
            TensorOp.names[name] = cls


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

    def map_region(self, arg_index: int, region: Region) -> Region:
        return region.broadcast(self.shape)
        # return super().map_region(arg_index, region)

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
        if shape is None:
            raise ValueError(f'Cannot reshape: {arg.shape} -> {shape}')
        orig = arg.shape
        size = 1
        for d in orig:
            size *= d
        seen_neg = False
        for d in shape:
            if d >= 0:
                if size % d:
                    raise ValueError(f'Cannot reshape: {arg.shape} -> {shape}')
                size //= d
            elif seen_neg:
                raise ValueError(f'Cannot reshape: {arg.shape} -> {shape}')
            else:
                seen_neg = True
        if seen_neg:
            shape = tuple(size if d < 0 else d for d in shape)
        return cls(arg, shape=shape, dtype=arg.dtype)


class SubscriptOp(UnaryOp):

    __slots__ = ('indices', )

    indices: Indices

    name = 'subscript'

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


class TransposeOp(UnaryOp):

    __slots__ = ('axes', )

    axes: Axes

    name = 'transpose'

    def __init__(self, arg: TensorType, axes: Axes, shape: Shape, dtype: DType):
        super().__init__(arg, shape, dtype)
        self.axes = axes

    def _options(self, opts: dict[str, Any]) -> None:
        opts['axes'] = self.axes

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.transpose(a, axes=self.axes)

    def map_region(self, arg_index: int, region: Region) -> Region:
        if isinstance(region, IndexedRegion):
            indices = tuple(region.indices[a] for a in self.axes)
            return IndexedRegion(self.shape, indices)
        return super().map_region(arg_index, region)

    @classmethod
    def _compile(cls, arg: TensorType, *, axes: Axes = None, **kwargs) -> Self:
        if axes is None:
            axes = tuple(reversed(range(arg.ndim)))
        elif len(axes) != arg.ndim:
            raise ValueError(f'Cannot transpose {arg.ndim}D tensor with {len(axes)} axes')
        else:
            axes = tuple(axes)
        orig_shape = arg.shape
        shape = tuple(orig_shape[i] for i in axes)
        return cls(arg, axes, shape=shape, dtype=arg.dtype)


class ElementWiseUnaryOp(UnaryOp):

    __slots__ = ()

    def map_region(self, arg_index: int, region: Region) -> Region:
        return region

    @classmethod
    def _compile(cls, arg: TensorType, **kwargs) -> Self:
        return cls(arg, shape=arg.shape, dtype=cls._compile_dtype(arg))

    @classmethod
    def _compile_dtype(cls, arg: TensorType) -> DType:
        return arg.dtype


class FunctionalOp(ElementWiseUnaryOp):

    __slots__ = ('func', )

    name = 'func'
    func: Functional

    def __init__(self, arg: TensorType, func: Functional, shape: Shape, dtype: DType):
        super().__init__(arg, shape, dtype)
        self.func = func

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return self.func(a if index is None else a[index])

    @classmethod
    def _compile(cls, arg: TensorType, func: Functional = None, **kwargs) -> Self:
        if func is None:
            raise ValueError(f'Cannot compile {cls.name} without a function')
        return cls(arg, func=func, shape=arg.shape, dtype=cls._compile_dtype(arg))



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


class AxisUnaryOp(UnaryOp):

    __slots__ = ('axis', )

    axis: Optional[tuple[int, ...]]

    def __init__(self, arg: TensorType, shape: Shape, dtype: DType, axis: Optional[tuple[int, ...]]):
        super().__init__(arg, shape, dtype)
        self.axis = axis

    def _options(self, opts: dict[str, Any]) -> None:
        super()._options(opts)
        if self.axis is not None:
            opts['axis'] = self.axis

    @classmethod
    def _compile(cls, arg: TensorType, *, axis: AxisChoice = None, **kwargs) -> Self:
        axis = cls._compile_axis(arg, axis=axis, **kwargs)
        # axis = (axis, ) if isinstance(axis, int) else tuple(axis) if axis else None
        # if axis:
        #     ndims = len(arg.shape)
        #     axis = tuple(ndims + a if a < 0 else a for a in axis)
        #     if not all(0 <= a < ndims for a in axis):
        #         raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")

        shape = cls._compile_shape(arg, axis=axis, **kwargs)
        return cls(arg, axis=axis, shape=shape, dtype=arg.dtype)

    @classmethod
    def _compile_axis(cls, arg: TensorType, *, axis: AxisChoice = None, **kwargs) -> Optional[Axes]:
        axis = (axis, ) if isinstance(axis, int) else tuple(axis) if axis else None
        if axis:
            ndims = len(arg.shape)
            axis = tuple(ndims + a if a < 0 else a for a in axis)
            if not all(0 <= a < ndims for a in axis):
                raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")
            return axis
        return None

    @classmethod
    def _compile_shape(cls, arg: TensorType, *, axis: Axes, **kwargs) -> Shape:
        return arg.shape


class ExpandDimsOp(AxisUnaryOp):

    __slots__ = ()

    name = 'expand_dims'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.expand_dims(a, axis=self.axis)

    @classmethod
    def _compile_axis(cls, arg: TensorType, *, axis: AxisChoice = None, **kwargs) -> Optional[Axes]:
        axis = (axis, ) if isinstance(axis, int) else tuple(axis) if axis else None
        if axis:
            ndims = len(arg.shape) + len(axis)
            axis = tuple(ndims + a if a < 0 else a for a in axis)
            if not all(0 <= a < ndims for a in axis):
                raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")
            return axis
        return None

    @classmethod
    def _compile_shape(cls, arg: TensorType, *, axis: Axes, **kwargs) -> Shape:
        shape = [0] * (len(arg.shape) + len(axis))
        for a in axis:
            shape[a] = 1
        a = 0
        for s in arg.shape:
            while shape[a] == 1:
                a += 1
            shape[a] = s
            a += 1
        return tuple(shape)


class ReduceOp(AxisUnaryOp):

    __slots__ = ('keepdims', )

    keepdims: bool

    def __init__(self, arg: TensorType, shape: Shape, dtype: DType, axis: Optional[tuple[int, ...]], keepdims: bool):
        super().__init__(arg, shape, dtype, axis)
        self.keepdims = keepdims

    def map_region(self, arg_index: int, region: Region) -> Region:
        if isinstance(region, IndexedRegion):
            orig_list: list = list(region.indices)
            for a in self.axis:
                orig_list[a] = None
            indices = tuple(ind for ind in orig_list if ind is not None)
            if len(indices) != len(self.shape):
                raise ValueError(f'Cannot reduce {len(orig_list)} indices to {len(self.shape)}')
            return IndexedRegion(self.shape, indices)
        return super().map_region(arg_index, region)

    def _options(self, opts: dict[str, Any]) -> None:
        super()._options(opts)
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

        shape = cls._compile_shape(arg, axis=axis, keepdims=keepdims, **kwargs)
        return cls(arg, axis=axis, shape=shape, dtype=arg.dtype, keepdims=keepdims)

    @classmethod
    def _compile_shape(cls, arg: TensorType, *, axis: Axes, keepdims: bool = False, **kwargs) -> Shape:
        return reduce_shape(arg.shape, axis, keepdims)


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


class SoftMaxOp(ReduceOp):

    __slots__ = ()

    name = 'softmax'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.softmax(a, axis=self.axis, keepdims=self.keepdims)


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


class LogSumExpOp(ReduceOp):

    __slots__ = ()

    name = 'logsumexp'

    def _evaluate(self, a: Array, index: Array = None) -> Array:
        return ten.logsumexp(a, axis=self.axis, keepdims=self.keepdims)


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
            if ls[-1] != rs[-1]:
                raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')
            shape = broadcast_shapes(ls[:-2], rs[:-1]) + (ls[-2], )
        else:
            if ls[-1] != rs[-2]:
                raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')
            shape = broadcast_shapes(ls[:-2], rs[:-2]) + (ls[-2], rs[-1])
        dtype = promote_types(left.dtype, right.dtype)  # ten.promote_types(a.dtype, b.dtype)
        return cls(left, right, shape=shape, dtype=dtype)


class ElementWiseBinaryOp(BinaryOp):

    __slots__ = ()

    def map_region(self, arg_index: int, region: Region) -> Region:
        return region

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
    def functional(arg: TensorType, *, func: Functional) -> FunctionalOp:
        return FunctionalOp.compile(arg, func=func)

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

    @staticmethod
    def logsumexp(arg: TensorType, *, axis: AxisChoice = None, keepdims: bool = False) -> ReduceOp:
        return LogSumExpOp.compile(arg, axis=axis, keepdims=keepdims)

    @staticmethod
    def expand_dims(arg: TensorType, *, axis: AxisChoice = None) -> ExpandDimsOp:
        return ExpandDimsOp.compile(arg, axis=axis)

    @staticmethod
    def transpose(arg: TensorType, *, axes: Axes = None) -> TransposeOp:
        return TransposeOp.compile(arg, axes=axes)


__all__ = [
    'TensorOp', 'TensorOps',
    'broadcast_shapes', 'promote_types',
]
