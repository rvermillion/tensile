#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from typing import Any, Optional, Self, Sequence
from .. import ten
from .common import Array, AxisChoice, Axes, Base, DType, Functional, Indices, Shape, TensorType
from .util import broadcast_shapes, promote_types
from .region import IndexedRegion, Region, Regions, RegionIndex


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
    def ndim(self) -> int:
        return len(self.shape)

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

    def evaluate(self, region: Region = None) -> Array:
        raise NotImplementedError()

    def map_region(self, arg_index: int, region: Region) -> Region:
        return Regions.full(self.shape)

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

    def print_evaluating(self, *args: Array, region: Region = None) -> None:
        disp = ', '.join(str(arg) for arg in args)
        if region is None:
            self.info(f'Evaluating {self} with ({disp})')
        else:
            self.info(f'Evaluating {self} with ({disp}) for region {region}')

    def name_with_options(self) -> str:
        if opts := self.options():
            return self.name + '[' + ', '.join(f'{k}={v!r}' for k, v in opts.items()) + ']'
        return self.name

    def derivation(self, short: bool = False) -> str:
        return self.name_with_options() + '(' + ", ".join(arg.display(short=short) for arg in self.args) + ')'

    def __repr__(self):
        return self.name_with_options()

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

    def evaluate(self, region: Region = None) -> Array:
        a = self.arg.get()
        self.print_evaluating(a, region=region)
        return self._evaluate(a, region=region)

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        raise NotImplementedError()


def can_broadcast(from_shape: Shape, to_shape: Shape) -> bool:
    return True


class BroadcastOp(UnaryOp):

    __slots__ = ()

    name = 'broadcast'

    def _options(self, opts: dict[str, Any]) -> None:
        opts['shape'] = self.shape

    def _evaluate(self, a: Array, region: Region = None) -> Array:
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

    def _evaluate(self, a: Array, region: Region = None) -> Array:
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

    def _evaluate(self, a: Array, region: Region = None) -> Array:
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

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.transpose(a, axes=self.axes)

    def map_region(self, arg_index: int, region: Region) -> Region:
        if isinstance(region, IndexedRegion):
            indices = tuple(region.indices[a] for a in self.axes)
            return Regions.indexed(self.shape, indices)
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


class ClipOp(ElementWiseUnaryOp):

    __slots__ = ('min', 'max')

    min: Optional[Array]
    max: Optional[Array]

    name = 'clip'

    # noinspection PyShadowingBuiltins
    def __init__(self, arg: TensorType, min: Optional[Array], max: Optional[Array], shape: Shape, dtype: DType):
        super().__init__(arg, shape, dtype)
        if min is None and max is None:
            raise ValueError('Must specify at least one of min or max')
        self.min = min
        self.max = max

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.clip(a, self.min, self.max)

    @classmethod
    def _compile(cls, arg: TensorType, min: ten.Scalar | Array | None = None, max: ten.Scalar | Array | None = None, **kwargs) -> Self:
        if min is not None and not ten.is_array(min):
            min = ten.array(min)
        if max is not None and not ten.is_array(max):
            max = ten.array(max)
        return cls(arg, min=min, max=max, shape=arg.shape, dtype=cls._compile_dtype(arg))


class FunctionOp(ElementWiseUnaryOp):

    __slots__ = ()

    func: Functional

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        # return self.func(a if index is None else a[index])
        return self.func(a)


class FunctionalOp(FunctionOp):

    __slots__ = ('func', )

    name = 'func'
    func: Functional

    def __init__(self, arg: TensorType, func: Functional, shape: Shape, dtype: DType):
        super().__init__(arg, shape, dtype)
        self.func = func

    @classmethod
    def _compile(cls, arg: TensorType, func: Functional = None, **kwargs) -> Self:
        if func is None:
            raise ValueError(f'Cannot compile {cls.name} without a function')
        return cls(arg, func=func, shape=arg.shape, dtype=cls._compile_dtype(arg))



class ExpOp(FunctionOp):

    __slots__ = ()

    name = 'exp'
    func = ten.exp


class ExpM1Op(FunctionOp):

    __slots__ = ()

    name = 'expm1'
    func = ten.expm1


class LogOp(FunctionOp):

    __slots__ = ()

    name = 'log'
    func = ten.log


class SqrtOp(FunctionOp):

    __slots__ = ()

    name = 'sqrt'
    func = ten.sqrt


class SquareOp(FunctionOp):

    __slots__ = ()

    name = 'square'
    func = ten.square


class FloorOp(FunctionOp):

    __slots__ = ()

    name = 'floor'
    func = ten.floor



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
            ndim = len(arg.shape)
            axis = tuple(ndim + a if a < 0 else a for a in axis)
            if not all(0 <= a < ndim for a in axis):
                raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")
            return axis
        return None

    @classmethod
    def _compile_shape(cls, arg: TensorType, *, axis: Axes, **kwargs) -> Shape:
        return arg.shape


class ExpandDimsOp(AxisUnaryOp):

    __slots__ = ()

    name = 'expand_dims'

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.expand_dims(a, axis=self.axis)

    def map_region(self, arg_index: int, region: Region) -> Region:
        if isinstance(region, IndexedRegion):
            indices: list[Optional[RegionIndex]] = [None] * self.ndim
            for a in self.axis:
                indices[a] = RegionIndex.single(0)
            a = 0
            for ind in region.indices:
                while indices[a] is not None:
                    a += 1
                indices[a] = ind
                a += 1
            return Regions.indexed(self.shape, tuple(indices))
        return Region.from_key(self.shape, ...)

    @classmethod
    def _compile_axis(cls, arg: TensorType, *, axis: AxisChoice = None, **kwargs) -> Optional[Axes]:
        axis = (axis, ) if isinstance(axis, int) else tuple(axis) if axis else None
        if axis:
            ndim = len(arg.shape) + len(axis)
            axis = tuple(ndim + a if a < 0 else a for a in axis)
            if not all(0 <= a < ndim for a in axis):
                raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")
            return axis
        raise ValueError(f'Cannot expand dims on {arg.shape} with axis {axis}')

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


class SqueezeOp(AxisUnaryOp):

    __slots__ = ()

    name = 'squeeze'

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.squeeze(a, axis=self.axis)

    def map_region(self, arg_index: int, region: Region) -> Region:
        if isinstance(region, IndexedRegion):
            indices = list(region.indices)
            for a in self.axis:
                del indices[a]
            return Regions.indexed(self.shape, tuple(indices))
        return Region.from_key(self.shape, ...)

    @classmethod
    def _compile_axis(cls, arg: TensorType, *, axis: AxisChoice = None, **kwargs) -> Optional[Axes]:
        ndim = arg.ndim
        if axis is None:
            axis = tuple(i for i in range(ndim) if i == 1)
        elif isinstance(axis, int):
            if axis < 0:
                axis += ndim
            axis = axis,
        else:
            axis = tuple(a + ndim if a < 0 else a for a in axis)
        if axis:
            if not all(0 <= a < ndim for a in axis):
                raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")
            shape = arg.shape
            if not all(shape[a] == 1 for a in axis):
                raise ValueError(f"Can only squeeze axes of dimension 1: {axis} out of bounds for shape {arg.shape}")
            return axis
        raise ValueError(f'Cannot squeeze {arg.shape} with axis {axis}')

    @classmethod
    def _compile_shape(cls, arg: TensorType, *, axis: Axes, **kwargs) -> Shape:
        shape = list(arg.shape)
        for a in axis:
            del shape[a]
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
            return Regions.indexed(self.shape, indices)
        return super().map_region(arg_index, region)

    def _options(self, opts: dict[str, Any]) -> None:
        super()._options(opts)
        if self.keepdims:
            opts['keepdims'] = self.keepdims

    @classmethod
    def _compile(cls, arg: TensorType, *, axis: AxisChoice = None, keepdims: bool = False, **kwargs) -> Self:
        axis = (axis, ) if isinstance(axis, int) else tuple(axis) if axis else None
        if axis:
            ndim = len(arg.shape)
            axis = tuple(ndim + a if a < 0 else a for a in axis)
            if not all(0 <= a < ndim for a in axis):
                raise ValueError(f"Axis {axis} out of bounds for shape {arg.shape}")

        shape = cls._compile_shape(arg, axis=axis, keepdims=keepdims, **kwargs)
        return cls(arg, axis=axis, shape=shape, dtype=arg.dtype, keepdims=keepdims)

    @classmethod
    def _compile_shape(cls, arg: TensorType, *, axis: Axes, keepdims: bool = False, **kwargs) -> Shape:
        return reduce_shape(arg.shape, axis, keepdims)


class MinOp(ReduceOp):

    __slots__ = ()

    name = 'min'

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.min(a, axis=self.axis, keepdims=self.keepdims)


class MaxOp(ReduceOp):

    __slots__ = ()

    name = 'max'

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.max(a, axis=self.axis, keepdims=self.keepdims)


class MeanOp(ReduceOp):

    __slots__ = ()

    name = 'mean'

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.mean(a, axis=self.axis, keepdims=self.keepdims)


class SoftMaxOp(ReduceOp):

    __slots__ = ()

    name = 'softmax'

    def _evaluate(self, a: Array, region: Region = None) -> Array:
        return ten.softmax(a, axis=self.axis, keepdims=self.keepdims)


class SumOp(ReduceOp):

    __slots__ = ()

    name = 'sum'

    def _evaluate(self, a: Array, region: Region = None) -> Array:
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

    def _evaluate(self, a: Array, region: Region = None) -> Array:
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

    def evaluate(self, region: Region = None) -> Array:
        left = self.left.get()
        right = self.right.get()
        self.print_evaluating(left, right, region=region)
        return self._evaluate(left, right, region=region)

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        raise NotImplementedError()

    # def update(self, tensor: 'EventTensor', event: TensorEvent, arg_index: int) -> Optional[TensorEvent]:
    #     tensor.data = None
    #     return None


class MatMulOp(BinaryOp):

    __slots__ = ()

    name = 'matmul'

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        # if index is None:
        return a @ b
        # return a[index] @ b[index]

    def map_region(self, arg_index: int, region: Region) -> Region:
        shape = self.shape
        if region.empty:
            return Regions.empty(shape)
        if not 0 <= arg_index <= 1:
            raise ValueError(f'Invalid arg index {arg_index} for {self}')
        if isinstance(region, IndexedRegion):
            # We special case vector-matrix and matrix-vector multiplication

            if self.left.ndim == 1:
                # The left arg is a 1D vector...

                if arg_index == 0 or self.right.ndim == 1:
                    # If the left vector is the one that changed or the right is also a 1D vector
                    # we return a full region
                    return Regions.full(shape)

                # If the right arg is a matrix and is the one that changed, then we
                # pass through all but the last contracted index which we set to fully changed
                indices = list(region.indices[:-1])
                indices[-1] = Regions.full_index(shape[-1])
            elif self.right.ndim == 1:
                # The right arg is a 1D vector....
                if arg_index == 0:
                    # If the left arg is the one that changed, then we pass through all
                    # but the last index, which is contracted out of the shape
                    indices = list(region.indices[:-1])
                else:
                    return Regions.full(shape)
            else:
                # It's a (possibly batched) 2D by 2D matmul so we copy the region indexes
                # and then change the right one based on the way changes flow through matrix
                # multiplication

                indices = list(region.indices)
                if arg_index == 0:
                    # If the left arg changed, then we make the last index be full
                    indices[-1] = Regions.full_index(shape[-1])
                else:
                    # If the right arg changed, then we make the second to last index be full
                    indices[-2] = Regions.full_index(shape[-2])
            return Regions.indexed(shape, tuple(indices))
        return Regions.full(shape)

    @classmethod
    def _compile(cls, left: TensorType, right: TensorType, **kwargs) -> Self:
        ls = left.shape
        rs = right.shape
        ldim = len(ls)
        rdim = len(rs)
        if ldim == 0 or rdim == 0:
            raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')

        if ldim == 1:
            if rdim == 1:
                if ls[-1] != rs[-1]:
                    raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')
                shape = ()
            else:
                if ls[-1] != rs[-2]:
                    raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')
                shape = rs[:-2] + (rs[-1],)
        elif rdim == 1:
            if ls[-1] != rs[-1]:
                raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')
            shape = ls[:-1]
        else:
            if ls[-1] != rs[-2]:
                raise ValueError(f'Shape mismatch for {cls.name}: {ls} != {rs}')
            matrix_shape = ls[-2], rs[-1]
            if ldim > 2 or rdim > 2:
                lbs = ls[:-2]
                rbs = rs[:-2]
                batch_shape = broadcast_shapes(lbs, rbs)
                if batch_shape != lbs:
                    left = left.broadcast(batch_shape)
                if batch_shape != rbs:
                    right = right.broadcast(batch_shape)
                shape = batch_shape + matrix_shape
            else:
                shape = matrix_shape
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

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        # if index is None:
        return a > b
        # return a[index] > b[index]


class GreaterEqualOp(CompareOp):

    __slots__ = ()

    name = 'greater_equal'

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        # if index is None:
        return a >= b
        # return a[index] >= b[index]


class AddOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'add'

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        # if index is None:
        return a + b
        # return a[index] + b[index]

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

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        # if index is None:
        return a - b
        # return a[index] - b[index]


class MulOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'mul'

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        # if index is None:
        return a * b
        # return a[index] * b[index]

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

class MaximumOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'maximum'

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        return ten.maximum(a, b)


class MinimumOp(ElementWiseBinaryOp):

    __slots__ = ()

    name = 'minimum'

    def _evaluate(self, a: Array, b: Array, region: Region = None) -> Array:
        return ten.minimum(a, b)


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
    def maximum(left: TensorType, right: TensorType) -> MaximumOp:
        return MaximumOp.compile(left, right)

    @staticmethod
    def minimum(left: TensorType, right: TensorType) -> MinimumOp:
        return MinimumOp.compile(left, right)

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
    def squeeze(arg: TensorType, *, axis: AxisChoice = None) -> SqueezeOp:
        return SqueezeOp.compile(arg, axis=axis)

    # noinspection PyShadowingBuiltins
    @staticmethod
    def clip(arg: TensorType, *, min: ten.Scalar|Array|None = None, max: ten.Scalar|Array|None = None) -> ClipOp:
        return ClipOp.compile(arg, min=min, max=max)

    @staticmethod
    def transpose(arg: TensorType, *, axes: Axes = None) -> TransposeOp:
        return TransposeOp.compile(arg, axes=axes)


__all__ = [
    'TensorOp', 'TensorOps',
    'broadcast_shapes', 'promote_types',
]
