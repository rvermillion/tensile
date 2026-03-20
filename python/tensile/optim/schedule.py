#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

import types
from typing import ParamSpec

from ..common import *


def zero(): return ten.array(0.)


def noop(*args, **kwargs):
    pass


class OptimizerSchedule(Object):

    __slots__ = ()

    def __call__(self, step: Array) -> Array:
        raise NotImplementedError()

    constant: ClassVar[bool] = False


class LRSchedule(Protocol):

    def __call__(self, step: Array) -> Array: ...


def coerce_lr_schedule(spec: Any = None, /, **kwargs) -> Any:
    if isinstance(spec, BaseSchedule):
        return spec
    return lr_meta.default_coerce(spec, **kwargs)


lr_meta = meta.Meta.for_class(LRSchedule, build=True)
lr_meta.coerce = coerce_lr_schedule


@provides(OptimizerSchedule, 'function')
@provides(LRSchedule, 'function')
class FunctionSchedule(OptimizerSchedule):

    __slots__ = ('schedule',)

    schedule: Callable[[Array], Array]

    def __call__(self, step: Array) -> Array:
        return self.schedule(step)


@meta.provides_from_type(LRSchedule, types.FunctionType, types.MethodType)
@meta.provides_from_type(OptimizerSchedule, types.FunctionType, types.MethodType)
def provide_function(schedule: Callable[[Array], Array]) -> FunctionSchedule:
    return FunctionSchedule(schedule=schedule)


@provides(OptimizerSchedule, 'constant')
@provides(LRSchedule, 'constant')
class ConstantSchedule(OptimizerSchedule):

    __slots__ = ('value',)

    value: Array
    constant = True

    def _coerce_value(self, value) -> Array:
        return value if ten.is_array(value) else ten.array(value)

    def __call__(self, step: Array) -> Array:
        return self.value


@meta.provides_from_type(LRSchedule, int, float)
@meta.provides_from_type(OptimizerSchedule, int, float)
def provide_constant(value: int | float) -> ConstantSchedule:
    return ConstantSchedule(value=ten.array(value))


class BaseSchedule(OptimizerSchedule):

    __slots__ = ()

    def value_for_step(self, step: Array) -> Array:
        raise NotImplementedError()

    def __call__(self, step: Array) -> Array:
        return self.value_for_step(step)


# zero_schedule = ConstantSchedule(value=0.)

def zero_schedule_factory():
    return ConstantSchedule(value=0.)


def constant_array_factory(value: ten.Scalar, dtype: DType = None) -> Callable[[], Array]:
    return lambda: ten.array(value, dtype=dtype)


@provides(OptimizerSchedule, 'ramp')
class RampSchedule(BaseSchedule):

    start_step: Annotated[Array, field(default_factory=zero)]
    stop_step: Annotated[Array, field()]
    before: Annotated[OptimizerSchedule, field(default_factory=zero_schedule_factory)]
    after: Annotated[OptimizerSchedule, field()]

    def _coerce_start_step(self, start_step) -> Array:
        return ten.array(start_step)

    def _coerce_stop_step(self, stop_step) -> Array:
        return ten.array(stop_step)

    def value_for_step(self, step: Array) -> Array:
        # ten.debug_eval(step)

        before = self.before(step)
        after = self.after(step)
        percent = ten.clip( (step - self.start_step) / (self.stop_step - self.start_step), 0., 1.)
        start = self.before(self.start_step)
        stop = self.after(self.stop_step)
        ramp = start + percent * (stop - start)

        value = ten.where(step < self.start_step, before, ten.where(step > self.stop_step, after, ramp))

        return value


@provides(OptimizerSchedule, 'cosine')
class CosineSchedule(BaseSchedule):

    __slots__ = ('zero_step', 'pi_step', 'base')

    zero_step: Annotated[Array, field(default_factory=zero)]
    pi_step: Annotated[Array, field()]
    base: Annotated[OptimizerSchedule, field()]
    scale: Annotated[OptimizerSchedule, field()]

    def _coerce_zero_step(self, zero_step) -> Array:
        return ten.array(zero_step)

    def _coerce_pi_step(self, pi_step) -> Array:
        return ten.array(pi_step)

    def value_for_step(self, step: Array) -> Array:
        # ten.debug_eval(step)

        progress = (step - self.zero_step) / ten.maximum(1, self.pi_step - self.zero_step)
        cosine = 0.5 * (1 + ten.cos(ten.pi * progress))
        value = self.base(step) - self.scale(step) * cosine
        return value


class BinaryOpSchedule(BaseSchedule):

    __slots__ = ('left', 'right')

    left: Annotated[OptimizerSchedule, field()]
    right: Annotated[OptimizerSchedule, field()]

    @staticmethod
    def op(left: Array, right: Array) -> Array:
        raise NotImplementedError()

    def value_for_step(self, step: Array) -> Array:
        # ten.debug_eval(step)

        return ten.maximum(self.left(step), self.right(step))


@provides(OptimizerSchedule, 'max')
class MaxSchedule(BinaryOpSchedule):

    __slots__ = ()

    op = ten.maximum


@provides(OptimizerSchedule, 'min')
class MinSchedule(BinaryOpSchedule):

    __slots__ = ()

    op = ten.minimum


@provides(OptimizerSchedule, 'warmup')
class WarmupSchedule(BaseSchedule):

    warmup_steps: Annotated[Array, field()]
    after_warmup: Annotated[OptimizerSchedule, field()]

    def _coerce_warmup_steps(self, warmup_steps) -> Array:
        return ten.array(warmup_steps)

    def warmup(self, percent: Array) -> Array:
        return percent * self.after_warmup(self.warmup_steps)

    def value_for_step(self, step: Array) -> Array:
        # ten.debug_eval(step)

        warmup = self.warmup(step)
        after_warmup = self.after_warmup(step)

        value = ten.where(step < self.warmup_steps, warmup, after_warmup)

        return value


@provides(LRSchedule, 'warmup')
class WarmupLRSchedule(BaseSchedule):

    warmup_steps: Annotated[Array, field()]
    base: Annotated[Array, field(default_factory=constant_array_factory(1e-3))]

    def _coerce_warmup_steps(self, warmup_steps) -> Array:
        return ten.array(warmup_steps)

    def _coerce_base(self, base) -> Array:
        return ten.array(base)

    def value_for_step(self, step: Array) -> Array:
        # ten.debug_eval(step)

        warmup_lr = self.base * step / self.warmup_steps
        after_warmup_lr = self.lr_after_warmup(step)

        lr = ten.where(step < self.warmup_steps, warmup_lr, after_warmup_lr)

        return lr

    def lr_after_warmup(self, step: Array) -> Array:
        return self.base

    def _repr_args(self, **options) -> str:
        return f'warmup={self.warmup_steps}, base={self.base}'

    def report_message(self, step, lr) -> str:
        return (
            '  - set learning rate at step {step} to {lr:.9f} (base {base:.9f}, warmup {warmup})'.format(
                step=step, lr=lr, base=self.base, warmup=self.warmup_steps
            )
        )


@meta.provides(LRSchedule, 'cosine')
class CosineLRSchedule(WarmupLRSchedule):

    total_steps: Annotated[Array, field(default_factory=zero)]
    min: Annotated[Array, field(default_factory=constant_array_factory(1e-5))]

    # def __init__(self, total_steps: int, min_lr: float = 1e-5, **kwargs):
    #     super().__init__(**kwargs)
    #     self.total_steps = total_steps
    #     self.min_lr = ten.array(min_lr)

    def _coerce_total_steps(self, total_steps) -> Array:
        return ten.array(total_steps)

    def _coerce_min(self, min_lr) -> Array:
        return ten.array(min_lr)

    def lr_after_warmup(self, step) -> Array:
        progress = (step - self.warmup_steps) / ten.maximum(1, self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1 + ten.cos(ten.pi * progress))
        lr = self.min + (self.base - self.min) * cosine
        return lr

    def _repr_args(self, **options) -> str:
        return super()._repr_args(**options) + f', total={self.total_steps}, min={self.min}'


P = ParamSpec('P')


def call_every(n: int, fn: Callable[P, bool], start: int = 1) -> Callable[P, bool]:
    i = start
    def inner(*args: P.args, **kwargs: P.kwargs):
        nonlocal i
        if i % n == 0:
            i += 1
            return fn(*args, **kwargs)
        else:
            i += 1
            return False
    return inner
