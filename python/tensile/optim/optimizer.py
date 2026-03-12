#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from ..infra import RootObject
from ..util.stats import get_stats
from ..nn import Module
from ..nn.common import *

from .schedule import LRSchedule, OptimizerSchedule
from .types import OptimizerStep, OptimizerStepHandler, TrainFunction, Batch


class BasicStepHandler(RootObject, Generic[Batch]):

    __slots__ = ('on_start', 'on_end')

    on_start: Callable[['Optimizer', Batch], None]
    on_end: Callable[['Optimizer', Array, Batch], None]

    def __init__(self,
                 on_start: Callable[['Optimizer', Batch], None],
                 on_end: Callable[['Optimizer', Array, Batch], None]):
        self.on_start = on_start
        self.on_end = on_end


class OptimizerSchedules:

    __slots__ = ('schedules',)

    schedules: dict[str, OptimizerSchedule]

    def __init__(self, schedules: dict[str, OptimizerSchedule] = None, /, **kwargs):
        self.schedules = {}
        if schedules:
            for name, schedule in schedules.items():
                if isinstance(schedule, OptimizerSchedule):
                    self.schedules[name] = schedule
        if kwargs:
            for name, schedule in kwargs.items():
                if isinstance(schedule, OptimizerSchedule):
                    self.schedules[name] = schedule

    def alias(self, include_constant: bool = False, aliases: dict[str, str] = None) -> dict[str, OptimizerSchedule]:
        if aliases is None:
            def alias(name: str) -> str: return name
        else:
            def alias(name: str) -> str: return aliases.get(name, name)
        if include_constant:
            return {alias(k): v for k, v in self.schedules.items()}
        return {alias(k): v for k, v in self.schedules.items() if not v.constant}

    def get(self, step: Array, include_constant: bool = False, aliases: dict[str, str] = None) -> dict[str, float]:
        if aliases is None:
            def alias(name: str) -> str: return name
        else:
            def alias(name: str) -> str: return aliases.get(name, name)
        if include_constant:
            return {alias(k): v(step).item() for k, v in self.schedules.items()}
        return {alias(k): v(step).item() for k, v in self.schedules.items() if not v.constant}

    def get_static(self) -> dict[str, Array]:
        step = ten.array(0, dtype=ten.int32)
        return {k: v(step) for k, v in self.schedules.items() if v.constant}



class OptimizerParamGroup(Object):

    __slots__ = ('group', 'params', 'param_tree', 'schedules')

    group: Annotated[int, field(
        doc='The group index for this optimizer group.'
    )]
    params: Annotated[set[str], field(
        doc='The parameters for this optimizer group.',
        default_factory=set,
    )]
    schedules: Annotated[OptimizerSchedules, field(
        doc='The schedules for this optimizer group.',
        default_factory=OptimizerSchedules,
    )]

    def filter_tree(self, arrays: tree.Tree[Array]) -> tree.Tree[Array]:
        group_paths = self.params
        if isinstance(arrays, Module):
            is_trainable = Module.Helpers.is_trainable_parameter_entry
            def include(e: tree.TreeEntry):
                return e.path in group_paths and is_trainable(e)
        else:
            def include(e: tree.TreeEntry):
                return e.path in group_paths and ten.is_array(e.value)
        return tree.filter(arrays, include=include)

    def filter_params(self, arrays: tree.Tree[Array]) -> list[Array]:
        group_paths = self.params
        params = []
        if isinstance(arrays, Module):
            is_trainable = Module.Helpers.is_trainable_parameter_entry
            def include(e: tree.TreeEntry):
                return e.path in group_paths and is_trainable(e)
        else:
            def include(e: tree.TreeEntry):
                return e.path in group_paths and ten.is_array(e.value)

        def add_to_params(e: tree.TreeEntry):
            params.append(e.value)

        tree.apply(arrays, add_to_params, include=include)
        return params

    def current_schedule(self, step: Array, include_constant: bool = False) -> dict[str, float]:
        return self.schedules.get(step, include_constant)

    def _repr_args(self, **options) -> str:
        return f'{self.group}: {len(self.params)} params'


class Optimizer(Object, Generic[Batch]):

    __slots__ = ('model', 'learning_rate', 'eps', 'current_step', 'param_groups', 'step_handler', 'backend')

    model: Annotated[Module, field(
        doc='The model to optimize.'
    )]
    learning_rate: Annotated[LRSchedule, field(
        doc='The learning rate schedule for this optimizer.',
        coerce=True,
    )]
    eps: Annotated[float, field(
        doc='The epsilon value for this optimizer.'
    )]
    current_step: Annotated[Array, field(
        doc='The current step of the optimizer.',
        default_factory=lambda: ten.array(0, dtype=ten.int32)
    )]
    param_groups: Annotated[list[OptimizerParamGroup], field(
        doc='The parameter groups for this optimizer.',
        default_factory=list,
    )]
    step_handler: Annotated[Optional[OptimizerStepHandler[Batch]], field(
        doc='The step handler for this optimizer.',
    )]

    schedule_names: ClassVar[tuple[str, ...]] = ('learning_rate', )
    spec_names: ClassVar[tuple[str, ...]] = ()
    aliases: ClassVar[dict[str, str]] = {}

    def _coerce_param_groups(self, param_groups: Any) -> Optional[list[OptimizerParamGroup]]:
        if param_groups is None: return None
        return [self._build_param_group(g, param_group) for g, param_group in enumerate(param_groups)]

    def postinit(self, spec: Spec):
        super().postinit(spec)
        if not self.param_groups:
            self.param_groups.append(OptimizerParamGroup(
                group=0,
                params=set(n for n, p in tree.flatten(self.model.trainable_parameters())),
                schedules=self.default_schedules(),
            ))
        self._validate_param_groups(self.param_groups)

    def _build_param_group(self, group: int, spec: Any) -> OptimizerParamGroup:
        if isinstance(spec, OptimizerParamGroup): return spec
        if isinstance(spec, dict):
            spec = spec.copy()
            params = spec.pop('params', None)
            group = spec.pop('group', group)
            if not params: raise ValueError(f'No parameters specified for group {group}')
            rest = self._coerce_param_group_specs(spec)
            return coerce(OptimizerParamGroup, {
                'group': group,
                'params': spec.get('params', []),
                'schedules': rest,
            })
        raise ValueError(f'Invalid parameter group spec: {spec}')

    def _coerce_param_group_specs(self, specs: dict[str, Any]) -> dict[str, Any]:
        coerced = {}
        if lr := specs.get('learning_rate'):
            coerced['learning_rate'] = coerce(LRSchedule, lr)
        return coerced

    def _validate_param_groups(self, param_groups: list[OptimizerParamGroup]) -> None:
        if len(param_groups) == 1: return
        for g, group in enumerate(param_groups):
            for name in group.params:
                for other in param_groups[g+1:]:
                    if name in other.params:
                        raise ValueError(f'Duplicate parameter name: {name}')

    def trainable_parameters(self, model: Module = None, group: int = None):
        if model is None: model = self.model

        if group is None:
            return model.trainable_parameters()
        return self.param_groups[group].filter_tree(model)

    def start_step(self, batch: Batch) -> None:
        if step_handler := self.step_handler:
            step_handler.on_start(self, batch)

    def get_hyperparameters(self) -> dict[str, Any]:
        stats = {}
        step = self.current_step
        for g, param_group in enumerate(self.param_groups):
            if schedule := param_group.current_schedule(step, include_constant=True):
                stats[str(g)] = {n: s for n, s in schedule.items()}
        return stats

    def finish_step(self, loss: Array, batch: Batch) -> None:
        if step_handler := self.step_handler:
            step_handler.on_end(self, loss, batch)
        self.current_step += 1

    def grad_stats(self, grads: Iterable[tuple[str, Array]]) -> dict[str, Any]:
        return get_stats(grads)

    def default_schedules(self) -> OptimizerSchedules:
        return OptimizerSchedules({name: getattr(self, name) for name in self.schedule_names})

    def current_schedule(self, param_group: int = 0) -> dict[str, float]:
        return self.param_groups[param_group].schedules.get(self.current_step, aliases=self.aliases)

    def stepper(self, model: Module, train_fn: TrainFunction[Batch]) -> OptimizerStep[Batch]:
        raise NotImplementedError()

    def alias_spec(self, name: str) -> str:
        return self.aliases.get(name, name)

    def backend_schedules(self, current: bool = True) -> dict[str, Any]:
        spec = {}
        if current:
            step = self.current_step
            for name in self.schedule_names:
                schedule = getattr(self, name)
                if schedule is not None:
                    alias = self.alias_spec(name)
                    spec[alias] = schedule(step)
        else:
            for name in self.schedule_names:
                schedule = getattr(self, name)
                if schedule is not None:
                    alias = self.alias_spec(name)
                    spec[alias] = schedule
        return spec

    def backend_spec(self) -> dict[str, Any]:
        spec = {}
        for name in self.spec_names:
            value = getattr(self, name)
            if value is not None:
                spec[name] = value
        return spec


class BaseSGDOptimizer(Optimizer):

    __slots__ = ('momentum', 'weight_decay', 'dampening', 'nesterov')

    momentum: Annotated[OptimizerSchedule, field(
        doc='The momentum for SGD.'
    )]
    weight_decay: Annotated[OptimizerSchedule, field(
        doc='The weight_decay for SGD.'
    )]
    dampening: Annotated[OptimizerSchedule, field(
        doc='The dampening for SGD.'
    )]
    nesterov: Annotated[bool, field(
        doc='Whether to use Nesterov momentum for SGD.'
    )]

    schedule_names = (*Optimizer.schedule_names, 'momentum', 'weight_decay', 'dampening')
    spec_names = ('nesterov',)

    def _coerce_momentum(self, spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    def _coerce_weight_decay(self, spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    def _coerce_dampening(self, spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    def _coerce_param_group_specs(self, specs: dict[str, Any]) -> dict[str, Any]:
        coerced = super()._coerce_param_group_specs(specs)
        if weight_decay := specs.get('weight_decay'):
            coerced['weight_decay'] = self._coerce_weight_decay(weight_decay)
        if momentum := specs.get('momentum'):
            coerced['momentum'] = self._coerce_momentum(momentum)
        if dampening := specs.get('dampening'):
            coerced['dampening'] = self._coerce_dampening(dampening)
        return coerced


class BaseAdamWOptimizer(Optimizer):

    __slots__ = ('weight_decay', 'betas',)

    weight_decay: Annotated[Optional[OptimizerSchedule], field(
        doc='The weight decay coefficient for AdamW.'
    )]
    betas: Annotated[Optional[tuple[float, float]], field(
        doc='The betas for AdamW.'
    )]

    schedule_names = (*Optimizer.schedule_names, 'weight_decay')
    spec_names = ('eps', 'betas')

    def _coerce_weight_decay(self, spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    def _coerce_betas(self, betas: Any):
        if betas is None: return None
        if isinstance(betas, tuple): return betas
        if isinstance(betas, list): return tuple(betas[:2])
        raise ValueError(f'Invalid betas: {betas}')

    def _coerce_param_group_specs(self, specs: dict[str, Any]) -> dict[str, Any]:
        coerced = super()._coerce_param_group_specs(specs)
        if lr := specs.get('learning_rate'):
            coerced['learning_rate'] = coerce(LRSchedule, lr)
        if weight_decay := specs.get('weight_decay'):
            coerced['weight_decay'] = self._coerce_weight_decay(weight_decay)
        return coerced


def add_spec(spec: dict[str, Any], **values: Any) -> dict[str, Any]:
    for key, value in values.items():
        if value is not None: spec[key] = value
    return spec


def add_to_schedules(schedules: dict[str, OptimizerSchedule] = None, /,  **kwargs) -> dict[str, OptimizerSchedule]:
    if schedules is None: schedules = {}
    for name, schedule in kwargs.items():
        if isinstance(schedule, OptimizerSchedule):
            schedules[name] = schedule
    return schedules

