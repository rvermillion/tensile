#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

from ..infra import RootObject
from ..infra.util import StringBuffer
from ..nn import Module
from ..nn.common import *

from .schedule import LRSchedule, OptimizerSchedule
from .types import (
    GradientHandler, OptimizerStep, OptimizerStepHandler, TrainFunction, Batch,
    OptimizerStartStep, OptimizerEndStep
)


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


class OptimizerConfig(Object):

    __slots__ = ('schedules', 'learning_rate')

    schedules: Annotated[OptimizerSchedules, field(
        doc='The schedules for this optimizer.',
    )]
    learning_rate: Annotated[LRSchedule, field(
        doc='The learning rate schedule for this optimizer.',
        coerce=True,
    )]

    algorithm: ClassVar[str] = 'generic'
    schedule_names: ClassVar[tuple[str, ...]] = ('learning_rate', )
    hyperparameter_names: ClassVar[tuple[str, ...]] = ()

    def _lazy_schedules(self) -> OptimizerSchedules:
        return OptimizerSchedules({name: getattr(self, name) for name in self.schedule_names})

    def inherit(self, spec: dict[str, Any]|None) -> dict[str, Any]:
        if spec is None: spec = {}
        algo = spec.setdefault('kind', self.algorithm)
        if self.algorithm != algo:
            return spec
        for name in self.schedule_names:
            if name not in spec:
                spec[name] = getattr(self, name)
        for name in self.hyperparameter_names:
            if name not in spec:
                spec[name] = getattr(self, name)
        return spec

    def backend_hyperparameters(self, aliases: dict[str, str|None]) -> dict[str, Any]:
        hyper = {}
        for name in self.hyperparameter_names:
            alias = aliases.get(name, name)
            if alias is not None:
                value = getattr(self, name)
                if value is not None:
                    hyper[alias] = value
        return hyper

    def _repr_args(self, **options) -> str:
        buff = StringBuffer()
        for name in self.schedule_names:
            value = getattr(self, name)
            if value is not None:
                buff.append(f'{name}={value!r}, ')
        for name in self.hyperparameter_names:
            value = getattr(self, name)
            if value is not None:
                buff.append(f'{name}={value!r}, ')
        return str(buff)


def filter_tree(arrays: tree.Tree[Array], paths: set[str]) -> tree.Tree[Array]:
    return tree.filter(arrays, include=lambda e: e.path in paths and ten.is_array(e.value))



class OptimizerParamGroup(Object):

    __slots__ = ('optimizer', 'group', 'params', 'param_tree', 'schedules', 'config')

    optimizer: Annotated['Optimizer', field(
        doc='The optimizer this param group belongs to.'
    )]
    group: Annotated[int, field(
        doc='The group index for this optimizer group.'
    )]
    param_tree: Annotated[dict[str, dict[str, str]], Optional[dict[str, dict[str, dict]]]]
    params: Annotated[set[str], field(
        doc='The parameters for this optimizer group.',
        default_factory=set,
    )]
    schedules: Annotated[OptimizerSchedules, field(
        doc='The schedules for this optimizer group.',
    )]
    config: Annotated[OptimizerConfig, field(
        doc='The config for this optimizer param group.',
    )]

    def _coerce_params(self, spec: Any) -> set[str]:
        if spec is None: return set()
        if isinstance(spec, str): return {spec}
        if isinstance(spec, Iterable): return set(spec)
        raise ValueError(f'Invalid parameter spec: {spec}')

    def _coerce_config(self, spec: Any) -> OptimizerConfig:
        if isinstance(spec, OptimizerConfig): return spec
        default_config = self.optimizer.config
        if default_config is None:
            if spec is None: raise ValueError('No default config specified')
        else:
            spec = default_config.inherit(spec)
        return coerce(OptimizerConfig, spec)

    def _lazy_schedules(self) -> OptimizerSchedules:
        return self.config.schedules

    def filter_tree(self, arrays: tree.Tree[Array]) -> tree.Tree[Array]:
        return filter_tree(arrays, self.params)

    def filter_params(self, arrays: tree.Tree[Array]) -> list[Array]:
        group_paths = self.params
        params = []
        def include(e: tree.TreeEntry):
            return e.path in group_paths and ten.is_array(e.value)

        def add_to_params(e: tree.TreeEntry):
            params.append(e.value)

        tree.apply(arrays, add_to_params, include=include)
        return params

    def current_schedule(self, step: Array, include_constant: bool = False) -> dict[str, float]:
        return self.schedules.get(step, include_constant)

    def get_hyperparameters(self, step: Array) -> dict[str, Any]:
        if schedule := self.current_schedule(step, include_constant=True):
            hyper = {n: s for n, s in schedule.items()}
        else:
            hyper = {}
        return hyper

    def _repr_args(self, **options) -> str:
        return f'{self.group}: {len(self.params)} params'


class Optimizer(Object, Generic[Batch]):

    __slots__ = ('model', 'params', 'config', 'all_params', 'learning_rate', 'eps', 'current_step', 'param_groups',
                 'step_handler', 'backends')

    model: Annotated[Module, field(
        doc='The model to optimize.'
    )]
    params: Annotated[Optional[set[str]], field(
        doc='The parameters for this optimizer.',
    )]
    all_params: Annotated[set[str], field(
        doc='All parameters for this optimizer.',
    )]
    config: Annotated[Optional[OptimizerConfig], field(
        doc='The config for this optimizer.',
        coerce=True,
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

    algorithm: ClassVar[str]
    schedule_names: ClassVar[tuple[str, ...]] = ('learning_rate', )
    hyperparameter_names: ClassVar[tuple[str, ...]] = ()
    hyperparameter_aliases: ClassVar[dict[str, str]] = {}

    def _coerce_param_groups(self, spec: Any) -> Optional[list[OptimizerParamGroup]]:
        if spec is None:
            return []
        elif isinstance(spec, Iterable):
            param_group_specs = list(spec)
            all_params = set()
            rest_group = None
            for g, pg_spec in enumerate(param_group_specs):
                if not isinstance(pg_spec, dict):
                    raise ValueError(f'Invalid parameter group spec: {pg_spec}')
                pg_spec = param_group_specs[g] = pg_spec.copy()
                pg_spec['optimizer'] = self
                pg_spec['group'] = g
                params = pg_spec.get('params')
                if params is None:
                    if rest_group is None:
                        rest_group = pg_spec
                    else:
                        raise ValueError(f'No parameters specified for group {g}')
                else:
                    all_params.update(params)
            if rest_group is not None:
                if self.params is None:
                    params = set(path for path, v in tree.flatten(self.model.trainable_parameters())) - all_params
                else:
                    params = set(self.params) - all_params
                rest_group['params'] = params

            return [OptimizerParamGroup(pg_spec) for pg_spec in param_group_specs]
        else:
            raise ValueError(f'Invalid parameter group spec: {spec}')

    def _lazy_all_params(self) -> set[str]:
        all_params = set()
        for pg in self.param_groups:
            all_params.update(pg.params)
        return all_params

    def postinit(self, spec: Spec):
        super().postinit(spec)
        if self.params is None:
            params = set(n for n, p in tree.flatten(self.model.trainable_parameters()))
        else:
            params = self.params
        if not self.param_groups:
            self.param_groups.append(OptimizerParamGroup(
                group=0,
                params=params,
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
            return OptimizerParamGroup({
                'optimizer': self,
                'group': group,
                'algorithm': self.algorithm,
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

    def trainable_parameters(self, model: Module = None, group: int = None) -> Tree[Array]:
        if model is None: model = self.model

        if group is None:
            params = filter_tree(model, self.all_params)
        else:
            params = self.param_groups[group].filter_tree(model)
        ten.debug_eval(params)
        return params

    def start_step(self, batch: Batch) -> None:
        if step_handler := self.step_handler:
            step_handler.on_start(self, batch)

    def get_hyperparameters(self, group: int = None) -> dict[str, Any]:
        step = self.current_step
        if group is None:
            hyper = {}
            for g, param_group in enumerate(self.param_groups):
                hyper[str(g)] = param_group.get_hyperparameters(step)
        else:
            return self.param_groups[group].get_hyperparameters(step)
        return hyper

    def finish_step(self, loss: Array, batch: Batch) -> None:
        if step_handler := self.step_handler:
            step_handler.on_end(self, loss, batch)
        self.current_step += 1

    def default_schedules(self) -> OptimizerSchedules:
        return OptimizerSchedules({name: getattr(self, name) for name in self.schedule_names})

    def current_schedule(self, param_group: int = 0) -> dict[str, float]:
        return self.param_groups[param_group].schedules.get(self.current_step, aliases=self.hyperparameter_aliases)

    def stepper(self, train_fn: TrainFunction[Batch], *,
                grad_handlers: Sequence[GradientHandler] = None,
                start_step: OptimizerStartStep = None,
                end_step: OptimizerEndStep = None,
                **kwargs,
                ) -> OptimizerStep[Batch]:
        raise NotImplementedError()

    def alias_hyperparameter(self, name: str) -> str:
        return self.hyperparameter_aliases.get(name, name)

    def backend_schedules(self, current: bool = True) -> dict[str, Any]:
        hyper = {}
        if current:
            step = self.current_step
            for name in self.schedule_names:
                schedule = getattr(self, name)
                if schedule is not None:
                    alias = self.alias_hyperparameter(name)
                    hyper[alias] = schedule(step)
        else:
            for name in self.schedule_names:
                schedule = getattr(self, name)
                if schedule is not None:
                    alias = self.alias_hyperparameter(name)
                    hyper[alias] = schedule
        return hyper

    def backend_hyperparameters(self) -> dict[str, Any]:
        hyper = {}
        for name in self.hyperparameter_names:
            value = getattr(self, name)
            if value is not None:
                hyper[name] = value
        return hyper

    @classmethod
    def _coerce_from_mapping(cls, spec: Mapping[str, Any], /, **kwargs):
        if 'kind' not in spec and 'kind' not in kwargs:
            kwargs['kind'] = ten.ten_kind
        return super()._coerce_from_mapping(spec, **kwargs)


@provides(OptimizerConfig, 'sgd')
class SGDConfig(OptimizerConfig):

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

    def _coerce_weight_decay(self, spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    algorithm: ClassVar[str] = 'sgd'
    schedule_names = (*Optimizer.schedule_names, 'momentum', 'weight_decay', 'dampening')
    hyperparameter_names = ('nesterov',)


@provides(OptimizerConfig, 'adamw')
class AdamWConfig(OptimizerConfig):

    __slots__ = ('weight_decay', 'betas', 'eps',)

    weight_decay: Annotated[Optional[OptimizerSchedule], field(
        doc='The weight decay coefficient for AdamW.'
    )]
    eps: Annotated[float, field(
        doc='The epsilon value for this optimizer.',
        default=1e-6,
    )]
    betas: Annotated[Optional[tuple[float, float]], field(
        doc='The betas for AdamW.'
    )]

    def _coerce_weight_decay(self, spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    def _coerce_betas(self, betas: Any):
        if betas is None: return None
        if isinstance(betas, tuple): return betas
        if isinstance(betas, list): return tuple(betas[:2])
        raise ValueError(f'Invalid betas: {betas}')

    algorithm: ClassVar[str] = 'adamw'
    schedule_names = (*OptimizerConfig.schedule_names, 'weight_decay')
    hyperparameter_names = ('eps', 'betas')



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

