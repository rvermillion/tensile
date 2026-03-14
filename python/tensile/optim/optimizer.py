#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

"""Optimizer abstractions and configuration helpers.

This module defines the core optimizer-facing data structures used by the
training stack:

- :class:`BasicStepHandler` for start/end step callbacks
- :class:`OptimizerSchedules` for named schedule management
- :class:`OptimizerConfig` and concrete config types for algorithm settings
- :class:`OptimizerParamGroup` for grouping parameters under shared schedules
- :class:`Optimizer` as the abstract optimizer interface

The design separates optimizer configuration from backend implementation so
multiple execution backends can reuse the same scheduling and parameter-group
logic.
"""
from collections.abc import Container
from pathlib import Path

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
    """Simple callback container invoked at the start and end of each optimizer step.

    The handler is intentionally lightweight: it just stores two callables and
    lets :class:`Optimizer` invoke them during step execution.

    Attributes:
        on_start: Called before gradients are computed for a batch.
        on_end: Called after the loss has been computed and the step is about
            to finish.
    """

    __slots__ = ('on_start', 'on_end')

    on_start: Callable[['Optimizer', Batch], None]
    on_end: Callable[['Optimizer', Array, Batch], None]

    def __init__(self,
                 on_start: Callable[['Optimizer', Batch], None],
                 on_end: Callable[['Optimizer', Array, Batch], None]):
        self.on_start = on_start
        self.on_end = on_end


class OptimizerSchedules(RootObject):
    """Container for named optimizer schedules.

    The object behaves like a small registry for per-optimizer schedule objects
    such as learning rate, momentum, or weight decay schedules. It can return
    either the schedules themselves or their current scalar values for a given
    step.
    """

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
        """Return schedules keyed by aliased names.

        Args:
            include_constant: Whether to include schedules marked as constant.
            aliases: Optional mapping from internal schedule names to backend or
                display names.

        Returns:
            A dictionary of schedule objects keyed by aliased names.
        """
        if aliases is None:
            def alias(name: str) -> str: return name
        else:
            def alias(name: str) -> str: return aliases.get(name, name)
        if include_constant:
            return {alias(k): v for k, v in self.schedules.items()}
        return {alias(k): v for k, v in self.schedules.items() if not v.constant}

    def get(self, step: Array, include_constant: bool = False, aliases: dict[str, str] = None) -> dict[str, float]:
        """Evaluate schedules at a given step.

        Args:
            step: Current optimizer step.
            include_constant: Whether to include constant schedules.
            aliases: Optional mapping from internal names to alternate keys.

        Returns:
            A mapping from schedule name to evaluated scalar value.
        """
        if aliases is None:
            def alias(name: str) -> str: return name
        else:
            def alias(name: str) -> str: return aliases.get(name, name)
        if include_constant:
            return {alias(k): v(step).item() for k, v in self.schedules.items()}
        return {alias(k): v(step).item() for k, v in self.schedules.items() if not v.constant}

    def get_static(self) -> dict[str, Array]:
        """Return schedules whose values are constant across all steps."""
        step = ten.array(0, dtype=ten.int32)
        return {k: v(step) for k, v in self.schedules.items() if v.constant}


class OptimizerConfig(Object):
    """Base configuration object for optimizers.

    Subclasses declare algorithm-specific schedules and hyperparameters. The
    config is responsible for exposing values in a backend-friendly form while
    still preserving higher-level schedule objects.
    """

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
        """Build the schedule registry from declared schedule fields."""
        return OptimizerSchedules({name: getattr(self, name) for name in self.schedule_names})

    def inherit(self, spec: dict[str, Any]|None) -> dict[str, Any]:
        """Fill missing values in a config spec from this config.

        This is primarily used when parameter groups partially override a
        default optimizer configuration.

        Args:
            spec: Partial config specification.

        Returns:
            A new or updated specification containing inherited defaults. If the
            requested algorithm differs from this config's algorithm, the spec is
            returned unchanged.
        """
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
        """Return non-scheduled hyperparameters, applying backend aliases.

        Any alias mapped to ``None`` is omitted from the result.
        """
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


def filter_tree(arrays: tree.Tree[Array], paths: Container[str]) -> tree.Tree[Array]:
    """Filter a parameter or gradient tree down to a set of named paths."""
    return tree.filter(arrays, include=lambda e: e.path in paths and ten.is_array(e.value))



class OptimizerParamGroup(Object):
    """A named subset of optimizer parameters sharing one configuration.

    Parameter groups allow different subsets of model parameters to use distinct
    schedules or hyperparameters while still being managed by a single
    optimizer object.
    """

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
        """Normalize parameter specifications into a set of parameter paths."""
        if spec is None: return set()
        if isinstance(spec, str): return {spec}
        if isinstance(spec, Iterable): return set(spec)
        raise ValueError(f'Invalid parameter spec: {spec}')

    def _coerce_config(self, spec: Any) -> OptimizerConfig:
        """Coerce a group config, inheriting defaults from the parent optimizer."""
        if isinstance(spec, OptimizerConfig): return spec
        default_config = self.optimizer.config
        if default_config is None:
            if spec is None: raise ValueError('No default config specified')
        else:
            spec = default_config.inherit(spec)
        return coerce(OptimizerConfig, spec)

    def _lazy_schedules(self) -> OptimizerSchedules:
        """Expose the schedule registry from the resolved config."""
        return self.config.schedules

    def filter_tree(self, arrays: tree.Tree[Array]) -> tree.Tree[Array]:
        """Return only entries belonging to this parameter group."""
        return filter_tree(arrays, self.params)

    def filter_params(self, arrays: tree.Tree[Array]) -> list[Array]:
        """Extract this group's arrays from a tree as a flat list."""
        group_paths = self.params
        params = []
        def include(e: tree.TreeEntry):
            return e.path in group_paths and ten.is_array(e.value)

        def add_to_params(e: tree.TreeEntry):
            params.append(e.value)

        tree.apply(arrays, add_to_params, include=include)
        return params

    def current_schedule(self, step: Array, include_constant: bool = False) -> dict[str, float]:
        """Evaluate this group's schedules for the given step."""
        return self.schedules.get(step, include_constant)

    def get_hyperparameters(self, step: Array) -> dict[str, Any]:
        """Return current schedule values as backend-ready hyperparameters."""
        if schedule := self.current_schedule(step, include_constant=True):
            hyper = {n: s for n, s in schedule.items()}
        else:
            hyper = {}
        return hyper

    def _repr_args(self, **options) -> str:
        return f'{self.group}: {len(self.params)} params'


class OptimizerParamInfo(RootObject):
    """Information about a single optimizer parameter."""

    __slots__ = ('path', 'group', 'state')

    path: Annotated[str, field()]
    group: Annotated[OptimizerParamGroup, field()]
    state: Annotated[dict[str, Any], field()]

    def __init__(self, path: str, group: OptimizerParamGroup, state: dict[str, Any]):
        super().__init__()
        self.path = path
        self.group = group
        self.state = state

    def _repr_args(self, **options) -> str:
        return self.path + f', group={self.group.group}'


class Optimizer(Object, Generic[Batch]):
    """Abstract optimizer wrapper shared across tensor backends.

    This class owns parameter selection, grouping, scheduling, and lifecycle
    hooks, while backend-specific subclasses implement the actual stepping
    logic in :meth:`stepper`.
    """

    __slots__ = ('model', 'params', 'config', 'all_params', 'learning_rate', 'eps', 'current_step', 'param_groups',
                 'step_handler', 'backends')

    model: Annotated[Module, field(
        doc='The model to optimize.'
    )]
    params: Annotated[Optional[set[str]], field(
        doc='The parameters for this optimizer.',
    )]
    all_params: Annotated[dict[str, OptimizerParamInfo], field(
        doc='All parameters for this optimizer.',
    )]
    config: Annotated[Optional[OptimizerConfig], field(
        doc='The config for this optimizer.',
        coerce=True,
    )]
    current_step: Annotated[Array, field(
        doc='The current step of the optimizer.',
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

    def _lazy_all_params(self) -> dict[str, OptimizerParamInfo]:
        all_params = {}
        for param_group in self.param_groups:
            for name in param_group.params:
                all_params[name] = self.build_param_info(name, param_group)
        return all_params

    _step_dtype: ClassVar[Annotated[ten.DType, field(ignore=True)]] = ten.uint64

    def _lazy_current_step(self) -> Array:
        return ten.array(0, dtype=self._step_dtype)

    def _coerce_current_step(self, spec: Any) -> Array:
        if spec is None:
            return ten.array(0, dtype=self._step_dtype)
        elif isinstance(spec, Array):
            return ten.as_type(spec, self._step_dtype)
        elif isinstance(spec, int):
            return ten.array(spec, dtype=self._step_dtype)
        else:
            raise TypeError(f'Invalid current_step specification: {spec}')

    def set_current_step(self, step: Array) -> None:
        self.current_step = step

    def build_param_info(self, path: str, group: OptimizerParamGroup) -> OptimizerParamInfo:
        return OptimizerParamInfo(path, group, {})

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
                config=self.config,
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

    def load(self, path: str|Path, **kwargs) -> None:
        """Load the optimizer state from a file."""
        if isinstance(path, str): path = Path(path)
        return self._load(path, **kwargs)

    def _load(self, path: Path, **kwargs) -> None:
        """Load the optimizer state from a file."""
        raise NotImplementedError()

    def save(self, path: str|Path, **kwargs) -> None:
        """Save the optimizer state to a file."""
        if isinstance(path, str): path = Path(path)
        return self._save(path, **kwargs)

    def _save(self, path: str|Path, **kwargs) -> None:
        """Save the optimizer state to a file."""
        raise NotImplementedError()

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
        else:
            kind = kwargs.get('kind', spec.get('kind'))
            if kind == 'native':
                kwargs['kind'] = 'native.' + ten.ten_kind
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



@provides(OptimizerConfig, 'adam')
class AdamConfig(OptimizerConfig):

    __slots__ = ('betas', 'eps', 'bias_correction')

    eps: Annotated[float, field(
        doc='The epsilon value for this optimizer.',
        default=1e-8,
    )]
    betas: Annotated[Optional[tuple[float, float]], field(
        doc='The betas for AdamW.',
        default=(0.9, 0.999),
    )]
    bias_correction: Annotated[bool, field(
        doc='Whether to use bias correction for Adam.',
        default=False,
    )]

    def _coerce_betas(self, betas: Any):
        if betas is None: return None
        if isinstance(betas, tuple): return betas
        if isinstance(betas, list): return tuple(betas[:2])
        raise ValueError(f'Invalid betas: {betas}')

    algorithm: ClassVar[str] = 'adam'
    hyperparameter_names = ('eps', 'betas', 'bias_correction')


@provides(OptimizerConfig, 'adamw')
class AdamWConfig(AdamConfig):

    __slots__ = ('weight_decay',)

    weight_decay: Annotated[Optional[OptimizerSchedule], field(
        doc='The weight decay coefficient for AdamW.'
    )]

    def _coerce_weight_decay(self, spec: Any) -> Optional[OptimizerSchedule]:
        if spec is None: return None
        return coerce(OptimizerSchedule, spec)

    algorithm: ClassVar[str] = 'adamw'
    schedule_names = (*AdamConfig.schedule_names, 'weight_decay')



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

