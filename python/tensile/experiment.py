#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from collections import defaultdict

import json

import time
from pathlib import Path

import yaml

from .common import *
from .nn import Module
from .util import plot
from .util.load import load
from .util.metric import Metric, ArrayMetric
from .util.state import StateAware


Params = dict[str, Any]
ReadOnlyParams = Mapping[str, Any]


def update_params(orig: Params, new: Params, inplace: bool = False) -> Params:
    params = orig if inplace else orig.copy()
    params.update(new)
    return params


def default_params(orig: Params, new: Params, inplace: bool = False) -> Params:
    params = orig if inplace else orig.copy()
    for p, v in new.items():
        if p not in params:
            params[p] = v
    return params


def coerce_params(spec: Any, params: Params = None) -> Params:
    if params is None: params = {}
    if isinstance(spec, Mapping):
        params.update(spec)
    return params


class Experiment(StateAware):

    __slots__ = ('name', 'qname', 'descriptor', 'parent',
                 'params', 'all_params', 'unflat_params', 'arch',
                 'experiments', 'metrics', 'seed', 'start_time', 'end_time',
                 'work_dir', 'path', 'model_dir', 'output')

    name: Annotated[str, field(
        doc='The name of the experiment'
    )]
    qname: Annotated[str, field(
        doc='The qualified name of the experiment'
    )]
    descriptor: Annotated[str, field(
        doc='The descriptor of the experiment'
    )]
    parent: Annotated[Optional['Experiment'], field(
        doc='The parent experiment, if any'
    )]
    params: Annotated[Params, field(
        doc='The parameters for this experiment',
        default_factory=dict,
    )]
    all_params: Annotated[Params, field(
        doc='The parameters for this experiment, including inherited parameters',
    )]
    unflat_params: Annotated[Params, field(
        doc='The parameters for this experiment',
    )]
    experiments: Annotated[list['Experiment'], field(
        doc='The list of sub-experiments for this experiment, if any',
        default_factory=list,
    )]
    metrics: Annotated[dict[str, Metric], field(
        default_factory=dict
    )]
    arch: Annotated[str, field(
        doc='The model architecture',
        default='',
    )]
    seed: Annotated[int, field(
        doc='The random seed to set before running the bucket',
    )]
    output: Annotated[list[str], field(
        doc='The list of lines output by this experiment',
        default_factory=list,
    )]
    start_time: Annotated[Optional[float], field(
        doc='The timestamp when the experiment started',
    )]
    end_time: Annotated[Optional[float], field(
        doc='The timestamp when the experiment ended',
    )]
    work_dir: Annotated[Path, field(
        doc='The path to the work directory',
    )]
    path: Annotated[Path, field(
        doc='The path to the experiment directory',
    )]
    model_dir: Annotated[Path, field(
        doc='The directory to store models in',
    )]

    def _coerce_model_dir(self, spec: Any) -> Path:
        return self.work_dir / spec

    def _lazy_model_dir(self) -> Path:
        return self.work_dir / f'models/{self.name}'

    def _lazy_work_dir(self) -> Path:
        return Path('./work').absolute()

    def _lazy_path(self) -> Path:
        if parent := self.parent:
            parent_dir = parent.path
        else:
            parent_dir = self.work_dir

        path = parent_dir / self.name
        # if not work_dir.exists():
        #     work_dir.mkdir(parents=True)
        return path

    def _lazy_all_params(self) -> Params:
        params = {}
        if parent := self.parent:
            params.update(parent.all_params)

        params.update(self.params)

        if defaults := self.fixed_param_defaults():
            for name, value in defaults.items():
                params.setdefault(name, value)
        return params

    def _coerce_params(self, spec: Any) -> Params:
        params = {}

        if spec is None: return params

        if isinstance(spec, str):
            spec = Path(spec)
        if isinstance(spec, Path):
            spec = load(spec)
        if isinstance(spec, Mapping):
            spec = tree.flatdict(spec)
            params.update(spec)
        else:
            raise ValueError(f"Invalid parameter specification: {spec}")

        return params

    def _lazy_unflat_params(self) -> Params:
        return tree.unflatdict(self.all_params)

    def _lazy_descriptor(self) -> str:
        desc_params = self.descriptor_params
        if desc_params:
            return self.name + '(' + ', '.join(f'{k}={v}' for k, v in desc_params.items()) + ')'
        return self.name

    def _lazy_qname(self) -> str:
        if parent := self.parent:
            return parent.qname + '/' + self.name
        return self.name

    def _lazy_seed(self) -> int:
        seed = self.params.get('seed', 0)
        if not seed:
            if parent := self.parent:
                seed = parent.seed
        return seed

    @property
    def descriptor_params(self) -> dict[str, Any]:
        params = self.params
        bucket_keys = params.keys()
        abbrevs = self.hyperparam_abbrevs
        return {abbrevs.get(pk, pk): params[pk] for pk in bucket_keys}

    @property
    def elapsed_time(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    # def set_param(self, name: str, value: Any) -> None:
    #     old_value = self.params.get(name)
    #     if old_value != value:
    #         self.params[name] = value
    #         self.param_changed(name, value, old_value)
    #
    # def param_changed(self, name: str, value: Any, old_value: Any) -> None:
    #     pass
    #
    # def get_param_defaults_with_prefix(self, prefix: str) -> Params:
    #     if parent := self.parent:
    #         params = parent.get_param_defaults_with_prefix(prefix)
    #     else:
    #         params = {}
    #     plen = len(prefix)
    #     for name, value in self.param_defaults.items():
    #         if name.startswith(prefix):
    #             params[name[plen:]] = value
    #     return params

    # def all_params(self, flat: bool = None) -> Params:
    #     params = self.get_params_with_prefix('')
    #     if flat is None:
    #         return params
    #     params = tree.flatdict(params)
    #     if not flat:
    #         params = tree.unflatten(params.items())
    #         if not isinstance(params, dict):
    #             raise ValueError(f"Expected dict, got {type(params).__name__}")
    #     return params

    def get_params_with_prefix(self, prefix: str) -> Params:
        # if parent := self.parent:
        #     params = parent.get_params_with_prefix(prefix)
        # else:
        #     params = {}
        params = {}
        plen = len(prefix)
        for name, value in self.all_params.items():
            if name.startswith(prefix):
                params[name[plen:]] = value
        # if with_defaults:
        #     for name, value in self.get_param_defaults_with_prefix(prefix).items():
        #         params.setdefault(name, value)
        return params

    def get_param(self, name: str, default: Any = None) -> Any:
        if params := self.all_params:
            if name in params:
                return params[name]
        # if default is None:
        #     default = self.param_defaults.get(name)
        # if parent := self.parent:
        #     return parent.get_param(name, default)
        return default

    def collect_params(self, name: str, params: dict[str, Any]) -> None:
        raise NotImplementedError()
        # if defaults := self.param_defaults.get(name):
        #     if isinstance(defaults, dict):
        #         for k, v in defaults.items():
        #             params.setdefault(k, v)
        if own := self.params.get(name):
            if isinstance(own, dict):
                params.update(own)

    def get_params(self, name: str) -> dict[str, Any]:
        raise NotImplementedError()
        params = {}
        if parent := self.parent:
            parent.collect_params(name, params)
        self.collect_params(name, params)
        return params

    def count_experiments(self, depth: int = 1) -> int:
        if depth == 0:
            return 1
        elif depth == 1:
            return len(self.experiments)
        elif depth > 1:
            count = 0
            for experiment in self.experiments:
                count += experiment.count_experiments(depth - 1)
            return count
        return 0

    def add_experiment(self, **spec) -> 'Experiment':
        experiment = Experiment.coerce(parent=self, **spec)
        self.experiments.append(experiment)
        return experiment

    def add_metric(self, metric: Metric, name: str = None):
        if name is None:
            name = metric.name
        if isinstance(metric, Metric):
            self.metrics[name] = metric
        else:
            raise ValueError(f"Invalid metric type for '{name}': {type(metric)}")
        if parent := self.parent:
            parent.add_metric(metric, self.name + '/' + name)

    def get_metric(self, name: str) -> Optional[Metric]:
        return self.metrics.get(name)

    def batch_data(self, b: int) -> Iterable[tuple[Array, Array]]:
        if parent := self.parent:
            return parent.batch_data(b)
        raise NotImplementedError()

    def get_path(self, rel: str, write: bool = False) -> Path:
        path = self.path / rel
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_module(self, spec: Any, name: str = None, save: bool|Path = True) -> Module:
        if spec is None:
            raise ValueError('module spec cannot be None')
        if isinstance(spec, Module):
            module = spec
        elif isinstance(spec, dict):
            module = Module.from_args(**spec)
        else:
            raise ValueError(f'unknown module spec type: {type(spec)}')
        if isinstance(save, Path):
            path = save
        elif name is not None:
            path = (self.model_dir / name).with_suffix('.safetensors')
        else:
            path = None

        if path is not None:
            if path.exists():
                module.load_weights(path)
            elif parent := self.parent:
                return parent.get_module(module, name, save=save)

            if save:
                path.parent.mkdir(parents=True, exist_ok=True)
                ten.save_tensors(path, tree.flatdict(module.parameters()))
        return module

    def run(self):
        self.start()
        self.run_self()
        self.run_experiments()
        self.finish()

    def start(self):
        self.start_time = time.time()
        self.metrics.clear()
        self.header(f'Experiment[{self.descriptor}] Starting')

    def run_self(self):
        if self.seed:
            ten.random.seed(self.seed)

    def run_experiments(self):
        for exp in self.experiments:
            if self.before_child_start(exp):
                exp.run()
                self.after_child_finish(exp)

    def before_child_start(self, child: 'Experiment') -> bool:
        return True

    def after_child_finish(self, child: 'Experiment') -> None:
        pass

    def plot_charts(self, metrics: dict[str, Metric] = None, *,
                    prefix: str = '', suffix: str = '',
                    # grid: dict[str, dict[str, Array]] = None,
                    with_grid: bool = True):
        if metrics is None: metrics = self.metrics
        if metrics:
            # if charts := self.get_param('charts'):
            #     for name, spec in charts.items():
            #         pass
            #     return
            array_metrics: dict[str, Array] = {}

            metrics_by_name = defaultdict(dict[str, Metric])

            ylim = dict(top=4., bottom=0.)

            for name, metric in metrics.items():
                if isinstance(metric, ArrayMetric):
                    array_metrics[name] = metric.array
                metrics_by_name[metric.name][name] = metric

            if len(metrics) <= 8:
                chart = plot.Chart.from_metrics(metrics, smoothing=0.9, ylim=ylim)

                chart.plot(out=self.get_path(f"{prefix}metrics{suffix}.png", write=True))

            if with_grid:

                d = 1
                rows = self.count_experiments(d)
                while rows == 1:
                    d += 1
                    rows = self.count_experiments(d)
                    if rows == 0:
                        rows = 1
                        break

                chart = plot.Chart.split_grid(metrics, smoothing=0.9, rows=rows, ylim=ylim)
                chart.plot(out=self.get_path(f"{prefix}grid{suffix}.png", write=True))

                for name, grid_metrics in metrics_by_name.items():
                    chart = plot.Chart.split_grid(grid_metrics, title=name, smoothing=0.9, rows=rows, ylim=ylim)
                    chart.plot(out=self.get_path(f"{prefix}grid-{name}{suffix}.png", write=True))

    def load_metrics(self, path: Path = None) -> dict[str, Metric]:
        if path is None: path = self.get_path(f"metrics.safetensors")
        if path.exists():
            metrics = {}
            arrays = ten.load_tensors(path)
            if arrays:
                for name, array in arrays.items():
                    if name.endswith('.steps'):
                        name = name[:-6]
                        steps = array.tolist()
                        values = arrays[name + '.values'].tolist()
                        if isinstance(steps, list) and isinstance(values, list):
                            metrics[name] = Metric.from_steps(name, steps, values)
                        else:
                            raise ValueError(f"Invalid metric format for {name}: steps={type(steps)}, values={type(values)}")
                    elif not name.endswith('.values'):
                        metrics[name] = Metric.from_array(name, array)
            return metrics
        else:
            raise ValueError('No metrics found at: ' + str(path))

    def save_metrics(self, metrics: dict[str, Metric] = None):
        if metrics is None: metrics = self.metrics
        if metrics:
            tensors: dict[str, Array] = {}
            for name, metric in metrics.items():
                metric.add_tensors(tensors, name)

            if tensors:
                ten.save_tensors(self.get_path(f"metrics.safetensors", write=True), tensors)

    def finish(self):
        self.end_time = time.time()
        prefix = ''  # self.qname + '-'

        self.plot_charts()
        self.save_metrics()

        self.metrics.clear()

        self.header(f'Experiment[{self.descriptor}] Finished')
        if output := self.output:

            with open(self.get_path(f"{prefix}output.txt", write=True), 'w') as f:
                f.write('\n'.join(output))

            self.output.clear()

        state = self.state_dict()
        with open(self.get_path(f"{prefix}experiment.json", write=True), 'w') as f:
            json.dump(state, f, indent=2)
        with open(self.get_path(f"{prefix}experiment.yaml", write=True), 'w') as f:
            yaml.dump(state, f, indent=2)

    def fixed_param_defaults(self) -> Params:
        return {}

    def _add_state(self, state: dict[str, Any]):
        state['name'] = self.name
        if kind := Experiment.meta.registry.get_kind(self.__class__):
            state['kind'] = kind
        else:
            state['kind'] = self.meta.qname
        if parent := self.parent:
            state['parent'] = parent.qname
        state.update(
            descriptor=self.descriptor,
            seed=self.seed,
            params=self.unflat_params,
            path=str(self.path),
        )
        if metrics := self.metrics:
            state['metrics'] = list(metrics)
        if experiments := self.experiments:
            state['experiments'] = [exp.qname for exp in experiments]
        if start_time := self.start_time:
            state['start_time'] = start_time
            if end_time := self.end_time:
                state['end_time'] = end_time
                state['elapsed_time'] = end_time - start_time
        return state

    def write_prefix(self) -> str:
        return self.name

    def write_output(self, msg: str, *, echo: bool = True, prefix: str = None, passthru: bool = True):
        prefixed_msg = msg if prefix is None else prefix + ': ' + msg
        self.output.append(prefixed_msg)
        if passthru:
            if parent := self.parent:
                prefix = self.write_prefix() if prefix is None else self.write_prefix() + '/' + prefix
                parent.write_output(msg, echo=echo, prefix=prefix)
                return
        if echo: print(prefixed_msg)

    def header(self, title: str, width: int = 120):
        self.write_output('')
        width -= len(title)
        if width < 0:
            self.write_output(title)
        else:
            p = width // 2
            s = p+1 if width % 2 else p
            self.write_output(" ".join(("=" * p, title, "=" * s)))

    def print(self, *args):
        self.write_output(" ".join(map(str, args)))

    def _repr_args(self, **options) -> str:
        return self.qname

    hyperparam_abbrevs: ClassVar[dict[str, str]] = {
        'learning_rate': 'lr',
        'learning_rate_decay': 'lrd',
        'seed': 's',
        'lr_decay': 'lrd',
    }

class CachedInputExperiment(Experiment):

    __slots__ = ('in_chunk_size', 'in_dim', 'chunks_per_epoch', 'input_dir',
                 'time_per_epoch')

    in_chunk_size: Annotated[int, field(
        doc='The size of each input chunk',
        default=1024,
    )]
    in_dim: Annotated[int, field(
        doc='The dimensionality of each input vector',
    )]
    chunks_per_epoch: Annotated[int, field(
        doc='The number of chunks to generate per epoch',
        default=10,
    )]
    time_per_epoch: Annotated[float, field(
        doc='The amount time to generate chunks for an epoch',
        default=0.0,
    )]
    input_dir: Annotated[Path, field(
        doc='The directory to store input chunks',
    )]

    def _coerce_input_dir(self, spec: Any) -> Path:
        return self.work_dir / spec

    def _lazy_input_dir(self) -> Path:
        return self.work_dir / f'inputs/{self.name}'

    def get_module(self, spec: Any, name: str = None, save: bool | Path = True) -> Module:
        if save is True and name is not None:
            save = (self.model_dir/ name).with_suffix('.safetensors')
        return super().get_module(spec, name, save)

    def generate_input_chunk(self) -> Array:
        return ten.random.normal(scale=2., shape=(self.in_chunk_size, self.in_dim))

    def get_input_chunk(self, i: int, name: str = 'inputs', key: str = 'input') -> Array:
        input_file = self.input_dir / f"{name}-{i}.safetensors"
        in_dim = self.in_dim
        if input_file.exists():
            arrays = ten.load_tensors(input_file)
        else:
            arrays = {
                key: self.generate_input_chunk()
            }
            input_file.parent.mkdir(parents=True, exist_ok=True)
            ten.save_tensors(input_file, arrays)
        ten.eval(arrays)
        inputs = arrays[key]
        if inputs.shape[-1] != in_dim:
             raise ValueError(f'Expected input chunk to have shape (..., {in_dim}), got: {inputs.shape}')
        return inputs

    def iter_batch(self, b: int) -> Iterable[Array]:
        chunk_size = self.in_chunk_size
        time_per_epoch = self.time_per_epoch
        if time_per_epoch > 0.0:
            i = 0
            end_time = time.time() + time_per_epoch
            while True:
                chunk = self.get_input_chunk(i)
                for s in range(0, chunk_size, b):
                    if time.time() > end_time:
                        return
                    yield chunk[s:s+b]
                i += 1
        else:
            for i in range(self.chunks_per_epoch):
                chunk = self.get_input_chunk(i)
                for s in range(0, chunk_size, b):
                    yield chunk[s:s+b]


@provides(Experiment, 'teacher')
class TeacherExperiment(CachedInputExperiment):

    __slots__ = ('teacher',)

    teacher: Annotated[Module, field(
        doc='The teacher model to use',
    )]

    def _coerce_teacher(self, spec: Any) -> Module:
        return self.get_module(spec, 'teacher')

    def batch_data(self, b: int) -> Iterable[tuple[Array, Array]]:
        teacher = self.teacher
        for inputs in self.iter_batch(b):
            targets = teacher(inputs)
            yield inputs, targets

    def start(self):
        super().start()

        self.header('teacher')
        self.print(self.teacher.structure())


# class ExperimentBucket(Object):
#
#     __slots__ = ('experiment', 'name', 'descriptor', 'params', 'param_defaults', 'work_dir', 'report_every', 'seed')
#
#     name: Annotated[str, field(
#         doc='The name of the experiment'
#     )]
#     descriptor: Annotated[str, field(
#         doc='The descriptor of the experiment'
#     )]
#     experiment: Annotated[Experiment, field(
#         doc='The experiment this bucket belongs to'
#     )]
#     params: Annotated[dict[str, Any], field(
#         default_factory=dict
#     )]
#
#     param_defaults: Annotated[dict[str, Any], field(
#         default_factory=dict
#     )]
#
#     work_dir: Annotated[Path, field()]
#     report_every: Annotated[int, field(
#         doc='The number of steps between each report',
#         default=100,
#     )]
#     seed: Annotated[Optional[int], field(
#         doc='The random seed to set before running the bucket',
#     )]
#
#     def _lazy_work_dir(self) -> Path:
#         return self.experiment.work_dir
#
#     def _lazy_descriptor(self) -> str:
#         desc_params = self.descriptor_params
#         if desc_params:
#             return self.name + '-' + '-'.join(f'{k}={v}' for k, v in desc_params.items())
#         return self.name
#
#     @property
#     def descriptor_params(self) -> dict[str, Any]:
#         params = self.params
#         bucket_keys = params.keys()
#         defaults = self.param_defaults
#         return {
#             default_hyperparam_abbrevs.get(pk, pk): params[pk]
#             for pk in bucket_keys
#             if pk in params and (pk not in defaults or params[pk] != defaults[pk])
#         }
#
#     def get_param(self, name: str) -> Any:
#         if params := self.params:
#             if name in params:
#                 return params[name]
#         return self.param_defaults.get(name)
#
#     def run(self):
#         if self.seed is not None:
#             ten.random.seed(self.seed)
#
#         start = time.perf_counter()
#         steps = self._run()
#         end = time.perf_counter()
#         ksteps = steps/1000.
#         self.print(f'time elapsed: {(end - start)/ksteps:.4f} seconds per thousand steps')
#
#     def _run(self) -> int:
#         raise NotImplementedError()
#
#     def print(self, *args):
#         print(f'Bucket[{self.descriptor}]:', *args)
#
#     header = staticmethod(Experiment.header)

Sweeps = Iterable[Params]

SweepLike = Union['Sweep', Sequence['SweepLike'], Mapping[str, Any]]


class Sweep(StateAware):

    __slots__ = ()

    def sweep(self, params: Params) -> Iterable[Params]:
        for sweep in self.iter_params():
            yield update_params(params, sweep, inplace=True)

    def iter_params(self) -> Iterable[Params]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(list(self.iter_params()))

    def __add__(self, other) -> 'Sweep':
        if not isinstance(other, Sweep):
            raise TypeError(f"Unsupported operand type for *: 'Sweep' and '{type(other).__name__}'")
        return SequenceSweep(sweeps=[self, other])

    def __mul__(self, other) -> 'Sweep':
        if not isinstance(other, Sweep):
            raise TypeError(f"Unsupported operand type for *: 'Sweep' and '{type(other).__name__}'")
        return ProductSweep(left=self, right=other)

    @classmethod
    def _coerce_from_sequence(cls, spec: Sequence, /, **kwargs) -> 'Sweep':
        return SequenceSweep(sweeps=spec, **kwargs)

    @classmethod
    def all(cls, *sweeps: SweepLike) -> 'Sweep':
        if sweeps:
            if len(sweeps) == 1:
                return cls.coerce(sweeps[0])
            return SequenceSweep(sweeps=sweeps)
        return null_sweep

    @classmethod
    def join(cls, *sweeps: SweepLike) -> 'Sweep':
        if sweeps:
            if len(sweeps) == 1:
                return cls.coerce(sweeps[0])
            return JoinSweep(sweeps=sweeps)
        return null_sweep

    @classmethod
    def unpack(cls, spec: dict[str, Any]) -> 'Sweep':
        sweeps = []
        for param, values in spec.items():
            if not isinstance(param, str):
                raise ValueError(f"Invalid parameter name: {param}")
            if not isinstance(values, Iterable):
                raise ValueError(f"Invalid sweep values for parameter '{param}': {values}")
            sweeps.append(ParamSweep(param=param, values=values))
        if len(sweeps) == 1:
            return sweeps[0]
        if sweeps:
            product = None
            for sweep in sweeps:
                if product is None:
                    product = sweep
                else:
                    product = ProductSweep(left=product, right=sweep)
            # noinspection PyTypeChecker
            return product
        return null_sweep


@provides(Sweep, 'null')
class NullSweep(Sweep):

    __slots__ = ()

    def iter_params(self) -> Iterable[Params]:
        return {},

    def __len__(self) -> int:
        return 0


null_sweep = NullSweep()


@provides(Sweep, 'param')
class ParamSweep(Sweep):

    __slots__ = ('param', 'values')

    param: Annotated[str, field(required=True)]
    values: Annotated[list[Any], field(required=True)]

    def _coerce_values(self, spec: Any) -> list[Any]:
        if isinstance(spec, Iterable):
            return list(spec)
        else:
            raise ValueError(f"Invalid sweep specification: {spec}")

    def iter_params(self) -> Iterable[Params]:
        param = self.param
        for value in self.values:
            yield {param: value}

    def __len__(self) -> int:
        return len(self.values)


@provides(Sweep, 'dict')
class DictSweep(Sweep):

    __slots__ = ('params', )

    params: Annotated[Params, field(required=True)]

    def iter_params(self) -> Iterable[Params]:
        return self.params,

    def __len__(self) -> int:
        return 1


@provides(Sweep, 'sequence')
class SequenceSweep(Sweep):

    __slots__ = ('sweeps',)

    sweeps: Annotated[tuple[Sweep, ...], field(required=True)]

    def _coerce_sweeps(self, spec: Any) -> tuple[Sweep, ...]:
        if isinstance(spec, Iterable):
            return tuple(Sweep.coerce(s) for s in spec)
        else:
            raise ValueError(f"Invalid sweep specification: {spec}")

    def iter_params(self) -> Iterable[Params]:
        for sweep in self.sweeps:
            yield from sweep.iter_params()

    def __len__(self) -> int:
        return sum(len(sweep) for sweep in self.sweeps)


@provides(Sweep, 'join')
class JoinSweep(Sweep):

    __slots__ = ('sweeps',)

    sweeps: Annotated[tuple[Sweep, ...], field(required=True)]

    def _coerce_sweeps(self, spec: Any) -> tuple[Sweep, ...]:
        if isinstance(spec, Iterable):
            return tuple(Sweep.coerce(s) for s in spec)
        else:
            raise ValueError(f"Invalid sweep specification: {spec}")

    def iter_params(self) -> Iterable[Params]:
        for sweeps in zip(*(sweep.iter_params() for sweep in self.sweeps)):
            params = {}
            for s in sweeps:
                yield update_params(params, s, inplace=True)

    def __len__(self) -> int:
        return min(len(sweep) for sweep in self.sweeps)


@provides(Sweep, 'product')
class ProductSweep(Sweep):

    __slots__ = ('left', 'right')

    left: Annotated[Sweep, field(
        required=True,
        coerce=True,
    )]
    right: Annotated[Sweep, field(
        required=True,
        coerce=True,
    )]

    def iter_params(self) -> Iterable[Params]:
        for left in self.left.iter_params():
            for right in self.right.iter_params():
                yield update_params(left, right)

    def __len__(self) -> int:
        return len(self.left) * len(self.right)


@provides(Experiment, 'sweep')
class SweepExperiment(Experiment):

    __slots__ = ('child', 'sweeps')

    child: Annotated[dict[str, Any], field(
        doc='The specification for the child experiment to sweep over',
        required=True,
    )]
    sweeps: Annotated[Sweep, field(
        doc='The parameter sweep to perform',
        required=True,
        coerce=True,
    )]

    def run_self(self):
        super().run_self()
        spec = self.child.copy()
        # orig_params = spec.get('params', {})
        spec['parent'] = self
        # name = spec.get('name', 'exp')
        name = self.name
        i = 0
        for params in self.sweeps.sweep({}):
            spec['name'] = "-".join(f"{k}={v}" for k, v in params.items())
            spec['params'] = params
            child = Experiment.coerce(**spec)
            self.experiments.append(child)
            # child.run()
            i += 1

    # def run_experiments(self):
    #     # They were already run in run_self....
    #     pass

    # def add_metric(self, metric: Metric, name: str = None):
    #     if parent := self.parent:
    #         parent.add_metric(metric, name)
    #     else:
    #         super().add_metric(metric, name)

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)
        state['sweeps'] = self.sweeps.state_dict()


class TrainingExperiment(Experiment):

    __slots__ = ('num_epochs', 'batch_size', 'report_every', 'steps')

    num_epochs: Annotated[int, field(
        doc='The number of epochs to train',
        default=1,
    )]
    batch_size: Annotated[int, field(
        doc='The batch size to use for training',
        default=8,
    )]
    report_every: Annotated[int, field(
        doc='The number of steps between each report',
        default=100,
    )]
    steps: Annotated[int, field(
        doc='The number of steps trained',
        default=0,
    )]

    def run_self(self):
        super().run_self()

        start = time.perf_counter()
        self.steps = self.train()
        end = time.perf_counter()
        ksteps = self.steps/1000.
        self.print(f'time elapsed: {(end - start)/ksteps:.4f} seconds per thousand steps')

    def train(self) -> int:
        raise NotImplementedError()

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)
        state.update(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            report_every=self.report_every,
            steps=self.steps,
        )


class ModelTrainingExperiment(TrainingExperiment):

    __slots__ = ('model', )

    model: Annotated[Module, field(
        doc='The model to train',
    )]

    def _coerce_model(self, spec: Any) -> Module:
        return self.get_module(spec, "model")

    def _add_state(self, state: dict[str, Any]):
        super()._add_state(state)
        state.update(
            model=self.model.structure(),
            arch=self.arch,
        )


