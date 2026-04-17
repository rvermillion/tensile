#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from .. import ten, Array
from ..infra import RootObject
from .buffer import ArrayBuffer


class Metric(RootObject):

    __slots__ = ('name',)

    name: str

    def add_tensors(self, tensors: dict[str, Array], name: str = None):
        pass

    def _repr_args(self, **options) -> str:
        return self.name

    @classmethod
    def from_array(cls, name: str, array: Array) -> 'Metric':
        return ArrayMetric(name, array)

    @classmethod
    def from_buffer(cls, name: str, buffer: ArrayBuffer) -> 'Metric':
        return ArrayMetric(name, buffer.fetch())

    @classmethod
    def from_changes(cls, name: str, changes: list[tuple[int, float]], last_step: int = None) -> 'Metric':
        steps = [s for s, v in changes]
        values = [v for s, v in changes]
        return cls.from_steps(name, steps, values, last_step=last_step)

    @classmethod
    def from_steps(cls, name: str, steps: list[int], values: list[float], last_step: int = None) -> 'Metric':
        if last_step is not None:
            steps.append(last_step)
            values.append(values[-1])
        return StepMetric(name, steps, values)


class ArrayMetric(Metric):

    __slots__ = ('array',)

    array: Array

    def __init__(self, name: str, array: Array):
        self.name = name
        self.array = array

    def add_tensors(self, tensors: dict[str, Array], name: str = None):
        if name is None: name = self.name
        tensors[name] = self.array


class StepMetric(Metric):

    __slots__ = ('steps', 'values')

    steps: list[int]
    values: list[float]

    def __init__(self, name: str, steps: list[int], values: list[float]):
        self.name = name
        self.steps = steps
        self.values = values

    def add_tensors(self, tensors: dict[str, Array], name: str = None):
        if name is None: name = self.name
        tensors[name + '.steps'] = ten.array(self.steps)
        tensors[name + '.values'] = ten.array(self.values)

