#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ...infrastructure import field
from ...infrastructure.types import Annotated, Any, Optional, Self, TypeVar, Union
from ...config import Config


Params = dict[str, Any]


def spec_from_dict(cls: type[Config], params: Params) -> Params:
    fields = cls.meta.fields

    return {
        k: v
        for k, v in params.items()
        if k in fields
    }


def spec_from_args(cls: type[Config], args: 'ModuleArgs') -> Params:
    fields = cls.meta.instance_fields
    arg_fields = args.meta.instance_fields

    return {
        k: f.get(args)
        for k, f in arg_fields.items()
        if k in fields
    }


# class QuantizationConfig(Config):
#     group_size: int
#     bits: int

A = TypeVar('A', bound='ModuleArgs')


class ModuleArgs(Config):

    kind: Annotated[Optional[str], field(inherit=False)] = None
    dropout: Annotated[Optional[dict[str, Any]], field()] = None

    def make_args(self, name: str, /, cls: type[A] = None, **kwargs) -> A:
        if cls is None: cls = self.__class__
        return cls.from_dict(kwargs, parent=self, step=name)

    def args_like(self, spec: dict[str, Any] = None, /, **kwargs) -> A:
        data = self._config_data
        if spec is not None: kwargs.update(spec)
        return self.__class__.from_dict(kwargs, parent=data.parent.config, step=data.step)

    @classmethod
    def from_dict(cls, params: Params, parent: Config = None, step: str = None) -> Self:
        return cls.construct(params, parent=parent, step=step)

    @classmethod
    def from_args(cls, args: 'ModuleArgs', parent: Config = None, step: str = None) -> Self:
        if args is None:
            raise ValueError('args must not be None')
        data = args._config_data
        if parent is None: parent = data.parent
        if step is None: step = data.step

        new_args = cls.construct(*data.locals, step=step, parent=parent)
        if data.defaults:
            new_args.set_defaults(**data.defaults)
        return new_args
        # noinspection PyArgumentList
        # return cls.construct(spec_from_args(cls, args), step=data.step, parent=data.parent)

    @classmethod
    def combine_args(cls, *args: Union[Params, 'ModuleArgs'], **kwargs) -> Self:
        spec = {}

        for arg in args:
            spec.update(spec_from_dict(cls, arg) if isinstance(arg, dict) else spec_from_args(cls, arg))

        if kwargs:
            spec.update(spec_from_dict(cls, kwargs))

        # noinspection PyArgumentList
        return cls(values=spec)


    def get_first(self, *args: str, default: Any = ...) -> Any:
        val = Config.get_first(self, *args, default=default)
        if val is ...:
            raise AttributeError(f'{self} does not have attributes {", ".join(args)}')
        return val
        # for arg in args:
        #     val = Config.get(self, arg, default=...)
        #     if val is not ...:
        #         return val
        # if default is ...:
        #     raise AttributeError(f'{self} does not have attributes {", ".join(args)}')
        # return default

    def get(self, arg: str, default: Any = ...) -> Any:
        val = Config.get(self, arg, default=default)
        if val is ...:
            raise AttributeError(f'{self} does not have attributes {arg}')
        return val
        # for arg in args:
        #     val = Config.get(self, arg, default=...)
        #     if val is not ...:
        #         return val
        # if default is ...:
        #     raise AttributeError(f'{self} does not have attributes {", ".join(args)}')
        # return default

    # def set_defaults(self, **kwargs):
    #     for arg, default in kwargs.items():
    #         value = getattr(self, arg, None)
    #         if value is None:
    #             setattr(self, arg, default)
    #         elif isinstance(default, dict):
    #             if isinstance(value, ModuleArgs):
    #                 value.set_defaults(**default)
    #             elif isinstance(value, dict):
    #                 value.update(default)

    # noinspection PyUnusedLocal
    @classmethod
    def with_fallback(cls, **kwargs) -> type[Self]:
        return cls

    @classmethod
    def update_args(cls, args: 'ModuleArgs', **kwargs):
        for arg, value in kwargs.items():
            if value is not None:
                setattr(args, arg, value)

