#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from types import NoneType
from typing import Any, Callable, Mapping, TypeVar, get_origin


special_class_names: dict[type, str] = {}

special_class_qnames: dict[type, str] = {}


def register_special_class_name(cls: type, name: str, qname: str = None, override: bool = False):
    if qname is None:
        qname = name
    if not override:
        if cls in special_class_names:
            raise ValueError(f'class {cls} already has a special name: {special_class_names[cls]}')
        if cls in special_class_qnames:
            raise ValueError(f'class {cls} already has a special qname: {special_class_qnames[cls]}')
    special_class_names[cls] = name
    special_class_qnames[cls] = qname


def register_special_class_names(names: dict[type, str], qnames: dict[type, str] = None, override: bool = False):
    for cls, name in names.items():
        register_special_class_name(cls, name, qname=None if qnames is None else qnames.get(cls), override=override)


register_special_class_names({
    str: 'str',
    bool: 'bool',
    int: 'int',
    float: 'float',
    dict: 'dict',
    list: 'list',
    tuple: 'tuple',
    set: 'set',
    get_origin(Callable): 'callable',
    NoneType: 'None',
    Any: 'Any'
})


def class_name(cls: type) -> str:
    if name := special_class_names.get(cls):
        return name
    return cls.__qualname__


def class_qname(cls: type) -> str:
    if qname := special_class_qnames.get(cls):
        return qname
    mod = cls.__module__
    if mod == '__main__':
        mod = 'main'
    return mod + '.' + cls.__qualname__


F = TypeVar('F', bound=Callable)


def name_function(func: F, name: str, qualname: str = None, cls: type = None, namespace: str = None) -> F:
    func._orig_names = (func.__name__, func.__qualname__)
    func.__name__ = name
    if qualname is None:
        if namespace is None and cls is not None:
            namespace = class_qname(cls)
        if namespace is None:
            func.__qualname__ = name
        else:
            func.__qualname__ = namespace + '.' + name
    else:
        func.__qualname__ = qualname
    return func

def process_specs(spec: Mapping[str, Any] = None, /, kind: str = None, **kwargs) -> tuple[str, Mapping[str, Any]]:
    if spec is None:
        spec = kwargs
    else:
        spec = dict(spec)
        spec.update(kwargs)
        if kind is None:
            kind = spec.pop('kind', None)
    # if kind is None and len(spec) == 1:
    #     (kind, spec), = spec.items()
    return kind, spec
