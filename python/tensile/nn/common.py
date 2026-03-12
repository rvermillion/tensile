#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

from typing import (Annotated, Any, Callable, ClassVar, Generic, Iterable, Iterator, Mapping, Optional,
                    Protocol, Sequence, Self, TYPE_CHECKING, TypeAlias, TypeVar, Union)

from ..shims import ten, Array, DType, Shape
from ..infra import (
    Predicate, PredicateLike, Object, coerce, meta, field, predicates,
    provides, Spec, Transform, TransformLike, tree
)
from ..infra.tree import Tree, TreeEntry
from ..infra.types import Factory, Keywords, PredicateFunction, TransformFunction, JSON


class Activation(Protocol):

    def __call__(self, x: Array) -> Array: ...


class BinaryActivation(Protocol):

    def __call__(self, a: Array, b: Array) -> Array: ...


__all__ = [
    'Activation',
    'Annotated',
    'Any',
    'Array',
    'BinaryActivation',
    'Callable',
    'ClassVar',
    'DType',
    'Generic',
    'Factory',
    'Iterable',
    'Iterator',
    'JSON',
    'Keywords',
    'Mapping',
    'Object',
    'Optional',
    'Predicate',
    'PredicateFunction',
    'PredicateLike',
    'Protocol',
    'Self',
    'Sequence',
    'Shape',
    'Spec',
    'Transform',
    'TransformFunction',
    'TransformLike',
    'Tree',
    'TreeEntry',
    'TypeAlias',
    'TypeVar',
    'TYPE_CHECKING',
    'Union',
    'coerce',
    'field',
    'meta',
    'predicates',
    'provides',
    'ten',
    'tree',
]