#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

from typing import (Annotated, Any, ClassVar, Generic, Optional,
                    Protocol, Self, TYPE_CHECKING, TypeAlias, TypeVar, Union)
from collections.abc import (Callable, Iterable, Iterator, Mapping, Sequence)

from . import ten
from .ten import Array, DType, Shape, AxisSelector

from .infra import (
    Predicate, PredicateLike, Object, coerce, meta, field, predicates,
    provides, Spec, Transform, TransformLike, tree
)
from .infra.tree import Tree, TreeEntry
from .infra.types import Factory, Keywords, PredicateFunction, TransformFunction, JSON, JSONObject, JSONList


Slice = slice

full_slice: AxisSelector = slice(None)


__all__ = [
    'Annotated',
    'Any',
    'Array',
    'Callable',
    'ClassVar',
    'DType',
    'Generic',
    'Factory',
    'Iterable',
    'Iterator',
    'JSON',
    'JSONObject',
    'JSONList',
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
    'Slice',
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