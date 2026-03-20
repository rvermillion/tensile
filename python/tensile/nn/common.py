#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.


from ..common import *

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