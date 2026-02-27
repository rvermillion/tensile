#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from typing import Any, TypeVar
from .types import Predicate


U = TypeVar('U', contravariant=True)
P = TypeVar('P', bound=Predicate)


def is_instance(cls: type) -> Predicate:
    def is_instance_pred(value) -> bool:
        return isinstance(value, cls)
    return is_instance_pred


def starts_with(prefix: str) -> Predicate[str]:
    def starts_with_pred(value: str) -> bool:
        return value is not None and value.startswith(prefix)
    return starts_with_pred


def ends_with(suffix: str) -> Predicate[str]:
    def ends_with_pred(value: str) -> bool:
        return value is not None and value.endswith(suffix)
    return ends_with_pred


def contains(part: str) -> Predicate[str]:
    def contains_pred(value: str) -> bool:
        return value is not None and value.find(part) >= 0
    return contains_pred


def matches(pattern: str) -> Predicate[str]:
    import re
    regex = re.compile(pattern)
    def entry_predicate(e: str) -> bool:
        return regex.match(e) is not None
    return entry_predicate


# noinspection PyUnusedLocal
def always(value: Any) -> bool:
    return True


# noinspection PyUnusedLocal
def never(value: Any) -> bool:
    return False


def invert(pred: P) -> P:
    def inverse(e) -> bool:
        return not pred(e)
    return inverse


def xor(a: P, b: P) -> P:
    return lambda value: not b(value) if a(value) else b(value)


# noinspection PyShadowingBuiltins
def all(*predicates: P) -> P:
    preds = []
    for p in predicates:
        if p is None: continue
        if p is never: return p
        if p is not always:
            preds.append(p)
    count = len(preds)
    if count == 0:
        pred = always
    elif count == 1:
        pred = preds[0]
    elif count == 2:
        a, b = preds
        def pred(e) -> bool:
            return a(e) and b(e)
    elif count == 3:
        a, b, c = preds
        def pred(e) -> bool:
            return a(e) and b(e) and c(e)
    else:
        half = count // 2
        pred = all(all(*preds[:half]), all(*preds[half:]))
    return pred


# noinspection PyShadowingBuiltins
def any(*predicates: P) -> P:
    preds = []
    for p in predicates:
        if p is None: continue
        if p is always: return p
        if p is not never:
            preds.append(p)
    count = len(preds)
    if count == 0:
        pred = never
    elif count == 1:
        pred = preds[0]
    elif count == 2:
        a, b = preds
        def pred(e) -> bool:
            return a(e) and b(e)
    elif count == 3:
        a, b, c = preds
        def pred(e) -> bool:
            return a(e) and b(e) and c(e)
    else:
        half = count // 2
        pred = any(any(*preds[:half]), any(*preds[half:]))
    return pred


__all__ = [
    'contains',
    'ends_with',
    'invert',
    'is_instance',
    'starts_with',
    'xor',
]
