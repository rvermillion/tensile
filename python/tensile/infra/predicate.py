#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
from collections.abc import Sized
from typing import Any, Callable, ClassVar, Container, Generic, Iterable, Mapping, Optional, Sequence, TypeVar
import re
import operator as op
from builtins import all as ball, any as bany

import tensile.infra as infra

from .types import Comparison, PredicateFunction, PredicateLike, TransformFunction, TransformLike, TYPE_CHECKING, missing
from .meta import meta_coerce_class, meta_configure_coerce
from .util import class_qname, name_function, tie_call, spread_mapping

X = TypeVar('X')
U = TypeVar('U', contravariant=True)
P = TypeVar('P', bound=PredicateFunction)

if TYPE_CHECKING:
    from .transform import Transform

# noinspection PyUnusedLocal
def always(value: Any) -> bool:
    return True


# noinspection PyUnusedLocal
def never(value: Any) -> bool:
    return False


def is_none(value: Any) -> bool:
    return value is None


def is_not_none(value: Any) -> bool:
    return value is not None


def is_true(value: Any) -> bool:
    return bool(value)


def is_false(value: Any) -> bool:
    return not bool(value)


def is_instance(cls: type|tuple[type, ...]) -> PredicateFunction:
    if isinstance(cls, tuple):
        if len(cls) == 0: return never
        if len(cls) == 1:
            cls = cls[0]
        else:
            def pred(value) -> bool:
                return isinstance(value, cls)
            return name_function(pred, f'is_instance[{tuple(map(class_qname, cls))}]')
    if isinstance(cls, type):
        def pred(value) -> bool:
            return isinstance(value, cls)
        return name_function(pred, f'is_instance[{class_qname(cls)}]')
    else:
        raise TypeError(f"Invalid class: {cls}")



def starts_with(prefix: str) -> PredicateFunction[str]:
    def pred(value: str) -> bool:
        return value is not None and value.startswith(prefix)
    return name_function(pred, f'starts_with[{prefix}]')


def ends_with(suffix: str) -> PredicateFunction[str]:
    def pred(value: str) -> bool:
        return value is not None and value.endswith(suffix)
    return name_function(pred, f'ends_with[{suffix}]')


def contains(part: str) -> PredicateFunction[str]:
    def pred(value: str) -> bool:
        return value is not None and value.find(part) >= 0
    return name_function(pred, f'contains[{part}]')


def matches(pattern: str) -> PredicateFunction[str]:
    import re
    regex = re.compile(pattern)
    def pred(e: str) -> bool:
        return regex.match(e) is not None
    return name_function(pred, f'matches[{pattern}]')


def invert(pred: P) -> P:
    if pred is always: return never
    if pred is never: return always
    if pred is is_false: return is_true
    if pred is is_true: return is_false

    def inverse(e) -> bool:
        return not pred(e)
    return name_function(inverse, f'not[{pred.__name__}]')


def xor(a: P, b: P) -> P:
    def pred(value) -> bool:
        return not b(value) if a(value) else b(value)
    return name_function(pred, f"xor[{a.__name__}, {b.__name__}]")


def transform(txf: TransformFunction[X, U], predicate: PredicateFunction[U]) -> PredicateFunction[X]:
    if predicate is always: return always
    if predicate is never: return never
    def evaluate(value: U) -> bool:
        return predicate(txf(value))
    return name_function(evaluate, f'transform[{txf.__name__}, {predicate.__name__}]')

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
        return always
    elif count == 1:
        return preds[0]
    elif count == 2:
        a, b = preds
        def pred(x) -> bool:
            return a(x) and b(x)
        return name_function(pred, f'and[{", ".join(p.__name__ for p in preds)}]')
    elif count == 3:
        a, b, c = preds
        def pred(x) -> bool:
            return a(x) and b(x) and c(x)
    elif count == 4:
        a, b, c, d = preds
        def pred(x) -> bool:
            return a(x) and b(x) and c(x) and d(x)
    else:
        half = count // 2
        pred = all(all(*preds[:half]), all(*preds[half:]))
    return name_function(pred, f'all[{", ".join(p.__name__ for p in preds)}]')


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
        return never
    elif count == 1:
        return preds[0]
    elif count == 2:
        a, b = preds
        def pred(x) -> bool:
            return a(x) or b(x)
        return name_function(pred, f'or[{", ".join(p.__name__ for p in preds)}]')
    elif count == 3:
        a, b, c = preds
        def pred(x) -> bool:
            return a(x) or b(x) or c(x)
    elif count == 4:
        a, b, c, d = preds
        def pred(x) -> bool:
            return a(x) or b(x) or c(x) or d(x)
    else:
        half = count // 2
        pred = any(any(*preds[:half]), any(*preds[half:]))
    return name_function(pred, f'any[{", ".join(p.__name__ for p in preds)}]')


def has_attr(name: str) -> PredicateFunction:
    def has_attr_pred(obj: Any) -> bool:
        value = getattr(obj, name, missing)
        return value is not missing
    return name_function(has_attr_pred, f'has_attr[{name}]')


def attr(name: str, pred: PredicateFunction, if_missing: bool = False) -> PredicateFunction:
    if pred is never:
        return invert(has_attr(name)) if if_missing else never
    if pred is always:
        return always if if_missing else has_attr(name)
    def attr_pred(obj: Any) -> bool:
        value = getattr(obj, name, missing)
        return if_missing if value is missing else pred(value)
    return name_function(attr_pred, f'attr[{name}, {pred.__name__}]')


def attrs(if_missing: bool = False, /, **preds: PredicateFunction) -> PredicateFunction[Mapping]:
    return all(*(attr(n, p, if_missing=if_missing) for n, p in preds.items()))


def has(item: X) -> PredicateFunction[Container[X]]:
    def pred(obj: Container) -> bool:
        return item in obj
    return name_function(pred, f'has[{item}]')


def has_not(item: X) -> PredicateFunction[Container[X]]:
    def pred(obj: Container) -> bool:
        return item not in obj
    return name_function(pred, f'has_not[{item}]')


# noinspection PyShadowingNames
def key(key: str, pred: PredicateFunction, if_missing: bool = False) -> PredicateFunction[Mapping]:
    if pred is never:
        return has_not(key) if if_missing else never
    if pred is always:
        return always if if_missing else has(key)
    def key_pred(obj: Any) -> bool:
        try:
            value = obj[key]
        except (KeyError, TypeError, IndexError):
            return if_missing
        return pred(value)
    return name_function(key_pred, f'key[{key!r}, {pred.__name__}]')


def keys(if_missing: bool = False, /, **preds: PredicateFunction) -> PredicateFunction[Mapping]:
    return all(*(key(k, p, if_missing=if_missing) for k, p in preds.items()))


def any_item(pred: PredicateFunction[U]) -> PredicateFunction[Iterable[U]]:
    if pred is never: return never
    def any_item_pred(items: Iterable) -> bool:
        for item in items:
            if pred(item):
                return True
        return False
    return any_item_pred


def all_items(pred: PredicateFunction[U]) -> PredicateFunction[Iterable[U]]:
    if pred is always: return always
    def any_item_pred(items: Iterable) -> bool:
        for item in items:
            if not pred(item):
                return False
        return True
    return any_item_pred


def every_n(n: int, offset: int = 0) -> PredicateFunction[int]:
    if n <= 0: raise ValueError("every must be positive")
    offset = (n - offset) % n
    def every_n(step: int) -> bool:
        return step % n == offset
    what = 'step' if n == 1 else 'steps'
    if offset == 0:
        name = f'every_{n}_{what}'
    else:
        name = f'every_{n}_{what}_offset_{offset}'
    return name_function(every_n, name)


class Predicate(Generic[U]):

    __slots__ = ('evaluate',)

    evaluate: PredicateFunction[U]
    not_evaluate: PredicateFunction[U]

    def __init__(self, evaluate: PredicateFunction[U]):
        self.evaluate = evaluate

    @property
    def is_always(self) -> bool:
        return self.evaluate is always

    @property
    def is_never(self) -> bool:
        return self.evaluate is never

    @property
    def is_constant(self) -> bool:
        return self.is_always or self.is_never

    @property
    def __name__(self) -> str:
        return self.evaluate.__name__

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return invert(self.evaluate)

    def describe(self, arg: str) -> str:
        return f'{self.evaluate.__name__}({arg})'

    def implies(self, other: PredicateFunction[U]) -> bool:
        return self._implies(coerce(other), True)

    def is_implied_by(self, other: PredicateFunction[U]) -> bool:
        return self._is_implied_by(coerce(other), True)

    def denies(self, other: PredicateFunction[U]) -> bool:
        return self._denies(coerce(other), True)

    def is_denied_by(self, other: PredicateFunction[U]) -> bool:
        return self._is_denied_by(coerce(other), True)

    def _implies(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_never or
            (isinstance(other, self.__class__) and self._same_implies(other)) or
            (reverse and other._is_implied_by(self, False))
        )

    def _same_implies(self, other: 'Predicate[U]') -> bool:
        return False

    def _is_implied_by(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_always or
            (isinstance(other, self.__class__) and self._same_is_implied_by(other)) or
            (reverse and other._implies(self, False))
        )

    def _same_is_implied_by(self, other: 'Predicate[U]') -> bool:
        return False

    def _denies(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and (
                self.is_never or (
                (isinstance(other, self.__class__) and self._same_denies(other)) or
                (reverse and other._is_denied_by(self, False))
            ))
        )

    def _same_denies(self, other: 'Predicate[U]') -> bool:
        return False

    def _is_denied_by(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            not self.is_always and (
                (isinstance(other, self.__class__) and self._same_is_denied_by(other)) or
                (reverse and other._denies(self, False))
            )
        )

    def _same_is_denied_by(self, other: 'Predicate[U]') -> bool:
        return False

    def _eq_tuple(self) -> tuple:
        return (self.evaluate,)

    def __call__(self, value: U) -> bool:
        return self.evaluate(value)

    def __eq__(self, other: Any) -> bool:
        return self is other or (
            isinstance(other, self.__class__) and
            self._eq_tuple() == other._eq_tuple()
        )

    def __hash__(self) -> int:
        return hash(self._eq_tuple())

    def __and__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return Predicates.all(self, other)

    def __rand__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return Predicates.all(other, self)

    def __or__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return Predicates.any(self, other)

    def __ror__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return Predicates.any(other, self)

    def __xor__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return Predicates.xor(self, other)

    def __rxor__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return Predicates.xor(other, self)

    def __invert__(self) -> 'Predicate[U]':
        return InversePredicate(self)

    def __repr__(self):
        return 'Predicate(' + self.describe('x') + ')'


tie_call(Predicate, 'evaluate')

def coerce(spec: PredicateLike[U]) -> Predicate[U]:
    if isinstance(spec, Predicate): return spec
    if spec is always or spec is None: return Predicates.always
    if spec is never: return Predicates.never
    if callable(spec):
        # noinspection PyTypeChecker
        return Predicate(spec)
    if isinstance(spec, Mapping):
        if len(spec) == 1:
            k, = spec.keys()
            if k != 'kind':
                value = spec[k]
                if factory := named_factories.get(k):
                    return factory(value)
                raise ValueError(f"Invalid key: {k!r}")
    if isinstance(spec, str):
        if singleton := named_singletons.get(spec):
            return singleton
        if factory := named_factories.get(spec):
            return factory(None)
        raise ValueError(f"Invalid string spec: {spec!r}")
    if isinstance(spec, Sequence):
        return all_factory(spec)
    raise ValueError(f"Invalid predicate spec: {spec}")


# noinspection PyUnusedLocal
def full_coerce(spec: Any = None, /, **kwargs) -> Predicate:
    if spec is None:
        spec = kwargs
    elif kwargs and isinstance(spec, Mapping):
        kwargs.update(spec)
        spec = kwargs
    return coerce(spec)


# noinspection PyTypeChecker
meta_configure_coerce(Predicate, full_coerce)
# meta_configure_coerce(PredicateFunction, full_coerce)


def all_factory(predicates: Iterable[PredicateLike[U]]) -> Predicate[U]:
    return Predicates.all(*(coerce(p) for p in predicates))


def any_factory(predicates: Iterable[PredicateLike[U]]) -> Predicate[U]:
    return Predicates.any(*(coerce(p) for p in predicates))


def not_factory(predicate: PredicateLike[U]) -> Predicate[U]:
    return Predicates.invert(coerce(predicate))


short_circuit = False
short_circuit_not = short_circuit
short_circuit_and = short_circuit
short_circuit_or = short_circuit


class AlwaysPredicate(Predicate[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(always)

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return never

    def __and__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return coerce(other) if short_circuit_and else super().__and__(other)

    def __rand__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return coerce(other) if short_circuit_and else super().__rand__(other)

    def __or__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return self if short_circuit_or else super().__or__(other)

    def __ror__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return self if short_circuit_or else super().__ror__(other)

    def __invert__(self) -> 'Predicate[U]':
        return Predicates.never if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return 'always'

    is_always = True
    is_never = False
    is_constant = True


class NeverPredicate(Predicate[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(never)

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return always

    def __and__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return self if short_circuit_and else super().__and__(other)

    def __rand__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return self if short_circuit_and else super().__rand__(other)

    def __or__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return coerce(other) if short_circuit_or else super().__or__(other)

    def __ror__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        return coerce(other) if short_circuit_or else super().__ror__(other)

    def __invert__(self) -> 'Predicate[U]':
        return Predicates.always if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return 'never'

    is_always = False
    is_never = True
    is_constant = True


class IsTruePredicate(Predicate[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(is_true)

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return is_false

    def _denies(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return super()._denies(other, reverse) or other.evaluate is is_false

    def _is_denied_by(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return super()._is_denied_by(other, reverse) or other.evaluate is is_false

    def __invert__(self) -> 'Predicate[U]':
        return Predicates.is_false if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return f'bool({arg})'


class IsFalsePredicate(Predicate[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(is_false)

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return is_true

    def _denies(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return super()._denies(other, reverse) or other.evaluate is is_true

    def _is_denied_by(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return super()._is_denied_by(other, reverse) or other.evaluate is is_true

    def __invert__(self) -> 'Predicate[U]':
        return Predicates.is_true if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return f'not bool({arg})'



class InversePredicate(Predicate[U]):

    __slots__ = ('inverse',)

    inverse: Predicate[U]

    def __init__(self, inverse: Predicate[U]):
        super().__init__(inverse.not_evaluate)
        self.inverse = inverse

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return self.inverse.evaluate

    def _eq_tuple(self) -> tuple:
        return self.inverse,

    def _denies(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            self.is_never and (
                self.inverse._is_implied_by(other, True) or
                (reverse and other._is_denied_by(self, False))
            )
        )

    def __invert__(self) -> 'Predicate[U]':
        return self.inverse if short_circuit_not else InversePredicate(self)

    def describe(self, arg: str) -> str:
        return '~' + self.inverse.describe(arg)
        # return 'not(' + self.inverse.describe(arg) + ')'


def _combine_all(preds: Sequence[Predicate[U]]) -> PredicateFunction[U]:
    evals = [p.evaluate for p in preds]
    for i, this in enumerate(preds[:-1]):
        if evals[i] is not always:
            for j, that in enumerate(preds[i+1:], start=1):
                if this.implies(that):
                    evals[i+j] = always
                elif that.implies(this):
                    # We pull it forward so that we don't mess up short-circuit AND
                    evals[i] = evals[i+j]
                    evals[i+j] = always
                elif this.denies(that):
                    return never
                elif that.denies(this):
                    return never
    return all(*evals)


def _combine_any(preds: Sequence[Predicate[U]]) -> PredicateFunction[U]:
    evals = [p.evaluate for p in preds]
    for i, this in enumerate(preds[:-1]):
        if evals[i] is not never:
            for j, that in enumerate(preds[i+1:], start=1):
                if this.implies(that):
                    # We pull it forward so that we don't mess up short-circuit AND

                    evals[i] = evals[i+j]
                    evals[i+j] = never
                elif that.implies(this):
                    evals[i+j] = never
                elif this.implies(~that):
                    return always
                elif that.implies(~this):
                    return always
    return any(*evals)


class AllPredicate(Predicate[U]):

    __slots__ = ('predicates', )

    predicates: tuple[Predicate[U], ...]

    def __init__(self, predicates: Iterable[Predicate[U]]):
        predicates = tuple(predicates)
        evaluate = _combine_all(predicates)
        super().__init__(evaluate)
        self.predicates = predicates

    def _eq_tuple(self) -> tuple:
        return self.predicates

    def _implies(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_never or
            bany(p._implies(other, True) for p in self.predicates) or
            (reverse and other._is_implied_by(self, False))
        )

    def _is_implied_by(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_always or
            ball(p._is_implied_by(other, True) for p in self.predicates) or
            (reverse and other._implies(self, False))
        )

    def _denies(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            self.is_never and (
                bany(p._denies(other, True) for p in self.predicates) or
                (reverse and other._is_denied_by(self, False))
            )
        )

    def _is_denied_by(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            not self.is_always and (
                bany(p._is_denied_by(other, True) for p in self.predicates) or
                (reverse and other._denies(self, False))
            )
        )

    def __and__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        if short_circuit_and:
            if isinstance(other, AllPredicate):
                return Predicates.all(*self.predicates, *other.predicates)
            else:
                return Predicates.all(*self.predicates, other)
        else:
            return super().__and__(other)

    def __rand__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        if short_circuit_and:
            if isinstance(other, AllPredicate):
                return Predicates.all(*other.predicates, *self.predicates)
            else:
                return Predicates.all(other, *self.predicates)
        else:
            return super().__rand__(other)

    def describe(self, arg: str) -> str:
        predicates = self.predicates
        if len(predicates) == 2:
            return '(' + predicates[0].describe(arg) + ' & ' + predicates[1].describe(arg) + ')'
            # return 'and(' + ', '.join(p.describe(arg) for p in predicates) + ')'
        else:
            return 'all(' + ', '.join(p.describe(arg) for p in predicates) + ')'


class AnyPredicate(Predicate[U]):

    __slots__ = ('predicates', )

    predicates: tuple[Predicate[U], ...]

    def __init__(self, predicates: Iterable[Predicate[U]], evaluate: PredicateFunction[U] = None):
        predicates = tuple(predicates)
        if evaluate is None: evaluate = _combine_any(predicates)
        super().__init__(evaluate)
        self.predicates = predicates

    def _eq_tuple(self) -> tuple:
        return self.predicates

    def _implies(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_never or
            ball(p._implies(other, True) for p in self.predicates) or
            (reverse and other._is_implied_by(self, False))
        )

    def _is_implied_by(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_always or
            bany(p._is_implied_by(other, True) for p in self.predicates) or
            (reverse and other._implies(self, False))
        )

    def _denies(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            self.is_never and (
                ball(p._denies(other, True) for p in self.predicates) or
                (reverse and other._is_denied_by(self, False))
            )
        )

    def _is_denied_by(self, other: Predicate[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            not self.is_always and (
                ball(p._is_denied_by(other, True) for p in self.predicates) or
                (reverse and other._denies(self, False))
            )
        )

    def __or__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        if short_circuit_or:
            if isinstance(other, AnyPredicate):
                return Predicates.any(*self.predicates, *other.predicates)
            else:
                return Predicates.any(*self.predicates, other)
        else:
            return super().__or__(other)

    def __ror__(self, other: PredicateFunction[U]) -> 'Predicate[U]':
        if short_circuit_or:
            if isinstance(other, AnyPredicate):
                return Predicates.any(*other.predicates, *self.predicates)
            else:
                return Predicates.any(other, *self.predicates)
        else:
            return super().__ror__(other)

    def describe(self, arg: str) -> str:
        predicates = self.predicates
        if len(predicates) == 2:
            return '(' + predicates[0].describe(arg) + ' | ' + predicates[1].describe(arg) + ')'
        else:
            return 'any(' + ', '.join(p.describe(arg) for p in predicates) + ')'


class XorPredicate(Predicate[U]):

    __slots__ = ('left', 'right')

    left: Predicate[U]
    right: Predicate[U]

    def __init__(self, left: Predicate[U], right: Predicate[U]):
        if left.denies(right) or right.denies(left):
            evaluate = any(left, right)
        elif left.implies(right):
            if right.implies(left):
                evaluate = never
            else:
                # xor is true only when left=False and right=True
                evaluate = all(left.not_evaluate, right.evaluate)
        elif right.implies(left):
            # xor is true only when right=False and left=True
            evaluate = all(right.not_evaluate, left.evaluate)
        else:
            evaluate = xor(left.evaluate, right.evaluate)
        super().__init__(evaluate)
        self.left = left
        self.right = right

    def _eq_tuple(self) -> tuple:
        return self.left, self.right

    def describe(self, arg: str) -> str:
        return 'xor(' + self.left.describe(arg) + ', ' + self.right.describe(arg) + ')'


compare_symbols = {
    op.eq: '==',
    op.ne: '!=',
    op.gt: '>',
    op.ge: '>=',
    op.le: '<=',
    op.lt: '<',
    op.is_: 'is',
    op.is_not: 'is not',
}

# noinspection PyUnusedLocal
def never_comparison(x: Any, y: Any) -> bool:
    return False


inverses: dict[Comparison, Comparison] = {
    op.eq: op.ne,
    op.ne: op.eq,
    op.gt: op.le,
    op.ge: op.lt,
    op.lt: op.ge,
    op.le: op.gt,
    op.is_: op.is_not,
    op.is_not: op.is_,
}


implications: dict[tuple[Comparison, Comparison], Comparison] = {
    (op.eq, op.eq): lambda x, y: x == y,
    (op.ne, op.ne): lambda x, y: x == y,
    (op.eq, op.gt): lambda x, y: x > y,
    (op.eq, op.ge): lambda x, y: x >= y,
    (op.eq, op.lt): lambda x, y: x < y,
    (op.eq, op.le): lambda x, y: x <= y,
    (op.gt, op.gt): lambda x, y: x >= y,
    (op.ge, op.ge): lambda x, y: x >= y,
    (op.gt, op.ge): lambda x, y: x >= y,
    (op.ge, op.gt): lambda x, y: x > y,
    (op.lt, op.lt): lambda x, y: x <= y,
    (op.le, op.le): lambda x, y: x <= y,
    (op.lt, op.le): lambda x, y: x <= y,
    (op.le, op.lt): lambda x, y: x < y,
    (op.is_, op.is_): lambda x, y: x is y,
    (op.is_not, op.is_not): lambda x, y: x is y,
    (op.is_, op.eq): lambda x, y: x == y,
    (op.is_, op.ne): lambda x, y: x != y,
    (op.is_, op.gt): lambda x, y: x > y,
    (op.is_, op.ge): lambda x, y: x >= y,
    (op.is_, op.lt): lambda x, y: x < y,
    (op.is_, op.le): lambda x, y: x <= y,
}


denials: dict[tuple[Comparison, Comparison], Comparison] = {
    (op.eq, op.eq): lambda x, y: x != y,
    (op.is_, op.ne): lambda x, y: x == y,
    (op.eq, op.ne): lambda x, y: x == y,
    (op.ne, op.eq): lambda x, y: x == y,
    (op.ne, op.is_): lambda x, y: x == y,

    (op.is_, op.gt): lambda x, y: x <= y,
    (op.eq, op.gt): lambda x, y: x <= y,
    (op.lt, op.gt): lambda x, y: x <= y,
    (op.le, op.gt): lambda x, y: x <= y,

    (op.is_, op.ge): lambda x, y: x < y,
    (op.eq, op.ge): lambda x, y: x < y,
    (op.lt, op.ge): lambda x, y: x <= y,
    (op.le, op.ge): lambda x, y: x < y,

    (op.is_, op.lt): lambda x, y: x >= y,
    (op.eq, op.lt): lambda x, y: x >= y,
    (op.gt, op.lt): lambda x, y: x >= y,
    (op.ge, op.lt): lambda x, y: x >= y,

    (op.is_, op.le): lambda x, y: x > y,
    (op.eq, op.le): lambda x, y: x > y,
    (op.gt, op.le): lambda x, y: x >= y,
    (op.ge, op.le): lambda x, y: x > y,

    (op.is_, op.is_): lambda x, y: x is not y,
    (op.is_, op.is_not): lambda x, y: x is y,
    (op.is_not, op.is_): lambda x, y: x is y,
}

class ComparePredicate(Predicate[U]):

    __slots__ = ('compare', 'arg', 'inverse')

    compare: Comparison[U]
    arg: U
    inverse: Optional['ComparePredicate[U]']

    def __init__(self, compare: Comparison[U], arg: U, inverse: Optional['ComparePredicate[U]'] = None):
        if compare not in compare_symbols:
            raise ValueError(f"Invalid comparison operator: {compare}")
        def evaluate(value: U) -> bool:
            return compare(value, arg)
        evaluate = name_function(evaluate, 'compare[' + compare_symbols[compare] + f' {arg}]')
        super().__init__(evaluate)
        self.compare = compare
        self.arg = arg
        self.inverse = inverse

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        if inverse := self.inverse:
            return inverse.evaluate
        compare = inverses[self.compare]
        arg = self.arg
        def evaluate(value: U) -> bool:
            return compare(value, arg)
        return name_function(evaluate, 'compare[' + compare_symbols[compare] + f' {arg}]')

    def _eq_tuple(self) -> tuple:
        return self.compare, self.arg

    def _same_implies(self, other: 'ComparePredicate[U]') -> bool:
        return implications.get((self.compare, other.compare), never_comparison)(self.arg, other.arg)

    def _same_is_implied_by(self, other: 'ComparePredicate[U]') -> bool:
        return implications.get((other.compare, self.compare), never_comparison)(other.arg, self.arg)

    def _same_denies(self, other: 'ComparePredicate[U]') -> bool:
        return denials.get((self.compare, other.compare), never_comparison)(self.arg, other.arg)

    def _same_is_denied_by(self, other: 'ComparePredicate[U]') -> bool:
        return denials.get((other.compare, self.compare), never_comparison)(other.arg, self.arg)

    def __invert__(self) -> 'Predicate[U]':
        if short_circuit_not:
            if inverse := self.inverse: return inverse
        return ComparePredicate(inverses[self.compare], self.arg, self)

    def describe(self, arg: str) -> str:
        if inverse := self.inverse:
            return f'~{inverse.describe(arg)}'
        return f'({arg} {compare_symbols[self.compare]} {self.arg!r})'


class AttrPredicate(Predicate[U]):

    __slots__ = ('name', 'predicate', 'if_missing')

    name: str
    predicate: Predicate[U]
    if_missing: bool

    def __init__(self, name: str, predicate: Predicate[U], if_missing: bool = False):
        evaluate = attr(name, predicate.evaluate, if_missing=if_missing)
        super().__init__(evaluate)
        self.name = name
        self.predicate = predicate
        self.if_missing = bool(if_missing)

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return attr(self.name, self.predicate.not_evaluate, if_missing=not self.if_missing)

    def _eq_tuple(self) -> tuple:
        return self.name, self.predicate, self.if_missing

    def _same_implies(self, other: 'AttrPredicate[U]') -> bool:
        return (
            other.name == self.name and
            (other.if_missing or not self.if_missing) and
            self.predicate._implies(other.predicate, True)
        )

    def _same_is_implied_by(self, other: 'AttrPredicate[U]') -> bool:
        return (
            other.name == self.name and
            (self.if_missing or not other.if_missing) and
            self.predicate._is_implied_by(other.predicate, True)
        )

    def _same_denies(self, other: 'AttrPredicate[U]') -> bool:
        return (
            other.name == self.name and
            (self.if_missing or not other.if_missing) and
            self.predicate._denies(other.predicate, True)
        )

    def _same_is_denied_by(self, other: 'AttrPredicate[U]') -> bool:
        return (
            other.name == self.name and
            (not self.if_missing or other.if_missing) and
            self.predicate._is_denied_by(other.predicate, True)
        )

    def describe(self, arg: str) -> str:
        return self.predicate.describe(f'{arg}.{self.name}')


class KeyPredicate(Predicate[U]):

    __slots__ = ('key', 'predicate', 'if_missing')

    key: str
    predicate: Predicate[U]
    if_missing: bool

    def __init__(self, name: str, predicate: Predicate[U], if_missing: bool = False):
        evaluate = key(name, predicate.evaluate, if_missing=if_missing)
        super().__init__(evaluate)
        self.key = name
        self.predicate = predicate
        self.if_missing = if_missing

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return key(self.key, self.predicate.not_evaluate, if_missing=not self.if_missing)

    def _eq_tuple(self) -> tuple:
        return self.key, self.predicate, self.if_missing

    def _same_implies(self, other: 'KeyPredicate[U]') -> bool:
        return (
            other.key == self.key and
            (other.if_missing or not self.if_missing) and
            self.predicate._implies(other.predicate, True)
        )

    def _same_is_implied_by(self, other: 'KeyPredicate[U]') -> bool:
        return (
            other.key == self.key and
            (self.if_missing or not other.if_missing) and
            self.predicate._is_implied_by(other.predicate, True)
        )

    def _same_denies(self, other: 'KeyPredicate[U]') -> bool:
        return (
            other.key == self.key and
            (self.if_missing or not other.if_missing) and
            self.predicate._denies(other.predicate, True)
        )

    def _same_is_denied_by(self, other: 'KeyPredicate[U]') -> bool:
        return (
            other.key == self.key and
            (not self.if_missing or other.if_missing) and
            self.predicate._is_denied_by(other.predicate, True)
        )

    def describe(self, arg: str) -> str:
        return self.predicate.describe(f'{arg}[{self.key!r}]')


class TransformPredicate(Predicate[U]):

    __slots__ = ('transform', 'predicate')

    transform: 'Transform[U, Any]'
    predicate: Predicate[Any]

    def __init__(self, txf: 'Transform[U, X]', predicate: Predicate[X]):
        super().__init__(transform(txf.transform, predicate.evaluate))
        self.transform = txf
        self.predicate = predicate

    @property
    def not_evaluate(self) -> PredicateFunction[U]:
        return transform(self.transform, self.predicate.not_evaluate)

    def _eq_tuple(self) -> tuple:
        return self.transform, self.predicate

    def _same_implies(self, other: 'TransformPredicate[U]') -> bool:
        return (
            other.transform == self.transform and
            self.predicate._implies(other.predicate, True)
        )

    def _same_is_implied_by(self, other: 'TransformPredicate[U]') -> bool:
        return (
            other.transform == self.transform and
            self.predicate._is_implied_by(other.predicate, True)
        )

    def _same_denies(self, other: 'TransformPredicate[U]') -> bool:
        return (
            other.transform == self.transform and
            self.predicate._denies(other.predicate, True)
        )

    def _same_is_denied_by(self, other: 'TransformPredicate[U]') -> bool:
        return (
            other.transform == self.transform and
            self.predicate._is_denied_by(other.predicate, True)
        )

    def describe(self, arg: str) -> str:
        return self.predicate.describe(self.transform.describe(arg))


class IsInstanceOfAnyPredicate(AnyPredicate[Any]):

    __slots__ = ('cls',)

    cls: tuple[type, ...]

    def __init__(self, cls: tuple[type, ...]):
        predicates = tuple(IsInstancePredicate(c) for c in cls)
        super().__init__(predicates, is_instance(cls))
        self.cls = cls

    def _eq_tuple(self) -> tuple:
        return self.cls,

    def _same_implies(self, other: 'IsInstancePredicate',) -> bool:
        return ball(issubclass(cls, other.cls) for cls in self.cls)

    def _is_implied_by(self, other: Predicate[U], reverse: bool) -> bool:
        return super()._is_implied_by(other, reverse) or (
            isinstance(other, IsInstancePredicate) and
            issubclass(other.cls, self.cls)
        )

    def describe(self, arg: str) -> str:
        return f'isinstance({arg}, ({", ".join(map(class_qname, self.cls))}))'


class IsInstancePredicate(Predicate[Any]):

    __slots__ = ('cls',)

    cls: type

    def __init__(self, cls: type):
        super().__init__(is_instance(cls))
        self.cls = cls

    def _eq_tuple(self) -> tuple:
        return self.cls,

    def _same_implies(self, other: 'IsInstancePredicate',) -> bool:
        return issubclass(self.cls, other.cls)

    def _is_implied_by(self, other: Predicate[U], reverse: bool) -> bool:
        return super()._is_implied_by(other, reverse) or (
            isinstance(other, IsInstancePredicate) and
            issubclass(other.cls, self.cls)
        )

    def describe(self, arg: str) -> str:
        return f'isinstance({arg}, {class_qname(self.cls)})'


class DelegatePredicate(Predicate[U]):

    __slots__ = ('delegate', 'describe',)

    delegate: Predicate[U]
    describe: Callable[[str], str]

    def __init__(self, delegate: Predicate[U], describe: Callable[[str], str]):
        super().__init__(delegate.evaluate)
        self.delegate = delegate
        self.describe = describe

    def _implies(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return self.delegate._implies(other, reverse)

    def _is_denied_by(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return self.delegate._is_denied_by(other, reverse)

    def _is_implied_by(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return self.delegate._is_implied_by(other, reverse)

    def _denies(self, other: 'Predicate[U]', reverse: bool) -> bool:
        return self.delegate._denies(other, reverse)

    def _eq_tuple(self) -> tuple:
        return self.delegate._eq_tuple()


class CustomPredicate(Predicate[U]):

    __slots__ = ('_implies', '_denies', '_is_implied_by', '_is_denied_by', 'describe', 'eq_tuple', 'info')

    _implies: PredicateFunction[Predicate]
    _denies: PredicateFunction[Predicate]
    _is_implied_by: PredicateFunction[Predicate]
    _is_denied_by: PredicateFunction[Predicate]
    eq_tuple: tuple
    info: Any
    describe: Callable[[str], str]


    def __init__(self, evaluate: PredicateFunction[U], implies: PredicateFunction[Predicate] = None,
                 denies: PredicateFunction[Predicate] = None, is_implied_by: PredicateFunction[Predicate] = None,
                 is_denied_by: PredicateFunction[Predicate] = None,
                 eq_tuple: tuple = None,
                 describe: Callable[[Predicate, str], str] = None,
                 info: Any = None):
        super().__init__(evaluate)
        self.predicate = evaluate
        self._implies = Predicate._implies if implies is None else implies
        self._denies = Predicate._denies if denies is None else denies
        self._is_implied_by = Predicate._is_implied_by if is_implied_by is None else is_implied_by
        self._is_denied_by = Predicate._is_denied_by if is_denied_by is None else is_denied_by
        self.info = info
        self.eq_tuple = tuple(eq_tuple) if eq_tuple else ()
        if describe is None:
            self.describe = Predicate.describe
        else:
            self.describe = lambda arg: describe(self, arg)


    def _eq_tuple(self) -> tuple:
        return self.eq_tuple


def describe(pred: Predicate[U], desc: Callable[[str], str]) -> Predicate[U]:
    return DelegatePredicate(pred, desc)


class Predicates:

    __slots__ = ()

    @staticmethod
    def invert(evaluate: PredicateLike[U]) -> Predicate[U]:
        predicate = coerce(evaluate)
        return InversePredicate(predicate)

    @staticmethod
    def all(*predicates: PredicateLike[U]) -> Predicate[U]:
        return AllPredicate(map(coerce, predicates))

    @staticmethod
    def any(*predicates: PredicateLike[U]) -> Predicate[U]:
        return AnyPredicate(map(coerce, predicates))

    @staticmethod
    def xor(left: PredicateLike[U], right: PredicateLike[U]) -> Predicate[U]:
        return XorPredicate(coerce(left), coerce(right))

    @staticmethod
    def is_in(container: Container[X]) -> Predicate[X]:
        def pred(x: Container[X]) -> bool:
            return x in container
        base = Predicate(name_function(pred, f'is_in[{container!r}]'))
        return Predicates.is_str & describe(base, lambda p, arg: f'({arg} in {container!r})')

    @staticmethod
    def eq(value: X) -> Predicate[X]:
        return ComparePredicate(op.eq, value)

    @staticmethod
    def ne(value: X) -> Predicate[X]:
        return ComparePredicate(op.ne, value)

    @staticmethod
    def gt(value: X) -> Predicate[X]:
        return ComparePredicate(op.gt, value)

    @staticmethod
    def ge(value: X) -> Predicate[X]:
        return ComparePredicate(op.ge, value)

    @staticmethod
    def lt(value: X) -> Predicate[X]:
        return ComparePredicate(op.lt, value)

    @staticmethod
    def le(value: X) -> Predicate[X]:
        return ComparePredicate(op.le, value)

    @staticmethod
    def is_(value: X) -> Predicate[X]:
        return ComparePredicate(op.is_, value)

    @staticmethod
    def is_not(value: X) -> Predicate[X]:
        return ComparePredicate(op.is_not, value)

    @staticmethod
    def transform(txf: TransformLike[U, X], predicate: PredicateLike[X]) -> Predicate[U]:
        return TransformPredicate(infra.transforms.coerce(txf), coerce(predicate))

    @staticmethod
    def length(predicate: PredicateLike[int]) -> Predicate[Sized]:
        return Predicates.transform(infra.transforms.length, coerce(predicate))

    @staticmethod
    def contains(s: X) -> Predicate[Container[X]]:
        def pred(x: Container[X]) -> bool:
            return s in x
        base = Predicate(name_function(pred, f'contains[{s!r}]'))
        return Predicates.is_str & describe(base, lambda p, arg: f'({s!r} in {arg})')

    @staticmethod
    def matches(s: str|re.Pattern) -> Predicate[str]:
        if isinstance(s, re.Pattern):
            pat = s
            s = pat.pattern
        else:
            pat = re.compile(s)
        def pred(x: str) -> bool:
            return pat.match(x) is not None
        base = Predicate(name_function(pred, f'matches[/{s}/]'))
        return Predicates.is_str & describe(base, lambda p, arg: f'({arg} ~= /{s}/)')

    @staticmethod
    def starts_with(s: str) -> Predicate[str]:
        def pred(x: str) -> bool:
            return x is not None and x.startswith(s)
        base = Predicate(name_function(pred, f'starts_with[{s!r}]'))
        return Predicates.is_str & describe(base, lambda p, arg: f'{arg}.startswith({s!r})')

    @staticmethod
    def ends_with(s: str) -> Predicate[str]:
        def pred(x: str) -> bool:
            return x is not None and x.endswith(s)
        base = Predicate(name_function(pred, f'ends_with[{s!r}]'))
        return Predicates.is_str & describe(base, lambda p, arg: f'{arg}.endswith({s!r})')

    @staticmethod
    def every_n(n: int, offset: int = 0) -> Predicate[int]:
        if n == 1:
            return Predicates.always
        return Predicates.function(every_n(n, offset))

    @staticmethod
    def describe(pred: Predicate[U], describe: Callable[[str], str]) -> Predicate[U]:
        return DelegatePredicate(pred, describe)

    @staticmethod
    def with_attr(name: str, predicate: PredicateLike, if_missing: bool = False) -> Predicate:
        return AttrPredicate(name, coerce(predicate), if_missing)

    @staticmethod
    def with_key(name: str, predicate: PredicateLike, if_missing: bool = False) -> Predicate:
        return KeyPredicate(name, coerce(predicate), if_missing)

    @staticmethod
    def is_instance(*classes: str|type) -> Predicate:
        if len(classes) == 1:
            cls = classes[0]
            return IsInstancePredicate(meta_coerce_class(cls))
        if len(classes) == 0:
            return Predicates.never
        return IsInstanceOfAnyPredicate(tuple(meta_coerce_class(c) for c in classes))

    @staticmethod
    def coerce(spec: PredicateLike[U]) -> Predicate[U]:
        return coerce(spec)

    @staticmethod
    def function(evaluate: PredicateFunction[X], name: str = None) -> Predicate[X]:
        if name is not None:
            name_function(evaluate, name)
        return Predicate(evaluate)

    if TYPE_CHECKING:
        @staticmethod
        def custom(evaluate: PredicateLike[U], *, implies: PredicateFunction[Predicate] = None, denies: PredicateFunction[Predicate] = None, is_implied_by: PredicateFunction[Predicate] = None, is_denied_by: PredicateFunction[Predicate] = None, describe: Callable[['CustomPredicate', str], str] = None, info: Any = None) -> Predicate[U]: ...
    else:
        @staticmethod
        def custom(evaluate: PredicateLike[U], **kwargs) -> Predicate[U]:
            return custom(evaluate, **kwargs)

    @staticmethod
    def from_function(evaluate: PredicateFunction[U]) -> Predicate[U]:
        if callable(evaluate):
            return Predicate(evaluate)
        raise TypeError(f'Expected callable, got {evaluate!r}')

    @staticmethod
    def named(name: str) -> Predicate:
        t = getattr(Predicates, name)
        if isinstance(t, Predicate):
            return t
        raise ValueError(f"Invalid predicate name: {name}")

    @staticmethod
    def register(name: str, factory: TransformFunction[Any, Predicate], overwrite: bool = False) -> None:
        if not overwrite and name in named_factories:
            raise ValueError(f'Predicate factory for {name} already exists')
        named_factories[name] = factory

    always: Predicate[Any] = AlwaysPredicate()
    never: Predicate[Any] = NeverPredicate()
    is_true: Predicate[Any] = IsTruePredicate()
    is_false: Predicate[Any] = IsFalsePredicate()
    is_none: Predicate[Any] = is_(None)
    not_none: Predicate[Any] = ~is_none
    is_str: Predicate[Any] = IsInstancePredicate(str)
    is_int: Predicate[Any] = IsInstancePredicate(int)
    is_float: Predicate[Any] = IsInstancePredicate(float)
    is_bool: Predicate[Any] = IsInstancePredicate(bool)
    PredicateFunction: ClassVar[type[Predicate]] = Predicate


def custom(evaluate: PredicateLike[U] = None, implies: PredicateFunction[Predicate] = None, denies: PredicateFunction[Predicate] = None, is_implied_by: PredicateFunction[Predicate] = None, is_denied_by: PredicateFunction[Predicate] = None, describe: Callable[['CustomPredicate', str], str] = None, info: Any = None) -> Predicate[U]:
    if evaluate is None: raise ValueError("evaluate cannot be None")
    return CustomPredicate(evaluate, implies=implies, denies=denies, is_implied_by=is_implied_by, is_denied_by=is_denied_by, describe=describe, info=info)


def build_lambda(expr: str) -> Predicate:
    pred: Callable = eval('lambda x: bool(' + expr + ')', globals(), {})
    if not callable(pred):
        raise ValueError(f'Invalid lambda expression: {expr}')
    return Predicates.function(pred)


named_factories: dict[str, TransformFunction[Any, Predicate]] = {
    'gt': Predicates.gt,
    'ge': Predicates.ge,
    'lt': Predicates.lt,
    'le': Predicates.le,
    'eq': Predicates.eq,
    'ne': Predicates.ne,
    'is': Predicates.is_,
    'is_not': Predicates.is_not,
    'isnt': Predicates.is_not,
    'all': all_factory,
    'any': any_factory,
    'not': not_factory,
    'attr': spread_mapping(Predicates.with_attr),
    'key': spread_mapping(Predicates.with_key),
    'contains': Predicates.contains,
    'matches': Predicates.matches,
    'starts_with': Predicates.starts_with,
    'ends_with': Predicates.ends_with,
    'startswith': Predicates.starts_with,
    'endswith': Predicates.ends_with,
    'every_n': Predicates.every_n,
    'custom': spread_mapping(custom),
    'lambda': build_lambda,
}

named_singletons: dict[str, Predicate[Any]] = {
    name: pred for name, pred in Predicates.__dict__.items() if isinstance(pred, Predicate)
}


__all__ = [
    'Predicates',
    'Predicate',
    'always',
    'contains',
    'ends_with',
    'invert',
    'is_instance',
    'never',
    'starts_with',
    'xor',
]
