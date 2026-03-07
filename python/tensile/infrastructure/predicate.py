#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from typing import Any, Callable, Container, Generic, Iterable, Mapping, Sequence, TypeVar
import re
import operator as op
from builtins import all as ball, any as bany

from .types import Comparison, Predicate, Transform, missing
from .meta import provides
from .util import class_qname, name_function

X = TypeVar('X')
U = TypeVar('U', contravariant=True)
P = TypeVar('P', bound=Predicate)

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


def is_instance(cls: type) -> Predicate:
    def pred(value) -> bool:
        return isinstance(value, cls)
    return name_function(pred, f'is_instance[{class_qname(cls)}]')


def starts_with(prefix: str) -> Predicate[str]:
    def pred(value: str) -> bool:
        return value is not None and value.startswith(prefix)
    return name_function(pred, f'starts_with[{prefix}]')


def ends_with(suffix: str) -> Predicate[str]:
    def pred(value: str) -> bool:
        return value is not None and value.endswith(suffix)
    return name_function(pred, f'ends_with[{suffix}]')


def contains(part: str) -> Predicate[str]:
    def pred(value: str) -> bool:
        return value is not None and value.find(part) >= 0
    return name_function(pred, f'contains[{part}]')


def matches(pattern: str) -> Predicate[str]:
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


def transform(txf: Transform[X, U], predicate: Predicate[U]) -> Predicate[X]:
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


def has_attr(name: str) -> Predicate:
    def has_attr_pred(obj: Any) -> bool:
        value = getattr(obj, name, missing)
        return value is not missing
    return name_function(has_attr_pred, f'has_attr[{name}]')


def attr(name: str, pred: Predicate, if_missing: bool = False) -> Predicate:
    if pred is never:
        return invert(has_attr(name)) if if_missing else never
    if pred is always:
        return always if if_missing else has_attr(name)
    def attr_pred(obj: Any) -> bool:
        value = getattr(obj, name, missing)
        return if_missing if value is missing else pred(value)
    return name_function(attr_pred, f'attr[{name}, {pred.__name__}]')


def attrs(if_missing: bool = False, /, **preds: Predicate) -> Predicate[Mapping]:
    return all(*(attr(n, p, if_missing=if_missing) for n, p in preds.items()))


def has(item: X) -> Predicate[Container[X]]:
    def pred(obj: Container) -> bool:
        return item in obj
    return name_function(pred, f'has[{item}]')


def has_not(item: X) -> Predicate[Container[X]]:
    def pred(obj: Container) -> bool:
        return item not in obj
    return name_function(pred, f'has_not[{item}]')


# noinspection PyShadowingNames
def key(key: str, pred: Predicate, if_missing: bool = False) -> Predicate[Mapping]:
    if pred is never:
        return has_not(key) if if_missing else never
    if pred is always:
        return always if if_missing else has(key)
    def key_pred(obj: Any) -> bool:
        value = obj.get(key, missing)
        return if_missing if value is missing else pred(value)
    return name_function(key_pred, f'key[{key!r}, {pred.__name__}]')


def keys(if_missing: bool = False, /, **preds: Predicate) -> Predicate[Mapping]:
    return all(*(key(k, p, if_missing=if_missing) for k, p in preds.items()))


def any_item(pred: Predicate[U]) -> Predicate[Iterable[U]]:
    if pred is never: return never
    def any_item_pred(items: Iterable) -> bool:
        for item in items:
            if pred(item):
                return True
        return False
    return any_item_pred


def all_items(pred: Predicate[U]) -> Predicate[Iterable[U]]:
    if pred is always: return always
    def any_item_pred(items: Iterable) -> bool:
        for item in items:
            if not pred(item):
                return False
        return True
    return any_item_pred


@provides(Predicate, 'default')
def coerce(spec: Any) -> 'PredicateObject[Any]':
    if isinstance(spec, PredicateObject): return spec
    if spec is always or spec is None: return Predicates.always
    if spec is never: return Predicates.never
    if callable(spec):
        # noinspection PyTypeChecker
        return PredicateObject(spec)
    if isinstance(spec, Mapping):
        spec_keys = list(spec.keys())
        if len(spec_keys) == 1:
            k = spec_keys[0]
            value = spec[k]
            if factory := named_factories.get(k):
                return factory(value)
            raise ValueError(f"Invalid key: {k!r}")
    if isinstance(spec, Sequence):
        return all_factory(spec)
    raise ValueError(f"Invalid predicate spec: {spec}")


class PredicateObject(Generic[U]):

    __slots__ = ('evaluate',)

    evaluate: Predicate[U]
    not_evaluate: Predicate[U]

    def __init__(self, evaluate: Predicate[U]):
        self.evaluate = evaluate

    @property
    def is_always(self) -> bool:
        return self.evaluate is always

    @property
    def is_never(self) -> bool:
        return self.evaluate is never

    @property
    def __name__(self) -> str:
        return self.evaluate.__name__

    @property
    def not_evaluate(self) -> Predicate[U]:
        return invert(self.evaluate)

    def describe(self, arg: str) -> str:
        return f'{self.evaluate.__name__}({arg})'

    def implies(self, other: Predicate[U]) -> bool:
        return self._implies(coerce(other), True)

    def is_implied_by(self, other: Predicate[U]) -> bool:
        return self._is_implied_by(coerce(other), True)

    def denies(self, other: Predicate[U]) -> bool:
        return self._denies(coerce(other), True)

    def is_denied_by(self, other: Predicate[U]) -> bool:
        return self._is_denied_by(coerce(other), True)

    def _implies(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_never or
            (isinstance(other, self.__class__) and self._same_implies(other)) or
            (reverse and other._is_implied_by(self, False))
        )

    def _same_implies(self, other: 'PredicateObject[U]') -> bool:
        return False

    def _is_implied_by(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_always or
            (isinstance(other, self.__class__) and self._same_is_implied_by(other)) or
            (reverse and other._implies(self, False))
        )

    def _same_is_implied_by(self, other: 'PredicateObject[U]') -> bool:
        return False

    def _denies(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and (
                self.is_never or (
                (isinstance(other, self.__class__) and self._same_denies(other)) or
                (reverse and other._is_denied_by(self, False))
            ))
        )

    def _same_denies(self, other: 'PredicateObject[U]') -> bool:
        return False

    def _is_denied_by(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            not self.is_always and (
                (isinstance(other, self.__class__) and self._same_is_denied_by(other)) or
                (reverse and other._denies(self, False))
            )
        )

    def _same_is_denied_by(self, other: 'PredicateObject[U]') -> bool:
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

    def __and__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return Predicates.all(self, other)

    def __rand__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return Predicates.all(other, self)

    def __or__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return Predicates.any(self, other)

    def __ror__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return Predicates.any(other, self)

    def __xor__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return Predicates.xor(self, other)

    def __rxor__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return Predicates.xor(other, self)

    def __invert__(self) -> 'PredicateObject[U]':
        return InversePredicate(self)

    def __repr__(self):
        return 'Predicate(' + self.describe('x') + ')'


# def coerce(evaluate: Optional[Predicate[U]]) -> PredicateObject[U]:
#     if evaluate is always or evaluate is None: return Predicates.always
#     if evaluate is never: return Predicates.never
#     if isinstance(evaluate, PredicateObject): return evaluate
#     if callable(evaluate): return PredicateObject(evaluate)
#     raise ValueError(f"Invalid predicate function: {evaluate}")


def all_factory(predicates: Iterable[Any]) -> PredicateObject:
    return Predicates.all(*(Predicates.coerce(p) for p in predicates))


def any_factory(predicates: Iterable[Any]) -> PredicateObject:
    return Predicates.any(*(Predicates.coerce(p) for p in predicates))


def not_factory(predicate: Any) -> PredicateObject:
    return Predicates.invert(Predicates.coerce(predicate))


short_circuit = False
short_circuit_not = short_circuit
short_circuit_and = short_circuit
short_circuit_or = short_circuit


class AlwaysPredicate(PredicateObject[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(always)

    @property
    def not_evaluate(self) -> Predicate[U]:
        return never

    def __and__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return coerce(other) if short_circuit_and else super().__and__(other)

    def __rand__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return coerce(other) if short_circuit_and else super().__rand__(other)

    def __or__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return self if short_circuit_or else super().__or__(other)

    def __ror__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return self if short_circuit_or else super().__ror__(other)

    def __invert__(self) -> 'PredicateObject[U]':
        return Predicates.never if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return 'always'

    is_always = True
    is_never = False


class NeverPredicate(PredicateObject[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(never)

    @property
    def not_evaluate(self) -> Predicate[U]:
        return always

    def __and__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return self if short_circuit_and else super().__and__(other)

    def __rand__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return self if short_circuit_and else super().__rand__(other)

    def __or__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return coerce(other) if short_circuit_or else super().__or__(other)

    def __ror__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        return coerce(other) if short_circuit_or else super().__ror__(other)

    def __invert__(self) -> 'PredicateObject[U]':
        return Predicates.always if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return 'never'

    is_always = False
    is_never = True


class IsTruePredicate(PredicateObject[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(is_true)

    @property
    def not_evaluate(self) -> Predicate[U]:
        return is_false

    def _denies(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return super()._denies(other, reverse) or other.evaluate is is_false

    def _is_denied_by(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return super()._is_denied_by(other, reverse) or other.evaluate is is_false

    def __invert__(self) -> 'PredicateObject[U]':
        return Predicates.is_false if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return f'bool({arg})'


class IsFalsePredicate(PredicateObject[Any]):

    __slots__ = ()

    def __init__(self):
        super().__init__(is_false)

    @property
    def not_evaluate(self) -> Predicate[U]:
        return is_true

    def _denies(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return super()._denies(other, reverse) or other.evaluate is is_true

    def _is_denied_by(self, other: 'PredicateObject[U]', reverse: bool) -> bool:
        return super()._is_denied_by(other, reverse) or other.evaluate is is_true

    def __invert__(self) -> 'PredicateObject[U]':
        return Predicates.is_true if short_circuit_not else super().__invert__()

    def describe(self, arg: str) -> str:
        return f'not bool({arg})'



class InversePredicate(PredicateObject[U]):

    __slots__ = ('inverse',)

    inverse: PredicateObject[U]

    def __init__(self, inverse: PredicateObject[U]):
        super().__init__(inverse.not_evaluate)
        self.inverse = inverse

    @property
    def not_evaluate(self) -> Predicate[U]:
        return self.inverse.evaluate

    def _eq_tuple(self) -> tuple:
        return self.inverse,

    def _denies(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            self.is_never and (
                self.inverse._is_implied_by(other, True) or
                (reverse and other._is_denied_by(self, False))
            )
        )

    def __invert__(self) -> 'PredicateObject[U]':
        return self.inverse if short_circuit_not else InversePredicate(self)

    def describe(self, arg: str) -> str:
        return '~' + self.inverse.describe(arg)
        # return 'not(' + self.inverse.describe(arg) + ')'


def _combine_all(preds: Sequence[PredicateObject[U]]) -> Predicate[U]:
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


def _combine_any(preds: Sequence[PredicateObject[U]]) -> Predicate[U]:
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


class AllPredicate(PredicateObject[U]):

    __slots__ = ('predicates', )

    predicates: tuple[PredicateObject[U], ...]

    def __init__(self, predicates: Iterable[PredicateObject[U]]):
        predicates = tuple(predicates)
        evaluate = _combine_all(predicates)
        super().__init__(evaluate)
        self.predicates = predicates

    def _eq_tuple(self) -> tuple:
        return self.predicates

    def _implies(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_never or
            bany(p._implies(other, True) for p in self.predicates) or
            (reverse and other._is_implied_by(self, False))
        )

    def _is_implied_by(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_always or
            ball(p._is_implied_by(other, True) for p in self.predicates) or
            (reverse and other._implies(self, False))
        )

    def _denies(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            self.is_never and (
                bany(p._denies(other, True) for p in self.predicates) or
                (reverse and other._is_denied_by(self, False))
            )
        )

    def _is_denied_by(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            not self.is_always and (
                bany(p._is_denied_by(other, True) for p in self.predicates) or
                (reverse and other._denies(self, False))
            )
        )

    def __and__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        if short_circuit_and:
            if isinstance(other, AllPredicate):
                return Predicates.all(*self.predicates, *other.predicates)
            else:
                return Predicates.all(*self.predicates, other)
        else:
            return super().__and__(other)

    def __rand__(self, other: Predicate[U]) -> 'PredicateObject[U]':
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


class AnyPredicate(PredicateObject[U]):

    __slots__ = ('predicates', )

    predicates: tuple[PredicateObject[U], ...]

    def __init__(self, predicates: Iterable[PredicateObject[U]]):
        predicates = tuple(predicates)
        evaluate = _combine_any(predicates)
        super().__init__(evaluate)
        self.predicates = predicates

    def _eq_tuple(self) -> tuple:
        return self.predicates

    def _implies(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_never or
            ball(p._implies(other, True) for p in self.predicates) or
            (reverse and other._is_implied_by(self, False))
        )

    def _is_implied_by(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is other or
            self.evaluate is other.evaluate or
            self.is_always or
            bany(p._is_implied_by(other, True) for p in self.predicates) or
            (reverse and other._implies(self, False))
        )

    def _denies(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            self.is_never and (
                ball(p._denies(other, True) for p in self.predicates) or
                (reverse and other._is_denied_by(self, False))
            )
        )

    def _is_denied_by(self, other: PredicateObject[U], reverse: bool) -> bool:
        return (
            self is not other and
            self.evaluate is not other.evaluate and
            not self.is_always and (
                ball(p._is_denied_by(other, True) for p in self.predicates) or
                (reverse and other._denies(self, False))
            )
        )

    def __or__(self, other: Predicate[U]) -> 'PredicateObject[U]':
        if short_circuit_or:
            if isinstance(other, AnyPredicate):
                return Predicates.any(*self.predicates, *other.predicates)
            else:
                return Predicates.any(*self.predicates, other)
        else:
            return super().__or__(other)

    def __ror__(self, other: Predicate[U]) -> 'PredicateObject[U]':
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


class XorPredicate(PredicateObject[U]):

    __slots__ = ('left', 'right')

    left: PredicateObject[U]
    right: PredicateObject[U]

    def __init__(self, left: PredicateObject[U], right: PredicateObject[U]):
        if left.denies(right) or right.denies(left):
            evaluate = always
        elif left.implies(right) or right.implies(left):
            evaluate = never
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

class ComparePredicate(PredicateObject[U]):

    __slots__ = ('compare', 'arg')

    compare: Comparison[U]
    arg: U

    def __init__(self, compare: Comparison[U], arg: U):
        if compare not in compare_symbols:
            raise ValueError(f"Invalid comparison operator: {compare}")
        def evaluate(value: U) -> bool:
            return compare(value, arg)
        evaluate = name_function(evaluate, 'compare[' + compare_symbols[compare] + f' {arg}]')
        super().__init__(evaluate)
        self.compare = compare
        self.arg = arg
        super().__init__(evaluate)
        self.compare = compare
        self.arg = arg

    @property
    def not_evaluate(self) -> Predicate[U]:
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

    def describe(self, arg: str) -> str:
        return f'({arg} {compare_symbols[self.compare]} {self.arg})'


class AttrPredicate(PredicateObject[U]):

    __slots__ = ('name', 'predicate', 'if_missing')

    name: str
    predicate: PredicateObject[U]
    if_missing: bool

    def __init__(self, name: str, predicate: PredicateObject[U], if_missing: bool = False):
        evaluate = attr(name, predicate.evaluate, if_missing=if_missing)
        super().__init__(evaluate)
        self.name = name
        self.predicate = predicate
        self.if_missing = bool(if_missing)

    @property
    def not_evaluate(self) -> Predicate[U]:
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
        return self.describe(f'{arg}.{self.name}')


class KeyPredicate(PredicateObject[U]):

    __slots__ = ('key', 'predicate', 'if_missing')

    key: str
    predicate: PredicateObject[U]
    if_missing: bool

    def __init__(self, name: str, predicate: PredicateObject[U], if_missing: bool = False):
        evaluate = key(name, predicate.evaluate, if_missing=if_missing)
        super().__init__(evaluate)
        self.key = name
        self.predicate = predicate
        self.if_missing = if_missing

    @property
    def not_evaluate(self) -> Predicate[U]:
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


class TransformPredicate(PredicateObject[U]):

    __slots__ = ('transform', 'predicate')

    transform: Transform[U, Any]
    predicate: PredicateObject[Any]

    def __init__(self, txf: Transform[U, X], predicate: PredicateObject[X]):
        super().__init__(transform(txf, predicate.evaluate))
        self.transform = txf
        self.predicate = predicate

    @property
    def not_evaluate(self) -> Predicate[U]:
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
        return self.predicate.describe(f'{self.transform.__name__}({arg})')



class IsInstancePredicate(PredicateObject[Any]):

    __slots__ = ('cls',)

    cls: type

    def __init__(self, cls: type):
        super().__init__(is_instance(cls))
        self.cls = cls

    def _eq_tuple(self) -> tuple:
        return self.cls,

    def _same_implies(self, other: 'IsInstancePredicate',) -> bool:
        return issubclass(self.cls, other.cls)

    def _is_implied_by(self, other: PredicateObject[U], reverse: bool) -> bool:
        return super()._is_implied_by(other, reverse) or (
            isinstance(other, IsInstancePredicate) and
            issubclass(other.cls, self.cls)
        )

    def describe(self, arg: str) -> str:
        return f'isinstance({arg}, {class_qname(self.cls)})'


class Predicates:

    __slots__ = ()

    @staticmethod
    def invert(evaluate: Predicate[U]) -> PredicateObject[U]:
        predicate = coerce(evaluate)
        return InversePredicate(predicate)

    @staticmethod
    def all(*predicates: Predicate[U]) -> PredicateObject[U]:
        return AllPredicate(map(coerce, predicates))

    @staticmethod
    def any(*predicates: Predicate[U]) -> PredicateObject[U]:
        return AnyPredicate(map(coerce, predicates))

    @staticmethod
    def xor(left: Predicate[U], right: Predicate[U]) -> PredicateObject[U]:
        return XorPredicate(coerce(left), coerce(right))

    @staticmethod
    def eq(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.eq, value)

    @staticmethod
    def ne(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.ne, value)

    @staticmethod
    def gt(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.gt, value)

    @staticmethod
    def ge(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.ge, value)

    @staticmethod
    def lt(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.lt, value)

    @staticmethod
    def le(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.le, value)

    @staticmethod
    def is_(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.is_, value)

    @staticmethod
    def is_not(value: X) -> PredicateObject[X]:
        return ComparePredicate(op.is_not, value)

    @staticmethod
    def transform(txf: Transform[U, X], predicate: Predicate[X]) -> PredicateObject[U]:
        return TransformPredicate(txf, coerce(predicate))

    @staticmethod
    def contains(s: str) -> PredicateObject[str]:
        def pred(x: str) -> bool:
            return s in x
        return Predicates.is_str & PredicateObject(name_function(pred, f'contains[{s!r}]'))

    @staticmethod
    def matches(s: str) -> PredicateObject[str]:
        pat = re.compile(s)
        def pred(x: str) -> bool:
            return pat.match(x) is not None
        return Predicates.is_str & PredicateObject(name_function(pred, f'matches[/{s}/]'))

    @staticmethod
    def starts_with(s: str) -> PredicateObject[str]:
        def pred(x: str) -> bool:
            return x is not None and x.startswith(s)
        return Predicates.is_str & PredicateObject(name_function(pred, f'starts_with[{s!r}]'))

    @staticmethod
    def ends_with(s: str) -> PredicateObject[str]:
        def pred(x: str) -> bool:
            return x is not None and x.endswith(s)
        return Predicates.is_str & PredicateObject(name_function(pred, f'ends_with[{s!r}]'))

    @staticmethod
    def with_attr(name: str, pred: Predicate, if_missing: bool = False) -> PredicateObject:
        return AttrPredicate(name, coerce(pred), if_missing)

    @staticmethod
    def with_key(name: str, pred: Predicate, if_missing: bool = False) -> PredicateObject:
        return KeyPredicate(name, coerce(pred), if_missing)

    @staticmethod
    def is_instance(cls: type) -> PredicateObject:
        return IsInstancePredicate(cls)

    coerce = staticmethod(coerce)

    @staticmethod
    def predicate(evaluate: Predicate[U]) -> PredicateObject[U]:
        return coerce(evaluate)

    @staticmethod
    def from_function(evaluate: Predicate[U]) -> PredicateObject[U]:
        if callable(evaluate):
            return PredicateObject(evaluate)
        raise TypeError(f'Expected callable, got {evaluate!r}')

    always: PredicateObject[Any] = AlwaysPredicate()
    never: PredicateObject[Any] = NeverPredicate()
    is_true: PredicateObject[Any] = IsTruePredicate()
    is_false: PredicateObject[Any] = IsFalsePredicate()
    is_none: PredicateObject[Any] = is_(None)

    is_str: PredicateObject[Any] = IsInstancePredicate(str)


named_factories: dict[str, Callable[[Any], PredicateObject[Any]]] = {
    'gt': Predicates.gt,
    'ge': Predicates.ge,
    'lt': Predicates.lt,
    'le': Predicates.le,
    'eq': Predicates.eq,
    'ne': Predicates.ne,
    'is': Predicates.is_,
    'is_not': Predicates.is_not,
    'all': all_factory,
    'any': any_factory,
    'not': not_factory,
}

named: dict[str, PredicateObject[Any]] = {
    'always': Predicates.always,
    'never': Predicates.never,
    'is_str': Predicates.is_str,
}


def test():

    def test_p(ap):
        print('-' * 80)
        print(ap)
        print(ap.evaluate.__name__)

    def test_pa(p, *args):
        test_p(p)
        for arg in args:
            print('x =', arg)
            print(p.describe('x'), '>', p(arg))

    def test_all(*p):
        ap = Predicates.all(*p)
        test_p(ap)

    def test_any(*p):
        ap = Predicates.any(*p)
        test_p(ap)

    is_str = Predicates.is_instance(str)
    is_int = Predicates.is_instance(int)
    is_float = Predicates.is_instance(float)
    is_bool = Predicates.is_instance(bool)

    gt_40 = Predicates.gt(40)
    eq_23 = Predicates.eq(23)
    gt_10 = Predicates.gt(10)
    ge_11 = Predicates.ge(11)
    gt_50 = Predicates.gt(50)
    lt_40 = Predicates.lt(40)
    lt_10 = Predicates.lt(10)
    le_10 = Predicates.le(10)

    test_all(gt_40, gt_10, ge_11, gt_50)
    test_any(gt_40, gt_10, ge_11, gt_50)
    test_all(lt_10, le_10)
    test_any(lt_10, le_10)
    test_all(gt_40, lt_10)
    test_any(gt_40, lt_10)
    test_all(gt_10, lt_40, Predicates.always)
    test_any(gt_10, lt_40, Predicates.always)

    test_all(is_int, is_bool)
    test_any(is_int, is_bool)

    test_p(gt_10 | ~gt_50 | ~eq_23)

    test_p(~~gt_10)

    test_p(Predicates.always | Predicates.never)
    test_p(Predicates.always & Predicates.never)
    test_p(~Predicates.never | Predicates.always)
    test_p(~Predicates.never | Predicates.never)
    test_p(~Predicates.never & Predicates.always)
    test_p(is_str & is_str)

    test_p(Predicates.all(is_int, gt_10, is_bool))

    test_pa(~gt_10, 10)
    test_pa(ge_11, 11)

    g = [1, 2, 3]
    h = [
        {'foo': 'bar'},
        {'foo': 2},
        {'foo': 'bat'},
    ]

    test_pa(Predicates.with_key('foo', is_str & Predicates.starts_with('ba')), *h)
    test_pa(Predicates.with_key('foo', is_str & Predicates.matches('b.[tu]')), *h)

    foo = Predicates.coerce([gt_50, gt_10, ge_11])
    test_p(foo)

    test_p(Predicates.is_true & Predicates.is_false)
    test_p(Predicates.is_true | Predicates.is_false)
    test_p(Predicates.eq(100) & Predicates.eq(200))
    test_p(Predicates.eq(100) | Predicates.eq(200))

    exit(0)


# test()

__all__ = [
    'Predicate',
    'Predicates',
    'always',
    'contains',
    'ends_with',
    'invert',
    'is_instance',
    'never',
    'starts_with',
    'xor',
]
