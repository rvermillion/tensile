
from tensile.infrastructure import predicates

gt_40 = predicates.gt(40)
eq_23 = predicates.eq(23)
gt_10 = predicates.gt(10)
ge_11 = predicates.ge(11)
gt_50 = predicates.gt(50)
lt_40 = predicates.lt(40)
lt_10 = predicates.lt(10)
le_10 = predicates.le(10)
gt_5 = predicates.gt(5)

gt_foo = predicates.gt('foo')
ge_foo = predicates.ge('foo')


def test_compare():

    assert gt_10(11)
    assert not gt_10(10)
    assert not gt_10(0)
    assert ge_foo('foo')
    assert gt_foo('food')
    assert not gt_foo('bar')


def test_str():

    is_str = predicates.is_str
    isnt_str = ~is_str
    contains_foo = predicates.contains('foo')
    startswith_foo = predicates.starts_with('foo')
    endswith_foo = predicates.ends_with('foo')
    matches_fo_d = predicates.matches('fo.d')

    assert is_str('foo')
    assert not is_str(123)
    assert isnt_str(123)
    assert not isnt_str('bar')
    assert contains_foo('food')
    assert startswith_foo('food')
    assert not endswith_foo('food')
    assert endswith_foo('snafoo')
    assert matches_fo_d('food')
    assert matches_fo_d('foid')

def test_tuple():

    gt_10_2 = predicates.with_key(2, gt_10)

    longer_than_5 = predicates.length(gt_5)

    a = list(range(10))
    b = tuple(range(5))

    assert gt_10_2((7, 4, 11))
    assert not gt_10_2((7, 4, 9))
    assert not gt_10_2((7, 4))
    assert longer_than_5(a)
    assert not longer_than_5(b)


def test_implication():

    p = gt_40 & lt_10
    assert p.is_never

    p = gt_40 & gt_10
    assert p.evaluate is gt_40.evaluate

    p = gt_40 | gt_10
    assert p.evaluate is gt_10.evaluate

    p = predicates.is_int & predicates.is_bool
    assert p.evaluate is predicates.is_bool.evaluate


def old():

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
        ap = predicates.all(*p)
        test_p(ap)

    def test_any(*p):
        ap = predicates.any(*p)
        test_p(ap)

    is_str = predicates.is_instance(str)
    is_int = predicates.is_instance(int)
    is_float = predicates.is_instance(float)
    is_bool = predicates.is_instance(bool)

    gt_40 = predicates.gt(40)
    eq_23 = predicates.eq(23)
    gt_10 = predicates.gt(10)
    ge_11 = predicates.ge(11)
    gt_50 = predicates.gt(50)
    lt_40 = predicates.lt(40)
    lt_10 = predicates.lt(10)
    le_10 = predicates.le(10)

    test_all(gt_40, gt_10, ge_11, gt_50)
    test_any(gt_40, gt_10, ge_11, gt_50)
    test_all(lt_10, le_10)
    test_any(lt_10, le_10)
    test_all(gt_40, lt_10)
    test_any(gt_40, lt_10)
    test_all(gt_10, lt_40, predicates.always)
    test_any(gt_10, lt_40, predicates.always)

    test_all(is_int, is_bool)
    test_any(is_int, is_bool)

    test_p(gt_10 | ~gt_50 | ~eq_23)

    test_p(~~gt_10)

    test_p(predicates.always | predicates.never)
    test_p(predicates.always & predicates.never)
    test_p(~predicates.never | predicates.always)
    test_p(~predicates.never | predicates.never)
    test_p(~predicates.never & predicates.always)
    test_p(is_str & is_str)

    test_p(predicates.all(is_int, gt_10, is_bool))

    test_pa(~gt_10, 10)
    test_pa(ge_11, 11)

    g = [1, 2, 3]
    h = [
        {'foo': 'bar'},
        {'foo': 2},
        {'foo': 'bat'},
    ]

    test_pa(predicates.with_key('foo', is_str & predicates.starts_with('ba')), *h)
    test_pa(predicates.with_key('foo', is_str & predicates.matches('b.[tu]')), *h)

    foo = predicates.coerce([gt_50, gt_10, ge_11])
    test_p(foo)

    test_p(predicates.is_true & predicates.is_false)
    test_p(predicates.is_true | predicates.is_false)
    test_p(predicates.eq(100) & predicates.eq(200))
    test_p(predicates.eq(100) | predicates.eq(200))

    exit(0)


# test()
