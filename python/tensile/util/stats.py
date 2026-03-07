#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from typing import Any, Callable, Iterable

from ..infrastructure import function
from .. import ten, Array

Stat = Callable[[ten.Array], ten.Scalar|ten.Array]

ArrayStat = Callable[[ten.Array], ten.Array]

ScalarStat = Callable[[ten.Array], ten.Scalar]


def make_scalar_stat(stat: Callable[..., ten.ArrayOrScalar], *args, **kwargs) -> ScalarStat:
    def stat_fn(x: ten.Array, *args, **kwargs) -> ten.Scalar:
        return stat(x).item()
    return stat_fn


default_stats: dict[str, Stat] = {
    "l2": make_scalar_stat(ten.norm),
    "max_abs": make_scalar_stat(function.compose(ten.abs, ten.max)),
    "mean_abs": make_scalar_stat(function.compose(ten.abs, ten.mean)),
}

all_stats: dict[str, Stat] = {
    **default_stats
}


def _stat_dict(names: tuple[str, ...], stats: dict[str, Stat]) -> dict[str, Stat]:
    if stats:
        if names:
            stats.update({name: all_stats[name] for name in names})
    elif names:
        stats = {name: all_stats[name] for name in names}
    else:
        stats = default_stats
    return stats


def _array_stat(x: ten.Array, stats: dict[str, Stat]) -> dict[str, Any]:
    return {n: s(ten.detach(x)) for n, s in stats.items()} if stats else {}


def get_stats(arrays: Array|dict[str, Array]|Iterable[tuple[str, Array]], *names: str, **stats: ScalarStat) -> dict[str, Any]:
    stats = _stat_dict(names, stats)
    if ten.is_array(arrays): return _array_stat(arrays, stats)
    if isinstance(arrays, dict): arrays = arrays.items()
    out = {}
    for name, g in arrays:
        out[name] = _array_stat(g, stats)
    return out

