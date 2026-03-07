#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import json
import re
from pathlib import Path
from typing import Annotated, Any, Self

from ..infrastructure import Object, field
from .. import ten
from .stats import get_stats
from ..infrastructure import Predicate, Predicates


CompareResult = str|dict[str, 'CompareResult']|None


# eps: float = 0.
default_eps: float = 1e-7

default_ignore_keys: set[str] = {'start_time', 'end_time'}


def default_ignore_key(key: str) -> bool:
    return key in default_ignore_keys or key.startswith('_')


class Comparison(Object):

    __slots__ = ('left', 'right', 'lname', 'rname', 'eps', 'ignore_key', 'ignore_file', 'ignore_path')

    left: Annotated[Path, field()]
    right: Annotated[Path, field()]
    lname: Annotated[str, field()]
    rname: Annotated[str, field()]
    eps: Annotated[float, field(default=default_eps)]
    ignore_key: Annotated[Predicate[str], field(default=default_ignore_key)]
    ignore_file: Annotated[Predicate[str], field(default=Predicates.never)]
    ignore_path: Annotated[Predicate[str], field(default=Predicates.never)]

    def _lazy_lname(self):
        return str(self.left)

    def _lazy_rname(self):
        return str(self.right)

    def compare_tensors(self, left: ten.Array, right: ten.Array, path: str) -> CompareResult:
        if left.shape != right.shape:
            return f"{self.lname}.shape != {self.rname}.shape: {left.shape} != {right.shape}"
        if ten.allclose(left, right):
            return None
        else:
            return {
                self.lname: get_stats(left),
                self.rname: get_stats(right),
                'delta': get_stats(left-right)
            }

    def compare_float(self, left: float, right: float, path: str) -> CompareResult:
        delta = left - right
        if abs(delta) > self.eps:
            return {
                # 'desc': f"{self.lname} is not within {self.eps} of {self.rname}: {left} - {right} = {delta}"
                self.lname: left,
                self.rname: right,
                'delta': delta,
                # 'eps': self.eps,
            }
        return None

    def compare_any(self, left: Any, right: Any, path: str, key_prefix: str = '.') -> CompareResult:
        if left is None:
            if right is None:
                return None
            return f'{path}: missing in {self.lname}'
        elif right is None:
            return f'{path}: missing in {self.rname}'
        elif ten.is_array(left) and ten.is_array(right):
            return self.compare_tensors(left, right, path)
        elif type(left) == type(right):
            if isinstance(left, dict):
                return self.compare_dict(left, right, path, key_prefix=key_prefix)
            elif isinstance(left, list):
                return self.compare_list(left, right, path)
            elif isinstance(left, float):
                return self.compare_float(left, right, path)

        if left == right:
            return None
        return f"{self.lname} is not equal to {self.rname}: {left} != {right}"

    def compare_dict(self, left: dict[str, Any], right: dict[str, Any], path: str, key_prefix: str = '.') -> CompareResult:
        out = {}
        if left or right:
            keys = sorted(left.keys() | right.keys())
            for k in keys:
                if self.ignore_key(k):
                    continue
                kpath = f'{path}{key_prefix}{k}'
                if self.ignore_path(kpath):
                    continue
                if k in left:
                    if k in right:
                        if r := self.compare_any(left[k], right[k], kpath):
                            out[k] = r
                    else:
                        out[k] = f'{kpath}: missing in {self.rname}'
                else:
                    out[k] = f'{kpath}: missing in {self.lname}'
        return out

    def compare_list(self, left: list[Any], right: list[Any], path: str) -> CompareResult:
        out = {}
        count = min(len(left), len(right))
        for k in range(count):
            kpath = f'{path}.{k}'
            if self.ignore_path(kpath):
                continue
            if r := self.compare_any(left[k], right[k], kpath):
                out[str(k)] = r
        if count < len(left):
            out[f'{count}:{len(left)}'] = f'{path}: missing in {self.rname}'
        if count < len(right):
            out[f'{count}:{len(right)}'] = f'{path}: missing in {self.lname}'
        return out

    def compare_dir(self, left: Path, right: Path, path: str) -> CompareResult:
        lfiles = set(p.name for p in left.iterdir())
        rfiles = set(p.name for p in right.iterdir())
        names = sorted(lfiles | rfiles)
        out = {}
        for name in names:
            if self.ignore_file(name): continue
            fpath = f'{path}/{name}'
            if self.ignore_path(fpath): continue
            if not self.ignore_file(name):
                lpath = left.joinpath(name)
                rpath = right.joinpath(name)
                if r:= self.compare_path(lpath, rpath, fpath):
                    out[name] = r
        return out

    def compare_file(self, left: Path, right: Path, path: str) -> CompareResult:
        suffix = left.suffix
        if suffix != right.suffix:
            return f'{path}: different: {left} != {right} (suffix: {suffix})'
        if suffix == '.safetensors':
            larrays = ten.load_tensors(str(left))
            rarrays = ten.load_tensors(str(right))
            if r := self.compare_dict(larrays, rarrays, path, key_prefix='/'):
                return r
            return None
        elif suffix == '.json':
            try:
                ljson = json.loads(left.read_text())
                rjson = json.loads(right.read_text())
            except Exception as e:
                return f'error: {e}'
            return self.compare_any(ljson, rjson, path, key_prefix='/')
        return f'{path}: unknown suffix: {suffix}'

    def compare_path(self, left: Path, right: Path, path: str) -> CompareResult:
        if left.exists():
            if right.exists():
                if left.is_dir():
                    if right.is_dir():
                        return self.compare_dir(left, right, path)
                    else:
                        return f'{path}: {self.lname} is a directory, {self.rname} is not'
                if right.is_dir():
                    return f'{path}: {self.rname} is a directory, {self.lname} is not'
                else:
                    return self.compare_file(left, right, path)
            else:
                return f'{path}: does not exist in {self.rname}'
        else:
            if right.exists():
                return f'{path}: does not exist in {self.lname}'
            else:
                return None

    def compare(self, path: str ='') -> CompareResult:
        return self.compare_path(self.left, self.right, path)

    @classmethod
    def build(cls, left: Path, right: Path, lname: str, rname: str, eps: float = default_eps,
              ignore_key: Predicate[str] = None,
              ignore_keys: set[str] = None,
              ignore_key_pattern: str = None,
              ignore_file: Predicate[str] = None,
              ignore_files: set[str] = None,
              ignore_file_pattern: str = None,
              ignore_path: Predicate[str] = None,
              ignore_paths: set[str] = None,
              ignore_path_pattern: str = None,
              ) -> Self:

        ignore_key = compose_predicate(ignore_key, ignore_keys, ignore_key_pattern)

        ignore_file = compose_predicate(ignore_file, ignore_files, ignore_file_pattern)

        ignore_path = compose_predicate(ignore_path, ignore_paths, ignore_path_pattern)

        return cls(left=left, right=right, lname=lname, rname=rname, eps=eps,
                   ignore_key=ignore_key, ignore_file=ignore_file, ignore_path=ignore_path)


def compose_predicate(ignore_key: Predicate[str] = None, ignore_keys: set[str] = None, ignore_pattern: str = None) -> Predicate[str]:
    preds: list[Predicate[str]] = []

    if ignore_key is not None: preds.append(ignore_key)
    if ignore_keys: preds.append(lambda key: key in ignore_keys)
    if ignore_pattern:
        ignore_re = re.compile(ignore_pattern)
        preds.append(lambda key: bool(ignore_re.match(key)))
    if not preds:
        preds.append(default_ignore_key)

    return Predicates.any(*preds)
