#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import re
from pathlib import Path
import glob

from .types import *
from ..infra import log


loaders: dict[str, Callable[[Path], dict]] = {}

try:
    import yaml

    def load_yaml(path: Path) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    loaders[".yaml"] = load_yaml
except ModuleNotFoundError:

    def load_yaml(path: Path) -> dict:
        raise ModuleNotFoundError("yaml module not found, cannot load yaml files")


try:
    import json

    def load_json(path: Path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    loaders[".json"] = load_json
except ModuleNotFoundError:

    def load_json(path: Path) -> dict:
        raise ModuleNotFoundError("json module not found, cannot load yaml files")


def try_load(path: Path) -> Optional[dict[str, Any]]:
    if path.exists():
        if loader := loaders.get(path.suffix):
            return loader(path)
        else:
            log.warn("Unsupported file format [{}] for {}", path.suffix, path)
    else:
        for suffix, loader in loaders.items():
            file = path.with_suffix(suffix)
            if file.exists():
                return loader(file)
    return None


def find_highest(path: Path) -> Optional[Path]:
    parent = path.parent
    name = path.name
    parts = name.split('*')
    if len(parts) == 1:
        return path if path.exists() else None

    pattern = r'(\d+)'.join(parts)
    matcher = re.compile(pattern)
    max_tup = (0,) * (len(parts) - 1)
    file = None
    for file in parent.glob(name):
        try:
            tup = tuple(map(int, matcher.match(file.name).groups()))
        except ValueError:
            continue

        for m, t in zip(max_tup, tup):
            if t < m:
                break
            elif t == m:
                continue
            else:
                max_tup = tup
                break
    return file
