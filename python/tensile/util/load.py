#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from collections.abc import Callable
from typing import Any, Optional
from pathlib import Path

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


def load(path: Path, search: bool = False) -> dict[str, Any]:
    if path.exists():
        if loader := loaders.get(path.suffix):
            return loader(path)
        else:
            raise IOError(f"Unsupported file format: {path.suffix}")
    elif search:
        for suffix, loader in loaders.items():
            file = path.with_suffix(suffix)
            if file.exists():
                return loader(file)
    raise FileNotFoundError(f"No such file: {path}")


def try_load(path: Path, search: bool = False) -> Optional[dict[str, Any]]:
    if path.exists():
        if loader := loaders.get(path.suffix):
            return loader(path)
    elif search:
        for suffix, loader in loaders.items():
            file = path.with_suffix(suffix)
            if file.exists():
                return loader(file)
    return None


__all__ = [
    'load',
    'try_load',
]