#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from pathlib import Path
import glob
import requests

from .. import ten
from ..repo import Repo, repo_to_local_path
from ..infra import coerce, meta, log
from ..infra.types import *
from .architecture import Architecture
from .model import Model


M = TypeVar('M', bound=Model)


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


def try_load_config(config_path: Path) -> Optional[dict[str, Any]]:
    if config_path.exists():
        if loader := loaders.get(config_path.suffix):
            return loader(config_path)
        else:
            log.warn(f"Unsupported config file format: {config_path.suffix}")
    else:
        for suffix, loader in loaders.items():
            config_file = config_path.with_suffix(suffix)
            if config_file.exists():
                return loader(config_file)
    return None


def load_config(model_path: Path) -> dict[str, Any]:
    if model_path.is_dir():
        config = try_load_config(model_path / "config") or try_load_config(model_path / model_path.name)
    else:
        config = try_load_config(model_path)
    if config: return config
    raise FileNotFoundError(f"Config file not found in {model_path}")


# noinspection PyShadowingNames,PyPep8Naming
def load_model(
    path: Path,
    lazy: bool = False,
    extra_config: dict = None,
    load_weights: bool = None,
    Model: type[M] = Model,
) -> M:
    """
    Load and initialize the model from a given path.

    :param path: (Path): The path to load the model from.
    :param lazy: (bool): If False eval the model parameters to make sure they are
        loaded in memory before returning, otherwise they will be loaded
        when needed. Default: ``False``
    :param extra_config: (dict, optional): Optional configuration parameters for the
        model. Defaults to an empty dictionary.
    :param load_weights: (bool): If True, load the model weights from the
    :param Model: (type[M]): The model class to load. Defaults to the ``Model`` class.

    :returns nn.Module: The loaded and initialized model.

    :raises FileNotFoundError: If the weight files (.safetensors) are not found.
    :raises ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(path)

    if extra_config:
        config.update(extra_config)

    weight_files = glob.glob(str(path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(path / "weight*.safetensors"))

    if load_weights and not weight_files:
        log.error(f"No safetensors found in {path}")
        raise FileNotFoundError(f"No safetensors found in {path}")

    weights = {}
    for wf in weight_files:
        w = ten.load_tensors(wf)
        weights.update(w)

    config.setdefault('kind', config.get('model_type'))

    model = meta.coerce(Model, config)
    # model = CommonModel.from_config(config)

    # model_class, model_args_class = get_model_classes(config)
    #
    # model_args = model_args_class.from_dict(config)
    # model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # if (quantization := config.get("quantization", None)) is not None:
    #
    #     def class_predicate(p, m):
    #         # Handle custom per layer quantizations
    #         if p in config["quantization"]:
    #             go = config["quantization"][p]
    #         elif not hasattr(m, "to_quantized"):
    #             go = False
    #         # Handle legacy models which may not have everything quantized
    #         else: go = f"{p}.scales" in weights
    #         if go:
    #             return True
    #         return False
    #
    #     nn.quantize(
    #         model,
    #         group_size=quantization["group_size"],
    #         bits=quantization["bits"],
    #         class_predicate=class_predicate,
    #     )
    #
    # print(model.structure())

    if weights:
        # params = model.parameters()
        model.load_weights(list(weights.items()))

    if not lazy:
        ten.eval(model.parameters())

    model.eval()
    return model


model_dirs: list[Path] = [
    p.resolve() for p in [Path(__file__).parent / "_configs"] if p.is_dir()
]


def _check_path(path: Path) -> Optional[Path]:
    if path.exists():
        return path.resolve()
    else:
        if '.' in path.name:
            def add_suffix(p, s):
                return p.with_name(p.name + s)
        else:
            def add_suffix(p, s):
                return p.with_suffix(s)
        for suffix in loaders:
            file = add_suffix(path, suffix)
            if file.exists():
                return file.resolve()
    return None


def find_model_path(path: str|Path) -> Path:
    if path is None: raise ValueError(f"Path must be specified: {path}")
    if isinstance(path, str): path = Path(path)
    if isinstance(path, Path):
        if model_path := _check_path(path):
            return model_path
        for model_dir in model_dirs:
            if model_path := _check_path(model_dir / path):
                if model_path.exists():
                    break
                else:
                    model_path = None
        # else:
        #     model_path = repo_to_local_path(path)
        if model_path is None:
            raise FileNotFoundError(f"Could not find path: {path}")
        return model_path.resolve()
    raise ValueError(f"Path must be a string or Path: {path}")


# noinspection PyShadowingNames,PyPep8Naming
def load_model_from_config(config: dict[str, Any], Model: type[M] = Model) -> M:
    return coerce(Model, config)


def fetch_hf_config(repo: str) -> dict:
    url = f'https://huggingface.co/{repo}/raw/main/config.json'
    return requests.get(url).json()


def model_from_pretrained(name: str) -> Model:
    try:
        model_path = find_model_path(name.lower())
        with open(model_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        config = None

    if config is None:
        repo = coerce(Repo, name)
        repo_config = repo.fetch_config()
        arch = Architecture.from_config(repo_config)
        config = arch.convert(repo.name, repo.qname, org=repo.org)

    return load_model_from_config(config)


Model.from_pretrained = model_from_pretrained

