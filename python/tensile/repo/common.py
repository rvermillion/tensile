#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
import os
from pathlib import Path

from ..infra import Object, field
from ..infra.types import Annotated, ClassVar, TYPE_CHECKING


class Repo(Object):

    __slots__ = ('path', 'name', 'org', 'local_path', 'download', 'weight_files')

    path: Annotated[str, field(
        doc='The name of the repository.',
        required=True,
    )]
    name: Annotated[str, field(
        doc='The org of the repository.',
    )]
    org: Annotated[str, field(
        doc='The org of the repository.',
    )]

    local_path: Annotated[Path, field(
        doc='The local path to the repository.',
    )]

    download: Annotated[bool, field(
        doc='Whether to download the repository.',
        default=True,
    )]

    weight_files: Annotated[list[Path], field(
        doc='The weight files in the repository.',
    )]

    def _lazy_name(self):
        return self.path.split('/')[-1]

    def _lazy_org(self):
        return self.path.split('/')[0]

    def _lazy_local_path(self) -> Path:
        raise ValueError("Local path not initialized")

    def _lazy_weight_files(self) -> list[Path]:
        weight_files = list(self.local_path.glob("model*.safetensors"))

        if not weight_files:
            # Try weight for back-compat
            weight_files = list(self.local_path.glob("weight*.safetensors"))

        return weight_files

    def _repr_args(self, **options) -> str:
        return self.qname

    @property
    def qname(self) -> str:
        return self.short_kind + ':' + self.path

    def fetch_config(self) -> dict:
        raise NotImplementedError(self)

    @classmethod
    def _coerce_from_str(cls, spec: str, /, kind: str = None, **kwargs):
        if ':' in spec:
            if kind is None:
                kind, repo = spec.split(':', maxsplit=1)
            else:
                _, repo = spec.split(':', maxsplit=1)
        else:
            if kind is None: kind = default_repo_provider
            repo = spec
        kwargs['kind'] = get_repo_provider(kind)
        kwargs.setdefault('path', repo)
        return cls._coerce_from_mapping(kwargs)

    kind: ClassVar[Annotated[str, field(
        doc='The kind of repository.',
    )]]
    short_kind: ClassVar[Annotated[str, field(
        doc='The kind of repository.',
    )]]


if TYPE_CHECKING:
    def repo_to_local_path(repo: str|Path, local_files_only: bool = True, **kwargs) -> Path: ...

else:
    def repo_to_local_path(repo: str|Path, **kwargs) -> Path:
        repo = Repo.coerce(str(repo), **kwargs)
        local_path = repo.local_path
        if not local_path.exists():
            raise FileNotFoundError(
                f"Repo {repo} not found. Please download it first."
            )
        return local_path


repo_provider_aliases = {}


default_repo_provider: str

def get_repo_provider(repo_provider: str) -> str:
    repo_provider = repo_provider.lower().strip()
    repo_provider = repo_provider_aliases.get(repo_provider, repo_provider)
    return repo_provider


def set_default_repo_provider(repo_provider: str, load: str|bool = False):
    global default_repo_provider
    default_repo_provider = get_repo_provider(repo_provider)
    if load:
        import importlib
        if isinstance(load, str):
            importlib.import_module(load, __package__)
        else:
            importlib.import_module(f".{default_repo_provider}", __package__)


def get_default_repo_provider() -> str:
    return default_repo_provider


def add_repo_provider_alias(alias: str, provider: str):
    repo_provider_aliases[alias] = get_repo_provider(provider)


set_default_repo_provider(os.getenv("REPO_PROVIDER", "huggingface"), load=True)

