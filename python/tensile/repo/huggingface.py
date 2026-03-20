#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from pathlib import Path
import requests

from ..infra import provides

from .common import Repo, repo_provider_aliases

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise ImportError("Run `pip install huggingface_hub` to use Hugging Face Hub.")

repo_provider = "huggingface"
short_repo_provider = 'hf'

repo_provider_aliases[short_repo_provider] = repo_provider

def repo_to_local_path(repo: str|Path, local_files_only: bool = True) -> Path:
    return Path(snapshot_download(str(repo), local_files_only=local_files_only))


@provides(Repo, repo_provider, short_repo_provider)
class HuggingFaceRepo(Repo):

    __slots__ = ()

    def _lazy_local_path(self) -> Path:
        return repo_to_local_path(self.path, local_files_only=not self.download)

    def fetch_config(self) -> dict:
        url = f'https://huggingface.co/{self.path}/raw/main/config.json'
        return requests.get(url).json()


    kind = repo_provider
    short_kind = short_repo_provider


__all__ = []
