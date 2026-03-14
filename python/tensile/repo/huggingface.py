#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from pathlib import Path

from ..infra import provides

from .common import Repo, repo_provider_aliases

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise ImportError("Run `pip install huggingface_hub` to use Hugging Face Hub.")

repo_provider = "huggingface"

repo_provider_aliases['hf'] = repo_provider

def repo_to_local_path(repo: str|Path, local_files_only: bool = True) -> Path:
    return Path(snapshot_download(str(repo), local_files_only=local_files_only))


@provides(Repo, repo_provider)
class HuggingFaceRepo(Repo):

    __slots__ = ()

    def _lazy_local_path(self) -> Path:
        return repo_to_local_path(self.name, local_files_only=not self.download)

    kind = repo_provider



__all__ = []
