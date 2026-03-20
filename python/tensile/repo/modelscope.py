#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


from pathlib import Path

from ..infra import provides
from .common import Repo, repo_provider_aliases

try:
    from modelscope import snapshot_download
except ImportError:
    raise ImportError("Run `pip install modelscope` to use ModelScope.")

repo_provider = "modelscope"

repo_provider_aliases['ms'] = repo_provider


def repo_to_local_path(repo: str|Path, local_files_only: bool = True) -> Path:
    raise NotImplementedError()


@provides(Repo, repo_provider, 'ms')
class ModelscopeRepo(Repo):

    __slots__ = ()

    def _lazy_local_path(self) -> Path:
        return repo_to_local_path(self.name, local_files_only=not self.download)

    kind = repo_provider



__all__ = []
