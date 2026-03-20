#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from pathlib import Path

from ..nn.quantization import QuantizableModuleArgs
from ..nn.module import *
from ..repo import Repo
from ..infra import field
from ..infra.types import *
from .. import Array, ten


class ModelArgs(QuantizableModuleArgs):
    name: str
    org: str
    model_type: str = 'unknown'
    repo: str = None
    remap_weights: Optional[dict[str, str]] = None


class Model(CompiledModule):
    """Model class for handling the loading, initialization, and management of a compiled model.

    This class is designed to encapsulate the operations required to manage a model, including
    initialization, loading weights, and ensuring compatibility with specific repositories. It is meant
    to work seamlessly with a repository containing model weights, providing options for lazy loading
    and weight sanitization.

    :ivar args: (ModelArgs) The model arguments used to configure the instance.
    :ivar name: (str) The name of the model.
    :ivar model_type: (str) The type of the model, indicating its architecture or purpose.
    :ivar repo: (Repo?) The repository containing the model weights, or None if not specified.
    """

    __slots__ = ('name', 'org', 'model_type', 'repo')

    args: Annotated[ModelArgs, field(
        doc="The arguments for this model.",
        required=True,
    )]
    name: Annotated[str, field(
        doc="The name of the model.",
        required=True,
    )]
    org: Annotated[str, field(
        doc="The organization of the model.",
    )]
    model_type: Annotated[str, field(
        doc="The type of the model, indicating its architecture or purpose.",
    )]
    repo: Annotated[Optional[Repo], field(
        doc='The repository containing the model weights.',
    )]

    # noinspection PyShadowingNames
    def init_from_args(self, args: ModelArgs):
        super().init_from_args(args)
        self.name = args.name
        self.org = args.org
        self.model_type = args.model_type
        repo = args.repo
        if repo is None:
            if self.org is not None:
                repo = f'hf:{self.org}/{self.name}'
        self.repo = None if repo is None else Repo.coerce(repo)

    def sanitize_weights(self, weights: dict[str, Array]) -> dict[str, Array]:
        if remap_weights := self.args.remap_weights:
            self.warn('Remapping weights for {}: {}', self, remap_weights)
            weights = {n: w for k, w in weights.items() if (n := remap_weights.get(k, k)) is not None}
        return weights

    def load_model(self, path: Path = None, *,
                   lazy: bool = False,
                   load_weights: bool = None,
                   ) -> None:
        """
        Load and initialize the model from a given path.

        :param path: (Path?): The path to load the model from. If not provided,
            the model will be loaded from the repository. Default: ``None``
        :param lazy: (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        :param load_weights: (bool): If True, load the model weights from the

        :raises FileNotFoundError: If the weight files (.safetensors) are not found.
        :raises ValueError: If the model class or args class are not found or cannot be instantiated.
        """

        if path is None:
            if repo := self.repo:
                path = repo.local_path
            else:
                self.warn('No path or repo provided')
                if load_weights:
                    raise FileNotFoundError(f"No safetensors found in {path}")
                return
        else:
            repo = Repo.coerce(name=self.name, local_path=path)

        weight_files = repo.weight_files

        if load_weights and not weight_files:
            self.error(f"No safetensors found in {path}")
            raise FileNotFoundError(f"No safetensors found in {path}")
        elif weight_files and load_weights is None:
            load_weights = True

        if load_weights:
            weights = {}
            for wf in weight_files:
                w = ten.load_tensors(wf)
                weights.update(w)

            weights = self.sanitize_weights(weights)

            if weights:
                # params = model.parameters()
                self.load_weights(list(weights.items()), strict=True)

        if not lazy:
            ten.eval(self.parameters())

        self.eval()

    def _extra_structure(self) -> str:
        return f'{self.name}, model_type={self.model_type}'

    @classmethod
    def from_pretrained(cls, repo: str) -> 'Model':
        raise NotImplementedError('This should be replaced if tensile.models.load is imported')

    Args = ModelArgs
