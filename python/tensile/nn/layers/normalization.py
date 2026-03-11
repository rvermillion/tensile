#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import Annotated, Callable, Optional, ten, field, provides
from ..module import CompiledModule, ModuleArgs


class NormalizationArgs(ModuleArgs):
    dims: int
    eps: float = 1e-5
    affine: bool = True
    bias: bool = True


class Normalization(CompiledModule):

    __slots__ = ('dims', 'eps',)

    dims: Annotated[int, field(
        doc="The feature dimensions of the input to normalize over",
        changed=CompiledModule.recompile,
    )]
    eps: Annotated[float, field(
        doc="A small additive constant for numerical stability",
        default=1e-5,
        changed=CompiledModule.recompile,
    )]

    def init_from_args(self, args: NormalizationArgs):
        super().init_from_args(args)
        self.dims = args.dims
        self.eps = args.eps

    @property
    def input_features(self) -> int:
        return self.dims

    @property
    def output_features(self) -> int:
        return self.dims

    Args = NormalizationArgs


@provides(Normalization, 'layer')
class LayerNorm(Normalization):
    r"""Applies layer normalization [1] on the inputs.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    [1]: https://arxiv.org/abs/1607.06450

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization
        bias (bool): If True include a translation to the affine
            transformation. If set to False the transformation is not really affine
            just scaling.
    """

    __slots__ = ("weight", "bias")

    weight: Annotated[Optional[ten.Array], field(
        doc="The learnable weights of the layer.",
        parameter=True,
    )]
    bias: Annotated[Optional[ten.Array], field(
        doc="The learnable bias of the layer.",
        parameter=True,
    )]

    def init_from_args(self, args: NormalizationArgs):
        super().init_from_args(args)

        if args.affine:
            dims = self.dims
            self.weight = ten.ones((dims,))
            self.bias = ten.zeros((dims,)) if args.bias else None

    def _extra_structure(self):
        return f"{self.dims}, eps={self.eps}, affine={'weight' in self}"

    def build_call(self, train: bool = False, **options) -> Callable:
        eps = self.eps

        def call(x):
            return ten.fast.layer_norm(x, self.weight, self.bias, eps)

        return call


@provides(Normalization, 'rms')
class RMSNorm(Normalization):
    r"""Applies Root Mean Square normalization [1] to the inputs.

    Computes

    ..  math::

        y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \gamma

    where :math:`\gamma` is a learned per feature dimension parameter initialized at
    1.

    Note the accumulation for the mean is done in 32-bit precision.

    [1]: https://arxiv.org/abs/1910.07467

    Args:
        dims (int): The feature dimension of the input to normalize over
        eps (float): A small additive constant for numerical stability
    """

    __slots__ = ("weight",)

    weight: Annotated[ten.Array, field(
        doc="The learnable weights of the layer.",
        parameter=True,
    )]

    def init_from_args(self, args: NormalizationArgs):
        super().init_from_args(args)
        self.dims = args.dims
        self.eps = args.get_first('rms_norm_eps', 'eps', default=1e-5)
        self.weight = ten.ones((args.dims,))

    def _extra_structure(self):
        w = self.weight
        return f"{() if w is None else w.shape[0]}, eps={self.eps}"

    def build_call(self, train: bool = False, **options) -> Callable:
        eps = self.eps

        def call(x: ten.Array) -> ten.Array:
            ten.debug_eval(x)
            return ten.fast.rms_norm(x, self.weight, eps)

        return call


