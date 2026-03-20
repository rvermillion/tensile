#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import Annotated, Array, Callable, Optional, ten, field, provides
from ..module import CompiledModule, FunctionModule, Functional, Module, ModuleArgs, WrapperModule


class NormalizationArgs(ModuleArgs):
    dims: int
    eps: float = 1e-5
    affine: bool = True
    bias: bool = True


class Normalization(FunctionModule):

    __slots__ = ('dims', 'eps', 'affine')

    dims: Annotated[int, field(
        doc="The feature dimensions of the input to normalize over",
        changed=CompiledModule.recompile,
    )]
    eps: Annotated[float, field(
        doc="A small additive constant for numerical stability",
        default=1e-5,
        changed=CompiledModule.recompile,
    )]
    affine: Annotated[bool, field(
        default=False,
        changed=CompiledModule.recompile,
    )]

    def init_from_args(self, args: NormalizationArgs):
        super().init_from_args(args)
        self.dims = args.dims
        self.eps = args.eps
        self.affine = args.affine

    @property
    def in_dim(self) -> int:
        return self.dims

    @property
    def out_dim(self) -> int:
        return self.dims

    Args = NormalizationArgs


class AffineNorm(Normalization):

    __slots__ = ("weight", "bias")

    weight: Annotated[Optional[ten.Array], field(
        doc="The learnable weights of the layer.",
        parameter=True,
    )]
    bias: Annotated[Optional[ten.Array], field(
        doc="The learnable bias of the layer.",
        parameter=True,
    )]

    def init_from_args(self, args: NormalizationArgs, bias: bool = None):
        super().init_from_args(args)

        if bias is None:
            bias = args.bias

        if args.affine:
            dims = self.dims
            self.weight = ten.ones((dims,))
            self.bias = ten.zeros((dims,)) if bias else None
        else:
            self.weight = None
            self.bias = None

    def _extra_structure(self):
        return f"{self.dims}, eps={self.eps}, affine={self.weight is not None}"



@provides(Normalization, 'layer')
class LayerNorm(AffineNorm):
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

    __slots__ = ()

    def build_call(self, mode: CompiledModule.Mode, **options) -> Functional:
        eps = self.eps

        def call(x: Array, /) -> Array:
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
        self.eps = args.get_first('rms_norm_eps', 'eps', default=1e-5)
        self.weight = ten.ones((args.dims,))

    def _extra_structure(self):
        w = self.weight
        return f"{() if w is None else w.shape[0]}, eps={self.eps}"

    def build_call(self, mode: CompiledModule.Mode, **options) -> Functional:
        eps = self.eps

        def call(x: Array, /) -> Array:
            return ten.fast.rms_norm(x, self.weight, eps)

        return call


@provides(Normalization, 'instance')
class InstanceNorm(AffineNorm):
    r"""Applies instance normalization [1] on the inputs.

    Computes

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively. Both are of size :attr:`dims`,
    if :attr:`affine` is ``True``.

    Args:
        dims (int): The number of features of the input.
        eps (float): A value added to the denominator for numerical stability. Default: ``1e-5``.
        affine (bool): Default: ``False``.

    Shape:
      - Input: :math:`(..., C)` where :math:`C` is equal to :attr:`dims`.
      - Output: Same shape as the input.

    Examples:
        >>> from tensile import ten
        >>> from tensile.nn.layers import Normalization
        >>> x = ten.random.normal((8, 4, 4, 16))
        >>> inorm = Normalization.coerce(dims=16, kind='instance')
        >>> output = inorm(x)

    References:
        [1]: https://arxiv.org/abs/1607.08022
    """

    __slots__ = ()

    def init_from_args(self, args: NormalizationArgs, bias: bool = False):
        super().init_from_args(args, bias=True)

    def build_call(self, mode: CompiledModule.Mode, **options) -> Functional:
        eps = self.eps
        if self.weight is None:
            def call(x: Array) -> Array:
                reduction_axes = tuple(range(1, x.ndim - 1))
                # Compute stats
                mean = ten.mean(x, axis=reduction_axes, keepdims=True)
                var = ten.var(x, axis=reduction_axes, keepdims=True)
                # Normalize
                x = (x - mean) * ten.rsqrt(var + eps)
                # Scale and shift if necessary
                return x
        else:
            def call(x: Array, /) -> Array:
                reduction_axes = tuple(range(1, x.ndim - 1))
                # Compute stats
                mean = ten.mean(x, axis=reduction_axes, keepdims=True)
                var = ten.var(x, axis=reduction_axes, keepdims=True)
                # Normalize
                x = (x - mean) * ten.rsqrt(var + eps)
                # Scale and shift if necessary
                return self.weight * x + self.bias
        return call


@provides(Normalization, 'group')
class GroupNorm(AffineNorm):
    r"""Applies Group Normalization [1] to the inputs.

    Computes the same normalization as layer norm, namely

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively. However, the mean and
    variance are computed over the spatial dimensions and each group of
    features. In particular, the input is split into num_groups across the
    feature dimension.

    The feature dimension is assumed to be the last dimension and the dimensions
    that precede it (except the first) are considered the spatial dimensions.

    [1]: https://arxiv.org/abs/1803.08494

    Args:
        num_groups (int): Number of groups to separate the features into
        dims (int): The feature dimensions of the input to normalize over
        eps (float): A small additive constant for numerical stability
        affine (bool): If True learn an affine transform to apply after the
            normalization.
        pytorch_compatible (bool): If True perform the group normalization in
            the same order/grouping as PyTorch.
    """

    __slots__ = ("num_groups", "pytorch_compatible")

    num_groups: Annotated[int, field()]
    pytorch_compatible: Annotated[bool, field(
        default=True,
        changed=CompiledModule.recompile,
    )]

    def init_from_args(self, args: NormalizationArgs, bias: bool = True):
        super().init_from_args(args, bias=True)

        self.num_groups = args.get('num_groups')
        self.pytorch_compatible = args.get('pytorch_compatible', True)

    def _extra_structure(self):
        return (
            f"{self.num_groups}, {self.dims}, eps={self.eps}, "
            f"affine={self.weight is not None}, pytorch_compatible={self.pytorch_compatible}"
        )

    def build_call(self, mode: CompiledModule.Mode, **options) -> Functional:
        eps = self.eps
        num_groups = self.num_groups

        if self.pytorch_compatible:
            def group_norm(x: Array, /) -> Array:
                batch, *rest, dims = x.shape
                group_size = dims // num_groups

                # Split into groups
                x = x.reshape(batch, -1, num_groups, group_size)
                x = x.transpose(0, 2, 1, 3).reshape(batch, num_groups, -1)

                # Normalize
                x = ten.fast.layer_norm(x, eps=eps, weight=None, bias=None)

                x = x.reshape(batch, num_groups, -1, group_size)
                x = ten.transpose(x, (0, 2, 1, 3)).reshape(batch, *rest, dims)
                return x
        else:
            def group_norm(x: Array, /) -> Array:
                batch, *rest, dims = x.shape

                # Split into groups
                x = x.reshape(batch, -1, num_groups)

                # Normalize
                means = ten.mean(x, axis=1, keepdims=True)
                var = ten.var(x, axis=1, keepdims=True)
                x = (x - means) * ten.rsqrt(var + eps)
                x = x.reshape(batch, *rest, dims)

                return x

        if self.weight is None:
            call = group_norm
        else:
            def call(x: Array, /) -> Array:
                return self.weight * group_norm(x) + self.bias
        return call


@provides(Normalization, 'batch')
class BatchNorm(Normalization):
    r"""Applies Batch Normalization over a 2D or 3D input.

    Computes

    .. math::

        y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

    where :math:`\gamma` and :math:`\beta` are learned per feature dimension
    parameters initialized at 1 and 0 respectively.

    The input shape is specified as ``NC`` or ``NLC``, where ``N`` is the
    batch, ``C`` is the number of features or channels, and ``L`` is the
    sequence length. The output has the same shape as the input. For
    four-dimensional arrays, the shape is ``NHWC``, where ``H`` and ``W`` are
    the height and width respectively.

    For more information on Batch Normalization, see the original paper `Batch
    Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

    Args:
        num_features (int): The feature dimension to normalize over.
        eps (float, optional): A small additive constant for numerical
            stability. Default: ``1e-5``.
        momentum (float, optional): The momentum for updating the running
            mean and variance. Default: ``0.1``.
        affine (bool, optional): If ``True``, apply a learned affine
            transformation after the normalization. Default: ``True``.
        track_running_stats (bool, optional): If ``True``, track the
            running mean and variance. Default: ``True``.

    Examples:
        >>> from tensile import ten
        >>> from tensile.nn.layers import Normalization
        >>> x = ten.random.normal((5, 4))
        >>> bnorm = Normalization.coerce(num_features=4, affine=True, kind='batch')
        >>> output = bnorm(x)
    """

    __slots__ = ("weight", "bias", "num_features", "momentum", "track_running_stats", "running_mean", "running_var")

    weight: Annotated[Optional[Array], field(
        parameter=True,
    )]
    bias: Annotated[Optional[Array], field(
        parameter=True,
    )]
    num_features: Annotated[int, field(
    )]
    momentum: Annotated[float, field(
        default=0.1,
    )]
    track_running_stats: Annotated[bool, field(
        default=True
    )]
    running_mean: Annotated[Optional[Array], field(
        parameter=False,
    )]
    running_var: Annotated[Optional[Array], field(
        parameter=False,
    )]

    def init_from_args(self, args: NormalizationArgs):
        super().init_from_args(args)

        num_features = args.get('num_features')
        momentum = args.get('momentum')
        affine = args.affine
        track_running_stats = args.get('track_running_stats')

        self.num_features = num_features
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = ten.ones((num_features,))
            self.bias = ten.zeros((num_features,))
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = ten.zeros((num_features,))
            self.running_var = ten.ones((num_features,))
        else:
            self.running_mean = None
            self.running_var = None

    def _extra_structure(self):
        return (
            f"{self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.weight is not None}, "
            f"track_running_stats={self.track_running_stats}"
        )

    def build_call(self, mode: CompiledModule.Mode, **options) -> Functional:
        mu = self.momentum
        eps = self.eps

        if self.track_running_stats:
            if mode.is_train():
                def get_stats(x: Array) -> tuple[Array, Array]:
                    reduction_axes = tuple(range(0, x.ndim - 1))

                    mean = ten.mean(x, axis=reduction_axes)
                    var = ten.var(x, axis=reduction_axes)
                    self.running_mean = (1 - mu) * self.running_mean + mu * mean
                    self.running_var = (1 - mu) * self.running_var + mu * var
                    return mean, var
            else:
                def get_stats(x: Array) -> tuple[Array, Array]:
                    return self.running_mean, self.running_var
        else:
            def get_stats(x: Array) -> tuple[Array, Array]:
                reduction_axes = tuple(range(0, x.ndim - 1))

                mean = ten.mean(x, axis=reduction_axes)
                var = ten.var(x, axis=reduction_axes)
                return mean, var

        if self.weight is None:
            def call(x: Array, /) -> Array:
                if x.ndim < 2 or x.ndim > 4:
                    raise ValueError(f"Expected input tensor to have 2, 3 or 4 dimensions, but got {x.ndim}")

                mean, var = get_stats(x)
                x = (x - mean) * ten.rsqrt(var + eps)
                return x
        else:

            def call(x: Array, /) -> Array:
                if x.ndim < 2 or x.ndim > 4:
                    raise ValueError(f"Expected input tensor to have 2, 3 or 4 dimensions, but got {x.ndim}")

                mean, var = get_stats(x)
                x = (x - mean) * ten.rsqrt(var + eps)
                return self.weight * x + self.bias

        return call


class NormedModuleArgs(ModuleArgs):

    pre_norm: Optional[NormalizationArgs] = None
    body: ModuleArgs
    post_norm: Optional[NormalizationArgs] = None



@provides(Module, 'normed')
class NormedModule(WrapperModule, FunctionModule):
    r"""A module that normalizes its inputs."""

    __slots__ = ('pre_norm', 'post_norm')

    pre_norm: Annotated[Optional[Normalization], field(
        doc="Prenormalization for the module.",
    )]
    body: Annotated[FunctionModule, field(
        doc="The module to wrap.",
        ignore=True,
    )]
    post_norm: Annotated[Optional[Normalization], field(
        doc="Postnormalization for the module.",
    )]


    def init_from_args(self, args: NormedModuleArgs):
        super().init_from_args(args)

        self.pre_norm = self.build_pre_norm(args)
        self.post_norm = self.build_post_norm(args)

    def build_pre_norm(self, args: NormedModuleArgs) -> Optional[Normalization]:
        if norm_args := args.pre_norm:
            return Normalization.from_args(norm_args)
        return None

    def build_post_norm(self, args: NormedModuleArgs) -> Optional[Normalization]:
        if norm_args := args.post_norm:
            return Normalization.from_args(norm_args)
        return None

    def build_call(self, mode: CompiledModule.Mode, **options) -> Functional:
        pre_norm = self.pre_norm
        body = self.body
        post_norm = self.post_norm

        if pre_norm is None:
            if post_norm is None:
                return body

            def call(x: Array, /) -> Array:
                h = body(x)
                return post_norm(h)
        else:
            if post_norm is None:
                def call(x: Array, /) -> Array:
                    h = pre_norm(x)
                    return body(h)
            else:
                def call(x: Array) -> Array:
                    h = pre_norm(x)
                    h = body(h)
                    return post_norm(h)
        return call

    def __getattr__(self, item):
        # print(f'Getting item: {item}')
        return getattr(self.body, item)

    unwrapped_keys = {*WrapperModule.unwrapped_keys, 'pre_norm', 'post_norm'}

    Args = NormedModuleArgs
