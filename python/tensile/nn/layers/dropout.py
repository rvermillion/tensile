#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ...infrastructure.util import name_function
from ...infrastructure.function import identity
from ..common import *
from ..module import *


def dropout_type(d: int) -> str:
    return 'Dropout' if d == 1 else f'Dropout{d}D'

@provides(Module, 'dropout')
class Dropout(CompiledModule):
    r"""Randomly zero a portion of the elements during training.

    The dimensionality d can be 1, 2 or 3.

    d=1 The remaining elements are multiplied with :math:`\frac{1}{1-p}` where
        :math:`p` is the probability of zeroing an element. This is done so the
        expected value of a given element will remain the same.

    d=2 Apply 2D channel-wise dropout during training.

        Randomly zero out entire channels independently with probability :math:`p`.
        This layer expects the channels to be last, i.e. the input shape should be
        ``NWHC`` or ``WHC`` where:``N`` is the batch dimension,``H`` is the input
        image height,``W`` is the input image width, and``C`` is the number of
        input channels

        The remaining channels are scaled by :math:`\frac{1}{1-p}` to
        maintain the expected value of each element. Unlike traditional dropout,
        which zeros individual entries, this layer zeros entire channels. This is
        beneficial for early convolution layers where adjacent pixels are
        correlated. In such case, traditional dropout may not effectively
        regularize activations. For more details, see [1].

        [1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
        Efficient Object Localization Using Convolutional Networks. CVPR 2015.

    d=3 Apply 3D channel-wise dropout during training.

        Randomly zero out entire channels independently with probability :math:`p`.
        This layer expects the channels to be last, i.e., the input shape should be
        `NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
        `H` is the input image height, `W` is the input image width, and `C` is
        the number of input channels.

        The remaining channels are scaled by :math:`\frac{1}{1-p}` to
        maintain the expected value of each element. Unlike traditional dropout,
        which zeros individual entries, this layer zeros entire channels. This is
        often beneficial for convolutional layers processing 3D data, like in
        medical imaging or video processing.

    Args:
        p (float): The probability to zero an element
        d (int): The dimensionality of the dropout
    """

    __slots__ = ('d', 'p',)

    d: Annotated[int, field(
        doc="The dimensionality of the dropout",
        default=1
    )]
    p: Annotated[float, field(
        doc="The probability to zero an element",
        default=0.5,
        readonly=True
    )]

    @property
    def notp(self) -> float:
        return 1. - self.p

    @notp.setter
    def notp(self, value: float):
        p = 1. - value
        if p < 0 or p >= 1:
            raise ValueError(f"The dropout probability {p} is not in [0, 1)")
        self.p = p

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)

        d = args.get('d', default=1)
        p = args.get('p', default=0.5)

        if p < 0 or p >= 1:
            raise ValueError(f"The dropout probability {p} is not in [0, 1)")
        if d < 1 or d > 3:
            raise ValueError(f"The number of {d} must be in [1, 3]")

        self.d = d
        self.notp = 1 - p

    def _repr_type(self, **options) -> str:
        return dropout_type(self.d)

    def _repr_args(self, **options) -> str:
        return f"p={self.p:3.2f}"

    def _extra_structure(self) -> str:
        return self._repr_args()

    def build_call(self, train: bool = False, **options) -> Callable:
        d = self.d
        notp = self.notp
        if train or notp == 1:
            scale = 1 / notp

            if d == 1:
                def call(x: Array) -> Array:
                    mask = ten.random.bernoulli(p=notp, shape=x.shape)
                    return (mask * x) * scale
            elif d == 2:
                def call(x: Array) -> Array:
                    if x.ndim not in (3, 4):
                        raise ValueError(
                            f"Received input with {x.ndim} dimensions. Expected 3 or 4 dimensions."
                        )

                    # Dropout is applied on the whole channel
                    # 3D input: (1, 1, C)
                    # 4D input: (B, 1, 1, C)
                    mask_shape = list(x.shape)
                    mask_shape[-2] = mask_shape[-3] = 1

                    mask = ten.random.bernoulli(p=notp, shape=mask_shape)
                    return (mask * x) * scale
            elif d == 3:
                def call(x: Array) -> Array:
                    if x.ndim not in (4, 5):
                        raise ValueError(
                            f"Received input with {x.ndim} dimensions. Expected 4 or 5 dimensions."
                        )

                    # Dropout is applied on the whole channel
                    # 4D input: (1, 1, 1, C)
                    # 5D input: (B, 1, 1, 1, C)
                    mask_shape = list(x.shape)
                    mask_shape[-2] = mask_shape[-3] = mask_shape[-4] = 1

                    mask = ten.random.bernoulli(p=notp, shape=mask_shape)
                    return (mask * x) * scale

            else:
                raise ValueError(f"The number of {d} must be in [1, 3]")
            return call
        else:
            return identity

    @classmethod
    def dropout(cls, mod: Callable[..., Array], p: float = 0.1, d: int = 1, **kwargs) -> Callable[..., Array]:
        if p == 0.:
            return mod

        dropout = cls.coerce(p=p, d=d, **kwargs)
        if not callable(dropout):
            raise TypeError(f'Dropout must be callable, got {dropout}')

        def call(*args, **kwargs) -> Array:
            return dropout(mod(*args, **kwargs))

        return name_function(call, dropout_type(d) + f'[{p}]({mod})')



__all__ = ['Dropout']