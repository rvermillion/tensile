#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


import math

from ..common import *
from ..module import CompiledModule, Module, ModuleArgs


class ConvolutionArgs(ModuleArgs):
    d: int = 1
    in_channels: int
    out_channels: int
    kernel_size: int|tuple[int, ...]
    stride: int|tuple[int, ...] = 1
    padding: int|tuple[int, ...] = 0
    dilation: int|tuple[int, ...] = 1
    groups: int = 1
    bias: bool = True


@provides(Module, 'convolution')
class Convolution(CompiledModule):

    __slots__ = ('weight', 'bias', 'd', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'groups',)

    weight: Annotated[Array, field(
        parameter=True,
    )]
    bias: Annotated[Optional[Array], field(
        parameter=True,
    )]
    d: int
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int

    def init_from_args(self, args: ConvolutionArgs):
        super().init_from_args(args)

        d = args.d
        in_channels = args.in_channels
        out_channels = args.out_channels
        kernel_size = args.kernel_size
        padding = args.padding
        dilation = args.dilation
        stride = args.stride
        groups = args.groups

        bias = args.bias

        if in_channels % groups != 0:
            raise ValueError(
                f"The number of input channels ({in_channels}) must be "
                f"divisible by the number of groups ({groups})"
            )

        if not 0 < d <= 3:
            raise ValueError(f"Dimensionality must be between 1 and 3, got {d}")


        if d == 1:
            kernel_size, stride, padding, dilation = map(
                lambda x: (x,) if isinstance(x, int) else x,
                (kernel_size, stride, padding, dilation),
            )
        elif d == 2:
            kernel_size, stride, padding, dilation = map(
                lambda x: (x, x) if isinstance(x, int) else x,
                (kernel_size, stride, padding, dilation),
            )
        elif d == 3:
            kernel_size, stride, padding, dilation = map(
                lambda x: (x, x, x) if isinstance(x, int) else x,
                (kernel_size, stride, padding, dilation),
            )

        for p in (kernel_size, stride, padding, dilation):
            if len(p) != d:
                raise ValueError(f'Wrong size tuple for dimension {d}: {p}')


        total_size = 1
        for size in kernel_size:
            total_size *= size

        scale = math.sqrt(1 / (in_channels * total_size))
        self.weight = ten.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels // groups),
        )

        self.d = d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.bias = ten.zeros((self.out_channels,)) if bias else None

    def build_call(self, train: bool = False, **options) -> Callable:
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        groups = self.groups
        if self.d == 1:
            conv = ten.conv1d
            stride = stride[0]
            padding = padding[0]
            dilation = dilation[0]
        elif self.d == 2:
            conv = ten.conv2d
        elif self.d == 3:
            conv = ten.conv3d
        else:
            raise ValueError(f"Invalid number of dimensions ({self.d})")

        def call(x):
            y = conv(x, self.weight, self.bias, stride, padding, dilation, groups)
            return y

        return call

    @property
    def in_dim(self) -> int:
        return self.in_channels

    @property
    def out_dim(self) -> int:
        return self.out_channels

    def _extra_structure(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, "
            f"bias={'bias' in self}"
        )
