#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from .common import *
from .module import CompiledModule, ModuleArgs


EncodePosition = Callable[[Array, int], Array]


class PositionEncoder(CompiledModule):

    __slots__ = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def build_call(self, train: bool = False, **options) -> EncodePosition:
        raise NotImplementedError()

    @property
    def in_dim(self) -> int:
        return -1

    @property
    def out_dim(self) -> int:
        return -1

    class Args(ModuleArgs):

        dims: int
        shift: int = 0
        traditional: bool = False
        base: float = 10000,
        scale: float = 1.0
        max_positions: int = 2048,
        scaling: dict[str, Any] = None


def identity_encoder(x: Array, offset: int = 0) -> Array:
    return x


@provides(PositionEncoder, 'identity', 'none')
class Identity(PositionEncoder):
    """A position encoder that does nothing."""

    def build_call(self, train: bool = False, **options) -> EncodePosition:
        return identity_encoder


rope_cache: dict[tuple, EncodePosition] = {}


def build_rope_call(dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0,
                    shift: int = 0) -> EncodePosition:
    try:
        call = rope_cache[dims, traditional, base, scale, shift]
    except KeyError:

        rope = ten.fast.rope

        if shift == 0:
            def call(x: Array, offset: int = 0) -> Array:
                # noinspection PyTypeChecker
                return rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
        else:
            rem = dims - shift

            def call(x: Array, offset: int = 0) -> Array:
                # noinspection PyTypeChecker
                return rope(
                    x[..., shift:],
                    rem,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )

        rope_cache[dims, traditional, base, scale, shift] = call

    return call



@provides(PositionEncoder, 'rope')
class RoPE(PositionEncoder):
    """Implements the rotary positional encoding.

    The traditional implementation rotates consecutive pairs of elements in the
    feature dimension while the default implementation rotates pairs with
    stride half the feature dimensions for efficiency.

    For more details see `RoFormer: Enhanced Transformer with Rotary Position
    Embedding <https://arxiv.org/abs/2104.09864>`_.

    Args:
        dims (int): The feature dimensions to be rotated. If the input feature
            is larger than dims then the rest is left unchanged.
        traditional (bool, optional): If set to ``True`` choose the traditional
            implementation which is slightly less efficient. Default: ``False``.
        base (float, optional): The base used to compute angular frequency for
            each dimension in the positional encodings. Default: ``10000``.
        scale (float, optional): The scale used to scale the positions. Default: ``1.0``.
    """

    __slots__ = ('dims', 'shift', 'traditional', 'base', 'scale')

    dims: int
    shift: int
    traditional: bool
    base: float
    scale: float

    def init_from_args(self, args: PositionEncoder.Args):
        super().init_from_args(args)
        self.dims = args.dims
        self.shift = args.shift
        self.traditional = args.traditional
        self.base = args.base
        self.scale = args.scale

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dims,

    def _extra_structure(self):
        return f"{self.dims}, traditional={self.traditional}"

    def build_call(self, train: bool = False, **options) -> EncodePosition:
        return build_rope_call(self.dims, self.traditional, self.base, self.scale)


@provides(PositionEncoder, 'rope.llama3')
class Llama3RoPE(PositionEncoder):
    __slots__ = ('dims', 'max_positions', 'traditional', 'base', 'scaling')

    dims: Annotated[int, field(
        doc="The feature dimensions to be rotated. If the input feature is larger than dims then the rest is left unchanged.",
    )]
    max_positions: Annotated[int, field(
        doc="The maximum number of tokens that can be encoded.",
        default=2048,
    )]
    traditional: Annotated[bool, field(
        doc="If set to ``True`` choose the traditional implementation which is slightly less efficient.",
        default=False,
    )]
    base: Annotated[float, field(
        doc="The base used to compute angular frequency for each dimension in the positional encodings.",
        default=10000,
    )]
    scaling: Annotated[dict[str, Any], field(
        doc="The scaling configuration for the RoPE.",
        default_factory=dict,
        tree=False,
    )]

    def init_from_args(self, args: ModuleArgs):
        super().init_from_args(args)
        self.dims = args.dims
        self.max_positions = args.get('max_positions', 2048)
        self.traditional = args.get('traditional', False)
        self.base = args.get('base', 10000)
        self.scaling = args.get('scaling', {})

    def build_call(self, train: bool = False, **options) -> EncodePosition:
        scaling_config = self.scaling

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_positions",
            8192,
        )

        return self.build_rope(self.dims, traditional=self.traditional, base=self.base,
                               factor=factor, low_freq_factor=low_freq_factor, high_freq_factor=high_freq_factor,
                               old_context_len=old_context_len)
        return self.rope

    def _extra_structure(self):
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_positions={self.max_positions}"
        )


    cache: ClassVar[dict[tuple, EncodePosition]] = {}

    @classmethod
    def build_rope(cls, dims: int, traditional: bool, base: float, factor: float, low_freq_factor: float, high_freq_factor: float, old_context_len: int) -> EncodePosition:
        key = dims, traditional, base, factor, low_freq_factor, high_freq_factor, old_context_len

        if rope := cls.cache.get(key):
            return rope

        freqs = base ** (ten.arange(0, dims, 2) / dims)
        wavelens = 2 * ten.pi * freqs
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor


        freqs = ten.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        freqs = ten.where(is_medium_freq, smooth_freqs, freqs)

        def rope(x, offset: int = 0):
            return ten.fast.rope(
                x,
                dims,
                traditional=traditional,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=freqs,
            )

        cls.cache[key] = rope
        return rope


__all__ = [
    'PositionEncoder',
]