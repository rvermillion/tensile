#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.
import math

from .common import *
from .module import CompiledModule, ModuleArgs


EncodePosition = Callable[[Array, int], Array]


class PositionEncoderArgs(ModuleArgs):

    dims: int
    shift: int = 0
    traditional: bool = False
    base: float = 10000,
    scale: float = 1.0
    max_positions: int = 2048,
    original_max_positions: int = 2048,
    partial_rotary_factor: float = 1.0
    scaling: dict[str, Any] = None


class PositionEncoder(CompiledModule):

    __slots__ = ()

    args: Annotated[PositionEncoderArgs, field(ignore=True)]

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def build_call(self, mode: CompiledModule.Mode, **options) -> EncodePosition:
        raise NotImplementedError()

    @property
    def in_dim(self) -> int:
        return -1

    @property
    def out_dim(self) -> int:
        return -1

    Args = PositionEncoderArgs

    if TYPE_CHECKING:

        __call__: EncodePosition


def identity_encoder(x: Array, offset: int = 0) -> Array:
    return x


@provides(PositionEncoder, 'identity', 'none')
class Identity(PositionEncoder):
    """A position encoder that does nothing."""

    def build_call(self, mode: CompiledModule.Mode, **options) -> EncodePosition:
        return identity_encoder


rope_cache: dict[tuple, EncodePosition] = {}


def shift_encode(encode: EncodePosition, shift: int) -> EncodePosition:
    if shift > 0:
        def shifted_encode(x: Array, offset: int = 0) -> Array:
            encoded = encode(x[..., shift:], offset=offset)
            return ten.concatenate([x[..., :shift], encoded], axis=-1)
        return shifted_encode
    return encode


def build_rope_call(dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0,
                    shift: int = 0) -> EncodePosition:
    try:
        call = rope_cache[dims, traditional, base, scale, shift]
    except KeyError:

        rope = ten.fast.rope

        rem = dims - shift

        def call(x: Array, offset: int = 0) -> Array:
            # noinspection PyTypeChecker
            return rope(
                x,
                rem,
                traditional=traditional,
                base=base,
                scale=scale,
                offset=offset,
            )

        call = shift_encode(call, shift)

        rope_cache[dims, traditional, base, scale, shift] = call

    return call


def build_fast_rope_call(dims: int, *, traditional: bool, base: float|None, scale: float,
                         mscale: float = 1.0,
                         freqs: Optional[Array] = None) -> EncodePosition:
    rope = ten.fast.rope
    if mscale == 1.0:
        def call(x, offset=0):
            return rope(
                x,
                dims,
                traditional=traditional,
                base=base,
                scale=scale,
                offset=offset,
                freqs=freqs,
            )
    else:
        def call(x, offset=0):
            x[..., :dims] = mscale * x[..., :dims]
            return rope(
                x,
                dims,
                traditional=traditional,
                base=base,
                scale=scale,
                offset=offset,
                freqs=freqs,
            )

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

    __slots__ = ('dims', 'traditional', 'base', 'scale', 'scaling', 'freqs')

    dims: Annotated[int, field(
        doc="The number of dimensions to be used for the RoPE."
    )]
    traditional: Annotated[bool, field(
        doc="Whether to use the traditional implementation of RoPE.",
        default=False,
    )]
    base: Annotated[float, field(
        doc="The base used to compute angular frequency for each dimension in the positional encodings.",
        default=10000,
    )]
    scale: Annotated[float, field(
        doc="The scale used to scale the positions.",
        default=1.0,
    )]
    scaling: Annotated[dict[str, Any], field(
        doc="The scaling configuration for the RoPE.",
        default_factory=dict,
        tree=False,
    )]
    freqs: Annotated[Optional[Array], field(
        doc="The frequency scaling factors for the RoPE.",
        tree=False,
    )]


    def init_from_args(self, args: PositionEncoderArgs):
        super().init_from_args(args)
        self.dims = args.dims
        self.traditional = args.traditional
        self.base = args.base
        self.scale = args.scale
        self.scaling = args.scaling
        self.freqs = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dims,

    def _extra_structure(self):
        return f"{self.dims}, traditional={self.traditional}"

    def build_call(self, mode: CompiledModule.Mode, **options) -> EncodePosition:
        return build_fast_rope_call(self.dims, traditional=self.traditional, base=self.base,
                                    scale=self.scale, freqs=self.freqs)


@provides(PositionEncoder, 'rope-longrope', 'rope-su')
class SuScaledRoPE(RoPE):

    __slots__ = ()


    def init_from_args(self, args: PositionEncoder.Args):
        super().init_from_args(args)

        self.scaling = scaling = args.scaling

        dims = self.dims
        base = self.base

        partial_rotary_factor = args.partial_rotary_factor
        if partial_rotary_factor != 1.0:
            dims = int(dims * partial_rotary_factor)

        long_factor = scaling.get('long_factor', 1.0)
        max_positions = args.max_positions
        original_max_positions = args.original_max_positions

        freqs = base ** (ten.arange(0, dims, 2, dtype=ten.float32) / dims)

        self.freqs = ten.array(long_factor, dtype=ten.float32) * freqs

        def default_scale(f):
            return math.sqrt(1 + math.log(f) / math.log(original_max_positions))

        factor = max_positions / original_max_positions
        self.scale = 1.0 if factor <= 1.0 else default_scale(factor)

    def build_call(self, mode: CompiledModule.Mode, **options) -> EncodePosition:

        partial_rotary_factor = self.args.partial_rotary_factor

        if partial_rotary_factor < 1.0:
            dims = int(self.dims * partial_rotary_factor)

            rope = build_fast_rope_call(dims, traditional=self.traditional, base=None,
                                        scale=1.0, mscale=self.scale, freqs=self.freqs)

            def call(x: Array, offset: int = 0) -> Array:
                partial = rope(x[..., :dims], offset)
                return ten.concatenate([partial, x[..., dims:]], axis=-1)

            return call

        return build_fast_rope_call(self.dims, traditional=self.traditional, base=None,
                                    scale=1.0, mscale=self.scale, freqs=self.freqs)



@provides(PositionEncoder, 'rope-llama3')
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

    def build_call(self, mode: CompiledModule.Mode, **options) -> EncodePosition:
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

        rope = build_fast_rope_call(dims, traditional=traditional, base=None,
                                    scale=1.0, mscale=1.0, freqs=freqs)

        cls.cache[key] = rope
        return rope


@provides(PositionEncoder, 'rope-yarn')
class YarnRoPE(RoPE):

    __slots__ = ('mscale')

    mscale: Annotated[float, field(
    )]

    def init_from_args(self, args: PositionEncoderArgs):
        super().init_from_args(args)
        dims = self.dims
        base = self.base
        scaling = self.scaling

        scaling_factor = scaling.get('scaling_factor', 1.0)
        beta_fast = scaling.get('beta_fast', 32)
        beta_slow = scaling.get('beta_slow', 1)
        mscale = scaling.get('mscale', 1)
        mscale_all_dim = scaling.get('mscale_all_dim', 0)

        original_max_positions = args.original_max_positions

        def yarn_find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                original_max_positions / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity

            linear_func = (ten.arange(dim, dtype=ten.float32) - min_val) / (
                max_val - min_val
            )
            return ten.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (ten.arange(0, dims, 2, dtype=ten.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self.freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )

    def build_call(self, mode: CompiledModule.Mode, **options) -> EncodePosition:
        return build_fast_rope_call(self.dims, traditional=self.traditional, base=None,
                                    scale=1.0, mscale=self.mscale, freqs=self.freqs)


__all__ = [
    'PositionEncoder',
]