#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

from ..common import *
# from ...util import normalize
from ..module import *
from ..attention import Attention
from .dropout import Dropout
from .mlp import MLP
from .normalization import Normalization


if TYPE_CHECKING:
    import patchlm.cache


class DecoderLayerArgs(ModuleArgs):

    kind: Annotated[str, field(inherit='layer_kind')] = 'transformer'
    hidden_size: int

    attention: Attention.Args
    # intermediate_size: int
    # rms_norm_eps: float
    # num_attention_heads: int
    # head_dim: Optional[int] = None
    # num_key_value_heads: Optional[int] = None
    # attention_bias: bool = False
    mlp: MLP.Args
    layernorm: Normalization.Args
    input_layernorm: Annotated[Normalization.Args, field(
        doc="The input normalization layer of the decoder layer",
        aliases=['layernorm'],
    )]
    post_attention_layernorm: Annotated[Normalization.Args, field(
        doc="The post-attention normalization layer of the decoder layer",
        aliases=['layernorm'],
    )]
    # mlp_bias: bool = False
    # rope_theta: float = 10000
    # rope_traditional: bool = False
    # rope_scaling: Optional[dict[str, Union[float, str]]] = None


class DecoderLayer(CompiledModule):

    __slots__ = ('hidden_size', 'self_attn', 'mlp', 'input_layernorm',
                 'post_attention_layernorm')

    # num_attention_heads: Annotated[int, field(
    #     doc="The number of attention heads in the layer",
    # )]
    hidden_size: Annotated[int, field(
        doc="The hidden size of the layer",
    )]
    self_attn: Annotated[Attention, field(
        doc="The self-attention module of the decoder layer",
        changed=CompiledModule.recompile,
    )]
    mlp: Annotated[MLP, field(
        doc="The MLP module of the decoder layer",
        changed=CompiledModule.recompile,
    )]
    input_layernorm: Annotated[Module, field(
        doc="The input layer normalization module of the decoder layer",
        changed=CompiledModule.recompile,
    )]
    post_attention_layernorm: Annotated[Module, field(
        doc="The post-attention layer normalization module of the decoder layer",
        changed=CompiledModule.recompile,
    )]

    def init_from_args(self, args: DecoderLayerArgs):
        super().init_from_args(args)

        # num_attention_heads = args.num_attention_heads
        hidden_size = args.hidden_size

        # self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.self_attn = self.build_self_attn(args)
        self.mlp = self.build_mlp(args)
        self.input_layernorm = self.build_input_layernorm(args)
        self.post_attention_layernorm = self.build_post_attention_layernorm(args)

    @property
    def in_dim(self) -> int:
        return self.hidden_size

    @property
    def out_dim(self) -> int:
        return self.hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self.self_attn.n_heads

    def build_self_attn(self, args: DecoderLayerArgs) -> Attention:
        return Attention.from_args(args.attention)

    def build_mlp(self, args: DecoderLayerArgs) -> MLP:
        mlp_args = args.mlp.set_defaults(
            in_dim=self.hidden_size,
            out_dim=self.hidden_size,
        )
        return MLP.from_args(mlp_args)

    def build_input_layernorm(self, args: DecoderLayerArgs) -> Module:
        norm_args = args.input_layernorm.set_defaults(
            dims=self.hidden_size,
            kind='rms',
        )
        return Normalization.from_args(norm_args)

    def build_post_attention_layernorm(self, args: DecoderLayerArgs) -> Module:
        norm_args = args.post_attention_layernorm.set_defaults(
            dims=self.hidden_size,
            kind='rms',
        )
        return Normalization.from_args(norm_args)

    # def build_layernorm(self, args: DecoderLayerArgs) -> Module:
    #     norm_args = args.layernorm.set_defaults(
    #         dims=self.hidden_size,
    #         eps=args.rms_norm_eps,
    #         kind='rms',
    #     )
    #     return Normalization.from_args(norm_args)

    def nilpotent(self, scale: float = None, precision: DType = None):
        self.self_attn.nilpotent()
        self.mlp.nilpotent()

    Args = DecoderLayerArgs


meta.for_class(DecoderLayer).configure_registry(
    modules='patchlm.models',
    append_kind=True,
    default_kind='transformer'
)


@meta.provides(DecoderLayer, 'transformer')
class TransformerBlock(DecoderLayer):

    __slots__ = ()

    def build_call(self, train: bool = False, **options) -> Callable:
        input_layernorm = self.input_layernorm
        self_attn = self.self_attn
        post_attention_layernorm = self.post_attention_layernorm
        mlp = self.mlp

        if train:
            if dropout := self.args.dropout:
                if 'self_attn' in dropout:
                    self_attn = Dropout.dropout(self_attn, p=dropout['self_attn'])
                if 'mlp' in dropout:
                    mlp = Dropout.dropout(mlp, p=dropout['mlp'])

        def call(x: Array) -> Array:
            """
            Processes the input tensor `x` through a series of transformations including
            layer normalization, self-attention, and a multi-layer perceptron (MLP). The
            function performs residual connections after both the self-attention and MLP
            operations.

            :param x: Input tensor to be processed. (B, L, D)
            :type x: Array
            :return: Processed tensor after applying layer normalization, self-attention,
                MLP, and residual connections.
            :rtype: Array
            """

            def test():
                def stats(desc: str, y: Array):
                    m = ten.max(ten.abs(y), axis=-1)
                    print(f"{desc} Max Abs Min: {ten.min(m)}, Max: {ten.max(m)}, Mean: {ten.mean(m)}")
                    r = ten.sqrt(ten.mean(ten.pow(y, 2), axis=-1))
                    rmsmin = ten.min(r)
                    rmsmax = ten.max(r)
                    rmsmean = ten.mean(r)
                    print(f"{desc} RMS Min: {rmsmin}, Max: {rmsmax}, Mean: {rmsmean}")
                    return
                stats('input', x)
                stats('x_norm', x_norm)
                ilwm = ten.abs(input_layernorm.weight)
                print(f"weights: Max Abs Min: {ten.min(ilwm)}, Max: {ten.max(ilwm)}, Mean: {ten.mean(ilwm)}")

            x_norm = input_layernorm(x)
            r = self_attn(x_norm)
            h = x + r
            h_norm = post_attention_layernorm(h)
            r = mlp(h_norm)
            out = h + r
            return out

        return call


def reflect(x: Array, b: Array) -> Array:
    """Computes the reflection of vector x across the plane normal to vector b using Geometric Algebra (-bxb).

    Args:
        x: Input vector to be reflected.
        b: Normal vector to the reflection plane. Must be a unit vector for proper reflection.

    Returns:
        The reflected vector.
    """
    return x - 2.*ten.matmul(ten.expand_dims(x, axis=-2), ten.expand_dims(b, axis=-1))[..., 0] * b


def rotate(x: Array, a: Array, b: Array) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    return reflect(reflect(x, b), a)


def ensure_dtype(x: Array, dtype: DType) -> Array:
    return x if x.dtype is dtype else ten.as_type(x, dtype)


def dtype_ensurer(dtype: DType) -> Callable[[Array], Array]:
    def ensure(x: Array) -> Array:
        return x if x.dtype is dtype else ten.as_type(x, dtype)
    return ensure


ensure_f16 = dtype_ensurer(ten.float16)
ensure_f32 = dtype_ensurer(ten.float32)


def rotate_ensure(x: Array, a: Array, b: Array, ensure: Callable[[Array], Array]) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    r = reflect(reflect(ensure(x), ensure(b)), ensure(a))
    return ensure_dtype(r, x.dtype)


def rotate_f16(x: Array, a: Array, b: Array) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    return rotate_ensure(x, a, b, ensure_f16)


def rotate_f32(x: Array, a: Array, b: Array) -> Array:
    """Computes the rotation using Geometric Algebra via abxba """
    return rotate_ensure(x, a, b, ensure_f32)


precision_rotators = {
    'f16': rotate_f16,
    'f32': rotate_f32,
    'default': rotate_f32,
}


def pct_diff(a: Array, b: Array) -> Array:
    return (b-a)/ten.sqrt(ten.sum(ten.square(a), axis=-1, keepdims=True))


@meta.provides(DecoderLayer, 'transformer.rotational')
@meta.provides(TransformerBlock, 'rotational')
class RotationalTransformerBlock(DecoderLayer):
    """A transformer block that uses geometric rotations instead of additive attention.

    This block modifies the standard transformer by replacing the additive attention
    mechanism with a geometric rotation. Instead of adding the attention output to
    the input, it reflects the input across a plane determined by the attention
    output plus a reference direction (e0 by default, but can be set by reference_dim).

    The rotation is implemented efficiently using vector operations:
    1. Compute attention output b
    2. Add reference direction to b[reference_dim]
    3. Reflect input x across b using h = x - 2(b•x)b
    4. Flip sign of reference dim component h[reference_dim]

    This geometric transformation allows the model to capture relationships that
    would be difficult to represent through pure addition.
    """

    reference_dim: int

    # kind = 'rotational'

    def init_from_args(self, args: DecoderLayerArgs, reference_dim: int = 0, **kwargs):
        super().init_from_args(args)
        self.reference_dim = reference_dim

    def build_call(self, train: bool = False, **options) -> Callable:
        input_layernorm = self.input_layernorm
        self_attn = self.self_attn
        post_attention_layernorm = self.post_attention_layernorm
        mlp = self.mlp
        reference_dim = self.reference_dim

        def decode(x: Array, call: 'patchlm.cache.ModelCall', cache: Optional['patchlm.cache.KVCache'] = None) -> Array:
            """Apply rotational transformer block to input.

            Args:
                x: Input tensor of shape [batch, seq_len, hidden_size]
                call: ModelCall object containing attention mask and other parameters
                cache: Optional KV cache for efficient inference

            Returns:
                Output tensor of same shape as input after applying attention,
                rotation, and MLP transformations.
            """
            x_norm = input_layernorm(x)

            b = self_attn(x_norm)

            b[..., reference_dim] += 1.

            b_norm = ten.norm(b)

            x_f32 = ensure_f32(x)
            b_f32 = ensure_f32(b_norm)

            h = reflect(x_f32, b_f32)
            # ax.eval(h)
            h[..., reference_dim] = -h[..., reference_dim]
            h = ensure_dtype(h, x.dtype)

            h_norm = post_attention_layernorm(h)
            r = mlp(h_norm)
            out = h + r

            return out

        return decode


@meta.provides(DecoderLayer, 'transformer.rotational.learned')
@meta.provides(TransformerBlock, 'rotational.learned')
@meta.provides(RotationalTransformerBlock, 'learned')
class LearnedReferenceRotationalTransformerBlock(DecoderLayer):
    """A transformer block that uses geometric rotations instead of additive attention.

    This block modifies the standard transformer by replacing the additive attention
    mechanism with a geometric rotation. Instead of adding the attention output to
    the input, it reflects the input across a plane determined by the attention
    output plus a reference direction (e0 by default, but can be set by reference_dim).

    The rotation is implemented efficiently using vector operations:
    1. Compute attention output b
    2. Add reference direction to b[reference_dim]
    3. Reflect input x across b using h = x - 2(b•x)b
    4. Flip sign of reference dim component h[reference_dim]

    This geometric transformation allows the model to capture relationships that
    would be difficult to represent through pure addition.
    """

    reference_vector: Array
    rotate_dtype: Callable[[Array, Array, Array], Array]

    # kind = 'rotational.learned'

    def init_from_args(self, args: DecoderLayerArgs, rotation_dtype: str = 'default', **kwargs):
        super().init_from_args(args)

        self.reference_vector = ten.random.normal(shape=(self.hidden_size,))
        self.rotate_dtype = precision_rotators[rotation_dtype]

    def build_call(self, train: bool = False, **options) -> Callable:
        input_layernorm = self.input_layernorm
        self_attn = self.self_attn
        post_attention_layernorm = self.post_attention_layernorm
        mlp = self.mlp
        reference_vector = self.reference_vector
        rotate_dtype = self.rotate_dtype

        def decode(x: Array, call: 'patchlm.cache.ModelCall', cache: Optional['patchlm.cache.KVCache'] = None) -> Array:
            """Apply rotational transformer block to input.

            Args:
                x: Input tensor of shape [batch, seq_len, hidden_size]
                call: ModelCall object containing attention mask and other parameters
                cache: Optional KV cache for efficient inference

            Returns:
                Output tensor of same shape as input after applying attention,
                rotation, and MLP transformations.
            """
            x_norm = input_layernorm(x)

            a = ten.norm(reference_vector)

            b_delta = self_attn(x_norm, call, cache)

            b = ten.norm(b_delta + a)

            # b_delta = (b-a)[0]

            h = rotate_dtype(x, a, b)

            # h_delta = pct_diff(x, h)[0]
            # h_delta = (h-x)[0]

            h_norm = post_attention_layernorm(h)
            r = mlp(h_norm)
            out = h + r

            # delta = pct_diff(x, out)[0]
            # delta = (out - x)[0]
            # ax.eval(delta)

            # ax.eval(delta, b_delta, h_delta)
            # print('b', ax.max(b_delta, axis=-1))
            # print('h', ax.max(h_delta, axis=-1))
            # print('x', ax.max(delta, axis=-1))

            return out

        return decode
