#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

import math

from ..common import *
from ..module import CompiledModule
from ..quantization import QuantizableModuleArgs


class EmbeddingArgs(QuantizableModuleArgs):
    num_embeddings: Annotated[int, field(
        doc="The number of embeddings to learn.",
        aliases=['input_dim'],
        inherit="vocab_size",
    )]
    output_dim: Annotated[int, field(
        doc="The dimensionality of the embeddings.",
        inherit="hidden_size",
    )]


class Embedding(CompiledModule):
    """Implements a simple lookup table that maps each input integer to a
    high-dimensional vector.

    Typically used to embed discrete tokens for processing by neural networks.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
           Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
    """

    __slots__ = ('weight', )

    weight: Annotated[Array, field(
        doc="The learnable weights of the layer.",
        parameter=True,
    )]

    def init_from_args(self, args: EmbeddingArgs):
        super().init_from_args(args)
        dims = args.output_dim
        num_embeddings = args.num_embeddings

        scale = math.sqrt(1 / dims)
        self.weight = ten.random.normal(shape=(num_embeddings, dims), scale=scale)

    @property
    def num_embeddings(self) -> int:
        return 0 if (w := self.weight) is None else w.shape[0]

    @property
    def out_dim(self) -> int:
        return 0 if (w := self.weight) is None else w.shape[1]

    @property
    def in_dim(self) -> int:
        return self.num_embeddings

    def build_call(self, train: bool = False, **options):
        def call(x):
            return self.weight[x]
        return call

    def _extra_structure(self):
        return f'{self.num_embeddings}, {self.output_dim}'

    def as_linear(self, x):
        """
        Call the embedding layer as a linear layer.

        Use this for example when input embedding and output projection
        weights are tied.
        """
        return x @ self.weight.T

    @classmethod
    def refine_implementation(cls, args: EmbeddingArgs) -> type[Self]:
        return QuantizedEmbedding if args.quantization.group_size > 0 else cls

    Args = EmbeddingArgs

    # def to_quantized(self, group_size: int = 64, bits: int = 4):
    #     """Return a :obj:`QuantizedEmbedding` layer that approximates this embedding layer."""
    #     return QuantizedEmbedding.from_embedding(self, group_size, bits)


class QuantizedEmbedding(CompiledModule):
    """The same as :obj:`Embedding` but with a  quantized weight matrix.

    :obj:`QuantizedEmbedding` also provides a :meth:`from_embedding`
    classmethod to convert embedding layers to :obj:`QuantizedEmbedding`
    layers.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
           Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
        group_size (Optional[int]): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``None``.
        bits (Optional[int]): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``None``.
        mode (str): The quantization method to use (see
           :func:`mlx.core.quantize`). Default: ``"affine"``.
    """

    __slots__ = ('num_embeddings', 'output_dims', 'weight', 'scales', 'biases', 'group_size', 'bits', 'mode')

    num_embeddings: int
    output_dims: int
    weight: Annotated[Array, field(
        doc="The learnable weights of the layer.",
        parameter=False,
    )]
    scales: Annotated[Array, field(
        doc="The learnable scales of the layer.",
        parameter=False,
    )]
    biases: Annotated[Optional[Array], field(
        doc="The learnable biases of the layer.",
        parameter=False,
    )]
    group_size: Annotated[int, field(
        doc="The group size to use for the quantized weight.",
        default=64,
    )]
    bits: Annotated[int, field(
        doc="The number of bits to use for quantization.",
        default=8,
    )]
    mode: Annotated[str, field(
        doc="The quantization mode to use. Currently only ``affine`` is supported.",
        default="affine",
    )]

    def init_from_args(self, args: EmbeddingArgs):
        super().init_from_args(args)
        dims = args.output_dim
        num_embeddings = args.num_embeddings

        quant = args.quantization

        if quant is None:
            raise ValueError()

        # Quantization config
        group_size = quant.get("group_size", 64)
        bits = quant.get("bits", 8)
        mode = quant.get("mode", "affine")

        self.group_size, self.bits = _defaults_for_mode(mode, group_size, bits)
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1 / dims)
        weight = ten.random.normal(shape=(num_embeddings, dims), scale=scale)

        self.weight, self.scales, *biases = ten.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None
        self.num_embeddings = num_embeddings
        self.output_dims = dims

        # Freeze this model's parameters
        self.freeze()

    @property
    def in_dim(self) -> int:
        return self.num_embeddings

    @property
    def out_dim(self) -> int:
        return self.output_dims

    def build_call(self, train: bool = False, **options) -> Callable:

        group_size = self.group_size
        bits = self.bits
        mode = self.mode

        def call(x):
            biases = self.biases
            # ten.debug_eval(x)
            y = ten.dequantize(
                self.weight[x],
                scales=self.scales[x],
                biases=biases[x] if biases is not None else None,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            return y
        return call

    def as_linear(self, x):
        """
        Call the quantized embedding layer as a quantized linear layer.

        Use this for example when input embedding and output projection
        weights are tied.
        """
        return ten.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def _extra_structure(self):
        return (
            f"{self.num_embeddings}, {self.output_dims}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    @classmethod
    def from_embedding(
        cls,
        embedding_layer: Embedding,
        group_size: int = None,
        bits: int = None,
        mode: str = "affine",
    ):
        """Create a :obj:`QuantizedEmbedding` layer from an :obj:`Embedding` layer."""
        num_embeddings, dims = embedding_layer.weight.shape
        ql = cls(num_embeddings=num_embeddings, output_dims=dims,
                 group_size=group_size, bits=bits, mode=mode)
        ql.weight, ql.scales, *biases = ten.quantize(
            embedding_layer.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None
        return ql


def _defaults_for_mode(mode, group_size, bits):
    mode_defaults = {
        "affine": (64, 4),
        "mxfp4": (32, 4),
        "nvfp4": (16, 4),
        "mxfp8": (32, 8),
    }
    default_group_size, default_bits = mode_defaults[mode]
    return group_size or default_group_size, bits or default_bits
