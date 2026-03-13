#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.

# import patchlm.cache

from .common import *
from .module import ForwardContext, Module, CompiledModule
from ..nn.layers import DecoderLayer, Embedding, Normalization
from ..nn.attention.mask import AttentionMasker, create_causal_mask, make_additive_masker
from .cache import ModelCache, KVCache
from .quantization import QuantizableModuleArgs


class LanguageModelArgs(QuantizableModuleArgs):
    model_type: Annotated[str, 'Model type identifier']
    hidden_size: int
    vocab_size: int
    tokenizer: Optional[str] = None

    embedding: Embedding.Args = None
    layers: Sequence[DecoderLayer.Args]
    norm: Normalization.Args = None

    _config_default_step = 'model'


class LanguageModelContext(ForwardContext):

    __slots__ = ('mask', 'cache', 'layer_cache', 'call')

    mask: Annotated[Optional[Array], field(
        doc='The mask to apply to the input'
    )]
    cache: Annotated[Optional[ModelCache], field(
        doc='The cache to use for this forward pass'
    )]
    layer_cache: Annotated[Optional[KVCache], field(
        doc='The kv cache for the current layer to use for this forward pass'
    )]

    def get_mask(self, n: int, dtype: DType = ten.float32) -> Optional[Array]:
        if n == 1:
            return None
        mask = self.mask
        if mask is not None and mask.dtype == dtype:
            if n <= mask.shape[0]:
                # reuse a portion of the mask if we can
                return mask[:n, :n]
        mask = create_causal_mask(n, dtype=dtype)
        self.mask = mask
        return mask

    def get_masker(self, n: int, dtype: DType = ten.float32) -> Optional[AttentionMasker]:
        if n == 1: return None
        return make_additive_masker(self.get_mask(n, dtype=dtype))


@provides(Module, "lm")
class LanguageModel(CompiledModule):

    __slots__ = ('embed_tokens', 'layers', 'norm', 'vocab_size',
                 'num_hidden_layers', 'hidden_size')

    args: Annotated[LanguageModelArgs, field(ignore=True)]

    vocab_size: int
    num_hidden_layers: int
    hidden_size: int

    embed_tokens: Module
    layers: list[DecoderLayer]
    norm: Module

    def init_from_args(self, args: LanguageModelArgs):
        super().init_from_args(args)

        self.vocab_size = args.vocab_size
        self.hidden_size = args.hidden_size
        self.num_hidden_layers = args.get_first('layers.count', 'num_hidden_layers')  #args.num_hidden_layers

        assert self.vocab_size > 0
        assert self.num_hidden_layers > 0

        self.embed_tokens = self.build_embed_tokens(args)
        self.layers = self.build_layers(args)
        self.norm = self.build_norm(args)

    def build_embed_tokens(self, args: LanguageModelArgs) -> Module:
        embedding_args = args.embedding.set_defaults(
            num_embeddings=self.vocab_size,
            output_dim=self.hidden_size,
        )
        return Embedding.from_args(embedding_args)

    def build_layers(self, args: LanguageModelArgs) -> list[DecoderLayer]:
        return [
            self.build_layer(args, l) for l in range(self.num_hidden_layers)
        ]

    # noinspection PyMethodMayBeStatic
    def build_layer(self, args: LanguageModelArgs, l: int) -> DecoderLayer:
        layer_args = args.layers[l]
        return DecoderLayer.from_args(layer_args)

    def build_norm(self, args: LanguageModelArgs) -> Module:
        norm_args = args.norm.set_defaults(
            dims=self.hidden_size,
            kind='rms',
        )
        return Normalization.from_args(norm_args)

    def build_cache(self) -> Optional[ModelCache]:
        if self.training: return None
        return meta.coerce(ModelCache, model=self)

    def build_call(self, train: bool = False, **options) -> Callable:

        def call(inputs: Array) -> Array:
            ten.debug_eval(inputs)
            h = self.embed_tokens(inputs)

            if ctx := LanguageModelContext.get_current():
                cache = ctx.cache

                if cache is None:
                    cache = ctx.cache = self.build_cache()

            else:
                ctx = LanguageModelContext(model=self)
                cache = None

            with ctx.push():

                if cache is None:
                    for layer in self.layers:
                        h = layer(h)
                else:

                    for layer, cache_layer in zip(self.layers, cache.layers):
                        ctx.layer_cache = cache_layer
                        h = layer(h)

                out = self.norm(h)


            return out
        return call

    @property
    def in_dim(self) -> int:
        return self.vocab_size

    @property
    def out_dim(self) -> int:
        return self.hidden_size

    ForwardContext = LanguageModelContext


meta.for_class(LanguageModel).configure_registry(
    modules='patchlm.models',
    append_kind=True,
)


@meta.provides(LanguageModel, 'standard')
class StandardLanguageModel(LanguageModel):

    pass
    # kind = 'standard'


