#  Copyright (c) 2025-2026. Richard Vermillion. All Rights Reserved.


from ..common import *
from ..module import FunctionModule, Module, CompiledModule
from ..attention.context import AttentionContext
from ..cache import ModelCache
from ..quantization import QuantizableModuleArgs
from .embedding import Embedding
from .normalization import Normalization
from .transformer import DecoderLayer


class LMArgs(QuantizableModuleArgs):
    model_type: Annotated[str, 'Model type identifier']
    hidden_dim: int
    vocab_size: int
    window_size: Optional[int] = None
    tokenizer: Optional[str] = None

    embedding: Embedding.Args = None
    layers: Sequence[DecoderLayer.Args]
    norm: Normalization.Args = None

    _config_default_step = 'model'


@provides(Module, "lm")
class LM(FunctionModule):

    __slots__ = ('embed_tokens', 'layers', 'norm', 'vocab_size', 'window_size',
                 'num_hidden_layers', 'hidden_dim')

    args: Annotated[LMArgs, field(ignore=True)]

    vocab_size: int
    window_size: Annotated[int, field(
        default=0
    )]
    num_hidden_layers: int
    hidden_dim: int

    embed_tokens: Module
    layers: list[DecoderLayer]
    norm: Module

    def init_from_args(self, args: LMArgs):
        super().init_from_args(args)

        self.vocab_size = args.vocab_size
        self.hidden_dim = args.hidden_dim
        self.window_size = args.window_size or 0
        self.num_hidden_layers = args.get_first('layers.count', 'num_hidden_layers')  #args.num_hidden_layers

        assert self.vocab_size > 0
        assert self.num_hidden_layers > 0

        self.embed_tokens = self.build_embed_tokens(args)
        self.layers = self.build_layers(args)
        self.norm = self.build_norm(args)

    def build_embed_tokens(self, args: LMArgs) -> Module:
        embedding_args = args.embedding.set_defaults(
            num_embeddings=self.vocab_size,
            output_dim=self.hidden_dim,
        )
        return Embedding.from_args(embedding_args)

    def build_layers(self, args: LMArgs) -> list[DecoderLayer]:
        return [
            self.build_layer(args, l) for l in range(self.num_hidden_layers)
        ]

    # noinspection PyMethodMayBeStatic
    def build_layer(self, args: LMArgs, l: int) -> DecoderLayer:
        layer_args = args.layers[l]
        return DecoderLayer.from_args(layer_args)

    def build_norm(self, args: LMArgs) -> Module:
        norm_args = args.norm.set_defaults(
            dims=self.hidden_dim,
            kind='rms',
        )
        return Normalization.from_args(norm_args)

    def build_cache(self) -> Optional[ModelCache]:
        return meta.coerce(ModelCache, model=self)

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        embed_tokens = self.embed_tokens
        layers = self.layers
        norm = self.norm
        if mode.is_train():
            def build_cache():return None
        else:
            build_cache = self.build_cache

        def call(inputs: Array) -> Array:
            # ten.debug_eval(inputs)
            h = embed_tokens(inputs)

            if ctx := AttentionContext.get_current():
                cache = ctx.cache

                if cache is None:
                    cache = ctx.cache = build_cache()

            else:
                ctx = AttentionContext(model=self)
                cache = None

            with ctx.push():

                if cache is None:
                    for layer in layers:
                        h = layer(h)
                else:
                    for layer, cache_layer in zip(layers, cache.layers):
                        ctx.layer_cache = cache_layer
                        h = layer(h)

                out = norm(h)

            return out
        return call

    def build_forward_context(self, model: 'Module' = None, **kwargs) -> AttentionContext:
        if model is None: model = self
        return AttentionContext.coerce(model=model, window_size=self.window_size, **kwargs)

    @property
    def in_dim(self) -> int:
        return self.vocab_size

    @property
    def out_dim(self) -> int:
        return self.hidden_dim

    ForwardContext = AttentionContext


meta.for_class(LM).configure_registry(
    modules='tensile.models',
    append_kind=True,
)


@meta.provides(LM, 'standard')
class StandardLanguageModel(LM):

    pass
    # kind = 'standard'


