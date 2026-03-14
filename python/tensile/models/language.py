#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..nn.module import *
from ..nn.lm import LM, LMArgs
from ..infra import field, meta, provides
from ..infra.types import *
from ..shims import Array
from .model import Model, ModelArgs


class LanguageModelArgs(ModelArgs):
    model_type: str = 'language'
    model: LMArgs
    lm_head: Optional[ModuleArgs] = None
    remap_weights: dict[str, str] = None


@provides(Model, 'language')
class LanguageModel(Model):

    __slots__ = ('model', 'lm_head')

    args: Annotated[LanguageModelArgs, field(
        doc="The arguments for this model.",
        init_order=0,
    )]
    model: Annotated[LM, field(
        doc="The LM model instance.",
    )]
    lm_head: Annotated[Optional[Module], field(
        doc="The optional LM head.",
    )]

    def init_from_args(self, args: LanguageModelArgs):
        super().init_from_args(args)
        self.model = self.build_model(args)
        self.lm_head = self.build_lm_head(args)

    def build_model(self, args: LanguageModelArgs) -> LM:
        return LM.from_args(args.model)

    def build_lm_head(self, args: LanguageModelArgs) -> Optional[Module]:
        if lm_head_args := args.lm_head:
            lm_head_args.set_defaults(
                input_dims=self.model.hidden_size,
                output_dims=self.model.vocab_size,
            )
            return self.build_proj_from_args(lm_head_args)
        return None

    def build_call(self, train: bool = False, **options) -> Callable:
        model = self.model
        lm_head = self.lm_head

        if lm_head is None:
            embed_tokens = self.model.embed_tokens

            tied_embeddings = getattr(embed_tokens, 'as_linear')

            def call(inputs: Array):
                out = model(inputs)
                return tied_embeddings(out)

        else:

            def call(inputs: Array):
                out = model(inputs)
                return lm_head(out)
        return call

    def build_forward_context(self, model: Module = None, **kwargs) -> ForwardContext:
        if model is None: model = self
        return self.model.build_forward_context(model=model, **kwargs)

    @property
    def in_dim(self) -> int:
        if model := self.model:
            return model.in_dim
        return -1

    @property
    def out_dim(self) -> int:
        if lm_head := self.lm_head:
            return lm_head.out_dim
        elif model := self.model:
            return model.in_dim
        return super().out_dim

    @property
    def layers(self):
        return self.model.layers


    Args = LanguageModelArgs


meta.for_class(LanguageModel).configure_registry(
    modules='tensile.models',
    append_kind=True,
)

provides(LanguageModel, 'default')(LanguageModel)

