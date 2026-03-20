#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..nn.module import *
from ..nn.layers.lm import LM, LMArgs
from ..infra import field, meta, provides
from ..infra.types import *
from ..ten import Array
from .model import Model, ModelArgs

if TYPE_CHECKING:
    import tensile.nn.attention.context


class LanguageModelArgs(ModelArgs):
    model_type: str = 'language'
    lm: LMArgs
    lm_head: Optional[ModuleArgs] = None


@provides(Model, 'language')
class LanguageModel(Model):

    __slots__ = ('lm', 'lm_head')

    args: Annotated[LanguageModelArgs, field(
        doc="The arguments for this model.",
        init_order=0,
    )]
    lm: Annotated[LM, field(
        doc="The LM model instance.",
    )]
    lm_head: Annotated[Optional[Module], field(
        doc="The optional LM head.",
    )]

    def init_from_args(self, args: LanguageModelArgs):
        super().init_from_args(args)
        self.lm = self.build_lm(args)
        self.lm_head = self.build_lm_head(args)

    def build_lm(self, args: LanguageModelArgs) -> LM:
        return LM.from_args(args.lm)

    def build_lm_head(self, args: LanguageModelArgs) -> Optional[Module]:
        if lm_head_args := args.lm_head:
            lm_head_args.set_defaults(
                in_dim=self.lm.hidden_dim,
                out_dim=self.lm.vocab_size,
            )
            return self.init_proj(lm_head_args)
        return None

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        lm = self.lm
        lm_head = self.lm_head

        if lm_head is None:
            embed_tokens = lm.embed_tokens

            lm_head = getattr(embed_tokens, 'as_linear')

        def call(inputs: Array):
            out = lm(inputs)
            return lm_head(out)
        return call

    def build_forward_context(self, model: Module = None, **kwargs) -> 'tensile.nn.attention.context.AttentionContext':
        if model is None: model = self
        return self.lm.build_forward_context(model=model, **kwargs)

    @property
    def in_dim(self) -> int:
        if lm := self.lm:
            return lm.in_dim
        return -1

    @property
    def out_dim(self) -> int:
        if lm_head := self.lm_head:
            return lm_head.out_dim
        elif lm := self.lm:
            return lm.in_dim
        return super().out_dim

    @property
    def layers(self):
        return self.lm.layers

    default_weight_aliases = {
        'model': 'lm'
    }

    Args = LanguageModelArgs


meta.for_class(LanguageModel).configure_registry(
    modules='tensile.models',
    append_kind=True,
)

provides(LanguageModel, 'default')(LanguageModel)

