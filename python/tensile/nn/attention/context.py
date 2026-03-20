#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


from ..common import *
from ..module import ForwardContext
from ...train import TrainingContext
from .mask import AttentionMasker, create_causal_mask, make_additive_masker

if TYPE_CHECKING:
    import tensile.nn.cache


@provides(ForwardContext, 'attention')
class AttentionContext(TrainingContext):

    __slots__ = ('mask', 'cache', 'layer_cache', 'window_size')

    mask: Annotated[Optional[Array], field(
        doc='The mask to apply to the input'
    )]
    cache: Annotated[Optional['tensile.nn.cache.ModelCache'], field(
        doc='The cache to use for this forward pass'
    )]
    layer_cache: Annotated[Optional['tensile.nn.cache.KVCache'], field(
        doc='The kv cache for the current layer to use for this forward pass'
    )]
    window_size: Annotated[int, field(
        doc='The window size to use for this forward pass',
        default=0,
    )]

    @property
    def offset(self) -> int:
        cache = self.layer_cache
        return 0 if cache is None else cache.offset

    def get_mask(self, n: int, dtype: DType = ten.float32) -> Optional[Array]:
        if n == 1:
            return None
        mask = self.mask
        if mask is not None and mask.dtype == dtype:
            if n <= mask.shape[0]:
                # reuse a portion of the mask if we can
                return mask[:n, :n]
        mask = create_causal_mask(n, window_size=self.window_size, dtype=dtype)
        self.mask = mask
        return mask

    def get_masker(self, n: int, dtype: DType = ten.float32) -> Optional[AttentionMasker]:
        if n == 1: return None
        return make_additive_masker(self.get_mask(n, dtype=dtype))


