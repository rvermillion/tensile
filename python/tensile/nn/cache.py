#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from pathlib import Path

from ..infra import Loadable, Storable
from ..infra.util import noop
from .attention import Attend, AttentionMasker
from .attention.mask import create_causal_mask
from .attention.kv import KVBuffer

from .common import *
from ..util.buffer import ArrayBuffer

if TYPE_CHECKING:
    import tensile.nn
    import tensile.nn.layers



class KVCache(Object, Storable, Loadable):

    __slots__ = ('skipped_keys', 'attended_keys')

    skipped_keys: Annotated[int, field(default=0)]
    attended_keys: Annotated[int, field(default=0)]

    _is_segmented: ClassVar[bool] = False

    @property
    def offset(self) -> int:
        return 0

    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False

    def seek(self, position: int):
        raise NotImplementedError()


    def attention(self, queries: Array, keys: Array, values: Array, /, *,
                  scale: float = None,
                  masker: AttentionMasker = ...,
                  attend: Attend = ...,
                  offset: int = 0,
                  update: bool = True,
                  **extra,
                  ) -> Array:
        raise NotImplementedError()

    # def partial_attention(self, queries: Array,
    #                       scale: float,
    #                       mask: Optional[Array]) -> tuple[Array, Array, Array]:
    #     raise NotImplementedError()

    def update_kv(self, keys: Array, values: Array) -> None:
        raise NotImplementedError()

    def update_and_fetch_kv(self, keys: Array, values: Array) -> tuple[Array, Array]:
        raise NotImplemented()

    def branch(self, branches: Array):
        raise NotImplemented()

    def arrays_to_save(self, arrays: dict, prefix: str = ''):
        raise NotImplementedError()

    def report(self, **kwargs):
        print('***', self)


class ModelCache(Object, Storable):

    __slots__ = ('model', 'layers', 'position', 'tokens', 'mask', 'square_causal_mask')

    model: Annotated['tensile.nn.lm.LM', field(
        doc="The model for which this cache is used."
    )]
    layers: Annotated[list[Optional[KVCache]], field(
        doc="The KV caches for each layer of the model."
    )]
    position: Annotated[int, field(
        doc="The current position in the sequence for which the cache is valid.",
        default=0
    )]
    tokens: Annotated[ArrayBuffer, field(
        doc="The sequence of tokens for which the cache is valid."
    )]
    mask: Annotated[Optional[Array], field(
        doc="The attention mask for the sequence of tokens."
    )]
    square_causal_mask: Annotated[bool, field(
        doc="Whether to use a square causal mask for attention.",
        default=True,
    )]

    def _lazy_tokens(self) -> ArrayBuffer:
        return ArrayBuffer()

    def _lazy_layers(self) -> list[Optional[KVCache]]:
        return [None] * len(self.model.layers)

    def branch(self, idx: Array):
        self.tokens.branch(idx)
        for layer in self.layers:
            layer.branch(idx)

    def rewind(self, steps: int):
        self.seek(self.position - steps)

    def seek(self, position: int):
        if position < 0 or position > self.position:
            raise ValueError(f'Position must be between 0 and {self.position}')
        self.tokens.length = position
        for layer in self.layers:
            layer.seek(position)
        self.position = position

    @property
    def skipped_segments(self) -> int:
        skipped = 0
        return skipped

    @property
    def attended_segments(self) -> int:
        attended = 0
        return attended

    @property
    def skipped_keys(self) -> int:
        skipped = 0
        for c in self.layers:
            skipped += c.skipped_keys
        return skipped

    @property
    def attended_keys(self) -> int:
        attended = 0
        for c in self.layers:
            attended += c.attended_keys
        return attended

    def create_mask(self, length: int, dtype: ten.DType):
        if length > 1:
            window_size = None
            if self.square_causal_mask:
                offset = 0
            else:
                cache = self.layers[0]
                if hasattr(cache, "max_size"):
                    offset = min(cache.max_size, cache.offset)
                    window_size = cache.max_size
                else:
                    offset = cache.offset
            mask = create_causal_mask(length, offset)
            mask = ten.as_type(mask, dtype)
        else:
            mask = None
        return mask

    def eval_state(self):
        ten.eval([c.state for c in self.layers])

    def save(self, file: str):
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving model cache in {path}')
        arrays = {}
        metadata = {}
        self.store(arrays, metadata)
        ten.save_safetensors(file, arrays, metadata)

    def store(self, arrays: dict, metadata: dict, prefix: str = ''):
        for lid, layer in enumerate(self.layers):
            layer.store(arrays, metadata, prefix=f'{prefix}layer.{lid}.')

        super().store(arrays, metadata, prefix)

    def report(self, **kwargs):
        for layer in self.layers:
            layer.report(**kwargs)
        attended, skipped = self.attended_segments, self.skipped_segments
        total = attended + skipped
        if total != 0:
            print(f'=== skipped {skipped} out of {total} segments ({100.0*skipped/total:.3f}%)')
        attended, skipped = self.attended_keys, self.skipped_keys
        total = attended + skipped
        if total > 0:
            print(f'=== skipped {skipped} out of {total} keys ({100.0*skipped/total:.3f}%)')

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterable[KVCache]:
        return iter(self.layers)

    def _repr_args(self) -> str:
        return f'layers={len(self.layers)}'


class LayerKVCache(KVCache):

    __slots__ = ['layer', 'layer_id', 'stream_id',
                 'kvs',
                 # 'keys', 'values', 'offset', 'step', 'batch_count', 'batch_step',
                 'n_heads', 'n_kv_heads', 'kv_heads_per_head', 'k_head_dim', 'v_head_dim',
                 # 'kv_capacity',
                 # 'skipped_keys', 'attended_keys',
                 'debug',
                 'print',
                 ]

    layer: Annotated['tensile.nn.layers.transformer.DecoderLayer', field(required=True)]
    layer_id: Annotated[int, field(required=True)]
    stream_id: Annotated[int, field(default=0)]
    kvs: KVBuffer
    n_heads: int
    n_kv_heads: int
    kv_heads_per_head: int
    k_head_dim: int
    v_head_dim: int
    print: Callable
    debug: Annotated[bool, field(default=False)]

    # def __init__(self, layer: 'patchlm.nn.lm.DecoderLayer', layer_id: int,
    #              stream_id: int = 0,
    #              step: int = 256, batch_step: int = 2, debug: bool = False):
    #     super().__init__()
    #     # layer = layer.unwrap()
    #
    #     # 2e^{a} = e^{b}  ln(2)+a=b  a=b-ln(2)
    #
    #     self.layer = layer
    #     self.layer_id = layer_id
    #     self.stream_id = stream_id
    #     self.kvs = KVBuffer(pos_start=0)
    #     self.skipped_keys = 0
    #     self.attended_keys = 0
    #     self.debug = debug
    #
    #     layer_attn = layer.self_attn
    #
    #     self.n_heads = layer_attn.n_heads
    #     self.n_kv_heads = layer_attn.n_kv_heads
    #     self.k_head_dim = layer_attn.k_head_dim
    #     self.v_head_dim = layer_attn.v_head_dim
    #     self.kv_heads_per_head = layer_attn.kv_heads_per_head
    #     self.print = print if debug else noop

    def postinit(self, spec: Spec):
        super().postinit(spec)
        self.kvs = KVBuffer(pos_start=0)

        layer_attn = self.layer.self_attn

        self.n_heads = layer_attn.n_heads
        self.n_kv_heads = layer_attn.n_kv_heads
        self.k_head_dim = layer_attn.k_head_dim
        self.v_head_dim = layer_attn.v_head_dim
        self.kv_heads_per_head = layer_attn.kv_heads_per_head
        self.print = print if self.debug else noop

    @property
    def offset(self) -> int:
        return self.kvs.pos_end

    def attention(self, queries: Array, keys: Array, values: Array, /, *,
                  scale: float = None,
                  masker: AttentionMasker = None,
                  attend: Attend = None,
                  offset: int = 0,
                  update: bool = True,
                  segment_queries: Array = None,
                  **extra,
                  ) -> Array:

        if attend is None:
            raise ValueError('Attend must be provided')


        # B, n_heads, Q, D = queries.shape
        Q = queries.shape[-2]

        if self.layer_id == 0:
            pass

        offset = self.offset

        if update:
            if keys.ndim < 5:
                keys = ten.expand_dims(keys, axis=2)
            if values.ndim < 5:
                values = ten.expand_dims(values, axis=2)
            keys, values = self.update_and_fetch_kv(keys, values)
        else:
            keys, values = self.fetch_kv(n_batches=keys.shape[0])

        self.attended_keys += self.offset * Q

        out = attend(queries, keys, values, scale=scale, masker=masker, offset=offset)

        return out

    def fire_grow(self, old_capacity: int, new_capacity: int):
        if self.layer_id == 0:
            self.print(f'Grew kv capacity from {old_capacity} to {new_capacity}')

    def update_kv(self, keys: Array, values: Array) -> None:
        self.kvs.update(keys, values)

    def fetch_kv(self, n_batches: int = None) -> tuple[Array, Array]:
        return self.kvs.fetch_kv(n_batches=n_batches)

    def update_and_fetch_kv(self, keys: Array, values: Array) -> tuple[Array, Array]:
        self.update_kv(keys, values)
        return self.fetch_kv(n_batches=keys.shape[0])

    def branch(self, branches: Array):
        self.kvs.branch(branches)

    # def cut_segment(self, start: int = 0, stop: int = None, n_batches: int = None, trim: bool = False) -> KVSegment:
    #     seq_slice = slice(start, stop)
    #     keys = self.keys[..., seq_slice, :]
    #     vals = self.values[..., seq_slice, :]
    #     if n_batches is None or n_batches > keys.shape[0]:
    #         segment = KVSegment(start, keys, vals)
    #     else:
    #         segment = KVSegment(start, keys[:n_batches, ...], vals[:n_batches, ...])
    #     if trim:
    #         self.keys = keys
    #         self.values = vals
    #         self.offset -= start
    #     return segment

    @property
    def state(self):
        return self.kvs.state

    @state.setter
    def state(self, v):
        self.kvs.state = v

    def is_trimmable(self):
        return True

    # def trim(self, n):
    #     n = min(self.offset, n)
    #     self.offset -= n
    #     return n

    def seek(self, position: int):
        if position < 0 or position > self.offset:
            raise ValueError(f'Seek position must be between 0 and {self.offset}')
        self.kvs.set_pos_end(position)

    def _store_arrays(self, arrays: dict, prefix: str = ''):
        keys, values = self.fetch_kv()
        arrays[prefix + 'keys'] = keys
        arrays[prefix + 'values'] = values

    def _repr_args(self) -> str:
        total = self.skipped_keys + self.attended_keys
        skipped_pct = 0.0 if total == 0 else 100.0 * self.skipped_keys / total
        return f'layer={self.layer_id}, length={self.offset}, attends={total}, skipped={skipped_pct:.2f}%'


__all__ = [
    'KVCache',
    'ModelCache',
]