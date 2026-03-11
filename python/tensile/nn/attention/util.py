#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..util import fast_sdp_attention
from .types import AttentionScorer, QKV


# noinspection PyPep8Naming
def reshape_qkv5(queries: Array, keys: Array, values: Array, reshape_four: bool = False, common_head_dim: bool = True) -> QKV:
    qdim = queries.ndim
    kdim = keys.ndim
    kshape = keys.shape
    qshape = queries.shape

    if qdim != 5:
        raise ValueError(f'queries should have 5 dimensions not {qdim}')
    if kdim != 5:
        raise ValueError(f'keys should have 5 dimensions not {kdim}')
    if values.ndim < 5:
        raise ValueError(f'values should have at least 5 dimensions not {values.ndim}')

    if kshape[:-1] != values.shape[:kdim-1]:
        raise ValueError(f'The keys and values should have the same batch and head dims: keys={keys.shape} values={values.shape}')

    B, n_q_heads, n_kv_heads_per_q_head, Q, D = qshape

    B_k, n_q_heads_per_kv_head, n_kv_heads, K, D_k = kshape

    if common_head_dim and D_k != D:
        raise ValueError(f'keys should have the same head dim as queries: {D_k} != {D}')
    if B_k != B:
        raise ValueError(f'keys should have the same batch dim as queries: {B_k} != {B}')

    if n_q_heads == n_q_heads_per_kv_head:
        if n_kv_heads_per_q_head == n_kv_heads or n_kv_heads_per_q_head == 1:
            return queries, keys, values
    if n_kv_heads == n_kv_heads_per_q_head:
        if n_q_heads_per_kv_head == n_q_heads or n_q_heads_per_kv_head == 1:
            return queries, keys, values
    if n_kv_heads_per_q_head == 1 and n_q_heads_per_kv_head == 1:
        return queries, keys, values

    q_n_heads = n_q_heads * n_kv_heads_per_q_head
    kv_n_heads = n_kv_heads * n_q_heads_per_kv_head

    if q_n_heads != kv_n_heads:
        raise ValueError(f'The number of heads must be equal: {q_n_heads} != {kv_n_heads}')

    if reshape_four:
        queries = queries.reshape(B, q_n_heads, Q, D)
        keys = keys.reshape(B, q_n_heads, K, D_k)
        values = values.reshape(B, q_n_heads, K, *values.shape[4:])
    else:
        raise ValueError(f'Need to reshape to four dims: {qshape} != {kshape}')

    return queries, keys, values


# noinspection PyPep8Naming
def reshape_qkv4(queries: Array, keys: Array, values: Array, force_five: bool = True, common_head_dim: bool = True) -> QKV:
    qdim = queries.ndim
    kdim = keys.ndim
    kshape = keys.shape
    qshape = queries.shape

    if qdim != 4:
        raise ValueError(f'queries should have 4 dimensions not {qdim}')
    if kdim != 4:
        raise ValueError(f'keys should have 4 dimensions not {kdim}')
    if values.ndim < 4:
        raise ValueError(f'values should have at least 4 dimensions not {values.ndim}')

    if kshape[:-1] != values.shape[:kdim-1]:
        raise ValueError(f'The keys and values should have the same batch and head dims: keys={keys.shape} values={values.shape}')

    B, n_q_heads, Q, D = qshape

    B_k, n_kv_heads, K, D_k = kshape

    if common_head_dim and D_k != D:
        raise ValueError(f'keys should have the same head dim as queries: {D_k} != {D}')
    if B_k != B:
        raise ValueError(f'keys should have the same batch dim as queries: {B_k} != {B}')

    if n_q_heads == n_kv_heads:
        if force_five:
            queries = ten.expand_dims(queries, axis=2)
            keys = ten.expand_dims(keys, axis=2)
            values = ten.expand_dims(values, axis=2)
        return queries, keys, values

    elif n_q_heads > n_kv_heads:
        if n_q_heads % n_kv_heads != 0:
            raise ValueError('The number of query heads must be divisible by the number of kv heads')
        queries_per_kv_head = n_q_heads // n_kv_heads
        queries = queries.reshape(B, n_kv_heads, queries_per_kv_head, Q, D)
        keys = ten.expand_dims(keys, axis=2)
        values = ten.expand_dims(values, axis=2)
    elif n_q_heads < n_kv_heads:
        if n_kv_heads % n_q_heads != 0:
            raise ValueError('The number of kv heads must be divisible by the number of query heads')
        kvs_per_q_head = n_kv_heads // n_q_heads
        queries = ten.expand_dims(queries, axis=2)
        keys = keys.reshape(B, n_q_heads, kvs_per_q_head, K, D_k)
        values = values.reshape(B, n_q_heads, kvs_per_q_head, K, *values.shape[3:])

    return queries, keys, values


# noinspection PyPep8Naming
def reshape_qkv(queries: Array, keys: Array, values: Array, force_five: bool = True, common_head_dim: bool = True) -> QKV:
    qdim = queries.ndim
    kdim = keys.ndim

    if qdim == 4:
        if kdim == 5:
            kshape = keys.shape
            vshape = values.shape
            keys = keys.reshape(kshape[0], kshape[1] * kshape[2], kshape[3], kshape[4])
            values = values.reshape(vshape[0], vshape[1] * vshape[2], vshape[3], vshape[4])
        if kdim == 4:
            return reshape_qkv4(queries, keys, values, force_five=force_five, common_head_dim=common_head_dim)
        elif kdim == 5:
            # TODO: Fix this. Raise and exception for now.
            raise NotImplementedError()
            queries = ten.expand_dims(queries, axis=2)
            return reshape_qkv5(queries, keys, values, reshape_four=not force_five, common_head_dim=common_head_dim)
        else:
            raise ValueError(f'keys should have 4 or 5 dimensions not {kdim}')
    elif qdim == 5:
        if kdim == 4:
            qshape = queries.shape
            queries = queries.reshape(qshape[0], qshape[1] * qshape[2], qshape[3], qshape[4])
            return reshape_qkv4(queries, keys, values, force_five=force_five, common_head_dim=common_head_dim)
        elif kdim == 5:
            return reshape_qkv5(queries, keys, values, reshape_four=not force_five, common_head_dim=common_head_dim)
        else:
            raise ValueError(f'keys should have 4 or 5 dimensions not {kdim}')
    else:
        raise ValueError(f'queries should have 4 or 5 dimensions not {qdim}')


def weight_values(scores: Array, values: Array, attn_dtype: DType = None) -> Array:
    # Use softmax to get the attention weights
    weights = ten.softmax(scores, axis=-1)

    v = values if attn_dtype is None else ten.as_type(values, attn_dtype)

    # Apply the attention weights to the values
    out = ten.as_type(ten.matmul(weights, v), values.dtype)

    return out


def fix_mask(mask: Optional[Array], dtype: DType = None) -> Optional[Array]:
    if mask is None:
        return None
    if mask.dtype == ten.bool_:
        mask = mask * -1e9
    if dtype is not None:
        mask = ten.as_type(mask, dtype)
    return mask



sdpa_attention_scorer: AttentionScorer



def sdpa_attention_scorer(queries: Array, keys_t: Array, qs: Optional[slice], kvs: Optional[slice], /,
                          offset: int = 0, **extra) -> Array:
    """
    Computes scaled dot-product attention scores by performing matrix multiplication
    between scaled queries and transposed keys within the specified slices.

    :param offset:
    :param queries: Input tensor representing the queries. (B, *H, Q, D)
    :param keys_t: Transposed keys tensor.                 (B, *H, D, K)
    :param qs: Slice defining the range for the queries.
    :param kvs: Slice defining the range for the keys and values.
    :param extra: Additional keyword arguments (optional; this can act as a placeholder
                  for any implementation-specific parameters).
    :return: A tensor containing the scaled dot-product attention scores. (B, *H, Q, K)

    """
    return ten.matmul(queries, keys_t)


def shift_slice(s: slice, offset: int = 0) -> slice:
    return s if offset == 0 else slice(s.start + offset, s.stop + offset, s.step)


__all__ = [
    'fast_sdp_attention',
    'reshape_qkv',
    'shift_slice',
    'sdpa_attention_scorer',
]