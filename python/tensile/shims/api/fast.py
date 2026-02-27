#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from typing import Optional

from .types import *


def rms_norm(x: Array, weight: Optional[Array] = None, eps: float = ...,  **kwargs) -> Array:
    """
    rms_norm(x: array, weight: Optional[array], eps: float, *, stream: Union[None, Stream, Device] = None) -> array

            Root Mean Square normalization (RMS norm).

            The normalization is with respect to the last axis of the input ``x``.

            Args:
                x (array): Input array.
                weight (array, optional): A multiplicative weight to scale the result by.
                  The ``weight`` should be one-dimensional with the same size
                  as the last axis of ``x``. If set to ``None`` then no scaling happens.
                eps (float): A small additive constant for numerical stability.

            Returns:
                array: The output array.
    """
    raise NotImplementedError()


def rope(a: Array, dims: int,  *args, **kwargs) -> Array:
    """
    rope(a: array, dims: int, *, traditional: bool, base: Optional[float], scale: float, offset: Union[int, array], freqs: Optional[array] = None, stream: Union[None, Stream, Device] = None) -> array

            Apply rotary positional encoding to the input.

            The input is expected to be at least 3D with shape ``(B, *, T, D)`` where:
              * ``B`` is the batch size.
              * ``T`` is the sequence length.
              * ``D`` is the feature dimension.

            Args:
                a (array): The input array.
                dims (int): The feature dimensions to be rotated. If the input feature
                  is larger than dims then the rest is left unchanged.
                traditional (bool): If set to ``True`` choose the traditional
                  implementation which rotates consecutive dimensions.
                base (float, optional): The base used to compute angular frequency for
                  each dimension in the positional encodings. Exactly one of ``base`` and
                  ``freqs`` must be ``None``.
                scale (float): The scale used to scale the positions.
                offset (int or array): The position offset to start at. If an
                  :obj:`array` is given it can be a scalar or vector of ``B``
                  offsets for each example in the batch.
                freqs (array, optional): Optional frequencies to use with RoPE.
                  If set, the ``base`` parameter must be ``None``. Default: ``None``.

            Returns:
                array: The output array.
    """
    raise NotImplementedError()
