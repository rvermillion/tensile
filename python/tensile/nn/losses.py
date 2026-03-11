#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
import math
from typing import Literal

from ..infrastructure.util import name_function
from .common import *


# if ten.ten_kind == 'mlx':
#     from mlx.nn.losses import cross_entropy
# elif ten.ten_kind == 'torch':
#     from torch.nn.functional import cross_entropy as torch_cross_entropy
#     def cross_entropy(logits: Array, labels: Array, reduction: str = 'mean') -> Array:
#         if logits.ndim > 2:
#             logits = logits.reshape(-1, logits.shape[-1])
#             labels = labels.reshape(-1)
#             loss = torch_cross_entropy(logits, labels, reduction=reduction)
#             return loss
#         return torch_cross_entropy(logits, labels, reduction=reduction)
# else:
#     def cross_entropy(logits: Array, labels: Array, reduction: str = 'mean') -> Array:
#         raise RuntimeError(f'cross_entropy is not supported for {ten.ten_kind}')


class LossFunction(Protocol):

    def __call__(self, predictions: Array, targets: Array) -> Array: ...


class ExtraLossFunction(Protocol):

    def __call__(self, predictions: Array, targets: Array, *extra: Array) -> Array: ...




Reduction = Literal["none", "mean", "sum"]


reductions: set[str] = {'none', 'mean', 'sum'}


loss_functions: dict[str, LossFunction] = {}


def loss_function_name(name: str, defaults: dict[str, Any] = None, /, **options) -> str:
    if defaults:
        options = {opt: val for opt, val in options.items() if val != defaults.get(opt)}
    if options: return name + '[' + ", ".join(f'{k}={v}' for k, v in options.items()) + ']'
    return name


def cache_and_name_loss_function(loss_fn: LossFunction, name: str) -> LossFunction:
    loss_fn = name_function(loss_fn, name)
    loss_functions[name] = loss_fn
    return loss_fn


@provides(LossFunction, 'cross_entropy', spread=True)
def provide_cross_entropy_loss(reduction: Reduction = 'mean', axis: int = -1,
                               label_smoothing: float = 0.0, skip_pad: int = None) -> LossFunction:

    if reduction not in reductions:
        raise ValueError(f"Invalid reduction: {reduction}")
    if label_smoothing < 0 or label_smoothing >= 1:
        raise ValueError(f"Label smoothing must in [0, 1), got {label_smoothing}.")

    loss_name =  loss_function_name('cross_entropy',
                                    {'axis': -1, 'label_smoothing': 0.0, 'skip_pad': None},
                                    reduction=reduction, skip_pad=skip_pad,
                                    axis=axis, label_smoothing=label_smoothing)


    if loss_fn := loss_functions.get(loss_name):
        return loss_fn

    if skip_pad is not None:
        if reduction == 'mean':
            def loss_fn(logits: Array, labels: Array) -> Array:
                losses = cross_entropy(logits, labels, axis=axis, reduction='none')

                mask = ten.as_type(labels != skip_pad, losses.dtype)

                n = ten.maximum(ten.sum(mask), ten.array(1, dtype=losses.dtype))

                loss = ten.sum(losses * mask) / n

                return loss
        elif reduction == 'sum':
            def loss_fn(logits: Array, labels: Array) -> Array:
                losses = cross_entropy(logits, labels, axis=axis, reduction='none')

                mask = ten.as_type(labels != skip_pad, losses.dtype)

                return ten.sum(losses * mask)
        elif reduction == 'none':
            def loss_fn(logits: Array, labels: Array) -> Array:
                losses = cross_entropy(logits, labels, axis=axis, reduction='none')
                mask = ten.as_type(labels != skip_pad, losses.dtype)

                return losses * mask
    else:
        def loss_fn(logits: Array, targets: Array, weights: Optional[Array] = None) -> Array:
            """
            Computes the cross entropy loss.

            Args:
                logits (array): The unnormalized logits.
                targets (array): The ground truth values. These can be class indices or
                    probabilities for each class. If the ``targets`` are class indices,
                    then ``targets`` shape should match the ``logits`` shape with
                    the ``axis`` dimension removed. If the ``targets`` are probabilities
                    (or one-hot encoded), then the ``targets`` shape should be the same as
                    the ``logits`` shape.
                weights (array, optional): Optional weights for each target. Default: ``None``.
                axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
                label_smoothing (float, optional): Label smoothing factor. Default: ``0``.
                reduction (str, optional): Specifies the reduction to apply to the output:
                    ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

            Returns:
                array: The computed cross entropy loss.

            Examples:
                >>> import mlx.core as mx
                >>> import mlx.nn as nn
                >>>
                >>> # Class indices as targets
                >>> logits = Array([[2.0, -1.0], [-1.0, 2.0]])
                >>> targets = Array([0, 1])
                >>> nn.losses.cross_entropy(logits, targets)
                array([0.0485873, 0.0485873], dtype=float32)
                >>>
                >>> # Probabilities (or one-hot vectors) as targets
                >>> logits = Array([[2.0, -1.0], [-1.0, 2.0]])
                >>> targets = Array([[0.9, 0.1], [0.1, 0.9]])
                >>> nn.losses.cross_entropy(logits, targets)
                array([0.348587, 0.348587], dtype=float32)
            """

            # Whether targets are class indices or probabilities
            targets_as_probs = targets.ndim == logits.ndim

            def _drop_dim(shape, ax):
                shape = list(shape)
                shape.pop(ax)
                return tuple(shape)

            # Check shapes in two cases: targets as class indices and targets as probabilities
            if (targets_as_probs and targets.shape != logits.shape) or (
                not targets_as_probs and targets.shape != _drop_dim(logits.shape, axis)
            ):
                raise ValueError(
                    f"Targets shape {targets.shape} does not match logits shape {logits.shape}."
                )

            if targets_as_probs:
                score = ten.sum(logits * targets, axis=axis)
            else:
                score = ten.take_along_axis(logits, ten.expand_dims(targets, axis), axis).squeeze(axis)

            logsumexp_logits = ten.logsumexp(logits, axis=axis)
            if label_smoothing > 0:
                # Adjust the true class score with label smoothing
                adjusted_score = (1 - label_smoothing) * score

                # Calculate the mean logit across the classes for smoothed loss
                mean_logits = logits.mean(axis=axis)
                smoothed_loss = -mean_logits * label_smoothing

                # Combine the adjusted score and smoothed loss with the logsumexp logits
                loss = logsumexp_logits - adjusted_score + smoothed_loss
            else:
                loss = logsumexp_logits - score

            # Apply weights if provided
            if weights is not None:
                if weights.shape != loss.shape:
                    raise ValueError(
                        f"Weights with shape {weights.shape} is not the same as "
                        f"output loss with shape {loss.shape}."
                    )
                loss *= weights

            return loss

        loss_fn = _make_reduction(loss_fn, reduction)

    return cache_and_name_loss_function(loss_fn, loss_name)


@provides(LossFunction, 'l1', spread=True)
def provide_l1_loss(reduction: Reduction = 'mean') -> LossFunction:
    loss_name = loss_function_name('l1', reduction=reduction)
    if loss_fn := loss_functions.get(loss_name):
        return loss_fn
    loss_fn = _make_reduction(l1_loss, reduction)
    return cache_and_name_loss_function(loss_fn, loss_name)


@provides(LossFunction, 'mse', spread=True)
def provide_mse_loss(reduction: Reduction = 'mean') -> LossFunction:
    loss_name = loss_function_name('mse', reduction=reduction)
    if loss_fn := loss_functions.get(loss_name):
        return loss_fn
    loss_fn = _make_reduction(mse_loss, reduction)
    return cache_and_name_loss_function(loss_fn, loss_name)


@provides(LossFunction, 'nll', spread=True)
def provide_nll_loss(reduction: Reduction = 'none', axis: int = -1) -> LossFunction:
    loss_name = loss_function_name('nll', {'axis': -1}, reduction=reduction, axis=axis)
    if loss_fn := loss_functions.get(loss_name):
        return loss_fn
    def loss_fn(inputs: Array, targets: Array) -> Array:
        """
        Computes the negative log likelihood loss.

        Args:
            inputs (array): The predicted distribution in log space.
            targets (array): The target values.
            axis (int, optional): The distribution axis. Default: ``-1``.
            reduction (str, optional): Specifies the reduction to apply to the output:
              ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

        Returns:
            array: The computed NLL loss.
        """
        return -ten.take_along_axis(inputs, targets[..., None], axis).squeeze(-1)
    loss_fn = _make_reduction(l1_loss, reduction)
    return cache_and_name_loss_function(loss_fn, loss_name)


class LossAndGradFunction(Protocol):

    def __call__(self, x: Array, y: Array) -> tuple[Array, Any]: ...


def _reduce(loss: Array, reduction: Reduction) -> Array:
    if reduction == 'mean':
        return ten.mean(loss)
    if reduction == 'sum':
        return ten.sum(loss)
    if reduction == 'none':
        return loss
    raise ValueError(f"Invalid reduction: {reduction}")


def _make_reduction(loss: LossFunction, reduction: Reduction = 'mean') -> LossFunction:
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        def reduced_loss(x, y):
            return ten.mean(loss(x, y))
    elif reduction == 'sum':
        def reduced_loss(x, y):
            return ten.sum(loss(x, y))
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return reduced_loss


def _validate_shape(predictions: Array, targets: Array) -> None:
    if predictions.shape != targets.shape:
        raise ValueError(f"Predictions shape {predictions.shape} does not match targets shape {targets.shape}")


def cross_entropy(
    logits: Array,
    targets: Array,
    weights: Optional[Array] = None,
    axis: int = -1,
    label_smoothing: float = 0.0,
    reduction: Reduction = "none",
) -> Array:
    """
    Computes the cross entropy loss.

    Args:
        logits (array): The unnormalized logits.
        targets (array): The ground truth values. These can be class indices or
            probabilities for each class. If the ``targets`` are class indices,
            then ``targets`` shape should match the ``logits`` shape with
            the ``axis`` dimension removed. If the ``targets`` are probabilities
            (or one-hot encoded), then the ``targets`` shape should be the same as
            the ``logits`` shape.
        weights (array, optional): Optional weights for each target. Default: ``None``.
        axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
        label_smoothing (float, optional): Label smoothing factor. Default: ``0``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed cross entropy loss.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>>
        >>> # Class indices as targets
        >>> logits = Array([[2.0, -1.0], [-1.0, 2.0]])
        >>> targets = Array([0, 1])
        >>> nn.losses.cross_entropy(logits, targets)
        array([0.0485873, 0.0485873], dtype=float32)
        >>>
        >>> # Probabilities (or one-hot vectors) as targets
        >>> logits = Array([[2.0, -1.0], [-1.0, 2.0]])
        >>> targets = Array([[0.9, 0.1], [0.1, 0.9]])
        >>> nn.losses.cross_entropy(logits, targets)
        array([0.348587, 0.348587], dtype=float32)
    """
    if label_smoothing < 0 or label_smoothing >= 1:
        raise ValueError(f"Label smoothing must in [0, 1), got {label_smoothing}.")

    # Whether targets are class indices or probabilities
    targets_as_probs = targets.ndim == logits.ndim

    def _drop_dim(shape, axis):
        shape = list(shape)
        shape.pop(axis)
        return tuple(shape)

    # Check shapes in two cases: targets as class indices and targets as probabilities
    if (targets_as_probs and targets.shape != logits.shape) or (
        not targets_as_probs and targets.shape != _drop_dim(logits.shape, axis)
    ):
        raise ValueError(
            f"Targets shape {targets.shape} does not match logits shape {logits.shape}."
        )

    if targets_as_probs:
        score = ten.sum(logits * targets, axis=axis)
    else:
        score = ten.take_along_axis(logits, ten.expand_dims(targets, axis), axis).squeeze(
            axis
        )

    logsumexp_logits = ten.logsumexp(logits, axis=axis)
    if label_smoothing > 0:
        # Adjust the true class score with label smoothing
        adjusted_score = (1 - label_smoothing) * score

        # Calculate the mean logit across the classes for smoothed loss
        mean_logits = logits.mean(axis=axis)
        smoothed_loss = -mean_logits * label_smoothing

        # Combine the adjusted score and smoothed loss with the logsumexp logits
        loss = logsumexp_logits - adjusted_score + smoothed_loss
    else:
        loss = logsumexp_logits - score

    # Apply weights if provided
    if weights is not None:
        if weights.shape != loss.shape:
            raise ValueError(
                f"Weights with shape {weights.shape} is not the same as "
                f"output loss with shape {loss.shape}."
            )
        loss *= weights

    return _reduce(loss, reduction)


def l1_loss(predictions: Array, targets: Array) -> Array:
    """
    Computes the L1 loss.

    Args:
        predictions (array): The predicted values.
        targets (array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Returns:
        array: The computed L1 loss.
    """
    _validate_shape(predictions, targets)
    return ten.abs(predictions - targets)


def mse_loss(predictions: Array, targets: Array) -> Array:
    """
    Computes the mean squared error loss.

    Args:
        predictions (array): The predicted values.
        targets (array): The target values.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

    Returns:
        array: The computed mean squared error loss.
    """
    _validate_shape(predictions, targets)
    return ten.square(predictions - targets)


def nll_loss(predictions: Array, targets: Array, axis: int = -1, reduction: Reduction = "none") -> Array:
    """
    Computes the negative log likelihood loss.

    Args:
        predictions (array): The predicted distribution in log space.
        targets (array): The target values.
        axis (int, optional): The distribution axis. Default: ``-1``.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

    Returns:
        array: The computed NLL loss.
    """
    loss = -ten.take_along_axis(predictions, targets[..., None], axis).squeeze(-1)

    return _reduce(loss, reduction)


@provides(LossFunction, 'gaussian_nll', spread=True)
def provide_gaussian_nll_loss(full: bool = False, eps: float = 1e-6, reduction: Reduction = "mean") -> LossFunction:
    loss_name = loss_function_name('gaussian_nll', {'full': False, 'eps': 1e-6}, reduction=reduction, full=full, eps=eps)
    if loss_fn := loss_functions.get(loss_name):
        return loss_fn
    if full:
        # noinspection PyShadowingBuiltins
        def loss_fn(predictions: Array, targets: Array, vars: Array = None) -> Array:
            r"""
            Computes the negative log likelihood loss for a Gaussian distribution.

            The loss is given by:

            .. math::
                \frac{1}{2}\left(\log\left(\max\left(\text{vars},
                \ \epsilon\right)\right) + \frac{\left(\text{inputs} - \text{targets} \right)^2}
                {\max\left(\text{vars}, \ \epsilon \right)}\right) + \text{const.}

            where ``inputs`` are the predicted means and ``vars`` are the the
            predicted variances.

            Args:
                predictions (array): The predicted expectation of the Gaussian distribution.
                targets (array): The target values (samples from the Gaussian distribution).
                vars (array): The predicted variance of the Gaussian distribution.
                full (bool, optional): Whether to include the constant term in the loss calculation.
                    Default: ``False``.
                eps (float, optional): Small positive constant for numerical stability.
                    Default: ``1e-6``.
                reduction (str, optional): Specifies the reduction to apply to the output:
                  ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

            Returns:
                array: The Gaussian NLL loss.
            """
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Inputs shape {predictions.shape} does not match targets shape {targets.shape}."
                )

            if predictions.shape != vars.shape:
                raise ValueError(
                    f"Inputs shape {predictions.shape} does not match vars shape {vars.shape}."
                )

            # For stability
            vars = ten.maximum(vars, eps)
            loss = 0.5 * (ten.log(vars) + ten.square(targets - predictions) / vars)

            loss += 0.5 * math.log(2 * math.pi)

            return loss
    else:
        # noinspection PyShadowingBuiltins
        def loss_fn(predictions: Array, targets: Array, vars: Array = None) -> Array:
            r"""
            Computes the negative log likelihood loss for a Gaussian distribution.

            The loss is given by:

            .. math::
                \frac{1}{2}\left(\log\left(\max\left(\text{vars},
                \ \epsilon\right)\right) + \frac{\left(\text{inputs} - \text{targets} \right)^2}
                {\max\left(\text{vars}, \ \epsilon \right)}\right) + \text{const.}

            where ``inputs`` are the predicted means and ``vars`` are the the
            predicted variances.

            Args:
                predictions (array): The predicted expectation of the Gaussian distribution.
                targets (array): The target values (samples from the Gaussian distribution).
                vars (array): The predicted variance of the Gaussian distribution.
                full (bool, optional): Whether to include the constant term in the loss calculation.
                    Default: ``False``.
                eps (float, optional): Small positive constant for numerical stability.
                    Default: ``1e-6``.
                reduction (str, optional): Specifies the reduction to apply to the output:
                  ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

            Returns:
                array: The Gaussian NLL loss.
            """
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Inputs shape {predictions.shape} does not match targets shape {targets.shape}."
                )

            if predictions.shape != vars.shape:
                raise ValueError(
                    f"Inputs shape {predictions.shape} does not match vars shape {vars.shape}."
                )

            # For stability
            vars = ten.maximum(vars, eps)
            loss = 0.5 * (ten.log(vars) + ten.square(targets - predictions) / vars)

            return loss

    loss_fn = _make_reduction(loss_fn, reduction)
    return cache_and_name_loss_function(loss_fn, loss_name)




use_softmax = False

if use_softmax:
    def logits_to_probs(logits: Array, axis: int = -1) -> Array:
        return ten.softmax(logits, axis=axis, keepdims=True)

    def logits_to_log_probs(logits: Array, axis: int = -1) -> Array:
        return ten.log(logits_to_probs(logits, axis=axis))

else:
    def logits_to_log_probs(logits: Array, axis: int = -1) -> Array:
        logits_max = ten.max(logits, axis=axis, keepdims=True)
        logits_norm = logits - logits_max   # numerical stability
        log_probs = logits_norm - ten.logsumexp(logits_norm, axis=-1, keepdims=True)
        return log_probs


    def logits_to_probs(logits: Array, axis: int = -1) -> Array:
        return ten.exp(logits_to_log_probs(logits))

