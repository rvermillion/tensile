#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .. import ten, Array


default_colors = plt.cm.tab10.colors  # or plt.cm.Dark2.colors for muted tones


def plot_metrics(
    series: dict[str, Array],
    title: str = "",
    xlabel: str = "Step",
    ylabel: str = "Value",
    out: str | Path | None = None,
    figsize: tuple = (10, 6),
    log_y: bool = False,
    smoothing: float = 0.0,
    xlim: dict[str, float] = None,
    ylim: dict[str, float] = None,
    colors = default_colors,
):
    """Plot one or more named metric series.

    Args:
        series: {"loss": [...], "probe_norm": [...], ...}
        out: filepath to save (e.g. "plots/loss.png"). Shows interactively if None.
        log_y: log scale on y-axis
        smoothing: EMA smoothing factor in [0, 1). 0 = raw, 0.9 = heavy smooth.
            Raw data shown as faint line, smoothed as solid.
    """
    fig, ax = plt.subplots(figsize=figsize)

    c = 0
    xmax = 0
    for name, values in series.items():
        values = ten.to_numpy(values)
        cnt = len(values)
        if xmax < cnt: xmax = cnt
        x = np.arange(cnt)

        if smoothing > 0:
            ax.plot(x, values, alpha=0.25, linewidth=0.8, color=colors[c])
            smoothed = _ema(values, smoothing)
            ax.plot(x, smoothed, label=name, linewidth=2, color=colors[c])
        else:
            ax.plot(x, values, label=name, linewidth=1.5, color=colors[c])
        c = (c + 1) % len(colors)

    if log_y:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(**xlim)
    else:
        ax.set_xlim(left=0, right=xmax)
    if ylim:
        ax.set_ylim(**ylim)
    plt.tight_layout()

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_grid(
    panels: dict[str, dict[str, Array]],
    out: str | Path | None = None,
    smoothing: float = 0.0,
    log_y: bool = False,
    xlim: dict[str, float] = None,
    ylim: dict[str, float] = None,
    colors = default_colors,
):
    """Multiple subplots. Keys become subplot titles.

    Usage:
        plot_grid({
            "Loss": {"sgd": loss_sgd, "pbdca": loss_pbdca},
            "Probe Norm": {"layer_0": norms_0, "layer_1": norms_1},
        }, out="plots/dashboard.png")
    """
    n = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, sharey=True)

    for idx, (panel_title, series) in enumerate(panels.items()):
        ax = axes[idx // cols][idx % cols]
        xmax = -float('inf')
        c = 0
        for name, values in series.items():
            values = ten.to_numpy(values)
            cnt = len(values)
            if xmax < cnt: xmax = cnt
            x = np.arange(cnt)
            if smoothing > 0:
                ax.plot(x, values, alpha=0.2, linewidth=0.8, color=colors[c])
                ax.plot(x, _ema(values, smoothing), label=name, linewidth=2, color=colors[c])
            else:
                ax.plot(x, values, label=name, linewidth=1.5, color=colors[c])
            c = (c + 1) % len(colors)

        ax.set_title(panel_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        if xlim:
            ax.set_xlim(**xlim)
        else:
            ax.set_xlim(left=0, right=xmax)
        if ylim:
            ax.set_ylim(**ylim)

    # hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


__all__ = [
    'plot_metrics',
    'plot_grid',
]