#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.
from collections import defaultdict

import math

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ..common import *
from .metric import Metric, ArrayMetric, StepMetric


default_colors = plt.cm.tab10.colors  # or plt.cm.Dark2.colors for muted tones


class Series(Object):

    __slots__ = ('label', 'color', 'smoothing')

    label: Annotated[str, field()]
    color: Annotated[str | None, field(default=None)]
    smoothing: Annotated[float, field()]

    def _lazy_label(self) -> str:
        return 'Series'

    def plot(self, ax: plt.Axes,
             label: str = None,
             smoothing: float = 0.0,
             color: str | None = None,
    ) -> tuple[int, float]:
        raise NotImplementedError()

    @classmethod
    def from_metric(cls, metric: Metric, **kwargs) -> 'Series':
        if isinstance(metric, ArrayMetric):
            return ArrayMetricSeries(metric=metric, **kwargs)
        if isinstance(metric, StepMetric):
            return StepMetricSeries(metric=metric, **kwargs)
        raise ValueError(f"Unsupported metric type: {type(metric).__name__}")

    @classmethod
    def from_metrics(cls, metrics: dict[str, Metric], **kwargs) -> list['Series']:
        return [Series.from_metric(metric, name=name, **kwargs) for name, metric in metrics.items()]

    @classmethod
    def split_panels(cls, metrics: dict[str, Metric], sep: str = '/', last: bool = True, **kwargs) -> dict[str, dict[str, 'Series']]:
        split = {}
        for name, metric in metrics.items():
            i = name.rfind(sep) if last else name.find(sep)
            if i < 0:
                exp = m = name
            else:
                exp = name[:i]
                m = name[i+1:]
            series = Series.from_metric(metric, name=name, **kwargs)
            if exp in split:
                split[exp][m] = series
            else:
                split[exp] = {m: series}
        return split


class ArrayMetricSeries(Series):

    __slots__ = ('metric', )

    metric: Annotated[ArrayMetric, field()]

    def _lazy_label(self) -> str:
        return self.metric.name

    def plot(self, ax: plt.Axes,
             label: str = None,
             smoothing: float = 0.0,
             color: str | None = None,
    ) -> tuple[int, float]:
        if label is None: label = self.label
        if color is None: color = self.color

        values = ten.to_numpy(self.metric.array)
        cnt = len(values)
        x = np.arange(cnt)

        max_y = np.max(values)
        if smoothing > 0:
            smoothed = _ema(values, smoothing)
            max_smoothed = np.max(smoothed)
            values = np.minimum(values, max_smoothed*3.)
            ax.plot(x, values, alpha=0.25, linewidth=0.8, color=color)
            ax.plot(x, smoothed, label=label, linewidth=2, color=color)
        else:
            ax.plot(x, values, label=label, linewidth=1.5, color=color)

        return cnt, max_y.item()

    def _repr_args(self, **options) -> str:
        return self.metric.name


class StepMetricSeries(Series):

    __slots__ = ('metric', )

    metric: Annotated[StepMetric, field()]

    def _lazy_label(self) -> str:
        return self.metric.name

    def plot(self, ax: plt.Axes,
             label: str = None,
             smoothing: float = 0.0,
             color: str | None = None,
    ) -> tuple[int, float]:
        if label is None: label = self.label
        if color is None: color = self.color

        metric = self.metric

        max_y = np.max(metric.values)
        cnt = metric.steps[-1]
        ax.plot(metric.steps, metric.values, drawstyle='steps-post', color=color, label=label, linewidth=1.5)

        return cnt, max_y.item()

    def _repr_args(self, **options) -> str:
        return self.metric.name


class Chart(Object):

    __slots__ = ('title', 'xlabel', 'ylabel', 'figsize', 'smoothing', 'colors',
                 'xlim', 'ylim')

    title: Annotated[str, field(default="")]
    xlabel: Annotated[str, field(default="Step")]
    ylabel: Annotated[str, field(default="Value")]
    figsize: Annotated[tuple[int, int], field(default=(10, 6))]
    smoothing: Annotated[float, field(default=0.0)]
    colors: Annotated[list[str], field(default=default_colors)]
    xlim: Annotated[Optional[dict[str, Any]], field()]
    ylim: Annotated[Optional[dict[str, Any]], field()]

    def plot(self,
             out: str | Path | None = None,
             *,
             title: str = None,
             xlabel: str = None,
             ylabel: str = None,
             figsize: tuple[int, int] = None,
             log_y: bool = False,
             smoothing: float = None,
             xlim: dict[str, float] = None,
             ylim: dict[str, float] = None,
             colors = default_colors,
             **kwargs):
        if title is None: title = self.title
        if xlabel is None: xlabel = self.xlabel
        if ylabel is None: ylabel = self.ylabel
        if figsize is None: figsize = self.figsize
        if smoothing is None: smoothing = self.smoothing
        if xlim is None: xlim = self.xlim
        if ylim is None: ylim = self.ylim

        fig = self.draw(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            log_y=log_y,
            smoothing=smoothing,
            colors=colors,
            xlim=xlim,
            ylim=ylim,
            **kwargs
        )

        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


    def draw(self, *,
             title: str,
             xlabel: str,
             ylabel: str,
             figsize: tuple[int, int],
             log_y: bool,
             smoothing: float,
             colors,
             xlim: dict[str, float] = None,
             ylim: dict[str, float] = None,
             **kwargs) -> plt.Figure:
        raise NotImplementedError()

    @classmethod
    def from_metrics(cls, metrics: dict[str, Metric], **kwargs) -> 'Chart':
        series = Series.from_metrics(metrics)
        return SeriesChart(series=series, **kwargs)

    @classmethod
    def split_grid(cls, metrics: dict[str, Metric], **kwargs) -> 'Chart':
        panels = Series.split_panels(metrics, **kwargs)
        return GridChart(panels=panels, **kwargs)

    @classmethod
    def grid_per_metric(cls, metrics: dict[str, Metric], **kwargs) -> list['GridChart']:
        charts = []
        panels = defaultdict(dict[str, Series])
        for name, metric in metrics.items():
            panels[metric.name][name] = Series.from_metric(metric)

        for title, panel in panels.items():
            charts.append(GridChart(panels=panel, title=title, **kwargs))

        return charts


@provides(Chart, 'series')
class SeriesChart(Chart):

    __slots__ = ('series',)

    series: Annotated[list[Series], field(default_factory=list)]

    def _coerce_series(self, spec: Any) -> list[Series]:
        if spec is None: return []
        if isinstance(spec, Mapping):
            return [Series.coerce(s, name=name) for name, s in spec.items()]
        if isinstance(spec, Iterable):
            return [Series.coerce(s) for s in spec]
        raise ValueError(f'Cannot coerce to list[Series]: {spec}')

    def draw(self, *,
             title: str,
             xlabel: str,
             ylabel: str,
             figsize: tuple[int, int],
             log_y: bool,
             smoothing: float,
             colors,
             xlim: dict[str, float] = None,
             ylim: dict[str, float] = None,
             **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)

        c = 0
        xmax = 0
        ymax = 0.
        series_ax = ax
        for series in self.series:

            cnt, ytop = series.plot(series_ax, smoothing=smoothing, color=colors[c])

            if xmax < cnt: xmax = cnt
            if ymax < ytop: ymax = ytop

            # if series_ax is ax:
            #     series_ax = ax.twinx()

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
        # else:
        #     ax.set_ylim(bottom=0)
        plt.tight_layout()

        return fig


@provides(Chart, 'grid')
class GridChart(Chart):

    __slots__ = ('panels', 'cols', 'rows')

    panels: Annotated[dict[str, dict[str, Series]], field(default_factory=dict)]
    cols: Annotated[int, field(default=0)]
    rows: Annotated[int, field(default=0)]

    def draw(self,
        smoothing: float = 0.0,
        log_y: bool = False,
        xlim: dict[str, float] = None,
        ylim: dict[str, float] = None,
        colors = default_colors,
        cols: int = 0,
        rows: int = 0,
         **kwargs,

    ):
        """Multiple subplots. Keys become subplot titles.

        Usage:
            plot_grid({
                "Loss": {"sgd": loss_sgd, "pbdca": loss_pbdca},
                "Probe Norm": {"layer_0": norms_0, "layer_1": norms_1},
            }, out="plots/dashboard.png")
        """

        panels = self.panels
        if cols == 0: cols = self.cols
        if rows == 0: rows = self.rows

        n = len(panels)
        if cols == 0:
            if rows == 0:
                if n < 4:
                    cols = 3
                else:
                    cols = min(int(math.ceil(n**0.5)), 6)
                rows = (n + cols - 1) // cols
            else:
                cols = (n + rows - 1) // rows
        elif rows == 0:
            rows = (n + cols - 1) // cols
        elif rows * cols < n:
            raise ValueError(f'Rows ({rows}) and columns ({cols}) specified for {n} panels, ')


        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, sharey=True)

        c = 0
        ncolors = len(colors)
        for idx, (panel_title, metrics) in enumerate(panels.items()):
            ax = axes[idx // cols][idx % cols]
            xmax = -float('inf')
            for name, series in metrics.items():
                if series is not None:
                    cnt, ymax = series.plot(ax, label=name, smoothing=smoothing, color=colors[c])
                    if xmax < cnt: xmax = cnt
            c = (c + 1) % ncolors

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
            # else:
            #     ax.set_ylim(bottom=0)

        # hide unused subplots
        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)

        plt.tight_layout()

        return fig

def plot_metric(
    ax,
    metric: Metric,
    label: str = None,
    smoothing: float = 0.0,
    color: str | None = None,
) -> int:
    if label is None: label = metric.name

    if isinstance(metric, ArrayMetric):
        values = ten.to_numpy(metric.array)
        cnt = len(values)
        x = np.arange(cnt)

        if smoothing > 0:
            ax.plot(x, values, alpha=0.25, linewidth=0.8, color=color)
            smoothed = _ema(values, smoothing)
            ax.plot(x, smoothed, label=label, linewidth=2, color=color)
        else:
            ax.plot(x, values, label=label, linewidth=1.5, color=color)
    elif isinstance(metric, StepMetric):
        cnt = len(metric.steps)
        ax.plot(metric.steps, metric.values, drawstyle='steps-post', color=color, label=label, linewidth=1.5)
    else:
        cnt = 0

    return cnt


def plot_metrics(
    metrics: dict[str, Metric],
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
    series = {name: Series.from_metric(metric) for name, metric in metrics.items()}
    chart = SeriesChart(series=series)

    chart.plot(out=out)


def xplot_metrics(
    metrics: dict[str, Metric],
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
        metrics: {"loss": [...], "probe_norm": [...], ...}
        out: filepath to save (e.g. "plots/loss.png"). Shows interactively if None.
        log_y: log scale on y-axis
        smoothing: EMA smoothing factor in [0, 1). 0 = raw, 0.9 = heavy smooth.
            Raw data shown as faint line, smoothed as solid.
    """
    fig, ax = plt.subplots(figsize=figsize)

    c = 0
    xmax = 0
    ymax = 0.
    for name, metric in metrics.items():

        series = Series.from_metric(metric)

        cnt, max_y = series.plot(ax, label=name, smoothing=smoothing, color=colors[c])

        # cnt = plot_metric(ax, metric, label=name, smoothing=smoothing, color=colors[c])
        # values = ten.to_numpy(values)
        # cnt = len(values)
        if xmax < cnt: xmax = cnt
        if ymax < max_y: ymax = max_y
        # x = np.arange(cnt)
        #
        # if smoothing > 0:
        #     ax.plot(x, values, alpha=0.25, linewidth=0.8, color=colors[c])
        #     smoothed = _ema(values, smoothing)
        #     ax.plot(x, smoothed, label=name, linewidth=2, color=colors[c])
        # else:
        #     ax.plot(x, values, label=name, linewidth=1.5, color=colors[c])
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


def plot_metrics_grid(
    panels: dict[str, dict[str, Metric]],
    out: str | Path | None = None,
    smoothing: float = 0.0,
    log_y: bool = False,
    xlim: dict[str, float] = None,
    ylim: dict[str, float] = None,
    colors = default_colors,
    cols: int = 0,
    rows: int = 0
):
    """Multiple subplots. Keys become subplot titles.

    Usage:
        plot_grid({
            "Loss": {"sgd": loss_sgd, "pbdca": loss_pbdca},
            "Probe Norm": {"layer_0": norms_0, "layer_1": norms_1},
        }, out="plots/dashboard.png")
    """
    n = len(panels)
    if cols == 0:
        if rows == 0:
            if n < 4:
                cols = 3
            else:
                cols = min(int(math.ceil(n**0.5)), 6)
            rows = (n + cols - 1) // cols
        else:
            cols = (n + rows - 1) // rows
    elif rows == 0:
        rows = (n + cols - 1) // cols
    elif rows * cols < n:
        raise ValueError(f'Rows ({rows}) and columns ({cols}) specified for {n} panels, ')


    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, sharey=True)

    c = 0
    for idx, (panel_title, metrics) in enumerate(panels.items()):
        ax = axes[idx // cols][idx % cols]
        xmax = -float('inf')
        for name, metric in metrics.items():
            if metric is not None:
                series = Series.from_metric(metric)

                cnt = series.plot(ax, label=name, smoothing=smoothing, color=colors[c])

                # series = Series.from_metric(metric)
                # cnt = plot_metric(ax, metric, label=name, smoothing=smoothing, color=colors[c])
                if xmax < cnt: xmax = cnt
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


def plot_grid(
    panels: dict[str, dict[str, Array]],
    out: str | Path | None = None,
    smoothing: float = 0.0,
    log_y: bool = False,
    xlim: dict[str, float] = None,
    ylim: dict[str, float] = None,
    colors = default_colors,
    cols: int = 0,
    rows: int = 0
):
    """Multiple subplots. Keys become subplot titles.

    Usage:
        plot_grid({
            "Loss": {"sgd": loss_sgd, "pbdca": loss_pbdca},
            "Probe Norm": {"layer_0": norms_0, "layer_1": norms_1},
        }, out="plots/dashboard.png")
    """
    n = len(panels)
    if cols == 0:
        if rows == 0:
            if n < 4:
                cols = 3
            else:
                cols = min(int(math.ceil(n**0.5)), 6)
            rows = (n + cols - 1) // cols
        else:
            cols = (n + rows - 1) // rows
    elif rows == 0:
        rows = (n + cols - 1) // cols
    elif rows * cols < n:
        raise ValueError(f'Rows ({rows}) and columns ({cols}) specified for {n} panels, ')


    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, sharey=True)

    c = 0
    for idx, (panel_title, series) in enumerate(panels.items()):
        ax = axes[idx // cols][idx % cols]
        xmax = -float('inf')
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


def split_panels(s: dict[str, Metric], sep: str = '/', last: bool = True) -> dict[str, dict[str, Metric]]:
    split = {}
    for name, array in s.items():
        i = name.rfind(sep) if last else name.find(sep)
        if i < 0:
            exp = m = name
        else:
            exp = name[:i]
            m = name[i+1:]
        if exp in split:
            split[exp][m] = array
        else:
            split[exp] = {m: array}
    return split


def split_plot_grid(metrics: dict[str, Metric], grid_suffix: str = '', out: Path = None, smoothing: float = None, rows: int = 0):
    grid_panels = split_panels(metrics)
    num_panels = len(grid_panels)
    if num_panels > 1:
        if num_panels == rows and len(grid_panels) % rows == 0:
            grid_panels = {}
            for name, array in metrics.items():
                grid_panels[name] = {name: array}
            plot_metrics_grid(grid_panels, out=out, smoothing=0.9, rows=rows)
        else:
            plot_metrics_grid(grid_panels, out=out, smoothing=0.9, rows=rows)


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