"""
Visualization utilities for non-convex optimization benchmarking.

Provides:
  - 2-D landscape heatmaps with trajectory overlays
  - Convergence curves (function value and gradient norm vs. iteration)
  - Comparison bar charts across optimizers
  - Hessian eigenvalue distribution histograms
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .functions import BenchmarkFunction
from .optimizers import OptimizeResult
from .benchmark import BenchmarkResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_CMAP = "viridis"
_DEFAULT_FIGSIZE = (8, 6)


def _ensure_ax(ax: Optional[Axes], figsize: Tuple[float, float] = _DEFAULT_FIGSIZE) -> Tuple[Figure, Axes]:
    """Return (fig, ax), creating a new figure if *ax* is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


# ---------------------------------------------------------------------------
# 2-D Landscape heatmap
# ---------------------------------------------------------------------------


def plot_landscape_2d(
    func: BenchmarkFunction,
    bounds: Optional[Tuple[float, float]] = None,
    n_grid: int = 200,
    log_scale: bool = False,
    trajectories: Optional[List[np.ndarray]] = None,
    trajectory_labels: Optional[List[str]] = None,
    trajectory_colors: Optional[List[str]] = None,
    mark_global_min: bool = True,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot a 2-D contour heatmap of *func* with optional trajectory overlays.

    Parameters
    ----------
    func:
        A 2-D benchmark function.
    bounds:
        ``(low, high)`` for both axes.  Defaults to *func.default_bounds*.
    n_grid:
        Grid resolution.
    log_scale:
        If True, display :math:`\\log(1 + f(x,y))`.
    trajectories:
        List of arrays of shape *(T, 2)*, one per algorithm.
    trajectory_labels:
        Labels for the legend corresponding to *trajectories*.
    trajectory_colors:
        Colors for each trajectory.
    mark_global_min:
        If True, mark the known global minimizer with a star.
    ax:
        Existing :class:`Axes` to draw on.
    title:
        Figure title.

    Returns
    -------
    (Figure, Axes)
    """
    if func.dim != 2:
        raise ValueError("plot_landscape_2d requires a 2-D function.")

    fig, ax = _ensure_ax(ax)
    lo, hi = bounds if bounds is not None else func.default_bounds

    xs = np.linspace(lo, hi, n_grid)
    ys = np.linspace(lo, hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.vectorize(lambda a, b: func(np.array([a, b])))(X, Y)
    if log_scale:
        Z = np.log1p(Z - Z.min())

    ax.contourf(X, Y, Z, levels=50, cmap=_DEFAULT_CMAP, alpha=0.85)
    ax.contour(X, Y, Z, levels=20, colors="white", linewidths=0.3, alpha=0.4)

    if trajectories is not None:
        colors = trajectory_colors or [
            f"C{i}" for i in range(len(trajectories))
        ]
        labels = trajectory_labels or [f"Trajectory {i}" for i in range(len(trajectories))]
        for traj, color, label in zip(trajectories, colors, labels):
            traj = np.asarray(traj)
            ax.plot(traj[:, 0], traj[:, 1], "-o", color=color, markersize=3,
                    linewidth=1.5, label=label)
            ax.plot(traj[0, 0], traj[0, 1], "s", color=color, markersize=8, zorder=5)
            ax.plot(traj[-1, 0], traj[-1, 1], "*", color=color, markersize=12, zorder=5)
        ax.legend(fontsize=8, loc="upper right")

    if mark_global_min:
        gmin = func.global_min_location()
        ax.plot(gmin[0], gmin[1], "r*", markersize=14, zorder=6,
                label="Global min", markeredgecolor="white", markeredgewidth=0.5)
        if trajectories is None:
            ax.legend(fontsize=8)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title or f"{func.name} landscape")
    return fig, ax


# ---------------------------------------------------------------------------
# Convergence curves
# ---------------------------------------------------------------------------


def plot_convergence(
    results: Union[OptimizeResult, List[OptimizeResult]],
    labels: Optional[List[str]] = None,
    log_y: bool = True,
    metric: str = "values",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot convergence curves (function value or gradient norm vs. iteration).

    Parameters
    ----------
    results:
        One or more :class:`~benchmarking.optimizers.OptimizeResult` objects.
    labels:
        Legend labels, one per result.
    log_y:
        If True, use a log scale on the y-axis.
    metric:
        ``"values"`` for function values, ``"grad_norms"`` for gradient norms.
    ax:
        Existing :class:`Axes` to draw on.
    title:
        Figure title.
    """
    if isinstance(results, OptimizeResult):
        results = [results]

    fig, ax = _ensure_ax(ax)
    labels = labels or [f"Run {i}" for i in range(len(results))]

    for res, label in zip(results, labels):
        data = getattr(res, metric)
        ax.plot(range(len(data)), data, label=label, linewidth=1.5)

    if log_y:
        ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("Iteration")
    ylabel = "Function value" if metric == "values" else "Gradient norm"
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Convergence ({ylabel})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    return fig, ax


# ---------------------------------------------------------------------------
# Optimizer comparison bar chart
# ---------------------------------------------------------------------------


def plot_optimizer_comparison(
    benchmark_result: BenchmarkResult,
    function_name: str,
    metric: str = "final_value",
    log_y: bool = False,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Bar chart comparing optimizers on a single function.

    Parameters
    ----------
    benchmark_result:
        Aggregated results from :class:`~benchmarking.benchmark.BenchmarkRunner`.
    function_name:
        Which function to show.
    metric:
        Column from :meth:`~benchmarking.benchmark.BenchmarkResult.as_dataframe`
        to aggregate (e.g. ``"final_value"``, ``"n_iterations"``).
    log_y:
        If True, use a log scale on the y-axis.
    ax:
        Existing :class:`Axes` to draw on.
    title:
        Figure title.
    """
    fig, ax = _ensure_ax(ax)
    df = benchmark_result.as_dataframe()
    sub = df[df["function"] == function_name]
    if sub.empty:
        raise ValueError(f"No data found for function '{function_name}'.")

    agg = sub.groupby("optimizer")[metric].agg(["mean", "std"]).reset_index()
    opts = agg["optimizer"].tolist()
    means = agg["mean"].to_numpy()
    stds = agg["std"].to_numpy()

    x = np.arange(len(opts))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[f"C{i}" for i in range(len(opts))],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(opts, rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title or f"{function_name}: {metric} by optimizer")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return fig, ax


# ---------------------------------------------------------------------------
# Hessian eigenvalue histogram
# ---------------------------------------------------------------------------


def plot_hessian_spectrum(
    eigenvalues: np.ndarray,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    bins: int = 60,
) -> Tuple[Figure, Axes]:
    """Histogram of Hessian eigenvalues.

    Parameters
    ----------
    eigenvalues:
        Flat array of eigenvalues collected via
        :meth:`~benchmarking.geometry.LandscapeGeometry.hessian_spectrum`.
    ax:
        Existing :class:`Axes` to draw on.
    title:
        Figure title.
    bins:
        Number of histogram bins.
    """
    fig, ax = _ensure_ax(ax)
    ax.hist(eigenvalues, bins=bins, color="steelblue", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="λ = 0")
    ax.set_xlabel("Eigenvalue (λ)")
    ax.set_ylabel("Count")
    ax.set_title(title or "Hessian eigenvalue distribution")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax


# ---------------------------------------------------------------------------
# Multi-panel benchmark summary
# ---------------------------------------------------------------------------


def plot_benchmark_summary(
    benchmark_result: BenchmarkResult,
    figsize: Tuple[float, float] = (14, 5),
) -> Figure:
    """Convenience multi-panel figure: convergence + optimizer comparison.

    Creates one row of panels, each showing the mean final value per
    optimizer for a different function.

    Parameters
    ----------
    benchmark_result:
        Aggregated benchmark results.
    figsize:
        Total figure size.

    Returns
    -------
    Figure
    """
    df = benchmark_result.as_dataframe()
    funcs = sorted(df["function"].unique())
    n = len(funcs)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]

    for ax, fname in zip(axes, funcs):
        try:
            plot_optimizer_comparison(benchmark_result, fname, ax=ax)
        except Exception:
            pass

    fig.tight_layout()
    return fig
