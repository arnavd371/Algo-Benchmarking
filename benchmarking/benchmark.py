"""
Benchmarking framework: run multiple optimizers across multiple functions and
collect performance metrics.

Usage example::

    from benchmarking import BenchmarkRunner, get_function, get_optimizer

    runner = BenchmarkRunner()
    runner.add_function("rastrigin", dim=2)
    runner.add_function("ackley", dim=2)
    runner.add_optimizer("adam", lr=1e-2)
    runner.add_optimizer("lbfgs")
    results = runner.run(n_trials=5, seed=42)
    print(results.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .functions import BenchmarkFunction, get_function
from .optimizers import BaseOptimizer, OptimizeResult, get_optimizer


# ---------------------------------------------------------------------------
# Single-trial result
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """Holds the outcome of a single (function, optimizer, trial) run."""

    function_name: str
    optimizer_name: str
    trial: int
    x0: np.ndarray
    result: OptimizeResult

    @property
    def converged(self) -> bool:
        return self.result.success

    @property
    def final_value(self) -> float:
        return self.result.fun

    @property
    def n_iterations(self) -> int:
        return self.result.nit

    @property
    def n_func_evals(self) -> int:
        return self.result.nfev

    @property
    def elapsed_seconds(self) -> float:
        return self.result.elapsed_seconds


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------


class BenchmarkResult:
    """Aggregated results from a :class:`BenchmarkRunner` run."""

    def __init__(self, trials: List[TrialResult]) -> None:
        self.trials = trials

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def as_dataframe(self) -> pd.DataFrame:
        """Return a tidy :class:`pandas.DataFrame` with one row per trial."""
        rows = []
        for t in self.trials:
            rows.append(
                {
                    "function": t.function_name,
                    "optimizer": t.optimizer_name,
                    "trial": t.trial,
                    "final_value": t.final_value,
                    "n_iterations": t.n_iterations,
                    "n_func_evals": t.n_func_evals,
                    "converged": t.converged,
                    "elapsed_s": t.elapsed_seconds,
                }
            )
        return pd.DataFrame(rows)

    def summary(self, decimals: int = 4) -> str:
        """Return a human-readable per-(function, optimizer) summary."""
        df = self.as_dataframe()
        agg = (
            df.groupby(["function", "optimizer"])
            .agg(
                mean_final_value=("final_value", "mean"),
                std_final_value=("final_value", "std"),
                mean_iterations=("n_iterations", "mean"),
                convergence_rate=("converged", "mean"),
                mean_elapsed_s=("elapsed_s", "mean"),
            )
            .round(decimals)
            .reset_index()
        )
        return agg.to_string(index=False)

    def best_optimizer_per_function(self) -> Dict[str, str]:
        """Return the name of the best optimizer (lowest mean final value) per function."""
        df = self.as_dataframe()
        best = (
            df.groupby(["function", "optimizer"])["final_value"]
            .mean()
            .reset_index()
        )
        idx = best.groupby("function")["final_value"].idxmin()
        return dict(zip(best.loc[idx, "function"], best.loc[idx, "optimizer"]))

    def convergence_trajectories(
        self, function_name: str, optimizer_name: str
    ) -> List[np.ndarray]:
        """Return value trajectories for all trials of a (function, optimizer) pair."""
        return [
            t.result.values
            for t in self.trials
            if t.function_name == function_name
            and t.optimizer_name == optimizer_name
        ]

    def __len__(self) -> int:
        return len(self.trials)

    def __repr__(self) -> str:
        funcs = len({t.function_name for t in self.trials})
        opts = len({t.optimizer_name for t in self.trials})
        return f"BenchmarkResult(functions={funcs}, optimizers={opts}, trials={len(self)})"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs a grid of (function × optimizer) experiments.

    Parameters
    ----------
    default_lr:
        Default learning rate passed to gradient-based optimizers.
    default_max_iter:
        Default iteration budget.
    default_tol:
        Default convergence tolerance.
    """

    def __init__(
        self,
        default_lr: float = 1e-2,
        default_max_iter: int = 5_000,
        default_tol: float = 1e-6,
    ) -> None:
        self.default_lr = default_lr
        self.default_max_iter = default_max_iter
        self.default_tol = default_tol

        self._functions: List[BenchmarkFunction] = []
        self._optimizers: List[BaseOptimizer] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_function(self, name_or_func, dim: int = 2, **kwargs) -> "BenchmarkRunner":
        """Add a function to the benchmark grid.

        Parameters
        ----------
        name_or_func:
            Either a string name (e.g. ``"rastrigin"``) or an instance of
            :class:`~benchmarking.functions.BenchmarkFunction`.
        dim:
            Dimensionality (ignored if an instance is passed).
        """
        if isinstance(name_or_func, str):
            self._functions.append(get_function(name_or_func, dim=dim))
        elif isinstance(name_or_func, BenchmarkFunction):
            self._functions.append(name_or_func)
        else:
            raise TypeError(f"Expected str or BenchmarkFunction, got {type(name_or_func)}.")
        return self

    def add_optimizer(self, name_or_opt, **kwargs) -> "BenchmarkRunner":
        """Add an optimizer to the benchmark grid.

        Parameters
        ----------
        name_or_opt:
            Either a string name (e.g. ``"adam"``) or an instance of
            :class:`~benchmarking.optimizers.BaseOptimizer`.
        **kwargs:
            Keyword arguments forwarded to the optimizer constructor (only
            when *name_or_opt* is a string).  Defaults to
            :attr:`default_lr`, :attr:`default_max_iter`, and
            :attr:`default_tol` if not provided.
        """
        if isinstance(name_or_opt, str):
            kwargs.setdefault("lr", self.default_lr)
            kwargs.setdefault("max_iter", self.default_max_iter)
            kwargs.setdefault("tol", self.default_tol)
            self._optimizers.append(get_optimizer(name_or_opt, **kwargs))
        elif isinstance(name_or_opt, BaseOptimizer):
            self._optimizers.append(name_or_opt)
        else:
            raise TypeError(f"Expected str or BaseOptimizer, got {type(name_or_opt)}.")
        return self

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        n_trials: int = 10,
        bounds: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ) -> BenchmarkResult:
        """Execute the full benchmark grid.

        Parameters
        ----------
        n_trials:
            Number of random starting points per (function, optimizer) pair.
        bounds:
            Override the function's default bounds for starting-point sampling.
        seed:
            Master random seed; each trial gets a deterministic sub-seed.

        Returns
        -------
        BenchmarkResult
        """
        if not self._functions:
            raise ValueError("No functions registered. Call add_function() first.")
        if not self._optimizers:
            raise ValueError("No optimizers registered. Call add_optimizer() first.")

        rng = np.random.default_rng(seed)
        all_trials: List[TrialResult] = []

        for func in self._functions:
            lo, hi = bounds if bounds is not None else func.default_bounds
            for opt in self._optimizers:
                for trial_idx in range(n_trials):
                    x0 = rng.uniform(lo, hi, size=func.dim)
                    res = opt.optimize(func, func.gradient, x0)
                    all_trials.append(
                        TrialResult(
                            function_name=func.name,
                            optimizer_name=opt.name,
                            trial=trial_idx,
                            x0=x0,
                            result=res,
                        )
                    )

        return BenchmarkResult(all_trials)

    # ------------------------------------------------------------------
    # Convenience class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def default_suite(cls, dim: int = 2) -> "BenchmarkRunner":
        """Build a runner pre-loaded with all built-in functions and optimizers.

        Parameters
        ----------
        dim:
            Dimensionality for all scalable functions.
        """
        runner = cls()
        for fname in ["rastrigin", "ackley", "rosenbrock", "levy", "griewank", "sphere"]:
            runner.add_function(fname, dim=dim)
        runner.add_function("himmelblau")
        for oname in ["gradient_descent", "sgd_momentum", "adam", "rmsprop", "lbfgs"]:
            runner.add_optimizer(oname)
        return runner

    def __repr__(self) -> str:
        return (
            f"BenchmarkRunner("
            f"functions={[f.name for f in self._functions]}, "
            f"optimizers={[o.name for o in self._optimizers]})"
        )
