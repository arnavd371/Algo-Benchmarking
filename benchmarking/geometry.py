"""
Landscape geometry analysis for non-convex optimization functions.

Provides tools to characterise the local geometry of a function at a given
point, to estimate the density of critical points in a search region, and to
compute landscape-level statistics that are useful for benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import eigvalsh

from .functions import BenchmarkFunction


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CriticalPointInfo:
    """Information about a (numerically located) critical point."""

    location: np.ndarray
    value: float
    grad_norm: float
    hessian_eigenvalues: np.ndarray
    point_type: str  # "local minimum" | "local maximum" | "saddle point" | "unknown"

    def __repr__(self) -> str:
        return (
            f"CriticalPointInfo(type={self.point_type!r}, "
            f"value={self.value:.6g}, "
            f"grad_norm={self.grad_norm:.2e})"
        )


@dataclass
class LandscapeStats:
    """Aggregated statistics describing a landscape sample."""

    n_samples: int
    value_mean: float
    value_std: float
    value_min: float
    value_max: float
    grad_norm_mean: float
    grad_norm_std: float
    n_saddle_estimates: int = 0
    n_local_min_estimates: int = 0
    smoothness_index: float = 0.0  # normalised std of gradient norms
    ruggedness_index: float = 0.0  # normalised value range
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------


class LandscapeGeometry:
    """Analyse and characterise the geometry of a benchmark function's landscape.

    Parameters
    ----------
    func:
        The :class:`~benchmarking.functions.BenchmarkFunction` to analyze.
    """

    def __init__(self, func: BenchmarkFunction) -> None:
        self.func = func

    # ------------------------------------------------------------------
    # Point-level geometry
    # ------------------------------------------------------------------

    def local_geometry(self, x: np.ndarray, eps: float = 1e-5) -> CriticalPointInfo:
        """Compute local curvature information at *x*.

        Parameters
        ----------
        x:
            Query point.
        eps:
            Step size for finite-difference Hessian.

        Returns
        -------
        CriticalPointInfo
        """
        x = np.asarray(x, dtype=float)
        val = self.func(x)
        grad = self.func.gradient(x)
        grad_norm = float(np.linalg.norm(grad))
        H = self.func.hessian(x, eps=eps)
        eigvals = eigvalsh(H)
        point_type = self._classify_from_eigvals(eigvals, grad_norm)
        return CriticalPointInfo(
            location=x.copy(),
            value=val,
            grad_norm=grad_norm,
            hessian_eigenvalues=eigvals,
            point_type=point_type,
        )

    @staticmethod
    def _classify_from_eigvals(eigvals: np.ndarray, grad_norm: float, tol: float = 1e-4) -> str:
        """Classify a point given its Hessian eigenvalues and gradient norm."""
        if grad_norm > tol:
            return "non-critical"
        pos = np.all(eigvals > tol)
        neg = np.all(eigvals < -tol)
        if pos:
            return "local minimum"
        if neg:
            return "local maximum"
        mixed = np.any(eigvals > tol) and np.any(eigvals < -tol)
        if mixed:
            return "saddle point"
        return "unknown"

    # ------------------------------------------------------------------
    # Landscape-level statistics
    # ------------------------------------------------------------------

    def sample_landscape(
        self,
        n_samples: int = 1000,
        bounds: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ) -> LandscapeStats:
        """Randomly sample the landscape and compute summary statistics.

        Parameters
        ----------
        n_samples:
            Number of random points to evaluate.
        bounds:
            ``(low, high)`` for each coordinate.  Defaults to
            ``func.default_bounds``.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        LandscapeStats
        """
        rng = np.random.default_rng(seed)
        lo, hi = bounds if bounds is not None else self.func.default_bounds
        points = rng.uniform(lo, hi, size=(n_samples, self.func.dim))
        values = np.array([self.func(p) for p in points])
        grad_norms = np.array([np.linalg.norm(self.func.gradient(p)) for p in points])

        v_range = float(values.max() - values.min())
        g_mean = float(grad_norms.mean())

        smoothness = float(grad_norms.std() / (g_mean + 1e-12))
        ruggedness = v_range / (abs(float(values.mean())) + 1e-12)

        return LandscapeStats(
            n_samples=n_samples,
            value_mean=float(values.mean()),
            value_std=float(values.std()),
            value_min=float(values.min()),
            value_max=float(values.max()),
            grad_norm_mean=g_mean,
            grad_norm_std=float(grad_norms.std()),
            smoothness_index=smoothness,
            ruggedness_index=ruggedness,
        )

    def hessian_spectrum(
        self,
        n_samples: int = 200,
        bounds: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample Hessian eigenvalues across the landscape.

        Returns
        -------
        np.ndarray
            Array of shape *(n_samples × dim,)* containing all eigenvalues.
        """
        rng = np.random.default_rng(seed)
        lo, hi = bounds if bounds is not None else self.func.default_bounds
        points = rng.uniform(lo, hi, size=(n_samples, self.func.dim))
        all_eigvals: List[np.ndarray] = []
        for p in points:
            H = self.func.hessian(p)
            all_eigvals.append(eigvalsh(H))
        return np.concatenate(all_eigvals)

    def estimate_saddle_density(
        self,
        n_samples: int = 500,
        bounds: Optional[Tuple[float, float]] = None,
        grad_tol: float = 1e-2,
        seed: Optional[int] = None,
    ) -> float:
        """Estimate the fraction of near-critical points that are saddle points.

        A point is considered *near-critical* if its gradient norm is below
        *grad_tol*.  Among near-critical points, a saddle is identified by
        having both positive and negative Hessian eigenvalues.

        Returns
        -------
        float
            Estimated saddle-point density (0–1).  Returns NaN if no
            near-critical points were found.
        """
        rng = np.random.default_rng(seed)
        lo, hi = bounds if bounds is not None else self.func.default_bounds
        points = rng.uniform(lo, hi, size=(n_samples, self.func.dim))

        near_critical = []
        for p in points:
            g = self.func.gradient(p)
            if np.linalg.norm(g) < grad_tol:
                near_critical.append(p)

        if not near_critical:
            return float("nan")

        n_saddle = sum(
            1
            for p in near_critical
            if self._classify_from_eigvals(
                eigvalsh(self.func.hessian(p)), 0.0, tol=1e-4
            )
            == "saddle point"
        )
        return n_saddle / len(near_critical)

    # ------------------------------------------------------------------
    # Trajectory geometry
    # ------------------------------------------------------------------

    def analyse_trajectory(self, trajectory: np.ndarray) -> List[CriticalPointInfo]:
        """Compute local geometry at every point in an optimization trajectory.

        Alias: :meth:`analyze_trajectory`.

        Parameters
        ----------
        trajectory:
            Array of shape *(T, d)*.

        Returns
        -------
        list of CriticalPointInfo
        """
        return [self.local_geometry(pt) for pt in trajectory]

    # American English alias
    analyze_trajectory = analyse_trajectory

    def curvature_along_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Return the maximum absolute Hessian eigenvalue along a trajectory.

        This is a proxy for the *local curvature* at each iterate.

        Returns
        -------
        np.ndarray
            Shape *(T,)*.
        """
        result = []
        for pt in trajectory:
            H = self.func.hessian(pt)
            eigvals = eigvalsh(H)
            result.append(float(np.max(np.abs(eigvals))))
        return np.array(result)

    def gradient_norms(self, trajectory: np.ndarray) -> np.ndarray:
        """Return gradient norms along a trajectory.

        Returns
        -------
        np.ndarray
            Shape *(T,)*.
        """
        return np.array([np.linalg.norm(self.func.gradient(pt)) for pt in trajectory])

    # ------------------------------------------------------------------
    # Basin of attraction estimation
    # ------------------------------------------------------------------

    def basin_of_attraction_volume(
        self,
        target: np.ndarray,
        optimizer_fn,
        n_trials: int = 200,
        bounds: Optional[Tuple[float, float]] = None,
        convergence_radius: float = 0.5,
        seed: Optional[int] = None,
    ) -> float:
        """Estimate the volume fraction of the search space that converges to *target*.

        *optimizer_fn* should accept a starting point and return the converged
        solution, e.g. ``lambda x0: scipy.optimize.minimize(f, x0).x``.

        Returns
        -------
        float
            Fraction of random starts that converge within *convergence_radius*
            of *target*.
        """
        rng = np.random.default_rng(seed)
        lo, hi = bounds if bounds is not None else self.func.default_bounds
        starts = rng.uniform(lo, hi, size=(n_trials, self.func.dim))
        in_basin = sum(
            1
            for x0 in starts
            if np.linalg.norm(optimizer_fn(x0) - target) <= convergence_radius
        )
        return in_basin / n_trials
