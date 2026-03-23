"""Tests for benchmarking.optimizers."""

import numpy as np
import pytest

from benchmarking.functions import Sphere, Rosenbrock
from benchmarking.optimizers import (
    GradientDescent,
    SGDMomentum,
    Adam,
    RMSprop,
    LBFGS,
    get_optimizer,
    ALL_OPTIMIZERS,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sphere_2d():
    return Sphere(dim=2)


def _run(optimizer, func, x0):
    return optimizer.optimize(func, func.gradient, np.asarray(x0, dtype=float))


# ---------------------------------------------------------------------------
# Convergence on convex sphere (easy test)
# ---------------------------------------------------------------------------


class TestConvergenceOnSphere:
    """All optimizers should converge to near-zero on the 2-D sphere."""

    @pytest.mark.parametrize("OptimizerCls,kwargs", [
        (GradientDescent, {"lr": 0.1, "max_iter": 5000}),
        (SGDMomentum,     {"lr": 0.05, "momentum": 0.9, "max_iter": 5000}),
        (Adam,            {"lr": 0.1, "max_iter": 5000}),
        (RMSprop,         {"lr": 0.1, "max_iter": 5000}),
        (LBFGS,           {"max_iter": 500}),
    ])
    def test_converges_on_sphere(self, OptimizerCls, kwargs):
        f = _sphere_2d()
        opt = OptimizerCls(**kwargs, tol=1e-8)
        res = _run(opt, f, [2.0, -3.0])
        assert res.fun < 1e-5, (
            f"{OptimizerCls.__name__} did not converge on sphere; "
            f"final value = {res.fun:.4e}"
        )


# ---------------------------------------------------------------------------
# Trajectory consistency
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_trajectory_shape(self):
        f = _sphere_2d()
        opt = Adam(lr=0.1, max_iter=200, tol=1e-12)
        res = _run(opt, f, [1.0, 1.0])
        T = res.trajectory.shape[0]
        assert res.trajectory.shape == (T, 2)
        assert len(res.values) == T
        assert len(res.grad_norms) == T

    def test_values_non_increasing_on_sphere_gd(self):
        """Gradient descent on a convex function should be monotone."""
        f = _sphere_2d()
        opt = GradientDescent(lr=0.1, max_iter=500, tol=1e-12, line_search=False)
        res = _run(opt, f, [2.0, 3.0])
        diffs = np.diff(res.values)
        # Allow for tiny numerical noise
        assert np.all(diffs <= 1e-8), "GD produced non-monotone values on sphere."

    def test_record_every(self):
        f = _sphere_2d()
        opt = Adam(lr=0.1, max_iter=100, tol=1e-12, record_every=10)
        res = _run(opt, f, [1.0, 1.0])
        # With record_every=10 over 100 iters we expect roughly 10 recorded points
        assert len(res.values) <= 15  # a bit of tolerance for first/last

    def test_first_trajectory_point_matches_x0(self):
        f = _sphere_2d()
        x0 = np.array([1.5, -2.5])
        opt = GradientDescent(lr=0.1, max_iter=100, tol=1e-12)
        res = _run(opt, f, x0)
        np.testing.assert_allclose(res.trajectory[0], x0, atol=1e-12)


# ---------------------------------------------------------------------------
# get_optimizer registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_optimizers_constructable(self):
        for name in ALL_OPTIMIZERS:
            opt = get_optimizer(name)
            assert hasattr(opt, "optimize")

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError):
            get_optimizer("does_not_exist")

    def test_kwargs_passed(self):
        opt = get_optimizer("adam", lr=0.5)
        assert opt.lr == 0.5


# ---------------------------------------------------------------------------
# Line search GD
# ---------------------------------------------------------------------------


class TestLineSearch:
    def test_line_search_gd_converges(self):
        f = _sphere_2d()
        opt = GradientDescent(lr=1.0, line_search=True, max_iter=500, tol=1e-8)
        res = _run(opt, f, [3.0, -4.0])
        assert res.fun < 1e-5
