"""Tests for benchmarking.symbolic."""

import numpy as np
import pytest
import sympy as sp

from benchmarking.symbolic import SymbolicAnalyzer


# ---------------------------------------------------------------------------
# Sphere (simple closed-form)
# ---------------------------------------------------------------------------


class TestSymbolicSphere:
    def setup_method(self):
        self.sa = SymbolicAnalyzer.from_sphere(dim=2)

    def test_value_at_origin(self):
        assert self.sa.value_at([0.0, 0.0]) == pytest.approx(0.0)

    def test_gradient_at_origin_is_zero(self):
        g = self.sa.gradient_at([0.0, 0.0])
        np.testing.assert_allclose(g, [0.0, 0.0], atol=1e-12)

    def test_hessian_is_2I(self):
        H = self.sa.hessian_at([1.0, 2.0])
        np.testing.assert_allclose(H, 2 * np.eye(2), atol=1e-10)

    def test_gradient_matches_numerical(self):
        from benchmarking.functions import Sphere
        f = Sphere(dim=2)
        pt = [1.5, -0.7]
        sym_g = self.sa.gradient_at(pt)
        num_g = f.gradient(np.array(pt))
        np.testing.assert_allclose(sym_g, num_g, rtol=1e-5)


# ---------------------------------------------------------------------------
# Himmelblau (known critical points)
# ---------------------------------------------------------------------------


class TestSymbolicHimmelblau:
    def setup_method(self):
        self.sa = SymbolicAnalyzer.from_himmelblau()

    def test_value_at_known_minimum(self):
        val = self.sa.value_at([3.0, 2.0])
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_gradient_near_zero_at_minimum(self):
        g = self.sa.gradient_at([3.0, 2.0])
        assert np.linalg.norm(g) < 1e-8

    def test_hessian_positive_definite_at_minimum(self):
        H = self.sa.hessian_at([3.0, 2.0])
        eigvals = np.linalg.eigvalsh(H)
        assert np.all(eigvals > 0), f"Eigenvalues not all positive: {eigvals}"

    def test_critical_points_found(self):
        pts = self.sa.find_critical_points()
        assert len(pts) >= 1

    def test_classify_known_minimum(self):
        pts = self.sa.find_critical_points()
        # At least one should be classified as a local minimum
        classifications = [self.sa.classify_critical_point(p) for p in pts]
        assert "local minimum" in classifications


# ---------------------------------------------------------------------------
# Rosenbrock (symbolic gradient check)
# ---------------------------------------------------------------------------


class TestSymbolicRosenbrock:
    def setup_method(self):
        self.sa = SymbolicAnalyzer.from_rosenbrock(dim=2)

    def test_value_at_global_min(self):
        assert self.sa.value_at([1.0, 1.0]) == pytest.approx(0.0, abs=1e-10)

    def test_gradient_at_global_min_is_zero(self):
        g = self.sa.gradient_at([1.0, 1.0])
        np.testing.assert_allclose(g, [0.0, 0.0], atol=1e-8)


# ---------------------------------------------------------------------------
# Trajectory annotation
# ---------------------------------------------------------------------------


class TestTrajectoryInfo:
    def test_trajectory_info_shape(self):
        sa = SymbolicAnalyzer.from_sphere(dim=2)
        traj = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
        info = sa.convergence_trajectory_info(traj)
        assert len(info) == 3
        for entry in info:
            assert "point" in entry
            assert "value" in entry
            assert "grad_norm" in entry
            assert "hessian_eigenvalues" in entry

    def test_final_point_has_zero_grad(self):
        sa = SymbolicAnalyzer.from_sphere(dim=2)
        traj = np.array([[1.0, 1.0], [0.0, 0.0]])
        info = sa.convergence_trajectory_info(traj)
        assert info[-1]["grad_norm"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Summary / repr
# ---------------------------------------------------------------------------


class TestDisplayHelpers:
    def test_summary_contains_function_name(self):
        sa = SymbolicAnalyzer.from_himmelblau()
        s = sa.summary()
        assert "Himmelblau" in s

    def test_repr(self):
        sa = SymbolicAnalyzer.from_sphere(dim=3)
        r = repr(sa)
        assert "SymbolicAnalyzer" in r and "3" in r
