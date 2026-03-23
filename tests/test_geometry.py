"""Tests for benchmarking.geometry."""

import numpy as np
import pytest

from benchmarking.functions import Sphere, Rastrigin, Himmelblau
from benchmarking.geometry import LandscapeGeometry, CriticalPointInfo, LandscapeStats


class TestLocalGeometry:
    def test_sphere_at_origin_is_local_min(self):
        f = Sphere(dim=2)
        geom = LandscapeGeometry(f)
        info = geom.local_geometry(np.array([0.0, 0.0]))
        assert info.point_type == "local minimum"
        assert info.value == pytest.approx(0.0)
        assert np.all(info.hessian_eigenvalues > 0)

    def test_non_critical_point(self):
        f = Sphere(dim=2)
        geom = LandscapeGeometry(f)
        info = geom.local_geometry(np.array([1.0, 1.0]))
        # Gradient is non-zero at (1, 1) → classified as non-critical
        assert info.point_type == "non-critical"

    def test_critical_point_info_type(self):
        f = Sphere(dim=2)
        geom = LandscapeGeometry(f)
        info = geom.local_geometry(np.zeros(2))
        assert isinstance(info, CriticalPointInfo)


class TestLandscapeSampling:
    def test_sample_landscape_returns_stats(self):
        f = Rastrigin(dim=2)
        geom = LandscapeGeometry(f)
        stats = geom.sample_landscape(n_samples=50, seed=42)
        assert isinstance(stats, LandscapeStats)
        assert stats.n_samples == 50
        assert stats.value_mean >= stats.value_min
        assert stats.value_max >= stats.value_mean

    def test_sphere_value_min_near_zero(self):
        f = Sphere(dim=2)
        geom = LandscapeGeometry(f)
        stats = geom.sample_landscape(n_samples=200, seed=0)
        assert stats.value_min >= -1e-10  # sphere is non-negative

    def test_hessian_spectrum_length(self):
        f = Sphere(dim=2)
        geom = LandscapeGeometry(f)
        eigvals = geom.hessian_spectrum(n_samples=30, seed=0)
        assert len(eigvals) == 30 * 2  # 30 samples × dim=2


class TestTrajectoryGeometry:
    def test_trajectory_analysis_length(self):
        f = Sphere(dim=2)
        geom = LandscapeGeometry(f)
        traj = np.array([[2.0, 2.0], [1.0, 1.0], [0.1, 0.1]])
        infos = geom.analyse_trajectory(traj)
        assert len(infos) == 3

    def test_curvature_along_sphere_trajectory_constant(self):
        f = Sphere(dim=2)
        geom = LandscapeGeometry(f)
        # Sphere has Hessian 2I everywhere → max |eigenvalue| = 2
        traj = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 3.0]])
        curvs = geom.curvature_along_trajectory(traj)
        np.testing.assert_allclose(curvs, 2.0, atol=1e-4)

    def test_gradient_norms_decrease_along_convergence(self):
        from benchmarking.optimizers import Adam
        f = Sphere(dim=2)
        opt = Adam(lr=0.1, max_iter=200, tol=1e-10)
        res = opt.optimize(f, f.gradient, np.array([3.0, 3.0]))
        geom = LandscapeGeometry(f)
        gnorms = geom.gradient_norms(res.trajectory)
        # The trajectory should generally be converging (norms decrease)
        # Relaxed check: final norm < initial norm
        assert gnorms[-1] < gnorms[0]
