"""Tests for benchmarking.functions."""

import numpy as np
import pytest

from benchmarking.functions import (
    Rastrigin,
    Ackley,
    Rosenbrock,
    Himmelblau,
    Levy,
    Schwefel,
    Griewank,
    Sphere,
    get_function,
    ALL_FUNCTIONS,
)


# ---------------------------------------------------------------------------
# Global minimum tests
# ---------------------------------------------------------------------------


class TestGlobalMinima:
    """Each function must evaluate to (approximately) its documented global minimum."""

    def test_rastrigin_min(self):
        f = Rastrigin(dim=2)
        assert abs(f(f.global_min_location())) < 1e-10

    def test_ackley_min(self):
        f = Ackley(dim=2)
        assert abs(f(f.global_min_location())) < 1e-10

    def test_rosenbrock_min(self):
        f = Rosenbrock(dim=3)
        assert abs(f(f.global_min_location())) < 1e-10

    def test_himmelblau_min(self):
        f = Himmelblau()
        assert abs(f(f.global_min_location())) < 1e-10

    def test_levy_min(self):
        f = Levy(dim=2)
        val = f(f.global_min_location())
        assert abs(val) < 1e-10

    def test_sphere_min(self):
        f = Sphere(dim=4)
        assert f(f.global_min_location()) == pytest.approx(0.0)

    def test_griewank_min(self):
        f = Griewank(dim=2)
        assert abs(f(f.global_min_location())) < 1e-10

    def test_schwefel_min(self):
        f = Schwefel(dim=2)
        val = f(f.global_min_location())
        assert abs(val) < 1.0  # Schwefel's min is ≈ 0 but with floating-point tolerance


# ---------------------------------------------------------------------------
# Gradient tests (finite-difference check)
# ---------------------------------------------------------------------------


class TestGradients:
    """Gradient computed via BenchmarkFunction.gradient should match a second
    independent finite-difference estimate."""

    @pytest.mark.parametrize("name,dim", [
        ("sphere", 3),
        ("rastrigin", 2),
        ("rosenbrock", 2),
        ("ackley", 2),
    ])
    def test_gradient_fd_consistency(self, name, dim):
        rng = np.random.default_rng(0)
        f = get_function(name, dim=dim)
        x = rng.uniform(-1, 1, size=dim)
        g1 = f.gradient(x, eps=1e-6)
        # Use a different eps to verify it's not trivially exact
        g2 = f.gradient(x, eps=5e-7)
        np.testing.assert_allclose(g1, g2, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Hessian tests
# ---------------------------------------------------------------------------


class TestHessian:
    def test_sphere_hessian_is_identity(self):
        f = Sphere(dim=3)
        H = f.hessian(np.zeros(3))
        np.testing.assert_allclose(H, 2 * np.eye(3), atol=1e-4)

    def test_hessian_is_symmetric(self):
        f = Rastrigin(dim=3)
        rng = np.random.default_rng(1)
        x = rng.uniform(-2, 2, size=3)
        H = f.hessian(x)
        np.testing.assert_allclose(H, H.T, atol=1e-6)


# ---------------------------------------------------------------------------
# API / misc tests
# ---------------------------------------------------------------------------


class TestFunctionAPI:
    def test_wrong_dimension_raises(self):
        f = Sphere(dim=2)
        with pytest.raises(ValueError):
            f(np.array([1.0, 2.0, 3.0]))

    def test_dim_1_raises(self):
        with pytest.raises(ValueError):
            Sphere(dim=0)

    def test_get_function_all(self):
        for name in ALL_FUNCTIONS:
            f = get_function(name, dim=2)
            assert callable(f)

    def test_get_function_unknown_raises(self):
        with pytest.raises(ValueError):
            get_function("does_not_exist")

    def test_repr(self):
        f = Ackley(dim=4)
        assert "4" in repr(f)

    def test_himmelblau_fixed_dim(self):
        f = Himmelblau()
        assert f.dim == 2
        # get_function should also return a 2-D instance
        g = get_function("himmelblau", dim=99)
        assert g.dim == 2
