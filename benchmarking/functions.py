"""
Non-convex benchmark functions for optimization testing.

Each function exposes a NumPy-compatible callable interface together with
metadata about its known global minimum, search bounds, and dimensionality.
"""

from __future__ import annotations

import abc
from typing import ClassVar, Tuple

import numpy as np


class BenchmarkFunction(abc.ABC):
    """Abstract base class for all benchmark functions.

    Subclasses must implement :meth:`evaluate`, and should set the class-level
    attributes *name*, *global_min_value*, *global_min_location*, and
    *default_bounds*.
    """

    name: ClassVar[str] = "BenchmarkFunction"
    global_min_value: ClassVar[float] = 0.0
    default_bounds: ClassVar[Tuple[float, float]] = (-5.0, 5.0)

    def __init__(self, dim: int = 2) -> None:
        if dim < 1:
            raise ValueError("Dimensionality must be at least 1.")
        self.dim = dim

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(
                f"{self.name} expects a 1-D array of length {self.dim}, "
                f"got shape {x.shape}."
            )
        return float(self.evaluate(x))

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Compute the function value at *x*."""

    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Finite-difference gradient (central differences)."""
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x)
        for i in range(self.dim):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (self.evaluate(xp) - self.evaluate(xm)) / (2 * eps)
        return grad

    def hessian(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Finite-difference Hessian (central second differences)."""
        x = np.asarray(x, dtype=float)
        H = np.zeros((self.dim, self.dim))
        f0 = self.evaluate(x)
        for i in range(self.dim):
            for j in range(i, self.dim):
                xpp = x.copy()
                xmm = x.copy()
                xpm = x.copy()
                xmp = x.copy()
                xpp[i] += eps
                xpp[j] += eps
                xmm[i] -= eps
                xmm[j] -= eps
                xpm[i] += eps
                xpm[j] -= eps
                xmp[i] -= eps
                xmp[j] += eps
                val = (
                    self.evaluate(xpp)
                    - self.evaluate(xpm)
                    - self.evaluate(xmp)
                    + self.evaluate(xmm)
                ) / (4 * eps ** 2)
                H[i, j] = val
                H[j, i] = val
        return H

    def global_min_location(self) -> np.ndarray:
        """Return the known global minimizer (repeated for each dimension)."""
        return np.zeros(self.dim)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


# ---------------------------------------------------------------------------
# Concrete benchmark functions
# ---------------------------------------------------------------------------


class Rastrigin(BenchmarkFunction):
    """Rastrigin function — highly multimodal with a large number of local minima.

    .. math::
        f(\\mathbf{x}) = An + \\sum_{i=1}^{n} \\left[ x_i^2 - A \\cos(2\\pi x_i) \\right]

    Global minimum: *f*(0, …, 0) = 0.
    """

    name = "Rastrigin"
    global_min_value = 0.0
    default_bounds = (-5.12, 5.12)

    def __init__(self, dim: int = 2, A: float = 10.0) -> None:
        super().__init__(dim)
        self.A = A

    def evaluate(self, x: np.ndarray) -> float:
        return self.A * self.dim + np.sum(x ** 2 - self.A * np.cos(2 * np.pi * x))

    def global_min_location(self) -> np.ndarray:
        return np.zeros(self.dim)


class Ackley(BenchmarkFunction):
    """Ackley function — nearly flat outer region with a deep central pit.

    .. math::
        f(\\mathbf{x}) = -a \\exp\\left(-b \\sqrt{\\frac{1}{n}\\sum x_i^2}\\right)
        - \\exp\\left(\\frac{1}{n}\\sum \\cos(cx_i)\\right) + a + e

    Global minimum: *f*(0, …, 0) = 0.
    """

    name = "Ackley"
    global_min_value = 0.0
    default_bounds = (-32.768, 32.768)

    def __init__(self, dim: int = 2, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> None:
        super().__init__(dim)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x: np.ndarray) -> float:
        n = self.dim
        sum1 = np.sqrt(np.sum(x ** 2) / n)
        sum2 = np.sum(np.cos(self.c * x)) / n
        return -self.a * np.exp(-self.b * sum1) - np.exp(sum2) + self.a + np.e

    def global_min_location(self) -> np.ndarray:
        return np.zeros(self.dim)


class Rosenbrock(BenchmarkFunction):
    """Rosenbrock (banana) function — a curved, narrow valley.

    .. math::
        f(\\mathbf{x}) = \\sum_{i=1}^{n-1} \\left[
            100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
        \\right]

    Global minimum: *f*(1, …, 1) = 0.
    """

    name = "Rosenbrock"
    global_min_value = 0.0
    default_bounds = (-5.0, 10.0)

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def global_min_location(self) -> np.ndarray:
        return np.ones(self.dim)


class Himmelblau(BenchmarkFunction):
    """Himmelblau function — four identical local minima (2-D only).

    .. math::
        f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    Global minima (all equal to 0):
    (3, 2), (−2.805, 3.131), (−3.779, −3.283), (3.584, −1.848).
    """

    name = "Himmelblau"
    global_min_value = 0.0
    default_bounds = (-5.0, 5.0)

    def __init__(self) -> None:
        super().__init__(dim=2)

    def evaluate(self, x: np.ndarray) -> float:
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def global_min_location(self) -> np.ndarray:
        return np.array([3.0, 2.0])


class Levy(BenchmarkFunction):
    """Levy function — multimodal with several local minima.

    Global minimum: *f*(1, …, 1) = 0.
    """

    name = "Levy"
    global_min_value = 0.0
    default_bounds = (-10.0, 10.0)

    def evaluate(self, x: np.ndarray) -> float:
        w = 1.0 + (x - 1.0) / 4.0
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum(
            (w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)
        )
        term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
        return term1 + term2 + term3

    def global_min_location(self) -> np.ndarray:
        return np.ones(self.dim)


class Schwefel(BenchmarkFunction):
    """Schwefel function — deceptive: global minimum far from next-best local minima.

    .. math::
        f(\\mathbf{x}) = 418.9829 n - \\sum_{i=1}^{n} x_i \\sin(\\sqrt{|x_i|})

    Global minimum: *f*(420.9687, …) ≈ 0.
    """

    name = "Schwefel"
    global_min_value = 0.0
    default_bounds = (-500.0, 500.0)

    def evaluate(self, x: np.ndarray) -> float:
        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def global_min_location(self) -> np.ndarray:
        return np.full(self.dim, 420.9687)


class Griewank(BenchmarkFunction):
    """Griewank function — product term introduces widespread local minima.

    .. math::
        f(\\mathbf{x}) = \\frac{1}{4000}\\sum x_i^2
        - \\prod \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1

    Global minimum: *f*(0, …, 0) = 0.
    """

    name = "Griewank"
    global_min_value = 0.0
    default_bounds = (-600.0, 600.0)

    def evaluate(self, x: np.ndarray) -> float:
        sum_term = np.sum(x ** 2) / 4000.0
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        return sum_term - prod_term + 1.0

    def global_min_location(self) -> np.ndarray:
        return np.zeros(self.dim)


class Sphere(BenchmarkFunction):
    """Sphere function — the canonical convex baseline.

    .. math::
        f(\\mathbf{x}) = \\sum_{i=1}^{n} x_i^2

    Global minimum: *f*(0, …, 0) = 0.
    """

    name = "Sphere"
    global_min_value = 0.0
    default_bounds = (-5.0, 5.0)

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def global_min_location(self) -> np.ndarray:
        return np.zeros(self.dim)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_FUNCTION_REGISTRY: dict[str, type[BenchmarkFunction]] = {
    "rastrigin": Rastrigin,
    "ackley": Ackley,
    "rosenbrock": Rosenbrock,
    "himmelblau": Himmelblau,
    "levy": Levy,
    "schwefel": Schwefel,
    "griewank": Griewank,
    "sphere": Sphere,
}

ALL_FUNCTIONS = list(_FUNCTION_REGISTRY.keys())


def get_function(name: str, dim: int = 2) -> BenchmarkFunction:
    """Instantiate a benchmark function by *name*.

    Parameters
    ----------
    name:
        Case-insensitive function name (e.g. ``"rastrigin"``).
    dim:
        Dimensionality.  Ignored for :class:`Himmelblau` which is fixed at 2.

    Returns
    -------
    BenchmarkFunction
        An instance of the requested function.
    """
    key = name.lower()
    if key not in _FUNCTION_REGISTRY:
        raise ValueError(
            f"Unknown function '{name}'. "
            f"Available: {sorted(_FUNCTION_REGISTRY)}."
        )
    cls = _FUNCTION_REGISTRY[key]
    if cls is Himmelblau:
        return cls()
    return cls(dim=dim)
