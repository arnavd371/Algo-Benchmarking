"""
Algorithmic Benchmarking in Non-Convex Optimization Landscapes.

Analyzes the geometry of high-dimensional non-convex optimization landscapes
using symbolic computation to track convergence trajectories.
"""

from .functions import (
    BenchmarkFunction,
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
from .symbolic import SymbolicAnalyzer
from .geometry import LandscapeGeometry
from .optimizers import (
    GradientDescent,
    SGDMomentum,
    Adam,
    RMSprop,
    LBFGS,
    get_optimizer,
    ALL_OPTIMIZERS,
)
from .benchmark import BenchmarkRunner, BenchmarkResult

__all__ = [
    "BenchmarkFunction",
    "Rastrigin",
    "Ackley",
    "Rosenbrock",
    "Himmelblau",
    "Levy",
    "Schwefel",
    "Griewank",
    "Sphere",
    "get_function",
    "ALL_FUNCTIONS",
    "SymbolicAnalyzer",
    "LandscapeGeometry",
    "GradientDescent",
    "SGDMomentum",
    "Adam",
    "RMSprop",
    "LBFGS",
    "get_optimizer",
    "ALL_OPTIMIZERS",
    "BenchmarkRunner",
    "BenchmarkResult",
]
