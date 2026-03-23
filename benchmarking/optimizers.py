"""
Optimization algorithms for benchmarking on non-convex landscapes.

Each optimizer follows a common interface:
  - ``optimize(func, x0, ...)`` → ``OptimizeResult``
  - ``trajectory`` attribute stores the iterate history after a run.

All algorithms record per-iteration metrics to enable trajectory analysis.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OptimizeResult:
    """Result of an optimization run."""

    x: np.ndarray                         # best point found
    fun: float                            # function value at *x*
    nit: int                              # number of iterations
    nfev: int                             # number of function evaluations
    success: bool                         # converged within tolerance?
    message: str                          # human-readable status
    trajectory: np.ndarray                # shape (T, dim)
    values: np.ndarray                    # f-values along the trajectory
    grad_norms: np.ndarray                # gradient norms along trajectory
    elapsed_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"OptimizeResult(fun={self.fun:.6g}, nit={self.nit}, "
            f"nfev={self.nfev}, success={self.success})"
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseOptimizer(abc.ABC):
    """Abstract base class for all optimizers."""

    name: str = "BaseOptimizer"

    def __init__(
        self,
        lr: float = 1e-3,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        record_every: int = 1,
    ) -> None:
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.record_every = record_every

    @abc.abstractmethod
    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizeResult:
        """Run the optimizer.

        Parameters
        ----------
        func:
            Objective function ``f(x) → float``.
        grad_fn:
            Gradient function ``∇f(x) → np.ndarray``.
        x0:
            Starting point.

        Returns
        -------
        OptimizeResult
        """

    def _make_result(
        self,
        x: np.ndarray,
        fun: float,
        nit: int,
        nfev: int,
        success: bool,
        message: str,
        trajectory: List[np.ndarray],
        values: List[float],
        grad_norms: List[float],
        t0: float,
    ) -> OptimizeResult:
        return OptimizeResult(
            x=x.copy(),
            fun=fun,
            nit=nit,
            nfev=nfev,
            success=success,
            message=message,
            trajectory=np.array(trajectory),
            values=np.array(values),
            grad_norms=np.array(grad_norms),
            elapsed_seconds=time.perf_counter() - t0,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self.lr}, max_iter={self.max_iter})"


# ---------------------------------------------------------------------------
# Gradient Descent
# ---------------------------------------------------------------------------


class GradientDescent(BaseOptimizer):
    """Vanilla gradient descent with optional Armijo line search.

    Parameters
    ----------
    lr:
        Initial step size (learning rate).
    line_search:
        Whether to apply the Armijo backtracking line search at each step.
    """

    name = "GradientDescent"

    def __init__(
        self,
        lr: float = 1e-3,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        record_every: int = 1,
        line_search: bool = False,
    ) -> None:
        super().__init__(lr=lr, max_iter=max_iter, tol=tol, record_every=record_every)
        self.line_search = line_search

    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizeResult:
        t0 = time.perf_counter()
        x = np.asarray(x0, dtype=float).copy()
        traj: List[np.ndarray] = []
        vals: List[float] = []
        gnorms: List[float] = []
        nfev = 0

        for nit in range(1, self.max_iter + 1):
            f_val = func(x)
            g = grad_fn(x)
            nfev += 2
            g_norm = float(np.linalg.norm(g))

            if nit % self.record_every == 0 or nit == 1:
                traj.append(x.copy())
                vals.append(f_val)
                gnorms.append(g_norm)

            if g_norm < self.tol:
                return self._make_result(
                    x, f_val, nit, nfev, True, "Converged (gradient norm < tol)",
                    traj, vals, gnorms, t0,
                )

            step = self.lr
            if self.line_search:
                step, extra_fev = _armijo_line_search(func, x, g, f_val)
                nfev += extra_fev

            x -= step * g

        f_val = func(x)
        g = grad_fn(x)
        nfev += 2
        traj.append(x.copy())
        vals.append(f_val)
        gnorms.append(float(np.linalg.norm(g)))
        return self._make_result(
            x, f_val, nit, nfev, False, "Maximum iterations reached",
            traj, vals, gnorms, t0,
        )


# ---------------------------------------------------------------------------
# SGD with Momentum
# ---------------------------------------------------------------------------


class SGDMomentum(BaseOptimizer):
    """Stochastic Gradient Descent with Nesterov-style momentum."""

    name = "SGDMomentum"

    def __init__(
        self,
        lr: float = 1e-3,
        momentum: float = 0.9,
        nesterov: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        record_every: int = 1,
    ) -> None:
        super().__init__(lr=lr, max_iter=max_iter, tol=tol, record_every=record_every)
        self.momentum = momentum
        self.nesterov = nesterov

    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizeResult:
        t0 = time.perf_counter()
        x = np.asarray(x0, dtype=float).copy()
        v = np.zeros_like(x)
        traj: List[np.ndarray] = []
        vals: List[float] = []
        gnorms: List[float] = []
        nfev = 0

        for nit in range(1, self.max_iter + 1):
            if self.nesterov:
                x_look = x - self.momentum * v
                g = grad_fn(x_look)
            else:
                g = grad_fn(x)
            nfev += 1

            f_val = func(x)
            nfev += 1
            g_norm = float(np.linalg.norm(g))

            if nit % self.record_every == 0 or nit == 1:
                traj.append(x.copy())
                vals.append(f_val)
                gnorms.append(g_norm)

            if g_norm < self.tol:
                return self._make_result(
                    x, f_val, nit, nfev, True, "Converged (gradient norm < tol)",
                    traj, vals, gnorms, t0,
                )

            v = self.momentum * v + self.lr * g
            x -= v

        f_val = func(x)
        traj.append(x.copy())
        vals.append(f_val)
        gnorms.append(float(np.linalg.norm(grad_fn(x))))
        nfev += 2
        return self._make_result(
            x, f_val, nit, nfev, False, "Maximum iterations reached",
            traj, vals, gnorms, t0,
        )


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


class Adam(BaseOptimizer):
    """Adam optimizer (Kingma & Ba, 2015).

    Parameters
    ----------
    lr:
        Learning rate (α).
    beta1:
        Exponential decay rate for the first moment estimate.
    beta2:
        Exponential decay rate for the second moment estimate.
    epsilon:
        Small constant for numerical stability.
    """

    name = "Adam"

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        record_every: int = 1,
    ) -> None:
        super().__init__(lr=lr, max_iter=max_iter, tol=tol, record_every=record_every)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizeResult:
        t0 = time.perf_counter()
        x = np.asarray(x0, dtype=float).copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        traj: List[np.ndarray] = []
        vals: List[float] = []
        gnorms: List[float] = []
        nfev = 0

        for nit in range(1, self.max_iter + 1):
            g = grad_fn(x)
            f_val = func(x)
            nfev += 2
            g_norm = float(np.linalg.norm(g))

            if nit % self.record_every == 0 or nit == 1:
                traj.append(x.copy())
                vals.append(f_val)
                gnorms.append(g_norm)

            if g_norm < self.tol:
                return self._make_result(
                    x, f_val, nit, nfev, True, "Converged (gradient norm < tol)",
                    traj, vals, gnorms, t0,
                )

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            m_hat = m / (1 - self.beta1 ** nit)
            v_hat = v / (1 - self.beta2 ** nit)
            x -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        f_val = func(x)
        traj.append(x.copy())
        vals.append(f_val)
        gnorms.append(float(np.linalg.norm(grad_fn(x))))
        nfev += 2
        return self._make_result(
            x, f_val, nit, nfev, False, "Maximum iterations reached",
            traj, vals, gnorms, t0,
        )


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------


class RMSprop(BaseOptimizer):
    """RMSprop optimizer.

    Parameters
    ----------
    lr:
        Learning rate.
    decay:
        Exponential moving-average decay for the squared gradient.
    epsilon:
        Small constant for numerical stability.
    """

    name = "RMSprop"

    def __init__(
        self,
        lr: float = 1e-3,
        decay: float = 0.9,
        epsilon: float = 1e-8,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        record_every: int = 1,
    ) -> None:
        super().__init__(lr=lr, max_iter=max_iter, tol=tol, record_every=record_every)
        self.decay = decay
        self.epsilon = epsilon

    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizeResult:
        t0 = time.perf_counter()
        x = np.asarray(x0, dtype=float).copy()
        sq_avg = np.ones_like(x)
        traj: List[np.ndarray] = []
        vals: List[float] = []
        gnorms: List[float] = []
        nfev = 0

        for nit in range(1, self.max_iter + 1):
            g = grad_fn(x)
            f_val = func(x)
            nfev += 2
            g_norm = float(np.linalg.norm(g))

            if nit % self.record_every == 0 or nit == 1:
                traj.append(x.copy())
                vals.append(f_val)
                gnorms.append(g_norm)

            if g_norm < self.tol:
                return self._make_result(
                    x, f_val, nit, nfev, True, "Converged (gradient norm < tol)",
                    traj, vals, gnorms, t0,
                )

            sq_avg = self.decay * sq_avg + (1 - self.decay) * g ** 2
            x -= self.lr * g / (np.sqrt(sq_avg) + self.epsilon)

        f_val = func(x)
        traj.append(x.copy())
        vals.append(f_val)
        gnorms.append(float(np.linalg.norm(grad_fn(x))))
        nfev += 2
        return self._make_result(
            x, f_val, nit, nfev, False, "Maximum iterations reached",
            traj, vals, gnorms, t0,
        )


# ---------------------------------------------------------------------------
# L-BFGS (wraps scipy)
# ---------------------------------------------------------------------------


class LBFGS(BaseOptimizer):
    """L-BFGS optimizer (wraps :func:`scipy.optimize.minimize` with method ``L-BFGS-B``).

    Parameters
    ----------
    lr:
        Not used directly (scipy handles step sizes internally); kept for API
        consistency.
    m:
        Number of past iterates to keep in memory.
    """

    name = "LBFGS"

    def __init__(
        self,
        lr: float = 1.0,
        m: int = 10,
        max_iter: int = 1000,
        tol: float = 1e-6,
        record_every: int = 1,
    ) -> None:
        super().__init__(lr=lr, max_iter=max_iter, tol=tol, record_every=record_every)
        self.m = m

    def optimize(
        self,
        func: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizeResult:
        t0 = time.perf_counter()
        x0 = np.asarray(x0, dtype=float)
        traj: List[np.ndarray] = []
        vals: List[float] = []
        gnorms: List[float] = []
        nfev_counter = [0]
        iter_counter = [0]

        def callback(xk: np.ndarray) -> None:
            iter_counter[0] += 1
            if iter_counter[0] % self.record_every == 0:
                fk = func(xk)
                gk = grad_fn(xk)
                traj.append(xk.copy())
                vals.append(fk)
                gnorms.append(float(np.linalg.norm(gk)))

        def func_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            nfev_counter[0] += 1
            return func(x), grad_fn(x)

        res = scipy_minimize(
            func_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol, "maxcor": self.m},
            callback=callback,
        )

        if not traj:
            traj.append(res.x.copy())
            vals.append(float(res.fun))
            g = grad_fn(res.x)
            gnorms.append(float(np.linalg.norm(g)))

        return self._make_result(
            res.x,
            float(res.fun),
            iter_counter[0],
            nfev_counter[0],
            bool(res.success),
            res.message,
            traj,
            vals,
            gnorms,
            t0,
        )


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


_OPTIMIZER_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "gradient_descent": GradientDescent,
    "sgd_momentum": SGDMomentum,
    "adam": Adam,
    "rmsprop": RMSprop,
    "lbfgs": LBFGS,
}

ALL_OPTIMIZERS = list(_OPTIMIZER_REGISTRY.keys())


def get_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """Instantiate an optimizer by *name*.

    Parameters
    ----------
    name:
        Case-insensitive optimizer name (e.g. ``"adam"``).
    **kwargs:
        Passed to the optimizer constructor.

    Returns
    -------
    BaseOptimizer
    """
    key = name.lower()
    if key not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {sorted(_OPTIMIZER_REGISTRY)}."
        )
    return _OPTIMIZER_REGISTRY[key](**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _armijo_line_search(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    grad: np.ndarray,
    f0: float,
    alpha0: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_steps: int = 50,
) -> Tuple[float, int]:
    """Backtracking Armijo line search.

    Returns
    -------
    (step_size, n_function_evaluations)
    """
    alpha = alpha0
    descent = -np.dot(grad, grad)  # directional derivative along -grad
    nfev = 0
    for _ in range(max_steps):
        x_new = x - alpha * grad
        f_new = func(x_new)
        nfev += 1
        if f_new <= f0 + c * alpha * descent:
            return alpha, nfev
        alpha *= rho
    return alpha, nfev
