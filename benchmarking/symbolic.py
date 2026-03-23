"""
Symbolic computation module for non-convex optimization landscapes.

Uses SymPy to derive exact gradients, Hessians, and locate critical points for
low-dimensional benchmark functions.  The results can be used to validate
numerical approximations and to study the geometry of the landscapes
analytically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp


class SymbolicAnalyzer:
    """Symbolic analyzer for a scalar function of *n* real variables.

    Parameters
    ----------
    expr:
        A SymPy expression representing the function.
    symbols:
        Ordered list of SymPy symbols (the independent variables).
    name:
        Optional human-readable name for display purposes.

    Examples
    --------
    Analyse the 2-D Himmelblau function symbolically::

        import sympy as sp
        from benchmarking.symbolic import SymbolicAnalyzer

        x, y = sp.symbols('x y', real=True)
        expr = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        sa = SymbolicAnalyzer(expr, [x, y], name="Himmelblau")
        print(sa.gradient_expr())
        print(sa.hessian_expr())
    """

    def __init__(
        self,
        expr: sp.Expr,
        symbols: Sequence[sp.Symbol],
        name: str = "f",
    ) -> None:
        self.expr = sp.expand(expr)
        self.symbols: List[sp.Symbol] = list(symbols)
        self.name = name
        self._dim = len(self.symbols)

        # Cached symbolic objects
        self._gradient: Optional[List[sp.Expr]] = None
        self._hessian: Optional[sp.Matrix] = None

    # ------------------------------------------------------------------
    # Symbolic derivations
    # ------------------------------------------------------------------

    def gradient_expr(self) -> List[sp.Expr]:
        """Return the symbolic gradient as a list of SymPy expressions."""
        if self._gradient is None:
            self._gradient = [sp.diff(self.expr, s) for s in self.symbols]
        return self._gradient

    def hessian_expr(self) -> sp.Matrix:
        """Return the symbolic Hessian matrix."""
        if self._hessian is None:
            grad = self.gradient_expr()
            self._hessian = sp.Matrix(
                [[sp.diff(g, s) for s in self.symbols] for g in grad]
            )
        return self._hessian

    def laplacian_expr(self) -> sp.Expr:
        """Return the symbolic Laplacian (trace of the Hessian)."""
        return sp.trace(self.hessian_expr())

    # ------------------------------------------------------------------
    # Numerical evaluation helpers
    # ------------------------------------------------------------------

    def gradient_at(self, point: Sequence[float]) -> np.ndarray:
        """Evaluate the symbolic gradient numerically at *point*."""
        subs = dict(zip(self.symbols, point))
        return np.array([float(g.subs(subs)) for g in self.gradient_expr()])

    def hessian_at(self, point: Sequence[float]) -> np.ndarray:
        """Evaluate the symbolic Hessian numerically at *point*."""
        subs = dict(zip(self.symbols, point))
        H = self.hessian_expr()
        return np.array(
            [[float(H[i, j].subs(subs)) for j in range(self._dim)] for i in range(self._dim)]
        )

    def value_at(self, point: Sequence[float]) -> float:
        """Evaluate the symbolic expression numerically at *point*."""
        subs = dict(zip(self.symbols, point))
        return float(self.expr.subs(subs))

    # ------------------------------------------------------------------
    # Critical point analysis
    # ------------------------------------------------------------------

    def find_critical_points(
        self, domain: Optional[Dict[sp.Symbol, sp.Interval]] = None
    ) -> List[Dict[sp.Symbol, sp.Expr]]:
        """Solve ∇f = 0 and return critical points.

        Parameters
        ----------
        domain:
            Optional mapping from symbol to a SymPy Interval to restrict
            solutions (not guaranteed to filter perfectly for all solvers).

        Returns
        -------
        list of dicts
            Each dict maps symbols to their values at a critical point.
            Returns an empty list if SymPy cannot find solutions.
        """
        grad = self.gradient_expr()
        try:
            sols = sp.solve(grad, self.symbols, dict=True)
        except NotImplementedError:
            return []
        if domain is not None:
            filtered = []
            for sol in sols:
                if all(
                    domain[s].contains(sol.get(s, sp.nan))
                    for s in domain
                    if s in sol
                ):
                    filtered.append(sol)
            return filtered
        return sols

    def classify_critical_point(
        self, point: Dict[sp.Symbol, sp.Expr]
    ) -> str:
        """Classify a critical point using the Hessian's eigenvalue signs.

        Uses numerical (floating-point) evaluation for speed; falls back to
        ``"degenerate / inconclusive"`` if the point contains complex numbers
        or the Hessian cannot be evaluated.

        Returns
        -------
        str
            One of ``"local minimum"``, ``"local maximum"``, ``"saddle point"``,
            or ``"degenerate / inconclusive"``.
        """
        # Build a numeric point, skipping complex solutions
        try:
            num_point = []
            for s in self.symbols:
                val = complex(point.get(s, sp.Integer(0)).evalf())
                if abs(val.imag) > 1e-6:
                    return "degenerate / inconclusive"
                num_point.append(val.real)
        except Exception:
            return "degenerate / inconclusive"

        try:
            H_num = self.hessian_at(num_point)
            eigvals = np.linalg.eigvalsh(H_num)
        except Exception:
            return "degenerate / inconclusive"

        tol = 1e-6
        pos = np.any(eigvals > tol)
        neg = np.any(eigvals < -tol)

        if pos and neg:
            return "saddle point"
        if pos:
            return "local minimum"
        if neg:
            return "local maximum"
        return "degenerate / inconclusive"

    # ------------------------------------------------------------------
    # Factory methods for built-in functions
    # ------------------------------------------------------------------

    @classmethod
    def from_rastrigin(cls, dim: int = 2, A: float = 10.0) -> "SymbolicAnalyzer":
        """Build a :class:`SymbolicAnalyzer` for the Rastrigin function."""
        syms = sp.symbols(f"x0:{dim}", real=True)
        expr = A * dim + sum(xi ** 2 - A * sp.cos(2 * sp.pi * xi) for xi in syms)
        return cls(expr, list(syms), name=f"Rastrigin_{dim}D")

    @classmethod
    def from_rosenbrock(cls, dim: int = 2) -> "SymbolicAnalyzer":
        """Build a :class:`SymbolicAnalyzer` for the Rosenbrock function."""
        syms = sp.symbols(f"x0:{dim}", real=True)
        expr = sum(
            100 * (syms[i + 1] - syms[i] ** 2) ** 2 + (1 - syms[i]) ** 2
            for i in range(dim - 1)
        )
        return cls(expr, list(syms), name=f"Rosenbrock_{dim}D")

    @classmethod
    def from_himmelblau(cls) -> "SymbolicAnalyzer":
        """Build a :class:`SymbolicAnalyzer` for the Himmelblau function."""
        x, y = sp.symbols("x y", real=True)
        expr = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
        return cls(expr, [x, y], name="Himmelblau")

    @classmethod
    def from_ackley(cls, dim: int = 2) -> "SymbolicAnalyzer":
        """Build a :class:`SymbolicAnalyzer` for the Ackley function."""
        syms = sp.symbols(f"x0:{dim}", real=True)
        n = dim
        sum1 = sp.sqrt(sum(xi ** 2 for xi in syms) / n)
        sum2 = sum(sp.cos(2 * sp.pi * xi) for xi in syms) / n
        expr = -20 * sp.exp(-sp.Rational(1, 5) * sum1) - sp.exp(sum2) + 20 + sp.E
        return cls(expr, list(syms), name=f"Ackley_{dim}D")

    @classmethod
    def from_sphere(cls, dim: int = 2) -> "SymbolicAnalyzer":
        """Build a :class:`SymbolicAnalyzer` for the Sphere function."""
        syms = sp.symbols(f"x0:{dim}", real=True)
        expr = sum(xi ** 2 for xi in syms)
        return cls(expr, list(syms), name=f"Sphere_{dim}D")

    # ------------------------------------------------------------------
    # Trajectory tracking
    # ------------------------------------------------------------------

    def convergence_trajectory_info(
        self, trajectory: np.ndarray
    ) -> List[dict]:
        """Annotate each point in an optimization trajectory.

        Parameters
        ----------
        trajectory:
            Array of shape *(T, d)* where *T* is the number of iterates and
            *d* equals ``self._dim``.

        Returns
        -------
        list of dicts
            Each entry has keys ``"point"``, ``"value"``, ``"grad_norm"``,
            and ``"hessian_eigenvalues"``.
        """
        info = []
        for pt in trajectory:
            val = self.value_at(pt)
            grad = self.gradient_at(pt)
            H = self.hessian_at(pt)
            eigvals = np.linalg.eigvalsh(H)
            info.append(
                {
                    "point": np.asarray(pt),
                    "value": val,
                    "grad_norm": float(np.linalg.norm(grad)),
                    "hessian_eigenvalues": eigvals,
                }
            )
        return info

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a multi-line string summarising the function's symbolic properties."""
        lines = [
            f"Function : {self.name}",
            f"Variables: {self.symbols}",
            f"Expression:\n  {self.expr}",
            "",
            "Gradient:",
        ]
        for s, g in zip(self.symbols, self.gradient_expr()):
            lines.append(f"  ∂f/∂{s} = {g}")
        lines += ["", "Hessian (symbolic):"]
        H = self.hessian_expr()
        for i in range(self._dim):
            row = "  [" + ",  ".join(str(H[i, j]) for j in range(self._dim)) + "]"
            lines.append(row)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SymbolicAnalyzer(name={self.name!r}, dim={self._dim})"
