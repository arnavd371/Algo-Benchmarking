"""
Demo: Algorithmic Benchmarking in Non-Convex Optimization Landscapes.

Runs a suite of experiments and produces:
  1. A 2-D landscape heatmap with optimization trajectories.
  2. Convergence curves for each optimizer.
  3. A bar-chart comparison of final objective values.
  4. A Hessian eigenvalue spectrum plot.
  5. Symbolic analysis printed to stdout.

Figures are saved to the current directory as PNG files.
"""

import os
import sys

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to files
import matplotlib.pyplot as plt

from benchmarking import (
    get_function,
    get_optimizer,
    BenchmarkRunner,
    SymbolicAnalyzer,
    LandscapeGeometry,
)
from benchmarking.visualization import (
    plot_landscape_2d,
    plot_convergence,
    plot_optimizer_comparison,
    plot_hessian_spectrum,
    plot_benchmark_summary,
)


# ---------------------------------------------------------------------------
# 1. Symbolic analysis of the Himmelblau function
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Symbolic Analysis – Himmelblau Function")
print("=" * 60)

sa = SymbolicAnalyzer.from_himmelblau()
print(sa.summary())
print()

print("Critical points and their classification:")
cpts = sa.find_critical_points()
for pt in cpts:
    cls = sa.classify_critical_point(pt)
    # Skip complex solutions (symbolic solver may return them)
    try:
        xy = [complex(pt.get(sa.symbols[0], 0)), complex(pt.get(sa.symbols[1], 0))]
        if any(abs(v.imag) > 1e-8 for v in xy):
            continue
        print(f"  x={xy[0].real:.4f}, y={xy[1].real:.4f}  →  {cls}")
    except Exception:
        continue
print()


# ---------------------------------------------------------------------------
# 2. Optimize multiple algorithms on Rastrigin (2-D)
# ---------------------------------------------------------------------------

print("=" * 60)
print("2. Running optimizers on 2-D Rastrigin")
print("=" * 60)

rng = np.random.default_rng(42)
func_r = get_function("rastrigin", dim=2)

optimizers_cfg = [
    ("gradient_descent", {"lr": 1e-3, "max_iter": 5000}),
    ("adam",             {"lr": 5e-2, "max_iter": 5000}),
    ("rmsprop",          {"lr": 5e-2, "max_iter": 5000}),
    ("lbfgs",            {"max_iter": 500}),
]

x0 = rng.uniform(-3, 3, size=2)
print(f"Starting point: {x0}")
print()

results = {}
trajectories = []
labels = []
for name, kwargs in optimizers_cfg:
    opt = get_optimizer(name, **kwargs)
    res = opt.optimize(func_r, func_r.gradient, x0.copy())
    results[name] = res
    trajectories.append(res.trajectory)
    labels.append(name)
    print(f"  {name:20s}  f={res.fun:.6f}  iters={res.nit:5d}  converged={res.success}")

print()


# ---------------------------------------------------------------------------
# 3. Landscape heatmap with trajectories
# ---------------------------------------------------------------------------

print("3. Plotting Rastrigin landscape with trajectories …")

fig, ax = plot_landscape_2d(
    func_r,
    trajectories=trajectories,
    trajectory_labels=labels,
    title="Rastrigin landscape – optimizer trajectories",
)
fig.savefig("rastrigin_landscape.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("   Saved: rastrigin_landscape.png")


# ---------------------------------------------------------------------------
# 4. Convergence curves
# ---------------------------------------------------------------------------

print("4. Plotting convergence curves …")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_convergence(
    list(results.values()),
    labels=list(results.keys()),
    metric="values",
    log_y=True,
    ax=axes[0],
    title="Function value vs. iteration",
)
plot_convergence(
    list(results.values()),
    labels=list(results.keys()),
    metric="grad_norms",
    log_y=True,
    ax=axes[1],
    title="Gradient norm vs. iteration",
)
fig.tight_layout()
fig.savefig("convergence_curves.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("   Saved: convergence_curves.png")


# ---------------------------------------------------------------------------
# 5. Full benchmark suite: multiple functions × multiple optimizers
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("5. Full benchmark suite (3 functions × 4 optimizers × 5 trials)")
print("=" * 60)

runner = BenchmarkRunner(default_lr=5e-2, default_max_iter=2000)
for fname in ["sphere", "rastrigin", "rosenbrock"]:
    runner.add_function(fname, dim=2)
for oname in ["gradient_descent", "sgd_momentum", "adam", "lbfgs"]:
    runner.add_optimizer(oname)

bench_result = runner.run(n_trials=5, seed=0)
print(bench_result.summary())
print()
print("Best optimizer per function:")
for fname, oname in bench_result.best_optimizer_per_function().items():
    print(f"  {fname}: {oname}")
print()

fig_summary = plot_benchmark_summary(bench_result, figsize=(14, 4))
fig_summary.savefig("benchmark_summary.png", dpi=120, bbox_inches="tight")
plt.close(fig_summary)
print("   Saved: benchmark_summary.png")


# ---------------------------------------------------------------------------
# 6. Hessian spectrum
# ---------------------------------------------------------------------------

print()
print("6. Sampling Hessian eigenvalue spectrum of Rastrigin …")

geom = LandscapeGeometry(func_r)
eigvals = geom.hessian_spectrum(n_samples=150, seed=0)
fig, ax = plot_hessian_spectrum(
    eigvals,
    title="Hessian eigenvalue spectrum – Rastrigin (2-D)",
)
fig.savefig("hessian_spectrum.png", dpi=120, bbox_inches="tight")
plt.close(fig)
print("   Saved: hessian_spectrum.png")


# ---------------------------------------------------------------------------
# 7. Symbolic trajectory annotation
# ---------------------------------------------------------------------------

print()
print("7. Symbolic trajectory annotation (Adam on Sphere) …")

sa_sphere = SymbolicAnalyzer.from_sphere(dim=2)
func_s = get_function("sphere", dim=2)
opt_adam = get_optimizer("adam", lr=0.1, max_iter=50)
res_sphere = opt_adam.optimize(func_s, func_s.gradient, np.array([2.0, -3.0]))

info = sa_sphere.convergence_trajectory_info(res_sphere.trajectory[:5])
for i, entry in enumerate(info):
    print(
        f"  iter {i:2d}  f={entry['value']:.6f}  "
        f"||∇f||={entry['grad_norm']:.4f}  "
        f"λ_min={entry['hessian_eigenvalues'].min():.4f}  "
        f"λ_max={entry['hessian_eigenvalues'].max():.4f}"
    )

print()
print("Demo complete.  Output files written to the current directory.")
