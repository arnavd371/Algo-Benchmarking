# Algo-Benchmarking

**Project 1 – Algorithmic Benchmarking in Non-Convex Optimization Landscapes**

Analyzes the geometry of high-dimensional non-convex optimization landscapes using
symbolic computation to track convergence trajectories.

---

## Features

| Module | What it provides |
|---|---|
| `benchmarking.functions` | 8 standard non-convex benchmark functions (Rastrigin, Ackley, Rosenbrock, Himmelblau, Levy, Schwefel, Griewank, Sphere) with finite-difference gradient & Hessian |
| `benchmarking.symbolic` | SymPy-based exact gradients, Hessians, critical-point detection and classification, trajectory annotation |
| `benchmarking.geometry` | Hessian spectrum sampling, landscape statistics (smoothness, ruggedness), saddle-point density estimation, curvature along trajectories |
| `benchmarking.optimizers` | Gradient Descent (± Armijo line search), SGD with Nesterov momentum, Adam, RMSprop, L-BFGS |
| `benchmarking.benchmark` | Grid-runner: sweep any set of functions × optimizers × random starts; returns a tidy DataFrame of metrics |
| `benchmarking.visualization` | 2-D heatmaps with trajectory overlays, convergence curves, bar-chart comparisons, Hessian eigenvalue histograms |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`, `pytest`

---

## Quick Start

```python
from benchmarking import BenchmarkRunner

# Run all built-in functions × all built-in optimizers, 5 trials each
runner = BenchmarkRunner.default_suite(dim=2)
results = runner.run(n_trials=5, seed=42)
print(results.summary())
print(results.best_optimizer_per_function())
```

### Symbolic landscape analysis

```python
from benchmarking import SymbolicAnalyzer

sa = SymbolicAnalyzer.from_himmelblau()
print(sa.summary())

for pt in sa.find_critical_points():
    print(pt, "→", sa.classify_critical_point(pt))
```

### Trajectory with geometry info

```python
import numpy as np
from benchmarking import get_function, get_optimizer, LandscapeGeometry

func = get_function("sphere", dim=2)
opt  = get_optimizer("adam", lr=0.1, max_iter=200)
res  = opt.optimize(func, func.gradient, np.array([3.0, -2.0]))

geom = LandscapeGeometry(func)
curvs = geom.curvature_along_trajectory(res.trajectory)
print("Max curvature along trajectory:", curvs.max())
```

### Visualisation

```python
from benchmarking import get_function, get_optimizer
from benchmarking.visualization import plot_landscape_2d
import numpy as np, matplotlib.pyplot as plt

func = get_function("rastrigin", dim=2)
opt  = get_optimizer("adam", lr=0.05, max_iter=500)
res  = opt.optimize(func, func.gradient, np.array([2.0, -1.5]))

fig, ax = plot_landscape_2d(func, trajectories=[res.trajectory], trajectory_labels=["Adam"])
plt.show()
```

---

## Running the Demo

```bash
python examples/demo.py
```

Produces four PNG plots in the current directory:
- `rastrigin_landscape.png` – 2-D landscape with optimizer trajectories
- `convergence_curves.png` – function value & gradient norm vs. iteration
- `benchmark_summary.png` – per-optimizer bar chart across functions
- `hessian_spectrum.png` – histogram of landscape Hessian eigenvalues

---

## Running Tests

```bash
pytest tests/ -q
```

---

## Project Structure

```
Algo-Benchmarking/
├── benchmarking/
│   ├── __init__.py
│   ├── functions.py       # Benchmark function library
│   ├── symbolic.py        # SymPy-based symbolic analysis
│   ├── geometry.py        # Landscape geometry & statistics
│   ├── optimizers.py      # Optimization algorithms
│   ├── benchmark.py       # Experiment runner
│   └── visualization.py   # Plotting utilities
├── tests/
│   ├── test_functions.py
│   ├── test_optimizers.py
│   ├── test_symbolic.py
│   ├── test_geometry.py
│   └── test_benchmark.py
├── examples/
│   └── demo.py
└── requirements.txt
```
