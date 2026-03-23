"""
Algo-Benchmarking Web App
=========================

Interactive Streamlit application for exploring non-convex optimization
landscapes and benchmarking optimisation algorithms.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from benchmarking import (
    get_function,
    get_optimizer,
    BenchmarkRunner,
    LandscapeGeometry,
)
from benchmarking.visualization import (
    plot_landscape_2d,
    plot_convergence,
    plot_hessian_spectrum,
    plot_benchmark_summary,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Algo-Benchmarking",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

SECTIONS = [
    "🏠 Introduction",
    "🗺️ Function Explorer",
    "🚀 Optimisation",
    "📊 Benchmark Suite",
    "🔬 Hessian Spectrum",
]

st.sidebar.title("Algo-Benchmarking")
st.sidebar.caption("Non-Convex Optimisation Visualiser")
section = st.sidebar.radio("Navigate to", SECTIONS)

FUNCTION_NAMES = [
    "rastrigin", "ackley", "rosenbrock", "himmelblau",
    "levy", "schwefel", "griewank", "sphere",
]

OPTIMIZER_NAMES = [
    "gradient_descent", "sgd_momentum", "adam", "rmsprop", "lbfgs",
]

FUNCTION_DESCRIPTIONS = {
    "rastrigin":   "Highly multimodal — many evenly-spaced local minima. Global min: **f(0,…,0) = 0**.",
    "ackley":      "Nearly flat outer region with a deep central pit. Global min: **f(0,…,0) = 0**.",
    "rosenbrock":  "Curved, narrow banana-shaped valley. Global min: **f(1,…,1) = 0**.",
    "himmelblau":  "Four symmetric local minima of equal value (2-D only). All minima = **0**.",
    "levy":        "Multimodal with several local minima. Global min: **f(1,…,1) = 0**.",
    "schwefel":    "Deceptive: global minimum lies far from next-best local minima. Global min ≈ **0**.",
    "griewank":    "Product term introduces widespread local minima. Global min: **f(0,…,0) = 0**.",
    "sphere":      "Canonical convex baseline — single bowl-shaped minimum. Global min: **f(0,…,0) = 0**.",
}

OPTIMIZER_DESCRIPTIONS = {
    "gradient_descent": "Vanilla gradient descent with optional Armijo backtracking line search.",
    "sgd_momentum":     "SGD with Nesterov-style momentum for acceleration.",
    "adam":             "Adaptive moment estimation (Kingma & Ba, 2015).",
    "rmsprop":          "Root-mean-square propagation — per-parameter adaptive learning rates.",
    "lbfgs":            "Quasi-Newton L-BFGS method (wraps SciPy).",
}

# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Running optimiser…")
def run_optimizer(func_name: str, opt_name: str, x0: list, lr: float, max_iter: int) -> dict:
    func = get_function(func_name, dim=2)
    kwargs: dict = {"max_iter": max_iter}
    if opt_name != "lbfgs":
        kwargs["lr"] = lr
    opt = get_optimizer(opt_name, **kwargs)
    res = opt.optimize(func, func.gradient, np.array(x0))
    return {
        "fun": res.fun,
        "nit": res.nit,
        "success": res.success,
        "trajectory": res.trajectory.tolist(),
        "values": res.values,
        "grad_norms": res.grad_norms,
        "time": res.time,
        "nfev": res.nfev,
    }


@st.cache_data(show_spinner="Running benchmark suite…")
def run_benchmark(func_names: tuple, opt_names: tuple, n_trials: int, seed: int, lr: float, max_iter: int):
    runner = BenchmarkRunner(default_lr=lr, default_max_iter=max_iter)
    for fname in func_names:
        runner.add_function(fname, dim=2)
    for oname in opt_names:
        runner.add_optimizer(oname)
    result = runner.run(n_trials=n_trials, seed=seed)
    return result


@st.cache_data(show_spinner="Sampling Hessian spectrum…")
def compute_hessian_spectrum(func_name: str, n_samples: int, seed: int) -> np.ndarray:
    func = get_function(func_name, dim=2)
    geom = LandscapeGeometry(func)
    return geom.hessian_spectrum(n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# Helper: close matplotlib figures
# ---------------------------------------------------------------------------

def show_fig(fig: plt.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ===========================================================================
# Section: Introduction
# ===========================================================================

if section == SECTIONS[0]:
    st.title("📈 Algo-Benchmarking")
    st.subheader("Interactive Non-Convex Optimisation Landscape Visualiser")

    st.markdown(
        """
        This web app lets you **explore non-convex optimisation landscapes** and
        compare how different gradient-based algorithms navigate them.

        ### What you can do

        | Section | Description |
        |---|---|
        | 🗺️ **Function Explorer** | Visualise any of the 8 benchmark functions as a 2-D heatmap |
        | 🚀 **Optimisation** | Run one or more algorithms from a custom starting point and watch their trajectories |
        | 📊 **Benchmark Suite** | Grid-sweep a set of functions × optimisers, compare results with bar charts |
        | 🔬 **Hessian Spectrum** | Sample Hessian eigenvalues across the landscape to measure curvature distribution |

        ### Available benchmark functions

        | Function | Characteristic |
        |---|---|
        | **Rastrigin** | Highly multimodal — evenly-spaced local minima |
        | **Ackley** | Nearly flat outer region with a deep central pit |
        | **Rosenbrock** | Curved, narrow banana-shaped valley |
        | **Himmelblau** | Four symmetric local minima (2-D only) |
        | **Levy** | Multimodal with several local minima |
        | **Schwefel** | Deceptive: global min far from next-best local minima |
        | **Griewank** | Widespread local minima from product term |
        | **Sphere** | Convex baseline — single bowl-shaped minimum |

        ### Available optimisers
        **Gradient Descent**, **SGD with Momentum**, **Adam**, **RMSprop**, **L-BFGS**

        ---
        Use the **sidebar** to navigate between sections.
        """
    )

# ===========================================================================
# Section: Function Explorer
# ===========================================================================

elif section == SECTIONS[1]:
    st.title("🗺️ Function Explorer")
    st.markdown("Explore the 2-D landscape of any benchmark function.")

    col_ctrl, col_plot = st.columns([1, 2])

    with col_ctrl:
        func_name = st.selectbox("Benchmark function", FUNCTION_NAMES, index=0,
                                 format_func=str.capitalize)
        st.caption(FUNCTION_DESCRIPTIONS[func_name])
        log_scale = st.checkbox("Logarithmic colour scale", value=False)
        n_grid = st.slider("Grid resolution", min_value=50, max_value=300, value=150, step=25)
        mark_min = st.checkbox("Mark global minimum", value=True)

        func = get_function(func_name, dim=2)
        lo, hi = func.default_bounds
        st.markdown(f"**Default bounds:** [{lo}, {hi}]")
        st.markdown(f"**Global min value:** {func.global_min_value}")

    with col_plot:
        func = get_function(func_name, dim=2)
        fig, ax = plot_landscape_2d(
            func,
            n_grid=n_grid,
            log_scale=log_scale,
            mark_global_min=mark_min,
            title=f"{func_name.capitalize()} landscape",
        )
        show_fig(fig)

    # Function expression info
    with st.expander("ℹ️ Function description"):
        desc_map = {
            "rastrigin":  r"$f(\mathbf{x}) = An + \sum_{i=1}^{n}\left[x_i^2 - A\cos(2\pi x_i)\right],\quad A=10$",
            "ackley":     r"$f(\mathbf{x}) = -20\exp\!\left(-0.2\sqrt{\tfrac{1}{n}\sum x_i^2}\right) - \exp\!\left(\tfrac{1}{n}\sum\cos(2\pi x_i)\right) + 20 + e$",
            "rosenbrock": r"$f(\mathbf{x}) = \sum_{i=1}^{n-1}\left[100(x_{i+1}-x_i^2)^2+(1-x_i)^2\right]$",
            "himmelblau": r"$f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2$",
            "levy":       r"$f(\mathbf{x}) = \sin^2(\pi w_1) + \sum_{i=1}^{n-1}(w_i-1)^2[1+10\sin^2(\pi w_i+1)] + (w_n-1)^2[1+\sin^2(2\pi w_n)]$",
            "schwefel":   r"$f(\mathbf{x}) = 418.9829\,n - \sum_{i=1}^{n} x_i\sin\!\left(\sqrt{|x_i|}\right)$",
            "griewank":   r"$f(\mathbf{x}) = \tfrac{1}{4000}\sum x_i^2 - \prod\cos\!\left(\tfrac{x_i}{\sqrt{i}}\right) + 1$",
            "sphere":     r"$f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2$",
        }
        st.latex(desc_map.get(func_name, ""))

# ===========================================================================
# Section: Optimisation
# ===========================================================================

elif section == SECTIONS[2]:
    st.title("🚀 Optimisation")
    st.markdown(
        "Run one or more optimisers from a chosen starting point and "
        "visualise their trajectories and convergence curves."
    )

    # --- Controls ---
    c1, c2 = st.columns(2)
    with c1:
        func_name = st.selectbox("Benchmark function", FUNCTION_NAMES, index=0,
                                 format_func=str.capitalize)
        st.caption(FUNCTION_DESCRIPTIONS[func_name])

    with c2:
        selected_opts = st.multiselect(
            "Optimisers", OPTIMIZER_NAMES,
            default=["adam", "gradient_descent"],
            format_func=lambda s: s.replace("_", " ").title(),
        )

    func = get_function(func_name, dim=2)
    lo, hi = func.default_bounds

    with st.expander("⚙️ Advanced settings", expanded=False):
        ec1, ec2 = st.columns(2)
        with ec1:
            lr = st.number_input("Learning rate (ignored for L-BFGS)", min_value=1e-5,
                                  max_value=1.0, value=0.05, step=0.005, format="%.4f")
            max_iter = st.number_input("Max iterations", min_value=100, max_value=10000,
                                        value=2000, step=100)
        with ec2:
            x0_x = st.slider("Start x₁", min_value=float(lo), max_value=float(hi),
                               value=float(np.clip(2.0, lo, hi)), step=0.1)
            x0_y = st.slider("Start x₂", min_value=float(lo), max_value=float(hi),
                               value=float(np.clip(-1.5, lo, hi)), step=0.1)

    run_btn = st.button("▶ Run optimisation", type="primary")

    if not selected_opts:
        st.info("Select at least one optimiser above.")
        st.stop()

    if run_btn or "opt_results" in st.session_state:
        if run_btn:
            # Clear cached results when user clicks run
            st.session_state.pop("opt_results", None)
            results_data = {}
            for opt_name in selected_opts:
                results_data[opt_name] = run_optimizer(
                    func_name, opt_name, [x0_x, x0_y],
                    lr, int(max_iter),
                )
            st.session_state["opt_results"] = results_data
            st.session_state["opt_func"] = func_name
            st.session_state["opt_x0"] = [x0_x, x0_y]

        results_data = st.session_state.get("opt_results", {})
        used_func = st.session_state.get("opt_func", func_name)
        func = get_function(used_func, dim=2)

        # --- Results table ---
        st.subheader("Results")
        rows = []
        for opt_name, r in results_data.items():
            rows.append({
                "Optimiser": opt_name.replace("_", " ").title(),
                "Final f(x)": f"{r['fun']:.6g}",
                "Iterations": r["nit"],
                "Converged": "✅" if r["success"] else "❌",
                "Time (s)": f"{r['time']:.3f}",
                "Function evals": r["nfev"],
            })
        st.table(rows)

        # --- Landscape + trajectories ---
        trajectories = [np.array(r["trajectory"]) for r in results_data.values()]
        labels = [n.replace("_", " ").title() for n in results_data]

        col_land, col_conv = st.columns(2)
        with col_land:
            st.subheader("Landscape & trajectories")
            log_scale = st.checkbox("Log colour scale", value=False, key="opt_log")
            fig, ax = plot_landscape_2d(
                func,
                trajectories=trajectories,
                trajectory_labels=labels,
                log_scale=log_scale,
                title=f"{used_func.capitalize()} – trajectories",
            )
            show_fig(fig)

        with col_conv:
            st.subheader("Convergence curves")
            from benchmarking.optimizers import OptimizeResult

            # Reconstruct lightweight OptimizeResult-like objects for plotting
            class _FakeResult:
                def __init__(self, values, grad_norms):
                    self.values = values
                    self.grad_norms = grad_norms

            fake_results = [_FakeResult(r["values"], r["grad_norms"]) for r in results_data.values()]

            fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
            plot_convergence(fake_results, labels=labels, metric="values",
                             log_y=True, ax=axes[0], title="Function value")
            plot_convergence(fake_results, labels=labels, metric="grad_norms",
                             log_y=True, ax=axes[1], title="Gradient norm")
            fig2.tight_layout()
            show_fig(fig2)

# ===========================================================================
# Section: Benchmark Suite
# ===========================================================================

elif section == SECTIONS[3]:
    st.title("📊 Benchmark Suite")
    st.markdown(
        "Run a grid sweep across multiple functions and optimisers, "
        "then compare results with bar charts."
    )

    c1, c2 = st.columns(2)
    with c1:
        sel_funcs = st.multiselect(
            "Functions to benchmark", FUNCTION_NAMES,
            default=["sphere", "rastrigin", "rosenbrock"],
            format_func=str.capitalize,
        )
    with c2:
        sel_opts = st.multiselect(
            "Optimisers to benchmark", OPTIMIZER_NAMES,
            default=["gradient_descent", "adam", "lbfgs"],
            format_func=lambda s: s.replace("_", " ").title(),
        )

    with st.expander("⚙️ Settings"):
        bc1, bc2 = st.columns(2)
        with bc1:
            n_trials = st.slider("Trials per (function, optimiser)", 1, 20, 5)
            seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        with bc2:
            b_lr = st.number_input("Learning rate (non-L-BFGS)", min_value=1e-4, max_value=1.0,
                                    value=0.05, step=0.005, format="%.4f")
            b_max_iter = st.number_input("Max iterations", min_value=100, max_value=5000,
                                          value=1000, step=100)

    if not sel_funcs or not sel_opts:
        st.info("Select at least one function and one optimiser.")
        st.stop()

    run_bench = st.button("▶ Run benchmark", type="primary")

    if run_bench:
        st.session_state.pop("bench_result", None)

    if run_bench or "bench_result" in st.session_state:
        if "bench_result" not in st.session_state or run_bench:
            result = run_benchmark(
                tuple(sel_funcs), tuple(sel_opts),
                int(n_trials), int(seed),
                float(b_lr), int(b_max_iter),
            )
            st.session_state["bench_result"] = result

        result = st.session_state["bench_result"]

        st.subheader("Summary")
        st.text(result.summary())

        st.subheader("Best optimiser per function")
        best = result.best_optimizer_per_function()
        best_rows = [{"Function": fn.capitalize(), "Best optimiser": on.replace("_", " ").title()}
                     for fn, on in best.items()]
        st.table(best_rows)

        st.subheader("Final value comparison")
        fig = plot_benchmark_summary(result, figsize=(max(8, len(sel_funcs) * 4), 5))
        show_fig(fig)

        st.subheader("Raw data")
        df = result.as_dataframe()
        st.dataframe(df, use_container_width=True)

# ===========================================================================
# Section: Hessian Spectrum
# ===========================================================================

elif section == SECTIONS[4]:
    st.title("🔬 Hessian Spectrum")
    st.markdown(
        "Sample the Hessian eigenvalue distribution across the landscape "
        "to measure local curvature and identify saddle-point density."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        func_name = st.selectbox("Benchmark function", FUNCTION_NAMES, index=0,
                                 format_func=str.capitalize)
        st.caption(FUNCTION_DESCRIPTIONS[func_name])
        n_samples = st.slider("Number of sample points", 50, 500, 150, step=25)
        h_seed = st.number_input("Seed", min_value=0, max_value=9999, value=0, step=1)
        bins = st.slider("Histogram bins", 20, 120, 60, step=10)
        run_h = st.button("▶ Compute spectrum", type="primary")

    with c2:
        if run_h:
            st.session_state.pop("hess_eigvals", None)
            st.session_state.pop("hess_func", None)

        if run_h or "hess_eigvals" in st.session_state:
            if "hess_eigvals" not in st.session_state or run_h:
                eigvals = compute_hessian_spectrum(func_name, int(n_samples), int(h_seed))
                st.session_state["hess_eigvals"] = eigvals
                st.session_state["hess_func"] = func_name

            eigvals = st.session_state["hess_eigvals"]
            used_func = st.session_state.get("hess_func", func_name)

            fig, ax = plot_hessian_spectrum(
                eigvals,
                bins=bins,
                title=f"Hessian eigenvalue spectrum – {used_func.capitalize()}",
            )
            show_fig(fig)

            # Statistics
            n_pos = int(np.sum(eigvals > 0))
            n_neg = int(np.sum(eigvals < 0))
            n_zero = int(np.sum(np.abs(eigvals) < 1e-8))

            st.markdown("**Eigenvalue statistics**")
            st.markdown(
                f"- Total eigenvalues sampled: **{len(eigvals)}**\n"
                f"- Positive (convex directions): **{n_pos}** ({100*n_pos/len(eigvals):.1f}%)\n"
                f"- Negative (concave directions): **{n_neg}** ({100*n_neg/len(eigvals):.1f}%)\n"
                f"- Near-zero: **{n_zero}**\n"
                f"- Min: **{eigvals.min():.4f}** | Max: **{eigvals.max():.4f}** | "
                f"Mean: **{eigvals.mean():.4f}**"
            )

            with st.expander("ℹ️ What does this mean?"):
                st.markdown(
                    """
                    The **Hessian** at a point describes the local curvature of the function.
                    Its **eigenvalues** tell you about the principal curvature directions:

                    - **All positive** → local minimum (convex region)
                    - **All negative** → local maximum
                    - **Mixed signs** → **saddle point** (common in deep-learning landscapes)
                    - **Near-zero** → flat region; gradient-based methods slow down

                    Highly non-convex functions like **Rastrigin** or **Schwefel** will show
                    a wide distribution with many negative eigenvalues and saddle points,
                    while the **Sphere** will show only positive eigenvalues.
                    """
                )
        else:
            st.info("Configure settings on the left and click **Compute spectrum**.")
