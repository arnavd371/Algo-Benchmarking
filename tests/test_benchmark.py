"""Tests for benchmarking.benchmark (BenchmarkRunner + BenchmarkResult)."""

import numpy as np
import pytest
import pandas as pd

from benchmarking.benchmark import BenchmarkRunner, BenchmarkResult, TrialResult
from benchmarking.functions import Sphere, Rastrigin
from benchmarking.optimizers import Adam, GradientDescent


class TestBenchmarkRunner:
    def _make_runner(self):
        runner = BenchmarkRunner(default_lr=0.1, default_max_iter=500)
        runner.add_function("sphere", dim=2)
        runner.add_optimizer("adam")
        return runner

    def test_run_returns_benchmark_result(self):
        runner = self._make_runner()
        result = runner.run(n_trials=3, seed=0)
        assert isinstance(result, BenchmarkResult)

    def test_correct_number_of_trials(self):
        runner = self._make_runner()
        result = runner.run(n_trials=5, seed=1)
        assert len(result) == 5  # 1 function × 1 optimizer × 5 trials

    def test_multi_function_multi_optimizer(self):
        runner = BenchmarkRunner(default_lr=0.1, default_max_iter=300)
        runner.add_function("sphere", dim=2)
        runner.add_function("rosenbrock", dim=2)
        runner.add_optimizer("adam")
        runner.add_optimizer("lbfgs")
        result = runner.run(n_trials=2, seed=0)
        # 2 functions × 2 optimizers × 2 trials = 8
        assert len(result) == 8

    def test_no_functions_raises(self):
        runner = BenchmarkRunner()
        runner.add_optimizer("adam")
        with pytest.raises(ValueError, match="No functions"):
            runner.run(n_trials=1)

    def test_no_optimizers_raises(self):
        runner = BenchmarkRunner()
        runner.add_function("sphere")
        with pytest.raises(ValueError, match="No optimizers"):
            runner.run(n_trials=1)

    def test_add_function_instance(self):
        runner = BenchmarkRunner()
        runner.add_function(Sphere(dim=3))
        runner.add_optimizer("adam")
        result = runner.run(n_trials=2, seed=0)
        assert len(result) == 2

    def test_add_optimizer_instance(self):
        runner = BenchmarkRunner()
        runner.add_function("sphere")
        runner.add_optimizer(Adam(lr=0.1, max_iter=200))
        result = runner.run(n_trials=2, seed=0)
        assert len(result) == 2

    def test_invalid_function_type_raises(self):
        runner = BenchmarkRunner()
        with pytest.raises(TypeError):
            runner.add_function(123)

    def test_invalid_optimizer_type_raises(self):
        runner = BenchmarkRunner()
        with pytest.raises(TypeError):
            runner.add_optimizer(123)


class TestBenchmarkResult:
    def _make_result(self):
        runner = BenchmarkRunner(default_lr=0.1, default_max_iter=300)
        runner.add_function("sphere", dim=2)
        runner.add_function("rastrigin", dim=2)
        runner.add_optimizer("adam")
        runner.add_optimizer("gradient_descent")
        return runner.run(n_trials=5, seed=42)

    def test_as_dataframe_shape(self):
        result = self._make_result()
        df = result.as_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20  # 2 funcs × 2 opts × 5 trials

    def test_as_dataframe_columns(self):
        result = self._make_result()
        df = result.as_dataframe()
        expected = {"function", "optimizer", "trial", "final_value",
                    "n_iterations", "n_func_evals", "converged", "elapsed_s"}
        assert expected.issubset(set(df.columns))

    def test_summary_str(self):
        result = self._make_result()
        s = result.summary()
        assert "sphere" in s.lower() or "Sphere" in s
        assert "adam" in s.lower() or "Adam" in s

    def test_best_optimizer_per_function(self):
        result = self._make_result()
        best = result.best_optimizer_per_function()
        assert "Sphere" in best or "sphere" in {k.lower() for k in best}

    def test_convergence_trajectories(self):
        result = self._make_result()
        trajs = result.convergence_trajectories("Sphere", "Adam")
        assert len(trajs) == 5
        for t in trajs:
            assert len(t) >= 1

    def test_repr(self):
        result = self._make_result()
        r = repr(result)
        assert "BenchmarkResult" in r


class TestDefaultSuite:
    def test_default_suite_runs(self):
        runner = BenchmarkRunner.default_suite(dim=2)
        result = runner.run(n_trials=1, seed=0)
        df = result.as_dataframe()
        # 7 functions (6 scalable + himmelblau) × 5 optimizers × 1 trial = 35
        assert len(df) == 35
