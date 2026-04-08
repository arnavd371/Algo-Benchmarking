/**
 * optimizers.js — Gradient-based optimisation algorithms (JavaScript port)
 *
 * Implements: Gradient Descent (with Armijo backtracking), SGD + Momentum,
 * Adam, and RMSprop — mirroring the Python benchmarking.optimizers module.
 *
 * Each optimizer function has the signature:
 *   optimize(f, x0, options) → OptimizeResult
 *
 * where OptimizeResult = {
 *   x          : number[]   – final point
 *   fun        : number     – final function value
 *   trajectory : number[][] – all iterates, shape (T+1, 2)
 *   values     : number[]   – function value at each iterate
 *   gradNorms  : number[]   – gradient norm at each iterate
 *   nit        : number     – number of iterations taken
 *   success    : boolean    – true if gradient norm fell below tol
 * }
 */

'use strict';

// ---------------------------------------------------------------------------
// Per-optimizer colour palette (shared with app.js)
// ---------------------------------------------------------------------------

const OPT_COLORS = {
  gradient_descent: '#e74c3c',
  sgd_momentum:     '#3498db',
  adam:             '#27ae60',
  rmsprop:          '#e67e22',
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

const OPTIMIZERS = {
  gradient_descent: {
    name: 'Gradient Descent',
    description: 'Vanilla gradient descent with Armijo backtracking line search.',
    defaultLr: 0.01,
    optimize: gdOptimize,
  },
  sgd_momentum: {
    name: 'SGD + Momentum',
    description: 'Gradient descent with Nesterov momentum for acceleration.',
    defaultLr: 0.01,
    optimize: momentumOptimize,
  },
  adam: {
    name: 'Adam',
    description: 'Adaptive moment estimation (Kingma & Ba, 2015).',
    defaultLr: 0.05,
    optimize: adamOptimize,
  },
  rmsprop: {
    name: 'RMSprop',
    description: 'Root-mean-square propagation with per-parameter adaptive rates.',
    defaultLr: 0.05,
    optimize: rmspropOptimize,
  },
};

const OPTIMIZER_KEYS = ['gradient_descent', 'sgd_momentum', 'adam', 'rmsprop'];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _norm(v) {
  return Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0));
}

function _buildResult(f, x, trajectory, values, gradNorms, tol) {
  const lastGrad = gradNorms.length ? gradNorms[gradNorms.length - 1] : Infinity;
  return {
    x,
    fun: f(x),
    trajectory,
    values,
    gradNorms,
    nit: trajectory.length - 1,
    success: lastGrad < tol,
  };
}

// ---------------------------------------------------------------------------
// Gradient Descent with Armijo backtracking line search
// ---------------------------------------------------------------------------

function gdOptimize(f, x0, { lr = 0.01, maxIter = 2000, tol = 1e-6 } = {}) {
  let x = x0.slice();
  const trajectory = [x.slice()];
  const values = [f(x)];
  const gradNorms = [];

  for (let iter = 0; iter < maxIter; iter++) {
    const g = gradient(f, x);
    const gnorm = _norm(g);
    gradNorms.push(gnorm);
    if (gnorm < tol) break;

    // Armijo backtracking
    let alpha = lr;
    const f0 = values[values.length - 1];
    const slope = g.reduce((s, gi) => s + gi * gi, 0); // ||g||^2
    for (let k = 0; k < 40; k++) {
      const xTry = x.map((xi, i) => xi - alpha * g[i]);
      if (f(xTry) <= f0 - 0.5 * alpha * slope) break;
      alpha *= 0.5;
      if (alpha < 1e-14) break;
    }

    x = x.map((xi, i) => xi - alpha * g[i]);
    trajectory.push(x.slice());
    values.push(f(x));
  }

  return _buildResult(f, x, trajectory, values, gradNorms, tol);
}

// ---------------------------------------------------------------------------
// SGD with Nesterov Momentum
// ---------------------------------------------------------------------------

function momentumOptimize(f, x0, { lr = 0.01, momentum = 0.9, maxIter = 2000, tol = 1e-6 } = {}) {
  let x = x0.slice();
  let vel = new Array(x.length).fill(0);
  const trajectory = [x.slice()];
  const values = [f(x)];
  const gradNorms = [];

  for (let iter = 0; iter < maxIter; iter++) {
    // Nesterov look-ahead point
    const xLook = x.map((xi, i) => xi - momentum * vel[i]);
    const g = gradient(f, xLook);
    const gnorm = _norm(gradient(f, x)); // actual gradient norm at x
    gradNorms.push(gnorm);
    if (gnorm < tol) break;

    vel = vel.map((vi, i) => momentum * vi + lr * g[i]);
    x = x.map((xi, i) => xi - vel[i]);
    trajectory.push(x.slice());
    values.push(f(x));
  }

  return _buildResult(f, x, trajectory, values, gradNorms, tol);
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

function adamOptimize(f, x0, { lr = 0.05, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, maxIter = 2000, tol = 1e-6 } = {}) {
  let x = x0.slice();
  let m = new Array(x.length).fill(0);
  let v = new Array(x.length).fill(0);
  const trajectory = [x.slice()];
  const values = [f(x)];
  const gradNorms = [];

  for (let t = 1; t <= maxIter; t++) {
    const g = gradient(f, x);
    const gnorm = _norm(g);
    gradNorms.push(gnorm);
    if (gnorm < tol) break;

    m = m.map((mi, i) => beta1 * mi + (1 - beta1) * g[i]);
    v = v.map((vi, i) => beta2 * vi + (1 - beta2) * g[i] * g[i]);

    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    x = x.map((xi, i) => {
      const mhat = m[i] / bc1;
      const vhat = v[i] / bc2;
      return xi - lr * mhat / (Math.sqrt(vhat) + eps);
    });

    trajectory.push(x.slice());
    values.push(f(x));
  }

  return _buildResult(f, x, trajectory, values, gradNorms, tol);
}

// ---------------------------------------------------------------------------
// RMSprop
// ---------------------------------------------------------------------------

function rmspropOptimize(f, x0, { lr = 0.05, decay = 0.99, eps = 1e-8, maxIter = 2000, tol = 1e-6 } = {}) {
  let x = x0.slice();
  let s = new Array(x.length).fill(0);
  const trajectory = [x.slice()];
  const values = [f(x)];
  const gradNorms = [];

  for (let iter = 0; iter < maxIter; iter++) {
    const g = gradient(f, x);
    const gnorm = _norm(g);
    gradNorms.push(gnorm);
    if (gnorm < tol) break;

    s = s.map((si, i) => decay * si + (1 - decay) * g[i] * g[i]);
    x = x.map((xi, i) => xi - lr * g[i] / (Math.sqrt(s[i]) + eps));
    trajectory.push(x.slice());
    values.push(f(x));
  }

  return _buildResult(f, x, trajectory, values, gradNorms, tol);
}
