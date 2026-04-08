/**
 * functions.js — Benchmark function library (JavaScript port)
 *
 * Implements the same 8 non-convex benchmark functions as the Python library,
 * together with finite-difference gradient / Hessian helpers and a grid builder
 * used by the Plotly visualisations.
 */

'use strict';

// ---------------------------------------------------------------------------
// Function registry
// ---------------------------------------------------------------------------

const FUNCTIONS = {
  rastrigin: {
    name: 'Rastrigin',
    bounds: [-5.12, 5.12],
    globalMin: 0.0,
    globalMinLoc: [0, 0],
    description: 'Highly multimodal — many evenly-spaced local minima surrounding the global minimum at the origin.',
    formula: 'f(x) = 10n + Σ [ xᵢ² − 10 cos(2π xᵢ) ]',
    fn(x) {
      const A = 10;
      return A * x.length + x.reduce((s, xi) => s + xi * xi - A * Math.cos(2 * Math.PI * xi), 0);
    },
  },

  ackley: {
    name: 'Ackley',
    bounds: [-32.768, 32.768],
    globalMin: 0.0,
    globalMinLoc: [0, 0],
    description: 'Nearly flat outer region with a deep, narrow central pit — difficult for gradient-based methods far from the origin.',
    formula: 'f(x) = −20 exp(−0.2 √(Σxᵢ²/n)) − exp(Σcos(2πxᵢ)/n) + 20 + e',
    fn(x) {
      const n = x.length;
      const sum1 = Math.sqrt(x.reduce((s, xi) => s + xi * xi, 0) / n);
      const sum2 = x.reduce((s, xi) => s + Math.cos(2 * Math.PI * xi), 0) / n;
      return -20 * Math.exp(-0.2 * sum1) - Math.exp(sum2) + 20 + Math.E;
    },
  },

  rosenbrock: {
    name: 'Rosenbrock',
    bounds: [-5.0, 10.0],
    globalMin: 0.0,
    globalMinLoc: [1, 1],
    description: 'The "banana" function — a curved, narrow valley. Easy to find the valley, but hard to follow it to the minimum.',
    formula: 'f(x) = Σ [ 100 (xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)² ]',
    fn(x) {
      let s = 0;
      for (let i = 0; i < x.length - 1; i++) {
        s += 100 * Math.pow(x[i + 1] - x[i] * x[i], 2) + Math.pow(1 - x[i], 2);
      }
      return s;
    },
  },

  himmelblau: {
    name: 'Himmelblau',
    bounds: [-5.0, 5.0],
    globalMin: 0.0,
    globalMinLoc: [3, 2],
    description: 'Four symmetric local minima of equal value (2-D only). All four minima equal 0; the function is degenerate in the sense that no unique global minimum exists.',
    formula: 'f(x, y) = (x² + y − 11)² + (x + y² − 7)²',
    fn(x) {
      return Math.pow(x[0] * x[0] + x[1] - 11, 2) + Math.pow(x[0] + x[1] * x[1] - 7, 2);
    },
  },

  levy: {
    name: 'Levy',
    bounds: [-10.0, 10.0],
    globalMin: 0.0,
    globalMinLoc: [1, 1],
    description: 'Multimodal function with several local minima. Global minimum at (1, …, 1).',
    formula: 'f(x) = sin²(πw₁) + Σ(wᵢ−1)²[1+10sin²(πwᵢ+1)] + (wₙ−1)²[1+sin²(2πwₙ)], where wᵢ = 1+(xᵢ−1)/4',
    fn(x) {
      const w = x.map(xi => 1 + (xi - 1) / 4);
      const term1 = Math.pow(Math.sin(Math.PI * w[0]), 2);
      let term2 = 0;
      for (let i = 0; i < w.length - 1; i++) {
        term2 += Math.pow(w[i] - 1, 2) * (1 + 10 * Math.pow(Math.sin(Math.PI * w[i] + 1), 2));
      }
      const wn = w[w.length - 1];
      const term3 = Math.pow(wn - 1, 2) * (1 + Math.pow(Math.sin(2 * Math.PI * wn), 2));
      return term1 + term2 + term3;
    },
  },

  schwefel: {
    name: 'Schwefel',
    bounds: [-500.0, 500.0],
    globalMin: 0.0,
    globalMinLoc: [420.9687, 420.9687],
    description: 'Deceptive: the global minimum is geometrically remote from the next-best local minima, making it very hard for local search methods.',
    formula: 'f(x) = 418.9829 n − Σ xᵢ sin(√|xᵢ|)',
    fn(x) {
      return 418.9829 * x.length - x.reduce((s, xi) => s + xi * Math.sin(Math.sqrt(Math.abs(xi))), 0);
    },
  },

  griewank: {
    name: 'Griewank',
    bounds: [-600.0, 600.0],
    globalMin: 0.0,
    globalMinLoc: [0, 0],
    description: 'A product term creates widespread local minima distributed across the entire domain.',
    formula: 'f(x) = Σxᵢ²/4000 − Π cos(xᵢ/√i) + 1',
    fn(x) {
      const sum = x.reduce((s, xi) => s + xi * xi, 0) / 4000;
      const prod = x.reduce((p, xi, i) => p * Math.cos(xi / Math.sqrt(i + 1)), 1);
      return sum - prod + 1;
    },
  },

  sphere: {
    name: 'Sphere',
    bounds: [-5.0, 5.0],
    globalMin: 0.0,
    globalMinLoc: [0, 0],
    description: 'The canonical convex baseline — a smooth bowl shape with a single global minimum at the origin.',
    formula: 'f(x) = Σ xᵢ²',
    fn(x) {
      return x.reduce((s, xi) => s + xi * xi, 0);
    },
  },
};

// Ordered list of function keys for UI display
const FUNCTION_KEYS = ['rastrigin', 'ackley', 'rosenbrock', 'himmelblau', 'levy', 'schwefel', 'griewank', 'sphere'];

// ---------------------------------------------------------------------------
// Calculus helpers
// ---------------------------------------------------------------------------

/**
 * Finite-difference gradient (central differences).
 * @param {Function} f  - scalar function of Array → Number
 * @param {number[]} x  - point
 * @param {number} eps  - step size
 * @returns {number[]}
 */
function gradient(f, x, eps = 1e-5) {
  return x.map((_, i) => {
    const xp = x.slice(); xp[i] += eps;
    const xm = x.slice(); xm[i] -= eps;
    return (f(xp) - f(xm)) / (2 * eps);
  });
}

/**
 * Finite-difference Hessian for a 2-D function.
 * Returns a 2×2 array [[h00, h01], [h10, h11]].
 * @param {Function} f
 * @param {number[]} x  - length-2 point
 * @param {number} eps
 * @returns {number[][]}
 */
function hessian2D(f, x, eps = 1e-4) {
  const H = [[0, 0], [0, 0]];
  for (let i = 0; i < 2; i++) {
    for (let j = i; j < 2; j++) {
      const xpp = x.slice(); xpp[i] += eps; xpp[j] += eps;
      const xmm = x.slice(); xmm[i] -= eps; xmm[j] -= eps;
      const xpm = x.slice(); xpm[i] += eps; xpm[j] -= eps;
      const xmp = x.slice(); xmp[i] -= eps; xmp[j] += eps;
      const val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps);
      H[i][j] = val;
      H[j][i] = val;
    }
  }
  return H;
}

/**
 * Eigenvalues of a 2×2 symmetric matrix via the quadratic formula.
 * @param {number[][]} H  - 2×2 matrix
 * @returns {number[]} [lambda1, lambda2] (lambda1 >= lambda2)
 */
function eigenvalues2D(H) {
  const a = H[0][0], b = H[0][1], d = H[1][1];
  const trace = a + d;
  const disc = Math.sqrt(Math.max(0, (a - d) * (a - d) + 4 * b * b));
  return [(trace + disc) / 2, (trace - disc) / 2];
}

// ---------------------------------------------------------------------------
// Grid builder for Plotly contour / heatmap plots
// ---------------------------------------------------------------------------

/**
 * Evaluate fn on an nGrid×nGrid grid over [lo, hi]².
 * Returns { xs, ys, z } compatible with Plotly contour traces.
 * @param {Function} fn
 * @param {number} lo
 * @param {number} hi
 * @param {number} nGrid
 * @returns {{ xs: number[], ys: number[], z: number[][] }}
 */
function buildGrid(fn, lo, hi, nGrid = 120) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < nGrid; i++) {
    xs.push(lo + (i / (nGrid - 1)) * (hi - lo));
    ys.push(lo + (i / (nGrid - 1)) * (hi - lo));
  }
  const z = ys.map(y => xs.map(x => fn([x, y])));
  return { xs, ys, z };
}
