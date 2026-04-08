/**
 * app.js — Main application: wires UI controls to functions, optimizers, and Plotly plots.
 *
 * Tabs:
 *   1. Home         – introduction
 *   2. Landscape    – 2-D contour heatmap for any benchmark function
 *   3. Optimise     – run ≥1 optimizer from a chosen start point; show trajectories + convergence
 *   4. Compare      – run all selected optimizers from the same start; bar-chart comparison
 *   5. Spectrum     – Hessian eigenvalue histogram
 */

'use strict';

// ---------------------------------------------------------------------------
// Plotly layout defaults
// ---------------------------------------------------------------------------

const PLOT_BG = '#ffffff';
const PAPER_BG = '#f8f9fa';
const FONT_FAMILY = "'Segoe UI', system-ui, sans-serif";

function baseLayout(title, xLabel = 'x₁', yLabel = 'x₂') {
  return {
    title: { text: title, font: { size: 15 } },
    xaxis: { title: xLabel, gridcolor: '#e0e0e0' },
    yaxis: { title: yLabel, gridcolor: '#e0e0e0' },
    plot_bgcolor: PLOT_BG,
    paper_bgcolor: PAPER_BG,
    font: { family: FONT_FAMILY, size: 12 },
    margin: { l: 55, r: 20, t: 45, b: 50 },
    legend: { bgcolor: 'rgba(255,255,255,0.8)', bordercolor: '#ccc', borderwidth: 1 },
  };
}

const PLOTLY_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Build a Plotly contour trace from a grid object. */
function contourTrace(grid, logScale, colorscale = 'Viridis') {
  let z = grid.z;
  if (logScale) {
    const flat = z.flat();
    const minZ = Math.min(...flat);
    z = z.map(row => row.map(v => Math.log1p(Math.max(0, v - minZ))));
  }
  return {
    type: 'contour',
    x: grid.xs,
    y: grid.ys,
    z,
    colorscale,
    contours: { coloring: 'heatmap', showlabels: false },
    ncontours: 25,
    showscale: true,
    colorbar: { thickness: 14, len: 0.9 },
    hovertemplate: 'x₁=%{x:.3f}<br>x₂=%{y:.3f}<br>f=%{z:.4f}<extra></extra>',
  };
}

/** Build a scatter trace for a trajectory. */
function trajectoryTrace(traj, color, name) {
  return {
    type: 'scatter',
    x: traj.map(p => p[0]),
    y: traj.map(p => p[1]),
    mode: 'lines+markers',
    line: { color, width: 2 },
    marker: { size: 4, color },
    name,
    hovertemplate: 'x₁=%{x:.3f}<br>x₂=%{y:.3f}<extra>' + name + '</extra>',
  };
}

/** Scatter trace for start or global-min marker. */
function markerTrace(x, y, color, symbol, name, size = 14) {
  return {
    type: 'scatter',
    x: [x], y: [y],
    mode: 'markers',
    marker: { symbol, color, size, line: { color: '#fff', width: 1.5 } },
    name,
    hovertemplate: `${name}<br>(%{x:.3f}, %{y:.3f})<extra></extra>`,
  };
}

/** Format a number for display (3–4 significant figures). */
function fmt(n) {
  if (n === null || n === undefined || !isFinite(n)) return 'N/A';
  return Math.abs(n) < 1e-4 || Math.abs(n) > 1e6 ? n.toExponential(3) : n.toPrecision(4);
}

// ---------------------------------------------------------------------------
// ① LANDSCAPE TAB
// ---------------------------------------------------------------------------

function initLandscapeTab() {
  const funcSel = document.getElementById('land-func');
  const gridSlider = document.getElementById('land-grid');
  const gridVal = document.getElementById('land-grid-val');
  const logCheck = document.getElementById('land-log');
  const markCheck = document.getElementById('land-mark-min');

  // Populate function selector
  FUNCTION_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = FUNCTIONS[k].name;
    funcSel.appendChild(opt);
  });

  gridSlider.addEventListener('input', () => { gridVal.textContent = gridSlider.value; });

  document.getElementById('land-plot-btn').addEventListener('click', plotLandscape);
  funcSel.addEventListener('change', updateLandscapeInfo);

  updateLandscapeInfo();
  plotLandscape();
}

function updateLandscapeInfo() {
  const key = document.getElementById('land-func').value;
  const f = FUNCTIONS[key];
  document.getElementById('land-func-name').textContent = f.name;
  document.getElementById('land-func-desc').textContent = f.description;
  document.getElementById('land-func-formula').textContent = f.formula;
  document.getElementById('land-bounds').textContent = `[${f.bounds[0]}, ${f.bounds[1]}]`;
  document.getElementById('land-global-min').textContent = f.globalMin;
  document.getElementById('land-global-min-loc').textContent = `(${f.globalMinLoc.map(v => v.toFixed(4)).join(', ')})`;
}

function plotLandscape() {
  const key = document.getElementById('land-func').value;
  const nGrid = parseInt(document.getElementById('land-grid').value, 10);
  const logScale = document.getElementById('land-log').checked;
  const markMin = document.getElementById('land-mark-min').checked;

  const funcDef = FUNCTIONS[key];
  const [lo, hi] = funcDef.bounds;
  const grid = buildGrid(funcDef.fn, lo, hi, nGrid);

  const traces = [contourTrace(grid, logScale)];
  if (markMin) {
    traces.push(markerTrace(funcDef.globalMinLoc[0], funcDef.globalMinLoc[1], '#ff4136', 'star', 'Global min', 16));
  }

  const layout = baseLayout(`${funcDef.name} landscape${logScale ? ' (log scale)' : ''}`);
  layout.xaxis.range = [lo, hi];
  layout.yaxis.range = [lo, hi];

  Plotly.react('landscape-plot', traces, layout, PLOTLY_CONFIG);
  updateLandscapeInfo();
}

// ---------------------------------------------------------------------------
// ② OPTIMISE TAB
// ---------------------------------------------------------------------------

function initOptimiseTab() {
  // Populate function selector
  const funcSel = document.getElementById('opt-func');
  FUNCTION_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = FUNCTIONS[k].name;
    funcSel.appendChild(opt);
  });

  // Sliders
  ['opt-x1', 'opt-x2'].forEach(id => {
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(id + '-val');
    slider.addEventListener('input', () => {
      valSpan.textContent = parseFloat(slider.value).toFixed(2);
      updateOptStartMarker();
    });
  });

  document.getElementById('opt-lr').addEventListener('input', function () {
    document.getElementById('opt-lr-val').textContent = parseFloat(this.value).toFixed(4);
  });

  document.getElementById('opt-func').addEventListener('change', onOptFuncChange);
  document.getElementById('opt-run-btn').addEventListener('click', runOptimise);

  onOptFuncChange();
}

function onOptFuncChange() {
  const key = document.getElementById('opt-func').value;
  const [lo, hi] = FUNCTIONS[key].bounds;
  const mid = ((lo + hi) / 2) * 0.4; // start off-centre

  ['opt-x1', 'opt-x2'].forEach(id => {
    const slider = document.getElementById(id);
    slider.min = lo;
    slider.max = hi;
    slider.step = (hi - lo) / 200;
    slider.value = mid;
    document.getElementById(id + '-val').textContent = parseFloat(mid).toFixed(2);
  });

  // Empty landscape (no trajectory yet)
  drawOptLandscape([]);
}

function drawOptLandscape(results) {
  const key = document.getElementById('opt-func').value;
  const logScale = document.getElementById('opt-log').checked;
  const funcDef = FUNCTIONS[key];
  const [lo, hi] = funcDef.bounds;
  const grid = buildGrid(funcDef.fn, lo, hi, 100);

  const traces = [contourTrace(grid, logScale)];
  traces.push(markerTrace(funcDef.globalMinLoc[0], funcDef.globalMinLoc[1], '#ff4136', 'star', 'Global min', 16));

  // Start marker
  const x1 = parseFloat(document.getElementById('opt-x1').value);
  const x2 = parseFloat(document.getElementById('opt-x2').value);
  traces.push(markerTrace(x1, x2, '#f1c40f', 'square', 'Start', 12));

  // Trajectories
  results.forEach(({ result, optKey }) => {
    traces.push(trajectoryTrace(result.trajectory, OPT_COLORS[optKey], OPTIMIZERS[optKey].name));
    // End point
    const last = result.trajectory[result.trajectory.length - 1];
    traces.push(markerTrace(last[0], last[1], OPT_COLORS[optKey], 'circle', OPTIMIZERS[optKey].name + ' end', 10));
  });

  const layout = baseLayout(`${funcDef.name} — trajectories`);
  layout.xaxis.range = [lo, hi];
  layout.yaxis.range = [lo, hi];

  Plotly.react('opt-landscape', traces, layout, PLOTLY_CONFIG);

  // Allow clicking on landscape to set start point
  const div = document.getElementById('opt-landscape');
  div.removeAllListeners && div.removeAllListeners('plotly_click');
  div.on('plotly_click', data => {
    if (!data.points.length) return;
    const pt = data.points[0];
    // Only respond to clicks on the contour (trace 0)
    if (pt.curveNumber !== 0) return;
    const nx = pt.x, ny = pt.y;
    document.getElementById('opt-x1').value = nx;
    document.getElementById('opt-x2').value = ny;
    document.getElementById('opt-x1-val').textContent = nx.toFixed(2);
    document.getElementById('opt-x2-val').textContent = ny.toFixed(2);
    updateOptStartMarker();
  });
}

function updateOptStartMarker() {
  // Re-draw landscape without re-running optimizer
  drawOptLandscape([]);
}

function runOptimise() {
  const key = document.getElementById('opt-func').value;
  const funcDef = FUNCTIONS[key];
  const x1 = parseFloat(document.getElementById('opt-x1').value);
  const x2 = parseFloat(document.getElementById('opt-x2').value);
  const lr = parseFloat(document.getElementById('opt-lr').value);
  const maxIter = parseInt(document.getElementById('opt-maxiter').value, 10);

  // Which optimizers are selected?
  const selectedOpts = OPTIMIZER_KEYS.filter(k =>
    document.getElementById('opt-check-' + k) && document.getElementById('opt-check-' + k).checked
  );

  if (selectedOpts.length === 0) {
    alert('Please select at least one optimiser.');
    return;
  }

  const btn = document.getElementById('opt-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Running…';

  // Run in a timeout to allow the spinner to render
  setTimeout(() => {
    const results = [];
    selectedOpts.forEach(optKey => {
      const optDef = OPTIMIZERS[optKey];
      const result = optDef.optimize(funcDef.fn, [x1, x2], { lr, maxIter });
      results.push({ result, optKey });
    });

    // Landscape with trajectories
    drawOptLandscape(results);

    // Convergence plots
    plotConvergence('opt-conv-values', results, 'Function value', 'values', true);
    plotConvergence('opt-conv-grads', results, 'Gradient norm', 'gradNorms', true);
    document.getElementById('opt-conv-row').classList.remove('d-none');

    // Results table
    buildResultsTable('opt-results-body', results);
    document.getElementById('opt-results-card').classList.remove('d-none');

    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-play-circle me-1"></i>Run';
  }, 20);
}

function plotConvergence(divId, results, yLabel, field, logY) {
  const traces = results.map(({ result, optKey }) => ({
    type: 'scatter',
    x: Array.from({ length: result[field].length }, (_, i) => i),
    y: result[field],
    mode: 'lines',
    name: OPTIMIZERS[optKey].name,
    line: { color: OPT_COLORS[optKey], width: 2 },
  }));

  const layout = {
    ...baseLayout(yLabel + ' vs iteration', 'Iteration', yLabel),
    yaxis: {
      title: yLabel,
      type: logY ? 'log' : 'linear',
      gridcolor: '#e0e0e0',
    },
    height: 300,
    margin: { l: 60, r: 20, t: 40, b: 45 },
  };

  Plotly.react(divId, traces, layout, PLOTLY_CONFIG);
}

function buildResultsTable(tbodyId, results) {
  const tbody = document.getElementById(tbodyId);
  tbody.innerHTML = '';
  results.forEach(({ result, optKey }) => {
    const tr = document.createElement('tr');
    const convergedBadge = result.success
      ? '<span class="badge bg-success">Yes</span>'
      : '<span class="badge bg-warning text-dark">No</span>';
    tr.innerHTML = `
      <td><span class="dot" style="background:${OPT_COLORS[optKey]}"></span> ${OPTIMIZERS[optKey].name}</td>
      <td class="text-end font-mono">${fmt(result.fun)}</td>
      <td class="text-end">${result.nit}</td>
      <td class="text-center">${convergedBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ---------------------------------------------------------------------------
// ③ COMPARE TAB
// ---------------------------------------------------------------------------

function initCompareTab() {
  const funcSel = document.getElementById('cmp-func');
  FUNCTION_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = FUNCTIONS[k].name;
    funcSel.appendChild(opt);
  });

  ['cmp-x1', 'cmp-x2'].forEach(id => {
    const slider = document.getElementById(id);
    const valSpan = document.getElementById(id + '-val');
    slider.addEventListener('input', () => {
      valSpan.textContent = parseFloat(slider.value).toFixed(2);
    });
  });

  document.getElementById('cmp-lr').addEventListener('input', function () {
    document.getElementById('cmp-lr-val').textContent = parseFloat(this.value).toFixed(4);
  });

  document.getElementById('cmp-func').addEventListener('change', onCmpFuncChange);
  document.getElementById('cmp-run-btn').addEventListener('click', runCompare);

  onCmpFuncChange();
}

function onCmpFuncChange() {
  const key = document.getElementById('cmp-func').value;
  const [lo, hi] = FUNCTIONS[key].bounds;
  const mid = ((lo + hi) / 2) * 0.4;

  ['cmp-x1', 'cmp-x2'].forEach(id => {
    const slider = document.getElementById(id);
    slider.min = lo; slider.max = hi;
    slider.step = (hi - lo) / 200;
    slider.value = mid;
    document.getElementById(id + '-val').textContent = parseFloat(mid).toFixed(2);
  });
}

function runCompare() {
  const key = document.getElementById('cmp-func').value;
  const funcDef = FUNCTIONS[key];
  const x1 = parseFloat(document.getElementById('cmp-x1').value);
  const x2 = parseFloat(document.getElementById('cmp-x2').value);
  const lr = parseFloat(document.getElementById('cmp-lr').value);
  const maxIter = parseInt(document.getElementById('cmp-maxiter').value, 10);

  const selectedOpts = OPTIMIZER_KEYS.filter(k =>
    document.getElementById('cmp-check-' + k) && document.getElementById('cmp-check-' + k).checked
  );

  if (selectedOpts.length === 0) {
    alert('Please select at least one optimiser.');
    return;
  }

  const btn = document.getElementById('cmp-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Running…';

  setTimeout(() => {
    const results = [];
    selectedOpts.forEach(optKey => {
      const result = OPTIMIZERS[optKey].optimize(funcDef.fn, [x1, x2], { lr, maxIter });
      results.push({ result, optKey });
    });

    // Landscape with all trajectories
    const [lo, hi] = funcDef.bounds;
    const logScale = document.getElementById('cmp-log').checked;
    const grid = buildGrid(funcDef.fn, lo, hi, 100);
    const traces = [contourTrace(grid, logScale)];
    traces.push(markerTrace(funcDef.globalMinLoc[0], funcDef.globalMinLoc[1], '#ff4136', 'star', 'Global min', 16));
    traces.push(markerTrace(x1, x2, '#f1c40f', 'square', 'Start', 12));

    results.forEach(({ result, optKey }) => {
      traces.push(trajectoryTrace(result.trajectory, OPT_COLORS[optKey], OPTIMIZERS[optKey].name));
      const last = result.trajectory[result.trajectory.length - 1];
      traces.push(markerTrace(last[0], last[1], OPT_COLORS[optKey], 'circle', OPTIMIZERS[optKey].name + ' end', 10));
    });

    const mapLayout = baseLayout(`${funcDef.name} — all trajectories`);
    mapLayout.xaxis.range = [lo, hi];
    mapLayout.yaxis.range = [lo, hi];
    Plotly.react('cmp-landscape', traces, mapLayout, PLOTLY_CONFIG);

    // Convergence curves (both metrics side-by-side)
    plotConvergence('cmp-conv-values', results, 'Function value', 'values', true);
    plotConvergence('cmp-conv-grads', results, 'Gradient norm', 'gradNorms', true);

    // Bar chart — final function values
    const barTraces = [{
      type: 'bar',
      x: results.map(({ optKey }) => OPTIMIZERS[optKey].name),
      y: results.map(({ result }) => result.fun),
      marker: { color: results.map(({ optKey }) => OPT_COLORS[optKey]) },
      text: results.map(({ result }) => fmt(result.fun)),
      textposition: 'outside',
    }];
    const barLayout = {
      ...baseLayout('Final f(x) by optimiser', 'Optimiser', 'Final f(x)'),
      yaxis: { title: 'Final f(x)', type: 'log', gridcolor: '#e0e0e0' },
      showlegend: false,
      height: 320,
      margin: { l: 60, r: 20, t: 40, b: 80 },
    };
    Plotly.react('cmp-bar', barTraces, barLayout, PLOTLY_CONFIG);

    // Iterations bar chart
    const iterTraces = [{
      type: 'bar',
      x: results.map(({ optKey }) => OPTIMIZERS[optKey].name),
      y: results.map(({ result }) => result.nit),
      marker: { color: results.map(({ optKey }) => OPT_COLORS[optKey]) },
      text: results.map(({ result }) => result.nit),
      textposition: 'outside',
    }];
    const iterLayout = {
      ...baseLayout('Iterations to convergence', 'Optimiser', 'Iterations'),
      showlegend: false,
      height: 320,
      margin: { l: 60, r: 20, t: 40, b: 80 },
    };
    Plotly.react('cmp-iter-bar', iterTraces, iterLayout, PLOTLY_CONFIG);

    // Results table
    buildResultsTable('cmp-results-body', results);

    document.getElementById('cmp-results-section').classList.remove('d-none');

    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-bar-chart me-1"></i>Compare';
  }, 20);
}

// ---------------------------------------------------------------------------
// ④ HESSIAN SPECTRUM TAB
// ---------------------------------------------------------------------------

function initSpectrumTab() {
  const funcSel = document.getElementById('spec-func');
  FUNCTION_KEYS.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = FUNCTIONS[k].name;
    funcSel.appendChild(opt);
  });

  document.getElementById('spec-samples').addEventListener('input', function () {
    document.getElementById('spec-samples-val').textContent = this.value;
  });

  document.getElementById('spec-run-btn').addEventListener('click', runSpectrum);
}

function runSpectrum() {
  const key = document.getElementById('spec-func').value;
  const funcDef = FUNCTIONS[key];
  const nSamples = parseInt(document.getElementById('spec-samples').value, 10);
  const bins = parseInt(document.getElementById('spec-bins').value, 10);

  const btn = document.getElementById('spec-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Computing…';

  setTimeout(() => {
    const [lo, hi] = funcDef.bounds;
    const allEigvals = [];

    for (let i = 0; i < nSamples; i++) {
      const x = [lo + Math.random() * (hi - lo), lo + Math.random() * (hi - lo)];
      const H = hessian2D(funcDef.fn, x);
      const eigs = eigenvalues2D(H);
      allEigvals.push(...eigs);
    }

    // Histogram
    const trace = {
      type: 'histogram',
      x: allEigvals,
      nbinsx: bins,
      marker: { color: '#3498db', opacity: 0.8 },
      name: 'Eigenvalues',
    };
    const zeroLine = {
      type: 'scatter',
      x: [0, 0],
      y: [0, null], // will be overridden by yref='paper'
      mode: 'lines',
      line: { color: '#e74c3c', width: 2, dash: 'dash' },
      name: 'λ = 0',
      yaxis: 'y',
    };
    const layout = {
      ...baseLayout(`Hessian eigenvalue spectrum — ${funcDef.name}`, 'Eigenvalue (λ)', 'Count'),
      shapes: [{
        type: 'line',
        x0: 0, x1: 0,
        y0: 0, y1: 1,
        yref: 'paper',
        line: { color: '#e74c3c', width: 2, dash: 'dash' },
      }],
      annotations: [{
        x: 0, y: 1, xref: 'x', yref: 'paper',
        text: 'λ = 0',
        showarrow: false,
        font: { color: '#e74c3c', size: 11 },
        xanchor: 'left', yanchor: 'top',
        xshift: 5,
      }],
      height: 380,
    };

    Plotly.react('spec-plot', [trace], layout, PLOTLY_CONFIG);

    // Statistics
    const nPos = allEigvals.filter(v => v > 1e-8).length;
    const nNeg = allEigvals.filter(v => v < -1e-8).length;
    const nZero = allEigvals.length - nPos - nNeg;
    const minV = Math.min(...allEigvals);
    const maxV = Math.max(...allEigvals);
    const meanV = allEigvals.reduce((s, v) => s + v, 0) / allEigvals.length;

    document.getElementById('spec-stats').innerHTML = `
      <div class="row g-2 text-center">
        <div class="col-6 col-md-3">
          <div class="card border-0 bg-light py-2">
            <div class="fs-4 fw-bold text-success">${nPos}</div>
            <div class="small text-muted">Positive<br><span class="text-success">convex directions</span></div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="card border-0 bg-light py-2">
            <div class="fs-4 fw-bold text-danger">${nNeg}</div>
            <div class="small text-muted">Negative<br><span class="text-danger">concave / saddle</span></div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="card border-0 bg-light py-2">
            <div class="fs-4 fw-bold text-secondary">${nZero}</div>
            <div class="small text-muted">Near-zero<br><span class="text-secondary">flat regions</span></div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="card border-0 bg-light py-2">
            <div class="fs-5 fw-bold">${(100 * nNeg / allEigvals.length).toFixed(1)}%</div>
            <div class="small text-muted">Saddle-point<br>density</div>
          </div>
        </div>
      </div>
      <div class="mt-3 small text-muted">
        Min: <b>${fmt(minV)}</b> &nbsp;|&nbsp; Max: <b>${fmt(maxV)}</b> &nbsp;|&nbsp; Mean: <b>${fmt(meanV)}</b>
        &nbsp;|&nbsp; Total eigenvalues: <b>${allEigvals.length}</b>
      </div>
    `;
    document.getElementById('spec-stats-card').classList.remove('d-none');

    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-activity me-1"></i>Compute spectrum';
  }, 20);
}

// ---------------------------------------------------------------------------
// Bootstrap tab activation callbacks (lazy-init plots)
// ---------------------------------------------------------------------------

const _tabInited = {};

document.addEventListener('DOMContentLoaded', () => {
  initLandscapeTab();
  initOptimiseTab();
  initCompareTab();
  initSpectrumTab();

  // Resize plots when a tab becomes visible (Plotly needs this)
  document.querySelectorAll('[data-bs-toggle="tab"]').forEach(btn => {
    btn.addEventListener('shown.bs.tab', () => {
      window.dispatchEvent(new Event('resize'));
    });
  });
});
