#!/usr/bin/env python
"""Generate self-contained HTML annotation tool for reclassifying Msln+ cells into tiers.

Loads detection JSON, extracts cell positions/SNR/tier data, and produces an
interactive HTML with:
- Pannable/zoomable canvas spatial map (colored dots by tier)
- Click-to-select with info panel
- Lasso/brush multi-select
- Tier reassignment buttons (including "Remove" for false positives)
- Live summary bar with counts
- Sidebar table sorted by SNR (descending), click-to-pan
- Export button to download annotations JSON

Usage:
    python scripts/generate_msln_annotation.py

    # Custom input/output:
    python scripts/generate_msln_annotation.py \\
        --input /path/to/detections_msln_bg_subtracted.json \\
        --output /path/to/msln_annotation.html
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_INPUT = (
    '/fs/pool/pool-mann-edwin/psilo_output/tp_full/'
    '20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct/'
    'detections_msln_bg_subtracted.json'
)
DEFAULT_OUTPUT = '/fs/pool/pool-mann-edwin/brain_fish_output/msln_annotation.html'


def main():
    parser = argparse.ArgumentParser(description='Generate Msln annotation HTML')
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help='Path to detections_msln_bg_subtracted.json')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Output HTML path')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f'ERROR: Input not found: {input_path}', file=sys.stderr)
        sys.exit(1)

    print(f'Loading {input_path} ...')
    with open(input_path) as f:
        detections = json.load(f)

    print(f'  {len(detections)} detections loaded')

    # Extract cell data
    uids = []
    xs = []
    ys = []
    snrs = []
    bg_subs = []
    tiers = []

    for det in detections:
        uid = det.get('uid', '')
        pos = det.get('global_center_um')
        if pos is None or len(pos) != 2:
            continue
        feat = det.get('features', {})
        msln_tier = feat.get('msln_tier', '')
        msln_snr = feat.get('msln_snr', 0.0)
        msln_bg_sub = feat.get('msln_bg_sub', 0.0)

        if not np.isfinite(pos[0]) or not np.isfinite(pos[1]):
            continue

        uids.append(uid)
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        snrs.append(float(msln_snr))
        bg_subs.append(float(msln_bg_sub))
        tiers.append(msln_tier)

    n = len(uids)
    print(f'  {n} valid cells extracted')

    # Tier encoding: 0=Msln+(2-3x), 1=Msln++(3-4x), 2=Msln+++(>4x)
    tier_map = {
        'Msln+ (2-3x)': 0,
        'Msln++ (3-4x)': 1,
        'Msln+++ (>4x)': 2,
    }
    tier_indices = [tier_map.get(t, 0) for t in tiers]

    # Serialize positions as Float32Array (compact comma-separated)
    x_str = ','.join(f'{v:.1f}' for v in xs)
    y_str = ','.join(f'{v:.1f}' for v in ys)
    snr_str = ','.join(f'{v:.2f}' for v in snrs)
    bg_str = ','.join(f'{v:.1f}' for v in bg_subs)
    tier_str = ','.join(str(t) for t in tier_indices)
    uid_json = json.dumps(uids)

    # Sort indices by SNR descending for sidebar table
    sorted_indices = sorted(range(n), key=lambda i: -snrs[i])
    sorted_idx_str = ','.join(str(i) for i in sorted_indices)

    # Counts per tier
    from collections import Counter
    tier_counts = Counter(tier_indices)

    # Spatial extent for initial view
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    print(f'  X range: {x_min:.1f} - {x_max:.1f} um')
    print(f'  Y range: {y_min:.1f} - {y_max:.1f} um')
    print(f'  Tiers: + = {tier_counts.get(0,0)}, ++ = {tier_counts.get(1,0)}, +++ = {tier_counts.get(2,0)}')

    html = _generate_html(
        n=n,
        x_str=x_str, y_str=y_str,
        snr_str=snr_str, bg_str=bg_str,
        tier_str=tier_str,
        uid_json=uid_json,
        sorted_idx_str=sorted_idx_str,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f'  Written {output_path} ({size_mb:.1f} MB)')


def _generate_html(*, n, x_str, y_str, snr_str, bg_str, tier_str,
                   uid_json, sorted_idx_str, x_min, x_max, y_min, y_max):
    """Build the complete self-contained HTML string."""

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Msln+ Tier Annotation</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #1a1a2e; color: #e0e0e0; overflow: hidden; height: 100vh; }}

#top-bar {{
  display: flex; align-items: center; gap: 12px;
  padding: 6px 14px; background: #16213e; border-bottom: 1px solid #333;
  font-size: 13px; flex-wrap: wrap; min-height: 40px;
}}
#top-bar .tier-badge {{
  padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 12px;
}}
#top-bar .title {{ font-size: 15px; font-weight: 700; margin-right: 16px; }}
.tier-plus  {{ background: #4caf50; color: #000; }}
.tier-pplus {{ background: #ff9800; color: #000; }}
.tier-ppplus {{ background: #f44336; color: #fff; }}
.tier-removed {{ background: #666; color: #fff; }}
#save-btn {{
  margin-left: auto; padding: 5px 16px; background: #2196F3; color: #fff;
  border: none; border-radius: 4px; cursor: pointer; font-size: 13px; font-weight: 600;
}}
#save-btn:hover {{ background: #1976D2; }}

#main {{ display: flex; height: calc(100vh - 42px); }}

#canvas-wrap {{
  flex: 1; position: relative; overflow: hidden; background: #0f0f23;
}}
canvas#map {{ display: block; width: 100%; height: 100%; cursor: crosshair; }}

#info-panel {{
  position: absolute; top: 10px; left: 10px;
  background: rgba(22,33,62,0.92); border: 1px solid #444;
  border-radius: 6px; padding: 10px 14px; font-size: 12px;
  display: none; min-width: 240px; z-index: 10;
}}
#info-panel .label {{ color: #888; font-size: 11px; }}
#info-panel .value {{ font-weight: 600; margin-bottom: 4px; }}
#info-panel .tier-btns {{ display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }}
#info-panel .tier-btns button {{
  padding: 4px 10px; border: none; border-radius: 3px; cursor: pointer;
  font-size: 12px; font-weight: 600;
}}
.btn-plus {{ background: #4caf50; color: #000; }}
.btn-pplus {{ background: #ff9800; color: #000; }}
.btn-ppplus {{ background: #f44336; color: #fff; }}
.btn-remove {{ background: #666; color: #fff; }}
.btn-plus:hover {{ background: #66bb6a; }}
.btn-pplus:hover {{ background: #ffb74d; }}
.btn-ppplus:hover {{ background: #ef5350; }}
.btn-remove:hover {{ background: #888; }}

#mode-indicator {{
  position: absolute; bottom: 10px; left: 10px;
  background: rgba(22,33,62,0.85); border: 1px solid #444;
  border-radius: 4px; padding: 5px 12px; font-size: 12px; z-index: 10;
}}

#sidebar {{
  width: 320px; background: #16213e; border-left: 1px solid #333;
  display: flex; flex-direction: column; overflow: hidden;
}}
#sidebar-header {{
  padding: 8px 12px; font-size: 13px; font-weight: 600;
  border-bottom: 1px solid #333; display: flex; align-items: center; gap: 8px;
}}
#sidebar-header input {{
  flex: 1; padding: 4px 8px; background: #0f0f23; border: 1px solid #444;
  border-radius: 3px; color: #e0e0e0; font-size: 12px;
}}
#cell-table-wrap {{
  flex: 1; overflow-y: auto; overflow-x: hidden;
}}
#cell-table {{
  width: 100%; border-collapse: collapse; font-size: 11px;
}}
#cell-table th {{
  position: sticky; top: 0; background: #1a1a3e; padding: 4px 6px;
  text-align: left; border-bottom: 1px solid #444; cursor: pointer;
  user-select: none;
}}
#cell-table th:hover {{ background: #252550; }}
#cell-table td {{
  padding: 3px 6px; border-bottom: 1px solid #222; cursor: pointer;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 120px;
}}
#cell-table tr:hover {{ background: #1e3a5f; }}
#cell-table tr.selected {{ background: #2a4a6f; }}
#cell-table .tier-dot {{
  display: inline-block; width: 10px; height: 10px; border-radius: 50%;
  margin-right: 4px; vertical-align: middle;
}}

/* Lasso overlay */
#lasso-svg {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none; z-index: 5;
}}
#lasso-svg path {{
  fill: rgba(33,150,243,0.15); stroke: #2196F3; stroke-width: 2;
  stroke-dasharray: 6,3;
}}

/* Bulk action bar */
#bulk-bar {{
  position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
  background: rgba(22,33,62,0.95); border: 1px solid #2196F3;
  border-radius: 6px; padding: 8px 16px; font-size: 13px;
  display: none; z-index: 10; text-align: center;
}}
#bulk-bar .tier-btns {{ display: flex; gap: 6px; margin-top: 6px; justify-content: center; }}
#bulk-bar .tier-btns button {{
  padding: 5px 14px; border: none; border-radius: 3px; cursor: pointer;
  font-size: 12px; font-weight: 600;
}}

/* Scrollbar styling */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: #1a1a2e; }}
::-webkit-scrollbar-thumb {{ background: #444; border-radius: 4px; }}
</style>
</head>
<body>

<div id="top-bar">
  <span class="title">Msln+ Tier Annotation</span>
  <span class="tier-badge tier-plus" id="count-plus">+ : 0</span>
  <span class="tier-badge tier-pplus" id="count-pplus">++ : 0</span>
  <span class="tier-badge tier-ppplus" id="count-ppplus">+++ : 0</span>
  <span class="tier-badge tier-removed" id="count-removed">Removed: 0</span>
  <span style="color:#888; font-size:12px;" id="count-changed">0 changed</span>
  <button id="save-btn" onclick="exportAnnotations()">Save Annotations</button>
</div>

<div id="main">
  <div id="canvas-wrap">
    <canvas id="map"></canvas>
    <svg id="lasso-svg"><path id="lasso-path" d=""></path></svg>
    <div id="info-panel">
      <div class="label">UID</div>
      <div class="value" id="info-uid" style="font-size:11px; word-break:break-all;"></div>
      <div class="label">SNR</div>
      <div class="value" id="info-snr"></div>
      <div class="label">BG-Sub Intensity</div>
      <div class="value" id="info-bg"></div>
      <div class="label">Current Tier</div>
      <div class="value" id="info-tier"></div>
      <div style="margin-top:6px; font-size:11px; color:#888;">Assign tier:</div>
      <div class="tier-btns">
        <button class="btn-plus" onclick="assignTier(0)">Msln+</button>
        <button class="btn-pplus" onclick="assignTier(1)">Msln++</button>
        <button class="btn-ppplus" onclick="assignTier(2)">Msln+++</button>
        <button class="btn-remove" onclick="assignTier(3)">Remove</button>
      </div>
    </div>
    <div id="mode-indicator">
      <span id="mode-text">Click: select | Shift+drag: lasso | Scroll: zoom | Drag: pan</span>
    </div>
    <div id="bulk-bar">
      <div id="bulk-count">0 cells selected</div>
      <div class="tier-btns">
        <button class="btn-plus" onclick="bulkAssign(0)">Msln+</button>
        <button class="btn-pplus" onclick="bulkAssign(1)">Msln++</button>
        <button class="btn-ppplus" onclick="bulkAssign(2)">Msln+++</button>
        <button class="btn-remove" onclick="bulkAssign(3)">Remove</button>
      </div>
      <div style="margin-top:4px; font-size:11px; color:#aaa;">Press Escape to deselect</div>
    </div>
  </div>

  <div id="sidebar">
    <div id="sidebar-header">
      <span>Cells (by SNR)</span>
      <input type="text" id="search-box" placeholder="Filter by UID..." oninput="filterTable()">
    </div>
    <div id="cell-table-wrap">
      <table id="cell-table">
        <thead><tr>
          <th onclick="sortTable('tier')" style="width:30px;">T</th>
          <th onclick="sortTable('snr')">SNR</th>
          <th onclick="sortTable('bg')">BG-Sub</th>
          <th onclick="sortTable('uid')">UID (suffix)</th>
        </tr></thead>
        <tbody id="cell-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
// =========================================================================
// Data
// =========================================================================
const N = {n};
const cellX = new Float32Array([{x_str}]);
const cellY = new Float32Array([{y_str}]);
const cellSNR = new Float32Array([{snr_str}]);
const cellBG = new Float32Array([{bg_str}]);
const origTier = new Uint8Array([{tier_str}]);
const curTier = new Uint8Array(origTier);  // mutable copy
const uids = {uid_json};
const snrSortedIdx = new Int32Array([{sorted_idx_str}]);

// Tier labels and colors
const TIER_LABELS = ['Msln+ (2-3x)', 'Msln++ (3-4x)', 'Msln+++ (>4x)', 'Removed'];
const TIER_COLORS = ['#4caf50', '#ff9800', '#f44336', '#666666'];
const TIER_COLORS_DIM = ['#2e7d32', '#e65100', '#c62828', '#444444'];

// Spatial extent
const DATA_X_MIN = {x_min:.1f};
const DATA_X_MAX = {x_max:.1f};
const DATA_Y_MIN = {y_min:.1f};
const DATA_Y_MAX = {y_max:.1f};

// =========================================================================
// Canvas + View State
// =========================================================================
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
let W, H;  // canvas pixel dims
let dpr = window.devicePixelRatio || 1;

// View transform: screen = (data - panX) * zoom, panX/panY in data coords
let zoom = 1;
let panX = DATA_X_MIN;
let panY = DATA_Y_MIN;

// Selection
let selectedIdx = -1;         // single-click selection
let selectedSet = new Set();  // lasso multi-select
let hoveredIdx = -1;

// Interaction state
let isDragging = false;
let isLassoing = false;
let dragStartScreen = null;
let lassoPoints = [];  // screen coords

function resize() {{
  const wrap = document.getElementById('canvas-wrap');
  W = wrap.clientWidth;
  H = wrap.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}

function fitAll() {{
  const pad = 0.05;
  const dx = DATA_X_MAX - DATA_X_MIN;
  const dy = DATA_Y_MAX - DATA_Y_MIN;
  const zx = W / (dx * (1 + 2 * pad));
  const zy = H / (dy * (1 + 2 * pad));
  zoom = Math.min(zx, zy);
  panX = DATA_X_MIN - (W / zoom - dx) / 2;
  panY = DATA_Y_MIN - (H / zoom - dy) / 2;
}}

// Data coord -> screen pixel
function toScreenX(dx) {{ return (dx - panX) * zoom; }}
function toScreenY(dy) {{ return (dy - panY) * zoom; }}
// Screen pixel -> data coord
function toDataX(sx) {{ return sx / zoom + panX; }}
function toDataY(sy) {{ return sy / zoom + panY; }}

// =========================================================================
// Spatial index (grid for fast nearest-neighbor)
// =========================================================================
const GRID_CELL = 200;  // um per grid cell
let gridCols, gridRows, grid;

function buildSpatialGrid() {{
  const gw = DATA_X_MAX - DATA_X_MIN + GRID_CELL;
  const gh = DATA_Y_MAX - DATA_Y_MIN + GRID_CELL;
  gridCols = Math.ceil(gw / GRID_CELL);
  gridRows = Math.ceil(gh / GRID_CELL);
  grid = new Array(gridCols * gridRows);
  for (let i = 0; i < grid.length; i++) grid[i] = [];
  for (let i = 0; i < N; i++) {{
    const gc = Math.floor((cellX[i] - DATA_X_MIN) / GRID_CELL);
    const gr = Math.floor((cellY[i] - DATA_Y_MIN) / GRID_CELL);
    const idx = gr * gridCols + gc;
    if (idx >= 0 && idx < grid.length) grid[idx].push(i);
  }}
}}

function findNearest(dx, dy, maxDist) {{
  // Search grid cells within maxDist
  const gc0 = Math.max(0, Math.floor((dx - maxDist - DATA_X_MIN) / GRID_CELL));
  const gc1 = Math.min(gridCols - 1, Math.floor((dx + maxDist - DATA_X_MIN) / GRID_CELL));
  const gr0 = Math.max(0, Math.floor((dy - maxDist - DATA_Y_MIN) / GRID_CELL));
  const gr1 = Math.min(gridRows - 1, Math.floor((dy + maxDist - DATA_Y_MIN) / GRID_CELL));
  let best = -1, bestD2 = maxDist * maxDist;
  for (let gr = gr0; gr <= gr1; gr++) {{
    for (let gc = gc0; gc <= gc1; gc++) {{
      const bucket = grid[gr * gridCols + gc];
      for (let k = 0; k < bucket.length; k++) {{
        const i = bucket[k];
        const d2 = (cellX[i] - dx) ** 2 + (cellY[i] - dy) ** 2;
        if (d2 < bestD2) {{ bestD2 = d2; best = i; }}
      }}
    }}
  }}
  return best;
}}

function findInLasso(screenPoints) {{
  // Point-in-polygon test for all cells
  const result = [];
  for (let i = 0; i < N; i++) {{
    const sx = toScreenX(cellX[i]);
    const sy = toScreenY(cellY[i]);
    if (pointInPolygon(sx, sy, screenPoints)) {{
      result.push(i);
    }}
  }}
  return result;
}}

function pointInPolygon(px, py, poly) {{
  let inside = false;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {{
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {{
      inside = !inside;
    }}
  }}
  return inside;
}}

// =========================================================================
// Rendering
// =========================================================================
let needsRedraw = true;

function draw() {{
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0f0f23';
  ctx.fillRect(0, 0, W, H);

  // Dot radius in pixels (scales with zoom to stay visible)
  const baseR = Math.max(2, Math.min(6, zoom * 15));

  // Draw all dots
  for (let i = 0; i < N; i++) {{
    const sx = toScreenX(cellX[i]);
    const sy = toScreenY(cellY[i]);
    // Cull off-screen
    if (sx < -10 || sx > W + 10 || sy < -10 || sy > H + 10) continue;

    const isSel = i === selectedIdx || selectedSet.has(i);
    const tier = curTier[i];
    ctx.fillStyle = isSel ? '#fff' : TIER_COLORS[tier];
    ctx.beginPath();
    ctx.arc(sx, sy, isSel ? baseR + 2 : baseR, 0, Math.PI * 2);
    ctx.fill();

    // Selection ring
    if (isSel) {{
      ctx.strokeStyle = TIER_COLORS[tier];
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, baseR + 5, 0, Math.PI * 2);
      ctx.stroke();
    }}
  }}

  // Hover highlight
  if (hoveredIdx >= 0 && hoveredIdx !== selectedIdx && !selectedSet.has(hoveredIdx)) {{
    const sx = toScreenX(cellX[hoveredIdx]);
    const sy = toScreenY(cellY[hoveredIdx]);
    ctx.strokeStyle = '#ffffff88';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, baseR + 3, 0, Math.PI * 2);
    ctx.stroke();
  }}

  // Scale bar
  drawScaleBar();
}}

function drawScaleBar() {{
  // Find nice scale bar length
  const targetPx = 120;
  const targetUm = targetPx / zoom;
  const niceSteps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
  let barUm = niceSteps[0];
  for (const s of niceSteps) {{
    if (Math.abs(s - targetUm) < Math.abs(barUm - targetUm)) barUm = s;
  }}
  const barPx = barUm * zoom;

  const x0 = W - barPx - 20;
  const y0 = H - 20;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x0, y0); ctx.lineTo(x0 + barPx, y0);
  ctx.moveTo(x0, y0 - 5); ctx.lineTo(x0, y0 + 5);
  ctx.moveTo(x0 + barPx, y0 - 5); ctx.lineTo(x0 + barPx, y0 + 5);
  ctx.stroke();

  ctx.fillStyle = '#fff';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(barUm >= 1000 ? (barUm / 1000) + ' mm' : barUm + ' um',
               x0 + barPx / 2, y0 - 8);
}}

function scheduleRedraw() {{
  if (!needsRedraw) {{
    needsRedraw = true;
    requestAnimationFrame(() => {{
      needsRedraw = false;
      draw();
    }});
  }}
}}

// =========================================================================
// Interaction
// =========================================================================

canvas.addEventListener('mousedown', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;

  if (e.shiftKey) {{
    // Start lasso
    isLassoing = true;
    lassoPoints = [[sx, sy]];
    document.getElementById('lasso-path').setAttribute('d', '');
    return;
  }}

  // Start pan
  isDragging = true;
  dragStartScreen = [e.clientX, e.clientY];
  canvas.style.cursor = 'grabbing';
}});

canvas.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;

  if (isLassoing) {{
    lassoPoints.push([sx, sy]);
    // Update SVG path
    let d = 'M ' + lassoPoints[0][0] + ' ' + lassoPoints[0][1];
    for (let i = 1; i < lassoPoints.length; i++) {{
      d += ' L ' + lassoPoints[i][0] + ' ' + lassoPoints[i][1];
    }}
    d += ' Z';
    document.getElementById('lasso-path').setAttribute('d', d);
    return;
  }}

  if (isDragging && dragStartScreen) {{
    const dx = (e.clientX - dragStartScreen[0]) / zoom;
    const dy = (e.clientY - dragStartScreen[1]) / zoom;
    panX -= dx;
    panY -= dy;
    dragStartScreen = [e.clientX, e.clientY];
    scheduleRedraw();
    return;
  }}

  // Hover detection
  const dx = toDataX(sx);
  const dy = toDataY(sy);
  const hitDist = Math.max(20, 50 / zoom);  // um
  const newHover = findNearest(dx, dy, hitDist);
  if (newHover !== hoveredIdx) {{
    hoveredIdx = newHover;
    canvas.style.cursor = newHover >= 0 ? 'pointer' : 'crosshair';
    scheduleRedraw();
  }}
}});

canvas.addEventListener('mouseup', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;

  if (isLassoing) {{
    isLassoing = false;
    document.getElementById('lasso-path').setAttribute('d', '');
    if (lassoPoints.length > 5) {{
      const inside = findInLasso(lassoPoints);
      if (inside.length > 0) {{
        selectedSet = new Set(inside);
        selectedIdx = -1;
        showBulkBar(inside.length);
        hideInfoPanel();
      }}
    }}
    lassoPoints = [];
    scheduleRedraw();
    return;
  }}

  if (isDragging) {{
    const wasDrag = dragStartScreen &&
      (Math.abs(e.clientX - dragStartScreen[0]) > 3 ||
       Math.abs(e.clientY - dragStartScreen[1]) > 3);
    isDragging = false;
    dragStartScreen = null;
    canvas.style.cursor = 'crosshair';

    if (wasDrag) return;  // was pan, not click

    // Click to select
    const dx = toDataX(sx);
    const dy = toDataY(sy);
    const hitDist = Math.max(20, 50 / zoom);
    const hit = findNearest(dx, dy, hitDist);

    if (hit >= 0) {{
      selectCell(hit);
    }} else {{
      deselectAll();
    }}
    scheduleRedraw();
  }}
}});

canvas.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;

  // Zoom centered on mouse position
  const dx = toDataX(sx);
  const dy = toDataY(sy);

  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  zoom *= factor;
  zoom = Math.max(0.001, Math.min(100, zoom));

  // Adjust pan so point under mouse stays fixed
  panX = dx - sx / zoom;
  panY = dy - sy / zoom;

  scheduleRedraw();
}}, {{ passive: false }});

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') {{
    deselectAll();
    scheduleRedraw();
  }}
  if (e.key === 'f' || e.key === 'F') {{
    fitAll();
    scheduleRedraw();
  }}
}});

// =========================================================================
// Selection & Annotation
// =========================================================================

function selectCell(idx) {{
  selectedIdx = idx;
  selectedSet.clear();
  hideBulkBar();
  showInfoPanel(idx);
  highlightTableRow(idx);
  scheduleRedraw();
}}

function deselectAll() {{
  selectedIdx = -1;
  selectedSet.clear();
  hideInfoPanel();
  hideBulkBar();
  clearTableHighlight();
  scheduleRedraw();
}}

function showInfoPanel(idx) {{
  const panel = document.getElementById('info-panel');
  panel.style.display = 'block';
  document.getElementById('info-uid').textContent = uids[idx];
  document.getElementById('info-snr').textContent = cellSNR[idx].toFixed(2);
  document.getElementById('info-bg').textContent = cellBG[idx].toFixed(1);
  document.getElementById('info-tier').textContent = TIER_LABELS[curTier[idx]];
}}

function hideInfoPanel() {{
  document.getElementById('info-panel').style.display = 'none';
}}

function showBulkBar(count) {{
  const bar = document.getElementById('bulk-bar');
  bar.style.display = 'block';
  document.getElementById('bulk-count').textContent = count + ' cells selected';
}}

function hideBulkBar() {{
  document.getElementById('bulk-bar').style.display = 'none';
}}

function assignTier(tier) {{
  if (selectedIdx >= 0) {{
    curTier[selectedIdx] = tier;
    showInfoPanel(selectedIdx);
    updateTableRow(selectedIdx);
    updateCounts();
    scheduleRedraw();
  }}
}}

function bulkAssign(tier) {{
  for (const idx of selectedSet) {{
    curTier[idx] = tier;
    updateTableRow(idx);
  }}
  updateCounts();
  scheduleRedraw();
}}

function updateCounts() {{
  let c = [0, 0, 0, 0];
  let changed = 0;
  for (let i = 0; i < N; i++) {{
    c[curTier[i]]++;
    if (curTier[i] !== origTier[i]) changed++;
  }}
  document.getElementById('count-plus').textContent = '+ : ' + c[0];
  document.getElementById('count-pplus').textContent = '++ : ' + c[1];
  document.getElementById('count-ppplus').textContent = '+++ : ' + c[2];
  document.getElementById('count-removed').textContent = 'Removed: ' + c[3];
  document.getElementById('count-changed').textContent = changed + ' changed';
}}

// =========================================================================
// Export
// =========================================================================

function exportAnnotations() {{
  const result = [];
  for (let i = 0; i < N; i++) {{
    result.push({{
      uid: uids[i],
      msln_tier: TIER_LABELS[curTier[i]],
      msln_snr: parseFloat(cellSNR[i].toFixed(2)),
      msln_bg_sub: parseFloat(cellBG[i].toFixed(1)),
      changed: curTier[i] !== origTier[i],
    }});
  }}
  const blob = new Blob([JSON.stringify(result, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'msln_tier_annotations.json';
  a.click();
  URL.revokeObjectURL(url);
}}

// =========================================================================
// Sidebar Table
// =========================================================================
let tableOrder = snrSortedIdx.slice();  // current display order
let tableFilter = '';
let currentSortCol = 'snr';
let currentSortAsc = false;
const tableRowMap = new Map();  // idx -> tr element

function buildTable() {{
  const tbody = document.getElementById('cell-tbody');
  tbody.innerHTML = '';
  tableRowMap.clear();

  for (let k = 0; k < tableOrder.length; k++) {{
    const i = tableOrder[k];
    const uid = uids[i];

    // Apply filter
    if (tableFilter && !uid.toLowerCase().includes(tableFilter)) continue;

    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    tr.onclick = () => {{
      selectCell(i);
      panToCell(i);
    }};

    // Tier dot
    const tdTier = document.createElement('td');
    tdTier.innerHTML = '<span class="tier-dot" style="background:' + TIER_COLORS[curTier[i]] + '"></span>';
    tr.appendChild(tdTier);

    // SNR
    const tdSNR = document.createElement('td');
    tdSNR.textContent = cellSNR[i].toFixed(2);
    tr.appendChild(tdSNR);

    // BG-Sub
    const tdBG = document.createElement('td');
    tdBG.textContent = cellBG[i].toFixed(0);
    tr.appendChild(tdBG);

    // UID suffix (last part after last _)
    const tdUID = document.createElement('td');
    const parts = uid.split('_');
    tdUID.textContent = parts.slice(-2).join('_');
    tdUID.title = uid;
    tr.appendChild(tdUID);

    tbody.appendChild(tr);
    tableRowMap.set(i, tr);
  }}
}}

function updateTableRow(idx) {{
  const tr = tableRowMap.get(idx);
  if (!tr) return;
  const dot = tr.querySelector('.tier-dot');
  if (dot) dot.style.background = TIER_COLORS[curTier[idx]];
}}

function highlightTableRow(idx) {{
  clearTableHighlight();
  const tr = tableRowMap.get(idx);
  if (tr) {{
    tr.classList.add('selected');
    tr.scrollIntoView({{ block: 'nearest' }});
  }}
}}

function clearTableHighlight() {{
  document.querySelectorAll('#cell-table tr.selected').forEach(
    tr => tr.classList.remove('selected')
  );
}}

function filterTable() {{
  tableFilter = document.getElementById('search-box').value.toLowerCase().trim();
  buildTable();
}}

function sortTable(col) {{
  if (currentSortCol === col) {{
    currentSortAsc = !currentSortAsc;
  }} else {{
    currentSortCol = col;
    currentSortAsc = col === 'uid';  // default ascending for UID, descending for numbers
  }}

  const arr = Array.from({{ length: N }}, (_, i) => i);
  const dir = currentSortAsc ? 1 : -1;

  if (col === 'snr') {{
    arr.sort((a, b) => dir * (cellSNR[a] - cellSNR[b]));
  }} else if (col === 'bg') {{
    arr.sort((a, b) => dir * (cellBG[a] - cellBG[b]));
  }} else if (col === 'tier') {{
    arr.sort((a, b) => dir * (curTier[a] - curTier[b]));
  }} else if (col === 'uid') {{
    arr.sort((a, b) => dir * uids[a].localeCompare(uids[b]));
  }}

  tableOrder = new Int32Array(arr);
  buildTable();
}}

function panToCell(idx) {{
  // Center view on cell with smooth-ish jump
  const targetZoom = Math.max(zoom, 0.3);
  zoom = targetZoom;
  panX = cellX[idx] - W / (2 * zoom);
  panY = cellY[idx] - H / (2 * zoom);
  scheduleRedraw();
}}

// =========================================================================
// Init
// =========================================================================
window.addEventListener('resize', () => {{ resize(); fitAll(); scheduleRedraw(); }});

resize();
buildSpatialGrid();
fitAll();
updateCounts();
buildTable();
scheduleRedraw();

</script>
</body>
</html>'''


if __name__ == '__main__':
    main()
