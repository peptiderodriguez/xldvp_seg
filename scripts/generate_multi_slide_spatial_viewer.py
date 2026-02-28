#!/usr/bin/env python
"""Multi-slide scrollable spatial viewer with ROI drawing.

Loads classified detections from multiple slides and renders a scrollable HTML
with one canvas panel per slide (3-column grid), cells colored by marker class
(positive=red, negative=blue). Supports interactive ROI drawing (circle,
rectangle, freeform polygon) with JSON export.

Usage:
    # Auto-discover from pipeline output directory
    python scripts/generate_multi_slide_spatial_viewer.py \
        --input-dir /path/to/output/ \
        --detection-glob "cell_detections_classified.json" \
        --group-field tdTomato_class \
        --title "Senescence tdTomato" \
        --output spatial_viewer.html

    # Explicit list of detection files
    python scripts/generate_multi_slide_spatial_viewer.py \
        --detections slide1/cell_detections_classified.json \
                     slide2/cell_detections_classified.json \
        --group-field tdTomato_class \
        --output spatial_viewer.html
"""
import argparse
import html as html_mod
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree


# --- Auto-eps via KNN knee method ---

def compute_auto_eps(positions, k=10):
    """Compute optimal DBSCAN eps using KNN distance knee/elbow method."""
    n = len(positions)
    if n < k + 1:
        return None
    tree = KDTree(positions)
    dists, _ = tree.query(positions, k=k + 1)
    knn_dists = np.sort(dists[:, -1])
    x_norm = np.linspace(0, 1, n)
    y_range = knn_dists[-1] - knn_dists[0]
    if y_range < 1e-9:
        return float(knn_dists[0])
    y_norm = (knn_dists - knn_dists[0]) / y_range
    diffs = y_norm - x_norm
    elbow_idx = int(np.argmax(diffs))
    return float(knn_dists[elbow_idx])


# --- Data loading ---

def load_slide_data(path, group_field):
    """Load a classified detection JSON and split by group field.

    Args:
        path: Path to classified detection JSON.
        group_field: Field name to group by (e.g. 'tdTomato_class').

    Returns:
        Dict with slide data, or None if no valid data.
    """
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
        return None

    with open(path) as f:
        detections = json.load(f)

    group_cells = {}  # group_label -> list of (x, y, area)

    if not isinstance(detections, list):
        print(f"  WARNING: {path} is not a JSON list, skipping", file=sys.stderr)
        return None

    for det in detections:
        # Check top-level first, then features dict (classify_markers stores there)
        val = det.get(group_field)
        if val is None:
            val = det.get('features', {}).get(group_field)
        if val is None:
            continue
        group = str(val)

        pos = det.get('global_center_um')
        if pos is None or len(pos) != 2:
            continue
        x_val, y_val = pos[0], pos[1]
        if not isinstance(x_val, (int, float)) or not isinstance(y_val, (int, float)):
            continue
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue

        area = det.get('features', {}).get('area_um2', 0.0)
        if not isinstance(area, (int, float)) or np.isnan(area):
            area = 0.0

        group_cells.setdefault(group, []).append((pos[0], pos[1], area))

    if not group_cells:
        return None

    groups_out = []
    for label, cells in sorted(group_cells.items()):
        arr = np.array(cells, dtype=np.float32)
        auto_eps = compute_auto_eps(arr[:, :2], k=10)
        groups_out.append({
            'label': label,
            'n': len(cells),
            'x': arr[:, 0].tolist(),
            'y': arr[:, 1].tolist(),
            'a': arr[:, 2].tolist(),
            'auto_eps': auto_eps,
        })

    return {
        'groups': groups_out,
        'n_cells': sum(g['n'] for g in groups_out),
        'x_range': [float(min(min(g['x']) for g in groups_out)),
                     float(max(max(g['x']) for g in groups_out))],
        'y_range': [float(min(min(g['y']) for g in groups_out)),
                     float(max(max(g['y']) for g in groups_out))],
    }


def discover_slides(input_dir, detection_glob):
    """Discover per-slide detection files in subdirectories.

    Returns list of (slide_name, detection_path) tuples.
    """
    input_dir = Path(input_dir)
    results = []
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        matches = list(subdir.glob(detection_glob))
        if matches:
            results.append((subdir.name, matches[0]))
    return results


def assign_group_colors(slides_data):
    """Assign colors to groups across all slides.

    Binary classification: positive=red, negative=blue.
    Falls back to a palette for 3+ groups.
    """
    all_groups = set()
    for _, data in slides_data:
        for g in data['groups']:
            all_groups.add(g['label'])

    # Binary classification shortcut
    if all_groups == {'positive', 'negative'}:
        color_map = {'positive': '#e63946', 'negative': '#457b9d'}
    elif len(all_groups) <= 2:
        palette = ['#e63946', '#457b9d']
        color_map = {lbl: palette[i] for i, lbl in enumerate(sorted(all_groups))}
    else:
        palette = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990',
        ]
        color_map = {lbl: palette[i % len(palette)]
                     for i, lbl in enumerate(sorted(all_groups))}

    for _, data in slides_data:
        for g in data['groups']:
            g['color'] = color_map[g['label']]

    return color_map


# --- HTML generation ---

def generate_html(slides_data, output_path, color_map, title='Multi-Slide Spatial Viewer'):
    """Generate self-contained scrollable HTML with ROI drawing.

    Args:
        slides_data: List of (slide_name, data_dict) tuples.
        output_path: Output HTML file path.
        color_map: Dict of group_label -> hex color.
        title: Page title.
    """
    title = html_mod.escape(title)

    # Serialize slide data to compact JS
    slides_js_parts = []
    for name, data in slides_data:
        groups_js = []
        for g in data['groups']:
            x_str = ','.join(f'{v:.1f}' for v in g['x'])
            y_str = ','.join(f'{v:.1f}' for v in g['y'])
            a_str = ','.join(f'{v:.1f}' for v in g['a'])
            auto_eps = g.get('auto_eps')
            auto_eps_js = f'{auto_eps:.1f}' if auto_eps is not None else '100'
            groups_js.append(
                f'{{label:{json.dumps(g["label"])},color:"{g["color"]}",n:{g["n"]},'
                f'x:new Float32Array([{x_str}]),'
                f'y:new Float32Array([{y_str}]),'
                f'a:new Float32Array([{a_str}]),'
                f'autoEps:{auto_eps_js},'
                f'clusters:[]}}'
            )
        slide_js = (
            f'{{name:{json.dumps(name)},n:{data["n_cells"]},'
            f'xr:[{data["x_range"][0]:.1f},{data["x_range"][1]:.1f}],'
            f'yr:[{data["y_range"][0]:.1f},{data["y_range"][1]:.1f}],'
            f'groups:[{",".join(groups_js)}]}}'
        )
        slides_js_parts.append(slide_js)

    slides_js = ',\n'.join(slides_js_parts)

    # Legend items
    legend_js_parts = []
    for lbl in sorted(color_map.keys()):
        legend_js_parts.append(f'{{label:{json.dumps(lbl)},color:"{color_map[lbl]}"}}')
    legend_js = ','.join(legend_js_parts)

    n_slides = len(slides_data)
    n_cols = min(3, n_slides)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }}
  #main {{ display: flex; width: 100vw; height: 100vh; }}
  #grid {{
    flex: 1;
    display: grid;
    grid-template-columns: repeat({n_cols}, 1fr);
    grid-auto-rows: 350px;
    gap: 2px;
    padding: 2px;
    overflow-y: auto;
  }}
  .panel {{
    position: relative;
    overflow: hidden;
    background: #111122;
    border: 1px solid #333;
    border-radius: 4px;
    cursor: grab;
  }}
  .panel.dragging {{ cursor: grabbing; }}
  .panel canvas {{ position: absolute; top: 0; left: 0; }}
  .panel .draw-overlay {{ z-index: 5; pointer-events: none; }}
  .panel.draw-mode .draw-overlay {{ pointer-events: auto; cursor: crosshair; }}
  .panel-label {{
    position: absolute; top: 4px; left: 6px; z-index: 10;
    font-size: 11px; color: #ccc; background: rgba(17,17,34,0.8);
    padding: 2px 6px; border-radius: 3px; pointer-events: none;
  }}
  .panel-count {{
    position: absolute; bottom: 4px; left: 6px; z-index: 10;
    font-size: 10px; color: #888; pointer-events: none;
  }}
  #sidebar {{
    width: 260px; min-width: 220px; background: rgba(26,26,46,0.95);
    border-left: 1px solid #444; overflow-y: auto; padding: 10px;
    display: flex; flex-direction: column; gap: 10px;
  }}
  #sidebar h3 {{ font-size: 13px; color: #aaa; margin-bottom: 4px; }}
  .leg-item {{
    display: flex; align-items: center; gap: 6px; padding: 3px 4px;
    cursor: pointer; border-radius: 3px; user-select: none; font-size: 11px;
  }}
  .leg-item:hover {{ background: rgba(255,255,255,0.08); }}
  .leg-item.hidden {{ opacity: 0.25; text-decoration: line-through; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .leg-label {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .ctrl-group {{ border-top: 1px solid #333; padding-top: 8px; }}
  .ctrl-row {{ display: flex; align-items: center; gap: 6px; font-size: 11px; margin-bottom: 4px; }}
  .ctrl-row input[type=range] {{ width: 90px; }}
  .btn {{ background: #2a2a4a; border: 1px solid #555; color: #ccc; padding: 4px 8px;
    border-radius: 3px; cursor: pointer; font-size: 11px; }}
  .btn:hover {{ background: #3a3a5a; }}
  .btn.active {{ background: #3a5a3a; border-color: #6a6; color: #fff; }}
  .mode-btn {{ min-width: 46px; text-align: center; }}
  #cluster-status {{ font-size: 10px; color: #888; margin-top: 2px; }}
  #roi-list {{ max-height: 200px; overflow-y: auto; }}
  .roi-item {{
    display: flex; align-items: center; gap: 4px; padding: 2px 4px;
    font-size: 11px; border-radius: 3px;
  }}
  .roi-item:hover {{ background: rgba(255,255,255,0.05); }}
  .roi-item .roi-name {{
    flex: 1; min-width: 0; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; cursor: text;
  }}
  .roi-item .roi-name:focus {{ outline: 1px solid #555; background: #1a1a2e; }}
  .roi-item .roi-stats {{ color: #888; font-size: 10px; white-space: nowrap; }}
  .roi-del {{ cursor: pointer; color: #a55; font-size: 13px; }}
  .roi-del:hover {{ color: #f66; }}
  select {{ background: #1a1a2e; color: #ccc; border: 1px solid #555; padding: 3px 6px;
    border-radius: 3px; font-size: 11px; width: 100%; }}
</style>
</head>
<body>
<div id="main">
  <div id="grid"></div>
  <div id="sidebar">
    <div>
      <h3>{title}</h3>
      <div style="font-size:10px;color:#888;margin-bottom:4px;">{n_slides} slides</div>
      <div id="leg-items"></div>
      <div style="margin-top:6px;display:flex;gap:4px;">
        <button class="btn" id="btn-all">Show All</button>
        <button class="btn" id="btn-none">Hide All</button>
      </div>
    </div>
    <div class="ctrl-group">
      <h3>Navigation</h3>
      <select id="slide-select">
        <option value="">Jump to slide...</option>
      </select>
    </div>
    <div class="ctrl-group">
      <h3>Clustering</h3>
      <div class="ctrl-row">
        <span>Eps scale</span>
        <input type="range" id="eps-slider" min="0.25" max="3.0" value="1.0" step="0.05">
        <span id="eps-val">1.00</span><span>x</span>
      </div>
      <div class="ctrl-row">
        <span>Min cells</span>
        <input type="range" id="min-cells-slider" min="3" max="50" value="10" step="1">
        <span id="min-cells-val">10</span>
      </div>
      <div id="cluster-status"></div>
    </div>
    <div class="ctrl-group">
      <h3>Display</h3>
      <div class="ctrl-row">
        <span>Dot size</span>
        <input type="range" id="dot-size" min="1" max="10" value="3" step="0.5">
        <span id="dot-val">3</span>
      </div>
      <div class="ctrl-row">
        <span>Opacity</span>
        <input type="range" id="opacity" min="0.1" max="1" value="0.8" step="0.05">
        <span id="op-val">0.80</span>
      </div>
      <div class="ctrl-row">
        <span>Hulls</span>
        <input type="checkbox" id="show-hulls" checked>
        <span>Labels</span>
        <input type="checkbox" id="show-labels" checked>
      </div>
      <div class="ctrl-row">
        <button class="btn" id="btn-reset">Reset Zoom</button>
      </div>
    </div>
    <div class="ctrl-group">
      <h3>ROI Drawing</h3>
      <div class="ctrl-row" style="flex-wrap:wrap;gap:3px;">
        <button class="btn mode-btn" id="mode-pan" data-mode="pan">Pan</button>
        <button class="btn mode-btn" id="mode-circle" data-mode="circle">Circle</button>
        <button class="btn mode-btn" id="mode-rect" data-mode="rect">Rect</button>
        <button class="btn mode-btn" id="mode-poly" data-mode="polygon">Poly</button>
      </div>
      <div style="font-size:10px;color:#666;margin:2px 0;">
        Circle: click+drag. Rect: click+drag.<br>
        Polygon: click vertices, dbl-click close.
      </div>
      <div id="roi-list"></div>
      <div class="ctrl-row" style="margin-top:4px;">
        <button class="btn" id="btn-download-roi">Download ROIs JSON</button>
      </div>
      <div class="ctrl-row">
        <input type="checkbox" id="roi-filter">
        <span>Filter cells by ROIs</span>
      </div>
      <div id="roi-stats" style="font-size:10px;color:#888;"></div>
    </div>
    <div class="ctrl-group" style="font-size:10px;color:#666;">
      Scroll to zoom, drag to pan.<br>
      Click legend to toggle groups.<br>
      Eps auto-computed per group.
    </div>
  </div>
</div>
<script>
const SLIDES = [{slides_js}];
const LEGEND = [{legend_js}];
const MIN_HULL = 24;

// ===================== DBSCAN =====================
function dbscan(x, y, n, eps, minPts) {{
  const labels = new Int32Array(n).fill(-1);
  if (n === 0 || eps <= 0) return labels;
  const grid = new Map();
  for (let i = 0; i < n; i++) {{
    const gx = Math.floor(x[i] / eps);
    const gy = Math.floor(y[i] / eps);
    const key = gx * 100003 + gy;
    let cell = grid.get(key);
    if (!cell) {{ cell = []; grid.set(key, cell); }}
    cell.push(i);
  }}
  const eps2 = eps * eps;
  function getNeighbors(idx) {{
    const px = x[idx], py = y[idx];
    const gx = Math.floor(px / eps);
    const gy = Math.floor(py / eps);
    const result = [];
    for (let dx = -1; dx <= 1; dx++) {{
      for (let dy = -1; dy <= 1; dy++) {{
        const cell = grid.get((gx + dx) * 100003 + (gy + dy));
        if (!cell) continue;
        for (let k = 0; k < cell.length; k++) {{
          const j = cell[k];
          const ddx = x[j] - px, ddy = y[j] - py;
          if (ddx * ddx + ddy * ddy <= eps2) result.push(j);
        }}
      }}
    }}
    return result;
  }}
  let clusterId = 0;
  const visited = new Uint8Array(n);
  for (let i = 0; i < n; i++) {{
    if (visited[i]) continue;
    visited[i] = 1;
    const nbrs = getNeighbors(i);
    if (nbrs.length < minPts) continue;
    labels[i] = clusterId;
    const queue = [];
    for (let k = 0; k < nbrs.length; k++) {{
      if (nbrs[k] !== i) queue.push(nbrs[k]);
    }}
    let qi = 0;
    while (qi < queue.length) {{
      const j = queue[qi++];
      if (!visited[j]) {{
        visited[j] = 1;
        const jnbrs = getNeighbors(j);
        if (jnbrs.length >= minPts) {{
          for (let k = 0; k < jnbrs.length; k++) {{
            if (!visited[jnbrs[k]]) queue.push(jnbrs[k]);
          }}
        }}
      }}
      if (labels[j] === -1) labels[j] = clusterId;
    }}
    clusterId++;
  }}
  return labels;
}}

// ===================== Convex hull =====================
function convexHull(points) {{
  const n = points.length;
  if (n < 3) return points.slice();
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const pts = [points[0]];
  for (let i = 1; i < n; i++) {{
    if (points[i][0] !== points[i-1][0] || points[i][1] !== points[i-1][1])
      pts.push(points[i]);
  }}
  if (pts.length < 3) return pts;
  function cross(O, A, B) {{
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
  }}
  const lower = [];
  for (const p of pts) {{
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
      lower.pop();
    lower.push(p);
  }}
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {{
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
      upper.pop();
    upper.push(p);
  }}
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}}

// ===================== Re-cluster =====================
function reclusterAll() {{
  const mult = parseFloat(document.getElementById('eps-slider').value);
  const minCells = parseInt(document.getElementById('min-cells-slider').value);
  let totalClusters = 0, totalHulls = 0;
  const t0 = performance.now();
  for (const slide of SLIDES) {{
    for (const group of slide.groups) {{
      const eps = group.autoEps * mult;
      const labels = dbscan(group.x, group.y, group.n, eps, minCells);
      const clusterMap = new Map();
      for (let i = 0; i < group.n; i++) {{
        const cl = labels[i];
        if (cl === -1) continue;
        let arr = clusterMap.get(cl);
        if (!arr) {{ arr = []; clusterMap.set(cl, arr); }}
        arr.push(i);
      }}
      group.clusters = [];
      let num = 0;
      for (const [clId, indices] of clusterMap) {{
        num++;
        totalClusters++;
        const pts = [];
        let sx = 0, sy = 0, totalArea = 0;
        for (const idx of indices) {{
          const px = group.x[idx], py = group.y[idx];
          pts.push([px, py]);
          sx += px; sy += py;
          totalArea += group.a[idx];
        }}
        const cx = sx / indices.length;
        const cy = sy / indices.length;
        let hull = [];
        if (indices.length >= MIN_HULL) {{
          hull = convexHull(pts);
          if (hull.length >= 3) totalHulls++;
          else hull = [];
        }}
        group.clusters.push({{
          label: group.label + ' #' + num,
          n: indices.length,
          areaUm2: totalArea,
          hull: hull,
          cx: cx,
          cy: cy,
        }});
      }}
    }}
  }}
  const dt = (performance.now() - t0).toFixed(0);
  document.getElementById('cluster-status').textContent =
    totalClusters + ' clusters (' + totalHulls + ' hulls) ' + dt + 'ms';
}}

// ===================== State =====================
const hidden = new Set();
let dotSize = 3, dotAlpha = 0.8, showHulls = true, showLabels = true;
let drawMode = 'pan';  // pan | circle | rect | polygon

// ROI storage
const rois = [];  // {{id, slide, type, data, name}}
let roiCounter = 0;
let roiFilterActive = false;

// Polygon in-progress state
let polySlide = null;
let polyVerts = [];

// Drawing in-progress state
let drawStart = null;  // {{x, y}} in data coords
let drawCurrent = null;

// Panel state
const panels = [];
let activePanel = null;
let rafId = 0;
const rafDirty = new Set();

function scheduleRender(p) {{
  rafDirty.add(p);
  if (!rafId) {{
    rafId = requestAnimationFrame(() => {{
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    }});
  }}
}}

function scheduleRenderAll() {{
  panels.forEach(p => rafDirty.add(p));
  if (!rafId) {{
    rafId = requestAnimationFrame(() => {{
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    }});
  }}
}}

// ===================== ROI geometry tests =====================
function pointInCircle(px, py, cx, cy, r) {{
  const dx = px - cx, dy = py - cy;
  return dx * dx + dy * dy <= r * r;
}}

function pointInRect(px, py, x1, y1, x2, y2) {{
  const minX = Math.min(x1, x2), maxX = Math.max(x1, x2);
  const minY = Math.min(y1, y2), maxY = Math.max(y1, y2);
  return px >= minX && px <= maxX && py >= minY && py <= maxY;
}}

function pointInPolygon(px, py, verts) {{
  // Ray casting algorithm
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {{
    const xi = verts[i][0], yi = verts[i][1];
    const xj = verts[j][0], yj = verts[j][1];
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {{
      inside = !inside;
    }}
  }}
  return inside;
}}

function pointInROI(px, py, roi) {{
  if (roi.type === 'circle') {{
    return pointInCircle(px, py, roi.data.cx, roi.data.cy, roi.data.r);
  }} else if (roi.type === 'rect') {{
    return pointInRect(px, py, roi.data.x1, roi.data.y1, roi.data.x2, roi.data.y2);
  }} else if (roi.type === 'polygon') {{
    return pointInPolygon(px, py, roi.data.verts);
  }}
  return false;
}}

function cellPassesROIFilter(px, py, slideName) {{
  if (!roiFilterActive || rois.length === 0) return true;
  // Check all ROIs for this slide
  for (const roi of rois) {{
    if (roi.slide === slideName && pointInROI(px, py, roi)) return true;
  }}
  return false;
}}

// ===================== Coordinate transforms =====================
function screenToData(p, sx, sy) {{
  return [(sx - p.panX) / p.zoom, (sy - p.panY) / p.zoom];
}}

function dataToScreen(p, dx, dy) {{
  return [dx * p.zoom + p.panX, dy * p.zoom + p.panY];
}}

// ===================== Panel init =====================
const observer = new IntersectionObserver((entries) => {{
  for (const entry of entries) {{
    const idx = parseInt(entry.target.dataset.idx);
    const p = panels[idx];
    if (entry.isIntersecting) {{
      p.visible = true;
      scheduleRender(p);
    }} else {{
      p.visible = false;
    }}
  }}
}}, {{ root: document.getElementById('grid'), threshold: 0.01 }});

function initPanels() {{
  const grid = document.getElementById('grid');
  const select = document.getElementById('slide-select');

  SLIDES.forEach((slide, idx) => {{
    const div = document.createElement('div');
    div.className = 'panel';
    div.dataset.idx = idx;

    const labelEl = document.createElement('div');
    labelEl.className = 'panel-label';
    labelEl.textContent = slide.name;
    const countEl = document.createElement('div');
    countEl.className = 'panel-count';

    const canvas = document.createElement('canvas');
    const drawCanvas = document.createElement('canvas');
    drawCanvas.className = 'draw-overlay';

    div.appendChild(labelEl);
    div.appendChild(countEl);
    div.appendChild(canvas);
    div.appendChild(drawCanvas);
    grid.appendChild(div);

    const ctx = canvas.getContext('2d');
    const dctx = drawCanvas.getContext('2d');

    const state = {{
      div, canvas, ctx, drawCanvas, dctx, countEl, slide, idx,
      zoom: 1, panX: 0, panY: 0,
      dragStartX: 0, dragStartY: 0, panStartX: 0, panStartY: 0,
      visible: false,
    }};
    panels.push(state);
    observer.observe(div);

    // Pan/zoom on data canvas
    canvas.addEventListener('mousedown', e => {{
      if (drawMode !== 'pan') return;
      activePanel = state;
      div.classList.add('dragging');
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.panStartX = state.panX;
      state.panStartY = state.panY;
      e.preventDefault();
    }});

    // Drawing events on overlay canvas
    drawCanvas.addEventListener('mousedown', e => {{
      if (drawMode === 'pan') return;
      const rect = div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(state, sx, sy);

      if (drawMode === 'polygon') {{
        if (polySlide !== slide.name) {{
          polySlide = slide.name;
          polyVerts = [];
        }}
        polyVerts.push([dx, dy]);
        renderDrawOverlay(state);
      }} else {{
        drawStart = {{ x: dx, y: dy, panel: state }};
        drawCurrent = {{ x: dx, y: dy }};
      }}
      e.preventDefault();
    }});

    drawCanvas.addEventListener('mousemove', e => {{
      if (drawMode === 'pan') return;
      if (!drawStart || drawStart.panel !== state) return;
      const rect = div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(state, sx, sy);
      drawCurrent = {{ x: dx, y: dy }};
      renderDrawOverlay(state);
    }});

    drawCanvas.addEventListener('mouseup', e => {{
      if (drawMode === 'pan' || drawMode === 'polygon') return;
      if (!drawStart || drawStart.panel !== state) return;
      const rect = div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(state, sx, sy);

      if (drawMode === 'circle') {{
        const cdx = dx - drawStart.x, cdy = dy - drawStart.y;
        const r = Math.sqrt(cdx * cdx + cdy * cdy);
        if (r > 1) {{
          addROI(slide.name, 'circle', {{ cx: drawStart.x, cy: drawStart.y, r }});
        }}
      }} else if (drawMode === 'rect') {{
        const w = Math.abs(dx - drawStart.x), h = Math.abs(dy - drawStart.y);
        if (w > 1 && h > 1) {{
          addROI(slide.name, 'rect', {{ x1: drawStart.x, y1: drawStart.y, x2: dx, y2: dy }});
        }}
      }}
      drawStart = null;
      drawCurrent = null;
      renderDrawOverlay(state);
    }});

    drawCanvas.addEventListener('dblclick', e => {{
      if (drawMode !== 'polygon') return;
      if (polySlide === slide.name && polyVerts.length >= 3) {{
        addROI(slide.name, 'polygon', {{ verts: polyVerts.slice() }});
      }}
      polyVerts = [];
      polySlide = null;
      renderDrawOverlay(state);
      e.preventDefault();
    }});

    // Wheel zoom on both canvases
    function handleWheel(e) {{
      e.preventDefault();
      const rect = div.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      state.panX = mx - factor * (mx - state.panX);
      state.panY = my - factor * (my - state.panY);
      state.zoom *= factor;
      state.zoom = Math.max(0.01, Math.min(200, state.zoom));
      scheduleRender(state);
      renderDrawOverlay(state);
    }}
    canvas.addEventListener('wheel', handleWheel, {{ passive: false }});
    drawCanvas.addEventListener('wheel', handleWheel, {{ passive: false }});

    // Slide dropdown
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = slide.name;
    select.appendChild(opt);
  }});

  // Global mouse handlers for pan
  window.addEventListener('mousemove', e => {{
    if (!activePanel) return;
    activePanel.panX = activePanel.panStartX + (e.clientX - activePanel.dragStartX);
    activePanel.panY = activePanel.panStartY + (e.clientY - activePanel.dragStartY);
    scheduleRender(activePanel);
  }});
  window.addEventListener('mouseup', () => {{
    if (activePanel) {{
      activePanel.div.classList.remove('dragging');
      activePanel = null;
    }}
  }});
}}

// ===================== Resize / fit =====================
function resizePanels() {{
  const dpr = window.devicePixelRatio || 1;
  panels.forEach(p => {{
    const rect = p.div.getBoundingClientRect();
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    p.cw = w; p.ch = h;
    // Data canvas
    p.canvas.width = w * dpr;
    p.canvas.height = h * dpr;
    p.canvas.style.width = w + 'px';
    p.canvas.style.height = h + 'px';
    p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Draw overlay canvas
    p.drawCanvas.width = w * dpr;
    p.drawCanvas.height = h * dpr;
    p.drawCanvas.style.width = w + 'px';
    p.drawCanvas.style.height = h + 'px';
    p.dctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }});
}}

function fitPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const s = p.slide;
  const dataW = s.xr[1] - s.xr[0];
  const dataH = s.yr[1] - s.yr[0];
  if (dataW <= 0 || dataH <= 0) {{
    p.zoom = 1;
    p.panX = cw / 2 - s.xr[0];
    p.panY = ch / 2 - s.yr[0];
    return;
  }}
  const pad = 0.05;
  p.zoom = Math.min(cw / (dataW * (1 + 2 * pad)), ch / (dataH * (1 + 2 * pad)));
  p.panX = (cw - dataW * p.zoom) / 2 - s.xr[0] * p.zoom;
  p.panY = (ch - dataH * p.zoom) / 2 - s.yr[0] * p.zoom;
}}

// ===================== Render data panel =====================
function renderPanel(p) {{
  if (!p.visible) return;
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const ctx = p.ctx;
  ctx.save();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#111122';
  ctx.fillRect(0, 0, cw, ch);
  ctx.translate(p.panX, p.panY);
  ctx.scale(p.zoom, p.zoom);

  const r = dotSize / p.zoom;
  const halfR = r / 2;
  let total = 0;
  const slideName = p.slide.name;

  for (const group of p.slide.groups) {{
    if (hidden.has(group.label)) continue;
    ctx.globalAlpha = dotAlpha;
    ctx.fillStyle = group.color;
    const n = group.n;
    const gx = group.x, gy = group.y;
    if (roiFilterActive && rois.length > 0) {{
      for (let i = 0; i < n; i++) {{
        if (cellPassesROIFilter(gx[i], gy[i], slideName)) {{
          ctx.fillRect(gx[i] - halfR, gy[i] - halfR, r, r);
          total++;
        }}
      }}
    }} else {{
      for (let i = 0; i < n; i++) {{
        ctx.fillRect(gx[i] - halfR, gy[i] - halfR, r, r);
      }}
      total += n;
    }}

    // Cluster hulls
    if (showHulls && group.clusters) {{
      ctx.globalAlpha = 1;
      for (const cl of group.clusters) {{
        if (!cl.hull || cl.hull.length < 3) continue;
        const path = new Path2D();
        path.moveTo(cl.hull[0][0], cl.hull[0][1]);
        for (let i = 1; i < cl.hull.length; i++) {{
          path.lineTo(cl.hull[i][0], cl.hull[i][1]);
        }}
        path.closePath();
        ctx.setLineDash([6 / p.zoom, 4 / p.zoom]);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2.5 / p.zoom;
        ctx.stroke(path);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.2 / p.zoom;
        ctx.stroke(path);
        ctx.setLineDash([]);

        if (showLabels) {{
          const fontSize = 11 / p.zoom;
          ctx.font = fontSize + 'px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const areaStr = cl.areaUm2 >= 1000
            ? (cl.areaUm2 / 1000).toFixed(1) + 'K'
            : Math.round(cl.areaUm2).toString();
          const line1 = cl.n + ' cells';
          const line2 = areaStr + ' um\\u00B2';
          const lh = fontSize * 1.2;
          ctx.fillStyle = '#000';
          ctx.fillText(line1, cl.cx + 0.5 / p.zoom, cl.cy - lh / 2 + 0.5 / p.zoom);
          ctx.fillText(line2, cl.cx + 0.5 / p.zoom, cl.cy + lh / 2 + 0.5 / p.zoom);
          ctx.fillStyle = '#fff';
          ctx.fillText(line1, cl.cx, cl.cy - lh / 2);
          ctx.fillText(line2, cl.cx, cl.cy + lh / 2);
        }}
      }}
    }}
  }}
  ctx.restore();
  p.countEl.textContent = total.toLocaleString() + ' cells';
}}

// ===================== Render draw overlay =====================
function renderDrawOverlay(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const dctx = p.dctx;
  dctx.clearRect(0, 0, cw, ch);

  dctx.save();
  dctx.translate(p.panX, p.panY);
  dctx.scale(p.zoom, p.zoom);

  const lw = 1.5 / p.zoom;

  // Draw existing ROIs for this slide
  for (const roi of rois) {{
    if (roi.slide !== p.slide.name) continue;
    dctx.strokeStyle = '#ffcc00';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.8;
    dctx.setLineDash([4 / p.zoom, 3 / p.zoom]);

    if (roi.type === 'circle') {{
      dctx.beginPath();
      dctx.arc(roi.data.cx, roi.data.cy, roi.data.r, 0, Math.PI * 2);
      dctx.stroke();
    }} else if (roi.type === 'rect') {{
      const x = Math.min(roi.data.x1, roi.data.x2);
      const y = Math.min(roi.data.y1, roi.data.y2);
      const w = Math.abs(roi.data.x2 - roi.data.x1);
      const h = Math.abs(roi.data.y2 - roi.data.y1);
      dctx.strokeRect(x, y, w, h);
    }} else if (roi.type === 'polygon') {{
      dctx.beginPath();
      dctx.moveTo(roi.data.verts[0][0], roi.data.verts[0][1]);
      for (let i = 1; i < roi.data.verts.length; i++) {{
        dctx.lineTo(roi.data.verts[i][0], roi.data.verts[i][1]);
      }}
      dctx.closePath();
      dctx.stroke();
    }}
    dctx.setLineDash([]);

    // ROI label
    const fontSize = 10 / p.zoom;
    dctx.font = fontSize + 'px system-ui';
    dctx.fillStyle = '#ffcc00';
    dctx.globalAlpha = 0.9;
    dctx.textAlign = 'left';
    dctx.textBaseline = 'top';
    let labelX, labelY;
    if (roi.type === 'circle') {{
      labelX = roi.data.cx - roi.data.r;
      labelY = roi.data.cy - roi.data.r - fontSize * 1.2;
    }} else if (roi.type === 'rect') {{
      labelX = Math.min(roi.data.x1, roi.data.x2);
      labelY = Math.min(roi.data.y1, roi.data.y2) - fontSize * 1.2;
    }} else {{
      labelX = roi.data.verts[0][0];
      labelY = roi.data.verts[0][1] - fontSize * 1.2;
    }}
    dctx.fillText(roi.name, labelX, labelY);
  }}

  // Draw in-progress shape
  if (drawStart && drawCurrent && drawStart.panel === p) {{
    dctx.strokeStyle = '#00ff88';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.7;
    dctx.setLineDash([3 / p.zoom, 2 / p.zoom]);

    if (drawMode === 'circle') {{
      const dx = drawCurrent.x - drawStart.x;
      const dy = drawCurrent.y - drawStart.y;
      const r = Math.sqrt(dx * dx + dy * dy);
      dctx.beginPath();
      dctx.arc(drawStart.x, drawStart.y, r, 0, Math.PI * 2);
      dctx.stroke();
    }} else if (drawMode === 'rect') {{
      const x = Math.min(drawStart.x, drawCurrent.x);
      const y = Math.min(drawStart.y, drawCurrent.y);
      const w = Math.abs(drawCurrent.x - drawStart.x);
      const h = Math.abs(drawCurrent.y - drawStart.y);
      dctx.strokeRect(x, y, w, h);
    }}
    dctx.setLineDash([]);
  }}

  // Draw in-progress polygon
  if (drawMode === 'polygon' && polySlide === p.slide.name && polyVerts.length > 0) {{
    dctx.strokeStyle = '#00ff88';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.7;
    dctx.beginPath();
    dctx.moveTo(polyVerts[0][0], polyVerts[0][1]);
    for (let i = 1; i < polyVerts.length; i++) {{
      dctx.lineTo(polyVerts[i][0], polyVerts[i][1]);
    }}
    dctx.stroke();
    // Draw vertices
    const vr = 3 / p.zoom;
    dctx.fillStyle = '#00ff88';
    for (const v of polyVerts) {{
      dctx.beginPath();
      dctx.arc(v[0], v[1], vr, 0, Math.PI * 2);
      dctx.fill();
    }}
  }}

  dctx.restore();
}}

// ===================== ROI management =====================
function addROI(slide, type, data) {{
  roiCounter++;
  const roi = {{
    id: 'ROI_' + roiCounter,
    slide,
    type,
    data,
    name: 'ROI ' + roiCounter,
  }};
  rois.push(roi);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
}}

function deleteROI(id) {{
  const idx = rois.findIndex(r => r.id === id);
  if (idx >= 0) rois.splice(idx, 1);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
}}

function updateROIList() {{
  const div = document.getElementById('roi-list');
  div.innerHTML = '';
  for (const roi of rois) {{
    const item = document.createElement('div');
    item.className = 'roi-item';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'roi-name';
    nameSpan.contentEditable = true;
    nameSpan.textContent = roi.name;
    nameSpan.title = roi.slide + ' | ' + roi.type;
    nameSpan.onblur = () => {{ roi.name = nameSpan.textContent.trim() || roi.id; }};
    nameSpan.onkeydown = (e) => {{ if (e.key === 'Enter') {{ e.preventDefault(); nameSpan.blur(); }} }};

    const statsSpan = document.createElement('span');
    statsSpan.className = 'roi-stats';
    statsSpan.dataset.roiId = roi.id;

    const delBtn = document.createElement('span');
    delBtn.className = 'roi-del';
    delBtn.textContent = '\\u00d7';
    delBtn.onclick = () => deleteROI(roi.id);

    item.appendChild(nameSpan);
    item.appendChild(statsSpan);
    item.appendChild(delBtn);
    div.appendChild(item);
  }}
}}

function updateROIStats() {{
  // Count cells inside each ROI
  for (const roi of rois) {{
    let count = 0;
    for (const slide of SLIDES) {{
      if (slide.name !== roi.slide) continue;
      for (const group of slide.groups) {{
        if (hidden.has(group.label)) continue;
        for (let i = 0; i < group.n; i++) {{
          if (pointInROI(group.x[i], group.y[i], roi)) count++;
        }}
      }}
    }}
    const el = document.querySelector('[data-roi-id="' + roi.id + '"]');
    if (el) el.textContent = count.toLocaleString();
  }}

  // Overall stats
  const statsDiv = document.getElementById('roi-stats');
  if (rois.length === 0) {{
    statsDiv.textContent = '';
  }} else {{
    statsDiv.textContent = rois.length + ' ROI(s) drawn';
  }}
}}

function downloadROIs() {{
  const out = {{ rois: [] }};
  for (const roi of rois) {{
    const entry = {{ id: roi.id, slide: roi.slide, type: roi.type, name: roi.name }};
    if (roi.type === 'circle') {{
      entry.center_um = [roi.data.cx, roi.data.cy];
      entry.radius_um = roi.data.r;
    }} else if (roi.type === 'rect') {{
      entry.vertices_um = [
        [roi.data.x1, roi.data.y1],
        [roi.data.x2, roi.data.y1],
        [roi.data.x2, roi.data.y2],
        [roi.data.x1, roi.data.y2],
      ];
    }} else if (roi.type === 'polygon') {{
      entry.vertices_um = roi.data.verts;
    }}
    out.rois.push(entry);
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'rois.json';
  a.click();
  URL.revokeObjectURL(url);
}}

// ===================== Legend =====================
function initLegend() {{
  const legDiv = document.getElementById('leg-items');
  const groupTotals = {{}};
  SLIDES.forEach(s => {{
    s.groups.forEach(g => {{
      groupTotals[g.label] = (groupTotals[g.label] || 0) + g.n;
    }});
  }});
  LEGEND.forEach(leg => {{
    const item = document.createElement('div');
    item.className = 'leg-item';
    const count = groupTotals[leg.label] || 0;
    const dot = document.createElement('span');
    dot.className = 'leg-dot';
    dot.style.background = leg.color;
    const label = document.createElement('span');
    label.className = 'leg-label';
    label.title = leg.label;
    label.textContent = leg.label + ' (' + count.toLocaleString() + ')';
    item.appendChild(dot);
    item.appendChild(label);
    item.onclick = () => {{
      if (hidden.has(leg.label)) hidden.delete(leg.label);
      else hidden.add(leg.label);
      item.classList.toggle('hidden');
      scheduleRenderAll();
    }};
    legDiv.appendChild(item);
  }});
}}

// ===================== Controls =====================
function initControls() {{
  const epsSlider = document.getElementById('eps-slider');
  const minCellsSlider = document.getElementById('min-cells-slider');

  epsSlider.oninput = e => {{
    document.getElementById('eps-val').textContent = parseFloat(e.target.value).toFixed(2);
  }};
  epsSlider.onchange = () => {{
    reclusterAll();
    scheduleRenderAll();
  }};
  minCellsSlider.oninput = e => {{
    document.getElementById('min-cells-val').textContent = e.target.value;
  }};
  minCellsSlider.onchange = () => {{
    reclusterAll();
    scheduleRenderAll();
  }};

  document.getElementById('dot-size').oninput = e => {{
    dotSize = parseFloat(e.target.value);
    document.getElementById('dot-val').textContent = dotSize;
    scheduleRenderAll();
  }};
  document.getElementById('opacity').oninput = e => {{
    dotAlpha = parseFloat(e.target.value);
    document.getElementById('op-val').textContent = dotAlpha.toFixed(2);
    scheduleRenderAll();
  }};
  document.getElementById('show-hulls').onchange = e => {{
    showHulls = e.target.checked;
    scheduleRenderAll();
  }};
  document.getElementById('show-labels').onchange = e => {{
    showLabels = e.target.checked;
    scheduleRenderAll();
  }};
  document.getElementById('btn-all').onclick = () => {{
    hidden.clear();
    document.querySelectorAll('.leg-item').forEach(el => el.classList.remove('hidden'));
    scheduleRenderAll();
  }};
  document.getElementById('btn-none').onclick = () => {{
    LEGEND.forEach(l => hidden.add(l.label));
    document.querySelectorAll('.leg-item').forEach(el => el.classList.add('hidden'));
    scheduleRenderAll();
  }};
  document.getElementById('btn-reset').onclick = () => {{
    resizePanels();
    panels.forEach(fitPanel);
    scheduleRenderAll();
  }};

  // Slide jump
  document.getElementById('slide-select').onchange = e => {{
    const idx = parseInt(e.target.value);
    if (!isNaN(idx) && panels[idx]) {{
      panels[idx].div.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    }}
  }};

  // Draw mode buttons
  document.querySelectorAll('.mode-btn').forEach(btn => {{
    btn.onclick = () => {{
      drawMode = btn.dataset.mode;
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      // Toggle draw overlay pointer events
      panels.forEach(p => {{
        if (drawMode === 'pan') {{
          p.div.classList.remove('draw-mode');
        }} else {{
          p.div.classList.add('draw-mode');
        }}
      }});
      // Clear all in-progress drawing state
      drawStart = null;
      drawCurrent = null;
      if (drawMode !== 'polygon') {{
        polyVerts = [];
        polySlide = null;
      }}
      panels.forEach(p => renderDrawOverlay(p));
    }};
  }});
  // Set pan as default active
  document.getElementById('mode-pan').classList.add('active');

  // ROI controls
  document.getElementById('btn-download-roi').onclick = downloadROIs;
  document.getElementById('roi-filter').onchange = e => {{
    roiFilterActive = e.target.checked;
    scheduleRenderAll();
  }};
}}

// ===================== Init =====================
initPanels();
initLegend();
initControls();

function fullInit() {{
  resizePanels();
  panels.forEach(fitPanel);
  reclusterAll();
  scheduleRenderAll();
}}

setTimeout(fullInit, 100);
window.addEventListener('resize', () => {{
  resizePanels();
  scheduleRenderAll();
  panels.forEach(p => renderDrawOverlay(p));
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-slide scrollable spatial viewer with ROI drawing')
    parser.add_argument('--input-dir',
                        help='Directory containing per-slide subdirectories')
    parser.add_argument('--detections', nargs='+',
                        help='Explicit list of detection JSON files')
    parser.add_argument('--detection-glob', default='cell_detections_classified.json',
                        help='Glob pattern for detection files within slide subdirs '
                             '(default: cell_detections_classified.json)')
    parser.add_argument('--group-field', required=True,
                        help='Field name to group/color by (e.g. tdTomato_class)')
    parser.add_argument('--title', default='Multi-Slide Spatial Viewer',
                        help='Page title')
    parser.add_argument('--output', default='spatial_viewer.html',
                        help='Output HTML file path')
    args = parser.parse_args()

    if not args.input_dir and not args.detections:
        parser.error('Provide either --input-dir or --detections')

    # Discover or use explicit files
    if args.input_dir:
        slide_files = discover_slides(args.input_dir, args.detection_glob)
        if not slide_files:
            print(f"Error: no detection files matching '{args.detection_glob}' "
                  f"found in {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(slide_files)} slides in {args.input_dir}")
    else:
        slide_files = []
        for p in args.detections:
            path = Path(p)
            name = path.parent.name or path.stem
            slide_files.append((name, path))

    # Load data
    slides_data = []
    for name, path in slide_files:
        print(f"  Loading {name}...", end='', flush=True)
        data = load_slide_data(path, args.group_field)
        if data is None:
            print(" skipped (no data)")
            continue
        slides_data.append((name, data))
        print(f" {data['n_cells']} cells, {len(data['groups'])} groups")

    if not slides_data:
        print("Error: no valid slide data loaded", file=sys.stderr)
        sys.exit(1)

    # Assign colors
    color_map = assign_group_colors(slides_data)
    print(f"\nGroups: {', '.join(f'{k} ({v})' for k, v in color_map.items())}")

    # Generate HTML
    total_cells = sum(d['n_cells'] for _, d in slides_data)
    print(f"\nGenerating HTML for {len(slides_data)} slides, "
          f"{total_cells:,} total cells...")
    generate_html(slides_data, args.output, color_map, title=args.title)


if __name__ == '__main__':
    main()
