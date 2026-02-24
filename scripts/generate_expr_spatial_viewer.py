#!/usr/bin/env python
"""Interactive HTML viewer for expression group spatial data across brain FISH scenes.

Reads per-scene zone data (detections_zoned.json + zone_metadata.json) from
the expression-gated spatial analysis pipeline, maps each cell to a simplified
expression group (12 groups, quad-negative excluded), computes convex hulls for
DBSCAN clusters, and generates a single self-contained HTML file with:
- 2x4 grid of scene panels (canvas-based, independent pan/zoom)
- Clickable legend to toggle expression groups on/off
- Dashed convex hull outlines + cluster labels for visible groups
- Dot size / opacity controls

Usage:
    python scripts/generate_expr_spatial_viewer.py \
        --input-dir /fs/pool/pool-mann-edwin/brain_fish_output \
        --subdir zones_expr_spatial_v2 \
        --output expr_spatial_viewer.html
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


# --- Expression group mapping ---

# Fixed color palette for 12 simplified expression groups.
# Slc17a7 family = warm, Gad1 family = cool, co-express/other = distinct.
GROUP_COLORS = {
    'Slc17a7 only':         '#e6194b',  # red
    'Slc17a7+/Htr2a+':     '#f58231',  # orange
    'Slc17a7+/Ntrk2+':     '#ffe119',  # yellow
    'Slc17a7+/Htr2a+/Ntrk2+': '#fabebe',  # light pink
    'Gad1 only':            '#4363d8',  # blue
    'Gad1+/Htr2a+':        '#42d4f4',  # cyan
    'Gad1+/Ntrk2+':        '#911eb4',  # purple
    'Gad1+/Htr2a+/Ntrk2+': '#aaffc3',  # mint
    'Slc17a7+/Gad1+':      '#f032e6',  # magenta
    'Htr2a only':           '#3cb44b',  # green
    'Ntrk2 only':           '#469990',  # teal
    'Htr2a+/Ntrk2+':       '#9a6324',  # brown
}

# Ordered list for consistent legend rendering
GROUP_ORDER = [
    'Slc17a7 only', 'Slc17a7+/Htr2a+', 'Slc17a7+/Ntrk2+', 'Slc17a7+/Htr2a+/Ntrk2+',
    'Gad1 only', 'Gad1+/Htr2a+', 'Gad1+/Ntrk2+', 'Gad1+/Htr2a+/Ntrk2+',
    'Slc17a7+/Gad1+', 'Htr2a only', 'Ntrk2 only', 'Htr2a+/Ntrk2+',
]


def parse_expression_pattern(cell_type):
    """Parse 'Slc17a7+/Htr2a-/Ntrk2-/Gad1-' into dict of marker -> bool."""
    markers = {}
    for part in cell_type.split('/'):
        m = re.match(r'^(\w+)([+-])$', part)
        if m:
            markers[m.group(1)] = (m.group(2) == '+')
    return markers


def classify_expression_group(cell_type):
    """Map full expression pattern to simplified group label.

    Returns None for quad-negative and unrecognized patterns (excluded).
    """
    if cell_type == 'other':
        return None

    markers = parse_expression_pattern(cell_type)
    if len(markers) != 4:
        return None

    slc = markers.get('Slc17a7', False)
    htr = markers.get('Htr2a', False)
    ntrk = markers.get('Ntrk2', False)
    gad = markers.get('Gad1', False)

    # Quad-negative excluded
    if not any([slc, htr, ntrk, gad]):
        return None

    # Co-expression of both major markers — Slc17a7 (VGLUT1, excitatory) + Gad1
    # (GABAergic, inhibitory) is a distinct phenotype; lump all 4 Htr2a/Ntrk2 combos.
    if slc and gad:
        return 'Slc17a7+/Gad1+'

    # Slc17a7 family (no Gad1)
    if slc and not gad:
        if htr and ntrk:
            return 'Slc17a7+/Htr2a+/Ntrk2+'
        if htr:
            return 'Slc17a7+/Htr2a+'
        if ntrk:
            return 'Slc17a7+/Ntrk2+'
        return 'Slc17a7 only'

    # Gad1 family (no Slc17a7)
    if gad and not slc:
        if htr and ntrk:
            return 'Gad1+/Htr2a+/Ntrk2+'
        if htr:
            return 'Gad1+/Htr2a+'
        if ntrk:
            return 'Gad1+/Ntrk2+'
        return 'Gad1 only'

    # Neither Slc17a7 nor Gad1
    if htr and ntrk:
        return 'Htr2a+/Ntrk2+'
    if htr:
        return 'Htr2a only'
    if ntrk:
        return 'Ntrk2 only'

    return None


def compute_convex_hull(positions, min_cells=24):
    """Compute convex hull vertices for a set of 2D positions.

    Returns list of [x, y] pairs (hull polygon), or empty list if too few cells
    or degenerate geometry.
    """
    if len(positions) < min_cells or len(positions) < 3:
        return []

    # Need at least 3 unique positions to form a polygon
    unique = np.unique(positions, axis=0)
    if len(unique) < 3:
        return []

    try:
        hull = ConvexHull(positions)
        return positions[hull.vertices].tolist()
    except Exception:
        return []


def load_scene_data(scene_dir, subdir, cluster_eps=500.0, cluster_min_cells=10,
                    min_hull_cells=24):
    """Load scene zone data and re-cluster with DBSCAN at the requested eps.

    Ignores upstream DBSCAN clustering (which used auto-eps and produced
    tissue-wide clusters). Instead, runs fresh DBSCAN per simplified
    expression group on cell positions in um.

    Returns dict with scene data or None if files missing.
    """
    zone_dir = scene_dir / subdir
    det_path = zone_dir / 'detections_zoned.json'
    meta_path = zone_dir / 'zone_metadata.json'

    if not det_path.exists() or not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    with open(det_path) as f:
        detections = json.load(f)

    # Build zone_id -> cell_type lookup (only need expression pattern)
    zone_cell_type = {}
    for z in meta['zones']:
        zone_cell_type[z['structure_id']] = z['cell_type']

    # Group cells by simplified expression group
    group_all_cells = {}  # group_label -> list of [x, y]

    for det in detections:
        zone_id = det.get('zone_id')
        if zone_id is None or zone_id not in zone_cell_type:
            continue

        group = classify_expression_group(zone_cell_type[zone_id])
        if group is None:
            continue

        pos = det.get('global_center_um')
        if pos is None or len(pos) != 2:
            continue

        group_all_cells.setdefault(group, []).append(pos)

    # Re-cluster each group with DBSCAN at the requested eps
    groups_out = []
    for group_label in GROUP_ORDER:
        cells = group_all_cells.get(group_label, [])
        if not cells:
            continue

        cells_arr = np.array(cells, dtype=np.float32)

        # Run DBSCAN on this group's positions
        db = DBSCAN(eps=cluster_eps, min_samples=cluster_min_cells)
        db_labels = db.fit_predict(cells_arr)

        clusters = []
        unique_labels = sorted(set(db_labels))
        cluster_num = 0
        for cl_id in unique_labels:
            if cl_id == -1:
                continue  # noise/scattered — dots only, no hull
            cluster_num += 1
            mask = db_labels == cl_id
            cl_positions = cells_arr[mask]
            hull = compute_convex_hull(cl_positions, min_cells=min_hull_cells)
            cx = float(cl_positions[:, 0].mean())
            cy = float(cl_positions[:, 1].mean())
            clusters.append({
                'label': f'{group_label} #{cluster_num}',
                'n': int(mask.sum()),
                'hull': hull,
                'cx': cx,
                'cy': cy,
            })

        groups_out.append({
            'label': group_label,
            'color': GROUP_COLORS[group_label],
            'n': len(cells),
            'x': cells_arr[:, 0].tolist(),
            'y': cells_arr[:, 1].tolist(),
            'clusters': clusters,
        })

    if not groups_out:
        return None

    # Compute coordinate ranges for auto-fit
    all_x = []
    all_y = []
    for g in groups_out:
        all_x.extend(g['x'])
        all_y.extend(g['y'])

    total_cells = sum(g['n'] for g in groups_out)

    return {
        'groups': groups_out,
        'n_cells': total_cells,
        'x_range': [float(min(all_x)), float(max(all_x))],
        'y_range': [float(min(all_y)), float(max(all_y))],
    }


def discover_scenes(input_dir, subdir):
    """Discover all scene directories containing zone data.

    Returns list of (scene_label, scene_path) tuples sorted by name.
    """
    input_dir = Path(input_dir)
    scenes = []

    for sample_dir in sorted(input_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        # Skip non-sample dirs (e.g., files)
        for scene_dir in sorted(sample_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not scene_dir.name.startswith('scene'):
                continue
            zone_dir = scene_dir / subdir
            if zone_dir.exists():
                label = f"{sample_dir.name}/{scene_dir.name}"
                scenes.append((label, scene_dir))

    return scenes


def generate_html(scenes_data, output_path):
    """Generate self-contained interactive HTML with 2x4 scene grid."""

    # Serialize scene data to compact JS
    scenes_js_parts = []
    for label, data in scenes_data:
        groups_js = []
        for g in data['groups']:
            # Compact float formatting
            x_str = ','.join(f'{v:.1f}' for v in g['x'])
            y_str = ','.join(f'{v:.1f}' for v in g['y'])

            clusters_js = []
            for c in g['clusters']:
                hull_str = json.dumps(c['hull'], separators=(',', ':'))
                clusters_js.append(
                    f'{{label:{json.dumps(c["label"])},n:{c["n"]},'
                    f'hull:{hull_str},cx:{c["cx"]:.1f},cy:{c["cy"]:.1f}}}'
                )

            groups_js.append(
                f'{{label:{json.dumps(g["label"])},color:"{g["color"]}",n:{g["n"]},'
                f'x:new Float32Array([{x_str}]),'
                f'y:new Float32Array([{y_str}]),'
                f'clusters:[{",".join(clusters_js)}]}}'
            )

        scene_js = (
            f'{{name:{json.dumps(label)},n:{data["n_cells"]},'
            f'xr:[{data["x_range"][0]:.1f},{data["x_range"][1]:.1f}],'
            f'yr:[{data["y_range"][0]:.1f},{data["y_range"][1]:.1f}],'
            f'groups:[{",".join(groups_js)}]}}'
        )
        scenes_js_parts.append(scene_js)

    scenes_js = ',\n'.join(scenes_js_parts)

    # Legend data (global)
    legend_js_parts = []
    for label in GROUP_ORDER:
        color = GROUP_COLORS[label]
        legend_js_parts.append(f'{{label:{json.dumps(label)},color:"{color}"}}')
    legend_js = ','.join(legend_js_parts)

    n_scenes = len(scenes_data)
    # Grid layout: up to 4 columns
    n_cols = min(4, n_scenes)
    n_rows = (n_scenes + n_cols - 1) // n_cols

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Expression Group Spatial Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }}
  #main {{ display: flex; width: 100vw; height: 100vh; }}
  #grid {{
    flex: 1;
    display: grid;
    grid-template-columns: repeat({n_cols}, 1fr);
    grid-template-rows: repeat({n_rows}, 1fr);
    gap: 2px;
    padding: 2px;
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
    width: 220px; min-width: 180px; background: rgba(26,26,46,0.95);
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
</style>
</head>
<body>
<div id="main">
  <div id="grid"></div>
  <div id="sidebar">
    <div>
      <h3>Expression Groups</h3>
      <div id="leg-items"></div>
      <div style="margin-top:6px;display:flex;gap:4px;">
        <button class="btn" id="btn-all">Show All</button>
        <button class="btn" id="btn-none">Hide All</button>
      </div>
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
    <div class="ctrl-group" style="font-size:10px;color:#666;">
      Scroll to zoom, drag to pan.<br>
      Click legend to toggle groups.
    </div>
  </div>
</div>
<script>
const SCENES = [{scenes_js}];
const LEGEND = [{legend_js}];

// State
const hidden = new Set();
let dotSize = 3, dotAlpha = 0.8, showHulls = true, showLabels = true;

// Panel state array
const panels = [];
let activePanel = null;  // panel currently being dragged
let rafId = 0;           // requestAnimationFrame ID for throttling
let rafDirty = new Set(); // panels needing re-render

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

function initPanels() {{
  const grid = document.getElementById('grid');
  SCENES.forEach((scene, idx) => {{
    const div = document.createElement('div');
    div.className = 'panel';
    const labelEl = document.createElement('div');
    labelEl.className = 'panel-label';
    labelEl.textContent = scene.name;
    const countEl = document.createElement('div');
    countEl.className = 'panel-count';
    const canvas = document.createElement('canvas');
    div.appendChild(labelEl);
    div.appendChild(countEl);
    div.appendChild(canvas);
    grid.appendChild(div);

    const ctx = canvas.getContext('2d');

    const state = {{
      div, canvas, ctx, countEl, scene,
      zoom: 1, panX: 0, panY: 0,
      dragStartX: 0, dragStartY: 0, panStartX: 0, panStartY: 0,
    }};
    panels.push(state);

    // Mousedown on panel starts drag
    div.addEventListener('mousedown', e => {{
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
      activePanel = state;
      div.classList.add('dragging');
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.panStartX = state.panX;
      state.panStartY = state.panY;
      e.preventDefault();
    }});

    // Zoom (per-panel, since wheel doesn't propagate the same way)
    div.addEventListener('wheel', e => {{
      e.preventDefault();
      const rect = div.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
      state.panX = mx - factor * (mx - state.panX);
      state.panY = my - factor * (my - state.panY);
      state.zoom *= factor;
      state.zoom = Math.max(0.01, Math.min(200, state.zoom));
      scheduleRender(state);
    }}, {{passive: false}});
  }});

  // Single global mousemove/mouseup listeners (instead of one per panel)
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

function resizePanels() {{
  const dpr = window.devicePixelRatio || 1;
  panels.forEach(p => {{
    const rect = p.div.getBoundingClientRect();
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    p.cw = w; p.ch = h;
    p.canvas.width = w * dpr;
    p.canvas.height = h * dpr;
    p.canvas.style.width = w + 'px';
    p.canvas.style.height = h + 'px';
    p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }});
}}

function fitPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const s = p.scene;
  const dataW = s.xr[1] - s.xr[0];
  const dataH = s.yr[1] - s.yr[0];
  if (dataW <= 0 || dataH <= 0) {{
    // Degenerate: single point or identical positions — center on data
    p.zoom = 1;
    p.panX = cw / 2 - s.xr[0];
    p.panY = ch / 2 - s.yr[0];
    return;
  }}
  const pad = 0.05;  // 5% padding
  p.zoom = Math.min(cw / (dataW * (1 + 2*pad)), ch / (dataH * (1 + 2*pad)));
  p.panX = (cw - dataW * p.zoom) / 2 - s.xr[0] * p.zoom;
  p.panY = (ch - dataH * p.zoom) / 2 - s.yr[0] * p.zoom;
}}

function renderPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const ctx = p.ctx;

  ctx.save();
  ctx.clearRect(0, 0, cw, ch);

  // Background
  ctx.fillStyle = '#111122';
  ctx.fillRect(0, 0, cw, ch);

  ctx.translate(p.panX, p.panY);
  ctx.scale(p.zoom, p.zoom);

  const r = dotSize / p.zoom;
  const halfR = r / 2;
  let total = 0;

  for (const group of p.scene.groups) {{
    if (hidden.has(group.label)) continue;

    // Draw dots
    ctx.globalAlpha = dotAlpha;
    ctx.fillStyle = group.color;
    const n = group.n;
    const gx = group.x, gy = group.y;
    for (let i = 0; i < n; i++) {{
      ctx.fillRect(gx[i] - halfR, gy[i] - halfR, r, r);
    }}
    total += n;

    // Draw cluster hulls
    if (showHulls) {{
      ctx.globalAlpha = 1;
      for (const cl of group.clusters) {{
        if (cl.hull.length < 3) continue;
        const path = new Path2D();
        path.moveTo(cl.hull[0][0], cl.hull[0][1]);
        for (let i = 1; i < cl.hull.length; i++) {{
          path.lineTo(cl.hull[i][0], cl.hull[i][1]);
        }}
        path.closePath();

        // Black outer stroke
        ctx.setLineDash([6/p.zoom, 4/p.zoom]);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2.5 / p.zoom;
        ctx.stroke(path);
        // White inner stroke for contrast
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.2 / p.zoom;
        ctx.stroke(path);
        ctx.setLineDash([]);

        // Cluster label at centroid
        if (showLabels) {{
          const fontSize = Math.max(8, Math.min(13, 11 / p.zoom));
          ctx.font = fontSize + 'px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          // Text shadow
          ctx.fillStyle = '#000';
          ctx.fillText(cl.label, cl.cx + 0.5/p.zoom, cl.cy + 0.5/p.zoom);
          ctx.fillStyle = '#fff';
          ctx.fillText(cl.label, cl.cx, cl.cy);
        }}
      }}
    }}
  }}

  ctx.restore();
  p.countEl.textContent = total.toLocaleString() + ' cells';
}}

function renderAll() {{
  // Immediate render (used for init and non-interactive updates)
  panels.forEach(renderPanel);
}}

// --- Legend ---
function initLegend() {{
  const legDiv = document.getElementById('leg-items');
  // Compute total per group across all scenes for counts
  const groupTotals = {{}};
  SCENES.forEach(s => {{
    s.groups.forEach(g => {{
      groupTotals[g.label] = (groupTotals[g.label] || 0) + g.n;
    }});
  }});

  LEGEND.forEach(leg => {{
    const item = document.createElement('div');
    item.className = 'leg-item';
    const count = groupTotals[leg.label] || 0;
    item.innerHTML =
      '<span class="leg-dot" style="background:' + leg.color + '"></span>' +
      '<span class="leg-label" title="' + leg.label + '">' + leg.label +
      ' (' + count.toLocaleString() + ')</span>';
    item.onclick = () => {{
      if (hidden.has(leg.label)) hidden.delete(leg.label);
      else hidden.add(leg.label);
      item.classList.toggle('hidden');
      renderAll();
    }};
    legDiv.appendChild(item);
  }});
}}

// --- Controls ---
function initControls() {{
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
    renderAll();
  }};
  document.getElementById('show-labels').onchange = e => {{
    showLabels = e.target.checked;
    renderAll();
  }};
  document.getElementById('btn-all').onclick = () => {{
    hidden.clear();
    document.querySelectorAll('.leg-item').forEach(el => el.classList.remove('hidden'));
    renderAll();
  }};
  document.getElementById('btn-none').onclick = () => {{
    LEGEND.forEach(l => hidden.add(l.label));
    document.querySelectorAll('.leg-item').forEach(el => el.classList.add('hidden'));
    renderAll();
  }};
  document.getElementById('btn-reset').onclick = () => {{
    panels.forEach(p => {{ fitPanel(p); }});
    resizePanels();
    renderAll();
  }};
}}

// --- Init ---
initPanels();
initLegend();
initControls();

function fullInit() {{
  resizePanels();
  panels.forEach(fitPanel);
  renderAll();
}}

// Wait for layout
setTimeout(fullInit, 50);
window.addEventListener('resize', () => {{
  resizePanels();
  scheduleRenderAll();
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML viewer for expression group spatial data')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing sample subdirs (e.g., brain_fish_output)')
    parser.add_argument('--subdir', default='zones_expr_spatial_v2',
                        help='Subdirectory name within each scene (default: zones_expr_spatial_v2)')
    parser.add_argument('--output', default=None,
                        help='Output HTML path (default: <input-dir>/expr_spatial_viewer.html)')
    parser.add_argument('--cluster-eps', type=float, default=500.0,
                        help='DBSCAN eps in um for spatial re-clustering (default: 500)')
    parser.add_argument('--cluster-min-cells', type=int, default=10,
                        help='DBSCAN min_samples for cluster membership (default: 10)')
    parser.add_argument('--min-hull-cells', type=int, default=24,
                        help='Minimum cells in a cluster to draw hull outline (default: 24)')

    args = parser.parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_dir / 'expr_spatial_viewer.html'

    # Discover scenes
    print(f"Scanning {input_dir} for scenes with {args.subdir}...", flush=True)
    scenes = discover_scenes(input_dir, args.subdir)
    print(f"  Found {len(scenes)} scenes", flush=True)

    if not scenes:
        print("ERROR: No scenes found with zone data", file=sys.stderr)
        sys.exit(1)

    # Load data for each scene
    scenes_data = []
    total_cells = 0
    for label, scene_dir in scenes:
        print(f"Loading {label}...", flush=True)
        data = load_scene_data(scene_dir, args.subdir,
                               cluster_eps=args.cluster_eps,
                               cluster_min_cells=args.cluster_min_cells,
                               min_hull_cells=args.min_hull_cells)
        if data is None:
            print(f"  Skipped (no valid data)", flush=True)
            continue
        n_groups = len(data['groups'])
        n_clusters = sum(len(g['clusters']) for g in data['groups'])
        n_hulls = sum(1 for g in data['groups'] for c in g['clusters'] if c['hull'])
        print(f"  {data['n_cells']} cells, {n_groups} groups, "
              f"{n_clusters} clusters ({n_hulls} with hulls)", flush=True)
        scenes_data.append((label, data))
        total_cells += data['n_cells']

    if not scenes_data:
        print("ERROR: No valid scene data loaded", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {total_cells} cells across {len(scenes_data)} scenes", flush=True)

    # Generate HTML
    print(f"Generating HTML...", flush=True)
    generate_html(scenes_data, output_path)
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
