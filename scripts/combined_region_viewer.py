#!/usr/bin/env python
"""Combined whole-slide region viewer + per-region PCA/UMAP explorer.

Left pane: whole-slide map with fluorescence background and clickable region
contours. Right pane: per-region UMAP + PC1 vs PC2 + clustering metrics with
4-way coloring (kmeans / leiden / hdbscan·PCA / hdbscan·UMAP).

Click a region on the map (or in the sidebar) → right pane loads that region's
cached PCA/UMAP data. Selected region is highlighted on the map.

Usage:
    python scripts/combined_region_viewer.py \\
        --detections cell_detections_with_organs.json \\
        --label-map labels_*_filled.npy \\
        --czi-path slide.czi \\
        --display-channels "4,2" \\
        --min-cells 1000 \\
        --feature-groups morph,channel \\
        --output combined_viewer.html
"""

import argparse
import base64
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.analysis.cluster_features import select_feature_names  # noqa: E402
from xldvp_seg.analysis.region_clustering import process_region  # noqa: E402
from xldvp_seg.utils.image_utils import percentile_normalize  # noqa: E402
from xldvp_seg.utils.json_utils import fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402
from xldvp_seg.visualization.encoding import safe_json  # noqa: E402
from xldvp_seg.visualization.fluorescence import read_czi_thumbnail_channels  # noqa: E402
from xldvp_seg.visualization.region_viewer import extract_region_contours  # noqa: E402

logger = get_logger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections", required=True, help="Detections JSON with organ_id")
    parser.add_argument("--label-map", required=True, help="Region label map .npy")
    parser.add_argument("--czi-path", required=True, help="CZI for fluorescence background")
    parser.add_argument(
        "--display-channels", default="4,2", help="Comma-separated channels for background"
    )
    parser.add_argument(
        "--scale", type=float, default=1 / 256, help="Thumbnail scale (default: 1/256)"
    )
    parser.add_argument("--scene", type=int, default=0, help="CZI scene index")
    parser.add_argument("--nuc-stats", help="region_nuc_stats.json for sidebar info")
    parser.add_argument("--min-cells", type=int, default=1000, help="Min nucleated cells")
    parser.add_argument("--feature-groups", default="morph,channel")
    parser.add_argument("--var-cutoff", type=float, default=0.90)
    parser.add_argument("--max-pcs", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-knn", type=int, default=15)
    parser.add_argument("--hdbscan-min-size", type=int, default=50)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--max-points-plot", type=int, default=5000)
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible clustering (default: module default 42 + warning)",
    )
    return parser.parse_args(argv)


def build_fluor_thumbnails(czi_path, channels, scale, scene):
    """Build base64-encoded JPEG thumbnails for each requested channel."""
    thumbs = []
    for ch in channels:
        ch_data, _, _, _ = read_czi_thumbnail_channels(
            czi_path, display_channels=[ch], scale_factor=scale, scene=scene
        )
        norm = percentile_normalize(ch_data[0])
        buf = io.BytesIO()
        Image.fromarray(norm).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        thumbs.append((f"ch{ch}", b64))
    # Also build a combined RGB thumbnail if 2+ channels
    if len(channels) >= 2:
        arrs = []
        for ch in channels[:3]:
            ch_data, _, _, _ = read_czi_thumbnail_channels(
                czi_path, display_channels=[ch], scale_factor=scale, scene=scene
            )
            arrs.append(percentile_normalize(ch_data[0]))
        while len(arrs) < 3:
            arrs.append(np.zeros_like(arrs[0]))
        rgb = np.stack(arrs[:3], axis=-1)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        name = "_".join(f"ch{c}" for c in channels[:3])
        thumbs.insert(0, (name, b64))  # combined first = default
    return thumbs


def generate_combined_html(
    region_data,
    contours,
    fluor_thumbnails,
    label_shape,
    output_path,
    nuc_stats=None,
    title="Combined Region + UMAP Viewer",
):
    """Emit the 2-pane combined viewer HTML."""
    sorted_regions = sorted(region_data.items(), key=lambda x: -x[1]["hopkins"])
    region_info = []
    for rid, data in sorted_regions:
        info = {
            "id": rid,
            "n_cells": data["n_cells"],
            "hopkins": data["hopkins"],
            "silhouette": data["silhouette"],
            "best_k": data["best_k"],
        }
        if nuc_stats and rid in nuc_stats:
            s = nuc_stats[rid]
            info["mean_nuc"] = s.get("mean_nuc", s.get("mean", "?"))
        region_info.append(info)

    # Keep contour payload tight: list of [region_id, flat_xy] per region
    contour_payload = {int(rid): pts for rid, pts in contours.items()}

    H, W = label_shape

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; display: flex; height: 100vh; overflow: hidden; }}
#sidebar {{ width: 240px; min-width: 240px; background: #111; border-right: 1px solid #333; display: flex; flex-direction: column; }}
#sidebar-header {{ padding: 12px; border-bottom: 1px solid #333; }}
#sidebar-header h2 {{ font-size: 14px; color: #fff; margin-bottom: 6px; }}
#sidebar-header .summary {{ font-size: 11px; color: #888; }}
#search {{ width: 100%; padding: 6px 8px; background: #1a1a1a; border: 1px solid #333; color: #e0e0e0; font-family: inherit; font-size: 11px; margin-top: 6px; }}
.sort-btn {{ padding: 3px 6px; background: #1a1a1a; border: 1px solid #333; color: #888; font-family: inherit; font-size: 10px; cursor: pointer; border-radius: 3px; }}
.sort-btn:hover {{ background: #222; }}
.sort-btn.active {{ background: #1a2a1a; border-color: #4caf50; color: #4caf50; }}
#region-list {{ flex: 1; overflow-y: auto; }}
.region-item {{ padding: 6px 12px; border-bottom: 1px solid #222; cursor: pointer; }}
.region-item:hover {{ background: #1a1a1a; }}
.region-item.active {{ background: #1a2a1a; border-left: 3px solid #4caf50; }}
.region-item .rid {{ font-weight: bold; color: #fff; font-size: 11px; }}
.region-item .meta {{ font-size: 10px; color: #888; margin-top: 2px; }}

#spatial-pane {{ width: 45%; min-width: 400px; background: #0a0a0a; border-right: 1px solid #333; position: relative; display: flex; flex-direction: column; }}
#spatial-toolbar {{ display: flex; gap: 6px; padding: 6px; background: #111; border-bottom: 1px solid #333; flex-wrap: wrap; align-items: center; font-size: 10px; color: #888; }}
#spatial-toolbar button, #spatial-toolbar select {{ padding: 3px 8px; background: #1a1a1a; border: 1px solid #333; color: #ddd; font-family: inherit; font-size: 10px; cursor: pointer; border-radius: 3px; }}
#spatial-toolbar button:hover {{ background: #222; }}
#canvas-wrap {{ flex: 1; position: relative; overflow: hidden; }}
#bg-canvas, #contour-canvas {{ position: absolute; top: 0; left: 0; }}

#right-pane {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 500px; }}
#top-bar {{ padding: 8px 16px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
#top-bar .metric {{ font-size: 11px; }}
#top-bar .metric .val {{ font-weight: bold; color: #fff; }}
#top-bar .metric .label {{ color: #888; }}
.color-btn {{ padding: 3px 6px; background: #1a1a1a; border: 1px solid #333; color: #888; font-family: inherit; font-size: 10px; cursor: pointer; border-radius: 3px; margin-left: 2px; }}
.color-btn:hover {{ background: #222; }}
.color-btn.active {{ background: #1a2a1a; border-color: #4caf50; color: #4caf50; }}
#plots {{ flex: 1; display: flex; gap: 4px; padding: 8px; overflow: hidden; min-height: 280px; }}
.plot-container {{ flex: 1; position: relative; background: #0d0d0d; border: 1px solid #222; border-radius: 4px; }}
.plot-container canvas {{ width: 100%; height: 100%; }}
.plot-title {{ position: absolute; top: 6px; left: 50%; transform: translateX(-50%); font-size: 11px; color: #aaa; pointer-events: none; z-index: 1; }}
#loadings {{ padding: 6px 16px; border-top: 1px solid #333; font-size: 10px; color: #888; max-height: 70px; overflow-y: auto; }}
#loadings .pc-row {{ margin-bottom: 2px; }}
#loadings .pc-label {{ color: #aaa; font-weight: bold; }}
#loadings .feat {{ color: #666; }}
#loadings .pos {{ color: #4caf50; }}
#loadings .neg {{ color: #ef5350; }}
#sil-chart {{ padding: 4px 16px 8px; border-top: 1px solid #222; }}
#sil-chart canvas {{ width: 100%; height: 80px; }}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <h2>Regions</h2>
    <div class="summary">{len(sorted_regions)} regions</div>
    <input id="search" placeholder="Filter regions..." oninput="filterRegions(this.value)">
    <div style="margin-top:6px;display:flex;gap:4px;flex-wrap:wrap">
      <button class="sort-btn active" data-sort="hopkins-desc" onclick="sortRegions(this)">H ↓</button>
      <button class="sort-btn" data-sort="hopkins-asc" onclick="sortRegions(this)">H ↑</button>
      <button class="sort-btn" data-sort="cells" onclick="sortRegions(this)">Cells ↓</button>
      <button class="sort-btn" data-sort="id" onclick="sortRegions(this)">ID</button>
    </div>
  </div>
  <div id="region-list"></div>
</div>

<div id="spatial-pane">
  <div id="spatial-toolbar">
    <button onclick="fitView()">Fit</button>
    <span>BG:</span>
    <select id="bg-select" onchange="setBg(this.value)"></select>
    <span>Thickness:</span>
    <input type="range" id="thick" min="0.5" max="4" step="0.1" value="1.2" oninput="redrawContours()" style="width:60px">
  </div>
  <div id="canvas-wrap">
    <canvas id="bg-canvas"></canvas>
    <canvas id="contour-canvas"></canvas>
  </div>
</div>

<div id="right-pane">
  <div id="top-bar">
    <div class="metric"><span class="label">Region </span><span class="val" id="m-region">—</span></div>
    <div class="metric"><span class="label">Cells </span><span class="val" id="m-cells">—</span></div>
    <div class="metric"><span class="label">Hopkins </span><span class="val" id="m-hopkins">—</span></div>
    <div class="metric"><span class="label">#cl </span><span class="val" id="m-k">—</span></div>
    <div class="metric"><span class="label">Noise </span><span class="val" id="m-noise">—</span></div>
    <div class="metric"><span class="label">Var </span><span class="val" id="m-var">—</span></div>
    <div class="metric" style="margin-left:auto">
      <span class="label">Color </span>
      <button class="color-btn active" data-color="kmeans" onclick="setColoring(this)">kmeans</button>
      <button class="color-btn" data-color="leiden" onclick="setColoring(this)">leiden</button>
      <button class="color-btn" data-color="hdbscan_pca" onclick="setColoring(this)">hdb·PCA</button>
      <button class="color-btn" data-color="hdbscan_umap" onclick="setColoring(this)">hdb·UMAP</button>
    </div>
  </div>
  <div id="plots">
    <div class="plot-container" style="flex:2"><div class="plot-title">UMAP (on top PCs)</div><canvas id="cumap"></canvas></div>
    <div class="plot-container"><div class="plot-title">PC1 vs PC2</div><canvas id="c12"></canvas></div>
  </div>
  <div id="loadings"></div>
  <div id="sil-chart"><canvas id="sil-canvas"></canvas></div>
</div>

<script>
const REGIONS = {safe_json({str(rid): data for rid, data in sorted_regions})};
const REGION_INFO = {safe_json(region_info)};
const CONTOURS = {safe_json(contour_payload)};
const BGS = {safe_json(dict(fluor_thumbnails))};
const LABEL_SHAPE = [{H}, {W}];  // [height, width] in thumbnail pixels
const CLUSTER_COLORS = [
  '#ff3860','#4dff5e','#4d8bff','#ffab30','#cc4dff',
  '#3ff2ff','#ff4df5','#d7ff4d','#ff7eb0','#3ff2c7'
];
const NOISE_COLOR = '#555';

let currentRegion = null;
let currentSort = 'hopkins-desc';
let currentColoring = 'kmeans';
let currentBg = Object.keys(BGS)[0] || null;

// Spatial pane state
let spatialScale = 1, spatialTx = 0, spatialTy = 0;
let bgImage = null;

// ---------- Sidebar ----------
function sortRegions(btn) {{
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentSort = btn.dataset.sort;
  buildSidebar();
}}
function buildSidebar() {{
  const items = [...REGION_INFO];
  if (currentSort === 'hopkins-desc') items.sort((a, b) => b.hopkins - a.hopkins);
  else if (currentSort === 'hopkins-asc') items.sort((a, b) => a.hopkins - b.hopkins);
  else if (currentSort === 'cells') items.sort((a, b) => b.n_cells - a.n_cells);
  else if (currentSort === 'id') items.sort((a, b) => a.id - b.id);
  const list = document.getElementById('region-list');
  list.innerHTML = '';
  items.forEach(r => {{
    const div = document.createElement('div');
    div.className = 'region-item' + (String(r.id) === currentRegion ? ' active' : '');
    div.dataset.rid = r.id;
    let meta = r.n_cells.toLocaleString() + ' | H=' + r.hopkins;
    if (r.mean_nuc !== undefined) meta += ' | nuc=' + r.mean_nuc;
    div.innerHTML = '<div class="rid">Region ' + r.id + '</div><div class="meta">' + meta + '</div>';
    div.onclick = () => selectRegion(String(r.id) === currentRegion ? null : String(r.id));
    list.appendChild(div);
  }});
}}
function filterRegions(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('.region-item').forEach(el => {{
    el.style.display = el.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

// ---------- BG select ----------
function populateBgSelect() {{
  const sel = document.getElementById('bg-select');
  sel.innerHTML = '';
  Object.keys(BGS).forEach(name => {{
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
  }});
  if (currentBg) sel.value = currentBg;
}}
function setBg(name) {{
  currentBg = name;
  loadBg();
}}
function loadBg() {{
  if (!currentBg || !BGS[currentBg]) return;
  bgImage = new Image();
  bgImage.onload = () => {{ fitView(); }};
  bgImage.src = 'data:image/jpeg;base64,' + BGS[currentBg];
}}

// ---------- Spatial canvas ----------
function fitView() {{
  const wrap = document.getElementById('canvas-wrap');
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const sx = cw / LABEL_SHAPE[1], sy = ch / LABEL_SHAPE[0];
  spatialScale = Math.min(sx, sy) * 0.95;
  spatialTx = (cw - LABEL_SHAPE[1] * spatialScale) / 2;
  spatialTy = (ch - LABEL_SHAPE[0] * spatialScale) / 2;
  redrawSpatial();
}}
function sizeCanvases() {{
  const wrap = document.getElementById('canvas-wrap');
  const dpr = window.devicePixelRatio || 1;
  ['bg-canvas', 'contour-canvas'].forEach(id => {{
    const c = document.getElementById(id);
    c.width = wrap.clientWidth * dpr;
    c.height = wrap.clientHeight * dpr;
    c.style.width = wrap.clientWidth + 'px';
    c.style.height = wrap.clientHeight + 'px';
    c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  }});
}}
function redrawSpatial() {{
  sizeCanvases();
  drawBg();
  drawContours();
}}
function drawBg() {{
  const c = document.getElementById('bg-canvas');
  const ctx = c.getContext('2d');
  const wrap = document.getElementById('canvas-wrap');
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, wrap.clientWidth, wrap.clientHeight);
  if (bgImage && bgImage.complete) {{
    ctx.save();
    ctx.translate(spatialTx, spatialTy);
    ctx.scale(spatialScale, spatialScale);
    ctx.drawImage(bgImage, 0, 0, LABEL_SHAPE[1], LABEL_SHAPE[0]);
    ctx.restore();
  }}
}}
function drawContours() {{
  const c = document.getElementById('contour-canvas');
  const ctx = c.getContext('2d');
  const wrap = document.getElementById('canvas-wrap');
  ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
  const thick = parseFloat(document.getElementById('thick').value);
  ctx.save();
  ctx.translate(spatialTx, spatialTy);
  ctx.scale(spatialScale, spatialScale);

  // First pass: draw non-selected regions (thin HSL outlines)
  ctx.lineWidth = thick / spatialScale;
  for (const [rid, pts] of Object.entries(CONTOURS)) {{
    if (rid === currentRegion) continue;
    const hue = (parseInt(rid) * 137) % 360;
    ctx.strokeStyle = 'hsl(' + hue + ',70%,55%)';
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {{
      if (i === 0) ctx.moveTo(pts[i][0], pts[i][1]);
      else ctx.lineTo(pts[i][0], pts[i][1]);
    }}
    ctx.closePath();
    ctx.stroke();
  }}

  // Second pass: selected region — bold highlight
  if (currentRegion && CONTOURS[currentRegion]) {{
    const pts = CONTOURS[currentRegion];
    // Dim everything else with a semi-transparent overlay
    ctx.restore();
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, wrap.clientWidth, wrap.clientHeight);
    ctx.translate(spatialTx, spatialTy);
    ctx.scale(spatialScale, spatialScale);
    // Filled yellow tint + thick white outer ring + cyan inner
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {{
      if (i === 0) ctx.moveTo(pts[i][0], pts[i][1]);
      else ctx.lineTo(pts[i][0], pts[i][1]);
    }}
    ctx.closePath();
    ctx.fillStyle = 'rgba(255, 235, 59, 0.28)';
    ctx.fill();
    ctx.lineWidth = (thick * 4) / spatialScale;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
    ctx.lineWidth = (thick * 1.8) / spatialScale;
    ctx.strokeStyle = '#00e5ff';
    ctx.stroke();
  }}
  ctx.restore();
}}

function redrawContours() {{ drawContours(); }}

// Click-on-region: point-in-polygon
function pointInPoly(x, y, pts) {{
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {{
    const xi = pts[i][0], yi = pts[i][1];
    const xj = pts[j][0], yj = pts[j][1];
    const intersect = ((yi > y) !== (yj > y)) &&
      (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi);
    if (intersect) inside = !inside;
  }}
  return inside;
}}
function onSpatialClick(e) {{
  const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
  const mx = e.clientX - wrap.left;
  const my = e.clientY - wrap.top;
  const lx = (mx - spatialTx) / spatialScale;
  const ly = (my - spatialTy) / spatialScale;
  let best = null, bestArea = Infinity;
  for (const [rid, pts] of Object.entries(CONTOURS)) {{
    if (pointInPoly(lx, ly, pts)) {{
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (const p of pts) {{
        if (p[0] < xmin) xmin = p[0]; if (p[0] > xmax) xmax = p[0];
        if (p[1] < ymin) ymin = p[1]; if (p[1] > ymax) ymax = p[1];
      }}
      const a = (xmax - xmin) * (ymax - ymin);
      if (a < bestArea) {{ best = rid; bestArea = a; }}
    }}
  }}
  if (best === null || best === currentRegion) selectRegion(null);
  else selectRegion(best);
}}
// Pan/zoom
let drag = null;
function onSpatialWheel(e) {{
  e.preventDefault();
  const wrap = document.getElementById('canvas-wrap').getBoundingClientRect();
  const mx = e.clientX - wrap.left, my = e.clientY - wrap.top;
  const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
  spatialTx = mx - (mx - spatialTx) * factor;
  spatialTy = my - (my - spatialTy) * factor;
  spatialScale *= factor;
  redrawSpatial();
}}
function onSpatialMouseDown(e) {{
  drag = {{ x: e.clientX, y: e.clientY, tx: spatialTx, ty: spatialTy, moved: false }};
}}
function onSpatialMouseMove(e) {{
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) drag.moved = true;
  spatialTx = drag.tx + dx; spatialTy = drag.ty + dy;
  redrawSpatial();
}}
function onSpatialMouseUp(e) {{ drag = null; }}

// ---------- UMAP pane (right) ----------
function setColoring(btn) {{
  document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentColoring = btn.dataset.color;
  if (currentRegion) selectRegion(currentRegion);
}}
function labelsFor(d) {{
  if (currentColoring === 'leiden') return d.labels_leiden || d.labels;
  if (currentColoring === 'hdbscan_pca') return d.labels_hdbscan_pca || d.labels;
  if (currentColoring === 'hdbscan_umap') return d.labels_hdbscan_umap || d.labels;
  return d.labels_kmeans || d.labels;
}}

function selectRegion(rid) {{
  currentRegion = rid;
  document.querySelectorAll('.region-item').forEach(el => {{
    el.classList.toggle('active', rid !== null && el.dataset.rid === rid);
  }});
  drawContours();
  if (rid === null) {{
    ['m-region','m-cells','m-hopkins','m-k','m-noise','m-var'].forEach(id => {{
      document.getElementById(id).textContent = '—';
    }});
    ['cumap','c12','sil-canvas'].forEach(id => {{
      const c = document.getElementById(id);
      if (c) {{ const ctx = c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle='#0d0d0d'; ctx.fillRect(0,0,c.width,c.height); }}
    }});
    document.getElementById('loadings').innerHTML = '';
    return;
  }}
  const d = REGIONS[rid];
  if (!d) return;
  document.getElementById('m-region').textContent = rid;
  document.getElementById('m-cells').textContent = d.n_cells.toLocaleString();
  document.getElementById('m-hopkins').textContent = d.hopkins;
  const nc = (d.n_clusters || {{}})[currentColoring];
  const nn = (d.n_noise || {{}})[currentColoring] || 0;
  document.getElementById('m-k').textContent = (nc !== undefined ? nc : '—') + ' (' + currentColoring + ')';
  document.getElementById('m-noise').textContent = nn > 0 ? nn + ' (' + (100*nn/d.n_plotted).toFixed(1) + '%)' : '—';
  document.getElementById('m-var').textContent = d.var_explained.map(v => (v*100).toFixed(1)+'%').join('/');

  const labs = labelsFor(d);
  drawScatter('cumap', d.umap_x, d.umap_y, labs, 'UMAP1', 'UMAP2');
  drawScatter('c12', d.pc1, d.pc2, labs, 'PC1', 'PC2');
  drawLoadings(d.top_loadings);
  const silDiv = document.getElementById('sil-chart');
  if (currentColoring === 'kmeans') {{
    silDiv.style.display = '';
    drawElbowChart(d.inertia_per_k, d.silhouette_per_k, d.best_k);
  }} else {{
    silDiv.style.display = 'none';
  }}
}}

function drawScatter(canvasId, xs, ys, labels, xLabel, yLabel) {{
  const canvas = document.getElementById(canvasId);
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height, pad = 30;
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, w, h);
  if (!xs || xs.length === 0) return;
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < xs.length; i++) {{
    if (xs[i] < xMin) xMin = xs[i]; if (xs[i] > xMax) xMax = xs[i];
    if (ys[i] < yMin) yMin = ys[i]; if (ys[i] > yMax) yMax = ys[i];
  }}
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
  const sx = v => pad + (v - xMin) / xRange * (w - 2*pad);
  const sy = v => h - pad - (v - yMin) / yRange * (h - 2*pad);
  ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const gx = pad + i*(w-2*pad)/4, gy = pad + i*(h-2*pad)/4;
    ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, h-pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w-pad, gy); ctx.stroke();
  }}
  const alpha = xs.length > 2000 ? 0.65 : xs.length > 500 ? 0.8 : 0.9;
  const radius = xs.length > 2000 ? 1.5 : xs.length > 500 ? 2.0 : 2.8;
  for (let i = 0; i < xs.length; i++) {{
    const lbl = labels[i];
    if (lbl < 0) {{ ctx.fillStyle = NOISE_COLOR; ctx.globalAlpha = alpha*0.6; }}
    else {{ ctx.fillStyle = CLUSTER_COLORS[lbl % CLUSTER_COLORS.length]; ctx.globalAlpha = alpha; }}
    ctx.beginPath();
    ctx.arc(sx(xs[i]), sy(ys[i]), radius, 0, Math.PI*2);
    ctx.fill();
  }}
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#666'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  ctx.fillText(xLabel, w/2, h-4);
  ctx.save(); ctx.translate(10, h/2); ctx.rotate(-Math.PI/2); ctx.fillText(yLabel, 0, 0); ctx.restore();
}}
function drawLoadings(loadings) {{
  const el = document.getElementById('loadings');
  el.innerHTML = '';
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  for (const [pc, feats] of Object.entries(loadings)) {{
    const row = document.createElement('div');
    row.className = 'pc-row';
    let html = '<span class="pc-label">' + esc(pc) + ':</span> ';
    html += feats.map(f => {{
      const cls = f.loading >= 0 ? 'pos' : 'neg';
      return '<span class="feat">' + esc(f.feature) + '</span>(<span class="' + cls + '">' + f.loading.toFixed(2) + '</span>)';
    }}).join(' ');
    row.innerHTML = html;
    el.appendChild(row);
  }}
}}
function drawElbowChart(inertiaPerK, silPerK, bestK) {{
  const canvas = document.getElementById('sil-canvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const H = 80;
  canvas.width = rect.width * dpr; canvas.height = H * dpr;
  canvas.style.width = rect.width + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = H;
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, w, h);
  const ks = Object.keys(inertiaPerK).map(Number).sort((a,b) => a-b);
  if (ks.length === 0) return;
  const ins = ks.map(k => inertiaPerK[k]);
  let insMin = Infinity, insMax = -Infinity;
  for (const v of ins) {{ if (v < insMin) insMin = v; if (v > insMax) insMax = v; }}
  const insRange = insMax - insMin || 1;
  const padL = 70, padR = 20, padT = 14, padB = 16;
  const plotW = w - padL - padR, plotH = h - padT - padB;
  const sx = i => padL + (ks.length > 1 ? (i/(ks.length-1))*plotW : plotW/2);
  const sy = v => padT + (1 - (v - insMin)/insRange) * plotH;
  ctx.strokeStyle = '#3a3a3a'; ctx.lineWidth = 1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(sx(0), sy(ins[0])); ctx.lineTo(sx(ks.length-1), sy(ins[ks.length-1])); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ks.forEach((k, i) => {{ const x = sx(i), y = sy(ins[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }});
  ctx.stroke();
  ks.forEach((k, i) => {{
    const x = sx(i), y = sy(ins[i]);
    const isBest = k === bestK;
    ctx.fillStyle = isBest ? '#4caf50' : '#888';
    ctx.beginPath(); ctx.arc(x, y, isBest ? 5 : 3, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = isBest ? '#4caf50' : '#666';
    ctx.font = (isBest ? 'bold ' : '') + '9px monospace'; ctx.textAlign = 'center';
    ctx.fillText('k=' + k, x, h-4);
    if (isBest) ctx.fillText('ELBOW', x, y-8);
  }});
  ctx.fillStyle = '#888'; ctx.font = '9px monospace'; ctx.textAlign = 'left';
  ctx.fillText('Inertia by k (elbow method)', 2, 10);
  const bestSil = silPerK[bestK];
  if (bestSil !== undefined) ctx.fillText('sil@elbow: ' + bestSil.toFixed(2), 2, 22);
}}

// ---------- Wiring ----------
window.addEventListener('resize', () => {{ fitView(); if (currentRegion) selectRegion(currentRegion); }});
const cwrap = document.getElementById('canvas-wrap');
cwrap.addEventListener('wheel', onSpatialWheel, {{ passive: false }});
cwrap.addEventListener('click', onSpatialClick);
cwrap.addEventListener('mousedown', onSpatialMouseDown);
window.addEventListener('mousemove', onSpatialMouseMove);
window.addEventListener('mouseup', onSpatialMouseUp);

window.addEventListener('keydown', e => {{ if (e.key === 'Escape') selectRegion(null); }});
populateBgSelect();
buildSidebar();
loadBg();
setTimeout(() => {{ if (REGION_INFO.length > 0) selectRegion(String(REGION_INFO[0].id)); }}, 100);
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    size_mb = Path(output_path).stat().st_size / 1e6
    logger.info("Wrote %s (%d regions, %.1f MB)", output_path, len(sorted_regions), size_mb)


def main():
    args = parse_args()
    rng = np.random.default_rng(42)

    logger.info("Loading label map from %s ...", args.label_map)
    label_map = np.load(args.label_map)
    logger.info("Label map shape: %s, %d regions", label_map.shape, len(np.unique(label_map)) - 1)

    logger.info("Extracting per-region contours...")
    contours = extract_region_contours(label_map)
    logger.info("Extracted contours for %d regions", len(contours))

    channels = [int(c.strip()) for c in args.display_channels.split(",")]
    logger.info("Loading CZI thumbnails (channels=%s, scale=%.4f)...", channels, args.scale)
    fluor_thumbnails = build_fluor_thumbnails(args.czi_path, channels, args.scale, args.scene)
    logger.info("Built %d background thumbnails", len(fluor_thumbnails))

    logger.info("Loading detections from %s ...", args.detections)
    detections = fast_json_load(args.detections)
    logger.info("Loaded %d detections", len(detections))

    # Group by organ, filter to nucleated, min-cells
    regions: dict[int, list] = defaultdict(list)
    for det in detections:
        oid = det.get("organ_id", 0)
        if oid == 0:
            continue
        nn = det.get("features", {}).get("n_nuclei")
        try:
            nn_val = float(nn) if nn is not None else None
        except (TypeError, ValueError):
            continue
        if nn_val is None or not math.isfinite(nn_val) or int(nn_val) < 1:
            continue
        regions[oid].append(det)
    del detections
    logger.info("Found %d regions with nucleated cells", len(regions))

    regions = {rid: dets for rid, dets in regions.items() if len(dets) >= args.min_cells}
    logger.info("%d regions with >= %d nucleated cells", len(regions), args.min_cells)

    feature_groups = {g.strip() for g in args.feature_groups.split(",")}
    sample_dets = next(iter(regions.values()))
    feature_names = select_feature_names(sample_dets, feature_groups)
    logger.info("Selected %d features from groups %s", len(feature_names), sorted(feature_groups))

    region_data = {}
    for i, (rid, dets) in enumerate(sorted(regions.items())):
        logger.info(
            "Processing region %d (%d/%d, %d cells)...", rid, i + 1, len(regions), len(dets)
        )
        result = process_region(
            dets,
            feature_names,
            args.max_k,
            args.max_points_plot,
            rng,
            seed=args.seed,
            var_cutoff=args.var_cutoff,
            max_pcs=args.max_pcs,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            leiden_resolution=args.leiden_resolution,
            leiden_knn=args.leiden_knn,
            hdbscan_min_size=args.hdbscan_min_size,
        )
        if result is not None:
            region_data[rid] = result

    logger.info("Processed %d regions", len(region_data))

    # Intersect: only keep regions that have BOTH contours AND PCA data
    keep = set(region_data.keys()) & set(contours.keys())
    region_data = {rid: region_data[rid] for rid in keep}
    contours = {rid: contours[rid] for rid in keep}
    logger.info("Final viewer has %d regions (intersection of contours + feature data)", len(keep))

    nuc_stats = None
    if args.nuc_stats:
        with open(args.nuc_stats) as f:
            raw = json.load(f)
        nuc_stats = {int(k): v for k, v in raw.items()}

    generate_combined_html(
        region_data,
        contours,
        fluor_thumbnails,
        label_map.shape,
        args.output,
        nuc_stats=nuc_stats,
    )


if __name__ == "__main__":
    main()
