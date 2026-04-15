#!/usr/bin/env python
"""Per-region PCA + UMAP scatter plots with four clustering methods.

For each organ region with enough nucleated cells:
1. PCA reduces features to enough PCs to capture --var-cutoff (default 90%)
2. UMAP projects those PCs to 2D for visualization
3. Four clusterings are computed, selectable in the viewer:
   - K-means (elbow method on inertia) on PCA space — linear/convex
   - Leiden on kNN graph built from PCA — non-linear, scverse-standard
   - HDBSCAN on PCA space — density-based, principled
   - HDBSCAN on UMAP space — density-based, matches visual lobes

Generates an interactive HTML with UMAP (primary) and PC1 vs PC2 (reference).
Coloring toggle switches between the four methods. HDBSCAN noise → gray.

Metrics per region:
- n_pcs_90: PCs needed for --var-cutoff (intrinsic dimensionality proxy)
- Hopkins statistic: clustering tendency (0.5 = random, >0.7 = clustered)
- Silhouette score: cohesion vs separation for best k
- Calinski-Harabasz: between-vs-within cluster variance ratio

Usage:
    python scripts/region_pca_viewer.py \
        --detections cell_detections_with_organs.json \
        --min-cells 1000 \
        --feature-groups morph,channel \
        --var-cutoff 0.90 \
        --output region_pca_viewer.html
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.analysis.cluster_features import select_feature_names
from xldvp_seg.analysis.region_clustering import process_region
from xldvp_seg.utils.json_utils import fast_json_load
from xldvp_seg.utils.logging import get_logger
from xldvp_seg.visualization.encoding import safe_json

logger = get_logger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections", required=True, help="Detections JSON with organ_id")
    parser.add_argument(
        "--min-cells", type=int, default=1000, help="Min nucleated cells per region (default: 1000)"
    )
    parser.add_argument(
        "--feature-groups",
        default="morph,channel",
        help="Comma-separated feature groups (default: morph,channel)",
    )
    parser.add_argument(
        "--max-k", type=int, default=8, help="Max k for k-means search (default: 8)"
    )
    parser.add_argument(
        "--var-cutoff",
        type=float,
        default=0.90,
        help="PCA cumulative variance cutoff (default: 0.90)",
    )
    parser.add_argument(
        "--max-pcs",
        type=int,
        default=50,
        help="Hard cap on PCs to feed UMAP (default: 50)",
    )
    parser.add_argument(
        "--umap-neighbors", type=int, default=15, help="UMAP n_neighbors (default: 15)"
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)"
    )
    parser.add_argument(
        "--leiden-resolution",
        type=float,
        default=1.0,
        help="Leiden resolution (higher = more clusters; default: 1.0)",
    )
    parser.add_argument(
        "--leiden-knn", type=int, default=15, help="Leiden kNN graph neighbors (default: 15)"
    )
    parser.add_argument(
        "--hdbscan-min-size",
        type=int,
        default=50,
        help="HDBSCAN min_cluster_size (default: 50)",
    )
    parser.add_argument(
        "--max-points-plot",
        type=int,
        default=5000,
        help="Max points per region in HTML (subsample for browser perf, default: 5000)",
    )
    parser.add_argument("--nuc-stats", help="Pre-computed region_nuc_stats.json for sidebar info")
    parser.add_argument("--output", required=True, help="Output HTML path")
    return parser.parse_args(argv)


def generate_pca_html(region_data, output_path, nuc_stats=None):
    """Generate interactive HTML with per-region PCA scatter plots."""
    # Sort regions by cell count descending
    sorted_regions = sorted(region_data.items(), key=lambda x: -x[1]["n_cells"])

    # Build region info for sidebar
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

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Region PCA Viewer — {len(sorted_regions)} regions</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; display: flex; height: 100vh; overflow: hidden; }}
#sidebar {{ width: 280px; min-width: 280px; background: #111; border-right: 1px solid #333; display: flex; flex-direction: column; }}
#sidebar-header {{ padding: 12px; border-bottom: 1px solid #333; }}
#sidebar-header h2 {{ font-size: 14px; color: #fff; margin-bottom: 6px; }}
#sidebar-header .summary {{ font-size: 11px; color: #888; }}
#search {{ width: 100%; padding: 6px 8px; background: #1a1a1a; border: 1px solid #333; color: #e0e0e0; font-family: inherit; font-size: 11px; margin-top: 6px; }}
.sort-btn, .color-btn {{ padding: 3px 8px; background: #1a1a1a; border: 1px solid #333; color: #888; font-family: inherit; font-size: 10px; cursor: pointer; border-radius: 3px; }}
.sort-btn:hover, .color-btn:hover {{ background: #222; }}
.sort-btn.active, .color-btn.active {{ background: #1a2a1a; border-color: #4caf50; color: #4caf50; }}
.color-btn {{ margin-left: 2px; }}
#region-list {{ flex: 1; overflow-y: auto; }}
.region-item {{ padding: 8px 12px; border-bottom: 1px solid #222; cursor: pointer; transition: background 0.15s; }}
.region-item:hover {{ background: #1a1a1a; }}
.region-item.active {{ background: #1a2a1a; border-left: 3px solid #4caf50; }}
.region-item .rid {{ font-weight: bold; color: #fff; }}
.region-item .meta {{ font-size: 10px; color: #888; margin-top: 2px; }}
.region-item .hopkins-bar {{ display: inline-block; height: 4px; background: #4caf50; margin-left: 4px; vertical-align: middle; }}
#main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
#top-bar {{ padding: 10px 16px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
#top-bar .metric {{ font-size: 11px; }}
#top-bar .metric .val {{ font-weight: bold; color: #fff; }}
#top-bar .metric .label {{ color: #888; }}
#plots {{ flex: 1; display: flex; gap: 4px; padding: 8px; overflow: hidden; }}
.plot-container {{ flex: 1; position: relative; background: #0d0d0d; border: 1px solid #222; border-radius: 4px; }}
.plot-container canvas {{ width: 100%; height: 100%; }}
.plot-title {{ position: absolute; top: 6px; left: 50%; transform: translateX(-50%); font-size: 11px; color: #aaa; pointer-events: none; z-index: 1; }}
#loadings {{ padding: 8px 16px; border-top: 1px solid #333; font-size: 10px; color: #888; max-height: 80px; overflow-y: auto; }}
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
    <h2>Region PCA</h2>
    <div class="summary">{len(sorted_regions)} regions, morph+channel features</div>
    <input id="search" placeholder="Filter regions..." oninput="filterRegions(this.value)">
    <div style="margin-top:6px;display:flex;gap:4px">
      <button class="sort-btn active" data-sort="hopkins-desc" onclick="sortRegions(this)">Hopkins ↓</button>
      <button class="sort-btn" data-sort="hopkins-asc" onclick="sortRegions(this)">Hopkins ↑</button>
      <button class="sort-btn" data-sort="cells" onclick="sortRegions(this)">Cells ↓</button>
      <button class="sort-btn" data-sort="sil" onclick="sortRegions(this)">Sil ↓</button>
    </div>
  </div>
  <div id="region-list"></div>
</div>
<div id="main">
  <div id="top-bar">
    <div class="metric"><span class="label">Region </span><span class="val" id="m-region">—</span></div>
    <div class="metric"><span class="label">Cells </span><span class="val" id="m-cells">—</span></div>
    <div class="metric"><span class="label">Hopkins </span><span class="val" id="m-hopkins">—</span></div>
    <div class="metric"><span class="label">Silhouette </span><span class="val" id="m-sil">—</span></div>
    <div class="metric"><span class="label">Best k </span><span class="val" id="m-k">—</span></div>
    <div class="metric"><span class="label">CH </span><span class="val" id="m-ch">—</span></div>
    <div class="metric"><span class="label">PCs@var </span><span class="val" id="m-npcs">—</span></div>
    <div class="metric"><span class="label">Var PC1/2/3 </span><span class="val" id="m-var">—</span></div>
    <div class="metric"><span class="label">Noise </span><span class="val" id="m-noise">—</span></div>
    <div class="metric" style="margin-left:auto">
      <span class="label">Color </span>
      <button class="color-btn active" data-color="kmeans" onclick="setColoring(this)">kmeans</button>
      <button class="color-btn" data-color="leiden" onclick="setColoring(this)">leiden</button>
      <button class="color-btn" data-color="hdbscan_pca" onclick="setColoring(this)">hdbscan·PCA</button>
      <button class="color-btn" data-color="hdbscan_umap" onclick="setColoring(this)">hdbscan·UMAP</button>
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
const CLUSTER_COLORS = [
  '#ff3860','#4dff5e','#4d8bff','#ffab30','#cc4dff',
  '#3ff2ff','#ff4df5','#d7ff4d','#ff7eb0','#3ff2c7'
];

let currentRegion = null;
let currentSort = 'hopkins-desc';
let currentColoring = 'kmeans';
const NOISE_COLOR = '#555';  // HDBSCAN -1 noise points

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

function sortRegions(btn) {{
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentSort = btn.dataset.sort;
  const sorted = [...REGION_INFO];
  if (currentSort === 'hopkins-desc') sorted.sort((a, b) => b.hopkins - a.hopkins);
  else if (currentSort === 'hopkins-asc') sorted.sort((a, b) => a.hopkins - b.hopkins);
  else if (currentSort === 'cells') sorted.sort((a, b) => b.n_cells - a.n_cells);
  else if (currentSort === 'sil') sorted.sort((a, b) => b.silhouette - a.silhouette);
  buildSidebar(sorted);
}}

function buildSidebar(items) {{
  if (!items) {{
    items = [...REGION_INFO];
    items.sort((a, b) => b.hopkins - a.hopkins);
  }}
  const list = document.getElementById('region-list');
  list.innerHTML = '';
  items.forEach(r => {{
    const div = document.createElement('div');
    div.className = 'region-item';
    div.dataset.rid = r.id;
    const hopkinsW = Math.round(r.hopkins * 40);
    const hopkinsColor = r.hopkins > 0.75 ? '#4caf50' : r.hopkins > 0.6 ? '#ff9800' : '#666';
    let meta = r.n_cells.toLocaleString() + ' cells | H=' + r.hopkins + ' | k=' + r.best_k;
    if (r.mean_nuc !== undefined) meta += ' | nuc=' + r.mean_nuc;
    div.innerHTML = '<div class="rid">Region ' + r.id + '</div>' +
      '<div class="meta">' + meta +
      ' <span class="hopkins-bar" style="width:' + hopkinsW + 'px;background:' + hopkinsColor + '"></span></div>';
    div.onclick = () => selectRegion(String(r.id));
    list.appendChild(div);
  }});
}}

function filterRegions(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('.region-item').forEach(el => {{
    el.style.display = el.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

function selectRegion(rid) {{
  currentRegion = rid;
  document.querySelectorAll('.region-item').forEach(el => {{
    el.classList.toggle('active', el.dataset.rid === rid);
  }});
  const d = REGIONS[rid];
  if (!d) return;
  document.getElementById('m-region').textContent = rid;
  document.getElementById('m-cells').textContent = d.n_cells.toLocaleString() + ' (' + d.n_plotted + ' plotted)';
  document.getElementById('m-hopkins').textContent = d.hopkins;
  document.getElementById('m-hopkins').style.color = d.hopkins > 0.75 ? '#4caf50' : d.hopkins > 0.6 ? '#ff9800' : '#ef5350';
  document.getElementById('m-sil').textContent = d.silhouette;
  document.getElementById('m-k').textContent = d.best_k;
  document.getElementById('m-ch').textContent = d.calinski_harabasz.toLocaleString();
  document.getElementById('m-npcs').textContent = d.n_pcs_used;
  document.getElementById('m-var').textContent = d.var_explained.map(v => (v * 100).toFixed(1) + '%').join(' / ');

  // Method-aware cluster count + noise
  const nClusters = (d.n_clusters || {{}})[currentColoring];
  const nNoise = (d.n_noise || {{}})[currentColoring] || 0;
  const nPlotted = d.n_plotted || 0;
  if (nClusters !== undefined) {{
    document.getElementById('m-k').textContent = nClusters + ' (' + currentColoring + ')';
  }}
  document.getElementById('m-noise').textContent =
    nNoise > 0 ? nNoise + ' (' + (100 * nNoise / nPlotted).toFixed(1) + '%)' : '—';

  const labs = labelsFor(d);
  drawScatter('cumap', d.umap_x, d.umap_y, labs, 'UMAP1', 'UMAP2');
  drawScatter('c12', d.pc1, d.pc2, labs, 'PC1', 'PC2');
  drawLoadings(d.top_loadings);
  // Elbow chart is only meaningful for k-means; hide otherwise
  const silChartDiv = document.getElementById('sil-chart');
  if (currentColoring === 'kmeans') {{
    silChartDiv.style.display = '';
    drawElbowChart(d.inertia_per_k, d.silhouette_per_k, d.best_k);
  }} else {{
    silChartDiv.style.display = 'none';
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
  const w = rect.width, h = rect.height;
  const pad = 30;

  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, w, h);

  if (!xs || xs.length === 0) return;

  // Avoid Math.min/max(...arr) — stack overflow on large arrays
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < xs.length; i++) {{
    if (xs[i] < xMin) xMin = xs[i]; if (xs[i] > xMax) xMax = xs[i];
    if (ys[i] < yMin) yMin = ys[i]; if (ys[i] > yMax) yMax = ys[i];
  }}
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const sx = (v) => pad + (v - xMin) / xRange * (w - 2 * pad);
  const sy = (v) => h - pad - (v - yMin) / yRange * (h - 2 * pad);

  // Grid lines
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const gx = pad + i * (w - 2 * pad) / 4;
    const gy = pad + i * (h - 2 * pad) / 4;
    ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, h - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w - pad, gy); ctx.stroke();
  }}

  // Points — HDBSCAN noise (label < 0) rendered in gray
  const alpha = xs.length > 2000 ? 0.65 : xs.length > 500 ? 0.8 : 0.9;
  const radius = xs.length > 2000 ? 1.5 : xs.length > 500 ? 2.0 : 2.8;
  for (let i = 0; i < xs.length; i++) {{
    const lbl = labels[i];
    if (lbl < 0) {{
      ctx.fillStyle = NOISE_COLOR;
      ctx.globalAlpha = alpha * 0.6;  // dim noise further
    }} else {{
      ctx.fillStyle = CLUSTER_COLORS[lbl % CLUSTER_COLORS.length];
      ctx.globalAlpha = alpha;
    }}
    ctx.beginPath();
    ctx.arc(sx(xs[i]), sy(ys[i]), radius, 0, Math.PI * 2);
    ctx.fill();
  }}
  ctx.globalAlpha = 1;

  // Axis labels
  ctx.fillStyle = '#666';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, w / 2, h - 4);
  ctx.save();
  ctx.translate(10, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
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
  canvas.width = rect.width * dpr;
  canvas.height = H * dpr;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = H;

  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, w, h);

  const ks = Object.keys(inertiaPerK).map(Number).sort((a, b) => a - b);
  if (ks.length === 0) return;
  const ins = ks.map(k => inertiaPerK[k]);

  let insMin = Infinity, insMax = -Infinity;
  for (const v of ins) {{ if (v < insMin) insMin = v; if (v > insMax) insMax = v; }}
  const insRange = insMax - insMin || 1;

  const padL = 70, padR = 20, padT = 14, padB = 16;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const sx = (i) => padL + (ks.length > 1 ? (i / (ks.length - 1)) * plotW : plotW / 2);
  const sy = (v) => padT + (1 - (v - insMin) / insRange) * plotH;

  // Label + line connecting endpoints (the "elbow line")
  ctx.strokeStyle = '#3a3a3a';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(sx(0), sy(ins[0]));
  ctx.lineTo(sx(ks.length - 1), sy(ins[ks.length - 1]));
  ctx.stroke();
  ctx.setLineDash([]);

  // Inertia curve
  ctx.strokeStyle = '#4caf50';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ks.forEach((k, i) => {{
    const x = sx(i), y = sy(ins[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Points + k labels
  ks.forEach((k, i) => {{
    const x = sx(i), y = sy(ins[i]);
    const isBest = k === bestK;
    ctx.fillStyle = isBest ? '#4caf50' : '#888';
    ctx.beginPath();
    ctx.arc(x, y, isBest ? 5 : 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = isBest ? '#4caf50' : '#666';
    ctx.font = (isBest ? 'bold ' : '') + '9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('k=' + k, x, h - 4);
    if (isBest) {{
      ctx.fillStyle = '#4caf50';
      ctx.fillText('ELBOW', x, y - 8);
    }}
  }});

  // Title + silhouette hint for best k
  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('Inertia (WCSS) by k — elbow method', 2, 10);
  const bestSil = silPerK[bestK];
  if (bestSil !== undefined) {{
    ctx.fillText('sil@elbow: ' + bestSil.toFixed(2), 2, 22);
  }}
}}

// Resize handler
window.addEventListener('resize', () => {{
  if (currentRegion) selectRegion(currentRegion);
}});

buildSidebar();
// Auto-select first region
if (REGION_INFO.length > 0) selectRegion(String(REGION_INFO[0].id));
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

    # Load detections
    logger.info("Loading detections from %s ...", args.detections)
    detections = fast_json_load(args.detections)
    logger.info("Loaded %d detections", len(detections))

    # Group by organ_id, filter to nucleated cells
    regions = defaultdict(list)
    for det in detections:
        oid = det.get("organ_id", 0)
        if oid == 0:
            continue
        nn = det.get("features", {}).get("n_nuclei")
        if nn is None or int(nn) < 1:
            continue
        regions[oid].append(det)
    del detections  # Free ~60GB — detections are now partitioned into regions
    logger.info("Found %d regions with nucleated cells", len(regions))

    # Filter by min-cells
    regions = {rid: dets for rid, dets in regions.items() if len(dets) >= args.min_cells}
    logger.info("%d regions with >= %d nucleated cells", len(regions), args.min_cells)

    # Select features from first region's detections
    feature_groups = {g.strip() for g in args.feature_groups.split(",")}
    sample_dets = next(iter(regions.values()))
    feature_names = select_feature_names(sample_dets, feature_groups)
    logger.info("Selected %d features from groups %s", len(feature_names), sorted(feature_groups))

    # Process each region
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
        else:
            logger.warning("Region %d: too few valid cells, skipping", rid)

    logger.info("Processed %d regions successfully", len(region_data))

    # Load optional nuc stats
    nuc_stats = None
    if args.nuc_stats:
        with open(args.nuc_stats) as f:
            raw = json.load(f)
        nuc_stats = {int(k): v for k, v in raw.items()}

    # Generate HTML
    generate_pca_html(region_data, args.output, nuc_stats=nuc_stats)

    # Print summary sorted by Hopkins (most clustered first)
    logger.info("")
    logger.info("=== Clustering tendency summary (sorted by Hopkins) ===")
    by_hopkins = sorted(region_data.items(), key=lambda x: -x[1]["hopkins"])
    for rid, d in by_hopkins[:20]:
        ve = d["var_explained"]
        var_str = "+".join(f"{v * 100:.1f}%" for v in ve[:3])
        logger.info(
            "  Region %3d: %5d cells | Hopkins=%.3f | Sil=%.3f | k=%d | Var=%s",
            rid,
            d["n_cells"],
            d["hopkins"],
            d["silhouette"],
            d["best_k"],
            var_str,
        )
    if len(by_hopkins) > 20:
        logger.info("  ... and %d more regions", len(by_hopkins) - 20)

    # Summary stats
    hopkins_vals = [d["hopkins"] for d in region_data.values()]
    logger.info("")
    logger.info(
        "Hopkins: mean=%.3f, median=%.3f, min=%.3f, max=%.3f",
        np.mean(hopkins_vals),
        np.median(hopkins_vals),
        np.min(hopkins_vals),
        np.max(hopkins_vals),
    )
    highly_clustered = sum(1 for h in hopkins_vals if h > 0.75)
    logger.info(
        "%d/%d regions (%.0f%%) have Hopkins > 0.75 (strong clustering)",
        highly_clustered,
        len(hopkins_vals),
        100 * highly_clustered / len(hopkins_vals) if hopkins_vals else 0,
    )


if __name__ == "__main__":
    main()
