#!/usr/bin/env python
"""Global feature-space clustering + spatial divergence analysis.

Inverse of `combined_region_viewer.py`: clusters ALL nucleated cells
(``n_nuclei >= 1``) in feature space globally, then looks at each cluster's
spatial distribution across organ regions.

Goal: find clusters that look the same in feature space but live in
multiple distinct organ regions (same morphology, different anatomy).

Pipeline:
1. Load detections, keep ``n_nuclei >= 1``, group features across ALL regions.
2. Drop zero-variance features, StandardScaler, PCA to ``--var-cutoff`` (default 90%).
3. Leiden on PCA k-NN graph (full set, via pynndescent for scale).
4. UMAP on a subsample for visualization.
5. Per-cluster: region distribution, entropy, #major regions (cells ≥10% of cluster).
6. Divergence score: entropy × ``n_major_regions`` — high = multi-anatomy cluster.

Output: interactive HTML viewer with:
- Left: global UMAP (subsample) colored by cluster; sortable sidebar.
- Right: spatial region map. Click a cluster → regions highlighted by density.
- Ranked table: top N clusters by divergence (most spatially multi-modal).

Usage:
    python scripts/global_cluster_spatial_viewer.py \\
        --detections cell_detections_with_organs.json \\
        --label-map labels_*_filled.npy \\
        --czi-path slide.czi --display-channels "4,2" \\
        --feature-groups morph,channel \\
        --var-cutoff 0.90 \\
        --leiden-resolution 1.0 \\
        --output global_cluster_viewer.html
"""

import argparse
import base64
import io
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.analysis.cluster_features import (  # noqa: E402
    _extract_feature_matrix,
    select_feature_names,
)
from xldvp_seg.analysis.region_clustering import (  # noqa: E402
    find_optimal_k_elbow,
)
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
    parser.add_argument("--display-channels", default="4,2", help="Channels for background")
    parser.add_argument("--scale", type=float, default=1 / 256)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument(
        "--min-nuclei", type=int, default=1, help="Keep cells with n_nuclei >= this (default 1)"
    )
    parser.add_argument(
        "--min-cells-per-region",
        type=int,
        default=100,
        help="Skip regions with fewer cells (default 100)",
    )
    parser.add_argument("--feature-groups", default="morph,channel")
    parser.add_argument("--var-cutoff", type=float, default=0.90)
    parser.add_argument("--max-pcs", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-knn", type=int, default=15)
    parser.add_argument("--max-k", type=int, default=8, help="Max k for k-means elbow")
    parser.add_argument("--hdbscan-min-size", type=int, default=500)
    parser.add_argument(
        "--max-umap-points",
        type=int,
        default=30000,
        help="Subsample global UMAP to this many points (browser perf)",
    )
    parser.add_argument(
        "--major-region-threshold",
        type=float,
        default=0.10,
        help="A region is 'major' for a cluster if it holds >= this fraction of the cluster (default 0.10)",
    )
    parser.add_argument("--nuc-stats", help="Optional region_nuc_stats.json for display")
    parser.add_argument("--output", required=True, help="Output HTML path")

    # --- Rare-cell-population mode (optional) ---
    rare = parser.add_argument_group(
        "rare-mode",
        "Visualize pre-computed rare cell populations from "
        "`xlseg discover-rare-cells`. When --rare-mode is set, the viewer "
        "skips its own clustering and reads `rare_pop_id` from detections.",
    )
    rare.add_argument(
        "--rare-mode",
        action="store_true",
        help="Enable rare-mode viewer (reads rare_pop_id from detections).",
    )
    rare.add_argument(
        "--rare-cluster-summary",
        type=Path,
        default=None,
        help="Path to cluster_summary.csv produced by discover_rare_cell_types.py. "
        "Required when --rare-mode.",
    )
    rare.add_argument(
        "--rare-linkage-matrix",
        type=Path,
        default=None,
        help="Optional scipy linkage matrix (.npy) on cluster centroids for an "
        "inline-SVG dendrogram. Preferred over --rare-dendrogram-png.",
    )
    rare.add_argument(
        "--rare-cluster-ids",
        type=Path,
        default=None,
        help="Optional cluster_ids.npy (matching the linkage matrix order).",
    )
    rare.add_argument(
        "--rare-dendrogram-png",
        type=Path,
        default=None,
        help="Optional fallback pre-rendered dendrogram PNG if linkage-matrix unavailable.",
    )

    args = parser.parse_args(argv)

    if args.rare_mode and args.rare_cluster_summary is None:
        parser.error("--rare-mode requires --rare-cluster-summary <cluster_summary.csv>")

    return args


def leiden_on_knn_graph(X, *, n_neighbors=15, resolution=1.0, seed=42):
    """Leiden on an approximate k-NN graph built with pynndescent.

    Unlike ``region_clustering.cluster_leiden`` (which uses sklearn
    NearestNeighbors — OK for up to ~10K points), this scales to 500K+ cells.
    Uses pynndescent for approximate kNN, igraph + leidenalg for community
    detection.
    """
    import igraph as ig
    import leidenalg
    import pynndescent

    n = X.shape[0]
    n_neighbors = min(n_neighbors + 1, n - 1)  # +1 for self

    logger.info("Building approximate kNN graph (pynndescent) for %d points...", n)
    ann = pynndescent.NNDescent(
        X.astype(np.float32),
        n_neighbors=n_neighbors,
        random_state=seed,
        n_jobs=-1,
    )
    ann.prepare()
    idxs, _ = ann.neighbor_graph

    # Build undirected edge list (skip self at column 0)
    src = np.repeat(np.arange(n), n_neighbors - 1)
    dst = idxs[:, 1:].reshape(-1)
    edges = np.column_stack([src, dst]).tolist()

    logger.info("Building igraph (%d edges)...", len(edges))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g = g.simplify(multiple=True, loops=True)

    logger.info("Running Leiden (resolution=%.2f)...", resolution)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.array(partition.membership, dtype=np.int32)
    logger.info("Leiden found %d communities", len(np.unique(labels)))
    return labels


def _load_rare_cluster_summary(csv_path: Path) -> list[dict]:
    """Read cluster_summary.csv produced by discover_rare_cell_types.py.

    Returns list of dicts with typed fields (size int, persistence/moran_i float,
    stable bool).
    """
    import csv as _csv

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            moran_raw = r.get("moran_i", "")
            moran_val: float | None = None
            if moran_raw not in ("", None):
                try:
                    moran_val = float(moran_raw)
                    if not math.isfinite(moran_val):
                        moran_val = None
                except ValueError:
                    moran_val = None
            rows.append(
                {
                    "cluster_id": int(r["cluster_id"]),
                    "size": int(r["size"]),
                    "hdbscan_persistence": float(r.get("hdbscan_persistence", 0.0) or 0.0),
                    "moran_i": moran_val,
                    "stable": (r.get("stable", "False") or "").lower() in ("true", "1", "yes"),
                    "noise_pct": float(r.get("noise_pct", 0.0) or 0.0),
                    "top_regions": r.get("top_regions", ""),
                    "top_morph_features": r.get("top_morph_features", ""),
                }
            )
    return rows


def _build_rare_dendrogram_svg(
    linkage: np.ndarray,
    cluster_ids: np.ndarray,
    summary_by_id: dict[int, dict],
    width: int = 900,
    height: int = 260,
    margin_l: int = 40,
    margin_r: int = 20,
    margin_t: int = 20,
    margin_b: int = 40,
) -> str:
    """Render Ward-linkage dendrogram as inline SVG with clickable leaves.

    Leaves are colored by rarity tier (smallest 20% red, largest 20% gray).
    Data attribute ``data-cid`` on each leaf circle wires into the viewer's
    existing ``selectCluster(cid)`` JS handler.

    Returns empty string if fewer than 2 clusters.
    """
    if linkage.shape[0] < 1 or len(cluster_ids) < 3:
        return ""

    from scipy.cluster.hierarchy import dendrogram

    d = dendrogram(linkage, no_plot=True)
    icoord = np.asarray(d["icoord"])  # (K-1, 4)
    dcoord = np.asarray(d["dcoord"])
    leaves = d["leaves"]  # permutation mapping leaf index → cluster-ids row index
    leaf_ordered_ids = [int(cluster_ids[i]) for i in leaves]

    # Rarity tiers by size
    sizes = np.array([summary_by_id.get(cid, {}).get("size", 0) for cid in leaf_ordered_ids])
    if len(sizes):
        p20, p80 = np.percentile(sizes, [20, 80])
    else:
        p20, p80 = 0, 0

    def _color_for(cid: int) -> str:
        info = summary_by_id.get(cid)
        # Missing-from-summary is abnormal — treat as exploratory (not stable).
        if info is None or not info.get("stable", False):
            return "#888888"
        sz = info.get("size", 0)
        if sz <= p20:
            return "#e6194b"
        if sz >= p80:
            return "#5a5a5a"
        return "#bfef45"

    # Scale coords
    x_min, x_max = icoord.min(), icoord.max()
    y_max = max(1.0, dcoord.max())
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def _sx(x):
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w if x_max > x_min else margin_l

    def _sy(y):
        return margin_t + plot_h - (y / y_max) * plot_h

    # Draw links as polylines
    polylines = []
    for i in range(len(icoord)):
        xs = [_sx(x) for x in icoord[i]]
        ys = [_sy(y) for y in dcoord[i]]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="#888" stroke-width="1.2"/>'
        )

    # Draw leaves: little circles + labels
    leaves_svg = []
    # Leaf x positions from icoord: each leaf occupies odd position index (5, 15, 25, ...)
    # Actually scipy's dendrogram leaf positions are at x = 5, 15, 25 ... by default
    n_leaves = len(leaf_ordered_ids)
    leaf_xs = [5 + 10 * i for i in range(n_leaves)]
    for idx, cid in enumerate(leaf_ordered_ids):
        cx = _sx(leaf_xs[idx])
        cy = _sy(0)
        color = _color_for(cid)
        sz = summary_by_id.get(cid, {}).get("size", 0)
        stable = summary_by_id.get(cid, {}).get("stable", True)
        leaves_svg.append(
            f'<g class="rare-leaf" data-cid="{cid}" cursor="pointer" '
            f'onclick="selectCluster({cid})">'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" '
            f'stroke="#fff" stroke-width="{1 if stable else 0.5}"/>'
            f'<text x="{cx:.1f}" y="{cy+18:.1f}" text-anchor="middle" '
            f'font-size="9" fill="#ccc">{cid}</text>'
            f'<title>cluster {cid}: {sz:,} cells{"" if stable else " (exploratory)"}</title>'
            f"</g>"
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#0d0d0d">'
        + "".join(polylines)
        + "".join(leaves_svg)
        + '<text x="10" y="15" fill="#aaa" font-size="10">Ward linkage · '
        "red=rare · olive=typical · gray=common/exploratory</text>" + "</svg>"
    )


def generate_rare_html(
    kept_detections: list[dict],
    region_ids: np.ndarray,
    summary: list[dict],
    linkage: np.ndarray,
    cluster_ids: np.ndarray,
    dendrogram_png_b64: str | None,
    output_path: Path,
    *,
    # Backward-compat kwargs for callers still passing the legacy signature.
    # These are NOT used by the current taxonomy-only rare-mode template;
    # spatial-map rendering belongs to a follow-up feature. Accepted and
    # ignored so older callers don't break.
    contours: dict | None = None,
    fluor_b64: str = "",
    img_w: int = 0,
    img_h: int = 0,
) -> None:
    """Generate rare-mode viewer HTML (taxonomy dendrogram + cluster sidebar).

    Simpler than ``generate_html`` — no UMAP (HDBSCAN in PCA doesn't emit one
    by default, and adding one would double runtime). Focused on two things
    that matter for rare-population review: cluster taxonomy (dendrogram) and
    per-cluster metrics (size, persistence, Moran's I, stable flag).

    Spatial map rendering is intentionally out of scope for this view — the
    default viewer (``--no-rare-mode``) already handles spatial.
    """
    del contours, fluor_b64, img_w, img_h  # accepted for compat; unused here.
    summary_by_id = {row["cluster_id"]: row for row in summary}

    # Prefer inline SVG dendrogram; fall back to PNG if provided.
    dendrogram_svg = _build_rare_dendrogram_svg(linkage, cluster_ids, summary_by_id)
    if not dendrogram_svg and dendrogram_png_b64:
        dendrogram_svg = (
            f'<img src="data:image/png;base64,{dendrogram_png_b64}" '
            f'style="width:100%;max-height:280px;background:#0d0d0d"/>'
        )
    if not dendrogram_svg:
        dendrogram_svg = (
            '<div style="padding:30px;color:#888">Fewer than 3 clusters — '
            "dendrogram skipped.</div>"
        )

    # Per-cluster spatial stats (region distribution) — reuse existing helper.
    labels_arr = np.array([int(d.get("rare_pop_id", -1)) for d in kept_detections])
    spatial_stats = compute_cluster_spatial_stats(labels_arr, region_ids, major_threshold=0.10)

    # Prepare cluster rows for sidebar
    rows_json = []
    for row in summary:
        cid = row["cluster_id"]
        spat = spatial_stats.get(cid, {}) if isinstance(spatial_stats, dict) else {}
        rows_json.append(
            {
                "cluster_id": cid,
                "size": row["size"],
                "persistence": row["hdbscan_persistence"],
                "moran_i": row["moran_i"],
                "stable": row["stable"],
                "n_major_regions": spat.get("n_major", 0),
                "top_region_frac": round(spat.get("top_region_frac", 0.0), 3),
                "top_regions": row.get("top_regions", ""),
                "top_features": row.get("top_morph_features", ""),
            }
        )

    data_json = safe_json(rows_json)
    n_stable = sum(1 for r in summary if r.get("stable"))
    summary_noise_pct = float(summary[0]["noise_pct"]) if summary else 0.0
    banner = ""
    if n_stable == 0:
        banner = (
            '<div style="background:#5a1a1a;color:#fff;padding:8px 14px">'
            "No STABLE clusters found. Showing all clusters as exploratory.</div>"
        )

    # Inline CSS + minimal JS
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Rare Cell Populations</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0a;color:#e0e0e0;font-family:system-ui,sans-serif;display:flex;height:100vh;overflow:hidden;font-size:12px}}
#sb{{width:340px;min-width:340px;background:#111;border-right:1px solid #222;overflow-y:auto}}
.hdr{{padding:12px 14px;border-bottom:1px solid #222}}
.hdr h1{{font-size:15px;margin-bottom:4px}}
.hdr .sub{{color:#888;font-size:11px}}
.controls{{padding:8px 14px;border-bottom:1px solid #1a1a1a;display:flex;flex-wrap:wrap;gap:6px}}
.controls button{{background:#222;color:#ddd;border:1px solid #333;padding:4px 8px;border-radius:3px;cursor:pointer;font-size:10px}}
.controls button:hover{{background:#333}}
.controls button.active{{background:#2d4a2d;border-color:#4a7a4a}}
.r{{margin:2px 8px;padding:6px 10px;border-radius:3px;cursor:pointer;border-left:3px solid transparent;display:flex;align-items:center;gap:8px}}
.r.stable{{border-left-color:#4caf50}}
.r.exploratory{{border-left-color:#888;opacity:0.65}}
.r.sel{{outline:1px solid #fff;background:#1a1a1a}}
.r:hover{{background:#1a1a1a}}
.r .info{{flex:1;min-width:0}}
.r .nm{{font-weight:600}}
.r .stat{{color:#888;font-size:10px;margin-top:1px}}
#main{{flex:1;display:flex;flex-direction:column;overflow:hidden}}
#dendro{{background:#0d0d0d;border-bottom:1px solid #222;overflow-x:auto;min-height:280px;max-height:300px}}
#spatial{{flex:1;background:#000;position:relative;overflow:auto}}
#spatial .stage{{position:relative;display:inline-block;transform-origin:0 0}}
#spatial .stage img{{display:block}}
#spatial .stage svg{{position:absolute;top:0;left:0;pointer-events:auto}}
.cluster-poly{{cursor:pointer}}
.cluster-poly.sel{{stroke-width:3 !important}}
#detail{{position:fixed;top:320px;right:14px;background:rgba(17,17,17,0.95);padding:10px 14px;border-radius:4px;max-width:320px;font-size:11px;border:1px solid #333;z-index:100}}
#detail h3{{font-size:12px;margin-bottom:4px}}
#detail table{{font-size:10px;border-collapse:collapse;margin-top:4px}}
#detail td{{padding:1px 8px 1px 0;color:#bbb}}
</style></head><body>
<div id=sb>
<div class=hdr>
<h1>Rare Cell Populations</h1>
<div class=sub>{len(summary)} clusters · {n_stable} stable · noise {100 * summary_noise_pct:.1f}%</div>
</div>
{banner}
<div class=controls>
<b style="margin-right:4px">Sort</b>
<button onclick="sortBy('size',1)">size↑</button>
<button onclick="sortBy('persistence',-1)">persistence↓</button>
<button onclick="sortBy('moran_i',-1)">Moran↓</button>
</div>
<div class=controls>
<b style="margin-right:4px">Filter</b>
<button id=fallbtn onclick="setFilter('all')" class=active>All</button>
<button id=fstablebtn onclick="setFilter('stable')">Stable only</button>
<button id=fsmallbtn onclick="setFilter('smallest')">Smallest 20</button>
</div>
<div id=rl></div>
</div>
<div id=main>
<div id=dendro>{dendrogram_svg}</div>
<div id=spatial></div>
</div>
<div id=detail><h3>Click a cluster or leaf to inspect</h3></div>
<script>
const R = {data_json};
let currentRows = R.slice();
let selectedCid = null;
let filter = 'all';

function render() {{
  const el = document.getElementById('rl');
  let rows = currentRows;
  if (filter === 'stable') rows = rows.filter(r => r.stable);
  if (filter === 'smallest') rows = [...rows].sort((a,b)=>a.size-b.size).slice(0, 20);
  el.innerHTML = '';
  for (const r of rows) {{
    const d = document.createElement('div');
    d.className = 'r ' + (r.stable ? 'stable' : 'exploratory') +
      (r.cluster_id === selectedCid ? ' sel' : '');
    d.dataset.cid = r.cluster_id;
    const moranStr = (r.moran_i == null) ? '—' : r.moran_i.toFixed(2);
    d.innerHTML = '<div class=info>' +
      '<div class=nm>Cluster ' + r.cluster_id + '</div>' +
      '<div class=stat>' + r.size.toLocaleString() + ' cells · ' +
      'persist ' + r.persistence.toFixed(3) + ' · Moran ' + moranStr + '</div>' +
      '</div>';
    d.onclick = () => selectCluster(r.cluster_id);
    el.appendChild(d);
  }}
}}

function sortBy(key, dir) {{
  currentRows.sort((a,b) => dir * (a[key] - b[key]));
  render();
}}

function setFilter(f) {{
  filter = f;
  document.querySelectorAll('.controls button').forEach(b => {{
    if (b.id === 'fallbtn' || b.id === 'fstablebtn' || b.id === 'fsmallbtn') {{
      b.classList.remove('active');
    }}
  }});
  document.getElementById({{'all':'fallbtn','stable':'fstablebtn','smallest':'fsmallbtn'}}[f]).classList.add('active');
  render();
}}

function selectCluster(cid) {{
  selectedCid = cid;
  const r = R.find(x => x.cluster_id === cid);
  if (r) {{
    const el = document.getElementById('detail');
    const topReg = r.top_regions.split(';').map(s=>s.trim()).filter(Boolean).slice(0,5);
    const topFeat = r.top_features.split(';').map(s=>s.trim()).filter(Boolean).slice(0,8);
    const moranStr = (r.moran_i == null) ? '—' : r.moran_i.toFixed(3);
    el.innerHTML = '<h3>Cluster ' + cid + (r.stable ? ' (stable)' : ' (exploratory)') + '</h3>' +
      '<div>' + r.size.toLocaleString() + ' cells | ' +
      'persist ' + r.persistence.toFixed(3) + ' | Moran I ' + moranStr + '</div>' +
      '<div style="margin-top:4px"><b>Top regions:</b> ' + topReg.join(', ') + '</div>' +
      '<div style="margin-top:4px"><b>Top features (z):</b><br>' +
      topFeat.map(f => '&nbsp;&nbsp;' + f).join('<br>') + '</div>';
  }}
  // Highlight leaf in dendrogram
  document.querySelectorAll('.rare-leaf circle').forEach(c => {{
    c.setAttribute('stroke-width', c.closest('.rare-leaf').dataset.cid == cid ? '3' : '1');
  }});
  render();
}}

render();
</script>
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote rare-mode viewer: %s (%.1f MB)", output_path, len(html) / 1e6)


def compute_cluster_spatial_stats(
    cluster_labels, region_ids, *, major_threshold=0.10, count_scale=1.0
):
    """For each cluster, compute region distribution + divergence score.

    Returns dict keyed by cluster_id:
        {
            "n_cells": int,
            "n_regions_touched": int,
            "region_dist": {rid: count, ...},      # regions with at least 1 cell
            "major_regions": [rid, rid, ...],      # regions with >= major_threshold
            "n_major": int,
            "entropy": float (normalized to [0,1]),
            "top_region_frac": float (largest region's share),
            "divergence": float (n_major * entropy — high = multi-modal)
        }
    """
    result = {}
    n_regions_total = len(np.unique(region_ids[region_ids > 0]))
    log_denom = np.log(max(n_regions_total, 2))

    for cid in np.unique(cluster_labels):
        if cid < 0:
            continue
        mask = cluster_labels == cid
        cluster_regions = region_ids[mask]
        # Only count cells in assigned regions (region_id > 0)
        cluster_regions = cluster_regions[cluster_regions > 0]
        n_cells = len(cluster_regions)
        if n_cells == 0:
            continue

        dist = Counter(cluster_regions.tolist())
        major_regions = [rid for rid, c in dist.items() if c / n_cells >= major_threshold]

        probs = np.array(list(dist.values()), dtype=float) / n_cells
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1))))
        entropy_norm = entropy / log_denom if log_denom > 0 else 0.0

        # Concentration / focal metrics
        sorted_probs = np.sort(probs)[::-1]  # descending
        cum = np.cumsum(sorted_probs)
        # k_90: regions needed to cover 90% of cluster's cells
        k_90 = int(np.searchsorted(cum, 0.90) + 1)
        # top_k fractions
        top_region_frac = float(sorted_probs[0])
        top3_frac = float(sorted_probs[:3].sum())

        # Legacy "divergence" (entropy-weighted, rewards ubiquitous clusters)
        divergence = len(major_regions) * entropy_norm

        # Focal multi-modal score: rewards clusters concentrated in a FEW distinct regions.
        # Uses top3 concentration * capped major count; returns 0 for k_90<=1 (organ-specific).
        if k_90 <= 1:
            focal_multimodal = 0.0  # single-region clusters aren't "multi-modal"
        else:
            focal_multimodal = top3_frac * min(len(major_regions), 5)

        # Scale up counts if this clustering ran on a subsample (so the min-cells
        # threshold in the viewer means the same thing across all 4 methods).
        result[int(cid)] = {
            "n_cells": int(round(n_cells * count_scale)),
            "n_regions_touched": len(dist),
            "region_dist": {int(k): int(round(v * count_scale)) for k, v in dist.items()},
            "major_regions": sorted([int(r) for r in major_regions]),
            "n_major": len(major_regions),
            "entropy": round(entropy_norm, 3),
            "top_region_frac": round(top_region_frac, 3),
            "top3_frac": round(top3_frac, 3),
            "k_90": int(k_90),
            "divergence": round(divergence, 3),
            "focal_multimodal": round(focal_multimodal, 3),
        }
    return result


def build_fluor_thumbnails(czi_path, channels, scale, scene):
    """Base64 JPEG thumbnails per channel + combined RGB."""
    thumbs = []
    arrs_for_rgb = []
    for ch in channels:
        ch_data, _, _, _ = read_czi_thumbnail_channels(
            czi_path, display_channels=[ch], scale_factor=scale, scene=scene
        )
        norm = percentile_normalize(ch_data[0])
        arrs_for_rgb.append(norm)
        buf = io.BytesIO()
        Image.fromarray(norm).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        thumbs.append((f"ch{ch}", b64))
    if len(channels) >= 2:
        while len(arrs_for_rgb) < 3:
            arrs_for_rgb.append(np.zeros_like(arrs_for_rgb[0]))
        rgb = np.stack(arrs_for_rgb[:3], axis=-1)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        name = "_".join(f"ch{c}" for c in channels[:3])
        thumbs.insert(0, (name, b64))
    return thumbs


def generate_html(
    umap_coords,  # (N_sub, 2)
    sub_labels,  # dict[method] -> (N_sub,) int array
    cluster_stats_all,  # dict[method] -> dict[cid] -> spatial stats
    contours,  # dict[rid] -> pts
    fluor_thumbnails,
    label_shape,
    output_path,
    title="Global Cluster + Spatial Divergence Viewer",
):
    """Emit 2-pane HTML: left = global UMAP, right = spatial map, 4 clustering toggles."""
    H, W = label_shape

    # Per-method: sidebar info + cluster->region_dist
    cluster_info_by_method = {}
    cluster_region_dist_by_method = {}
    for method, stats in cluster_stats_all.items():
        sorted_clusters = sorted(stats.items(), key=lambda x: -x[1]["divergence"])
        cluster_info_by_method[method] = [
            {
                "id": cid,
                "n_cells": s["n_cells"],
                "n_major": s["n_major"],
                "entropy": s["entropy"],
                "divergence": s["divergence"],
                "top_region_frac": s["top_region_frac"],
                "top3_frac": s["top3_frac"],
                "k_90": s["k_90"],
                "focal_multimodal": s["focal_multimodal"],
                "major_regions": s["major_regions"][:8],
            }
            for cid, s in sorted_clusters
        ]
        cluster_region_dist_by_method[method] = {
            str(cid): s["region_dist"] for cid, s in stats.items()
        }

    # Subsample labels per method (for UMAP coloring)
    sub_labels_by_method = {m: arr.tolist() for m, arr in sub_labels.items()}

    contour_payload = {int(rid): pts for rid, pts in contours.items()}

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; font-size: 12px; display: flex; height: 100vh; overflow: hidden; }}
#sidebar {{ width: 300px; min-width: 300px; background: #111; border-right: 1px solid #333; display: flex; flex-direction: column; }}
#sidebar h2 {{ font-size: 14px; color: #fff; padding: 12px 12px 6px; }}
#sidebar .summary {{ font-size: 11px; color: #888; padding: 0 12px 8px; border-bottom: 1px solid #333; }}
.sort-bar {{ padding: 6px 12px; display: flex; gap: 4px; flex-wrap: wrap; border-bottom: 1px solid #222; }}
.sort-btn, .method-btn {{ padding: 3px 8px; background: #1a1a1a; border: 1px solid #333; color: #888; font-family: inherit; font-size: 10px; cursor: pointer; border-radius: 3px; }}
.sort-btn:hover, .method-btn:hover {{ background: #222; }}
.sort-btn.active, .method-btn.active {{ background: #1a2a1a; border-color: #4caf50; color: #4caf50; }}
#cluster-list {{ flex: 1; overflow-y: auto; }}
.cluster-item {{ padding: 8px 12px; border-bottom: 1px solid #222; cursor: pointer; }}
.cluster-item:hover {{ background: #1a1a1a; }}
.cluster-item.active {{ background: #1a2a1a; border-left: 3px solid #4caf50; }}
.cluster-item .cid {{ font-weight: bold; color: #fff; }}
.cluster-item .meta {{ font-size: 10px; color: #888; margin-top: 2px; }}
.cluster-item .regions {{ font-size: 10px; color: #aaa; margin-top: 2px; }}
.divergence-bar {{ height: 3px; background: #333; margin-top: 3px; overflow: hidden; }}
.divergence-bar .fill {{ height: 100%; background: #f032e6; }}

#main {{ flex: 1; display: flex; overflow: hidden; }}
.pane {{ flex: 1; position: relative; display: flex; flex-direction: column; min-width: 400px; }}
.pane + .pane {{ border-left: 1px solid #333; }}
.pane-title {{ padding: 8px 12px; background: #111; border-bottom: 1px solid #222; color: #aaa; font-size: 11px; }}
.canvas-wrap {{ flex: 1; position: relative; overflow: hidden; }}
canvas {{ position: absolute; top: 0; left: 0; }}
#info {{ padding: 8px 12px; background: #111; border-top: 1px solid #222; font-size: 11px; color: #bbb; max-height: 130px; overflow-y: auto; }}
.info-row {{ margin-bottom: 4px; }}
.info-row .label {{ color: #888; }}
.info-row .val {{ color: #fff; font-weight: bold; }}
</style>
</head><body>
<div id="sidebar">
  <h2>Global Clusters</h2>
  <div class="summary">Cluster by:</div>
  <div class="sort-bar" style="border-top: 1px solid #333">
    <button class="method-btn active" data-method="leiden" onclick="setMethod(this)">leiden</button>
    <button class="method-btn" data-method="kmeans" onclick="setMethod(this)">kmeans</button>
    <button class="method-btn" data-method="hdbscan_pca" onclick="setMethod(this)">hdb·PCA</button>
    <button class="method-btn" data-method="hdbscan_umap" onclick="setMethod(this)">hdb·UMAP</button>
  </div>
  <div class="summary" id="method-summary">—</div>
  <div class="sort-bar">
    <button class="sort-btn active" data-sort="focal" onclick="sortClusters(this)" title="Focal multi-modal (best for 2-5 distinct regions)">Focal ↓</button>
    <button class="sort-btn" data-sort="k90_asc" onclick="sortClusters(this)" title="Fewest regions to cover 90% of cluster">k90 ↑</button>
    <button class="sort-btn" data-sort="divergence" onclick="sortClusters(this)" title="Entropy-weighted — rewards ubiquitous clusters">Div ↓</button>
    <button class="sort-btn" data-sort="n_major" onclick="sortClusters(this)">#Maj ↓</button>
    <button class="sort-btn" data-sort="cells" onclick="sortClusters(this)">Cells ↓</button>
    <button class="sort-btn" data-sort="top_frac" onclick="sortClusters(this)">Top% ↑</button>
  </div>
  <div id="cluster-list"></div>
</div>

<div id="main">
  <div class="pane">
    <div class="pane-title">Global UMAP (all cells, colored by cluster) &mdash; click in scatter to pick cluster</div>
    <div class="canvas-wrap" id="umap-wrap"><canvas id="umap-canvas"></canvas></div>
  </div>
  <div class="pane">
    <div class="pane-title">
      Spatial distribution &mdash; selected cluster's cells by organ region
      &nbsp;&nbsp;
      <span style="color:#888">Min cells/region:</span>
      <input type="number" id="min-cells-input" value="250" min="1" step="10"
             style="width:70px;background:#1a1a1a;color:#e0e0e0;border:1px solid #333;padding:2px 4px;font-family:inherit;font-size:10px"
             oninput="drawSpatial()">
    </div>
    <div class="canvas-wrap" id="spatial-wrap">
      <canvas id="bg-canvas"></canvas>
      <canvas id="heat-canvas"></canvas>
    </div>
    <div id="info">Click a cluster in the sidebar or on the UMAP to see its spatial distribution.</div>
  </div>
</div>

<script>
const UMAP_X = {safe_json(umap_coords[:, 0].tolist())};
const UMAP_Y = {safe_json(umap_coords[:, 1].tolist())};
const SUB_LABELS_BY_METHOD = {safe_json(sub_labels_by_method)};
const CLUSTER_INFO_BY_METHOD = {safe_json(cluster_info_by_method)};
const CLUSTER_REGION_DIST_BY_METHOD = {safe_json(cluster_region_dist_by_method)};
const CONTOURS = {safe_json(contour_payload)};
const BGS = {safe_json(dict(fluor_thumbnails))};
const LABEL_SHAPE = [{H}, {W}];
const PALETTE = [
  '#ff3860','#4dff5e','#4d8bff','#ffab30','#cc4dff',
  '#3ff2ff','#ff4df5','#d7ff4d','#ff7eb0','#3ff2c7',
  '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
  '#42d4f4','#f032e6','#bfef45','#fabebe','#469990'
];
const NOISE = '#555';
let currentCluster = null;
let currentSort = 'focal';
let currentMethod = 'leiden';
let bgImg = null;

// Proxies that key into the active method
function clusterInfo() {{ return CLUSTER_INFO_BY_METHOD[currentMethod] || []; }}
function clusterRegionDist() {{ return CLUSTER_REGION_DIST_BY_METHOD[currentMethod] || {{}}; }}
function umapLabels() {{ return SUB_LABELS_BY_METHOD[currentMethod] || []; }}

function clusterColor(cid) {{
  if (cid < 0) return NOISE;
  return PALETTE[cid % PALETTE.length];
}}
function setMethod(btn) {{
  document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentMethod = btn.dataset.method;
  currentCluster = null;  // reset selection when method changes
  updateMethodSummary();
  buildSidebar();
  drawUmap();
  drawSpatial();
  updateInfo();
}}
function updateMethodSummary() {{
  const info = clusterInfo();
  let noiseCount = 0;
  const labs = umapLabels();
  for (const l of labs) if (l < 0) noiseCount++;
  const noisePct = labs.length ? (100 * noiseCount / labs.length).toFixed(1) : '0';
  document.getElementById('method-summary').textContent =
    info.length + ' clusters' + (noiseCount > 0 ? ' · ' + noisePct + '% noise' : '');
}}
function sortClusters(btn) {{
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentSort = btn.dataset.sort;
  buildSidebar();
}}
function buildSidebar() {{
  const items = [...clusterInfo()];
  if (currentSort === 'focal') items.sort((a, b) => b.focal_multimodal - a.focal_multimodal);
  else if (currentSort === 'k90_asc') items.sort((a, b) => (a.k_90 || 99) - (b.k_90 || 99));
  else if (currentSort === 'divergence') items.sort((a, b) => b.divergence - a.divergence);
  else if (currentSort === 'n_major') items.sort((a, b) => b.n_major - a.n_major);
  else if (currentSort === 'cells') items.sort((a, b) => b.n_cells - a.n_cells);
  else if (currentSort === 'top_frac') items.sort((a, b) => b.top_region_frac - a.top_region_frac);

  const list = document.getElementById('cluster-list');
  list.innerHTML = '';
  // Scale the progress bar to whichever metric is the active sort (fall back to focal)
  const getVal = c => {{
    if (currentSort === 'k90_asc') return 1 / Math.max(c.k_90 || 1, 1);
    if (currentSort === 'divergence') return c.divergence;
    if (currentSort === 'cells') return c.n_cells;
    if (currentSort === 'n_major') return c.n_major;
    if (currentSort === 'top_frac') return c.top_region_frac;
    return c.focal_multimodal;
  }};
  const maxVal = Math.max(...items.map(getVal), 0.01);
  items.forEach(c => {{
    const div = document.createElement('div');
    div.className = 'cluster-item' + (String(c.id) === currentCluster ? ' active' : '');
    div.dataset.cid = c.id;
    const pct = Math.round((getVal(c) / maxVal) * 100);
    const regionsStr = c.major_regions.length
      ? 'Major: [' + c.major_regions.join(', ') + ']'
      : '<em>no major regions</em>';
    div.innerHTML = '<div class="cid" style="color:' + clusterColor(c.id) + '">Cluster ' + c.id + '</div>' +
      '<div class="meta">' + c.n_cells.toLocaleString() + ' cells · focal=' + c.focal_multimodal.toFixed(2) +
      ' · k90=' + c.k_90 + ' · top=' + (c.top_region_frac*100).toFixed(0) + '%</div>' +
      '<div class="regions">' + regionsStr + '</div>' +
      '<div class="divergence-bar"><div class="fill" style="width:' + pct + '%"></div></div>';
    div.onclick = () => selectCluster(String(c.id) === currentCluster ? null : String(c.id));
    list.appendChild(div);
  }});
}}

function selectCluster(cid) {{
  currentCluster = cid;
  document.querySelectorAll('.cluster-item').forEach(el => {{
    el.classList.toggle('active', cid !== null && el.dataset.cid === cid);
  }});
  drawUmap();
  drawSpatial();
  updateInfo();
}}
function updateInfo() {{
  const el = document.getElementById('info');
  if (currentCluster === null) {{
    el.innerHTML = 'Click a cluster in the sidebar or on the UMAP to see its spatial distribution.';
    return;
  }}
  const info = clusterInfo().find(c => String(c.id) === currentCluster);
  if (!info) return;
  let html = '<div class="info-row"><span class="label">Cluster </span><span class="val" style="color:' + clusterColor(info.id) + '">' + info.id + '</span>';
  html += ' &nbsp; <span class="label">Cells </span><span class="val">' + info.n_cells.toLocaleString() + '</span>';
  html += ' &nbsp; <span class="label">Divergence </span><span class="val">' + info.divergence.toFixed(2) + '</span>';
  html += ' &nbsp; <span class="label">Entropy </span><span class="val">' + info.entropy.toFixed(2) + '</span>';
  html += ' &nbsp; <span class="label">Top region fraction </span><span class="val">' + (info.top_region_frac*100).toFixed(1) + '%</span></div>';
  html += '<div class="info-row"><span class="label">Major regions (≥10% of cluster): </span>';
  html += info.major_regions.length ? '<span class="val">' + info.major_regions.join(', ') + '</span>' : '<em>none</em>';
  html += '</div>';
  el.innerHTML = html;
}}

// ---- UMAP pane ----
let umapBounds = null;
function drawUmap() {{
  const canvas = document.getElementById('umap-canvas');
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = wrap.clientWidth * dpr; canvas.height = wrap.clientHeight * dpr;
  canvas.style.width = wrap.clientWidth + 'px'; canvas.style.height = wrap.clientHeight + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = wrap.clientWidth, h = wrap.clientHeight, pad = 20;
  ctx.fillStyle = '#0d0d0d';
  ctx.fillRect(0, 0, w, h);
  if (UMAP_X.length === 0) return;
  if (!umapBounds) {{
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < UMAP_X.length; i++) {{
      if (UMAP_X[i] < xMin) xMin = UMAP_X[i]; if (UMAP_X[i] > xMax) xMax = UMAP_X[i];
      if (UMAP_Y[i] < yMin) yMin = UMAP_Y[i]; if (UMAP_Y[i] > yMax) yMax = UMAP_Y[i];
    }}
    umapBounds = {{xMin, xMax, yMin, yMax}};
  }}
  const {{xMin, xMax, yMin, yMax}} = umapBounds;
  const sx = v => pad + (v - xMin) / (xMax - xMin + 1e-9) * (w - 2 * pad);
  const sy = v => h - pad - (v - yMin) / (yMax - yMin + 1e-9) * (h - 2 * pad);
  const selectedId = currentCluster !== null ? parseInt(currentCluster) : null;
  // First pass: non-selected (dim). Use labels for currentMethod.
  const labs = umapLabels();
  for (let i = 0; i < UMAP_X.length; i++) {{
    const lbl = labs[i];
    if (selectedId !== null && lbl !== selectedId) {{
      ctx.fillStyle = '#222';
      ctx.globalAlpha = 0.4;
    }} else {{
      ctx.fillStyle = clusterColor(lbl);
      ctx.globalAlpha = selectedId !== null ? 0.95 : 0.55;
    }}
    ctx.beginPath();
    ctx.arc(sx(UMAP_X[i]), sy(UMAP_Y[i]), selectedId !== null && lbl === selectedId ? 2.5 : 1.5, 0, Math.PI * 2);
    ctx.fill();
  }}
  ctx.globalAlpha = 1;
}}

// ---- Spatial pane ----
function loadBg() {{
  if (!BGS) return;
  const key = Object.keys(BGS)[0];
  if (!key) return;
  bgImg = new Image();
  bgImg.onload = () => drawSpatial();
  bgImg.src = 'data:image/jpeg;base64,' + BGS[key];
}}
function drawSpatial() {{
  const bgCanvas = document.getElementById('bg-canvas');
  const heatCanvas = document.getElementById('heat-canvas');
  const wrap = bgCanvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  [bgCanvas, heatCanvas].forEach(c => {{
    c.width = wrap.clientWidth * dpr; c.height = wrap.clientHeight * dpr;
    c.style.width = wrap.clientWidth + 'px'; c.style.height = wrap.clientHeight + 'px';
    c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  }});
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  const scale = Math.min(cw / LABEL_SHAPE[1], ch / LABEL_SHAPE[0]) * 0.95;
  const tx = (cw - LABEL_SHAPE[1] * scale) / 2;
  const ty = (ch - LABEL_SHAPE[0] * scale) / 2;

  const bgCtx = bgCanvas.getContext('2d');
  bgCtx.fillStyle = '#0a0a0a'; bgCtx.fillRect(0, 0, cw, ch);
  if (bgImg && bgImg.complete) {{
    bgCtx.save(); bgCtx.translate(tx, ty); bgCtx.scale(scale, scale);
    bgCtx.globalAlpha = 0.6;
    bgCtx.drawImage(bgImg, 0, 0, LABEL_SHAPE[1], LABEL_SHAPE[0]);
    bgCtx.restore();
  }}
  bgCtx.globalAlpha = 1;

  const heatCtx = heatCanvas.getContext('2d');
  heatCtx.save(); heatCtx.translate(tx, ty); heatCtx.scale(scale, scale);
  heatCtx.lineWidth = 1.0 / scale;

  if (currentCluster === null) {{
    // show all contours dim
    heatCtx.strokeStyle = '#3a3a3a';
    for (const [rid, pts] of Object.entries(CONTOURS)) {{
      heatCtx.beginPath();
      for (let i = 0; i < pts.length; i++) {{
        if (i === 0) heatCtx.moveTo(pts[i][0], pts[i][1]);
        else heatCtx.lineTo(pts[i][0], pts[i][1]);
      }}
      heatCtx.closePath(); heatCtx.stroke();
    }}
    heatCtx.restore();
    return;
  }}

  // Shade regions by cluster density
  const dist = clusterRegionDist()[currentCluster] || {{}};
  const minInput = document.getElementById('min-cells-input');
  const minCells = minInput ? Math.max(1, parseInt(minInput.value || '1', 10) || 1) : 1;
  let maxCount = 0;
  for (const v of Object.values(dist)) if (v > maxCount) maxCount = v;
  const clusterHex = clusterColor(parseInt(currentCluster));

  for (const [rid, pts] of Object.entries(CONTOURS)) {{
    const count = dist[rid] || 0;
    if (count < minCells) continue;  // hide regions below the threshold
    const alpha = maxCount > 0 ? (count / maxCount) : 0;
    heatCtx.beginPath();
    for (let i = 0; i < pts.length; i++) {{
      if (i === 0) heatCtx.moveTo(pts[i][0], pts[i][1]);
      else heatCtx.lineTo(pts[i][0], pts[i][1]);
    }}
    heatCtx.closePath();
    heatCtx.fillStyle = clusterHex;
    heatCtx.globalAlpha = 0.25 + 0.6 * alpha;
    heatCtx.fill();
    heatCtx.globalAlpha = 1;
    heatCtx.strokeStyle = '#ffffff';
    heatCtx.lineWidth = 1.5 / scale;
    heatCtx.stroke();
  }}
  heatCtx.restore();
}}

// ---- UMAP click → select cluster ----
document.getElementById('umap-canvas').addEventListener('click', (e) => {{
  const wrap = document.getElementById('umap-wrap').getBoundingClientRect();
  const mx = e.clientX - wrap.left;
  const my = e.clientY - wrap.top;
  if (!umapBounds) return;
  const w = wrap.width, h = wrap.height, pad = 20;
  const {{xMin, xMax, yMin, yMax}} = umapBounds;
  const sx = v => pad + (v - xMin) / (xMax - xMin + 1e-9) * (w - 2 * pad);
  const sy = v => h - pad - (v - yMin) / (yMax - yMin + 1e-9) * (h - 2 * pad);
  // Find closest point
  let bestDist = 25, bestLabel = null;  // within 5px
  const labs = umapLabels();
  for (let i = 0; i < UMAP_X.length; i++) {{
    const dx = sx(UMAP_X[i]) - mx, dy = sy(UMAP_Y[i]) - my;
    const d = dx * dx + dy * dy;
    if (d < bestDist) {{ bestDist = d; bestLabel = labs[i]; }}
  }}
  if (bestLabel !== null) selectCluster(String(bestLabel));
  else selectCluster(null);
}});

window.addEventListener('resize', () => {{ drawUmap(); drawSpatial(); }});
updateMethodSummary();
buildSidebar();
loadBg();
drawUmap();
drawSpatial();
</script>
</body></html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)
    size_mb = Path(output_path).stat().st_size / 1e6
    total_clusters = sum(len(s) for s in cluster_stats_all.values())
    logger.info(
        "Wrote %s (%d UMAP pts, %d total clusters across 4 methods, %.1f MB)",
        output_path,
        len(umap_coords),
        total_clusters,
        size_mb,
    )


def _run_rare_mode(args) -> int:
    """Rare-mode viewer: reads `rare_pop_id` from detections + pre-computed
    cluster summary, skips Leiden/k-means/HDBSCAN, writes dendrogram + spatial
    map HTML.
    """
    logger.info("Rare mode: loading detections %s", args.detections)
    detections = fast_json_load(args.detections)
    kept = [d for d in detections if d.get("organ_id", 0) > 0 and "rare_pop_id" in d]
    del detections
    logger.info("  %d cells with organ_id>0 AND rare_pop_id labeled", len(kept))

    summary = _load_rare_cluster_summary(args.rare_cluster_summary)
    logger.info(
        "  %d clusters in summary (%d stable)",
        len(summary),
        sum(1 for r in summary if r["stable"]),
    )

    linkage = np.zeros((0, 4), dtype=np.float32)
    cluster_ids = np.array([], dtype=np.int32)
    if args.rare_linkage_matrix:
        linkage = np.load(args.rare_linkage_matrix, allow_pickle=False)
        logger.info("  linkage loaded: shape=%s", linkage.shape)
    if args.rare_cluster_ids:
        cluster_ids = np.load(args.rare_cluster_ids, allow_pickle=False)
    elif linkage.size > 0:
        # Derive cluster_ids from summary in stable-first order
        cluster_ids = np.array([r["cluster_id"] for r in summary])

    dendrogram_png_b64 = None
    if args.rare_dendrogram_png and args.rare_dendrogram_png.exists():
        dendrogram_png_b64 = base64.b64encode(args.rare_dendrogram_png.read_bytes()).decode()

    # Taxonomy-only view: no CZI thumbnail load (saves ~5s + ~50MB RAM).
    region_ids = np.array([d["organ_id"] for d in kept])
    generate_rare_html(
        kept_detections=kept,
        region_ids=region_ids,
        summary=summary,
        linkage=linkage,
        cluster_ids=cluster_ids,
        dendrogram_png_b64=dendrogram_png_b64,
        output_path=Path(args.output),
    )
    return 0


def main():
    args = parse_args()

    # --- Rare-mode short-circuit ---
    if args.rare_mode:
        return _run_rare_mode(args)

    rng = np.random.default_rng(42)

    # --- Load detections, filter ---
    logger.info("Loading detections from %s ...", args.detections)
    detections = fast_json_load(args.detections)
    logger.info("Loaded %d detections", len(detections))

    # Filter: keep n_nuclei >= min_nuclei AND organ_id > 0
    kept = []
    for det in detections:
        oid = det.get("organ_id", 0)
        if oid == 0:
            continue
        nn = det.get("features", {}).get("n_nuclei")
        try:
            nn_val = float(nn) if nn is not None else None
        except (TypeError, ValueError):
            continue
        if nn_val is None or not math.isfinite(nn_val) or int(nn_val) < args.min_nuclei:
            continue
        kept.append(det)
    del detections
    logger.info("Kept %d cells with organ_id>0 AND n_nuclei>=%d", len(kept), args.min_nuclei)

    # Also drop cells from tiny regions (noise)
    region_counts = Counter(d["organ_id"] for d in kept)
    good_regions = {r for r, c in region_counts.items() if c >= args.min_cells_per_region}
    kept = [d for d in kept if d["organ_id"] in good_regions]
    logger.info(
        "After dropping regions with <%d cells: %d cells across %d regions",
        args.min_cells_per_region,
        len(kept),
        len(good_regions),
    )

    # --- Feature selection + matrix ---
    feature_groups = {g.strip() for g in args.feature_groups.split(",")}
    feature_names = select_feature_names(kept, feature_groups)
    logger.info("Using %d features from groups %s", len(feature_names), sorted(feature_groups))

    X, _, valid_idx = _extract_feature_matrix(kept, feature_names)
    if X is None:
        logger.error("No valid cells after feature extraction")
        sys.exit(1)
    kept = [kept[i] for i in valid_idx]
    region_ids = np.array([d["organ_id"] for d in kept])
    logger.info("Feature matrix: %s", X.shape)

    # Drop zero-variance features, scale
    variances = np.var(X, axis=0)
    nonconstant = variances > 1e-12
    X = X[:, nonconstant]
    X_scaled = StandardScaler().fit_transform(X).astype(np.float32)

    # --- PCA ---
    full_max = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca_full = PCA(n_components=full_max, random_state=42)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_cutoff = int(np.searchsorted(cumvar, args.var_cutoff) + 1)
    n_pcs = max(2, min(n_for_cutoff, args.max_pcs, full_max))
    X_pca = pca_full.transform(X_scaled)[:, :n_pcs]
    logger.info(
        "PCA: %d PCs capture %.1f%% variance",
        n_pcs,
        cumvar[n_pcs - 1] * 100,
    )

    # --- UMAP on FULL set ---
    n_total = X_pca.shape[0]
    from umap import UMAP

    logger.info("Running UMAP on FULL %d cells (this can take 30-60 min)...", n_total)
    umap_model = UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=42,
        low_memory=True,  # spill intermediate matrices to disk if needed
        n_jobs=-1,
    )
    X_umap_full = umap_model.fit_transform(X_pca)
    logger.info("UMAP done. Shape: %s", X_umap_full.shape)

    # Subsample for the viewer's scatter plot (browser perf)
    if n_total > args.max_umap_points:
        sub_idx = rng.choice(n_total, size=args.max_umap_points, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n_total)
    X_umap = X_umap_full[sub_idx]  # only for display

    # --- 4 clusterings ---
    # Leiden on full kNN graph (scales via pynndescent)
    labels_leiden = leiden_on_knn_graph(
        X_pca, n_neighbors=args.leiden_knn, resolution=args.leiden_resolution
    )

    # K-means with elbow — on full PCA
    logger.info("K-means elbow on %d cells (k=2..%d)...", n_total, args.max_k)
    best_k, sil, labels_kmeans, ch, sil_per_k, inertia_per_k = find_optimal_k_elbow(
        X_pca, max_k=args.max_k, rng=rng
    )
    logger.info("K-means elbow picked k=%d (silhouette=%.3f)", best_k, sil)

    # HDBSCAN on FULL PCA space (parallel core-distance — faster than default)
    import hdbscan

    logger.info("HDBSCAN on FULL %d cells in PCA space (parallel)...", n_total)
    hdb_full = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_size,
        min_samples=None,
        core_dist_n_jobs=-1,
    )
    labels_hdb_pca = hdb_full.fit_predict(X_pca).astype(np.int32)
    logger.info(
        "HDBSCAN-PCA (full): %d clusters, %d noise (%.1f%%)",
        len(np.unique(labels_hdb_pca[labels_hdb_pca >= 0])),
        int(np.sum(labels_hdb_pca < 0)),
        100 * np.sum(labels_hdb_pca < 0) / n_total,
    )

    # HDBSCAN on FULL UMAP (2D — fast even for 520K)
    logger.info("HDBSCAN on FULL UMAP coords (%d cells)...", n_total)
    hdb_umap = hdbscan.HDBSCAN(
        min_cluster_size=args.hdbscan_min_size,
        min_samples=None,
        core_dist_n_jobs=-1,
    )
    labels_hdb_umap = hdb_umap.fit_predict(X_umap_full).astype(np.int32)
    logger.info(
        "HDBSCAN-UMAP (full): %d clusters, %d noise (%.1f%%)",
        len(np.unique(labels_hdb_umap[labels_hdb_umap >= 0])),
        int(np.sum(labels_hdb_umap < 0)),
        100 * np.sum(labels_hdb_umap < 0) / n_total,
    )

    # --- Per-cluster spatial stats per method (all on FULL region_ids now) ---
    cluster_stats_all = {
        "leiden": compute_cluster_spatial_stats(
            labels_leiden, region_ids, major_threshold=args.major_region_threshold
        ),
        "kmeans": compute_cluster_spatial_stats(
            labels_kmeans, region_ids, major_threshold=args.major_region_threshold
        ),
        "hdbscan_pca": compute_cluster_spatial_stats(
            labels_hdb_pca, region_ids, major_threshold=args.major_region_threshold
        ),
        "hdbscan_umap": compute_cluster_spatial_stats(
            labels_hdb_umap, region_ids, major_threshold=args.major_region_threshold
        ),
    }

    # Subsample labels for the viewer's scatter (all 4 methods now on full set).
    sub_labels = {
        "leiden": labels_leiden[sub_idx],
        "kmeans": labels_kmeans[sub_idx],
        "hdbscan_pca": labels_hdb_pca[sub_idx],
        "hdbscan_umap": labels_hdb_umap[sub_idx],
    }

    # --- Print divergence summary per method ---
    for method, stats in cluster_stats_all.items():
        logger.info("")
        logger.info("=== Top 10 %s clusters by divergence ===", method)
        by_div = sorted(stats.items(), key=lambda x: -x[1]["divergence"])
        for cid, s in by_div[:10]:
            logger.info(
                "  C%3d: %6d cells | div=%.2f | #maj=%d | entropy=%.2f | top=%.0f%% | major=%s",
                cid,
                s["n_cells"],
                s["divergence"],
                s["n_major"],
                s["entropy"],
                s["top_region_frac"] * 100,
                s["major_regions"][:5],
            )

    # --- Spatial contours + thumbnails ---
    logger.info("Loading label map + contours for spatial pane...")
    label_map = np.load(args.label_map)
    contours = extract_region_contours(label_map)
    channels = [int(c.strip()) for c in args.display_channels.split(",")]
    fluor_thumbnails = build_fluor_thumbnails(args.czi_path, channels, args.scale, args.scene)

    # Only keep contours for regions that appear in at least one cluster (any method)
    touched_regions = set()
    for stats in cluster_stats_all.values():
        for s in stats.values():
            touched_regions.update(s["region_dist"].keys())
    contours = {rid: pts for rid, pts in contours.items() if rid in touched_regions}
    logger.info("Keeping %d region contours for viewer", len(contours))

    # --- Generate HTML ---
    generate_html(
        X_umap,
        sub_labels,
        cluster_stats_all,
        contours,
        fluor_thumbnails,
        label_map.shape,
        args.output,
    )


if __name__ == "__main__":
    main()
