#!/usr/bin/env python3
"""Spatial analysis of pancreatic islets from a completed islet detection run.

Defines islets as the union of marker-positive (bright) cells within a spatial
buffer, recruits nearby unclassified cells, then computes per-islet features:
  - Morphometry: area, perimeter, circularity, elongation (PCA)
  - Composition: per-marker counts/fractions, Shannon entropy, dominant type
  - Spatial: nearest-neighbor distances between marker types, radial distribution,
    mantle-core index, mixing index
  - Atypical flags: low_beta, alpha_dominant, inverted_architecture, highly_segregated

Outputs:
  - islet_summary.csv          — one row per islet with all features
  - islet_detections.json      — enriched detections with islet_id + marker_class
  - html/islet_analysis.html   — visual overview with cards, pie charts, histograms

Usage:
  python scripts/analyze_islets.py \\
      --run-dir /path/to/islet_output \\
      --czi-path /path/to/slide.czi \\
      --buffer-um 25 --min-cells 5
"""

import argparse
import base64
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.io.czi_loader import get_loader
from run_segmentation import classify_islet_marker, compute_islet_marker_thresholds

try:
    import hdf5plugin  # noqa: F401 — registers LZ4 codec for h5py
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PADDING = 40
PINK = (255, 0, 255)
_BASE_MARKER_COLORS = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]
_MARKER_HTML_COLORS = ['#ff3333', '#33cc33', '#3333ff']


def build_marker_colors(marker_map):
    """Build marker_class -> RGB color dict from marker_map."""
    colors = {'multi': (255, 170, 0), 'none': (128, 128, 128)}
    for i, name in enumerate(marker_map.keys()):
        colors[name] = _BASE_MARKER_COLORS[i] if i < len(_BASE_MARKER_COLORS) else (200, 200, 200)
    return colors


# ---------------------------------------------------------------------------
# 1. Islet definition
# ---------------------------------------------------------------------------

def load_detections(run_dir):
    """Load detections JSON from run directory."""
    det_path = Path(run_dir) / 'islet_detections.json'
    if not det_path.exists():
        logger.error(f"Detections not found: {det_path}")
        sys.exit(1)
    with open(det_path) as f:
        dets = json.load(f)
    print(f"Loaded {len(dets)} detections from {det_path}")
    return dets


def classify_all(detections, marker_map):
    """Classify all detections by marker type. Modifies in place."""
    marker_thresholds = compute_islet_marker_thresholds(detections, marker_map=marker_map)
    counts = Counter()
    for det in detections:
        mc, _ = classify_islet_marker(det.get('features', {}), marker_thresholds, marker_map=marker_map)
        det['marker_class'] = mc
        counts[mc] += 1
    print(f"Marker classification: {dict(counts)}")
    return marker_thresholds


def define_islets(detections, pixel_size, buffer_um=25.0, min_cells=5):
    """Define islets as the union of marker-positive (bright) cells + spatial buffer.

    Simple approach:
    1. Take all marker+ cells (any endocrine marker above threshold)
    2. Group them into connected components: two marker+ cells within buffer_um
       belong to the same islet (single-linkage via KDTree)
    3. Recruit 'none' cells within buffer_um of any marker+ cell in the islet
    4. Filter by min_cells

    Returns islet_groups: {islet_id: [det_indices]}
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix

    # Separate marker+ vs none
    endocrine_idx = []
    none_idx = []
    for i, det in enumerate(detections):
        gc = det.get('global_center', det.get('center'))
        if gc is None:
            continue
        mc = det.get('marker_class', 'none')
        if mc != 'none':  # 'multi' cells are endocrine and should seed islets
            endocrine_idx.append(i)
        else:
            none_idx.append(i)

    if not endocrine_idx:
        print("No endocrine cells found — cannot define islets")
        return {}

    print(f"Defining islets from {len(endocrine_idx)} marker+ cells (buffer={buffer_um} um)...")

    # Build coordinates in um
    endo_coords_um = np.array([
        [detections[i].get('global_center', detections[i].get('center', [0, 0]))[0] * pixel_size,
         detections[i].get('global_center', detections[i].get('center', [0, 0]))[1] * pixel_size]
        for i in endocrine_idx
    ])

    # Connected components: marker+ cells within buffer_um are in the same islet
    tree = cKDTree(endo_coords_um)
    pairs = tree.query_pairs(r=buffer_um)

    n = len(endocrine_idx)
    if pairs:
        rows, cols = zip(*pairs)
        rows, cols = list(rows), list(cols)
        # Symmetric: add both (i,j) and (j,i)
        data = [True] * len(rows) * 2
        all_rows = rows + cols
        all_cols = cols + rows
        adj = coo_matrix((data, (all_rows, all_cols)), shape=(n, n), dtype=bool)
    else:
        adj = coo_matrix((n, n), dtype=bool)

    n_components, labels = connected_components(adj, directed=False)

    # Assign islet_id to endocrine cells
    for row_i, det_i in enumerate(endocrine_idx):
        detections[det_i]['islet_id'] = int(labels[row_i])

    # Count per component
    component_sizes = Counter(labels)
    valid_components = {c for c, sz in component_sizes.items() if sz >= min_cells}
    n_singletons = sum(1 for c, sz in component_sizes.items() if sz < min_cells)
    print(f"  {len(valid_components)} islets (>= {min_cells} marker+ cells), "
          f"{n_singletons} small groups discarded")

    # Mark small-component cells as unassigned
    for row_i, det_i in enumerate(endocrine_idx):
        if labels[row_i] not in valid_components:
            detections[det_i]['islet_id'] = -1

    # Recruit 'none' cells within buffer_um of any islet's marker+ cells
    if none_idx and valid_components:
        # Build KDTree of valid endocrine cells only
        valid_mask = np.array([labels[r] in valid_components for r in range(n)])
        valid_coords = endo_coords_um[valid_mask]
        valid_labels = labels[valid_mask]

        if len(valid_coords) > 0:
            valid_tree = cKDTree(valid_coords)
            none_coords_um = np.array([
                [detections[i].get('global_center', detections[i].get('center', [0, 0]))[0] * pixel_size,
                 detections[i].get('global_center', detections[i].get('center', [0, 0]))[1] * pixel_size]
                for i in none_idx
            ])
            dists, indices = valid_tree.query(none_coords_um)
            recruited = 0
            for j, (d, idx) in enumerate(zip(dists, indices)):
                if d <= buffer_um:
                    detections[none_idx[j]]['islet_id'] = int(valid_labels[idx])
                    recruited += 1
                else:
                    detections[none_idx[j]]['islet_id'] = -1
            print(f"  Recruited {recruited}/{len(none_idx)} none cells into islets")
        else:
            for i in none_idx:
                detections[i]['islet_id'] = -1
    else:
        for i in none_idx:
            detections[i]['islet_id'] = -1

    # Build islet groups
    islet_groups = {}
    for i, det in enumerate(detections):
        iid = det.get('islet_id', -1)
        if iid >= 0:
            islet_groups.setdefault(iid, []).append(i)

    return islet_groups


# ---------------------------------------------------------------------------
# 2. Islet morphometry
# ---------------------------------------------------------------------------

def compute_islet_shape(cells, pixel_size):
    """Compute shape features from cell coordinates via ConvexHull.

    Returns dict with area_um2, perimeter_um, circularity, elongation, centroid_um.
    """
    coords_um = np.array([
        [c.get('global_center', c.get('center', [0, 0]))[0] * pixel_size,
         c.get('global_center', c.get('center', [0, 0]))[1] * pixel_size]
        for c in cells
    ])

    result = {
        'n_cells': len(cells),
        'centroid_x_um': float(np.mean(coords_um[:, 0])),
        'centroid_y_um': float(np.mean(coords_um[:, 1])),
    }

    if len(coords_um) < 3:
        # Can't compute ConvexHull with <3 points
        result.update({
            'area_um2': 0.0,
            'perimeter_um': 0.0,
            'circularity': 0.0,
            'elongation': 1.0,
        })
        return result

    try:
        hull = ConvexHull(coords_um)
        area = hull.volume  # In 2D, volume = area
        perimeter = hull.area  # In 2D, area = perimeter
    except Exception:
        # Collinear or degenerate points — use bounding box as fallback
        extent = coords_um.ptp(axis=0)
        area = max(float(extent[0] * extent[1]), 1.0)
        perimeter = 2.0 * (extent[0] + extent[1])

    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Elongation via PCA eigenvalue ratio
    centered = coords_um - coords_um.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    elongation = float(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else 1.0

    result.update({
        'area_um2': float(area),
        'perimeter_um': float(perimeter),
        'circularity': float(circularity),
        'elongation': float(elongation),
    })
    return result


# ---------------------------------------------------------------------------
# 3. Composition metrics
# ---------------------------------------------------------------------------

def compute_composition(cells, marker_map):
    """Compute per-marker counts, fractions, Shannon entropy, dominant type."""
    marker_names = list(marker_map.keys())
    counts = Counter(c.get('marker_class', 'none') for c in cells)
    n = len(cells)

    result = {}
    fractions = {}
    for name in marker_names:
        cnt = counts.get(name, 0)
        frac = cnt / n if n > 0 else 0.0
        result[f'n_{name}'] = cnt
        result[f'frac_{name}'] = round(frac, 4)
        fractions[name] = frac

    result['n_multi'] = counts.get('multi', 0)
    result['n_none'] = counts.get('none', 0)
    result['frac_multi'] = round(counts.get('multi', 0) / n, 4) if n > 0 else 0.0
    result['frac_none'] = round(counts.get('none', 0) / n, 4) if n > 0 else 0.0

    # Shannon entropy over endocrine compartment (renormalize marker fractions to sum to 1)
    endo_total = sum(fractions.values())
    if endo_total > 0:
        ps = [f / endo_total for f in fractions.values() if f > 0]
        entropy = -sum(p * math.log2(p) for p in ps)
    else:
        entropy = 0.0
    result['entropy'] = round(entropy, 4)

    # Dominant type
    if fractions:
        dominant = max(fractions, key=fractions.get)
        result['dominant_type'] = dominant
        result['dominant_frac'] = round(fractions[dominant], 4)
    else:
        result['dominant_type'] = 'none'
        result['dominant_frac'] = 0.0

    return result


# ---------------------------------------------------------------------------
# 4. Spatial metrics
# ---------------------------------------------------------------------------

def compute_spatial_metrics(cells, pixel_size, marker_map):
    """Compute inter-cell spatial metrics within an islet.

    Returns dict with:
      - nn_{typeA}_{typeB}_um: mean nearest-neighbor distance from typeA to typeB
      - radial_{type}: mean normalized radial position (0=center, 1=edge)
      - mantle_core_index: radial_alpha - radial_beta (positive = classic architecture)
      - mixing_index: fraction of k=6 NN that differ in marker type
      - cell_density: cells per 1000 um2
    """
    marker_names = list(marker_map.keys())
    result = {}

    # Build coordinate arrays by type
    type_coords = {}
    all_coords = []
    all_types = []
    for c in cells:
        gc = c.get('global_center', c.get('center'))
        if gc is None:
            continue
        xy_um = np.array([gc[0] * pixel_size, gc[1] * pixel_size])
        mc = c.get('marker_class', 'none')
        all_coords.append(xy_um)
        all_types.append(mc)
        type_coords.setdefault(mc, []).append(xy_um)

    if len(all_coords) < 3:
        return result

    all_coords = np.array(all_coords)
    centroid = all_coords.mean(axis=0)

    # Cell density
    try:
        hull = ConvexHull(all_coords)
        hull_area = hull.volume  # 2D area
        result['cell_density_per_1000um2'] = round(len(all_coords) / hull_area * 1000, 2) if hull_area > 0 else 0.0
    except Exception:
        result['cell_density_per_1000um2'] = 0.0

    # Nearest-neighbor distances between type pairs
    type_arrays = {t: np.array(v) for t, v in type_coords.items()}
    for a in marker_names:
        for b in marker_names:
            if a not in type_arrays or b not in type_arrays:
                continue
            if len(type_arrays[a]) == 0 or len(type_arrays[b]) == 0:
                continue
            if a == b and len(type_arrays[a]) < 2:
                continue
            tree_b = cKDTree(type_arrays[b])
            k = 2 if a == b else 1  # skip self when same type
            dists, _ = tree_b.query(type_arrays[a], k=k)
            if a == b:
                dists = dists[:, 1] if dists.ndim > 1 else dists
            else:
                dists = dists[:, 0] if dists.ndim > 1 else dists
            result[f'nn_{a}_{b}_um'] = round(float(np.mean(dists)), 2)

    # Radial distribution (normalized: 0 = center, 1 = max radius)
    radii = np.linalg.norm(all_coords - centroid, axis=1)
    r_max = float(radii.max())
    max_r = r_max if r_max > 0 else 1.0

    for name in marker_names:
        if name in type_arrays and len(type_arrays[name]) > 0:
            type_radii = np.linalg.norm(type_arrays[name] - centroid, axis=1)
            result[f'radial_{name}'] = round(float(np.mean(type_radii / max_r)), 4)

    # Mantle-core index: radial_gcg - radial_ins (positive = classic mouse architecture)
    # Uses explicit marker names, not positional — robust to --marker-channels order
    r_alpha = result.get('radial_gcg')
    r_beta = result.get('radial_ins')
    if r_alpha is not None and r_beta is not None:
        result['mantle_core_index'] = round(r_alpha - r_beta, 4)

    # Mixing index: fraction of k-NN that differ in type (endocrine cells only)
    endo_mask = [t not in ('none', 'multi') for t in all_types]
    endo_coords_arr = all_coords[endo_mask]
    endo_types = [t for t, m in zip(all_types, endo_mask) if m]
    if len(endo_coords_arr) > 6:
        tree_endo = cKDTree(endo_coords_arr)
        k = min(7, len(endo_coords_arr))  # 6 neighbors + self
        _, nn_idx = tree_endo.query(endo_coords_arr, k=k)
        n_diff = 0
        n_total = 0
        for i in range(len(endo_coords_arr)):
            for j in nn_idx[i, 1:]:  # skip self
                if j < len(endo_types):
                    n_total += 1
                    if endo_types[j] != endo_types[i]:
                        n_diff += 1
        result['mixing_index'] = round(n_diff / n_total, 4) if n_total > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# 5. Atypical detection
# ---------------------------------------------------------------------------

def detect_atypical(composition, spatial, marker_names):
    """Flag atypical islet arrangements.

    Returns list of string flags.
    """
    flags = []

    # Use explicit marker names — robust to --marker-channels ordering
    frac_beta = composition.get('frac_ins', 0)
    if frac_beta < 0.30:
        flags.append('low_beta')

    frac_alpha = composition.get('frac_gcg', 0)
    if frac_alpha > 0.50:
        flags.append('alpha_dominant')

    mci = spatial.get('mantle_core_index')
    if mci is not None and mci < -0.1:
        flags.append('inverted_architecture')

    mixing = spatial.get('mixing_index')
    if mixing is not None and mixing < 0.1:
        flags.append('highly_segregated')

    return flags


# ---------------------------------------------------------------------------
# 5b. Cell-level and advanced per-islet metrics
# ---------------------------------------------------------------------------

def compute_cell_type_sizes(cells, pixel_size, marker_map):
    """Per-marker-type cell area statistics in um2."""
    result = {}
    px2 = pixel_size ** 2
    for name in list(marker_map.keys()) + ['multi', 'none']:
        areas = []
        for c in cells:
            if c.get('marker_class') != name:
                continue
            a = c.get('features', {}).get('area', 0)
            if a > 0:
                areas.append(a * px2)
        if areas:
            arr = np.array(areas)
            result[f'cellarea_mean_{name}'] = round(float(arr.mean()), 1)
            result[f'cellarea_median_{name}'] = round(float(np.median(arr)), 1)
    return result


def compute_coexpression(cells, marker_map, marker_thresholds):
    """Quantify co-expression patterns among 'multi' cells.

    marker_thresholds is (norm_ranges, ch_thresholds, ratio_min) tuple from
    compute_islet_marker_thresholds(), or None.
    Checks which channels exceed their normalized threshold for each multi cell,
    then counts co-expression pairs (e.g. coexpr_gcg_ins).
    """
    if not marker_thresholds:
        return {}
    norm_ranges, ch_thresholds, _ratio_min = marker_thresholds
    n_multi_cells = 0
    pair_counts = Counter()
    for c in cells:
        if c.get('marker_class') != 'multi':
            continue
        n_multi_cells += 1
        feats = c.get('features', {})
        above = []
        for name, ch_idx in marker_map.items():
            ch_key = f'ch{ch_idx}'
            lo, hi = norm_ranges.get(ch_key, (0, 1))
            raw_val = feats.get(f'{ch_key}_mean', 0)
            norm_val = max(0.0, min(1.0, (raw_val - lo) / (hi - lo))) if hi > lo else 0.0
            threshold = ch_thresholds.get(ch_key, 0.5)
            if norm_val >= threshold:
                above.append(name)
        for ai in range(len(above)):
            for bi in range(ai + 1, len(above)):
                pair = tuple(sorted([above[ai], above[bi]]))
                pair_counts[pair] += 1
    result = {}
    for (a, b), cnt in pair_counts.items():
        result[f'coexpr_{a}_{b}'] = cnt
    result['n_multi_cells'] = n_multi_cells
    result['n_coexpr_pairs'] = sum(pair_counts.values())
    return result


def compute_border_core(cells, pixel_size, marker_map):
    """Split cells into border vs core by radial position.

    Border = outer 30% of area (radius >= sqrt(0.7) ~= 0.84 of max radius).
    Returns per-marker border/core fractions and counts.
    """
    coords = []
    types = []
    for c in cells:
        gc = c.get('global_center', c.get('center'))
        if gc is None:
            continue
        coords.append([gc[0] * pixel_size, gc[1] * pixel_size])
        types.append(c.get('marker_class', 'none'))
    if len(coords) < 5:
        return {}
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    max_r = float(radii.max())
    if max_r == 0:
        return {}
    norm_radii = radii / max_r
    border_mask = norm_radii >= math.sqrt(0.70)  # outer 30% by area
    result = {}
    for name in marker_map.keys():
        type_mask = np.array([t == name for t in types])
        n_type = int(type_mask.sum())
        if n_type > 0:
            n_border = int((type_mask & border_mask).sum())
            result[f'border_frac_{name}'] = round(n_border / n_type, 3)
            result[f'core_frac_{name}'] = round(1.0 - n_border / n_type, 3)
    result['n_border'] = int(border_mask.sum())
    result['n_core'] = len(coords) - result['n_border']
    return result


def compute_density_gradient(cells, pixel_size):
    """Compare cell density in core vs periphery using median-radius split.

    Returns core/border density per 1000 um2 and their ratio.
    """
    coords = []
    for c in cells:
        gc = c.get('global_center', c.get('center'))
        if gc is None:
            continue
        coords.append([gc[0] * pixel_size, gc[1] * pixel_size])
    if len(coords) < 10:
        return {}
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    max_r = float(radii.max())
    if max_r == 0:
        return {}
    median_r = float(np.median(radii))
    inner_area = math.pi * median_r ** 2
    outer_area = math.pi * max_r ** 2 - inner_area
    n_inner = int((radii <= median_r).sum())
    n_outer = len(coords) - n_inner
    core_dens = (n_inner / inner_area * 1000) if inner_area > 0 else 0
    border_dens = (n_outer / outer_area * 1000) if outer_area > 0 else 0
    return {
        'core_density_per_1000um2': round(core_dens, 2),
        'border_density_per_1000um2': round(border_dens, 2),
        'density_ratio': round(core_dens / border_dens, 2) if border_dens > 0 else 0.0,
    }


def compute_radial_profile(cells, pixel_size, marker_map, n_bins=5):
    """Radial composition profile: marker fractions in concentric bins.

    Returns per-marker semicolon-separated bin fractions and gradient slope.
    Positive gradient = marker enriched at periphery.
    """
    coords = []
    types = []
    for c in cells:
        gc = c.get('global_center', c.get('center'))
        if gc is None:
            continue
        coords.append([gc[0] * pixel_size, gc[1] * pixel_size])
        types.append(c.get('marker_class', 'none'))
    if len(coords) < n_bins * 2:
        return {}
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    max_r = float(radii.max())
    if max_r == 0:
        return {}
    norm_radii = radii / max_r
    bin_edges = np.linspace(0, 1, n_bins + 1)
    types_arr = np.array(types)
    result = {}
    for name in marker_map.keys():
        fracs = []
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b < n_bins - 1:
                bin_mask = (norm_radii >= lo) & (norm_radii < hi)
            else:
                bin_mask = (norm_radii >= lo) & (norm_radii <= hi)
            n_bin = int(bin_mask.sum())
            n_type = int(((types_arr == name) & bin_mask).sum())
            fracs.append(round(n_type / n_bin, 3) if n_bin > 0 else 0.0)
        result[f'radial_profile_{name}'] = ';'.join(str(v) for v in fracs)
        if len(fracs) >= 2 and sum(fracs) > 0:
            slope = float(np.polyfit(np.arange(n_bins), fracs, 1)[0])
            result[f'radial_gradient_{name}'] = round(slope, 4)
    return result


def compute_homogeneity(cells, pixel_size, marker_map):
    """Spatial homogeneity: variance of marker fractions across quadrants.

    Score 0-1 (higher = more spatially uniform distribution of cell types).
    """
    coords = []
    types = []
    for c in cells:
        gc = c.get('global_center', c.get('center'))
        if gc is None:
            continue
        coords.append([gc[0] * pixel_size, gc[1] * pixel_size])
        types.append(c.get('marker_class', 'none'))
    if len(coords) < 20:
        return {}
    coords = np.array(coords)
    types_arr = np.array(types)
    centroid = coords.mean(axis=0)
    quadrant_fracs = {name: [] for name in marker_map.keys()}
    for qx in [False, True]:
        for qy in [False, True]:
            q_mask = ((coords[:, 0] >= centroid[0]) == qx) & \
                     ((coords[:, 1] >= centroid[1]) == qy)
            n_q = int(q_mask.sum())
            if n_q < 3:
                continue
            for name in marker_map.keys():
                n_type = int(((types_arr == name) & q_mask).sum())
                quadrant_fracs[name].append(n_type / n_q)
    result = {}
    variances = []
    for name in marker_map.keys():
        if len(quadrant_fracs[name]) >= 3:
            v = float(np.var(quadrant_fracs[name]))
            result[f'homogeneity_var_{name}'] = round(v, 4)
            variances.append(v)
    if variances:
        # Sigmoid-like scaling: 1/(1+20*var) gives better dynamic range than linear
        result['homogeneity_score'] = round(
            1.0 / (1.0 + 20.0 * float(np.mean(variances))), 3)
    return result


def compute_packing_density(cells, pixel_size):
    """Ratio of total cell mask area to convex hull area (cellularity proxy)."""
    coords = []
    cell_areas_um2 = []
    px2 = pixel_size ** 2
    for c in cells:
        gc = c.get('global_center', c.get('center'))
        if gc is None:
            continue
        coords.append([gc[0] * pixel_size, gc[1] * pixel_size])
        a = c.get('features', {}).get('area', 0)
        if a > 0:
            cell_areas_um2.append(a * px2)
    if len(coords) < 3 or not cell_areas_um2:
        return {}
    coords = np.array(coords)
    try:
        hull = ConvexHull(coords)
        hull_area = hull.volume
    except Exception:
        return {}
    if hull_area <= 0:
        return {}
    total_cell_area = sum(cell_areas_um2)
    return {
        'packing_density': round(total_cell_area / hull_area, 4),
        'total_cell_area_um2': round(total_cell_area, 1),
    }


# ---------------------------------------------------------------------------
# 6. Master analysis function
# ---------------------------------------------------------------------------

def analyze_all_islets(detections, islet_groups, pixel_size, marker_map,
                       marker_thresholds=None):
    """Run all analyses on all islets. Returns list of islet feature dicts."""
    marker_names = list(marker_map.keys())
    results = []

    for iid in sorted(islet_groups.keys()):
        idx_list = islet_groups[iid]
        cells = [detections[i] for i in idx_list]
        if len(cells) < 3:
            continue

        row = {'islet_id': iid}
        row.update(compute_islet_shape(cells, pixel_size))
        row.update(compute_composition(cells, marker_map))
        spatial = compute_spatial_metrics(cells, pixel_size, marker_map)
        row.update(spatial)
        row.update(compute_cell_type_sizes(cells, pixel_size, marker_map))
        row.update(compute_coexpression(cells, marker_map, marker_thresholds))
        row.update(compute_border_core(cells, pixel_size, marker_map))
        row.update(compute_density_gradient(cells, pixel_size))
        row.update(compute_radial_profile(cells, pixel_size, marker_map))
        row.update(compute_homogeneity(cells, pixel_size, marker_map))
        row.update(compute_packing_density(cells, pixel_size))
        row['atypical_flags'] = detect_atypical(row, spatial, marker_names)
        row['n_flags'] = len(row['atypical_flags'])

        results.append(row)

    return results


def compute_tissue_level(islet_features, detections, pixel_size):
    """Compute tissue-level statistics across all islets.

    Mutates islet_features to add per-islet nn_islet_distance_um and
    dist_to_tissue_edge_um. Returns dict of global tissue-level stats.
    """
    result = {}
    if not islet_features:
        return result

    # Islet-islet proximity (centroid nearest-neighbor distances)
    indexed_centroids = []
    for i, f in enumerate(islet_features):
        if 'centroid_x_um' in f:
            indexed_centroids.append((i, [f['centroid_x_um'], f['centroid_y_um']]))
    if len(indexed_centroids) >= 2:
        orig_indices, coords_list = zip(*indexed_centroids)
        centroids = np.array(coords_list)
        tree = cKDTree(centroids)
        dists, _ = tree.query(centroids, k=2)
        nn_dists = dists[:, 1]
        result['mean_inter_islet_um'] = round(float(nn_dists.mean()), 1)
        result['median_inter_islet_um'] = round(float(np.median(nn_dists)), 1)
        result['min_inter_islet_um'] = round(float(nn_dists.min()), 1)
        result['max_inter_islet_um'] = round(float(nn_dists.max()), 1)
        for j, orig_idx in enumerate(orig_indices):
            islet_features[orig_idx]['nn_islet_distance_um'] = round(
                float(nn_dists[j]), 1)

    # Total endocrine area
    total_islet_area = sum(f.get('area_um2', 0) for f in islet_features)
    result['total_islet_area_um2'] = round(total_islet_area, 0)
    result['n_islets'] = len(islet_features)

    # Tissue area from convex hull of ALL cell positions
    all_coords = []
    for d in detections:
        gc = d.get('global_center', d.get('center'))
        if gc is not None:
            all_coords.append([gc[0] * pixel_size, gc[1] * pixel_size])
    if len(all_coords) >= 3:
        all_coords_arr = np.array(all_coords)
        try:
            tissue_hull = ConvexHull(all_coords_arr)
            tissue_area = tissue_hull.volume
            result['tissue_area_um2'] = round(tissue_area, 0)
            if tissue_area > 0:
                result['endocrine_area_fraction'] = round(
                    total_islet_area / tissue_area, 4)
                tissue_area_mm2 = tissue_area / 1e6
                result['islet_density_per_mm2'] = round(
                    len(islet_features) / tissue_area_mm2, 2)

            # Distance to tissue edge per islet (exact via hull facet equations)
            equations = tissue_hull.equations  # (n_facets, ndim+1)
            for f in islet_features:
                if 'centroid_x_um' in f and 'centroid_y_um' in f:
                    pt = np.array([f['centroid_x_um'], f['centroid_y_um']])
                    signed_dists = equations[:, :2] @ pt + equations[:, 2]
                    max_sd = float(signed_dists.max())
                    # Interior: all signed_dists <= 0, distance = -max
                    f['dist_to_tissue_edge_um'] = round(
                        -max_sd, 1) if max_sd <= 0 else 0.0
        except Exception:
            pass

    # Size-composition correlations (Spearman)
    if len(islet_features) >= 5:
        from scipy.stats import spearmanr
        sizes = [f['n_cells'] for f in islet_features]
        frac_keys = [k for k in islet_features[0].keys()
                     if k.startswith('frac_') and k not in ('frac_multi', 'frac_none')]
        for key in frac_keys:
            fracs = [f.get(key, 0) for f in islet_features]
            try:
                rho, pval = spearmanr(sizes, fracs)
                if not np.isnan(rho):
                    result[f'corr_size_{key}'] = round(float(rho), 3)
                    result[f'corr_size_{key}_p'] = round(float(pval), 4)
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------

def export_csv(islet_features, output_path):
    """Export islet features to CSV. Does not mutate input."""
    import csv
    if not islet_features:
        print("No islets to export")
        return

    # Work on copies to avoid mutating shared data
    rows = []
    for row in islet_features:
        r = dict(row)
        flags = r.get('atypical_flags', [])
        r['atypical_flags'] = ';'.join(flags) if isinstance(flags, list) else (flags or '')
        rows.append(r)

    # Collect all keys across all rows (not just first — later rows may have extra keys)
    all_keys = dict.fromkeys(k for r in rows for k in r.keys())
    fieldnames = list(all_keys)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {output_path} ({len(rows)} islets)")


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def export_detections(detections, output_path):
    """Export enriched detections JSON (with islet_id + marker_class)."""
    clean = sanitize_for_json(detections)
    with open(output_path, 'w') as f:
        json.dump(clean, f, indent=1)
    print(f"Saved enriched detections: {output_path}")


# ---------------------------------------------------------------------------
# 8. HTML visualization
# ---------------------------------------------------------------------------

def pct_norm(img):
    """Percentile-normalize a multi-channel image, preserving zero (padding) pixels."""
    out = np.zeros_like(img, dtype=np.uint8)
    valid = np.any(img > 0, axis=-1)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(float)
        vals = ch[valid]
        if len(vals) == 0:
            continue
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99.5)
        if hi > lo:
            out[:, :, c] = np.clip(255 * (ch - lo) / (hi - lo), 0, 255).astype(np.uint8)
    return out


def draw_dashed_contours(img, contours, color, thickness=1, dash_len=6, gap_len=4):
    """Draw dashed contour lines on img in-place."""
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 2:
            continue
        all_pts = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dist = np.linalg.norm(p2 - p1)
            if dist < 1:
                continue
            n_steps = max(int(dist), 1)
            for t in np.linspace(0, 1, n_steps, endpoint=False):
                all_pts.append((p1 + t * (p2 - p1)).astype(int))
        cycle = dash_len + gap_len
        for i, pt in enumerate(all_pts):
            if (i % cycle) < dash_len:
                cv2.circle(img, tuple(pt), 0, color, thickness)


def render_islet_card(islet_feat, cells, masks, tile_vis, tile_x, tile_y,
                      tile_h, tile_w, pixel_size, marker_colors, marker_map):
    """Render a single islet card with crop image + stats. Returns HTML string or None."""
    marker_names = list(marker_map.keys())

    # Collect cell info within this tile
    mask_labels = []
    cell_info = []
    for d in cells:
        gc = d.get('global_center', d.get('center', [0, 0]))
        cx_rel = gc[0] - tile_x
        cy_rel = gc[1] - tile_y
        ml = d.get('mask_label')
        cell_info.append((cx_rel, cy_rel, ml, d.get('marker_class', 'none')))
        if ml is not None and ml > 0:
            mask_labels.append(ml)

    if not mask_labels:
        return None

    # Union mask + dilate for islet boundary
    union_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    for ml in mask_labels:
        union_mask |= (masks == ml).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(union_mask, kernel)

    ys, xs = np.where(dilated > 0)
    if len(xs) == 0:
        return None

    x_min = max(0, int(xs.min()) - PADDING)
    x_max = min(tile_w, int(xs.max()) + PADDING)
    y_min = max(0, int(ys.min()) - PADDING)
    y_max = min(tile_h, int(ys.max()) + PADDING)

    crop = tile_vis[y_min:y_max, x_min:x_max].copy()

    # Dashed cell contours colored by marker
    for _cx, _cy, ml, mc in cell_info:
        color = marker_colors.get(mc, (128, 128, 128))
        if ml is not None and ml > 0:
            mask_crop = (masks[y_min:y_max, x_min:x_max] == ml).astype(np.uint8)
            if mask_crop.any():
                cnts, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                draw_dashed_contours(crop, cnts, color, thickness=1)

    # Solid pink islet boundary
    boundary_crop = dilated[y_min:y_max, x_min:x_max]
    cnts, _ = cv2.findContours(boundary_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnts, -1, PINK, 2, cv2.LINE_AA)

    _, buf = cv2.imencode('.png', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf).decode()

    # --- Stats panel ---
    iid = islet_feat['islet_id']
    n = islet_feat['n_cells']
    area = islet_feat.get('area_um2', 0)
    circ = islet_feat.get('circularity', 0)
    elong = islet_feat.get('elongation', 1)
    entropy = islet_feat.get('entropy', 0)
    dominant = islet_feat.get('dominant_type', '?')
    dom_frac = islet_feat.get('dominant_frac', 0)
    mci = islet_feat.get('mantle_core_index')
    mixing = islet_feat.get('mixing_index')
    density = islet_feat.get('cell_density_per_1000um2', 0)
    flags = islet_feat.get('atypical_flags', [])
    if isinstance(flags, str):
        flags = flags.split(';') if flags else []

    # Composition bar
    total = max(n, 1)
    bar_parts = []
    for i, name in enumerate(marker_names):
        cnt = islet_feat.get(f'n_{name}', 0)
        if cnt > 0:
            html_c = _MARKER_HTML_COLORS[i] if i < len(_MARKER_HTML_COLORS) else '#cccccc'
            bar_parts.append(f'<div style="width:{100*cnt/total:.0f}%;background:{html_c}" '
                             f'title="{name} {cnt} ({100*cnt/total:.0f}%)"></div>')
    n_multi = islet_feat.get('n_multi', 0)
    if n_multi > 0:
        bar_parts.append(f'<div style="width:{100*n_multi/total:.0f}%;background:#ffaa00" '
                         f'title="multi {n_multi}"></div>')
    n_none = islet_feat.get('n_none', 0)
    if n_none > 0:
        bar_parts.append(f'<div style="width:{100*n_none/total:.0f}%;background:#555" '
                         f'title="none {n_none}"></div>')
    bar_html = (f'<div style="display:flex;height:10px;width:100%;border-radius:4px;'
                f'overflow:hidden;margin:4px 0">{"".join(bar_parts)}</div>')

    # Count labels
    count_parts = []
    for i, name in enumerate(marker_names):
        cnt = islet_feat.get(f'n_{name}', 0)
        html_c = _MARKER_HTML_COLORS[i] if i < len(_MARKER_HTML_COLORS) else '#ccc'
        count_parts.append(f'<span style="color:{html_c}">{name}:{cnt}</span>')
    if n_multi > 0:
        count_parts.append(f'<span style="color:#ffaa00">multi:{n_multi}</span>')
    counts_str = ' '.join(count_parts)

    # Spatial stats line
    spatial_parts = []
    if mci is not None:
        spatial_parts.append(f'MCI={mci:.2f}')
    if mixing is not None:
        spatial_parts.append(f'mix={mixing:.2f}')
    spatial_parts.append(f'dens={density:.1f}/1000um\u00b2')
    spatial_str = ' | '.join(spatial_parts)

    # New per-islet metrics
    new_parts = []
    packing = islet_feat.get('packing_density')
    if packing is not None:
        new_parts.append(f'pack={packing:.2f}')
    dratio = islet_feat.get('density_ratio')
    if dratio is not None:
        new_parts.append(f'd_ratio={dratio:.1f}')
    homo = islet_feat.get('homogeneity_score')
    if homo is not None:
        new_parts.append(f'homo={homo:.2f}')
    nn_islet = islet_feat.get('nn_islet_distance_um')
    if nn_islet is not None:
        new_parts.append(f'nn_islet={nn_islet:.0f}um')
    edge_dist = islet_feat.get('dist_to_tissue_edge_um')
    if edge_dist is not None:
        new_parts.append(f'edge={edge_dist:.0f}um')
    new_str = ' | '.join(new_parts)

    # Border/core bar
    n_border = islet_feat.get('n_border', 0)
    n_core = islet_feat.get('n_core', 0)
    border_bar = ''
    if n_border + n_core > 0:
        bp = 100 * n_border / (n_border + n_core)
        border_bar = (f'<div style="display:flex;height:6px;width:100%;border-radius:3px;'
                      f'overflow:hidden;margin:2px 0">'
                      f'<div style="width:{100-bp:.0f}%;background:#555" title="core {n_core}"></div>'
                      f'<div style="width:{bp:.0f}%;background:#ff00ff" title="border {n_border}"></div></div>')

    # Co-expression note
    coexpr_parts = []
    for key, val in islet_feat.items():
        if key.startswith('coexpr_') and val > 0:
            pair = key.replace('coexpr_', '').replace('_', '+')
            coexpr_parts.append(f'{pair}:{val}')
    coexpr_str = ' '.join(coexpr_parts)

    # Flags
    flag_html = ''
    if flags:
        flag_html = (' '.join(f'<span style="color:#ff6666;background:#330000;padding:1px 4px;'
                               f'border-radius:3px;font-size:10px">{f}</span>' for f in flags))

    # Pie chart SVG
    pie_svg = _make_pie_svg(islet_feat, marker_names, size=60)

    card = f'''
    <div class="card" style="display:inline-block;margin:10px;background:#111;border:2px solid #333;
         border-radius:8px;padding:10px;vertical-align:top;max-width:{crop.shape[1]+40}px">
        <img src="data:image/png;base64,{b64}" style="display:block;border-radius:4px">
        <div style="color:white;font-family:monospace;font-size:12px;margin-top:6px">
            <div style="display:flex;align-items:center;gap:8px">
                <b style="color:#ff00ff">Islet {iid}</b>
                {pie_svg}
                <span>{n} cells</span>
            </div>
            {bar_html}
            <div>{counts_str}</div>
            <div style="color:#aaa;font-size:11px">
                area={area:.0f}um\u00b2 circ={circ:.2f} elong={elong:.1f}
                H={entropy:.2f} dom={dominant}({dom_frac:.0%})
            </div>
            <div style="color:#aaa;font-size:11px">{spatial_str}</div>
            {f'<div style="color:#aaa;font-size:11px">{new_str}</div>' if new_str else ''}
            {border_bar}
            {f'<div style="color:#ffaa00;font-size:10px">bihormonal: {coexpr_str}</div>' if coexpr_str else ''}
            {f'<div style="margin-top:4px">{flag_html}</div>' if flag_html else ''}
        </div>
    </div>'''
    return card


def _make_pie_svg(islet_feat, marker_names, size=60):
    """Tiny SVG pie chart of marker composition."""
    r = size / 2
    cx, cy = r, r
    slices = []
    colors = list(_MARKER_HTML_COLORS[:len(marker_names)]) + ['#ffaa00', '#555']
    labels = list(marker_names) + ['multi', 'none']
    values = [islet_feat.get(f'n_{n}', 0) for n in marker_names]
    values.append(islet_feat.get('n_multi', 0))
    values.append(islet_feat.get('n_none', 0))
    total = sum(values)
    if total == 0:
        return ''

    angle = -90  # start at top
    for val, color in zip(values, colors):
        if val == 0:
            continue
        frac = val / total
        sweep = frac * 360
        if sweep >= 359.99:
            # Full circle
            slices.append(f'<circle cx="{cx}" cy="{cy}" r="{r-1}" fill="{color}" opacity="0.8"/>')
            break
        end_angle = angle + sweep
        x1 = cx + (r - 1) * math.cos(math.radians(angle))
        y1 = cy + (r - 1) * math.sin(math.radians(angle))
        x2 = cx + (r - 1) * math.cos(math.radians(end_angle))
        y2 = cy + (r - 1) * math.sin(math.radians(end_angle))
        large = 1 if sweep > 180 else 0
        slices.append(
            f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{r-1},{r-1} 0 {large} 1 '
            f'{x2:.1f},{y2:.1f} Z" fill="{color}" opacity="0.8"/>'
        )
        angle = end_angle

    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
            f'style="flex-shrink:0">{"".join(slices)}</svg>')


def generate_html(islet_features, detections, islet_groups, tile_info, tile_masks_cache,
                  tile_vis_cache, pixel_size, marker_map, marker_colors, slide_name,
                  tile_h, tile_w, output_path, tissue_stats=None):
    """Generate full HTML analysis page."""
    marker_names = list(marker_map.keys())

    # Map islet -> best tile (most interior)
    islet_tile_map = {}
    for iid in islet_groups:
        cells = [detections[i] for i in islet_groups[iid]]
        coords = np.array([
            c.get('global_center', c.get('center', [0, 0])) for c in cells
        ])
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        best_tile = None
        best_margin = -1e9
        for (tx, ty) in tile_info:
            margin = min(xmin - tx, (tx + tile_w) - xmax, ymin - ty, (ty + tile_h) - ymax)
            if margin > best_margin:
                best_margin = margin
                best_tile = (tx, ty)
        if best_tile is not None:
            islet_tile_map[iid] = best_tile

    # Sort by cell count descending
    sorted_feats = sorted(islet_features, key=lambda x: x['n_cells'], reverse=True)

    card_parts = []
    rendered = 0
    for feat in sorted_feats:
        iid = feat['islet_id']
        tile_key = islet_tile_map.get(iid)
        if tile_key is None or tile_key not in tile_masks_cache:
            continue
        tx, ty = tile_key
        masks = tile_masks_cache[tile_key]
        tile_vis = tile_vis_cache[tile_key]

        # Filter to cells within this tile (avoid cross-tile mask label collisions)
        all_cells = [detections[i] for i in islet_groups[iid]]
        cells = [c for c in all_cells
                 if tx <= c.get('global_center', c.get('center', [0, 0]))[0] < tx + tile_w
                 and ty <= c.get('global_center', c.get('center', [0, 0]))[1] < ty + tile_h]
        if not cells:
            continue

        card = render_islet_card(feat, cells, masks, tile_vis, tx, ty, tile_h, tile_w,
                                 pixel_size, marker_colors, marker_map)
        if card:
            card_parts.append(card)
            rendered += 1
    cards_html = ''.join(card_parts)

    # Summary stats
    n_islets = len(islet_features)
    total_cells = sum(f['n_cells'] for f in islet_features)
    median_area = np.median([f['area_um2'] for f in islet_features]) if islet_features else 0
    n_flagged = sum(1 for f in islet_features if f.get('n_flags', 0) > 0)

    # Histogram data (area distribution)
    areas = [f['area_um2'] for f in islet_features]
    area_hist_svg = _make_histogram_svg(areas, 'Islet Area (um\u00b2)', width=300, height=100)

    # Composition summary pie
    total_comp = Counter()
    for f in islet_features:
        for name in marker_names:
            total_comp[name] += f.get(f'n_{name}', 0)
        total_comp['multi'] += f.get('n_multi', 0)
        total_comp['none'] += f.get('n_none', 0)
    summary_pie = _make_summary_pie_svg(total_comp, marker_names, size=80)

    # Legend
    legend_parts = []
    for i, name in enumerate(marker_names):
        c = _MARKER_HTML_COLORS[i] if i < len(_MARKER_HTML_COLORS) else '#ccc'
        legend_parts.append(f'<span style="color:{c}">{name}</span>')
    legend_parts.append('<span style="color:#ffaa00">multi</span>')
    legend_parts.append('<span style="color:#888">none</span>')
    legend_html = ' | '.join(legend_parts)

    # Tissue-level summary
    tissue_html = ''
    if tissue_stats:
        t_parts = []
        eaf = tissue_stats.get('endocrine_area_fraction')
        if eaf is not None:
            t_parts.append(f'<div class="stat">endocrine area: <b>{eaf:.1%}</b></div>')
        idmm = tissue_stats.get('islet_density_per_mm2')
        if idmm is not None:
            t_parts.append(f'<div class="stat">density: <b>{idmm:.1f}</b> islets/mm\u00b2</div>')
        mii = tissue_stats.get('median_inter_islet_um')
        if mii is not None:
            t_parts.append(f'<div class="stat">inter-islet: <b>{mii:.0f}</b> um (median)</div>')
        # Size-composition correlations
        corr_parts = []
        for key, val in tissue_stats.items():
            if key.startswith('corr_size_frac_') and not key.endswith('_p'):
                marker = key.replace('corr_size_frac_', '')
                pval = tissue_stats.get(f'{key}_p', 1.0)
                sig = '*' if pval < 0.05 else ''
                corr_parts.append(f'{marker}: rho={val:.2f}{sig}')
        if corr_parts:
            t_parts.append(f'<div class="stat">size-composition: {" | ".join(corr_parts)}</div>')
        if t_parts:
            tissue_html = (f'<div class="summary" style="border-left:3px solid #ff00ff">'
                          f'{"".join(t_parts)}</div>')

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Islet Analysis</title>
<style>
body {{ background: #000; margin: 20px; color: white; font-family: monospace; }}
h1 {{ font-family: sans-serif; }}
.summary {{ background: #111; padding: 16px; border-radius: 8px; margin-bottom: 20px;
            display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }}
.stat {{ font-size: 14px; }}
.stat b {{ color: #ff00ff; font-size: 18px; }}
.sub {{ color: #aaa; font-size: 13px; margin-bottom: 10px; }}
</style></head><body>
<h1>Islet Spatial Analysis &mdash; {slide_name}</h1>
<div class="summary">
    {summary_pie}
    <div>
        <div class="stat"><b>{n_islets}</b> islets | <b>{total_cells}</b> cells</div>
        <div class="stat">median area: <b>{median_area:.0f}</b> um\u00b2</div>
        <div class="stat"><b>{n_flagged}</b> flagged atypical</div>
    </div>
    {area_hist_svg}
</div>
{tissue_html}
<div class="sub">
    <span style="color:#ff00ff">pink = islet boundary</span> | {legend_html} (dashed contours) |
    MCI = mantle-core index | mix = mixing index
</div>
{cards_html}
<div style="color:#555;font-size:11px;margin-top:20px">
    Generated by analyze_islets.py | {rendered} islet cards rendered
</div>
</body></html>'''

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Saved HTML: {output_path} ({len(html)/1024:.0f} KB)")


def _make_histogram_svg(values, title, width=300, height=100, n_bins=15):
    """Simple SVG histogram."""
    if not values:
        return ''
    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = max(counts) if max(counts) > 0 else 1
    bar_w = width / n_bins

    bars = []
    for i, cnt in enumerate(counts):
        h = (cnt / max_count) * (height - 20)
        x = i * bar_w
        y = height - 20 - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-1:.1f}" height="{h:.1f}" '
                    f'fill="#ff00ff" opacity="0.6"/>')

    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'style="flex-shrink:0">'
            f'<text x="{width/2}" y="12" fill="#aaa" font-size="10" text-anchor="middle">{title}</text>'
            f'{"".join(bars)}'
            f'<text x="0" y="{height-2}" fill="#666" font-size="9">{edges[0]:.0f}</text>'
            f'<text x="{width}" y="{height-2}" fill="#666" font-size="9" text-anchor="end">{edges[-1]:.0f}</text>'
            f'</svg>')


def _make_summary_pie_svg(counts, marker_names, size=80):
    """Summary pie chart SVG for the header."""
    colors = list(_MARKER_HTML_COLORS[:len(marker_names)]) + ['#ffaa00', '#555']
    labels = list(marker_names) + ['multi', 'none']
    values = [counts.get(n, 0) for n in labels]
    total = sum(values)
    if total == 0:
        return ''

    r = size / 2
    cx, cy = r, r
    slices = []
    angle = -90
    for val, color, label in zip(values, colors, labels):
        if val == 0:
            continue
        frac = val / total
        sweep = frac * 360
        if sweep >= 359.99:
            slices.append(f'<circle cx="{cx}" cy="{cy}" r="{r-1}" fill="{color}" opacity="0.8"/>')
            break
        end_angle = angle + sweep
        x1 = cx + (r - 1) * math.cos(math.radians(angle))
        y1 = cy + (r - 1) * math.sin(math.radians(angle))
        x2 = cx + (r - 1) * math.cos(math.radians(end_angle))
        y2 = cy + (r - 1) * math.sin(math.radians(end_angle))
        large = 1 if sweep > 180 else 0
        slices.append(
            f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} A{r-1},{r-1} 0 {large} 1 '
            f'{x2:.1f},{y2:.1f} Z" fill="{color}" opacity="0.8"/>'
        )
        angle = end_angle

    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
            f'style="flex-shrink:0">{"".join(slices)}</svg>')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyze pancreatic islets: morphometry, composition, spatial metrics'
    )
    parser.add_argument('--run-dir', required=True,
                        help='Path to completed islet run output directory')
    parser.add_argument('--czi-path', required=True,
                        help='Path to CZI file (for loading display channels)')
    parser.add_argument('--buffer-um', type=float, default=25.0,
                        help='Buffer distance in um: marker+ cells within this distance '
                             'are grouped into the same islet, and none/multi cells within '
                             'this distance are recruited (default: 25)')
    parser.add_argument('--min-cells', type=int, default=5,
                        help='Minimum marker+ cells to form an islet (default: 5)')
    parser.add_argument('--display-channels', type=str, default='2,3,5',
                        help='R,G,B channel indices for display (default: 2,3,5)')
    parser.add_argument('--marker-channels', type=str, default='gcg:2,ins:3,sst:5',
                        help='Marker-to-channel mapping (default: gcg:2,ins:3,sst:5)')
    parser.add_argument('--no-html', action='store_true',
                        help='Skip HTML generation (just CSV + JSON)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    run_dir = Path(args.run_dir)
    czi_path = Path(args.czi_path)

    # Parse config
    display_chs = [int(x.strip()) for x in args.display_channels.split(',')]
    marker_map = {}
    for pair in args.marker_channels.split(','):
        name, ch = pair.strip().split(':')
        marker_map[name.strip()] = int(ch.strip())
    marker_names = list(marker_map.keys())

    # 1. Load detections
    detections = load_detections(run_dir)
    if not detections:
        print("No detections — exiting")
        sys.exit(0)

    # 2. Classify markers (need enough detections for reliable thresholds)
    marker_thresholds = classify_all(detections, marker_map)
    n_endocrine = sum(1 for d in detections if d.get('marker_class', 'none') != 'none')
    if n_endocrine == 0:
        print("ERROR: No marker-positive cells found. Cannot define islets.")
        sys.exit(1)

    # 3. Get pixel size from CZI
    print(f"Loading CZI metadata from {czi_path}...")
    loader = get_loader(czi_path, load_to_ram=False, channel=display_chs[0])
    pixel_size = loader.get_pixel_size()
    x_start = loader.x_start
    y_start = loader.y_start
    print(f"  pixel_size={pixel_size:.4f} um/px")

    # 4. Define islets (union of bright cells + buffer)
    islet_groups = define_islets(detections, pixel_size,
                                buffer_um=args.buffer_um,
                                min_cells=args.min_cells)

    if not islet_groups:
        print("No islets defined — exiting")
        sys.exit(0)

    print(f"{len(islet_groups)} islets defined")

    # 5. Analyze all islets
    print("Computing islet features...")
    islet_features = analyze_all_islets(detections, islet_groups, pixel_size, marker_map,
                                        marker_thresholds=marker_thresholds)
    print(f"  {len(islet_features)} islets analyzed")

    # 5b. Tissue-level analysis
    print("Computing tissue-level metrics...")
    tissue_stats = compute_tissue_level(islet_features, detections, pixel_size)

    # Print summary
    if islet_features:
        areas = [f['area_um2'] for f in islet_features]
        cells_per = [f['n_cells'] for f in islet_features]
        n_flagged = sum(1 for f in islet_features if f.get('n_flags', 0) > 0)
        print(f"\n--- Summary ---")
        print(f"  Islets: {len(islet_features)}")
        print(f"  Cells: {sum(cells_per)} total, median {np.median(cells_per):.0f}/islet")
        print(f"  Area: median {np.median(areas):.0f} um2, range {min(areas):.0f}-{max(areas):.0f}")
        print(f"  Flagged atypical: {n_flagged}")
        for name in marker_names:
            fracs = [f.get(f'frac_{name}', 0) for f in islet_features]
            print(f"  {name}: median frac {np.median(fracs):.2f}")
        mcis = [f.get('mantle_core_index') for f in islet_features if f.get('mantle_core_index') is not None]
        if mcis:
            print(f"  Mantle-core index: median {np.median(mcis):.3f}")
        mixes = [f.get('mixing_index') for f in islet_features if f.get('mixing_index') is not None]
        if mixes:
            print(f"  Mixing index: median {np.median(mixes):.3f}")
        # Tissue-level
        eaf = tissue_stats.get('endocrine_area_fraction')
        if eaf is not None:
            print(f"  Endocrine area fraction: {eaf:.2%}")
        idmm = tissue_stats.get('islet_density_per_mm2')
        if idmm is not None:
            print(f"  Islet density: {idmm:.1f} /mm2")
        mii = tissue_stats.get('median_inter_islet_um')
        if mii is not None:
            print(f"  Inter-islet distance: median {mii:.0f} um")

    # 6. Export CSV
    csv_path = run_dir / 'islet_summary.csv'
    export_csv(islet_features, csv_path)

    # 7. Export enriched detections
    det_out = run_dir / 'islet_detections_analyzed.json'
    export_detections(detections, det_out)

    # 8. HTML visualization
    if not args.no_html:
        print("\nGenerating HTML visualization...")

        tiles_dir = run_dir / 'tiles'
        if not tiles_dir.exists():
            print(f"Tiles directory not found: {tiles_dir} — skipping HTML")
        else:
            # Discover tiles
            tile_info = {}
            for td in sorted(tiles_dir.iterdir()):
                if not td.is_dir() or not td.name.startswith('tile_'):
                    continue
                parts = td.name.split('_')
                if len(parts) < 3:
                    continue
                try:
                    tx, ty = int(parts[1]), int(parts[2])
                    tile_info[(tx, ty)] = td
                except ValueError:
                    continue

            if not tile_info:
                print("No tile directories found — skipping HTML")
            else:
                # Load CZI display channels
                loader = get_loader(czi_path, load_to_ram=True, channel=display_chs[0])
                ch_data = {}
                for ch in display_chs:
                    print(f"  Loading channel {ch}...")
                    loader.load_channel(ch)
                    ch_data[ch] = loader._channel_data[ch]

                # Get tile dimensions from first tile
                first_td = next(iter(tile_info.values()))
                mask_path = first_td / 'islet_masks.h5'
                with h5py.File(mask_path, 'r') as f:
                    first_masks = f['masks'][:]
                tile_h, tile_w = first_masks.shape[:2]
                del first_masks

                # Load needed tiles
                needed_tiles = set()
                for iid in islet_groups:
                    cells = [detections[i] for i in islet_groups[iid]]
                    for c in cells:
                        gc = c.get('global_center', c.get('center', [0, 0]))
                        for (tx, ty) in tile_info:
                            if tx <= gc[0] < tx + tile_w and ty <= gc[1] < ty + tile_h:
                                needed_tiles.add((tx, ty))
                                break

                tile_masks_cache = {}
                tile_vis_cache = {}
                for (tx, ty) in sorted(needed_tiles):
                    td = tile_info.get((tx, ty))
                    if td is None:
                        continue
                    mp = td / 'islet_masks.h5'
                    if not mp.exists():
                        continue
                    with h5py.File(mp, 'r') as f:
                        masks = f['masks'][:]
                    tile_masks_cache[(tx, ty)] = masks
                    th, tw = masks.shape[:2]
                    rel_tx = tx - x_start
                    rel_ty = ty - y_start
                    rgb_chs = []
                    for ch in display_chs[:3]:
                        if ch in ch_data:
                            rgb_chs.append(ch_data[ch][rel_ty:rel_ty+th, rel_tx:rel_tx+tw])
                        else:
                            rgb_chs.append(np.zeros((th, tw), dtype=np.uint16))
                    while len(rgb_chs) < 3:
                        rgb_chs.append(np.zeros((th, tw), dtype=np.uint16))
                    tile_vis_cache[(tx, ty)] = pct_norm(np.stack(rgb_chs, axis=-1))
                    print(f"  Loaded tile ({tx},{ty})")

                marker_colors = build_marker_colors(marker_map)
                html_path = run_dir / 'html' / 'islet_analysis.html'
                generate_html(islet_features, detections, islet_groups, tile_info,
                              tile_masks_cache, tile_vis_cache, pixel_size, marker_map,
                              marker_colors, czi_path.stem, tile_h, tile_w, html_path,
                              tissue_stats=tissue_stats)

    print("\nDone!")


if __name__ == '__main__':
    main()
