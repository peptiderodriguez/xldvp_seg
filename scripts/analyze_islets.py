#!/usr/bin/env python3
"""Spatial analysis of pancreatic islets from a completed islet detection run.

Finds islets by tissue-level signal: sums percentile-normalized marker channels
(Gcg, Ins, Sst), Otsu-thresholds the endocrine signal image, then takes all
Cellpose-detected cells inside each region.  Per-islet features:
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

try:
    import hdf5plugin  # noqa: F401 — registers LZ4 codec for h5py
except ImportError:
    pass

from segmentation.utils.json_utils import sanitize_for_json

logger = logging.getLogger(__name__)


def load_czi_direct(czi_path, channels, strip_height=4096):
    """Load CZI channels directly using aicspylibczi, bypassing CZILoader.

    CZILoader has issues with multi-scene CZI files (S dimension causes
    CDimCoordinatesOverspecifiedException). This function reads channels
    via read_mosaic() without passing S, which works correctly.

    Returns:
        (pixel_size, x_start, y_start, ch_data) where ch_data is {ch: np.ndarray}
    """
    from aicspylibczi import CziFile

    reader = CziFile(str(czi_path))
    bbox = reader.get_mosaic_scene_bounding_box(index=0)
    x_start, y_start, width, height = bbox.x, bbox.y, bbox.w, bbox.h

    # Pixel size from CZI metadata (no hardcoded fallback)
    pixel_size = None
    try:
        scaling = reader.meta.find('.//Scaling/Items/Distance[@Id="X"]/Value')
        if scaling is not None:
            pixel_size = float(scaling.text) * 1e6
    except Exception:
        pass

    ch_data = {}
    n_strips = (height + strip_height - 1) // strip_height
    for ch in channels:
        logger.info("  Loading channel %d...", ch)
        arr = np.empty((height, width), dtype=np.uint16)
        for i in range(n_strips):
            y_off = i * strip_height
            h = min(strip_height, height - y_off)
            strip = reader.read_mosaic(
                region=(x_start, y_start + y_off, width, h),
                scale_factor=1, C=ch
            )
            arr[y_off:y_off + h, :] = np.squeeze(strip)
        ch_data[ch] = arr
        logger.info("  Channel %d loaded: %.2f GB", ch, arr.nbytes / 1e9)

    return pixel_size, x_start, y_start, ch_data


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
# 1. Median-based marker classification
# ---------------------------------------------------------------------------

def compute_cell_medians(detections, tiles_dir, ch_data, marker_map, x_start, y_start):
    """Compute per-cell median intensity for each marker channel from HDF5 masks + CZI.

    Groups detections by tile_origin, loads the per-tile HDF5 mask, extracts
    the masked region from the full-slide CZI channel arrays, and stores
    ``ch{idx}_median`` in each detection's features dict.
    """
    from scipy import ndimage

    tile_groups = {}
    for i, det in enumerate(detections):
        to = det.get('tile_origin')
        if to is None:
            continue
        key = (int(to[0]), int(to[1]))
        tile_groups.setdefault(key, []).append(i)

    n_processed = 0
    for (tx, ty), det_indices in tile_groups.items():
        td = tiles_dir / f'tile_{tx}_{ty}'
        mask_path = td / 'islet_masks.h5'
        if not mask_path.exists():
            continue

        with h5py.File(mask_path, 'r') as f:
            masks = f['masks'][:]

        slices = ndimage.find_objects(masks)
        th, tw = masks.shape[:2]

        # Tile position relative to CZI mosaic origin
        rel_tx = tx - x_start
        rel_ty = ty - y_start

        for i in det_indices:
            det = detections[i]
            ml = det.get('tile_mask_label', det.get('mask_label'))
            if ml is None or ml <= 0 or ml > len(slices):
                continue
            sl = slices[ml - 1]  # 1-indexed
            if sl is None:
                continue

            feats = det.setdefault('features', {})
            y_sl, x_sl = sl
            cell_mask = masks[sl] == ml

            for name, ch_idx in marker_map.items():
                if ch_idx not in ch_data:
                    continue
                ch_region = ch_data[ch_idx][rel_ty + y_sl.start:rel_ty + y_sl.stop,
                                            rel_tx + x_sl.start:rel_tx + x_sl.stop]
                vals = ch_region[cell_mask]
                vals = vals[vals > 0]  # exclude CZI padding zeros
                feats[f'ch{ch_idx}_median'] = float(np.median(vals)) if len(vals) > 0 else 0.0

            n_processed += 1

        del masks

    logger.info("  Computed medians for %d/%d cells across %d tiles",
                n_processed, len(detections), len(tile_groups))


def classify_by_percentile(detections, marker_map, percentile=95):
    """Classify cells by marker channel median intensity using a percentile threshold.

    For each marker channel, the threshold is ``np.percentile(all_medians, percentile)``.
    A cell above threshold for exactly one marker gets that marker name.
    If two markers are within 1.5x of each other: ``'multi'``.
    Below all thresholds: ``'none'``.

    Returns:
        thresholds: dict {marker_name: float} for downstream use.
    """
    # Collect all non-zero medians per channel
    ch_vals = {}
    for name, ch_idx in marker_map.items():
        key = f'ch{ch_idx}_median'
        vals = [d.get('features', {}).get(key) for d in detections]
        ch_vals[name] = [v for v in vals if v is not None and v > 0]

    thresholds = {}
    for name, vals in ch_vals.items():
        thresholds[name] = float(np.percentile(vals, percentile)) if vals else float('inf')

    logger.info("  Percentile thresholds (p%s):", percentile)
    for name, t in thresholds.items():
        ch_idx = marker_map[name]
        n_above = sum(1 for v in ch_vals[name] if v >= t)
        logger.info("    %s (ch%d): %.1f  (%d cells above)", name, ch_idx, t, n_above)

    counts = Counter()
    for det in detections:
        feats = det.get('features', {})
        above = {}
        for name, ch_idx in marker_map.items():
            val = feats.get(f'ch{ch_idx}_median', 0)
            if val >= thresholds[name]:
                above[name] = val

        if not above:
            det['marker_class'] = 'none'
        elif len(above) == 1:
            det['marker_class'] = next(iter(above))
        else:
            sorted_markers = sorted(above.items(), key=lambda x: x[1], reverse=True)
            dominant_val = sorted_markers[0][1]
            runner_up_val = sorted_markers[1][1]
            if runner_up_val > 0 and dominant_val < 1.5 * runner_up_val:
                det['marker_class'] = 'multi'
            else:
                det['marker_class'] = sorted_markers[0][0]

        counts[det['marker_class']] += 1

    logger.info("  Classification: %s", dict(counts))
    return thresholds


# ---------------------------------------------------------------------------
# 2. Islet definition
# ---------------------------------------------------------------------------

def load_detections(run_dir):
    """Load detections JSON from run directory."""
    det_path = Path(run_dir) / 'islet_detections.json'
    if not det_path.exists():
        logger.error(f"Detections not found: {det_path}")
        sys.exit(1)
    with open(det_path) as f:
        dets = json.load(f)
    logger.info("Loaded %d detections from %s", len(dets), det_path)
    return dets


def find_islet_regions(ch_data, marker_map, pixel_size, downsample=4,
                       blur_sigma_um=10.0, close_um=10.0,
                       min_area_um2=500.0, buffer_um=25.0,
                       otsu_multiplier=1.5):
    """Find islet regions from tissue-level endocrine signal.

    Sums percentile-normalized marker channels to produce a total endocrine
    signal image, then Otsu-thresholds and extracts connected components.

    Args:
        ch_data: {ch_idx: np.ndarray} full-res channel arrays
        marker_map: {name: ch_idx} marker channel mapping
        pixel_size: um per pixel at full resolution
        downsample: downsampling factor (default 8)
        blur_sigma_um: Gaussian blur sigma in um (~1 cell diameter)
        close_um: morphological closing kernel in um
        min_area_um2: minimum region area in um^2
        buffer_um: dilation buffer in um (capture border cells)
        otsu_multiplier: multiply Otsu threshold by this (>1 = stricter)

    Returns:
        (region_labels, downsample) where region_labels is a labeled 2D array
        at downsampled resolution. Label 0 = background, 1..N = islet regions.
    """
    from scipy.ndimage import gaussian_filter, label as ndi_label
    from scipy.ndimage import binary_closing, binary_dilation
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    ds_pixel_size = pixel_size * downsample

    # Downsample each marker channel and percentile-normalize to [0, 1]
    normalized = []
    for name, ch_idx in marker_map.items():
        if ch_idx not in ch_data:
            continue
        full = ch_data[ch_idx]
        arr = full[::downsample, ::downsample].astype(np.float32)
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            normalized.append(np.zeros_like(arr))
            continue
        p1, p99 = np.percentile(nonzero, [1, 99])
        if p99 > p1:
            arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
        else:
            arr = np.zeros_like(arr)
        # Re-zero CZI padding pixels (downsample may alias, re-check from full)
        arr[full[::downsample, ::downsample] == 0] = 0
        normalized.append(arr)
        logger.info("    %s (ch%d): p1=%.0f p99=%.0f", name, ch_idx, p1, p99)

    if not normalized:
        logger.warning("  No marker channels available for tissue-level finding")
        return np.zeros((1, 1), dtype=np.int32), downsample, np.zeros((1, 1), dtype=np.float32)

    # Sum normalized channels → total endocrine signal
    signal = np.sum(normalized, axis=0)

    # Gaussian blur (sigma in downsampled pixels)
    sigma_px = blur_sigma_um / ds_pixel_size
    signal = gaussian_filter(signal, sigma=sigma_px)

    # Otsu on non-zero pixels
    nonzero_signal = signal[signal > 0]
    if len(nonzero_signal) < 100:
        logger.warning("  Too few non-zero pixels for Otsu thresholding")
        return np.zeros(signal.shape, dtype=np.int32), downsample, signal

    otsu_raw = threshold_otsu(nonzero_signal)
    otsu_t = otsu_raw * otsu_multiplier
    binary = signal >= otsu_t

    # Morphological close (fill inter-cell gaps)
    close_px = max(1, int(round(close_um / ds_pixel_size)))
    struct = np.ones((close_px * 2 + 1, close_px * 2 + 1), dtype=bool)
    binary = binary_closing(binary, structure=struct)

    # Remove small regions
    min_area_px = max(1, int(round(min_area_um2 / (ds_pixel_size ** 2))))
    binary = remove_small_objects(binary, min_size=min_area_px)

    # Dilate by buffer_um (capture border cells)
    buffer_px = max(1, int(round(buffer_um / ds_pixel_size)))
    buf_struct = np.ones((buffer_px * 2 + 1, buffer_px * 2 + 1), dtype=bool)
    binary = binary_dilation(binary, structure=buf_struct)

    # Label connected components
    region_labels, n_regions = ndi_label(binary)

    logger.info("  Tissue-level islet finding: %s downsampled image (%.2f um/px)",
                signal.shape, ds_pixel_size)
    logger.info("    Otsu: raw=%.3f x%s = %.3f, %d candidate regions",
                otsu_raw, otsu_multiplier, otsu_t, n_regions)
    logger.info("    blur=%.1fpx close=%dpx min_area=%dpx buffer=%dpx",
                sigma_px, close_px, min_area_px, buffer_px)

    return region_labels, downsample, signal


def assign_cells_to_regions(detections, region_labels, downsample,
                            x_start, y_start, min_cells=5):
    """Assign detections to labeled islet regions by spatial lookup.

    Each detection's global_center is mapped to the downsampled region label
    image. Regions with fewer than min_cells detections are dropped.

    Returns:
        islet_groups: {region_id: [det_indices]}
    """
    h, w = region_labels.shape
    groups = {}

    for i, det in enumerate(detections):
        gc = det.get('global_center', det.get('center'))
        if gc is None:
            det['islet_id'] = -1
            continue

        # Global mosaic coords → array coords → downsampled coords
        ax = round(gc[0]) - x_start
        ay = round(gc[1]) - y_start
        dx = ax // downsample
        dy = ay // downsample

        # Bounds check
        if 0 <= dx < w and 0 <= dy < h:
            rid = int(region_labels[dy, dx])
        else:
            rid = 0

        if rid > 0:
            det['islet_id'] = rid
            groups.setdefault(rid, []).append(i)
        else:
            det['islet_id'] = -1

    # Filter by min_cells
    small = [rid for rid, indices in groups.items() if len(indices) < min_cells]
    for rid in small:
        for i in groups[rid]:
            detections[i]['islet_id'] = -1
        del groups[rid]

    n_assigned = sum(len(v) for v in groups.values())
    logger.info("  Assigned %d/%d cells to %d islet regions "
                "(dropped %d regions with < %d cells)",
                n_assigned, len(detections), len(groups), len(small), min_cells)

    return groups


def save_region_diagnostic(endocrine_signal, region_labels, islet_features,
                           quality_dropped_ids, ds_pixel_size, output_path):
    """Save diagnostic PNG showing all candidate regions on endocrine signal.

    Green = kept by quality filter, red = dropped by quality filter,
    yellow = too few cells (never analyzed).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h, w = endocrine_signal.shape
    n_regions = region_labels.max()

    # Build kept set
    kept_ids = {f['islet_id'] for f in islet_features}

    # RGB canvas from endocrine signal (grayscale)
    sig_norm = endocrine_signal.copy()
    if sig_norm.max() > 0:
        sig_norm = sig_norm / sig_norm.max()
    rgb = np.stack([sig_norm, sig_norm, sig_norm], axis=-1)

    # Draw region boundaries by finding edges of each label, dilated for visibility
    from scipy.ndimage import binary_erosion, binary_dilation
    thickness = 5  # pixels
    thick_struct = np.ones((thickness * 2 + 1, thickness * 2 + 1), dtype=bool)
    for rid in range(1, n_regions + 1):
        mask = region_labels == rid
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        if not boundary.any():
            continue
        # Thicken boundary
        boundary = binary_dilation(boundary, structure=thick_struct)

        if rid in kept_ids:
            color = [0, 1, 0]       # green = kept
        elif rid in quality_dropped_ids:
            color = [1, 0, 0]       # red = dropped by quality filter
        else:
            color = [1, 1, 0]       # yellow = too few cells

        rgb[boundary] = color

    # Plot with scale bar
    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=150)
    ax.imshow(rgb, origin='upper')
    n_yellow = n_regions - len(kept_ids) - len(quality_dropped_ids)
    ax.set_title(f'Tissue-level islet candidates: '
                 f'{len(kept_ids)} kept (green), '
                 f'{len(quality_dropped_ids)} dropped (red), '
                 f'{n_yellow} <min_cells (yellow)',
                 fontsize=10)
    ax.axis('off')

    # Scale bar: 500 um
    bar_px = 500.0 / ds_pixel_size
    ax.plot([w - bar_px - 20, w - 20], [h - 30, h - 30], 'w-', linewidth=3)
    ax.text(w - bar_px / 2 - 20, h - 40, '500 um', color='white',
            ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved region diagnostic: %s", output_path)


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


def compute_coexpression(cells, marker_map, thresholds):
    """Quantify co-expression patterns among 'multi' cells.

    thresholds: dict {marker_name: threshold_value} from classify_by_percentile().
    For each multi cell, checks which marker channels exceed their threshold,
    then counts co-expression pairs (e.g. coexpr_gcg_ins).
    """
    if not thresholds:
        return {}
    n_multi_cells = 0
    pair_counts = Counter()
    for c in cells:
        if c.get('marker_class') != 'multi':
            continue
        n_multi_cells += 1
        feats = c.get('features', {})
        above = []
        for name, ch_idx in marker_map.items():
            val = feats.get(f'ch{ch_idx}_median', 0)
            if val >= thresholds.get(name, float('inf')):
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
                       thresholds=None):
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

        # Mean dominant marker intensity for marker+ cells (quality metric)
        marker_cells = [c for c in cells if c.get('marker_class') not in ('none', 'multi')]
        if marker_cells:
            dom_vals = []
            for c in marker_cells:
                mc = c['marker_class']
                ch_idx = marker_map.get(mc)
                if ch_idx is not None:
                    dom_vals.append(c.get('features', {}).get(f'ch{ch_idx}_median', 0))
            row['mean_marker_intensity'] = float(np.mean(dom_vals)) if dom_vals else 0.0
        else:
            row['mean_marker_intensity'] = 0.0
        spatial = compute_spatial_metrics(cells, pixel_size, marker_map)
        row.update(spatial)
        row.update(compute_cell_type_sizes(cells, pixel_size, marker_map))
        row.update(compute_coexpression(cells, marker_map, thresholds))
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
        logger.info("No islets to export")
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
    logger.info("Saved CSV: %s (%d islets)", output_path, len(rows))


# sanitize_for_json imported from segmentation.utils.json_utils


def export_detections(detections, output_path):
    """Export enriched detections JSON (with islet_id + marker_class)."""
    clean = sanitize_for_json(detections)
    with open(output_path, 'w') as f:
        json.dump(clean, f)
    logger.info("Saved enriched detections: %s", output_path)


# ---------------------------------------------------------------------------
# 8. HTML visualization
# ---------------------------------------------------------------------------

def pct_norm(img, pop_ranges=None):
    """Percentile-normalize a multi-channel image, preserving zero (padding) pixels.

    Args:
        img: (H, W, C) uint16 array
        pop_ranges: optional list of (lo, hi) per channel from population stats.
            If provided, uses these instead of per-tile percentiles for consistent
            display across tiles (matching classification thresholds).
    """
    out = np.zeros_like(img, dtype=np.uint8)
    valid = np.any(img > 0, axis=-1)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(float)
        if pop_ranges is not None and c < len(pop_ranges) and pop_ranges[c] is not None:
            lo, hi = pop_ranges[c]
        else:
            vals = ch[valid]
            if len(vals) == 0:
                continue
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99.5)
        if hi > lo:
            out[:, :, c] = np.clip(255 * (ch - lo) / (hi - lo), 0, 255).astype(np.uint8)
    return out


def draw_dashed_contours(img, contours, color=None, thickness=1, dash_len=6, gap_len=4):
    """Draw alternating black/white dashed contour lines on img in-place."""
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
                cv2.circle(img, tuple(pt), 0, (0, 0, 0), thickness)       # black dash
            else:
                cv2.circle(img, tuple(pt), 0, (255, 255, 255), thickness)  # white dash


def render_islet_card(islet_feat, cells, masks, tile_vis, tile_x, tile_y,
                      tile_h, tile_w, pixel_size, marker_colors, marker_map):
    """Render a single islet card with crop image + stats. Returns HTML string or None."""
    marker_names = list(marker_map.keys())

    # Collect cell info within this tile
    cell_info = []
    has_marker_cells = False
    for d in cells:
        gc = d.get('global_center', d.get('center', [0, 0]))
        cx_rel = gc[0] - tile_x
        cy_rel = gc[1] - tile_y
        # Use tile_mask_label for HDF5 lookup (new data), fall back to mask_label (old data)
        ml = d.get('tile_mask_label', d.get('mask_label'))
        mc = d.get('marker_class', 'none')
        cell_info.append((cx_rel, cy_rel, ml, mc))
        if mc != 'none' and ml is not None and ml > 0:
            has_marker_cells = True

    if not has_marker_cells:
        return None

    # Union mask of marker+ cells only for islet boundary
    mh, mw = masks.shape[:2]
    union_mask = np.zeros((mh, mw), dtype=np.uint8)
    for _cx, _cy, ml, mc in cell_info:
        if mc == 'none' or ml is None or ml <= 0:
            continue
        union_mask |= (masks == ml).astype(np.uint8)

    # Morphological close + dilate → single cohesive boundary
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    boundary = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, close_k)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    boundary = cv2.dilate(boundary, dilate_k)

    # Keep only the largest connected component
    n_labels, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(boundary)
    if n_labels > 1:
        # Label 0 is background; find largest foreground component
        largest = 1 + np.argmax(cc_stats[1:, cv2.CC_STAT_AREA])
        boundary = (cc_labels == largest).astype(np.uint8)

    ys, xs = np.where(boundary > 0)
    if len(xs) == 0:
        return None

    x_min = max(0, int(xs.min()) - PADDING)
    x_max = min(mw, int(xs.max()) + PADDING)
    y_min = max(0, int(ys.min()) - PADDING)
    y_max = min(mh, int(ys.max()) + PADDING)

    crop = tile_vis[y_min:y_max, x_min:x_max].copy()

    # Draw dashed contours for marker+ cells only
    for _cx, _cy, ml, mc in cell_info:
        if mc == 'none' or ml is None or ml <= 0:
            continue
        mask_crop = (masks[y_min:y_max, x_min:x_max] == ml).astype(np.uint8)
        if not mask_crop.any():
            continue
        cnts, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = marker_colors.get(mc, (128, 128, 128))
        draw_dashed_contours(crop, cnts, color, thickness=2)

    # Solid pink islet boundary (single cohesive shape)
    boundary_crop = boundary[y_min:y_max, x_min:x_max]
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

        # Filter to cells whose mask_label belongs to THIS tile (tile_origin match)
        # mask_labels are per-tile (1..N), so a cell's mask_label is only valid
        # in the masks.h5 from its origin tile, not any overlapping tile
        all_cells = [detections[i] for i in islet_groups[iid]]
        cells = [c for c in all_cells
                 if tuple(c.get('tile_origin', [-1, -1])) == (tx, ty)]
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
    logger.info("Saved HTML: %s (%.0f KB)", output_path, len(html) / 1024)


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
                        help='Path to CZI file (for loading marker + display channels)')
    parser.add_argument('--buffer-um', type=float, default=25.0,
                        help='Dilation buffer in um applied to tissue-level islet '
                             'regions to capture border cells (default: 25)')
    parser.add_argument('--blur-sigma-um', type=float, default=10.0,
                        help='Gaussian blur sigma in um for tissue signal smoothing '
                             '(default: 10, ~1 cell diameter)')
    parser.add_argument('--close-um', type=float, default=10.0,
                        help='Morphological closing kernel in um to fill inter-cell '
                             'gaps within islets (default: 10)')
    parser.add_argument('--min-area-um2', type=float, default=500.0,
                        help='Minimum islet region area in um^2 (default: 500)')
    parser.add_argument('--otsu-multiplier', type=float, default=1.5,
                        help='Multiply Otsu threshold by this factor (>1 = stricter, '
                             'default: 1.5)')
    parser.add_argument('--min-cells', type=int, default=5,
                        help='Minimum cells per islet region (default: 5)')
    parser.add_argument('--display-channels', type=str, default='2,3,5',
                        help='R,G,B channel indices for display (default: 2,3,5)')
    parser.add_argument('--marker-channels', type=str, default='gcg:2,ins:3,sst:5',
                        help='Marker-to-channel mapping (default: gcg:2,ins:3,sst:5)')
    parser.add_argument('--marker-percentile', type=float, default=95,
                        help='Percentile threshold for marker classification: cells with '
                             'median intensity above this percentile are marker-positive '
                             '(default: 95 = top 5%%)')
    parser.add_argument('--quality-filter', choices=['none', 'q25'], default='q25',
                        help='Filter regions by marker-positive cell fraction. '
                             'q25=drop bottom quartile (default: q25)')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in um/px. If omitted, read from CZI metadata. '
                             'If CZI metadata is missing, an error is raised.')
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
        logger.warning("No detections — exiting")
        sys.exit(0)

    # 2. Load CZI channels (marker + display, deduplicated)
    all_chs = sorted(set(display_chs) | set(marker_map.values()))
    logger.info("Loading CZI channels %s from %s...", all_chs, czi_path)
    czi_pixel_size, x_start, y_start, ch_data = load_czi_direct(czi_path, all_chs)
    if args.pixel_size is not None:
        pixel_size = args.pixel_size
        if czi_pixel_size is not None:
            logger.info("  Overriding CZI pixel_size=%.4f with --pixel-size=%.4f",
                         czi_pixel_size, pixel_size)
    elif czi_pixel_size is not None:
        pixel_size = czi_pixel_size
    else:
        logger.error("pixel_size must be provided via --pixel-size or read from "
                      "--czi-path metadata (CZI metadata did not contain pixel size)")
        sys.exit(1)
    logger.info("  pixel_size=%.4f um/px", pixel_size)

    # 3. Find islet regions from tissue-level signal
    logger.info("Finding islet regions from tissue signal...")
    region_labels, ds, endocrine_signal = find_islet_regions(
        ch_data, marker_map, pixel_size, buffer_um=args.buffer_um,
        blur_sigma_um=args.blur_sigma_um, close_um=args.close_um,
        min_area_um2=args.min_area_um2, otsu_multiplier=args.otsu_multiplier)

    # 4. Assign cells to regions
    islet_groups = assign_cells_to_regions(
        detections, region_labels, ds, x_start, y_start,
        min_cells=args.min_cells)

    if not islet_groups:
        logger.warning("No islet regions found — exiting")
        sys.exit(0)

    logger.info("%d islet regions with cells", len(islet_groups))

    # 5. Compute per-cell median intensity from HDF5 masks + CZI
    tiles_dir = run_dir / 'tiles'
    logger.info("Computing per-cell median intensities...")
    compute_cell_medians(detections, tiles_dir, ch_data, marker_map, x_start, y_start)

    # 6. Classify by percentile threshold (all cells, whole-population)
    logger.info("Classifying markers (p%s)...", args.marker_percentile)
    thresholds = classify_by_percentile(detections, marker_map,
                                        percentile=args.marker_percentile)

    # 7. Clear non-islet marker classifications (background noise)
    islet_cell_indices = set()
    for indices in islet_groups.values():
        islet_cell_indices.update(indices)

    n_cleared = 0
    for i, det in enumerate(detections):
        if i not in islet_cell_indices and det.get('marker_class', 'none') != 'none':
            det['marker_class'] = 'none'
            n_cleared += 1
    if n_cleared:
        logger.info("  Cleared %d non-islet marker classifications", n_cleared)
    final_counts = Counter(d.get('marker_class', 'none') for d in detections)
    logger.info("  Final classification: %s", dict(final_counts))

    # 8. Analyze all islets
    logger.info("Computing islet features...")
    islet_features = analyze_all_islets(detections, islet_groups, pixel_size, marker_map,
                                        thresholds=thresholds)
    logger.info("  %d islets analyzed", len(islet_features))

    # 9. Quality filter: keep islets with above-median tissue endocrine signal
    #    Per-region median of the summed+normalized endocrine image (same as diagnostic).
    #    Median is robust to bright-pixel outliers within a region.
    quality_dropped_ids = set()
    if args.quality_filter != 'none' and len(islet_features) >= 4:
        n_before = len(islet_features)

        # Compute median endocrine signal per region from the tissue image
        for f in islet_features:
            rid = f['islet_id']
            region_mask = region_labels == rid
            region_vals = endocrine_signal[region_mask]
            f['_median_signal'] = float(np.median(region_vals)) if len(region_vals) > 0 else 0.0

        signals = [f['_median_signal'] for f in islet_features]
        cutoff = float(np.median(signals))
        keep = [f for f in islet_features if f['_median_signal'] >= cutoff]
        drop = [f for f in islet_features if f['_median_signal'] < cutoff]

        if drop:
            quality_dropped_ids = {f['islet_id'] for f in drop}
            for iid in quality_dropped_ids:
                islet_groups.pop(iid, None)
            for d in detections:
                if d.get('islet_id', -1) in quality_dropped_ids:
                    d['islet_id'] = -1
            islet_features = keep
            drop_info = ', '.join(
                f"{f['islet_id']}({f['_median_signal']:.3f})"
                for f in drop)
            logger.info("  Quality filter (median tissue signal, cutoff=%.3f): "
                        "kept %d/%d, dropped %d (id/signal: %s)",
                        cutoff, len(keep), n_before, len(drop), drop_info)
        else:
            logger.info("  Quality filter: all %d islets above median=%.3f",
                        n_before, cutoff)

        # Clean up temp key
        for f in islet_features:
            f.pop('_median_signal', None)

    # 9b. Save region diagnostic image
    ds_pixel_size = pixel_size * ds
    diag_path = run_dir / 'islet_regions_diagnostic.png'
    save_region_diagnostic(endocrine_signal, region_labels, islet_features,
                           quality_dropped_ids, ds_pixel_size, diag_path)

    # 10. Tissue-level analysis
    logger.info("Computing tissue-level metrics...")
    tissue_stats = compute_tissue_level(islet_features, detections, pixel_size)

    # Print summary
    if islet_features:
        areas = [f['area_um2'] for f in islet_features]
        cells_per = [f['n_cells'] for f in islet_features]
        n_flagged = sum(1 for f in islet_features if f.get('n_flags', 0) > 0)
        logger.info("--- Summary ---")
        logger.info("  Islets: %d", len(islet_features))
        logger.info("  Cells: %d total, median %.0f/islet",
                     sum(cells_per), np.median(cells_per))
        logger.info("  Area: median %.0f um2, range %.0f-%.0f",
                     np.median(areas), min(areas), max(areas))
        logger.info("  Flagged atypical: %d", n_flagged)
        for name in marker_names:
            fracs = [f.get(f'frac_{name}', 0) for f in islet_features]
            logger.info("  %s: median frac %.2f", name, np.median(fracs))
        mcis = [f.get('mantle_core_index') for f in islet_features if f.get('mantle_core_index') is not None]
        if mcis:
            logger.info("  Mantle-core index: median %.3f", np.median(mcis))
        mixes = [f.get('mixing_index') for f in islet_features if f.get('mixing_index') is not None]
        if mixes:
            logger.info("  Mixing index: median %.3f", np.median(mixes))
        # Tissue-level
        eaf = tissue_stats.get('endocrine_area_fraction')
        if eaf is not None:
            logger.info("  Endocrine area fraction: %.2f%%", eaf * 100)
        idmm = tissue_stats.get('islet_density_per_mm2')
        if idmm is not None:
            logger.info("  Islet density: %.1f /mm2", idmm)
        mii = tissue_stats.get('median_inter_islet_um')
        if mii is not None:
            logger.info("  Inter-islet distance: median %.0f um", mii)

    # 11. Export CSV
    csv_path = run_dir / 'islet_summary.csv'
    export_csv(islet_features, csv_path)

    # 12. Export enriched detections
    det_out = run_dir / 'islet_detections_analyzed.json'
    export_detections(detections, det_out)

    # 13. HTML visualization
    if not args.no_html:
        logger.info("Generating HTML visualization...")

        if not tiles_dir.exists():
            logger.warning("Tiles directory not found: %s — skipping HTML", tiles_dir)
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
                logger.warning("No tile directories found — skipping HTML")
            else:
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

                # Compute normalization ranges from islet region pixels only
                # so marker signal is visible (whole-slide p99.5 is too wide)
                islet_mask = region_labels > 0
                pop_ranges = []
                for ch in display_chs[:3]:
                    if ch in ch_data:
                        ds_arr = ch_data[ch][::ds, ::ds]
                        h = min(ds_arr.shape[0], islet_mask.shape[0])
                        w = min(ds_arr.shape[1], islet_mask.shape[1])
                        vals = ds_arr[:h, :w][islet_mask[:h, :w]]
                        vals = vals[vals > 0]
                        if len(vals) > 0:
                            lo = float(np.percentile(vals, 1))
                            hi = float(np.percentile(vals, 95))
                            pop_ranges.append((lo, hi))
                        else:
                            pop_ranges.append(None)
                    else:
                        pop_ranges.append(None)
                logger.info("  Islet-region normalization ranges: %s", pop_ranges)

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
                    tile_vis_cache[(tx, ty)] = pct_norm(
                        np.stack(rgb_chs, axis=-1), pop_ranges=pop_ranges)
                    logger.info("  Loaded tile (%d,%d)", tx, ty)

                marker_colors = build_marker_colors(marker_map)
                html_path = run_dir / 'html' / 'islet_analysis.html'
                generate_html(islet_features, detections, islet_groups, tile_info,
                              tile_masks_cache, tile_vis_cache, pixel_size, marker_map,
                              marker_colors, czi_path.stem, tile_h, tile_w, html_path,
                              tissue_stats=tissue_stats)

    logger.info("Done!")


if __name__ == '__main__':
    main()
