#!/usr/bin/env python
"""Interactive HTML viewer for spatial structure analysis with KDE density,
SAM2 structural regions, and cell dots.

Three visualization layers (all toggleable):
1. SAM2 structural regions — precomputed boundary polygons from density maps
2. KDE density contours — iso-density lines per cell type (JS, interactive bandwidth)
3. Cell dots — individual neurons colored by marker class

Two modes:
- Base mode (no GPU): KDE + dots, all computed in JS
- SAM2 mode (--sam2-regions, GPU): adds precomputed region boundaries

Usage:
    # Base mode (fast, no GPU)
    python scripts/generate_spatial_structure_viewer.py \\
        --detections brain_fish_output/*/detections_neurons_hybrid.json \\
        --group-field marker_class --top-n 12 \\
        --title "Brain FISH Spatial Structure" \\
        --output spatial_structure_viewer.html

    # With regions (requires GPU)
    python scripts/generate_spatial_structure_viewer.py \\
        --detections brain_fish_output/*/detections_neurons_hybrid.json \\
        --group-field marker_class --top-n 12 --sam2-regions \\
        --title "Brain FISH Spatial Structure (SAM2)" \\
        --output spatial_structure_sam2_viewer.html
"""
import argparse
import json
import sys
import html as html_mod
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
AUTO_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990',
    '#e6beff', '#9a6324', '#ffe119', '#aaffc3', '#800000',
    '#ffd8b1', '#000075', '#a9a9a9', '#808000', '#ff69b4',
]
OTHER_COLOR = '#555555'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_group(det, group_field):
    """Return group label for one detection, or None to skip."""
    val = det.get(group_field)
    if val is None:
        val = det.get('features', {}).get(group_field)
    return str(val) if val is not None else None


def load_file_data(path, group_field, weight_field=None):
    """Load detection JSON and group cells by field.

    Returns dict with groups, n_cells, x_range, y_range, or None.
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        detections = json.load(f)

    group_cells = {}  # label -> list of (x, y, weight)

    for det in detections:
        group = extract_group(det, group_field)
        if group is None:
            continue

        pos = det.get('global_center_um')
        if pos is None or len(pos) != 2:
            continue
        if not (np.isfinite(pos[0]) and np.isfinite(pos[1])):
            continue

        weight = 1.0
        if weight_field:
            w = det.get(weight_field)
            if w is None:
                w = det.get('features', {}).get(weight_field)
            if isinstance(w, (int, float)) and not np.isnan(w):
                weight = float(w)

        group_cells.setdefault(group, []).append((pos[0], pos[1], weight))

    if not group_cells:
        return None

    groups_out = []
    for label, cells in group_cells.items():
        arr = np.array(cells, dtype=np.float32)
        groups_out.append({
            'label': label,
            'color': None,
            'n': len(cells),
            'x': arr[:, 0].tolist(),
            'y': arr[:, 1].tolist(),
            'w': arr[:, 2].tolist(),
        })

    if not groups_out:
        return None

    all_x, all_y = [], []
    for g in groups_out:
        all_x.extend(g['x'])
        all_y.extend(g['y'])

    return {
        'groups': groups_out,
        'n_cells': sum(g['n'] for g in groups_out),
        'x_range': [float(min(all_x)), float(max(all_x))],
        'y_range': [float(min(all_y)), float(max(all_y))],
    }


def apply_colors(scenes_data, top_n, exclude_groups, include_groups=None):
    """Assign colors to top-N groups globally, lump rest into 'other'.

    Args:
        scenes_data: List of (label, data_dict) tuples. Modified in place.
        top_n: Keep top N groups by cell count.
        exclude_groups: Set of labels to drop entirely.
        include_groups: If set, only keep these groups (no 'other' lumping).

    Returns:
        color_map dict.
    """
    # Count globally
    global_counts = {}
    for _, data in scenes_data:
        for g in data['groups']:
            if g['label'] in exclude_groups:
                continue
            if include_groups and g['label'] not in include_groups:
                continue
            global_counts[g['label']] = global_counts.get(g['label'], 0) + g['n']

    sorted_groups = sorted(global_counts.items(), key=lambda x: -x[1])

    # When include_groups is set, keep ALL included groups (no "other" lumping)
    effective_top_n = len(sorted_groups) if include_groups else top_n
    top_labels = set()
    for i, (lbl, _) in enumerate(sorted_groups):
        if i < effective_top_n:
            top_labels.add(lbl)

    color_map = {}
    for i, (lbl, _) in enumerate(sorted_groups):
        if lbl in top_labels:
            color_map[lbl] = AUTO_COLORS[i % len(AUTO_COLORS)]
    if not include_groups:
        color_map['other'] = OTHER_COLOR

    for _, data in scenes_data:
        new_groups = {}
        for g in data['groups']:
            if g['label'] in exclude_groups:
                continue
            if include_groups and g['label'] not in include_groups:
                continue

            target = g['label'] if g['label'] in top_labels else 'other'
            if target not in new_groups:
                new_groups[target] = {
                    'label': target, 'color': color_map[target],
                    'n': 0, 'x': [], 'y': [], 'w': [],
                }
            ng = new_groups[target]
            ng['n'] += g['n']
            ng['x'].extend(g['x'])
            ng['y'].extend(g['y'])
            ng['w'].extend(g['w'])

        ordered = []
        for lbl, _ in sorted_groups:
            if lbl in new_groups and lbl in top_labels:
                ordered.append(new_groups[lbl])
        if 'other' in new_groups:
            ordered.append(new_groups['other'])

        data['groups'] = ordered
        data['n_cells'] = sum(g['n'] for g in ordered)

    return color_map


# ---------------------------------------------------------------------------
# SAM2 region computation (optional, requires GPU)
# ---------------------------------------------------------------------------

def compute_kde_image(positions, types, colors, resolution=1024, bandwidth_um=600):
    """Render multi-channel KDE density as RGB image for SAM2.

    Each cell type's density map is rendered as a separate color channel,
    then composited into an RGB image using the type's assigned color.

    Args:
        positions: (N, 2) array of cell positions in um.
        types: (N,) array of type indices.
        colors: List of (R, G, B) tuples per type (0-255).
        resolution: Output image resolution (pixels on long axis).
        bandwidth_um: KDE bandwidth in micrometers.

    Returns:
        (rgb_image, x_range, y_range, pixel_size_um) tuple.
    """
    from scipy.ndimage import gaussian_filter

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    data_w = x_max - x_min
    data_h = y_max - y_min
    if data_w <= 0 or data_h <= 0:
        return None, None, None, None

    # Compute pixel size and grid dimensions
    pixel_size = max(data_w, data_h) / resolution
    nx = max(1, int(np.ceil(data_w / pixel_size)))
    ny = max(1, int(np.ceil(data_h / pixel_size)))
    sigma = bandwidth_um / pixel_size

    # Accumulate density per type
    unique_types = np.unique(types)
    rgb = np.zeros((ny, nx, 3), dtype=np.float64)

    for ti in unique_types:
        mask = types == ti
        if not np.any(mask):
            continue
        px = positions[mask, 0]
        py = positions[mask, 1]

        # Bin into 2D histogram
        xi = np.clip(((px - x_min) / pixel_size).astype(int), 0, nx - 1)
        yi = np.clip(((py - y_min) / pixel_size).astype(int), 0, ny - 1)
        grid = np.zeros((ny, nx), dtype=np.float64)
        np.add.at(grid, (yi, xi), 1)

        # Gaussian blur
        if sigma > 0.5:
            grid = gaussian_filter(grid, sigma=sigma)

        # Normalize to [0, 1] (per-type)
        gmax = grid.max()
        if gmax > 0:
            grid /= gmax

        # Add to RGB using type color
        c = colors[int(ti)]
        rgb[:, :, 0] += grid * (c[0] / 255.0)
        rgb[:, :, 1] += grid * (c[1] / 255.0)
        rgb[:, :, 2] += grid * (c[2] / 255.0)

    # Normalize and convert to uint8
    rgb_max = rgb.max()
    if rgb_max > 0:
        rgb /= rgb_max
    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)

    return rgb_uint8, (x_min, x_max), (y_min, y_max), pixel_size


def run_sam2_regions(kde_image, positions, types, type_labels, type_colors,
                     x_range, y_range, pixel_size, min_region_cells=50):
    """Run SAM2 automatic mask generation on KDE density image.

    Args:
        kde_image: (H, W, 3) uint8 RGB image from compute_kde_image.
        positions: (N, 2) array of cell positions.
        types: (N,) array of type indices.
        type_labels: List of type label strings.
        type_colors: List of hex color strings per type.
        x_range: (x_min, x_max) tuple.
        y_range: Not used directly (x_range and pixel_size define mapping).
        pixel_size: Micrometers per pixel.
        min_region_cells: Minimum cells inside a region to keep it.

    Returns:
        List of region dicts with boundary, composition, label, etc.
    """
    import cv2

    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print('ERROR: SAM2 not available. Install sam2 package for --sam2-regions.',
              file=sys.stderr)
        return []

    import torch

    # Build SAM2 model
    sam2_checkpoint = None
    sam2_config = None
    # Search for SAM2 checkpoint
    for candidate in [
        Path.home() / 'sam2_hiera_large.pt',
        Path.home() / 'models' / 'sam2_hiera_large.pt',
        Path('/tmp/sam2_hiera_large.pt'),
    ]:
        if candidate.exists():
            sam2_checkpoint = str(candidate)
            sam2_config = 'sam2_hiera_l'
            break

    if sam2_checkpoint is None:
        # Try small model
        for candidate in [
            Path.home() / 'sam2_hiera_small.pt',
            Path.home() / 'models' / 'sam2_hiera_small.pt',
        ]:
            if candidate.exists():
                sam2_checkpoint = str(candidate)
                sam2_config = 'sam2_hiera_s'
                break

    if sam2_checkpoint is None:
        print('ERROR: No SAM2 checkpoint found. Place sam2_hiera_large.pt in ~/models/',
              file=sys.stderr)
        return []

    print(f'Loading SAM2 model: {sam2_checkpoint}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=100,
    )

    print(f'Running SAM2 on KDE image ({kde_image.shape[1]}x{kde_image.shape[0]})...')
    masks = mask_generator.generate(kde_image)
    print(f'SAM2 found {len(masks)} masks')

    # Sort by area descending
    masks.sort(key=lambda m: m['area'], reverse=True)

    # Process each mask: compute composition, extract boundary
    x_min = x_range[0]
    y_min = y_range[0]
    regions = []

    # Pre-compute pixel coordinates once for all masks (C1+M5 fix)
    # Use first mask to get dimensions (SAM2 masks all share the same shape)
    if not masks:
        return []
    h0, w0 = masks[0]['segmentation'].shape
    cx_all = np.clip(((positions[:, 0] - x_min) / pixel_size).astype(int), 0, w0 - 1)
    cy_all = np.clip(((positions[:, 1] - y_min) / pixel_size).astype(int), 0, h0 - 1)

    for mask_data in masks:
        seg = mask_data['segmentation']  # bool array (H, W)
        h, w = seg.shape

        # Clip to this mask's dimensions (should match h0,w0 but be safe)
        cx = np.minimum(cx_all, w - 1)
        cy = np.minimum(cy_all, h - 1)

        inside_full = seg[cy, cx].astype(bool)

        n_inside = inside_full.sum()
        if n_inside < min_region_cells:
            continue

        # Composition: count each type inside
        composition = {}
        for ti in np.unique(types):
            count = (inside_full & (types == ti)).sum()
            if count > 0:
                composition[type_labels[int(ti)]] = int(count)

        # Dominant type
        dominant = max(composition, key=composition.get)
        dominant_frac = composition[dominant] / n_inside

        # Secondary type
        sorted_comp = sorted(composition.items(), key=lambda x: -x[1])
        if len(sorted_comp) > 1 and sorted_comp[1][1] / n_inside > 0.2:
            label = f'{dominant}/{sorted_comp[1][0]}'
        else:
            label = dominant

        # Extract boundary polygon using OpenCV
        seg_uint8 = seg.astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)

        # Simplify with RDP
        epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
        contour = cv2.approxPolyDP(contour, epsilon, True)

        # Convert pixel coordinates to um coordinates
        boundary = []
        for pt in contour.reshape(-1, 2):
            boundary.append({
                'x': round(float(pt[0] * pixel_size + x_min), 1),
                'y': round(float(pt[1] * pixel_size + y_min), 1),
            })

        if len(boundary) < 3:
            continue

        # Compute elongation from mask moments
        moments = cv2.moments(seg_uint8)
        if moments['m00'] > 0:
            mu20 = moments['mu20'] / moments['m00']
            mu02 = moments['mu02'] / moments['m00']
            mu11 = moments['mu11'] / moments['m00']
            d = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
            major = mu20 + mu02 + d
            minor = mu20 + mu02 - d
            elongation = round(np.sqrt(max(major, 1e-9) / max(minor, 1e-9)), 2)
        else:
            elongation = 1.0

        # Assign color from dominant type
        dominant_idx = type_labels.index(dominant) if dominant in type_labels else 0
        color = type_colors[dominant_idx] if dominant_idx < len(type_colors) else '#888888'

        area_um2 = round(mask_data['area'] * pixel_size * pixel_size, 0)

        regions.append({
            'id': len(regions),
            'type': dominant,
            'label': label,
            'color': color,
            'composition': {k: round(v / n_inside, 3) for k, v in composition.items()},
            'n_cells': int(n_inside),
            'area_um2': area_um2,
            'elongation': elongation,
            'dominant_frac': round(dominant_frac, 3),
            'boundary': boundary,
        })

    print(f'Kept {len(regions)} regions (>={min_region_cells} cells each)')
    return regions


# ---------------------------------------------------------------------------
# CV region computation (per-type Canny + contours, no GPU needed)
# ---------------------------------------------------------------------------

def compute_cv_regions(positions, types, type_labels, type_colors,
                       bandwidth_um=600, resolution=1024,
                       min_region_cells=50):
    """Compute structural regions per cell type using KDE + Otsu thresholding.

    For each cell type:
    1. Bin positions into 2D histogram
    2. Gaussian blur with bandwidth_um
    3. Threshold at multiple levels (Otsu + fixed fractions)
    4. Canny edge detection on density map
    5. findContours on thresholded regions (filled areas, not just edges)
    6. RDP simplify contour polygons
    7. Count cells inside each contour, label by type

    Args:
        positions: (N, 2) array of cell positions in um.
        types: (N,) array of type indices.
        type_labels: List of type label strings.
        type_colors: List of hex color strings per type.
        bandwidth_um: KDE bandwidth in micrometers.
        resolution: Grid resolution (pixels on long axis).
        min_region_cells: Minimum cells inside a region to keep.
        canny_low: Canny low threshold.
        canny_high: Canny high threshold.

    Returns:
        List of region dicts (same format as regions).
    """
    import cv2
    from scipy.ndimage import gaussian_filter

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    data_w = x_max - x_min
    data_h = y_max - y_min
    if data_w <= 0 or data_h <= 0:
        return []

    pixel_size = max(data_w, data_h) / resolution
    nx = max(1, int(np.ceil(data_w / pixel_size)))
    ny = max(1, int(np.ceil(data_h / pixel_size)))
    sigma = bandwidth_um / pixel_size

    # Pre-compute pixel coordinates for cell-in-region counting
    cx_all = np.clip(((positions[:, 0] - x_min) / pixel_size).astype(int), 0, nx - 1)
    cy_all = np.clip(((positions[:, 1] - y_min) / pixel_size).astype(int), 0, ny - 1)

    unique_types = np.unique(types)
    regions = []

    for ti in unique_types:
        type_mask = types == ti
        n_type = type_mask.sum()
        if n_type < min_region_cells:
            continue

        label = type_labels[int(ti)]
        color = type_colors[int(ti)] if int(ti) < len(type_colors) else '#888888'

        # Bin into 2D histogram
        px = positions[type_mask, 0]
        py = positions[type_mask, 1]
        xi = np.clip(((px - x_min) / pixel_size).astype(int), 0, nx - 1)
        yi = np.clip(((py - y_min) / pixel_size).astype(int), 0, ny - 1)
        grid = np.zeros((ny, nx), dtype=np.float64)
        np.add.at(grid, (yi, xi), 1)

        # Gaussian blur
        if sigma > 0.5:
            grid = gaussian_filter(grid, sigma=sigma)

        # Normalize to uint8 for OpenCV
        gmax = grid.max()
        if gmax <= 0:
            continue
        grid_u8 = (grid / gmax * 255).clip(0, 255).astype(np.uint8)

        # Multi-level thresholding: Otsu + fixed fractions of max
        # This finds both dense cores and broad territory
        otsu_thresh, _ = cv2.threshold(grid_u8, 0, 255, cv2.THRESH_OTSU)
        thresholds = sorted(set([
            max(10, int(otsu_thresh * 0.5)),   # broad territory
            max(15, int(otsu_thresh)),          # Otsu level
            max(20, int(otsu_thresh * 1.5)),    # dense cores
        ]))

        for thresh_val in thresholds:
            # Threshold to binary
            _, binary = cv2.threshold(grid_u8, thresh_val, 255, cv2.THRESH_BINARY)

            # Morphological close to fill small gaps
            kernel_size = max(3, int(sigma * 0.5)) | 1  # ensure odd
            morph_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)

            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Skip tiny contours
                area_px = cv2.contourArea(contour)
                if area_px < 50:
                    continue

                # RDP simplification
                epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
                contour = cv2.approxPolyDP(contour, epsilon, True)

                if len(contour) < 3:
                    continue

                # Count cells of THIS type inside contour
                # Create mask from contour
                cmask = np.zeros((ny, nx), dtype=np.uint8)
                cv2.drawContours(cmask, [contour], 0, 255, -1)

                # Count this type's cells inside
                inside_this = cmask[cy_all[type_mask], cx_all[type_mask]] > 0
                n_inside_type = int(inside_this.sum())
                if n_inside_type < min_region_cells:
                    continue

                # Count ALL cell types inside for composition
                inside_all = cmask[cy_all, cx_all] > 0
                n_inside_total = int(inside_all.sum())
                if n_inside_total == 0:
                    continue

                composition = {}
                for tj in unique_types:
                    count = int((inside_all & (types == tj)).sum())
                    if count > 0:
                        composition[type_labels[int(tj)]] = count

                dominant = max(composition, key=composition.get)
                dominant_frac = composition[dominant] / n_inside_total

                # Convert contour pixels to um coordinates
                boundary = []
                for pt in contour.reshape(-1, 2):
                    boundary.append({
                        'x': round(float(pt[0] * pixel_size + x_min), 1),
                        'y': round(float(pt[1] * pixel_size + y_min), 1),
                    })

                # Elongation from moments
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    mu20 = moments['mu20'] / moments['m00']
                    mu02 = moments['mu02'] / moments['m00']
                    mu11 = moments['mu11'] / moments['m00']
                    d = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
                    major = mu20 + mu02 + d
                    minor = mu20 + mu02 - d
                    elongation = round(np.sqrt(max(major, 1e-9) / max(minor, 1e-9)), 2)
                else:
                    elongation = 1.0

                area_um2 = round(area_px * pixel_size * pixel_size, 0)

                # Determine threshold level label
                if thresh_val <= thresholds[0]:
                    level_tag = 'broad'
                elif thresh_val >= thresholds[-1]:
                    level_tag = 'core'
                else:
                    level_tag = 'mid'

                regions.append({
                    'id': len(regions),
                    'type': label,
                    'label': f'{label} ({level_tag})',
                    'color': color,
                    'composition': {k: round(v / n_inside_total, 3)
                                    for k, v in composition.items()},
                    'n_cells': n_inside_total,
                    'area_um2': area_um2,
                    'elongation': elongation,
                    'dominant_frac': round(dominant_frac, 3),
                    'boundary': boundary,
                })

    # Sort by area descending
    regions.sort(key=lambda r: r['area_um2'], reverse=True)
    # Re-index
    for i, r in enumerate(regions):
        r['id'] = i

    print(f'CV regions: {len(regions)} regions from {len(unique_types)} types '
          f'(>={min_region_cells} cells each)')
    return regions


# ---------------------------------------------------------------------------
# Spatial autocorrelation regions (enrichment-based)
# ---------------------------------------------------------------------------

def compute_autocorrelation_regions(positions, types, type_labels, type_colors,
                                    bandwidth_um=600, resolution=512,
                                    min_region_cells=30):
    """Compute spatial autocorrelation regions via local enrichment analysis.

    Unlike CV regions (which threshold on absolute density — where a type is
    dense), this finds regions where each type is significantly ENRICHED
    relative to the local cell density.  A type may be dense in an area simply
    because ALL types are dense there; enrichment normalises that away and
    reveals true spatial structure.

    Method per cell type:
    1. KDE for this type  +  KDE for all types combined
    2. Local proportion surface  =  type_kde / total_kde
    3. Enrichment  =  local_proportion / global_proportion
    4. Contour at enrichment levels  1.5×, 2.5×, 4.0×
    5. Morphological close  →  cv2.findContours  →  RDP simplify
    6. Count cells inside each contour for composition stats

    Returns list of region dicts (same schema as compute_cv_regions).
    """
    import cv2
    from scipy.ndimage import gaussian_filter

    x_all = positions[:, 0].astype(np.float64)
    y_all = positions[:, 1].astype(np.float64)
    n = len(positions)
    if n == 0:
        return []

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    span_x = x_max - x_min
    span_y = y_max - y_min
    if span_x < 1 or span_y < 1:
        return []

    # Grid with aspect ratio preservation
    if span_x >= span_y:
        nx = resolution
        ny = max(1, int(resolution * span_y / span_x))
    else:
        ny = resolution
        nx = max(1, int(resolution * span_x / span_y))

    pixel_size_x = span_x / nx
    pixel_size_y = span_y / ny
    pixel_size = (pixel_size_x + pixel_size_y) / 2
    sigma_px = bandwidth_um / pixel_size

    # Bin all cells into grid
    cx_all = np.clip(((x_all - x_min) / span_x * nx).astype(int), 0, nx - 1)
    cy_all = np.clip(((y_all - y_min) / span_y * ny).astype(int), 0, ny - 1)

    # Total KDE (smoothed histogram of all cells)
    hist_all = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(hist_all, (cy_all, cx_all), 1)
    kde_all = gaussian_filter(hist_all, sigma=sigma_px)

    # Mask: require meaningful local density (at least ~1 cell per kernel area)
    density_threshold = max(kde_all.max() * 0.01, 1e-10)
    valid = kde_all > density_threshold

    unique_types = np.unique(types)
    regions = []

    # Enrichment thresholds: mild → strong
    enrichment_thresholds = [1.5, 2.5, 4.0, 8.0]
    level_labels = ['enriched', 'clustered', 'hotspot', 'core']

    for ti in unique_types:
        type_mask = types == ti
        n_type = int(type_mask.sum())
        idx = int(ti)
        label = type_labels[idx] if idx < len(type_labels) else f'type_{idx}'
        color = type_colors[idx] if idx < len(type_colors) else '#888888'

        if n_type < min_region_cells:
            continue

        global_prop = n_type / n

        # Type-specific KDE
        hist_type = np.zeros((ny, nx), dtype=np.float64)
        np.add.at(hist_type, (cy_all[type_mask], cx_all[type_mask]), 1)
        kde_type = gaussian_filter(hist_type, sigma=sigma_px)

        # Local proportion and enrichment
        prop_surface = np.zeros_like(kde_type)
        prop_surface[valid] = kde_type[valid] / kde_all[valid]

        enrichment = np.zeros_like(prop_surface)
        if global_prop > 0:
            enrichment[valid] = prop_surface[valid] / global_prop

        # Contour at each enrichment level (highest first so inner overrides)
        for thresh, level_label in zip(enrichment_thresholds, level_labels):
            binary = (enrichment >= thresh).astype(np.uint8) * 255

            # Morphological close to fill small gaps
            kern_size = max(3, int(sigma_px * 0.3)) | 1  # odd
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kern_size, kern_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area_px = cv2.contourArea(contour)
                if area_px < 50:
                    continue

                # RDP simplification
                epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
                contour = cv2.approxPolyDP(contour, epsilon, True)
                if len(contour) < 3:
                    continue

                # Count cells of THIS type inside contour
                cmask = np.zeros((ny, nx), dtype=np.uint8)
                cv2.drawContours(cmask, [contour], 0, 255, -1)
                inside_this = cmask[cy_all[type_mask], cx_all[type_mask]] > 0
                n_inside_type = int(inside_this.sum())
                if n_inside_type < min_region_cells:
                    continue

                # Count ALL cell types inside for composition
                inside_all = cmask[cy_all, cx_all] > 0
                n_inside_total = int(inside_all.sum())
                if n_inside_total == 0:
                    continue

                composition = {}
                for tj in unique_types:
                    count = int((inside_all & (types == tj)).sum())
                    if count > 0:
                        composition[type_labels[int(tj)]] = count

                dominant = max(composition, key=composition.get)
                dominant_frac = composition[dominant] / n_inside_total

                # Convert contour pixels to um coordinates
                boundary = []
                for pt in contour.reshape(-1, 2):
                    boundary.append({
                        'x': round(float(pt[0] * pixel_size_x + x_min), 1),
                        'y': round(float(pt[1] * pixel_size_y + y_min), 1),
                    })

                # Elongation from moments
                moments = cv2.moments(contour)
                if moments['m00'] > 0:
                    mu20 = moments['mu20'] / moments['m00']
                    mu02 = moments['mu02'] / moments['m00']
                    mu11 = moments['mu11'] / moments['m00']
                    d = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
                    major = mu20 + mu02 + d
                    minor = mu20 + mu02 - d
                    elongation = round(
                        np.sqrt(max(major, 1e-9) / max(minor, 1e-9)), 2)
                else:
                    elongation = 1.0

                area_um2 = round(area_px * pixel_size_x * pixel_size_y, 0)

                regions.append({
                    'id': len(regions),
                    'type': label,
                    'label': f'{label} ({level_label})',
                    'color': color,
                    'composition': {k: round(v / n_inside_total, 3)
                                    for k, v in composition.items()},
                    'n_cells': n_inside_total,
                    'area_um2': area_um2,
                    'elongation': elongation,
                    'dominant_frac': round(dominant_frac, 3),
                    'boundary': boundary,
                })

    # Sort by area descending, re-index
    regions.sort(key=lambda r: r['area_um2'], reverse=True)
    for i, r in enumerate(regions):
        r['id'] = i

    n_types_found = len(set(r['type'] for r in regions))
    print(f'Autocorrelation regions: {len(regions)} regions from '
          f'{n_types_found} types (>={min_region_cells} cells each)')
    return regions


# ---------------------------------------------------------------------------
# Graph-based pattern detection
# ---------------------------------------------------------------------------

def compute_graph_patterns(positions, types, type_labels, type_colors,
                           connect_radius_um=150, min_cluster_cells=8,
                           boundary_dilate_um=50):
    """Detect spatial patterns via graph-based connected components.

    Unlike KDE/enrichment (grid-based smoothing), this works at the cell level:

    1. Per type: KDTree → connect cells within *connect_radius_um*
    2. Connected components → discrete clusters
    3. Per cluster, classify spatial arrangement:
       - **linear**  : elongation > 4  (thin strand)
       - **arc**     : elongation > 3 AND significant quadratic curvature
       - **ring**    : circularity > 0.65 AND hollowness > 0.55
       - **cluster** : compact blob (everything else)
    4. Boundary: rasterise points → dilate → findContours → RDP simplify

    Returns list of region dicts (same schema as other region functions),
    with ``pattern`` field added.
    """
    import cv2
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(positions)
    if n == 0:
        return []

    x_all = positions[:, 0].astype(np.float64)
    y_all = positions[:, 1].astype(np.float64)

    # Global bounding box (for cell-inside-contour tests later)
    gx_min, gx_max = x_all.min(), x_all.max()
    gy_min, gy_max = y_all.min(), y_all.max()

    unique_types = np.unique(types)
    regions = []

    for ti in unique_types:
        type_mask = types == ti
        n_type = int(type_mask.sum())
        idx = int(ti)
        label = type_labels[idx] if idx < len(type_labels) else f'type_{idx}'
        color = type_colors[idx] if idx < len(type_colors) else '#888888'

        if n_type < min_cluster_cells:
            continue

        # Positions of this type only
        tp = positions[type_mask]  # (n_type, 2)

        # Build KDTree and find connected pairs
        tree = cKDTree(tp)
        pairs = tree.query_pairs(r=connect_radius_um)

        if not pairs:
            # No connections — each cell is isolated
            continue

        # Sparse adjacency matrix → connected components
        rows, cols = zip(*pairs)
        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n_type, n_type))
        adj = adj + adj.T

        n_components, comp_labels = connected_components(adj, directed=False)

        for ci in range(n_components):
            cmask = comp_labels == ci
            nc = int(cmask.sum())
            if nc < min_cluster_cells:
                continue

            pts = tp[cmask]  # (nc, 2) — positions of this cluster
            cx_mean = pts[:, 0].mean()
            cy_mean = pts[:, 1].mean()

            # ----- Pattern classification -----

            # PCA
            centered = pts - pts.mean(axis=0)
            cov = np.cov(centered.T) if nc > 2 else np.eye(2)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]  # descending
            lam1 = max(eigvals[0], 1e-10)
            lam2 = max(eigvals[1], 1e-10)
            linearity = lam1 / (lam1 + lam2)   # 0.5=isotropic, 1.0=line
            elongation = np.sqrt(lam1 / lam2)

            # Circle fit (center = mean, check radius consistency)
            radii = np.sqrt((pts[:, 0] - cx_mean)**2 +
                            (pts[:, 1] - cy_mean)**2)
            mean_r = radii.mean()
            if mean_r > 1e-6:
                circularity = 1.0 - (radii.std() / mean_r)
            else:
                circularity = 0.0
            # Hollowness: are points concentrated away from center?
            hollowness = np.median(radii) / max(radii.max(), 1e-6)

            # Curvature: project onto PC1, fit quadratic, check R²
            eigvecs = np.linalg.eigh(cov)[1]
            pc1 = eigvecs[:, -1]  # largest eigenvector
            pc2 = eigvecs[:, -2]
            proj1 = centered @ pc1  # along major axis
            proj2 = centered @ pc2  # perpendicular
            has_curvature = False
            if nc > 5 and elongation > 2.5:
                # Fit quadratic: proj2 = a*proj1^2 + b*proj1 + c
                coeffs = np.polyfit(proj1, proj2, 2)
                pred = np.polyval(coeffs, proj1)
                ss_res = ((proj2 - pred)**2).sum()
                ss_tot = ((proj2 - proj2.mean())**2).sum()
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                # Significant curvature if quadratic explains > 30% and
                # curvature coefficient is non-trivial
                if r2 > 0.3 and abs(coeffs[0]) > 1e-6:
                    has_curvature = True

            # Classify
            if elongation > 4 and not has_curvature:
                pattern = 'linear'
            elif elongation > 3 and has_curvature:
                pattern = 'arc'
            elif circularity > 0.65 and hollowness > 0.55 and elongation < 3:
                pattern = 'ring'
            else:
                pattern = 'cluster'

            # ----- Boundary via rasterisation -----

            # Local bounding box with padding
            pad = boundary_dilate_um
            bx_min = pts[:, 0].min() - pad
            bx_max = pts[:, 0].max() + pad
            by_min = pts[:, 1].min() - pad
            by_max = pts[:, 1].max() + pad
            bw = bx_max - bx_min
            bh = by_max - by_min

            # Rasterise at ~5 um/pixel (or min 64 px)
            target_px = max(64, min(512, int(max(bw, bh) / 5)))
            if bw >= bh:
                rnx = target_px
                rny = max(1, int(target_px * bh / bw))
            else:
                rny = target_px
                rnx = max(1, int(target_px * bw / bh))

            rpx = bw / max(rnx, 1)  # um per pixel x
            rpy = bh / max(rny, 1)  # um per pixel y

            # Rasterise points
            px = np.clip(((pts[:, 0] - bx_min) / bw * rnx).astype(int),
                         0, rnx - 1)
            py = np.clip(((pts[:, 1] - by_min) / bh * rny).astype(int),
                         0, rny - 1)
            raster = np.zeros((rny, rnx), dtype=np.uint8)
            raster[py, px] = 255

            # Dilate to connect nearby points and form contiguous region
            dilate_px = max(2, int(connect_radius_um / rpx * 0.5))
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            raster = cv2.dilate(raster, kern)
            # Slight close to fill interior gaps
            close_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px + 1, dilate_px + 1))
            raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, close_kern)

            contours, _ = cv2.findContours(
                raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            epsilon = max(1.0, 0.008 * cv2.arcLength(contour, True))
            contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(contour) < 3:
                continue

            # Convert pixel → um
            boundary = []
            for pt in contour.reshape(-1, 2):
                boundary.append({
                    'x': round(float(pt[0] * rpx + bx_min), 1),
                    'y': round(float(pt[1] * rpy + by_min), 1),
                })

            # Composition: count ALL cell types inside this boundary
            # Use raster mask (fast) instead of per-point pointPolygonTest
            cmask = np.zeros((rny, rnx), dtype=np.uint8)
            cv2.drawContours(cmask, [contour], 0, 255, -1)
            all_px = np.clip(
                ((positions[:, 0] - bx_min) / bw * rnx).astype(int),
                0, rnx - 1)
            all_py = np.clip(
                ((positions[:, 1] - by_min) / bh * rny).astype(int),
                0, rny - 1)
            inside_all = cmask[all_py, all_px] > 0
            n_inside_total = int(inside_all.sum())

            composition = {}
            for tj in unique_types:
                count = int((inside_all & (types == tj)).sum())
                if count > 0:
                    composition[type_labels[int(tj)]] = count

            if n_inside_total == 0:
                n_inside_total = nc
                composition = {label: nc}

            dominant = max(composition, key=composition.get)
            dominant_frac = composition[dominant] / max(n_inside_total, 1)

            # Area from contour
            contour_area_px = cv2.contourArea(contour)
            area_um2 = round(contour_area_px * rpx * rpy, 0)

            # Elongation from contour moments
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                mu20 = moments['mu20'] / moments['m00']
                mu02 = moments['mu02'] / moments['m00']
                mu11 = moments['mu11'] / moments['m00']
                d = np.sqrt(4 * mu11**2 + (mu20 - mu02)**2)
                major = mu20 + mu02 + d
                minor = mu20 + mu02 - d
                cont_elong = round(
                    np.sqrt(max(major, 1e-9) / max(minor, 1e-9)), 2)
            else:
                cont_elong = round(elongation, 2)

            regions.append({
                'id': len(regions),
                'type': label,
                'label': f'{label} ({pattern}, n={nc})',
                'color': color,
                'pattern': pattern,
                'composition': {k: round(v / max(n_inside_total, 1), 3)
                                for k, v in composition.items()},
                'n_cells': n_inside_total,
                'area_um2': area_um2,
                'elongation': cont_elong,
                'dominant_frac': round(dominant_frac, 3),
                'boundary': boundary,
            })

    # Sort by area descending, re-index
    regions.sort(key=lambda r: r['area_um2'], reverse=True)
    for i, r in enumerate(regions):
        r['id'] = i

    n_types_found = len(set(r['type'] for r in regions))
    patterns_summary = {}
    for r in regions:
        p = r['pattern']
        patterns_summary[p] = patterns_summary.get(p, 0) + 1
    pat_str = ', '.join(f'{v} {k}' for k, v in sorted(patterns_summary.items()))
    print(f'Graph patterns: {len(regions)} regions from {n_types_found} types '
          f'(>={min_cluster_cells} cells each): {pat_str}')
    return regions


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def hex_to_rgb(h):
    """Convert '#rrggbb' to (r, g, b) tuple."""
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def generate_html(scenes_data, output_path, title='Spatial Structure Viewer',
                  kde_bandwidth=600):
    """Generate self-contained interactive HTML viewer.

    Args:
        scenes_data: List of (label, data_dict) tuples.
        output_path: Where to write the HTML file.
        title: Page title.
        kde_bandwidth: Default KDE bandwidth in um.
    """
    n_scenes = len(scenes_data)
    if n_scenes <= 1:
        n_cols, n_rows = 1, 1
    elif n_scenes <= 2:
        n_cols, n_rows = 2, 1
    elif n_scenes <= 4:
        n_cols, n_rows = 2, 2
    elif n_scenes <= 6:
        n_cols, n_rows = 3, 2
    elif n_scenes <= 9:
        n_cols, n_rows = 3, 3
    else:
        n_cols = 4
        n_rows = (n_scenes + n_cols - 1) // n_cols

    # Serialize scene data to JS
    scenes_js_parts = []
    for label, data in scenes_data:
        groups_js = []
        for g in data['groups']:
            x_str = ','.join(f'{v:.1f}' for v in g['x'])
            y_str = ','.join(f'{v:.1f}' for v in g['y'])
            w_str = ','.join(f'{v:.3f}' for v in g['w'])

            groups_js.append(
                f'{{label:{json.dumps(g["label"])},color:"{g["color"]}",n:{g["n"]},'
                f'x:new Float32Array([{x_str}]),'
                f'y:new Float32Array([{y_str}]),'
                f'w:new Float32Array([{w_str}])}}'
            )

        # regions (if present)
        def _serialize_regions(reg_list):
            parts = []
            for r in reg_list:
                bnd = ','.join(f'[{p["x"]},{p["y"]}]' for p in r['boundary'])
                comp = ','.join(f'{json.dumps(k)}:{v}'
                                for k, v in r['composition'].items())
                parts.append(
                    f'{{id:{r["id"]},type:{json.dumps(r.get("type",""))},'
                    f'label:{json.dumps(r["label"])},'
                    f'color:"{r["color"]}",n:{r["n_cells"]},'
                    f'area:{r["area_um2"]},elong:{r["elongation"]},'
                    f'dfrac:{r["dominant_frac"]},'
                    f'comp:{{{comp}}},'
                    f'bnd:[{bnd}]}}'
                )
            return '[' + ','.join(parts) + ']'

        regions = data.get('regions', [])
        regions_js_str = _serialize_regions(regions)

        # Multi-scale regions (graph-patterns with multiple radii)
        region_scales = data.get('region_scales')
        if region_scales:
            scales_parts = []
            for scale_key in sorted(region_scales.keys(), key=int):
                scales_parts.append(
                    f'{json.dumps(scale_key)}:{_serialize_regions(region_scales[scale_key])}'
                )
            scales_js = '{' + ','.join(scales_parts) + '}'
        else:
            scales_js = 'null'

        scene_js = (
            f'{{name:{json.dumps(label)},n:{data["n_cells"]},'
            f'xr:[{data["x_range"][0]:.1f},{data["x_range"][1]:.1f}],'
            f'yr:[{data["y_range"][0]:.1f},{data["y_range"][1]:.1f}],'
            f'groups:[{",".join(groups_js)}],'
            f'regions:{regions_js_str},'
            f'regionScales:{scales_js}}}'
        )
        scenes_js_parts.append(scene_js)

    scenes_js = ',\n'.join(scenes_js_parts)

    # Build legend items from first scene's group order
    legend_items = []
    if scenes_data:
        for g in scenes_data[0][1]['groups']:
            total = sum(
                gg['n'] for _, d in scenes_data
                for gg in d['groups'] if gg['label'] == g['label']
            )
            legend_items.append((g['label'], g['color'], total))

    legend_html_parts = []
    for lbl, color, count in legend_items:
        safe_label = lbl.replace('&', '&amp;').replace('<', '&lt;').replace('"', '&quot;')
        legend_html_parts.append(
            f'<div class="leg-item" data-label="{safe_label}">'
            f'<span class="leg-swatch" style="background:{color}"></span>'
            f'<span class="leg-text">{safe_label} ({count:,})</span>'
            f'</div>'
        )
    legend_html = '\n'.join(legend_html_parts)

    has_regions = any(data.get('regions') for _, data in scenes_data)
    has_multiscale = any(data.get('region_scales') for _, data in scenes_data)

    # HTML-escape title
    title = html_mod.escape(title)

    # Build optional regions sidebar section
    regions_sidebar = ''
    if has_regions:
        # Scale dropdown (only for multi-scale graph patterns)
        scale_dropdown = ''
        if has_multiscale:
            # Get scale keys from first scene that has them
            scale_keys = []
            for _, data in scenes_data:
                if data.get('region_scales'):
                    scale_keys = sorted(data['region_scales'].keys(), key=int)
                    break
            default_idx = len(scale_keys) // 2
            default_scale = scale_keys[default_idx]
            scale_arr_js = ','.join(str(k) for k in scale_keys)
            scale_dropdown = f'''
    <div class="ctrl-row">
      <span>Scale</span>
      <input type="range" id="region-scale" min="0" max="{len(scale_keys)-1}" value="{default_idx}" style="flex:1">
      <span id="region-scale-label" style="min-width:60px;text-align:right">{default_scale} \u00b5m</span>
    </div>'''

        regions_sidebar = f'''<div class="ctrl-group">
    <h3>Regions</h3>
    <div class="ctrl-row">
      <span>Show</span>
      <input type="checkbox" id="show-regions" checked>
      <span>Labels</span>
      <input type="checkbox" id="show-region-labels" checked>
    </div>
    <div class="ctrl-row">
      <span>Opacity</span>
      <input type="range" id="region-opacity" min="0" max="0.8" value="0.25" step="0.05">
      <span class="val" id="region-op-val">0.25</span>
    </div>
    <div class="ctrl-row">
      <span>Boundaries</span>
      <input type="checkbox" id="show-region-bnd" checked>
    </div>{scale_dropdown}
  </div>'''

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ display:flex; height:100vh; background:#0a0a1a; color:#ddd; font:12px/1.4 system-ui,sans-serif; overflow:hidden; }}
#sidebar {{
  width:260px; min-width:260px; padding:10px; overflow-y:auto; background:#14142a;
  border-right:1px solid #333; display:flex; flex-direction:column; gap:10px;
}}
#sidebar h3 {{ font-size:13px; margin-bottom:4px; color:#aac; }}
.ctrl-group {{ background:#1a1a35; border-radius:6px; padding:8px; }}
.ctrl-row {{ display:flex; align-items:center; gap:6px; margin:4px 0; font-size:11px; }}
.ctrl-row span:first-child {{ min-width:65px; color:#99a; }}
.ctrl-row input[type=range] {{ flex:1; height:4px; accent-color:#6af; }}
.ctrl-row input[type=checkbox] {{ accent-color:#6af; }}
.ctrl-row .val {{ min-width:36px; text-align:right; color:#adf; font-variant-numeric:tabular-nums; }}
.leg-item {{ display:flex; align-items:center; gap:6px; padding:2px 4px; cursor:pointer; border-radius:3px; }}
.leg-item:hover {{ background:#222244; }}
.leg-item.hidden {{ opacity:0.35; }}
.leg-swatch {{ width:12px; height:12px; border-radius:2px; flex-shrink:0; }}
.leg-text {{ font-size:11px; }}
.btn {{ background:#2a2a4a; border:1px solid #444; color:#ccc; padding:3px 8px; border-radius:3px; cursor:pointer; font-size:11px; }}
.btn:hover {{ background:#3a3a5a; }}
#grid {{
  flex:1; display:grid;
  grid-template-columns: repeat({n_cols}, 1fr);
  grid-template-rows: repeat({n_rows}, 1fr);
  gap:2px; padding:2px;
}}
.panel {{ position:relative; overflow:hidden; background:#111122; border:1px solid #333; border-radius:4px; cursor:grab; }}
.panel.dragging {{ cursor:grabbing; }}
.panel canvas {{ position:absolute; top:0; left:0; }}
.panel-label {{ position:absolute; top:4px; left:6px; font-size:11px; color:#8af; z-index:2; pointer-events:none;
  text-shadow:0 0 4px #000, 0 0 2px #000; }}
.panel-count {{ position:absolute; top:4px; right:6px; font-size:10px; color:#8a8; z-index:2; pointer-events:none;
  text-shadow:0 0 4px #000; }}
</style>
</head>
<body>
<div id="sidebar">
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <h3>{title}</h3>
    <button id="fit-all-btn" style="padding:2px 8px;font-size:11px;cursor:pointer;" title="Reset zoom on all panels (or double-click a panel)">Fit</button>
  </div>

  {regions_sidebar}

  <div class="ctrl-group">
    <h3>KDE Density</h3>
    <div class="ctrl-row">
      <span>Show</span>
      <input type="checkbox" id="show-kde" checked>
    </div>
    <div class="ctrl-row">
      <span>Bandwidth</span>
      <input type="range" id="kde-bw" min="50" max="2000" value="{kde_bandwidth}" step="25">
      <span class="val" id="kde-bw-val">{kde_bandwidth}</span><span>um</span>
    </div>
    <div class="ctrl-row">
      <span>Levels</span>
      <input type="range" id="kde-levels" min="1" max="6" value="3" step="1">
      <span class="val" id="kde-levels-val">3</span>
    </div>
    <div class="ctrl-row">
      <span>Opacity</span>
      <input type="range" id="kde-opacity" min="0.1" max="1.0" value="0.5" step="0.05">
      <span class="val" id="kde-op-val">0.50</span>
    </div>
    <div class="ctrl-row">
      <span>Fill</span>
      <input type="checkbox" id="kde-fill" checked>
      <span>Lines</span>
      <input type="checkbox" id="kde-lines" checked>
    </div>
  </div>

  <div class="ctrl-group">
    <h3>Cell Dots</h3>
    <div class="ctrl-row">
      <span>Show</span>
      <input type="checkbox" id="show-dots" checked>
    </div>
    <div class="ctrl-row">
      <span>Size</span>
      <input type="range" id="dot-size" min="1" max="10" value="3" step="0.5">
      <span class="val" id="dot-val">3</span>
    </div>
    <div class="ctrl-row">
      <span>Opacity</span>
      <input type="range" id="dot-opacity" min="0.1" max="1.0" value="0.7" step="0.05">
      <span class="val" id="dot-op-val">0.70</span>
    </div>
  </div>

  <div class="ctrl-group">
    <h3>Cell Types</h3>
    <div style="margin-bottom:6px;display:flex;gap:4px;">
      <button class="btn" id="btn-all">Show All</button>
      <button class="btn" id="btn-none">Hide All</button>
    </div>
    <div id="leg-items">
      {legend_html}
    </div>
  </div>

  <div class="ctrl-group">
    <h3>Info</h3>
    <div id="status" style="font-size:10px;color:#888;"></div>
  </div>
</div>

<div id="grid"></div>

<script>
'use strict';

const SCENES = [{scenes_js}];

// --- State ---
const hidden = new Set();
let showDots = true, dotSize = 3, dotAlpha = 0.7;
let showKDE = true, kdeBW = {kde_bandwidth}, kdeLevels = 3, kdeAlpha = 0.5, kdeFill = true, kdeLines = true;
let showRegions = true, regionAlpha = 0.25, showRegionLabels = true, showRegionBnd = true;

// --- KDE caches ---
const kdeCache = new Map();  // panel_idx -> {{bw, levels, data, groups_key}}

// --- Panel state ---
const panels = [];
let activePanel = null;
let rafId = 0;
let rafDirty = new Set();

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

// --- Color utilities ---
function hexToRgb(h) {{
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}}

function rgbToHex(r, g, b) {{
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}}

function colorWithAlpha(hex, a) {{
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${{r}},${{g}},${{b}},${{a}})`;
}}

// =========================================================================
// Color utilities
// =========================================================================

// =========================================================================
// KDE: Histogram2D + Gaussian blur + Marching Squares
// =========================================================================

function computeHistogram2D(x, y, w, nx, ny, xr, yr) {{
  const grid = new Float32Array(ny * nx);
  const sx = nx / (xr[1] - xr[0]);
  const sy = ny / (yr[1] - yr[0]);
  const n = x.length;
  for (let i = 0; i < n; i++) {{
    const gx = Math.min(Math.floor((x[i] - xr[0]) * sx), nx - 1);
    const gy = Math.min(Math.floor((y[i] - yr[0]) * sy), ny - 1);
    if (gx >= 0 && gy >= 0) {{
      grid[gy * nx + gx] += (w ? w[i] : 1);
    }}
  }}
  return grid;
}}

function gaussianBlur1D(src, nx, ny, sigma, horizontal) {{
  const radius = Math.ceil(sigma * 3);
  const kernel = new Float32Array(2 * radius + 1);
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {{
    kernel[i + radius] = Math.exp(-0.5 * (i / sigma) * (i / sigma));
    ksum += kernel[i + radius];
  }}
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  const dst = new Float32Array(ny * nx);

  if (horizontal) {{
    for (let row = 0; row < ny; row++) {{
      for (let col = 0; col < nx; col++) {{
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {{
          const c = Math.min(Math.max(col + k, 0), nx - 1);
          sum += src[row * nx + c] * kernel[k + radius];
        }}
        dst[row * nx + col] = sum;
      }}
    }}
  }} else {{
    for (let col = 0; col < nx; col++) {{
      for (let row = 0; row < ny; row++) {{
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {{
          const r = Math.min(Math.max(row + k, 0), ny - 1);
          sum += src[r * nx + col] * kernel[k + radius];
        }}
        dst[row * nx + col] = sum;
      }}
    }}
  }}
  return dst;
}}

function gaussianBlur(grid, nx, ny, sigma) {{
  if (sigma < 0.5) return grid;
  const tmp = gaussianBlur1D(grid, nx, ny, sigma, true);
  return gaussianBlur1D(tmp, nx, ny, sigma, false);
}}

// Marching squares: extract iso-contour polygons from scalar grid
function marchingSquares(grid, nx, ny, threshold, xr, yr) {{
  // Use bin-center convention consistent with computeHistogram2D
  const stepX = (xr[1] - xr[0]) / nx;
  const stepY = (yr[1] - yr[0]) / ny;

  // Build edge segments
  const segments = [];
  for (let row = 0; row < ny - 1; row++) {{
    for (let col = 0; col < nx - 1; col++) {{
      const tl = grid[row * nx + col] >= threshold ? 1 : 0;
      const tr = grid[row * nx + col + 1] >= threshold ? 1 : 0;
      const br = grid[(row + 1) * nx + col + 1] >= threshold ? 1 : 0;
      const bl = grid[(row + 1) * nx + col] >= threshold ? 1 : 0;
      let code = (tl << 3) | (tr << 2) | (br << 1) | bl;

      if (code === 0 || code === 15) continue;

      // Interpolation helpers
      const x0 = xr[0] + col * stepX;
      const y0 = yr[0] + row * stepY;
      const vTL = grid[row * nx + col];
      const vTR = grid[row * nx + col + 1];
      const vBR = grid[(row + 1) * nx + col + 1];
      const vBL = grid[(row + 1) * nx + col];

      function lerp(v1, v2) {{
        const d = v2 - v1;
        return Math.abs(d) < 1e-10 ? 0.5 : (threshold - v1) / d;
      }}

      // Edge midpoints with linear interpolation
      const top = [x0 + lerp(vTL, vTR) * stepX, y0];
      const right = [x0 + stepX, y0 + lerp(vTR, vBR) * stepY];
      const bottom = [x0 + lerp(vBL, vBR) * stepX, y0 + stepY];
      const left = [x0, y0 + lerp(vTL, vBL) * stepY];

      // Saddle-point disambiguation: use center value to pick correct topology
      if (code === 5 || code === 10) {{
        const center = (vTL + vTR + vBR + vBL) / 4;
        if (center >= threshold) {{
          // Connect same-sign corners: code 5 -> [[left,bottom],[top,right]]
          //                            code 10 -> [[top,left],[bottom,right]]
          if (code === 5) code = 17;  // sentinel for alt config
          else code = 18;
        }}
      }}

      // Segment lookup table with saddle disambiguation
      let segs;
      switch (code) {{
        case 1:  segs = [[left, bottom]]; break;
        case 2:  segs = [[bottom, right]]; break;
        case 3:  segs = [[left, right]]; break;
        case 4:  segs = [[right, top]]; break;
        case 5:  segs = [[left, top], [bottom, right]]; break;
        case 17: segs = [[left, bottom], [top, right]]; break;  // code 5 alt
        case 6:  segs = [[bottom, top]]; break;
        case 7:  segs = [[left, top]]; break;
        case 8:  segs = [[top, left]]; break;
        case 9:  segs = [[top, bottom]]; break;
        case 10: segs = [[top, right], [left, bottom]]; break;
        case 18: segs = [[top, left], [bottom, right]]; break;  // code 10 alt
        case 11: segs = [[top, right]]; break;
        case 12: segs = [[right, left]]; break;
        case 13: segs = [[right, bottom]]; break;
        case 14: segs = [[bottom, left]]; break;
        default: segs = null;
      }}

      if (segs) {{
        for (const seg of segs) segments.push(seg);
      }}
    }}
  }}

  // Chain segments into polygons using endpoint spatial hash for O(n) total
  if (segments.length === 0) return [];

  const eps = stepX * 0.01;
  const eps2 = eps * eps;

  // String-based endpoint hash (safe for large coordinates)
  function ptKey(p) {{
    return Math.round(p[0] / eps) + ',' + Math.round(p[1] / eps);
  }}

  // Build endpoint hash: key -> list of {{segIdx, endIdx (0 or 1)}}
  const endHash = new Map();
  for (let i = 0; i < segments.length; i++) {{
    for (let e = 0; e < 2; e++) {{
      const k = ptKey(segments[i][e]);
      if (!endHash.has(k)) endHash.set(k, []);
      endHash.get(k).push({{ si: i, ei: e }});
    }}
  }}

  function dist2(a, b) {{
    const dx = a[0] - b[0], dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }}

  const polys = [];
  const used = new Uint8Array(segments.length);

  for (let start = 0; start < segments.length; start++) {{
    if (used[start]) continue;
    used[start] = 1;
    const poly = [segments[start][0], segments[start][1]];

    let found = true;
    while (found) {{
      found = false;
      const tail = poly[poly.length - 1];
      const rx = Math.round(tail[0] / eps);
      const ry = Math.round(tail[1] / eps);
      // Check hash bucket and neighboring buckets for rounding robustness
      for (let dx = -1; dx <= 1; dx++) {{
        for (let dy = -1; dy <= 1; dy++) {{
          const bucket = endHash.get((rx + dx) + ',' + (ry + dy));
          if (!bucket) continue;
          for (const entry of bucket) {{
            if (used[entry.si]) continue;
            const seg = segments[entry.si];
            if (dist2(tail, seg[entry.ei]) < eps2) {{
              poly.push(seg[1 - entry.ei]);
              used[entry.si] = 1;
              found = true;
              break;
            }}
            if (dist2(tail, seg[1 - entry.ei]) < eps2) {{
              poly.push(seg[entry.ei]);
              used[entry.si] = 1;
              found = true;
              break;
            }}
          }}
          if (found) break;
        }}
        if (found) break;
      }}
    }}
    if (poly.length >= 3) polys.push(poly);
  }}

  return polys;
}}

function computeKDE(scene, nxGrid, nyGrid, bandwidthUm, nLevels) {{
  const xr = scene.xr, yr = scene.yr;
  const dataW = xr[1] - xr[0], dataH = yr[1] - yr[0];
  if (dataW <= 0 || dataH <= 0) return null;

  const pixelSize = Math.max(dataW, dataH) / Math.max(nxGrid, nyGrid);
  const sigma = bandwidthUm / pixelSize;

  // Adaptive grid: maintain aspect ratio
  let nx, ny;
  if (dataW > dataH) {{
    nx = nxGrid;
    ny = Math.max(1, Math.round(nxGrid * dataH / dataW));
  }} else {{
    ny = nyGrid;
    nx = Math.max(1, Math.round(nyGrid * dataW / dataH));
  }}

  const result = [];

  for (let gi = 0; gi < scene.groups.length; gi++) {{
    const g = scene.groups[gi];
    if (hidden.has(g.label)) continue;
    if (g.n < 5) continue;

    const hist = computeHistogram2D(g.x, g.y, g.w, nx, ny, xr, yr);
    const blurred = gaussianBlur(hist, nx, ny, sigma);

    // Find max density
    let maxD = 0;
    for (let i = 0; i < blurred.length; i++) {{
      if (blurred[i] > maxD) maxD = blurred[i];
    }}
    if (maxD <= 0) continue;

    // Extract contours at multiple levels
    const contours = [];
    for (let li = 1; li <= nLevels; li++) {{
      const frac = li / (nLevels + 1);
      const threshold = maxD * frac;
      const polys = marchingSquares(blurred, nx, ny, threshold, xr, yr);
      contours.push({{ level: frac, polys }});
    }}

    result.push({{ gi, color: g.color, contours }});
  }}

  return result;
}}

function drawKDEContours(ctx, kdeData, scene, panZoom, opacity, fill, lines) {{
  if (!kdeData) return;

  for (const entry of kdeData) {{
    const color = entry.color;
    const [r, g, b] = hexToRgb(color);

    for (let li = entry.contours.length - 1; li >= 0; li--) {{
      const {{ level, polys }} = entry.contours[li];

      for (const poly of polys) {{
        if (poly.length < 3) continue;

        const path = new Path2D();
        path.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {{
          path.lineTo(poly[i][0], poly[i][1]);
        }}
        path.closePath();

        if (fill) {{
          ctx.globalAlpha = opacity * (1 - level) * 0.6;
          ctx.fillStyle = `rgba(${{r}},${{g}},${{b}},1)`;
          ctx.fill(path);
        }}

        if (lines) {{
          ctx.globalAlpha = opacity;
          ctx.strokeStyle = color;
          ctx.lineWidth = (1.5 - level * 0.5) / panZoom;
          ctx.stroke(path);
        }}
      }}
    }}
  }}
}}

// =========================================================================
// Regions (precomputed)
// =========================================================================

function drawRegions(ctx, regions, panZoom, opacity, showLabels, showBoundaries) {{
  if (!regions || regions.length === 0) return;

  for (const reg of regions) {{
    if (reg.bnd.length < 3) continue;
    if (reg.type && hidden.has(reg.type)) continue;

    const path = new Path2D();
    path.moveTo(reg.bnd[0][0], reg.bnd[0][1]);
    for (let i = 1; i < reg.bnd.length; i++) {{
      path.lineTo(reg.bnd[i][0], reg.bnd[i][1]);
    }}
    path.closePath();

    // Fill
    ctx.globalAlpha = opacity;
    ctx.fillStyle = reg.color;
    ctx.fill(path);

    // Boundary
    if (showBoundaries) {{
      ctx.globalAlpha = Math.min(opacity * 3, 0.9);
      ctx.strokeStyle = reg.color;
      ctx.lineWidth = 2.5 / panZoom;
      ctx.stroke(path);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.0 / panZoom;
      ctx.stroke(path);
    }}

    // Label
    if (showLabels) {{
      // Centroid of boundary
      let cx = 0, cy = 0;
      for (const pt of reg.bnd) {{ cx += pt[0]; cy += pt[1]; }}
      cx /= reg.bnd.length;
      cy /= reg.bnd.length;

      const fontSize = 11 / panZoom;
      ctx.font = `bold ${{fontSize}}px system-ui`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.globalAlpha = 0.9;

      const line1 = reg.label;
      const line2 = reg.n + ' cells (' + (reg.dfrac * 100).toFixed(0) + '%)';
      const lh = fontSize * 1.3;

      // Shadow
      ctx.fillStyle = '#000';
      ctx.fillText(line1, cx + 0.8/panZoom, cy - lh/2 + 0.8/panZoom);
      ctx.fillText(line2, cx + 0.8/panZoom, cy + lh/2 + 0.8/panZoom);
      // Text
      ctx.fillStyle = '#fff';
      ctx.fillText(line1, cx, cy - lh/2);
      ctx.fillText(line2, cx, cy + lh/2);
    }}
  }}
}}

// =========================================================================
// Cell Dots
// =========================================================================

function drawDots(ctx, groups, panZoom, size, opacity) {{
  ctx.globalAlpha = opacity;
  const r = size / panZoom;
  const halfR = r / 2;
  let total = 0;

  for (const g of groups) {{
    if (hidden.has(g.label)) continue;
    ctx.fillStyle = g.color;
    const n = g.n, gx = g.x, gy = g.y;
    for (let i = 0; i < n; i++) {{
      ctx.fillRect(gx[i] - halfR, gy[i] - halfR, r, r);
    }}
    total += n;
  }}
  return total;
}}

// =========================================================================
// Render pipeline
// =========================================================================

function getVisibleGroupsKey() {{
  return Array.from(hidden).sort().join('|') + ':' + hidden.size;
}}

function renderPanel(p) {{
  const cw = p.cw || p.div.getBoundingClientRect().width;
  const ch = p.ch || p.div.getBoundingClientRect().height;
  const ctx = p.ctx;
  const scene = p.scene;

  ctx.save();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#111122';
  ctx.fillRect(0, 0, cw, ch);

  ctx.translate(p.panX, p.panY);
  ctx.scale(p.zoom, p.zoom);

  const gKey = getVisibleGroupsKey();

  // Layer 1: Regions (lowest)
  if (showRegions && scene.regions && scene.regions.length > 0) {{
    drawRegions(ctx, scene.regions, p.zoom, regionAlpha, showRegionLabels, showRegionBnd);
  }}

  // Layer 2: KDE contours
  if (showKDE) {{
    const cacheKey = p.idx;
    const cached = kdeCache.get(cacheKey);
    let kdeData;
    const kdeRes = 200;  // fixed grid resolution for KDE
    if (cached && cached.bw === kdeBW && cached.levels === kdeLevels && cached.gKey === gKey) {{
      kdeData = cached.data;
    }} else {{
      kdeData = computeKDE(scene, kdeRes, kdeRes, kdeBW, kdeLevels);
      kdeCache.set(cacheKey, {{ bw: kdeBW, levels: kdeLevels, gKey: gKey, data: kdeData }});
    }}
    drawKDEContours(ctx, kdeData, scene, p.zoom, kdeAlpha, kdeFill, kdeLines);
  }}

  // Layer 4: Cell dots (top)
  let total = 0;
  if (showDots) {{
    total = drawDots(ctx, scene.groups, p.zoom, dotSize, dotAlpha);
  }} else {{
    for (const g of scene.groups) {{
      if (!hidden.has(g.label)) total += g.n;
    }}
  }}

  ctx.restore();
  p.countEl.textContent = total.toLocaleString() + ' cells';
}}

// =========================================================================
// Panel initialization
// =========================================================================

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
      div, canvas, ctx, countEl, scene, idx,
      zoom: 1, panX: 0, panY: 0,
      dragStartX: 0, dragStartY: 0, panStartX: 0, panStartY: 0,
      cw: 0, ch: 0,
    }};
    panels.push(state);

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

    div.addEventListener('dblclick', e => {{
      fitPanel(state);
      scheduleRender(state);
    }});
  }});

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
    p.zoom = 1; p.panX = cw / 2; p.panY = ch / 2;
    return;
  }}
  const pad = 0.05;
  p.zoom = Math.min(cw / (dataW * (1 + 2*pad)), ch / (dataH * (1 + 2*pad)));
  p.panX = (cw - dataW * p.zoom) / 2 - s.xr[0] * p.zoom;
  p.panY = (ch - dataH * p.zoom) / 2 - s.yr[0] * p.zoom;
}}

// =========================================================================
// Legend + Controls wiring
// =========================================================================

function initLegend() {{
  document.getElementById('fit-all-btn').onclick = () => {{
    panels.forEach(fitPanel);
    scheduleRenderAll();
  }};

  document.querySelectorAll('.leg-item').forEach(el => {{
    el.addEventListener('click', () => {{
      const label = el.dataset.label;
      if (hidden.has(label)) {{
        hidden.delete(label);
        el.classList.remove('hidden');
      }} else {{
        hidden.add(label);
        el.classList.add('hidden');
      }}
      invalidateCaches();
      scheduleRenderAll();
    }});
  }});

  document.getElementById('btn-all').onclick = () => {{
    hidden.clear();
    document.querySelectorAll('.leg-item').forEach(el => el.classList.remove('hidden'));
    invalidateCaches();
    scheduleRenderAll();
  }};
  document.getElementById('btn-none').onclick = () => {{
    document.querySelectorAll('.leg-item').forEach(el => {{
      hidden.add(el.dataset.label);
      el.classList.add('hidden');
    }});
    invalidateCaches();
    scheduleRenderAll();
  }};
}}

function invalidateCaches() {{
  kdeCache.clear();
}}

function initControls() {{
  // Dot controls
  const dotShowEl = document.getElementById('show-dots');
  dotShowEl.onchange = () => {{ showDots = dotShowEl.checked; scheduleRenderAll(); }};
  document.getElementById('dot-size').oninput = e => {{
    dotSize = parseFloat(e.target.value);
    document.getElementById('dot-val').textContent = dotSize;
    scheduleRenderAll();
  }};
  document.getElementById('dot-opacity').oninput = e => {{
    dotAlpha = parseFloat(e.target.value);
    document.getElementById('dot-op-val').textContent = dotAlpha.toFixed(2);
    scheduleRenderAll();
  }};

  // Debounce helper for expensive recomputation sliders
  let _debounceTimers = {{}};
  function debouncedRecompute(id, ms) {{
    clearTimeout(_debounceTimers[id]);
    _debounceTimers[id] = setTimeout(() => {{
      invalidateCaches();
      scheduleRenderAll();
    }}, ms);
  }}

  // KDE controls
  const kdeShowEl = document.getElementById('show-kde');
  kdeShowEl.onchange = () => {{ showKDE = kdeShowEl.checked; invalidateCaches(); scheduleRenderAll(); }};
  document.getElementById('kde-bw').oninput = e => {{
    kdeBW = parseInt(e.target.value);
    document.getElementById('kde-bw-val').textContent = kdeBW;
    debouncedRecompute('kde-bw', 200);
  }};
  document.getElementById('kde-levels').oninput = e => {{
    kdeLevels = parseInt(e.target.value);
    document.getElementById('kde-levels-val').textContent = kdeLevels;
    debouncedRecompute('kde-levels', 200);
  }};
  document.getElementById('kde-opacity').oninput = e => {{
    kdeAlpha = parseFloat(e.target.value);
    document.getElementById('kde-op-val').textContent = kdeAlpha.toFixed(2);
    scheduleRenderAll();
  }};
  document.getElementById('kde-fill').onchange = e => {{
    kdeFill = e.target.checked;
    scheduleRenderAll();
  }};
  document.getElementById('kde-lines').onchange = e => {{
    kdeLines = e.target.checked;
    scheduleRenderAll();
  }};

  // SAM2 Region controls (if present)
  const regionShowEl = document.getElementById('show-regions');
  if (regionShowEl) {{
    regionShowEl.onchange = () => {{ showRegions = regionShowEl.checked; scheduleRenderAll(); }};
    document.getElementById('region-opacity').oninput = e => {{
      regionAlpha = parseFloat(e.target.value);
      document.getElementById('region-op-val').textContent = regionAlpha.toFixed(2);
      scheduleRenderAll();
    }};
    document.getElementById('show-region-labels').onchange = e => {{
      showRegionLabels = e.target.checked;
      scheduleRenderAll();
    }};
    document.getElementById('show-region-bnd').onchange = e => {{
      showRegionBnd = e.target.checked;
      scheduleRenderAll();
    }};

    // Multi-scale region slider
    const scaleEl = document.getElementById('region-scale');
    if (scaleEl) {{
      const SCALE_KEYS = [{scale_arr_js}];
      const scaleLabelEl = document.getElementById('region-scale-label');
      // Apply default scale
      const defaultKey = String(SCALE_KEYS[parseInt(scaleEl.value)]);
      for (const sc of SCENES) {{
        if (sc.regionScales && sc.regionScales[defaultKey]) {{
          sc.regions = sc.regionScales[defaultKey];
        }}
      }}
      let _scaleTimer = null;
      scaleEl.oninput = () => {{
        const idx = parseInt(scaleEl.value);
        const val = SCALE_KEYS[idx];
        if (scaleLabelEl) scaleLabelEl.textContent = val + ' \u00b5m';
        if (_scaleTimer) clearTimeout(_scaleTimer);
        _scaleTimer = setTimeout(() => {{
          const key = String(val);
          for (const sc of SCENES) {{
            if (sc.regionScales && sc.regionScales[key]) {{
              sc.regions = sc.regionScales[key];
            }}
          }}
          scheduleRenderAll();
        }}, 200);
      }};
    }}
  }}
}}

// =========================================================================
// Init
// =========================================================================

function fullInit() {{
  initPanels();
  resizePanels();
  panels.forEach(fitPanel);
  initLegend();
  initControls();
  scheduleRenderAll();

  const totalCells = SCENES.reduce((s, sc) => s + sc.n, 0);
  const totalRegions = SCENES.reduce((s, sc) => s + (sc.regions ? sc.regions.length : 0), 0);
  document.getElementById('status').textContent =
    SCENES.length + ' scenes | ' + totalCells.toLocaleString() + ' cells' +
    (totalRegions > 0 ? ' | ' + totalRegions + ' regions' : '');
}}

setTimeout(fullInit, 50);
window.addEventListener('resize', () => {{
  resizePanels();
  panels.forEach(fitPanel);
  scheduleRenderAll();
}});
</script>
</body>
</html>'''

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    total_cells = sum(d['n_cells'] for _, d in scenes_data)
    total_regions = sum(len(d.get('regions', [])) for _, d in scenes_data)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f'Wrote {output_path} ({size_mb:.1f} MB)')
    print(f'  {n_scenes} scenes, {total_cells:,} cells, {total_regions} regions')
    print(f'  Grid: {n_cols}x{n_rows}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive spatial structure viewer with KDE density, '
                    'regions, and cell dots')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir',
                             help='Directory with sample subdirs, each containing detection JSON')
    input_group.add_argument('--detections', nargs='+', metavar='FILE',
                             help='Direct detection JSON file(s)')

    parser.add_argument('--subdir', default='',
                        help='Subdirectory within each sample dir to find detection JSON (--input-dir mode)')
    parser.add_argument('--detection-filename', default='detections_neurons_hybrid.json',
                        help='Detection filename pattern (--input-dir mode, default: detections_neurons_hybrid.json)')
    parser.add_argument('--group-field', default='marker_class',
                        help='Field to group by (default: marker_class)')
    parser.add_argument('--top-n', type=int, default=12,
                        help='Keep top N groups by cell count (default: 12)')
    parser.add_argument('--exclude-groups', default='',
                        help='Comma-separated group labels to drop')
    parser.add_argument('--include-groups', default='',
                        help='Comma-separated group labels to keep (exclusive, no "other")')
    parser.add_argument('--weight-field', default=None,
                        help='Detection field for KDE weighting (e.g., ch3_corrected)')

    parser.add_argument('--kde-bandwidth', type=int, default=600,
                        help='Default KDE bandwidth in um (default: 600)')

    parser.add_argument('--cv-regions', action='store_true',
                        help='Compute per-type structural regions via Canny + contours (no GPU)')
    parser.add_argument('--autocorrelation', action='store_true',
                        help='Compute spatial autocorrelation regions (enrichment vs local density)')
    parser.add_argument('--graph-patterns', action='store_true',
                        help='Graph-based pattern detection (lines, arcs, rings, clusters)')
    parser.add_argument('--connect-radius', type=float, nargs='+', default=[150],
                        help='Max distance(s) (um) to connect cells in graph. '
                             'Multiple values enable scale toggle (e.g. --connect-radius 75 150 300)')
    parser.add_argument('--sam2-regions', action='store_true',
                        help='Compute SAM2 structural regions from density maps (requires GPU)')
    parser.add_argument('--region-resolution', type=int, default=1024,
                        help='KDE image resolution for region computation (default: 1024)')
    parser.add_argument('--min-region-cells', type=int, default=50,
                        help='Minimum cells per region (default: 50)')


    parser.add_argument('--output', default=None,
                        help='Output HTML path')
    parser.add_argument('--title', default=None,
                        help='Page title')

    args = parser.parse_args()

    exclude_groups = set(g.strip() for g in args.exclude_groups.split(',') if g.strip())
    include_groups = set(g.strip() for g in args.include_groups.split(',') if g.strip()) or None

    # Load data
    scenes_data = []

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f'ERROR: Input directory not found: {input_dir}', file=sys.stderr)
            sys.exit(1)

        # Look for sample subdirectories
        for sample_dir in sorted(input_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if args.subdir:
                search_dir = sample_dir / args.subdir
            else:
                search_dir = sample_dir

            if not search_dir.exists():
                continue

            det_file = search_dir / args.detection_filename
            if not det_file.exists():
                # Try glob
                candidates = list(search_dir.glob('detections*.json'))
                if candidates:
                    det_file = candidates[0]
                else:
                    continue

            data = load_file_data(det_file, args.group_field, args.weight_field)
            if data:
                label = sample_dir.name
                scenes_data.append((label, data))
                print(f'  Loaded {label}: {data["n_cells"]:,} cells, '
                      f'{len(data["groups"])} groups')
    else:
        for path in args.detections:
            path = Path(path)
            data = load_file_data(path, args.group_field, args.weight_field)
            if data:
                # Derive label from path (include grandparent for uniqueness)
                parent = path.parent.name
                grandparent = path.parent.parent.name
                if parent in ('.', ''):
                    label = path.stem
                elif grandparent in ('.', '', 'brain_fish_output'):
                    label = parent
                else:
                    label = f'{grandparent}/{parent}'
                scenes_data.append((label, data))
                print(f'  Loaded {label}: {data["n_cells"]:,} cells, '
                      f'{len(data["groups"])} groups')

    if not scenes_data:
        print('ERROR: No valid detection data found.', file=sys.stderr)
        sys.exit(1)

    print(f'\nLoaded {len(scenes_data)} scenes')

    # Apply colors
    color_map = apply_colors(scenes_data, args.top_n, exclude_groups, include_groups)
    print(f'Assigned colors to {len(color_map)} groups')

    # Region computation (optional)
    has_region_mode = (args.cv_regions or args.autocorrelation
                       or args.graph_patterns or args.sam2_regions)
    if has_region_mode:
        if args.graph_patterns:
            mode = 'Graph-based pattern detection'
        elif args.autocorrelation:
            mode = 'Spatial autocorrelation (enrichment)'
        elif args.cv_regions:
            mode = 'CV (Canny + contours)'
        else:
            mode = 'SAM2'
        print(f'\n--- {mode} Region Computation ---')

        for label, data in scenes_data:
            print(f'\nScene: {label}')

            # Collect all visible cells
            all_x, all_y, all_types = [], [], []
            type_labels = []
            type_colors = []
            for gi, g in enumerate(data['groups']):
                type_labels.append(g['label'])
                type_colors.append(g['color'])
                for i in range(g['n']):
                    all_x.append(g['x'][i])
                    all_y.append(g['y'][i])
                    all_types.append(gi)

            if not all_x:
                data['regions'] = []
                continue

            positions = np.array(list(zip(all_x, all_y)), dtype=np.float32)
            types_arr = np.array(all_types, dtype=np.int32)

            if args.graph_patterns:
                # Graph-based connected components + pattern classification
                radii = sorted(args.connect_radius)
                if len(radii) > 1:
                    # Multi-scale: compute for each radius
                    scales = {}
                    for r in radii:
                        print(f'  radius={r} um ...')
                        scales[str(int(r))] = compute_graph_patterns(
                            positions, types_arr, type_labels, type_colors,
                            connect_radius_um=r,
                            min_cluster_cells=args.min_region_cells,
                            boundary_dilate_um=r * 0.4,
                        )
                    data['region_scales'] = scales
                    # Default to middle scale
                    data['regions'] = scales[str(int(radii[len(radii) // 2]))]
                else:
                    regions = compute_graph_patterns(
                        positions, types_arr, type_labels, type_colors,
                        connect_radius_um=radii[0],
                        min_cluster_cells=args.min_region_cells,
                        boundary_dilate_um=radii[0] * 0.4,
                    )
                    data['regions'] = regions
                    data['region_scales'] = None
            elif args.autocorrelation:
                # Spatial autocorrelation (enrichment vs local density)
                data['regions'] = compute_autocorrelation_regions(
                    positions, types_arr, type_labels, type_colors,
                    bandwidth_um=args.kde_bandwidth,
                    resolution=args.region_resolution,
                    min_region_cells=args.min_region_cells,
                )
            elif args.cv_regions:
                # Per-type Canny + contours (no GPU)
                data['regions'] = compute_cv_regions(
                    positions, types_arr, type_labels, type_colors,
                    bandwidth_um=args.kde_bandwidth,
                    resolution=args.region_resolution,
                    min_region_cells=args.min_region_cells,
                )
            else:
                # SAM2 (requires GPU)
                kde_colors = [hex_to_rgb(c) for c in type_colors]
                kde_img, x_range, y_range, pixel_size = compute_kde_image(
                    positions, types_arr, kde_colors,
                    resolution=args.region_resolution,
                    bandwidth_um=args.kde_bandwidth,
                )
                if kde_img is None:
                    data['regions'] = []
                    continue

                data['regions'] = run_sam2_regions(
                    kde_img, positions, types_arr, type_labels, type_colors,
                    x_range, y_range, pixel_size,
                    min_region_cells=args.min_region_cells,
                )

    # Generate HTML
    title = args.title or f'Spatial Structure: {args.group_field}'
    output = args.output
    if output is None:
        if args.input_dir:
            output = str(Path(args.input_dir) / 'spatial_structure_viewer.html')
        else:
            output = 'spatial_structure_viewer.html'

    generate_html(
        scenes_data, output,
        title=title,
        kde_bandwidth=args.kde_bandwidth,
    )


if __name__ == '__main__':
    main()
