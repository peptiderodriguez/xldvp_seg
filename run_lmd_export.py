#!/usr/bin/env python3
"""
Unified LMD Export Tool - Export detections to Leica LMD format.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Handles the complete pipeline:
1. Load detections + filter by score/annotations
2. Load biological clusters (from cluster_detections.py)
3. Extract contours from H5 masks (if needed)
4. Post-process contours (dilate + RDP simplify)
5. Order singles and clusters by nearest-neighbor path on slide
6. Generate spatial controls for ALL samples (singles and clusters)
7. Assign wells in serpentine order with alternating target/control
8. Export to Leica LMD XML via py-lmd

Usage:
    # Full pipeline with clusters and controls
    python run_lmd_export.py \\
        --detections detections.json \\
        --cell-type nmj \\
        --crosses reference_crosses.json \\
        --clusters clusters.json \\
        --tiles-dir /path/to/tiles \\
        --output-dir lmd_export \\
        --export --generate-controls

    # Generate cross placement HTML (step before export)
    python run_lmd_export.py \\
        --detections detections.json \\
        --output-dir lmd_export \\
        --generate-cross-html
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_detections(detections_path):
    """Load detections from JSON file."""
    with open(detections_path, 'r') as f:
        return json.load(f)


def load_annotations(annotations_path):
    """
    Load annotations and return set of positive UIDs.

    Supports formats:
    - {"positive": [...], "negative": [...]}
    - {"annotations": {"uid": "yes/no", ...}}
    - Plain list of UIDs
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    positive_uids = set()

    if 'positive' in data:
        positive_uids.update(data['positive'])
    elif 'annotations' in data:
        for uid, label in data['annotations'].items():
            if label.lower() in ('yes', 'positive', 'true', '1'):
                positive_uids.add(uid)
    elif isinstance(data, list):
        positive_uids.update(data)

    return positive_uids


def filter_detections(detections, positive_uids=None, min_score=None):
    """Filter detections by annotations and/or score."""
    filtered = []
    for det in detections:
        uid = det.get('uid', det.get('id', ''))

        if positive_uids is not None and uid not in positive_uids:
            continue

        if min_score is not None:
            score = det.get('rf_prediction', det.get('score', 0))
            if score is None:
                score = 0
            if score < min_score:
                continue

        filtered.append(det)
    return filtered


def load_clusters(clusters_path):
    """Load and validate clusters JSON from cluster_detections.py."""
    with open(clusters_path, 'r') as f:
        data = json.load(f)

    if 'main_clusters' not in data or 'outliers' not in data:
        raise ValueError(f"Invalid clusters file: missing 'main_clusters' or 'outliers' keys")

    return data


def get_detection_coordinates(det):
    """Extract (x, y) pixel coordinates from detection."""
    if 'global_center' in det:
        return det['global_center']
    if 'center' in det:
        return det['center']
    return None


# ---------------------------------------------------------------------------
# Contour extraction from H5 masks
# ---------------------------------------------------------------------------

def extract_contour_from_mask(mask, label):
    """Extract the outer contour for a specific label in a mask array."""
    import cv2

    binary = (mask == label).astype(np.uint8)
    if binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def extract_contours_for_detections(detections, tiles_dir, pixel_size,
                                    mask_filename='nmj_masks.h5',
                                    dilation_um=0.5, rdp_epsilon=5.0):
    """
    Extract and process contours for detections from H5 mask files.

    Skips detections that already have 'contour_um' set.

    Returns dict mapping uid -> contour data.
    """
    import hdf5plugin  # noqa: F401 - must import before h5py for LZ4
    import h5py
    from segmentation.lmd.contour_processing import process_contour

    tiles_dir = Path(tiles_dir)

    # Group by tile
    by_tile = {}
    for det in detections:
        # Skip if contour already exists
        if det.get('contour_um') is not None:
            continue

        tile_origin = det.get('tile_origin', [0, 0])
        tile_name = f"tile_{tile_origin[0]}_{tile_origin[1]}"
        if tile_name not in by_tile:
            by_tile[tile_name] = []
        by_tile[tile_name].append(det)

    if not by_tile:
        print("    All detections already have contours, skipping extraction.")
        return {}

    results = {}
    for tile_idx, (tile_name, tile_dets) in enumerate(by_tile.items()):
        if (tile_idx + 1) % 20 == 0:
            print(f"    Tile {tile_idx + 1}/{len(by_tile)}...")

        mask_path = tiles_dir / tile_name / mask_filename
        if not mask_path.exists():
            continue

        with h5py.File(mask_path, 'r') as hf:
            masks = hf['masks'][:]

        for det in tile_dets:
            uid = det.get('uid', det.get('id', ''))
            label = det.get('mask_label')
            if label is None:
                # Fallback for single-GPU path: parse from id (e.g., "nmj_3" -> 3)
                det_id = det.get('id', '')
                try:
                    label = int(det_id.split('_')[-1])
                except (ValueError, IndexError):
                    continue

            contour_local = extract_contour_from_mask(masks, label)
            if contour_local is None:
                continue

            # Convert to global coordinates
            tile_origin = det.get('tile_origin', [0, 0])
            contour_global = contour_local.astype(float)
            contour_global[:, 0] += tile_origin[0]
            contour_global[:, 1] += tile_origin[1]

            # Apply post-processing (dilation + RDP)
            processed, stats = process_contour(
                contour_global.tolist(),
                pixel_size_um=pixel_size,
                dilation_um=dilation_um,
                rdp_epsilon=rdp_epsilon,
                return_stats=True,
            )

            if processed is None:
                continue

            results[uid] = {
                'contour_global_px': contour_global.tolist(),
                'contour_um': processed.tolist(),
                'area_um2': stats['area_after_um2'],
                'n_points': stats['points_after'],
            }

    return results


# ---------------------------------------------------------------------------
# Nearest-neighbor path ordering
# ---------------------------------------------------------------------------

def nearest_neighbor_order(points, start_idx=None):
    """Order points using nearest-neighbor algorithm with KDTree for efficiency.

    Uses scipy.spatial.cKDTree to find nearest unvisited neighbor at each step,
    replacing the naive O(n^2) inner loop with O(n log n) lookups.

    Returns index order.
    """
    from scipy.spatial import cKDTree

    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]

    points_arr = np.array(points, dtype=np.float64)

    if start_idx is None:
        # Start from top-left-most point
        start_idx = int(np.argmin(points_arr[:, 0] + points_arr[:, 1]))

    # Build KDTree for efficient nearest-neighbor queries
    tree = cKDTree(points_arr)

    visited = np.zeros(n, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True

    current = start_idx
    for _ in range(n - 1):
        # Query progressively more neighbors until we find an unvisited one.
        # Start with k=2 (self + 1 neighbor); increase if all found are visited.
        k = min(8, n)  # Start with a reasonable batch size
        while True:
            dists, indices = tree.query(points_arr[current], k=k)
            # Make sure dists/indices are arrays even for k=1
            if np.ndim(dists) == 0:
                dists = np.array([dists])
                indices = np.array([indices])

            # Find first unvisited neighbor (sorted by distance)
            found = False
            for idx in indices:
                if idx < n and not visited[idx]:
                    order.append(int(idx))
                    visited[idx] = True
                    current = int(idx)
                    found = True
                    break

            if found:
                break

            # All k neighbors were visited; increase k
            if k >= n:
                # All points visited (shouldn't happen but guard against it)
                break
            k = min(k * 2, n)

    return order


# ---------------------------------------------------------------------------
# Serpentine well generation (384-well plate, 4 quadrants)
# ---------------------------------------------------------------------------

def generate_quadrant_serpentine(quadrant, start_corner='auto'):
    """Generate wells for a 384-well quadrant in serpentine order."""
    even_rows = ['B', 'D', 'F', 'H', 'J', 'L', 'N']
    odd_rows = ['C', 'E', 'G', 'I', 'K', 'M', 'O']
    even_cols = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    odd_cols = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

    if quadrant == 'B2':
        rows, cols = even_rows, even_cols
    elif quadrant == 'B3':
        rows, cols = even_rows, odd_cols
    elif quadrant == 'C2':
        rows, cols = odd_rows, even_cols
    elif quadrant == 'C3':
        rows, cols = odd_rows, odd_cols
    else:
        raise ValueError(f"Unknown quadrant: {quadrant}")

    if start_corner == 'auto':
        start_corner = 'TL' if quadrant.startswith('B') else 'BR'

    if start_corner == 'TL':
        row_order = rows
        first_row_left_to_right = True
    elif start_corner == 'TR':
        row_order = rows
        first_row_left_to_right = False
    elif start_corner == 'BL':
        row_order = list(reversed(rows))
        first_row_left_to_right = True
    elif start_corner == 'BR':
        row_order = list(reversed(rows))
        first_row_left_to_right = False
    else:
        raise ValueError(f"Unknown start_corner: {start_corner}")

    wells = []
    for i, row in enumerate(row_order):
        if i % 2 == 0:
            col_order = cols if first_row_left_to_right else list(reversed(cols))
        else:
            col_order = list(reversed(cols)) if first_row_left_to_right else cols
        for col in col_order:
            wells.append(f"{row}{col}")

    return wells


def generate_wells_serpentine_4_quadrants(n_wells):
    """
    Generate wells in serpentine order across 4 quadrants: B2 -> B3 -> C3 -> C2.

    Uses auto start corners that minimize travel between quadrants:
      B2: TL start (B2 -> ... -> N22)
      B3: determined from last well of B2
      C3: determined from last well of B3
      C2: determined from last well of C3

    Each quadrant has 77 wells (7 rows x 11 cols). Total = 308 wells.
    """
    if n_wells <= 0:
        return []

    MAX_WELLS = 308  # 4 quadrants x 77 wells each
    if n_wells > MAX_WELLS:
        raise ValueError(
            f"Requested {n_wells} wells but 384-well plate only has {MAX_WELLS} usable wells "
            f"(4 quadrants x 77). Reduce the number of detections or split across multiple plates."
        )

    quadrant_order = ['B2', 'B3', 'C3', 'C2']
    all_wells = []

    for i, quad in enumerate(quadrant_order):
        if i == 0:
            wells = generate_quadrant_serpentine(quad, start_corner='TL')
        else:
            # Determine start corner from last well of previous quadrant
            prev_well = all_wells[-1]
            prev_row, prev_col = prev_well[0], int(prev_well[1:])
            top_rows = set('BCDEFGH')
            is_top = prev_row in top_rows
            is_left = prev_col <= 12

            if is_top and is_left:
                start = 'TL'
            elif is_top and not is_left:
                start = 'TR'
            elif not is_top and is_left:
                start = 'BL'
            else:
                start = 'BR'

            wells = generate_quadrant_serpentine(quad, start_corner=start)

        all_wells.extend(wells)
        if len(all_wells) >= n_wells:
            return all_wells[:n_wells]

    return all_wells[:n_wells]


# ---------------------------------------------------------------------------
# Spatial control generation
# ---------------------------------------------------------------------------

# 8 cardinal directions
CONTROL_DIRECTIONS = {
    'E':  np.array([1, 0]),
    'NE': np.array([1, -1]) / np.sqrt(2),
    'N':  np.array([0, -1]),
    'NW': np.array([-1, -1]) / np.sqrt(2),
    'W':  np.array([-1, 0]),
    'SW': np.array([-1, 1]) / np.sqrt(2),
    'S':  np.array([0, 1]),
    'SE': np.array([1, 1]) / np.sqrt(2),
}


def _make_polygon(contour):
    """Create a validated Shapely Polygon from a contour array. Returns None on failure."""
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
    except ImportError:
        return None
    try:
        contour = np.asarray(contour)
        if len(contour) < 3:
            return None
        poly = Polygon(contour)
        if not poly.is_valid:
            poly = make_valid(poly)
        return poly
    except Exception:
        return None


def _build_spatial_index(precomputed_polygons):
    """Build spatial index from valid polygons.

    Uses Shapely STRtree for O(log N) intersection queries instead of O(N) linear scan.

    Args:
        precomputed_polygons: list of Shapely Polygon objects (or None entries)

    Returns:
        STRtree instance, or None if no valid polygons or STRtree unavailable.
    """
    try:
        from shapely import STRtree
    except ImportError:
        return None
    valid_polys = [p for p in precomputed_polygons if p is not None]
    if not valid_polys:
        return None
    return STRtree(valid_polys)


def _check_overlap_indexed(shifted_contour, spatial_tree):
    """Check overlap using spatial index -- O(log N) instead of O(N).

    Args:
        shifted_contour: numpy array (N, 2) of the candidate contour
        spatial_tree: STRtree instance (or None for no-collision case)

    Returns True if overlap detected, False if no overlap.
    """
    candidate_poly = _make_polygon(shifted_contour)
    if candidate_poly is None or not candidate_poly.is_valid:
        return True  # Treat invalid as overlapping (skip)
    if spatial_tree is None:
        return False
    return len(spatial_tree.query(candidate_poly, predicate='intersects')) > 0


def _check_overlap_precomputed(shifted_contour, precomputed_polygons):
    """Check if shifted contour overlaps any precomputed polygon (linear fallback).

    Args:
        shifted_contour: numpy array (N, 2) of the candidate contour
        precomputed_polygons: list of Shapely Polygon objects (or None entries)

    Returns True if overlap detected, False if no overlap.
    """
    candidate_poly = _make_polygon(shifted_contour)
    if candidate_poly is None:
        return True  # Can't validate, assume overlap

    for existing_poly in precomputed_polygons:
        if existing_poly is None:
            continue
        try:
            if candidate_poly.intersects(existing_poly):
                return True
        except Exception:
            return True
    return False


def generate_spatial_control(contour_um, precomputed_polygons,
                             offset_um=100.0, max_attempts=3,
                             spatial_tree=None):
    """
    Generate a control contour by shifting in 8 directions.

    Every sample MUST have a control. If all 8 directions at the initial
    offset overlap, we increase the offset by 50% and retry, up to
    max_attempts times.

    Args:
        contour_um: Contour in um, shape (N, 2)
        precomputed_polygons: List of precomputed Shapely Polygon objects (fallback)
        offset_um: Starting offset distance in um
        max_attempts: Number of times to increase offset if all directions fail
        spatial_tree: Optional STRtree for O(log N) overlap checks

    Returns:
        (shifted_contour_um, direction_name, actual_offset_um) tuple.
        Always returns a result (falls back to largest-gap direction).
    """
    contour_arr = np.array(contour_um)

    # Choose overlap check function based on available index
    if spatial_tree is not None:
        _check = lambda s: _check_overlap_indexed(s, spatial_tree)
    else:
        _check = lambda s: _check_overlap_precomputed(s, precomputed_polygons)

    for attempt in range(max_attempts):
        current_offset = offset_um * (1.5 ** attempt)

        for direction_name, direction_vec in CONTROL_DIRECTIONS.items():
            offset_vec = direction_vec * current_offset
            shifted = contour_arr + offset_vec

            if not _check(shifted):
                return shifted.tolist(), direction_name, current_offset

    # Fallback: use first direction (E) at largest attempted offset.
    # This ensures every sample always gets a control.
    fallback_offset = offset_um * (1.5 ** max_attempts)
    fallback_vec = CONTROL_DIRECTIONS['E'] * fallback_offset
    shifted = contour_arr + fallback_vec
    return shifted.tolist(), 'E_fallback', fallback_offset


def generate_cluster_control(cluster_contours_um, precomputed_polygons,
                             offset_um=100.0, max_attempts=3,
                             spatial_tree=None):
    """
    Generate control for a cluster: shift ALL member contours by the SAME vector.

    Args:
        cluster_contours_um: List of contour arrays in um
        precomputed_polygons: List of precomputed Shapely Polygon objects (fallback)
        offset_um: Starting offset distance in um
        max_attempts: Number of times to increase offset if all directions fail
        spatial_tree: Optional STRtree for O(log N) overlap checks

    Returns (list_of_shifted_contours_um, direction_name, actual_offset_um).
    Always returns a result.
    """
    # Choose overlap check function based on available index
    if spatial_tree is not None:
        _check = lambda s: _check_overlap_indexed(s, spatial_tree)
    else:
        _check = lambda s: _check_overlap_precomputed(s, precomputed_polygons)

    for attempt in range(max_attempts):
        current_offset = offset_um * (1.5 ** attempt)

        for direction_name, direction_vec in CONTROL_DIRECTIONS.items():
            offset_vec = direction_vec * current_offset

            # Check all member contours for collisions
            any_overlap = False
            for contour_um in cluster_contours_um:
                shifted = np.array(contour_um) + offset_vec
                if _check(shifted):
                    any_overlap = True
                    break

            if not any_overlap:
                shifted_contours = [(np.array(c) + offset_vec).tolist()
                                    for c in cluster_contours_um]
                return shifted_contours, direction_name, current_offset

    # Fallback
    fallback_offset = offset_um * (1.5 ** max_attempts)
    fallback_vec = CONTROL_DIRECTIONS['E'] * fallback_offset
    shifted_contours = [(np.array(c) + fallback_vec).tolist()
                        for c in cluster_contours_um]
    return shifted_contours, 'E_fallback', fallback_offset


# ---------------------------------------------------------------------------
# Well assignment with controls
# ---------------------------------------------------------------------------

def assign_wells_with_controls(ordered_singles, ordered_single_ctrls,
                               ordered_clusters, ordered_cluster_ctrls):
    """
    Assign wells in serpentine order: alternating Target -> Control.

    Singles first, then clusters. Every sample has a control.
    Each single gets 2 wells (target, ctrl). Each cluster gets 2 wells (cluster, ctrl).

    Returns list of (shape_dict, well) tuples in well order.
    """
    # Total wells needed: 2 per single + 2 per cluster
    n_wells = 2 * len(ordered_singles) + 2 * len(ordered_clusters)
    wells = generate_wells_serpentine_4_quadrants(n_wells)

    assignments = []
    well_idx = 0

    # Singles: Target -> Control -> Target -> Control ...
    for single, ctrl in zip(ordered_singles, ordered_single_ctrls):
        single['well'] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
        assignments.append(single)
        well_idx += 1

        ctrl['well'] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
        assignments.append(ctrl)
        well_idx += 1

    # Clusters: Cluster -> Control -> Cluster -> Control ...
    for cluster, ctrl in zip(ordered_clusters, ordered_cluster_ctrls):
        cluster['well'] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
        assignments.append(cluster)
        well_idx += 1

        ctrl['well'] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
        assignments.append(ctrl)
        well_idx += 1

    return assignments, wells[:well_idx]


# ---------------------------------------------------------------------------
# Build unified export data
# ---------------------------------------------------------------------------

def build_export_data(assignments, well_order, metadata):
    """Build the unified export JSON structure."""
    shapes = []
    n_singles = n_single_controls = n_clusters = n_cluster_controls = 0
    n_detections_in_clusters = 0

    for item in assignments:
        shapes.append(item)
        t = item.get('type', '')
        if t == 'single':
            n_singles += 1
        elif t == 'single_control':
            n_single_controls += 1
        elif t == 'cluster':
            n_clusters += 1
            n_detections_in_clusters += item.get('n_members', item.get('n_nmjs', 0))
        elif t == 'cluster_control':
            n_cluster_controls += 1

    return {
        'metadata': metadata,
        'summary': {
            'n_singles': n_singles,
            'n_single_controls': n_single_controls,
            'n_clusters': n_clusters,
            'n_cluster_controls': n_cluster_controls,
            'n_detections_in_clusters': n_detections_in_clusters,
            'total_wells_used': len(well_order),
        },
        'shapes': shapes,
        'well_order': well_order,
    }


# ---------------------------------------------------------------------------
# LMD XML export
# ---------------------------------------------------------------------------

def export_to_lmd_xml(shapes, crosses_data, output_path, flip_y=True):
    """
    Export shapes to Leica LMD XML via py-lmd.

    Handles single contours (single, single_control) and
    multi-contour shapes (cluster, cluster_control).
    """
    from lmd.lib import Collection, Shape
    from lmd.tools import makeCross

    pixel_size = crosses_data['pixel_size_um']
    image_height_um = crosses_data['image_height_px'] * pixel_size

    # Calibration from first 3 crosses
    crosses = crosses_data['crosses']
    if len(crosses) < 3:
        raise ValueError("Need at least 3 reference crosses for calibration")

    calibration_points = np.array([
        [c['x_um'], c['y_um'] if not flip_y else image_height_um - c['y_um']]
        for c in crosses[:3]
    ])

    collection = Collection(calibration_points=calibration_points)

    # Add reference crosses
    for c in crosses:
        x_um = c['x_um']
        y_um = c['y_um'] if not flip_y else image_height_um - c['y_um']
        cross = makeCross(
            center=np.array([x_um, y_um]),
            arm_length=100, arm_width=10,
        )
        collection.add_shape(Shape(
            points=cross, well="CAL", name=f"RefCross_{c['id']}"
        ))

    # Add shapes
    for shape in shapes:
        well = shape.get('well', 'A1')
        shape_type = shape.get('type', 'single')

        # Collect contours for this shape
        contours_um = []
        if shape_type in ('cluster', 'cluster_control'):
            for c in shape.get('contours_um', []):
                if c and len(c) >= 3:
                    contours_um.append(np.array(c))
        else:
            c = shape.get('contour_um')
            if c and len(c) >= 3:
                contours_um.append(np.array(c))

        for idx, polygon_um in enumerate(contours_um):
            polygon_um = polygon_um.copy()
            if flip_y:
                polygon_um[:, 1] = image_height_um - polygon_um[:, 1]

            # Close polygon
            if not np.allclose(polygon_um[0], polygon_um[-1]):
                polygon_um = np.vstack([polygon_um, polygon_um[0]])

            uid = shape.get('uid', '')
            if shape_type == 'cluster':
                uids = shape.get('member_uids', [])
                name = uids[idx] if idx < len(uids) else f"{uid}_member{idx}"
            elif shape_type == 'cluster_control':
                name = f"{uid}_ctrl_member{idx}"
            else:
                name = uid

            collection.new_shape(polygon_um, well=well, name=name)

    output_path = Path(output_path)
    collection.save(str(output_path))
    print(f"  Exported LMD XML: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Cross placement HTML (unchanged from before)
# ---------------------------------------------------------------------------

def generate_cross_placement_html(detections, output_dir, pixel_size_um,
                                  image_width_px, image_height_px):
    """Generate HTML page for interactively placing reference crosses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_x, all_y = [], []
    for det in detections:
        coords = get_detection_coordinates(det)
        if coords:
            all_x.append(coords[0])
            all_y.append(coords[1])

    if not all_x:
        print("ERROR: No detection coordinates found")
        return None

    svg_width = 1200
    svg_height = int(svg_width * image_height_px / image_width_px)
    scale_x = svg_width / image_width_px
    scale_y = svg_height / image_height_px

    detection_circles = []
    for det in detections:
        coords = get_detection_coordinates(det)
        if not coords:
            continue
        svg_x = coords[0] * scale_x
        svg_y = coords[1] * scale_y
        detection_circles.append(
            f'<circle cx="{svg_x:.1f}" cy="{svg_y:.1f}" r="3" fill="lime" opacity="0.7"/>'
        )

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>LMD Reference Cross Placement</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; margin-bottom: 10px; }}
        .instructions {{ background: #16213e; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .instructions ul {{ margin: 10px 0; padding-left: 20px; }}
        .canvas-container {{ position: relative; border: 2px solid #333; border-radius: 8px;
                            overflow: hidden; background: #000; }}
        #mainSvg {{ display: block; cursor: crosshair; }}
        .cross-list {{ margin-top: 20px; background: #16213e; padding: 15px; border-radius: 8px; }}
        .cross-item {{ display: flex; justify-content: space-between; align-items: center;
                      padding: 8px; margin: 5px 0; background: #0f3460; border-radius: 4px; }}
        .cross-item button {{ background: #e94560; color: white; border: none;
                             padding: 5px 10px; border-radius: 4px; cursor: pointer; }}
        .buttons {{ margin-top: 20px; display: flex; gap: 10px; }}
        .btn {{ padding: 12px 24px; border: none; border-radius: 6px;
               cursor: pointer; font-size: 16px; font-weight: bold; }}
        .btn-primary {{ background: #00d4ff; color: #000; }}
        .btn-secondary {{ background: #333; color: #fff; }}
        .btn-success {{ background: #00ff88; color: #000; }}
        .info {{ margin-top: 15px; color: #888; font-size: 14px; }}
        .cross-marker {{ pointer-events: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LMD Reference Cross Placement</h1>
        <div class="instructions">
            <strong>Instructions:</strong>
            <ul>
                <li>Click on the image to place reference crosses (minimum 3 required)</li>
                <li>Crosses should be placed at identifiable landmarks visible under the LMD</li>
                <li>Green dots show detection locations for reference</li>
            </ul>
        </div>
        <div class="canvas-container">
            <svg id="mainSvg" width="{svg_width}" height="{svg_height}"
                 viewBox="0 0 {svg_width} {svg_height}">
                <rect width="100%" height="100%" fill="#111"/>
                <g id="detections">{''.join(detection_circles)}</g>
                <g id="crosses"></g>
            </svg>
        </div>
        <div class="cross-list">
            <h3>Reference Crosses: <span id="crossCount">0</span></h3>
            <div id="crossListItems"></div>
        </div>
        <div class="buttons">
            <button class="btn btn-secondary" onclick="clearCrosses()">Clear All</button>
            <button class="btn btn-primary" onclick="undoLast()">Undo Last</button>
            <button class="btn btn-success" onclick="saveCrosses()">Save Crosses</button>
        </div>
        <div class="info">
            <p>Image size: {image_width_px} x {image_height_px} px | Pixel size: {pixel_size_um:.4f} um/px</p>
            <p>Total detections shown: {len(detections)}</p>
        </div>
    </div>
    <script>
        const imageWidth = {image_width_px};
        const imageHeight = {image_height_px};
        const svgWidth = {svg_width};
        const svgHeight = {svg_height};
        const pixelSize = {pixel_size_um};
        let crosses = [];

        document.getElementById('mainSvg').addEventListener('click', function(e) {{
            const rect = this.getBoundingClientRect();
            const svgX = e.clientX - rect.left;
            const svgY = e.clientY - rect.top;
            const imgX = svgX / svgWidth * imageWidth;
            const imgY = svgY / svgHeight * imageHeight;
            const umX = imgX * pixelSize;
            const umY = imgY * pixelSize;
            addCross(imgX, imgY, umX, umY, svgX, svgY);
        }});

        function addCross(imgX, imgY, umX, umY, svgX, svgY) {{
            const id = crosses.length + 1;
            crosses.push({{ id, x_px: imgX, y_px: imgY, x_um: umX, y_um: umY }});
            const crossGroup = document.getElementById('crosses');
            const crossSize = 15;
            const cross = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            cross.setAttribute('id', 'cross_' + id);
            cross.setAttribute('class', 'cross-marker');
            cross.innerHTML = `
                <line x1="${{svgX - crossSize}}" y1="${{svgY}}" x2="${{svgX + crossSize}}" y2="${{svgY}}" stroke="red" stroke-width="3"/>
                <line x1="${{svgX}}" y1="${{svgY - crossSize}}" x2="${{svgX}}" y2="${{svgY + crossSize}}" stroke="red" stroke-width="3"/>
                <circle cx="${{svgX}}" cy="${{svgY}}" r="20" fill="none" stroke="red" stroke-width="2"/>
                <text x="${{svgX + 25}}" y="${{svgY + 5}}" fill="red" font-size="14" font-weight="bold">${{id}}</text>
            `;
            crossGroup.appendChild(cross);
            updateCrossList();
        }}

        function updateCrossList() {{
            document.getElementById('crossCount').textContent = crosses.length;
            const listDiv = document.getElementById('crossListItems');
            listDiv.innerHTML = crosses.map((c, i) => `
                <div class="cross-item">
                    <span>Cross ${{c.id}}: (${{c.x_px.toFixed(0)}}, ${{c.y_px.toFixed(0)}}) px</span>
                    <button onclick="removeCross(${{i}})">Remove</button>
                </div>
            `).join('');
        }}

        function removeCross(index) {{
            const cross = crosses[index];
            const elem = document.getElementById('cross_' + cross.id);
            if (elem) elem.remove();
            crosses.splice(index, 1);
            updateCrossList();
        }}

        function clearCrosses() {{
            document.getElementById('crosses').innerHTML = '';
            crosses = [];
            updateCrossList();
        }}

        function undoLast() {{
            if (crosses.length > 0) removeCross(crosses.length - 1);
        }}

        function saveCrosses() {{
            if (crosses.length < 3) {{ alert('Please place at least 3 reference crosses!'); return; }}
            const data = {{
                image_width_px: imageWidth, image_height_px: imageHeight,
                pixel_size_um: pixelSize,
                crosses: crosses.map(c => ({{
                    id: c.id, x_px: c.x_px, y_px: c.y_px, x_um: c.x_um, y_um: c.y_um
                }}))
            }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'reference_crosses.json'; a.click();
            URL.revokeObjectURL(url);
            alert('Saved ' + crosses.length + ' reference crosses!');
        }}
    </script>
</body>
</html>'''

    html_path = output_dir / 'place_crosses.html'
    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Generated cross placement HTML: {html_path}")
    return html_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Unified LMD Export - detections to Leica LMD format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full pipeline with clusters and controls
  python run_lmd_export.py \\
      --detections detections.json \\
      --cell-type nmj \\
      --crosses reference_crosses.json \\
      --clusters clusters.json \\
      --tiles-dir /path/to/tiles \\
      --output-dir lmd_export \\
      --export --generate-controls

  # Generate HTML for placing reference crosses
  python run_lmd_export.py \\
      --detections detections.json \\
      --output-dir lmd_export \\
      --generate-cross-html
''',
    )

    # Input files
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to detections JSON file')
    parser.add_argument('--cell-type', type=str, default=None,
                        help='Cell type (nmj, mk, vessel, mesothelium). '
                             'Auto-derives mask filename if --mask-filename not set.')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to annotations JSON (filters to positives only)')
    parser.add_argument('--crosses', type=str, default=None,
                        help='Path to reference crosses JSON')
    parser.add_argument('--clusters', type=str, default=None,
                        help='Path to clusters JSON from cluster_detections.py')

    # Contour extraction
    parser.add_argument('--tiles-dir', type=str, default=None,
                        help='Path to tiles/ directory with H5 masks')
    parser.add_argument('--mask-filename', type=str, default=None,
                        help='Mask filename within each tile dir (default: auto from --cell-type, or nmj_masks.h5)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for LMD files')
    parser.add_argument('--output-name', type=str, default='shapes',
                        help='Base name for output files')

    # Actions
    parser.add_argument('--generate-cross-html', action='store_true',
                        help='Generate HTML for placing reference crosses')
    parser.add_argument('--export', action='store_true',
                        help='Export to LMD XML (requires --crosses)')

    # Image metadata
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in um (auto-detect from detections if not set)')
    parser.add_argument('--image-width', type=int, default=None,
                        help='Image width in pixels')
    parser.add_argument('--image-height', type=int, default=None,
                        help='Image height in pixels')

    # Filtering
    parser.add_argument('--min-score', type=float, default=None,
                        help='Minimum rf_prediction score (filters detections)')

    # Zone filtering
    parser.add_argument('--zone-filter', type=str, default=None,
                        help='Include only these zone IDs (comma-separated, e.g. "1,3,5")')
    parser.add_argument('--zone-exclude', type=str, default=None,
                        help='Exclude these zone IDs (comma-separated, e.g. "0")')

    # Controls
    parser.add_argument('--generate-controls', action='store_true',
                        help='Generate spatial control regions for every target')
    parser.add_argument('--control-offset-um', type=float, default=100.0,
                        help='Offset distance for controls in um (default: 100)')

    # Contour processing
    parser.add_argument('--dilation-um', type=float, default=0.5,
                        help='Contour dilation in um (default: 0.5)')
    parser.add_argument('--rdp-epsilon', type=float, default=5.0,
                        help='RDP simplification epsilon in pixels (default: 5)')

    # Options
    parser.add_argument('--no-flip-y', action='store_true',
                        help='Do not flip Y axis for stage coordinates')

    args = parser.parse_args()

    # Auto-derive mask filename from cell type
    if args.mask_filename is None:
        if args.cell_type:
            args.mask_filename = f'{args.cell_type}_masks.h5'
        else:
            args.mask_filename = 'nmj_masks.h5'  # backward-compatible default

    # -----------------------------------------------------------------------
    # Load detections
    # -----------------------------------------------------------------------
    print(f"Loading detections from: {args.detections}")
    all_detections = load_detections(args.detections)
    print(f"  Loaded {len(all_detections)} detections")

    # Keep original list for cluster index lookups (cluster_detections.py
    # indices reference the ORIGINAL unfiltered list, since it filters internally)
    detections = list(all_detections)

    # Filter by annotations
    if args.annotations:
        print(f"Loading annotations from: {args.annotations}")
        positive_uids = load_annotations(args.annotations)
        print(f"  Found {len(positive_uids)} positive annotations")
        detections = filter_detections(detections, positive_uids=positive_uids)
        print(f"  Filtered to {len(detections)} positive detections")

    # Filter by score (skip if --clusters provided, clustering already filtered)
    if args.min_score is not None and not args.clusters:
        before = len(detections)
        detections = filter_detections(detections, min_score=args.min_score)
        print(f"  Score filter (>= {args.min_score}): {before} -> {len(detections)}")
    elif args.min_score is not None and args.clusters:
        print(f"  Score filter skipped (clustering already filtered at >= {args.min_score})")

    # Filter by zone (from assign_tissue_zones.py)
    if args.zone_filter:
        zone_ids = {int(z) for z in args.zone_filter.split(',')}
        before = len(detections)
        detections = [d for d in detections if d.get('zone_id') in zone_ids]
        print(f"  Zone filter (include {zone_ids}): {before} -> {len(detections)}")
    if args.zone_exclude:
        exclude_ids = {int(z) for z in args.zone_exclude.split(',')}
        before = len(detections)
        detections = [d for d in detections if d.get('zone_id') not in exclude_ids]
        print(f"  Zone exclude ({exclude_ids}): {before} -> {len(detections)}")

    if len(detections) == 0:
        print("ERROR: No detections to export!")
        return

    # Auto-detect metadata
    pixel_size = args.pixel_size
    if pixel_size is None:
        # Try to get pixel size from detection features
        for det in detections:
            feat = det.get('features', {})
            if 'pixel_size_um' in feat:
                pixel_size = feat['pixel_size_um']
                break
        if pixel_size is None:
            print("ERROR: pixel_size_um is required. Provide via --pixel-size or ensure "
                  "detections JSON contains pixel_size_um in detection features.")
            return
        print(f"  Pixel size (from detections): {pixel_size} um/px")
    else:
        print(f"  Pixel size (from CLI): {pixel_size} um/px")

    image_width = args.image_width
    image_height = args.image_height
    if image_width is None or image_height is None:
        max_x = max_y = 0
        for det in all_detections:
            coords = get_detection_coordinates(det)
            if coords:
                max_x = max(max_x, coords[0])
                max_y = max(max_y, coords[1])
        if max_x == 0 or max_y == 0:
            print("ERROR: Could not estimate image dimensions from detections. "
                  "Provide --image-width and --image-height.")
            return
        if image_width is None:
            image_width = int(max_x * 1.1)
        if image_height is None:
            image_height = int(max_y * 1.1)
        print(f"  Image size (estimated): {image_width} x {image_height} px")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Generate cross placement HTML
    # -----------------------------------------------------------------------
    if args.generate_cross_html:
        generate_cross_placement_html(
            detections, output_dir, pixel_size, image_width, image_height
        )

    # -----------------------------------------------------------------------
    # Export pipeline
    # -----------------------------------------------------------------------
    if args.export:
        if not args.crosses:
            print("ERROR: --crosses required for export. First use --generate-cross-html")
            return

        print(f"\nLoading crosses from: {args.crosses}")
        with open(args.crosses, 'r') as f:
            crosses_data = json.load(f)

        # Override metadata from CLI
        if args.pixel_size:
            crosses_data['pixel_size_um'] = args.pixel_size
        if args.image_width:
            crosses_data['image_width_px'] = args.image_width
        if args.image_height:
            crosses_data['image_height_px'] = args.image_height

        # -------------------------------------------------------------------
        # Step 1: Separate singles vs clustered detections
        # -------------------------------------------------------------------
        if args.clusters:
            print(f"\nLoading clusters from: {args.clusters}")
            cluster_data = load_clusters(args.clusters)

            # Cluster indices reference the ORIGINAL unfiltered detections list
            # (cluster_detections.py runs its own score filtering internally)
            outlier_indices = [o.get('detection_index', o.get('nmj_index'))
                               for o in cluster_data['outliers']]

            single_dets = [all_detections[i] for i in outlier_indices if i < len(all_detections)]
            cluster_groups = []
            for c in cluster_data['main_clusters']:
                member_indices = c.get('detection_indices', c.get('nmj_indices', []))
                members = [all_detections[i] for i in member_indices if i < len(all_detections)]
                if members:
                    cluster_groups.append({
                        'id': c['id'],
                        'members': members,
                        'cx': c.get('cx', 0),
                        'cy': c.get('cy', 0),
                    })

            print(f"  Singles: {len(single_dets)}")
            print(f"  Clusters: {len(cluster_groups)} "
                  f"({sum(len(cg['members']) for cg in cluster_groups)} detections)")
        else:
            # No clusters file: all detections are singles
            single_dets = detections
            cluster_groups = []
            print(f"  All {len(single_dets)} detections treated as singles (no --clusters)")

        # -------------------------------------------------------------------
        # Step 2: Extract contours from H5 masks if needed
        # -------------------------------------------------------------------
        all_dets_needing_contours = list(single_dets)
        for cg in cluster_groups:
            all_dets_needing_contours.extend(cg['members'])

        # Promote pipeline-processed contours: contour_dilated_um -> contour_um
        _promoted = 0
        for d in all_dets_needing_contours:
            if d.get('contour_um') is None and d.get('contour_dilated_um') is not None:
                d['contour_um'] = d['contour_dilated_um']
                _promoted += 1
        if _promoted:
            print(f"  Used {_promoted} pre-processed contours from pipeline (contour_dilated_um)")

        need_extraction = any(
            d.get('contour_um') is None and d.get('outer_contour_global') is None
            for d in all_dets_needing_contours
        )

        if need_extraction and args.tiles_dir:
            print(f"\nExtracting contours from H5 masks ({args.tiles_dir})...")
            contour_results = extract_contours_for_detections(
                all_dets_needing_contours, args.tiles_dir, pixel_size,
                mask_filename=args.mask_filename,
                dilation_um=args.dilation_um,
                rdp_epsilon=args.rdp_epsilon,
            )
            print(f"  Extracted {len(contour_results)} contours")

            # Attach contours to detections
            for det in all_dets_needing_contours:
                uid = det.get('uid', det.get('id', ''))
                if uid in contour_results:
                    det['contour_um'] = contour_results[uid]['contour_um']
                    det['area_um2'] = contour_results[uid]['area_um2']
        elif need_extraction and not args.tiles_dir:
            # Try to process existing outer_contour_global
            from segmentation.lmd.contour_processing import process_contour
            print("\nProcessing existing contours (dilation + RDP)...")
            processed_count = 0
            for det in all_dets_needing_contours:
                if det.get('contour_um') is not None:
                    continue
                contour_px = det.get('outer_contour_global')
                if contour_px is None:
                    continue
                processed, stats = process_contour(
                    contour_px, pixel_size_um=pixel_size,
                    dilation_um=args.dilation_um,
                    rdp_epsilon=args.rdp_epsilon,
                    return_stats=True,
                )
                if processed is not None:
                    det['contour_um'] = processed.tolist()
                    det['area_um2'] = stats['area_after_um2']
                    processed_count += 1
            print(f"  Processed {processed_count} contours")

        # -------------------------------------------------------------------
        # Step 3: Order singles by nearest-neighbor path
        # -------------------------------------------------------------------
        print("\nOrdering singles by nearest-neighbor path on slide...")
        singles_with_contours = []
        singles_positions = []
        for det in single_dets:
            if det.get('contour_um') is None:
                continue
            singles_with_contours.append(det)
            coords = get_detection_coordinates(det)
            if coords:
                singles_positions.append((coords[0], coords[1]))
            else:
                singles_positions.append((0, 0))

        if singles_positions:
            nn_order = nearest_neighbor_order(singles_positions)
            ordered_singles_dets = [singles_with_contours[i] for i in nn_order]
            last_pos = singles_positions[nn_order[-1]]
            total_dist = sum(
                np.linalg.norm(np.array(singles_positions[nn_order[i]]) -
                               np.array(singles_positions[nn_order[i+1]]))
                for i in range(len(nn_order) - 1)
            )
            print(f"  Ordered {len(ordered_singles_dets)} singles, "
                  f"path: {total_dist * pixel_size / 1000:.1f} mm")
        else:
            ordered_singles_dets = []
            last_pos = (0, 0)

        # -------------------------------------------------------------------
        # Step 4: Order clusters by nearest-neighbor (from last single)
        # -------------------------------------------------------------------
        print("Ordering clusters by nearest-neighbor path...")
        clusters_with_contours = []
        cluster_centroids = []

        for cg in cluster_groups:
            # Check all members have contours
            member_contours = []
            member_uids = []
            for m in cg['members']:
                contour = m.get('contour_um')
                if contour is not None:
                    member_contours.append(contour)
                    member_uids.append(m.get('uid', m.get('id', '')))

            if not member_contours:
                print(f"  WARNING: Cluster {cg['id']} has no contours, skipping entirely")
                continue

            dropped = len(cg['members']) - len(member_contours)
            if dropped > 0:
                print(f"  WARNING: Cluster {cg['id']} lost {dropped}/{len(cg['members'])} members (contour extraction failed)")

            clusters_with_contours.append({
                'id': cg['id'],
                'members': cg['members'],
                'member_contours_um': member_contours,
                'member_uids': member_uids,
                'cx': cg['cx'],
                'cy': cg['cy'],
            })
            cluster_centroids.append((cg['cx'], cg['cy']))

        if cluster_centroids:
            # Start from nearest cluster to last single position
            dists = [np.linalg.norm(np.array(cc) - np.array(last_pos))
                     for cc in cluster_centroids]
            start_cluster = int(np.argmin(dists))
            cluster_order = nearest_neighbor_order(cluster_centroids, start_idx=start_cluster)
            ordered_clusters_data = [clusters_with_contours[i] for i in cluster_order]
            print(f"  Ordered {len(ordered_clusters_data)} clusters")
        else:
            ordered_clusters_data = []

        # -------------------------------------------------------------------
        # Well capacity check (before expensive control generation)
        # -------------------------------------------------------------------
        MAX_WELLS = 308  # 4 quadrants x 77 wells each on 384-well plate
        n_items = len(ordered_singles_dets) + len(ordered_clusters_data)
        n_wells_needed = n_items * 2 if args.generate_controls else n_items
        if n_wells_needed > MAX_WELLS:
            print(f"\n{'='*70}")
            print(f"WELL CAPACITY EXCEEDED")
            print(f"{'='*70}")
            print(f"  Singles:  {len(ordered_singles_dets)}")
            print(f"  Clusters: {len(ordered_clusters_data)}")
            print(f"  Controls: {'yes (x2)' if args.generate_controls else 'no'}")
            print(f"  Wells needed: {n_wells_needed}")
            print(f"  Wells available: {MAX_WELLS} (384-well plate, 4 quadrants)")
            print(f"  Overflow: {n_wells_needed - MAX_WELLS} wells")
            print(f"\nOptions:")
            print(f"  1. Increase --min-score to reduce detections")
            print(f"  2. Split detections across multiple plates")
            print(f"  3. Run without --generate-controls (halves well usage)")
            print(f"{'='*70}")
            return

        # -------------------------------------------------------------------
        # Step 5: Generate controls
        # -------------------------------------------------------------------
        if args.generate_controls:
            print(f"\nGenerating controls (offset: {args.control_offset_um} um)...")
            print("  Every sample will have a control.")

            # Precompute Shapely polygons for all detection contours (avoids
            # recreating Polygon objects on every overlap check)
            precomputed_polygons = []
            for det in ordered_singles_dets:
                c = det.get('contour_um')
                if c and len(c) >= 3:
                    precomputed_polygons.append(_make_polygon(c))

            for cdata in ordered_clusters_data:
                for c in cdata['member_contours_um']:
                    if c and len(c) >= 3:
                        precomputed_polygons.append(_make_polygon(c))

            print(f"  Precomputed {len(precomputed_polygons)} collision polygons")

            # Build spatial index for O(log N) overlap checks.
            # STRtree is static (built once from all detection polygons).
            # New controls are NOT added to the index — this is acceptable
            # since controls are offset in different directions and unlikely
            # to overlap each other.
            detection_tree = _build_spatial_index(precomputed_polygons)
            if detection_tree is not None:
                print("  Using STRtree spatial index for overlap checks")
            else:
                print("  STRtree unavailable, using linear overlap scan (install shapely>=2.0 for speedup)")

            # Generate single controls
            ordered_single_ctrls = []
            fallback_count = 0
            for det in ordered_singles_dets:
                contour_um = det.get('contour_um')
                shifted, direction, actual_offset = generate_spatial_control(
                    contour_um, precomputed_polygons,
                    offset_um=args.control_offset_um,
                    spatial_tree=detection_tree,
                )
                if 'fallback' in direction:
                    fallback_count += 1

                uid = det.get('uid', det.get('id', ''))
                ordered_single_ctrls.append({
                    'type': 'single_control',
                    'uid': uid + '_ctrl',
                    'control_of': uid,
                    'contour_um': shifted,
                    'offset_direction': direction,
                    'offset_um': actual_offset,
                    'area_um2': det.get('area_um2', 0),
                })

            print(f"  Single controls: {len(ordered_single_ctrls)} "
                  f"({fallback_count} used fallback offset)")

            # Generate cluster controls
            ordered_cluster_ctrls = []
            cluster_fallback = 0
            for cdata in ordered_clusters_data:
                shifted_contours, direction, actual_offset = generate_cluster_control(
                    cdata['member_contours_um'], precomputed_polygons,
                    offset_um=args.control_offset_um,
                    spatial_tree=detection_tree,
                )
                if 'fallback' in direction:
                    cluster_fallback += 1

                ordered_cluster_ctrls.append({
                    'type': 'cluster_control',
                    'uid': f"cluster_{cdata['id']}_ctrl",
                    'control_of_cluster': cdata['id'],
                    'contours_um': shifted_contours,
                    'offset_direction': direction,
                    'offset_um': actual_offset,
                })

            print(f"  Cluster controls: {len(ordered_cluster_ctrls)} "
                  f"({cluster_fallback} used fallback offset)")
        else:
            ordered_single_ctrls = []
            ordered_cluster_ctrls = []

        # -------------------------------------------------------------------
        # Step 6: Build shape dicts for export
        # -------------------------------------------------------------------
        # Convert singles to shape dicts
        ordered_singles = []
        for det in ordered_singles_dets:
            uid = det.get('uid', det.get('id', ''))
            ordered_singles.append({
                'type': 'single',
                'uid': uid,
                'contour_um': det.get('contour_um'),
                'area_um2': det.get('area_um2', 0),
                'global_center': det.get('global_center'),
            })

        # Convert clusters to shape dicts
        ordered_clusters = []
        for cdata in ordered_clusters_data:
            total_area = sum(
                m.get('area_um2', 0) for m in cdata['members']
                if m.get('contour_um') is not None
            )
            ordered_clusters.append({
                'type': 'cluster',
                'uid': f"cluster_{cdata['id']}",
                'cluster_id': cdata['id'],
                'n_members': len(cdata['member_contours_um']),
                'contours_um': cdata['member_contours_um'],
                'member_uids': cdata['member_uids'],
                'total_area_um2': total_area,
                'cx': cdata['cx'],
                'cy': cdata['cy'],
            })

        # -------------------------------------------------------------------
        # Step 7: Assign wells
        # -------------------------------------------------------------------
        try:
            if args.generate_controls:
                print("\nAssigning wells (serpentine, B2->B3->C3->C2, alternating target/control)...")
                assignments, well_order = assign_wells_with_controls(
                    ordered_singles, ordered_single_ctrls,
                    ordered_clusters, ordered_cluster_ctrls,
                )
            else:
                # No controls: just assign sequentially
                all_shapes = ordered_singles + ordered_clusters
                n_wells = len(all_shapes)
                well_order = generate_wells_serpentine_4_quadrants(n_wells)
                for i, shape in enumerate(all_shapes):
                    shape['well'] = well_order[i] if i < len(well_order) else f"overflow_{i}"
                assignments = all_shapes
        except ValueError as e:
            # Well overflow — save partial results (shapes without well assignments)
            print(f"\nERROR: {e}")
            partial_shapes = ordered_singles + ordered_single_ctrls + ordered_clusters + ordered_cluster_ctrls
            partial_path = output_dir / f"{args.output_name}_NO_WELLS.json"
            partial_data = {
                'metadata': {
                    'cell_type': args.cell_type or 'unknown',
                    'error': str(e),
                    'n_singles': len(ordered_singles),
                    'n_clusters': len(ordered_clusters),
                    'controls': args.generate_controls,
                },
                'shapes': partial_shapes,
            }
            with open(partial_path, 'w') as f:
                json.dump(partial_data, f)
            print(f"  Partial results saved to: {partial_path}")
            print(f"  (shapes without well assignments — prune list and re-run)")
            return

        print(f"  Total wells used: {len(well_order)}")
        if well_order:
            print(f"  First well: {well_order[0]}, Last well: {well_order[-1]}")

        # -------------------------------------------------------------------
        # Step 8: Build and save export data
        # -------------------------------------------------------------------
        metadata = {
            'cell_type': args.cell_type or 'unknown',
            'plate_format': '384',
            'quadrant_order': ['B2', 'B3', 'C3', 'C2'],
            'pixel_size_um': pixel_size,
            'dilation_um': args.dilation_um,
            'rdp_epsilon_px': args.rdp_epsilon,
            'control_offset_um': args.control_offset_um,
        }

        export_data = build_export_data(assignments, well_order, metadata)

        # Save JSON (timestamped + symlink)
        from segmentation.utils.timestamps import timestamped_path, update_symlink
        json_path = output_dir / f"{args.output_name}_with_controls.json"
        ts_json = timestamped_path(json_path)
        with open(ts_json, 'w') as f:
            json.dump(export_data, f)
        update_symlink(json_path, ts_json)
        print(f"\n  Saved export JSON: {ts_json}")

        # Save CSV summary (timestamped + symlink)
        csv_path = output_dir / f"{args.output_name}_summary.csv"
        ts_csv = timestamped_path(csv_path)
        with open(ts_csv, 'w') as f:
            f.write('well,type,uid,area_um2,n_contours,offset_direction\n')
            for shape in assignments:
                well = shape.get('well', '')
                stype = shape.get('type', '')
                uid = shape.get('uid', '')
                area = shape.get('area_um2', shape.get('total_area_um2', 0))
                n_contours = len(shape.get('contours_um', [])) or (1 if shape.get('contour_um') else 0)
                direction = shape.get('offset_direction', '')
                f.write(f"{well},{stype},{uid},{area:.2f},{n_contours},{direction}\n")
        update_symlink(csv_path, ts_csv)
        print(f"  Saved CSV summary: {ts_csv}")

        # Export LMD XML (timestamped + symlink)
        xml_path = output_dir / f"{args.output_name}.xml"
        ts_xml = timestamped_path(xml_path)
        try:
            export_to_lmd_xml(
                assignments, crosses_data, ts_xml,
                flip_y=not args.no_flip_y,
            )
            update_symlink(xml_path, ts_xml)
            print(f"  Saved LMD XML: {ts_xml}")
        except ImportError:
            print("  WARNING: py-lmd not installed, skipping XML export.")
            print("  Install with: pip install py-lmd")
        except Exception as e:
            print(f"  WARNING: XML export failed: {e}")

        # Print summary
        s = export_data['summary']
        print(f"\n{'='*60}")
        print(f"EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"  Singles:          {s['n_singles']}")
        print(f"  Single controls:  {s['n_single_controls']}")
        print(f"  Clusters:         {s['n_clusters']}")
        print(f"  Cluster controls: {s['n_cluster_controls']}")
        print(f"  Detections in clusters: {s['n_detections_in_clusters']}")
        print(f"  Total wells used: {s['total_wells_used']}")
        print(f"{'='*60}")

    if not args.generate_cross_html and not args.export:
        print("No action specified. Use --generate-cross-html or --export")


if __name__ == '__main__':
    main()
