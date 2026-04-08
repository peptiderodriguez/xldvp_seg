"""
Pure-logic functions for LMD (Laser Microdissection) export.

Handles loading, filtering, contour extraction, spatial control generation,
well assignment, XML export, and cross-placement HTML generation.

Promoted from ``run_lmd_export.py`` so they are importable as a library.
The top-level script retains only CLI orchestration (main, _run_single_slide,
run_batch_export).
"""

from pathlib import Path

import numpy as np

from xldvp_seg.exceptions import DataLoadError, ExportError
from xldvp_seg.lmd.well_plate import generate_plate_wells
from xldvp_seg.utils.json_utils import fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_detections(detections_path):
    """Load detections from JSON file."""
    return fast_json_load(str(detections_path))


def load_annotations(annotations_path):
    """
    Load annotations and return set of positive UIDs.

    Supports formats:
    - {"positive": [...], "negative": [...]}
    - {"annotations": {"uid": "yes/no", ...}}
    - Plain list of UIDs
    """
    data = fast_json_load(str(annotations_path))

    positive_uids = set()

    if "positive" in data:
        positive_uids.update(data["positive"])
    elif "annotations" in data:
        for uid, label in data["annotations"].items():
            if label.lower() in ("yes", "positive", "true", "1"):
                positive_uids.add(uid)
    elif isinstance(data, list):
        positive_uids.update(data)

    return positive_uids


def filter_detections(detections, positive_uids=None, min_score=None):
    """Filter detections by annotations and/or score."""
    filtered = []
    for det in detections:
        uid = det.get("uid", det.get("id", ""))

        if positive_uids is not None and uid not in positive_uids:
            continue

        if min_score is not None:
            score = det.get("rf_prediction", det.get("score", 0))
            if score is None:
                score = 0
            if score < min_score:
                continue

        filtered.append(det)
    return filtered


def load_clusters(clusters_path):
    """Load and validate clusters JSON from cluster_detections.py."""
    data = fast_json_load(str(clusters_path))

    if "main_clusters" not in data or "outliers" not in data:
        raise DataLoadError("Invalid clusters file: missing 'main_clusters' or 'outliers' keys")

    return data


def get_detection_coordinates(det):
    """Extract (x, y) global pixel coordinates from detection.

    Uses ``global_center`` (slide-level coordinates) only.
    Never falls back to ``center`` which is tile-local.
    """
    if "global_center" in det:
        return det["global_center"]
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


def extract_contours_for_detections(
    detections,
    tiles_dir,
    pixel_size,
    mask_filename="nmj_masks.h5",
    dilation_um=0.5,
    rdp_epsilon=5.0,
    max_area_change_pct=None,
    max_dilation_area_pct=None,
):
    """
    Extract and process contours for detections from H5 mask files.

    Skips detections that already have 'contour_um' set.

    Returns dict mapping uid -> contour data.
    """
    import h5py
    import hdf5plugin  # noqa: F401 - must import before h5py for LZ4

    from xldvp_seg.lmd.contour_processing import process_contour

    tiles_dir = Path(tiles_dir)

    # Group by tile
    by_tile = {}
    for det in detections:
        # Skip if contour already exists
        if det.get("contour_um") is not None:
            continue

        tile_origin = det.get("tile_origin", [0, 0])
        tile_name = f"tile_{tile_origin[0]}_{tile_origin[1]}"
        if tile_name not in by_tile:
            by_tile[tile_name] = []
        by_tile[tile_name].append(det)

    if not by_tile:
        logger.info("All detections already have contours, skipping extraction.")
        return {}

    results = {}
    for tile_idx, (tile_name, tile_dets) in enumerate(by_tile.items()):
        if (tile_idx + 1) % 20 == 0:
            logger.info("Tile %d/%d...", tile_idx + 1, len(by_tile))

        mask_path = tiles_dir / tile_name / mask_filename
        if not mask_path.exists():
            continue

        with h5py.File(mask_path, "r") as hf:
            masks = hf["masks"][:]

        for det in tile_dets:
            uid = det.get("uid", det.get("id", ""))
            label = det.get("mask_label")
            if label is None:
                # Fallback for single-GPU path: parse from id (e.g., "nmj_3" -> 3)
                det_id = det.get("id", "")
                try:
                    label = int(det_id.split("_")[-1])
                except (ValueError, IndexError):
                    continue

            contour_local = extract_contour_from_mask(masks, label)
            if contour_local is None:
                continue

            # Convert to global coordinates
            tile_origin = det.get("tile_origin", [0, 0])
            contour_global = contour_local.astype(float)
            contour_global[:, 0] += tile_origin[0]
            contour_global[:, 1] += tile_origin[1]

            # Apply post-processing (adaptive RDP + adaptive dilation)
            processed, stats = process_contour(
                contour_global.tolist(),
                pixel_size_um=pixel_size,
                dilation_um=dilation_um,
                rdp_epsilon=rdp_epsilon,
                max_area_change_pct=max_area_change_pct,
                max_dilation_area_pct=max_dilation_area_pct,
                return_stats=True,
            )

            if processed is None:
                continue

            results[uid] = {
                "contour_global_px": contour_global.tolist(),
                "contour_um": processed.tolist(),
                "area_um2": stats["area_after_um2"],
                "n_points": stats["points_after"],
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
# Spatial control generation
# ---------------------------------------------------------------------------

# 8 cardinal directions
CONTROL_DIRECTIONS = {
    "E": np.array([1, 0]),
    "NE": np.array([1, -1]) / np.sqrt(2),
    "N": np.array([0, -1]),
    "NW": np.array([-1, -1]) / np.sqrt(2),
    "W": np.array([-1, 0]),
    "SW": np.array([-1, 1]) / np.sqrt(2),
    "S": np.array([0, 1]),
    "SE": np.array([1, 1]) / np.sqrt(2),
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
        logger.debug("Polygon construction failed for contour", exc_info=True)
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
    return len(spatial_tree.query(candidate_poly, predicate="intersects")) > 0


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
            logger.debug("Intersection check failed", exc_info=True)
            return True
    return False


def generate_spatial_control(
    contour_um, precomputed_polygons, offset_um=100.0, max_attempts=3, spatial_tree=None
):
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
        _check = lambda s: _check_overlap_indexed(s, spatial_tree)  # noqa: E731
    else:
        _check = lambda s: _check_overlap_precomputed(s, precomputed_polygons)  # noqa: E731

    for attempt in range(max_attempts):
        current_offset = offset_um * (1.5**attempt)

        for direction_name, direction_vec in CONTROL_DIRECTIONS.items():
            offset_vec = direction_vec * current_offset
            shifted = contour_arr + offset_vec

            if not _check(shifted):
                return shifted.tolist(), direction_name, current_offset

    # Fallback: use first direction (E) at largest attempted offset.
    # This ensures every sample always gets a control.
    fallback_offset = offset_um * (1.5**max_attempts)
    fallback_vec = CONTROL_DIRECTIONS["E"] * fallback_offset
    shifted = contour_arr + fallback_vec
    return shifted.tolist(), "E_fallback", fallback_offset


def generate_cluster_control(
    cluster_contours_um, precomputed_polygons, offset_um=100.0, max_attempts=3, spatial_tree=None
):
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
        _check = lambda s: _check_overlap_indexed(s, spatial_tree)  # noqa: E731
    else:
        _check = lambda s: _check_overlap_precomputed(s, precomputed_polygons)  # noqa: E731

    for attempt in range(max_attempts):
        current_offset = offset_um * (1.5**attempt)

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
                shifted_contours = [
                    (np.array(c) + offset_vec).tolist() for c in cluster_contours_um
                ]
                return shifted_contours, direction_name, current_offset

    # Fallback
    fallback_offset = offset_um * (1.5**max_attempts)
    fallback_vec = CONTROL_DIRECTIONS["E"] * fallback_offset
    shifted_contours = [(np.array(c) + fallback_vec).tolist() for c in cluster_contours_um]
    return shifted_contours, "E_fallback", fallback_offset


# ---------------------------------------------------------------------------
# Well assignment with controls
# ---------------------------------------------------------------------------


def assign_wells_with_controls(
    ordered_singles, ordered_single_ctrls, ordered_clusters, ordered_cluster_ctrls
):
    """
    Assign wells in serpentine order: alternating Target -> Control.

    Singles first, then clusters. Every sample has a control.
    Each single gets 2 wells (target, ctrl). Each cluster gets 2 wells (cluster, ctrl).

    Returns list of (shape_dict, well) tuples in well order.
    """
    # Total wells needed: 2 per single + 2 per cluster
    n_wells = 2 * len(ordered_singles) + 2 * len(ordered_clusters)
    wells = generate_plate_wells(n_wells)

    assignments = []
    well_idx = 0

    # Singles: Target -> Control -> Target -> Control ...
    for single, ctrl in zip(ordered_singles, ordered_single_ctrls):
        single["well"] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
        assignments.append(single)
        well_idx += 1

        ctrl["well"] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
        assignments.append(ctrl)
        well_idx += 1

    # Clusters: Cluster -> Control -> Cluster -> Control ...
    for cluster, ctrl in zip(ordered_clusters, ordered_cluster_ctrls):
        cluster["well"] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
        assignments.append(cluster)
        well_idx += 1

        ctrl["well"] = wells[well_idx] if well_idx < len(wells) else f"overflow_{well_idx}"
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
        t = item.get("type", "")
        if t == "single":
            n_singles += 1
        elif t == "single_control":
            n_single_controls += 1
        elif t == "cluster":
            n_clusters += 1
            n_detections_in_clusters += item.get("n_members", item.get("n_nmjs", 0))
        elif t == "cluster_control":
            n_cluster_controls += 1

    return {
        "metadata": metadata,
        "summary": {
            "n_singles": n_singles,
            "n_single_controls": n_single_controls,
            "n_clusters": n_clusters,
            "n_cluster_controls": n_cluster_controls,
            "n_detections_in_clusters": n_detections_in_clusters,
            "total_wells_used": len(well_order),
        },
        "shapes": shapes,
        "well_order": well_order,
    }


# ---------------------------------------------------------------------------
# LMD XML export
# ---------------------------------------------------------------------------

from xldvp_seg.lmd.contour_processing import (
    transform_native_to_display as _transform_native_to_display,
)


def export_to_lmd_xml(shapes, crosses_data, output_path, flip_y=True):
    """
    Export shapes to Leica LMD XML via py-lmd.

    Handles single contours (single, single_control) and
    multi-contour shapes (cluster, cluster_control).

    Reads display_transform from crosses_data to apply the same coordinate
    transforms (flip_horizontal, rotate_cw_90) to contours so they match
    the cross calibration space.
    """
    from lmd.lib import Collection

    pixel_size = crosses_data["pixel_size_um"]
    orig_w_um = crosses_data["image_width_px"] * pixel_size
    orig_h_um = crosses_data["image_height_px"] * pixel_size

    # Read display transforms from crosses metadata
    dt = crosses_data.get("display_transform", {})
    flip_h = dt.get("flip_horizontal", False)
    rot90 = dt.get("rotate_cw_90", False)

    # After rotation, display dimensions swap
    display_h_um = orig_w_um if rot90 else orig_h_um

    if flip_h or rot90:
        logger.info("Display transforms: flip_h=%s, rot90=%s", flip_h, rot90)
        logger.info("Display height for LMD Y-flip: %.0f um", display_h_um)

    # Calibration from first 3 crosses
    crosses = crosses_data["crosses"]
    if len(crosses) < 3:
        raise ExportError("Need at least 3 reference crosses for calibration")

    # Crosses are already in display space; Y-flip for LMD convention
    calibration_points = np.array(
        [[c["x_um"], c["y_um"] if not flip_y else display_h_um - c["y_um"]] for c in crosses[:3]]
    )

    collection = Collection(calibration_points=calibration_points)
    # Note: calibration points in the XML header are sufficient for LMD.
    # Visual cross shapes are NOT added — they confuse the LMD software.

    # Add shapes — transform contours from native CZI space to display space
    for shape in shapes:
        well = shape.get("well", "A1")
        shape_type = shape.get("type", "single")

        # Collect contours for this shape
        contours_um = []
        if shape_type in ("cluster", "cluster_control"):
            for c in shape.get("contours_um", []):
                if c and len(c) >= 3:
                    contours_um.append(np.array(c))
        else:
            c = shape.get("contour_um")
            if c and len(c) >= 3:
                contours_um.append(np.array(c))

        for idx, polygon_um in enumerate(contours_um):
            # Apply display transforms to match cross coordinate space
            polygon_um = _transform_native_to_display(
                np.array(polygon_um), orig_w_um, orig_h_um, flip_h, rot90
            )
            if flip_y:
                polygon_um[:, 1] = display_h_um - polygon_um[:, 1]

            # Close polygon
            if not np.allclose(polygon_um[0], polygon_um[-1]):
                polygon_um = np.vstack([polygon_um, polygon_um[0]])

            uid = shape.get("uid", "")
            if shape_type == "cluster":
                uids = shape.get("member_uids", [])
                name = uids[idx] if idx < len(uids) else f"{uid}_member{idx}"
            elif shape_type == "cluster_control":
                name = f"{uid}_ctrl_member{idx}"
            else:
                name = uid

            collection.new_shape(polygon_um, well=well, name=name)

    output_path = Path(output_path)
    collection.save(str(output_path))
    logger.info("Exported LMD XML: %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# Cross placement HTML
# ---------------------------------------------------------------------------


def generate_cross_placement_html(
    detections, output_dir, pixel_size_um, image_width_px, image_height_px
):
    """Generate HTML page for interactively placing reference crosses."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for val in [pixel_size_um, image_width_px, image_height_px]:
        if not np.isfinite(val):
            raise ExportError(f"Non-finite value in cross placement: {val}")

    all_x, all_y = [], []
    for det in detections:
        coords = get_detection_coordinates(det)
        if coords:
            all_x.append(coords[0])
            all_y.append(coords[1])

    if not all_x:
        logger.error("No detection coordinates found")
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

    html_content = f"""<!DOCTYPE html>
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
</html>"""

    html_path = output_dir / "place_crosses.html"
    with open(html_path, "w") as f:
        f.write(html_content)

    logger.info("Generated cross placement HTML: %s", html_path)
    return html_path
