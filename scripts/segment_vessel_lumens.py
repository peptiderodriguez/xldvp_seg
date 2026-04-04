#!/usr/bin/env python3
"""Lumen-first vessel detection using SAM2 auto-mask on OME-Zarr tiles.

Multi-scale approach: reads pre-built OME-Zarr pyramids (flat-field corrected),
runs SAM2 automatic mask generation per tile, filters for dark interior + high
boundary contrast, deduplicates across tiles and scales, then validates each
candidate lumen by proximity of marker-positive cells (e.g. SMA+, CD31+).

Architecture:
    OME-Zarr (multi-resolution pyramid)
      -> Multi-scale tiling (zarr levels 1-4 = 2x, 4x, 8x, 16x)
      -> SAM2 auto-mask per tile -> candidate lumens
      -> Filter: size bounds + local contrast ratio + interior darkness
      -> Cross-tile IoU dedup
      -> Cross-scale merge (keep highest-resolution)
      -> Biological validation: marker+ cells near lumen boundary
      -> Vessel characterization (via xldvp_seg.analysis.vessel_characterization)
      -> Output: vessel_lumens.json, cell_detections_vessels.json, vessel_summary.csv

Example:
    python scripts/segment_vessel_lumens.py \\
        --zarr-path /path/to/slide.zarr \\
        --detections /path/to/cell_detections_classified.json \\
        --marker-names SMA,CD31 \\
        --rgb-channels 1,3,0 \\
        --output-dir /path/to/output/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Percentile normalization
# ---------------------------------------------------------------------------


def percentile_normalize_uint8(arr: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    """Normalize uint16 array to uint8 using percentile stretch.

    Excludes zero pixels (CZI padding) from percentile computation. Returns
    a uint8 array suitable for SAM2 RGB input.

    Args:
        arr: Input array (typically uint16).
        low: Lower percentile for clipping.
        high: Upper percentile for clipping.

    Returns:
        uint8 array with values in [0, 255].
    """
    if arr.size == 0:
        return arr.astype(np.uint8)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo = float(np.percentile(nonzero, low))
    hi = float(np.percentile(nonzero, high))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip((arr.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255)
    return clipped.astype(np.uint8)


# ---------------------------------------------------------------------------
# Multi-scale tiling
# ---------------------------------------------------------------------------


def generate_tiles(
    level_shape: tuple[int, ...], tile_size: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """Generate (y_start, x_start, h, w) tiles covering a zarr level.

    Args:
        level_shape: Shape as (C, H, W).
        tile_size: Tile side length in pixels.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        List of (y, x, h, w) tuples.
    """
    _, H, W = level_shape
    step = tile_size - overlap
    if step <= 0:
        step = tile_size
    tiles = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            h = min(tile_size, H - y)
            w = min(tile_size, W - x)
            # Skip tiny edge tiles that are mostly overlap
            if h > overlap and w > overlap:
                tiles.append((y, x, h, w))
    return tiles


# ---------------------------------------------------------------------------
# Multi-scale zarr tile reading (reusable)
# ---------------------------------------------------------------------------


def resolve_zarr_scales(
    zarr_root: Any,
    scales: list[int],
) -> list[tuple[int, int, str, int]]:
    """Resolve requested scales against available zarr pyramid levels.

    For scales that exceed the zarr pyramid (e.g., 64x when zarr only has
    levels 0–4 = up to 16x), maps to the coarsest available level plus an
    extra downsample factor applied in-script.

    Args:
        zarr_root: Opened zarr group with numeric level keys ('0', '1', ...).
        scales: List of scale factors (must be powers of 2).

    Returns:
        List of (scale, zarr_level, level_key, extra_downsample) tuples.

    Raises:
        ValueError: If any scale is not a power of 2.
    """
    # Validate power-of-2
    for s in scales:
        if s < 1 or (s & (s - 1)) != 0:
            raise ValueError(
                f"Scale {s} is not a power of 2. Scales must be powers of 2 "
                f"(e.g., 2, 4, 8, 16, 32, 64)."
            )

    max_zarr_level = max(int(k) for k in zarr_root.keys() if k.isdigit())
    resolved = []

    for s in scales:
        level = int(np.log2(s))
        level_key = str(level)
        if level_key in zarr_root:
            resolved.append((s, level, level_key, 1))
        elif level > max_zarr_level:
            extra_ds = 2 ** (level - max_zarr_level)
            coarse_key = str(max_zarr_level)
            logger.info(
                "Scale %dx: zarr level %d not found, will read level %d + %dx downsample",
                s,
                level,
                max_zarr_level,
                extra_ds,
            )
            resolved.append((s, max_zarr_level, coarse_key, extra_ds))
        else:
            logger.warning("Zarr level %d (scale %dx) not found, skipping", level, s)

    return resolved


def get_effective_shape(
    level_data: Any,
    extra_ds: int,
) -> tuple[int, int, int]:
    """Get the effective (C, H, W) shape after optional extra downsampling.

    Args:
        level_data: Zarr array with shape (C, H, W).
        extra_ds: Extra downsample factor (1 = no downsampling).

    Returns:
        (C, H, W) tuple representing the shape SAM2 will see.
    """
    c, h, w = level_data.shape
    if extra_ds > 1:
        return (c, h // extra_ds, w // extra_ds)
    return (c, h, w)


def read_zarr_tile(
    level_data: Any,
    channel: int,
    tile_y: int,
    tile_x: int,
    tile_h: int,
    tile_w: int,
    extra_ds: int = 1,
) -> np.ndarray:
    """Read a single-channel tile from zarr with optional extra downsampling.

    When extra_ds > 1, reads the corresponding larger region from the zarr
    level and downsamples with cv2.INTER_AREA (anti-aliased averaging).

    Args:
        level_data: Zarr array with shape (C, H, W).
        channel: Channel index to read.
        tile_y: Tile y-offset in the effective (downsampled) coordinate space.
        tile_x: Tile x-offset in the effective (downsampled) coordinate space.
        tile_h: Tile height in the effective coordinate space.
        tile_w: Tile width in the effective coordinate space.
        extra_ds: Extra downsample factor (1 = direct read, no downsampling).

    Returns:
        2D numpy array of shape (tile_h, tile_w), same dtype as zarr source
        (or resized via INTER_AREA if extra_ds > 1).
    """
    if extra_ds > 1:
        import cv2

        # Map effective coords back to zarr-level coords
        zy = tile_y * extra_ds
        zx = tile_x * extra_ds
        zh = min(tile_h * extra_ds, level_data.shape[1] - zy)
        zw = min(tile_w * extra_ds, level_data.shape[2] - zx)
        raw = np.array(level_data[channel, zy : zy + zh, zx : zx + zw])
        # Compute actual output size (may differ from tile_h/tile_w at edges
        # due to rounding in integer division)
        out_h = min(tile_h, int(np.ceil(zh / extra_ds)))
        out_w = min(tile_w, int(np.ceil(zw / extra_ds)))
        return cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        return np.array(level_data[channel, tile_y : tile_y + tile_h, tile_x : tile_x + tile_w])


# ---------------------------------------------------------------------------
# Per-tile SAM2 lumen detection
# ---------------------------------------------------------------------------


def detect_lumens_in_tile(
    tile_rgb_uint8: np.ndarray,
    pixel_size_um: float,
    min_area_um2: float,
    min_contrast: float,
    sam2_auto: Any,
) -> list[dict]:
    """Run SAM2 auto-mask on a tile and filter for lumen candidates.

    Filters applied:
      1. Size bounds (min area, <80% of tile).
      2. Interior darkness: median interior intensity < tile tissue median.
      3. Local contrast: boundary ring median / interior median >= min_contrast.

    Args:
        tile_rgb_uint8: (H, W, 3) uint8 RGB tile.
        pixel_size_um: Pixel size at this scale level.
        min_area_um2: Minimum lumen area in um^2.
        min_contrast: Minimum boundary/interior contrast ratio.
        sam2_auto: SAM2AutomaticMaskGenerator instance.

    Returns:
        List of lumen candidate dicts with tile-local contours and metrics.
    """
    import cv2

    results = sam2_auto.generate(tile_rgb_uint8)

    # Compute tile tissue median for darkness check
    gray = np.mean(tile_rgb_uint8.astype(np.float32), axis=2)
    tissue_mask = gray > 10  # exclude true background (CZI padding)
    tile_tissue_median = float(np.median(gray[tissue_mask])) if tissue_mask.any() else 128.0

    lumens = []
    min_area_px = min_area_um2 / (pixel_size_um**2)
    max_area_px = 0.8 * tile_rgb_uint8.shape[0] * tile_rgb_uint8.shape[1]

    for r in results:
        mask = r["segmentation"]
        area_px = r["area"]

        # --- Size filter ---
        if area_px < min_area_px or area_px > max_area_px:
            continue

        # --- Interior darkness ---
        interior_vals = gray[mask]
        if len(interior_vals) == 0:
            continue
        interior_median = float(np.median(interior_vals))
        if interior_median >= tile_tissue_median:
            continue  # not darker than surrounding tissue

        # --- Local contrast: dilated ring around mask ---
        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(mask_uint8, kernel)
        ring = (dilated > 0) & (~mask)
        if not ring.any():
            continue
        boundary_median = float(np.median(gray[ring]))
        contrast = boundary_median / max(interior_median, 1.0)
        if contrast < min_contrast:
            continue

        # --- Extract contour ---
        contours_cv, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_cv:
            continue
        contour = max(contours_cv, key=cv2.contourArea)
        contour_pts = contour.reshape(-1, 2)  # (N, 2) as (x, y) in tile coords

        if len(contour_pts) < 3:
            continue

        lumens.append(
            {
                "contour_tile_xy": contour_pts,
                "area_px": area_px,
                "interior_median": interior_median,
                "boundary_median": boundary_median,
                "contrast_ratio": round(contrast, 3),
                "sam2_iou": float(r.get("predicted_iou", 0)),
                "sam2_stability": float(r.get("stability_score", 0)),
            }
        )

    return lumens


# ---------------------------------------------------------------------------
# Coordinate conversion: tile-local pixels -> global um
# ---------------------------------------------------------------------------


def tile_contour_to_global_um(
    contour_tile_xy: np.ndarray,
    tile_y: int,
    tile_x: int,
    scale_factor: int,
    base_pixel_size_um: float,
) -> np.ndarray:
    """Convert tile-local pixel contour to global um coordinates.

    Args:
        contour_tile_xy: (N, 2) array of (x, y) pixel coords within the tile.
        tile_y: Tile y-offset in pixels at this level.
        tile_x: Tile x-offset in pixels at this level.
        scale_factor: Downscale factor (2, 4, 8, 16).
        base_pixel_size_um: Pixel size at level 0 (full resolution).

    Returns:
        (N, 2) float64 array of (x, y) coordinates in global um.
    """
    pixel_size_at_level = base_pixel_size_um * scale_factor
    global_xy_um = contour_tile_xy.astype(np.float64) * pixel_size_at_level
    global_xy_um[:, 0] += tile_x * pixel_size_at_level  # x offset
    global_xy_um[:, 1] += tile_y * pixel_size_at_level  # y offset
    return global_xy_um


# ---------------------------------------------------------------------------
# Cross-tile IoU dedup
# ---------------------------------------------------------------------------


def dedup_lumens_iou(lumens: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Remove duplicate lumens from overlapping tiles using contour IoU.

    For each overlapping pair (IoU > threshold), keeps the one with higher
    SAM2 stability score. Uses Shapely STRtree for spatial indexing.

    Args:
        lumens: List of lumen dicts with 'contour_global_um' keys.
        iou_threshold: IoU threshold for considering two lumens duplicates.

    Returns:
        Deduplicated list of lumen dicts.
    """
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    if len(lumens) <= 1:
        return lumens

    # Build Shapely polygons from contours (in global um)
    polys = []
    valid_indices = []
    for i, lumen in enumerate(lumens):
        pts = lumen.get("contour_global_um")
        if pts is None or len(pts) < 3:
            continue
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty:
                polys.append(poly)
                valid_indices.append(i)
        except Exception as e:
            logger.debug("Polygon operation failed: %s", e)
            continue

    if len(polys) <= 1:
        return [lumens[i] for i in valid_indices]

    # Build STRtree + reverse lookup for older Shapely that returns geometry objects
    tree = STRtree(polys)
    poly_id_to_idx = {id(p): i for i, p in enumerate(polys)}
    suppressed = set()

    for qi in range(len(polys)):
        if qi in suppressed:
            continue
        poly_q = polys[qi]

        # Query nearby geometries
        try:
            candidates = tree.query(poly_q)
        except Exception as e:
            logger.debug("Polygon operation failed: %s", e)
            continue

        for ci_raw in candidates:
            # STRtree.query returns indices in newer Shapely, geometry objects in older
            if isinstance(ci_raw, int):
                ci = ci_raw
            else:
                ci = poly_id_to_idx.get(id(ci_raw))
                if ci is None:
                    continue

            if ci <= qi or ci in suppressed:
                continue

            poly_c = polys[ci]
            try:
                inter = poly_q.intersection(poly_c).area
                union = poly_q.union(poly_c).area
            except Exception:
                continue

            if union <= 0:
                continue
            iou = inter / union
            if iou >= iou_threshold:
                # Suppress the one with lower stability
                stab_q = lumens[valid_indices[qi]].get("sam2_stability", 0)
                stab_c = lumens[valid_indices[ci]].get("sam2_stability", 0)
                if stab_q >= stab_c:
                    suppressed.add(ci)
                else:
                    suppressed.add(qi)
                    break  # qi is suppressed, stop checking against it

    kept = [lumens[valid_indices[i]] for i in range(len(polys)) if i not in suppressed]
    logger.info(
        "IoU dedup: %d -> %d lumens (threshold %.2f)", len(lumens), len(kept), iou_threshold
    )
    return kept


# ---------------------------------------------------------------------------
# Cross-scale merge
# ---------------------------------------------------------------------------


def merge_across_scales(lumens: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Merge lumens detected at different scales, keeping highest-resolution.

    Process scales finest-first (smallest scale factor = highest resolution).
    At each coarser scale, only add lumens that don't overlap with already-
    accepted ones.

    Args:
        lumens: List of lumen dicts, each with 'scale' and 'contour_global_um'.
        iou_threshold: IoU threshold for considering cross-scale overlap.

    Returns:
        Merged list of lumen dicts.
    """
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    if len(lumens) <= 1:
        return lumens

    # Group by scale
    by_scale: dict[int, list[dict]] = {}
    for lumen in lumens:
        s = lumen.get("scale", 1)
        by_scale.setdefault(s, []).append(lumen)

    accepted: list[dict] = []
    accepted_polys: list[Polygon] = []

    for scale in sorted(by_scale.keys()):  # finest first (2, 4, 8, 16)
        scale_lumens = by_scale[scale]
        n_added = 0

        # Rebuild STRtree from accepted lumens for fast spatial queries.
        # Tree is rebuilt per-scale (not per-lumen) to amortize cost.
        accepted_tree = STRtree(accepted_polys) if accepted_polys else None
        accepted_id_to_idx = {id(p): i for i, p in enumerate(accepted_polys)}

        for lumen in scale_lumens:
            pts = lumen.get("contour_global_um")
            if pts is None or len(pts) < 3:
                continue
            try:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_valid or poly.is_empty:
                    continue
            except Exception as e:
                logger.debug("Polygon operation failed: %s", e)
                continue

            # Check overlap with already-accepted (finer-scale) lumens via STRtree
            overlaps = False
            if accepted_tree is not None:
                try:
                    candidates = accepted_tree.query(poly)
                except Exception as e:
                    logger.debug("STRtree query failed: %s", e)
                    candidates = []

                for ci_raw in candidates:
                    ci = ci_raw if isinstance(ci_raw, int) else accepted_id_to_idx.get(id(ci_raw))
                    if ci is None:
                        continue
                    ap = accepted_polys[ci]
                    try:
                        inter = poly.intersection(ap).area
                        union = poly.union(ap).area
                        if union > 0 and (inter / union) >= iou_threshold:
                            overlaps = True
                            break
                    except Exception as e:
                        logger.debug("Polygon operation failed: %s", e)
                        continue

            if not overlaps:
                accepted.append(lumen)
                accepted_polys.append(poly)
                n_added += 1

        logger.info("  Scale %dx: accepted %d / %d lumens", scale, n_added, len(scale_lumens))

    logger.info("Cross-scale merge: %d -> %d lumens", len(lumens), len(accepted))
    return accepted


# ---------------------------------------------------------------------------
# Biological validation with marker+ cells
# ---------------------------------------------------------------------------


def validate_lumens_with_cells(
    lumens: list[dict],
    cell_positions_um: np.ndarray,
    cell_features: list[dict],
    marker_names: list[str],
    radius_min: float = 20.0,
    radius_max: float = 100.0,
    min_coverage: float = 0.1,
    save_debug: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Validate lumens by checking for marker+ cells near the boundary.

    For each lumen, finds cells within a scale-adaptive radius of the lumen
    boundary and computes coverage (fraction of boundary 'staffed' by cells).

    Args:
        lumens: List of lumen candidate dicts.
        cell_positions_um: (M, 2) array of cell positions in um.
        cell_features: List of feature dicts for each cell.
        marker_names: Marker names used for filtering (informational).
        radius_min: Minimum assignment radius in um.
        radius_max: Maximum assignment radius in um.
        min_coverage: Minimum boundary cell coverage to validate.
        save_debug: If True, keep rejected lumens with rejection reasons.

    Returns:
        Tuple of (validated_lumens, rejected_lumens).
    """
    import cv2
    from scipy.spatial import cKDTree

    if len(cell_positions_um) == 0:
        logger.warning("No cell positions provided for validation")
        return [], lumens if save_debug else []

    tree = cKDTree(cell_positions_um)

    validated = []
    rejected = []

    for lumen in lumens:
        contour = lumen.get("contour_global_um")
        if contour is None or len(contour) < 3:
            lumen["rejection_reason"] = "degenerate_contour"
            if save_debug:
                rejected.append(lumen)
            continue

        centroid = contour.mean(axis=0)

        # Scale-adaptive radius based on equivalent diameter
        equiv_diam = lumen.get("equiv_diameter_um", 50.0)
        radius = float(np.clip(0.5 * equiv_diam, radius_min, radius_max))

        # Prepare contour for cv2.pointPolygonTest (float32, Nx1x2 format).
        # cv2.pointPolygonTest returns signed distance: positive = outside,
        # negative = inside, 0 = on boundary. Single C++ call replaces both
        # Shapely contains() and boundary.distance().
        contour_cv = contour.astype(np.float32).reshape(-1, 1, 2)

        # Validate contour is usable (non-degenerate area)
        contour_area = cv2.contourArea(contour_cv)
        if contour_area <= 0:
            lumen["rejection_reason"] = "zero_area_contour"
            if save_debug:
                rejected.append(lumen)
            continue

        # Find cells near lumen centroid (generous initial search).
        # Use bounding box diagonal for elongated lumens — equiv_diam underestimates
        # the extent for oblique aorta cuts where major >> minor axis.
        bbox_diag = float(np.sqrt(np.ptp(contour[:, 0]) ** 2 + np.ptp(contour[:, 1]) ** 2))
        search_radius = radius + max(equiv_diam, bbox_diag) / 2
        nearby_idx = tree.query_ball_point(centroid, search_radius)

        if not nearby_idx:
            lumen["rejection_reason"] = "no_cells_nearby"
            if save_debug:
                rejected.append(lumen)
            continue

        # Fine filter: signed distance from cell to lumen boundary via cv2.
        # Positive = outside (wall cell), negative = inside (skip).
        assigned_cells = []
        for idx in nearby_idx:
            pt = (float(cell_positions_um[idx, 0]), float(cell_positions_um[idx, 1]))
            dist = cv2.pointPolygonTest(contour_cv, pt, measureDist=True)
            if dist < 0:
                continue  # cell is inside lumen — not a wall cell
            if dist <= radius:
                assigned_cells.append({"cell_idx": int(idx), "distance_um": round(float(dist), 2)})

        if not assigned_cells:
            lumen["rejection_reason"] = "no_cells_within_radius"
            if save_debug:
                rejected.append(lumen)
            continue

        # Compute mean cell diameter for coverage calculation
        cell_diams = []
        for ac in assigned_cells:
            a = cell_features[ac["cell_idx"]].get("area_um2", 0)
            if a > 0:
                cell_diams.append(2 * np.sqrt(a / np.pi))
        mean_cell_diam = float(np.mean(cell_diams)) if cell_diams else 10.0

        # Coverage: fraction of boundary 'staffed' by marker+ cells
        perimeter = lumen.get("perimeter_um", float(cv2.arcLength(contour_cv, True)))
        expected_cells = perimeter / max(mean_cell_diam, 1.0)
        coverage = len(assigned_cells) / max(expected_cells, 1.0)

        if coverage < min_coverage:
            lumen["rejection_reason"] = f"low_coverage_{coverage:.3f}"
            if save_debug:
                rejected.append(lumen)
            continue

        # Attach validation metadata
        lumen["n_assigned_cells"] = len(assigned_cells)
        lumen["coverage"] = round(float(coverage), 3)
        lumen["assigned_cell_indices"] = [ac["cell_idx"] for ac in assigned_cells]
        lumen["cell_distances_um"] = [ac["distance_um"] for ac in assigned_cells]
        lumen["mean_cell_distance_um"] = round(
            float(np.mean([ac["distance_um"] for ac in assigned_cells])), 2
        )
        lumen["assignment_radius_um"] = round(radius, 2)
        validated.append(lumen)

    logger.info(
        "Validation: %d / %d lumens validated (%.0f%%), %d rejected",
        len(validated),
        len(lumens),
        100.0 * len(validated) / max(len(lumens), 1),
        len(rejected),
    )
    return validated, rejected


# ---------------------------------------------------------------------------
# Vessel characterization (calls into shared module if available)
# ---------------------------------------------------------------------------


def characterize_vessel(
    lumen: dict,
    detections: list[dict],
    marker_names: list[str],
    class_keys: list[str],
) -> dict:
    """Characterize a validated lumen as a vessel.

    Computes lumen morphometry, wall morphometry, marker composition,
    spatial layering, and assigns a vessel type. Falls back to inline
    implementations if the shared vessel_characterization module is not
    yet available.

    Args:
        lumen: Validated lumen dict with assigned cell data.
        detections: Full detection list.
        marker_names: List of marker names.
        class_keys: List of class keys (e.g., 'SMA_class').

    Returns:
        Vessel characterization dict.
    """
    contour = lumen["contour_global_um"]
    assigned_indices = lumen.get("assigned_cell_indices", [])

    from xldvp_seg.analysis.vessel_characterization import (
        analyze_marker_composition as _amc,
    )
    from xldvp_seg.analysis.vessel_characterization import (
        assign_vessel_type as _avt,
    )
    from xldvp_seg.analysis.vessel_characterization import (
        compute_lumen_morphometry as _clm,
    )
    from xldvp_seg.analysis.vessel_characterization import (
        compute_wall_morphometry as _cwm,
    )
    from xldvp_seg.analysis.vessel_characterization import (
        detect_spatial_layering_from_boundary as _dsl_boundary,
    )
    from xldvp_seg.utils.detection_utils import extract_positions_um

    lumen_morph = _clm(contour)

    # Extract cell positions + areas in lockstep (extract_positions_um may skip
    # cells with unresolvable coordinates — must keep areas aligned).
    assigned_dets = [detections[i] for i in assigned_indices]
    valid_positions = []
    valid_areas = []
    pixel_size_inferred = None
    for det in assigned_dets:
        pos_arr, pixel_size_inferred = extract_positions_um(
            [det], pixel_size_um=pixel_size_inferred
        )
        if len(pos_arr) == 1:
            valid_positions.append(pos_arr[0])
            valid_areas.append(det.get("features", {}).get("area_um2", 0))
    cell_positions_um = (
        np.array(valid_positions, dtype=np.float64)
        if valid_positions
        else np.empty((0, 2), dtype=np.float64)
    )
    cell_areas = np.array(valid_areas, dtype=np.float64)
    wall_morph = _cwm(contour, cell_positions_um, cell_areas if cell_areas.any() else None)
    composition = _amc(detections, assigned_indices, marker_names, class_keys)

    # Spatial layering using distance from lumen boundary (not centroid)
    cell_distances = lumen.get("cell_distances_um", [])
    class_keys_map = dict(zip(marker_names, class_keys))
    layering = (
        _dsl_boundary(detections, assigned_indices, cell_distances, class_keys_map)
        if len(assigned_indices) >= 10 and len(cell_distances) == len(assigned_indices)
        else {}
    )

    morphometry = {**lumen_morph, **wall_morph}
    # Use lumen_equiv_diameter_um as vessel_diameter_um for vessel typing
    morphometry["vessel_diameter_um"] = lumen_morph.get("lumen_equiv_diameter_um", 0)
    morphometry["wall_extent_um"] = wall_morph.get("wall_thickness_um", 0)

    morphology = "strip" if lumen_morph.get("lumen_elongation", 1.0) > 4.0 else "ring"
    vessel_type = _avt(
        morphology=morphology,
        composition=composition,
        layering=layering,
        morphometry=morphometry,
    )

    return {
        "morphology": morphology,
        "vessel_type": vessel_type,
        "morphometry": morphometry,
        "composition": composition,
        "layering": layering if layering else None,
    }


# ---------------------------------------------------------------------------
# Cell data extraction
# ---------------------------------------------------------------------------


def classify_markers_by_snr_percentile(
    detections: list[dict],
    marker_names: list[str],
    snr_keys: list[str],
    percentile: float = 95.0,
) -> int:
    """Tag detections with {marker}_class based on top-N% SNR per channel.

    Modifies detections in-place. Each cell is tagged as positive for a marker
    if its SNR in that channel is >= the percentile threshold. Cells above
    threshold for ANY marker are considered marker-positive.

    Args:
        detections: Full detection list (modified in-place).
        marker_names: Marker names (e.g., ["SMA", "CD31"]).
        snr_keys: Corresponding SNR feature keys (e.g., ["ch1_snr", "ch3_snr"]).
        percentile: Percentile threshold (default 95 = top 5%).

    Returns:
        Number of cells tagged as positive for at least one marker.
    """
    # Compute threshold per channel
    thresholds = {}
    for marker, snr_key in zip(marker_names, snr_keys):
        vals = np.array([d.get("features", {}).get(snr_key, 0) for d in detections])
        thresholds[marker] = float(np.percentile(vals, percentile))
        logger.info(
            "  %s (%s): p%.0f threshold = %.3f", marker, snr_key, percentile, thresholds[marker]
        )

    # Tag each cell
    n_positive = 0
    for d in detections:
        feat = d.setdefault("features", {})
        any_pos = False
        for marker, snr_key in zip(marker_names, snr_keys):
            is_pos = feat.get(snr_key, 0) >= thresholds[marker]
            feat[f"{marker}_class"] = "positive" if is_pos else "negative"
            if is_pos:
                any_pos = True
        if any_pos:
            n_positive += 1

    logger.info(
        "SNR percentile classification: %d / %d cells positive (%.1f%%)",
        n_positive,
        len(detections),
        100.0 * n_positive / max(len(detections), 1),
    )
    return n_positive


def extract_cell_data(
    detections: list[dict], marker_names: str
) -> tuple[np.ndarray, list[dict], list[int]]:
    """Extract positions and features for marker+ cells from detections.

    Selects cells that are positive for ANY of the specified markers.

    Args:
        detections: Full detection list.
        marker_names: Comma-separated marker names.

    Returns:
        Tuple of (positions_um, features_list, original_indices).
    """
    from xldvp_seg.utils.detection_utils import extract_positions_um

    names = [m.strip() for m in marker_names.split(",")]
    class_keys = [f"{name}_class" for name in names]

    # Select marker+ cells (OR logic)
    positive_idx = []
    for i, det in enumerate(detections):
        feat = det.get("features", {})
        for key in class_keys:
            if feat.get(key) == "positive" or det.get(key) == "positive":
                positive_idx.append(i)
                break

    logger.info(
        "Selected %d / %d marker+ cells (%s)",
        len(positive_idx),
        len(detections),
        ", ".join(names),
    )

    if not positive_idx:
        return np.empty((0, 2), dtype=np.float64), [], []

    # Extract positions one-by-one to maintain alignment (extract_positions_um
    # may skip detections with unresolvable coords). First call infers pixel_size.
    pos_dets = [detections[i] for i in positive_idx]
    _, pixel_size = extract_positions_um(pos_dets[:1])  # infer pixel_size from first det

    valid_pos = []
    valid_feats = []
    valid_idx = []
    for i, det in zip(positive_idx, pos_dets):
        pos_arr, _ = extract_positions_um([det], pixel_size_um=pixel_size)
        if len(pos_arr) == 1:
            valid_pos.append(pos_arr[0])
            valid_feats.append(det.get("features", {}))
            valid_idx.append(i)

    if len(valid_pos) < len(positive_idx):
        logger.warning(
            "Dropped %d cells with unresolvable coordinates",
            len(positive_idx) - len(valid_pos),
        )

    positions = (
        np.array(valid_pos, dtype=np.float64) if valid_pos else np.empty((0, 2), dtype=np.float64)
    )
    return positions, valid_feats, valid_idx


# ---------------------------------------------------------------------------
# Pixel size from OME-Zarr metadata or CZI
# ---------------------------------------------------------------------------


def get_pixel_size(zarr_root: Any, czi_path: str | None = None) -> float:
    """Extract base pixel size (level 0) from OME-Zarr metadata or CZI.

    Tries OME-NGFF multiscales metadata first. Falls back to CZI metadata
    if zarr_root lacks the information.

    Args:
        zarr_root: Opened zarr group.
        czi_path: Optional CZI path for fallback metadata.

    Returns:
        Pixel size in um at level 0.
    """
    # Try OME-NGFF metadata
    try:
        multiscales = zarr_root.attrs["multiscales"]
        datasets = multiscales[0]["datasets"]
        # Level 0 scale transform
        transforms = datasets[0]["coordinateTransformations"]
        for t in transforms:
            if t["type"] == "scale":
                scale = t["scale"]
                # OME-NGFF axes: [c, y, x] -> y scale is pixel size
                if len(scale) >= 3:
                    pixel_size = float(scale[1])  # y-axis scale
                    if pixel_size > 0:
                        logger.info("Pixel size from OME-Zarr metadata: %.4f um", pixel_size)
                        return pixel_size
    except (KeyError, IndexError, TypeError):
        pass

    # Try CZI metadata
    if czi_path and Path(czi_path).exists():
        try:
            from aicspylibczi import CziFile

            czi = CziFile(str(czi_path))
            metadata = czi.meta
            scaling = metadata.find('.//Scaling/Items/Distance[@Id="X"]/Value')
            if scaling is not None:
                pixel_size = float(scaling.text) * 1e6
                logger.info("Pixel size from CZI metadata: %.4f um", pixel_size)
                return pixel_size
        except Exception as e:
            logger.warning("Failed to read CZI metadata: %s", e)

    # Fallback
    logger.warning(
        "Could not determine pixel size from zarr or CZI metadata. "
        "Falling back to 0.1725 um/px -- verify against CZI metadata!"
    )
    return 0.1725


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def build_vessel_record(
    vessel_id: int,
    lumen: dict,
    characterization: dict,
    scale: int,
) -> dict:
    """Build a vessel record for JSON output.

    Combines lumen geometry, SAM2 metrics, validation data, and vessel
    characterization into a single flat-ish dict. Contours are stored as
    lists-of-lists for JSON serialization.

    Args:
        vessel_id: Unique vessel ID.
        lumen: Validated lumen dict.
        characterization: Output of characterize_vessel().
        scale: Scale factor at which this lumen was detected.

    Returns:
        Vessel record dict.
    """
    contour = lumen.get("contour_global_um")
    contour_list = contour.tolist() if isinstance(contour, np.ndarray) else contour

    record = {
        "vessel_id": vessel_id,
        "scale": scale,
        "morphology": characterization.get("morphology", "unknown"),
        "vessel_type": characterization.get("vessel_type", "unclassified"),
        "contour_global_um": contour_list,
        # Lumen geometry
        "area_px": lumen.get("area_px"),
        "equiv_diameter_um": lumen.get("equiv_diameter_um"),
        "perimeter_um": lumen.get("perimeter_um"),
        # SAM2 metrics
        "sam2_iou": lumen.get("sam2_iou"),
        "sam2_stability": lumen.get("sam2_stability"),
        # Contrast metrics
        "contrast_ratio": lumen.get("contrast_ratio"),
        "interior_median": lumen.get("interior_median"),
        "boundary_median": lumen.get("boundary_median"),
        # Validation metrics
        "n_assigned_cells": lumen.get("n_assigned_cells", 0),
        "coverage": lumen.get("coverage", 0),
        "mean_cell_distance_um": lumen.get("mean_cell_distance_um"),
        "assignment_radius_um": lumen.get("assignment_radius_um"),
        "assigned_cell_indices": lumen.get("assigned_cell_indices", []),
        "cell_distances_um": lumen.get("cell_distances_um", []),
        # Morphometry + composition
        "morphometry": characterization.get("morphometry"),
        "composition": characterization.get("composition"),
        "layering": characterization.get("layering"),
    }
    return record


def write_vessel_csv(vessels: list[dict], csv_path: Path, marker_names: list[str]) -> None:
    """Write vessel summary CSV.

    Args:
        vessels: List of vessel record dicts.
        csv_path: Output CSV path.
        marker_names: Marker names for dynamic columns.
    """
    if not vessels:
        logger.info("No vessels to write to CSV")
        return

    # Dynamic marker columns
    marker_cols = []
    for name in marker_names:
        marker_cols.extend([f"n_{name.lower()}", f"{name.lower()}_frac"])
    marker_cols.extend(["n_double_pos", "dominant_marker"])

    fieldnames = [
        "vessel_id",
        "scale",
        "morphology",
        "vessel_type",
        "equiv_diameter_um",
        "perimeter_um",
        "n_assigned_cells",
        "coverage",
        "mean_cell_distance_um",
        "contrast_ratio",
        "sam2_iou",
        "sam2_stability",
    ] + marker_cols

    # Add morphometry columns that exist
    morph_cols = [
        "lumen_area_um2",
        "lumen_perimeter_um",
        "lumen_equiv_diameter_um",
        "lumen_elongation",
        "lumen_circularity",
        "centroid_x_um",
        "centroid_y_um",
    ]
    fieldnames.extend(morph_cols)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in vessels:
            row = {k: v.get(k) for k in fieldnames}
            # Flatten composition and morphometry into row
            comp = v.get("composition") or {}
            morph = v.get("morphometry") or {}
            for k2, v2 in comp.items():
                if k2 in fieldnames:
                    row[k2] = v2
            for k2, v2 in morph.items():
                if k2 in fieldnames:
                    row[k2] = v2
            writer.writerow(row)

    logger.info("Saved CSV summary (%d vessels) to %s", len(vessels), csv_path)


def tag_cell_detections(
    detections: list[dict],
    vessels: list[dict],
    prefix: str = "lumen_vessel",
) -> None:
    """Tag cell detections with lumen-vessel membership.

    Modifies detections in-place, adding vessel_id, vessel_type, and morphology
    to each assigned cell's features.

    Args:
        detections: Full detection list (modified in-place).
        vessels: List of vessel record dicts.
        prefix: Feature key prefix.
    """
    field_id = f"{prefix}_id"
    field_type = f"{prefix}_type"
    field_morph = f"{prefix}_morphology"
    field_dist = f"{prefix}_distance_um"

    # Initialize all detections
    for d in detections:
        feat = d.setdefault("features", {})
        feat[field_id] = -1
        feat[field_type] = "none"
        feat[field_morph] = "none"

    # Apply vessel assignments
    for v in vessels:
        vid = v["vessel_id"]
        vtype = v.get("vessel_type", "unclassified")
        vmorph = v.get("morphology", "unknown")
        indices = v.get("assigned_cell_indices", [])
        distances = v.get("cell_distances_um", [])

        for j, idx in enumerate(indices):
            if 0 <= idx < len(detections):
                feat = detections[idx].setdefault("features", {})
                feat[field_id] = vid
                feat[field_type] = vtype
                feat[field_morph] = vmorph
                if j < len(distances):
                    feat[field_dist] = distances[j]

    # Summary
    assigned = sum(1 for d in detections if d.get("features", {}).get(field_id, -1) >= 0)
    logger.info("Tagged %d / %d detections with vessel membership", assigned, len(detections))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lumen-first vessel detection using SAM2 on OME-Zarr tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--zarr-path", required=True, help="Path to OME-Zarr store")
    parser.add_argument(
        "--detections",
        required=True,
        help="Top-5%% marker+ cell detections JSON (from classify_markers.py)",
    )
    parser.add_argument(
        "--czi-path", default=None, help="CZI path for metadata fallback (pixel size)"
    )
    parser.add_argument(
        "--marker-names",
        required=True,
        help="Comma-separated marker names (e.g., SMA,CD31)",
    )
    parser.add_argument(
        "--marker-snr-channels",
        default=None,
        help="Comma-separated SNR feature keys per marker (e.g., ch1_snr,ch3_snr). "
        "When provided, computes top-5%% SNR per channel and tags cells as "
        "{marker}_class=positive/negative. Use when detections lack _class fields. "
        "Must match --marker-names order.",
    )
    parser.add_argument(
        "--marker-percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for SNR-based marker classification (default: 95 = top 5%%)",
    )
    parser.add_argument(
        "--rgb-channels",
        required=True,
        help="3 zarr channel indices for SAM2 RGB input (e.g., 1,3,0)",
    )
    parser.add_argument(
        "--scales",
        default="2,4,8,16",
        help="Comma-separated scale factors (zarr levels = log2(scale))",
    )
    parser.add_argument(
        "--tile-size", type=int, default=3000, help="Tile size in pixels at each level"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=750,
        help="Tile overlap in pixels (default: 750, ~25%% of tile — ensures most lumens "
        "fit entirely in at least one tile)",
    )
    parser.add_argument(
        "--min-lumen-area-um2",
        type=float,
        default=50.0,
        help="Minimum lumen area in um^2",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=1.5,
        help="Minimum boundary/interior contrast ratio",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.1,
        help="Minimum boundary marker cell coverage",
    )
    parser.add_argument(
        "--assignment-radius-min",
        type=float,
        default=20.0,
        help="Minimum cell-to-lumen assignment radius in um",
    )
    parser.add_argument(
        "--assignment-radius-max",
        type=float,
        default=100.0,
        help="Maximum cell-to-lumen assignment radius in um",
    )
    parser.add_argument(
        "--tile-shard",
        default=None,
        help="Tile sharding for multi-GPU: INDEX/TOTAL (e.g., 0/4). "
        "Each shard processes every TOTAL-th tile starting at INDEX. "
        "Run N shards in parallel (one per GPU), then merge results.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge shard results instead of running detection. "
        "Reads vessel_lumens_shard_*.json from --output-dir.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save rejected lumens for debugging",
    )

    # SAM2 auto-mask tuning
    parser.add_argument(
        "--sam2-points-per-side",
        type=int,
        default=32,
        help="SAM2 auto-mask: points per side of the grid",
    )
    parser.add_argument(
        "--sam2-pred-iou-thresh",
        type=float,
        default=0.7,
        help="SAM2 auto-mask: predicted IoU threshold",
    )
    parser.add_argument(
        "--sam2-stability-thresh",
        type=float,
        default=0.8,
        help="SAM2 auto-mask: stability score threshold",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Handle --merge-shards mode (skip detection, just merge + validate + characterize)
    if args.merge_shards:
        _merge_shards_and_finish(args)
        return
    t0 = time.time()

    zarr_path = Path(args.zarr_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse channel indices
    rgb_channels = [int(c.strip()) for c in args.rgb_channels.split(",")]
    if len(rgb_channels) != 3:
        logger.error("--rgb-channels must specify exactly 3 channels, got %d", len(rgb_channels))
        sys.exit(1)

    scales = sorted(int(s.strip()) for s in args.scales.split(","))
    marker_names = [m.strip() for m in args.marker_names.split(",")]
    class_keys = [f"{name}_class" for name in marker_names]

    # Parse tile shard
    shard_idx, shard_total = 0, 1
    if args.tile_shard:
        parts = args.tile_shard.split("/")
        if len(parts) != 2:
            logger.error("--tile-shard must be INDEX/TOTAL (e.g., 0/4)")
            sys.exit(1)
        shard_idx, shard_total = int(parts[0]), int(parts[1])
        logger.info("Tile shard: %d / %d", shard_idx, shard_total)

    logger.info("=" * 70)
    logger.info("LUMEN-FIRST VESSEL DETECTION")
    logger.info("=" * 70)
    logger.info("  Zarr: %s", zarr_path)
    logger.info("  Detections: %s", args.detections)
    logger.info("  Markers: %s", marker_names)
    logger.info("  RGB channels: %s", rgb_channels)
    logger.info("  Scales: %s", scales)
    logger.info("  Tile size: %d px, overlap: %d px", args.tile_size, args.overlap)
    logger.info("  Min lumen area: %.1f um^2", args.min_lumen_area_um2)
    logger.info("  Min contrast: %.2f", args.min_contrast)
    logger.info("  Min coverage: %.2f", args.min_coverage)
    logger.info(
        "  Assignment radius: %.0f - %.0f um",
        args.assignment_radius_min,
        args.assignment_radius_max,
    )
    logger.info("  Output: %s", output_dir)

    # ---- 1. Open zarr ----
    import zarr

    _zarr_v3 = int(zarr.__version__.split(".")[0]) >= 3
    if _zarr_v3:
        root = zarr.open_group(str(zarr_path), mode="r")
    else:
        root = zarr.open(str(zarr_path), mode="r")

    # Resolve scales against zarr pyramid (supports beyond-pyramid downsampling)
    try:
        available_levels = resolve_zarr_scales(root, scales)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    if not available_levels:
        logger.error("No valid zarr levels found for scales %s", scales)
        sys.exit(1)

    # ---- 2. Get pixel size ----
    base_pixel_size = get_pixel_size(root, args.czi_path)
    logger.info("Base pixel size: %.4f um/px", base_pixel_size)

    # ---- 3. Load SAM2 (single GPU per shard task) ----
    logger.info("Loading SAM2...")
    from xldvp_seg.models.manager import get_model_manager
    from xldvp_seg.utils.device import get_default_device

    device = get_default_device()
    manager = get_model_manager(device=device)
    _, _ = manager.get_sam2()

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_lumen = SAM2AutomaticMaskGenerator(
        manager._sam2_model,
        points_per_side=args.sam2_points_per_side,
        pred_iou_thresh=args.sam2_pred_iou_thresh,
        stability_score_thresh=args.sam2_stability_thresh,
        min_mask_region_area=100,
        crop_n_layers=1,
    )
    logger.info(
        "SAM2 ready on %s (points_per_side=%d, iou_thresh=%.2f, stability=%.2f)",
        device,
        args.sam2_points_per_side,
        args.sam2_pred_iou_thresh,
        args.sam2_stability_thresh,
    )

    # ---- 4. Multi-scale lumen detection (multi-GPU) ----

    import cv2
    from tqdm import tqdm

    from xldvp_seg.utils.device import empty_cache

    # Scale-specific size bounds (min/max diameter in um).
    # Auto-computed for scales not in the dict: min = scale * 7.5, max = 100000.
    _size_bounds = {
        2: (15, 300),
        4: (50, 800),
        8: (200, 2000),
        16: (500, 100000),
        32: (1000, 100000),
        64: (2000, 100000),
    }

    all_lumens: list[dict] = []

    for scale, level, level_key, extra_ds in available_levels:
        level_data = root[level_key]
        pixel_size_at_level = base_pixel_size * scale
        eff_shape = get_effective_shape(level_data, extra_ds)

        all_tiles = generate_tiles(eff_shape, args.tile_size, args.overlap)
        # Round-robin shard: each shard gets every N-th tile
        tiles = [t for i, t in enumerate(all_tiles) if i % shard_total == shard_idx]
        logger.info(
            "Scale %dx: %d/%d tiles (shard %d/%d, level %d%s, %.3f um/px)",
            scale,
            len(tiles),
            len(all_tiles),
            shard_idx,
            shard_total,
            level,
            f" + {extra_ds}x downsample" if extra_ds > 1 else "",
            pixel_size_at_level,
        )

        min_diam, max_diam = _size_bounds.get(scale, (scale * 7.5, 100000))
        scale_lumens_before = len(all_lumens)

        for tile_y, tile_x, tile_h, tile_w in tqdm(
            tiles, desc=f"Scale {scale}x", unit="tile", leave=True
        ):
            # Read channels, crop to actual data extent (no zero-padding)
            actual_h, actual_w = tile_h, tile_w
            channels_ok = True
            channel_arrays = []
            for ch_idx in rgb_channels:
                try:
                    raw = read_zarr_tile(
                        level_data, ch_idx, tile_y, tile_x, tile_h, tile_w, extra_ds
                    )
                except Exception as e:
                    logger.debug(
                        "Tile ch=%d y=%d x=%d scale %dx: %s", ch_idx, tile_y, tile_x, scale, e
                    )
                    channels_ok = False
                    break
                actual_h = min(actual_h, raw.shape[0])
                actual_w = min(actual_w, raw.shape[1])
                channel_arrays.append(raw)

            if channels_ok:
                rgb = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)
                for i, raw in enumerate(channel_arrays):
                    rgb[:, :, i] = percentile_normalize_uint8(raw[:actual_h, :actual_w])

                tile_lumens = detect_lumens_in_tile(
                    rgb,
                    pixel_size_at_level,
                    args.min_lumen_area_um2,
                    args.min_contrast,
                    sam2_lumen,
                )

                for lumen in tile_lumens:
                    lumen["contour_global_um"] = tile_contour_to_global_um(
                        lumen["contour_tile_xy"], tile_y, tile_x, scale, base_pixel_size
                    )
                    lumen["scale"] = scale
                    area_um2 = lumen["area_px"] * pixel_size_at_level**2
                    lumen["equiv_diameter_um"] = round(2 * np.sqrt(area_um2 / np.pi), 2)
                    lumen["perimeter_um"] = round(
                        float(
                            cv2.arcLength(
                                lumen["contour_tile_xy"].reshape(-1, 1, 2).astype(np.int32), True
                            )
                        )
                        * pixel_size_at_level,
                        2,
                    )
                    del lumen["contour_tile_xy"]

                    d = lumen["equiv_diameter_um"]
                    if min_diam <= d <= max_diam:
                        all_lumens.append(lumen)

        empty_cache()

        n_new = len(all_lumens) - scale_lumens_before
        logger.info("  -> %d new candidates at scale %dx (%d total)", n_new, scale, len(all_lumens))

    # ---- 5. Save shard or proceed to dedup ----
    if shard_total > 1:
        # Save shard results for later merging
        shard_out = output_dir / f"vessel_lumens_shard_{shard_idx}.json"
        atomic_json_dump(all_lumens, str(shard_out))
        logger.info(
            "Shard %d/%d: saved %d candidates to %s",
            shard_idx,
            shard_total,
            len(all_lumens),
            shard_out,
        )
        return  # merge step runs separately via --merge-shards

    if not all_lumens:
        logger.warning("No lumen candidates found at any scale. Exiting.")
        atomic_json_dump([], str(output_dir / "vessel_lumens.json"))
        logger.info("Wrote empty vessel_lumens.json")
        return

    logger.info("Total candidates before dedup: %d", len(all_lumens))

    # ---- 6. Cross-tile dedup ----
    all_lumens = dedup_lumens_iou(all_lumens, iou_threshold=0.3)

    # ---- 7. Cross-scale merge ----
    all_lumens = merge_across_scales(all_lumens, iou_threshold=0.3)
    logger.info("After cross-scale merge: %d lumens", len(all_lumens))

    # ---- 8. Validate, characterize, output (shared with --merge-shards) ----
    _validate_characterize_and_output(all_lumens, args, output_dir, marker_names, class_keys)

    elapsed = time.time() - t0
    logger.info("Completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)


def _validate_characterize_and_output(
    all_lumens: list[dict],
    args: argparse.Namespace,
    output_dir: Path,
    marker_names: list[str],
    class_keys: list[str],
) -> None:
    """Validate lumens, characterize vessels, write all output files.

    Shared by both the single-run path in main() and the --merge-shards path.
    """
    logger.info("Loading cell detections from %s...", args.detections)
    detections = fast_json_load(str(args.detections))
    logger.info("  %d detections loaded", len(detections))

    # If --marker-snr-channels is provided, classify cells by top-N% SNR
    # before selecting marker+ cells. This handles detections that lack _class fields.
    if args.marker_snr_channels:
        snr_keys = [k.strip() for k in args.marker_snr_channels.split(",")]
        if len(snr_keys) != len(marker_names):
            logger.error(
                "--marker-snr-channels has %d keys but --marker-names has %d markers",
                len(snr_keys),
                len(marker_names),
            )
            sys.exit(1)
        logger.info("Classifying markers by top-%.0f%% SNR...", 100 - args.marker_percentile)
        classify_markers_by_snr_percentile(
            detections, marker_names, snr_keys, args.marker_percentile
        )

    cell_positions, cell_features, cell_indices = extract_cell_data(detections, args.marker_names)
    if len(cell_positions) == 0:
        logger.error("No marker+ cells found. Cannot validate lumens.")
        sys.exit(1)

    validated, rejected = validate_lumens_with_cells(
        all_lumens,
        cell_positions,
        cell_features,
        marker_names,
        radius_min=args.assignment_radius_min,
        radius_max=args.assignment_radius_max,
        min_coverage=args.min_coverage,
        save_debug=args.save_debug,
    )

    if not validated:
        logger.warning("No lumens passed biological validation.")
        atomic_json_dump([], str(output_dir / "vessel_lumens.json"))
        if args.save_debug and rejected:
            atomic_json_dump(
                [
                    {
                        "rejection_reason": r.get("rejection_reason", "unknown"),
                        "equiv_diameter_um": r.get("equiv_diameter_um"),
                        "scale": r.get("scale"),
                        "contrast_ratio": r.get("contrast_ratio"),
                    }
                    for r in rejected
                ],
                str(output_dir / "vessel_lumens_rejected.json"),
            )
        return

    # Characterize each vessel
    logger.info("Characterizing %d validated vessels...", len(validated))
    vessels: list[dict] = []

    for vid, lumen in enumerate(validated):
        # Map cell indices from cell_positions space to detections space (lockstep)
        assigned_cell_pos_idx = lumen.get("assigned_cell_indices", [])
        old_distances = lumen.get("cell_distances_um", [])
        new_indices = []
        new_distances = []
        for k, j in enumerate(assigned_cell_pos_idx):
            if j < len(cell_indices):
                new_indices.append(cell_indices[j])
                if k < len(old_distances):
                    new_distances.append(old_distances[k])
        lumen["assigned_cell_indices"] = new_indices
        lumen["cell_distances_um"] = new_distances
        lumen["n_assigned_cells"] = len(new_indices)

        characterization = characterize_vessel(lumen, detections, marker_names, class_keys)
        record = build_vessel_record(vid, lumen, characterization, lumen.get("scale", 0))
        vessels.append(record)

    tag_cell_detections(detections, vessels, prefix="lumen_vessel")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("LUMEN VESSEL DETECTION SUMMARY")
    logger.info("=" * 70)
    logger.info("  Total validated vessels: %d", len(vessels))
    type_counts: dict[str, int] = {}
    morph_counts: dict[str, int] = {}
    scale_counts: dict[int, int] = {}
    for v in vessels:
        vt = v.get("vessel_type", "unclassified")
        vm = v.get("morphology", "unknown")
        vs = v.get("scale", 0)
        type_counts[vt] = type_counts.get(vt, 0) + 1
        morph_counts[vm] = morph_counts.get(vm, 0) + 1
        scale_counts[vs] = scale_counts.get(vs, 0) + 1
    logger.info("  Vessel types: %s", type_counts)
    logger.info("  Morphology: %s", morph_counts)
    logger.info("  By scale: %s", scale_counts)
    logger.info("  Total assigned cells: %d", sum(v.get("n_assigned_cells", 0) for v in vessels))

    # Write outputs
    atomic_json_dump(vessels, str(output_dir / "vessel_lumens.json"))
    logger.info("Saved %d vessel records to %s", len(vessels), output_dir / "vessel_lumens.json")

    det_out = output_dir / "cell_detections_vessels.json"
    atomic_json_dump(detections, str(det_out))
    logger.info("Saved tagged detections to %s", det_out)

    vessel_cells = [d for d in detections if d.get("features", {}).get("lumen_vessel_id", -1) >= 0]
    atomic_json_dump(vessel_cells, str(output_dir / "cell_detections_vessel_only.json"))
    logger.info("Saved %d vessel-only detections", len(vessel_cells))

    write_vessel_csv(vessels, output_dir / "vessel_summary.csv", marker_names)

    if args.save_debug and rejected:
        atomic_json_dump(
            [
                {
                    "rejection_reason": r.get("rejection_reason", "unknown"),
                    "equiv_diameter_um": r.get("equiv_diameter_um"),
                    "scale": r.get("scale"),
                    "contrast_ratio": r.get("contrast_ratio"),
                    "interior_median": r.get("interior_median"),
                    "boundary_median": r.get("boundary_median"),
                    "sam2_iou": r.get("sam2_iou"),
                    "sam2_stability": r.get("sam2_stability"),
                }
                for r in rejected
            ],
            str(output_dir / "vessel_lumens_rejected.json"),
        )
        logger.info("Saved %d rejected lumens for debugging", len(rejected))


def _merge_shards_and_finish(args: argparse.Namespace) -> None:
    """Merge shard results, dedup, validate, characterize, and write output.

    Called when --merge-shards is specified. Reads vessel_lumens_shard_*.json
    from --output-dir, concatenates, runs cross-tile dedup + cross-scale merge,
    then continues with biological validation and characterization.
    """
    import glob

    output_dir = Path(args.output_dir)
    shard_files = sorted(glob.glob(str(output_dir / "vessel_lumens_shard_*.json")))
    if not shard_files:
        logger.error("No shard files found in %s", output_dir)
        sys.exit(1)

    logger.info("Merging %d shard files...", len(shard_files))
    all_lumens: list[dict] = []
    for sf in shard_files:
        shard_data = fast_json_load(sf)
        logger.info("  %s: %d candidates", Path(sf).name, len(shard_data))
        # Convert contour lists back to numpy arrays
        for lumen in shard_data:
            c = lumen.get("contour_global_um")
            if c is not None and not isinstance(c, np.ndarray):
                lumen["contour_global_um"] = np.array(c, dtype=np.float64)
        all_lumens.extend(shard_data)
    logger.info("Total candidates from all shards: %d", len(all_lumens))

    if not all_lumens:
        logger.warning("No candidates after merging shards.")
        atomic_json_dump([], str(output_dir / "vessel_lumens.json"))
        return

    # Dedup + merge
    all_lumens = dedup_lumens_iou(all_lumens, iou_threshold=0.3)
    all_lumens = merge_across_scales(all_lumens, iou_threshold=0.3)
    logger.info("After dedup + cross-scale merge: %d lumens", len(all_lumens))

    # Validate, characterize, output (shared code path with main)
    marker_names = [m.strip() for m in args.marker_names.split(",")]
    class_keys = [f"{name}_class" for name in marker_names]
    _validate_characterize_and_output(all_lumens, args, output_dir, marker_names, class_keys)


if __name__ == "__main__":
    main()
