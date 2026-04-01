"""
Contour post-processing for LMD export.

Functions for preparing detection contours for laser microdissection:
1. Dilation - add buffer around contour so laser cuts outside the target
2. RDP simplification - reduce point count for LMD hardware limits
3. Validation - fix invalid polygons (self-intersections, etc.)

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Usage:
    from segmentation.lmd.contour_processing import process_contour, process_contours_batch

    # Single contour (pixel_size_um from CZI metadata)
    processed = process_contour(contour_px, pixel_size_um=pixel_size)

    # Batch processing
    results = process_contours_batch(contours_px, pixel_size_um=pixel_size)
"""

from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# Default parameters
# Removed: DEFAULT_PIXEL_SIZE_UM — pixel_size_um must come from CZI metadata
DEFAULT_DILATION_UM = 0.5  # Dilate by 0.5um
DEFAULT_RDP_EPSILON = 5  # RDP epsilon in pixels


def transform_native_to_display(pts_xy_um, orig_w_um, orig_h_um, flip_h, rot90):
    """Transform contour [x, y] um from native CZI space to display space.

    Applies the same transforms that napari_place_crosses.py applied to the
    image, so contours end up in the same coordinate system as the crosses.

    In [x, y] coordinates:
      flip_h:  x' = orig_w - x,  y' = y
      rot90:   x' = orig_h - y,  y' = x   (CW 90 deg)
    """
    pts = pts_xy_um.copy()
    if flip_h:
        pts[:, 0] = orig_w_um - pts[:, 0]
    if rot90:
        x_new = orig_h_um - pts[:, 1]
        y_new = pts[:, 0].copy()
        pts[:, 0] = x_new
        pts[:, 1] = y_new
    return pts


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Apply Ramer-Douglas-Peucker simplification using OpenCV.

    Args:
        points: Array of shape (N, 2) with coordinates
        epsilon: Maximum distance threshold for point removal (in same units as points)

    Returns:
        Simplified array of points, shape (M, 2) where M <= N
    """
    if len(points) < 3:
        return points

    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(points, epsilon, closed=True)
    return simplified.reshape(-1, 2)


def adaptive_rdp_simplify(
    contour_px: np.ndarray,
    max_area_change_pct: float = 5.0,
    max_epsilon: float = 20.0,
) -> tuple[np.ndarray, float]:
    """Find the largest RDP epsilon that keeps shape deviation within tolerance.

    Binary-searches over epsilon to maximise point reduction while keeping
    the symmetric difference between the original and simplified polygons
    within *max_area_change_pct* % of the original area.  Symmetric
    difference catches corner-cutting and area gain that pure area change
    would miss.

    Args:
        contour_px: (N, 2) contour in pixel coordinates.
        max_area_change_pct: Maximum allowed symmetric-difference / original-area
            (in percent, default 5.0).
        max_epsilon: Upper bound of the epsilon search range (pixels).

    Returns:
        ``(simplified_contour, epsilon_used)`` — the simplified (M, 2) array
        and the epsilon that was applied.  If the contour is already simple
        (< 10 points) or cannot be simplified within tolerance, the original
        contour is returned with ``epsilon_used = 0.0``.
    """
    contour_px = np.asarray(contour_px, dtype=np.float32)

    if len(contour_px) < 10:
        return contour_px, 0.0

    # Build original polygon once
    orig_poly = Polygon(contour_px)
    orig_poly = validate_polygon(orig_poly)
    if orig_poly is None or orig_poly.is_empty or orig_poly.area < 0.1:
        return contour_px, 0.0

    tolerance = max_area_change_pct / 100.0

    lo, hi = 0.0, max_epsilon
    best_contour = contour_px
    best_epsilon = 0.0

    for _ in range(20):
        mid = (lo + hi) / 2.0
        simplified = rdp_simplify(contour_px, mid)
        if len(simplified) < 3:
            hi = mid
            continue

        simp_poly = Polygon(simplified)
        if not simp_poly.is_valid:
            simp_poly = make_valid(simp_poly)
            if simp_poly.geom_type != "Polygon":
                polys = [g for g in getattr(simp_poly, "geoms", []) if g.geom_type == "Polygon"]
                if polys:
                    simp_poly = max(polys, key=lambda p: p.area)
                else:
                    hi = mid
                    continue

        if simp_poly.is_empty:
            hi = mid
            continue

        sym_diff_area = orig_poly.symmetric_difference(simp_poly).area
        deviation = sym_diff_area / orig_poly.area

        if deviation <= tolerance:
            best_contour = simplified
            best_epsilon = mid
            lo = mid  # try larger epsilon for more reduction
        else:
            hi = mid  # too much deviation

    return best_contour, best_epsilon


def adaptive_dilate(
    contour_px: np.ndarray,
    max_area_change_pct: float = 5.0,
    max_dilation_px: float = 20.0,
) -> tuple[np.ndarray, float]:
    """Find the largest dilation that keeps area increase within tolerance.

    Binary-searches over buffer distance to maximise laser clearance while
    keeping the area increase within *max_area_change_pct* % of the original.

    Args:
        contour_px: (N, 2) contour in pixel coordinates.
        max_area_change_pct: Maximum allowed area increase (percent, default 5.0).
        max_dilation_px: Upper bound of the dilation search range (pixels).

    Returns:
        ``(dilated_contour, dilation_used)`` — the dilated (M, 2) array
        and the buffer distance that was applied.  If dilation is not possible
        within tolerance, the original contour is returned with
        ``dilation_used = 0.0``.
    """
    contour_px = np.asarray(contour_px, dtype=np.float32)

    if len(contour_px) < 3:
        return contour_px, 0.0

    orig_poly = Polygon(contour_px)
    orig_poly = validate_polygon(orig_poly)
    if orig_poly is None or orig_poly.is_empty or orig_poly.area < 0.1:
        return contour_px, 0.0

    tolerance = max_area_change_pct / 100.0

    lo, hi = 0.0, max_dilation_px
    best_contour = contour_px
    best_dilation = 0.0

    for _ in range(20):
        mid = (lo + hi) / 2.0
        dilated_poly = orig_poly.buffer(mid)

        if dilated_poly.is_empty:
            hi = mid
            continue

        if dilated_poly.geom_type == "MultiPolygon":
            dilated_poly = max(dilated_poly.geoms, key=lambda p: p.area)

        area_increase = (dilated_poly.area - orig_poly.area) / orig_poly.area

        if area_increase <= tolerance:
            best_contour = np.array(dilated_poly.exterior.coords)[:-1].astype(np.float32)
            best_dilation = mid
            lo = mid  # try larger dilation for more clearance
        else:
            hi = mid  # too much area increase

    return best_contour, best_dilation


def validate_polygon(poly: Polygon) -> Polygon | None:
    """
    Validate and fix a Shapely polygon.

    Handles:
    - Self-intersecting polygons
    - MultiPolygon results (takes largest)
    - GeometryCollection results (takes largest polygon)
    - Empty or tiny polygons

    Args:
        poly: Shapely Polygon object

    Returns:
        Valid Polygon or None if unfixable
    """
    if poly.is_valid:
        return poly

    # Try to fix invalid polygon
    poly = make_valid(poly)

    if poly.geom_type == "Polygon":
        return poly
    elif poly.geom_type == "MultiPolygon":
        # Take the largest polygon
        return max(poly.geoms, key=lambda p: p.area)
    elif poly.geom_type == "GeometryCollection":
        # Extract polygons and take the largest
        polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
        if polys:
            return max(polys, key=lambda p: p.area)
        return None
    else:
        return None


def dilate_contour(
    contour: np.ndarray, dilation: float, _poly: Polygon = None
) -> np.ndarray | None:
    if len(contour) < 3:
        return None

    if _poly is None:
        _poly = Polygon(contour)
        _poly = validate_polygon(_poly)

    if _poly is None or _poly.is_empty or _poly.area < 0.1:
        return None

    poly_dilated = _poly.buffer(dilation)

    if poly_dilated.is_empty:
        return None

    if poly_dilated.geom_type == "MultiPolygon":
        poly_dilated = max(poly_dilated.geoms, key=lambda p: p.area)

    coords = np.array(poly_dilated.exterior.coords)[:-1]

    return coords


def erode_contour(contour: np.ndarray, erosion: float, _poly: Polygon = None) -> np.ndarray | None:
    if erosion <= 0:
        return contour
    if len(contour) < 3:
        return None

    if _poly is None:
        _poly = Polygon(contour)
        _poly = validate_polygon(_poly)

    if _poly is None or _poly.is_empty or _poly.area < 0.1:
        return None

    poly_eroded = _poly.buffer(-erosion)

    if poly_eroded.is_empty:
        return None

    if poly_eroded.geom_type == "MultiPolygon":
        poly_eroded = max(poly_eroded.geoms, key=lambda p: p.area)

    if poly_eroded.area < 0.1:
        return None

    coords = np.array(poly_eroded.exterior.coords)[:-1]
    return coords


def erode_contour_percent(contour: np.ndarray, erode_pct: float) -> np.ndarray | None:
    """
    Erode a contour by a percentage of its characteristic size (sqrt(area)).

    Args:
        contour: Contour points, shape (N, 2)
        erode_pct: Erosion fraction (e.g. 0.05 = 5% of sqrt(area))

    Returns:
        Eroded contour points, or None if the polygon collapsed.
    """
    if erode_pct <= 0:
        return contour
    if len(contour) < 3:
        return None

    poly = Polygon(contour)
    poly = validate_polygon(poly)

    if poly is None or poly.is_empty or poly.area < 0.1:
        return None

    erosion = np.sqrt(poly.area) * erode_pct

    # Apply erosion directly (avoid re-creating Polygon in erode_contour)
    poly_eroded = poly.buffer(-erosion)
    if poly_eroded.is_empty:
        return None
    if poly_eroded.geom_type == "MultiPolygon":
        poly_eroded = max(poly_eroded.geoms, key=lambda p: p.area)
    if poly_eroded.area < 0.1:
        return None
    return np.array(poly_eroded.exterior.coords)[:-1]


def process_contour(
    contour_px: list[list[float]],
    pixel_size_um: float = None,
    dilation_um: float = DEFAULT_DILATION_UM,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    erosion_um: float = 0.0,
    erode_pct: float = 0.0,
    max_area_change_pct: float | None = None,
    max_dilation_area_pct: float | None = None,
    return_stats: bool = False,
) -> np.ndarray | None | tuple[np.ndarray | None, dict[str, Any]]:
    """
    Process a single contour: validate, simplify, optionally dilate/erode.

    Processing order (all in pixel space to avoid precision loss):
    1. Simplify via RDP (adaptive or fixed epsilon)
    2. Dilate in pixel space (laser buffer for LMD)
    3. Erode in pixel space (if erosion_um > 0 or erode_pct > 0)
    4. Convert to micrometers only at the end

    Args:
        contour_px: Contour points in pixels, list of [x, y] pairs
        pixel_size_um: Pixel size in micrometers
        dilation_um: Dilation amount in micrometers (default 0.5).
            Ignored when *max_dilation_area_pct* is set.
        rdp_epsilon: RDP simplification epsilon in pixels (default 5).
            Ignored when *max_area_change_pct* is set.
        erosion_um: Erosion amount in micrometers (default 0.0 = no erosion).
            Applied AFTER simplification and dilation, in pixel space.
        erode_pct: Erosion as fraction of sqrt(area) (default 0.0 = no erosion).
            Applied AFTER simplification and dilation. If both erosion_um and
            erode_pct are set, erosion_um takes priority.
        max_area_change_pct: When set, use adaptive RDP that binary-searches
            for the largest epsilon keeping symmetric-difference deviation
            within this percentage of the original area.  Overrides
            *rdp_epsilon*.
        max_dilation_area_pct: When set, use adaptive dilation that
            binary-searches for the largest buffer distance keeping area
            increase within this percentage.  Overrides *dilation_um*.
        return_stats: If True, return (contour, stats_dict)

    Returns:
        Processed contour in micrometers as numpy array, or None if invalid.
        If return_stats=True, returns (contour, stats_dict) where stats contains:
            - points_before: original point count
            - points_after: simplified point count
            - area_before_um2: original area
            - area_after_um2: final area
    """
    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required — must come from CZI metadata")
    stats = {
        "points_before": 0,
        "points_after": 0,
        "area_before_um2": 0,
        "area_after_um2": 0,
        "valid": False,
    }

    if contour_px is None or len(contour_px) < 3:
        if return_stats:
            return None, stats
        return None

    contour_px = np.array(contour_px)
    stats["points_before"] = len(contour_px)

    # Construct pixel polygon once, reuse for area and dilation
    px_poly = Polygon(contour_px)
    px_poly = validate_polygon(px_poly)
    if px_poly is not None and not px_poly.is_empty:
        stats["area_before_um2"] = px_poly.area * pixel_size_um * pixel_size_um

    # All operations in pixel space to avoid precision loss from
    # repeated px->um->px conversions.

    # Step 1: Simplify — adaptive (symmetric-difference-bounded) or fixed epsilon
    if max_area_change_pct is not None and max_area_change_pct > 0:
        simplified_px, _eps = adaptive_rdp_simplify(
            contour_px, max_area_change_pct=max_area_change_pct
        )
    else:
        simplified_px = rdp_simplify(contour_px, rdp_epsilon)

    # Step 2: Dilate in pixel space (laser buffer).
    # Adaptive mode: binary-search for largest buffer within area tolerance.
    # Fixed mode: apply dilation_um directly.
    # If dilation fails (e.g. tiny polygon), we keep the undilated contour
    # rather than returning None — having a contour without laser buffer is
    # better than losing the detection entirely at LMD export.
    if max_dilation_area_pct is not None and max_dilation_area_pct > 0:
        dilated_px, _dil = adaptive_dilate(simplified_px, max_area_change_pct=max_dilation_area_pct)
        simplified_px = dilated_px
    else:
        dilation_px = dilation_um / pixel_size_um
        if dilation_px > 0:
            dilated_px = dilate_contour(simplified_px, dilation_px)
            if dilated_px is not None:
                simplified_px = dilated_px

    # Step 3: Erode in pixel space (if requested)
    if erosion_um > 0:
        erosion_px = erosion_um / pixel_size_um
        eroded_px = erode_contour(simplified_px, erosion_px)
        if eroded_px is None:
            if return_stats:
                return None, stats
            return None
        simplified_px = eroded_px
    elif erode_pct > 0:
        eroded_px = erode_contour_percent(simplified_px, erode_pct)
        if eroded_px is None:
            if return_stats:
                return None, stats
            return None
        simplified_px = eroded_px

    if len(simplified_px) < 3:
        if return_stats:
            return None, stats
        return None

    stats["points_after"] = len(simplified_px)

    # Calculate final area in pixel space (fix self-intersecting post-RDP polygons)
    final_poly_px = Polygon(simplified_px)
    if not final_poly_px.is_valid:
        final_poly_px = make_valid(final_poly_px)
        if final_poly_px.geom_type != "Polygon":
            polys = [g for g in getattr(final_poly_px, "geoms", []) if g.geom_type == "Polygon"]
            if polys:
                final_poly_px = max(polys, key=lambda p: p.area)
    if not final_poly_px.is_empty:
        stats["area_after_um2"] = final_poly_px.area * pixel_size_um * pixel_size_um

    # Convert to microns only at the end
    simplified_um = simplified_px * pixel_size_um

    stats["valid"] = True

    if return_stats:
        return simplified_um, stats
    return simplified_um


def process_contours_batch(
    contours_px: list[list[list[float]]],
    pixel_size_um: float = None,
    dilation_um: float = DEFAULT_DILATION_UM,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    erosion_um: float = 0.0,
    erode_pct: float = 0.0,
    max_area_change_pct: float | None = None,
    max_dilation_area_pct: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Process a batch of contours with statistics.

    Args:
        contours_px: List of contours, each a list of [x, y] points in pixels
        pixel_size_um: Pixel size in micrometers
        dilation_um: Dilation amount in micrometers
        rdp_epsilon: RDP simplification epsilon in pixels
        erosion_um: Erosion amount in micrometers (default 0.0)
        erode_pct: Erosion as fraction of sqrt(area) (default 0.0)
        max_area_change_pct: Adaptive RDP tolerance (see process_contour)
        max_dilation_area_pct: Adaptive dilation tolerance (see process_contour)
        verbose: Print progress

    Returns:
        Dictionary with:
            - contours_um: List of processed contours (None for invalid)
            - valid_count: Number of successfully processed contours
            - skipped_count: Number of invalid/skipped contours
            - total_points_before: Sum of original point counts
            - total_points_after: Sum of simplified point counts
            - point_reduction_pct: Percentage reduction in points
            - mean_area_before_um2: Mean original area
            - mean_area_after_um2: Mean final area
            - area_change_pct: Percentage increase in area (from dilation)
    """
    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required — must come from CZI metadata")
    results = {
        "contours_um": [],
        "valid_count": 0,
        "skipped_count": 0,
        "total_points_before": 0,
        "total_points_after": 0,
        "point_reduction_pct": 0,
        "mean_area_before_um2": 0,
        "mean_area_after_um2": 0,
        "area_change_pct": 0,
    }

    areas_before = []
    areas_after = []

    for i, contour_px in enumerate(contours_px):
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"  Processing contour {i + 1}/{len(contours_px)}...")

        processed, stats = process_contour(
            contour_px,
            pixel_size_um=pixel_size_um,
            dilation_um=dilation_um,
            rdp_epsilon=rdp_epsilon,
            erosion_um=erosion_um,
            erode_pct=erode_pct,
            max_area_change_pct=max_area_change_pct,
            max_dilation_area_pct=max_dilation_area_pct,
            return_stats=True,
        )

        results["contours_um"].append(processed)
        results["total_points_before"] += stats["points_before"]
        results["total_points_after"] += stats["points_after"]

        if stats["valid"]:
            results["valid_count"] += 1
            if stats["area_before_um2"] > 0:
                areas_before.append(stats["area_before_um2"])
            if stats["area_after_um2"] > 0:
                areas_after.append(stats["area_after_um2"])
        else:
            results["skipped_count"] += 1

    # Calculate summary stats
    if results["total_points_before"] > 0:
        results["point_reduction_pct"] = (
            1 - results["total_points_after"] / results["total_points_before"]
        ) * 100

    if areas_before:
        results["mean_area_before_um2"] = np.mean(areas_before)
    if areas_after:
        results["mean_area_after_um2"] = np.mean(areas_after)

    if results["mean_area_before_um2"] > 0:
        results["area_change_pct"] = (
            results["mean_area_after_um2"] / results["mean_area_before_um2"] - 1
        ) * 100

    return results


# Convenience function for getting contours from detections
def process_detection_contours(
    detections: list[dict],
    contour_key: str = "outer_contour_global",
    pixel_size_um: float = None,
    dilation_um: float = DEFAULT_DILATION_UM,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    erosion_um: float = 0.0,
    erode_pct: float = 0.0,
    max_area_change_pct: float | None = None,
    max_dilation_area_pct: float | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray | None], dict[str, Any]]:
    """
    Process contours from a list of detection dictionaries.

    Args:
        detections: List of detection dicts with contour data
        contour_key: Key to access contour in each detection
        pixel_size_um: Pixel size in micrometers
        dilation_um: Dilation amount in micrometers
        rdp_epsilon: RDP simplification epsilon in pixels
        erosion_um: Erosion amount in micrometers (default 0.0)
        erode_pct: Erosion as fraction of sqrt(area) (default 0.0)
        max_area_change_pct: Adaptive RDP tolerance (see process_contour)
        max_dilation_area_pct: Adaptive dilation tolerance (see process_contour)
        verbose: Print progress

    Returns:
        Tuple of (processed_contours, stats_dict)
    """
    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required — must come from CZI metadata")
    contours_px = [d.get(contour_key) for d in detections]

    if verbose:
        logger.info(f"Processing {len(contours_px)} contours...")
        if max_dilation_area_pct is not None and max_dilation_area_pct > 0:
            logger.info(f"  Adaptive dilation: max {max_dilation_area_pct:.1f}% area increase")
        else:
            logger.info(f"  Dilation: +{dilation_um} um")
        if max_area_change_pct is not None and max_area_change_pct > 0:
            logger.info(f"  Adaptive RDP: max {max_area_change_pct:.1f}% shape deviation")
        else:
            logger.info(f"  RDP epsilon: {rdp_epsilon} px")
        if erosion_um > 0:
            logger.info(f"  Erosion: -{erosion_um} um")
        if erode_pct > 0:
            logger.info(f"  Erosion: {erode_pct*100:.1f}% of sqrt(area)")

    results = process_contours_batch(
        contours_px,
        pixel_size_um=pixel_size_um,
        dilation_um=dilation_um,
        rdp_epsilon=rdp_epsilon,
        erosion_um=erosion_um,
        erode_pct=erode_pct,
        max_area_change_pct=max_area_change_pct,
        max_dilation_area_pct=max_dilation_area_pct,
        verbose=verbose,
    )

    if verbose:
        logger.info(f"  Valid: {results['valid_count']}, Skipped: {results['skipped_count']}")
        logger.info(f"  Point reduction: {results['point_reduction_pct']:.1f}%")
        logger.info(
            f"  Area change: {results['mean_area_before_um2']:.1f} -> {results['mean_area_after_um2']:.1f} um2 ({results['area_change_pct']:+.1f}%)"
        )

    return results["contours_um"], results
