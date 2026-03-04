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

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any
from shapely.geometry import Polygon
from shapely.validation import make_valid

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# Default parameters
# Removed: DEFAULT_PIXEL_SIZE_UM — pixel_size_um must come from CZI metadata
DEFAULT_DILATION_UM = 0.5      # Dilate by 0.5um
DEFAULT_RDP_EPSILON = 5        # RDP epsilon in pixels


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


def validate_polygon(poly: Polygon) -> Optional[Polygon]:
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

    if poly.geom_type == 'Polygon':
        return poly
    elif poly.geom_type == 'MultiPolygon':
        # Take the largest polygon
        return max(poly.geoms, key=lambda p: p.area)
    elif poly.geom_type == 'GeometryCollection':
        # Extract polygons and take the largest
        polys = [g for g in poly.geoms if g.geom_type == 'Polygon']
        if polys:
            return max(polys, key=lambda p: p.area)
        return None
    else:
        return None


def dilate_contour(contour: np.ndarray, dilation: float,
                   _poly: Polygon = None) -> Optional[np.ndarray]:
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

    if poly_dilated.geom_type == 'MultiPolygon':
        poly_dilated = max(poly_dilated.geoms, key=lambda p: p.area)

    coords = np.array(poly_dilated.exterior.coords)[:-1]

    return coords


def erode_contour(contour: np.ndarray, erosion: float,
                  _poly: Polygon = None) -> Optional[np.ndarray]:
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

    if poly_eroded.geom_type == 'MultiPolygon':
        poly_eroded = max(poly_eroded.geoms, key=lambda p: p.area)

    if poly_eroded.area < 0.1:
        return None

    coords = np.array(poly_eroded.exterior.coords)[:-1]
    return coords


def erode_contour_percent(contour: np.ndarray, erode_pct: float) -> Optional[np.ndarray]:
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
    if poly_eroded.geom_type == 'MultiPolygon':
        poly_eroded = max(poly_eroded.geoms, key=lambda p: p.area)
    if poly_eroded.area < 0.1:
        return None
    return np.array(poly_eroded.exterior.coords)[:-1]


def process_contour(
    contour_px: List[List[float]],
    pixel_size_um: float = None,
    dilation_um: float = DEFAULT_DILATION_UM,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    erosion_um: float = 0.0,
    erode_pct: float = 0.0,
    return_stats: bool = False
) -> Optional[np.ndarray] | Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Process a single contour: validate, dilate, simplify, and optionally erode.

    Processing order (all in pixel space to avoid precision loss):
    1. Dilate in pixel space (validates/fixes polygon geometry)
    2. Apply RDP simplification in pixel space
    3. Erode in pixel space (if erosion_um > 0 or erode_pct > 0)
    4. Convert to micrometers only at the end

    Args:
        contour_px: Contour points in pixels, list of [x, y] pairs
        pixel_size_um: Pixel size in micrometers
        dilation_um: Dilation amount in micrometers (default 0.5)
        rdp_epsilon: RDP simplification epsilon in pixels (default 5)
        erosion_um: Erosion amount in micrometers (default 0.0 = no erosion).
            Applied AFTER dilation and RDP, in pixel space.
        erode_pct: Erosion as fraction of sqrt(area) (default 0.0 = no erosion).
            Applied AFTER dilation and RDP. If both erosion_um and erode_pct
            are set, erosion_um takes priority.
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
        'points_before': 0,
        'points_after': 0,
        'area_before_um2': 0,
        'area_after_um2': 0,
        'valid': False,
    }

    if contour_px is None or len(contour_px) < 3:
        if return_stats:
            return None, stats
        return None

    contour_px = np.array(contour_px)
    stats['points_before'] = len(contour_px)

    # Construct pixel polygon once, reuse for area and dilation
    px_poly = Polygon(contour_px)
    px_poly = validate_polygon(px_poly)
    if px_poly is not None and not px_poly.is_empty:
        stats['area_before_um2'] = px_poly.area * pixel_size_um * pixel_size_um

    # All operations in pixel space to avoid precision loss from
    # repeated px->um->px conversions.
    # Step 1: Dilate in pixel space
    dilation_px = dilation_um / pixel_size_um
    dilated_px = dilate_contour(contour_px, dilation_px, _poly=px_poly)
    if dilated_px is None:
        if return_stats:
            return None, stats
        return None

    # Step 2: RDP simplify in pixel space
    simplified_px = rdp_simplify(dilated_px, rdp_epsilon)

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

    stats['points_after'] = len(simplified_px)

    # Calculate final area in pixel space (fix self-intersecting post-RDP polygons)
    final_poly_px = Polygon(simplified_px)
    if not final_poly_px.is_valid:
        final_poly_px = make_valid(final_poly_px)
        if final_poly_px.geom_type != 'Polygon':
            polys = [g for g in getattr(final_poly_px, 'geoms', []) if g.geom_type == 'Polygon']
            if polys:
                final_poly_px = max(polys, key=lambda p: p.area)
    if not final_poly_px.is_empty:
        stats['area_after_um2'] = final_poly_px.area * pixel_size_um * pixel_size_um

    # Convert to microns only at the end
    simplified_um = simplified_px * pixel_size_um

    stats['valid'] = True

    if return_stats:
        return simplified_um, stats
    return simplified_um


def process_contours_batch(
    contours_px: List[List[List[float]]],
    pixel_size_um: float = None,
    dilation_um: float = DEFAULT_DILATION_UM,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    erosion_um: float = 0.0,
    erode_pct: float = 0.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Process a batch of contours with statistics.

    Args:
        contours_px: List of contours, each a list of [x, y] points in pixels
        pixel_size_um: Pixel size in micrometers
        dilation_um: Dilation amount in micrometers
        rdp_epsilon: RDP simplification epsilon in pixels
        erosion_um: Erosion amount in micrometers (default 0.0)
        erode_pct: Erosion as fraction of sqrt(area) (default 0.0)
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
        'contours_um': [],
        'valid_count': 0,
        'skipped_count': 0,
        'total_points_before': 0,
        'total_points_after': 0,
        'point_reduction_pct': 0,
        'mean_area_before_um2': 0,
        'mean_area_after_um2': 0,
        'area_change_pct': 0,
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
            return_stats=True
        )

        results['contours_um'].append(processed)
        results['total_points_before'] += stats['points_before']
        results['total_points_after'] += stats['points_after']

        if stats['valid']:
            results['valid_count'] += 1
            if stats['area_before_um2'] > 0:
                areas_before.append(stats['area_before_um2'])
            if stats['area_after_um2'] > 0:
                areas_after.append(stats['area_after_um2'])
        else:
            results['skipped_count'] += 1

    # Calculate summary stats
    if results['total_points_before'] > 0:
        results['point_reduction_pct'] = (
            1 - results['total_points_after'] / results['total_points_before']
        ) * 100

    if areas_before:
        results['mean_area_before_um2'] = np.mean(areas_before)
    if areas_after:
        results['mean_area_after_um2'] = np.mean(areas_after)

    if results['mean_area_before_um2'] > 0:
        results['area_change_pct'] = (
            results['mean_area_after_um2'] / results['mean_area_before_um2'] - 1
        ) * 100

    return results


# Convenience function for getting contours from detections
def process_detection_contours(
    detections: List[Dict],
    contour_key: str = 'outer_contour_global',
    pixel_size_um: float = None,
    dilation_um: float = DEFAULT_DILATION_UM,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    erosion_um: float = 0.0,
    erode_pct: float = 0.0,
    verbose: bool = True
) -> Tuple[List[Optional[np.ndarray]], Dict[str, Any]]:
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
        verbose: Print progress

    Returns:
        Tuple of (processed_contours, stats_dict)
    """
    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required — must come from CZI metadata")
    contours_px = [d.get(contour_key) for d in detections]

    if verbose:
        logger.info(f"Processing {len(contours_px)} contours...")
        logger.info(f"  Dilation: +{dilation_um} um")
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
        verbose=verbose
    )

    if verbose:
        logger.info(f"  Valid: {results['valid_count']}, Skipped: {results['skipped_count']}")
        logger.info(f"  Point reduction: {results['point_reduction_pct']:.1f}%")
        logger.info(f"  Area change: {results['mean_area_before_um2']:.1f} -> {results['mean_area_after_um2']:.1f} um2 ({results['area_change_pct']:+.1f}%)")

    return results['contours_um'], results
