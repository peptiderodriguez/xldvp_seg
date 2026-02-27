"""
Multi-scale vessel detection utilities.

Provides functions for:
- Coordinate scaling between resolutions
- IoU computation for contour deduplication
- Merging detections across scales
- Tile grid generation at different scales

Scale factors:
- 1: Full resolution (0.17 µm/px typical)
- 4: 1/4 resolution (0.69 µm/px)
- 8: 1/8 resolution (1.38 µm/px)
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# Scale-specific detection parameters
SCALE_PARAMS = {
    # Ultra-coarse scale (1/64x): Aorta and major vessels
    64: {
        'min_diameter_um': 1000,
        'max_diameter_um': 10000,
        'min_circularity': 0.10,
        'min_ring_completeness': 0.15,
        'description': 'Aorta and major vessels (>1000 µm)',
    },
    # Very coarse scale (1/32x): Large arteries
    32: {
        'min_diameter_um': 500,
        'max_diameter_um': 5000,
        'min_circularity': 0.12,
        'min_ring_completeness': 0.20,
        'description': 'Large arteries (500-5000 µm)',
    },
    # Coarse scale (1/16x): Very large vessels only
    16: {
        'min_diameter_um': 200,
        'max_diameter_um': 3000,
        'min_circularity': 0.15,
        'min_ring_completeness': 0.25,
        'description': 'Very large arteries (>200 µm)',
    },
    # Coarse scale (1/8x): Large vessels only
    8: {
        'min_diameter_um': 100,
        'max_diameter_um': 1000,
        'min_circularity': 0.2,
        'min_ring_completeness': 0.3,
        'description': 'Large arteries (100-1000 µm)',
    },
    # Medium scale (1/4x): Medium vessels
    4: {
        'min_diameter_um': 50,
        'max_diameter_um': 300,
        'min_circularity': 0.2,
        'min_ring_completeness': 0.3,
        'description': 'Medium vessels (50-300 µm)',
    },
    # Medium-fine scale (1/2x): Medium-small vessels
    2: {
        'min_diameter_um': 20,
        'max_diameter_um': 150,
        'min_circularity': 0.2,
        'min_ring_completeness': 0.3,
        'description': 'Medium-small vessels (20-150 µm)',
    },
    # Fine scale (1x): Small vessels and capillaries
    1: {
        'min_diameter_um': 5,
        'max_diameter_um': 75,
        'min_circularity': 0.15,
        'min_ring_completeness': 0.25,
        'description': 'Small vessels and capillaries (5-75 µm)',
    },
}


def get_scale_params(scale_factor: int) -> Dict:
    """
    Get detection parameters appropriate for a given scale.

    Args:
        scale_factor: Downsampling factor (1, 4, or 8)

    Returns:
        Dict of detection parameters
    """
    if scale_factor in SCALE_PARAMS:
        return SCALE_PARAMS[scale_factor].copy()

    # For intermediate scales, interpolate
    if scale_factor >= 8:
        return SCALE_PARAMS[8].copy()
    elif scale_factor >= 4:
        return SCALE_PARAMS[4].copy()
    else:
        return SCALE_PARAMS[1].copy()


def scale_contour(contour: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Scale contour coordinates from scaled space to full resolution.

    Returns float64 to preserve precision. The caller is responsible for
    casting to int32 after all coordinate transformations (e.g., adding
    tile offsets) are complete. This avoids truncation errors that are
    amplified at high scale factors (e.g., up to 64 px error at scale 64).

    Args:
        contour: Contour array shape (N, 1, 2) or (N, 2)
        scale_factor: The scale at which contour was detected

    Returns:
        Contour in full-resolution coordinates (float64)
    """
    if contour is None:
        return None

    contour = np.asarray(contour)
    scaled = contour.copy().astype(np.float64)
    scaled *= scale_factor
    return scaled


def scale_point(point: Tuple[float, float], scale_factor: int) -> Tuple[float, float]:
    """
    Scale a point from scaled space to full resolution.

    Args:
        point: (x, y) tuple in scaled coordinates
        scale_factor: The scale at which point was detected

    Returns:
        (x, y) in full-resolution coordinates
    """
    return (point[0] * scale_factor, point[1] * scale_factor)


def compute_iou_contours(
    contour1: np.ndarray,
    contour2: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    max_render_size: int = 512,
) -> float:
    """
    Compute Intersection over Union (IoU) between two contours.

    Uses mask-based IoU computation by rendering contours to binary masks.
    Contours are translated to local coordinates and downsampled so the
    combined bounding box fits within max_render_size pixels on each axis.
    IoU is scale-invariant, so downsampling preserves accuracy while
    keeping mask allocation bounded (max ~512x512 = 262K pixels vs
    potentially millions for large vessels at full resolution).

    Args:
        contour1: First contour (N, 1, 2) or (N, 2)
        contour2: Second contour (M, 1, 2) or (M, 2)
        image_shape: (height, width) for mask rendering. If None, computed from contours.
        max_render_size: Maximum pixels per axis for mask rendering (default 512).

    Returns:
        IoU value between 0 and 1
    """
    if contour1 is None or contour2 is None:
        return 0.0

    # Ensure contours are in correct format
    c1 = contour1.reshape(-1, 1, 2) if contour1.ndim == 2 else contour1
    c2 = contour2.reshape(-1, 1, 2) if contour2.ndim == 2 else contour2

    # Compute bounding boxes
    x1, y1, w1, h1 = cv2.boundingRect(c1)
    x2, y2, w2, h2 = cv2.boundingRect(c2)

    # Quick rejection if bounding boxes don't overlap
    if (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
        return 0.0

    # Translate contours to local coordinates (origin at min of both bounding boxes)
    min_x = min(x1, x2)
    min_y = min(y1, y2)

    local_w = max(x1 + w1, x2 + w2) - min_x + 2
    local_h = max(y1 + h1, y2 + h2) - min_y + 2

    c1_local = c1.copy().astype(np.float64)
    c2_local = c2.copy().astype(np.float64)
    c1_local[:, :, 0] -= min_x
    c1_local[:, :, 1] -= min_y
    c2_local[:, :, 0] -= min_x
    c2_local[:, :, 1] -= min_y

    # Downsample if combined bbox exceeds max_render_size — IoU is scale-invariant
    scale = 1.0
    if local_w > max_render_size or local_h > max_render_size:
        scale = min(max_render_size / local_w, max_render_size / local_h)
        c1_local = (c1_local * scale).astype(np.int32)
        c2_local = (c2_local * scale).astype(np.int32)
        local_w = int(local_w * scale) + 2
        local_h = int(local_h * scale) + 2
    else:
        c1_local = c1_local.astype(np.int32)
        c2_local = c2_local.astype(np.int32)

    # Always use local (possibly downsampled) dimensions — ignore image_shape
    # since contours have been translated to local coordinates
    render_shape = (local_h, local_w)

    # Create masks
    mask1 = np.zeros(render_shape, dtype=np.uint8)
    mask2 = np.zeros(render_shape, dtype=np.uint8)

    try:
        cv2.drawContours(mask1, [c1_local], -1, 1, -1)
        cv2.drawContours(mask2, [c2_local], -1, 1, -1)
    except cv2.error:
        return 0.0

    # Compute IoU
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    if union == 0:
        return 0.0

    return intersection / union


def compute_iou_from_masks(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU directly from binary masks.

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        IoU value between 0 and 1
    """
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    if union == 0:
        return 0.0

    return intersection / union


def merge_detections_across_scales(
    detections: List[Dict],
    iou_threshold: float = 0.3,
    tile_size: int = 3000,
) -> List[Dict]:
    """
    Merge detections from different scales, removing duplicates.

    Sorted by area descending — the largest detection is kept when duplicates
    overlap above iou_threshold.

    Uses spatial grid indexing for O(n*k) performance instead of O(n^2),
    where k is the average number of nearby detections per grid cell.

    Args:
        detections: List of detection dicts, each with 'outer' contour
                   and 'scale_detected' field
        iou_threshold: IoU above which detections are considered duplicates
        (Always keeps larger contour area detection)
        tile_size: Tile size in pixels (grid cell = 10% of tile_size)

    Returns:
        Deduplicated list of detections
    """
    import time
    import sys
    from collections import defaultdict

    if not detections:
        return []

    # Pre-compute contours in cv2 format and bounding boxes
    prepared = []  # (det, contour_cv2, bbox, area)
    skipped_no_outer = 0
    for det in detections:
        outer = det.get('outer')
        if outer is None:
            skipped_no_outer += 1
            continue
        try:
            c = np.asarray(outer).reshape(-1, 1, 2).astype(np.int32)
            bbox = cv2.boundingRect(c)  # (x, y, w, h)
            area = abs(cv2.contourArea(c))
            prepared.append((det, c, bbox, area))
        except Exception:
            skipped_no_outer += 1

    logger.info(f"Merge: {len(prepared)} valid contours, {skipped_no_outer} skipped (no outer)")

    # Sort by area descending (largest first = highest priority)
    prepared.sort(key=lambda t: t[3], reverse=True)

    # Grid cell size = 10% of tile size. Detections larger than this span multiple
    # cells (fine — they get inserted into all cells they touch). What matters is
    # that each cell contains few detections for fast neighbor lookups.
    cell_size = max(300, tile_size // 10)
    logger.info(f"Merge: grid cell size = {cell_size}px (tile_size={tile_size})")

    def _bbox_cells(bbox):
        """Return set of grid cell keys that a bounding box overlaps."""
        x, y, w, h = bbox
        x1, y1 = x // cell_size, y // cell_size
        x2, y2 = (x + w) // cell_size, (y + h) // cell_size
        cells = set()
        for gx in range(x1, x2 + 1):
            for gy in range(y1, y2 + 1):
                cells.add((gx, gy))
        return cells

    def _bboxes_overlap(b1, b2):
        """Fast bounding box overlap check."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    # Spatial grid: cell_key -> list of (index_in_merged, bbox, contour)
    grid = defaultdict(list)
    merged = []
    merged_contours = []  # parallel list of (contour_cv2, bbox)
    duplicate_count = 0
    iou_checks = 0
    
    # Progress tracking for large merge operations
    total_start = time.time()
    progress_interval = max(100, len(prepared) // 100) if len(prepared) > 0 else 100
    max_detections_per_cell = 0
    cell_with_max = None

    for det_idx, (det, contour, bbox, area) in enumerate(prepared):
        cells = _bbox_cells(bbox)

        # Collect candidate indices from overlapping grid cells (deduplicate)
        candidate_indices = set()
        for cell in cells:
            cell_size_check = len(grid[cell])
            if cell_size_check > max_detections_per_cell:
                max_detections_per_cell = cell_size_check
                cell_with_max = cell
            for idx in grid[cell]:
                candidate_indices.add(idx)

        is_duplicate = False
        for idx in candidate_indices:
            existing_c, existing_bbox = merged_contours[idx]
            # Fast bbox pre-check before expensive IoU
            if not _bboxes_overlap(bbox, existing_bbox):
                continue
            iou_checks += 1
            iou = compute_iou_contours(contour, existing_c)
            if iou > iou_threshold:
                is_duplicate = True
                duplicate_count += 1
                if duplicate_count <= 5:
                    existing_area = abs(cv2.contourArea(existing_c))
                    logger.debug(
                        f"Duplicate #{duplicate_count} (IoU={iou:.3f}): "
                        f"dropping area={area:.0f}px (scale 1/{det.get('scale_detected', '?')}x), "
                        f"keeping area={existing_area:.0f}px (scale 1/{merged[idx].get('scale_detected', '?')}x)"
                    )
                break

        if not is_duplicate:
            new_idx = len(merged)
            merged.append(det)
            merged_contours.append((contour, bbox))
            for cell in cells:
                grid[cell].append(new_idx)
        
        # Progress logging
        if (det_idx + 1) % progress_interval == 0:
            elapsed = time.time() - total_start
            rate = (det_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = len(prepared) - (det_idx + 1)
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"Merge progress: {det_idx + 1}/{len(prepared)} ({100.0*(det_idx+1)/len(prepared):.1f}%) — "
                f"{iou_checks} IoU checks, {len(merged)} kept, {elapsed:.1f}s elapsed, ETA {eta:.1f}s"
            )

    logger.debug(f"Skipped {skipped_no_outer} detections with no outer contour")
    
    # Summary statistics
    elapsed_total = time.time() - total_start
    avg_checks_per_det = iou_checks / len(prepared) if len(prepared) > 0 else 0
    logger.info(
        f"Merge completed: {iou_checks} total IoU checks in {elapsed_total:.1f}s "
        f"({iou_checks/elapsed_total:.0f} checks/s, {avg_checks_per_det:.1f} checks/detection)"
    )
    logger.info(
        f"Merged {len(detections)} detections → {len(merged)} "
        f"(removed {len(detections) - len(merged)} duplicates, "
        f"{iou_checks} IoU checks vs {len(prepared) * (len(prepared) - 1) // 2} brute-force)"
    )

    return merged


def generate_tile_grid_at_scale(
    mosaic_width: int,
    mosaic_height: int,
    tile_size: int,
    scale_factor: int,
    overlap: int = 0
) -> List[Tuple[int, int]]:
    """
    Generate tile grid coordinates at a specific scale.

    The returned coordinates are in the SCALED coordinate system.
    To get full-res coordinates, multiply by scale_factor.

    Args:
        mosaic_width: Full-resolution mosaic width
        mosaic_height: Full-resolution mosaic height
        tile_size: Tile size in scaled pixels
        scale_factor: Downsampling factor (1, 4, 8, etc.)
        overlap: Overlap between tiles in scaled pixels

    Returns:
        List of (tile_x, tile_y) in scaled coordinates
    """
    # Compute scaled mosaic dimensions
    scaled_width = mosaic_width // scale_factor
    scaled_height = mosaic_height // scale_factor

    tiles = []
    step = tile_size - overlap

    y = 0
    while y < scaled_height:
        x = 0
        while x < scaled_width:
            tiles.append((x, y))
            x += step
        y += step

    logger.debug(
        f"Generated {len(tiles)} tiles at scale 1/{scale_factor}x "
        f"(tile_size={tile_size}, overlap={overlap})"
    )

    return tiles


def convert_detection_to_full_res(
    detection: Dict,
    scale_factor: int,
    tile_x_scaled: int,
    tile_y_scaled: int,
    smooth: bool = True,
    smooth_base_factor: float = 3.0,
) -> Dict:
    """
    Convert a detection from scaled coordinates to full resolution.

    Updates all coordinate fields in the detection dict. Optionally applies
    B-spline smoothing AFTER upscaling to remove stair-step artifacts that
    arise from coarse-scale detection (e.g., 1/32x contours have 32px jumps).

    The smoothing factor scales with the detection scale: at 1x no smoothing
    is needed, at 1/32x the factor is base * 32 to handle 32px stair-steps.

    Args:
        detection: Detection dict with 'outer', 'inner', 'center' fields
        scale_factor: Scale at which detection was made
        tile_x_scaled: Tile X origin in scaled coordinates
        tile_y_scaled: Tile Y origin in scaled coordinates
        smooth: Whether to apply B-spline smoothing (default True)
        smooth_base_factor: Base smoothing factor, scaled by scale_factor

    Returns:
        Detection with coordinates in full resolution
    """
    det = detection.copy()

    # Scale contours — stay in float64 until after offset, then cast to int32.
    # This avoids truncation errors amplified by large scale factors (up to
    # scale_factor pixels of displacement when casting before offset).
    tile_offset = np.array([tile_x_scaled * scale_factor, tile_y_scaled * scale_factor], dtype=np.float64)

    # Import smoothing function (lazy to avoid circular imports)
    _smooth_fn = None
    if smooth and scale_factor > 1:
        try:
            from segmentation.detection.strategies.vessel import smooth_contour_spline
            _smooth_fn = smooth_contour_spline
        except ImportError:
            pass

    # Smoothing factor scales with detection scale — coarser = more smoothing
    effective_smoothing = smooth_base_factor * scale_factor

    # Arc contours are open curves — don't force periodic spline closure
    is_arc = det.get('is_arc', False)
    contour_closed = not is_arc

    if 'outer' in det and det['outer'] is not None:
        det['outer'] = scale_contour(det['outer'], scale_factor)
        det['outer'] += tile_offset
        if _smooth_fn is not None:
            det['outer'] = _smooth_fn(det['outer'], smoothing=effective_smoothing, force_closed=contour_closed)
        det['outer'] = np.asarray(det['outer']).astype(np.int32)

    if 'inner' in det and det['inner'] is not None:
        det['inner'] = scale_contour(det['inner'], scale_factor)
        det['inner'] += tile_offset
        if _smooth_fn is not None:
            det['inner'] = _smooth_fn(det['inner'], smoothing=effective_smoothing, force_closed=contour_closed)
        det['inner'] = np.asarray(det['inner']).astype(np.int32)

    # Scale center/centroid point
    for key in ('center', 'centroid'):
        if key in det:
            cx, cy = det[key]
            det[key] = [
                (cx + tile_x_scaled) * scale_factor,
                (cy + tile_y_scaled) * scale_factor
            ]

    # Scale center points inside features dict
    feats = det.get('features', {})
    if isinstance(feats, dict):
        for center_key in ('outer_center', 'inner_center'):
            if center_key in feats and feats[center_key] is not None:
                cx, cy = feats[center_key]
                feats[center_key] = [
                    (cx + tile_x_scaled) * scale_factor,
                    (cy + tile_y_scaled) * scale_factor
                ]

    # Add scale metadata
    det['scale_detected'] = scale_factor

    return det


def run_multiscale_detection(
    tile_getter: Callable,
    detect_fn: Callable,
    mosaic_width: int,
    mosaic_height: int,
    tile_size: int = 4000,
    scales: Optional[List[int]] = None,
    pixel_size_um: float = 0.17,
    channel: int = 0,
    iou_threshold: float = 0.3,
    progress_callback: Optional[Callable] = None,
    **detect_kwargs
) -> List[Dict]:
    """
    Run vessel detection at multiple scales and merge results.

    This is the main entry point for multi-scale detection.

    Args:
        tile_getter: Function(tile_x, tile_y, tile_size, channel, scale_factor) -> ndarray
        detect_fn: Function(tile, pixel_size_um=..., **kwargs) -> List[Dict]
        mosaic_width: Full-resolution mosaic width
        mosaic_height: Full-resolution mosaic height
        tile_size: Tile size in pixels (same at all scales)
        scales: List of scale factors to use, ordered coarse to fine
        pixel_size_um: Pixel size at full resolution in µm
        channel: Channel to process
        iou_threshold: IoU threshold for deduplication
        progress_callback: Optional callback(scale, tiles_done, total_tiles)
        **detect_kwargs: Additional kwargs passed to detect_fn

    Returns:
        List of deduplicated detections in full-resolution coordinates
    """
    if scales is None:
        scales = [32, 16, 8, 4, 2]

    all_detections = []

    for scale in scales:
        scale_pixel_size = pixel_size_um * scale
        scale_params = get_scale_params(scale)

        # Get tile grid for this scale
        tiles = generate_tile_grid_at_scale(
            mosaic_width, mosaic_height, tile_size, scale
        )

        logger.info(
            f"Scale 1/{scale}x: {len(tiles)} tiles, "
            f"pixel_size={scale_pixel_size:.3f} µm, "
            f"target: {scale_params['description']}"
        )

        for i, (tile_x, tile_y) in enumerate(tiles):
            # Get tile at this scale
            tile = tile_getter(tile_x, tile_y, tile_size, channel, scale)
            if tile is None:
                continue

            # Detect with scale-appropriate parameters
            detections = detect_fn(
                tile,
                pixel_size_um=scale_pixel_size,
                min_diameter_um=scale_params['min_diameter_um'],
                max_diameter_um=scale_params['max_diameter_um'],
                **detect_kwargs
            )

            # Convert to full resolution coordinates
            for det in detections:
                det_fullres = convert_detection_to_full_res(
                    det, scale, tile_x, tile_y
                )
                all_detections.append(det_fullres)

            if progress_callback:
                progress_callback(scale, i + 1, len(tiles))

        logger.info(
            f"Scale 1/{scale}x: Found {len([d for d in all_detections if d.get('scale_detected') == scale])} detections"
        )

    # Merge across scales (keep larger/more complete detection)
    merged = merge_detections_across_scales(
        all_detections,
        iou_threshold=iou_threshold,
        tile_size=tile_size,
    )

    return merged
