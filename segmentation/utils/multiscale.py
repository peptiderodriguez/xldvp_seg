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

    Args:
        contour: Contour array shape (N, 1, 2) or (N, 2)
        scale_factor: The scale at which contour was detected

    Returns:
        Contour in full-resolution coordinates
    """
    if contour is None:
        return None

    scaled = contour.copy().astype(np.float64)
    scaled *= scale_factor
    return scaled.astype(np.int32)


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
    image_shape: Optional[Tuple[int, int]] = None
) -> float:
    """
    Compute Intersection over Union (IoU) between two contours.

    Uses mask-based IoU computation by rendering contours to binary masks.
    Contours are translated to local coordinates to avoid huge mask images
    when working with global mosaic coordinates.

    Args:
        contour1: First contour (N, 1, 2) or (N, 2)
        contour2: Second contour (M, 1, 2) or (M, 2)
        image_shape: (height, width) for mask rendering. If None, computed from contours.

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
    # This avoids creating huge mask images when coordinates are in global mosaic space
    min_x = min(x1, x2)
    min_y = min(y1, y2)

    c1_local = c1.copy()
    c2_local = c2.copy()
    c1_local[:, :, 0] -= min_x
    c1_local[:, :, 1] -= min_y
    c2_local[:, :, 0] -= min_x
    c2_local[:, :, 1] -= min_y

    # Determine image shape for mask rendering (now in local coordinates)
    if image_shape is None:
        local_max_x = max(x1 + w1, x2 + w2) - min_x + 10
        local_max_y = max(y1 + h1, y2 + h2) - min_y + 10
        image_shape = (local_max_y, local_max_x)

    # Create masks
    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)

    try:
        cv2.drawContours(mask1, [c1_local], -1, 1, -1)  # Filled
        cv2.drawContours(mask2, [c2_local], -1, 1, -1)  # Filled
    except cv2.error:
        # Contours may be invalid or outside image bounds
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
    prefer_finer_scale: bool = True
) -> List[Dict]:
    """
    Merge detections from different scales, removing duplicates.

    When the same vessel is detected at multiple scales, keeps the
    detection from the finest scale (or coarsest if prefer_finer_scale=False).

    Args:
        detections: List of detection dicts, each with 'outer' contour
                   and 'scale_detected' field
        iou_threshold: IoU above which detections are considered duplicates
        prefer_finer_scale: If True, keep finer scale detection; else coarser

    Returns:
        Deduplicated list of detections
    """
    if not detections:
        return []

    # Sort by scale (finer first if prefer_finer_scale)
    sorted_dets = sorted(
        detections,
        key=lambda d: d.get('scale_detected', 1),
        reverse=not prefer_finer_scale
    )

    # DEBUG: Count detections with valid outer contours
    valid_outer_count = sum(1 for d in sorted_dets if d.get('outer') is not None)
    logger.info(f"DEBUG: {valid_outer_count}/{len(sorted_dets)} detections have valid 'outer' contour")

    # DEBUG: Print first detection's outer info
    if sorted_dets:
        first = sorted_dets[0]
        outer = first.get('outer')
        if outer is not None:
            logger.info(f"DEBUG: First detection outer shape: {outer.shape}, dtype: {outer.dtype}")
            logger.info(f"DEBUG: First detection outer bounds: x=[{outer[:,:,0].min()}, {outer[:,:,0].max()}], y=[{outer[:,:,1].min()}, {outer[:,:,1].max()}]")
        else:
            logger.info(f"DEBUG: First detection has no 'outer' contour. Keys: {list(first.keys())}")

    merged = []
    skipped_no_outer = 0
    duplicate_count = 0

    for det in sorted_dets:
        outer = det.get('outer')
        if outer is None:
            skipped_no_outer += 1
            continue

        is_duplicate = False

        for existing in merged:
            existing_outer = existing.get('outer')
            if existing_outer is None:
                continue

            iou = compute_iou_contours(outer, existing_outer)
            if iou > iou_threshold:
                is_duplicate = True
                duplicate_count += 1
                # Log first few duplicates for debugging
                if duplicate_count <= 5:
                    logger.info(
                        f"DEBUG: Duplicate #{duplicate_count} (IoU={iou:.3f}): "
                        f"scale {det.get('scale_detected')} vs {existing.get('scale_detected')}, "
                        f"new bounds x=[{outer[:,:,0].min()}, {outer[:,:,0].max()}], "
                        f"existing bounds x=[{existing_outer[:,:,0].min()}, {existing_outer[:,:,0].max()}]"
                    )
                break

        if not is_duplicate:
            merged.append(det)

    logger.info(f"DEBUG: Skipped {skipped_no_outer} detections with no outer contour")
    logger.info(
        f"Merged {len(detections)} detections → {len(merged)} "
        f"(removed {len(detections) - len(merged)} duplicates)"
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
    tile_y_scaled: int
) -> Dict:
    """
    Convert a detection from scaled coordinates to full resolution.

    Updates all coordinate fields in the detection dict.

    Args:
        detection: Detection dict with 'outer', 'inner', 'center' fields
        scale_factor: Scale at which detection was made
        tile_x_scaled: Tile X origin in scaled coordinates
        tile_y_scaled: Tile Y origin in scaled coordinates

    Returns:
        Detection with coordinates in full resolution
    """
    det = detection.copy()

    # Scale contours
    if 'outer' in det and det['outer'] is not None:
        det['outer'] = scale_contour(det['outer'], scale_factor)
        # Add tile offset in full-res coordinates
        det['outer'] += np.array([tile_x_scaled * scale_factor, tile_y_scaled * scale_factor])

    if 'inner' in det and det['inner'] is not None:
        det['inner'] = scale_contour(det['inner'], scale_factor)
        det['inner'] += np.array([tile_x_scaled * scale_factor, tile_y_scaled * scale_factor])

    # Scale center point
    if 'center' in det:
        cx, cy = det['center']
        det['center'] = [
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
    scales: List[int] = [8, 4, 1],
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

    # Merge across scales (finer scale takes precedence)
    merged = merge_detections_across_scales(
        all_detections,
        iou_threshold=iou_threshold,
        prefer_finer_scale=True
    )

    return merged
