"""
Mask-overlap deduplication for detections.

Removes duplicate detections caused by tile overlap by comparing actual mask
pixels in global coordinates. When two masks overlap significantly, the larger
detection is kept.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).
"""

import numpy as np

try:
    import hdf5plugin  # Register LZ4 filter for h5py
except ImportError:
    pass
import h5py

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def deduplicate_by_mask_overlap(detections, tiles_dir, min_overlap_fraction=0.1,
                                mask_filename=None, sort_by='area'):
    """Remove duplicate detections by checking actual mask pixel overlap.

    For each pair of detections, checks if their masks overlap in global
    coordinates. When masks overlap by more than min_overlap_fraction of
    the smaller mask, keeps the higher-priority detection.

    Args:
        detections: List of detection dicts with 'tile_origin', 'mask_label', 'features'
        tiles_dir: Path to tiles directory containing mask h5 files
        min_overlap_fraction: Minimum overlap (fraction of smaller mask) to consider duplicates
        mask_filename: Name of the mask HDF5 file in each tile dir (e.g. 'nmj_masks.h5').
            Must be provided explicitly by the caller.
        sort_by: Priority for keeping detections during dedup.
            'area' (default): larger detections win (backward-compat)
            'confidence': higher-scoring detections win (uses rf_prediction/score/sam2_score)

    Returns:
        List of deduplicated detections
    """
    from pathlib import Path

    if not detections:
        return []

    if mask_filename is None:
        raise ValueError(
            "mask_filename must be provided (e.g. '{cell_type}_masks.h5'). "
            "No default is assumed to avoid cell-type-specific assumptions."
        )

    tiles_dir = Path(tiles_dir)

    # Load all masks and compute global bounding boxes
    det_info = []  # List of (det, global_bbox, global_mask_coords)

    # Cache loaded mask files
    mask_cache = {}

    for det in detections:
        tile_origin = tuple(det.get('tile_origin', [0, 0]))
        tile_x, tile_y = tile_origin
        tile_id = f"tile_{tile_x}_{tile_y}"
        mask_label = det.get('mask_label')

        if mask_label is None:
            # Try to extract from ID as fallback
            det_id = det.get('id', '')
            try:
                mask_label = int(det_id.split('_')[-1])
            except (ValueError, IndexError):
                det_info.append((det, None, None))
                continue

        # Load masks if not cached
        if tile_id not in mask_cache:
            masks_file = tiles_dir / tile_id / mask_filename
            if masks_file.exists():
                try:
                    with h5py.File(masks_file, 'r') as f:
                        mask_cache[tile_id] = f['masks'][:]
                except Exception as e:
                    logger.warning(f"Failed to load masks from {masks_file}: {e}")
                    mask_cache[tile_id] = None
            else:
                mask_cache[tile_id] = None

        masks = mask_cache.get(tile_id)
        if masks is None:
            det_info.append((det, None, None))
            continue

        # Get mask pixels in local coords
        local_ys, local_xs = np.where(masks == mask_label)
        if len(local_ys) == 0:
            det_info.append((det, None, None))
            continue

        # Convert to global coords
        global_xs = local_xs + tile_x
        global_ys = local_ys + tile_y

        # Compute bounding box (x_min, y_min, x_max, y_max)
        bbox = (int(global_xs.min()), int(global_ys.min()),
                int(global_xs.max()), int(global_ys.max()))

        # Store global mask coords as set for fast overlap checking
        global_coords = set(zip(global_xs.tolist(), global_ys.tolist()))

        det_info.append((det, bbox, global_coords))

    # Sort by priority descending (keep higher-priority ones)
    if sort_by == 'confidence':
        def _confidence_key(item):
            det = item[0]
            score = det.get('rf_prediction')
            if score is None:
                score = det.get('score')
            if score is None:
                score = det.get('features', {}).get('sam2_score', 0)
            return score if score is not None else 0
        det_info.sort(key=_confidence_key, reverse=True)
    else:
        det_info.sort(key=lambda x: x[0].get('features', {}).get('area', 0), reverse=True)

    # Greedy deduplication: keep detection if it doesn't significantly overlap with any kept detection
    kept = []
    kept_info = []  # (bbox, global_coords) for kept detections

    for det, bbox, coords in det_info:
        if bbox is None or coords is None:
            # Can't check overlap, keep it
            kept.append(det)
            continue

        # Check overlap with all kept detections
        is_duplicate = False
        for kept_bbox, kept_coords in kept_info:
            # Quick bbox overlap check first
            if (bbox[0] > kept_bbox[2] or bbox[2] < kept_bbox[0] or
                    bbox[1] > kept_bbox[3] or bbox[3] < kept_bbox[1]):
                continue

            # Check actual pixel overlap
            overlap = len(coords & kept_coords)
            smaller_size = min(len(coords), len(kept_coords))

            if smaller_size > 0 and overlap / smaller_size >= min_overlap_fraction:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)
            kept_info.append((bbox, coords))

    n_removed = len(detections) - len(kept)
    if n_removed > 0:
        logger.info(f"Mask overlap dedup: {len(detections)} -> {len(kept)} ({n_removed} duplicates removed)")

    return kept
