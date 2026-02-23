"""
Mask-overlap deduplication for detections.

Removes duplicate detections caused by tile overlap by comparing actual mask
pixels in global coordinates. When two masks overlap significantly, the larger
detection is kept.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).
"""

import numpy as np
from scipy import ndimage

try:
    import hdf5plugin  # Register LZ4 filter for h5py
except ImportError:
    pass
import h5py

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

# Encode (x, y) as single int64 for fast numpy set intersection
_COORD_STRIDE = 300000  # Must exceed max slide dimension in pixels


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
    n_total = len(detections)
    print(f"[dedup] Starting dedup for {n_total} detections...", flush=True)

    # Load all masks and compute global bounding boxes
    det_info = []  # List of (det, global_bbox, encoded_coords)

    # Cache: tile_id -> (masks_array, find_objects_slices)
    mask_cache = {}

    for i, det in enumerate(detections):
        if i % 5000 == 0 and i > 0:
            print(f"[dedup] Loading masks: {i}/{n_total}", flush=True)

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

        # Load masks + precompute find_objects if not cached
        if tile_id not in mask_cache:
            masks_file = tiles_dir / tile_id / mask_filename
            if masks_file.exists():
                try:
                    with h5py.File(masks_file, 'r') as f:
                        masks_array = f['masks'][:]
                    slices = ndimage.find_objects(masks_array)
                    mask_cache[tile_id] = (masks_array, slices)
                except Exception as e:
                    logger.warning(f"Failed to load masks from {masks_file}: {e}")
                    mask_cache[tile_id] = None
            else:
                mask_cache[tile_id] = None

        cached = mask_cache.get(tile_id)
        if cached is None:
            det_info.append((det, None, None))
            continue

        masks_array, slices = cached

        # Extract mask coordinates via find_objects bbox (O(bbox) not O(H×W))
        if mask_label < 1 or mask_label > len(slices):
            det_info.append((det, None, None))
            continue
        sl = slices[mask_label - 1]  # find_objects is 1-indexed
        if sl is None:
            det_info.append((det, None, None))
            continue

        local_ys, local_xs = np.where(masks_array[sl] == mask_label)
        if len(local_ys) == 0:
            det_info.append((det, None, None))
            continue

        # Adjust for slice offset + convert to global coords
        global_xs = local_xs + sl[1].start + tile_x
        global_ys = local_ys + sl[0].start + tile_y

        # Compute bounding box (x_min, y_min, x_max, y_max)
        bbox = (int(global_xs.min()), int(global_ys.min()),
                int(global_xs.max()), int(global_ys.max()))

        # Encode coords as sorted int64 array for fast numpy intersection
        encoded = np.sort(global_ys.astype(np.int64) * _COORD_STRIDE + global_xs.astype(np.int64))

        det_info.append((det, bbox, encoded))

    # Free mask cache — no longer needed
    del mask_cache

    print(f"[dedup] Masks loaded. Running overlap check...", flush=True)

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
    kept_info = []  # (bbox, encoded_coords) for kept detections

    for i, (det, bbox, coords) in enumerate(det_info):
        if i % 10000 == 0 and i > 0:
            print(f"[dedup] Overlap check: {i}/{n_total} processed, {len(kept)} kept", flush=True)

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

            # Fast numpy intersection on sorted int64 arrays
            overlap = len(np.intersect1d(coords, kept_coords, assume_unique=True))
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
    print(f"[dedup] Done: {len(detections)} -> {len(kept)} ({n_removed} removed)", flush=True)

    return kept
