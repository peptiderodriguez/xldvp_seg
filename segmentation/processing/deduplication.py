"""
Mask-overlap deduplication for detections.

Removes duplicate detections caused by tile overlap by comparing actual mask
pixels in global coordinates. When two masks overlap significantly, the larger
detection is kept.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Performance optimizations:
- Batch HDF5 file handles: all tile HDF5 files are opened upfront and kept
  open for the duration of mask loading, avoiding repeated open/close on GPFS.
- Set-based intersection with early exit: overlap checking uses Python sets
  with counting up to threshold, avoiding full intersection computation.
- Mask cache eviction: tiles are evicted from the mask cache once all their
  detections have been processed, reducing peak memory from O(all_tiles) to
  O(active_tiles).
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

# Default stride for encoding (x, y) as single int64 for fast numpy set
# intersection.  This is overridden dynamically in the dedup loop below when
# the actual data exceeds this value.
_COORD_STRIDE_DEFAULT = 1_000_000


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
    from collections import Counter

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

    # --- Fix 3 (part 1): Count detections per tile for cache eviction ---
    tile_det_counts = Counter()
    for det in detections:
        tile_origin = tuple(det.get('tile_origin', [0, 0]))
        tile_x, tile_y = tile_origin
        tile_id = f"tile_{tile_x}_{tile_y}"
        tile_det_counts[tile_id] += 1

    # --- Fix 1: Batch-open all HDF5 file handles upfront ---
    # Discover unique tile IDs and open their HDF5 files once, keeping handles
    # open for the duration of mask loading.  This avoids 5000+ open/close
    # round-trips on GPFS network mounts.
    unique_tile_ids = set(tile_det_counts.keys())
    h5_handles = {}  # tile_id -> open h5py.File (or None if missing/failed)
    for tile_id in unique_tile_ids:
        masks_file = tiles_dir / tile_id / mask_filename
        if masks_file.exists():
            try:
                h5_handles[tile_id] = h5py.File(masks_file, 'r')
            except Exception as e:
                logger.warning(f"Failed to open HDF5 file {masks_file}: {e}")
                h5_handles[tile_id] = None
        else:
            h5_handles[tile_id] = None
    print(f"[dedup] Opened {sum(1 for v in h5_handles.values() if v is not None)} "
          f"HDF5 files out of {len(unique_tile_ids)} tiles", flush=True)

    # Load all masks and compute global bounding boxes.
    # We collect raw (global_xs, global_ys) first, then encode after the loop
    # using a dynamically computed stride to avoid coordinate collisions on
    # mosaics wider than 1M pixels.
    det_info_raw = []  # List of (det, global_bbox, global_xs, global_ys) or (det, None, None, None)
    observed_max_x = 0  # Track actual max x coordinate across all detections
    n_maskless = 0  # Count detections without valid mask data

    # Cache: tile_id -> (masks_array, find_objects_slices)
    mask_cache = {}

    for i, det in enumerate(detections):
        if i % 5000 == 0 and i > 0:
            print(f"[dedup] Loading masks: {i}/{n_total}", flush=True)

        tile_origin = tuple(det.get('tile_origin', [0, 0]))
        tile_x, tile_y = tile_origin
        tile_id = f"tile_{tile_x}_{tile_y}"
        mask_label = det.get('mask_label')

        if mask_label is None or mask_label == 0:
            # Try to extract from ID as fallback
            det_id = det.get('id', '')
            try:
                mask_label = int(det_id.split('_')[-1])
            except (ValueError, IndexError):
                n_maskless += 1
                det_info_raw.append((det, None, None, None))
                # --- Fix 3: Decrement tile count and evict if done ---
                tile_det_counts[tile_id] -= 1
                if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                    del mask_cache[tile_id]
                continue

        # Ensure int (JSON deserialization can produce float, e.g. 3.0)
        mask_label = int(mask_label)

        # Load masks + precompute find_objects if not cached
        if tile_id not in mask_cache:
            h5f = h5_handles.get(tile_id)
            if h5f is not None:
                try:
                    masks_array = h5f['masks'][:]
                    slices = ndimage.find_objects(masks_array)
                    mask_cache[tile_id] = (masks_array, slices)
                except Exception as e:
                    logger.warning(f"Failed to load masks from tile {tile_id}: {e}")
                    mask_cache[tile_id] = None
            else:
                mask_cache[tile_id] = None

        cached = mask_cache.get(tile_id)
        if cached is None:
            n_maskless += 1
            det_info_raw.append((det, None, None, None))
            # --- Fix 3: Decrement tile count and evict if done ---
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        masks_array, slices = cached

        # Extract mask coordinates via find_objects bbox (O(bbox) not O(H×W))
        if mask_label < 1 or mask_label > len(slices):
            n_maskless += 1
            det_info_raw.append((det, None, None, None))
            # --- Fix 3: Decrement tile count and evict if done ---
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue
        sl = slices[mask_label - 1]  # find_objects is 1-indexed
        if sl is None:
            n_maskless += 1
            det_info_raw.append((det, None, None, None))
            # --- Fix 3: Decrement tile count and evict if done ---
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        local_ys, local_xs = np.where(masks_array[sl] == mask_label)
        if len(local_ys) == 0:
            n_maskless += 1
            det_info_raw.append((det, None, None, None))
            # --- Fix 3: Decrement tile count and evict if done ---
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        # Adjust for slice offset + convert to global coords
        global_xs = local_xs + sl[1].start + tile_x
        global_ys = local_ys + sl[0].start + tile_y

        # Compute bounding box (x_min, y_min, x_max, y_max)
        bbox = (int(global_xs.min()), int(global_ys.min()),
                int(global_xs.max()), int(global_ys.max()))

        # Track max x for dynamic stride computation
        observed_max_x = max(observed_max_x, int(global_xs.max()))

        det_info_raw.append((det, bbox, global_xs, global_ys))

        # --- Fix 3: Decrement tile count and evict if done ---
        tile_det_counts[tile_id] -= 1
        if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
            del mask_cache[tile_id]

    # --- Fix 1: Close all HDF5 file handles ---
    for tile_id, h5f in h5_handles.items():
        if h5f is not None:
            try:
                h5f.close()
            except Exception:
                pass
    del h5_handles

    # Free any remaining mask cache entries (should be empty after eviction)
    for _tid in list(mask_cache):
        cached = mask_cache.pop(_tid)
        if cached is not None:
            del cached
    del mask_cache
    del tile_det_counts

    if n_maskless > 0:
        logger.warning(f"{n_maskless}/{n_total} detections have no valid mask data; "
                       f"kept without overlap check")

    # Compute dynamic stride: must exceed max x coordinate to avoid collisions.
    # Use at least _COORD_STRIDE_DEFAULT (1M) for safety on small images.
    coord_stride = max(observed_max_x + 1, _COORD_STRIDE_DEFAULT)
    if coord_stride > _COORD_STRIDE_DEFAULT:
        logger.info(f"Coordinate stride increased to {coord_stride} (max x={observed_max_x})")

    # --- Fix 2 (part 1): Encode coordinates as tuples for set-based overlap ---
    # We encode (x, y) pairs as single int64 values using coord_stride, then
    # convert to frozenset for O(1) membership testing during overlap checks.
    det_info = []  # List of (det, global_bbox, encoded_frozenset, n_coords)
    for det, bbox, gxs, gys in det_info_raw:
        if bbox is None:
            det_info.append((det, None, None, 0))
        else:
            encoded = gys.astype(np.int64) * coord_stride + gxs.astype(np.int64)
            det_info.append((det, bbox, frozenset(encoded.tolist()), len(encoded)))
    del det_info_raw

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

    # Greedy deduplication with spatial grid acceleration.
    # Duplicates only exist in tile overlap zones, so we partition detections
    # into grid cells and only check neighbors. O(n*k) instead of O(n*kept).

    # Choose grid cell size: max bbox dimension across all detections, or fallback 500px.
    # This ensures each detection spans at most 2x2 grid cells.
    max_dim = 0
    for _, bbox, _, _ in det_info:
        if bbox is not None:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_dim = max(max_dim, w, h)
    grid_cell = max(max_dim + 1, 500)
    print(f"[dedup] Spatial grid cell size: {grid_cell} px", flush=True)

    from collections import defaultdict
    grid = defaultdict(list)  # (gx, gy) -> list of indices into kept_data
    # kept_data stores (bbox, coord_frozenset, n_coords) for each kept detection
    kept_data = []

    def _grid_cells(bbox):
        """Return all grid cells that a bbox touches."""
        gx0 = bbox[0] // grid_cell
        gy0 = bbox[1] // grid_cell
        gx1 = bbox[2] // grid_cell
        gy1 = bbox[3] // grid_cell
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                yield (gx, gy)

    kept = []

    for i, (det, bbox, coord_set, n_coords) in enumerate(det_info):
        if i % 50000 == 0 and i > 0:
            print(f"[dedup] Overlap check: {i}/{n_total} processed, {len(kept)} kept", flush=True)

        if bbox is None or coord_set is None:
            # No mask data — keep detection without overlap check.
            # Summary warning already logged above; skip per-detection spam.
            kept.append(det)
            kept_data.append((None, None, 0))
            continue

        # Collect candidate indices from neighboring grid cells (deduplicated)
        candidate_indices = set()
        for cell in _grid_cells(bbox):
            for idx in grid[cell]:
                candidate_indices.add(idx)

        is_duplicate = False
        for idx in candidate_indices:
            kept_bbox, kept_set, kept_n = kept_data[idx]
            if kept_bbox is None:
                continue
            # Bbox overlap check
            if (bbox[0] > kept_bbox[2] or bbox[2] < kept_bbox[0] or
                    bbox[1] > kept_bbox[3] or bbox[3] < kept_bbox[1]):
                continue

            # --- Fix 2: Set-based overlap with early exit ---
            # We only need to know if overlap >= threshold_count, so we count
            # up to the threshold and break early.  Iterate over the smaller
            # set for efficiency.
            smaller_size = min(n_coords, kept_n)
            if smaller_size == 0:
                continue
            threshold_count = int(smaller_size * min_overlap_fraction)
            if threshold_count == 0:
                # Any overlap at all exceeds fraction of 0 pixels
                threshold_count = 1

            # Iterate over the smaller set, probe the larger
            if n_coords <= kept_n:
                probe_set = kept_set
                iterate_set = coord_set
            else:
                probe_set = coord_set
                iterate_set = kept_set

            overlap = 0
            for c in iterate_set:
                if c in probe_set:
                    overlap += 1
                    if overlap >= threshold_count:
                        break

            if overlap >= threshold_count:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_idx = len(kept)
            kept.append(det)
            kept_data.append((bbox, coord_set, n_coords))
            # Insert into all grid cells this detection touches
            for cell in _grid_cells(bbox):
                grid[cell].append(kept_idx)

    n_removed = len(detections) - len(kept)
    if n_removed > 0:
        logger.info(f"Mask overlap dedup: {len(detections)} -> {len(kept)} ({n_removed} duplicates removed)")
    print(f"[dedup] Done: {len(detections)} -> {len(kept)} ({n_removed} removed)", flush=True)

    return kept
