"""
Deduplication for detections from tiled image processing.

Removes duplicate detections caused by tile overlap. Two methods:

1. **mask_overlap** (default): Loads full mask pixels from HDF5, encodes as
   frozenset of global coordinates, checks pixel-level overlap. Accurate but
   memory-intensive for large detections.

2. **iou_nms**: Extracts contour polygons from HDF5 masks, builds a Shapely
   STRtree for O(n log n) spatial queries, then runs greedy NMS based on
   contour IoU. Uses ~100x less memory per detection (polygon vs pixel
   frozenset) and is faster for large slides.

Both methods work with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Performance optimizations:
- Batch HDF5 file handles: all tile HDF5 files are opened upfront and kept
  open for the duration of mask loading, avoiding repeated open/close on GPFS.
- Set-based intersection with early exit (mask_overlap): overlap checking uses
  Python sets with counting up to threshold, avoiding full intersection.
- STRtree spatial index (iou_nms): O(n log n) candidate finding vs O(n^2).
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

from xldvp_seg.exceptions import ConfigError, DetectionError
from xldvp_seg.utils.logging import get_logger

try:
    from shapely.errors import GEOSException
except ImportError:
    GEOSException = Exception

logger = get_logger(__name__)

# Default stride for encoding (x, y) as single int64 for fast numpy set
# intersection.  This is overridden dynamically in the dedup loop below when
# the actual data exceeds this value.
_COORD_STRIDE_DEFAULT = 1_000_000


def deduplicate_by_mask_overlap(
    detections, tiles_dir, min_overlap_fraction=0.1, mask_filename=None, sort_by="area"
):
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
    from collections import Counter
    from pathlib import Path

    if not detections:
        return []

    if mask_filename is None:
        raise ConfigError(
            "mask_filename must be provided (e.g. '{cell_type}_masks.h5'). "
            "No default is assumed to avoid cell-type-specific assumptions."
        )

    tiles_dir = Path(tiles_dir)
    n_total = len(detections)
    logger.info(f"Starting dedup for {n_total} detections...")

    # --- Fix 3 (part 1): Count detections per tile for cache eviction ---
    tile_det_counts = Counter()
    for det in detections:
        tile_origin = tuple(det.get("tile_origin", [0, 0]))
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
                h5_handles[tile_id] = h5py.File(masks_file, "r")
            except Exception as e:
                logger.warning(f"Failed to open HDF5 file {masks_file}: {e}")
                h5_handles[tile_id] = None
        else:
            h5_handles[tile_id] = None
    logger.info(
        f"Opened {sum(1 for v in h5_handles.values() if v is not None)} "
        f"HDF5 files out of {len(unique_tile_ids)} tiles"
    )

    # Load all masks and compute global bounding boxes.
    # We collect raw (global_xs, global_ys) first, then encode after the loop
    # using a dynamically computed stride to avoid coordinate collisions on
    # mosaics wider than 1M pixels.
    det_info_raw = []  # List of (det, global_bbox, global_xs, global_ys) or (det, None, None, None)
    observed_max_x = 0  # Track actual max x coordinate across all detections
    observed_max_y = 0  # Track actual max y for int64 overflow guard (Phase 4c)
    n_maskless = 0  # Count detections without valid mask data

    # Cache: tile_id -> (masks_array, find_objects_slices)
    mask_cache = {}

    for i, det in enumerate(detections):
        if i % 5000 == 0 and i > 0:
            logger.info(f"Loading masks: {i}/{n_total}")

        tile_origin = tuple(det.get("tile_origin", [0, 0]))
        tile_x, tile_y = tile_origin
        tile_id = f"tile_{tile_x}_{tile_y}"
        mask_label = det.get("mask_label")

        if mask_label is None or mask_label == 0:
            # Try to extract from ID as fallback
            det_id = det.get("id", "")
            try:
                mask_label = int(det_id.split("_")[-1])
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
                    masks_array = h5f["masks"][:]
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
        bbox = (
            int(global_xs.min()),
            int(global_ys.min()),
            int(global_xs.max()),
            int(global_ys.max()),
        )

        # Track max x for dynamic stride computation
        observed_max_x = max(observed_max_x, int(global_xs.max()))
        observed_max_y = max(observed_max_y, int(global_ys.max()))

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
            except Exception as e:
                logger.debug("Failed to close HDF5 handle for tile %s: %s", tile_id, e)
    del h5_handles

    # Free any remaining mask cache entries (should be empty after eviction)
    for _tid in list(mask_cache):
        cached = mask_cache.pop(_tid)
        if cached is not None:
            del cached
    del mask_cache
    del tile_det_counts

    if n_maskless > 0:
        logger.warning(
            f"{n_maskless}/{n_total} detections have no valid mask data; "
            f"kept without overlap check"
        )

    # Compute dynamic stride: must exceed max x coordinate to avoid collisions.
    # Use at least _COORD_STRIDE_DEFAULT (1M) for safety on small images.
    coord_stride = max(observed_max_x + 1, _COORD_STRIDE_DEFAULT)
    if coord_stride > _COORD_STRIDE_DEFAULT:
        logger.info(f"Coordinate stride increased to {coord_stride} (max x={observed_max_x})")
    # Phase 4c / E.1: slide-size sanity check. The actual frozenset tuple
    # encoding uses Python ints (arbitrary precision), so there's no numeric
    # overflow risk. This guard exists to refuse absurdly large slides where
    # the frozenset hash table would be pathologically slow, and to flag
    # that IoU NMS is the better dedup choice for such inputs.
    if observed_max_y * coord_stride >= 2**63:
        raise DetectionError(
            f"Slide too large for tuple-encoded dedup: observed_max_y="
            f"{observed_max_y}, coord_stride={coord_stride}. Use "
            f"--dedup-method iou_nms for large slides."
        )

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

    logger.info("Masks loaded. Running overlap check...")

    # Sort by priority descending (keep higher-priority ones)
    if sort_by == "confidence":

        def _confidence_key(item):
            det = item[0]
            score = det.get("rf_prediction")
            if score is None:
                score = det.get("score")
            if score is None:
                score = det.get("features", {}).get("sam2_score", 0)
            return score if score is not None else 0

        det_info.sort(key=_confidence_key, reverse=True)
    else:
        det_info.sort(key=lambda x: x[0].get("features", {}).get("area", 0), reverse=True)

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
    logger.info(f"Spatial grid cell size: {grid_cell} px")

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
            logger.info(f"Overlap check: {i}/{n_total} processed, {len(kept)} kept")

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
            if (
                bbox[0] > kept_bbox[2]
                or bbox[2] < kept_bbox[0]
                or bbox[1] > kept_bbox[3]
                or bbox[3] < kept_bbox[1]
            ):
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
        logger.info(
            f"Mask overlap dedup: {len(detections)} -> {len(kept)} ({n_removed} duplicates removed)"
        )
    logger.info(f"Done: {len(detections)} -> {len(kept)} ({n_removed} removed)")

    return kept


def deduplicate_by_iou_nms(
    detections,
    tiles_dir,
    iou_threshold=0.2,
    mask_filename=None,
    sort_by="area",
):
    """Remove duplicate detections using contour IoU with Shapely STRtree NMS.

    For each pair of spatially overlapping detections (found via STRtree),
    computes IoU from contour polygons extracted from HDF5 masks. When
    IoU > threshold, keeps the higher-priority detection.

    This is an alternative to deduplicate_by_mask_overlap() that uses
    ~100x less memory per detection (polygon vs pixel frozenset) and
    O(n log n) spatial queries via STRtree.

    Args:
        detections: List of detection dicts with tile_origin, mask_label
        tiles_dir: Path to tiles directory with HDF5 mask files
        iou_threshold: IoU threshold for suppression (default: 0.2)
        mask_filename: Name of HDF5 mask file in each tile dir
        sort_by: Priority ordering - 'area' (larger wins) or 'confidence'
                 (higher rf_prediction/score wins)

    Returns:
        Deduplicated list of detections.
    """
    from collections import Counter
    from pathlib import Path

    import cv2
    from shapely import STRtree
    from shapely.geometry import Polygon

    if not detections:
        return []

    if mask_filename is None:
        raise ConfigError(
            "mask_filename must be provided (e.g. '{cell_type}_masks.h5'). "
            "No default is assumed to avoid cell-type-specific assumptions."
        )

    tiles_dir = Path(tiles_dir)
    n_total = len(detections)
    logger.info(
        f"Starting IoU NMS dedup for {n_total} detections " f"(threshold={iou_threshold})..."
    )

    # --- Count detections per tile for cache eviction ---
    tile_det_counts = Counter()
    for det in detections:
        tile_origin = tuple(det.get("tile_origin", [0, 0]))
        tile_x, tile_y = tile_origin
        tile_id = f"tile_{tile_x}_{tile_y}"
        tile_det_counts[tile_id] += 1

    # --- Batch-open all HDF5 file handles upfront ---
    unique_tile_ids = set(tile_det_counts.keys())
    h5_handles = {}
    for tile_id in unique_tile_ids:
        masks_file = tiles_dir / tile_id / mask_filename
        if masks_file.exists():
            try:
                h5_handles[tile_id] = h5py.File(masks_file, "r")
            except Exception as e:
                logger.warning(f"Failed to open HDF5 file {masks_file}: {e}")
                h5_handles[tile_id] = None
        else:
            h5_handles[tile_id] = None
    logger.info(
        f"Opened {sum(1 for v in h5_handles.values() if v is not None)} "
        f"HDF5 files out of {len(unique_tile_ids)} tiles"
    )

    # --- Load masks and extract contour polygons ---
    # Cache: tile_id -> (masks_array, find_objects_slices)
    mask_cache = {}
    polygons = []  # Shapely polygons, one per detection (or None)
    n_maskless = 0

    for i, det in enumerate(detections):
        if i % 5000 == 0 and i > 0:
            logger.info(f"Extracting contours: {i}/{n_total}")

        tile_origin = tuple(det.get("tile_origin", [0, 0]))
        tile_x, tile_y = tile_origin
        tile_id = f"tile_{tile_x}_{tile_y}"
        mask_label = det.get("mask_label")

        if mask_label is None or mask_label == 0:
            det_id = det.get("id", "")
            try:
                mask_label = int(det_id.split("_")[-1])
            except (ValueError, IndexError):
                n_maskless += 1
                polygons.append(None)
                tile_det_counts[tile_id] -= 1
                if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                    del mask_cache[tile_id]
                continue

        mask_label = int(mask_label)

        # Load masks + precompute find_objects if not cached
        if tile_id not in mask_cache:
            h5f = h5_handles.get(tile_id)
            if h5f is not None:
                try:
                    masks_array = h5f["masks"][:]
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
            polygons.append(None)
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        masks_array, slices = cached

        if mask_label < 1 or mask_label > len(slices):
            n_maskless += 1
            polygons.append(None)
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        sl = slices[mask_label - 1]
        if sl is None:
            n_maskless += 1
            polygons.append(None)
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        # Extract binary mask for this label within its bounding box
        local_mask = (masks_array[sl] == mask_label).astype(np.uint8)
        if local_mask.sum() == 0:
            n_maskless += 1
            polygons.append(None)
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        # Extract contour from binary mask
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            n_maskless += 1
            polygons.append(None)
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue

        # Use largest contour (by area) if multiple fragments exist
        contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze()  # (N, 1, 2) -> (N, 2)

        if contour.ndim != 2 or len(contour) < 3:
            # Fewer than 3 points: degenerate mask, treat as maskless
            n_maskless += 1
            polygons.append(None)
            tile_det_counts[tile_id] -= 1
            if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                del mask_cache[tile_id]
            continue
        else:
            # Convert local contour coords to global coordinates
            # cv2 contour format is (x_local, y_local) within the slice bbox
            global_coords = [
                (float(pt[0]) + sl[1].start + tile_x, float(pt[1]) + sl[0].start + tile_y)
                for pt in contour
            ]
            poly = Polygon(global_coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
                # buffer(0) can produce MultiPolygon from self-intersecting shapes
                if poly.geom_type == "MultiPolygon":
                    poly = max(poly.geoms, key=lambda g: g.area)
            if poly.is_empty or poly.area == 0:
                n_maskless += 1
                polygons.append(None)
                tile_det_counts[tile_id] -= 1
                if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
                    del mask_cache[tile_id]
                continue

        polygons.append(poly)

        # Evict tile from cache if all its detections have been processed
        tile_det_counts[tile_id] -= 1
        if tile_det_counts[tile_id] <= 0 and tile_id in mask_cache:
            del mask_cache[tile_id]

    # --- Close all HDF5 file handles ---
    for tile_id, h5f in h5_handles.items():
        if h5f is not None:
            try:
                h5f.close()
            except Exception as e:
                logger.debug("Failed to close HDF5 handle for tile %s: %s", tile_id, e)
    del h5_handles

    # Free remaining mask cache
    for _tid in list(mask_cache):
        cached = mask_cache.pop(_tid)
        if cached is not None:
            del cached
    del mask_cache
    del tile_det_counts

    if n_maskless > 0:
        logger.warning(
            f"{n_maskless}/{n_total} detections have no valid mask/contour; "
            f"kept without overlap check"
        )

    # --- Build STRtree from valid polygons ---
    # STRtree needs a list of geometries; we map index -> detection index
    valid_indices = [i for i, p in enumerate(polygons) if p is not None]
    valid_polygons = [polygons[i] for i in valid_indices]

    if not valid_polygons:
        logger.info("No valid polygons for IoU NMS; returning all detections unchanged")
        return detections

    tree = STRtree(valid_polygons)
    # Mapping: valid_indices[pos] = det_idx (list is O(1) by position)
    det_idx_to_valid_pos = {det_idx: pos for pos, det_idx in enumerate(valid_indices)}

    logger.info(
        f"Contours loaded. Built STRtree with {len(valid_polygons)} polygons. " f"Running NMS..."
    )

    # --- Sort by priority ---
    if sort_by == "confidence":

        def _confidence_key(idx):
            det = detections[idx]
            score = det.get("rf_prediction")
            if score is None:
                score = det.get("score")
            if score is None:
                score = det.get("features", {}).get("sam2_score", 0)
            return score if score is not None else 0

        priority_order = sorted(valid_indices, key=_confidence_key, reverse=True)
    else:
        # Sort by area descending (larger detections win)
        priority_order = sorted(
            valid_indices,
            key=lambda idx: detections[idx].get("features", {}).get("area", 0),
            reverse=True,
        )

    # --- Greedy NMS ---
    kept = set()
    suppressed = set()

    for i in priority_order:
        if i in suppressed:
            continue
        kept.add(i)

        # Query STRtree for candidates whose bounding boxes overlap this polygon
        valid_pos = det_idx_to_valid_pos[i]
        candidate_positions = tree.query(valid_polygons[valid_pos])

        for j_pos in candidate_positions:
            j = valid_indices[j_pos]
            if j == i or j in kept or j in suppressed:
                continue

            # Compute IoU between the two polygons
            poly_i = valid_polygons[valid_pos]
            poly_j = valid_polygons[j_pos]
            try:
                intersection_area = poly_i.intersection(poly_j).area
            except (ValueError, TypeError, GEOSException):
                # Geometry errors (rare): skip this pair
                continue
            if intersection_area == 0:
                continue
            union_area = poly_i.area + poly_j.area - intersection_area
            if union_area > 0 and intersection_area / union_area > iou_threshold:
                suppressed.add(j)

    # Include maskless detections (kept without overlap check)
    maskless_indices = {i for i, p in enumerate(polygons) if p is None}

    # Build final result preserving original order
    keep_set = kept | maskless_indices
    result = [det for idx, det in enumerate(detections) if idx in keep_set]

    n_removed = n_total - len(result)
    if n_removed > 0:
        logger.info(
            f"IoU NMS dedup: {n_total} -> {len(result)} "
            f"({n_removed} duplicates removed, threshold={iou_threshold})"
        )
    logger.info(f"Done: {n_total} -> {len(result)} ({n_removed} removed)")

    return result
