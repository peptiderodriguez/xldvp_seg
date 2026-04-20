"""Post-dedup processing: contour extraction, background correction, feature extraction.

After deduplication the surviving detections go through three phases:

**Phase 1 — Contour extraction + quick medians** (per-tile, parallelized):
    Extract the original contour from the HDF5 segmentation mask and
    store it in the detection dict.  Compute per-cell median intensity
    per channel from the **original** binary mask region.

**Phase 2 — Background estimation** (global):
    Build KD-tree from global cell positions and the quick medians,
    estimate per-cell local background for each channel.

**Phase 3 — Intensity feature extraction on corrected pixels** (per-tile, parallelized):
    Subtract per-cell background from the pixel data within each
    detection's **original** mask, then extract per-channel intensity
    features from the corrected pixels.  Morphological features
    (computed during initial detection) are preserved unchanged.

Contour simplification (RDP) and dilation are deferred to LMD export
time so that features are always computed from the true segmentation
mask, not a reduced approximation.

Phases 1 and 3 are parallelized with ThreadPoolExecutor (auto-detects CPU
count from SLURM_CPUS_PER_TASK or os.cpu_count()).  Thread-safety: SHM is
read-only, HDF5 reads are independent per-tile, per-detection mutations are
on unique objects, and NumPy/cv2 release the GIL.
"""

import os

import cv2
import h5py
import numpy as np
from scipy.ndimage import find_objects

from xldvp_seg.analysis.background import _extract_centroids, local_background_subtract
from xldvp_seg.detection.strategies.mixins import MultiChannelFeatureMixin
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Thread-safety note: MultiChannelFeatureMixin is stateless — all methods are pure
# functions that take inputs and return outputs with no instance state mutation.
# This singleton is safe for concurrent use across ThreadPoolExecutor workers.
# DO NOT add instance state to this class without refactoring to per-thread instances.
_channel_mixin = MultiChannelFeatureMixin()

# Thread pool sizing: SLURM_CPUS_PER_TASK or os.cpu_count(), capped at 32
try:
    _MAX_WORKERS = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "")), 32)
except (ValueError, TypeError):
    _MAX_WORKERS = min(os.cpu_count() or 4, 32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _contour_from_binary(binary_mask: np.ndarray) -> np.ndarray | None:
    """Extract the largest external contour from a boolean mask.

    Args:
        binary_mask: (H, W) boolean array.

    Returns (N, 2) int array in local tile coordinates, or ``None``.
    """
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def _extract_intensity_features(
    mask: np.ndarray,
    tile_channels: dict[int, np.ndarray],
    *,
    include_zeros: bool = False,
) -> dict[str, float]:
    """Extract per-channel intensity features from *tile_channels* within *mask*.

    Uses ``MultiChannelFeatureMixin.extract_multichannel_features()`` which
    produces 15 features per channel plus inter-channel ratios.

    Args:
        mask: Boolean mask defining the region to extract features from.
        include_zeros: Pass True for background-corrected data where
            zero-valued pixels are real signal (not CZI padding).
    """
    channels_dict = {f"ch{ch}": data for ch, data in sorted(tile_channels.items())}
    return _channel_mixin.extract_multichannel_features(
        mask,
        channels_dict,
        compute_ratios=True,
        _include_zeros=include_zeros,
    )


# ---------------------------------------------------------------------------
# Tile data readers
# ---------------------------------------------------------------------------


def _read_tile_from_shm(
    slide_shm_arr: np.ndarray,
    ch_to_slot: dict[int, int],
    tile_x: int,
    tile_y: int,
    tile_h: int,
    tile_w: int,
    x_start: int,
    y_start: int,
) -> dict[int, np.ndarray]:
    """Slice per-channel 2-D arrays from shared memory for one tile.

    Returns ``{czi_channel_index: (tile_h, tile_w) uint16_array}``.
    """
    rel_x = tile_x - x_start
    rel_y = tile_y - y_start
    channels: dict[int, np.ndarray] = {}
    for czi_ch, slot in sorted(ch_to_slot.items()):
        channels[czi_ch] = slide_shm_arr[rel_y : rel_y + tile_h, rel_x : rel_x + tile_w, slot]
    return channels


def _read_tile_from_loader(
    loader,
    ch_indices: list[int],
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> dict[int, np.ndarray]:
    """Read per-channel 2-D arrays from CZI loader (fallback when no SHM)."""
    channels: dict[int, np.ndarray] = {}
    for ch in ch_indices:
        tile = loader.get_tile(tile_x, tile_y, tile_size, channel=ch)
        if tile is not None and tile.size > 0:
            if tile.ndim == 3:
                tile = tile[:, :, 0]
            channels[ch] = tile
    return channels


def _load_tile_channels(
    use_shm: bool,
    slide_shm_arr,
    ch_to_slot,
    tile_x,
    tile_y,
    tile_h,
    tile_w,
    x_start,
    y_start,
    loader,
    _ch_indices,
    tile_size,
) -> dict[int, np.ndarray]:
    """Load tile channel data from SHM or loader, clipping to mask dims."""
    tile_channels: dict[int, np.ndarray] = {}
    if use_shm:
        tile_channels = _read_tile_from_shm(
            slide_shm_arr,
            ch_to_slot,
            tile_x,
            tile_y,
            tile_h,
            tile_w,
            x_start,
            y_start,
        )
    elif loader is not None:
        tile_channels = _read_tile_from_loader(loader, _ch_indices, tile_x, tile_y, tile_size)
        # Clip to mask dims — edge tiles may be smaller than tile_size
        for ck in list(tile_channels):
            ca = tile_channels[ck]
            if ca.shape[0] > tile_h or ca.shape[1] > tile_w:
                tile_channels[ck] = ca[:tile_h, :tile_w]
    return tile_channels


def _parse_tile_key(tile_key: str) -> tuple[int, int]:
    """Parse ``'X_Y'`` tile key to ``(tile_x, tile_y)``."""
    parts = tile_key.split("_")
    return int(parts[0]), int(parts[1])


# ---------------------------------------------------------------------------
# Per-tile worker functions (for ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def _phase1_tile(
    tile_key: str,
    tile_dets: list[dict],
    tiles_dir,
    mask_filename: str,
    use_shm: bool,
    slide_shm_arr,
    ch_to_slot,
    x_start: int,
    y_start: int,
    loader,
    ch_indices: list[int],
    tile_size: int,
    has_data_source: bool,
    contour_processing: bool,
    pixel_size_um: float,
) -> tuple[int, int]:
    """Process one tile for Phase 1 (contour extraction + quick medians).

    Extracts the original contour from the HDF5 segmentation mask and
    computes quick median intensity per channel from the **original** mask
    region (no dilation or RDP — those are deferred to LMD export).

    Note: HDF5 mask files are read in both Phase 1 and Phase 3 independently.
    On network filesystems (GPFS), this adds I/O latency. A future optimization
    could cache masks from Phase 1 for Phase 3 reuse.

    Returns ``(n_ok, n_fail)``.
    """
    import hdf5plugin  # noqa: F401 — register LZ4 codec

    tile_x, tile_y = _parse_tile_key(tile_key)
    tile_name = f"tile_{tile_x}_{tile_y}"
    n_ok = 0
    n_fail = 0

    mask_path = tiles_dir / tile_name / mask_filename
    if not mask_path.exists():
        return 0, len(tile_dets)

    with h5py.File(mask_path, "r") as hf:
        masks_arr = hf["masks"][:]
    tile_h, tile_w = masks_arr.shape[:2]

    # Compute bboxes for all labels ONCE per tile in C (scipy).
    # Without this, doing (masks_arr == label) per detection is O(N_cells × tile_pixels).
    bboxes = find_objects(masks_arr)
    n_labels = len(bboxes)

    tile_channels: dict[int, np.ndarray] = {}
    if has_data_source:
        tile_channels = _load_tile_channels(
            use_shm,
            slide_shm_arr,
            ch_to_slot,
            tile_x,
            tile_y,
            tile_h,
            tile_w,
            x_start,
            y_start,
            loader,
            ch_indices,
            tile_size,
        )

    for det in tile_dets:
        label = det.get("mask_label")
        if label is None or label <= 0 or label > n_labels:
            n_fail += 1
            continue

        bbox = bboxes[label - 1]
        if bbox is None:
            n_fail += 1
            continue

        r_slice, c_slice = bbox
        # bbox-sized mask, not tile-sized
        crop_mask = masks_arr[r_slice, c_slice] == label
        if not crop_mask.any():
            n_fail += 1
            continue

        if contour_processing:
            contour_local = _contour_from_binary(crop_mask)
            if contour_local is not None:
                # cv2 returns (x, y) = (col, row) coords in bbox-local space.
                # Shift to tile-local, then to global.
                origin = det.get("tile_origin", [0, 0])
                contour_global = contour_local.astype(np.float64)
                contour_global[:, 0] += c_slice.start + origin[0]
                contour_global[:, 1] += r_slice.start + origin[1]
                det["contour_px"] = contour_global.tolist()
                det["contour_um"] = (contour_global * pixel_size_um).tolist()

        n_ok += 1

        quick_medians = {}
        if tile_channels:
            for ch, data in tile_channels.items():
                crop = data[r_slice, c_slice]
                pixels = crop[crop_mask].astype(np.float32)
                nonzero_pixels = pixels[pixels > 0]
                quick_medians[ch] = (
                    float(np.median(nonzero_pixels)) if len(nonzero_pixels) > 0 else 0.0
                )
        det["_bg_quick_medians"] = quick_medians

    return n_ok, n_fail


def _phase3_tile(
    tile_key: str,
    tile_dets: list[dict],
    tiles_dir,
    mask_filename: str,
    use_shm: bool,
    slide_shm_arr,
    ch_to_slot,
    x_start: int,
    y_start: int,
    loader,
    ch_indices: list[int],
    tile_size: int,
    has_data_source: bool,
    per_cell_bg: dict[int, dict[int, float]],
    pixel_size_um: float,
) -> tuple[int, int]:
    """Process one tile for Phase 3 (feature extraction on corrected pixels).

    Returns ``(n_ok, n_fail)``.
    """
    import hdf5plugin  # noqa: F401

    tile_x, tile_y = _parse_tile_key(tile_key)
    tile_name = f"tile_{tile_x}_{tile_y}"
    n_ok = 0
    n_fail = 0

    mask_path = tiles_dir / tile_name / mask_filename
    if not mask_path.exists():
        return 0, len(tile_dets)

    with h5py.File(mask_path, "r") as hf:
        masks_arr = hf["masks"][:]
    tile_h, tile_w = masks_arr.shape[:2]

    if not has_data_source:
        return 0, len(tile_dets)

    tile_channels = _load_tile_channels(
        use_shm,
        slide_shm_arr,
        ch_to_slot,
        tile_x,
        tile_y,
        tile_h,
        tile_w,
        x_start,
        y_start,
        loader,
        ch_indices,
        tile_size,
    )
    if not tile_channels:
        return 0, len(tile_dets)

    # Compute bboxes for all labels ONCE per tile in C (scipy).
    # Without this, per-detection bbox extraction is O(N_cells × tile_pixels).
    bboxes = find_objects(masks_arr)
    n_labels = len(bboxes)

    for det in tile_dets:
        det_idx = det["_postdedup_idx"]
        label = det.get("mask_label")
        if label is None or label <= 0 or label > n_labels:
            n_fail += 1
            continue

        bbox = bboxes[label - 1]
        if bbox is None:
            n_fail += 1
            continue

        r_slice, c_slice = bbox
        crop_mask = masks_arr[r_slice, c_slice] == label
        if not crop_mask.any():
            n_fail += 1
            continue

        bg = per_cell_bg.get(det_idx, {})
        feat = det.setdefault("features", {})
        has_bg = bool(bg)

        raw_crops: dict[int, np.ndarray] = {}
        for ch, data in tile_channels.items():
            raw_crops[ch] = data[r_slice, c_slice].astype(np.float32)

        if has_bg:
            # Raw pass: stats only — ratios are computed on the corrected data below.
            raw_channels_dict = {f"ch{ch}": data for ch, data in sorted(raw_crops.items())}
            raw_feats = _channel_mixin.extract_multichannel_features(
                crop_mask,
                raw_channels_dict,
                compute_ratios=False,
                _include_zeros=False,
            )
            for k, v in raw_feats.items():
                feat[f"{k}_raw"] = v
            for ch, crop in raw_crops.items():
                ch_bg = bg.get(ch, 0.0)
                if ch_bg > 0:
                    crop[crop_mask] = np.maximum(crop[crop_mask] - ch_bg, 0.0)
            intensity_feats = _extract_intensity_features(
                crop_mask,
                raw_crops,
                include_zeros=True,
            )
        else:
            intensity_feats = _extract_intensity_features(crop_mask, raw_crops)

        feat.update(intensity_feats)

        for ch in tile_channels:
            ch_bg = bg.get(ch, 0.0)
            if ch_bg > 0:
                raw_median = feat.get(f"ch{ch}_median_raw", 0.0)
                feat[f"ch{ch}_background"] = ch_bg
                feat[f"ch{ch}_snr"] = float(raw_median / ch_bg)
            else:
                feat[f"ch{ch}_background"] = 0.0
                feat[f"ch{ch}_snr"] = 0.0

        feat["area_um2"] = float(int(crop_mask.sum()) * pixel_size_um**2)

        n_ok += 1

    return n_ok, n_fail


# ---------------------------------------------------------------------------
# Multi-process workers (ProcessPool path — workaround for h5py phil lock)
#
# h5py has a global Python-level lock (``phil``) that serializes all HDF5
# operations across threads in a single process. ThreadPool on Phase 1/3
# therefore hits the lock on every ``hf["masks"][:]`` read, capping effective
# parallelism at ~3× regardless of worker count. Running workers in separate
# processes (each with their own ``phil``) restores true parallelism.
#
# The task functions below live at module scope so ``ProcessPoolExecutor`` can
# pickle them by qualname. They operate on minimal task packets (not full
# detection dicts) to keep per-task pickling cheap, and return update dicts
# that the main process merges with :func:`_apply_tile_updates`.
# ---------------------------------------------------------------------------


def _load_tile_channels_in_worker(
    ctx,
    tile_x: int,
    tile_y: int,
    tile_h: int,
    tile_w: int,
) -> dict[int, np.ndarray]:
    """Slice per-channel 2-D arrays from the worker's attached SHM."""
    cfg = ctx.slide_config
    rel_x = tile_x - cfg.x_start
    rel_y = tile_y - cfg.y_start
    out: dict[int, np.ndarray] = {}
    for czi_ch, slot in sorted(cfg.ch_to_slot.items()):
        out[czi_ch] = ctx.slide_arr[rel_y : rel_y + tile_h, rel_x : rel_x + tile_w, slot]
    return out


def _read_tile_masks(ctx, tile_x: int, tile_y: int):
    """Read ``masks_arr`` for one tile; return ``(masks_arr, bboxes)`` or ``None``."""
    from pathlib import Path

    import h5py

    cfg = ctx.slide_config
    tile_dir = Path(cfg.tiles_dir) / f"tile_{tile_x}_{tile_y}"
    mask_path = tile_dir / cfg.mask_filename
    if not mask_path.exists():
        return None
    with h5py.File(str(mask_path), "r") as hf:
        masks_arr = hf["masks"][:]
    bboxes = find_objects(masks_arr)
    return masks_arr, bboxes


def _phase1_mp_task(task: dict, ctx) -> dict[int, dict]:
    """Worker: compute Phase 1 updates (contour + quick medians) for one tile.

    Returns a dict ``{det_idx: {...update fields...}}``. Fields may include
    ``contour_px``, ``contour_um``, ``_bg_quick_medians``. Detections that
    fail validation are simply omitted from the result.
    """
    tile_x, tile_y = _parse_tile_key(task["tile_key"])
    contour_processing = bool(task.get("contour_processing", True))
    pixel_size_um = float(task["pixel_size_um"])
    dets = task["dets"]  # [{"idx": int, "mask_label": int, "tile_origin": [ox, oy]}, ...]

    read = _read_tile_masks(ctx, tile_x, tile_y)
    if read is None:
        return {}
    masks_arr, bboxes = read
    tile_h, tile_w = masks_arr.shape[:2]
    n_labels = len(bboxes)

    tile_channels = _load_tile_channels_in_worker(ctx, tile_x, tile_y, tile_h, tile_w)

    updates: dict[int, dict] = {}
    for d in dets:
        label = d["mask_label"]
        if label is None or label <= 0 or label > n_labels:
            continue
        bbox = bboxes[label - 1]
        if bbox is None:
            continue
        r_slice, c_slice = bbox
        crop_mask = masks_arr[r_slice, c_slice] == label
        if not crop_mask.any():
            continue

        det_upd: dict = {}
        if contour_processing:
            contour_local = _contour_from_binary(crop_mask)
            if contour_local is not None:
                ox, oy = d.get("tile_origin", [0, 0])
                cg = contour_local.astype(np.float64)
                cg[:, 0] += c_slice.start + ox
                cg[:, 1] += r_slice.start + oy
                det_upd["contour_px"] = cg.tolist()
                det_upd["contour_um"] = (cg * pixel_size_um).tolist()

        quick_medians: dict[int, float] = {}
        for ch, data in tile_channels.items():
            crop = data[r_slice, c_slice]
            pixels = crop[crop_mask].astype(np.float32)
            nz = pixels[pixels > 0]
            quick_medians[ch] = float(np.median(nz)) if len(nz) > 0 else 0.0
        det_upd["_bg_quick_medians"] = quick_medians

        updates[d["idx"]] = det_upd

    return updates


def _phase3_mp_task(task: dict, ctx) -> dict[int, dict]:
    """Worker: compute Phase 3 updates (bg-corrected intensity features) for one tile.

    Returns a dict ``{det_idx: {"features": {...}}}`` where ``features`` is
    the diff to merge into each detection's feature dict (NOT a replacement).
    """
    tile_x, tile_y = _parse_tile_key(task["tile_key"])
    pixel_size_um = float(task["pixel_size_um"])
    dets = task["dets"]
    # per_cell_bg is keyed by int det_idx → {ch: bg} but JSON/pickle round-trips
    # often stringify int keys; normalize defensively.
    raw_bg = task.get("per_cell_bg", {})
    per_cell_bg: dict[int, dict[int, float]] = {
        int(k): {int(ch): float(v) for ch, v in chmap.items()} for k, chmap in raw_bg.items()
    }

    read = _read_tile_masks(ctx, tile_x, tile_y)
    if read is None:
        return {}
    masks_arr, bboxes = read
    tile_h, tile_w = masks_arr.shape[:2]
    n_labels = len(bboxes)

    tile_channels = _load_tile_channels_in_worker(ctx, tile_x, tile_y, tile_h, tile_w)
    if not tile_channels:
        return {}

    updates: dict[int, dict] = {}
    for d in dets:
        label = d["mask_label"]
        det_idx = d["idx"]
        if label is None or label <= 0 or label > n_labels:
            continue
        bbox = bboxes[label - 1]
        if bbox is None:
            continue
        r_slice, c_slice = bbox
        crop_mask = masks_arr[r_slice, c_slice] == label
        if not crop_mask.any():
            continue

        bg = per_cell_bg.get(det_idx, {})
        has_bg = bool(bg)
        features: dict = {}

        raw_crops: dict[int, np.ndarray] = {}
        for ch, data in tile_channels.items():
            raw_crops[ch] = data[r_slice, c_slice].astype(np.float32)

        if has_bg:
            raw_channels_dict = {f"ch{ch}": arr for ch, arr in sorted(raw_crops.items())}
            raw_feats = _channel_mixin.extract_multichannel_features(
                crop_mask,
                raw_channels_dict,
                compute_ratios=False,
                _include_zeros=False,
            )
            for k, v in raw_feats.items():
                features[f"{k}_raw"] = v
            for ch, crop in raw_crops.items():
                ch_bg = bg.get(ch, 0.0)
                if ch_bg > 0:
                    crop[crop_mask] = np.maximum(crop[crop_mask] - ch_bg, 0.0)
            intensity_feats = _extract_intensity_features(crop_mask, raw_crops, include_zeros=True)
        else:
            intensity_feats = _extract_intensity_features(crop_mask, raw_crops)

        features.update(intensity_feats)

        for ch in tile_channels:
            ch_bg = bg.get(ch, 0.0)
            if ch_bg > 0:
                raw_median = features.get(f"ch{ch}_median_raw", 0.0)
                features[f"ch{ch}_background"] = ch_bg
                features[f"ch{ch}_snr"] = float(raw_median / ch_bg)
            else:
                features[f"ch{ch}_background"] = 0.0
                features[f"ch{ch}_snr"] = 0.0

        features["area_um2"] = float(int(crop_mask.sum()) * pixel_size_um**2)

        updates[det_idx] = {"features": features}

    return updates


def _build_phase1_tasks(
    by_tile: dict[str, list[dict]],
    *,
    contour_processing: bool,
    pixel_size_um: float,
) -> list[dict]:
    """Build minimal, picklable task packets for Phase 1."""
    tasks: list[dict] = []
    for tile_key, tile_dets in by_tile.items():
        refs: list[dict] = []
        for i, det in enumerate(tile_dets):
            label = det.get("mask_label")
            if label is None:
                continue
            origin = det.get("tile_origin", [0, 0])
            # Strict primitive types for pickle safety
            refs.append(
                {
                    "idx": int(det["_postdedup_idx"]),
                    "mask_label": int(label),
                    "tile_origin": [int(origin[0]), int(origin[1])],
                    "_local_idx": i,
                }
            )
        if refs:
            tasks.append(
                {
                    "tile_key": tile_key,
                    "dets": refs,
                    "contour_processing": bool(contour_processing),
                    "pixel_size_um": float(pixel_size_um),
                }
            )
    return tasks


def _build_phase3_tasks(
    by_tile: dict[str, list[dict]],
    per_cell_bg: dict[int, dict[int, float]],
    *,
    pixel_size_um: float,
) -> list[dict]:
    """Build minimal, picklable task packets for Phase 3.

    ``per_cell_bg`` is sliced per tile so each task only carries the bg entries
    for cells in that tile — avoids broadcasting a multi-MB global dict.
    """
    tasks: list[dict] = []
    for tile_key, tile_dets in by_tile.items():
        refs: list[dict] = []
        bg_slice: dict[int, dict[int, float]] = {}
        for det in tile_dets:
            label = det.get("mask_label")
            if label is None:
                continue
            det_idx = int(det["_postdedup_idx"])
            refs.append({"idx": det_idx, "mask_label": int(label)})
            cell_bg = per_cell_bg.get(det_idx)
            if cell_bg:
                bg_slice[det_idx] = {int(ch): float(v) for ch, v in cell_bg.items()}
        if refs:
            tasks.append(
                {
                    "tile_key": tile_key,
                    "dets": refs,
                    "per_cell_bg": bg_slice,
                    "pixel_size_um": float(pixel_size_um),
                }
            )
    return tasks


def _apply_tile_updates(detections: list[dict], updates: dict[int, dict]) -> int:
    """Merge worker-returned update dicts into the main-process detection list.

    Semantics:
      - ``features`` key: merged into existing ``det['features']`` via ``dict.update``
      - all other keys: replace existing value
    Returns the number of detections mutated.
    """
    n = 0
    for det_idx, upd in updates.items():
        if det_idx < 0 or det_idx >= len(detections):
            continue
        det = detections[det_idx]
        for k, v in upd.items():
            if k == "features":
                det.setdefault("features", {}).update(v)
            else:
                det[k] = v
        n += 1
    return n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_detections_post_dedup(
    detections: list[dict],
    tiles_dir,
    pixel_size_um: float,
    mask_filename: str = "cell_masks.h5",
    *,
    # Data sources (provide SHM or loader — SHM preferred)
    slide_shm_arr: np.ndarray | None = None,
    ch_to_slot: dict[int, int] | None = None,
    x_start: int = 0,
    y_start: int = 0,
    loader=None,
    ch_indices: list[int] | None = None,
    tile_size: int = 3000,
    # Processing toggles
    contour_processing: bool = True,
    background_correction: bool = True,
    bg_neighbors: int = 30,
    # Nuclear counting (Phase 4)
    count_nuclei: bool = False,
    nuc_channel_idx: int | None = None,
    min_nuclear_area: int = 50,
    cellpose_model=None,
    sam2_predictor=None,
    # Multi-GPU Phase 4 (preferred when SHM available)
    num_gpus: int = 0,
    shm_name: str | None = None,
    sam2_checkpoint=None,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    extract_sam2_embeddings: bool = True,
    # Deprecated — accepted for YAML/CLI compat, silently ignored
    dilation_um: float = 0.0,
    rdp_epsilon: float = 0.0,
) -> dict:
    """Run all post-dedup processing on *detections* **in-place**.

    Three-phase pipeline:

    1. **Contour extraction + quick medians** — extract original contours
       from HDF5 masks, compute quick median intensity per channel from
       the original mask region.
    2. **Background estimation** — KD-tree on global cell positions,
       estimate per-cell local background for each channel.
    3. **Intensity feature extraction on corrected pixels** — subtract
       per-cell background, then extract per-channel intensity features
       from the original mask.  Morphological features from initial
       detection are preserved.

    Contour simplification (RDP) and dilation are deferred to LMD
    export time so features always reflect the true segmentation mask.

    Args:
        detections: Deduped detection dicts (mutated in-place).
        tiles_dir: Path to tile output directory containing HDF5 mask files.
        pixel_size_um: Pixel size from CZI metadata.
        mask_filename: HDF5 mask file name inside each tile dir.
        slide_shm_arr: Shared-memory array (H, W, n_channels).  Preferred.
        ch_to_slot: ``{czi_channel_index: shm_slot_index}`` mapping.
        x_start, y_start: Mosaic origin for SHM coordinate conversion.
        loader: CZI loader (fallback when SHM is unavailable).
        ch_indices: CZI channel indices for loader fallback (required when
            using loader without ch_to_slot).
        tile_size: Tile edge length in pixels (used with loader fallback).
        contour_processing: Whether to extract contours from HDF5 masks.
        background_correction: Whether to run local bg subtraction.
        bg_neighbors: KD-tree neighbor count for bg subtraction.
        dilation_um: Deprecated — ignored (dilation moved to LMD export).
        rdp_epsilon: Deprecated — ignored (RDP moved to LMD export).

    Returns:
        Summary dict with processing statistics.
    """
    from pathlib import Path

    import hdf5plugin  # noqa: F401 — register LZ4 codec before h5py

    tiles_dir = Path(tiles_dir)
    n_total = len(detections)

    if n_total == 0:
        logger.info("No detections — skipping post-dedup processing")
        return {"n_processed": 0}

    # Decide data source
    use_shm = slide_shm_arr is not None and ch_to_slot is not None
    if not use_shm and loader is None:
        logger.warning(
            "Neither shared memory nor CZI loader provided. "
            "Skipping feature re-extraction (contour processing only)."
        )
    _ch_indices = sorted(ch_to_slot.keys()) if ch_to_slot else (ch_indices or [])
    has_data_source = use_shm or (loader is not None and len(_ch_indices) > 0)
    if not use_shm and loader is not None and not _ch_indices:
        logger.warning("Loader provided but no ch_indices — feature re-extraction will be skipped")

    # CZI loader read_mosaic is NOT thread-safe — cap workers to 1 when using loader fallback
    effective_workers = _MAX_WORKERS
    if not use_shm and loader is not None:
        effective_workers = 1
        logger.info("CZI loader path is not thread-safe — using single-threaded post-dedup")

    logger.info("=" * 50)
    logger.info("POST-DEDUP PROCESSING")
    logger.info("=" * 50)
    logger.info("  Detections: %d", n_total)
    logger.info("  Contour extraction: %s", "ON" if contour_processing else "DISABLED")
    logger.info("  Background correction: %s (k=%d)", background_correction, bg_neighbors)
    logger.info(
        "  Data source: %s", "shared memory" if use_shm else ("CZI loader" if loader else "NONE")
    )

    # Assign stable indices before grouping (survives across phases)
    for i, det in enumerate(detections):
        det["_postdedup_idx"] = i

    # --- Group detections by tile ---
    by_tile: dict[str, list[dict]] = {}
    for det in detections:
        origin = det.get("tile_origin", [0, 0])
        key = f"{origin[0]}_{origin[1]}"
        by_tile.setdefault(key, []).append(det)

    n_tiles = len(by_tile)
    logger.info("  Tiles with detections: %d", n_tiles)

    # --- Counters ---
    n_contour_ok = 0
    n_contour_fail = 0
    n_features_ok = 0
    n_features_fail = 0

    # ==================================================================
    # PHASE 1: Contour extraction + quick median extraction (parallelized)
    # ==================================================================
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm as _tqdm

    # Multi-process path (ProcessPool) — preferred when SHM is available.
    # Workaround for h5py phil lock; see _phase1_mp_task docstring.
    use_mp_postdedup = use_shm and shm_name is not None

    if use_mp_postdedup:
        from xldvp_seg.processing.multiprocess_tiles import (
            SharedSlideConfig,
            TileProcessor,
        )

        slide_cfg = SharedSlideConfig(
            shm_name=shm_name,
            shm_shape=tuple(slide_shm_arr.shape),
            shm_dtype=str(slide_shm_arr.dtype),
            ch_to_slot=dict(ch_to_slot),
            x_start=int(x_start),
            y_start=int(y_start),
            tiles_dir=str(tiles_dir),
            mask_filename=str(mask_filename),
        )
        processor = TileProcessor(slide_cfg)

        tasks = _build_phase1_tasks(
            by_tile,
            contour_processing=contour_processing,
            pixel_size_um=pixel_size_um,
        )
        logger.info("-" * 40)
        logger.info(
            "Phase 1: Contour extraction + quick median extraction (ProcessPool, %d workers)",
            processor.num_workers,
        )

        with _tqdm(total=len(tasks), desc="Phase 1") as pbar:
            for task, updates in processor.run(
                _phase1_mp_task,
                tasks,
                desc="Phase 1",
                largest_first_key=lambda t: len(t["dets"]),
            ):
                tile_key = task["tile_key"]
                expected = len(task["dets"])
                got = _apply_tile_updates(detections, updates)
                n_contour_ok += got
                n_contour_fail += max(0, expected - got)
                pbar.update(1)
    else:
        logger.info("-" * 40)
        logger.info(
            "Phase 1: Contour extraction + quick median extraction (ThreadPool, %d workers)",
            effective_workers,
        )

        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {
                pool.submit(
                    _phase1_tile,
                    k,
                    v,
                    tiles_dir,
                    mask_filename,
                    use_shm,
                    slide_shm_arr,
                    ch_to_slot,
                    x_start,
                    y_start,
                    loader,
                    _ch_indices,
                    tile_size,
                    has_data_source,
                    contour_processing,
                    pixel_size_um,
                ): k
                for k, v in by_tile.items()
            }
            with _tqdm(total=len(futures), desc="Phase 1") as pbar:
                for f in as_completed(futures):
                    tile_key = futures[f]
                    try:
                        ok, fail = f.result()
                        n_contour_ok += ok
                        n_contour_fail += fail
                    except Exception:
                        logger.error("Phase 1 failed for tile %s", tile_key, exc_info=True)
                        n_contour_fail += len(by_tile[tile_key])
                    pbar.update(1)

    logger.info("  Phase 1 complete: %d contours ok, %d failed", n_contour_ok, n_contour_fail)

    total_phase1 = n_contour_ok + n_contour_fail
    if total_phase1 > 0 and n_contour_fail / total_phase1 > 0.05:
        logger.warning(
            "Phase 1 failure rate %.1f%% (%d/%d) exceeds 5%% threshold. "
            "Check tile data integrity.",
            100 * n_contour_fail / total_phase1,
            n_contour_fail,
            total_phase1,
        )

    # ==================================================================
    # PHASE 2: Background estimation (KD-tree)
    # ==================================================================
    # {detection_index: {channel: bg_value}}
    per_cell_bg: dict[int, dict[int, float]] = {}

    if background_correction and has_data_source:
        logger.info("-" * 40)
        logger.info("Phase 2: Background estimation (k=%d)", bg_neighbors)

        centroids = _extract_centroids(detections)

        # Discover which channels have data
        bg_channels: set[int] = set()
        for det in detections:
            bg_channels.update(det.get("_bg_quick_medians", {}).keys())
        bg_channels_sorted = sorted(bg_channels)

        # Build the KD-tree once and reuse across all channels
        _cached_tree_and_indices = None

        for ch in bg_channels_sorted:
            values = np.array(
                [d.get("_bg_quick_medians", {}).get(ch, 0.0) for d in detections],
                dtype=np.float64,
            )
            _, ch_bg, _cached_tree_and_indices = local_background_subtract(
                values,
                centroids,
                bg_neighbors,
                tree_and_indices=_cached_tree_and_indices,
            )

            for i in range(len(detections)):
                per_cell_bg.setdefault(i, {})[ch] = float(ch_bg[i])

            median_bg = float(np.median(ch_bg))
            logger.info("  ch%d: median bg=%.1f", ch, median_bg)

        logger.info("  Phase 2 complete: %d channels estimated", len(bg_channels_sorted))
    else:
        logger.info("Phase 2: Background correction disabled or no data source — skipping")

    # ==================================================================
    # PHASE 3: Feature extraction on background-corrected pixels (parallelized)
    # ==================================================================
    try:
        if use_mp_postdedup:
            from xldvp_seg.processing.multiprocess_tiles import (
                SharedSlideConfig,
                TileProcessor,
            )

            slide_cfg = SharedSlideConfig(
                shm_name=shm_name,
                shm_shape=tuple(slide_shm_arr.shape),
                shm_dtype=str(slide_shm_arr.dtype),
                ch_to_slot=dict(ch_to_slot),
                x_start=int(x_start),
                y_start=int(y_start),
                tiles_dir=str(tiles_dir),
                mask_filename=str(mask_filename),
            )
            processor = TileProcessor(slide_cfg)

            tasks = _build_phase3_tasks(by_tile, per_cell_bg, pixel_size_um=pixel_size_um)
            logger.info("-" * 40)
            logger.info(
                "Phase 3: Feature extraction on corrected pixels (ProcessPool, %d workers)",
                processor.num_workers,
            )

            with _tqdm(total=len(tasks), desc="Phase 3") as pbar:
                for task, updates in processor.run(
                    _phase3_mp_task,
                    tasks,
                    desc="Phase 3",
                    largest_first_key=lambda t: len(t["dets"]),
                ):
                    expected = len(task["dets"])
                    got = _apply_tile_updates(detections, updates)
                    n_features_ok += got
                    n_features_fail += max(0, expected - got)
                    pbar.update(1)
        else:
            logger.info("-" * 40)
            logger.info(
                "Phase 3: Feature extraction on corrected pixels (ThreadPool, %d workers)",
                effective_workers,
            )
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(
                        _phase3_tile,
                        k,
                        v,
                        tiles_dir,
                        mask_filename,
                        use_shm,
                        slide_shm_arr,
                        ch_to_slot,
                        x_start,
                        y_start,
                        loader,
                        _ch_indices,
                        tile_size,
                        has_data_source,
                        per_cell_bg,
                        pixel_size_um,
                    ): k
                    for k, v in by_tile.items()
                }
                with _tqdm(total=len(futures), desc="Phase 3") as pbar:
                    for f in as_completed(futures):
                        tile_key = futures[f]
                        try:
                            ok, fail = f.result()
                            n_features_ok += ok
                            n_features_fail += fail
                        except Exception:
                            logger.error("Phase 3 failed for tile %s", tile_key, exc_info=True)
                            n_features_fail += len(by_tile[tile_key])
                        pbar.update(1)

        total_phase3 = n_features_ok + n_features_fail
        if total_phase3 > 0 and n_features_fail / total_phase3 > 0.05:
            logger.warning(
                "Phase 3 failure rate %.1f%% (%d/%d) exceeds 5%% threshold. "
                "Check data source and feature extraction.",
                100 * n_features_fail / total_phase3,
                n_features_fail,
                total_phase3,
            )
    finally:
        # --- Cleanup temporary keys ---
        for det in detections:
            det.pop("_bg_quick_medians", None)
            det.pop("_postdedup_idx", None)

    # ==================================================================
    # PHASE 4: Nuclear counting (optional)
    # ==================================================================
    n_nuclei_counted = 0
    use_multigpu_phase4 = (
        count_nuclei
        and nuc_channel_idx is not None
        and num_gpus >= 1
        and use_shm
        and shm_name is not None
    )

    if use_multigpu_phase4:
        logger.info("-" * 40)
        logger.info("Phase 4: Nuclear counting (multi-GPU, %d workers)", num_gpus)
        from xldvp_seg.pipeline.multigpu_phase4 import run_multigpu_phase4

        n_nuclei_counted = run_multigpu_phase4(
            by_tile,
            detections,
            num_gpus=num_gpus,
            tiles_dir=tiles_dir,
            mask_filename=mask_filename,
            pixel_size_um=pixel_size_um,
            min_nuclear_area=min_nuclear_area,
            slide_shm_arr=slide_shm_arr,
            shm_name=shm_name,
            nuc_channel_idx=nuc_channel_idx,
            ch_to_slot=ch_to_slot,
            x_start=x_start,
            y_start=y_start,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            extract_sam2_embeddings=extract_sam2_embeddings,
        )
        logger.info("  Phase 4 complete: %d cells enriched with nuclear counts", n_nuclei_counted)

    elif count_nuclei and nuc_channel_idx is not None and cellpose_model is not None:
        logger.info("-" * 40)
        logger.info("Phase 4: Nuclear counting (single-process fallback)")

        from xldvp_seg.analysis.nuclear_count import (
            _percentile_normalize_to_uint8,
            count_nuclei_for_tile,
        )

        for tile_key, tile_dets in by_tile.items():
            tile_x, tile_y = _parse_tile_key(tile_key)

            # Load cell masks for this tile
            tile_dir = tiles_dir / f"tile_{tile_x}_{tile_y}"
            mask_path = tile_dir / mask_filename
            if not mask_path.exists():
                continue
            with h5py.File(str(mask_path), "r") as hf:
                cell_masks = hf["masks"][:]
            tile_h, tile_w = cell_masks.shape[:2]

            # Get nuclear channel tile from SHM or loader
            if use_shm and nuc_channel_idx in ch_to_slot:
                slot = ch_to_slot[nuc_channel_idx]
                nuc_tile = slide_shm_arr[
                    tile_y - y_start : tile_y - y_start + tile_h,
                    tile_x - x_start : tile_x - x_start + tile_w,
                    slot,
                ]
            elif loader is not None:
                nuc_tile = loader.get_tile(tile_x, tile_y, tile_size, channel=nuc_channel_idx)
                if nuc_tile is not None:
                    if nuc_tile.ndim == 3:
                        nuc_tile = nuc_tile[:, :, 0]
                    nuc_tile = nuc_tile[:tile_h, :tile_w]
            else:
                continue

            if nuc_tile is None or nuc_tile.size == 0:
                continue

            # Set SAM2 image if available (nuclear channel only)
            if sam2_predictor is not None:
                nuc_uint8 = _percentile_normalize_to_uint8(nuc_tile)
                nuc_rgb = np.stack([nuc_uint8] * 3, axis=-1)
                sam2_predictor.set_image(nuc_rgb)

            results, n_nuc = count_nuclei_for_tile(
                cell_masks,
                nuc_tile,
                cellpose_model,
                pixel_size_um,
                min_nuclear_area,
                tile_x,
                tile_y,
                sam2_predictor=sam2_predictor,
            )

            if sam2_predictor is not None:
                try:
                    sam2_predictor.reset_predictor()
                except Exception as e:
                    logger.debug("SAM2 predictor reset failed: %s", e)

            # Enrich detections: summary metrics in features, per-nucleus list at top-level
            for det in tile_dets:
                mask_label = det.get("tile_mask_label", det.get("mask_label"))
                if mask_label is not None and int(mask_label) in results:
                    nuc_feats = results[int(mask_label)]
                    features = det.setdefault("features", {})
                    # Summary metrics go in features (flat, classifier-compatible)
                    for k in (
                        "n_nuclei",
                        "nuclear_area_um2",
                        "nuclear_area_fraction",
                        "largest_nucleus_um2",
                        "nuclear_solidity",
                        "nuclear_eccentricity",
                    ):
                        if k in nuc_feats:
                            features[k] = nuc_feats[k]
                    # Per-nucleus detail list at top-level (not in features — avoids JSON bloat
                    # and doesn't break flat-feature assumptions in train_classifier.py)
                    if "nuclei" in nuc_feats and nuc_feats["nuclei"]:
                        det["nuclei"] = nuc_feats["nuclei"]
                    n_nuclei_counted += 1

        logger.info("  Phase 4 complete: %d cells enriched with nuclear counts", n_nuclei_counted)
    elif count_nuclei:
        logger.warning(
            "Phase 4 skipped: count_nuclei=True but missing prerequisites: "
            "nuc_channel_idx=%s, num_gpus=%d, use_shm=%s, shm_name=%s, "
            "cellpose_model=%s",
            nuc_channel_idx,
            num_gpus,
            use_shm,
            shm_name is not None,
            cellpose_model is not None,
        )

    # --- Summary ---
    # Compute corrected channels list for metadata
    corrected_channels = sorted({ch for bg in per_cell_bg.values() for ch in bg})

    summary = {
        "n_processed": n_total,
        "n_tiles": n_tiles,
        "n_contour_ok": n_contour_ok,
        "n_contour_fail": n_contour_fail,
        "n_features_ok": n_features_ok,
        "n_features_fail": n_features_fail,
        "contour_processing": contour_processing,
        "background_correction": background_correction,
        "bg_neighbors": bg_neighbors,
        "corrected_channels": corrected_channels,
        "count_nuclei": count_nuclei,
        "n_nuclei_counted": n_nuclei_counted,
    }

    logger.info("=" * 50)
    logger.info("Post-dedup processing complete:")
    logger.info("  Contours: %d ok, %d failed", n_contour_ok, n_contour_fail)
    logger.info("  Features: %d re-extracted, %d skipped", n_features_ok, n_features_fail)
    if corrected_channels:
        logger.info("  Background-corrected channels: %s", corrected_channels)

    return summary
