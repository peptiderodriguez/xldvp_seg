"""Post-dedup processing: contour extraction, background correction, feature extraction.

After deduplication the surviving detections go through three phases:

**Phase 1 — Contour extraction + quick medians** (per-tile, parallelized):
    Extract the original contour from the HDF5 segmentation mask and
    store it in the detection dict.  Compute per-cell median intensity
    per channel from the **original** binary mask region.

**Phase 2 — Background estimation** (global):
    Build KD-tree from global cell positions and the quick means,
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

from xldvp_seg.detection.strategies.mixins import MultiChannelFeatureMixin
from xldvp_seg.pipeline.background import _extract_centroids, local_background_subtract
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Re-use a singleton mixin instance for channel stats extraction
_channel_mixin = MultiChannelFeatureMixin()

# Thread pool sizing: SLURM_CPUS_PER_TASK or os.cpu_count(), capped at 32
# Thread-safety: MultiChannelFeatureMixin has NO instance state — all methods
# are pure functions.  If instance state is ever added, this singleton must be
# replaced with per-thread instances or a lock.
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
        if label is None:
            n_fail += 1
            continue

        # Compute once, reuse for contour extraction and quick medians
        original_mask = (masks_arr == label).astype(bool)
        if not original_mask.any():
            n_fail += 1
            continue

        if contour_processing:
            contour_local = _contour_from_binary(original_mask)
            if contour_local is not None:
                origin = det.get("tile_origin", [0, 0])
                contour_global = contour_local.astype(np.float32)
                contour_global[:, 0] += origin[0]
                contour_global[:, 1] += origin[1]
                det["contour_px"] = contour_global.tolist()
                det["contour_um"] = (contour_global * pixel_size_um).tolist()

        n_ok += 1

        quick_medians = {}
        if tile_channels:
            for ch, data in tile_channels.items():
                pixels = data[original_mask].astype(np.float32)
                quick_medians[ch] = float(np.median(pixels))
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

    for det in tile_dets:
        det_idx = det["_postdedup_idx"]
        label = det.get("mask_label")
        if label is None:
            n_fail += 1
            continue

        # Always use the original segmentation mask from HDF5
        original_mask = (masks_arr == label).astype(bool)

        if not original_mask.any():
            n_fail += 1
            continue

        rows = np.where(original_mask.any(axis=1))[0]
        cols = np.where(original_mask.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        crop_mask = original_mask[r0:r1, c0:c1]

        bg = per_cell_bg.get(det_idx, {})
        feat = det.setdefault("features", {})
        has_bg = bool(bg)

        raw_crops: dict[int, np.ndarray] = {}
        for ch, data in tile_channels.items():
            raw_crops[ch] = data[r0:r1, c0:c1].astype(np.float32)

        if has_bg:
            raw_feats = _extract_intensity_features(crop_mask, raw_crops)
            for k, v in raw_feats.items():
                if "_ratio" not in k and "_diff" not in k and "_specificity" not in k:
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

        # Always compute area_um2 from the original mask
        feat["area_um2"] = float(int(crop_mask.sum()) * pixel_size_um**2)

        n_ok += 1

    return n_ok, n_fail


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
    # Deprecated — accepted for YAML/CLI compat, silently ignored
    dilation_um: float = 0.0,
    rdp_epsilon: float = 0.0,
) -> dict:
    """Run all post-dedup processing on *detections* **in-place**.

    Three-phase pipeline:

    1. **Contour extraction + quick means** — extract original contours
       from HDF5 masks, compute quick mean intensity per channel from
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
    # PHASE 1: Contour extraction + quick mean extraction (parallelized)
    # ==================================================================
    logger.info("-" * 40)
    logger.info(
        "Phase 1: Contour extraction + quick mean extraction (%d workers)", effective_workers
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm as _tqdm

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
    logger.info("-" * 40)
    logger.info("Phase 3: Feature extraction on corrected pixels (%d workers)", effective_workers)

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

    # --- Cleanup temporary keys ---
    for det in detections:
        det.pop("_bg_quick_medians", None)
        det.pop("_postdedup_idx", None)

    # ==================================================================
    # PHASE 4: Nuclear counting (optional, single-threaded — uses GPU)
    # ==================================================================
    n_nuclei_counted = 0
    if count_nuclei and nuc_channel_idx is not None and cellpose_model is not None:
        logger.info("-" * 40)
        logger.info("Phase 4: Nuclear counting (Cellpose on nuclear channel)")

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
                except Exception:
                    pass

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
            "Phase 4 skipped: count_nuclei=True but missing nuc_channel_idx=%s, "
            "cellpose_model=%s",
            nuc_channel_idx,
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
