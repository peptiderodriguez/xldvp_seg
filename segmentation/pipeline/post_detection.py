"""Post-dedup processing: contour dilation, background correction, feature extraction.

After deduplication the surviving detections go through three phases:

**Phase 1 — Contour dilation + quick means** (per-tile):
    Extract contour from HDF5 mask, dilate + RDP, compute quick mean
    intensity per channel from the dilated mask region.

**Phase 2 — Background estimation** (global):
    Build KD-tree from global cell positions and the quick means,
    estimate per-cell local background for each channel.

**Phase 3 — Feature extraction on corrected pixels** (per-tile):
    Subtract per-cell background from the pixel data within each
    detection's dilated mask, *then* extract all features (morph +
    per-channel intensity) from the corrected pixels.  This ensures
    every feature — including std, percentiles, etc. — reflects the
    background-corrected signal.

The two tile passes both read from shared memory (essentially free) or
the CZI loader (acceptable cost on resume).
"""

import cv2
import numpy as np
import h5py

from segmentation.utils.logging import get_logger
from segmentation.utils.detection_utils import safe_to_uint8
from segmentation.utils.feature_extraction import extract_morphological_features
from segmentation.detection.strategies.mixins import MultiChannelFeatureMixin
from segmentation.lmd.contour_processing import (
    dilate_contour,
    rdp_simplify,
    DEFAULT_DILATION_UM,
    DEFAULT_RDP_EPSILON,
)
from segmentation.pipeline.background import local_background_subtract, _extract_centroids

logger = get_logger(__name__)

# Re-use a singleton mixin instance for channel stats extraction
_channel_mixin = MultiChannelFeatureMixin()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contour_from_mask(mask_arr: np.ndarray, label: int) -> np.ndarray | None:
    """Extract the largest external contour for *label* in *mask_arr*.

    Returns (N, 2) int array in local tile coordinates, or ``None``.
    """
    binary = (mask_arr == label).astype(np.uint8)
    if binary.sum() == 0:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def _dilated_mask_from_contour(
    contour_local: np.ndarray,
    tile_h: int,
    tile_w: int,
    dilation_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Dilate a contour and rasterise to a binary mask clipped to tile bounds.

    Returns ``(dilated_contour, dilated_mask)`` — both in local tile coords.
    ``dilated_contour`` is (M, 2) float, ``dilated_mask`` is (tile_h, tile_w) bool.
    """
    dilated = dilate_contour(contour_local.astype(float), dilation_px)
    if dilated is None or len(dilated) < 3:
        return None, None

    # Clip to tile bounds
    dilated[:, 0] = np.clip(dilated[:, 0], 0, tile_w - 1)
    dilated[:, 1] = np.clip(dilated[:, 1], 0, tile_h - 1)

    # Rasterise to binary mask
    mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    pts = np.round(dilated).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return dilated, mask.astype(bool)


def _rasterize_contour(contour_local: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    """Rasterize a contour (local coords) into a boolean mask."""
    mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    pts = np.round(contour_local).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _extract_intensity_features(
    dilated_mask: np.ndarray,
    tile_channels: dict[int, np.ndarray],
) -> dict[str, float]:
    """Extract per-channel intensity features from *tile_channels* within *dilated_mask*.

    Uses ``MultiChannelFeatureMixin.extract_multichannel_features()`` which
    produces 15 features per channel plus inter-channel ratios.
    """
    channels_dict = {f"ch{ch}": data for ch, data in sorted(tile_channels.items())}
    return _channel_mixin.extract_multichannel_features(
        dilated_mask, channels_dict, compute_ratios=True,
    )


def _extract_morph_features(
    dilated_mask: np.ndarray,
    tile_rgb: np.ndarray,
    tile_global_mean: float | None = None,
) -> dict[str, float]:
    """Extract 22 base morphological features from *tile_rgb* within *dilated_mask*."""
    return extract_morphological_features(dilated_mask, tile_rgb, tile_global_mean)


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
        channels[czi_ch] = slide_shm_arr[rel_y:rel_y + tile_h, rel_x:rel_x + tile_w, slot]
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
    slide_shm_arr, ch_to_slot, tile_x, tile_y, tile_h, tile_w,
    x_start, y_start, loader, _ch_indices, tile_size,
) -> dict[int, np.ndarray]:
    """Load tile channel data from SHM or loader, clipping to mask dims."""
    tile_channels: dict[int, np.ndarray] = {}
    if use_shm:
        tile_channels = _read_tile_from_shm(
            slide_shm_arr, ch_to_slot, tile_x, tile_y, tile_h, tile_w, x_start, y_start,
        )
    elif loader is not None:
        tile_channels = _read_tile_from_loader(loader, _ch_indices, tile_x, tile_y, tile_size)
        # Clip to mask dims — edge tiles may be smaller than tile_size
        for ck in list(tile_channels):
            ca = tile_channels[ck]
            if ca.shape[0] > tile_h or ca.shape[1] > tile_w:
                tile_channels[ck] = ca[:tile_h, :tile_w]
    return tile_channels


def _tile_rgb_from_channels(
    tile_channels: dict[int, np.ndarray],
    display_channels: list[int] | None = None,
) -> np.ndarray:
    """Build an RGB uint8 tile from up to 3 channel arrays for morph features.

    If *display_channels* is given (list of CZI channel indices for R, G, B),
    use those.  Otherwise use the first 3 channel indices available.
    """
    ch_keys = sorted(tile_channels.keys())
    if display_channels:
        rgb_keys = display_channels[:3]
    else:
        rgb_keys = ch_keys[:3]

    layers = []
    for k in rgb_keys:
        arr = tile_channels.get(k)
        if arr is not None:
            layers.append(safe_to_uint8(arr))
        else:
            layers.append(np.zeros_like(next(iter(tile_channels.values())), dtype=np.uint8))

    # Pad to 3 if fewer channels
    while len(layers) < 3:
        layers.append(np.zeros_like(layers[0]))

    return np.stack(layers, axis=-1)


def _parse_tile_key(tile_key: str) -> tuple[int, int]:
    """Parse ``'X_Y'`` tile key to ``(tile_x, tile_y)``."""
    parts = tile_key.split("_")
    return int(parts[0]), int(parts[1])


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
    display_channels: list[int] | None = None,
    # Processing toggles
    contour_processing: bool = True,
    dilation_um: float = DEFAULT_DILATION_UM,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    background_correction: bool = True,
    bg_neighbors: int = 30,
) -> dict:
    """Run all post-dedup processing on *detections* **in-place**.

    Three-phase pipeline:

    1. **Contour dilation + quick means** — extract contours from HDF5
       masks, dilate + RDP, compute quick mean intensity per channel.
    2. **Background estimation** — KD-tree on global cell positions,
       estimate per-cell local background for each channel.
    3. **Feature extraction on corrected pixels** — subtract per-cell
       background from the pixel data, then extract all features.

    All features (intensity, morph) are computed on background-corrected
    pixel data, so std, percentiles, etc. all reflect the corrected signal.

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
        display_channels: CZI channel indices for R/G/B morph extraction.
        contour_processing: Whether to dilate + RDP contours.
        dilation_um, rdp_epsilon: Contour processing parameters.
        background_correction: Whether to run local bg subtraction.
        bg_neighbors: KD-tree neighbor count for bg subtraction.

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

    dilation_px = dilation_um / pixel_size_um if contour_processing else 0.0

    logger.info("=" * 50)
    logger.info("POST-DEDUP PROCESSING")
    logger.info("=" * 50)
    logger.info("  Detections: %d", n_total)
    if contour_processing:
        logger.info("  Contour dilation: +%.2f um (%.1f px)", dilation_um, dilation_px)
        logger.info("  RDP epsilon: %.1f px", rdp_epsilon)
    else:
        logger.info("  Contour processing: DISABLED")
    logger.info("  Background correction: %s (k=%d)", background_correction, bg_neighbors)
    logger.info("  Data source: %s", "shared memory" if use_shm else ("CZI loader" if loader else "NONE"))

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
    # PHASE 1: Contour dilation + quick mean extraction
    # ==================================================================
    logger.info("-" * 40)
    logger.info("Phase 1: Contour dilation + quick mean extraction")

    for tile_idx, (tile_key, tile_dets) in enumerate(by_tile.items()):
        tile_x, tile_y = _parse_tile_key(tile_key)
        tile_name = f"tile_{tile_x}_{tile_y}"

        if (tile_idx + 1) % 50 == 0 or tile_idx == 0:
            logger.info(
                "  [P1] Tile %d/%d (%s, %d detections)",
                tile_idx + 1, n_tiles, tile_name, len(tile_dets),
            )

        # Load masks from HDF5
        mask_path = tiles_dir / tile_name / mask_filename
        if not mask_path.exists():
            logger.warning("Mask file not found: %s — skipping %d detections", mask_path, len(tile_dets))
            n_contour_fail += len(tile_dets)
            continue

        with h5py.File(mask_path, "r") as hf:
            masks_arr = hf["masks"][:]
        tile_h, tile_w = masks_arr.shape[:2]

        # Load tile channels for quick means
        tile_channels: dict[int, np.ndarray] = {}
        if has_data_source:
            tile_channels = _load_tile_channels(
                use_shm, slide_shm_arr, ch_to_slot, tile_x, tile_y,
                tile_h, tile_w, x_start, y_start, loader, _ch_indices, tile_size,
            )

        # Process each detection
        for det in tile_dets:
            label = det.get("mask_label")
            if label is None:
                n_contour_fail += 1
                continue

            # Extract contour from original mask
            contour_local = _contour_from_mask(masks_arr, label)
            if contour_local is None:
                n_contour_fail += 1
                continue

            # Dilate contour (or use original)
            if contour_processing and dilation_px > 0:
                dilated_contour, dilated_mask = _dilated_mask_from_contour(
                    contour_local, tile_h, tile_w, dilation_px,
                )
                if dilated_contour is None:
                    dilated_mask = (masks_arr == label).astype(bool)
                    dilated_contour = rdp_simplify(contour_local.astype(np.float32), rdp_epsilon)
                else:
                    dilated_contour = rdp_simplify(dilated_contour.astype(np.float32), rdp_epsilon)
                n_contour_ok += 1

                # Store dilated contour (global coordinates)
                origin = det.get("tile_origin", [0, 0])
                contour_global = dilated_contour.copy()
                contour_global[:, 0] += origin[0]
                contour_global[:, 1] += origin[1]
                det["contour_dilated_px"] = contour_global.tolist()
                det["contour_dilated_um"] = (contour_global * pixel_size_um).tolist()
            else:
                dilated_mask = (masks_arr == label).astype(bool)
                n_contour_ok += 1

            # Quick mean per channel (for background estimation)
            quick_means = {}
            if tile_channels and dilated_mask.any():
                for ch, data in tile_channels.items():
                    pixels = data[dilated_mask].astype(np.float32)
                    quick_means[ch] = float(np.mean(pixels)) if len(pixels) > 0 else 0.0
            det["_bg_quick_means"] = quick_means

        del masks_arr
        if tile_channels:
            del tile_channels

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
            bg_channels.update(det.get("_bg_quick_means", {}).keys())
        bg_channels_sorted = sorted(bg_channels)

        for ch in bg_channels_sorted:
            values = np.array(
                [d.get("_bg_quick_means", {}).get(ch, 0.0) for d in detections],
                dtype=np.float64,
            )
            _, ch_bg = local_background_subtract(values, centroids, bg_neighbors)

            for i in range(len(detections)):
                per_cell_bg.setdefault(i, {})[ch] = float(ch_bg[i])

            median_bg = float(np.median(ch_bg))
            logger.info("  ch%d: median bg=%.1f", ch, median_bg)

        logger.info("  Phase 2 complete: %d channels estimated", len(bg_channels_sorted))
    else:
        logger.info("Phase 2: Background correction disabled or no data source — skipping")

    # ==================================================================
    # PHASE 3: Feature extraction on background-corrected pixels
    # ==================================================================
    logger.info("-" * 40)
    logger.info("Phase 3: Feature extraction on corrected pixels")

    # Build detection-index lookup
    det_to_idx = {id(det): i for i, det in enumerate(detections)}

    for tile_idx, (tile_key, tile_dets) in enumerate(by_tile.items()):
        tile_x, tile_y = _parse_tile_key(tile_key)
        tile_name = f"tile_{tile_x}_{tile_y}"

        if (tile_idx + 1) % 50 == 0 or tile_idx == 0:
            logger.info(
                "  [P3] Tile %d/%d (%s, %d detections)",
                tile_idx + 1, n_tiles, tile_name, len(tile_dets),
            )

        # Need HDF5 masks for non-dilation fallback
        mask_path = tiles_dir / tile_name / mask_filename
        if not mask_path.exists():
            n_features_fail += len(tile_dets)
            continue

        with h5py.File(mask_path, "r") as hf:
            masks_arr = hf["masks"][:]
        tile_h, tile_w = masks_arr.shape[:2]

        # Load tile channels
        if not has_data_source:
            n_features_fail += len(tile_dets)
            del masks_arr
            continue

        tile_channels = _load_tile_channels(
            use_shm, slide_shm_arr, ch_to_slot, tile_x, tile_y,
            tile_h, tile_w, x_start, y_start, loader, _ch_indices, tile_size,
        )
        if not tile_channels:
            n_features_fail += len(tile_dets)
            del masks_arr
            continue

        # Build RGB tile for morph features (uncorrected — shape features unaffected)
        tile_rgb = _tile_rgb_from_channels(tile_channels, display_channels)
        valid = np.max(tile_rgb, axis=2) > 0
        tile_global_mean = float(np.mean(tile_rgb[valid])) if valid.any() else 0.0

        # Process each detection
        for det in tile_dets:
            det_idx = det_to_idx[id(det)]
            label = det.get("mask_label")
            if label is None:
                n_features_fail += 1
                continue

            # Reconstruct dilated mask
            if det.get("contour_dilated_px") is not None:
                # Rasterize from stored contour (convert global → local)
                contour_global = np.array(det["contour_dilated_px"])
                origin = det.get("tile_origin", [0, 0])
                contour_local = contour_global.copy()
                contour_local[:, 0] -= origin[0]
                contour_local[:, 1] -= origin[1]
                dilated_mask = _rasterize_contour(contour_local, tile_h, tile_w)
            else:
                dilated_mask = (masks_arr == label).astype(bool)

            if not dilated_mask.any():
                n_features_fail += 1
                continue

            # Get bounding box for efficient crop-based correction
            rows = np.where(dilated_mask.any(axis=1))[0]
            cols = np.where(dilated_mask.any(axis=0))[0]
            r0, r1 = rows[0], rows[-1] + 1
            c0, c1 = cols[0], cols[-1] + 1
            crop_mask = dilated_mask[r0:r1, c0:c1]

            bg = per_cell_bg.get(det_idx, {})
            feat = det.setdefault("features", {})

            # --- Intensity features from bg-corrected pixel crops ---
            corrected_crops: dict[int, np.ndarray] = {}
            for ch, data in tile_channels.items():
                crop = data[r0:r1, c0:c1].astype(np.float32)
                ch_bg = bg.get(ch, 0.0)
                if ch_bg > 0:
                    crop[crop_mask] = np.maximum(crop[crop_mask] - ch_bg, 0.0)
                corrected_crops[ch] = crop

            intensity_feats = _extract_intensity_features(crop_mask, corrected_crops)
            feat.update(intensity_feats)

            # Store raw means and background metadata
            quick_means = det.get("_bg_quick_means", {})
            for ch in tile_channels:
                raw_mean = quick_means.get(ch, 0.0)
                ch_bg = bg.get(ch, 0.0)
                feat[f"ch{ch}_mean_raw"] = raw_mean
                if ch_bg > 0:
                    feat[f"ch{ch}_background"] = ch_bg
                    feat[f"ch{ch}_snr"] = float(raw_mean / ch_bg) if ch_bg > 0 else 0.0

            # --- Morphological features from uncorrected RGB crop ---
            crop_rgb = tile_rgb[r0:r1, c0:c1]
            morph_feats = _extract_morph_features(crop_mask, crop_rgb, tile_global_mean)
            if morph_feats:
                for k, v in morph_feats.items():
                    feat[k] = v
                feat["area_um2"] = float(morph_feats.get("area", 0) * pixel_size_um ** 2)

            n_features_ok += 1

        del masks_arr, tile_channels, tile_rgb

    # --- Cleanup temporary keys ---
    for det in detections:
        det.pop("_bg_quick_means", None)

    # --- Summary ---
    # Compute corrected channels list for metadata
    corrected_channels = sorted({
        ch for bg in per_cell_bg.values() for ch in bg
    })

    summary = {
        "n_processed": n_total,
        "n_tiles": n_tiles,
        "n_contour_ok": n_contour_ok,
        "n_contour_fail": n_contour_fail,
        "n_features_ok": n_features_ok,
        "n_features_fail": n_features_fail,
        "contour_processing": contour_processing,
        "dilation_um": dilation_um,
        "rdp_epsilon": rdp_epsilon,
        "background_correction": background_correction,
        "bg_neighbors": bg_neighbors,
        "corrected_channels": corrected_channels,
    }

    logger.info("=" * 50)
    logger.info("Post-dedup processing complete:")
    logger.info("  Contours: %d ok, %d failed", n_contour_ok, n_contour_fail)
    logger.info("  Features: %d re-extracted, %d skipped", n_features_ok, n_features_fail)
    if corrected_channels:
        logger.info("  Background-corrected channels: %s", corrected_channels)

    return summary
