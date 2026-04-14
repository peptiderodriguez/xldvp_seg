#!/usr/bin/env python3
"""Coarse-to-fine threshold-based lumen detection from OME-Zarr.

Replaces SAM2-based lumen detection with a pure numpy/scipy/cv2 pipeline:
channel-sum + Otsu threshold + connected components, refined across scales.

No GPU required. Suitable for login nodes or any SLURM partition.

Algorithm:
  1. Discover lumens at EVERY scale (64x -> 16x -> 8x -> 4x -> 2x), masking out
     already-found lumens at each step. Coarse scales use whole-image processing,
     fine scales use tiling.
  2. Refine each discovered lumen to the finest available scale for precise edges.
  3. Extract contours + metadata (contour_global_um, area_um2, darkness_tier, etc.)
  4. Optional HTML viewer via generate_contour_viewer.py

Usage::

    python scripts/detect_vessel_lumens_threshold.py \\
        --zarr-path slide.ome.zarr \\
        --scales 2,4,8,16,64 \\
        --otsu-multiplier 0.8 \\
        --blur-sigma-um 5.0 \\
        --min-area-um2 50 \\
        --output-dir vessel_lumens_threshold/ \\
        --czi-path slide.czi \\
        --display-channels 1,3,0 \\
        --channel-names "SMA,CD31,nuc" \\
        --output vessel_lumens_threshold.html
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter
from scipy.ndimage import label as ndi_label
from skimage.filters import threshold_otsu
from skimage.morphology import disk as skimage_disk
from skimage.segmentation import watershed as skimage_watershed

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger, setup_logging
from xldvp_seg.utils.zarr_io import (
    generate_tiles,
    get_effective_shape,
    read_all_channels_crop,
    resolve_zarr_scales,
)
from xldvp_seg.utils.zarr_io import read_all_channels as _read_all_channels

logger = get_logger(__name__)

# Physical distance for boundary ring dilation (~one vessel wall thickness)
_BOUNDARY_RING_UM = 10.0


# ---------------------------------------------------------------------------
# Core image processing
# ---------------------------------------------------------------------------


def _percentile_normalize(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize a single channel to [0, 1].

    Uses p1/p99 from non-zero pixels. Re-zeros CZI padding (original zeros).

    Args:
        arr: 2D float32 array.

    Returns:
        Normalized float32 array in [0, 1].
    """
    zero_mask = arr == 0
    nonzero = arr[~zero_mask]
    if len(nonzero) == 0:
        return np.zeros_like(arr)
    p1, p99 = np.percentile(nonzero, [1, 99])
    del nonzero  # free before creating normalized array
    if p99 <= p1:
        return np.zeros_like(arr)
    # Normalize to [0, 1] — returns a new array, does NOT modify input
    result = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
    result[zero_mask] = 0.0
    return result


def _channel_sum_and_threshold(
    channels: list[np.ndarray],
    pixel_size_um: float,
    blur_sigma_um: float,
    otsu_multiplier: float,
    morph_close_um: float = 10.0,
    morph_open_um: float = 5.0,
    local_threshold: bool = True,
    block_size_um: float = 100.0,
    threshold_fraction: float = 0.5,
    fill_expansion: float = 1.5,
    global_dark_floor: float | None = None,  # accepted but unused here (used in refinement)
    channel_indices: list[int] | None = None,  # subset of channels to sum (default: all)
    output_complement: bool = False,  # output tissue regions instead of dark holes
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize channels, sum, blur, threshold (inverted for dark lumens).

    Supports two thresholding modes:

    - **Local** (default): Gaussian local-mean with hysteresis fill.
      1. Compute local mean via large-sigma Gaussian (``block_size_um``).
      2. **Seed mask**: ``signal < local_mean * threshold_fraction`` — strict,
         catches definitely-dark lumen cores.
      3. **Fill mask**: ``signal < local_mean * threshold_fraction * fill_expansion``
         — permissive, includes lumen boundaries + interior.
      4. **Hysteresis**: keep fill-mask connected components that contain at
         least one seed pixel. Fills lumen interiors even when the boundary
         has gaps wider than morphological closing could bridge.
      This rejects inter-tissue-lobe gaps (uniformly dark → local mean also
      dark → threshold low → gap not detected).

    - **Global** (``local_threshold=False``): Otsu on non-zero pixels, scaled
      by ``otsu_multiplier``. Legacy approach with morphological closing.

    Args:
        channels: List of 2D float32 channel arrays.
        pixel_size_um: Pixel size at the current scale in micrometres.
        blur_sigma_um: Gaussian blur sigma in micrometres (noise smoothing).
        otsu_multiplier: Multiply Otsu threshold (only used when local_threshold=False).
        morph_close_um: Morphological closing radius in um (global mode only).
        morph_open_um: Morphological opening radius in um (remove noise).
        local_threshold: If True, use local Gaussian-mean with hysteresis fill.
        block_size_um: Gaussian sigma for local mean computation, in micrometres.
        threshold_fraction: Seed threshold (pixel < local_mean * fraction).
        fill_expansion: Fill threshold multiplier on top of threshold_fraction.
            E.g., fraction=0.5, expansion=1.5 → fill at 0.75 of local mean.

    Returns:
        Tuple of (binary_mask, signal_sum, threshold_value).
        binary_mask: bool array where True = lumen candidate (dark region).
        signal_sum: float32 blurred summed signal.
        threshold_value: Otsu threshold (global mode) or 0.0 (local mode).
    """
    # Percentile-normalize each channel independently, sum incrementally to save memory.
    # Select channel subset if specified
    if channel_indices is not None:
        channels = [channels[i] for i in channel_indices if i < len(channels)]

    signal = np.zeros(channels[0].shape, dtype=np.float32)
    for i in range(len(channels)):
        channels[i] = _percentile_normalize(channels[i])
        signal += channels[i]
        channels[i] = None  # free memory immediately
    del channels

    # Gaussian blur (noise smoothing — small sigma, e.g. 5um)
    sigma_px = blur_sigma_um / pixel_size_um
    if sigma_px > 0.5:
        signal = gaussian_filter(signal, sigma=sigma_px)

    # Check for sufficient data
    nonzero_signal = signal[signal > 0]
    if len(nonzero_signal) < 100:
        logger.warning("Too few non-zero pixels (%d) for thresholding", len(nonzero_signal))
        return np.zeros(signal.shape, dtype=bool), signal, 0.0

    if local_threshold:
        # Local Gaussian-mean threshold with seeded watershed.
        #
        # 1. Compute local mean via large-sigma Gaussian.
        # 2. Find seed pixels (strict: signal < local_mean * fraction).
        # 3. Define growth mask (permissive: signal < local_mean * fraction * expansion).
        # 4. Watershed from seed components into growth mask — each seed expands
        #    to fill its dark basin but stops at bright walls (vessel walls) and
        #    at territory of neighboring seeds (prevents capillary merging).
        block_sigma_px = max(1.0, block_size_um / pixel_size_um)
        local_mean = gaussian_filter(signal, sigma=block_sigma_px)

        # Seed: strict — definitely-dark lumen cores
        seed_thresh = local_mean * threshold_fraction
        seed_mask = (signal > 0) & (signal < seed_thresh)

        # Label seed components as watershed markers
        seed_labels, n_seeds = ndi_label(seed_mask)

        if n_seeds > 0:
            # Growth mask: permissive dark region where seeds can expand into
            growth_thresh = local_mean * min(threshold_fraction * fill_expansion, 0.95)
            growth_mask = (signal > 0) & (signal < growth_thresh)

            # Watershed: signal is the elevation map (bright = ridges = walls).
            # Seeds grow downhill into dark basins, stop at bright boundaries.
            ws_labels = skimage_watershed(signal, markers=seed_labels, mask=growth_mask)
            binary = ws_labels > 0
        else:
            binary = seed_mask

        threshold_val = 0.0
    else:
        # Legacy global Otsu
        if np.std(nonzero_signal) < 1e-6:
            logger.warning("Near-zero variance in signal — cannot threshold")
            return np.zeros(signal.shape, dtype=bool), signal, 0.0
        otsu_raw = threshold_otsu(nonzero_signal)
        otsu_t = otsu_raw * otsu_multiplier
        binary = (signal > 0) & (signal <= otsu_t)
        threshold_val = otsu_raw

        # Morphological close (fill gaps — only in global mode; local mode uses hysteresis)
        close_px = max(1, int(round(morph_close_um / pixel_size_um)))
        binary = binary_closing(binary, structure=skimage_disk(close_px))

    # Morphological open (remove noise specks — both modes)
    open_px = max(1, int(round(morph_open_um / pixel_size_um)))
    binary = binary_opening(binary, structure=skimage_disk(open_px))

    # Output complement mode: tissue regions instead of dark holes
    if output_complement:
        tissue_floor = 0.05  # minimal signal to count as tissue
        binary = ~binary & (signal > tissue_floor)

    return binary, signal, threshold_val


def _label_and_filter(
    binary: np.ndarray,
    pixel_size_um: float,
    min_area_um2: float,
    max_area_um2: float | None = None,
) -> tuple[np.ndarray, int]:
    """Label connected components and remove those outside area bounds.

    Args:
        binary: 2D bool mask.
        pixel_size_um: Pixel size in um at this scale.
        min_area_um2: Minimum area in um^2.
        max_area_um2: Maximum area in um^2 (None = no upper bound).

    Returns:
        Tuple of (labels, n_kept). Labels is int32 with 0=background.
    """
    labels, n_components = ndi_label(binary)
    if n_components == 0:
        return labels, 0

    area_per_px_um2 = pixel_size_um**2
    min_area_px = max(1, int(round(min_area_um2 / area_per_px_um2)))
    max_area_px = None
    if max_area_um2 is not None:
        max_area_px = int(round(max_area_um2 / area_per_px_um2))

    # Compute component sizes using bincount (fast for large label arrays)
    sizes = np.bincount(labels.ravel())
    keep_mask = sizes >= min_area_px
    if max_area_px is not None:
        keep_mask &= sizes <= max_area_px
    keep_mask[0] = False  # background is never kept

    # Relabel kept components contiguously
    old_to_new = np.zeros(len(sizes), dtype=np.int32)
    new_label = 0
    for old_label in range(1, len(sizes)):
        if keep_mask[old_label]:
            new_label += 1
            old_to_new[old_label] = new_label

    if new_label == 0:
        return np.zeros_like(labels), 0

    labels = old_to_new[labels]
    return labels, new_label


# ---------------------------------------------------------------------------
# Step 1: Coarsest-scale discovery
# ---------------------------------------------------------------------------


def discover_coarse_lumens(
    zarr_root: Any,
    scale_info: tuple[int, int, str, int],
    base_pixel_size: float,
    blur_sigma_um: float,
    otsu_multiplier: float,
    min_area_um2: float,
    existing_lumens: list[dict] | None = None,
    save_debug: bool = False,
    output_dir: Path | None = None,
    threshold_kwargs: dict | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Discover lumen seeds at a scale via whole-image processing.

    Suitable for coarse scales (64x, 16x) where the image fits in memory.
    Optionally masks out regions covered by ``existing_lumens``.

    Args:
        zarr_root: Opened zarr group.
        scale_info: (scale, zarr_level, level_key, extra_ds) tuple.
        base_pixel_size: Pixel size at level 0 in um.
        blur_sigma_um: Gaussian blur sigma in um.
        otsu_multiplier: Otsu threshold multiplier.
        min_area_um2: Minimum lumen area in um^2.
        existing_lumens: Previously discovered lumens to mask out (actual masks, not bboxes).
        save_debug: Save debug masks as PNG.
        output_dir: Directory for debug output.

    Returns:
        Tuple of (lumens, labels, signal).
        lumens: List of dicts with bbox, mask, area info per lumen.
        labels: 2D int32 label array.
        signal: 2D float32 blurred sum image.
    """
    threshold_kwargs = threshold_kwargs or {}
    scale, zarr_level, level_key, extra_ds = scale_info
    pixel_size = base_pixel_size * scale
    level_data = zarr_root[level_key]
    eff_shape = get_effective_shape(level_data, extra_ds)

    logger.info(
        "Discovery at %dx (level %s + %dx ds, shape %s, pixel=%.2f um)",
        scale,
        level_key,
        extra_ds,
        eff_shape,
        pixel_size,
    )

    channels = _read_all_channels(level_data, extra_ds)
    logger.info("  Read %d channels, shape %s", len(channels), channels[0].shape)

    binary, signal, otsu_raw = _channel_sum_and_threshold(
        channels, pixel_size, blur_sigma_um, otsu_multiplier, **threshold_kwargs
    )
    dark_pct = 100.0 * binary.sum() / max(1, binary.size)
    if threshold_kwargs.get("local_threshold", False):
        logger.info(
            "  Local threshold (sigma=%.0fum, frac=%.2f), dark pixels=%d (%.1f%%)",
            threshold_kwargs.get("block_size_um", 100),
            threshold_kwargs.get("threshold_fraction", 0.5),
            binary.sum(),
            dark_pct,
        )
    else:
        logger.info(
            "  Otsu=%.4f, threshold=%.4f, dark pixels=%d (%.1f%%)",
            otsu_raw,
            otsu_raw * otsu_multiplier,
            binary.sum(),
            dark_pct,
        )

    # Mask out already-discovered lumens using actual masks (not bboxes)
    if existing_lumens:
        n_masked = 0
        for lumen in existing_lumens:
            src_scale = lumen["refined_scale"]
            ratio = src_scale / scale
            by = int(lumen["bbox_y"] * ratio)
            bx = int(lumen["bbox_x"] * ratio)
            bh = max(1, int(lumen["bbox_h"] * ratio))
            bw = max(1, int(lumen["bbox_w"] * ratio))
            ey1 = max(0, by)
            ex1 = max(0, bx)
            ey2 = min(binary.shape[0], by + bh)
            ex2 = min(binary.shape[1], bx + bw)
            if ey2 <= ey1 or ex2 <= ex1:
                continue
            mask = lumen.get("mask_at_scale")
            if mask is not None:
                resized = cv2.resize(
                    mask.astype(np.uint8),
                    (ex2 - ex1, ey2 - ey1),
                    interpolation=cv2.INTER_NEAREST,
                )
                binary[ey1:ey2, ex1:ex2] &= resized == 0
            else:
                binary[ey1:ey2, ex1:ex2] = False
            n_masked += 1
        if n_masked:
            logger.info("  Masked out %d existing lumens (mask-based)", n_masked)

    labels, n_kept = _label_and_filter(binary, pixel_size, min_area_um2)
    logger.info("  %d coarse lumens after size filter (>%.0f um^2)", n_kept, min_area_um2)

    if save_debug and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save binary mask
        cv2.imwrite(
            str(output_dir / f"debug_binary_{scale}x.png"),
            (binary.astype(np.uint8) * 255),
        )
        # Save labels (visible colors)
        if n_kept > 0:
            label_vis = ((labels % 255) + 1).astype(np.uint8)
            label_vis[labels == 0] = 0
            cv2.imwrite(str(output_dir / f"debug_labels_{scale}x.png"), label_vis)

    # Extract per-lumen bboxes and metadata using find_objects for O(1) bbox lookup
    from scipy.ndimage import find_objects

    lumens = []
    area_per_px_um2 = pixel_size**2
    pad = 3  # padding for dilation boundary ring
    img_h, img_w = labels.shape
    slices = find_objects(labels)  # list of (slice_y, slice_x) per label
    for lbl in range(1, n_kept + 1):
        sl = slices[lbl - 1]
        if sl is None:
            continue
        bbox_y = sl[0].start
        bbox_x = sl[1].start
        bbox_h = sl[0].stop - sl[0].start
        bbox_w = sl[1].stop - sl[1].start

        # Crop to padded bbox for interior/boundary computation
        cy1 = max(0, bbox_y - pad)
        cx1 = max(0, bbox_x - pad)
        cy2 = min(img_h, bbox_y + bbox_h + pad)
        cx2 = min(img_w, bbox_x + bbox_w + pad)
        crop_mask = labels[cy1:cy2, cx1:cx2] == lbl
        crop_signal = signal[cy1:cy2, cx1:cx2]
        area_px = int(crop_mask.sum())

        interior_median = float(np.median(crop_signal[crop_mask]))
        crop_u8 = crop_mask.astype(np.uint8)
        _dil_px = max(2, int(round(_BOUNDARY_RING_UM / pixel_size)))
        dilated = cv2.dilate(crop_u8, np.ones((2 * _dil_px + 1,) * 2, np.uint8), iterations=1)
        boundary = (dilated > 0) & (~crop_mask) & (crop_signal > 0)
        boundary_median = float(np.median(crop_signal[boundary])) if boundary.any() else 0.0
        contrast = boundary_median / max(interior_median, 1e-6)

        # Store mask cropped to exact bbox (no padding)
        # .copy() to avoid holding a view into the larger crop_mask array
        mask_crop = crop_mask[
            bbox_y - cy1 : bbox_y - cy1 + bbox_h, bbox_x - cx1 : bbox_x - cx1 + bbox_w
        ].copy()

        lumens.append(
            {
                "label": lbl,
                "bbox_y": bbox_y,
                "bbox_x": bbox_x,
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "area_px": area_px,
                "area_um2": area_px * area_per_px_um2,
                "discovery_scale": scale,
                "refined_scale": scale,
                "interior_median": interior_median,
                "boundary_median": boundary_median,
                "contrast_ratio": contrast,
                "mask_at_scale": mask_crop,
            }
        )

    logger.info(
        "  Median contrast ratio: %.2f",
        np.median([l["contrast_ratio"] for l in lumens]) if lumens else 0.0,
    )
    return lumens, labels, signal


# ---------------------------------------------------------------------------
# Mask re-derivation from zarr (for memory-efficient operation)
# ---------------------------------------------------------------------------


def rederive_mask_from_zarr(
    lumen: dict,
    zarr_root: Any,
    scale_infos_by_scale: dict[int, tuple],
    base_pixel_size: float,
    blur_sigma_um: float,
    otsu_multiplier: float,
    threshold_kwargs: dict | None = None,
) -> np.ndarray | None:
    """Re-derive a lumen's binary mask from zarr data at its current refined_scale.

    Reads a padded crop around the lumen's bbox, thresholds it, and identifies
    the connected component at the bbox center. Used to recover masks that were
    dropped to save memory.

    Returns the mask cropped to exact bbox dimensions, or None if no valid
    component is found.
    """
    rs = lumen["refined_scale"]
    si = scale_infos_by_scale.get(rs)
    if si is None:
        return None

    scale, _, level_key, extra_ds = si
    pixel_size = base_pixel_size * scale

    level_data = zarr_root[level_key]
    eff_shape = get_effective_shape(level_data, extra_ds)

    by, bx, bh, bw = lumen["bbox_y"], lumen["bbox_x"], lumen["bbox_h"], lumen["bbox_w"]

    # 20% padding (matches refine_lumens_at_scale)
    pad_y = max(2, int(bh * 0.2))
    pad_x = max(2, int(bw * 0.2))
    crop_y = max(0, by - pad_y)
    crop_x = max(0, bx - pad_x)
    crop_h = min(bh + 2 * pad_y, eff_shape[1] - crop_y)
    crop_w = min(bw + 2 * pad_x, eff_shape[2] - crop_x)

    if crop_h < 3 or crop_w < 3:
        return None

    channels = _read_all_channels(level_data, extra_ds, crop_y, crop_x, crop_h, crop_w)
    binary, _, _ = _channel_sum_and_threshold(
        channels, pixel_size, blur_sigma_um, otsu_multiplier, **(threshold_kwargs or {})
    )

    if not binary.any():
        return None

    labels, n_comp = ndi_label(binary)
    if n_comp == 0:
        return None

    # Find component at bbox center (in crop coordinates)
    cy = by - crop_y + bh // 2
    cx = bx - crop_x + bw // 2
    cy = min(max(0, cy), labels.shape[0] - 1)
    cx = min(max(0, cx), labels.shape[1] - 1)
    target_label = int(labels[cy, cx])

    # Fallback 1: search 5x5 neighborhood
    if target_label == 0:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]:
                    if labels[ny, nx] > 0:
                        target_label = int(labels[ny, nx])
                        break
            if target_label > 0:
                break

    # Fallback 2: component with max overlap inside bbox region
    if target_label == 0:
        rel_y = max(0, by - crop_y)
        rel_x = max(0, bx - crop_x)
        bbox_region = labels[rel_y : rel_y + bh, rel_x : rel_x + bw]
        if bbox_region.size > 0:
            counts = np.bincount(bbox_region.ravel())
            counts[0] = 0  # ignore background
            if counts.max() > 0:
                target_label = int(counts.argmax())

    if target_label == 0:
        return None

    # Extract mask cropped to exact bbox coords, padded to (bh, bw)
    comp_mask = labels == target_label
    rel_y = max(0, by - crop_y)
    rel_x = max(0, bx - crop_x)
    end_y = min(rel_y + bh, comp_mask.shape[0])
    end_x = min(rel_x + bw, comp_mask.shape[1])
    sliced = comp_mask[rel_y:end_y, rel_x:end_x]
    if sliced.shape == (bh, bw):
        return sliced.copy()
    # Pad to expected dimensions (edge lumens where crop is clipped)
    mask = np.zeros((bh, bw), dtype=bool)
    mask[: sliced.shape[0], : sliced.shape[1]] = sliced
    return mask


def rederive_masks_batch(
    lumens: list[dict],
    zarr_root: Any,
    scale_infos_by_scale: dict[int, tuple],
    base_pixel_size: float,
    blur_sigma_um: float,
    otsu_multiplier: float,
    label: str = "",
) -> int:
    """Re-derive masks for all lumens where mask_at_scale is None.

    Returns the number of successfully re-derived masks.
    """
    n_rederived = 0
    n_failed = 0
    for l in lumens:
        if l.get("mask_at_scale") is not None:
            continue
        l["mask_at_scale"] = rederive_mask_from_zarr(
            l,
            zarr_root,
            scale_infos_by_scale,
            base_pixel_size,
            blur_sigma_um,
            otsu_multiplier,
        )
        if l["mask_at_scale"] is not None:
            n_rederived += 1
        else:
            n_failed += 1
    if n_rederived or n_failed:
        logger.info(
            "Re-derived %d masks%s%s",
            n_rederived,
            f" ({n_failed} failed)" if n_failed else "",
            f" for {label}" if label else "",
        )
    return n_rederived


# ---------------------------------------------------------------------------
# Per-scale checkpointing
# ---------------------------------------------------------------------------

# Keys to strip from lumen dicts before saving to checkpoint
_CHECKPOINT_STRIP_KEYS = {"mask_at_scale", "label"}


def save_discovery_checkpoint(
    all_lumens: list[dict],
    scale: int,
    output_dir: Path,
    parameters: dict,
) -> Path:
    """Save cumulative lumen discovery state after completing a scale."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "scale": scale,
        "cumulative_n_lumens": len(all_lumens),
        "parameters": parameters,
        "lumens": [
            {k: v for k, v in l.items() if k not in _CHECKPOINT_STRIP_KEYS} for l in all_lumens
        ],
    }
    path = output_dir / f"checkpoint_discovery_scale{scale}x.json"
    atomic_json_dump(checkpoint, str(path))
    logger.info("Saved checkpoint: %d lumens through %dx → %s", len(all_lumens), scale, path.name)
    return path


def load_discovery_checkpoint(
    output_dir: Path,
    scales: list[int],
    parameters: dict,
) -> tuple[list[dict], int]:
    """Load the most advanced (finest-scale) valid checkpoint.

    Args:
        output_dir: Directory containing checkpoint files.
        scales: List of scales sorted descending (coarsest first).
        parameters: Current run parameters for validation.

    Returns:
        (lumens, last_completed_scale). Returns ([], 0) if no checkpoint found.
    """
    # Check from finest to coarsest — use the most advanced checkpoint
    for scale in reversed(scales):
        path = output_dir / f"checkpoint_discovery_scale{scale}x.json"
        if not path.exists():
            continue
        try:
            checkpoint = fast_json_load(str(path))
        except Exception:
            logger.warning("Failed to load checkpoint %s, skipping", path.name)
            continue

        # Validate parameters match
        stored_params = checkpoint.get("parameters", {})
        mismatches = []
        for key in ("otsu_multiplier", "blur_sigma_um", "min_area_um2", "tile_size", "scales"):
            stored = stored_params.get(key)
            current = parameters.get(key)
            if stored is not None and current is not None and stored != current:
                mismatches.append(f"{key}: {stored} → {current}")
        if mismatches:
            logger.warning(
                "Checkpoint %s has different parameters: %s. Starting fresh.",
                path.name,
                ", ".join(mismatches),
            )
            return [], 0

        lumens = checkpoint.get("lumens", [])
        # Ensure mask_at_scale is None (not stored in checkpoint)
        for l in lumens:
            l["mask_at_scale"] = None
        logger.info(
            "Loaded checkpoint: %d lumens through %dx from %s",
            len(lumens),
            scale,
            path.name,
        )
        return lumens, scale

    return [], 0


# ---------------------------------------------------------------------------
# Step 2: Refine at finer scales
# ---------------------------------------------------------------------------


def refine_lumens_at_scale(
    lumens: list[dict],
    zarr_root: Any,
    src_scale: int,
    dst_scale_info: tuple[int, int, str, int],
    base_pixel_size: float,
    blur_sigma_um: float,
    otsu_multiplier: float,
    save_debug: bool = False,
    output_dir: Path | None = None,
    threshold_kwargs: dict | None = None,
) -> list[dict]:
    """Refine lumen boundaries by re-thresholding at a finer scale.

    For each lumen, reads a crop at the finer scale, re-thresholds locally,
    and keeps dark components overlapping the coarse seed. If the coarse seed
    splits at finer scale, takes the union.

    Args:
        lumens: List of lumen dicts from coarser scale.
        zarr_root: Opened zarr group.
        src_scale: The scale at which lumens were previously defined.
        dst_scale_info: Target finer scale (scale, level, key, extra_ds).
        base_pixel_size: Level-0 pixel size in um.
        blur_sigma_um: Gaussian blur sigma in um.
        otsu_multiplier: Otsu threshold multiplier.
        save_debug: Save debug crops as PNG.
        output_dir: Debug output directory.

    Returns:
        Updated lumens list with refined boundaries.
    """
    dst_scale, _, dst_level_key, dst_extra_ds = dst_scale_info
    dst_pixel_size = base_pixel_size * dst_scale
    dst_level_data = zarr_root[dst_level_key]
    dst_eff_shape = get_effective_shape(dst_level_data, dst_extra_ds)
    scale_ratio = src_scale / dst_scale  # e.g., 64/16 = 4

    logger.info(
        "Step 2: Refining %d lumens from %dx to %dx (ratio=%.0f, dst shape=%s)",
        len(lumens),
        src_scale,
        dst_scale,
        scale_ratio,
        dst_eff_shape,
    )

    refined_count = 0
    for i, lumen in enumerate(lumens):
        # Map coarse bbox to finer-scale coords with 20% padding
        pad_frac = 0.2
        by = int(lumen["bbox_y"] * scale_ratio)
        bx = int(lumen["bbox_x"] * scale_ratio)
        bh = int(lumen["bbox_h"] * scale_ratio)
        bw = int(lumen["bbox_w"] * scale_ratio)
        pad_y = max(1, int(bh * pad_frac))
        pad_x = max(1, int(bw * pad_frac))

        # Clamp to image bounds
        crop_y = max(0, by - pad_y)
        crop_x = max(0, bx - pad_x)
        crop_h = min(bh + 2 * pad_y, dst_eff_shape[1] - crop_y)
        crop_w = min(bw + 2 * pad_x, dst_eff_shape[2] - crop_x)

        if crop_h < 3 or crop_w < 3:
            continue

        # Read all channels in a single zarr access (fast I/O)
        channels = read_all_channels_crop(
            dst_level_data, dst_extra_ds, crop_y, crop_x, crop_h, crop_w
        )

        # Simple median-based threshold for refinement crops.
        # The crop is small and centered on a known lumen — no need for the
        # full watershed pipeline. Median is robust to the lumen/wall ratio.
        signal = np.zeros(channels[0].shape, dtype=np.float32)
        for ch_data in channels:
            signal += _percentile_normalize(ch_data)

        # Blur + threshold: dark if signal < median * fraction
        sigma_px = blur_sigma_um / dst_pixel_size
        if sigma_px > 0.5:
            signal = gaussian_filter(signal, sigma=sigma_px)

        nonzero = signal[signal > 0]
        if len(nonzero) < 10:
            continue
        crop_median = float(np.median(nonzero))
        frac = (threshold_kwargs or {}).get("threshold_fraction", 0.5)
        # Local threshold OR globally dark (p5 floor) — catches lumen interiors
        # in large chambers where the local median is itself dark.
        # The global_dark_floor is passed via threshold_kwargs from discovery.
        global_dark_floor = (threshold_kwargs or {}).get("global_dark_floor", None)
        binary = (signal > 0) & (signal < crop_median * frac)
        if global_dark_floor is not None:
            binary |= (signal > 0) & (signal < global_dark_floor)

        if not binary.any():
            continue

        # Upscale coarse mask to finer resolution as seed
        coarse_mask = lumen.get("mask_at_scale")
        if coarse_mask is None:
            continue
        seed_h = int(round(coarse_mask.shape[0] * scale_ratio))
        seed_w = int(round(coarse_mask.shape[1] * scale_ratio))
        if seed_h < 1 or seed_w < 1:
            continue
        seed_upscaled = cv2.resize(
            coarse_mask.astype(np.uint8), (seed_w, seed_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Place seed in crop coordinates
        seed_in_crop = np.zeros(binary.shape, dtype=bool)
        # Offset of the original bbox within the padded crop
        off_y = by - crop_y
        off_x = bx - crop_x
        # Clamp seed placement
        sy1 = max(0, off_y)
        sx1 = max(0, off_x)
        sy2 = min(binary.shape[0], off_y + seed_h)
        sx2 = min(binary.shape[1], off_x + seed_w)
        # Corresponding region in seed
        ry1 = sy1 - off_y
        rx1 = sx1 - off_x
        ry2 = ry1 + (sy2 - sy1)
        rx2 = rx1 + (sx2 - sx1)
        if (
            sy2 > sy1
            and sx2 > sx1
            and ry2 <= seed_upscaled.shape[0]
            and rx2 <= seed_upscaled.shape[1]
        ):
            seed_in_crop[sy1:sy2, sx1:sx2] = seed_upscaled[ry1:ry2, rx1:rx2]

        # Label dark components in the crop
        crop_labels, n_crop = ndi_label(binary)
        if n_crop == 0:
            continue

        # Keep components that overlap the seed (union if seed splits)
        refined_mask = np.zeros(binary.shape, dtype=bool)
        for lbl in range(1, n_crop + 1):
            comp = crop_labels == lbl
            overlap = comp & seed_in_crop
            if overlap.any():
                refined_mask |= comp

        if not refined_mask.any():
            continue

        # Update lumen with refined data at dst_scale
        ys, xs = np.where(refined_mask)
        new_by = int(ys.min())
        new_bx = int(xs.min())
        new_bh = int(ys.max()) - new_by + 1
        new_bw = int(xs.max()) - new_bx + 1
        area_px = int(refined_mask.sum())
        area_per_px_um2 = dst_pixel_size**2

        # Interior / boundary brightness
        interior_median = float(np.median(signal[refined_mask]))
        comp_u8 = refined_mask.astype(np.uint8)
        _dil_px = max(2, int(round(_BOUNDARY_RING_UM / dst_pixel_size)))
        dilated = cv2.dilate(comp_u8, np.ones((2 * _dil_px + 1,) * 2, np.uint8), iterations=1)
        boundary = (dilated > 0) & (~refined_mask) & (signal > 0)
        boundary_median = float(np.median(signal[boundary])) if boundary.any() else 0.0
        contrast = boundary_median / max(interior_median, 1e-6)

        # Store in GLOBAL finer-scale coords (crop_y + local offset)
        lumen["bbox_y"] = crop_y + new_by
        lumen["bbox_x"] = crop_x + new_bx
        lumen["bbox_h"] = new_bh
        lumen["bbox_w"] = new_bw
        lumen["area_px"] = area_px
        lumen["area_um2"] = area_px * area_per_px_um2
        lumen["refined_scale"] = dst_scale
        lumen["interior_median"] = interior_median
        lumen["boundary_median"] = boundary_median
        lumen["contrast_ratio"] = contrast
        # .copy() prevents retaining the larger refined_mask crop array
        lumen["mask_at_scale"] = refined_mask[
            new_by : new_by + new_bh, new_bx : new_bx + new_bw
        ].copy()
        refined_count += 1

        if save_debug and output_dir is not None and i < 20:
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(output_dir / f"debug_refine_{dst_scale}x_lumen{i}.png"),
                (refined_mask.astype(np.uint8) * 255),
            )

        if (i + 1) % 500 == 0:
            logger.info(
                "  Refining %d/%d lumens at %dx (%d refined so far)",
                i + 1,
                len(lumens),
                dst_scale,
                refined_count,
            )

    logger.info("  Refined %d/%d lumens at %dx", refined_count, len(lumens), dst_scale)
    return lumens


# ---------------------------------------------------------------------------
# Step 3: Fine-scale discovery (small lumens)
# ---------------------------------------------------------------------------


def discover_fine_lumens(
    zarr_root: Any,
    scale_info: tuple[int, int, str, int],
    base_pixel_size: float,
    existing_lumens: list[dict],
    blur_sigma_um: float,
    otsu_multiplier: float,
    min_area_um2: float,
    max_area_um2: float | None,
    tile_size: int,
    overlap: int = 750,
    tissue_mask: np.ndarray | None = None,
    tissue_mask_scale: int = 1,
    save_debug: bool = False,
    output_dir: Path | None = None,
    threshold_kwargs: dict | None = None,
) -> list[dict]:
    """Discover lumens at a scale via tiled processing.

    Tiles the image, thresholds each tile, and finds dark components that
    do not overlap existing (already-discovered) lumens.

    If ``tissue_mask`` is provided (a boolean array from a coarser scale),
    tiles that fall entirely outside tissue are skipped for speed.

    Args:
        zarr_root: Opened zarr group.
        scale_info: (scale, level, key, extra_ds) for this fine scale.
        base_pixel_size: Level-0 pixel size in um.
        existing_lumens: Previously discovered lumens (will be masked out).
        blur_sigma_um: Gaussian blur sigma in um.
        otsu_multiplier: Otsu threshold multiplier.
        min_area_um2: Minimum area for new lumens.
        max_area_um2: Maximum area for new lumens.
        tile_size: Tile size in pixels.
        overlap: Tile overlap in pixels.
        tissue_mask: Boolean tissue mask at ``tissue_mask_scale`` for fast tile skipping.
        save_debug: Save debug images.
        output_dir: Debug output directory.

    Returns:
        List of new lumen dicts discovered at this scale.
    """
    scale, _, level_key, extra_ds = scale_info
    pixel_size = base_pixel_size * scale
    level_data = zarr_root[level_key]
    eff_shape = get_effective_shape(level_data, extra_ds)

    max_str = f"{max_area_um2:.0f}" if max_area_um2 is not None else "none"
    logger.info(
        "Discovery at %dx (shape=%s, pixel=%.2f um, min=%.0f, max=%s um^2)",
        scale,
        eff_shape,
        pixel_size,
        min_area_um2,
        max_str,
    )

    # Build a coarse-resolution exclusion mask from actual lumen masks (not bboxes).
    # The mask stays at the coarsest available scale to keep memory tiny.
    # For each tile at this scale, we map its footprint to the exclusion mask coords
    # and mask out only pixels that fall inside an actual lumen polygon.
    excl_mask: np.ndarray | None = None
    excl_scale: int = 0
    if existing_lumens:
        # Find the coarsest scale among existing lumens for the exclusion mask
        excl_scale = max(l["discovery_scale"] for l in existing_lumens)
        # We need the image dimensions at excl_scale — derive from this scale's shape
        ratio_to_excl = scale / excl_scale  # e.g., 4/64 = 0.0625
        excl_h = max(1, int(eff_shape[1] * ratio_to_excl))
        excl_w = max(1, int(eff_shape[2] * ratio_to_excl))
        excl_mask = np.zeros((excl_h, excl_w), dtype=bool)

        for lumen in existing_lumens:
            src_scale = lumen["refined_scale"]
            # Map lumen bbox to exclusion mask coords
            ratio = src_scale / excl_scale
            by = int(lumen["bbox_y"] * ratio)
            bx = int(lumen["bbox_x"] * ratio)
            bh = max(1, int(lumen["bbox_h"] * ratio))
            bw = max(1, int(lumen["bbox_w"] * ratio))
            mask = lumen.get("mask_at_scale")
            if mask is not None:
                # Resize the actual lumen mask to exclusion scale
                resized = cv2.resize(
                    mask.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST
                )
                ey1 = max(0, by)
                ex1 = max(0, bx)
                ey2 = min(excl_h, by + bh)
                ex2 = min(excl_w, bx + bw)
                # Clip resized mask to fit
                my1 = ey1 - by
                mx1 = ex1 - bx
                my2 = my1 + (ey2 - ey1)
                mx2 = mx1 + (ex2 - ex1)
                excl_mask[ey1:ey2, ex1:ex2] |= resized[my1:my2, mx1:mx2] > 0
            else:
                # Fallback: use bbox if mask not available
                ey1 = max(0, by)
                ex1 = max(0, bx)
                ey2 = min(excl_h, by + bh)
                ex2 = min(excl_w, bx + bw)
                excl_mask[ey1:ey2, ex1:ex2] = True

        excl_pct = 100.0 * excl_mask.sum() / max(1, excl_mask.size)
        logger.info(
            "  Exclusion mask: %d lumens painted at %dx (%dx%d, %.1f%% excluded)",
            len(existing_lumens),
            excl_scale,
            excl_w,
            excl_h,
            excl_pct,
        )
    # Generate tiles
    tiles = generate_tiles(eff_shape, tile_size, overlap)
    logger.info("  Processing %d tiles (%d px, %d overlap)", len(tiles), tile_size, overlap)

    # Pre-compute tissue mask ratio for fast tile skipping
    tissue_ratio = tissue_mask_scale / scale if tissue_mask is not None else 0
    n_skipped_tissue = 0

    new_lumens: list[dict] = []
    area_per_px_um2 = pixel_size**2

    for ti, (ty, tx, th, tw) in enumerate(tiles):
        # Fast skip: check if tile overlaps any tissue in the coarse tissue mask
        if tissue_mask is not None:
            # Map tile coords to tissue mask coords
            tm_y1 = max(0, int(ty * tissue_ratio))
            tm_x1 = max(0, int(tx * tissue_ratio))
            tm_y2 = min(tissue_mask.shape[0], int((ty + th) * tissue_ratio) + 1)
            tm_x2 = min(tissue_mask.shape[1], int((tx + tw) * tissue_ratio) + 1)
            if tm_y2 > tm_y1 and tm_x2 > tm_x1:
                if not tissue_mask[tm_y1:tm_y2, tm_x1:tm_x2].any():
                    n_skipped_tissue += 1
                    continue

        # Read all channels for this tile
        channels = _read_all_channels(level_data, extra_ds, ty, tx, th, tw)

        # Threshold
        binary, signal, otsu_raw = _channel_sum_and_threshold(
            channels,
            pixel_size,
            blur_sigma_um,
            otsu_multiplier,
            **(threshold_kwargs or {}),
        )

        if not binary.any():
            continue

        # Mask out existing lumens using the coarse exclusion mask
        if excl_mask is not None:
            # Map tile footprint to exclusion mask coords
            excl_ratio = scale / excl_scale  # e.g., 4/64 = 0.0625
            em_y1 = max(0, int(ty * excl_ratio))
            em_x1 = max(0, int(tx * excl_ratio))
            em_y2 = min(excl_mask.shape[0], int((ty + th) * excl_ratio) + 1)
            em_x2 = min(excl_mask.shape[1], int((tx + tw) * excl_ratio) + 1)
            if em_y2 > em_y1 and em_x2 > em_x1:
                excl_crop = excl_mask[em_y1:em_y2, em_x1:em_x2]
                if excl_crop.any():
                    # Upscale exclusion crop to tile resolution and apply
                    excl_tile = cv2.resize(
                        excl_crop.astype(np.uint8),
                        (tw, th),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    binary[excl_tile > 0] = False

        if not binary.any():
            continue

        # Label and filter
        labels, n_comp = _label_and_filter(binary, pixel_size, min_area_um2, max_area_um2)
        if n_comp == 0:
            continue

        for lbl in range(1, n_comp + 1):
            comp = labels == lbl
            ys, xs = np.where(comp)
            local_by = int(ys.min())
            local_bx = int(xs.min())
            local_bh = int(ys.max()) - local_by + 1
            local_bw = int(xs.max()) - local_bx + 1
            area_px = int(comp.sum())

            # Interior / boundary brightness
            interior_median = float(np.median(signal[comp]))
            comp_u8 = comp.astype(np.uint8)
            _dil_px = max(2, int(round(_BOUNDARY_RING_UM / pixel_size)))
            dilated = cv2.dilate(comp_u8, np.ones((2 * _dil_px + 1,) * 2, np.uint8), iterations=1)
            boundary_ring = (dilated > 0) & (~comp) & (signal > 0)
            boundary_median = (
                float(np.median(signal[boundary_ring])) if boundary_ring.any() else 0.0
            )
            contrast = boundary_median / max(interior_median, 1e-6)

            # Global coords at this scale
            global_by = ty + local_by
            global_bx = tx + local_bx

            new_lumens.append(
                {
                    "bbox_y": global_by,
                    "bbox_x": global_bx,
                    "bbox_h": local_bh,
                    "bbox_w": local_bw,
                    "area_px": area_px,
                    "area_um2": area_px * area_per_px_um2,
                    "discovery_scale": scale,
                    "refined_scale": scale,
                    "interior_median": interior_median,
                    "boundary_median": boundary_median,
                    "contrast_ratio": contrast,
                    # .copy() prevents retaining entire tile-sized comp array (~9MB)
                    "mask_at_scale": comp[
                        local_by : local_by + local_bh, local_bx : local_bx + local_bw
                    ].copy(),
                }
            )

        if (ti + 1) % 50 == 0:
            logger.info(
                "  Tile %d/%d: %d new candidates so far", ti + 1, len(tiles), len(new_lumens)
            )
            gc.collect()  # free accumulated zarr chunk caches

    if n_skipped_tissue:
        logger.info("  Skipped %d/%d tiles (no tissue)", n_skipped_tissue, len(tiles))
    logger.info("  %d raw candidates from %dx tiles", len(new_lumens), scale)

    # Cross-tile dedup: IoU >= 0.3, keep larger
    if len(new_lumens) > 1:
        new_lumens = _dedup_by_bbox_iou(new_lumens, iou_threshold=0.3)
        logger.info("  %d candidates after cross-tile dedup", len(new_lumens))

    return new_lumens


def _dedup_by_bbox_iou(lumens: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """Deduplicate lumens by bounding-box IoU, keeping the larger one.

    Args:
        lumens: List of lumen dicts with bbox_y, bbox_x, bbox_h, bbox_w, area_px.
        iou_threshold: IoU threshold above which to suppress.

    Returns:
        Deduplicated list.
    """
    if len(lumens) <= 1:
        return lumens

    # Sort by area descending (keep larger first)
    lumens_sorted = sorted(lumens, key=lambda l: l["area_px"], reverse=True)
    kept: list[dict] = []
    suppressed = set()

    for i, li in enumerate(lumens_sorted):
        if i in suppressed:
            continue
        kept.append(li)
        # Suppress smaller lumens that overlap
        y1_i = li["bbox_y"]
        x1_i = li["bbox_x"]
        y2_i = y1_i + li["bbox_h"]
        x2_i = x1_i + li["bbox_w"]
        area_i = li["bbox_h"] * li["bbox_w"]

        for j in range(i + 1, len(lumens_sorted)):
            if j in suppressed:
                continue
            lj = lumens_sorted[j]
            y1_j = lj["bbox_y"]
            x1_j = lj["bbox_x"]
            y2_j = y1_j + lj["bbox_h"]
            x2_j = x1_j + lj["bbox_w"]

            # Intersection
            iy1 = max(y1_i, y1_j)
            ix1 = max(x1_i, x1_j)
            iy2 = min(y2_i, y2_j)
            ix2 = min(x2_i, x2_j)

            if iy2 <= iy1 or ix2 <= ix1:
                continue

            inter = (iy2 - iy1) * (ix2 - ix1)
            area_j = lj["bbox_h"] * lj["bbox_w"]
            union = area_i + area_j - inter
            iou = inter / max(union, 1)

            if iou >= iou_threshold:
                suppressed.add(j)

    return kept


# ---------------------------------------------------------------------------
# Step 4: Extract contours and metadata
# ---------------------------------------------------------------------------


def _darkness_tier(interior_median: float) -> str:
    """Classify lumen darkness from normalized interior median.

    Args:
        interior_median: Median of the normalized channel-sum inside the lumen.

    Returns:
        One of 'very_dark', 'dark', 'moderate', 'light'.
    """
    if interior_median < 0.05:
        return "very_dark"
    elif interior_median < 0.15:
        return "dark"
    elif interior_median < 0.30:
        return "moderate"
    else:
        return "light"


def extract_contours(
    lumens: list[dict],
    base_pixel_size: float,
) -> list[dict]:
    """Extract cv2 contours and build output records.

    Converts per-lumen binary masks to polygon contours in global micrometres.

    Args:
        lumens: List of lumen dicts with mask_at_scale, bbox_y/x, refined_scale.
        base_pixel_size: Level-0 pixel size in um.

    Returns:
        List of output-ready lumen records with contour_global_um and metadata.
    """
    records: list[dict] = []

    for lumen in lumens:
        mask = lumen.get("mask_at_scale")
        if mask is None or mask.size == 0:
            continue

        # Pad mask by 1px so findContours can close edge-touching contours
        padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
        contours_cv, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours_cv:
            continue

        # Take the largest contour
        contour_px = max(contours_cv, key=cv2.contourArea)
        contour_px = contour_px.squeeze(1)  # (N, 2) in (x, y) format from cv2

        if len(contour_px) < 3:
            continue

        # Remove 1px padding offset
        contour_px = contour_px - 1

        # Convert to global um coordinates
        refined_scale = lumen["refined_scale"]
        pixel_size_at_scale = base_pixel_size * refined_scale
        # contour_px is in local mask coords; add bbox offset for global coords
        global_x_px = contour_px[:, 0].astype(np.float64) + lumen["bbox_x"]
        global_y_px = contour_px[:, 1].astype(np.float64) + lumen["bbox_y"]
        contour_global_um = np.column_stack(
            [global_x_px * pixel_size_at_scale, global_y_px * pixel_size_at_scale]
        )

        # Compute metrics
        area_um2 = lumen["area_um2"]
        equiv_diameter = np.sqrt(4.0 * area_um2 / np.pi)
        # Perimeter from contour in um
        diffs = np.diff(contour_global_um, axis=0, append=contour_global_um[:1])
        perimeter_um = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

        # Stable UID from contour centroid (deterministic from geometry)
        centroid_x_um = float(np.mean(contour_global_um[:, 0]))
        centroid_y_um = float(np.mean(contour_global_um[:, 1]))
        uid = f"lumen_{centroid_x_um:.2f}_{centroid_y_um:.2f}"

        records.append(
            {
                "uid": uid,
                "contour_global_um": contour_global_um.tolist(),
                "centroid_x_um": round(centroid_x_um, 2),
                "centroid_y_um": round(centroid_y_um, 2),
                "area_um2": round(area_um2, 1),
                "equiv_diameter_um": round(float(equiv_diameter), 2),
                "perimeter_um": round(perimeter_um, 1),
                "interior_median": round(lumen["interior_median"], 4),
                "boundary_median": round(lumen["boundary_median"], 4),
                "contrast_ratio": round(lumen["contrast_ratio"], 2),
                "discovery_scale": lumen["discovery_scale"],
                "refined_scale": lumen["refined_scale"],
                "darkness_tier": _darkness_tier(lumen["interior_median"]),
                "n_marker_wall": lumen.get("n_marker_wall", 0),
            }
        )

    return records


# ---------------------------------------------------------------------------
# Step 5: Viewer generation
# ---------------------------------------------------------------------------


def _generate_viewer(
    output_json: Path,
    output_html: Path,
    group_field: str,
    czi_path: Path | None,
    display_channels: str,
    channel_names: str | None,
    scale_factor: float,
    scene: int,
    max_contours: int,
    title: str = "Threshold Lumen Detection",
) -> None:
    """Launch generate_contour_viewer.py as a subprocess."""
    script_dir = Path(__file__).resolve().parent
    viewer_script = script_dir / "generate_contour_viewer.py"

    if not viewer_script.exists():
        logger.warning("Contour viewer script not found at %s — skipping viewer.", viewer_script)
        return

    cmd = [
        sys.executable,
        str(viewer_script),
        "--contours",
        str(output_json),
        "--group-field",
        group_field,
        "--max-contours",
        str(max_contours),
        "--title",
        title,
        "--output",
        str(output_html),
    ]

    if czi_path is not None:
        cmd.extend(["--czi-path", str(czi_path)])
        cmd.extend(["--display-channels", display_channels])
        cmd.extend(["--scale-factor", str(scale_factor)])
        cmd.extend(["--scene", str(scene)])
        if channel_names:
            cmd.extend(["--channel-names", channel_names])

    logger.info("Generating viewer: %s", output_html.name)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Viewer generation failed (exit %d):\n%s", result.returncode, result.stderr)
    else:
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.debug("  viewer: %s", line)
        logger.info("Viewer written: %s", output_html)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Coarse-to-fine threshold-based lumen detection from OME-Zarr. "
            "Uses channel-sum + Otsu threshold + connected components (no GPU)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input ---
    p.add_argument(
        "--zarr-path",
        required=True,
        type=Path,
        help="Path to OME-Zarr file (e.g. slide.ome.zarr).",
    )
    p.add_argument(
        "--scales",
        default="2,4,8,16,64",
        help=(
            "Comma-separated scales for processing. Includes discovery + refinement targets. "
            "Default: 2,4,8,16,64."
        ),
    )
    p.add_argument(
        "--discovery-scales",
        default=None,
        help=(
            "Comma-separated scales for lumen discovery (default: coarsest scale only). "
            "Scales in --scales but not here are refinement-only. "
            "E.g., --scales 8,16,64 --discovery-scales 64,16 discovers at 64x+16x, refines to 8x."
        ),
    )

    p.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel indices to include in the sum (default: all). "
        "E.g., --channels 0,3 for nuc+PM only.",
    )
    p.add_argument(
        "--output-complement",
        action="store_true",
        help="Output the complement of dark regions (tissue/organ regions instead of lumens).",
    )

    # --- Threshold parameters ---
    p.add_argument(
        "--no-local-threshold",
        action="store_true",
        help="Disable local thresholding and use global Otsu instead (legacy behavior).",
    )
    p.add_argument(
        "--block-size-um",
        type=float,
        default=2400.0,
        help="Gaussian sigma for local mean computation in um (default: 2400). "
        "Tuned for whole-mount cross-sections with large chambers. "
        "Reduce to 100-500 for tight tissue or small vessel panels.",
    )
    p.add_argument(
        "--threshold-fraction",
        type=float,
        default=0.5,
        help="Local threshold fraction: pixel < local_mean * fraction (default: 0.5).",
    )
    p.add_argument(
        "--fill-expansion",
        type=float,
        default=1.5,
        help="Hysteresis fill expansion: fill_thresh = fraction * expansion (default: 1.5).",
    )
    p.add_argument(
        "--otsu-multiplier",
        type=float,
        default=0.8,
        help="Multiply Otsu threshold (only with --no-local-threshold) (default: 0.8).",
    )
    p.add_argument(
        "--blur-sigma-um",
        type=float,
        default=5.0,
        help="Gaussian blur sigma in micrometres (default: 5.0).",
    )
    p.add_argument(
        "--min-area-um2",
        type=float,
        default=50.0,
        help="Minimum lumen area in um^2 (default: 50.0).",
    )
    p.add_argument(
        "--tissue-threshold",
        type=float,
        default=0.05,
        help="Tissue mask threshold on normalized signal sum (default: 0.05).",
    )

    # --- Tiling ---
    p.add_argument(
        "--tile-size",
        type=int,
        default=3000,
        help="Tile size in pixels for fine-scale discovery (default: 3000).",
    )

    # --- Marker validation (pre-filter before refinement) ---
    p.add_argument(
        "--marker-cells-json",
        type=Path,
        default=None,
        help="Path to marker-classified cell detections JSON. If provided, "
        "only lumens with >=min-marker-cells nearby are refined (huge speedup).",
    )
    p.add_argument(
        "--min-marker-cells",
        type=int,
        default=1,
        help="Minimum marker+ cells in wall to keep for refinement (default: 1).",
    )
    p.add_argument(
        "--marker-classes",
        type=str,
        default="SMA,CD31",
        help="Comma-separated list of marker names to use for pre-filter. "
        "Script looks for '{NAME}_class == positive' in each cell's features. "
        "Default: SMA,CD31. For LYVE1 slide use 'SMA,LYVE1' or just 'LYVE1'.",
    )

    # --- Output ---
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for JSON and debug images.",
    )
    p.add_argument(
        "--save-debug",
        action="store_true",
        help="Save intermediate binary masks as uint8 PNGs.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available. Skips already-completed discovery scales.",
    )

    # --- Viewer ---
    p.add_argument(
        "--czi-path",
        type=Path,
        default=None,
        help="CZI file for fluorescence background in viewer.",
    )
    p.add_argument(
        "--display-channels",
        default="1,3,0",
        help="Comma-separated CZI channel indices for R,G,B viewer (default: 1,3,0).",
    )
    p.add_argument(
        "--channel-names",
        default=None,
        help="Comma-separated channel display names (e.g. 'SMA,CD31,nuc').",
    )
    p.add_argument(
        "--scale-factor",
        type=float,
        default=0.0625,
        help="CZI downsample factor for viewer background (default: 0.0625 = 1/16).",
    )
    p.add_argument(
        "--scene",
        type=int,
        default=0,
        help="CZI scene index for multi-scene files (default: 0).",
    )
    p.add_argument(
        "--skip-viewer",
        action="store_true",
        help="Skip HTML viewer generation.",
    )
    p.add_argument(
        "--max-contours",
        type=int,
        default=50_000,
        help="Maximum contours to include in viewer (default: 50000).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file path (default: <output-dir>/vessel_lumens_threshold.html).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for threshold-based lumen detection."""
    args = parse_args(argv)
    setup_logging(level="INFO")
    t0 = time.time()

    # Validate inputs
    if not args.zarr_path.exists():
        logger.error("Zarr path not found: %s", args.zarr_path)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse scales (sorted descending = coarsest first)
    scales = sorted([int(s.strip()) for s in args.scales.split(",")], reverse=True)
    if not scales:
        logger.error("No valid scales provided")
        sys.exit(1)

    # Discovery scales: which scales run lumen discovery (vs refinement-only)
    if args.discovery_scales:
        discovery_scales = set(int(s.strip()) for s in args.discovery_scales.split(","))
    else:
        discovery_scales = {scales[0]}  # default: coarsest only
    logger.info("  Discovery scales: %s", sorted(discovery_scales, reverse=True))

    logger.info("=" * 70)
    logger.info("THRESHOLD-BASED LUMEN DETECTION")
    logger.info("=" * 70)
    logger.info("  Zarr: %s", args.zarr_path)
    logger.info("  Scales: %s", scales)
    logger.info("  Otsu multiplier: %.2f", args.otsu_multiplier)
    logger.info("  Blur sigma: %.1f um", args.blur_sigma_um)
    logger.info("  Min area: %.0f um^2", args.min_area_um2)
    logger.info("  Tile size: %d px", args.tile_size)
    logger.info("  Output: %s", args.output_dir)

    # Open zarr
    import zarr

    zarr_root = zarr.open(str(args.zarr_path), mode="r")

    # Read base pixel size from zarr attributes
    base_pixel_size = None
    # Try OME-Zarr multiscales metadata
    multiscales = zarr_root.attrs.get("multiscales", [])
    if multiscales:
        datasets = multiscales[0].get("datasets", [])
        if datasets:
            transforms = datasets[0].get("coordinateTransformations", [])
            for t in transforms:
                if t.get("type") == "scale":
                    # OME-Zarr scale is in physical units; take Y dimension
                    scale_vals = t["scale"]
                    # Typically (C, Y, X) or (Y, X) — take the second-to-last
                    base_pixel_size = float(scale_vals[-1])
                    break
    if base_pixel_size is None:
        # Fallback: try top-level attrs
        base_pixel_size = zarr_root.attrs.get("pixel_size_um")
    if base_pixel_size is None:
        logger.error(
            "Cannot determine pixel size from zarr metadata. "
            "Ensure OME-Zarr has 'multiscales' with coordinateTransformations."
        )
        sys.exit(1)

    logger.info("  Base pixel size: %.4f um", base_pixel_size)

    # Resolve scales to zarr levels
    scale_infos = resolve_zarr_scales(zarr_root, scales)
    if not scale_infos:
        logger.error("No valid scales resolved")
        sys.exit(1)

    # Sort by scale descending (coarsest first)
    scale_infos.sort(key=lambda x: x[0], reverse=True)
    scale_infos_by_scale = {si[0]: si for si in scale_infos}

    # -----------------------------------------------------------------------
    # Step 1: Discover lumens at EVERY scale (coarsest first)
    # At each scale, mask out already-found lumens from previous scales.
    # -----------------------------------------------------------------------
    use_local = not args.no_local_threshold
    channel_indices = [int(c.strip()) for c in args.channels.split(",")] if args.channels else None
    threshold_kwargs = {
        "local_threshold": use_local,
        "block_size_um": args.block_size_um,
        "threshold_fraction": args.threshold_fraction,
        "fill_expansion": args.fill_expansion,
        "channel_indices": channel_indices,
        "output_complement": args.output_complement,
    }

    checkpoint_params = {
        "otsu_multiplier": args.otsu_multiplier,
        "blur_sigma_um": args.blur_sigma_um,
        "min_area_um2": args.min_area_um2,
        "tile_size": args.tile_size,
        "scales": args.scales,
        "local_threshold": use_local,
        "block_size_um": args.block_size_um,
        "threshold_fraction": args.threshold_fraction,
    }

    all_lumens: list[dict] = []
    last_completed_scale = 0

    if args.resume:
        all_lumens, last_completed_scale = load_discovery_checkpoint(
            args.output_dir,
            scales,
            checkpoint_params,
        )
        if last_completed_scale > 0:
            logger.info(
                "RESUMED from checkpoint: %d lumens through %dx scale",
                len(all_lumens),
                last_completed_scale,
            )
        else:
            logger.info("No valid checkpoint found, starting fresh")
    overlap = min(750, args.tile_size // 4)
    # Threshold for tiled vs whole-image: scales where the image fits in ~8 GB
    # Level 4 is ~6.5K x 14.6K × 4ch × 2B = 760 MB → whole-image ok
    # Level 3 is ~13K x 29K × 4ch × 2B = 3 GB → borderline, use tiling
    # 16x = 6.5K x 14.6K = 95M px → 4ch × 2B = 760 MB, fits in 64GB whole-image.
    # Threshold for whole-image vs tiled processing. Depends on available RAM.
    # 4ch × float32 = 16 bytes/px → 500M px = 8 GB, 2B px = 32 GB.
    # On 370GB nodes, 2B is safe. On smaller nodes, use 500M.
    tile_threshold_pixels = 2_000_000_000

    # Tissue mask: built from the coarsest-scale signal to skip background tiles.
    # A boolean array at the coarsest scale — True = tissue present.
    tissue_mask: np.ndarray | None = None
    tissue_mask_scale: int = 1

    coarsest_scale = scale_infos[0][0]  # sorted descending

    for si in scale_infos:
        scale, _, level_key, extra_ds = si

        # Skip scales not designated for discovery
        if scale not in discovery_scales:
            continue

        # Skip scales already completed (resume mode)
        # Scales are sorted descending (64, 16, 8, 4, 2).
        # last_completed_scale = 16 means 64x and 16x are done → skip scale >= 16.
        if last_completed_scale > 0 and scale >= last_completed_scale:
            logger.info("Skipping %dx (already completed in checkpoint)", scale)
            # Still need tissue mask even when skipping — rebuild from coarsest
            if tissue_mask is None:
                coarsest_si = scale_infos[0]
                cs, _, ck, cds = coarsest_si
                cl = zarr_root[ck]
                cs_pixel = base_pixel_size * cs
                chs = _read_all_channels(cl, cds)
                _, tm_signal, _ = _channel_sum_and_threshold(
                    chs,
                    cs_pixel,
                    args.blur_sigma_um,
                    args.otsu_multiplier,
                    **threshold_kwargs,
                )
                tissue_mask = tm_signal > args.tissue_threshold
                tissue_mask_scale = cs
                logger.info(
                    "  Rebuilt tissue mask at %dx: %.1f%% tissue (%s)",
                    cs,
                    100.0 * tissue_mask.sum() / max(1, tissue_mask.size),
                    tissue_mask.shape,
                )
            continue

        level_data = zarr_root[level_key]
        eff_shape = get_effective_shape(level_data, extra_ds)
        n_pixels = eff_shape[1] * eff_shape[2]

        use_tiling = n_pixels > tile_threshold_pixels

        # Per-scale block size: keep the same pixel-space sigma across scales.
        # block_size_um is the sigma at the coarsest scale. At finer scales,
        # scale down proportionally so the kernel covers the same pixel area.
        effective_block_um = args.block_size_um * (scale / coarsest_scale)
        scale_threshold_kwargs = {**threshold_kwargs, "block_size_um": effective_block_um}

        if use_tiling:
            # Tiled discovery (8x, 4x, 2x) with tissue mask for fast tile skipping
            new_lumens = discover_fine_lumens(
                zarr_root,
                si,
                base_pixel_size,
                existing_lumens=all_lumens,
                blur_sigma_um=args.blur_sigma_um,
                otsu_multiplier=args.otsu_multiplier,
                min_area_um2=args.min_area_um2,
                max_area_um2=None,  # no upper bound — discover all sizes
                tile_size=args.tile_size,
                overlap=overlap,
                tissue_mask=tissue_mask,
                tissue_mask_scale=tissue_mask_scale,
                save_debug=args.save_debug,
                output_dir=args.output_dir,
                threshold_kwargs=scale_threshold_kwargs,
            )
        else:
            # Whole-image discovery (64x, 16x)
            new_lumens, _, coarse_signal = discover_coarse_lumens(
                zarr_root,
                si,
                base_pixel_size,
                args.blur_sigma_um,
                args.otsu_multiplier,
                args.min_area_um2,
                existing_lumens=all_lumens if all_lumens else None,
                save_debug=args.save_debug,
                output_dir=args.output_dir,
                threshold_kwargs=scale_threshold_kwargs,
            )
            # Build tissue mask from the first (coarsest) whole-image signal
            if tissue_mask is None and coarse_signal is not None:
                tissue_mask = coarse_signal > args.tissue_threshold
                tissue_mask_scale = scale
                tissue_pct = 100.0 * tissue_mask.sum() / max(1, tissue_mask.size)
                logger.info(
                    "  Built tissue mask at %dx: %.1f%% tissue (%s)",
                    scale,
                    tissue_pct,
                    tissue_mask.shape,
                )
                # Compute global p10 floor for refinement — pixels below this are
                # definitely dark (lumen interior), even if the local threshold
                # misses them (e.g., center of large heart chambers).
                nonzero_signal = coarse_signal[coarse_signal > 0]
                if len(nonzero_signal) > 100:
                    global_p10 = float(np.percentile(nonzero_signal, 10))
                    threshold_kwargs["global_dark_floor"] = global_p10
                    logger.info("  Global p10 floor: %.4f", global_p10)

        all_lumens.extend(new_lumens)

        # Save checkpoint after each discovery scale
        save_discovery_checkpoint(all_lumens, scale, args.output_dir, checkpoint_params)

        logger.info(
            "After %dx discovery: +%d new lumens, %d total",
            scale,
            len(new_lumens),
            len(all_lumens),
        )

    if not all_lumens:
        logger.warning("No lumens found at any scale. Exiting.")
        out_path = args.output_dir / "vessel_lumens_threshold.json"
        atomic_json_dump([], str(out_path))
        logger.info("Wrote empty output: %s", out_path)
        logger.info("Total time: %.1fs", time.time() - t0)
        return

    # -----------------------------------------------------------------------
    # Step 2: Refine each lumen to the finest available scale (2x)
    # Only refine lumens discovered at coarser scales (discovery_scale > finest)
    # -----------------------------------------------------------------------
    finest_scale_info = scale_infos[-1]  # sorted descending, last = finest
    finest_scale = finest_scale_info[0]

    # Pre-filter using marker cells if provided (huge speedup — skip refining FPs)
    if args.marker_cells_json:
        from scipy.spatial import cKDTree

        marker_names = [m.strip() for m in args.marker_classes.split(",") if m.strip()]
        marker_keys = [f"{m}_class" for m in marker_names]
        logger.info(
            "Loading marker cells for pre-filter: %s (classes: %s)",
            args.marker_cells_json,
            ", ".join(marker_keys),
        )
        marker_cells = fast_json_load(str(args.marker_cells_json))
        marker_pos = []
        for c in marker_cells:
            feats = c.get("features", {})
            if any(feats.get(k) == "positive" for k in marker_keys):
                pos = c.get("global_center_um")
                if pos:
                    marker_pos.append(pos)
        marker_pos = np.array(marker_pos) if marker_pos else np.empty((0, 2))
        marker_tree = cKDTree(marker_pos) if len(marker_pos) > 0 else None
        logger.info("  %d marker+ cells loaded", len(marker_pos))

        n_before = len(all_lumens)
        kept = []
        for l in all_lumens:
            if marker_tree is None:
                break
            # Coarse centroid in um
            rs = l["refined_scale"]
            ps = base_pixel_size * rs
            cx = (l["bbox_x"] + l["bbox_w"] / 2) * ps
            cy = (l["bbox_y"] + l["bbox_h"] / 2) * ps
            equiv_r = np.sqrt(l["area_um2"] / np.pi)
            nearby = marker_tree.query_ball_point([cx, cy], equiv_r + 30.0)
            if len(nearby) >= args.min_marker_cells:
                l["n_marker_wall"] = len(nearby)
                kept.append(l)
        all_lumens = kept
        logger.info(
            "  Marker pre-filter (>=%d cells): %d → %d lumens",
            args.min_marker_cells,
            n_before,
            len(all_lumens),
        )
        del marker_cells, marker_pos, marker_tree
        gc.collect()

    lumens_to_refine = [l for l in all_lumens if l["discovery_scale"] > finest_scale]
    lumens_already_fine = [l for l in all_lumens if l["discovery_scale"] <= finest_scale]

    if lumens_to_refine:
        logger.info(
            "Step 2: Refining %d lumens to %dx (already at %dx: %d)",
            len(lumens_to_refine),
            finest_scale,
            finest_scale,
            len(lumens_already_fine),
        )
        # Refine progressively through each finer scale.
        # Group by refined_scale to avoid mixing lumens at different scales
        # (refine_lumens_at_scale assumes all lumens share the same src_scale).
        for si in scale_infos[1:]:  # skip coarsest, go finer
            target_scale = si[0]
            # Group lumens by their current refined_scale
            by_src: dict[int, list[dict]] = {}
            for l in lumens_to_refine:
                rs = l["refined_scale"]
                if rs > target_scale:
                    by_src.setdefault(rs, []).append(l)
            if not by_src:
                continue
            for src_scale, batch in sorted(by_src.items(), reverse=True):
                # On resume, masks may be None — re-derive from zarr
                n_missing = sum(1 for l in batch if l.get("mask_at_scale") is None)
                if n_missing:
                    rederive_masks_batch(
                        batch,
                        zarr_root,
                        scale_infos_by_scale,
                        base_pixel_size,
                        args.blur_sigma_um,
                        args.otsu_multiplier,
                        label=f"refinement {src_scale}x→{target_scale}x",
                    )
                # refine_lumens_at_scale mutates lumens in-place (updates
                # bbox, mask_at_scale, refined_scale) so no reassignment needed.
                refine_lumens_at_scale(
                    batch,
                    zarr_root,
                    src_scale=src_scale,
                    dst_scale_info=si,
                    base_pixel_size=base_pixel_size,
                    blur_sigma_um=args.blur_sigma_um,
                    otsu_multiplier=args.otsu_multiplier,
                    save_debug=args.save_debug,
                    output_dir=args.output_dir,
                    threshold_kwargs=threshold_kwargs,
                )

        all_lumens = lumens_to_refine + lumens_already_fine
        logger.info("After refinement: %d total lumens", len(all_lumens))

    logger.info("Total lumens before contour extraction: %d", len(all_lumens))

    # On resume, masks may be None — re-derive from zarr
    n_missing = sum(1 for l in all_lumens if l.get("mask_at_scale") is None)
    if n_missing:
        rederive_masks_batch(
            all_lumens,
            zarr_root,
            scale_infos_by_scale,
            base_pixel_size,
            args.blur_sigma_um,
            args.otsu_multiplier,
            label="contour extraction",
        )

    # -----------------------------------------------------------------------
    # Step 4: Extract contours + metadata
    # -----------------------------------------------------------------------
    records = extract_contours(all_lumens, base_pixel_size)
    logger.info("Extracted %d valid contours from %d lumens", len(records), len(all_lumens))

    # -----------------------------------------------------------------------
    # Step 4b: Inline feature extraction (morph + channel stats)
    # -----------------------------------------------------------------------
    # Lumens still have mask_at_scale in memory; zarr is open. Extract features
    # now to avoid a separate pass over the zarr.
    logger.info("Extracting features for %d lumens...", len(records))
    from xldvp_seg.detection.strategies.mixins import MultiChannelFeatureMixin

    _channel_mixin = MultiChannelFeatureMixin()
    n_features_ok = 0
    n_skipped_large = 0
    for ri, rec in enumerate(records):
        # Match record to lumen by position
        rs = rec["refined_scale"]
        pixel_size_at_scale = base_pixel_size * rs

        # Rasterize contour to get mask (fast, avoids lookup issues)
        pts = np.asarray(rec["contour_global_um"], dtype=np.float64)
        pts_px = pts / pixel_size_at_scale
        x_min = max(0, int(np.floor(pts_px[:, 0].min())))
        y_min = max(0, int(np.floor(pts_px[:, 1].min())))
        x_max = int(np.ceil(pts_px[:, 0].max()))
        y_max = int(np.ceil(pts_px[:, 1].max()))
        bw = x_max - x_min + 1
        bh = y_max - y_min + 1
        if bw < 2 or bh < 2:
            rec["features"] = {}
            continue

        # Skip very large lumens (background FPs)
        if rec["area_um2"] > 1_000_000:
            rec["features"] = {}
            n_skipped_large += 1
            continue

        local_pts = (pts_px - [x_min, y_min]).astype(np.int32)
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [local_pts], 1)
        mask = mask.astype(bool)

        if not mask.any():
            rec["features"] = {}
            continue

        # Read zarr crop
        si = scale_infos_by_scale.get(rs)
        if si is None:
            rec["features"] = {}
            continue
        _, _, lk, eds = si
        ld = zarr_root[lk]
        channels = read_all_channels_crop(ld, eds, y_min, x_min, bh, bw)

        # Morph features from mask
        features = {}
        mask_u8 = mask.astype(np.uint8)
        contours_cv, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_cv:
            contour = max(contours_cv, key=cv2.contourArea)
            area_px = int(mask.sum())
            perim_px = cv2.arcLength(contour, True)
            circ = min((4 * np.pi * area_px) / max(perim_px**2, 1e-10), 1.0)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area_px / max(hull_area, 1e-10)
            features["morph_circularity"] = round(circ, 4)
            features["morph_solidity"] = round(solidity, 4)
            if len(contour) >= 5:
                try:
                    (_, (ma, MA), _) = cv2.fitEllipse(contour)
                    features["morph_eccentricity"] = round(
                        np.sqrt(1 - (ma / max(MA, 1e-10)) ** 2) if ma < MA else 0.0, 4
                    )
                    features["morph_aspect_ratio"] = round(MA / max(ma, 1e-10), 4)
                except cv2.error:
                    pass

        # Channel stats
        ch_dict = {f"ch{ci}": ch for ci, ch in enumerate(channels)}
        ch_features = _channel_mixin.extract_multichannel_features(
            mask, ch_dict, _include_zeros=True
        )
        features.update(ch_features)

        rec["features"] = features
        n_features_ok += 1

        if (ri + 1) % 5000 == 0:
            logger.info("  Features: %d/%d lumens (%d ok)", ri + 1, len(records), n_features_ok)

    logger.info(
        "Feature extraction: %d/%d lumens with features (%d skipped large)",
        n_features_ok,
        len(records),
        n_skipped_large,
    )

    # Sort by area descending
    records.sort(key=lambda r: r["area_um2"], reverse=True)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("THRESHOLD LUMEN DETECTION SUMMARY")
    logger.info("=" * 70)
    logger.info("  Total lumens: %d", len(records))
    if records:
        areas = [r["area_um2"] for r in records]
        logger.info(
            "  Area: median=%.0f, min=%.0f, max=%.0f um^2",
            np.median(areas),
            min(areas),
            max(areas),
        )
        diameters = [r["equiv_diameter_um"] for r in records]
        logger.info(
            "  Diameter: median=%.1f, min=%.1f, max=%.1f um",
            np.median(diameters),
            min(diameters),
            max(diameters),
        )
        contrasts = [r["contrast_ratio"] for r in records]
        logger.info(
            "  Contrast: median=%.2f, min=%.2f, max=%.2f",
            np.median(contrasts),
            min(contrasts),
            max(contrasts),
        )

        # Darkness tier distribution
        tier_counts: dict[str, int] = {}
        for r in records:
            tier = r["darkness_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        for tier in ["very_dark", "dark", "moderate", "light"]:
            if tier in tier_counts:
                logger.info(
                    "  %s: %d (%.1f%%)",
                    tier,
                    tier_counts[tier],
                    100.0 * tier_counts[tier] / len(records),
                )

        # Scale distribution
        scale_counts: dict[int, int] = {}
        for r in records:
            s = r["discovery_scale"]
            scale_counts[s] = scale_counts.get(s, 0) + 1
        for s in sorted(scale_counts.keys(), reverse=True):
            logger.info("  discovered@%dx: %d", s, scale_counts[s])

    # Write output
    out_path = args.output_dir / "vessel_lumens_threshold.json"
    atomic_json_dump(records, str(out_path))
    logger.info("Wrote %d lumens to %s", len(records), out_path)

    # -----------------------------------------------------------------------
    # Step 5: Viewer
    # -----------------------------------------------------------------------
    if not args.skip_viewer:
        html_path = args.output or (args.output_dir / "vessel_lumens_threshold.html")
        _generate_viewer(
            output_json=out_path,
            output_html=html_path,
            group_field="darkness_tier",
            czi_path=args.czi_path,
            display_channels=args.display_channels,
            channel_names=args.channel_names,
            scale_factor=args.scale_factor,
            scene=args.scene,
            max_contours=args.max_contours,
        )

    elapsed = time.time() - t0
    logger.info("Total time: %.1fs (%.1f min)", elapsed, elapsed / 60.0)


if __name__ == "__main__":
    main()
