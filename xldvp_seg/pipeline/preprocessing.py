"""Slide-wide preprocessing: photobleach correction, flat-field, Reinhard normalization.

Applies slide-level corrections to all loaded channels before tile processing.
All corrections modify the channel data arrays in-place.
"""

import gc
import json

import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def apply_slide_preprocessing(args, all_channel_data, loader):
    """Apply slide-wide preprocessing: photobleach correction, flat-field, Reinhard.

    Modifies all_channel_data and loader in-place.

    Args:
        args: Parsed CLI args (reads photobleaching_correction, normalize_features,
              norm_params_file, channel)
        all_channel_data: dict of channel_idx -> 2D array (modified in place)
        loader: CZI loader (channel_data updated in place)
    """
    # EXPERIMENTAL: Photobleaching correction — results may be unreliable.
    # This feature needs rework before production use.
    if getattr(args, "photobleaching_correction", False):
        logger.warning("Photobleach correction is EXPERIMENTAL and may produce unreliable results.")
        _apply_photobleach_correction(args, all_channel_data, loader)

    # Apply flat-field illumination correction (smooth out regional intensity gradients)
    if getattr(args, "normalize_features", True):
        _apply_flat_field_correction(args, all_channel_data, loader)

    # Apply Reinhard normalization if params file provided (whole-slide, before tiling)
    if getattr(args, "norm_params_file", None):
        _apply_reinhard_normalization(args, all_channel_data, loader)


def _apply_photobleach_correction(args, all_channel_data, loader):
    """Apply slide-wide photobleaching correction to fix horizontal/vertical banding."""
    from segmentation.preprocessing.illumination import (
        estimate_band_severity,
        normalize_rows_columns,
    )

    logger.info("Applying slide-wide photobleaching correction...")

    for ch, ch_data in all_channel_data.items():
        original_dtype = ch_data.dtype

        # Report severity before
        severity_before = estimate_band_severity(ch_data)
        logger.info(
            f"  Channel {ch} before: row_cv={severity_before['row_cv']:.1f}%, "
            f"col_cv={severity_before['col_cv']:.1f}% ({severity_before['severity']})"
        )

        # Apply row/column normalization to fix banding
        # Note: uses float32 internally, may need ~2x memory temporarily
        corrected = normalize_rows_columns(ch_data)

        # Convert back to original dtype (in-place clip to avoid extra copy)
        if original_dtype == np.uint16:
            np.clip(corrected, 0, 65535, out=corrected)
            corrected = corrected.astype(np.uint16)
        elif original_dtype == np.uint8:
            np.clip(corrected, 0, 255, out=corrected)
            corrected = corrected.astype(np.uint8)
        else:
            corrected = corrected.astype(original_dtype)

        # Write corrected data back into the existing array (may be an SHM view).
        # Using np.copyto preserves the SHM backing — dict assignment would replace
        # the view with a new array, disconnecting from shared memory.
        np.copyto(all_channel_data[ch], corrected)
        if hasattr(loader, "set_channel_data"):
            loader.set_channel_data(ch, all_channel_data[ch])

        # Report severity after
        severity_after = estimate_band_severity(corrected)
        logger.info(
            f"  Channel {ch} after:  row_cv={severity_after['row_cv']:.1f}%, "
            f"col_cv={severity_after['col_cv']:.1f}% ({severity_after['severity']})"
        )

        # Drop the local reference so GC can free any intermediate float64 arrays
        del corrected

        # Force garbage collection after each channel to free float64 intermediate
        gc.collect()

    logger.info("Photobleaching correction complete.")


def _apply_flat_field_correction(args, all_channel_data, loader):
    """Apply flat-field illumination correction to smooth regional intensity gradients."""
    from segmentation.preprocessing.flat_field import estimate_illumination_profile

    logger.info(f"\n{'='*70}")
    logger.info("FLAT-FIELD ILLUMINATION CORRECTION")
    logger.info(f"{'='*70}")
    logger.info("Estimating slide-level illumination profile...")

    illumination_profile = estimate_illumination_profile(all_channel_data)

    for ch in all_channel_data:
        illumination_profile.correct_channel_inplace(all_channel_data[ch], ch)
        # Sync corrected data back to loader for all channels
        if hasattr(loader, "set_channel_data"):
            loader.set_channel_data(ch, all_channel_data[ch])
        elif ch == args.channel:
            loader.channel_data = all_channel_data[ch]

    gc.collect()
    logger.info("Flat-field correction complete.")


def _apply_reinhard_normalization(args, all_channel_data, loader):
    """Apply cross-slide Reinhard normalization (LAB space, median/MAD)."""
    from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN

    logger.info(f"\n{'='*70}")
    logger.info("CROSS-SLIDE REINHARD NORMALIZATION (median/MAD)")
    logger.info(f"{'='*70}")
    logger.info(f"Loading params from: {args.norm_params_file}")

    with open(args.norm_params_file) as f:
        norm_params = json.load(f)

    # Validate required keys
    required_keys = {"L_median", "L_mad", "a_median", "a_mad", "b_median", "b_mad"}
    missing_keys = required_keys - set(norm_params.keys())
    if missing_keys:
        raise ValueError(
            f"Normalization params file missing required keys: {missing_keys}. "
            f"Required: {required_keys}"
        )

    logger.info(
        f"  Target: L_median={norm_params['L_median']:.2f}, L_mad={norm_params['L_mad']:.2f}"
    )
    logger.info(
        f"  Target: a_median={norm_params['a_median']:.2f}, a_mad={norm_params['a_mad']:.2f}"
    )
    logger.info(
        f"  Target: b_median={norm_params['b_median']:.2f}, b_mad={norm_params['b_mad']:.2f}"
    )
    if "n_slides" in norm_params:
        logger.info(
            f"  Computed from {norm_params['n_slides']} slides, {norm_params.get('n_total_pixels', '?')} pixels"
        )

    # Build RGB image for normalization
    # Loader produces individual 2D channels — primary_data is always 2D.
    primary_data = loader.channel_data
    if primary_data.ndim != 2:
        raise ValueError(f"Unexpected channel data shape for normalization: {primary_data.shape}")
    # Single channel: convert to uint8 FIRST, then stack 3x.
    # This avoids creating a 3-channel uint16 copy (3x memory waste).
    single_u8 = primary_data
    if single_u8.dtype == np.uint16:
        logger.info(
            f"  Converting single-channel uint16 -> uint8 before stacking ({single_u8.nbytes / 1e9:.1f} GB)"
        )
        single_u8 = (single_u8 >> 8).astype(np.uint8)
    elif single_u8.dtype != np.uint8:
        from segmentation.utils.detection_utils import safe_to_uint8

        single_u8 = safe_to_uint8(single_u8)
    rgb_for_norm = np.stack([single_u8] * 3, axis=-1)
    del single_u8

    logger.info(
        f"  RGB shape: {rgb_for_norm.shape}, dtype: {rgb_for_norm.dtype} ({rgb_for_norm.nbytes / 1e9:.1f} GB)"
    )
    logger.info(
        "  Applying Reinhard normalization (this normalizes tissue blocks, preserves background)..."
    )

    normalized_rgb = apply_reinhard_normalization_MEDIAN(rgb_for_norm, norm_params)
    del rgb_for_norm
    gc.collect()

    # Update channel data with normalized values.
    # Use np.copyto to write back into existing arrays (may be SHM views).
    # Single channel -- take first channel from normalized RGB
    normalized_single = normalized_rgb[:, :, 0]
    np.copyto(all_channel_data[args.channel], normalized_single)
    if hasattr(loader, "set_channel_data"):
        loader.set_channel_data(args.channel, all_channel_data[args.channel])
    del normalized_single

    del normalized_rgb
    gc.collect()

    logger.info("  Reinhard normalization complete.")
