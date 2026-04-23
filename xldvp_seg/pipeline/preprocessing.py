"""Slide-wide preprocessing: photobleach correction, flat-field, Reinhard normalization.

Applies slide-level corrections to all loaded channels before tile processing.
All corrections modify the channel data arrays in-place.
"""

import gc
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from xldvp_seg.exceptions import DataLoadError
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Flat-field cache filename + lock suffix. Placed inside slide_output_dir so it
# survives across shards and runs with --resume, and invalidates naturally when
# a new run_timestamped dir is created for a different slide.
FLAT_FIELD_CACHE_FILENAME = "flat_field_profile.npz"
FLAT_FIELD_LOCK_FILENAME = "flat_field_profile.computing"


def apply_slide_preprocessing(args, all_channel_data, loader, *, slide_output_dir=None):
    """Apply slide-wide preprocessing: photobleach correction, flat-field, Reinhard.

    Modifies all_channel_data and loader in-place.

    Args:
        args: Parsed CLI args (reads photobleaching_correction, normalize_features,
              norm_params_file, channel)
        all_channel_data: dict of channel_idx -> 2D array (modified in place)
        loader: CZI loader (channel_data updated in place)
        slide_output_dir: Where to read/write the flat-field profile cache. Pass
            when available so multi-node shard runs and ``--resume`` reuses the
            ~1-2h slide-wide illumination estimate instead of recomputing it per
            shard. Pass ``None`` to disable caching (single-node ephemeral runs).
    """
    # RGB brightfield CZIs: skip all preprocessing.
    # Photobleach/flat-field/Reinhard are designed for 2D uint16 fluorescence and
    # either crash or produce garbage on 3D uint8 RGB brightfield data.
    if any(arr.ndim == 3 and arr.dtype == np.uint8 for arr in all_channel_data.values()):
        logger.info("RGB brightfield detected — skipping photobleach/flat-field/Reinhard")
        return

    # EXPERIMENTAL: Photobleaching correction — results may be unreliable.
    # This feature needs rework before production use.
    if getattr(args, "photobleaching_correction", False):
        logger.warning("Photobleach correction is EXPERIMENTAL and may produce unreliable results.")
        _apply_photobleach_correction(args, all_channel_data, loader)

    # Apply flat-field illumination correction (smooth out regional intensity gradients)
    if getattr(args, "normalize_features", True):
        _apply_flat_field_correction(
            args, all_channel_data, loader, slide_output_dir=slide_output_dir
        )

    # Apply Reinhard normalization if params file provided (whole-slide, before tiling)
    if getattr(args, "norm_params_file", None):
        _apply_reinhard_normalization(args, all_channel_data, loader)


def _apply_photobleach_correction(args, all_channel_data, loader):
    """Apply slide-wide photobleaching correction to fix horizontal/vertical banding."""
    from xldvp_seg.preprocessing.illumination import (
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


def _apply_flat_field_correction(args, all_channel_data, loader, *, slide_output_dir=None):
    """Apply flat-field illumination correction to smooth regional intensity gradients.

    When ``slide_output_dir`` is given, a cached ``flat_field_profile.npz`` in
    that directory is used if its metadata (CZI path / mtime / size / channel
    list / slide shape) matches the current run. Otherwise the profile is
    computed fresh and written back to the cache for subsequent shards or
    ``--resume`` runs to reuse.
    """
    logger.info(f"\n{'='*70}")
    logger.info("FLAT-FIELD ILLUMINATION CORRECTION")
    logger.info(f"{'='*70}")

    cache_meta = _flat_field_cache_meta(args, all_channel_data)
    cache_path = None
    lock_path = None
    if slide_output_dir is not None:
        out_dir = Path(slide_output_dir)
        cache_path = out_dir / FLAT_FIELD_CACHE_FILENAME
        lock_path = out_dir / FLAT_FIELD_LOCK_FILENAME

    illumination_profile = (
        _load_flat_field_cache(cache_path, cache_meta) if cache_path is not None else None
    )
    if illumination_profile is not None:
        logger.info(
            "Reusing cached flat-field profile (%d channels, block_size=%d).",
            len(illumination_profile.grids),
            illumination_profile.block_size,
        )
    else:
        illumination_profile = _acquire_or_wait_for_profile(
            all_channel_data, cache_path, lock_path, cache_meta
        )

    for ch in all_channel_data:
        illumination_profile.correct_channel_inplace(all_channel_data[ch], ch)
        # Sync corrected data back to loader for all channels
        if hasattr(loader, "set_channel_data"):
            loader.set_channel_data(ch, all_channel_data[ch])
        elif ch == args.channel:
            loader.channel_data = all_channel_data[ch]

    gc.collect()
    logger.info("Flat-field correction complete.")


def _acquire_or_wait_for_profile(all_channel_data, cache_path, lock_path, cache_meta):
    """Compute the profile locally, or wait for another shard's write.

    Loops until either (a) we acquire the compute lock and produce the profile
    ourselves (releasing the lock in a ``finally`` so an exception mid-compute
    never leaks the lock), or (b) another shard writes a valid cache while we
    wait. If a wait returns without a cache hit (e.g. the other shard crashed
    and its lock was force-removed), we retry the acquire instead of computing
    without a lock, preventing concurrent redundant recomputes.
    """
    from xldvp_seg.preprocessing.flat_field import estimate_illumination_profile

    # No cache configured → compute inline, nothing to coordinate.
    if cache_path is None or lock_path is None:
        logger.info("Estimating slide-level illumination profile...")
        return estimate_illumination_profile(all_channel_data)

    while True:
        if _try_acquire_compute_lock(lock_path):
            try:
                logger.info("Estimating slide-level illumination profile...")
                profile = estimate_illumination_profile(all_channel_data)
                try:
                    profile.save(cache_path, metadata=cache_meta)
                    logger.info("Saved flat-field profile cache: %s", cache_path.name)
                except OSError as exc:
                    logger.warning("Could not write flat-field cache %s: %s", cache_path, exc)
                return profile
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

        # Lock held by another process — wait for the cache to land.
        logger.info(
            "Flat-field profile is being computed by another process (lock=%s). Waiting...",
            lock_path.name,
        )
        _wait_for_compute(cache_path, lock_path)
        profile = _load_flat_field_cache(cache_path, cache_meta)
        if profile is not None:
            logger.info(
                "Loaded flat-field profile written by another shard (%d channels).",
                len(profile.grids),
            )
            return profile
        # Wait exited without a cache — other shard likely crashed and the
        # stale lock was removed. Loop back and retry the acquire so only one
        # of the waiters recomputes.


def _flat_field_cache_meta(args, all_channel_data: dict) -> dict:
    """Build the cache-key metadata dict for the current preprocessing call.

    Anything that changes what the flat-field estimator *sees* must be in here,
    or a stale cache from a previous run with different settings could be
    loaded. That includes CZI identity (path/mtime/size), channel selection,
    slide shape, and whether photobleach correction runs before flat-field
    (photobleach mutates the SHM data before flat-field estimates its profile,
    so toggling it changes the correct profile).
    """
    czi_path_str = str(getattr(args, "czi_path", ""))
    czi_mtime: float = 0.0
    czi_size: int = 0
    try:
        if czi_path_str:
            st = os.stat(czi_path_str)
            czi_mtime = float(st.st_mtime)
            czi_size = int(st.st_size)
    except OSError:
        pass
    any_arr = next(iter(all_channel_data.values()))
    return {
        "czi_path": czi_path_str,
        "czi_mtime": czi_mtime,
        "czi_size": czi_size,
        "channels": sorted(int(c) for c in all_channel_data),
        "slide_shape": [int(s) for s in any_arr.shape[:2]],
        "photobleaching_correction": bool(getattr(args, "photobleaching_correction", False)),
    }


def _load_flat_field_cache(cache_path: Path | None, expected_meta: dict):
    """Return a valid :class:`IlluminationProfile` from cache, or ``None``.

    Any mismatch between cached metadata and the current run (czi identity,
    shape, channel list) is treated as a miss so a stale slide's profile can
    never contaminate a different run.
    """
    from xldvp_seg.preprocessing.flat_field import IlluminationProfile

    if cache_path is None or not cache_path.exists():
        return None
    try:
        profile, cached_meta = IlluminationProfile.load(cache_path)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, EOFError) as exc:
        # Covers: algorithm_version mismatch (ValueError), missing npz key
        # (KeyError), truncated/corrupted file (BadZipFile/EOFError), FS errors
        # (OSError). Any of these means we should recompute rather than crash.
        logger.info("Flat-field cache at %s unreadable (%s) — recomputing.", cache_path.name, exc)
        return None
    for key in (
        "czi_path",
        "czi_size",
        "channels",
        "slide_shape",
        "photobleaching_correction",
    ):
        if cached_meta.get(key) != expected_meta.get(key):
            logger.info(
                "Flat-field cache stale: %s differs " "(cached=%r, current=%r). Recomputing.",
                key,
                cached_meta.get(key),
                expected_meta.get(key),
            )
            return None
    if abs(cached_meta.get("czi_mtime", 0.0) - expected_meta.get("czi_mtime", 0.0)) > 1.0:
        logger.info("Flat-field cache stale: czi_mtime differs. Recomputing.")
        return None
    logger.info("Loaded flat-field profile from cache: %s", cache_path.name)
    return profile


def _try_acquire_compute_lock(lock_path: Path | None) -> bool:
    """Atomic advisory lock via ``O_CREAT | O_EXCL``.

    Returns ``True`` if the lock was acquired (caller must compute + write the
    cache + release the lock). Returns ``False`` if another process is already
    computing — caller should wait.
    """
    if lock_path is None:
        return True
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError as exc:
        logger.warning(
            "Could not create flat-field lock %s: %s — proceeding without lock", lock_path, exc
        )
        return True  # degrade to no-lock rather than blocking
    try:
        os.write(fd, f"pid={os.getpid()}\nhost={os.uname().nodename}\n".encode())
    finally:
        os.close(fd)
    return True


def _wait_for_compute(
    cache_path: Path, lock_path: Path, *, poll_sec: int = 30, stale_sec: int = 3 * 3600
) -> None:
    """Poll for the cache file to land, handling stale locks.

    Returns once either (a) the cache file exists, (b) the lock disappeared
    (caller should retry), or (c) the lock mtime exceeds ``stale_sec`` (lock is
    forcibly removed and caller retries).
    """
    start = time.time()
    while True:
        if cache_path.exists():
            return
        if not lock_path.exists():
            return
        try:
            lock_age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if lock_age > stale_sec:
            logger.warning(
                "Flat-field compute lock is %.1f h old — removing and retrying.", lock_age / 3600
            )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return
        if time.time() - start > stale_sec:
            logger.warning("Giving up waiting for flat-field cache — will recompute locally.")
            return
        time.sleep(poll_sec)


def _apply_reinhard_normalization(args, all_channel_data, loader):
    """Apply cross-slide Reinhard normalization (LAB space, median/MAD)."""
    from xldvp_seg.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN

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
        raise DataLoadError(
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
        from xldvp_seg.utils.detection_utils import safe_to_uint8

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
