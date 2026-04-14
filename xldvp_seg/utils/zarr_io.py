"""Shared OME-Zarr I/O helpers for multi-scale pyramid reading.

Provides scale resolution, tile generation, and single/multi-channel tile reading
with optional extra downsampling for scales beyond the zarr pyramid.

Used by:
  - scripts/detect_vessel_lumens_threshold.py
  - scripts/extract_lumen_features.py

TODO: scripts/segment_vessel_lumens.py (1,736 lines) has its own copies of similar
helpers. A future refactor could migrate it to use this module as well.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def get_effective_shape(
    level_data: Any,
    extra_ds: int,
) -> tuple[int, int, int]:
    """Get the effective (C, H, W) shape after optional extra downsampling.

    Args:
        level_data: Zarr array with shape (C, H, W).
        extra_ds: Extra downsample factor (1 = no downsampling).

    Returns:
        (C, H, W) tuple representing the effective shape.
    """
    c, h, w = level_data.shape
    if extra_ds > 1:
        return (c, h // extra_ds, w // extra_ds)
    return (c, h, w)


# ---------------------------------------------------------------------------
# Scale resolution
# ---------------------------------------------------------------------------


def resolve_zarr_scales(
    zarr_root: Any,
    scales: list[int],
) -> list[tuple[int, int, str, int]]:
    """Resolve requested scales against available zarr pyramid levels.

    For scales that exceed the zarr pyramid (e.g., 64x when zarr only has
    levels 0-4 = up to 16x), maps to the coarsest available level plus an
    extra downsample factor applied in-script.

    Args:
        zarr_root: Opened zarr group with numeric level keys ('0', '1', ...).
        scales: List of scale factors (must be powers of 2).

    Returns:
        List of (scale, zarr_level, level_key, extra_downsample) tuples.

    Raises:
        ValueError: If any scale is not a power of 2.
    """
    # Validate power-of-2
    for s in scales:
        if s < 1 or (s & (s - 1)) != 0:
            raise ValueError(
                f"Scale {s} is not a power of 2. Scales must be powers of 2 "
                f"(e.g., 2, 4, 8, 16, 32, 64)."
            )

    max_zarr_level = max(int(k) for k in zarr_root.keys() if k.isdigit())
    resolved = []

    for s in scales:
        level = int(np.log2(s))
        level_key = str(level)
        if level_key in zarr_root:
            resolved.append((s, level, level_key, 1))
        elif level > max_zarr_level:
            extra_ds = 2 ** (level - max_zarr_level)
            coarse_key = str(max_zarr_level)
            logger.info(
                "Scale %dx: zarr level %d not found, will read level %d + %dx downsample",
                s,
                level,
                max_zarr_level,
                extra_ds,
            )
            resolved.append((s, max_zarr_level, coarse_key, extra_ds))
        else:
            logger.warning("Zarr level %d (scale %dx) not found, skipping", level, s)

    return resolved


def resolve_zarr_level(zarr_root: Any, scale: int) -> tuple[Any, int]:
    """Resolve a single scale to a zarr level array + extra_ds factor.

    Unlike :func:`resolve_zarr_scales` (which resolves a list and returns
    metadata tuples), this returns the zarr array directly — convenient when
    you only need one scale at a time.

    Args:
        zarr_root: Opened zarr group with numeric level keys ('0', '1', ...).
        scale: Scale factor (must be a power of 2).

    Returns:
        (level_data, extra_ds) where *level_data* is the zarr array and
        *extra_ds* is the additional downsample factor (1 = none).

    Raises:
        ValueError: If no numeric keys exist in the zarr root.
    """
    target_level = int(np.log2(scale))
    available = sorted(int(k) for k in zarr_root.keys() if k.isdigit())
    if not available:
        raise ValueError("No numeric keys in zarr root")

    if target_level in available:
        return zarr_root[str(target_level)], 1

    # Fall back to coarsest available level <= target, else the overall coarsest
    best = (
        max(lv for lv in available if lv <= target_level)
        if any(lv <= target_level for lv in available)
        else available[-1]
    )
    extra_ds = scale // (2**best)
    return zarr_root[str(best)], max(1, extra_ds)


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------


def generate_tiles(
    level_shape: tuple[int, ...], tile_size: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """Generate (y_start, x_start, h, w) tiles covering a zarr level.

    Args:
        level_shape: Shape as (C, H, W).
        tile_size: Tile side length in pixels.
        overlap: Overlap between adjacent tiles in pixels.

    Returns:
        List of (y, x, h, w) tuples.
    """
    _, H, W = level_shape
    step = tile_size - overlap
    if step <= 0:
        step = tile_size
    tiles = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            h = min(tile_size, H - y)
            w = min(tile_size, W - x)
            # Skip tiny edge tiles that are mostly overlap
            if h > overlap and w > overlap:
                tiles.append((y, x, h, w))
    return tiles


# ---------------------------------------------------------------------------
# Single-channel tile reading
# ---------------------------------------------------------------------------


def read_zarr_tile(
    level_data: Any,
    channel: int,
    tile_y: int,
    tile_x: int,
    tile_h: int,
    tile_w: int,
    extra_ds: int = 1,
) -> np.ndarray:
    """Read a single-channel tile from zarr with optional extra downsampling.

    When extra_ds > 1, reads the corresponding larger region from the zarr
    level and downsamples with cv2.INTER_AREA (anti-aliased averaging).

    Args:
        level_data: Zarr array with shape (C, H, W).
        channel: Channel index to read.
        tile_y: Tile y-offset in the effective (downsampled) coordinate space.
        tile_x: Tile x-offset in the effective (downsampled) coordinate space.
        tile_h: Tile height in the effective coordinate space.
        tile_w: Tile width in the effective coordinate space.
        extra_ds: Extra downsample factor (1 = direct read, no downsampling).

    Returns:
        2D numpy array of shape (tile_h, tile_w), same dtype as zarr source
        (or resized via INTER_AREA if extra_ds > 1).
    """
    if extra_ds > 1:
        # Map effective coords back to zarr-level coords
        zy = tile_y * extra_ds
        zx = tile_x * extra_ds
        zh = min(tile_h * extra_ds, level_data.shape[1] - zy)
        zw = min(tile_w * extra_ds, level_data.shape[2] - zx)
        raw = np.array(level_data[channel, zy : zy + zh, zx : zx + zw])
        # Compute actual output size (may differ from tile_h/tile_w at edges
        # due to rounding in integer division)
        out_h = min(tile_h, int(np.ceil(zh / extra_ds)))
        out_w = min(tile_w, int(np.ceil(zw / extra_ds)))
        return cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        return np.array(level_data[channel, tile_y : tile_y + tile_h, tile_x : tile_x + tile_w])


# ---------------------------------------------------------------------------
# Multi-channel tile reading
# ---------------------------------------------------------------------------


def read_all_channels(
    level_data: Any,
    extra_ds: int,
    tile_y: int = 0,
    tile_x: int = 0,
    tile_h: int | None = None,
    tile_w: int | None = None,
) -> list[np.ndarray]:
    """Read all channels from a zarr level (optionally cropped to a tile).

    Reads each channel individually via :func:`read_zarr_tile`. For small
    crops where I/O latency dominates, prefer :func:`read_all_channels_crop`
    which issues a single multi-channel zarr read.

    Args:
        level_data: Zarr array with shape (C, H, W).
        extra_ds: Extra downsample factor.
        tile_y: Y offset in effective coordinates.
        tile_x: X offset in effective coordinates.
        tile_h: Height in effective coordinates (None = full height).
        tile_w: Width in effective coordinates (None = full width).

    Returns:
        List of 2D float32 arrays, one per channel.
    """
    eff_shape = get_effective_shape(level_data, extra_ds)
    n_channels = eff_shape[0]
    if tile_h is None:
        tile_h = eff_shape[1]
    if tile_w is None:
        tile_w = eff_shape[2]

    channels = []
    for ch in range(n_channels):
        arr = read_zarr_tile(level_data, ch, tile_y, tile_x, tile_h, tile_w, extra_ds)
        channels.append(arr.astype(np.float32))
    return channels


def read_all_channels_crop(
    level_data: Any,
    extra_ds: int,
    y: int,
    x: int,
    h: int,
    w: int,
) -> list[np.ndarray]:
    """Read all channels for a crop region with zero-padding at boundaries.

    Unlike :func:`read_all_channels`, this issues a single multi-channel zarr
    read (better I/O on network filesystems) and zero-pads the result when
    the crop extends beyond the image boundary, guaranteeing output arrays of
    exactly shape ``(h, w)``.

    Args:
        level_data: Zarr array with shape (C, H, W).
        extra_ds: Extra downsample factor (1 = direct read).
        y: Crop y-offset in effective (downsampled) coordinate space.
        x: Crop x-offset in effective (downsampled) coordinate space.
        h: Crop height in effective coordinates.
        w: Crop width in effective coordinates.

    Returns:
        List of 2D float32 arrays of shape (h, w), one per channel.
    """
    # Clamp negative coordinates to 0
    y = max(0, y)
    x = max(0, x)

    if extra_ds > 1:
        raw_y, raw_x = y * extra_ds, x * extra_ds
        raw_h, raw_w = h * extra_ds, w * extra_ds
        raw_h = min(raw_h, level_data.shape[1] - raw_y)
        raw_w = min(raw_w, level_data.shape[2] - raw_x)
        if raw_h <= 0 or raw_w <= 0:
            return [np.zeros((h, w), dtype=np.float32) for _ in range(level_data.shape[0])]
        # Single multi-channel read
        raw = np.asarray(level_data[:, raw_y : raw_y + raw_h, raw_x : raw_x + raw_w])
        channels = []
        for ch in range(raw.shape[0]):
            channels.append(
                cv2.resize(raw[ch].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            )
        return channels
    else:
        arr_h = min(h, level_data.shape[1] - y)
        arr_w = min(w, level_data.shape[2] - x)
        if arr_h <= 0 or arr_w <= 0:
            return [np.zeros((h, w), dtype=np.float32) for _ in range(level_data.shape[0])]
        # Single multi-channel read
        raw = np.asarray(level_data[:, y : y + arr_h, x : x + arr_w]).astype(np.float32)
        channels = []
        for ch in range(raw.shape[0]):
            if raw[ch].shape == (h, w):
                channels.append(raw[ch])
            else:
                # Zero-pad to requested size at image boundary
                padded = np.zeros((h, w), dtype=np.float32)
                padded[: raw[ch].shape[0], : raw[ch].shape[1]] = raw[ch]
                channels.append(padded)
        return channels
