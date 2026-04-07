"""CZI thumbnail loading and base64 encoding for HTML viewers."""

import base64
import io
import re

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def encode_channel_b64(ch_array):
    """Encode a single-channel uint8 array as PNG base64 string."""
    from PIL import Image

    img = Image.fromarray(ch_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    del buf
    return b64


def parse_scene_index(name):
    """Extract CZI scene index from a slide/panel name.

    Supports patterns like ``scene_3``, ``scene03``, or ``Scene_3``.
    Returns 0 (the default scene) when the name doesn't encode a scene index,
    so single-scene CZIs and multi-slide directories work unchanged.
    """
    m = re.match(r"(?i)scene[_-]?(\d+)", name)
    return int(m.group(1)) if m else 0


def read_czi_thumbnail_channels(czi_path, display_channels, scale_factor=0.0625, scene=0):
    """Read CZI mosaic channels at low resolution, return per-channel uint8 arrays.

    Uses aicspylibczi read_mosaic(scale_factor=...) for memory-efficient loading.
    Each channel is percentile-normalised to uint8 independently.

    Args:
        czi_path: Path to CZI file (str or Path).
        display_channels: List of channel indices to read (up to 3).
        scale_factor: Downsampling factor (default 1/16 = 0.0625).
        scene: CZI scene index (0-based, default 0).

    Returns:
        channel_arrays: list of uint8 arrays, one per channel (height x width).
        pixel_size_um: pixel size in um at full resolution, or None.
        mosaic_x: mosaic origin x in full-resolution pixels.
        mosaic_y: mosaic origin y in full-resolution pixels.
    """
    from aicspylibczi import CziFile

    czi = CziFile(str(czi_path))

    # Get pixel size from metadata — try CZILoader first (more robust), then aicspylibczi
    pixel_size_um = None
    try:
        from xldvp_seg.io.czi_loader import CZILoader

        loader = CZILoader(str(czi_path), scene=scene)
        pixel_size_um = loader.get_pixel_size()
        del loader  # close file handle
    except Exception:
        logger.debug("CZI pixel size extraction failed", exc_info=True)
    if pixel_size_um is None:
        try:
            scaling = czi.get_scaling()
            if scaling and len(scaling) >= 1:
                pixel_size_um = scaling[0] * 1e6  # m -> um
        except Exception:
            logger.debug("Bounding box scaling failed", exc_info=True)

    # Get mosaic bounding box for the scene
    try:
        bbox = czi.get_mosaic_scene_bounding_box(index=scene)
    except Exception:
        bbox = czi.get_mosaic_bounding_box()
    region = (bbox.x, bbox.y, bbox.w, bbox.h)
    mosaic_x = bbox.x
    mosaic_y = bbox.y
    logger.info(
        "CZI scene %d: %dx%d px at (%d,%d), scale=%s",
        scene,
        bbox.w,
        bbox.h,
        bbox.x,
        bbox.y,
        scale_factor,
    )

    channel_arrays = []
    for ch in display_channels:
        logger.info("Reading channel %d...", ch)
        try:
            img = czi.read_mosaic(C=ch, region=region, scale_factor=scale_factor)
            img = np.squeeze(img)
            logger.info("  Channel %d: %s %s", ch, img.shape, img.dtype)
        except Exception as exc:
            logger.warning("  Channel %d FAILED: %s", ch, exc)
            channel_arrays.append(None)
            continue

        # Percentile-normalise to uint8 (exclude zeros which are CZI padding)
        nonzero_mask = img > 0 if img.dtype != np.uint8 else None
        valid = img[nonzero_mask] if nonzero_mask is not None else img.ravel()
        if len(valid) == 0:
            channel_arrays.append(np.zeros(img.shape[:2], dtype=np.uint8))
            continue
        p_low = float(np.percentile(valid, 1))
        p_high = float(np.percentile(valid, 99.5))
        del valid  # free memory before float32 conversion
        if p_high <= p_low:
            p_high = p_low + 1.0
        # In-place normalisation to avoid extra float32 copy
        result = img.astype(np.float32)
        result -= p_low
        result /= p_high - p_low
        np.clip(result, 0.0, 1.0, out=result)
        result *= 255
        result = result.astype(np.uint8)
        if nonzero_mask is not None:
            result[~nonzero_mask] = 0  # preserve CZI padding as black
        channel_arrays.append(result)

    return channel_arrays, pixel_size_um, mosaic_x, mosaic_y
