"""Image processing utilities and HDF5 helpers for HTML export.

Extracted from ``html_export.py`` for reuse across the I/O package.

**Image utilities:** percentile_normalize, draw_mask_contour, image_to_base64,
    get_largest_connected_component, compose_tile_rgb
**HDF5 helpers:** create_hdf5_dataset, HDF5_COMPRESSION_KWARGS, HDF5_COMPRESSION_NAME
**HTML helpers:** _esc (HTML/JS string escaping)
"""

import base64
import html as html_mod
from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage

from xldvp_seg.utils.logging import get_logger
from xldvp_seg.utils.mask_cleanup import get_largest_connected_component  # noqa: F401

logger = get_logger(__name__)


def _esc(value) -> str:
    """Escape a value for safe insertion into HTML/JS strings.

    Prevents XSS by escaping <, >, &, ", and ' characters.
    """
    return html_mod.escape(str(value), quote=True)


# Try to use LZ4 compression (faster than gzip), fallback to gzip
try:
    import hdf5plugin

    # LZ4 is ~3-5x faster than gzip with similar compression ratio for image masks
    HDF5_COMPRESSION_KWARGS = hdf5plugin.LZ4(nbytes=0)  # Returns dict-like for **unpacking
    HDF5_COMPRESSION_NAME = "LZ4"
except ImportError:
    HDF5_COMPRESSION_KWARGS = {"compression": "gzip"}
    HDF5_COMPRESSION_NAME = "gzip"


def create_hdf5_dataset(f, name, data):
    """Create HDF5 dataset with best available compression (LZ4 or gzip)."""
    if isinstance(HDF5_COMPRESSION_KWARGS, dict):
        f.create_dataset(name, data=data, **HDF5_COMPRESSION_KWARGS)
    else:
        # hdf5plugin filter object — pass as compression kwarg
        f.create_dataset(name, data=data, **dict(HDF5_COMPRESSION_KWARGS))


def percentile_normalize(image, p_low=1, p_high=99.5, global_percentiles=None):
    """
    Normalize image using percentiles.

    Percentiles are computed on non-zero pixels only (CZI padding is exactly 0).
    Zero pixels stay black after normalization.

    Args:
        image: 2D or 3D numpy array
        p_low: Lower percentile for normalization (default 1)
        p_high: Upper percentile for normalization (default 99.5)
        global_percentiles: Optional dict {channel_index: (low_val, high_val)}.
            When provided and the channel index exists in the dict, use these
            precomputed percentile values instead of computing from the crop.
            Only applies to multi-channel (3D) images.

    Returns:
        uint8 normalized image
    """
    if image.ndim == 2:
        # Percentile on non-zero pixels only (exclude CZI padding)
        nonzero = image[image > 0]
        if len(nonzero) == 0:
            return np.zeros_like(image, dtype=np.uint8)
        low_val = np.percentile(nonzero, p_low)
        high_val = np.percentile(nonzero, p_high)
        if high_val > low_val:
            normalized = (image.astype(np.float32) - low_val) / (high_val - low_val) * 255
            result = np.clip(normalized, 0, 255).astype(np.uint8)
            result[image == 0] = 0  # Keep padding black
            return result
        if image.dtype == np.uint16:
            return (image / 256).astype(np.uint8)
        from xldvp_seg.utils.detection_utils import safe_to_uint8

        return safe_to_uint8(image)
    else:
        # Multi-channel: valid pixel = any channel > 0
        h, w, c = image.shape
        valid_mask = np.max(image, axis=2) > 0
        result = np.zeros_like(image, dtype=np.uint8)
        for ch in range(c):
            ch_data = image[:, :, ch]
            valid_pixels = ch_data[valid_mask]
            if len(valid_pixels) == 0:
                continue
            if global_percentiles is not None and ch in global_percentiles:
                low_val, high_val = global_percentiles[ch]
            else:
                low_val = np.percentile(valid_pixels, p_low)
                high_val = np.percentile(valid_pixels, p_high)
            if high_val > low_val:
                normalized = (ch_data.astype(np.float32) - low_val) / (high_val - low_val) * 255
                result[:, :, ch] = np.clip(normalized, 0, 255).astype(np.uint8)
            else:
                if image.dtype == np.uint16:
                    result[:, :, ch] = (ch_data / 256).astype(np.uint8)
                else:
                    from xldvp_seg.utils.detection_utils import safe_to_uint8

                    result[:, :, ch] = safe_to_uint8(ch_data)
        # Keep padding pixels black
        result[~valid_mask] = 0
        return result


def draw_mask_contour(
    img_array, mask, color=(0, 255, 0), thickness=2, dotted=False, use_cv2=True, bw_dashed=False
):
    """
    Draw mask contour on image.

    Args:
        img_array: RGB image array (or grayscale, will be converted)
        mask: Binary mask
        color: RGB tuple for contour color
        thickness: Contour thickness in pixels
        dotted: Whether to use dotted line
        use_cv2: Use OpenCV for faster, smoother contours (default True)
        bw_dashed: Draw alternating black/white dashed contour (overrides color)

    Returns:
        Image with contour drawn (always RGB)
    """
    import cv2

    # Ensure RGB
    if img_array.ndim == 2:
        img_out = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_out = img_array.copy()

    if bw_dashed:
        # Thin green dashed contour line
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        dash_len, gap_len = 8, 5
        line_thickness = thickness
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            cycle = dash_len + gap_len
            n_pts = len(pts)
            i = 0
            while i < n_pts:
                j = min(i + dash_len, n_pts - 1)
                if j > i:
                    cv2.line(img_out, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), line_thickness)
                i += cycle
        return img_out

    if use_cv2 and not dotted:
        # Use cv2.drawContours for smooth, thick lines
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Convert RGB to BGR for cv2, then back
        cv2.drawContours(img_out, contours, -1, color, thickness)
        return img_out

    # Fallback to dilation method
    dilated = ndimage.binary_dilation(mask, iterations=thickness)
    contour = dilated & ~mask
    ys, xs = np.where(contour)

    if len(ys) == 0:
        return img_out

    if dotted:
        # Subsample every 3rd pixel for dotted effect
        dot_mask = np.zeros(len(ys), dtype=bool)
        dot_mask[::3] = True
        ys_draw, xs_draw = ys[dot_mask], xs[dot_mask]
    else:
        ys_draw, xs_draw = ys, xs
    valid = (
        (ys_draw >= 0)
        & (ys_draw < img_out.shape[0])
        & (xs_draw >= 0)
        & (xs_draw < img_out.shape[1])
    )
    img_out[ys_draw[valid], xs_draw[valid]] = color

    return img_out


def image_to_base64(img_array, format="JPEG", quality=85):
    """
    Convert numpy array or PIL image to base64 string.

    Args:
        img_array: numpy array or PIL Image
        format: Image format ('JPEG' or 'PNG')
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded string
    """
    if isinstance(img_array, np.ndarray):
        pil_img = Image.fromarray(img_array)
    else:
        pil_img = img_array

    # Use JPEG for opaque images (smaller, faster). PNG only for transparency.
    has_alpha = pil_img.mode in ("RGBA", "LA", "PA")
    buffer = BytesIO()
    if has_alpha:
        pil_img.save(buffer, format="PNG", optimize=True)
        mime_type = "png"
    else:
        out_format = format.upper() if format else "JPEG"
        if pil_img.mode != "RGB" and out_format == "JPEG":
            pil_img = pil_img.convert("RGB")
        pil_img.save(buffer, format=out_format, quality=quality)
        mime_type = out_format.lower()

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), mime_type


# =============================================================================
# TILE RGB COMPOSITION (shared by pipeline resume + regenerate_html.py)
# =============================================================================


def compose_tile_rgb(
    channel_arrays,
    tile_x,
    tile_y,
    tile_size,
    display_channels,
    x_start,
    y_start,
    mosaic_h,
    mosaic_w,
):
    """Extract a tile region and compose RGB from display channels.

    Args:
        channel_arrays: List of 2D arrays indexed by channel number (None for missing).
        tile_x, tile_y: Tile origin in mosaic (global) coordinates.
        tile_size: Tile dimension in pixels.
        display_channels: List of channel indices for [R, G, B].
        x_start, y_start: Mosaic origin offset.
        mosaic_h, mosaic_w: Mosaic array dimensions.

    Returns:
        (h, w, 3) uint8 array with per-channel percentile normalization,
        or None if tile is entirely outside bounds.
    """
    # Convert to array coordinates (subtract mosaic origin)
    ay = tile_y - y_start
    ax = tile_x - x_start

    # Clamp to array bounds
    ay_end = min(ay + tile_size, mosaic_h)
    ax_end = min(ax + tile_size, mosaic_w)
    ay = max(0, ay)
    ax = max(0, ax)

    if ay_end <= ay or ax_end <= ax:
        return None

    h = ay_end - ay
    w = ax_end - ax
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i, ch_idx in enumerate(display_channels[:3]):
        if ch_idx < len(channel_arrays) and channel_arrays[ch_idx] is not None:
            ch_data = channel_arrays[ch_idx][ay:ay_end, ax:ax_end]
            # Percentile normalize to uint8 (non-zero pixels only)
            valid = ch_data > 0
            if valid.any():
                p1 = np.percentile(ch_data[valid], 1)
                p99 = np.percentile(ch_data[valid], 99.5)
                if p99 > p1:
                    norm = np.clip(
                        (ch_data.astype(np.float32) - p1) / (p99 - p1) * 255,
                        0,
                        255,
                    ).astype(np.uint8)
                    norm[~valid] = 0
                    rgb[:, :, i] = norm
    return rgb
