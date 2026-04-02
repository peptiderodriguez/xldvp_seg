"""Find ROI regions via summed marker signal and Otsu thresholding.

Extracted from ``examples/islet/analyze_islets.py::find_islet_regions()``.
The logic is generalised: any set of marker channels can be summed and
thresholded to find bright tissue regions (islets, tumour foci, etc.).
"""

from __future__ import annotations

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def find_regions_by_marker_signal(
    channel_data: dict[int, np.ndarray],
    marker_channels: list[int],
    pixel_size: float,
    downsample: int = 4,
    blur_sigma_um: float = 10.0,
    otsu_multiplier: float = 1.5,
    min_area_um2: float = 500.0,
    buffer_um: float = 25.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Find tissue regions with strong marker signal.

    Downsamples the requested marker channels, percentile-normalises each to
    [0, 1], sums them, applies Gaussian blur, Otsu-thresholds (with
    multiplier), morphologically closes, removes small objects, dilates by a
    buffer, and labels connected components.

    Args:
        channel_data: ``{channel_index: np.ndarray}`` full-resolution 2-D
            arrays (uint16 or float).
        marker_channels: List of channel indices to sum.
        pixel_size: Micrometres per pixel at full resolution.
        downsample: Downsampling factor (default 4).
        blur_sigma_um: Gaussian blur sigma in micrometres.
        otsu_multiplier: Multiply Otsu threshold by this factor (>1 = stricter).
        min_area_um2: Minimum region area in um^2.
        buffer_um: Dilation buffer in um (captures border cells).

    Returns:
        Tuple of ``(region_labels, downsample_factor, signal_heatmap)``.
        *region_labels* is a 2-D int32 array at downsampled resolution where
        0 = background and 1..N = region labels.  *signal_heatmap* is the
        blurred summed signal (float32, same shape).
    """
    from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
    from scipy.ndimage import label as ndi_label
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    ds_pixel_size = pixel_size * downsample

    # Downsample + percentile-normalise each marker channel to [0, 1]
    normalised: list[np.ndarray] = []
    for ch_idx in marker_channels:
        if ch_idx not in channel_data:
            logger.warning("Channel %d not in channel_data — skipped", ch_idx)
            continue
        full = channel_data[ch_idx]
        arr = full[::downsample, ::downsample].astype(np.float32)
        nonzero = arr[arr > 0]
        if len(nonzero) == 0:
            normalised.append(np.zeros_like(arr))
            continue
        p1, p99 = np.percentile(nonzero, [1, 99])
        if p99 > p1:
            arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
        else:
            arr = np.zeros_like(arr)
        # Re-zero CZI padding pixels
        arr[full[::downsample, ::downsample] == 0] = 0
        normalised.append(arr)
        logger.debug("  ch%d: p1=%.0f p99=%.0f", ch_idx, p1, p99)

    if not normalised:
        logger.warning("No valid marker channels — returning empty labels")
        # Compute downsampled shape from first available channel
        first_ch = next(iter(channel_data.values()))
        ds_shape = (first_ch.shape[0] // downsample, first_ch.shape[1] // downsample)
        return (
            np.zeros(ds_shape, dtype=np.int32),
            downsample,
            np.zeros(ds_shape, dtype=np.float32),
        )

    # Sum normalised channels
    signal = np.sum(normalised, axis=0).astype(np.float32)

    # Gaussian blur
    sigma_px = blur_sigma_um / ds_pixel_size
    signal = gaussian_filter(signal, sigma=sigma_px)

    # Otsu on non-zero pixels
    nonzero_signal = signal[signal > 0]
    if len(nonzero_signal) < 100:
        logger.warning("Too few non-zero pixels (%d) for Otsu", len(nonzero_signal))
        return np.zeros(signal.shape, dtype=np.int32), downsample, signal

    if np.std(nonzero_signal) < 1e-6:
        logger.warning("Near-zero variance — cannot threshold")
        return np.zeros(signal.shape, dtype=np.int32), downsample, signal

    otsu_raw = threshold_otsu(nonzero_signal)
    otsu_t = otsu_raw * otsu_multiplier
    binary = signal >= otsu_t

    # Morphological close
    close_um = 10.0
    close_px = max(1, int(round(close_um / ds_pixel_size)))
    struct = np.ones((close_px * 2 + 1, close_px * 2 + 1), dtype=bool)
    binary = binary_closing(binary, structure=struct)

    # Remove small objects
    min_area_px = max(1, int(round(min_area_um2 / (ds_pixel_size**2))))
    binary = remove_small_objects(binary, min_size=min_area_px)

    # Dilate by buffer
    buffer_px = max(1, int(round(buffer_um / ds_pixel_size)))
    buf_struct = np.ones((buffer_px * 2 + 1, buffer_px * 2 + 1), dtype=bool)
    binary = binary_dilation(binary, structure=buf_struct)

    # Label connected components
    region_labels, n_regions = ndi_label(binary)

    logger.info(
        "find_regions_by_marker_signal: %d regions (ds=%d, otsu=%.3f x%.1f=%.3f, "
        "blur=%.1fpx, close=%dpx, min=%dpx, buffer=%dpx)",
        n_regions,
        downsample,
        otsu_raw,
        otsu_multiplier,
        otsu_t,
        sigma_px,
        close_px,
        min_area_px,
        buffer_px,
    )
    return region_labels.astype(np.int32), downsample, signal
