"""Find ROI regions that are roughly circular (e.g. pancreatic islets).

Identifies bright connected components in a single channel, then filters by
area (from a diameter range) and circularity (4*pi*area/perimeter^2).
"""

from __future__ import annotations

import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def find_circular_regions(
    channel_data: dict[int, np.ndarray],
    channel_idx: int,
    pixel_size: float,
    downsample: int = 8,
    min_diameter_um: float = 400.0,
    max_diameter_um: float = 3000.0,
    min_circularity: float = 0.4,
) -> tuple[np.ndarray, int]:
    """Find roughly-circular bright regions in a single channel.

    1. Downsample the channel.
    2. Otsu-threshold tissue vs background.
    3. Morphological close + fill holes.
    4. Label connected components.
    5. Filter by area (derived from diameter range) and circularity.

    Args:
        channel_data: ``{channel_index: np.ndarray}`` full-resolution 2-D arrays.
        channel_idx: Which channel to use.
        pixel_size: Micrometres per pixel at full resolution.
        downsample: Downsampling factor (default 8).
        min_diameter_um: Minimum diameter in um.
        max_diameter_um: Maximum diameter in um.
        min_circularity: Minimum circularity (4*pi*area/perimeter^2).

    Returns:
        ``(region_labels, downsample_factor)`` where *region_labels* is a 2-D
        int32 array at downsampled resolution (0 = background, 1..N = regions).
    """
    import math

    from scipy.ndimage import binary_closing, binary_fill_holes
    from scipy.ndimage import label as ndi_label
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops

    if channel_idx not in channel_data:
        logger.warning("Channel %d not in channel_data", channel_idx)
        return np.zeros((1, 1), dtype=np.int32), downsample

    full = channel_data[channel_idx]
    arr = full[::downsample, ::downsample].astype(np.float32)

    ds_pixel_size = pixel_size * downsample

    # Otsu threshold
    nonzero = arr[arr > 0]
    if len(nonzero) < 100:
        logger.warning("Too few non-zero pixels for Otsu (%d)", len(nonzero))
        return np.zeros(arr.shape, dtype=np.int32), downsample

    if np.std(nonzero) < 1e-6:
        logger.warning("Near-zero variance — cannot threshold")
        return np.zeros(arr.shape, dtype=np.int32), downsample

    thresh = threshold_otsu(nonzero)
    binary = arr >= thresh

    # Morphological close + fill holes
    close_px = max(1, int(round(20.0 / ds_pixel_size)))
    struct = np.ones((close_px * 2 + 1, close_px * 2 + 1), dtype=bool)
    binary = binary_closing(binary, structure=struct)
    binary = binary_fill_holes(binary)

    # Label connected components
    labels_all, n_all = ndi_label(binary)

    # Area bounds (in downsampled pixels)
    min_area_px = math.pi * (min_diameter_um / (2 * ds_pixel_size)) ** 2
    max_area_px = math.pi * (max_diameter_um / (2 * ds_pixel_size)) ** 2

    # Filter by area + circularity
    props = regionprops(labels_all)
    keep_labels: set[int] = set()

    for rp in props:
        area = rp.area
        perimeter = rp.perimeter
        if perimeter == 0:
            continue
        circularity = (4 * math.pi * area) / (perimeter**2)
        if min_area_px <= area <= max_area_px and circularity >= min_circularity:
            keep_labels.add(rp.label)

    # Build filtered label array with consecutive labels
    if not keep_labels:
        logger.info("find_circular_regions: 0 regions passed filters (from %d candidates)", n_all)
        return np.zeros(arr.shape, dtype=np.int32), downsample

    filtered = np.zeros_like(labels_all, dtype=np.int32)
    new_label = 0
    for old_label in sorted(keep_labels):
        new_label += 1
        filtered[labels_all == old_label] = new_label

    logger.info(
        "find_circular_regions: %d / %d regions passed (area=[%.0f, %.0f] px, circ>=%.2f)",
        new_label,
        n_all,
        min_area_px,
        max_area_px,
        min_circularity,
    )
    return filtered, downsample
