"""
Cross-slide stain normalization for H&E/brightfield images.

Provides percentile-based normalization to harmonize intensity distributions
across multiple slides, correcting for staining variation and scanner differences.
"""

import numpy as np
from typing import Tuple, Optional


def compute_global_percentiles(
    slides_data: list,
    p_low: float = 1.0,
    p_high: float = 99.0,
    n_samples: int = 100000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global percentiles across all slides for normalization.

    Args:
        slides_data: List of RGB arrays (each slide)
        p_low: Lower percentile (default 1.0)
        p_high: Upper percentile (default 99.0)
        n_samples: Number of pixels to sample per slide

    Returns:
        (low_values, high_values): Arrays of shape (3,) for RGB channels
    """
    all_samples = []

    for slide_rgb in slides_data:
        # Handle both RGB (H, W, 3) and grayscale (H, W)
        if slide_rgb.ndim == 3:
            h, w, c = slide_rgb.shape
            n_pixels = h * w
            n_sample = min(n_samples, n_pixels)

            # Random sampling with direct 2D indexing (no reshape/flatten)
            row_indices = np.random.randint(0, h, size=n_sample)
            col_indices = np.random.randint(0, w, size=n_sample)
            samples = slide_rgb[row_indices, col_indices, :].copy()
        elif slide_rgb.ndim == 2:
            h, w = slide_rgb.shape
            n_pixels = h * w
            n_sample = min(n_samples, n_pixels)

            # Random sampling for grayscale with direct 2D indexing
            row_indices = np.random.randint(0, h, size=n_sample)
            col_indices = np.random.randint(0, w, size=n_sample)
            samples = slide_rgb[row_indices, col_indices].copy().reshape(-1, 1)  # Reshape to (N, 1) for vstack compatibility
        else:
            raise ValueError(f"Unexpected slide shape: {slide_rgb.shape}")

        all_samples.append(samples)

    # Combine all samples
    combined = np.vstack(all_samples)

    # Compute global percentiles per channel (both in single pass)
    low_vals, high_vals = np.percentile(combined, [p_low, p_high], axis=0)

    return low_vals, high_vals


def normalize_to_percentiles(
    image: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """
    Normalize image to match target percentile range.

    Memory-optimized: processes one channel at a time to avoid creating
    a full float32 copy of the entire image (4x memory reduction).

    Args:
        image: RGB image (H, W, 3) or grayscale (H, W)
        target_low: Target values for p_low percentile (shape: 3 for RGB, scalar for grayscale)
        target_high: Target values for p_high percentile (shape: 3 for RGB, scalar for grayscale)
        p_low: Lower percentile
        p_high: Upper percentile

    Returns:
        Normalized image (uint8)
    """
    # Handle RGB vs grayscale
    if image.ndim == 3 and image.shape[2] == 3:
        # RGB image - allocate output as uint8 directly (not float32!)
        normalized = np.empty_like(image, dtype=np.uint8)

        for c in range(3):
            # Work with one channel at a time in float32 (much smaller temp array)
            channel = image[:, :, c].astype(np.float32)

            # Compute current percentiles (both in single pass)
            curr_low, curr_high = np.percentile(channel, [p_low, p_high])

            # Avoid division by zero
            if curr_high - curr_low < 1:
                normalized[:, :, c] = image[:, :, c]
                continue

            # Rescale to target range
            # Map [curr_low, curr_high] → [target_low, target_high]
            scale = (target_high[c] - target_low[c]) / (curr_high - curr_low)
            channel_normalized = (channel - curr_low) * scale + target_low[c]

            # Clip and convert to uint8, write directly to output
            normalized[:, :, c] = np.clip(channel_normalized, 0, 255).astype(np.uint8)

            # Free temporary float32 arrays immediately
            del channel, channel_normalized

    elif image.ndim == 2:
        # Grayscale image
        channel = image.astype(np.float32)

        # Compute current percentiles (both in single pass)
        curr_low, curr_high = np.percentile(channel, [p_low, p_high])

        # Avoid division by zero
        if curr_high - curr_low < 1:
            normalized = image.copy()
        else:
            # Rescale to target range
            target_low_val = target_low if np.isscalar(target_low) else target_low[0]
            target_high_val = target_high if np.isscalar(target_high) else target_high[0]
            scale = (target_high_val - target_low_val) / (curr_high - curr_low)
            channel_normalized = (channel - curr_low) * scale + target_low_val

            # Clip and convert to uint8
            normalized = np.clip(channel_normalized, 0, 255).astype(np.uint8)
            del channel, channel_normalized

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return normalized


def normalize_slide_to_reference(
    image: np.ndarray,
    reference_image: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """
    Normalize a slide to match the intensity distribution of a reference slide.

    Args:
        image: RGB image to normalize
        reference_image: Reference RGB image
        p_low: Lower percentile
        p_high: Upper percentile

    Returns:
        Normalized RGB image
    """
    # Compute reference percentiles (all channels and both percentiles in one pass)
    ref_low, ref_high = np.percentile(reference_image, [p_low, p_high], axis=(0, 1))

    # Normalize to match reference
    return normalize_to_percentiles(image, ref_low, ref_high, p_low, p_high)


def percentile_normalize_rgb(
    image: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    target_range: Tuple[float, float] = (0, 255)
) -> np.ndarray:
    """
    Simple per-slide percentile normalization (no cross-slide reference).

    Args:
        image: RGB image
        p_low: Lower percentile to clip
        p_high: Upper percentile to clip
        target_range: Output range (default 0-255)

    Returns:
        Normalized image
    """
    target_low = np.full(3, target_range[0], dtype=np.float32)
    target_high = np.full(3, target_range[1], dtype=np.float32)

    return normalize_to_percentiles(image, target_low, target_high, p_low, p_high)
