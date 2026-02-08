"""
Cross-slide stain normalization for H&E/brightfield images.

Provides percentile-based and Reinhard normalization to harmonize intensity
distributions across multiple slides, correcting for staining variation and
scanner differences.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


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
        normalized = np.zeros_like(image, dtype=np.uint8)

        for c in range(3):
            # Work with one channel at a time in float32 (much smaller temp array)
            channel = image[:, :, c].astype(np.float32)

            # Sample pixels for percentile estimation (avoid massive temp arrays in np.percentile)
            # Use fast random 2D indexing instead of np.random.choice (much faster for huge arrays)
            n_sample = min(1000000, channel.size)
            if n_sample < channel.size:
                # Random 2D sampling (allows replacement, negligible collision probability)
                h, w = channel.shape
                row_idx = np.random.randint(0, h, size=n_sample)
                col_idx = np.random.randint(0, w, size=n_sample)
                sample = channel[row_idx, col_idx]
            else:
                sample = channel.ravel()
            curr_low, curr_high = np.percentile(sample, [p_low, p_high])

            # Avoid division by zero
            if curr_high - curr_low < 1:
                normalized[:, :, c] = image[:, :, c]
                del channel  # Free temp array before continue
                continue

            # Rescale to target range IN-PLACE (avoid creating temp arrays!)
            # Map [curr_low, curr_high] → [target_low, target_high]
            scale = (target_high[c] - target_low[c]) / (curr_high - curr_low)

            # In-place operations to minimize memory (no intermediate temps)
            channel -= curr_low
            channel *= scale
            channel += target_low[c]

            # Clip in-place and convert to uint8
            np.clip(channel, 0, 255, out=channel)
            normalized[:, :, c] = channel.astype(np.uint8)

            # Free temporary float32 array immediately
            del channel

    elif image.ndim == 2:
        # Grayscale image
        channel = image.astype(np.float32)

        # Sample pixels for percentile estimation (avoid massive temp arrays in np.percentile)
        # Use fast random 2D indexing instead of np.random.choice (much faster for huge arrays)
        n_sample = min(1000000, channel.size)
        if n_sample < channel.size:
            # Random 2D sampling (allows replacement, negligible collision probability)
            h, w = channel.shape
            row_idx = np.random.randint(0, h, size=n_sample)
            col_idx = np.random.randint(0, w, size=n_sample)
            sample = channel[row_idx, col_idx]
        else:
            sample = channel.ravel()
        curr_low, curr_high = np.percentile(sample, [p_low, p_high])

        # Avoid division by zero
        if curr_high - curr_low < 1:
            normalized = image.copy()
            del channel  # Free temp array
        else:
            # Rescale to target range IN-PLACE (avoid creating temp arrays!)
            target_low_val = target_low if np.isscalar(target_low) else target_low[0]
            target_high_val = target_high if np.isscalar(target_high) else target_high[0]
            scale = (target_high_val - target_low_val) / (curr_high - curr_low)

            # In-place operations to minimize memory (no intermediate temps)
            channel -= curr_low
            channel *= scale
            channel += target_low_val

            # Clip in-place and convert to uint8
            np.clip(channel, 0, 255, out=channel)
            normalized = channel.astype(np.uint8)
            del channel

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


# ============================================================================
# Reinhard Normalization (Lab color space)
# ============================================================================

def compute_reinhard_params_from_samples(
    slide_samples: list,
) -> Dict[str, float]:
    """
    Compute global Reinhard normalization parameters from sampled pixels.

    Uses median and MAD (median absolute deviation) for robust statistics
    that are less sensitive to outliers than mean/std.
    Uses cv2 for LAB conversion (consistent with apply function).

    Args:
        slide_samples: List of RGB arrays (N_samples, 3) from each slide

    Returns:
        Dictionary with Lab channel statistics:
        {
            'L_median': float, 'L_mad': float,
            'a_median': float, 'a_mad': float,
            'b_median': float, 'b_mad': float,
            'n_slides': int,
            'n_total_pixels': int,
            'method': 'reinhard_median'
        }
    """
    all_lab_samples = []

    for samples_rgb in slide_samples:
        # Convert RGB samples to Lab color space using cv2 (consistent with apply function)
        # Input: (N, 3) array in range [0, 255]
        # cv2.cvtColor expects uint8 input for COLOR_RGB2LAB
        samples_uint8 = np.clip(samples_rgb, 0, 255).astype(np.uint8)
        samples_img = samples_uint8.reshape(1, -1, 3)
        samples_lab = cv2.cvtColor(samples_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        samples_lab = samples_lab.reshape(-1, 3)  # Back to (N, 3)

        # Convert from cv2 LAB encoding to standard LAB scale
        samples_lab[:, 0] = samples_lab[:, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
        samples_lab[:, 1] = samples_lab[:, 1] - 128.0           # a: [0,255] -> [-128,127]
        samples_lab[:, 2] = samples_lab[:, 2] - 128.0           # b: [0,255] -> [-128,127]

        all_lab_samples.append(samples_lab)

    # Combine all samples from all slides
    combined_lab = np.vstack(all_lab_samples)

    # Compute median and MAD for each Lab channel (robust to outliers)
    L_median = np.median(combined_lab[:, 0])
    L_mad = np.median(np.abs(combined_lab[:, 0] - L_median))

    a_median = np.median(combined_lab[:, 1])
    a_mad = np.median(np.abs(combined_lab[:, 1] - a_median))

    b_median = np.median(combined_lab[:, 2])
    b_mad = np.median(np.abs(combined_lab[:, 2] - b_median))

    return {
        'L_median': float(L_median),
        'L_mad': float(L_mad),
        'a_median': float(a_median),
        'a_mad': float(a_mad),
        'b_median': float(b_median),
        'b_mad': float(b_mad),
        'n_slides': len(slide_samples),
        'n_total_pixels': int(combined_lab.shape[0]),
        'method': 'reinhard_median'
    }



# Deprecated alias: identical to compute_reinhard_params_from_samples (which already
# uses median/MAD). Kept for backward compatibility with external scripts.
compute_reinhard_params_from_samples_MEDIAN = compute_reinhard_params_from_samples


def apply_reinhard_normalization_MEDIAN(
    image: np.ndarray,
    params: Dict[str, float],
    variance_threshold: float = 15.0,
    tile_size: int = 10000,
    block_size: int = 7
) -> np.ndarray:
    """
    Apply Reinhard normalization with MEDIAN/MAD (robust version).

    Uses block-level tissue detection (same as Phase 1 compute_normalization_params.py):
    512x512 blocks, K-means calibrated threshold / 10, cv2 LAB conversion.
    All pixels within tissue blocks are normalized; background blocks unchanged.

    Args:
        image: RGB image (H, W, 3) in range [0, 255], dtype uint8
        params: Dictionary with Lab statistics from Phase 1
                Must contain: L_median, L_mad, a_median, a_mad, b_median, b_mad
        variance_threshold: (unused, kept for API compatibility)
        tile_size: (unused, kept for API compatibility)
        block_size: (unused, kept for API compatibility)

    Returns:
        Normalized RGB image (H, W, 3), dtype uint8
    """
    import cv2
    from segmentation.detection.tissue import calibrate_tissue_threshold, filter_tissue_tiles

    h, w, c = image.shape
    block_sz = 512

    # Step 1: Block-level tissue detection (same method as Phase 1)
    logger.info(f"  Tissue detection: block-level ({block_sz}x{block_sz}), same as Phase 1")
    blocks = []
    for y in range(0, h, block_sz):
        for x in range(0, w, block_sz):
            blocks.append({'x': x, 'y': y})

    logger.info(f"  Total blocks: {len(blocks)}")

    # Calibrate threshold using K-means (same as Phase 1)
    var_threshold = calibrate_tissue_threshold(
        blocks,
        image_array=image,
        calibration_samples=min(100, len(blocks)),
        block_size=block_sz,
        tile_size=block_sz
    )
    logger.info(f"  Calibrated variance threshold: {var_threshold:.1f}")

    # Reduce by 10x (same as Phase 1)
    var_threshold /= 10.0
    logger.info(f"  Reduced threshold (10x): {var_threshold:.1f}")

    # Filter to tissue blocks
    tissue_blocks = filter_tissue_tiles(
        blocks,
        var_threshold,
        image_array=image,
        tile_size=block_sz,
        block_size=block_sz,
        n_workers=8,
        show_progress=False
    )

    logger.info(f"  Tissue blocks: {len(tissue_blocks)} / {len(blocks)} ({100*len(tissue_blocks)/len(blocks):.1f}%)")

    if len(tissue_blocks) == 0:
        logger.warning("  No tissue blocks found, returning original image")
        return image.copy()

    del blocks

    # Step 2: Sample random pixels from tissue blocks for stats (same as Phase 1)
    n_samples = 1000000
    logger.info(f"  Sampling {n_samples:,} pixels from tissue blocks for stats...")

    # Vectorized sampling: pick random blocks, random offsets within each
    block_origins = np.array([(b['x'], b['y']) for b in tissue_blocks])
    block_indices = np.random.randint(0, len(tissue_blocks), size=n_samples)
    selected_origins = block_origins[block_indices]

    # Per-block-bounded offsets (matches compute_normalization_params.py)
    # Prevents over-sampling boundary pixels at image edges
    max_x_offsets = np.minimum(block_sz, w - selected_origins[:, 0])
    max_y_offsets = np.minimum(block_sz, h - selected_origins[:, 1])
    max_x_offsets = np.maximum(max_x_offsets, 1)  # avoid zero range
    max_y_offsets = np.maximum(max_y_offsets, 1)
    x_offsets = (np.random.random(n_samples) * max_x_offsets).astype(np.intp)
    y_offsets = (np.random.random(n_samples) * max_y_offsets).astype(np.intp)
    xs = selected_origins[:, 0] + x_offsets
    ys = selected_origins[:, 1] + y_offsets

    samples = image[ys, xs]  # (n_samples, 3) uint8

    del block_origins, block_indices, selected_origins, x_offsets, y_offsets, xs, ys

    # Convert to LAB using cv2 (same as Phase 1)
    samples_img = samples.reshape(1, -1, 3)
    lab = cv2.cvtColor(samples_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = lab.reshape(-1, 3)
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
    lab[:, 1] = lab[:, 1] - 128.0           # a: [0,255] -> [-128,127]
    lab[:, 2] = lab[:, 2] - 128.0           # b: [0,255] -> [-128,127]

    del samples, samples_img

    # Compute image-level MEDIAN and MAD
    L_img_median = np.median(lab[:, 0])
    L_img_mad = np.median(np.abs(lab[:, 0] - L_img_median))
    a_img_median = np.median(lab[:, 1])
    a_img_mad = np.median(np.abs(lab[:, 1] - a_img_median))
    b_img_median = np.median(lab[:, 2])
    b_img_mad = np.median(np.abs(lab[:, 2] - b_img_median))

    del lab

    # Log internal stats for verification
    logger.info(f"  Image stats ({n_samples:,} samples from {len(tissue_blocks)} tissue blocks):")
    logger.info(f"    L: median={L_img_median:.2f}, MAD={L_img_mad:.2f}  (target: {params['L_median']:.2f}, {params['L_mad']:.2f})")
    logger.info(f"    a: median={a_img_median:.2f}, MAD={a_img_mad:.2f}  (target: {params['a_median']:.2f}, {params['a_mad']:.2f})")
    logger.info(f"    b: median={b_img_median:.2f}, MAD={b_img_mad:.2f}  (target: {params['b_median']:.2f}, {params['b_mad']:.2f})")

    # Step 3: Normalize all pixels in tissue blocks
    result = image.copy()

    for block in tissue_blocks:
        bx, by = block['x'], block['y']
        by_end = min(by + block_sz, h)
        bx_end = min(bx + block_sz, w)

        block_img = image[by:by_end, bx:bx_end]

        # Convert to LAB using cv2 (same as Phase 1)
        block_lab = cv2.cvtColor(block_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        block_lab[:, :, 0] = block_lab[:, :, 0] * 100.0 / 255.0
        block_lab[:, :, 1] = block_lab[:, :, 1] - 128.0
        block_lab[:, :, 2] = block_lab[:, :, 2] - 128.0

        # Apply normalization: (x - img_median) / img_mad * target_mad + target_median
        if L_img_mad > 1.0:
            block_lab[:, :, 0] = (block_lab[:, :, 0] - L_img_median) / L_img_mad * params['L_mad'] + params['L_median']
        else:
            block_lab[:, :, 0] = params['L_median']

        if a_img_mad > 1.0:
            block_lab[:, :, 1] = (block_lab[:, :, 1] - a_img_median) / a_img_mad * params['a_mad'] + params['a_median']
        else:
            block_lab[:, :, 1] = params['a_median']

        if b_img_mad > 1.0:
            block_lab[:, :, 2] = (block_lab[:, :, 2] - b_img_median) / b_img_mad * params['b_mad'] + params['b_median']
        else:
            block_lab[:, :, 2] = params['b_median']

        # Convert back to cv2 LAB encoding and then to RGB
        block_lab[:, :, 0] = block_lab[:, :, 0] * 255.0 / 100.0
        block_lab[:, :, 1] = block_lab[:, :, 1] + 128.0
        block_lab[:, :, 2] = block_lab[:, :, 2] + 128.0

        block_lab = np.clip(block_lab, 0, 255).astype(np.uint8)
        block_rgb = cv2.cvtColor(block_lab, cv2.COLOR_LAB2RGB)

        result[by:by_end, bx:bx_end] = block_rgb
        del block_img, block_lab, block_rgb

    return result


def apply_reinhard_normalization(
    image: np.ndarray,
    params: Dict[str, float],
    variance_threshold: float = 15.0,
    tile_size: int = 10000,
    block_size: int = 7
) -> np.ndarray:
    """
    Apply Reinhard normalization with block-level tissue detection.

    Uses median/MAD statistics (robust to outliers) and cv2 for LAB conversion
    (consistent with Phase 1 compute_normalization_params.py).

    Delegates to apply_reinhard_normalization_MEDIAN which implements the
    block-level tissue detection approach matching Phase 1.

    Args:
        image: RGB image (H, W, 3) in range [0, 255], dtype uint8
        params: Dictionary with Lab statistics (must contain L_median, L_mad, etc.)
        variance_threshold: (unused, kept for API compatibility)
        tile_size: (unused, kept for API compatibility)
        block_size: (unused, kept for API compatibility)

    Returns:
        Normalized RGB image (H, W, 3), dtype uint8
    """
    return apply_reinhard_normalization_MEDIAN(
        image, params,
        variance_threshold=variance_threshold,
        tile_size=tile_size,
        block_size=block_size
    )
