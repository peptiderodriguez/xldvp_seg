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

_LAB_KEYS = ('L_median', 'L_mad', 'a_median', 'a_mad', 'b_median', 'b_mad')


def extract_slide_norm_params(slide_thresh):
    """Extract Otsu threshold and per-slide LAB stats from a per-slide threshold dict.

    Args:
        slide_thresh: Per-slide dict from tissue_thresholds JSON, or None.

    Returns:
        tuple: (otsu_threshold, slide_lab_stats) — either or both may be None.
    """
    if not slide_thresh:
        return None, None
    otsu = slide_thresh.get('otsu_threshold', slide_thresh.get('intensity_threshold'))
    slab = {k: slide_thresh[k] for k in _LAB_KEYS if k in slide_thresh}
    return otsu, slab if len(slab) == 6 else None


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

    Thin wrapper around :func:`segmentation.io.html_export.percentile_normalize`,
    which is the canonical implementation. This wrapper preserves the original
    call signature for backward compatibility.

    Args:
        image: RGB image (H, W, 3) or grayscale (H, W)
        p_low: Lower percentile to clip
        p_high: Upper percentile to clip
        target_range: Output range (default 0-255). Only (0, 255) is supported;
            other ranges fall back to the legacy normalize_to_percentiles path.

    Returns:
        Normalized uint8 image
    """
    # The canonical percentile_normalize always maps to [0, 255].
    # If caller requests a non-default target_range, fall back to the old path.
    if target_range != (0, 255):
        target_low = np.full(3, target_range[0], dtype=np.float32)
        target_high = np.full(3, target_range[1], dtype=np.float32)
        return normalize_to_percentiles(image, target_low, target_high, p_low, p_high)

    from segmentation.io.html_export import percentile_normalize
    return percentile_normalize(image, p_low=p_low, p_high=p_high)


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
    otsu_threshold: float = None,
    slide_lab_stats: Dict[str, float] = None,
    **_legacy_kwargs,
) -> np.ndarray:
    """
    Apply Reinhard normalization with pixel-level tissue masking.

    Iterates over 512x512 blocks for memory efficiency. Within each block,
    pixels with (gray > 0) & (gray < otsu_threshold) are tissue — only those
    get normalized. Background pixels are left untouched.

    Args:
        image: RGB image (H, W, 3) in range [0, 255], dtype uint8
        params: Dictionary with global target Lab statistics from Phase 1
                Must contain: L_median, L_mad, a_median, a_mad, b_median, b_mad
        otsu_threshold: Per-slide Otsu threshold from step 1 JSON.
            If None, computed from image.
        slide_lab_stats: Per-slide LAB stats dict from step 1 JSON with keys:
            L_median, L_mad, a_median, a_mad, b_median, b_mad.
            If None, computed from image by sampling tissue pixels.
        **_legacy_kwargs: Accepts (and ignores) former dead parameters
            ``variance_threshold``, ``tile_size``, ``block_size``,
            ``precomputed_variance_threshold``, ``precomputed_intensity_threshold``
            for backward compatibility.

    Returns:
        Normalized RGB image (H, W, 3), dtype uint8
    """
    from segmentation.detection.tissue import compute_otsu_threshold

    h, w, c = image.shape
    block_sz = 512
    gray = None  # Lazy — only computed if otsu_threshold or slide_lab_stats is missing

    # Step 1: Get Otsu threshold (from step 1 JSON or compute)
    if otsu_threshold is not None:
        otsu_val = otsu_threshold
        logger.info(f"  Using pre-computed Otsu threshold: {otsu_val:.1f}")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        otsu_val = compute_otsu_threshold(gray)
        logger.info(f"  Computed Otsu threshold from image: {otsu_val:.1f}")

    # Step 2: Get per-slide LAB stats (from step 1 JSON or compute by sampling)
    if slide_lab_stats is not None:
        L_src_median = slide_lab_stats['L_median']
        L_src_mad = slide_lab_stats['L_mad']
        a_src_median = slide_lab_stats['a_median']
        a_src_mad = slide_lab_stats['a_mad']
        b_src_median = slide_lab_stats['b_median']
        b_src_mad = slide_lab_stats['b_mad']
        logger.info(f"  Using pre-computed slide LAB stats: L={L_src_median:.2f}±{L_src_mad:.2f}")
    else:
        # Fallback: rejection-sample 1M tissue pixels and compute LAB stats
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        logger.info(f"  Computing slide LAB stats by sampling tissue pixels...")
        n_samples = 1_000_000
        max_attempts = n_samples * 5
        collected = []
        total_collected = 0
        attempt = 0
        while total_collected < n_samples and attempt < max_attempts:
            batch = min(n_samples * 2, max_attempts - attempt)
            rows = np.random.randint(0, h, size=batch)
            cols = np.random.randint(0, w, size=batch)
            g = gray[rows, cols]
            mask = (g > 0) & (g < otsu_val)
            tissue_rows = rows[mask]
            tissue_cols = cols[mask]
            if len(tissue_rows) > 0:
                rgb_pixels = image[tissue_rows, tissue_cols]  # (K, 3)
                collected.append(rgb_pixels)
                total_collected += len(rgb_pixels)
            attempt += batch

        if not collected:
            logger.warning("  No tissue pixels found for LAB stats, returning original image")
            del gray
            return image.copy()

        all_pixels = np.concatenate(collected)[:n_samples]
        # Convert to LAB
        px_img = all_pixels.reshape(1, -1, 3).astype(np.uint8)
        px_lab = cv2.cvtColor(px_img, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
        px_lab[:, 0] = px_lab[:, 0] * 100.0 / 255.0
        px_lab[:, 1] -= 128.0
        px_lab[:, 2] -= 128.0

        L_src_median = float(np.median(px_lab[:, 0]))
        L_src_mad = float(np.median(np.abs(px_lab[:, 0] - L_src_median)))
        a_src_median = float(np.median(px_lab[:, 1]))
        a_src_mad = float(np.median(np.abs(px_lab[:, 1] - a_src_median)))
        b_src_median = float(np.median(px_lab[:, 2]))
        b_src_mad = float(np.median(np.abs(px_lab[:, 2] - b_src_median)))
        del all_pixels, px_img, px_lab, collected
        logger.info(f"  Computed slide LAB stats: L={L_src_median:.2f}±{L_src_mad:.2f}")

    if gray is not None:
        del gray

    # Log source vs target stats
    logger.info(f"  Source stats: L={L_src_median:.2f}±{L_src_mad:.2f}, a={a_src_median:.2f}±{a_src_mad:.2f}, b={b_src_median:.2f}±{b_src_mad:.2f}")
    logger.info(f"  Target stats: L={params['L_median']:.2f}±{params['L_mad']:.2f}, a={params['a_median']:.2f}±{params['a_mad']:.2f}, b={params['b_median']:.2f}±{params['b_mad']:.2f}")

    # Step 3: Normalize tissue pixels block-by-block (512x512 for memory efficiency)
    result = image.copy()
    tissue_px_count = 0
    total_px_count = 0

    for by in range(0, h, block_sz):
        for bx in range(0, w, block_sz):
            by_end = min(by + block_sz, h)
            bx_end = min(bx + block_sz, w)

            block_img = result[by:by_end, bx:bx_end]
            block_gray = cv2.cvtColor(block_img, cv2.COLOR_RGB2GRAY)

            # Tissue mask: non-black and below Otsu
            tissue_mask = (block_gray > 0) & (block_gray < otsu_val)
            n_tissue = int(tissue_mask.sum())
            total_px_count += block_gray.size

            if n_tissue == 0:
                continue
            tissue_px_count += n_tissue

            # Convert entire block to LAB
            block_lab = cv2.cvtColor(block_img, cv2.COLOR_RGB2LAB).astype(np.float32)
            block_lab[:, :, 0] = block_lab[:, :, 0] * 100.0 / 255.0
            block_lab[:, :, 1] = block_lab[:, :, 1] - 128.0
            block_lab[:, :, 2] = block_lab[:, :, 2] - 128.0

            # Normalize ONLY tissue pixels
            if L_src_mad > 1e-6:
                block_lab[:, :, 0][tissue_mask] = (block_lab[:, :, 0][tissue_mask] - L_src_median) / L_src_mad * params['L_mad'] + params['L_median']
            else:
                block_lab[:, :, 0][tissue_mask] = params['L_median']

            if a_src_mad > 1e-6:
                block_lab[:, :, 1][tissue_mask] = (block_lab[:, :, 1][tissue_mask] - a_src_median) / a_src_mad * params['a_mad'] + params['a_median']
            else:
                block_lab[:, :, 1][tissue_mask] = params['a_median']

            if b_src_mad > 1e-6:
                block_lab[:, :, 2][tissue_mask] = (block_lab[:, :, 2][tissue_mask] - b_src_median) / b_src_mad * params['b_mad'] + params['b_median']
            else:
                block_lab[:, :, 2][tissue_mask] = params['b_median']

            # Convert back to RGB — only tissue pixels changed, background preserved
            block_lab[:, :, 0] = block_lab[:, :, 0] * 255.0 / 100.0
            block_lab[:, :, 1] = block_lab[:, :, 1] + 128.0
            block_lab[:, :, 2] = block_lab[:, :, 2] + 128.0

            block_lab = np.clip(block_lab, 0, 255).astype(np.uint8)
            block_rgb = cv2.cvtColor(block_lab, cv2.COLOR_LAB2RGB)

            result[by:by_end, bx:bx_end] = block_rgb
            del block_lab, block_rgb

    logger.info(f"  Normalized {tissue_px_count:,} tissue pixels ({100*tissue_px_count/max(total_px_count,1):.1f}% of image)")

    return result


def apply_reinhard_normalization(
    image: np.ndarray,
    params: Dict[str, float],
    otsu_threshold: float = None,
    slide_lab_stats: Dict[str, float] = None,
    **_legacy_kwargs,
) -> np.ndarray:
    """
    Apply Reinhard normalization with pixel-level tissue masking.

    Delegates to apply_reinhard_normalization_MEDIAN.

    Args:
        image: RGB image (H, W, 3) in range [0, 255], dtype uint8
        params: Dictionary with global target Lab statistics
        otsu_threshold: Per-slide Otsu threshold from step 1 JSON
        slide_lab_stats: Per-slide LAB source stats from step 1 JSON
        **_legacy_kwargs: Accepts (and ignores) former dead parameters
            for backward compatibility.

    Returns:
        Normalized RGB image (H, W, 3), dtype uint8
    """
    return apply_reinhard_normalization_MEDIAN(
        image, params,
        otsu_threshold=otsu_threshold,
        slide_lab_stats=slide_lab_stats,
    )
