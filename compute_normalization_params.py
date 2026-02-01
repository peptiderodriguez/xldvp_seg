#!/usr/bin/env python3
"""
Compute global normalization parameters from all 16 slides.
Saves parameters to JSON for use in parallel segmentation jobs.
"""

import numpy as np
import json
from pathlib import Path
from segmentation.io.czi_loader import get_loader
from segmentation.utils.logging import setup_logging, get_logger
from segmentation.preprocessing.stain_normalization import compute_global_percentiles

setup_logging()
logger = get_logger(__name__)

def sample_pixels_from_slide(czi_path, channel=0, n_samples=500000):
    """Sample random pixels from a slide."""
    logger.info(f"Sampling from {czi_path.name}...")

    try:
        loader = get_loader(str(czi_path), load_to_ram=True, channel=channel)
        channel_data = loader.get_channel_data(channel)

        if channel_data is None:
            logger.warning(f"  No data loaded for {czi_path.name}")
            return None

        # Get shape
        if len(channel_data.shape) == 3:  # RGB
            h, w, c = channel_data.shape
            is_rgb = (c == 3)
        else:
            h, w = channel_data.shape
            is_rgb = False

        logger.info(f"  Shape: {channel_data.shape}, RGB: {is_rgb}")

        # Sample random pixels efficiently (without flattening)
        n_pixels = h * w
        n_sample = min(n_samples, n_pixels)

        # Generate random 2D coordinates directly (with replacement)
        # For 500k samples from billions of pixels, duplicates are negligible (~0.016%)
        # This is MUCH faster than np.random.choice with replace=False on billions of elements
        row_indices = np.random.randint(0, h, size=n_sample)
        col_indices = np.random.randint(0, w, size=n_sample)

        # Index directly into the array (much faster than flattening entire array)
        if is_rgb:
            samples = channel_data[row_indices, col_indices, :].copy()  # Shape: (n_sample, 3)
        else:
            samples = channel_data[row_indices, col_indices].copy()  # Shape: (n_sample,)

        mean_intensity = samples.mean()
        logger.info(f"  Mean intensity: {mean_intensity:.1f}")

        # CRITICAL: Free memory immediately after sampling
        loader.close()  # Release all loader resources including internal data cache
        del loader, channel_data, row_indices, col_indices
        import gc
        gc.collect()

        return samples

    except Exception as e:
        logger.error(f"  Failed to sample from {czi_path.name}: {e}")
        return None

def compute_percentiles_from_samples(all_samples, p_low=1.0, p_high=99.0):
    """
    Compute global percentiles from pre-sampled pixel data.

    This is a wrapper around the module's compute_global_percentiles() that
    works with already-sampled data (for memory-efficient processing).
    """
    # Stack all samples
    combined = np.vstack(all_samples)
    logger.info(f"Computing percentiles from {len(combined):,} total samples...")

    # Compute percentiles per channel (using optimized single-pass percentile)
    if combined.ndim == 2 and combined.shape[1] == 3:  # RGB
        low_vals, high_vals = np.percentile(combined, [p_low, p_high], axis=0)
    else:
        low_vals, high_vals = np.percentile(combined, [p_low, p_high])

    return low_vals, high_vals

def main():
    logger.info("="*70)
    logger.info("COMPUTING GLOBAL NORMALIZATION PARAMETERS")
    logger.info("="*70)

    czi_dir = Path("/viper/ptmp2/edrod/2025_11_18")
    slides = sorted(czi_dir.glob("2025_11_18_*.czi"))

    logger.info(f"Found {len(slides)} slides")

    # Sample from all slides
    np.random.seed(42)  # Reproducible sampling
    all_samples = []

    for czi_path in slides:
        samples = sample_pixels_from_slide(czi_path, channel=0, n_samples=500000)
        if samples is not None:
            all_samples.append(samples)

        # Force garbage collection after each slide to prevent memory accumulation
        import gc
        gc.collect()

    if len(all_samples) == 0:
        logger.error("No samples collected!")
        return

    # Compute global percentiles
    logger.info("")
    logger.info("="*70)
    target_low, target_high = compute_percentiles_from_samples(all_samples, p_low=1.0, p_high=99.0)

    # Save parameters
    params = {
        'n_slides': len(all_samples),
        'p_low': 1.0,
        'p_high': 99.0,
        'target_low': target_low.tolist() if hasattr(target_low, 'tolist') else [target_low],
        'target_high': target_high.tolist() if hasattr(target_high, 'tolist') else [target_high],
        'slides': [s.name for s in slides]
    }

    output_file = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/normalization_params_all16.json")
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)

    logger.info("="*70)
    logger.info("GLOBAL NORMALIZATION PARAMETERS:")
    if isinstance(target_low, np.ndarray) and len(target_low) == 3:
        logger.info(f"  R: [{target_low[0]:.1f}, {target_high[0]:.1f}]")
        logger.info(f"  G: [{target_low[1]:.1f}, {target_high[1]:.1f}]")
        logger.info(f"  B: [{target_low[2]:.1f}, {target_high[2]:.1f}]")
    else:
        logger.info(f"  Range: [{target_low:.1f}, {target_high:.1f}]")
    logger.info(f"\nSaved to: {output_file}")
    logger.info("="*70)

    logger.info("\nNext step: Launch 8 parallel jobs with --norm-params-file flag")

if __name__ == "__main__":
    main()
