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
    """Sample random pixels from TISSUE REGIONS only (using variance-based detection)."""
    logger.info(f"Sampling from {czi_path.name}...")

    try:
        from segmentation.detection.tissue import calibrate_tissue_threshold, filter_tissue_tiles

        loader = get_loader(str(czi_path), load_to_ram=True, channel=channel)
        channel_data = loader.get_channel_data(channel)

        if channel_data is None:
            logger.warning(f"  No data loaded for {czi_path.name}")
            return None

        # Ensure RGB
        if len(channel_data.shape) == 2:
            channel_data = np.stack([channel_data] * 3, axis=-1)

        h, w, c = channel_data.shape
        logger.info(f"  Shape: {channel_data.shape}")

        # Use proper tissue detection at block level (fast)
        logger.info(f"  Detecting tissue blocks using variance thresholding...")

        block_size = 512
        blocks = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                blocks.append({'x': x, 'y': y})

        logger.info(f"  Created {len(blocks)} blocks, calibrating tissue threshold...")

        # Calibrate tissue threshold
        variance_threshold = calibrate_tissue_threshold(
            blocks,
            image_array=channel_data,
            calibration_samples=min(100, len(blocks)),
            block_size=block_size,
            tile_size=block_size
        )

        logger.info(f"  Tissue variance threshold (computed): {variance_threshold:.1f}")

        # Reduce threshold by 10x to detect more tissue (especially lighter-stained regions)
        variance_threshold = variance_threshold / 10.0
        logger.info(f"  Tissue variance threshold (reduced 10x): {variance_threshold:.1f}")
        logger.info(f"  Filtering to tissue blocks...")

        # Filter to tissue blocks only
        tissue_blocks = filter_tissue_tiles(
            blocks,
            variance_threshold,
            image_array=channel_data,
            tile_size=block_size,
            block_size=block_size,
            n_workers=8,
            show_progress=False
        )

        if len(tissue_blocks) == 0:
            logger.warning(f"  No tissue blocks found in {czi_path.name}!")
            return None

        logger.info(f"  Found {len(tissue_blocks)} tissue blocks ({100*len(tissue_blocks)/len(blocks):.1f}%)")
        logger.info(f"  Sampling {n_samples} pixels from tissue blocks...")

        # Sample random pixels from tissue blocks only (no per-pixel checks needed!)
        tissue_samples = []

        for _ in range(n_samples):
            # Pick random tissue block
            block = tissue_blocks[np.random.randint(len(tissue_blocks))]

            # Pick random pixel within that block
            y = block['y'] + np.random.randint(0, min(block_size, h - block['y']))
            x = block['x'] + np.random.randint(0, min(block_size, w - block['x']))

            tissue_samples.append(channel_data[y, x, :])

        samples = np.array(tissue_samples)  # (N, 3)
        logger.info(f"  Sampled {len(samples)} tissue pixels, mean intensity: {samples.mean():.1f}")

        # Close and clear to prevent memory accumulation across slides
        loader.close()
        from segmentation.io.czi_loader import clear_cache
        clear_cache()
        del loader, channel_data
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

    # Compute global median/MAD for Reinhard normalization
    import cv2
    logger.info("")
    logger.info("="*70)
    logger.info("Computing global median/MAD for Reinhard normalization...")

    # Stack all samples
    combined = np.vstack(all_samples)  # (N, 3) RGB
    logger.info(f"Total samples: {len(combined):,}")

    # Convert to LAB
    tissue_img = combined.reshape(1, -1, 3).astype(np.uint8)
    lab = cv2.cvtColor(tissue_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = lab.reshape(-1, 3)  # Back to (N, 3)

    # Convert to actual LAB scale
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
    lab[:, 1] = lab[:, 1] - 128.0          # a: [0,255] -> [-128,127]
    lab[:, 2] = lab[:, 2] - 128.0          # b: [0,255] -> [-128,127]

    # Compute median and MAD for each channel
    medians = np.median(lab, axis=0)
    mads = np.median(np.abs(lab - medians), axis=0)

    # Save parameters in Reinhard format
    params = {
        'L_median': float(medians[0]),
        'L_mad': float(mads[0]),
        'a_median': float(medians[1]),
        'a_mad': float(mads[1]),
        'b_median': float(medians[2]),
        'b_mad': float(mads[2]),
        'n_slides': len(all_samples),
        'n_total_pixels': len(combined),
        'method': 'reinhard_median',
        'slides': [s.name for s in slides],
        'samples_per_slide': 500000,
        'sampling_method': 'tissue_aware_10x_lower_threshold',
        'tile_size': 3000,
        'block_size': 512
    }

    output_file = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides_MEDIAN_NEW.json")
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)

    logger.info("="*70)
    logger.info("GLOBAL REINHARD NORMALIZATION PARAMETERS:")
    logger.info(f"  L: median={medians[0]:.2f}, MAD={mads[0]:.2f}")
    logger.info(f"  a: median={medians[1]:.2f}, MAD={mads[1]:.2f}")
    logger.info(f"  b: median={medians[2]:.2f}, MAD={mads[2]:.2f}")
    logger.info("="*70)
    logger.info(f"\nSaved to: {output_file}")
    logger.info("="*70)

    logger.info("\nNext step: Launch 8 parallel jobs with --norm-params-file flag")

if __name__ == "__main__":
    main()
