#!/usr/bin/env python3
"""
Compute Reinhard normalization parameters from 8 slides using tissue-aware sampling.

Samples 500k pixels per slide from tissue-containing tiles only (not random from whole slide).
This provides better representative statistics for normalization.
"""

import numpy as np
import json
from pathlib import Path
from segmentation.io.czi_loader import get_loader
from segmentation.utils.logging import setup_logging, get_logger
from segmentation.preprocessing.stain_normalization import compute_reinhard_params_from_samples
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    filter_tissue_tiles,
    compute_pixel_level_tissue_mask
)

setup_logging()
logger = get_logger(__name__)


def create_tile_grid(width, height, tile_size=3000, overlap=0):
    """Create a grid of tile coordinates."""
    tiles = []
    step = tile_size - overlap
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            tiles.append({'x': x, 'y': y})
    return tiles


def sample_pixels_from_tissue_tiles(czi_path, channel=0, n_samples=500000, tile_size=3000, block_size=512):
    """
    Sample random pixels from tissue-containing tiles only.

    Args:
        czi_path: Path to CZI file
        channel: Channel index
        n_samples: Total number of pixels to sample
        tile_size: Tile size for tissue detection
        block_size: Block size for variance calculation

    Returns:
        RGB array of sampled pixels (N, 3) or None on failure
    """
    logger.info(f"Processing {czi_path.name}...")

    try:
        # Load slide to RAM
        logger.info("  Loading slide to RAM...")
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
            # Convert grayscale to RGB for consistency
            channel_data = np.stack([channel_data] * 3, axis=-1)
            h, w, c = channel_data.shape

        logger.info(f"  Shape: {channel_data.shape}, RGB: {is_rgb}")

        # Create tile grid with 15% overlap
        # Overlap allows picking most complete mask when cells split across tiles
        overlap = int(tile_size * 0.15)  # 15% overlap = 450 pixels for 3000px tiles
        logger.info(f"  Creating tile grid (tile_size={tile_size}, overlap={overlap})...")
        tiles = create_tile_grid(w, h, tile_size=tile_size, overlap=overlap)
        logger.info(f"  Total tiles: {len(tiles)}")

        # Calibrate tissue threshold using K-means
        logger.info("  Calibrating tissue threshold...")
        variance_threshold = calibrate_tissue_threshold(
            tiles,
            image_array=channel_data,
            calibration_samples=100,
            block_size=block_size,
            tile_size=tile_size
        )

        # Filter to tissue tiles
        logger.info("  Filtering to tissue tiles...")
        tissue_tiles = filter_tissue_tiles(
            tiles,
            variance_threshold,
            image_array=channel_data,
            tile_size=tile_size,
            block_size=block_size,
            n_workers=8,
            show_progress=True
        )

        if len(tissue_tiles) == 0:
            logger.warning(f"  No tissue tiles found in {czi_path.name}")
            return None

        logger.info(f"  Tissue tiles: {len(tissue_tiles)} / {len(tiles)} ({100*len(tissue_tiles)/len(tiles):.1f}%)")

        # Sample pixels from tissue tiles only
        logger.info(f"  Sampling {n_samples:,} pixels from tissue tiles...")
        samples = []

        # Calculate samples per tile (distribute evenly)
        samples_per_tile = max(1, n_samples // len(tissue_tiles))

        for tile in tissue_tiles:
            tile_x, tile_y = tile['x'], tile['y']

            # Extract tile from channel_data
            tile_img = channel_data[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]

            # Skip if tile is too small
            if tile_img.shape[0] < tile_size // 2 or tile_img.shape[1] < tile_size // 2:
                continue

            # Compute pixel-level tissue mask for this tile
            # Only sample from tissue pixels, not background
            tissue_mask = compute_pixel_level_tissue_mask(
                tile_img,
                variance_threshold,
                block_size=7
            )

            # Get coordinates of tissue pixels
            tissue_coords = np.argwhere(tissue_mask)  # Returns (N, 2) array of (y, x)

            if len(tissue_coords) == 0:
                # No tissue pixels in this tile (shouldn't happen if tile passed detection)
                logger.warning(f"    Tile at ({tile_x}, {tile_y}) has no tissue pixels, skipping")
                continue

            # Sample from tissue pixels only
            n_tile_samples = min(samples_per_tile, len(tissue_coords))

            if len(tissue_coords) > n_tile_samples:
                # Randomly select tissue pixel indices
                sampled_indices = np.random.choice(len(tissue_coords), n_tile_samples, replace=False)
                sampled_coords = tissue_coords[sampled_indices]
            else:
                # Use all tissue pixels
                sampled_coords = tissue_coords

            # Extract RGB values at sampled tissue pixel coordinates
            tile_samples = tile_img[sampled_coords[:, 0], sampled_coords[:, 1], :].copy()

            samples.append(tile_samples)

            # Stop if we have enough samples
            if sum(len(s) for s in samples) >= n_samples:
                break

        # Check if we collected any samples
        if len(samples) == 0:
            logger.warning(f"  No valid samples collected from tissue tiles in {czi_path.name}")
            loader.close()
            del loader, channel_data, tiles, tissue_tiles
            import gc
            gc.collect()
            return None

        # Combine all samples
        all_samples = np.vstack(samples)

        # Trim to exact n_samples if we over-sampled
        if len(all_samples) > n_samples:
            all_samples = all_samples[:n_samples]

        logger.info(f"  Collected {len(all_samples):,} pixels (mean intensity: {all_samples.mean():.1f})")

        # CRITICAL: Free memory immediately after sampling
        loader.close()
        del loader, channel_data, tiles, tissue_tiles, samples
        import gc
        gc.collect()

        return all_samples

    except Exception as e:
        logger.error(f"  Failed to sample from {czi_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    logger.info("="*70)
    logger.info("COMPUTING REINHARD NORMALIZATION PARAMETERS (8 SLIDES)")
    logger.info("="*70)

    # All 16 slides: complete cohort for robust reference statistics
    slide_names = [
        "2025_11_18_FGC1.czi",
        "2025_11_18_FGC2.czi",
        "2025_11_18_FGC3.czi",
        "2025_11_18_FGC4.czi",
        "2025_11_18_FHU1.czi",
        "2025_11_18_FHU2.czi",
        "2025_11_18_FHU3.czi",
        "2025_11_18_FHU4.czi",
        "2025_11_18_MGC1.czi",
        "2025_11_18_MGC2.czi",
        "2025_11_18_MGC3.czi",
        "2025_11_18_MGC4.czi",
        "2025_11_18_MHU1.czi",
        "2025_11_18_MHU2.czi",
        "2025_11_18_MHU3.czi",
        "2025_11_18_MHU4.czi",
    ]

    czi_dir = Path("/viper/ptmp2/edrod/2025_11_18")
    slides = [czi_dir / name for name in slide_names]

    # Verify all slides exist
    missing = [s for s in slides if not s.exists()]
    if missing:
        logger.error(f"Missing slides: {[s.name for s in missing]}")
        return

    logger.info(f"Processing {len(slides)} slides:")
    for s in slides:
        logger.info(f"  - {s.name}")
    logger.info("")

    # Sample from all slides
    np.random.seed(42)  # Reproducible sampling
    all_samples = []

    for czi_path in slides:
        samples = sample_pixels_from_tissue_tiles(
            czi_path,
            channel=0,
            n_samples=500000,
            tile_size=3000,
            block_size=512
        )

        if samples is not None:
            all_samples.append(samples)

        # Force garbage collection after each slide
        import gc
        gc.collect()
        logger.info("")

    if len(all_samples) == 0:
        logger.error("No samples collected!")
        return

    # Compute Reinhard parameters in Lab color space
    logger.info("="*70)
    logger.info("COMPUTING REINHARD PARAMETERS (Lab color space)...")
    params = compute_reinhard_params_from_samples(all_samples)

    # Add metadata
    params['method'] = 'reinhard'
    params['slides'] = slide_names
    params['samples_per_slide'] = 500000
    params['sampling_method'] = 'tissue_aware'
    params['tile_size'] = 3000
    params['block_size'] = 512

    # Save parameters
    output_file = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides.json")
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)

    logger.info("="*70)
    logger.info("REINHARD NORMALIZATION PARAMETERS (Lab color space):")
    logger.info(f"  L channel: mean={params['L_mean']:.2f}, std={params['L_std']:.2f}")
    logger.info(f"  a channel: mean={params['a_mean']:.2f}, std={params['a_std']:.2f}")
    logger.info(f"  b channel: mean={params['b_mean']:.2f}, std={params['b_std']:.2f}")
    logger.info(f"  Total pixels: {params['n_total_pixels']:,}")
    logger.info(f"  Slides: {params['n_slides']}")
    logger.info(f"\nSaved to: {output_file}")
    logger.info("="*70)

    logger.info("\nNext step: Launch parallel segmentation jobs with these parameters")


if __name__ == "__main__":
    main()
