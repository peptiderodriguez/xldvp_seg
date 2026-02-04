#!/usr/bin/env python3
"""
Validate median-based Reinhard normalization on all 16 slides.
Creates comparison images: Original | Mean-normalized | Median-normalized
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import cv2
from segmentation.io.czi_loader import CZILoader
from segmentation.preprocessing.reinhard_norm import ReinhardNormalizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/viper/ptmp2/edrod/xldvp_seg_fresh")
SLIDE_DIR = Path("/viper/ptmp2/edrod/2025_11_18")
MEAN_PARAMS = BASE_DIR / "reinhard_params_16slides.json"
MEDIAN_PARAMS = BASE_DIR / "reinhard_params_16slides_MEDIAN.json"
OUTPUT_DIR = BASE_DIR / "validation_median_vs_mean"
OUTPUT_DIR.mkdir(exist_ok=True)

# Tile size (same as Phase 1)
TILE_SIZE = 3000

def load_params(params_file):
    """Load parameters from JSON"""
    with open(params_file) as f:
        return json.load(f)

def create_normalizer(params, method="reinhard"):
    """Create normalizer from parameters"""
    normalizer = ReinhardNormalizer()

    if method == "reinhard":
        # Mean-based
        normalizer.target_means = np.array([
            params["L_mean"],
            params["a_mean"],
            params["b_mean"]
        ])
        normalizer.target_stds = np.array([
            params["L_std"],
            params["a_std"],
            params["b_std"]
        ])
    else:
        # Median-based
        normalizer.target_means = np.array([
            params["L_median"],
            params["a_median"],
            params["b_median"]
        ])
        normalizer.target_stds = np.array([
            params["L_mad"],
            params["a_mad"],
            params["b_mad"]
        ])

    return normalizer

def extract_region(slide_path, size):
    """Extract a region with tissue from the slide using tissue detection"""
    from segmentation.io.czi_loader import get_loader
    from segmentation.detection.tissue import calibrate_tissue_threshold, filter_tissue_tiles

    logger.info(f"  Loading slide to RAM...")
    loader = get_loader(str(slide_path), load_to_ram=True, channel=0)
    channel_data = loader.get_channel_data(0)

    if channel_data is None:
        raise ValueError(f"Failed to load channel data")

    # Get shape
    if len(channel_data.shape) == 3:  # RGB
        h, w, c = channel_data.shape
    else:  # Grayscale - convert to RGB
        channel_data = np.stack([channel_data] * 3, axis=-1)
        h, w, c = channel_data.shape

    logger.info(f"  Finding tissue regions...")

    # Create tile grid
    tile_size = 3000
    overlap = int(tile_size * 0.15)
    tiles = []
    step = tile_size - overlap
    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tiles.append({'x': x, 'y': y})

    logger.info(f"  Created {len(tiles)} tiles, calibrating tissue threshold...")

    # Calibrate tissue threshold
    variance_threshold = calibrate_tissue_threshold(
        tiles,
        image_array=channel_data,
        calibration_samples=100,
        block_size=512,
        tile_size=tile_size
    )

    # Filter to tissue tiles
    tissue_tiles = filter_tissue_tiles(
        tiles,
        variance_threshold,
        image_array=channel_data,
        tile_size=tile_size,
        block_size=512,
        n_workers=8,
        show_progress=False
    )

    if len(tissue_tiles) == 0:
        raise ValueError("No tissue tiles found in slide!")

    logger.info(f"  Found {len(tissue_tiles)} tissue tiles")

    # Randomly select up to 4 tissue tiles
    import random
    random.seed(42)  # Reproducible
    n_tiles = min(4, len(tissue_tiles))
    selected_tiles = random.sample(tissue_tiles, n_tiles)

    tiles_and_masks = []
    for i, tile in enumerate(selected_tiles):
        # Extract full tile
        x1 = tile['x']
        y1 = tile['y']
        x2 = x1 + tile_size
        y2 = y1 + tile_size

        logger.info(f"  Tile {i+1}/{n_tiles}: Extracting {tile_size}x{tile_size} at y=[{y1}:{y2}], x=[{x1}:{x2}]")
        tile_data = channel_data[y1:y2, x1:x2, :].copy()

        # Compute pixel-level tissue mask for this tile
        from segmentation.detection.tissue import compute_pixel_level_tissue_mask
        tissue_mask = compute_pixel_level_tissue_mask(
            tile_data,
            variance_threshold,
            block_size=512
        )

        logger.info(f"  Tile {i+1}: {tissue_mask.sum()} / {tissue_mask.size} pixels are tissue ({100*tissue_mask.sum()/tissue_mask.size:.1f}%)")

        tiles_and_masks.append((tile_data, tissue_mask))

    # Free memory
    from segmentation.io.czi_loader import clear_cache
    clear_cache()
    del loader, channel_data
    import gc
    gc.collect()

    return tiles_and_masks

def normalize_median_mad(img, target_medians, target_mads, tissue_mask=None):
    """
    Normalize image using MEDIAN and MAD (both source and target).

    Args:
        img: RGB image (H, W, 3) uint8
        target_medians: Target median values for L, a, b channels
        target_mads: Target MAD values for L, a, b channels
        tissue_mask: Optional boolean mask (H, W) - only normalize masked pixels

    Returns:
        Normalized RGB image uint8
    """
    # Convert to LAB (returns uint8 encoding)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Convert from uint8 encoding to actual LAB scale
    # L: [0, 255] -> [0, 100]
    # a, b: [0, 255] -> [-128, 127] (with 128 as zero)
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0  # L channel
    lab[:, :, 1] = lab[:, :, 1] - 128.0          # a channel
    lab[:, :, 2] = lab[:, :, 2] - 128.0          # b channel

    # Debug: check LAB values before normalization
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"    LAB before norm - L: [{lab[:,:,0].min():.1f}, {lab[:,:,0].max():.1f}], a: [{lab[:,:,1].min():.1f}, {lab[:,:,1].max():.1f}], b: [{lab[:,:,2].min():.1f}, {lab[:,:,2].max():.1f}]")

    # Normalize each channel using median/MAD (only on tissue pixels if mask provided)
    for i in range(3):
        if tissue_mask is not None:
            # Compute stats only on tissue pixels
            tissue_pixels = lab[:, :, i][tissue_mask]
            source_median = np.median(tissue_pixels)
            source_mad = np.median(np.abs(tissue_pixels - source_median))
        else:
            source_median = np.median(lab[:, :, i])
            source_mad = np.median(np.abs(lab[:, :, i] - source_median))

        logger.info(f"    Ch{i}: source median={source_median:.2f}, mad={source_mad:.2f}, target median={target_medians[i]:.2f}, mad={target_mads[i]:.2f}")

        if source_mad > 0:  # Avoid division by zero
            if tissue_mask is not None:
                # Only normalize tissue pixels
                lab[:, :, i][tissue_mask] = ((lab[:, :, i][tissue_mask] - source_median) / source_mad) * target_mads[i] + target_medians[i]
            else:
                # Normalize all pixels
                lab[:, :, i] = ((lab[:, :, i] - source_median) / source_mad) * target_mads[i] + target_medians[i]

    # Debug: check LAB values after normalization
    logger.info(f"    LAB after norm - L: [{lab[:,:,0].min():.1f}, {lab[:,:,0].max():.1f}], a: [{lab[:,:,1].min():.1f}, {lab[:,:,1].max():.1f}], b: [{lab[:,:,2].min():.1f}, {lab[:,:,2].max():.1f}]")

    # Clip to valid LAB range (actual scale)
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)    # L: 0-100
    lab[:, :, 1] = np.clip(lab[:, :, 1], -128, 127) # a: -128 to 127
    lab[:, :, 2] = np.clip(lab[:, :, 2], -128, 127) # b: -128 to 127

    # Convert back from actual LAB scale to uint8 encoding for cvtColor
    lab[:, :, 0] = lab[:, :, 0] * 255.0 / 100.0  # L: [0,100] -> [0,255]
    lab[:, :, 1] = lab[:, :, 1] + 128.0          # a: [-128,127] -> [0,255]
    lab[:, :, 2] = lab[:, :, 2] + 128.0          # b: [-128,127] -> [0,255]

    # Convert back to RGB
    lab = lab.astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb

def normalize_mean_std(img, target_means, target_stds, tissue_mask=None):
    """
    Normalize image using MEAN and STD (standard Reinhard).

    Args:
        img: RGB image (H, W, 3) uint8
        target_means: Target mean values for L, a, b channels
        target_stds: Target std values for L, a, b channels
        tissue_mask: Optional boolean mask (H, W) - only normalize masked pixels

    Returns:
        Normalized RGB image uint8
    """
    # Convert to LAB (returns uint8 encoding)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Convert from uint8 encoding to actual LAB scale
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
    lab[:, :, 1] = lab[:, :, 1] - 128.0          # a: [0,255] -> [-128,127]
    lab[:, :, 2] = lab[:, :, 2] - 128.0          # b: [0,255] -> [-128,127]

    # Normalize each channel using mean/std (only on tissue pixels if mask provided)
    for i in range(3):
        if tissue_mask is not None:
            # Compute stats only on tissue pixels
            tissue_pixels = lab[:, :, i][tissue_mask]
            source_mean = np.mean(tissue_pixels)
            source_std = np.std(tissue_pixels)
        else:
            source_mean = np.mean(lab[:, :, i])
            source_std = np.std(lab[:, :, i])

        if source_std > 0:  # Avoid division by zero
            if tissue_mask is not None:
                # Only normalize tissue pixels
                lab[:, :, i][tissue_mask] = ((lab[:, :, i][tissue_mask] - source_mean) / source_std) * target_stds[i] + target_means[i]
            else:
                # Normalize all pixels
                lab[:, :, i] = ((lab[:, :, i] - source_mean) / source_std) * target_stds[i] + target_means[i]

    # Clip to valid LAB range (actual scale)
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)
    lab[:, :, 1] = np.clip(lab[:, :, 1], -128, 127)
    lab[:, :, 2] = np.clip(lab[:, :, 2], -128, 127)

    # Convert back from actual LAB scale to uint8 encoding
    lab[:, :, 0] = lab[:, :, 0] * 255.0 / 100.0
    lab[:, :, 1] = lab[:, :, 1] + 128.0
    lab[:, :, 2] = lab[:, :, 2] + 128.0

    # Convert back to RGB
    lab = lab.astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb

def create_comparison(original, mean_norm, median_norm):
    """Create side-by-side comparison image"""
    h, w = original.shape[:2]
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = original
    comparison[:, w:2*w] = mean_norm
    comparison[:, 2*w:] = median_norm
    return comparison

def add_labels(img):
    """Add text labels to comparison image"""
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()

    h, w = img.shape[:2]
    labels = ["Original", "Mean-based", "Median-based"]
    positions = [w//6, w//2 + w//6, 5*w//6]

    for label, x_pos in zip(labels, positions):
        # Black background for text
        bbox = draw.textbbox((x_pos, 30), label, font=font, anchor="mt")
        draw.rectangle(bbox, fill="black")
        # White text
        draw.text((x_pos, 30), label, fill="white", font=font, anchor="mt")

    return np.array(pil_img)

def main():
    logger.info("="*70)
    logger.info("VALIDATION: MEDIAN vs MEAN Reinhard Normalization")
    logger.info("="*70)

    # Load parameters
    logger.info("Loading parameters...")
    mean_params = load_params(MEAN_PARAMS)
    median_params = load_params(MEDIAN_PARAMS)

    logger.info(f"Mean-based: L={mean_params['L_mean']:.2f}±{mean_params['L_std']:.2f}")
    logger.info(f"Median-based: L={median_params['L_median']:.2f}±{median_params['L_mad']:.2f}")

    # Create normalizers
    logger.info("Creating normalizers...")
    mean_normalizer = create_normalizer(mean_params, method="reinhard")
    median_normalizer = create_normalizer(median_params, method="reinhard_median")

    # Process all 16 slides
    slides = sorted(SLIDE_DIR.glob("*.czi"))
    logger.info(f"Found {len(slides)} slides")

    for i, slide_path in enumerate(slides, 1):
        logger.info(f"\n[{i}/{len(slides)}] Processing {slide_path.name}...")

        try:
            # Extract multiple tissue tiles with masks
            tiles_and_masks = extract_region(slide_path, TILE_SIZE)

            # Process all tiles and collect comparisons
            tile_comparisons = []
            for tile_idx, (original, tissue_mask) in enumerate(tiles_and_masks, 1):
                logger.info(f"  Tile {tile_idx}/{len(tiles_and_masks)}: min={original.min()}, max={original.max()}, mean={original.mean():.2f}")

                # Normalize with both methods (only tissue pixels)
                target_means = np.array([mean_params['L_mean'], mean_params['a_mean'], mean_params['b_mean']])
                target_stds = np.array([mean_params['L_std'], mean_params['a_std'], mean_params['b_std']])
                mean_norm = normalize_mean_std(original, target_means, target_stds, tissue_mask)

                target_medians = np.array([median_params['L_median'], median_params['a_median'], median_params['b_median']])
                target_mads = np.array([median_params['L_mad'], median_params['a_mad'], median_params['b_mad']])
                median_norm = normalize_median_mad(original, target_medians, target_mads, tissue_mask)

                # Create comparison for this tile
                comparison = create_comparison(original, mean_norm, median_norm)
                tile_comparisons.append(comparison)

                del original, tissue_mask, mean_norm, median_norm, comparison

            # Stack all tile comparisons vertically
            logger.info(f"  Stacking {len(tile_comparisons)} tiles into composite image...")

            # Add labels to first row only
            tile_comparisons[0] = add_labels(tile_comparisons[0])

            # Stack all tiles
            composite = np.vstack(tile_comparisons)

            # Save composite
            output_file = OUTPUT_DIR / f"{slide_path.stem}_4tiles_comparison.png"
            Image.fromarray(composite).save(output_file, quality=95)
            logger.info(f"  Saved: {output_file}")

            # Free memory
            del tile_comparisons, composite
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"  ERROR processing {slide_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    logger.info("\n" + "="*70)
    logger.info("VALIDATION COMPLETE!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
