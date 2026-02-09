#!/usr/bin/env python3
"""
Compute consensus stain normalization parameters from all slides.

Samples tissue patches from all slides, fits a Macenko normalizer on the
pooled data to learn the average stain characteristics, and saves the
normalizer for later tile-by-tile application.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Tuple
from aicspylibczi import CziFile
from skimage import filters, morphology
import argparse
from segmentation.preprocessing.reinhard_norm import ReinhardNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_slide_thumbnail(czi_path: str, target_size: int = 4000) -> np.ndarray:
    """Load a downsampled version of the slide for patch sampling."""
    logger.info(f"Loading thumbnail from {Path(czi_path).name}...")

    czi = CziFile(czi_path)

    # Load at reduced resolution
    # scale_factor=0.1 gives ~10k pixels for 100k full res
    img_data = czi.read_mosaic(C=0, scale_factor=0.05)  # 5% of full resolution

    # Extract RGB from shape (H, W, C) or (H, W, C, 1)
    if img_data.ndim == 4:
        # Shape: (H, W, C, 1) -> (H, W, C)
        img_rgb = img_data[:, :, :3, 0]
    elif img_data.ndim == 3:
        # Shape: (H, W, C) -> (H, W, C)
        img_rgb = img_data[:, :, :3]
    else:
        raise ValueError(f"Unexpected image shape: {img_data.shape}")

    logger.info(f"  Loaded thumbnail: {img_rgb.shape}")
    return img_rgb


def detect_tissue_mask(rgb_image: np.ndarray) -> np.ndarray:
    """Create binary tissue mask using Otsu thresholding on grayscale."""
    # Convert to grayscale
    gray = np.mean(rgb_image, axis=2).astype(np.uint8)

    # Otsu threshold (tissue is darker than background for H&E)
    threshold_val = filters.threshold_otsu(gray)
    tissue_mask = gray < threshold_val

    # Clean up mask (remove small noise, fill holes)
    tissue_mask = morphology.remove_small_holes(tissue_mask, area_threshold=1000)
    tissue_mask = morphology.closing(tissue_mask, morphology.disk(5))

    return tissue_mask


def sample_tissue_patches(
    rgb_image: np.ndarray,
    tissue_mask: np.ndarray,
    n_patches: int = 50,
    patch_size: int = 256
) -> List[np.ndarray]:
    """Sample random tissue patches from the image."""
    h, w = tissue_mask.shape
    patches = []

    # Find tissue coordinates
    tissue_coords = np.argwhere(tissue_mask)

    if len(tissue_coords) == 0:
        logger.warning("No tissue detected in slide!")
        return []

    # Sample random tissue locations
    n_attempts = n_patches * 10  # Try multiple times
    for _ in range(n_attempts):
        if len(patches) >= n_patches:
            break

        # Random tissue pixel
        idx = np.random.randint(0, len(tissue_coords))
        y, x = tissue_coords[idx]

        # Extract patch centered at this location
        y_start = max(0, y - patch_size // 2)
        x_start = max(0, x - patch_size // 2)
        y_end = min(h, y_start + patch_size)
        x_end = min(w, x_start + patch_size)

        # Skip if patch too small
        if (y_end - y_start) < patch_size // 2 or (x_end - x_start) < patch_size // 2:
            continue

        patch = rgb_image[y_start:y_end, x_start:x_end, :]

        # Check if patch has enough tissue (>50%)
        patch_mask = tissue_mask[y_start:y_end, x_start:x_end]
        tissue_fraction = np.mean(patch_mask)

        if tissue_fraction > 0.5:
            patches.append(patch)

    logger.info(f"  Sampled {len(patches)} patches")
    return patches


def main():
    parser = argparse.ArgumentParser(description='Compute consensus stain normalization parameters')
    parser.add_argument('--slides', nargs='+', required=True, help='List of CZI slide paths')
    parser.add_argument('--output', required=True, help='Output pickle file for normalizer')
    parser.add_argument('--n-patches', type=int, default=50, help='Patches per slide (default: 50)')
    parser.add_argument('--patch-size', type=int, default=256, help='Patch size (default: 256)')
    parser.add_argument('--method', default='reinhard', choices=['reinhard'],
                        help='Normalization method (default: reinhard)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("COMPUTING CONSENSUS STAIN NORMALIZATION PARAMETERS")
    logger.info("=" * 80)
    logger.info(f"Number of slides: {len(args.slides)}")
    logger.info(f"Patches per slide: {args.n_patches}")
    logger.info(f"Method: {args.method}")
    logger.info("")

    all_patches = []

    # Sample patches from each slide
    for i, slide_path in enumerate(args.slides, 1):
        logger.info(f"[{i}/{len(args.slides)}] Processing {Path(slide_path).name}")

        try:
            # Load thumbnail
            thumbnail = load_slide_thumbnail(slide_path, target_size=2000)

            # Detect tissue
            tissue_mask = detect_tissue_mask(thumbnail)
            tissue_fraction = np.mean(tissue_mask)
            logger.info(f"  Tissue coverage: {tissue_fraction*100:.1f}%")

            # Sample patches
            patches = sample_tissue_patches(
                thumbnail,
                tissue_mask,
                n_patches=args.n_patches,
                patch_size=args.patch_size
            )

            all_patches.extend(patches)

        except Exception as e:
            logger.error(f"  Failed to process slide: {e}")
            continue

    logger.info("")
    logger.info(f"Total patches collected: {len(all_patches)}")

    if len(all_patches) == 0:
        logger.error("No patches collected! Cannot fit normalizer.")
        return 1

    # Standardize patch sizes by resizing to common size
    logger.info("Standardizing patch sizes...")
    target_size = args.patch_size
    standardized_patches = []
    for patch in all_patches:
        if patch.shape[0] != target_size or patch.shape[1] != target_size:
            import cv2
            patch = cv2.resize(patch, (target_size, target_size))
        standardized_patches.append(patch)

    # Stack patches into single array for fitting
    logger.info("Pooling patches...")
    pooled_data = np.vstack([p.reshape(-1, 3) for p in standardized_patches])
    logger.info(f"Pooled data shape: {pooled_data.shape}")

    # Fit Reinhard normalizer on consensus data
    logger.info("")
    logger.info(f"Fitting {args.method.upper()} normalizer on pooled data...")

    try:
        normalizer = ReinhardNormalizer()

        # Fit on all standardized patches to get consensus statistics
        logger.info(f"  Computing consensus LAB statistics from {len(standardized_patches)} patches...")
        normalizer.fit_batch(standardized_patches)

        logger.info(f"  Target LAB means: {normalizer.target_means}")
        logger.info(f"  Target LAB stds:  {normalizer.target_stds}")
        logger.info("  Normalizer fitted successfully!")

        # Save normalizer
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        normalizer.save(output_path)

        logger.info(f"Saved normalizer to: {output_path}")
        logger.info("")
        logger.info("=" * 80)
        logger.info("DONE! Use this normalizer for tile-by-tile normalization.")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Failed to fit normalizer: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
