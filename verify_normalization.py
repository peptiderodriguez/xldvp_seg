#!/usr/bin/env python3
"""
Standalone verification script for Reinhard normalization.

Loads a CZI slide, applies normalization (which now uses Phase 1's tissue
detection method), then re-samples tissue pixels using the SAME method
to verify output statistics match targets.

Usage:
    python verify_normalization.py /path/to/slide.czi --params reinhard_params.json
"""

import sys
import os
import json
import argparse
import logging
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation.io.czi_loader import get_loader
from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN
from segmentation.detection.tissue import calibrate_tissue_threshold, filter_tissue_tiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def sample_tissue_stats_phase1_method(image, block_sz=512, n_samples=500000):
    """
    Sample tissue stats using Phase 1's exact method:
    block-level variance detection, random sampling from tissue blocks, cv2 LAB.
    """
    h, w = image.shape[:2]

    blocks = []
    for y in range(0, h, block_sz):
        for x in range(0, w, block_sz):
            blocks.append({'x': x, 'y': y})

    # Calibrate threshold (same as Phase 1)
    var_threshold = calibrate_tissue_threshold(
        blocks,
        image_array=image,
        calibration_samples=min(100, len(blocks)),
        block_size=block_sz,
        tile_size=block_sz
    )
    logger.info(f"  Calibrated threshold: {var_threshold:.1f}, reduced 10x: {var_threshold/10:.1f}")
    var_threshold /= 10.0

    # Filter tissue blocks
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

    if not tissue_blocks:
        logger.warning("  No tissue blocks found!")
        return None

    # Vectorized sampling from tissue blocks
    block_origins = np.array([(b['x'], b['y']) for b in tissue_blocks])
    block_indices = np.random.randint(0, len(tissue_blocks), size=n_samples)
    selected_origins = block_origins[block_indices]
    x_offsets = np.random.randint(0, block_sz, size=n_samples)
    y_offsets = np.random.randint(0, block_sz, size=n_samples)
    xs = np.clip(selected_origins[:, 0] + x_offsets, 0, w - 1)
    ys = np.clip(selected_origins[:, 1] + y_offsets, 0, h - 1)
    samples = image[ys, xs]  # (n_samples, 3) uint8

    # Convert to LAB using cv2 (same as Phase 1)
    samples_img = samples.reshape(1, -1, 3)
    lab = cv2.cvtColor(samples_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = lab.reshape(-1, 3)
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0
    lab[:, 1] = lab[:, 1] - 128.0
    lab[:, 2] = lab[:, 2] - 128.0

    L_med = np.median(lab[:, 0])
    L_mad = np.median(np.abs(lab[:, 0] - L_med))
    a_med = np.median(lab[:, 1])
    a_mad = np.median(np.abs(lab[:, 1] - a_med))
    b_med = np.median(lab[:, 2])
    b_mad = np.median(np.abs(lab[:, 2] - b_med))

    return {
        'L_median': float(L_med), 'L_mad': float(L_mad),
        'a_median': float(a_med), 'a_mad': float(a_mad),
        'b_median': float(b_med), 'b_mad': float(b_mad),
        'n_tissue_blocks': len(tissue_blocks),
        'n_samples': n_samples,
    }


def verify_slide(czi_path, params_file):
    with open(params_file, 'r') as f:
        params = json.load(f)

    slide_name = os.path.basename(czi_path)
    logger.info(f"{'='*70}")
    logger.info(f"VERIFYING: {slide_name}")
    logger.info(f"{'='*70}")

    # Load slide
    logger.info(f"Loading slide into RAM...")
    loader = get_loader(czi_path, load_to_ram=True, channel=0)
    h, w = loader.channel_data.shape[:2]
    logger.info(f"  Loaded: {w} x {h}, {loader.channel_data.nbytes/(1024**3):.2f} GB")

    # PRE-NORMALIZATION
    logger.info(f"\n--- PRE-NORMALIZATION ---")
    np.random.seed(42)
    pre = sample_tissue_stats_phase1_method(loader.channel_data)
    if pre:
        log_stats("PRE", pre, params)

    # NORMALIZE
    logger.info(f"\n--- APPLYING NORMALIZATION ---")
    logger.info(f"  Target: L={params['L_median']:.2f}±{params['L_mad']:.2f}, "
                f"a={params['a_median']:.2f}±{params['a_mad']:.2f}, "
                f"b={params['b_median']:.2f}±{params['b_mad']:.2f}")

    normalized = apply_reinhard_normalization_MEDIAN(loader.channel_data, params)
    loader.channel_data = normalized
    del normalized
    import gc; gc.collect()

    # POST-NORMALIZATION
    logger.info(f"\n--- POST-NORMALIZATION ---")
    np.random.seed(42)  # Same seed for comparable sampling
    post = sample_tissue_stats_phase1_method(loader.channel_data)
    if post:
        log_stats("POST", post, params)

    loader.close()
    import gc; gc.collect()

    return {'slide': slide_name, 'pre': pre, 'post': post, 'targets': params}


def log_stats(label, stats, targets):
    logger.info(f"  {label} ({stats['n_samples']:,} samples from {stats['n_tissue_blocks']} tissue blocks):")
    for ch in ['L', 'a', 'b']:
        med = stats[f'{ch}_median']
        mad = stats[f'{ch}_mad']
        t_med = targets[f'{ch}_median']
        t_mad = targets[f'{ch}_mad']
        logger.info(f"    {ch}: median={med:.2f} (target {t_med:.2f}, diff {med-t_med:+.2f}), "
                    f"MAD={mad:.2f} (target {t_mad:.2f}, diff {mad-t_mad:+.2f})")


def main():
    parser = argparse.ArgumentParser(description='Verify Reinhard normalization output')
    parser.add_argument('czi_paths', nargs='+', help='Path(s) to CZI slide(s)')
    parser.add_argument('--params', required=True, help='Path to Reinhard params JSON')
    args = parser.parse_args()

    for czi_path in args.czi_paths:
        if not os.path.exists(czi_path):
            logger.error(f"File not found: {czi_path}")
            continue
        verify_slide(czi_path, args.params)


if __name__ == '__main__':
    main()
