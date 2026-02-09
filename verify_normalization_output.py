#!/usr/bin/env python3
"""
Verify normalization output by computing LAB statistics on normalized slide.
Samples tissue regions at full resolution instead of downsampling.
"""
import numpy as np
from skimage import color
from scipy.ndimage import uniform_filter
import json
import sys
from segmentation.io.czi_loader import get_loader

def compute_tissue_lab_stats(image, variance_threshold=50.0, n_samples=100, patch_size=512):
    """
    Compute LAB statistics by sampling tissue patches at full resolution.

    Args:
        image: RGB image (H, W, 3), uint8
        variance_threshold: Variance threshold for tissue detection
        n_samples: Number of patches to sample
        patch_size: Size of each patch

    Returns:
        dict with L/a/b median and MAD statistics
    """
    h, w = image.shape[:2]

    # Find tissue regions
    print(f"  Detecting tissue regions...")
    gray = np.mean(image, axis=2).astype(np.float32)
    local_var = uniform_filter(gray, size=7)**2 - uniform_filter(gray**2, size=7)

    # Get coordinates of tissue pixels
    tissue_coords = np.argwhere(local_var > variance_threshold)

    if len(tissue_coords) < 1000:
        print(f"  WARNING: Only {len(tissue_coords)} tissue pixels found!")
        return None

    print(f"  Found {len(tissue_coords):,} tissue pixels")

    # Sample random patches centered on tissue
    all_lab_pixels = []

    for i in range(min(n_samples, len(tissue_coords))):
        # Pick random tissue pixel as center
        center_y, center_x = tissue_coords[np.random.randint(len(tissue_coords))]

        # Extract patch
        y0 = max(0, center_y - patch_size // 2)
        y1 = min(h, center_y + patch_size // 2)
        x0 = max(0, center_x - patch_size // 2)
        x1 = min(w, center_x + patch_size // 2)

        patch = image[y0:y1, x0:x1, :]

        # Convert to LAB
        if patch.dtype == np.uint8:
            patch_float = patch.astype(np.float32) / 255.0
        else:
            patch_float = patch.astype(np.float32)

        patch_lab = color.rgb2lab(patch_float)

        # Detect tissue in this patch
        patch_gray = np.mean(patch, axis=2).astype(np.float32)
        patch_var = uniform_filter(patch_gray, size=7)**2 - uniform_filter(patch_gray**2, size=7)
        patch_tissue_mask = patch_var > variance_threshold

        if np.sum(patch_tissue_mask) > 10:
            tissue_lab_pixels = patch_lab[patch_tissue_mask]
            all_lab_pixels.append(tissue_lab_pixels)

    if not all_lab_pixels:
        print("  ERROR: No tissue pixels found in any patches!")
        return None

    # Combine all sampled tissue pixels
    all_lab = np.vstack(all_lab_pixels)
    print(f"  Sampled {len(all_lab):,} tissue pixels from {len(all_lab_pixels)} patches")

    # Compute statistics
    L_median = np.median(all_lab[:, 0])
    L_mad = np.median(np.abs(all_lab[:, 0] - L_median))

    a_median = np.median(all_lab[:, 1])
    a_mad = np.median(np.abs(all_lab[:, 1] - a_median))

    b_median = np.median(all_lab[:, 2])
    b_mad = np.median(np.abs(all_lab[:, 2] - b_median))

    return {
        'L_median': float(L_median),
        'L_mad': float(L_mad),
        'a_median': float(a_median),
        'a_mad': float(a_mad),
        'b_median': float(b_median),
        'b_mad': float(b_mad),
        'n_pixels': len(all_lab)
    }


def verify_normalized_slide(czi_path, target_params_file, channel=0):
    """Verify that normalized slide matches target parameters."""

    print(f"\n{'='*70}")
    print(f"NORMALIZATION VERIFICATION")
    print(f"{'='*70}")
    print(f"Slide: {czi_path}")
    print(f"Target params: {target_params_file}")

    # Load target parameters
    with open(target_params_file, 'r') as f:
        target = json.load(f)

    print(f"\nTARGET PARAMETERS:")
    print(f"  L: median={target['L_median']:.2f}, MAD={target['L_mad']:.2f}")
    print(f"  a: median={target['a_median']:.2f}, MAD={target['a_mad']:.2f}")
    print(f"  b: median={target['b_median']:.2f}, MAD={target['b_mad']:.2f}")

    # Load slide (assumes already normalized in loader cache)
    print(f"\nLoading slide channel {channel}...")
    loader = get_loader(czi_path, load_to_ram=True, channel=channel)

    # Compute statistics
    print(f"\nComputing output statistics on normalized slide...")
    output_stats = compute_tissue_lab_stats(
        loader.channel_data,
        variance_threshold=target.get('variance_threshold', 50.0),
        n_samples=100,
        patch_size=512
    )

    if output_stats is None:
        print("\n❌ VERIFICATION FAILED: Could not compute statistics")
        return False

    print(f"\nOUTPUT STATISTICS:")
    print(f"  L: median={output_stats['L_median']:.2f}, MAD={output_stats['L_mad']:.2f}")
    print(f"  a: median={output_stats['a_median']:.2f}, MAD={output_stats['a_mad']:.2f}")
    print(f"  b: median={output_stats['b_median']:.2f}, MAD={output_stats['b_mad']:.2f}")
    print(f"  Pixels sampled: {output_stats['n_pixels']:,}")

    # Compute errors
    print(f"\nDEVIATION FROM TARGET:")
    L_med_err = abs(output_stats['L_median'] - target['L_median'])
    L_mad_err = abs(output_stats['L_mad'] - target['L_mad'])
    a_med_err = abs(output_stats['a_median'] - target['a_median'])
    a_mad_err = abs(output_stats['a_mad'] - target['a_mad'])
    b_med_err = abs(output_stats['b_median'] - target['b_median'])
    b_mad_err = abs(output_stats['b_mad'] - target['b_mad'])

    print(f"  L median error: {L_med_err:.2f} ({L_med_err/target['L_median']*100:.1f}%)")
    print(f"  L MAD error: {L_mad_err:.2f} ({L_mad_err/target['L_mad']*100:.1f}%)")
    print(f"  a median error: {a_med_err:.2f}")
    print(f"  a MAD error: {a_mad_err:.2f} ({a_mad_err/target['a_mad']*100:.1f}%)")
    print(f"  b median error: {b_med_err:.2f}")
    print(f"  b MAD error: {b_mad_err:.2f} ({b_mad_err/target['b_mad']*100:.1f}%)")

    # Check if within acceptable tolerance (±5% for medians, ±10% for MADs)
    tolerance_met = (
        L_med_err / target['L_median'] < 0.05 and
        a_med_err < 2.0 and  # Absolute for a/b since they're near zero
        b_med_err < 2.0 and
        L_mad_err / target['L_mad'] < 0.10 and
        a_mad_err / target['a_mad'] < 0.10 and
        b_mad_err / target['b_mad'] < 0.10
    )

    print(f"\n{'='*70}")
    if tolerance_met:
        print("✅ VERIFICATION PASSED: Output matches target within tolerance")
    else:
        print("⚠️  VERIFICATION MARGINAL: Some deviations exceed tolerance")
    print(f"{'='*70}\n")

    return tolerance_met


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python verify_normalization_output.py <czi_path> <params_file>")
        sys.exit(1)

    verify_normalized_slide(sys.argv[1], sys.argv[2])
