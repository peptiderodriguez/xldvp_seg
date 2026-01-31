#!/usr/bin/env python3
"""
Check RGB intensity distributions across all 16 slides.
Helps determine if cross-slide normalization is needed.
"""

import numpy as np
from pathlib import Path
from segmentation.io.czi_loader import get_loader
import json

def sample_slide_intensities(czi_path, n_samples=50000):
    """Sample random pixels from a slide to estimate intensity distribution."""
    loader = get_loader(str(czi_path), load_to_ram=False, channel=0)

    # Get slide dimensions
    height, width = loader.height, loader.width

    # Sample random positions
    np.random.seed(42)
    sample_y = np.random.randint(0, height, n_samples)
    sample_x = np.random.randint(0, width, n_samples)

    # Read pixels
    pixels = []
    for y, x in zip(sample_y, sample_x):
        try:
            # Read small patch around point
            patch = loader.get_region(x, y, min(10, width-x), min(10, height-y))
            if patch is not None and patch.size > 0:
                pixels.append(patch[0, 0])  # Center pixel
        except:
            continue

    pixels = np.array(pixels)

    if len(pixels) == 0:
        return None

    # Compute statistics
    stats = {
        'mean_r': float(np.mean(pixels[:, 0])),
        'mean_g': float(np.mean(pixels[:, 1])),
        'mean_b': float(np.mean(pixels[:, 2])),
        'std_r': float(np.std(pixels[:, 0])),
        'std_g': float(np.std(pixels[:, 1])),
        'std_b': float(np.std(pixels[:, 2])),
        'p01_r': float(np.percentile(pixels[:, 0], 1)),
        'p99_r': float(np.percentile(pixels[:, 0], 99)),
        'p01_g': float(np.percentile(pixels[:, 1], 1)),
        'p99_g': float(np.percentile(pixels[:, 1], 99)),
        'p01_b': float(np.percentile(pixels[:, 2], 1)),
        'p99_b': float(np.percentile(pixels[:, 2], 99)),
        'mean_intensity': float(np.mean(pixels)),
        'n_samples': len(pixels)
    }

    return stats

def main():
    czi_dir = Path("/viper/ptmp2/edrod/2025_11_18")
    slides = sorted(czi_dir.glob("*.czi"))

    print(f"Analyzing {len(slides)} slides...")
    print()

    all_stats = {}

    for czi_path in slides:
        slide_name = czi_path.stem
        print(f"Processing {slide_name}...", flush=True)

        stats = sample_slide_intensities(czi_path)

        if stats:
            all_stats[slide_name] = stats
            print(f"  Mean RGB: ({stats['mean_r']:.1f}, {stats['mean_g']:.1f}, {stats['mean_b']:.1f})")
            print(f"  Mean intensity: {stats['mean_intensity']:.1f}")
            print(f"  P01-P99 range: R[{stats['p01_r']:.0f}-{stats['p99_r']:.0f}] "
                  f"G[{stats['p01_g']:.0f}-{stats['p99_g']:.0f}] "
                  f"B[{stats['p01_b']:.0f}-{stats['p99_b']:.0f}]")
        else:
            print(f"  Failed to sample")
        print()

    # Compute cross-slide statistics
    mean_intensities = [s['mean_intensity'] for s in all_stats.values()]
    print("=" * 70)
    print("CROSS-SLIDE VARIATION:")
    print(f"  Mean intensity across slides: {np.mean(mean_intensities):.1f} ± {np.std(mean_intensities):.1f}")
    print(f"  Range: {np.min(mean_intensities):.1f} - {np.max(mean_intensities):.1f}")
    print(f"  Coefficient of variation: {np.std(mean_intensities) / np.mean(mean_intensities) * 100:.1f}%")
    print("=" * 70)

    # Save results
    output_file = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/slide_intensity_stats.json")
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Recommendation
    cv = np.std(mean_intensities) / np.mean(mean_intensities) * 100
    if cv < 5:
        print("\n✓ Low variation - normalization optional")
    elif cv < 10:
        print("\n⚠ Moderate variation - normalization recommended")
    else:
        print("\n⚠⚠ High variation - normalization strongly recommended")

if __name__ == "__main__":
    main()
