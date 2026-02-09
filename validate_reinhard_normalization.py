#!/usr/bin/env python3
"""
Validate Reinhard normalization by visualizing before/after on sample tiles.

This script:
1. Loads Reinhard parameters from JSON
2. Extracts representative tiles from test slides
3. Applies normalization
4. Creates side-by-side visualizations
5. Shows RGB histograms and statistics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

# Add segmentation module to path
sys.path.insert(0, str(Path(__file__).parent))

from segmentation.io.czi_loader import get_loader
from segmentation.preprocessing.stain_normalization import (
    apply_reinhard_normalization,
    apply_reinhard_normalization_MEDIAN
)
from segmentation.detection.tissue import has_tissue


def load_reinhard_params(params_file):
    """Load Reinhard parameters from JSON (supports both mean/std and median/MAD)."""
    with open(params_file, 'r') as f:
        params = json.load(f)

    print(f"Loaded Reinhard parameters from {params_file}")

    # Check if it's median-based or mean-based
    if 'method' in params and params['method'] == 'reinhard_median':
        print(f"  Method: MEDIAN-based (robust to outliers)")
        print(f"  L: median={params['L_median']:.3f}, mad={params['L_mad']:.3f}")
        print(f"  a: median={params['a_median']:.3f}, mad={params['a_mad']:.3f}")
        print(f"  b: median={params['b_median']:.3f}, mad={params['b_mad']:.3f}")
    else:
        print(f"  Method: MEAN-based (traditional)")
        print(f"  L: mean={params['L_mean']:.3f}, std={params['L_std']:.3f}")
        print(f"  a: mean={params['a_mean']:.3f}, std={params['a_std']:.3f}")
        print(f"  b: mean={params['b_mean']:.3f}, std={params['b_std']:.3f}")

    print(f"  Computed from {params['n_slides']} slides, {params['n_total_pixels']:,} pixels")
    return params


def extract_sample_tiles(slide_path, n_tiles=6, tile_size=3000, variance_threshold=15.0):
    """
    Extract diverse tissue tiles from a slide.

    Returns tiles that represent different tissue regions (bright, dark, dense, sparse).
    """
    print(f"\nExtracting {n_tiles} sample tiles from {slide_path.name}...")

    # Load slide
    loader = get_loader(slide_path, load_to_ram=True, channel=0)
    channel_data = loader.get_channel_data(0)
    h, w = channel_data.shape[:2]

    print(f"  Slide size: {w} × {h}")

    # Sample tiles from different regions
    tiles = []
    coords = []

    # Grid sampling with spacing
    step_x = w // (n_tiles // 2 + 1)
    step_y = h // (n_tiles // 3 + 1)

    attempts = 0
    max_attempts = 100

    for y_offset in [step_y, 2*step_y]:
        for x_offset in [step_x, 2*step_x, 3*step_x]:
            if len(tiles) >= n_tiles:
                break

            attempts += 1
            if attempts > max_attempts:
                break

            # Extract tile
            x = min(x_offset, w - tile_size)
            y = min(y_offset, h - tile_size)

            tile = channel_data[y:y+tile_size, x:x+tile_size].copy()

            # Check if it has tissue
            has_tissue_flag, tissue_frac = has_tissue(tile, variance_threshold, block_size=512)

            if has_tissue_flag and tissue_frac > 0.2:
                tiles.append(tile)
                coords.append((x, y))
                print(f"    Tile {len(tiles)}: ({x}, {y}), tissue={tissue_frac:.1%}")

    print(f"  Extracted {len(tiles)} tiles with tissue")

    # Free memory
    loader.close()
    del channel_data, loader

    return tiles, coords


def compute_tile_stats(tile):
    """Compute RGB statistics for a tile."""
    return {
        'mean_r': tile[:, :, 0].mean(),
        'mean_g': tile[:, :, 1].mean(),
        'mean_b': tile[:, :, 2].mean(),
        'std_r': tile[:, :, 0].std(),
        'std_g': tile[:, :, 1].std(),
        'std_b': tile[:, :, 2].std(),
    }


def visualize_normalization(tiles_before, tiles_after, coords, output_path, slide_name):
    """
    Create comprehensive visualization of before/after normalization.
    """
    n_tiles = len(tiles_before)

    # Create figure with gridspec
    fig = plt.figure(figsize=(20, 4*n_tiles))
    gs = gridspec.GridSpec(n_tiles, 5, figure=fig, wspace=0.3, hspace=0.3)

    fig.suptitle(f'Reinhard Normalization Validation - {slide_name}',
                 fontsize=16, fontweight='bold', y=0.995)

    for i, (tile_before, tile_after, (x, y)) in enumerate(zip(tiles_before, tiles_after, coords)):

        # Before image
        ax_before = fig.add_subplot(gs[i, 0])
        ax_before.imshow(tile_before)
        ax_before.set_title(f'Before\n({x}, {y})', fontsize=10)
        ax_before.axis('off')

        # After image
        ax_after = fig.add_subplot(gs[i, 1])
        ax_after.imshow(tile_after)
        ax_after.set_title('After', fontsize=10)
        ax_after.axis('off')

        # Difference (enhanced for visibility)
        diff = np.abs(tile_after.astype(float) - tile_before.astype(float))
        ax_diff = fig.add_subplot(gs[i, 2])
        im = ax_diff.imshow(diff.mean(axis=2), cmap='hot', vmin=0, vmax=50)
        ax_diff.set_title(f'Difference\n(max={diff.max():.1f})', fontsize=10)
        ax_diff.axis('off')
        plt.colorbar(im, ax=ax_diff, fraction=0.046)

        # RGB histograms - Before
        ax_hist_before = fig.add_subplot(gs[i, 3])
        for channel, color, label in [(0, 'red', 'R'), (1, 'green', 'G'), (2, 'blue', 'B')]:
            hist, bins = np.histogram(tile_before[:, :, channel], bins=50, range=(0, 255))
            ax_hist_before.plot(bins[:-1], hist, color=color, alpha=0.7, label=label, linewidth=1.5)
        ax_hist_before.set_title('Before RGB Histogram', fontsize=9)
        ax_hist_before.set_xlim(0, 255)
        ax_hist_before.legend(fontsize=8)
        ax_hist_before.grid(alpha=0.3)

        # RGB histograms - After
        ax_hist_after = fig.add_subplot(gs[i, 4])
        for channel, color, label in [(0, 'red', 'R'), (1, 'green', 'G'), (2, 'blue', 'B')]:
            hist, bins = np.histogram(tile_after[:, :, channel], bins=50, range=(0, 255))
            ax_hist_after.plot(bins[:-1], hist, color=color, alpha=0.7, label=label, linewidth=1.5)
        ax_hist_after.set_title('After RGB Histogram', fontsize=9)
        ax_hist_after.set_xlim(0, 255)
        ax_hist_after.legend(fontsize=8)
        ax_hist_after.grid(alpha=0.3)

        # Statistics text
        stats_before = compute_tile_stats(tile_before)
        stats_after = compute_tile_stats(tile_after)

        ax_before.text(0.02, 0.98,
                      f"RGB: ({stats_before['mean_r']:.1f}, {stats_before['mean_g']:.1f}, {stats_before['mean_b']:.1f})",
                      transform=ax_before.transAxes, fontsize=8, va='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_after.text(0.02, 0.98,
                     f"RGB: ({stats_after['mean_r']:.1f}, {stats_after['mean_g']:.1f}, {stats_after['mean_b']:.1f})",
                     transform=ax_after.transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def print_statistics_summary(tiles_before, tiles_after):
    """Print summary statistics comparing before/after."""
    print("\n" + "="*70)
    print("STATISTICS SUMMARY")
    print("="*70)

    # Compute overall statistics
    all_before = np.concatenate([t.reshape(-1, 3) for t in tiles_before])
    all_after = np.concatenate([t.reshape(-1, 3) for t in tiles_after])

    print("\nOverall RGB Statistics:")
    print(f"  Before: R={all_before[:, 0].mean():.1f}±{all_before[:, 0].std():.1f}, "
          f"G={all_before[:, 1].mean():.1f}±{all_before[:, 1].std():.1f}, "
          f"B={all_before[:, 2].mean():.1f}±{all_before[:, 2].std():.1f}")
    print(f"  After:  R={all_after[:, 0].mean():.1f}±{all_after[:, 0].std():.1f}, "
          f"G={all_after[:, 1].mean():.1f}±{all_after[:, 1].std():.1f}, "
          f"B={all_after[:, 2].mean():.1f}±{all_after[:, 2].std():.1f}")

    # Per-tile variation
    tile_means_before = [t.mean(axis=(0, 1)) for t in tiles_before]
    tile_means_after = [t.mean(axis=(0, 1)) for t in tiles_after]

    var_before = np.std([m.mean() for m in tile_means_before])
    var_after = np.std([m.mean() for m in tile_means_after])

    print(f"\nInter-tile intensity variation:")
    print(f"  Before: {var_before:.2f} (std of tile means)")
    print(f"  After:  {var_after:.2f} (std of tile means)")
    print(f"  Reduction: {(1 - var_after/var_before)*100:.1f}%")

    print("="*70)


def main():
    """Main validation workflow."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate Reinhard normalization')
    parser.add_argument('--params', type=str,
                       default='reinhard_params_8slides.json',
                       help='Path to Reinhard parameters JSON')
    parser.add_argument('--slides', type=str, nargs='+',
                       default=['2025_11_18_FGC2.czi', '2025_11_18_MHU1.czi'],
                       help='Test slides (not in training set)')
    parser.add_argument('--slide-dir', type=str,
                       default='/viper/ptmp2/edrod/czi',
                       help='Directory containing slides')
    parser.add_argument('--output-dir', type=str,
                       default='validation_reinhard',
                       help='Output directory for visualizations')
    parser.add_argument('--n-tiles', type=int, default=6,
                       help='Number of tiles per slide')
    parser.add_argument('--tile-size', type=int, default=3000,
                       help='Tile size in pixels')

    args = parser.parse_args()

    # Setup
    params_file = Path(args.params)
    slide_dir = Path(args.slide_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("REINHARD NORMALIZATION VALIDATION")
    print("="*70)

    # Load parameters
    params = load_reinhard_params(params_file)

    # Process each test slide
    for slide_name in args.slides:
        slide_path = slide_dir / slide_name

        if not slide_path.exists():
            print(f"\nWarning: {slide_path} not found, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {slide_name}")
        print(f"{'='*70}")

        # Extract sample tiles
        tiles_before, coords = extract_sample_tiles(
            slide_path,
            n_tiles=args.n_tiles,
            tile_size=args.tile_size
        )

        if len(tiles_before) == 0:
            print(f"  No tissue tiles found, skipping {slide_name}")
            continue

        # Apply normalization (use correct function based on method)
        use_median = params.get('method') == 'reinhard_median'
        norm_func = apply_reinhard_normalization_MEDIAN if use_median else apply_reinhard_normalization
        method_name = "MEDIAN-based" if use_median else "MEAN-based"

        print(f"\nApplying {method_name} Reinhard normalization to {len(tiles_before)} tiles...")
        tiles_after = []
        for i, tile in enumerate(tiles_before):
            tile_norm = norm_func(
                tile,
                params,
                variance_threshold=15.0,
                tile_size=10000,
                block_size=7
            )
            tiles_after.append(tile_norm)
            print(f"  Tile {i+1}/{len(tiles_before)} normalized")

        # Print statistics
        print_statistics_summary(tiles_before, tiles_after)

        # Visualize
        output_path = output_dir / f'validation_{slide_path.stem}.png'
        visualize_normalization(tiles_before, tiles_after, coords, output_path, slide_name)

    print(f"\n{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nVisualizations saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Review the visualizations")
    print("  2. Check if normalization reduces inter-tile variation")
    print("  3. Verify tissue appearance looks natural")
    print("  4. If satisfied, proceed to Phase 2 segmentation")


if __name__ == '__main__':
    main()
