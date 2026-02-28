#!/usr/bin/env python3
"""
Generate before/after comparison images for photobleach + flat-field correction.

Produces:
  1. Full-slide thumbnails (before vs after) for each channel
  2. Zoomed crop comparisons
  3. Row/column mean profiles showing banding removal

Usage:
    python scripts/visualize_corrections.py \
        --czi-path /path/to/slide.czi \
        --channel 1 \
        --output-dir /tmp/correction_demo/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.io.czi_loader import get_loader, get_czi_metadata
from segmentation.preprocessing.illumination import normalize_rows_columns, estimate_band_severity
from segmentation.preprocessing.flat_field import estimate_illumination_profile


def downsample(img, factor):
    """Simple block-mean downsampling."""
    h, w = img.shape
    h2 = h // factor * factor
    w2 = w // factor * factor
    return img[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def save_thumbnail(data, path, title, vmax=None, cmap='gray'):
    """Save a downsampled thumbnail with title."""
    factor = max(1, max(data.shape) // 2000)
    thumb = downsample(data.astype(np.float32), factor)
    if vmax is None:
        vmax = np.percentile(thumb[thumb > 0], 99.5) if np.any(thumb > 0) else 1
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(thumb, cmap=cmap, vmin=0, vmax=vmax, aspect='equal')
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return vmax


def save_profile_comparison(before, after, path, channel, direction='row'):
    """Plot row or column mean profiles before and after correction."""
    axis = 1 if direction == 'row' else 0
    before_means = np.mean(before.astype(np.float32), axis=axis)
    after_means = np.mean(after.astype(np.float32), axis=axis)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    ax1.plot(before_means, color='red', alpha=0.7, linewidth=0.5)
    ax1.set_title(f'Ch{channel} {direction.title()} Means — BEFORE correction', fontsize=12)
    ax1.set_ylabel('Mean intensity')
    cv_before = np.std(before_means) / max(np.mean(before_means), 1e-6) * 100
    ax1.text(0.98, 0.95, f'CV = {cv_before:.1f}%', transform=ax1.transAxes,
             ha='right', va='top', fontsize=11, color='red',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.plot(after_means, color='blue', alpha=0.7, linewidth=0.5)
    ax2.set_title(f'Ch{channel} {direction.title()} Means — AFTER correction', fontsize=12)
    ax2.set_ylabel('Mean intensity')
    ax2.set_xlabel(f'{direction.title()} index')
    cv_after = np.std(after_means) / max(np.mean(after_means), 1e-6) * 100
    ax2.text(0.98, 0.95, f'CV = {cv_after:.1f}%', transform=ax2.transAxes,
             ha='right', va='top', fontsize=11, color='blue',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {direction} CV: {cv_before:.1f}% → {cv_after:.1f}%")


def save_crop_comparison(before, after, path, channel, crop_size=2000):
    """Side-by-side zoomed crop from the center of the slide."""
    h, w = before.shape
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - crop_size // 2)
    x0 = max(0, cx - crop_size // 2)
    y1 = min(h, y0 + crop_size)
    x1 = min(w, x0 + crop_size)

    crop_b = before[y0:y1, x0:x1].astype(np.float32)
    crop_a = after[y0:y1, x0:x1].astype(np.float32)

    # Shared vmax from before (so you can see the correction effect)
    vmax = np.percentile(crop_b[crop_b > 0], 99.5) if np.any(crop_b > 0) else 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(crop_b, cmap='gray', vmin=0, vmax=vmax)
    ax1.set_title(f'Ch{channel} Center Crop — BEFORE', fontsize=13)
    ax1.axis('off')

    ax2.imshow(crop_a, cmap='gray', vmin=0, vmax=vmax)
    ax2.set_title(f'Ch{channel} Center Crop — AFTER', fontsize=13)
    ax2.axis('off')

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_illumination_field(profile, channel, path):
    """Visualize the estimated illumination grid."""
    grid = profile.grids[channel]
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(grid, cmap='inferno', aspect='equal')
    ax.set_title(f'Ch{channel} Illumination Profile (coarse grid, slide_mean={profile.slide_means[channel]:.0f})',
                 fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Block median intensity')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize photobleach + flat-field corrections')
    parser.add_argument('--czi-path', required=True, help='Path to CZI file')
    parser.add_argument('--channel', type=int, default=1, help='Channel to visualize')
    parser.add_argument('--all-channels', action='store_true', help='Visualize all channels')
    parser.add_argument('--output-dir', default='/tmp/correction_demo/', help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CZI: {args.czi_path}")
    loader = get_loader(args.czi_path, load_to_ram=True, channel=args.channel)
    pixel_size = loader.get_pixel_size()
    w, h = loader.mosaic_size
    print(f"  Mosaic: {w} x {h} px")
    print(f"  Pixel size: {pixel_size} um/px")

    meta = get_czi_metadata(args.czi_path)
    n_channels = meta.get('n_channels', 1)
    channels = list(range(n_channels)) if args.all_channels else [args.channel]

    # Load all requested channels
    all_channel_data = {}
    for ch in channels:
        print(f"  Loading channel {ch}...")
        data = loader.get_channel_data(ch)
        if data is None:
            loader.load_channel(ch)
            data = loader.get_channel_data(ch)
        all_channel_data[ch] = data
        print(f"    Shape: {data.shape}, dtype: {data.dtype}")

    # --- Step 1: Save raw copies ---
    raw_data = {ch: all_channel_data[ch].copy() for ch in channels}

    # --- Step 2: Flat-field correction only ---
    print("\n=== FLAT-FIELD CORRECTION ===")
    profile = estimate_illumination_profile(all_channel_data)
    for ch in channels:
        profile.correct_channel_inplace(all_channel_data[ch], ch)
    flatfield_after = {ch: all_channel_data[ch] for ch in channels}

    # --- Generate images ---
    print("\n=== GENERATING IMAGES ===")
    for ch in channels:
        print(f"\nChannel {ch}:")

        severity_before = estimate_band_severity(raw_data[ch])
        severity_after = estimate_band_severity(flatfield_after[ch])
        print(f"  Before: row_cv={severity_before['row_cv']:.1f}%, col_cv={severity_before['col_cv']:.1f}%")
        print(f"  After:  row_cv={severity_after['row_cv']:.1f}%, col_cv={severity_after['col_cv']:.1f}%")

        # 1. Full-slide thumbnails: raw vs flat-field
        print("  Saving thumbnails...")
        vmax = save_thumbnail(raw_data[ch],
                              output_dir / f'ch{ch}_1_raw.png',
                              f'Ch{ch} RAW (no correction)')
        save_thumbnail(flatfield_after[ch],
                       output_dir / f'ch{ch}_2_flatfield.png',
                       f'Ch{ch} After Flat-field Correction', vmax=vmax)

        # 2. Illumination profile heatmap
        print("  Saving illumination profile...")
        save_illumination_field(profile, ch, output_dir / f'ch{ch}_illumination_profile.png')

        # 3. Row/column profiles: raw vs flat-field
        print("  Saving row/column profiles...")
        save_profile_comparison(raw_data[ch], flatfield_after[ch],
                                output_dir / f'ch{ch}_row_profiles.png', ch, 'row')
        save_profile_comparison(raw_data[ch], flatfield_after[ch],
                                output_dir / f'ch{ch}_col_profiles.png', ch, 'column')

        # 4. Zoomed center crop comparison
        print("  Saving crop comparisons...")
        save_crop_comparison(raw_data[ch], flatfield_after[ch],
                             output_dir / f'ch{ch}_crop_comparison.png', ch)

    print(f"\nDone! Images saved to: {output_dir}")
    print("Files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
