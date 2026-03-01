#!/usr/bin/env python3
"""
Preview preprocessing effects on a CZI channel at reduced resolution.

Generates before/after PNG comparisons for flat-field correction,
photobleach correction, and stain normalization.  Lightweight enough
to run on a login node at 1/8 scale.

Usage:
    python scripts/preview_preprocessing.py \
        --czi-path /path/to/slide.czi \
        --channel 1 \
        --preprocessing flat_field \
        --output-dir /tmp/preview/

    python scripts/preview_preprocessing.py \
        --czi-path /path/to/slide.czi \
        --channel 0 \
        --preprocessing all \
        --scale-factor 8 \
        --output-dir /tmp/preview/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.io.czi_loader import get_loader, get_czi_metadata
from segmentation.preprocessing.flat_field import estimate_illumination_profile
from segmentation.preprocessing.illumination import (
    correct_photobleaching,
    normalize_rows_columns,
)


def downsample(img, factor):
    """Block-mean downsampling."""
    h, w = img.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    return img[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def load_channel(czi_path, channel, scale_factor):
    """Load a single channel, optionally downsampled."""
    loader = get_loader(str(czi_path))
    meta = get_czi_metadata(czi_path)
    n_channels = meta["n_channels"]
    if channel < 0 or channel >= n_channels:
        print(f"ERROR: channel {channel} out of range (0..{n_channels - 1})", file=sys.stderr)
        sys.exit(1)

    print(f"Loading channel {channel} from {czi_path.name}...")
    data = loader.load_channel_to_ram(channel)
    if scale_factor > 1:
        print(f"  Downsampling {scale_factor}x: {data.shape} -> ", end="")
        data = downsample(data.astype(np.float32), scale_factor)
        print(f"{data.shape}")
    return data, meta


def make_histogram(before, after, title, path):
    """Save side-by-side intensity histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    mask_b = before > 0
    mask_a = after > 0
    vmax = np.percentile(before[mask_b], 99.5) if mask_b.any() else 1

    axes[0].hist(before[mask_b].ravel(), bins=200, color="steelblue", alpha=0.8)
    axes[0].set_title("Before")
    axes[0].set_xlabel("Intensity")
    axes[0].set_xlim(0, vmax * 1.1)

    axes[1].hist(after[mask_a].ravel(), bins=200, color="coral", alpha=0.8)
    axes[1].set_title("After")
    axes[1].set_xlabel("Intensity")
    axes[1].set_xlim(0, vmax * 1.1)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved histogram: {path}")


def make_comparison(before, after, title, path):
    """Save side-by-side before/after image."""
    mask = before > 0
    vmax = np.percentile(before[mask], 99.5) if mask.any() else 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(before, cmap="gray", vmin=0, vmax=vmax, aspect="equal")
    axes[0].set_title("Before", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(after, cmap="gray", vmin=0, vmax=vmax, aspect="equal")
    axes[1].set_title("After", fontsize=12)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison: {path}")


def apply_flat_field(data, output_dir, channel, meta):
    """Apply flat-field correction and save comparison."""
    print("Applying flat-field correction...")
    before = data.copy()
    # estimate_illumination_profile expects dict {ch: array}
    profile = estimate_illumination_profile({channel: data}, block_size=64)
    corrected = data.copy()
    profile.correct_channel_inplace(corrected, channel)

    ch_name = meta["channels"][channel]["name"] if channel < len(meta["channels"]) else f"ch{channel}"
    make_comparison(before, corrected, f"Flat-Field Correction — {ch_name}", output_dir / f"ch{channel}_flat_field.png")
    make_histogram(before, corrected, f"Flat-Field Histogram — {ch_name}", output_dir / f"ch{channel}_flat_field_hist.png")
    return corrected


def apply_photobleach(data, output_dir, channel, meta):
    """Apply photobleach correction and save comparison."""
    print("Applying photobleach correction...")
    before = data.copy()
    corrected = correct_photobleaching(data.copy())

    ch_name = meta["channels"][channel]["name"] if channel < len(meta["channels"]) else f"ch{channel}"
    make_comparison(before, corrected, f"Photobleach Correction — {ch_name}", output_dir / f"ch{channel}_photobleach.png")
    make_histogram(before, corrected, f"Photobleach Histogram — {ch_name}", output_dir / f"ch{channel}_photobleach_hist.png")
    return corrected


def apply_row_col_norm(data, output_dir, channel, meta):
    """Apply row/column normalization and save comparison."""
    print("Applying row/column normalization...")
    before = data.copy()
    corrected = normalize_rows_columns(data.copy())

    ch_name = meta["channels"][channel]["name"] if channel < len(meta["channels"]) else f"ch{channel}"
    make_comparison(before, corrected, f"Row/Column Normalization — {ch_name}", output_dir / f"ch{channel}_rowcol_norm.png")
    make_histogram(before, corrected, f"Row/Col Norm Histogram — {ch_name}", output_dir / f"ch{channel}_rowcol_norm_hist.png")
    return corrected


def main():
    parser = argparse.ArgumentParser(
        description="Preview preprocessing effects on a CZI channel"
    )
    parser.add_argument("--czi-path", required=True, help="Path to CZI file")
    parser.add_argument("--channel", type=int, required=True, help="Channel index")
    parser.add_argument(
        "--preprocessing",
        choices=["flat_field", "photobleach", "rowcol", "all"],
        default="all",
        help="Which preprocessing to preview (default: all)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for PNGs")
    parser.add_argument(
        "--scale-factor",
        type=int,
        default=8,
        help="Downsample factor (default: 8 = 1/8 resolution)",
    )
    args = parser.parse_args()

    czi_path = Path(args.czi_path)
    if not czi_path.exists():
        print(f"ERROR: CZI not found: {czi_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_channel(czi_path, args.channel, args.scale_factor)

    steps = {
        "flat_field": apply_flat_field,
        "photobleach": apply_photobleach,
        "rowcol": apply_row_col_norm,
    }

    if args.preprocessing == "all":
        for name, func in steps.items():
            func(data.copy(), output_dir, args.channel, meta)
    else:
        steps[args.preprocessing](data, output_dir, args.channel, meta)

    print(f"\nDone. Output in: {output_dir}")


if __name__ == "__main__":
    main()
