#!/usr/bin/env python3
"""
Validate cross-slide normalization by comparing intensity distributions.

Compares:
1. Non-normalized slides (should have different distributions)
2. Normalized slides (should have similar distributions)

Generates plots for visual confirmation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from segmentation.io.czi_loader import CZILoader
from segmentation.utils.logging import get_logger
import random

logger = get_logger(__name__)

def sample_pixels_from_slide(czi_path, n_samples=10000):
    """Sample random pixels from a slide."""
    logger.info(f"Sampling {n_samples} pixels from {czi_path.name}...")

    loader = CZILoader(str(czi_path))
    loader.load_channel_to_ram(0)  # Load RGB channel

    # Get image data
    img = loader.channel_data
    h, w = img.shape[:2]

    # Sample random pixels
    random_y = np.random.randint(0, h, n_samples)
    random_x = np.random.randint(0, w, n_samples)

    if img.ndim == 3:  # RGB
        samples = img[random_y, random_x, :]  # Shape: (n_samples, 3)
    else:  # Grayscale
        samples = img[random_y, random_x]
        samples = samples[:, np.newaxis]  # Shape: (n_samples, 1)

    loader.close()

    return samples

def load_normalization_params(params_file):
    """Load normalization parameters."""
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params

def normalize_samples(samples, params):
    """Apply normalization to samples using global params."""
    normalized = np.zeros_like(samples, dtype=np.float32)

    for c in range(samples.shape[1]):
        channel_key = str(c)
        if channel_key not in params:
            logger.warning(f"No params for channel {c}, skipping normalization")
            normalized[:, c] = samples[:, c]
            continue

        p1_global = params[channel_key]['p1']
        p99_global = params[channel_key]['p99']

        # Compute current percentiles
        p1_current = np.percentile(samples[:, c], 1)
        p99_current = np.percentile(samples[:, c], 99)

        # Rescale: [current_p1, current_p99] -> [global_p1, global_p99]
        if p99_current > p1_current:
            normalized[:, c] = (samples[:, c] - p1_current) / (p99_current - p1_current) * (p99_global - p1_global) + p1_global
        else:
            normalized[:, c] = samples[:, c]

        # Clip to valid range
        normalized[:, c] = np.clip(normalized[:, c], 0, 255)

    return normalized.astype(np.uint8)

def plot_distributions(slide_data, title, output_file):
    """Plot intensity distributions for multiple slides."""
    n_slides = len(slide_data)
    n_channels = slide_data[0]['samples'].shape[1]

    fig, axes = plt.subplots(1, n_channels, figsize=(6*n_channels, 5))
    if n_channels == 1:
        axes = [axes]

    channel_names = ['Red', 'Green', 'Blue'] if n_channels == 3 else ['Intensity']
    colors = ['red', 'green', 'blue'] if n_channels == 3 else ['gray']

    for c in range(n_channels):
        ax = axes[c]

        for slide_info in slide_data:
            samples = slide_info['samples'][:, c]
            ax.hist(samples, bins=50, alpha=0.5, label=slide_info['name'], density=True)

        ax.set_xlabel('Intensity')
        ax.set_ylabel('Density')
        ax.set_title(f'{channel_names[c]} Channel')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot: {output_file}")
    plt.close()

def plot_percentile_comparison(slide_data_unnorm, slide_data_norm, output_file):
    """Plot P1 and P99 values across slides to show normalization effect."""
    n_channels = slide_data_unnorm[0]['samples'].shape[1]

    fig, axes = plt.subplots(2, n_channels, figsize=(6*n_channels, 8))
    if n_channels == 1:
        axes = axes.reshape(2, 1)

    channel_names = ['Red', 'Green', 'Blue'] if n_channels == 3 else ['Intensity']

    for c in range(n_channels):
        slide_names = [s['name'] for s in slide_data_unnorm]

        # P1 values
        p1_unnorm = [np.percentile(s['samples'][:, c], 1) for s in slide_data_unnorm]
        p1_norm = [np.percentile(s['samples'][:, c], 1) for s in slide_data_norm]

        axes[0, c].plot(p1_unnorm, 'o-', label='Non-normalized', markersize=8)
        axes[0, c].plot(p1_norm, 's-', label='Normalized', markersize=8)
        axes[0, c].set_ylabel('P1 (1st percentile)')
        axes[0, c].set_title(f'{channel_names[c]} Channel - P1')
        axes[0, c].legend()
        axes[0, c].grid(True, alpha=0.3)
        axes[0, c].set_xticks(range(len(slide_names)))
        axes[0, c].set_xticklabels(slide_names, rotation=45, ha='right', fontsize=8)

        # P99 values
        p99_unnorm = [np.percentile(s['samples'][:, c], 99) for s in slide_data_unnorm]
        p99_norm = [np.percentile(s['samples'][:, c], 99) for s in slide_data_norm]

        axes[1, c].plot(p99_unnorm, 'o-', label='Non-normalized', markersize=8)
        axes[1, c].plot(p99_norm, 's-', label='Normalized', markersize=8)
        axes[1, c].set_ylabel('P99 (99th percentile)')
        axes[1, c].set_title(f'{channel_names[c]} Channel - P99')
        axes[1, c].legend()
        axes[1, c].grid(True, alpha=0.3)
        axes[1, c].set_xticks(range(len(slide_names)))
        axes[1, c].set_xticklabels(slide_names, rotation=45, ha='right', fontsize=8)

    fig.suptitle('Normalization Effect: P1 and P99 Values Across Slides', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot: {output_file}")
    plt.close()

def main():
    logger.info("="*70)
    logger.info("NORMALIZATION VALIDATION")
    logger.info("="*70)

    # Configuration
    czi_dir = Path("/viper/ptmp2/edrod/2025_11_18")
    params_file = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/normalization_params_all16.json")
    output_dir = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/validation_plots")
    output_dir.mkdir(exist_ok=True)

    # Select subset of slides for validation (avoid loading all 16)
    slide_names = [
        "2025_11_18_FGC1.czi",
        "2025_11_18_FGC3.czi",
        "2025_11_18_FHU2.czi",
        "2025_11_18_MGC1.czi",
        "2025_11_18_MHU3.czi",
    ]

    n_samples = 20000  # Sample pixels per slide

    # Load normalization parameters
    logger.info(f"Loading normalization parameters from {params_file}")
    norm_params = load_normalization_params(params_file)
    logger.info(f"Parameters loaded: {list(norm_params.keys())}")

    # Sample pixels from slides
    slide_data_unnorm = []
    slide_data_norm = []

    for slide_name in slide_names:
        czi_path = czi_dir / slide_name
        if not czi_path.exists():
            logger.warning(f"Slide not found: {czi_path}")
            continue

        # Sample pixels
        samples = sample_pixels_from_slide(czi_path, n_samples)

        # Store unnormalized
        slide_data_unnorm.append({
            'name': slide_name.replace('2025_11_18_', '').replace('.czi', ''),
            'samples': samples
        })

        # Apply normalization
        samples_norm = normalize_samples(samples, norm_params)
        slide_data_norm.append({
            'name': slide_name.replace('2025_11_18_', '').replace('.czi', ''),
            'samples': samples_norm
        })

    logger.info(f"Sampled {len(slide_data_unnorm)} slides")

    # Generate plots
    logger.info("Generating distribution plots...")

    # Plot 1: Non-normalized distributions (should be different)
    plot_distributions(
        slide_data_unnorm,
        "Non-Normalized Slides - Intensity Distributions (Should Vary)",
        output_dir / "distributions_unnormalized.png"
    )

    # Plot 2: Normalized distributions (should be similar)
    plot_distributions(
        slide_data_norm,
        "Normalized Slides - Intensity Distributions (Should Be Similar)",
        output_dir / "distributions_normalized.png"
    )

    # Plot 3: Percentile comparison
    plot_percentile_comparison(
        slide_data_unnorm,
        slide_data_norm,
        output_dir / "percentile_comparison.png"
    )

    logger.info("="*70)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Plots saved to: {output_dir}")
    logger.info("")
    logger.info("Expected results:")
    logger.info("  1. distributions_unnormalized.png: Different curves per slide")
    logger.info("  2. distributions_normalized.png: Overlapping curves (similar)")
    logger.info("  3. percentile_comparison.png: Normalized lines should be flat")
    logger.info("")

if __name__ == "__main__":
    main()
