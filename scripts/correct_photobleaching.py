#!/usr/bin/env python3
"""
Photobleaching Band Correction for CZI Tiles

Corrects horizontal and vertical photobleaching bands in microscopy images
and re-runs lumen detection to compare vessel counts before/after correction.

Uses the segmentation.preprocessing module for correction algorithms.

Usage:
    python scripts/correct_photobleaching.py \
        --czi-path /path/to/slide.czi \
        --tile-x 150000 --tile-y 40000 \
        --tile-size 2000 --channel 2 \
        --output /tmp/lumen_tile3_corrected.png
"""

import argparse
import sys
from pathlib import Path

# Use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.io.czi_loader import CZILoader
from segmentation.preprocessing import (
    correct_photobleaching,
    estimate_band_severity,
)


def detect_vessels_lumen_based(
    image: np.ndarray,
    pixel_size_um: float = 0.22,
    min_diameter_um: float = 10.0,
    max_diameter_um: float = 500.0,
    min_circularity: float = 0.3,
    max_aspect_ratio: float = 4.0,
    wall_brightness_percentile: float = 75,
    lumen_darkness_percentile: float = 25,
) -> list:
    """
    Detect vessels by finding dark lumen regions surrounded by bright walls.

    Algorithm:
    1. Find dark regions (below lumen_darkness_percentile)
    2. For each dark region, check for bright ring (SMA wall) around it
    3. Filter by size, circularity, aspect ratio

    Args:
        image: 2D grayscale image (SMA channel)
        pixel_size_um: Microns per pixel
        min_diameter_um: Minimum vessel outer diameter in microns
        max_diameter_um: Maximum vessel outer diameter in microns
        min_circularity: Minimum circularity (0-1)
        max_aspect_ratio: Maximum aspect ratio
        wall_brightness_percentile: Percentile threshold for wall detection
        lumen_darkness_percentile: Percentile threshold for lumen detection

    Returns:
        List of detected vessels with contours and measurements
    """
    # Normalize image to float
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        return []

    # Calculate thresholds
    lumen_thresh = np.percentile(img_norm, lumen_darkness_percentile)
    wall_thresh = np.percentile(img_norm, wall_brightness_percentile)

    # Create lumen mask (dark regions)
    lumen_mask = (img_norm < lumen_thresh).astype(np.uint8) * 255

    # Create wall mask (bright regions)
    wall_mask = (img_norm > wall_thresh).astype(np.uint8) * 255

    # Clean up lumen mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    lumen_mask = cv2.morphologyEx(lumen_mask, cv2.MORPH_OPEN, kernel_small)
    lumen_mask = cv2.morphologyEx(lumen_mask, cv2.MORPH_CLOSE, kernel_medium)

    # Find lumen contours
    contours, _ = cv2.findContours(
        lumen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate size thresholds
    min_diameter_px = min_diameter_um / pixel_size_um
    max_diameter_px = max_diameter_um / pixel_size_um
    min_area_px = np.pi * (min_diameter_px / 2) ** 2 * 0.25  # 25% of circular area
    max_area_px = np.pi * (max_diameter_px / 2) ** 2

    vessels = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Size filter
        if area < min_area_px or area > max_area_px:
            continue

        # Fit ellipse for shape analysis
        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        center, (minor_axis, major_axis), angle = ellipse

        # Aspect ratio filter
        if minor_axis > 0:
            aspect_ratio = major_axis / minor_axis
        else:
            continue

        if aspect_ratio > max_aspect_ratio:
            continue

        # Circularity filter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            continue

        if circularity < min_circularity:
            continue

        # Check for bright wall around the lumen
        # Create a dilated mask to sample the wall region
        lumen_single = np.zeros_like(lumen_mask)
        cv2.drawContours(lumen_single, [cnt], 0, 255, -1)

        # Dilate to get wall region
        dilation_px = max(5, int(min_diameter_px * 0.2))
        kernel_wall = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px, dilation_px)
        )
        wall_region = cv2.dilate(lumen_single, kernel_wall) - lumen_single

        # Sample wall intensity
        wall_pixels = img_norm[wall_region > 0]
        if len(wall_pixels) < 10:
            continue

        wall_intensity = np.mean(wall_pixels)

        # Sample lumen intensity
        lumen_pixels = img_norm[lumen_single > 0]
        lumen_intensity = np.mean(lumen_pixels)

        # Wall should be significantly brighter than lumen
        intensity_ratio = wall_intensity / (lumen_intensity + 1e-8)
        if intensity_ratio < 1.3:  # Wall should be at least 30% brighter
            continue

        # Check wall has SMA signal (above threshold)
        wall_sma_fraction = np.mean(wall_pixels > wall_thresh)
        if wall_sma_fraction < 0.2:  # At least 20% of wall should be SMA+
            continue

        # Calculate diameter from lumen area
        diameter_px = 2 * np.sqrt(area / np.pi)
        diameter_um = diameter_px * pixel_size_um

        vessels.append({
            'contour': cnt,
            'lumen_area_px': area,
            'diameter_um': diameter_um,
            'center': center,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'wall_intensity': wall_intensity,
            'lumen_intensity': lumen_intensity,
            'intensity_ratio': intensity_ratio,
            'wall_sma_fraction': wall_sma_fraction,
        })

    return vessels


def visualize_correction(
    original: np.ndarray,
    corrected: np.ndarray,
    vessels_original: list,
    vessels_corrected: list,
    output_path: str,
    pixel_size_um: float = 0.22,
):
    """
    Create visualization comparing original and corrected images with vessel detections.

    Args:
        original: Original image with photobleaching bands
        corrected: Corrected image
        vessels_original: Vessels detected in original
        vessels_corrected: Vessels detected in corrected
        output_path: Path to save the visualization
        pixel_size_um: Microns per pixel
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Normalize for display
    def normalize_for_display(img):
        p2, p98 = np.percentile(img, [2, 98])
        return np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)

    orig_display = normalize_for_display(original)
    corr_display = normalize_for_display(corrected)

    # Row 1: Before correction
    # Panel 1: Original image
    axes[0, 0].imshow(orig_display, cmap='gray')
    axes[0, 0].set_title(f'Original Image\n(with photobleaching bands)', fontsize=12)
    axes[0, 0].axis('off')

    # Panel 2: Row/column intensity profiles
    row_means_orig = np.mean(original, axis=1)
    col_means_orig = np.mean(original, axis=0)

    ax_profile = axes[0, 1]
    ax_profile.plot(row_means_orig, label='Row means', alpha=0.7)
    ax_profile.plot(col_means_orig, label='Column means', alpha=0.7)
    ax_profile.set_xlabel('Position (pixels)')
    ax_profile.set_ylabel('Mean Intensity')
    ax_profile.set_title('Original Intensity Profiles\n(bands visible as spikes/dips)')
    ax_profile.legend()
    ax_profile.grid(True, alpha=0.3)

    # Panel 3: Original with vessels
    orig_rgb = np.stack([orig_display] * 3, axis=-1)
    for v in vessels_original:
        cv2.drawContours(orig_rgb, [v['contour']], 0, (0, 1, 0), 2)
        cx, cy = v['center']
        diameter = v['diameter_um']
        cv2.putText(
            orig_rgb, f'{diameter:.0f}um',
            (int(cx) + 10, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 1, 0), 1
        )
    axes[0, 2].imshow(orig_rgb)
    axes[0, 2].set_title(f'Original: {len(vessels_original)} vessels detected', fontsize=12)
    axes[0, 2].axis('off')

    # Row 2: After correction
    # Panel 4: Corrected image
    axes[1, 0].imshow(corr_display, cmap='gray')
    axes[1, 0].set_title('Corrected Image\n(bands removed)', fontsize=12)
    axes[1, 0].axis('off')

    # Panel 5: Corrected intensity profiles
    row_means_corr = np.mean(corrected, axis=1)
    col_means_corr = np.mean(corrected, axis=0)

    ax_profile2 = axes[1, 1]
    ax_profile2.plot(row_means_corr, label='Row means', alpha=0.7)
    ax_profile2.plot(col_means_corr, label='Column means', alpha=0.7)
    ax_profile2.set_xlabel('Position (pixels)')
    ax_profile2.set_ylabel('Mean Intensity')
    ax_profile2.set_title('Corrected Intensity Profiles\n(more uniform)')
    ax_profile2.legend()
    ax_profile2.grid(True, alpha=0.3)

    # Panel 6: Corrected with vessels
    corr_rgb = np.stack([corr_display] * 3, axis=-1)
    for v in vessels_corrected:
        cv2.drawContours(corr_rgb, [v['contour']], 0, (0, 1, 0), 2)
        cx, cy = v['center']
        diameter = v['diameter_um']
        cv2.putText(
            corr_rgb, f'{diameter:.0f}um',
            (int(cx) + 10, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 1, 0), 1
        )
    axes[1, 2].imshow(corr_rgb)
    axes[1, 2].set_title(f'Corrected: {len(vessels_corrected)} vessels detected', fontsize=12)
    axes[1, 2].axis('off')

    # Add summary statistics
    fig.suptitle(
        f'Photobleaching Band Correction - Vessel Detection Comparison\n'
        f'Tile size: {original.shape[1]}x{original.shape[0]} px | '
        f'Pixel size: {pixel_size_um:.3f} um/px',
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Photobleaching band correction for CZI tiles'
    )
    parser.add_argument(
        '--czi-path', required=True,
        help='Path to CZI file'
    )
    parser.add_argument(
        '--tile-x', type=int, required=True,
        help='Tile X coordinate (mosaic coordinates)'
    )
    parser.add_argument(
        '--tile-y', type=int, required=True,
        help='Tile Y coordinate (mosaic coordinates)'
    )
    parser.add_argument(
        '--tile-size', type=int, default=2000,
        help='Tile size in pixels (default: 2000)'
    )
    parser.add_argument(
        '--channel', type=int, default=2,
        help='Channel to extract (default: 2 for SMA)'
    )
    parser.add_argument(
        '--output', default='/tmp/lumen_tile3_corrected.png',
        help='Output path for visualization'
    )
    parser.add_argument(
        '--no-row-col', action='store_true',
        help='Skip row/column normalization'
    )
    parser.add_argument(
        '--no-morph', action='store_true',
        help='Skip morphological background subtraction'
    )
    parser.add_argument(
        '--morph-kernel', type=int, default=101,
        help='Morphological kernel size (default: 101)'
    )
    parser.add_argument(
        '--min-diameter', type=float, default=10.0,
        help='Minimum vessel diameter in microns (default: 10)'
    )
    parser.add_argument(
        '--max-diameter', type=float, default=500.0,
        help='Maximum vessel diameter in microns (default: 500)'
    )

    args = parser.parse_args()

    # Load CZI and extract tile
    print(f"Loading CZI: {args.czi_path}")
    loader = CZILoader(args.czi_path)

    print(f"Mosaic size: {loader.width} x {loader.height}")
    print(f"Mosaic origin: ({loader.x_start}, {loader.y_start})")
    print(f"Extracting tile at ({args.tile_x}, {args.tile_y}), size {args.tile_size}x{args.tile_size}, channel {args.channel}")

    # Get tile
    tile = loader._get_tile_from_czi(
        args.tile_x, args.tile_y, args.tile_size, args.channel
    )

    if tile is None or tile.size == 0:
        print("Error: Could not extract tile from CZI")
        sys.exit(1)

    print(f"Tile shape: {tile.shape}, dtype: {tile.dtype}")
    print(f"Tile intensity range: {tile.min()} - {tile.max()}")

    # Get pixel size
    pixel_size_um = loader.get_pixel_size()
    print(f"Pixel size: {pixel_size_um:.4f} um/px")

    # Apply correction
    print("\nApplying photobleaching correction...")
    corrected = correct_photobleaching(
        tile,
        row_col_normalize=not args.no_row_col,
        morph_subtract=not args.no_morph,
        morph_kernel_size=args.morph_kernel
    )

    print(f"Corrected intensity range: {corrected.min():.1f} - {corrected.max():.1f}")

    # Detect vessels in original
    print("\nDetecting vessels in original image...")
    vessels_original = detect_vessels_lumen_based(
        tile,
        pixel_size_um=pixel_size_um,
        min_diameter_um=args.min_diameter,
        max_diameter_um=args.max_diameter,
    )
    print(f"  Found {len(vessels_original)} vessels")

    # Detect vessels in corrected
    print("Detecting vessels in corrected image...")
    vessels_corrected = detect_vessels_lumen_based(
        corrected,
        pixel_size_um=pixel_size_um,
        min_diameter_um=args.min_diameter,
        max_diameter_um=args.max_diameter,
    )
    print(f"  Found {len(vessels_corrected)} vessels")

    # Print comparison statistics
    print("\n" + "=" * 60)
    print("VESSEL DETECTION COMPARISON")
    print("=" * 60)
    print(f"Original image:  {len(vessels_original)} vessels")
    print(f"Corrected image: {len(vessels_corrected)} vessels")
    print(f"Difference:      {len(vessels_corrected) - len(vessels_original):+d} vessels")

    if vessels_original:
        diameters_orig = [v['diameter_um'] for v in vessels_original]
        print(f"\nOriginal vessel diameters:")
        print(f"  Min: {min(diameters_orig):.1f} um")
        print(f"  Max: {max(diameters_orig):.1f} um")
        print(f"  Mean: {np.mean(diameters_orig):.1f} um")

    if vessels_corrected:
        diameters_corr = [v['diameter_um'] for v in vessels_corrected]
        print(f"\nCorrected vessel diameters:")
        print(f"  Min: {min(diameters_corr):.1f} um")
        print(f"  Max: {max(diameters_corr):.1f} um")
        print(f"  Mean: {np.mean(diameters_corr):.1f} um")

    # Calculate band severity using module function
    severity_orig = estimate_band_severity(tile)
    severity_corr = estimate_band_severity(corrected)

    print(f"\nIntensity uniformity (CV = coefficient of variation):")
    print(f"  Original rows:   CV = {severity_orig['row_cv']:.2f}%")
    print(f"  Corrected rows:  CV = {severity_corr['row_cv']:.2f}%")
    print(f"  Original cols:   CV = {severity_orig['col_cv']:.2f}%")
    print(f"  Corrected cols:  CV = {severity_corr['col_cv']:.2f}%")
    print(f"  Original severity: {severity_orig['severity']}")
    print(f"  Corrected severity: {severity_corr['severity']}")
    print("=" * 60)

    # Create visualization
    print(f"\nGenerating visualization...")
    visualize_correction(
        tile, corrected,
        vessels_original, vessels_corrected,
        args.output,
        pixel_size_um=pixel_size_um
    )

    # Cleanup
    loader.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
