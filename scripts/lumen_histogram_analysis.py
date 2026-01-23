#!/usr/bin/env python3
"""
Lumen Size Histogram Analysis

Processes multiple tiles with photobleaching correction and generates
a histogram of lumen sizes across all tiles.

Usage:
    python scripts/lumen_histogram_analysis.py
"""

import sys
from pathlib import Path

# Use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.io.czi_loader import CZILoader


def normalize_rows_columns(image: np.ndarray, target_mean: float = None) -> np.ndarray:
    """
    Normalize each row and column to have consistent mean intensity.
    """
    img = image.astype(np.float64)

    if target_mean is None:
        target_mean = np.mean(img)

    # Step 1: Normalize rows
    row_means = np.mean(img, axis=1, keepdims=True)
    row_means = np.where(row_means > 0, row_means, 1)
    row_corrected = img * (target_mean / row_means)

    # Step 2: Normalize columns on the row-corrected image
    col_means = np.mean(row_corrected, axis=0, keepdims=True)
    col_means = np.where(col_means > 0, col_means, 1)
    corrected = row_corrected * (target_mean / col_means)

    return corrected


def morphological_background_subtraction(
    image: np.ndarray,
    kernel_size: int = 101
) -> np.ndarray:
    """
    Remove low-frequency background using morphological opening.
    """
    img = image.astype(np.float64)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    background = cv2.morphologyEx(
        img.astype(np.float32),
        cv2.MORPH_OPEN,
        kernel
    )

    corrected = img - background.astype(np.float64)
    corrected = corrected + np.mean(background)

    return corrected


def combined_correction(
    image: np.ndarray,
    row_col_normalize: bool = True,
    morph_subtract: bool = True,
    morph_kernel_size: int = 101
) -> np.ndarray:
    """
    Apply combined photobleaching correction.
    """
    corrected = image.astype(np.float64)

    if row_col_normalize:
        corrected = normalize_rows_columns(corrected)

    if morph_subtract:
        corrected = morphological_background_subtraction(
            corrected,
            kernel_size=morph_kernel_size
        )

    return corrected


def detect_vessels_lumen_based(
    image: np.ndarray,
    pixel_size_um: float = 0.1725,
    min_diameter_um: float = 10.0,
    max_diameter_um: float = 500.0,
    min_circularity: float = 0.3,
    max_aspect_ratio: float = 4.0,
    wall_brightness_percentile: float = 75,
    lumen_darkness_percentile: float = 25,
) -> list:
    """
    Detect vessels by finding dark lumen regions surrounded by bright walls.
    """
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        return []

    lumen_thresh = np.percentile(img_norm, lumen_darkness_percentile)
    wall_thresh = np.percentile(img_norm, wall_brightness_percentile)

    lumen_mask = (img_norm < lumen_thresh).astype(np.uint8) * 255
    wall_mask = (img_norm > wall_thresh).astype(np.uint8) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    lumen_mask = cv2.morphologyEx(lumen_mask, cv2.MORPH_OPEN, kernel_small)
    lumen_mask = cv2.morphologyEx(lumen_mask, cv2.MORPH_CLOSE, kernel_medium)

    contours, _ = cv2.findContours(
        lumen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_diameter_px = min_diameter_um / pixel_size_um
    max_diameter_px = max_diameter_um / pixel_size_um
    min_area_px = np.pi * (min_diameter_px / 2) ** 2 * 0.25
    max_area_px = np.pi * (max_diameter_px / 2) ** 2

    vessels = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area_px or area > max_area_px:
            continue

        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        center, (minor_axis, major_axis), angle = ellipse

        if minor_axis > 0:
            aspect_ratio = major_axis / minor_axis
        else:
            continue

        if aspect_ratio > max_aspect_ratio:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            continue

        if circularity < min_circularity:
            continue

        lumen_single = np.zeros_like(lumen_mask)
        cv2.drawContours(lumen_single, [cnt], 0, 255, -1)

        dilation_px = max(5, int(min_diameter_px * 0.2))
        kernel_wall = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_px, dilation_px)
        )
        wall_region = cv2.dilate(lumen_single, kernel_wall) - lumen_single

        wall_pixels = img_norm[wall_region > 0]
        if len(wall_pixels) < 10:
            continue

        wall_intensity = np.mean(wall_pixels)

        lumen_pixels = img_norm[lumen_single > 0]
        lumen_intensity = np.mean(lumen_pixels)

        intensity_ratio = wall_intensity / (lumen_intensity + 1e-8)
        if intensity_ratio < 1.15:  # More permissive (was 1.3)
            continue

        wall_sma_fraction = np.mean(wall_pixels > wall_thresh)
        if wall_sma_fraction < 0.1:  # More permissive (was 0.2)
            continue

        diameter_px = 2 * np.sqrt(area / np.pi)
        diameter_um = diameter_px * pixel_size_um
        lumen_area_um2 = area * (pixel_size_um ** 2)

        vessels.append({
            'contour': cnt,
            'lumen_area_px': area,
            'lumen_area_um2': lumen_area_um2,
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


def visualize_tile(
    original: np.ndarray,
    corrected: np.ndarray,
    vessels: list,
    output_path: str,
    tile_x: int,
    tile_y: int,
    tile_idx: int,
    pixel_size_um: float = 0.1725,
):
    """
    Create visualization for a single tile showing original, corrected, and detected vessels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def normalize_for_display(img):
        p2, p98 = np.percentile(img, [2, 98])
        return np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)

    orig_display = normalize_for_display(original)
    corr_display = normalize_for_display(corrected)

    # Panel 1: Original image
    axes[0].imshow(orig_display, cmap='gray')
    axes[0].set_title(f'Original Image\n(with photobleaching bands)', fontsize=11)
    axes[0].axis('off')

    # Panel 2: Corrected image
    axes[1].imshow(corr_display, cmap='gray')
    axes[1].set_title('Corrected Image\n(bands removed)', fontsize=11)
    axes[1].axis('off')

    # Panel 3: Corrected with vessels
    corr_rgb = np.stack([corr_display] * 3, axis=-1)
    for v in vessels:
        cv2.drawContours(corr_rgb, [v['contour']], 0, (0, 1, 0), 2)
        cx, cy = v['center']
        diameter = v['diameter_um']
        cv2.putText(
            corr_rgb, f'{diameter:.0f}um',
            (int(cx) + 5, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 1, 0), 1
        )
    axes[2].imshow(corr_rgb)
    axes[2].set_title(f'Detected Vessels: {len(vessels)}\n(with diameter labels)', fontsize=11)
    axes[2].axis('off')

    fig.suptitle(
        f'Tile {tile_idx} at ({tile_x}, {tile_y}) | Size: {original.shape[1]}x{original.shape[0]} px | '
        f'Pixel size: {pixel_size_um:.4f} um/px',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Visualization saved to: {output_path}")


def create_histogram(
    all_lumen_areas: list,
    output_path: str,
):
    """
    Create histogram of lumen sizes.
    """
    if not all_lumen_areas:
        print("No lumen areas to plot!")
        return

    areas = np.array(all_lumen_areas)

    # Calculate statistics
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    std_area = np.std(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine bin count and edges
    n_bins = min(50, len(areas) // 3 + 1)
    n_bins = max(n_bins, 10)

    # Use log scale for bins if range is large
    if max_area / min_area > 100:
        bins = np.logspace(np.log10(min_area), np.log10(max_area), n_bins + 1)
        ax.set_xscale('log')
    else:
        bins = np.linspace(min_area, max_area, n_bins + 1)

    # Plot histogram
    counts, bin_edges, patches = ax.hist(
        areas,
        bins=bins,
        color='steelblue',
        edgecolor='black',
        alpha=0.7
    )

    # Add mean and median lines
    ax.axvline(mean_area, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_area:.1f} um^2')
    ax.axvline(median_area, color='orange', linestyle='-', linewidth=2, label=f'Median: {median_area:.1f} um^2')

    # Labels and title
    ax.set_xlabel('Lumen Area (um^2)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(
        f'Lumen Size Distribution\n'
        f'Total Vessels: {len(areas)} | Mean: {mean_area:.1f} um^2 | Median: {median_area:.1f} um^2 | Std: {std_area:.1f} um^2',
        fontsize=14, fontweight='bold'
    )

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add text box with statistics
    textstr = '\n'.join([
        f'Total vessels: {len(areas)}',
        f'Mean: {mean_area:.2f} um^2',
        f'Median: {median_area:.2f} um^2',
        f'Std: {std_area:.2f} um^2',
        f'Min: {min_area:.2f} um^2',
        f'Max: {max_area:.2f} um^2',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nHistogram saved to: {output_path}")


def main():
    # Configuration
    czi_path = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
    channel = 2  # SMA channel
    tile_size = 2000
    pixel_size_um = 0.1725

    # Tile locations
    tiles = [
        (50000, 30000),
        (100000, 50000),
        (150000, 40000),
        (180000, 70000),
        (80000, 80000),
    ]

    # Detection parameters (consistent across all tiles)
    # Using more permissive parameters to detect more vessels
    detection_params = {
        'pixel_size_um': pixel_size_um,
        'min_diameter_um': 3.0,        # Very small to catch capillaries
        'max_diameter_um': 1000.0,     # Allow larger vessels
        'min_circularity': 0.15,       # Very permissive for irregular shapes
        'max_aspect_ratio': 6.0,       # Allow more elongated shapes
        'wall_brightness_percentile': 65,  # More permissive
        'lumen_darkness_percentile': 35,   # More permissive
    }

    print("=" * 70)
    print("LUMEN SIZE HISTOGRAM ANALYSIS")
    print("=" * 70)
    print(f"CZI file: {czi_path}")
    print(f"Channel: {channel} (SMA)")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Pixel size: {pixel_size_um} um/px")
    print(f"Number of tiles: {len(tiles)}")
    print("=" * 70)

    # Load CZI
    print(f"\nLoading CZI file...")
    loader = CZILoader(czi_path)
    print(f"Mosaic size: {loader.width} x {loader.height}")
    print(f"Mosaic origin: ({loader.x_start}, {loader.y_start})")

    # Actual pixel size from metadata
    actual_pixel_size = loader.get_pixel_size()
    print(f"Pixel size from metadata: {actual_pixel_size:.4f} um/px")

    # Use the specified pixel size
    detection_params['pixel_size_um'] = pixel_size_um

    all_lumen_areas = []
    all_vessels = []

    # Process each tile
    for idx, (tile_x, tile_y) in enumerate(tiles, start=1):
        print(f"\n{'='*50}")
        print(f"Processing Tile {idx}: ({tile_x}, {tile_y})")
        print("="*50)

        # Extract tile
        tile = loader._get_tile_from_czi(tile_x, tile_y, tile_size, channel)

        if tile is None or tile.size == 0:
            print(f"  Warning: Could not extract tile at ({tile_x}, {tile_y}), skipping")
            continue

        print(f"  Tile shape: {tile.shape}, dtype: {tile.dtype}")
        print(f"  Tile intensity range: {tile.min()} - {tile.max()}")

        # Apply photobleaching correction
        print("  Applying photobleaching correction...")
        corrected = combined_correction(
            tile,
            row_col_normalize=True,
            morph_subtract=True,
            morph_kernel_size=101
        )
        print(f"  Corrected intensity range: {corrected.min():.1f} - {corrected.max():.1f}")

        # Detect vessels
        print("  Detecting vessels (lumen-first method)...")
        vessels = detect_vessels_lumen_based(corrected, **detection_params)
        print(f"  Found {len(vessels)} vessels")

        # Collect lumen areas
        for v in vessels:
            all_lumen_areas.append(v['lumen_area_um2'])
            all_vessels.append({
                'tile_idx': idx,
                'tile_x': tile_x,
                'tile_y': tile_y,
                **v
            })

        # Create visualization
        output_path = f"/tmp/lumen_corrected_tile{idx}_{tile_x}_{tile_y}.png"
        visualize_tile(
            tile, corrected, vessels,
            output_path,
            tile_x, tile_y, idx,
            pixel_size_um=pixel_size_um
        )

    # Close loader
    loader.close()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if all_lumen_areas:
        areas = np.array(all_lumen_areas)
        print(f"Total vessels detected: {len(areas)}")
        print(f"Mean lumen area: {np.mean(areas):.2f} um^2")
        print(f"Median lumen area: {np.median(areas):.2f} um^2")
        print(f"Std lumen area: {np.std(areas):.2f} um^2")
        print(f"Min lumen area: {np.min(areas):.2f} um^2")
        print(f"Max lumen area: {np.max(areas):.2f} um^2")

        # Create histogram
        create_histogram(all_lumen_areas, "/tmp/lumen_size_histogram.png")
    else:
        print("No vessels detected across all tiles!")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    # Print output file locations
    print("\nOutput files:")
    for idx, (tile_x, tile_y) in enumerate(tiles, start=1):
        print(f"  - /tmp/lumen_corrected_tile{idx}_{tile_x}_{tile_y}.png")
    print(f"  - /tmp/lumen_size_histogram.png")


if __name__ == '__main__':
    main()
