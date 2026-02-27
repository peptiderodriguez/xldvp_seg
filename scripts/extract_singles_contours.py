#!/usr/bin/env python3
"""
Extract contours for single (outlier) detections from mask files.

Reads the original mask H5 files, extracts contours for each outlier detection,
converts to global coordinates, applies post-processing, and saves results.

Supports any cell type via the --cell-type flag, which determines the mask
filename pattern ({cell_type}_masks.h5).
"""

# Fix HDF5 plugin issues - must be set BEFORE importing h5py
import os
import sys
from pathlib import Path
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np
import hdf5plugin  # Must import BEFORE h5py to register LZ4 filter
import h5py
import cv2
from typing import List, Dict, Tuple, Optional

from segmentation.lmd.contour_processing import process_contour


def extract_contour_from_mask(mask: np.ndarray, label: int) -> Optional[np.ndarray]:
    """
    Extract the outer contour for a specific label in a mask.

    Args:
        mask: Label mask array (H, W) with integer labels
        label: The label value to extract

    Returns:
        Contour as numpy array of shape (N, 2) with [x, y] coordinates, or None
    """
    # Create binary mask for this label
    binary = (mask == label).astype(np.uint8)

    if binary.sum() == 0:
        return None

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Reshape from (N, 1, 2) to (N, 2) and convert to [x, y] format
    contour = largest.reshape(-1, 2)

    return contour


def local_to_global_contour(contour_local: np.ndarray, tile_origin: Tuple[int, int]) -> np.ndarray:
    """
    Convert local tile coordinates to global mosaic coordinates.

    Args:
        contour_local: Contour in tile coordinates, shape (N, 2) as [x, y]
        tile_origin: (tile_x, tile_y) origin of tile in global coordinates

    Returns:
        Contour in global coordinates
    """
    contour_global = contour_local.copy().astype(float)
    contour_global[:, 0] += tile_origin[0]  # Add tile_x to x coordinates
    contour_global[:, 1] += tile_origin[1]  # Add tile_y to y coordinates
    return contour_global


def main(base_dir: Path, tiles_dir: Path, clusters_path: Path,
         detections_path: Path, output_path: Path, pixel_size_um: float,
         cell_type: str = "nmj"):
    print("=" * 70)
    print("EXTRACT CONTOURS FOR SINGLES (OUTLIERS)")
    print("=" * 70)

    mask_filename = f"{cell_type}_masks.h5"

    # Load cluster data to get outlier indices
    print("\n[1/5] Loading cluster data...")
    with open(clusters_path) as f:
        cluster_data = json.load(f)

    outliers = cluster_data.get('outliers', [])
    outlier_indices = [o.get('detection_index', o.get('nmj_index')) for o in outliers]
    print(f"  Found {len(outlier_indices)} outliers")

    # Load all detections
    print("\n[2/5] Loading detections...")
    with open(detections_path) as f:
        all_detections = json.load(f)

    positives = [d for d in all_detections if d.get('rf_prediction', 0) >= 0.5]
    print(f"  {len(positives)} positive detections")

    # Get outlier detections (with bounds checking)
    outlier_detections = []
    skipped_indices = []
    for idx in outlier_indices:
        if idx < 0 or idx >= len(positives):
            skipped_indices.append(idx)
            continue
        outlier_detections.append(positives[idx])
    if skipped_indices:
        print(f"  WARNING: {len(skipped_indices)} outlier indices out of range "
              f"(max valid: {len(positives) - 1}): {skipped_indices[:10]}"
              f"{'...' if len(skipped_indices) > 10 else ''}")

    # Group by tile for efficient processing
    print("\n[3/5] Grouping by tile...")
    by_tile = {}
    for det in outlier_detections:
        tile_origin = det.get('tile_origin', [0, 0])
        tile_name = f"tile_{tile_origin[0]}_{tile_origin[1]}"
        if tile_name not in by_tile:
            by_tile[tile_name] = []
        by_tile[tile_name].append(det)

    print(f"  {len(by_tile)} tiles contain outliers")

    # Process each tile
    print("\n[4/5] Extracting contours from masks...")
    results = []
    success_count = 0
    fail_count = 0

    areas_before = []
    areas_after = []

    for tile_idx, (tile_name, tile_dets) in enumerate(by_tile.items()):
        if (tile_idx + 1) % 10 == 0:
            print(f"  Processing tile {tile_idx + 1}/{len(by_tile)}...")

        # Load mask file
        mask_path = tiles_dir / tile_name / mask_filename
        if not mask_path.exists():
            print(f"  WARNING: Mask file not found: {mask_path}")
            fail_count += len(tile_dets)
            continue

        with h5py.File(mask_path, 'r') as hf:
            masks = hf['masks'][:]

        # Process each detection in this tile
        for det in tile_dets:
            # Get mask label (centroid-based lookup stored in detection dict)
            label = det.get('mask_label')
            if label is None:
                fail_count += 1
                continue

            # Extract contour
            contour_local = extract_contour_from_mask(masks, label)
            if contour_local is None:
                fail_count += 1
                continue

            # Convert to global coordinates
            tile_origin = det.get('tile_origin', [0, 0])
            contour_global = local_to_global_contour(contour_local, tile_origin)

            # Apply post-processing (dilate + RDP)
            contour_processed, stats = process_contour(
                contour_global.tolist(),
                pixel_size_um=pixel_size_um,
                return_stats=True
            )

            if contour_processed is None:
                fail_count += 1
                continue

            # Track areas
            areas_before.append(stats['area_before_um2'])
            areas_after.append(stats['area_after_um2'])

            # Build result
            result = {
                'uid': det.get('uid'),
                'id': det.get('id'),
                'tile_name': tile_name,
                'tile_origin': tile_origin,
                'center': det.get('center'),
                'global_center': det.get('global_center'),
                'global_center_um': det.get('global_center_um'),
                'rf_prediction': det.get('rf_prediction'),
                'features': det.get('features'),
                'outer_contour_global': contour_global.tolist(),
                'outer_contour_processed_um': contour_processed.tolist(),
                'area_before_um2': stats['area_before_um2'],
                'area_after_um2': stats['area_after_um2'],
                'points_before': stats['points_before'],
                'points_after': stats['points_after'],
            }
            results.append(result)
            success_count += 1

    print(f"\n  Success: {success_count}, Failed: {fail_count}")

    # Save results
    print("\n[5/5] Saving results...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f"  Saved to: {output_path}")

    # Print area statistics
    print("\n" + "=" * 70)
    print("AREA STATISTICS (SINGLES)")
    print("=" * 70)

    if areas_before:
        arr_before = np.array(areas_before)
        arr_after = np.array(areas_after)

        print(f"\nBEFORE post-processing (n={len(areas_before)}):")
        print(f"  Range:  {arr_before.min():.1f} - {arr_before.max():.1f} µm²")
        print(f"  Mean:   {arr_before.mean():.1f} µm² (std: {arr_before.std():.1f})")
        print(f"  Median: {np.median(arr_before):.1f} µm²")

        print(f"\nAFTER post-processing (dilated +0.5µm, RDP):")
        print(f"  Range:  {arr_after.min():.1f} - {arr_after.max():.1f} µm²")
        print(f"  Mean:   {arr_after.mean():.1f} µm² (std: {arr_after.std():.1f})")
        print(f"  Median: {np.median(arr_after):.1f} µm²")

        print(f"\nArea increase: {arr_before.mean():.1f} → {arr_after.mean():.1f} µm² (+{(arr_after.mean()/arr_before.mean() - 1)*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract contours for single (outlier) detections from mask files')
    parser.add_argument('--base-dir', type=Path, required=True,
                        help='Base output directory (e.g., nmj_output/experiment_name)')
    parser.add_argument('--tiles-dir', type=Path, required=True,
                        help='Directory containing tile subdirectories with {cell_type}_masks.h5 files')
    parser.add_argument('--cell-type', type=str, default='nmj',
                        help='Cell type, used to construct mask filename as {cell_type}_masks.h5 (default: nmj)')
    parser.add_argument('--clusters', type=Path, default=None,
                        help='Path to clusters JSON (default: <base-dir>/nmj_clusters_375_425_500_1000.json)')
    parser.add_argument('--detections', type=Path, default=None,
                        help='Path to detections JSON (default: <base-dir>/nmj_detections_classified_v2.json)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output path for results JSON (default: <base-dir>/lmd_export/singles_with_contours.json)')
    parser.add_argument('--pixel-size', type=float, default=0.1725,
                        help='Pixel size in micrometers (default: 0.1725)')
    args = parser.parse_args()

    clusters_path = args.clusters if args.clusters else args.base_dir / "nmj_clusters_375_425_500_1000.json"
    detections_path = args.detections if args.detections else args.base_dir / "nmj_detections_classified_v2.json"
    output_path = args.output if args.output else args.base_dir / "lmd_export" / "singles_with_contours.json"

    main(
        base_dir=args.base_dir,
        tiles_dir=args.tiles_dir,
        clusters_path=clusters_path,
        detections_path=detections_path,
        output_path=output_path,
        pixel_size_um=args.pixel_size,
        cell_type=args.cell_type,
    )
