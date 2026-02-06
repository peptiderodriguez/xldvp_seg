#!/usr/bin/env python3
"""
Extract contours for single (outlier) NMJs from mask files.

Reads the original mask H5 files, extracts contours for each outlier NMJ,
converts to global coordinates, applies post-processing, and saves results.
"""

# Fix HDF5 plugin issues - must be set BEFORE importing h5py
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import json
import numpy as np
import hdf5plugin  # Must import BEFORE h5py to register LZ4 filter
import h5py
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from contour_processing import process_contour

# Paths
BASE_DIR = Path("/home/dude/nmj_output/20251107_Fig5_full_classified_v3")
TILES_DIR = Path("/home/dude/nmj_output/20251107_Fig5_full_multichannel/20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch/tiles")
CLUSTERS_PATH = BASE_DIR / "nmj_clusters_375_425_500_1000.json"
DETECTIONS_PATH = BASE_DIR / "nmj_detections_classified_v2.json"
OUTPUT_PATH = BASE_DIR / "lmd_export" / "singles_with_contours.json"

PIXEL_SIZE_UM = 0.1725


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


def main():
    print("=" * 70)
    print("EXTRACT CONTOURS FOR SINGLES (OUTLIERS)")
    print("=" * 70)

    # Load cluster data to get outlier indices
    print("\n[1/5] Loading cluster data...")
    with open(CLUSTERS_PATH) as f:
        cluster_data = json.load(f)

    outliers = cluster_data.get('outliers', [])
    outlier_indices = [o['nmj_index'] for o in outliers]
    print(f"  Found {len(outlier_indices)} outliers")

    # Load all detections
    print("\n[2/5] Loading detections...")
    with open(DETECTIONS_PATH) as f:
        all_detections = json.load(f)

    positives = [d for d in all_detections if d.get('rf_prediction') == 1]
    print(f"  {len(positives)} positive detections")

    # Get outlier detections
    outlier_detections = [positives[idx] for idx in outlier_indices]

    # Group by tile for efficient processing
    print("\n[3/5] Grouping by tile...")
    by_tile = {}
    for det in outlier_detections:
        tile_name = det.get('tile_name')
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
        mask_path = TILES_DIR / tile_name / "nmj_masks.h5"
        if not mask_path.exists():
            print(f"  WARNING: Mask file not found: {mask_path}")
            fail_count += len(tile_dets)
            continue

        with h5py.File(mask_path, 'r') as hf:
            masks = hf['masks'][:]

        # Process each detection in this tile
        for det in tile_dets:
            # Get detection label from id (e.g., "det_5" -> 5)
            det_id = det.get('id', '')
            try:
                label = int(det_id.split('_')[-1])
            except (ValueError, IndexError):
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
                pixel_size_um=PIXEL_SIZE_UM,
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
                'rf_confidence': det.get('rf_confidence'),
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
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to: {OUTPUT_PATH}")

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
    main()
