#!/usr/bin/env python3
"""
Generate complete LMD export with singles + clusters.

Creates:
1. Well assignments (384-well, B2+C2 quadrants, serpentine order)
2. Processed contours for all NMJs (dilation +0.5µm, RDP simplification)
3. LMD XML file for Leica LMD7

Usage:
    python generate_full_lmd_export.py
"""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import json
import numpy as np
import hdf5plugin
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
OUTPUT_DIR = BASE_DIR / "lmd_export"

PIXEL_SIZE_UM = 0.1725


def generate_quadrant_serpentine(quadrant: str, start_corner: str = 'auto') -> List[str]:
    """Generate wells for a 384-well quadrant in serpentine order."""
    even_rows = ['B', 'D', 'F', 'H', 'J', 'L', 'N']
    odd_rows = ['C', 'E', 'G', 'I', 'K', 'M', 'O']
    even_cols = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    odd_cols = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

    if quadrant == 'B2':
        rows, cols = even_rows, even_cols
    elif quadrant == 'B3':
        rows, cols = even_rows, odd_cols
    elif quadrant == 'C2':
        rows, cols = odd_rows, even_cols
    elif quadrant == 'C3':
        rows, cols = odd_rows, odd_cols
    else:
        raise ValueError(f"Unknown quadrant: {quadrant}")

    if start_corner == 'auto':
        start_corner = 'TL' if quadrant.startswith('B') else 'BR'

    if start_corner == 'TL':
        row_order = rows
        first_row_left_to_right = True
    elif start_corner == 'TR':
        row_order = rows
        first_row_left_to_right = False
    elif start_corner == 'BL':
        row_order = list(reversed(rows))
        first_row_left_to_right = True
    elif start_corner == 'BR':
        row_order = list(reversed(rows))
        first_row_left_to_right = False
    else:
        raise ValueError(f"Unknown start_corner: {start_corner}")

    wells = []
    for i, row in enumerate(row_order):
        if i % 2 == 0:
            col_order = cols if first_row_left_to_right else list(reversed(cols))
        else:
            col_order = list(reversed(cols)) if first_row_left_to_right else cols
        for col in col_order:
            wells.append(f"{row}{col}")

    return wells


def generate_multi_quadrant_serpentine(quadrants: List[str]) -> List[str]:
    """Generate wells for multiple quadrants with minimal movement between them."""
    if not quadrants:
        return []

    all_wells = []
    for i, quad in enumerate(quadrants):
        if i == 0:
            wells = generate_quadrant_serpentine(quad, start_corner='auto')
        else:
            prev_well = all_wells[-1]
            prev_row, prev_col = prev_well[0], int(prev_well[1:])
            top_rows = set('BCDEFGH')
            is_top = prev_row in top_rows
            is_left = prev_col <= 12

            if is_top and is_left:
                start = 'TL'
            elif is_top and not is_left:
                start = 'TR'
            elif not is_top and is_left:
                start = 'BL'
            else:
                start = 'BR'

            wells = generate_quadrant_serpentine(quad, start_corner=start)

        all_wells.extend(wells)

    return all_wells


def nearest_neighbor_order(points: List[Tuple[float, float]], start_idx: int = None) -> List[int]:
    """Order points using nearest-neighbor algorithm."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]

    points_arr = np.array(points)

    if start_idx is None:
        start_idx = np.argmin(points_arr[:, 0] + points_arr[:, 1])

    visited = [False] * n
    order = [start_idx]
    visited[start_idx] = True

    current = start_idx
    for _ in range(n - 1):
        min_dist = float('inf')
        nearest = -1

        for j in range(n):
            if not visited[j]:
                dist = np.sqrt((points_arr[current, 0] - points_arr[j, 0])**2 +
                              (points_arr[current, 1] - points_arr[j, 1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = j

        if nearest >= 0:
            order.append(nearest)
            visited[nearest] = True
            current = nearest

    return order


def extract_contour_from_mask(mask: np.ndarray, label: int) -> Optional[np.ndarray]:
    """Extract the outer contour for a specific label in a mask."""
    binary = (mask == label).astype(np.uint8)
    if binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def extract_contours_for_detections(detections: List[Dict], tiles_dir: Path) -> Dict[str, Dict]:
    """Extract and process contours for a list of detections."""
    # Group by tile
    by_tile = {}
    for det in detections:
        tile_name = det.get('tile_name')
        if tile_name not in by_tile:
            by_tile[tile_name] = []
        by_tile[tile_name].append(det)

    results = {}
    for tile_idx, (tile_name, tile_dets) in enumerate(by_tile.items()):
        if (tile_idx + 1) % 20 == 0:
            print(f"    Tile {tile_idx + 1}/{len(by_tile)}...")

        mask_path = tiles_dir / tile_name / "nmj_masks.h5"
        if not mask_path.exists():
            continue

        with h5py.File(mask_path, 'r') as hf:
            masks = hf['masks'][:]

        for det in tile_dets:
            det_id = det.get('id', '')
            uid = det.get('uid', det_id)

            try:
                label = int(det_id.split('_')[-1])
            except (ValueError, IndexError):
                continue

            contour_local = extract_contour_from_mask(masks, label)
            if contour_local is None:
                continue

            # Convert to global coordinates
            tile_origin = det.get('tile_origin', [0, 0])
            contour_global = contour_local.astype(float)
            contour_global[:, 0] += tile_origin[0]
            contour_global[:, 1] += tile_origin[1]

            # Apply post-processing
            processed, stats = process_contour(
                contour_global.tolist(),
                pixel_size_um=PIXEL_SIZE_UM,
                return_stats=True
            )

            if processed is None:
                continue

            results[uid] = {
                'uid': uid,
                'id': det_id,
                'tile_name': tile_name,
                'global_center': det.get('global_center'),
                'global_center_um': det.get('global_center_um'),
                'contour_global_px': contour_global.tolist(),
                'contour_processed_um': processed.tolist(),
                'area_before_um2': stats['area_before_um2'],
                'area_after_um2': stats['area_after_um2'],
                'points_before': stats['points_before'],
                'points_after': stats['points_after'],
            }

    return results


def main():
    print("=" * 70)
    print("FULL LMD EXPORT: SINGLES + CLUSTERS")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading data...")
    with open(CLUSTERS_PATH) as f:
        cluster_data = json.load(f)

    with open(DETECTIONS_PATH) as f:
        all_detections = json.load(f)

    positives = [d for d in all_detections if d.get('rf_prediction') == 1]
    print(f"  Total positive NMJs: {len(positives)}")

    # Get outliers (singles)
    outliers = cluster_data.get('outliers', [])
    outlier_indices = [o['nmj_index'] for o in outliers]
    outlier_dets = [positives[idx] for idx in outlier_indices]
    print(f"  Singles (outliers): {len(outlier_dets)}")

    # Get clusters
    clusters = cluster_data.get('main_clusters', [])
    clustered_indices = set()
    for c in clusters:
        clustered_indices.update(c['nmj_indices'])
    print(f"  Clusters: {len(clusters)} containing {len(clustered_indices)} NMJs")

    # Extract contours for singles
    print("\n[2/7] Extracting contours for singles...")
    singles_contours = extract_contours_for_detections(outlier_dets, TILES_DIR)
    print(f"  Extracted: {len(singles_contours)}/{len(outlier_dets)}")

    # Extract contours for clustered NMJs
    print("\n[3/7] Extracting contours for clustered NMJs...")
    clustered_dets = [positives[idx] for idx in clustered_indices]
    clustered_contours = extract_contours_for_detections(clustered_dets, TILES_DIR)
    print(f"  Extracted: {len(clustered_contours)}/{len(clustered_dets)}")

    # Generate well order
    quadrants = ['B2', 'C2']
    print(f"\n[4/7] Generating well order ({' + '.join(quadrants)} quadrants)...")
    all_wells = generate_multi_quadrant_serpentine(quadrants)
    print(f"  Total wells available: {len(all_wells)}")

    # Order singles by nearest-neighbor
    print("\n[5/7] Ordering singles by nearest-neighbor on slide...")
    singles_with_contours = []
    singles_positions = []

    for det in outlier_dets:
        uid = det.get('uid', det.get('id'))
        if uid in singles_contours:
            singles_with_contours.append({
                'detection': det,
                'contour_data': singles_contours[uid]
            })
            center = det.get('global_center', det.get('center', [0, 0]))
            singles_positions.append((center[0], center[1]))

    if singles_positions:
        nn_order = nearest_neighbor_order(singles_positions)
        ordered_singles = [singles_with_contours[i] for i in nn_order]

        # Calculate path length
        total_dist = 0
        for i in range(len(nn_order) - 1):
            p1 = singles_positions[nn_order[i]]
            p2 = singles_positions[nn_order[i + 1]]
            total_dist += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        print(f"  Ordered {len(ordered_singles)} singles, path: {total_dist * PIXEL_SIZE_UM / 1000:.1f} mm")

        last_single_pos = singles_positions[nn_order[-1]]
    else:
        ordered_singles = []
        last_single_pos = (0, 0)

    # Order clusters by nearest-neighbor
    print("\n[6/7] Ordering clusters by nearest-neighbor on slide...")
    cluster_centroids = [(c['cx'], c['cy']) for c in clusters]

    if cluster_centroids and last_single_pos:
        dists = [np.sqrt((cx - last_single_pos[0])**2 + (cy - last_single_pos[1])**2)
                 for cx, cy in cluster_centroids]
        start_cluster = np.argmin(dists)
    else:
        start_cluster = 0

    if cluster_centroids:
        cluster_nn_order = nearest_neighbor_order(cluster_centroids, start_idx=start_cluster)
        ordered_clusters = [clusters[i] for i in cluster_nn_order]

        total_dist = 0
        for i in range(len(cluster_nn_order) - 1):
            p1 = cluster_centroids[cluster_nn_order[i]]
            p2 = cluster_centroids[cluster_nn_order[i + 1]]
            total_dist += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        print(f"  Ordered {len(ordered_clusters)} clusters, path: {total_dist * PIXEL_SIZE_UM / 1000:.1f} mm")
    else:
        ordered_clusters = []

    # Build final export
    print("\n[7/7] Building export...")

    n_singles = len(ordered_singles)
    n_clusters = len(ordered_clusters)
    total_needed = n_singles + n_clusters

    if total_needed > len(all_wells):
        print(f"  WARNING: Need {total_needed} wells but only have {len(all_wells)}!")

    export_data = {
        'metadata': {
            'plate_format': '384',
            'quadrants': quadrants,
            'pixel_size_um': PIXEL_SIZE_UM,
            'dilation_um': 0.5,
            'rdp_epsilon_px': 5,
        },
        'summary': {
            'n_singles': n_singles,
            'n_clusters': n_clusters,
            'n_nmjs_in_clusters': sum(c['n'] for c in ordered_clusters),
            'total_wells': total_needed,
        },
        'singles': [],
        'clusters': [],
        'well_order': all_wells[:total_needed],
    }

    # Add singles
    for i, item in enumerate(ordered_singles):
        det = item['detection']
        contour = item['contour_data']
        well = all_wells[i]

        export_data['singles'].append({
            'well': well,
            'well_index': i + 1,
            'uid': contour['uid'],
            'global_center_um': det.get('global_center_um'),
            'contour_um': contour['contour_processed_um'],
            'area_um2': contour['area_after_um2'],
            'n_points': contour['points_after'],
        })

    # Add clusters
    for i, cluster in enumerate(ordered_clusters):
        well = all_wells[n_singles + i]

        # Get contours for all NMJs in cluster
        cluster_nmjs = []
        for idx in cluster['nmj_indices']:
            det = positives[idx]
            uid = det.get('uid', det.get('id'))
            if uid in clustered_contours:
                cluster_nmjs.append({
                    'uid': uid,
                    'global_center_um': det.get('global_center_um'),
                    'contour_um': clustered_contours[uid]['contour_processed_um'],
                    'area_um2': clustered_contours[uid]['area_after_um2'],
                    'n_points': clustered_contours[uid]['points_after'],
                })

        export_data['clusters'].append({
            'well': well,
            'well_index': n_singles + i + 1,
            'cluster_id': cluster['id'],
            'centroid_um': [cluster['cx'] * PIXEL_SIZE_UM, cluster['cy'] * PIXEL_SIZE_UM],
            'n_nmjs': len(cluster_nmjs),
            'total_area_um2': sum(nmj['area_um2'] for nmj in cluster_nmjs),
            'nmjs': cluster_nmjs,
        })

    # Save
    output_path = OUTPUT_DIR / "lmd_export_full.json"
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"  Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    print(f"  Plate: 384-well ({' + '.join(quadrants)} quadrants)")
    print(f"  Total wells: {total_needed} / {len(all_wells)} available")
    print(f"")
    print(f"  SINGLES: {n_singles}")
    print(f"    Wells: {all_wells[0]} → {all_wells[n_singles-1]}")
    singles_area = sum(s['area_um2'] for s in export_data['singles'])
    print(f"    Total area: {singles_area:.0f} µm²")
    print(f"")
    print(f"  CLUSTERS: {n_clusters}")
    print(f"    Wells: {all_wells[n_singles]} → {all_wells[total_needed-1]}")
    print(f"    NMJs in clusters: {export_data['summary']['n_nmjs_in_clusters']}")
    clusters_area = sum(c['total_area_um2'] for c in export_data['clusters'])
    print(f"    Total area: {clusters_area:.0f} µm²")
    print(f"")
    print(f"  TOTAL NMJs: {n_singles + export_data['summary']['n_nmjs_in_clusters']}")
    print(f"  TOTAL AREA: {singles_area + clusters_area:.0f} µm²")
    print("=" * 70)


if __name__ == '__main__':
    main()
