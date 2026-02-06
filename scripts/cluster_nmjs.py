#!/usr/bin/env python3
"""
Biological clustering of NMJ detections for LMD well assignment.

Two-stage greedy spatial clustering with area constraint:
  Round 1 (500 um): Tight spatial groups
  Round 2 (1000 um): Remaining NMJs with looser distance

Cluster target: total NMJ area per cluster = 375-425 um2 (midpoint 400).
Overshoot rule: add NMJ only if |total + nmj - 400| < |total - 400|.

Usage:
    python scripts/cluster_nmjs.py \\
        --detections nmj_detections.json \\
        --pixel-size 0.1725 \\
        --area-min 375 --area-max 425 \\
        --dist-round1 500 --dist-round2 1000 \\
        --min-score 0.5 \\
        --output nmj_clusters.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def get_nmj_area_um2(det: Dict, pixel_size: float) -> float:
    """Compute area in um2 from detection features or contour."""
    # Check for pre-computed area
    if 'area_um2' in det:
        return det['area_um2']

    # Check features dict
    features = det.get('features', {})
    if 'area_um2' in features:
        return features['area_um2']
    if 'area' in features:
        return features['area'] * pixel_size * pixel_size

    # Fallback: compute from contour if available
    for key in ('outer_contour_global', 'outer_contour'):
        contour = det.get(key)
        if contour is not None and len(contour) >= 3:
            pts = np.array(contour, dtype=np.float64)
            # Shoelace formula
            x, y = pts[:, 0], pts[:, 1]
            area_px = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            return area_px * pixel_size * pixel_size

    return 0.0


def get_nmj_center(det: Dict) -> Optional[Tuple[float, float]]:
    """Get (x, y) center coordinates from detection."""
    if 'global_center' in det:
        gc = det['global_center']
        return (float(gc[0]), float(gc[1]))
    if 'center' in det:
        c = det['center']
        return (float(c[0]), float(c[1]))
    return None


def cluster_nmjs_greedy_area(
    indices: List[int],
    coords: np.ndarray,
    areas: np.ndarray,
    dist_threshold: float,
    area_min: float,
    area_max: float,
) -> Tuple[List[List[int]], List[int]]:
    """
    Single-round greedy spatial clustering with area constraint.

    Args:
        indices: Detection indices to cluster
        coords: Array (N, 2) of [x, y] positions in um
        areas: Array (N,) of areas in um2
        dist_threshold: Max distance in um to add to cluster
        area_min: Target area lower bound
        area_max: Target area upper bound

    Returns:
        (clusters, remaining) where clusters is list of index lists,
        remaining is list of unclustered indices
    """
    midpoint = (area_min + area_max) / 2.0

    unclustered = set(range(len(indices)))
    failed_seeds = set()  # Seeds that couldn't grow - prevent infinite loop
    clusters = []

    # Sort by position (top-left to bottom-right) for deterministic seeding
    sort_order = np.argsort(coords[:, 0] + coords[:, 1])

    while unclustered - failed_seeds:
        # Pick first unclustered (non-failed) in sorted order as seed
        seed = None
        for idx in sort_order:
            if idx in unclustered and idx not in failed_seeds:
                seed = idx
                break
        if seed is None:
            break

        cluster = [seed]
        unclustered.remove(seed)
        total_area = areas[seed]

        # Grow cluster
        while True:
            if total_area >= area_max:
                break

            # Find nearest unclustered within distance threshold
            best_idx = None
            best_dist = float('inf')

            # Use cluster centroid for distance
            cluster_coords = coords[cluster]
            centroid = cluster_coords.mean(axis=0)

            for idx in unclustered:
                dist = np.linalg.norm(coords[idx] - centroid)
                if dist < best_dist and dist <= dist_threshold:
                    best_dist = dist
                    best_idx = idx

            if best_idx is None:
                break

            # Closer-to-midpoint rule
            candidate_area = areas[best_idx]
            new_total = total_area + candidate_area
            if abs(new_total - midpoint) <= abs(total_area - midpoint):
                cluster.append(best_idx)
                unclustered.remove(best_idx)
                total_area = new_total
            else:
                # Adding would overshoot beyond midpoint benefit - stop this cluster
                break

        # Only keep as cluster if >1 member
        if len(cluster) > 1:
            clusters.append([indices[i] for i in cluster])
        else:
            # Seed couldn't grow - mark as failed and return to unclustered
            unclustered.add(cluster[0])
            failed_seeds.add(cluster[0])

    remaining = [indices[i] for i in unclustered]
    return clusters, remaining


def two_stage_clustering(
    detections: List[Dict],
    pixel_size: float,
    area_min: float = 375.0,
    area_max: float = 425.0,
    dist_round1: float = 500.0,
    dist_round2: float = 1000.0,
    min_score: float = 0.5,
) -> Dict:
    """
    Orchestrate two-round clustering.

    Returns dict with clusters, outliers, summary, and parameters.
    """
    # Filter detections by score
    filtered_indices = []
    for i, det in enumerate(detections):
        score = det.get('rf_prediction', det.get('score', 0))
        if score is None:
            score = 0
        if score >= min_score:
            filtered_indices.append(i)

    if not filtered_indices:
        return {
            'parameters': {
                'area_range': [area_min, area_max],
                'distance_thresholds': [dist_round1, dist_round2],
                'min_score': min_score,
                'pixel_size_um': pixel_size,
            },
            'summary': {'n_clusters': 0, 'n_singles': 0, 'n_nmjs_in_clusters': 0, 'n_total_filtered': 0},
            'main_clusters': [],
            'outliers': [],
        }

    # Compute coordinates (in um) and areas
    coords_um = []
    areas_um2 = []
    valid_indices = []

    for i in filtered_indices:
        det = detections[i]
        center = get_nmj_center(det)
        if center is None:
            continue
        area = get_nmj_area_um2(det, pixel_size)
        coords_um.append((center[0] * pixel_size, center[1] * pixel_size))
        areas_um2.append(area)
        valid_indices.append(i)

    if not valid_indices:
        return {
            'parameters': {
                'area_range': [area_min, area_max],
                'distance_thresholds': [dist_round1, dist_round2],
                'min_score': min_score,
                'pixel_size_um': pixel_size,
            },
            'summary': {'n_clusters': 0, 'n_singles': 0, 'n_nmjs_in_clusters': 0, 'n_total_filtered': 0},
            'main_clusters': [],
            'outliers': [],
        }

    coords_arr = np.array(coords_um)
    areas_arr = np.array(areas_um2)

    # Local indices for clustering (0..N-1 within valid set)
    local_indices = list(range(len(valid_indices)))

    # Round 1: tight distance
    print(f"  Round 1: dist_threshold={dist_round1} um ...")
    clusters_r1, remaining_r1 = cluster_nmjs_greedy_area(
        local_indices, coords_arr, areas_arr,
        dist_threshold=dist_round1, area_min=area_min, area_max=area_max,
    )
    print(f"    Clusters: {len(clusters_r1)}, Remaining: {len(remaining_r1)}")

    # Round 2: looser distance on remaining
    print(f"  Round 2: dist_threshold={dist_round2} um ...")
    if remaining_r1:
        remaining_coords = coords_arr[remaining_r1]
        remaining_areas = areas_arr[remaining_r1]
        clusters_r2_local, remaining_r2_local = cluster_nmjs_greedy_area(
            list(range(len(remaining_r1))), remaining_coords, remaining_areas,
            dist_threshold=dist_round2, area_min=area_min, area_max=area_max,
        )
        # Map back to valid_indices
        clusters_r2 = [[remaining_r1[j] for j in cl] for cl in clusters_r2_local]
        remaining_final = [remaining_r1[j] for j in remaining_r2_local]
    else:
        clusters_r2 = []
        remaining_final = []

    print(f"    Clusters: {len(clusters_r2)}, Remaining: {len(remaining_final)}")

    all_clusters = clusters_r1 + clusters_r2

    # Build output
    main_clusters = []
    for cid, cluster_local_idxs in enumerate(all_clusters):
        member_global_idxs = [valid_indices[li] for li in cluster_local_idxs]
        cluster_coords = coords_arr[cluster_local_idxs]
        cluster_areas = areas_arr[cluster_local_idxs]
        cx, cy = cluster_coords.mean(axis=0)

        main_clusters.append({
            'id': cid,
            'nmj_indices': member_global_idxs,
            'cx': float(cx / pixel_size),  # Back to pixel coords for consistency
            'cy': float(cy / pixel_size),
            'n': len(member_global_idxs),
            'total_area_um2': float(cluster_areas.sum()),
        })

    outliers = [{'nmj_index': valid_indices[li]} for li in remaining_final]

    n_in_clusters = sum(c['n'] for c in main_clusters)

    result = {
        'parameters': {
            'area_range': [area_min, area_max],
            'distance_thresholds': [dist_round1, dist_round2],
            'min_score': min_score,
            'pixel_size_um': pixel_size,
        },
        'summary': {
            'n_clusters': len(main_clusters),
            'n_singles': len(outliers),
            'n_nmjs_in_clusters': n_in_clusters,
            'n_total_filtered': len(valid_indices),
        },
        'main_clusters': main_clusters,
        'outliers': outliers,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Cluster NMJ detections for LMD well assignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
    python scripts/cluster_nmjs.py \\
        --detections nmj_detections.json \\
        --output nmj_clusters.json
''',
    )
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to detections JSON')
    parser.add_argument('--pixel-size', type=float, default=0.1725,
                        help='Pixel size in um (default: 0.1725)')
    parser.add_argument('--area-min', type=float, default=375.0,
                        help='Cluster area lower bound in um2 (default: 375)')
    parser.add_argument('--area-max', type=float, default=425.0,
                        help='Cluster area upper bound in um2 (default: 425)')
    parser.add_argument('--dist-round1', type=float, default=500.0,
                        help='Distance threshold for round 1 in um (default: 500)')
    parser.add_argument('--dist-round2', type=float, default=1000.0,
                        help='Distance threshold for round 2 in um (default: 1000)')
    parser.add_argument('--min-score', type=float, default=0.5,
                        help='Minimum rf_prediction score (default: 0.5)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for clusters JSON')

    args = parser.parse_args()

    # Load detections
    print(f"Loading detections from: {args.detections}")
    with open(args.detections, 'r') as f:
        detections = json.load(f)
    print(f"  Total detections: {len(detections)}")

    # Run clustering
    print(f"\nClustering (area target: {args.area_min}-{args.area_max} um2)...")
    result = two_stage_clustering(
        detections,
        pixel_size=args.pixel_size,
        area_min=args.area_min,
        area_max=args.area_max,
        dist_round1=args.dist_round1,
        dist_round2=args.dist_round2,
        min_score=args.min_score,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Summary
    s = result['summary']
    print(f"\nResults:")
    print(f"  Filtered NMJs (score >= {args.min_score}): {s['n_total_filtered']}")
    print(f"  Clusters: {s['n_clusters']}")
    print(f"  NMJs in clusters: {s['n_nmjs_in_clusters']}")
    print(f"  Singles (outliers): {s['n_singles']}")

    if result['main_clusters']:
        areas = [c['total_area_um2'] for c in result['main_clusters']]
        sizes = [c['n'] for c in result['main_clusters']]
        print(f"\n  Cluster areas: {min(areas):.1f} - {max(areas):.1f} um2 "
              f"(mean {np.mean(areas):.1f}, median {np.median(areas):.1f})")
        print(f"  Cluster sizes: {min(sizes)} - {max(sizes)} NMJs "
              f"(mean {np.mean(sizes):.1f})")

    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
