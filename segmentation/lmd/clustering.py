"""
Biological clustering of detections for LMD well assignment.

Two-stage greedy spatial clustering with area constraint:
  Round 1 (500 um): Tight spatial groups
  Round 2 (1000 um): Remaining detections with looser distance

Cluster target: total detection area per cluster = 375-425 um2 (midpoint 400).
Overshoot rule: add detection only if |total + area - 400| < |total - 400|.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Usage:
    from segmentation.lmd.clustering import two_stage_clustering

    result = two_stage_clustering(detections, pixel_size=0.1725)
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Dict, Tuple, Optional

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def get_detection_area_um2(det: Dict, pixel_size: float) -> float:
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


def get_detection_center(det: Dict) -> Optional[Tuple[float, float]]:
    """
    Get (x, y) center coordinates from detection, in PIXELS.

    Returns pixel coordinates from 'global_center' or 'center' fields.
    Callers (e.g. two_stage_clustering) are responsible for converting
    to microns by multiplying by pixel_size before distance calculations.
    """
    if 'global_center' in det:
        gc = det['global_center']
        return (float(gc[0]), float(gc[1]))
    if 'center' in det:
        c = det['center']
        return (float(c[0]), float(c[1]))
    return None


def cluster_detections_greedy_area(
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

    # Build spatial index for O(log N) nearest-neighbor queries
    tree = cKDTree(coords)

    # Sort by position (top-left to bottom-right) for deterministic seeding
    sort_order = np.lexsort((coords[:, 1], coords[:, 0]))

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
        cluster_centroid = coords[seed].copy()
        centroid_weight = areas[seed]

        # Grow cluster
        while True:
            if total_area >= area_max:
                break

            # Find nearest unclustered within distance threshold using cKDTree
            candidates = tree.query_ball_point(cluster_centroid, dist_threshold)
            best_idx = None
            best_dist = float('inf')
            for idx in candidates:
                if idx not in unclustered:
                    continue
                dist = np.linalg.norm(coords[idx] - cluster_centroid)
                if dist < best_dist:
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
                # Area-weighted centroid update — heavier detections pull
                # the centroid more, producing shorter LMD laser paths.
                # Uses running centroid_weight to avoid O(n) re-sum per step
                # and prevent floating-point drift from repeated summation.
                new_weight = centroid_weight + areas[best_idx]
                if new_weight > 0:
                    cluster_centroid = (cluster_centroid * centroid_weight + coords[best_idx] * areas[best_idx]) / new_weight
                centroid_weight = new_weight
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
            'summary': {'n_clusters': 0, 'n_singles': 0, 'n_detections_in_clusters': 0, 'n_total_filtered': 0},
            'main_clusters': [],
            'outliers': [],
        }

    # Compute coordinates (in um) and areas
    coords_um = []
    areas_um2 = []
    valid_indices = []

    for i in filtered_indices:
        det = detections[i]
        center = get_detection_center(det)
        if center is None:
            continue
        area = get_detection_area_um2(det, pixel_size)
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
            'summary': {'n_clusters': 0, 'n_singles': 0, 'n_detections_in_clusters': 0, 'n_total_filtered': 0},
            'main_clusters': [],
            'outliers': [],
        }

    coords_arr = np.array(coords_um)
    areas_arr = np.array(areas_um2)

    # Local indices for clustering (0..N-1 within valid set)
    local_indices = list(range(len(valid_indices)))

    # Round 1: tight distance
    logger.info(f"  Round 1: dist_threshold={dist_round1} um ...")
    clusters_r1, remaining_r1 = cluster_detections_greedy_area(
        local_indices, coords_arr, areas_arr,
        dist_threshold=dist_round1, area_min=area_min, area_max=area_max,
    )
    logger.info(f"    Clusters: {len(clusters_r1)}, Remaining: {len(remaining_r1)}")

    # Round 2: looser distance on remaining
    logger.info(f"  Round 2: dist_threshold={dist_round2} um ...")
    if remaining_r1:
        remaining_coords = coords_arr[remaining_r1]
        remaining_areas = areas_arr[remaining_r1]
        clusters_r2_local, remaining_r2_local = cluster_detections_greedy_area(
            list(range(len(remaining_r1))), remaining_coords, remaining_areas,
            dist_threshold=dist_round2, area_min=area_min, area_max=area_max,
        )
        # Map back to valid_indices
        clusters_r2 = [[remaining_r1[j] for j in cl] for cl in clusters_r2_local]
        remaining_final = [remaining_r1[j] for j in remaining_r2_local]
    else:
        clusters_r2 = []
        remaining_final = []

    logger.info(f"    Clusters: {len(clusters_r2)}, Remaining: {len(remaining_final)}")

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
            'detection_indices': member_global_idxs,
            'cx': float(cx / pixel_size),  # Back to pixel coords for consistency
            'cy': float(cy / pixel_size),
            'n': len(member_global_idxs),
            'total_area_um2': float(cluster_areas.sum()),
        })

    outliers = [{'detection_index': valid_indices[li]} for li in remaining_final]

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
            'n_detections_in_clusters': n_in_clusters,
            'n_total_filtered': len(valid_indices),
        },
        'main_clusters': main_clusters,
        'outliers': outliers,
    }

    return result
