"""Sliding window spatial sampling along ROI skeleton paths for LMD well planning."""

import numpy as np
from scipy.spatial import cKDTree

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def compute_skeleton_paths(poly, resolution=1.0, min_path_length=20):
    """Compute ordered paths along the morphological skeleton of a polygon.

    Args:
        poly: Shapely Polygon.
        resolution: Rasterization resolution in same units as polygon coords.
        min_path_length: Minimum skeleton path length in pixels to keep.

    Returns:
        List of Nx2 arrays, each an ordered path in polygon coordinate space.
    """
    from shapely import contains_xy
    from skimage.morphology import skeletonize

    xmin, ymin, xmax, ymax = poly.bounds
    margin = 5
    w = int((xmax - xmin) / resolution) + 2 * margin
    h = int((ymax - ymin) / resolution) + 2 * margin
    yy, xx = np.mgrid[0:h, 0:w]
    px = xx * resolution + xmin - margin * resolution
    py = yy * resolution + ymin - margin * resolution
    mask = contains_xy(poly, px.ravel(), py.ravel()).reshape(h, w)

    skel = skeletonize(mask)
    skel_coords = np.argwhere(skel)
    if len(skel_coords) == 0:
        return []

    skel_um = np.column_stack(
        [
            skel_coords[:, 1] * resolution + xmin - margin * resolution,
            skel_coords[:, 0] * resolution + ymin - margin * resolution,
        ]
    )

    # Build adjacency and trace paths from endpoints
    skel_tree = cKDTree(skel_um)
    nbrs = skel_tree.query_ball_tree(skel_tree, r=resolution * 1.5)
    degree = np.array([len(n) - 1 for n in nbrs])
    endpoints = np.where(degree == 1)[0]

    def trace(start):
        path = [start]
        visited_local.add(start)
        cur = start
        while True:
            nxt = [n for n in nbrs[cur] if n != cur and n not in visited_local]
            if not nxt:
                break
            cur = nxt[0]
            path.append(cur)
            visited_local.add(cur)
        return path

    visited_local = set()
    paths = []
    for ep in endpoints:
        if ep not in visited_local:
            p = trace(ep)
            if len(p) >= min_path_length:
                paths.append(skel_um[p])

    # Trace from remaining junctions
    junctions = np.where(degree >= 3)[0]
    for jp in junctions:
        if jp not in visited_local:
            p = trace(jp)
            if len(p) >= min_path_length:
                paths.append(skel_um[p])

    return paths


def place_windows_along_paths(paths, radius, step):
    """Place window centers at regular intervals along skeleton paths.

    Args:
        paths: List of Nx2 arrays from compute_skeleton_paths.
        radius: Window radius in coordinate units.
        step: Distance between window centers along path.

    Returns:
        Nx2 array of window center coordinates.
    """
    centers = []
    for pts in paths:
        diffs = np.diff(pts, axis=0)
        arc = np.concatenate([[0], np.cumsum(np.sqrt((diffs**2).sum(axis=1)))])
        total = arc[-1]
        for d in np.arange(radius, total - radius / 2, step):
            idx = min(np.searchsorted(arc, d), len(pts) - 1)
            centers.append(pts[idx])
    return np.array(centers) if centers else np.empty((0, 2))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def spatially_balanced_sample(positions, areas, center, target_lo, target_hi, min_cells=10):
    """Farthest-point area-matched sampling within a window.

    Starts from the cell closest to center, then iteratively adds the cell
    farthest from all selected cells, accumulating area until the target range
    is reached. Guarantees spatial spread across the window.

    Args:
        positions: Nx2 array of cell positions (only available/unclaimed cells).
        areas: Length-N array of cell areas.
        center: (x, y) window center.
        target_lo: Minimum total area.
        target_hi: Maximum total area.
        min_cells: Minimum cells required before accepting the sample.

    Returns:
        (selected_indices, total_area) or ([], 0.0) if target can't be met.
    """
    if len(positions) == 0 or areas.sum() < target_lo:
        return [], 0.0

    dists_to_center = np.sqrt(((positions - center) ** 2).sum(axis=1))
    start = np.argmin(dists_to_center)
    selected = [start]
    selected_set = {start}
    running = areas[start]

    # Incremental min-distance tracking: update only with the newly added point
    min_dists = np.sqrt(((positions - positions[start]) ** 2).sum(axis=1))
    min_dists[start] = -1

    while running < target_hi and len(selected) < len(positions):
        candidates = np.where(min_dists > 0)[0]
        if len(candidates) == 0:
            break

        # Try candidates by distance (farthest first), skip those too large
        sorted_cands = candidates[np.argsort(-min_dists[candidates])]
        added = False
        for ci in sorted_cands:
            if running + areas[ci] <= target_hi:
                selected.append(ci)
                selected_set.add(ci)
                running += areas[ci]
                added = True
                # Incrementally update min distances with new point
                new_dists = np.sqrt(((positions - positions[ci]) ** 2).sum(axis=1))
                min_dists = np.minimum(min_dists, new_dists)
                min_dists[ci] = -1
                break
        if not added:
            # No candidate fits — all remaining cells would exceed target_hi
            break

        if running >= target_lo and len(selected) >= min_cells:
            break

    if target_lo <= running <= target_hi:
        return list(selected), float(running)
    return [], 0.0


def run_sampling(roi_positions, roi_areas, roi_indices, centers, radius, target_lo, target_hi):
    """Run greedy window sampling along centerline positions.

    Args:
        roi_positions: Nx2 positions of cells inside ROI.
        roi_areas: Length-N areas.
        roi_indices: Length-N global detection indices.
        centers: Mx2 window center positions.
        radius: Window radius.
        target_lo: Min area per window.
        target_hi: Max area per window.

    Returns:
        (windows, n_rejected) where windows is a list of dicts.
    """
    tree = cKDTree(roi_positions)
    pools = {
        wi: list(tree.query_ball_point([cx, cy], r=radius)) for wi, (cx, cy) in enumerate(centers)
    }

    used = set()
    windows = []
    n_rejected = 0
    remaining = set(range(len(centers)))

    # Cache available area per window (updated after each assignment)
    avail_area_cache = {wi: sum(roi_areas[i] for i in pools[wi]) for wi in remaining}

    while remaining:
        # Pick window with most available area (O(n) lookup, not O(n*m))
        best = max(remaining, key=lambda wi: avail_area_cache.get(wi, 0))
        avail = [i for i in pools[best] if i not in used]
        avail_areas = roi_areas[avail]

        if avail_areas.sum() < target_lo:
            n_rejected += 1
            remaining.discard(best)
            continue

        avail_pos = roi_positions[avail]
        picked_local, running = spatially_balanced_sample(
            avail_pos, avail_areas, centers[best], target_lo, target_hi
        )

        if picked_local:
            picked_roi = [avail[i] for i in picked_local]
            used.update(picked_roi)
            windows.append(
                {
                    "window_id": len(windows),
                    "center_x": float(centers[best][0]),
                    "center_y": float(centers[best][1]),
                    "n_cells": len(picked_roi),
                    "total_area_um2": round(running, 1),
                    "cell_indices": [int(roi_indices[i]) for i in picked_roi],
                }
            )
            # Update cache for remaining windows that share claimed cells
            claimed = set(picked_roi)
            for wi in remaining:
                if wi != best:
                    overlap = claimed & set(pools[wi])
                    if overlap:
                        avail_area_cache[wi] -= sum(roi_areas[i] for i in overlap)
        else:
            n_rejected += 1
        remaining.discard(best)
        avail_area_cache.pop(best, None)

    # Sort by x position for consistent ordering
    windows.sort(key=lambda w: w["center_x"])
    for i, w in enumerate(windows):
        w["window_id"] = i

    return windows, n_rejected


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def grid_search(
    roi_positions,
    roi_areas,
    roi_indices,
    paths,
    target_lo,
    target_hi,
    n_cells,
    radii=None,
    overlaps=None,
):
    """Search (radius, overlap) space for zero-rejection configurations.

    Args:
        roi_positions, roi_areas, roi_indices: Cell data inside ROI.
        paths: Skeleton paths.
        target_lo, target_hi: Area target range.
        n_cells: Total cells in ROI.
        radii: List of radii to try (um).
        overlaps: List of overlap fractions to try.

    Returns:
        List of result dicts, sorted by coverage descending.
    """
    if radii is None:
        radii = list(range(40, 105, 5))
    if overlaps is None:
        overlaps = [1 / 3, 0.4, 0.5, 0.6, 2 / 3]

    results = []
    for radius in radii:
        for ov in overlaps:
            step = 2 * radius * (1 - ov)
            centers = place_windows_along_paths(paths, radius, step)
            if len(centers) == 0:
                continue

            windows, n_rej = run_sampling(
                roi_positions, roi_areas, roi_indices, centers, radius, target_lo, target_hi
            )
            n_sampled = sum(w["n_cells"] for w in windows)
            results.append(
                {
                    "radius_um": radius,
                    "overlap_pct": round(ov * 100),
                    "step_um": round(step),
                    "n_placed": len(centers),
                    "n_filled": len(windows),
                    "n_rejected": n_rej,
                    "n_sampled": n_sampled,
                    "coverage_pct": round(n_sampled / n_cells * 100, 1),
                }
            )

    results.sort(key=lambda r: (-r["coverage_pct"], r["n_rejected"]))
    return results
