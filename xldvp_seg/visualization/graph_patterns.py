"""Spatial graph pattern detection for viewer region overlays."""

import numpy as np

from xldvp_seg.utils.logging import get_logger

from .data_loading import compute_auto_eps  # noqa: F401 — re-export for convenience

logger = get_logger(__name__)


def compute_graph_patterns(
    positions,
    types,
    type_labels,
    type_colors,
    connect_radius_um=150,
    min_cluster_cells=8,
    boundary_dilate_um=50,
    _cached_trees=None,
):
    """Detect spatial patterns via graph-based connected components.

    Per type: KDTree -> connect cells within connect_radius_um, connected
    components -> discrete clusters, classify pattern (linear/arc/ring/cluster),
    boundary via rasterise -> dilate -> findContours -> RDP simplify.

    Args:
        positions: (N, 2) array of cell positions in um.
        types: (N,) array of integer type indices.
        type_labels: List of type label strings.
        type_colors: List of hex color strings per type.
        connect_radius_um: Radius for connecting cells (default 150 um).
        min_cluster_cells: Minimum cells per cluster (default 8).
        boundary_dilate_um: Dilation radius for boundary (default 50 um).
        _cached_trees: Optional dict {type_index: (points, cKDTree)} to reuse
            across multiple radii. Pass the same dict for each call.

    Returns:
        List of region dicts with boundary polygons, composition, pattern.
    """
    import cv2
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(positions)
    if n == 0:
        return []

    unique_types = np.unique(types)
    regions = []

    for ti in unique_types:
        type_mask = types == ti
        n_type = int(type_mask.sum())
        idx = int(ti)
        label = type_labels[idx] if idx < len(type_labels) else f"type_{idx}"
        color = type_colors[idx] if idx < len(type_colors) else "#888888"

        if n_type < min_cluster_cells:
            continue

        # Reuse KDTree across radii if cached
        if _cached_trees is not None and idx in _cached_trees:
            tp, tree = _cached_trees[idx]
        else:
            tp = positions[type_mask]  # (n_type, 2)
            tree = cKDTree(tp)
            if _cached_trees is not None:
                _cached_trees[idx] = (tp, tree)

        pairs = tree.query_pairs(r=connect_radius_um)

        if not pairs:
            continue

        rows, cols = zip(*pairs)
        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix((data, (rows, cols)), shape=(n_type, n_type))
        adj = adj + adj.T

        n_components, comp_labels = connected_components(adj, directed=False)

        for ci in range(n_components):
            cmask = comp_labels == ci
            nc = int(cmask.sum())
            if nc < min_cluster_cells:
                continue

            pts = tp[cmask]
            cx_mean = pts[:, 0].mean()
            cy_mean = pts[:, 1].mean()

            # Pattern classification via PCA
            centered = pts - pts.mean(axis=0)
            cov = np.cov(centered.T) if nc > 2 else np.eye(2)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            lam1 = max(eigvals[0], 1e-10)
            lam2 = max(eigvals[1], 1e-10)
            elongation = np.sqrt(lam1 / lam2)

            # Circle fit
            radii = np.sqrt((pts[:, 0] - cx_mean) ** 2 + (pts[:, 1] - cy_mean) ** 2)
            mean_r = radii.mean()
            circularity = (1.0 - radii.std() / mean_r) if mean_r > 1e-6 else 0.0
            hollowness = np.median(radii) / max(radii.max(), 1e-6)

            # Curvature check
            has_curvature = False
            if nc > 5 and elongation > 2.5:
                eigvecs = np.linalg.eigh(cov)[1]
                pc1 = eigvecs[:, -1]
                pc2 = eigvecs[:, -2]
                proj1 = centered @ pc1
                proj2 = centered @ pc2
                coeffs = np.polyfit(proj1, proj2, 2)
                pred = np.polyval(coeffs, proj1)
                ss_res = ((proj2 - pred) ** 2).sum()
                ss_tot = ((proj2 - proj2.mean()) ** 2).sum()
                r2 = 1 - ss_res / max(ss_tot, 1e-10)
                if r2 > 0.3 and abs(coeffs[0]) > 1e-6:
                    has_curvature = True

            if elongation > 4 and not has_curvature:
                pattern = "linear"
            elif elongation > 3 and has_curvature:
                pattern = "arc"
            elif circularity > 0.65 and hollowness > 0.55 and elongation < 3:
                pattern = "ring"
            else:
                pattern = "cluster"

            # Boundary via rasterisation
            pad = boundary_dilate_um
            bx_min = pts[:, 0].min() - pad
            bx_max = pts[:, 0].max() + pad
            by_min = pts[:, 1].min() - pad
            by_max = pts[:, 1].max() + pad
            bw = bx_max - bx_min
            bh = by_max - by_min

            target_px = max(64, min(512, int(max(bw, bh) / 5)))
            if bw >= bh:
                rnx = target_px
                rny = max(1, int(target_px * bh / bw))
            else:
                rny = target_px
                rnx = max(1, int(target_px * bw / bh))

            rpx = bw / max(rnx, 1)
            rpy = bh / max(rny, 1)

            px = np.clip(((pts[:, 0] - bx_min) / bw * rnx).astype(int), 0, rnx - 1)
            py = np.clip(((pts[:, 1] - by_min) / bh * rny).astype(int), 0, rny - 1)
            raster = np.zeros((rny, rnx), dtype=np.uint8)
            raster[py, px] = 255

            dilate_px = max(2, int(connect_radius_um / rpx * 0.5))
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
            )
            raster = cv2.dilate(raster, kern)
            close_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_px + 1, dilate_px + 1)
            )
            raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE, close_kern)

            contours, _ = cv2.findContours(raster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            epsilon = max(1.0, 0.008 * cv2.arcLength(contour, True))
            contour = cv2.approxPolyDP(contour, epsilon, True)
            if len(contour) < 3:
                continue

            boundary = []
            for pt in contour.reshape(-1, 2):
                boundary.append(
                    {
                        "x": round(float(pt[0] * rpx + bx_min), 1),
                        "y": round(float(pt[1] * rpy + by_min), 1),
                    }
                )

            # Composition: count all cell types inside boundary
            cmask_img = np.zeros((rny, rnx), dtype=np.uint8)
            cv2.drawContours(cmask_img, [contour], 0, 255, -1)
            all_px = np.clip(((positions[:, 0] - bx_min) / bw * rnx).astype(int), 0, rnx - 1)
            all_py = np.clip(((positions[:, 1] - by_min) / bh * rny).astype(int), 0, rny - 1)
            inside_all = cmask_img[all_py, all_px] > 0
            n_inside_total = int(inside_all.sum())

            composition = {}
            for tj in unique_types:
                count = int((inside_all & (types == tj)).sum())
                if count > 0:
                    composition[type_labels[int(tj)]] = count

            if n_inside_total == 0:
                n_inside_total = nc
                composition = {label: nc}

            dominant = max(composition, key=composition.get)
            dominant_frac = composition[dominant] / max(n_inside_total, 1)

            # Normalize composition to fractions
            composition = {k: round(v / max(n_inside_total, 1), 3) for k, v in composition.items()}

            contour_area_px = cv2.contourArea(contour)
            area_um2 = round(contour_area_px * rpx * rpy, 0)

            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                mu20 = moments["mu20"] / moments["m00"]
                mu02 = moments["mu02"] / moments["m00"]
                mu11 = moments["mu11"] / moments["m00"]
                d = np.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)
                major = mu20 + mu02 + d
                minor = mu20 + mu02 - d
                cont_elong = round(np.sqrt(max(major, 1e-9) / max(minor, 1e-9)), 2)
            else:
                cont_elong = round(elongation, 2)

            regions.append(
                {
                    "id": len(regions),
                    "type": label,
                    "label": f"{label} ({pattern}, n={nc})",
                    "color": color,
                    "pattern": pattern,
                    "composition": composition,
                    "n_cells": n_inside_total,
                    "area_um2": area_um2,
                    "elongation": cont_elong,
                    "dominant_frac": round(dominant_frac, 3),
                    "boundary": boundary,
                }
            )

    # Sort by area descending, re-index
    regions.sort(key=lambda r: r["area_um2"], reverse=True)
    for i, r in enumerate(regions):
        r["id"] = i

    n_types_found = len(set(r["type"] for r in regions)) if regions else 0
    patterns_summary = {}
    for r in regions:
        p = r["pattern"]
        patterns_summary[p] = patterns_summary.get(p, 0) + 1
    pat_str = ", ".join(f"{v} {k}" for k, v in sorted(patterns_summary.items()))
    logger.info(
        "Graph patterns (r=%dum): %d regions from %d types (>=%d cells): %s",
        connect_radius_um,
        len(regions),
        n_types_found,
        min_cluster_cells,
        pat_str,
    )
    return regions
