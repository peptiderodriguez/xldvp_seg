"""Shared graph topology analysis for spatial structure detection.

Provides reusable metrics for classifying connected components of cells as
rings, arcs, strips, or clusters.  Used by both
``scripts/detect_curvilinear_patterns.py`` (mesothelium strip detection) and
``scripts/detect_vessel_structures.py`` (vessel ring/arc/strip detection).

Graph construction uses scipy.sparse for scalability (handles 100K+ cells),
with NetworkX only for per-component topology analysis on small subgraphs.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sparse_cc
from scipy.spatial import ConvexHull, cKDTree

from segmentation.utils.logging import get_logger

__all__ = [
    "build_radius_graph_sparse",
    "build_component_subgraph",
    "double_bfs_diameter",
    "component_linearity",
    "component_width",
    "path_length_um",
    "ring_score",
    "arc_fraction",
    "elongation",
    "circularity",
    "hollowness",
    "has_curvature",
    "safe_hull_area",
    "bounding_box_aspect_ratio",
    "compute_all_metrics",
]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_radius_graph_sparse(
    positions: np.ndarray, radius: float
) -> tuple[int, np.ndarray, set[tuple[int, int]]]:
    """Build a radius graph using scipy sparse matrix and find connected components.

    Args:
        positions: (N, 2) array of cell positions in µm.
        radius: connection radius in µm.

    Returns:
        n_comp: number of connected components.
        labels: (N,) array of component labels per cell.
        pairs: set of (i, j) edge pairs (for building nx subgraphs later).
    """
    n = len(positions)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=radius)
    logger.info("  KD-tree: %d edges from %d nodes (r=%.0f µm)", len(pairs), n, radius)

    if not pairs:
        return n, np.arange(n), set()

    rows, cols = zip(*pairs)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj + adj.T

    n_comp, labels = sparse_cc(adj, directed=False)
    return n_comp, labels, pairs


def build_component_subgraph(pairs: set[tuple[int, int]], comp_nodes: set[int] | list[int]):
    """Build a NetworkX subgraph for a single connected component.

    Only imports networkx when called (lazy import for the scipy-only fast path).

    Args:
        pairs: set of (i, j) edge pairs from the full graph.
        comp_nodes: set or array of node indices in this component.

    Returns:
        nx.Graph for this component.
    """
    import networkx as nx

    comp_set = set(comp_nodes) if not isinstance(comp_nodes, set) else comp_nodes
    G = nx.Graph()
    G.add_nodes_from(comp_set)
    for i, j in pairs:
        if i in comp_set and j in comp_set:
            G.add_edge(i, j)
    return G


# ---------------------------------------------------------------------------
# Strip metrics (from detect_curvilinear_patterns.py)
# ---------------------------------------------------------------------------


def double_bfs_diameter(G, comp_nodes: set[int] | list[int]) -> tuple[int, list[int]]:
    """Compute pseudo-diameter of a component via double-BFS.

    Returns (diameter, path_nodes) where diameter is the number of hops and
    path_nodes is the list of node indices along the diameter path.

    Exact for trees, tight lower-bound for general graphs.
    """
    import networkx as nx

    subgraph = G.subgraph(comp_nodes)
    start = min(comp_nodes)  # deterministic for reproducibility
    far1_dist = nx.single_source_shortest_path_length(subgraph, start)
    far1_node = max(far1_dist, key=far1_dist.get)
    far2_paths = nx.single_source_shortest_path(subgraph, far1_node)
    far2_node = max(far2_paths, key=lambda n: len(far2_paths[n]))
    diameter = len(far2_paths[far2_node]) - 1
    path_nodes = far2_paths[far2_node]
    return diameter, path_nodes


def component_linearity(diameter: int, n_nodes: int) -> float:
    """Linearity score: graph_diameter / sqrt(n_nodes).

    High for strips/linear structures, low for compact blobs.
    """
    if n_nodes <= 0:
        return 0.0
    return diameter / np.sqrt(n_nodes)


def component_width(
    positions: np.ndarray, comp_nodes: list[int], path: list[int], percentile: int = 95
) -> float:
    """Perpendicular half-width of a component relative to its diameter path.

    Computes the distance from each cell to the nearest segment of the diameter
    path polyline, then returns the given percentile (default 95th). This is the
    **half-width** (radius from centerline), not the full width. For a symmetric
    structure of full width W, this returns ~W/2.

    When using ``--max-strip-width``, set the threshold to the expected half-width.

    Vectorized: O(C * P) numpy operations, no Python inner loops.
    """
    if len(path) < 2:
        return 0.0
    pts = np.asarray(positions[list(comp_nodes)], dtype=np.float64)
    path_arr = np.asarray(path)
    seg_a = positions[path_arr[:-1]]  # (P, 2)
    seg_b = positions[path_arr[1:]]  # (P, 2)
    ab = seg_b - seg_a  # (P, 2)
    ab_sq = np.sum(ab * ab, axis=1)  # (P,)
    # Broadcast: (C, 1, 2) vs (1, P, 2)
    ap = pts[:, None, :] - seg_a[None, :, :]  # (C, P, 2)
    t = np.clip(
        np.sum(ap * ab[None, :, :], axis=2) / np.maximum(ab_sq[None, :], 1e-12),
        0.0,
        1.0,
    )  # (C, P)
    proj = seg_a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (C, P, 2)
    dists = np.sqrt(np.sum((pts[:, None, :] - proj) ** 2, axis=2))  # (C, P)
    min_dists = dists.min(axis=1)  # (C,) — min distance to any segment
    return float(np.percentile(min_dists, percentile))


def path_length_um(positions: np.ndarray, path_nodes: list[int]) -> float:
    """Physical length along a diameter path in µm."""
    if len(path_nodes) < 2:
        return 0.0
    path_arr = np.asarray(path_nodes)
    diffs = positions[path_arr[1:]] - positions[path_arr[:-1]]
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


# ---------------------------------------------------------------------------
# Ring / arc metrics — graph topology (NEW)
# ---------------------------------------------------------------------------


def ring_score(positions: np.ndarray, G, comp_nodes: set[int] | list[int]) -> float:
    """Angular connectivity score: do cells form a connected ring?

    Orders cells by angle from centroid and checks if angularly-adjacent
    cells are also connected in the graph.

    Returns 0.0 if the component's bounding-box aspect ratio > 3 (angular
    ordering is unreliable for very elongated shapes).

    Returns:
        float in [0, 1]. Ring ~0.7-1.0, blob ~0.3-0.5, strip ~0.1-0.3.
    """
    comp_list = list(comp_nodes)
    pts = positions[comp_list]
    n = len(pts)
    if n < 4:
        return 0.0

    # Aspect ratio guard — angular ordering fails for elongated ellipses
    ar = bounding_box_aspect_ratio(pts)
    if ar > 3.0:
        return 0.0

    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)

    # Map component node IDs to their position in comp_list
    # (G uses the original node indices)
    connected = 0
    for i in range(n):
        node_a = comp_list[order[i]]
        node_b = comp_list[order[(i + 1) % n]]
        if G.has_edge(node_a, node_b):
            connected += 1

    return connected / n


def arc_fraction(positions: np.ndarray, G, comp_nodes: set[int] | list[int]) -> float:
    """Max contiguous arc: longest run of angularly-connected pairs / n.

    Handles gaps in rings (e.g., CD31/LYVE1 gaps in vessel wall).
    Returns 0.0 if aspect ratio > 3 (angular ordering unreliable for
    very elongated shapes — same guard as ring_score).

    Returns:
        float in [0, 1]. Full ring ~1.0, half-ring ~0.5, quarter ~0.25.
    """
    comp_list = list(comp_nodes)
    pts = positions[comp_list]
    n = len(pts)
    if n < 4:
        return 0.0

    # Same AR guard as ring_score — angular ordering breaks for elongated shapes
    ar = bounding_box_aspect_ratio(pts)
    if ar > 3.0:
        return 0.0

    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)

    # Build binary array: 1 if angularly-adjacent pair is graph-connected
    connected = np.zeros(n, dtype=bool)
    for i in range(n):
        node_a = comp_list[order[i]]
        node_b = comp_list[order[(i + 1) % n]]
        connected[i] = G.has_edge(node_a, node_b)

    # Find longest contiguous run of True (wrapping around)
    if connected.all():
        return 1.0

    # Double the array to handle wrap-around
    doubled = np.concatenate([connected, connected])
    max_run = 0
    current_run = 0
    for v in doubled:
        if v:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    # Cap at n (full circle)
    return min(max_run, n) / n


# ---------------------------------------------------------------------------
# Ring / arc metrics — geometric / PCA (from vessel_community_analysis.py)
# ---------------------------------------------------------------------------


def elongation(pts: np.ndarray) -> float:
    """PCA elongation: sqrt(λ1 / λ2).

    High for strips/linear, low for rings/blobs.
    """
    if len(pts) < 3:
        return 1.0
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    lam1 = max(eigvals[0], 1e-10)
    lam2 = max(eigvals[1], 1e-10)
    return float(np.sqrt(lam1 / lam2))


def circularity(pts: np.ndarray) -> float:
    """Radial distance uniformity: 1 - std(radii) / mean(radii).

    High for rings (cells at ~same distance from centroid).
    """
    if len(pts) < 3:
        return 0.0
    centroid = pts.mean(axis=0)
    radii = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))
    mean_r = radii.mean()
    if mean_r < 1e-6:
        return 0.0
    return float(max(0.0, 1.0 - radii.std() / mean_r))


def hollowness(pts: np.ndarray) -> float:
    """Hollowness: median(radii) / max(radii).

    High for rings (cells concentrated on perimeter), low for filled blobs.
    """
    if len(pts) < 3:
        return 0.0
    centroid = pts.mean(axis=0)
    radii = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))
    max_r = radii.max()
    if max_r < 1e-6:
        return 0.0
    return float(np.median(radii) / max_r)


def has_curvature(pts: np.ndarray) -> bool:
    """Detect curvature via 2nd-order polynomial fit in PCA space.

    Returns True if the component is elongated (elongation > 2.5) and a
    quadratic fit explains significant variance (R² > 0.3).
    """
    n = len(pts)
    if n < 10:
        return False  # need ≥10 points for reliable quadratic fit (3 params)

    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals_raw, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals_raw[::-1]  # descending
    lam1 = max(eigvals[0], 1e-10)
    lam2 = max(eigvals[1], 1e-10)
    elong = np.sqrt(lam1 / lam2)

    if elong <= 2.5:
        return False

    pc1 = eigvecs[:, -1]
    pc2 = eigvecs[:, -2]
    proj1 = centered @ pc1
    proj2 = centered @ pc2
    coeffs = np.polyfit(proj1, proj2, 2)
    pred = np.polyval(coeffs, proj1)
    ss_res = ((proj2 - pred) ** 2).sum()
    ss_tot = ((proj2 - proj2.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return bool(r2 > 0.3 and abs(coeffs[0]) > 1e-6)


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def safe_hull_area(pts: np.ndarray) -> float:
    """Convex hull area, returning 0 for degenerate cases."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # 2D: volume = area
    except Exception:
        return 0.0


def bounding_box_aspect_ratio(pts: np.ndarray) -> float:
    """Aspect ratio from PCA-aligned bounding box (major / minor extent).

    Used as a guard for ring_score — angular ordering is unreliable when AR > 3.
    """
    if len(pts) < 3:
        return 1.0
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvecs = np.linalg.eigh(cov)[1]
    # Project onto principal axes
    proj = centered @ eigvecs
    extents = proj.max(axis=0) - proj.min(axis=0)
    minor = max(extents.min(), 1e-10)
    major = extents.max()
    return float(major / minor)


def compute_all_metrics(positions: np.ndarray, G, comp_nodes: set[int] | list[int]) -> dict:
    """Compute all topology + geometric metrics for a single component.

    Convenience function that returns a dict with all metrics.
    Requires a NetworkX subgraph (build with ``build_component_subgraph``).

    Returns:
        dict with keys: linearity, diameter, length_um, width_um,
        ring_score, arc_fraction, elongation, circularity, hollowness,
        has_curvature, hull_area_um2, aspect_ratio, n_cells.
    """
    comp_list = list(comp_nodes)
    pts = positions[comp_list]
    n = len(comp_list)

    # Graph topology
    diam, path = double_bfs_diameter(G, comp_nodes)
    lin = component_linearity(diam, n)
    length = path_length_um(positions, path)
    width = component_width(positions, comp_list, path)

    # Ring / arc — graph
    rs = ring_score(positions, G, comp_nodes)
    af = arc_fraction(positions, G, comp_nodes)

    # Ring / arc — geometric / PCA
    elong = elongation(pts)
    circ = circularity(pts)
    hollow = hollowness(pts)
    curv = has_curvature(pts)

    # Geometry
    hull = safe_hull_area(pts)
    ar = bounding_box_aspect_ratio(pts)

    return {
        "n_cells": n,
        "diameter": diam,
        "length_um": round(length, 1),
        "width_um": round(width, 1),
        "linearity": round(lin, 3),
        "ring_score": round(rs, 3),
        "arc_fraction": round(af, 3),
        "elongation": round(elong, 3),
        "circularity": round(circ, 3),
        "hollowness": round(hollow, 3),
        "has_curvature": curv,
        "hull_area_um2": round(hull, 1),
        "aspect_ratio": round(ar, 3),
    }
