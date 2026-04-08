"""Spatial pattern detection: classify cell clusters as strips, arcs, or blobs.

Provides reusable functions for detecting curvilinear (strip/ribbon) spatial
patterns from cell detections.  Builds KD-tree radius graphs on marker-positive
cells, extracts connected components, and classifies each component as **strip**
or **cluster** based on graph diameter normalized by component size.

Typical workflow::

    from xldvp_seg.analysis.pattern_detection import (
        select_positive_cells,
        classify_components,
        refine_strip_cells,
    )

    positive_idx = select_positive_cells(detections, snr_channel=MARKER_CH, snr_threshold=1.5)
    labels, stats, G = classify_components(positions, radius=50, ...)
    labels = refine_strip_cells(positions, labels, radius=50, method="k_core", ...)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xldvp_seg.exceptions import ConfigError
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "select_positive_cells",
    "classify_components",
    "refine_strip_cells",
]


def select_positive_cells(
    detections: list[dict[str, Any]],
    *,
    snr_channel: int | None = None,
    snr_threshold: float = 1.5,
    marker_filter: str | None = None,
) -> list[int]:
    """Return indices of positive cells based on SNR or marker filter.

    Exactly one of *snr_channel* or *marker_filter* must be provided.

    Args:
        detections: List of detection dicts (each with a ``features`` sub-dict).
        snr_channel: Channel index for SNR-based filtering (uses ``ch{N}_snr``
            key from each detection's features).
        snr_threshold: Minimum SNR value for a cell to be considered positive
            (default 1.5).  Only used when *snr_channel* is set.
        marker_filter: A ``"key==value"`` expression to match against detection
            fields or their ``features`` sub-dict (e.g. ``"MSLN_class==positive"``).

    Returns:
        List of integer indices into *detections* for positive cells.

    Raises:
        ValueError: If neither or both of *snr_channel* and *marker_filter*
            are provided, or if *marker_filter* is malformed.
    """
    if (snr_channel is None) == (marker_filter is None):
        raise ConfigError("Exactly one of snr_channel or marker_filter must be provided")

    positive_idx: list[int] = []

    if snr_channel is not None:
        snr_key = f"ch{snr_channel}_snr"
        for i, d in enumerate(detections):
            snr = d.get("features", {}).get(snr_key, 0)
            if snr >= snr_threshold:
                positive_idx.append(i)
        logger.info(
            "SNR filter: %s >= %.1f -> %d positive cells",
            snr_key,
            snr_threshold,
            len(positive_idx),
        )
    else:
        assert marker_filter is not None  # for type checker
        if "==" not in marker_filter:
            raise ConfigError(
                f"Malformed marker_filter: expected 'key==value', got {marker_filter!r}"
            )
        key, value = marker_filter.split("==", 1)
        key, value = key.strip(), value.strip()
        # Handle boolean/numeric values stored as non-string types
        match_values: set[Any] = {value}
        if value.lower() in ("true", "false"):
            match_values.add(value.lower() == "true")
        for i, d in enumerate(detections):
            v1 = d.get(key)
            v2 = d.get("features", {}).get(key)
            if v1 in match_values or v2 in match_values:
                positive_idx.append(i)
        logger.info("Marker filter: %s -> %d positive cells", marker_filter, len(positive_idx))

    return positive_idx


def classify_components(
    positions: np.ndarray,
    radius: float,
    min_component_size: int,
    linearity_threshold: float,
    min_strip_cells: int = 0,
    min_strip_length: float = 0,
    max_strip_width: float = 0,
) -> tuple[list[str], list[dict[str, Any]], Any]:
    """Classify cells by connected component shape analysis.

    For each connected component in the radius graph:
      - Compute graph diameter (longest shortest path)
      - Linearity score = diameter / sqrt(n_nodes)
      - High score -> strip, low score -> blob

    Args:
        positions: ``(N, 2)`` array of cell positions in microns.
        radius: Connection radius in microns for graph edges.
        min_component_size: Components smaller than this are classified as noise.
        linearity_threshold: Minimum ``diameter / sqrt(n)`` ratio for strip
            classification.
        min_strip_cells: Minimum cells in a strip component.  Components passing
            linearity but below this size become ``"cluster"``.  0 = no filter.
        min_strip_length: Minimum physical length in microns along the diameter
            path.  0 = no filter.
        max_strip_width: Maximum perpendicular width in microns.  Components
            wider than this become ``"cluster"``.  0 = no filter.

    Returns:
        Tuple of ``(labels, comp_stats, G)`` where *labels* is a list of
        per-node strings (``"strip"``, ``"cluster"``, or ``"noise"``),
        *comp_stats* is a list of component metric dicts sorted by linearity,
        and *G* is the networkx Graph.
    """
    import networkx as nx
    from scipy.spatial import KDTree

    from xldvp_seg.utils.graph_topology import (
        component_linearity,
        component_width,
        double_bfs_diameter,
        path_length_um,
    )

    # Build radius graph
    logger.info("Building KD-tree radius graph (r=%.0f um)...", radius)
    tree = KDTree(positions)
    pairs = tree.query_pairs(r=radius)
    logger.info("  %d edges from %d nodes", len(pairs), len(positions))

    G = nx.Graph()
    G.add_nodes_from(range(len(positions)))
    G.add_edges_from(pairs)

    # Connected components
    components = list(nx.connected_components(G))
    logger.info("  %d connected components", len(components))

    # Size distribution
    sizes = sorted([len(c) for c in components], reverse=True)
    logger.info("  Largest 10 components: %s", sizes[:10])
    logger.info(
        "  Components >= %d nodes: %d",
        min_component_size,
        sum(1 for s in sizes if s >= min_component_size),
    )

    # Analyze each component
    labels: list[str] = ["noise"] * len(positions)  # default
    comp_stats: list[dict[str, Any]] = []

    n_strips = 0
    n_blobs = 0
    strip_cells = 0
    blob_cells = 0

    for comp_idx, comp_nodes in enumerate(components):
        n = len(comp_nodes)

        if n < min_component_size:
            # Too small -- leave as noise
            continue

        # Graph diameter, physical length, width, linearity -- via shared module
        diameter, path_nodes = double_bfs_diameter(G, comp_nodes)
        length_um = path_length_um(positions, path_nodes)
        width_um = component_width(positions, list(comp_nodes), path_nodes)
        linearity = component_linearity(diameter, n)
        is_strip = (
            linearity >= linearity_threshold
            and (min_strip_cells == 0 or n >= min_strip_cells)
            and (min_strip_length == 0 or length_um >= min_strip_length)
            and (max_strip_width == 0 or width_um <= max_strip_width)
        )

        comp_stats.append(
            {
                "component": comp_idx,
                "n_cells": n,
                "diameter": diameter,
                "length_um": float(length_um),
                "width_um": float(width_um),
                "linearity": float(linearity),
                "classification": "strip" if is_strip else "cluster",
            }
        )

        label = "strip" if is_strip else "cluster"
        for node in comp_nodes:
            labels[node] = label

        if is_strip:
            n_strips += 1
            strip_cells += n
        else:
            n_blobs += 1
            blob_cells += n

    # Summary
    comp_stats.sort(key=lambda x: x["linearity"], reverse=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPONENT ANALYSIS (threshold: linearity >= %.1f)", linearity_threshold)
    logger.info("=" * 70)
    logger.info("  Strip components: %d (%d cells)", n_strips, strip_cells)
    logger.info("  Blob components:  %d (%d cells)", n_blobs, blob_cells)

    noise_count = sum(1 for lab in labels if lab == "noise")
    logger.info("  Noise (small components): %d cells", noise_count)

    # Linearity distribution
    lin_values = [c["linearity"] for c in comp_stats]
    if lin_values:
        logger.info("")
        logger.info("LINEARITY DISTRIBUTION (components >= %d cells)", min_component_size)
        for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
            n_above = sum(1 for v in lin_values if v >= thresh)
            cells_above = sum(c["n_cells"] for c in comp_stats if c["linearity"] >= thresh)
            logger.info(
                "  linearity >= %4.1f: %4d components, %6d cells",
                thresh,
                n_above,
                cells_above,
            )

    # Top components
    logger.info("")
    logger.info("TOP 20 COMPONENTS BY LINEARITY (>= %d cells)", min_component_size)
    logger.info(
        "%5s %6s %8s %9s %8s %10s %8s",
        "comp",
        "cells",
        "diameter",
        "length",
        "width",
        "linearity",
        "class",
    )
    for c in comp_stats[:20]:
        logger.info(
            "%5d %6d %8d %8.0f um %7.0f um %10.2f %8s",
            c["component"],
            c["n_cells"],
            c["diameter"],
            c["length_um"],
            c["width_um"],
            c["linearity"],
            c["classification"],
        )

    return labels, comp_stats, G


def refine_strip_cells(
    positions: np.ndarray,
    labels: list[str],
    radius: float,
    method: str,
    threshold: float,
    G_all: Any | None = None,
) -> list[str]:
    """Per-cell refinement: demote hanger-on cells from 'strip' to 'cluster'.

    Three methods:

    - **betweenness**: Drop bottom N% by betweenness centrality within each
      strip component.
    - **degree_ratio**: Drop cells where < *threshold* fraction of neighbors
      are strip cells.
    - **k_core**: Keep only the k-core of each strip component's subgraph.

    Args:
        positions: ``(N, 2)`` array of all positive cell positions.
        labels: Per-cell labels (``"strip"``, ``"cluster"``, ``"noise"``).
        radius: Connection radius in microns (used only if *G_all* is None).
        method: One of ``"betweenness"``, ``"degree_ratio"``, ``"k_core"``.
        threshold: Method-specific threshold.  Pass 0 for method defaults
            (betweenness=30, degree_ratio=0.5, k_core=2).
        G_all: Pre-built networkx Graph (avoids redundant KD-tree rebuild).

    Returns:
        Updated labels list (some ``"strip"`` demoted to ``"cluster"``).
    """
    import networkx as nx

    # Defaults
    if threshold == 0:
        if method == "betweenness":
            threshold = 30  # drop bottom 30%
        elif method == "degree_ratio":
            threshold = 0.5  # at least 50% strip neighbors
        elif method == "k_core":
            threshold = 2  # k=2

    strip_indices = [i for i, lab in enumerate(labels) if lab == "strip"]
    if not strip_indices:
        return labels

    logger.info(
        "Refining %d strip cells with method=%s, threshold=%s",
        len(strip_indices),
        method,
        threshold,
    )

    # Reuse graph from classify_components if provided
    if G_all is None:
        from scipy.spatial import KDTree

        tree = KDTree(positions)
        pairs = tree.query_pairs(r=radius)
        G_all = nx.Graph()
        G_all.add_nodes_from(range(len(positions)))
        G_all.add_edges_from(pairs)

    strip_set = set(strip_indices)
    labels = list(labels)  # copy
    demoted = 0

    if method == "betweenness":
        # Build subgraph of strip cells only
        G_strip = G_all.subgraph(strip_indices).copy()
        # Process each connected component separately
        for comp_nodes in nx.connected_components(G_strip):
            if len(comp_nodes) < 3:
                continue
            sub = G_strip.subgraph(comp_nodes)
            # Approximate for large components (exact is O(V*E))
            if len(comp_nodes) > 500:
                bc = nx.betweenness_centrality(sub, k=min(200, len(comp_nodes)))
            else:
                bc = nx.betweenness_centrality(sub)
            # Rank-based demotion: bottom N% of betweenness centrality
            sorted_nodes = sorted(bc.items(), key=lambda x: x[1])
            n_demote = int(len(sorted_nodes) * threshold / 100)
            for node, val in sorted_nodes[:n_demote]:
                labels[node] = "cluster"
                demoted += 1

    elif method == "degree_ratio":
        # For each strip cell, fraction of neighbors that are also strip
        for i in strip_indices:
            neighbors = set(G_all.neighbors(i))
            if not neighbors:
                labels[i] = "cluster"
                demoted += 1
                continue
            strip_neighbors = neighbors & strip_set
            ratio = len(strip_neighbors) / len(neighbors)
            if ratio < threshold:
                labels[i] = "cluster"
                demoted += 1

    elif method == "k_core":
        k = int(threshold)
        G_strip = G_all.subgraph(strip_indices).copy()
        # k-core: maximal subgraph where all nodes have degree >= k
        core = nx.k_core(G_strip, k=k)
        core_nodes = set(core.nodes())
        for i in strip_indices:
            if i not in core_nodes:
                labels[i] = "cluster"
                demoted += 1

    logger.info(
        "  Refinement demoted %d cells (strip: %d -> %d)",
        demoted,
        len(strip_indices),
        len(strip_indices) - demoted,
    )

    return labels
