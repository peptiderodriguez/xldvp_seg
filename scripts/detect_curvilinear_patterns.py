#!/usr/bin/env python3
"""Detect curvilinear (strip/ribbon) spatial patterns in classified detections.

Builds a KD-tree radius graph on marker-positive cells, extracts connected
components, then classifies each component as **strip** or **blob** based on
graph diameter normalized by component size.

Strip components have high ``diameter / sqrt(n_nodes)`` — the graph path
through them is long relative to their size. This works for curved strips
because graph diameter follows the actual path through the component, unlike
PCA which assumes a linear axis.

Each detection is tagged with a ``{prefix}_pattern`` field in features:
  - ``strip``     — member of a component classified as curvilinear
  - ``cluster``   — member of a compact/blob component
  - ``noise``     — positive but in a tiny component (< min-component-size)
  - ``other``     — not positive for the marker

Example:
    python detect_curvilinear_patterns.py \\
        --detections classified.json \\
        --snr-channel 2 --snr-threshold 1.5 \\
        --radius 50 --min-component-size 10 \\
        --linearity-threshold 3.0 \\
        --output-prefix msln
"""

import argparse
from pathlib import Path

import numpy as np

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect curvilinear spatial patterns via graph component analysis"
    )
    parser.add_argument("--detections", required=True, help="Path to classified detection JSON")

    # Marker selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--snr-channel",
        type=int,
        help="Channel index for SNR-based filtering (uses ch{N}_snr from features)",
    )
    group.add_argument(
        "--marker-filter",
        help='Filter expression for positive cells (e.g., "MSLN_class==positive")',
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=1.5,
        help="SNR threshold for positive classification (default: 1.5)",
    )

    # Graph params
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Connection radius in µm for graph edges. Should be ~1.5-2x the "
        "typical cell-to-cell spacing. Larger values bridge gaps but may merge "
        "separate structures. 50-100 µm typical for mesothelial strips (default: 50)",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=10,
        help="Components smaller than this are classified as noise (default: 10)",
    )
    parser.add_argument(
        "--linearity-threshold",
        type=float,
        default=3.0,
        help="Min diameter/sqrt(n) ratio to classify as strip (default: 3.0)",
    )
    parser.add_argument(
        "--min-strip-cells",
        type=int,
        default=0,
        help="Min cells in a strip component (components passing linearity but below "
        "this size are classified as cluster instead). 0 = no filter (default: 0)",
    )
    parser.add_argument(
        "--min-strip-length",
        type=float,
        default=0,
        help="Min physical length in µm along the diameter path for a strip. "
        "0 = no filter (default: 0)",
    )
    parser.add_argument(
        "--max-strip-width",
        type=float,
        default=0,
        help="Max perpendicular width in µm. Components wider than this are "
        "classified as cluster even if they pass linearity. Mesothelial strips "
        "are typically 50-200µm wide. 0 = no filter (default: 0)",
    )

    # Cell-level refinement (trim hangers-on from strip components)
    parser.add_argument(
        "--refine-method",
        choices=["none", "betweenness", "degree_ratio", "k_core"],
        default="none",
        help="Per-cell refinement to trim absorbed non-strip cells. "
        "betweenness: keep top N%% by betweenness centrality. "
        "degree_ratio: keep cells where most neighbors are also strip cells. "
        "k_core: keep k-core backbone (removes leaf-like appendages). "
        "(default: none)",
    )
    parser.add_argument(
        "--refine-threshold",
        type=float,
        default=0,
        help="Threshold for refinement. "
        "betweenness: percentile cutoff (e.g., 30 = drop bottom 30%%). "
        "degree_ratio: min fraction of neighbors in same strip (e.g., 0.5). "
        "k_core: minimum degree k (e.g., 2). "
        "(default: 0 = method-specific default)",
    )

    # Output
    parser.add_argument(
        "--output-prefix",
        default="marker",
        help='Prefix for tagged field names (default: "marker" → "marker_pattern")',
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as input detections)",
    )

    return parser.parse_args()


def select_positive_cells(detections, args):
    """Return indices of positive cells based on SNR or marker filter."""
    positive_idx = []

    if args.snr_channel is not None:
        snr_key = f"ch{args.snr_channel}_snr"
        for i, d in enumerate(detections):
            snr = d.get("features", {}).get(snr_key, 0)
            if snr >= args.snr_threshold:
                positive_idx.append(i)
        logger.info(
            "SNR filter: %s >= %.1f → %d positive cells",
            snr_key,
            args.snr_threshold,
            len(positive_idx),
        )
    else:
        if "==" not in args.marker_filter:
            logger.error("Malformed --marker-filter: expected 'key==value'")
            raise SystemExit(1)
        key, value = args.marker_filter.split("==", 1)
        key, value = key.strip(), value.strip()
        # Handle boolean/numeric values stored as non-string types
        match_values = {value}
        if value.lower() in ("true", "false"):
            match_values.add(value.lower() == "true")
        for i, d in enumerate(detections):
            v1 = d.get(key)
            v2 = d.get("features", {}).get(key)
            if v1 in match_values or v2 in match_values:
                positive_idx.append(i)
        logger.info("Marker filter: %s → %d positive cells", args.marker_filter, len(positive_idx))

    return positive_idx


def extract_aligned_positions(detections, positive_idx, extract_positions_um):
    """Extract positions ensuring exact index alignment with positive_idx."""
    positive_dets = [detections[i] for i in positive_idx]

    # Infer pixel_size from the batch
    _, pixel_size = extract_positions_um(positive_dets)
    px_str = f"{pixel_size:.4f}" if pixel_size else "N/A"

    # Resolve per-detection to maintain exact index alignment.
    # extract_positions_um silently skips unresolvable detections without
    # reporting which indices survived. Per-detection calls are the only way
    # to track the mapping between input indices and output positions.
    valid_positive_idx = []
    valid_positions = []
    for i, det in zip(positive_idx, positive_dets):
        pos_arr, _ = extract_positions_um([det], pixel_size_um=pixel_size)
        if len(pos_arr) == 1:
            valid_positive_idx.append(i)
            valid_positions.append(pos_arr[0])

    if len(valid_positive_idx) < len(positive_idx):
        logger.warning(
            "Dropped %d cells with unresolvable coordinates (%d → %d)",
            len(positive_idx) - len(valid_positive_idx),
            len(positive_idx),
            len(valid_positive_idx),
        )

    positions = (
        np.array(valid_positions, dtype=np.float64)
        if valid_positions
        else np.empty((0, 2), dtype=np.float64)
    )
    logger.info("  Positions extracted: %d cells, pixel_size=%s µm", len(positions), px_str)
    return valid_positive_idx, positions


def _component_width(positions, comp_nodes, path, percentile=95):
    """Perpendicular width — delegates to shared graph_topology module."""
    from segmentation.utils.graph_topology import component_width

    return component_width(positions, comp_nodes, path, percentile)


def classify_components(
    positions,
    radius,
    min_component_size,
    linearity_threshold,
    min_strip_cells=0,
    min_strip_length=0,
    max_strip_width=0,
):
    """Classify cells by connected component shape analysis.

    For each connected component in the radius graph:
      - Compute graph diameter (longest shortest path)
      - Linearity score = diameter / sqrt(n_nodes)
      - High score → strip, low score → blob

    Returns per-node labels and component stats.
    """
    import networkx as nx

    from segmentation.utils.graph_topology import (
        component_linearity,
        component_width,
        double_bfs_diameter,
        path_length_um,
    )

    # Build radius graph
    logger.info("Building KD-tree radius graph (r=%.0f µm)...", radius)
    from scipy.spatial import KDTree

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
    labels = ["noise"] * len(positions)  # default
    comp_stats = []

    n_strips = 0
    n_blobs = 0
    strip_cells = 0
    blob_cells = 0

    for comp_idx, comp_nodes in enumerate(components):
        n = len(comp_nodes)

        if n < min_component_size:
            # Too small — leave as noise
            continue

        # Graph diameter, physical length, width, linearity — via shared module
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
            "%5d %6d %8d %8.0f µm %7.0f µm %10.2f %8s",
            c["component"],
            c["n_cells"],
            c["diameter"],
            c["length_um"],
            c["width_um"],
            c["linearity"],
            c["classification"],
        )

    return labels, comp_stats, G


def refine_strip_cells(positions, labels, radius, method, threshold, G_all=None):
    """Per-cell refinement: demote hanger-on cells from 'strip' to 'cluster'.

    Three methods:
      betweenness: Drop bottom N% by betweenness centrality within each strip component.
      degree_ratio: Drop cells where < threshold fraction of neighbors are strip cells.
      k_core: Keep only the k-core of each strip component's subgraph.

    Args:
        positions: (N, 2) array of all positive cell positions.
        labels: list of per-cell labels ("strip", "cluster", "noise").
        radius: connection radius in µm (used only if G_all is None).
        method: "betweenness", "degree_ratio", or "k_core".
        threshold: method-specific threshold.
        G_all: pre-built networkx Graph (avoids redundant KD-tree rebuild).

    Returns:
        Updated labels list (some "strip" demoted to "cluster").
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
            # Find percentile cutoff
            values = sorted(bc.values())
            cutoff_idx = int(len(values) * threshold / 100)
            cutoff_val = values[min(cutoff_idx, len(values) - 1)]
            for node, val in bc.items():
                if val < cutoff_val:
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
        "  Refinement demoted %d cells (strip: %d → %d)",
        demoted,
        len(strip_indices),
        len(strip_indices) - demoted,
    )

    return labels


def tag_detections(detections, positive_idx, labels, prefix):
    """Tag each detection with pattern classification."""
    field = f"{prefix}_pattern"

    for d in detections:
        d.setdefault("features", {})[field] = "other"

    for j, orig_idx in enumerate(positive_idx):
        detections[orig_idx]["features"][field] = labels[j]

    counts = {}
    for d in detections:
        c = d.get("features", {}).get(field, "other")
        counts[c] = counts.get(c, 0) + 1
    logger.info("Final tagged classes: %s", counts)

    return field


def main():
    args = parse_args()

    from segmentation.utils.detection_utils import extract_positions_um
    from segmentation.utils.json_utils import atomic_json_dump, fast_json_load

    # Load
    logger.info("Loading %s...", args.detections)
    detections = fast_json_load(args.detections)
    logger.info("  %d detections loaded", len(detections))

    # Select positive cells
    positive_idx = select_positive_cells(detections, args)
    if len(positive_idx) < 2:
        logger.error("Only %d positive cells — not enough for analysis", len(positive_idx))
        raise SystemExit(1)

    # Extract positions
    positive_idx, positions = extract_aligned_positions(
        detections, positive_idx, extract_positions_um
    )
    if len(positions) < 2:
        logger.error("Only %d resolved positions — not enough", len(positions))
        raise SystemExit(1)

    # Classify
    labels, comp_stats, G = classify_components(
        positions,
        args.radius,
        args.min_component_size,
        args.linearity_threshold,
        args.min_strip_cells,
        args.min_strip_length,
        args.max_strip_width,
    )

    # Per-cell refinement (trim hangers-on)
    if args.refine_method != "none":
        labels = refine_strip_cells(
            positions,
            labels,
            args.radius,
            args.refine_method,
            args.refine_threshold,
            G_all=G,
        )

    # Tag
    field = tag_detections(detections, positive_idx, labels, args.output_prefix)

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.detections).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    det_out = out_dir / f"cell_detections_{args.output_prefix}_strip_tagged.json"
    logger.info("Saving tagged detections to %s...", det_out)
    atomic_json_dump(detections, str(det_out))

    # Write strip-only JSON for fast viewer generation
    strip_dets = [d for d in detections if d.get("features", {}).get(field) == "strip"]
    strip_out = out_dir / f"cell_detections_{args.output_prefix}_strip_only.json"
    logger.info("Saving %d strip-only detections to %s...", len(strip_dets), strip_out)
    atomic_json_dump(strip_dets, str(strip_out))

    stats_out = out_dir / f"{args.output_prefix}_component_stats.json"
    atomic_json_dump(comp_stats, str(stats_out))
    logger.info("Saved component stats to %s", stats_out)
    logger.info("Done. Group field for viewer: %s", field)


if __name__ == "__main__":
    main()
