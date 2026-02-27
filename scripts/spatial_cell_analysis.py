#!/usr/bin/env python3
"""
Spatial cell analysis: RF embedding, morphological UMAP, and network analysis.

Three independent modes activated by flags. Can run one, two, or all three.

Modes:
  --rf-embedding     RF leaf-node co-occurrence UMAP (supervised embedding)
  --morph-umap       Morphological feature UMAP with small-multiple coloring
  --spatial-network  Delaunay-based cell adjacency graph + community detection

Usage:
    python scripts/spatial_cell_analysis.py \
        --detections detections.json \
        --output-dir analysis_output/ \
        --classifier rf_classifier.pkl \
        --rf-embedding --morph-umap --spatial-network

Dependencies: umap-learn, networkx, scipy, matplotlib, scikit-learn
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.utils.logging import get_logger, setup_logging
from segmentation.utils.json_utils import NumpyEncoder
from segmentation.utils.detection_utils import (
    load_detections,
    extract_feature_matrix,
)

logger = get_logger(__name__)


def get_positions_um(detections, pixel_size):
    """Extract cell positions in microns.

    Tries global_center_um first, then global_center * pixel_size.
    Returns (positions, valid_indices) where positions is (n, 2) in [x, y].
    """
    positions = []
    valid_indices = []
    for i, det in enumerate(detections):
        feats = det.get('features', {})
        # Try um coordinates first
        center_um = det.get('global_center_um', feats.get('global_center_um'))
        if center_um is not None and len(center_um) == 2:
            positions.append(center_um)
            valid_indices.append(i)
            continue
        # Fall back to pixel coordinates * pixel_size
        center_px = det.get('global_center', feats.get('global_center'))
        if center_px is not None and len(center_px) == 2:
            positions.append([center_px[0] * pixel_size, center_px[1] * pixel_size])
            valid_indices.append(i)
    return np.array(positions, dtype=np.float64), valid_indices


def subsample_for_umap(max_samples, n_samples, rng=None):
    """Return (fit_idx, all_idx) index arrays for UMAP subsampling.

    If n_samples <= max_samples, both are np.arange(n_samples).
    Otherwise, fit_idx is a random subset of max_samples indices.
    """
    all_idx = np.arange(n_samples)
    if n_samples <= max_samples:
        return all_idx, all_idx
    if rng is None:
        rng = np.random.default_rng(42)
    fit_idx = rng.choice(n_samples, max_samples, replace=False)
    fit_idx.sort()
    return fit_idx, all_idx


def run_umap(X, fit_idx, all_idx, n_neighbors=30, min_dist=0.1, metric='euclidean'):
    """Fit UMAP on X[fit_idx], transform X[all_idx]. Returns full embedding."""
    import umap as umap_lib

    n_jobs = min(int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)), 64)
    n_nbrs = max(min(n_neighbors, len(fit_idx) - 1), 2)

    reducer = umap_lib.UMAP(
        n_neighbors=n_nbrs,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        random_state=42,
        n_jobs=n_jobs,
        low_memory=True,
    )

    if len(fit_idx) == len(all_idx):
        # No subsampling needed
        embedding = reducer.fit_transform(X)
    else:
        logger.info(f"  Fitting UMAP on {len(fit_idx):,} / {len(all_idx):,} samples...")
        reducer.fit(X[fit_idx])
        embedding = reducer.transform(X)

    return embedding


def parse_marker_filter(filter_str):
    """Parse a marker filter string into a callable predicate.

    Supported formats:
        'ch0_mean>100'           — numeric comparison on a feature
        'tdTomato_class==positive' — string equality on a feature
        'SMA_class!=negative'    — string inequality

    Returns a function that takes a detection dict and returns bool.
    """
    # Try numeric pattern first: feature_name operator number
    m = re.match(r'^(\w+)\s*([><=!]+)\s*([0-9.eE+-]+)$', filter_str.strip())
    if m:
        feat_name, op, val_str = m.group(1), m.group(2), m.group(3)
        threshold = float(val_str)

        ops = {
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
        }
        if op not in ops:
            logger.error(f"Unsupported operator '{op}' in marker filter")
            sys.exit(1)
        cmp_fn = ops[op]

        def predicate(det):
            feats = det.get('features', {})
            val = feats.get(feat_name, 0.0)
            if val is None:
                return False
            return cmp_fn(float(val), threshold)

        return predicate

    # Try string pattern: feature_name == or != string_value
    m = re.match(r'^(\w+)\s*(==|!=)\s*(\w+)$', filter_str.strip())
    if m:
        feat_name, op, str_val = m.group(1), m.group(2), m.group(3)

        def predicate(det):
            feats = det.get('features', {})
            val = feats.get(feat_name)
            if val is None:
                return False
            if op == '==':
                return str(val) == str_val
            else:
                return str(val) != str_val

        return predicate

    logger.error(f"Cannot parse marker filter: '{filter_str}' "
                 f"(expected e.g. 'ch0_mean>100' or 'tdTomato_class==positive')")
    sys.exit(1)


def save_enriched_detections(detections, output_path):
    """Save detections with enriched fields to JSON (no indent, timestamped)."""
    from segmentation.utils.timestamps import save_with_timestamp
    save_with_timestamp(output_path, detections, fmt='json', json_encoder=NumpyEncoder)
    logger.info(f"Saved enriched detections ({len(detections):,} entries)")


# ---------------------------------------------------------------------------
# Mode 1: RF Embedding
# ---------------------------------------------------------------------------

def run_rf_embedding(detections, classifier_path, output_dir, *,
                     max_umap_samples=50000, umap_neighbors=30, umap_min_dist=0.1):
    """Generate UMAP from RF leaf-node co-occurrence matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classifier
    logger.info(f"Loading classifier from {classifier_path}...")
    from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
    clf_data = load_nmj_rf_classifier(str(classifier_path))
    pipeline = clf_data['pipeline']
    feature_names = clf_data['feature_names']

    # Extract the raw RF from pipeline
    from sklearn.pipeline import Pipeline as SkPipeline
    if isinstance(pipeline, SkPipeline):
        # Find the RF step
        rf = None
        for name, step in pipeline.named_steps.items():
            if hasattr(step, 'apply') and hasattr(step, 'feature_importances_'):
                rf = step
                break
        if rf is None:
            rf = pipeline[-1]
    else:
        rf = pipeline
    logger.info(f"RF: {rf.n_estimators} trees, {rf.n_features_in_} features")

    # Extract features
    X, valid_idx = extract_feature_matrix(detections, feature_names)
    logger.info(f"Feature matrix: {X.shape[0]:,} x {X.shape[1]} (from {len(detections):,} detections)")

    if X.shape[0] < 10:
        logger.warning("Too few detections with features for RF embedding, skipping")
        return detections

    # Apply all pipeline preprocessing steps before the RF estimator
    if isinstance(pipeline, SkPipeline) and len(pipeline) > 1:
        X_transformed = X
        for name, step in list(pipeline.named_steps.items())[:-1]:
            X_transformed = step.transform(X_transformed)
        X_scaled = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X_scaled = X

    # Get leaf indices
    logger.info("Computing RF leaf indices...")
    leaf_indices = rf.apply(X_scaled)  # (n_samples, n_trees)
    logger.info(f"  Leaf index matrix: {leaf_indices.shape}")

    # UMAP on leaf indices with hamming distance
    logger.info(f"Running UMAP on leaf indices (metric=hamming, "
                f"n_neighbors={umap_neighbors}, min_dist={umap_min_dist})...")
    fit_idx, all_idx = subsample_for_umap(max_umap_samples, leaf_indices.shape[0])
    embedding = run_umap(leaf_indices, fit_idx, all_idx,
                         n_neighbors=umap_neighbors, min_dist=umap_min_dist,
                         metric='hamming')

    # Write back to detections
    for row, det_idx in enumerate(valid_idx):
        det = detections[det_idx]
        if 'features' not in det:
            det['features'] = {}
        det['features']['rf_umap_x'] = float(embedding[row, 0])
        det['features']['rf_umap_y'] = float(embedding[row, 1])

    # Get rf_prediction scores for coloring
    scores = np.array([
        detections[i].get('rf_prediction',
                          detections[i].get('features', {}).get('rf_prediction', 0.0))
        for i in valid_idx
    ], dtype=np.float32)

    # Feature importance
    importances = rf.feature_importances_
    feat_importance = sorted(zip(feature_names, importances),
                             key=lambda x: x[1], reverse=True)

    # Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. UMAP colored by rf_prediction
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=scores, cmap='RdYlGn', s=3, alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='RF prediction')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'RF Leaf Embedding ({len(valid_idx):,} cells, {rf.n_estimators} trees)')
    fig.savefig(output_dir / 'rf_embedding_umap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved rf_embedding_umap.png")

    # 2. Feature importance bar chart (top 20)
    top_n = min(20, len(feat_importance))
    top_feats = feat_importance[:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [f[0] for f in reversed(top_feats)]
    values = [f[1] for f in reversed(top_feats)]
    ax.barh(names, values, color='steelblue')
    ax.set_xlabel('Gini Importance')
    ax.set_title(f'Top {top_n} RF Feature Importances')
    fig.savefig(output_dir / 'rf_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved rf_feature_importance.png")

    return detections


# ---------------------------------------------------------------------------
# Mode 2: Morphological UMAP
# ---------------------------------------------------------------------------

# Morph features to look for (in priority order)
_MORPH_FEATURES = [
    'area', 'perimeter', 'circularity', 'solidity', 'aspect_ratio',
    'extent', 'equiv_diameter', 'eccentricity', 'mean_intensity',
    'skeleton_length', 'convex_area', 'major_axis_length',
    'minor_axis_length', 'orientation',
]


def run_morph_umap(detections, output_dir, *,
                   max_umap_samples=50000, umap_neighbors=30, umap_min_dist=0.1):
    """UMAP of interpretable morph features, colored as small multiples."""
    from sklearn.preprocessing import StandardScaler

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover which morph features are present (union of first 10 with features)
    sample_feats = {}
    n_sampled = 0
    for det in detections:
        feats = det.get('features', {})
        if feats:
            sample_feats.update(feats)
            n_sampled += 1
            if n_sampled >= 10:
                break

    present_morph = [f for f in _MORPH_FEATURES if f in sample_feats]

    # Also grab channel stats if present
    ch_pattern = re.compile(r'^ch\d+_(mean|std)$')
    ch_features = sorted(k for k in sample_feats if ch_pattern.match(k))

    all_features = present_morph + ch_features
    if len(all_features) < 3:
        logger.warning(f"Only {len(all_features)} morph features found, need >= 3. Skipping morph UMAP.")
        return detections

    logger.info(f"Morph UMAP: {len(present_morph)} morph + {len(ch_features)} channel features")

    # Extract feature matrix
    X, valid_idx = extract_feature_matrix(detections, all_features)
    logger.info(f"  Feature matrix: {X.shape[0]:,} x {X.shape[1]}")

    if X.shape[0] < 10:
        logger.warning("Too few detections for morph UMAP, skipping")
        return detections

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # UMAP
    logger.info(f"Running morph UMAP (n_neighbors={umap_neighbors}, min_dist={umap_min_dist})...")
    fit_idx, all_idx = subsample_for_umap(max_umap_samples, X_scaled.shape[0])
    embedding = run_umap(X_scaled, fit_idx, all_idx,
                         n_neighbors=umap_neighbors, min_dist=umap_min_dist)
    logger.info(f"  Embedding: {embedding.shape}")

    # Write back to detections
    for row, det_idx in enumerate(valid_idx):
        det = detections[det_idx]
        if 'features' not in det:
            det['features'] = {}
        det['features']['morph_umap_x'] = float(embedding[row, 0])
        det['features']['morph_umap_y'] = float(embedding[row, 1])

    # Build raw feature values for coloring
    feature_values = {}
    for j, fname in enumerate(all_features):
        feature_values[fname] = X[:, j]  # unscaled values

    # Small multiples grid
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Choose features to plot (morph first, then channel, cap at 16)
    plot_features = all_features[:16]
    n_plots = len(plot_features)
    ncols = 4
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, fname in enumerate(plot_features):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        vals = feature_values[fname]
        # Clip to p1-p99 for better color range
        vmin, vmax = np.percentile(vals[vals != 0], [1, 99]) if np.any(vals != 0) else (0, 1)
        if vmin == vmax:
            vmax = vmin + 1
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=vals, cmap='viridis', s=1, alpha=0.4,
                        vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(fname, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(f'Morphological UMAP ({len(valid_idx):,} cells)', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / 'morph_umap_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved morph_umap_grid.png")

    # Score-colored plot if rf_prediction is available
    scores = np.array([
        detections[i].get('rf_prediction',
                          detections[i].get('features', {}).get('rf_prediction', None))
        for i in valid_idx
    ])
    has_scores = not all(s is None for s in scores)
    if has_scores:
        scores = np.array([float(s) if s is not None else 0.0 for s in scores])
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=scores, cmap='RdYlGn', s=3, alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label='RF prediction')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Morph UMAP by RF Score ({len(valid_idx):,} cells)')
        fig.savefig(output_dir / 'morph_umap_by_score.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Saved morph_umap_by_score.png")

    return detections


# ---------------------------------------------------------------------------
# Mode 3: Spatial Network Analysis
# ---------------------------------------------------------------------------

def run_spatial_network(detections, output_dir, *,
                        pixel_size=None, marker_filter=None,
                        max_edge_distance=50.0, min_component_cells=3):
    """Build Delaunay-based cell adjacency graph, find components + communities."""
    if pixel_size is None:
        raise ValueError("pixel_size is required — must come from CZI metadata")
    import networkx as nx
    from scipy.spatial import Delaunay, ConvexHull

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter detections if marker_filter provided
    if marker_filter is not None:
        pred = parse_marker_filter(marker_filter)
        filtered_idx = [i for i, d in enumerate(detections) if pred(d)]
        logger.info(f"Marker filter '{marker_filter}': {len(detections):,} -> {len(filtered_idx):,}")
    else:
        filtered_idx = list(range(len(detections)))

    if len(filtered_idx) < 3:
        logger.warning(f"Only {len(filtered_idx)} cells after filtering, need >= 3. Skipping network.")
        return detections

    # Get positions
    filtered_dets = [detections[i] for i in filtered_idx]
    positions, pos_valid = get_positions_um(filtered_dets, pixel_size)

    if len(positions) < 3:
        logger.warning(f"Only {len(positions)} cells with valid positions, need >= 3. Skipping network.")
        return detections

    # Map pos_valid back to original detection indices
    det_indices = [filtered_idx[vi] for vi in pos_valid]
    logger.info(f"Building Delaunay triangulation for {len(positions):,} cells...")

    # Delaunay triangulation (can fail on collinear points)
    try:
        tri = Delaunay(positions)
    except Exception as e:
        logger.warning(f"Delaunay triangulation failed ({type(e).__name__}): {e}")
        logger.warning("  This can happen when cells are nearly collinear. Skipping network.")
        return detections

    # Build networkx graph
    G = nx.Graph()
    for node_id in range(len(positions)):
        G.add_node(node_id, det_idx=det_indices[node_id],
                    x_um=float(positions[node_id, 0]),
                    y_um=float(positions[node_id, 1]))

    # Extract edges from Delaunay simplices
    edge_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                if a > b:
                    a, b = b, a
                edge_set.add((a, b))

    # Add edges with distance, prune by max_edge_distance
    n_total_edges = len(edge_set)
    n_kept = 0
    for a, b in edge_set:
        dist = np.linalg.norm(positions[a] - positions[b])
        if dist <= max_edge_distance:
            G.add_edge(a, b, weight=float(dist))
            n_kept += 1

    logger.info(f"  Delaunay edges: {n_total_edges:,} total, {n_kept:,} kept "
                f"(<= {max_edge_distance} um)")

    # Connected components
    components = list(nx.connected_components(G))
    # Filter by min size
    large_components = [c for c in components if len(c) >= min_component_cells]
    logger.info(f"  Connected components: {len(components):,} total, "
                f"{len(large_components):,} with >= {min_component_cells} cells")

    # Per-component metrics
    comp_rows = []
    for comp_id, comp_nodes in enumerate(large_components):
        comp_nodes = sorted(comp_nodes)
        subgraph = G.subgraph(comp_nodes)
        comp_positions = positions[comp_nodes]

        n_cells = len(comp_nodes)
        mean_degree = 2.0 * subgraph.number_of_edges() / max(n_cells, 1)

        # Convex hull area
        hull_area = 0.0
        if n_cells >= 3:
            try:
                hull = ConvexHull(comp_positions)
                hull_area = float(hull.volume)  # 2D: volume = area
            except Exception as e:
                logger.debug(f"  ConvexHull failed for component {comp_id} ({n_cells} cells): {e}")

        density = n_cells / hull_area if hull_area > 0 else float('nan')

        # Graph diameter (longest shortest path)
        try:
            diameter = nx.diameter(subgraph)
        except nx.NetworkXError:
            diameter = 0

        # Mean channel features for the component
        ch_means = {}
        ch_pattern = re.compile(r'^ch\d+_mean$')
        for node in comp_nodes:
            det = detections[G.nodes[node]['det_idx']]
            feats = det.get('features', {})
            for k, v in feats.items():
                if ch_pattern.match(k) and isinstance(v, (int, float)):
                    ch_means.setdefault(k, []).append(float(v))
        ch_avg = {k: np.mean(v) for k, v in ch_means.items()}

        row = {
            'component_id': comp_id,
            'n_cells': n_cells,
            'hull_area_um2': hull_area,
            'density': density,
            'diameter': diameter,
            'mean_degree': mean_degree,
        }
        row.update(ch_avg)
        comp_rows.append(row)

        # Write component_id back to detections
        for node in comp_nodes:
            det_idx = G.nodes[node]['det_idx']
            if 'features' not in detections[det_idx]:
                detections[det_idx]['features'] = {}
            detections[det_idx]['features']['component_id'] = comp_id

    # Save component summary CSV
    if comp_rows:
        csv_path = output_dir / 'component_summary.csv'
        # Union all keys across rows (some components may have different channel features)
        fieldnames_set = set()
        for row in comp_rows:
            fieldnames_set.update(row.keys())
        # Stable order: fixed columns first, then sorted channel columns
        fixed_cols = ['component_id', 'n_cells', 'hull_area_um2', 'density', 'diameter', 'mean_degree']
        extra_cols = sorted(fieldnames_set - set(fixed_cols))
        fieldnames = fixed_cols + extra_cols
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comp_rows)
        logger.info(f"  Saved component_summary.csv ({len(comp_rows)} components)")

    # Community detection (Louvain) on full pruned graph
    # Use weight=None so all edges are equal — the stored weight is distance_um,
    # and Louvain groups by high weight, which would invert the spatial semantics.
    logger.info("Running Louvain community detection...")
    try:
        communities = nx.community.louvain_communities(G, weight=None, seed=42)
        non_trivial = sum(1 for c in communities if len(c) > 1)
        logger.info(f"  Found {len(communities)} communities ({non_trivial} with 2+ cells)")
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                det_idx = G.nodes[node]['det_idx']
                if 'features' not in detections[det_idx]:
                    detections[det_idx]['features'] = {}
                detections[det_idx]['features']['community_id'] = comm_id
    except Exception as e:
        logger.warning(f"Louvain community detection failed: {e}")
        communities = None

    # Save GraphML
    graphml_path = output_dir / 'graph.graphml'
    nx.write_graphml(G, str(graphml_path))
    logger.info(f"  Saved graph.graphml ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    # Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # 1. Spatial network colored by component
    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw edges (light gray) — LineCollection for performance
    segments = [[positions[a], positions[b]] for a, b in G.edges()]
    if segments:
        lc = LineCollection(segments, colors='#cccccc', linewidths=0.3, alpha=0.4)
        ax.add_collection(lc)

    # Color nodes by component_id
    node_colors = np.full(len(positions), -1, dtype=int)
    for comp_id, comp_nodes in enumerate(large_components):
        for node in comp_nodes:
            node_colors[node] = comp_id

    # Unclustered cells
    unclustered = node_colors == -1
    if np.any(unclustered):
        ax.scatter(positions[unclustered, 0], positions[unclustered, 1],
                   c='#dddddd', s=3, alpha=0.3, label='unclustered', rasterized=True)

    # Clustered cells
    clustered = ~unclustered
    if np.any(clustered):
        sc = ax.scatter(positions[clustered, 0], positions[clustered, 1],
                        c=node_colors[clustered], cmap='tab20', s=5, alpha=0.6,
                        rasterized=True)

    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(f'Spatial Network ({len(positions):,} cells, {len(large_components)} components, '
                 f'max edge {max_edge_distance} um)')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    fig.savefig(output_dir / 'spatial_network.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved spatial_network.png")

    # 2. Communities plot
    if communities is not None and len(communities) > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        comm_colors = np.full(len(positions), -1, dtype=int)
        for comm_id, comm_nodes in enumerate(communities):
            for node in comm_nodes:
                comm_colors[node] = comm_id

        sc = ax.scatter(positions[:, 0], positions[:, 1],
                        c=comm_colors, cmap='tab20', s=5, alpha=0.6, rasterized=True)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Y (um)')
        ax.set_title(f'Louvain Communities ({len(communities)} communities)')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        fig.savefig(output_dir / 'spatial_communities.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"  Saved spatial_communities.png")

    return detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Spatial cell analysis: RF embedding, morph UMAP, network analysis')
    parser.add_argument('--detections', required=True,
                        help='Path to detections JSON')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for plots, CSV, and enriched JSON')
    parser.add_argument('--classifier', default=None,
                        help='Path to RF classifier (.pkl/.joblib) — required for --rf-embedding')
    parser.add_argument('--rf-embedding', action='store_true',
                        help='Mode 1: RF leaf-node co-occurrence UMAP')
    parser.add_argument('--morph-umap', action='store_true',
                        help='Mode 2: Morphological feature UMAP (small multiples)')
    parser.add_argument('--spatial-network', action='store_true',
                        help='Mode 3: Delaunay-based spatial network analysis')
    parser.add_argument('--score-threshold', type=float, default=None,
                        help='Filter detections by rf_prediction >= threshold')
    parser.add_argument('--marker-filter', default=None,
                        help='Filter for spatial network, e.g. "ch0_mean>100"')
    parser.add_argument('--max-edge-distance', type=float, default=50.0,
                        help='Max edge distance in um for network (default: 50)')
    parser.add_argument('--min-component-cells', type=int, default=3,
                        help='Min cells per connected component (default: 3)')
    parser.add_argument('--pixel-size', type=float, required=True,
                        help='Pixel size in um/px. Must match the CZI metadata of the source image.')
    parser.add_argument('--max-umap-samples', type=int, default=50000,
                        help='Subsample limit for UMAP (default: 50000)')
    parser.add_argument('--umap-neighbors', type=int, default=30,
                        help='UMAP n_neighbors (default: 30)')
    parser.add_argument('--umap-min-dist', type=float, default=0.1,
                        help='UMAP min_dist (default: 0.1)')
    args = parser.parse_args()
    setup_logging(level="INFO")

    if not (args.rf_embedding or args.morph_umap or args.spatial_network):
        parser.error("At least one mode required: --rf-embedding, --morph-umap, --spatial-network")

    if args.rf_embedding and not args.classifier:
        parser.error("--classifier is required when using --rf-embedding")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detections (shared across all modes)
    detections = load_detections(args.detections, args.score_threshold)

    if not detections:
        logger.error("No detections after filtering — nothing to analyze")
        sys.exit(1)

    # Run selected modes
    if args.rf_embedding:
        logger.info("=" * 60)
        logger.info("Mode 1: RF Embedding")
        logger.info("=" * 60)
        detections = run_rf_embedding(
            detections, args.classifier, output_dir,
            max_umap_samples=args.max_umap_samples,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

    if args.morph_umap:
        logger.info("=" * 60)
        logger.info("Mode 2: Morphological UMAP")
        logger.info("=" * 60)
        detections = run_morph_umap(
            detections, output_dir,
            max_umap_samples=args.max_umap_samples,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

    if args.spatial_network:
        logger.info("=" * 60)
        logger.info("Mode 3: Spatial Network Analysis")
        logger.info("=" * 60)
        detections = run_spatial_network(
            detections, output_dir,
            pixel_size=args.pixel_size,
            marker_filter=args.marker_filter,
            max_edge_distance=args.max_edge_distance,
            min_component_cells=args.min_component_cells,
        )

    # Save enriched detections
    enriched_path = output_dir / 'detections_enriched.json'
    save_enriched_detections(detections, enriched_path)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
