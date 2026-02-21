#!/usr/bin/env python3
"""
Feature-based clustering of RF-positive islet cells.

Uses UMAP dimensionality reduction + HDBSCAN clustering to discover cell types
among RF-positive detections based on per-channel marker expression.

Auto-labels clusters by dominant marker:
  - Highest ch2_mean (Gcg) -> "alpha"
  - Highest ch3_mean (Ins) -> "beta"
  - Highest ch5_mean (Sst) -> "delta"
  - Mixed/low -> "other"

Outputs:
  - islet_detections_clustered.json  — detections + cluster_id, cluster_label, umap_x, umap_y
  - cluster_summary.csv              — per-cluster stats
  - umap_plot.png                    — UMAP colored by cluster
  - marker_violin.png                — marker intensity distributions per cluster
  - islet_spatial.h5ad               — AnnData format for scanpy
  - islet_spatial.csv                — flat CSV with coords + clusters

Usage:
  python scripts/cluster_by_features.py \\
      --detections /path/to/islet_detections.json \\
      --output-dir /path/to/clustering_output \\
      --threshold 0.5

Dependencies: umap-learn, hdbscan, anndata, matplotlib, pandas
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_detections(detections_path, threshold=0.5):
    """Load RF-positive detections above threshold."""
    with open(detections_path) as f:
        all_dets = json.load(f)

    positive = []
    for det in all_dets:
        score = det.get('rf_prediction', det.get('score'))
        if score is not None and score >= threshold:
            positive.append(det)
        elif score is None:
            # No classifier — include all
            positive.append(det)

    return positive


def extract_feature_matrix(detections, feature_prefix='ch', feature_names_override=None):
    """Extract feature matrix from detections.

    Selects features matching the prefix (default: per-channel stats),
    or uses an explicit list if feature_names_override is provided.

    Returns:
        X: numpy array (n_cells, n_features)
        feature_names: list of feature name strings
        valid_indices: indices of detections with complete features
    """
    if feature_names_override:
        feature_names = list(feature_names_override)
    else:
        # Collect all feature names matching prefix
        all_names = set()
        for det in detections:
            feats = det.get('features', {})
            for k in feats:
                if k.startswith(feature_prefix) and isinstance(feats[k], (int, float)):
                    all_names.add(k)
        feature_names = sorted(all_names)

    if not feature_names:
        return None, [], []

    # Build matrix
    rows = []
    valid_indices = []
    for i, det in enumerate(detections):
        feats = det.get('features', {})
        row = []
        valid = True
        for name in feature_names:
            val = feats.get(name)
            if val is None or not isinstance(val, (int, float)):
                valid = False
                break
            row.append(float(val))
        if valid:
            rows.append(row)
            valid_indices.append(i)

    X = np.array(rows, dtype=np.float64)
    return X, feature_names, valid_indices


def normalize_marker_features(X, feature_names):
    """Normalize marker features using population p1-p99.5 percentile stretch.

    Same normalization as classify_islet_marker() / HTML display so clustering
    matches visual appearance.

    Args:
        X: (n_cells, n_features) raw feature matrix
        feature_names: list of feature names

    Returns:
        X_norm: (n_cells, n_features) normalized to [0, 1]
        norm_ranges: dict mapping feature_name -> (lo, hi) percentile values
    """
    X_norm = np.zeros_like(X)
    norm_ranges = {}
    for j in range(X.shape[1]):
        col = X[:, j]
        lo = np.percentile(col, 1)
        hi = np.percentile(col, 99.5)
        norm_ranges[feature_names[j]] = (lo, hi)
        if hi > lo:
            X_norm[:, j] = np.clip((col - lo) / (hi - lo), 0, 1)
        else:
            X_norm[:, j] = 0.0
        print(f"  {feature_names[j]}: range [{lo:.1f}, {hi:.1f}] -> [0, 1]")
    return X_norm, norm_ranges


def auto_label_clusters(detections, labels, valid_indices, norm_ranges=None):
    """Auto-label clusters by dominant marker expression.

    For each cluster, compute mean of ch2_mean (Gcg), ch3_mean (Ins), ch5_mean (Sst).
    When norm_ranges is provided, normalize values before comparing (so each channel
    is on the same scale and the highest-signal channel doesn't dominate).

    Args:
        norm_ranges: dict mapping 'ch2_mean' etc to (lo, hi) percentile ranges.
            If provided, values are normalized to [0,1] before comparison.
    """
    cluster_labels = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    marker_keys = {
        'alpha': 'ch2_mean',   # Gcg
        'beta': 'ch3_mean',    # Ins
        'delta': 'ch5_mean',   # Sst
    }

    for cl in sorted(unique_labels):
        mask = labels == cl
        cluster_dets = [detections[valid_indices[i]] for i, m in enumerate(mask) if m]

        marker_means = {}
        for label, key in marker_keys.items():
            vals = [d.get('features', {}).get(key, 0) for d in cluster_dets]
            mean_val = np.mean(vals) if vals else 0
            # Normalize if ranges provided
            if norm_ranges and key in norm_ranges:
                lo, hi = norm_ranges[key]
                if hi > lo:
                    mean_val = max(0, (mean_val - lo) / (hi - lo))
                else:
                    mean_val = 0.0
            marker_means[label] = mean_val

        if max(marker_means.values()) == 0:
            cluster_labels[cl] = 'other'
        else:
            best = max(marker_means, key=marker_means.get)
            # Only label if dominant marker is clearly above baseline (>0.1 normalized)
            if norm_ranges and marker_means[best] < 0.1:
                cluster_labels[cl] = 'other'
            else:
                cluster_labels[cl] = best

    cluster_labels[-1] = 'noise'
    return cluster_labels


def run_clustering(args):
    """Main clustering pipeline."""
    from sklearn.preprocessing import StandardScaler

    try:
        import umap
    except ImportError:
        print("ERROR: umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    try:
        import hdbscan
    except ImportError:
        print("ERROR: hdbscan not installed. Run: pip install hdbscan")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detections
    print(f"Loading detections from {args.detections}...")
    detections = load_detections(args.detections, threshold=args.threshold)
    print(f"  {len(detections)} RF-positive detections (threshold >= {args.threshold})")

    if len(detections) < args.min_cluster_size:
        print(f"ERROR: Not enough detections ({len(detections)}) for clustering "
              f"(need at least {args.min_cluster_size})")
        sys.exit(1)

    # Extract features
    print("Extracting feature matrix...")
    marker_features = ['ch2_mean', 'ch3_mean', 'ch5_mean']  # Gcg, Ins, Sst

    if args.marker_only:
        print("  Using normalized marker channels only (ch2_mean, ch3_mean, ch5_mean)")
        X, feature_names, valid_indices = extract_feature_matrix(
            detections, feature_names_override=marker_features
        )
    else:
        X, feature_names, valid_indices = extract_feature_matrix(
            detections, feature_prefix=args.feature_prefix
        )

    if X is None or len(X) == 0:
        print("ERROR: No valid features found")
        sys.exit(1)

    print(f"  Feature matrix: {X.shape[0]} cells x {X.shape[1]} features")

    # Normalize and scale
    norm_ranges = None
    if args.marker_only:
        # Population percentile normalization (matches HTML display)
        print("  Applying p1-p99.5 percentile normalization...")
        X_scaled, norm_ranges = normalize_marker_features(X, feature_names)
        # No StandardScaler needed — already [0, 1] and all same units
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # Replace NaN/inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # UMAP embedding
    print(f"Running UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled)
    print(f"  UMAP embedding: {embedding.shape}")

    # HDBSCAN clustering
    print(f"Running HDBSCAN (min_cluster_size={args.min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # Auto-label clusters
    cluster_label_map = auto_label_clusters(detections, labels, valid_indices, norm_ranges=norm_ranges)
    print(f"  Cluster labels: {cluster_label_map}")

    # Enrich detections with cluster info
    valid_set = set(valid_indices)
    for i, idx in enumerate(valid_indices):
        det = detections[idx]
        cl = int(labels[i])
        det['cluster_id'] = cl
        det['cluster_label'] = cluster_label_map.get(cl, 'other')
        det['umap_x'] = float(embedding[i, 0])
        det['umap_y'] = float(embedding[i, 1])

    # Add sentinel values for non-clustered detections (missing features)
    for i, det in enumerate(detections):
        if i not in valid_set and 'cluster_id' not in det:
            det['cluster_id'] = None
            det['cluster_label'] = 'unclassified'
            det['umap_x'] = None
            det['umap_y'] = None

    # Save enriched detections
    clustered_path = output_dir / 'islet_detections_clustered.json'
    with open(clustered_path, 'w') as f:
        json.dump(detections, f, indent=2, default=str)
    print(f"  Saved: {clustered_path}")

    # Build summary DataFrame
    rows = []
    for i, idx in enumerate(valid_indices):
        det = detections[idx]
        gc = det.get('global_center', det.get('center', [0, 0]))
        row = {
            'uid': det.get('uid', det.get('id', f'cell_{i}')),
            'x': gc[0] if isinstance(gc, (list, tuple)) else 0,
            'y': gc[1] if isinstance(gc, (list, tuple)) else 0,
            'cluster_id': det.get('cluster_id', -1),
            'cluster_label': det.get('cluster_label', 'other'),
            'umap_x': det.get('umap_x', 0),
            'umap_y': det.get('umap_y', 0),
        }
        # Add marker intensities
        feats = det.get('features', {})
        for key in ['ch2_mean', 'ch3_mean', 'ch5_mean']:
            row[key] = feats.get(key, 0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Cluster summary
    summary = df.groupby('cluster_label').agg(
        n_cells=('uid', 'count'),
        ch2_mean_gcg=('ch2_mean', 'mean'),
        ch3_mean_ins=('ch3_mean', 'mean'),
        ch5_mean_sst=('ch5_mean', 'mean'),
        x_mean=('x', 'mean'),
        y_mean=('y', 'mean'),
        x_std=('x', 'std'),
        y_std=('y', 'std'),
    ).round(2)

    summary_path = output_dir / 'cluster_summary.csv'
    summary.to_csv(summary_path)
    print(f"  Saved: {summary_path}")
    print(summary.to_string())

    # Spatial CSV
    csv_path = output_dir / 'islet_spatial.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # AnnData export
    try:
        import anndata
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        adata = anndata.AnnData(
            X=X_clean,
            obs=df.set_index('uid'),
        )
        adata.var_names = feature_names
        adata.obsm['X_umap'] = embedding
        h5ad_path = output_dir / 'islet_spatial.h5ad'
        adata.write(h5ad_path)
        print(f"  Saved: {h5ad_path}")
    except ImportError:
        print("  Skipping .h5ad export (anndata not installed)")

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # UMAP plot
        fig, ax = plt.subplots(figsize=(10, 8))
        color_map = {
            'alpha': 'red',
            'beta': 'green',
            'delta': 'blue',
            'other': 'gray',
            'noise': 'lightgray',
        }
        for label_name in df['cluster_label'].unique():
            mask = df['cluster_label'] == label_name
            color = color_map.get(label_name, 'gray')
            ax.scatter(
                df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                c=color, label=label_name, s=5, alpha=0.6,
            )
        ax.legend(markerscale=4)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Islet Cell Clustering ({len(df)} cells, {n_clusters} clusters)')
        umap_path = output_dir / 'umap_plot.png'
        fig.savefig(umap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {umap_path}")

        # Violin plot of marker intensities per cluster
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        marker_cols = [('ch2_mean', 'Gcg (alpha)'), ('ch3_mean', 'Ins (beta)'), ('ch5_mean', 'Sst (delta)')]

        for ax, (col, title) in zip(axes, marker_cols):
            cluster_labels_sorted = sorted(df['cluster_label'].unique())
            data = [df.loc[df['cluster_label'] == cl, col].values for cl in cluster_labels_sorted]
            # Filter out clusters with <2 points (violinplot KDE requires >=2)
            valid_mask = [len(d) >= 2 for d in data]
            data_filtered = [d for d, v in zip(data, valid_mask) if v]
            tick_labels = [cl for cl, v in zip(cluster_labels_sorted, valid_mask) if v]
            if data_filtered:
                parts = ax.violinplot(data_filtered, showmeans=True, showmedians=True)
                ax.set_xticks(range(1, len(tick_labels) + 1))
                ax.set_xticklabels(tick_labels, rotation=45)
            ax.set_title(title)
            ax.set_ylabel('Intensity')

        fig.suptitle('Marker Expression by Cluster')
        fig.tight_layout()
        violin_path = output_dir / 'marker_violin.png'
        fig.savefig(violin_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {violin_path}")

    except ImportError:
        print("  Skipping plots (matplotlib not installed)")

    print(f"\nDone! {n_clusters} clusters found in {len(valid_indices)} cells.")


def main():
    parser = argparse.ArgumentParser(
        description='Feature-based clustering of RF-positive islet cells'
    )
    parser.add_argument('--detections', required=True,
                        help='Path to islet_detections.json')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for clustering results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum rf_prediction score (default: 0.5)')
    parser.add_argument('--marker-only', action='store_true',
                        help='Use only 3 normalized marker channels (ch2/Gcg, ch3/Ins, ch5/Sst)')
    parser.add_argument('--feature-prefix', type=str, default='ch',
                        help='Feature name prefix to select (default: "ch" for channel stats)')
    parser.add_argument('--n-neighbors', type=int, default=30,
                        help='UMAP n_neighbors (default: 30)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist (default: 0.1)')
    parser.add_argument('--min-cluster-size', type=int, default=50,
                        help='HDBSCAN min_cluster_size (default: 50)')
    parser.add_argument('--min-samples', type=int, default=None,
                        help='HDBSCAN min_samples (default: None, uses min_cluster_size)')
    args = parser.parse_args()

    run_clustering(args)


if __name__ == '__main__':
    main()
