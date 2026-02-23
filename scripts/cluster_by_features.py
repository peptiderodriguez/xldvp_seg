#!/usr/bin/env python3
"""
Feature-based clustering of cell detections using UMAP + HDBSCAN.

Works with ANY cell type's channel features. Auto-discovers channel features
from detections (keys matching ch\\d+_*) and labels clusters by dominant marker.

For islet detections (backward compat): auto-detects ch2/ch3/ch5 and labels
alpha/beta/delta when --marker-channels is not specified.

Feature groups:
  - "morph":   all morphological features (= shape + color, backward compat)
  - "shape":   pure geometry (area, circularity, solidity, aspect_ratio, etc.)
  - "color":   intensity/color (gray_mean, hue_mean, relative_brightness, etc.)
  - "sam2":    SAM2 embedding features (sam2_0..sam2_255)
  - "channel": per-channel stats (ch0_mean, ch1_std, ch0_ch2_ratio, etc.)
  - "deep":    deep features (resnet_*, dinov2_*)

Auto-labels clusters by dominant marker:
  - For each cluster, z-score of mean expression per marker channel
  - Highest z-score marker name becomes the cluster label
  - Mixed/low -> "other"

Outputs:
  - detections_clustered.json   -- detections + cluster_id, cluster_label, umap_x, umap_y
  - cluster_summary.csv         -- per-cluster stats
  - umap_plot.png               -- UMAP colored by cluster
  - marker_violin.png           -- marker intensity distributions per cluster
  - spatial.h5ad                -- AnnData format for scanpy
  - spatial.csv                 -- flat CSV with coords + clusters

Usage:
  # Islet (backward compat -- auto-detects ch2/ch3/ch5 as alpha/beta/delta)
  python scripts/cluster_by_features.py \\
      --detections /path/to/islet_detections.json \\
      --output-dir /path/to/clustering_output

  # Tissue pattern with 4 channels, exclude ch3 (bad stain)
  python scripts/cluster_by_features.py \\
      --detections /path/to/tissue_pattern_detections.json \\
      --output-dir /path/to/clustering_output \\
      --marker-channels "msln:2,pm:1" \\
      --exclude-channels "3" \\
      --feature-groups "morph,sam2,channel"

Dependencies: umap-learn, hdbscan, anndata, matplotlib, pandas
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Default islet marker mapping (backward compatibility)
_ISLET_MARKER_DEFAULTS = {
    'alpha': 2,   # Gcg
    'beta': 3,    # Ins
    'delta': 5,   # Sst
}


def sanitize_for_json(obj):
    """Recursively replace NaN/inf with None in nested structures.

    json.dump(default=) only fires for non-serializable types.
    Python float NaN IS serializable (outputs non-standard "NaN" token),
    so we must walk the structure recursively to catch them.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def parse_marker_channels(marker_str):
    """Parse --marker-channels string into {name: channel_index} dict.

    Format: "name1:ch_idx1,name2:ch_idx2" e.g. "msln:2,pm:1" or "alpha:2,beta:3,delta:5"
    Returns: {'msln': 2, 'pm': 1}
    """
    if not marker_str:
        return None
    result = {}
    for pair in marker_str.split(','):
        pair = pair.strip()
        if ':' not in pair:
            print(f"WARNING: Ignoring malformed marker-channel pair '{pair}' (expected name:index)")
            continue
        name, idx_str = pair.split(':', 1)
        name = name.strip()
        try:
            idx = int(idx_str.strip())
        except ValueError:
            print(f"WARNING: Ignoring non-integer channel index in '{pair}'")
            continue
        result[name] = idx
    return result if result else None


def parse_exclude_channels(exclude_str):
    """Parse --exclude-channels string into set of channel indices.

    Format: "3" or "0,3,5"
    Returns: {3} or {0, 3, 5}
    """
    if not exclude_str:
        return set()
    result = set()
    for part in exclude_str.split(','):
        part = part.strip()
        try:
            result.add(int(part))
        except ValueError:
            print(f"WARNING: Ignoring non-integer channel index '{part}' in --exclude-channels")
    return result


def discover_channels_from_features(detections):
    """Auto-discover channel indices present in detection features.

    Looks for keys matching ch(\\d+)_* pattern and returns sorted unique indices.

    Returns: sorted list of channel indices, e.g. [0, 1, 2, 3, 5]
    """
    ch_indices = set()
    ch_pattern = re.compile(r'^ch(\d+)_')
    for det in detections:
        feats = det.get('features', {})
        for key in feats:
            m = ch_pattern.match(key)
            if m:
                ch_indices.add(int(m.group(1)))
    return sorted(ch_indices)


def discover_marker_channels(detections, exclude_channels=None):
    """Auto-discover marker channels: channels with _mean features, minus excluded.

    Falls back to islet defaults if ch2/ch3/ch5 all present.

    Returns: {name: channel_index} dict or None if no channels found
    """
    available = discover_channels_from_features(detections)
    if exclude_channels:
        available = [ch for ch in available if ch not in exclude_channels]

    if not available:
        return None

    # Check for islet defaults (backward compat)
    islet_channels = set(_ISLET_MARKER_DEFAULTS.values())
    if islet_channels.issubset(set(available)):
        return dict(_ISLET_MARKER_DEFAULTS)

    # Generic: name channels by index
    return {f'ch{ch}': ch for ch in available}


_COLOR_FEATURES = frozenset({
    'red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std',
    'gray_mean', 'gray_std', 'hue_mean', 'saturation_mean', 'value_mean',
    'relative_brightness', 'intensity_variance', 'dark_fraction',
})

_SHAPE_FEATURES = frozenset({
    'area', 'area_um2', 'perimeter', 'circularity', 'solidity', 'aspect_ratio',
    'extent', 'equiv_diameter', 'nuclear_complexity',
})


def classify_feature_group(key):
    """Classify a feature key into its group.

    Returns one of: 'shape', 'color', 'morph' (=shape+color), 'sam2',
    'channel', 'deep', or None if unrecognized.

    'morph' is a virtual group that matches both 'shape' and 'color' features,
    preserving backward compatibility with --feature-groups morph,sam2,channel.
    """
    if re.match(r'^ch\d+', key):
        return 'channel'
    if key.startswith('sam2_'):
        return 'sam2'
    if key.startswith('resnet_') or key.startswith('dinov2_'):
        return 'deep'
    if key in _SHAPE_FEATURES:
        return 'shape'
    if key in _COLOR_FEATURES:
        return 'color'
    # Unknown non-prefixed keys default to shape
    if isinstance(key, str) and not key.startswith(('ch', 'sam2', 'resnet', 'dinov2')):
        return 'shape'
    return None


def get_channel_index_from_key(key):
    """Extract channel index from a channel feature key like 'ch2_mean' or 'ch0_ch2_ratio'.

    Returns set of channel indices referenced by this key.
    """
    return {int(m) for m in re.findall(r'ch(\d+)', key)}


def select_feature_names(detections, feature_groups, exclude_channels=None):
    """Select feature names from detections based on requested groups and exclusions.

    Args:
        detections: list of detection dicts
        feature_groups: set of group names. Recognized groups:
            'morph' (= shape + color, backward compat), 'shape', 'color',
            'sam2', 'channel', 'deep'
        exclude_channels: set of channel indices to exclude from 'channel' group

    Returns: sorted list of feature name strings
    """
    if exclude_channels is None:
        exclude_channels = set()

    # Expand 'morph' into shape + color for backward compatibility
    expanded = set(feature_groups)
    if 'morph' in expanded:
        expanded.discard('morph')
        expanded.add('shape')
        expanded.add('color')

    # Collect all numeric feature names from detections
    all_names = set()
    for det in detections:
        feats = det.get('features', {})
        for k, v in feats.items():
            if isinstance(v, (int, float)):
                all_names.add(k)

    # Filter by group and exclusions
    selected = []
    for name in sorted(all_names):
        group = classify_feature_group(name)
        if group is None or group not in expanded:
            continue

        # For channel features, check if any referenced channel is excluded
        if group == 'channel' and exclude_channels:
            referenced = get_channel_index_from_key(name)
            if referenced & exclude_channels:
                continue

        selected.append(name)

    return selected


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
            # No classifier -- include all
            positive.append(det)

    return positive


def extract_feature_matrix(detections, feature_names):
    """Extract feature matrix from detections using explicit feature name list.

    Returns:
        X: numpy array (n_cells, n_features)
        feature_names: list of feature name strings (same as input, for convenience)
        valid_indices: indices of detections with complete features
    """
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
            if val is None or not isinstance(val, (int, float)) or isinstance(val, bool):
                valid = False
                break
            fval = float(val)
            if not math.isfinite(fval):
                valid = False
                break
            row.append(fval)
        if valid:
            rows.append(row)
            valid_indices.append(i)

    if not rows:
        return None, feature_names, []

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


def auto_label_clusters(detections, labels, valid_indices, marker_channels,
                        norm_ranges=None):
    """Auto-label clusters by dominant marker expression (z-score normalized).

    For each cluster, compute mean of each marker channel's _mean feature.
    When norm_ranges is provided, normalize values before comparing.
    Otherwise, compute population-level z-scores per marker so that channels
    with different baseline intensities are compared fairly.

    Args:
        detections: list of detection dicts
        labels: cluster labels array from HDBSCAN
        valid_indices: indices into detections for valid cells
        marker_channels: dict {name: channel_index} e.g. {'alpha': 2, 'msln': 2}
        norm_ranges: dict mapping 'chN_mean' to (lo, hi) percentile ranges.
            If provided, values are normalized to [0,1] before comparison.

    Returns:
        cluster_label_map: {cluster_id: label_string}
    """
    cluster_labels = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    # Build marker key mapping: {marker_name: feature_key}
    marker_keys = {name: f'ch{idx}_mean' for name, idx in marker_channels.items()}

    # Compute population-level stats per marker for z-score normalization
    # (used when norm_ranges is not provided)
    pop_stats = {}  # {feature_key: (mean, std)}
    if not norm_ranges and marker_keys:
        all_valid_dets = [detections[vi] for vi in valid_indices]
        for label, key in marker_keys.items():
            all_vals = np.array([d.get('features', {}).get(key, 0) for d in all_valid_dets],
                                dtype=np.float64)
            pop_stats[key] = (np.mean(all_vals), max(np.std(all_vals), 1e-12))

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
            elif key in pop_stats:
                # Z-score: how many SDs above population mean
                pmean, pstd = pop_stats[key]
                mean_val = (mean_val - pmean) / pstd
            marker_means[label] = mean_val

        if not marker_means or max(marker_means.values()) == 0:
            cluster_labels[cl] = 'other'
        else:
            best = max(marker_means, key=marker_means.get)
            # Only label if dominant marker is clearly above baseline
            if norm_ranges and marker_means[best] < 0.1:
                cluster_labels[cl] = 'other'
            elif not norm_ranges and marker_means[best] < 0.5:
                # z-score < 0.5 SD above mean — not distinctive
                cluster_labels[cl] = 'other'
            else:
                cluster_labels[cl] = best

    cluster_labels[-1] = 'noise'
    return cluster_labels


def get_marker_mean_keys(marker_channels):
    """Get the chN_mean feature keys for each marker channel.

    Returns: list of (marker_name, feature_key) tuples, sorted by channel index.
    """
    return sorted(
        [(name, f'ch{idx}_mean') for name, idx in marker_channels.items()],
        key=lambda x: marker_channels[x[0]]
    )


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

    # Parse channel configuration
    marker_channels = parse_marker_channels(args.marker_channels)
    exclude_channels = parse_exclude_channels(args.exclude_channels)
    feature_groups = {g.strip() for g in args.feature_groups.split(',')}
    valid_groups = {'morph', 'shape', 'color', 'sam2', 'channel', 'deep'}
    invalid_groups = feature_groups - valid_groups
    if invalid_groups:
        print(f"WARNING: Unknown feature groups ignored: {invalid_groups}")
        feature_groups &= valid_groups

    # Load detections
    print(f"Loading detections from {args.detections}...")
    detections = load_detections(args.detections, threshold=args.threshold)
    print(f"  {len(detections)} RF-positive detections (threshold >= {args.threshold})")

    if len(detections) < args.min_cluster_size:
        print(f"ERROR: Not enough detections ({len(detections)}) for clustering "
              f"(need at least {args.min_cluster_size})")
        sys.exit(1)

    # Discover channels
    all_channels = discover_channels_from_features(detections)
    print(f"  Channels found in features: {all_channels}")
    if exclude_channels:
        print(f"  Excluding channels: {sorted(exclude_channels)}")

    # Resolve marker channels
    if marker_channels is None:
        marker_channels = discover_marker_channels(detections, exclude_channels)
        if marker_channels is None:
            print("WARNING: No marker channels found in features. "
                  "Cluster labeling will use 'other' for all clusters.")
            marker_channels = {}
        else:
            # Check if we got islet defaults
            if marker_channels == _ISLET_MARKER_DEFAULTS:
                print("  Auto-detected islet marker channels: "
                      "alpha=ch2 (Gcg), beta=ch3 (Ins), delta=ch5 (Sst)")
            else:
                print(f"  Auto-detected marker channels: {marker_channels}")
    else:
        # User-specified -- validate channels exist
        for name, idx in list(marker_channels.items()):
            if idx not in all_channels:
                print(f"  WARNING: Marker '{name}' references ch{idx} which has no "
                      f"features in detections (available: {all_channels})")
            if idx in exclude_channels:
                print(f"  WARNING: Marker '{name}' references ch{idx} which is in "
                      f"--exclude-channels. Removing from markers.")
                del marker_channels[name]
        print(f"  Marker channels: {marker_channels}")

    # Extract features
    print("Extracting feature matrix...")
    marker_mean_keys = [f'ch{idx}_mean' for idx in marker_channels.values()]

    if args.marker_only:
        if not marker_channels:
            print("ERROR: --marker-only requires marker channels but none found")
            sys.exit(1)
        feature_names = sorted(marker_mean_keys)
        print(f"  Using normalized marker channels only: {feature_names}")
        X, feature_names, valid_indices = extract_feature_matrix(
            detections, feature_names
        )
    else:
        feature_names = select_feature_names(
            detections, feature_groups, exclude_channels
        )
        print(f"  Selected {len(feature_names)} features from groups {sorted(feature_groups)}")
        # Print breakdown by group
        group_counts = {}
        for fn in feature_names:
            g = classify_feature_group(fn)
            group_counts[g] = group_counts.get(g, 0) + 1
        for g in sorted(group_counts):
            print(f"    {g}: {group_counts[g]} features")

        X, feature_names, valid_indices = extract_feature_matrix(
            detections, feature_names
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
        # No StandardScaler needed -- already [0, 1] and all same units
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
    labels = clusterer.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # Auto-label clusters
    cluster_label_map = auto_label_clusters(
        detections, labels, valid_indices, marker_channels, norm_ranges=norm_ranges
    )
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
    clustered_path = output_dir / 'detections_clustered.json'
    with open(clustered_path, 'w') as f:
        json.dump(sanitize_for_json(detections), f, indent=2)
    print(f"  Saved: {clustered_path}")

    # Build summary DataFrame
    marker_mean_pairs = get_marker_mean_keys(marker_channels)  # [(name, key), ...]
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
        for marker_name, key in marker_mean_pairs:
            row[key] = feats.get(key, 0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Cluster summary -- dynamically build aggregation for available markers
    agg_dict = {'n_cells': ('uid', 'count')}
    for marker_name, key in marker_mean_pairs:
        col_label = f'{key}_{marker_name}'
        if key in df.columns:
            agg_dict[col_label] = (key, 'mean')
    agg_dict['x_mean'] = ('x', 'mean')
    agg_dict['y_mean'] = ('y', 'mean')
    agg_dict['x_std'] = ('x', 'std')
    agg_dict['y_std'] = ('y', 'std')

    summary = df.groupby('cluster_label').agg(**agg_dict).round(2)

    summary_path = output_dir / 'cluster_summary.csv'
    summary.to_csv(summary_path)
    print(f"  Saved: {summary_path}")
    print(summary.to_string())

    # Spatial CSV
    csv_path = output_dir / 'spatial.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # AnnData export
    try:
        import anndata
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        obs_df = df.copy()
        if obs_df['uid'].duplicated().any():
            obs_df['uid'] = (obs_df['uid'] + '_' +
                             obs_df.groupby('uid').cumcount().astype(str))
        adata = anndata.AnnData(
            X=X_clean,
            obs=obs_df.set_index('uid'),
        )
        adata.var_names = feature_names
        adata.obsm['X_umap'] = embedding
        h5ad_path = output_dir / 'spatial.h5ad'
        adata.write(h5ad_path)
        print(f"  Saved: {h5ad_path}")
    except ImportError:
        print("  Skipping .h5ad export (anndata not installed)")

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Build dynamic color map for cluster labels
        # Use a colormap that supports many labels
        all_label_names = sorted(df['cluster_label'].unique())
        # Fixed colors for known labels
        fixed_colors = {
            'alpha': 'red', 'beta': 'green', 'delta': 'blue',
            'other': 'gray', 'noise': 'lightgray', 'unclassified': 'lightgray',
        }
        # Generate distinct colors for unknown labels using tab10/tab20
        tab_colors = plt.cm.tab10.colors + plt.cm.tab20.colors
        color_idx = 0
        color_map = {}
        for label_name in all_label_names:
            if label_name in fixed_colors:
                color_map[label_name] = fixed_colors[label_name]
            else:
                color_map[label_name] = tab_colors[color_idx % len(tab_colors)]
                color_idx += 1

        # UMAP plot
        fig, ax = plt.subplots(figsize=(10, 8))
        for label_name in all_label_names:
            mask = df['cluster_label'] == label_name
            color = color_map.get(label_name, 'gray')
            ax.scatter(
                df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                c=[color], label=label_name, s=5, alpha=0.6,
            )
        ax.legend(markerscale=4)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'Cell Clustering ({len(df)} cells, {n_clusters} clusters)')
        umap_path = output_dir / 'umap_plot.png'
        fig.savefig(umap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {umap_path}")

        # Violin plot of marker intensities per cluster
        if marker_mean_pairs:
            n_markers = len(marker_mean_pairs)
            fig, axes = plt.subplots(1, n_markers, figsize=(5 * n_markers, 5))
            if n_markers == 1:
                axes = [axes]

            for ax, (marker_name, col) in zip(axes, marker_mean_pairs):
                if col not in df.columns:
                    ax.set_title(f'{marker_name} (ch{marker_channels[marker_name]}) - no data')
                    continue
                cluster_labels_sorted = sorted(df['cluster_label'].unique())
                data = [df.loc[df['cluster_label'] == cl, col].values
                        for cl in cluster_labels_sorted]
                # Filter out clusters with <2 points (violinplot KDE requires >=2)
                valid_mask = [len(d) >= 2 for d in data]
                data_filtered = [d for d, v in zip(data, valid_mask) if v]
                tick_labels = [cl for cl, v in zip(cluster_labels_sorted, valid_mask) if v]
                if data_filtered:
                    parts = ax.violinplot(data_filtered, showmeans=True, showmedians=True)
                    ax.set_xticks(range(1, len(tick_labels) + 1))
                    ax.set_xticklabels(tick_labels, rotation=45)
                ax.set_title(f'{marker_name} (ch{marker_channels[marker_name]})')
                ax.set_ylabel('Intensity')

            fig.suptitle('Marker Expression by Cluster')
            fig.tight_layout()
            violin_path = output_dir / 'marker_violin.png'
            fig.savefig(violin_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {violin_path}")
        else:
            print("  Skipping violin plot (no marker channels)")

    except ImportError:
        print("  Skipping plots (matplotlib not installed)")

    print(f"\nDone! {n_clusters} clusters found in {len(valid_indices)} cells.")

    # Sub-clustering (optional)
    if getattr(args, 'subcluster', False):
        run_subclustering(
            detections, output_dir, marker_channels, exclude_channels,
            subcluster_features=args.subcluster_features,
            subcluster_min_size=args.subcluster_min_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
        # Re-save detections with subcluster fields
        clustered_path = output_dir / 'detections_clustered.json'
        with open(clustered_path, 'w') as f:
            json.dump(sanitize_for_json(detections), f, indent=2)
        print(f"  Updated: {clustered_path} (with subcluster fields)")


def run_subclustering(detections, output_dir, marker_channels, exclude_channels,
                      subcluster_features='morph,sam2', subcluster_min_size=50,
                      n_neighbors=30, min_dist=0.1):
    """Sub-cluster each parent cluster by appearance features (morph+sam2).

    For each parent cluster with enough cells, runs UMAP + HDBSCAN on
    appearance-only features to find morphologically distinct sub-populations.

    Modifies detections in-place, adding:
      - subcluster_id: int (cluster ID within parent)
      - subcluster_label: str ("{parent}_{letter}")
      - sub_umap_x, sub_umap_y: float (UMAP coords within parent)

    Outputs per parent cluster:
      - subclusters/{parent}/umap_subcluster.png
      - subclusters/{parent}/morph_violin.png

    Master output:
      - subclusters/subcluster_summary.csv
    """
    from sklearn.preprocessing import StandardScaler
    try:
        import umap
    except ImportError:
        print("ERROR: umap-learn not installed")
        return
    try:
        import hdbscan
    except ImportError:
        print("ERROR: hdbscan not installed")
        return

    subcluster_dir = output_dir / 'subclusters'
    subcluster_dir.mkdir(exist_ok=True)

    # Feature selection: appearance only
    sub_groups = {g.strip() for g in subcluster_features.split(',')}
    sub_feature_names = select_feature_names(detections, sub_groups, exclude_channels)
    if not sub_feature_names:
        print("ERROR: No features found for subclustering")
        return

    group_counts = {}
    for fn in sub_feature_names:
        g = classify_feature_group(fn)
        group_counts[g] = group_counts.get(g, 0) + 1
    print(f"\n{'='*60}")
    print(f"SUB-CLUSTERING by appearance ({len(sub_feature_names)} features: "
          f"{', '.join(f'{g}:{n}' for g, n in sorted(group_counts.items()))})")
    print(f"{'='*60}")

    # Group detections by parent cluster label
    parent_groups = {}  # {label: [det_index, ...]}
    for i, det in enumerate(detections):
        label = det.get('cluster_label')
        if label and label not in ('noise', 'unclassified'):
            parent_groups.setdefault(label, []).append(i)

    min_for_sub = max(subcluster_min_size * 3, 100)
    master_rows = []

    for parent_label in sorted(parent_groups.keys()):
        det_indices = parent_groups[parent_label]
        if len(det_indices) < min_for_sub:
            print(f"\n  '{parent_label}': {len(det_indices)} cells — "
                  f"skipping (need >= {min_for_sub})")
            continue

        print(f"\n--- Sub-clustering '{parent_label}' ({len(det_indices)} cells) ---")

        # Build feature matrix for this subset only
        sub_dets = [detections[di] for di in det_indices]
        X_sub, _, sub_valid_local = extract_feature_matrix(sub_dets, sub_feature_names)

        if X_sub is None or len(X_sub) < min_for_sub:
            n = 0 if X_sub is None else len(X_sub)
            print(f"  {n} valid cells — skipping")
            continue

        print(f"  Feature matrix: {X_sub.shape[0]} x {X_sub.shape[1]}")

        # Scale
        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X_sub),
                                 nan=0.0, posinf=0.0, neginf=0.0)

        # PCA if many features (SAM2 alone is 256D)
        n_features = X_scaled.shape[1]
        if n_features > 50:
            from sklearn.decomposition import PCA
            n_comp = min(50, X_scaled.shape[0] - 1, n_features)
            pca = PCA(n_components=n_comp, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
            var_expl = pca.explained_variance_ratio_.sum() * 100
            print(f"  PCA: {n_features} -> {n_comp} dims ({var_expl:.1f}% variance)")

        # UMAP
        n_nbrs = min(n_neighbors, len(X_scaled) - 1)
        reducer = umap.UMAP(n_neighbors=n_nbrs, min_dist=min_dist,
                            n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)

        # HDBSCAN
        mcs = max(min(subcluster_min_size, len(X_scaled) // 5), 10)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
        labels = clusterer.fit_predict(embedding)

        n_sc = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"  {n_sc} sub-clusters, {n_noise} noise")
        if n_sc == 0:
            print("  No sub-clusters found")
            continue

        # Map IDs to letters (A, B, C...) sorted by count descending
        label_ids = sorted(set(labels) - {-1})
        counts = {lid: int((labels == lid).sum()) for lid in label_ids}
        sorted_ids = sorted(label_ids, key=lambda x: -counts[x])
        alpha_map = {}
        for i, lid in enumerate(sorted_ids):
            alpha_map[lid] = chr(ord('A') + i) if i < 26 else f'sub{i}'
        alpha_map[-1] = 'noise'

        # Per-subcluster stats
        morph_keys = ['area_um2', 'circularity', 'eccentricity', 'solidity',
                      'aspect_ratio']
        for lid in sorted_ids:
            sc_mask = labels == lid
            sc_local_indices = [sub_valid_local[j]
                                for j, m in enumerate(sc_mask) if m]
            sc_dets = [sub_dets[li] for li in sc_local_indices]

            row = {
                'parent': parent_label,
                'subcluster': alpha_map[lid],
                'label': f"{parent_label}_{alpha_map[lid]}",
                'n_cells': counts[lid],
            }
            for mk in morph_keys:
                vals = [d.get('features', {}).get(mk)
                        for d in sc_dets]
                vals = [v for v in vals
                        if v is not None and isinstance(v, (int, float))]
                row[mk] = round(float(np.mean(vals)), 2) if vals else None

            for mname, midx in sorted(marker_channels.items(),
                                       key=lambda x: x[1]):
                mk = f'ch{midx}_mean'
                vals = [d.get('features', {}).get(mk)
                        for d in sc_dets]
                vals = [v for v in vals if v is not None]
                row[f'{mname}_mean'] = (round(float(np.mean(vals)), 1)
                                        if vals else None)

            master_rows.append(row)

        # Write back to detections
        for local_pos, local_valid_idx in enumerate(sub_valid_local):
            det_idx = det_indices[local_valid_idx]
            sc_id = int(labels[local_pos])
            detections[det_idx]['subcluster_id'] = sc_id
            detections[det_idx]['subcluster_label'] = (
                f"{parent_label}_{alpha_map[sc_id]}")
            detections[det_idx]['sub_umap_x'] = float(embedding[local_pos, 0])
            detections[det_idx]['sub_umap_y'] = float(embedding[local_pos, 1])

        # --- Plots ---
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            parent_dir = subcluster_dir / parent_label
            parent_dir.mkdir(exist_ok=True)

            # UMAP colored by subcluster
            fig, ax = plt.subplots(figsize=(10, 8))
            tab_colors = (list(plt.cm.tab10.colors)
                          + list(plt.cm.tab20.colors))
            for i, lid in enumerate(sorted_ids):
                mask = labels == lid
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=[tab_colors[i % len(tab_colors)]],
                    label=f"{alpha_map[lid]} (n={counts[lid]})",
                    s=5, alpha=0.6,
                )
            noise_mask = labels == -1
            if noise_mask.any():
                ax.scatter(
                    embedding[noise_mask, 0], embedding[noise_mask, 1],
                    c='lightgray', label=f'noise (n={n_noise})',
                    s=3, alpha=0.3,
                )
            ax.legend(markerscale=4)
            ax.set_title(
                f"Sub-clusters of '{parent_label}' ({len(X_scaled)} cells)")
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            umap_path = parent_dir / 'umap_subcluster.png'
            fig.savefig(umap_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {umap_path}")

            # Morph violin: area + circularity per subcluster
            morph_plot_keys = ['area_um2', 'circularity', 'eccentricity']
            fig, axes = plt.subplots(
                1, len(morph_plot_keys),
                figsize=(5 * len(morph_plot_keys), 5))
            if len(morph_plot_keys) == 1:
                axes = [axes]
            for ax, mk in zip(axes, morph_plot_keys):
                data = []
                tick_labels = []
                for lid in sorted_ids:
                    sc_mask = labels == lid
                    sc_local = [sub_valid_local[j]
                                for j, m in enumerate(sc_mask) if m]
                    vals = [sub_dets[li].get('features', {}).get(mk, 0)
                            for li in sc_local]
                    if len(vals) >= 2:
                        data.append(vals)
                        tick_labels.append(alpha_map[lid])
                if data:
                    parts = ax.violinplot(
                        data, showmeans=True, showmedians=True)
                    ax.set_xticks(range(1, len(tick_labels) + 1))
                    ax.set_xticklabels(tick_labels)
                ax.set_title(mk)
                ax.set_ylabel(mk)
            fig.suptitle(f"Morphology: '{parent_label}' sub-clusters")
            fig.tight_layout()
            violin_path = parent_dir / 'morph_violin.png'
            fig.savefig(violin_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {violin_path}")

            # Marker violin per subcluster
            if marker_channels:
                marker_pairs = get_marker_mean_keys(marker_channels)
                n_markers = len(marker_pairs)
                fig, axes = plt.subplots(1, n_markers,
                                         figsize=(5 * n_markers, 5))
                if n_markers == 1:
                    axes = [axes]
                for ax, (mname, mkey) in zip(axes, marker_pairs):
                    data = []
                    tick_labels = []
                    for lid in sorted_ids:
                        sc_mask = labels == lid
                        sc_local = [sub_valid_local[j]
                                    for j, m in enumerate(sc_mask) if m]
                        vals = [sub_dets[li].get('features', {}).get(mkey, 0)
                                for li in sc_local]
                        if len(vals) >= 2:
                            data.append(vals)
                            tick_labels.append(alpha_map[lid])
                    if data:
                        parts = ax.violinplot(
                            data, showmeans=True, showmedians=True)
                        ax.set_xticks(range(1, len(tick_labels) + 1))
                        ax.set_xticklabels(tick_labels)
                    ax.set_title(f'{mname} (ch{marker_channels[mname]})')
                    ax.set_ylabel('Intensity')
                fig.suptitle(f"Markers: '{parent_label}' sub-clusters")
                fig.tight_layout()
                mk_violin_path = parent_dir / 'marker_violin.png'
                fig.savefig(mk_violin_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {mk_violin_path}")

        except ImportError:
            print("  Skipping plots (matplotlib not installed)")

    # Master summary
    if master_rows:
        summary_df = pd.DataFrame(master_rows)
        summary_path = subcluster_dir / 'subcluster_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSub-cluster summary:")
        print(summary_df.to_string(index=False))
        print(f"\nSaved: {summary_path}")

    return master_rows


def main():
    parser = argparse.ArgumentParser(
        description='Feature-based clustering of cell detections (any cell type)'
    )
    parser.add_argument('--detections', default=None,
                        help='Path to detections JSON file (required unless --subcluster-input)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for clustering results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum rf_prediction score (default: 0.5)')
    parser.add_argument('--marker-channels', type=str, default=None,
                        help='Marker channels for cluster labeling: "name:ch_idx,..." '
                        'e.g. "msln:2,pm:1" or "alpha:2,beta:3,delta:5". '
                        'If not given, auto-detects from features (islet defaults if ch2/ch3/ch5 present)')
    parser.add_argument('--exclude-channels', type=str, default=None,
                        help='Channel indices to exclude from feature matrix: "3" or "0,3,5"')
    parser.add_argument('--feature-groups', type=str, default='morph,sam2,channel',
                        help='Comma-separated feature groups: morph (=shape+color), '
                        'shape, color, sam2, channel, deep (default: "morph,sam2,channel")')
    parser.add_argument('--marker-only', action='store_true',
                        help='Use only normalized marker channel _mean features '
                        '(population p1-p99.5 percentile stretch)')
    parser.add_argument('--n-neighbors', type=int, default=30,
                        help='UMAP n_neighbors (default: 30)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist (default: 0.1)')
    parser.add_argument('--min-cluster-size', type=int, default=50,
                        help='HDBSCAN min_cluster_size (default: 50)')
    parser.add_argument('--min-samples', type=int, default=None,
                        help='HDBSCAN min_samples (default: None, uses min_cluster_size)')
    parser.add_argument('--subcluster', action='store_true',
                        help='After main clustering, sub-cluster each parent cluster '
                        'by appearance features (morph+sam2)')
    parser.add_argument('--subcluster-input', type=str, default=None,
                        help='Path to pre-clustered detections JSON for standalone '
                        'subclustering (skips main clustering)')
    parser.add_argument('--subcluster-features', type=str, default='shape,sam2',
                        help='Feature groups for subclustering (default: "shape,sam2")')
    parser.add_argument('--subcluster-min-size', type=int, default=50,
                        help='HDBSCAN min_cluster_size for sub-clusters (default: 50)')
    args = parser.parse_args()

    if args.subcluster_input:
        # Standalone subclustering on pre-clustered detections
        pass  # --detections not needed
    elif not args.detections:
        parser.error("--detections is required (unless using --subcluster-input)")

    if args.subcluster_input:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        marker_channels = parse_marker_channels(args.marker_channels)
        exclude_channels = parse_exclude_channels(args.exclude_channels)

        print(f"Loading pre-clustered detections from {args.subcluster_input}...")
        with open(args.subcluster_input) as f:
            detections = json.load(f)
        print(f"  {len(detections)} detections")

        if marker_channels is None:
            marker_channels = (discover_marker_channels(detections, exclude_channels)
                               or {})
            if marker_channels:
                print(f"  Auto-detected marker channels: {marker_channels}")

        run_subclustering(
            detections, output_dir, marker_channels, exclude_channels,
            subcluster_features=args.subcluster_features,
            subcluster_min_size=args.subcluster_min_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )

        # Save enriched detections
        out_path = output_dir / 'detections_subclustered.json'
        with open(out_path, 'w') as f:
            json.dump(sanitize_for_json(detections), f, indent=2)
        print(f"\nSaved: {out_path}")
        return

    run_clustering(args)


if __name__ == '__main__':
    main()
