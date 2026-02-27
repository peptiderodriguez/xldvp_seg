#!/usr/bin/env python3
"""
Automatic tissue zone assignment using spatially-constrained agglomerative clustering.

Discovers spatially coherent zones with distinct marker profiles — handles laminar
layers, circular nuclei, irregular blobs, anything. Clusters cells by marker
similarity while forcing clusters to be spatially contiguous.

Input:  detection JSON from run_segmentation.py (tissue_pattern, islet, etc.)
Output: annotated JSON with zone_id/zone_label per cell, zone map visualizations

Usage:
    # Auto-discover zones
    python scripts/assign_tissue_zones.py \
        --detections tissue_pattern_detections.json \
        --output-dir zones/

    # Manual zone count
    python scripts/assign_tissue_zones.py \
        --detections tissue_pattern_detections.json \
        --n-zones 6 \
        --output-dir zones/

    # Tune spatial vs marker weight
    python scripts/assign_tissue_zones.py \
        --detections tissue_pattern_detections.json \
        --spatial-weight 0.5 \
        --output-dir zones/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


# ── Loading ───────────────────────────────────────────────────────────

_SKIP_FEATURE_KEYS = {
    'global_center_um', 'tile_origin', 'rf_prediction', 'mask_filename',
    'detection_channel', 'uid', 'mask_label', 'tile_idx',
}

_DEEP_FEATURE_PREFIXES = ('resnet_', 'dinov2_')


def _should_exclude_key(key, exclude_channels):
    """Check if a feature key should be excluded based on channel indices.

    Matches:
      - ch{idx}_*  (e.g. ch3_mean, ch3_std, ch3_p99)
      - *_ch{idx}_* cross-channel keys (e.g. ch0_ch3_ratio, ch0_ch3_diff)
      - ch{idx} in ratio/diff keys where it appears as either operand
    """
    if not exclude_channels:
        return False
    for idx in exclude_channels:
        tag = f'ch{idx}'
        # Matches ch{idx}_ at start (per-channel features)
        if key.startswith(tag + '_'):
            return True
        # Matches _ch{idx}_ in middle (cross-channel ratios/diffs like ch0_ch3_ratio)
        if f'_{tag}_' in key:
            return True
        # Matches _ch{idx} at end (shouldn't normally happen but be safe)
        if key.endswith(f'_{tag}'):
            return True
    return False


def _auto_detect_feature_keys(detections, mode='channels', exclude_channels=None):
    """Auto-detect feature keys from detection dicts.

    Args:
        mode: 'channels' — only ch*_mean keys (default, backward compat)
              'all' — all numeric features except deep features and metadata
              'morph_sam2' — morphological + SAM2 embeddings (no ch*_mean, no deep)
        exclude_channels: Set of channel indices to exclude (e.g. {3} removes ch3_*)

    Returns sorted list of feature key names.
    """
    key_set = set()
    sample = detections[:min(50, len(detections))]
    for det in sample:
        feats = det.get('features', det)
        for k, v in feats.items():
            if k in _SKIP_FEATURE_KEYS:
                continue
            if not isinstance(v, (int, float)):
                continue
            if any(k.startswith(p) for p in _DEEP_FEATURE_PREFIXES):
                continue

            if mode == 'channels':
                if k.startswith('ch') and k.endswith('_mean'):
                    key_set.add(k)
            elif mode == 'morph_sam2':
                if not k.startswith('ch'):
                    key_set.add(k)
            else:  # 'all'
                key_set.add(k)

    if exclude_channels:
        key_set = {k for k in key_set if not _should_exclude_key(k, exclude_channels)}

    return sorted(key_set)


def load_and_prepare(detections_path, marker_channels=None, min_score=0.0,
                     feature_mode='channels', n_pca=None, exclude_channels=None):
    """Load detection JSON, extract positions + features.

    Args:
        detections_path: Path to detection JSON
        marker_channels: List of feature keys to use. If None, auto-detect
            based on feature_mode.
        min_score: Minimum rf_prediction score to include
        feature_mode: 'channels' (ch*_mean only), 'all' (morph+SAM2+channels),
            'morph_sam2' (morph+SAM2, no channel means)
        n_pca: If set, reduce features to this many dimensions via PCA after
            z-scoring. Useful when feature count is high (>50).
        exclude_channels: Set of channel indices to exclude (e.g. {3}).

    Returns:
        detections: Original detection list (filtered)
        positions: (N, 2) array of [x_um, y_um]
        features: (N, n_features) array of z-scored (and optionally PCA'd) features
        channel_keys: List of feature key names used
    """
    with open(detections_path) as f:
        data = json.load(f)

    # Handle both flat list and wrapped format
    if isinstance(data, dict):
        raw = data.get('detections', data.get('features', []))
    else:
        raw = data

    # Filter by score and require position
    detections = []
    for det in raw:
        feats = det.get('features', det)
        center_um = det.get('global_center_um') or feats.get('global_center_um')
        if center_um is None:
            continue

        rf = det.get('rf_prediction')
        if rf is None:
            rf = feats.get('rf_prediction')
        if rf is not None and float(rf) < min_score:
            continue

        detections.append(det)

    if not detections:
        print("ERROR: No cells with positions found after filtering")
        sys.exit(1)

    # Auto-detect feature keys
    if marker_channels is None:
        channel_keys = _auto_detect_feature_keys(detections, mode=feature_mode,
                                                  exclude_channels=exclude_channels)
    else:
        # Apply exclusion to explicitly provided keys too
        if exclude_channels:
            channel_keys = [k for k in marker_channels
                            if not _should_exclude_key(k, exclude_channels)]
        else:
            channel_keys = marker_channels

    if not channel_keys:
        print("ERROR: No features found")
        sys.exit(1)

    # Extract arrays
    positions = np.zeros((len(detections), 2))
    features_raw = np.zeros((len(detections), len(channel_keys)))

    for i, det in enumerate(detections):
        feats = det.get('features', det)
        center_um = det.get('global_center_um') or feats.get('global_center_um')
        positions[i] = [float(center_um[0]), float(center_um[1])]
        for j, key in enumerate(channel_keys):
            val = feats.get(key, 0)
            fval = float(val) if val is not None else 0.0
            features_raw[i, j] = fval if not np.isnan(fval) else 0.0

    # Z-score features
    scaler = StandardScaler()
    features = scaler.fit_transform(features_raw)

    # Optional PCA dimensionality reduction
    if n_pca is not None and n_pca < features.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_pca, random_state=42)
        features = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {len(channel_keys)} features -> {n_pca} components "
              f"({explained:.1%} variance explained)")

    print(f"Loaded {len(detections)} cells, {len(channel_keys)} features")
    print(f"  Features: {channel_keys[:10]}{'...' if len(channel_keys) > 10 else ''}")
    print(f"  Tissue extent: {np.ptp(positions[:, 0]):.0f} x {np.ptp(positions[:, 1]):.0f} um")

    return detections, positions, features, channel_keys


# ── Clustering ────────────────────────────────────────────────────────

def build_spatial_graph(positions, k_neighbors=15):
    """Build k-NN spatial connectivity graph from cell positions.

    Returns sparse connectivity matrix suitable for AgglomerativeClustering.
    """
    k = min(k_neighbors, len(positions) - 1)
    if k < 1:
        return None
    graph = kneighbors_graph(positions, n_neighbors=k, mode='connectivity',
                             include_self=False)
    # Symmetrize (agglomerative needs symmetric connectivity)
    graph = graph + graph.T
    graph[graph > 1] = 1
    return graph


def find_zones_elbow(features, connectivity, min_zones=3, max_zones=20,
                     scale=1.0):
    """Find natural n_zones via dendrogram merge distance elbow.

    Args:
        scale: Zone count scale factor (must be positive; 0.5 = 2x more zones).

    Runs agglomerative clustering ONCE with compute_distances=True, examines
    top-level merge distances, and finds the elbow (Kneedle method -- max
    perpendicular distance from first-to-last line) to determine the natural
    cluster count. Much faster than silhouette sweep (1 fit vs 18).

    Args:
        features: (N, D) feature matrix (already combined with spatial)
        connectivity: Sparse k-NN graph
        min_zones: Floor for zone count
        max_zones: Ceiling for zone count
        scale: Divide natural k by this. 0.5 = 2x more zones, 2.0 = half.

    Returns:
        labels: (N,) zone assignments
        chosen_k: Number of zones chosen
        elbow_info: Dict with 'natural_k' and 'merge_dists' (for plotting)
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    # Run once at min_zones -- compute_distances gives ALL N-1 merge distances
    model = AgglomerativeClustering(
        n_clusters=min_zones, connectivity=connectivity, linkage='ward',
        compute_distances=True,
    )
    model.fit(features)

    dists = model.distances_
    n_top = min(max_zones, len(dists))
    top_dists = dists[-n_top:]

    # Kneedle: max perpendicular distance to line from first to last point
    # top_dists[j] = merge from (n_top-j+1) to (n_top-j) clusters
    n = len(top_dists)
    if n < 3:
        natural_k = min_zones
    else:
        x = np.arange(n, dtype=float)
        y = top_dists.copy()

        x_norm = (x - x[0]) / max(x[-1] - x[0], 1e-12)
        y_norm = (y - y[0]) / max(y[-1] - y[0], 1e-12)

        dx = x_norm[-1] - x_norm[0]
        dy = y_norm[-1] - y_norm[0]
        line_len = np.sqrt(dx**2 + dy**2)

        if line_len < 1e-12:
            natural_k = (min_zones + max_zones) // 2
        else:
            perp_dist = np.abs(
                dy * x_norm - dx * y_norm
                + x_norm[-1] * y_norm[0] - y_norm[-1] * x_norm[0]
            ) / line_len
            if perp_dist.max() < 0.01:
                # No clear elbow (constant/linear merge distances)
                natural_k = (min_zones + max_zones) // 2
            else:
                elbow_idx = np.argmax(perp_dist)
                natural_k = n_top - elbow_idx

    natural_k = max(min_zones, min(max_zones, natural_k))

    # Apply scale factor
    scaled_k = int(round(natural_k / scale))
    scaled_k = max(min_zones, min(max_zones, scaled_k))

    print(f"  Dendrogram elbow: natural k={natural_k}", end='')
    if scale != 1.0:
        print(f", scale={scale} -> k={scaled_k}")
    else:
        print(f" -> k={scaled_k}")

    # Re-run at chosen k if different from min_zones
    if scaled_k != min_zones:
        model2 = AgglomerativeClustering(
            n_clusters=scaled_k, connectivity=connectivity, linkage='ward',
        )
        labels = model2.fit_predict(features)
    else:
        labels = model.labels_

    elbow_info = {'natural_k': natural_k, 'merge_dists': top_dists}
    return labels, scaled_k, elbow_info


def find_zones_fixed(features, connectivity, n_zones):
    """Cluster with a fixed number of zones."""
    model = AgglomerativeClustering(
        n_clusters=n_zones, connectivity=connectivity, linkage='ward',
    )
    labels = model.fit_predict(features)
    return labels


def _subsample_and_propagate(features, positions, k_neighbors, n_zones,
                              spatial_weight, max_direct_cells, min_zones,
                              max_zones, scale):
    """Cluster a subsample and propagate labels to all cells via k-NN.

    For datasets > max_direct_cells, randomly samples max_direct_cells cells,
    runs full agglomerative clustering on the subsample, then assigns each
    remaining cell to the zone of its nearest spatial neighbor in the subsample.

    Returns:
        labels: (N,) zone assignments for ALL cells
        n_found: Number of zones
        elbow_info: Dict with elbow info (or None if fixed k)
    """
    N = len(features)
    rng = np.random.RandomState(42)
    sub_idx = np.sort(rng.choice(N, max_direct_cells, replace=False))

    print(f"\n  Subsampling: {N:,} -> {max_direct_cells:,} cells for clustering")

    sub_features = features[sub_idx]
    sub_positions = positions[sub_idx]

    # Build connectivity on subsample positions
    sub_connectivity = build_spatial_graph(sub_positions, k_neighbors=k_neighbors)

    # Augment with spatial coordinates
    if spatial_weight > 0:
        pos_scaled = StandardScaler().fit_transform(sub_positions)
        combined = np.hstack([
            sub_features * (1.0 - spatial_weight),
            pos_scaled * spatial_weight,
        ])
    else:
        combined = sub_features

    # Cluster the subsample
    if n_zones is not None:
        print(f"  Clustering subsample into {n_zones} zones (fixed)...")
        labels_sub = find_zones_fixed(combined, sub_connectivity, n_zones)
        n_found = n_zones
        elbow_info = None
    else:
        print(f"  Auto-selecting zones (elbow, range {min_zones}-{max_zones})...")
        labels_sub, n_found, elbow_info = find_zones_elbow(
            combined, sub_connectivity, min_zones=min_zones,
            max_zones=max_zones, scale=scale,
        )

    # Propagate to remaining cells via nearest spatial neighbor
    remaining_mask = np.ones(N, dtype=bool)
    remaining_mask[sub_idx] = False
    n_remaining = remaining_mask.sum()

    print(f"  Propagating labels to {n_remaining:,} remaining cells...")
    tree = KDTree(sub_positions)

    labels_all = np.empty(N, dtype=int)
    labels_all[sub_idx] = labels_sub

    _, nn_idx = tree.query(positions[remaining_mask])
    labels_all[remaining_mask] = labels_sub[nn_idx]

    print(f"  Subsample + propagation complete: {n_found} zones")
    return labels_all, n_found, elbow_info


def find_zones(features, positions, connectivity, n_zones=None, spatial_weight=0.3,
               min_zones=3, max_zones=20, max_direct_cells=250000, scale=1.0,
               k_neighbors=15):
    """Main zone-finding entry point.

    If spatial_weight > 0, augments the marker feature matrix with scaled
    spatial coordinates before clustering.

    For large datasets (> max_direct_cells), subsamples, clusters the subsample,
    then propagates labels to remaining cells via nearest spatial neighbor.

    Args:
        features: (N, n_markers) z-scored marker features
        positions: (N, 2) spatial positions in um
        connectivity: Sparse k-NN graph (used for direct path)
        n_zones: Fixed number of zones (None = auto via elbow)
        spatial_weight: Weight for spatial coordinates (0 = markers only, 1 = equal)
        min_zones: Min zones for auto search
        max_zones: Max zones for auto search
        max_direct_cells: Subsample threshold (default: 250K)
        scale: Zone count scale factor (0.5 = 2x more zones)
        k_neighbors: k-NN graph connectivity (for subsample path)

    Returns:
        labels: (N,) zone assignments
        n_zones_found: Actual number of zones
        elbow_info: Dict with 'natural_k' and 'merge_dists' (None if fixed k)
    """
    N = len(features)

    # Guard: degenerate cases
    if N <= 1:
        return np.zeros(N, dtype=int), max(1, N), None
    min_zones = min(min_zones, N)
    max_zones = min(max_zones, N)
    if n_zones is not None:
        n_zones = min(n_zones, N)

    # Subsample + propagate for large datasets
    if N > max_direct_cells:
        labels, n_found, elbow_info = _subsample_and_propagate(
            features, positions, k_neighbors=k_neighbors, n_zones=n_zones,
            spatial_weight=spatial_weight, max_direct_cells=max_direct_cells,
            min_zones=min_zones, max_zones=max_zones, scale=scale,
        )
        return labels, n_found, elbow_info

    # Direct clustering path
    # Augment features with spatial coordinates
    if spatial_weight > 0:
        pos_scaled = StandardScaler().fit_transform(positions)
        pos_weighted = pos_scaled * spatial_weight
        marker_weighted = features * (1.0 - spatial_weight)
        combined = np.hstack([marker_weighted, pos_weighted])
    else:
        combined = features

    if n_zones is not None:
        print(f"\nClustering into {n_zones} zones (fixed)...")
        labels = find_zones_fixed(combined, connectivity, n_zones)
        return labels, n_zones, None
    else:
        print(f"\nAuto-selecting zones (elbow, range {min_zones}-{max_zones})...")
        labels, best_k, elbow_info = find_zones_elbow(
            combined, connectivity, min_zones=min_zones, max_zones=max_zones,
            scale=scale,
        )
        return labels, best_k, elbow_info


# ── Post-processing ──────────────────────────────────────────────────

def merge_small_zones(labels, positions, min_cells=20):
    """Merge zones with fewer than min_cells into their nearest spatial neighbor.

    Uses KDTree to find the nearest cell in a different (large enough) zone.
    """
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    small_zones = set(unique[counts < min_cells])

    if not small_zones:
        return labels

    large_mask = np.array([l not in small_zones for l in labels])
    if not large_mask.any():
        # All zones are small — keep them as-is
        return labels

    large_tree = KDTree(positions[large_mask])
    large_labels = labels[large_mask]

    for zone_id in small_zones:
        zone_mask = labels == zone_id
        zone_positions = positions[zone_mask]
        _, nearest_idx = large_tree.query(zone_positions)
        nearest_labels = large_labels[nearest_idx]
        # Assign each cell to the label of its nearest large-zone neighbor
        labels[zone_mask] = nearest_labels

    n_merged = len(small_zones)
    print(f"  Merged {n_merged} small zones (< {min_cells} cells)")
    return labels


def renumber_labels(labels):
    """Renumber labels to 0..k-1 contiguously."""
    unique = sorted(set(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in labels])


# ── Characterization ─────────────────────────────────────────────────

def characterize_zones(labels, positions, features_raw, channel_keys):
    """Compute per-zone summary statistics.

    Args:
        labels: (N,) zone assignments
        positions: (N, 2) positions in um
        features_raw: (N, n_markers) raw (un-z-scored) marker values
        channel_keys: List of channel key names

    Returns:
        List of zone metadata dicts
    """
    zones = []
    for zone_id in sorted(set(labels)):
        mask = labels == zone_id
        zone_pos = positions[mask]
        zone_feats = features_raw[mask]

        # Marker profile: mean and std per channel
        marker_profile = {}
        for j, key in enumerate(channel_keys):
            ch_name = key.replace('_mean', '')
            marker_profile[ch_name] = {
                'mean': float(zone_feats[:, j].mean()),
                'std': float(zone_feats[:, j].std()),
                'median': float(np.median(zone_feats[:, j])),
            }

        # Spatial stats
        centroid = zone_pos.mean(axis=0)
        x_span = np.ptp(zone_pos[:, 0])
        y_span = np.ptp(zone_pos[:, 1])

        zones.append({
            'zone_id': int(zone_id),
            'n_cells': int(mask.sum()),
            'centroid_um': [float(centroid[0]), float(centroid[1])],
            'extent_um': [float(x_span), float(y_span)],
            'marker_profile': marker_profile,
        })

    return zones


def auto_label_zones(zone_metadata, channel_keys, channel_names=None,
                     features_raw=None):
    """Name zones by their dominant marker using greedy cross-zone matching.

    Uses a two-pass approach:
    1. For each channel, rank zones by mean expression. Build a zone-by-channel
       z-score matrix (z-scored across zones, per channel).
    2. Greedy assignment: repeatedly find the (zone, channel) pair with the
       highest z-score, assign that label, remove both from contention.
    This ensures unique labels — no two zones share the same marker label.

    Labels like "Slc17a7-high", "Gad1-high", "mixed".

    Args:
        zone_metadata: List from characterize_zones()
        channel_keys: Channel feature keys
        channel_names: Optional friendly names (e.g. ['Slc17a7', 'Htr2a', ...])
        features_raw: Unused (kept for API compat).
    """
    if channel_names is None:
        channel_names = [k.replace('_mean', '').replace('ch', 'ch') for k in channel_keys]

    n_zones = len(zone_metadata)
    n_ch = len(channel_keys)
    ch_names = [k.replace('_mean', '') for k in channel_keys]

    # Build zone-by-channel mean matrix
    zone_means = np.zeros((n_zones, n_ch))
    for i, zone in enumerate(zone_metadata):
        for j, ch_name in enumerate(ch_names):
            zone_means[i, j] = zone['marker_profile'][ch_name]['mean']

    # Z-score each channel ACROSS zones (column-wise) — which zone stands out?
    zone_z = np.zeros_like(zone_means)
    for j in range(n_ch):
        col = zone_means[:, j]
        mu, sigma = col.mean(), col.std()
        if sigma < 1e-12:
            zone_z[:, j] = 0
        else:
            zone_z[:, j] = (col - mu) / sigma

    # Greedy assignment: pick best (zone, channel) pair, assign, remove both
    assigned_zones = {}  # zone_idx -> (ch_idx, z_score)
    available_zones = set(range(n_zones))
    available_channels = set(range(n_ch))

    while available_zones and available_channels:
        best_z = -np.inf
        best_pair = None
        for i in available_zones:
            for j in available_channels:
                if zone_z[i, j] > best_z:
                    best_z = zone_z[i, j]
                    best_pair = (i, j)

        if best_pair is None or best_z < 0.3:
            break

        zi, cj = best_pair
        assigned_zones[zi] = (cj, best_z)
        available_zones.discard(zi)
        available_channels.discard(cj)

    # Apply labels
    for i, zone in enumerate(zone_metadata):
        if i in assigned_zones:
            ch_idx, z_val = assigned_zones[i]
            friendly = channel_names[ch_idx]
            zone['zone_label'] = f"{friendly}-high"
            zone['dominant_channel'] = ch_names[ch_idx]
            zone['dominant_z'] = float(z_val)
        else:
            zone['zone_label'] = 'mixed'
            zone['dominant_channel'] = None
            zone['dominant_z'] = 0.0


def gate_label_zones(zone_metadata, channel_keys, channel_names, features_raw,
                     gate_thresholds=None, method='otsu', percentile=75):
    """Label zones by +/- marker combination using intensity thresholds.

    For each zone, checks whether the zone's mean expression for each marker
    exceeds the threshold. Labels like "Slc17a7+/Htr2a+/Ntrk2-/Gad1-".

    Args:
        zone_metadata: List from characterize_zones()
        channel_keys: Channel feature keys (e.g. ['ch0_mean', ...])
        channel_names: Friendly names (e.g. ['Slc17a7', 'Htr2a', ...])
        features_raw: (N, n_channels) raw marker values (all cells, for threshold calc)
        gate_thresholds: Optional dict {marker_name: threshold}. Missing → auto.
        method: 'otsu' or 'percentile' (default: 'otsu')
        percentile: Percentile cutoff when method='percentile' (default: 75,
            meaning top 25% of cells are positive).
    """
    if gate_thresholds is None:
        gate_thresholds = {}

    ch_names = [k.replace('_mean', '') for k in channel_keys]

    # Compute thresholds on all cells
    thresholds = {}
    for j, (key, name) in enumerate(zip(channel_keys, channel_names)):
        if name in gate_thresholds:
            thresholds[name] = gate_thresholds[name]
            src = 'manual'
        elif method == 'percentile':
            thresholds[name] = float(np.percentile(features_raw[:, j], percentile))
            src = f'p{percentile}'
        else:
            thresholds[name] = otsu_threshold(features_raw[:, j])
            src = 'otsu'
        n_pos = int((features_raw[:, j] > thresholds[name]).sum())
        pct = 100.0 * n_pos / len(features_raw)
        print(f"  {name}: threshold = {thresholds[name]:.1f} ({src}), "
              f"{n_pos}/{len(features_raw)} positive ({pct:.1f}%)")

    # Label each zone by combination
    for zone in zone_metadata:
        parts = []
        for j, (ch_name, name) in enumerate(zip(ch_names, channel_names)):
            zone_mean = zone['marker_profile'][ch_name]['mean']
            is_pos = zone_mean > thresholds[name]
            parts.append(f"{name}{'+'if is_pos else '-'}")
        zone['zone_label'] = '/'.join(parts)
        zone['gate_thresholds'] = {name: float(thresholds[name])
                                    for name in channel_names}

    return thresholds


# ── Marker Gating ────────────────────────────────────────────────────

def otsu_threshold(values):
    """Find optimal threshold using Otsu's method (maximize between-class variance)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    best_thresh = sorted_vals[n // 2]  # fallback to median
    best_var = 0

    # Test ~200 candidate thresholds spanning the data range
    candidates = np.linspace(np.percentile(values, 5), np.percentile(values, 95), 200)
    for t in candidates:
        lo = values[values <= t]
        hi = values[values > t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        w0 = len(lo) / n
        w1 = len(hi) / n
        var_between = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t

    return best_thresh


def gate_cells(features_raw, channel_keys, channel_names, gate_markers,
               gate_thresholds=None, method='otsu', percentile=75):
    """Classify cells into +/- groups by marker thresholds.

    Args:
        features_raw: (N, n_channels) raw marker values
        channel_keys: List of channel keys (e.g. ['ch0_mean', ...])
        channel_names: Friendly names (e.g. ['Slc17a7', ...])
        gate_markers: List of marker names to gate on (must match channel_names)
        gate_thresholds: Optional dict of {marker_name: threshold}.
            Missing markers get auto-threshold via method.
        method: 'otsu' or 'percentile' (default: 'otsu')
        percentile: Percentile cutoff when method='percentile' (default: 75).

    Returns:
        labels: (N,) integer labels for each group
        group_metadata: List of dicts with group info
        thresholds: Dict of {marker_name: threshold_value}
    """
    if gate_thresholds is None:
        gate_thresholds = {}

    # Map marker names to feature column indices
    name_to_idx = {}
    for j, name in enumerate(channel_names):
        name_to_idx[name] = j
        name_to_idx[name.lower()] = j

    gate_indices = []
    gate_names = []
    thresholds = {}

    for marker in gate_markers:
        idx = name_to_idx.get(marker)
        if idx is None:
            idx = name_to_idx.get(marker.lower())
        if idx is None:
            # Try matching channel key directly
            for j, key in enumerate(channel_keys):
                if marker in key:
                    idx = j
                    break
        if idx is None:
            print(f"WARNING: marker '{marker}' not found in channels {channel_names}")
            continue

        gate_indices.append(idx)
        friendly = channel_names[idx]
        gate_names.append(friendly)

        # Determine threshold
        if friendly in gate_thresholds:
            thresholds[friendly] = gate_thresholds[friendly]
        elif marker in gate_thresholds:
            thresholds[friendly] = gate_thresholds[marker]
        elif method == 'percentile':
            thresh = float(np.percentile(features_raw[:, idx], percentile))
            thresholds[friendly] = thresh
            n_pos = int((features_raw[:, idx] > thresh).sum())
            print(f"  {friendly}: p{percentile} threshold = {thresh:.1f} "
                  f"({n_pos}/{len(features_raw)} positive, "
                  f"{100*n_pos/len(features_raw):.1f}%)")
        else:
            thresh = otsu_threshold(features_raw[:, idx])
            thresholds[friendly] = thresh
            n_pos = int((features_raw[:, idx] > thresh).sum())
            print(f"  {friendly}: Otsu threshold = {thresh:.1f} "
                  f"({n_pos}/{len(features_raw)} positive, "
                  f"{100*n_pos/len(features_raw):.1f}%)")

    if not gate_indices:
        print("ERROR: No valid gate markers found")
        sys.exit(1)

    # Classify each cell: binary code from +/- for each gate marker
    n_gates = len(gate_indices)
    binary_codes = np.zeros(len(features_raw), dtype=int)
    for bit, (idx, name) in enumerate(zip(gate_indices, gate_names)):
        positive = features_raw[:, idx] > thresholds[name]
        binary_codes += positive.astype(int) * (2 ** (n_gates - 1 - bit))

    # Build group labels: e.g. "Slc17a7+/Htr2a-"
    n_groups = 2 ** n_gates
    code_to_label = {}
    for code in range(n_groups):
        parts = []
        for bit in range(n_gates):
            is_pos = (code >> (n_gates - 1 - bit)) & 1
            parts.append(f"{gate_names[bit]}{'+'if is_pos else '-'}")
        code_to_label[code] = '/'.join(parts)

    # Renumber to contiguous labels (only groups with cells)
    present_codes = sorted(set(binary_codes))
    code_to_zone = {code: i for i, code in enumerate(present_codes)}
    labels = np.array([code_to_zone[c] for c in binary_codes])

    # Build metadata per group
    group_metadata = []
    for code in present_codes:
        zone_id = code_to_zone[code]
        mask = binary_codes == code
        group_metadata.append({
            'zone_id': zone_id,
            'gate_code': int(code),
            'zone_label': code_to_label[code],
            'n_cells': int(mask.sum()),
        })

    print(f"\nGated on {n_gates} markers → {len(present_codes)} groups:")
    for g in group_metadata:
        print(f"  Group {g['zone_id']}: {g['zone_label']:30s} n={g['n_cells']}")

    return labels, group_metadata, thresholds


# ── Spatial Structure Detection ──────────────────────────────────────

def _kdistance_elbow(positions, k=10):
    """Find optimal DBSCAN eps via k-distance elbow (Kneedle method).

    Computes distance to k-th nearest neighbor for all points, sorts descending,
    and finds the elbow (point of maximum curvature) in the curve.

    Returns eps in the same units as positions (um).
    """
    tree = KDTree(positions)
    dists, _ = tree.query(positions, k=min(k + 1, len(positions)))
    k_dists = np.sort(dists[:, -1])[::-1]  # sorted descending

    # Normalize to [0,1] for curvature computation
    n = len(k_dists)
    x = np.linspace(0, 1, n)
    y_min, y_max = k_dists[-1], k_dists[0]
    if y_max - y_min < 1e-12:
        return float(np.median(k_dists) * 3)
    y = (k_dists - y_min) / (y_max - y_min)

    # Draw line from first to last point; elbow = max perpendicular distance
    # (this is the Kneedle / max-distance-to-diagonal method)
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    line_len = np.sqrt(dx**2 + dy**2)
    if line_len < 1e-12:
        return float(np.median(k_dists) * 3)

    # Perpendicular distance from each point to the line
    perp_dist = np.abs(dy * x - dx * y + x[-1]*y[0] - y[-1]*x[0]) / line_len
    elbow_idx = np.argmax(perp_dist)
    eps = float(k_dists[elbow_idx])

    # Sanity floor: at least median NN * 2
    median_nn = float(np.median(dists[:, 1]))
    eps = max(eps, median_nn * 2)

    return eps


def _compute_kdistances(positions, labels, zone_metadata, k=10):
    """Compute k-distance curves per cell type for plotting.

    Returns dict: {cell_type_label: (sorted_kdists_descending, elbow_eps)}
    """
    result = {}
    for z in zone_metadata:
        type_id = z['zone_id']
        type_label = z['zone_label']
        type_mask = labels == type_id
        type_pos = positions[type_mask]
        if len(type_pos) < k + 1:
            continue
        elbow_eps = _kdistance_elbow(type_pos, k=k)
        tree = KDTree(type_pos)
        dists, _ = tree.query(type_pos, k=min(k + 1, len(type_pos)))
        k_dists = np.sort(dists[:, -1])[::-1]
        result[type_label] = (k_dists, elbow_eps)
    return result


def find_structures_per_type(positions, labels, zone_metadata, eps_um=None,
                              min_cells=10, eps_scale=1.0):
    """Find spatial structures (clusters) within each cell type using DBSCAN.

    For each gated cell type, runs DBSCAN on positions to find spatially
    coherent groups. Isolated cells become "scattered".

    Args:
        positions: (N, 2) positions in um
        labels: (N,) cell type labels from gating
        zone_metadata: List of zone/group metadata dicts
        eps_um: DBSCAN neighborhood radius in um. If None, auto-detect via
            k-distance elbow per cell type.
        min_cells: Minimum cells for a structure (DBSCAN min_samples)
        eps_scale: Scale factor for auto eps (e.g. 0.5 = half elbow)

    Returns:
        structure_labels: (N,) combined labels (type_id * 100 + structure_id)
        structure_metadata: List of structure dicts
    """
    from sklearn.cluster import DBSCAN

    structure_labels = np.full(len(positions), -1, dtype=int)
    structure_metadata = []
    struct_id_counter = 0

    for z in zone_metadata:
        type_id = z['zone_id']
        type_label = z['zone_label']
        type_mask = labels == type_id
        type_pos = positions[type_mask]

        if len(type_pos) < min_cells:
            # Too few cells, all scattered
            idx = np.where(type_mask)[0]
            for i in idx:
                structure_labels[i] = struct_id_counter
            structure_metadata.append({
                'structure_id': struct_id_counter,
                'cell_type': type_label,
                'spatial_label': 'scattered',
                'zone_label': f"{type_label} (scattered)",
                'n_cells': int(type_mask.sum()),
                'centroid_um': [float(type_pos[:, 0].mean()),
                                float(type_pos[:, 1].mean())] if len(type_pos) > 0 else [0, 0],
            })
            struct_id_counter += 1
            continue

        # Auto-determine eps from k-distance elbow
        if eps_um is None:
            local_eps = _kdistance_elbow(type_pos, k=min_cells) * eps_scale
        else:
            local_eps = eps_um

        db = DBSCAN(eps=local_eps, min_samples=min_cells)
        db_labels = db.fit_predict(type_pos)

        # Map back to global indices
        idx = np.where(type_mask)[0]
        n_clusters = max(db_labels.max() + 1, 0)
        noise_mask = db_labels == -1

        for cluster_id in range(n_clusters):
            cluster_mask = db_labels == cluster_id
            cluster_pos = type_pos[cluster_mask]
            cluster_idx = idx[cluster_mask]

            for i in cluster_idx:
                structure_labels[i] = struct_id_counter

            structure_metadata.append({
                'structure_id': struct_id_counter,
                'cell_type': type_label,
                'spatial_label': f"cluster_{cluster_id}",
                'zone_label': f"{type_label} #{cluster_id+1}",
                'n_cells': int(cluster_mask.sum()),
                'centroid_um': [float(cluster_pos[:, 0].mean()),
                                float(cluster_pos[:, 1].mean())],
                'extent_um': [float(np.ptp(cluster_pos[:, 0])),
                              float(np.ptp(cluster_pos[:, 1]))],
                'density_cells_per_mm2': float(
                    cluster_mask.sum() / max(
                        np.ptp(cluster_pos[:, 0]) * np.ptp(cluster_pos[:, 1]) * 1e-6,
                        1e-12)),
            })
            struct_id_counter += 1

        # Noise / scattered cells
        if noise_mask.any():
            noise_idx = idx[noise_mask]
            for i in noise_idx:
                structure_labels[i] = struct_id_counter
            noise_pos = type_pos[noise_mask]
            structure_metadata.append({
                'structure_id': struct_id_counter,
                'cell_type': type_label,
                'spatial_label': 'scattered',
                'zone_label': f"{type_label} (scattered)",
                'n_cells': int(noise_mask.sum()),
                'centroid_um': [float(noise_pos[:, 0].mean()),
                                float(noise_pos[:, 1].mean())],
            })
            struct_id_counter += 1

        print(f"  {type_label}: {n_clusters} structures + "
              f"{noise_mask.sum()} scattered (eps={local_eps:.0f} um)")

    return structure_labels, structure_metadata


def merge_rare_groups(labels, zone_metadata, positions, features_raw,
                      channel_keys, min_group_cells=24):
    """Merge expression groups with fewer total cells into 'other'.

    Operates BEFORE find_structures — merges rare gated groups so DBSCAN
    isn't run on tiny populations. After merging, relabels and rebuilds
    zone_metadata.

    Args:
        labels: (N,) zone_id per cell
        zone_metadata: list of zone dicts from gate_cells()
        positions: (N,2) positions in um
        features_raw: (N, n_channels) raw feature matrix
        channel_keys: list of feature keys
        min_group_cells: groups with fewer cells are merged into 'other'

    Returns:
        labels, zone_metadata (updated in-place label array + new metadata)
    """
    # Count cells per group
    keep = []
    merge_ids = []
    for z in zone_metadata:
        if z['n_cells'] >= min_group_cells:
            keep.append(z)
        else:
            merge_ids.append(z['zone_id'])

    if not merge_ids:
        return labels, zone_metadata

    n_merged = sum(z['n_cells'] for z in zone_metadata if z['zone_id'] in merge_ids)
    merge_set = set(merge_ids)
    print(f"  Merging {len(merge_ids)} rare groups ({n_merged} cells) into 'other'")

    # Assign a new "other" zone_id
    other_id = max(z['zone_id'] for z in zone_metadata) + 1
    for i in range(len(labels)):
        if labels[i] in merge_set:
            labels[i] = other_id

    # Build "other" metadata
    other_mask = labels == other_id
    other_pos = positions[other_mask]
    other_feats = features_raw[other_mask]
    other_meta = {
        'zone_id': other_id,
        'zone_label': 'other',
        'n_cells': int(other_mask.sum()),
        'centroid_um': [float(other_pos[:, 0].mean()),
                        float(other_pos[:, 1].mean())] if other_mask.any() else [0, 0],
        'extent_um': [float(np.ptp(other_pos[:, 0])),
                      float(np.ptp(other_pos[:, 1]))] if other_mask.any() else [0, 0],
        'marker_profile': {},
    }
    for j, key in enumerate(channel_keys):
        ch_name = key.replace('_mean', '')
        vals = other_feats[:, j]
        other_meta['marker_profile'][ch_name] = {
            'mean': float(vals.mean()) if len(vals) > 0 else 0.0,
            'std': float(vals.std()) if len(vals) > 0 else 0.0,
            'median': float(np.median(vals)) if len(vals) > 0 else 0.0,
        }

    # Renumber: keep zones get 0..N-1, other gets N
    new_meta = []
    old_to_new = {}
    for new_id, z in enumerate(keep):
        old_to_new[z['zone_id']] = new_id
        z['zone_id'] = new_id
        new_meta.append(z)
    old_to_new[other_id] = len(new_meta)
    other_meta['zone_id'] = len(new_meta)
    new_meta.append(other_meta)

    new_labels = np.array([old_to_new.get(int(lbl), other_meta['zone_id'])
                           for lbl in labels])

    return new_labels, new_meta


# ── Output ────────────────────────────────────────────────────────────

def assign_and_save(detections, labels, zone_metadata, output_path):
    """Add zone_id and zone_label to each detection and write JSON."""
    # Build zone label lookup
    label_lookup = {z['zone_id']: z['zone_label'] for z in zone_metadata}

    for det, zone_id in zip(detections, labels):
        det['zone_id'] = int(zone_id)
        det['zone_label'] = label_lookup.get(int(zone_id), f'zone_{zone_id}')

    with open(output_path, 'w') as f:
        json.dump(detections, f)

    print(f"Saved {len(detections)} zoned detections to {output_path}")


# ── Visualization ─────────────────────────────────────────────────────

def get_zone_cmap(n_zones):
    """Get a categorical colormap with enough distinct colors."""
    if n_zones <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_zones]
    elif n_zones <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_zones]
    else:
        colors = plt.cm.gist_ncar(np.linspace(0.05, 0.95, n_zones))
    return ListedColormap(colors)


def plot_zone_map(positions, labels, zone_metadata, output_path, title='Tissue Zones',
                  color_by_group=False):
    """Scatter plot of cells colored by zone with legend.

    Args:
        color_by_group: If True, color by expression group (cell_type field)
            instead of individual structure. Structures within the same
            expression group share a color. Useful for gate+structures mode.
    """
    if color_by_group and zone_metadata and 'cell_type' in zone_metadata[0]:
        _plot_zone_map_by_group(positions, labels, zone_metadata, output_path, title)
        return

    n_zones = len(zone_metadata)
    cmap = get_zone_cmap(n_zones)

    fig, ax = plt.subplots(figsize=(12, 10))
    for zone in zone_metadata:
        zid = zone['zone_id']
        mask = labels == zid
        color = cmap(zid / max(n_zones - 1, 1))
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   c=[color], s=3, alpha=0.6, label=f"Z{zid}: {zone['zone_label']} (n={zone['n_cells']})")

    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1), markerscale=3,
              borderaxespad=0)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_zone_map_by_group(positions, labels, zone_metadata, output_path, title):
    """Zone map colored by expression group, not individual structure."""
    # Collect unique expression groups (cell_type)
    groups = []
    seen = set()
    for z in zone_metadata:
        ct = z.get('cell_type', z['zone_label'])
        if ct not in seen:
            groups.append(ct)
            seen.add(ct)

    n_groups = len(groups)
    cmap = get_zone_cmap(n_groups)
    group_to_color = {g: cmap(i / max(n_groups - 1, 1)) for i, g in enumerate(groups)}

    # Build per-group masks (merging all structures of same expression type)
    group_cells = {g: 0 for g in groups}
    group_masks = {g: np.zeros(len(positions), dtype=bool) for g in groups}
    for z in zone_metadata:
        ct = z.get('cell_type', z['zone_label'])
        mask = labels == z['zone_id']
        group_masks[ct] |= mask
        group_cells[ct] += z['n_cells']

    fig, ax = plt.subplots(figsize=(14, 10))
    for g in groups:
        mask = group_masks[g]
        if not mask.any():
            continue
        color = group_to_color[g]
        n = group_cells[g]
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   c=[color], s=3, alpha=0.6, label=f"{g} (n={n})")

    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), markerscale=3,
              borderaxespad=0)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_marker_profiles(zone_metadata, channel_keys, channel_names, output_path):
    """Bar chart of mean marker expression per zone."""
    n_zones = len(zone_metadata)
    n_channels = len(channel_keys)
    cmap = get_zone_cmap(n_zones)

    fig, ax = plt.subplots(figsize=(max(8, n_channels * 1.5), 6))
    x = np.arange(n_channels)
    width = 0.8 / n_zones

    for i, zone in enumerate(zone_metadata):
        means = []
        for key in channel_keys:
            ch_name = key.replace('_mean', '')
            means.append(zone['marker_profile'][ch_name]['mean'])
        color = cmap(i / max(n_zones - 1, 1))
        ax.bar(x + i * width, means, width, color=color, alpha=0.8,
               label=f"Z{zone['zone_id']}: {zone['zone_label']}")

    ax.set_xlabel('Channel')
    ax.set_ylabel('Mean intensity')
    ax.set_title('Marker Expression by Zone')
    ax.set_xticks(x + width * (n_zones - 1) / 2)
    ax.set_xticklabels(channel_names, rotation=30, ha='right')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1),
              borderaxespad=0)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_zone_map_with_density(positions, labels, zone_metadata, output_path,
                               bin_size_um=50, sigma_bins=1.5,
                               color_by_group=False):
    """Zone map overlaid on cell density heatmap."""
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(figsize=(14, 10))

    # Density heatmap background
    xs, ys = positions[:, 0], positions[:, 1]
    x_bins = max(10, int((xs.max() - xs.min()) / bin_size_um))
    y_bins = max(10, int((ys.max() - ys.min()) / bin_size_um))
    heatmap, x_edges, y_edges = np.histogram2d(xs, ys, bins=[x_bins, y_bins])
    heatmap = gaussian_filter(heatmap, sigma=sigma_bins)
    extent = [x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]]  # inverted Y
    ax.imshow(heatmap.T, origin='upper', extent=extent, cmap='Greys', alpha=0.4,
              aspect='equal', interpolation='bilinear')

    # Zone scatter on top — color by group if structures mode
    if color_by_group and zone_metadata and 'cell_type' in zone_metadata[0]:
        groups = []
        seen = set()
        for z in zone_metadata:
            ct = z.get('cell_type', z['zone_label'])
            if ct not in seen:
                groups.append(ct)
                seen.add(ct)
        n_groups = len(groups)
        cmap = get_zone_cmap(n_groups)
        group_to_color = {g: cmap(i / max(n_groups - 1, 1)) for i, g in enumerate(groups)}

        group_masks = {g: np.zeros(len(positions), dtype=bool) for g in groups}
        group_cells = {g: 0 for g in groups}
        for z in zone_metadata:
            ct = z.get('cell_type', z['zone_label'])
            group_masks[ct] |= (labels == z['zone_id'])
            group_cells[ct] += z['n_cells']

        for g in groups:
            mask = group_masks[g]
            if not mask.any():
                continue
            ax.scatter(positions[mask, 0], positions[mask, 1],
                       c=[group_to_color[g]], s=3, alpha=0.5,
                       label=f"{g} (n={group_cells[g]})")
    else:
        n_zones = len(zone_metadata)
        cmap = get_zone_cmap(n_zones)
        for zone in zone_metadata:
            zid = zone['zone_id']
            mask = labels == zid
            color = cmap(zid / max(n_zones - 1, 1))
            ax.scatter(positions[mask, 0], positions[mask, 1],
                       c=[color], s=3, alpha=0.5, label=f"Z{zid}: {zone['zone_label']}")

    ax.set_xlabel('X (um)')
    ax.set_ylabel('Y (um)')
    ax.set_title('Tissue Zones over Density')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1), markerscale=3,
              borderaxespad=0)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_gate_histograms(features_raw, channel_keys, channel_names, gate_markers,
                         thresholds, output_path):
    """Plot histograms of gated marker intensities with threshold lines."""
    # Resolve gate marker indices
    name_to_idx = {}
    for j, name in enumerate(channel_names):
        name_to_idx[name] = j
        name_to_idx[name.lower()] = j

    gate_indices = []
    gate_names = []
    for marker in gate_markers:
        idx = name_to_idx.get(marker)
        if idx is None:
            idx = name_to_idx.get(marker.lower())
        if idx is None:
            for j, key in enumerate(channel_keys):
                if marker in key:
                    idx = j
                    break
        if idx is not None:
            gate_indices.append(idx)
            gate_names.append(channel_names[idx])

    n_gates = len(gate_indices)
    fig, axes = plt.subplots(1, n_gates, figsize=(5 * n_gates, 4))
    if n_gates == 1:
        axes = [axes]

    for ax, idx, name in zip(axes, gate_indices, gate_names):
        vals = features_raw[:, idx]
        ax.hist(vals, bins=80, color='steelblue', alpha=0.7, edgecolor='none')
        thresh = thresholds[name]
        ax.axvline(thresh, color='red', linewidth=2, linestyle='--',
                   label=f'threshold = {thresh:.0f}')
        n_pos = (vals > thresh).sum()
        n_neg = (vals <= thresh).sum()
        ax.set_title(f'{name}')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Cell count')
        ax.legend(fontsize=9)
        ax.text(0.97, 0.95, f'+: {n_pos}\n−: {n_neg}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.suptitle('Marker Gating Thresholds', fontsize=13, y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_kdistance(kdist_data, eps_used, output_path):
    """Plot k-distance curves per cell type with elbow and chosen eps marked.

    Args:
        kdist_data: dict from _compute_kdistances: {label: (sorted_kdists, elbow_eps)}
        eps_used: The eps actually used for DBSCAN (manual or auto)
        output_path: Where to save the plot
    """
    n = len(kdist_data)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 4), squeeze=False)
    axes = axes[0]

    for ax, (label, (k_dists, elbow_eps)) in zip(axes, kdist_data.items()):
        x = np.arange(len(k_dists))
        ax.plot(x, k_dists, color='steelblue', linewidth=1.5)

        # Mark auto elbow
        elbow_idx = np.searchsorted(-k_dists, -elbow_eps)  # find index where k_dists drops below elbow_eps
        elbow_idx = min(elbow_idx, len(k_dists) - 1)
        ax.axhline(elbow_eps, color='green', linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f'auto elbow = {elbow_eps:.0f} um')

        # Mark the eps actually used
        if eps_used is not None and abs(eps_used - elbow_eps) > 1:
            ax.axhline(eps_used, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
                        label=f'eps used = {eps_used:.0f} um')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Cell (sorted by k-dist)')
        ax.set_ylabel('k-distance (um)')
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('k-Distance Curves (elbow = natural DBSCAN eps)', fontsize=12, y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_dendrogram_elbow(elbow_info, chosen_k, output_path):
    """Plot merge distances with elbow point marked.

    Args:
        elbow_info: Dict with 'natural_k' and 'merge_dists' from find_zones_elbow()
        chosen_k: Final zone count (after scaling)
        output_path: Where to save the plot
    """
    if elbow_info is None:
        return

    merge_dists = elbow_info['merge_dists']
    natural_k = elbow_info['natural_k']
    n_top = len(merge_dists)

    # x-axis: number of clusters after each merge
    # merge_dists[j] = merge from (n_top-j+1) to (n_top-j) clusters
    n_clusters = np.arange(n_top, 0, -1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_clusters, merge_dists, 'bo-', linewidth=1.5, markersize=5)
    ax.axvline(natural_k, color='green', linestyle='--', alpha=0.7,
               label=f'elbow k={natural_k}')
    if chosen_k != natural_k:
        ax.axvline(chosen_k, color='red', linestyle='--', alpha=0.7,
                   label=f'final k={chosen_k}')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Merge distance (Ward)')
    ax.set_title('Dendrogram Elbow (merge distances)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Automatic tissue zone assignment via spatially-constrained clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--detections', required=True,
                        help='Path to detection JSON from run_segmentation.py')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for zoned results')

    # Zone parameters
    parser.add_argument('--n-zones', type=int, default=None,
                        help='Fixed number of zones (default: auto via dendrogram elbow)')
    parser.add_argument('--min-zones', type=int, default=3,
                        help='Min zones for auto search (default: 3)')
    parser.add_argument('--max-zones', type=int, default=20,
                        help='Max zones for auto search (default: 20)')
    parser.add_argument('--max-direct-cells', type=int, default=250000,
                        help='Max cells for direct clustering. Above this, subsample '
                             'and propagate via k-NN (default: 250000)')
    parser.add_argument('--n-zones-scale', type=float, default=1.0,
                        help='Scale factor for auto zone count. 0.5 = 2x more zones '
                             '(finer resolution), 2.0 = half (coarser). Default: 1.0')
    parser.add_argument('--spatial-weight', type=float, default=0.15,
                        help='Weight for spatial coordinates vs markers, 0-1 (default: 0.15)')
    parser.add_argument('--k-neighbors', type=int, default=15,
                        help='k-NN spatial graph connectivity (default: 15)')
    parser.add_argument('--min-cells-per-zone', type=int, default=20,
                        help='Merge zones smaller than this (default: 20)')

    # Feature selection
    parser.add_argument('--marker-channels', type=str, default=None,
                        help='Comma-separated channel keys (default: all ch*_mean)')
    parser.add_argument('--channel-names', type=str, default=None,
                        help='Comma-separated friendly channel names for labels')
    parser.add_argument('--min-score', type=float, default=0.0,
                        help='Minimum RF score to include (default: 0.0 = all)')
    parser.add_argument('--all-features', action='store_true',
                        help='Use all features (morph + SAM2 + channels) instead of '
                             'just ch*_mean. For whole-tissue region discovery.')
    parser.add_argument('--morph-sam2', action='store_true',
                        help='Use morph + SAM2 features only (no channel means). '
                             'For tissue region discovery without marker-specific channels.')
    parser.add_argument('--exclude-channels', type=str, default=None,
                        help='Comma-separated channel indices to exclude from zone '
                             'discovery (e.g. "3" or "3,5"). Filters out all features '
                             'starting with ch{idx}_ and cross-channel keys like '
                             'ch0_ch3_ratio, ch0_ch3_diff.')
    parser.add_argument('--n-pca', type=int, default=None,
                        help='Reduce features to N PCA components before clustering. '
                             'Auto-set to 50 when --all-features or --morph-sam2 and '
                             'feature count > 50. Set 0 to disable.')

    # Zone labeling
    parser.add_argument('--label-by-gate', action='store_true',
                        help='Label spatial zones by +/- marker combination instead '
                             'of single dominant marker. Runs spatial agglomerative '
                             'clustering for zone boundaries, then applies Otsu '
                             'thresholds to each zone mean to assign +/- labels. '
                             'E.g. "Slc17a7+/Htr2a+/Ntrk2-/Gad1-".')
    parser.add_argument('--label-gate-thresholds', type=str, default=None,
                        help='Manual thresholds for --label-by-gate (comma-sep, same '
                             'order as --channel-names). Default: auto via --gate-method.')
    parser.add_argument('--gate-method', type=str, default='otsu',
                        choices=['otsu', 'percentile'],
                        help='Threshold method for --label-by-gate and --gate. '
                             '"otsu" = Otsu bimodal split (default). '
                             '"percentile" = top N%% positive.')
    parser.add_argument('--gate-percentile', type=float, default=75,
                        help='Percentile cutoff when --gate-method=percentile. '
                             'Default: 75 (top 25%% are positive).')

    # Marker gating mode
    parser.add_argument('--gate', type=str, default=None,
                        help='Gate markers (comma-sep). Classifies cells into +/- '
                             'combinations. E.g. "Slc17a7,Htr2a" → 4 groups: '
                             'Slc17a7+/Htr2a+, Slc17a7+/Htr2a-, etc.')
    parser.add_argument('--gate-thresholds', type=str, default=None,
                        help='Manual thresholds per gate marker (comma-sep, same order '
                             'as --gate). E.g. "2000,1500". Default: Otsu auto-threshold.')
    parser.add_argument('--positive-only', action='store_true',
                        help='In gate mode, only keep cells positive for the first '
                             'gate marker. Removes the double-negative population.')
    parser.add_argument('--find-structures', action='store_true',
                        help='After gating, find spatial structures (DBSCAN clusters) '
                             'within each cell type. Shows where each population is '
                             'concentrated.')
    parser.add_argument('--structure-eps', type=float, default=None,
                        help='DBSCAN radius in um for structure detection '
                             '(default: auto from nearest-neighbor distances)')
    parser.add_argument('--structure-min-cells', type=int, default=10,
                        help='Minimum cells per spatial structure (default: 10)')
    parser.add_argument('--structure-eps-scale', type=float, default=1.0,
                        help='Scale factor for auto eps (e.g. 0.5 = half elbow, '
                             '2.0 = double). Only used when --structure-eps is not set.')
    parser.add_argument('--min-group-cells', type=int, default=24,
                        help='Merge expression groups with fewer total cells into '
                             '"other" for cleaner visualization (default: 24)')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Open napari viewer after zone assignment')

    args = parser.parse_args()

    if args.n_zones_scale <= 0:
        parser.error("--n-zones-scale must be positive")
    if args.n_zones is not None and args.n_zones_scale != 1.0:
        print("WARNING: --n-zones-scale is ignored when --n-zones is set (fixed zone count)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse channel arguments
    marker_channels = None
    if args.marker_channels:
        marker_channels = [c.strip() for c in args.marker_channels.split(',')]
        # Allow shorthand like "0,1,2" → "ch0_mean,ch1_mean,ch2_mean"
        expanded = []
        for c in marker_channels:
            if c.isdigit():
                expanded.append(f'ch{c}_mean')
            else:
                expanded.append(c)
        marker_channels = expanded

    channel_names = None
    if args.channel_names:
        channel_names = [c.strip() for c in args.channel_names.split(',')]

    # Parse exclude channels
    exclude_channels = None
    if args.exclude_channels:
        exclude_channels = set()
        for idx_str in args.exclude_channels.split(','):
            idx_str = idx_str.strip()
            if idx_str.isdigit():
                exclude_channels.add(int(idx_str))
            else:
                parser.error(f"--exclude-channels: '{idx_str}' is not a valid channel index")
        print(f"Excluding channels: {sorted(exclude_channels)}")

    # Determine feature mode
    if args.morph_sam2:
        feature_mode = 'morph_sam2'
    elif args.all_features:
        feature_mode = 'all'
    else:
        feature_mode = 'channels'

    # Auto PCA for high-dimensional feature modes
    n_pca = args.n_pca
    if n_pca is None and feature_mode in ('all', 'morph_sam2'):
        n_pca = 50  # auto PCA when using full feature sets
    if n_pca == 0:
        n_pca = None  # --n-pca 0 disables

    # ── Load ──────────────────────────────────────────────────────
    detections, positions, features, channel_keys = load_and_prepare(
        args.detections, marker_channels=marker_channels, min_score=args.min_score,
        feature_mode=feature_mode, n_pca=n_pca, exclude_channels=exclude_channels,
    )

    if channel_names is None:
        channel_names = [k.replace('_mean', '') for k in channel_keys]

    # ── Build raw features for characterization (always ch*_mean) ─────
    # Even when clustering on all features, zone profiles use channel means
    ch_mean_keys = _auto_detect_feature_keys(detections, mode='channels',
                                                exclude_channels=exclude_channels)
    if not ch_mean_keys:
        ch_mean_keys = channel_keys  # fallback if no ch*_mean found
    features_raw = np.zeros((len(detections), len(ch_mean_keys)))
    for i, det in enumerate(detections):
        feats = det.get('features', det)
        for j, key in enumerate(ch_mean_keys):
            val = feats.get(key, 0)
            fval = float(val) if val is not None else 0.0
            features_raw[i, j] = fval if not np.isnan(fval) else 0.0
    # Use ch_mean_keys for characterization/plots, channel_keys for clustering
    char_keys = ch_mean_keys
    if channel_names is None or len(channel_names) != len(char_keys):
        char_names = [k.replace('_mean', '') for k in char_keys]
    else:
        char_names = channel_names

    # ── Parse gate args ───────────────────────────────────────────
    gate_markers = None
    gate_thresholds = None
    if args.gate:
        gate_markers = [m.strip() for m in args.gate.split(',')]
        if args.gate_thresholds:
            thresh_vals = [float(t) for t in args.gate_thresholds.split(',')]
            gate_thresholds = dict(zip(gate_markers, thresh_vals))

    elbow_info = None
    n_zones_pre_merge = None
    features_raw_all = features_raw.copy()  # keep unfiltered copy for histograms

    if gate_markers:
        # ── Gating mode ──────────────────────────────────────────
        print(f"\nMarker gating mode: {gate_markers}")
        labels, zone_metadata, thresholds = gate_cells(
            features_raw, char_keys, char_names,
            gate_markers, gate_thresholds,
            method=args.gate_method, percentile=args.gate_percentile,
        )
        # Add marker profiles to gated groups
        for z in zone_metadata:
            mask = labels == z['zone_id']
            zone_feats = features_raw[mask]
            zone_pos = positions[mask]
            z['marker_profile'] = {}
            for j, key in enumerate(char_keys):
                ch_name = key.replace('_mean', '')
                z['marker_profile'][ch_name] = {
                    'mean': float(zone_feats[:, j].mean()),
                    'std': float(zone_feats[:, j].std()),
                    'median': float(np.median(zone_feats[:, j])),
                }
            z['centroid_um'] = [float(zone_pos[:, 0].mean()),
                                float(zone_pos[:, 1].mean())]
            z['extent_um'] = [float(np.ptp(zone_pos[:, 0])),
                              float(np.ptp(zone_pos[:, 1]))]
        n_zones = len(zone_metadata)
    else:
        # ── Spatial clustering mode ───────────────────────────────
        if len(positions) <= args.max_direct_cells:
            print(f"\nBuilding k-NN spatial graph (k={args.k_neighbors})...")
            connectivity = build_spatial_graph(positions, k_neighbors=args.k_neighbors)
        else:
            print(f"\n  {len(positions):,} cells > {args.max_direct_cells:,} max_direct_cells"
                  f" — will subsample")
            connectivity = None  # subsample path builds its own

        # Auto min_zones: at least n_marker_channels so each marker can have its own zone
        min_zones = max(args.min_zones, len(char_keys))
        labels, n_zones, elbow_info = find_zones(
            features, positions, connectivity,
            n_zones=args.n_zones,
            spatial_weight=args.spatial_weight,
            min_zones=min_zones,
            max_zones=max(args.max_zones, min_zones),
            max_direct_cells=args.max_direct_cells,
            scale=args.n_zones_scale,
            k_neighbors=args.k_neighbors,
        )
        n_zones_pre_merge = n_zones  # save for dendrogram plot

        # Merge small zones
        labels = merge_small_zones(labels, positions, min_cells=args.min_cells_per_zone)
        labels = renumber_labels(labels)
        n_zones = len(set(labels))
        print(f"\nFinal: {n_zones} zones")

        # Characterize (always based on channel means for interpretability)
        zone_metadata = characterize_zones(labels, positions, features_raw, char_keys)

        if args.label_by_gate:
            # Label zones by +/- combination (Otsu on zone means)
            print("\nLabeling zones by marker combination (+/-)...")
            label_gate_thresholds = None
            if args.label_gate_thresholds:
                thresh_vals = [float(t) for t in args.label_gate_thresholds.split(',')]
                label_gate_thresholds = dict(zip(char_names, thresh_vals))
            thresholds = gate_label_zones(
                zone_metadata, char_keys, char_names, features_raw,
                gate_thresholds=label_gate_thresholds,
                method=args.gate_method, percentile=args.gate_percentile,
            )
        else:
            auto_label_zones(zone_metadata, char_keys, char_names,
                             features_raw=features_raw)
            thresholds = None

    # ── Positive-only filter (gate mode) ──────────────────────────
    if gate_markers and args.positive_only:
        # Keep only cells positive for the first gate marker
        first_marker = gate_markers[0]
        keep_zones = set()
        for z in zone_metadata:
            # Zone label contains "MarkerName+" if positive for that marker
            if f"{first_marker}+" in z['zone_label']:
                keep_zones.add(z['zone_id'])
        if keep_zones:
            keep_mask = np.array([labels[i] in keep_zones for i in range(len(labels))])
            n_before = len(detections)
            detections = [d for d, k in zip(detections, keep_mask) if k]
            positions = positions[keep_mask]
            features_raw = features_raw[keep_mask]
            labels = labels[keep_mask]

            # Renumber and rebuild metadata
            labels = renumber_labels(labels)
            zone_metadata = [z for z in zone_metadata if z['zone_id'] in keep_zones]
            for new_id, z in enumerate(zone_metadata):
                z['zone_id'] = new_id
            n_zones = len(zone_metadata)
            print(f"\n  --positive-only: kept {first_marker}+ cells: "
                  f"{n_before} -> {len(detections)}")

    for z in zone_metadata:
        print(f"  Zone {z['zone_id']}: {z['zone_label']:30s} "
              f"n={z['n_cells']:5d}  centroid=({z['centroid_um'][0]:.0f}, {z['centroid_um'][1]:.0f})")

    # ── Merge rare expression groups ──────────────────────────────
    if gate_markers and args.min_group_cells > 0:
        labels, zone_metadata = merge_rare_groups(
            labels, zone_metadata, positions, features_raw,
            char_keys, min_group_cells=args.min_group_cells)
        n_zones = len(zone_metadata)

    # ── Find spatial structures within each type ──────────────────
    kdist_data = None
    if gate_markers and args.find_structures:
        # Compute k-distance curves (for auto-eps and plotting)
        kdist_data = _compute_kdistances(
            positions, labels, zone_metadata, k=args.structure_min_cells)

        print(f"\nFinding spatial structures per cell type...")
        labels, zone_metadata = find_structures_per_type(
            positions, labels, zone_metadata,
            eps_um=args.structure_eps,
            min_cells=args.structure_min_cells,
            eps_scale=args.structure_eps_scale,
        )
        # Add marker profiles to structures
        for z in zone_metadata:
            mask = labels == z['structure_id']
            if not mask.any():
                continue
            z_feats = features_raw[mask]
            z_pos = positions[mask]
            z['marker_profile'] = {}
            for j, key in enumerate(char_keys):
                ch_name = key.replace('_mean', '')
                z['marker_profile'][ch_name] = {
                    'mean': float(z_feats[:, j].mean()),
                    'std': float(z_feats[:, j].std()),
                    'median': float(np.median(z_feats[:, j])),
                }
            if 'centroid_um' not in z:
                z['centroid_um'] = [float(z_pos[:, 0].mean()),
                                    float(z_pos[:, 1].mean())]
            if 'extent_um' not in z:
                z['extent_um'] = [float(np.ptp(z_pos[:, 0])),
                                  float(np.ptp(z_pos[:, 1]))]
        # Remap for output: structure_id → zone_id
        for z in zone_metadata:
            z['zone_id'] = z['structure_id']
        n_zones = len(zone_metadata)
        print(f"\n  Total: {n_zones} structures across all cell types")
        for z in zone_metadata:
            print(f"    {z['zone_id']:3d}: {z['zone_label']:35s} n={z['n_cells']:5d}")

    # ── Save ──────────────────────────────────────────────────────
    print("\nSaving outputs...")

    # Annotated detections
    assign_and_save(detections, labels, zone_metadata,
                    output_dir / 'detections_zoned.json')

    # Zone metadata
    meta_output = {
        'n_zones': n_zones,
        'mode': 'gate' if gate_markers else 'cluster',
        'parameters': {
            'spatial_weight': args.spatial_weight,
            'k_neighbors': args.k_neighbors,
            'min_cells_per_zone': args.min_cells_per_zone,
            'n_zones_requested': args.n_zones,
            'min_zones': args.min_zones,
            'max_zones': args.max_zones,
            'clustering_features': channel_keys,
            'marker_channels': char_keys,
            'channel_names': char_names,
            'feature_mode': feature_mode,
            'n_pca': n_pca,
            'min_score': args.min_score,
            'exclude_channels': sorted(exclude_channels) if exclude_channels else None,
        },
        'zones': zone_metadata,
    }
    if gate_markers:
        meta_output['gate_markers'] = gate_markers
        meta_output['gate_thresholds'] = {k: float(v) for k, v in thresholds.items()}
    if args.label_by_gate and thresholds:
        meta_output['label_by_gate'] = True
        meta_output['gate_thresholds'] = {k: float(v) for k, v in thresholds.items()}
    if elbow_info is not None:
        meta_output['elbow'] = {
            'natural_k': int(elbow_info['natural_k']),
            'n_zones_scale': args.n_zones_scale,
            'merge_dists': [float(d) for d in elbow_info['merge_dists']],
        }
    meta_output['parameters']['max_direct_cells'] = args.max_direct_cells
    meta_output['parameters']['n_zones_scale'] = args.n_zones_scale
    with open(output_dir / 'zone_metadata.json', 'w') as f:
        json.dump(meta_output, f)
    print(f"  Saved: {output_dir / 'zone_metadata.json'}")

    # ── Visualize ─────────────────────────────────────────────────
    print("\nGenerating visualizations...")

    is_structures = gate_markers and args.find_structures
    if is_structures:
        title = f'Spatial Structures: {", ".join(gate_markers)} ({n_zones} structures)'
    elif gate_markers:
        title = f'Marker Gating: {", ".join(gate_markers)} ({n_zones} groups)'
    else:
        title = f'Tissue Zones ({n_zones} zones, spatial_weight={args.spatial_weight})'
    plot_zone_map(positions, labels, zone_metadata,
                  output_dir / 'zone_map.png', title=title,
                  color_by_group=is_structures)

    plot_marker_profiles(zone_metadata, char_keys, char_names,
                         output_dir / 'zone_marker_profiles.png')

    plot_zone_map_with_density(positions, labels, zone_metadata,
                               output_dir / 'zone_map_with_density.png',
                               color_by_group=is_structures)

    if elbow_info is not None:
        plot_dendrogram_elbow(elbow_info, n_zones_pre_merge,
                              output_dir / 'dendrogram_elbow.png')

    if gate_markers and thresholds:
        plot_gate_histograms(features_raw_all, char_keys, char_names,
                             gate_markers, thresholds,
                             output_dir / 'gate_histograms.png')

    if args.label_by_gate and thresholds:
        plot_gate_histograms(features_raw_all, char_keys, char_names,
                             char_names, thresholds,
                             output_dir / 'gate_histograms.png')

    if kdist_data:
        plot_kdistance(kdist_data, args.structure_eps,
                       output_dir / 'kdistance_elbow.png')

    # ── Napari ────────────────────────────────────────────────────
    if args.visualize:
        try:
            import napari
            viewer = napari.Viewer(title='Tissue Zones')
            cmap = get_zone_cmap(n_zones)
            colors = [cmap(labels[i] / max(n_zones - 1, 1)) for i in range(len(labels))]
            # Napari points: [y, x] convention
            pts = positions[:, ::-1]
            viewer.add_points(pts, face_color=colors, size=5, name='cells')
            napari.run()
        except ImportError:
            print("  napari not available, skipping interactive viewer")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\nDone. Outputs in {output_dir}/")
    print(f"  detections_zoned.json     — {len(detections)} cells with zone_id + zone_label")
    print(f"  zone_metadata.json        — {n_zones} zone summaries")
    print(f"  zone_map.png              — cells colored by zone")
    print(f"  zone_marker_profiles.png  — marker expression per zone")
    print(f"  zone_map_with_density.png — zones over density heatmap")
    if elbow_info is not None:
        print(f"  dendrogram_elbow.png      — auto k-selection (elbow method)")


if __name__ == '__main__':
    main()
