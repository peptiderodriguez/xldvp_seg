"""Utility functions for feature-based clustering of cell detections.

Provides channel discovery, feature selection, feature matrix extraction,
marker normalization, cluster auto-labeling, and spatial smoothing.
These are used by ``scripts/cluster_by_features.py`` and are importable
for programmatic use.

Feature groups:
  - "morph":   all morphological features (= shape + color, backward compat)
  - "shape":   pure geometry (area, circularity, solidity, aspect_ratio, etc.)
  - "color":   intensity/color (gray_mean, hue_mean, relative_brightness, etc.)
  - "sam2":    SAM2 embedding features (sam2_0..sam2_255)
  - "channel": per-channel stats (ch0_mean, ch1_std, ch0_ch2_ratio, etc.)
  - "deep":    deep features (resnet_*, dinov2_*)
"""

import math
import re
import types
from pathlib import Path

import numpy as np

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load, sanitize_for_json
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default islet marker mapping (backward compatibility)
_ISLET_MARKER_DEFAULTS = {
    "alpha": 2,  # Gcg
    "beta": 3,  # Ins
    "delta": 5,  # Sst
}

_COLOR_FEATURES = frozenset(
    {
        "red_mean",
        "red_std",
        "green_mean",
        "green_std",
        "blue_mean",
        "blue_std",
        "gray_mean",
        "gray_std",
        "hue_mean",
        "saturation_mean",
        "value_mean",
        "relative_brightness",
        "intensity_variance",
        "dark_fraction",
    }
)

_SHAPE_FEATURES = frozenset(
    {
        "area",
        "area_um2",
        "perimeter",
        "circularity",
        "solidity",
        "aspect_ratio",
        "extent",
        "equiv_diameter",
        "nuclear_complexity",
    }
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_marker_channels(marker_str):
    """Parse --marker-channels string into {name: channel_index} dict.

    Format: "name1:ch_idx1,name2:ch_idx2" e.g. "msln:2,pm:1" or "alpha:2,beta:3,delta:5"
    Returns: {'msln': 2, 'pm': 1}
    """
    if not marker_str:
        return None
    result = {}
    for pair in marker_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            logger.warning(
                "Ignoring malformed marker-channel pair '%s' (expected name:index)", pair
            )
            continue
        name, idx_str = pair.split(":", 1)
        name = name.strip()
        try:
            idx = int(idx_str.strip())
        except ValueError:
            logger.warning("Ignoring non-integer channel index in '%s'", pair)
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
    for part in exclude_str.split(","):
        part = part.strip()
        try:
            result.add(int(part))
        except ValueError:
            logger.warning("Ignoring non-integer channel index '%s' in --exclude-channels", part)
    return result


# ---------------------------------------------------------------------------
# Channel discovery
# ---------------------------------------------------------------------------


def discover_channels_from_features(detections):
    """Auto-discover channel indices present in detection features.

    Looks for keys matching ch(\\d+)_* pattern and returns sorted unique indices.

    Returns: sorted list of channel indices, e.g. [0, 1, 2, 3, 5]
    """
    ch_indices = set()
    ch_pattern = re.compile(r"^ch(\d+)_")
    for det in detections:
        feats = det.get("features", {})
        if not feats:
            continue
        for key in feats:
            m = ch_pattern.match(key)
            if m:
                ch_indices.add(int(m.group(1)))
        if ch_indices:
            break
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
    return {f"ch{ch}": ch for ch in available}


# ---------------------------------------------------------------------------
# Feature classification & selection
# ---------------------------------------------------------------------------


def classify_feature_group(key):
    """Classify a feature key into its group.

    Returns one of: 'shape', 'color', 'morph' (=shape+color), 'sam2',
    'channel', 'deep', or None if unrecognized.

    'morph' is a virtual group that matches both 'shape' and 'color' features,
    preserving backward compatibility with --feature-groups morph,sam2,channel.
    """
    if re.match(r"^ch\d+", key):
        return "channel"
    if key.startswith("sam2_"):
        return "sam2"
    if key.startswith("resnet_") or key.startswith("dinov2_"):
        return "deep"
    if key in _SHAPE_FEATURES:
        return "shape"
    if key in _COLOR_FEATURES:
        return "color"
    # Unknown non-prefixed keys default to shape
    if isinstance(key, str) and not key.startswith(("ch", "sam2", "resnet", "dinov2")):
        return "shape"
    return None


def get_channel_index_from_key(key):
    """Extract channel index from a channel feature key like 'ch2_mean' or 'ch0_ch2_ratio'.

    Returns set of channel indices referenced by this key.
    """
    return {int(m) for m in re.findall(r"ch(\d+)", key)}


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
    if "morph" in expanded:
        expanded.discard("morph")
        expanded.add("shape")
        expanded.add("color")

    # Collect all numeric feature names from first detection with features
    all_names = set()
    for det in detections:
        feats = det.get("features", {})
        if not feats:
            continue
        for k, v in feats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                all_names.add(k)
        if all_names:
            break

    # Filter by group and exclusions
    selected = []
    for name in sorted(all_names):
        group = classify_feature_group(name)
        if group is None or group not in expanded:
            continue

        # For channel features, check if any referenced channel is excluded
        if group == "channel" and exclude_channels:
            referenced = get_channel_index_from_key(name)
            if referenced & exclude_channels:
                continue

        selected.append(name)

    return selected


# ---------------------------------------------------------------------------
# Detection loading
# ---------------------------------------------------------------------------


def load_detections_filtered(detections_path, threshold=0.5):
    """Load RF-positive detections above threshold.

    Includes ALL detections that lack a score (no classifier applied).
    This differs from :func:`xldvp_seg.utils.detection_utils.load_detections`
    which defaults unscored detections to 0.0.
    """
    all_dets = fast_json_load(str(detections_path))

    positive = []
    for det in all_dets:
        score = det.get("rf_prediction", det.get("score"))
        if score is not None and score >= threshold:
            positive.append(det)
        elif score is None:
            # No classifier -- include all
            positive.append(det)

    return positive


# ---------------------------------------------------------------------------
# Feature matrix extraction & normalization
# ---------------------------------------------------------------------------


def extract_feature_matrix(detections, feature_names):
    """Extract feature matrix from detections using explicit feature name list.

    Strict mode: skips detections with any missing/non-finite feature value.
    Returns a 3-tuple (X, feature_names, valid_indices).

    Note: differs from ``xldvp_seg.utils.detection_utils.extract_feature_matrix``
    which returns a 2-tuple (X, valid_indices), uses float32, and fills missing
    values with 0 instead of skipping the row.

    Returns:
        X: numpy array (n_cells, n_features) or None if empty
        feature_names: list of feature name strings (same as input, for convenience)
        valid_indices: indices of detections with complete features
    """
    if not feature_names:
        return None, [], []

    # Build matrix — strict: skip rows with any missing/non-finite value
    rows = []
    valid_indices = []
    for i, det in enumerate(detections):
        feats = det.get("features", {})
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
        logger.info("  %s: range [%.1f, %.1f] -> [0, 1]", feature_names[j], lo, hi)
    return X_norm, norm_ranges


# ---------------------------------------------------------------------------
# Cluster labeling
# ---------------------------------------------------------------------------


def auto_label_clusters(detections, labels, valid_indices, marker_channels, norm_ranges=None):
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
    marker_keys = {name: f"ch{idx}_mean" for name, idx in marker_channels.items()}

    # Compute population-level stats per marker for z-score normalization
    # (used when norm_ranges is not provided)
    pop_stats = {}  # {feature_key: (mean, std)}
    if not norm_ranges and marker_keys:
        all_valid_dets = [detections[vi] for vi in valid_indices]
        for label, key in marker_keys.items():
            all_vals = np.array(
                [d.get("features", {}).get(key, 0) for d in all_valid_dets], dtype=np.float64
            )
            pop_stats[key] = (np.mean(all_vals), max(np.std(all_vals), 1e-12))

    for cl in sorted(unique_labels):
        mask = labels == cl
        cluster_dets = [detections[valid_indices[i]] for i, m in enumerate(mask) if m]

        marker_means = {}
        for label, key in marker_keys.items():
            vals = [d.get("features", {}).get(key, 0) for d in cluster_dets]
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

        # Label as "C{id}" with marker annotation if one is enriched
        enriched = None
        if marker_means and max(marker_means.values()) > 0:
            best = max(marker_means, key=marker_means.get)
            threshold = 0.1 if norm_ranges else 0.5
            if marker_means[best] >= threshold:
                enriched = best

        if enriched:
            cluster_labels[cl] = f"C{cl} ({enriched}-high)"
        else:
            cluster_labels[cl] = f"C{cl}"

    cluster_labels[-1] = "noise"
    return cluster_labels


def get_marker_mean_keys(marker_channels):
    """Get the chN_mean feature keys for each marker channel.

    Returns: list of (marker_name, feature_key) tuples, sorted by channel index.
    """
    return sorted(
        [(name, f"ch{idx}_mean") for name, idx in marker_channels.items()],
        key=lambda x: marker_channels[x[0]],
    )


# ---------------------------------------------------------------------------
# Spatial smoothing
# ---------------------------------------------------------------------------


def spatial_smooth_features(X, positions_um, k=15, sim_threshold=0.5, n_pca=50):
    """Feature-gated spatial smoothing.

    For each cell, replaces its features with a weighted average of its
    spatial neighbors' features, where weight = cosine_similarity(cell, neighbor)
    in PCA space. Only neighbors above sim_threshold contribute.

    This prevents smoothing across tissue boundaries (e.g., hepatocyte next to
    endothelial cell) and preserves rare cell types that differ from their
    spatial neighbors in feature space.

    Args:
        X: Feature matrix (n_cells, n_features), already scaled.
        positions_um: Cell positions in microns (n_cells, 2).
        k: Number of spatial nearest neighbors.
        sim_threshold: Minimum cosine similarity to include neighbor (0-1).
        n_pca: PCA dimensions for similarity computation.

    Returns:
        X_smooth: Smoothed feature matrix (same shape as X).
    """
    from scipy.spatial import KDTree
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    n_cells = X.shape[0]
    if n_cells < k + 1:
        return X.copy()

    # PCA for similarity computation (faster than full feature space)
    n_components = min(n_pca, X.shape[1], n_cells - 1)
    X_pca = PCA(n_components=n_components).fit_transform(X)
    # L2-normalize for cosine similarity via dot product
    X_pca_norm = normalize(X_pca, norm="l2")

    # Build spatial KD-tree
    tree = KDTree(positions_um)
    distances, indices = tree.query(positions_um, k=k + 1)  # +1 includes self

    X_smooth = np.zeros_like(X, dtype=np.float64)
    n_smoothed = 0

    for i in range(n_cells):
        neighbor_idx = indices[i, 1:]  # exclude self

        # Cosine similarity in PCA space
        sims = X_pca_norm[neighbor_idx] @ X_pca_norm[i]

        # Gate: only neighbors above threshold
        mask = sims >= sim_threshold
        if mask.sum() == 0:
            # No similar neighbors -- keep original features
            X_smooth[i] = X[i]
            continue

        # Weighted average: weight = cosine similarity
        weights = sims[mask]
        weights = weights / weights.sum()  # normalize

        X_smooth[i] = weights @ X[neighbor_idx[mask]]
        n_smoothed += 1

    logger.info(
        "[spatial-smooth] %d/%d cells smoothed (k=%d, sim_threshold=%.2f)",
        n_smoothed,
        n_cells,
        k,
        sim_threshold,
    )

    return X_smooth


# ---------------------------------------------------------------------------
# run_clustering & run_subclustering
# ---------------------------------------------------------------------------


def _run_clustering_impl(args):
    """Main clustering pipeline (internal, takes namespace-like object)."""
    import os

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed. Run: pip install umap-learn")
        raise

    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed. Run: pip install hdbscan")
        raise

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse channel configuration
    marker_channels = parse_marker_channels(args.marker_channels)
    exclude_channels = parse_exclude_channels(args.exclude_channels)
    feature_groups = {g.strip() for g in args.feature_groups.split(",")}
    valid_groups = {"morph", "shape", "color", "sam2", "channel", "deep"}
    invalid_groups = feature_groups - valid_groups
    if invalid_groups:
        logger.warning("Unknown feature groups ignored: %s", invalid_groups)
        feature_groups &= valid_groups

    # Load detections
    logger.info("Loading detections from %s...", args.detections)
    detections = load_detections_filtered(args.detections, threshold=args.threshold)
    logger.info("  %d RF-positive detections (threshold >= %s)", len(detections), args.threshold)

    # Expression gating — filter to cells above a percentile of a channel mean
    if args.gate_channel is not None:
        gate_key = f"ch{args.gate_channel}_mean"
        gate_values = []
        for d in detections:
            v = d.get("features", {}).get(gate_key)
            if v is not None:
                gate_values.append(float(v))
        if not gate_values:
            raise ValueError(f"No detections have feature '{gate_key}' -- cannot gate")
        gate_cutoff = float(np.percentile(gate_values, args.gate_percentile))
        before = len(detections)
        detections = [
            d for d in detections if float(d.get("features", {}).get(gate_key, 0)) >= gate_cutoff
        ]
        logger.info(
            "  Gated on %s >= p%.0f (%.1f): %d -> %d detections (top %.0f%%)",
            gate_key,
            args.gate_percentile,
            gate_cutoff,
            before,
            len(detections),
            100 - args.gate_percentile,
        )

    if len(detections) < args.min_cluster_size:
        raise ValueError(
            f"Not enough detections ({len(detections)}) for clustering "
            f"(need at least {args.min_cluster_size})"
        )

    # Discover channels
    all_channels = discover_channels_from_features(detections)
    logger.info("  Channels found in features: %s", all_channels)
    if exclude_channels:
        logger.info("  Excluding channels: %s", sorted(exclude_channels))

    # Resolve marker channels
    if marker_channels is None:
        marker_channels = discover_marker_channels(detections, exclude_channels)
        if marker_channels is None:
            logger.warning(
                "No marker channels found in features. "
                "Cluster labeling will use 'other' for all clusters."
            )
            marker_channels = {}
        else:
            # Check if we got islet defaults
            if marker_channels == _ISLET_MARKER_DEFAULTS:
                logger.info(
                    "  Auto-detected islet marker channels: "
                    "alpha=ch2 (Gcg), beta=ch3 (Ins), delta=ch5 (Sst)"
                )
            else:
                logger.info("  Auto-detected marker channels: %s", marker_channels)
    else:
        # User-specified -- validate channels exist
        for name, idx in list(marker_channels.items()):
            if idx not in all_channels:
                logger.warning(
                    "Marker '%s' references ch%d which has no "
                    "features in detections (available: %s)",
                    name,
                    idx,
                    all_channels,
                )
            if idx in exclude_channels:
                logger.warning(
                    "Marker '%s' references ch%d which is in "
                    "--exclude-channels. Removing from markers.",
                    name,
                    idx,
                )
                del marker_channels[name]
        logger.info("  Marker channels: %s", marker_channels)

    # Extract features
    logger.info("Extracting feature matrix...")
    marker_mean_keys = [f"ch{idx}_mean" for idx in marker_channels.values()]

    if args.marker_only:
        if not marker_channels:
            raise ValueError("marker_only requires marker channels but none found")
        feature_names = sorted(marker_mean_keys)
        logger.info("  Using normalized marker channels only: %s", feature_names)
        X, feature_names, valid_indices = extract_feature_matrix(detections, feature_names)
    else:
        feature_names = select_feature_names(detections, feature_groups, exclude_channels)
        logger.info(
            "  Selected %d features from groups %s", len(feature_names), sorted(feature_groups)
        )
        # Print breakdown by group
        group_counts = {}
        for fn in feature_names:
            g = classify_feature_group(fn)
            group_counts[g] = group_counts.get(g, 0) + 1
        for g in sorted(group_counts):
            logger.info("    %s: %d features", g, group_counts[g])

        X, feature_names, valid_indices = extract_feature_matrix(detections, feature_names)

    if X is None or len(X) == 0:
        raise ValueError("No valid features found in detections")

    logger.info("  Feature matrix: %d cells x %d features", X.shape[0], X.shape[1])

    # Normalize and scale
    norm_ranges = None
    if args.marker_only:
        # Population percentile normalization (matches HTML display)
        logger.info("  Applying p1-p99.5 percentile normalization...")
        X_scaled, norm_ranges = normalize_marker_features(X, feature_names)
        # No StandardScaler needed -- already [0, 1] and all same units
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    # Replace NaN/inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature-gated spatial smoothing (opt-in)
    if args.spatial_smooth:
        from xldvp_seg.utils.detection_utils import extract_positions_um

        # Extract positions only for valid detections (aligned with X_scaled rows)
        valid_dets_for_positions = [detections[i] for i in valid_indices]
        positions, _px_size = extract_positions_um(valid_dets_for_positions)
        if positions is not None and len(positions) == X_scaled.shape[0]:
            X_original = X_scaled.copy()  # noqa: F841 — keep unsmoothed for comparison
            X_scaled = spatial_smooth_features(
                X_scaled,
                positions,
                k=args.smooth_k,
                sim_threshold=args.smooth_sim_threshold,
            )
        else:
            n_pos = len(positions) if positions is not None else 0
            logger.warning(
                "[spatial-smooth] Could not extract positions for all cells "
                "(%d/%d), skipping smoothing",
                n_pos,
                X_scaled.shape[0],
            )

    # Optional PCA pre-reduction for large feature sets
    if X_scaled.shape[1] > 50:
        from sklearn.decomposition import PCA

        pca_target = 0.95
        logger.info(
            "  PCA pre-reduction: %d dims -> %.0f%% variance...",
            X_scaled.shape[1],
            pca_target * 100,
        )
        pca = PCA(n_components=pca_target, random_state=42)
        X_umap = pca.fit_transform(X_scaled)
        total_var = pca.explained_variance_ratio_.sum()
        if total_var < pca_target:
            logger.warning(
                "  All %d components kept but only %.1f%% variance explained (< %.0f%% target)",
                X_umap.shape[1],
                total_var * 100,
                pca_target * 100,
            )
        else:
            logger.info(
                "  PCA: %d -> %d dims (%.1f%% variance)",
                X_scaled.shape[1],
                X_umap.shape[1],
                total_var * 100,
            )
    else:
        X_umap = X_scaled

    # Dimensionality reduction (UMAP and/or t-SNE)
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    umap_embedding = None
    tsne_embedding = None

    if args.methods in ("umap", "both"):
        logger.info(
            "Running UMAP (n_neighbors=%d, min_dist=%s, n_jobs=%d)...",
            args.n_neighbors,
            args.min_dist,
            n_jobs,
        )
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            n_components=2,
            random_state=42,
            n_jobs=n_jobs,
            low_memory=True,
        )
        umap_embedding = reducer.fit_transform(X_umap)
        logger.info("  UMAP embedding: %s", umap_embedding.shape)

    if args.methods in ("tsne", "both"):
        from sklearn.manifold import TSNE

        logger.info(
            "Running t-SNE (perplexity=%d, max_iter=%d, n_jobs=%d)...",
            args.perplexity,
            args.tsne_n_iter,
            n_jobs,
        )
        tsne = TSNE(
            perplexity=args.perplexity,
            max_iter=args.tsne_n_iter,
            random_state=42,
            n_jobs=n_jobs,
        )
        tsne_embedding = tsne.fit_transform(X_umap)
        logger.info("  t-SNE embedding: %s", tsne_embedding.shape)

    # For HDBSCAN, prefer UMAP embedding (better for density-based clustering)
    # Fall back to t-SNE if UMAP wasn't run
    cluster_embedding = umap_embedding if umap_embedding is not None else tsne_embedding

    # Clustering
    if args.clustering == "leiden":
        logger.info("Running Leiden clustering (resolution=%s)...", args.resolution)
        import anndata as ad
        import scanpy as sc

        adata_tmp = ad.AnnData(X=cluster_embedding)
        sc.pp.neighbors(adata_tmp, n_neighbors=args.n_neighbors, use_rep="X")
        sc.tl.leiden(adata_tmp, resolution=args.resolution, random_state=42)
        labels = adata_tmp.obs["leiden"].astype(int).values
        n_clusters = len(set(labels))
        n_noise = 0
        logger.info("  Found %d clusters (Leiden resolution=%s)", n_clusters, args.resolution)
        del adata_tmp
    else:
        logger.info("Running HDBSCAN (min_cluster_size=%d)...", args.min_cluster_size)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
        labels = clusterer.fit_predict(cluster_embedding)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info("  Found %d clusters, %d noise points", n_clusters, n_noise)

    # Auto-label clusters
    cluster_label_map = auto_label_clusters(
        detections, labels, valid_indices, marker_channels, norm_ranges=norm_ranges
    )
    logger.info("  Cluster labels: %s", cluster_label_map)

    # Enrich detections with cluster info
    valid_set = set(valid_indices)
    for i, idx in enumerate(valid_indices):
        det = detections[idx]
        cl = int(labels[i])
        det["cluster_id"] = cl
        det["cluster_label"] = cluster_label_map.get(cl, "other")
        if umap_embedding is not None:
            det["umap_x"] = float(umap_embedding[i, 0])
            det["umap_y"] = float(umap_embedding[i, 1])
        if tsne_embedding is not None:
            det["tsne_x"] = float(tsne_embedding[i, 0])
            det["tsne_y"] = float(tsne_embedding[i, 1])

    # Add sentinel values for non-clustered detections (missing features)
    for i, det in enumerate(detections):
        if i not in valid_set and "cluster_id" not in det:
            det["cluster_id"] = None
            det["cluster_label"] = "unclassified"
            if umap_embedding is not None:
                det["umap_x"] = None
                det["umap_y"] = None
            if tsne_embedding is not None:
                det["tsne_x"] = None
                det["tsne_y"] = None

    # Save enriched detections
    clustered_path = output_dir / "detections_clustered.json"
    atomic_json_dump(sanitize_for_json(detections), str(clustered_path))
    logger.info("  Saved: %s", clustered_path)

    # Build summary DataFrame
    marker_mean_pairs = get_marker_mean_keys(marker_channels)  # [(name, key), ...]
    rows = []
    for i, idx in enumerate(valid_indices):
        det = detections[idx]
        gc = det.get("global_center", [0, 0])
        row = {
            "uid": det.get("uid", det.get("id", f"cell_{i}")),
            "x": gc[0] if isinstance(gc, (list, tuple)) else 0,
            "y": gc[1] if isinstance(gc, (list, tuple)) else 0,
            "cluster_id": det.get("cluster_id", -1),
            "cluster_label": det.get("cluster_label", "other"),
        }
        if umap_embedding is not None:
            row["umap_x"] = det.get("umap_x", 0)
            row["umap_y"] = det.get("umap_y", 0)
        if tsne_embedding is not None:
            row["tsne_x"] = det.get("tsne_x", 0)
            row["tsne_y"] = det.get("tsne_y", 0)
        # Add marker intensities
        feats = det.get("features", {})
        for marker_name, key in marker_mean_pairs:
            row[key] = feats.get(key, 0)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Cluster summary -- dynamically build aggregation for available markers
    agg_dict = {"n_cells": ("uid", "count")}
    for marker_name, key in marker_mean_pairs:
        col_label = f"{key}_{marker_name}"
        if key in df.columns:
            agg_dict[col_label] = (key, "mean")
    agg_dict["x_mean"] = ("x", "mean")
    agg_dict["y_mean"] = ("y", "mean")
    agg_dict["x_std"] = ("x", "std")
    agg_dict["y_std"] = ("y", "std")

    summary = df.groupby("cluster_label").agg(**agg_dict).round(2)

    summary_path = output_dir / "cluster_summary.csv"
    summary.to_csv(summary_path)
    logger.info("  Saved: %s", summary_path)
    logger.info("\n%s", summary.to_string())

    # Spatial CSV
    csv_path = output_dir / "spatial.csv"
    df.to_csv(csv_path, index=False)
    logger.info("  Saved: %s", csv_path)

    # AnnData export
    try:
        import anndata

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        obs_df = df.copy()
        if obs_df["uid"].duplicated().any():
            obs_df["uid"] = obs_df["uid"] + "_" + obs_df.groupby("uid").cumcount().astype(str)
        adata = anndata.AnnData(
            X=X_clean,
            obs=obs_df.set_index("uid"),
        )
        adata.var_names = feature_names
        if umap_embedding is not None:
            adata.obsm["X_umap"] = umap_embedding
        if tsne_embedding is not None:
            adata.obsm["X_tsne"] = tsne_embedding
        h5ad_path = output_dir / "spatial.h5ad"
        adata.write(h5ad_path)
        logger.info("  Saved: %s", h5ad_path)
    except ImportError:
        logger.info("  Skipping .h5ad export (anndata not installed)")

    # Plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.spatial import ConvexHull

        # Build dynamic color map for cluster labels
        # Use a colormap that supports many labels
        all_label_names = sorted(df["cluster_label"].unique())
        # Fixed colors for known labels
        fixed_colors = {
            "alpha": "red",
            "beta": "green",
            "delta": "blue",
            "other": "gray",
            "noise": "lightgray",
            "unclassified": "lightgray",
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

        # --- Marker ring colors ---
        # Fixed warm palette distinct from the cluster tab10/tab20 colors.
        # Assigned in order of marker_mean_pairs (sorted by channel index).
        _MARKER_RING_PALETTE = [
            "#e6194b",  # red
            "#f58231",  # orange
            "#ffe119",  # yellow
            "#f032e6",  # magenta
            "#42d4f4",  # cyan
            "#bfef45",  # lime
            "#dcbeff",  # lavender
            "#fabed4",  # pink
        ]
        # Build {marker_name: ring_color}
        marker_ring_palette = {}
        use_marker_rings = getattr(args, "marker_rings", True) and bool(marker_mean_pairs)
        if use_marker_rings:
            for i, (mname, _key) in enumerate(marker_mean_pairs):
                marker_ring_palette[mname] = _MARKER_RING_PALETTE[i % len(_MARKER_RING_PALETTE)]

            # Compute per-cell dominant marker via z-score (threshold > 1.0 SD).
            # z-score is computed across all cells in the plot.
            marker_cols = [key for _name, key in marker_mean_pairs if key in df.columns]
            if marker_cols:
                marker_vals = df[marker_cols].values.astype(float)
                col_mean = marker_vals.mean(axis=0)
                col_std = marker_vals.std(axis=0)
                col_std[col_std == 0] = 1.0  # avoid divide-by-zero for constant channels
                zscores = (marker_vals - col_mean) / col_std  # shape: (n_cells, n_markers)

                # For each cell: find the marker with the highest z-score above 1.0
                dominant_ring = []
                for row_z in zscores:
                    best_i = int(np.argmax(row_z))
                    if row_z[best_i] > 1.0:
                        col_key = marker_cols[best_i]
                        # Resolve col_key back to marker name
                        mname = next((n for n, k in marker_mean_pairs if k == col_key), None)
                        dominant_ring.append(marker_ring_palette.get(mname) if mname else None)
                    else:
                        dominant_ring.append(None)
                df["_marker_ring_color"] = dominant_ring
            else:
                use_marker_rings = False

        def _plot_embedding(
            ax,
            emb_x,
            emb_y,
            df,
            color_map,
            all_label_names,
            title,
            xlabel,
            ylabel,
            marker_ring_palette=None,
        ):
            """Plot a single embedding with dashed convex hull outlines per cluster.

            If marker_ring_palette is provided (non-empty dict) and df contains a
            '_marker_ring_color' column, a second scatter pass draws hollow rings
            around dots colored by the cell's dominant marker channel.
            """
            for label_name in all_label_names:
                mask = df["cluster_label"] == label_name
                color = color_map.get(label_name, "gray")
                ax.scatter(
                    emb_x[mask],
                    emb_y[mask],
                    c=[color],
                    label=label_name,
                    s=1,
                    alpha=0.3,
                )
                # Draw cluster outline (convex hull, dashed, no fill)
                if label_name not in ("noise", "unclassified", "other") and mask.sum() >= 3:
                    pts = np.column_stack([emb_x[mask].values, emb_y[mask].values])
                    try:
                        hull = ConvexHull(pts)
                        hull_pts = pts[hull.vertices]
                        hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon
                        ax.plot(
                            hull_pts[:, 0],
                            hull_pts[:, 1],
                            color=color,
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.8,
                        )
                    except Exception:
                        pass  # Degenerate hull (collinear points)

            # Marker rings: second scatter pass, one per marker color
            ring_handles = []
            if marker_ring_palette and "_marker_ring_color" in df.columns:
                for mname, ring_color in marker_ring_palette.items():
                    ring_mask = df["_marker_ring_color"] == ring_color
                    if not ring_mask.any():
                        continue
                    ax.scatter(
                        emb_x[ring_mask],
                        emb_y[ring_mask],
                        facecolors="none",
                        edgecolors=ring_color,
                        linewidths=0.5,
                        s=18,
                        alpha=0.85,
                        zorder=3,
                    )
                    # Proxy artist for legend entry
                    import matplotlib.lines as mlines

                    handle = mlines.Line2D(
                        [],
                        [],
                        marker="o",
                        color="none",
                        markerfacecolor="none",
                        markeredgecolor=ring_color,
                        markeredgewidth=0.8,
                        markersize=6,
                        label=mname,
                    )
                    ring_handles.append(handle)

            # Cluster legend (existing)
            cluster_legend = ax.legend(markerscale=4, fontsize=8, loc="upper left")

            # Marker ring legend (second legend, lower left)
            if ring_handles:
                ax.add_artist(cluster_legend)  # keep cluster legend visible
                ax.legend(
                    handles=ring_handles,
                    title="Dominant marker",
                    fontsize=7,
                    title_fontsize=7,
                    loc="lower left",
                    framealpha=0.7,
                )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

        main_title = f"Cell Clustering ({len(df)} cells, {n_clusters} clusters)"
        _ring_palette_arg = marker_ring_palette if use_marker_rings else {}

        if umap_embedding is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            _plot_embedding(
                ax,
                df["umap_x"],
                df["umap_y"],
                df,
                color_map,
                all_label_names,
                main_title,
                "UMAP 1",
                "UMAP 2",
                marker_ring_palette=_ring_palette_arg,
            )
            umap_path = output_dir / "umap_plot.png"
            fig.savefig(umap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved: %s", umap_path)

        if tsne_embedding is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            _plot_embedding(
                ax,
                df["tsne_x"],
                df["tsne_y"],
                df,
                color_map,
                all_label_names,
                main_title,
                "t-SNE 1",
                "t-SNE 2",
                marker_ring_palette=_ring_palette_arg,
            )
            tsne_path = output_dir / "tsne_plot.png"
            fig.savefig(tsne_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved: %s", tsne_path)

        if umap_embedding is not None and tsne_embedding is not None:
            fig, (ax_u, ax_t) = plt.subplots(1, 2, figsize=(20, 8))
            _plot_embedding(
                ax_u,
                df["umap_x"],
                df["umap_y"],
                df,
                color_map,
                all_label_names,
                "UMAP",
                "UMAP 1",
                "UMAP 2",
                marker_ring_palette=_ring_palette_arg,
            )
            _plot_embedding(
                ax_t,
                df["tsne_x"],
                df["tsne_y"],
                df,
                color_map,
                all_label_names,
                "t-SNE",
                "t-SNE 1",
                "t-SNE 2",
                marker_ring_palette=_ring_palette_arg,
            )
            fig.suptitle(main_title)
            fig.tight_layout()
            combined_path = output_dir / "umap_tsne_plot.png"
            fig.savefig(combined_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved: %s", combined_path)

        # Violin plot of marker intensities per cluster
        if marker_mean_pairs:
            n_markers = len(marker_mean_pairs)
            fig, axes = plt.subplots(1, n_markers, figsize=(5 * n_markers, 5))
            if n_markers == 1:
                axes = [axes]

            for ax, (marker_name, col) in zip(axes, marker_mean_pairs):
                if col not in df.columns:
                    ax.set_title(f"{marker_name} (ch{marker_channels[marker_name]}) - no data")
                    continue
                cluster_labels_sorted = sorted(df["cluster_label"].unique())
                data = [
                    df.loc[df["cluster_label"] == cl, col].values for cl in cluster_labels_sorted
                ]
                # Filter out clusters with <2 points (violinplot KDE requires >=2)
                valid_mask = [len(d) >= 2 for d in data]
                data_filtered = [d for d, v in zip(data, valid_mask) if v]
                tick_labels = [cl for cl, v in zip(cluster_labels_sorted, valid_mask) if v]
                if data_filtered:
                    parts = ax.violinplot(
                        data_filtered, showmeans=True, showmedians=True
                    )  # noqa: F841
                    ax.set_xticks(range(1, len(tick_labels) + 1))
                    ax.set_xticklabels(tick_labels, rotation=45)
                ax.set_title(f"{marker_name} (ch{marker_channels[marker_name]})")
                ax.set_ylabel("Intensity")

            fig.suptitle("Marker Expression by Cluster")
            fig.tight_layout()
            violin_path = output_dir / "marker_violin.png"
            fig.savefig(violin_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved: %s", violin_path)
        else:
            logger.info("  Skipping violin plot (no marker channels)")

    except ImportError:
        logger.info("  Skipping plots (matplotlib not installed)")

    # Interactive HTML (plotly) — white background, click legend to highlight clusters
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        _PLOTLY_COLORS = [
            "#e6194b",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#bfef45",
            "#fabebe",
            "#469990",
            "#e6beff",
            "#9a6324",
            "#ffe119",
            "#aaffc3",
            "#800000",
            "#ffd8b1",
            "#000075",
            "#a9a9a9",
            "#808000",
            "#ff69b4",
            "#dcbeff",
            "#00ffff",
            "#ff6347",
            "#7cfc00",
        ]

        def _build_interactive(df, x_col, y_col, title, xlabel, ylabel, out_path, detections=None):
            """Build plotly interactive with white bg + click-to-highlight clusters.

            Also adds marker profile traces (single+, double+) if detections have
            marker_profile or *_class fields.
            """
            fig = go.Figure()
            # Background: all points as white dots (shows UMAP structure)
            fig.add_trace(
                go.Scattergl(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    marker=dict(size=2, color="rgba(255,255,255,0.15)"),
                    name="all cells",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Per-cluster colored traces (start hidden, click legend to show)
            labels_list = sorted(df["cluster_label"].unique())
            has_uid = "uid" in df.columns
            for i, label in enumerate(labels_list):
                if label == "noise":
                    continue
                mask = df["cluster_label"] == label
                n = mask.sum()
                fig.add_trace(
                    go.Scattergl(
                        x=df.loc[mask, x_col],
                        y=df.loc[mask, y_col],
                        mode="markers",
                        marker=dict(
                            size=2, color=_PLOTLY_COLORS[i % len(_PLOTLY_COLORS)], opacity=0.3
                        ),
                        name=f"{label} ({n})",
                        text=df.loc[mask, "uid"] if has_uid else None,
                        hovertemplate=(
                            "%{text}<extra>" + str(label) + "</extra>" if has_uid else None
                        ),
                        visible="legendonly",
                    )
                )

            # Marker profile traces (double+, single+)
            if detections is not None:
                # Find marker_profile or *_class fields
                sample_feat = detections[0].get("features", {}) if detections else {}
                class_keys = [k for k in sample_feat if k.endswith("_class")]
                marker_names_list = [k.replace("_class", "") for k in class_keys]

                if marker_names_list and len(df) == len(detections):
                    # Build marker profile column
                    profiles = []
                    for det in detections:
                        feat = det.get("features", {})
                        parts = []
                        for m in marker_names_list:
                            cls = feat.get(f"{m}_class", "negative")
                            parts.append(f"{m}+" if cls == "positive" else f"{m}-")
                        profiles.append("/".join(parts))
                    df["_marker_profile"] = profiles

                    # Add traces for positive profiles
                    profile_counts = df["_marker_profile"].value_counts()
                    for profile, count in profile_counts.items():
                        n_pos = profile.count("+")
                        if n_pos == 0:
                            continue  # skip all-negative
                        mask = df["_marker_profile"] == profile
                        color = "#ff0000" if n_pos >= 2 else "#ffaa00"
                        fig.add_trace(
                            go.Scattergl(
                                x=df.loc[mask, x_col],
                                y=df.loc[mask, y_col],
                                mode="markers",
                                marker=dict(
                                    size=4,
                                    color=color,
                                    opacity=0.7,
                                    symbol="diamond" if n_pos >= 2 else "circle",
                                ),
                                name=f"\U0001f4cd {profile} ({count})",
                                text=df.loc[mask, "uid"] if has_uid else None,
                                hovertemplate=(
                                    "%{text}<extra>" + profile + "</extra>" if has_uid else None
                                ),
                                visible="legendonly",
                            )
                        )
            # "Show All" / "Hide All" buttons
            n_traces = len(fig.data)
            fig.update_layout(
                template="plotly_dark",
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                width=1400,
                height=900,
                legend=dict(itemsizing="constant", font=dict(size=9)),
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        x=0.0,
                        y=1.12,
                        buttons=[
                            dict(
                                label="Show All",
                                method="update",
                                args=[{"visible": [True] * n_traces}],
                            ),
                            dict(
                                label="Hide All",
                                method="update",
                                args=[{"visible": [True] + ["legendonly"] * (n_traces - 1)}],
                            ),
                        ],
                    )
                ],
            )
            pio.write_html(fig, str(out_path))
            logger.info("  Saved: %s", out_path)

        # Pass detections for marker profile overlays (only valid_indices subset)
        valid_dets = [detections[i] for i in valid_indices] if valid_indices else None

        if umap_embedding is not None and "umap_x" in df.columns:
            _build_interactive(
                df,
                "umap_x",
                "umap_y",
                f"{main_title} -- click legend to highlight",
                "UMAP 1",
                "UMAP 2",
                output_dir / "umap_interactive.html",
                detections=valid_dets,
            )

        if tsne_embedding is not None and "tsne_x" in df.columns:
            _build_interactive(
                df,
                "tsne_x",
                "tsne_y",
                f"{main_title} (t-SNE) -- click legend to highlight",
                "t-SNE 1",
                "t-SNE 2",
                output_dir / "tsne_interactive.html",
                detections=valid_dets,
            )

    except ImportError:
        logger.info("  Skipping interactive plots (plotly not installed)")

    logger.info("Done! %d clusters found in %d cells.", n_clusters, len(valid_indices))

    # Sub-clustering (optional)
    if getattr(args, "subcluster", False):
        run_subclustering(
            detections,
            output_dir,
            marker_channels,
            exclude_channels,
            subcluster_features=args.subcluster_features,
            subcluster_min_size=args.subcluster_min_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
        # Re-save detections with subcluster fields
        clustered_path = output_dir / "detections_clustered.json"
        atomic_json_dump(sanitize_for_json(detections), str(clustered_path))
        logger.info("  Updated: %s (with subcluster fields)", clustered_path)

    # Trajectory analysis (optional)
    if args.trajectory:
        try:
            import anndata as ad
            import scanpy as sc

            logger.info("--- Trajectory Analysis ---")
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            adata = ad.AnnData(X=X_clean)
            adata.var_names = feature_names
            adata.obs["cluster_label"] = [
                detections[i].get("cluster_label", "unknown") for i in valid_indices
            ]
            adata.obs["cluster_label"] = adata.obs["cluster_label"].astype("category")
            if umap_embedding is not None:
                adata.obsm["X_umap"] = umap_embedding

            logger.info("  Computing neighbors (n_neighbors=%d)...", args.n_neighbors)
            sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep="X")

            logger.info("  Computing diffusion map...")
            sc.tl.diffmap(adata)

            logger.info("  Computing PAGA...")
            sc.tl.paga(adata, groups="cluster_label")
            # Compute PAGA node positions (required before draw_graph with init_pos='paga')
            sc.pl.paga(adata, show=False)

            logger.info("  Computing force-directed layout...")
            sc.tl.draw_graph(adata, init_pos="paga")

            # Diffusion pseudotime
            root_cluster = args.root_cluster
            if root_cluster is None:
                root_cluster = adata.obs["cluster_label"].value_counts().index[0]
            root_mask = adata.obs["cluster_label"] == root_cluster
            has_dpt = False
            if root_mask.any():
                adata.uns["iroot"] = int(np.where(root_mask)[0][0])
                logger.info("  Computing diffusion pseudotime (root: %s)...", root_cluster)
                sc.tl.dpt(adata)
                has_dpt = True

            h5ad_path = output_dir / "trajectory.h5ad"
            adata.write(h5ad_path)
            logger.info("  Saved: %s", h5ad_path)

            # Enrich detections
            for i, idx in enumerate(valid_indices):
                det = detections[idx]
                if has_dpt and "dpt_pseudotime" in adata.obs.columns:
                    det["dpt_pseudotime"] = float(adata.obs["dpt_pseudotime"].iloc[i])
                if "X_diffmap" in adata.obsm:
                    for dc in range(min(3, adata.obsm["X_diffmap"].shape[1])):
                        det[f"diffmap_{dc}"] = float(adata.obsm["X_diffmap"][i, dc])
                if "X_draw_graph_fa" in adata.obsm:
                    det["fa_x"] = float(adata.obsm["X_draw_graph_fa"][i, 0])
                    det["fa_y"] = float(adata.obsm["X_draw_graph_fa"][i, 1])

            atomic_json_dump(
                sanitize_for_json(detections), output_dir / "detections_clustered.json"
            )

            # Plots
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.paga(adata, ax=ax, show=False, fontsize=8)
            ax.set_title("PAGA: Cluster Connectivity")
            fig.savefig(output_dir / "paga.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved: paga.png")

            if "X_draw_graph_fa" in adata.obsm:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.draw_graph(adata, color="cluster_label", ax=ax, show=False, size=1, alpha=0.3)
                fig.savefig(output_dir / "force_directed.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("  Saved: force_directed.png")

            if has_dpt and umap_embedding is not None:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.umap(
                    adata,
                    color="dpt_pseudotime",
                    ax=ax,
                    show=False,
                    size=1,
                    alpha=0.3,
                    color_map="viridis",
                )
                fig.savefig(output_dir / "pseudotime_umap.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("  Saved: pseudotime_umap.png")

            if "X_diffmap" in adata.obsm:
                fig, ax = plt.subplots(figsize=(10, 8))
                sc.pl.diffmap(adata, color="cluster_label", ax=ax, show=False, size=1, alpha=0.3)
                fig.savefig(output_dir / "diffusion_map.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("  Saved: diffusion_map.png")

            # Interactive pseudotime
            if has_dpt and umap_embedding is not None:
                try:
                    import plotly.express as px
                    import plotly.io as pio

                    df["dpt_pseudotime"] = adata.obs["dpt_pseudotime"].values
                    fig_pt = px.scatter(
                        df,
                        x="umap_x",
                        y="umap_y",
                        color="dpt_pseudotime",
                        hover_data=(
                            ["uid", "cluster_label"] if "uid" in df.columns else ["cluster_label"]
                        ),
                        title="Diffusion Pseudotime on UMAP",
                        color_continuous_scale="viridis",
                        opacity=0.4,
                    )
                    fig_pt.update_traces(marker=dict(size=2))
                    fig_pt.update_layout(template="plotly_dark", width=1400, height=900)
                    pio.write_html(fig_pt, str(output_dir / "pseudotime_interactive.html"))
                    logger.info("  Saved: pseudotime_interactive.html")
                except ImportError:
                    pass

            logger.info("  Trajectory analysis complete!")
        except ImportError as e:
            logger.info("  Skipping trajectory analysis (%s)", e)
        except Exception as e:
            logger.warning("  Trajectory analysis failed: %s", e, exc_info=True)


def run_clustering(**kwargs):
    """Feature-based clustering of cell detections.

    Public API wrapper that accepts keyword arguments and delegates to the
    internal implementation. All parameters have defaults matching the CLI
    argparse defaults.

    Args:
        detections: Path to detections JSON file.
        output_dir: Output directory for clustering results.
        feature_groups: Comma-separated feature groups (default: "morph,sam2,channel").
        methods: Dim reduction methods: "umap", "tsne", or "both" (default: "both").
        clustering: Clustering algorithm: "leiden" or "hdbscan" (default: "leiden").
        resolution: Leiden resolution (default: 1.0).
        n_neighbors: UMAP n_neighbors (default: 30).
        min_dist: UMAP min_dist (default: 0.1).
        threshold: Minimum rf_prediction score (default: 0.5).
        min_cluster_size: HDBSCAN min_cluster_size (default: 50).
        marker_channels: Marker channels string "name:idx,..." (default: "").
        exclude_channels: Channel indices to exclude "0,3,5" (default: "").
        marker_rings: Draw marker expression rings on plots (default: True).
        trajectory: Run trajectory analysis (default: False).
        root_cluster: Cluster label for pseudotime root (default: None).
        spatial_smooth: Apply spatial smoothing (default: False).
        smooth_k: Spatial smoothing neighbors (default: 15).
        smooth_sim_threshold: Cosine similarity threshold for smoothing (default: 0.5).
        marker_only: Use only marker channel features (default: False).
        gate_channel: Gate on this channel index before clustering (default: None).
        gate_percentile: Percentile threshold for gating (default: 90).
        perplexity: t-SNE perplexity (default: 30).
        tsne_n_iter: t-SNE iterations (default: 1000).
        min_samples: HDBSCAN min_samples (default: None).
        subcluster: Run sub-clustering after main clustering (default: False).
        subcluster_features: Feature groups for subclustering (default: "shape,sam2").
        subcluster_min_size: HDBSCAN min_cluster_size for sub-clusters (default: 50).
        subcluster_input: Path to pre-clustered detections for standalone subclustering.
    """
    defaults = dict(
        detections=None,
        output_dir=None,
        feature_groups="morph,sam2,channel",
        methods="both",
        clustering="leiden",
        resolution=1.0,
        n_neighbors=30,
        min_dist=0.1,
        threshold=0.5,
        min_cluster_size=50,
        marker_channels="",
        exclude_channels="",
        marker_rings=True,
        trajectory=False,
        root_cluster=None,
        spatial_smooth=False,
        smooth_k=15,
        smooth_sim_threshold=0.5,
        marker_only=False,
        gate_channel=None,
        gate_percentile=90,
        perplexity=30,
        tsne_n_iter=1000,
        min_samples=None,
        subcluster=False,
        subcluster_features="shape,sam2",
        subcluster_min_size=50,
        subcluster_input=None,
    )
    # Merge caller kwargs over defaults
    merged = {**defaults, **kwargs}
    ns = types.SimpleNamespace(**merged)
    _run_clustering_impl(ns)


def run_subclustering(
    detections,
    output_dir,
    marker_channels,
    exclude_channels,
    subcluster_features="morph,sam2",
    subcluster_min_size=50,
    n_neighbors=30,
    min_dist=0.1,
):
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
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed")
        return
    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed")
        return

    output_dir = Path(output_dir)
    subcluster_dir = output_dir / "subclusters"
    subcluster_dir.mkdir(exist_ok=True)

    # Feature selection: appearance only
    sub_groups = {g.strip() for g in subcluster_features.split(",")}
    sub_feature_names = select_feature_names(detections, sub_groups, exclude_channels)
    if not sub_feature_names:
        logger.error("No features found for subclustering")
        return

    group_counts = {}
    for fn in sub_feature_names:
        g = classify_feature_group(fn)
        group_counts[g] = group_counts.get(g, 0) + 1
    logger.info(
        "%s\nSUB-CLUSTERING by appearance (%d features: %s)\n%s",
        "=" * 60,
        len(sub_feature_names),
        ", ".join(f"{g}:{n}" for g, n in sorted(group_counts.items())),
        "=" * 60,
    )

    # Group detections by parent cluster label
    parent_groups = {}  # {label: [det_index, ...]}
    for i, det in enumerate(detections):
        label = det.get("cluster_label")
        if label and label not in ("noise", "unclassified"):
            parent_groups.setdefault(label, []).append(i)

    min_for_sub = max(subcluster_min_size * 3, 100)
    master_rows = []

    for parent_label in sorted(parent_groups.keys()):
        det_indices = parent_groups[parent_label]
        if len(det_indices) < min_for_sub:
            logger.info(
                "  '%s': %d cells -- skipping (need >= %d)",
                parent_label,
                len(det_indices),
                min_for_sub,
            )
            continue

        logger.info("--- Sub-clustering '%s' (%d cells) ---", parent_label, len(det_indices))

        # Build feature matrix for this subset only
        sub_dets = [detections[di] for di in det_indices]
        X_sub, _, sub_valid_local = extract_feature_matrix(sub_dets, sub_feature_names)

        if X_sub is None or len(X_sub) < min_for_sub:
            n = 0 if X_sub is None else len(X_sub)
            logger.info("  %d valid cells -- skipping", n)
            continue

        logger.info("  Feature matrix: %d x %d", X_sub.shape[0], X_sub.shape[1])

        # Scale
        scaler = StandardScaler()
        X_scaled = np.nan_to_num(scaler.fit_transform(X_sub), nan=0.0, posinf=0.0, neginf=0.0)

        # PCA if many features (SAM2 alone is 256D)
        n_features = X_scaled.shape[1]
        if n_features > 50:
            from sklearn.decomposition import PCA

            n_comp = min(50, X_scaled.shape[0] - 1, n_features)
            pca = PCA(n_components=n_comp, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)
            var_expl = pca.explained_variance_ratio_.sum() * 100
            logger.info("  PCA: %d -> %d dims (%.1f%% variance)", n_features, n_comp, var_expl)

        # UMAP
        n_nbrs = min(n_neighbors, len(X_scaled) - 1)
        reducer = umap.UMAP(n_neighbors=n_nbrs, min_dist=min_dist, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)

        # HDBSCAN
        mcs = max(min(subcluster_min_size, len(X_scaled) // 5), 10)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
        labels = clusterer.fit_predict(embedding)

        n_sc = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info("  %d sub-clusters, %d noise", n_sc, n_noise)
        if n_sc == 0:
            logger.info("  No sub-clusters found")
            continue

        # Map IDs to letters (A, B, C...) sorted by count descending
        label_ids = sorted(set(labels) - {-1})
        counts = {lid: int((labels == lid).sum()) for lid in label_ids}
        sorted_ids = sorted(label_ids, key=lambda x: -counts[x])
        alpha_map = {}
        for i, lid in enumerate(sorted_ids):
            alpha_map[lid] = chr(ord("A") + i) if i < 26 else f"sub{i}"
        alpha_map[-1] = "noise"

        # Per-subcluster stats
        morph_keys = ["area_um2", "circularity", "eccentricity", "solidity", "aspect_ratio"]
        for lid in sorted_ids:
            sc_mask = labels == lid
            sc_local_indices = [sub_valid_local[j] for j, m in enumerate(sc_mask) if m]
            sc_dets = [sub_dets[li] for li in sc_local_indices]

            row = {
                "parent": parent_label,
                "subcluster": alpha_map[lid],
                "label": f"{parent_label}_{alpha_map[lid]}",
                "n_cells": counts[lid],
            }
            for mk in morph_keys:
                vals = [d.get("features", {}).get(mk) for d in sc_dets]
                vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
                row[mk] = round(float(np.mean(vals)), 2) if vals else None

            for mname, midx in sorted(marker_channels.items(), key=lambda x: x[1]):
                mk = f"ch{midx}_mean"
                vals = [d.get("features", {}).get(mk) for d in sc_dets]
                vals = [v for v in vals if v is not None]
                row[f"{mname}_mean"] = round(float(np.mean(vals)), 1) if vals else None

            master_rows.append(row)

        # Write back to detections
        for local_pos, local_valid_idx in enumerate(sub_valid_local):
            det_idx = det_indices[local_valid_idx]
            sc_id = int(labels[local_pos])
            detections[det_idx]["subcluster_id"] = sc_id
            detections[det_idx]["subcluster_label"] = f"{parent_label}_{alpha_map[sc_id]}"
            detections[det_idx]["sub_umap_x"] = float(embedding[local_pos, 0])
            detections[det_idx]["sub_umap_y"] = float(embedding[local_pos, 1])

        # --- Plots ---
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            parent_dir = subcluster_dir / parent_label
            parent_dir.mkdir(exist_ok=True)

            # UMAP colored by subcluster
            fig, ax = plt.subplots(figsize=(10, 8))
            tab_colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
            for i, lid in enumerate(sorted_ids):
                mask = labels == lid
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[tab_colors[i % len(tab_colors)]],
                    label=f"{alpha_map[lid]} (n={counts[lid]})",
                    s=5,
                    alpha=0.6,
                )
            noise_mask = labels == -1
            if noise_mask.any():
                ax.scatter(
                    embedding[noise_mask, 0],
                    embedding[noise_mask, 1],
                    c="lightgray",
                    label=f"noise (n={n_noise})",
                    s=3,
                    alpha=0.3,
                )
            ax.legend(markerscale=4)
            ax.set_title(f"Sub-clusters of '{parent_label}' ({len(X_scaled)} cells)")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            umap_path = parent_dir / "umap_subcluster.png"
            fig.savefig(umap_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved: %s", umap_path)

            # Morph violin: area + circularity per subcluster
            morph_plot_keys = ["area_um2", "circularity", "eccentricity"]
            fig, axes = plt.subplots(1, len(morph_plot_keys), figsize=(5 * len(morph_plot_keys), 5))
            if len(morph_plot_keys) == 1:
                axes = [axes]
            for ax, mk in zip(axes, morph_plot_keys):
                data = []
                tick_labels = []
                for lid in sorted_ids:
                    sc_mask = labels == lid
                    sc_local = [sub_valid_local[j] for j, m in enumerate(sc_mask) if m]
                    vals = [sub_dets[li].get("features", {}).get(mk, 0) for li in sc_local]
                    if len(vals) >= 2:
                        data.append(vals)
                        tick_labels.append(alpha_map[lid])
                if data:
                    parts = ax.violinplot(data, showmeans=True, showmedians=True)  # noqa: F841
                    ax.set_xticks(range(1, len(tick_labels) + 1))
                    ax.set_xticklabels(tick_labels)
                ax.set_title(mk)
                ax.set_ylabel(mk)
            fig.suptitle(f"Morphology: '{parent_label}' sub-clusters")
            fig.tight_layout()
            violin_path = parent_dir / "morph_violin.png"
            fig.savefig(violin_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved: %s", violin_path)

            # Marker violin per subcluster
            if marker_channels:
                marker_pairs = get_marker_mean_keys(marker_channels)
                n_markers = len(marker_pairs)
                fig, axes = plt.subplots(1, n_markers, figsize=(5 * n_markers, 5))
                if n_markers == 1:
                    axes = [axes]
                for ax, (mname, mkey) in zip(axes, marker_pairs):
                    data = []
                    tick_labels = []
                    for lid in sorted_ids:
                        sc_mask = labels == lid
                        sc_local = [sub_valid_local[j] for j, m in enumerate(sc_mask) if m]
                        vals = [sub_dets[li].get("features", {}).get(mkey, 0) for li in sc_local]
                        if len(vals) >= 2:
                            data.append(vals)
                            tick_labels.append(alpha_map[lid])
                    if data:
                        parts = ax.violinplot(data, showmeans=True, showmedians=True)  # noqa: F841
                        ax.set_xticks(range(1, len(tick_labels) + 1))
                        ax.set_xticklabels(tick_labels)
                    ax.set_title(f"{mname} (ch{marker_channels[mname]})")
                    ax.set_ylabel("Intensity")
                fig.suptitle(f"Markers: '{parent_label}' sub-clusters")
                fig.tight_layout()
                mk_violin_path = parent_dir / "marker_violin.png"
                fig.savefig(mk_violin_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("  Saved: %s", mk_violin_path)

        except ImportError:
            logger.info("  Skipping plots (matplotlib not installed)")

    # Master summary
    if master_rows:
        summary_df = pd.DataFrame(master_rows)
        summary_path = subcluster_dir / "subcluster_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Sub-cluster summary:\n%s", summary_df.to_string(index=False))
        logger.info("Saved: %s", summary_path)

    return master_rows
