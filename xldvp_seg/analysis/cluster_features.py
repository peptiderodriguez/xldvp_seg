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

import numpy as np

from xldvp_seg.utils.json_utils import fast_json_load
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
