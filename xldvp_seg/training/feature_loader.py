"""Feature loading and filtering for classifier training.

Provides functions to filter feature names by feature set (morph, morph_sam2,
channel_stats, all) and to load feature matrices matched against annotation
labels from detection JSON files.
"""

import numpy as np

from xldvp_seg.utils.json_utils import fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def is_morph_feature(name):
    """Check if feature is morph + channel stats (excludes SAM2/ResNet/DINOv2 embeddings)."""
    return not name.startswith(("sam2_", "resnet_", "dinov2_"))


def is_morph_sam2_feature(name):
    """Check if a feature name is morphological or SAM2."""
    return not name.startswith(("resnet_", "dinov2_"))


def is_channel_stats_feature(name):
    """Check if a feature name is a per-channel intensity stat or inter-channel ratio."""
    return name.startswith("ch") or name.startswith("channel_")


def filter_feature_names(feature_names, feature_set):
    """Filter feature names based on the requested feature set."""
    if feature_set == "morph":
        return [n for n in feature_names if is_morph_feature(n)]
    elif feature_set == "morph_sam2":
        return [n for n in feature_names if is_morph_sam2_feature(n)]
    elif feature_set == "channel_stats":
        return [n for n in feature_names if is_channel_stats_feature(n)]
    else:  # 'all'
        return feature_names


def load_features_and_annotations(detections_path, annotations_path, feature_set="all"):
    """Load features from detections and match with annotations.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (0=negative, 1=positive)
        feature_names: List of feature names
    """
    # Load detections
    detections = fast_json_load(str(detections_path))
    logger.info("Loaded %d detections", len(detections))

    # Load annotations
    annotations = fast_json_load(str(annotations_path))

    positive_ids = set(annotations.get("positive", []))
    negative_ids = set(annotations.get("negative", []))
    logger.info("Annotations: %d positive, %d negative", len(positive_ids), len(negative_ids))

    # Build lookup by various ID formats
    det_by_id = {}
    for det in detections:
        # tile_origin is [x, y]
        tile_origin = det.get("tile_origin", [0, 0])
        tile_x = int(tile_origin[0])
        tile_y = int(tile_origin[1])
        nmj_id = det.get("id", "")

        # Full ID format used in annotations: tile_x_tile_y_nmj_N
        full_id = f"{tile_x}_{tile_y}_{nmj_id}"
        det_by_id[full_id] = det

        # Also store by uid for fallback
        uid = det.get("uid", "")
        if uid:
            det_by_id[uid] = det

    # Extract features for annotated samples
    X = []
    y = []
    feature_names = []
    matched_pos = 0
    matched_neg = 0

    # First pass: determine which features are scalar (not lists/arrays)
    sample_det = None
    for det in detections:
        if det.get("features"):
            sample_det = det
            break

    if sample_det:
        all_features = sample_det.get("features", {})
        # Only keep scalar features (exclude bbox, embeddings stored as lists, etc.)
        all_scalar_names = sorted(
            [
                k
                for k, v in all_features.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
        )
        feature_names = filter_feature_names(all_scalar_names, feature_set)
        logger.info(
            "Feature set '%s': %d features (from %d total scalar)",
            feature_set,
            len(feature_names),
            len(all_scalar_names),
        )

    for sample_id in positive_ids:
        if sample_id in det_by_id:
            det = det_by_id[sample_id]
            features = det.get("features", {})
            if features and feature_names:
                X.append([float(features.get(k, 0)) for k in feature_names])
                y.append(1)
                matched_pos += 1

    for sample_id in negative_ids:
        if sample_id in det_by_id:
            det = det_by_id[sample_id]
            features = det.get("features", {})
            if features and feature_names:
                X.append([float(features.get(k, 0)) for k in feature_names])
                y.append(0)
                matched_neg += 1

    logger.info("Matched: %d positive, %d negative", matched_pos, matched_neg)
    logger.info("Feature dimensions: %d", len(feature_names))

    return np.array(X), np.array(y), feature_names
