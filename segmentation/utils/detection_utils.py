"""Shared detection loading and feature extraction utilities.

These were duplicated across spatial_cell_analysis.py, cluster_by_features.py,
and apply_classifier.py.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_detections(path, score_threshold=None):
    """Load detections JSON, optionally filtering by rf_prediction.

    Args:
        path: Path to detections JSON file.
        score_threshold: If not None, keep only detections with
            ``rf_prediction >= score_threshold``. Looks in both the
            top-level key and ``features.rf_prediction``.

    Returns:
        List of detection dicts.
    """
    path = Path(path)
    if not path.exists():
        logger.error("Detections file not found: %s", path)
        sys.exit(1)

    logger.info("Loading detections from %s...", path)
    with open(path) as f:
        detections = json.load(f)
    logger.info("Loaded %s detections", f"{len(detections):,}")

    if score_threshold is not None:
        before = len(detections)
        detections = [
            d for d in detections
            if d.get('rf_prediction', d.get('features', {}).get('rf_prediction', 0.0))
            >= score_threshold
        ]
        logger.info("Score filter >= %s: %s -> %s",
                     score_threshold, f"{before:,}", f"{len(detections):,}")

    return detections


def extract_feature_matrix(detections, feature_names):
    """Extract a feature matrix from detections for given feature names.

    Pre-allocates a numpy array and fills it row by row. Detections
    without features are skipped. Missing feature keys default to 0;
    list/tuple values default to 0; None defaults to 0.

    Args:
        detections: List of detection dicts, each with a ``features`` sub-dict.
        feature_names: Ordered list of feature key strings.

    Returns:
        ``(X, valid_indices)`` where ``X`` is ``(n_valid, n_features)``
        float32 array and ``valid_indices`` maps rows back to detection
        list indices.
    """
    n_det = len(detections)
    n_feat = len(feature_names)
    X = np.zeros((n_det, n_feat), dtype=np.float32)
    valid_indices = []

    for i, det in enumerate(detections):
        feats = det.get('features', {})
        if not feats:
            continue
        for j, name in enumerate(feature_names):
            val = feats.get(name, 0)
            if isinstance(val, (list, tuple)):
                val = 0
            X[i, j] = float(val) if val is not None else 0.0
        valid_indices.append(i)

    X_valid = X[valid_indices]
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
    return X_valid, valid_indices
