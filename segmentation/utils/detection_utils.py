"""Shared detection loading and feature extraction utilities.

These were duplicated across spatial_cell_analysis.py, cluster_by_features.py,
and apply_classifier.py.
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def safe_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Safely convert any numeric array to uint8 [0, 255].

    Handles float32/float64 tiles that may be in [0, 1] range (e.g., after
    photobleaching correction or normalization), uint16 tiles, and already-uint8.
    Bare ``.astype(np.uint8)`` on float [0, 1] data truncates everything to 0.

    Args:
        arr: Input array of any numeric dtype.

    Returns:
        uint8 array with values in [0, 255].
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 256).astype(np.uint8)
    arr = arr.astype(np.float32)
    arr_max = arr.max()
    if arr_max <= 0:
        return np.zeros(arr.shape[:2] + ((3,) if arr.ndim == 3 else ()), dtype=np.uint8)
    if arr_max <= 1.0 + 1e-6:
        return (arr * 255).clip(0, 255).astype(np.uint8)
    elif arr_max <= 255.0:
        return (arr * (255.0 / arr_max)).clip(0, 255).astype(np.uint8)
    else:
        return arr.clip(0, 255).astype(np.uint8)


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
        raise FileNotFoundError(f"Detections file not found: {path}")

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
