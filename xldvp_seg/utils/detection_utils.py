"""Shared detection loading and feature extraction utilities.

These were duplicated across spatial_cell_analysis.py, cluster_by_features.py,
and apply_classifier.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xldvp_seg.exceptions import DataLoadError
from xldvp_seg.utils.json_utils import fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


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
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = arr.astype(np.float32)
    arr_max = arr.max()
    if arr_max <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    if arr_max <= 1.0 + 1e-6:
        return (arr * 255).clip(0, 255).astype(np.uint8)
    else:
        return arr.clip(0, 255).astype(np.uint8)


def load_detections(path: str | Path, score_threshold: float | None = None) -> list[dict]:
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
    detections = fast_json_load(str(path))
    logger.info("Loaded %s detections", f"{len(detections):,}")

    if score_threshold is not None:
        before = len(detections)
        detections = [
            d
            for d in detections
            if d.get("rf_prediction", d.get("features", {}).get("rf_prediction", 0.0))
            >= score_threshold
        ]
        logger.info(
            "Score filter >= %s: %s -> %s", score_threshold, f"{before:,}", f"{len(detections):,}"
        )

    return detections


def apply_marker_filter(detections: list[dict], filter_expr: str | None) -> list[dict]:
    """Filter detections by a marker expression like ``"MSLN_class==positive"``.

    Checks both top-level ``det[key]`` and ``det["features"][key]`` (since
    classify_markers.py stores marker classes in features).

    Args:
        detections: List of detection dicts.
        filter_expr: String in ``"key==value"`` format.

    Returns:
        Filtered list of detection dicts.
    """
    if not filter_expr:
        return detections
    if "==" not in filter_expr:
        logger.warning(
            "Marker filter '%s' has no '==' operator, ignoring. Format: key==value",
            filter_expr,
        )
        return detections

    key, value = filter_expr.split("==", 1)
    key = key.strip()
    value = value.strip()

    before = len(detections)
    filtered = [
        d for d in detections if d.get(key) == value or d.get("features", {}).get(key) == value
    ]
    logger.info("Marker filter %s==%s: %s -> %s", key, value, f"{before:,}", f"{len(filtered):,}")
    return filtered


def extract_positions_um(
    detections: list[dict],
    pixel_size_um: float | None = None,
    return_indices: bool = False,
) -> tuple[np.ndarray, float | None] | tuple[np.ndarray, float | None, list[int]]:
    """Extract cell positions in microns from a list of detection dicts.

    Resolution order for each detection:
      1. ``global_center_um`` (top-level or inside ``features``)
      2. ``global_center`` * *pixel_size_um* (if *pixel_size_um* is provided)
      3. ``global_x`` / ``global_y`` * *pixel_size_um* from features or arg
      4. If *pixel_size_um* is ``None``, try to infer from ``area`` /
         ``area_um2`` (ratio gives px_size^2).
      5. Skip the detection if none of the above work.

    Args:
        detections: List of detection dicts.
        pixel_size_um: Pixel size in micrometers.  If ``None``, the
            function attempts to infer it from ``area`` / ``area_um2``
            in the first detection's features.
        return_indices: If ``True``, also return a list of original
            detection indices that were successfully resolved.

    Returns:
        ``(positions, pixel_size_um)`` when *return_indices* is False, or
        ``(positions, pixel_size_um, valid_indices)`` when True.
        *positions* is an ``(N, 2)`` float64 array of ``[x, y]``
        coordinates in microns (only rows for which a position could be
        resolved).
    """
    # Try to infer pixel_size_um if not provided
    if pixel_size_um is None and detections:
        feats = detections[0].get("features", {})
        area_px = feats.get("area")
        area_um2 = feats.get("area_um2")
        if area_px and area_um2 and area_px > 0:
            pixel_size_um = float(np.sqrt(area_um2 / area_px))

    positions = []
    valid_indices = []
    for i, det in enumerate(detections):
        feats = det.get("features", {})

        # 1. global_center_um
        center_um = det.get("global_center_um")
        if center_um is None:
            center_um = feats.get("global_center_um")
        if center_um is not None and len(center_um) == 2:
            x, y = float(center_um[0]), float(center_um[1])
            if np.isfinite(x) and np.isfinite(y):
                positions.append([x, y])
                valid_indices.append(i)
                continue

        # 2. global_center * pixel_size_um
        center_px = det.get("global_center")
        if center_px is None:
            center_px = feats.get("global_center")
        if center_px is not None and len(center_px) == 2 and pixel_size_um:
            x = float(center_px[0]) * pixel_size_um
            y = float(center_px[1]) * pixel_size_um
            if np.isfinite(x) and np.isfinite(y):
                positions.append([x, y])
                valid_indices.append(i)
                continue

        # 3. global_x / global_y
        gx = det.get("global_x")
        gy = det.get("global_y")
        if gx is not None and gy is not None:
            ps = feats.get("pixel_size_um") or pixel_size_um
            if ps and isinstance(ps, (int, float)):
                x = float(gx) * float(ps)
                y = float(gy) * float(ps)
                if np.isfinite(x) and np.isfinite(y):
                    positions.append([x, y])
                    valid_indices.append(i)
                    continue

        # Could not resolve position for this detection — skip
        continue

    arr = np.array(positions, dtype=np.float64) if positions else np.empty((0, 2), dtype=np.float64)
    if return_indices:
        return arr, pixel_size_um, valid_indices
    return arr, pixel_size_um


def load_rf_classifier(model_path: str) -> dict:
    """Load a trained random forest classifier from a pickle/joblib file.

    Supports two serialization formats:

    * **sklearn Pipeline** — a bare ``Pipeline`` object; feature names are
      loaded from a sidecar JSON file if available.
    * **Dict format** — ``{'model': <rf>, 'feature_names': [...], ...}``
      optionally with ``'scaler'``.

    This is a cell-type-agnostic wrapper.  The original
    ``load_nmj_rf_classifier`` in the NMJ strategy module delegates here.

    Args:
        model_path: Path to ``.pkl`` or ``.joblib`` file.

    Returns:
        Dict with keys ``'pipeline'`` (sklearn Pipeline or estimator),
        ``'feature_names'`` (list[str]), ``'type'`` (``'rf'``),
        ``'raw_meta'`` (dict).
    """
    import json as _json
    from pathlib import Path as _Path

    import joblib
    from sklearn.pipeline import Pipeline

    try:
        model_data = joblib.load(model_path)
    except (EOFError, ModuleNotFoundError, FileNotFoundError) as e:
        raise DataLoadError(f"Failed to load classifier from {model_path}: {e}") from e

    if isinstance(model_data, Pipeline):
        pipeline = model_data

        model_dir = _Path(model_path).parent
        # Check for feature names sidecar (legacy NMJ naming, then generic)
        feature_names_path = None
        for name in (
            "nmj_classifier_feature_names.json",
            "classifier_feature_names.json",
            "feature_names.json",
        ):
            candidate = model_dir / name
            if candidate.exists():
                feature_names_path = candidate
                break

        if feature_names_path is not None:
            with open(feature_names_path) as f:
                feature_names = _json.load(f)
            logger.info("Loaded feature names from %s", feature_names_path)
        else:
            n_features = pipeline.named_steps["rf"].n_features_in_
            feature_names = [f"feature_{i}" for i in range(n_features)]
            logger.warning(
                "No feature names file found, using generic names for %d features",
                n_features,
            )

        result = {
            "pipeline": pipeline,
            "feature_names": feature_names,
            "type": "rf",
            "raw_meta": {},  # Legacy Pipeline format has no metadata
        }
        logger.info("Loaded RF Pipeline classifier with %d features", len(feature_names))

    else:
        rf_model = model_data.get("model", model_data.get("classifier"))
        if rf_model is None:
            raise ValueError(
                f"Classifier at {model_path} has no 'model' or 'classifier' key. "
                f"Keys found: {list(model_data.keys())}"
            )
        if not hasattr(rf_model, "predict"):
            raise ValueError(
                f"Classifier at {model_path}: loaded object has no predict() method. "
                f"Type: {type(rf_model).__name__}"
            )

        if "scaler" in model_data:
            pipeline = Pipeline([("scaler", model_data["scaler"]), ("rf", rf_model)])
        else:
            pipeline = rf_model

        result = {
            "pipeline": pipeline,
            "feature_names": model_data.get("feature_names", []),
            "type": "rf",
            "raw_meta": model_data,  # Preserve training metadata for provenance
        }
        logger.info(
            "Loaded RF classifier (legacy format) with %d features",
            len(result["feature_names"]),
        )
        if "accuracy" in model_data:
            logger.info("  Accuracy: %s", model_data["accuracy"])

    return result


def extract_feature_matrix(
    detections: list[dict], feature_names: list[str]
) -> tuple[np.ndarray, list[int]]:
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
        feats = det.get("features", {})
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


# ---------------------------------------------------------------------------
# Contour field access (backwards-compatible with old field names)
# ---------------------------------------------------------------------------


def get_contour_px(det: dict) -> list[list[float]] | None:
    """Get contour in pixels from a detection dict.

    Handles both new (``contour_px``) and legacy (``contour_dilated_px``)
    field names.  Returns ``None`` if neither is present.

    Uses explicit ``is not None`` checks (not ``or``) to avoid skipping
    an empty list ``[]`` and falling through to the legacy field.
    """
    c = det.get("contour_px")
    if c is not None:
        return c
    return det.get("contour_dilated_px")


def get_contour_um(det: dict) -> list[list[float]] | None:
    """Get contour in micrometers from a detection dict.

    Handles both new (``contour_um``) and legacy (``contour_dilated_um``)
    field names.  Returns ``None`` if neither is present.
    """
    c = det.get("contour_um")
    if c is not None:
        return c
    return det.get("contour_dilated_um")
