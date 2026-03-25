#!/usr/bin/env python3
"""
Train NMJ classifier using extracted multi-channel features.

Uses Random Forest on raw (unscaled) features — RF is scale-invariant,
so no StandardScaler is needed. This keeps the prediction path simple:
just pass raw features to classifier.predict_proba().

Feature sets:
  --feature-set morph          → morphological features only (~78)
  --feature-set morph_sam2     → morph + SAM2 embeddings (~334)
  --feature-set channel_stats  → per-channel intensity features only (ch*_ prefixed)
  --feature-set all            → all scalar features (default)
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

from segmentation.utils.json_utils import fast_json_load
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Feature name prefixes (documentation only — actual filtering uses is_*_feature() functions below)
MORPH_PREFIXES = (
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "extent",
    "circularity",
    "aspect_ratio",
    "compactness",
    "convex_area",
    "filled_area",
    "euler_number",
    "equivalent_diameter",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "feret_diameter",
    "skeleton",
    "hu_moment",
    "centroid",
    "bbox",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "intensity_range",
    "median_intensity",
    "intensity_skew",
    "intensity_kurtosis",
    "percentile",
    "entropy",
    "haralick",
    "gabor",
    "lbp",
    "gradient",
    "edge",
    "texture",
    "shape",
    "concavity",
    "roughness",
    "branching",
    "endpoint",
    "curvature",
    # Multi-channel stats (ch0-ch5 for 6-channel islet, ch0-ch2 for 3-channel NMJ)
    "ch0_",
    "ch1_",
    "ch2_",
    "ch3_",
    "ch4_",
    "ch5_",
    "channel_",
)
SAM2_PREFIX = "sam2_"


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
    logger.info(f"Loaded {len(detections)} detections")

    # Load annotations
    annotations = fast_json_load(str(annotations_path))

    positive_ids = set(annotations.get("positive", []))
    negative_ids = set(annotations.get("negative", []))
    logger.info(f"Annotations: {len(positive_ids)} positive, {len(negative_ids)} negative")

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
    feature_names = None
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
            f"Feature set '{feature_set}': {len(feature_names)} features "
            f"(from {len(all_scalar_names)} total scalar)"
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

    logger.info(f"Matched: {matched_pos} positive, {matched_neg} negative")
    logger.info(f"Feature dimensions: {len(feature_names) if feature_names else 0}")

    return np.array(X), np.array(y), feature_names


def main():
    parser = argparse.ArgumentParser(description="Train NMJ Feature Classifier")
    parser.add_argument(
        "--detections",
        type=str,
        default=None,
        help="Path to nmj_detections.json (required for training)",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to annotations JSON (required for training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model (required for training)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="Number of trees in Random Forest"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="all",
        choices=["morph", "morph_sam2", "channel_stats", "all"],
        help="Feature subset to use (default: all). "
        "channel_stats: per-channel intensity features only (ch*_ prefixed)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Classifier name (e.g. "vessel_v1"). Auto-generated if not provided.',
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register classifier in registry after training. Implied when --name is provided.",
    )
    parser.add_argument(
        "--list-classifiers",
        action="store_true",
        help="List available classifiers from registry and exit (no training).",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        default="unknown",
        help='Cell type for metadata (default: "unknown").',
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional description string for the classifier.",
    )
    args = parser.parse_args()

    setup_logging()

    # List classifiers and exit if requested
    if args.list_classifiers:
        from segmentation.utils.classifier_registry import list_classifiers

        list_classifiers()
        return

    # Validate required args for training
    missing = []
    if not args.detections:
        missing.append("--detections")
    if not args.annotations:
        missing.append("--annotations")
    if not args.output_dir:
        missing.append("--output-dir")
    if missing:
        parser.error(f"the following arguments are required for training: {', '.join(missing)}")

    # Load data
    logger.info("Loading features and annotations...")
    X, y, feature_names = load_features_and_annotations(
        args.detections, args.annotations, feature_set=args.feature_set
    )

    if len(X) < 50:
        logger.error(f"Not enough samples: {len(X)}")
        return

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")

    # RF hyperparams (no scaler — RF is scale-invariant)
    rf_params = dict(
        n_estimators=args.n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # --- Step 1: Cross-validation on training set ---
    logger.info("\nCross-validation (5-fold on train set)...")
    cv_clf = RandomForestClassifier(**rf_params)
    cv_scores = cross_val_score(cv_clf, X_train, y_train, cv=5, scoring="f1")
    logger.info(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # --- Step 2: Evaluate on held-out test set ---
    logger.info(f"\nEvaluating on held-out test set ({len(X_test)} samples)...")
    eval_clf = RandomForestClassifier(**rf_params)
    eval_clf.fit(X_train, y_train)
    y_pred = eval_clf.predict(X_test)

    logger.info("\n" + "=" * 50)
    logger.info("CLASSIFICATION REPORT (held-out test set)")
    logger.info("=" * 50)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    test_accuracy = (y_pred == y_test).mean()

    # --- Step 3: Retrain on ALL data for the final model ---
    logger.info(f"\nRetraining on ALL {len(X)} samples for final model...")
    final_clf = RandomForestClassifier(**rf_params)
    final_clf.fit(X, y)

    # Feature importance (from final model)
    logger.info("\nTop 20 most important features:")
    importances = final_clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    for i, idx in enumerate(indices):
        logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # --- Step 4: Save model (no scaler!) ---
    # Save raw RF classifier directly — no Pipeline wrapping needed since RF
    # is scale-invariant. load_nmj_rf_classifier() handles wrapping if needed.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"nmj_classifier_rf_{timestamp}.pkl"
    clf_name = args.name or f"rf_{args.feature_set}_{datetime.now().strftime('%Y%m%d')}"
    joblib.dump(
        {
            "model": final_clf,
            "classifier": final_clf,  # Legacy key for backward compat
            "feature_names": feature_names,
            "feature_set": args.feature_set,
            "test_accuracy": test_accuracy,
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "n_total": len(y),
            "name": clf_name,
            "trained_at": datetime.now().isoformat(),
            "training_annotations_path": os.path.abspath(args.annotations),
            "cell_type": getattr(args, "cell_type", "unknown") or "unknown",
            "description": getattr(args, "description", "") or "",
        },
        model_path,
    )

    # Symlink latest classifier for easy reference
    latest_link = output_dir / "nmj_classifier_rf_latest.pkl"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(model_path.name)
    except OSError:
        pass
    logger.info(f"\nModel saved to: {model_path}")

    # Auto-register if --register or --name provided
    if args.register or args.name:
        from segmentation.utils.classifier_registry import register_classifier

        # Derive training slide name from detections path (e.g. .../slide_name/run_dir/det.json)
        _det_path = Path(args.detections).resolve()
        _training_slide = _det_path.parent.parent.name if _det_path.parent.parent.exists() else None
        reg_meta = {
            "feature_set": args.feature_set,
            "cv_f1_mean": float(cv_scores.mean()),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "cell_type": getattr(args, "cell_type", "unknown") or "unknown",
            "trained_at": datetime.now().isoformat(),
            "training_slide": _training_slide,
            "training_annotations_path": os.path.abspath(args.annotations),
            "description": getattr(args, "description", "") or "",
        }
        register_classifier(clf_name, model_path, reg_meta)

    # --- Step 5: Self-test — load and verify the saved model ---
    logger.info("\nSelf-test: loading saved model and verifying predictions...")
    loaded = joblib.load(model_path)
    loaded_model = loaded["model"]
    loaded_names = loaded["feature_names"]

    assert len(loaded_names) == len(
        feature_names
    ), f"Feature name mismatch: {len(loaded_names)} vs {len(feature_names)}"

    # Verify predictions match on a subset
    test_probs = loaded_model.predict_proba(X[:10])[:, 1]
    expected_probs = final_clf.predict_proba(X[:10])[:, 1]
    assert np.allclose(test_probs, expected_probs), "Saved model predictions don't match!"

    # Also verify legacy key works
    loaded_legacy = loaded["classifier"]
    legacy_probs = loaded_legacy.predict_proba(X[:10])[:, 1]
    assert np.allclose(
        legacy_probs, expected_probs
    ), "Legacy classifier key predictions don't match!"

    logger.info("Self-test PASSED")
    logger.info("\nSummary:")
    logger.info(f"  Feature set: {args.feature_set} ({len(feature_names)} features)")
    logger.info(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    logger.info(f"  Test accuracy: {test_accuracy:.4f}")
    logger.info(f"  Final model trained on: {len(X)} samples")
    logger.info(f"  Saved to: {model_path}")


if __name__ == "__main__":
    main()
