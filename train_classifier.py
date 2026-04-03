#!/usr/bin/env python3
"""
Train RF classifier from annotated detections.

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

from xldvp_seg.training.feature_loader import (
    load_features_and_annotations,
)
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train RF Feature Classifier")
    parser.add_argument(
        "--detections",
        type=str,
        default=None,
        help="Path to detections JSON (required for training)",
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
        from xldvp_seg.utils.classifier_registry import list_classifiers

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
    # is scale-invariant. load_rf_classifier() handles wrapping if needed.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"rf_classifier_{timestamp}.pkl"
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
            "feature_extraction": "original_mask",
        },
        model_path,
    )

    # Symlink latest classifier for easy reference
    latest_link = output_dir / "rf_classifier_latest.pkl"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(model_path.name)
    except OSError:
        pass
    logger.info(f"\nModel saved to: {model_path}")

    # Auto-register if --register or --name provided
    if args.register or args.name:
        from xldvp_seg.utils.classifier_registry import register_classifier

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
