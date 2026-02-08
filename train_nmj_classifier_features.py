#!/usr/bin/env python3
"""
Train NMJ classifier using extracted multi-channel features.
Uses Random Forest on ~2,400 features instead of ResNet on images.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def load_features_and_annotations(detections_path, annotations_path):
    """Load features from detections and match with annotations.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (0=negative, 1=positive)
        feature_names: List of feature names
    """
    # Load detections
    with open(detections_path) as f:
        detections = json.load(f)
    logger.info(f"Loaded {len(detections)} detections")

    # Load annotations
    with open(annotations_path) as f:
        annotations = json.load(f)

    positive_ids = set(annotations.get('positive', []))
    negative_ids = set(annotations.get('negative', []))
    logger.info(f"Annotations: {len(positive_ids)} positive, {len(negative_ids)} negative")

    # Build lookup by various ID formats
    det_by_id = {}
    for det in detections:
        # tile_origin is [x, y]
        tile_origin = det.get('tile_origin', [0, 0])
        tile_x = int(tile_origin[0])
        tile_y = int(tile_origin[1])
        nmj_id = det.get('id', '')

        # Full ID format used in annotations: tile_x_tile_y_nmj_N
        full_id = f"{tile_x}_{tile_y}_{nmj_id}"
        det_by_id[full_id] = det

        # Also store by uid for fallback
        uid = det.get('uid', '')
        if uid:
            det_by_id[uid] = det

        # Also store by just nmj_id for fallback
        det_by_id[nmj_id] = det

    # Extract features for annotated samples
    X = []
    y = []
    feature_names = None
    matched_pos = 0
    matched_neg = 0

    # First pass: determine which features are scalar (not lists/arrays)
    sample_det = None
    for det in detections:
        if det.get('features'):
            sample_det = det
            break

    if sample_det:
        all_features = sample_det.get('features', {})
        # Only keep scalar features (exclude bbox, embeddings, etc.)
        feature_names = sorted([k for k, v in all_features.items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool)])
        logger.info(f"Using {len(feature_names)} scalar features (excluded lists/arrays)")

    for sample_id in positive_ids:
        if sample_id in det_by_id:
            det = det_by_id[sample_id]
            features = det.get('features', {})
            if features and feature_names:
                X.append([float(features.get(k, 0)) for k in feature_names])
                y.append(1)
                matched_pos += 1

    for sample_id in negative_ids:
        if sample_id in det_by_id:
            det = det_by_id[sample_id]
            features = det.get('features', {})
            if features and feature_names:
                X.append([float(features.get(k, 0)) for k in feature_names])
                y.append(0)
                matched_neg += 1

    logger.info(f"Matched: {matched_pos} positive, {matched_neg} negative")
    logger.info(f"Feature dimensions: {len(feature_names) if feature_names else 0}")

    return np.array(X), np.array(y), feature_names


def main():
    parser = argparse.ArgumentParser(description='Train NMJ Feature Classifier')
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to nmj_detections.json')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to annotations JSON')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for model')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees in Random Forest')
    args = parser.parse_args()

    setup_logging()

    # Load data
    logger.info("Loading features and annotations...")
    X, y, feature_names = load_features_and_annotations(
        args.detections, args.annotations
    )

    if len(X) < 50:
        logger.error(f"Not enough samples: {len(X)}")
        return

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Test: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    logger.info(f"\nTraining Random Forest with {args.n_estimators} trees...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    logger.info("\n" + "="*50)
    logger.info("CLASSIFICATION REPORT")
    logger.info("="*50)
    logger.info("\n" + classification_report(y_test, y_pred,
                                             target_names=['Negative', 'Positive']))

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Cross-validation using Pipeline to avoid data leakage
    # (scaler is fit only on training folds, not on validation data)
    logger.info("\nCross-validation (5-fold)...")
    cv_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )),
    ])
    cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=5)
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Feature importance
    logger.info("\nTop 20 most important features:")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    for i, idx in enumerate(indices):
        logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "nmj_classifier_rf.pkl"
    joblib.dump({
        'classifier': clf,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': (y_pred == y_test).mean(),
        'n_positive': int(y.sum()),
        'n_negative': int(len(y) - y.sum()),
    }, model_path)

    logger.info(f"\nModel saved to: {model_path}")
    logger.info(f"Test accuracy: {(y_pred == y_test).mean():.4f}")


if __name__ == '__main__':
    main()
