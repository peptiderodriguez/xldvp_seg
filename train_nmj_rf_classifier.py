#!/usr/bin/env python3
"""
Train NMJ Random Forest classifier using extracted morphological features.

Uses the 2300+ features extracted during segmentation (area, solidity, texture, etc.)
to train a Random Forest classifier. This is more interpretable than CNN and leverages
all the feature engineering work.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)


def load_annotations(annotations_path: Path) -> tuple:
    """Load annotations from JSON file."""
    with open(annotations_path) as f:
        data = json.load(f)

    # Support both formats
    if 'positive' in data:
        positive_ids = set(data['positive'])
        negative_ids = set(data['negative'])
    elif 'annotations' in data:
        positive_ids = {k for k, v in data['annotations'].items() if v == 'yes'}
        negative_ids = {k for k, v in data['annotations'].items() if v == 'no'}
    else:
        raise ValueError("Unknown annotation format")

    return positive_ids, negative_ids


def load_detections(detections_path: Path) -> dict:
    """Load detections with features from JSON."""
    with open(detections_path) as f:
        data = json.load(f)

    # Index by multiple possible ID formats for matching
    indexed = {}
    for d in data:
        # Primary: uid
        if 'uid' in d:
            indexed[d['uid']] = d

        # Alternative 1: construct tile-based ID without slide name
        # Format: {tileX}_{tileY}_{id} (matches HTML export format)
        if 'tile_origin' in d and 'id' in d:
            tile_x, tile_y = d['tile_origin']
            alt_id = f"{tile_x}_{tile_y}_{d['id']}"
            indexed[alt_id] = d

        # Alternative 2: construct tile-based ID with slide name
        # Format: {slide}_{tileX}_{tileY}_{id}
        if 'tile_origin' in d and 'id' in d and 'slide_name' in d:
            tile_x, tile_y = d['tile_origin']
            alt_id2 = f"{d['slide_name']}_{tile_x}_{tile_y}_{d['id']}"
            indexed[alt_id2] = d

        # Alternative 3: just the id
        if 'id' in d:
            indexed[d['id']] = d

    return indexed


def extract_feature_matrix(detections: dict, uids: list, feature_names: list = None) -> tuple:
    """Extract feature matrix for given UIDs."""
    if not uids:
        return np.array([]), [], []

    # Get feature names from first detection, excluding non-scalar features
    if feature_names is None:
        sample = next(iter(detections.values()))
        feature_names = []
        for k, v in sorted(sample['features'].items()):
            if isinstance(v, (int, float, np.integer, np.floating)):
                feature_names.append(k)
            # Skip lists like bbox, centroid

    X = []
    valid_uids = []

    for uid in uids:
        if uid in detections:
            features = detections[uid]['features']
            row = []
            for fn in feature_names:
                val = features.get(fn, 0)
                # Handle any remaining non-scalars
                if isinstance(val, (list, tuple)):
                    val = 0
                row.append(float(val) if val is not None else 0)
            X.append(row)
            valid_uids.append(uid)

    return np.array(X, dtype=np.float32), valid_uids, feature_names


def train_rf_classifier(
    annotations_path: str,
    detections_path: str,
    output_dir: str,
    n_estimators: int = 100,
    max_depth: int = None,
    test_size: float = 0.2,
):
    """Train Random Forest classifier on extracted features."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading annotations...")
    positive_ids, negative_ids = load_annotations(Path(annotations_path))
    logger.info(f"  Positive: {len(positive_ids)}")
    logger.info(f"  Negative: {len(negative_ids)}")

    logger.info("Loading detections with features...")
    detections = load_detections(Path(detections_path))
    logger.info(f"  Total detections: {len(detections)}")

    # Extract features for annotated samples
    logger.info("Extracting feature matrices...")

    # Get feature names
    sample = next(iter(detections.values()))
    feature_names = sorted(sample['features'].keys())
    logger.info(f"  Number of features: {len(feature_names)}")

    # Positive samples
    X_pos, valid_pos, _ = extract_feature_matrix(detections, list(positive_ids), feature_names)
    y_pos = np.ones(len(valid_pos))

    # Negative samples
    X_neg, valid_neg, _ = extract_feature_matrix(detections, list(negative_ids), feature_names)
    y_neg = np.zeros(len(valid_neg))

    logger.info(f"  Matched positive samples: {len(valid_pos)}")
    logger.info(f"  Matched negative samples: {len(valid_neg)}")

    if len(valid_pos) == 0 or len(valid_neg) == 0:
        raise ValueError("No matching samples found! Check annotation UIDs match detection UIDs.")

    # Combine
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(f"\nTraining set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    logger.info(f"\nTraining Random Forest (n_estimators={n_estimators})...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]

    accuracy = (y_pred == y_test).mean()
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST ACCURACY: {accuracy:.4f}")
    logger.info(f"{'='*50}")

    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # Cross-validation
    logger.info("\nCross-validation (5-fold)...")
    X_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
    logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Feature importance
    logger.info("\nTop 20 most important features:")
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1][:20]
    for i, idx in enumerate(indices):
        logger.info(f"  {i+1:2d}. {feature_names[idx]:30s} = {importance[idx]:.4f}")

    # Save model
    model_path = output_dir / 'nmj_rf_classifier.pkl'
    model_data = {
        'model': rf,
        'scaler': scaler,
        'feature_names': feature_names,
        'accuracy': accuracy,
        'cv_accuracy': cv_scores.mean(),
    }
    joblib.dump(model_data, model_path)
    logger.info(f"\nModel saved to: {model_path}")

    # Save feature importance
    importance_path = output_dir / 'feature_importance.json'
    importance_data = {
        feature_names[i]: float(importance[i])
        for i in np.argsort(importance)[::-1]
    }
    with open(importance_path, 'w') as f:
        json.dump(importance_data, f, indent=2)
    logger.info(f"Feature importance saved to: {importance_path}")

    return rf, scaler, feature_names, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train NMJ Random Forest Classifier')
    parser.add_argument('--annotations', required=True, help='Path to annotations JSON')
    parser.add_argument('--detections', required=True, help='Path to detections JSON with features')
    parser.add_argument('--output-dir', default='/home/dude/nmj_output/rf_classifier', help='Output directory')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max-depth', type=int, default=None, help='Max tree depth')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')

    args = parser.parse_args()

    train_rf_classifier(
        args.annotations,
        args.detections,
        args.output_dir,
        args.n_estimators,
        args.max_depth,
        args.test_size,
    )


if __name__ == '__main__':
    main()
