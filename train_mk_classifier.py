#!/usr/bin/env python3
"""
Train a Random Forest classifier for MK (megakaryocyte) cell filtering.

Pipeline: 10% segmentation -> HTML annotation -> convert_annotations_to_training.py
          -> training_data.json -> **this script** -> mk_classifier.pkl
          -> run_unified_FAST.py --mk-classifier mk_classifier.pkl (100% run)

Input:  training_data.json from convert_annotations_to_training.py
Output: .pkl compatible with run_unified_FAST.py's apply_classifier()
        Keys: 'classifier', 'feature_names', 'n_samples', 'accuracy', etc.

Usage:
    python train_mk_classifier.py \
        --training-data training_data.json \
        --output mk_classifier.pkl \
        --morph-only          # optional: exclude sam2_emb_* and resnet_* features
        --n-estimators 500    # optional (default 500)
        --test-size 0.2       # optional
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def load_training_data(path):
    """Load training_data.json produced by convert_annotations_to_training.py."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def filter_mk_samples(data):
    """Keep only MK cell_type samples."""
    samples = data['training_samples']
    mk_samples = [s for s in samples if s['cell_type'] == 'mk']
    return mk_samples


def get_morph_feature_names(feature_names):
    """Return only morphological feature names (exclude sam2_emb_* and resnet_*)."""
    return [f for f in feature_names if not f.startswith('sam2_emb_') and not f.startswith('resnet_')]


def samples_to_arrays(samples, feature_names):
    """Convert list of sample dicts to (X, y) numpy arrays."""
    X = np.array([
        [s['features'].get(name, 0.0) for name in feature_names]
        for s in samples
    ], dtype=np.float32)
    y = np.array([s['label'] for s in samples])
    return X, y


def print_class_balance(y, label="Dataset"):
    """Print class balance summary."""
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    total = len(y)
    print(f"\n  {label}: {total} samples")
    print(f"    Positive (true MK): {n_pos} ({100*n_pos/total:.1f}%)")
    print(f"    Negative (false MK): {n_neg} ({100*n_neg/total:.1f}%)")
    ratio = n_pos / n_neg if n_neg > 0 else float('inf')
    print(f"    Pos/Neg ratio: {ratio:.2f}")


def print_feature_importances(clf, feature_names, top_n=20):
    """Print top feature importances."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n  Top-{min(top_n, len(feature_names))} Feature Importances:")
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        print(f"    {i+1:3d}. {feature_names[idx]:30s}  {importances[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Random Forest classifier for MK cell filtering'
    )
    parser.add_argument('--training-data', type=str, required=True,
                        help='Path to training_data.json from convert_annotations_to_training.py')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for classifier .pkl')
    parser.add_argument('--morph-only', action='store_true',
                        help='Use only morphological features (exclude sam2_emb_*, resnet_*)')
    parser.add_argument('--n-estimators', type=int, default=500,
                        help='Number of trees in Random Forest (default: 500)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data held out for testing (default: 0.2)')

    args = parser.parse_args()

    print("=" * 70)
    print("MK RANDOM FOREST CLASSIFIER TRAINING")
    print("=" * 70)

    # -- 1. Load data ------------------------------------------------------
    print(f"\nLoading training data: {args.training_data}")
    data = load_training_data(args.training_data)

    all_feature_names = sorted(data['feature_names'])
    print(f"  Total feature names in file: {len(all_feature_names)}")

    # -- 2. Filter to MK only ---------------------------------------------
    mk_samples = filter_mk_samples(data)
    if len(mk_samples) == 0:
        print("\nERROR: No MK samples found in training data!")
        sys.exit(1)
    print(f"  MK samples: {len(mk_samples)}")

    # -- 3. Select features ------------------------------------------------
    if args.morph_only:
        feature_names = get_morph_feature_names(all_feature_names)
        feature_set = 'morph_only'
        print(f"\n  Using morphological features only: {len(feature_names)}")
    else:
        feature_names = all_feature_names
        feature_set = 'all'
        print(f"\n  Using all features: {len(feature_names)}")

    n_morph = len([f for f in feature_names if not f.startswith('sam2_emb_') and not f.startswith('resnet_')])
    n_sam2 = len([f for f in feature_names if f.startswith('sam2_emb_')])
    n_resnet = len([f for f in feature_names if f.startswith('resnet_')])
    print(f"    Morphological: {n_morph}")
    print(f"    SAM2 embeddings: {n_sam2}")
    print(f"    ResNet embeddings: {n_resnet}")

    # -- 4. Build arrays ---------------------------------------------------
    X, y = samples_to_arrays(mk_samples, feature_names)
    print(f"\n  Array shape: X={X.shape}, y={y.shape}")
    print_class_balance(y, "Full MK dataset")

    # Check for NaN/Inf
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan > 0 or n_inf > 0:
        print(f"\n  WARNING: Found {n_nan} NaN and {n_inf} Inf values - replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # -- 5. Train/test split -----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\n  Train/test split ({1 - args.test_size:.0%}/{args.test_size:.0%}):")
    print_class_balance(y_train, "Train")
    print_class_balance(y_test, "Test")

    # -- 6. Cross-validation on train set only -----------------------------
    print("\n" + "-" * 50)
    print("5-FOLD STRATIFIED CROSS-VALIDATION (train set only)")
    print("-" * 50)

    clf_cv = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf_cv, X_train, y_train, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(clf_cv, X_train, y_train, cv=cv, scoring='f1')

    print(f"\n  CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"    Per fold: {[f'{s:.4f}' for s in cv_acc]}")
    print(f"  CV F1:       {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"    Per fold: {[f'{s:.4f}' for s in cv_f1]}")

    # -- 7. Train on train set, evaluate on held-out test ------------------
    print("\n" + "-" * 50)
    print("HELD-OUT TEST SET EVALUATION")
    print("-" * 50)

    # NOTE: No scaler! RF is scale-invariant, and run_unified_FAST.py's
    # apply_classifier() passes raw features directly (line 685).
    clf_eval = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )
    clf_eval.fit(X_train, y_train)
    y_pred = clf_eval.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    for line in report.split('\n'):
        print(f"    {line}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted Neg  Predicted Pos")
    print(f"    Actual Neg   {cm[0, 0]:>12d}  {cm[0, 1]:>12d}")
    print(f"    Actual Pos   {cm[1, 0]:>12d}  {cm[1, 1]:>12d}")

    print_feature_importances(clf_eval, feature_names, top_n=20)

    # -- 8. Retrain on ALL data for final model ----------------------------
    print("\n" + "-" * 50)
    print("FINAL MODEL (retrained on ALL data)")
    print("-" * 50)

    clf_final = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )
    clf_final.fit(X, y)
    print(f"\n  Trained on {X.shape[0]} samples, {X.shape[1]} features")

    print_feature_importances(clf_final, feature_names, top_n=20)

    # -- 9. Save pkl -------------------------------------------------------
    # Key names MUST match what run_unified_FAST.py expects:
    #   clf_data['classifier']     (line 650)
    #   clf_data['feature_names']  (line 651)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clf_data = {
        'classifier': clf_final,       # NOT 'model' like NMJ script
        'feature_names': feature_names, # sorted, same order used in training
        'n_samples': int(X.shape[0]),
        'accuracy': float(acc),
        'cv_accuracy': float(cv_acc.mean()),
        'cell_type': 'mk',
        'feature_set': feature_set,
    }

    joblib.dump(clf_data, output_path)
    print(f"\n  Saved classifier to: {output_path}")
    print(f"    Keys: {list(clf_data.keys())}")

    # -- 10. Self-test: reload and verify ----------------------------------
    print("\n" + "-" * 50)
    print("SELF-TEST: reload and verify predict()")
    print("-" * 50)

    loaded = joblib.load(output_path)
    assert 'classifier' in loaded, "Missing 'classifier' key"
    assert 'feature_names' in loaded, "Missing 'feature_names' key"
    assert loaded['feature_names'] == feature_names, "Feature names mismatch"

    # Predict on a dummy zero vector
    dummy_X = np.zeros((1, len(feature_names)), dtype=np.float32)
    pred = loaded['classifier'].predict(dummy_X)
    proba = loaded['classifier'].predict_proba(dummy_X)
    print(f"  predict() on zeros   -> class={pred[0]}, proba={proba[0]}")

    # Predict on first real sample
    real_X = X[:1]
    pred = loaded['classifier'].predict(real_X)
    proba = loaded['classifier'].predict_proba(real_X)
    print(f"  predict() on sample  -> class={pred[0]}, proba={proba[0]}")

    print(f"\n  Self-test PASSED")

    # -- Summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Classifier: {output_path}")
    print(f"  Cell type:  mk")
    print(f"  Features:   {len(feature_names)} ({feature_set})")
    print(f"  Samples:    {X.shape[0]}")
    print(f"  CV Acc:     {cv_acc.mean():.4f}")
    print(f"  Test Acc:   {acc:.4f}")
    print(f"  Test F1:    {f1:.4f}")
    print(f"\n  To use in 100% segmentation:")
    print(f"    python run_unified_FAST.py ... --mk-classifier {output_path}")
    print()


if __name__ == "__main__":
    main()
