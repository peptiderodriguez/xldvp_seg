#!/usr/bin/env python3
"""Train a generic cell-quality classifier from multiple annotated slides.

Tests whether a morph-only RF trained on N-1 slides generalizes to the held-out
slide. If it does, you never need to annotate again for PM+nuc cell detection.

Usage:
    # After annotating 3 slides:
    python scripts/train_generic_classifier.py \
        --inputs slide1_det.json:slide1_annot.json \
                 slide2_det.json:slide2_annot.json \
                 slide3_det.json:slide3_annot.json \
        --output-dir generic_classifier/

    # Test an existing classifier on a new slide (no annotation needed):
    python scripts/train_generic_classifier.py \
        --inputs new_slide_det.json \
        --classifier generic_classifier/generic_morph_classifier.pkl \
        --output-dir new_slide_scored/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
from collections import Counter

from segmentation.utils.json_utils import fast_json_load, atomic_json_dump
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

# Morph features that are intensity-independent and should generalize across slides
MORPH_FEATURES = [
    "area", "perimeter", "solidity", "eccentricity", "elongation",
    "major_axis_length", "minor_axis_length", "circularity",
    "equivalent_diameter", "extent", "convex_area", "filled_area",
    "euler_number", "orientation", "hu_moment_0", "hu_moment_1",
    "hu_moment_2", "hu_moment_3", "hu_moment_4", "hu_moment_5",
    "hu_moment_6",
]


def load_slide(det_path, annot_path=None):
    """Load detections + optional annotations, return (detections, labels_dict)."""
    dets = fast_json_load(str(det_path))
    if isinstance(dets, dict):
        dets = dets.get("detections", [])

    labels = {}
    if annot_path and Path(annot_path).exists():
        annot = fast_json_load(str(annot_path))
        if isinstance(annot, dict):
            # Format: {uid: label} or {"annotations": {uid: label}}
            labels = annot.get("annotations", annot)
            # Normalize: 1 = positive (real cell), 0 = negative (false positive)
            normalized = {}
            for uid, val in labels.items():
                if isinstance(val, (int, float)):
                    normalized[uid] = int(val)
                elif isinstance(val, str):
                    normalized[uid] = 1 if val.lower() in ("1", "yes", "positive", "true") else 0
                elif isinstance(val, dict):
                    # {uid: {label: 1}} format from HTML export
                    normalized[uid] = int(val.get("label", val.get("value", 0)))
            labels = normalized

    return dets, labels


def extract_morph_vector(det):
    """Extract morph feature vector from a detection. Returns (vector, valid)."""
    features = det.get("features", {})
    vec = []
    for f in MORPH_FEATURES:
        val = features.get(f)
        if val is None:
            return None, False
        try:
            vec.append(float(val))
        except (TypeError, ValueError):
            return None, False
    return np.array(vec), True


def build_dataset(slides):
    """Build X, y, slide_ids from list of (detections, labels, slide_name)."""
    X_all, y_all, slide_ids = [], [], []

    for dets, labels, slide_name in slides:
        if not labels:
            continue
        uid_to_det = {d.get("uid", ""): d for d in dets}
        for uid, label in labels.items():
            det = uid_to_det.get(uid)
            if det is None:
                continue
            vec, valid = extract_morph_vector(det)
            if valid:
                X_all.append(vec)
                y_all.append(label)
                slide_ids.append(slide_name)

    if not X_all:
        return None, None, None

    return np.array(X_all), np.array(y_all), slide_ids


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train RF on train set, evaluate on test set. Returns (clf, metrics)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, {
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_pos_rate": round(y_train.mean(), 4),
        "test_pos_rate": round(y_test.mean(), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train generic cell-quality classifier from multiple slides",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Detection:annotation pairs (colon-separated). "
             "E.g., det1.json:annot1.json det2.json:annot2.json. "
             "Annotation can be omitted for score-only mode.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--classifier", type=Path, default=None,
        help="Existing classifier to apply (skip training, just score)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse inputs ---
    slides = []
    for inp in args.inputs:
        parts = inp.split(":")
        det_path = parts[0]
        annot_path = parts[1] if len(parts) > 1 else None
        slide_name = Path(det_path).stem
        logger.info(f"Loading {slide_name}...")
        dets, labels = load_slide(det_path, annot_path)
        logger.info(f"  {len(dets):,} detections, {len(labels):,} annotations")
        slides.append((dets, labels, slide_name))

    # --- Score-only mode ---
    if args.classifier:
        import joblib
        clf = joblib.load(str(args.classifier))
        logger.info(f"Loaded classifier: {args.classifier}")
        for dets, _, slide_name in slides:
            n_scored = 0
            for det in dets:
                vec, valid = extract_morph_vector(det)
                if valid:
                    score = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])
                    det["rf_prediction"] = round(score, 4)
                    det.get("features", {})["rf_prediction"] = round(score, 4)
                    n_scored += 1
            out_path = args.output_dir / f"{slide_name}_scored.json"
            atomic_json_dump(dets, str(out_path))
            logger.info(f"  Scored {n_scored:,}/{len(dets):,} detections -> {out_path}")
        return

    # --- Build combined dataset ---
    X, y, slide_ids = build_dataset(slides)
    if X is None:
        logger.error("No annotated detections found")
        sys.exit(1)

    slide_names = sorted(set(slide_ids))
    logger.info(f"Combined dataset: {len(X):,} cells, {int(y.sum()):,} positive, "
                f"{len(y) - int(y.sum()):,} negative, {len(slide_names)} slides")

    # --- Leave-one-slide-out cross-validation ---
    logger.info("\n=== Leave-One-Slide-Out Cross-Validation ===")
    slide_ids_arr = np.array(slide_ids)
    loo_results = []

    for held_out in slide_names:
        train_mask = slide_ids_arr != held_out
        test_mask = slide_ids_arr == held_out

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        _, metrics = train_and_evaluate(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
        metrics["held_out"] = held_out
        loo_results.append(metrics)

        logger.info(
            f"  Hold out {held_out}: F1={metrics['f1']:.3f} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"(train={metrics['n_train']}, test={metrics['n_test']})"
        )

    if loo_results:
        mean_f1 = np.mean([r["f1"] for r in loo_results])
        logger.info(f"\n  Mean LOO F1: {mean_f1:.3f}")

        if mean_f1 >= 0.85:
            logger.info("  -> Generic classifier GENERALIZES well. Training on all data.")
        else:
            logger.info("  -> Generalization is weak. Per-slide classifiers recommended.")

    # --- Train on all data ---
    logger.info("\n=== Training on all slides combined ===")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # 5-fold CV on combined data
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
    logger.info(f"  5-fold CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Fit final model
    clf.fit(X, y)

    # Save
    import joblib
    clf_path = args.output_dir / "generic_morph_classifier.pkl"
    joblib.dump(clf, str(clf_path))
    logger.info(f"  Saved: {clf_path}")

    # Feature importance
    importances = sorted(
        zip(MORPH_FEATURES, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    logger.info("  Top features:")
    for feat, imp in importances[:10]:
        logger.info(f"    {feat}: {imp:.4f}")

    # Save summary
    summary = {
        "n_slides": len(slide_names),
        "slides": slide_names,
        "n_total": len(X),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "feature_set": "morph",
        "features": MORPH_FEATURES,
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std": round(float(cv_scores.std()), 4),
        "loo_results": loo_results,
        "loo_mean_f1": round(float(mean_f1), 4) if loo_results else None,
        "feature_importance": {f: round(float(i), 4) for f, i in importances},
    }
    atomic_json_dump(summary, str(args.output_dir / "generic_classifier_summary.json"))
    logger.info(f"\nDone. Classifier: {clf_path}")
    logger.info(f"Apply to new slides with: python scripts/train_generic_classifier.py "
                f"--inputs new_det.json --classifier {clf_path} --output-dir scored/")


if __name__ == "__main__":
    main()
