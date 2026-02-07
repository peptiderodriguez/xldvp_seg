#!/usr/bin/env python3
"""
Compare feature subsets for RF classifier training.

Systematically evaluates different feature combinations using stratified
cross-validation, so the user can decide which features to extract for
the full 100% run.

Usage:
    python scripts/compare_feature_sets.py \
        --detections /path/to/nmj_detections.json \
        --annotations /path/to/annotations.json \
        --output-dir /path/to/output

    # Sort by precision instead of F1
    python scripts/compare_feature_sets.py \
        --detections detections.json \
        --annotations annotations.json \
        --sort-by precision
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Feature categories defined by regex patterns.
# Order matters: first match wins for each feature name.
FEATURE_CATEGORIES = [
    ("sam2", re.compile(r"^sam2_\d+$")),
    ("resnet_context", re.compile(r"^resnet_ctx_\d+$")),
    ("resnet_masked", re.compile(r"^resnet_\d+$")),
    ("dinov2_context", re.compile(r"^dinov2_ctx_\d+$")),
    ("dinov2_masked", re.compile(r"^dinov2_\d+$")),
    ("multichannel", re.compile(r"^ch\d+_")),
]

# Combination definitions: (display_name, list of required categories)
# Only tested when ALL constituent categories have at least 1 feature.
FEATURE_COMBOS = [
    ("morphological", ["morphological"]),
    ("morph+sam2", ["morphological", "sam2"]),
    ("morph+multichannel", ["morphological", "multichannel"]),
    ("morph+sam2+multichannel", ["morphological", "sam2", "multichannel"]),
    ("morph+dinov2_context", ["morphological", "dinov2_context"]),
    ("morph+dinov2_combined", ["morphological", "dinov2_masked", "dinov2_context"]),
    ("morph+resnet_combined", ["morphological", "resnet_masked", "resnet_context"]),
    ("all_features", None),  # None = use everything
]


def categorize_features(feature_names):
    """Assign each feature name to a category.

    Returns:
        dict mapping category -> list of feature names
    """
    categories = {cat: [] for cat, _ in FEATURE_CATEGORIES}
    categories["morphological"] = []

    for name in feature_names:
        matched = False
        for cat, pattern in FEATURE_CATEGORIES:
            if pattern.match(name):
                categories[cat].append(name)
                matched = True
                break
        if not matched:
            categories["morphological"].append(name)

    return categories


def load_annotations(annotations_path):
    """Load annotations supporting both formats.

    Format 1: {positive: [...], negative: [...]}
    Format 2: {annotations: {uid: "yes"/"no", ...}}

    Returns:
        (positive_ids: set, negative_ids: set)
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    # Format 1: explicit positive/negative lists
    if "positive" in annotations or "negative" in annotations:
        positive_ids = set(annotations.get("positive", []))
        negative_ids = set(annotations.get("negative", []))
        return positive_ids, negative_ids

    # Format 2: annotations dict with "yes"/"no" values
    if "annotations" in annotations:
        ann_dict = annotations["annotations"]
        positive_ids = {uid for uid, val in ann_dict.items()
                        if str(val).lower() in ("yes", "true", "1", "positive")}
        negative_ids = {uid for uid, val in ann_dict.items()
                        if str(val).lower() in ("no", "false", "0", "negative")}
        return positive_ids, negative_ids

    logger.error("Unrecognized annotation format. Expected 'positive'/'negative' "
                 "lists or 'annotations' dict.")
    sys.exit(1)


def build_detection_lookup(detections):
    """Build lookup dict mapping various ID formats -> detection."""
    det_by_id = {}
    for det in detections:
        tile_origin = det.get("tile_origin", [0, 0])
        tile_x = int(tile_origin[0])
        tile_y = int(tile_origin[1])
        det_id = det.get("id", "")

        # tile_x_tile_y_id format
        full_id = f"{tile_x}_{tile_y}_{det_id}"
        det_by_id[full_id] = det

        # uid fallback
        uid = det.get("uid", "")
        if uid:
            det_by_id[uid] = det

        # bare id fallback
        if det_id:
            det_by_id[det_id] = det

    return det_by_id


def load_features_and_labels(detections_path, annotations_path):
    """Load detections and match with annotations.

    Returns:
        X: ndarray (n_samples, n_features)
        y: ndarray (n_samples,) with 0/1 labels
        feature_names: list of str
        n_positive: int
        n_negative: int
    """
    with open(detections_path) as f:
        detections = json.load(f)
    logger.info(f"Loaded {len(detections)} detections")

    positive_ids, negative_ids = load_annotations(annotations_path)
    logger.info(f"Annotations: {len(positive_ids)} positive, {len(negative_ids)} negative")

    det_by_id = build_detection_lookup(detections)

    # Determine scalar feature names from first detection with features
    feature_names = None
    for det in detections:
        if det.get("features"):
            feature_names = sorted(
                k for k, v in det["features"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
            break

    if not feature_names:
        logger.error("No scalar features found in detections")
        sys.exit(1)

    X = []
    y = []
    matched_pos = 0
    matched_neg = 0

    for sample_id in positive_ids:
        det = det_by_id.get(sample_id)
        if det and det.get("features"):
            X.append([float(det["features"].get(k, 0)) for k in feature_names])
            y.append(1)
            matched_pos += 1

    for sample_id in negative_ids:
        det = det_by_id.get(sample_id)
        if det and det.get("features"):
            X.append([float(det["features"].get(k, 0)) for k in feature_names])
            y.append(0)
            matched_neg += 1

    logger.info(f"Matched: {matched_pos} positive, {matched_neg} negative")

    if matched_pos + matched_neg < 20:
        logger.error(f"Too few matched samples ({matched_pos + matched_neg}). "
                     "Check that annotation IDs match detection IDs.")
        sys.exit(1)

    return (np.array(X, dtype=np.float64), np.array(y, dtype=int),
            feature_names, matched_pos, matched_neg)


def evaluate_feature_set(X, y, feature_indices, n_folds, n_estimators):
    """Run stratified k-fold CV on a feature subset.

    Returns:
        dict with accuracy/precision/recall/f1, each having mean and std
    """
    X_sub = X[:, feature_indices]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_idx, test_idx in skf.split(X_sub, y):
        X_train, X_test = X_sub[train_idx], X_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        tp = int(np.sum((y_pred == 1) & (y_test == 1)))
        fp = int(np.sum((y_pred == 1) & (y_test == 0)))
        fn = int(np.sum((y_pred == 0) & (y_test == 1)))
        tn = int(np.sum((y_pred == 0) & (y_test == 0)))

        acc = (tp + tn) / len(y_test)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)

    return {
        metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for metric, vals in fold_metrics.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare feature subsets for RF classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--detections", required=True,
                        help="Path to detections JSON")
    parser.add_argument("--annotations", required=True,
                        help="Path to annotations JSON")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: alongside detections)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Cross-validation folds (default: 5)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="RF trees (default: 200)")
    parser.add_argument("--sort-by", default="f1",
                        choices=["accuracy", "precision", "recall", "f1"],
                        help="Sort results by this metric (default: f1)")
    args = parser.parse_args()

    setup_logging()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.detections).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, feature_names, n_pos, n_neg = load_features_and_labels(
        args.detections, args.annotations
    )
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    n_samples = len(y)
    logger.info(f"Total samples: {n_samples} ({n_pos} pos, {n_neg} neg)")
    logger.info(f"Total features: {len(feature_names)}")

    # Categorize features
    categories = categorize_features(feature_names)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    logger.info("Feature categories:")
    for cat, names in sorted(categories.items()):
        if names:
            logger.info(f"  {cat}: {len(names)} features")

    # Evaluate each combination
    results = []
    for combo_name, required_cats in FEATURE_COMBOS:
        if required_cats is None:
            # all_features: use everything
            indices = list(range(len(feature_names)))
            combo_feature_names = list(feature_names)
        else:
            # Check all required categories have features
            missing = [c for c in required_cats if not categories.get(c)]
            if missing:
                logger.info(f"Skipping {combo_name}: missing {', '.join(missing)}")
                continue

            combo_feature_names = []
            for cat in required_cats:
                combo_feature_names.extend(categories[cat])
            indices = [name_to_idx[n] for n in combo_feature_names]

        if not indices:
            continue

        n_feat = len(indices)
        logger.info(f"Evaluating {combo_name} ({n_feat} features)...")

        metrics = evaluate_feature_set(X, y, indices, args.n_folds, args.n_estimators)

        results.append({
            "name": combo_name,
            "n_features": n_feat,
            "feature_names": sorted(combo_feature_names),
            **metrics,
        })

    # Sort by chosen metric
    results.sort(key=lambda r: r[args.sort_by]["mean"], reverse=True)

    # Print comparison table
    header = (f"\nFeature Set Comparison ({args.n_folds}-fold CV, "
              f"{args.n_estimators} trees, {n_samples} samples)")
    print(header)
    print("=" * len(header.strip()))
    print(f"{'Feature Set':<28} {'n_feat':>6}  {'Accuracy':>10}  "
          f"{'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print("-" * 88)

    for r in results:
        print(f"{r['name']:<28} {r['n_features']:>6}  "
              f"{r['accuracy']['mean']:>10.3f}  "
              f"{r['precision']['mean']:>10.3f}  "
              f"{r['recall']['mean']:>10.3f}  "
              f"{r['f1']['mean']:>10.3f}")

    print()

    # Save JSON
    output_data = {
        "metadata": {
            "detections_file": str(Path(args.detections).resolve()),
            "annotations_file": str(Path(args.annotations).resolve()),
            "n_samples": n_samples,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_folds": args.n_folds,
            "n_estimators": args.n_estimators,
            "sort_by": args.sort_by,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    output_path = output_dir / "feature_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
