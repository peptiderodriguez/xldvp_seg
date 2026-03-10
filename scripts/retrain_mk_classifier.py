#!/usr/bin/env python3
"""Retrain MK classifier with unified training data from all 16 slides.

Merges original training data (13 slides, 1725 samples) with rescued slide
annotations (FGC2/FGC4/MHU4), optionally incorporates re-extracted SAM2
embeddings, trains a new RF classifier, and re-scores all detections.

Usage:
    # Step 1: Build unified training set + train classifier
    python scripts/retrain_mk_classifier.py train \
        --original-training mk_training_data_2026-02-11.json \
        --rescued-base-dir mk_clf084_dataset \
        --rescued-annotations mk_annotations_2026-03-06_rejected3_unnorm_100pct.json \
        --sam2-embeddings sam2_embeddings_all.json \
        --output-dir retrained_classifier/

    # Step 2: Re-score all detections
    python scripts/retrain_mk_classifier.py score \
        --classifier retrained_classifier/mk_classifier_YYYY-MM-DD.pkl \
        --full-detections all_mks_with_rejected3_full.json \
        --output all_mks_rescored.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

RESCUED_SLIDES = ["2025_11_18_FGC2", "2025_11_18_FGC4", "2025_11_18_MHU4"]
PIXEL_SIZE_UM = 0.1725


def load_original_training(path):
    """Load original training data from mk_training_data JSON."""
    with open(path) as f:
        data = json.load(f)
    samples = data["training_samples"]
    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = sum(1 for s in samples if s["label"] == 0)
    print(f"Original training: {len(samples)} samples ({n_pos} pos, {n_neg} neg)")
    return samples


def build_rescued_samples(base_dir, annotations_path, full_detections_path):
    """Build training samples from rescued slide tile features + annotations.

    Annotation format: {"positive": [], "negative": [uid_list]}
    Cells NOT in negative list are implicit positives (hand-curated by user).

    IMPORTANT: Tile features contain all pre-dedup candidates (thousands per slide).
    Only cells present in the full detections JSON were reviewed by the user.
    We filter to those UIDs to avoid treating unannotated candidates as positives.
    """
    base_dir = Path(base_dir)

    # Load full detections to get valid UIDs for rescued slides
    print(f"Loading full detections to get valid UIDs...")
    with open(full_detections_path) as f:
        full_dets = json.load(f)
    valid_uids = set()
    for det in full_dets:
        slide = det.get("slide", "")
        if any(rs in slide for rs in ["FGC2", "FGC4", "MHU4"]):
            valid_uids.add(det.get("uid", ""))
    print(f"  {len(valid_uids)} valid UIDs from rescued slides")

    with open(annotations_path) as f:
        annotations = json.load(f)

    # Parse negative UIDs — handle multiple possible formats
    neg_uids = set()
    if "mk" in annotations and isinstance(annotations["mk"], dict):
        neg_uids.update(annotations["mk"].get("negative", []))
    else:
        neg_uids.update(annotations.get("negative", []))

    print(f"Rescued annotations: {len(neg_uids)} negatives")

    samples = []
    for slide in RESCUED_SLIDES:
        tile_base = base_dir / slide / "mk" / "tiles"
        if not tile_base.exists():
            print(f"  WARNING: {tile_base} not found, skipping")
            continue

        slide_pos = 0
        slide_neg = 0
        tile_dirs = sorted([d for d in tile_base.iterdir() if d.is_dir()])

        for tile_dir in tile_dirs:
            feat_file = tile_dir / "features.json"
            if not feat_file.exists():
                continue

            with open(feat_file) as f:
                tile_feats = json.load(f)

            for det in tile_feats:
                uid = det.get("uid", "")

                # Include if: in final detection set (implicit positive)
                # or in negative annotation list (explicit negative)
                if uid in neg_uids:
                    label = 0
                elif uid in valid_uids:
                    label = 1
                else:
                    continue  # pre-dedup candidate, not reviewed

                features = det.get("features", {})

                area_px = features.get("area", 0)
                sample = {
                    "slide": slide,
                    "tile_id": tile_dir.name,
                    "det_id": det.get("id", ""),
                    "features": features,
                    "area_px": area_px,
                    "area_um2": area_px * (PIXEL_SIZE_UM ** 2),
                    "uid": uid,
                    "label": label,
                    "cell_type": "mk",
                }
                samples.append(sample)

                if label == 1:
                    slide_pos += 1
                else:
                    slide_neg += 1

        print(f"  {slide}: {slide_pos} pos, {slide_neg} neg")

    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = sum(1 for s in samples if s["label"] == 0)
    print(f"Rescued total: {len(samples)} samples ({n_pos} pos, {n_neg} neg)")
    return samples


def merge_sam2(samples, embeddings_path):
    """Merge SAM2 embeddings into training samples by UID."""
    with open(embeddings_path) as f:
        embeddings = json.load(f)
    print(f"SAM2 embeddings: {len(embeddings)} UIDs")

    n_updated = 0
    n_missing = 0
    for sample in samples:
        uid = sample["uid"]
        if uid in embeddings:
            emb = embeddings[uid]
            for i, v in enumerate(emb):
                sample["features"][f"sam2_{i}"] = v
            n_updated += 1
        else:
            # Ensure sam2 keys exist (as zeros if missing)
            for i in range(256):
                sample["features"].setdefault(f"sam2_{i}", 0.0)
            n_missing += 1

    print(f"  Updated: {n_updated}, missing: {n_missing}")
    if n_missing > 0:
        print(f"  WARNING: {n_missing} samples still have zero SAM2 embeddings")


def train_classifier(samples, output_dir, n_estimators=500, feature_set="all",
                     exclude_color=False):
    """Train RF classifier on unified training data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        train_test_split,
    )
    from sklearn.metrics import classification_report, confusion_matrix

    # Color/intensity features derived from RGB crop rendering — not biologically meaningful
    COLOR_FEATURES = {
        "red_mean", "red_std", "green_mean", "green_std", "blue_mean", "blue_std",
        "hue_mean", "saturation_mean", "value_mean", "gray_mean", "gray_std",
        "relative_brightness", "intensity_variance", "dark_fraction",
        "dark_region_fraction",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all feature names across samples
    all_feature_names = set()
    for s in samples:
        all_feature_names.update(s["features"].keys())

    # Filter by feature set
    if feature_set == "morph":
        feature_names = sorted(
            f for f in all_feature_names
            if not f.startswith("sam2_")
            and not f.startswith("resnet_")
            and not f.startswith("dinov2_")
        )
    elif feature_set == "morph_sam2":
        feature_names = sorted(
            f for f in all_feature_names
            if not f.startswith("resnet_")
            and not f.startswith("dinov2_")
        )
    else:  # "all"
        feature_names = sorted(all_feature_names)

    if exclude_color:
        before = len(feature_names)
        feature_names = [f for f in feature_names if f not in COLOR_FEATURES]
        print(f"Excluded {before - len(feature_names)} color features")

    print(f"\nFeature set '{feature_set}': {len(feature_names)} features")

    # Build X, y
    X = np.zeros((len(samples), len(feature_names)))
    y = np.zeros(len(samples), dtype=int)

    for i, s in enumerate(samples):
        for j, fname in enumerate(feature_names):
            v = s["features"].get(fname, 0.0)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                v = 0.0
            X[i, j] = float(v)
        y[i] = s["label"]

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"Training set: {len(samples)} samples ({n_pos} pos, {n_neg} neg)")

    # SAM2 status check
    sam2_cols = [j for j, f in enumerate(feature_names) if f.startswith("sam2_")]
    if sam2_cols:
        sam2_nonzero = np.count_nonzero(X[:, sam2_cols])
        sam2_total = len(sam2_cols) * len(samples)
        pct = 100 * sam2_nonzero / sam2_total if sam2_total > 0 else 0
        print(f"SAM2 features: {len(sam2_cols)} dims, {pct:.1f}% non-zero values")
        if pct < 1:
            print(
                "  WARNING: SAM2 features are mostly zeros — "
                "consider running SAM2 extraction first"
            )

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(
        f"\nTrain: {len(X_train)} ({int(y_train.sum())} pos), "
        f"Test: {len(X_test)} ({int(y_test.sum())} pos)"
    )

    # 5-fold CV on training set
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
    print(f"5-fold CV F1: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

    cv_acc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"5-fold CV Acc: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")

    # Evaluate on held-out test set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = float((y_pred == y_test).mean())
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Retrain on ALL data
    print(f"\nRetraining on all {len(X)} samples...")
    clf_final = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf_final.fit(X, y)

    # Feature importance (top 20)
    importances = clf_final.feature_importances_
    top_idx = np.argsort(importances)[::-1][:20]
    print("\nTop 20 features by importance:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

    # Save classifier
    import pickle

    timestamp = datetime.now().strftime("%Y-%m-%d")
    pkl_path = output_dir / f"mk_classifier_{timestamp}.pkl"

    model_data = {
        "classifier": clf_final,
        "feature_names": feature_names,
        "n_samples": len(samples),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": test_acc,
        "cv_accuracy": float(cv_acc.mean()),
        "cv_f1": float(cv_f1.mean()),
        "cell_type": "mk",
        "feature_set": feature_set,
        "trained_at": timestamp,
        "slides": sorted(set(s["slide"] for s in samples)),
        "n_slides": len(set(s["slide"] for s in samples)),
    }

    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nSaved classifier to {pkl_path}")

    # Save training data JSON
    training_data = {
        "X": [s["features"] for s in samples],
        "y": [s["label"] for s in samples],
        "feature_names": feature_names,
        "sample_ids": [s["uid"] for s in samples],
        "training_samples": samples,
        "source": {
            "original_training": len([
                s for s in samples
                if not any(rs in s["slide"] for rs in ["FGC2", "FGC4", "MHU4"])
            ]),
            "rescued_training": len([
                s for s in samples
                if any(rs in s["slide"] for rs in ["FGC2", "FGC4", "MHU4"])
            ]),
            "timestamp": timestamp,
        },
    }
    td_path = output_dir / f"mk_training_data_{timestamp}.json"
    with open(td_path, "w") as f:
        json.dump(training_data, f)
    print(f"Saved training data to {td_path}")

    return clf_final, feature_names


def score_detections(classifier_path, detections_path, output_path=None):
    """Re-score all detections with trained classifier."""
    import pickle

    print(f"Loading classifier from {classifier_path}...")
    with open(classifier_path, "rb") as f:
        model_data = pickle.load(f)

    clf = model_data["classifier"]
    feature_names = model_data["feature_names"]
    print(
        f"  {len(feature_names)} features, "
        f"trained on {model_data.get('n_samples', '?')} samples"
    )

    print(f"Loading detections from {detections_path}...")
    with open(detections_path) as f:
        detections = json.load(f)
    print(f"  {len(detections)} detections")

    # Build feature matrix
    X = np.zeros((len(detections), len(feature_names)))
    for i, det in enumerate(detections):
        feat_key = "features" if "features" in det else "features_morph_color"
        feats = det.get(feat_key, {})
        for j, fname in enumerate(feature_names):
            v = feats.get(fname, 0.0)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                v = 0.0
            X[i, j] = float(v)

    # Score
    probs = clf.predict_proba(X)
    pos_idx = list(clf.classes_).index(1)
    scores = probs[:, pos_idx]

    # Update detections
    for i, det in enumerate(detections):
        old_score = det.get("mk_score")
        det["mk_score"] = float(scores[i])
        if old_score is not None:
            det["mk_score_old"] = old_score

    # Summary
    print(f"\nScore distribution:")
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        n_above = int((scores >= threshold).sum())
        print(f"  >= {threshold:.2f}: {n_above} ({100 * n_above / len(scores):.1f}%)")

    # Per-slide summary at 0.80
    slides = sorted(set(d["slide"] for d in detections))
    print(f"\nPer-slide (at >= 0.80):")
    for slide in slides:
        slide_scores = [d["mk_score"] for d in detections if d["slide"] == slide]
        n_above = sum(1 for s in slide_scores if s >= 0.80)
        pct = 100 * n_above / len(slide_scores) if slide_scores else 0
        print(f"  {slide}: {n_above}/{len(slide_scores)} ({pct:.1f}%)")

    # Save
    if output_path is None:
        output_path = str(detections_path).replace(".json", "_rescored.json")
    with open(output_path, "w") as f:
        json.dump(detections, f)
    print(f"\nSaved {len(detections)} re-scored detections to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Train command
    tr = sub.add_parser(
        "train", help="Build unified training set and train classifier"
    )
    tr.add_argument(
        "--original-training", type=Path, required=True,
        help="Original training data JSON (mk_training_data_2026-02-11.json)",
    )
    tr.add_argument(
        "--rescued-base-dir", type=Path, required=True,
        help="Base dir with rescued slide tile features ({slide}/mk/tiles/)",
    )
    tr.add_argument(
        "--rescued-annotations", type=Path, required=True,
        help="Rescued slide annotations JSON",
    )
    tr.add_argument(
        "--full-detections", type=Path, required=True,
        help="Full detection JSON (all_mks_with_rejected3_full.json) — "
             "used to filter rescued tile features to final detections only",
    )
    tr.add_argument(
        "--sam2-embeddings", type=Path, nargs="+", default=None,
        help="SAM2 embeddings JSON(s) from extract_sam2_embeddings.py "
             "(multiple files will be merged)",
    )
    tr.add_argument(
        "--output-dir", type=Path, default=Path("retrained_classifier"),
        help="Output directory for classifier + training data",
    )
    tr.add_argument(
        "--n-estimators", type=int, default=500,
        help="Number of RF trees (default: 500)",
    )
    tr.add_argument(
        "--feature-set", choices=["all", "morph", "morph_sam2"], default="all",
        help="Feature subset to use (default: all)",
    )
    tr.add_argument(
        "--exclude-color", action="store_true",
        help="Exclude RGB/HSV crop color features (red_mean, hue_mean, etc.)",
    )

    # Score command
    sc = sub.add_parser(
        "score", help="Re-score all detections with trained classifier"
    )
    sc.add_argument(
        "--classifier", type=Path, required=True,
        help="Trained classifier PKL",
    )
    sc.add_argument(
        "--full-detections", type=Path, required=True,
        help="Full detection JSON to re-score",
    )
    sc.add_argument(
        "--output", type=Path, default=None,
        help="Output path (default: *_rescored.json)",
    )

    args = parser.parse_args()

    if args.command == "train":
        # Load original training data
        original = load_original_training(args.original_training)

        # Build rescued samples from tile features + annotations
        rescued = build_rescued_samples(
            args.rescued_base_dir, args.rescued_annotations,
            args.full_detections,
        )

        # Combine
        all_samples = original + rescued

        # Deduplicate by UID
        seen = set()
        deduped = []
        for s in all_samples:
            if s["uid"] not in seen:
                seen.add(s["uid"])
                deduped.append(s)
        if len(deduped) < len(all_samples):
            print(
                f"\nDeduplicated: {len(all_samples)} -> {len(deduped)} samples"
            )
        all_samples = deduped

        # Merge SAM2 embeddings if provided
        if args.sam2_embeddings:
            # Support multiple embedding files (e.g., original13 + rescued3)
            merged_emb = {}
            for emb_path in args.sam2_embeddings:
                print(f"\nLoading SAM2 from {emb_path}...")
                with open(emb_path) as f:
                    emb = json.load(f)
                merged_emb.update(emb)
                print(f"  {len(emb)} UIDs (total: {len(merged_emb)})")

            # Write merged to temp file for merge_sam2()
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(merged_emb, tmp)
                tmp_path = tmp.name
            merge_sam2(all_samples, tmp_path)
            Path(tmp_path).unlink()

        # Train
        train_classifier(
            all_samples,
            args.output_dir,
            n_estimators=args.n_estimators,
            feature_set=args.feature_set,
            exclude_color=args.exclude_color,
        )

    elif args.command == "score":
        score_detections(
            str(args.classifier),
            str(args.full_detections),
            output_path=str(args.output) if args.output else None,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
