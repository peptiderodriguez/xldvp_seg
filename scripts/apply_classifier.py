#!/usr/bin/env python3
"""
Apply a trained RF classifier to existing NMJ detections (CPU-only).

Scores every detection in the JSON with rf_prediction, without re-running
detection or feature extraction. Enables a "detect once, classify later"
workflow where 100% detection is done once (expensive GPU), then classifiers
can be trained and applied iteratively (cheap CPU).

Usage:
    python scripts/apply_classifier.py \
        --detections $RUN_DIR/nmj_detections.json \
        --classifier ./checkpoints/nmj_classifier_rf_morph_sam2.pkl \
        --output $RUN_DIR/nmj_detections_scored.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from segmentation.utils.logging import get_logger, setup_logging
from segmentation.utils.json_utils import NumpyEncoder, atomic_json_dump
from segmentation.utils.detection_utils import extract_feature_matrix, load_rf_classifier

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Apply trained RF classifier to existing detections (CPU-only)')
    parser.add_argument('--detections', required=True,
                        help='Path to detections JSON file')
    parser.add_argument('--classifier', required=True,
                        help='Path to RF classifier (.pkl or .joblib)')
    parser.add_argument('--output', default=None,
                        help='Output path (default: <input>_scored.json)')
    args = parser.parse_args()
    setup_logging(level="INFO")

    det_path = Path(args.detections)
    if not det_path.exists():
        logger.error(f"Detections file not found: {det_path}")
        sys.exit(1)

    # Resolve classifier: registry name or file path
    from segmentation.utils.classifier_registry import resolve_classifier
    try:
        clf_path = resolve_classifier(args.classifier)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Default output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = det_path.with_name(det_path.stem + '_scored.json')

    # Load detections
    logger.info(f"Loading detections from {det_path}...")
    with open(det_path) as f:
        detections = json.load(f)
    logger.info(f"Loaded {len(detections):,} detections")

    # Load classifier
    logger.info(f"Loading classifier from {clf_path}...")
    clf_data = load_rf_classifier(str(clf_path))
    pipeline = clf_data['pipeline']
    feature_names = clf_data['feature_names']
    logger.info(f"Classifier has {len(feature_names)} features")

    # Build provenance info from the already-loaded clf_data (avoids double pkl load)
    from segmentation.utils.classifier_registry import build_classifier_info
    classifier_info = build_classifier_info(clf_path, clf_data.get('raw_meta', {}))
    logger.info(f"Classifier: {classifier_info['classifier_name']} "
                f"(features={classifier_info['feature_set']}, "
                f"F1={classifier_info.get('cv_f1', '?')})")

    # Extract feature matrix (pre-allocated numpy array)
    X_valid, valid_indices = extract_feature_matrix(detections, feature_names)

    logger.info(f"Extracted features for {len(valid_indices):,} / {len(detections):,} detections")

    # Check feature name coverage — warn if classifier expects features not in detections
    if valid_indices:
        sample_features = detections[valid_indices[0]].get('features', {})
        present = sum(1 for fn in feature_names if fn in sample_features)
        missing = len(feature_names) - present
        if missing > 0:
            logger.warning(f"Feature mismatch: {present}/{len(feature_names)} classifier features "
                           f"found in detections ({missing} missing, will default to 0)")
            if present < len(feature_names) * 0.5:
                logger.warning(f"  >50% features missing — scores may be unreliable. "
                               f"Was the classifier trained on a different feature set?")

    if not valid_indices:
        logger.warning("No detections have features — setting all rf_prediction = 0.0")
        for det in detections:
            det['rf_prediction'] = 0.0
            det['classifier_info'] = classifier_info
    else:
        X = X_valid

        # Score — handle edge case where classifier has only 1 class
        proba = pipeline.predict_proba(X)
        if proba.shape[1] == 1:
            logger.warning("Classifier has only 1 class — all scores set to 0.0")
            scores = np.zeros(len(X))
        else:
            scores = proba[:, 1]

        # Apply scores to valid detections
        for idx, score in zip(valid_indices, scores):
            detections[idx]['rf_prediction'] = float(score)
            detections[idx]['classifier_info'] = classifier_info

        # Set 0.0 for detections without features
        for i, det in enumerate(detections):
            if 'rf_prediction' not in det or det['rf_prediction'] is None:
                det['rf_prediction'] = 0.0
                det['classifier_info'] = classifier_info

        # Score distribution summary
        all_scores = np.array([det['rf_prediction'] for det in detections])
        logger.info(f"Score distribution:")
        logger.info(f"  min={all_scores.min():.3f}  median={np.median(all_scores):.3f}  "
                     f"max={all_scores.max():.3f}  mean={all_scores.mean():.3f}")
        for thresh in [0.3, 0.5, 0.7, 0.9]:
            n_above = (all_scores >= thresh).sum()
            logger.info(f"  >= {thresh}: {n_above:,} ({100*n_above/len(all_scores):.1f}%)")

    # Save with timestamp
    from segmentation.utils.timestamps import timestamped_path, update_symlink
    ts_out = timestamped_path(out_path)
    logger.info(f"Saving scored detections to {ts_out}...")
    atomic_json_dump(detections, str(ts_out))
    update_symlink(out_path, ts_out)

    # Write sidecar provenance file
    sidecar_path = ts_out.parent / 'classifier_info.json'
    atomic_json_dump(classifier_info, sidecar_path)
    logger.info(f"Classifier provenance written to {sidecar_path}")

    logger.info(f"Done. {len(detections):,} detections scored.")


if __name__ == '__main__':
    main()
