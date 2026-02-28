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

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
from segmentation.utils.logging import get_logger, setup_logging
from segmentation.utils.json_utils import NumpyEncoder
from segmentation.utils.detection_utils import extract_feature_matrix

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

    clf_path = Path(args.classifier)
    if not clf_path.exists():
        logger.error(f"Classifier file not found: {clf_path}")
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
    clf_data = load_nmj_rf_classifier(str(clf_path))
    pipeline = clf_data['pipeline']
    feature_names = clf_data['feature_names']
    logger.info(f"Classifier has {len(feature_names)} features")

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
    else:
        X = X_valid

        # Score
        scores = pipeline.predict_proba(X)[:, 1]

        # Apply scores to valid detections
        for idx, score in zip(valid_indices, scores):
            detections[idx]['rf_prediction'] = float(score)

        # Set 0.0 for detections without features
        for i, det in enumerate(detections):
            if 'rf_prediction' not in det or det['rf_prediction'] is None:
                det['rf_prediction'] = 0.0

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
    with open(ts_out, 'w') as f:
        json.dump(detections, f, cls=NumpyEncoder)
    update_symlink(out_path, ts_out)
    logger.info(f"Done. {len(detections):,} detections scored.")


if __name__ == '__main__':
    main()
