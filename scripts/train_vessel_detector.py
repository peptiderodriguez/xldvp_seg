#!/usr/bin/env python3
"""
Train vessel detector (vessel vs non-vessel binary classifier).

This trains a Random Forest classifier to distinguish true vessels
from false positives (artifacts, noise, other structures).

This is DIFFERENT from the vessel type classifier (capillary/arteriole/artery).
This classifier answers: "Is this candidate actually a vessel?"

Input:
    - Annotations JSON with yes/no labels (from HTML review interface)
    - Detections JSON with features from candidate detection

Output:
    - Trained model file (.joblib)
    - Feature importance JSON
    - Training metrics JSON with precision/recall/F1
    - Visualizations (confusion matrix, feature importance)

Usage:
    python scripts/train_vessel_detector.py \\
        --annotations /path/to/annotations.json \\
        --detections /path/to/vessel_detections.json \\
        --output-dir /path/to/output

Annotation format (from HTML review):
    {
        "vessel_uid_1": "yes",
        "vessel_uid_2": "no",
        "vessel_uid_3": "yes",
        ...
    }

    Or nested format:
    {
        "annotations": {
            "vessel_uid_1": "yes",
            ...
        }
    }
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.classification.vessel_detector_rf import (
    VesselDetectorRF,
    VESSEL_DETECTION_FEATURES,
    MINIMAL_DETECTION_FEATURES,
)

# =============================================================================
# SIZE CLASS DEFINITIONS FOR STRATIFIED SAMPLING
# =============================================================================

# Size class thresholds (in microns)
SIZE_CLASS_THRESHOLDS = {
    0: (0, 50),       # small: <50 um
    1: (50, 200),     # medium: 50-200 um
    2: (200, float('inf'))  # large: >200 um
}

SIZE_CLASS_NAMES = {
    0: 'small',
    1: 'medium',
    2: 'large'
}


def get_size_class(diameter_um: float) -> int:
    """Get size class for a vessel based on outer diameter."""
    if diameter_um < 50:
        return 0  # small
    elif diameter_um < 200:
        return 1  # medium
    else:
        return 2  # large


def extract_diameter(detection: Dict) -> float:
    """Extract diameter from a detection dict."""
    features = detection.get('features', detection)

    # Try different possible field names for diameter
    diameter = features.get('outer_diameter_um')
    if diameter is None:
        diameter = features.get('diameter_um')
    if diameter is None:
        # Fall back to estimating from area
        area = features.get('outer_area_um2') or features.get('area')
        if area is not None and area > 0:
            diameter = 2 * np.sqrt(area / np.pi)

    return diameter if diameter is not None else np.nan


def analyze_size_distribution(
    diameters: np.ndarray,
    labels: np.ndarray,
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Analyze the size distribution of vessels.

    Args:
        diameters: Array of vessel diameters in microns
        labels: Array of labels (0=negative, 1=positive)
        prefix: Optional prefix for log messages

    Returns:
        Dict with distribution statistics and warnings
    """
    valid_mask = ~np.isnan(diameters)
    valid_diameters = diameters[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_diameters) == 0:
        return {'warning': 'No valid diameter measurements found'}

    # Get size classes
    size_classes = np.array([get_size_class(d) for d in valid_diameters])

    # Count per class
    from collections import Counter
    class_counts = Counter(size_classes)
    total = len(size_classes)

    # Calculate percentages
    distribution = {}
    warnings = []

    logger.info(f"\n{prefix}Size Distribution Analysis:")
    logger.info("-" * 50)

    for class_id in range(3):
        count = class_counts.get(class_id, 0)
        pct = 100 * count / total if total > 0 else 0
        class_name = SIZE_CLASS_NAMES[class_id]
        threshold = SIZE_CLASS_THRESHOLDS[class_id]

        distribution[class_name] = {
            'count': count,
            'percentage': pct,
            'threshold_um': threshold,
        }

        # Count positive/negative within class
        class_mask = size_classes == class_id
        if class_mask.sum() > 0:
            pos_in_class = valid_labels[class_mask].sum()
            neg_in_class = class_mask.sum() - pos_in_class
            pos_pct = 100 * pos_in_class / class_mask.sum()
        else:
            pos_in_class = neg_in_class = pos_pct = 0

        logger.info(f"  {class_name:10s} ({threshold[0]}-{threshold[1]} um): {count:5d} ({pct:5.1f}%) "
                   f"[pos: {int(pos_in_class)}, neg: {int(neg_in_class)}, pos%: {pos_pct:.1f}%]")

        # Generate warnings for imbalanced distribution
        if pct < 10 and total > 50:
            warnings.append(f"UNDERREPRESENTED: {class_name} has only {pct:.1f}% of samples")
        if pct > 70:
            warnings.append(f"OVERREPRESENTED: {class_name} has {pct:.1f}% of samples")

    # Overall statistics
    logger.info(f"\n  Total valid samples: {total}")
    logger.info(f"  Diameter range: {valid_diameters.min():.1f} - {valid_diameters.max():.1f} um")
    logger.info(f"  Median diameter: {np.median(valid_diameters):.1f} um")

    # Log warnings
    if warnings:
        logger.warning("\n  SIZE DISTRIBUTION WARNINGS:")
        for w in warnings:
            logger.warning(f"    - {w}")

    return {
        'distribution': distribution,
        'warnings': warnings,
        'total': total,
        'diameter_range': (float(valid_diameters.min()), float(valid_diameters.max())),
        'median_diameter': float(np.median(valid_diameters)),
    }


def stratified_sample_by_size(
    X_list: List[Dict],
    y_list: List[int],
    uids: List[str],
    detections: Dict[str, Dict],
    samples_per_class: int = None,
    min_samples_per_class: int = 10,
    random_seed: int = 42,
) -> tuple:
    """
    Perform stratified sampling to balance vessel size classes.

    This ensures the training set has equal representation across size classes,
    preventing the classifier from being biased toward the most common size.

    Args:
        X_list: List of feature dictionaries
        y_list: List of labels (0 or 1)
        uids: List of UIDs
        detections: Detection dictionary for diameter lookup
        samples_per_class: Target samples per class (None = use min class count)
        min_samples_per_class: Minimum samples to require per class
        random_seed: Random seed

    Returns:
        Tuple of (X_balanced, y_balanced, uids_balanced)
    """
    np.random.seed(random_seed)

    # Extract diameters
    diameters = np.array([extract_diameter(detections.get(uid, {})) for uid in uids])
    valid_mask = ~np.isnan(diameters)

    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        logger.warning(f"  {n_invalid} samples have missing diameter - will be excluded")

    # Filter to valid samples
    X_valid = [X_list[i] for i in range(len(X_list)) if valid_mask[i]]
    y_valid = np.array([y_list[i] for i in range(len(y_list)) if valid_mask[i]])
    uids_valid = [uids[i] for i in range(len(uids)) if valid_mask[i]]
    diameters_valid = diameters[valid_mask]

    # Get size classes
    size_classes = np.array([get_size_class(d) for d in diameters_valid])

    # Count per class (separately for positive and negative)
    class_pos_counts = {}
    class_neg_counts = {}
    for class_id in range(3):
        class_mask = size_classes == class_id
        if class_mask.sum() > 0:
            class_pos_counts[class_id] = int((y_valid[class_mask] == 1).sum())
            class_neg_counts[class_id] = int((y_valid[class_mask] == 0).sum())
        else:
            class_pos_counts[class_id] = 0
            class_neg_counts[class_id] = 0

    # Determine samples per class
    if samples_per_class is None:
        # Use minimum of (smallest positive class, smallest negative class)
        min_pos = min(class_pos_counts.values()) if class_pos_counts else 0
        min_neg = min(class_neg_counts.values()) if class_neg_counts else 0
        samples_per_class = min(min_pos, min_neg)

    if samples_per_class < min_samples_per_class:
        logger.warning(f"  Insufficient samples for balanced stratification. "
                      f"Min class has {samples_per_class} samples, need {min_samples_per_class}")
        logger.warning("  Using class weights instead of undersampling")
        return X_valid, y_valid.tolist(), uids_valid

    logger.info(f"\n  Stratified sampling: {samples_per_class} samples per class per label")

    # Sample from each class
    selected_indices = []

    for class_id in range(3):
        class_mask = size_classes == class_id
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            logger.warning(f"    {SIZE_CLASS_NAMES[class_id]}: NO SAMPLES")
            continue

        # Sample positive and negative separately within each class
        pos_indices = class_indices[y_valid[class_indices] == 1]
        neg_indices = class_indices[y_valid[class_indices] == 0]

        # Sample equal numbers from positive and negative
        n_pos = min(len(pos_indices), samples_per_class)
        n_neg = min(len(neg_indices), samples_per_class)

        if n_pos > 0:
            sampled_pos = np.random.choice(pos_indices, n_pos, replace=False)
            selected_indices.extend(sampled_pos.tolist())

        if n_neg > 0:
            sampled_neg = np.random.choice(neg_indices, n_neg, replace=False)
            selected_indices.extend(sampled_neg.tolist())

        logger.info(f"    {SIZE_CLASS_NAMES[class_id]:10s}: {n_pos} pos, {n_neg} neg "
                   f"(of {len(pos_indices)} pos, {len(neg_indices)} neg available)")

    # Create balanced dataset
    X_balanced = [X_valid[i] for i in selected_indices]
    y_balanced = [int(y_valid[i]) for i in selected_indices]
    uids_balanced = [uids_valid[i] for i in selected_indices]

    logger.info(f"\n  Balanced dataset: {len(X_balanced)} samples "
               f"({sum(1 for y in y_balanced if y == 1)} pos, {sum(1 for y in y_balanced if y == 0)} neg)")

    return X_balanced, y_balanced, uids_balanced

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_annotations(annotations_path: Path) -> Dict[str, str]:
    """
    Load vessel annotations from JSON file.

    Supports multiple formats:
    1. Direct mapping: {"uid": "yes"/"no", ...}
    2. Nested: {"annotations": {"uid": "yes"/"no", ...}}

    Args:
        annotations_path: Path to annotations JSON

    Returns:
        Dictionary mapping vessel UID to label (yes/no)
    """
    with open(annotations_path) as f:
        data = json.load(f)

    # Handle nested format
    if 'annotations' in data:
        return data['annotations']

    # Direct mapping
    return data


def load_detections(detections_path: Path) -> Dict[str, Dict]:
    """
    Load vessel detections with features from JSON.

    Args:
        detections_path: Path to detections JSON

    Returns:
        Dictionary mapping UIDs to detection dicts
    """
    with open(detections_path) as f:
        data = json.load(f)

    indexed = {}
    for d in data:
        # Primary: uid
        if 'uid' in d:
            indexed[d['uid']] = d

        # Alternative formats for matching
        if 'tile_origin' in d and 'id' in d:
            tile_x, tile_y = d['tile_origin']
            alt_id = f"{tile_x}_{tile_y}_{d['id']}"
            indexed[alt_id] = d

            if 'slide_name' in d:
                alt_id2 = f"{d['slide_name']}_{tile_x}_{tile_y}_{d['id']}"
                indexed[alt_id2] = d

        if 'id' in d:
            indexed[d['id']] = d

    return indexed


def extract_training_data(
    detections: Dict[str, Dict],
    annotations: Dict[str, str],
    feature_names: List[str],
) -> tuple:
    """
    Extract feature matrix and labels from annotated vessels.

    Args:
        detections: Detection dictionary indexed by UID
        annotations: Mapping of UID to yes/no label
        feature_names: List of feature names to extract

    Returns:
        Tuple of (X_features, y_labels, valid_uids)
    """
    X_list = []
    y_list = []
    valid_uids = []

    for uid, label in annotations.items():
        if uid not in detections:
            logger.debug(f"UID not found in detections: {uid}")
            continue

        det = detections[uid]
        features = det.get('features', {})

        # If features are at top level (not nested)
        if not features and any(key in det for key in feature_names[:5]):
            features = det

        X_list.append(features)
        y_list.append(label)
        valid_uids.append(uid)

    if not X_list:
        raise ValueError("No matching samples found! Check annotation UIDs.")

    return X_list, y_list, valid_uids


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = 'Vessel Detection Confusion Matrix'
) -> None:
    """Plot and save confusion matrix visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))

        # Normalize for color mapping
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=['Non-Vessel', 'Vessel'],
            yticklabels=['Non-Vessel', 'Vessel'],
            ax=ax,
            cbar_kws={'label': 'Proportion'}
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Confusion matrix saved to: {output_path}")

    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping confusion matrix plot")


def plot_feature_importance(
    importance: Dict[str, float],
    output_path: Path,
    top_n: int = 20,
    title: str = 'Feature Importance for Vessel Detection'
) -> None:
    """Plot and save feature importance visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Sort and get top N
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(names))
        ax.barh(y_pos, scores, align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Feature importance plot saved to: {output_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping feature importance plot")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path
) -> None:
    """Plot precision-recall curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, 'b-', linewidth=2, label=f'AP = {avg_precision:.3f}')
        ax.fill_between(recall, precision, alpha=0.3)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Precision-recall curve saved to: {output_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping PR curve plot")


def main():
    parser = argparse.ArgumentParser(
        description='Train vessel detector (vessel vs non-vessel classifier)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_vessel_detector.py \\
        --annotations annotations.json \\
        --detections vessel_detections.json

    # With custom parameters
    python train_vessel_detector.py \\
        --annotations annotations.json \\
        --detections vessel_detections.json \\
        --n-estimators 200 \\
        --max-depth 20 \\
        --output-dir ./models

    # Use minimal features
    python train_vessel_detector.py \\
        --annotations annotations.json \\
        --detections vessel_detections.json \\
        --minimal-features
        """
    )

    parser.add_argument(
        '--annotations', '-a',
        required=True,
        help='Path to annotations JSON (yes/no labels from HTML review)'
    )
    parser.add_argument(
        '--detections', '-d',
        required=True,
        help='Path to vessel detections JSON with features'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./vessel_detector_output',
        help='Output directory (default: ./vessel_detector_output)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in Random Forest (default: 100)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=15,
        help='Maximum tree depth (default: 15, None for unlimited)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for test set (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--minimal-features',
        action='store_true',
        help='Use minimal feature set (when vessel-specific features unavailable)'
    )

    # Stratification options
    parser.add_argument(
        '--stratify-by-size',
        action='store_true',
        help='Balance training data across vessel size classes (small <50um, medium 50-200um, large >200um)'
    )
    parser.add_argument(
        '--samples-per-size-class',
        type=int,
        default=None,
        help='Target samples per size class per label (default: auto = min class count)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("=" * 60)
    logger.info("VESSEL DETECTOR TRAINING")
    logger.info("=" * 60)

    logger.info("\nLoading annotations...")
    annotations = load_annotations(Path(args.annotations))
    logger.info(f"  Total annotations: {len(annotations)}")

    # Count by label
    label_counts = {}
    for label in annotations.values():
        label_lower = label.lower()
        label_counts[label_lower] = label_counts.get(label_lower, 0) + 1

    for label, count in sorted(label_counts.items()):
        logger.info(f"    {label}: {count}")

    logger.info("\nLoading detections...")
    detections = load_detections(Path(args.detections))
    logger.info(f"  Total detections: {len(detections)}")

    # Determine feature set
    if args.minimal_features:
        feature_names = MINIMAL_DETECTION_FEATURES.copy()
        logger.info(f"\nUsing minimal features: {len(feature_names)}")
    else:
        feature_names = VESSEL_DETECTION_FEATURES.copy()
        logger.info(f"\nUsing full features: {len(feature_names)}")

    # Extract training data
    logger.info("\nExtracting training data...")
    X_list, y_list, valid_uids = extract_training_data(detections, annotations, feature_names)
    logger.info(f"  Matched samples: {len(X_list)}")

    # Convert labels to binary
    y_binary = [1 if str(y).lower() in ('yes', 'vessel', 'true', '1') else 0 for y in y_list]

    # Apply stratified sampling by size if requested
    if args.stratify_by_size:
        logger.info("\n" + "=" * 60)
        logger.info("STRATIFIED SAMPLING BY VESSEL SIZE")
        logger.info("=" * 60)

        # Extract diameters for size analysis
        diameters = np.array([extract_diameter(detections.get(uid, {})) for uid in valid_uids])

        # Analyze original distribution
        analyze_size_distribution(diameters, np.array(y_binary), prefix="Original ")

        # Apply stratified sampling
        X_list, y_binary, valid_uids = stratified_sample_by_size(
            X_list, y_binary, valid_uids, detections,
            samples_per_class=args.samples_per_size_class,
            min_samples_per_class=10,
            random_seed=42,
        )

        # Analyze balanced distribution
        diameters_balanced = np.array([extract_diameter(detections.get(uid, {})) for uid in valid_uids])
        analyze_size_distribution(diameters_balanced, np.array(y_binary), prefix="Balanced ")

    # Split data for evaluation
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_list, y_binary,
        test_size=args.test_size,
        random_state=42,
        stratify=y_binary
    )
    logger.info(f"\n  Training set: {len(X_train)}")
    logger.info(f"  Test set: {len(X_test)}")

    # Train classifier
    logger.info("\n" + "=" * 60)
    logger.info("Training vessel detector")
    logger.info("=" * 60)

    detector = VesselDetectorRF(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        feature_names=feature_names,
    )

    train_metrics = detector.train(
        X_train, y_train,
        feature_names=feature_names,
        cv_folds=args.cv_folds,
        stratify_by_size=args.stratify_by_size,
        verbose=True
    )

    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set")
    logger.info("=" * 60)

    eval_metrics = detector.evaluate(X_test, y_test, verbose=True)

    # Feature importance analysis
    logger.info("\n" + "=" * 60)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 60)

    importance = detector.get_feature_importance()
    top_features = detector.get_top_features(15)

    logger.info("\nTop 15 most important features:")
    for i, (name, score) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {name:35s} = {score:.4f}")

    # Save model
    model_path = output_dir / 'vessel_detector.joblib'
    detector.save(model_path)

    # Save feature importance
    importance_path = output_dir / 'feature_importance.json'
    with open(importance_path, 'w') as f:
        json.dump(importance, f)
    logger.info(f"\nFeature importance saved to: {importance_path}")

    # Save metrics
    metrics = {
        'train': train_metrics,
        'test': {
            'accuracy': eval_metrics['accuracy'],
            'precision': eval_metrics['precision'],
            'recall': eval_metrics['recall'],
            'f1_score': eval_metrics['f1_score'],
            'auc_roc': eval_metrics.get('auc_roc', 0.0),
            'confusion_matrix': eval_metrics['confusion_matrix'],
        },
        'feature_names': feature_names,
        'n_samples': len(X_list),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'model_path': str(model_path),
        'stratify_by_size': args.stratify_by_size,
        'samples_per_size_class': args.samples_per_size_class,
    }

    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, default=str)
    logger.info(f"Training metrics saved to: {metrics_path}")

    # Generate visualizations
    cm = np.array(eval_metrics['confusion_matrix'])

    plot_confusion_matrix(
        cm,
        output_dir / 'confusion_matrix.png',
        title=f'Vessel Detection (F1: {eval_metrics["f1_score"]:.2%})'
    )

    plot_feature_importance(
        importance,
        output_dir / 'feature_importance.png',
        top_n=15,
        title='Top 15 Feature Importances for Vessel Detection'
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Test accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"Test precision: {eval_metrics['precision']:.4f}")
    logger.info(f"Test recall: {eval_metrics['recall']:.4f}")
    logger.info(f"Test F1 score: {eval_metrics['f1_score']:.4f}")
    logger.info(f"CV accuracy: {train_metrics['cv_accuracy_mean']:.4f} (+/- {train_metrics['cv_accuracy_std'] * 2:.4f})")

    return metrics


if __name__ == '__main__':
    main()
