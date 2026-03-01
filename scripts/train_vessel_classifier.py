#!/usr/bin/env python3
"""
Train vessel type classifier using annotated vessel data.

This script trains a Random Forest classifier to classify vessels as
capillary, arteriole, or artery based on morphological features.

Input:
    - Detections JSON with vessel features (from run_segmentation.py --cell-type vessel)
    - Annotations JSON mapping vessel IDs to types

Output:
    - Trained model file (.joblib)
    - Feature importance JSON
    - Confusion matrix visualization (PNG)
    - Training metrics JSON

Usage:
    python scripts/train_vessel_classifier.py \\
        --annotations /path/to/annotations.json \\
        --detections /path/to/vessel_detections.json \\
        --output-dir /path/to/output

Annotation format (annotations.json):
    {
        "vessel_uid_1": "capillary",
        "vessel_uid_2": "arteriole",
        "vessel_uid_3": "artery",
        ...
    }

    Or alternative format:
    {
        "annotations": {
            "vessel_uid_1": "capillary",
            ...
        }
    }
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.classification.vessel_classifier import (
    VesselClassifier,
    VESSEL_CORE_FEATURES,
    MORPHOLOGICAL_FEATURES,
    DEFAULT_FEATURES,
    FULL_FEATURES,
)
from segmentation.classification.feature_selection import (
    select_optimal_features,
    compare_feature_sets,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_annotations(annotations_path: Path) -> Dict[str, str]:
    """
    Load vessel type annotations from JSON file.

    Supports multiple formats:
    1. Direct mapping: {"uid": "vessel_type", ...}
    2. Nested: {"annotations": {"uid": "vessel_type", ...}}

    Args:
        annotations_path: Path to annotations JSON

    Returns:
        Dictionary mapping vessel UID to vessel type
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

    Builds index by multiple possible ID formats for flexible matching.

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
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix and labels from annotated vessels.

    Args:
        detections: Detection dictionary indexed by UID
        annotations: Mapping of UID to vessel type
        feature_names: List of feature names to extract

    Returns:
        Tuple of (X, y, valid_uids) where:
        - X: Feature matrix (N, D)
        - y: Label array (N,)
        - valid_uids: List of UIDs that were successfully matched
    """
    X_list = []
    y_list = []
    valid_uids = []

    for uid, vessel_type in annotations.items():
        if uid not in detections:
            logger.warning(f"UID not found in detections: {uid}")
            continue

        det = detections[uid]
        features = det.get('features', {})

        # Extract feature values
        row = []
        for fname in feature_names:
            val = features.get(fname, 0)
            if isinstance(val, (list, tuple)):
                val = 0
            elif val is None:
                val = 0
            row.append(float(val))

        X_list.append(row)
        y_list.append(vessel_type)
        valid_uids.append(uid)

    if not X_list:
        raise ValueError("No matching samples found! Check annotation UIDs.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    return X, y, valid_uids


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plot and save confusion matrix visualization.

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save PNG
        title: Plot title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))

        # Normalize confusion matrix for color mapping
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=cm,  # Show actual counts
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
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
    title: str = 'Feature Importance'
) -> None:
    """
    Plot and save feature importance visualization.

    Args:
        importance: Dictionary of feature importance scores
        output_path: Path to save PNG
        top_n: Number of top features to show
        title: Plot title
    """
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
        ax.barh(y_pos, scores, align='center')
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


def train_vessel_classifier(
    annotations_path: str,
    detections_path: str,
    output_dir: str,
    n_estimators: int = 100,
    max_depth: int = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    feature_selection: bool = True,
    use_core_features_only: bool = False,
    use_full_features: bool = False,
) -> Dict[str, Any]:
    """
    Train vessel type classifier.

    Args:
        annotations_path: Path to annotations JSON
        detections_path: Path to detections JSON
        output_dir: Output directory for model and metrics
        n_estimators: Number of RF trees
        max_depth: Maximum tree depth
        test_size: Fraction for test set
        cv_folds: Number of CV folds
        feature_selection: Run feature selection analysis
        use_core_features_only: Use only vessel-specific features
        use_full_features: Use full 2326 features (22 morph + 256 SAM2 + 2048 ResNet)

    Returns:
        Dictionary of training results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading annotations...")
    annotations = load_annotations(Path(annotations_path))
    logger.info(f"  Total annotations: {len(annotations)}")

    # Count by type
    type_counts = {}
    for vtype in annotations.values():
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    for vtype, count in sorted(type_counts.items()):
        logger.info(f"    {vtype}: {count}")

    logger.info("Loading detections...")
    detections = load_detections(Path(detections_path))
    logger.info(f"  Total detections: {len(detections)}")

    # Determine feature set
    if use_core_features_only:
        feature_names = VESSEL_CORE_FEATURES.copy()
        logger.info(f"Using core vessel features only: {len(feature_names)}")
    elif use_full_features:
        feature_names = FULL_FEATURES.copy()
        logger.info(f"Using full 2326 features (morph + SAM2 + ResNet): {len(feature_names)}")
    else:
        feature_names = DEFAULT_FEATURES.copy()
        logger.info(f"Using default features: {len(feature_names)}")

    # Extract training data
    logger.info("Extracting training data...")
    X, y, valid_uids = extract_training_data(detections, annotations, feature_names)
    logger.info(f"  Matched samples: {len(X)}")
    logger.info(f"  Features: {len(feature_names)}")

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )
    logger.info(f"  Training set: {len(X_train)}")
    logger.info(f"  Test set: {len(X_test)}")

    # Train classifier
    logger.info("\n" + "=" * 60)
    logger.info("Training vessel classifier")
    logger.info("=" * 60)

    classifier = VesselClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        feature_names=feature_names,
    )

    train_metrics = classifier.train(
        X_train, y_train,
        feature_names=feature_names,
        cv_folds=cv_folds,
        verbose=True
    )

    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set")
    logger.info("=" * 60)

    eval_metrics = classifier.evaluate(X_test, y_test, verbose=True)

    # Feature importance analysis
    logger.info("\n" + "=" * 60)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 60)

    importance = classifier.get_feature_importance()
    top_features = classifier.get_top_features(20)

    logger.info("\nTop 20 most important features:")
    for i, (name, score) in enumerate(top_features, 1):
        logger.info(f"  {i:2d}. {name:35s} = {score:.4f}")

    # Optional: Run full feature selection analysis
    if feature_selection:
        logger.info("\n" + "=" * 60)
        logger.info("Running Feature Selection Analysis")
        logger.info("=" * 60)

        # Compare feature sets
        feature_sets = {
            'all_features': feature_names,
            'vessel_core': VESSEL_CORE_FEATURES,
            'morphological': MORPHOLOGICAL_FEATURES,
            'top_10': [f[0] for f in top_features[:10]],
        }

        comparison = compare_feature_sets(
            X, y,
            feature_names=feature_names,
            feature_sets=feature_sets,
            cv_folds=cv_folds,
        )

        # RFECV for optimal feature subset
        logger.info("\nRunning RFECV...")
        optimal = select_optimal_features(
            X, y,
            feature_names=feature_names,
            method='rfecv',
            min_features=5,
            cv_folds=cv_folds,
        )
        logger.info(f"RFECV selected {optimal['n_features']} features")

    # Save model
    model_path = output_dir / 'vessel_classifier.joblib'
    classifier.save(model_path)

    # Save feature importance
    importance_path = output_dir / 'feature_importance.json'
    with open(importance_path, 'w') as f:
        json.dump(importance, f)
    logger.info(f"Feature importance saved to: {importance_path}")

    # Save metrics
    metrics = {
        'train': train_metrics,
        'test': {
            'accuracy': eval_metrics['accuracy'],
            'classification_report': eval_metrics['classification_report'],
        },
        'feature_names': feature_names,
        'n_samples': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    if feature_selection:
        metrics['feature_comparison'] = comparison
        metrics['optimal_features'] = optimal

    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, default=str)
    logger.info(f"Training metrics saved to: {metrics_path}")

    # Generate visualizations
    cm = np.array(eval_metrics['confusion_matrix'])
    class_names = classifier.label_encoder.classes_.tolist()

    plot_confusion_matrix(
        cm, class_names,
        output_dir / 'confusion_matrix.png',
        title=f'Vessel Classification (Acc: {eval_metrics["accuracy"]:.2%})'
    )

    plot_feature_importance(
        importance,
        output_dir / 'feature_importance.png',
        top_n=20,
        title='Top 20 Feature Importances'
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Test accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"CV accuracy: {train_metrics['cv_accuracy_mean']:.4f} (+/- {train_metrics['cv_accuracy_std'] * 2:.4f})")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train vessel type classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train_vessel_classifier.py \\
        --annotations annotations.json \\
        --detections vessel_detections.json

    # With custom parameters
    python train_vessel_classifier.py \\
        --annotations annotations.json \\
        --detections vessel_detections.json \\
        --n-estimators 200 \\
        --max-depth 10 \\
        --output-dir ./models

    # Use only vessel-specific features
    python train_vessel_classifier.py \\
        --annotations annotations.json \\
        --detections vessel_detections.json \\
        --core-features-only
        """
    )

    parser.add_argument(
        '--annotations', '-a',
        required=True,
        help='Path to annotations JSON (mapping vessel UIDs to types)'
    )
    parser.add_argument(
        '--detections', '-d',
        required=True,
        help='Path to vessel detections JSON with features'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./vessel_classifier_output',
        help='Output directory (default: ./vessel_classifier_output)'
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
        default=None,
        help='Maximum tree depth (default: unlimited)'
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
        '--no-feature-selection',
        action='store_true',
        help='Skip feature selection analysis (faster)'
    )
    parser.add_argument(
        '--core-features-only',
        action='store_true',
        help='Use only vessel-specific features (diameter, wall thickness, etc.)'
    )
    parser.add_argument(
        '--full-features',
        action='store_true',
        help='Use full 2326 features (22 morphological + 256 SAM2 + 2048 ResNet)'
    )

    args = parser.parse_args()

    train_vessel_classifier(
        annotations_path=args.annotations,
        detections_path=args.detections,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        feature_selection=not args.no_feature_selection,
        use_core_features_only=args.core_features_only,
        use_full_features=args.full_features,
    )


if __name__ == '__main__':
    main()
