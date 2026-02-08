"""
Vessel detection classifier (vessel vs non-vessel).

This classifier answers: "Is this candidate actually a vessel, or a false positive?"
It is DIFFERENT from vessel type classification (capillary/arteriole/artery).

This is the FIRST STAGE classifier in a two-stage pipeline:
1. VesselDetectorRF: Is this a vessel? (binary: vessel vs non-vessel)
2. ArteryVeinClassifier: What type of vessel? (artery vs vein)

Features used:
    - Morphological: area, perimeter, circularity, solidity, aspect_ratio, etc.
    - Intensity: mean colors (RGB), gray_mean, gray_std, HSV features
    - Texture: relative_brightness, intensity_variance, dark_fraction
    - Vessel-specific: wall_thickness, ring_completeness, diameter, etc.

Usage:
    # Training
    detector = VesselDetectorRF()
    detector.train_from_files(annotations_json, detections_json)
    detector.save('vessel_detector.joblib')

    # Inference
    detector = VesselDetectorRF.load('vessel_detector.joblib')
    is_vessel, confidence = detector.predict_vessel(features)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

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
    """
    Get size class for a vessel based on outer diameter.

    Args:
        diameter_um: Vessel outer diameter in microns

    Returns:
        Size class (0=small, 1=medium, 2=large)
    """
    if np.isnan(diameter_um):
        return 1  # Default to medium if unknown
    if diameter_um < 50:
        return 0  # small
    elif diameter_um < 200:
        return 1  # medium
    else:
        return 2  # large


# Features for vessel detection (morphological + intensity)
# These help distinguish vessels from false positives like tissue artifacts
VESSEL_DETECTION_FEATURES = [
    # Morphological features
    'area',
    'perimeter',
    'circularity',
    'solidity',
    'aspect_ratio',
    'extent',
    'equiv_diameter',
    # Intensity features
    'red_mean', 'red_std',
    'green_mean', 'green_std',
    'blue_mean', 'blue_std',
    'gray_mean', 'gray_std',
    # HSV features
    'hue_mean',
    'saturation_mean',
    'value_mean',
    # Texture features
    'relative_brightness',
    'intensity_variance',
    'dark_fraction',
    'nuclear_complexity',
    # Vessel-specific features (if available)
    'outer_diameter_um',
    'inner_diameter_um',
    'wall_thickness_mean_um',
    'wall_thickness_std_um',
    'ring_completeness',
    'wall_area_um2',
    'lumen_area_um2',
]

# Minimal feature set for when vessel-specific features are not available
MINIMAL_DETECTION_FEATURES = [
    'area',
    'perimeter',
    'circularity',
    'solidity',
    'aspect_ratio',
    'extent',
    'gray_mean',
    'gray_std',
    'relative_brightness',
    'intensity_variance',
]


class VesselDetectorRF:
    """
    Random Forest classifier for binary vessel detection.

    Classifies candidates into two categories:
    - vessel: True positive - this is actually a vessel
    - non_vessel: False positive - artifact, noise, or other structure

    This classifier is designed to be used AFTER candidate detection
    (which runs in permissive mode to minimize false negatives) to
    filter out false positives.

    Attributes:
        model: Trained RandomForestClassifier
        feature_names: List of feature names used
        trained: Whether model has been trained
        metrics: Training metrics (accuracy, precision, recall, F1)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = 'balanced',
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize vessel detector.

        Args:
            n_estimators: Number of trees in forest (default 100)
            max_depth: Maximum tree depth (default 15, None for unlimited)
            min_samples_split: Min samples to split internal node
            min_samples_leaf: Min samples at leaf node
            class_weight: 'balanced' to handle class imbalance
            random_state: Random seed for reproducibility
            feature_names: List of feature names to use (None for defaults)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_names = feature_names or VESSEL_DETECTION_FEATURES.copy()
        self.trained = False
        self.metrics: Dict[str, Any] = {}

    def _extract_diameter_from_dict(
        self,
        features_dict: Dict[str, Any]
    ) -> float:
        """
        Extract vessel diameter from a features dictionary.

        Args:
            features_dict: Dictionary of features from vessel detection

        Returns:
            Diameter in microns, or NaN if not available
        """
        # Try different possible field names for diameter
        diameter = features_dict.get('outer_diameter_um')
        if diameter is None:
            diameter = features_dict.get('diameter_um')
        if diameter is None:
            # Fall back to estimating from area
            area = features_dict.get('outer_area_um2') or features_dict.get('area')
            if area is not None and area > 0:
                diameter = 2 * np.sqrt(area / np.pi)

        return float(diameter) if diameter is not None else np.nan

    def _extract_features_from_dict(
        self,
        features_dict: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Extract feature vector from a features dictionary.

        Args:
            features_dict: Dictionary of features from vessel detection
            feature_names: List of feature names to extract

        Returns:
            1D numpy array of feature values
        """
        if feature_names is None:
            feature_names = self.feature_names

        values = []
        for name in feature_names:
            val = features_dict.get(name, 0)
            # Handle non-scalar values
            if isinstance(val, (list, tuple)):
                val = 0
            elif val is None:
                val = 0
            elif isinstance(val, bool):
                val = 1 if val else 0
            values.append(float(val))

        return np.array(values, dtype=np.float32)

    def _prepare_training_data(
        self,
        X: Union[np.ndarray, List[Dict]],
        y: Union[np.ndarray, List[str], List[bool]],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data, handling both array and dict inputs.

        Args:
            X: Feature matrix (N, D) or list of feature dicts
            y: Labels as array, list of strings ('yes'/'no'), or list of bools

        Returns:
            Tuple of (X_array, y_encoded) where y_encoded is 1 for vessel, 0 for non-vessel
        """
        # Handle dict input
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X_array = np.array([
                self._extract_features_from_dict(d, feature_names)
                for d in X
            ])
        else:
            X_array = np.asarray(X, dtype=np.float32)

        # Handle different label formats
        y_array = np.asarray(y)

        # Convert to binary: 1 = vessel, 0 = non-vessel
        if y_array.dtype == bool:
            y_encoded = y_array.astype(int)
        elif y_array.dtype.kind in ('U', 'S', 'O'):  # String types
            # Accept: 'yes', 'vessel', 'true', '1' -> 1
            # Accept: 'no', 'non_vessel', 'false', '0' -> 0
            y_encoded = np.array([
                1 if str(label).lower() in ('yes', 'vessel', 'true', '1') else 0
                for label in y_array
            ], dtype=int)
        else:
            y_encoded = y_array.astype(int)

        return X_array, y_encoded

    def train(
        self,
        X: Union[np.ndarray, List[Dict]],
        y: Union[np.ndarray, List[str], List[bool]],
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
        stratify_by_size: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the binary vessel detector.

        Args:
            X: Feature matrix (N, D) or list of feature dicts
            y: Labels (N,) - bool, int, or strings ('yes'/'no', 'vessel'/'non_vessel')
            feature_names: Names of features in X (required if X is array)
            cv_folds: Number of cross-validation folds
            stratify_by_size: If True and X is list of dicts, stratify CV by vessel size
            verbose: Print training progress

        Returns:
            Dictionary of training metrics
        """
        if feature_names is not None:
            self.feature_names = feature_names

        # Extract size classes if stratifying by size (only works with dict input)
        size_classes = None
        if stratify_by_size and isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            diameters = np.array([self._extract_diameter_from_dict(d) for d in X])
            size_classes = np.array([get_size_class(d) for d in diameters])
            if verbose:
                from collections import Counter
                counts = Counter(size_classes)
                logger.info("Size class distribution:")
                for class_id in range(3):
                    logger.info(f"  {SIZE_CLASS_NAMES[class_id]}: {counts.get(class_id, 0)}")

        # Prepare data
        X_array, y_encoded = self._prepare_training_data(X, y, feature_names)

        if verbose:
            logger.info(f"Training vessel detector with {len(X_array)} samples")
            logger.info(f"Features: {len(self.feature_names)}")
            unique, counts = np.unique(y_encoded, return_counts=True)
            for idx, count in zip(unique, counts):
                label = 'vessel' if idx == 1 else 'non_vessel'
                logger.info(f"  {label}: {count} samples")

        # Handle NaN/Inf
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # RF is scale-invariant, so no StandardScaler needed.
        # Pass raw features directly to avoid data leakage from fitting scaler on full data.

        # Cross-validation
        if verbose:
            logger.info(f"Running {cv_folds}-fold cross-validation...")

        # Create stratification groups combining label and size class
        if stratify_by_size and size_classes is not None:
            # Combine label and size class for compound stratification
            # e.g., label=0, size=1 -> group=1; label=1, size=1 -> group=4
            stratify_groups = y_encoded * 3 + size_classes
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X_array, y_encoded, cv=cv.split(X_array, stratify_groups), scoring='accuracy')
        else:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X_array, y_encoded, cv=cv, scoring='accuracy')

        if verbose:
            logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train on full dataset
        self.model.fit(X_array, y_encoded)
        self.trained = True

        # Compute training metrics
        y_pred = self.model.predict(X_array)
        y_proba = self.model.predict_proba(X_array)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_encoded, y_pred, average='binary', pos_label=1
        )

        # ROC AUC
        try:
            auc = roc_auc_score(y_encoded, y_proba)
        except ValueError:
            auc = 0.0

        train_accuracy = (y_pred == y_encoded).mean()

        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc),
            'n_samples': len(X_array),
            'n_features': len(self.feature_names),
            'class_distribution': {
                'vessel': int(np.sum(y_encoded == 1)),
                'non_vessel': int(np.sum(y_encoded == 0)),
            },
        }

        if verbose:
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"AUC-ROC: {auc:.4f}")

        return self.metrics

    def train_from_files(
        self,
        annotations_path: Union[str, Path],
        detections_path: Union[str, Path],
        cv_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train from annotation and detection JSON files.

        Args:
            annotations_path: Path to annotations JSON (from HTML review)
                Format: {"uid": "yes"/"no", ...} or {"annotations": {"uid": "yes"/"no", ...}}
            detections_path: Path to detections JSON
                Format: [{"uid": "...", "features": {...}}, ...]
            cv_folds: Number of cross-validation folds
            verbose: Print training progress

        Returns:
            Dictionary of training metrics
        """
        # Load annotations
        with open(annotations_path) as f:
            annotations_data = json.load(f)

        # Handle nested format
        if 'annotations' in annotations_data:
            annotations = annotations_data['annotations']
        else:
            annotations = annotations_data

        # Load detections
        with open(detections_path) as f:
            detections_list = json.load(f)

        # Build detection index by UID
        detections_by_uid = {}
        for d in detections_list:
            if 'uid' in d:
                detections_by_uid[d['uid']] = d
            if 'id' in d:
                detections_by_uid[d['id']] = d
            # Alternative formats
            if 'tile_origin' in d and 'id' in d:
                tile_x, tile_y = d['tile_origin']
                alt_id = f"{tile_x}_{tile_y}_{d['id']}"
                detections_by_uid[alt_id] = d

        # Match annotations to detections
        X_list = []
        y_list = []
        matched_count = 0
        unmatched_uids = []

        for uid, label in annotations.items():
            if uid not in detections_by_uid:
                unmatched_uids.append(uid)
                continue

            det = detections_by_uid[uid]
            features = det.get('features', det)  # Features might be at top level or nested

            X_list.append(features)
            y_list.append(label)
            matched_count += 1

        if verbose:
            logger.info(f"Matched {matched_count}/{len(annotations)} annotations")
            if unmatched_uids:
                logger.warning(f"Unmatched UIDs: {unmatched_uids[:10]}{'...' if len(unmatched_uids) > 10 else ''}")

        if not X_list:
            raise ValueError("No matching samples found! Check annotation UIDs match detection UIDs.")

        return self.train(X_list, y_list, cv_folds=cv_folds, verbose=verbose)

    def predict_vessel(
        self,
        features: Union[Dict[str, Any], np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Predict whether a candidate is a vessel.

        Args:
            features: Feature dictionary or array for a single candidate

        Returns:
            Tuple of (is_vessel: bool, confidence: float)
            - is_vessel: True if classified as vessel
            - confidence: Probability of being a vessel (0.0 to 1.0)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Extract features
        if isinstance(features, dict):
            X = self._extract_features_from_dict(features).reshape(1, -1)
        else:
            X = np.asarray(features, dtype=np.float32).reshape(1, -1)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Predict directly (RF is scale-invariant, no scaler needed)
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        is_vessel = bool(prediction == 1)
        confidence = float(proba[1])  # Probability of vessel class

        return is_vessel, confidence

    def predict_batch(
        self,
        features_list: List[Union[Dict[str, Any], np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for multiple candidates.

        Args:
            features_list: List of feature dictionaries or arrays

        Returns:
            Tuple of (is_vessel_array, confidence_array)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Extract features
        if isinstance(features_list[0], dict):
            X = np.array([self._extract_features_from_dict(f) for f in features_list])
        else:
            X = np.asarray(features_list, dtype=np.float32)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Predict directly (RF is scale-invariant, no scaler needed)
        predictions = self.model.predict(X)
        probas = self.model.predict_proba(X)

        is_vessel = predictions == 1
        confidence = probas[:, 1]

        return is_vessel, confidence

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance rankings.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def evaluate(
        self,
        X: Union[np.ndarray, List[Dict]],
        y: Union[np.ndarray, List[str], List[bool]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate detector on test data.

        Args:
            X: Feature matrix or list of feature dicts
            y: True labels
            verbose: Print evaluation results

        Returns:
            Dictionary with accuracy, precision, recall, F1, confusion matrix
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Prepare data
        X_array, y_true = self._prepare_training_data(X, y)
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Predict directly (RF is scale-invariant, no scaler needed)
        y_pred = self.model.predict(X_array)
        y_proba = self.model.predict_proba(X_array)[:, 1]

        # Compute metrics
        accuracy = (y_pred == y_true).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1
        )

        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=['non_vessel', 'vessel'],
            output_dict=True
        )

        if verbose:
            logger.info(f"\nTest Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"AUC-ROC: {auc:.4f}")
            logger.info("\nClassification Report:")
            print(classification_report(
                y_true, y_pred,
                target_names=['non_vessel', 'vessel']
            ))
            logger.info("\nConfusion Matrix:")
            logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
            logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save trained model to file.

        Args:
            path: Output file path (use .joblib extension)
        """
        if not self.trained:
            raise RuntimeError("Cannot save untrained model. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'class_weight': self.class_weight,
                'random_state': self.random_state,
            },
            'model_type': 'vessel_detector',
            'version': '1.0',
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VesselDetectorRF':
        """
        Load trained model from file.

        Args:
            path: Path to saved model file

        Returns:
            VesselDetectorRF instance with loaded model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Verify model type
        if model_data.get('model_type') != 'vessel_detector':
            logger.warning(f"Model type mismatch: expected 'vessel_detector', got '{model_data.get('model_type')}'")

        # Create instance with saved config
        config = model_data.get('config', {})
        instance = cls(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 15),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=config.get('random_state', 42),
            feature_names=model_data['feature_names'],
        )

        # Restore trained state
        instance.model = model_data['model']
        instance.metrics = model_data.get('metrics', {})
        instance.trained = True

        logger.info(f"Model loaded from: {path}")
        logger.info(f"  F1 Score: {instance.metrics.get('f1_score', 'N/A'):.4f}")
        logger.info(f"  Features: {len(instance.feature_names)}")

        return instance


def train_vessel_detector(
    annotations_json: Union[str, Path],
    detections_json: Union[str, Path],
    **kwargs
) -> VesselDetectorRF:
    """
    Convenience function to train a vessel detector.

    Args:
        annotations_json: Path to annotations JSON (from HTML review)
        detections_json: Path to detections JSON
        **kwargs: Additional arguments passed to VesselDetectorRF

    Returns:
        Trained VesselDetectorRF instance
    """
    detector = VesselDetectorRF(**kwargs)
    detector.train_from_files(annotations_json, detections_json)
    return detector


def predict_vessel(
    model: VesselDetectorRF,
    features: Dict[str, Any]
) -> Tuple[bool, float]:
    """
    Convenience function to predict vessel status.

    Args:
        model: Trained VesselDetectorRF instance
        features: Feature dictionary for candidate

    Returns:
        Tuple of (is_vessel: bool, confidence: float)
    """
    return model.predict_vessel(features)
