"""
Multi-class vessel type classifier.

Classifies vessels into 6 types based on marker profiles and morphological features:
- artery: Large SMA+ ring, thick wall, CD31+ at lumen
- arteriole: Smaller SMA+ ring, CD31+ at lumen
- vein: Thin SMA+/- wall, larger lumen, CD31+ at lumen
- capillary: No SMA, CD31+ tubular, small diameter
- lymphatic: LYVE1+, CD31-, irregular shape
- collecting_lymphatic: LYVE1+, SMA+

This classifier uses a Random Forest trained on:
1. Marker intensity scores (SMA, CD31, LYVE1 in wall and lumen)
2. Marker ratios (SMA/CD31, LYVE1/CD31)
3. Morphological features (diameter, wall thickness, circularity)
4. Detection provenance (which marker detected this vessel)

Usage:
    # Training
    classifier = VesselTypeClassifier()
    classifier.train(X, y, feature_names)
    classifier.save('vessel_type_classifier.joblib')

    # Inference
    classifier = VesselTypeClassifier.load('vessel_type_classifier.joblib')
    vessel_type, confidence = classifier.predict(features)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


# =============================================================================
# VESSEL TYPE DEFINITIONS
# =============================================================================

VESSEL_TYPES = [
    'artery',
    'arteriole',
    'vein',
    'capillary',
    'lymphatic',
    'collecting_lymphatic',
]

# Vessel type characteristics for documentation and rule-based fallback
VESSEL_TYPE_CHARACTERISTICS = {
    'artery': {
        'description': 'Large muscular vessel with thick SMA+ wall',
        'sma_wall': 'high',
        'cd31_lumen': 'present',
        'lyve1': 'absent',
        'diameter_range_um': (100, 5000),
        'wall_thickness_ratio': 'high (>0.1)',
    },
    'arteriole': {
        'description': 'Small muscular vessel with moderate SMA+ wall',
        'sma_wall': 'moderate-high',
        'cd31_lumen': 'present',
        'lyve1': 'absent',
        'diameter_range_um': (10, 100),
        'wall_thickness_ratio': 'moderate (0.05-0.15)',
    },
    'vein': {
        'description': 'Vessel with thin SMA+/- wall and large lumen',
        'sma_wall': 'low-moderate',
        'cd31_lumen': 'present',
        'lyve1': 'absent',
        'diameter_range_um': (15, 3000),
        'wall_thickness_ratio': 'low (<0.05)',
    },
    'capillary': {
        'description': 'Smallest vessel, no muscular wall',
        'sma_wall': 'absent',
        'cd31_lumen': 'present (tubular)',
        'lyve1': 'absent',
        'diameter_range_um': (3, 10),
        'wall_thickness_ratio': 'none (single endothelial layer)',
    },
    'lymphatic': {
        'description': 'LYVE1+ vessel with irregular shape',
        'sma_wall': 'absent',
        'cd31_lumen': 'absent or weak',
        'lyve1': 'present',
        'diameter_range_um': (10, 200),
        'wall_thickness_ratio': 'thin, irregular',
    },
    'collecting_lymphatic': {
        'description': 'Large LYVE1+ vessel with SMA+ wall',
        'sma_wall': 'present (discontinuous)',
        'cd31_lumen': 'absent or weak',
        'lyve1': 'present',
        'diameter_range_um': (50, 500),
        'wall_thickness_ratio': 'moderate (discontinuous SMA)',
    },
}


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Key features for vessel type classification
# These are the most discriminative features for distinguishing vessel types
TYPE_FEATURES = [
    # Marker intensity scores - wall region
    'sma_wall_mean',
    'cd31_wall_mean',
    'lyve1_wall_mean',
    # Marker intensity scores - lumen region
    'sma_lumen_mean',
    'cd31_lumen_mean',
    'lyve1_lumen_mean',
    # Marker ratios (highly discriminative)
    'sma_cd31_wall_ratio',
    'lyve1_cd31_ratio',
    'sma_wall_lumen_contrast',
    # Morphology
    'outer_diameter_um',
    'inner_diameter_um',
    'wall_thickness_mean_um',
    'wall_thickness_ratio',  # wall_thickness / outer_diameter
    'ring_completeness',
    'circularity',
    'tubularity',  # For capillaries - length/diameter ratio
    # Detection provenance
    'detected_by_sma',
    'detected_by_cd31',
    'detected_by_lyve1',
]

# Extended features including all morphological and intensity features
EXTENDED_TYPE_FEATURES = TYPE_FEATURES + [
    # Additional morphology
    'aspect_ratio',
    'solidity',
    'convexity',
    'lumen_wall_ratio',
    'wall_uniformity',
    'wall_asymmetry',
    # Additional intensity features
    'wall_intensity_std',
    'wall_lumen_contrast',
    'wall_background_contrast',
    # Size features (for stratification)
    'log_diameter',
    'size_class',
]

# Minimal feature set when only basic features are available
MINIMAL_TYPE_FEATURES = [
    'outer_diameter_um',
    'wall_thickness_mean_um',
    'circularity',
    'ring_completeness',
    'sma_wall_mean',
    'cd31_wall_mean',
]


# =============================================================================
# VESSEL TYPE CLASSIFIER
# =============================================================================

class VesselTypeClassifier:
    """
    Multi-class classifier for vessel type classification.

    Classifies vessels into 6 types based on marker profiles and morphology:
    - artery: Large SMA+ ring, thick wall, CD31+ at lumen
    - arteriole: Smaller SMA+ ring, CD31+ at lumen
    - vein: Thin SMA+/- wall, larger lumen, CD31+ at lumen
    - capillary: No SMA, CD31+ tubular, small diameter
    - lymphatic: LYVE1+, CD31-, irregular shape
    - collecting_lymphatic: LYVE1+, SMA+

    The classifier uses a Random Forest trained on marker intensity features,
    marker ratios, and morphological features.

    Attributes:
        model: Trained RandomForestClassifier
        scaler: StandardScaler for feature normalization
        label_encoder: LabelEncoder for class labels
        feature_names: List of feature names used
        trained: Whether model has been trained
        metrics: Training metrics (accuracy, precision, recall, F1)
    """

    # Class labels
    VESSEL_TYPES = VESSEL_TYPES

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = 'balanced',
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize vessel type classifier.

        Args:
            n_estimators: Number of trees in forest (default 200)
            max_depth: Maximum tree depth (default 20)
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

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(VESSEL_TYPES)
        self.feature_names = feature_names or TYPE_FEATURES.copy()
        self.trained = False
        self.metrics: Dict[str, Any] = {}

    def _compute_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute derived features important for vessel type classification.

        Args:
            features: Original feature dictionary

        Returns:
            Feature dictionary with derived features added
        """
        features = features.copy()

        # Wall thickness ratio (wall_thickness / outer_diameter)
        wall_thickness = features.get('wall_thickness_mean_um', 0)
        outer_diameter = features.get('outer_diameter_um', 1)
        if outer_diameter > 0:
            features['wall_thickness_ratio'] = wall_thickness / outer_diameter
        else:
            features['wall_thickness_ratio'] = 0

        # SMA/CD31 wall ratio
        sma_wall = features.get('sma_wall_mean', 0)
        cd31_wall = features.get('cd31_wall_mean', 0)
        if cd31_wall is not None and cd31_wall > 0:
            features['sma_cd31_wall_ratio'] = sma_wall / cd31_wall if sma_wall is not None else 0
        else:
            features['sma_cd31_wall_ratio'] = float('inf') if sma_wall and sma_wall > 0 else 0

        # LYVE1/CD31 ratio (for lymphatic detection)
        lyve1_wall = features.get('lyve1_wall_mean', 0)
        if cd31_wall is not None and cd31_wall > 0:
            features['lyve1_cd31_ratio'] = lyve1_wall / cd31_wall if lyve1_wall is not None else 0
        else:
            features['lyve1_cd31_ratio'] = float('inf') if lyve1_wall and lyve1_wall > 0 else 0

        # SMA wall-lumen contrast
        sma_lumen = features.get('sma_lumen_mean', 0)
        if sma_wall is not None and sma_wall > 0 and sma_lumen is not None:
            features['sma_wall_lumen_contrast'] = (sma_wall - sma_lumen) / sma_wall
        else:
            features['sma_wall_lumen_contrast'] = 0

        # Tubularity (for capillary detection) - approximated from aspect ratio
        # Higher tubularity = more elongated = likely capillary longitudinal section
        aspect_ratio = features.get('aspect_ratio', 1)
        features['tubularity'] = aspect_ratio if aspect_ratio > 1 else 1 / aspect_ratio if aspect_ratio > 0 else 1

        # Log diameter for scale-invariant comparisons
        if outer_diameter is not None and outer_diameter > 0:
            features['log_diameter'] = np.log(outer_diameter + 1)
        else:
            features['log_diameter'] = 0

        # Size class (0=capillary, 1=arteriole, 2=small_artery, 3=artery)
        if outer_diameter is not None:
            if outer_diameter < 10:
                features['size_class'] = 0
            elif outer_diameter < 50:
                features['size_class'] = 1
            elif outer_diameter < 150:
                features['size_class'] = 2
            else:
                features['size_class'] = 3
        else:
            features['size_class'] = 1  # Default to arteriole

        # Detection provenance (binary flags)
        # These should be set by the detection pipeline, but provide defaults
        if 'detected_by_sma' not in features:
            features['detected_by_sma'] = 1 if (sma_wall or 0) > 50 else 0
        if 'detected_by_cd31' not in features:
            features['detected_by_cd31'] = 1 if (cd31_wall or 0) > 50 else 0
        if 'detected_by_lyve1' not in features:
            features['detected_by_lyve1'] = 1 if (lyve1_wall or 0) > 50 else 0

        return features

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

        # Compute derived features first
        features_dict = self._compute_derived_features(features_dict)

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
            elif val == float('inf') or val == float('-inf'):
                val = 0  # Replace inf with 0
            values.append(float(val))

        return np.array(values, dtype=np.float32)

    def _prepare_training_data(
        self,
        X: Union[np.ndarray, List[Dict]],
        y: Union[np.ndarray, List[str]],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data, handling both array and dict inputs.

        Args:
            X: Feature matrix (N, D) or list of feature dicts
            y: Labels as array or list of strings (vessel type names)

        Returns:
            Tuple of (X_array, y_encoded)
        """
        # Handle dict input
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X_array = np.array([
                self._extract_features_from_dict(d, feature_names)
                for d in X
            ])
        else:
            X_array = np.asarray(X, dtype=np.float32)

        # Handle string labels
        y_array = np.asarray(y)
        if y_array.dtype.kind in ('U', 'S', 'O'):  # String types
            # Validate labels are valid vessel types
            invalid_labels = set(y_array) - set(VESSEL_TYPES)
            if invalid_labels:
                logger.warning(f"Unknown vessel type labels: {invalid_labels}")
                # Map unknown labels to closest match or 'vein' as default
                y_array = np.array([
                    label if label in VESSEL_TYPES else 'vein'
                    for label in y_array
                ])
            y_encoded = self.label_encoder.transform(y_array)
        else:
            y_encoded = y_array.astype(int)

        return X_array, y_encoded

    def train(
        self,
        X: Union[np.ndarray, List[Dict]],
        y: Union[np.ndarray, List[str]],
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the vessel type classifier on labeled data.

        Args:
            X: Feature matrix (N, D) or list of feature dicts
            y: Labels (N,) - strings (vessel type names from VESSEL_TYPES)
            feature_names: Names of features in X
            cv_folds: Number of cross-validation folds
            verbose: Print training progress

        Returns:
            Dictionary of training metrics
        """
        if feature_names is not None:
            self.feature_names = feature_names

        # Prepare data
        X_array, y_encoded = self._prepare_training_data(X, y, feature_names)

        if verbose:
            logger.info(f"Training vessel type classifier with {len(X_array)} samples")
            logger.info(f"Features: {len(self.feature_names)}")
            unique, counts = np.unique(y_encoded, return_counts=True)
            for idx, count in zip(unique, counts):
                label = self.label_encoder.inverse_transform([idx])[0]
                logger.info(f"  {label}: {count} samples")

        # Handle NaN/Inf
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)

        # Cross-validation
        if cv_folds > 1 and len(X_array) >= cv_folds:
            if verbose:
                logger.info(f"Running {cv_folds}-fold cross-validation...")

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='accuracy')

            if verbose:
                logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        else:
            cv_scores = np.array([0.0])
            if verbose:
                logger.info("Skipping cross-validation (insufficient samples)")

        # Train on full dataset
        self.model.fit(X_scaled, y_encoded)
        self.trained = True

        # Compute training metrics
        y_pred = self.model.predict(X_scaled)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_encoded, y_pred, average='weighted'
        )

        train_accuracy = (y_pred == y_encoded).mean()

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(y_encoded, y_pred, average=None)

        per_class_metrics = {}
        for i, label in enumerate(self.label_encoder.classes_):
            if i < len(per_class_precision):
                per_class_metrics[label] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(per_class_f1[i]),
                    'support': int(per_class_support[i]) if i < len(per_class_support) else 0,
                }

        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(X_array),
            'n_features': len(self.feature_names),
            'n_classes': len(VESSEL_TYPES),
            'class_distribution': {
                label: int(count)
                for label, count in zip(
                    self.label_encoder.classes_,
                    np.bincount(y_encoded, minlength=len(VESSEL_TYPES))
                )
            },
            'per_class_metrics': per_class_metrics,
        }

        if verbose:
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Precision (weighted): {precision:.4f}")
            logger.info(f"Recall (weighted): {recall:.4f}")
            logger.info(f"F1 Score (weighted): {f1:.4f}")

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
            annotations_path: Path to annotations JSON
                Format: {"uid": "artery"/"vein"/"capillary"/etc., ...}
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

        # Filter to only valid vessel type annotations
        valid_annotations = {
            uid: label.lower() for uid, label in annotations.items()
            if label.lower() in VESSEL_TYPES
        }

        if verbose:
            logger.info(f"Found {len(valid_annotations)} valid vessel type annotations")
            type_counts = {}
            for label in valid_annotations.values():
                type_counts[label] = type_counts.get(label, 0) + 1
            for vtype, count in sorted(type_counts.items()):
                logger.info(f"  {vtype}: {count}")

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

        # Match annotations to detections
        X_list = []
        y_list = []
        matched_count = 0

        for uid, label in valid_annotations.items():
            if uid not in detections_by_uid:
                continue

            det = detections_by_uid[uid]
            features = det.get('features', det)

            X_list.append(features)
            y_list.append(label)
            matched_count += 1

        if verbose:
            logger.info(f"Matched {matched_count}/{len(valid_annotations)} annotations to detections")

        if not X_list:
            raise ValueError("No matching samples found! Check annotation UIDs match detection UIDs.")

        return self.train(X_list, y_list, cv_folds=cv_folds, verbose=verbose)

    def predict(
        self,
        features: Union[Dict[str, Any], np.ndarray]
    ) -> Tuple[str, float]:
        """
        Predict vessel type for a single vessel.

        Args:
            features: Feature dictionary or array for a single vessel

        Returns:
            Tuple of (vessel_type: str, confidence: float)
            - vessel_type: One of VESSEL_TYPES
            - confidence: Probability of predicted class (0.0 to 1.0)
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

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]

        vessel_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(proba))

        return vessel_type, confidence

    def predict_batch(
        self,
        features_list: List[Union[Dict[str, Any], np.ndarray]]
    ) -> List[Tuple[str, float]]:
        """
        Predict vessel types for multiple vessels.

        Args:
            features_list: List of feature dictionaries or arrays

        Returns:
            List of (vessel_type, confidence) tuples
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

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probas = self.model.predict_proba(X_scaled)

        vessel_types = self.label_encoder.inverse_transform(predictions)
        confidences = np.max(probas, axis=1)

        return list(zip(vessel_types.tolist(), confidences.tolist()))

    def predict_proba(
        self,
        features: Union[Dict[str, Any], np.ndarray]
    ) -> Dict[str, float]:
        """
        Get class probabilities for all vessel types.

        Args:
            features: Feature dictionary or array for a single vessel

        Returns:
            Dictionary mapping each vessel type to its probability
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

        # Scale and predict
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]

        return dict(zip(self.label_encoder.classes_, proba.tolist()))

    def predict_top_k(
        self,
        features: Union[Dict[str, Any], np.ndarray],
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most likely vessel types with probabilities.

        Args:
            features: Feature dictionary or array for a single vessel
            k: Number of top predictions to return

        Returns:
            List of (vessel_type, probability) tuples, sorted by probability descending
        """
        proba_dict = self.predict_proba(features)
        sorted_types = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_types[:k]

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
        y: Union[np.ndarray, List[str]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.

        Args:
            X: Feature matrix or list of feature dicts
            y: True labels (vessel type names)
            verbose: Print evaluation results

        Returns:
            Dictionary with accuracy, precision, recall, F1, confusion matrix
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Prepare data
        X_array, y_true = self._prepare_training_data(X, y)
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Predict
        X_scaled = self.scaler.transform(X_array)
        y_pred = self.model.predict(X_scaled)

        # Compute metrics
        accuracy = (y_pred == y_true).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        if verbose:
            logger.info(f"\nTest Accuracy: {accuracy:.4f}")
            logger.info(f"Precision (weighted): {precision:.4f}")
            logger.info(f"Recall (weighted): {recall:.4f}")
            logger.info(f"F1 Score (weighted): {f1:.4f}")
            logger.info("\nClassification Report:")
            print(classification_report(
                y_true, y_pred,
                target_names=self.label_encoder.classes_
            ))
            logger.info("\nConfusion Matrix:")
            print(cm)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
        }

    @staticmethod
    def rule_based_classify(features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Fall-back rule-based classification using marker profiles and morphology.

        Used when no trained model is available. This implements the biological
        rules for vessel type identification based on marker expression patterns.

        Args:
            features: Vessel features dictionary

        Returns:
            Tuple of (vessel_type, confidence)
        """
        # Extract key features with defaults
        diameter = features.get('outer_diameter_um', 0)
        wall_thickness = features.get('wall_thickness_mean_um', 0)
        sma_wall = features.get('sma_wall_mean', 0) or 0
        cd31_wall = features.get('cd31_wall_mean', 0) or 0
        lyve1_wall = features.get('lyve1_wall_mean', 0) or 0
        ring_completeness = features.get('ring_completeness', 0) or 0

        # Calculate derived features
        wall_thickness_ratio = wall_thickness / max(diameter, 1)

        # Lymphatic detection (LYVE1+ is definitive)
        if lyve1_wall > 50:
            if sma_wall > 50:
                return 'collecting_lymphatic', 0.75
            else:
                return 'lymphatic', 0.75

        # Capillary: very small, no SMA
        if diameter < 10 and sma_wall < 30:
            return 'capillary', 0.70

        # Distinguish artery vs vein vs arteriole based on SMA and wall thickness
        if sma_wall > 80 and ring_completeness > 0.7:
            # Strong SMA+ indicates muscular vessel
            if diameter >= 100 and wall_thickness_ratio > 0.1:
                return 'artery', 0.70
            elif diameter >= 10:
                return 'arteriole', 0.65
        elif sma_wall < 50 and cd31_wall > 30:
            # Low SMA, CD31+ suggests vein
            if diameter > 20:
                return 'vein', 0.60
            else:
                return 'capillary', 0.55

        # Default classification based on size
        if diameter < 10:
            return 'capillary', 0.50
        elif diameter < 100:
            if wall_thickness_ratio > 0.08:
                return 'arteriole', 0.50
            else:
                return 'vein', 0.50
        else:
            if wall_thickness_ratio > 0.08:
                return 'artery', 0.50
            else:
                return 'vein', 0.50

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
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
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
            'model_type': 'vessel_type_classifier',
            'version': '1.0',
            'vessel_types': VESSEL_TYPES,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VesselTypeClassifier':
        """
        Load trained model from file.

        Args:
            path: Path to saved model file

        Returns:
            VesselTypeClassifier instance with loaded model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Verify model type
        if model_data.get('model_type') != 'vessel_type_classifier':
            logger.warning(
                f"Model type mismatch: expected 'vessel_type_classifier', "
                f"got '{model_data.get('model_type')}'"
            )

        # Create instance with saved config
        config = model_data.get('config', {})
        instance = cls(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 20),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            class_weight=config.get('class_weight', 'balanced'),
            random_state=config.get('random_state', 42),
            feature_names=model_data['feature_names'],
        )

        # Restore trained state
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.label_encoder = model_data['label_encoder']
        instance.metrics = model_data.get('metrics', {})
        instance.trained = True

        logger.info(f"Model loaded from: {path}")
        logger.info(f"  Accuracy: {instance.metrics.get('cv_accuracy_mean', 'N/A'):.4f}")
        logger.info(f"  F1 Score: {instance.metrics.get('f1_score', 'N/A'):.4f}")
        logger.info(f"  Features: {len(instance.feature_names)}")
        logger.info(f"  Classes: {len(instance.label_encoder.classes_)}")

        return instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def classify_vessel_type(
    features: Dict[str, Any],
    classifier: Optional[VesselTypeClassifier] = None,
    model_path: Optional[Union[str, Path]] = None
) -> Tuple[str, float]:
    """
    Convenience function to classify a single vessel's type.

    Uses ML classifier if available, falls back to rule-based.

    Args:
        features: Vessel features dictionary
        classifier: Pre-loaded VesselTypeClassifier instance
        model_path: Path to saved model (loads if classifier not provided)

    Returns:
        Tuple of (vessel_type, confidence)
    """
    # Try ML classification
    if classifier is not None:
        try:
            return classifier.predict(features)
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, using rule-based")

    # Try loading model from path
    if model_path is not None:
        try:
            classifier = VesselTypeClassifier.load(model_path)
            return classifier.predict(features)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

    # Fall back to rule-based
    return VesselTypeClassifier.rule_based_classify(features)


def get_vessel_type_description(vessel_type: str) -> Dict[str, Any]:
    """
    Get detailed description of a vessel type.

    Args:
        vessel_type: One of VESSEL_TYPES

    Returns:
        Dictionary with vessel type characteristics
    """
    return VESSEL_TYPE_CHARACTERISTICS.get(vessel_type, {})
