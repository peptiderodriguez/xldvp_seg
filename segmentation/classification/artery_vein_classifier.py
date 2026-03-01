"""
Artery vs Vein classifier.

This is a SECOND-STAGE classifier that runs ONLY on confirmed vessels
(after VesselDetectorRF has classified candidates as true vessels).

The classifier distinguishes between:
- Artery: thick muscular walls, smaller lumen relative to wall, more circular
- Vein: thinner walls, larger lumen relative to wall, often more irregular

Features used:
    - Wall thickness (mean, std, ratio to diameter)
    - Diameter (outer, inner)
    - Wall/lumen ratio
    - Circularity and aspect ratio
    - Position features (optional: depth in tissue, proximity to other structures)
    - Intensity features (wall density, lumen characteristics)

Usage:
    # Training
    classifier = ArteryVeinClassifier()
    classifier.train(X, y)
    classifier.save('artery_vein_classifier.joblib')

    # Inference (after VesselDetectorRF confirms it's a vessel)
    classifier = ArteryVeinClassifier.load('artery_vein_classifier.joblib')
    vessel_type, confidence = classifier.predict(features)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# Features specifically relevant for artery vs vein classification
ARTERY_VEIN_FEATURES = [
    # Wall thickness features (arteries have thicker walls)
    'wall_thickness_mean_um',
    'wall_thickness_median_um',
    'wall_thickness_std_um',
    'wall_thickness_min_um',
    'wall_thickness_max_um',
    # Diameter features
    'outer_diameter_um',
    'inner_diameter_um',
    # Derived ratios (important distinguishing features)
    'wall_lumen_ratio',  # wall_area / lumen_area - higher for arteries
    'wall_thickness_ratio',  # wall_thickness / outer_diameter - higher for arteries
    # Area features
    'wall_area_um2',
    'lumen_area_um2',
    'outer_area_um2',
    # Shape features (arteries tend to be more circular)
    'circularity',
    'aspect_ratio',
    'ring_completeness',
    # Intensity features (wall density)
    'gray_mean',
    'gray_std',
    'intensity_variance',
    # Color features (may indicate staining differences)
    'red_mean',
    'green_mean',
    'blue_mean',
    'saturation_mean',
]

# Minimal feature set when only basic vessel features are available
MINIMAL_ARTERY_VEIN_FEATURES = [
    'wall_thickness_mean_um',
    'outer_diameter_um',
    'inner_diameter_um',
    'circularity',
    'aspect_ratio',
    'gray_mean',
]


class ArteryVeinClassifier:
    """
    Random Forest classifier for artery vs vein classification.

    This classifier should only be applied to CONFIRMED vessels
    (after stage 1 vessel detection).

    Attributes:
        model: Trained RandomForestClassifier
        scaler: StandardScaler for feature normalization
        label_encoder: LabelEncoder for class labels
        feature_names: List of feature names used
        trained: Whether model has been trained
        metrics: Training metrics (accuracy, precision, recall, F1)
    """

    # Class labels
    LABELS = ['artery', 'vein']

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 3,
        class_weight: str = 'balanced',
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize artery/vein classifier.

        Args:
            n_estimators: Number of trees in forest (default 100)
            max_depth: Maximum tree depth (default 10)
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

        # NOTE: StandardScaler is unnecessary for RF (scale-invariant) but kept
        # for backward compatibility with previously trained/serialized models.
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.LABELS)
        self.feature_names = feature_names or ARTERY_VEIN_FEATURES.copy()
        self.trained = False
        self.metrics: Dict[str, Any] = {}

    def _compute_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute derived features that are important for artery/vein classification.

        Args:
            features: Original feature dictionary

        Returns:
            Feature dictionary with derived features added
        """
        features = features.copy()

        # Wall/lumen ratio
        wall_area = features.get('wall_area_um2', 0)
        lumen_area = features.get('lumen_area_um2', 1)  # Avoid division by zero
        features['wall_lumen_ratio'] = wall_area / max(lumen_area, 1)

        # Wall thickness ratio (relative to diameter)
        wall_thickness = features.get('wall_thickness_mean_um', 0)
        outer_diameter = features.get('outer_diameter_um', 1)
        features['wall_thickness_ratio'] = wall_thickness / max(outer_diameter, 1)

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
            y: Labels as array or list of strings ('artery'/'vein')

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
        Train the artery/vein classifier.

        Args:
            X: Feature matrix (N, D) or list of feature dicts
            y: Labels (N,) - strings ('artery', 'vein')
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
            logger.info(f"Training artery/vein classifier with {len(X_array)} samples")
            logger.info(f"Features: {len(self.feature_names)}")
            unique, counts = np.unique(y_encoded, return_counts=True)
            for idx, count in zip(unique, counts):
                label = self.label_encoder.inverse_transform([idx])[0]
                logger.info(f"  {label}: {count} samples")

        # Handle NaN/Inf
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Cross-validation using Pipeline to avoid data leakage
        # (scaler is fit only on training folds, not on validation data)
        if cv_folds > 1 and len(X_array) >= cv_folds:
            if verbose:
                logger.info(f"Running {cv_folds}-fold cross-validation...")

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            cv_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    n_jobs=-1,
                )),
            ])
            cv_scores = cross_val_score(cv_pipeline, X_array, y_encoded, cv=cv, scoring='accuracy')

            if verbose:
                logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        else:
            cv_scores = np.array([0.0])
            if verbose:
                logger.info("Skipping cross-validation (insufficient samples)")

        # Train on full dataset (fit scaler on all data only for final model)
        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, y_encoded)
        self.trained = True

        # Compute training metrics
        y_pred = self.model.predict(X_scaled)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_encoded, y_pred, average='weighted'
        )

        train_accuracy = (y_pred == y_encoded).mean()

        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'n_samples': len(X_array),
            'n_features': len(self.feature_names),
            'class_distribution': {
                label: int(count)
                for label, count in zip(
                    self.label_encoder.classes_,
                    np.bincount(y_encoded)
                )
            },
        }

        if verbose:
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")

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
                Format: {"uid": "artery"/"vein", ...}
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

        # Filter to only artery/vein annotations
        annotations = {
            uid: label for uid, label in annotations.items()
            if label.lower() in ('artery', 'vein')
        }

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

        for uid, label in annotations.items():
            if uid not in detections_by_uid:
                continue

            det = detections_by_uid[uid]
            features = det.get('features', det)

            X_list.append(features)
            y_list.append(label.lower())
            matched_count += 1

        if verbose:
            logger.info(f"Matched {matched_count}/{len(annotations)} artery/vein annotations")

        if not X_list:
            raise ValueError("No matching samples found!")

        return self.train(X_list, y_list, cv_folds=cv_folds, verbose=verbose)

    def predict(
        self,
        features: Union[Dict[str, Any], np.ndarray]
    ) -> Tuple[str, float]:
        """
        Predict artery or vein for a confirmed vessel.

        Args:
            features: Feature dictionary or array for a single vessel

        Returns:
            Tuple of (vessel_type: str, confidence: float)
            - vessel_type: 'artery' or 'vein'
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for multiple vessels.

        Args:
            features_list: List of feature dictionaries or arrays

        Returns:
            Tuple of (vessel_types, confidences)
            - vessel_types: Array of 'artery' or 'vein' strings
            - confidences: Array of confidence scores
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

        return vessel_types, confidences

    def predict_proba(
        self,
        features: Union[Dict[str, Any], np.ndarray]
    ) -> Dict[str, float]:
        """
        Get class probabilities for artery and vein.

        Args:
            features: Feature dictionary or array for a single vessel

        Returns:
            Dictionary {'artery': prob, 'vein': prob}
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
            y: True labels ('artery' or 'vein')
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
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
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
        Fall-back rule-based classification using wall thickness ratios.

        Used when no trained model is available.

        Args:
            features: Vessel features dictionary

        Returns:
            Tuple of (vessel_type, confidence)
        """
        wall_thickness = features.get('wall_thickness_mean_um', 0)
        outer_diameter = features.get('outer_diameter_um', 1)
        lumen_area = features.get('lumen_area_um2', 1)
        wall_area = features.get('wall_area_um2', 0)

        # Calculate ratios
        wall_thickness_ratio = wall_thickness / max(outer_diameter, 1)
        wall_lumen_ratio = wall_area / max(lumen_area, 1)

        # Arteries have thicker walls relative to diameter and lumen
        # Typical threshold: arteries have wall_thickness_ratio > 0.1
        if wall_thickness_ratio > 0.1 or wall_lumen_ratio > 0.5:
            return 'artery', 0.6
        else:
            return 'vein', 0.6

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
            'model_type': 'artery_vein_classifier',
            'version': '1.0',
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ArteryVeinClassifier':
        """
        Load trained model from file.

        Args:
            path: Path to saved model file

        Returns:
            ArteryVeinClassifier instance with loaded model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Verify model type
        if model_data.get('model_type') != 'artery_vein_classifier':
            logger.warning(f"Model type mismatch: expected 'artery_vein_classifier', got '{model_data.get('model_type')}'")

        # Create instance with saved config
        config = model_data.get('config', {})
        instance = cls(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 10),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 3),
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
        f1 = instance.metrics.get('f1_score')
        logger.info(f"  F1 Score: {f1:.4f}" if f1 is not None else "  F1 Score: N/A")
        logger.info(f"  Features: {len(instance.feature_names)}")

        return instance
