"""
ML-based vessel type classifier.

Replaces hard-coded diameter thresholds with a Random Forest classifier
using morphological features (diameter, wall_thickness, circularity, etc.)
to classify vessels as capillary, arteriole, or artery.

Features used:
    - Vessel-specific: diameter, wall_thickness, lumen_area, wall_area, etc.
    - Morphological: circularity, aspect_ratio, solidity, etc.
    - Color/intensity: mean intensities, texture features

Usage:
    # Training
    classifier = VesselClassifier()
    classifier.train(X, y, feature_names)
    classifier.save('model.joblib')

    # Inference
    classifier = VesselClassifier.load('model.joblib')
    predictions, confidence = classifier.predict(features)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


# Vessel type labels
VESSEL_TYPES = ['capillary', 'arteriole', 'artery']

# Core vessel-specific features (most discriminative)
VESSEL_CORE_FEATURES = [
    'outer_diameter_um',
    'inner_diameter_um',
    'wall_thickness_mean_um',
    'wall_thickness_median_um',
    'wall_thickness_std_um',
    'wall_thickness_min_um',
    'wall_thickness_max_um',
    'lumen_area_um2',
    'wall_area_um2',
    'outer_area_um2',
    'aspect_ratio',
    'circularity',
    'ring_completeness',
]

# Extended morphological features (from 22 standard features)
MORPHOLOGICAL_FEATURES = [
    'area',
    'perimeter',
    'solidity',
    'extent',
    'equiv_diameter',
    'red_mean', 'red_std',
    'green_mean', 'green_std',
    'blue_mean', 'blue_std',
    'gray_mean', 'gray_std',
    'hue_mean', 'saturation_mean', 'value_mean',
    'relative_brightness',
    'intensity_variance',
    'dark_fraction',
    'nuclear_complexity',
]

# Combined default features (vessel-specific + morphological = ~35 features)
DEFAULT_FEATURES = VESSEL_CORE_FEATURES + MORPHOLOGICAL_FEATURES

# SAM2 embedding features (256D)
SAM2_FEATURES = [f'sam2_{i}' for i in range(256)]

# ResNet-50 features (2048D)
RESNET_FEATURES = [f'resnet_{i}' for i in range(2048)]

# Full feature set (single-pass): 22 base morph + 13 vessel-specific + 256 SAM2 + 2048 ResNet
# Note: Full pipeline uses masked+context (4096 ResNet, 2048 DINOv2) for up to 6478 total
FULL_FEATURES = DEFAULT_FEATURES + SAM2_FEATURES + RESNET_FEATURES


class VesselClassifier:
    """
    Random Forest classifier for vessel type classification.

    Classifies vessels into three types based on morphological features:
    - capillary: smallest vessels, thin walls
    - arteriole: medium vessels, thicker walls
    - artery: large vessels, thick muscular walls

    Attributes:
        model: Trained RandomForestClassifier
        scaler: StandardScaler for feature normalization
        label_encoder: LabelEncoder for class labels
        feature_names: List of feature names used
        trained: Whether model has been trained
        metrics: Training metrics (accuracy, cv_score, etc.)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = 'balanced',
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize vessel classifier.

        Args:
            n_estimators: Number of trees in forest (default 100)
            max_depth: Maximum tree depth (None for unlimited)
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
        self.feature_names = feature_names or DEFAULT_FEATURES.copy()
        self.trained = False
        self.metrics: Dict[str, Any] = {}

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
            y: Labels as array or list of strings

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
            if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                # First call (training) — fit the encoder
                y_encoded = self.label_encoder.fit_transform(y_array)
            else:
                # Subsequent calls (evaluation) — use existing mapping
                y_encoded = self.label_encoder.transform(y_array)
        else:
            y_encoded = y_array.astype(int)
            # Fit label encoder with known classes if not already fitted
            if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                self.label_encoder.fit(VESSEL_TYPES)

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
        Train the classifier on annotated vessel data.

        Args:
            X: Feature matrix (N, D) or list of feature dicts
            y: Labels (N,) - integers or strings ('capillary', 'arteriole', 'artery')
            feature_names: Names of features in X (required if X is array)
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
            logger.info(f"Training vessel classifier with {len(X_array)} samples")
            logger.info(f"Features: {len(self.feature_names)}")
            unique, counts = np.unique(y_encoded, return_counts=True)
            for idx, count in zip(unique, counts):
                label = self.label_encoder.inverse_transform([idx])[0]
                logger.info(f"  {label}: {count} samples")

        # Handle NaN/Inf
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Cross-validation using Pipeline to avoid data leakage
        # (scaler is fit only on training folds, not on validation data)
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

        # Train on full dataset (fit scaler on all data only for final model)
        X_scaled = self.scaler.fit_transform(X_array)
        self.model.fit(X_scaled, y_encoded)
        self.trained = True

        # Compute training metrics
        y_pred = self.model.predict(X_scaled)
        train_accuracy = (y_pred == y_encoded).mean()

        self.metrics = {
            'train_accuracy': float(train_accuracy),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'n_samples': len(X_array),
            'n_features': len(self.feature_names),
            'class_distribution': dict(zip(
                self.label_encoder.classes_.tolist(),
                [int(c) for c in np.bincount(y_encoded, minlength=len(self.label_encoder.classes_))]
            )),
        }

        if verbose:
            logger.info(f"Training accuracy: {train_accuracy:.4f}")

        return self.metrics

    def predict(
        self,
        X: Union[np.ndarray, List[Dict], Dict],
        return_confidence: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Predict vessel types with confidence scores.

        Args:
            X: Feature matrix (N, D), list of feature dicts, or single dict
            return_confidence: Whether to return confidence scores

        Returns:
            If return_confidence=True: (predictions, confidence)
                - predictions: Array of vessel type strings
                - confidence: Array of confidence scores [0, 1]
            If return_confidence=False: predictions only

        Raises:
            RuntimeError: If model not trained
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Handle single dict input
        single_input = False
        if isinstance(X, dict):
            X = [X]
            single_input = True

        # Convert dict input to array
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X_array = np.array([
                self._extract_features_from_dict(d)
                for d in X
            ])
        else:
            X_array = np.asarray(X, dtype=np.float32)

        # Handle NaN/Inf
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Scale features
        X_scaled = self.scaler.transform(X_array)

        # Predict
        y_pred = self.model.predict(X_scaled)
        predictions = self.label_encoder.inverse_transform(y_pred)

        if return_confidence:
            # Get probability of predicted class as confidence
            probas = self.model.predict_proba(X_scaled)
            confidence = np.max(probas, axis=1)

            if single_input:
                return predictions[0], confidence[0]
            return predictions, confidence
        else:
            if single_input:
                return predictions[0]
            return predictions

    def predict_proba(
        self,
        X: Union[np.ndarray, List[Dict], Dict]
    ) -> np.ndarray:
        """
        Get class probabilities for each vessel type.

        Args:
            X: Feature matrix (N, D), list of feature dicts, or single dict

        Returns:
            Probability matrix (N, 3) with columns [capillary, arteriole, artery]
            or (3,) for single input
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Handle single dict input
        single_input = False
        if isinstance(X, dict):
            X = [X]
            single_input = True

        # Convert dict input to array
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X_array = np.array([
                self._extract_features_from_dict(d)
                for d in X
            ])
        else:
            X_array = np.asarray(X, dtype=np.float32)

        # Handle NaN/Inf
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Scale and predict
        X_scaled = self.scaler.transform(X_array)
        probas = self.model.predict_proba(X_scaled)

        if single_input:
            return probas[0]
        return probas

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
            y: True labels
            verbose: Print evaluation results

        Returns:
            Dictionary with accuracy, confusion matrix, classification report
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
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        if verbose:
            logger.info(f"\nTest Accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            print(classification_report(
                y_true, y_pred,
                target_names=self.label_encoder.classes_
            ))
            logger.info("\nConfusion Matrix:")
            print(cm)

        return {
            'accuracy': float(accuracy),
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
            'version': '1.0',
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VesselClassifier':
        """
        Load trained model from file.

        Args:
            path: Path to saved model file

        Returns:
            VesselClassifier instance with loaded model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Create instance with saved config
        config = model_data.get('config', {})
        instance = cls(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth'),
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
        logger.info(f"  Features: {len(instance.feature_names)}")

        return instance

    @staticmethod
    def rule_based_classify(features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Fall-back rule-based classification using diameter thresholds.

        This is the original hard-coded classification that the ML model replaces.
        Used when no trained model is available.

        Args:
            features: Vessel features dictionary

        Returns:
            Tuple of (vessel_type, confidence)
        """
        diameter = features.get('outer_diameter_um', 0)

        if diameter < 10:
            return 'capillary', 0.8
        elif diameter < 100:
            return 'arteriole', 0.7
        else:
            return 'artery', 0.6


def classify_vessel(
    features: Dict[str, Any],
    classifier: Optional[VesselClassifier] = None,
    model_path: Optional[Union[str, Path]] = None
) -> Tuple[str, float]:
    """
    Convenience function to classify a single vessel.

    Uses ML classifier if available, falls back to rule-based.

    Args:
        features: Vessel features dictionary
        classifier: Pre-loaded VesselClassifier instance
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
            classifier = VesselClassifier.load(model_path)
            return classifier.predict(features)
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")

    # Fall back to rule-based
    return VesselClassifier.rule_based_classify(features)
