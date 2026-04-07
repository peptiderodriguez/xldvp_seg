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
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xldvp_seg.classification.base import BaseVesselClassifier
from xldvp_seg.exceptions import ClassificationError
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# Features specifically relevant for artery vs vein classification
ARTERY_VEIN_FEATURES = [
    # Wall thickness features (arteries have thicker walls)
    "wall_thickness_mean_um",
    "wall_thickness_median_um",
    "wall_thickness_std_um",
    "wall_thickness_min_um",
    "wall_thickness_max_um",
    # Diameter features
    "outer_diameter_um",
    "inner_diameter_um",
    # Derived ratios (important distinguishing features)
    "wall_lumen_ratio",  # wall_area / lumen_area - higher for arteries
    "wall_thickness_ratio",  # wall_thickness / outer_diameter - higher for arteries
    # Area features
    "wall_area_um2",
    "lumen_area_um2",
    "outer_area_um2",
    # Shape features (arteries tend to be more circular)
    "circularity",
    "aspect_ratio",
    "ring_completeness",
    # Intensity features (wall density)
    "gray_mean",
    "gray_std",
    "intensity_variance",
    # Color features (may indicate staining differences)
    "red_mean",
    "green_mean",
    "blue_mean",
    "saturation_mean",
]

# Minimal feature set when only basic vessel features are available
MINIMAL_ARTERY_VEIN_FEATURES = [
    "wall_thickness_mean_um",
    "outer_diameter_um",
    "inner_diameter_um",
    "circularity",
    "aspect_ratio",
    "gray_mean",
]


class ArteryVeinClassifier(BaseVesselClassifier):
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
    LABELS = ["artery", "vein"]

    MODEL_TYPE = "artery_vein_classifier"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 3,
        class_weight: str = "balanced",
        random_state: int = 42,
        feature_names: list[str] | None = None,
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
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            feature_names=feature_names,
            default_features=ARTERY_VEIN_FEATURES,
            use_scaler=True,
            label_classes=self.LABELS,
        )

    def _coerce_value(self, v: Any) -> float:
        """Handle bool values in addition to base coercion."""
        if v is None:
            return 0.0
        if isinstance(v, (list, tuple)):
            return 0.0
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        return float(v)

    def _compute_derived_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Compute derived features that are important for artery/vein classification.

        Args:
            features: Original feature dictionary

        Returns:
            Feature dictionary with derived features added
        """
        features = features.copy()

        # Wall/lumen ratio
        wall_area = features.get("wall_area_um2", 0)
        lumen_area = features.get("lumen_area_um2", 1)  # Avoid division by zero
        features["wall_lumen_ratio"] = wall_area / max(lumen_area, 1)

        # Wall thickness ratio (relative to diameter)
        wall_thickness = features.get("wall_thickness_mean_um", 0)
        outer_diameter = features.get("outer_diameter_um", 1)
        features["wall_thickness_ratio"] = wall_thickness / max(outer_diameter, 1)

        return features

    def _prepare_training_data(
        self,
        X: np.ndarray | list[dict],
        y: np.ndarray | list[str],
        feature_names: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            X_array = np.array([self._extract_features_from_dict(d, feature_names) for d in X])
        else:
            X_array = np.asarray(X, dtype=np.float32)

        # Handle string labels
        y_array = np.asarray(y)
        if y_array.dtype.kind in ("U", "S", "O"):  # String types
            y_encoded = self.label_encoder.transform(y_array)
        else:
            y_encoded = y_array.astype(int)

        return X_array, y_encoded

    def train(
        self,
        X: np.ndarray | list[dict],
        y: np.ndarray | list[str],
        feature_names: list[str] | None = None,
        cv_folds: int = 5,
        verbose: bool = True,
    ) -> dict[str, Any]:
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
            cv_pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=self.n_estimators,
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            min_samples_leaf=self.min_samples_leaf,
                            class_weight=self.class_weight,
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            cv_scores = cross_val_score(cv_pipeline, X_array, y_encoded, cv=cv, scoring="accuracy")

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
            y_encoded, y_pred, average="weighted"
        )

        train_accuracy = (y_pred == y_encoded).mean()

        self.metrics = {
            "train_accuracy": float(train_accuracy),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "n_samples": len(X_array),
            "n_features": len(self.feature_names),
            "class_distribution": {
                label: int(count)
                for label, count in zip(self.label_encoder.classes_, np.bincount(y_encoded))
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
        annotations_path: str | Path,
        detections_path: str | Path,
        cv_folds: int = 5,
        verbose: bool = True,
    ) -> dict[str, Any]:
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
        if "annotations" in annotations_data:
            annotations = annotations_data["annotations"]
        else:
            annotations = annotations_data

        # Filter to only artery/vein annotations
        annotations = {
            uid: label for uid, label in annotations.items() if label.lower() in ("artery", "vein")
        }

        # Load detections
        with open(detections_path) as f:
            detections_list = json.load(f)

        # Build detection index by UID
        detections_by_uid = {}
        for d in detections_list:
            if "uid" in d:
                detections_by_uid[d["uid"]] = d
            if "id" in d:
                detections_by_uid[d["id"]] = d

        # Match annotations to detections
        X_list = []
        y_list = []
        matched_count = 0

        for uid, label in annotations.items():
            if uid not in detections_by_uid:
                continue

            det = detections_by_uid[uid]
            features = det.get("features", det)

            X_list.append(features)
            y_list.append(label.lower())
            matched_count += 1

        if verbose:
            logger.info(f"Matched {matched_count}/{len(annotations)} artery/vein annotations")

        if not X_list:
            raise ValueError("No matching samples found!")

        return self.train(X_list, y_list, cv_folds=cv_folds, verbose=verbose)

    def predict(self, features: dict[str, Any] | np.ndarray) -> tuple[str, float]:
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
            raise ClassificationError("Model not trained. Call train() or load() first.")

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
        self, features_list: list[dict[str, Any] | np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
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
            raise ClassificationError("Model not trained. Call train() or load() first.")

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

    def predict_proba(self, features: dict[str, Any] | np.ndarray) -> dict[str, float]:
        """
        Get class probabilities for artery and vein.

        Args:
            features: Feature dictionary or array for a single vessel

        Returns:
            Dictionary {'artery': prob, 'vein': prob}
        """
        if not self.trained:
            raise ClassificationError("Model not trained. Call train() or load() first.")

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

    def evaluate(
        self, X: np.ndarray | list[dict], y: np.ndarray | list[str], verbose: bool = True
    ) -> dict[str, Any]:
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
            raise ClassificationError("Model not trained. Call train() or load() first.")

        # Prepare data
        X_array, y_true = self._prepare_training_data(X, y)
        X_array = np.nan_to_num(X_array, nan=0, posinf=0, neginf=0)

        # Predict
        X_scaled = self.scaler.transform(X_array)
        y_pred = self.model.predict(X_scaled)

        # Compute metrics
        accuracy = (y_pred == y_true).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True
        )

        if verbose:
            logger.info(f"\nTest Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info("\nClassification Report:")
            logger.info(
                "\n%s",
                classification_report(y_true, y_pred, target_names=self.label_encoder.classes_),
            )
            logger.info("\nConfusion Matrix:")
            logger.info("\n%s", cm)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    @staticmethod
    def rule_based_classify(features: dict[str, Any]) -> tuple[str, float]:
        """
        Fall-back rule-based classification using wall thickness ratios.

        Used when no trained model is available.

        Args:
            features: Vessel features dictionary

        Returns:
            Tuple of (vessel_type, confidence)
        """
        wall_thickness = features.get("wall_thickness_mean_um", 0)
        outer_diameter = features.get("outer_diameter_um", 1)
        lumen_area = features.get("lumen_area_um2", 1)
        wall_area = features.get("wall_area_um2", 0)

        # Calculate ratios
        wall_thickness_ratio = wall_thickness / max(outer_diameter, 1)
        wall_lumen_ratio = wall_area / max(lumen_area, 1)

        # Arteries have thicker walls relative to diameter and lumen
        # Typical threshold: arteries have wall_thickness_ratio > 0.1
        if wall_thickness_ratio > 0.1 or wall_lumen_ratio > 0.5:
            return "artery", 0.6
        else:
            return "vein", 0.6

    @classmethod
    def load(cls, path: str | Path) -> "ArteryVeinClassifier":
        """
        Load trained model from file.

        Args:
            path: Path to saved model file

        Returns:
            ArteryVeinClassifier instance with loaded model
        """
        instance = super().load(path)
        f1 = instance.metrics.get("f1_score")
        logger.info(f"  F1 Score: {f1:.4f}" if f1 is not None else "  F1 Score: N/A")
        return instance
