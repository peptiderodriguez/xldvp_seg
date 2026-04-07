"""Base class for RF-based vessel classifiers.

Provides shared infrastructure for all vessel classifier variants:
- RF parameter storage and model creation
- Feature extraction with coerce/derived-feature hooks
- joblib save/load serialization
- Feature importance ranking

Subclasses implement predict(), train(), evaluate(), and optionally override
_coerce_value() and _compute_derived_features() for type-specific behavior.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from xldvp_seg.exceptions import ClassificationError
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


class BaseVesselClassifier(abc.ABC):
    """Abstract base for sklearn RF vessel classifiers.

    Provides shared infrastructure: RF parameter storage, feature extraction,
    joblib save/load, feature importance. Subclasses implement predict/train/evaluate.

    Class attributes:
        MODEL_TYPE: Identifier string written into serialized model files.
            Subclasses MUST set this to a unique, non-empty string.
    """

    MODEL_TYPE: str = ""  # Subclasses MUST override

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """Train the classifier. Subclasses must implement."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        feature_names: list[str] | None = None,
        default_features: list[str] | None = None,
        use_scaler: bool = True,
        label_classes: list[str] | None = None,
    ):
        """Initialize the classifier.

        Args:
            n_estimators: Number of trees in forest.
            max_depth: Maximum tree depth (None for unlimited).
            min_samples_split: Min samples to split internal node.
            min_samples_leaf: Min samples at leaf node.
            class_weight: 'balanced' to handle class imbalance.
            random_state: Random seed for reproducibility.
            feature_names: List of feature names to use (None uses default_features).
            default_features: Default feature list when feature_names is None.
            use_scaler: Whether to use StandardScaler (True for most classifiers,
                False for VesselDetectorRF which skips scaling).
            label_classes: If provided, pre-fit the LabelEncoder with these classes.
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

        self.use_scaler = use_scaler
        if use_scaler:
            # NOTE: StandardScaler is unnecessary for RF (scale-invariant) but kept
            # for backward compatibility with previously trained/serialized models.
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.label_encoder = LabelEncoder()
        if label_classes is not None:
            self.label_encoder.fit(label_classes)

        self.feature_names = feature_names or (default_features.copy() if default_features else [])
        self.trained = False
        self.metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Feature extraction hooks (override in subclasses as needed)
    # ------------------------------------------------------------------

    def _coerce_value(self, v: Any) -> float:
        """Convert a single feature value to float.

        Override in subclasses to handle bool, inf, or other special values.
        The base implementation handles None and list/tuple; everything else
        is cast via ``float()``.
        """
        if v is None:
            return 0.0
        if isinstance(v, (list, tuple)):
            return 0.0
        return float(v)

    def _compute_derived_features(self, features_dict: dict[str, Any]) -> dict[str, Any]:
        """Hook for subclasses to add derived features before extraction.

        Called by ``_extract_features_from_dict`` before iterating over
        feature names. The base implementation returns the dict unchanged.

        Subclasses that override this MUST return a *copy* of the input dict
        (to avoid mutating the caller's data).
        """
        return features_dict

    def _extract_features_from_dict(
        self, features_dict: dict[str, Any], feature_names: list[str] | None = None
    ) -> np.ndarray:
        """Extract feature vector from a features dictionary.

        Applies ``_compute_derived_features`` first, then iterates over
        ``feature_names`` (falling back to ``self.feature_names``), calling
        ``_coerce_value`` on each value.

        Args:
            features_dict: Dictionary of features from vessel detection.
            feature_names: List of feature names to extract (default: self.feature_names).

        Returns:
            1D numpy array of feature values (float32).
        """
        if feature_names is None:
            feature_names = self.feature_names

        # Let subclasses inject derived features
        features_dict = self._compute_derived_features(features_dict)

        values = []
        for name in feature_names:
            val = features_dict.get(name, 0)
            values.append(self._coerce_value(val))

        return np.array(values, dtype=np.float32)

    # ------------------------------------------------------------------
    # Feature importance (identical across all 4 classifiers)
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance rankings.

        Returns:
            Dictionary mapping feature names to importance scores.

        Raises:
            RuntimeError: If model not trained.
        """
        if not self.trained:
            raise ClassificationError("Model not trained. Call train() or load() first.")

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features.

        Args:
            n: Number of features to return.

        Returns:
            List of (feature_name, importance) tuples sorted by importance.
        """
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    # ------------------------------------------------------------------
    # Serialization (save / load)
    # ------------------------------------------------------------------

    def _save_extra(self) -> dict[str, Any]:
        """Hook for subclasses to add extra keys to the serialized dict.

        Override to persist additional state (e.g., vessel_types list).
        The returned dict is merged into the top-level model_data.
        """
        return {}

    def _load_extra(self, model_data: dict[str, Any]) -> None:
        """Hook for subclasses to restore extra state from a serialized dict.

        Override to restore anything added by ``_save_extra``.
        """
        pass

    def save(self, path: str | Path) -> None:
        """Save trained model to file.

        Args:
            path: Output file path (use .joblib extension).

        Raises:
            RuntimeError: If model not trained.
        """
        if not self.trained:
            raise ClassificationError("Cannot save untrained model. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data: dict[str, Any] = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
            },
            "model_type": self.MODEL_TYPE,
            "version": "1.0",
        }

        # Add scaler and label_encoder only when used
        if self.use_scaler:
            model_data["scaler"] = self.scaler
        if hasattr(self.label_encoder, "classes_") and len(self.label_encoder.classes_) > 0:
            model_data["label_encoder"] = self.label_encoder

        # Let subclasses add extra keys
        model_data.update(self._save_extra())

        joblib.dump(model_data, path)
        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> BaseVesselClassifier:
        """Load trained model from file.

        Args:
            path: Path to saved model file.

        Returns:
            Classifier instance with loaded model.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If file cannot be deserialized.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            model_data = joblib.load(path)
        except (EOFError, ModuleNotFoundError) as e:
            raise ClassificationError(f"Failed to load classifier from {path}: {e}") from e

        # Verify model type
        expected = cls.MODEL_TYPE
        actual = model_data.get("model_type")
        if expected and actual != expected:
            logger.warning(f"Model type mismatch: expected '{expected}', got '{actual}'")

        # Create instance with saved config
        config = model_data.get("config", {})
        instance = cls(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth"),
            min_samples_split=config.get("min_samples_split", 5),
            min_samples_leaf=config.get("min_samples_leaf", 2),
            class_weight=config.get("class_weight", "balanced"),
            random_state=config.get("random_state", 42),
            feature_names=model_data["feature_names"],
        )

        # Restore trained state
        instance.model = model_data["model"]
        if "scaler" in model_data and instance.use_scaler:
            instance.scaler = model_data["scaler"]
        if "label_encoder" in model_data:
            instance.label_encoder = model_data["label_encoder"]
        instance.metrics = model_data.get("metrics", {})
        instance.trained = True

        # Let subclasses restore extra state
        instance._load_extra(model_data)

        logger.info(f"Model loaded from: {path}")
        logger.info(f"  Features: {len(instance.feature_names)}")

        return instance
