"""Tests for BaseVesselClassifier from xldvp_seg/classification/base.py."""

import numpy as np
import pytest

from xldvp_seg.classification.base import BaseVesselClassifier


class ConcreteClassifier(BaseVesselClassifier):
    """Minimal concrete subclass for testing."""

    MODEL_TYPE = "test"

    def train(self, *a, **kw):
        pass


class TestBaseVesselClassifierAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseVesselClassifier()


class TestBaseVesselClassifierDefaults:
    def test_default_attributes(self):
        clf = ConcreteClassifier()
        assert clf.n_estimators == 100
        assert clf.max_depth is None
        assert clf.min_samples_split == 5
        assert clf.min_samples_leaf == 2
        assert clf.class_weight == "balanced"
        assert clf.random_state == 42
        assert clf.feature_names == []
        assert clf.trained is False
        assert clf.use_scaler is True
        assert clf.scaler is not None
        assert clf.MODEL_TYPE == "test"

    def test_custom_params(self):
        clf = ConcreteClassifier(n_estimators=50, max_depth=10, feature_names=["a", "b"])
        assert clf.n_estimators == 50
        assert clf.max_depth == 10
        assert clf.feature_names == ["a", "b"]


class TestBaseVesselClassifierSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        clf = ConcreteClassifier(feature_names=["f1", "f2", "f3"])

        # Fit on trivial data
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = np.array([0] * 10 + [1] * 10)
        clf.model.fit(X, y)
        clf.label_encoder.fit(["class_a", "class_b"])
        if clf.scaler is not None:
            clf.scaler.fit(X)
        clf.trained = True

        path = tmp_path / "test.joblib"
        clf.save(path)

        loaded = ConcreteClassifier.load(path)
        assert loaded.feature_names == ["f1", "f2", "f3"]
        assert loaded.MODEL_TYPE == "test"
        assert loaded.trained is True

    def test_save_untrained_raises(self, tmp_path):
        from xldvp_seg.exceptions import ClassificationError

        clf = ConcreteClassifier()
        with pytest.raises(ClassificationError):
            clf.save(tmp_path / "bad.joblib")


class TestBaseVesselClassifierFeatureImportance:
    def _make_trained(self):
        clf = ConcreteClassifier(feature_names=["area", "circularity", "solidity"])
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 3))
        y = np.array([0] * 15 + [1] * 15)
        clf.model.fit(X, y)
        clf.trained = True
        return clf

    def test_get_feature_importance(self):
        clf = self._make_trained()
        imp = clf.get_feature_importance()
        assert isinstance(imp, dict)
        assert set(imp.keys()) == {"area", "circularity", "solidity"}
        assert all(isinstance(v, float) for v in imp.values())

    def test_get_top_features(self):
        clf = self._make_trained()
        top = clf.get_top_features(n=2)
        assert isinstance(top, list)
        assert len(top) == 2
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)
        # Sorted descending by importance
        assert top[0][1] >= top[1][1]

    def test_feature_importance_untrained_raises(self):
        from xldvp_seg.exceptions import ClassificationError

        clf = ConcreteClassifier(feature_names=["a"])
        with pytest.raises(ClassificationError):
            clf.get_feature_importance()
