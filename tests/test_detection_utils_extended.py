"""Extended tests for xldvp_seg.utils.detection_utils.

Covers:
- safe_to_uint8 (all dtypes and edge cases)
- extract_positions_um (all resolution paths and edges)
- load_rf_classifier (missing file, format handling)
"""

import numpy as np
import pytest

from xldvp_seg.utils.detection_utils import (
    extract_positions_um,
    load_rf_classifier,
    safe_to_uint8,
)

# ---------------------------------------------------------------------------
# safe_to_uint8
# ---------------------------------------------------------------------------


class TestSafeToUint8:
    def test_uint8_passthrough(self):
        """uint8 input should be returned unchanged (same object)."""
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = safe_to_uint8(arr)
        assert result is arr
        assert result.dtype == np.uint8

    def test_uint16_divided(self):
        """uint16 input should be divided by 256."""
        arr = np.array([0, 256, 65535], dtype=np.uint16)
        result = safe_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == 255

    def test_float_01_scaled(self):
        """Float [0, 1] input should be scaled to [0, 255]."""
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = safe_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[1] == 127 or result[1] == 128  # rounding
        assert result[2] == 255

    def test_float_above_1_clipped(self):
        """Float > 1.0 should be clipped to 255."""
        arr = np.array([0.0, 100.0, 300.0], dtype=np.float32)
        result = safe_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0] == 0
        assert result[1] == 100
        assert result[2] == 255

    def test_float_negative_clipped(self):
        """Negative float values should be clipped to 0."""
        arr = np.array([-10.0, 0.0, 128.0], dtype=np.float32)
        result = safe_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[0] == 0

    def test_empty_array(self):
        """Empty array should not crash."""
        arr = np.array([], dtype=np.float32)
        result = safe_to_uint8(arr)
        assert result.dtype == np.uint8
        assert len(result) == 0

    def test_all_zeros(self):
        """All-zero array should return all zeros."""
        arr = np.zeros(10, dtype=np.float32)
        result = safe_to_uint8(arr)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 0)

    def test_float64_input(self):
        """float64 should also work (converted to float32 internally)."""
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = safe_to_uint8(arr)
        assert result.dtype == np.uint8
        assert result[2] == 255

    def test_2d_array(self):
        """2D arrays should work without issue."""
        arr = np.array([[0, 128, 255], [50, 100, 200]], dtype=np.uint8)
        result = safe_to_uint8(arr)
        assert result is arr  # uint8 passthrough


# ---------------------------------------------------------------------------
# extract_positions_um
# ---------------------------------------------------------------------------


class TestExtractPositionsUm:
    def test_global_center_um(self):
        """Detection with global_center_um should be used directly."""
        dets = [
            {"global_center_um": [100.0, 200.0], "features": {}},
            {"global_center_um": [300.0, 400.0], "features": {}},
        ]
        positions, px = extract_positions_um(dets)
        assert positions.shape == (2, 2)
        np.testing.assert_allclose(positions[0], [100.0, 200.0])
        np.testing.assert_allclose(positions[1], [300.0, 400.0])

    def test_global_center_times_pixel_size(self):
        """Detection with global_center + pixel_size should be multiplied."""
        dets = [{"global_center": [200, 400], "features": {}}]
        positions, px = extract_positions_um(dets, pixel_size_um=0.5)
        np.testing.assert_allclose(positions[0], [100.0, 200.0])

    def test_global_x_y(self):
        """Detection with global_x/global_y should be resolved."""
        dets = [
            {
                "global_x": 200,
                "global_y": 400,
                "features": {"pixel_size_um": 0.5},
            },
        ]
        positions, px = extract_positions_um(dets)
        np.testing.assert_allclose(positions[0], [100.0, 200.0])

    def test_nan_global_center_um_skipped(self):
        """NaN in global_center_um should skip the detection."""
        dets = [
            {"global_center_um": [float("nan"), 200.0], "features": {}},
            {"global_center_um": [100.0, 200.0], "features": {}},
        ]
        positions, px = extract_positions_um(dets)
        assert positions.shape == (1, 2)

    def test_return_indices(self):
        """return_indices=True should give valid indices matching positions."""
        dets = [
            {"global_center_um": [100.0, 200.0], "features": {}},
            {"features": {}},  # no position
            {"global_center_um": [300.0, 400.0], "features": {}},
        ]
        positions, px, indices = extract_positions_um(dets, return_indices=True)
        assert positions.shape == (2, 2)
        assert indices == [0, 2]

    def test_empty_detections(self):
        """Empty detections should return empty array."""
        positions, px = extract_positions_um([])
        assert positions.shape == (0, 2)

    def test_infer_pixel_size_from_area(self):
        """pixel_size should be inferred from area/area_um2 ratio."""
        dets = [
            {
                "global_center": [100, 200],
                "features": {"area": 100, "area_um2": 25.0},
            },
        ]
        positions, px = extract_positions_um(dets)
        # px = sqrt(25/100) = 0.5
        assert px == pytest.approx(0.5, abs=0.01)
        np.testing.assert_allclose(positions[0], [50.0, 100.0])

    def test_no_position_info_skipped(self):
        """Detections without any position info should be skipped."""
        dets = [{"features": {"area": 100}}]
        positions, px = extract_positions_um(dets, pixel_size_um=0.5)
        assert positions.shape == (0, 2)

    def test_global_center_um_in_features(self):
        """global_center_um inside features dict should also work."""
        dets = [{"features": {"global_center_um": [50.0, 100.0]}}]
        positions, px = extract_positions_um(dets)
        assert positions.shape == (1, 2)
        np.testing.assert_allclose(positions[0], [50.0, 100.0])


# ---------------------------------------------------------------------------
# load_rf_classifier
# ---------------------------------------------------------------------------


class TestLoadRfClassifier:
    def test_missing_file_raises(self):
        """Missing file should raise DataLoadError (also an IOError)."""
        from xldvp_seg.exceptions import DataLoadError

        with pytest.raises(DataLoadError, match="Failed to load"):
            load_rf_classifier("/nonexistent/path/model.pkl")

    def test_dict_format(self, tmp_path):
        """Dict format with model key should be loaded correctly."""
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        # Fit on dummy data
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        rf.fit(X, y)

        model_path = tmp_path / "model.pkl"
        joblib.dump(
            {
                "model": rf,
                "feature_names": ["f1", "f2", "f3"],
                "accuracy": 0.95,
            },
            model_path,
        )

        result = load_rf_classifier(str(model_path))
        assert result["type"] == "rf"
        assert result["feature_names"] == ["f1", "f2", "f3"]
        assert hasattr(result["pipeline"], "predict")

    def test_pipeline_format(self, tmp_path):
        """sklearn Pipeline format should be loaded correctly."""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        pipe = Pipeline([("scaler", StandardScaler()), ("rf", rf)])
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        pipe.fit(X, y)

        model_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipe, model_path)

        # Also save feature names sidecar
        import json

        with open(tmp_path / "classifier_feature_names.json", "w") as f:
            json.dump(["f1", "f2", "f3"], f)

        result = load_rf_classifier(str(model_path))
        assert result["type"] == "rf"
        assert result["feature_names"] == ["f1", "f2", "f3"]

    def test_dict_without_model_key_raises(self, tmp_path):
        """Dict without model/classifier key should raise DataLoadError."""
        import joblib

        from xldvp_seg.exceptions import DataLoadError

        model_path = tmp_path / "bad_model.pkl"
        joblib.dump({"feature_names": ["f1"]}, model_path)

        with pytest.raises(DataLoadError, match="no 'model' or 'classifier' key"):
            load_rf_classifier(str(model_path))
