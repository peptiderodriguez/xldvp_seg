"""Regression tests for Phase A.1 fix: joblib.load exception coverage.

Truncated pickle files raise pickle.UnpicklingError which was NOT in the
original except tuple; my first fix broadened the tuple but still missed
UnpicklingError specifically. This test pins the behavior.
"""

import pickle

import pytest

from xldvp_seg.exceptions import ClassificationError, DataLoadError
from xldvp_seg.utils.detection_utils import load_rf_classifier


class TestTruncatedPickle:
    def test_truncated_pickle_wraps_as_dataload_error(self, tmp_path):
        """Feed a truncated pickle to load_rf_classifier → DataLoadError."""
        path = tmp_path / "corrupt.pkl"
        # Truncate the pickle header — deterministically triggers
        # pickle.UnpicklingError on load.
        path.write_bytes(pickle.dumps({"ok": 1})[:5])

        with pytest.raises(DataLoadError, match="Failed to load"):
            load_rf_classifier(str(path))

    def test_random_bytes_wraps_as_dataload_error(self, tmp_path):
        """Random non-pickle bytes also wrap cleanly."""
        path = tmp_path / "garbage.pkl"
        path.write_bytes(b"\x00\x01\x02not-a-pickle-at-all")
        with pytest.raises(DataLoadError):
            load_rf_classifier(str(path))

    def test_base_classifier_load_wraps_unpickling_error(self, tmp_path):
        """Concrete BaseVesselClassifier subclass also wraps pickle errors."""
        from xldvp_seg.classification.vessel_detector_rf import VesselDetectorRF

        path = tmp_path / "corrupt.pkl"
        path.write_bytes(pickle.dumps({"ok": 1})[:5])

        with pytest.raises(ClassificationError, match="Failed to load"):
            VesselDetectorRF.load(path)
