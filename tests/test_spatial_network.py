"""Tests for xldvp_seg.analysis.spatial_network.

Covers:
- parse_marker_filter with numeric and string patterns
- _get_positions_um with various detection formats
- Edge cases: empty detections, missing positions
"""

import numpy as np
import pytest

from xldvp_seg.analysis.spatial_network import (
    _get_positions_um,
    parse_marker_filter,
)


class TestParseMarkerFilter:
    def test_numeric_greater_than(self):
        """'ch0_mean>100' should return a callable that filters correctly."""
        pred = parse_marker_filter("ch0_mean>100")
        det_above = {"features": {"ch0_mean": 150.0}}
        det_below = {"features": {"ch0_mean": 50.0}}
        assert pred(det_above) is True
        assert pred(det_below) is False

    def test_numeric_greater_equal(self):
        pred = parse_marker_filter("ch0_mean>=100")
        assert pred({"features": {"ch0_mean": 100.0}}) is True
        assert pred({"features": {"ch0_mean": 99.9}}) is False

    def test_numeric_less_than(self):
        pred = parse_marker_filter("ch0_mean<100")
        assert pred({"features": {"ch0_mean": 50.0}}) is True
        assert pred({"features": {"ch0_mean": 150.0}}) is False

    def test_numeric_equal(self):
        pred = parse_marker_filter("ch0_mean==100")
        assert pred({"features": {"ch0_mean": 100.0}}) is True
        assert pred({"features": {"ch0_mean": 99.0}}) is False

    def test_numeric_not_equal(self):
        pred = parse_marker_filter("ch0_mean!=100")
        assert pred({"features": {"ch0_mean": 99.0}}) is True
        assert pred({"features": {"ch0_mean": 100.0}}) is False

    def test_string_equality(self):
        """'SMA_class==positive' should match string features."""
        pred = parse_marker_filter("SMA_class==positive")
        det_pos = {"features": {"SMA_class": "positive"}}
        det_neg = {"features": {"SMA_class": "negative"}}
        assert pred(det_pos) is True
        assert pred(det_neg) is False

    def test_string_inequality(self):
        pred = parse_marker_filter("SMA_class!=negative")
        assert pred({"features": {"SMA_class": "positive"}}) is True
        assert pred({"features": {"SMA_class": "negative"}}) is False

    def test_missing_feature_returns_false(self):
        pred = parse_marker_filter("ch0_mean>100")
        assert pred({"features": {}}) is False  # defaults to 0.0

    def test_missing_feature_none_returns_false(self):
        pred = parse_marker_filter("SMA_class==positive")
        assert pred({"features": {}}) is False

    def test_invalid_filter_raises(self):
        """Unparseable filter strings should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_marker_filter("invalid filter string!!!")

    def test_negative_threshold(self):
        pred = parse_marker_filter("score>-0.5")
        assert pred({"features": {"score": 0.0}}) is True
        assert pred({"features": {"score": -1.0}}) is False


class TestGetPositionsUm:
    def test_with_global_center_um(self):
        """Detections with global_center_um should be resolved directly."""
        dets = [
            {"global_center_um": [100.0, 200.0], "features": {}},
            {"global_center_um": [300.0, 400.0], "features": {}},
        ]
        positions, valid_indices = _get_positions_um(dets, pixel_size=0.5)
        assert positions.shape == (2, 2)
        assert len(valid_indices) == 2
        np.testing.assert_allclose(positions[0], [100.0, 200.0])

    def test_with_global_center_and_pixel_size(self):
        """Detections with global_center + pixel_size should be multiplied."""
        dets = [
            {"global_center": [200, 400], "features": {}},
        ]
        positions, valid_indices = _get_positions_um(dets, pixel_size=0.5)
        assert positions.shape == (1, 2)
        np.testing.assert_allclose(positions[0], [100.0, 200.0])

    def test_empty_detections(self):
        """Empty detections should return empty arrays."""
        positions, valid_indices = _get_positions_um([], pixel_size=0.5)
        assert positions.shape == (0, 2)
        assert len(valid_indices) == 0

    def test_some_missing_positions(self):
        """Detections missing position info should be skipped."""
        dets = [
            {"global_center_um": [100.0, 200.0], "features": {}},
            {"features": {}},  # no position info at all
            {"global_center_um": [300.0, 400.0], "features": {}},
        ]
        positions, valid_indices = _get_positions_um(dets, pixel_size=0.5)
        assert positions.shape == (2, 2)
        assert len(valid_indices) == 2
        assert valid_indices == [0, 2]

    def test_with_global_x_y(self):
        """Detections with global_x/global_y should be resolved."""
        dets = [
            {
                "global_x": 200,
                "global_y": 400,
                "features": {"pixel_size_um": 0.5},
            },
        ]
        positions, valid_indices = _get_positions_um(dets, pixel_size=None)
        assert positions.shape == (1, 2)
        np.testing.assert_allclose(positions[0], [100.0, 200.0])

    def test_nan_global_center_um_skipped(self):
        """NaN in global_center_um should skip the detection."""
        dets = [
            {"global_center_um": [float("nan"), 200.0], "features": {}},
            {"global_center_um": [100.0, 200.0], "features": {}},
        ]
        positions, valid_indices = _get_positions_um(dets, pixel_size=0.5)
        assert positions.shape == (1, 2)
        assert valid_indices == [1]
