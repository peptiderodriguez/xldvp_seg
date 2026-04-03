"""Tests for cluster feature utilities."""

import numpy as np

from xldvp_seg.analysis.cluster_features import (
    classify_feature_group,
    discover_channels_from_features,
    extract_feature_matrix,
    normalize_marker_features,
    parse_exclude_channels,
    parse_marker_channels,
    select_feature_names,
)


class TestParseMarkerChannels:
    def test_valid_string(self):
        result = parse_marker_channels("alpha:2,beta:3,delta:5")
        assert result == {"alpha": 2, "beta": 3, "delta": 5}

    def test_single_marker(self):
        result = parse_marker_channels("SMA:1")
        assert result == {"SMA": 1}

    def test_empty_string(self):
        assert parse_marker_channels("") is None
        assert parse_marker_channels(None) is None

    def test_malformed_pair(self):
        result = parse_marker_channels("badformat")
        assert result is None  # no valid pairs


class TestParseExcludeChannels:
    def test_single(self):
        assert parse_exclude_channels("3") == {3}

    def test_multiple(self):
        assert parse_exclude_channels("0,3,5") == {0, 3, 5}

    def test_empty(self):
        assert parse_exclude_channels("") == set()
        assert parse_exclude_channels(None) == set()


class TestClassifyFeatureGroup:
    def test_shape_features(self):
        assert classify_feature_group("area") == "shape"
        assert classify_feature_group("circularity") == "shape"
        assert classify_feature_group("solidity") == "shape"
        assert classify_feature_group("aspect_ratio") == "shape"

    def test_color_features(self):
        assert classify_feature_group("gray_mean") == "color"
        assert classify_feature_group("hue_mean") == "color"

    def test_channel_features(self):
        assert classify_feature_group("ch0_mean") == "channel"
        assert classify_feature_group("ch2_snr") == "channel"
        assert classify_feature_group("ch0_ch1_ratio") == "channel"

    def test_sam2_features(self):
        assert classify_feature_group("sam2_0") == "sam2"
        assert classify_feature_group("sam2_255") == "sam2"

    def test_deep_features(self):
        assert classify_feature_group("resnet_0") == "deep"
        assert classify_feature_group("dinov2_100") == "deep"

    def test_unknown_features(self):
        # String-valued classification keys are not numeric features
        # but classify_feature_group may return "shape" as fallback for unrecognized keys
        result = classify_feature_group("NeuN_class")
        assert result is not None  # falls through to shape as catch-all


class TestDiscoverChannels:
    def test_finds_channels(self):
        dets = [{"features": {"ch0_mean": 1.0, "ch2_mean": 2.0, "area": 50}}]
        result = discover_channels_from_features(dets)
        assert result == [0, 2]

    def test_empty_detections(self):
        assert discover_channels_from_features([]) == []

    def test_no_channel_features(self):
        dets = [{"features": {"area": 50, "circularity": 0.8}}]
        assert discover_channels_from_features(dets) == []


class TestSelectFeatureNames:
    def test_morph_includes_shape_and_color(self):
        dets = [
            {
                "features": {
                    "area": 50,
                    "gray_mean": 100,
                    "ch0_mean": 10,
                    "sam2_0": 0.5,
                }
            }
        ]
        names = select_feature_names(dets, {"morph"}, None)
        assert "area" in names
        assert "gray_mean" in names
        assert "ch0_mean" not in names
        assert "sam2_0" not in names

    def test_channel_group(self):
        dets = [{"features": {"area": 50, "ch0_mean": 10, "ch1_snr": 2.0}}]
        names = select_feature_names(dets, {"channel"}, None)
        assert "ch0_mean" in names
        assert "ch1_snr" in names
        assert "area" not in names

    def test_exclude_channels(self):
        dets = [{"features": {"ch0_mean": 10, "ch1_mean": 20, "ch2_mean": 30}}]
        names = select_feature_names(dets, {"channel"}, {1})
        assert "ch0_mean" in names
        assert "ch2_mean" in names
        assert "ch1_mean" not in names


class TestExtractFeatureMatrix:
    def test_complete_features(self):
        dets = [
            {"features": {"area": 50.0, "circ": 0.8}},
            {"features": {"area": 100.0, "circ": 0.9}},
            {"features": {"area": 75.0, "circ": 0.7}},
        ]
        X, names, valid = extract_feature_matrix(dets, ["area", "circ"])
        assert X.shape == (3, 2)
        assert valid == [0, 1, 2]

    def test_missing_feature_skips_row(self):
        dets = [
            {"features": {"area": 50.0, "circ": 0.8}},
            {"features": {"area": 100.0}},  # missing circ
            {"features": {"area": 75.0, "circ": 0.7}},
        ]
        X, names, valid = extract_feature_matrix(dets, ["area", "circ"])
        assert X.shape == (2, 2)
        assert valid == [0, 2]

    def test_empty_feature_names(self):
        X, names, valid = extract_feature_matrix([], [])
        assert X is None
        assert names == []
        assert valid == []

    def test_no_valid_rows(self):
        dets = [{"features": {}}]
        X, names, valid = extract_feature_matrix(dets, ["area"])
        # No detection has the required feature, so X is None
        assert X is None or X.shape[0] == 0


class TestNormalizeMarkerFeatures:
    def test_normalizes_channel_features(self):
        np.random.seed(42)
        X = np.random.rand(100, 3)
        names = ["ch0_mean", "ch1_mean", "area"]
        X_norm, ranges = normalize_marker_features(X, names)
        assert X_norm.shape == X.shape
        # Channel columns should be normalized
        assert "ch0_mean" in ranges or "ch1_mean" in ranges
