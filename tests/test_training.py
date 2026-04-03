"""Tests for training/feature_loader utilities."""

from xldvp_seg.training.feature_loader import (
    filter_feature_names,
    is_channel_stats_feature,
    is_morph_feature,
    is_morph_sam2_feature,
)


class TestIsMorphFeature:
    def test_morph_features(self):
        assert is_morph_feature("area") is True
        assert is_morph_feature("circularity") is True
        assert is_morph_feature("ch0_mean") is True  # morph includes channel stats

    def test_non_morph_features(self):
        assert is_morph_feature("sam2_0") is False
        assert is_morph_feature("resnet_5") is False
        assert is_morph_feature("dinov2_100") is False


class TestIsMorphSam2Feature:
    def test_includes_morph_and_sam2(self):
        assert is_morph_sam2_feature("area") is True
        assert is_morph_sam2_feature("sam2_0") is True
        assert is_morph_sam2_feature("sam2_255") is True

    def test_excludes_deep(self):
        assert is_morph_sam2_feature("resnet_5") is False
        assert is_morph_sam2_feature("dinov2_100") is False


class TestIsChannelStatsFeature:
    def test_channel_features(self):
        assert is_channel_stats_feature("ch0_mean") is True
        assert is_channel_stats_feature("ch3_snr") is True
        assert is_channel_stats_feature("ch0_ch1_ratio") is True

    def test_non_channel_features(self):
        assert is_channel_stats_feature("area") is False
        assert is_channel_stats_feature("sam2_0") is False
        assert is_channel_stats_feature("circularity") is False


class TestFilterFeatureNames:
    def test_all_returns_everything(self):
        names = ["area", "sam2_0", "resnet_5", "ch0_mean"]
        result = filter_feature_names(names, "all")
        assert result == names

    def test_morph_excludes_deep(self):
        names = ["area", "ch0_mean", "sam2_0", "resnet_5", "dinov2_100"]
        result = filter_feature_names(names, "morph")
        assert "area" in result
        assert "ch0_mean" in result
        assert "sam2_0" not in result
        assert "resnet_5" not in result

    def test_morph_sam2(self):
        names = ["area", "sam2_0", "resnet_5"]
        result = filter_feature_names(names, "morph_sam2")
        assert "area" in result
        assert "sam2_0" in result
        assert "resnet_5" not in result

    def test_channel_stats(self):
        names = ["area", "ch0_mean", "ch1_snr", "sam2_0"]
        result = filter_feature_names(names, "channel_stats")
        assert "ch0_mean" in result
        assert "ch1_snr" in result
        assert "area" not in result
        assert "sam2_0" not in result

    def test_empty_input(self):
        assert filter_feature_names([], "morph") == []
