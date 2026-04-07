"""Tests for xldvp_seg.analysis.marker_classification.

Covers the core science functions: classify_otsu, classify_otsu_half,
classify_gmm, extract_marker_values, classify_single_marker, and
plot_distribution.
"""

import numpy as np
import pytest

from xldvp_seg.analysis.marker_classification import (
    classify_gmm,
    classify_otsu,
    classify_otsu_half,
    classify_single_marker,
    extract_marker_values,
    plot_distribution,
)


def _make_detections(n, channel=1, feature_key="ch1_mean", values=None):
    """Build detection dicts with features.{feature_key} populated."""
    np.random.seed(42)
    if values is None:
        values = np.random.rand(n) * 100.0
    dets = []
    for i, v in enumerate(values):
        dets.append(
            {
                "uid": f"test_{i}",
                "global_center": [float(i * 10), float(i * 10)],
                "features": {feature_key: float(v)},
            }
        )
    return dets


def _bimodal_values(n_low=500, n_high=500, mu_low=1.0, sigma_low=0.3, mu_high=5.0, sigma_high=0.5):
    np.random.seed(42)
    low = np.random.normal(mu_low, sigma_low, n_low)
    high = np.random.normal(mu_high, sigma_high, n_high)
    return np.concatenate([low, high])


class TestClassifyOtsu:
    def test_bimodal_separation(self):
        values = _bimodal_values()
        threshold, mask = classify_otsu(values)
        assert 1.0 < threshold < 5.0
        n_pos = mask.sum()
        assert 400 < n_pos < 600

    def test_too_few_values(self):
        values = np.array([0.0] * 95 + [1.0] * 5)
        threshold, mask = classify_otsu(values)
        assert threshold == 0.0
        assert mask.sum() == 0

    def test_zero_variance(self):
        values = np.full(100, 42.0)
        threshold, mask = classify_otsu(values)
        assert threshold == 0.0
        assert mask.sum() == 0

    def test_all_zeros(self):
        values = np.zeros(200)
        threshold, mask = classify_otsu(values)
        assert threshold == 0.0
        assert mask.sum() == 0
        assert len(mask) == 200

    def test_mask_length_matches_input(self):
        values = _bimodal_values()
        _, mask = classify_otsu(values)
        assert len(mask) == len(values)

    def test_single_element(self):
        """Single element should not crash."""
        values = np.array([5.0])
        threshold, mask = classify_otsu(values)
        assert threshold == 0.0
        assert mask.sum() == 0

    def test_nan_values(self):
        """NaN in values should be handled gracefully (not crash)."""
        values = np.array([1.0, 2.0, float("nan"), 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0])
        threshold, mask = classify_otsu(values)
        assert len(mask) == len(values)

    def test_positive_requires_above_zero(self):
        values = np.array([0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0])
        _, mask = classify_otsu(values)
        assert not mask[0]
        assert not mask[1]
        assert not mask[2]


class TestClassifyOtsuHalf:
    def test_threshold_is_half(self):
        values = _bimodal_values()
        t_full, _ = classify_otsu(values)
        t_half, _ = classify_otsu_half(values)
        assert abs(t_half - t_full / 2.0) < 0.01

    def test_more_permissive(self):
        values = _bimodal_values()
        _, mask_full = classify_otsu(values)
        _, mask_half = classify_otsu_half(values)
        assert mask_half.sum() >= mask_full.sum()

    def test_too_few_values(self):
        values = np.array([0.0] * 95 + [1.0] * 5)
        threshold, mask = classify_otsu_half(values)
        assert threshold == 0.0
        assert mask.sum() == 0


class TestClassifyGmm:
    def test_bimodal_separation(self):
        values = _bimodal_values(mu_low=1.0, mu_high=10.0)
        threshold, mask = classify_gmm(values)
        n_pos = mask.sum()
        assert 350 < n_pos < 650
        # Threshold must fall between the two modes
        assert (
            1.0 < threshold < 10.0
        ), f"GMM threshold {threshold:.2f} not between modes 1.0 and 10.0"

    def test_too_few_values(self):
        values = np.array([1.0, 2.0, 3.0])
        threshold, mask = classify_gmm(values)
        assert threshold == 0.0
        assert mask.sum() == 0

    def test_unimodal_still_classifies(self):
        """Unimodal data: GMM still returns a result (may warn about poor separation)."""
        np.random.seed(42)
        values = np.random.normal(5.0, 0.1, 200)
        threshold, mask = classify_gmm(values)
        # Should not crash; threshold and mask should be valid
        assert isinstance(threshold, float)
        assert len(mask) == 200

    def test_zero_variance(self):
        values = np.full(50, 42.0)
        threshold, mask = classify_gmm(values)
        assert threshold == 0.0
        assert mask.sum() == 0


class TestExtractMarkerValues:
    def test_normal_extraction(self):
        dets = [
            {"features": {"ch2_mean": 10.0}},
            {"features": {"ch2_mean": 20.0}},
            {"features": {"ch2_mean": 30.0}},
        ]
        vals = extract_marker_values(dets, channel=2, feature="mean")
        np.testing.assert_array_equal(vals, [10.0, 20.0, 30.0])

    def test_snr_feature(self):
        dets = [
            {"features": {"ch1_snr": 2.5}},
            {"features": {"ch1_snr": 0.5}},
        ]
        vals = extract_marker_values(dets, channel=1, feature="snr")
        np.testing.assert_array_equal(vals, [2.5, 0.5])

    def test_use_raw_fallback(self):
        dets_raw = [{"features": {"ch0_mean_raw": 99.0, "ch0_mean": 50.0}}]
        vals = extract_marker_values(dets_raw, channel=0, feature="mean", use_raw=True)
        np.testing.assert_array_equal(vals, [99.0])

        dets_no_raw = [{"features": {"ch0_mean": 50.0}}]
        vals_fb = extract_marker_values(dets_no_raw, channel=0, feature="mean", use_raw=True)
        np.testing.assert_array_equal(vals_fb, [50.0])

    def test_missing_key_returns_zeros(self):
        dets = [{"features": {"ch0_mean": 10.0}}, {"features": {"ch0_mean": 20.0}}]
        vals = extract_marker_values(dets, channel=5, feature="mean")
        np.testing.assert_array_equal(vals, [0.0, 0.0])

    def test_missing_features_dict(self):
        dets = [{"uid": "no_features"}, {"features": {"ch1_mean": 5.0}}]
        vals = extract_marker_values(dets, channel=1, feature="mean")
        np.testing.assert_array_equal(vals, [0.0, 5.0])


class TestClassifySingleMarker:
    def test_snr_method_realistic(self, tmp_path):
        np.random.seed(42)
        n = 1000
        snr_low = np.random.normal(0.5, 0.2, n // 2)
        snr_high = np.random.normal(3.0, 0.5, n // 2)
        snr_all = np.concatenate([snr_low, snr_high])

        dets = []
        for i, s in enumerate(snr_all):
            dets.append(
                {
                    "uid": f"cell_{i}",
                    "global_center": [float(i * 10), float(i * 10)],
                    "features": {"ch1_snr": float(max(s, 0))},
                }
            )

        result = classify_single_marker(
            dets,
            channel=1,
            marker_name="SMA",
            method="snr",
            output_dir=tmp_path,
            snr_threshold=1.5,
        )

        assert 350 < result["n_positive"] < 650
        assert result["n_positive"] + result["n_negative"] == n

        for det in dets:
            assert "SMA_class" in det["features"]
            assert "SMA_value" in det["features"]
            assert det["features"]["SMA_class"] in ("positive", "negative")

    def test_otsu_method(self, tmp_path):
        values = _bimodal_values(mu_low=2.0, mu_high=20.0)
        dets = _make_detections(len(values), feature_key="ch1_mean", values=values)
        result = classify_single_marker(
            dets,
            channel=1,
            marker_name="NeuN",
            method="otsu",
            output_dir=tmp_path,
        )
        assert 1.0 < result["threshold"] < 20.0
        assert 350 < result["n_positive"] < 650

    def test_mutates_in_place(self, tmp_path):
        values = _bimodal_values()
        dets = _make_detections(len(values), feature_key="ch1_mean", values=values)
        classify_single_marker(
            dets,
            channel=1,
            marker_name="TestMK",
            method="otsu",
            output_dir=tmp_path,
        )
        for det in dets:
            assert "TestMK_class" in det["features"]
            assert "TestMK_value" in det["features"]
            assert "TestMK_class" not in det

    def test_returns_summary_dict(self, tmp_path):
        values = _bimodal_values()
        dets = _make_detections(len(values), feature_key="ch1_mean", values=values)
        result = classify_single_marker(
            dets,
            channel=1,
            marker_name="Marker1",
            method="otsu",
            output_dir=tmp_path,
        )
        expected_keys = {
            "marker",
            "method",
            "threshold",
            "n_positive",
            "n_negative",
            "pct_positive",
        }
        assert expected_keys.issubset(result.keys())
        assert result["marker"] == "Marker1"
        assert isinstance(result["threshold"], float)
        assert isinstance(result["n_positive"], int)

    def test_empty_detections(self, tmp_path):
        result = classify_single_marker(
            [],
            channel=1,
            marker_name="Empty",
            method="snr",
            output_dir=tmp_path,
        )
        assert result["n_positive"] == 0
        assert result["n_negative"] == 0

    def test_cv_filter(self, tmp_path):
        np.random.seed(42)
        dets = []
        for i in range(200):
            cv_val = 0.5 if i < 150 else 2.0
            dets.append(
                {
                    "uid": f"cell_{i}",
                    "global_center": [float(i * 10), float(i * 10)],
                    "features": {"ch1_snr": 5.0, "ch1_cv": float(cv_val)},
                }
            )
        result = classify_single_marker(
            dets,
            channel=1,
            marker_name="CVTest",
            method="snr",
            output_dir=tmp_path,
            snr_threshold=1.5,
            cv_max=1.0,
        )
        assert result["n_positive"] == 150
        assert result["cv_filtered"] == 50

    def test_gmm_method(self, tmp_path):
        """GMM method works through classify_single_marker with correct threshold."""
        values = _bimodal_values(mu_low=1.0, mu_high=10.0)
        dets = _make_detections(len(values), feature_key="ch1_mean", values=values)
        result = classify_single_marker(
            dets,
            channel=1,
            marker_name="GMM_Test",
            method="gmm",
            output_dir=tmp_path,
        )
        assert (
            1.0 < result["threshold"] < 10.0
        ), f"GMM threshold {result['threshold']:.2f} not between modes"
        assert 350 < result["n_positive"] < 650

    def test_otsu_half_method(self, tmp_path):
        """otsu_half through classify_single_marker is more permissive than otsu."""
        values = _bimodal_values(mu_low=2.0, mu_high=20.0)
        dets_otsu = _make_detections(len(values), feature_key="ch1_mean", values=values)
        dets_half = _make_detections(len(values), feature_key="ch1_mean", values=values)
        result_otsu = classify_single_marker(
            dets_otsu,
            channel=1,
            marker_name="OtsuFull",
            method="otsu",
            output_dir=tmp_path,
        )
        result_half = classify_single_marker(
            dets_half,
            channel=1,
            marker_name="OtsuHalf",
            method="otsu_half",
            output_dir=tmp_path,
        )
        assert result_half["n_positive"] >= result_otsu["n_positive"]
        assert result_half["threshold"] < result_otsu["threshold"]

    def test_unknown_method_raises(self, tmp_path):
        dets = _make_detections(10, feature_key="ch1_snr", values=np.ones(10) * 5.0)
        with pytest.raises(ValueError, match="Unknown method"):
            classify_single_marker(
                dets,
                channel=1,
                marker_name="Bad",
                method="nonexistent",
                output_dir=tmp_path,
            )

    def test_snr_without_snr_feature_raises(self, tmp_path):
        dets = _make_detections(10, feature_key="ch1_mean", values=np.ones(10) * 50.0)
        for d in dets:
            d["features"].pop("ch1_snr", None)
        from xldvp_seg.exceptions import DataLoadError

        with pytest.raises(DataLoadError, match="SNR method requires"):
            classify_single_marker(
                dets,
                channel=1,
                marker_name="NoSNR",
                method="snr",
                output_dir=tmp_path,
            )

    def test_distribution_plot_created(self, tmp_path):
        values = _bimodal_values()
        dets = _make_detections(len(values), feature_key="ch1_mean", values=values)
        classify_single_marker(
            dets,
            channel=1,
            marker_name="PlotCheck",
            method="otsu",
            output_dir=tmp_path,
        )
        assert (tmp_path / "PlotCheck_distribution.png").exists()


class TestPlotDistribution:
    def test_creates_png(self, tmp_path):
        np.random.seed(42)
        values = np.random.normal(50, 10, 500)
        out = tmp_path / "test_dist.png"
        plot_distribution(values, 50.0, "TestMarker", "otsu", 250, 250, out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_empty_values_noop(self, tmp_path):
        out = tmp_path / "empty_dist.png"
        plot_distribution(np.array([]), 0.0, "Empty", "otsu", 0, 0, out)
        assert not out.exists()

    def test_log_scale_wide_range(self, tmp_path):
        np.random.seed(42)
        values = np.concatenate(
            [
                np.random.normal(1, 0.5, 500),
                np.random.normal(1000, 100, 500),
            ]
        )
        values = np.maximum(values, 0)
        out = tmp_path / "log_dist.png"
        plot_distribution(values, 100.0, "WideRange", "gmm", 500, 500, out)
        assert out.exists()
