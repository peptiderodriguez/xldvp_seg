"""Tests for xldvp_seg.analysis.background.correct_all_channels.

Covers:
- Raw value preservation (ch{N}_median_raw)
- Background and SNR keys created
- Corrected values <= raw values
- Double-correction guard
- Edge: single detection (global median fallback)
- Edge: all-zero channel values
"""

import numpy as np

from xldvp_seg.analysis.background import correct_all_channels


def _make_detections(n, n_channels=2, seed=42):
    """Build n detection dicts with ch0_* and ch1_* features and global_center."""
    rng = np.random.default_rng(seed)
    dets = []
    for i in range(n):
        feat = {}
        for ch in range(n_channels):
            base = rng.uniform(50, 200)
            feat[f"ch{ch}_mean"] = float(base + rng.normal(0, 10))
            feat[f"ch{ch}_median"] = float(base)
            feat[f"ch{ch}_min"] = float(base * 0.5)
            feat[f"ch{ch}_max"] = float(base * 1.5)
            feat[f"ch{ch}_p5"] = float(base * 0.6)
            feat[f"ch{ch}_p25"] = float(base * 0.8)
            feat[f"ch{ch}_p75"] = float(base * 1.2)
            feat[f"ch{ch}_p95"] = float(base * 1.4)
            feat[f"ch{ch}_std"] = float(rng.uniform(5, 20))
            feat[f"ch{ch}_cv"] = float(feat[f"ch{ch}_std"] / max(feat[f"ch{ch}_mean"], 1e-6))
        dets.append(
            {
                "global_center": [float(rng.uniform(0, 1000)), float(rng.uniform(0, 1000))],
                "features": feat,
            }
        )
    return dets


class TestCorrectAllChannels:
    def test_raw_keys_created(self):
        """After correction, ch{N}_median_raw keys must exist."""
        dets = _make_detections(50)
        channels = correct_all_channels(dets)
        assert 0 in channels
        assert 1 in channels
        for d in dets:
            for ch in channels:
                assert f"ch{ch}_median_raw" in d["features"]
                assert f"ch{ch}_mean_raw" in d["features"]

    def test_background_and_snr_keys_created(self):
        """After correction, ch{N}_background and ch{N}_snr must exist."""
        dets = _make_detections(50)
        channels = correct_all_channels(dets)
        for d in dets:
            for ch in channels:
                assert f"ch{ch}_background" in d["features"]
                assert f"ch{ch}_snr" in d["features"]
                assert isinstance(d["features"][f"ch{ch}_background"], float)
                assert isinstance(d["features"][f"ch{ch}_snr"], float)

    def test_corrected_le_raw(self):
        """Corrected ch{N}_median values must be <= raw values."""
        dets = _make_detections(50)
        correct_all_channels(dets)
        for d in dets:
            for ch in [0, 1]:
                raw = d["features"][f"ch{ch}_median_raw"]
                corrected = d["features"][f"ch{ch}_median"]
                assert corrected <= raw + 1e-6, f"Corrected {corrected} > raw {raw} for ch{ch}"

    def test_corrected_non_negative(self):
        """Corrected values must be >= 0."""
        dets = _make_detections(50)
        correct_all_channels(dets)
        for d in dets:
            for ch in [0, 1]:
                assert d["features"][f"ch{ch}_median"] >= 0.0

    def test_double_correction_guard(self):
        """If ch{N}_background already exists, function skips correction."""
        dets = _make_detections(50)
        # First correction
        channels1 = correct_all_channels(dets)
        assert len(channels1) == 2

        # Save corrected values
        saved_medians = [d["features"]["ch0_median"] for d in dets]

        # Second correction attempt - should be skipped
        channels2 = correct_all_channels(dets)
        assert channels2 == []

        # Values should be unchanged
        for i, d in enumerate(dets):
            assert d["features"]["ch0_median"] == saved_medians[i]

    def test_double_correction_uses_raw(self):
        """If ch{N}_median_raw exists but no _background, uses raw for correction."""
        dets = _make_detections(50)
        # Manually add _raw keys but NOT _background
        for d in dets:
            for ch in [0, 1]:
                d["features"][f"ch{ch}_median_raw"] = d["features"][f"ch{ch}_median"]
        channels = correct_all_channels(dets)
        # Should correct (no _background keys found)
        assert len(channels) == 2
        # Should use _raw values (not re-read corrected)
        for d in dets:
            for ch in channels:
                assert f"ch{ch}_background" in d["features"]

    def test_single_detection_global_median_fallback(self):
        """With 1 detection, should use global median fallback (not crash)."""
        dets = _make_detections(1)
        channels = correct_all_channels(dets)
        assert len(channels) == 2
        for ch in channels:
            assert f"ch{ch}_background" in dets[0]["features"]
            assert f"ch{ch}_snr" in dets[0]["features"]

    def test_all_zero_channel_values(self):
        """All-zero channel values should not crash."""
        dets = _make_detections(20)
        for d in dets:
            d["features"]["ch0_mean"] = 0.0
            d["features"]["ch0_median"] = 0.0
            d["features"]["ch0_min"] = 0.0
            d["features"]["ch0_max"] = 0.0
            d["features"]["ch0_p5"] = 0.0
            d["features"]["ch0_p25"] = 0.0
            d["features"]["ch0_p75"] = 0.0
            d["features"]["ch0_p95"] = 0.0
        channels = correct_all_channels(dets)
        assert 0 in channels
        # All corrected values should be 0
        for d in dets:
            assert d["features"]["ch0_median"] == 0.0
            assert d["features"]["ch0_background"] == 0.0

    def test_empty_detections(self):
        """Empty list should return empty channels."""
        channels = correct_all_channels([])
        assert channels == []

    def test_no_channel_keys(self):
        """Detections without ch{N}_mean keys should return empty."""
        dets = [{"global_center": [0, 0], "features": {"area": 100}}]
        channels = correct_all_channels(dets)
        assert channels == []

    def test_custom_centroids(self):
        """Providing pre-computed centroids should work."""
        dets = _make_detections(50)
        centroids = np.array([[d["global_center"][0], d["global_center"][1]] for d in dets])
        channels = correct_all_channels(dets, centroids=centroids)
        assert len(channels) == 2

    def test_snr_positive_when_signal_above_background(self):
        """SNR should be > 1 when median_raw > background."""
        dets = _make_detections(100)
        correct_all_channels(dets)
        # At least some cells should have SNR > 1 given varied intensities
        snr_values = [d["features"]["ch0_snr"] for d in dets]
        assert any(s > 1.0 for s in snr_values)

    def test_snr_zero_when_background_zero(self):
        """SNR should be 0 when background is 0 (division guard)."""
        # This is implicitly tested by the all-zero case, but let's verify
        dets = _make_detections(1)
        dets[0]["features"]["ch0_median"] = 0.0
        correct_all_channels(dets)
        assert dets[0]["features"]["ch0_snr"] == 0.0

    def test_intensity_suffixes_all_corrected(self):
        """All intensity suffixes should have _raw counterparts."""
        dets = _make_detections(50)
        channels = correct_all_channels(dets)
        suffixes = ["mean", "median", "min", "max", "p5", "p25", "p75", "p95"]
        for d in dets:
            for ch in channels:
                for suffix in suffixes:
                    raw_key = f"ch{ch}_{suffix}_raw"
                    assert raw_key in d["features"], f"Missing {raw_key}"
