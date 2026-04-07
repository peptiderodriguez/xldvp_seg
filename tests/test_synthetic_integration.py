"""Integration tests using synthetic data (no CZI, no GPU, no mocks)."""

import numpy as np

from xldvp_seg.core.slide_analysis import SlideAnalysis

# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------


def _make_synthetic_detections(n=30):
    rng = np.random.default_rng(42)
    dets = []
    for i in range(n):
        dets.append(
            {
                "uid": f"test_cell_{i}",
                "global_center": [float(i * 10), float(i * 5)],
                "features": {
                    "area": float(rng.integers(500, 5000)),
                    "area_um2": float(rng.integers(50, 500)),
                    "ch0_mean": float(rng.uniform(10, 200)),
                    "ch0_median": float(rng.uniform(10, 200)),
                    "ch0_min": float(rng.uniform(0, 50)),
                    "ch0_max": float(rng.uniform(100, 255)),
                    "ch1_mean": float(rng.uniform(5, 150)),
                    "ch1_median": float(rng.uniform(5, 150)),
                    "circularity": float(rng.uniform(0.3, 1.0)),
                    "eccentricity": float(rng.uniform(0.0, 0.9)),
                    "solidity": float(rng.uniform(0.7, 1.0)),
                },
            }
        )
    return dets


# ---------------------------------------------------------------------------
# Test: SlideAnalysis roundtrip
# ---------------------------------------------------------------------------


class TestSlideAnalysisRoundtrip:
    def test_from_detections_and_filter(self):
        dets = _make_synthetic_detections(30)
        slide = SlideAnalysis.from_detections(dets)

        assert slide.n_detections == 30
        assert len(slide.detections) == 30

        # Features DataFrame
        df = slide.features_df
        assert df.shape[0] == 30
        assert "area" in df.columns
        assert "ch0_mean" in df.columns

    def test_filter_by_area(self):
        dets = _make_synthetic_detections(30)
        slide = SlideAnalysis.from_detections(dets)

        # Manual filter (SlideAnalysis.filter doesn't support arbitrary features,
        # so we filter detections and create a new slide)
        big = [d for d in slide.detections if d["features"]["area"] > 1000]
        filtered = SlideAnalysis.from_detections(big)
        assert filtered.n_detections <= 30
        assert filtered.n_detections > 0
        for d in filtered.detections:
            assert d["features"]["area"] > 1000

    def test_to_anndata(self):
        dets = _make_synthetic_detections(30)
        slide = SlideAnalysis.from_detections(dets)
        adata = slide.to_anndata()

        assert adata.shape[0] == 30
        assert adata.X.shape[0] == 30
        # var should have feature_group
        assert "feature_group" in adata.var.columns
        # obs should have slide_name
        assert "slide_name" in adata.obs.columns
        # X should be float32
        assert adata.X.dtype == np.float32


# ---------------------------------------------------------------------------
# Test: Marker classification (otsu method on real data)
# ---------------------------------------------------------------------------


class TestMarkersRealClassify:
    def test_otsu_classifies_bimodal_signal(self):
        from xldvp_seg.api import tl

        rng = np.random.default_rng(99)
        dets = []
        for i in range(30):
            # First 15: high ch0_mean (100-200), last 15: low ch0_mean (5-20)
            if i < 15:
                ch0_mean = float(rng.uniform(100, 200))
            else:
                ch0_mean = float(rng.uniform(5, 20))
            dets.append(
                {
                    "uid": f"marker_cell_{i}",
                    "global_center": [float(i * 10), float(i * 5)],
                    "features": {
                        "area": float(rng.integers(500, 5000)),
                        "ch0_mean": ch0_mean,
                        "ch0_median": ch0_mean * 0.9,
                    },
                }
            )

        slide = SlideAnalysis.from_detections(dets)
        result = tl.markers(
            slide,
            marker_channels=[0],
            marker_names=["test"],
            method="otsu",
            intensity_feature="mean",
        )

        # Verify marker fields exist
        for d in result.detections:
            feat = d["features"]
            assert "test_class" in feat
            assert "test_value" in feat
            assert feat["test_class"] in ("positive", "negative")

        # With a bimodal signal, we should get some positive and some negative
        classes = [d["features"]["test_class"] for d in result.detections]
        assert "positive" in classes, "Expected some positive classifications"
        assert "negative" in classes, "Expected some negative classifications"

        # marker_profile should be set
        for d in result.detections:
            assert "marker_profile" in d["features"]


# ---------------------------------------------------------------------------
# Test: Score with real RF classifier
# ---------------------------------------------------------------------------


class TestScoreRealRF:
    def test_score_with_trained_rf(self, tmp_path):
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        from xldvp_seg.api import tl

        dets = _make_synthetic_detections(30)
        slide = SlideAnalysis.from_detections(dets)

        # Train a small RF on synthetic features
        feature_names = ["area", "circularity", "solidity"]
        X = np.array([[d["features"][f] for f in feature_names] for d in dets], dtype=np.float32)
        # Label: area > 2000 -> positive
        y = (X[:, 0] > 2000).astype(int)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)

        # Save in dict format (what load_rf_classifier expects)
        clf_path = tmp_path / "test_rf.pkl"
        joblib.dump(
            {
                "model": rf,
                "feature_names": feature_names,
                "type": "rf",
            },
            clf_path,
        )

        result = tl.score(slide, classifier=clf_path)

        # All detections should now have rf_prediction
        scored = [d for d in result.detections if "rf_prediction" in d]
        assert len(scored) == 30
        for d in scored:
            assert 0.0 <= d["rf_prediction"] <= 1.0


# ---------------------------------------------------------------------------
# Test: Background correction
# ---------------------------------------------------------------------------


class TestBackgroundCorrectionReal:
    def test_correct_all_channels(self):
        from xldvp_seg.pipeline.background import correct_all_channels

        rng = np.random.default_rng(42)
        dets = []
        for i in range(20):
            dets.append(
                {
                    "uid": f"bg_cell_{i}",
                    "global_center": [float(i * 20), float(i * 10)],
                    "features": {
                        "ch0_mean": float(rng.uniform(50, 200)),
                        "ch0_median": float(rng.uniform(50, 200)),
                        "ch0_min": float(rng.uniform(0, 50)),
                        "ch0_max": float(rng.uniform(100, 255)),
                    },
                }
            )

        channels = correct_all_channels(dets, n_neighbors=5)

        assert 0 in channels, "Channel 0 should have been corrected"

        for d in dets:
            feat = d["features"]
            # Raw values preserved
            assert "ch0_median_raw" in feat
            assert "ch0_mean_raw" in feat
            # Background estimated
            assert "ch0_background" in feat
            assert feat["ch0_background"] >= 0
            # SNR computed
            assert "ch0_snr" in feat
            # Corrected median should be <= raw median
            assert feat["ch0_median"] <= feat["ch0_median_raw"] + 1e-9
