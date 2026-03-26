"""Roundtrip tests verifying pipeline handoffs with realistic detection formats.

These tests use detection dicts matching the REAL pipeline output format
(not simplified test fixtures). They verify that each pipeline stage's
output is correctly consumed by the next stage.

No GPU, no CZI, no SLURM -- runs in CI in seconds.

Key format facts verified here:
  - classify_markers.py stores marker classes INSIDE det["features"]
  - rf_prediction is at the TOP LEVEL of the detection dict
  - quality_filter_detections.py sets det["rf_prediction"] = 1.0 for passing
  - contour_dilated_px is at the TOP LEVEL (list of [x,y] pairs)
  - global_center / global_center_um are at the TOP LEVEL
  - features dict: area, solidity, eccentricity, sam2_*, ch*_mean/median/background/snr/median_raw
"""

import numpy as np
import pytest

from segmentation.core import SlideAnalysis
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load


def _make_pipeline_detections(n=20):
    """Create detections matching real pipeline output format.

    Includes all the fields that matter for handoff correctness:
    - Top-level: uid, rf_prediction, global_center, global_center_um,
      contour_dilated_px, cell_type
    - features dict: morphology, SAM2 embeddings, per-channel stats
      (ch0_mean, ch0_median, ch0_background, ch0_snr, ch0_median_raw),
      marker classes (NeuN_class, tdTomato_class), marker_profile

    Key: marker classes go in features dict, NOT top-level.
    This matches what classify_markers.py actually produces.
    """
    rng = np.random.RandomState(42)
    dets = []
    for i in range(n):
        # Alternate marker assignments
        neun_class = "positive" if i % 3 != 0 else "negative"
        tdt_class = "positive" if i % 4 == 0 else "negative"
        neun_sign = "+" if neun_class == "positive" else "-"
        tdt_sign = "+" if tdt_class == "positive" else "-"
        marker_profile = f"NeuN{neun_sign}/tdTomato{tdt_sign}"

        x_px = 1000 + i * 50
        y_px = 2000 + i * 30
        pixel_size = 0.325

        # Contour: small square around center (global pixel coords)
        contour = [
            [x_px - 10, y_px - 10],
            [x_px + 10, y_px - 10],
            [x_px + 10, y_px + 10],
            [x_px - 10, y_px + 10],
        ]

        det = {
            # --- Top-level fields ---
            "uid": f"testslide_cell_{x_px}_{y_px}",
            "cell_type": "cell",
            "rf_prediction": round(0.2 + i * 0.04, 2),  # 0.2 to 0.96
            "global_center": [x_px, y_px],
            "global_center_um": [round(x_px * pixel_size, 3), round(y_px * pixel_size, 3)],
            "contour_dilated_px": contour,
            # --- Features dict (everything else) ---
            "features": {
                # Morphological features
                "area": int(area_px := 300 + rng.randint(0, 500)),
                "area_um2": round(area_px * pixel_size**2, 3),
                "solidity": round(0.7 + rng.random() * 0.3, 3),
                "eccentricity": round(rng.random() * 0.8, 3),
                "perimeter": round(60 + rng.random() * 40, 2),
                "major_axis_length": round(20 + rng.random() * 30, 2),
                "minor_axis_length": round(10 + rng.random() * 20, 2),
                # SAM2 embeddings (abbreviated: 4 dims for tests)
                "sam2_0": round(float(rng.randn()), 4),
                "sam2_1": round(float(rng.randn()), 4),
                "sam2_2": round(float(rng.randn()), 4),
                "sam2_3": round(float(rng.randn()), 4),
                # Per-channel stats (channel 0)
                "ch0_mean": round(50 + rng.random() * 200, 2),
                "ch0_median": round(40 + rng.random() * 180, 2),
                "ch0_background": round(10 + rng.random() * 30, 2),
                "ch0_snr": round(1.0 + rng.random() * 4.0, 3),
                "ch0_median_raw": round(40 + rng.random() * 180, 2),
                # Per-channel stats (channel 1)
                "ch1_mean": round(30 + rng.random() * 150, 2),
                "ch1_median": round(25 + rng.random() * 130, 2),
                "ch1_background": round(8 + rng.random() * 25, 2),
                "ch1_snr": round(0.8 + rng.random() * 3.5, 3),
                "ch1_median_raw": round(25 + rng.random() * 130, 2),
                # Marker classes -- INSIDE features (classify_markers.py format)
                "NeuN_class": neun_class,
                "NeuN_value": round(rng.random() * 5.0, 3),
                "NeuN_threshold": 1.5,
                "tdTomato_class": tdt_class,
                "tdTomato_value": round(rng.random() * 3.0, 3),
                "tdTomato_threshold": 1.2,
                "marker_profile": marker_profile,
            },
        }
        dets.append(det)
    return dets


# ---------------------------------------------------------------------------
# Detection -> SlideAnalysis handoff
# ---------------------------------------------------------------------------


class TestDetectionToSlideAnalysis:
    """Verify SlideAnalysis correctly reads real pipeline detection format."""

    def test_features_df_includes_marker_class_from_features(self):
        """Marker classes stored in det['features'] appear in features_df."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        df = slide.features_df
        assert "NeuN_class" in df.columns
        assert "tdTomato_class" in df.columns
        # Verify actual values propagated
        assert set(df["NeuN_class"].unique()) == {"positive", "negative"}

    def test_features_df_includes_rf_prediction_from_toplevel(self):
        """rf_prediction stored at top level appears in features_df."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        df = slide.features_df
        assert "rf_prediction" in df.columns
        # Should match the top-level values we set
        assert df["rf_prediction"].iloc[0] == pytest.approx(0.2, abs=0.01)

    def test_filter_marker_positive_from_features(self):
        """filter(marker=...) finds classes stored in features dict."""
        dets = _make_pipeline_detections(20)
        slide = SlideAnalysis.from_detections(dets)
        pos = slide.filter(marker="NeuN", positive=True)
        # i % 3 != 0 -> positive. Out of 0..19, multiples of 3: {0,3,6,9,12,15,18} = 7 neg
        expected_pos = 20 - 7  # 13
        assert pos.n_detections == expected_pos
        # Every remaining detection should be positive
        for det in pos.detections:
            assert det["features"]["NeuN_class"] == "positive"

    def test_filter_marker_negative_from_features(self):
        """filter(marker=..., positive=False) works with features dict."""
        dets = _make_pipeline_detections(20)
        slide = SlideAnalysis.from_detections(dets)
        neg = slide.filter(marker="NeuN", positive=False)
        expected_neg = 7  # multiples of 3 in 0..19
        assert neg.n_detections == expected_neg
        for det in neg.detections:
            assert det["features"]["NeuN_class"] == "negative"

    def test_filter_score_threshold(self):
        """filter(score_threshold=...) uses top-level rf_prediction."""
        dets = _make_pipeline_detections(20)
        slide = SlideAnalysis.from_detections(dets)
        # rf_prediction goes from 0.2 to 0.96 in steps of 0.04
        # >= 0.5: i where 0.2 + i*0.04 >= 0.5 -> i >= 7.5 -> i >= 8
        filtered = slide.filter(score_threshold=0.5)
        expected = 20 - 8  # indices 8..19 = 12
        assert filtered.n_detections == expected
        for det in filtered.detections:
            assert det["rf_prediction"] >= 0.5

    def test_filter_combined_score_and_marker(self):
        """Combined score + marker filter works with real format."""
        dets = _make_pipeline_detections(20)
        slide = SlideAnalysis.from_detections(dets)
        filtered = slide.filter(score_threshold=0.5, marker="NeuN", positive=True)
        # Score >= 0.5: indices 8..19
        # NeuN positive: i % 3 != 0 -> in 8..19, multiples of 3: {9, 12, 15, 18} = 4 neg
        # So 12 - 4 = 8 pass both
        assert filtered.n_detections == 8
        for det in filtered.detections:
            assert det["rf_prediction"] >= 0.5
            assert det["features"]["NeuN_class"] == "positive"

    def test_chained_filter_preserves_format(self):
        """Chaining .filter().filter() preserves detection dict format."""
        dets = _make_pipeline_detections(20)
        slide = SlideAnalysis.from_detections(dets)
        result = slide.filter(score_threshold=0.5).filter(marker="NeuN", positive=True)
        # Same count as combined filter
        assert result.n_detections == 8
        # Verify full detection structure is intact after chaining
        det = result.detections[0]
        assert "uid" in det
        assert "rf_prediction" in det
        assert "global_center" in det
        assert "contour_dilated_px" in det
        assert "features" in det
        assert "area" in det["features"]
        assert "NeuN_class" in det["features"]


# ---------------------------------------------------------------------------
# SlideAnalysis -> AnnData handoff
# ---------------------------------------------------------------------------


class TestSlideAnalysisToAnndata:
    """Verify to_anndata() works with real pipeline format."""

    def test_anndata_has_marker_in_obs(self):
        """AnnData obs contains marker_class columns from features dict."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        adata = slide.to_anndata()
        assert "NeuN_class" in adata.obs.columns
        assert "tdTomato_class" in adata.obs.columns
        assert set(adata.obs["NeuN_class"].unique()) == {"positive", "negative"}

    def test_anndata_has_rf_prediction_in_obs(self):
        """AnnData obs contains rf_prediction."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        adata = slide.to_anndata()
        assert "rf_prediction" in adata.obs.columns
        assert adata.obs["rf_prediction"].notna().all()

    def test_anndata_has_spatial_coords(self):
        """AnnData obsm['spatial'] populated from global_center_um."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        adata = slide.to_anndata()
        assert "spatial" in adata.obsm
        assert adata.obsm["spatial"].shape == (10, 2)
        # Verify coordinates are in microns (not pixels)
        # First detection: global_center_um = [1000*0.325, 2000*0.325] = [325.0, 650.0]
        assert adata.obsm["spatial"][0, 0] == pytest.approx(325.0, abs=0.1)
        assert adata.obsm["spatial"][0, 1] == pytest.approx(650.0, abs=0.1)

    def test_anndata_morph_features_in_X(self):
        """Morphological features in X, not embeddings."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        adata = slide.to_anndata()
        var_names = list(adata.var_names)
        assert "area" in var_names
        assert "solidity" in var_names
        assert "eccentricity" in var_names
        # Channel stats should also be in X
        assert "ch0_mean" in var_names
        assert "ch0_snr" in var_names

    def test_anndata_sam2_in_obsm(self):
        """SAM2 features in obsm['X_sam2'], not in X."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        adata = slide.to_anndata()
        assert "X_sam2" in adata.obsm
        assert adata.obsm["X_sam2"].shape == (10, 4)
        # SAM2 should NOT be in X
        assert "sam2_0" not in list(adata.var_names)
        assert "sam2_3" not in list(adata.var_names)


# ---------------------------------------------------------------------------
# Quality filter -> SlideAnalysis handoff
# ---------------------------------------------------------------------------


class TestQualityFilterFormat:
    """Verify quality filter output format is consumed correctly."""

    def test_filtered_detections_have_rf_prediction(self):
        """After quality filter, rf_prediction=1.0 for passing cells."""
        dets = _make_pipeline_detections(10)
        # Simulate quality_filter_detections.py: set rf_prediction=1.0 for passing
        for det in dets:
            det["rf_prediction"] = 1.0
        slide = SlideAnalysis.from_detections(dets)
        filtered = slide.filter(score_threshold=0.5)
        assert filtered.n_detections == 10  # all pass at 1.0

    def test_filtered_then_classified_roundtrip(self):
        """Quality filter -> marker classify -> SlideAnalysis -> filter(marker=...) works."""
        dets = _make_pipeline_detections(20)
        # Step 1: Simulate quality filter (set rf_prediction=1.0 for passing)
        for det in dets:
            det["rf_prediction"] = 1.0
        # Step 2: Marker classes are already in features (from _make_pipeline_detections)
        # Step 3: Load into SlideAnalysis
        slide = SlideAnalysis.from_detections(dets)
        # Step 4: filter(score_threshold=0.5) then filter(marker=..., positive=True)
        result = slide.filter(score_threshold=0.5).filter(marker="NeuN", positive=True)
        # All 20 pass score (rf_prediction=1.0), then NeuN positive: 13
        assert result.n_detections == 13
        for det in result.detections:
            assert det["features"]["NeuN_class"] == "positive"
            assert det["rf_prediction"] == 1.0

    def test_quality_filter_does_not_require_features_marker(self):
        """Quality filter works on detections that have no marker classes yet."""
        dets = _make_pipeline_detections(10)
        # Remove marker classes from features (pre-classification state)
        for det in dets:
            for key in list(det["features"].keys()):
                if "_class" in key or "_value" in key or "_threshold" in key:
                    del det["features"][key]
            if "marker_profile" in det["features"]:
                del det["features"]["marker_profile"]
            det["rf_prediction"] = 1.0
        slide = SlideAnalysis.from_detections(dets)
        filtered = slide.filter(score_threshold=0.5)
        assert filtered.n_detections == 10


# ---------------------------------------------------------------------------
# Contour data handoff
# ---------------------------------------------------------------------------


class TestContourHandoff:
    """Verify contour data flows through pipeline stages."""

    def test_contour_dilated_px_preserved_through_filter(self):
        """Contours survive SlideAnalysis.filter()."""
        dets = _make_pipeline_detections(10)
        slide = SlideAnalysis.from_detections(dets)
        filtered = slide.filter(score_threshold=0.5)
        for det in filtered.detections:
            contour = det.get("contour_dilated_px")
            assert contour is not None
            assert len(contour) == 4  # our test square
            # Verify structure: list of [x, y] pairs
            assert len(contour[0]) == 2

    def test_contours_property_returns_arrays(self):
        """SlideAnalysis.contours converts to numpy arrays."""
        dets = _make_pipeline_detections(5)
        slide = SlideAnalysis.from_detections(dets)
        contours = slide.contours
        assert len(contours) == 5
        for c in contours:
            assert isinstance(c, np.ndarray)
            assert c.shape == (4, 2)

    def test_save_load_preserves_contours(self, tmp_path):
        """save() then load() preserves contour data via atomic_json_dump."""
        dets = _make_pipeline_detections(5)
        slide = SlideAnalysis.from_detections(dets)
        out = tmp_path / "test_dets.json"
        slide.save(out)

        loaded = fast_json_load(out)
        assert len(loaded) == 5
        for orig, reloaded in zip(dets, loaded):
            assert reloaded["contour_dilated_px"] == orig["contour_dilated_px"]
            assert reloaded["global_center"] == orig["global_center"]
            assert reloaded["global_center_um"] == orig["global_center_um"]


# ---------------------------------------------------------------------------
# Background correction feature format
# ---------------------------------------------------------------------------


class TestBackgroundCorrectionFormat:
    """Verify bg-corrected features are in the expected format."""

    def test_bg_corrected_features_in_features_dict(self):
        """ch0_background, ch0_snr, ch0_median_raw in features dict."""
        dets = _make_pipeline_detections(5)
        slide = SlideAnalysis.from_detections(dets)
        df = slide.features_df
        # Background correction fields from pipeline
        assert "ch0_background" in df.columns
        assert "ch0_snr" in df.columns
        assert "ch0_median_raw" in df.columns
        # Second channel too
        assert "ch1_background" in df.columns
        assert "ch1_snr" in df.columns

    def test_bg_features_survive_roundtrip(self, tmp_path):
        """Background features survive save/load roundtrip."""
        dets = _make_pipeline_detections(5)
        out = tmp_path / "bg_test.json"
        atomic_json_dump(dets, out)
        loaded = fast_json_load(out)

        slide = SlideAnalysis.from_detections(loaded)
        df = slide.features_df
        # Verify values are numeric and reasonable
        assert df["ch0_background"].dtype in (np.float64, np.float32, float)
        assert (df["ch0_snr"] > 0).all()

    def test_bg_features_in_anndata_X(self):
        """Background correction features end up in AnnData X (not obsm)."""
        dets = _make_pipeline_detections(5)
        slide = SlideAnalysis.from_detections(dets)
        adata = slide.to_anndata()
        var_names = list(adata.var_names)
        assert "ch0_background" in var_names
        assert "ch0_snr" in var_names
        assert "ch0_median_raw" in var_names


# ---------------------------------------------------------------------------
# apply_marker_filter utility tests
# ---------------------------------------------------------------------------


class TestApplyMarkerFilter:
    """Tests for the shared apply_marker_filter utility."""

    def test_filter_toplevel_key(self):
        """Filter matches top-level detection key."""
        from segmentation.utils.detection_utils import apply_marker_filter

        dets = [{"MSLN_class": "positive"}, {"MSLN_class": "negative"}]
        result = apply_marker_filter(dets, "MSLN_class==positive")
        assert len(result) == 1
        assert result[0]["MSLN_class"] == "positive"

    def test_filter_features_key(self):
        """Filter matches key inside features dict (real pipeline format)."""
        from segmentation.utils.detection_utils import apply_marker_filter

        dets = [
            {"features": {"MSLN_class": "positive"}},
            {"features": {"MSLN_class": "negative"}},
        ]
        result = apply_marker_filter(dets, "MSLN_class==positive")
        assert len(result) == 1

    def test_filter_no_match(self):
        """Filter returns empty list when nothing matches."""
        from segmentation.utils.detection_utils import apply_marker_filter

        dets = [{"features": {"MSLN_class": "negative"}}]
        result = apply_marker_filter(dets, "MSLN_class==positive")
        assert len(result) == 0

    def test_filter_none_returns_all(self):
        """None filter returns all detections."""
        from segmentation.utils.detection_utils import apply_marker_filter

        dets = [{"a": 1}, {"b": 2}]
        result = apply_marker_filter(dets, None)
        assert len(result) == 2

    def test_filter_empty_string_returns_all(self):
        """Empty string filter returns all detections."""
        from segmentation.utils.detection_utils import apply_marker_filter

        result = apply_marker_filter([{"a": 1}], "")
        assert len(result) == 1

    def test_filter_malformed_no_equals(self):
        """Malformed filter (no ==) returns all with warning."""
        from segmentation.utils.detection_utils import apply_marker_filter

        dets = [{"MSLN_class": "positive"}]
        result = apply_marker_filter(dets, "MSLN_class positive")
        assert len(result) == 1  # returns all, not filtered

    def test_filter_spaces_around_equals(self):
        """Spaces around == are handled."""
        from segmentation.utils.detection_utils import apply_marker_filter

        dets = [{"features": {"MSLN_class": "positive"}}]
        result = apply_marker_filter(dets, "MSLN_class == positive")
        assert len(result) == 1

    def test_filter_empty_detections(self):
        """Empty detection list returns empty."""
        from segmentation.utils.detection_utils import apply_marker_filter

        result = apply_marker_filter([], "MSLN_class==positive")
        assert len(result) == 0
