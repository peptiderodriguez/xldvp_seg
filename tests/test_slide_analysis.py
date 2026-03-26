"""Tests for segmentation.core.slide_analysis.SlideAnalysis.

Tests the central state object wrapping pipeline output: construction,
filtering, DataFrame export, AnnData export, and repr behaviour.

Run with: pytest tests/test_slide_analysis.py -v
"""

import numpy as np

from segmentation.core import SlideAnalysis


def _make_detections(n=5):
    """Create sample detection dicts matching real pipeline output format.

    classify_markers.py stores marker classes inside det["features"],
    NOT at the top level. This fixture matches that real format.
    """
    dets = []
    for i in range(n):
        marker_class = "positive" if i % 2 == 0 else "negative"
        marker_profile = f"NeuN{'+' if i % 2 == 0 else '-'}"
        dets.append(
            {
                "uid": f"slide_cell_{i * 100}_{i * 200}",
                "rf_prediction": 0.3 + i * 0.15,  # 0.3, 0.45, 0.6, 0.75, 0.9
                "global_center": [i * 100, i * 200],
                "features": {
                    "area": 500 + i * 100,
                    "solidity": 0.8 + i * 0.02,
                    "eccentricity": 0.3 + i * 0.05,
                    "sam2_0": float(i),
                    "sam2_1": float(i + 1),
                    # Marker classes in features (real pipeline format)
                    "NeuN_class": marker_class,
                    "marker_profile": marker_profile,
                },
            }
        )
    return dets


class TestSlideAnalysisConstruction:
    def test_from_detections(self):
        dets = _make_detections(5)
        slide = SlideAnalysis.from_detections(dets)
        assert slide.n_detections == 5
        assert len(slide) == 5

    def test_from_empty_list(self):
        slide = SlideAnalysis.from_detections([])
        assert slide.n_detections == 0

    def test_default_init(self):
        slide = SlideAnalysis()
        # No detections provided, no file path -> empty list
        assert slide.n_detections == 0


class TestFeaturesDataFrame:
    def test_features_df_shape(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        df = slide.features_df
        assert len(df) == 5
        assert df.index.name == "uid"
        assert "area" in df.columns

    def test_features_df_contains_morph(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        df = slide.features_df
        assert "solidity" in df.columns

    def test_features_df_contains_sam2(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        df = slide.features_df
        assert "sam2_0" in df.columns
        assert "sam2_1" in df.columns

    def test_features_df_contains_rf_prediction(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        df = slide.features_df
        assert "rf_prediction" in df.columns

    def test_features_df_contains_marker_class(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        df = slide.features_df
        assert "NeuN_class" in df.columns

    def test_features_df_contains_marker_profile(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        df = slide.features_df
        assert "marker_profile" in df.columns

    def test_features_df_uid_index_values(self):
        dets = _make_detections(3)
        slide = SlideAnalysis.from_detections(dets)
        df = slide.features_df
        expected_uids = [d["uid"] for d in dets]
        assert list(df.index) == expected_uids

    def test_features_df_empty(self):
        slide = SlideAnalysis.from_detections([])
        assert slide.features_df.empty


class TestFiltering:
    def test_filter_score_threshold(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        filtered = slide.filter(score_threshold=0.5)
        # rf_predictions are 0.3, 0.45, 0.6, 0.75, 0.9 -- 3 >= 0.5
        assert filtered.n_detections == 3

    def test_filter_marker_positive(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        pos = slide.filter(marker="NeuN", positive=True)
        # indices 0, 2, 4 are positive
        assert pos.n_detections == 3

    def test_filter_marker_negative(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        neg = slide.filter(marker="NeuN", positive=False)
        # indices 1, 3 are negative
        assert neg.n_detections == 2

    def test_filter_returns_new_instance(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        filtered = slide.filter(score_threshold=0.5)
        assert filtered is not slide
        assert slide.n_detections == 5  # original unchanged

    def test_filter_combined(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        filtered = slide.filter(score_threshold=0.5, marker="NeuN", positive=True)
        # Score >= 0.5: indices 2 (0.6), 3 (0.75), 4 (0.9)
        # NeuN positive: indices 0, 2, 4
        # Intersection: indices 2, 4
        assert filtered.n_detections == 2

    def test_filter_no_criteria(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        filtered = slide.filter()
        assert filtered.n_detections == 5

    def test_filter_all_excluded(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        filtered = slide.filter(score_threshold=1.0)
        assert filtered.n_detections == 0

    def test_chained_filters(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        result = slide.filter(score_threshold=0.5).filter(marker="NeuN", positive=True)
        assert result.n_detections == 2


class TestAnnDataExport:
    def test_to_anndata_shape(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        adata = slide.to_anndata()
        assert adata.n_obs == 5

    def test_to_anndata_sam2_obsm(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        adata = slide.to_anndata()
        assert "X_sam2" in adata.obsm
        assert adata.obsm["X_sam2"].shape == (5, 2)

    def test_to_anndata_obs_columns(self):
        slide = SlideAnalysis.from_detections(_make_detections(5))
        adata = slide.to_anndata()
        assert "NeuN_class" in adata.obs.columns

    def test_to_anndata_empty(self):
        slide = SlideAnalysis.from_detections([])
        adata = slide.to_anndata()
        assert adata.n_obs == 0

    def test_to_anndata_morph_in_X(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        adata = slide.to_anndata()
        # area and solidity should be in X (morph features)
        assert "area" in list(adata.var_names)
        assert "solidity" in list(adata.var_names)

    def test_to_anndata_sam2_not_in_X(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        adata = slide.to_anndata()
        # sam2 features should NOT be in X (they go to obsm)
        assert "sam2_0" not in list(adata.var_names)

    def test_to_anndata_X_finite(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        adata = slide.to_anndata()
        assert np.all(np.isfinite(adata.X))


class TestRepr:
    def test_repr_after_access(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        # Access detections to populate _detections from _detections_override
        _ = slide.detections
        r = repr(slide)
        assert "SlideAnalysis" in r
        assert "n=3" in r

    def test_repr_before_access(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        # Before accessing .detections, repr should show "?" (no lazy load triggered)
        r = repr(slide)
        assert "SlideAnalysis" in r
        assert "n=?" in r

    def test_repr_unknown_when_not_loaded(self):
        slide = SlideAnalysis()
        r = repr(slide)
        assert "SlideAnalysis" in r
        # No detections loaded yet, should show "?"
        assert "n=?" in r


class TestMetadataProperties:
    def test_cell_type_default(self):
        slide = SlideAnalysis.from_detections([])
        assert slide.cell_type == "unknown"

    def test_slide_name_default(self):
        slide = SlideAnalysis.from_detections([])
        # No output_dir and no summary -> "unknown"
        assert slide.slide_name == "unknown"

    def test_pixel_size_default(self):
        slide = SlideAnalysis.from_detections([])
        assert slide.pixel_size_um == 0.0

    def test_output_dir_none(self):
        slide = SlideAnalysis.from_detections([])
        assert slide.output_dir is None

    def test_detections_path_none(self):
        slide = SlideAnalysis.from_detections([])
        assert slide.detections_path is None


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        dets = _make_detections(3)
        slide = SlideAnalysis.from_detections(dets)
        out = tmp_path / "test_detections.json"
        slide.save(out)
        assert out.exists()

        # Verify saved file is valid JSON and has correct count
        from segmentation.utils.json_utils import fast_json_load

        loaded = fast_json_load(out)
        assert len(loaded) == 3
        assert loaded[0]["uid"] == dets[0]["uid"]
