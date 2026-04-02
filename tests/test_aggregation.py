"""Tests for segmentation.analysis.aggregation module.

Tests slide-level and cohort-level feature aggregation, plus AnnData conversion.
"""

import numpy as np
import pandas as pd

from segmentation.analysis.aggregation import (
    aggregate_cohort,
    aggregate_slide,
    cohort_to_anndata,
)
from segmentation.core import SlideAnalysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detections(n=10, prefix="slide"):
    """Create sample detections with numeric features and marker classes."""
    dets = []
    for i in range(n):
        dets.append(
            {
                "uid": f"{prefix}_cell_{i * 100}_{i * 200}",
                "rf_prediction": 0.5 + (i % 3) * 0.15,
                "NeuN_class": "positive" if i % 2 == 0 else "negative",
                "marker_profile": "NeuN+" if i % 2 == 0 else "NeuN-",
                "global_center": [i * 100, i * 200],
                "features": {
                    "area": 500 + i * 50,
                    "solidity": 0.85 + (i % 5) * 0.02,
                    "eccentricity": 0.3 + (i % 4) * 0.1,
                },
            }
        )
    return dets


def _make_slide(n=10, prefix="slide", name="test_slide"):
    """Create a SlideAnalysis with a unique slide name via output_dir."""
    # Use a non-existent path purely for the .name property (slide_name)
    from pathlib import Path

    return SlideAnalysis.from_detections(
        _make_detections(n, prefix=prefix),
        output_dir=Path(f"/tmp/fake/{name}"),
    )


# ---------------------------------------------------------------------------
# aggregate_slide
# ---------------------------------------------------------------------------


class TestAggregateSlide:

    def test_ungrouped_single_row(self):
        """Ungrouped aggregation produces exactly one row."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide)
        assert len(result) == 1

    def test_ungrouped_has_stats(self):
        """Ungrouped aggregation includes mean, median, std, count for features."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide)
        assert "area_mean" in result.columns
        assert "area_median" in result.columns
        assert "area_std" in result.columns
        assert "area_count" in result.columns

    def test_ungrouped_has_solidity_and_eccentricity(self):
        """Multiple numeric features produce aggregated columns."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide)
        assert "solidity_mean" in result.columns
        assert "eccentricity_mean" in result.columns

    def test_ungrouped_index_is_slide_name(self):
        """Index of ungrouped result is the slide name."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide)
        assert result.index[0] == slide.slide_name

    def test_ungrouped_area_mean_value(self):
        """Check that area_mean is computed correctly from known data."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide)
        # areas are 500, 550, 600, ..., 950 → mean = 725
        expected_mean = np.mean([500 + i * 50 for i in range(10)])
        assert abs(result["area_mean"].iloc[0] - expected_mean) < 1e-6

    def test_grouped_by_marker_profile(self):
        """Grouping by marker_profile produces one row per group."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide, group_by="marker_profile")
        # NeuN+ (i=0,2,4,6,8) and NeuN- (i=1,3,5,7,9) = 2 groups
        assert len(result) == 2

    def test_grouped_index_is_group_values(self):
        """Grouped result has group values as index."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide, group_by="marker_profile")
        assert set(result.index) == {"NeuN+", "NeuN-"}

    def test_grouped_has_multiindex_columns(self):
        """Grouped result columns are '{feature}_{stat}' formatted."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide, group_by="marker_profile")
        assert "area_mean" in result.columns
        assert "area_count" in result.columns

    def test_grouped_count_per_group(self):
        """Each group should have the correct count."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide, group_by="marker_profile")
        # Each group has 5 cells
        assert result.loc["NeuN+", "area_count"] == 5
        assert result.loc["NeuN-", "area_count"] == 5

    def test_empty_detections(self):
        """Empty detections produce an empty DataFrame."""
        slide = SlideAnalysis.from_detections([])
        result = aggregate_slide(slide)
        assert result.empty

    def test_single_detection(self):
        """Single detection produces valid aggregation."""
        slide = SlideAnalysis.from_detections(_make_detections(1))
        result = aggregate_slide(slide)
        assert len(result) == 1
        assert result["area_mean"].iloc[0] == 500

    def test_nonexistent_group_by_falls_through(self):
        """If group_by column doesn't exist, falls back to ungrouped."""
        slide = SlideAnalysis.from_detections(_make_detections(10))
        result = aggregate_slide(slide, group_by="nonexistent_column")
        # Should behave like ungrouped (single row)
        assert len(result) == 1

    def test_excludes_sam2_embeddings(self):
        """SAM2 embedding columns should not appear in aggregated output."""
        dets = _make_detections(5)
        for d in dets:
            d["features"]["sam2_0"] = 0.1
            d["features"]["sam2_1"] = 0.2
        slide = SlideAnalysis.from_detections(dets)
        result = aggregate_slide(slide)
        sam2_cols = [c for c in result.columns if "sam2_" in c]
        assert len(sam2_cols) == 0


# ---------------------------------------------------------------------------
# aggregate_cohort
# ---------------------------------------------------------------------------


class TestAggregateCohort:

    def test_multiple_slides(self):
        """Cohort aggregation produces one row per slide."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(3)]
        result = aggregate_cohort(slides)
        assert len(result) == 3

    def test_index_is_slide_name(self):
        """Cohort result is indexed by slide_name."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(3)]
        result = aggregate_cohort(slides)
        assert result.index.name == "slide_name"

    def test_has_n_cells_column(self):
        """Ungrouped cohort includes n_cells column."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(2)]
        result = aggregate_cohort(slides)
        assert "n_cells" in result.columns
        assert result["n_cells"].iloc[0] == 10

    def test_grouped_cohort(self):
        """Cohort with group_by produces rows for each slide x group combination."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(2)]
        result = aggregate_cohort(slides, group_by="marker_profile")
        # 2 slides x 2 groups = 4 rows
        assert len(result) == 4

    def test_grouped_cohort_has_group_column(self):
        """Grouped cohort includes a 'group' column."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(2)]
        result = aggregate_cohort(slides, group_by="marker_profile")
        assert "group" in result.columns
        assert set(result["group"]) == {"NeuN+", "NeuN-"}

    def test_single_slide_cohort(self):
        """Cohort with a single slide works."""
        slides = [_make_slide(10, name="only_slide")]
        result = aggregate_cohort(slides)
        assert len(result) == 1

    def test_empty_slides_skipped(self):
        """Slides with no detections are skipped in cohort aggregation."""
        slides = [
            _make_slide(10, prefix="real", name="real_slide"),
            SlideAnalysis.from_detections([]),  # empty
        ]
        result = aggregate_cohort(slides)
        assert len(result) == 1

    def test_all_empty_slides(self):
        """All-empty cohort produces empty DataFrame."""
        slides = [
            SlideAnalysis.from_detections([]),
            SlideAnalysis.from_detections([]),
        ]
        result = aggregate_cohort(slides)
        assert result.empty

    def test_different_detection_counts(self):
        """Slides with different detection counts aggregate correctly."""
        slides = [
            _make_slide(5, prefix="small", name="small_slide"),
            _make_slide(20, prefix="large", name="large_slide"),
        ]
        result = aggregate_cohort(slides)
        assert len(result) == 2
        # n_cells should reflect actual counts
        n_cells = result["n_cells"].tolist()
        assert 5 in n_cells
        assert 20 in n_cells


# ---------------------------------------------------------------------------
# cohort_to_anndata
# ---------------------------------------------------------------------------


class TestCohortToAnndata:

    def test_basic_conversion(self):
        """Basic cohort-to-AnnData conversion produces correct shape."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(3)]
        cohort = aggregate_cohort(slides)
        adata = cohort_to_anndata(cohort)
        assert adata.n_obs == 3

    def test_feature_columns_in_var(self):
        """AnnData var_names should contain the numeric feature columns."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(3)]
        cohort = aggregate_cohort(slides)
        adata = cohort_to_anndata(cohort)
        assert adata.n_vars > 0
        # n_cells is metadata (obs), not a feature (var)
        assert "n_cells" in adata.obs.columns
        assert "n_cells" not in list(adata.var_names)

    def test_x_has_no_nans(self):
        """AnnData X matrix should have NaN replaced with 0."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(3)]
        cohort = aggregate_cohort(slides)
        adata = cohort_to_anndata(cohort)
        assert not np.isnan(adata.X).any()

    def test_x_dtype_is_float32(self):
        """AnnData X should be float32."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(3)]
        cohort = aggregate_cohort(slides)
        adata = cohort_to_anndata(cohort)
        assert adata.X.dtype == np.float32

    def test_with_metadata(self):
        """cohort_to_anndata with metadata joins it into obs."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(3)]
        cohort = aggregate_cohort(slides)
        meta = pd.DataFrame(
            {"condition": ["control", "treatment", "treatment"]},
            index=cohort.index,
        )
        adata = cohort_to_anndata(cohort, metadata=meta)
        assert "condition" in adata.obs.columns
        assert list(adata.obs["condition"]) == ["control", "treatment", "treatment"]

    def test_grouped_cohort_has_group_in_obs(self):
        """Grouped cohort includes 'group' column in obs."""
        slides = [_make_slide(10, prefix=f"s{i}", name=f"slide_{i}") for i in range(2)]
        cohort = aggregate_cohort(slides, group_by="marker_profile")
        adata = cohort_to_anndata(cohort)
        assert "group" in adata.obs.columns
