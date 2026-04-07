"""Tests for xldvp_seg.api -- analysis tool wrappers, plotting, and I/O.

Uses unittest.mock.patch to avoid heavy imports (sklearn, scripts, GPU models).
Tests that tl.score(), tl.markers(), tl.train(), pl.umap(),
and pp.detect() pass arguments correctly and return expected results.

Run with: pytest tests/test_api.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xldvp_seg.api import io, pl, pp, tl
from xldvp_seg.core import SlideAnalysis


def _make_detections(n=5):
    """Create sample detection dicts with features for scoring."""
    dets = []
    for i in range(n):
        dets.append(
            {
                "uid": f"slide_cell_{i * 100}_{i * 200}",
                "rf_prediction": 0.5,
                "global_center": [i * 100, i * 200],
                "features": {
                    "area": 500 + i * 100,
                    "solidity": 0.8 + i * 0.02,
                    "ch1_mean": 100.0 + i * 10,
                    "ch1_snr": 1.0 + i * 0.5,
                    "ch1_background": 50.0,
                    "ch1_median_raw": 80.0 + i * 5,
                    "sam2_0": float(i),
                    "sam2_1": float(i + 1),
                },
            }
        )
    return dets


class TestTlScore:
    """Tests for tl.score() -- RF classifier scoring."""

    @patch("xldvp_seg.utils.detection_utils.load_rf_classifier")
    def test_score_applies_predictions(self, mock_load_clf):
        """Verify score() calls predict_proba and writes rf_prediction."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array(
            [
                [0.2, 0.8],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.5, 0.5],
                [0.3, 0.7],
            ]
        )
        mock_load_clf.return_value = {
            "pipeline": mock_pipeline,
            "feature_names": ["area", "solidity"],
            "type": "rf",
            "raw_meta": {},
        }

        dets = _make_detections(5)
        slide = SlideAnalysis.from_detections(dets)
        result = tl.score(slide, classifier="fake_classifier.pkl")

        assert result is slide
        mock_load_clf.assert_called_once_with("fake_classifier.pkl")
        mock_pipeline.predict_proba.assert_called_once()
        X_arg = mock_pipeline.predict_proba.call_args[0][0]
        assert X_arg.shape == (5, 2)
        assert slide.detections[0]["rf_prediction"] == pytest.approx(0.8)
        assert slide.detections[1]["rf_prediction"] == pytest.approx(0.4)

    @patch("xldvp_seg.utils.detection_utils.load_rf_classifier")
    def test_score_empty_detections(self, mock_load_clf):
        """Score on empty slide returns slide without calling classifier."""
        mock_load_clf.return_value = {
            "pipeline": MagicMock(),
            "feature_names": ["area"],
            "type": "rf",
            "raw_meta": {},
        }
        slide = SlideAnalysis.from_detections([])
        result = tl.score(slide, classifier="clf.pkl")
        assert result is slide
        mock_load_clf.return_value["pipeline"].predict_proba.assert_not_called()

    @patch("xldvp_seg.utils.detection_utils.load_rf_classifier")
    def test_score_missing_features(self, mock_load_clf):
        """Detections missing required features are skipped."""
        mock_pipeline = MagicMock()
        mock_load_clf.return_value = {
            "pipeline": mock_pipeline,
            "feature_names": ["area", "nonexistent_feature"],
            "type": "rf",
            "raw_meta": {},
        }
        dets = _make_detections(3)
        slide = SlideAnalysis.from_detections(dets)
        result = tl.score(slide, classifier="clf.pkl")
        assert result is slide
        mock_pipeline.predict_proba.assert_not_called()

    @patch("xldvp_seg.utils.detection_utils.load_rf_classifier")
    def test_score_custom_field(self, mock_load_clf):
        """Score with custom score_field name."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
        mock_load_clf.return_value = {
            "pipeline": mock_pipeline,
            "feature_names": ["area"],
            "type": "rf",
            "raw_meta": {},
        }
        dets = _make_detections(1)
        slide = SlideAnalysis.from_detections(dets)
        tl.score(slide, classifier="clf.pkl", score_field="custom_score")
        assert "custom_score" in slide.detections[0]
        assert slide.detections[0]["custom_score"] == pytest.approx(0.9)

    @patch("xldvp_seg.utils.detection_utils.load_rf_classifier")
    def test_score_invalidates_features_df_cache(self, mock_load_clf):
        """Scoring should clear the cached features_df."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array(
            [
                [0.2, 0.8],
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        )
        mock_load_clf.return_value = {
            "pipeline": mock_pipeline,
            "feature_names": ["area"],
            "type": "rf",
            "raw_meta": {},
        }
        dets = _make_detections(3)
        slide = SlideAnalysis.from_detections(dets)
        _ = slide.features_df
        assert slide._features_df is not None

        tl.score(slide, classifier="clf.pkl")
        assert slide._features_df is None

    @patch("xldvp_seg.utils.detection_utils.load_rf_classifier")
    def test_score_partial_features(self, mock_load_clf):
        """When some detections have features and some don't, only valid ones scored."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_load_clf.return_value = {
            "pipeline": mock_pipeline,
            "feature_names": ["area", "solidity"],
            "type": "rf",
            "raw_meta": {},
        }
        dets = _make_detections(2)
        # Remove solidity from the second detection
        del dets[1]["features"]["solidity"]
        slide = SlideAnalysis.from_detections(dets)
        tl.score(slide, classifier="clf.pkl")
        # Only 1 detection should be scored
        mock_pipeline.predict_proba.assert_called_once()
        X_arg = mock_pipeline.predict_proba.call_args[0][0]
        assert X_arg.shape == (1, 2)


class TestTlMarkers:
    """Tests for tl.markers() -- marker classification."""

    def test_markers_empty_detections_returns_early(self):
        """markers() on empty slide should return slide immediately."""
        slide = SlideAnalysis.from_detections([])

        mock_classify = MagicMock()

        with patch(
            "xldvp_seg.analysis.marker_classification.classify_single_marker",
            mock_classify,
        ):
            result = tl.markers(slide, marker_channels=[1], marker_names=["NeuN"])
            assert result is slide
            # classify_single_marker should NOT have been called
            mock_classify.assert_not_called()

    def test_markers_calls_classify_per_channel(self):
        """markers() should call classify_single_marker for each channel."""
        mock_classify = MagicMock(
            return_value={
                "n_positive": 3,
                "n_negative": 2,
                "threshold": 1.5,
            }
        )

        dets = _make_detections(5)
        slide = SlideAnalysis.from_detections(dets)

        with patch(
            "xldvp_seg.analysis.marker_classification.classify_single_marker",
            mock_classify,
        ):
            result = tl.markers(
                slide,
                marker_channels=[1, 2],
                marker_names=["NeuN", "CD31"],
            )

        assert result is slide
        # Should have been called twice (once per marker)
        assert mock_classify.call_count == 2
        # Verify first call had correct channel/name
        first_call_kwargs = mock_classify.call_args_list[0]
        assert first_call_kwargs.kwargs["channel"] == 1
        assert first_call_kwargs.kwargs["marker_name"] == "NeuN"
        second_call_kwargs = mock_classify.call_args_list[1]
        assert second_call_kwargs.kwargs["channel"] == 2
        assert second_call_kwargs.kwargs["marker_name"] == "CD31"

    def test_markers_builds_marker_profile(self):
        """markers() should build marker_profile from marker classes."""

        def fake_classify(detections, channel, marker_name, **kwargs):
            target = "positive" if channel == 1 else "negative"
            for det in detections:
                feat = det.setdefault("features", {})
                feat[f"{marker_name}_class"] = target
            return {"n_positive": len(detections), "n_negative": 0, "threshold": 1.5}

        dets = _make_detections(3)
        slide = SlideAnalysis.from_detections(dets)

        with patch(
            "xldvp_seg.analysis.marker_classification.classify_single_marker",
            fake_classify,
        ):
            tl.markers(
                slide,
                marker_channels=[1, 2],
                marker_names=["NeuN", "CD31"],
            )

        # All detections should have marker_profile
        for det in slide.detections:
            feat = det.get("features", {})
            assert "marker_profile" in feat
            assert feat["marker_profile"] == "NeuN+/CD31-"

    def test_markers_invalidates_features_df_cache(self):
        """markers() should clear cached features_df."""
        mock_classify = MagicMock(
            return_value={
                "n_positive": 2,
                "n_negative": 1,
                "threshold": 1.0,
            }
        )

        dets = _make_detections(3)
        slide = SlideAnalysis.from_detections(dets)
        _ = slide.features_df
        assert slide._features_df is not None

        with patch(
            "xldvp_seg.analysis.marker_classification.classify_single_marker",
            mock_classify,
        ):
            tl.markers(slide, marker_channels=[1], marker_names=["NeuN"])

        assert slide._features_df is None

    def test_marker_profile_logic(self):
        """Test the marker_profile construction logic in isolation."""
        dets = _make_detections(3)
        for i, det in enumerate(dets):
            feat = det.setdefault("features", {})
            feat["NeuN_class"] = "positive" if i % 2 == 0 else "negative"
            feat["CD31_class"] = "negative" if i % 2 == 0 else "positive"

        marker_names = ["NeuN", "CD31"]
        for det in dets:
            feat = det.get("features", {})
            parts = []
            for name in marker_names:
                cls = feat.get(f"{name}_class", "negative")
                parts.append(f"{name}+" if cls == "positive" else f"{name}-")
            feat["marker_profile"] = "/".join(parts)

        assert dets[0]["features"]["marker_profile"] == "NeuN+/CD31-"
        assert dets[1]["features"]["marker_profile"] == "NeuN-/CD31+"
        assert dets[2]["features"]["marker_profile"] == "NeuN+/CD31-"


class TestTlTrain:
    """Tests for tl.train() -- RF classifier training."""

    def test_train_requires_detections_path(self):
        """train() should raise ValueError if slide has no detections_path."""
        slide = SlideAnalysis.from_detections(_make_detections(3))
        assert slide.detections_path is None
        with pytest.raises(ValueError, match="no detections_path"):
            tl.train(slide, annotations="annotations.json")

    def test_train_returns_dict(self, tmp_path):
        """train() should return a dict with training metrics."""
        import json

        det_path = tmp_path / "cell_detections.json"
        dets = _make_detections(10)
        det_path.write_text(json.dumps(dets))

        slide = SlideAnalysis.load(tmp_path)
        assert slide.detections_path is not None

        ann_path = tmp_path / "annotations.json"
        annotations = {dets[i]["uid"]: (1 if i < 5 else 0) for i in range(10)}
        ann_path.write_text(json.dumps(annotations))

        fake_X = np.random.rand(10, 3)
        fake_y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        fake_feature_names = ["area", "solidity", "sam2_0"]

        mock_load_fn = MagicMock(return_value=(fake_X, fake_y, fake_feature_names))
        mock_rf_instance = MagicMock()
        mock_rf_instance.fit = MagicMock()
        mock_rf_cls = MagicMock(return_value=mock_rf_instance)
        mock_cv = MagicMock(return_value=np.array([0.9, 0.85, 0.88, 0.92, 0.87]))
        mock_joblib = MagicMock()

        with (
            patch(
                "xldvp_seg.training.feature_loader.load_features_and_annotations",
                mock_load_fn,
            ),
            patch(
                "sklearn.ensemble.RandomForestClassifier",
                mock_rf_cls,
            ),
            patch(
                "sklearn.model_selection.cross_val_score",
                mock_cv,
            ),
            patch(
                "joblib.dump",
                mock_joblib.dump,
            ),
        ):
            output_pkl = tmp_path / "test_classifier.pkl"
            result = tl.train(
                slide,
                annotations=str(ann_path),
                feature_set="morph",
                output_path=str(output_pkl),
            )

            assert isinstance(result, dict)
            assert result["feature_set"] == "morph"
            assert "cv_f1_mean" in result
            assert "cv_f1_std" in result
            assert "n_positive" in result
            assert "n_negative" in result
            assert result["n_positive"] == 5
            assert result["n_negative"] == 5

            # Verify load_features_and_annotations was called with correct args
            mock_load_fn.assert_called_once()
            call_args = mock_load_fn.call_args
            assert str(det_path) in call_args[0][0]
            assert str(ann_path) in call_args[0][1]

            # Verify RF was trained on all data
            mock_rf_instance.fit.assert_called_once()

    def test_train_no_annotated_raises(self, tmp_path):
        """train() should raise ValueError if no annotated detections found."""
        import json

        det_path = tmp_path / "cell_detections.json"
        dets = _make_detections(5)
        det_path.write_text(json.dumps(dets))

        slide = SlideAnalysis.load(tmp_path)
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps({}))

        # Return empty X,y from load_features_and_annotations
        mock_load_fn = MagicMock(return_value=(np.array([]), np.array([]), []))

        with patch(
            "xldvp_seg.training.feature_loader.load_features_and_annotations",
            mock_load_fn,
        ):
            with pytest.raises(ValueError, match="No annotated detections"):
                tl.train(slide, annotations=str(ann_path))


class TestPpDetect:
    """Tests for pp.detect() -- command builder."""

    def test_detect_returns_command_string(self):
        cmd = pp.detect(czi_path="/fake/slide.czi", cell_type="cell")
        assert isinstance(cmd, str)
        assert "xlseg detect" in cmd
        assert "--czi-path /fake/slide.czi" in cmd
        assert "--cell-type cell" in cmd

    def test_detect_includes_channel_spec(self):
        cmd = pp.detect(
            czi_path="/fake/slide.czi",
            channel_spec="cyto=PM,nuc=488",
        )
        assert "cyto=PM,nuc=488" in cmd

    def test_detect_includes_output_dir(self):
        cmd = pp.detect(czi_path="/fake/slide.czi", output_dir="/tmp/out")
        assert "--output-dir /tmp/out" in cmd

    def test_detect_does_not_execute(self):
        """detect() must return a string, not run anything."""
        result = pp.detect(czi_path="/nonexistent.czi")
        assert isinstance(result, str)


class TestPlUmap:
    """Tests for pl.umap() -- UMAP visualization."""

    def test_umap_requires_detections_path(self):
        slide = SlideAnalysis.from_detections(_make_detections(3))
        assert slide.detections_path is None
        with pytest.raises(ValueError, match="no detections_path"):
            pl.umap(slide)

    @patch("xldvp_seg.analysis.cluster_features.run_clustering")
    def test_umap_delegates_to_run_clustering(self, mock_run, tmp_path):
        """umap() should call run_clustering with methods='umap'."""
        import json

        det_path = tmp_path / "cell_detections.json"
        det_path.write_text(json.dumps(_make_detections(5)))
        slide = SlideAnalysis.load(tmp_path)

        out_dir = tmp_path / "umap_out"
        result = pl.umap(slide, output_dir=str(out_dir))

        assert result is slide
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["methods"] == "umap"
        assert str(det_path) in call_kwargs.kwargs["detections"]

    @patch("xldvp_seg.analysis.cluster_features.run_clustering")
    def test_umap_forwards_kwargs(self, mock_run, tmp_path):
        """Extra kwargs should be forwarded to run_clustering."""
        import json

        det_path = tmp_path / "cell_detections.json"
        det_path.write_text(json.dumps(_make_detections(5)))
        slide = SlideAnalysis.load(tmp_path)

        pl.umap(slide, output_dir=str(tmp_path / "out"), n_neighbors=50, min_dist=0.3)

        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["n_neighbors"] == 50
        assert call_kwargs["min_dist"] == 0.3


class TestPpInspect:
    """Tests for pp.inspect() -- CZI metadata inspection."""

    @patch("xldvp_seg.io.czi_loader.get_czi_metadata")
    def test_inspect_returns_metadata_dict(self, mock_get_meta):
        """inspect() should return the dict from get_czi_metadata."""
        expected = {"channels": [{"name": "AF488", "wavelength": 488}], "n_scenes": 1}
        mock_get_meta.return_value = expected

        result = pp.inspect("/fake/slide.czi")

        assert result is expected
        mock_get_meta.assert_called_once_with("/fake/slide.czi")

    @patch("xldvp_seg.io.czi_loader.get_czi_metadata")
    def test_inspect_converts_path_to_str(self, mock_get_meta):
        """inspect() should accept a Path object and convert to str."""
        from pathlib import Path

        mock_get_meta.return_value = {}
        pp.inspect(Path("/fake/slide.czi"))
        mock_get_meta.assert_called_once_with("/fake/slide.czi")


class TestIoReadProteomics:
    """Tests for io.read_proteomics() -- CSV and dvp-io paths."""

    def test_read_proteomics_csv(self, tmp_path):
        """read_proteomics() without search_engine reads a plain CSV."""
        import pandas as pd

        csv_path = tmp_path / "proteomics.csv"
        csv_path.write_text("well_id,ProteinA,ProteinB\nW1,1.0,2.0\nW2,3.0,4.0\n")

        result = io.read_proteomics(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["W1", "W2"]
        assert "ProteinA" in result.columns
        assert "ProteinB" in result.columns

    @patch("xldvp_seg.analysis.omic_linker.OmicLinker")
    def test_read_proteomics_via_dvpio(self, mock_linker_cls, tmp_path):
        """read_proteomics() with search_engine delegates to OmicLinker."""
        import pandas as pd

        mock_linker = MagicMock()
        mock_linker._proteomics = pd.DataFrame({"ProteinA": [1.0]}, index=["W1"])
        mock_linker_cls.return_value = mock_linker

        report_path = tmp_path / "report.tsv"
        report_path.write_text("dummy\n")

        result = io.read_proteomics(report_path, search_engine="diann")

        mock_linker.load_proteomics_report.assert_called_once()
        call_args = mock_linker.load_proteomics_report.call_args
        assert call_args[0][1] == "diann"
        assert isinstance(result, pd.DataFrame)


class TestTlCluster:
    """Tests for tl.cluster() -- feature clustering."""

    @patch("xldvp_seg.analysis.cluster_features.run_clustering")
    def test_cluster_requires_detections_path(self, mock_run):
        """cluster() raises if slide has no detections_path."""
        slide = SlideAnalysis.from_detections(_make_detections(3))
        with pytest.raises(Exception, match="no detections_path"):
            tl.cluster(slide)
        mock_run.assert_not_called()

    @patch("xldvp_seg.analysis.cluster_features.run_clustering")
    def test_cluster_forwards_kwargs(self, mock_run, tmp_path):
        """cluster() should forward feature_groups, methods, resolution to run_clustering."""
        import json

        det_path = tmp_path / "cell_detections.json"
        det_path.write_text(json.dumps(_make_detections(5)))
        slide = SlideAnalysis.load(tmp_path)

        out_dir = tmp_path / "cluster_out"
        tl.cluster(
            slide,
            feature_groups="morph,sam2",
            methods="umap",
            resolution=0.5,
            output_dir=str(out_dir),
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args.kwargs
        assert call_kwargs["feature_groups"] == "morph,sam2"
        assert call_kwargs["methods"] == "umap"
        assert call_kwargs["resolution"] == 0.5

    @patch("xldvp_seg.analysis.cluster_features.run_clustering")
    def test_cluster_returns_slide(self, mock_run, tmp_path):
        """cluster() should return the slide object."""
        import json

        det_path = tmp_path / "cell_detections.json"
        det_path.write_text(json.dumps(_make_detections(5)))
        slide = SlideAnalysis.load(tmp_path)

        result = tl.cluster(slide, output_dir=str(tmp_path / "out"))

        assert result is slide


class TestTlSpatial:
    """Tests for tl.spatial() -- spatial network analysis."""

    @patch("xldvp_seg.analysis.spatial_network.run_spatial_network")
    def test_spatial_returns_slide(self, mock_run):
        """spatial() should return the slide object."""
        mock_run.return_value = None
        dets = _make_detections(5)
        slide = SlideAnalysis.from_detections(dets)

        result = tl.spatial(slide, output_dir="/tmp/spatial_test_out")

        assert result is slide

    @patch("xldvp_seg.analysis.spatial_network.run_spatial_network")
    def test_spatial_calls_run_spatial_network(self, mock_run, tmp_path):
        """spatial() should call run_spatial_network with slide's detections."""
        mock_run.return_value = None
        dets = _make_detections(3)
        slide = SlideAnalysis.from_detections(dets)

        tl.spatial(slide, output_dir=str(tmp_path), max_edge_distance=75.0)

        mock_run.assert_called_once()
        call_args, call_kwargs = mock_run.call_args
        # First positional arg is detections list
        assert call_args[0] == dets
        assert call_kwargs["max_edge_distance"] == 75.0

    @patch("xldvp_seg.analysis.spatial_network.run_spatial_network")
    def test_spatial_updates_detections_when_result_returned(self, mock_run, tmp_path):
        """When run_spatial_network returns detections, slide._detections is updated."""
        updated_dets = _make_detections(3)
        for det in updated_dets:
            det["community_id"] = 42
        mock_run.return_value = updated_dets

        slide = SlideAnalysis.from_detections(_make_detections(3))
        tl.spatial(slide, output_dir=str(tmp_path))

        assert slide.detections[0].get("community_id") == 42


class TestIoToSpatialdata:
    """Tests for io.to_spatialdata() -- SpatialData export."""

    @patch("xldvp_seg.io.spatialdata_export.build_anndata")
    def test_to_spatialdata_builds_anndata(self, mock_build):
        """to_spatialdata() calls build_anndata with the slide's detections."""
        import anndata as ad

        mock_adata = MagicMock(spec=ad.AnnData)
        mock_adata.shape = (5, 10)
        mock_build.return_value = mock_adata

        dets = _make_detections(5)
        slide = SlideAnalysis.from_detections(dets)
        slide.summary["cell_type"] = "cell"

        # Patch spatialdata import to avoid needing the package
        with patch.dict("sys.modules", {"spatialdata": None, "spatialdata.models": None}):
            # When spatialdata is missing, should fall back to h5ad
            with patch.object(mock_adata, "write_h5ad"):
                io.to_spatialdata(slide, output_path="/tmp/test_spatialdata.zarr")

        mock_build.assert_called_once_with(dets, "cell")

    @patch("xldvp_seg.io.spatialdata_export.build_anndata")
    def test_to_spatialdata_returns_none_for_empty(self, mock_build):
        """to_spatialdata() returns None when slide has no detections."""
        slide = SlideAnalysis.from_detections([])
        result = io.to_spatialdata(slide)
        assert result is None
        mock_build.assert_not_called()
