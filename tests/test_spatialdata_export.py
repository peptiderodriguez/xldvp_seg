"""Tests for SpatialData export module."""

import numpy as np
import pytest

ad = pytest.importorskip("anndata")

from xldvp_seg.io.spatialdata_export import (
    _discover_features,
    _discover_obs_classes,
    build_anndata,
)


def _make_detections(n=50):
    """Create minimal detections with features for SpatialData export."""
    np.random.seed(42)
    dets = []
    for i in range(n):
        det = {
            "uid": f"cell_{i}",
            "cell_type": "cell",
            "slide_name": "test_slide",
            "pixel_size_um": 0.5,
            "global_center": [100.0 + i * 10, 200.0 + i * 5],
            "global_center_um": [50.0 + i * 5, 100.0 + i * 2.5],
            "features": {
                "area": float(np.random.uniform(50, 500)),
                "circularity": float(np.random.uniform(0.3, 1.0)),
                "solidity": float(np.random.uniform(0.7, 1.0)),
                "aspect_ratio": float(np.random.uniform(1.0, 3.0)),
                "ch0_mean": float(np.random.uniform(0, 100)),
                "ch0_median": float(np.random.uniform(0, 80)),
                "ch1_mean": float(np.random.uniform(0, 200)),
                "ch1_snr": float(np.random.uniform(0.5, 5.0)),
                "sam2_0": float(np.random.randn()),
                "sam2_1": float(np.random.randn()),
                "sam2_2": float(np.random.randn()),
            },
            "contour_px": [[0, 0], [10, 0], [10, 10], [0, 10]],
        }
        dets.append(det)
    return dets


class TestDiscoverFeatures:
    def test_finds_morph_and_channel(self):
        dets = _make_detections(5)
        morph, channel, embeddings = _discover_features(dets)
        # Morph features should include basic shape features
        all_feats = list(morph) + list(channel)
        assert "area" in all_feats or "circularity" in all_feats
        # Channel stats: ch0_mean may be in morph or channel depending on categorization
        assert any("ch0" in f for f in all_feats)

    def test_excludes_embeddings(self):
        dets = _make_detections(5)
        morph, channel, embeddings = _discover_features(dets)
        all_names = list(morph) + list(channel)
        assert not any(n.startswith("sam2_") for n in all_names)
        # SAM2 should be in embeddings
        assert any("sam2" in k for k in embeddings)

    def test_empty_detections(self):
        morph, channel, embeddings = _discover_features([])
        assert len(morph) == 0
        assert len(channel) == 0


class TestDiscoverObsClasses:
    def test_finds_class_columns(self):
        dets = _make_detections(3)
        for d in dets:
            d["features"]["NeuN_class"] = "positive"
            d["features"]["CD31_class"] = "negative"
        classes = _discover_obs_classes(dets)
        assert "NeuN_class" in classes
        assert "CD31_class" in classes

    def test_no_class_columns(self):
        dets = _make_detections(3)
        classes = _discover_obs_classes(dets)
        assert len(classes) == 0


class TestBuildAnndata:
    def test_shape(self):
        dets = _make_detections(50)
        adata = build_anndata(dets, "cell")
        assert adata.n_obs == 50
        assert adata.X.shape[0] == 50
        assert adata.X.shape[1] > 0  # at least some features in X

    def test_obs_columns(self):
        dets = _make_detections(10)
        adata = build_anndata(dets, "cell")
        assert "cell_type" in adata.obs.columns
        assert "slide_name" in adata.obs.columns
        assert "uid" in adata.obs.columns

    def test_spatial_in_obsm(self):
        dets = _make_detections(10)
        adata = build_anndata(dets, "cell")
        assert "spatial" in adata.obsm
        assert adata.obsm["spatial"].shape == (10, 2)

    def test_sam2_in_obsm_not_x(self):
        dets = _make_detections(10)
        adata = build_anndata(dets, "cell")
        # SAM2 features should be in obsm as X_sam2, not in X
        assert "X_sam2" in adata.obsm, f"Expected X_sam2 in obsm, got: {list(adata.obsm.keys())}"
        assert adata.obsm["X_sam2"].shape[0] == 10
        # X should not contain sam2 features
        var_names = list(adata.var_names)
        assert not any(v.startswith("sam2_") for v in var_names)

    def test_unique_obs_names(self):
        dets = _make_detections(10)
        adata = build_anndata(dets, "cell")
        assert len(set(adata.obs_names)) == len(adata.obs_names)

    def test_with_marker_classes(self):
        dets = _make_detections(10)
        for d in dets[:5]:
            d["features"]["NeuN_class"] = "positive"
        for d in dets[5:]:
            d["features"]["NeuN_class"] = "negative"
        adata = build_anndata(dets, "cell")
        # NeuN_class should appear in obs
        assert "NeuN_class" in adata.obs.columns


class TestBuildShapes:
    def test_shapes_from_contours(self):
        geopandas = pytest.importorskip("geopandas")
        from xldvp_seg.io.spatialdata_export import build_shapes

        dets = _make_detections(5)
        shapes = build_shapes(dets, "cell", tiles_dir=None, pixel_size_um=0.5)
        assert isinstance(shapes, dict)
        # Should have at least one layer
        if shapes:
            for name, gdf in shapes.items():
                assert isinstance(gdf, geopandas.GeoDataFrame)
                assert len(gdf) > 0


class TestRoundtripZarr:
    def test_export_and_reload(self, tmp_path):
        spatialdata = pytest.importorskip("spatialdata")
        from xldvp_seg.io.spatialdata_export import export_spatialdata

        dets = _make_detections(20)
        output = tmp_path / "test_spatialdata.zarr"
        export_spatialdata(
            dets,
            output_path=str(output),
            cell_type="cell",
            pixel_size_um=0.5,
            run_squidpy=False,
        )
        assert output.exists()

        sdata = spatialdata.read_zarr(str(output))
        # Verify table exists and has correct shape
        assert (
            hasattr(sdata, "tables") and sdata.tables
        ), "SpatialData has no tables after roundtrip"
        table_name = list(sdata.tables.keys())[0]
        table = sdata.tables[table_name]
        assert table.n_obs == 20
