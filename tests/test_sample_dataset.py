"""Tests for segmentation.datasets.sample()."""

import pytest


class TestSampleDataset:

    def test_returns_required_keys(self):
        from segmentation.datasets import sample
        data = sample()
        for key in ["detections", "n_clusters", "cluster_labels",
                     "pixel_size_um", "channel_names"]:
            assert key in data, f"Missing key: {key}"

    def test_detection_count(self):
        from segmentation.datasets import sample
        data = sample(n_cells=100)
        assert len(data["detections"]) == 100

    def test_custom_params(self):
        from segmentation.datasets import sample
        data = sample(n_cells=50, n_clusters=3, n_channels=2)
        assert len(data["detections"]) == 50
        assert data["n_clusters"] == 3
        assert len(data["channel_names"]) == 2

    def test_detection_has_features(self):
        from segmentation.datasets import sample
        det = sample(n_cells=5)["detections"][0]
        assert "features" in det
        assert "area" in det["features"]

    def test_sam2_embeddings_present(self):
        from segmentation.datasets import sample
        det = sample(n_cells=5)["detections"][0]
        assert "sam2_0" in det["features"]
        assert "sam2_255" in det["features"]

    def test_channel_features_present(self):
        from segmentation.datasets import sample
        det = sample(n_cells=5, n_channels=3)["detections"][0]
        for ch in range(3):
            assert f"ch{ch}_mean" in det["features"]
            assert f"ch{ch}_snr" in det["features"]

    def test_global_center(self):
        from segmentation.datasets import sample
        det = sample(n_cells=5)["detections"][0]
        assert "global_center" in det
        assert len(det["global_center"]) == 2
        assert "global_center_um" in det

    def test_cluster_labels_valid(self):
        from segmentation.datasets import sample
        data = sample(n_cells=100, n_clusters=5)
        assert all(0 <= l < 5 for l in data["cluster_labels"])

    def test_reproducibility(self):
        from segmentation.datasets import sample
        d1 = sample(seed=42)
        d2 = sample(seed=42)
        assert d1["detections"][0]["global_center"] == d2["detections"][0]["global_center"]

    def test_different_seeds_differ(self):
        from segmentation.datasets import sample
        d1 = sample(seed=1)
        d2 = sample(seed=2)
        assert d1["detections"][0]["global_center"] != d2["detections"][0]["global_center"]
