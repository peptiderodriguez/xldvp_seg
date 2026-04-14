"""Tests for xldvp_seg.analysis.region_segmentation — pure functions on synthetic data."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xldvp_seg.analysis.region_segmentation import (
    build_tissue_mask,
    clean_labels,
    compute_region_nuc_stats,
    fill_labels,
)
from xldvp_seg.utils.image_utils import percentile_normalize

# ---------------------------------------------------------------------------
# percentile_normalize
# ---------------------------------------------------------------------------


class TestPercentileNormalize:
    def test_zero_input(self):
        arr = np.zeros((10, 10), dtype=np.uint16)
        result = percentile_normalize(arr)
        assert result.dtype == np.uint8
        assert result.max() == 0

    def test_uniform_input(self):
        arr = np.full((10, 10), 1000, dtype=np.uint16)
        result = percentile_normalize(arr)
        assert result.dtype == np.uint8
        # Uniform → all same value after normalize (or 0 if range is 0)

    def test_bimodal(self):
        arr = np.zeros((100, 100), dtype=np.uint16)
        arr[:50] = 500
        arr[50:] = 2000
        result = percentile_normalize(arr)
        assert result.dtype == np.uint8
        assert result.max() > 200
        assert result[:50].mean() < result[50:].mean()

    def test_uint8_passthrough(self):
        arr = np.array([[100, 200]], dtype=np.uint8)
        result = percentile_normalize(arr)
        assert np.array_equal(result, arr)

    def test_preserves_zeros(self):
        arr = np.zeros((100, 100), dtype=np.uint16)
        arr[20:80, 20:80] = np.random.default_rng(42).integers(500, 2000, size=(60, 60))
        result = percentile_normalize(arr)
        # Nonzero region should be bright, zero region stays zero
        assert result[20:80, 20:80].mean() > 50
        assert result[0, 0] == 0


# ---------------------------------------------------------------------------
# build_tissue_mask
# ---------------------------------------------------------------------------


class TestBuildTissueMask:
    def test_circular_blob(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        yy, xx = np.mgrid[0:200, 0:200]
        img[((yy - 100) ** 2 + (xx - 100) ** 2) < 60**2] = 200
        mask = build_tissue_mask(img, erode=3)
        assert mask.dtype == np.uint8
        assert mask.max() == 1
        # Center should be tissue
        assert mask[100, 100] == 1
        # Corners should be background
        assert mask[0, 0] == 0

    def test_empty_image(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        mask = build_tissue_mask(img)
        assert mask.sum() == 0

    def test_erode_parameter(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[20:180, 20:180] = 200
        mask_light = build_tissue_mask(img, erode=2)
        mask_heavy = build_tissue_mask(img, erode=15)
        # Heavy erosion → smaller mask
        assert mask_heavy.sum() < mask_light.sum()


# ---------------------------------------------------------------------------
# clean_labels
# ---------------------------------------------------------------------------


class TestCleanLabels:
    def test_removes_small_regions(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:20, 10:20] = 1  # 100 pixels
        labels[50:80, 50:80] = 2  # 900 pixels
        tissue = np.ones((100, 100), dtype=np.uint8)
        cleaned = clean_labels(labels, tissue, min_area=200)
        assert 1 not in np.unique(cleaned)
        assert 2 in np.unique(cleaned)

    def test_removes_low_tissue_overlap(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[0:30, 0:30] = 1  # 900 pixels
        tissue = np.zeros((100, 100), dtype=np.uint8)
        tissue[50:100, 50:100] = 1  # tissue is elsewhere
        cleaned = clean_labels(labels, tissue, min_area=100, min_tissue_overlap=0.5)
        assert 1 not in np.unique(cleaned)

    def test_keeps_valid_regions(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[30:70, 30:70] = 1
        tissue = np.ones((100, 100), dtype=np.uint8)
        cleaned = clean_labels(labels, tissue, min_area=100)
        assert 1 in np.unique(cleaned)

    def test_resizes_tissue_mask(self):
        labels = np.zeros((200, 200), dtype=np.int32)
        labels[50:150, 50:150] = 1
        tissue = np.ones((100, 100), dtype=np.uint8)  # different size
        cleaned = clean_labels(labels, tissue)
        assert 1 in np.unique(cleaned)


# ---------------------------------------------------------------------------
# fill_labels
# ---------------------------------------------------------------------------


class TestFillLabels:
    def test_fills_gaps(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:40, 10:40] = 1
        labels[60:90, 60:90] = 2
        tissue = np.ones((100, 100), dtype=np.uint8)
        filled = fill_labels(labels, tissue, fill_interstitial=False)
        # All tissue pixels should have a label
        assert (filled[tissue > 0] == 0).sum() == 0

    def test_masks_to_tissue(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:90, 10:90] = 1
        tissue = np.zeros((100, 100), dtype=np.uint8)
        tissue[20:80, 20:80] = 1
        filled = fill_labels(labels, tissue)
        # Outside tissue should be 0
        assert filled[0, 0] == 0
        assert filled[95, 95] == 0

    def test_interstitial_gets_own_id(self):
        # Two regions surrounding an unlabeled hole.
        # expand_labels will try to fill from both sides but can't reach the
        # center if it's equidistant. Use a 3-region layout where the hole
        # is genuinely enclosed.
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[0:45, :] = 1  # top half
        labels[55:100, :] = 2  # bottom half
        labels[:, 0:45] = 3  # left side (overwrites some of 1 and 2)
        labels[:, 55:100] = 4  # right side
        labels[45:55, 45:55] = 0  # enclosed hole in center
        tissue = np.ones((100, 100), dtype=np.uint8)
        filled = fill_labels(labels, tissue, fill_interstitial=True)
        hole_id = filled[50, 50]
        assert hole_id != 0
        # Should be a NEW id (not 1,2,3,4) because the hole is enclosed
        assert hole_id > 4

    def test_interstitial_disabled(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:90, 10:90] = 1
        labels[40:60, 40:60] = 0
        tissue = np.ones((100, 100), dtype=np.uint8)
        filled = fill_labels(labels, tissue, fill_interstitial=False)
        # Without interstitial fill, expand_labels absorbs hole into label 1
        assert filled[50, 50] == 1


# ---------------------------------------------------------------------------
# compute_region_nuc_stats
# ---------------------------------------------------------------------------


class TestComputeRegionNucStats:
    def _make_dets(self, organ_ids, n_nuclei_vals):
        return [
            {"organ_id": oid, "features": {"n_nuclei": nn}}
            for oid, nn in zip(organ_ids, n_nuclei_vals)
        ]

    def test_excludes_zero_nuclei(self):
        dets = self._make_dets([1, 1, 1, 1], [0, 1, 2, 1])
        stats = compute_region_nuc_stats(dets, exclude_zero_nuclei=True)
        assert stats[1]["count"] == 3  # 0-nucleus cell excluded
        assert stats[1]["mean_nuc"] == pytest.approx(4 / 3, abs=0.01)

    def test_includes_zero_when_disabled(self):
        dets = self._make_dets([1, 1, 1], [0, 1, 2])
        stats = compute_region_nuc_stats(dets, exclude_zero_nuclei=False)
        assert stats[1]["count"] == 3

    def test_excludes_background(self):
        dets = self._make_dets([0, 1, 0], [1, 1, 1])
        stats = compute_region_nuc_stats(dets)
        assert 0 not in stats
        assert 1 in stats

    def test_nuc_dist(self):
        dets = self._make_dets([1] * 5, [1, 1, 2, 2, 3])
        stats = compute_region_nuc_stats(dets)
        assert stats[1]["nuc_dist"] == {"1": 2, "2": 2, "3": 1}

    def test_median(self):
        dets = self._make_dets([1] * 5, [1, 1, 1, 2, 3])
        stats = compute_region_nuc_stats(dets)
        assert stats[1]["median_nuc"] == 1

    def test_missing_organ_id(self):
        dets = [{"features": {"n_nuclei": 1}}]  # no organ_id
        stats = compute_region_nuc_stats(dets)
        assert len(stats) == 0

    def test_missing_n_nuclei(self):
        dets = [{"organ_id": 1, "features": {}}]  # no n_nuclei
        stats = compute_region_nuc_stats(dets)
        assert len(stats) == 0


# ---------------------------------------------------------------------------
# segment_regions (mocked SAM2)
# ---------------------------------------------------------------------------


class TestSegmentRegionsMocked:
    def test_builds_label_map_from_masks(self):
        """Patch SAM2 and verify label map structure."""
        # Create fake SAM2 masks
        fake_masks = [
            {"segmentation": np.zeros((50, 50), dtype=bool), "area": 100},
            {"segmentation": np.zeros((50, 50), dtype=bool), "area": 400},
        ]
        fake_masks[0]["segmentation"][5:15, 5:15] = True  # small region
        fake_masks[1]["segmentation"][25:45, 25:45] = True  # larger region

        mock_gen = MagicMock()
        mock_gen.generate.return_value = fake_masks

        with patch("xldvp_seg.analysis.region_segmentation._load_sam2_model") as mock_load:
            mock_load.return_value = MagicMock()

            with patch.dict(
                "sys.modules",
                {
                    "sam2": MagicMock(),
                    "sam2.automatic_mask_generator": MagicMock(
                        SAM2AutomaticMaskGenerator=MagicMock(return_value=mock_gen)
                    ),
                    "sam2.build_sam": MagicMock(),
                },
            ):
                from xldvp_seg.analysis.region_segmentation import segment_regions

                img = np.zeros((50, 50, 3), dtype=np.uint8)
                labels, masks = segment_regions(img, sigma=0.5)

                assert labels.shape == (50, 50)
                assert labels.dtype == np.int32
                assert len(np.unique(labels)) >= 3  # bg + 2 regions
                assert len(masks) == 2
