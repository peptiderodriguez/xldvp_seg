"""Tests for xldvp_seg.roi — ROI-restricted cell detection utilities."""

import json

import numpy as np

from xldvp_seg.roi.circular_objects import find_circular_regions
from xldvp_seg.roi.common import (
    extract_region_bboxes,
    filter_detections_by_roi_mask,
    filter_tiles_by_rois,
    number_rois_spatial,
)
from xldvp_seg.roi.from_file import load_rois_from_mask, load_rois_from_polygons
from xldvp_seg.roi.marker_threshold import find_regions_by_marker_signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_with_blobs(shape=(1000, 1000), n_blobs=3, radius=80, seed=42):
    """Create a uint16 image with *n_blobs* bright circular blobs on dark bg."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 50, size=shape, dtype=np.uint16)
    centers = [(200, 200), (200, 700), (700, 500)][:n_blobs]
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for cy, cx in centers:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        img[mask] = rng.randint(500, 1000, dtype=np.uint16)
    return img, centers


def _label_from_blobs(shape=(1000, 1000), n_blobs=3, radius=80):
    """Create a label array with known circular blobs (consecutive labels)."""
    labels = np.zeros(shape, dtype=np.int32)
    centers = [(200, 200), (200, 700), (700, 500)][:n_blobs]
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for i, (cy, cx) in enumerate(centers, start=1):
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        labels[mask] = i
    return labels, centers


# ---------------------------------------------------------------------------
# find_regions_by_marker_signal
# ---------------------------------------------------------------------------


class TestFindRegionsByMarkerSignal:

    def test_finds_bright_regions(self):
        img, _ = _make_image_with_blobs(n_blobs=3, radius=80)
        channel_data = {0: img}
        labels, ds, signal = find_regions_by_marker_signal(
            channel_data,
            marker_channels=[0],
            pixel_size=1.0,
            downsample=2,
            blur_sigma_um=5.0,
            otsu_multiplier=1.0,
            min_area_um2=10.0,
            buffer_um=5.0,
        )
        n_regions = int(labels.max())
        assert n_regions >= 2, f"Expected >=2 regions, got {n_regions}"
        assert labels.shape == (500, 500)
        assert signal.shape == labels.shape

    def test_no_signal_returns_empty(self):
        """All-zero image should return no regions."""
        channel_data = {0: np.zeros((200, 200), dtype=np.uint16)}
        labels, ds, signal = find_regions_by_marker_signal(
            channel_data,
            marker_channels=[0],
            pixel_size=1.0,
            downsample=2,
        )
        assert int(labels.max()) == 0

    def test_missing_channel_skipped(self):
        """Requesting a missing channel should not crash."""
        img, _ = _make_image_with_blobs(n_blobs=1)
        channel_data = {0: img}
        labels, ds, _ = find_regions_by_marker_signal(
            channel_data,
            marker_channels=[0, 99],  # ch 99 does not exist
            pixel_size=1.0,
            downsample=2,
        )
        # Should still work with the one valid channel
        assert labels.shape[0] > 0


# ---------------------------------------------------------------------------
# extract_region_bboxes
# ---------------------------------------------------------------------------


class TestExtractRegionBboxes:

    def test_correct_coords(self):
        labels, centers = _label_from_blobs(shape=(1000, 1000), n_blobs=2, radius=40)
        ds_labels = labels[::2, ::2]  # downsample=2
        rois = extract_region_bboxes(
            ds_labels,
            downsample=2,
            x_start=100,
            y_start=200,
            full_width=1000,
            full_height=1000,
            padding_px=10,
        )
        assert len(rois) == 2
        for roi in rois:
            assert roi["roi_id"] > 0
            assert roi["height"] > 0
            assert roi["width"] > 0
            # gx0 = ax0 + x_start
            assert roi["gx0"] == roi["ax0"] + 100
            assert roi["gy0"] == roi["ay0"] + 200
            assert roi["area_px"] > 0

    def test_padding_clipped_to_boundaries(self):
        """Padding should not exceed slide dimensions."""
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[0:5, 0:5] = 1  # region touching the top-left edge
        rois = extract_region_bboxes(
            labels, downsample=1, x_start=0, y_start=0, full_width=50, full_height=50, padding_px=20
        )
        assert len(rois) == 1
        assert rois[0]["ay0"] == 0  # clipped to 0
        assert rois[0]["ax0"] == 0

    def test_empty_labels(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        rois = extract_region_bboxes(labels, 1, 0, 0, 100, 100)
        assert rois == []


# ---------------------------------------------------------------------------
# number_rois_spatial
# ---------------------------------------------------------------------------


class TestNumberRoisSpatial:

    def _make_rois(self):
        """Three ROIs: two in top row, one in bottom row."""
        return [
            {"gy0": 100, "gx0": 500, "height": 50, "width": 50, "roi_id": 0},
            {"gy0": 100, "gx0": 100, "height": 50, "width": 50, "roi_id": 0},
            {"gy0": 600, "gx0": 300, "height": 50, "width": 50, "roi_id": 0},
        ]

    def test_sequential_ids(self):
        rois = self._make_rois()
        result = number_rois_spatial(rois, pixel_size_um=1.0, row_tolerance_um=200)
        # Sorted IDs should be 1, 2, 3
        ids = sorted(r["roi_id"] for r in result)
        assert ids == [1, 2, 3]
        # The roi at gx0=100 (top-left) should get id=1
        id_by_gx0 = {r["gx0"]: r["roi_id"] for r in result}
        assert id_by_gx0[100] == 1  # top-left
        assert id_by_gx0[500] == 2  # top-right
        assert id_by_gx0[300] == 3  # bottom row

    def test_grid_mode(self):
        rois = self._make_rois()
        result = number_rois_spatial(rois, pixel_size_um=1.0, row_tolerance_um=200, grid_mode=True)
        labels = [r["grid_label"] for r in result]
        assert "A1" in labels
        assert "A2" in labels
        assert "B1" in labels

    def test_empty_input(self):
        assert number_rois_spatial([]) == []


# ---------------------------------------------------------------------------
# filter_tiles_by_rois
# ---------------------------------------------------------------------------


class TestFilterTilesByRois:

    def test_keeps_overlapping(self):
        rois = [{"gx0": 100, "gy0": 100, "width": 200, "height": 200}]
        tiles = [
            {"x": 0, "y": 0},  # overlaps (tile 0..256 vs roi 100..300)
            {"x": 500, "y": 500},  # no overlap
            {"x": 200, "y": 200},  # overlaps
        ]
        kept = filter_tiles_by_rois(tiles, rois, tile_size=256)
        assert len(kept) == 2
        assert {"x": 500, "y": 500} not in kept

    def test_empty_rois(self):
        tiles = [{"x": 0, "y": 0}]
        assert filter_tiles_by_rois(tiles, [], tile_size=256) == []

    def test_no_overlap(self):
        rois = [{"gx0": 1000, "gy0": 1000, "width": 50, "height": 50}]
        tiles = [{"x": 0, "y": 0}]
        assert filter_tiles_by_rois(tiles, rois, tile_size=100) == []


# ---------------------------------------------------------------------------
# filter_detections_by_roi_mask
# ---------------------------------------------------------------------------


class TestFilterDetectionsByRoiMask:

    def test_filters_correctly(self):
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:40, 20:40] = 1
        labels[60:80, 60:80] = 2
        detections = [
            {"global_center": [30, 30]},  # inside label 1
            {"global_center": [70, 70]},  # inside label 2
            {"global_center": [5, 5]},  # outside
        ]
        kept = filter_detections_by_roi_mask(detections, labels, downsample=1, x_start=0, y_start=0)
        assert len(kept) == 2
        assert kept[0]["roi_id"] == 1
        assert kept[1]["roi_id"] == 2

    def test_with_downsample(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1
        # Detection at global (25, 25), ds=2 → label[12, 12] → inside label 1
        detections = [{"global_center": [25, 25]}]
        kept = filter_detections_by_roi_mask(detections, labels, downsample=2, x_start=0, y_start=0)
        assert len(kept) == 1
        assert kept[0]["roi_id"] == 1

    def test_empty_detections(self):
        labels = np.ones((10, 10), dtype=np.int32)
        assert filter_detections_by_roi_mask([], labels, 1, 0, 0) == []

    def test_with_mosaic_origin(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1
        # Detection at global (110, 210), origin=(100, 200), ds=1
        # → label coords: (110-100, 210-200) = (10, 10) → inside
        detections = [{"global_center": [110, 210]}]
        kept = filter_detections_by_roi_mask(
            detections, labels, downsample=1, x_start=100, y_start=200
        )
        assert len(kept) == 1


# ---------------------------------------------------------------------------
# find_circular_regions
# ---------------------------------------------------------------------------


class TestFindCircularRegions:

    def test_finds_circular_blobs(self):
        """Synthetic image with 2 well-separated solid circular blobs."""
        shape = (400, 400)
        # Black background with two uniform bright circles
        img = np.zeros(shape, dtype=np.uint16)
        # Add sparse dim tissue so Otsu has a bimodal distribution
        rng = np.random.RandomState(42)
        tissue_mask = rng.random(shape) < 0.3  # 30% of pixels are dim tissue
        img[tissue_mask] = 100
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        for cy, cx in [(100, 100), (300, 300)]:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 50**2
            img[mask] = 800
        channel_data = {0: img}
        # pixel_size=0.5, ds=1 → ds_pixel_size=0.5, close_px=40 (big but blobs
        # are solid so closing only smooths edges).
        # Actually use ds=4 so the close kernel is more reasonable
        labels, ds = find_circular_regions(
            channel_data,
            channel_idx=0,
            pixel_size=0.5,
            downsample=4,
            min_diameter_um=20,
            max_diameter_um=200,
            min_circularity=0.4,
        )
        n = int(labels.max())
        assert n >= 1, f"Expected >=1 circular region, got {n}"

    def test_no_regions_when_too_small(self):
        """Blobs smaller than min_diameter_um should be rejected."""
        shape = (500, 500)
        img = np.zeros(shape, dtype=np.uint16)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        mask = (xx - 250) ** 2 + (yy - 250) ** 2 <= 10**2
        img[mask] = 800
        channel_data = {0: img}
        labels, ds = find_circular_regions(
            channel_data,
            channel_idx=0,
            pixel_size=1.0,
            downsample=2,
            min_diameter_um=200,
            max_diameter_um=1000,
            min_circularity=0.5,
        )
        assert int(labels.max()) == 0

    def test_missing_channel(self):
        channel_data = {0: np.zeros((100, 100), dtype=np.uint16)}
        labels, ds = find_circular_regions(channel_data, channel_idx=99, pixel_size=1.0)
        assert int(labels.max()) == 0


# ---------------------------------------------------------------------------
# load_rois_from_polygons
# ---------------------------------------------------------------------------


class TestLoadRoisFromPolygons:

    def test_flat_list_format(self, tmp_path):
        """Flat list of region dicts with vertices_px."""
        polygons = [
            {"vertices_px": [[10, 10], [10, 90], [90, 90], [90, 10]]},
            {"vertices_px": [[110, 110], [110, 190], [190, 190], [190, 110]]},
        ]
        json_path = tmp_path / "regions.json"
        json_path.write_text(json.dumps(polygons))
        labels, ds = load_rois_from_polygons(json_path, image_shape=(200, 200))
        assert ds == 1
        assert int(labels.max()) == 2
        # Check that the first polygon area is labelled 1
        assert labels[50, 50] == 1
        # Check that the second polygon area is labelled 2
        assert labels[150, 150] == 2
        # Background should be 0
        assert labels[0, 0] == 0

    def test_nested_format(self, tmp_path):
        """Nested format like annotate_bone_regions.py output."""
        data = {
            "slides": {
                "slide1": {
                    "femur": {"vertices_px": [[20, 20], [20, 80], [80, 80], [80, 20]]},
                }
            }
        }
        json_path = tmp_path / "nested.json"
        json_path.write_text(json.dumps(data))
        labels, ds = load_rois_from_polygons(json_path, image_shape=(100, 100))
        assert int(labels.max()) == 1
        assert labels[50, 50] == 1

    def test_empty_json(self, tmp_path):
        json_path = tmp_path / "empty.json"
        json_path.write_text("[]")
        labels, ds = load_rois_from_polygons(json_path, image_shape=(100, 100))
        assert int(labels.max()) == 0

    def test_scale_factor(self, tmp_path):
        """Vertices at half scale should be upscaled."""
        polygons = [{"vertices_px": [[5, 5], [5, 45], [45, 45], [45, 5]]}]
        json_path = tmp_path / "scaled.json"
        json_path.write_text(json.dumps(polygons))
        labels, ds = load_rois_from_polygons(json_path, image_shape=(100, 100), scale_factor=2.0)
        # Vertices [5,5]->[10,10] and [45,45]->[90,90]
        assert labels[50, 50] == 1
        assert labels[0, 0] == 0


# ---------------------------------------------------------------------------
# load_rois_from_mask
# ---------------------------------------------------------------------------


class TestLoadRoisFromMask:

    def test_npy_file(self, tmp_path):
        labels_in = np.zeros((100, 100), dtype=np.int32)
        labels_in[10:50, 10:50] = 1
        labels_in[60:90, 60:90] = 2
        npy_path = tmp_path / "mask.npy"
        np.save(npy_path, labels_in)
        labels_out, ds = load_rois_from_mask(npy_path)
        assert ds == 1
        np.testing.assert_array_equal(labels_out, labels_in)

    def test_image_file(self, tmp_path):
        from PIL import Image

        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[20:40, 20:40] = 1
        arr[60:80, 60:80] = 2
        img_path = tmp_path / "mask.png"
        Image.fromarray(arr).save(img_path)
        labels_out, ds = load_rois_from_mask(img_path)
        assert ds == 1
        assert int(labels_out.max()) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_single_region_bbox(self):
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1
        rois = extract_region_bboxes(labels, 1, 0, 0, 50, 50, padding_px=0)
        assert len(rois) == 1
        assert rois[0]["roi_id"] == 1
        assert rois[0]["height"] == 10
        assert rois[0]["width"] == 10

    def test_number_rois_single(self):
        rois = [{"gy0": 0, "gx0": 0, "height": 10, "width": 10, "roi_id": 0}]
        result = number_rois_spatial(rois)
        assert result[0]["roi_id"] == 1

    def test_filter_detections_all_outside(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        detections = [{"global_center": [5, 5]}]
        kept = filter_detections_by_roi_mask(detections, labels, 1, 0, 0)
        assert kept == []
