"""Tests for segmentation.processing.deduplication.deduplicate_by_iou_nms.

Creates temporary HDF5 mask files with known rectangular masks, then runs
IoU NMS deduplication to verify correct duplicate removal behavior.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

try:
    import hdf5plugin
except ImportError:
    pass
import h5py

from segmentation.processing.deduplication import deduplicate_by_iou_nms


def _create_tile_masks(tiles_dir, tile_origin, masks_array, mask_filename="test_masks.h5"):
    """Helper: write a masks array to an HDF5 file under tiles_dir/tile_X_Y/."""
    tile_x, tile_y = tile_origin
    tile_dir = Path(tiles_dir) / f"tile_{tile_x}_{tile_y}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(tile_dir / mask_filename, "w") as f:
        f.create_dataset("masks", data=masks_array, dtype=np.int32)


def _make_rect_mask(shape, row_start, row_end, col_start, col_end, label):
    """Helper: create a labeled mask array with a single rectangle."""
    mask = np.zeros(shape, dtype=np.int32)
    mask[row_start:row_end, col_start:col_end] = label
    return mask


def _make_detection(tile_origin, mask_label, area=None, det_id=None):
    """Helper: build a minimal detection dict."""
    det = {
        "tile_origin": list(tile_origin),
        "mask_label": mask_label,
        "features": {"area": area or 100},
    }
    if det_id is not None:
        det["id"] = det_id
    return det


@pytest.fixture
def tiles_dir():
    """Temporary directory for tile HDF5 files."""
    d = tempfile.mkdtemp(prefix="dedup_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestDeduplicateByIoUNMS:
    MASK_FILE = "test_masks.h5"

    def test_empty_detections(self, tiles_dir):
        """Empty input should return empty list."""
        result = deduplicate_by_iou_nms(
            [], tiles_dir, iou_threshold=0.2, mask_filename=self.MASK_FILE
        )
        assert result == []

    def test_identical_rectangles_one_removed(self, tiles_dir):
        """Two detections with identical masks on the same tile: one should be removed."""
        # Create a 200x200 tile with two labels occupying the SAME region
        mask = np.zeros((200, 200), dtype=np.int32)
        mask[50:150, 50:150] = 1  # label 1
        # For label 2, overlay the same region (both labels share pixels, but
        # HDF5 stores a single label per pixel). Instead, put label 2 on a
        # second tile at the same origin so both detections map to overlapping
        # global coordinates.
        mask1 = _make_rect_mask((200, 200), 50, 150, 50, 150, label=1)
        mask2 = _make_rect_mask((200, 200), 50, 150, 50, 150, label=1)

        _create_tile_masks(tiles_dir, (0, 0), mask1, self.MASK_FILE)
        # Second tile at same origin but different tile dir requires same coords.
        # Easier: use two labels on one tile via combined mask.
        combined = np.zeros((200, 200), dtype=np.int32)
        combined[50:150, 50:150] = 1
        # Overwrite to have two separate labels:
        combined2 = np.zeros((200, 200), dtype=np.int32)
        combined2[50:150, 50:150] = 2
        # Merge: label 1 in top half, label 2 in bottom half -- but that won't
        # overlap. Instead, use two tiles at overlapping origins.

        # Strategy: tile A at origin (0, 0) with label=1 at rows 50:150, cols 50:150
        # tile B at origin (0, 0) -- same tile, just two labels stacked.
        # We can't have two labels in the same pixel, so use two tiles at
        # slightly offset origins with masks in the overlap zone.

        # Tile A at origin (0, 0): label 1 at local [50:150, 50:150] -> global [50:150, 50:150]
        # Tile B at origin (10, 10): label 1 at local [40:140, 40:140] -> global [50:150, 50:150]
        mask_a = _make_rect_mask((200, 200), 50, 150, 50, 150, label=1)
        mask_b = _make_rect_mask((200, 200), 40, 140, 40, 140, label=1)

        _create_tile_masks(tiles_dir, (0, 0), mask_a, self.MASK_FILE)
        _create_tile_masks(tiles_dir, (10, 10), mask_b, self.MASK_FILE)

        det_a = _make_detection((0, 0), mask_label=1, area=10000, det_id="slide_cell_0_0_1")
        det_b = _make_detection((10, 10), mask_label=1, area=10000, det_id="slide_cell_10_10_1")

        result = deduplicate_by_iou_nms(
            [det_a, det_b], tiles_dir, iou_threshold=0.2, mask_filename=self.MASK_FILE
        )
        assert len(result) == 1

    def test_non_overlapping_both_kept(self, tiles_dir):
        """Two non-overlapping rectangles should both be kept."""
        # Tile at origin (0, 0) with two well-separated labels
        mask = np.zeros((500, 500), dtype=np.int32)
        mask[10:60, 10:60] = 1     # top-left
        mask[300:350, 300:350] = 2  # bottom-right (far away)

        _create_tile_masks(tiles_dir, (0, 0), mask, self.MASK_FILE)

        det_a = _make_detection((0, 0), mask_label=1, area=2500, det_id="s_c_0_0_1")
        det_b = _make_detection((0, 0), mask_label=2, area=2500, det_id="s_c_0_0_2")

        result = deduplicate_by_iou_nms(
            [det_a, det_b], tiles_dir, iou_threshold=0.2, mask_filename=self.MASK_FILE
        )
        assert len(result) == 2

    def test_partial_overlap_above_threshold(self, tiles_dir):
        """Two rectangles with high overlap (IoU > threshold) should deduplicate to one."""
        # Tile A at origin (0, 0): label 1 covers [0:100, 0:100]
        # Tile B at origin (10, 10): label 1 covers [0:100, 0:100]
        # Global A = [0:100, 0:100], Global B = [10:110, 10:110]
        # Intersection = [10:100, 10:100] = 90x90 = 8100
        # Union = 10000 + 10000 - 8100 = 11900
        # IoU = 8100/11900 ~ 0.68 >> 0.2
        mask_a = _make_rect_mask((200, 200), 0, 100, 0, 100, label=1)
        mask_b = _make_rect_mask((200, 200), 0, 100, 0, 100, label=1)

        _create_tile_masks(tiles_dir, (0, 0), mask_a, self.MASK_FILE)
        _create_tile_masks(tiles_dir, (10, 10), mask_b, self.MASK_FILE)

        det_a = _make_detection((0, 0), mask_label=1, area=10000, det_id="s_c_0_0_1")
        det_b = _make_detection((10, 10), mask_label=1, area=10000, det_id="s_c_10_10_1")

        result = deduplicate_by_iou_nms(
            [det_a, det_b], tiles_dir, iou_threshold=0.2, mask_filename=self.MASK_FILE
        )
        assert len(result) == 1

    def test_partial_overlap_below_threshold(self, tiles_dir):
        """Two rectangles with small overlap (IoU < threshold) should both be kept."""
        # Tile A at origin (0, 0): label 1 at [0:100, 0:100]
        # Tile B at origin (90, 90): label 1 at [0:100, 0:100]
        # Global A = [0:100, 0:100], Global B = [90:190, 90:190]
        # Intersection = [90:100, 90:100] = 10x10 = 100
        # Union = 10000 + 10000 - 100 = 19900
        # IoU = 100/19900 ~ 0.005 << 0.2
        mask_a = _make_rect_mask((200, 200), 0, 100, 0, 100, label=1)
        mask_b = _make_rect_mask((200, 200), 0, 100, 0, 100, label=1)

        _create_tile_masks(tiles_dir, (0, 0), mask_a, self.MASK_FILE)
        _create_tile_masks(tiles_dir, (90, 90), mask_b, self.MASK_FILE)

        det_a = _make_detection((0, 0), mask_label=1, area=10000, det_id="s_c_0_0_1")
        det_b = _make_detection((90, 90), mask_label=1, area=10000, det_id="s_c_90_90_1")

        result = deduplicate_by_iou_nms(
            [det_a, det_b], tiles_dir, iou_threshold=0.2, mask_filename=self.MASK_FILE
        )
        assert len(result) == 2

    def test_requires_mask_filename(self, tiles_dir):
        """Calling without mask_filename should raise ValueError."""
        det = _make_detection((0, 0), mask_label=1)
        with pytest.raises(ValueError, match="mask_filename must be provided"):
            deduplicate_by_iou_nms([det], tiles_dir, iou_threshold=0.2)

    def test_missing_mask_file_keeps_detection(self, tiles_dir):
        """Detection whose tile has no HDF5 file should be kept (maskless fallback)."""
        det = _make_detection((999, 999), mask_label=1, det_id="s_c_999_999_1")
        result = deduplicate_by_iou_nms(
            [det], tiles_dir, iou_threshold=0.2, mask_filename=self.MASK_FILE
        )
        assert len(result) == 1

    def test_larger_area_wins_by_default(self, tiles_dir):
        """With sort_by='area', the detection with larger area should be kept."""
        # Two overlapping detections: one large, one small
        # Large: label 1 at [0:100, 0:100] on tile (0,0)
        # Small: label 1 at [0:50, 0:50] on tile (0,0)  -> global [0:50, 0:50]
        # But same tile can't have overlapping labels. Use two tiles:
        # Large on tile (0,0): [0:100, 0:100]
        # Small on tile (5,5): [0:50, 0:50] -> global [5:55, 5:55]
        # Overlap region: [5:55, 5:55] = 50x50 = 2500
        # Small polygon area ~ 50*50 = 2500, Large ~ 100*100 = 10000
        # IoU = 2500 / (10000 + 2500 - 2500) = 2500/10000 = 0.25 > 0.2
        mask_large = _make_rect_mask((200, 200), 0, 100, 0, 100, label=1)
        mask_small = _make_rect_mask((200, 200), 0, 50, 0, 50, label=1)

        _create_tile_masks(tiles_dir, (0, 0), mask_large, self.MASK_FILE)
        _create_tile_masks(tiles_dir, (5, 5), mask_small, self.MASK_FILE)

        det_large = _make_detection((0, 0), mask_label=1, area=10000, det_id="s_c_0_0_1")
        det_small = _make_detection((5, 5), mask_label=1, area=2500, det_id="s_c_5_5_1")

        result = deduplicate_by_iou_nms(
            [det_small, det_large],
            tiles_dir,
            iou_threshold=0.2,
            mask_filename=self.MASK_FILE,
            sort_by="area",
        )
        assert len(result) == 1
        # The larger detection should be the one kept
        assert result[0]["features"]["area"] == 10000
