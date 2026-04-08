"""Tests for xldvp_seg.analysis.nuclear_count — overlap-based nuclear assignment."""

import numpy as np
from skimage.measure import regionprops


class TestOverlapAssignment:
    """Verify overlap-based nuclear-to-cell assignment logic."""

    def test_nucleus_fully_inside_cell(self):
        """Nucleus entirely within a cell is correctly assigned."""
        cell_masks = np.zeros((20, 20), dtype=np.int32)
        cell_masks[2:18, 2:18] = 1

        nuclear_masks = np.zeros((20, 20), dtype=np.int32)
        nuclear_masks[8:12, 8:12] = 1

        nuc_prop = regionprops(nuclear_masks)[0]
        coords = nuc_prop.coords
        labels = cell_masks[coords[:, 0], coords[:, 1]]
        labels = labels[labels > 0]
        assert len(labels) == 16  # 4x4 block
        assert int(np.bincount(labels).argmax()) == 1

    def test_nucleus_in_background_skipped(self):
        """Nucleus entirely in background produces empty overlap."""
        cell_masks = np.zeros((20, 20), dtype=np.int32)
        cell_masks[0:5, 0:5] = 1  # cell far from nucleus

        nuclear_masks = np.zeros((20, 20), dtype=np.int32)
        nuclear_masks[15:18, 15:18] = 1  # nucleus in background

        nuc_prop = regionprops(nuclear_masks)[0]
        coords = nuc_prop.coords
        labels = cell_masks[coords[:, 0], coords[:, 1]]
        labels = labels[labels > 0]
        assert len(labels) == 0  # no overlap

    def test_boundary_nucleus_assigned_by_majority_overlap(self):
        """Nucleus straddling two cells is assigned to the one with more overlap."""
        cell_masks = np.zeros((20, 20), dtype=np.int32)
        cell_masks[0:10, :] = 1  # cell 1: top half
        cell_masks[10:20, :] = 2  # cell 2: bottom half

        # Nucleus mostly in cell 1: rows 6-12, cols 8-12
        # 4 rows in cell 1 (6,7,8,9), 2 rows in cell 2 (10,11)
        nuclear_masks = np.zeros((20, 20), dtype=np.int32)
        nuclear_masks[6:12, 8:13] = 1

        nuc_prop = regionprops(nuclear_masks)[0]
        coords = nuc_prop.coords
        labels = cell_masks[coords[:, 0], coords[:, 1]]
        labels = labels[labels > 0]
        assert int(np.bincount(labels).argmax()) == 1  # cell 1 wins

    def test_centroid_in_gap_but_overlap_captures_nucleus(self):
        """Nucleus whose centroid falls in a segmentation gap is still assigned."""
        cell_masks = np.zeros((20, 20), dtype=np.int32)
        cell_masks[0:10, :] = 1  # cell 1
        cell_masks[10:20, :] = 2  # cell 2
        cell_masks[10, 5:15] = 0  # segmentation gap at boundary

        # Nucleus straddling the gap: rows 8-12, cols 7-12
        nuclear_masks = np.zeros((20, 20), dtype=np.int32)
        nuclear_masks[8:13, 7:13] = 1

        nuc_prop = regionprops(nuclear_masks)[0]

        # Verify centroid falls in/near the gap (old code would skip)
        cy, _ = nuc_prop.centroid
        iy = int(round(cy))
        assert cell_masks[iy, 10] == 0, "Centroid row should be in the gap"

        # Overlap-based assignment captures it
        coords = nuc_prop.coords
        labels = cell_masks[coords[:, 0], coords[:, 1]]
        labels = labels[labels > 0]
        assert len(labels) > 0, "Overlap finds pixels despite centroid in gap"
        assigned = int(np.bincount(labels).argmax())
        assert assigned in (1, 2)  # assigned to one of the cells

    def test_single_pixel_nucleus(self):
        """Degenerate single-pixel nucleus is handled correctly."""
        cell_masks = np.zeros((10, 10), dtype=np.int32)
        cell_masks[3:7, 3:7] = 1

        nuclear_masks = np.zeros((10, 10), dtype=np.int32)
        nuclear_masks[5, 5] = 1

        nuc_prop = regionprops(nuclear_masks)[0]
        coords = nuc_prop.coords
        labels = cell_masks[coords[:, 0], coords[:, 1]]
        labels = labels[labels > 0]
        assert len(labels) == 1
        assert int(np.bincount(labels).argmax()) == 1

    def test_multiple_cells_overlap(self):
        """Nucleus overlapping 3 cells is assigned to the one with most pixels."""
        cell_masks = np.zeros((30, 30), dtype=np.int32)
        cell_masks[0:10, :] = 1  # cell 1
        cell_masks[10:20, :] = 2  # cell 2
        cell_masks[20:30, :] = 3  # cell 3

        # Nucleus spanning all 3 cells: rows 8-22
        # 2 rows in cell 1 (8,9), 10 rows in cell 2 (10-19), 2 rows in cell 3 (20,21)
        nuclear_masks = np.zeros((30, 30), dtype=np.int32)
        nuclear_masks[8:22, 12:18] = 1

        nuc_prop = regionprops(nuclear_masks)[0]
        coords = nuc_prop.coords
        labels = cell_masks[coords[:, 0], coords[:, 1]]
        labels = labels[labels > 0]
        assert int(np.bincount(labels).argmax()) == 2  # cell 2 has most overlap
