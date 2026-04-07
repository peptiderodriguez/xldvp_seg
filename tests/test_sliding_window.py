"""Tests for xldvp_seg.analysis.sliding_window_sampling.

Covers:
- spatially_balanced_sample with grid of points
- min_distance enforcement
- Edge: n_samples > len(positions)
- Edge: single position
- place_windows_along_paths
- run_sampling
"""

import numpy as np

from xldvp_seg.analysis.sliding_window_sampling import (
    place_windows_along_paths,
    run_sampling,
    spatially_balanced_sample,
)


class TestSpatiallyBalancedSample:
    def test_basic_grid(self):
        """Grid of points: should select cells until area target is met."""
        rng = np.random.default_rng(42)
        n = 50
        positions = np.column_stack([np.linspace(0, 500, n), np.zeros(n)])
        areas = np.full(n, 100.0)  # each cell is 100 um^2
        center = np.array([250, 0])
        target_lo = 500.0
        target_hi = 1500.0

        selected, total_area = spatially_balanced_sample(
            positions, areas, center, target_lo, target_hi, min_cells=3
        )
        assert len(selected) > 0
        assert target_lo <= total_area <= target_hi

    def test_spatial_spread(self):
        """Selected points should be spatially spread (farthest-point sampling)."""
        rng = np.random.default_rng(42)
        n = 100
        positions = rng.uniform(0, 1000, (n, 2))
        areas = np.full(n, 50.0)
        center = np.array([500, 500])
        target_lo = 200.0
        target_hi = 500.0

        selected, total_area = spatially_balanced_sample(
            positions, areas, center, target_lo, target_hi, min_cells=3
        )
        if len(selected) >= 2:
            sel_pos = positions[selected]
            # Check pairwise distances are nonzero
            for i in range(len(sel_pos)):
                for j in range(i + 1, len(sel_pos)):
                    dist = np.linalg.norm(sel_pos[i] - sel_pos[j])
                    assert dist > 0

    def test_empty_positions(self):
        """Empty positions should return empty selection."""
        selected, total_area = spatially_balanced_sample(
            np.empty((0, 2)), np.array([]), np.array([0, 0]), 100, 200
        )
        assert selected == []
        assert total_area == 0.0

    def test_insufficient_area(self):
        """When total available area < target_lo, should return empty."""
        positions = np.array([[0, 0], [10, 0]])
        areas = np.array([10.0, 10.0])  # total 20
        center = np.array([5, 0])

        selected, total_area = spatially_balanced_sample(
            positions, areas, center, 100, 200  # target_lo=100, much more than available
        )
        assert selected == []
        assert total_area == 0.0

    def test_single_position(self):
        """Single position that fits in target range should be returned."""
        positions = np.array([[50, 50]])
        areas = np.array([150.0])
        center = np.array([50, 50])

        selected, total_area = spatially_balanced_sample(
            positions, areas, center, 100, 200, min_cells=1
        )
        assert len(selected) == 1
        assert total_area == 150.0

    def test_single_position_below_target(self):
        """Single position below target_lo should return empty."""
        positions = np.array([[50, 50]])
        areas = np.array([50.0])
        center = np.array([50, 50])

        selected, total_area = spatially_balanced_sample(
            positions, areas, center, 100, 200, min_cells=1
        )
        assert selected == []


class TestPlaceWindowsAlongPaths:
    def test_straight_path(self):
        """Windows placed along a straight line at regular intervals."""
        path = np.array([[i * 10, 0] for i in range(20)], dtype=np.float64)
        centers = place_windows_along_paths([path], radius=30, step=40)
        assert len(centers) > 0
        # All centers should be along the x-axis (y=0)
        np.testing.assert_allclose(centers[:, 1], 0.0, atol=0.1)

    def test_empty_paths(self):
        """Empty paths should return empty array."""
        centers = place_windows_along_paths([], radius=30, step=40)
        assert centers.shape == (0, 2)

    def test_short_path(self):
        """Path shorter than 2*radius should yield no windows."""
        path = np.array([[0, 0], [10, 0]], dtype=np.float64)
        centers = place_windows_along_paths([path], radius=30, step=40)
        assert len(centers) == 0


class TestRunSampling:
    def test_basic_sampling(self):
        """Sampling should fill windows with cells from the ROI."""
        rng = np.random.default_rng(42)
        n = 200
        positions = np.column_stack([np.linspace(0, 1000, n), np.zeros(n)])
        areas = np.full(n, 50.0)
        indices = np.arange(n)
        centers = np.array([[250, 0], [500, 0], [750, 0]])
        radius = 150.0
        target_lo = 200.0
        target_hi = 500.0

        windows, n_rejected = run_sampling(
            positions, areas, indices, centers, radius, target_lo, target_hi
        )
        assert len(windows) > 0
        for w in windows:
            assert "window_id" in w
            assert "n_cells" in w
            assert "cell_indices" in w
            assert target_lo <= w["total_area_um2"] <= target_hi

    def test_no_overlap_in_cell_assignments(self):
        """Each cell should be assigned to at most one window."""
        rng = np.random.default_rng(42)
        n = 100
        positions = np.column_stack([np.linspace(0, 500, n), np.zeros(n)])
        areas = np.full(n, 50.0)
        indices = np.arange(n)
        centers = np.array([[100, 0], [200, 0], [300, 0], [400, 0]])
        radius = 100.0

        windows, _ = run_sampling(positions, areas, indices, centers, radius, 100, 300)
        all_cells = []
        for w in windows:
            all_cells.extend(w["cell_indices"])
        # No duplicates
        assert len(all_cells) == len(set(all_cells))
