"""Tests for xldvp_seg.analysis.background.local_background_subtract."""

import numpy as np

from xldvp_seg.analysis.background import local_background_subtract


class TestLocalBackgroundSubtract:
    def test_basic_correction(self):
        """Values should decrease after background subtraction."""
        np.random.seed(42)
        n = 100
        centroids = np.random.rand(n, 2) * 1000
        values = np.random.rand(n) * 100 + 50  # all positive
        corrected, bg, tree_info = local_background_subtract(values, centroids)
        assert len(corrected) == n
        assert len(bg) == n
        assert np.all(corrected >= 0)

    def test_kdtree_caching(self):
        """tree_and_indices from first call should be reusable."""
        np.random.seed(42)
        n = 100
        centroids = np.random.rand(n, 2) * 1000
        values1 = np.random.rand(n) * 100
        values2 = np.random.rand(n) * 200
        _, _, tree_info = local_background_subtract(values1, centroids)
        assert tree_info is not None
        # Reuse tree
        corrected2, _, tree_info2 = local_background_subtract(
            values2, centroids, tree_and_indices=tree_info
        )
        assert len(corrected2) == n

    def test_few_cells_fallback(self):
        """With fewer cells than n_neighbors, should use global median."""
        centroids = np.array([[0, 0], [100, 100], [200, 200]])
        values = np.array([10.0, 20.0, 30.0])
        corrected, bg, _ = local_background_subtract(values, centroids, n_neighbors=30)
        assert len(corrected) == 3
        assert np.all(corrected >= 0)

    def test_corrected_non_negative(self):
        """Even with high background, corrected values should be >= 0."""
        np.random.seed(42)
        n = 50
        centroids = np.random.rand(n, 2) * 1000
        values = np.ones(n) * 5  # very low values
        corrected, _, _ = local_background_subtract(values, centroids)
        assert np.all(corrected >= 0)

    def test_uniform_values(self):
        """Uniform field: correction should be small or zero."""
        np.random.seed(42)
        n = 100
        centroids = np.random.rand(n, 2) * 1000
        values = np.ones(n) * 100.0
        corrected, bg, _ = local_background_subtract(values, centroids)
        # Background should be close to 100, corrected close to 0
        assert np.mean(corrected) < 10  # mostly corrected away

    def test_return_types(self):
        """Check that return types are correct numpy arrays and tuple."""
        np.random.seed(42)
        n = 50
        centroids = np.random.rand(n, 2) * 1000
        values = np.random.rand(n) * 100
        corrected, bg, tree_info = local_background_subtract(values, centroids)
        assert isinstance(corrected, np.ndarray)
        assert isinstance(bg, np.ndarray)
        assert isinstance(tree_info, tuple)
        assert len(tree_info) == 2  # (tree, indices)

    def test_cached_tree_gives_same_bg(self):
        """Using cached tree should give identical results to a fresh build."""
        np.random.seed(42)
        n = 100
        centroids = np.random.rand(n, 2) * 1000
        values = np.random.rand(n) * 100
        corrected1, bg1, tree_info = local_background_subtract(values, centroids)
        corrected2, bg2, _ = local_background_subtract(
            values, centroids, tree_and_indices=tree_info
        )
        np.testing.assert_array_equal(corrected1, corrected2)
        np.testing.assert_array_equal(bg1, bg2)

    def test_few_cells_bg_is_global_median(self):
        """Fallback should set background to global median for all cells."""
        values = np.array([10.0, 20.0, 30.0])
        centroids = np.array([[0, 0], [100, 100], [200, 200]])
        _, bg, _ = local_background_subtract(values, centroids, n_neighbors=30)
        expected_bg = np.median(values)
        np.testing.assert_array_equal(bg, np.full(3, expected_bg))

    def test_high_value_outlier(self):
        """A single bright outlier should still produce non-negative correction."""
        np.random.seed(42)
        n = 100
        centroids = np.random.rand(n, 2) * 1000
        values = np.ones(n) * 50.0
        values[0] = 10000.0  # bright outlier
        corrected, bg, _ = local_background_subtract(values, centroids)
        assert np.all(corrected >= 0)
        # Outlier should retain most of its value (bg from neighbors ~ 50)
        assert corrected[0] > 9000

    def test_nan_values_no_crash(self):
        """NaN values in input should not crash (numpy propagates NaN gracefully)."""
        np.random.seed(42)
        n = 50
        centroids = np.random.rand(n, 2) * 1000
        values = np.random.rand(n) * 100
        values[5] = np.nan
        # Should not raise — NaN propagation is acceptable
        corrected, bg, _ = local_background_subtract(values, centroids)
        assert len(corrected) == n

    def test_negative_values_clipped(self):
        """Negative input values should result in corrected >= 0."""
        np.random.seed(42)
        n = 50
        centroids = np.random.rand(n, 2) * 1000
        values = np.random.rand(n) * 100 - 50  # some negative
        corrected, bg, _ = local_background_subtract(values, centroids)
        assert np.all(corrected >= 0)

    def test_background_positive_when_nonzero_values(self):
        """When values are positive, at least some background should be > 0."""
        np.random.seed(42)
        n = 100
        centroids = np.random.rand(n, 2) * 1000
        values = np.random.rand(n) * 100 + 10  # all > 10
        corrected, bg, _ = local_background_subtract(values, centroids)
        assert np.any(bg > 0)
        assert np.any(corrected < values)
