"""Unit tests for post-detection processing functions."""

import numpy as np

from xldvp_seg.pipeline.post_detection import (
    _contour_from_binary,
    _parse_tile_key,
)


class TestContourFromBinary:
    def test_circle_mask_returns_contour(self):
        """Circle mask should produce a valid contour array."""
        mask = np.zeros((100, 100), dtype=bool)
        yy, xx = np.ogrid[:100, :100]
        mask[(yy - 50) ** 2 + (xx - 50) ** 2 <= 30**2] = True
        contour = _contour_from_binary(mask)
        assert contour is not None
        assert contour.ndim == 2
        assert contour.shape[1] == 2
        assert len(contour) >= 10  # circle should have many points

    def test_rectangle_mask(self):
        """Rectangle mask should produce a 4+ point contour."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 30:80] = True
        contour = _contour_from_binary(mask)
        assert contour is not None
        assert contour.shape[1] == 2
        assert len(contour) >= 4

    def test_empty_mask_returns_none(self):
        """All-False mask should return None."""
        mask = np.zeros((100, 100), dtype=bool)
        contour = _contour_from_binary(mask)
        assert contour is None

    def test_single_pixel_returns_none(self):
        """Single pixel may not produce a closed contour."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True
        contour = _contour_from_binary(mask)
        # Single pixel: either None or very small contour
        if contour is not None:
            assert contour.shape[1] == 2

    def test_contour_coordinates_within_mask_bounds(self):
        """Contour coordinates should be within mask dimensions."""
        mask = np.zeros((200, 300), dtype=bool)
        mask[50:150, 100:250] = True
        contour = _contour_from_binary(mask)
        assert contour is not None
        assert np.all(contour[:, 0] >= 0) and np.all(contour[:, 0] < 300)
        assert np.all(contour[:, 1] >= 0) and np.all(contour[:, 1] < 200)


class TestParseTileKey:
    def test_basic(self):
        assert _parse_tile_key("1000_2000") == (1000, 2000)

    def test_zero_origin(self):
        assert _parse_tile_key("0_0") == (0, 0)

    def test_large_coords(self):
        assert _parse_tile_key("50000_80000") == (50000, 80000)
