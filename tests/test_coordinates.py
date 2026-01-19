"""
Tests for coordinate handling utilities.

Tests the coordinate conversion functions in segmentation/processing/coordinates.py
to ensure correct handling of [x, y] vs [row, col] conventions.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.processing.coordinates import (
    regionprop_centroid_to_xy,
    xy_to_array_index,
    array_index_to_xy,
    extract_crop_bounds,
    generate_uid,
)


class TestRegionpropCentroidToXY:
    """Tests for regionprop_centroid_to_xy function."""

    def test_regionprop_centroid_to_xy(self, mock_regionprop):
        """Test conversion from regionprops centroid (row, col) to [x, y] format."""
        # mock_regionprop.centroid = (100.5, 200.5)  # (y, x) in regionprops
        result = regionprop_centroid_to_xy(mock_regionprop)

        # Should return [x, y] = [200.5, 100.5]
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == 200.5  # x = col
        assert result[1] == 100.5  # y = row

    def test_regionprop_centroid_to_xy_integer_coords(self):
        """Test with integer coordinates."""
        prop = MagicMock()
        prop.centroid = (50, 100)  # (y, x)

        result = regionprop_centroid_to_xy(prop)

        assert result == [100.0, 50.0]  # [x, y]

    def test_regionprop_centroid_to_xy_returns_floats(self):
        """Test that result always contains floats."""
        prop = MagicMock()
        prop.centroid = (50, 100)

        result = regionprop_centroid_to_xy(prop)

        assert all(isinstance(v, float) for v in result)


class TestXYToArrayIndex:
    """Tests for xy_to_array_index function."""

    def test_xy_to_array_index_basic(self):
        """Test basic conversion from x, y to row, col."""
        x, y = 100.5, 50.5
        row, col = xy_to_array_index(x, y)

        # row = int(y), col = int(x)
        assert row == 50
        assert col == 100

    def test_xy_to_array_index_roundtrip(self):
        """Test that xy_to_array_index and array_index_to_xy are inverses."""
        original_x, original_y = 150, 75

        # Convert to array index
        row, col = xy_to_array_index(original_x, original_y)

        # Convert back to xy
        recovered_x, recovered_y = array_index_to_xy(row, col)

        # Should get back original values
        assert recovered_x == original_x
        assert recovered_y == original_y

    def test_xy_to_array_index_returns_integers(self):
        """Test that result always contains integers."""
        row, col = xy_to_array_index(100.7, 50.3)

        assert isinstance(row, int)
        assert isinstance(col, int)


class TestExtractCropBounds:
    """Tests for extract_crop_bounds function."""

    def test_extract_crop_bounds_within_image(self):
        """Test crop bounds when fully within image boundaries."""
        center_x, center_y = 256, 256
        crop_size = 100
        image_width, image_height = 512, 512

        x1, y1, x2, y2 = extract_crop_bounds(
            center_x, center_y, crop_size, image_width, image_height
        )

        # Crop should be centered on (256, 256)
        assert x1 == 206  # 256 - 50
        assert y1 == 206  # 256 - 50
        assert x2 == 306  # 256 + 50
        assert y2 == 306  # 256 + 50

        # Verify dimensions
        assert x2 - x1 == crop_size
        assert y2 - y1 == crop_size

    def test_extract_crop_bounds_at_left_edge(self):
        """Test crop bounds when center is near left edge."""
        center_x, center_y = 30, 256
        crop_size = 100
        image_width, image_height = 512, 512

        x1, y1, x2, y2 = extract_crop_bounds(
            center_x, center_y, crop_size, image_width, image_height
        )

        # x1 should be clipped to 0
        assert x1 == 0
        assert x2 == 80  # 30 + 50
        assert y1 == 206
        assert y2 == 306

    def test_extract_crop_bounds_at_top_edge(self):
        """Test crop bounds when center is near top edge."""
        center_x, center_y = 256, 30
        crop_size = 100
        image_width, image_height = 512, 512

        x1, y1, x2, y2 = extract_crop_bounds(
            center_x, center_y, crop_size, image_width, image_height
        )

        # y1 should be clipped to 0
        assert x1 == 206
        assert x2 == 306
        assert y1 == 0
        assert y2 == 80  # 30 + 50

    def test_extract_crop_bounds_at_right_edge(self):
        """Test crop bounds when center is near right edge."""
        center_x, center_y = 490, 256
        crop_size = 100
        image_width, image_height = 512, 512

        x1, y1, x2, y2 = extract_crop_bounds(
            center_x, center_y, crop_size, image_width, image_height
        )

        # x2 should be clipped to image_width
        assert x1 == 440  # 490 - 50
        assert x2 == 512  # clipped

    def test_extract_crop_bounds_at_bottom_edge(self):
        """Test crop bounds when center is near bottom edge."""
        center_x, center_y = 256, 500
        crop_size = 100
        image_width, image_height = 512, 512

        x1, y1, x2, y2 = extract_crop_bounds(
            center_x, center_y, crop_size, image_width, image_height
        )

        # y2 should be clipped to image_height
        assert y1 == 450  # 500 - 50
        assert y2 == 512  # clipped

    def test_extract_crop_bounds_at_corner(self):
        """Test crop bounds when center is at corner."""
        center_x, center_y = 10, 10
        crop_size = 100
        image_width, image_height = 512, 512

        x1, y1, x2, y2 = extract_crop_bounds(
            center_x, center_y, crop_size, image_width, image_height
        )

        # Both x1 and y1 should be clipped to 0
        assert x1 == 0
        assert y1 == 0
        assert x2 == 60  # 10 + 50
        assert y2 == 60  # 10 + 50

    def test_extract_crop_bounds_at_edges(self):
        """Test crop bounds at multiple edge conditions."""
        # Test that bounds are always within [0, image_size]
        for center_x in [0, 256, 511]:
            for center_y in [0, 256, 511]:
                x1, y1, x2, y2 = extract_crop_bounds(
                    center_x, center_y, 100, 512, 512
                )

                assert x1 >= 0
                assert y1 >= 0
                assert x2 <= 512
                assert y2 <= 512
                assert x1 < x2
                assert y1 < y2


class TestGenerateUID:
    """Tests for generate_uid function."""

    def test_generate_uid_format(self):
        """Test that UID has correct format."""
        slide_name = "2025_11_18_FGC1"
        cell_type = "mk"
        global_x = 12345.6
        global_y = 67890.3

        uid = generate_uid(slide_name, cell_type, global_x, global_y)

        # Expected format: {slide_name}_{cell_type}_{round(x)}_{round(y)}
        expected = "2025_11_18_FGC1_mk_12346_67890"
        assert uid == expected

    def test_generate_uid_rounding(self):
        """Test that coordinates are properly rounded."""
        uid = generate_uid("slide", "hspc", 100.4, 200.6)
        assert uid == "slide_hspc_100_201"

        uid = generate_uid("slide", "hspc", 100.5, 200.5)
        assert uid == "slide_hspc_100_200"  # Python rounds 0.5 to even

    def test_generate_uid_different_cell_types(self):
        """Test UID generation for different cell types."""
        cell_types = ["mk", "hspc", "nmj", "vessel"]

        for cell_type in cell_types:
            uid = generate_uid("slide", cell_type, 100, 200)
            assert f"_{cell_type}_" in uid

    def test_generate_uid_uniqueness(self):
        """Test that different coordinates produce different UIDs."""
        uid1 = generate_uid("slide", "mk", 100, 200)
        uid2 = generate_uid("slide", "mk", 100, 201)
        uid3 = generate_uid("slide", "mk", 101, 200)

        assert uid1 != uid2
        assert uid1 != uid3
        assert uid2 != uid3
