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
    # New UID parsing and migration functions
    parse_uid,
    migrate_uid_format,
    is_spatial_uid,
    # Coordinate validation functions
    CoordinateValidationError,
    validate_xy_coordinates,
    validate_array_indices,
    validate_bbox_xyxy,
    # Coordinate labeling helpers
    create_coordinate_dict,
    format_coordinates_for_export,
    convert_detections_to_spatial_uids,
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


class TestParseUID:
    """Tests for parse_uid function."""

    def test_parse_spatial_uid(self):
        """Test parsing a spatial format UID."""
        uid = "slide_01_mk_12345_67890"
        parsed = parse_uid(uid)

        assert parsed['slide_name'] == 'slide_01'
        assert parsed['cell_type'] == 'mk'
        assert parsed['global_x'] == 12345
        assert parsed['global_y'] == 67890
        assert parsed['global_id'] is None
        assert parsed['is_spatial'] is True

    def test_parse_legacy_uid(self):
        """Test parsing a legacy format UID."""
        uid = "slide_01_mk_123"
        parsed = parse_uid(uid)

        assert parsed['slide_name'] == 'slide_01'
        assert parsed['cell_type'] == 'mk'
        assert parsed['global_x'] is None
        assert parsed['global_y'] is None
        assert parsed['global_id'] == 123
        assert parsed['is_spatial'] is False

    def test_parse_uid_with_underscores_in_slide(self):
        """Test parsing UID where slide name contains underscores."""
        uid = "2025_11_18_FGC1_hspc_5000_3000"
        parsed = parse_uid(uid)

        assert parsed['slide_name'] == '2025_11_18_FGC1'
        assert parsed['cell_type'] == 'hspc'
        assert parsed['global_x'] == 5000
        assert parsed['global_y'] == 3000

    def test_parse_uid_all_cell_types(self):
        """Test parsing UIDs for different cell types."""
        for cell_type in ['mk', 'hspc', 'nmj', 'vessel']:
            uid = f"slide_{cell_type}_100_200"
            parsed = parse_uid(uid)
            assert parsed['cell_type'] == cell_type

    def test_parse_uid_invalid_raises(self):
        """Test that invalid UID raises ValueError."""
        with pytest.raises(ValueError):
            parse_uid("invalid_uid_format")


class TestMigrateUIDFormat:
    """Tests for migrate_uid_format function."""

    def test_migrate_legacy_to_spatial(self):
        """Test migrating a legacy UID to spatial format."""
        old_uid = "slide_01_mk_123"
        new_uid = migrate_uid_format(old_uid, 12345.6, 67890.3)

        assert new_uid == "slide_01_mk_12346_67890"

    def test_migrate_preserves_slide_name(self):
        """Test that migration preserves complex slide names."""
        old_uid = "2025_11_18_FGC1_hspc_456"
        new_uid = migrate_uid_format(old_uid, 5000, 3000)

        assert new_uid == "2025_11_18_FGC1_hspc_5000_3000"


class TestIsSpatialUID:
    """Tests for is_spatial_uid function."""

    def test_spatial_uid_returns_true(self):
        """Test that spatial UIDs return True."""
        assert is_spatial_uid("slide_mk_12345_67890") is True
        assert is_spatial_uid("slide_hspc_100_200") is True

    def test_legacy_uid_returns_false(self):
        """Test that legacy UIDs return False."""
        assert is_spatial_uid("slide_mk_123") is False

    def test_invalid_uid_returns_false(self):
        """Test that invalid UIDs return False."""
        assert is_spatial_uid("invalid_format") is False


class TestValidateXYCoordinates:
    """Tests for validate_xy_coordinates function."""

    def test_valid_coordinates_pass(self):
        """Test that valid coordinates don't raise."""
        validate_xy_coordinates(100, 200, 512, 512)  # Should not raise

    def test_negative_x_raises(self):
        """Test that negative x raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_xy_coordinates(-1, 200, 512, 512)

    def test_negative_y_raises(self):
        """Test that negative y raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_xy_coordinates(100, -1, 512, 512)

    def test_x_exceeds_width_raises(self):
        """Test that x >= width raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_xy_coordinates(512, 200, 512, 512)

    def test_y_exceeds_height_raises(self):
        """Test that y >= height raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_xy_coordinates(100, 512, 512, 512)

    def test_allow_negative_flag(self):
        """Test that allow_negative flag works."""
        validate_xy_coordinates(-10, -20, 512, 512, allow_negative=True)


class TestValidateArrayIndices:
    """Tests for validate_array_indices function."""

    def test_valid_indices_pass(self):
        """Test that valid indices don't raise."""
        validate_array_indices(100, 200, 512, 512)  # Should not raise

    def test_negative_row_raises(self):
        """Test that negative row raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_array_indices(-1, 200, 512, 512)

    def test_row_exceeds_height_raises(self):
        """Test that row >= height raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_array_indices(512, 200, 512, 512)


class TestValidateBboxXYXY:
    """Tests for validate_bbox_xyxy function."""

    def test_valid_bbox_pass(self):
        """Test that valid bbox doesn't raise."""
        validate_bbox_xyxy((100, 100, 200, 200), 512, 512)  # Should not raise

    def test_negative_coord_raises(self):
        """Test that negative coordinates raise error."""
        with pytest.raises(CoordinateValidationError):
            validate_bbox_xyxy((-1, 100, 200, 200), 512, 512)

    def test_exceeds_bounds_raises(self):
        """Test that exceeding bounds raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_bbox_xyxy((100, 100, 600, 200), 512, 512)

    def test_invalid_dimensions_raises(self):
        """Test that x1 >= x2 or y1 >= y2 raises error."""
        with pytest.raises(CoordinateValidationError):
            validate_bbox_xyxy((200, 100, 100, 200), 512, 512)  # x1 > x2


class TestCreateCoordinateDict:
    """Tests for create_coordinate_dict function."""

    def test_basic_coordinate_dict(self):
        """Test basic coordinate dict creation."""
        result = create_coordinate_dict(100.5, 200.5)

        assert result['x'] == 100.5
        assert result['y'] == 200.5
        assert result['center_xy'] == [100.5, 200.5]

    def test_with_prefix(self):
        """Test coordinate dict with prefix."""
        result = create_coordinate_dict(100, 200, prefix="global_")

        assert result['global_x'] == 100
        assert result['global_y'] == 200
        assert result['global_center_xy'] == [100, 200]

    def test_with_rounded(self):
        """Test coordinate dict with rounded values."""
        result = create_coordinate_dict(100.6, 200.4, include_rounded=True)

        assert result['x_rounded'] == 101
        assert result['y_rounded'] == 200


class TestFormatCoordinatesForExport:
    """Tests for format_coordinates_for_export function."""

    def test_basic_export_format(self):
        """Test basic export format."""
        result = format_coordinates_for_export(12345.6, 67890.3)

        assert result['global_x_px'] == 12345.6
        assert result['global_y_px'] == 67890.3
        assert result['global_center_xy'] == [12345.6, 67890.3]

    def test_with_pixel_size(self):
        """Test export format with pixel size conversion."""
        result = format_coordinates_for_export(1000, 2000, pixel_size_um=0.22)

        assert result['global_x_um'] == 220.0
        assert result['global_y_um'] == 440.0

    def test_with_local_coords(self):
        """Test export format with local coordinates."""
        result = format_coordinates_for_export(
            12345, 67890,
            local_x=345, local_y=890,
            tile_origin_x=12000, tile_origin_y=67000
        )

        assert result['local_x_px'] == 345
        assert result['local_y_px'] == 890
        assert result['tile_origin_xy'] == [12000, 67000]


class TestConvertDetectionsToSpatialUIDs:
    """Tests for convert_detections_to_spatial_uids function."""

    def test_convert_with_center_field(self):
        """Test conversion when detections have center field."""
        detections = [
            {'global_id': 123, 'center': [1000, 2000]},
            {'global_id': 456, 'center': [3000, 4000]},
        ]
        result = convert_detections_to_spatial_uids(detections, "slide_01", "mk")

        assert result[0]['uid'] == "slide_01_mk_1000_2000"
        assert result[1]['uid'] == "slide_01_mk_3000_4000"

    def test_preserves_legacy_global_id(self):
        """Test that legacy global_id is preserved."""
        detections = [{'global_id': 123, 'center': [1000, 2000]}]
        result = convert_detections_to_spatial_uids(detections, "slide", "mk")

        assert result[0]['legacy_global_id'] == 123

    def test_handles_missing_coordinates(self):
        """Test that detections without coordinates are passed through."""
        detections = [{'some_field': 'value'}]
        result = convert_detections_to_spatial_uids(detections, "slide", "mk")

        assert 'uid' not in result[0]
        assert result[0]['some_field'] == 'value'
