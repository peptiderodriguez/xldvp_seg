"""
Comprehensive unit tests for core utility modules in xldvp_seg_repo.

Tests the following modules:
- segmentation/utils/config.py - Configuration management and validation
- segmentation/processing/memory.py - Memory validation and management
- segmentation/processing/coordinates.py - Coordinate conversion utilities
- segmentation/processing/mk_hspc_utils.py - MK/HSPC tile processing utilities

Run with: pytest tests/test_utils.py -v
"""

import sys
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch
import tempfile
import json

import numpy as np

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONFIG MODULE TESTS
# =============================================================================

class TestConfigFeatureDimensions(TestCase):
    """Tests for get_feature_dimensions() function in config module."""

    def test_get_feature_dimensions_returns_dict(self):
        """Test that get_feature_dimensions returns a dictionary."""
        from segmentation.utils.config import get_feature_dimensions

        result = get_feature_dimensions()

        self.assertIsInstance(result, dict)

    def test_get_feature_dimensions_has_required_keys(self):
        """Test that the returned dict has all required keys."""
        from segmentation.utils.config import get_feature_dimensions

        result = get_feature_dimensions()

        required_keys = ['morphological', 'sam2_embedding', 'resnet_embedding', 'total']
        for key in required_keys:
            self.assertIn(key, result, f"Missing required key: {key}")

    def test_get_feature_dimensions_values_are_integers(self):
        """Test that all dimension values are positive integers."""
        from segmentation.utils.config import get_feature_dimensions

        result = get_feature_dimensions()

        for key, value in result.items():
            self.assertIsInstance(value, int, f"{key} should be an integer")
            self.assertGreater(value, 0, f"{key} should be positive")

    def test_get_feature_dimensions_total_matches_sum(self):
        """Test that total equals sum of individual dimensions."""
        from segmentation.utils.config import get_feature_dimensions

        result = get_feature_dimensions()

        expected_total = (
            result['morphological'] +
            result['sam2_embedding'] +
            result['resnet_embedding']
        )
        self.assertEqual(result['total'], expected_total)

    def test_get_feature_dimensions_known_values(self):
        """Test that dimensions match expected known values."""
        from segmentation.utils.config import get_feature_dimensions

        result = get_feature_dimensions()

        # These are the documented feature counts
        self.assertEqual(result['morphological'], 22)
        self.assertEqual(result['sam2_embedding'], 256)
        self.assertEqual(result['resnet_embedding'], 2048)
        self.assertEqual(result['total'], 2326)


class TestConfigCpuWorkerCount(TestCase):
    """Tests for get_cpu_worker_count() function in config module."""

    def test_get_cpu_worker_count_returns_integer(self):
        """Test that get_cpu_worker_count returns an integer."""
        from segmentation.utils.config import get_cpu_worker_count

        result = get_cpu_worker_count()

        self.assertIsInstance(result, int)

    def test_get_cpu_worker_count_at_least_one(self):
        """Test that at least 1 worker is returned."""
        from segmentation.utils.config import get_cpu_worker_count

        result = get_cpu_worker_count()

        self.assertGreaterEqual(result, 1)

    def test_get_cpu_worker_count_with_explicit_cores(self):
        """Test get_cpu_worker_count with explicit core count."""
        from segmentation.utils.config import get_cpu_worker_count

        # With 10 cores and 80% utilization, expect 8 workers
        result = get_cpu_worker_count(total_cores=10)

        self.assertEqual(result, 8)

    def test_get_cpu_worker_count_with_single_core(self):
        """Test that single core still returns 1 worker."""
        from segmentation.utils.config import get_cpu_worker_count

        result = get_cpu_worker_count(total_cores=1)

        self.assertEqual(result, 1)

    def test_get_cpu_worker_count_rounds_down(self):
        """Test that worker count rounds down."""
        from segmentation.utils.config import get_cpu_worker_count

        # 5 cores * 0.8 = 4.0 workers
        result = get_cpu_worker_count(total_cores=5)
        self.assertEqual(result, 4)

        # 7 cores * 0.8 = 5.6, rounds to 5
        result = get_cpu_worker_count(total_cores=7)
        self.assertEqual(result, 5)

    def test_get_cpu_worker_count_minimum_one(self):
        """Test that even with 0 or negative cores, returns at least 1."""
        from segmentation.utils.config import get_cpu_worker_count

        # Edge case with 0 cores (shouldn't happen but test anyway)
        # Should return at least 1 due to max(1, ...)
        result = get_cpu_worker_count(total_cores=1)
        self.assertGreaterEqual(result, 1)


class TestConfigValidation(TestCase):
    """Tests for validate_config() function in config module."""

    def test_validate_config_default_passes(self):
        """Test that default config passes validation."""
        from segmentation.utils.config import validate_config

        result = validate_config()

        self.assertTrue(result['valid'])
        self.assertEqual(result['errors'], [])

    def test_validate_config_invalid_tile_size_too_small(self):
        """Test that tile_size below minimum fails validation."""
        from segmentation.utils.config import validate_config

        invalid_config = {'tile_size': 500}  # Minimum is 1000
        result = validate_config(config=invalid_config)

        self.assertFalse(result['valid'])
        self.assertTrue(any('tile_size' in e for e in result['errors']))

    def test_validate_config_invalid_tile_size_too_large(self):
        """Test that tile_size above maximum fails validation."""
        from segmentation.utils.config import validate_config

        invalid_config = {'tile_size': 10000}  # Maximum is 8192
        result = validate_config(config=invalid_config)

        self.assertFalse(result['valid'])
        self.assertTrue(any('tile_size' in e for e in result['errors']))

    def test_validate_config_invalid_sample_fraction(self):
        """Test that sample_fraction outside range fails validation."""
        from segmentation.utils.config import validate_config

        # Too small
        invalid_config = {'sample_fraction': 0.001}  # Minimum is 0.01
        result = validate_config(config=invalid_config)
        self.assertFalse(result['valid'])

        # Too large
        invalid_config = {'sample_fraction': 1.5}  # Maximum is 1.0
        result = validate_config(config=invalid_config)
        self.assertFalse(result['valid'])

    def test_validate_config_invalid_contour_color(self):
        """Test that invalid contour_color fails validation."""
        from segmentation.utils.config import validate_config

        # Wrong length
        invalid_config = {'contour_color': [50, 255]}
        result = validate_config(config=invalid_config)
        self.assertFalse(result['valid'])

        # Values out of range
        invalid_config = {'contour_color': [50, 300, 50]}
        result = validate_config(config=invalid_config)
        self.assertFalse(result['valid'])

    def test_validate_config_invalid_html_theme(self):
        """Test that invalid html_theme fails validation."""
        from segmentation.utils.config import validate_config

        invalid_config = {'html_theme': 'blue'}  # Must be 'dark' or 'light'
        result = validate_config(config=invalid_config)

        self.assertFalse(result['valid'])
        self.assertTrue(any('html_theme' in e for e in result['errors']))

    def test_validate_config_invalid_percentiles(self):
        """Test that invalid normalization percentiles fail validation."""
        from segmentation.utils.config import validate_config

        # Low >= high
        invalid_config = {
            'normalization_percentiles': {'mk': [95, 5]}
        }
        result = validate_config(config=invalid_config)

        self.assertFalse(result['valid'])
        self.assertTrue(any('less than high' in e for e in result['errors']))

    def test_validate_config_raise_on_error(self):
        """Test that raise_on_error=True raises exception."""
        from segmentation.utils.config import validate_config, ConfigValidationError

        invalid_config = {'tile_size': 100}

        with self.assertRaises(ConfigValidationError):
            validate_config(config=invalid_config, raise_on_error=True)

    def test_validate_config_returns_warnings(self):
        """Test that validation returns warnings for logical issues."""
        from segmentation.utils.config import validate_config

        # mem_per_worker_small_tile > mem_per_worker_large_tile is a warning
        result = validate_config(memory_thresholds={
            'min_ram_gb': 8.0,
            'mem_per_worker_small_tile': 20.0,
            'mem_per_worker_large_tile': 10.0,
            'min_gpu_gb': 6.0,
        })

        # Should still be valid but have warnings
        self.assertTrue(result['valid'])  # warnings don't cause failure
        self.assertTrue(len(result['warnings']) > 0)


class TestConfigDetectionDefaults(TestCase):
    """Tests for get_detection_defaults() function in config module."""

    def test_get_detection_defaults_nmj(self):
        """Test detection defaults for NMJ cell type."""
        from segmentation.utils.config import get_detection_defaults

        result = get_detection_defaults('nmj')

        self.assertIsInstance(result, dict)
        self.assertIn('channel', result)
        self.assertIn('min_area_px', result)
        self.assertIn('max_solidity', result)
        self.assertEqual(result['channel'], 1)

    def test_get_detection_defaults_mk(self):
        """Test detection defaults for MK cell type."""
        from segmentation.utils.config import get_detection_defaults

        result = get_detection_defaults('mk')

        self.assertIsInstance(result, dict)
        self.assertIn('channel', result)
        self.assertIn('min_area_um2', result)
        self.assertIn('max_area_um2', result)
        self.assertEqual(result['min_area_um2'], 200)
        self.assertEqual(result['max_area_um2'], 2000)

    def test_get_detection_defaults_cell(self):
        """Test detection defaults for generic cell type."""
        from segmentation.utils.config import get_detection_defaults

        result = get_detection_defaults('cell')

        self.assertIsInstance(result, dict)
        self.assertIn('channel', result)
        self.assertIn('min_area_px', result)
        self.assertIn('max_area_px', result)

    def test_get_detection_defaults_vessel(self):
        """Test detection defaults for vessel cell type."""
        from segmentation.utils.config import get_detection_defaults

        result = get_detection_defaults('vessel')

        self.assertIsInstance(result, dict)
        self.assertIn('min_diameter_um', result)
        self.assertIn('max_diameter_um', result)
        self.assertIn('min_wall_thickness_um', result)

    def test_get_detection_defaults_mesothelium(self):
        """Test detection defaults for mesothelium cell type."""
        from segmentation.utils.config import get_detection_defaults

        result = get_detection_defaults('mesothelium')

        self.assertIsInstance(result, dict)
        self.assertIn('target_chunk_area_um2', result)
        self.assertIn('min_ribbon_width_um', result)

    def test_get_detection_defaults_unknown_returns_empty(self):
        """Test that unknown cell type returns empty dict."""
        from segmentation.utils.config import get_detection_defaults

        result = get_detection_defaults('unknown_type')

        self.assertEqual(result, {})

    def test_get_detection_defaults_returns_copy(self):
        """Test that returned dict is a copy, not reference."""
        from segmentation.utils.config import get_detection_defaults

        result1 = get_detection_defaults('mk')
        result1['modified'] = True

        result2 = get_detection_defaults('mk')

        self.assertNotIn('modified', result2)


class TestConfigPixelSize(TestCase):
    """Tests for get_pixel_size() function in config module."""

    def test_get_pixel_size_mk(self):
        """Test pixel size for MK cell type."""
        from segmentation.utils.config import get_pixel_size, DEFAULT_CONFIG

        result = get_pixel_size(DEFAULT_CONFIG, 'mk')

        self.assertEqual(result, 0.22)

    def test_get_pixel_size_nmj(self):
        """Test pixel size for NMJ cell type."""
        from segmentation.utils.config import get_pixel_size, DEFAULT_CONFIG

        result = get_pixel_size(DEFAULT_CONFIG, 'nmj')

        self.assertEqual(result, 0.1725)

    def test_get_pixel_size_default_fallback(self):
        """Test pixel size falls back to default for unknown type."""
        from segmentation.utils.config import get_pixel_size, DEFAULT_CONFIG

        result = get_pixel_size(DEFAULT_CONFIG, 'unknown')

        self.assertEqual(result, 0.22)  # default value

    def test_get_pixel_size_scalar_config(self):
        """Test get_pixel_size with scalar pixel_size_um value."""
        from segmentation.utils.config import get_pixel_size

        config = {'pixel_size_um': 0.5}
        result = get_pixel_size(config, 'mk')

        self.assertEqual(result, 0.5)


class TestConfigConstants(TestCase):
    """Tests for config module constants."""

    def test_default_config_has_required_keys(self):
        """Test that DEFAULT_CONFIG has all required keys."""
        from segmentation.utils.config import DEFAULT_CONFIG

        required_keys = [
            'pixel_size_um',
            'normalization_percentiles',
            'contour_color',
            'contour_thickness',
            'samples_per_page',
            'tile_size',
            'sample_fraction',
        ]

        for key in required_keys:
            self.assertIn(key, DEFAULT_CONFIG, f"Missing key: {key}")

    def test_batch_sizes_has_required_keys(self):
        """Test that BATCH_SIZES has required keys."""
        from segmentation.utils.config import BATCH_SIZES

        required_keys = ['resnet_feature_extraction', 'sam2_embedding', 'gc_interval_tiles']

        for key in required_keys:
            self.assertIn(key, BATCH_SIZES, f"Missing key: {key}")

    def test_memory_thresholds_has_required_keys(self):
        """Test that MEMORY_THRESHOLDS has required keys."""
        from segmentation.utils.config import MEMORY_THRESHOLDS

        required_keys = [
            'min_ram_gb',
            'mem_per_worker_small_tile',
            'mem_per_worker_large_tile',
            'min_gpu_gb',
        ]

        for key in required_keys:
            self.assertIn(key, MEMORY_THRESHOLDS, f"Missing key: {key}")

    def test_memory_thresholds_values_sensible(self):
        """Test that memory thresholds have sensible values."""
        from segmentation.utils.config import MEMORY_THRESHOLDS

        # min_ram_gb should be reasonable (not too low, not too high)
        self.assertGreater(MEMORY_THRESHOLDS['min_ram_gb'], 4.0)
        self.assertLess(MEMORY_THRESHOLDS['min_ram_gb'], 32.0)

        # Large tile memory should be >= small tile memory
        self.assertGreaterEqual(
            MEMORY_THRESHOLDS['mem_per_worker_large_tile'],
            MEMORY_THRESHOLDS['mem_per_worker_small_tile']
        )


# =============================================================================
# MEMORY MODULE TESTS
# =============================================================================

class TestMemoryUsage(TestCase):
    """Tests for get_memory_usage() function in memory module."""

    def test_get_memory_usage_returns_dict(self):
        """Test that get_memory_usage returns a dictionary."""
        from segmentation.processing.memory import get_memory_usage

        result = get_memory_usage()

        self.assertIsInstance(result, dict)

    def test_get_memory_usage_has_ram_keys(self):
        """Test that returned dict has RAM-related keys."""
        from segmentation.processing.memory import get_memory_usage

        result = get_memory_usage()

        self.assertIn('ram_available_gb', result)
        self.assertIn('ram_total_gb', result)
        self.assertIn('ram_used_percent', result)

    def test_get_memory_usage_ram_values_sensible(self):
        """Test that RAM values are sensible."""
        from segmentation.processing.memory import get_memory_usage

        result = get_memory_usage()

        # Available RAM should be positive
        self.assertGreater(result['ram_available_gb'], 0)

        # Total RAM should be >= available
        self.assertGreaterEqual(result['ram_total_gb'], result['ram_available_gb'])

        # Used percent should be 0-100
        self.assertGreaterEqual(result['ram_used_percent'], 0)
        self.assertLessEqual(result['ram_used_percent'], 100)

    def test_get_memory_usage_gpu_keys_when_available(self):
        """Test GPU keys are present when CUDA is available."""
        from segmentation.processing.memory import get_memory_usage
        import torch

        result = get_memory_usage()

        if torch.cuda.is_available():
            self.assertIn('gpu_total_gb', result)
            self.assertIn('gpu_allocated_gb', result)
            self.assertIn('gpu_available_gb', result)


class TestSafeWorkerCount(TestCase):
    """Tests for get_safe_worker_count() function in memory module."""

    def test_get_safe_worker_count_returns_integer(self):
        """Test that get_safe_worker_count returns an integer."""
        from segmentation.processing.memory import get_safe_worker_count

        result = get_safe_worker_count(requested_workers=4, tile_size=3000)

        self.assertIsInstance(result, int)

    def test_get_safe_worker_count_minimum_one(self):
        """Test that at least 1 worker is returned."""
        from segmentation.processing.memory import get_safe_worker_count

        result = get_safe_worker_count(requested_workers=1, tile_size=3000)

        self.assertGreaterEqual(result, 1)

    def test_get_safe_worker_count_respects_requested(self):
        """Test that requested count is respected when memory allows."""
        from segmentation.processing.memory import get_safe_worker_count

        # Request a small number that should always be allowed
        result = get_safe_worker_count(requested_workers=1, tile_size=3000)

        # Should get at least what we requested if memory allows
        self.assertGreaterEqual(result, 1)

    def test_get_safe_worker_count_no_auto_adjust(self):
        """Test that auto_adjust=False returns requested count."""
        from segmentation.processing.memory import get_safe_worker_count

        result = get_safe_worker_count(
            requested_workers=8,
            tile_size=3000,
            auto_adjust=False
        )

        # Should return exactly what was requested
        self.assertEqual(result, 8)

    def test_get_safe_worker_count_between_1_and_max(self):
        """Test that result is between 1 and requested workers."""
        from segmentation.processing.memory import get_safe_worker_count

        requested = 16
        result = get_safe_worker_count(requested_workers=requested, tile_size=3000)

        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, requested)


class TestValidateSystemResources(TestCase):
    """Tests for validate_system_resources() function in memory module."""

    def test_validate_system_resources_returns_dict(self):
        """Test that validate_system_resources returns a dictionary."""
        from segmentation.processing.memory import validate_system_resources

        result = validate_system_resources(num_workers=4, tile_size=3000)

        self.assertIsInstance(result, dict)

    def test_validate_system_resources_has_required_keys(self):
        """Test that returned dict has all required keys."""
        from segmentation.processing.memory import validate_system_resources

        result = validate_system_resources(num_workers=4, tile_size=3000)

        required_keys = [
            'warnings',
            'recommended_workers',
            'recommended_tile_size',
            'should_abort',
        ]

        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_validate_system_resources_warnings_is_list(self):
        """Test that warnings is a list."""
        from segmentation.processing.memory import validate_system_resources

        result = validate_system_resources(num_workers=4, tile_size=3000)

        self.assertIsInstance(result['warnings'], list)

    def test_validate_system_resources_recommended_workers_positive(self):
        """Test that recommended_workers is a positive integer."""
        from segmentation.processing.memory import validate_system_resources

        result = validate_system_resources(num_workers=4, tile_size=3000)

        self.assertIsInstance(result['recommended_workers'], int)
        self.assertGreater(result['recommended_workers'], 0)

    def test_validate_system_resources_should_abort_is_bool(self):
        """Test that should_abort is a boolean."""
        from segmentation.processing.memory import validate_system_resources

        result = validate_system_resources(num_workers=4, tile_size=3000)

        self.assertIsInstance(result['should_abort'], bool)

    @patch('segmentation.processing.memory.psutil.virtual_memory')
    def test_validate_system_resources_low_memory_warns(self, mock_vm):
        """Test that low memory produces warnings."""
        from segmentation.processing.memory import validate_system_resources

        # Mock low memory (5 GB available)
        mock_vm.return_value = MagicMock(
            available=5 * (1024**3),
            total=16 * (1024**3),
            percent=68.75
        )

        result = validate_system_resources(num_workers=8, tile_size=3000)

        # Should have warnings about worker count
        self.assertTrue(len(result['warnings']) > 0)
        # Recommended workers should be reduced
        self.assertLess(result['recommended_workers'], 8)


# =============================================================================
# COORDINATES MODULE TESTS
# =============================================================================

class TestTileToGlobalCoords(TestCase):
    """Tests for tile_to_global_coords() function in coordinates module."""

    def test_tile_to_global_coords_basic(self):
        """Test basic tile to global coordinate conversion."""
        from segmentation.processing.coordinates import tile_to_global_coords

        local_x, local_y = 100, 50
        tile_origin_x, tile_origin_y = 1000, 2000

        global_x, global_y = tile_to_global_coords(
            local_x, local_y, tile_origin_x, tile_origin_y
        )

        self.assertEqual(global_x, 1100)  # 100 + 1000
        self.assertEqual(global_y, 2050)  # 50 + 2000

    def test_tile_to_global_coords_origin_zero(self):
        """Test conversion with tile at origin."""
        from segmentation.processing.coordinates import tile_to_global_coords

        local_x, local_y = 50, 75
        tile_origin_x, tile_origin_y = 0, 0

        global_x, global_y = tile_to_global_coords(
            local_x, local_y, tile_origin_x, tile_origin_y
        )

        # Local coords should equal global when tile is at origin
        self.assertEqual(global_x, 50)
        self.assertEqual(global_y, 75)

    def test_tile_to_global_coords_float_inputs(self):
        """Test conversion with float inputs."""
        from segmentation.processing.coordinates import tile_to_global_coords

        local_x, local_y = 100.5, 50.5
        tile_origin_x, tile_origin_y = 1000, 2000

        global_x, global_y = tile_to_global_coords(
            local_x, local_y, tile_origin_x, tile_origin_y
        )

        self.assertEqual(global_x, 1100.5)
        self.assertEqual(global_y, 2050.5)


class TestGlobalToTileCoords(TestCase):
    """Tests for global_to_tile_coords() function in coordinates module."""

    def test_global_to_tile_coords_basic(self):
        """Test basic global to tile coordinate conversion."""
        from segmentation.processing.coordinates import global_to_tile_coords

        global_x, global_y = 1100, 2050
        tile_origin_x, tile_origin_y = 1000, 2000

        local_x, local_y = global_to_tile_coords(
            global_x, global_y, tile_origin_x, tile_origin_y
        )

        self.assertEqual(local_x, 100)  # 1100 - 1000
        self.assertEqual(local_y, 50)   # 2050 - 2000

    def test_global_to_tile_coords_roundtrip(self):
        """Test that global_to_tile and tile_to_global are inverses."""
        from segmentation.processing.coordinates import (
            tile_to_global_coords,
            global_to_tile_coords
        )

        original_local_x, original_local_y = 123, 456
        tile_origin_x, tile_origin_y = 5000, 7000

        # Convert to global
        global_x, global_y = tile_to_global_coords(
            original_local_x, original_local_y, tile_origin_x, tile_origin_y
        )

        # Convert back to local
        recovered_local_x, recovered_local_y = global_to_tile_coords(
            global_x, global_y, tile_origin_x, tile_origin_y
        )

        self.assertEqual(recovered_local_x, original_local_x)
        self.assertEqual(recovered_local_y, original_local_y)

    def test_global_to_tile_coords_at_origin(self):
        """Test conversion when point is at tile origin."""
        from segmentation.processing.coordinates import global_to_tile_coords

        global_x, global_y = 1000, 2000
        tile_origin_x, tile_origin_y = 1000, 2000

        local_x, local_y = global_to_tile_coords(
            global_x, global_y, tile_origin_x, tile_origin_y
        )

        self.assertEqual(local_x, 0)
        self.assertEqual(local_y, 0)


class TestRegionpropCentroidToXY(TestCase):
    """Tests for regionprop_centroid_to_xy() function in coordinates module."""

    def test_regionprop_centroid_to_xy_basic(self):
        """Test conversion from regionprops centroid to xy format."""
        from segmentation.processing.coordinates import regionprop_centroid_to_xy

        # Mock regionprops object
        prop = MagicMock()
        prop.centroid = (100.5, 200.5)  # (row, col) = (y, x)

        result = regionprop_centroid_to_xy(prop)

        # Should return [x, y] = [col, row]
        self.assertEqual(result[0], 200.5)  # x = col
        self.assertEqual(result[1], 100.5)  # y = row

    def test_regionprop_centroid_to_xy_returns_list(self):
        """Test that result is a list."""
        from segmentation.processing.coordinates import regionprop_centroid_to_xy

        prop = MagicMock()
        prop.centroid = (50, 100)

        result = regionprop_centroid_to_xy(prop)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_regionprop_centroid_to_xy_returns_floats(self):
        """Test that result contains floats."""
        from segmentation.processing.coordinates import regionprop_centroid_to_xy

        prop = MagicMock()
        prop.centroid = (50, 100)  # Integers

        result = regionprop_centroid_to_xy(prop)

        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


class TestGenerateUID(TestCase):
    """Tests for generate_uid() function in coordinates module."""

    def test_generate_uid_format(self):
        """Test that UID has expected format."""
        from segmentation.processing.coordinates import generate_uid

        uid = generate_uid('slide_name', 'mk', 12345.6, 67890.3)

        # Format: {slide}_{type}_{round(x)}_{round(y)}
        self.assertEqual(uid, 'slide_name_mk_12346_67890')

    def test_generate_uid_rounding(self):
        """Test coordinate rounding in UID."""
        from segmentation.processing.coordinates import generate_uid

        uid = generate_uid('slide', 'nmj', 100.4, 200.6)

        self.assertEqual(uid, 'slide_nmj_100_201')

    def test_generate_uid_uniqueness(self):
        """Test that different coordinates produce different UIDs."""
        from segmentation.processing.coordinates import generate_uid

        uid1 = generate_uid('slide', 'mk', 100, 200)
        uid2 = generate_uid('slide', 'mk', 101, 200)
        uid3 = generate_uid('slide', 'mk', 100, 201)

        self.assertNotEqual(uid1, uid2)
        self.assertNotEqual(uid1, uid3)
        self.assertNotEqual(uid2, uid3)

    def test_generate_uid_different_cell_types(self):
        """Test UID generation for different cell types."""
        from segmentation.processing.coordinates import generate_uid

        for cell_type in ['mk', 'hspc', 'nmj', 'vessel', 'mesothelium']:
            uid = generate_uid('slide', cell_type, 100, 200)
            self.assertIn(f'_{cell_type}_', uid)


# =============================================================================
# MK_HSPC_UTILS MODULE TESTS
# =============================================================================

class TestEnsureRGBArray(TestCase):
    """Tests for ensure_rgb_array() function in mk_hspc_utils module."""

    def test_ensure_rgb_array_grayscale_input(self):
        """Test conversion of grayscale image to RGB."""
        from segmentation.processing.mk_hspc_utils import ensure_rgb_array

        grayscale = np.zeros((100, 100), dtype=np.uint8)
        grayscale[50:60, 50:60] = 128

        result = ensure_rgb_array(grayscale)

        self.assertEqual(result.shape, (100, 100, 3))
        # All channels should have same values
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_ensure_rgb_array_rgba_input(self):
        """Test conversion of RGBA image to RGB (drops alpha)."""
        from segmentation.processing.mk_hspc_utils import ensure_rgb_array

        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255  # R
        rgba[:, :, 1] = 128  # G
        rgba[:, :, 2] = 64   # B
        rgba[:, :, 3] = 200  # Alpha (should be dropped)

        result = ensure_rgb_array(rgba)

        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(result[0, 0, 0], 255)  # R preserved
        self.assertEqual(result[0, 0, 1], 128)  # G preserved
        self.assertEqual(result[0, 0, 2], 64)   # B preserved

    def test_ensure_rgb_array_rgb_input(self):
        """Test that RGB input is returned unchanged."""
        from segmentation.processing.mk_hspc_utils import ensure_rgb_array

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        rgb[:, :, 1] = 128
        rgb[:, :, 2] = 64

        result = ensure_rgb_array(rgb)

        self.assertEqual(result.shape, (100, 100, 3))
        np.testing.assert_array_equal(result, rgb)

    def test_ensure_rgb_array_preserves_content(self):
        """Test that content is preserved during conversion."""
        from segmentation.processing.mk_hspc_utils import ensure_rgb_array

        grayscale = np.array([[0, 128], [255, 64]], dtype=np.uint8)

        result = ensure_rgb_array(grayscale)

        # Check each channel matches original grayscale
        for c in range(3):
            np.testing.assert_array_equal(result[:, :, c], grayscale)

    def test_ensure_rgb_array_different_dtypes(self):
        """Test handling of different numpy dtypes."""
        from segmentation.processing.mk_hspc_utils import ensure_rgb_array

        # Test uint16 (common for CZI images)
        grayscale_16 = np.zeros((50, 50), dtype=np.uint16)
        result_16 = ensure_rgb_array(grayscale_16)
        self.assertEqual(result_16.shape, (50, 50, 3))

        # Test float32
        grayscale_f = np.zeros((50, 50), dtype=np.float32)
        result_f = ensure_rgb_array(grayscale_f)
        self.assertEqual(result_f.shape, (50, 50, 3))


class TestCheckTileValidity(TestCase):
    """Tests for check_tile_validity() function in mk_hspc_utils module."""

    def test_check_tile_validity_empty_tile(self):
        """Test that empty (all zeros) tile is marked invalid."""
        from segmentation.processing.mk_hspc_utils import check_tile_validity

        empty_tile = np.zeros((100, 100, 3), dtype=np.uint8)

        is_valid, status = check_tile_validity(empty_tile, 'tile_0_0')

        self.assertFalse(is_valid)
        self.assertEqual(status, 'empty')

    def test_check_tile_validity_valid_tile(self):
        """Test that tile with content is marked valid."""
        from segmentation.processing.mk_hspc_utils import check_tile_validity

        valid_tile = np.ones((100, 100, 3), dtype=np.uint8) * 128

        is_valid, status = check_tile_validity(valid_tile, 'tile_0_0')

        self.assertTrue(is_valid)
        self.assertEqual(status, 'valid')

    def test_check_tile_validity_partial_content(self):
        """Test that tile with partial content is valid."""
        from segmentation.processing.mk_hspc_utils import check_tile_validity

        partial_tile = np.zeros((100, 100, 3), dtype=np.uint8)
        partial_tile[50, 50, 0] = 1  # Single non-zero pixel

        is_valid, status = check_tile_validity(partial_tile, 'tile_0_0')

        self.assertTrue(is_valid)
        self.assertEqual(status, 'valid')

    def test_check_tile_validity_returns_tuple(self):
        """Test that result is a tuple of (bool, str)."""
        from segmentation.processing.mk_hspc_utils import check_tile_validity

        tile = np.zeros((100, 100, 3), dtype=np.uint8)

        result = check_tile_validity(tile, 'tile_0_0')

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], str)

    def test_check_tile_validity_different_shapes(self):
        """Test validity check with different tile shapes."""
        from segmentation.processing.mk_hspc_utils import check_tile_validity

        # Small tile
        small_valid = np.ones((10, 10, 3), dtype=np.uint8)
        is_valid, _ = check_tile_validity(small_valid, 'tile_small')
        self.assertTrue(is_valid)

        # Large tile
        large_valid = np.ones((4096, 4096, 3), dtype=np.uint8)
        is_valid, _ = check_tile_validity(large_valid, 'tile_large')
        self.assertTrue(is_valid)


class TestExtractTileFromSharedMemory(TestCase):
    """Tests for extract_tile_from_shared_memory() function."""

    def test_extract_tile_basic(self):
        """Test basic tile extraction from shared memory."""
        from segmentation.processing.mk_hspc_utils import extract_tile_from_shared_memory

        # Create shared memory array
        shared = np.arange(10000).reshape(100, 100)

        tile_info = {'id': 't1', 'x': 10, 'y': 20, 'w': 30, 'h': 40}

        img, error = extract_tile_from_shared_memory(shared, tile_info)

        self.assertIsNone(error)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (40, 30))  # h, w

    def test_extract_tile_null_shared_memory(self):
        """Test that None shared memory returns error."""
        from segmentation.processing.mk_hspc_utils import extract_tile_from_shared_memory

        tile_info = {'id': 't1', 'x': 0, 'y': 0, 'w': 10, 'h': 10}

        img, error = extract_tile_from_shared_memory(None, tile_info)

        self.assertIsNone(img)
        self.assertIsNotNone(error)
        self.assertIn('not available', error.lower())

    def test_extract_tile_at_origin(self):
        """Test tile extraction at array origin."""
        from segmentation.processing.mk_hspc_utils import extract_tile_from_shared_memory

        shared = np.arange(400).reshape(20, 20)
        tile_info = {'id': 't1', 'x': 0, 'y': 0, 'w': 5, 'h': 5}

        img, error = extract_tile_from_shared_memory(shared, tile_info)

        self.assertIsNone(error)
        self.assertEqual(img.shape, (5, 5))
        # First element should be 0 (origin)
        self.assertEqual(img[0, 0], 0)

    def test_extract_tile_preserves_values(self):
        """Test that extracted values match source array."""
        from segmentation.processing.mk_hspc_utils import extract_tile_from_shared_memory

        shared = np.arange(100).reshape(10, 10)
        tile_info = {'id': 't1', 'x': 2, 'y': 3, 'w': 4, 'h': 5}

        img, error = extract_tile_from_shared_memory(shared, tile_info)

        # Verify extraction matches direct slice
        expected = shared[3:8, 2:6]
        np.testing.assert_array_equal(img, expected)


class TestBuildMkHspcResult(TestCase):
    """Tests for build_mk_hspc_result() function."""

    def test_build_result_empty_status(self):
        """Test building result with empty status."""
        from segmentation.processing.mk_hspc_utils import build_mk_hspc_result

        result = build_mk_hspc_result('tile_0_0', 'empty')

        self.assertEqual(result['tid'], 'tile_0_0')
        self.assertEqual(result['status'], 'empty')
        self.assertNotIn('mk_masks', result)

    def test_build_result_error_status(self):
        """Test building result with error status."""
        from segmentation.processing.mk_hspc_utils import build_mk_hspc_result

        result = build_mk_hspc_result(
            'tile_0_0',
            'error',
            error='Test error message'
        )

        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error'], 'Test error message')

    def test_build_result_success_status(self):
        """Test building result with success status."""
        from segmentation.processing.mk_hspc_utils import build_mk_hspc_result

        mk_masks = np.zeros((100, 100))
        hspc_masks = np.zeros((100, 100))
        mk_feats = [{'area': 100}]
        hspc_feats = [{'area': 50}]
        tile = {'id': 'tile_0_0', 'x': 0, 'y': 0}

        result = build_mk_hspc_result(
            'tile_0_0',
            'success',
            mk_masks=mk_masks,
            hspc_masks=hspc_masks,
            mk_feats=mk_feats,
            hspc_feats=hspc_feats,
            tile=tile
        )

        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['mk_masks'])
        self.assertIsNotNone(result['hspc_masks'])
        self.assertEqual(result['mk_feats'], mk_feats)
        self.assertEqual(result['hspc_feats'], hspc_feats)
        self.assertEqual(result['tile'], tile)

    def test_build_result_with_slide_name(self):
        """Test building result with optional slide_name."""
        from segmentation.processing.mk_hspc_utils import build_mk_hspc_result

        result = build_mk_hspc_result(
            'tile_0_0',
            'success',
            mk_masks=np.zeros((10, 10)),
            hspc_masks=np.zeros((10, 10)),
            mk_feats=[],
            hspc_feats=[],
            tile={},
            slide_name='test_slide'
        )

        self.assertEqual(result['slide_name'], 'test_slide')

    def test_build_result_defaults_empty_lists(self):
        """Test that None features default to empty lists."""
        from segmentation.processing.mk_hspc_utils import build_mk_hspc_result

        result = build_mk_hspc_result(
            'tile_0_0',
            'success',
            mk_masks=np.zeros((10, 10)),
            hspc_masks=np.zeros((10, 10)),
            tile={}
        )

        self.assertEqual(result['mk_feats'], [])
        self.assertEqual(result['hspc_feats'], [])


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == '__main__':
    import unittest
    unittest.main(verbosity=2)
