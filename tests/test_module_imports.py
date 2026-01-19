"""
Test module imports for the segmentation package.

Verifies that all key modules and functions can be imported correctly
after refactoring. These tests serve as a smoke test to catch import
errors, circular dependencies, and missing exports.

This test file can be run with either pytest or unittest:
    pytest tests/test_module_imports.py -v
    python -m unittest tests.test_module_imports -v
"""

import unittest

# Optional pytest import for running with pytest
try:
    import pytest
except ImportError:
    pytest = None


class TestHTMLExportImports(unittest.TestCase):
    """Test imports from segmentation.io.html_export module."""

    def test_utility_function_imports(self):
        """Test that utility functions can be imported."""
        from segmentation.io.html_export import (
            create_hdf5_dataset,
            draw_mask_contour,
            percentile_normalize,
            get_largest_connected_component,
        )
        assert callable(create_hdf5_dataset)
        assert callable(draw_mask_contour)
        assert callable(percentile_normalize)
        assert callable(get_largest_connected_component)

    def test_mk_hspc_export_imports(self):
        """Test that MK/HSPC export functions can be imported."""
        from segmentation.io.html_export import (
            export_mk_hspc_html_from_ram,
            load_samples_from_ram,
            create_mk_hspc_index,
            generate_mk_hspc_page_html,
            generate_mk_hspc_pages,
        )
        assert callable(export_mk_hspc_html_from_ram)
        assert callable(load_samples_from_ram)
        assert callable(create_mk_hspc_index)
        assert callable(generate_mk_hspc_page_html)
        assert callable(generate_mk_hspc_pages)

    def test_general_export_imports(self):
        """Test that general export functions can be imported."""
        from segmentation.io.html_export import (
            export_samples_to_html,
            generate_annotation_page,
            generate_index_page,
        )
        assert callable(export_samples_to_html)
        assert callable(generate_annotation_page)
        assert callable(generate_index_page)

    def test_all_html_export_imports_at_once(self):
        """Test importing all HTML export functions at once."""
        from segmentation.io.html_export import (
            create_hdf5_dataset,
            draw_mask_contour,
            percentile_normalize,
            get_largest_connected_component,
            export_mk_hspc_html_from_ram,
            load_samples_from_ram,
            create_mk_hspc_index,
            generate_mk_hspc_page_html,
            generate_mk_hspc_pages,
            export_samples_to_html,
            generate_annotation_page,
            generate_index_page,
        )
        # All imports succeeded if we reach this point


class TestConfigImports(unittest.TestCase):
    """Test imports from segmentation.utils.config module."""

    def test_feature_dimension_constants(self):
        """Test that feature dimension constants can be imported."""
        from segmentation.utils.config import (
            MORPHOLOGICAL_FEATURES_COUNT,
            SAM2_EMBEDDING_DIMENSION,
            RESNET_EMBEDDING_DIMENSION,
            TOTAL_FEATURES_PER_CELL,
        )
        assert MORPHOLOGICAL_FEATURES_COUNT == 22
        assert SAM2_EMBEDDING_DIMENSION == 256
        assert RESNET_EMBEDDING_DIMENSION == 2048
        assert TOTAL_FEATURES_PER_CELL == 2326

    def test_pixel_size_constant(self):
        """Test that default pixel size constant can be imported."""
        from segmentation.utils.config import DEFAULT_PIXEL_SIZE_UM
        assert DEFAULT_PIXEL_SIZE_UM == 0.1725

    def test_helper_functions(self):
        """Test that helper functions can be imported and called."""
        from segmentation.utils.config import (
            get_feature_dimensions,
            get_cpu_worker_count,
            validate_config,
        )
        assert callable(get_feature_dimensions)
        assert callable(get_cpu_worker_count)
        assert callable(validate_config)

        # Test get_feature_dimensions returns expected structure
        dims = get_feature_dimensions()
        assert isinstance(dims, dict)
        assert 'morphological' in dims
        assert 'sam2_embedding' in dims
        assert 'resnet_embedding' in dims
        assert 'total' in dims
        assert dims['total'] == dims['morphological'] + dims['sam2_embedding'] + dims['resnet_embedding']

    def test_config_constants_and_defaults(self):
        """Test that config defaults and constants can be imported."""
        from segmentation.utils.config import (
            DEFAULT_CONFIG,
            DETECTION_DEFAULTS,
            DEFAULT_PATHS,
        )
        assert isinstance(DEFAULT_CONFIG, dict)
        assert isinstance(DETECTION_DEFAULTS, dict)
        assert isinstance(DEFAULT_PATHS, dict)

    def test_config_getter_functions(self):
        """Test that config getter functions can be imported."""
        from segmentation.utils.config import (
            get_default_path,
            get_output_dir,
            get_detection_defaults,
            get_pixel_size,
            get_normalization_percentiles,
            load_config,
            save_config,
        )
        assert callable(get_default_path)
        assert callable(get_output_dir)
        assert callable(get_detection_defaults)
        assert callable(get_pixel_size)
        assert callable(get_normalization_percentiles)
        assert callable(load_config)
        assert callable(save_config)


class TestModelManagerImports(unittest.TestCase):
    """Test imports from segmentation.models module."""

    def test_model_manager_imports(self):
        """Test that model manager can be imported."""
        from segmentation.models import (
            ModelManager,
            get_model_manager,
            find_checkpoint,
        )
        assert ModelManager is not None
        assert callable(get_model_manager)
        assert callable(find_checkpoint)


class TestProcessingModuleImports(unittest.TestCase):
    """Test imports from segmentation.processing modules."""

    def test_memory_module_imports(self):
        """Test that memory module functions can be imported."""
        from segmentation.processing.memory import (
            validate_system_resources,
            get_safe_worker_count,
            get_memory_usage,
            log_memory_status,
        )
        assert callable(validate_system_resources)
        assert callable(get_safe_worker_count)
        assert callable(get_memory_usage)
        assert callable(log_memory_status)

    def test_mk_hspc_utils_imports(self):
        """Test that MK/HSPC utilities can be imported."""
        from segmentation.processing.mk_hspc_utils import (
            ensure_rgb_array,
            check_tile_validity,
            prepare_tile_for_detection,
            build_mk_hspc_result,
            extract_tile_from_shared_memory,
        )
        assert callable(ensure_rgb_array)
        assert callable(check_tile_validity)
        assert callable(prepare_tile_for_detection)
        assert callable(build_mk_hspc_result)
        assert callable(extract_tile_from_shared_memory)

    def test_coordinates_module_imports(self):
        """Test that coordinates module can be imported."""
        from segmentation.processing.coordinates import (
            tile_to_global_coords,
            global_to_tile_coords,
            regionprop_centroid_to_xy,
            xy_to_array_index,
            array_index_to_xy,
            extract_crop_bounds,
            extract_crop,
            generate_uid,
            mask_to_crop_coords,
        )
        assert callable(tile_to_global_coords)
        assert callable(global_to_tile_coords)
        assert callable(regionprop_centroid_to_xy)
        assert callable(xy_to_array_index)
        assert callable(array_index_to_xy)
        assert callable(extract_crop_bounds)
        assert callable(extract_crop)
        assert callable(generate_uid)
        assert callable(mask_to_crop_coords)


class TestDetectionModuleImports(unittest.TestCase):
    """Test imports from segmentation.detection modules."""

    def test_registry_imports(self):
        """Test that strategy registry can be imported."""
        from segmentation.detection.registry import StrategyRegistry
        assert StrategyRegistry is not None
        assert callable(StrategyRegistry.list_strategies)
        assert callable(StrategyRegistry.create)
        assert callable(StrategyRegistry.register)

    def test_tissue_detection_imports(self):
        """Test that tissue detection functions can be imported."""
        from segmentation.detection.tissue import (
            calculate_block_variances,
            has_tissue,
            calibrate_tissue_threshold,
            filter_tissue_tiles,
        )
        assert callable(calculate_block_variances)
        assert callable(has_tissue)
        assert callable(calibrate_tissue_threshold)
        assert callable(filter_tissue_tiles)


class TestIOModuleImports(unittest.TestCase):
    """Test imports from segmentation.io modules."""

    def test_czi_loader_imports(self):
        """Test that CZI loader can be imported."""
        from segmentation.io.czi_loader import CZILoader
        assert CZILoader is not None


class TestPackageLevelImports(unittest.TestCase):
    """Test that package-level imports work correctly."""

    def test_segmentation_package_import(self):
        """Test that main segmentation package can be imported."""
        import segmentation
        assert segmentation is not None

    def test_subpackage_imports(self):
        """Test that subpackages can be imported."""
        from segmentation import io
        from segmentation import utils
        from segmentation import models
        from segmentation import detection
        from segmentation import processing
        assert io is not None
        assert utils is not None
        assert models is not None
        assert detection is not None
        assert processing is not None


if __name__ == "__main__":
    if pytest is not None:
        pytest.main([__file__, "-v"])
    else:
        unittest.main(verbosity=2)
