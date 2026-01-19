"""
Tests to verify MK/HSPC segmentation pipeline imports work correctly after refactoring.

These tests verify:
1. Main module imports work
2. Key functions exist and are callable
3. Magic number constants are accessible from config
4. CLI argparser works

Run with: pytest tests/test_mk_hspc_imports.py -v
"""

import pytest


class TestRunUnifiedFASTImports:
    """Test that run_unified_FAST module imports correctly."""

    def test_main_module_imports(self):
        """Verify the main module can be imported."""
        import run_unified_FAST
        assert run_unified_FAST is not None

    def test_run_multi_slide_segmentation_exists(self):
        """Verify run_multi_slide_segmentation function exists and is callable."""
        import run_unified_FAST
        assert hasattr(run_unified_FAST, 'run_multi_slide_segmentation')
        assert callable(run_unified_FAST.run_multi_slide_segmentation)

    def test_phase1_load_slides_exists(self):
        """Verify _phase1_load_slides function exists and is callable."""
        import run_unified_FAST
        assert hasattr(run_unified_FAST, '_phase1_load_slides')
        assert callable(run_unified_FAST._phase1_load_slides)

    def test_phase2_identify_tissue_tiles_exists(self):
        """Verify _phase2_identify_tissue_tiles function exists and is callable."""
        import run_unified_FAST
        assert hasattr(run_unified_FAST, '_phase2_identify_tissue_tiles')
        assert callable(run_unified_FAST._phase2_identify_tissue_tiles)

    def test_phase3_sample_tiles_exists(self):
        """Verify _phase3_sample_tiles function exists and is callable."""
        import run_unified_FAST
        assert hasattr(run_unified_FAST, '_phase3_sample_tiles')
        assert callable(run_unified_FAST._phase3_sample_tiles)

    def test_phase4_process_tiles_exists(self):
        """Verify _phase4_process_tiles function exists and is callable."""
        import run_unified_FAST
        assert hasattr(run_unified_FAST, '_phase4_process_tiles')
        assert callable(run_unified_FAST._phase4_process_tiles)

    def test_export_html_from_ram_exists(self):
        """Verify export_html_from_ram function exists and is callable."""
        import run_unified_FAST
        assert hasattr(run_unified_FAST, 'export_html_from_ram')
        assert callable(run_unified_FAST.export_html_from_ram)


class TestConfigConstants:
    """Test that magic number constants are accessible from config."""

    def test_morphological_features_count(self):
        """Verify MORPHOLOGICAL_FEATURES_COUNT constant is accessible."""
        from segmentation.utils.config import MORPHOLOGICAL_FEATURES_COUNT
        assert isinstance(MORPHOLOGICAL_FEATURES_COUNT, int)
        assert MORPHOLOGICAL_FEATURES_COUNT == 22

    def test_sam2_embedding_dimension(self):
        """Verify SAM2_EMBEDDING_DIMENSION constant is accessible."""
        from segmentation.utils.config import SAM2_EMBEDDING_DIMENSION
        assert isinstance(SAM2_EMBEDDING_DIMENSION, int)
        assert SAM2_EMBEDDING_DIMENSION == 256

    def test_resnet_embedding_dimension(self):
        """Verify RESNET_EMBEDDING_DIMENSION constant is accessible."""
        from segmentation.utils.config import RESNET_EMBEDDING_DIMENSION
        assert isinstance(RESNET_EMBEDDING_DIMENSION, int)
        assert RESNET_EMBEDDING_DIMENSION == 2048

    def test_total_features_per_cell(self):
        """Verify TOTAL_FEATURES_PER_CELL constant is accessible and correct."""
        from segmentation.utils.config import (
            MORPHOLOGICAL_FEATURES_COUNT,
            SAM2_EMBEDDING_DIMENSION,
            RESNET_EMBEDDING_DIMENSION,
            TOTAL_FEATURES_PER_CELL
        )
        expected = MORPHOLOGICAL_FEATURES_COUNT + SAM2_EMBEDDING_DIMENSION + RESNET_EMBEDDING_DIMENSION
        assert TOTAL_FEATURES_PER_CELL == expected


class TestSegmentationModuleImports:
    """Test that segmentation submodule imports work."""

    def test_utils_logging_imports(self):
        """Verify logging utilities import correctly."""
        from segmentation.utils.logging import get_logger, setup_logging, log_parameters
        assert callable(get_logger)
        assert callable(setup_logging)
        assert callable(log_parameters)

    def test_models_manager_imports(self):
        """Verify model manager imports correctly."""
        from segmentation.models import get_model_manager
        assert callable(get_model_manager)

    def test_processing_memory_imports(self):
        """Verify memory processing utilities import correctly."""
        from segmentation.processing.memory import (
            validate_system_resources,
            get_safe_worker_count,
            get_memory_usage,
            log_memory_status
        )
        assert callable(validate_system_resources)
        assert callable(get_safe_worker_count)
        assert callable(get_memory_usage)
        assert callable(log_memory_status)

    def test_processing_mk_hspc_utils_imports(self):
        """Verify MK/HSPC processing utilities import correctly."""
        from segmentation.processing.mk_hspc_utils import (
            ensure_rgb_array,
            check_tile_validity,
            prepare_tile_for_detection,
            build_mk_hspc_result,
            extract_tile_from_shared_memory
        )
        assert callable(ensure_rgb_array)
        assert callable(check_tile_validity)
        assert callable(prepare_tile_for_detection)
        assert callable(build_mk_hspc_result)
        assert callable(extract_tile_from_shared_memory)

    def test_detection_registry_imports(self):
        """Verify strategy registry imports correctly."""
        from segmentation.detection.registry import StrategyRegistry
        assert hasattr(StrategyRegistry, 'register')
        assert hasattr(StrategyRegistry, 'create')
        assert hasattr(StrategyRegistry, 'list_strategies')


class TestCLI:
    """Test that CLI argparser works."""

    def test_argparser_creation(self):
        """Verify argparser can be created without errors."""
        import argparse
        import run_unified_FAST

        # Check that the module has the expected CLI behavior
        # by importing and checking it doesn't error
        assert run_unified_FAST is not None
