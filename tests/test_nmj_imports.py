"""
Test NMJ (Neuromuscular Junction) pipeline imports.

Verifies that all key modules and functions for NMJ segmentation
can be imported correctly after refactoring.

Usage:
    python tests/test_nmj_imports.py  # Run standalone
    pytest tests/test_nmj_imports.py -v  # Run with pytest (if available)
"""

import sys
from pathlib import Path

# Add repo root to path so imports work when running from tests/ directory
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestNMJStrategyImports:
    """Test imports from NMJ detection strategy module."""

    def test_nmj_strategy_import(self):
        """Test that NMJStrategy class can be imported."""
        from segmentation.detection.strategies.nmj import NMJStrategy
        assert NMJStrategy is not None
        # Verify it has required methods
        assert hasattr(NMJStrategy, 'segment')
        assert hasattr(NMJStrategy, 'detect')
        assert hasattr(NMJStrategy, 'filter')
        assert hasattr(NMJStrategy, 'classify')

    def test_nmj_classifier_loaders(self):
        """Test that classifier loading functions can be imported."""
        from segmentation.detection.strategies.nmj import (
            load_nmj_classifier,
            load_nmj_rf_classifier,
            load_classifier,
        )
        assert callable(load_nmj_classifier)
        assert callable(load_nmj_rf_classifier)
        assert callable(load_classifier)

    def test_nmj_simple_detection_function(self):
        """Test that simple detection function can be imported."""
        from segmentation.detection.strategies.nmj import detect_nmjs_simple
        assert callable(detect_nmjs_simple)


class TestStrategyRegistryImports:
    """Test that NMJ strategy is registered in the registry."""

    def test_registry_import(self):
        """Test that StrategyRegistry can be imported."""
        from segmentation.detection.registry import StrategyRegistry
        assert StrategyRegistry is not None

    def test_nmj_strategy_registered(self):
        """Test that NMJ strategy is registered in the registry."""
        from segmentation.detection.registry import StrategyRegistry
        strategies = StrategyRegistry.list_strategies()
        assert 'nmj' in strategies

    def test_nmj_strategy_creation(self):
        """Test that NMJ strategy can be created from registry."""
        from segmentation.detection.registry import StrategyRegistry
        strategy = StrategyRegistry.create('nmj')
        assert strategy is not None
        assert strategy.name == 'nmj'


class TestMultiChannelMixinImports:
    """Test imports from multi-channel feature mixin."""

    def test_mixin_import(self):
        """Test that MultiChannelFeatureMixin can be imported."""
        from segmentation.detection.strategies.mixins import MultiChannelFeatureMixin
        assert MultiChannelFeatureMixin is not None

    def test_mixin_methods_exist(self):
        """Test that mixin has expected methods."""
        from segmentation.detection.strategies.mixins import MultiChannelFeatureMixin
        assert hasattr(MultiChannelFeatureMixin, 'extract_channel_stats')
        assert hasattr(MultiChannelFeatureMixin, 'extract_multichannel_features')
        assert hasattr(MultiChannelFeatureMixin, 'extract_channel_intensity_simple')


class TestHTMLExportImports:
    """Test imports from HTML export module for NMJ."""

    def test_export_samples_to_html(self):
        """Test that export_samples_to_html can be imported."""
        from segmentation.io.html_export import export_samples_to_html
        assert callable(export_samples_to_html)

    def test_utility_functions(self):
        """Test that utility functions can be imported."""
        from segmentation.io.html_export import (
            percentile_normalize,
            draw_mask_contour,
            image_to_base64,
        )
        assert callable(percentile_normalize)
        assert callable(draw_mask_contour)
        assert callable(image_to_base64)


class TestEntryPointScriptImports:
    """Test that entry point scripts can be imported."""

    def test_run_nmj_segmentation_import(self):
        """Test that run_nmj_segmentation module can be imported."""
        import run_nmj_segmentation
        assert run_nmj_segmentation is not None
        # Check main functions exist
        assert hasattr(run_nmj_segmentation, 'detect_nmjs')
        assert hasattr(run_nmj_segmentation, 'process_czi_for_nmj')
        assert hasattr(run_nmj_segmentation, 'main')

    def test_run_segmentation_import(self):
        """Test that run_segmentation module can be imported."""
        import run_segmentation
        assert run_segmentation is not None
        # Check main function exists
        assert hasattr(run_segmentation, 'main')


class TestFeatureExtractionImports:
    """Test imports for feature extraction used by NMJ."""

    def test_morphological_features(self):
        """Test that morphological feature extraction can be imported."""
        from segmentation.utils.feature_extraction import extract_morphological_features
        assert callable(extract_morphological_features)

    def test_feature_dimension_constants(self):
        """Test that feature dimension constants can be imported."""
        from segmentation.utils.feature_extraction import (
            SAM2_EMBEDDING_DIM,
            RESNET50_FEATURE_DIM,
        )
        assert SAM2_EMBEDDING_DIM == 256
        assert RESNET50_FEATURE_DIM == 2048


class TestCZILoaderImports:
    """Test CZI loader imports used by NMJ pipeline."""

    def test_czi_loader_import(self):
        """Test that CZILoader can be imported."""
        from segmentation.io.czi_loader import CZILoader, get_loader
        assert CZILoader is not None
        assert callable(get_loader)


def run_tests_standalone():
    """Run tests without pytest."""
    import sys

    test_classes = [
        TestNMJStrategyImports,
        TestStrategyRegistryImports,
        TestMultiChannelMixinImports,
        TestHTMLExportImports,
        TestEntryPointScriptImports,
        TestFeatureExtractionImports,
        TestCZILoaderImports,
    ]

    print("Testing NMJ Pipeline Imports...")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    print(f"  {test_class.__name__}.{method_name}...", end=" ")
                    getattr(instance, method_name)()
                    print("OK")
                    passed += 1
                except Exception as e:
                    print(f"FAILED: {e}")
                    failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        import sys
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        import sys
        sys.exit(run_tests_standalone())
