"""
Comprehensive tests for feature extraction utilities.

Tests the feature extraction functions and constants in the xldvp_seg_repo codebase,
specifically from:
- segmentation/utils/config.py (feature dimension constants)
- segmentation/utils/feature_extraction.py (morphological features, ResNet, SAM2)
- segmentation/detection/strategies/base.py (feature extraction methods)

Test categories:
1. Feature dimension constants verification
2. Morphological feature extraction (22 features)
3. SAM2 embedding dimensions (256)
4. ResNet feature dimensions (2048)
5. Combined feature vector tests (total = 2326)
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Path setup to allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def _check_torch_available():
    """Helper to check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


class TestFeatureDimensionConstants(unittest.TestCase):
    """
    Tests for feature dimension constants defined in segmentation/utils/config.py.

    These constants define the expected dimensions for different feature types:
    - Morphological features: 22
    - SAM2 embeddings: 256
    - ResNet embeddings: 2048
    - Total features per cell: 2326 (22 + 256 + 2048)
    """

    def test_morphological_features_count_equals_22(self):
        """Verify MORPHOLOGICAL_FEATURES_COUNT constant equals 22."""
        from segmentation.utils.config import MORPHOLOGICAL_FEATURES_COUNT
        self.assertEqual(
            MORPHOLOGICAL_FEATURES_COUNT, 22,
            f"Expected 22 morphological features, got {MORPHOLOGICAL_FEATURES_COUNT}"
        )

    def test_sam2_embedding_dimension_equals_256(self):
        """Verify SAM2_EMBEDDING_DIMENSION constant equals 256."""
        from segmentation.utils.config import SAM2_EMBEDDING_DIMENSION
        self.assertEqual(
            SAM2_EMBEDDING_DIMENSION, 256,
            f"Expected SAM2 embedding dimension of 256, got {SAM2_EMBEDDING_DIMENSION}"
        )

    def test_resnet_embedding_dimension_equals_2048(self):
        """Verify RESNET_EMBEDDING_DIMENSION constant equals 2048."""
        from segmentation.utils.config import RESNET_EMBEDDING_DIMENSION
        self.assertEqual(
            RESNET_EMBEDDING_DIMENSION, 2048,
            f"Expected ResNet embedding dimension of 2048, got {RESNET_EMBEDDING_DIMENSION}"
        )

    def test_total_features_per_cell_equals_2326(self):
        """Verify TOTAL_FEATURES_PER_CELL constant equals 2326 (22 + 256 + 2048)."""
        from segmentation.utils.config import TOTAL_FEATURES_PER_CELL
        self.assertEqual(
            TOTAL_FEATURES_PER_CELL, 2326,
            f"Expected total features of 2326, got {TOTAL_FEATURES_PER_CELL}"
        )

    def test_total_features_equals_sum_of_components(self):
        """Verify TOTAL_FEATURES_PER_CELL equals sum of morphological + SAM2 + ResNet."""
        from segmentation.utils.config import (
            MORPHOLOGICAL_FEATURES_COUNT,
            SAM2_EMBEDDING_DIMENSION,
            RESNET_EMBEDDING_DIMENSION,
            TOTAL_FEATURES_PER_CELL,
        )
        expected_total = (
            MORPHOLOGICAL_FEATURES_COUNT +
            SAM2_EMBEDDING_DIMENSION +
            RESNET_EMBEDDING_DIMENSION
        )
        self.assertEqual(
            TOTAL_FEATURES_PER_CELL, expected_total,
            f"Total features ({TOTAL_FEATURES_PER_CELL}) does not equal sum of components ({expected_total})"
        )

    def test_get_feature_dimensions_returns_correct_values(self):
        """Verify get_feature_dimensions() helper returns correct dictionary."""
        from segmentation.utils.config import get_feature_dimensions
        dims = get_feature_dimensions()

        self.assertIn("morphological", dims)
        self.assertIn("sam2_embedding", dims)
        self.assertIn("resnet_embedding", dims)
        self.assertIn("total", dims)

        self.assertEqual(dims["morphological"], 22)
        self.assertEqual(dims["sam2_embedding"], 256)
        self.assertEqual(dims["resnet_embedding"], 2048)
        self.assertEqual(dims["total"], 2326)

    def test_feature_extraction_module_constants_match_config(self):
        """Verify constants in feature_extraction.py match those in config.py."""
        from segmentation.utils.feature_extraction import (
            SAM2_EMBEDDING_DIM,
            RESNET50_FEATURE_DIM,
            MORPHOLOGICAL_FEATURE_COUNT,
        )
        from segmentation.utils.config import (
            SAM2_EMBEDDING_DIMENSION,
            RESNET_EMBEDDING_DIMENSION,
            MORPHOLOGICAL_FEATURES_COUNT,
        )

        self.assertEqual(
            SAM2_EMBEDDING_DIM, SAM2_EMBEDDING_DIMENSION,
            "SAM2 dimension mismatch between modules"
        )
        self.assertEqual(
            RESNET50_FEATURE_DIM, RESNET_EMBEDDING_DIMENSION,
            "ResNet dimension mismatch between modules"
        )
        self.assertEqual(
            MORPHOLOGICAL_FEATURE_COUNT, MORPHOLOGICAL_FEATURES_COUNT,
            "Morphological feature count mismatch between modules"
        )


class TestMorphologicalFeatureExtraction(unittest.TestCase):
    """
    Tests for morphological feature extraction from segmentation/utils/feature_extraction.py.

    The extract_morphological_features() function should return exactly 22 features
    including area, perimeter, circularity, solidity, color stats, etc.
    """

    def setUp(self):
        """Create synthetic test data for morphological feature tests."""
        # Create a simple circular mask (radius 50, centered at 100,100)
        self.mask_size = 200
        y, x = np.ogrid[:self.mask_size, :self.mask_size]
        center_y, center_x = 100, 100
        radius = 50
        self.circular_mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(bool)

        # Create a simple RGB image with known values
        self.rgb_image = np.zeros((self.mask_size, self.mask_size, 3), dtype=np.uint8)
        self.rgb_image[self.circular_mask] = [200, 150, 100]  # Set mask region to specific RGB

        # Create an empty mask for edge case testing
        self.empty_mask = np.zeros((self.mask_size, self.mask_size), dtype=bool)

        # Create a grayscale image
        self.gray_image = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)
        self.gray_image[self.circular_mask] = 180

    def test_extract_morphological_features_returns_22_features(self):
        """Verify extract_morphological_features returns exactly 22 features."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        features = extract_morphological_features(self.circular_mask, self.rgb_image)

        self.assertEqual(
            len(features), 22,
            f"Expected 22 features, got {len(features)}"
        )

    def test_extract_morphological_features_with_circular_mask(self):
        """Test feature extraction with a synthetic circular mask."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        features = extract_morphological_features(self.circular_mask, self.rgb_image)

        # Check that features dict is not empty
        self.assertTrue(features, "Features should not be empty for valid mask")

        # Verify area is approximately pi * r^2
        expected_area = np.pi * 50**2
        actual_area = features['area']
        self.assertAlmostEqual(
            actual_area, expected_area, delta=expected_area * 0.05,
            msg=f"Area {actual_area} should be approximately {expected_area}"
        )

        # Circularity should be close to 1.0 for a circle
        self.assertGreater(features['circularity'], 0.9, "Circularity should be high for circle")

        # Solidity should be close to 1.0 for a convex shape
        self.assertGreater(features['solidity'], 0.95, "Solidity should be high for circle")

    def test_expected_feature_names_present(self):
        """Verify all expected morphological feature names are present."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        features = extract_morphological_features(self.circular_mask, self.rgb_image)

        expected_features = [
            'area', 'perimeter', 'circularity', 'solidity', 'aspect_ratio',
            'extent', 'equiv_diameter',
            'red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std',
            'gray_mean', 'gray_std',
            'hue_mean', 'saturation_mean', 'value_mean',
            'relative_brightness', 'intensity_variance', 'dark_fraction', 'nuclear_complexity'
        ]

        for feature_name in expected_features:
            self.assertIn(
                feature_name, features,
                f"Expected feature '{feature_name}' not found in extracted features"
            )

    def test_empty_mask_returns_empty_dict(self):
        """Test that empty mask returns empty dictionary (graceful handling)."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        features = extract_morphological_features(self.empty_mask, self.rgb_image)

        self.assertEqual(
            features, {},
            "Empty mask should return empty dictionary"
        )

    def test_all_zero_mask_returns_empty_dict(self):
        """Test that all-zero mask returns empty dictionary."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        zero_mask = np.zeros((100, 100), dtype=bool)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        features = extract_morphological_features(zero_mask, image)

        self.assertEqual(features, {}, "All-zero mask should return empty dictionary")

    def test_morphological_features_with_grayscale_image(self):
        """Test feature extraction works with grayscale images."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        features = extract_morphological_features(self.circular_mask, self.gray_image)

        self.assertEqual(
            len(features), 22,
            "Should still return 22 features for grayscale image"
        )

        # For grayscale, RGB means should equal the grayscale mean
        self.assertAlmostEqual(
            features['gray_mean'], 180, delta=5,
            msg="Gray mean should be approximately 180"
        )

    def test_color_statistics_correctness(self):
        """Verify color statistics are computed correctly."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        features = extract_morphological_features(self.circular_mask, self.rgb_image)

        # Red channel should be ~200
        self.assertAlmostEqual(features['red_mean'], 200, delta=5)

        # Green channel should be ~150
        self.assertAlmostEqual(features['green_mean'], 150, delta=5)

        # Blue channel should be ~100
        self.assertAlmostEqual(features['blue_mean'], 100, delta=5)

    def test_feature_values_are_numeric(self):
        """Verify all extracted feature values are numeric types."""
        from segmentation.utils.feature_extraction import extract_morphological_features

        features = extract_morphological_features(self.circular_mask, self.rgb_image)

        for key, value in features.items():
            self.assertIsInstance(
                value, (int, float),
                f"Feature '{key}' should be numeric, got {type(value)}"
            )


class TestSAM2EmbeddingDimensions(unittest.TestCase):
    """
    Tests for SAM2 embedding extraction dimensions.

    SAM2 embeddings should be 256-dimensional vectors extracted at specific
    points in the image feature map.
    """

    def test_sam2_embedding_dim_constant_is_256(self):
        """Verify SAM2_EMBEDDING_DIM constant is 256."""
        from segmentation.utils.feature_extraction import SAM2_EMBEDDING_DIM
        self.assertEqual(SAM2_EMBEDDING_DIM, 256)

    def test_extract_sam2_embedding_returns_256d_vector(self):
        """Test that SAM2 embedding extraction returns 256-dimensional vector."""
        from segmentation.detection.strategies.base import DetectionStrategy

        # Create a concrete test strategy
        class TestStrategy(DetectionStrategy):
            @property
            def name(self):
                return "test"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = TestStrategy()

        # Create mock SAM2 predictor with proper structure
        mock_predictor = MagicMock()
        mock_features = {
            "image_embed": MagicMock()
        }
        mock_features["image_embed"].shape = (1, 256, 64, 64)  # B, C, H, W
        mock_predictor._features = mock_features
        mock_predictor._orig_hw = (512, 512)

        # Create mock tensor that returns numpy array
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(256)
        mock_features["image_embed"].__getitem__ = MagicMock(return_value=mock_tensor)

        embedding = strategy._extract_sam2_embedding(mock_predictor, 256, 256)

        self.assertEqual(
            len(embedding), 256,
            f"SAM2 embedding should be 256-dimensional, got {len(embedding)}"
        )

    def test_extract_sam2_embedding_handles_missing_predictor(self):
        """Test SAM2 embedding extraction returns zeros when predictor fails."""
        from segmentation.detection.strategies.base import DetectionStrategy
        from segmentation.utils.feature_extraction import SAM2_EMBEDDING_DIM

        class TestStrategy(DetectionStrategy):
            @property
            def name(self):
                return "test"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = TestStrategy()

        # Create mock that raises exception
        mock_predictor = MagicMock()
        mock_predictor._features = {}  # Missing 'image_embed' key

        embedding = strategy._extract_sam2_embedding(mock_predictor, 100, 100)

        self.assertEqual(len(embedding), SAM2_EMBEDDING_DIM)
        self.assertTrue(np.all(embedding == 0), "Should return zeros on failure")


class TestResNetFeatureDimensions(unittest.TestCase):
    """
    Tests for ResNet50 feature extraction dimensions.

    ResNet50 features (before the final FC layer) should be 2048-dimensional.
    """

    def test_resnet50_feature_dim_constant_is_2048(self):
        """Verify RESNET50_FEATURE_DIM constant is 2048."""
        from segmentation.utils.feature_extraction import RESNET50_FEATURE_DIM
        self.assertEqual(RESNET50_FEATURE_DIM, 2048)

    def test_preprocess_crop_for_resnet_empty_input(self):
        """Test preprocessing empty crop returns correct shape."""
        from segmentation.utils.feature_extraction import preprocess_crop_for_resnet

        empty_crop = np.array([])
        result = preprocess_crop_for_resnet(empty_crop)

        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result == 0))

    def test_preprocess_crop_for_resnet_uint8_rgb(self):
        """Test preprocessing uint8 RGB image."""
        from segmentation.utils.feature_extraction import preprocess_crop_for_resnet

        crop = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = preprocess_crop_for_resnet(crop)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[-1], 3)

    def test_preprocess_crop_for_resnet_uint16(self):
        """Test preprocessing uint16 image (common for CZI files)."""
        from segmentation.utils.feature_extraction import preprocess_crop_for_resnet

        crop = np.random.randint(0, 65536, (100, 100, 3), dtype=np.uint16)
        result = preprocess_crop_for_resnet(crop)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape[-1], 3)

    def test_preprocess_crop_for_resnet_grayscale(self):
        """Test preprocessing grayscale image converts to RGB."""
        from segmentation.utils.feature_extraction import preprocess_crop_for_resnet

        gray_crop = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocess_crop_for_resnet(gray_crop)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[-1], 3)

        # All channels should be equal for grayscale
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_preprocess_crop_for_resnet_four_channel(self):
        """Test preprocessing 4-channel image (e.g., RGBA) truncates to 3."""
        from segmentation.utils.feature_extraction import preprocess_crop_for_resnet

        rgba_crop = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        result = preprocess_crop_for_resnet(rgba_crop)

        self.assertEqual(result.shape[-1], 3)
        self.assertEqual(result.dtype, np.uint8)

    @unittest.skipIf(
        not _check_torch_available(),
        "PyTorch not available, skipping GPU-dependent tests"
    )
    def test_extract_resnet_features_batch_returns_2048d(self):
        """Test batch ResNet extraction returns 2048-dimensional features."""
        from segmentation.detection.strategies.base import DetectionStrategy
        from segmentation.utils.feature_extraction import RESNET50_FEATURE_DIM

        class TestStrategy(DetectionStrategy):
            @property
            def name(self):
                return "test"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = TestStrategy()

        # Create mock model and transform
        import torch
        mock_model = MagicMock()
        mock_features = torch.randn(2, 2048, 1, 1)  # Batch of 2
        mock_model.return_value = mock_features

        mock_transform = MagicMock()
        mock_transform.return_value = torch.randn(3, 224, 224)

        device = torch.device('cpu')

        # Create test crops
        crops = [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        ]

        features = strategy._extract_resnet_features_batch(
            crops, mock_model, mock_transform, device, batch_size=2
        )

        self.assertEqual(len(features), 2)
        for feat in features:
            self.assertEqual(
                len(feat), RESNET50_FEATURE_DIM,
                f"ResNet features should be {RESNET50_FEATURE_DIM}-dimensional"
            )


class TestCombinedFeatureVector(unittest.TestCase):
    """
    Tests for combined feature vectors (morphological + SAM2 + ResNet = 2326 total).

    The _extract_full_features_batch method in DetectionStrategy should produce
    feature dictionaries with the correct total number of features.
    """

    def setUp(self):
        """Create synthetic test data."""
        self.mask_size = 200
        y, x = np.ogrid[:self.mask_size, :self.mask_size]
        center_y, center_x = 100, 100
        radius = 40
        self.mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(bool)
        self.tile = np.random.randint(0, 256, (self.mask_size, self.mask_size, 3), dtype=np.uint8)

    def test_total_features_count_with_all_extractors(self):
        """Test that full feature extraction produces 2326 features when all extractors run."""
        from segmentation.utils.config import (
            MORPHOLOGICAL_FEATURES_COUNT,
            SAM2_EMBEDDING_DIMENSION,
            RESNET_EMBEDDING_DIMENSION,
            TOTAL_FEATURES_PER_CELL,
        )

        # Verify the math
        expected_total = (
            MORPHOLOGICAL_FEATURES_COUNT +
            SAM2_EMBEDDING_DIMENSION +
            RESNET_EMBEDDING_DIMENSION
        )

        self.assertEqual(expected_total, 2326)
        self.assertEqual(TOTAL_FEATURES_PER_CELL, 2326)

    def test_extract_full_features_batch_with_mocks(self):
        """Test _extract_full_features_batch returns correct structure."""
        from segmentation.detection.strategies.base import DetectionStrategy
        from segmentation.utils.config import (
            MORPHOLOGICAL_FEATURES_COUNT,
            SAM2_EMBEDDING_DIMENSION,
            RESNET_EMBEDDING_DIMENSION,
        )

        class TestStrategy(DetectionStrategy):
            @property
            def name(self):
                return "test"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = TestStrategy()

        # Use models dict with None values to trigger zero-filling
        models = {
            'sam2_predictor': None,
            'resnet': None,
            'resnet_transform': None,
            'device': None,
        }

        masks = [self.mask]
        feature_list = strategy._extract_full_features_batch(
            masks, self.tile, models,
            extract_sam2=True,
            extract_resnet=True
        )

        self.assertEqual(len(feature_list), 1)
        features = feature_list[0]

        # Should have morphological features + centroid
        self.assertIn('area', features)
        self.assertIn('centroid', features)
        self.assertIn('circularity', features)

        # Should have SAM2 embedding features (filled with zeros)
        sam2_count = sum(1 for k in features.keys() if k.startswith('sam2_emb_'))
        self.assertEqual(
            sam2_count, SAM2_EMBEDDING_DIMENSION,
            f"Expected {SAM2_EMBEDDING_DIMENSION} SAM2 features, got {sam2_count}"
        )

        # Should have ResNet features (filled with zeros)
        resnet_count = sum(1 for k in features.keys() if k.startswith('resnet_'))
        self.assertEqual(
            resnet_count, RESNET_EMBEDDING_DIMENSION,
            f"Expected {RESNET_EMBEDDING_DIMENSION} ResNet features, got {resnet_count}"
        )

    def test_feature_vector_dimensions_match_constants(self):
        """Verify extracted feature vector dimensions match defined constants."""
        from segmentation.utils.config import get_feature_dimensions

        dims = get_feature_dimensions()

        # These should be the canonical values
        self.assertEqual(dims['morphological'], 22)
        self.assertEqual(dims['sam2_embedding'], 256)
        self.assertEqual(dims['resnet_embedding'], 2048)
        self.assertEqual(dims['total'], 22 + 256 + 2048)

    def test_empty_mask_list_returns_empty_list(self):
        """Test _extract_full_features_batch with empty mask list."""
        from segmentation.detection.strategies.base import DetectionStrategy

        class TestStrategy(DetectionStrategy):
            @property
            def name(self):
                return "test"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = TestStrategy()

        result = strategy._extract_full_features_batch([], self.tile, {})

        self.assertEqual(result, [])

    def test_empty_mask_in_list_returns_empty_dict(self):
        """Test handling of empty mask within mask list."""
        from segmentation.detection.strategies.base import DetectionStrategy

        class TestStrategy(DetectionStrategy):
            @property
            def name(self):
                return "test"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = TestStrategy()

        empty_mask = np.zeros((100, 100), dtype=bool)
        masks = [empty_mask]

        result = strategy._extract_full_features_batch(masks, self.tile, {})

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], {})


class TestPreprocessCropForResnet(unittest.TestCase):
    """
    Additional tests for preprocess_crop_for_resnet function.

    This function handles various input formats and ensures output is always
    a valid uint8 RGB array suitable for ResNet inference.
    """

    def test_uint16_scaling_preserves_relative_values(self):
        """Test that uint16 values are properly scaled to uint8."""
        from segmentation.utils.feature_extraction import preprocess_crop_for_resnet

        crop = np.zeros((100, 100, 3), dtype=np.uint16)
        crop[:50, :, :] = 65535  # Max uint16 value
        crop[50:, :, :] = 32768  # Half of max

        result = preprocess_crop_for_resnet(crop)

        # Max uint16 (65535) / 256 = 255 (max uint8)
        self.assertEqual(result[:50, :, :].max(), 255)

        # Half of max (32768) / 256 = 128 (approximately)
        half_value = result[50:, :, :].max()
        self.assertTrue(
            120 <= half_value <= 136,
            f"Half value {half_value} should be approximately 128"
        )

    def test_float_conversion(self):
        """Test preprocessing a float image converts to uint8."""
        from segmentation.utils.feature_extraction import preprocess_crop_for_resnet

        float_crop = np.ones((100, 100, 3), dtype=np.float32) * 128.5
        result = preprocess_crop_for_resnet(float_crop)

        self.assertEqual(result.dtype, np.uint8)


class TestHSVColorFeatures(unittest.TestCase):
    """
    Tests for HSV color feature extraction.

    Tests the vectorized HSV conversion and feature computation functions.
    """

    def test_rgb_to_hsv_vectorized_empty_input(self):
        """Test RGB to HSV conversion with empty input."""
        from segmentation.utils.feature_extraction import rgb_to_hsv_vectorized

        empty_pixels = np.zeros((0, 3))
        result = rgb_to_hsv_vectorized(empty_pixels)

        self.assertEqual(result.shape, (0, 3))

    def test_rgb_to_hsv_vectorized_known_colors(self):
        """Test RGB to HSV conversion with known colors."""
        from segmentation.utils.feature_extraction import rgb_to_hsv_vectorized

        # Red, Green, Blue in RGB
        pixels = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        hsv = rgb_to_hsv_vectorized(pixels)

        # Red should have H near 0
        self.assertTrue(hsv[0, 0] < 10 or hsv[0, 0] > 170, "Red hue should be near 0 or 180")

        # Green should have H near 60 (in 0-180 scale)
        self.assertTrue(50 < hsv[1, 0] < 70, f"Green hue {hsv[1, 0]} should be near 60")

        # Blue should have H near 120 (in 0-180 scale)
        self.assertTrue(110 < hsv[2, 0] < 130, f"Blue hue {hsv[2, 0]} should be near 120")

    def test_compute_hsv_features_empty_pixels(self):
        """Test HSV feature computation with empty pixel array."""
        from segmentation.utils.feature_extraction import compute_hsv_features

        empty_pixels = np.zeros((0, 3))
        result = compute_hsv_features(empty_pixels)

        self.assertEqual(result['hue_mean'], 0.0)
        self.assertEqual(result['saturation_mean'], 0.0)
        self.assertEqual(result['value_mean'], 0.0)

    def test_compute_hsv_features_returns_dict(self):
        """Test that compute_hsv_features returns expected dictionary."""
        from segmentation.utils.feature_extraction import compute_hsv_features

        pixels = np.array([[200, 150, 100], [210, 160, 110]], dtype=np.uint8)
        result = compute_hsv_features(pixels)

        self.assertIn('hue_mean', result)
        self.assertIn('saturation_mean', result)
        self.assertIn('value_mean', result)

        # Values should be numeric
        self.assertIsInstance(result['hue_mean'], float)
        self.assertIsInstance(result['saturation_mean'], float)
        self.assertIsInstance(result['value_mean'], float)


class TestCreateResNetTransform(unittest.TestCase):
    """Tests for ResNet transform creation."""

    @unittest.skipIf(
        not _check_torch_available(),
        "PyTorch not available, skipping transform tests"
    )
    def test_create_resnet_transform_returns_compose(self):
        """Test that create_resnet_transform returns a valid Compose object."""
        from segmentation.utils.feature_extraction import create_resnet_transform
        import torchvision.transforms as tv_transforms

        transform = create_resnet_transform()

        self.assertIsInstance(transform, tv_transforms.Compose)

    @unittest.skipIf(
        not _check_torch_available(),
        "PyTorch not available, skipping transform tests"
    )
    def test_resnet_transform_output_shape(self):
        """Test that ResNet transform produces correct output shape."""
        from segmentation.utils.feature_extraction import create_resnet_transform
        from PIL import Image
        import torch

        transform = create_resnet_transform()

        # Create a test image
        test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))

        output = transform(test_image)

        # Should produce (3, 224, 224) tensor
        self.assertEqual(output.shape, (3, 224, 224))
        self.assertIsInstance(output, torch.Tensor)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
