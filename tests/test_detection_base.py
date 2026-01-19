"""
Tests for Detection dataclass and DetectionStrategy base class.

Tests the base detection infrastructure in segmentation/detection/strategies/base.py.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from segmentation.detection.strategies.base import Detection, DetectionStrategy


class TestDetectionDataclass:
    """Tests for the Detection dataclass."""

    def test_detection_area_property(self, sample_mask):
        """Test that area property correctly counts mask pixels."""
        detection = Detection(
            mask=sample_mask,
            centroid=[256.0, 256.0],
            features={},
        )

        # Area should be the sum of True values in mask
        expected_area = int(sample_mask.sum())
        assert detection.area == expected_area
        assert isinstance(detection.area, int)

    def test_detection_area_empty_mask(self, empty_mask):
        """Test area property with empty mask."""
        detection = Detection(
            mask=empty_mask,
            centroid=[0.0, 0.0],
            features={},
        )

        assert detection.area == 0

    def test_detection_area_rectangular_mask(self, simple_rectangular_mask):
        """Test area property with rectangular mask (100x100 = 10000 pixels)."""
        detection = Detection(
            mask=simple_rectangular_mask,
            centroid=[150.0, 150.0],
            features={},
        )

        assert detection.area == 10000

    def test_detection_to_dict(self, sample_mask):
        """Test conversion to dictionary."""
        features = {'eccentricity': 0.1, 'solidity': 0.98}
        detection = Detection(
            mask=sample_mask,
            centroid=[256.5, 256.5],
            features=features,
            id="test_id",
            score=0.95,
        )

        result = detection.to_dict()

        # Check all expected keys
        assert 'centroid' in result
        assert 'features' in result
        assert 'id' in result
        assert 'score' in result
        assert 'area' in result

        # Check values
        assert result['centroid'] == [256.5, 256.5]
        assert result['features'] == features
        assert result['id'] == "test_id"
        assert result['score'] == 0.95
        assert result['area'] == int(sample_mask.sum())

        # Mask should NOT be in dict (too large for serialization)
        assert 'mask' not in result

    def test_detection_to_dict_minimal(self, empty_mask):
        """Test to_dict with minimal/default values."""
        detection = Detection(
            mask=empty_mask,
            centroid=[0.0, 0.0],
        )

        result = detection.to_dict()

        assert result['centroid'] == [0.0, 0.0]
        assert result['features'] == {}
        assert result['id'] is None
        assert result['score'] is None
        assert result['area'] == 0

    def test_detection_default_values(self, sample_mask):
        """Test that Detection has correct default values."""
        detection = Detection(
            mask=sample_mask,
            centroid=[100.0, 200.0],
        )

        assert detection.features == {}
        assert detection.id is None
        assert detection.score is None

    def test_detection_with_all_fields(self, sample_mask):
        """Test Detection with all fields populated."""
        features = {
            'area': 7854,
            'eccentricity': 0.05,
            'solidity': 0.99,
            'mean_intensity': 200.0,
        }

        detection = Detection(
            mask=sample_mask,
            centroid=[256.0, 256.0],
            features=features,
            id="slide_mk_256_256",
            score=0.987,
        )

        assert detection.mask is sample_mask
        assert detection.centroid == [256.0, 256.0]
        assert detection.features == features
        assert detection.id == "slide_mk_256_256"
        assert detection.score == 0.987


class TestDetectionStrategyComputeFeatures:
    """Tests for DetectionStrategy.compute_features method."""

    def _create_test_strategy(self):
        """Create a concrete test strategy implementation."""
        class TestStrategy(DetectionStrategy):
            @property
            def name(self):
                return "test"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        return TestStrategy()

    def test_strategy_compute_features_returns_dict(self, sample_mask, sample_tile):
        """Test that compute_features returns a dictionary."""
        strategy = self._create_test_strategy()

        features = strategy.compute_features(sample_mask, sample_tile)

        assert isinstance(features, dict)

    def test_strategy_compute_features_empty_mask(self, empty_mask, sample_tile):
        """Test compute_features with empty mask returns empty dict."""
        strategy = self._create_test_strategy()

        features = strategy.compute_features(empty_mask, sample_tile)

        assert features == {}

    def test_strategy_compute_features_has_required_keys(self, sample_mask, sample_tile):
        """Test that compute_features returns all required feature keys."""
        strategy = self._create_test_strategy()

        features = strategy.compute_features(sample_mask, sample_tile)

        required_keys = ['area', 'centroid', 'eccentricity', 'solidity',
                         'mean_intensity', 'perimeter']

        for key in required_keys:
            assert key in features, f"Missing required key: {key}"

    def test_strategy_compute_features_centroid_format(self, sample_mask, sample_tile):
        """Test that centroid is in [x, y] format."""
        strategy = self._create_test_strategy()

        features = strategy.compute_features(sample_mask, sample_tile)

        centroid = features['centroid']
        assert isinstance(centroid, list)
        assert len(centroid) == 2
        assert all(isinstance(c, float) for c in centroid)

        # For our circular mask centered at (256, 256)
        assert abs(centroid[0] - 256) < 2  # x
        assert abs(centroid[1] - 256) < 2  # y

    def test_strategy_compute_features_area_matches_mask(self, simple_rectangular_mask, sample_tile):
        """Test that computed area matches mask.sum()."""
        strategy = self._create_test_strategy()

        features = strategy.compute_features(simple_rectangular_mask, sample_tile)

        assert features['area'] == int(simple_rectangular_mask.sum())
        assert features['area'] == 10000  # 100x100

    def test_strategy_compute_features_with_grayscale(self, sample_mask, sample_tile_grayscale):
        """Test compute_features handles grayscale tiles."""
        strategy = self._create_test_strategy()

        features = strategy.compute_features(sample_mask, sample_tile_grayscale)

        # Should still work and return features
        assert 'area' in features
        assert 'mean_intensity' in features

    def test_strategy_name_property(self):
        """Test that strategy name property works."""
        strategy = self._create_test_strategy()

        assert strategy.name == "test"

    def test_strategy_get_config(self):
        """Test that get_config returns strategy configuration."""
        class ConfiguredStrategy(DetectionStrategy):
            def __init__(self, threshold=0.5, min_area=100):
                self.threshold = threshold
                self.min_area = min_area

            @property
            def name(self):
                return "configured"

            def segment(self, tile, models):
                return []

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = ConfiguredStrategy(threshold=0.7, min_area=200)
        config = strategy.get_config()

        assert config['strategy'] == 'configured'
        assert config['threshold'] == 0.7
        assert config['min_area'] == 200


class TestDetectionStrategyDetect:
    """Tests for the complete detect() pipeline."""

    def test_strategy_detect_empty_result(self, sample_tile):
        """Test detect() with no masks returns empty list."""
        class EmptyStrategy(DetectionStrategy):
            @property
            def name(self):
                return "empty"

            def segment(self, tile, models):
                return []  # No masks

            def filter(self, masks, features, pixel_size_um):
                return []

        strategy = EmptyStrategy()
        detections = strategy.detect(sample_tile, {}, pixel_size_um=0.22)

        assert detections == []

    def test_strategy_detect_with_masks(self, sample_tile, sample_mask):
        """Test detect() pipeline with masks."""
        class SimpleStrategy(DetectionStrategy):
            def __init__(self, mask_to_return):
                self._mask = mask_to_return

            @property
            def name(self):
                return "simple"

            def segment(self, tile, models):
                return [self._mask]

            def filter(self, masks, features, pixel_size_um):
                # Accept all masks
                return [
                    Detection(
                        mask=m,
                        centroid=f['centroid'],
                        features=f,
                    )
                    for m, f in zip(masks, features)
                ]

        strategy = SimpleStrategy(sample_mask)
        detections = strategy.detect(sample_tile, {}, pixel_size_um=0.22)

        assert len(detections) == 1
        assert isinstance(detections[0], Detection)
        assert detections[0].area > 0
