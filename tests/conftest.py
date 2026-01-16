"""
Pytest fixtures for xldvp_seg_repo tests.

Provides shared fixtures for creating sample data, mocks, and temporary directories.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock


@pytest.fixture
def sample_tile():
    """
    Create a simple 512x512 RGB numpy array with some shapes.

    Contains:
    - A white circle in the center
    - A white rectangle in the top-left
    - Background is black

    Returns:
        np.ndarray: 512x512x3 uint8 array
    """
    tile = np.zeros((512, 512, 3), dtype=np.uint8)

    # Draw a circle in the center (radius 50)
    center_y, center_x = 256, 256
    radius = 50
    y_indices, x_indices = np.ogrid[:512, :512]
    circle_mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    tile[circle_mask] = [255, 255, 255]

    # Draw a rectangle in the top-left corner
    tile[50:100, 50:150] = [200, 200, 200]

    return tile


@pytest.fixture
def sample_mask(sample_tile):
    """
    Binary mask derived from the sample tile.

    Creates a mask from the white circle in the center of sample_tile.

    Returns:
        np.ndarray: 512x512 boolean array
    """
    # Create mask from the circle region
    mask = np.zeros((512, 512), dtype=bool)
    center_y, center_x = 256, 256
    radius = 50
    y_indices, x_indices = np.ogrid[:512, :512]
    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    return mask


@pytest.fixture
def empty_mask():
    """
    Empty binary mask for testing edge cases.

    Returns:
        np.ndarray: 512x512 boolean array of all False
    """
    return np.zeros((512, 512), dtype=bool)


@pytest.fixture
def simple_rectangular_mask():
    """
    Simple rectangular mask for predictable area calculations.

    Creates a 100x100 pixel square mask.

    Returns:
        np.ndarray: 512x512 boolean array with 100x100 True region
    """
    mask = np.zeros((512, 512), dtype=bool)
    mask[100:200, 100:200] = True
    return mask


@pytest.fixture
def mock_loader(sample_tile):
    """
    Mock CZI loader that returns sample tiles.

    Provides a mock object simulating CZI loader behavior with:
    - read_tile() method returning sample_tile
    - mosaic_dims property returning (512, 512)
    - pixel_size_um property returning 0.22

    Returns:
        MagicMock: Mock CZI loader object
    """
    loader = MagicMock()
    loader.read_tile.return_value = sample_tile
    loader.mosaic_dims = (512, 512)
    loader.pixel_size_um = 0.22
    loader.get_tile_origin.return_value = (0, 0)

    # Mock context manager behavior
    loader.__enter__ = MagicMock(return_value=loader)
    loader.__exit__ = MagicMock(return_value=False)

    return loader


@pytest.fixture
def temp_output_dir():
    """
    Temporary directory for test outputs.

    Creates a temporary directory that is automatically cleaned up after the test.

    Yields:
        Path: Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="bm_mk_seg_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_tile_grayscale():
    """
    Create a simple 512x512 grayscale numpy array with shapes.

    Returns:
        np.ndarray: 512x512 uint8 array
    """
    tile = np.zeros((512, 512), dtype=np.uint8)

    # Draw a circle in the center
    center_y, center_x = 256, 256
    radius = 50
    y_indices, x_indices = np.ogrid[:512, :512]
    circle_mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    tile[circle_mask] = 200

    return tile


@pytest.fixture
def sample_tile_uint16():
    """
    Create a 16-bit sample tile (common for CZI images).

    Returns:
        np.ndarray: 512x512x3 uint16 array
    """
    tile = np.zeros((512, 512, 3), dtype=np.uint16)

    # Draw a circle with high 16-bit values
    center_y, center_x = 256, 256
    radius = 50
    y_indices, x_indices = np.ogrid[:512, :512]
    circle_mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
    tile[circle_mask] = [50000, 50000, 50000]

    return tile


@pytest.fixture
def mock_regionprop():
    """
    Mock scikit-image regionprops object.

    Provides a mock with centroid at (row=100, col=200) = (y=100, x=200).

    Returns:
        MagicMock: Mock regionprops object
    """
    prop = MagicMock()
    # Regionprops returns centroid as (row, col) = (y, x)
    prop.centroid = (100.5, 200.5)  # (y, x) in regionprops convention
    prop.area = 7854  # Approximate area of circle with radius 50
    prop.eccentricity = 0.0
    prop.solidity = 1.0
    prop.mean_intensity = 200.0
    prop.perimeter = 314.0
    return prop
