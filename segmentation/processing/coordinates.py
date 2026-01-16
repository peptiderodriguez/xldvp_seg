"""
Coordinate handling utilities for the segmentation pipeline.

Convention: All stored coordinates are [x, y] (horizontal, vertical).

- Scikit-image regionprops returns centroids as (row, col) = (y, x)
- NumPy arrays are indexed as [row, col] = [y, x]
- Our storage format is [x, y] for consistency with image coordinates

This module provides helpers to reduce coordinate conversion errors.
"""

import numpy as np
from typing import Tuple, List, Union


def regionprop_centroid_to_xy(prop) -> List[float]:
    """
    Convert scikit-image regionprops centroid to [x, y] format.

    Args:
        prop: A regionprops object with .centroid attribute

    Returns:
        [x, y] as list of floats
    """
    # prop.centroid is (row, col) = (y, x)
    return [float(prop.centroid[1]), float(prop.centroid[0])]


def xy_to_array_index(x: float, y: float) -> Tuple[int, int]:
    """
    Convert [x, y] coordinates to numpy array indices [row, col].

    Args:
        x: Horizontal coordinate
        y: Vertical coordinate

    Returns:
        (row, col) tuple for array indexing
    """
    return (int(y), int(x))


def array_index_to_xy(row: int, col: int) -> Tuple[int, int]:
    """
    Convert numpy array indices [row, col] to [x, y] coordinates.

    Args:
        row: Row index (y)
        col: Column index (x)

    Returns:
        (x, y) tuple
    """
    return (col, row)


def extract_crop_bounds(
    center_x: float,
    center_y: float,
    crop_size: int,
    image_width: int,
    image_height: int
) -> Tuple[int, int, int, int]:
    """
    Calculate crop bounds centered on a point, clipped to image boundaries.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        crop_size: Desired crop size (square)
        image_width: Image width for clipping
        image_height: Image height for clipping

    Returns:
        (x1, y1, x2, y2) bounds for cropping
    """
    half = crop_size // 2

    x1 = max(0, int(center_x - half))
    x2 = min(image_width, int(center_x + half))
    y1 = max(0, int(center_y - half))
    y2 = min(image_height, int(center_y + half))

    return (x1, y1, x2, y2)


def extract_crop(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: int,
    copy: bool = True
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a square crop centered on a point.

    Args:
        image: Source image (2D or 3D array)
        center_x: X coordinate of center
        center_y: Y coordinate of center
        crop_size: Desired crop size (square)
        copy: Whether to copy the data (default True to avoid modifying source)

    Returns:
        (crop, (x1, y1, x2, y2)) - the crop and its bounds
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = extract_crop_bounds(center_x, center_y, crop_size, w, h)

    # NumPy indexing is [row, col] = [y, x]
    crop = image[y1:y2, x1:x2]

    if copy:
        crop = crop.copy()

    return crop, (x1, y1, x2, y2)


def global_to_tile_coords(
    global_x: float,
    global_y: float,
    tile_origin_x: int,
    tile_origin_y: int
) -> Tuple[float, float]:
    """
    Convert global mosaic coordinates to tile-relative coordinates.

    Args:
        global_x: Global X coordinate
        global_y: Global Y coordinate
        tile_origin_x: Tile's X origin in global coords
        tile_origin_y: Tile's Y origin in global coords

    Returns:
        (local_x, local_y) coordinates relative to tile
    """
    return (global_x - tile_origin_x, global_y - tile_origin_y)


def tile_to_global_coords(
    local_x: float,
    local_y: float,
    tile_origin_x: int,
    tile_origin_y: int
) -> Tuple[float, float]:
    """
    Convert tile-relative coordinates to global mosaic coordinates.

    Args:
        local_x: X coordinate relative to tile
        local_y: Y coordinate relative to tile
        tile_origin_x: Tile's X origin in global coords
        tile_origin_y: Tile's Y origin in global coords

    Returns:
        (global_x, global_y) coordinates in mosaic space
    """
    return (local_x + tile_origin_x, local_y + tile_origin_y)


def generate_uid(
    slide_name: str,
    cell_type: str,
    global_x: float,
    global_y: float
) -> str:
    """
    Generate a unique identifier for a detection.

    Format: {slide_name}_{cell_type}_{round(x)}_{round(y)}

    Args:
        slide_name: Name of the slide
        cell_type: Type of cell (mk, hspc, nmj, vessel)
        global_x: Global X coordinate
        global_y: Global Y coordinate

    Returns:
        Unique identifier string
    """
    return f"{slide_name}_{cell_type}_{round(global_x)}_{round(global_y)}"


def mask_to_crop_coords(
    mask_ys: np.ndarray,
    mask_xs: np.ndarray,
    tile_origin_y: int,
    tile_origin_x: int,
    crop_y1: int,
    crop_x1: int,
    crop_height: int,
    crop_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map mask pixel coordinates from tile space to crop space (vectorized).

    Args:
        mask_ys: Y coordinates of mask pixels (in tile space)
        mask_xs: X coordinates of mask pixels (in tile space)
        tile_origin_y: Tile Y origin in global space
        tile_origin_x: Tile X origin in global space
        crop_y1: Crop start Y in global space
        crop_x1: Crop start X in global space
        crop_height: Crop height
        crop_width: Crop width

    Returns:
        (valid_crop_ys, valid_crop_xs) - coordinates within crop bounds
    """
    # Convert to global coords
    global_ys = mask_ys + tile_origin_y
    global_xs = mask_xs + tile_origin_x

    # Convert to crop coords
    crop_ys = global_ys - crop_y1
    crop_xs = global_xs - crop_x1

    # Bounds check
    valid = (
        (crop_ys >= 0) & (crop_ys < crop_height) &
        (crop_xs >= 0) & (crop_xs < crop_width)
    )

    return crop_ys[valid].astype(int), crop_xs[valid].astype(int)
