"""
Coordinate handling utilities for the segmentation pipeline.

Convention: All stored coordinates are [x, y] (horizontal, vertical).

Coordinate System:
    - Origin: Top-left corner (0, 0)
    - X-axis: Horizontal, increases to the right (columns)
    - Y-axis: Vertical, increases downward (rows)

Key Conversions:
    - Scikit-image regionprops returns centroids as (row, col) = (y, x)
    - NumPy arrays are indexed as [row, col] = [y, x]
    - Our storage format is [x, y] for consistency with image coordinates

UID Format:
    All cell types use spatial UIDs: {slide}_{celltype}_{round(x)}_{round(y)}
    Example: "2025_11_18_FGC1_mk_12346_67890"

See docs/COORDINATE_SYSTEM.md for full specification.
"""

import re
import numpy as np
from typing import Tuple, List, Union, Optional, Dict, Any


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
    return f"{slide_name}_{cell_type}_{int(round(global_x))}_{int(round(global_y))}"


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


# =============================================================================
# UID Parsing and Migration
# =============================================================================

def parse_uid(uid: str) -> Dict[str, Any]:
    """
    Parse a spatial UID into its components.

    Supports both the canonical spatial format and legacy global_id format.

    Args:
        uid: UID string in format "{slide}_{celltype}_{x}_{y}" or "{slide}_{celltype}_{global_id}"

    Returns:
        Dict with keys:
        - 'slide_name': Slide identifier
        - 'cell_type': Cell type (mk, hspc, nmj, vessel)
        - 'global_x': X coordinate (int) or None for legacy format
        - 'global_y': Y coordinate (int) or None for legacy format
        - 'global_id': Legacy global_id (int) or None for spatial format
        - 'is_spatial': True if spatial UID format, False if legacy

    Examples:
        >>> parse_uid("slide_01_mk_12345_67890")
        {'slide_name': 'slide_01', 'cell_type': 'mk', 'global_x': 12345, 'global_y': 67890, 'global_id': None, 'is_spatial': True}

        >>> parse_uid("slide_01_mk_123")
        {'slide_name': 'slide_01', 'cell_type': 'mk', 'global_x': None, 'global_y': None, 'global_id': 123, 'is_spatial': False}
    """
    # Known cell types
    cell_types = ['mk', 'hspc', 'nmj', 'vessel', 'cell', 'mesothelium', 'islet', 'tissue_pattern']

    # Try to find the cell type in the UID
    cell_type = None
    cell_type_pos = -1

    for ct in cell_types:
        # Look for _{celltype}_ pattern
        pattern = f'_{ct}_'
        pos = uid.rfind(pattern)
        if pos != -1:
            cell_type = ct
            cell_type_pos = pos
            break

    if cell_type is None:
        raise ValueError(f"Could not parse cell type from UID: {uid}")

    # Extract slide name (everything before _{celltype}_)
    slide_name = uid[:cell_type_pos]

    # Extract the suffix after the cell type
    suffix = uid[cell_type_pos + len(f'_{cell_type}_'):]

    # Check if spatial format (contains underscore with two numbers)
    parts = suffix.split('_')

    if len(parts) >= 2 and parts[0].lstrip('-').isdigit() and parts[1].lstrip('-').isdigit():
        # Spatial format: {x}_{y}
        global_x = int(parts[0])
        global_y = int(parts[1])
        return {
            'slide_name': slide_name,
            'cell_type': cell_type,
            'global_x': global_x,
            'global_y': global_y,
            'global_id': None,
            'is_spatial': True,
        }
    elif len(parts) == 1 and parts[0].isdigit():
        # Legacy format: {global_id}
        global_id = int(parts[0])
        return {
            'slide_name': slide_name,
            'cell_type': cell_type,
            'global_x': None,
            'global_y': None,
            'global_id': global_id,
            'is_spatial': False,
        }
    else:
        raise ValueError(f"Could not parse coordinates from UID suffix: {suffix}")


def migrate_uid_format(
    old_uid: str,
    global_x: float,
    global_y: float
) -> str:
    """
    Convert a legacy global_id UID to the spatial UID format.

    Args:
        old_uid: Legacy UID in format "{slide}_{celltype}_{global_id}"
        global_x: Global X coordinate
        global_y: Global Y coordinate

    Returns:
        New UID in spatial format "{slide}_{celltype}_{round(x)}_{round(y)}"

    Examples:
        >>> migrate_uid_format("slide_01_mk_123", 12345.6, 67890.3)
        'slide_01_mk_12346_67890'
    """
    parsed = parse_uid(old_uid)
    return generate_uid(parsed['slide_name'], parsed['cell_type'], global_x, global_y)


def is_spatial_uid(uid: str) -> bool:
    """
    Check if a UID uses the spatial format.

    Args:
        uid: UID string to check

    Returns:
        True if spatial format ({slide}_{celltype}_{x}_{y}), False otherwise

    Examples:
        >>> is_spatial_uid("slide_01_mk_12345_67890")
        True
        >>> is_spatial_uid("slide_01_mk_123")
        False
    """
    try:
        parsed = parse_uid(uid)
        return parsed['is_spatial']
    except ValueError:
        return False


# =============================================================================
# Coordinate Validation
# =============================================================================

class CoordinateValidationError(ValueError):
    """Exception raised for invalid coordinates."""
    pass


def validate_xy_coordinates(
    x: float,
    y: float,
    width: int,
    height: int,
    allow_negative: bool = False,
    context: str = ""
) -> None:
    """
    Validate that x, y coordinates are within valid bounds.

    Args:
        x: X coordinate (horizontal)
        y: Y coordinate (vertical)
        width: Image/tile width
        height: Image/tile height
        allow_negative: Whether to allow negative coordinates (default False)
        context: Optional context string for error messages

    Raises:
        CoordinateValidationError: If coordinates are out of bounds

    Examples:
        >>> validate_xy_coordinates(100, 200, 512, 512)  # OK
        >>> validate_xy_coordinates(600, 200, 512, 512)  # Raises error
    """
    ctx = f" ({context})" if context else ""

    if not allow_negative:
        if x < 0:
            raise CoordinateValidationError(f"X coordinate {x} is negative{ctx}")
        if y < 0:
            raise CoordinateValidationError(f"Y coordinate {y} is negative{ctx}")

    if x >= width:
        raise CoordinateValidationError(
            f"X coordinate {x} exceeds width {width}{ctx}"
        )
    if y >= height:
        raise CoordinateValidationError(
            f"Y coordinate {y} exceeds height {height}{ctx}"
        )


def validate_array_indices(
    row: int,
    col: int,
    height: int,
    width: int,
    context: str = ""
) -> None:
    """
    Validate that row, col indices are within valid bounds for array indexing.

    Args:
        row: Row index (y direction)
        col: Column index (x direction)
        height: Array height (number of rows)
        width: Array width (number of columns)
        context: Optional context string for error messages

    Raises:
        CoordinateValidationError: If indices are out of bounds

    Examples:
        >>> validate_array_indices(100, 200, 512, 512)  # OK
        >>> validate_array_indices(600, 200, 512, 512)  # Raises error
    """
    ctx = f" ({context})" if context else ""

    if row < 0:
        raise CoordinateValidationError(f"Row index {row} is negative{ctx}")
    if col < 0:
        raise CoordinateValidationError(f"Column index {col} is negative{ctx}")
    if row >= height:
        raise CoordinateValidationError(
            f"Row index {row} exceeds height {height}{ctx}"
        )
    if col >= width:
        raise CoordinateValidationError(
            f"Column index {col} exceeds width {width}{ctx}"
        )


def validate_bbox_xyxy(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    context: str = ""
) -> None:
    """
    Validate a bounding box in (x1, y1, x2, y2) format.

    Args:
        bbox: Tuple of (x1, y1, x2, y2)
        width: Image width
        height: Image height
        context: Optional context string for error messages

    Raises:
        CoordinateValidationError: If bbox is invalid
    """
    x1, y1, x2, y2 = bbox
    ctx = f" ({context})" if context else ""

    if x1 < 0 or y1 < 0:
        raise CoordinateValidationError(
            f"Bbox ({x1}, {y1}, {x2}, {y2}) has negative coordinates{ctx}"
        )
    if x2 > width or y2 > height:
        raise CoordinateValidationError(
            f"Bbox ({x1}, {y1}, {x2}, {y2}) exceeds bounds ({width}, {height}){ctx}"
        )
    if x1 >= x2 or y1 >= y2:
        raise CoordinateValidationError(
            f"Bbox ({x1}, {y1}, {x2}, {y2}) has invalid dimensions{ctx}"
        )


# =============================================================================
# Coordinate Labeling Helpers
# =============================================================================

def create_coordinate_dict(
    x: float,
    y: float,
    prefix: str = "",
    include_rounded: bool = False
) -> Dict[str, Any]:
    """
    Create a dictionary with explicitly labeled coordinates.

    Args:
        x: X coordinate
        y: Y coordinate
        prefix: Optional prefix for keys (e.g., "global_", "local_")
        include_rounded: Whether to include rounded integer versions

    Returns:
        Dict with labeled coordinate fields

    Examples:
        >>> create_coordinate_dict(123.4, 567.8, prefix="global_")
        {'global_x': 123.4, 'global_y': 567.8, 'global_center_xy': [123.4, 567.8]}

        >>> create_coordinate_dict(123.4, 567.8, include_rounded=True)
        {'x': 123.4, 'y': 567.8, 'center_xy': [123.4, 567.8], 'x_rounded': 123, 'y_rounded': 568}
    """
    result = {
        f'{prefix}x': x,
        f'{prefix}y': y,
        f'{prefix}center_xy': [x, y],
    }

    if include_rounded:
        result[f'{prefix}x_rounded'] = round(x)
        result[f'{prefix}y_rounded'] = round(y)

    return result


def format_coordinates_for_export(
    global_x: float,
    global_y: float,
    local_x: Optional[float] = None,
    local_y: Optional[float] = None,
    tile_origin_x: Optional[int] = None,
    tile_origin_y: Optional[int] = None,
    pixel_size_um: Optional[float] = None
) -> Dict[str, Any]:
    """
    Format coordinates for JSON/CSV export with explicit labeling.

    All coordinate fields include _xy suffix to indicate [x, y] ordering.

    Args:
        global_x: Global X coordinate in pixels
        global_y: Global Y coordinate in pixels
        local_x: Local X coordinate in tile (optional)
        local_y: Local Y coordinate in tile (optional)
        tile_origin_x: Tile origin X (optional)
        tile_origin_y: Tile origin Y (optional)
        pixel_size_um: Pixel size in microns for um conversion (optional)

    Returns:
        Dict with labeled coordinate fields

    Examples:
        >>> coords = format_coordinates_for_export(12345.6, 67890.3, pixel_size_um=0.22)
        >>> coords['global_center_xy']
        [12345.6, 67890.3]
        >>> coords['global_x_um']
        2716.032
    """
    result = {
        'global_x_px': global_x,
        'global_y_px': global_y,
        'global_center_xy': [global_x, global_y],
    }

    if local_x is not None and local_y is not None:
        result['local_x_px'] = local_x
        result['local_y_px'] = local_y
        result['local_center_xy'] = [local_x, local_y]

    if tile_origin_x is not None and tile_origin_y is not None:
        result['tile_origin_x'] = tile_origin_x
        result['tile_origin_y'] = tile_origin_y
        result['tile_origin_xy'] = [tile_origin_x, tile_origin_y]

    if pixel_size_um is not None:
        result['global_x_um'] = global_x * pixel_size_um
        result['global_y_um'] = global_y * pixel_size_um

    return result


# =============================================================================
# Batch Conversion Utilities
# =============================================================================

def convert_detections_to_spatial_uids(
    detections: List[Dict[str, Any]],
    slide_name: str,
    cell_type: str
) -> List[Dict[str, Any]]:
    """
    Convert a list of detections to use spatial UIDs.

    For detections that have a legacy global_id, this generates a new spatial UID
    based on the detection's coordinates.

    Args:
        detections: List of detection dicts with 'global_x'/'global_y' or 'center' fields
        slide_name: Slide name for UID generation
        cell_type: Cell type for UID generation

    Returns:
        Updated list of detections with 'uid' field using spatial format

    Examples:
        >>> dets = [{'global_id': 123, 'center': [1000, 2000]}]
        >>> result = convert_detections_to_spatial_uids(dets, "slide_01", "mk")
        >>> result[0]['uid']
        'slide_01_mk_1000_2000'
    """
    result = []
    uid_counts = {}  # Track UID collisions from round() on dense detections

    for det in detections:
        det_copy = det.copy()

        # Get coordinates
        if 'center' in det:
            global_x, global_y = det['center'][0], det['center'][1]
        elif 'global_x' in det and 'global_y' in det:
            global_x, global_y = det['global_x'], det['global_y']
        elif 'global_center' in det:
            global_x, global_y = det['global_center'][0], det['global_center'][1]
        elif 'centroid' in det:
            global_x, global_y = det['centroid'][0], det['centroid'][1]
        else:
            # No coordinates available, skip UID generation
            result.append(det_copy)
            continue

        # Generate spatial UID with collision detection
        base_uid = generate_uid(slide_name, cell_type, global_x, global_y)
        if base_uid in uid_counts:
            uid_counts[base_uid] += 1
            det_copy['uid'] = f"{base_uid}_{uid_counts[base_uid]}"
        else:
            uid_counts[base_uid] = 0
            det_copy['uid'] = base_uid

        # Keep legacy global_id for backwards compatibility if present
        if 'global_id' in det:
            det_copy['legacy_global_id'] = det['global_id']

        result.append(det_copy)

    return result
