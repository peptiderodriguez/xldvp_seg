"""
MK/HSPC-specific utilities for tile processing and worker functions.

This module provides utility functions shared by tile worker functions,
reducing code duplication while maintaining backward compatibility.

Functions:
    ensure_rgb_array: Convert image to RGB format
    check_tile_validity: Check if tile has valid content
    prepare_tile_for_detection: Normalize tile for detection
    build_mk_hspc_result: Build result dict for MK/HSPC workers

Usage:
    from segmentation.processing.mk_hspc_utils import (
        ensure_rgb_array,
        check_tile_validity,
        prepare_tile_for_detection,
        build_mk_hspc_result,
    )
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


def ensure_rgb_array(img_data: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format (H, W, 3).

    Handles common input formats:
    - Grayscale (H, W) -> Replicated to RGB (H, W, 3)
    - RGBA (H, W, 4) -> Drops alpha channel (H, W, 3)
    - RGB (H, W, 3) -> Unchanged

    Args:
        img_data: Input image array in any supported format.

    Returns:
        RGB image array with shape (H, W, 3).

    Examples:
        >>> gray = np.zeros((100, 100), dtype=np.uint8)
        >>> rgb = ensure_rgb_array(gray)
        >>> rgb.shape
        (100, 100, 3)

        >>> rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        >>> rgb = ensure_rgb_array(rgba)
        >>> rgb.shape
        (100, 100, 3)
    """
    if img_data.ndim == 2:
        # Grayscale -> RGB
        return np.stack([img_data] * 3, axis=-1)
    elif img_data.shape[2] == 1:
        # Single-channel (H, W, 1) -> RGB
        return np.stack([img_data[:, :, 0]] * 3, axis=-1)
    elif img_data.shape[2] == 4:
        # RGBA -> RGB (drop alpha)
        return img_data[:, :, :3]
    else:
        # Already RGB (or 2-channel, keep as-is)
        return img_data


def check_tile_validity(img_rgb: np.ndarray, tile_id: str) -> Tuple[bool, str]:
    """
    Check if tile has valid content for processing.

    Validates that the tile:
    - Has non-zero content (not a black tile)

    Args:
        img_rgb: RGB image array.
        tile_id: Tile identifier for logging/error messages.

    Returns:
        Tuple of (is_valid, status_str):
        - is_valid: True if tile can be processed, False otherwise
        - status_str: 'valid' if OK, or descriptive status ('empty')

    Examples:
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> valid, status = check_tile_validity(img, 'tile_0_0')
        >>> valid, status
        (False, 'empty')

        >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        >>> valid, status = check_tile_validity(img, 'tile_0_0')
        >>> valid, status
        (True, 'valid')
    """
    if img_rgb.max() == 0:
        return False, 'empty'
    return True, 'valid'


def prepare_tile_for_detection(
    img_rgb: np.ndarray,
    normalize: bool = True,
    p_low: int = 5,
    p_high: int = 95
) -> np.ndarray:
    """
    Prepare tile for detection by applying percentile normalization.

    Args:
        img_rgb: RGB image array.
        normalize: Whether to apply percentile normalization. Default True.
        p_low: Low percentile for normalization (default 5).
        p_high: High percentile for normalization (default 95).

    Returns:
        Normalized image array (same dtype as input or uint8 after norm).

    Note:
        Tissue validation (has_tissue check) is done separately as it's
        optional in some processing modes.

    Examples:
        >>> img = np.random.randint(0, 65535, (100, 100, 3), dtype=np.uint16)
        >>> normed = prepare_tile_for_detection(img)
        >>> normed.dtype
        dtype('uint8')
    """
    if normalize:
        from segmentation.io.html_export import percentile_normalize
        return percentile_normalize(img_rgb, p_low=p_low, p_high=p_high)
    return img_rgb


def build_mk_hspc_result(
    tid: str,
    status: str,
    mk_masks: Optional[np.ndarray] = None,
    hspc_masks: Optional[np.ndarray] = None,
    mk_feats: Optional[List[Dict[str, Any]]] = None,
    hspc_feats: Optional[List[Dict[str, Any]]] = None,
    tile: Optional[Dict[str, Any]] = None,
    slide_name: Optional[str] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build result dict for MK/HSPC tile worker functions.

    Creates a standardized result dictionary with consistent structure
    across all worker function modes.

    Args:
        tid: Tile identifier.
        status: Processing status ('success', 'error', 'empty', 'no_tissue').
        mk_masks: MK mask array (for success status).
        hspc_masks: HSPC mask array (for success status).
        mk_feats: MK feature list (for success status).
        hspc_feats: HSPC feature list (for success status).
        tile: Tile info dict with coordinates (for success status).
        slide_name: Optional slide name (for multi-slide modes).
        error: Error message (for error status).

    Returns:
        Result dictionary with keys appropriate to the status:
        - Always: tid, status
        - On success: mk_masks, hspc_masks, mk_feats, hspc_feats, tile
        - On error: error
        - Optionally: slide_name (when provided)

    Examples:
        >>> result = build_mk_hspc_result('tile_0_0', 'empty')
        >>> result['status']
        'empty'

        >>> result = build_mk_hspc_result(
        ...     'tile_0_0', 'success',
        ...     mk_masks=np.zeros((100, 100)),
        ...     hspc_masks=np.zeros((100, 100)),
        ...     mk_feats=[],
        ...     hspc_feats=[],
        ...     tile={'id': 'tile_0_0', 'x': 0, 'y': 0}
        ... )
        >>> result['status']
        'success'
    """
    result = {
        'tid': tid,
        'status': status,
    }

    if status == 'success':
        result.update({
            'mk_masks': mk_masks,
            'hspc_masks': hspc_masks,
            'mk_feats': mk_feats if mk_feats is not None else [],
            'hspc_feats': hspc_feats if hspc_feats is not None else [],
            'tile': tile,
        })
    elif status == 'error':
        result['error'] = error if error else 'Unknown error'

    # Add slide_name if provided (multi-slide mode)
    if slide_name is not None:
        result['slide_name'] = slide_name

    return result


def extract_tile_from_shared_memory(
    shared_image: np.ndarray,
    tile: Dict[str, Any],
    mosaic_origin: Tuple[int, int] = (0, 0)
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Extract tile region from shared memory array.

    Tile coordinates are global CZI mosaic coordinates. The mosaic_origin
    is subtracted to convert to 0-indexed array coordinates.

    Args:
        shared_image: Shared memory array containing full image.
        tile: Tile dict with 'x', 'y', 'w', 'h' keys (global coords).
        mosaic_origin: (ox, oy) mosaic origin to subtract for array indexing.

    Returns:
        Tuple of (image_data, error_message):
        - image_data: Extracted tile array, or None on error
        - error_message: Error description, or None on success

    Examples:
        >>> import numpy as np
        >>> shared = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        >>> tile = {'id': 't1', 'x': 0, 'y': 0, 'w': 100, 'h': 100}
        >>> img, err = extract_tile_from_shared_memory(shared, tile)
        >>> img.shape
        (100, 100)
    """
    if shared_image is None:
        return None, "Shared memory not available"

    try:
        ox, oy = mosaic_origin
        img = shared_image[
            tile['y'] - oy:tile['y'] - oy + tile['h'],
            tile['x'] - ox:tile['x'] - ox + tile['w']
        ]
        if img.size == 0:
            return None, f"Empty crop extracted from tile {tile.get('id', 'unknown')}"
        return img.copy(), None
    except Exception as e:
        return None, f"Memory slice error: {e}"
