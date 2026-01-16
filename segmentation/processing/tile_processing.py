"""
Tile processing helper functions for the unified segmentation pipeline.

This module provides helper functions to reduce nesting in the main tile processing
loop by extracting common patterns into reusable, well-typed functions.

Functions:
    build_detection_params: Factory for cell-type-specific detection parameters
    load_and_validate_tile: Load and validate tile data from loader
    enrich_detection_features: Add global coordinates and metadata to detections
    save_tile_outputs: Save masks and features to disk

Usage:
    from segmentation.processing.tile_processing import (
        build_detection_params,
        load_and_validate_tile,
        enrich_detection_features,
        save_tile_outputs,
    )

    # Build detection parameters
    params = build_detection_params('nmj', args, pixel_size_um)

    # Load and validate tile
    tile_rgb, cd31_data = load_and_validate_tile(
        loader, tile_x, tile_y, tile_size, channel
    )

    # Enrich features with global coordinates
    enrich_detection_features(features_list, tile_x, tile_y, slide_name, pixel_size_um, 'nmj')

    # Save outputs
    result = save_tile_outputs(tile_out_dir, 'nmj', masks, features_list)
"""

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import h5py

from segmentation.io.html_export import create_hdf5_dataset
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def build_detection_params(
    cell_type: str,
    args: Any,
    pixel_size_um: float
) -> Dict[str, Any]:
    """
    Factory function to build cell-type-specific detection parameters.

    Creates a parameter dictionary with appropriate values for each supported
    cell type, extracting values from command-line arguments.

    Args:
        cell_type: Type of cell to detect. Supported types:
            - 'nmj': Neuromuscular junctions
            - 'mk': Megakaryocytes
            - 'hspc': Hematopoietic stem/progenitor cells
            - 'vessel': Blood vessel cross-sections
            - 'mesothelium': Mesothelial ribbons
        args: Namespace object containing command-line arguments with
            cell-type-specific parameters.
        pixel_size_um: Pixel size in micrometers for unit conversions.

    Returns:
        Dictionary containing detection parameters for the specified cell type.

    Raises:
        ValueError: If cell_type is not one of the supported types.

    Examples:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     intensity_percentile=99,
        ...     min_area=150,
        ...     min_skeleton_length=30,
        ...     max_solidity=0.85
        ... )
        >>> params = build_detection_params('nmj', args, 0.22)
        >>> params['intensity_percentile']
        99
    """
    if cell_type == 'nmj':
        return {
            'intensity_percentile': getattr(args, 'intensity_percentile', 99),
            'min_area': getattr(args, 'min_area', 150),
            'min_skeleton_length': getattr(args, 'min_skeleton_length', 30),
            'max_solidity': getattr(args, 'max_solidity', 0.85),
        }

    elif cell_type == 'mk':
        return {
            'mk_min_area': getattr(args, 'mk_min_area', 1000),
            'mk_max_area': getattr(args, 'mk_max_area', 100000),
        }

    elif cell_type == 'hspc':
        # HSPC uses Cellpose which auto-detects parameters
        return {}

    elif cell_type == 'vessel':
        return {
            'min_vessel_diameter_um': getattr(args, 'min_vessel_diameter', 10),
            'max_vessel_diameter_um': getattr(args, 'max_vessel_diameter', 1000),
            'min_wall_thickness_um': getattr(args, 'min_wall_thickness', 2),
            'max_aspect_ratio': getattr(args, 'max_aspect_ratio', 4.0),
            'min_circularity': getattr(args, 'min_circularity', 0.3),
            'min_ring_completeness': getattr(args, 'min_ring_completeness', 0.5),
            'pixel_size_um': pixel_size_um,
            'classify_vessel_types': getattr(args, 'classify_vessel_types', False),
            'canny_low': getattr(args, 'canny_low', None),
            'canny_high': getattr(args, 'canny_high', None),
        }

    elif cell_type == 'mesothelium':
        return {
            'target_chunk_area_um2': getattr(args, 'target_chunk_area', 1500),
            'min_ribbon_width_um': getattr(args, 'min_ribbon_width', 5),
            'max_ribbon_width_um': getattr(args, 'max_ribbon_width', 30),
            'min_fragment_area_um2': getattr(args, 'min_fragment_area', 1500),
            'pixel_size_um': pixel_size_um,
        }

    else:
        raise ValueError(
            f"Unknown cell type: {cell_type}. "
            f"Supported types: nmj, mk, hspc, vessel, mesothelium"
        )


def load_and_validate_tile(
    loader: Any,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    channel: int,
    cell_type: Optional[str] = None,
    cd31_channel: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load tile data from loader and validate it has content.

    Handles the common pattern of:
    1. Loading tile data from the CZI loader
    2. Validating the tile is not None and has non-zero content
    3. Converting grayscale to RGB if needed
    4. Optionally loading CD31 channel for vessel validation

    Args:
        loader: CZILoader instance with get_tile() method.
        tile_x: X coordinate of tile origin in global mosaic coordinates.
        tile_y: Y coordinate of tile origin in global mosaic coordinates.
        tile_size: Size of the tile in pixels (assumes square tiles).
        channel: Channel index to load for main tile data.
        cell_type: Optional cell type string. If 'vessel' and cd31_channel
            is provided, will load CD31 data.
        cd31_channel: Optional channel index for CD31 data. Only used when
            cell_type is 'vessel'.

    Returns:
        Tuple of (tile_rgb, cd31_data):
            - tile_rgb: RGB image array (H, W, 3) as uint16 or None if invalid
            - cd31_data: CD31 channel array as float32 or None if not requested/available

        Returns (None, None) if:
            - tile_data is None or empty
            - tile_data has max value of 0 (no signal)

    Examples:
        >>> tile_rgb, cd31 = load_and_validate_tile(
        ...     loader, 0, 0, 3000, channel=1
        ... )
        >>> if tile_rgb is not None:
        ...     process_tile(tile_rgb)

        >>> # With CD31 validation for vessels
        >>> tile_rgb, cd31 = load_and_validate_tile(
        ...     loader, 0, 0, 3000, channel=0,
        ...     cell_type='vessel', cd31_channel=1
        ... )
    """
    # Load main channel tile
    tile_data = loader.get_tile(tile_x, tile_y, tile_size, channel=channel)

    # Validate tile data exists
    if tile_data is None or tile_data.size == 0:
        logger.debug(f"Tile ({tile_x}, {tile_y}): No data returned from loader")
        return None, None

    # Validate tile has content (not all zeros)
    if tile_data.max() == 0:
        logger.debug(f"Tile ({tile_x}, {tile_y}): Max value is 0, skipping")
        return None, None

    # Convert grayscale to RGB
    if tile_data.ndim == 2:
        tile_rgb = np.stack([tile_data] * 3, axis=-1)
    else:
        tile_rgb = tile_data

    # Optionally load CD31 channel for vessel validation
    cd31_data = None
    if cell_type == 'vessel' and cd31_channel is not None:
        cd31_tile = loader.get_tile(tile_x, tile_y, tile_size, channel=cd31_channel)
        if cd31_tile is not None and cd31_tile.size > 0:
            cd31_data = cd31_tile.astype(np.float32)
        else:
            logger.debug(
                f"Tile ({tile_x}, {tile_y}): CD31 channel {cd31_channel} "
                "returned no data"
            )

    return tile_rgb, cd31_data


def enrich_detection_features(
    features_list: List[Dict[str, Any]],
    tile_x: int,
    tile_y: int,
    slide_name: str,
    pixel_size_um: float,
    cell_type: str
) -> None:
    """
    Enrich detection features with global coordinates and metadata.

    Modifies each feature dict in-place to add:
    - uid: Universal identifier (slide_celltype_globalX_globalY)
    - global_center: [x, y] in global mosaic pixel coordinates
    - global_center_um: [x, y] in micrometers
    - tile_origin: [tile_x, tile_y] origin of source tile
    - slide_name: Name of the source slide

    For vessel detections, also transforms contours to global coordinates.

    Args:
        features_list: List of feature dictionaries from detection. Each dict
            must have a 'center' key with [local_x, local_y] coordinates.
            Modified in-place.
        tile_x: X coordinate of tile origin in global mosaic coordinates.
        tile_y: Y coordinate of tile origin in global mosaic coordinates.
        slide_name: Name of the slide (typically the CZI filename stem).
        pixel_size_um: Pixel size in micrometers for coordinate conversion.
        cell_type: Type of cell for UID generation.

    Returns:
        None. The features_list is modified in-place.

    Note:
        The coordinate system follows the convention:
        - All coordinates are stored as [x, y] (horizontal, vertical)
        - global_center = [local_x + tile_x, local_y + tile_y]
        - UIDs use rounded global coordinates to reduce collision probability

    Examples:
        >>> features = [
        ...     {'id': 'nmj_1', 'center': [100.5, 200.3], 'features': {...}},
        ...     {'id': 'nmj_2', 'center': [300.0, 400.0], 'features': {...}},
        ... ]
        >>> enrich_detection_features(
        ...     features, tile_x=1000, tile_y=2000,
        ...     slide_name='slide1', pixel_size_um=0.22, cell_type='nmj'
        ... )
        >>> features[0]['uid']
        'slide1_nmj_1101_2200'
        >>> features[0]['global_center']
        [1100.5, 2200.3]
    """
    for feat in features_list:
        # Extract local center coordinates
        local_cx, local_cy = feat['center']

        # Calculate global coordinates
        global_cx = tile_x + local_cx
        global_cy = tile_y + local_cy

        # Create universal ID: slide_celltype_globalX_globalY
        # Use round() to reduce collision probability for nearby cells
        uid = f"{slide_name}_{cell_type}_{round(global_cx)}_{round(global_cy)}"

        # Add enriched fields
        feat['uid'] = uid
        feat['global_center'] = [float(global_cx), float(global_cy)]
        feat['global_center_um'] = [
            float(global_cx * pixel_size_um),
            float(global_cy * pixel_size_um)
        ]
        feat['tile_origin'] = [tile_x, tile_y]
        feat['slide_name'] = slide_name

        # Handle vessel-specific contour transformations
        if cell_type == 'vessel':
            _transform_vessel_contours(feat, tile_x, tile_y)


def _transform_vessel_contours(
    feat: Dict[str, Any],
    tile_x: int,
    tile_y: int
) -> None:
    """
    Transform vessel contours from local to global coordinates.

    Helper function for enrich_detection_features. Handles the OpenCV contour
    format where each point is nested as [[x, y]].

    Args:
        feat: Feature dictionary with optional 'outer_contour' and
            'inner_contour' keys. Modified in-place.
        tile_x: X coordinate of tile origin.
        tile_y: Y coordinate of tile origin.
    """
    # Transform outer contour
    if 'outer_contour' in feat:
        feat['outer_contour_global'] = [
            [pt[0][0] + tile_x, pt[0][1] + tile_y]
            for pt in feat['outer_contour']
        ]

    # Transform inner contour
    if 'inner_contour' in feat:
        feat['inner_contour_global'] = [
            [pt[0][0] + tile_x, pt[0][1] + tile_y]
            for pt in feat['inner_contour']
        ]


def save_tile_outputs(
    tile_out_dir: Union[str, Path],
    cell_type: str,
    masks: np.ndarray,
    features_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Save tile detection outputs (masks and features) to disk.

    Creates the output directory if needed and saves:
    - Masks to HDF5 file with LZ4/gzip compression
    - Features to JSON file

    Args:
        tile_out_dir: Directory to save outputs. Created if it doesn't exist.
        cell_type: Type of cell for filename prefix (e.g., 'nmj', 'mk').
        masks: 2D numpy array of instance masks (uint16 or uint32).
            Each unique non-zero value represents a different detection.
        features_list: List of feature dictionaries for each detection.

    Returns:
        Dictionary containing:
            - masks_path: Path to saved HDF5 masks file
            - features_path: Path to saved JSON features file
            - detection_count: Number of detections saved
            - success: Boolean indicating if save was successful

    Raises:
        OSError: If directory creation or file writing fails.

    Examples:
        >>> result = save_tile_outputs(
        ...     '/output/tile_0_0', 'nmj', masks, features
        ... )
        >>> print(result)
        {
            'masks_path': '/output/tile_0_0/nmj_masks.h5',
            'features_path': '/output/tile_0_0/nmj_features.json',
            'detection_count': 5,
            'success': True
        }
    """
    tile_out_dir = Path(tile_out_dir)

    result = {
        'masks_path': None,
        'features_path': None,
        'detection_count': len(features_list),
        'success': False,
    }

    try:
        # Create directory if needed
        tile_out_dir.mkdir(parents=True, exist_ok=True)

        # Save masks to HDF5 with compression
        masks_path = tile_out_dir / f"{cell_type}_masks.h5"
        with h5py.File(masks_path, 'w') as f:
            # Convert to uint16 for storage efficiency
            # (supports up to 65535 detections per tile)
            create_hdf5_dataset(f, 'masks', masks.astype(np.uint16))

        result['masks_path'] = str(masks_path)

        # Save features to JSON
        features_path = tile_out_dir / f"{cell_type}_features.json"
        with open(features_path, 'w') as f:
            json.dump(features_list, f, indent=2)

        result['features_path'] = str(features_path)
        result['success'] = True

        logger.debug(
            f"Saved {len(features_list)} detections to {tile_out_dir}: "
            f"masks={masks_path.name}, features={features_path.name}"
        )

    except OSError as e:
        logger.error(f"Failed to save tile outputs to {tile_out_dir}: {e}")
        raise

    return result


def process_tile_complete(
    loader: Any,
    segmenter: Any,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    channel: int,
    cell_type: str,
    params: Dict[str, Any],
    slide_name: str,
    pixel_size_um: float,
    tile_out_dir: Union[str, Path],
    cd31_channel: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Complete tile processing pipeline combining load, detect, enrich, and save.

    This is a convenience function that combines all tile processing steps
    into a single call, handling errors gracefully.

    Args:
        loader: CZILoader instance.
        segmenter: UnifiedSegmenter instance.
        tile_x: X coordinate of tile origin.
        tile_y: Y coordinate of tile origin.
        tile_size: Tile size in pixels.
        channel: Main channel index.
        cell_type: Type of cell to detect.
        params: Detection parameters dict.
        slide_name: Name of the source slide.
        pixel_size_um: Pixel size in micrometers.
        tile_out_dir: Directory to save outputs.
        cd31_channel: Optional CD31 channel for vessel validation.

    Returns:
        Dictionary with processing results, or None if tile was skipped/failed:
            - features_list: Enriched feature dictionaries
            - masks: Detection mask array
            - detection_count: Number of detections
            - tile_origin: [tile_x, tile_y]
            - saved_files: Output from save_tile_outputs

    Examples:
        >>> result = process_tile_complete(
        ...     loader, segmenter, 0, 0, 3000, 1, 'nmj', params,
        ...     'slide1', 0.22, '/output/tile_0_0'
        ... )
        >>> if result:
        ...     print(f"Found {result['detection_count']} detections")
    """
    # Load and validate tile
    tile_rgb, cd31_data = load_and_validate_tile(
        loader, tile_x, tile_y, tile_size, channel,
        cell_type=cell_type, cd31_channel=cd31_channel
    )

    if tile_rgb is None:
        return None

    try:
        # Run detection
        masks, features_list = segmenter.process_tile(
            tile_rgb, cell_type, params, cd31_channel=cd31_data
        )

        if len(features_list) == 0:
            return None

        # Enrich features with global coordinates
        enrich_detection_features(
            features_list, tile_x, tile_y, slide_name, pixel_size_um, cell_type
        )

        # Save outputs
        saved_files = save_tile_outputs(tile_out_dir, cell_type, masks, features_list)

        return {
            'features_list': features_list,
            'masks': masks,
            'detection_count': len(features_list),
            'tile_origin': [tile_x, tile_y],
            'saved_files': saved_files,
        }

    except Exception as e:
        logger.error(f"Error processing tile ({tile_x}, {tile_y}): {e}")
        return None
