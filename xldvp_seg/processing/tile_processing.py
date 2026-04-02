"""
Tile processing helper functions for the unified segmentation pipeline.

Functions:
    process_single_tile: Common per-tile detection with CUDA retry, mask_label, UID
    enrich_detection_features: Add global coordinates and metadata to detections
    save_tile_outputs: Save masks and features to disk
"""

import gc
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from segmentation.io.html_export import create_hdf5_dataset
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def detections_to_features_list(detections, cell_type):
    """
    Convert a list of Detection objects to the expected features_list format.

    Args:
        detections: List of Detection objects from strategy.detect()
        cell_type: Cell type string for ID prefix

    Returns:
        List of feature dicts with 'id', 'center', 'features', and optionally
        'rf_prediction' and vessel contour fields.
    """
    if detections is None:
        return []

    features_list = []
    for i, det in enumerate(detections, start=1):
        feat_dict = {
            "id": det.id if det.id else f"{cell_type}_{i}",
            "center": det.centroid,  # [x, y] format
            "features": det.features.copy(),
        }
        # Include RF prediction score at top level.
        # Prefer classifier-set feature keys (rf_prediction for islet, prob_nmj for NMJ)
        # to avoid exposing solidity/sam2_score as "rf_prediction" when no classifier ran.
        if "rf_prediction" in det.features:
            feat_dict["rf_prediction"] = det.features["rf_prediction"]
        elif "prob_nmj" in det.features:
            feat_dict["rf_prediction"] = det.features["prob_nmj"]
        else:
            feat_dict["rf_prediction"] = det.score if det.score is not None else 0.0

        # For vessels, lift contours from features to top level
        if cell_type == "vessel":
            if "outer_contour" in feat_dict["features"]:
                feat_dict["outer_contour"] = feat_dict["features"].pop("outer_contour")
            if "inner_contour" in feat_dict["features"]:
                feat_dict["inner_contour"] = feat_dict["features"].pop("inner_contour")
            if "sma_contour" in feat_dict["features"]:
                feat_dict["sma_contour"] = feat_dict["features"].pop("sma_contour")

        features_list.append(feat_dict)
    return features_list


def process_single_tile(
    tile_rgb: np.ndarray,
    extra_channel_tiles: dict[int, np.ndarray] | None,
    strategy,
    models: dict[str, Any],
    pixel_size_um: float,
    cell_type: str,
    slide_name: str,
    tile_x: int,
    tile_y: int,
    cd31_channel_data: np.ndarray | None = None,
    channel_names: dict[int, str] | None = None,
    max_retries: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    """
    Process a single tile through detection pipeline with CUDA retry.

    This is the common per-tile logic shared by both single-GPU and multi-GPU paths.
    Encapsulates:
    1. CUDA retry with exponential backoff around strategy.detect()
    2. Detection-to-dict conversion
    3. Centroid-based mask_label lookup with 3x3 fallback
    4. UID + global coords + contour globalization

    Args:
        tile_rgb: uint8 RGB tile for detection/visual models (H, W, 3)
        extra_channel_tiles: Dict mapping channel index to uint16 2D arrays
            for per-channel feature extraction (or None)
        strategy: DetectionStrategy instance (NMJ, MK, Vessel, Cell, etc.)
        models: Dict with loaded models (sam2_predictor, resnet, etc.)
        pixel_size_um: Pixel size in micrometers
        cell_type: Cell type string ('nmj', 'mk', 'vessel', 'cell', 'mesothelium')
        slide_name: Slide name for UID generation
        tile_x: Global X coordinate of tile origin
        tile_y: Global Y coordinate of tile origin
        cd31_channel_data: Optional CD31 channel for vessel validation
        channel_names: Optional channel name mapping for vessels
        max_retries: Max CUDA retry attempts (default 3)

    Returns:
        Tuple of (label_array, features_list) or None if detection failed.
        - label_array: uint32 mask array with detection IDs
        - features_list: List of feature dicts with uid, global_center, mask_label, etc.
    """
    import torch

    # --- Step 1: Run detection with CUDA retry ---
    masks = None
    detections = None

    for attempt in range(max_retries):
        try:
            # TODO: Vessel detect() requires extra kwargs (cd31_channel, channel_names,
            # tile_x, tile_y, tile_size) that other strategies don't accept.
            # Unifying the signature would require all strategies to accept **kwargs.
            if cell_type == "vessel":
                masks, detections = strategy.detect(
                    tile_rgb,
                    models,
                    pixel_size_um,
                    cd31_channel=cd31_channel_data,
                    extra_channels=extra_channel_tiles,
                    channel_names=channel_names,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    tile_size=tile_rgb.shape[0],
                )
            elif cell_type in ("nmj", "mk", "cell", "islet", "tissue_pattern"):
                masks, detections = strategy.detect(
                    tile_rgb,
                    models,
                    pixel_size_um,
                    extra_channels=extra_channel_tiles,
                    extract_features=True,
                )
            else:
                # Mesothelium and others — basic detect()
                result = strategy.detect(tile_rgb, models, pixel_size_um)
                # Some strategies return tuple, some return list
                if isinstance(result, tuple) and len(result) == 2:
                    masks, detections = result
                else:
                    detections = result
                    masks = None
            break  # Success

        except torch.cuda.OutOfMemoryError as oom_err:
            if attempt < max_retries - 1:
                logger.warning(
                    f"CUDA OOM on tile ({tile_x}, {tile_y}) attempt "
                    f"{attempt + 1}/{max_retries}: {oom_err}"
                )
                gc.collect()
                from segmentation.utils.device import empty_cache as _empty_cache

                _empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                logger.error(f"All {max_retries} CUDA retries failed for tile ({tile_x}, {tile_y})")
                raise

        except RuntimeError as rt_err:
            err_str = str(rt_err).lower()
            is_gpu_error = (
                "cuda" in err_str
                or "mps" in err_str
                or "out of memory" in err_str
                or "device-side assert" in err_str
            )
            if is_gpu_error and attempt < max_retries - 1:
                logger.warning(
                    f"GPU RuntimeError on tile ({tile_x}, {tile_y}) attempt "
                    f"{attempt + 1}/{max_retries}: {rt_err}"
                )
                gc.collect()
                from segmentation.utils.device import empty_cache as _empty_cache

                _empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                time.sleep(1.0 * (attempt + 1))
                continue
            else:
                raise

    # --- Step 2: Convert detections to features_list format ---
    if detections is None or len(detections) == 0:
        return None

    features_list = detections_to_features_list(detections, cell_type)
    if not features_list:
        return None

    # Build label array if not returned by strategy
    if masks is None:
        h, w = tile_rgb.shape[:2]
        masks = np.zeros((h, w), dtype=np.uint32)
        for i, det in enumerate(detections, start=1):
            if hasattr(det, "mask") and det.mask is not None:
                masks[det.mask] = i

    # --- Step 3: Centroid-based mask_label lookup with 3x3 fallback ---
    for feat in features_list:
        local_cx, local_cy = feat["center"]
        mask_y = int(round(local_cy))
        mask_x = int(round(local_cx))
        # Clamp to valid range
        mask_y = max(0, min(mask_y, masks.shape[0] - 1))
        mask_x = max(0, min(mask_x, masks.shape[1] - 1))
        actual_mask_label = int(masks[mask_y, mask_x])

        if actual_mask_label == 0:
            # Centroid on background — search 7x7 neighborhood (concave shapes)
            found_label = 0
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny = max(0, min(mask_y + dy, masks.shape[0] - 1))
                    nx = max(0, min(mask_x + dx, masks.shape[1] - 1))
                    if masks[ny, nx] > 0:
                        found_label = int(masks[ny, nx])
                        break
                if found_label > 0:
                    break
            actual_mask_label = found_label
            if found_label == 0:
                logger.warning(
                    f"Could not find mask label for detection at "
                    f"({local_cx}, {local_cy}) in tile ({tile_x}, {tile_y})"
                )

        feat["mask_label"] = actual_mask_label

    # Remove detections with mask_label=0 (no valid mask found)
    features_list = [f for f in features_list if f.get("mask_label", 0) > 0]

    # --- Step 4: UID + global coords + contour globalization ---
    enrich_detection_features(features_list, tile_x, tile_y, slide_name, pixel_size_um, cell_type)

    # Set id=uid for consistency with multigpu_nmj path
    for feat in features_list:
        feat["id"] = feat["uid"]

    return masks, features_list


def enrich_detection_features(
    features_list: list[dict[str, Any]],
    tile_x: int,
    tile_y: int,
    slide_name: str,
    pixel_size_um: float,
    cell_type: str,
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
        center = feat.get("center")
        if center is None or len(center) < 2:
            logger.warning("Detection missing 'center' key, skipping enrichment")
            continue
        local_cx, local_cy = center

        # Calculate global coordinates
        global_cx = tile_x + local_cx
        global_cy = tile_y + local_cy

        # Create universal ID: slide_celltype_globalX_globalY
        # Use round() to reduce collision probability for nearby cells
        uid = f"{slide_name}_{cell_type}_{int(round(global_cx))}_{int(round(global_cy))}"

        # Add enriched fields
        feat["uid"] = uid
        feat["global_center"] = [float(global_cx), float(global_cy)]
        feat["global_center_um"] = [
            float(global_cx * pixel_size_um),
            float(global_cy * pixel_size_um),
        ]
        feat["tile_origin"] = [tile_x, tile_y]
        feat["slide_name"] = slide_name
        feat["pixel_size_um"] = float(pixel_size_um)

        # Handle vessel-specific contour transformations
        if cell_type == "vessel":
            _transform_vessel_contours(feat, tile_x, tile_y)


def _transform_vessel_contours(feat: dict[str, Any], tile_x: int, tile_y: int) -> None:
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
    # Contours may be OpenCV format [[x, y]] or flat [x, y] (from .tolist())
    if feat.get("outer_contour") is not None:
        feat["outer_contour_global"] = [
            (
                [pt[0] + tile_x, pt[1] + tile_y]
                if isinstance(pt[0], (int, float))
                else [pt[0][0] + tile_x, pt[0][1] + tile_y]
            )
            for pt in feat["outer_contour"]
        ]

    # Transform inner contour
    if feat.get("inner_contour") is not None:
        feat["inner_contour_global"] = [
            (
                [pt[0] + tile_x, pt[1] + tile_y]
                if isinstance(pt[0], (int, float))
                else [pt[0][0] + tile_x, pt[0][1] + tile_y]
            )
            for pt in feat["inner_contour"]
        ]

    # Transform SMA contour
    if feat.get("sma_contour") is not None:
        feat["sma_contour_global"] = [
            (
                [pt[0] + tile_x, pt[1] + tile_y]
                if isinstance(pt[0], (int, float))
                else [pt[0][0] + tile_x, pt[0][1] + tile_y]
            )
            for pt in feat["sma_contour"]
        ]


def save_tile_outputs(
    tile_out_dir: str | Path, cell_type: str, masks: np.ndarray, features_list: list[dict[str, Any]]
) -> dict[str, Any]:
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
        "masks_path": None,
        "features_path": None,
        "detection_count": len(features_list),
        "success": False,
    }

    try:
        # Create directory if needed
        tile_out_dir.mkdir(parents=True, exist_ok=True)

        # Save masks to HDF5 with compression
        masks_path = tile_out_dir / f"{cell_type}_masks.h5"
        with h5py.File(masks_path, "w") as f:
            # Keep original dtype (uint32 supports >65535 labels per tile)
            create_hdf5_dataset(f, "masks", masks)

        result["masks_path"] = str(masks_path)

        # Save features to JSON
        features_path = tile_out_dir / f"{cell_type}_features.json"
        from segmentation.utils.json_utils import NumpyEncoder

        with open(features_path, "w") as f:
            json.dump(features_list, f, cls=NumpyEncoder)

        result["features_path"] = str(features_path)
        result["success"] = True

        logger.debug(
            f"Saved {len(features_list)} detections to {tile_out_dir}: "
            f"masks={masks_path.name}, features={features_path.name}"
        )

    except OSError as e:
        logger.error(f"Failed to save tile outputs to {tile_out_dir}: {e}")
        raise

    return result
