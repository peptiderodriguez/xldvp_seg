"""
Mask cleanup utilities for LMD (Laser Microdissection) export.

Addresses two common mask quality issues:
1. Fragmented masks - Single detection contains many disconnected pieces
2. Masks with holes - Donut-shaped masks with internal gaps

Usage:
    from segmentation.utils.mask_cleanup import cleanup_mask, recompute_mask_features

    # Clean up a mask (keep largest component, fill holes)
    cleaned_mask = cleanup_mask(mask, keep_largest=True, fill_internal_holes=True)

    # Recompute features after cleanup
    features = recompute_mask_features(cleaned_mask, image, pixel_size_um)
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Dict, Any


def get_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component (removes fragments).

    Args:
        mask: Binary mask (2D boolean or uint8 array)

    Returns:
        Binary mask with only the largest connected component
    """
    if not mask.any():
        return mask.copy()

    # Label connected components
    labeled, num_features = ndimage.label(mask.astype(bool))

    if num_features <= 1:
        return mask.copy()

    # Find sizes of all components
    sizes = ndimage.sum(mask.astype(bool), labeled, range(1, num_features + 1))

    # Keep only the largest
    largest_label = np.argmax(sizes) + 1
    return (labeled == largest_label).astype(mask.dtype)


def fill_holes(
    mask: np.ndarray,
    max_hole_area_fraction: float = 0.5,
) -> np.ndarray:
    """
    Fill internal holes in a binary mask.

    Preserves large holes (e.g., vessel lumens) by only filling holes
    smaller than max_hole_area_fraction of the total mask area.

    Args:
        mask: Binary mask (2D boolean or uint8 array)
        max_hole_area_fraction: Maximum hole size to fill as fraction of mask area.
            Holes larger than this are preserved (default 0.5 = 50%).
            Set to 1.0 to fill all holes regardless of size.

    Returns:
        Binary mask with small internal holes filled
    """
    if not mask.any():
        return mask.copy()

    mask_bool = mask.astype(bool)
    mask_area = mask_bool.sum()

    # Fill all holes first
    filled = ndimage.binary_fill_holes(mask_bool)

    # Find holes by subtracting original from filled
    holes = filled & ~mask_bool

    if not holes.any():
        return mask.copy()

    # If we want to preserve large holes
    if max_hole_area_fraction < 1.0:
        max_hole_area = mask_area * max_hole_area_fraction

        # Label individual holes
        labeled_holes, num_holes = ndimage.label(holes)

        if num_holes > 0:
            # Find hole sizes
            hole_sizes = ndimage.sum(holes, labeled_holes, range(1, num_holes + 1))

            # Create mask of holes to preserve (large holes)
            preserve_mask = np.zeros_like(holes)
            for i, size in enumerate(hole_sizes, 1):
                if size > max_hole_area:
                    preserve_mask |= (labeled_holes == i)

            # Result = filled mask minus preserved large holes
            result = filled & ~preserve_mask
            return result.astype(mask.dtype)

    return filled.astype(mask.dtype)


def cleanup_mask(
    mask: np.ndarray,
    keep_largest: bool = True,
    fill_internal_holes: bool = True,
    max_hole_area_fraction: float = 0.5,
) -> np.ndarray:
    """
    Unified cleanup pipeline for segmentation masks.

    Applies cleanup operations in order:
    1. Keep largest connected component (if enabled)
    2. Fill internal holes (if enabled)

    Args:
        mask: Binary mask (2D boolean or uint8 array)
        keep_largest: If True, keep only largest connected component (default True)
        fill_internal_holes: If True, fill internal holes (default True)
        max_hole_area_fraction: Max hole size to fill as fraction of mask area (default 0.5)

    Returns:
        Cleaned binary mask
    """
    result = mask.copy()

    if not result.any():
        return result

    if keep_largest:
        result = get_largest_connected_component(result)

    if fill_internal_holes:
        result = fill_holes(result, max_hole_area_fraction)

    return result


def recompute_mask_features(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    pixel_size_um: float = 0.1725,
) -> Dict[str, Any]:
    """
    Recompute mask features after cleanup.

    Computes:
    - area_px, area_um2: Mask area
    - centroid_xy: [x, y] center for crop positioning
    - bbox_xyxy: [x1, y1, x2, y2] bounding box
    - solidity, circularity: Shape descriptors

    Args:
        mask: Binary mask (2D array)
        image: Optional RGB image for intensity features
        pixel_size_um: Pixel size in micrometers (default 0.1725)

    Returns:
        Dictionary with recomputed features
    """
    features = {}

    if not mask.any():
        features['area_px'] = 0
        features['area_um2'] = 0.0
        features['centroid_xy'] = [0.0, 0.0]
        features['bbox_xyxy'] = [0, 0, 0, 0]
        return features

    mask_bool = mask.astype(bool)

    # Area
    area_px = int(mask_bool.sum())
    features['area_px'] = area_px
    features['area_um2'] = area_px * (pixel_size_um ** 2)

    # Centroid (x, y order for consistency with existing code)
    ys, xs = np.where(mask_bool)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    features['centroid_xy'] = [centroid_x, centroid_y]

    # Bounding box
    features['bbox_xyxy'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    # Shape features (solidity, circularity)
    try:
        from skimage import measure

        # Solidity = area / convex_area
        props = measure.regionprops(mask_bool.astype(np.uint8))[0]
        convex_area = props.convex_area if hasattr(props, 'convex_area') else area_px
        features['solidity'] = area_px / convex_area if convex_area > 0 else 1.0

        # Circularity = 4 * pi * area / perimeter^2
        perimeter = props.perimeter if hasattr(props, 'perimeter') else 0
        if perimeter > 0:
            features['circularity'] = 4 * np.pi * area_px / (perimeter ** 2)
        else:
            features['circularity'] = 1.0

    except Exception:
        features['solidity'] = 1.0
        features['circularity'] = 1.0

    return features


def apply_cleanup_to_detection(
    mask: np.ndarray,
    feat: Dict[str, Any],
    label_array: np.ndarray,
    det_id: int,
    pixel_size_um: float = 0.1725,
    keep_largest: bool = True,
    fill_internal_holes: bool = True,
    max_hole_area_fraction: float = 0.5,
) -> np.ndarray:
    """
    Apply cleanup to a single detection and update the label array and features.

    This is a convenience function for the common pattern in process_tile_safe()
    where we need to clean up a mask and update both the label array and features.

    Args:
        mask: Binary mask for this detection (extracted as label_array == det_id)
        feat: Feature dictionary for this detection (modified in place)
        label_array: Full label array for the tile (modified in place)
        det_id: Detection ID in the label array
        pixel_size_um: Pixel size in micrometers
        keep_largest: If True, keep only largest connected component
        fill_internal_holes: If True, fill internal holes
        max_hole_area_fraction: Max hole size to fill as fraction of mask area

    Returns:
        Cleaned mask (label_array is also updated in place)
    """
    # Apply cleanup
    cleaned_mask = cleanup_mask(
        mask,
        keep_largest=keep_largest,
        fill_internal_holes=fill_internal_holes,
        max_hole_area_fraction=max_hole_area_fraction,
    )

    # Update label array
    label_array[mask] = 0  # Clear original
    label_array[cleaned_mask.astype(bool)] = det_id  # Set cleaned

    # Update features
    feat['area'] = int(cleaned_mask.sum())
    feat['area_um2'] = feat['area'] * (pixel_size_um ** 2)

    # Recompute centroid for crop centering
    if cleaned_mask.any():
        ys, xs = np.where(cleaned_mask)
        feat['center'] = [float(xs.mean()), float(ys.mean())]

    return cleaned_mask
