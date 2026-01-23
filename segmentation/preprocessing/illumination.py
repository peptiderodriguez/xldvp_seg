"""
Illumination and photobleaching correction for microscopy images.

This module provides functions to correct common illumination artifacts in
tiled microscopy images, including:
- Horizontal and vertical photobleaching bands from stitched acquisitions
- Uneven background illumination
- Vignetting

Usage:
    from segmentation.preprocessing import correct_photobleaching

    # Simple correction with defaults
    corrected = correct_photobleaching(image)

    # Custom correction
    from segmentation.preprocessing import normalize_rows_columns, morphological_background_subtraction
    corrected = normalize_rows_columns(image)
    corrected = morphological_background_subtraction(corrected, kernel_size=151)
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def normalize_rows_columns(
    image: np.ndarray,
    target_mean: Optional[float] = None
) -> np.ndarray:
    """
    Normalize each row and column to have consistent mean intensity.

    This corrects for horizontal and vertical photobleaching bands by:
    1. Computing row-wise means and normalizing rows
    2. Computing column-wise means and normalizing columns

    Args:
        image: 2D grayscale image (any numeric dtype)
        target_mean: Target mean intensity. If None, uses global mean.

    Returns:
        Corrected image with consistent row/column intensities (float64)

    Example:
        >>> corrected = normalize_rows_columns(tile)
        >>> # Row and column means will now be uniform
    """
    img = image.astype(np.float64)

    # Use global mean as target if not specified
    if target_mean is None:
        target_mean = np.mean(img)

    # Step 1: Normalize rows
    row_means = np.mean(img, axis=1, keepdims=True)
    # Avoid division by zero
    row_means = np.where(row_means > 0, row_means, 1)
    row_corrected = img * (target_mean / row_means)

    # Step 2: Normalize columns on the row-corrected image
    col_means = np.mean(row_corrected, axis=0, keepdims=True)
    col_means = np.where(col_means > 0, col_means, 1)
    corrected = row_corrected * (target_mean / col_means)

    return corrected


def morphological_background_subtraction(
    image: np.ndarray,
    kernel_size: int = 101
) -> np.ndarray:
    """
    Remove low-frequency background using morphological opening.

    Uses a large structuring element to estimate the background,
    then subtracts it to remove illumination artifacts.

    Args:
        image: 2D grayscale image
        kernel_size: Size of the structuring element (should be odd).
                    Larger values remove larger-scale variations.
                    Typical range: 51-201 pixels.

    Returns:
        Background-subtracted image (float64)

    Example:
        >>> # Remove large-scale illumination gradients
        >>> corrected = morphological_background_subtraction(tile, kernel_size=101)
    """
    img = image.astype(np.float64)

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create large elliptical structuring element
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    # Estimate background using morphological opening
    # Opening = erosion followed by dilation
    # This removes bright structures smaller than the kernel
    background = cv2.morphologyEx(
        img.astype(np.float32),
        cv2.MORPH_OPEN,
        kernel
    )

    # Subtract background
    corrected = img - background.astype(np.float64)

    # Shift to positive values (add global mean of background)
    corrected = corrected + np.mean(background)

    return corrected


def correct_photobleaching(
    image: np.ndarray,
    row_col_normalize: bool = True,
    morph_subtract: bool = True,
    morph_kernel_size: int = 101,
    target_mean: Optional[float] = None
) -> np.ndarray:
    """
    Apply combined photobleaching correction.

    This is the main entry point for correcting photobleaching artifacts.
    It combines row/column normalization (for banding) with morphological
    background subtraction (for gradients).

    Args:
        image: 2D grayscale image
        row_col_normalize: Apply row/column normalization to fix bands
        morph_subtract: Apply morphological background subtraction
        morph_kernel_size: Kernel size for morphological operation
        target_mean: Target mean for normalization (None = use global mean)

    Returns:
        Corrected image (float64)

    Example:
        >>> from segmentation.preprocessing import correct_photobleaching
        >>>
        >>> # Full correction (recommended for most cases)
        >>> corrected = correct_photobleaching(tile)
        >>>
        >>> # Only fix banding, no background subtraction
        >>> corrected = correct_photobleaching(tile, morph_subtract=False)
        >>>
        >>> # Custom kernel for larger-scale gradients
        >>> corrected = correct_photobleaching(tile, morph_kernel_size=151)
    """
    corrected = image.astype(np.float64)

    if row_col_normalize:
        corrected = normalize_rows_columns(corrected, target_mean=target_mean)

    if morph_subtract:
        corrected = morphological_background_subtraction(
            corrected,
            kernel_size=morph_kernel_size
        )

    return corrected


def estimate_band_severity(image: np.ndarray) -> Dict[str, float]:
    """
    Estimate the severity of horizontal and vertical banding artifacts.

    Uses coefficient of variation (CV) of row and column means as a measure
    of banding severity. Higher CV indicates more severe banding.

    Args:
        image: 2D grayscale image

    Returns:
        Dictionary with:
        - row_cv: Coefficient of variation of row means (%)
        - col_cv: Coefficient of variation of column means (%)
        - severity: Overall severity rating ('none', 'mild', 'moderate', 'severe')

    Example:
        >>> stats = estimate_band_severity(tile)
        >>> print(f"Row CV: {stats['row_cv']:.1f}%, Severity: {stats['severity']}")
    """
    img = image.astype(np.float64)

    row_means = np.mean(img, axis=1)
    col_means = np.mean(img, axis=0)

    row_cv = np.std(row_means) / np.mean(row_means) * 100 if np.mean(row_means) > 0 else 0
    col_cv = np.std(col_means) / np.mean(col_means) * 100 if np.mean(col_means) > 0 else 0

    # Determine overall severity
    max_cv = max(row_cv, col_cv)
    if max_cv < 5:
        severity = 'none'
    elif max_cv < 10:
        severity = 'mild'
    elif max_cv < 20:
        severity = 'moderate'
    else:
        severity = 'severe'

    return {
        'row_cv': row_cv,
        'col_cv': col_cv,
        'max_cv': max_cv,
        'severity': severity
    }


def correct_tile_batch(
    tiles: list,
    **kwargs
) -> list:
    """
    Apply photobleaching correction to a batch of tiles.

    Args:
        tiles: List of 2D numpy arrays (tiles)
        **kwargs: Arguments passed to correct_photobleaching()

    Returns:
        List of corrected tiles

    Example:
        >>> corrected_tiles = correct_tile_batch(tiles, morph_kernel_size=101)
    """
    return [correct_photobleaching(tile, **kwargs) for tile in tiles]


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    CLAHE improves local contrast while limiting noise amplification,
    making it ideal for handling photobleaching and uneven staining.

    Args:
        image: 2D grayscale image (any dtype)
        clip_limit: Threshold for contrast limiting. Higher values give
                   more contrast but may amplify noise. Default 2.0.
        tile_size: Size of grid for histogram equalization. Smaller tiles
                  give more local adaptation. Default (8, 8).

    Returns:
        CLAHE-enhanced image (uint8, range 0-255)

    Example:
        >>> enhanced = apply_clahe(tile)
        >>> enhanced = apply_clahe(tile, clip_limit=3.0)  # More contrast
    """
    # Normalize to uint8
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img_uint8 = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_uint8 = np.zeros(image.shape, dtype=np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img_uint8)
