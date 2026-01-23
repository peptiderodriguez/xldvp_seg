"""
Vesselness filtering for enhancing tubular and ring-like structures.

Uses Frangi vesselness filter based on Hessian eigenvalue analysis to
detect vessel-like structures regardless of intensity, responding to
local geometry/structure instead.

Usage:
    from segmentation.preprocessing import compute_vesselness

    # Enhance vessel walls in SMA image
    vesselness = compute_vesselness(sma_image)

    # Multi-scale for various vessel sizes
    vesselness = compute_vesselness(sma_image, sigmas=range(1, 15))
"""

import numpy as np
from typing import Tuple, Union, Optional, Sequence
from skimage.filters import frangi, hessian


def compute_vesselness(
    image: np.ndarray,
    sigmas: Sequence[float] = range(1, 10),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: Optional[float] = None,
    black_ridges: bool = False,
) -> np.ndarray:
    """
    Compute Frangi vesselness to enhance tubular/ring-like structures.

    The Frangi filter analyzes local image geometry using Hessian eigenvalues
    to detect elongated structures (vessel walls). It responds to *structure*
    rather than intensity, making it robust to noise and intensity variations.

    Args:
        image: 2D grayscale image (will be normalized internally)
        sigmas: Scales for multi-scale analysis. Each sigma corresponds to
               a different vessel wall thickness. Default range(1,10) covers
               walls from ~2-20 pixels thick.
        alpha: Frangi correction constant for plate-like structures.
               Higher = more sensitivity to elongated structures.
        beta: Frangi correction constant for blob-like structures.
              Higher = less response to blob-like (non-vessel) structures.
        gamma: Frangi correction constant for background noise.
               If None, uses half the max Hessian norm.
        black_ridges: If True, detect dark ridges on bright background.
                     If False (default), detect bright ridges on dark background.
                     For SMA staining (bright vessels), use False.

    Returns:
        Vesselness map (float64, range ~0-1) where high values indicate
        vessel-like structures.

    Example:
        >>> # Basic usage
        >>> vesselness = compute_vesselness(sma_tile)
        >>>
        >>> # For larger vessels (thicker walls)
        >>> vesselness = compute_vesselness(sma_tile, sigmas=range(2, 20))
        >>>
        >>> # Threshold to get binary vessel mask
        >>> vessel_mask = vesselness > 0.01
    """
    # Normalize image to 0-1 range
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)

    # Compute Frangi vesselness
    vesselness = frangi(
        img_norm,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    )

    return vesselness


def compute_vesselness_multiscale(
    image: np.ndarray,
    sigma_ranges: list = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vesselness at multiple scale ranges and return both
    the combined vesselness and the scale at which max response occurred.

    Useful for detecting the approximate vessel size at each location.

    Args:
        image: 2D grayscale image
        sigma_ranges: List of (min_sigma, max_sigma) tuples for different
                     size ranges. Default covers small, medium, large vessels.
        **kwargs: Additional arguments passed to compute_vesselness()

    Returns:
        Tuple of (vesselness_map, scale_map) where:
        - vesselness_map: Maximum vesselness across all scales
        - scale_map: Index of the scale range with max response (0, 1, 2, ...)

    Example:
        >>> vesselness, scales = compute_vesselness_multiscale(tile)
        >>> small_vessels = vesselness * (scales == 0)  # Small scale responses
    """
    if sigma_ranges is None:
        sigma_ranges = [
            (1, 5),    # Small vessels / capillaries
            (3, 10),   # Medium vessels
            (8, 20),   # Large vessels
        ]

    vesselness_stack = []

    for min_sig, max_sig in sigma_ranges:
        sigmas = range(min_sig, max_sig + 1)
        v = compute_vesselness(image, sigmas=sigmas, **kwargs)
        vesselness_stack.append(v)

    vesselness_stack = np.stack(vesselness_stack, axis=0)

    # Get max across scales and which scale gave max
    vesselness_max = np.max(vesselness_stack, axis=0)
    scale_map = np.argmax(vesselness_stack, axis=0)

    return vesselness_max, scale_map


def enhance_ring_structures(
    image: np.ndarray,
    inner_sigmas: Sequence[float] = range(1, 8),
    outer_sigmas: Sequence[float] = range(5, 15),
) -> np.ndarray:
    """
    Enhance ring-like structures specifically (vessel cross-sections).

    Computes vesselness at two scale ranges and combines them to
    emphasize closed ring patterns over simple tubes.

    Args:
        image: 2D grayscale image
        inner_sigmas: Scales for inner wall detection
        outer_sigmas: Scales for outer wall detection

    Returns:
        Ring-enhanced vesselness map

    Note:
        This is experimental - basic Frangi may work better in practice.
    """
    # Inner wall response
    inner = compute_vesselness(image, sigmas=inner_sigmas)

    # Outer wall response
    outer = compute_vesselness(image, sigmas=outer_sigmas)

    # Combine - rings should have both inner and outer responses
    # Use geometric mean to require both
    combined = np.sqrt(inner * outer)

    return combined
