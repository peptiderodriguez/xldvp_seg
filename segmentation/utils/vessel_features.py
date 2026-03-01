"""
Vessel-specific feature extraction for blood vessel segmentation.

This module provides biologically and physically meaningful features
specifically designed for:
1. Vessel vs non-vessel classification
2. Artery vs vein classification

Features are organized into categories:
- Ring/Wall Features: Wall integrity and uniformity metrics
- Diameter/Size Features: Size measurements and ratios
- Shape Features: Geometric shape descriptors
- Intensity/Gradient Features: SMA signal characteristics
- Context Features: Background and contrast metrics

Target: ~25-30 vessel-specific features that complement the generic
morphological features for improved vessel classification.

Usage:
    from segmentation.utils.vessel_features import (
        extract_vessel_features,
        VESSEL_FEATURE_NAMES,
        VESSEL_FEATURE_COUNT,
    )

    features = extract_vessel_features(
        wall_mask, lumen_mask, sma_channel,
        outer_contour, inner_contour,
        pixel_size_um=0.22
    )
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, List
from scipy.ndimage import distance_transform_edt, sobel
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# FEATURE NAME DEFINITIONS
# =============================================================================

# Ring/Wall Features (5 features)
RING_WALL_FEATURE_NAMES = [
    'ring_completeness',       # Fraction of perimeter with SMA signal
    'wall_thickness_cv',       # Coefficient of variation of wall thickness (std/mean)
    'wall_asymmetry',          # max/min wall thickness ratio
    'lumen_wall_ratio',        # lumen_area / wall_area
    'wall_fraction',           # wall_area / outer_area
]

# Diameter/Size Features (4 features + 6 scale-invariant features)
DIAMETER_SIZE_FEATURE_NAMES = [
    'outer_diameter_um',       # Outer diameter in microns
    'inner_diameter_um',       # Inner diameter in microns
    'diameter_ratio',          # inner/outer diameter ratio (scale-invariant)
    'hydraulic_diameter',      # 4 * area / perimeter (flow characteristic)
]

# Log-transformed size features for scale-invariant ML (3 features)
# These reduce the impact of size on classification
LOG_SIZE_FEATURE_NAMES = [
    'log_area',                # log(outer_area) - reduces size bias
    'log_diameter',            # log(outer_diameter_um) - reduces size bias
    'log_perimeter',           # log(outer_perimeter) - reduces size bias
]

# Size class categorical feature (1 feature, encoded as numeric)
SIZE_CLASS_FEATURE_NAMES = [
    'size_class',              # 0=capillary(<10um), 1=arteriole(10-50um), 2=small_artery(50-150um), 3=artery(>150um)
]

# Shape Features (5 features)
SHAPE_FEATURE_NAMES = [
    'circularity',             # 4*pi*area/perimeter^2
    'ellipticity',             # 1 - (minor_axis/major_axis)
    'convexity',               # area / convex_hull_area
    'roughness',               # perimeter / convex_hull_perimeter
    'compactness',             # perimeter^2 / area
]

# Intensity/Gradient Features from SMA channel (6 features)
INTENSITY_GRADIENT_FEATURE_NAMES = [
    'wall_intensity_mean',     # Mean SMA intensity in wall region
    'wall_intensity_std',      # Std of SMA intensity in wall
    'lumen_intensity_mean',    # Mean intensity in lumen (should be low)
    'wall_lumen_contrast',     # (wall_mean - lumen_mean) / wall_mean
    'edge_gradient_mean',      # Mean gradient magnitude at outer boundary
    'edge_gradient_std',       # Edge sharpness consistency
]

# Context Features (2 features)
CONTEXT_FEATURE_NAMES = [
    'background_intensity',    # Mean intensity in region just outside vessel
    'wall_background_contrast', # Contrast between wall and background
]

# Additional derived features (5 features)
DERIVED_FEATURE_NAMES = [
    'wall_thickness_range',    # max - min wall thickness
    'wall_eccentricity',       # Measure of wall shape eccentricity
    'lumen_circularity',       # Circularity of lumen contour
    'center_offset',           # Distance between outer and inner centers
    'wall_coverage',           # Fraction of outer perimeter covered by wall
]

# Complete list of vessel-specific feature names
VESSEL_FEATURE_NAMES = (
    RING_WALL_FEATURE_NAMES +
    DIAMETER_SIZE_FEATURE_NAMES +
    LOG_SIZE_FEATURE_NAMES +         # New: log-transformed size features
    SIZE_CLASS_FEATURE_NAMES +       # New: size class categorical
    SHAPE_FEATURE_NAMES +
    INTENSITY_GRADIENT_FEATURE_NAMES +
    CONTEXT_FEATURE_NAMES +
    DERIVED_FEATURE_NAMES
)

VESSEL_FEATURE_COUNT = len(VESSEL_FEATURE_NAMES)  # Should be ~32 features

# Size class thresholds (in microns) for vessel categorization
# Based on standard vascular biology classifications
SIZE_CLASS_THRESHOLDS = {
    'capillary': (0, 10),        # 5-10 um (class 0)
    'arteriole': (10, 50),       # 10-50 um (class 1)
    'small_artery': (50, 150),   # 50-150 um (class 2)
    'artery': (150, 1e9)            # >150 um (class 3)
}

def get_size_class(diameter_um: float) -> int:
    """
    Get size class for a vessel based on outer diameter.

    Args:
        diameter_um: Outer diameter in microns

    Returns:
        Size class (0=capillary, 1=arteriole, 2=small_artery, 3=artery)
    """
    if diameter_um < 10:
        return 0  # capillary
    elif diameter_um < 50:
        return 1  # arteriole
    elif diameter_um < 150:
        return 2  # small_artery
    else:
        return 3  # artery

def get_size_class_name(size_class: int) -> str:
    """
    Get human-readable name for size class.

    Args:
        size_class: Numeric size class (0-3)

    Returns:
        Size class name string
    """
    names = ['capillary', 'arteriole', 'small_artery', 'artery']
    return names[min(size_class, 3)]


# =============================================================================
# MAIN FEATURE EXTRACTION FUNCTION
# =============================================================================

def extract_vessel_features(
    wall_mask: np.ndarray,
    lumen_mask: Optional[np.ndarray],
    sma_channel: np.ndarray,
    outer_contour: np.ndarray,
    inner_contour: Optional[np.ndarray],
    pixel_size_um: float = 0.22,
    binary_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract comprehensive vessel-specific features.

    This function extracts ~28 features specifically designed for vessel
    classification, covering wall characteristics, shape, intensity,
    and contextual information.

    Args:
        wall_mask: Boolean mask of vessel wall region (outer - lumen)
        lumen_mask: Boolean mask of lumen region (may be None for partial vessels)
        sma_channel: SMA (smooth muscle actin) channel image (grayscale)
        outer_contour: Outer vessel boundary contour (Nx1x2 array)
        inner_contour: Inner (lumen) boundary contour (may be None for partial)
        pixel_size_um: Pixel size in microns for unit conversion
        binary_mask: Optional binary mask from segmentation (for ring completeness)

    Returns:
        Dictionary with all vessel-specific features. Missing features
        (e.g., for partial vessels) will have None values.

    Example:
        >>> features = extract_vessel_features(
        ...     wall_mask, lumen_mask, sma_channel,
        ...     outer_contour, inner_contour,
        ...     pixel_size_um=0.22
        ... )
        >>> print(features['wall_thickness_cv'])
        0.15
    """
    features = {}

    # Ensure SMA channel is grayscale float
    if sma_channel.ndim == 3:
        sma_gray = np.mean(sma_channel[:, :, :3], axis=2).astype(np.float32)
    else:
        sma_gray = sma_channel.astype(np.float32)

    # Normalize SMA channel to 0-255 range
    sma_min, sma_max = sma_gray.min(), sma_gray.max()
    if sma_max - sma_min > 1e-8:
        sma_norm = ((sma_gray - sma_min) / (sma_max - sma_min) * 255).astype(np.float32)
    else:
        sma_norm = np.zeros_like(sma_gray)

    # Extract feature categories
    ring_wall_features = _extract_ring_wall_features(
        wall_mask, lumen_mask, outer_contour, inner_contour,
        pixel_size_um, binary_mask
    )
    features.update(ring_wall_features)

    diameter_features = _extract_diameter_features(
        wall_mask, lumen_mask, outer_contour, inner_contour, pixel_size_um
    )
    features.update(diameter_features)

    shape_features = _extract_shape_features(
        wall_mask, outer_contour, inner_contour
    )
    features.update(shape_features)

    intensity_features = _extract_intensity_features(
        wall_mask, lumen_mask, sma_norm, outer_contour
    )
    features.update(intensity_features)

    context_features = _extract_context_features(
        wall_mask, sma_norm, outer_contour
    )
    features.update(context_features)

    derived_features = _extract_derived_features(
        wall_mask, lumen_mask, outer_contour, inner_contour, pixel_size_um
    )
    features.update(derived_features)

    return features


# =============================================================================
# RING/WALL FEATURE EXTRACTION
# =============================================================================

def _extract_ring_wall_features(
    wall_mask: np.ndarray,
    lumen_mask: Optional[np.ndarray],
    outer_contour: np.ndarray,
    inner_contour: Optional[np.ndarray],
    pixel_size_um: float,
    binary_mask: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """
    Extract ring/wall integrity and uniformity features.

    These features characterize how complete and uniform the vessel wall is,
    which is important for distinguishing real vessels from artifacts.
    """
    features = {
        'ring_completeness': None,
        'wall_thickness_cv': None,
        'wall_asymmetry': None,
        'lumen_wall_ratio': None,
        'wall_fraction': None,
    }

    # Wall and outer areas
    wall_area = float(wall_mask.sum())
    if wall_area < 10:
        return features

    outer_area = wall_area
    if lumen_mask is not None:
        lumen_area = float(lumen_mask.sum())
        outer_area = wall_area + lumen_area
        features['lumen_wall_ratio'] = lumen_area / wall_area if wall_area > 0 else None

    features['wall_fraction'] = wall_area / outer_area if outer_area > 0 else None

    # Ring completeness - fraction of perimeter with SMA signal
    if binary_mask is not None and inner_contour is not None:
        features['ring_completeness'] = _compute_ring_completeness(
            outer_contour, inner_contour, binary_mask
        )

    # Wall thickness measurements
    if inner_contour is not None and lumen_mask is not None:
        wall_thicknesses = _measure_wall_thickness(
            wall_mask, lumen_mask, inner_contour, pixel_size_um
        )

        if len(wall_thicknesses) >= 5:
            mean_thickness = np.mean(wall_thicknesses)
            std_thickness = np.std(wall_thicknesses)
            min_thickness = np.min(wall_thicknesses)
            max_thickness = np.max(wall_thicknesses)

            # Coefficient of variation (std/mean, lower = more uniform = vessel-like)
            features['wall_thickness_cv'] = std_thickness / mean_thickness if mean_thickness > 0 else None

            # Wall asymmetry: max/min ratio
            features['wall_asymmetry'] = max_thickness / min_thickness if min_thickness > 0 else None

    return features


def _compute_ring_completeness(
    outer_contour: np.ndarray,
    inner_contour: np.ndarray,
    binary_mask: np.ndarray,
    num_samples: int = 72,
    pixel_size_um: float = 0.22,
) -> float:
    """
    Compute the fraction of the vessel perimeter that has SMA signal.

    Samples points around the mid-wall region and checks how many
    are positive in the binary mask.

    SIZE-ADAPTIVE SAMPLING:
    The number of sample points is now adapted based on vessel perimeter to
    ensure consistent sampling density across all vessel sizes:
    - Target: ~1 sample per 2 pixels of perimeter (or ~0.5um at 0.22um/px)
    - Minimum: 24 samples (for very small vessels)
    - Maximum: 360 samples (for very large vessels)

    This prevents size bias where:
    - Small vessels might have overlapping samples (biasing toward positive)
    - Large vessels might have gaps in sampling (biasing toward negative)
    """
    try:
        outer_ellipse = cv2.fitEllipse(outer_contour)
        inner_ellipse = cv2.fitEllipse(inner_contour)
    except cv2.error:
        return 0.0

    (cx_out, cy_out), (minor_out, major_out), angle_out = outer_ellipse
    (cx_in, cy_in), (minor_in, major_in), angle_in = inner_ellipse

    # SIZE-ADAPTIVE: Calculate perimeter and adapt sample count
    # Use average of outer and inner perimeter (mid-wall perimeter)
    avg_diameter = (max(major_out, minor_out) + max(major_in, minor_in)) / 2
    approx_perimeter_px = np.pi * avg_diameter  # Approximate perimeter

    # Target: 1 sample per 2 pixels of perimeter for consistent density
    adaptive_samples = int(approx_perimeter_px / 2)

    # Clamp to reasonable range
    MIN_SAMPLES = 24   # Minimum for very small vessels (~5um capillaries)
    MAX_SAMPLES = 360  # Maximum for very large vessels (~500um arteries)
    adaptive_samples = max(MIN_SAMPLES, min(MAX_SAMPLES, adaptive_samples))

    # Use adaptive samples instead of fixed num_samples
    actual_samples = adaptive_samples

    ring_points = 0
    ring_positive = 0

    for theta in np.linspace(0, 2 * np.pi, actual_samples, endpoint=False):
        # Mid-wall radius calculation
        a_out, b_out = major_out / 2, minor_out / 2
        a_in, b_in = major_in / 2, minor_in / 2
        angle_out_rad = np.radians(angle_out)

        cos_t = np.cos(theta - angle_out_rad)
        sin_t = np.sin(theta - angle_out_rad)

        r_out = (a_out * b_out) / np.sqrt((b_out * cos_t) ** 2 + (a_out * sin_t) ** 2 + 1e-8)
        r_in = (a_in * b_in) / np.sqrt((b_in * cos_t) ** 2 + (a_in * sin_t) ** 2 + 1e-8)
        r_mid = (r_out + r_in) / 2

        # Point at mid-wall
        px = int(cx_out + r_mid * np.cos(theta))
        py = int(cy_out + r_mid * np.sin(theta))

        if 0 <= py < binary_mask.shape[0] and 0 <= px < binary_mask.shape[1]:
            ring_points += 1
            if binary_mask[py, px] > 0:
                ring_positive += 1

    return ring_positive / (ring_points + 1e-8)


def _measure_wall_thickness(
    wall_mask: np.ndarray,
    lumen_mask: np.ndarray,
    inner_contour: np.ndarray,
    pixel_size_um: float,
    num_samples: int = 36,
) -> np.ndarray:
    """
    Measure wall thickness at multiple points around the vessel.

    Two measurement methods are used independently:

    1. **Contour-sampling** (primary): Samples evenly-spaced points along the
       inner contour and reads the distance-transform value at each point.
       Each value is the distance from the lumen boundary outward into the
       wall -- a one-sided (radial) thickness estimate.

    2. **Skeleton-based** (fallback): Skeletonises the wall mask and reads the
       distance-transform at each skeleton pixel, then doubles it to obtain a
       full wall-width (diameter) estimate.  Because this measures a
       fundamentally different quantity (full width vs. one-sided distance),
       the skeleton values are only used when contour-sampling returns fewer
       than ``num_samples // 2`` measurements.

    Returns:
        1-D array of wall thickness values in micrometers.  All values in a
        single call come from the same measurement method to keep statistics
        (mean, std, CV, asymmetry) internally consistent.
    """
    h, w = wall_mask.shape

    # Distance from lumen boundary into wall
    dist_from_lumen = distance_transform_edt(~lumen_mask)

    # ---- Method 1: contour-sampling (one-sided radial distance) ----
    contour_thicknesses: list = []
    step = max(1, len(inner_contour) // num_samples)

    for pt in inner_contour[::step]:
        px, py = pt[0]
        if 0 <= py < h and 0 <= px < w:
            if wall_mask[py, px] or lumen_mask[py, px]:
                ray_dist = dist_from_lumen[py, px]
                if ray_dist > 0:
                    contour_thicknesses.append(ray_dist * pixel_size_um)

    # If contour-sampling produced enough points, return those directly.
    min_required = max(num_samples // 2, 3)
    if len(contour_thicknesses) >= min_required:
        return np.array(contour_thicknesses)

    # ---- Method 2: skeleton-based (full wall-width estimate, fallback) ----
    try:
        skeleton = skeletonize(wall_mask)
        skeleton_distances = dist_from_lumen[skeleton]
        if len(skeleton_distances) > 0:
            # Multiply by 2 because skeleton sits at the medial axis, so the
            # distance-transform value is roughly half the total wall width.
            medial_thicknesses = skeleton_distances * 2 * pixel_size_um
            return medial_thicknesses
    except Exception:
        pass

    # Return whatever contour-sampling collected (may be empty)
    return np.array(contour_thicknesses) if contour_thicknesses else np.array([])


# =============================================================================
# DIAMETER/SIZE FEATURE EXTRACTION
# =============================================================================

def _extract_diameter_features(
    wall_mask: np.ndarray,
    lumen_mask: Optional[np.ndarray],
    outer_contour: np.ndarray,
    inner_contour: Optional[np.ndarray],
    pixel_size_um: float,
) -> Dict[str, Optional[float]]:
    """
    Extract diameter and size-related features.

    These features capture the size characteristics of the vessel,
    which correlate with vessel type (capillary, arteriole, artery).

    Includes log-transformed features and size class for scale-invariant ML.
    """
    features = {
        'outer_diameter_um': None,
        'inner_diameter_um': None,
        'diameter_ratio': None,
        'hydraulic_diameter': None,
        # Log-transformed features (scale-invariant)
        'log_area': None,
        'log_diameter': None,
        'log_perimeter': None,
        # Size class categorical feature
        'size_class': None,
    }

    # Fit ellipse to outer contour
    if len(outer_contour) >= 5:
        try:
            outer_ellipse = cv2.fitEllipse(outer_contour)
            (_, _), (minor_out, major_out), _ = outer_ellipse
            outer_diameter_um = max(major_out, minor_out) * pixel_size_um
            features['outer_diameter_um'] = float(outer_diameter_um)
        except cv2.error:
            pass

    # Fit ellipse to inner contour
    if inner_contour is not None and len(inner_contour) >= 5:
        try:
            inner_ellipse = cv2.fitEllipse(inner_contour)
            (_, _), (minor_in, major_in), _ = inner_ellipse
            inner_diameter_um = max(major_in, minor_in) * pixel_size_um
            features['inner_diameter_um'] = float(inner_diameter_um)

            # Diameter ratio
            if features['outer_diameter_um'] is not None and features['outer_diameter_um'] > 0:
                features['diameter_ratio'] = inner_diameter_um / features['outer_diameter_um']
        except cv2.error:
            pass

    # Hydraulic diameter: 4 * area / perimeter (flow characteristic)
    # For the lumen (where blood flows)
    if lumen_mask is not None:
        lumen_area = float(lumen_mask.sum()) * pixel_size_um ** 2
        if inner_contour is not None:
            lumen_perimeter = cv2.arcLength(inner_contour, True) * pixel_size_um
            if lumen_perimeter > 0:
                features['hydraulic_diameter'] = 4 * lumen_area / lumen_perimeter

    # =========================================================================
    # LOG-TRANSFORMED SIZE FEATURES (Scale-invariant for ML)
    # =========================================================================
    # These features reduce size bias in classifiers by compressing the scale
    # Log transform makes differences between small vessels proportionally
    # similar to differences between large vessels

    # Calculate outer area and perimeter for log features
    outer_area = cv2.contourArea(outer_contour)
    outer_area_um2 = outer_area * pixel_size_um ** 2
    outer_perimeter = cv2.arcLength(outer_contour, True)
    outer_perimeter_um = outer_perimeter * pixel_size_um

    # Log area (add 1 to avoid log(0))
    if outer_area_um2 > 0:
        features['log_area'] = float(np.log(outer_area_um2 + 1))

    # Log diameter
    if features['outer_diameter_um'] is not None and features['outer_diameter_um'] > 0:
        features['log_diameter'] = float(np.log(features['outer_diameter_um'] + 1))

    # Log perimeter
    if outer_perimeter_um > 0:
        features['log_perimeter'] = float(np.log(outer_perimeter_um + 1))

    # =========================================================================
    # SIZE CLASS CATEGORICAL FEATURE
    # =========================================================================
    # Explicit size binning helps RF classifiers and enables stratified sampling
    # Classes: 0=capillary(<10um), 1=arteriole(10-50um), 2=small_artery(50-150um), 3=artery(>150um)

    if features['outer_diameter_um'] is not None:
        features['size_class'] = float(get_size_class(features['outer_diameter_um']))

    return features


# =============================================================================
# SHAPE FEATURE EXTRACTION
# =============================================================================

def _extract_shape_features(
    wall_mask: np.ndarray,
    outer_contour: np.ndarray,
    inner_contour: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """
    Extract geometric shape descriptors.

    These features characterize the shape of the vessel cross-section,
    which helps distinguish vessels from other circular structures.
    """
    features = {
        'circularity': None,
        'ellipticity': None,
        'convexity': None,
        'roughness': None,
        'compactness': None,
    }

    # Outer contour metrics
    outer_area = cv2.contourArea(outer_contour)
    outer_perimeter = cv2.arcLength(outer_contour, True)

    if outer_area < 10 or outer_perimeter < 1:
        return features

    # Circularity: 4*pi*area/perimeter^2 (1.0 = perfect circle)
    features['circularity'] = 4 * np.pi * outer_area / (outer_perimeter ** 2)

    # Compactness: perimeter^2 / area (inverse of circularity * 4*pi)
    features['compactness'] = (outer_perimeter ** 2) / outer_area

    # Ellipticity: 1 - (minor_axis/major_axis)
    if len(outer_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(outer_contour)
            (_, _), (minor_ax, major_ax), _ = ellipse
            if major_ax > 0:
                features['ellipticity'] = 1.0 - (min(minor_ax, major_ax) / max(minor_ax, major_ax))
        except cv2.error:
            pass

    # Convexity and roughness using convex hull
    try:
        points = outer_contour.reshape(-1, 2)
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, volume is area

            # Get hull perimeter by summing edge lengths
            hull_vertices = points[hull.vertices]
            hull_perimeter = 0.0
            for i in range(len(hull_vertices)):
                p1 = hull_vertices[i]
                p2 = hull_vertices[(i + 1) % len(hull_vertices)]
                hull_perimeter += np.sqrt(np.sum((p2 - p1) ** 2))

            # Convexity: area / convex_hull_area (1.0 = convex)
            if hull_area > 0:
                features['convexity'] = outer_area / hull_area

            # Roughness: perimeter / convex_hull_perimeter (1.0 = smooth)
            if hull_perimeter > 0:
                features['roughness'] = outer_perimeter / hull_perimeter
    except Exception:
        pass

    return features


# =============================================================================
# INTENSITY/GRADIENT FEATURE EXTRACTION
# =============================================================================

def _extract_intensity_features(
    wall_mask: np.ndarray,
    lumen_mask: Optional[np.ndarray],
    sma_channel: np.ndarray,
    outer_contour: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Extract intensity and gradient features from SMA channel.

    These features characterize the SMA staining pattern, which is
    critical for vessel identification and type classification.
    """
    features = {
        'wall_intensity_mean': None,
        'wall_intensity_std': None,
        'lumen_intensity_mean': None,
        'wall_lumen_contrast': None,
        'edge_gradient_mean': None,
        'edge_gradient_std': None,
    }

    # Wall intensity statistics
    wall_pixels = sma_channel[wall_mask]
    if len(wall_pixels) > 0:
        features['wall_intensity_mean'] = float(np.mean(wall_pixels))
        features['wall_intensity_std'] = float(np.std(wall_pixels))

    # Lumen intensity (should be low for real vessels)
    if lumen_mask is not None:
        lumen_pixels = sma_channel[lumen_mask]
        if len(lumen_pixels) > 0:
            features['lumen_intensity_mean'] = float(np.mean(lumen_pixels))

            # Wall-lumen contrast
            if features['wall_intensity_mean'] is not None and features['wall_intensity_mean'] > 0:
                features['wall_lumen_contrast'] = (
                    (features['wall_intensity_mean'] - features['lumen_intensity_mean']) /
                    features['wall_intensity_mean']
                )

    # Edge gradient at outer boundary
    edge_gradients = _compute_edge_gradients(sma_channel, outer_contour)
    if len(edge_gradients) > 0:
        features['edge_gradient_mean'] = float(np.mean(edge_gradients))
        features['edge_gradient_std'] = float(np.std(edge_gradients))

    return features


def _compute_edge_gradients(
    image: np.ndarray,
    contour: np.ndarray,
    sample_step: int = 5,
) -> np.ndarray:
    """
    Compute gradient magnitudes along the contour boundary.

    Uses Sobel filters to compute gradient magnitude at points
    along the contour.
    """
    h, w = image.shape[:2]

    # Compute gradient magnitude using Sobel
    grad_x = sobel(image, axis=1)
    grad_y = sobel(image, axis=0)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Sample gradient at contour points
    gradients = []
    points = contour.reshape(-1, 2)

    for i in range(0, len(points), sample_step):
        px, py = points[i]
        if 0 <= py < h and 0 <= px < w:
            gradients.append(grad_mag[int(py), int(px)])

    return np.array(gradients)


# =============================================================================
# CONTEXT FEATURE EXTRACTION
# =============================================================================

def _extract_context_features(
    wall_mask: np.ndarray,
    sma_channel: np.ndarray,
    outer_contour: np.ndarray,
    background_margin: int = 10,
) -> Dict[str, Optional[float]]:
    """
    Extract contextual features from the region around the vessel.

    These features characterize the background and contrast,
    helping distinguish vessels from similarly shaped artifacts.
    """
    features = {
        'background_intensity': None,
        'wall_background_contrast': None,
    }

    h, w = sma_channel.shape[:2]

    # Create background mask: dilated outer region minus wall
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(outer_mask, [outer_contour], 0, 255, -1)

    # Dilate to get region just outside vessel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_margin * 2, background_margin * 2))
    dilated = cv2.dilate(outer_mask, kernel)

    # Background is dilated minus original
    background_mask = (dilated > 0) & (outer_mask == 0)

    background_pixels = sma_channel[background_mask]
    if len(background_pixels) > 0:
        features['background_intensity'] = float(np.mean(background_pixels))

        # Wall-background contrast
        wall_pixels = sma_channel[wall_mask]
        if len(wall_pixels) > 0:
            wall_mean = np.mean(wall_pixels)
            bg_mean = features['background_intensity']
            if bg_mean > 0:
                features['wall_background_contrast'] = (wall_mean - bg_mean) / bg_mean

    return features


# =============================================================================
# DERIVED FEATURE EXTRACTION
# =============================================================================

def _extract_derived_features(
    wall_mask: np.ndarray,
    lumen_mask: Optional[np.ndarray],
    outer_contour: np.ndarray,
    inner_contour: Optional[np.ndarray],
    pixel_size_um: float,
) -> Dict[str, Optional[float]]:
    """
    Extract additional derived features.

    These features combine multiple measurements to create
    more discriminative features for classification.
    """
    features = {
        'wall_thickness_range': None,
        'wall_eccentricity': None,
        'lumen_circularity': None,
        'center_offset': None,
        'wall_coverage': None,
    }

    # Wall thickness range
    if inner_contour is not None and lumen_mask is not None:
        wall_thicknesses = _measure_wall_thickness(
            wall_mask, lumen_mask, inner_contour, pixel_size_um
        )
        if len(wall_thicknesses) >= 2:
            features['wall_thickness_range'] = float(np.max(wall_thicknesses) - np.min(wall_thicknesses))

    # Wall eccentricity using ellipse fit
    if len(outer_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(outer_contour)
            (_, _), (minor_ax, major_ax), _ = ellipse
            if major_ax > 0 and minor_ax > 0:
                # Eccentricity = sqrt(1 - (b/a)^2) where a >= b
                a = max(major_ax, minor_ax) / 2
                b = min(major_ax, minor_ax) / 2
                features['wall_eccentricity'] = float(np.sqrt(1 - (b / a) ** 2))
        except cv2.error:
            pass

    # Lumen circularity
    if inner_contour is not None and len(inner_contour) >= 5:
        lumen_area = cv2.contourArea(inner_contour)
        lumen_perimeter = cv2.arcLength(inner_contour, True)
        if lumen_perimeter > 0:
            features['lumen_circularity'] = 4 * np.pi * lumen_area / (lumen_perimeter ** 2)

    # Center offset: distance between outer and inner centers
    if inner_contour is not None and len(outer_contour) >= 5 and len(inner_contour) >= 5:
        try:
            outer_ellipse = cv2.fitEllipse(outer_contour)
            inner_ellipse = cv2.fitEllipse(inner_contour)

            (cx_out, cy_out), _, _ = outer_ellipse
            (cx_in, cy_in), _, _ = inner_ellipse

            offset_px = np.sqrt((cx_out - cx_in) ** 2 + (cy_out - cy_in) ** 2)
            features['center_offset'] = float(offset_px * pixel_size_um)
        except cv2.error:
            pass

    # Wall coverage: what fraction of outer perimeter has wall
    if inner_contour is not None:
        outer_perimeter = cv2.arcLength(outer_contour, True)
        if outer_perimeter > 0:
            # Estimate coverage from wall area relative to expected thin ring
            outer_area = cv2.contourArea(outer_contour)
            inner_area = cv2.contourArea(inner_contour)
            wall_area = outer_area - inner_area

            # Expected wall area for complete ring (rough approximation)
            avg_radius = np.sqrt(outer_area / np.pi)
            expected_thickness = avg_radius * 0.1  # Assume 10% of radius
            expected_wall_area = np.pi * avg_radius ** 2 - np.pi * (avg_radius - expected_thickness) ** 2

            if expected_wall_area > 0:
                features['wall_coverage'] = min(1.0, wall_area / expected_wall_area)

    return features


# =============================================================================
# BATCH EXTRACTION HELPER
# =============================================================================

def extract_vessel_features_batch(
    candidates: List[Dict[str, Any]],
    tile: np.ndarray,
    pixel_size_um: float = 0.22,
) -> List[Dict[str, float]]:
    """
    Extract vessel features for multiple candidates in batch.

    This is a convenience function for processing multiple vessel
    candidates at once.

    Args:
        candidates: List of candidate dicts, each with:
            - 'outer': Outer contour
            - 'inner': Inner contour (optional)
            - 'binary': Binary mask (optional)
        tile: Image tile (SMA channel)
        pixel_size_um: Pixel size in microns

    Returns:
        List of feature dictionaries, one per candidate.
    """
    results = []
    h, w = tile.shape[:2]

    for cand in candidates:
        outer = cand['outer']
        inner = cand.get('inner')
        binary = cand.get('binary')

        # Create masks (use uint8 for cv2.drawContours, then convert to bool)
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        lumen_mask = None

        cv2.drawContours(wall_mask, [outer], 0, 1, -1)
        wall_mask = wall_mask.astype(bool)

        if inner is not None:
            lumen_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(lumen_mask, [inner], 0, 1, -1)
            lumen_mask = lumen_mask.astype(bool)

            # Wall is outer minus lumen
            wall_mask = wall_mask & ~lumen_mask

        # Extract features
        try:
            features = extract_vessel_features(
                wall_mask=wall_mask,
                lumen_mask=lumen_mask,
                sma_channel=tile,
                outer_contour=outer,
                inner_contour=inner,
                pixel_size_um=pixel_size_um,
                binary_mask=binary,
            )
            results.append(features)
        except Exception as e:
            logger.debug(f"Failed to extract vessel features: {e}")
            results.append({name: None for name in VESSEL_FEATURE_NAMES})

    return results


# =============================================================================
# FEATURE VECTOR CONVERSION
# =============================================================================

def vessel_features_to_vector(
    features: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Convert vessel feature dict to fixed-length numeric vector.

    This is useful for machine learning where a consistent feature
    vector is needed.

    Args:
        features: Feature dictionary from extract_vessel_features
        feature_names: List of feature names to include (default: VESSEL_FEATURE_NAMES)

    Returns:
        1D numpy array of feature values (None values become 0.0)
    """
    if feature_names is None:
        feature_names = VESSEL_FEATURE_NAMES

    vector = []
    for name in feature_names:
        val = features.get(name)
        if val is None or (isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val))):
            vector.append(0.0)
        else:
            vector.append(float(val))

    return np.array(vector)


# =============================================================================
# MULTI-CHANNEL FEATURE EXTRACTION
# =============================================================================

# Channel name mapping for common 4-channel vessel slides
# Note: This is slide-specific; actual mapping should be passed from CZI metadata
DEFAULT_CHANNEL_NAMES = {
    0: 'nuclear',    # AF488 - Nuclear stain
    1: 'sma',        # AF647 - Smooth muscle actin (primary detection)
    2: 'pm',         # AF750 - Plasma membrane marker
    3: 'cd31',       # AF555 - CD31 endothelial marker
}

# Multi-channel feature names (per channel)
MULTICHANNEL_WALL_FEATURES = [
    'wall_intensity_mean',
    'wall_intensity_std',
    'wall_intensity_median',
    'wall_intensity_cv',
]

MULTICHANNEL_LUMEN_FEATURES = [
    'lumen_intensity_mean',
    'lumen_intensity_std',
    'lumen_intensity_median',
    'lumen_intensity_cv',
]

# Cross-channel ratio feature names
CROSS_CHANNEL_RATIO_NAMES = [
    'sma_cd31_wall_ratio',        # SMA/CD31 in wall (high for muscular vessels)
    'sma_nuclear_wall_ratio',     # SMA/nuclear in wall
    'cd31_nuclear_lumen_ratio',   # CD31/nuclear at lumen edge
    'sma_pm_wall_ratio',          # SMA/PM in wall
    'cd31_lumen_wall_ratio',      # CD31 lumen vs wall (high if CD31 at lumen edge)
    'sma_wall_lumen_contrast',    # SMA contrast between wall and lumen
]


def extract_multichannel_intensity_features(
    wall_mask: np.ndarray,
    lumen_mask: Optional[np.ndarray],
    channels_data: Dict[int, np.ndarray],
    channel_names: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """
    Extract intensity features from multiple channels.

    For each channel, extracts intensity statistics from both the wall
    and lumen regions, enabling channel-specific characterization of
    the vessel.

    Args:
        wall_mask: Boolean mask of vessel wall region
        lumen_mask: Boolean mask of lumen region (optional)
        channels_data: Dict mapping channel index to channel image array
        channel_names: Optional dict mapping channel index to name
                      (default: DEFAULT_CHANNEL_NAMES)

    Returns:
        Dictionary with per-channel intensity features, e.g.:
        {
            'ch0_wall_intensity_mean': 45.2,
            'ch0_wall_intensity_std': 12.3,
            'nuclear_wall_intensity_mean': 45.2,  # Named alias
            ...
        }
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    features = {}

    for ch_idx, ch_data in channels_data.items():
        ch_name = channel_names.get(ch_idx, f'ch{ch_idx}')

        # Ensure channel data matches mask dimensions
        if ch_data.shape[:2] != wall_mask.shape[:2]:
            logger.debug(f"Channel {ch_idx} shape mismatch: {ch_data.shape} vs {wall_mask.shape}")
            continue

        # Convert to grayscale if needed
        if ch_data.ndim == 3:
            ch_gray = np.mean(ch_data, axis=2).astype(np.float32)
        else:
            ch_gray = ch_data.astype(np.float32)

        # Wall intensity features
        wall_pixels = ch_gray[wall_mask]
        if len(wall_pixels) > 0:
            mean_val = float(np.mean(wall_pixels))
            std_val = float(np.std(wall_pixels))
            median_val = float(np.median(wall_pixels))
            cv_val = std_val / mean_val if mean_val > 0 else 0.0

            # Store with numeric index prefix (always)
            features[f'ch{ch_idx}_wall_mean'] = mean_val
            features[f'ch{ch_idx}_wall_std'] = std_val
            features[f'ch{ch_idx}_wall_median'] = median_val
            features[f'ch{ch_idx}_wall_cv'] = cv_val

            # Store with named prefix (for convenience)
            features[f'{ch_name}_wall_mean'] = mean_val
            features[f'{ch_name}_wall_std'] = std_val
            features[f'{ch_name}_wall_median'] = median_val
            features[f'{ch_name}_wall_cv'] = cv_val

        # Lumen intensity features
        if lumen_mask is not None:
            lumen_pixels = ch_gray[lumen_mask]
            if len(lumen_pixels) > 0:
                mean_val = float(np.mean(lumen_pixels))
                std_val = float(np.std(lumen_pixels))
                median_val = float(np.median(lumen_pixels))
                cv_val = std_val / mean_val if mean_val > 0 else 0.0

                features[f'ch{ch_idx}_lumen_mean'] = mean_val
                features[f'ch{ch_idx}_lumen_std'] = std_val
                features[f'ch{ch_idx}_lumen_median'] = median_val
                features[f'ch{ch_idx}_lumen_cv'] = cv_val

                features[f'{ch_name}_lumen_mean'] = mean_val
                features[f'{ch_name}_lumen_std'] = std_val
                features[f'{ch_name}_lumen_median'] = median_val
                features[f'{ch_name}_lumen_cv'] = cv_val

    return features


def compute_channel_ratios(
    multichannel_features: Dict[str, float],
    channel_names: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """
    Compute biologically meaningful ratios between channels.

    These ratios help characterize vessel type and validate detection:
    - SMA/CD31: High in muscular arteries, lower in capillaries
    - CD31 lumen/wall: High if endothelium is at lumen boundary (validates lumen)
    - SMA wall/lumen contrast: High indicates clear vessel wall

    Args:
        multichannel_features: Dict with per-channel features from
                              extract_multichannel_intensity_features
        channel_names: Optional channel name mapping

    Returns:
        Dictionary with cross-channel ratio features
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNEL_NAMES

    ratios = {}

    def safe_ratio(num_key: str, denom_key: str) -> Optional[float]:
        """Compute ratio safely, returning None if denominator is 0."""
        num = multichannel_features.get(num_key, 0)
        denom = multichannel_features.get(denom_key, 0)
        if denom is None or denom == 0:
            return None
        return float(num / denom) if num is not None else None

    # SMA / CD31 ratio in wall (high for muscular arteries)
    ratios['sma_cd31_wall_ratio'] = safe_ratio('sma_wall_mean', 'cd31_wall_mean')

    # SMA / nuclear ratio in wall
    ratios['sma_nuclear_wall_ratio'] = safe_ratio('sma_wall_mean', 'nuclear_wall_mean')

    # CD31 / nuclear ratio in lumen (endothelial presence at lumen)
    ratios['cd31_nuclear_lumen_ratio'] = safe_ratio('cd31_lumen_mean', 'nuclear_lumen_mean')

    # SMA / PM ratio in wall
    ratios['sma_pm_wall_ratio'] = safe_ratio('sma_wall_mean', 'pm_wall_mean')

    # CD31 lumen vs wall ratio (higher if CD31 concentrated at lumen edge)
    cd31_lumen = multichannel_features.get('cd31_lumen_mean', 0)
    cd31_wall = multichannel_features.get('cd31_wall_mean', 0)
    if cd31_wall is not None and cd31_wall > 0:
        ratios['cd31_lumen_wall_ratio'] = float(cd31_lumen / cd31_wall) if cd31_lumen is not None else None
    else:
        ratios['cd31_lumen_wall_ratio'] = None

    # SMA wall-lumen contrast
    sma_wall = multichannel_features.get('sma_wall_mean', 0)
    sma_lumen = multichannel_features.get('sma_lumen_mean', 0)
    if sma_wall is not None and sma_wall > 0 and sma_lumen is not None:
        ratios['sma_wall_lumen_contrast'] = float((sma_wall - sma_lumen) / sma_wall)
    else:
        ratios['sma_wall_lumen_contrast'] = None

    return ratios


def extract_all_vessel_features_multichannel(
    wall_mask: np.ndarray,
    lumen_mask: Optional[np.ndarray],
    sma_channel: np.ndarray,
    outer_contour: np.ndarray,
    inner_contour: Optional[np.ndarray],
    pixel_size_um: float,
    channels_data: Optional[Dict[int, np.ndarray]] = None,
    channel_names: Optional[Dict[int, str]] = None,
    binary_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract complete vessel feature set including multi-channel features.

    This is the main entry point for multi-channel vessel feature extraction.
    It combines:
    1. Standard vessel features from SMA channel (~32 features)
    2. Multi-channel intensity features (~8 features × n_channels)
    3. Cross-channel ratios (~6 features)

    Args:
        wall_mask: Boolean mask of vessel wall
        lumen_mask: Boolean mask of lumen (optional)
        sma_channel: SMA (primary detection) channel image
        outer_contour: Outer vessel boundary contour
        inner_contour: Inner (lumen) boundary contour (optional)
        pixel_size_um: Pixel size in microns
        channels_data: Dict mapping channel index to image array (optional)
        channel_names: Dict mapping channel index to name (optional)
        binary_mask: Binary mask for ring completeness (optional)

    Returns:
        Dictionary with all vessel features (standard + multichannel + ratios)

    Example:
        >>> features = extract_all_vessel_features_multichannel(
        ...     wall_mask, lumen_mask, sma_channel,
        ...     outer_contour, inner_contour, 0.22,
        ...     channels_data={0: nuc, 1: sma, 2: pm, 3: cd31},
        ...     channel_names={0: 'nuclear', 1: 'sma', 2: 'pm', 3: 'cd31'}
        ... )
    """
    # Extract standard vessel features from SMA channel
    features = extract_vessel_features(
        wall_mask=wall_mask,
        lumen_mask=lumen_mask,
        sma_channel=sma_channel,
        outer_contour=outer_contour,
        inner_contour=inner_contour,
        pixel_size_um=pixel_size_um,
        binary_mask=binary_mask,
    )

    # If multi-channel data provided, extract additional features
    if channels_data is not None and len(channels_data) > 0:
        # Multi-channel intensity features
        multichannel_features = extract_multichannel_intensity_features(
            wall_mask=wall_mask,
            lumen_mask=lumen_mask,
            channels_data=channels_data,
            channel_names=channel_names,
        )
        features.update(multichannel_features)

        # Cross-channel ratios
        channel_ratios = compute_channel_ratios(
            multichannel_features=multichannel_features,
            channel_names=channel_names,
        )
        features.update(channel_ratios)

    return features


def get_vessel_feature_importance() -> Dict[str, str]:
    """
    Get descriptions of feature importance for documentation.

    Returns:
        Dictionary mapping feature names to their importance descriptions.
    """
    return {
        # Ring/Wall features
        'ring_completeness': 'High for vessels (>0.7), low for artifacts. Key discriminator.',
        'wall_thickness_cv': 'CV of wall thickness. Low (<0.3) for vessels, high for irregular structures.',
        'wall_asymmetry': 'Ratio < 3.0 typical for vessels, higher indicates artifacts.',
        'lumen_wall_ratio': 'Typically 1.0-5.0 for vessels, varies by type.',
        'wall_fraction': 'Typically 0.1-0.5 for vessels.',

        # Diameter features
        'outer_diameter_um': 'Key for vessel type: <10 capillary, 10-100 arteriole, >100 artery.',
        'inner_diameter_um': 'Lumen size, correlates with flow capacity.',
        'diameter_ratio': 'Typically 0.6-0.9 for vessels.',
        'hydraulic_diameter': 'Flow characteristic, useful for physiology correlation.',

        # Shape features
        'circularity': 'High (>0.5) for cross-sections, low for longitudinal cuts.',
        'ellipticity': 'Low (<0.5) for round vessels, higher for elongated.',
        'convexity': 'High (>0.9) for clean vessels, lower for irregular shapes.',
        'roughness': 'Close to 1.0 for smooth boundaries.',
        'compactness': 'Inverse of circularity, useful for some classifiers.',

        # Intensity features
        'wall_intensity_mean': 'SMA staining intensity. Higher in muscular vessels.',
        'wall_intensity_std': 'Uniformity of staining. Low indicates consistent staining.',
        'lumen_intensity_mean': 'Should be low (<50) for real lumens.',
        'wall_lumen_contrast': 'High (>0.5) indicates clear lumen. Key for vessel ID.',
        'edge_gradient_mean': 'Sharp edges indicate well-defined boundaries.',
        'edge_gradient_std': 'Low indicates consistent edge sharpness.',

        # Context features
        'background_intensity': 'Reference for contrast calculations.',
        'wall_background_contrast': 'High indicates vessel stands out from background.',

        # Derived features
        'wall_thickness_range': 'Low (<5 um) indicates uniform wall.',
        'wall_eccentricity': 'Ellipse eccentricity, 0=circle, 1=line.',
        'lumen_circularity': 'Round lumens (>0.7) typical for cross-sections.',
        'center_offset': 'Low (<5 um) indicates concentric structure.',
        'wall_coverage': 'Close to 1.0 for complete rings.',
    }
