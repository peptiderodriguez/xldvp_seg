"""
Preprocessing modules for microscopy image correction.

Includes:
- illumination: Photobleaching and illumination artifact correction, CLAHE
- vesselness: Frangi vesselness filtering for tubular/ring structure enhancement
- stain_normalization: Cross-slide intensity normalization
"""

from .illumination import (
    normalize_rows_columns,
    morphological_background_subtraction,
    correct_photobleaching,
    estimate_band_severity,
    apply_clahe,
)

from .vesselness import (
    compute_vesselness,
    compute_vesselness_multiscale,
    enhance_ring_structures,
)

from .stain_normalization import (
    percentile_normalize_rgb,
    compute_global_percentiles,
    normalize_to_percentiles,
    normalize_slide_to_reference,
)

__all__ = [
    # Illumination correction
    'normalize_rows_columns',
    'morphological_background_subtraction',
    'correct_photobleaching',
    'estimate_band_severity',
    'apply_clahe',
    # Vesselness filtering
    'compute_vesselness',
    'compute_vesselness_multiscale',
    'enhance_ring_structures',
    # Stain normalization
    'percentile_normalize_rgb',
    'compute_global_percentiles',
    'normalize_to_percentiles',
    'normalize_slide_to_reference',
]
