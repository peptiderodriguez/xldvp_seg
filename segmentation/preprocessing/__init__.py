"""
Preprocessing modules for microscopy image correction.

Includes:
- illumination: Photobleaching and illumination artifact correction, CLAHE
- vesselness: Frangi vesselness filtering for tubular/ring structure enhancement
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
]
