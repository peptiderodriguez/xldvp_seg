"""
Detection algorithms for the segmentation pipeline.

Provides:
- Tissue detection using K-means variance analysis
- CellDetector: Unified detector with shared models (SAM2, Cellpose, ResNet)
- Detection strategies for different cell types (NMJ, MK, vessel, etc.)
"""

from .tissue import (
    calculate_block_variances,
    has_tissue,
    calibrate_tissue_threshold,
    filter_tissue_tiles,
)

from .cell_detector import (
    CellDetector,
    extract_morphological_features,
)

from .strategies import (
    DetectionStrategy,
    Detection,
    NMJStrategy,
)

from .registry import StrategyRegistry

__all__ = [
    # Tissue detection
    'calculate_block_variances',
    'has_tissue',
    'calibrate_tissue_threshold',
    'filter_tissue_tiles',
    # Unified cell detector
    'CellDetector',
    'extract_morphological_features',
    # Detection strategies
    'DetectionStrategy',
    'Detection',
    'NMJStrategy',
    # Strategy registry
    'StrategyRegistry',
]
