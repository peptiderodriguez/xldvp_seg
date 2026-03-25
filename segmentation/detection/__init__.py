"""
Detection algorithms for the segmentation pipeline.

Provides:
- Tissue detection using K-means variance analysis
- CellDetector: Unified detector with shared models (SAM2, Cellpose, ResNet)
- Detection strategies for different cell types (NMJ, MK, vessel, etc.)
"""

from .cell_detector import (
    CellDetector,
    extract_morphological_features,
)
from .registry import StrategyRegistry
from .strategies import (
    Detection,
    DetectionStrategy,
    NMJStrategy,
)
from .tissue import (
    calculate_block_variances,
    calibrate_tissue_threshold,
    compute_tissue_thresholds,
    compute_variance_threshold,
    filter_tissue_tiles,
    has_tissue,
    is_tissue_block,
)

__all__ = [
    # Tissue detection
    "calculate_block_variances",
    "is_tissue_block",
    "has_tissue",
    "compute_variance_threshold",
    "compute_tissue_thresholds",
    "calibrate_tissue_threshold",
    "filter_tissue_tiles",
    # Unified cell detector
    "CellDetector",
    "extract_morphological_features",
    # Detection strategies
    "DetectionStrategy",
    "Detection",
    "NMJStrategy",
    # Strategy registry
    "StrategyRegistry",
]
