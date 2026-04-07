"""
Detection algorithms for the segmentation pipeline.

Provides:
- Tissue detection using K-means variance analysis
- CellDetector: Unified detector with shared models (SAM2, Cellpose, ResNet)
- Detection strategies for different cell types (NMJ, MK, vessel, etc.)
"""

import importlib

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

# --- Lazy imports for torch-dependent symbols from cell_detector ---
_LAZY_IMPORTS = {
    "CellDetector": "xldvp_seg.detection.cell_detector",
    "extract_morphological_features": "xldvp_seg.detection.cell_detector",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(mod, name)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Tissue detection
    "calculate_block_variances",
    "is_tissue_block",
    "has_tissue",
    "compute_variance_threshold",
    "compute_tissue_thresholds",
    "calibrate_tissue_threshold",
    "filter_tissue_tiles",
    # Unified cell detector (lazy — loaded on first access)
    "CellDetector",
    "extract_morphological_features",
    # Detection strategies
    "DetectionStrategy",
    "Detection",
    "NMJStrategy",
    # Strategy registry
    "StrategyRegistry",
]
