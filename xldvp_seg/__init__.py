"""
Segmentation package for unified cell segmentation pipeline.

Provides detection, processing, and I/O utilities for:
- MK (Megakaryocytes)
- HSPC (Hematopoietic Stem/Progenitor Cells)
- NMJ (Neuromuscular Junctions)
- Vessel (Blood Vessels)

Usage:
    from segmentation.io import CZILoader, get_loader
    from segmentation.detection import has_tissue, calibrate_tissue_threshold
    from segmentation.processing import DetectionPipeline, BatchProcessor
    from segmentation.preprocessing import correct_photobleaching
    from segmentation.utils import get_logger, setup_logging, load_config
"""

# Version — single source of truth is pyproject.toml
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("xldvp_seg")
except Exception:
    __version__ = "2.0.0"  # fallback for uninstalled editable mode

# Submodule imports are available directly
# Use lazy imports to avoid circular dependencies and unnecessary loading
# Individual modules should be imported explicitly:
#   from segmentation.io import CZILoader
#   from segmentation.utils.logging import get_logger

__all__ = [
    "io",
    "detection",
    "lmd",
    "processing",
    "preprocessing",
    "utils",
    "core",
    "api",
    "analysis",
    "models",
    "metrics",
    "datasets",
    "roi",
]

# Convenience re-exports
from segmentation.core import SlideAnalysis  # noqa: E402
