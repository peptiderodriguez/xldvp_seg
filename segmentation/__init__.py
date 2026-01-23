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

# Version
__version__ = "0.1.0"

# Submodule imports are available directly
# Use lazy imports to avoid circular dependencies and unnecessary loading
# Individual modules should be imported explicitly:
#   from segmentation.io import CZILoader
#   from segmentation.utils.logging import get_logger

__all__ = [
    "io",
    "detection",
    "processing",
    "preprocessing",
    "utils",
    "cli",
]
