"""
xldvp_seg — spatial cell segmentation and Deep Visual Proteomics pipeline.

Provides detection, processing, and I/O utilities for:
- Cell (generic 2-channel Cellpose + SAM2 embeddings)
- NMJ (Neuromuscular Junctions — intensity threshold + watershed)
- MK (Megakaryocytes — SAM2 auto-mask)
- Vessel (Blood Vessels — 3-contour SMA ring detection)
- Islet (Pancreatic Islets — Cellpose + GMM marker classification)
- Mesothelium (ridge detection for ribbon structures)
- Tissue Pattern (multi-channel Cellpose spatial frequency analysis)
- InstanSeg (alternative segmenter via --segmenter instanseg with --cell-type cell)

Usage:
    from xldvp_seg.io import CZILoader, get_loader
    from xldvp_seg.detection import has_tissue, calibrate_tissue_threshold
    from xldvp_seg.processing import DetectionPipeline, BatchProcessor
    from xldvp_seg.preprocessing import correct_photobleaching
    from xldvp_seg.utils import get_logger, setup_logging, load_config
"""

# Version — single source of truth is pyproject.toml
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("xldvp_seg")
except ImportError:
    __version__ = "2.0.0"  # fallback for uninstalled editable mode

# Submodule imports are available directly
# Use lazy imports to avoid circular dependencies and unnecessary loading
# Individual modules should be imported explicitly:
#   from xldvp_seg.io import CZILoader
#   from xldvp_seg.utils.logging import get_logger

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
from xldvp_seg.core import SlideAnalysis  # noqa: E402
