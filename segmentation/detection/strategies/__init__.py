"""
Detection strategy implementations.

Each strategy encapsulates the complete detection pipeline for a specific cell type:
- segment(): Initial mask generation (thresholding, model inference, etc.)
- filter(): Post-processing and classification

Available strategies:
- MKStrategy: Megakaryocyte detection (SAM2 auto mask + size filter + optional classifier)
- CellStrategy: Generic cell detection (Cellpose + SAM2 refinement)
- NMJStrategy: Neuromuscular junction detection (intensity threshold + elongation + ResNet)
- VesselStrategy: Blood vessel detection (ring structures via contour hierarchy + ellipse fitting)
- MesotheliumStrategy: Mesothelial ribbon detection for laser microdissection
- IsletStrategy: Pancreatic islet cell detection (Cellpose membrane+nuclear + SAM2)

Mixins:
- MultiChannelFeatureMixin: Extract per-channel features from multi-channel images
"""

from .base import DetectionStrategy, Detection
from .mk import MKStrategy
from .cell import CellStrategy, HSPCStrategy  # HSPCStrategy is backward compatibility alias
from .nmj import NMJStrategy
from .vessel import VesselStrategy
from .mesothelium import MesotheliumStrategy
from .islet import IsletStrategy
from .mixins import MultiChannelFeatureMixin

__all__ = [
    'DetectionStrategy',
    'Detection',
    'MKStrategy',
    'CellStrategy',
    'HSPCStrategy',  # Backward compatibility alias
    'NMJStrategy',
    'VesselStrategy',
    'MesotheliumStrategy',
    'IsletStrategy',
    'MultiChannelFeatureMixin',
]
