"""
Detection strategy implementations.

Each strategy encapsulates the complete detection pipeline for a specific cell type:
- segment(): Initial mask generation (thresholding, model inference, etc.)
- filter(): Post-processing and classification

Strategies self-register with the StrategyRegistry via @register_strategy decorators.
Importing this package triggers all registrations.

Available strategies:
- NMJStrategy: Neuromuscular junction detection (intensity threshold + morphology + watershed)
- MKStrategy: Megakaryocyte detection (SAM2 auto-mask + size filter)
- CellStrategy: Generic cell detection (Cellpose 2-channel cyto+nuc + SAM2 embeddings)
- VesselStrategy: Blood vessel detection (SMA+ ring detection, 3-contour hierarchy)
- IsletStrategy: Pancreatic islet cell detection (Cellpose membrane+nuclear + marker classification)
- TissuePatternStrategy: Whole-mount tissue pattern detection (Cellpose + spatial frequency)
- MesotheliumStrategy: Mesothelial ribbon detection (ridge detection for ribbon structures)

Mixins:
- MultiChannelFeatureMixin: Extract per-channel features from multi-channel images
"""

from .base import Detection, DetectionStrategy
from .cell import CellStrategy
from .instanseg import InstanSegStrategy
from .islet import IsletStrategy
from .mesothelium import MesotheliumStrategy
from .mixins import MultiChannelFeatureMixin
from .mk import MKStrategy
from .nmj import NMJStrategy
from .tissue_pattern import TissuePatternStrategy
from .vessel import VesselStrategy

__all__ = [
    "DetectionStrategy",
    "Detection",
    "MKStrategy",
    "CellStrategy",
    "NMJStrategy",
    "VesselStrategy",
    "MesotheliumStrategy",
    "IsletStrategy",
    "TissuePatternStrategy",
    "InstanSegStrategy",
    "MultiChannelFeatureMixin",
]
