"""
Centralized model loading and management.

Provides a singleton ModelManager that handles lazy loading, caching, and cleanup
of all ML models used in the segmentation pipeline.

This module consolidates model loading logic that was previously duplicated in:
- run_unified_FAST.py
- run_segmentation.py
- segmentation/detection/cell_detector.py
- segmentation/cli.py

Usage:
    from segmentation.models import get_model_manager

    manager = get_model_manager(device="cuda")
    sam2_predictor, sam2_auto = manager.get_sam2()
    manager.cleanup()
"""

import gc
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# CHECKPOINT PATH CONFIGURATION
# =============================================================================

# Central definition of checkpoint paths (previously scattered across files)
CHECKPOINT_PATHS = {
    'sam2': [
        Path(__file__).parent.parent.parent / "checkpoints" / "sam2.1_hiera_large.pt",
        Path.home() / ".cache" / "sam2" / "sam2.1_hiera_large.pt",
        Path("/ptmp/edrod/MKsegmentation/checkpoints/sam2.1_hiera_large.pt"),
    ],
    'cellpose': [
        Path(__file__).parent.parent.parent / "checkpoints" / "cpsam",
        Path.home() / ".cache" / "cellpose" / "cpsam",
    ],
    'resnet': [
        # ResNet uses pretrained weights by default, but custom checkpoints can be added
        Path(__file__).parent.parent.parent / "checkpoints" / "resnet50_custom.pth",
    ],
    'mk_classifier': [
        Path(__file__).parent.parent.parent / "checkpoints" / "best_model.pth",
        Path("/ptmp/edrod/MKsegmentation/checkpoints/best_model.pth"),
    ],
}

# SAM2 config path (relative to repo root)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# SAM2 auto mask generator defaults
SAM2_AUTO_DEFAULTS = {
    'points_per_side': 24,
    'pred_iou_thresh': 0.5,
    'stability_score_thresh': 0.4,
    'min_mask_region_area': 500,
    'crop_n_layers': 1,
}


def find_checkpoint(model_name: str) -> Optional[Path]:
    """
    Find checkpoint file for a model by searching known locations.

    Args:
        model_name: Name of model ('sam2', 'cellpose', 'resnet', 'mk_classifier')

    Returns:
        Path to checkpoint if found, None otherwise
    """
    if model_name not in CHECKPOINT_PATHS:
        logger.warning(f"Unknown model name: {model_name}")
        return None

    for path in CHECKPOINT_PATHS[model_name]:
        if path.exists():
            logger.debug(f"Found {model_name} checkpoint at {path}")
            return path

    logger.debug(f"No checkpoint found for {model_name}")
    return None


# =============================================================================
# MODEL MANAGER - SINGLETON PATTERN
# =============================================================================

# Global cache of ModelManager instances per device
_manager_cache: Dict[str, 'ModelManager'] = {}
_manager_cache_lock = threading.Lock()


def get_model_manager(device: str = "cuda") -> 'ModelManager':
    """
    Get or create a ModelManager for the specified device.

    This implements a thread-safe singleton pattern - only one manager exists per device.

    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        ModelManager instance for the device
    """
    device_key = str(device)
    with _manager_cache_lock:
        if device_key not in _manager_cache:
            _manager_cache[device_key] = ModelManager(device=device)
        return _manager_cache[device_key]


def clear_manager_cache():
    """Clear all cached ModelManager instances and free resources."""
    global _manager_cache
    with _manager_cache_lock:
        for key, manager in list(_manager_cache.items()):
            manager.cleanup()
        _manager_cache.clear()
    logger.info("Cleared model manager cache")


class ModelManager:
    """
    Centralized model loading and management.

    Provides lazy loading of ML models with automatic caching and cleanup.
    Models are only loaded when first accessed, minimizing initial memory footprint.

    Attributes:
        device: Torch device for model inference
        sam2_predictor: Loaded SAM2 image predictor (or None)
        sam2_auto: Loaded SAM2 automatic mask generator (or None)
        cellpose: Loaded Cellpose model (or None)
        resnet: Loaded ResNet feature extractor (or None)
        dinov2: Loaded DINOv2 feature extractor (or None)

    Example:
        manager = ModelManager(device="cuda")
        sam2_pred, sam2_auto = manager.get_sam2()
        cellpose = manager.get_cellpose()
        resnet, transform = manager.get_resnet()
        dinov2, dinov2_transform = manager.get_dinov2()
        manager.cleanup()
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize ModelManager.

        Args:
            device: Device for model inference ('cuda', 'cpu', or specific GPU)
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        logger.info(f"ModelManager initialized for device: {self.device}")

        # Lazy-loaded model storage
        self._sam2_predictor = None
        self._sam2_auto = None
        self._sam2_model = None  # Underlying model (shared between predictor/auto)
        self._cellpose = None
        self._resnet = None
        self._resnet_transform = None
        self._dinov2 = None
        self._dinov2_transform = None
        self._mk_classifier = None

    @property
    def sam2_predictor(self):
        """SAM2 image predictor (lazy loaded)."""
        if self._sam2_predictor is None:
            self._load_sam2()
        return self._sam2_predictor

    @property
    def sam2_auto(self):
        """SAM2 automatic mask generator (lazy loaded)."""
        if self._sam2_auto is None:
            self._load_sam2()
        return self._sam2_auto

    @property
    def cellpose(self):
        """Cellpose model (lazy loaded)."""
        if self._cellpose is None:
            self._load_cellpose()
        return self._cellpose

    @property
    def resnet(self):
        """ResNet feature extractor (lazy loaded)."""
        if self._resnet is None:
            self._load_resnet()
        return self._resnet

    @property
    def resnet_transform(self):
        """ResNet preprocessing transform (lazy loaded)."""
        if self._resnet_transform is None:
            self._load_resnet()
        return self._resnet_transform

    @property
    def dinov2(self):
        """DINOv2 feature extractor (lazy loaded)."""
        if self._dinov2 is None:
            self._load_dinov2()
        return self._dinov2

    @property
    def dinov2_transform(self):
        """DINOv2 preprocessing transform (lazy loaded)."""
        if self._dinov2_transform is None:
            self._load_dinov2()
        return self._dinov2_transform

    def get_sam2(self) -> Tuple[Any, Any]:
        """
        Get SAM2 predictor and automatic mask generator.

        Returns:
            Tuple of (SAM2ImagePredictor, SAM2AutomaticMaskGenerator)

        Raises:
            RuntimeError: If SAM2 checkpoint not found
        """
        if self._sam2_predictor is None:
            self._load_sam2()
        return self._sam2_predictor, self._sam2_auto

    def get_cellpose(self) -> Any:
        """
        Get Cellpose model.

        Returns:
            CellposeModel instance

        Raises:
            RuntimeError: If Cellpose loading fails
        """
        if self._cellpose is None:
            self._load_cellpose()
        return self._cellpose

    def get_resnet(self) -> Tuple[Any, Any]:
        """
        Get ResNet feature extractor and transform.

        Returns:
            Tuple of (ResNet model, preprocessing transform)
        """
        if self._resnet is None:
            self._load_resnet()
        return self._resnet, self._resnet_transform

    def get_dinov2(self) -> Tuple[Any, Any]:
        """
        Get DINOv2 feature extractor and transform.

        Returns:
            Tuple of (DINOv2 model, preprocessing transform).
            Model may be None if loading fails.
        """
        if self._dinov2 is None:
            self._load_dinov2()
        return self._dinov2, self._dinov2_transform

    def _load_sam2(self):
        """Load SAM2 models."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        checkpoint_path = find_checkpoint('sam2')
        if checkpoint_path is None:
            raise RuntimeError(
                "SAM2 checkpoint not found. Searched locations:\n" +
                "\n".join(f"  - {p}" for p in CHECKPOINT_PATHS['sam2'])
            )

        logger.info(f"Loading SAM2 from {checkpoint_path}...")
        self._sam2_model = build_sam2(SAM2_CONFIG, str(checkpoint_path), device=self.device)

        # Auto mask generator for MK detection
        self._sam2_auto = SAM2AutomaticMaskGenerator(
            self._sam2_model,
            **SAM2_AUTO_DEFAULTS
        )

        # Predictor for point prompts and embeddings
        self._sam2_predictor = SAM2ImagePredictor(self._sam2_model)
        logger.info("SAM2 loaded successfully")

    def _load_cellpose(self):
        """Load Cellpose-SAM model."""
        # Set cellpose model path before import if checkpoint exists
        checkpoint_path = find_checkpoint('cellpose')
        if checkpoint_path is not None:
            os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(checkpoint_path.parent)

        from cellpose.models import CellposeModel

        logger.info("Loading Cellpose-SAM...")
        self._cellpose = CellposeModel(
            pretrained_model='cpsam',
            gpu=torch.cuda.is_available() and self.device.type == 'cuda',
            device=self.device
        )
        logger.info("Cellpose loaded successfully")

    def _load_resnet(self):
        """Load ResNet-50 feature extractor."""
        logger.info("Loading ResNet-50...")

        # Use pretrained weights
        resnet = tv_models.resnet50(weights='DEFAULT')

        # Remove final FC layer to get 2048D features
        self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self._resnet.eval().to(self.device)

        # Standard ImageNet preprocessing
        self._resnet_transform = tv_transforms.Compose([
            tv_transforms.Resize(224),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        logger.info("ResNet-50 loaded successfully")

    def _load_dinov2(self):
        """Load DINOv2 ViT-L/14 feature extractor (1024D features)."""
        logger.info("Loading DINOv2 (dinov2_vitl14)...")

        try:
            self._dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self._dinov2.eval().to(self.device)

            # Same transform as ResNet (ImageNet normalization)
            self._dinov2_transform = tv_transforms.Compose([
                tv_transforms.Resize(224),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            logger.info("DINOv2 loaded successfully (1024D features)")
        except Exception as e:
            logger.warning(f"Failed to load DINOv2: {e}. DINOv2 features will be zeros.")
            self._dinov2 = None
            self._dinov2_transform = None

    def get_models_dict(self) -> Dict[str, Any]:
        """
        Get a dict of all loaded models (for passing to detection strategies).

        This provides backward compatibility with code that expects a models dict.

        Returns:
            Dict with keys: sam2_predictor, sam2_auto, cellpose, resnet, resnet_transform, device
        """
        return {
            'sam2_predictor': self._sam2_predictor,
            'sam2_auto': self._sam2_auto,
            'cellpose': self._cellpose,
            'resnet': self._resnet,
            'resnet_transform': self._resnet_transform,
            'dinov2': self._dinov2,
            'dinov2_transform': self._dinov2_transform,
            'device': self.device,
        }

    def cleanup(self):
        """
        Release all loaded models and free GPU memory.

        Call this when done with processing to free resources.
        """
        logger.info("Cleaning up ModelManager resources...")

        # Clear SAM2
        if self._sam2_predictor is not None:
            try:
                self._sam2_predictor.reset_predictor()
            except Exception:
                pass
            self._sam2_predictor = None
        self._sam2_auto = None
        self._sam2_model = None

        # Clear Cellpose
        self._cellpose = None

        # Clear ResNet
        self._resnet = None
        self._resnet_transform = None

        # Clear DINOv2
        self._dinov2 = None
        self._dinov2_transform = None

        # Force garbage collection
        gc.collect()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ModelManager cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False

    def __repr__(self) -> str:
        loaded = []
        if self._sam2_predictor is not None:
            loaded.append("SAM2")
        if self._cellpose is not None:
            loaded.append("Cellpose")
        if self._resnet is not None:
            loaded.append("ResNet")
        if self._dinov2 is not None:
            loaded.append("DINOv2")
        loaded_str = ", ".join(loaded) if loaded else "none"
        return f"ModelManager(device={self.device}, loaded=[{loaded_str}])"


__all__ = [
    'ModelManager',
    'get_model_manager',
    'clear_manager_cache',
    'find_checkpoint',
    'CHECKPOINT_PATHS',
    'SAM2_CONFIG',
    'SAM2_AUTO_DEFAULTS',
]
