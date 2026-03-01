"""
Unified cell detector with pluggable detection strategies.

This module provides the CellDetector class that holds shared models (SAM2, Cellpose,
ResNet) and a strategy pattern for cell-type specific detection.

Models are loaded lazily on first use to minimize memory footprint when not all
models are needed.

Usage:
    from segmentation.detection.cell_detector import CellDetector
    from segmentation.detection.strategies import NMJStrategy

    # Create detector with lazy-loaded models
    detector = CellDetector(device="cuda")

    # Run detection with one or more strategies
    results = detector.detect(tile, [NMJStrategy()], pixel_size_um=0.22)

    # Cleanup GPU memory when done
    detector.cleanup()

    # Or use as context manager for automatic cleanup
    with CellDetector(device="cuda") as detector:
        results = detector.detect(tile, strategies)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import gc

import numpy as np
import torch
import torchvision.models as tv_models

from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import (
    extract_resnet_features_batch,
    preprocess_crop_for_resnet,
    create_resnet_transform,
    extract_morphological_features,  # Import from shared module (Issue #7)
)

# Import base classes from strategies module
from segmentation.detection.strategies.base import Detection, DetectionStrategy

logger = get_logger(__name__)


class _LazyModelDict(dict):
    """
    Dictionary that lazy-loads models from CellDetector only when accessed.

    This avoids loading ResNet/DINOv2 when they're not needed (e.g., when using
    only morphological + SAM2 features for NMJ classification).
    """

    def __init__(self, detector: 'CellDetector'):
        super().__init__()
        self._detector = detector
        # Track which models have been loaded (must be set before any dict ops)
        self._loaded = {'device'}
        # Pre-populate with device (always available)
        super().__setitem__('device', detector.device)

    def __getitem__(self, key):
        if key not in self._loaded:
            self._load_model(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self._loaded:
            self._load_model(key)
        return super().get(key, default)

    def _load_model(self, key):
        """Load a model on-demand."""
        self._loaded.add(key)
        if key == 'sam2_predictor':
            self[key] = self._detector.sam2_predictor
        elif key == 'sam2_auto':
            self[key] = self._detector.sam2_auto
        elif key == 'cellpose':
            self[key] = self._detector.cellpose
        elif key == 'resnet':
            self[key] = self._detector.resnet
        elif key == 'resnet_transform':
            self[key] = self._detector.resnet_transform
        elif key == 'dinov2':
            self[key] = self._detector.dinov2
        elif key == 'dinov2_transform':
            self[key] = self._detector.dinov2_transform
        elif key == 'device':
            self[key] = self._detector.device
        # For any other key (like 'classifier'), just leave it as-is

    def __setitem__(self, key, value):
        """Allow setting arbitrary keys (e.g., classifier)."""
        self._loaded.add(key)
        super().__setitem__(key, value)


class CellDetector:
    """
    Unified cell detector with pluggable strategies.

    Holds shared models (SAM2, Cellpose, ResNet) that are loaded lazily on first use.
    Detection strategies can be plugged in for different cell types.

    Attributes:
        device: Torch device for model inference
        models: Dict of loaded models for use by strategies

    Example:
        # Basic usage
        detector = CellDetector(device="cuda")
        results = detector.detect(tile, [MKStrategy()], pixel_size_um=0.22)
        detector.cleanup()

        # Context manager usage (auto cleanup)
        with CellDetector(device="cuda") as detector:
            results = detector.detect(tile, strategies)
    """

    def __init__(
        self,
        sam2_checkpoint: Optional[str] = None,
        sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        cellpose_model: str = "cpsam",
        resnet_checkpoint: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize detector with shared models.

        Models are loaded lazily on first use to minimize initial memory footprint.

        Args:
            sam2_checkpoint: Path to SAM2 checkpoint. If None, auto-detects from
                             common locations (./checkpoints/, cluster paths).
            sam2_config: SAM2 config file path
            cellpose_model: Cellpose model name (default: 'cpsam')
            resnet_checkpoint: Path to custom ResNet checkpoint. If None, uses
                               pretrained ImageNet weights.
            device: Device for inference ('cuda', 'cpu', or specific GPU)
        """
        # Convert device string to torch.device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Store config for lazy loading
        self._sam2_checkpoint = sam2_checkpoint
        self._sam2_config = sam2_config
        self._cellpose_model = cellpose_model
        self._resnet_checkpoint = resnet_checkpoint

        # Lazy-loaded models
        self._sam2_predictor = None
        self._sam2_auto = None
        self._sam2_model = None
        self._cellpose = None
        self._resnet = None
        self._resnet_transform = None
        self._dinov2 = None
        self._dinov2_transform = None

        # Cached lazy model dict (created on first access)
        self._models_dict = None

        logger.info(f"CellDetector initialized (device={self.device})")

    @property
    def models(self) -> Dict[str, Any]:
        """
        Get models dict for strategies.

        Note: This uses internal attributes directly to avoid triggering lazy loading.
        Models will be loaded on-demand when strategies access them via the properties.
        The dict is cached so items can be added (e.g., 'classifier') and persist.

        Returns:
            Dict with keys: 'sam2_predictor', 'sam2_auto', 'cellpose', 'resnet',
                           'resnet_transform', 'dinov2', 'dinov2_transform', 'device'
        """
        # Return cached dict or create new one
        if self._models_dict is None:
            self._models_dict = _LazyModelDict(self)
        return self._models_dict

    @property
    def sam2_predictor(self):
        """Lazy load SAM2 predictor (for point prompts and embeddings)."""
        if self._sam2_predictor is None:
            self._load_sam2()
        return self._sam2_predictor

    @property
    def sam2_auto(self):
        """Lazy load SAM2 automatic mask generator."""
        if self._sam2_auto is None:
            self._load_sam2()
        return self._sam2_auto

    @property
    def cellpose(self):
        """Lazy load Cellpose model."""
        if self._cellpose is None:
            self._load_cellpose()
        return self._cellpose

    @property
    def resnet(self):
        """Lazy load ResNet feature extractor."""
        if self._resnet is None:
            self._load_resnet()
        return self._resnet

    @property
    def resnet_transform(self):
        """Get ResNet preprocessing transform."""
        if self._resnet_transform is None:
            self._load_resnet()
        return self._resnet_transform

    @property
    def dinov2(self):
        """Lazy load DINOv2 feature extractor."""
        if self._dinov2 is None:
            self._load_dinov2()
        return self._dinov2

    @property
    def dinov2_transform(self):
        """Get DINOv2 preprocessing transform."""
        if self._dinov2_transform is None:
            self._load_dinov2()
        return self._dinov2_transform

    def _load_dinov2(self):
        """Load DINOv2 feature extractor (ViT-L/14, 1024D features)."""
        logger.info("Loading DINOv2 (dinov2_vitl14)...")

        try:
            self._dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self._dinov2.eval().to(self.device)

            # Same transform as ResNet (ImageNet normalization)
            self._dinov2_transform = create_resnet_transform()

            logger.info("DINOv2 loaded successfully (1024D features)")
        except Exception as e:
            logger.warning(f"Failed to load DINOv2: {e}. DINOv2 features will be zeros.")
            self._dinov2 = None
            self._dinov2_transform = None

    def _find_sam2_checkpoint(self) -> Optional[Path]:
        """Find SAM2 checkpoint in common locations."""
        if self._sam2_checkpoint:
            cp = Path(self._sam2_checkpoint)
            if cp.exists():
                return cp

        # Auto-detect from common locations
        script_dir = Path(__file__).parent.parent.parent.resolve()
        checkpoint_candidates = [
            script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
            script_dir / "checkpoints" / "sam2.1_hiera_l.pt",
            Path("/ptmp/edrod/MKsegmentation/checkpoints/sam2.1_hiera_large.pt"),
        ]

        for cp in checkpoint_candidates:
            if cp.exists():
                return cp

        return None

    def _load_sam2(self):
        """Load SAM2 models (predictor and auto mask generator)."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        checkpoint_path = self._find_sam2_checkpoint()
        if checkpoint_path is None:
            logger.warning("SAM2 checkpoint not found, SAM2 features will be unavailable")
            return

        logger.info(f"Loading SAM2 from {checkpoint_path}...")

        # Build SAM2 model
        self._sam2_model = build_sam2(
            self._sam2_config,
            str(checkpoint_path),
            device=self.device
        )

        # Auto mask generator for automatic segmentation (MK detection)
        self._sam2_auto = SAM2AutomaticMaskGenerator(
            self._sam2_model,
            points_per_side=24,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.4,
            min_mask_region_area=500,
            crop_n_layers=1
        )

        # Predictor for point prompts (HSPC) and embeddings
        self._sam2_predictor = SAM2ImagePredictor(self._sam2_model)

        logger.info("SAM2 loaded successfully")

    def _load_cellpose(self):
        """Load Cellpose-SAM model."""
        from cellpose.models import CellposeModel

        logger.info(f"Loading Cellpose ({self._cellpose_model}) on {self.device}...")

        self._cellpose = CellposeModel(
            pretrained_model=self._cellpose_model,
            gpu=True,
            device=self.device
        )

        logger.info("Cellpose loaded successfully")

    def _load_resnet(self):
        """Load ResNet feature extractor."""
        logger.info("Loading ResNet-50...")

        # Load pretrained ResNet50
        resnet = tv_models.resnet50(weights='DEFAULT')

        # Remove final FC layer to get feature extractor (2048D output)
        self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self._resnet.eval().to(self.device)

        # Standard ImageNet preprocessing
        self._resnet_transform = create_resnet_transform()

        # Load custom checkpoint if provided
        if self._resnet_checkpoint:
            cp_path = Path(self._resnet_checkpoint)
            if cp_path.exists():
                logger.info(f"Loading ResNet checkpoint from {cp_path}...")
                state_dict = torch.load(cp_path, map_location=self.device)
                self._resnet.load_state_dict(state_dict, strict=False)

        logger.info("ResNet loaded successfully")

    def detect(
        self,
        tile: np.ndarray,
        strategies: List[DetectionStrategy],
        pixel_size_um: float = None
    ) -> Dict[str, List[Detection]]:
        """
        Run detection for multiple cell types in one pass.

        Args:
            tile: RGB image tile as numpy array (H, W, 3)
            strategies: List of detection strategies to run
            pixel_size_um: Pixel size for area conversion (from CZI metadata, required)

        Returns:
            Dict mapping cell type name to list of Detection objects
        """
        if pixel_size_um is None:
            raise ValueError("pixel_size_um must be provided (from CZI metadata)")

        results = {}

        for strategy in strategies:
            logger.debug(f"Running detection strategy: {strategy.name}")

            # 1. Segment using strategy-specific approach
            masks = strategy.segment(tile, self.models)
            logger.debug(f"  Generated {len(masks)} candidate masks")

            # 2. Extract features for each mask
            features = []
            for mask in masks:
                feat = strategy.compute_features(mask, tile)
                features.append(feat)

            # 3. Filter based on strategy criteria
            detections = strategy.filter(masks, features, pixel_size_um)
            logger.debug(f"  {len(detections)} detections after filtering")

            results[strategy.name] = detections

        return results

    def extract_sam2_embedding(self, cy: float, cx: float) -> np.ndarray:
        """
        Extract 256D SAM2 embedding at a point.

        Note: set_image() must be called first to set the image for embedding extraction.

        Args:
            cy: Y coordinate (row) in the image
            cx: X coordinate (column) in the image

        Returns:
            256D embedding vector as numpy array
        """
        if self._sam2_predictor is None:
            return np.zeros(256)

        try:
            shape = self._sam2_predictor._features["image_embed"].shape
            emb_h, emb_w = shape[2], shape[3]
            img_h, img_w = self._sam2_predictor._orig_hw[0]

            if img_h == 0 or img_w == 0:
                return np.zeros(256)

            emb_y = int(cy / img_h * emb_h)
            emb_x = int(cx / img_w * emb_w)
            emb_y = min(max(emb_y, 0), emb_h - 1)
            emb_x = min(max(emb_x, 0), emb_w - 1)

            return self._sam2_predictor._features["image_embed"][0, :, emb_y, emb_x].cpu().numpy()
        except Exception as e:
            logger.debug(f"Failed to extract SAM2 embedding at ({cx}, {cy}): {e}")
            return np.zeros(256)

    def extract_full_features(
        self,
        mask: np.ndarray,
        tile: np.ndarray,
        cy: float,
        cx: float
    ) -> Dict[str, Any]:
        """
        Extract full features for a single detection.

        Features include:
        - 22 base morphological/intensity features (up to ~78 with multi-channel)
        - 256 SAM2 embedding features
        - 2048 ResNet deep features

        Args:
            mask: Binary mask of detection
            tile: RGB image tile
            cy: Y centroid coordinate (for SAM2 embedding)
            cx: X centroid coordinate (for SAM2 embedding)

        Returns:
            Dict with all features
        """
        # 22 morphological features
        features = extract_morphological_features(mask, tile)

        # 256 SAM2 embedding features
        sam2_emb = self.extract_sam2_embedding(cy, cx)
        for i, v in enumerate(sam2_emb):
            features[f'sam2_{i}'] = float(v)

        # 2048 ResNet features from masked crop
        ys, xs = np.where(mask)
        if len(ys) > 0:
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop = tile[y1:y2+1, x1:x2+1].copy()
            crop_mask = mask[y1:y2+1, x1:x2+1]
            crop[~crop_mask] = 0

            resnet_feats = self._extract_resnet_single(crop)
            for i, v in enumerate(resnet_feats):
                features[f'resnet_{i}'] = float(v)
        else:
            for i in range(2048):
                features[f'resnet_{i}'] = 0.0

        return features

    def _extract_resnet_single(self, crop: np.ndarray) -> np.ndarray:
        """Extract 2048D ResNet features from a single crop."""
        from PIL import Image

        if crop.size == 0:
            return np.zeros(2048)

        try:
            processed = preprocess_crop_for_resnet(crop)
            pil_img = Image.fromarray(processed, mode='RGB')
            tensor = self.resnet_transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.resnet(tensor).cpu().numpy().flatten()
            return features
        except Exception as e:
            logger.debug(f"Failed to extract ResNet features: {e}")
            return np.zeros(2048)

    def extract_resnet_features_batch(
        self,
        crops: List[np.ndarray],
        batch_size: int = 16
    ) -> List[np.ndarray]:
        """
        Extract ResNet features for multiple crops in batches.

        This improves GPU utilization by processing multiple crops at once.

        Args:
            crops: List of image crops
            batch_size: Batch size for GPU processing (default 16)

        Returns:
            List of 2048D feature vectors
        """
        return extract_resnet_features_batch(
            crops,
            self.resnet,
            self.resnet_transform,
            self.device,
            batch_size=batch_size
        )

    def set_image(self, image: np.ndarray):
        """
        Set image for SAM2 predictor (required for point prompts and embeddings).

        Args:
            image: RGB image as numpy array, must be HxWxC format with C=3

        Raises:
            ValueError: If image is not in expected HxWxC RGB format
        """
        if self._sam2_predictor is not None:
            # Validate image format explicitly (don't assume)
            if image.ndim != 3:
                raise ValueError(f"Image must be 3D (HxWxC), got {image.ndim}D with shape {image.shape}")
            if image.shape[2] != 3:
                raise ValueError(f"Image must have 3 channels (RGB), got {image.shape[2]} channels. Shape: {image.shape}")

            # Ensure uint8 format
            if image.dtype != np.uint8:
                from segmentation.utils.detection_utils import safe_to_uint8
                image = safe_to_uint8(image)
            self._sam2_predictor.set_image(image)

    def reset_predictor(self):
        """Reset SAM2 predictor cache after processing a tile."""
        if self._sam2_predictor is not None:
            self._sam2_predictor.reset_predictor()

    def cleanup(self):
        """
        Release GPU memory.

        Call this when done with the detector to free GPU resources.
        """
        logger.info("Cleaning up CellDetector resources...")

        self._sam2_predictor = None
        self._sam2_auto = None
        self._sam2_model = None
        self._cellpose = None
        self._resnet = None
        self._resnet_transform = None
        self._dinov2 = None
        self._dinov2_transform = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False


# Re-export Detection and DetectionStrategy for convenience
__all__ = [
    'CellDetector',
    'Detection',
    'DetectionStrategy',
    'extract_morphological_features',
]
