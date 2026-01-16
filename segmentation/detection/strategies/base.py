"""
Base class for detection strategies.

Each detection strategy encapsulates the complete pipeline for detecting
a specific type of cell or structure in microscopy images.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import logging

from segmentation.utils.feature_extraction import SAM2_EMBEDDING_DIM, RESNET50_FEATURE_DIM

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    A single detection from the segmentation pipeline.

    Attributes:
        mask: Binary mask of the detection (HxW boolean array)
        centroid: Local centroid [x, y] within the tile
        features: Dictionary of computed features
        id: Optional local ID within tile
        score: Optional confidence score (from classifier)
    """
    mask: np.ndarray
    centroid: List[float]
    features: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    score: Optional[float] = None

    @property
    def area(self) -> int:
        """Area in pixels."""
        return int(self.mask.sum())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'centroid': self.centroid,
            'features': self.features,
            'id': self.id,
            'score': self.score,
            'area': self.area,
        }


class DetectionStrategy(ABC):
    """
    Abstract base class for cell/structure detection strategies.

    Each strategy implements:
    1. segment(): Generate candidate masks from a tile
    2. filter(): Filter and classify candidates

    The strategy pattern allows different cell types to use entirely
    different detection approaches while maintaining a consistent interface.

    Example usage:
        strategy = NMJStrategy(intensity_percentile=99.5)
        masks = strategy.segment(tile_data, models={})
        detections = strategy.filter(masks, features, pixel_size_um=0.22)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name (e.g., 'nmj', 'mk', 'vessel')."""
        pass

    @abstractmethod
    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Generate candidate masks from a tile image.

        Args:
            tile: Input tile image (HxW for grayscale, HxWxC for RGB)
            models: Dictionary of loaded models (e.g., SAM2, Cellpose, classifiers)
                   May be empty if strategy doesn't use learned models.

        Returns:
            List of binary masks (each HxW boolean array)
        """
        pass

    @abstractmethod
    def filter(
        self,
        masks: List[np.ndarray],
        features: List[Dict[str, Any]],
        pixel_size_um: float
    ) -> List[Detection]:
        """
        Filter and classify candidate masks.

        Args:
            masks: List of candidate masks from segment()
            features: List of feature dictionaries (one per mask)
            pixel_size_um: Pixel size for area calculations

        Returns:
            List of Detection objects that pass filtering criteria
        """
        pass

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float
    ) -> List[Detection]:
        """
        Complete detection pipeline: segment + compute features + filter.

        This is a convenience method that runs the full pipeline.
        Override in subclasses for custom behavior.

        Args:
            tile: Input tile image
            models: Dictionary of loaded models
            pixel_size_um: Pixel size for area calculations

        Returns:
            List of Detection objects
        """
        # Generate candidate masks
        masks = self.segment(tile, models)

        if not masks:
            return []

        # Compute features for each mask
        features = [self.compute_features(m, tile) for m in masks]

        # Filter and classify
        detections = self.filter(masks, features, pixel_size_um)

        return detections

    def compute_features(
        self,
        mask: np.ndarray,
        tile: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute features for a single mask.

        Default implementation computes basic morphological features.
        Override in subclasses for strategy-specific features.

        Args:
            mask: Binary mask
            tile: Original tile image

        Returns:
            Dictionary of features
        """
        from skimage.measure import regionprops

        if mask.sum() == 0:
            return {}

        # Get intensity image
        if tile.ndim == 3:
            gray = np.mean(tile[:, :, :3], axis=2)
        else:
            gray = tile.astype(float)

        # Use regionprops for basic features
        props = regionprops(mask.astype(int), intensity_image=gray)

        if not props:
            return {}

        prop = props[0]

        return {
            'area': int(prop.area),
            'centroid': [float(prop.centroid[1]), float(prop.centroid[0])],  # [x, y]
            'eccentricity': float(prop.eccentricity),
            'solidity': float(prop.solidity),
            'mean_intensity': float(prop.mean_intensity),
            'perimeter': float(prop.perimeter),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return strategy configuration for logging/reproducibility.

        Default implementation returns all instance attributes.
        """
        return {
            'strategy': self.name,
            **{k: v for k, v in self.__dict__.items()
               if not k.startswith('_') and not callable(v)}
        }

    # ===== Shared feature extraction methods =====
    # These methods are used by multiple strategies (NMJ, MK, Vessel, Cell)
    # and are provided here to avoid code duplication.

    def _extract_sam2_embedding(
        self,
        sam2_predictor,
        cy: float,
        cx: float
    ) -> np.ndarray:
        """
        Extract 256D SAM2 embedding at a point.

        The SAM2 image embedding is a spatial feature map. This extracts
        the 256D feature vector at the position corresponding to (cy, cx).

        Args:
            sam2_predictor: SAM2ImagePredictor with image already set
            cy: Y coordinate (row) in image
            cx: X coordinate (column) in image

        Returns:
            256D feature vector as numpy array
        """
        try:
            shape = sam2_predictor._features["image_embed"].shape
            emb_h, emb_w = shape[2], shape[3]
            img_h, img_w = sam2_predictor._orig_hw

            # Check for division by zero
            if img_h == 0 or img_w == 0:
                return np.zeros(SAM2_EMBEDDING_DIM)

            # Map image coordinates to embedding coordinates
            emb_y = int(cy / img_h * emb_h)
            emb_x = int(cx / img_w * emb_w)

            # Clamp to valid range
            emb_y = min(max(emb_y, 0), emb_h - 1)
            emb_x = min(max(emb_x, 0), emb_w - 1)

            return sam2_predictor._features["image_embed"][0, :, emb_y, emb_x].cpu().numpy()
        except Exception as e:
            logger.debug(f"Failed to extract SAM2 embedding at ({cx}, {cy}): {e}")
            return np.zeros(SAM2_EMBEDDING_DIM)

    def _extract_resnet_features_batch(
        self,
        crops: List[np.ndarray],
        model,
        transform,
        device,
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Extract ResNet features for multiple crops in batches.

        Batch processing improves GPU utilization by processing multiple
        crops at once rather than one at a time.

        Args:
            crops: List of image crops as numpy arrays
            model: ResNet model (nn.Sequential ending before FC layer)
            transform: Torchvision transform for preprocessing
            device: Torch device to use
            batch_size: Batch size for processing

        Returns:
            List of 2048D feature vectors as numpy arrays
        """
        import torch
        from PIL import Image

        if not crops:
            return []

        # Validate batch_size
        if batch_size is None or batch_size < 1:
            batch_size = 32
            logger.debug(f"Invalid batch_size, using default: {batch_size}")

        all_features = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_tensors = []
            valid_indices = []

            # Preprocess each crop
            for idx, crop in enumerate(batch_crops):
                try:
                    # Handle uint16 images
                    if crop.dtype == np.uint16:
                        crop = (crop / 256).astype(np.uint8)
                    elif crop.dtype != np.uint8:
                        crop = crop.astype(np.uint8)

                    # Ensure RGB format
                    if crop.ndim == 2:
                        crop = np.stack([crop, crop, crop], axis=-1)
                    elif crop.shape[-1] != 3:
                        crop = np.ascontiguousarray(crop[..., :3])

                    if crop.size == 0:
                        continue

                    pil_img = Image.fromarray(crop, mode='RGB')
                    tensor = transform(pil_img)
                    batch_tensors.append(tensor)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.debug(f"Failed to preprocess crop {idx}: {e}")

            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(device)

                with torch.no_grad():
                    features = model(batch_tensor)
                    features = features.squeeze(-1).squeeze(-1)
                    features = features.cpu().numpy()

                # Map features back to correct indices
                feature_idx = 0
                for idx in range(len(batch_crops)):
                    if idx in valid_indices:
                        all_features.append(features[feature_idx])
                        feature_idx += 1
                    else:
                        all_features.append(np.zeros(RESNET50_FEATURE_DIM))
            else:
                for _ in batch_crops:
                    all_features.append(np.zeros(RESNET50_FEATURE_DIM))

        return all_features

    def _percentile_normalize(
        self,
        image: np.ndarray,
        p_low: float = 1,
        p_high: float = 99.5
    ) -> np.ndarray:
        """
        Normalize image using percentiles.

        Args:
            image: Input image array
            p_low: Lower percentile (default 1)
            p_high: Upper percentile (default 99.5)

        Returns:
            Normalized uint8 image
        """
        if image.dtype == np.uint8:
            return image

        p_low_val, p_high_val = np.percentile(image, [p_low, p_high])

        if p_high_val <= p_low_val:
            return np.zeros_like(image, dtype=np.uint8)

        normalized = (image.astype(float) - p_low_val) / (p_high_val - p_low_val)
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        return normalized
