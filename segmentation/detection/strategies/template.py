"""
Template for creating new detection strategies.

To create a new strategy:
1. Copy this file to your_strategy.py
2. Rename YourStrategy class
3. Implement the required methods
4. Register in __init__.py
5. Add to run_segmentation.py create_strategy_for_cell_type()

Example usage:
    from segmentation.detection.strategies.your_strategy import YourStrategy

    strategy = YourStrategy(min_area_um=100, max_area_um=1000)
    detections = strategy.detect(tile_rgb, models, pixel_size_um=0.22)
"""

import gc
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .base import DetectionStrategy, Detection
from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import (
    extract_morphological_features,
    SAM2_EMBEDDING_DIM,
    RESNET50_FEATURE_DIM,
)

logger = get_logger(__name__)


class YourStrategy(DetectionStrategy):
    """
    [Brief description of what this strategy detects]

    Detection pipeline:
    1. [Step 1 - e.g., Initial segmentation method]
    2. [Step 2 - e.g., Filtering criteria]
    3. [Step 3 - e.g., Feature extraction]
    4. [Step 4 - e.g., Classification (optional)]

    Key parameters:
        min_area_um: Minimum object area in square micrometers
        max_area_um: Maximum object area in square micrometers
        [other parameters]
    """

    def __init__(
        self,
        min_area_um: float = 100.0,
        max_area_um: float = 10000.0,
        pixel_size_um: float = 0.22,
        extract_sam2_embeddings: bool = True,
        extract_resnet_features: bool = True,
        # Add your strategy-specific parameters here
    ):
        """
        Initialize the detection strategy.

        Args:
            min_area_um: Minimum area in square micrometers
            max_area_um: Maximum area in square micrometers
            pixel_size_um: Pixel size for area conversion
            extract_sam2_embeddings: Whether to extract SAM2 embeddings
            extract_resnet_features: Whether to extract ResNet features
        """
        self.min_area_um = min_area_um
        self.max_area_um = max_area_um
        self.pixel_size_um = pixel_size_um
        self.extract_sam2_embeddings = extract_sam2_embeddings
        self.extract_resnet_features = extract_resnet_features

        # Convert area thresholds to pixels
        self._update_pixel_thresholds()

    def _update_pixel_thresholds(self):
        """Update pixel-based thresholds from um-based values."""
        um2_per_px2 = self.pixel_size_um ** 2
        self.min_area_px = int(self.min_area_um / um2_per_px2)
        self.max_area_px = int(self.max_area_um / um2_per_px2)

    @property
    def name(self) -> str:
        """Return the strategy name (used for output directories and logging)."""
        return 'your_cell_type'  # Change this!

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Detect objects in a tile and return masks + detections.

        Args:
            tile: RGB image tile as numpy array (H, W, 3)
            models: Dict containing loaded models:
                - 'sam2': SAM2ImagePredictor (optional)
                - 'cellpose': CellposeModel (optional)
                - 'resnet': ResNet feature extractor (optional)
                - 'resnet_transform': Torchvision transform (optional)
                - 'classifier': Trained classifier (optional)
            pixel_size_um: Pixel size in micrometers (overrides init value)

        Returns:
            Tuple of:
                - masks: Labeled mask array (H, W) with integer labels
                - detections: List of Detection objects with features
        """
        # Update pixel size if provided
        if pixel_size_um is not None:
            self.pixel_size_um = pixel_size_um
            self._update_pixel_thresholds()

        # Validate input
        if tile is None or tile.size == 0:
            return np.zeros((0, 0), dtype=np.uint32), []

        h, w = tile.shape[:2]

        # =====================================================================
        # STEP 1: Initial Segmentation
        # =====================================================================
        # TODO: Implement your segmentation method here
        # Options:
        # - Thresholding (intensity, adaptive, Otsu)
        # - Cellpose (models.get('cellpose'))
        # - SAM2 auto-mask (models.get('sam2'))
        # - Contour detection
        # - etc.

        # Example: Simple intensity thresholding
        if tile.ndim == 3:
            gray = np.mean(tile, axis=2)
        else:
            gray = tile.astype(float)

        threshold = np.percentile(gray, 95)  # Adjust as needed
        binary = gray > threshold

        # Label connected components
        from skimage.measure import label, regionprops
        labeled = label(binary)
        props = regionprops(labeled, intensity_image=gray)

        # =====================================================================
        # STEP 2: Filtering
        # =====================================================================
        valid_detections = []
        valid_labels = []

        for prop in props:
            # Size filter
            if prop.area < self.min_area_px or prop.area > self.max_area_px:
                continue

            # TODO: Add your strategy-specific filters here
            # Examples:
            # - Circularity filter
            # - Solidity filter
            # - Aspect ratio filter
            # - Intensity filter

            # Get centroid (regionprops returns row, col = y, x)
            cy, cx = prop.centroid

            # Create mask for this object
            mask = (labeled == prop.label)

            valid_labels.append(prop.label)
            valid_detections.append({
                'label': prop.label,
                'mask': mask,
                'centroid': [float(cx), float(cy)],  # Store as [x, y]
                'area_px': prop.area,
                'area_um2': prop.area * (self.pixel_size_um ** 2),
                'cy': cy,
                'cx': cx,
            })

        # =====================================================================
        # STEP 3: Feature Extraction
        # =====================================================================
        detections = []

        for det in valid_detections:
            mask = det['mask']
            cy, cx = det['cy'], det['cx']

            # Extract morphological features (22 features)
            morph_features = extract_morphological_features(mask, tile)

            # Extract SAM2 embeddings (256 features)
            sam2_emb = np.zeros(SAM2_EMBEDDING_DIM)
            if self.extract_sam2_embeddings and 'sam2' in models:
                sam2_emb = self._extract_sam2_embedding(models['sam2'], cy, cx)

            # Extract ResNet features (2048 features)
            resnet_features = np.zeros(RESNET50_FEATURE_DIM)
            if self.extract_resnet_features and 'resnet' in models:
                resnet_features = self._extract_resnet_features(
                    tile, mask, cy, cx, models
                )

            # Combine all features
            features = {**morph_features}
            for i, v in enumerate(sam2_emb):
                features[f'sam2_{i}'] = float(v)
            for i, v in enumerate(resnet_features):
                features[f'resnet_{i}'] = float(v)

            # =====================================================================
            # STEP 4: Classification (Optional)
            # =====================================================================
            confidence = 1.0
            is_positive = True

            if 'classifier' in models and models['classifier'] is not None:
                # TODO: Implement classification logic
                # confidence, is_positive = self._classify(crop, models)
                pass

            if not is_positive:
                continue

            # Create Detection object
            detection = Detection(
                mask=mask,
                centroid=det['centroid'],
                features=features,
                id=f"{self.name}_{len(detections)+1}",
                score=confidence,  # confidence stored as score
            )
            detections.append(detection)

        # Create output mask (relabeled)
        output_mask = np.zeros((h, w), dtype=np.uint32)
        for i, det in enumerate(detections, 1):
            output_mask[det.mask] = i

        # Cleanup
        gc.collect()

        return output_mask, detections

    # _extract_sam2_embedding inherited from DetectionStrategy base class
    # _extract_resnet_features_batch inherited from DetectionStrategy base class (batch version)
    # _percentile_normalize inherited from DetectionStrategy base class

    def _extract_resnet_features(
        self,
        tile: np.ndarray,
        mask: np.ndarray,
        cy: float,
        cx: float,
        models: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract ResNet features from crop around detection.

        Note: This is a single-crop version for simplicity.
        For better performance, use the inherited _extract_resnet_features_batch()
        method from the base class which processes multiple crops at once.
        """
        try:
            from PIL import Image

            resnet = models.get('resnet')
            transform = models.get('resnet_transform')

            if resnet is None or transform is None:
                return np.zeros(RESNET50_FEATURE_DIM)

            # Extract crop
            crop_size = 300
            half = crop_size // 2
            h, w = tile.shape[:2]

            y1 = max(0, int(cy) - half)
            y2 = min(h, int(cy) + half)
            x1 = max(0, int(cx) - half)
            x2 = min(w, int(cx) + half)

            if y2 <= y1 or x2 <= x1:
                return np.zeros(RESNET50_FEATURE_DIM)

            crop = tile[y1:y2, x1:x2]

            # Ensure RGB
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)

            # Convert to PIL and process
            pil_img = Image.fromarray(crop.astype(np.uint8))
            tensor = transform(pil_img).unsqueeze(0)

            import torch
            device = next(resnet.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                features = resnet(tensor).cpu().numpy().flatten()

            return features
        except Exception as e:
            logger.debug(f"Failed to extract ResNet features: {e}")
            return np.zeros(RESNET50_FEATURE_DIM)
