"""
Base class for detection strategies.

Each detection strategy encapsulates the complete pipeline for detecting
a specific type of cell or structure in microscopy images.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

from segmentation.utils.feature_extraction import SAM2_EMBEDDING_DIM, RESNET50_FEATURE_DIM

# DINOv2 ViT-L/14 feature dimension
DINOV2_FEATURE_DIM = 1024

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
        models: Dict[str, Any],
        **kwargs
    ) -> List[np.ndarray]:
        """
        Generate candidate masks from a tile image.

        Args:
            tile: Input tile image (HxW for grayscale, HxWxC for RGB)
            models: Dictionary of loaded models (e.g., SAM2, Cellpose, classifiers)
                   May be empty if strategy doesn't use learned models.
            **kwargs: Strategy-specific parameters (e.g., pixel_size_um, cd31_channel)

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
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Complete detection pipeline: segment + compute features + filter.

        This is a convenience method that runs the full pipeline.
        Override in subclasses for custom behavior.

        Args:
            tile: Input tile image
            models: Dictionary of loaded models
            pixel_size_um: Pixel size for area calculations

        Returns:
            Tuple of (label_array, list of Detection objects)
        """
        # Generate candidate masks
        masks = self.segment(tile, models)

        if not masks:
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        # Compute features for each mask
        features = [self.compute_features(m, tile) for m in masks]

        # Filter and classify
        detections = self.filter(masks, features, pixel_size_um)

        # Build label array
        label_array = np.zeros(tile.shape[:2], dtype=np.uint32)
        for i, det in enumerate(detections, start=1):
            if det.mask is not None:
                label_array[det.mask] = i

        return label_array, detections

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

    # ===== Shared classifier and feature methods =====
    # These methods are used by multiple strategies and are provided here
    # to avoid code duplication.

    def classify_rf(
        self,
        detections: List[Detection],
        classifier,
        scaler,
        feature_names: List[str]
    ) -> List[Detection]:
        """
        Classify candidates using trained Random Forest model.

        Supports two formats:
        1. Separate classifier and scaler objects (legacy)
        2. sklearn Pipeline with scaler+RF combined (new format)
           - Pass pipeline as 'classifier', scaler=None

        Args:
            detections: List of Detection objects to classify
            classifier: Trained sklearn RandomForest model OR sklearn Pipeline
            scaler: StandardScaler for feature normalization (None if using Pipeline)
            feature_names: List of feature names expected by classifier

        Returns:
            List of Detection objects with updated scores (keeps ALL)
        """
        from sklearn.pipeline import Pipeline

        if not detections:
            return []

        # Extract features for each detection
        X = []
        valid_indices = []

        for i, det in enumerate(detections):
            if det.features:
                row = []
                for fn in feature_names:
                    val = det.features.get(fn, 0)
                    # Handle non-scalars
                    if isinstance(val, (list, tuple)):
                        val = 0
                    row.append(float(val) if val is not None else 0)
                X.append(row)
                valid_indices.append(i)

        if not X:
            return detections

        # Convert to numpy
        X = np.array(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Handle Pipeline vs separate scaler+classifier
        if isinstance(classifier, Pipeline):
            # Pipeline handles scaling internally
            probs = classifier.predict_proba(X)[:, 1]
        else:
            # Legacy: apply scaler separately
            if scaler is not None:
                X = scaler.transform(X)
            probs = classifier.predict_proba(X)[:, 1]

        # Update detections with scores (keep ALL - filter post-hoc by rf_prediction)
        for j, (idx, prob) in enumerate(zip(valid_indices, probs)):
            det = detections[idx]
            det.score = float(prob)
            det.features['rf_prediction'] = float(prob)
            det.features['confidence'] = float(prob)
            # Strategy-specific extras (e.g., NMJ sets prob_nmj)
            self._post_classify_rf_detection(det, prob)

        # Set score=0 for detections that couldn't be classified (missing features)
        for i, det in enumerate(detections):
            if det.score is None:
                det.score = 0.0
                det.features['rf_prediction'] = 0.0
                det.features['confidence'] = 0.0
                self._post_classify_rf_detection(det, 0.0)

        threshold = getattr(self, 'classifier_threshold', 0.5)
        n_above = sum(1 for d in detections if d.score >= threshold)
        logger.debug(f"RF classifier: {n_above}/{len(detections)} above {threshold}, keeping all")

        return detections

    def _post_classify_rf_detection(self, det: Detection, prob: float) -> None:
        """
        Hook for strategy-specific post-classification updates.

        Override in subclasses to set additional feature keys after RF
        classification (e.g., NMJ sets 'prob_nmj' for backward compat).

        Args:
            det: Detection object that was just scored
            prob: The RF probability assigned to this detection
        """
        pass

    @staticmethod
    def _zero_fill_deep_features(
        detection_dicts: list,
        has_dinov2: bool = False,
    ) -> None:
        """
        Zero-fill missing deep feature vectors (ResNet, DINOv2) on detection dicts.

        Call this after batch deep-feature extraction to ensure every detection
        has a consistent feature vector length, even if crop extraction failed
        for some detections.

        Args:
            detection_dicts: List of dicts, each with a 'features' sub-dict.
                Supports both {'features': {...}} dicts and Detection objects
                (via duck-typing on ['features'] access).
            has_dinov2: Whether DINOv2 features are expected (controls whether
                dinov2_* keys get zero-filled).
        """
        for det in detection_dicts:
            feats = det['features']
            if 'resnet_0' not in feats:
                for i in range(RESNET50_FEATURE_DIM):
                    feats[f'resnet_{i}'] = 0.0
            if 'resnet_ctx_0' not in feats:
                for i in range(RESNET50_FEATURE_DIM):
                    feats[f'resnet_ctx_{i}'] = 0.0
            if has_dinov2 and 'dinov2_0' not in feats:
                for i in range(DINOV2_FEATURE_DIM):
                    feats[f'dinov2_{i}'] = 0.0
            if has_dinov2 and 'dinov2_ctx_0' not in feats:
                for i in range(DINOV2_FEATURE_DIM):
                    feats[f'dinov2_ctx_{i}'] = 0.0

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
            # _orig_hw is a list containing a tuple: [(h, w)]
            img_h, img_w = sam2_predictor._orig_hw[0]

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

    def _extract_dinov2_features_batch(
        self,
        crops: List[np.ndarray],
        model,
        transform,
        device,
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Extract DINOv2 features for multiple crops in batches.

        Args:
            crops: List of image crops as numpy arrays
            model: DINOv2 model
            transform: Torchvision transform for preprocessing
            device: Torch device to use
            batch_size: Batch size for processing

        Returns:
            List of 1024D feature vectors as numpy arrays (for ViT-L/14)
        """
        import torch
        from PIL import Image

        if not crops:
            return []

        if batch_size is None or batch_size < 1:
            batch_size = 32

        all_features = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_tensors = []
            valid_indices = []

            for idx, crop in enumerate(batch_crops):
                try:
                    if crop.dtype == np.uint16:
                        crop = (crop / 256).astype(np.uint8)
                    elif crop.dtype != np.uint8:
                        crop = crop.astype(np.uint8)

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
                    logger.debug(f"Failed to preprocess crop for DINOv2 {idx}: {e}")

            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(device)

                with torch.no_grad():
                    # DINOv2 returns CLS token features directly
                    features = model(batch_tensor).cpu().numpy()

                feature_idx = 0
                for idx in range(len(batch_crops)):
                    if idx in valid_indices:
                        all_features.append(features[feature_idx])
                        feature_idx += 1
                    else:
                        all_features.append(np.zeros(DINOV2_FEATURE_DIM))
            else:
                for _ in batch_crops:
                    all_features.append(np.zeros(DINOV2_FEATURE_DIM))

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

    def _extract_full_features_batch(
        self,
        masks: List[np.ndarray],
        tile: np.ndarray,
        models: Dict[str, Any],
        extract_sam2: bool = True,
        extract_resnet: bool = True,
        resnet_batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Extract full feature set for multiple masks in batch.

        This method consolidates the feature extraction logic duplicated across
        strategies (NMJ, MK, etc.) into a single reusable method. For each mask,
        it extracts:
        - Base morphological features (22 features) using extract_morphological_features
        - SAM2 embeddings (256 features) at the mask centroid
        - ResNet features (2x2048 = 4096: masked + context) using efficient batch processing
        - DINOv2 features (2x1024 = 2048: masked + context) if dinov2 model provided

        Args:
            masks: List of boolean mask arrays (each HxW)
            tile: Image tile array (HxW for grayscale, HxWxC for RGB)
            models: Dictionary containing loaded models:
                - 'sam2_predictor': SAM2ImagePredictor (optional, for embeddings)
                - 'resnet': ResNet model without final FC layer (optional)
                - 'resnet_transform': torchvision transform for ResNet input
                - 'device': torch device for inference
            extract_sam2: Whether to extract SAM2 embeddings (default True)
            extract_resnet: Whether to extract ResNet features (default True)
            resnet_batch_size: Batch size for ResNet processing (default 32)

        Returns:
            List of feature dictionaries, one per mask. Each dict contains:
            - Morphological features (area, perimeter, solidity, etc.)
            - 'centroid': [x, y] coordinates
            - 'sam2_0' through 'sam2_255': SAM2 embedding values
            - 'resnet_0' through 'resnet_2047': ResNet feature values
            Empty dict returned for invalid masks.
        """
        from segmentation.utils.feature_extraction import extract_morphological_features

        if not masks:
            return []

        # Prepare tile for processing
        if tile.ndim == 2:
            tile_rgb = np.stack([tile, tile, tile], axis=-1)
        elif tile.shape[2] == 1:
            tile_rgb = np.concatenate([tile, tile, tile], axis=-1)
        else:
            tile_rgb = tile[:, :, :3]

        # Ensure uint8 format
        if tile_rgb.dtype == np.uint16:
            tile_rgb = (tile_rgb / 256).astype(np.uint8)
        elif tile_rgb.dtype != np.uint8:
            tile_rgb = tile_rgb.astype(np.uint8)

        # Get models
        sam2_predictor = models.get('sam2_predictor')
        resnet = models.get('resnet')
        resnet_transform = models.get('resnet_transform')
        device = models.get('device')

        # Set image for SAM2 embeddings if available
        if sam2_predictor is not None and extract_sam2:
            try:
                sam2_predictor.set_image(tile_rgb)
            except Exception as e:
                logger.debug(f"Failed to set SAM2 image: {e}")
                sam2_predictor = None

        # First pass: extract morphological features and SAM2 embeddings, collect crops
        feature_list = []
        crops_for_resnet = []  # Masked crops (background zeroed)
        crops_for_resnet_context = []  # Context crops (full tissue, unmasked)
        crop_indices = []

        for idx, mask in enumerate(masks):
            # Skip empty masks
            if mask.sum() == 0:
                feature_list.append({})
                continue

            # Extract morphological features (22 features)
            morph_features = extract_morphological_features(mask, tile_rgb)
            if not morph_features:
                feature_list.append({})
                continue

            # Compute centroid
            ys, xs = np.where(mask)
            if len(ys) == 0:
                feature_list.append({})
                continue

            cy, cx = float(np.mean(ys)), float(np.mean(xs))
            morph_features['centroid'] = [cx, cy]  # [x, y] format

            # Extract SAM2 embeddings (256D)
            if sam2_predictor is not None and extract_sam2:
                sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy, cx)
                for i, v in enumerate(sam2_emb):
                    morph_features[f'sam2_{i}'] = float(v)
            elif extract_sam2:
                # Fill with zeros if SAM2 not available
                for i in range(SAM2_EMBEDDING_DIM):
                    morph_features[f'sam2_{i}'] = 0.0

            # Prepare crops for batch deep feature extraction (ResNet and/or DINOv2)
            if extract_resnet or (models.get('dinov2') is not None):
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                if y2 > y1 and x2 > x1:
                    # Context crop (unmasked - full tissue context)
                    crop_context = tile_rgb[y1:y2+1, x1:x2+1].copy()
                    crops_for_resnet_context.append(crop_context)

                    # Masked crop (background zeroed out)
                    crop_masked = crop_context.copy()
                    crop_mask = mask[y1:y2+1, x1:x2+1]
                    crop_masked[~crop_mask] = 0
                    crops_for_resnet.append(crop_masked)
                    crop_indices.append(idx)

            feature_list.append(morph_features)

        # Batch ResNet feature extraction - masked (original)
        if crops_for_resnet and resnet is not None and resnet_transform is not None and extract_resnet:
            resnet_features_list = self._extract_resnet_features_batch(
                crops_for_resnet, resnet, resnet_transform, device, resnet_batch_size
            )

            # Assign masked ResNet features to correct detections
            for crop_idx, resnet_feats in zip(crop_indices, resnet_features_list):
                if feature_list[crop_idx]:  # Only if features dict exists
                    for i, v in enumerate(resnet_feats):
                        feature_list[crop_idx][f'resnet_{i}'] = float(v)

        # Batch ResNet feature extraction - context (unmasked)
        if crops_for_resnet_context and resnet is not None and resnet_transform is not None and extract_resnet:
            resnet_context_list = self._extract_resnet_features_batch(
                crops_for_resnet_context, resnet, resnet_transform, device, resnet_batch_size
            )

            # Assign context ResNet features to correct detections
            for crop_idx, resnet_feats in zip(crop_indices, resnet_context_list):
                if feature_list[crop_idx]:
                    for i, v in enumerate(resnet_feats):
                        feature_list[crop_idx][f'resnet_ctx_{i}'] = float(v)

        # Fill zeros for features without ResNet values
        if extract_resnet:
            for feat in feature_list:
                if feat and 'resnet_0' not in feat:
                    for i in range(RESNET50_FEATURE_DIM):
                        feat[f'resnet_{i}'] = 0.0
                if feat and 'resnet_ctx_0' not in feat:
                    for i in range(RESNET50_FEATURE_DIM):
                        feat[f'resnet_ctx_{i}'] = 0.0

        # Batch DINOv2 feature extraction (both masked and context)
        dinov2 = models.get('dinov2')
        dinov2_transform = models.get('dinov2_transform')

        if crops_for_resnet and dinov2 is not None and dinov2_transform is not None:
            # DINOv2 masked features
            dinov2_masked_list = self._extract_dinov2_features_batch(
                crops_for_resnet, dinov2, dinov2_transform, device, resnet_batch_size
            )
            for crop_idx, dino_feats in zip(crop_indices, dinov2_masked_list):
                if feature_list[crop_idx]:
                    for i, v in enumerate(dino_feats):
                        feature_list[crop_idx][f'dinov2_{i}'] = float(v)

            # DINOv2 context features
            dinov2_context_list = self._extract_dinov2_features_batch(
                crops_for_resnet_context, dinov2, dinov2_transform, device, resnet_batch_size
            )
            for crop_idx, dino_feats in zip(crop_indices, dinov2_context_list):
                if feature_list[crop_idx]:
                    for i, v in enumerate(dino_feats):
                        feature_list[crop_idx][f'dinov2_ctx_{i}'] = float(v)

        # Fill zeros for features without DINOv2 values
        if dinov2 is not None:
            for feat in feature_list:
                if feat and 'dinov2_0' not in feat:
                    for i in range(DINOV2_FEATURE_DIM):
                        feat[f'dinov2_{i}'] = 0.0
                if feat and 'dinov2_ctx_0' not in feat:
                    for i in range(DINOV2_FEATURE_DIM):
                        feat[f'dinov2_ctx_{i}'] = 0.0

        # Reset SAM2 predictor to free memory
        if sam2_predictor is not None and extract_sam2:
            try:
                sam2_predictor.reset_predictor()
            except Exception as e:
                logger.debug(f"Failed to reset SAM2 predictor: {e}")

        return feature_list
