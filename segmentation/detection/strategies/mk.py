"""
Megakaryocyte (MK) detection strategy.

MKs are large cells detected using:
1. SAM2 automatic mask generation (no Cellpose needed)
2. Size filtering (default 200-2000 um^2)
3. Overlap filtering (skip masks with >50% overlap with existing larger masks)
4. Optional ResNet classification (MK vs non-MK)

The detection pipeline:
- SAM2 auto mask generator proposes all candidate masks
- Size filter removes masks outside the expected MK area range
- Masks are sorted by area (largest first) to prioritize larger cells
- Overlap filter removes smaller masks that overlap >50% with already-accepted masks
- Optional classifier further filters false positives
"""

import gc
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .base import DetectionStrategy, Detection, _safe_to_uint8
from .mixins import MultiChannelFeatureMixin
from segmentation.utils.feature_extraction import (
    extract_morphological_features,
    SAM2_EMBEDDING_DIM,
    RESNET50_FEATURE_DIM,
)

logger = logging.getLogger(__name__)


# Issue #7: Local extract_morphological_features removed - now imported from shared module


class MKStrategy(DetectionStrategy, MultiChannelFeatureMixin):
    """
    Megakaryocyte detection strategy.

    MKs are large cells detected using:
    1. SAM2 auto-mask generation (no Cellpose)
    2. Size filtering (default 200-2000 um^2)
    3. Overlap filtering (removes smaller overlapping masks)
    4. Optional ResNet classification (MK vs non-MK)

    SAM2 Auto Mask Generator Configuration:
        - points_per_side: 24 (grid density for mask proposals)
        - pred_iou_thresh: 0.5 (minimum IOU threshold)
        - stability_score_thresh: 0.4 (minimum stability)
        - min_mask_region_area: 500 (minimum pixels)
        - crop_n_layers: 1 (hierarchical crop refinement)
    """

    def __init__(self,
                 min_area_um: float = 200.0,
                 max_area_um: float = 2000.0,
                 classifier_threshold: float = 0.5,
                 overlap_threshold: float = 0.5,
                 extract_deep_features: bool = False,
                 extract_sam2_embeddings: bool = True,
                 resnet_batch_size: int = 32):
        """
        Initialize MK detection strategy.

        Args:
            min_area_um: Minimum MK area in square micrometers (default 200)
            max_area_um: Maximum MK area in square micrometers (default 2000)
            classifier_threshold: Minimum classifier score to keep detection (default 0.5)
            overlap_threshold: Maximum overlap fraction with existing masks (default 0.5)
            extract_deep_features: Whether to extract ResNet+DINOv2 features (default False, opt-in)
            extract_sam2_embeddings: Whether to extract 256D SAM2 embeddings
            resnet_batch_size: Batch size for ResNet feature extraction
        """
        self.min_area_um = min_area_um
        self.max_area_um = max_area_um
        self.classifier_threshold = classifier_threshold
        self.overlap_threshold = overlap_threshold
        self.extract_deep_features = extract_deep_features
        self.extract_sam2_embeddings = extract_sam2_embeddings
        self.resnet_batch_size = resnet_batch_size

    @property
    def name(self) -> str:
        return "mk"

    def segment(self, tile: np.ndarray, models: Dict[str, Any], **kwargs) -> List[np.ndarray]:
        """
        Generate candidate binary masks from a tile image.

        Conforms to the DetectionStrategy base class interface by returning
        List[np.ndarray] (binary masks). Internally delegates to _segment_sam2()
        and extracts the 'segmentation' mask from each SAM2 result dict.

        Args:
            tile: RGB image tile (H, W, 3)
            models: Dict with 'sam2_auto' (SAM2AutomaticMaskGenerator)
            **kwargs: Additional strategy-specific parameters (unused)

        Returns:
            List of binary masks (each HxW boolean array)
        """
        sam2_results = self._segment_sam2(tile, models)
        return [result['segmentation'].astype(bool) for result in sam2_results]

    def _segment_sam2(self, tile: np.ndarray, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use SAM2 automatic mask generation to find MK candidates.

        SAM2 auto mask generator scans the image with a grid of points and
        proposes masks for any detected objects. Results include:
        - 'segmentation': Binary mask
        - 'area': Mask area in pixels
        - 'predicted_iou': SAM2's predicted IOU score
        - 'stability_score': Mask stability across perturbations

        Args:
            tile: RGB image tile (H, W, 3)
            models: Dict with 'sam2_auto' (SAM2AutomaticMaskGenerator)

        Returns:
            List of SAM2 result dicts with segmentation masks and metadata
        """
        sam2_auto = models.get('sam2_auto')
        if sam2_auto is None:
            raise RuntimeError("SAM2 automatic mask generator not loaded - required for MK detection")

        # SAM2 expects uint8 RGB image
        sam2_img = _safe_to_uint8(tile)

        # Generate all candidate masks
        sam2_results = sam2_auto.generate(sam2_img)

        return sam2_results

    def filter(self,
               masks_or_results: List[Any],
               features: List[Dict[str, Any]],
               pixel_size_um: float) -> List[Detection]:
        """
        Filter masks by size, overlap, and classifier score.

        Filtering steps:
        1. Size filter: Keep masks within [min_area_um, max_area_um]
        2. Sort by area (largest first) to prioritize larger MKs
        3. Overlap filter: Skip masks with >50% overlap with already-accepted masks
        4. Classifier filter: If mk_score available, filter by threshold

        Args:
            masks_or_results: List of SAM2 result dicts or binary masks
            features: List of feature dicts (with 'mk_score' if classifier used)
            pixel_size_um: Pixel size in micrometers

        Returns:
            List of Detection objects that passed all filters
        """
        from scipy import ndimage

        # Convert area thresholds from um^2 to pixels
        min_area_px = self.min_area_um / (pixel_size_um ** 2)
        max_area_px = self.max_area_um / (pixel_size_um ** 2)

        # Handle both SAM2 result dicts and raw masks
        results = []
        for item in masks_or_results:
            if isinstance(item, dict) and 'segmentation' in item:
                # SAM2 result dict
                mask = item['segmentation']
                area = item.get('area', mask.sum())
                metadata = {
                    'sam2_iou': float(item.get('predicted_iou', 0)),
                    'sam2_stability': float(item.get('stability_score', 0))
                }
            else:
                # Raw mask
                mask = item
                area = mask.sum()
                metadata = {}

            results.append({
                'mask': mask,
                'area': area,
                'metadata': metadata
            })

        # Step 1: Size filtering — pair each result with its original feature dict
        # so that filtering doesn't break the index correspondence
        valid_results = []
        for orig_idx, result in enumerate(results):
            if min_area_px <= result['area'] <= max_area_px:
                result['_orig_features'] = features[orig_idx] if orig_idx < len(features) else {}
                valid_results.append(result)

        # Step 2: Sort by area (largest first) — features travel with results
        valid_results.sort(key=lambda x: x['area'], reverse=True)

        # Step 3: Overlap filtering and detection creation
        detections = []
        accepted_mask = None  # Combined mask of all accepted detections

        for result in valid_results:
            mask = result['mask']

            # Ensure boolean type (critical for NVIDIA CUDA compatibility)
            if mask.dtype != bool:
                mask = (mask > 0.5).astype(bool)

            # Check overlap with already-accepted masks
            if accepted_mask is not None and accepted_mask.any():
                overlap = ((mask > 0) & (accepted_mask > 0)).sum()
                if overlap > self.overlap_threshold * mask.sum():
                    continue  # Skip this mask - too much overlap

            # Get features (aligned with sorted results)
            feat = result['_orig_features']

            # Step 4: Classifier filtering
            mk_score = feat.get('mk_score', 1.0)
            if mk_score < self.classifier_threshold:
                continue  # Skip - classifier rejected

            # Compute centroid
            cy, cx = ndimage.center_of_mass(mask)

            # Create detection - merge SAM2 metadata into features
            feat_with_metadata = feat.copy()
            feat_with_metadata.update(result['metadata'])
            feat_with_metadata['mk_score'] = mk_score

            det = Detection(
                mask=mask,
                centroid=[float(cx), float(cy)],  # [x, y] format per base class
                features=feat_with_metadata,
                score=mk_score
            )
            detections.append(det)

            # Update accepted mask
            if accepted_mask is None:
                accepted_mask = mask.copy()
            else:
                accepted_mask = accepted_mask | mask

        return detections

    def detect(self,
               tile: np.ndarray,
               models: Dict[str, Any],
               pixel_size_um: float = None,
               extract_features: bool = True,
               extra_channels: Dict[int, np.ndarray] = None) -> Tuple[np.ndarray, List[Detection]]:
        """
        Full MK detection pipeline with optimized batch processing.

        This overrides the base detect() for MK-specific optimizations:
        1. SAM2 auto mask generation
        2. Size filtering (before feature extraction for efficiency)
        3. Overlap filtering
        4. Batch feature extraction (morphological + SAM2 + ResNet)
        5. Per-channel features when extra_channels provided
        6. Optional classifier filtering

        Args:
            tile: RGB image tile (H, W, 3)
            models: Dict with loaded models:
                - 'sam2_auto': SAM2AutomaticMaskGenerator (required)
                - 'sam2_predictor': SAM2ImagePredictor (for embeddings)
                - 'resnet': ResNet model (for features)
                - 'resnet_transform': torchvision transform
                - 'mk_classifier': Optional classifier model
            pixel_size_um: Pixel size in micrometers (required, from CZI metadata)
            extract_features: Whether to extract full features
            extra_channels: Dict mapping channel index to 2D uint16 array for
                per-channel feature extraction (optional)

        Returns:
            Tuple of (label_array, list of Detection objects)
        """
        import torch
        from scipy import ndimage

        if pixel_size_um is None:
            raise ValueError("pixel_size_um is required for MK area filtering — do not rely on defaults")

        # Convert area thresholds from um^2 to pixels
        min_area_px = self.min_area_um / (pixel_size_um ** 2)
        max_area_px = self.max_area_um / (pixel_size_um ** 2)

        # Step 1: Generate masks with SAM2
        sam2_results = self._segment_sam2(tile, models)

        # Step 2: Pre-filter by size (before expensive feature extraction)
        valid_results = []
        for result in sam2_results:
            area = result['segmentation'].sum()
            if min_area_px <= area <= max_area_px:
                result['area'] = area
                valid_results.append(result)

        # Sort by area (largest first) for overlap priority
        valid_results.sort(key=lambda x: x['area'], reverse=True)

        # Free memory from SAM2 results
        del sam2_results
        gc.collect()

        # Step 3: Overlap filtering and collect valid detections
        valid_detections = []
        label_array = np.zeros(tile.shape[:2], dtype=np.uint32)
        det_id = 1

        for result in valid_results:
            mask = result['segmentation']

            # Ensure boolean type (critical for NVIDIA CUDA compatibility)
            if mask.dtype != bool:
                mask = (mask > 0.5).astype(bool)

            # Check overlap with already-accepted masks
            if label_array.max() > 0:
                overlap = ((mask > 0) & (label_array > 0)).sum()
                if overlap > self.overlap_threshold * mask.sum():
                    continue  # Skip - too much overlap with larger mask

            # Add to label array
            label_array[mask] = det_id

            # Compute centroid
            cy, cx = ndimage.center_of_mass(mask)

            valid_detections.append({
                'id': det_id,
                'mask': mask,
                'cy': cy,
                'cx': cx,
                'sam2_iou': float(result.get('predicted_iou', 0)),
                'sam2_stability': float(result.get('stability_score', 0))
            })
            det_id += 1

        # Free memory
        del valid_results
        gc.collect()

        # Step 4: Feature extraction (batch processing)
        detections = []

        if valid_detections and extract_features:
            # Set image for SAM2 embeddings if predictor available
            sam2_predictor = models.get('sam2_predictor')
            if sam2_predictor is not None and self.extract_sam2_embeddings:
                sam2_img = _safe_to_uint8(tile)
                sam2_predictor.set_image(sam2_img)

            # First pass: Extract morphological and SAM2 features, collect crops
            crops = []
            crops_context = []
            crop_indices = []

            for idx, det in enumerate(valid_detections):
                mask = det['mask']
                cy, cx = det['cy'], det['cx']

                # Extract morphological features
                morph = extract_morphological_features(mask, tile)

                # Extract per-channel features if extra_channels provided
                if extra_channels is not None:
                    channels_dict = {f'ch{k}': v for k, v in sorted(extra_channels.items()) if v is not None}
                    multichannel_feats = self.extract_multichannel_features(mask, channels_dict)
                    morph.update(multichannel_feats)

                # Extract SAM2 embeddings
                if sam2_predictor is not None and self.extract_sam2_embeddings:
                    sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy, cx)
                    for i, v in enumerate(sam2_emb):
                        morph[f'sam2_{i}'] = float(v)
                elif self.extract_sam2_embeddings:
                    logger.warning("SAM2 predictor unavailable - zero-filling 256D embeddings")
                    for i in range(256):
                        morph[f'sam2_{i}'] = 0.0

                det['features'] = morph

                # Prepare crops for batch ResNet/DINOv2 processing (masked + context)
                if self.extract_deep_features:
                    ys, xs = np.where(mask)
                    if len(ys) > 0:
                        y1, y2 = ys.min(), ys.max()
                        x1, x2 = xs.min(), xs.max()
                        crop_context = tile[y1:y2+1, x1:x2+1].copy()
                        crop_masked = crop_context.copy()
                        crop_mask = mask[y1:y2+1, x1:x2+1]
                        crop_masked[~crop_mask] = 0  # Zero out background
                        crops.append(crop_masked)
                        crops_context.append(crop_context)
                        crop_indices.append(idx)

            # Batch deep feature extraction (ResNet + DINOv2, masked + context)
            if self.extract_deep_features:
                resnet = models.get('resnet')
                resnet_transform = models.get('resnet_transform')
            else:
                resnet = None
                resnet_transform = None
            device = models.get('device')

            # ResNet masked
            if crops and resnet is not None and resnet_transform is not None:
                resnet_features_list = self._extract_resnet_features_batch(
                    crops, resnet, resnet_transform, device
                )
                for crop_idx, resnet_feats in zip(crop_indices, resnet_features_list):
                    for i, v in enumerate(resnet_feats):
                        valid_detections[crop_idx]['features'][f'resnet_{i}'] = float(v)

            # ResNet context
            if crops_context and resnet is not None and resnet_transform is not None:
                resnet_ctx_list = self._extract_resnet_features_batch(
                    crops_context, resnet, resnet_transform, device
                )
                for crop_idx, resnet_feats in zip(crop_indices, resnet_ctx_list):
                    for i, v in enumerate(resnet_feats):
                        valid_detections[crop_idx]['features'][f'resnet_ctx_{i}'] = float(v)

            # DINOv2 (only access model if extract_deep_features is enabled)
            if self.extract_deep_features:
                dinov2 = models.get('dinov2')
                dinov2_transform = models.get('dinov2_transform')
            else:
                dinov2 = None
                dinov2_transform = None

            if crops and dinov2 is not None and dinov2_transform is not None:
                dinov2_masked_list = self._extract_dinov2_features_batch(
                    crops, dinov2, dinov2_transform, device
                )
                for crop_idx, dino_feats in zip(crop_indices, dinov2_masked_list):
                    for i, v in enumerate(dino_feats):
                        valid_detections[crop_idx]['features'][f'dinov2_{i}'] = float(v)

                dinov2_ctx_list = self._extract_dinov2_features_batch(
                    crops_context, dinov2, dinov2_transform, device
                )
                for crop_idx, dino_feats in zip(crop_indices, dinov2_ctx_list):
                    for i, v in enumerate(dino_feats):
                        valid_detections[crop_idx]['features'][f'dinov2_ctx_{i}'] = float(v)

            # Fill zeros for detections that failed crop extraction
            if self.extract_deep_features:
                self._zero_fill_deep_features(valid_detections, has_dinov2=(dinov2 is not None))

            # Reset SAM2 predictor
            if sam2_predictor is not None:
                sam2_predictor.reset_predictor()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 5: Apply classifier if available
        mk_classifier = models.get('mk_classifier')
        mk_feature_names = models.get('mk_feature_names')

        # Build final Detection objects
        for det in valid_detections:
            features = det.get('features', {})

            # Apply classifier if available
            mk_score = 1.0
            if mk_classifier is not None and mk_feature_names is not None:
                mk_score = self._apply_classifier(features, mk_classifier, mk_feature_names)

            # Filter by classifier threshold
            if mk_score < self.classifier_threshold:
                # Remove from label array
                label_array[label_array == det['id']] = 0
                continue

            # Add SAM2 metadata to features
            features['sam2_iou'] = det['sam2_iou']
            features['sam2_stability'] = det['sam2_stability']
            features['mk_score'] = mk_score

            detection = Detection(
                mask=det['mask'],
                centroid=[float(det['cx']), float(det['cy'])],  # [x, y] format per base class
                features=features,
                score=mk_score
            )
            detections.append(detection)

        # Re-number label array to be contiguous
        if detections:
            new_label_array = np.zeros_like(label_array)
            for new_id, det in enumerate(detections, start=1):
                new_label_array[det.mask] = new_id
            label_array = new_label_array

        return label_array, detections

    # _extract_sam2_embedding inherited from DetectionStrategy base class
    # SAM2 embeddings are extracted in detect() at line ~374 (set_image + per-detection embedding).
    # _extract_resnet_features_batch inherited from DetectionStrategy base class

    def _apply_classifier(self,
                          features: Dict[str, float],
                          classifier,
                          feature_names: List[str]) -> float:
        """
        Apply trained classifier to features.

        Args:
            features: Dict of feature_name -> value
            classifier: Trained sklearn classifier with predict_proba
            feature_names: List of feature names in correct order

        Returns:
            Probability score (0-1) that this is a true MK
        """
        try:
            # Build feature vector in correct order
            X = np.array([[features.get(name, 0.0) for name in feature_names]])

            # Get probability of positive class
            proba = classifier.predict_proba(X)
            return float(proba[0, 1])  # Probability of class 1 (MK)
        except Exception as e:
            logger.warning(f"Classifier failed, defaulting to accept: {e}")
            return 1.0  # Default to accepting if classifier fails


__all__ = ['MKStrategy']
