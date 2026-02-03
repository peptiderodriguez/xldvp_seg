"""
NMJ (Neuromuscular Junction) detection strategy.

NMJs are detected using a unique multi-stage approach:
1. Intensity thresholding - NMJs appear as bright fluorescent regions
2. Solidity filtering - NMJs have branched shapes with low solidity (max_solidity=0.85)
3. Watershed expansion - Masks are expanded to capture full BTX signal
4. Smoothing - Binary smoothing before and after expansion
5. Optional ResNet classifier - Trained to distinguish NMJ vs non-NMJ
6. Full feature extraction (22 morphological + 256 SAM2 + 2048 ResNet = 2326 features)

This strategy uses intensity thresholding combined with morphological analysis,
and optionally extracts SAM2 embeddings and ResNet features for ML classification.

Reference parameters from the original pipeline:
- intensity_percentile: 99 (default)
- min_area: 150 pixels (default)
- min_skeleton_length: 30 pixels (default)
- max_solidity: 0.85 (default, lower = more branched)
"""

import gc
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from scipy import ndimage
from skimage.morphology import skeletonize, remove_small_objects, binary_opening, binary_closing, binary_dilation, disk
from skimage.measure import label, regionprops

from .base import DetectionStrategy, Detection
from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import (
    extract_morphological_features,
    SAM2_EMBEDDING_DIM,
    RESNET50_FEATURE_DIM,
)

logger = get_logger(__name__)


# Issue #7: Local extract_morphological_features removed - now imported from shared module


class NMJStrategy(DetectionStrategy):
    """
    NMJ detection strategy using intensity threshold + solidity filtering.

    NMJs in fluorescent muscle tissue images appear as bright, branched
    structures with low solidity (area/convex_hull). This strategy:

    1. Thresholds bright regions (high intensity percentile)
    2. Applies morphological cleanup (opening/closing)
    3. Filters by skeleton length and solidity (branched structures have low solidity)
    4. Smooths and expands masks for better coverage
    5. Full feature extraction (22 morphological + 256 SAM2 + 2048 ResNet = 2326 features)
    6. Optionally classifies with a trained ResNet model

    Parameters:
        intensity_percentile: Percentile threshold for bright regions (default 99)
        max_solidity: Maximum solidity to pass filter (default 0.85, lower = more branched)
        min_skeleton_length: Minimum skeleton length in pixels (default 30)
        min_area_px: Minimum area in pixels (default 150)
        max_area_px: Maximum area in pixels (default None = no limit)
        min_area_um: Minimum area in um^2 (default 25, used if pixel_size known)
        classifier_threshold: Confidence threshold for classifier (default 0.75)
        use_classifier: Whether to use ResNet classifier (default True if model provided)
        extract_resnet_features: Whether to extract 2048D ResNet features (default True)
        extract_sam2_embeddings: Whether to extract 256D SAM2 embeddings (default True)
        resnet_batch_size: Batch size for ResNet feature extraction (default 16)
    """

    def __init__(
        self,
        intensity_percentile: float = 99.0,
        max_solidity: float = 0.85,
        min_skeleton_length: int = 30,
        min_area_px: int = 150,
        max_area_px: Optional[int] = None,
        min_area_um: float = 25.0,
        classifier_threshold: float = 0.75,
        use_classifier: bool = True,
        extract_resnet_features: bool = True,
        extract_sam2_embeddings: bool = True,
        resnet_batch_size: int = 32
    ):
        self.intensity_percentile = intensity_percentile
        self.max_solidity = max_solidity
        self.min_skeleton_length = min_skeleton_length
        self.min_area_px = min_area_px
        self.max_area_px = max_area_px
        self.min_area_um = min_area_um
        self.classifier_threshold = classifier_threshold
        self.use_classifier = use_classifier
        self.extract_resnet_features = extract_resnet_features
        self.extract_sam2_embeddings = extract_sam2_embeddings
        self.resnet_batch_size = resnet_batch_size

    @property
    def name(self) -> str:
        return "nmj"

    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Segment NMJ candidates using intensity threshold + solidity filtering.

        This method does NOT use SAM2 or Cellpose - it uses simple thresholding
        followed by connected component analysis and solidity-based filtering.
        Masks are smoothed and expanded for better coverage.

        Args:
            tile: Input tile image (HxW grayscale or HxWx3 RGB)
            models: Not used for initial segmentation, but may contain classifier

        Returns:
            List of binary masks for NMJ candidates that pass solidity filter
        """
        # Convert to grayscale - use BTX channel (green/index 1) for thresholding
        # When tile is multi-channel RGB: R=nuclear(ch0), G=BTX(ch1), B=NFL(ch2)
        # We want to threshold on BTX only since that's the NMJ marker
        if tile.ndim == 3:
            gray = tile[:, :, 1].astype(float)  # BTX channel only
        else:
            gray = tile.astype(float)

        # Check for empty tile
        if gray.max() == 0:
            return []

        # Threshold bright regions
        threshold = np.percentile(gray, self.intensity_percentile)
        bright_mask = gray > threshold

        # Morphological cleanup
        # Opening removes small bright noise
        # Closing fills small holes in bright regions
        bright_mask = binary_opening(bright_mask, disk(1))
        bright_mask = binary_closing(bright_mask, disk(2))

        # Remove objects smaller than min_area
        bright_mask = remove_small_objects(bright_mask, min_size=self.min_area_px)

        # Label connected components
        labeled = label(bright_mask)
        props = regionprops(labeled, intensity_image=gray)

        # Filter by solidity (branched structures have low solidity)
        nmj_masks = []

        for prop in props:
            if prop.area < self.min_area_px:
                continue

            if self.max_area_px is not None and prop.area > self.max_area_px:
                continue

            # Create mask for this region
            region_mask = (labeled == prop.label)

            # Compute skeleton length
            skeleton = skeletonize(region_mask)
            skeleton_length = skeleton.sum()

            # Filter by skeleton length and solidity (low solidity = branched)
            if skeleton_length >= self.min_skeleton_length and prop.solidity <= self.max_solidity:
                # Smooth mask: closing then opening
                smoothed = binary_closing(region_mask, disk(2))
                smoothed = binary_opening(smoothed, disk(2))

                # Adaptive expansion: grow until signal drops
                expanded = self._expand_to_signal_edge(smoothed, gray)

                # Final smoothing to clean up watershed edges
                final = binary_closing(expanded, disk(1))
                final = binary_opening(final, disk(1))
                nmj_masks.append(final.astype(bool))

        return nmj_masks

    def compute_features(
        self,
        mask: np.ndarray,
        tile: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute NMJ-specific features including skeleton-based metrics.

        In addition to standard morphological features, computes:
        - skeleton_length: Length of medial axis in pixels
        - solidity: Area / convex hull area (low = branched)

        Args:
            mask: Binary mask of the NMJ
            tile: Original tile image

        Returns:
            Dictionary of features
        """
        if mask.sum() == 0:
            return {}

        # Get intensity image
        if tile.ndim == 3:
            gray = np.mean(tile[:, :, :3], axis=2)
        else:
            gray = tile.astype(float)

        # Compute skeleton
        skeleton = skeletonize(mask)
        skeleton_length = int(skeleton.sum())

        # Get regionprops
        props = regionprops(mask.astype(int), intensity_image=gray)

        if not props:
            return {}

        prop = props[0]

        return {
            'area': int(prop.area),
            'centroid': [float(prop.centroid[1]), float(prop.centroid[0])],  # [x, y]
            'skeleton_length': skeleton_length,
            'eccentricity': float(prop.eccentricity),
            'solidity': float(prop.solidity),
            'mean_intensity': float(prop.mean_intensity),
            'perimeter': float(prop.perimeter),
            'bbox': list(prop.bbox),  # [min_row, min_col, max_row, max_col]
        }

    def filter(
        self,
        masks: List[np.ndarray],
        features: List[Dict[str, Any]],
        pixel_size_um: float
    ) -> List[Detection]:
        """
        Filter NMJ candidates by area threshold.

        Note: Elongation filtering is already done in segment().
        This method applies additional area filtering in um^2.

        Args:
            masks: List of candidate masks
            features: List of feature dictionaries
            pixel_size_um: Pixel size for area conversion

        Returns:
            List of Detection objects that pass filtering
        """
        detections = []

        for mask, feat in zip(masks, features):
            if not feat:
                continue

            # Compute area in um^2
            area_um2 = feat['area'] * (pixel_size_um ** 2)

            # Filter by area in um^2
            if area_um2 < self.min_area_um:
                continue

            # Create Detection object
            detection = Detection(
                mask=mask,
                centroid=feat['centroid'],
                features={
                    **feat,
                    'area_um2': area_um2,
                },
                id=f"nmj_{len(detections) + 1}",
                score=None  # Will be set by classifier
            )

            detections.append(detection)

        return detections

    def classify(
        self,
        detections: List[Detection],
        tile: np.ndarray,
        classifier_model,
        transform,
        device,
        batch_size: int = 32
    ) -> List[Detection]:
        """
        Classify NMJ candidates using trained ResNet model.

        Args:
            detections: List of Detection objects to classify
            tile: Original tile image (for crop extraction)
            classifier_model: Loaded PyTorch ResNet model
            transform: Torchvision transform for preprocessing
            device: Torch device
            batch_size: Batch size for inference

        Returns:
            List of Detection objects with updated scores
        """
        import torch

        if not detections:
            return []

        # Convert tile to RGB for classifier
        if tile.ndim == 2:
            tile_rgb = np.stack([tile] * 3, axis=-1)
        else:
            tile_rgb = tile[:, :, :3] if tile.shape[2] >= 3 else np.stack([tile[:, :, 0]] * 3, axis=-1)

        # Normalize tile
        tile_rgb = self._percentile_normalize(tile_rgb)

        # Extract crops
        crops = []
        valid_indices = []

        for i, det in enumerate(detections):
            crop = self._extract_crop(tile_rgb, det.centroid)
            if crop is not None:
                crops.append(crop)
                valid_indices.append(i)

        if not crops:
            return detections

        # Batch inference
        classifier_model.eval()
        results = {}

        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i:i+batch_size]
                batch_indices = valid_indices[i:i+batch_size]

                # Transform and stack
                batch_tensors = torch.stack([transform(c) for c in batch_crops]).to(device)

                # Forward pass
                outputs = classifier_model(batch_tensors)
                probs = torch.softmax(outputs, dim=1)

                for j, (idx, prob) in enumerate(zip(batch_indices, probs)):
                    prob_nmj = prob[1].item()
                    results[idx] = prob_nmj

        # Update detections with scores
        classified_detections = []
        for i, det in enumerate(detections):
            if i in results:
                det.score = results[i]
                det.features['prob_nmj'] = results[i]
                det.features['confidence'] = results[i]

                # Filter by classifier threshold
                if results[i] >= self.classifier_threshold:
                    classified_detections.append(det)
            else:
                # No classification, keep if not using classifier
                if not self.use_classifier:
                    classified_detections.append(det)

        return classified_detections

    def classify_rf(
        self,
        detections: List[Detection],
        classifier,
        scaler,
        feature_names: List[str]
    ) -> List[Detection]:
        """
        Classify NMJ candidates using trained Random Forest model.

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
            List of Detection objects that pass classification
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

        # Update detections with scores
        classified_detections = []
        for j, (idx, prob) in enumerate(zip(valid_indices, probs)):
            det = detections[idx]
            det.score = prob
            det.features['prob_nmj'] = prob
            det.features['confidence'] = prob

            # Filter by classifier threshold
            if prob >= self.classifier_threshold:
                classified_detections.append(det)

        logger.debug(f"RF classifier: {len(classified_detections)}/{len(detections)} passed (threshold={self.classifier_threshold})")

        return classified_detections

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float,
        extract_full_features: bool = True,
        extra_channels: Dict[int, np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Complete NMJ detection pipeline with full feature extraction.

        When extra_channels is provided, extracts per-channel features from all
        channels, significantly increasing the feature count for better classification.

        Pipeline:
        1. Segment candidates (intensity threshold + solidity filtering)
        2. Extract full features (morphological + SAM2 + ResNet)
           - With extra_channels: ~2400+ features (22 morph/channel + ratios + embeddings)
           - Without: ~2326 features
        3. Filter by area
        4. Optional classifier filtering

        Args:
            tile: Input tile image (RGB - can be true multi-channel if all channels loaded)
            models: Dict with optional keys:
                - 'sam2_predictor': SAM2ImagePredictor (for embeddings)
                - 'resnet': ResNet model (for features)
                - 'resnet_transform': torchvision transform
                - 'device': torch device
                - 'classifier': Optional NMJ classifier model
                - 'transform': Transform for classifier
            pixel_size_um: Pixel size for area calculations
            extract_full_features: Whether to extract all features (default True)
            extra_channels: Dict mapping channel index to 2D array for per-channel
                feature extraction. Keys are channel numbers (0, 1, 2), values are
                grayscale tiles. If provided, extracts features from each channel.

        Returns:
            Tuple of (label_array, list of Detection objects with full features)
        """
        import torch

        # Step 1: Segment candidates (includes elongation filtering)
        masks = self.segment(tile, models)

        if not masks:
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        # Prepare image for SAM2/ResNet
        if tile.ndim == 2:
            tile_rgb = np.stack([tile] * 3, axis=-1)
        elif tile.shape[2] == 1:
            tile_rgb = np.concatenate([tile, tile, tile], axis=-1)
        else:
            tile_rgb = tile[:, :, :3]

        # Ensure uint8 format
        if tile_rgb.dtype != np.uint8:
            if tile_rgb.dtype == np.uint16:
                tile_rgb = (tile_rgb / 256).astype(np.uint8)
            else:
                tile_rgb = tile_rgb.astype(np.uint8)

        # Get models - only load ResNet/DINOv2 if we'll use them
        sam2_predictor = models.get('sam2_predictor')
        device = models.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Only access ResNet/DINOv2 if extract_resnet_features is enabled (avoids triggering lazy load)
        if self.extract_resnet_features:
            resnet = models.get('resnet')
            resnet_transform = models.get('resnet_transform')
        else:
            resnet = None
            resnet_transform = None

        # Set image for SAM2 embeddings if available
        if sam2_predictor is not None and self.extract_sam2_embeddings and extract_full_features:
            sam2_predictor.set_image(tile_rgb)

        # Step 2: Extract features for each mask
        valid_detections = []
        crops_for_resnet = []  # Masked crops
        crops_for_resnet_context = []  # Context crops (unmasked)
        crop_indices = []
        label_array = np.zeros(tile.shape[:2], dtype=np.uint32)
        det_id = 1

        for idx, mask in enumerate(masks):
            # Extract 22 morphological features
            morph_features = extract_morphological_features(mask, tile_rgb)
            if not morph_features:
                continue

            # Compute NMJ-specific features (skeleton-based)
            nmj_specific = self._compute_nmj_specific_features(mask, tile)
            morph_features.update(nmj_specific)

            # Extract per-channel features if extra_channels provided
            if extra_channels is not None and extract_full_features:
                multichannel_feats = self._compute_multichannel_features(mask, extra_channels)
                morph_features.update(multichannel_feats)

            # Get centroid
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cy, cx = np.mean(ys), np.mean(xs)

            # Extract SAM2 embeddings (256D)
            if sam2_predictor is not None and self.extract_sam2_embeddings and extract_full_features:
                sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy, cx)
                for i, v in enumerate(sam2_emb):
                    morph_features[f'sam2_{i}'] = float(v)
            elif extract_full_features:
                # Fill with zeros if SAM2 not available
                for i in range(256):
                    morph_features[f'sam2_{i}'] = 0.0

            # Prepare crops for batch ResNet/DINOv2 processing (masked + context)
            if self.extract_resnet_features and extract_full_features:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                # Context crop (unmasked - full tissue)
                crop_context = tile_rgb[y1:y2+1, x1:x2+1].copy()
                # Masked crop (background zeroed out)
                crop_masked = crop_context.copy()
                crop_mask = mask[y1:y2+1, x1:x2+1]
                crop_masked[~crop_mask] = 0
                crops_for_resnet.append(crop_masked)
                crops_for_resnet_context.append(crop_context)
                crop_indices.append(len(valid_detections))

            valid_detections.append({
                'idx': idx,
                'mask': mask,
                'centroid': [float(cx), float(cy)],  # [x, y]
                'features': morph_features,
                'cy': cy,
                'cx': cx
            })

            # Add to label array
            label_array[mask] = det_id
            det_id += 1

        # Batch ResNet feature extraction - masked
        if crops_for_resnet and resnet is not None and resnet_transform is not None and extract_full_features:
            resnet_features_list = self._extract_resnet_features_batch(
                crops_for_resnet, resnet, resnet_transform, device
            )

            # Assign masked ResNet features to correct detections
            for crop_idx, resnet_feats in zip(crop_indices, resnet_features_list):
                for i, v in enumerate(resnet_feats):
                    valid_detections[crop_idx]['features'][f'resnet_{i}'] = float(v)

        # Batch ResNet feature extraction - context (unmasked)
        if crops_for_resnet_context and resnet is not None and resnet_transform is not None and extract_full_features:
            resnet_context_list = self._extract_resnet_features_batch(
                crops_for_resnet_context, resnet, resnet_transform, device
            )

            # Assign context ResNet features to correct detections
            for crop_idx, resnet_feats in zip(crop_indices, resnet_context_list):
                for i, v in enumerate(resnet_feats):
                    valid_detections[crop_idx]['features'][f'resnet_ctx_{i}'] = float(v)

        # Batch DINOv2 feature extraction (masked + context)
        # Only access DINOv2 if extract_resnet_features is enabled (avoids triggering lazy load)
        if self.extract_resnet_features:
            dinov2 = models.get('dinov2')
            dinov2_transform = models.get('dinov2_transform')
        else:
            dinov2 = None
            dinov2_transform = None

        if crops_for_resnet and dinov2 is not None and dinov2_transform is not None and extract_full_features:
            # DINOv2 masked features
            dinov2_masked_list = self._extract_dinov2_features_batch(
                crops_for_resnet, dinov2, dinov2_transform, device
            )
            for crop_idx, dino_feats in zip(crop_indices, dinov2_masked_list):
                for i, v in enumerate(dino_feats):
                    valid_detections[crop_idx]['features'][f'dinov2_{i}'] = float(v)

            # DINOv2 context features
            dinov2_context_list = self._extract_dinov2_features_batch(
                crops_for_resnet_context, dinov2, dinov2_transform, device
            )
            for crop_idx, dino_feats in zip(crop_indices, dinov2_context_list):
                for i, v in enumerate(dino_feats):
                    valid_detections[crop_idx]['features'][f'dinov2_ctx_{i}'] = float(v)

        # Fill zeros for detections without ResNet/DINOv2 features (only if extraction is enabled)
        if extract_full_features and self.extract_resnet_features:
            for det in valid_detections:
                if 'resnet_0' not in det['features']:
                    for i in range(2048):
                        det['features'][f'resnet_{i}'] = 0.0
                if 'resnet_ctx_0' not in det['features']:
                    for i in range(2048):
                        det['features'][f'resnet_ctx_{i}'] = 0.0
                if dinov2 is not None and 'dinov2_0' not in det['features']:
                    for i in range(1024):  # DINOv2-L dimension
                        det['features'][f'dinov2_{i}'] = 0.0
                if dinov2 is not None and 'dinov2_ctx_0' not in det['features']:
                    for i in range(1024):
                        det['features'][f'dinov2_ctx_{i}'] = 0.0

        # Reset SAM2 predictor
        if sam2_predictor is not None and self.extract_sam2_embeddings:
            try:
                sam2_predictor.reset_predictor()
            except Exception as e:
                logger.debug(f"Failed to reset SAM2 predictor: {e}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        # Step 3: Filter by area and create Detection objects
        masks_list = [det['mask'] for det in valid_detections]
        features_list = [det['features'] for det in valid_detections]

        # Add centroid to features for filter method
        for det in valid_detections:
            det['features']['centroid'] = det['centroid']

        detections = self.filter(masks_list, features_list, pixel_size_um)

        if not detections:
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        # Step 4: Optional classifier filtering
        if self.use_classifier and 'classifier' in models:
            classifier = models['classifier']
            classifier_type = models.get('classifier_type', 'cnn')

            if classifier_type == 'rf':
                # Random Forest classifier - uses extracted features
                scaler = models.get('scaler')
                feature_names = models.get('feature_names', [])
                detections = self.classify_rf(
                    detections, classifier, scaler, feature_names
                )
            else:
                # CNN classifier - uses image crops
                transform = models.get('transform')
                if transform is not None and device is not None:
                    detections = self.classify(
                        detections, tile, classifier, transform, device
                    )

        # Re-build label array with only final detections
        final_label_array = np.zeros(tile.shape[:2], dtype=np.uint32)
        for new_id, det in enumerate(detections, start=1):
            final_label_array[det.mask] = new_id

        return final_label_array, detections

    def _compute_nmj_specific_features(
        self,
        mask: np.ndarray,
        tile: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute NMJ-specific features including skeleton-based metrics.

        Args:
            mask: Binary mask
            tile: Original tile image

        Returns:
            Dictionary of NMJ-specific features
        """
        if mask.sum() == 0:
            return {}

        # Get intensity image
        if tile.ndim == 3:
            gray = np.mean(tile[:, :, :3], axis=2)
        else:
            gray = tile.astype(float)

        # Compute skeleton
        skeleton = skeletonize(mask)
        skeleton_length = int(skeleton.sum())

        # Get regionprops
        props = regionprops(mask.astype(int), intensity_image=gray)
        if not props:
            return {}

        prop = props[0]

        return {
            'skeleton_length': skeleton_length,
            'solidity': float(prop.solidity),
            'eccentricity': float(prop.eccentricity),
            'mean_intensity': float(prop.mean_intensity),
            'bbox': list(prop.bbox),
        }

    # _extract_sam2_embedding inherited from DetectionStrategy base class
    # _extract_resnet_features_batch inherited from DetectionStrategy base class
    # _percentile_normalize inherited from DetectionStrategy base class

    def _compute_multichannel_features(
        self,
        mask: np.ndarray,
        extra_channels: Dict[int, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compute features from all available channels for better NMJ classification.

        Extracts intensity statistics, texture features, and inter-channel ratios
        from each channel. This provides the classifier with much richer information
        than single-channel features alone.

        Channel mapping for NMJ:
        - ch0: Nuclear (488nm) - should be LOW in real NMJs
        - ch1: BTX (647nm) - should be HIGH in real NMJs (target signal)
        - ch2: NFL (750nm) - neurofilament marker

        Args:
            mask: Binary mask of the NMJ candidate
            extra_channels: Dict mapping channel index to 2D grayscale tile

        Returns:
            Dict with per-channel features and inter-channel ratios
        """
        if mask.sum() == 0:
            return {}

        features = {}

        # Compute per-channel features
        channel_means = {}
        channel_stds = {}
        channel_maxes = {}
        channel_medians = {}

        for ch_idx, ch_data in sorted(extra_channels.items()):
            if ch_data is None:
                continue

            # Ensure shapes match
            if ch_data.shape != mask.shape:
                continue

            # Get masked pixels
            masked_pixels = ch_data[mask].astype(float)
            if len(masked_pixels) == 0:
                continue

            # Basic intensity statistics
            ch_mean = float(np.mean(masked_pixels))
            ch_std = float(np.std(masked_pixels))
            ch_max = float(np.max(masked_pixels))
            ch_min = float(np.min(masked_pixels))
            ch_median = float(np.median(masked_pixels))

            # Percentiles for robust statistics
            ch_p5 = float(np.percentile(masked_pixels, 5))
            ch_p25 = float(np.percentile(masked_pixels, 25))
            ch_p75 = float(np.percentile(masked_pixels, 75))
            ch_p95 = float(np.percentile(masked_pixels, 95))

            # Texture/distribution features
            ch_variance = float(np.var(masked_pixels))
            ch_skewness = float(self._safe_skewness(masked_pixels))
            ch_kurtosis = float(self._safe_kurtosis(masked_pixels))
            ch_iqr = ch_p75 - ch_p25  # Interquartile range
            ch_dynamic_range = ch_max - ch_min

            # Store for ratio calculations
            channel_means[ch_idx] = ch_mean
            channel_stds[ch_idx] = ch_std
            channel_maxes[ch_idx] = ch_max
            channel_medians[ch_idx] = ch_median

            # Add to features with channel prefix
            prefix = f'ch{ch_idx}'
            features[f'{prefix}_mean'] = ch_mean
            features[f'{prefix}_std'] = ch_std
            features[f'{prefix}_max'] = ch_max
            features[f'{prefix}_min'] = ch_min
            features[f'{prefix}_median'] = ch_median
            features[f'{prefix}_p5'] = ch_p5
            features[f'{prefix}_p25'] = ch_p25
            features[f'{prefix}_p75'] = ch_p75
            features[f'{prefix}_p95'] = ch_p95
            features[f'{prefix}_variance'] = ch_variance
            features[f'{prefix}_skewness'] = ch_skewness
            features[f'{prefix}_kurtosis'] = ch_kurtosis
            features[f'{prefix}_iqr'] = ch_iqr
            features[f'{prefix}_dynamic_range'] = ch_dynamic_range

        # Compute inter-channel ratios (critical for distinguishing NMJs from autofluorescence)
        # Real NMJs: high BTX (ch1), low nuclear (ch0)
        # Autofluorescence: high signal in all channels including nuclear

        if 0 in channel_means and 1 in channel_means:
            # BTX / nuclear ratio - should be HIGH for real NMJs
            btx = channel_means[1]
            nuclear = max(channel_means[0], 1)  # Avoid div by zero
            features['btx_nuclear_ratio'] = btx / nuclear
            features['btx_nuclear_diff'] = btx - channel_means[0]

        if 1 in channel_means and 2 in channel_means:
            # BTX / NFL ratio
            btx = channel_means[1]
            nfl = max(channel_means[2], 1)
            features['btx_nfl_ratio'] = btx / nfl
            features['btx_nfl_diff'] = btx - channel_means[2]

        if 0 in channel_means and 2 in channel_means:
            # Nuclear / NFL ratio
            nuclear = channel_means[0]
            nfl = max(channel_means[2], 1)
            features['nuclear_nfl_ratio'] = nuclear / nfl

        # Channel specificity: primary (BTX) vs max of other channels
        if 1 in channel_means:
            btx = channel_means[1]
            other_means = [v for k, v in channel_means.items() if k != 1]
            if other_means:
                max_other = max(other_means)
                features['channel_specificity'] = btx / max(max_other, 1)
                features['channel_specificity_diff'] = btx - max_other

        # Coefficient of variation per channel (useful for texture)
        for ch_idx in channel_means:
            if channel_stds.get(ch_idx, 0) > 0 and channel_means.get(ch_idx, 0) > 0:
                features[f'ch{ch_idx}_cv'] = channel_stds[ch_idx] / channel_means[ch_idx]

        return features

    def _safe_skewness(self, data: np.ndarray) -> float:
        """Compute skewness safely, returning 0 if not enough data."""
        if len(data) < 3:
            return 0.0
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            return 0.0

    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis safely, returning 0 if not enough data."""
        if len(data) < 4:
            return 0.0
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            return 0.0

    def _expand_to_signal_edge(
        self,
        mask: np.ndarray,
        intensity_image: np.ndarray,
        low_threshold_percentile: float = 95.0,
        max_area_growth: float = 1.0
    ) -> np.ndarray:
        """
        Expand mask to signal boundaries using watershed.

        Uses the initial mask as seeds and grows via watershed until
        reaching low-intensity boundaries. The expansion is constrained
        to regions above a lower intensity threshold and limited to
        a maximum proportional growth of the original mask area.

        Args:
            mask: Initial binary mask (seeds)
            intensity_image: Grayscale intensity image
            low_threshold_percentile: Lower threshold for expansion region (default 95)
            max_area_growth: Maximum fractional growth of mask area (default 1.0 = 100% growth)

        Returns:
            Expanded binary mask
        """
        from skimage.segmentation import watershed

        if mask.sum() == 0:
            return mask.astype(bool)

        # Calculate max expansion radius based on proportional area growth
        # To grow area by factor (1 + max_area_growth), radius grows by sqrt(1 + max_area_growth)
        # expansion_radius = r_orig * (sqrt(1 + max_area_growth) - 1)
        original_area = mask.sum()
        effective_radius = np.sqrt(original_area / np.pi)
        max_expansion_radius = int(np.ceil(effective_radius * (np.sqrt(1 + max_area_growth) - 1)))
        max_expansion_radius = max(max_expansion_radius, 2)  # At least 2 pixels

        # Create expansion region: lower threshold than initial detection
        low_threshold = np.percentile(intensity_image, low_threshold_percentile)
        expansion_region = intensity_image > low_threshold

        # Limit expansion to max_expansion_radius from original mask
        max_reach = binary_dilation(mask, disk(max_expansion_radius))
        expansion_region = expansion_region & max_reach

        # Use negative intensity as elevation map (watershed fills valleys)
        # Higher intensity = lower elevation = easier to fill
        elevation = -intensity_image.astype(float)

        # Create markers: 1 = seed (NMJ), 2 = background
        markers = np.zeros_like(mask, dtype=np.int32)
        markers[mask] = 1
        markers[~expansion_region] = 2  # Background outside expansion region

        # Run watershed
        labels = watershed(elevation, markers, mask=expansion_region)

        # Return expanded mask (label 1)
        return (labels == 1).astype(bool)

    def _extract_crop(
        self,
        tile_rgb: np.ndarray,
        centroid: List[float],
        zoom_factor: float = 7.5,
        output_size: int = 300
    ) -> Optional[Image.Image]:
        """
        Extract crop centered on centroid with zoom factor.

        Args:
            tile_rgb: RGB tile image (normalized to uint8)
            centroid: [x, y] center coordinates
            zoom_factor: How much context to include (larger = more zoomed out)
            output_size: Final crop size in pixels

        Returns:
            PIL Image of the crop, or None if invalid
        """
        cx, cy = int(centroid[0]), int(centroid[1])

        # Calculate crop size
        crop_size = int(output_size * zoom_factor / 7.5)
        half = crop_size // 2

        h, w = tile_rgb.shape[:2]
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        # Validate crop bounds before extracting
        if y2 <= y1 or x2 <= x1:
            return None

        crop = tile_rgb[y1:y2, x1:x2].copy()

        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None

        # Resize to output size
        pil_img = Image.fromarray(crop)
        pil_img = pil_img.resize((output_size, output_size), Image.LANCZOS)

        return pil_img


def load_nmj_classifier(model_path: str, device=None):
    """
    Load a trained NMJ ResNet18 classifier.

    Args:
        model_path: Path to checkpoint (.pth file)
        device: Torch device (default: CUDA if available)

    Returns:
        Tuple of (model, transform, device)
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return model, transform, device


def load_nmj_rf_classifier(model_path: str):
    """
    Load a trained NMJ Random Forest classifier.

    Supports two formats:
    1. Dict with 'model', 'scaler', 'feature_names' keys (legacy)
    2. sklearn Pipeline with scaler + RF (new format from train_morph_sam2_classifier.py)
       - Feature names loaded from companion JSON file

    Args:
        model_path: Path to pickle/joblib file (.pkl or .joblib)

    Returns:
        Dict with 'pipeline' (sklearn Pipeline), 'feature_names', 'type'='rf'
    """
    import joblib
    import json
    from pathlib import Path
    from sklearn.pipeline import Pipeline

    model_data = joblib.load(model_path)

    # Check if it's a Pipeline (new format) or dict (legacy format)
    if isinstance(model_data, Pipeline):
        # New format: Pipeline saved directly, feature names in companion JSON
        pipeline = model_data

        # Look for companion feature names file
        model_dir = Path(model_path).parent
        feature_names_path = model_dir / "nmj_classifier_feature_names.json"

        if feature_names_path.exists():
            with open(feature_names_path) as f:
                feature_names = json.load(f)
            logger.info(f"Loaded feature names from {feature_names_path}")
        else:
            # Try to infer feature count from model
            n_features = pipeline.named_steps['rf'].n_features_in_
            feature_names = [f"feature_{i}" for i in range(n_features)]
            logger.warning(f"No feature names file found, using generic names for {n_features} features")

        result = {
            'pipeline': pipeline,
            'feature_names': feature_names,
            'type': 'rf'
        }
        logger.info(f"Loaded RF Pipeline classifier with {len(feature_names)} features")

    else:
        # Legacy format: dict with model, scaler, feature_names
        # Wrap in pipeline-like structure for consistent interface
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if 'scaler' in model_data and 'model' in model_data:
            pipeline = Pipeline([
                ('scaler', model_data['scaler']),
                ('rf', model_data['model'])
            ])
        else:
            pipeline = model_data.get('model', model_data)

        result = {
            'pipeline': pipeline,
            'feature_names': model_data.get('feature_names', []),
            'type': 'rf'
        }
        logger.info(f"Loaded RF classifier (legacy format) with {len(result['feature_names'])} features")
        if 'accuracy' in model_data:
            logger.info(f"  Accuracy: {model_data['accuracy']}")

    return result


def load_classifier(model_path: str, device=None):
    """
    Load NMJ classifier - auto-detects CNN (.pth) vs RF (.pkl/.joblib).

    Args:
        model_path: Path to model file
        device: Torch device (for CNN only)

    Returns:
        Dict with classifier info and 'type' key ('cnn' or 'rf')
    """
    if model_path.endswith('.pkl') or model_path.endswith('.joblib'):
        return load_nmj_rf_classifier(model_path)
    else:
        model, transform, device = load_nmj_classifier(model_path, device)
        return {
            'model': model,
            'transform': transform,
            'device': device,
            'type': 'cnn'
        }


def _expand_to_signal_edge_simple(
    mask: np.ndarray,
    intensity_image: np.ndarray,
    low_threshold_percentile: float = 95.0,
    max_area_growth: float = 1.0
) -> np.ndarray:
    """
    Expand mask to signal boundaries using watershed.

    Standalone version for use with detect_nmjs_simple().

    Args:
        mask: Initial binary mask (seeds)
        intensity_image: Grayscale intensity image
        low_threshold_percentile: Lower threshold for expansion region (default 95)
        max_area_growth: Maximum fractional growth of mask area (default 1.0 = 100% growth)

    Returns:
        Expanded binary mask
    """
    from skimage.segmentation import watershed

    if mask.sum() == 0:
        return mask.astype(bool)

    # Calculate max expansion radius based on proportional area growth
    original_area = mask.sum()
    effective_radius = np.sqrt(original_area / np.pi)
    max_expansion_radius = int(np.ceil(effective_radius * (np.sqrt(1 + max_area_growth) - 1)))
    max_expansion_radius = max(max_expansion_radius, 2)  # At least 2 pixels

    # Create expansion region with lower threshold
    low_threshold = np.percentile(intensity_image, low_threshold_percentile)
    expansion_region = intensity_image > low_threshold

    # Limit expansion to max_expansion_radius from original mask
    max_reach = binary_dilation(mask, disk(max_expansion_radius))
    expansion_region = expansion_region & max_reach

    # Negative intensity as elevation (high intensity = low elevation)
    elevation = -intensity_image.astype(float)

    # Markers: 1 = seed, 2 = background
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[mask] = 1
    markers[~expansion_region] = 2

    # Watershed expansion
    labels = watershed(elevation, markers, mask=expansion_region)

    return (labels == 1).astype(bool)


def detect_nmjs_simple(
    image: np.ndarray,
    intensity_percentile: float = 99,
    min_area: int = 150,
    min_skeleton_length: int = 30,
    max_solidity: float = 0.85
) -> tuple:
    """
    Simple NMJ detection function (standalone, no class).

    This is the original detect_nmjs() function from run_nmj_segmentation.py,
    kept for backwards compatibility.

    Args:
        image: RGB or grayscale image
        intensity_percentile: Percentile for bright region threshold (default 99)
        min_area: Minimum NMJ area in pixels
        min_skeleton_length: Minimum skeleton length in pixels
        max_solidity: Maximum solidity (default 0.85, lower = more branched)

    Returns:
        Tuple of (nmj_masks, nmj_features):
        - nmj_masks: Label array with NMJ IDs (0 = background)
        - nmj_features: List of feature dictionaries
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = np.mean(image[:, :, :3], axis=2)
    else:
        gray = image.astype(float)

    # Check for empty image
    if gray.max() == 0:
        return np.zeros(image.shape[:2], dtype=np.uint32), []

    # Threshold bright regions
    threshold = np.percentile(gray, intensity_percentile)
    bright_mask = gray > threshold

    # Morphological cleanup
    bright_mask = binary_opening(bright_mask, disk(1))
    bright_mask = binary_closing(bright_mask, disk(2))
    bright_mask = remove_small_objects(bright_mask, min_size=min_area)

    # Label connected components
    labeled = label(bright_mask)
    props = regionprops(labeled, intensity_image=gray)

    # Filter by solidity (branched structures have low solidity)
    nmj_masks = np.zeros(image.shape[:2], dtype=np.uint32)
    nmj_features = []
    nmj_id = 1

    for prop in props:
        if prop.area < min_area:
            continue

        region_mask = labeled == prop.label
        skeleton = skeletonize(region_mask)
        skeleton_length = skeleton.sum()

        if skeleton_length >= min_skeleton_length and prop.solidity <= max_solidity:
            # Smooth mask: closing then opening
            smoothed = binary_closing(region_mask, disk(2))
            smoothed = binary_opening(smoothed, disk(2))

            # Adaptive expansion: grow until signal drops
            expanded = _expand_to_signal_edge_simple(smoothed, gray)

            # Final smoothing to clean up watershed edges
            final = binary_closing(expanded, disk(1))
            final = binary_opening(final, disk(1))

            nmj_masks[final] = nmj_id

            nmj_features.append({
                'id': f'nmj_{nmj_id}',
                'area': int(prop.area),
                'skeleton_length': int(skeleton_length),
                'eccentricity': float(prop.eccentricity),
                'mean_intensity': float(prop.mean_intensity),
                'centroid': [float(prop.centroid[1]), float(prop.centroid[0])],  # x, y
                'bbox': list(prop.bbox),
                'perimeter': float(prop.perimeter),
                'solidity': float(prop.solidity),
            })

            nmj_id += 1

    return nmj_masks, nmj_features
