"""
Tissue pattern detection strategy using Cellpose on summed detection channels.

Designed for FISH brain sections but generalizable to any multi-channel tissue:
- Sums configurable detection channels for Cellpose input (grayscale mode)
- No SAM2 refinement (dense tissue causes whole-region expansion)
- SAM2 embeddings still extracted per-cell for downstream features
- Per-channel feature extraction from all available channels

Default 5-channel CZI layout (brain FISH):
  Ch0: AF488  -> Slc17a7 (excitatory neurons, detection)
  Ch1: AF647  -> Htr2a   (serotonin receptor, analysis)
  Ch2: AF750  -> Ntrk2   (BDNF receptor, analysis)
  Ch3: AF555  -> Gad1    (inhibitory neurons, detection)
  Ch4: Hoechst -> Nuclear (tissue detection)
"""

import gc
import numpy as np
from scipy import ndimage
from typing import Dict, Any, List, Optional, Tuple

from .cell import CellStrategy
from .islet import _percentile_normalize_channel
from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import extract_morphological_features

logger = get_logger(__name__)


class TissuePatternStrategy(CellStrategy):
    """
    Tissue pattern cell detection using Cellpose on summed channels.

    Extends CellStrategy with:
    - Summed multi-channel Cellpose input (grayscale mode)
    - No SAM2 refinement (dense tissue)
    - Configurable detection and display channels
    - Multi-channel feature extraction

    Detection pipeline:
    1. Sum detection channels (percentile-normalized to uint8 each, then summed+clipped)
    2. Cellpose detection with channels=[0,0] (grayscale)
    3. Area filtering
    4. Feature extraction from all channels (mask-only, zero-excluded)
    5. SAM2 embedding extraction (using summed channel as pseudo-RGB)
    """

    def __init__(
        self,
        detection_channels: List[int] = None,
        nuclear_channel: int = 4,
        min_area_um: float = 20,
        max_area_um: float = 300,
        overlap_threshold: float = 0.5,
        min_mask_pixels: int = 10,
        extract_deep_features: bool = False,
        extract_sam2_embeddings: bool = True,
        resnet_batch_size: int = 32,
    ):
        super().__init__(
            min_area_um=min_area_um,
            max_area_um=max_area_um,
            overlap_threshold=overlap_threshold,
            min_mask_pixels=min_mask_pixels,
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=resnet_batch_size,
        )
        self.detection_channels = detection_channels if detection_channels is not None else [0, 3]
        self.nuclear_channel = nuclear_channel

    @property
    def name(self) -> str:
        return "tissue_pattern"

    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        extra_channels: Optional[Dict[int, np.ndarray]] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Segment cells using Cellpose on summed detection channels (grayscale).

        SAM2 refinement is intentionally SKIPPED for dense tissue (same reason
        as islet — point prompts expand to whole-tissue regions).

        Args:
            tile: RGB image array (HxWx3, uint8)
            models: Dict with 'cellpose'
            extra_channels: Dict mapping CZI channel index to 2D uint16 arrays.
                Must contain all detection_channels.

        Returns:
            List of boolean masks sorted by area (largest first)
        """
        self._last_summed_channel = None  # Reset for each tile

        cellpose = models.get('cellpose')

        if cellpose is None:
            raise RuntimeError("Cellpose model required for tissue_pattern detection")

        if extra_channels is None:
            logger.warning("No extra_channels for tissue_pattern — falling back to grayscale Cellpose")
            return super().segment(tile, models, **kwargs)

        # Sum detection channels (each percentile-normalized to uint8 first)
        ch_arrays = []
        for ch_idx in self.detection_channels:
            ch_raw = extra_channels.get(ch_idx)
            if ch_raw is None:
                logger.warning(
                    f"Missing detection channel {ch_idx} — falling back to grayscale Cellpose"
                )
                return super().segment(tile, models, **kwargs)
            ch_arrays.append(_percentile_normalize_channel(ch_raw))

        # Sum and clip to uint8
        summed = np.zeros(ch_arrays[0].shape, dtype=np.uint16)
        for ch_u8 in ch_arrays:
            summed += ch_u8.astype(np.uint16)
        summed = np.clip(summed, 0, 255).astype(np.uint8)

        # Store for detect() to use for SAM2 image setting
        self._last_summed_channel = summed

        # Cellpose grayscale: channels=[0,0]
        cellpose_masks, _, _ = cellpose.eval(summed, channels=[0, 0])

        # Get unique mask IDs (exclude background 0)
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        logger.info(f"Cellpose found {len(cellpose_ids)} cells")

        # Area filtering (same pattern as islet)
        pixel_area_um2 = kwargs.get('pixel_size_um', 0.1725) ** 2
        min_area_px = int(self.min_area_um / pixel_area_um2) if pixel_area_um2 > 0 else 10
        max_area_px = int(self.max_area_um / pixel_area_um2) if pixel_area_um2 > 0 else 100000

        # Compute areas for all cells in O(M) using bincount (not O(N×H×W) per cell)
        areas_all = np.bincount(cellpose_masks.ravel())
        areas = [(cp_id, areas_all[cp_id]) for cp_id in cellpose_ids]

        in_range = [(cp_id, a) for cp_id, a in areas if min_area_px <= a <= max_area_px]
        out_of_range = len(areas) - len(in_range)
        if out_of_range > 0:
            logger.info(f"  Area filter: {len(in_range)} in range [{self.min_area_um}-{self.max_area_um} um²], "
                        f"{out_of_range} rejected")

        # Extract masks using find_objects for bbox-scoped extraction (not full-tile scan per cell)
        slices = ndimage.find_objects(cellpose_masks)
        accepted_masks = []
        for cp_id, _ in in_range:
            sl = slices[cp_id - 1]  # find_objects is 1-indexed
            if sl is None:
                continue
            cp_mask = np.zeros(cellpose_masks.shape, dtype=bool)
            cp_mask[sl] = (cellpose_masks[sl] == cp_id)
            if cp_mask.sum() < self.min_mask_pixels:
                continue
            accepted_masks.append(cp_mask)

        # No SAM2 scores — Cellpose-only
        self._last_segment_scores = []

        logger.info(f"  Accepted {len(accepted_masks)} masks (>= {self.min_mask_pixels} px)")

        del cellpose_masks
        gc.collect()

        return accepted_masks

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float,
        extract_features: bool = True,
        extra_channels: Dict[int, np.ndarray] = None,
    ) -> Tuple[np.ndarray, List['Detection']]:
        """
        Complete tissue pattern detection pipeline.

        Args:
            tile: RGB image array (uint8)
            models: Dict with 'cellpose', 'sam2_predictor', etc.
            pixel_size_um: Pixel size in microns
            extract_features: Whether to extract features
            extra_channels: Dict mapping CZI channel index to 2D uint16 arrays

        Returns:
            Tuple of (label_array, list of Detection objects)
        """
        import torch
        from .base import Detection

        sam2_predictor = models.get('sam2_predictor')

        if models.get('cellpose') is None:
            raise RuntimeError("Cellpose model required for tissue_pattern detection")

        # Generate masks using summed-channel segment()
        masks = self.segment(tile, models, extra_channels=extra_channels,
                             pixel_size_um=pixel_size_um)

        if not masks:
            if sam2_predictor is not None:
                sam2_predictor.reset_predictor()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        # Set SAM2 image using summed channel as pseudo-RGB (for embedding extraction)
        if self.extract_sam2_embeddings and sam2_predictor is not None:
            summed = getattr(self, '_last_summed_channel', None)
            if summed is not None:
                sam2_rgb = np.stack([summed, summed, summed], axis=-1)
            else:
                sam2_rgb = tile
            sam2_predictor.set_image(sam2_rgb)

        # Feature extraction loop
        valid_detections = []
        crops_for_resnet = []
        crops_for_resnet_context = []
        crop_indices = []

        segment_scores = getattr(self, '_last_segment_scores', [])

        # Precompute tile_global_mean once (avoids recomputing per cell in extract_morphological_features)
        if tile.ndim == 3:
            global_valid = np.max(tile, axis=2) > 0
            tile_global_mean = float(np.mean(tile[global_valid])) if global_valid.any() else 0
        else:
            global_valid = tile > 0
            tile_global_mean = float(np.mean(tile[global_valid])) if global_valid.any() else 0
        del global_valid

        n_masks = len(masks)
        for idx, mask in enumerate(masks):
            if idx % 500 == 0:
                print(f"[tissue_pattern] Featurizing cell {idx}/{n_masks}", flush=True)
            feat = extract_morphological_features(mask, tile, tile_global_mean=tile_global_mean)
            if not feat:
                continue

            if idx < len(segment_scores):
                feat['sam2_score'] = float(segment_scores[idx])

            # Per-channel features from all channels
            if extra_channels is not None:
                channels_dict = {
                    f'ch{k}': v for k, v in sorted(extra_channels.items())
                    if v is not None
                }
                multichannel_feats = self.extract_multichannel_features(mask, channels_dict)
                feat.update(multichannel_feats)

            # Compute centroid from mask
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cx_val, cy_val = float(np.mean(xs)), float(np.mean(ys))
            feat['centroid'] = [cx_val, cy_val]

            # SAM2 embeddings (256D)
            if self.extract_sam2_embeddings and sam2_predictor is not None:
                sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy_val, cx_val)
                for i, v in enumerate(sam2_emb):
                    feat[f'sam2_{i}'] = float(v)
            elif self.extract_sam2_embeddings:
                for i in range(256):
                    feat[f'sam2_{i}'] = 0.0

            # Deep features (opt-in)
            if self.extract_deep_features:
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()
                    crop_context = tile[y1:y2+1, x1:x2+1].copy()
                    crop_masked = crop_context.copy()
                    crop_mask = mask[y1:y2+1, x1:x2+1]
                    crop_masked[~crop_mask] = 0
                    crops_for_resnet.append(crop_masked)
                    crops_for_resnet_context.append(crop_context)
                    crop_indices.append(len(valid_detections))

            valid_detections.append({
                'mask': mask,
                'centroid': [cx_val, cy_val],
                'features': feat,
            })

        # Batch deep feature extraction
        if self.extract_deep_features:
            device = models.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            resnet = models.get('resnet')
            resnet_transform = models.get('resnet_transform')

            if crops_for_resnet and resnet is not None and resnet_transform is not None:
                resnet_features = self._extract_resnet_features_batch(
                    crops_for_resnet, resnet, resnet_transform, device,
                    batch_size=self.resnet_batch_size,
                )
                for crop_idx, resnet_feat in zip(crop_indices, resnet_features):
                    for i, v in enumerate(resnet_feat):
                        valid_detections[crop_idx]['features'][f'resnet_{i}'] = float(v)

            if crops_for_resnet_context and resnet is not None and resnet_transform is not None:
                resnet_ctx_features = self._extract_resnet_features_batch(
                    crops_for_resnet_context, resnet, resnet_transform, device,
                    batch_size=self.resnet_batch_size,
                )
                for crop_idx, resnet_feat in zip(crop_indices, resnet_ctx_features):
                    for i, v in enumerate(resnet_feat):
                        valid_detections[crop_idx]['features'][f'resnet_ctx_{i}'] = float(v)

            dinov2 = models.get('dinov2')
            dinov2_transform = models.get('dinov2_transform')

            if crops_for_resnet and dinov2 is not None and dinov2_transform is not None:
                dinov2_masked = self._extract_dinov2_features_batch(
                    crops_for_resnet, dinov2, dinov2_transform, device,
                )
                for crop_idx, dino_feat in zip(crop_indices, dinov2_masked):
                    for i, v in enumerate(dino_feat):
                        valid_detections[crop_idx]['features'][f'dinov2_{i}'] = float(v)

                dinov2_ctx = self._extract_dinov2_features_batch(
                    crops_for_resnet_context, dinov2, dinov2_transform, device,
                )
                for crop_idx, dino_feat in zip(crop_indices, dinov2_ctx):
                    for i, v in enumerate(dino_feat):
                        valid_detections[crop_idx]['features'][f'dinov2_ctx_{i}'] = float(v)

            # Fill zeros for missing deep features
            for det in valid_detections:
                if 'resnet_0' not in det['features']:
                    for i in range(2048):
                        det['features'][f'resnet_{i}'] = 0.0
                if 'resnet_ctx_0' not in det['features']:
                    for i in range(2048):
                        det['features'][f'resnet_ctx_{i}'] = 0.0
                if dinov2 is not None and 'dinov2_0' not in det['features']:
                    for i in range(1024):
                        det['features'][f'dinov2_{i}'] = 0.0
                if dinov2 is not None and 'dinov2_ctx_0' not in det['features']:
                    for i in range(1024):
                        det['features'][f'dinov2_ctx_{i}'] = 0.0

        # Reset predictor state
        if sam2_predictor is not None:
            sam2_predictor.reset_predictor()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build Detection objects and filter by size
        features_list = [det['features'] for det in valid_detections]
        masks_list = [det['mask'] for det in valid_detections]

        detections = self.filter(masks_list, features_list, pixel_size_um)

        if not detections:
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        # Optional RF classifier scoring
        if 'classifier' in models:
            classifier_type = models.get('classifier_type', 'rf')
            if classifier_type == 'rf':
                classifier = models['classifier']
                scaler = models.get('scaler')
                feature_names = models.get('feature_names', [])
                detections = self.classify_rf(
                    detections, classifier, scaler, feature_names
                )

        # Build label array
        label_array = np.zeros(tile.shape[:2], dtype=np.uint32)
        for i, det in enumerate(detections, start=1):
            if det.mask is not None:
                label_array[det.mask] = i

        return label_array, detections

    def classify_rf(self, detections, classifier, scaler, feature_names):
        """Classify candidates using trained Random Forest (same as IsletStrategy)."""
        if not detections:
            return []

        X = []
        valid_indices = []

        for i, det in enumerate(detections):
            if det.features:
                row = []
                for fn in feature_names:
                    val = det.features.get(fn, 0)
                    if isinstance(val, (list, tuple)):
                        val = 0
                    row.append(float(val) if val is not None else 0)
                X.append(row)
                valid_indices.append(i)

        if not X:
            return detections

        X = np.array(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        probs = classifier.predict_proba(X)[:, 1]

        for j, (idx, prob) in enumerate(zip(valid_indices, probs)):
            det = detections[idx]
            det.score = float(prob)
            det.features['rf_prediction'] = float(prob)
            det.features['confidence'] = float(prob)

        for i, det in enumerate(detections):
            if det.score is None:
                det.score = 0.0
                det.features['rf_prediction'] = 0.0
                det.features['confidence'] = 0.0

        n_above = sum(1 for d in detections if d.score >= 0.5)
        logger.debug(f"RF classifier: {n_above}/{len(detections)} above 0.5, keeping all")

        return detections
