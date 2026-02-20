"""
Pancreatic islet cell detection strategy using Cellpose + SAM2.

Uses membrane (AF633) + nuclear (DAPI) channels as 2-channel Cellpose input
for cell segmentation, then SAM2 refinement. Extracts features from all 6
channels for downstream RF classification and clustering.

6-channel CZI layout:
  Ch0: Bright (brightfield, not used for detection)
  Ch1: AF633 (membrane marker — Cellpose input)
  Ch2: AF555/Gcg (alpha cells — display Red)
  Ch3: AF488/Ins (beta cells — display Green)
  Ch4: DAPI (nucleus — Cellpose input)
  Ch5: Cy7/Sst (delta cells — display Blue)
"""

import gc
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import ndimage

from .cell import CellStrategy
from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import extract_morphological_features

logger = get_logger(__name__)


def _percentile_normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize a single uint16 channel to uint8 using percentile clipping.

    Computes p1/p99.5 on non-zero pixels, maps linearly to 0-255.
    Zero pixels (CZI padding) stay 0.

    Args:
        channel: 2D uint16 array

    Returns:
        2D uint8 array
    """
    nonzero = channel[channel > 0]
    if len(nonzero) == 0:
        return np.zeros(channel.shape, dtype=np.uint8)

    p_low = np.percentile(nonzero, 1)
    p_high = np.percentile(nonzero, 99.5)

    if p_high <= p_low:
        return np.zeros(channel.shape, dtype=np.uint8)

    result = np.zeros(channel.shape, dtype=np.float32)
    mask = channel > 0
    result[mask] = (channel[mask].astype(np.float32) - p_low) / (p_high - p_low) * 255.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


class IsletStrategy(CellStrategy):
    """
    Pancreatic islet cell detection using Cellpose (membrane+nuclear) + SAM2.

    Extends CellStrategy with:
    - 2-channel Cellpose input (membrane + nuclear) instead of grayscale
    - Larger max_candidates for dense islet tissue
    - 6-channel feature extraction

    Detection pipeline:
    1. Construct 2-channel input: membrane (AF633) + nuclear (DAPI)
    2. Cellpose detection with channels=[1,2] (cytoplasm + nucleus)
    3. SAM2 refinement using centroids as point prompts
    4. Overlap filtering
    5. Feature extraction from all 6 channels (mask-only, zero-excluded)
    """

    def __init__(
        self,
        membrane_channel: int = 1,
        nuclear_channel: int = 4,
        min_area_um: float = 30,
        max_area_um: float = 500,
        max_candidates: int = 1000,
        overlap_threshold: float = 0.5,
        min_mask_pixels: int = 10,
        extract_deep_features: bool = False,
        extract_sam2_embeddings: bool = True,
        resnet_batch_size: int = 32,
    ):
        """
        Initialize islet detection strategy.

        Args:
            membrane_channel: CZI channel index for membrane marker (default 1, AF633)
            nuclear_channel: CZI channel index for nuclear marker (default 4, DAPI)
            min_area_um: Minimum cell area in um2 (default 30)
            max_area_um: Maximum cell area in um2 (default 500)
            max_candidates: Max Cellpose candidates per tile (default 1000)
            overlap_threshold: Skip masks with overlap > this fraction (default 0.5)
            min_mask_pixels: Minimum mask size in pixels (default 10)
            extract_deep_features: Whether to extract ResNet+DINOv2 (default False)
            extract_sam2_embeddings: Whether to extract SAM2 embeddings (default True)
            resnet_batch_size: Batch size for ResNet feature extraction (default 32)
        """
        super().__init__(
            min_area_um=min_area_um,
            max_area_um=max_area_um,
            max_candidates=max_candidates,
            overlap_threshold=overlap_threshold,
            min_mask_pixels=min_mask_pixels,
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=resnet_batch_size,
        )
        self.membrane_channel = membrane_channel
        self.nuclear_channel = nuclear_channel

    @property
    def name(self) -> str:
        return "islet"

    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        extra_channels: Optional[Dict[int, np.ndarray]] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Segment islet cells using 2-channel Cellpose (membrane + nuclear).

        SAM2 refinement is intentionally SKIPPED for islet because dense
        islet tissue causes SAM2 point prompts to return whole-tissue regions
        (100K+ px) instead of individual cells. Cellpose with membrane+nuclear
        already produces high-quality individual cell masks.

        SAM2 embeddings are still extracted per-cell in detect() for downstream
        classification features.

        Args:
            tile: RGB image array (HxWx3, uint8)
            models: Dict with 'cellpose' (and optionally 'sam2_predictor')
            extra_channels: Dict mapping CZI channel index to 2D uint16 arrays.
                Must contain membrane_channel and nuclear_channel.

        Returns:
            List of boolean masks sorted by area (largest first)
        """
        cellpose = models.get('cellpose')

        if cellpose is None:
            raise RuntimeError("Cellpose model required for islet detection")

        # --- Build 2-channel Cellpose input from extra_channels ---
        if extra_channels is None:
            logger.warning("No extra_channels for islet — falling back to grayscale Cellpose")
            return super().segment(tile, models, **kwargs)

        membrane_raw = extra_channels.get(self.membrane_channel)
        nuclear_raw = extra_channels.get(self.nuclear_channel)

        if membrane_raw is None or nuclear_raw is None:
            logger.warning(
                f"Missing membrane (ch{self.membrane_channel}) or nuclear "
                f"(ch{self.nuclear_channel}) — falling back to grayscale Cellpose"
            )
            return super().segment(tile, models, **kwargs)

        # Percentile-normalize uint16 -> uint8 per channel
        membrane_u8 = _percentile_normalize_channel(membrane_raw)
        nuclear_u8 = _percentile_normalize_channel(nuclear_raw)

        # Stack as 2-channel input: (H, W, 2)
        cellpose_input = np.stack([membrane_u8, nuclear_u8], axis=-1)

        # Cellpose: channels=[1,2] means ch0 of input=cytoplasm, ch1=nucleus
        cellpose_masks, _, _ = cellpose.eval(cellpose_input, channels=[1, 2])

        # Get unique mask IDs (exclude background 0)
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        logger.info(f"Cellpose found {len(cellpose_ids)} cells")

        # Limit candidates by area (keep largest)
        if len(cellpose_ids) > self.max_candidates:
            areas = [(cp_id, (cellpose_masks == cp_id).sum()) for cp_id in cellpose_ids]
            areas.sort(key=lambda x: x[1], reverse=True)
            cellpose_ids = np.array([a[0] for a in areas[:self.max_candidates]])
            logger.info(f"  Limited to {len(cellpose_ids)} largest candidates")

        # Extract individual boolean masks from Cellpose label array
        # No overlap filtering needed — Cellpose masks are non-overlapping by design
        accepted_masks = []
        for cp_id in cellpose_ids:
            cp_mask = (cellpose_masks == cp_id).astype(bool)
            if cp_mask.sum() < self.min_mask_pixels:
                continue
            accepted_masks.append(cp_mask)

        # No SAM2 scores — use None placeholder (detect() handles this)
        self._last_segment_scores = []

        logger.info(f"  Accepted {len(accepted_masks)} masks (>= {self.min_mask_pixels} px)")

        del cellpose_masks
        gc.collect()

        return accepted_masks

    def classify_rf(
        self,
        detections: list,
        classifier,
        scaler,
        feature_names: list,
    ) -> list:
        """
        Classify islet candidates using trained Random Forest.

        load_nmj_rf_classifier() always wraps into a Pipeline, so classifier
        is always a Pipeline that handles scaling internally.

        Args:
            detections: List of Detection objects to classify
            classifier: sklearn Pipeline (or bare RF model)
            scaler: Unused (Pipeline handles scaling), kept for interface compat
            feature_names: List of feature names expected by classifier

        Returns:
            List of Detection objects with updated scores (keeps ALL)
        """
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

        # Update detections with RF probability (keep ALL — filter post-hoc in HTML)
        for j, (idx, prob) in enumerate(zip(valid_indices, probs)):
            det = detections[idx]
            det.score = float(prob)
            det.features['rf_prediction'] = float(prob)
            det.features['confidence'] = float(prob)

        # Set score=0 for detections that couldn't be classified (missing features)
        for i, det in enumerate(detections):
            if det.score is None:
                det.score = 0.0
                det.features['rf_prediction'] = 0.0
                det.features['confidence'] = 0.0

        n_above = sum(1 for d in detections if d.score >= 0.5)
        logger.debug(f"RF classifier: {n_above}/{len(detections)} above 0.5, keeping all")

        return detections

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float,
        extract_features: bool = True,
        extra_channels: Dict[int, np.ndarray] = None,
    ) -> Tuple[np.ndarray, List['Detection']]:
        """
        Complete islet detection pipeline.

        Overrides CellStrategy.detect() to pass extra_channels into segment().

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
            raise RuntimeError("Cellpose model required for islet detection")

        # Generate masks using our 2-channel segment()
        masks = self.segment(tile, models, extra_channels=extra_channels)

        if not masks:
            if sam2_predictor is not None:
                sam2_predictor.reset_predictor()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        # Set SAM2 image for embedding extraction (segment() no longer uses SAM2)
        if self.extract_sam2_embeddings and sam2_predictor is not None:
            membrane_raw = extra_channels.get(self.membrane_channel) if extra_channels else None
            if membrane_raw is not None:
                membrane_u8 = _percentile_normalize_channel(membrane_raw)
                sam2_rgb = np.stack([membrane_u8, membrane_u8, membrane_u8], axis=-1)
            else:
                sam2_rgb = tile  # fallback to tile RGB
            sam2_predictor.set_image(sam2_rgb)

        # Feature extraction loop (reuse CellStrategy pattern)
        valid_detections = []
        crops_for_resnet = []
        crops_for_resnet_context = []
        crop_indices = []

        # No SAM2 scores from segment() (Cellpose-only)
        segment_scores = getattr(self, '_last_segment_scores', [])

        for idx, mask in enumerate(masks):
            feat = extract_morphological_features(mask, tile)
            if not feat:
                continue

            # SAM2 confidence score from segment()
            if idx < len(segment_scores):
                feat['sam2_score'] = float(segment_scores[idx])

            # Per-channel features from all 6 channels
            if extra_channels is not None:
                channels_dict = {
                    f'ch{k}': v for k, v in sorted(extra_channels.items())
                    if v is not None
                }
                multichannel_feats = self.extract_multichannel_features(mask, channels_dict)
                feat.update(multichannel_feats)

            # Compute centroid from mask (extract_morphological_features does NOT return centroid)
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
