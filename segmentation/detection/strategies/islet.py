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
from .base import Detection
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
    - Area pre-filter in segment() (no count cap)
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
        overlap_threshold: float = 0.5,
        min_mask_pixels: int = 10,
        extract_deep_features: bool = False,
        extract_sam2_embeddings: bool = True,
        resnet_batch_size: int = 32,
        marker_signal_factor: float = 2.0,
        gmm_prefilter_thresholds: dict = None,
    ):
        """
        Initialize islet detection strategy.

        Args:
            membrane_channel: CZI channel index for membrane marker (default 1, AF633)
            nuclear_channel: CZI channel index for nuclear marker (default 4, DAPI)
            overlap_threshold: Skip masks with overlap > this fraction (default 0.5)
            min_mask_pixels: Minimum mask size in pixels (default 10)
            extract_deep_features: Whether to extract ResNet+DINOv2 (default False)
            extract_sam2_embeddings: Whether to extract SAM2 embeddings (default True)
            resnet_batch_size: Batch size for ResNet feature extraction (default 32)
            marker_signal_factor: Divisor for GMM pre-filter threshold.
                Cells need marker signal > gmm_threshold/factor to get full features.
                Higher = more permissive (2.0 = half the classification threshold).
                Set to 0 to disable pre-filtering. (default 2.0)
            gmm_prefilter_thresholds: Pre-computed global GMM thresholds from pilot
                calibration phase. Dict mapping channel index → raw threshold value.
                If provided, uses these instead of per-tile GMM fitting.
                (default None = fit per-tile GMM)
        """
        super().__init__(
            min_area_um=20,     # permissive safety net in filter() (um²)
            max_area_um=2000,   # permissive safety net in filter() (um²)
            overlap_threshold=overlap_threshold,
            min_mask_pixels=min_mask_pixels,
            extract_deep_features=extract_deep_features,
            extract_sam2_embeddings=extract_sam2_embeddings,
            resnet_batch_size=resnet_batch_size,
        )
        self.membrane_channel = membrane_channel
        self.nuclear_channel = nuclear_channel
        self.marker_signal_factor = marker_signal_factor
        self.gmm_prefilter_thresholds = gmm_prefilter_thresholds

    @property
    def name(self) -> str:
        return "islet"

    def filter(
        self,
        masks: List[np.ndarray],
        features: List[Dict[str, Any]],
        pixel_size_um: float
    ) -> List[Detection]:
        """
        Convert masks to Detection objects (no area filtering).

        Computes area_um2, extracts centroid, creates Detection with id/score.
        No size filtering — Cellpose determines what constitutes a cell.
        """
        if not masks:
            return []

        pixel_area_um2 = pixel_size_um ** 2

        detections = []
        for i, (mask, feat) in enumerate(zip(masks, features)):
            area_px = mask.sum()

            # Get centroid
            centroid = feat.get('centroid', [0, 0])

            # Compute area in microns
            area_um2 = area_px * pixel_area_um2

            # Add computed area to features
            feat['area_um2'] = area_um2

            detections.append(Detection(
                mask=mask,
                centroid=centroid,
                features=feat,
                id=f'{self.name}_{i}',
                score=feat.get('sam2_score', feat.get('solidity', 0.0))
            ))

        return detections

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
        if extra_channels is None or self.membrane_channel not in extra_channels:
            raise ValueError(
                f"IsletStrategy requires extra_channels with membrane_channel={self.membrane_channel}. "
                f"Available: {list(extra_channels.keys()) if extra_channels else 'None'}"
            )

        membrane_raw = extra_channels.get(self.membrane_channel)
        nuclear_raw = extra_channels.get(self.nuclear_channel)

        if membrane_raw is None or nuclear_raw is None:
            raise ValueError(
                f"IsletStrategy requires both membrane (ch{self.membrane_channel}) and nuclear "
                f"(ch{self.nuclear_channel}) channels. Got membrane={membrane_raw is not None}, "
                f"nuclear={nuclear_raw is not None}. Available: {list(extra_channels.keys())}"
            )

        # Percentile-normalize uint16 -> uint8 per channel
        membrane_u8 = _percentile_normalize_channel(membrane_raw)
        nuclear_u8 = _percentile_normalize_channel(nuclear_raw)

        # Cache for detect() to reuse (avoids double normalization — E7)
        self._membrane_u8 = membrane_u8

        # Stack as 2-channel input: (H, W, 2)
        cellpose_input = np.stack([membrane_u8, nuclear_u8], axis=-1)

        # Cellpose: channels=[1,2] means ch0 of input=cytoplasm, ch1=nucleus
        cellpose_masks, _, _ = cellpose.eval(cellpose_input, channels=[1, 2])

        # Get unique mask IDs (exclude background 0)
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        logger.info(f"Cellpose found {len(cellpose_ids)} cells")
        print(f"[islet] Cellpose found {len(cellpose_ids)} cells", flush=True)

        # Extract individual boolean masks from Cellpose label array using find_objects
        # No area filtering — let Cellpose decide what's a cell, only reject
        # fragments smaller than min_mask_pixels (noise).
        slices = ndimage.find_objects(cellpose_masks)
        accepted_masks = []
        n_tiny = 0
        for cp_id in cellpose_ids:
            sl = slices[cp_id - 1]  # find_objects is 1-indexed
            if sl is None:
                continue
            cp_mask = np.zeros(cellpose_masks.shape, dtype=bool)
            cp_mask[sl] = (cellpose_masks[sl] == cp_id)
            if cp_mask.sum() < self.min_mask_pixels:
                n_tiny += 1
                continue
            accepted_masks.append(cp_mask)
        if n_tiny > 0:
            logger.info(f"  Rejected {n_tiny} fragments < {self.min_mask_pixels} pixels")

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
        # Reuse cached membrane_u8 from segment() to avoid double percentile normalization (E7)
        if self.extract_sam2_embeddings and sam2_predictor is not None:
            membrane_u8 = getattr(self, '_membrane_u8', None)
            if membrane_u8 is not None:
                sam2_rgb = np.stack([membrane_u8, membrane_u8, membrane_u8], axis=-1)
            else:
                sam2_rgb = tile  # fallback to tile RGB
            sam2_predictor.set_image(sam2_rgb)

        # Feature extraction loop (reuse CellStrategy pattern)
        valid_detections = []
        crops_for_resnet = []
        crops_for_resnet_context = []
        crop_indices = []

        # Convert tile to uint8 for morphological feature extraction
        # (extract_morphological_features assumes uint8 for HSV, dark_fraction, etc.)
        if tile.dtype != np.uint8:
            tile_u8 = np.zeros_like(tile, dtype=np.uint8)
            for ch in range(tile.shape[2] if tile.ndim == 3 else 1):
                ch_data = tile[:, :, ch] if tile.ndim == 3 else tile
                nonzero = ch_data[ch_data > 0]
                if len(nonzero) > 0:
                    p1, p99 = np.percentile(nonzero, [1, 99])
                    if p99 > p1:
                        if tile.ndim == 3:
                            tile_u8[:, :, ch] = np.clip((ch_data.astype(np.float32) - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                        else:
                            tile_u8 = np.clip((ch_data.astype(np.float32) - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                    # Preserve zero pixels (CZI padding)
                    if tile.ndim == 3:
                        tile_u8[:, :, ch][ch_data == 0] = 0
                    else:
                        tile_u8[ch_data == 0] = 0
        else:
            tile_u8 = tile

        # Precompute tile_global_mean once (avoids recomputing per cell in extract_morphological_features)
        if tile_u8.ndim == 3:
            global_valid = np.max(tile_u8, axis=2) > 0
            tile_global_mean = float(np.mean(tile_u8[global_valid])) if global_valid.any() else 0
        else:
            global_valid = tile_u8 > 0
            tile_global_mean = float(np.mean(tile_u8[global_valid])) if global_valid.any() else 0
        del global_valid

        # Determine marker channels (all except bright=0, membrane, nuclear)
        non_marker_chs = {0, self.membrane_channel, self.nuclear_channel}
        marker_chs = sorted(set(extra_channels.keys()) - non_marker_chs) if extra_channels else []

        # Build channels_dict once (reused for every cell)
        channels_dict = {}
        if extra_channels is not None:
            channels_dict = {
                f'ch{k}': v for k, v in sorted(extra_channels.items())
                if v is not None
            }

        # ====================================================================
        # Phase 0: Quick marker means for ALL cells (~0.01ms/cell)
        # Just enough to compute Otsu/2 thresholds for pre-filtering.
        # ====================================================================
        n_masks = len(masks)
        quick_data = []  # list of {mask, centroid, marker_means}
        for idx, mask in enumerate(masks):
            if idx % 2000 == 0:
                print(f"[islet] Phase 0 (marker scan) {idx}/{n_masks}", flush=True)

            # Centroid
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cx_val, cy_val = float(np.mean(xs)), float(np.mean(ys))

            # Quick marker channel means (raw uint16, zero-excluded)
            marker_means = {}
            for ch_idx in marker_chs:
                ch_data = extra_channels.get(ch_idx)
                if ch_data is not None:
                    masked_vals = ch_data[mask]
                    nonzero = masked_vals[masked_vals > 0]
                    marker_means[ch_idx] = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0

            quick_data.append({
                'mask': mask,
                'centroid': [cx_val, cy_val],
                'marker_means': marker_means,
            })

        print(f"[islet] Phase 0 done: {len(quick_data)}/{n_masks} cells scanned", flush=True)

        # ====================================================================
        # GMM pre-filter: use global thresholds from pilot calibration if
        # available, otherwise fit per-tile GMM as fallback. The pre-filter
        # applies a permissive margin (threshold / marker_signal_factor).
        # ====================================================================
        gmm_thresholds = {}
        if marker_chs and self.marker_signal_factor > 0:
            divisor = max(self.marker_signal_factor, 0.01)

            if self.gmm_prefilter_thresholds:
                # Use pre-computed global thresholds from pilot calibration
                for ch_idx in marker_chs:
                    if ch_idx in self.gmm_prefilter_thresholds:
                        raw_cutoff = self.gmm_prefilter_thresholds[ch_idx] / divisor
                        gmm_thresholds[ch_idx] = raw_cutoff
                        raw_vals = np.array([c['marker_means'].get(ch_idx, 0) for c in quick_data])
                        n_above = int(np.sum(raw_vals > raw_cutoff))
                        logger.info(f"  ch{ch_idx}: global GMM P50={self.gmm_prefilter_thresholds[ch_idx]:.0f}, "
                                    f"pre-filter={raw_cutoff:.0f} (/{divisor:.1f}), "
                                    f"{n_above}/{len(raw_vals)} pass")
                if gmm_thresholds:
                    print(f"[islet] Pre-filter (global GMM/{divisor:.0f}): "
                          + ", ".join(f"ch{k}={v:.1f}" for k, v in sorted(gmm_thresholds.items())),
                          flush=True)
            else:
                # Fallback: fit per-tile GMM
                from sklearn.mixture import GaussianMixture as _GaussianMixture

                for ch_idx in marker_chs:
                    raw_vals = np.array([c['marker_means'].get(ch_idx, 0) for c in quick_data])
                    raw_pos = raw_vals[raw_vals > 0]
                    if len(raw_pos) < 50:
                        continue

                    log_vals = np.log1p(raw_pos).reshape(-1, 1)
                    gmm = _GaussianMixture(n_components=2, random_state=42, max_iter=200)
                    gmm.fit(log_vals)

                    signal_idx = int(np.argmax(gmm.means_.flatten()))
                    bg_mean = float(np.exp(gmm.means_.flatten()[1 - signal_idx]))
                    sig_mean = float(np.exp(gmm.means_.flatten()[signal_idx]))

                    # Find raw threshold where P(signal) = 0.5 via binary search
                    lo_raw, hi_raw = 0.0, float(raw_vals.max())
                    for _ in range(50):
                        mid = (lo_raw + hi_raw) / 2
                        p = gmm.predict_proba(np.log1p(np.array([[mid]])))
                        if p[0, signal_idx] > 0.5:
                            hi_raw = mid
                        else:
                            lo_raw = mid
                    gmm_threshold = (lo_raw + hi_raw) / 2
                    raw_cutoff = gmm_threshold / divisor
                    gmm_thresholds[ch_idx] = raw_cutoff

                    n_above = int(np.sum(raw_vals > raw_cutoff))
                    logger.info(f"  ch{ch_idx}: tile GMM bg={bg_mean:.0f}, sig={sig_mean:.0f}, "
                                f"P50={gmm_threshold:.0f}, pre-filter={raw_cutoff:.0f} "
                                f"(/{divisor:.1f}), {n_above}/{len(raw_vals)} pass")

                if gmm_thresholds:
                    print(f"[islet] Pre-filter (tile GMM/{divisor:.0f}): "
                          + ", ".join(f"ch{k}={v:.1f}" for k, v in sorted(gmm_thresholds.items())),
                          flush=True)

        # Split cells into promising (any marker > GMM threshold) vs background
        promising_indices = []
        background_indices = []
        for i, cell in enumerate(quick_data):
            if gmm_thresholds:
                passes = any(
                    cell['marker_means'].get(ch, 0) > gmm_thresholds[ch]
                    for ch in gmm_thresholds
                )
            else:
                passes = True  # no thresholds → keep all
            if passes:
                promising_indices.append(i)
            else:
                background_indices.append(i)

        n_promising = len(promising_indices)
        n_background = len(background_indices)
        if n_background > 0:
            pct = 100 * n_background / len(quick_data)
            print(f"[islet] Pre-filter: {n_promising} promising + {n_background} background "
                  f"({pct:.0f}% skipped before feature extraction)", flush=True)

        # ====================================================================
        # Phase 1: Full features for PROMISING cells only
        # (morph + all 6 channels + SAM2 + deep features)
        # ====================================================================
        n_sam2_extracted = 0

        for pi, cell_idx in enumerate(promising_indices):
            if pi % 500 == 0:
                print(f"[islet] Featurizing cell {pi}/{n_promising}", flush=True)
            cell = quick_data[cell_idx]
            mask = cell['mask']
            cx_val, cy_val = cell['centroid']

            feat = extract_morphological_features(mask, tile_u8, tile_global_mean=tile_global_mean)
            if not feat:
                continue

            # Per-channel features from all 6 channels
            if channels_dict:
                multichannel_feats = self.extract_multichannel_features(mask, channels_dict)
                feat.update(multichannel_feats)

            feat['centroid'] = [cx_val, cy_val]

            # SAM2 embeddings (256D)
            if self.extract_sam2_embeddings and sam2_predictor is not None:
                sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy_val, cx_val)
                for si, v in enumerate(sam2_emb):
                    feat[f'sam2_{si}'] = float(v)
                n_sam2_extracted += 1
            elif self.extract_sam2_embeddings:
                for si in range(256):
                    feat[f'sam2_{si}'] = 0.0

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

        # ====================================================================
        # Phase 2: Minimal features for BACKGROUND cells
        # (marker means from Phase 0 + centroid + area — no morph, no SAM2)
        # ====================================================================
        for cell_idx in background_indices:
            cell = quick_data[cell_idx]
            mask = cell['mask']
            cx_val, cy_val = cell['centroid']

            feat = {
                'centroid': [cx_val, cy_val],
                'area': int(mask.sum()),
            }
            # Store the quick marker means as ch{idx}_mean features
            for ch_idx, val in cell['marker_means'].items():
                feat[f'ch{ch_idx}_mean'] = val

            # Zero SAM2 embeddings
            if self.extract_sam2_embeddings:
                for si in range(256):
                    feat[f'sam2_{si}'] = 0.0

            valid_detections.append({
                'mask': mask,
                'centroid': [cx_val, cy_val],
                'features': feat,
            })

        # Free quick_data (masks are now referenced by valid_detections)
        del quick_data

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

        if n_background > 0:
            print(f"[islet] Pre-filter saved: {n_promising} full + {n_background} minimal "
                  f"({100*n_background/len(valid_detections):.0f}% skipped, "
                  f"~{n_background * 1.8 / 3600:.1f}h SAM2 time saved)", flush=True)
        print(f"[islet] Feature extraction done: {len(valid_detections)}/{n_masks} valid", flush=True)

        # Reset predictor state and clean up cached membrane_u8
        self._membrane_u8 = None
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
