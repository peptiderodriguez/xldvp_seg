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
    - 2-channel Cellpose input (membrane + nuclear) or nuclei-only mode
    - Area pre-filter in segment() (no count cap)
    - 6-channel feature extraction

    Detection pipeline:
    1. Construct Cellpose input:
       - PM+nuclei mode: membrane + nuclear, channels=[1,2]
       - Nuclei-only mode (membrane_channel=None): DAPI only, channels=[0,0]
    2. Cellpose detection
    3. SAM2 refinement using centroids as point prompts
    4. Overlap filtering
    5. Feature extraction from all 6 channels (mask-only, zero-excluded)
    """

    def __init__(
        self,
        membrane_channel: Optional[int] = 1,
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
            membrane_channel: CZI channel index for membrane marker (default 1, AF633).
                Set to None for nuclei-only mode (DAPI grayscale, channels=[0,0]).
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
                calibration phase. Dict mapping channel index -> raw threshold value.
                If provided, uses these instead of per-tile GMM fitting.
                (default None = fit per-tile GMM)
        """
        super().__init__(
            min_area_um=20,     # permissive safety net in filter() (um^2)
            max_area_um=2000,   # permissive safety net in filter() (um^2)
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
        masks: List[Optional[np.ndarray]],
        features: List[Dict[str, Any]],
        pixel_size_um: float
    ) -> List[Detection]:
        """
        Convert masks to Detection objects (no area filtering).

        Computes area_um2, extracts centroid, creates Detection with id/score.
        No size filtering -- Cellpose determines what constitutes a cell.
        Supports mask=None (compact mode) -- uses pre-computed area from features.
        """
        if not features:
            return []

        pixel_area_um2 = pixel_size_um ** 2

        detections = []
        for i, (mask, feat) in enumerate(zip(masks, features)):
            # Get area from mask or pre-computed feature
            if mask is not None:
                area_px = mask.sum()
            else:
                area_px = feat.get('area', 0)

            centroid = feat.get('centroid', [0, 0])
            area_um2 = area_px * pixel_area_um2
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

        WARNING: This method creates full-tile boolean masks for each cell.
        For dense tiles (17K+ cells on 4000x4000), this uses ~260 GB.
        Use detect() instead, which uses compact mask representations.

        This method is kept for backward compatibility / external callers.

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

        # --- Build Cellpose input from extra_channels ---
        nuclear_raw = extra_channels.get(self.nuclear_channel) if extra_channels else None
        if nuclear_raw is None:
            raise ValueError(
                f"IsletStrategy requires extra_channels with nuclear_channel={self.nuclear_channel}. "
                f"Available: {list(extra_channels.keys()) if extra_channels else 'None'}"
            )

        nuclear_u8 = _percentile_normalize_channel(nuclear_raw)

        if self.membrane_channel is not None:
            # PM + nuclei mode: 2-channel Cellpose input
            membrane_raw = extra_channels.get(self.membrane_channel) if extra_channels else None
            if membrane_raw is None:
                raise ValueError(
                    f"IsletStrategy requires extra_channels with membrane_channel={self.membrane_channel}. "
                    f"Available: {list(extra_channels.keys()) if extra_channels else 'None'}"
                )
            membrane_u8 = _percentile_normalize_channel(membrane_raw)
            self._membrane_u8 = membrane_u8
            cellpose_input = np.stack([membrane_u8, nuclear_u8], axis=-1)
            cellpose_channels = [1, 2]  # ch0=cytoplasm, ch1=nucleus
        else:
            # Nuclei-only mode: grayscale DAPI
            self._membrane_u8 = nuclear_u8  # SAM2 uses this for embeddings
            cellpose_input = nuclear_u8
            cellpose_channels = [0, 0]  # grayscale/nuclei

        cellpose_masks, _, _ = cellpose.eval(cellpose_input, channels=cellpose_channels)

        # Get unique mask IDs (exclude background 0)
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        logger.info(f"Cellpose found {len(cellpose_ids)} cells")
        print(f"[islet] Cellpose found {len(cellpose_ids)} cells", flush=True)

        # Extract individual boolean masks from Cellpose label array using find_objects
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

    # classify_rf() inherited from DetectionStrategy base class

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float,
        extract_features: bool = True,
        extra_channels: Dict[int, np.ndarray] = None,
    ) -> Tuple[np.ndarray, List['Detection']]:
        """
        Complete islet detection pipeline with compact mask representation.

        Uses Cellpose label array + find_objects slices instead of full-tile
        boolean masks per cell. This reduces memory from O(n_cells * tile_area)
        to O(tile_area), preventing OOM on dense tiles (17K+ cells).

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

        cellpose = models.get('cellpose')
        sam2_predictor = models.get('sam2_predictor')

        if cellpose is None:
            raise RuntimeError("Cellpose model required for islet detection")

        # =================================================================
        # Cellpose segmentation (compact mode — NO full-tile boolean masks)
        # =================================================================
        nuclear_raw = extra_channels.get(self.nuclear_channel) if extra_channels else None
        if nuclear_raw is None:
            raise ValueError(
                f"IsletStrategy requires extra_channels with nuclear_channel={self.nuclear_channel}. "
                f"Available: {list(extra_channels.keys()) if extra_channels else 'None'}"
            )

        nuclear_u8 = _percentile_normalize_channel(nuclear_raw)

        if self.membrane_channel is not None:
            # PM + nuclei mode: 2-channel Cellpose input
            membrane_raw = extra_channels.get(self.membrane_channel) if extra_channels else None
            if membrane_raw is None:
                raise ValueError(
                    f"IsletStrategy requires extra_channels with membrane_channel={self.membrane_channel}. "
                    f"Available: {list(extra_channels.keys()) if extra_channels else 'None'}"
                )
            membrane_u8 = _percentile_normalize_channel(membrane_raw)
            self._membrane_u8 = membrane_u8
            cellpose_input = np.stack([membrane_u8, nuclear_u8], axis=-1)
            cellpose_channels = [1, 2]
        else:
            # Nuclei-only mode: grayscale DAPI
            self._membrane_u8 = nuclear_u8
            cellpose_input = nuclear_u8
            cellpose_channels = [0, 0]

        cellpose_masks, _, _ = cellpose.eval(cellpose_input, channels=cellpose_channels)

        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        logger.info(f"Cellpose found {len(cellpose_ids)} cells")
        print(f"[islet] Cellpose found {len(cellpose_ids)} cells", flush=True)

        # find_objects gives bbox slices for each label — O(bbox) access per cell
        cp_slices = ndimage.find_objects(cellpose_masks)

        # Filter tiny fragments using compact masks (no full-tile booleans)
        accepted_ids = []
        n_tiny = 0
        for cp_id in cellpose_ids:
            sl = cp_slices[cp_id - 1]
            if sl is None:
                n_tiny += 1
                continue
            n_px = int(np.sum(cellpose_masks[sl] == cp_id))
            if n_px < self.min_mask_pixels:
                n_tiny += 1
                continue
            accepted_ids.append(int(cp_id))
        if n_tiny > 0:
            logger.info(f"  Rejected {n_tiny} fragments < {self.min_mask_pixels} pixels")

        del cellpose_input, nuclear_u8
        gc.collect()

        tile_shape = tile.shape[:2]

        if not accepted_ids:
            del cellpose_masks, cp_slices
            if sam2_predictor is not None:
                sam2_predictor.reset_predictor()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return np.zeros(tile_shape, dtype=np.uint32), []

        # Set SAM2 image for embedding extraction
        # Uses _membrane_u8 (membrane in PM+nuc mode, DAPI in nuclei-only mode)
        if self.extract_sam2_embeddings and sam2_predictor is not None:
            sam2_img = self._membrane_u8
            sam2_rgb = np.stack([sam2_img, sam2_img, sam2_img], axis=-1)
            sam2_predictor.set_image(sam2_rgb)

        # Feature extraction prep
        valid_detections = []
        crops_for_resnet = []
        crops_for_resnet_context = []
        crop_indices = []

        # Convert tile to uint8 for morphological feature extraction
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
                    if tile.ndim == 3:
                        tile_u8[:, :, ch][ch_data == 0] = 0
                    else:
                        tile_u8[ch_data == 0] = 0
        else:
            tile_u8 = tile

        # Precompute tile_global_mean once
        if tile_u8.ndim == 3:
            global_valid = np.max(tile_u8, axis=2) > 0
            tile_global_mean = float(np.mean(tile_u8[global_valid])) if global_valid.any() else 0
        else:
            global_valid = tile_u8 > 0
            tile_global_mean = float(np.mean(tile_u8[global_valid])) if global_valid.any() else 0
        del global_valid

        # Determine marker channels (all except bright=0, membrane, nuclear)
        non_marker_chs = {0, self.nuclear_channel}
        if self.membrane_channel is not None:
            non_marker_chs.add(self.membrane_channel)
        marker_chs = sorted(set(extra_channels.keys()) - non_marker_chs) if extra_channels else []

        # Build channels_dict once (reused for every cell)
        channels_dict = {}
        if extra_channels is not None:
            channels_dict = {
                f'ch{k}': v for k, v in sorted(extra_channels.items())
                if v is not None
            }

        # ====================================================================
        # Phase 0: Quick marker means using COMPACT masks (slice-based)
        # Uses only local bbox-sized masks (~30x30 per cell, ~1 KB each)
        # instead of full-tile booleans (4000x4000, 15 MB each).
        # Memory: 17K cells x ~1 KB = ~17 MB vs 17K x 15 MB = 260 GB
        # ====================================================================
        n_cells = len(accepted_ids)
        quick_data = []

        for idx, cp_id in enumerate(accepted_ids):
            if idx % 2000 == 0:
                print(f"[islet] Phase 0 (marker scan) {idx}/{n_cells}", flush=True)

            sl = cp_slices[cp_id - 1]
            local_mask = (cellpose_masks[sl] == cp_id)

            # Centroid in tile coordinates (offset by slice start)
            local_ys, local_xs = np.where(local_mask)
            if len(local_ys) == 0:
                continue
            cx_val = float(np.mean(local_xs) + sl[1].start)
            cy_val = float(np.mean(local_ys) + sl[0].start)

            # Quick marker channel means (raw uint16, zero-excluded)
            marker_means = {}
            for ch_idx in marker_chs:
                ch_arr = extra_channels.get(ch_idx)
                if ch_arr is not None:
                    masked_vals = ch_arr[sl][local_mask]
                    nonzero = masked_vals[masked_vals > 0]
                    marker_means[ch_idx] = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0

            quick_data.append({
                'cp_id': cp_id,
                'n_pixels': int(local_mask.sum()),
                'centroid': [cx_val, cy_val],
                'marker_means': marker_means,
            })
            # local_mask is bbox-sized (~30x30), auto-freed on next iteration

        print(f"[islet] Phase 0 done: {len(quick_data)}/{n_cells} cells scanned", flush=True)

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
                passes = True  # no thresholds -> keep all
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
        # Uses padded bbox crops (2x bbox area) instead of full-tile masks.
        # A typical cell bbox is ~30x30, padded crop ~60x60 = ~3.6 KB.
        # Peak memory: ~4 KB per cell instead of 15 MB (full-tile boolean).
        # ====================================================================
        n_sam2_extracted = 0
        H, W = tile_shape

        for pi, cell_idx in enumerate(promising_indices):
            if pi % 500 == 0:
                print(f"[islet] Featurizing cell {pi}/{n_promising}", flush=True)
            cell = quick_data[cell_idx]
            cp_id = cell['cp_id']
            sl = cp_slices[cp_id - 1]
            cx_val, cy_val = cell['centroid']

            # Padded bbox: expand by 50% on each side (100% more area)
            by0, by1 = sl[0].start, sl[0].stop
            bx0, bx1 = sl[1].start, sl[1].stop
            bh, bw = by1 - by0, bx1 - bx0
            pad_y, pad_x = bh // 2, bw // 2
            py0 = max(0, by0 - pad_y)
            py1 = min(H, by1 + pad_y)
            px0 = max(0, bx0 - pad_x)
            px1 = min(W, bx1 + pad_x)
            pad_sl = (slice(py0, py1), slice(px0, px1))

            # Local mask within padded bbox (small: e.g. 60x60)
            local_mask = (cellpose_masks[pad_sl] == cp_id)

            # Crop tile_u8 to padded bbox for morph features
            tile_crop = tile_u8[pad_sl] if tile_u8.ndim == 2 else tile_u8[py0:py1, px0:px1]

            feat = extract_morphological_features(local_mask, tile_crop, tile_global_mean=tile_global_mean)
            if not feat:
                continue

            # Per-channel features: crop each channel to padded bbox
            if channels_dict:
                cropped_channels = {
                    name: ch_arr[pad_sl] for name, ch_arr in channels_dict.items()
                    if ch_arr is not None
                }
                multichannel_feats = self.extract_multichannel_features(local_mask, cropped_channels)
                feat.update(multichannel_feats)

            # Deep features crops (use original bbox, not padded)
            if self.extract_deep_features:
                crop_context = tile[by0:by1, bx0:bx1].copy()
                orig_local_mask = (cellpose_masks[sl] == cp_id)
                crop_masked = crop_context.copy()
                crop_masked[~orig_local_mask] = 0
                crops_for_resnet.append(crop_masked)
                crops_for_resnet_context.append(crop_context)
                crop_indices.append(len(valid_detections))

            feat['centroid'] = [cx_val, cy_val]

            # SAM2 embeddings (256D) — only needs centroid, not mask
            if self.extract_sam2_embeddings and sam2_predictor is not None:
                sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy_val, cx_val)
                for si, v in enumerate(sam2_emb):
                    feat[f'sam2_{si}'] = float(v)
                n_sam2_extracted += 1
            elif self.extract_sam2_embeddings:
                for si in range(256):
                    feat[f'sam2_{si}'] = 0.0

            valid_detections.append({
                'cp_id': cp_id,
                'centroid': [cx_val, cy_val],
                'features': feat,
            })

        # ====================================================================
        # Phase 2: Minimal features for BACKGROUND cells
        # (marker means from Phase 0 + centroid + area — no morph, no SAM2)
        # No mask needed — uses pre-computed n_pixels from Phase 0.
        # ====================================================================
        for cell_idx in background_indices:
            cell = quick_data[cell_idx]
            cx_val, cy_val = cell['centroid']

            feat = {
                'centroid': [cx_val, cy_val],
                'area': cell['n_pixels'],
            }
            # Store the quick marker means as ch{idx}_mean features
            for ch_idx, val in cell['marker_means'].items():
                feat[f'ch{ch_idx}_mean'] = val

            # Zero SAM2 embeddings
            if self.extract_sam2_embeddings:
                for si in range(256):
                    feat[f'sam2_{si}'] = 0.0

            valid_detections.append({
                'cp_id': cell['cp_id'],
                'centroid': [cx_val, cy_val],
                'features': feat,
            })

        # Free quick_data
        del quick_data

        if n_background > 0:
            n_valid = len(valid_detections)
            print(f"[islet] Pre-filter saved: {n_promising} full + {n_background} minimal "
                  f"({100*n_background/n_valid:.0f}% skipped, "
                  f"~{n_background * 1.8 / 3600:.1f}h SAM2 time saved)", flush=True)
        print(f"[islet] Feature extraction done: {len(valid_detections)}/{n_cells} valid", flush=True)

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
            self._zero_fill_deep_features(valid_detections, has_dinov2=(dinov2 is not None))

        # ====================================================================
        # Build label array directly from Cellpose labels (no full masks needed)
        # ====================================================================
        label_array = np.zeros(tile_shape, dtype=np.uint32)
        for det_idx, vd in enumerate(valid_detections, start=1):
            cp_id = vd['cp_id']
            sl = cp_slices[cp_id - 1]
            label_array[sl][cellpose_masks[sl] == cp_id] = det_idx

        # Free Cellpose label array and slices (no longer needed)
        del cellpose_masks, cp_slices
        gc.collect()

        # Build Detection objects (mask=None — label_array IS the mask)
        pixel_area_um2 = pixel_size_um ** 2
        detections = []
        for i, vd in enumerate(valid_detections):
            feat = vd['features']
            area_px = feat.get('area', 0)
            feat['area_um2'] = area_px * pixel_area_um2
            detections.append(Detection(
                mask=None,
                centroid=feat.get('centroid', vd['centroid']),
                features=feat,
                id=f'{self.name}_{i}',
                score=feat.get('sam2_score', feat.get('solidity', 0.0)),
            ))

        del valid_detections

        # Cleanup
        self._membrane_u8 = None
        if sam2_predictor is not None:
            sam2_predictor.reset_predictor()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        return label_array, detections
