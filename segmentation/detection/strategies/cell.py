"""
Generic cell detection strategy using Cellpose + SAM2.

A general-purpose cell detection pipeline:
1. Cellpose nuclei detection (auto-size detection)
2. SAM2 refinement using nuclei centroids as point prompts
3. Overlap filtering to avoid duplicate detections
4. Feature extraction (morphological + SAM2 embeddings + ResNet)
"""

import gc
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import ndimage
from skimage.measure import regionprops

from .base import DetectionStrategy, Detection
from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import (
    extract_morphological_features,
    SAM2_EMBEDDING_DIM,
    RESNET50_FEATURE_DIM,
)

logger = get_logger(__name__)


# Issue #7: Local extract_morphological_features removed - now imported from shared module


class CellStrategy(DetectionStrategy):
    """
    Generic cell detection strategy using Cellpose + SAM2 refinement.

    A general-purpose pipeline for detecting cells in microscopy images.
    Detection pipeline:
    1. Cellpose nuclei detection (auto size detection)
    2. SAM2 refinement using nuclei centroids as point prompts
    3. Overlap filtering (skip masks with >50% overlap with existing detections)
    4. Full feature extraction (22 morphological + 256 SAM2 + 2048 ResNet = 2326 features)

    Required models in detect():
        - 'cellpose': CellposeModel with cpsam pretrained model
        - 'sam2_predictor': SAM2ImagePredictor for point prompts
        - 'resnet': ResNet model for deep features (optional)
        - 'resnet_transform': Transform for ResNet (optional)
    """

    def __init__(
        self,
        min_area_um: float = 50,
        max_area_um: float = 200,
        max_candidates: int = 500,
        overlap_threshold: float = 0.5,
        min_mask_pixels: int = 10,
        resnet_batch_size: int = 32
    ):
        """
        Initialize cell detection strategy.

        Args:
            min_area_um: Minimum area in square microns (default 50)
            max_area_um: Maximum area in square microns (default 200)
            max_candidates: Maximum Cellpose candidates per tile (default 500)
            overlap_threshold: Skip masks with overlap > this fraction (default 0.5)
            min_mask_pixels: Minimum mask size in pixels (default 10)
            resnet_batch_size: Batch size for ResNet feature extraction (default 16)
        """
        self.min_area_um = min_area_um
        self.max_area_um = max_area_um
        self.max_candidates = max_candidates
        self.overlap_threshold = overlap_threshold
        self.min_mask_pixels = min_mask_pixels
        self.resnet_batch_size = resnet_batch_size

    @property
    def name(self) -> str:
        return "cell"

    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Generate cell candidate masks using Cellpose + SAM2 refinement.

        This method performs:
        1. Cellpose detection on the tile (grayscale mode, auto size)
        2. For each Cellpose detection, use its centroid as a SAM2 point prompt
        3. Take the best SAM2 mask (highest confidence)
        4. Filter by overlap with existing masks

        Args:
            tile: RGB image array (HxWx3)
            models: Dict with 'cellpose' and 'sam2_predictor'

        Returns:
            List of boolean masks, sorted by SAM2 confidence score
        """
        cellpose = models.get('cellpose')
        sam2_predictor = models.get('sam2_predictor')

        if cellpose is None:
            raise RuntimeError("Cellpose model required for cell detection")
        if sam2_predictor is None:
            raise RuntimeError("SAM2 predictor required for cell detection")

        # Ensure proper image format for SAM2
        if tile.dtype != np.uint8:
            tile_uint8 = (tile / 256).astype(np.uint8) if tile.dtype == np.uint16 else tile.astype(np.uint8)
        else:
            tile_uint8 = tile

        # Step 1: Cellpose detection (grayscale mode for nuclei)
        cellpose_masks, _, _ = cellpose.eval(tile, channels=[0, 0])

        # Get unique mask IDs (exclude background 0)
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        # Limit candidates to avoid processing thousands per tile
        if len(cellpose_ids) > self.max_candidates:
            # Sort by mask area (larger first) and take top N
            areas = [(cp_id, (cellpose_masks == cp_id).sum()) for cp_id in cellpose_ids]
            areas.sort(key=lambda x: x[1], reverse=True)
            cellpose_ids = np.array([a[0] for a in areas[:self.max_candidates]])

        # Step 2: Set image for SAM2 predictor
        sam2_predictor.set_image(tile_uint8)

        # Step 3: Collect candidates with SAM2 refinement
        candidates = []
        for cp_id in cellpose_ids:
            cp_mask = cellpose_masks == cp_id
            cy, cx = ndimage.center_of_mass(cp_mask)

            # Use centroid as SAM2 point prompt
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])

            masks_pred, scores, _ = sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # Take best mask (highest confidence)
            best_idx = np.argmax(scores)
            sam2_mask = masks_pred[best_idx]

            # Ensure boolean type
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)

            # Skip tiny masks
            if sam2_mask.sum() < self.min_mask_pixels:
                continue

            candidates.append({
                'mask': sam2_mask,
                'score': float(scores[best_idx]),
                'center': (cx, cy),
                'cellpose_id': int(cp_id)
            })

        # Sort by SAM2 confidence score (most confident first)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Step 4: Filter overlaps
        accepted_masks = []
        combined_mask = np.zeros(tile.shape[:2], dtype=bool)

        for cand in candidates:
            sam2_mask = cand['mask']

            # Check overlap with existing masks
            if combined_mask.any():
                overlap = (sam2_mask & combined_mask).sum()
                if overlap > self.overlap_threshold * sam2_mask.sum():
                    continue

            # Accept this mask
            accepted_masks.append(sam2_mask)
            combined_mask |= sam2_mask

        # Clean up
        del candidates, cellpose_masks
        gc.collect()

        return accepted_masks

    def filter(
        self,
        masks: List[np.ndarray],
        features: List[Dict[str, Any]],
        pixel_size_um: float
    ) -> List[Detection]:
        """
        Filter cell candidates by size.

        Args:
            masks: List of candidate masks
            features: List of feature dicts (one per mask)
            pixel_size_um: Pixel size in microns

        Returns:
            List of Detection objects passing size filter
        """
        if not masks:
            return []

        pixel_area_um2 = pixel_size_um ** 2
        min_area_px = self.min_area_um / pixel_area_um2
        max_area_px = self.max_area_um / pixel_area_um2

        detections = []
        for i, (mask, feat) in enumerate(zip(masks, features)):
            area_px = mask.sum()

            # Size filter
            if area_px < min_area_px or area_px > max_area_px:
                continue

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
                id=f'cell_{i}',
                score=feat.get('sam2_score', feat.get('solidity', 0.0))
            ))

        return detections

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float
    ) -> List[Detection]:
        """
        Complete cell detection pipeline with full feature extraction.

        This overrides the base detect() to add SAM2 embeddings and ResNet features.

        Args:
            tile: RGB image array
            models: Dict with 'cellpose', 'sam2_predictor', 'resnet', 'resnet_transform'
            pixel_size_um: Pixel size in microns

        Returns:
            List of Detection objects with full 2326-feature set
        """
        import torch

        # Get models
        cellpose = models.get('cellpose')
        sam2_predictor = models.get('sam2_predictor')
        resnet = models.get('resnet')
        resnet_transform = models.get('resnet_transform')
        device = models.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        if cellpose is None or sam2_predictor is None:
            raise RuntimeError("Cellpose and SAM2 predictor required for cell detection")

        # Generate masks using segment()
        masks = self.segment(tile, models)

        if not masks:
            # Reset predictor state
            sam2_predictor.reset_predictor()
            torch.cuda.empty_cache()
            return []

        # Compute features for each mask
        valid_detections = []
        crops_for_resnet = []
        crop_indices = []

        for idx, mask in enumerate(masks):
            # Basic morphological features
            feat = extract_morphological_features(mask, tile)
            if not feat:
                continue

            # Get centroid
            cy = feat.get('centroid', [0, 0])
            if isinstance(cy, list) and len(cy) >= 2:
                # centroid is [x, y], need cy, cx for embedding
                cx_val, cy_val = cy[0], cy[1]
            else:
                ys, xs = np.where(mask)
                cy_val, cx_val = np.mean(ys), np.mean(xs)

            # SAM2 embeddings (256D)
            sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy_val, cx_val)
            for i, v in enumerate(sam2_emb):
                feat[f'sam2_emb_{i}'] = float(v)

            # Prepare crop for batch ResNet processing
            ys, xs = np.where(mask)
            if len(ys) > 0:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                crop = tile[y1:y2+1, x1:x2+1].copy()
                crop_mask = mask[y1:y2+1, x1:x2+1]
                crop[~crop_mask] = 0

                crops_for_resnet.append(crop)
                crop_indices.append(idx)

            valid_detections.append({
                'idx': idx,
                'mask': mask,
                'centroid': [cx_val, cy_val],  # [x, y]
                'features': feat
            })

        # Batch ResNet feature extraction (using inherited method from base class)
        if crops_for_resnet and resnet is not None and resnet_transform is not None:
            resnet_features = self._extract_resnet_features_batch(
                crops_for_resnet, resnet, resnet_transform, device,
                batch_size=self.resnet_batch_size
            )

            # Assign ResNet features to correct detections
            for crop_idx, resnet_feat in zip(crop_indices, resnet_features):
                # Find matching detection
                for det in valid_detections:
                    if det['idx'] == crop_idx:
                        for i, v in enumerate(resnet_feat):
                            det['features'][f'resnet_{i}'] = float(v)
                        break

        # Fill zeros for detections without ResNet features
        for det in valid_detections:
            if 'resnet_0' not in det['features']:
                for i in range(2048):
                    det['features'][f'resnet_{i}'] = 0.0

        # Reset predictor state
        sam2_predictor.reset_predictor()
        torch.cuda.empty_cache()

        # Build Detection objects and filter by size
        features_list = [det['features'] for det in valid_detections]
        masks_list = [det['mask'] for det in valid_detections]

        return self.filter(masks_list, features_list, pixel_size_um)

    # _extract_sam2_embedding inherited from DetectionStrategy base class
    # _extract_resnet_features_batch inherited from DetectionStrategy base class

    def compute_features(
        self,
        mask: np.ndarray,
        tile: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute morphological features for a single mask.

        Note: For full feature extraction including SAM2 embeddings and
        ResNet features, use detect() which handles batch processing.

        Args:
            mask: Binary mask
            tile: Original tile image

        Returns:
            Dictionary of morphological features
        """
        return extract_morphological_features(mask, tile)


# Backward compatibility alias
HSPCStrategy = CellStrategy
