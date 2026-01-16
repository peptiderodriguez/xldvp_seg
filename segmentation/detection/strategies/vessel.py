"""
Vessel detection strategy.

Detects blood vessel cross-sections (ring structures) in SMA-stained tissue
using contour hierarchy analysis and ellipse fitting.

Full feature extraction: 22 morphological + 256 SAM2 + 2048 ResNet = 2326 features
plus vessel-specific features (wall thickness, diameters, etc.)
"""

import gc
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from .base import DetectionStrategy, Detection
from segmentation.utils.logging import get_logger
from segmentation.utils.feature_extraction import (
    extract_morphological_features,
    SAM2_EMBEDDING_DIM,
    RESNET50_FEATURE_DIM,
)

logger = get_logger(__name__)


# Issue #7: Local extract_morphological_features removed - now imported from shared module


class VesselStrategy(DetectionStrategy):
    """
    Vessel detection strategy for ring structures.

    Vessels are detected using:
    1. Canny edge detection + Otsu thresholding for SMA+ regions
    2. Contour hierarchy analysis (RETR_CCOMP) to find parent-child pairs
    3. Ellipse fitting for outer (adventitia) and inner (lumen) boundaries
    4. Wall thickness measurement via distance transform + skeleton analysis
    5. Optional CD31 validation (endothelial marker at lumen boundary)
    6. Full feature extraction (22 morphological + 256 SAM2 + 2048 ResNet = 2326 features)

    Ring structures are identified as outer contours that have inner contours
    (holes), representing the vessel wall surrounding the lumen.

    Parameters:
        min_diameter_um: Minimum outer diameter in microns (default: 10)
        max_diameter_um: Maximum outer diameter in microns (default: 1000)
        min_wall_thickness_um: Minimum wall thickness in microns (default: 2)
        max_aspect_ratio: Maximum major/minor axis ratio (default: 4.0)
            Higher values exclude longitudinal vessel sections
        min_circularity: Minimum circularity 0-1 (default: 0.3)
        min_ring_completeness: Minimum fraction of SMA+ perimeter (default: 0.5)
        canny_low: Low threshold for Canny (auto if None)
        canny_high: High threshold for Canny (auto if None)
        classify_vessel_types: Whether to auto-classify by diameter (default: False)
        extract_resnet_features: Whether to extract 2048D ResNet features (default: True)
        extract_sam2_embeddings: Whether to extract 256D SAM2 embeddings (default: True)
        resnet_batch_size: Batch size for ResNet feature extraction (default: 16)
    """

    def __init__(
        self,
        min_diameter_um: float = 10,
        max_diameter_um: float = 1000,
        min_wall_thickness_um: float = 2,
        max_aspect_ratio: float = 4.0,
        min_circularity: float = 0.3,
        min_ring_completeness: float = 0.5,
        canny_low: Optional[int] = None,
        canny_high: Optional[int] = None,
        classify_vessel_types: bool = False,
        extract_resnet_features: bool = True,
        extract_sam2_embeddings: bool = True,
        resnet_batch_size: int = 32,
    ):
        self.min_diameter_um = min_diameter_um
        self.max_diameter_um = max_diameter_um
        self.min_wall_thickness_um = min_wall_thickness_um
        self.max_aspect_ratio = max_aspect_ratio
        self.min_circularity = min_circularity
        self.min_ring_completeness = min_ring_completeness
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.classify_vessel_types = classify_vessel_types
        self.extract_resnet_features = extract_resnet_features
        self.extract_sam2_embeddings = extract_sam2_embeddings
        self.resnet_batch_size = resnet_batch_size

    @property
    def name(self) -> str:
        return "vessel"

    def segment(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float = 0.22,
        cd31_channel: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment ring structures using contour hierarchy.

        Uses Canny edge detection to find vessel edges, then contour hierarchy
        (RETR_CCOMP) to identify outer contours with inner contours (rings).

        Args:
            tile: RGB or grayscale image (SMA channel)
            models: Dict of models (not used for vessel detection)
            pixel_size_um: Pixel size in microns for filtering
            cd31_channel: Optional CD31 channel for validation

        Returns:
            List of ring candidate dicts with 'outer', 'inner', 'all_inner' contours
        """
        # Convert to grayscale
        if tile.ndim == 3:
            gray = np.mean(tile[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = tile.astype(np.float32)

        # Normalize to 0-255 for OpenCV
        gray_min, gray_max = gray.min(), gray.max()
        if gray_max - gray_min > 1e-8:
            gray_norm = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
        else:
            gray_norm = np.zeros_like(gray, dtype=np.uint8)

        # Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray_norm, (5, 5), 1.5)

        # Auto-calculate Canny thresholds using Otsu's method
        if self.canny_low is None or self.canny_high is None:
            otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny_low = int(otsu_thresh * 0.5)
            canny_high = int(otsu_thresh * 1.0)
        else:
            canny_low = self.canny_low
            canny_high = self.canny_high

        # Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)

        # Dilate edges slightly to close small gaps
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)

        # Fill detected edges to create binary regions
        binary = np.zeros_like(edges_dilated)

        # Find contours from edges
        edge_contours, _ = cv2.findContours(
            edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Fill closed contours
        for cnt in edge_contours:
            if cv2.contourArea(cnt) > 50:  # Skip tiny noise
                cv2.drawContours(binary, [cnt], 0, 255, -1)

        # Morphological cleanup
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # Find contours with hierarchy for ring detection
        # RETR_CCOMP gives 2-level hierarchy: outer contours and their direct holes
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or len(contours) == 0:
            return []

        hierarchy = hierarchy[0]  # Shape: (N, 4) where 4 = [next, prev, child, parent]

        # Find ring candidates: outer contours (parent=-1) that have children (holes)
        ring_candidates = []
        for i, (next_c, prev_c, child, parent) in enumerate(hierarchy):
            if parent == -1 and child != -1:  # Outer contour with at least one hole
                outer_contour = contours[i]

                # Collect all child contours (holes)
                inner_contours = []
                child_idx = child
                while child_idx != -1:
                    inner_contours.append(contours[child_idx])
                    child_idx = hierarchy[child_idx][0]  # Next sibling

                # Take the largest inner contour as the main lumen
                if inner_contours:
                    inner_contour = max(inner_contours, key=cv2.contourArea)
                    ring_candidates.append({
                        'outer': outer_contour,
                        'inner': inner_contour,
                        'all_inner': inner_contours,
                        'binary': binary,  # Store for ring completeness calculation
                    })

        return ring_candidates

    def extract_features(
        self,
        ring_candidate: Dict[str, Any],
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float = 0.22,
        cd31_channel: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract vessel-specific features from a ring candidate.

        Fits ellipses to outer and inner contours, measures wall thickness
        using distance transform and skeleton analysis.

        Args:
            ring_candidate: Dict with 'outer', 'inner', 'all_inner' contours
            tile: Original image for intensity measurements
            models: Dict of models (not used)
            pixel_size_um: Pixel size in microns
            cd31_channel: Optional CD31 channel for validation

        Returns:
            Dict of features, or None if candidate fails validation
        """
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize

        outer = ring_candidate['outer']
        inner = ring_candidate['inner']
        binary = ring_candidate.get('binary')

        # Need at least 5 points for ellipse fitting
        if len(outer) < 5 or len(inner) < 5:
            return None

        # Fit ellipses
        try:
            outer_ellipse = cv2.fitEllipse(outer)
            inner_ellipse = cv2.fitEllipse(inner)
        except cv2.error:
            return None

        # Extract ellipse parameters
        # fitEllipse returns: ((cx, cy), (minor_axis, major_axis), angle)
        (cx_out, cy_out), (minor_out, major_out), angle_out = outer_ellipse
        (cx_in, cy_in), (minor_in, major_in), angle_in = inner_ellipse

        # Calculate areas
        outer_area = cv2.contourArea(outer)
        inner_area = cv2.contourArea(inner)
        wall_area = outer_area - inner_area

        if wall_area <= 0 or inner_area <= 0:
            return None

        # Convert to diameters in microns
        outer_diameter_um = max(major_out, minor_out) * pixel_size_um
        inner_diameter_um = max(major_in, minor_in) * pixel_size_um

        # Size filtering
        if outer_diameter_um < self.min_diameter_um:
            return None
        if outer_diameter_um > self.max_diameter_um:
            return None

        # Aspect ratio filtering (exclude longitudinal sections)
        aspect_ratio_out = max(major_out, minor_out) / (min(major_out, minor_out) + 1e-8)
        if aspect_ratio_out > self.max_aspect_ratio:
            return None

        # Circularity filtering
        perimeter_out = cv2.arcLength(outer, True)
        circularity = 4 * np.pi * outer_area / (perimeter_out ** 2 + 1e-8)
        if circularity < self.min_circularity:
            return None

        # Create wall mask for measurements
        h, w = tile.shape[:2]
        wall_mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(wall_mask_temp, [outer], 0, 255, -1)
        cv2.drawContours(wall_mask_temp, [inner], 0, 0, -1)
        wall_region = wall_mask_temp > 0

        if wall_region.sum() == 0:
            return None

        # Calculate wall thickness using distance transform
        lumen_mask_temp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lumen_mask_temp, [inner], 0, 255, -1)

        # Distance from lumen boundary into wall
        dist_from_lumen = distance_transform_edt(~(lumen_mask_temp > 0))

        # Sample thickness at points along inner contour
        wall_thickness_values = []
        for pt in inner[::max(1, len(inner) // 36)]:  # Sample ~36 points
            px, py = pt[0]
            if 0 <= py < h and 0 <= px < w:
                if wall_region[py, px] or (lumen_mask_temp[py, px] > 0):
                    ray_dist = dist_from_lumen[py, px]
                    if ray_dist > 0:
                        wall_thickness_values.append(ray_dist * pixel_size_um)

        # Also measure using skeleton/medial axis approach
        try:
            skeleton = skeletonize(wall_region)
            skeleton_distances = dist_from_lumen[skeleton]
            if len(skeleton_distances) > 0:
                # Thickness is roughly 2x the distance to medial axis
                medial_thicknesses = skeleton_distances * 2 * pixel_size_um
                wall_thickness_values.extend(medial_thicknesses.tolist())
        except Exception as e:
            logger.debug(f"Skeleton analysis failed for wall thickness: {e}")

        if len(wall_thickness_values) < 5:
            return None

        wall_thicknesses = np.array(wall_thickness_values)
        wall_thickness_mean = float(np.mean(wall_thicknesses))
        wall_thickness_std = float(np.std(wall_thicknesses))
        wall_thickness_min = float(np.min(wall_thicknesses))
        wall_thickness_max = float(np.max(wall_thicknesses))
        wall_thickness_median = float(np.median(wall_thicknesses))

        # Wall thickness filtering
        if wall_thickness_mean < self.min_wall_thickness_um:
            return None

        # Calculate ring completeness (fraction of perimeter with SMA signal)
        ring_points = 0
        ring_positive = 0

        if binary is not None:
            for theta in np.linspace(0, 2 * np.pi, 72):
                # Mid-wall radius
                a_out, b_out = major_out / 2, minor_out / 2
                a_in, b_in = major_in / 2, minor_in / 2
                angle_out_rad = np.radians(angle_out)

                cos_t = np.cos(theta - angle_out_rad)
                sin_t = np.sin(theta - angle_out_rad)
                r_out = (a_out * b_out) / np.sqrt((b_out * cos_t) ** 2 + (a_out * sin_t) ** 2 + 1e-8)
                r_in = (a_in * b_in) / np.sqrt((b_in * cos_t) ** 2 + (a_in * sin_t) ** 2 + 1e-8)
                r_mid = (r_out + r_in) / 2

                # Point at mid-wall
                px = int(cx_out + r_mid * np.cos(theta))
                py = int(cy_out + r_mid * np.sin(theta))

                if 0 <= py < binary.shape[0] and 0 <= px < binary.shape[1]:
                    ring_points += 1
                    if binary[py, px] > 0:
                        ring_positive += 1

        ring_completeness = ring_positive / (ring_points + 1e-8)
        if ring_completeness < self.min_ring_completeness:
            return None

        # CD31 validation (if channel provided)
        cd31_validated = True
        cd31_score = 0.0
        if cd31_channel is not None:
            lumen_mask = np.zeros((h, w), dtype=np.uint8)
            wall_mask = np.zeros((h, w), dtype=np.uint8)

            cv2.drawContours(lumen_mask, [inner], 0, 255, -1)
            cv2.drawContours(wall_mask, [outer], 0, 255, -1)
            cv2.drawContours(wall_mask, [inner], 0, 0, -1)

            cd31_in_lumen = cd31_channel[lumen_mask > 0].mean() if (lumen_mask > 0).any() else 0
            cd31_in_wall = cd31_channel[wall_mask > 0].mean() if (wall_mask > 0).any() else 0

            # CD31 should be at lumen boundary, not in wall
            cd31_score = float(cd31_in_lumen / (cd31_in_wall + 1e-8))
            cd31_validated = cd31_in_lumen > cd31_in_wall * 0.8  # Some tolerance

        # Auto-classify vessel type by size
        vessel_type = 'unknown'
        if self.classify_vessel_types:
            if outer_diameter_um < 10:
                vessel_type = 'capillary'
            elif outer_diameter_um < 100:
                vessel_type = 'arteriole'
            else:
                vessel_type = 'artery'

        # Determine confidence level
        if ring_completeness > 0.8 and circularity > 0.6 and aspect_ratio_out < 2.0:
            confidence = 'high'
        elif ring_completeness > 0.6 and circularity > 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            # Diameters
            'outer_diameter_um': float(outer_diameter_um),
            'inner_diameter_um': float(inner_diameter_um),
            'major_axis_um': float(max(major_out, minor_out) * pixel_size_um),
            'minor_axis_um': float(min(major_out, minor_out) * pixel_size_um),
            # Wall thickness measurements
            'wall_thickness_mean_um': wall_thickness_mean,
            'wall_thickness_median_um': wall_thickness_median,
            'wall_thickness_std_um': wall_thickness_std,
            'wall_thickness_min_um': wall_thickness_min,
            'wall_thickness_max_um': wall_thickness_max,
            # Areas
            'lumen_area_um2': float(inner_area * pixel_size_um ** 2),
            'wall_area_um2': float(wall_area * pixel_size_um ** 2),
            'outer_area_um2': float(outer_area * pixel_size_um ** 2),
            # Shape metrics
            'orientation_deg': float(angle_out),
            'aspect_ratio': float(aspect_ratio_out),
            'circularity': float(circularity),
            'ring_completeness': float(ring_completeness),
            # Validation
            'cd31_validated': cd31_validated,
            'cd31_score': cd31_score,
            # Classification
            'vessel_type': vessel_type,
            'confidence': confidence,
            # Centers
            'outer_center': [float(cx_out), float(cy_out)],
            'inner_center': [float(cx_in), float(cy_in)],
        }

    def filter(
        self,
        masks: List[np.ndarray],
        features: List[Dict[str, Any]],
        pixel_size_um: float,
    ) -> List[Detection]:
        """
        Filter candidates based on extracted features.

        Note: For VesselStrategy, most filtering happens during extract_features().
        This method creates Detection objects from valid candidates.

        Args:
            masks: List of wall masks
            features: List of feature dicts from extract_features()
            pixel_size_um: Pixel size in microns

        Returns:
            List of Detection objects
        """
        detections = []

        for i, (mask, feat) in enumerate(zip(masks, features)):
            if feat is None:
                continue

            # Get centroid from outer center
            center = feat.get('outer_center', [0, 0])

            det = Detection(
                mask=mask,
                centroid=center,
                features=feat,
                id=f"vessel_{i + 1}",
                score=1.0 if feat.get('confidence') == 'high' else (
                    0.7 if feat.get('confidence') == 'medium' else 0.4
                ),
            )
            detections.append(det)

        return detections

    def detect(
        self,
        tile: np.ndarray,
        models: Dict[str, Any],
        pixel_size_um: float = 0.22,
        cd31_channel: Optional[np.ndarray] = None,
        extract_full_features: bool = True,
    ) -> Tuple[np.ndarray, List[Detection]]:
        """
        Complete vessel detection pipeline with full 2326 feature extraction.

        Pipeline:
        1. Segment ring candidates using contour hierarchy analysis
        2. Extract vessel-specific features (wall thickness, diameters, etc.)
        3. Extract full features (22 morphological + 256 SAM2 + 2048 ResNet)
        4. Filter by size and create Detection objects

        Args:
            tile: RGB or grayscale image (SMA channel)
            models: Dict with optional keys:
                - 'sam2_predictor': SAM2ImagePredictor (for embeddings)
                - 'resnet': ResNet model (for features)
                - 'resnet_transform': torchvision transform
                - 'device': torch device
            pixel_size_um: Pixel size in microns
            cd31_channel: Optional CD31 channel for validation
            extract_full_features: Whether to extract all 2326 features (default True)

        Returns:
            Tuple of (combined mask array, list of Detection objects with 2326 features)
        """
        import torch

        # Get ring candidates
        ring_candidates = self.segment(tile, models, pixel_size_um, cd31_channel)

        if not ring_candidates:
            return np.zeros(tile.shape[:2], dtype=np.uint32), []

        h, w = tile.shape[:2]

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

        # Get models
        sam2_predictor = models.get('sam2_predictor')
        resnet = models.get('resnet')
        resnet_transform = models.get('resnet_transform')
        device = models.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Set image for SAM2 embeddings if available
        if sam2_predictor is not None and self.extract_sam2_embeddings and extract_full_features:
            sam2_predictor.set_image(tile_rgb)

        # Extract features and create masks for valid candidates
        valid_candidates = []
        crops_for_resnet = []
        crop_indices = []

        for cand_idx, cand in enumerate(ring_candidates):
            # Extract vessel-specific features
            vessel_feat = self.extract_features(
                cand, tile, models, pixel_size_um, cd31_channel
            )

            if vessel_feat is None:
                continue

            # Create wall mask
            temp = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp, [cand['outer']], 0, 1, -1)
            cv2.drawContours(temp, [cand['inner']], 0, 0, -1)
            wall_mask = temp.astype(bool)

            if wall_mask.sum() == 0:
                continue

            # Extract 22 morphological features
            morph_feat = extract_morphological_features(wall_mask, tile_rgb)
            if not morph_feat:
                continue

            # Merge vessel-specific and morphological features
            all_features = {**morph_feat, **vessel_feat}

            # Get centroid
            center = vessel_feat.get('outer_center', [0, 0])
            cx, cy = center[0], center[1]

            # Extract SAM2 embeddings (256D)
            if sam2_predictor is not None and self.extract_sam2_embeddings and extract_full_features:
                sam2_emb = self._extract_sam2_embedding(sam2_predictor, cy, cx)
                for i, v in enumerate(sam2_emb):
                    all_features[f'sam2_emb_{i}'] = float(v)
            elif extract_full_features:
                # Fill with zeros if SAM2 not available
                for i in range(256):
                    all_features[f'sam2_emb_{i}'] = 0.0

            # Prepare crop for batch ResNet processing
            if self.extract_resnet_features and extract_full_features:
                ys, xs = np.where(wall_mask)
                if len(ys) > 0:
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()
                    crop = tile_rgb[y1:y2+1, x1:x2+1].copy()
                    crop_mask = wall_mask[y1:y2+1, x1:x2+1]
                    crop[~crop_mask] = 0  # Zero out background
                    crops_for_resnet.append(crop)
                    crop_indices.append(len(valid_candidates))

            valid_candidates.append({
                'mask': wall_mask,
                'features': all_features,
                'centroid': center,
                'outer_contour': cand['outer'].tolist(),
                'inner_contour': cand['inner'].tolist(),
            })

        # Batch ResNet feature extraction
        if crops_for_resnet and resnet is not None and resnet_transform is not None and extract_full_features:
            resnet_features_list = self._extract_resnet_features_batch(
                crops_for_resnet, resnet, resnet_transform, device
            )

            # Assign ResNet features to correct candidates
            for crop_idx, resnet_feats in zip(crop_indices, resnet_features_list):
                for i, v in enumerate(resnet_feats):
                    valid_candidates[crop_idx]['features'][f'resnet_{i}'] = float(v)

        # Fill zeros for candidates without ResNet features
        if extract_full_features:
            for cand in valid_candidates:
                if 'resnet_0' not in cand['features']:
                    for i in range(2048):
                        cand['features'][f'resnet_{i}'] = 0.0

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

        # Create Detection objects
        masks_list = [cand['mask'] for cand in valid_candidates]
        features_list = [cand['features'] for cand in valid_candidates]

        detections = self.filter(masks_list, features_list, pixel_size_um)

        # Add contour data to detections
        for i, det in enumerate(detections):
            if i < len(valid_candidates):
                det.features['outer_contour'] = valid_candidates[i]['outer_contour']
                det.features['inner_contour'] = valid_candidates[i]['inner_contour']

        # Build combined mask with overlap checking
        combined_mask = np.zeros((h, w), dtype=np.uint32)
        final_detections = []
        det_id = 1

        for det in detections:
            mask = det.mask

            # Check overlap with existing detections
            if combined_mask.max() > 0:
                overlap = (mask & (combined_mask > 0)).sum()
                if overlap > 0.5 * mask.sum():
                    continue

            combined_mask[mask] = det_id
            det.id = f"vessel_{det_id}"
            final_detections.append(det)
            det_id += 1

        return combined_mask, final_detections

    # _extract_sam2_embedding inherited from DetectionStrategy base class
    # _extract_resnet_features_batch inherited from DetectionStrategy base class

    def create_vessel_mask(
        self,
        outer_contour: np.ndarray,
        inner_contour: np.ndarray,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a wall mask from outer and inner contours.

        Args:
            outer_contour: Outer boundary contour
            inner_contour: Inner (lumen) boundary contour
            shape: (height, width) of output mask

        Returns:
            Boolean mask of vessel wall
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, [outer_contour], 0, 255, -1)
        cv2.drawContours(mask, [inner_contour], 0, 0, -1)
        return mask > 0
