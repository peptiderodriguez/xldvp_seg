#!/usr/bin/env python3
"""
Unified Cell Segmentation Pipeline

A general-purpose pipeline for detecting and classifying cells in CZI microscopy images.
Supports multiple cell types with shared infrastructure and full feature extraction.

Cell Types:
    - nmj: Neuromuscular junctions (intensity + elongation filter)
    - mk: Megakaryocytes (SAM2 automatic mask generation)
    - hspc: Hematopoietic stem/progenitor cells (Cellpose-SAM + SAM2 refinement)
    - vessel: Blood vessel cross-sections (ring detection via contour hierarchy)

Features extracted per cell: 2326 total
    - 22 morphological/intensity features
    - 256 SAM2 embedding features
    - 2048 ResNet-50 deep features
    + Cell-type specific features (elongation for NMJ, wall thickness for vessel, etc.)

Outputs:
    - {cell_type}_detections.json: All detections with universal IDs and global coordinates
    - {cell_type}_coordinates.csv: Quick export with center coordinates in pixels and µm
    - {cell_type}_masks.h5: Per-tile mask arrays
    - html/: Interactive HTML viewer for annotation

Usage:
    # NMJ detection
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj --channel 1

    # Vessel detection (SMA staining)
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --min-vessel-diameter 10 --max-vessel-diameter 500

    # Vessel with CD31 validation
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --cd31-channel 1
"""

import os
import gc
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from PIL import Image

# Import shared modules
from shared.tissue_detection import (
    calibrate_tissue_threshold,
    filter_tissue_tiles,
)
from shared.html_export import (
    export_samples_to_html,
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)

# Try LZ4 compression for HDF5
try:
    import hdf5plugin
    HDF5_COMPRESSION = hdf5plugin.LZ4(nbytes=0)
except ImportError:
    HDF5_COMPRESSION = {'compression': 'gzip'}


def create_hdf5_dataset(f, name, data):
    """Create HDF5 dataset with compression."""
    if isinstance(HDF5_COMPRESSION, dict):
        f.create_dataset(name, data=data, **HDF5_COMPRESSION)
    else:
        f.create_dataset(name, data=data, **HDF5_COMPRESSION)


# =============================================================================
# FEATURE EXTRACTION (shared across all cell types)
# =============================================================================

def extract_morphological_features(mask, image):
    """
    Extract 22 morphological/intensity features from a mask.

    Returns dict with: area, perimeter, circularity, solidity, aspect_ratio,
    extent, equiv_diameter, color stats (RGB, gray, HSV), texture features.
    """
    from skimage import measure

    if not mask.any():
        return {}

    area = int(mask.sum())
    props = measure.regionprops(mask.astype(int))[0]

    perimeter = props.perimeter if hasattr(props, 'perimeter') else 0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    solidity = props.solidity if hasattr(props, 'solidity') else 0
    aspect_ratio = props.major_axis_length / props.minor_axis_length if props.minor_axis_length > 0 else 1
    extent = props.extent if hasattr(props, 'extent') else 0
    equiv_diameter = props.equivalent_diameter if hasattr(props, 'equivalent_diameter') else 0

    # Intensity features
    if image.ndim == 3:
        masked_pixels = image[mask]
        red_mean, red_std = float(np.mean(masked_pixels[:, 0])), float(np.std(masked_pixels[:, 0]))
        green_mean, green_std = float(np.mean(masked_pixels[:, 1])), float(np.std(masked_pixels[:, 1]))
        blue_mean, blue_std = float(np.mean(masked_pixels[:, 2])), float(np.std(masked_pixels[:, 2]))
        gray = np.mean(masked_pixels, axis=1)
    else:
        gray = image[mask].astype(float)
        red_mean = green_mean = blue_mean = float(np.mean(gray))
        red_std = green_std = blue_std = float(np.std(gray))

    gray_mean, gray_std = float(np.mean(gray)), float(np.std(gray))

    # HSV features
    if image.ndim == 3:
        import colorsys
        hsv = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in masked_pixels[:100]])  # Sample for speed
        hue_mean = float(np.mean(hsv[:, 0]) * 180)
        sat_mean = float(np.mean(hsv[:, 1]) * 255)
        val_mean = float(np.mean(hsv[:, 2]) * 255)
    else:
        hue_mean = sat_mean = 0.0
        val_mean = gray_mean

    # Texture features
    relative_brightness = gray_mean - np.mean(image) if image.size > 0 else 0
    intensity_variance = float(np.var(gray))
    dark_fraction = float(np.mean(gray < 100))
    nuclear_complexity = gray_std

    return {
        'area': area,
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'solidity': float(solidity),
        'aspect_ratio': float(aspect_ratio),
        'extent': float(extent),
        'equiv_diameter': float(equiv_diameter),
        'red_mean': red_mean, 'red_std': red_std,
        'green_mean': green_mean, 'green_std': green_std,
        'blue_mean': blue_mean, 'blue_std': blue_std,
        'gray_mean': gray_mean, 'gray_std': gray_std,
        'hue_mean': hue_mean, 'saturation_mean': sat_mean, 'value_mean': val_mean,
        'relative_brightness': float(relative_brightness),
        'intensity_variance': intensity_variance,
        'dark_region_fraction': dark_fraction,
        'nuclear_complexity': nuclear_complexity,
    }


# =============================================================================
# UNIFIED SEGMENTER CLASS
# =============================================================================

class UnifiedSegmenter:
    """
    Unified segmenter for all cell types (MK, HSPC, NMJ, Vessel).

    Loads models once and provides cell-type specific detection with
    consistent feature extraction (2326 features per cell + type-specific).

    Detection methods:
        - NMJ: Intensity threshold + skeleton elongation filter
        - MK: SAM2 automatic mask generation + size filter
        - HSPC: Cellpose-SAM nuclei detection + SAM2 refinement
        - Vessel: Contour hierarchy for ring detection + ellipse fitting
    """

    def __init__(self, device=None, load_sam2=True, load_cellpose=True):
        """
        Initialize segmenter with required models.

        Args:
            device: torch device (auto-detect if None)
            load_sam2: Whether to load SAM2 (needed for MK, HSPC, and full features)
            load_cellpose: Whether to load Cellpose-SAM (needed for HSPC)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        print(f"Initializing UnifiedSegmenter on {self.device}")

        # ResNet for deep features (always loaded - 2048D features)
        print("  Loading ResNet-50...")
        resnet = tv_models.resnet50(weights='DEFAULT')
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.resnet.eval().to(self.device)
        self.resnet_transform = tv_transforms.Compose([
            tv_transforms.Resize(224),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # SAM2 for mask generation and embeddings
        self.sam2_auto = None
        self.sam2_predictor = None
        if load_sam2:
            self._load_sam2()

        # Cellpose-SAM for HSPC detection
        self.cellpose = None
        if load_cellpose:
            self._load_cellpose()

    def _load_sam2(self):
        """Load SAM2 models."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        # Find checkpoint
        script_dir = Path(__file__).parent.resolve()
        checkpoint_candidates = [
            script_dir / "checkpoints" / "sam2.1_hiera_large.pt",
            Path("/ptmp/edrod/MKsegmentation/checkpoints/sam2.1_hiera_large.pt"),
        ]
        checkpoint_path = None
        for cp in checkpoint_candidates:
            if cp.exists():
                checkpoint_path = cp
                break

        if checkpoint_path is None:
            print("  WARNING: SAM2 checkpoint not found, skipping SAM2")
            return

        print(f"  Loading SAM2 from {checkpoint_path}...")
        sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(sam2_config, str(checkpoint_path), device=self.device)

        # Auto mask generator for MK detection
        self.sam2_auto = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=24,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.4,
            min_mask_region_area=500,
            crop_n_layers=1
        )

        # Predictor for point prompts (HSPC) and embeddings
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def _load_cellpose(self):
        """Load Cellpose-SAM model."""
        from cellpose.models import CellposeModel

        print(f"  Loading Cellpose-SAM...")
        self.cellpose = CellposeModel(pretrained_model='cpsam', gpu=True, device=self.device)

    def extract_resnet_features(self, crop):
        """Extract 2048D ResNet features from crop."""
        if crop.size == 0:
            return np.zeros(2048)

        pil_img = Image.fromarray(crop)
        tensor = self.resnet_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.resnet(tensor).cpu().numpy().flatten()
        return features

    def extract_sam2_embedding(self, cy, cx):
        """Extract 256D SAM2 embedding at a point."""
        if self.sam2_predictor is None:
            return np.zeros(256)

        try:
            shape = self.sam2_predictor._features["image_embed"].shape
            emb_h, emb_w = shape[2], shape[3]
            img_h, img_w = self.sam2_predictor._orig_hw

            emb_y = int(cy / img_h * emb_h)
            emb_x = int(cx / img_w * emb_w)
            emb_y = min(max(emb_y, 0), emb_h - 1)
            emb_x = min(max(emb_x, 0), emb_w - 1)

            return self.sam2_predictor._features["image_embed"][0, :, emb_y, emb_x].cpu().numpy()
        except:
            return np.zeros(256)

    def extract_full_features(self, mask, image_rgb, cy, cx):
        """
        Extract all 2326 features for a detection.

        Args:
            mask: Binary mask
            image_rgb: RGB image
            cy, cx: Centroid coordinates

        Returns:
            Dict with all features
        """
        # 22 morphological features
        features = extract_morphological_features(mask, image_rgb)

        # 256 SAM2 embedding features
        sam2_emb = self.extract_sam2_embedding(cy, cx)
        for i, v in enumerate(sam2_emb):
            features[f'sam2_emb_{i}'] = float(v)

        # 2048 ResNet features from masked crop
        ys, xs = np.where(mask)
        if len(ys) > 0:
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop = image_rgb[y1:y2+1, x1:x2+1].copy()
            crop_mask = mask[y1:y2+1, x1:x2+1]
            crop[~crop_mask] = 0

            resnet_feats = self.extract_resnet_features(crop)
            for i, v in enumerate(resnet_feats):
                features[f'resnet_{i}'] = float(v)
        else:
            for i in range(2048):
                features[f'resnet_{i}'] = 0.0

        return features

    def detect_nmj(self, image_rgb, params):
        """
        Detect NMJs using intensity threshold + elongation filter.

        Args:
            image_rgb: RGB image array
            params: Dict with intensity_percentile, min_area, min_skeleton_length, min_elongation

        Returns:
            Tuple of (masks, features_list)
        """
        from skimage.morphology import skeletonize, remove_small_objects, binary_opening, binary_closing, disk
        from skimage.measure import label, regionprops
        from scipy import ndimage

        # Convert to grayscale
        if image_rgb.ndim == 3:
            gray = np.mean(image_rgb[:, :, :3], axis=2)
        else:
            gray = image_rgb.astype(float)

        # Threshold bright regions
        threshold = np.percentile(gray, params['intensity_percentile'])
        bright_mask = gray > threshold

        # Morphological cleanup
        bright_mask = binary_opening(bright_mask, disk(1))
        bright_mask = binary_closing(bright_mask, disk(2))
        bright_mask = remove_small_objects(bright_mask, min_size=params['min_area'])

        # Label connected components
        labeled = label(bright_mask)
        props = regionprops(labeled, intensity_image=gray)

        # Filter by elongation
        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        features_list = []
        det_id = 1

        # Set image for SAM2 embeddings if available
        if self.sam2_predictor is not None:
            self.sam2_predictor.set_image(image_rgb)

        for prop in props:
            if prop.area < params['min_area']:
                continue

            region_mask = labeled == prop.label
            skeleton = skeletonize(region_mask)
            skeleton_length = skeleton.sum()
            elongation = skeleton_length / np.sqrt(prop.area) if prop.area > 0 else 0

            if skeleton_length >= params['min_skeleton_length'] and elongation >= params['min_elongation']:
                masks[region_mask] = det_id

                cy, cx = prop.centroid

                # Extract full 2326 features
                features = self.extract_full_features(region_mask, image_rgb, cy, cx)
                features['skeleton_length'] = int(skeleton_length)
                features['elongation'] = float(elongation)
                features['eccentricity'] = float(prop.eccentricity)
                features['mean_intensity'] = float(prop.mean_intensity)

                features_list.append({
                    'id': f'nmj_{det_id}',
                    'center': [float(cx), float(cy)],
                    'features': features
                })

                det_id += 1

        # Clear SAM2 cache
        if self.sam2_predictor is not None:
            self.sam2_predictor.reset_predictor()

        return masks, features_list

    def detect_mk(self, image_rgb, params):
        """
        Detect MKs using SAM2 automatic mask generation.

        Args:
            image_rgb: RGB image array
            params: Dict with mk_min_area, mk_max_area

        Returns:
            Tuple of (masks, features_list)
        """
        from scipy import ndimage

        if self.sam2_auto is None:
            raise RuntimeError("SAM2 not loaded - required for MK detection")

        # Generate masks
        sam2_results = self.sam2_auto.generate(image_rgb)

        # Filter by size
        valid_results = []
        for result in sam2_results:
            area = result['segmentation'].sum()
            if params['mk_min_area'] <= area <= params['mk_max_area']:
                result['area'] = area
                valid_results.append(result)
        valid_results.sort(key=lambda x: x['area'], reverse=True)

        del sam2_results
        gc.collect()

        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        features_list = []
        det_id = 1

        # Set image for embeddings
        self.sam2_predictor.set_image(image_rgb)

        for result in valid_results:
            mask = result['segmentation']
            if mask.dtype != bool:
                mask = (mask > 0.5).astype(bool)

            # Check overlap
            if masks.max() > 0:
                overlap = ((mask > 0) & (masks > 0)).sum()
                if overlap > 0.5 * mask.sum():
                    continue

            masks[mask] = det_id
            cy, cx = ndimage.center_of_mass(mask)

            # Extract full features
            features = self.extract_full_features(mask, image_rgb, cy, cx)
            features['sam2_iou'] = float(result.get('predicted_iou', 0))
            features['sam2_stability'] = float(result.get('stability_score', 0))

            features_list.append({
                'id': f'mk_{det_id}',
                'center': [float(cx), float(cy)],
                'features': features
            })

            det_id += 1

        del valid_results
        gc.collect()
        torch.cuda.empty_cache()

        self.sam2_predictor.reset_predictor()

        return masks, features_list

    def detect_hspc(self, image_rgb, params):
        """
        Detect HSPCs using Cellpose-SAM + SAM2 refinement.

        Args:
            image_rgb: RGB image array
            params: Dict (currently unused, Cellpose auto-detects)

        Returns:
            Tuple of (masks, features_list)
        """
        from scipy import ndimage

        if self.cellpose is None:
            raise RuntimeError("Cellpose not loaded - required for HSPC detection")
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 not loaded - required for HSPC detection")

        # Cellpose detection
        cellpose_masks, _, _ = self.cellpose.eval(image_rgb, channels=[0, 0])

        # Get centroids
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        # Limit candidates
        MAX_CANDIDATES = 500
        if len(cellpose_ids) > MAX_CANDIDATES:
            areas = [(cp_id, (cellpose_masks == cp_id).sum()) for cp_id in cellpose_ids]
            areas.sort(key=lambda x: x[1], reverse=True)
            cellpose_ids = np.array([a[0] for a in areas[:MAX_CANDIDATES]])

        # Set image for SAM2
        self.sam2_predictor.set_image(image_rgb)

        # Collect candidates with SAM2 refinement
        candidates = []
        for cp_id in cellpose_ids:
            cp_mask = cellpose_masks == cp_id
            cy, cx = ndimage.center_of_mass(cp_mask)

            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])

            masks_pred, scores, _ = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            best_idx = np.argmax(scores)
            sam2_mask = masks_pred[best_idx]
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)

            if sam2_mask.sum() < 10:
                continue

            candidates.append({
                'mask': sam2_mask,
                'score': float(scores[best_idx]),
                'center': (cx, cy),
                'cp_id': cp_id
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)

        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        features_list = []
        det_id = 1

        for cand in candidates:
            sam2_mask = cand['mask']
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)

            # Check overlap
            if masks.max() > 0:
                overlap = ((sam2_mask > 0) & (masks > 0)).sum()
                if overlap > 0.5 * sam2_mask.sum():
                    continue

            masks[sam2_mask] = det_id
            cx, cy = cand['center']

            # Extract full features
            features = self.extract_full_features(sam2_mask, image_rgb, cy, cx)
            features['sam2_score'] = cand['score']
            features['cellpose_id'] = int(cand['cp_id'])

            features_list.append({
                'id': f'hspc_{det_id}',
                'center': [float(cx), float(cy)],
                'features': features
            })

            det_id += 1

        del candidates, cellpose_masks
        gc.collect()

        self.sam2_predictor.reset_predictor()
        torch.cuda.empty_cache()

        return masks, features_list

    def detect_vessel(self, image_rgb, params, cd31_channel=None):
        """
        Detect vessel cross-sections (ring structures) using Canny edge detection.

        Vessels appear as ring-like structures in SMA staining - an outer contour
        (adventitial side) with an inner contour (lumen). Uses Canny edge detection
        to find ring edges, then contour hierarchy to pair outer/inner boundaries.

        Args:
            image_rgb: RGB image array (SMA channel as grayscale repeated 3x, or actual RGB)
            params: Dict with:
                - min_vessel_diameter_um: minimum outer diameter
                - max_vessel_diameter_um: maximum outer diameter
                - min_wall_thickness_um: minimum wall thickness
                - max_aspect_ratio: maximum major/minor axis ratio (exclude longitudinal)
                - min_circularity: minimum circularity (0-1)
                - min_ring_completeness: minimum fraction of ring that must be SMA+
                - pixel_size_um: for converting pixels to microns
                - classify_vessel_types: whether to auto-classify by size
                - canny_low: Canny low threshold (default: auto)
                - canny_high: Canny high threshold (default: auto)
            cd31_channel: Optional CD31 channel for vessel validation

        Returns:
            Tuple of (masks, features_list)
        """
        import cv2
        from scipy import ndimage
        from scipy.ndimage import distance_transform_edt

        pixel_size = params.get('pixel_size_um', 0.22)

        # Convert size parameters from µm to pixels
        min_diameter_px = params.get('min_vessel_diameter_um', 10) / pixel_size
        max_diameter_px = params.get('max_vessel_diameter_um', 1000) / pixel_size
        min_wall_px = params.get('min_wall_thickness_um', 2) / pixel_size

        # Convert to grayscale
        if image_rgb.ndim == 3:
            gray = np.mean(image_rgb[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = image_rgb.astype(np.float32)

        # Normalize to 0-255 for OpenCV
        gray_norm = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)

        # Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray_norm, (5, 5), 1.5)

        # Auto-calculate Canny thresholds using Otsu's method
        if params.get('canny_low') is None or params.get('canny_high') is None:
            otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny_low = int(otsu_thresh * 0.5)
            canny_high = int(otsu_thresh * 1.0)
        else:
            canny_low = params.get('canny_low')
            canny_high = params.get('canny_high')

        # Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)

        # Dilate edges slightly to close small gaps, then find contours
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=1)

        # Fill the detected edges to create binary regions
        # Use flood fill from edges to create filled regions
        binary = np.zeros_like(edges_dilated)

        # Find contours from edges
        edge_contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None or len(contours) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint32), []

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
                        'all_inner': inner_contours
                    })

        # Process ring candidates
        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        features_list = []
        det_id = 1

        # Set image for SAM2 embeddings if available
        if self.sam2_predictor is not None:
            self.sam2_predictor.set_image(image_rgb)

        for cand in ring_candidates:
            outer = cand['outer']
            inner = cand['inner']

            # Need at least 5 points for ellipse fitting
            if len(outer) < 5 or len(inner) < 5:
                continue

            # Fit ellipses
            try:
                outer_ellipse = cv2.fitEllipse(outer)
                inner_ellipse = cv2.fitEllipse(inner)
            except cv2.error:
                continue

            # Extract ellipse parameters
            # fitEllipse returns: ((cx, cy), (minor_axis, major_axis), angle)
            (cx_out, cy_out), (minor_out, major_out), angle_out = outer_ellipse
            (cx_in, cy_in), (minor_in, major_in), angle_in = inner_ellipse

            # Calculate areas
            outer_area = cv2.contourArea(outer)
            inner_area = cv2.contourArea(inner)
            wall_area = outer_area - inner_area

            if wall_area <= 0 or inner_area <= 0:
                continue

            # Convert to diameters in µm
            outer_diameter_um = max(major_out, minor_out) * pixel_size
            inner_diameter_um = max(major_in, minor_in) * pixel_size

            # Size filtering
            if outer_diameter_um < params.get('min_vessel_diameter_um', 10):
                continue
            if outer_diameter_um > params.get('max_vessel_diameter_um', 1000):
                continue

            # Aspect ratio filtering (exclude longitudinal sections)
            aspect_ratio_out = max(major_out, minor_out) / (min(major_out, minor_out) + 1e-8)
            if aspect_ratio_out > params.get('max_aspect_ratio', 4.0):
                continue

            # Circularity filtering
            perimeter_out = cv2.arcLength(outer, True)
            circularity = 4 * np.pi * outer_area / (perimeter_out ** 2 + 1e-8)
            if circularity < params.get('min_circularity', 0.3):
                continue

            # Calculate wall thickness using distance transform (more accurate for irregular shapes)
            # Create wall mask
            wall_mask_temp = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(wall_mask_temp, [outer], 0, 255, -1)
            cv2.drawContours(wall_mask_temp, [inner], 0, 0, -1)
            wall_region = wall_mask_temp > 0

            if wall_region.sum() == 0:
                continue

            # Distance transform from inner boundary (lumen edge)
            lumen_mask_temp = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(lumen_mask_temp, [inner], 0, 255, -1)

            # Distance from lumen boundary into wall
            dist_from_lumen = distance_transform_edt(~(lumen_mask_temp > 0))

            # Distance from outer boundary into wall
            outer_mask_temp = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(outer_mask_temp, [outer], 0, 255, -1)
            dist_from_outer = distance_transform_edt(outer_mask_temp > 0)

            # Wall thickness at each wall pixel = distance to lumen + distance to outer
            # But we want local thickness, so sample along the medial axis
            wall_thickness_values = []

            # Sample thickness at points along inner contour
            for pt in inner[::max(1, len(inner)//36)]:  # Sample ~36 points
                px, py = pt[0]
                if 0 <= py < wall_region.shape[0] and 0 <= px < wall_region.shape[1]:
                    # Find thickness by ray casting outward
                    # Use the distance transform value at the medial axis
                    if wall_region[py, px] or (lumen_mask_temp[py, px] > 0):
                        # Cast ray outward to find wall thickness
                        ray_dist = dist_from_lumen[py, px]
                        if ray_dist > 0:
                            wall_thickness_values.append(ray_dist * pixel_size)

            # Also measure using the skeleton/medial axis approach
            from skimage.morphology import skeletonize
            skeleton = skeletonize(wall_region)
            skeleton_distances = dist_from_lumen[skeleton]
            if len(skeleton_distances) > 0:
                # Thickness is roughly 2x the distance to medial axis
                medial_thicknesses = skeleton_distances * 2 * pixel_size
                wall_thickness_values.extend(medial_thicknesses.tolist())

            if len(wall_thickness_values) < 5:
                continue

            wall_thicknesses = np.array(wall_thickness_values)
            wall_thickness_mean = float(np.mean(wall_thicknesses))
            wall_thickness_std = float(np.std(wall_thicknesses))
            wall_thickness_min = float(np.min(wall_thicknesses))
            wall_thickness_max = float(np.max(wall_thicknesses))
            wall_thickness_median = float(np.median(wall_thicknesses))

            # Wall thickness filtering
            if wall_thickness_mean < params.get('min_wall_thickness_um', 2):
                continue

            # Calculate ring completeness (fraction of perimeter with SMA signal)
            # Sample points along the expected ring and check if they're in the mask
            ring_points = 0
            ring_positive = 0
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
            if ring_completeness < params.get('min_ring_completeness', 0.5):
                continue

            # CD31 validation (if channel provided)
            cd31_validated = True
            cd31_score = 0.0
            if cd31_channel is not None:
                # Create masks for lumen and wall
                lumen_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                wall_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

                cv2.drawContours(lumen_mask, [inner], 0, 255, -1)
                cv2.drawContours(wall_mask, [outer], 0, 255, -1)
                cv2.drawContours(wall_mask, [inner], 0, 0, -1)

                cd31_in_lumen = cd31_channel[lumen_mask > 0].mean() if (lumen_mask > 0).any() else 0
                cd31_in_wall = cd31_channel[wall_mask > 0].mean() if (wall_mask > 0).any() else 0

                # CD31 should be at lumen boundary, not in wall
                cd31_score = cd31_in_lumen / (cd31_in_wall + 1e-8)
                cd31_validated = cd31_in_lumen > cd31_in_wall * 0.8  # Some tolerance

            # Create mask for this vessel (wall region only)
            vessel_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.drawContours(vessel_mask, [outer], 0, 255, -1)
            cv2.drawContours(vessel_mask, [inner], 0, 0, -1)
            vessel_mask_bool = vessel_mask > 0

            # Check overlap with existing detections
            if masks.max() > 0:
                overlap = (vessel_mask_bool & (masks > 0)).sum()
                if overlap > 0.5 * vessel_mask_bool.sum():
                    continue

            masks[vessel_mask_bool] = det_id

            # Extract full features
            cy, cx = cy_out, cx_out
            features = self.extract_full_features(vessel_mask_bool, image_rgb, cy, cx)

            # Add vessel-specific features
            features['outer_diameter_um'] = float(outer_diameter_um)
            features['inner_diameter_um'] = float(inner_diameter_um)
            features['major_axis_um'] = float(max(major_out, minor_out) * pixel_size)
            features['minor_axis_um'] = float(min(major_out, minor_out) * pixel_size)
            features['wall_thickness_mean_um'] = float(wall_thickness_mean)
            features['wall_thickness_median_um'] = float(wall_thickness_median)
            features['wall_thickness_std_um'] = float(wall_thickness_std)
            features['wall_thickness_min_um'] = float(wall_thickness_min)
            features['wall_thickness_max_um'] = float(wall_thickness_max)
            features['lumen_area_um2'] = float(inner_area * pixel_size ** 2)
            features['wall_area_um2'] = float(wall_area * pixel_size ** 2)
            features['orientation_deg'] = float(angle_out)
            features['aspect_ratio'] = float(aspect_ratio_out)
            features['circularity'] = float(circularity)
            features['ring_completeness'] = float(ring_completeness)
            features['cd31_validated'] = cd31_validated
            features['cd31_score'] = float(cd31_score)

            # Auto-classify vessel type by size (if enabled)
            vessel_type = 'unknown'
            if params.get('classify_vessel_types', False):
                if outer_diameter_um < 10:
                    vessel_type = 'capillary'
                elif outer_diameter_um < 100:
                    vessel_type = 'arteriole'
                else:
                    vessel_type = 'artery'
            features['vessel_type'] = vessel_type

            # Determine confidence level
            if ring_completeness > 0.8 and circularity > 0.6 and aspect_ratio_out < 2.0:
                confidence = 'high'
            elif ring_completeness > 0.6 and circularity > 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'
            features['confidence'] = confidence

            features_list.append({
                'id': f'vessel_{det_id}',
                'center': [float(cx), float(cy)],
                'outer_contour': outer.tolist(),
                'inner_contour': inner.tolist(),
                'features': features
            })

            det_id += 1

        # Clear SAM2 cache
        if self.sam2_predictor is not None:
            self.sam2_predictor.reset_predictor()

        return masks, features_list

    def detect_mesothelium(self, image_rgb, params):
        """
        Detect mesothelial ribbon structures and divide into ~1500 µm² chunks.

        Uses ridge detection (Meijering filter) to find thin ribbon structures,
        then extracts skeleton, walks along paths, and chunks by area.

        Args:
            image_rgb: RGB image array (mesothelin channel as grayscale repeated 3x)
            params: Dict with:
                - target_chunk_area_um2: Target area for each chunk (default 1500)
                - min_ribbon_width_um: Expected minimum ribbon width
                - max_ribbon_width_um: Expected maximum ribbon width
                - min_fragment_area_um2: Skip fragments smaller than this
                - pixel_size_um: For converting pixels to microns

        Returns:
            Tuple of (masks, features_list) where features_list contains chunk polygons
        """
        import cv2
        from scipy import ndimage
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import skeletonize, medial_axis, remove_small_objects
        from skimage.morphology import binary_closing, binary_opening, disk
        from skimage.filters import meijering, threshold_local
        from skimage.measure import label, regionprops

        pixel_size = params.get('pixel_size_um', 0.22)
        target_area_um2 = params.get('target_chunk_area_um2', 1500)
        min_fragment_um2 = params.get('min_fragment_area_um2', 1500)  # Skip small fragments

        # Convert width parameters from µm to pixels
        min_width_px = params.get('min_ribbon_width_um', 5) / pixel_size
        max_width_px = params.get('max_ribbon_width_um', 30) / pixel_size

        # Convert to grayscale
        if image_rgb.ndim == 3:
            gray = np.mean(image_rgb[:, :, :3], axis=2).astype(np.float32)
        else:
            gray = image_rgb.astype(np.float32)

        # Normalize to 0-1 for ridge detection
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        # Ridge detection using Meijering filter (optimized for neurite/line structures)
        sigmas = np.linspace(min_width_px * 0.5, max_width_px * 0.5, 5)
        ridges = meijering(gray_norm, sigmas=sigmas, black_ridges=False)

        # Threshold ridge response
        ridge_thresh = threshold_local(ridges, block_size=51, offset=-0.01)
        binary = ridges > ridge_thresh

        # Morphological cleanup
        binary = binary_opening(binary, disk(1))
        binary = binary_closing(binary, disk(2))
        binary = remove_small_objects(binary, min_size=int(min_fragment_um2 / (pixel_size ** 2) * 0.1))

        # Label connected components and filter by total area
        labeled = label(binary)
        props = regionprops(labeled)

        # Keep only fragments large enough to chunk
        valid_labels = []
        for prop in props:
            area_um2 = prop.area * (pixel_size ** 2)
            if area_um2 >= min_fragment_um2:
                valid_labels.append(prop.label)

        if len(valid_labels) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint32), []

        # Create cleaned binary with only valid fragments
        binary_clean = np.isin(labeled, valid_labels)

        # Extract medial axis with distance transform
        skeleton, distance = medial_axis(binary_clean, return_distance=True)
        local_width = distance * 2  # Full width at skeleton points

        # Parse skeleton into paths using skan if available, else simple approach
        try:
            from skan import Skeleton as SkanSkeleton, summarize
            skel_obj = SkanSkeleton(skeleton)
            paths = skel_obj.paths_list()
        except ImportError:
            # Fallback: trace paths manually
            paths = self._trace_skeleton_paths(skeleton)

        # Chunk each path by area
        masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        features_list = []
        chunk_id = 1

        for path_idx, path_coords in enumerate(paths):
            if len(path_coords) < 3:
                continue

            # Get local width at each path point
            widths_px = []
            for pt in path_coords:
                r, c = int(pt[0]), int(pt[1])
                if 0 <= r < local_width.shape[0] and 0 <= c < local_width.shape[1]:
                    widths_px.append(max(local_width[r, c], 1))
                else:
                    widths_px.append(min_width_px)
            widths_um = np.array(widths_px) * pixel_size

            # Walk along path, accumulating area until target reached
            chunks = []
            accumulated_area = 0
            chunk_start_idx = 0

            for i in range(1, len(path_coords)):
                # Segment length
                dx = (path_coords[i][1] - path_coords[i-1][1]) * pixel_size
                dy = (path_coords[i][0] - path_coords[i-1][0]) * pixel_size
                seg_length = np.sqrt(dx**2 + dy**2)

                # Average width
                avg_width = (widths_um[i] + widths_um[i-1]) / 2

                # Segment area
                accumulated_area += seg_length * avg_width

                # Check if we've reached target
                if accumulated_area >= target_area_um2:
                    chunks.append({
                        'start_idx': chunk_start_idx,
                        'end_idx': i,
                        'path_points': path_coords[chunk_start_idx:i+1],
                        'widths_px': widths_px[chunk_start_idx:i+1],
                        'area_um2': accumulated_area
                    })
                    chunk_start_idx = i
                    accumulated_area = 0

            # Handle remainder - merge with previous if too small
            if chunk_start_idx < len(path_coords) - 1:
                remainder_area = accumulated_area
                if remainder_area < target_area_um2 * 0.5 and len(chunks) > 0:
                    # Merge with previous chunk
                    prev = chunks[-1]
                    prev['end_idx'] = len(path_coords) - 1
                    prev['path_points'] = np.vstack([prev['path_points'], path_coords[chunk_start_idx+1:]])
                    prev['widths_px'] = list(prev['widths_px']) + widths_px[chunk_start_idx+1:]
                    prev['area_um2'] += remainder_area
                elif remainder_area >= min_fragment_um2 * 0.5:
                    # Keep as separate chunk if not too small
                    chunks.append({
                        'start_idx': chunk_start_idx,
                        'end_idx': len(path_coords) - 1,
                        'path_points': path_coords[chunk_start_idx:],
                        'widths_px': widths_px[chunk_start_idx:],
                        'area_um2': remainder_area
                    })

            # Convert chunks to polygons
            for chunk in chunks:
                polygon = self._skeleton_chunk_to_polygon(
                    chunk['path_points'],
                    chunk['widths_px'],
                    pixel_size
                )

                if polygon is None or len(polygon) < 4:
                    continue

                # Create mask for this chunk
                chunk_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                cv2.fillPoly(chunk_mask, [polygon.astype(np.int32)], 255)
                chunk_mask_bool = chunk_mask > 0

                if chunk_mask_bool.sum() == 0:
                    continue

                masks[chunk_mask_bool] = chunk_id

                # Calculate centroid
                cy, cx = ndimage.center_of_mass(chunk_mask_bool)

                # Create feature dict
                features = {
                    'area_um2': float(chunk['area_um2']),
                    'path_length_um': float(len(chunk['path_points']) * pixel_size),
                    'mean_width_um': float(np.mean(chunk['widths_px']) * pixel_size),
                    'n_vertices': len(polygon),
                    'branch_id': path_idx,
                }

                features_list.append({
                    'id': f'meso_{chunk_id}',
                    'center': [float(cx), float(cy)],
                    'polygon_image': polygon.tolist(),  # In image coordinates (pixels)
                    'features': features
                })

                chunk_id += 1

        return masks, features_list

    def _trace_skeleton_paths(self, skeleton):
        """
        Simple skeleton path tracing (fallback if skan not available).
        Returns list of paths, each path is Nx2 array of (row, col) coordinates.
        """
        from scipy import ndimage
        from collections import deque

        # Find endpoints and branch points
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel)
        neighbor_count = neighbor_count * skeleton

        endpoints = (neighbor_count == 1) & skeleton
        branchpoints = (neighbor_count >= 3) & skeleton

        # Label skeleton segments
        labeled, n_segments = ndimage.label(skeleton)

        paths = []
        visited = np.zeros_like(skeleton, dtype=bool)

        # Start from each endpoint
        endpoint_coords = np.argwhere(endpoints)

        for start in endpoint_coords:
            if visited[start[0], start[1]]:
                continue

            # Trace path from this endpoint
            path = [start]
            visited[start[0], start[1]] = True
            current = start

            while True:
                # Find unvisited neighbors
                r, c = current
                found_next = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < skeleton.shape[0] and
                            0 <= nc < skeleton.shape[1] and
                            skeleton[nr, nc] and
                            not visited[nr, nc]):
                            path.append(np.array([nr, nc]))
                            visited[nr, nc] = True
                            current = np.array([nr, nc])
                            found_next = True
                            break
                    if found_next:
                        break

                if not found_next:
                    break

            if len(path) >= 3:
                paths.append(np.array(path))

        return paths

    def _skeleton_chunk_to_polygon(self, path_points, widths_px, pixel_size):
        """
        Convert skeleton path with widths to closed polygon.
        """
        path_points = np.array(path_points)
        widths_px = np.array(widths_px)

        if len(path_points) < 2:
            return None

        left_boundary = []
        right_boundary = []

        for i in range(len(path_points)):
            half_width = widths_px[i] / 2

            # Get tangent direction
            if i == 0:
                tangent = path_points[1] - path_points[0]
            elif i == len(path_points) - 1:
                tangent = path_points[-1] - path_points[-2]
            else:
                tangent = path_points[i+1] - path_points[i-1]

            norm = np.linalg.norm(tangent)
            if norm < 1e-6:
                continue
            tangent = tangent / norm

            # Perpendicular
            perp = np.array([-tangent[1], tangent[0]])

            # Boundary points (row, col format)
            left_pt = path_points[i] + perp * half_width
            right_pt = path_points[i] - perp * half_width

            left_boundary.append(left_pt)
            right_boundary.append(right_pt)

        if len(left_boundary) < 2:
            return None

        # Create closed polygon: left forward, then right backward
        # Convert to (col, row) = (x, y) for cv2
        polygon = np.vstack([
            np.array(left_boundary)[:, ::-1],  # (row,col) to (col,row)
            np.array(right_boundary)[::-1, ::-1]
        ])

        return polygon

    def process_tile(self, image_rgb, cell_type, params, cd31_channel=None):
        """
        Process a tile for the specified cell type.

        Args:
            image_rgb: RGB image array
            cell_type: 'nmj', 'mk', 'hspc', 'vessel', or 'mesothelium'
            params: Cell-type specific parameters
            cd31_channel: Optional CD31 channel for vessel validation

        Returns:
            Tuple of (masks, features_list)
        """
        if cell_type == 'nmj':
            return self.detect_nmj(image_rgb, params)
        elif cell_type == 'mk':
            return self.detect_mk(image_rgb, params)
        elif cell_type == 'hspc':
            return self.detect_hspc(image_rgb, params)
        elif cell_type == 'vessel':
            return self.detect_vessel(image_rgb, params, cd31_channel=cd31_channel)
        elif cell_type == 'mesothelium':
            return self.detect_mesothelium(image_rgb, params)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")


# =============================================================================
# CZI LOADING
# =============================================================================

def load_czi(czi_path):
    """Load CZI file and return reader + metadata."""
    from aicspylibczi import CziFile

    reader = CziFile(str(czi_path))

    bbox = reader.get_mosaic_bounding_box()
    mosaic_info = {
        'x': bbox.x,
        'y': bbox.y,
        'width': bbox.w,
        'height': bbox.h,
    }

    # Get pixel size
    pixel_size_um = 0.22
    try:
        meta = reader.meta
        for elem in meta.iter():
            if 'ScalingX' in elem.tag or 'Scale' in elem.tag:
                if elem.text:
                    try:
                        val = float(elem.text)
                        if val < 1e-3:
                            pixel_size_um = val * 1e6
                            break
                    except:
                        pass
    except:
        pass

    return reader, mosaic_info, pixel_size_um


def generate_tile_grid(mosaic_info, tile_size):
    """Generate tile coordinates covering the mosaic."""
    tiles = []
    x_start = mosaic_info['x']
    y_start = mosaic_info['y']
    width = mosaic_info['width']
    height = mosaic_info['height']

    for y in range(y_start, y_start + height, tile_size):
        for x in range(x_start, x_start + width, tile_size):
            tiles.append({'x': x, 'y': y})

    return tiles


# =============================================================================
# SAMPLE CREATION FOR HTML
# =============================================================================

# =============================================================================
# LMD EXPORT (for mesothelium)
# =============================================================================

def export_to_leica_lmd(detections, output_path, pixel_size_um, image_height_px,
                        image_width_px=None, calibration_points=None,
                        add_fiducials=True, fiducial_positions=None,
                        flip_y=True):
    """
    Export mesothelium chunks to Leica LMD XML format using py-lmd library.

    Args:
        detections: List of detection dicts with 'polygon_image' (pixel coords)
        output_path: Path to save XML file
        pixel_size_um: Microns per pixel
        image_height_px: Image height in pixels (for Y flip)
        image_width_px: Image width in pixels (for calibration)
        calibration_points: Optional 3x2 array of calibration points in µm
        add_fiducials: Whether to add calibration cross markers
        fiducial_positions: List of (x, y) positions in µm for fiducial crosses
                           If None and add_fiducials=True, uses image corners
        flip_y: Whether to flip Y axis for stage coordinates

    Returns:
        Path to saved file, also saves metadata CSV with both coordinate systems
    """
    try:
        from lmd.lib import Collection, Shape
        from lmd.tools import makeCross
        has_pylmd = True
    except ImportError:
        print("WARNING: py-lmd not installed. Install with: pip install py-lmd")
        print("  Falling back to simple XML export...")
        has_pylmd = False

    if image_width_px is None:
        image_width_px = 10000 / pixel_size_um  # Default estimate

    img_width_um = image_width_px * pixel_size_um
    img_height_um = image_height_px * pixel_size_um

    # Default calibration points (corners of image in µm)
    if calibration_points is None:
        calibration_points = np.array([
            [0, 0],
            [0, img_height_um],
            [img_width_um, img_height_um]
        ])

    # Default fiducial positions (corners + center)
    if fiducial_positions is None and add_fiducials:
        margin = 500  # µm margin from edges
        fiducial_positions = [
            (margin, margin),  # Top-left
            (img_width_um - margin, margin),  # Top-right
            (margin, img_height_um - margin),  # Bottom-left
            (img_width_um - margin, img_height_um - margin),  # Bottom-right
        ]

    if not has_pylmd:
        return _export_lmd_simple(detections, output_path, pixel_size_um,
                                  image_height_px, flip_y, fiducial_positions)

    # Create collection
    collection = Collection(calibration_points=calibration_points)

    # Add fiducial crosses
    if add_fiducials and fiducial_positions:
        for i, (fx, fy) in enumerate(fiducial_positions):
            # makeCross creates a cross shape at specified location
            cross = makeCross(
                center=np.array([fx, fy]),
                arm_length=100,  # µm
                arm_width=10,    # µm
            )
            collection.add_shape(Shape(
                points=cross,
                well="CAL",  # Special well for calibration
                name=f"Fiducial_{i+1}"
            ))

    # Add each chunk as a shape
    for i, det in enumerate(detections):
        if 'polygon_image' not in det:
            continue

        polygon_px = np.array(det['polygon_image'])

        # Convert to µm coordinates
        polygon_um = polygon_px * pixel_size_um

        # Flip Y if needed (image Y increases down, stage Y may increase up)
        if flip_y:
            polygon_um[:, 1] = img_height_um - polygon_um[:, 1]

        # Close polygon if not already closed
        if not np.allclose(polygon_um[0], polygon_um[-1]):
            polygon_um = np.vstack([polygon_um, polygon_um[0]])

        # Add to collection
        chunk_name = det.get('uid', det.get('id', f'Chunk_{i+1:04d}'))
        collection.new_shape(polygon_um, well="A1", name=chunk_name)

    # Save to XML
    collection.save(str(output_path))
    print(f"  Exported {len(detections)} chunks to LMD XML: {output_path}")
    if add_fiducials:
        print(f"  Added {len(fiducial_positions)} fiducial crosses for calibration")

    # Also save metadata CSV with both coordinate systems
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write('chunk_name,centroid_x_px,centroid_y_px,centroid_x_um,centroid_y_um,area_um2,n_vertices\n')
        for det in detections:
            if 'polygon_image' not in det:
                continue
            name = det.get('uid', det.get('id', ''))
            cx_px, cy_px = det['center']
            cx_um = cx_px * pixel_size_um
            cy_um = cy_px * pixel_size_um
            if flip_y:
                cy_um = img_height_um - cy_um
            area = det['features'].get('area_um2', 0)
            n_verts = det['features'].get('n_vertices', len(det.get('polygon_image', [])))
            f.write(f'{name},{cx_px:.1f},{cy_px:.1f},{cx_um:.2f},{cy_um:.2f},{area:.2f},{n_verts}\n')
    print(f"  Saved coordinates CSV: {csv_path}")

    return output_path


def _export_lmd_simple(detections, output_path, pixel_size_um, image_height_px,
                       flip_y=True, fiducial_positions=None):
    """
    Simple XML export fallback when py-lmd is not installed.
    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    root = ET.Element("ImageData")

    # Global coordinates
    global_coords = ET.SubElement(root, "GlobalCoordinates")
    ET.SubElement(global_coords, "OffsetX").text = "0"
    ET.SubElement(global_coords, "OffsetY").text = "0"

    # Shape list
    shape_list = ET.SubElement(root, "ShapeList")

    for i, det in enumerate(detections):
        if 'polygon_image' not in det:
            continue

        polygon_px = np.array(det['polygon_image'])

        # Convert to µm
        polygon_um = polygon_px * pixel_size_um

        # Flip Y if needed
        if flip_y:
            polygon_um[:, 1] = (image_height_px * pixel_size_um) - polygon_um[:, 1]

        shape = ET.SubElement(shape_list, "Shape")

        # Name
        name = det.get('uid', det.get('id', f'Chunk_{i+1:04d}'))
        ET.SubElement(shape, "Name").text = name
        ET.SubElement(shape, "ShapeType").text = "Polygon"

        # Points
        point_list = ET.SubElement(shape, "PointList")
        for x, y in polygon_um:
            point = ET.SubElement(point_list, "Point")
            ET.SubElement(point, "X").text = f"{x:.2f}"
            ET.SubElement(point, "Y").text = f"{y:.2f}"

    # Pretty print
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    with open(output_path, 'w') as f:
        f.write(xml_str)

    print(f"  Exported {len(detections)} chunks to simple XML: {output_path}")
    return output_path


# =============================================================================
# SAMPLE CREATION FOR HTML
# =============================================================================

def create_sample_from_detection(tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name, zoom_factor=7.5):
    """Create an HTML sample from a detection."""
    det_id = feat['id']
    det_num = int(det_id.split('_')[-1])
    mask = masks == det_num

    if mask.sum() == 0:
        return None

    # Get centroid
    cy, cx = feat['center'][1], feat['center'][0]

    # Calculate crop size
    ys, xs = np.where(mask)
    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_extent = max(mask_h, mask_w)
    crop_size = int(mask_extent * zoom_factor) if mask_extent > 0 else 500

    half = crop_size // 2
    y1 = max(0, int(cy) - half)
    y2 = min(tile_rgb.shape[0], int(cy) + half)
    x1 = max(0, int(cx) - half)
    x2 = min(tile_rgb.shape[1], int(cx) + half)

    crop = tile_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Normalize and draw contour
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5)
    crop_with_contour = draw_mask_contour(crop_norm, crop_mask, color=(0, 255, 0), thickness=2)

    # Resize to 300x300
    pil_img = Image.fromarray(crop_with_contour)
    pil_img = pil_img.resize((300, 300), Image.LANCZOS)

    img_b64, mime = image_to_base64(pil_img, format='PNG')

    # Create unique ID
    uid = f"{slide_name}_{tile_x}_{tile_y}_{det_id}"

    # Get stats from features
    features = feat['features']
    area_um2 = features.get('area', 0) * (pixel_size_um ** 2)

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }

    # Add cell-type specific stats
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'sam2_score' in features:
        stats['confidence'] = features['sam2_score']

    return {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime,
        'stats': stats,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(args):
    """Main pipeline execution."""
    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    slide_name = czi_path.stem

    print(f"\n{'='*60}")
    print(f"UNIFIED SEGMENTATION PIPELINE")
    print(f"{'='*60}")
    print(f"Slide: {slide_name}")
    print(f"Cell type: {args.cell_type}")
    print(f"Channel: {args.channel}")
    print(f"{'='*60}\n")

    # Load CZI
    print("Loading CZI file...")
    reader, mosaic_info, pixel_size_um = load_czi(czi_path)

    print(f"  Mosaic: {mosaic_info['width']} x {mosaic_info['height']} px")
    print(f"  Origin: ({mosaic_info['x']}, {mosaic_info['y']})")
    print(f"  Pixel size: {pixel_size_um:.4f} um/px")

    # Generate tile grid
    print(f"\nGenerating tile grid (size={args.tile_size})...")
    all_tiles = generate_tile_grid(mosaic_info, args.tile_size)
    print(f"  Total tiles: {len(all_tiles)}")

    # Calibrate tissue threshold
    print("\nCalibrating tissue threshold...")
    variance_threshold = calibrate_tissue_threshold(
        all_tiles,
        reader=reader,
        x_start=mosaic_info['x'],
        y_start=mosaic_info['y'],
        calibration_samples=min(50, len(all_tiles)),
        channel=args.channel,
        tile_size=args.tile_size,
    )

    # Filter to tissue-containing tiles
    print("\nFiltering to tissue-containing tiles...")
    tissue_tiles = filter_tissue_tiles(
        all_tiles,
        variance_threshold,
        reader=reader,
        x_start=mosaic_info['x'],
        y_start=mosaic_info['y'],
        channel=args.channel,
        tile_size=args.tile_size,
    )

    if len(tissue_tiles) == 0:
        print("ERROR: No tissue-containing tiles found!")
        return

    # Sample from tissue tiles
    n_sample = max(1, int(len(tissue_tiles) * args.sample_fraction))
    sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
    sampled_tiles = [tissue_tiles[i] for i in sample_indices]

    print(f"\nSampled {len(sampled_tiles)} tiles ({args.sample_fraction*100:.0f}% of {len(tissue_tiles)} tissue tiles)")

    # Setup output directories
    slide_output_dir = output_dir / slide_name
    tiles_dir = slide_output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Initialize segmenter
    # Always load SAM2 for full 2326 features (22 morph + 256 SAM2 + 2048 ResNet)
    print("\nInitializing segmenter...")
    load_sam2 = True  # Required for full feature extraction
    load_cellpose = args.cell_type == 'hspc'

    segmenter = UnifiedSegmenter(
        load_sam2=load_sam2,
        load_cellpose=load_cellpose,
    )

    # Detection parameters
    if args.cell_type == 'nmj':
        params = {
            'intensity_percentile': args.intensity_percentile,
            'min_area': args.min_area,
            'min_skeleton_length': args.min_skeleton_length,
            'min_elongation': args.min_elongation,
        }
    elif args.cell_type == 'mk':
        params = {
            'mk_min_area': args.mk_min_area,
            'mk_max_area': args.mk_max_area,
        }
    elif args.cell_type == 'hspc':
        params = {}
    elif args.cell_type == 'vessel':
        params = {
            'min_vessel_diameter_um': args.min_vessel_diameter,
            'max_vessel_diameter_um': args.max_vessel_diameter,
            'min_wall_thickness_um': args.min_wall_thickness,
            'max_aspect_ratio': args.max_aspect_ratio,
            'min_circularity': args.min_circularity,
            'min_ring_completeness': args.min_ring_completeness,
            'pixel_size_um': pixel_size_um,
            'classify_vessel_types': args.classify_vessel_types,
        }
    elif args.cell_type == 'mesothelium':
        params = {
            'target_chunk_area_um2': args.target_chunk_area,
            'min_ribbon_width_um': args.min_ribbon_width,
            'max_ribbon_width_um': args.max_ribbon_width,
            'min_fragment_area_um2': args.min_fragment_area,
            'pixel_size_um': pixel_size_um,
        }
    else:
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    print(f"\nDetection params: {params}")

    # Process tiles
    print("\nProcessing tiles...")
    all_samples = []
    all_detections = []  # Universal list with global coordinates
    total_detections = 0

    for tile in tqdm(sampled_tiles, desc="Tiles"):
        tile_x = tile['x']
        tile_y = tile['y']

        try:
            # Read tile
            tile_data = reader.read_mosaic(
                region=(tile_x, tile_y, args.tile_size, args.tile_size),
                scale_factor=1,
                C=args.channel
            )

            if tile_data is None or tile_data.size == 0:
                continue

            tile_data = np.squeeze(tile_data)

            if tile_data.max() == 0:
                continue

            # Convert to RGB
            if tile_data.ndim == 2:
                tile_rgb = np.stack([tile_data] * 3, axis=-1)
            else:
                tile_rgb = tile_data

            # Read CD31 channel if specified (for vessel validation)
            cd31_channel = None
            if args.cell_type == 'vessel' and args.cd31_channel is not None:
                cd31_data = reader.read_mosaic(
                    region=(tile_x, tile_y, args.tile_size, args.tile_size),
                    scale_factor=1,
                    C=args.cd31_channel
                )
                if cd31_data is not None and cd31_data.size > 0:
                    cd31_channel = np.squeeze(cd31_data).astype(np.float32)

            # Detect cells
            masks, features_list = segmenter.process_tile(
                tile_rgb, args.cell_type, params, cd31_channel=cd31_channel
            )

            if len(features_list) == 0:
                continue

            # Add universal IDs and global coordinates to each detection
            for feat in features_list:
                local_cx, local_cy = feat['center']
                global_cx = tile_x + local_cx
                global_cy = tile_y + local_cy

                # Create universal ID: slide_celltype_globalX_globalY_localID
                uid = f"{slide_name}_{args.cell_type}_{int(global_cx)}_{int(global_cy)}"
                feat['uid'] = uid
                feat['global_center'] = [float(global_cx), float(global_cy)]
                feat['global_center_um'] = [float(global_cx * pixel_size_um), float(global_cy * pixel_size_um)]
                feat['tile_origin'] = [tile_x, tile_y]
                feat['slide_name'] = slide_name

                # Convert contours to global coordinates if present (vessels)
                if 'outer_contour' in feat:
                    feat['outer_contour_global'] = [[pt[0][0] + tile_x, pt[0][1] + tile_y]
                                                    for pt in feat['outer_contour']]
                if 'inner_contour' in feat:
                    feat['inner_contour_global'] = [[pt[0][0] + tile_x, pt[0][1] + tile_y]
                                                    for pt in feat['inner_contour']]

                all_detections.append(feat)

            # Save masks and features
            tile_id = f"tile_{tile_x}_{tile_y}"
            tile_out = tiles_dir / tile_id
            tile_out.mkdir(exist_ok=True)

            with h5py.File(tile_out / f"{args.cell_type}_masks.h5", 'w') as f:
                create_hdf5_dataset(f, 'masks', masks.astype(np.uint16))

            with open(tile_out / f"{args.cell_type}_features.json", 'w') as f:
                json.dump(features_list, f)

            # Create samples for HTML
            for feat in features_list:
                sample = create_sample_from_detection(
                    tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name
                )
                if sample:
                    all_samples.append(sample)
                    total_detections += 1

            del tile_data, tile_rgb, masks
            if cd31_channel is not None:
                del cd31_channel
            gc.collect()

        except Exception as e:
            print(f"\n  Error processing tile ({tile_x}, {tile_y}): {e}")
            continue

    print(f"\nTotal detections: {total_detections}")

    # Sort samples by area
    all_samples.sort(key=lambda x: x['stats'].get('area_um2', 0))

    # Export to HTML
    print(f"\nExporting to HTML ({len(all_samples)} samples)...")
    html_dir = slide_output_dir / "html"

    export_samples_to_html(
        all_samples,
        html_dir,
        args.cell_type,
        samples_per_page=args.samples_per_page,
        title=f"{args.cell_type.upper()} Detection - {slide_name}",
        subtitle=f"Pixel size: {pixel_size_um:.4f} um/px | {len(sampled_tiles)} tiles processed",
        extra_stats={
            'Tissue tiles': len(tissue_tiles),
            'Sampled': len(sampled_tiles),
        },
        page_prefix=f'{args.cell_type}_page',
    )

    # Save all detections with universal IDs and global coordinates
    detections_file = slide_output_dir / f'{args.cell_type}_detections.json'
    with open(detections_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"  Saved {len(all_detections)} detections to {detections_file}")

    # Export CSV with contour coordinates for easy import
    csv_file = slide_output_dir / f'{args.cell_type}_coordinates.csv'
    with open(csv_file, 'w') as f:
        # Header
        if args.cell_type == 'vessel':
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,outer_diameter_um,wall_thickness_um,confidence\n')
            for det in all_detections:
                feat = det.get('features', {})
                f.write(f"{det['uid']},{det['global_center'][0]:.1f},{det['global_center'][1]:.1f},"
                        f"{det['global_center_um'][0]:.2f},{det['global_center_um'][1]:.2f},"
                        f"{feat.get('outer_diameter_um', 0):.2f},{feat.get('wall_thickness_mean_um', 0):.2f},"
                        f"{feat.get('confidence', 'unknown')}\n")
        else:
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,area_um2\n')
            for det in all_detections:
                feat = det.get('features', {})
                area_um2 = feat.get('area', 0) * (pixel_size_um ** 2)
                f.write(f"{det['uid']},{det['global_center'][0]:.1f},{det['global_center'][1]:.1f},"
                        f"{det['global_center_um'][0]:.2f},{det['global_center_um'][1]:.2f},{area_um2:.2f}\n")
    print(f"  Saved coordinates to {csv_file}")

    # Export to Leica LMD format for mesothelium
    if args.cell_type == 'mesothelium' and len(all_detections) > 0:
        print(f"\nExporting to Leica LMD XML format...")
        lmd_file = slide_output_dir / f'{args.cell_type}_chunks.xml'
        export_to_leica_lmd(
            all_detections,
            lmd_file,
            pixel_size_um,
            image_height_px=mosaic_info['height'],
            image_width_px=mosaic_info['width'],
            add_fiducials=args.add_fiducials,
            flip_y=True,
        )

    # Save summary
    summary = {
        'slide_name': slide_name,
        'cell_type': args.cell_type,
        'pixel_size_um': pixel_size_um,
        'mosaic_width': mosaic_info['width'],
        'mosaic_height': mosaic_info['height'],
        'total_tiles': len(all_tiles),
        'tissue_tiles': len(tissue_tiles),
        'sampled_tiles': len(sampled_tiles),
        'total_detections': total_detections,
        'params': params,
        'detections_file': str(detections_file),
        'coordinates_file': str(csv_file),
    }

    with open(slide_output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {slide_output_dir}")
    print(f"HTML viewer: {html_dir / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified Cell Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument('--czi-path', type=str, required=True, help='Path to CZI file')
    parser.add_argument('--cell-type', type=str, required=True,
                        choices=['nmj', 'mk', 'hspc', 'vessel', 'mesothelium'],
                        help='Cell type to detect')

    # Output
    parser.add_argument('--output-dir', type=str, default='/home/dude/nmj_output', help='Output directory')

    # Tile processing
    parser.add_argument('--tile-size', type=int, default=3000, help='Tile size in pixels')
    parser.add_argument('--sample-fraction', type=float, default=0.10, help='Fraction of tissue tiles')
    parser.add_argument('--channel', type=int, default=1, help='Channel index')

    # NMJ parameters
    parser.add_argument('--intensity-percentile', type=float, default=99)
    parser.add_argument('--min-area', type=int, default=150)
    parser.add_argument('--min-skeleton-length', type=int, default=30)
    parser.add_argument('--min-elongation', type=float, default=1.5)

    # MK parameters
    parser.add_argument('--mk-min-area', type=int, default=1000)
    parser.add_argument('--mk-max-area', type=int, default=100000)

    # Vessel parameters
    parser.add_argument('--min-vessel-diameter', type=float, default=10,
                        help='Minimum vessel outer diameter in µm')
    parser.add_argument('--max-vessel-diameter', type=float, default=1000,
                        help='Maximum vessel outer diameter in µm')
    parser.add_argument('--min-wall-thickness', type=float, default=2,
                        help='Minimum vessel wall thickness in µm')
    parser.add_argument('--max-aspect-ratio', type=float, default=4.0,
                        help='Maximum aspect ratio (exclude longitudinal sections)')
    parser.add_argument('--min-circularity', type=float, default=0.3,
                        help='Minimum circularity for vessel detection')
    parser.add_argument('--min-ring-completeness', type=float, default=0.5,
                        help='Minimum ring completeness (fraction of SMA+ perimeter)')
    parser.add_argument('--cd31-channel', type=int, default=None,
                        help='CD31 channel index for vessel validation (optional)')
    parser.add_argument('--classify-vessel-types', action='store_true',
                        help='Auto-classify vessels by size (capillary/arteriole/artery)')

    # Mesothelium parameters (for LMD chunking)
    parser.add_argument('--target-chunk-area', type=float, default=1500,
                        help='Target area for mesothelium chunks in µm²')
    parser.add_argument('--min-ribbon-width', type=float, default=5,
                        help='Minimum expected ribbon width in µm')
    parser.add_argument('--max-ribbon-width', type=float, default=30,
                        help='Maximum expected ribbon width in µm')
    parser.add_argument('--min-fragment-area', type=float, default=1500,
                        help='Skip mesothelium fragments smaller than this (µm²)')
    parser.add_argument('--add-fiducials', action='store_true', default=True,
                        help='Add calibration cross markers to LMD export')
    parser.add_argument('--no-fiducials', dest='add_fiducials', action='store_false',
                        help='Do not add calibration markers')

    # Feature extraction (always enabled - 2326 features per detection)
    # Kept for backwards compatibility but no longer needed
    parser.add_argument('--extract-full-features', action='store_true',
                        help='(Deprecated) Full features always extracted')

    # HTML export
    parser.add_argument('--samples-per-page', type=int, default=300)

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()
