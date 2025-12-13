"""
Unified segmentation: MKs (SAM2) + HSPCs (Cellpose-SAM) in one pass.

Processes each tile once and outputs:
- MK masks (SAM2 automatic mask generation, size-filtered by mk-min/max-area)
- HSPC masks (Cellpose-SAM detection + SAM2 refinement, size-invariant)
- All features: 22 custom + 256 SAM2 + 2048 ResNet = 2326 per cell

Usage:
    python run_unified_segmentation.py \
        --czi-path /path/to/slide.czi \
        --output-dir /path/to/output \
        --mk-min-area 1000
"""

# Set cellpose model path FIRST, before any imports (cellpose caches this at import time)
import os
os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = '/ptmp/edrod/MKsegmentation/checkpoints'

import numpy as np
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
import torch.multiprocessing as mp
from multiprocessing import shared_memory
from PIL import Image
import psutil

# Global variable for worker process
segmenter = None
shared_image = None

def init_worker(mk_classifier_path, hspc_classifier_path, gpu_queue, mm_path, mm_shape, mm_dtype):
    """Initialize the segmenter and attach to shared memory map once per worker process."""
    global segmenter, shared_image
    
    # Assign GPU
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # Get a GPU ID from the queue
            gpu_id = gpu_queue.get(timeout=5)
            device = f"cuda:{gpu_id}"
        except:
            # Fallback: simple modulo or default
            n_gpus = torch.cuda.device_count()
            if n_gpus > 0:
                gpu_id = mp.current_process().pid % n_gpus
                device = f"cuda:{gpu_id}"
            else:
                device = "cuda"
    
    print(f"Worker {mp.current_process().pid} initialized on {device}", flush=True)
    
    # Initialize Segmenter
    segmenter = UnifiedSegmenter(
        mk_classifier_path=mk_classifier_path,
        hspc_classifier_path=hspc_classifier_path,
        device=device
    )

    # Attach to Memory Map (Read-Only)
    try:
        shared_image = np.memmap(mm_path, dtype=mm_dtype, mode='r', shape=mm_shape)
        print(f"Worker {mp.current_process().pid} attached to memmap: {mm_path}", flush=True)
    except Exception as e:
        print(f"Worker {mp.current_process().pid} FAILED to attach to memmap: {e}", flush=True)
        shared_image = None

def process_tile_worker(args):
    """
    Worker function for processing a single tile in a separate process.
    """
    # Unpack all arguments
    tile, _, _, _, output_dir, \
    mk_min_area, mk_max_area, variance_threshold, \
    calibration_block_size = args

    # Use global variables
    global segmenter, shared_image
    
    tid = tile['id']
    
    # Extract tile from Shared Memory
    if shared_image is None:
        return {'tid': tid, 'status': 'error', 'error': "Shared memory not available in worker"}

    try:
        # Slice directly from shared memory (no disk I/O)
        # Coordinate system of shared_image matches the tiling grid (0,0 is start of image ROI)
        img = shared_image[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]
        
        # Check if valid
        if img.size == 0:
             return {'tid': tid, 'status': 'error', 'error': f"Empty crop extracted for tile {tid}"}
             
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Memory slice error: {e}"}

    # Convert to RGB
    if img.ndim == 2:
        img_rgb = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 4:
        img_rgb = img[:, :, :3]
    else:
        img_rgb = img

    if img_rgb.max() == 0:
        return {'tid': tid, 'status': 'empty'}

    img_rgb = percentile_normalize(img_rgb, p_low=5, p_high=95)
    has_tissue_content, _ = has_tissue(img_rgb, variance_threshold, block_size=calibration_block_size)
    if not has_tissue_content:
        return {'tid': tid, 'status': 'no_tissue'}

    try:
        mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
            img_rgb, mk_min_area, mk_max_area
        )
        return {
            'tid': tid, 'status': 'success',
            'mk_masks': mk_masks, 'hspc_masks': hspc_masks,
            'mk_feats': mk_feats, 'hspc_feats': hspc_feats,
            'tile': tile
        }
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Processing error: {e}"}


_ROCM_PATCH_APPLIED = False

def _apply_rocm_patch_if_needed():
    """Apply ROCm INT_MAX fix lazily - call this before using SAM2."""
    global _ROCM_PATCH_APPLIED
    if _ROCM_PATCH_APPLIED:
        return
    _ROCM_PATCH_APPLIED = True

    try:
        import torch
        from typing import List, Dict, Any
        import sam2.utils.amg as amg

        def mask_to_rle_pytorch_rocm_safe(tensor: torch.Tensor) -> List[Dict[str, Any]]:
            """
            Encodes masks to an uncompressed RLE, with ROCm INT_MAX workaround.
            Moves tensor to CPU before nonzero() to avoid INT_MAX issues.
            """
            # Put in fortran order and flatten h,w
            b, h, w = tensor.shape
            tensor = tensor.permute(0, 2, 1).flatten(1)

            # Compute change indices
            diff = tensor[:, 1:] ^ tensor[:, :-1]

            # ROCm FIX: Move to CPU before nonzero() to avoid INT_MAX error
            diff_cpu = diff.cpu()
            change_indices = diff_cpu.nonzero()

            # Encode run length
            out = []
            for i in range(b):
                cur_idxs = change_indices[change_indices[:, 0] == i, 1]
                cur_idxs = torch.cat(
                    [
                        torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                        cur_idxs + 1,
                        torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
                    ]
                )
                btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
                counts = [] if tensor[i, 0] == 0 else [0]
                counts.extend(btw_idxs.detach().cpu().tolist())
                out.append({"size": [h, w], "counts": counts})
            return out

        # Apply the patch
        amg.mask_to_rle_pytorch = mask_to_rle_pytorch_rocm_safe
        print("[ROCm FIX] Patched sam2.utils.amg.mask_to_rle_pytorch for INT_MAX workaround")
    except ImportError as e:
        print(f"[ROCm FIX] Could not apply patch: {e}")

def get_pixel_size_from_czi(czi_path):
    """Extract pixel size in microns from CZI metadata."""
    from pylibCZIrw import czi as pyczi

    reader = pyczi.CziReader(str(czi_path))
    meta = reader.metadata
    reader.close()

    try:
        scaling = meta['ImageDocument']['Metadata']['Scaling']['Items']['Distance']
        px_x = px_y = None
        for item in scaling:
            if item.get('@Id') == 'X':
                px_x = float(item['Value']) * 1e6
            elif item.get('@Id') == 'Y':
                px_y = float(item['Value']) * 1e6
        if px_x and px_y:
            return px_x, px_y
    except (KeyError, TypeError):
        pass
    raise ValueError(f"Could not extract pixel size from {czi_path}")


def percentile_normalize(image, p_low=5, p_high=95):
    """
    Normalize image intensity using percentile scaling.
    Maps p_low to p_high percentile range to 0-255.
    Reduces slide-to-slide variation from staining differences.
    """
    if image.ndim == 2:
        low_val = np.percentile(image, p_low)
        high_val = np.percentile(image, p_high)
        if high_val > low_val:
            normalized = (image.astype(np.float32) - low_val) / (high_val - low_val) * 255
            return np.clip(normalized, 0, 255).astype(np.uint8)
        return image.astype(np.uint8)
    elif image.ndim == 3:
        # Normalize each channel independently
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            result[:, :, c] = percentile_normalize(image[:, :, c], p_low, p_high)
        return result
    return image


def calculate_block_variances(gray_image, block_size=512):
    """Calculate variance for each block in the image."""
    variances = []
    height, width = gray_image.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = gray_image[y:y+block_size, x:x+block_size]
            if block.size < (block_size * block_size) / 4:
                continue
            variances.append(np.var(block))

    return variances


def calibrate_tissue_threshold(tiles, reader=None, x_start=0, y_start=0, calibration_samples=50, block_size=512, image_array=None):
    """
    Auto-detect variance threshold using K-means clustering.
    Supports reading from CZI reader OR memory-mapped/shared array.
    """
    import cv2
    from sklearn.cluster import KMeans

    print(f"Calibrating tissue threshold (K-means 3-cluster on {calibration_samples} random tiles)...")

    # Sample 50 tiles
    n_calibration = calibration_samples
    np.random.seed(42)
    calibration_tiles = list(np.random.choice(tiles, min(n_calibration, len(tiles)), replace=False))

    all_variances = []
    empty_count = 0

    for tile in calibration_tiles:
        # Extract tile from Array (Preferred) or CZI
        if image_array is not None:
            # image_array is (H, W, C) or (H, W) matching the tiles grid
            img = image_array[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]
        elif reader is not None:
            roi = (x_start + tile['x'], y_start + tile['y'], tile['w'], tile['h'])
            try:
                img = reader.read(roi=roi, plane={'C': 0})
            except:
                continue
        else:
            continue

        # Empty tiles contribute low variance (important for calibration)
        if img.max() == 0:
            n_blocks = (tile['w'] // block_size) * (tile['h'] // block_size)
            all_variances.extend([0.0] * max(1, n_blocks))
            empty_count += 1
            continue

        # Percentile normalize (5-95%) to standardize variance across slides
        if img.ndim == 3:
            img_norm = percentile_normalize(img, p_low=5, p_high=95)
            gray = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
        else:
            gray = percentile_normalize(img, p_low=5, p_high=95)

        block_vars = calculate_block_variances(gray, block_size)
        if block_vars:
            all_variances.extend(block_vars)

    if len(all_variances) < 20:
        print("  WARNING: Not enough samples, using default threshold 15.0")
        return 15.0

    variances = np.array(all_variances)

    # K-means with 3 clusters: background (low var), tissue (medium var), artifacts (high var)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(variances.reshape(-1, 1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    # Find the bottom cluster (lowest center = background)
    sorted_indices = np.argsort(centers)
    bottom_cluster_idx = sorted_indices[0]

    # Threshold = max variance in the bottom cluster (to exclude all background blocks)
    bottom_cluster_variances = variances[labels == bottom_cluster_idx]
    threshold = float(np.max(bottom_cluster_variances)) if len(bottom_cluster_variances) > 0 else 15.0

    print(f"  K-means centers: {sorted(centers)[0]:.1f} (bg), {sorted(centers)[1]:.1f} (tissue), {sorted(centers)[2]:.1f} (outliers)")
    print(f"  Threshold (bg cluster max): {threshold:.1f}")
    print(f"  Sampled {len(calibration_tiles)} tiles ({empty_count} empty), {len(variances)} blocks")

    return threshold


def has_tissue(tile_image, variance_threshold, min_tissue_fraction=0.15, block_size=512):
    """
    Check if a tile contains tissue using block-based variance.
    """
    import cv2

    # Handle all-black tiles (empty CZI regions)
    if tile_image.max() == 0:
        return False, 0.0

    # Convert to grayscale (already normalized)
    if tile_image.ndim == 3:
        gray = cv2.cvtColor(tile_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = tile_image.astype(np.uint8)

    variances = calculate_block_variances(gray, block_size)

    if len(variances) == 0:
        return False, 0.0

    tissue_blocks = sum(1 for v in variances if v >= variance_threshold)
    tissue_fraction = tissue_blocks / len(variances)

    return tissue_fraction >= min_tissue_fraction, tissue_fraction


def extract_morphological_features(mask, image):
    """Extract 22 morphological/intensity features from a mask."""
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
        gray = image[mask]
        red_mean = green_mean = blue_mean = float(np.mean(gray))
        red_std = green_std = blue_std = float(np.std(gray))

    gray_mean, gray_std = float(np.mean(gray)), float(np.std(gray))

    # HSV features
    if image.ndim == 3:
        import colorsys
        hsv = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in masked_pixels])
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


class UnifiedSegmenter:
    """Unified segmenter for MKs and HSPCs."""

    def __init__(
        self,
        sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        mk_classifier_path=None,
        hspc_classifier_path=None,
        device=None
    ):
        _apply_rocm_patch_if_needed()
        from cellpose.models import CellposeModel
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        # Convert device to torch.device if it's a string
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Find SAM2 checkpoint
        sam2_base = Path("/ptmp/edrod/MKsegmentation")
        checkpoint_path = sam2_base / sam2_checkpoint

        print(f"Loading SAM2 from {checkpoint_path}...")
        sam2_model = build_sam2(sam2_config, str(checkpoint_path), device=self.device)

        # SAM2 for auto mask generation (MKs)
        self.sam2_auto = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=24, # Changed from 32 to 24
            pred_iou_thresh=0.4,
            stability_score_thresh=0.4,
            min_mask_region_area=500,
            crop_n_layers=1 # Added crop_n_layers
        )

        # SAM2 predictor for point prompts (HSPCs)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Cellpose-SAM for HSPC detection (v4+ with SAM backbone)
        print(f"Loading Cellpose-SAM model on {self.device}...")
        self.cellpose = CellposeModel(pretrained_model='cpsam', gpu=True, device=self.device)

        # ResNet for deep features
        print("Loading ResNet-50...")
        resnet = tv_models.resnet50(weights='DEFAULT')
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.resnet.eval().to(self.device)
        self.resnet_transform = tv_transforms.Compose([
            tv_transforms.Resize(224),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load classifiers if provided
        self.mk_classifier = None
        self.mk_feature_names = None
        self.hspc_classifier = None
        self.hspc_feature_names = None

        if mk_classifier_path:
            print(f"Loading MK classifier: {mk_classifier_path}")
            import joblib
            clf_data = joblib.load(mk_classifier_path)
            self.mk_classifier = clf_data['classifier']
            self.mk_feature_names = clf_data['feature_names']
            print(f"  Features: {len(self.mk_feature_names)}, Trained on {clf_data.get('n_samples', '?')} samples")

        if hspc_classifier_path:
            print(f"Loading HSPC classifier: {hspc_classifier_path}")
            import joblib
            clf_data = joblib.load(hspc_classifier_path)
            self.hspc_classifier = clf_data['classifier']
            self.hspc_feature_names = clf_data['feature_names']
            print(f"  Features: {len(self.hspc_feature_names)}, Trained on {clf_data.get('n_samples', '?')} samples")

        print("Models loaded.")

    def apply_classifier(self, features_dict, cell_type):
        """Apply classifier to features and return (is_positive, confidence).

        Args:
            features_dict: Dict of feature_name -> value
            cell_type: 'mk' or 'hspc'

        Returns:
            (bool, float): (is_positive, confidence_score)
        """
        if cell_type == 'mk':
            clf = self.mk_classifier
            feature_names = self.mk_feature_names
        else:
            clf = self.hspc_classifier
            feature_names = self.hspc_feature_names

        if clf is None:
            return True, 1.0  # No classifier = accept all

        # Build feature vector in correct order
        X = np.array([[features_dict.get(name, 0.0) for name in feature_names]])

        # Predict
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        confidence = proba[1] if pred == 1 else proba[0]

        return bool(pred == 1), float(confidence)

    def extract_resnet_features(self, crop):
        """Extract 2048D ResNet features from crop."""
        if crop.ndim == 2:
            crop = np.stack([crop]*3, axis=-1)
        pil_img = Image.fromarray(crop.astype(np.uint8))
        tensor = self.resnet_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.resnet(tensor).cpu().numpy().flatten()
        return features

    def extract_sam2_embedding(self, cy, cx):
        """Extract 256D SAM2 embedding at location."""
        try:
            emb_y = int(cy / 16)
            emb_x = int(cx / 16)
            shape = self.sam2_predictor._features.shape
            emb_y = min(max(0, emb_y), shape[2] - 1)
            emb_x = min(max(0, emb_x), shape[3] - 1)
            return self.sam2_predictor._features[0, :, emb_y, emb_x].cpu().numpy()
        except:
            return np.zeros(256)

    def process_tile(
        self,
        image_rgb,
        mk_min_area=1000,
        mk_max_area=100000
    ):
        """Process a single tile for both MKs and HSPCs.

        Note: mk_min_area/mk_max_area only filter MKs, not HSPCs.
        Cellpose-SAM handles HSPC detection without size parameters.

        Returns:
            mk_masks: Label array for MKs
            hspc_masks: Label array for HSPCs
            mk_features: List of feature dicts
            hspc_features: List of feature dicts
        """
        from scipy import ndimage

        # ============================================
        # MK Detection: SAM2 automatic mask generation
        # ============================================
        # Process MK detection FIRST (auto generator manages its own predictor)
        # This avoids holding embeddings in TWO predictors simultaneously
        sam2_results = self.sam2_auto.generate(image_rgb)

        # Filter by size and sort by area (largest first)
        valid_results = []
        for result in sam2_results:
            area = result['segmentation'].sum()
            if mk_min_area <= area <= mk_max_area:
                result['area'] = area
                valid_results.append(result)
        valid_results.sort(key=lambda x: x['area'], reverse=True)

        # Delete original sam2_results to free memory (can be 3GB+ of masks)
        del sam2_results
        import gc
        gc.collect()

        mk_masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        mk_features = []
        mk_id = 1

        for result in valid_results:
            mask = result['segmentation']

            # Check overlap with existing masks - skip if >50% overlaps (larger already added)
            if mk_masks.max() > 0:
                overlap = ((mask > 0) & (mk_masks > 0)).sum()
                if overlap > 0.5 * mask.sum():
                    continue

            # Add to label array
            mk_masks[mask] = mk_id

            # Get centroid
            cy, cx = ndimage.center_of_mass(mask)

            # Extract all features
            morph = extract_morphological_features(mask, image_rgb)

            # SAM2 embeddings
            sam2_emb = self.extract_sam2_embedding(cy, cx)
            for i, v in enumerate(sam2_emb):
                morph[f'sam2_emb_{i}'] = float(v)

            # ResNet features from masked crop
            ys, xs = np.where(mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop = image_rgb[y1:y2+1, x1:x2+1].copy()
            crop_mask = mask[y1:y2+1, x1:x2+1]
            crop[~crop_mask] = 0

            resnet_feats = self.extract_resnet_features(crop)
            for i, v in enumerate(resnet_feats):
                morph[f'resnet_{i}'] = float(v)

            # Apply classifier if available
            is_positive, confidence = self.apply_classifier(morph, 'mk')

            if not is_positive:
                # Remove from mask if classifier rejects
                mk_masks[mk_masks == mk_id] = 0
                continue

            mk_features.append({
                'id': f'mk_{mk_id}',
                'center': [float(cx), float(cy)],
                'sam2_iou': float(result.get('predicted_iou', 0)),
                'sam2_stability': float(result.get('stability_score', 0)),
                'classifier_confidence': confidence,
                'features': morph
            })

            mk_id += 1

        # Delete valid_results to free memory (large mask arrays)
        del valid_results
        gc.collect()

        # ============================================
        # HSPC Detection: Cellpose-SAM + SAM2 refinement
        # ============================================
        # Now set image for SAM2 predictor (after auto generator is done)
        # This ensures we only hold ONE predictor's embeddings at a time
        self.sam2_predictor.set_image(image_rgb)

        # Cellpose with grayscale mode, let cpsam auto-detect size
        cellpose_masks, _, _ = self.cellpose.eval(image_rgb, channels=[0,0])

        # Get Cellpose centroids
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        # Collect all HSPC candidates with SAM2 refinement
        hspc_candidates = []
        for cp_id in cellpose_ids:
            cp_mask = cellpose_masks == cp_id
            cy, cx = ndimage.center_of_mass(cp_mask)

            # Use centroid as SAM2 prompt
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])

            masks_pred, scores, _ = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # Take best mask
            best_idx = np.argmax(scores)
            sam2_mask = masks_pred[best_idx]
            sam2_score = float(scores[best_idx])

            if sam2_mask.sum() < 10:
                continue

            # Skip if overlaps significantly with MKs (it's probably part of an MK)
            mk_overlap = ((sam2_mask > 0) & (mk_masks > 0)).sum() / sam2_mask.sum()
            if mk_overlap > 0.5:
                continue

            hspc_candidates.append({
                'mask': sam2_mask,
                'score': sam2_score,
                'center': (cx, cy),
                'cp_id': cp_id
            })

        # Sort by SAM2 score (most confident first) and handle overlaps
        hspc_candidates.sort(key=lambda x: x['score'], reverse=True)

        hspc_masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        hspc_features = []
        hspc_id = 1

        for cand in hspc_candidates:
            sam2_mask = cand['mask']
            sam2_score = cand['score']
            cx, cy = cand['center']

            # Check overlap with existing HSPC masks - skip if >50% overlaps
            if hspc_masks.max() > 0:
                overlap = ((sam2_mask > 0) & (hspc_masks > 0)).sum()
                if overlap > 0.5 * sam2_mask.sum():
                    continue

            # Add to label array
            hspc_masks[sam2_mask] = hspc_id

            # Extract features
            morph = extract_morphological_features(sam2_mask, image_rgb)

            # SAM2 embeddings
            sam2_emb = self.extract_sam2_embedding(cy, cx)
            for i, v in enumerate(sam2_emb):
                morph[f'sam2_emb_{i}'] = float(v)

            # ResNet features
            ys, xs = np.where(sam2_mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop = image_rgb[y1:y2+1, x1:x2+1].copy()
            crop_mask = sam2_mask[y1:y2+1, x1:x2+1]
            crop[~crop_mask] = 0

            resnet_feats = self.extract_resnet_features(crop)
            for i, v in enumerate(resnet_feats):
                morph[f'resnet_{i}'] = float(v)

            # Apply classifier if available
            is_positive, confidence = self.apply_classifier(morph, 'hspc')

            if not is_positive:
                hspc_masks[hspc_masks == hspc_id] = 0
                continue

            hspc_features.append({
                'id': f'hspc_{hspc_id}',
                'center': [float(cx), float(cy)],
                'cellpose_id': int(cand['cp_id']),
                'sam2_score': sam2_score,
                'classifier_confidence': confidence,
                'features': morph
            })

            hspc_id += 1

        # Delete large temporary arrays to free memory
        del hspc_candidates
        del cellpose_masks
        gc.collect()

        # Clear SAM2 cached features after processing this tile
        self.sam2_predictor.reset_predictor()
        torch.cuda.empty_cache()

        return mk_masks, hspc_masks, mk_features, hspc_features


def run_unified_segmentation(
    czi_path,
    output_dir,
    mk_min_area=1000,
    mk_max_area=100000,
    tile_size=4096,
    overlap=512,
    sample_fraction=1.0,
    calibration_block_size=512,
    calibration_samples=50,
    num_workers=4,
    mk_classifier_path=None,
    hspc_classifier_path=None
):
    """Run unified MK + HSPC segmentation with multiprocessing."""
    from pylibCZIrw import czi as pyczi

    # Set start method to 'spawn' for GPU safety
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)

    czi_path = Path(czi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("UNIFIED SEGMENTATION: MK + HSPC (MULTIPROCESSING)")
    print(f"{'='*70}")
    print(f"CZI: {czi_path}")

    # Open CZI
    reader = pyczi.CziReader(str(czi_path))
    scenes = reader.scenes_bounding_rectangle
    if scenes:
        rect = scenes[0]
        x_start, y_start = rect.x, rect.y
        full_width, full_height = rect.w, rect.h
    else:
        bbox = reader.total_bounding_box
        x_start, y_start = bbox['X'][0], bbox['Y'][0]
        full_width = bbox['X'][1] - bbox['X'][0]
        full_height = bbox['Y'][1] - bbox['Y'][0]

    print(f"Image: {full_width} x {full_height}")

    # Load full image into Memory Map (File-backed)
    print("Loading image into Memory Map...", flush=True)
    # Note: read() returns (H, W, C) or (H, W)
    full_img = reader.read(plane={"C": 0, "T": 0, "Z": 0}, roi=(x_start, y_start, full_width, full_height))
    
    # Create temporary directory for memmap
    import shutil
    temp_mm_dir = output_dir / "temp_mm"
    temp_mm_dir.mkdir(parents=True, exist_ok=True)
    mm_path = temp_mm_dir / "image.dat"
    
    # Create writable memmap
    shm_arr = np.memmap(mm_path, dtype=full_img.dtype, mode='w+', shape=full_img.shape)
    
    # Copy image to memmap
    np.copyto(shm_arr, full_img)
    shm_arr.flush()
    print(f"Image loaded to Memory Map: {mm_path} ({full_img.nbytes / 1024**3:.2f} GB)")
    
    # Capture shape/dtype for workers
    mm_shape = full_img.shape
    mm_dtype = full_img.dtype
    
    # Free local memory
    del full_img
    del shm_arr # Close handle in main process
    
    # Create tiles
    n_tx = int(np.ceil(full_width / (tile_size - overlap)))
    n_ty = int(np.ceil(full_height / (tile_size - overlap)))
    tiles = []
    for ty in range(n_ty):
        for tx in range(n_tx):
            tiles.append({
                'id': len(tiles),
                'x': tx * (tile_size - overlap),
                'y': ty * (tile_size - overlap),
                'w': min(tile_size, full_width - tx * (tile_size - overlap)),
                'h': min(tile_size, full_height - ty * (tile_size - overlap))
            })

    print(f"Total tiles: {len(tiles)}")

    # Calibrate tissue threshold using MEMMAP array
    # Re-open memmap in read mode for calibration
    calib_arr = np.memmap(mm_path, dtype=mm_dtype, mode='r', shape=mm_shape)
    
    variance_threshold = calibrate_tissue_threshold(
        tiles, reader=None, x_start=x_start, y_start=y_start, 
        calibration_samples=calibration_samples, 
        block_size=calibration_block_size,
        image_array=calib_arr
    )
    
    del calib_arr # Close calibration handle

    reader.close() # Close reader in main process

    if sample_fraction < 1.0:
        n = max(1, int(len(tiles) * sample_fraction))
        np.random.seed(42)
        tiles = list(np.random.choice(tiles, n, replace=False))
        print(f"Sampling {len(tiles)} tiles for processing")

    # Prepare arguments for worker processes (NO reader, NO large objects)
    worker_args = []
    for tile in tiles:
        worker_args.append((
            tile, czi_path, x_start, y_start, output_dir,
            mk_min_area, mk_max_area, variance_threshold,
            calibration_block_size
        ))
    
    # Process tiles in parallel
    mk_count = 0  # Just count, don't accumulate features (already saved to disk)
    hspc_count = 0
    mk_gid = 1
    hspc_gid = 1

    # Setup GPU distribution
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            # Fill queue with GPU IDs for workers to consume
            for i in range(num_workers):
                gpu_queue.put(i % n_gpus)
    
    # Pass Memmap path to initializer
    # Note: passing path as string is safe for pickling
    init_args = (mk_classifier_path, hspc_classifier_path, gpu_queue, str(mm_path), mm_shape, mm_dtype)
    
    try:
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
                for result in pool.imap_unordered(process_tile_worker, worker_args):
                    pbar.update(1)
                    if result['status'] == 'success':
                        tile = result['tile']
                        tid = result['tid']
                        
                        # Process and save MK results
                        if result['mk_feats']:
                            mk_dir = output_dir / "mk" / "tiles"
                            mk_tile_dir = mk_dir / str(tid)
                            mk_tile_dir.mkdir(parents=True, exist_ok=True)
                            
                            with open(mk_tile_dir / "window.csv", 'w') as f:
                                f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")
                            
                            new_mk = np.zeros_like(result['mk_masks'])
                            mk_tile_cells = []
                            for feat in result['mk_feats']:
                                old_id = int(feat['id'].split('_')[1])
                                new_mk[result['mk_masks'] == old_id] = mk_gid
                                feat['id'] = f'det_{mk_gid - 1}'
                                feat['global_id'] = mk_gid
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                mk_tile_cells.append(mk_gid)
                                mk_count += 1  # Just count, features already saved to disk
                                mk_gid += 1
                            
                            with open(mk_tile_dir / "classes.csv", 'w') as f:
                                for c in mk_tile_cells: f.write(f"{c}\n")
                            with h5py.File(mk_tile_dir / "segmentation.h5", 'w') as f:
                                f.create_dataset('labels', data=new_mk[np.newaxis], compression='gzip')
                            with open(mk_tile_dir / "features.json", 'w') as f:
                                json.dump([{'id': m['id'], 'features': m['features']} for m in result['mk_feats']], f)

                        # Process and save HSPC results
                        if result['hspc_feats']:
                            hspc_dir = output_dir / "hspc" / "tiles"
                            hspc_tile_dir = hspc_dir / str(tid)
                            hspc_tile_dir.mkdir(parents=True, exist_ok=True)
                            
                            with open(hspc_tile_dir / "window.csv", 'w') as f:
                                f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")
                            
                            new_hspc = np.zeros_like(result['hspc_masks'])
                            hspc_tile_cells = []
                            for feat in result['hspc_feats']:
                                old_id = int(feat['id'].split('_')[1])
                                new_hspc[result['hspc_masks'] == old_id] = hspc_gid
                                feat['id'] = f'det_{hspc_gid - 1}'
                                feat['global_id'] = hspc_gid
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                hspc_tile_cells.append(hspc_gid)
                                hspc_count += 1  # Just count, features already saved to disk
                                hspc_gid += 1
                                
                            with open(hspc_tile_dir / "classes.csv", 'w') as f:
                                for c in hspc_tile_cells: f.write(f"{c}\n")
                            with h5py.File(hspc_tile_dir / "segmentation.h5", 'w') as f:
                                f.create_dataset('labels', data=new_hspc[np.newaxis], compression='gzip')
                            with open(hspc_tile_dir / "features.json", 'w') as f:
                                json.dump([{'id': h['id'], 'features': h['features']} for h in result['hspc_feats']], f)

                    elif result['status'] == 'error':
                        print(f"  Tile {result['tid']} error: {result['error']}")

                    # Explicit memory cleanup after processing each result
                    del result
                    import gc
                    if pbar.n % 5 == 0:  # Run GC every 5 tiles
                        gc.collect()
    finally:
        # Always clean up temp memmap directory
        if 'temp_mm_dir' in locals() and temp_mm_dir.exists():
            try:
                shutil.rmtree(temp_mm_dir)
                print(f"Cleaned up temp directory: {temp_mm_dir}")
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_mm_dir}: {e}")

    # Save summaries
    # Get pixel size from reader if possible (re-open temporarily)
    try:
        pixel_size_um = get_pixel_size_from_czi(czi_path)
    except:
        pixel_size_um = None

    summary = {
        'czi_path': str(czi_path),
        'pixel_size_um': pixel_size_um,
        'mk_count': mk_count,
        'hspc_count': hspc_count,
        'feature_count': '22 morphological + 256 SAM2 + 2048 ResNet = 2326'
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"MKs detected: {mk_count}")
    print(f"HSPCs detected: {hspc_count}")
    print(f"Features per cell: 2326 (22 + 256 + 2048)")
    print(f"Output: {output_dir}")
    print(f"  MK tiles: {output_dir}/mk/tiles/")
    print(f"  HSPC tiles: {output_dir}/hspc/tiles/")


def main():
    parser = argparse.ArgumentParser(description='Unified MK + HSPC segmentation')
    parser.add_argument('--czi-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mk-min-area-um', type=float, default=100,
                        help='Minimum MK area in µm² (only applies to MKs)')
    parser.add_argument('--mk-max-area-um', type=float, default=2100,
                        help='Maximum MK area in µm² (only applies to MKs)')
    parser.add_argument('--tile-size', type=int, default=4096)
    parser.add_argument('--overlap', type=int, default=512)
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--calibration-block-size', type=int, default=512,
                        help='Block size for variance calculation in tissue calibration and detection.')
    parser.add_argument('--calibration-samples', type=int, default=50,
                        help='Number of tiles to sample for tissue calibration.')
    parser.add_argument('--mk-classifier', type=str, help='Path to trained MK classifier (.pkl)')
    parser.add_argument('--hspc-classifier', type=str, help='Path to trained HSPC classifier (.pkl)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers for tile processing.')

    args = parser.parse_args()

    # Convert µm² to px² using pixel size (0.1725 µm/px)
    PIXEL_SIZE_UM = 0.1725
    um_to_px_factor = PIXEL_SIZE_UM ** 2  # 0.02975625
    mk_min_area_px = int(args.mk_min_area_um / um_to_px_factor)
    mk_max_area_px = int(args.mk_max_area_um / um_to_px_factor)

    print(f"MK area filter: {args.mk_min_area_um}-{args.mk_max_area_um} µm² = {mk_min_area_px}-{mk_max_area_px} px²")

    run_unified_segmentation(
        args.czi_path, args.output_dir,
        mk_min_area_px, mk_max_area_px,
        args.tile_size, args.overlap, args.sample_fraction,
        calibration_block_size=args.calibration_block_size,
        calibration_samples=args.calibration_samples,
        num_workers=args.num_workers,
        mk_classifier_path=args.mk_classifier,
        hspc_classifier_path=args.hspc_classifier
    )


if __name__ == "__main__":
    main()
