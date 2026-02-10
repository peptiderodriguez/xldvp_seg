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
import random
from pathlib import Path

# Early logging import for module-level logger
from segmentation.utils.logging import get_logger, setup_logging, log_parameters
from segmentation.utils.feature_extraction import extract_resnet_features_batch, preprocess_crop_for_resnet
from segmentation.io.czi_loader import get_loader, CZILoader
from segmentation.io.html_export import (
    create_hdf5_dataset,
    draw_mask_contour,
    percentile_normalize,
    get_largest_connected_component,
    image_to_base64,
    HDF5_COMPRESSION_KWARGS,
    HDF5_COMPRESSION_NAME,
    # MK/HSPC batch HTML export functions (consolidated)
    export_mk_hspc_html_from_ram as export_html_from_ram,
    load_samples_from_ram,
    create_mk_hspc_index as create_export_index,
    generate_mk_hspc_page_html as generate_export_page_html,
    generate_mk_hspc_pages as generate_export_pages,
)
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    compute_variance_threshold,
    compute_tissue_thresholds,
    filter_tissue_tiles,
    has_tissue,
    calculate_block_variances,
)
from segmentation.utils.config import (
    MORPHOLOGICAL_FEATURES_COUNT,
    SAM2_EMBEDDING_DIMENSION,
    RESNET_EMBEDDING_DIMENSION,
    TOTAL_FEATURES_PER_CELL,
    DEFAULT_PIXEL_SIZE_UM,
    RESNET_INFERENCE_BATCH_SIZE,
    CPU_UTILIZATION_FRACTION,
    get_feature_dimensions,
    get_cpu_worker_count,
)
from segmentation.utils.mask_cleanup import cleanup_mask, apply_cleanup_to_detection
from segmentation.preprocessing.stain_normalization import (
    percentile_normalize_rgb,
    compute_global_percentiles,
    normalize_to_percentiles,
    apply_reinhard_normalization,
    extract_slide_norm_params,
)

logger = get_logger(__name__)

# Auto-detect checkpoint directory (local or cluster)
_script_dir = Path(__file__).parent.resolve()
_checkpoint_candidates = [
    _script_dir / "checkpoints",  # Local: same dir as script
    Path.home() / ".cache" / "cellpose",  # User cache
    Path("/ptmp/edrod/MKsegmentation/checkpoints"),  # Cluster path
]
for _cp in _checkpoint_candidates:
    if _cp.exists():
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(_cp)
        break
else:
    # Default to local checkpoints dir (will be created)
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(_script_dir / "checkpoints")

import gc
import numpy as np
import cv2
import h5py
import json
import argparse
from tqdm import tqdm
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import psutil
from skimage.color import rgb2hed


def extract_hematoxylin_channel(rgb_image):
    """Extract hematoxylin (nuclear) channel from H&E stained RGB image.

    Uses color deconvolution to separate Hematoxylin (blue/purple nuclei)
    from Eosin (pink cytoplasm). Returns inverted hematoxylin channel
    as uint8 grayscale image suitable for Cellpose.

    Args:
        rgb_image: RGB image array (H, W, 3), uint8

    Returns:
        Grayscale image (H, W), uint8, with nuclei appearing bright
    """
    # Convert to HED color space (Hematoxylin-Eosin-DAB)
    hed = rgb2hed(rgb_image)

    # Extract hematoxylin channel (index 0)
    # Higher values = more hematoxylin staining
    hematoxylin = hed[:, :, 0]

    # Normalize to 0-255 range and invert so nuclei are bright
    # (Cellpose expects bright objects on dark background)
    h_min, h_max = hematoxylin.min(), hematoxylin.max()
    if h_max > h_min:
        hematoxylin_norm = (hematoxylin - h_min) / (h_max - h_min)
    else:
        hematoxylin_norm = np.zeros_like(hematoxylin)

    # Invert: high hematoxylin (nuclei) becomes bright
    hematoxylin_inv = (hematoxylin_norm * 255).astype(np.uint8)

    return hematoxylin_inv


# =============================================================================
# HTML EXPORT FUNCTIONS
# =============================================================================
# Note: HTML export functions (load_samples_from_ram, create_export_index,
# generate_export_page_html, generate_export_pages, export_html_from_ram)
# are now consolidated in segmentation.io.html_export and imported above.


# Global variables for worker process
segmenter = None
shared_image = None

# Global mask cleanup configuration (set via init_worker_cleanup_config)
cleanup_config = {
    'cleanup_masks': False,
    'fill_holes': True,
    'pixel_size_um': 0.1725,
    'hspc_nuclear_only': False,
}


def set_cleanup_config(cleanup_masks=False, fill_holes=True, pixel_size_um=0.1725, hspc_nuclear_only=False):
    """Set global cleanup configuration (call in main process before spawning workers)."""
    global cleanup_config
    cleanup_config = {
        'cleanup_masks': cleanup_masks,
        'fill_holes': fill_holes,
        'pixel_size_um': pixel_size_um,
        'hspc_nuclear_only': hspc_nuclear_only,
    }


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
        except Exception as e:
            # Fallback: simple modulo or default
            logger.debug(f"GPU queue timeout, using fallback assignment: {e}")
            n_gpus = torch.cuda.device_count()
            if n_gpus > 0:
                gpu_id = mp.current_process().pid % n_gpus
                device = f"cuda:{gpu_id}"
            else:
                device = "cuda"
    
    logger.info(f"Worker {mp.current_process().pid} initialized on {device}")
    
    # Initialize Segmenter
    segmenter = UnifiedSegmenter(
        mk_classifier_path=mk_classifier_path,
        hspc_classifier_path=hspc_classifier_path,
        device=device
    )

    # Attach to Memory Map (Read-Only)
    try:
        shared_image = np.memmap(mm_path, dtype=mm_dtype, mode='r', shape=mm_shape)
        logger.info(f"Worker {mp.current_process().pid} attached to memmap: {mm_path}")
    except Exception as e:
        logger.error(f"Worker {mp.current_process().pid} FAILED to attach to memmap: {e}")
        shared_image = None

def process_tile_worker(args):
    """
    Worker function for processing a single tile in a separate process.
    """
    # Unpack all arguments
    tile, _, _, _, output_dir, \
    mk_min_area, mk_max_area, hspc_min_area, hspc_max_area, variance_threshold, \
    calibration_block_size, intensity_threshold = args

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

    # Normalization disabled for H&E images - raw pixel values work better
    # img_rgb = percentile_normalize(img_rgb, p_low=5, p_high=95)
    has_tissue_content, _ = has_tissue(img_rgb, variance_threshold, block_size=calibration_block_size, intensity_threshold=intensity_threshold, modality='brightfield')
    if not has_tissue_content:
        return {'tid': tid, 'status': 'no_tissue'}

    try:
        mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
            img_rgb, mk_min_area, mk_max_area, hspc_min_area, hspc_max_area,
            hspc_nuclear_only=cleanup_config.get('hspc_nuclear_only', False),
            cleanup_masks=cleanup_config.get('cleanup_masks', False),
            fill_holes=cleanup_config.get('fill_holes', True),
            pixel_size_um=cleanup_config.get('pixel_size_um', 0.1725)
        )

        # Generate crops for each detection (for HTML export without reloading CZI)
        # Optionally apply mask cleanup if enabled in global config
        for feat in mk_feats:
            _, crop_result = process_detection_with_cleanup(
                feat, mk_masks, img_rgb, 'mk',
                cleanup_masks=cleanup_config['cleanup_masks'],
                fill_holes=cleanup_config['fill_holes'],
                pixel_size_um=cleanup_config['pixel_size_um'],
            )
            if crop_result:
                feat['crop_b64'] = crop_result['crop']
                feat['mask_b64'] = crop_result['mask']

        for feat in hspc_feats:
            _, crop_result = process_detection_with_cleanup(
                feat, hspc_masks, img_rgb, 'hspc',
                cleanup_masks=cleanup_config['cleanup_masks'],
                fill_holes=cleanup_config['fill_holes'],
                pixel_size_um=cleanup_config['pixel_size_um'],
            )
            if crop_result:
                feat['crop_b64'] = crop_result['crop']
                feat['mask_b64'] = crop_result['mask']

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
        logger.info("[ROCm FIX] Patched sam2.utils.amg.mask_to_rle_pytorch for INT_MAX workaround")
    except ImportError as e:
        logger.info(f"[ROCm FIX] Could not apply patch: {e}")

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


# Note: percentile_normalize is imported from segmentation.io.html_export
# Note: calibrate_tissue_threshold, filter_tissue_tiles, has_tissue, and
#       calculate_block_variances are imported from segmentation.detection.tissue


def generate_detection_crop(img_rgb, mask, centroid, crop_size=300, display_size=250, padding_fraction=0.3):
    """
    Generate a crop image for a single detection.

    Uses dynamic crop size based on mask bounding box to ensure 100% mask coverage.

    Args:
        img_rgb: RGB tile image (already normalized)
        mask: Binary mask for this detection
        centroid: (x, y) center of detection within tile (fallback if mask empty)
        crop_size: Minimum crop size (used if mask is smaller)
        display_size: Size to resize crop to
        padding_fraction: Padding around mask bbox as fraction of bbox size (default 0.3 = 30%)

    Returns:
        Base64-encoded JPEG string of the crop with mask contour
    """
    import cv2

    h, w = img_rgb.shape[:2]

    # Calculate bounding box from mask
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        bbox_y1, bbox_y2 = y_indices[0], y_indices[-1] + 1
        bbox_x1, bbox_x2 = x_indices[0], x_indices[-1] + 1

        bbox_h = bbox_y2 - bbox_y1
        bbox_w = bbox_x2 - bbox_x1
        bbox_center_x = (bbox_x1 + bbox_x2) // 2
        bbox_center_y = (bbox_y1 + bbox_y2) // 2

        # Dynamic crop size: max dimension + padding, but at least crop_size
        max_dim = max(bbox_h, bbox_w)
        padding = int(max_dim * padding_fraction)
        dynamic_size = max(crop_size, max_dim + 2 * padding)

        # Use bbox center for cropping
        cx, cy = bbox_center_x, bbox_center_y
    else:
        # Fallback to centroid if mask is empty
        cx, cy = int(centroid[0]), int(centroid[1])
        dynamic_size = crop_size

    half = dynamic_size // 2

    # Calculate crop bounds
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    if x2 <= x1 or y2 <= y1:
        return None

    # Extract crop
    crop = img_rgb[y1:y2, x1:x2].copy()

    # Extract corresponding mask region
    mask_crop = mask[y1:y2, x1:x2]

    # Resize to display_size (square)
    if crop.shape[0] != display_size or crop.shape[1] != display_size:
        crop = cv2.resize(crop, (display_size, display_size))
        mask_crop = cv2.resize(mask_crop.astype(np.uint8), (display_size, display_size), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Save raw crop and mask separately - outline drawn at HTML generation time
    crop_b64, _ = image_to_base64(crop)

    # Encode mask as compact PNG (1-bit effectively)
    mask_img = (mask_crop.astype(np.uint8) * 255)
    mask_b64, _ = image_to_base64(mask_img, format='PNG')

    return {'crop': crop_b64, 'mask': mask_b64}


def process_detection_with_cleanup(
    feat, masks_array, img_rgb, cell_type,
    cleanup_masks=False, fill_holes=True, pixel_size_um=0.1725
):
    """
    Process a single detection: optionally cleanup mask, generate crop.

    Args:
        feat: Feature dict for the detection
        masks_array: Label array containing all masks for this cell type
        img_rgb: RGB tile image
        cell_type: 'mk' or 'hspc' (for logging)
        cleanup_masks: If True, apply mask cleanup (largest component + fill holes)
        fill_holes: If True, fill internal holes (ignored if cleanup_masks=False)
        pixel_size_um: Pixel size for area calculation

    Returns:
        Tuple of (cleaned_mask, crop_result) or (original_mask, crop_result)
    """
    det_id = int(feat['id'].split('_')[1])
    mask = (masks_array == det_id)

    if cleanup_masks and mask.any():
        # Apply cleanup and update mask array + features in place (recomputes ALL features)
        mask = apply_cleanup_to_detection(
            mask, feat, masks_array, det_id,
            pixel_size_um=pixel_size_um,
            keep_largest=True,
            fill_internal_holes=fill_holes,
            max_hole_area_fraction=0.5,
            image=img_rgb,  # Pass image to recompute ALL morphological features
        )

    # Get centroid (may have been updated by cleanup)
    centroid = feat.get('center', [img_rgb.shape[1]//2, img_rgb.shape[0]//2])
    crop_result = generate_detection_crop(img_rgb, mask, centroid)

    return mask, crop_result


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

    # HSV features (vectorized for speed)
    if image.ndim == 3:
        from segmentation.utils.feature_extraction import compute_hsv_features
        hsv_feats = compute_hsv_features(masked_pixels, sample_size=100)
        hue_mean = hsv_feats['hue_mean']
        sat_mean = hsv_feats['saturation_mean']
        val_mean = hsv_feats['value_mean']
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

        # Find SAM2 checkpoint (auto-detect local or cluster)
        script_dir = Path(__file__).parent.resolve()
        sam2_candidates = [
            script_dir / sam2_checkpoint,  # Local: same dir as script
            script_dir / "checkpoints" / Path(sam2_checkpoint).name,  # Local checkpoints subdir
            Path("/ptmp/edrod/MKsegmentation") / sam2_checkpoint,  # Cluster path
        ]
        checkpoint_path = None
        for cp in sam2_candidates:
            if cp.exists():
                checkpoint_path = cp
                break
        if checkpoint_path is None:
            # Default to local path (will fail with helpful error if missing)
            checkpoint_path = script_dir / "checkpoints" / Path(sam2_checkpoint).name

        logger.info(f"Loading SAM2 from {checkpoint_path}...")
        sam2_model = build_sam2(sam2_config, str(checkpoint_path), device=self.device)

        # SAM2 for auto mask generation (MKs)
        self.sam2_auto = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=24, # Changed from 32 to 24
            pred_iou_thresh=0.5,  # Stricter filtering for speed
            stability_score_thresh=0.4,
            min_mask_region_area=500,
            crop_n_layers=1 # Added crop_n_layers
        )

        # SAM2 predictor for point prompts (HSPCs)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Cellpose-SAM for HSPC detection (v4+ with SAM backbone)
        logger.info(f"Loading Cellpose-SAM model on {self.device}...")
        self.cellpose = CellposeModel(pretrained_model='cpsam', gpu=True, device=self.device)

        # ResNet for deep features
        logger.info("Loading ResNet-50...")
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
            logger.info(f"Loading MK classifier: {mk_classifier_path}")
            import joblib
            clf_data = joblib.load(mk_classifier_path)
            self.mk_classifier = clf_data['classifier']
            self.mk_feature_names = clf_data['feature_names']
            logger.info(f"  Features: {len(self.mk_feature_names)}, Trained on {clf_data.get('n_samples', '?')} samples")

        if hspc_classifier_path:
            logger.info(f"Loading HSPC classifier: {hspc_classifier_path}")
            import joblib
            clf_data = joblib.load(hspc_classifier_path)
            self.hspc_classifier = clf_data['classifier']
            self.hspc_feature_names = clf_data['feature_names']
            logger.info(f"  Features: {len(self.hspc_feature_names)}, Trained on {clf_data.get('n_samples', '?')} samples")

        logger.info("Models loaded.")

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

    def extract_resnet_features_batch(self, crops, batch_size=16):
        """
        Extract ResNet features for multiple crops in batches for GPU efficiency.

        This method significantly improves GPU utilization by processing multiple
        crops at once instead of one at a time.

        Args:
            crops: List of image crops as numpy arrays
            batch_size: Number of crops to process at once (default 16)

        Returns:
            List of feature vectors as numpy arrays (2048D each)
        """
        return extract_resnet_features_batch(
            crops,
            self.resnet,
            self.resnet_transform,
            self.device,
            batch_size=batch_size
        )

    def extract_sam2_embedding(self, cy, cx):
        """Extract 256D SAM2 embedding at location.

        Args:
            cy: Y coordinate (row) in image space
            cx: X coordinate (col) in image space

        Returns:
            256D SAM2 embedding vector
        """
        try:
            # Get feature map shape and image dimensions for proper scaling
            # SAM2 features are stored as [batch, channels, H, W]
            if hasattr(self.sam2_predictor, '_features'):
                features = self.sam2_predictor._features
                if isinstance(features, dict) and "image_embed" in features:
                    features = features["image_embed"]
                shape = features.shape
                emb_h, emb_w = shape[2], shape[3]

                # Get original image dimensions for proper coordinate mapping
                if hasattr(self.sam2_predictor, '_orig_hw'):
                    img_h, img_w = self.sam2_predictor._orig_hw
                else:
                    # Fallback to assuming 16x downsampling
                    img_h, img_w = emb_h * 16, emb_w * 16

                # Map image coordinates to embedding coordinates
                emb_y = int(cy / img_h * emb_h)
                emb_x = int(cx / img_w * emb_w)
                emb_y = min(max(0, emb_y), emb_h - 1)
                emb_x = min(max(0, emb_x), emb_w - 1)

                return features[0, :, emb_y, emb_x].cpu().numpy()
            return np.zeros(256)
        except Exception as e:
            logger.debug(f"Failed to extract SAM2 embedding: {e}")
            return np.zeros(256)

    def process_tile(
        self,
        image_rgb,
        mk_min_area=1000,
        mk_max_area=100000,
        hspc_min_area=None,
        hspc_max_area=None,
        resnet_batch_size=16,
        hspc_nuclear_only=False,
        cleanup_masks=False,
        fill_holes=True,
        pixel_size_um=0.1725
    ):
        """Process a single tile for both MKs and HSPCs.

        Uses batch ResNet feature extraction for improved GPU utilization.

        Args:
            image_rgb: RGB image array
            mk_min_area: Minimum MK area in pixels
            mk_max_area: Maximum MK area in pixels
            hspc_min_area: Minimum HSPC area in pixels (None = no filter)
            hspc_max_area: Maximum HSPC area in pixels (None = no filter)
            resnet_batch_size: Batch size for ResNet feature extraction (default 16)
            hspc_nuclear_only: If True, use H&E deconvolution to extract hematoxylin
                              (nuclear) channel for HSPC detection (reduces false positives)
            cleanup_masks: If True, apply mask cleanup (largest component + fill holes) BEFORE feature extraction
            fill_holes: If True, fill internal holes during cleanup
            pixel_size_um: Pixel size for area calculations

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
        gc.collect()

        mk_masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        mk_valid_detections = []  # Collect valid detections for batch processing
        mk_id = 1

        # First pass: filter overlaps and collect valid MK detections
        for result in valid_results:
            mask = result['segmentation']
            # Ensure boolean type for indexing (critical for NVIDIA CUDA compatibility)
            if mask.dtype != bool:
                mask = (mask > 0.5).astype(bool)

            # Check overlap with existing masks - skip if >50% overlaps (larger already added)
            if mk_masks.max() > 0:
                overlap = ((mask > 0) & (mk_masks > 0)).sum()
                if overlap > 0.5 * mask.sum():
                    continue

            # Add to label array
            mk_masks[mask] = mk_id

            # Get centroid
            cy, cx = ndimage.center_of_mass(mask)

            mk_valid_detections.append({
                'id': mk_id,
                'mask': mask,
                'cy': cy,
                'cx': cx,
                'sam2_iou': float(result.get('predicted_iou', 0)),
                'sam2_stability': float(result.get('stability_score', 0))
            })
            mk_id += 1

        # Delete valid_results to free memory (large mask arrays)
        del valid_results
        gc.collect()

        # Batch feature extraction for MKs
        mk_features = []
        if mk_valid_detections:
            # Extract morphological and SAM2 features, collect crops
            mk_crops = []
            mk_crop_indices = []

            for idx, det in enumerate(mk_valid_detections):
                mask = det['mask']
                cy, cx = det['cy'], det['cx']

                # Apply mask cleanup BEFORE feature extraction (if enabled)
                if cleanup_masks:
                    mask = cleanup_mask(mask, keep_largest=True, fill_internal_holes=fill_holes, max_hole_area_fraction=0.5)
                    # Update mask in detection dict and label array
                    det['mask'] = mask
                    mk_masks[mk_masks == det['id']] = 0  # Clear old
                    mk_masks[mask] = det['id']  # Set cleaned
                    # Recompute centroid
                    if mask.any():
                        cy, cx = ndimage.center_of_mass(mask)
                        det['cy'], det['cx'] = cy, cx

                # Extract morphological features (from cleaned mask if cleanup enabled)
                morph = extract_morphological_features(mask, image_rgb)

                # SAM2 embeddings
                sam2_emb = self.extract_sam2_embedding(cy, cx)
                for i, v in enumerate(sam2_emb):
                    morph[f'sam2_{i}'] = float(v)

                det['morph'] = morph

                # Prepare crop for batch ResNet processing
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()
                    crop = image_rgb[y1:y2+1, x1:x2+1].copy()
                    crop_mask = mask[y1:y2+1, x1:x2+1]
                    crop[~crop_mask] = 0
                    mk_crops.append(crop)
                    mk_crop_indices.append(idx)

            # Batch ResNet feature extraction
            if mk_crops:
                resnet_features_list = self.extract_resnet_features_batch(mk_crops, batch_size=resnet_batch_size)

                # Assign ResNet features to correct detections
                for crop_idx, resnet_feats in zip(mk_crop_indices, resnet_features_list):
                    for i, v in enumerate(resnet_feats):
                        mk_valid_detections[crop_idx]['morph'][f'resnet_{i}'] = float(v)

            # Fill zeros for detections without valid crops
            # Fill missing ResNet features with zeros (batch initialization)
            for det in mk_valid_detections:
                if 'resnet_0' not in det['morph']:
                    det['morph'].update({f'resnet_{i}': 0.0 for i in range(2048)})

            # Build final MK features list
            for det in mk_valid_detections:
                mk_features.append({
                    'id': f'det_{det["id"]}',
                    'global_id': None,  # Will be set in save_tile_results
                    'uid': None,  # Will be set in save_tile_results
                    'center': [float(det['cx']), float(det['cy'])],
                    'sam2_iou': det['sam2_iou'],
                    'sam2_stability': det['sam2_stability'],
                    'features': det['morph']
                })

        torch.cuda.empty_cache()  # Clear GPU cache after MK processing

        # ============================================
        # HSPC Detection: Cellpose-SAM + SAM2 refinement
        # ============================================
        # Now set image for SAM2 predictor (after auto generator is done)
        # This ensures we only hold ONE predictor's embeddings at a time
        self.sam2_predictor.set_image(image_rgb)

        # Prepare image for Cellpose HSPC detection
        if hspc_nuclear_only:
            # Use H&E deconvolution to extract hematoxylin (nuclear) channel
            # This reduces false positives from non-nuclear structures
            hematoxylin = extract_hematoxylin_channel(image_rgb)
            # Cellpose expects (H, W) or (H, W, C) - pass grayscale directly
            cellpose_input = hematoxylin
            cellpose_channels = [0, 0]  # Grayscale mode
        else:
            # Original behavior: use full RGB with grayscale conversion
            cellpose_input = image_rgb
            cellpose_channels = [0, 0]

        # Cellpose with grayscale mode, fixed diameter=30 for small HSPCs
        # diameter=30 targets small cells (~20-50 µm²), prevents auto-detection of large merged regions
        cellpose_masks, _, _ = self.cellpose.eval(cellpose_input, channels=cellpose_channels, diameter=30)

        # Get Cellpose centroids and limit to top 500 by mask area
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
            # Ensure boolean type for indexing (critical for NVIDIA CUDA compatibility)
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)
            sam2_score = float(scores[best_idx])

            if sam2_mask.sum() < 10:
                continue

            # Keep all candidates - overlap filtering happens after sorting by confidence
            hspc_candidates.append({
                'mask': sam2_mask,
                'score': sam2_score,
                'center': (cx, cy),
                'cp_id': cp_id
            })

        # Sort by SAM2 score (most confident first) and handle overlaps
        hspc_candidates.sort(key=lambda x: x['score'], reverse=True)

        hspc_masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        hspc_valid_detections = []  # Collect valid detections for batch processing
        hspc_id = 1

        # First pass: filter overlaps and collect valid HSPC detections
        for cand in hspc_candidates:
            sam2_mask = cand['mask']
            # Ensure boolean type for indexing (critical for NVIDIA CUDA compatibility)
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)
            sam2_score = cand['score']
            cx, cy = cand['center']

            # HSPC size filter (if enabled)
            mask_area = sam2_mask.sum()
            if hspc_min_area is not None and mask_area < hspc_min_area:
                continue
            if hspc_max_area is not None and mask_area > hspc_max_area:
                continue

            # Check overlap with existing HSPC masks - skip if >50% overlaps
            if hspc_masks.max() > 0:
                overlap = ((sam2_mask > 0) & (hspc_masks > 0)).sum()
                if overlap > 0.5 * sam2_mask.sum():
                    continue

            # Add to label array
            hspc_masks[sam2_mask] = hspc_id

            hspc_valid_detections.append({
                'id': hspc_id,
                'mask': sam2_mask,
                'cy': cy,
                'cx': cx,
                'sam2_score': sam2_score,
                'cp_id': int(cand['cp_id'])
            })
            hspc_id += 1

        # Delete large temporary arrays to free memory
        del hspc_candidates
        del cellpose_masks
        gc.collect()

        # Batch feature extraction for HSPCs
        hspc_features = []
        if hspc_valid_detections:
            # Extract morphological and SAM2 features, collect crops
            hspc_crops = []
            hspc_crop_indices = []

            for idx, det in enumerate(hspc_valid_detections):
                mask = det['mask']
                cy, cx = det['cy'], det['cx']

                # Apply mask cleanup BEFORE feature extraction (if enabled)
                if cleanup_masks:
                    mask = cleanup_mask(mask, keep_largest=True, fill_internal_holes=fill_holes, max_hole_area_fraction=0.5)
                    # Update mask in detection dict and label array
                    det['mask'] = mask
                    hspc_masks[hspc_masks == det['id']] = 0  # Clear old
                    hspc_masks[mask] = det['id']  # Set cleaned
                    # Recompute centroid for SAM2 embedding extraction
                    if mask.any():
                        cy, cx = ndimage.center_of_mass(mask)
                        det['cy'], det['cx'] = cy, cx

                # Extract morphological features (from cleaned mask if cleanup enabled)
                morph = extract_morphological_features(mask, image_rgb)

                # SAM2 embeddings
                sam2_emb = self.extract_sam2_embedding(cy, cx)
                for i, v in enumerate(sam2_emb):
                    morph[f'sam2_{i}'] = float(v)

                det['morph'] = morph

                # Prepare crop for batch ResNet processing
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    y1, y2 = ys.min(), ys.max()
                    x1, x2 = xs.min(), xs.max()
                    crop = image_rgb[y1:y2+1, x1:x2+1].copy()
                    crop_mask = mask[y1:y2+1, x1:x2+1]
                    crop[~crop_mask] = 0
                    hspc_crops.append(crop)
                    hspc_crop_indices.append(idx)

            # Batch ResNet feature extraction
            if hspc_crops:
                resnet_features_list = self.extract_resnet_features_batch(hspc_crops, batch_size=resnet_batch_size)

                # Assign ResNet features to correct detections
                for crop_idx, resnet_feats in zip(hspc_crop_indices, resnet_features_list):
                    for i, v in enumerate(resnet_feats):
                        hspc_valid_detections[crop_idx]['morph'][f'resnet_{i}'] = float(v)

            # Fill zeros for detections without valid crops
            # Fill missing ResNet features with zeros (batch initialization)
            for det in hspc_valid_detections:
                if 'resnet_0' not in det['morph']:
                    det['morph'].update({f'resnet_{i}': 0.0 for i in range(2048)})

            # Build final HSPC features list
            for det in hspc_valid_detections:
                hspc_features.append({
                    'id': f'det_{det["id"]}',
                    'global_id': None,  # Will be set in save_tile_results
                    'uid': None,  # Will be set in save_tile_results
                    'center': [float(det['cx']), float(det['cy'])],
                    'cellpose_id': det['cp_id'],
                    'sam2_score': det['sam2_score'],
                    'features': det['morph']
                })

        # Clear SAM2 cached features after processing this tile
        self.sam2_predictor.reset_predictor()
        torch.cuda.empty_cache()

        return mk_masks, hspc_masks, mk_features, hspc_features


def shared_calibrate_tissue_threshold(tiles, image_array, calibration_samples, block_size, tile_size,
                                      modality=None, precomputed_thresholds=None, slide_name=None):
    """Calibrate tissue detection thresholds using K-means (variance) and Otsu (intensity).

    Args:
        tiles: List of tile dictionaries with x, y, w, h keys
        image_array: The full slide image array (loaded to RAM)
        calibration_samples: Number of tiles to sample for calibration
        block_size: Block size for variance calculation
        tile_size: Expected tile size
        modality: 'brightfield' for H&E (÷3 variance + pixel Otsu), None for default
        precomputed_thresholds: Dict mapping slide_name -> {variance_threshold, intensity_threshold}
            from step 1 JSON. If provided and slide_name found, skips calibration.
        slide_name: Name of the current slide (for looking up precomputed thresholds)

    Returns:
        tuple: (variance_threshold, intensity_threshold)
    """
    # Use precomputed thresholds from step 1 if available
    if precomputed_thresholds and slide_name:
        precomputed = precomputed_thresholds.get(slide_name)
        if precomputed is not None:
            otsu_thresh = precomputed.get('otsu_threshold',
                                          precomputed.get('intensity_threshold', 220.0))
            logger.info(f"Using pre-computed tissue thresholds from step 1 for '{slide_name}':")
            logger.info(f"  otsu_threshold={otsu_thresh:.1f} (variance_threshold=0, unused for brightfield)")
            return 0.0, otsu_thresh

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Sample tiles for calibration
    calib_count = min(calibration_samples, len(tiles))
    calib_tiles = random.sample(tiles, calib_count)
    logger.info(f"Calibrating tissue threshold using {calib_count} tiles...")

    def calc_tile_stats(tile):
        tile_img = image_array[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]

        if tile_img.max() == 0:
            return [0.0], [255.0], np.array([], dtype=np.uint8)

        if tile_img.ndim == 3:
            gray = cv2.cvtColor(tile_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = tile_img.astype(np.uint8) if tile_img.dtype != np.uint8 else tile_img

        block_vars, block_means = calculate_block_variances(gray, block_size)

        # Collect pixel samples for brightfield Otsu
        pixel_sample = np.array([], dtype=np.uint8)
        if modality == 'brightfield':
            flat = gray.ravel()
            n = min(2000, len(flat))
            pixel_sample = flat[np.random.choice(len(flat), n, replace=False)]

        return (block_vars if block_vars else [],
                block_means if block_means else [],
                pixel_sample)

    all_variances = []
    all_means = []
    all_pixel_samples = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(calc_tile_stats, tile): tile for tile in calib_tiles}
        for future in tqdm(as_completed(futures), total=len(calib_tiles), desc="Calibrating"):
            block_vars, block_means, pixel_sample = future.result()
            all_variances.extend(block_vars)
            all_means.extend(block_means)
            if len(pixel_sample) > 0:
                all_pixel_samples.append(pixel_sample)

    variances = np.array(all_variances)
    means = np.array(all_means)
    pixel_samples = np.concatenate(all_pixel_samples) if all_pixel_samples else None
    if len(variances) == 0:
        logger.warning("No variance samples collected, using defaults")
        return 50.0, 220.0

    logger.info(f"  Running K-means + Otsu on {len(variances)} block samples...")
    return compute_tissue_thresholds(variances, means, default_var=50.0,
                                     modality=modality, pixel_samples=pixel_samples)


def run_unified_segmentation(
    czi_path,
    output_dir,
    mk_min_area=1000,
    mk_max_area=100000,
    hspc_min_area=None,
    hspc_max_area=None,
    tile_size=4096,
    overlap=512,
    sample_fraction=1.0,
    calibration_block_size=512,
    calibration_samples=50,
    num_workers=4,
    mk_classifier_path=None,
    hspc_classifier_path=None,
    channel=0,
    normalization_method='none',
    norm_params_file=None
):
    """Run unified MK + HSPC segmentation with multiprocessing.

    Uses RAM-first architecture: loads entire channel into RAM once,
    then all tile processing references that RAM array.
    """
    import shutil

    # Set start method to 'spawn' for GPU safety
    if num_workers > 0:
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # Already set, verify it's 'spawn'
            if mp.get_start_method() != 'spawn':
                logger.warning(f"Multiprocessing start method already set to '{mp.get_start_method()}', expected 'spawn'")

    czi_path = Path(czi_path)
    slide_name = czi_path.stem  # Extract slide name for UID generation
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info("UNIFIED SEGMENTATION: MK + HSPC (RAM-FIRST ARCHITECTURE)")
    logger.info(f"{'='*70}")
    logger.info(f"CZI: {czi_path}")

    # Load channel into RAM using get_loader (uses global cache)
    logger.info(f"Loading channel {channel} into RAM...")
    loader = get_loader(czi_path, load_to_ram=True, channel=channel)

    x_start, y_start = loader.mosaic_origin
    full_width, full_height = loader.mosaic_size

    logger.info(f"Image: {full_width} x {full_height}")
    img_size_gb = loader.channel_data.nbytes / (1024**3)
    logger.info(f"Channel data loaded to RAM ({img_size_gb:.2f} GB)")

    # Load normalization params and per-slide tissue thresholds from JSON
    precomputed_thresholds = None
    if normalization_method == 'reinhard' and norm_params_file:
        logger.info(f"\n{'='*70}")
        logger.info("APPLYING SLIDE-LEVEL NORMALIZATION")
        logger.info(f"{'='*70}")
        logger.info(f"Loading parameters from: {norm_params_file}")

        with open(norm_params_file, 'r') as f:
            reinhard_params = json.load(f)

        # Extract per-slide tissue thresholds from step 1 (if present)
        precomputed_thresholds = reinhard_params.get('tissue_thresholds')
        if precomputed_thresholds:
            logger.info(f"  Found pre-computed tissue thresholds for {len(precomputed_thresholds)} slides")
        else:
            logger.warning(f"  No tissue_thresholds in params file — will fall back to recalibration")

        # Detect median/MAD vs mean/std
        if 'L_median' in reinhard_params and 'L_mad' in reinhard_params:
            from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN
            logger.info(f"  Method: MEDIAN/MAD (robust)")
            logger.info(f"  L: median={reinhard_params['L_median']:.2f}, MAD={reinhard_params['L_mad']:.2f}")
            logger.info(f"  a: median={reinhard_params['a_median']:.2f}, MAD={reinhard_params['a_mad']:.2f}")
            logger.info(f"  b: median={reinhard_params['b_median']:.2f}, MAD={reinhard_params['b_mad']:.2f}")
            logger.info(f"  Normalizing entire slide...")

            slide_thresh = precomputed_thresholds.get(slide_name) if precomputed_thresholds else None
            _otsu, _slab = extract_slide_norm_params(slide_thresh)
            channel_data_normalized = apply_reinhard_normalization_MEDIAN(
                loader.channel_data,
                reinhard_params,
                otsu_threshold=_otsu,
                slide_lab_stats=_slab,
            )
            loader.channel_data = channel_data_normalized
            del channel_data_normalized
            gc.collect()
            logger.info(f"  Slide normalized successfully")
        else:
            logger.info(f"  Method: MEAN/STD (classic)")
            logger.info(f"  L: mean={reinhard_params['L_mean']:.2f}, std={reinhard_params['L_std']:.2f}")
            logger.info(f"  a: mean={reinhard_params['a_mean']:.2f}, std={reinhard_params['a_std']:.2f}")
            logger.info(f"  b: mean={reinhard_params['b_mean']:.2f}, std={reinhard_params['b_std']:.2f}")
            logger.info(f"  Normalizing entire slide...")

            slide_thresh = precomputed_thresholds.get(slide_name) if precomputed_thresholds else None
            _otsu, _slab = extract_slide_norm_params(slide_thresh)
            channel_data_normalized = apply_reinhard_normalization(
                loader.channel_data,
                reinhard_params,
                otsu_threshold=_otsu,
                slide_lab_stats=_slab,
            )
            loader.channel_data = channel_data_normalized
            del channel_data_normalized
            gc.collect()
            logger.info(f"  Slide normalized successfully")

    # Create tiles (coordinates relative to mosaic origin)
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

    logger.info(f"Total tiles: {len(tiles)}")

    # Calibrate tissue threshold using RAM array directly
    # (will use precomputed thresholds from step 1 if available)
    variance_threshold, intensity_threshold = shared_calibrate_tissue_threshold(
        tiles=tiles,
        image_array=loader.channel_data,
        calibration_samples=calibration_samples,
        block_size=calibration_block_size,
        tile_size=tile_size,
        modality='brightfield',
        precomputed_thresholds=precomputed_thresholds,
        slide_name=slide_name,
    )

    # Create memmap for worker processes (they can't share numpy arrays directly)
    # Use /dev/shm if available for RAM-backed storage
    shm_path = Path("/dev/shm")
    use_ramdisk = False
    if shm_path.exists():
        try:
            shm_stat = os.statvfs("/dev/shm")
            shm_free_gb = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024**3)
            if shm_free_gb > img_size_gb * 1.5:
                use_ramdisk = True
                logger.info(f"  Using RAM-backed storage (/dev/shm): {shm_free_gb:.1f} GB free")
        except Exception as e:
            logger.debug(f"Could not check /dev/shm availability: {e}")

    if use_ramdisk:
        temp_mm_dir = shm_path / f"mkseg_{os.getpid()}"
    else:
        temp_mm_dir = output_dir / "temp_mm"
        logger.info(f"  Using disk-backed storage (fallback)")

    temp_mm_dir.mkdir(parents=True, exist_ok=True)
    mm_path = temp_mm_dir / "image.dat"

    # Create memmap and copy from loader's RAM array
    mm_shape = loader.channel_data.shape
    mm_dtype = loader.channel_data.dtype
    shm_arr = np.memmap(mm_path, dtype=mm_dtype, mode='w+', shape=mm_shape)
    np.copyto(shm_arr, loader.channel_data)
    shm_arr.flush()
    del shm_arr

    storage_type = "RAM" if use_ramdisk else "disk"
    logger.info(f"Memmap created for workers: {mm_path} ({storage_type})")

    # Clear loader's RAM (memmap now holds the data for workers)
    loader.close()

    if sample_fraction < 1.0:
        n = max(1, int(len(tiles) * sample_fraction))
        np.random.seed(42)
        tiles = list(np.random.choice(tiles, n, replace=False))
        logger.info(f"Sampling {len(tiles)} tiles for processing")

    # Prepare arguments for worker processes (NO reader, NO large objects)
    worker_args = []
    for tile in tiles:
        worker_args.append((
            tile, czi_path, x_start, y_start, output_dir,
            mk_min_area, mk_max_area, hspc_min_area, hspc_max_area, variance_threshold,
            calibration_block_size, intensity_threshold
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
                                feat['global_id'] = mk_gid  # Keep for backwards compatibility
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
                                feat['uid'] = f"{slide_name}_mk_{round(feat['center'][0])}_{round(feat['center'][1])}"
                                mk_tile_cells.append(mk_gid)
                                mk_count += 1  # Just count, features already saved to disk
                                mk_gid += 1

                            with open(mk_tile_dir / "classes.csv", 'w') as f:
                                for c in mk_tile_cells: f.write(f"{c}\n")
                            with h5py.File(mk_tile_dir / "segmentation.h5", 'w') as f:
                                create_hdf5_dataset(f, 'labels', new_mk[np.newaxis])
                            with open(mk_tile_dir / "features.json", 'w') as f:
                                json.dump(result['mk_feats'], f)

                            # Explicit cleanup to prevent memory accumulation
                            del new_mk, mk_tile_cells

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
                                feat['global_id'] = hspc_gid  # Keep for backwards compatibility
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
                                feat['uid'] = f"{slide_name}_hspc_{round(feat['center'][0])}_{round(feat['center'][1])}"
                                hspc_tile_cells.append(hspc_gid)
                                hspc_count += 1  # Just count, features already saved to disk
                                hspc_gid += 1

                            with open(hspc_tile_dir / "classes.csv", 'w') as f:
                                for c in hspc_tile_cells: f.write(f"{c}\n")
                            with h5py.File(hspc_tile_dir / "segmentation.h5", 'w') as f:
                                create_hdf5_dataset(f, 'labels', new_hspc[np.newaxis])
                            with open(hspc_tile_dir / "features.json", 'w') as f:
                                json.dump(result['hspc_feats'], f)

                            # Explicit cleanup to prevent memory accumulation
                            del new_hspc, hspc_tile_cells

                    elif result['status'] == 'error':
                        logger.error(f"Tile {result['tid']} error: {result['error']}")

                    # Explicit memory cleanup after processing each result
                    # Delete large mask arrays from result before del result
                    if 'mk_masks' in result:
                        del result['mk_masks']
                    if 'hspc_masks' in result:
                        del result['hspc_masks']
                    if 'mk_feats' in result:
                        del result['mk_feats']
                    if 'hspc_feats' in result:
                        del result['hspc_feats']
                    del result

                    # Run GC every tile to prevent accumulation
                    gc.collect()
    finally:
        # Always clean up temp memmap directory
        if 'temp_mm_dir' in locals() and temp_mm_dir.exists():
            try:
                shutil.rmtree(temp_mm_dir)
                logger.info(f"Cleaned up temp directory: {temp_mm_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_mm_dir}: {e}")

    # Save summaries
    # Get pixel size from reader if possible (re-open temporarily)
    try:
        pixel_size_um = get_pixel_size_from_czi(czi_path)
    except Exception as e:
        logger.debug(f"Could not get pixel size from CZI: {e}")
        pixel_size_um = None

    summary = {
        'czi_path': str(czi_path),
        'pixel_size_um': pixel_size_um,
        'mk_count': mk_count,
        'hspc_count': hspc_count,
        'feature_count': f'{MORPHOLOGICAL_FEATURES_COUNT} morphological + {SAM2_EMBEDDING_DIMENSION} SAM2 + {RESNET_EMBEDDING_DIMENSION} ResNet = {TOTAL_FEATURES_PER_CELL}'
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"MKs detected: {mk_count}")
    logger.info(f"HSPCs detected: {hspc_count}")
    logger.info(f"Features per cell: 2326 (22 + 256 + 2048)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"  MK tiles: {output_dir}/mk/tiles/")
    logger.info(f"  HSPC tiles: {output_dir}/hspc/tiles/")


def _phase1_load_slides(czi_paths, tile_size, overlap, channel, norm_method='none', norm_params=None):
    """
    Phase 1: Load all slides into RAM using CZILoader and optionally normalize.

    Loads each CZI file's specified channel into RAM for efficient tile access.
    Uses the global CZI loader cache for memory management.
    If normalization is requested, normalizes the ENTIRE slide before tile extraction.

    Args:
        czi_paths: List of paths to CZI files to load.
        tile_size: Size of tiles in pixels (used for logging context).
        overlap: Overlap between tiles in pixels (used for logging context).
        channel: Channel index to load from each CZI file.
        norm_method: Normalization method ('none', 'reinhard_median', 'reinhard', 'percentile')
        norm_params: Normalization parameters (dict with statistics)

    Returns:
        tuple: (slide_data, slide_loaders) where:
            - slide_data: dict mapping slide_name -> {'image': np.array, 'shape': tuple, 'czi_path': Path}
            - slide_loaders: dict mapping slide_name -> CZILoader (kept alive to maintain RAM data)
    """
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: LOADING ALL SLIDES INTO RAM (CZILoader)")
    logger.info(f"{'='*70}")

    slide_data = {}
    slide_loaders = {}
    total_size_gb = 0

    for slide_idx, czi_path in enumerate(czi_paths):
        czi_path = Path(czi_path)
        slide_name = czi_path.stem

        logger.info(f"\n[{slide_idx+1}/{len(czi_paths)}] Loading {slide_name} (channel {channel})...")

        try:
            loader = get_loader(czi_path, load_to_ram=True, channel=channel)

            full_width, full_height = loader.mosaic_size
            size_gb = loader.channel_data.nbytes / (1024**3)
            total_size_gb += size_gb

            # Apply slide-level normalization if requested (BEFORE tile extraction!)
            channel_data = loader.channel_data
            if norm_method == 'reinhard_median' and norm_params is not None:
                from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN
                logger.info(f"  Normalizing slide (reinhard_median, L={norm_params['L_median']:.2f}±{norm_params['L_mad']:.2f})...")
                _tissue_th = norm_params.get('tissue_thresholds', {}).get(slide_name)
                _otsu, _slab = extract_slide_norm_params(_tissue_th)
                channel_data_normalized = apply_reinhard_normalization_MEDIAN(
                    channel_data,
                    norm_params,
                    otsu_threshold=_otsu,
                    slide_lab_stats=_slab,
                )
                # Free original channel data to prevent OOM (normalization returns new array)
                loader.channel_data = None
                del channel_data
                channel_data = channel_data_normalized
                gc.collect()
                logger.info(f"  Slide normalized successfully (original data freed)")
            elif norm_method == 'reinhard' and norm_params is not None:
                logger.info(f"  Normalizing slide (reinhard mean/std)...")
                _tissue_th = norm_params.get('tissue_thresholds', {}).get(slide_name)
                _otsu, _slab = extract_slide_norm_params(_tissue_th)
                channel_data_normalized = apply_reinhard_normalization(
                    channel_data,
                    norm_params,
                    otsu_threshold=_otsu,
                    slide_lab_stats=_slab,
                )
                # Free original channel data to prevent OOM
                loader.channel_data = None
                del channel_data
                channel_data = channel_data_normalized
                gc.collect()
                logger.info(f"  Slide normalized successfully (original data freed)")

            slide_loaders[slide_name] = loader
            slide_data[slide_name] = {
                'image': channel_data,  # Use normalized data if applicable
                'shape': (full_width, full_height),
                'czi_path': czi_path
            }

            logger.info(f"  Loaded: {full_width} x {full_height} ({size_gb:.2f} GB)")

            # Release CziFile reader to free memory (data is now in numpy array)
            loader.release_reader()
            gc.collect()

        except Exception as e:
            logger.error(f"Loading {slide_name}: {e}")
            continue

    logger.info(f"\nTotal RAM used: {total_size_gb:.2f} GB")

    mem = psutil.virtual_memory()
    logger.info(f"System RAM: {mem.total/(1024**3):.1f} GB total, {mem.available/(1024**3):.1f} GB available")

    return slide_data, slide_loaders


def _calibrate_and_filter_tissue(all_tiles, n_slides, calibration_block_size, calibration_samples,
                                   get_tile_fn, desc_suffix="", modality=None,
                                   precomputed_thresholds=None):
    """Shared calibration + tissue filtering logic for Phase 2.

    Args:
        all_tiles: list of (slide_name, tile_dict) tuples
        n_slides: number of slides (for scaling calibration samples)
        calibration_block_size: block size for variance calculation
        calibration_samples: samples per slide for threshold calibration
        get_tile_fn: callable(slide_name, tile) -> np.ndarray or None
        desc_suffix: suffix for tqdm progress bars (e.g. " (streaming)")
        modality: 'brightfield' for H&E (÷3 variance + pixel Otsu), None for default
        precomputed_thresholds: Dict mapping slide_name -> {variance_threshold, intensity_threshold}
            from step 1 JSON. If provided, skips calibration and uses per-slide thresholds.

    Returns:
        tuple: (tissue_tiles, variance_threshold, intensity_threshold)
    """
    import cv2

    if precomputed_thresholds:
        # Use pre-computed per-slide thresholds — skip calibration entirely
        logger.info(f"\nUsing pre-computed tissue thresholds from step 1 for {len(precomputed_thresholds)} slides")
        for sn, th in precomputed_thresholds.items():
            otsu_val = th.get('otsu_threshold', th.get('intensity_threshold', 220.0))
            logger.info(f"  {sn}: otsu_threshold={otsu_val:.1f}")

        # Compute median Otsu threshold as aggregate for downstream code
        all_otsu = [th.get('otsu_threshold', th.get('intensity_threshold', 220.0))
                    for th in precomputed_thresholds.values()]
        variance_threshold = 0.0  # unused for brightfield
        intensity_threshold = float(np.median(all_otsu))
        logger.info(f"  Aggregate (median): otsu_threshold={intensity_threshold:.1f} (variance_threshold=0, unused for brightfield)")
    else:
        n_calib_samples = min(calibration_samples * n_slides, len(all_tiles))
        logger.info(f"\nCalibrating tissue threshold from {n_calib_samples} samples...")

        np.random.seed(42)
        calib_indices = np.random.choice(len(all_tiles), n_calib_samples, replace=False)
        calib_tiles = [all_tiles[idx] for idx in calib_indices]

        calib_threads = max(1, int(os.cpu_count() * 0.8))

        def calc_tile_stats(args):
            slide_name, tile = args
            tile_img = get_tile_fn(slide_name, tile)

            if tile_img is None or tile_img.max() == 0:
                return [0.0], [255.0], np.array([], dtype=np.uint8)

            if tile_img.ndim == 3:
                gray = cv2.cvtColor(tile_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = tile_img.astype(np.uint8) if tile_img.dtype != np.uint8 else tile_img

            block_vars, block_means = calculate_block_variances(gray, calibration_block_size)

            # Collect pixel samples for brightfield Otsu
            pixel_sample = np.array([], dtype=np.uint8)
            if modality == 'brightfield':
                flat = gray.ravel()
                n = min(2000, len(flat))
                pixel_sample = flat[np.random.choice(len(flat), n, replace=False)]

            return (block_vars if block_vars else [],
                    block_means if block_means else [],
                    pixel_sample)

        all_variances = []
        all_means = []
        all_pixel_samples = []
        with ThreadPoolExecutor(max_workers=calib_threads) as executor:
            futures = {executor.submit(calc_tile_stats, tile): tile for tile in calib_tiles}
            for future in tqdm(as_completed(futures), total=len(calib_tiles), desc=f"Calibrating{desc_suffix}"):
                block_vars, block_means, pixel_sample = future.result()
                all_variances.extend(block_vars)
                all_means.extend(block_means)
                if len(pixel_sample) > 0:
                    all_pixel_samples.append(pixel_sample)

        variances = np.array(all_variances)
        means = np.array(all_means)
        pixel_samples = np.concatenate(all_pixel_samples) if all_pixel_samples else None
        logger.info(f"  Running K-means + Otsu on {len(variances)} block samples...")
        variance_threshold, intensity_threshold = compute_tissue_thresholds(
            variances, means, modality=modality, pixel_samples=pixel_samples)

    logger.info(f"\nFiltering to tissue-containing tiles...")

    tissue_check_threads = max(1, int(os.cpu_count() * 0.8))
    logger.info(f"  Using {tissue_check_threads} threads for parallel tissue checking")

    def check_tile_tissue(args):
        slide_name, tile = args
        tile_img = get_tile_fn(slide_name, tile)

        if tile_img is None or tile_img.max() == 0:
            return None

        # Use per-slide thresholds if available, otherwise aggregate
        if precomputed_thresholds and slide_name in precomputed_thresholds:
            vt = 0.0  # unused for brightfield
            it = precomputed_thresholds[slide_name].get(
                'otsu_threshold',
                precomputed_thresholds[slide_name].get('intensity_threshold', 220.0))
        else:
            vt = variance_threshold
            it = intensity_threshold

        has_tissue_flag, _ = has_tissue(tile_img, vt, block_size=calibration_block_size,
                                        intensity_threshold=it, modality=modality)

        if has_tissue_flag:
            return (slide_name, tile)
        return None

    tissue_tiles = []
    with ThreadPoolExecutor(max_workers=tissue_check_threads) as executor:
        futures = {executor.submit(check_tile_tissue, tile_args): tile_args for tile_args in all_tiles}
        for future in tqdm(as_completed(futures), total=len(all_tiles), desc=f"Checking tissue{desc_suffix}"):
            result = future.result()
            if result is not None:
                tissue_tiles.append(result)

    logger.info(f"\nTissue tiles: {len(tissue_tiles)} / {len(all_tiles)} ({100*len(tissue_tiles)/len(all_tiles):.1f}%)")

    return tissue_tiles, variance_threshold, intensity_threshold


def _create_tile_grid(slide_names_and_sizes, tile_size, overlap):
    """Create tile grid for multiple slides.

    Args:
        slide_names_and_sizes: list of (slide_name, full_width, full_height) tuples
        tile_size: tile size in pixels
        overlap: overlap between tiles in pixels

    Returns:
        list of (slide_name, tile_dict) tuples
    """
    all_tiles = []
    for slide_name, full_width, full_height in slide_names_and_sizes:
        n_tx = int(np.ceil(full_width / (tile_size - overlap)))
        n_ty = int(np.ceil(full_height / (tile_size - overlap)))

        slide_tiles = []
        for ty in range(n_ty):
            for tx in range(n_tx):
                tile = {
                    'id': f"{slide_name}_{len(slide_tiles)}",
                    'x': tx * (tile_size - overlap),
                    'y': ty * (tile_size - overlap),
                    'w': min(tile_size, full_width - tx * (tile_size - overlap)),
                    'h': min(tile_size, full_height - ty * (tile_size - overlap))
                }
                slide_tiles.append(tile)

        logger.info(f"  {slide_name}: {len(slide_tiles)} tiles")
        all_tiles.extend((slide_name, tile) for tile in slide_tiles)

    return all_tiles


def _phase2_identify_tissue_tiles(slide_data, tile_size, overlap, variance_threshold, calibration_block_size, calibration_samples,
                                   precomputed_thresholds=None):
    """Phase 2: Create tiles and identify tissue-containing tiles (RAM mode)."""
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: IDENTIFYING TISSUE TILES")
    logger.info(f"{'='*70}")

    slide_sizes = [(name, data['shape'][0], data['shape'][1]) for name, data in slide_data.items()]
    all_tiles = _create_tile_grid(slide_sizes, tile_size, overlap)
    logger.info(f"\nTotal tiles across all slides: {len(all_tiles)}")

    def get_tile_from_ram(slide_name, tile):
        img = slide_data[slide_name]['image']
        return img[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]

    return _calibrate_and_filter_tissue(
        all_tiles, len(slide_data), calibration_block_size, calibration_samples, get_tile_from_ram,
        modality='brightfield',
        precomputed_thresholds=precomputed_thresholds,
    )


def _phase2_identify_tissue_tiles_streaming(slide_loaders, tile_size, overlap, variance_threshold, calibration_block_size, calibration_samples, channel,
                                             precomputed_thresholds=None):
    """Phase 2 (Streaming): Identify tissue tiles by reading from CZI on-demand."""
    from segmentation.processing.memory import log_memory_status

    log_memory_status("Phase 2 tissue detection (streaming) - START")

    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: IDENTIFYING TISSUE TILES (STREAMING MODE)")
    logger.info(f"{'='*70}")

    slide_sizes = [(name, loader.mosaic_size[0], loader.mosaic_size[1]) for name, loader in slide_loaders.items()]
    all_tiles = _create_tile_grid(slide_sizes, tile_size, overlap)
    logger.info(f"\nTotal tiles across all slides: {len(all_tiles)}")

    def get_tile_from_czi(slide_name, tile):
        loader = slide_loaders[slide_name]
        x_origin, y_origin = loader.mosaic_origin
        return loader.get_tile(x_origin + tile['x'], y_origin + tile['y'], tile_size, channel)

    tissue_tiles, variance_threshold, intensity_threshold = _calibrate_and_filter_tissue(
        all_tiles, len(slide_loaders), calibration_block_size, calibration_samples,
        get_tile_from_czi, desc_suffix=" (streaming)", modality='brightfield',
        precomputed_thresholds=precomputed_thresholds,
    )

    log_memory_status("Phase 2 tissue detection (streaming) - END")

    return tissue_tiles, variance_threshold, intensity_threshold


def _phase3_sample_tiles(tissue_tiles, sample_fraction):
    """
    Phase 3: Sample tiles from the combined pool of tissue-containing tiles.

    Performs random sampling across all slides to ensure representative coverage.
    Uses a fixed random seed for reproducibility.

    Args:
        tissue_tiles: list of (slide_name, tile_dict) tuples containing tissue.
        sample_fraction: Fraction of tiles to sample (0.0 to 1.0).

    Returns:
        list: Sampled tiles as (slide_name, tile_dict) tuples.
    """
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3: SAMPLING FROM COMBINED POOL")
    logger.info(f"{'='*70}")

    if sample_fraction < 1.0:
        n_sample = max(1, int(len(tissue_tiles) * sample_fraction))
        np.random.seed(42)
        sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
        sampled_tiles = [tissue_tiles[i] for i in sample_indices]
    else:
        sampled_tiles = tissue_tiles

    slide_counts = {}
    for slide_name, tile in sampled_tiles:
        slide_counts[slide_name] = slide_counts.get(slide_name, 0) + 1

    logger.info(f"Sampled {len(sampled_tiles)} tiles ({sample_fraction*100:.0f}% of tissue tiles)")
    logger.info(f"\nPer-slide distribution:")
    for slide_name in sorted(slide_counts.keys()):
        logger.info(f"  {slide_name}: {slide_counts[slide_name]} tiles")

    return sampled_tiles


def _phase4_process_tiles(
    sampled_tiles,
    slide_data,
    slide_loaders,
    output_base,
    mk_min_area,
    mk_max_area,
    variance_threshold,
    calibration_block_size,
    num_workers,
    mk_classifier_path,
    hspc_classifier_path,
    html_output_dir,
    samples_per_page,
    mk_min_area_um,
    mk_max_area_um,
    hspc_min_area=25,
    hspc_max_area=100,
    multi_gpu=False,
    num_gpus=4,
    norm_params=None,
    normalization_method='none',
    intensity_threshold=220,
    per_slide_thresholds=None,
):
    """
    Phase 4: Process sampled tiles using ML models (SAM2, Cellpose, ResNet).

    Sets up a multiprocessing worker pool, processes each tile for MK and HSPC
    detection, saves results to disk, generates summaries, and optionally exports
    HTML annotation pages.

    Args:
        sampled_tiles: list of (slide_name, tile_dict) tuples to process.
        slide_data: dict mapping slide_name -> {'image': np.array, 'shape': tuple, 'czi_path': Path}
        slide_loaders: dict mapping slide_name -> CZILoader
        output_base: Path to output directory.
        mk_min_area: Minimum MK area in pixels.
        mk_max_area: Maximum MK area in pixels.
        variance_threshold: Variance threshold for tissue detection.
        calibration_block_size: Block size for variance calculation.
        num_workers: Number of worker processes for multiprocessing pool.
        mk_classifier_path: Path to MK classifier model (optional).
        hspc_classifier_path: Path to HSPC classifier model (optional).
        html_output_dir: Path to HTML output directory (optional).
        samples_per_page: Number of samples per HTML page.
        mk_min_area_um: Minimum MK area in um^2 for HTML export filtering.
        mk_max_area_um: Maximum MK area in um^2 for HTML export filtering.
        multi_gpu: Enable multi-GPU mode (each GPU processes one tile at a time).
        num_gpus: Number of GPUs to use in multi-GPU mode.

    Returns:
        tuple: (total_mk, total_hspc) counts of detected cells.
    """
    import shutil

    logger.info(f"\n{'='*70}")
    logger.info("PHASE 4: PROCESSING SAMPLED TILES")
    if multi_gpu:
        logger.info(f"Mode: MULTI-GPU ({num_gpus} GPUs, 1 tile per GPU)")
    else:
        logger.info(f"Mode: Standard pool ({num_workers} workers)")
    logger.info(f"{'='*70}")

    slide_results = {name: {'mk_count': 0, 'hspc_count': 0, 'mk_gid': 1, 'hspc_gid': 1}
                     for name in slide_data.keys()}

    # Helper function to process a single result (shared by both modes)
    def process_result(result, slide_results):
        """Process a single tile result and save to disk."""
        if result['status'] == 'success':
            slide_name = result['slide_name']
            tile = result['tile']
            tid = result['tid']
            output_dir = output_base / slide_name
            output_dir.mkdir(parents=True, exist_ok=True)

            sr = slide_results[slide_name]

            if result.get('mk_feats'):
                mk_dir = output_dir / "mk" / "tiles"
                mk_tile_dir = mk_dir / str(tid)
                mk_tile_dir.mkdir(parents=True, exist_ok=True)

                with open(mk_tile_dir / "window.csv", 'w') as f:
                    f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

                new_mk = np.zeros_like(result['mk_masks'])
                mk_tile_cells = []
                for feat in result['mk_feats']:
                    old_id = int(feat['id'].split('_')[1])
                    new_mk[result['mk_masks'] == old_id] = sr['mk_gid']
                    feat['id'] = f'det_{sr["mk_gid"] - 1}'
                    feat['global_id'] = sr['mk_gid']  # Keep for backwards compatibility
                    feat['center'][0] += tile['x']
                    feat['center'][1] += tile['y']
                    # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
                    feat['uid'] = f"{slide_name}_mk_{round(feat['center'][0])}_{round(feat['center'][1])}"
                    mk_tile_cells.append(sr['mk_gid'])
                    sr['mk_count'] += 1
                    sr['mk_gid'] += 1

                with open(mk_tile_dir / "classes.csv", 'w') as f:
                    for c in mk_tile_cells: f.write(f"{c}\n")
                with h5py.File(mk_tile_dir / "segmentation.h5", 'w') as f:
                    create_hdf5_dataset(f, 'labels', new_mk[np.newaxis])
                with open(mk_tile_dir / "features.json", 'w') as f:
                    json.dump(result['mk_feats'], f)

                del new_mk, mk_tile_cells

            if result.get('hspc_feats'):
                hspc_dir = output_dir / "hspc" / "tiles"
                hspc_tile_dir = hspc_dir / str(tid)
                hspc_tile_dir.mkdir(parents=True, exist_ok=True)

                with open(hspc_tile_dir / "window.csv", 'w') as f:
                    f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

                new_hspc = np.zeros_like(result['hspc_masks'])
                hspc_tile_cells = []
                for feat in result['hspc_feats']:
                    old_id = int(feat['id'].split('_')[1])
                    new_hspc[result['hspc_masks'] == old_id] = sr['hspc_gid']
                    feat['id'] = f'det_{sr["hspc_gid"] - 1}'
                    feat['global_id'] = sr['hspc_gid']  # Keep for backwards compatibility
                    feat['center'][0] += tile['x']
                    feat['center'][1] += tile['y']
                    # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
                    feat['uid'] = f"{slide_name}_hspc_{round(feat['center'][0])}_{round(feat['center'][1])}"
                    hspc_tile_cells.append(sr['hspc_gid'])
                    sr['hspc_count'] += 1
                    sr['hspc_gid'] += 1

                with open(hspc_tile_dir / "classes.csv", 'w') as f:
                    for c in hspc_tile_cells: f.write(f"{c}\n")
                with h5py.File(hspc_tile_dir / "segmentation.h5", 'w') as f:
                    create_hdf5_dataset(f, 'labels', new_hspc[np.newaxis])
                with open(hspc_tile_dir / "features.json", 'w') as f:
                    json.dump(result['hspc_feats'], f)

                del new_hspc, hspc_tile_cells

        elif result['status'] == 'error':
            logger.error(f"Tile error: {result.get('error', 'unknown')}")

        # Cleanup result
        for key in ['mk_masks', 'hspc_masks', 'mk_feats', 'hspc_feats']:
            if key in result:
                del result[key]

    def collect_results_with_retry(
        processor,
        sampled_tiles,
        slide_results,
        process_result_fn,
        max_retries=2,
        timeout_per_tile=120,
        stall_timeout=300,
    ):
        """
        Collect results with retry logic for failed/timed-out tiles.

        Args:
            processor: MultiGPUTileProcessorSHM instance
            sampled_tiles: List of (slide_name, tile) tuples
            slide_results: Dict to accumulate results
            process_result_fn: Function to process each result
            max_retries: Max retries per tile before giving up
            timeout_per_tile: Timeout for each result collection
            stall_timeout: Timeout before declaring workers stalled

        Returns:
            tuple: (completed_count, failed_tiles) where failed_tiles is list of (slide_name, tile, error)
        """
        # Track pending tiles by their ID
        pending = {}  # tid -> (slide_name, tile, retry_count)
        for slide_name, tile in sampled_tiles:
            tid = tile.get('id', f"{tile['x']}_{tile['y']}")
            pending[tid] = (slide_name, tile, 0)

        completed = 0
        failed_tiles = []
        consecutive_timeouts = 0
        max_consecutive_timeouts = 3  # Declare workers dead after 3 consecutive timeouts

        with tqdm(total=len(sampled_tiles), desc="Processing tiles") as pbar:
            while pending or completed < len(sampled_tiles):
                result = processor.collect_result(timeout=timeout_per_tile)

                if result is None:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= max_consecutive_timeouts:
                        logger.error(f"Workers appear stalled after {consecutive_timeouts} consecutive timeouts")
                        logger.error(f"Completed {completed}/{len(sampled_tiles)} tiles, {len(pending)} pending")
                        # Mark all pending as failed
                        for tid, (slide_name, tile, retries) in pending.items():
                            failed_tiles.append((slide_name, tile, "Worker stall timeout"))
                        pending.clear()
                        break
                    continue

                # Reset timeout counter on successful result
                consecutive_timeouts = 0

                # Skip ready messages
                if result.get('status') == 'ready':
                    continue

                tid = result.get('tid')
                if tid and tid in pending:
                    slide_name, tile, retries = pending.pop(tid)

                    if result['status'] == 'success':
                        process_result_fn(result, slide_results)
                        completed += 1
                        pbar.update(1)
                    elif result['status'] == 'error':
                        error_msg = result.get('error', 'unknown')
                        if retries < max_retries:
                            logger.warning(f"Tile {tid} failed (attempt {retries+1}/{max_retries+1}): {error_msg}, retrying...")
                            pending[tid] = (slide_name, tile, retries + 1)
                            processor.submit_tile(slide_name, tile)
                        else:
                            logger.error(f"Tile {tid} failed after {max_retries+1} attempts: {error_msg}")
                            failed_tiles.append((slide_name, tile, error_msg))
                            completed += 1  # Count as "done" even if failed
                            pbar.update(1)
                    else:
                        # Empty, no_tissue, invalid - count as completed
                        completed += 1
                        pbar.update(1)

                # Cleanup result
                for key in ['mk_masks', 'hspc_masks', 'mk_feats', 'hspc_feats']:
                    if key in result:
                        del result[key]
                del result
                gc.collect()

        return completed, failed_tiles

    # Track failed tiles across the run
    all_failed_tiles = []

    # =========================================================================
    # MULTI-GPU MODE: Each GPU processes one tile at a time (SHARED MEMORY)
    # =========================================================================
    if multi_gpu:
        from segmentation.processing.multigpu_shm import (
            SharedSlideManager,
            MultiGPUTileProcessorSHM
        )

        # Create shared memory for all slides (zero-copy access for workers)
        # Copy one slide at a time to minimize peak memory usage
        logger.info("Moving slides to shared memory...")
        shm_manager = SharedSlideManager()
        for i, (slide_name, data) in enumerate(slide_data.items()):
            img = data['image']
            size_gb = img.nbytes / (1024**3)
            logger.info(f"  [{i+1}/{len(slide_data)}] {slide_name}: {size_gb:.1f} GB -> shared memory")
            shm_manager.add_slide(slide_name, img)
            # Free original array immediately after copying
            data['image'] = None
            del img
            gc.collect()
        logger.info(f"All {len(slide_data)} slides now in shared memory")

        try:
            with MultiGPUTileProcessorSHM(
                num_gpus=num_gpus,
                slide_info=shm_manager.get_slide_info(),
                mk_classifier_path=mk_classifier_path,
                hspc_classifier_path=hspc_classifier_path,
                mk_min_area=mk_min_area,
                mk_max_area=mk_max_area,
                hspc_max_area=hspc_max_area,
                variance_threshold=variance_threshold,
                calibration_block_size=calibration_block_size,
                cleanup_config=cleanup_config,
                norm_params=norm_params,
                normalization_method=normalization_method,
                intensity_threshold=intensity_threshold,
                modality='brightfield',
                per_slide_thresholds=per_slide_thresholds,
            ) as processor:
                # Submit all tiles (only coordinates, not data!)
                logger.info(f"Submitting {len(sampled_tiles)} tiles to {num_gpus} GPUs (shared memory)...")
                for slide_name, tile in sampled_tiles:
                    processor.submit_tile(slide_name, tile)

                # Collect results with retry logic
                logger.info("Collecting results...")
                completed, failed = collect_results_with_retry(
                    processor, sampled_tiles, slide_results, process_result
                )
                all_failed_tiles.extend(failed)
        finally:
            # Always cleanup shared memory
            logger.info("Cleaning up shared memory...")
            shm_manager.cleanup()

    # =========================================================================
    # STANDARD MODE: Now also uses shared memory for zero-copy worker access
    # =========================================================================
    else:
        from segmentation.processing.multigpu_shm import (
            SharedSlideManager,
            MultiGPUTileProcessorSHM
        )

        # Move slides from RAM to shared memory (zero-copy for workers)
        logger.info("Moving slides to shared memory...")
        shm_manager = SharedSlideManager()
        for i, (slide_name, data) in enumerate(slide_data.items()):
            img = data['image']
            if img is None:
                continue
            size_gb = img.nbytes / (1024**3)
            logger.info(f"  [{i+1}/{len(slide_data)}] {slide_name}: {size_gb:.1f} GB -> shared memory")
            shm_manager.add_slide(slide_name, img)
            # Free original array immediately after copying
            data['image'] = None
            del img
            gc.collect()
        logger.info(f"All slides now in shared memory")

        # Use same processor as multi-GPU mode but with num_workers GPUs
        actual_gpus = min(num_workers, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        logger.info(f"Using {actual_gpus} GPUs for processing")

        if normalization_method != 'none' and norm_params is None:
            logger.warning(f"Normalization method is '{normalization_method}' but norm_params is None — "
                           f"tiles will NOT be normalized by GPU workers. "
                           f"Ensure normalization was applied at slide level before calling _phase4_process_tiles().")

        try:
            with MultiGPUTileProcessorSHM(
                num_gpus=actual_gpus,
                slide_info=shm_manager.get_slide_info(),
                mk_classifier_path=mk_classifier_path,
                hspc_classifier_path=hspc_classifier_path,
                mk_min_area=mk_min_area,
                mk_max_area=mk_max_area,
                hspc_max_area=hspc_max_area,
                variance_threshold=variance_threshold,
                calibration_block_size=calibration_block_size,
                cleanup_config=cleanup_config,
                norm_params=norm_params,
                normalization_method=normalization_method,
                intensity_threshold=intensity_threshold,
                modality='brightfield',
                per_slide_thresholds=per_slide_thresholds,
            ) as processor:
                # Submit all tiles (only coordinates, not data!)
                logger.info(f"Submitting {len(sampled_tiles)} tiles to {actual_gpus} GPUs (shared memory)...")
                for slide_name, tile in sampled_tiles:
                    processor.submit_tile(slide_name, tile)

                # Collect results with retry logic
                logger.info("Collecting results...")
                completed, failed = collect_results_with_retry(
                    processor, sampled_tiles, slide_results, process_result
                )
                all_failed_tiles.extend(failed)
        finally:
            # Always cleanup shared memory
            logger.info("Cleaning up shared memory...")
            shm_manager.cleanup()

    total_mk = 0
    total_hspc = 0

    for slide_name, data in slide_data.items():
        sr = slide_results[slide_name]
        output_dir = output_base / slide_name

        if sr['mk_count'] > 0 or sr['hspc_count'] > 0:
            output_dir.mkdir(parents=True, exist_ok=True)

            pixel_size_um = None
            if slide_name in slide_loaders:
                try:
                    pixel_size_um = slide_loaders[slide_name].get_pixel_size()
                except Exception as e:
                    logger.debug(f"Failed to get pixel size from loader: {e}")
            if pixel_size_um is None:
                try:
                    pixel_size_um = get_pixel_size_from_czi(data['czi_path'])
                except Exception as e:
                    logger.debug(f"Failed to get pixel size from CZI: {e}")
                    pixel_size_um = None

            summary = {
                'czi_path': str(data['czi_path']),
                'pixel_size_um': pixel_size_um,
                'mk_count': sr['mk_count'],
                'hspc_count': sr['hspc_count'],
                'feature_count': f'{MORPHOLOGICAL_FEATURES_COUNT} morphological + {SAM2_EMBEDDING_DIMENSION} SAM2 + {RESNET_EMBEDDING_DIMENSION} ResNet = {TOTAL_FEATURES_PER_CELL}'
            }
            with open(output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

        total_mk += sr['mk_count']
        total_hspc += sr['hspc_count']
        logger.info(f"  {slide_name}: {sr['mk_count']} MKs, {sr['hspc_count']} HSPCs")

    if html_output_dir:
        # Check if images are still in RAM (they're freed when using shared memory mode)
        images_available = any(data.get('image') is not None for data in slide_data.values())
        if images_available:
            export_html_from_ram(
                slide_data=slide_data,
                output_base=output_base,
                html_output_dir=html_output_dir,
                samples_per_page=samples_per_page,
                mk_min_area_um=mk_min_area_um,
                mk_max_area_um=mk_max_area_um
            )
        else:
            logger.info("Skipping RAM-based HTML export (images freed for shared memory)")
            logger.info(f"Run 'python regenerate_html_fast.py --output-dir {output_base}' to generate HTML from saved crops")

    # Report failed tiles
    if all_failed_tiles:
        logger.error(f"\n{'='*70}")
        logger.error(f"FAILED TILES: {len(all_failed_tiles)} tiles failed to process")
        logger.error(f"{'='*70}")
        # Group by slide
        failed_by_slide = {}
        for slide_name, tile, error in all_failed_tiles:
            failed_by_slide.setdefault(slide_name, []).append((tile, error))
        for slide_name, failures in failed_by_slide.items():
            logger.error(f"  {slide_name}: {len(failures)} failed tiles")
            for tile, error in failures[:3]:  # Show first 3 errors per slide
                tid = tile.get('id', f"{tile['x']}_{tile['y']}")
                logger.error(f"    - {tid}: {error}")
            if len(failures) > 3:
                logger.error(f"    ... and {len(failures) - 3} more")

        # Save failed tiles to JSON for potential retry
        failed_json = output_base / "failed_tiles.json"
        with open(failed_json, 'w') as f:
            json.dump([
                {'slide': s, 'tile_id': t.get('id'), 'x': t['x'], 'y': t['y'], 'error': e}
                for s, t, e in all_failed_tiles
            ], f, indent=2)
        logger.error(f"Failed tiles saved to: {failed_json}")

        # Raise error to indicate incomplete run
        raise RuntimeError(f"Run incomplete: {len(all_failed_tiles)} tiles failed. See {failed_json} for details.")

    return total_mk, total_hspc


def run_multi_slide_segmentation(
    czi_paths,
    output_base,
    mk_min_area=1000,
    mk_max_area=100000,
    hspc_min_area=None,
    hspc_max_area=None,
    tile_size=4096,
    overlap=512,
    sample_fraction=1.0,
    calibration_block_size=512,
    calibration_samples=50,
    num_workers=4,
    mk_classifier_path=None,
    hspc_classifier_path=None,
    html_output_dir=None,
    samples_per_page=300,
    mk_min_area_um=200,
    mk_max_area_um=2000,
    channel=0,
    multi_gpu=False,
    num_gpus=4,
    normalize_slides=False,
    normalization_method='percentile',
    norm_params_file=None,
    norm_percentile_low=1.0,
    norm_percentile_high=99.0,
):
    """
    Process multiple slides with UNIFIED SAMPLING using RAM-first architecture.

    This function orchestrates a 4-phase pipeline:
    - Phase 1: Loads ALL slides into RAM using CZILoader
    - Phase 2: Identifies tissue-containing tiles across ALL slides
    - Phase 3: Samples from the combined pool (truly representative)
    - Phase 4: Processes sampled tiles with models loaded ONCE

    Each phase is implemented in a separate function for better maintainability:
    - _phase1_load_slides()
    - _phase2_identify_tissue_tiles()
    - _phase3_sample_tiles()
    - _phase4_process_tiles()
    """
    # Set start method to 'spawn' for GPU safety
    if num_workers > 0:
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # Already set, verify it's 'spawn'
            if mp.get_start_method() != 'spawn':
                logger.warning(f"Multiprocessing start method already set to '{mp.get_start_method()}', expected 'spawn'")

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Initialize normalization parameters (shared by both multi-GPU and standard paths)
    # Set up once here so both branches can use them.
    norm_params_for_workers = None
    norm_method_for_workers = 'none'

    precomputed_thresholds = None
    if normalize_slides and normalization_method == 'reinhard':
        if not norm_params_file:
            raise ValueError(
                "Reinhard normalization requires --norm-params-file to be specified. "
                "Please provide a JSON file with Reinhard Lab statistics."
            )
        with open(norm_params_file, 'r') as f:
            reinhard_params = json.load(f)
        if 'L_median' in reinhard_params and 'L_mad' in reinhard_params:
            norm_method_for_workers = 'reinhard_median'
        else:
            norm_method_for_workers = 'reinhard'
        norm_params_for_workers = reinhard_params

        # Extract per-slide tissue thresholds from step 1 (if present)
        precomputed_thresholds = reinhard_params.get('tissue_thresholds')
        if precomputed_thresholds:
            logger.info(f"Found pre-computed tissue thresholds for {len(precomputed_thresholds)} slides")
        else:
            logger.warning(f"No tissue_thresholds in params file — will fall back to recalibration")

    # =========================================================================
    # MULTI-GPU MODE: Memory-efficient streaming approach
    # =========================================================================
    if multi_gpu:
        from segmentation.processing.multigpu_shm import SharedSlideManager, MultiGPUTileProcessorSHM
        from segmentation.processing.memory import log_memory_status

        logger.info(f"\n{'='*70}")
        logger.info(f"UNIFIED SAMPLING SEGMENTATION: {len(czi_paths)} slides (RAM-FIRST MODE)")
        logger.info(f"{'='*70}")
        logger.info(f"Step 1: Load ALL slides to RAM (fast subsequent access)")
        logger.info(f"Step 2: Identify tissue tiles (from RAM - fast)")
        logger.info(f"Step 3: Sample {sample_fraction*100:.0f}% from combined pool")
        logger.info(f"Step 4: Copy RAM to shared memory, then process with GPUs")
        logger.info(f"{'='*70}")

        log_memory_status("Before Phase 1")

        if norm_params_for_workers is not None:
            logger.info(f"\n{'='*70}")
            logger.info("REINHARD NORMALIZATION (SLIDE-LEVEL)")
            logger.info(f"{'='*70}")
            logger.info(f"  Method: {norm_method_for_workers}")
            logger.info(f"  Normalization will be applied to each slide BEFORE tile extraction")

        # Phase 1: Load all slides to RAM for fast tissue detection
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 1: LOADING ALL SLIDES TO RAM")
        logger.info(f"{'='*70}")

        slide_loaders = {}
        slide_metadata = {}
        slide_is_rgb = {}  # Track which slides have RGB data
        for i, czi_path in enumerate(czi_paths):
            czi_path = Path(czi_path)
            slide_name = czi_path.stem
            log_memory_status(f"Before loading slide {i+1}/{len(czi_paths)} ({slide_name})")

            logger.info(f"[{i+1}/{len(czi_paths)}] Loading {slide_name} to RAM...")
            loader = get_loader(czi_path, load_to_ram=True, channel=channel)
            slide_loaders[slide_name] = loader
            slide_metadata[slide_name] = {
                'shape': loader.mosaic_size,
                'czi_path': czi_path
            }
            # Check if this slide has RGB data - use purpose-built method
            is_rgb = loader.is_channel_rgb(channel)
            slide_is_rgb[slide_name] = is_rgb

            # Get channel data for shape logging
            channel_data = loader.get_channel_data(channel)
            if is_rgb:
                logger.info(f"  {slide_name}: RGB data detected (shape: {channel_data.shape})")
            else:
                logger.info(f"  {slide_name}: Grayscale data (shape: {channel_data.shape if channel_data is not None else 'None'})")

            # Apply slide-level normalization if requested
            if norm_method_for_workers == 'reinhard_median' and norm_params_for_workers is not None:
                from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN
                logger.info(f"  Normalizing {slide_name} (reinhard_median, L={norm_params_for_workers['L_median']:.2f}±{norm_params_for_workers['L_mad']:.2f})...")
                _tissue_th = precomputed_thresholds.get(slide_name) if precomputed_thresholds else None
                _otsu, _slab = extract_slide_norm_params(_tissue_th)
                channel_data_normalized = apply_reinhard_normalization_MEDIAN(
                    channel_data,
                    norm_params_for_workers,
                    otsu_threshold=_otsu,
                    slide_lab_stats=_slab,
                )
                # Replace channel data in loader with normalized version
                loader.channel_data = channel_data_normalized
                del channel_data, channel_data_normalized
                logger.info(f"  {slide_name} normalized successfully")
                gc.collect()
            elif norm_method_for_workers == 'reinhard' and norm_params_for_workers is not None:
                logger.info(f"  Normalizing {slide_name} (reinhard mean/std)...")
                _tissue_th = precomputed_thresholds.get(slide_name) if precomputed_thresholds else None
                _otsu, _slab = extract_slide_norm_params(_tissue_th)
                channel_data_normalized = apply_reinhard_normalization(
                    channel_data,
                    norm_params_for_workers,
                    otsu_threshold=_otsu,
                    slide_lab_stats=_slab,
                )
                # Replace channel data in loader with normalized version
                loader.channel_data = channel_data_normalized
                del channel_data, channel_data_normalized
                gc.collect()
                logger.info(f"  {slide_name} normalized successfully")

            log_memory_status(f"After loading slide {i+1}")
            gc.collect()

        log_memory_status("After Phase 1 (all slides in RAM)")

        # Reinhard normalization already applied at slide level above (lines 2540-2568).
        # Do NOT pass norm params to GPU workers — would cause double normalization.

        # CROSS-SLIDE NORMALIZATION (percentile only, requires multiple slides)
        if normalize_slides and len(czi_paths) > 1 and normalization_method == 'percentile':
            logger.info(f"\n{'='*70}")
            logger.info("CROSS-SLIDE PERCENTILE NORMALIZATION")
            logger.info(f"{'='*70}")

            # Load normalization parameters from file if provided
            if norm_params_file:
                logger.info(f"Loading pre-computed normalization parameters from: {norm_params_file}")
                with open(norm_params_file, 'r') as f:
                    params = json.load(f)

                # Validate required keys in JSON file
                required_keys = {'target_low', 'target_high', 'p_low', 'p_high', 'n_slides'}
                missing_keys = required_keys - set(params.keys())
                if missing_keys:
                    raise ValueError(
                        f"Normalization parameters file '{norm_params_file}' is missing required keys: {missing_keys}. "
                        f"Required keys: {required_keys}"
                    )

                # Validate percentile values are in range [0, 100]
                p_low = params['p_low']
                p_high = params['p_high']

                if not isinstance(p_low, (int, float)) or not isinstance(p_high, (int, float)):
                    raise ValueError(
                        f"Percentile values must be numeric. Got p_low={type(p_low).__name__}, p_high={type(p_high).__name__}"
                    )

                if not (0 <= p_low <= 100):
                    raise ValueError(
                        f"Lower percentile (p_low) must be in range [0, 100]. Got: {p_low}"
                    )

                if not (0 <= p_high <= 100):
                    raise ValueError(
                        f"Upper percentile (p_high) must be in range [0, 100]. Got: {p_high}"
                    )

                if p_low >= p_high:
                    raise ValueError(
                        f"Lower percentile must be less than upper percentile. Got p_low={p_low}, p_high={p_high}"
                    )

                # Validate and convert target_low and target_high
                try:
                    target_low = np.array(params['target_low'], dtype=np.float32)
                    target_high = np.array(params['target_high'], dtype=np.float32)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Failed to convert target_low and target_high to arrays: {e}"
                    )

                # Validate target range has 3 channels (RGB)
                if target_low.shape != (3,) or target_high.shape != (3,):
                    raise ValueError(
                        f"Target ranges must have exactly 3 channels (RGB). "
                        f"Got target_low shape={target_low.shape}, target_high shape={target_high.shape}"
                    )

                # Validate target ranges are valid (low < high for each channel)
                for channel_idx in range(3):
                    channel_name = ['R', 'G', 'B'][channel_idx]
                    if target_low[channel_idx] >= target_high[channel_idx]:
                        raise ValueError(
                            f"Invalid target range for {channel_name} channel: "
                            f"target_low ({target_low[channel_idx]}) must be less than "
                            f"target_high ({target_high[channel_idx]})"
                        )

                # Log validated parameters
                logger.info(f"  Parameters computed from {params['n_slides']} slides")
                logger.info(f"  P{p_low}-P{p_high}")
            else:
                # Compute global percentiles from current slides
                logger.info(f"Computing global percentiles (P{norm_percentile_low}-P{norm_percentile_high}) from {len(slide_loaders)} slides...")

                # Collect channel data from all slides for global stats
                all_channel_data = []
                for slide_name, loader in slide_loaders.items():
                    channel_data = loader.get_channel_data(channel)
                    if channel_data is not None:
                        all_channel_data.append(channel_data)

                # Compute global target percentiles
                target_low, target_high = compute_global_percentiles(
                    all_channel_data,
                    p_low=norm_percentile_low,
                    p_high=norm_percentile_high,
                    n_samples=50000  # Sample 50k pixels per slide
                )

            logger.info(f"  Global target range:")
            logger.info(f"    R: [{target_low[0]:.1f}, {target_high[0]:.1f}]")
            logger.info(f"    G: [{target_low[1]:.1f}, {target_high[1]:.1f}]")
            logger.info(f"    B: [{target_low[2]:.1f}, {target_high[2]:.1f}]")

            # Normalize each slide
            for slide_name, loader in slide_loaders.items():
                logger.info(f"  Normalizing {slide_name}...")
                channel_data = loader.get_channel_data(channel)

                if channel_data is not None:
                    # Show before stats
                    before_mean = channel_data.mean(axis=(0,1))

                    # Normalize
                    normalized = normalize_to_percentiles(
                        channel_data,
                        target_low,
                        target_high,
                        p_low=norm_percentile_low,
                        p_high=norm_percentile_high
                    )

                    # Update loader's channel data
                    loader.channel_data = normalized

                    # Show after stats
                    after_mean = normalized.mean(axis=(0,1))

                    # Handle both RGB and grayscale cases
                    if normalized.ndim == 3 and normalized.shape[2] == 3:
                        # RGB case: log all 3 channels
                        logger.info(f"    Before: RGB=({before_mean[0]:.1f}, {before_mean[1]:.1f}, {before_mean[2]:.1f})")
                        logger.info(f"    After:  RGB=({after_mean[0]:.1f}, {after_mean[1]:.1f}, {after_mean[2]:.1f})")
                    else:
                        # Grayscale case: log single mean value
                        logger.info(f"    Before: Grayscale={before_mean:.1f}")
                        logger.info(f"    After:  Grayscale={after_mean:.1f}")

                logger.info("Percentile normalization complete!")
                log_memory_status("After percentile normalization")

        # Phase 2: Identify tissue tiles using streaming (on-demand reading)
        tissue_tiles, variance_threshold, intensity_threshold = _phase2_identify_tissue_tiles_streaming(
            slide_loaders, tile_size, overlap, None, calibration_block_size, calibration_samples, channel,
            precomputed_thresholds=precomputed_thresholds,
        )

        log_memory_status("After Phase 2 (tissue detection)")

        # Phase 3: Sample from combined pool
        sampled_tiles = _phase3_sample_tiles(tissue_tiles, sample_fraction)

        # Phase 4: Load slides to shared memory ONE AT A TIME, then process
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 4: COPYING RAM TO SHARED MEMORY & PROCESSING")
        logger.info(f"{'='*70}")

        shm_manager = SharedSlideManager()
        slide_results = {name: {'mk_count': 0, 'hspc_count': 0, 'mk_gid': 1, 'hspc_gid': 1}
                        for name in slide_loaders.keys()}

        # Determine which slides have sampled tiles
        slides_with_tiles = set(slide_name for slide_name, _ in sampled_tiles)
        logger.info(f"Copying {len(slides_with_tiles)} slides from RAM to shared memory...")

        failed_slides = set()
        for i, slide_name in enumerate(list(slides_with_tiles)):
            loader = slide_loaders[slide_name]
            log_memory_status(f"Before copying slide {i+1}/{len(slides_with_tiles)} ({slide_name})")

            try:
                # Get channel data from RAM (already loaded in Phase 1)
                channel_data = loader.get_channel_data(channel)
                if channel_data is None:
                    raise ValueError(f"Channel {channel} not loaded in RAM for {slide_name}")

                is_rgb = slide_is_rgb.get(slide_name, False)
                logger.info(f"  [{i+1}/{len(slides_with_tiles)}] Copying {slide_name} to shared memory "
                           f"({'RGB' if is_rgb else 'grayscale'}, {channel_data.nbytes / 1e9:.2f} GB)...")

                # Copy RAM data to shared memory (fast memcpy, no CZI reading)
                shm_manager.add_slide(slide_name, channel_data)

                log_memory_status(f"After copying slide {i+1}")
            except Exception as e:
                logger.error(f"Failed to copy {slide_name} to shared memory: {e}")
                logger.error(f"Skipping {slide_name} and its tiles")
                failed_slides.add(slide_name)
                # Remove failed slide's shared memory if it was partially created
                try:
                    shm_manager.cleanup_slide(slide_name)
                except Exception:
                    pass
            finally:
                gc.collect()

        # Filter out tiles from failed slides
        if failed_slides:
            original_count = len(sampled_tiles)
            sampled_tiles = [(sn, t) for sn, t in sampled_tiles if sn not in failed_slides]
            logger.warning(f"Removed {original_count - len(sampled_tiles)} tiles from {len(failed_slides)} failed slides")
            if not sampled_tiles:
                raise RuntimeError("All slides failed to load - aborting")

        log_memory_status("All slides in shared memory")

        # Now process with GPU workers
        total_mk = 0
        total_hspc = 0

        try:
            with MultiGPUTileProcessorSHM(
                num_gpus=num_gpus,
                slide_info=shm_manager.get_slide_info(),
                mk_classifier_path=mk_classifier_path,
                hspc_classifier_path=hspc_classifier_path,
                mk_min_area=mk_min_area,
                mk_max_area=mk_max_area,
                hspc_max_area=hspc_max_area,
                variance_threshold=variance_threshold,
                calibration_block_size=calibration_block_size,
                cleanup_config=cleanup_config,
                intensity_threshold=intensity_threshold,
                modality='brightfield',
                per_slide_thresholds=precomputed_thresholds,
                # Don't pass norm params — normalization already applied at slide level
            ) as processor:
                logger.info(f"Submitting {len(sampled_tiles)} tiles to {num_gpus} GPUs...")
                for slide_name, tile in sampled_tiles:
                    processor.submit_tile(slide_name, tile)

                logger.info("Collecting results...")
                with tqdm(total=len(sampled_tiles), desc="Processing tiles (multi-GPU)") as pbar:
                    collected = 0
                    while collected < len(sampled_tiles):
                        result = processor.collect_result(timeout=300)
                        if result is None:
                            logger.error(f"Timeout waiting for results ({collected}/{len(sampled_tiles)})")
                            break
                        if result.get('status') == 'ready':
                            continue

                        # Process result (save to disk)
                        if result['status'] == 'success':
                            slide_name = result['slide_name']
                            tile = result['tile']
                            tid = result['tid']
                            output_dir = output_base / slide_name
                            output_dir.mkdir(parents=True, exist_ok=True)

                            sr = slide_results[slide_name]

                            if result.get('mk_feats'):
                                mk_dir = output_dir / "mk" / "tiles" / str(tid)
                                mk_dir.mkdir(parents=True, exist_ok=True)
                                with open(mk_dir / "window.csv", 'w') as f:
                                    f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")
                                for feat in result['mk_feats']:
                                    feat['center'][0] += tile['x']
                                    feat['center'][1] += tile['y']
                                    # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
                                    feat['uid'] = f"{slide_name}_mk_{round(feat['center'][0])}_{round(feat['center'][1])}"
                                    sr['mk_count'] += 1
                                with open(mk_dir / "features.json", 'w') as f:
                                    json.dump(result['mk_feats'], f)

                            if result.get('hspc_feats'):
                                hspc_dir = output_dir / "hspc" / "tiles" / str(tid)
                                hspc_dir.mkdir(parents=True, exist_ok=True)
                                with open(hspc_dir / "window.csv", 'w') as f:
                                    f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")
                                for feat in result['hspc_feats']:
                                    feat['center'][0] += tile['x']
                                    feat['center'][1] += tile['y']
                                    # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
                                    feat['uid'] = f"{slide_name}_hspc_{round(feat['center'][0])}_{round(feat['center'][1])}"
                                    sr['hspc_count'] += 1
                                with open(hspc_dir / "features.json", 'w') as f:
                                    json.dump(result['hspc_feats'], f)

                        del result
                        pbar.update(1)
                        collected += 1

        finally:
            logger.info("Cleaning up shared memory...")
            shm_manager.cleanup()
            # Close CZI readers to prevent file descriptor leaks
            for loader in slide_loaders.values():
                try:
                    loader.close()
                except Exception as e:
                    logger.warning(f"Error closing loader: {e}")
            del slide_loaders
            gc.collect()

        # Calculate totals
        for sr in slide_results.values():
            total_mk += sr['mk_count']
            total_hspc += sr['hspc_count']

    # =========================================================================
    # STANDARD MODE: RAM-first architecture
    # =========================================================================
    else:
        logger.info(f"\n{'='*70}")
        logger.info(f"UNIFIED SAMPLING SEGMENTATION: {len(czi_paths)} slides (RAM-FIRST)")
        logger.info(f"{'='*70}")
        logger.info(f"Step 1: Load ALL slides into RAM using CZILoader")
        logger.info(f"Step 2: Identify tissue tiles across all slides")
        logger.info(f"Step 3: Sample {sample_fraction*100:.0f}% from combined pool")
        logger.info(f"Step 4: Process with models loaded ONCE")
        logger.info(f"{'='*70}")

        # Phase 1: Load all slides into RAM and normalize if requested
        slide_data, slide_loaders = _phase1_load_slides(
            czi_paths, tile_size, overlap, channel,
            norm_method=norm_method_for_workers,
            norm_params=norm_params_for_workers
        )

        # Phase 2: Identify tissue-containing tiles
        tissue_tiles, variance_threshold, intensity_threshold = _phase2_identify_tissue_tiles(
            slide_data, tile_size, overlap, None, calibration_block_size, calibration_samples,
            precomputed_thresholds=precomputed_thresholds,
        )

        # Phase 3: Sample from combined pool
        sampled_tiles = _phase3_sample_tiles(tissue_tiles, sample_fraction)

        # Phase 4: Process sampled tiles with ML models
        total_mk, total_hspc = _phase4_process_tiles(
            sampled_tiles=sampled_tiles,
            slide_data=slide_data,
            slide_loaders=slide_loaders,
            output_base=output_base,
            mk_min_area=mk_min_area,
            mk_max_area=mk_max_area,
            variance_threshold=variance_threshold,
            calibration_block_size=calibration_block_size,
            num_workers=num_workers,
            mk_classifier_path=mk_classifier_path,
            hspc_classifier_path=hspc_classifier_path,
            html_output_dir=html_output_dir,
            samples_per_page=samples_per_page,
            mk_min_area_um=mk_min_area_um,
            mk_max_area_um=mk_max_area_um,
            hspc_min_area=hspc_min_area,
            hspc_max_area=hspc_max_area,
            multi_gpu=False,  # Don't use multi-GPU in standard mode
            num_gpus=num_gpus,
            intensity_threshold=intensity_threshold,
            per_slide_thresholds=precomputed_thresholds,
        )

        # Clear slide data and loaders from RAM
        del slide_data
        for loader in slide_loaders.values():
            loader.close()
        del slide_loaders
        gc.collect()

    logger.info(f"\n{'='*70}")
    logger.info("ALL SLIDES COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total MKs: {total_mk}")
    logger.info(f"Total HSPCs: {total_hspc}")
    logger.info(f"Output: {output_base}")


def process_tile_gpu_only(args):
    """
    GPU-only worker: receives pre-normalized tile, does only GPU work.
    CPU pre/post processing happens in main process threads.
    """
    tile, img_rgb, mk_min_area, mk_max_area, hspc_min_area, hspc_max_area, slide_name = args

    global segmenter
    tid = tile['id']

    if segmenter is None:
        return {'tid': tid, 'status': 'error', 'error': "Segmenter not initialized", 'slide_name': slide_name}

    if img_rgb.max() == 0:
        return {'tid': tid, 'status': 'empty', 'slide_name': slide_name}

    try:
        mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
            img_rgb, mk_min_area, mk_max_area, hspc_min_area, hspc_max_area,
            hspc_nuclear_only=cleanup_config.get('hspc_nuclear_only', False),
            cleanup_masks=cleanup_config.get('cleanup_masks', False),
            fill_holes=cleanup_config.get('fill_holes', True),
            pixel_size_um=cleanup_config.get('pixel_size_um', 0.1725)
        )

        # Generate crops for each detection (with optional cleanup)
        for feat in mk_feats:
            _, crop_result = process_detection_with_cleanup(
                feat, mk_masks, img_rgb, 'mk',
                cleanup_masks=cleanup_config['cleanup_masks'],
                fill_holes=cleanup_config['fill_holes'],
                pixel_size_um=cleanup_config['pixel_size_um'],
            )
            if crop_result:
                feat['crop_b64'] = crop_result['crop']
                feat['mask_b64'] = crop_result['mask']

        for feat in hspc_feats:
            _, crop_result = process_detection_with_cleanup(
                feat, hspc_masks, img_rgb, 'hspc',
                cleanup_masks=cleanup_config['cleanup_masks'],
                fill_holes=cleanup_config['fill_holes'],
                pixel_size_um=cleanup_config['pixel_size_um'],
            )
            if crop_result:
                feat['crop_b64'] = crop_result['crop']
                feat['mask_b64'] = crop_result['mask']

        return {
            'tid': tid, 'status': 'success',
            'mk_masks': mk_masks, 'hspc_masks': hspc_masks,
            'mk_feats': mk_feats, 'hspc_feats': hspc_feats,
            'tile': tile, 'slide_name': slide_name
        }
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Processing error: {e}", 'slide_name': slide_name}


def preprocess_tile_cpu(slide_data, slide_name, tile, use_pinned_memory=True):
    """
    CPU pre-processing: extract tile from RAM (already normalized at slide-level).
    Runs in ThreadPoolExecutor in main process.

    Normalization is applied at SLIDE-LEVEL in Phase 1, not per-tile.
    The slide_data['image'] contains pre-normalized data if normalization was requested.

    If use_pinned_memory=True and CUDA available, allocates output in pinned memory
    for faster CPU->GPU transfer (DMA, doesn't block CPU).
    """
    img = slide_data[slide_name]['image']
    tile_img = img[tile['y']:tile['y']+tile['h'],
                   tile['x']:tile['x']+tile['w']].copy()

    # Convert to RGB
    if tile_img.ndim == 2:
        img_rgb = np.stack([tile_img]*3, axis=-1)
    elif tile_img.shape[2] == 4:
        img_rgb = tile_img[:, :, :3]
    else:
        img_rgb = tile_img

    # NOTE: Normalization is now applied at SLIDE-LEVEL in Phase 1, not per-tile!
    # The slide_data['image'] already contains normalized data if normalization was requested.
    # This ensures all tiles from a slide use consistent slide-level statistics.

    # Use pinned memory for faster GPU transfer (double-buffering optimization)
    if use_pinned_memory and torch.cuda.is_available():
        try:
            # Pin memory for DMA transfer to GPU without blocking CPU
            # torch.from_numpy shares memory with numpy array (zero-copy)
            img_tensor = torch.from_numpy(img_rgb).pin_memory()
            # Note: img_rgb still references same memory, now pinned
        except Exception as e:
            logger.info(f"Failed to pin memory, using regular memory: {e}")

    return (tile, img_rgb, slide_name)


def save_tile_results(result, output_base, slide_results, slide_results_lock):
    """
    CPU post-processing: save masks and features to disk.
    Runs in ThreadPoolExecutor in main process.

    Processes ONE tile at a time. Thread-safe via slide_results_lock.
    """
    if result['status'] != 'success':
        return result

    slide_name = result['slide_name']
    tile = result['tile']
    tid = result['tid']
    output_dir = output_base / slide_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Thread-safe access to per-slide counters
    with slide_results_lock:
        sr = slide_results[slide_name]

        # Reserve global IDs for this tile's cells (atomic operation)
        mk_start_gid = sr['mk_gid']
        mk_count = len(result.get('mk_feats', []))
        sr['mk_gid'] += mk_count
        sr['mk_count'] += mk_count

        hspc_start_gid = sr['hspc_gid']
        hspc_count = len(result.get('hspc_feats', []))
        sr['hspc_gid'] += hspc_count
        sr['hspc_count'] += hspc_count

    # Now do the actual I/O outside the lock (allows parallelism)
    if result['mk_feats']:
        mk_dir = output_dir / "mk" / "tiles"
        mk_tile_dir = mk_dir / str(tid)
        mk_tile_dir.mkdir(parents=True, exist_ok=True)

        with open(mk_tile_dir / "window.csv", 'w') as f:
            f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

        new_mk = np.zeros_like(result['mk_masks'])
        mk_tile_cells = []
        current_gid = mk_start_gid
        for feat in result['mk_feats']:
            old_id = int(feat['id'].split('_')[1])
            new_mk[result['mk_masks'] == old_id] = current_gid
            feat['id'] = f'det_{current_gid - 1}'
            feat['global_id'] = current_gid  # Keep for backwards compatibility
            feat['center'][0] += tile['x']
            feat['center'][1] += tile['y']
            # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
            feat['uid'] = f"{slide_name}_mk_{round(feat['center'][0])}_{round(feat['center'][1])}"
            mk_tile_cells.append(current_gid)
            current_gid += 1

        with open(mk_tile_dir / "classes.csv", 'w') as f:
            for c in mk_tile_cells:
                f.write(f"{c}\n")
        with h5py.File(mk_tile_dir / "segmentation.h5", 'w') as f:
            create_hdf5_dataset(f, 'labels', new_mk[np.newaxis])
        with open(mk_tile_dir / "features.json", 'w') as f:
            json.dump(result['mk_feats'], f)

    if result['hspc_feats']:
        hspc_dir = output_dir / "hspc" / "tiles"
        hspc_tile_dir = hspc_dir / str(tid)
        hspc_tile_dir.mkdir(parents=True, exist_ok=True)

        with open(hspc_tile_dir / "window.csv", 'w') as f:
            f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

        new_hspc = np.zeros_like(result['hspc_masks'])
        hspc_tile_cells = []
        current_gid = hspc_start_gid
        for feat in result['hspc_feats']:
            old_id = int(feat['id'].split('_')[1])
            new_hspc[result['hspc_masks'] == old_id] = current_gid
            feat['id'] = f'det_{current_gid - 1}'
            feat['global_id'] = current_gid  # Keep for backwards compatibility
            feat['center'][0] += tile['x']
            feat['center'][1] += tile['y']
            # Generate spatial UID: {slide}_{celltype}_{round(x)}_{round(y)}
            feat['uid'] = f"{slide_name}_hspc_{round(feat['center'][0])}_{round(feat['center'][1])}"
            hspc_tile_cells.append(current_gid)
            current_gid += 1

        with open(hspc_tile_dir / "classes.csv", 'w') as f:
            for c in hspc_tile_cells:
                f.write(f"{c}\n")
        with h5py.File(hspc_tile_dir / "segmentation.h5", 'w') as f:
            create_hdf5_dataset(f, 'labels', new_hspc[np.newaxis])
        with open(hspc_tile_dir / "features.json", 'w') as f:
            json.dump(result['hspc_feats'], f)

    return {'slide_name': slide_name, 'mk_count': mk_count, 'hspc_count': hspc_count}


def run_pipelined_segmentation(
    czi_paths,
    output_base,
    slide_data,  # Pre-loaded slides
    sampled_tiles,  # Pre-sampled (slide_name, tile) tuples
    variance_threshold,
    mk_min_area=1000,
    mk_max_area=100000,
    hspc_min_area=25,
    hspc_max_area=100,
    num_workers=1,
    mk_classifier_path=None,
    hspc_classifier_path=None,
    preprocess_threads=None,  # CPU threads for pre-processing (default: 80% of cores)
    save_threads=None,  # CPU threads for saving (default: shares the 80% pool)
    norm_params=None,  # Reinhard normalization parameters (dict with Lab stats)
    normalization_method='none',  # Normalization method ('none', 'reinhard', 'percentile')
    intensity_threshold=220,  # Max background intensity (Otsu-derived)
):
    """
    PIPELINED segmentation: parallel CPU pre/post processing + serial GPU.

    Architecture:
      CPU ThreadPool (pre)  -->  Queue  -->  GPU Worker  -->  Queue  -->  CPU ThreadPool (save)
           N threads                          1 process                       M threads

    Thread allocation (default 80% of CPU cores):
      - Pre-process: 60% of allocated threads (extracting tiles, normalizing)
      - Post-process: 40% of allocated threads (saving HDF5, features)

    This keeps the GPU constantly fed while CPU handles I/O in parallel.
    """
    # Calculate CPU thread allocation: use 80% of available cores
    total_cores = os.cpu_count() or 8
    cpu_pool_size = int(total_cores * 0.8)  # 80% of cores

    if preprocess_threads is None:
        preprocess_threads = max(4, int(cpu_pool_size * 0.6))  # 60% for pre-processing
    if save_threads is None:
        save_threads = max(2, int(cpu_pool_size * 0.4))  # 40% for post-processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import queue
    import threading
    import shutil

    output_base = Path(output_base)

    logger.info(f"\n{'='*70}")
    logger.info("PIPELINED PROCESSING")
    logger.info(f"{'='*70}")
    logger.info(f"Pre-process threads: {preprocess_threads}")
    logger.info(f"GPU workers: {num_workers}")
    logger.info(f"Save threads: {save_threads}")
    logger.info(f"Tiles to process: {len(sampled_tiles)}")
    logger.info(f"HDF5 compression: {HDF5_COMPRESSION_NAME}")
    logger.info(f"Pinned memory: {'enabled' if torch.cuda.is_available() else 'disabled (no CUDA)'}")
    logger.info(f"{'='*70}")

    # Setup GPU worker pool
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            for i in range(num_workers):
                gpu_queue.put(i % n_gpus)

    temp_init_dir = output_base / "temp_init"
    temp_init_dir.mkdir(parents=True, exist_ok=True)
    dummy_mm_path = temp_init_dir / "dummy.dat"
    dummy_arr = np.memmap(dummy_mm_path, dtype=np.uint8, mode='w+', shape=(100, 100, 3))
    dummy_arr.flush()
    del dummy_arr

    init_args = (mk_classifier_path, hspc_classifier_path, gpu_queue,
                 str(dummy_mm_path), (100, 100, 3), np.uint8)

    # Track results per slide
    slide_results = {Path(p).stem: {'mk_count': 0, 'hspc_count': 0, 'mk_gid': 1, 'hspc_gid': 1}
                     for p in czi_paths}
    slide_results_lock = threading.Lock()

    # Queues for pipelining
    preprocess_queue = queue.Queue(maxsize=preprocess_threads * 2)  # Buffer pre-processed tiles
    save_queue = queue.Queue()  # Results waiting to be saved

    # Stats
    stats = {'preprocessed': 0, 'processed': 0, 'saved': 0}
    stats_lock = threading.Lock()

    def preprocess_worker():
        """Thread that pre-processes tiles and feeds the queue."""
        with ThreadPoolExecutor(max_workers=preprocess_threads) as executor:
            futures = []
            for slide_name, tile in sampled_tiles:
                future = executor.submit(preprocess_tile_cpu, slide_data, slide_name, tile)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    preprocess_queue.put(result)
                    with stats_lock:
                        stats['preprocessed'] += 1
                except Exception as e:
                    logger.error(f"Preprocess error: {e}")

            # Signal end
            preprocess_queue.put(None)

    def save_worker():
        """Thread that saves results using a thread pool for parallel I/O."""
        with ThreadPoolExecutor(max_workers=save_threads) as save_executor:
            pending_saves = []
            while True:
                item = save_queue.get()
                if item is None:
                    break
                # Submit save job to thread pool (pass lock for thread-safe counter access)
                future = save_executor.submit(save_tile_results, item, output_base, slide_results, slide_results_lock)
                pending_saves.append(future)
                save_queue.task_done()

                # Check completed saves
                done = [f for f in pending_saves if f.done()]
                for f in done:
                    try:
                        f.result()
                        with stats_lock:
                            stats['saved'] += 1
                    except Exception as e:
                        logger.error(f"Save error: {e}")
                    pending_saves.remove(f)

            # Wait for remaining saves
            for future in as_completed(pending_saves):
                try:
                    future.result()
                    with stats_lock:
                        stats['saved'] += 1
                except Exception as e:
                    logger.error(f"Save error: {e}")

    try:
        # Start save worker thread
        save_thread = threading.Thread(target=save_worker, daemon=True)
        save_thread.start()

        # Start preprocess thread
        preprocess_thread = threading.Thread(target=preprocess_worker, daemon=True)
        preprocess_thread.start()

        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            logger.info(f"GPU worker pool ready")

            def gpu_input_generator():
                """Generate GPU inputs from preprocess queue."""
                while True:
                    item = preprocess_queue.get()
                    if item is None:
                        break
                    tile, img_rgb, slide_name = item
                    yield (tile, img_rgb, mk_min_area, mk_max_area, hspc_min_area, hspc_max_area, slide_name)

            with tqdm(total=len(sampled_tiles), desc="Processing") as pbar:
                for result in pool.imap_unordered(process_tile_gpu_only, gpu_input_generator()):
                    with stats_lock:
                        stats['processed'] += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'pre': stats['preprocessed'],
                        'gpu': stats['processed'],
                        'save': stats['saved']
                    })

                    if result['status'] == 'success':
                        save_queue.put(result)
                    elif result['status'] == 'error':
                        logger.error(f"GPU error: {result['error']}")

                    # Cleanup large arrays
                    for key in ['mk_masks', 'hspc_masks', 'mk_feats', 'hspc_feats']:
                        if key in result:
                            del result[key]

        # Wait for saves to complete
        save_queue.put(None)  # Signal save worker to stop
        save_thread.join(timeout=60)

    finally:
        if temp_init_dir.exists():
            try:
                shutil.rmtree(temp_init_dir)
            except Exception as e:
                logger.debug(f"Failed to cleanup temp init dir: {e}")

    # Summary
    total_mk = sum(sr['mk_count'] for sr in slide_results.values())
    total_hspc = sum(sr['hspc_count'] for sr in slide_results.values())

    logger.info(f"\n{'='*70}")
    logger.info("PIPELINED PROCESSING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total MKs: {total_mk}")
    logger.info(f"Total HSPCs: {total_hspc}")

    return slide_results


def main():
    # Setup logging first so all messages are visible
    setup_logging(level="INFO", console=True)

    parser = argparse.ArgumentParser(description='Unified MK + HSPC segmentation')
    parser.add_argument('--czi-path', help='Single CZI file path')
    parser.add_argument('--czi-paths', nargs='+', help='Multiple CZI file paths (models loaded once)')
    parser.add_argument('--output-dir', required=True, help='Output directory (subdirs created per slide if multiple)')
    parser.add_argument('--mk-min-area-um', type=float, default=200,
                        help='Minimum MK area in µm² (only applies to MKs)')
    parser.add_argument('--mk-max-area-um', type=float, default=2000,
                        help='Maximum MK area in µm² (only applies to MKs)')
    parser.add_argument('--hspc-min-area-um', type=float, default=0,
                        help='Minimum HSPC area in µm² (default 0 = no minimum, recommended 20)')
    parser.add_argument('--hspc-max-area-um', type=float, default=500,
                        help='Maximum HSPC area in µm² (default 500, use 0 to disable)')
    parser.add_argument('--hspc-nuclear-only', action='store_true',
                        help='Use H&E deconvolution to detect HSPCs on nuclear (hematoxylin) channel only. '
                             'Reduces false positives from non-nuclear structures in H&E stained tissue.')
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
    parser.add_argument('--multi-gpu', action='store_true',
                        help='Enable multi-GPU mode: each GPU processes one tile at a time')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use in multi-GPU mode (default: 4)')
    parser.add_argument('--html-output-dir', type=str, default=None,
                        help='Directory for HTML export (default: output-dir/../docs)')
    parser.add_argument('--samples-per-page', type=int, default=300,
                        help='Number of cell samples per HTML page')

    # Mask cleanup options for LMD export
    parser.add_argument('--cleanup-masks', action='store_true',
                        help='Enable mask cleanup (keep largest component, fill holes)')
    parser.add_argument('--no-fill-holes', action='store_true',
                        help='Disable hole filling when using --cleanup-masks (for vessels)')
    parser.add_argument('--max-hole-fraction', type=float, default=0.5,
                        help='Max hole size to fill as fraction of mask area (default: 0.5)')

    # Cross-slide normalization options
    parser.add_argument('--normalize-slides', action='store_true',
                        help='Apply cross-slide intensity normalization (recommended for multi-slide batches)')
    parser.add_argument('--normalization-method', type=str, default='percentile',
                        choices=['percentile', 'reinhard', 'none'],
                        help='Normalization method: percentile (default), reinhard (Lab color space), or none')
    parser.add_argument('--norm-percentile-low', type=float, default=1.0,
                        help='Lower percentile for normalization (default: 1.0)')
    parser.add_argument('--norm-percentile-high', type=float, default=99.0,
                        help='Upper percentile for normalization (default: 99.0)')
    parser.add_argument('--norm-params-file', type=str, default=None,
                        help='JSON file with pre-computed normalization parameters (for parallel jobs)')

    args = parser.parse_args()

    # Build list of CZI paths
    czi_paths = []
    if args.czi_paths:
        czi_paths = [Path(p) for p in args.czi_paths]
    elif args.czi_path:
        czi_paths = [Path(args.czi_path)]
    else:
        parser.error("Must provide either --czi-path or --czi-paths")

    # Validate all paths exist
    for p in czi_paths:
        if not p.exists():
            parser.error(f"CZI file not found: {p}")

    # Convert µm² to px² using pixel size (0.1725 µm/px)
    PIXEL_SIZE_UM = 0.1725
    um_to_px_factor = PIXEL_SIZE_UM ** 2  # 0.02975625
    mk_min_area_px = int(args.mk_min_area_um / um_to_px_factor)
    mk_max_area_px = int(args.mk_max_area_um / um_to_px_factor)
    hspc_min_area_px = int(args.hspc_min_area_um / um_to_px_factor) if args.hspc_min_area_um > 0 else None
    hspc_max_area_px = int(args.hspc_max_area_um / um_to_px_factor) if args.hspc_max_area_um > 0 else None

    logger.info(f"MK area filter: {args.mk_min_area_um}-{args.mk_max_area_um} µm² = {mk_min_area_px}-{mk_max_area_px} px²")
    if hspc_min_area_px or hspc_max_area_px:
        min_str = f"{args.hspc_min_area_um}" if hspc_min_area_px else "0"
        max_str = f"{args.hspc_max_area_um}" if hspc_max_area_px else "∞"
        min_px_str = f"{hspc_min_area_px}" if hspc_min_area_px else "0"
        max_px_str = f"{hspc_max_area_px}" if hspc_max_area_px else "∞"
        logger.info(f"HSPC area filter: {min_str}-{max_str} µm² = {min_px_str}-{max_px_str} px²")
    else:
        logger.info("HSPC area filter: disabled")

    # Set HTML output directory (default: output_dir/../docs)
    html_output_dir = args.html_output_dir
    if html_output_dir is None:
        html_output_dir = Path(args.output_dir).parent / "docs"
    html_output_dir = Path(html_output_dir)

    # Configure mask cleanup (for LMD export) and HSPC nuclear-only mode
    fill_holes = not args.no_fill_holes if args.cleanup_masks else True
    set_cleanup_config(
        cleanup_masks=args.cleanup_masks,
        fill_holes=fill_holes,
        pixel_size_um=PIXEL_SIZE_UM,
        hspc_nuclear_only=args.hspc_nuclear_only,
    )
    if args.cleanup_masks:
        logger.info(f"Mask cleanup ENABLED (fill_holes={fill_holes}, max_hole_fraction={args.max_hole_fraction})")
    if args.hspc_nuclear_only:
        logger.info("HSPC nuclear-only mode ENABLED (H&E deconvolution for hematoxylin channel)")

    # Process slides
    if len(czi_paths) == 1:
        # Single slide - use output_dir directly
        run_unified_segmentation(
            czi_paths[0], args.output_dir,
            mk_min_area_px, mk_max_area_px, hspc_min_area_px, hspc_max_area_px,
            args.tile_size, args.overlap, args.sample_fraction,
            calibration_block_size=args.calibration_block_size,
            calibration_samples=args.calibration_samples,
            num_workers=args.num_workers,
            mk_classifier_path=args.mk_classifier,
            hspc_classifier_path=args.hspc_classifier,
            normalization_method=args.normalization_method if hasattr(args, 'normalize_slides') and args.normalize_slides else 'none',
            norm_params_file=args.norm_params_file if hasattr(args, 'norm_params_file') else None
        )
    else:
        # Multiple slides - load models once, process all slides
        run_multi_slide_segmentation(
            czi_paths, args.output_dir,
            mk_min_area_px, mk_max_area_px, hspc_min_area_px, hspc_max_area_px,
            args.tile_size, args.overlap, args.sample_fraction,
            calibration_block_size=args.calibration_block_size,
            calibration_samples=args.calibration_samples,
            num_workers=args.num_workers,
            mk_classifier_path=args.mk_classifier,
            hspc_classifier_path=args.hspc_classifier,
            html_output_dir=str(html_output_dir),
            samples_per_page=args.samples_per_page,
            mk_min_area_um=args.mk_min_area_um,
            mk_max_area_um=args.mk_max_area_um,
            multi_gpu=args.multi_gpu,
            num_gpus=args.num_gpus,
            normalize_slides=args.normalize_slides,
            normalization_method=args.normalization_method,
            norm_params_file=args.norm_params_file,
            norm_percentile_low=args.norm_percentile_low,
            norm_percentile_high=args.norm_percentile_high,
        )


if __name__ == "__main__":
    main()
