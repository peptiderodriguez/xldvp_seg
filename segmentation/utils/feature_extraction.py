"""
Batch feature extraction utilities for GPU-efficient processing.

This module provides batch processing functions for ResNet feature extraction,
improving GPU utilization by processing multiple crops at once rather than
one at a time.

Usage:
    from shared.feature_extraction import extract_resnet_features_batch

    crops = [crop1, crop2, crop3, ...]  # List of numpy arrays
    features = extract_resnet_features_batch(crops, model, transform, device, batch_size=16)
"""

import gc
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Callable
import torchvision.transforms as tv_transforms

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def preprocess_crop_for_resnet(crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a crop for ResNet feature extraction.

    Handles:
    - uint16 to uint8 conversion (CZI images are often 16-bit)
    - Grayscale to RGB conversion
    - Channel dimension normalization

    Args:
        crop: Input image crop as numpy array

    Returns:
        Preprocessed crop as uint8 RGB numpy array
    """
    if crop.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Convert uint16 to uint8 if needed (CZI images are often 16-bit)
    if crop.dtype == np.uint16:
        crop = (crop / 256).astype(np.uint8)
    elif crop.dtype != np.uint8:
        crop = crop.astype(np.uint8)

    # Ensure RGB format (3 channels)
    if crop.ndim == 2:
        crop = np.stack([crop, crop, crop], axis=-1)
    elif crop.shape[-1] != 3:
        crop = np.ascontiguousarray(crop[..., :3])

    return crop


def extract_resnet_features_batch(
    crops: List[np.ndarray],
    model: torch.nn.Module,
    transform: Callable,
    device: torch.device,
    batch_size: int = 16
) -> List[np.ndarray]:
    """
    Extract ResNet features for multiple crops in batches for GPU efficiency.

    This function processes crops in batches rather than one at a time,
    significantly improving GPU utilization and throughput.

    Args:
        crops: List of image crops as numpy arrays (any dtype/channels)
        model: ResNet model (should be nn.Sequential ending before FC layer)
        transform: Torchvision transform for preprocessing
        device: Torch device to use
        batch_size: Number of crops to process at once (default 16)

    Returns:
        List of feature vectors as numpy arrays (2048D each)

    Example:
        >>> crops = [crop1, crop2, crop3, ...]
        >>> features = extract_resnet_features_batch(crops, resnet, transform, device)
        >>> for feat in features:
        ...     print(feat.shape)  # (2048,)
    """
    if not crops:
        return []

    all_features = []

    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        batch_tensors = []
        valid_indices = []

        # Preprocess each crop
        for idx, crop in enumerate(batch_crops):
            try:
                # Preprocess crop
                processed = preprocess_crop_for_resnet(crop)

                # Convert to PIL and apply transform
                pil_img = Image.fromarray(processed, mode='RGB')
                tensor = transform(pil_img)
                batch_tensors.append(tensor)
                valid_indices.append(idx)
            except Exception as e:
                # Will be handled with zeros
                logger.debug(f"Failed to preprocess crop {idx}: {e}")

        if batch_tensors:
            # Stack into batch tensor and process
            batch_tensor = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                features = model(batch_tensor)
                features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
                features = features.cpu().numpy()

            # Map features back to correct indices
            feature_idx = 0
            for idx in range(len(batch_crops)):
                if idx in valid_indices:
                    all_features.append(features[feature_idx])
                    feature_idx += 1
                else:
                    # Return zeros for failed crops
                    all_features.append(np.zeros(2048))

            # Clear intermediate tensors to prevent memory buildup
            del batch_tensor, features
        else:
            # All crops failed, return zeros
            for _ in batch_crops:
                all_features.append(np.zeros(2048))

        # Clear GPU memory periodically (every 10 batches) to prevent OOM on long runs
        batch_num = i // batch_size
        if batch_num > 0 and batch_num % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Final cleanup after all batches processed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return all_features


def extract_resnet_features_single(
    crop: np.ndarray,
    model: torch.nn.Module,
    transform: Callable,
    device: torch.device
) -> np.ndarray:
    """
    Extract ResNet features for a single crop.

    This is the original single-crop implementation, kept for compatibility.
    For better GPU utilization, prefer extract_resnet_features_batch().

    Args:
        crop: Image crop as numpy array
        model: ResNet model (should be nn.Sequential ending before FC layer)
        transform: Torchvision transform for preprocessing
        device: Torch device to use

    Returns:
        Feature vector as numpy array (2048D)
    """
    if crop.size == 0:
        return np.zeros(2048)

    try:
        processed = preprocess_crop_for_resnet(crop)
        pil_img = Image.fromarray(processed, mode='RGB')
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(tensor).cpu().numpy().flatten()
        return features
    except Exception as e:
        logger.debug(f"Failed to extract ResNet features: {e}")
        return np.zeros(2048)


def create_resnet_transform() -> tv_transforms.Compose:
    """
    Create the standard transform for ResNet feature extraction.

    Returns:
        Torchvision Compose transform
    """
    return tv_transforms.Compose([
        tv_transforms.Resize(224),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def rgb_to_hsv_vectorized(rgb_pixels: np.ndarray) -> np.ndarray:
    """
    Vectorized RGB to HSV conversion using matplotlib.

    This is ~10-50x faster than the loop-based colorsys.rgb_to_hsv approach.

    Args:
        rgb_pixels: Array of RGB pixels, shape (N, 3), values in range [0, 255]

    Returns:
        Array of HSV values, shape (N, 3), with:
            - H in range [0, 180] (OpenCV convention)
            - S in range [0, 255]
            - V in range [0, 255]

    Example:
        >>> pixels = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        >>> hsv = rgb_to_hsv_vectorized(pixels)
        >>> print(hsv.shape)  # (3, 3)
    """
    import matplotlib.colors as mcolors

    if rgb_pixels.size == 0:
        return np.zeros((0, 3))

    # Normalize to [0, 1] for matplotlib
    rgb_normalized = rgb_pixels.astype(np.float32) / 255.0

    # Ensure correct shape (N, 3) for matplotlib
    if rgb_normalized.ndim == 1:
        rgb_normalized = rgb_normalized.reshape(1, 3)

    # Use matplotlib's vectorized conversion
    # rgb_to_hsv expects shape (..., 3) and returns same shape
    hsv_normalized = mcolors.rgb_to_hsv(rgb_normalized)

    # Convert to OpenCV-style ranges: H [0-180], S [0-255], V [0-255]
    hsv = np.zeros_like(hsv_normalized)
    hsv[:, 0] = hsv_normalized[:, 0] * 180  # H: 0-1 -> 0-180
    hsv[:, 1] = hsv_normalized[:, 1] * 255  # S: 0-1 -> 0-255
    hsv[:, 2] = hsv_normalized[:, 2] * 255  # V: 0-1 -> 0-255

    return hsv


def compute_hsv_features(masked_pixels: np.ndarray, sample_size: int = 100) -> dict:
    """
    Compute HSV color features from masked pixels using vectorized conversion.

    Args:
        masked_pixels: RGB pixel values, shape (N, 3), values in [0, 255]
        sample_size: Number of pixels to sample for speed (default 100)

    Returns:
        Dict with 'hue_mean', 'saturation_mean', 'value_mean'
    """
    if len(masked_pixels) == 0:
        return {'hue_mean': 0.0, 'saturation_mean': 0.0, 'value_mean': 0.0}

    # Sample for speed if needed
    if len(masked_pixels) > sample_size:
        indices = np.random.choice(len(masked_pixels), sample_size, replace=False)
        sample = masked_pixels[indices]
    else:
        sample = masked_pixels

    # Vectorized HSV conversion
    hsv = rgb_to_hsv_vectorized(sample)

    return {
        'hue_mean': float(np.mean(hsv[:, 0])),
        'saturation_mean': float(np.mean(hsv[:, 1])),
        'value_mean': float(np.mean(hsv[:, 2])),
    }


# =============================================================================
# MORPHOLOGICAL FEATURE EXTRACTION (Issue #7 - Consolidated from 7 files)
# =============================================================================

# =============================================================================
# Feature dimension constants for THIS MODULE's extraction functions.
#
# These are the "single-pass" dimensions — i.e., what each individual extraction
# function produces.  The full-pipeline dimensions (which double ResNet/DINOv2
# for masked + context crops, and include NMJ-specific + multi-channel features)
# are defined in segmentation/utils/config.py.
#
#   This module (feature_extraction.py)    |  config.py (full pipeline)
#   ─────────────────────────────────────  |  ────────────────────────────
#   MORPHOLOGICAL_FEATURE_COUNT = 22       |  MORPHOLOGICAL_FEATURES_COUNT = 78
#     (base extract_morphological_features)|    (22 base + NMJ-specific + multi-ch)
#   RESNET50_FEATURE_DIM = 2048            |  RESNET_EMBEDDING_DIMENSION = 4096
#     (single-pass output dim)             |    (2 x 2048: masked + context)
#   SAM2_EMBEDDING_DIM = 256               |  SAM2_EMBEDDING_DIMENSION = 256
#   (DINOv2 ViT-L/14 = 1024 per pass)     |  DINOV2_EMBEDDING_DIMENSION = 2048
#                                          |    (2 x 1024: masked + context)
#                                          |  TOTAL_FEATURES_PER_CELL = 6478
#                                          |    (78 + 256 + 4096 + 2048)
# =============================================================================
SAM2_EMBEDDING_DIM = 256
RESNET50_FEATURE_DIM = 2048
MORPHOLOGICAL_FEATURE_COUNT = 22

# Vessel-specific feature count (imported from vessel_features module)
# This is re-exported here for convenience
VESSEL_FEATURE_COUNT = 28  # Approx count, see vessel_features.py for exact list


def extract_morphological_features(mask: np.ndarray, image: np.ndarray) -> dict:
    """
    Extract MORPHOLOGICAL_FEATURE_COUNT (22) base morphological/intensity features.

    This is the single canonical implementation - previously duplicated in 7 files.
    All strategy files should import from here.

    Note: The full-pipeline "morphological" feature set (~78 features as reported in
    config.py) includes these 22 PLUS NMJ-specific features (skeleton_length, etc.)
    and multi-channel per-channel statistics added by the detection strategies.

    Args:
        mask: Binary mask as numpy array
        image: Source image (RGB or grayscale)

    Returns:
        Dict with 22 features:
          Shape (7): area, perimeter, circularity, solidity, aspect_ratio, extent,
                     equiv_diameter
          Color (8): red_mean, red_std, green_mean, green_std, blue_mean, blue_std,
                     gray_mean, gray_std
          HSV (3):   hue_mean, saturation_mean, value_mean
          Texture (4): relative_brightness, intensity_variance, dark_fraction,
                       nuclear_complexity
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

    # Intensity features — exclude zero pixels (CZI padding)
    if image.ndim == 3:
        masked_pixels = image[mask]
        # Exclude pixels where all channels are zero (CZI padding)
        valid = np.max(masked_pixels, axis=1) > 0
        masked_pixels = masked_pixels[valid]
        if len(masked_pixels) == 0:
            return {}
        red_mean, red_std = float(np.mean(masked_pixels[:, 0])), float(np.std(masked_pixels[:, 0]))
        green_mean, green_std = float(np.mean(masked_pixels[:, 1])), float(np.std(masked_pixels[:, 1]))
        blue_mean, blue_std = float(np.mean(masked_pixels[:, 2])), float(np.std(masked_pixels[:, 2]))
        gray = np.mean(masked_pixels, axis=1)
    else:
        gray = image[mask].astype(float)
        # Exclude zero pixels (CZI padding)
        gray = gray[gray > 0]
        if len(gray) == 0:
            return {}
        red_mean = green_mean = blue_mean = float(np.mean(gray))
        red_std = green_std = blue_std = float(np.std(gray))

    gray_mean, gray_std = float(np.mean(gray)), float(np.std(gray))

    # HSV features (vectorized for speed)
    if image.ndim == 3:
        hsv_feats = compute_hsv_features(masked_pixels, sample_size=100)
        hue_mean = hsv_feats['hue_mean']
        sat_mean = hsv_feats['saturation_mean']
        val_mean = hsv_feats['value_mean']
    else:
        hue_mean = sat_mean = 0.0
        val_mean = gray_mean

    # Texture features
    # Exclude zero pixels from global mean (CZI padding)
    if image.ndim == 3:
        global_valid = np.max(image, axis=2) > 0
        relative_brightness = gray_mean - float(np.mean(image[global_valid])) if global_valid.any() else 0
    else:
        global_valid = image > 0
        relative_brightness = gray_mean - float(np.mean(image[global_valid])) if global_valid.any() else 0
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
        'hue_mean': float(hue_mean),
        'saturation_mean': float(sat_mean),
        'value_mean': float(val_mean),
        'relative_brightness': float(relative_brightness),
        'intensity_variance': float(intensity_variance),
        'dark_fraction': float(dark_fraction),
        'nuclear_complexity': float(nuclear_complexity),
    }
