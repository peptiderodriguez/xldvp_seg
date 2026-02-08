#!/usr/bin/env python3
"""
Run lumen-first vessel detection on 10% of tissue tiles with multi-scale detection.

This script:
1. Loads the CZI file
2. Uses tissue detection to find tissue tiles
3. Samples 10% of tissue tiles
4. Runs lumen_first detection at multiple scales (1/16, 1/8, 1/4, 1/2, 1x)
5. Merges detections across scales using IoU-based deduplication
6. Collects all vessel candidates
7. Exports results to JSON
8. Generates HTML for annotation

Multi-scale detection ensures:
- Large vessels (>200 µm) detected at 1/16x and 1/8x scales
- Medium vessels (50-200 µm) detected at 1/4x and 1/2x scales
- Small vessels (<50 µm) detected at 1x scale
- Overlapping detections are deduplicated (IoU > 0.3 = same vessel)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

from segmentation.io.czi_loader import CZILoader
from segmentation.utils.multiscale import (
    SCALE_PARAMS,
    scale_contour,
    scale_point,
    merge_detections_across_scales,
    compute_iou_contours,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)
from segmentation.detection.tissue import calibrate_tissue_threshold, filter_tissue_tiles
from segmentation.preprocessing import correct_photobleaching
from segmentation.io.html_export import (
    export_samples_to_html,
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)


# Default configuration (overridden by CLI arguments)
_DEFAULT_CZI_PATH = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
_DEFAULT_OUTPUT_DIR = "/home/dude/vessel_output/lumen_first_test"
CZI_PATH = _DEFAULT_CZI_PATH
OUTPUT_DIR = Path(_DEFAULT_OUTPUT_DIR)
CROPS_DIR = OUTPUT_DIR / "crops"  # Save raw crops for fast HTML regeneration
PIXEL_SIZE_UM = 0.1725
TILE_SIZE = 20000  # Large tiles (3.45mm coverage) for big vessels like aorta
SAMPLE_FRACTION = 1.0  # 100% of tissue tiles (full run)
SMA_CHANNEL = 2  # SMA is channel 2 (0=nuc, 1=CD31, 2=SMA, 3=PM)
NUC_CHANNEL = 0  # Nuclear (488nm)
CD31_CHANNEL = 1  # CD31 (555nm)

# Multi-scale detection configuration
MULTI_SCALE_ENABLED = True
SCALES = [64, 32, 16, 8, 4, 2, 1]  # From coarsest to finest (1/64 max)
IOU_THRESHOLD = 0.3  # IoU threshold for deduplication

# Lumen area thresholds per scale (in um^2) - computed from SCALE_PARAMS diameters
# Area = pi * (diameter/2)^2, so min_area for min_diameter and max_area for max_diameter
SCALE_LUMEN_PARAMS = {
    # Scale 64: 1000-10000 um diameter (1-10mm) -> huge vessels like aorta
    64: {
        'min_lumen_area_um2': 500000,  # ~800um diameter min
        'max_lumen_area_um2': 100000000,  # 10mm diameter max
        'min_ellipse_fit': 0.15,  # Very permissive
        'max_aspect_ratio': 8.0,  # Very elongated ok
        'min_wall_brightness_ratio': 1.03,  # Very low threshold
        'boundary_margin_fraction': 0.01,
    },
    # Scale 32: 500-5000 um diameter -> very large arteries
    32: {
        'min_lumen_area_um2': 100000,  # ~350um diameter min
        'max_lumen_area_um2': 25000000,  # 5mm diameter max
        'min_ellipse_fit': 0.20,
        'max_aspect_ratio': 7.0,
        'min_wall_brightness_ratio': 1.04,
        'boundary_margin_fraction': 0.015,
    },
    # Scale 16: 200-3000 um diameter -> ~31416 - 7068583 um^2 area
    16: {
        'min_lumen_area_um2': 20000,  # Slightly smaller than theoretical min to catch more
        'max_lumen_area_um2': 8000000,
        'min_ellipse_fit': 0.25,  # Very permissive for large vessels
        'max_aspect_ratio': 6.0,  # Large vessels can be quite elongated
        'min_wall_brightness_ratio': 1.05,  # Lower threshold for downsampled images
        'boundary_margin_fraction': 0.02,  # 2% margin (large vessels ok near edges)
    },
    # Scale 8: 100-1000 um diameter -> ~7854 - 785398 um^2 area
    8: {
        'min_lumen_area_um2': 5000,
        'max_lumen_area_um2': 1000000,
        'min_ellipse_fit': 0.30,
        'max_aspect_ratio': 5.5,
        'min_wall_brightness_ratio': 1.08,
        'boundary_margin_fraction': 0.02,
    },
    # Scale 4: 50-300 um diameter -> ~1963 - 70686 um^2 area
    4: {
        'min_lumen_area_um2': 1500,
        'max_lumen_area_um2': 100000,
        'min_ellipse_fit': 0.32,
        'max_aspect_ratio': 5.0,
        'min_wall_brightness_ratio': 1.10,
        'boundary_margin_fraction': 0.03,
    },
    # Scale 2: 20-150 um diameter -> ~314 - 17671 um^2 area
    2: {
        'min_lumen_area_um2': 200,
        'max_lumen_area_um2': 25000,
        'min_ellipse_fit': 0.33,
        'max_aspect_ratio': 5.0,
        'min_wall_brightness_ratio': 1.10,
        'boundary_margin_fraction': 0.04,
    },
    # Scale 1: 5-75 um diameter -> ~20 - 4418 um^2 area
    1: {
        'min_lumen_area_um2': 75,  # Original value raised slightly
        'max_lumen_area_um2': 6000,
        'min_ellipse_fit': 0.35,
        'max_aspect_ratio': 5.0,
        'min_wall_brightness_ratio': 1.10,
        'boundary_margin_fraction': 0.05,  # 5% margin for small vessels
    },
}

# Channel mapping for RGB display: R=SMA (detection channel), G=CD31, B=Nuclear
CHANNEL_RGB_MAP = {
    'R': SMA_CHANNEL,   # Red = SMA (what we're detecting)
    'G': CD31_CHANNEL,  # Green = CD31 (endothelial)
    'B': NUC_CHANNEL,   # Blue = Nuclear
}


def downsample_image(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Downsample an image by a scale factor.

    Uses INTER_AREA interpolation for best quality when downsampling.

    Args:
        image: Input image (2D or 3D array)
        scale_factor: Factor to downsample by (e.g., 8 means 1/8th resolution)

    Returns:
        Downsampled image
    """
    if scale_factor == 1:
        return image

    h, w = image.shape[:2]
    new_h = h // scale_factor
    new_w = w // scale_factor

    if new_h < 1 or new_w < 1:
        return None

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_ellipse_fit_quality(contour, ellipse, img_h, img_w):
    """Compute IoU between contour and its fitted ellipse."""
    if ellipse is None or len(contour) < 5:
        return 0.0

    (cx, cy), (ma, MA), angle = ellipse
    x, y, w, h = cv2.boundingRect(contour)
    margin = 5
    x1, y1 = max(0, x - margin), max(0, y - margin)
    w2, h2 = w + 2*margin, h + 2*margin

    contour_mask = np.zeros((h2, w2), dtype=np.uint8)
    contour_shifted = contour.copy()
    contour_shifted[:, 0, 0] -= x1
    contour_shifted[:, 0, 1] -= y1
    cv2.drawContours(contour_mask, [contour_shifted], 0, 255, -1)

    ellipse_mask = np.zeros((h2, w2), dtype=np.uint8)
    ellipse_shifted = ((cx - x1, cy - y1), (ma, MA), angle)
    try:
        cv2.ellipse(ellipse_mask, ellipse_shifted, 255, -1)
    except:
        return 0.0

    intersection = np.logical_and(contour_mask > 0, ellipse_mask > 0).sum()
    union = np.logical_or(contour_mask > 0, ellipse_mask > 0).sum()

    if union == 0:
        return 0.0
    return intersection / union


def detect_lumen_first(
    tile: np.ndarray,
    pixel_size_um: float = 0.1725,
    min_lumen_area_um2: float = 50,
    max_lumen_area_um2: float = 150000,
    min_ellipse_fit: float = 0.40,
    max_aspect_ratio: float = 5.0,
    min_wall_brightness_ratio: float = 1.15,
    wall_thickness_fraction: float = 0.4,
    boundary_margin_fraction: float = 0.0,
    scale_factor: int = 1,
):
    """
    Lumen-first vessel detection: find dark lumens, validate bright SMA+ wall.

    This approach:
    1. Finds dark regions using Otsu threshold
    2. Fits ellipses to validate shape (permissive)
    3. Checks for bright wall surrounding the lumen
    4. Allows irregular shapes to pass for classifier training

    Args:
        tile: Grayscale SMA image
        pixel_size_um: Pixel size in micrometers (adjusted for scale)
        min_lumen_area_um2: Minimum lumen area in um^2
        max_lumen_area_um2: Maximum lumen area in um^2
        min_ellipse_fit: Minimum ellipse fit quality (IoU, 0-1)
        max_aspect_ratio: Maximum major/minor axis ratio
        min_wall_brightness_ratio: Minimum wall/lumen intensity ratio
        wall_thickness_fraction: Wall region as fraction of lumen size
        boundary_margin_fraction: Fraction of tile size to exclude from edges (0-0.5)
                                  Set to 0 to disable boundary filtering at detection time
        scale_factor: Scale factor at which detection is running (for metadata)

    Returns:
        List of candidate dicts with vessel info, including 'scale_detected' field
    """
    # Convert area thresholds to pixels
    min_lumen_area_px = min_lumen_area_um2 / (pixel_size_um ** 2)
    max_lumen_area_px = max_lumen_area_um2 / (pixel_size_um ** 2)

    # Normalize to uint8
    if tile.ndim == 3:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY) if tile.shape[2] == 3 else tile[:, :, 0]
    else:
        gray = tile

    if gray.dtype != np.uint8:
        img_min, img_max = gray.min(), gray.max()
        if img_max - img_min > 1e-8:
            img_norm = ((gray - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            return []
    else:
        img_norm = gray.copy()

    # Float version for intensity measurements
    img_float = gray.astype(np.float32)
    if img_float.max() > 0:
        img_float = img_float / img_float.max()

    h, w = img_float.shape[:2]

    # Find dark regions using Otsu threshold
    blurred = cv2.GaussianBlur(img_norm, (9, 9), 2.5)
    otsu_thresh, lumen_binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_OPEN, kernel_open)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lumen_binary = cv2.morphologyEx(lumen_binary, cv2.MORPH_CLOSE, kernel_small)

    # Find contours
    contours, _ = cv2.findContours(lumen_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Size filter
        if area < min_lumen_area_px or area > max_lumen_area_px:
            continue
        if len(contour) < 5:
            continue

        # Fit ellipse
        try:
            ellipse = cv2.fitEllipse(contour)
        except:
            continue

        (cx, cy), (minor_axis, major_axis), angle = ellipse

        # Aspect ratio check
        if minor_axis > 0:
            aspect_ratio = major_axis / minor_axis
        else:
            continue

        if aspect_ratio > max_aspect_ratio:
            continue

        # Ellipse fit quality (IoU)
        fit_quality = compute_ellipse_fit_quality(contour, ellipse, h, w)
        if fit_quality < min_ellipse_fit:
            continue

        # Boundary filtering (optional) - skip vessels too close to tile edges
        # This prevents partial vessels from being detected at small scales
        # At large scales (16, 8), we want to detect large vessels even near edges
        if boundary_margin_fraction > 0:
            margin_x = int(w * boundary_margin_fraction)
            margin_y = int(h * boundary_margin_fraction)
            bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
            if (bbox_x < margin_x or bbox_y < margin_y or
                (bbox_x + bbox_w) > (w - margin_x) or
                (bbox_y + bbox_h) > (h - margin_y)):
                continue

        # Check for bright wall around lumen
        lumen_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lumen_mask, [contour], 0, 255, -1)

        avg_radius = (major_axis + minor_axis) / 4
        wall_thickness = max(3, int(avg_radius * wall_thickness_fraction))

        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * wall_thickness + 1, 2 * wall_thickness + 1)
        )
        dilated_mask = cv2.dilate(lumen_mask, kernel_dilate)
        wall_mask = dilated_mask - lumen_mask

        lumen_pixels = img_float[lumen_mask > 0]
        wall_pixels = img_float[wall_mask > 0]

        if len(lumen_pixels) < 10 or len(wall_pixels) < 10:
            continue

        lumen_mean = np.mean(lumen_pixels)
        wall_mean = np.mean(wall_pixels)

        if lumen_mean > 0:
            wall_lumen_ratio = wall_mean / lumen_mean
        else:
            wall_lumen_ratio = wall_mean / 0.01

        if wall_lumen_ratio < min_wall_brightness_ratio:
            continue

        # Get outer contour from dilated mask
        outer_contours, _ = cv2.findContours(
            dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outer_contour = max(outer_contours, key=cv2.contourArea) if outer_contours else contour

        # Calculate diameters
        inner_diameter_um = (major_axis + minor_axis) / 2 * pixel_size_um
        outer_diameter_um = inner_diameter_um + 2 * wall_thickness * pixel_size_um

        # Calculate wall thickness in um
        wall_thickness_um = wall_thickness * pixel_size_um

        candidates.append({
            'outer': outer_contour,
            'inner': contour,
            'lumen_contour': contour,
            'inner_ellipse': ellipse,
            'centroid': (cx, cy),
            'inner_area_px': area,
            'outer_area_px': cv2.contourArea(outer_contour),
            'inner_diameter_um': inner_diameter_um,
            'outer_diameter_um': outer_diameter_um,
            'wall_thickness_um': wall_thickness_um,
            'aspect_ratio': aspect_ratio,
            'ellipse_fit_quality': fit_quality,
            'wall_lumen_ratio': wall_lumen_ratio,
            'lumen_mean': float(lumen_mean),
            'wall_mean': float(wall_mean),
            'detection_method': 'lumen_first',
            'scale_detected': scale_factor,  # Track which scale found this vessel
        })

    return candidates


def extract_multichannel_features(vessel, tile_channels):
    """
    Extract intensity features from all channels for a vessel.

    Args:
        vessel: Vessel candidate dict with 'outer' and 'inner' contours
        tile_channels: Dict mapping channel name to pre-extracted tile arrays

    Returns:
        Dict with multi-channel features
    """
    features = {}

    # Get tile dimensions from first channel
    first_ch = next(iter(tile_channels.values()))
    h, w = first_ch.shape[:2]

    # Create wall and lumen masks
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    lumen_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(wall_mask, [vessel['outer']], 0, 255, -1)
    if vessel['inner'] is not None:
        cv2.drawContours(wall_mask, [vessel['inner']], 0, 0, -1)
        cv2.drawContours(lumen_mask, [vessel['inner']], 0, 255, -1)

    wall_pixels_mask = wall_mask > 0
    lumen_pixels_mask = lumen_mask > 0

    # Extract per-channel statistics
    for ch_name, tile_ch in tile_channels.items():

        # Wall statistics
        wall_pixels = tile_ch[wall_pixels_mask] if wall_pixels_mask.any() else np.array([0])
        features[f'{ch_name}_wall_mean'] = float(np.mean(wall_pixels))
        features[f'{ch_name}_wall_std'] = float(np.std(wall_pixels))
        features[f'{ch_name}_wall_median'] = float(np.median(wall_pixels))
        features[f'{ch_name}_wall_max'] = float(np.max(wall_pixels))

        # Lumen statistics
        lumen_pixels = tile_ch[lumen_pixels_mask] if lumen_pixels_mask.any() else np.array([0])
        features[f'{ch_name}_lumen_mean'] = float(np.mean(lumen_pixels))
        features[f'{ch_name}_lumen_std'] = float(np.std(lumen_pixels))
        features[f'{ch_name}_lumen_median'] = float(np.median(lumen_pixels))

        # Wall/lumen contrast
        if features[f'{ch_name}_lumen_mean'] > 0:
            features[f'{ch_name}_wall_lumen_ratio'] = features[f'{ch_name}_wall_mean'] / features[f'{ch_name}_lumen_mean']
        else:
            features[f'{ch_name}_wall_lumen_ratio'] = 0.0

    # Cross-channel ratios (biologically meaningful)
    # SMA/CD31 ratio in wall (arteries have high SMA, low CD31 in wall)
    if features.get('cd31_wall_mean', 0) > 0:
        features['sma_cd31_wall_ratio'] = features['sma_wall_mean'] / features['cd31_wall_mean']
    else:
        features['sma_cd31_wall_ratio'] = 0.0

    # CD31 in lumen vs wall (endothelium at lumen boundary)
    if features.get('cd31_wall_mean', 0) > 0:
        features['cd31_lumen_wall_ratio'] = features['cd31_lumen_mean'] / features['cd31_wall_mean']
    else:
        features['cd31_lumen_wall_ratio'] = 0.0

    # Nuclear density in wall (cellularity indicator)
    features['nuc_wall_intensity'] = features.get('nuc_wall_mean', 0)

    return features


def run_multiscale_detection_on_tile(
    tile_sma: np.ndarray,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    pixel_size_um: float,
    scales: list = None,
    iou_threshold: float = 0.3,
) -> list:
    """
    Run multi-scale lumen-first detection on a single tile.

    For each scale:
    1. Downsample the SMA tile by the scale factor
    2. Run detection with scale-appropriate parameters
    3. Scale contours back to full resolution
    4. Add tile offset to get global-within-tile coordinates

    Finally, merge detections across scales to remove duplicates.

    Args:
        tile_sma: Full-resolution SMA tile (grayscale)
        tile_x: Tile X origin in global mosaic coordinates
        tile_y: Tile Y origin in global mosaic coordinates
        tile_size: Size of the tile in pixels
        pixel_size_um: Pixel size at full resolution
        scales: List of scale factors (e.g., [16, 8, 4, 2, 1])
        iou_threshold: IoU threshold for deduplication

    Returns:
        List of merged vessel candidates with full-resolution coordinates
    """
    if scales is None:
        scales = SCALES

    all_detections = []
    scale_counts = {}

    for scale in scales:
        # Get scale-specific parameters
        params = SCALE_LUMEN_PARAMS.get(scale, SCALE_LUMEN_PARAMS[1])

        # Downsample the tile
        tile_scaled = downsample_image(tile_sma, scale)
        if tile_scaled is None:
            continue

        # Adjust pixel size for scale
        scaled_pixel_size = pixel_size_um * scale

        # Run detection at this scale
        candidates = detect_lumen_first(
            tile_scaled,
            pixel_size_um=scaled_pixel_size,
            min_lumen_area_um2=params['min_lumen_area_um2'],
            max_lumen_area_um2=params['max_lumen_area_um2'],
            min_ellipse_fit=params['min_ellipse_fit'],
            max_aspect_ratio=params['max_aspect_ratio'],
            min_wall_brightness_ratio=params['min_wall_brightness_ratio'],
            boundary_margin_fraction=params['boundary_margin_fraction'],
            scale_factor=scale,
        )

        scale_counts[scale] = len(candidates)

        # Scale contours back to full resolution
        # scale_contour returns float64; cast to int32 after scaling
        for cand in candidates:
            # Scale contours
            if cand['outer'] is not None:
                cand['outer'] = scale_contour(cand['outer'], scale).astype(np.int32)
            if cand['inner'] is not None:
                cand['inner'] = scale_contour(cand['inner'], scale).astype(np.int32)
            if cand.get('lumen_contour') is not None:
                cand['lumen_contour'] = scale_contour(cand['lumen_contour'], scale).astype(np.int32)

            # Scale centroid (local to tile)
            cx, cy = cand['centroid']
            cand['centroid'] = (cx * scale, cy * scale)

            # Scale pixel-based measurements back to full-res
            cand['inner_area_px'] = cand['inner_area_px'] * (scale ** 2)
            cand['outer_area_px'] = cand['outer_area_px'] * (scale ** 2)
            # Note: um-based measurements are already correct since pixel_size was adjusted

            # Scale ellipse if present
            if cand.get('inner_ellipse') is not None:
                (ecx, ecy), (minor, major), angle = cand['inner_ellipse']
                cand['inner_ellipse'] = (
                    (ecx * scale, ecy * scale),
                    (minor * scale, major * scale),
                    angle
                )

            all_detections.append(cand)

    # Log counts per scale
    total_before_merge = len(all_detections)
    if total_before_merge > 0:
        counts_str = ', '.join([f"1/{s}x:{scale_counts.get(s, 0)}" for s in scales])
        logger.debug(f"  Tile ({tile_x}, {tile_y}): {counts_str} = {total_before_merge} total before merge")

    # Merge across scales using IoU
    if len(all_detections) > 1:
        merged = merge_detections_across_scales(
            all_detections,
            iou_threshold=iou_threshold,
            prefer_finer_scale=True  # Prefer finer scale detections
        )
    else:
        merged = all_detections

    return merged


def extract_rgb_tile(channel_data, tile_x, tile_y, tile_size, loader):
    """
    Extract RGB tile from multi-channel data.

    Args:
        channel_data: Dict mapping channel index to image arrays
        tile_x, tile_y: Tile origin coordinates
        tile_size: Size of tile
        loader: CZILoader instance (for coordinate conversion)

    Returns:
        RGB numpy array (H, W, 3) with R=SMA, G=CD31, B=Nuclear
    """
    # Get tile bounds in array coordinates
    x_start = tile_x - loader.x_start
    y_start = tile_y - loader.y_start
    x_end = min(x_start + tile_size, loader.width)
    y_end = min(y_start + tile_size, loader.height)

    # Clamp to valid range
    x_start = max(0, x_start)
    y_start = max(0, y_start)

    if x_end <= x_start or y_end <= y_start:
        return None

    # Extract each channel
    tiles = {}
    for ch_idx, ch_data in channel_data.items():
        tiles[ch_idx] = ch_data[y_start:y_end, x_start:x_end].copy()

    # Apply per-channel photobleaching correction to fix banding artifacts
    for ch_idx in tiles:
        if tiles[ch_idx].size > 0:
            tiles[ch_idx] = correct_photobleaching(tiles[ch_idx])

    # Normalize each channel to 0-255
    def normalize_channel(img):
        if img.size == 0:
            return img.astype(np.uint8)
        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img_norm = np.clip((img - p2) / (p98 - p2) * 255, 0, 255)
        else:
            img_norm = np.zeros_like(img)
        return img_norm.astype(np.uint8)

    # Create RGB: R=SMA, G=CD31, B=Nuclear
    r_ch = normalize_channel(tiles[SMA_CHANNEL])
    g_ch = normalize_channel(tiles[CD31_CHANNEL])
    b_ch = normalize_channel(tiles[NUC_CHANNEL])

    rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
    return rgb


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


def create_vessel_sample(vessel, tile_img, tile_x, tile_y, slide_name, pixel_size_um, crop_size=200):
    """
    Create a sample dict for HTML export.

    Args:
        vessel: Vessel candidate dict from detection
        tile_img: Tile image (grayscale or RGB)
        tile_x, tile_y: Tile origin in global coordinates
        slide_name: Name of the slide
        pixel_size_um: Pixel size in micrometers
        crop_size: Size of crop in pixels (will be adaptive based on vessel size)

    Returns:
        Sample dict for HTML export
    """
    h, w = tile_img.shape[:2]

    # Get centroid in local tile coordinates
    cx, cy = int(vessel['centroid'][0]), int(vessel['centroid'][1])

    # Calculate global coordinates
    global_x = tile_x + cx
    global_y = tile_y + cy

    # Get outer contour bounding box
    outer_bbox = cv2.boundingRect(vessel['outer'])
    bbox_x, bbox_y, bbox_w, bbox_h = outer_bbox

    # Check if vessel touches tile boundaries (incomplete mask)
    margin = 5  # Pixels from edge to consider "touching boundary"
    if bbox_x < margin or bbox_y < margin or (bbox_x + bbox_w) > (w - margin) or (bbox_y + bbox_h) > (h - margin):
        return None  # Skip boundary vessels with incomplete masks

    # Adaptive crop size: 100% larger than outer mask (2x the bounding box)
    mask_size = max(bbox_w, bbox_h)
    adaptive_crop = int(mask_size * 2.0)  # 100% padding = 2x mask size
    adaptive_crop = max(crop_size, adaptive_crop)  # At least minimum crop size
    adaptive_crop = min(adaptive_crop, 800)  # Cap at 800px (increased from 500)
    half_size = adaptive_crop // 2

    # Calculate crop bounds (local coordinates) centered on vessel centroid
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size)
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size)

    # Skip if crop has invalid dimensions (vessel near edge)
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < mask_size or (y2 - y1) < mask_size:
        return None  # Skip if we can't fit the full mask

    # Extract crop
    if tile_img.ndim == 2:
        crop = tile_img[y1:y2, x1:x2]
        # Normalize to uint8 for OpenCV
        crop_min, crop_max = crop.min(), crop.max()
        if crop_max > crop_min:
            crop_uint8 = ((crop - crop_min) / (crop_max - crop_min) * 255).astype(np.uint8)
        else:
            crop_uint8 = np.zeros_like(crop, dtype=np.uint8)
        crop_rgb = cv2.cvtColor(crop_uint8, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = tile_img[y1:y2, x1:x2].copy()

    # Normalize each channel independently for visibility
    for c in range(3):
        ch = crop_rgb[:, :, c].astype(np.float32)
        p2, p98 = np.percentile(ch, (2, 98))
        if p98 > p2:
            crop_rgb[:, :, c] = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        else:
            crop_rgb[:, :, c] = 0

    # Create UID early for saving crop
    uid = f"{slide_name}_vessel_{int(global_x)}_{int(global_y)}"

    # Save raw crop (before contours) for fast HTML regeneration
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    crop_path = CROPS_DIR / f"{uid}.jpg"
    cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Create mask for contour drawing
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

    # Shift outer contour to crop coordinates
    outer_shifted = vessel['outer'].copy()
    outer_shifted[:, 0, 0] -= x1
    outer_shifted[:, 0, 1] -= y1
    cv2.drawContours(mask, [outer_shifted], 0, 255, -1)

    # If inner contour exists, subtract it to show wall only
    if vessel['inner'] is not None:
        inner_shifted = vessel['inner'].copy()
        inner_shifted[:, 0, 0] -= x1
        inner_shifted[:, 0, 1] -= y1
        cv2.drawContours(mask, [inner_shifted], 0, 0, -1)

    # Draw mask contour on crop (white solid for outer wall)
    crop_with_contour = draw_mask_contour(
        crop_rgb,
        mask > 0,
        color=(255, 255, 255),  # White for outer
        thickness=10,
        dotted=False
    )

    # Draw inner contour (white dotted for lumen)
    if vessel['inner'] is not None:
        inner_mask = np.zeros_like(mask)
        cv2.drawContours(inner_mask, [inner_shifted], 0, 255, -1)
        crop_with_contour = draw_mask_contour(
            crop_with_contour,
            inner_mask > 0,
            color=(255, 255, 255),  # White for lumen
            thickness=10,
            dotted=False  # Solid line same as outer
        )

    # Convert to base64
    img_b64, mime_type = image_to_base64(crop_with_contour, format='JPEG', quality=85)

    # Build sample dict (uid already created above for crop saving)
    sample = {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime_type,
        'stats': {
            'area_um2': vessel['outer_area_px'] * (pixel_size_um ** 2),
            'area_px': vessel['outer_area_px'],
            'inner_diameter_um': vessel['inner_diameter_um'],
            'outer_diameter_um': vessel['outer_diameter_um'],
            'wall_thickness_um': vessel['wall_thickness_um'],
            'solidity': vessel['ellipse_fit_quality'],
            'aspect_ratio': vessel['aspect_ratio'],
            'wall_lumen_ratio': vessel['wall_lumen_ratio'],
            'scale_detected': vessel.get('scale_detected', 1),  # For display in HTML
        },
        # Additional metadata for JSON export
        'features': {
            'global_center': [global_x, global_y],
            'local_center': [cx, cy],
            'tile_origin': [tile_x, tile_y],
            'inner_diameter_um': vessel['inner_diameter_um'],
            'outer_diameter_um': vessel['outer_diameter_um'],
            'wall_thickness_um': vessel['wall_thickness_um'],
            'aspect_ratio': vessel['aspect_ratio'],
            'ellipse_fit_quality': vessel['ellipse_fit_quality'],
            'wall_lumen_ratio': vessel['wall_lumen_ratio'],
            'lumen_mean': vessel['lumen_mean'],
            'wall_mean': vessel['wall_mean'],
            'detection_method': vessel['detection_method'],
            'scale_detected': vessel.get('scale_detected', 1),  # Track which scale found this vessel
            'outer_area_px': float(vessel['outer_area_px']),
            'inner_area_px': float(vessel['inner_area_px']),
            # Multi-channel intensity features
            **vessel.get('multichannel_features', {}),
        },
        # Store contours for potential later use (global coords)
        'outer_contour': vessel['outer'].tolist(),
        'inner_contour': vessel['inner'].tolist() if vessel['inner'] is not None else None,
        # Store shifted contours (crop-relative coords) for HTML regeneration
        'outer_contour_shifted': outer_shifted.tolist(),
        'inner_contour_shifted': inner_shifted.tolist() if vessel['inner'] is not None else None,
        'crop_offset': [x1, y1],  # Offset within tile
        'crop_size': [x2 - x1, y2 - y1],  # Crop dimensions
    }

    return sample


def main():
    print("=" * 60)
    if MULTI_SCALE_ENABLED:
        print(f"MULTI-SCALE LUMEN-FIRST VESSEL DETECTION ({SAMPLE_FRACTION*100:.0f}% sample)")
        print(f"Scales: {SCALES} (coarse to fine)")
    else:
        print(f"LUMEN-FIRST VESSEL DETECTION ({SAMPLE_FRACTION*100:.0f}% sample)")
    print("=" * 60)
    print(f"CZI file: {CZI_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Pixel size: {PIXEL_SIZE_UM} um/px")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Sample fraction: {SAMPLE_FRACTION * 100:.0f}%")
    if MULTI_SCALE_ENABLED:
        print(f"IoU threshold for deduplication: {IOU_THRESHOLD}")
        print("\nScale-specific parameters:")
        for scale in SCALES:
            params = SCALE_LUMEN_PARAMS[scale]
            print(f"  1/{scale}x: lumen area {params['min_lumen_area_um2']:.0f}-{params['max_lumen_area_um2']:.0f} um^2, "
                  f"ellipse_fit>{params['min_ellipse_fit']:.2f}, aspect<{params['max_aspect_ratio']:.1f}")
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_dir = OUTPUT_DIR / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    # Extract slide name
    slide_name = Path(CZI_PATH).stem

    # Load CZI - all 3 channels for RGB display
    print("Loading CZI file (3 channels for RGB display)...")

    # Load each channel separately
    loader_sma = CZILoader(CZI_PATH, load_to_ram=True, channel=SMA_CHANNEL, quiet=False)
    print("Loading CD31 channel...")
    loader_cd31 = CZILoader(CZI_PATH, load_to_ram=True, channel=CD31_CHANNEL, quiet=True)
    print("Loading nuclear channel...")
    loader_nuc = CZILoader(CZI_PATH, load_to_ram=True, channel=NUC_CHANNEL, quiet=True)

    # Use SMA loader as primary (for tissue detection)
    loader = loader_sma

    # Store all channel data
    channel_data = {
        SMA_CHANNEL: loader_sma.channel_data,
        CD31_CHANNEL: loader_cd31.channel_data,
        NUC_CHANNEL: loader_nuc.channel_data,
    }

    mosaic_info = {
        'x': loader.x_start,
        'y': loader.y_start,
        'width': loader.width,
        'height': loader.height,
    }

    print(f"  Mosaic size: {loader.width:,} x {loader.height:,} px")
    print(f"  Origin: ({loader.x_start}, {loader.y_start})")

    # Generate tile grid
    print("\nGenerating tile grid...")
    all_tiles = generate_tile_grid(mosaic_info, TILE_SIZE)
    print(f"  Total tiles: {len(all_tiles)}")

    # Calibrate tissue threshold
    print("\nCalibrating tissue threshold...")
    variance_threshold = calibrate_tissue_threshold(
        all_tiles,
        calibration_samples=min(50, len(all_tiles)),
        channel=SMA_CHANNEL,
        tile_size=TILE_SIZE,
        image_array=loader.channel_data,
    )

    # Filter to tissue tiles
    print("\nFiltering to tissue-containing tiles...")
    tissue_tiles = filter_tissue_tiles(
        all_tiles,
        variance_threshold,
        channel=SMA_CHANNEL,
        tile_size=TILE_SIZE,
        image_array=loader.channel_data,
    )
    print(f"  Tissue tiles: {len(tissue_tiles)} ({len(tissue_tiles)/len(all_tiles)*100:.1f}%)")

    if len(tissue_tiles) == 0:
        print("ERROR: No tissue tiles found!")
        return

    # Sample tiles
    n_sample = max(1, int(len(tissue_tiles) * SAMPLE_FRACTION))
    sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
    sampled_tiles = [tissue_tiles[i] for i in sample_indices]

    print(f"\nSampled {len(sampled_tiles)} tiles ({SAMPLE_FRACTION*100:.0f}% of {len(tissue_tiles)} tissue tiles)")

    # Process tiles
    if MULTI_SCALE_ENABLED:
        print(f"\nProcessing tiles with MULTI-SCALE lumen-first detection (scales: {SCALES})...")
    else:
        print("\nProcessing tiles with lumen-first detection...")
    all_samples = []
    total_vessels = 0
    total_vessels_before_merge = 0
    scale_detection_counts = {s: 0 for s in SCALES}  # Track detections per scale

    for tile_info in tqdm(sampled_tiles, desc="Processing tiles"):
        tile_x = tile_info['x']
        tile_y = tile_info['y']

        # Extract SMA tile for detection
        tile_sma = loader.get_tile(tile_x, tile_y, TILE_SIZE, channel=SMA_CHANNEL)

        if tile_sma is None or tile_sma.size == 0:
            continue

        # Apply photobleaching correction for detection
        tile_corrected = correct_photobleaching(tile_sma)

        if MULTI_SCALE_ENABLED:
            # Run multi-scale detection
            candidates = run_multiscale_detection_on_tile(
                tile_corrected,
                tile_x,
                tile_y,
                TILE_SIZE,
                pixel_size_um=PIXEL_SIZE_UM,
                scales=SCALES,
                iou_threshold=IOU_THRESHOLD,
            )
            # Track which scales contributed
            for cand in candidates:
                scale = cand.get('scale_detected', 1)
                if scale in scale_detection_counts:
                    scale_detection_counts[scale] += 1
        else:
            # Original single-scale detection
            candidates = detect_lumen_first(
                tile_corrected,
                pixel_size_um=PIXEL_SIZE_UM,
                min_lumen_area_um2=75,         # ~10um diameter minimum (slightly raised)
                max_lumen_area_um2=300000,     # Large vessels max
                min_ellipse_fit=0.35,          # Moderate shape requirement
                max_aspect_ratio=5.0,          # Allow oblique sections
                min_wall_brightness_ratio=1.10, # Moderate contrast requirement
                scale_factor=1,
            )

        total_vessels += len(candidates)

        # Log candidate count for debugging
        if len(candidates) > 50:
            print(f"  Tile ({tile_x}, {tile_y}): {len(candidates)} candidates (many!)")

        # Extract RGB tile for display (R=SMA, G=CD31, B=Nuclear)
        tile_rgb = extract_rgb_tile(channel_data, tile_x, tile_y, TILE_SIZE, loader)
        if tile_rgb is None:
            tile_rgb = cv2.cvtColor(
                ((tile_corrected - tile_corrected.min()) / (tile_corrected.max() - tile_corrected.min() + 1e-8) * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )

        # Pre-extract tile arrays for all channels ONCE (optimization)
        x_start = tile_x - loader.x_start
        y_start = tile_y - loader.y_start
        x_end = min(x_start + TILE_SIZE, loader.width)
        y_end = min(y_start + TILE_SIZE, loader.height)
        x_start = max(0, x_start)
        y_start = max(0, y_start)

        tile_channels = {
            'sma': channel_data[SMA_CHANNEL][y_start:y_end, x_start:x_end],
            'cd31': channel_data[CD31_CHANNEL][y_start:y_end, x_start:x_end],
            'nuc': channel_data[NUC_CHANNEL][y_start:y_end, x_start:x_end],
        }

        # Create samples for each candidate with multi-channel features
        print(f"  Processing {len(candidates)} vessels for tile ({tile_x}, {tile_y})...", flush=True)
        for i, vessel in enumerate(candidates):
            # Extract multi-channel intensity features (uses pre-extracted tiles)
            mc_features = extract_multichannel_features(vessel, tile_channels)
            # Add multi-channel features to vessel dict
            vessel['multichannel_features'] = mc_features

            if (i + 1) % 50 == 0 or i == 0:
                print(f"    Vessel {i+1}/{len(candidates)}...", flush=True)
            sample = create_vessel_sample(
                vessel,
                tile_rgb,  # Pass RGB tile for display
                tile_x,
                tile_y,
                slide_name,
                PIXEL_SIZE_UM
            )
            if sample is not None:  # Skip vessels near tile edges
                all_samples.append(sample)

    print(f"\n{'=' * 60}")
    print(f"DETECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total vessels detected: {total_vessels}")
    print(f"Total samples (after boundary filtering): {len(all_samples)}")

    if MULTI_SCALE_ENABLED:
        print(f"\nDetections by scale (after merge):")
        for scale in SCALES:
            count = scale_detection_counts[scale]
            pct = (count / total_vessels * 100) if total_vessels > 0 else 0
            target = SCALE_PARAMS.get(scale, {}).get('description', f'scale 1/{scale}x')
            print(f"  1/{scale}x: {count:5d} ({pct:5.1f}%) - {target}")

    if len(all_samples) == 0:
        print("No vessels found!")
        return

    # Sort samples by diameter (largest first)
    all_samples.sort(key=lambda x: x['stats']['outer_diameter_um'], reverse=True)

    # Calculate stats
    diameters = [s['stats']['outer_diameter_um'] for s in all_samples]
    print(f"\nDiameter statistics:")
    print(f"  Mean: {np.mean(diameters):.1f} um")
    print(f"  Median: {np.median(diameters):.1f} um")
    print(f"  Min: {np.min(diameters):.1f} um")
    print(f"  Max: {np.max(diameters):.1f} um")
    print(f"  Std: {np.std(diameters):.1f} um")

    # Save detections JSON
    print("\nSaving detections JSON...")
    detections_file = OUTPUT_DIR / "vessel_detections.json"

    # Prepare detections for JSON (remove image data, keep features)
    detections_json = []
    for sample in all_samples:
        detection = {
            'uid': sample['uid'],
            'features': sample['features'],
            'stats': sample['stats'],
            'outer_contour': sample['outer_contour'],
            'inner_contour': sample['inner_contour'],
        }
        detections_json.append(detection)

    # Build metadata dict
    json_metadata = {
        'slide_name': slide_name,
        'czi_path': str(CZI_PATH),
        'pixel_size_um': PIXEL_SIZE_UM,
        'tile_size': TILE_SIZE,
        'sample_fraction': SAMPLE_FRACTION,
        'total_tiles': len(all_tiles),
        'tissue_tiles': len(tissue_tiles),
        'sampled_tiles': len(sampled_tiles),
        'total_vessels': len(detections_json),
        'detection_method': 'lumen_first_multiscale' if MULTI_SCALE_ENABLED else 'lumen_first',
        'timestamp': datetime.now().isoformat(),
    }

    # Add multi-scale specific metadata
    if MULTI_SCALE_ENABLED:
        json_metadata['multiscale'] = {
            'enabled': True,
            'scales': SCALES,
            'iou_threshold': IOU_THRESHOLD,
            'detections_per_scale': {str(s): scale_detection_counts[s] for s in SCALES},
            'scale_params': {str(s): SCALE_LUMEN_PARAMS[s] for s in SCALES},
        }

    json_metadata['detections'] = detections_json

    with open(detections_file, 'w') as f:
        json.dump(json_metadata, f, indent=2, cls=NumpyEncoder)

    print(f"  Saved: {detections_file}")

    # Save coordinates CSV
    csv_file = OUTPUT_DIR / "vessel_coordinates.csv"
    with open(csv_file, 'w') as f:
        f.write("uid,global_x,global_y,outer_diameter_um,inner_diameter_um,wall_thickness_um,aspect_ratio,ellipse_fit_quality,wall_lumen_ratio,scale_detected\n")
        for det in detections_json:
            feat = det['features']
            center = feat['global_center']
            scale = feat.get('scale_detected', 1)
            f.write(f"{det['uid']},{center[0]},{center[1]},{feat['outer_diameter_um']:.2f},{feat['inner_diameter_um']:.2f},{feat['wall_thickness_um']:.2f},{feat['aspect_ratio']:.2f},{feat['ellipse_fit_quality']:.2f},{feat['wall_lumen_ratio']:.2f},{scale}\n")
    print(f"  Saved: {csv_file}")

    # Generate HTML
    print("\nGenerating HTML annotation pages...")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_samples, n_pages = export_samples_to_html(
        samples=all_samples,
        output_dir=html_dir,
        cell_type='vessel',
        samples_per_page=300,
        title="Vessel Detection (Lumen-First)",
        subtitle=f"10% sample from {slide_name}",
        file_name=slide_name,
        pixel_size_um=PIXEL_SIZE_UM,
        tiles_processed=len(sampled_tiles),
        tiles_total=len(all_tiles),
        tissue_tiles=len(tissue_tiles),
        timestamp=timestamp,
        experiment_name="lumen_first_test",
        channel_legend={"red": "SMA (647)", "green": "CD31 (555)", "blue": "Nuclear (488)"},
    )

    print(f"\n{'=' * 60}")
    print(f"EXPORT COMPLETE")
    print(f"{'=' * 60}")
    print(f"Detections: {detections_file}")
    print(f"Coordinates: {csv_file}")
    print(f"HTML viewer: {html_dir / 'index.html'}")
    print(f"Total samples: {n_samples}")
    print(f"Total pages: {n_pages}")

    # Cleanup
    loader.close()
    print("\nDone!")


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Multi-scale lumen-first vessel detection")
    _parser.add_argument('--czi-path', type=str, default=_DEFAULT_CZI_PATH,
                         help=f'Path to CZI file (default: {_DEFAULT_CZI_PATH})')
    _parser.add_argument('--output-dir', type=str, default=_DEFAULT_OUTPUT_DIR,
                         help=f'Output directory (default: {_DEFAULT_OUTPUT_DIR})')
    _args = _parser.parse_args()

    # Override module-level globals from CLI
    CZI_PATH = _args.czi_path
    OUTPUT_DIR = Path(_args.output_dir)
    CROPS_DIR = OUTPUT_DIR / "crops"

    main()
