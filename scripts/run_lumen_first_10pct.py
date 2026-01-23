#!/usr/bin/env python3
"""
Run lumen-first vessel detection on 10% of tissue tiles.

This script:
1. Loads the CZI file
2. Uses tissue detection to find tissue tiles
3. Samples 10% of tissue tiles
4. Runs lumen_first detection on each tile
5. Collects all vessel candidates
6. Exports results to JSON
7. Generates HTML for annotation
"""

import sys
sys.path.insert(0, '/home/dude/code/vessel_seg')

import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

from segmentation.io.czi_loader import CZILoader


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


# Configuration
CZI_PATH = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
OUTPUT_DIR = Path("/home/dude/vessel_output/lumen_first_test")
PIXEL_SIZE_UM = 0.1725
TILE_SIZE = 4000
SAMPLE_FRACTION = 0.10
SMA_CHANNEL = 2  # SMA is channel 2 (0=nuc, 1=CD31, 2=SMA, 3=PM)
NUC_CHANNEL = 0  # Nuclear (488nm)
CD31_CHANNEL = 1  # CD31 (555nm)

# Channel mapping for RGB display: R=SMA (detection channel), G=CD31, B=Nuclear
CHANNEL_RGB_MAP = {
    'R': SMA_CHANNEL,   # Red = SMA (what we're detecting)
    'G': CD31_CHANNEL,  # Green = CD31 (endothelial)
    'B': NUC_CHANNEL,   # Blue = Nuclear
}


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
        pixel_size_um: Pixel size in micrometers
        min_lumen_area_um2: Minimum lumen area in um^2
        max_lumen_area_um2: Maximum lumen area in um^2
        min_ellipse_fit: Minimum ellipse fit quality (IoU, 0-1)
        max_aspect_ratio: Maximum major/minor axis ratio
        min_wall_brightness_ratio: Minimum wall/lumen intensity ratio
        wall_thickness_fraction: Wall region as fraction of lumen size

    Returns:
        List of candidate dicts with vessel info
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
        })

    return candidates


def extract_multichannel_features(vessel, channel_data, tile_x, tile_y, tile_size, loader):
    """
    Extract intensity features from all channels for a vessel.

    Args:
        vessel: Vessel candidate dict with 'outer' and 'inner' contours
        channel_data: Dict mapping channel index to full mosaic arrays
        tile_x, tile_y: Tile origin coordinates
        tile_size: Size of tile
        loader: CZILoader instance

    Returns:
        Dict with multi-channel features
    """
    features = {}

    # Get tile bounds in array coordinates
    x_start = tile_x - loader.x_start
    y_start = tile_y - loader.y_start
    x_end = min(x_start + tile_size, loader.width)
    y_end = min(y_start + tile_size, loader.height)
    x_start = max(0, x_start)
    y_start = max(0, y_start)

    if x_end <= x_start or y_end <= y_start:
        return features

    h, w = y_end - y_start, x_end - x_start

    # Create wall and lumen masks
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    lumen_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(wall_mask, [vessel['outer']], 0, 255, -1)
    if vessel['inner'] is not None:
        cv2.drawContours(wall_mask, [vessel['inner']], 0, 0, -1)
        cv2.drawContours(lumen_mask, [vessel['inner']], 0, 255, -1)

    wall_pixels_mask = wall_mask > 0
    lumen_pixels_mask = lumen_mask > 0

    channel_names = {
        SMA_CHANNEL: 'sma',
        CD31_CHANNEL: 'cd31',
        NUC_CHANNEL: 'nuc',
    }

    # Extract per-channel statistics
    for ch_idx, ch_name in channel_names.items():
        ch_data = channel_data[ch_idx]
        tile_ch = ch_data[y_start:y_end, x_start:x_end]

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

    # Adaptive crop size based on vessel diameter
    vessel_diameter_px = vessel['outer_diameter_um'] / pixel_size_um
    adaptive_crop = max(crop_size, int(vessel_diameter_px * 2.0))  # At least 2x the vessel diameter
    adaptive_crop = min(adaptive_crop, 500)  # Cap at 500px
    half_size = adaptive_crop // 2

    # Calculate crop bounds (local coordinates)
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size)
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size)

    # Skip if crop has invalid dimensions (vessel near edge)
    if x2 <= x1 or y2 <= y1:
        return None

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

    # Normalize for visibility
    crop_rgb = percentile_normalize(crop_rgb, p_low=2, p_high=98)

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

    # Draw mask contour on crop
    crop_with_contour = draw_mask_contour(
        crop_rgb,
        mask > 0,
        color=(0, 255, 0),  # Green for outer
        thickness=2
    )

    # Draw inner contour in different color
    if vessel['inner'] is not None:
        inner_mask = np.zeros_like(mask)
        cv2.drawContours(inner_mask, [inner_shifted], 0, 255, -1)
        crop_with_contour = draw_mask_contour(
            crop_with_contour,
            inner_mask > 0,
            color=(255, 100, 100),  # Red for lumen
            thickness=2
        )

    # Convert to base64
    img_b64, mime_type = image_to_base64(crop_with_contour, format='JPEG', quality=85)

    # Create UID: slide_vessel_x_y
    uid = f"{slide_name}_vessel_{int(global_x)}_{int(global_y)}"

    # Build sample dict
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
            'outer_area_px': float(vessel['outer_area_px']),
            'inner_area_px': float(vessel['inner_area_px']),
            # Multi-channel intensity features
            **vessel.get('multichannel_features', {}),
        },
        # Store contours for potential later use
        'outer_contour': vessel['outer'].tolist(),
        'inner_contour': vessel['inner'].tolist() if vessel['inner'] is not None else None,
    }

    return sample


def main():
    print("=" * 60)
    print("LUMEN-FIRST VESSEL DETECTION (10% sample)")
    print("=" * 60)
    print(f"CZI file: {CZI_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Pixel size: {PIXEL_SIZE_UM} um/px")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Sample fraction: {SAMPLE_FRACTION * 100:.0f}%")
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
    print("\nProcessing tiles with lumen-first detection...")
    all_samples = []
    total_vessels = 0

    for tile_info in tqdm(sampled_tiles, desc="Processing tiles"):
        tile_x = tile_info['x']
        tile_y = tile_info['y']

        # Extract SMA tile for detection
        tile_sma = loader.get_tile(tile_x, tile_y, TILE_SIZE, channel=SMA_CHANNEL)

        if tile_sma is None or tile_sma.size == 0:
            continue

        # Apply photobleaching correction for detection
        tile_corrected = correct_photobleaching(tile_sma)

        # Run lumen-first detection on SMA channel
        candidates = detect_lumen_first(
            tile_corrected,
            pixel_size_um=PIXEL_SIZE_UM,
            min_lumen_area_um2=50,         # ~8um diameter minimum
            max_lumen_area_um2=150000,     # Very permissive max
            min_ellipse_fit=0.40,          # Permissive on shape
            max_aspect_ratio=5.0,          # Allow oblique sections
            min_wall_brightness_ratio=1.15, # Wall must be brighter than lumen
        )

        total_vessels += len(candidates)

        # Extract RGB tile for display (R=SMA, G=CD31, B=Nuclear)
        tile_rgb = extract_rgb_tile(channel_data, tile_x, tile_y, TILE_SIZE, loader)
        if tile_rgb is None:
            tile_rgb = cv2.cvtColor(
                ((tile_corrected - tile_corrected.min()) / (tile_corrected.max() - tile_corrected.min() + 1e-8) * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )

        # Create samples for each candidate with multi-channel features
        for vessel in candidates:
            # Extract multi-channel intensity features
            mc_features = extract_multichannel_features(
                vessel, channel_data, tile_x, tile_y, TILE_SIZE, loader
            )
            # Add multi-channel features to vessel dict
            vessel['multichannel_features'] = mc_features

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
    print(f"Total samples: {len(all_samples)}")

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

    with open(detections_file, 'w') as f:
        json.dump({
            'slide_name': slide_name,
            'czi_path': str(CZI_PATH),
            'pixel_size_um': PIXEL_SIZE_UM,
            'tile_size': TILE_SIZE,
            'sample_fraction': SAMPLE_FRACTION,
            'total_tiles': len(all_tiles),
            'tissue_tiles': len(tissue_tiles),
            'sampled_tiles': len(sampled_tiles),
            'total_vessels': len(detections_json),
            'detection_method': 'lumen_first',
            'timestamp': datetime.now().isoformat(),
            'detections': detections_json,
        }, f, indent=2, cls=NumpyEncoder)

    print(f"  Saved: {detections_file}")

    # Save coordinates CSV
    csv_file = OUTPUT_DIR / "vessel_coordinates.csv"
    with open(csv_file, 'w') as f:
        f.write("uid,global_x,global_y,outer_diameter_um,inner_diameter_um,wall_thickness_um,aspect_ratio,ellipse_fit_quality,wall_lumen_ratio\n")
        for det in detections_json:
            feat = det['features']
            center = feat['global_center']
            f.write(f"{det['uid']},{center[0]},{center[1]},{feat['outer_diameter_um']:.2f},{feat['inner_diameter_um']:.2f},{feat['wall_thickness_um']:.2f},{feat['aspect_ratio']:.2f},{feat['ellipse_fit_quality']:.2f},{feat['wall_lumen_ratio']:.2f}\n")
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
    main()
