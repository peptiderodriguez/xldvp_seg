#!/usr/bin/env python3
"""
SAM2-based vessel detection at 1/8 scale.

Approach:
- Load channels at 1/2 scale into RAM (4x smaller than full-res)
- Process at 1/8 scale (downsample 4x from 1/2 base)
- No diameter filters - detect all ring structures, filter in post-processing
- 4000px tiles cover 32000px at full res

Pipeline:
1. SAM2 generates mask proposals on SMA channel
2. Filter to lumens (dark inside, bright SMA wall around)
3. Watershed expand from lumen to find outer wall
4. Extract vessel measurements + save crops with contours

Output:
- vessel_detections_multiscale.json - all vessels with coordinates and measurements
- crops/ - vessel images with green (outer) and cyan (inner) contours
- index.html - viewer sorted by diameter
"""

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
# watershed removed - using dilate_until_signal_drops instead
from skimage.transform import resize
import json
import os
import sys
import random
import gc
import torch
from tqdm import tqdm

sys.path.insert(0, '/home/dude/code/vessel_seg')
from segmentation.io.czi_loader import CZILoader
from segmentation.preprocessing.illumination import correct_photobleaching

# Configuration
CZI_PATH = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
OUTPUT_DIR = "/home/dude/vessel_output/sam2_multiscale"
SAMPLE_FRACTION = 1.0  # 100% of tiles
TILE_OVERLAP = 0.5  # 50% overlap between tiles
TISSUE_VARIANCE_THRESHOLD = 50

# Base scale for RAM loading (finest scale we need)
# All coarser scales will downsample from this
BASE_SCALE = 2  # 1/2 scale

# Single scale configuration - 1/8 scale, no diameter filter
# 4000px tiles at 1/8 scale = 32000px coverage at full res
TILE_SIZE = 4000  # Default tile size
SCALES = [
    {'name': '1/64', 'scale_factor': 64, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1000},
    {'name': '1/32', 'scale_factor': 32, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1200},
    {'name': '1/16', 'scale_factor': 16, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1400},
    {'name': '1/8', 'scale_factor': 8, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 1700},
    {'name': '1/4', 'scale_factor': 4, 'min_diam_um': 0, 'max_diam_um': 999999, 'tile_size': 2000},
]


class DownsampledChannelCache:
    """
    Holds channels loaded at BASE_SCALE (1/2), provides tiles at any coarser scale.

    Memory usage: ~4x smaller than full-res (for BASE_SCALE=2).
    """

    def __init__(self, loader: CZILoader, channels: list, base_scale: int = 2, strip_height: int = 5000):
        """
        Load channels at base_scale into RAM using strip-by-strip loading.

        Args:
            loader: CZILoader instance (used for reading, not for RAM storage)
            channels: List of channel indices to load
            base_scale: Scale factor for base resolution (2 = 1/2 scale)
            strip_height: Height of strips for loading (in full-res pixels)
        """
        self.base_scale = base_scale
        self.channels = {}
        self.full_res_size = loader.mosaic_size  # (width, height) at full res
        self.loader = loader  # Keep reference for strip reading

        # Size at base scale
        self.base_width = self.full_res_size[0] // base_scale
        self.base_height = self.full_res_size[1] // base_scale

        print(f"\nLoading channels at 1/{base_scale} scale ({self.base_width} x {self.base_height})...")
        print(f"  (Full res: {self.full_res_size[0]} x {self.full_res_size[1]})")

        for ch in channels:
            print(f"  Loading channel {ch}...", flush=True)
            self.channels[ch] = self._load_channel_in_strips(ch, strip_height)
            mem_gb = self.channels[ch].nbytes / (1024**3)
            print(f"    Channel {ch}: {self.channels[ch].shape}, {mem_gb:.2f} GB")

        total_gb = sum(arr.nbytes for arr in self.channels.values()) / (1024**3)
        print(f"  Total RAM: {total_gb:.2f} GB")

    def _load_channel_in_strips(self, channel: int, strip_height: int) -> np.ndarray:
        """Load a channel in strips, downsampling each strip manually."""
        # Pre-allocate output array at base scale
        channel_array = np.empty((self.base_height, self.base_width), dtype=np.uint16)

        n_strips = (self.loader.height + strip_height - 1) // strip_height

        for i in tqdm(range(n_strips), desc=f"Loading ch{channel}"):
            # Full-res strip coordinates
            y_off = i * strip_height
            h = min(strip_height, self.loader.height - y_off)

            # Read strip at FULL resolution (scale_factor=1 to avoid aicspylibczi bug)
            strip = self.loader.reader.read_mosaic(
                region=(self.loader.x_start, self.loader.y_start + y_off, self.loader.width, h),
                scale_factor=1,
                C=channel
            )
            strip = np.squeeze(strip)

            # Manually downsample to base_scale using cv2.resize
            target_h = h // self.base_scale
            target_w = self.loader.width // self.base_scale
            if target_h > 0 and target_w > 0:
                strip_downsampled = cv2.resize(strip, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                continue

            # Calculate output position at base scale
            y_out = y_off // self.base_scale

            # Handle edge case where strip might be slightly smaller
            actual_h = min(strip_downsampled.shape[0], self.base_height - y_out)
            actual_w = min(strip_downsampled.shape[1], self.base_width)
            if actual_h > 0:
                channel_array[y_out:y_out + actual_h, :actual_w] = strip_downsampled[:actual_h, :actual_w]

            del strip, strip_downsampled
            gc.collect()

        return channel_array

    def get_tile(self, tile_x: int, tile_y: int, tile_size: int, channel: int, scale_factor: int) -> np.ndarray:
        """
        Get a tile at the specified scale.

        Args:
            tile_x, tile_y: Tile origin in SCALED coordinates (at target scale_factor)
            tile_size: Output tile size (e.g., 5000)
            channel: Channel index
            scale_factor: Target scale (must be >= base_scale)

        Returns:
            Tile data at target scale, shape (tile_size, tile_size) or smaller at edges
        """
        if scale_factor < self.base_scale:
            raise ValueError(f"scale_factor {scale_factor} < base_scale {self.base_scale}")

        if channel not in self.channels:
            raise ValueError(f"Channel {channel} not loaded")

        # Convert tile coords from target scale to base scale
        # tile_x is in target-scale space, need to convert to base-scale space
        relative_scale = scale_factor // self.base_scale

        # In target scale: tile covers tile_size pixels
        # In full res: tile covers tile_size * scale_factor pixels
        # In base scale: tile covers tile_size * scale_factor / base_scale = tile_size * relative_scale
        base_tile_size = tile_size * relative_scale

        # Convert tile origin from target-scale coords to base-scale coords
        # tile_x (target) * scale_factor = full_res_x
        # full_res_x / base_scale = base_x
        base_x = (tile_x * scale_factor) // self.base_scale
        base_y = (tile_y * scale_factor) // self.base_scale

        # Extract region from base-scale data
        data = self.channels[channel]
        y2 = min(base_y + base_tile_size, data.shape[0])
        x2 = min(base_x + base_tile_size, data.shape[1])

        if base_x >= data.shape[1] or base_y >= data.shape[0]:
            return None

        region = data[base_y:y2, base_x:x2]

        if region.size == 0:
            return None

        # Downsample from base scale to target scale if needed
        if relative_scale > 1:
            target_h = region.shape[0] // relative_scale
            target_w = region.shape[1] // relative_scale
            if target_h == 0 or target_w == 0:
                return None
            region = cv2.resize(region, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return region

    def release(self):
        """Release all channel data."""
        self.channels.clear()
        gc.collect()

# Channel indices
NUCLEAR = 0
CD31 = 1
SMA = 2
PM = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'tiles'), exist_ok=True)

# Clear crops folder at start of each run to avoid mixing with previous runs
crops_dir = os.path.join(OUTPUT_DIR, 'crops')
if os.path.exists(crops_dir):
    import shutil
    shutil.rmtree(crops_dir)
os.makedirs(crops_dir, exist_ok=True)

CROP_SIZE = 400  # pixels around vessel center for crops

def normalize_channel(arr, p_low=1, p_high=99):
    """Normalize to uint8 using percentile clipping (robust to outliers)."""
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, (p_low, p_high))
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-8) * 255, 0, 255)
    return arr.astype(np.uint8)

def save_vessel_crop(vessel, display_rgb, outer_contour, inner_contour, tile_x, tile_y, tile_size, scale_factor):
    """Save a crop of the vessel with contours drawn."""
    cx, cy = vessel['local_center']

    # Crop region in tile image coordinates
    half = CROP_SIZE // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(display_rgb.shape[1], cx + half)
    y2 = min(display_rgb.shape[0], cy + half)

    # Check for valid crop
    if x2 <= x1 or y2 <= y1:
        return None

    crop = display_rgb[y1:y2, x1:x2].copy()

    if crop.size == 0:
        return None

    # Draw contours on crop (translate to crop coordinates)
    # Contours are in tile coordinates, need to offset by crop origin
    outer_in_crop = outer_contour - np.array([x1, y1])
    inner_in_crop = inner_contour - np.array([x1, y1])

    cv2.drawContours(crop, [outer_in_crop], -1, (0, 255, 0), 2)  # Green for outer wall
    cv2.drawContours(crop, [inner_in_crop], -1, (0, 255, 255), 2)  # Cyan for inner lumen

    # Store crop offset for reference
    vessel['crop_offset'] = [x1, y1]
    vessel['crop_scale_factor'] = scale_factor

    crop_path = os.path.join(OUTPUT_DIR, 'crops', f"{vessel['uid']}.jpg")
    cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    return crop_path

def is_tissue_tile(tile, threshold=TISSUE_VARIANCE_THRESHOLD):
    return np.var(tile) > threshold

def verify_lumen_multichannel(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm, scale_factor=1):
    """Verify region is true lumen by checking multi-channel emptiness."""
    area = mask.sum()
    # Scale area thresholds: base values for 1/1 scale, adjust for current scale
    # At coarser scales, fewer pixels represent same physical area
    min_area = 50 // (scale_factor * scale_factor) if scale_factor > 1 else 50
    max_area = 500000 // (scale_factor * scale_factor) if scale_factor > 1 else 500000
    min_area = max(10, min_area)  # floor to avoid rejecting everything

    if area < min_area or area > max_area:
        return False, {}

    sma_inside = sma_norm[mask].mean()
    nuclear_inside = nuclear_norm[mask].mean()

    # Scale dilation kernel: 15px at 1/1 → smaller at coarser scales
    kernel_size = max(3, 15 // scale_factor)
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    surrounding = dilated.astype(bool) & ~mask

    min_surrounding = max(20, 100 // (scale_factor * scale_factor))
    if surrounding.sum() < min_surrounding:
        return False, {}

    sma_surrounding = sma_norm[surrounding].mean()
    nuclear_surrounding = nuclear_norm[surrounding].mean()

    sma_ratio = sma_inside / (sma_surrounding + 1)
    nuclear_ratio = nuclear_inside / (nuclear_surrounding + 1)

    # Lumen must simply be darker than surrounding wall (any amount)
    # Relaxed from sma_ratio < 0.85 to < 1.0 to catch more vessels
    is_valid = (sma_ratio < 1.0) and (nuclear_ratio < 1.5)

    stats = {
        'area': int(area),
        'sma_inside': float(sma_inside),
        'sma_wall': float(sma_surrounding),
        'sma_ratio': float(sma_ratio),
        'nuclear_ratio': float(nuclear_ratio),
    }

    return is_valid, stats

def dilate_until_signal_drops(lumens, cd31_norm, scale_factor=1, drop_ratio=0.85, max_iterations=50):
    """
    Expand from lumens by dilating until CD31 signal drops.

    CD31 marks endothelium (vessel lining), so this ensures we're capturing
    actual blood vessel boundaries.

    For each lumen:
    1. Start with the lumen mask
    2. Dilate iteratively
    3. Check mean CD31 intensity in the newly added ring
    4. Stop when ring intensity drops below drop_ratio * wall intensity

    Returns labels array where each lumen's expanded region has a unique label.
    """
    labels = np.zeros(cd31_norm.shape, dtype=np.int32)
    kernel = np.ones((3, 3), np.uint8)

    for idx, lumen in enumerate(lumens):
        mask = lumen['mask'].astype(np.uint8)
        label_id = idx + 1

        # Get initial wall intensity (the CD31+ ring around lumen)
        dilated_once = cv2.dilate(mask, kernel, iterations=1)
        initial_ring = (dilated_once > 0) & (mask == 0)
        if initial_ring.sum() == 0:
            labels[mask > 0] = label_id
            continue
        wall_intensity = cd31_norm[initial_ring].mean()

        # Threshold for stopping: when ring drops below this
        stop_threshold = wall_intensity * drop_ratio

        # Iteratively dilate - always do at least one dilation
        current_mask = dilated_once.copy()  # Start with one dilation already done
        for i in range(max_iterations - 1):  # -1 since we already did one
            dilated = cv2.dilate(current_mask, kernel, iterations=1)
            new_ring = (dilated > 0) & (current_mask == 0)

            if new_ring.sum() == 0:
                break

            ring_intensity = cd31_norm[new_ring].mean()

            # Stop if signal dropped
            if ring_intensity < stop_threshold:
                break

            # Only expand if signal is still strong
            current_mask = dilated

        # Assign label to final expanded region
        labels[current_mask > 0] = label_id

    return labels

def process_tile_at_scale(tile_x, tile_y, channel_cache, sam2_generator, scale_config, pixel_size_um=0.1725):
    """
    Process a tile at given scale using downsampled channel cache.

    tile_x, tile_y are in SCALED coordinates (not full-res).
    The channel_cache holds data at BASE_SCALE and downsamples further as needed.
    """
    scale_factor = scale_config['scale_factor']
    min_diam = scale_config['min_diam_um']
    max_diam = scale_config['max_diam_um']
    tile_size = scale_config.get('tile_size', TILE_SIZE)

    # Effective pixel size at this scale (larger pixels at coarser scales)
    effective_pixel_size = pixel_size_um * scale_factor

    # Get all channels at this scale from cache
    tiles = {}
    for ch in [NUCLEAR, CD31, SMA, PM]:
        tiles[ch] = channel_cache.get_tile(tile_x, tile_y, tile_size, ch, scale_factor)

    # Check for valid tile data (all channels must be present)
    if any(tiles[ch] is None or tiles[ch].size == 0 for ch in [NUCLEAR, CD31, SMA, PM]):
        return []

    # Apply photobleaching correction + normalization to all channels
    sma_corrected = correct_photobleaching(tiles[SMA].astype(np.float32))
    nuclear_corrected = correct_photobleaching(tiles[NUCLEAR].astype(np.float32))
    cd31_corrected = correct_photobleaching(tiles[CD31].astype(np.float32))
    pm_corrected = correct_photobleaching(tiles[PM].astype(np.float32))

    sma_norm = normalize_channel(sma_corrected)
    nuclear_norm = normalize_channel(nuclear_corrected)
    cd31_norm = normalize_channel(cd31_corrected)
    pm_norm = normalize_channel(pm_corrected)

    # SAM2 input: grayscale as RGB (for detection)
    sma_rgb = cv2.cvtColor(sma_norm, cv2.COLOR_GRAY2RGB)
    cd31_rgb = cv2.cvtColor(cd31_norm, cv2.COLOR_GRAY2RGB)

    # Display RGB: multi-channel for visualization (R=SMA, G=CD31, B=nuclear)
    display_rgb = np.stack([sma_norm, cd31_norm, nuclear_norm], axis=-1)

    # Run SAM2 on both SMA and CD31 channels to find more vessels
    masks_sma = sam2_generator.generate(sma_rgb)
    masks_cd31 = sam2_generator.generate(cd31_rgb)

    # Combine masks from both channels
    # Deduplication happens at the final merge step (merge_vessels_across_scales)
    masks = masks_sma + masks_cd31

    # Filter to lumens
    lumens = []
    for i, m in enumerate(masks):
        mask = m['segmentation']
        is_valid, stats = verify_lumen_multichannel(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm, scale_factor)
        if is_valid:
            lumens.append({'idx': i, 'mask': mask, 'stats': stats})

    if len(lumens) == 0:
        # Cleanup before returning
        del tiles, sma_corrected, nuclear_corrected, cd31_corrected, pm_corrected
        del sma_norm, nuclear_norm, cd31_norm, pm_norm, sma_rgb, display_rgb, masks
        gc.collect()
        return []

    # Watershed expansion
    labels = dilate_until_signal_drops(lumens, cd31_norm, scale_factor)

    # Extract vessels
    vessels = []
    for idx, lumen in enumerate(lumens):
        label_id = idx + 1

        wall_mask = (labels == label_id).astype(np.uint8)
        outer_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(outer_contours) == 0:
            continue
        outer_contour = max(outer_contours, key=cv2.contourArea)

        inner_mask = lumen['mask'].astype(np.uint8)
        inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(inner_contours) == 0:
            continue
        inner_contour = max(inner_contours, key=cv2.contourArea)

        # Compute measurements at tile scale, then convert to full resolution
        outer_area_tile = cv2.contourArea(outer_contour)
        inner_area_tile = cv2.contourArea(inner_contour)

        # Convert areas back to full resolution (multiply by scale_factor^2)
        area_scale = scale_factor * scale_factor
        outer_area = outer_area_tile * area_scale
        inner_area = inner_area_tile * area_scale
        wall_area = outer_area - inner_area

        # Fit ellipses
        if len(outer_contour) >= 5:
            outer_ellipse = cv2.fitEllipse(outer_contour)
            outer_diameter = (outer_ellipse[1][0] + outer_ellipse[1][1]) / 2 * effective_pixel_size
        else:
            outer_diameter = np.sqrt(outer_area / np.pi) * 2 * pixel_size_um

        if len(inner_contour) >= 5:
            inner_ellipse = cv2.fitEllipse(inner_contour)
            inner_diameter = (inner_ellipse[1][0] + inner_ellipse[1][1]) / 2 * effective_pixel_size
        else:
            inner_diameter = np.sqrt(inner_area / np.pi) * 2 * pixel_size_um

        # Filter by diameter range for this scale
        if outer_diameter < min_diam or outer_diameter > max_diam:
            continue

        # Skip invalid vessels (outer must be > inner)
        if outer_diameter <= inner_diameter:
            continue

        # Filter out vessels with outer/inner ratio > 2 (likely false positives)
        if inner_diameter > 0 and (outer_diameter / inner_diameter) > 2:
            continue

        wall_thickness = (outer_diameter - inner_diameter) / 2

        # Get centroid at tile scale
        M = cv2.moments(outer_contour)
        if M['m00'] > 0:
            cx_tile = int(M['m10'] / M['m00'])
            cy_tile = int(M['m01'] / M['m00'])
        else:
            cx_tile, cy_tile = inner_contour.mean(axis=0)[0].astype(int)

        # Scale contours and centroid back to full resolution
        # tile coords * scale_factor = full res coords
        cx_full = int(cx_tile * scale_factor)
        cy_full = int(cy_tile * scale_factor)

        outer_full = (outer_contour * scale_factor).astype(int)
        inner_full = (inner_contour * scale_factor).astype(int)

        # Global coordinates (tile_x/tile_y are in scaled coords, convert to full res)
        global_x = tile_x * scale_factor + cx_full
        global_y = tile_y * scale_factor + cy_full

        vessel = {
            'uid': f'vessel_{global_x}_{global_y}',
            'scale': scale_config['name'],
            'scale_factor': scale_factor,
            'tile_x': tile_x,  # in scaled coordinates
            'tile_y': tile_y,
            'local_center': [int(cx_tile), int(cy_tile)],  # in tile coordinates
            'global_center': [int(global_x), int(global_y)],  # in full-res coordinates
            'outer_contour': outer_full.tolist(),
            'inner_contour': inner_full.tolist(),
            'outer_diameter_um': float(outer_diameter),
            'inner_diameter_um': float(inner_diameter),
            'wall_thickness_um': float(wall_thickness),
            'outer_area_px': float(outer_area),
            'inner_area_px': float(inner_area),
            'wall_area_px': float(wall_area),
            **lumen['stats']
        }

        # Save crop with contours (use tile-scale contours for drawing on tile-scale image)
        crop_path = save_vessel_crop(vessel, display_rgb, outer_contour, inner_contour, tile_x, tile_y, tile_size, scale_factor)
        vessel['crop_path'] = crop_path

        vessels.append(vessel)

    # Cleanup large arrays before returning
    del tiles, sma_corrected, nuclear_corrected, cd31_corrected, pm_corrected
    del sma_norm, nuclear_norm, cd31_norm, pm_norm, sma_rgb, display_rgb, masks, labels
    gc.collect()

    return vessels

def merge_vessels_across_scales(vessels, iou_threshold=0.3):
    """Merge vessels detected at different scales, keeping the coarser scale version (more likely complete)."""
    # Sort by scale (coarsest first - higher denominator = coarser)
    sorted_vessels = sorted(vessels, key=lambda v: float(v['scale'].split('/')[1]) if '/' in v['scale'] else 1, reverse=True)

    merged = []
    used = set()

    for i, v1 in enumerate(sorted_vessels):
        if i in used:
            continue

        merged.append(v1)
        used.add(i)

        # Find overlapping vessels at finer scales (discard them, keep coarser)
        for j, v2 in enumerate(sorted_vessels):
            if j in used or j == i:
                continue

            # Quick distance check
            dx = abs(v1['global_center'][0] - v2['global_center'][0])
            dy = abs(v1['global_center'][1] - v2['global_center'][1])
            max_dist = max(v1['outer_diameter_um'], v2['outer_diameter_um']) / 0.1725

            if dx < max_dist and dy < max_dist:
                # Mark as used (keep v1 which is at coarser scale, more likely complete)
                used.add(j)

    return merged

def main():
    import sys
    print("=" * 60, flush=True)
    print("Multi-Scale SAM2 Vessel Detection", flush=True)
    print("=" * 60, flush=True)

    # Open CZI (don't load full-res into RAM)
    loader = CZILoader(CZI_PATH)
    mosaic_size = loader.mosaic_size
    print(f"\nMosaic size (full res): {mosaic_size}", flush=True)

    # Load all channels at BASE_SCALE (1/2) - 4x smaller than full res
    channel_cache = DownsampledChannelCache(loader, [NUCLEAR, CD31, SMA, PM], BASE_SCALE)

    # Load SAM2
    print("\nLoading SAM2...", flush=True)
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "/home/dude/code/xldvp_seg_repo/checkpoints/sam2.1_hiera_large.pt",
        device="cuda"
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=48,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.7,
        min_mask_region_area=100,
    )

    all_vessels = []

    # Process each scale
    for scale_config in SCALES:
        scale_factor = scale_config['scale_factor']
        tile_size = scale_config.get('tile_size', TILE_SIZE)
        stride = int(tile_size * (1 - TILE_OVERLAP))  # With 50% overlap, stride = tile_size/2

        print(f"\n{'='*40}", flush=True)
        print(f"Processing scale {scale_config['name']}", flush=True)
        print(f"  Scale factor: {scale_factor}x", flush=True)
        print(f"  Tile size: {tile_size}px (covers {tile_size * scale_factor}px at full res)", flush=True)
        print(f"  Stride: {stride}px ({int(TILE_OVERLAP*100)}% overlap)", flush=True)
        print(f"  Diameter range: {scale_config['min_diam_um']}-{scale_config['max_diam_um']} µm", flush=True)
        print(f"{'='*40}", flush=True)

        # Create tile grid for this scale (in SCALED coordinates)
        # With overlap, we use stride instead of tile_size for stepping
        scaled_mosaic_x = mosaic_size[0] // scale_factor
        scaled_mosaic_y = mosaic_size[1] // scale_factor
        n_tiles_x = max(1, (scaled_mosaic_x - tile_size) // stride + 1)
        n_tiles_y = max(1, (scaled_mosaic_y - tile_size) // stride + 1)
        total_tiles = n_tiles_x * n_tiles_y
        print(f"Scaled mosaic: {scaled_mosaic_x} x {scaled_mosaic_y}", flush=True)
        print(f"Tile grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles", flush=True)

        # Find tissue tiles (coordinates in scaled space)
        print("Identifying tissue tiles...", flush=True)
        tissue_tiles = []

        for ty in tqdm(range(n_tiles_y), desc="Scanning"):
            for tx in range(n_tiles_x):
                # tile_x, tile_y are in SCALED coordinates (using stride for stepping)
                tile_x = tx * stride
                tile_y = ty * stride

                # Check for tissue using the scaled tile (from cache)
                tile = channel_cache.get_tile(tile_x, tile_y, tile_size, SMA, scale_factor)
                if tile is not None and is_tissue_tile(tile):
                    tissue_tiles.append((tile_x, tile_y))

        print(f"Found {len(tissue_tiles)} tissue tiles", flush=True)

        # Sample (at 100%, this just takes all tiles)
        n_sample = max(1, int(len(tissue_tiles) * SAMPLE_FRACTION))
        sampled_tiles = random.sample(tissue_tiles, min(n_sample, len(tissue_tiles)))
        print(f"Processing {len(sampled_tiles)} tiles", flush=True)

        # Process
        scale_vessels = []
        for tile_x, tile_y in tqdm(sampled_tiles, desc="Processing"):
            try:
                vessels = process_tile_at_scale(
                    tile_x, tile_y, channel_cache, mask_generator, scale_config
                )
                scale_vessels.extend(vessels)
            except Exception as e:
                print(f"\nError at ({tile_x}, {tile_y}): {e}", flush=True)
                import traceback
                traceback.print_exc()
            finally:
                # Memory cleanup after each tile
                gc.collect()
                torch.cuda.empty_cache()

        print(f"Found {len(scale_vessels)} vessels at scale {scale_config['name']}", flush=True)
        all_vessels.extend(scale_vessels)

        # Cleanup between scales
        gc.collect()
        torch.cuda.empty_cache()

    # Merge across scales
    print(f"\nTotal vessels before merge: {len(all_vessels)}")
    merged_vessels = merge_vessels_across_scales(all_vessels)
    print(f"After merge: {len(merged_vessels)}")

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'vessel_detections_multiscale.json')
    with open(output_path, 'w') as f:
        json.dump(merged_vessels, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Stats
    if len(merged_vessels) > 0:
        diameters = [v['outer_diameter_um'] for v in merged_vessels]
        print(f"\nDiameter stats:")
        print(f"  min={min(diameters):.1f}, max={max(diameters):.1f}, mean={np.mean(diameters):.1f} µm")

        # By scale
        for scale_config in SCALES:
            count = sum(1 for v in merged_vessels if v['scale'] == scale_config['name'])
            print(f"  Scale {scale_config['name']}: {count} vessels")

    # Cleanup
    channel_cache.release()
    loader.close()

    # Generate HTML
    generate_html(merged_vessels)

    print("\nDone!")

def generate_html(vessels):
    """Generate HTML viewer using package template."""
    import base64
    from segmentation.io.html_export import export_vessel_samples_to_html

    print("\nGenerating HTML...")

    # Convert vessels to sample format expected by export_vessel_samples_to_html
    samples = []
    for v in vessels:
        crop_path = v.get('crop_path', '')
        if not crop_path or not os.path.exists(crop_path):
            continue

        # Read crop and convert to base64
        with open(crop_path, 'rb') as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')

        # Build features dict (package expects specific keys)
        features = {
            'outer_diameter_um': v.get('outer_diameter_um', 0),
            'inner_diameter_um': v.get('inner_diameter_um', 0),
            'wall_thickness_mean_um': v.get('wall_thickness_um', 0),
            'outer_area_px': v.get('outer_area_px', 0),
            'inner_area_px': v.get('inner_area_px', 0),
            'wall_area_px': v.get('wall_area_px', 0),
            'sma_ratio': v.get('sma_ratio', 0),
            'scale': v.get('scale', '?'),
            'global_x': v.get('global_center', [0, 0])[0],
            'global_y': v.get('global_center', [0, 0])[1],
        }

        samples.append({
            'uid': v.get('uid', ''),
            'image': img_base64,  # Just base64, export function adds prefix
            'features': features,
        })

    if not samples:
        print("No samples to export")
        return

    # Sort by diameter descending
    samples = sorted(samples, key=lambda s: -s['features'].get('outer_diameter_um', 0))

    # Export using package template
    export_vessel_samples_to_html(
        samples=samples,
        output_dir=OUTPUT_DIR,
        cell_type='vessel',
        samples_per_page=200,
        title='SAM2 Multiscale Vessel Detection',
        subtitle=f"Scales: {', '.join(s['name'] for s in SCALES)}",
        experiment_name='sam2_multiscale',
        file_name=os.path.basename(CZI_PATH),
    )

if __name__ == '__main__':
    main()
