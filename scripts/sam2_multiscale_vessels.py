#!/usr/bin/env python3
"""
Multi-scale SAM2 vessel detection.
- Scale 1/4: 20k source → 5k for SAM2 (detects large vessels >200µm)
- Scale 1: 5k tiles (detects medium vessels 10-200µm)
Merge and deduplicate across scales.
"""

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from skimage.segmentation import watershed
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
SAMPLE_FRACTION = 0.20
TISSUE_VARIANCE_THRESHOLD = 50

# Scale configurations (coarse to fine)
# 1/16 scale: 80000px tiles require ~51GB transient memory - needs cleanup between tiles
SCALES = [
    {'name': '1/16', 'source_size': 80000, 'sam2_size': 5000, 'scale': 0.0625, 'min_diam_um': 500, 'max_diam_um': 10000},
    {'name': '1/4', 'source_size': 20000, 'sam2_size': 5000, 'scale': 0.25, 'min_diam_um': 100, 'max_diam_um': 1000},
    {'name': '1/1', 'source_size': 5000, 'sam2_size': 5000, 'scale': 1.0, 'min_diam_um': 10, 'max_diam_um': 300},
]

# Channel indices
NUCLEAR = 0
CD31 = 1
SMA = 2
PM = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'tiles'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'crops'), exist_ok=True)

CROP_SIZE = 400  # pixels around vessel center for crops

def normalize_channel(arr, p_low=1, p_high=99):
    """Normalize to uint8 using percentile clipping (robust to outliers)."""
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, (p_low, p_high))
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-8) * 255, 0, 255)
    return arr.astype(np.uint8)

def save_vessel_crop(vessel, sma_rgb, tile_x, tile_y, source_size, scale):
    """Save a raw crop of the vessel (no contours - those are drawn at display time)."""
    cx, cy = vessel['local_center']

    # Crop region in SAM2 image coordinates
    half = CROP_SIZE // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(sma_rgb.shape[1], cx + half)
    y2 = min(sma_rgb.shape[0], cy + half)

    crop = sma_rgb[y1:y2, x1:x2].copy()

    # Store crop offset for contour drawing at display time
    vessel['crop_offset'] = [x1, y1]
    vessel['crop_scale'] = scale

    # Save raw crop (no contours)
    crop_path = os.path.join(OUTPUT_DIR, 'crops', f"{vessel['uid']}.jpg")
    cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

    return crop_path

def is_tissue_tile(tile, threshold=TISSUE_VARIANCE_THRESHOLD):
    return np.var(tile) > threshold

def verify_lumen_multichannel(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm):
    """Verify region is true lumen by checking multi-channel emptiness."""
    area = mask.sum()
    if area < 50 or area > 500000:
        return False, {}

    sma_inside = sma_norm[mask].mean()
    nuclear_inside = nuclear_norm[mask].mean()

    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    surrounding = dilated.astype(bool) & ~mask

    if surrounding.sum() < 100:
        return False, {}

    sma_surrounding = sma_norm[surrounding].mean()
    nuclear_surrounding = nuclear_norm[surrounding].mean()

    sma_ratio = sma_inside / (sma_surrounding + 1)
    nuclear_ratio = nuclear_inside / (nuclear_surrounding + 1)

    # Lumen must be darker than surrounding wall (sma_ratio < 0.85)
    # and not too nuclear-dense (nuclear_ratio < 1.2)
    is_valid = (sma_ratio < 0.85) and (nuclear_ratio < 1.2)

    stats = {
        'area': int(area),
        'sma_inside': float(sma_inside),
        'sma_wall': float(sma_surrounding),
        'sma_ratio': float(sma_ratio),
        'nuclear_ratio': float(nuclear_ratio),
    }

    return is_valid, stats

def watershed_expand(lumens, sma_norm):
    """Expand from lumens to find outer wall via watershed."""
    sma_inverted = 255 - sma_norm
    markers = np.zeros(sma_norm.shape, dtype=np.int32)

    for idx, lumen in enumerate(lumens):
        mask = lumen['mask']
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        markers[eroded > 0] = idx + 1

    background_thresh = np.percentile(sma_norm, 95)
    background = sma_norm > background_thresh
    markers[background] = len(lumens) + 1

    labels = watershed(sma_inverted, markers, mask=None)
    return labels

def process_tile_at_scale(tile_x, tile_y, loader, sam2_generator, scale_config, pixel_size_um=0.1725):
    """
    Process a tile at given scale.

    For scale < 1: extract larger region, downsample for SAM2, then scale coords back up.
    """
    source_size = scale_config['source_size']
    sam2_size = scale_config['sam2_size']
    scale = scale_config['scale']
    min_diam = scale_config['min_diam_um']
    max_diam = scale_config['max_diam_um']

    # Effective pixel size at this scale
    effective_pixel_size = pixel_size_um / scale

    # Get all channels
    tiles = {}
    for ch in [NUCLEAR, CD31, SMA, PM]:
        tiles[ch] = loader.get_tile(tile_x, tile_y, source_size, ch)

    # Apply photobleaching correction + normalization to all channels
    sma_corrected = correct_photobleaching(tiles[SMA].astype(np.float32))
    nuclear_corrected = correct_photobleaching(tiles[NUCLEAR].astype(np.float32))
    cd31_corrected = correct_photobleaching(tiles[CD31].astype(np.float32))
    pm_corrected = correct_photobleaching(tiles[PM].astype(np.float32))

    sma_norm_full = normalize_channel(sma_corrected)
    nuclear_norm_full = normalize_channel(nuclear_corrected)
    cd31_norm_full = normalize_channel(cd31_corrected)
    pm_norm_full = normalize_channel(pm_corrected)

    # Downsample if needed
    if scale < 1.0:
        sma_norm = cv2.resize(sma_norm_full, (sam2_size, sam2_size), interpolation=cv2.INTER_AREA)
        nuclear_norm = cv2.resize(nuclear_norm_full, (sam2_size, sam2_size), interpolation=cv2.INTER_AREA)
        cd31_norm = cv2.resize(cd31_norm_full, (sam2_size, sam2_size), interpolation=cv2.INTER_AREA)
        pm_norm = cv2.resize(pm_norm_full, (sam2_size, sam2_size), interpolation=cv2.INTER_AREA)
    else:
        sma_norm = sma_norm_full
        nuclear_norm = nuclear_norm_full
        cd31_norm = cd31_norm_full
        pm_norm = pm_norm_full

    # SAM2 input: SMA grayscale as RGB (for detection)
    sma_rgb = cv2.cvtColor(sma_norm, cv2.COLOR_GRAY2RGB)

    # Display RGB: multi-channel for visualization (R=SMA, G=CD31, B=nuclear)
    display_rgb = np.stack([sma_norm, cd31_norm, nuclear_norm], axis=-1)

    # Run SAM2
    masks = sam2_generator.generate(sma_rgb)

    # Filter to lumens
    lumens = []
    for i, m in enumerate(masks):
        mask = m['segmentation']
        is_valid, stats = verify_lumen_multichannel(mask, sma_norm, nuclear_norm, cd31_norm, pm_norm)
        if is_valid:
            lumens.append({'idx': i, 'mask': mask, 'stats': stats})

    if len(lumens) == 0:
        # Cleanup before returning
        del tiles, sma_corrected, sma_norm_full, masks
        gc.collect()
        return [], sma_rgb

    # Watershed expansion
    labels = watershed_expand(lumens, sma_norm)

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

        # Compute measurements at SAM2 scale, then convert
        outer_area_sam2 = cv2.contourArea(outer_contour)
        inner_area_sam2 = cv2.contourArea(inner_contour)

        # Convert areas back to full resolution
        area_scale = 1.0 / (scale * scale)
        outer_area = outer_area_sam2 * area_scale
        inner_area = inner_area_sam2 * area_scale
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

        wall_thickness = (outer_diameter - inner_diameter) / 2

        # Get centroid at SAM2 scale
        M = cv2.moments(outer_contour)
        if M['m00'] > 0:
            cx_sam2 = int(M['m10'] / M['m00'])
            cy_sam2 = int(M['m01'] / M['m00'])
        else:
            cx_sam2, cy_sam2 = inner_contour.mean(axis=0)[0].astype(int)

        # Scale contours and centroid back to full resolution
        cx = int(cx_sam2 / scale)
        cy = int(cy_sam2 / scale)

        outer_full = (outer_contour.astype(float) / scale).astype(int)
        inner_full = (inner_contour.astype(float) / scale).astype(int)

        # Global coordinates
        global_x = tile_x + cx
        global_y = tile_y + cy

        vessel = {
            'uid': f'vessel_{global_x}_{global_y}',
            'scale': scale_config['name'],
            'tile_x': tile_x,
            'tile_y': tile_y,
            'local_center': [int(cx), int(cy)],
            'global_center': [int(global_x), int(global_y)],
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

        # Save crop (use display_rgb for multi-channel visualization)
        crop_path = save_vessel_crop(vessel, display_rgb, tile_x, tile_y, source_size, scale)
        vessel['crop_path'] = crop_path

        vessels.append(vessel)

    # Cleanup large arrays before returning
    del tiles, sma_corrected, sma_norm_full, masks, labels
    if scale < 1.0:
        del sma_norm, nuclear_norm, cd31_norm, pm_norm
    gc.collect()

    return vessels, display_rgb

def compute_iou(contour1, contour2, img_shape):
    """Compute IoU between two contours."""
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask1, [np.array(contour1)], 1)
    cv2.fillPoly(mask2, [np.array(contour2)], 1)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)

def merge_vessels_across_scales(vessels, iou_threshold=0.3):
    """Merge vessels detected at different scales, keeping the finer scale version."""
    # Sort by scale (finest first)
    sorted_vessels = sorted(vessels, key=lambda v: -float(v['scale'].split('/')[1]) if '/' in v['scale'] else 1)

    merged = []
    used = set()

    for i, v1 in enumerate(sorted_vessels):
        if i in used:
            continue

        merged.append(v1)
        used.add(i)

        # Find overlapping vessels at coarser scales
        for j, v2 in enumerate(sorted_vessels):
            if j in used or j == i:
                continue

            # Quick distance check
            dx = abs(v1['global_center'][0] - v2['global_center'][0])
            dy = abs(v1['global_center'][1] - v2['global_center'][1])
            max_dist = max(v1['outer_diameter_um'], v2['outer_diameter_um']) / 0.1725

            if dx < max_dist and dy < max_dist:
                # Mark as used (keep v1 which is at finer scale)
                used.add(j)

    return merged

def main():
    print("=" * 60)
    print("Multi-Scale SAM2 Vessel Detection")
    print("=" * 60)

    loader = CZILoader(CZI_PATH)

    # Load all channels
    print("\nLoading all channels...")
    for ch in [NUCLEAR, CD31, SMA, PM]:
        print(f"  Loading channel {ch}...")
        loader.load_channel(ch)

    mosaic_size = loader.mosaic_size
    print(f"\nMosaic size: {mosaic_size}")

    # Load SAM2
    print("\nLoading SAM2...")
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "/home/dude/code/xldvp_seg_repo/checkpoints/sam2.1_hiera_large.pt",
        device="cuda"
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=100,
    )

    all_vessels = []

    # Process each scale
    for scale_config in SCALES:
        print(f"\n{'='*40}")
        print(f"Processing scale {scale_config['name']}")
        print(f"  Source tile: {scale_config['source_size']}px")
        print(f"  SAM2 input: {scale_config['sam2_size']}px")
        print(f"  Diameter range: {scale_config['min_diam_um']}-{scale_config['max_diam_um']} µm")
        print(f"{'='*40}")

        source_size = scale_config['source_size']

        # Create tile grid for this scale
        n_tiles_x = mosaic_size[0] // source_size
        n_tiles_y = mosaic_size[1] // source_size
        total_tiles = n_tiles_x * n_tiles_y
        print(f"Tile grid: {n_tiles_x} x {n_tiles_y} = {total_tiles} tiles")

        # Find tissue tiles
        print("Identifying tissue tiles...")
        tissue_tiles = []
        sample_size = 5000  # Size of sample regions for tissue check

        for ty in tqdm(range(n_tiles_y), desc="Scanning"):
            for tx in range(n_tiles_x):
                tile_x = tx * source_size
                tile_y = ty * source_size

                # For large tiles, sample multiple regions across the tile
                if source_size > sample_size * 2:
                    # Sample 9 regions (3x3 grid) across the large tile
                    has_tissue = False
                    step = (source_size - sample_size) // 2
                    for sy in range(3):
                        for sx in range(3):
                            sample_x = tile_x + sx * step
                            sample_y = tile_y + sy * step
                            sample = loader.get_tile(sample_x, sample_y, sample_size, SMA)
                            if sample is not None and is_tissue_tile(sample):
                                has_tissue = True
                                break
                        if has_tissue:
                            break
                    if has_tissue:
                        tissue_tiles.append((tile_x, tile_y))
                else:
                    # Small tiles - check directly
                    tile = loader.get_tile(tile_x, tile_y, min(source_size, sample_size), SMA)
                    if tile is not None and is_tissue_tile(tile):
                        tissue_tiles.append((tile_x, tile_y))

        print(f"Found {len(tissue_tiles)} tissue tiles")

        # Sample
        n_sample = max(1, int(len(tissue_tiles) * SAMPLE_FRACTION))
        sampled_tiles = random.sample(tissue_tiles, min(n_sample, len(tissue_tiles)))
        print(f"Sampling {len(sampled_tiles)} tiles")

        # Process
        scale_vessels = []
        for tile_x, tile_y in tqdm(sampled_tiles, desc="Processing"):
            try:
                vessels, _ = process_tile_at_scale(
                    tile_x, tile_y, loader, mask_generator, scale_config
                )
                scale_vessels.extend(vessels)
            except Exception as e:
                print(f"\nError at ({tile_x}, {tile_y}): {e}")
            finally:
                # Memory cleanup after each tile
                gc.collect()
                torch.cuda.empty_cache()

        print(f"Found {len(scale_vessels)} vessels at scale {scale_config['name']}")
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

    loader.close()

    # Generate HTML
    generate_html(merged_vessels)

    print("\nDone!")

def generate_html(vessels):
    """Generate HTML viewer from saved crops."""
    print("\nGenerating HTML...")

    # Sort by diameter descending
    vessels = sorted(vessels, key=lambda v: -v.get('outer_diameter_um', 0))

    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Multiscale Vessel Detection</title>
    <style>
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 20px; }
        h1 { text-align: center; }
        .stats { text-align: center; margin: 20px; padding: 15px; background: #333; border-radius: 8px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; }
        .card { background: #2a2a2a; border-radius: 8px; overflow: hidden; }
        .card img { width: 100%; height: 180px; object-fit: contain; background: #000; }
        .card-info { padding: 8px; font-size: 11px; }
        .diam { color: #4CAF50; font-weight: bold; font-size: 14px; }
        .wall { color: #2196F3; }
        .scale { color: #FF9800; }
        .legend { text-align: center; margin: 10px; }
        .green { color: #0f0; }
        .cyan { color: #0ff; }
    </style>
</head>
<body>
    <h1>Multiscale Vessel Detection</h1>
    <div class="stats">
        <strong>Total: ''' + str(len(vessels)) + ''' vessels</strong><br>
        Scale 1/16: ''' + str(sum(1 for v in vessels if v.get('scale') == '1/16')) + ''' |
        Scale 1/4: ''' + str(sum(1 for v in vessels if v.get('scale') == '1/4')) + ''' |
        Scale 1/1: ''' + str(sum(1 for v in vessels if v.get('scale') == '1/1')) + '''
    </div>
    <div class="legend">
        <span class="green">GREEN = outer wall</span> |
        <span class="cyan">CYAN = inner lumen</span>
    </div>
    <div class="grid">
'''

    for v in vessels:
        crop_path = v.get('crop_path', '')
        if crop_path and os.path.exists(crop_path):
            rel_path = os.path.relpath(crop_path, OUTPUT_DIR)
        else:
            continue

        diam = v.get('outer_diameter_um', 0)
        wall = v.get('wall_thickness_um', 0)
        scale_name = v.get('scale', '?')

        html += f'''
        <div class="card">
            <img src="{rel_path}">
            <div class="card-info">
                <div class="diam">⌀ {diam:.0f} µm</div>
                <div class="wall">wall: {wall:.1f} µm</div>
                <div class="scale">{scale_name}</div>
            </div>
        </div>
'''

    html += '''
    </div>
</body>
</html>
'''

    html_path = os.path.join(OUTPUT_DIR, 'index.html')
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"Saved HTML to {html_path}")

if __name__ == '__main__':
    main()
