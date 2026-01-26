#!/usr/bin/env python3
"""Generate HTML viewer for multiscale vessel detection results using package template."""

import json
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

sys.path.insert(0, '/home/dude/code/vessel_seg')
from segmentation.io.czi_loader import CZILoader
from segmentation.io.html_export import export_samples_to_html
from segmentation.preprocessing.illumination import correct_photobleaching

# Config
CZI_PATH = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
JSON_PATH = "/home/dude/vessel_output/sam2_multiscale/vessel_detections_multiscale.json"
OUTPUT_DIR = "/home/dude/vessel_output/sam2_multiscale"
CROP_SIZE = 400  # pixels around center

# Channel indices
NUCLEAR = 0
CD31 = 1
SMA = 2
PM = 3

def normalize_uint8(arr, apply_photobleach=True):
    """Normalize to uint8 with optional photobleaching correction."""
    arr = arr.astype(np.float32)
    if apply_photobleach:
        arr = correct_photobleaching(arr, morph_kernel_size=51).astype(np.float32)
    p1, p99 = np.percentile(arr, (1, 99))
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-8) * 255, 0, 255)
    return arr.astype(np.uint8)

def img_to_base64(img):
    """Convert numpy array to base64 JPEG."""
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

def draw_contours_on_crop(crop, outer, inner, crop_offset, crop_scale):
    """Draw outer (green) and inner (cyan) contours on crop."""
    if outer:
        if isinstance(outer[0][0], list):
            outer = [p[0] for p in outer]
        pts = (np.array(outer) * crop_scale - crop_offset).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(crop, [pts], -1, (0, 255, 0), 2)

    if inner:
        if isinstance(inner[0][0], list):
            inner = [p[0] for p in inner]
        pts = (np.array(inner) * crop_scale - crop_offset).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(crop, [pts], -1, (0, 255, 255), 2)

    return crop

def main():
    print("Loading vessels...")
    with open(JSON_PATH) as f:
        vessels = json.load(f)

    print(f"Found {len(vessels)} vessels")

    # Sort by area descending
    vessels = sorted(vessels, key=lambda v: -v.get('outer_area_px', 0))

    # Limit for HTML
    max_vessels = 200
    vessels = vessels[:max_vessels]

    print(f"Generating HTML for top {len(vessels)} vessels by diameter...")

    # Check if we have pre-saved crops
    has_crops = any(v.get('crop_path') and os.path.exists(v.get('crop_path', '')) for v in vessels)

    loader = None
    if not has_crops:
        # Need to load CZI for fallback
        print("Loading CZI (no pre-saved crops found)...")
        loader = CZILoader(CZI_PATH)
        for ch in [NUCLEAR, CD31, SMA, PM]:
            loader.load_channel(ch)

    # Prepare samples for package template
    samples = []
    for v in tqdm(vessels, desc="Preparing samples"):
        crop_path = v.get('crop_path', '')

        if crop_path and os.path.exists(crop_path):
            # Use pre-saved crop (already photobleach corrected + normalized, RGB)
            crop = cv2.imread(crop_path)
            if crop is None:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Draw contours
            crop_offset = v.get('crop_offset', [0, 0])
            crop_scale = v.get('crop_scale', 1.0)
            crop = draw_contours_on_crop(
                crop,
                v.get('outer_contour', []),
                v.get('inner_contour', []),
                crop_offset,
                crop_scale
            )
        else:
            # Fallback: extract from CZI with all channels
            if loader is None:
                continue

            cx, cy = v['global_center']
            x1 = max(0, cx - CROP_SIZE)
            y1 = max(0, cy - CROP_SIZE)

            # Get all channels and create RGB
            sma = loader.get_tile(x1, y1, CROP_SIZE * 2, SMA)
            cd31 = loader.get_tile(x1, y1, CROP_SIZE * 2, CD31)
            nuclear = loader.get_tile(x1, y1, CROP_SIZE * 2, NUCLEAR)

            if sma is None:
                continue

            # Normalize each channel with photobleaching correction
            sma_norm = normalize_uint8(sma)
            cd31_norm = normalize_uint8(cd31) if cd31 is not None else np.zeros_like(sma_norm)
            nuclear_norm = normalize_uint8(nuclear) if nuclear is not None else np.zeros_like(sma_norm)

            # Create RGB: R=SMA, G=CD31, B=nuclear
            crop = np.stack([sma_norm, cd31_norm, nuclear_norm], axis=-1)

            # Draw contours
            tile_x = v.get('tile_x', 0)
            tile_y = v.get('tile_y', 0)

            outer = v.get('outer_contour', [])
            inner = v.get('inner_contour', [])

            if outer:
                if isinstance(outer[0][0], list):
                    outer = [p[0] for p in outer]
                pts = np.array([[tile_x + p[0] - x1, tile_y + p[1] - y1] for p in outer])
                pts = pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(crop, [pts], -1, (0, 255, 0), 2)

            if inner:
                if isinstance(inner[0][0], list):
                    inner = [p[0] for p in inner]
                pts = np.array([[tile_x + p[0] - x1, tile_y + p[1] - y1] for p in inner])
                pts = pts.astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(crop, [pts], -1, (0, 255, 255), 2)

        # Convert to base64
        img_b64 = img_to_base64(crop)

        # Build sample dict for package template
        # Pixel size: 0.1725 µm/px (from CZI metadata)
        pixel_size_um = 0.1725
        area_px = v.get('outer_area_px', 0)
        area_um2 = area_px * (pixel_size_um ** 2)

        sample = {
            'uid': v.get('uid', 'unknown'),
            'image': img_b64,
            'stats': {
                'area_um2': area_um2,
                'area_px': area_px,
                'sma_ratio': v.get('sma_ratio', 0),
                'diameter_um': v.get('outer_diameter_um', 0),
                'scale': v.get('scale', '1/1'),
            }
        }
        samples.append(sample)

    if loader:
        loader.close()

    # Export using package template
    html_dir = os.path.join(OUTPUT_DIR, 'html')
    export_samples_to_html(
        samples=samples,
        output_dir=html_dir,
        cell_type='vessel',
        samples_per_page=100,
        title='Multiscale Vessel Detection',
        subtitle=f'Top {len(samples)} vessels by area (descending)',
        channel_legend={
            'red': 'SMA',
            'green': 'CD31',
            'blue': 'Nuclear'
        },
        extra_stats={
            'Total vessels': len(vessels),
            'Scale 1/4': sum(1 for v in vessels if v.get('scale') == '1/4'),
            'Scale 1/1': sum(1 for v in vessels if v.get('scale') == '1/1'),
        }
    )

    print(f"\nSaved to {html_dir}/index.html")

if __name__ == '__main__':
    main()
