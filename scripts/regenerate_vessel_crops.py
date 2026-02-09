#!/usr/bin/env python3
"""
Regenerate vessel crop images from saved detection JSON.

Reads vessel_detections_multiscale.json which contains contour coordinates,
then re-reads tile data from CZI to create annotated crop images.

Usage:
    python scripts/regenerate_vessel_crops.py
    python scripts/regenerate_vessel_crops.py --input /path/to/detections.json
"""

import numpy as np
import cv2
import json
import os
import sys
import gc
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.io.czi_loader import CZILoader

# Same constants as sam2_multiscale_vessels.py
CZI_PATH = "/home/dude/images/20251106_Fig2_nuc488_CD31_555_SMA647_PM750-EDFvar-stitch.czi"
OUTPUT_DIR = "/home/dude/vessel_output/sam2_multiscale"
NUCLEAR = 0
CD31 = 1
SMA = 2
PM = 3
BASE_SCALE = 2

SCALE_TILE_SIZES = {
    '1/64': 1000, '1/32': 1200, '1/16': 1400,
    '1/8': 1700, '1/4': 2000, '1/2': 2500,
}


def normalize_channel(arr, p_low=1, p_high=99):
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, (p_low, p_high))
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-8) * 255, 0, 255)
    return arr.astype(np.uint8)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate vessel crop images")
    parser.add_argument('--input', type=str,
                        default=os.path.join(OUTPUT_DIR, 'vessel_detections_multiscale.json'))
    args = parser.parse_args()

    # Load detections
    print(f"Loading detections from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    vessels = data['vessels']
    print(f"  {len(vessels)} vessels")

    # Setup output
    crops_dir = os.path.join(OUTPUT_DIR, 'crops')
    os.makedirs(crops_dir, exist_ok=True)

    # Load CZI
    print("Opening CZI...")
    loader = CZILoader(CZI_PATH)

    # Import DownsampledChannelCache from the main script
    from scripts.sam2_multiscale_vessels import DownsampledChannelCache
    channel_cache = DownsampledChannelCache(loader, [NUCLEAR, CD31, SMA, PM], BASE_SCALE)

    # Group vessels by (tile_x, tile_y, scale_factor) to batch tile reads
    tile_groups = {}
    for i, v in enumerate(vessels):
        key = (v['tile_x'], v['tile_y'], v['scale_factor'])
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append(i)

    print(f"  {len(tile_groups)} unique tiles to read")

    n_generated = 0
    n_skipped = 0

    for (tile_x, tile_y, scale_factor), indices in tqdm(tile_groups.items(), desc="Generating crops"):
        tile_size = SCALE_TILE_SIZES.get(f"1/{scale_factor}", 2000)

        # Read tile channels
        sma = channel_cache.get_tile(tile_x, tile_y, tile_size, SMA, scale_factor)
        cd31 = channel_cache.get_tile(tile_x, tile_y, tile_size, CD31, scale_factor)
        nuclear = channel_cache.get_tile(tile_x, tile_y, tile_size, NUCLEAR, scale_factor)

        if sma is None or cd31 is None or nuclear is None:
            n_skipped += len(indices)
            continue

        # Normalize and create display RGB
        sma_norm = normalize_channel(sma)
        cd31_norm = normalize_channel(cd31)
        nuclear_norm = normalize_channel(nuclear)
        display_rgb = np.stack([sma_norm, cd31_norm, nuclear_norm], axis=-1)

        for idx in indices:
            v = vessels[idx]
            # Contours in JSON are in full-res coordinates; convert back to tile scale
            outer_contour = np.array(v['outer_contour'], dtype=np.int32) // scale_factor
            inner_contour = np.array(v['inner_contour'], dtype=np.int32) // scale_factor

            if len(outer_contour) == 0 or len(inner_contour) == 0:
                n_skipped += 1
                continue

            # Bounding box of both contours
            all_points = np.vstack([outer_contour, inner_contour])
            x_min, y_min = all_points.min(axis=0).flatten()[:2]
            x_max, y_max = all_points.max(axis=0).flatten()[:2]

            mask_w = x_max - x_min
            mask_h = y_max - y_min
            crop_w = max(int(mask_w * 2), 100)
            crop_h = max(int(mask_h * 2), 100)

            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2

            x1 = max(0, cx - crop_w // 2)
            y1 = max(0, cy - crop_h // 2)
            x2 = min(display_rgb.shape[1], cx + crop_w // 2)
            y2 = min(display_rgb.shape[0], cy + crop_h // 2)

            if x2 <= x1 or y2 <= y1:
                n_skipped += 1
                continue

            crop_raw = display_rgb[y1:y2, x1:x2].copy()
            if crop_raw.size == 0:
                n_skipped += 1
                continue

            uid = v['uid']

            # Save raw
            raw_path = os.path.join(crops_dir, f"{uid}_raw.jpg")
            cv2.imwrite(raw_path, cv2.cvtColor(crop_raw, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Draw contours
            crop_contoured = crop_raw.copy()
            outer_in_crop = outer_contour.reshape(-1, 1, 2) - np.array([x1, y1])
            inner_in_crop = inner_contour.reshape(-1, 1, 2) - np.array([x1, y1])
            cv2.drawContours(crop_contoured, [outer_in_crop], -1, (0, 255, 0), 2)
            cv2.drawContours(crop_contoured, [inner_in_crop], -1, (0, 255, 255), 2)

            crop_path = os.path.join(crops_dir, f"{uid}.jpg")
            cv2.imwrite(crop_path, cv2.cvtColor(crop_contoured, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Update vessel dict paths
            v['crop_path'] = crop_path
            v['crop_path_raw'] = raw_path
            v['crop_offset'] = [int(x1), int(y1)]

            n_generated += 1

        del display_rgb, sma, cd31, nuclear
        gc.collect()

    print(f"\nDone! Generated {n_generated} crops, skipped {n_skipped}")

    # Save updated JSON with correct crop paths
    print(f"Updating {args.input} with new crop paths...")
    with open(args.input, 'w') as f:
        json.dump(data, f, indent=2)

    # Regenerate HTML
    print("Regenerating HTML...")
    from scripts.sam2_multiscale_vessels import generate_html
    generate_html(vessels)

    channel_cache.release()
    loader.close()
    print("All done!")


if __name__ == '__main__':
    main()
