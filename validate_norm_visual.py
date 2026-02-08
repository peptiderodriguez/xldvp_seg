#!/usr/bin/env python3
"""
Visual validation of Reinhard normalization and tissue detection.

For each slide produces:
1. Whole-slide tissue map: downsampled slide with tissue tile grid overlay
   (green = tissue, red = rejected, with K-means calibrated threshold)
2. Side-by-side raw vs normalized tiles from actual tissue regions

Output: /viper/ptmp2/edrod/xldvp_seg_fresh/verification_tiles/
"""

import json
import gc
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from segmentation.io.czi_loader import get_loader
from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    has_tissue,
    filter_tissue_tiles,
    calculate_block_variances,
)


CZI_DIR = Path("/viper/ptmp2/edrod/2025_11_18")
PARAMS_FILE = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides_MEDIAN_NEW.json")
OUTPUT_DIR = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/verification_tiles")
TILE_SIZE = 3000
BLOCK_SIZE = 512

# Pick slides from each group
SLIDES = [
    "2025_11_18_FGC1.czi",
    "2025_11_18_FGC3.czi",
    "2025_11_18_FHU2.czi",
    "2025_11_18_MGC1.czi",
    "2025_11_18_MHU4.czi",
]

TILES_PER_SLIDE = 3


def get_font(size=40):
    """Try to find a usable font."""
    for path in [
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def ensure_rgb(tile):
    """Convert tile to RGB uint8."""
    if tile is None:
        return None
    if tile.ndim == 2:
        tile = np.stack([tile] * 3, axis=-1)
    elif tile.shape[2] == 4:
        tile = tile[:, :, :3]
    if tile.dtype != np.uint8:
        if tile.max() <= 1.0:
            tile = (tile * 255).astype(np.uint8)
        else:
            tile = tile.astype(np.uint8)
    return tile


def generate_tissue_map(loader, slide_name, output_dir):
    """
    Generate a whole-slide overview with tissue detection overlay.

    Produces a downsampled slide image with:
    - Green rectangles around tiles classified as tissue
    - Red rectangles around tiles classified as background
    - The K-means calibrated threshold printed on the image
    - Tissue fraction stats
    """
    dims = loader.get_dimensions()
    h, w = dims['height'], dims['width']
    image_array = loader.get_channel_data(0)

    print(f"  Slide: {w} x {h}")

    # Build tile grid (same as pipeline)
    tiles = []
    for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
        for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
            tiles.append({'x': x, 'y': y, 'w': TILE_SIZE, 'h': TILE_SIZE})
    print(f"  Total tiles ({TILE_SIZE}x{TILE_SIZE}): {len(tiles)}")

    # Calibrate threshold using K-means (same as pipeline)
    print(f"  Calibrating tissue threshold (K-means on 100 sample tiles)...")
    threshold = calibrate_tissue_threshold(
        tiles=tiles,
        image_array=image_array,
        channel=0,
        tile_size=TILE_SIZE,
        block_size=BLOCK_SIZE,
        calibration_samples=100,
    )
    print(f"  Calibrated threshold: {threshold:.1f}")

    # Classify tiles using the same function as the pipeline
    print(f"  Filtering tissue tiles (same as pipeline: filter_tissue_tiles)...")
    tissue_tiles = filter_tissue_tiles(
        tiles=tiles,
        variance_threshold=threshold,
        image_array=image_array,
        channel=0,
        tile_size=TILE_SIZE,
        block_size=BLOCK_SIZE,
    )

    # Determine which tiles were rejected
    tissue_set = {(t['x'], t['y']) for t in tissue_tiles}
    bg_tiles = [t for t in tiles if (t['x'], t['y']) not in tissue_set]

    n_tissue = len(tissue_tiles)
    n_total = len(tiles)
    print(f"  Tissue tiles: {n_tissue} / {n_total} ({100*n_tissue/n_total:.1f}%)")

    # Create downsampled overview
    # Target ~2000px on the long side
    scale = max(1, max(w, h) // 2000)
    thumb_w, thumb_h = w // scale, h // scale
    print(f"  Creating overview at 1/{scale}x ({thumb_w} x {thumb_h})...")

    # Downsample the full slide
    rgb_array = ensure_rgb(image_array)
    thumb = cv2.resize(rgb_array, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    # Draw tile grid overlay
    img = Image.fromarray(thumb)
    draw = ImageDraw.Draw(img, 'RGBA')

    for tile in bg_tiles:
        x1 = tile['x'] // scale
        y1 = tile['y'] // scale
        x2 = (tile['x'] + TILE_SIZE) // scale
        y2 = (tile['y'] + TILE_SIZE) // scale
        # Semi-transparent red fill + red outline
        draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 40), outline=(255, 0, 0, 120), width=1)

    for tile in tissue_tiles:
        x1 = tile['x'] // scale
        y1 = tile['y'] // scale
        x2 = (tile['x'] + TILE_SIZE) // scale
        y2 = (tile['y'] + TILE_SIZE) // scale
        # Green outline only (no fill so tissue is visible)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 200), width=2)

    # Add legend
    font = get_font(24)
    font_sm = get_font(18)

    legend_y = 10
    draw.rectangle([10, legend_y, 420, legend_y + 120], fill=(255, 255, 255, 200))
    draw.text((20, legend_y + 5), f"{slide_name}", fill=(0, 0, 0), font=font)
    draw.rectangle([20, legend_y + 40, 40, legend_y + 55], fill=(0, 255, 0, 200), outline=(0, 255, 0))
    draw.text((50, legend_y + 37), f"Tissue: {n_tissue} tiles", fill=(0, 100, 0), font=font_sm)
    draw.rectangle([20, legend_y + 65, 40, legend_y + 80], fill=(255, 0, 0, 80), outline=(255, 0, 0))
    draw.text((50, legend_y + 62), f"Background: {len(bg_tiles)} tiles", fill=(150, 0, 0), font=font_sm)
    draw.text((20, legend_y + 90), f"Threshold: {threshold:.1f} (K-means)", fill=(80, 80, 80), font=font_sm)

    out_path = output_dir / f"tissue_map_{slide_name}.png"
    img = img.convert('RGB')
    img.save(out_path, quality=95)
    print(f"  Saved tissue map: {out_path}")

    del thumb, rgb_array
    gc.collect()

    return out_path, tissue_tiles, threshold


def save_side_by_side(raw, normalized, slide_name, tile_idx, tile_x, tile_y, var_score, output_dir):
    """Save raw and normalized tiles side by side."""
    h, w = raw.shape[:2]

    label_h = 80
    gap = 10
    canvas = np.ones((h + label_h, w * 2 + gap, 3), dtype=np.uint8) * 240
    canvas[label_h:, :w] = raw
    canvas[label_h:, w + gap:] = normalized

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    font = get_font(40)
    font_sm = get_font(28)

    draw.text((20, 15), "RAW", fill=(200, 0, 0), font=font)
    draw.text((w + gap + 20, 15), "REINHARD NORMALIZED", fill=(0, 130, 0), font=font)
    draw.text((20, 50), f"{slide_name}  pos=({tile_x},{tile_y})  var={var_score:.0f}", fill=(80, 80, 80), font=font_sm)

    out_path = output_dir / f"comparison_{slide_name}_tile{tile_idx}.png"
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    img.save(out_path, quality=95)
    print(f"    Saved: {out_path}")
    return out_path


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(PARAMS_FILE) as f:
        norm_params = json.load(f)
    print(f"Reinhard params: L_median={norm_params['L_median']:.2f}, L_MAD={norm_params['L_mad']:.2f}")
    print(f"                 a_median={norm_params['a_median']:.2f}, a_MAD={norm_params['a_mad']:.2f}")
    print(f"                 b_median={norm_params['b_median']:.2f}, b_MAD={norm_params['b_mad']:.2f}")

    all_outputs = []

    for slide_file in SLIDES:
        slide_name = slide_file.replace('.czi', '')
        czi_path = CZI_DIR / slide_file

        print(f"\n{'='*60}")
        print(f"Processing: {slide_name}")
        print(f"{'='*60}")

        loader = get_loader(str(czi_path), load_to_ram=True)
        loader.open()
        loader.load_channel(0)

        # --- Part 1: Whole-slide tissue map ---
        print(f"\n  --- TISSUE MAP ---")
        map_path, tissue_tiles, threshold = generate_tissue_map(loader, slide_name, OUTPUT_DIR)
        all_outputs.append(map_path)

        # --- Part 2: Raw vs normalized on tissue tiles ---
        print(f"\n  --- RAW vs NORMALIZED (top {TILES_PER_SLIDE} tissue tiles) ---")
        image_array = loader.get_channel_data(0)

        # Score tissue tiles by variance to pick the most interesting ones
        scored = []
        for tile in tissue_tiles:
            tx, ty = tile['x'], tile['y']
            tile_img = image_array[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
            if tile_img.ndim == 3:
                gray = tile_img.mean(axis=2)
            else:
                gray = tile_img.astype(np.float32)
            var = float(np.var(gray))
            scored.append((var, tx, ty))
        scored.sort(reverse=True)

        for idx, (var_score, tx, ty) in enumerate(scored[:TILES_PER_SLIDE]):
            print(f"\n  Tile {idx+1}: pos=({tx},{ty}), variance={var_score:.0f}")

            raw_tile = image_array[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE].copy()
            raw_tile = ensure_rgb(raw_tile)
            print(f"    Raw RGB mean: R={raw_tile[:,:,0].mean():.1f} G={raw_tile[:,:,1].mean():.1f} B={raw_tile[:,:,2].mean():.1f}")

            norm_tile = apply_reinhard_normalization(raw_tile.copy(), norm_params)
            print(f"    Norm RGB mean: R={norm_tile[:,:,0].mean():.1f} G={norm_tile[:,:,1].mean():.1f} B={norm_tile[:,:,2].mean():.1f}")

            out = save_side_by_side(raw_tile, norm_tile, slide_name, idx, tx, ty, var_score, OUTPUT_DIR)
            all_outputs.append(out)

            del raw_tile, norm_tile
            gc.collect()

        # Free slide
        loader.close()
        loader.clear_cache()
        del loader, image_array
        gc.collect()

    print(f"\n{'='*60}")
    print(f"DONE — {len(all_outputs)} images saved to: {OUTPUT_DIR}/")
    for p in all_outputs:
        print(f"  {p.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
