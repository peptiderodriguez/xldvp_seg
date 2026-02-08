#!/usr/bin/env python3
"""
Visual verification: pre/post normalization comparison for all 16 slides.

For each slide:
- Detects tissue blocks (Phase 1 method)
- Picks 4 sample tiles (one from each image quadrant), 2048x2048 regions
- Extracts pre-normalization tiles with green 512x512 block outlines
- Normalizes slide with apply_reinhard_normalization_MEDIAN
- Extracts post-normalization tiles with same green outlines
- Saves 2-column (pre, post) × 4-row comparison image

Also logs pre/post LAB median/MAD stats for each slide and prints summary table.
"""

import gc
import json
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from segmentation.io.czi_loader import get_loader, clear_cache
from segmentation.detection.tissue import calibrate_tissue_threshold, filter_tissue_tiles
from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN
from segmentation.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Paths
CZI_DIR = Path("/viper/ptmp2/edrod/2025_11_18")
PARAMS_FILE = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides_MEDIAN_NEW.json")
OUTPUT_DIR = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/verification_tiles")

BLOCK_SIZE = 512
TILE_SIZE = 2048  # 2048x2048 context region per sample tile
OUTPUT_TILE = 512  # downscale each tile to 512x512 for the output image


def compute_tissue_stats(image, tissue_blocks, block_sz=512, n_samples=1000000):
    """
    Compute LAB median/MAD from tissue blocks (same method as Phase 1 / normalization).

    Returns dict with L_median, L_mad, a_median, a_mad, b_median, b_mad.
    """
    h, w = image.shape[:2]
    block_origins = np.array([(b['x'], b['y']) for b in tissue_blocks])
    block_indices = np.random.randint(0, len(tissue_blocks), size=n_samples)
    selected_origins = block_origins[block_indices]

    # Per-block-bounded offsets (matches compute_normalization_params.py)
    max_x_offsets = np.minimum(block_sz, w - selected_origins[:, 0])
    max_y_offsets = np.minimum(block_sz, h - selected_origins[:, 1])
    max_x_offsets = np.maximum(max_x_offsets, 1)
    max_y_offsets = np.maximum(max_y_offsets, 1)
    x_offsets = (np.random.random(n_samples) * max_x_offsets).astype(np.intp)
    y_offsets = (np.random.random(n_samples) * max_y_offsets).astype(np.intp)
    xs = selected_origins[:, 0] + x_offsets
    ys = selected_origins[:, 1] + y_offsets

    samples = image[ys, xs]  # (n_samples, 3) uint8

    # Convert to LAB using cv2 (same as Phase 1)
    samples_img = samples.reshape(1, -1, 3)
    lab = cv2.cvtColor(samples_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = lab.reshape(-1, 3)
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
    lab[:, 1] = lab[:, 1] - 128.0           # a: [0,255] -> [-128,127]
    lab[:, 2] = lab[:, 2] - 128.0           # b: [0,255] -> [-128,127]

    L_med = float(np.median(lab[:, 0]))
    L_mad = float(np.median(np.abs(lab[:, 0] - L_med)))
    a_med = float(np.median(lab[:, 1]))
    a_mad = float(np.median(np.abs(lab[:, 1] - a_med)))
    b_med = float(np.median(lab[:, 2]))
    b_mad = float(np.median(np.abs(lab[:, 2] - b_med)))

    return {
        'L_median': L_med, 'L_mad': L_mad,
        'a_median': a_med, 'a_mad': a_mad,
        'b_median': b_med, 'b_mad': b_mad,
    }


def detect_tissue_blocks(image):
    """Detect tissue blocks using Phase 1 method (calibrate + /10 + filter)."""
    h, w = image.shape[:2]

    blocks = []
    for y in range(0, h, BLOCK_SIZE):
        for x in range(0, w, BLOCK_SIZE):
            blocks.append({'x': x, 'y': y})

    logger.info(f"  Total blocks: {len(blocks)}")

    var_threshold = calibrate_tissue_threshold(
        blocks,
        image_array=image,
        calibration_samples=min(100, len(blocks)),
        block_size=BLOCK_SIZE,
        tile_size=BLOCK_SIZE,
    )
    logger.info(f"  Calibrated threshold: {var_threshold:.1f}")

    var_threshold /= 10.0
    logger.info(f"  Reduced threshold (10x): {var_threshold:.1f}")

    tissue_blocks = filter_tissue_tiles(
        blocks,
        var_threshold,
        image_array=image,
        tile_size=BLOCK_SIZE,
        block_size=BLOCK_SIZE,
        n_workers=8,
        show_progress=False,
    )

    logger.info(f"  Tissue blocks: {len(tissue_blocks)} / {len(blocks)}")
    return tissue_blocks


def _count_tissue_in_region(tx, ty, tissue_block_set, tile_size, block_size):
    """Count how many tissue blocks fall within a tile region."""
    count = 0
    for by in range(ty, ty + tile_size, block_size):
        for bx in range(tx, tx + tile_size, block_size):
            if (bx, by) in tissue_block_set:
                count += 1
    return count


def pick_quadrant_tiles(image, tissue_blocks):
    """
    Pick 4 tile locations, one from each quadrant, choosing the densest
    tissue region (most tissue blocks within the 2048x2048 tile).

    Returns list of (x, y) top-left corners for TILE_SIZE x TILE_SIZE regions.
    """
    h, w = image.shape[:2]
    mid_x, mid_y = w // 2, h // 2

    # Build a set of tissue block origins for fast lookup
    tissue_set = set((b['x'], b['y']) for b in tissue_blocks)

    # Partition tissue blocks into 4 quadrants
    quadrants = {
        'TL': [], 'TR': [], 'BL': [], 'BR': [],
    }
    for b in tissue_blocks:
        bx, by = b['x'], b['y']
        if bx < mid_x and by < mid_y:
            quadrants['TL'].append(b)
        elif bx >= mid_x and by < mid_y:
            quadrants['TR'].append(b)
        elif bx < mid_x and by >= mid_y:
            quadrants['BL'].append(b)
        else:
            quadrants['BR'].append(b)

    tiles = []
    for qname in ['TL', 'TR', 'BL', 'BR']:
        qblocks = quadrants[qname]
        if len(qblocks) == 0:
            # Fallback: pick from any available tissue block
            qblocks = tissue_blocks

        # For each tissue block in this quadrant, score the 2048x2048 region
        # centered on it by counting how many tissue blocks it contains.
        # Sample up to 200 candidates to keep it fast.
        candidates = qblocks
        if len(candidates) > 200:
            indices = np.random.choice(len(candidates), 200, replace=False)
            candidates = [qblocks[i] for i in indices]

        best_score = -1
        best_tx, best_ty = 0, 0

        for b in candidates:
            bx, by = b['x'], b['y']
            tx = bx + BLOCK_SIZE // 2 - TILE_SIZE // 2
            ty = by + BLOCK_SIZE // 2 - TILE_SIZE // 2
            tx = max(0, min(tx, w - TILE_SIZE))
            ty = max(0, min(ty, h - TILE_SIZE))
            # Snap to block grid
            tx = (tx // BLOCK_SIZE) * BLOCK_SIZE
            ty = (ty // BLOCK_SIZE) * BLOCK_SIZE

            score = _count_tissue_in_region(tx, ty, tissue_set, TILE_SIZE, BLOCK_SIZE)
            if score > best_score:
                best_score = score
                best_tx, best_ty = tx, ty

        tiles.append((best_tx, best_ty))

    return tiles


def extract_tile(image, tx, ty):
    """
    Extract a TILE_SIZE x TILE_SIZE region.

    Returns RGB PIL Image (TILE_SIZE x TILE_SIZE).
    """
    h, w = image.shape[:2]
    ty_end = min(ty + TILE_SIZE, h)
    tx_end = min(tx + TILE_SIZE, w)

    tile = image[ty:ty_end, tx:tx_end].copy()

    # Pad if at edge
    if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
        padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        padded[:tile.shape[0], :tile.shape[1]] = tile
        tile = padded

    return Image.fromarray(tile)


def create_comparison_image(pre_tiles, post_tiles, slide_name):
    """
    Create a 2-column (pre, post) × 4-row comparison image.

    Each tile is downscaled to OUTPUT_TILE x OUTPUT_TILE.
    Output: (2 * OUTPUT_TILE) wide × (4 * OUTPUT_TILE + header) tall.
    """
    header_h = 40
    img_w = 2 * OUTPUT_TILE
    img_h = 4 * OUTPUT_TILE + header_h

    canvas = Image.new('RGB', (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header text
    try:
        font = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", 14)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_small = font

    draw.text((10, 10), f"{slide_name}", fill=(0, 0, 0), font=font)
    draw.text((OUTPUT_TILE // 2 - 30, 10), "Pre", fill=(0, 0, 200), font=font)
    draw.text((OUTPUT_TILE + OUTPUT_TILE // 2 - 30, 10), "Post", fill=(200, 0, 0), font=font)

    for row in range(4):
        pre_small = pre_tiles[row].resize((OUTPUT_TILE, OUTPUT_TILE), Image.LANCZOS)
        post_small = post_tiles[row].resize((OUTPUT_TILE, OUTPUT_TILE), Image.LANCZOS)

        y_offset = header_h + row * OUTPUT_TILE
        canvas.paste(pre_small, (0, y_offset))
        canvas.paste(post_small, (OUTPUT_TILE, y_offset))

        # Quadrant label
        q_label = ['TL', 'TR', 'BL', 'BR'][row]
        draw.text((5, y_offset + 5), q_label, fill=(255, 255, 0), font=font_small)

    return canvas


def process_slide(czi_path, params, output_dir):
    """Process a single slide: detect tissue, extract tiles, normalize, compare."""
    slide_name = czi_path.stem
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {slide_name}")
    logger.info(f"{'='*70}")

    t0 = time.time()

    # Step 1: Load slide
    logger.info("  Loading slide to RAM...")
    loader = get_loader(str(czi_path), load_to_ram=True, channel=0)
    image = loader.channel_data
    if image is None:
        logger.error(f"  Failed to load {slide_name}")
        loader.close()
        clear_cache()
        gc.collect()
        return None

    h, w = image.shape[:2]
    logger.info(f"  Image shape: {image.shape} ({image.nbytes / 1e9:.1f} GB)")

    # Step 2: Detect tissue blocks
    logger.info("  Detecting tissue blocks...")
    tissue_blocks = detect_tissue_blocks(image)

    if len(tissue_blocks) == 0:
        logger.warning(f"  No tissue found in {slide_name}, skipping")
        loader.close()
        clear_cache()
        gc.collect()
        return None

    # Step 3: Compute PRE-normalization stats
    logger.info("  Computing pre-normalization stats...")
    pre_stats = compute_tissue_stats(image, tissue_blocks)
    logger.info(f"    PRE  L: median={pre_stats['L_median']:.2f}, MAD={pre_stats['L_mad']:.2f}")
    logger.info(f"    PRE  a: median={pre_stats['a_median']:.2f}, MAD={pre_stats['a_mad']:.2f}")
    logger.info(f"    PRE  b: median={pre_stats['b_median']:.2f}, MAD={pre_stats['b_mad']:.2f}")

    # Step 4: Pick 4 tile locations (one per quadrant)
    logger.info("  Picking 4 sample tiles (one per quadrant)...")
    tile_locs = pick_quadrant_tiles(image, tissue_blocks)
    for i, (tx, ty) in enumerate(tile_locs):
        logger.info(f"    Tile {i} ({['TL','TR','BL','BR'][i]}): ({tx}, {ty})")

    # Step 5: Extract pre-normalization tiles with green outlines
    logger.info("  Extracting pre-normalization tiles...")
    pre_tiles = []
    for tx, ty in tile_locs:
        pil_tile = extract_tile(image, tx, ty)
        pre_tiles.append(pil_tile)

    # Step 6: Normalize the slide
    logger.info("  Normalizing slide (apply_reinhard_normalization_MEDIAN)...")
    normalized = apply_reinhard_normalization_MEDIAN(image, params)

    # Step 7: Compute POST-normalization stats (re-detect tissue on normalized image)
    # Use same tissue blocks from original detection for consistency
    logger.info("  Computing post-normalization stats...")
    post_stats = compute_tissue_stats(normalized, tissue_blocks)
    logger.info(f"    POST L: median={post_stats['L_median']:.2f}, MAD={post_stats['L_mad']:.2f}  (target: {params['L_median']:.2f}, {params['L_mad']:.2f})")
    logger.info(f"    POST a: median={post_stats['a_median']:.2f}, MAD={post_stats['a_mad']:.2f}  (target: {params['a_median']:.2f}, {params['a_mad']:.2f})")
    logger.info(f"    POST b: median={post_stats['b_median']:.2f}, MAD={post_stats['b_mad']:.2f}  (target: {params['b_median']:.2f}, {params['b_mad']:.2f})")

    # Step 8: Extract post-normalization tiles with same green outlines
    logger.info("  Extracting post-normalization tiles...")
    post_tiles = []
    for tx, ty in tile_locs:
        pil_tile = extract_tile(normalized, tx, ty)
        post_tiles.append(pil_tile)

    del normalized

    # Step 9: Create comparison image
    logger.info("  Creating comparison image...")
    comparison = create_comparison_image(pre_tiles, post_tiles, slide_name)

    output_path = output_dir / f"{slide_name}.png"
    comparison.save(str(output_path))
    logger.info(f"  Saved: {output_path}")

    del pre_tiles, post_tiles, comparison

    # Step 10: Cleanup
    logger.info("  Cleaning up...")
    loader.close()
    clear_cache()
    del loader, image, tissue_blocks
    gc.collect()

    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.1f}s")

    return {
        'slide': slide_name,
        'pre': pre_stats,
        'post': post_stats,
    }


def print_summary_table(all_stats, params):
    """Print summary table of pre/post stats vs targets."""
    logger.info(f"\n{'='*120}")
    logger.info("SUMMARY: Pre/Post Normalization Stats vs Targets")
    logger.info(f"{'='*120}")
    logger.info(f"Targets: L_median={params['L_median']:.2f}  L_mad={params['L_mad']:.2f}  "
                f"a_median={params['a_median']:.2f}  a_mad={params['a_mad']:.2f}  "
                f"b_median={params['b_median']:.2f}  b_mad={params['b_mad']:.2f}")
    logger.info(f"{'='*120}")

    header = f"{'Slide':<25}  {'L_med':>6} {'L_mad':>6}  {'a_med':>6} {'a_mad':>6}  {'b_med':>6} {'b_mad':>6}  |  {'L_med':>6} {'L_mad':>6}  {'a_med':>6} {'a_mad':>6}  {'b_med':>6} {'b_mad':>6}"
    logger.info(f"{'':25}  {'--- PRE ---':^42}  |  {'--- POST ---':^42}")
    logger.info(header)
    logger.info(f"{'-'*120}")

    for s in all_stats:
        pre = s['pre']
        post = s['post']
        line = (
            f"{s['slide']:<25}  "
            f"{pre['L_median']:6.2f} {pre['L_mad']:6.2f}  "
            f"{pre['a_median']:6.2f} {pre['a_mad']:6.2f}  "
            f"{pre['b_median']:6.2f} {pre['b_mad']:6.2f}  |  "
            f"{post['L_median']:6.2f} {post['L_mad']:6.2f}  "
            f"{post['a_median']:6.2f} {post['a_mad']:6.2f}  "
            f"{post['b_median']:6.2f} {post['b_mad']:6.2f}"
        )
        logger.info(line)

    logger.info(f"{'='*120}")


def main():
    logger.info("="*70)
    logger.info("VISUAL VERIFICATION: Pre/Post Normalization for All 16 Slides")
    logger.info("="*70)

    # Load normalization parameters
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    logger.info(f"Loaded params from {PARAMS_FILE}")
    logger.info(f"  L: median={params['L_median']:.2f}, MAD={params['L_mad']:.2f}")
    logger.info(f"  a: median={params['a_median']:.2f}, MAD={params['a_mad']:.2f}")
    logger.info(f"  b: median={params['b_median']:.2f}, MAD={params['b_mad']:.2f}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all CZI files
    slides = sorted(CZI_DIR.glob("2025_11_18_*.czi"))
    logger.info(f"Found {len(slides)} slides")

    np.random.seed(42)

    all_stats = []
    for i, czi_path in enumerate(slides):
        logger.info(f"\n[{i+1}/{len(slides)}] {czi_path.name}")
        result = process_slide(czi_path, params, OUTPUT_DIR)
        if result is not None:
            all_stats.append(result)

    # Print summary table
    if all_stats:
        print_summary_table(all_stats, params)

    # Final verification
    pngs = list(OUTPUT_DIR.glob("*.png"))
    logger.info(f"\nVerification: {len(pngs)}/{len(slides)} PNG files created in {OUTPUT_DIR}")
    for p in sorted(pngs):
        logger.info(f"  {p.name}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
