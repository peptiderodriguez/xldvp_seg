#!/usr/bin/env python3
"""
Compute global normalization parameters from all 16 slides.
Saves parameters to JSON for use in parallel segmentation jobs.
"""

import gc
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from segmentation.io.czi_loader import get_loader
from segmentation.utils.logging import setup_logging, get_logger
from segmentation.preprocessing.stain_normalization import (
    compute_global_percentiles,
    apply_reinhard_normalization,
)
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    filter_tissue_tiles,
)

setup_logging()
logger = get_logger(__name__)

def sample_pixels_from_slide(czi_path, channel=0, n_samples=1000000):
    """Sample random pixels from TISSUE REGIONS only (using variance-based detection)."""
    logger.info(f"Sampling from {czi_path.name}...")

    try:
        loader = get_loader(str(czi_path), load_to_ram=True, channel=channel)
        channel_data = loader.get_channel_data(channel)

        if channel_data is None:
            logger.warning(f"  No data loaded for {czi_path.name}")
            return None

        # Ensure RGB
        if len(channel_data.shape) == 2:
            channel_data = np.stack([channel_data] * 3, axis=-1)

        h, w, c = channel_data.shape
        logger.info(f"  Shape: {channel_data.shape}")

        # Use proper tissue detection at block level (fast)
        logger.info(f"  Detecting tissue blocks using variance thresholding...")

        block_size = 512
        blocks = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                blocks.append({'x': x, 'y': y})

        logger.info(f"  Created {len(blocks)} blocks, calibrating tissue threshold...")

        # Calibrate tissue threshold
        variance_threshold = calibrate_tissue_threshold(
            blocks,
            image_array=channel_data,
            calibration_samples=min(100, len(blocks)),
            block_size=block_size,
            tile_size=block_size
        )

        logger.info(f"  Tissue variance threshold (computed): {variance_threshold:.1f}")

        # Reduce threshold by 3x to detect more tissue (especially lighter-stained regions)
        variance_threshold = variance_threshold / 3.0
        logger.info(f"  Tissue variance threshold (reduced 3x): {variance_threshold:.1f}")
        logger.info(f"  Filtering to tissue blocks...")

        # Filter to tissue blocks only
        tissue_blocks = filter_tissue_tiles(
            blocks,
            variance_threshold,
            image_array=channel_data,
            tile_size=block_size,
            block_size=block_size,
            n_workers=8,
            show_progress=False
        )

        if len(tissue_blocks) == 0:
            logger.warning(f"  No tissue blocks found in {czi_path.name}!")
            return None

        logger.info(f"  Found {len(tissue_blocks)} tissue blocks ({100*len(tissue_blocks)/len(blocks):.1f}%)")
        logger.info(f"  Sampling {n_samples} pixels from tissue blocks...")

        # Vectorized sampling: pick random blocks, then random offsets within each
        block_origins = np.array([(b['x'], b['y']) for b in tissue_blocks])
        block_indices = np.random.randint(0, len(tissue_blocks), size=n_samples)
        selected_origins = block_origins[block_indices]

        # Compute max valid offsets per selected block (clip to image bounds)
        max_x_offsets = np.minimum(block_size, w - selected_origins[:, 0])
        max_y_offsets = np.minimum(block_size, h - selected_origins[:, 1])

        # Generate random offsets (uniform per-sample, respecting per-block max)
        x_offsets = (np.random.random(n_samples) * max_x_offsets).astype(np.intp)
        y_offsets = (np.random.random(n_samples) * max_y_offsets).astype(np.intp)

        xs = selected_origins[:, 0] + x_offsets
        ys = selected_origins[:, 1] + y_offsets

        # Clip to image bounds (safety)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        samples = channel_data[ys, xs]  # (N, 3) -- vectorized indexing

        del block_origins, block_indices, selected_origins
        del max_x_offsets, max_y_offsets, x_offsets, y_offsets, xs, ys
        logger.info(f"  Sampled {len(samples)} tissue pixels, median intensity: {np.median(samples):.1f}")

        # Close and clear to prevent memory accumulation across slides
        loader.close()
        from segmentation.io.czi_loader import clear_cache
        clear_cache()
        del loader, channel_data
        gc.collect()

        return samples

    except Exception as e:
        logger.error(f"  Failed to sample from {czi_path.name}: {e}")
        return None

def compute_percentiles_from_samples(all_samples, p_low=1.0, p_high=99.0):
    """
    Compute global percentiles from pre-sampled pixel data.

    This is a wrapper around the module's compute_global_percentiles() that
    works with already-sampled data (for memory-efficient processing).
    """
    # Stack all samples
    combined = np.vstack(all_samples)
    logger.info(f"Computing percentiles from {len(combined):,} total samples...")

    # Compute percentiles per channel (using optimized single-pass percentile)
    if combined.ndim == 2 and combined.shape[1] == 3:  # RGB
        low_vals, high_vals = np.percentile(combined, [p_low, p_high], axis=0)
    else:
        low_vals, high_vals = np.percentile(combined, [p_low, p_high])

    return low_vals, high_vals

def main():
    logger.info("="*70)
    logger.info("COMPUTING GLOBAL NORMALIZATION PARAMETERS")
    logger.info("="*70)

    czi_dir = Path("/viper/ptmp2/edrod/2025_11_18")
    slides = sorted(czi_dir.glob("2025_11_18_*.czi"))

    logger.info(f"Found {len(slides)} slides")

    # Sample from all slides
    np.random.seed(42)  # Reproducible sampling
    all_samples = []

    for czi_path in slides:
        samples = sample_pixels_from_slide(czi_path, channel=0, n_samples=1000000)
        if samples is not None:
            all_samples.append(samples)

        # Force garbage collection after each slide to prevent memory accumulation
        gc.collect()

    if len(all_samples) == 0:
        logger.error("No samples collected!")
        return

    # Compute global median/MAD for Reinhard normalization
    logger.info("")
    logger.info("="*70)
    logger.info("Computing global median/MAD for Reinhard normalization...")

    # Stack all samples
    combined = np.vstack(all_samples)  # (N, 3) RGB
    logger.info(f"Total samples: {len(combined):,}")

    # Convert to LAB — handle both uint8 and uint16 input
    if combined.dtype == np.uint16:
        tissue_img = np.clip(combined.reshape(1, -1, 3) / 256, 0, 255).astype(np.uint8)
    else:
        tissue_img = combined.reshape(1, -1, 3).astype(np.uint8)
    lab = cv2.cvtColor(tissue_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = lab.reshape(-1, 3)  # Back to (N, 3)

    # Convert to actual LAB scale
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
    lab[:, 1] = lab[:, 1] - 128.0          # a: [0,255] -> [-128,127]
    lab[:, 2] = lab[:, 2] - 128.0          # b: [0,255] -> [-128,127]

    # Compute median and MAD for each channel
    medians = np.median(lab, axis=0)
    mads = np.median(np.abs(lab - medians), axis=0)

    # Save parameters in Reinhard format
    params = {
        'L_median': float(medians[0]),
        'L_mad': float(mads[0]),
        'a_median': float(medians[1]),
        'a_mad': float(mads[1]),
        'b_median': float(medians[2]),
        'b_mad': float(mads[2]),
        'n_slides': len(all_samples),
        'n_total_pixels': len(combined),
        'method': 'reinhard_median',
        'slides': [s.name for s in slides],
        'samples_per_slide': 1000000,
        'sampling_method': 'tissue_aware_10x_lower_threshold',
        'tile_size': 3000,
        'block_size': 512
    }

    output_file = Path("/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides_MEDIAN_NEW.json")
    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)

    logger.info("="*70)
    logger.info("GLOBAL REINHARD NORMALIZATION PARAMETERS:")
    logger.info(f"  L: median={medians[0]:.2f}, MAD={mads[0]:.2f}")
    logger.info(f"  a: median={medians[1]:.2f}, MAD={mads[1]:.2f}")
    logger.info(f"  b: median={medians[2]:.2f}, MAD={mads[2]:.2f}")
    logger.info("="*70)
    logger.info(f"\nSaved to: {output_file}")
    logger.info("="*70)

    logger.info("\nNext step: Launch 8 parallel jobs with --norm-params-file flag")

    # ── Visual validation ─────────────────────────────────────────────
    logger.info("")
    logger.info("="*70)
    logger.info("VISUAL VALIDATION: tissue maps + raw vs normalized")
    logger.info("="*70)

    # Free the sampling data before loading slides for validation
    del all_samples, combined, tissue_img, lab
    gc.collect()

    generate_visual_validation(slides, params, output_file.parent / "verification_tiles")


def _get_font(size=40):
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


def _ensure_rgb(arr):
    """Convert array to RGB uint8."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def generate_visual_validation(slide_paths, norm_params, output_dir, tile_size=3000, block_size=512, tiles_per_slide=3):
    """
    For each slide, produce:
      1. tissue_map_{slide}.png  — downsampled whole-slide with green/red tile grid
      2. comparison_{slide}_tile{N}.png — side-by-side RAW vs REINHARD NORMALIZED
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    for czi_path in slide_paths:
        slide_name = czi_path.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Validating: {slide_name}")
        logger.info(f"{'='*60}")

        loader = get_loader(str(czi_path), load_to_ram=True, channel=0)
        image_array = loader.get_channel_data(0)
        if image_array is None:
            logger.warning(f"  Failed to load {slide_name}, skipping")
            loader.close()
            del loader
            gc.collect()
            continue

        h, w = image_array.shape[:2]
        logger.info(f"  Slide: {w} x {h}")

        # ── Tissue map ────────────────────────────────────────────────
        # Build tile grid (same as segmentation pipeline)
        tiles = []
        for ty in range(0, h - tile_size + 1, tile_size):
            for tx in range(0, w - tile_size + 1, tile_size):
                tiles.append({'x': tx, 'y': ty, 'w': tile_size, 'h': tile_size})
        logger.info(f"  Total tiles ({tile_size}x{tile_size}): {len(tiles)}")

        # Calibrate threshold (same K-means as pipeline)
        threshold = calibrate_tissue_threshold(
            tiles=tiles,
            image_array=image_array,
            calibration_samples=min(100, len(tiles)),
            block_size=block_size,
            tile_size=tile_size,
        )
        logger.info(f"  Calibrated threshold: {threshold:.1f}")

        # Filter using pipeline's function
        tissue_tiles = filter_tissue_tiles(
            tiles=tiles,
            variance_threshold=threshold,
            image_array=image_array,
            tile_size=tile_size,
            block_size=block_size,
        )
        tissue_set = {(t['x'], t['y']) for t in tissue_tiles}
        bg_tiles = [t for t in tiles if (t['x'], t['y']) not in tissue_set]
        logger.info(f"  Tissue tiles: {len(tissue_tiles)} / {len(tiles)} ({100*len(tissue_tiles)/len(tiles):.1f}%)")

        # Create downsampled overview (~2000px long side)
        scale = max(1, max(w, h) // 2000)
        thumb_w, thumb_h = w // scale, h // scale
        logger.info(f"  Creating overview at 1/{scale}x ({thumb_w}x{thumb_h})...")

        # Resize directly from image_array to avoid 23GB RGB copy
        if image_array.ndim == 2:
            thumb_gray = cv2.resize(image_array, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            thumb = np.stack([thumb_gray] * 3, axis=-1).astype(np.uint8)
            del thumb_gray
        else:
            src = image_array[:, :, :3] if image_array.shape[2] > 3 else image_array
            thumb = cv2.resize(src.astype(np.uint8), (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(thumb)
        draw = ImageDraw.Draw(img, 'RGBA')

        # Red tint on rejected tiles
        for t in bg_tiles:
            x1, y1 = t['x'] // scale, t['y'] // scale
            x2, y2 = (t['x'] + tile_size) // scale, (t['y'] + tile_size) // scale
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 40), outline=(255, 0, 0, 120), width=1)

        # Green outline on tissue tiles
        for t in tissue_tiles:
            x1, y1 = t['x'] // scale, t['y'] // scale
            x2, y2 = (t['x'] + tile_size) // scale, (t['y'] + tile_size) // scale
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 200), width=2)

        # Legend
        font = _get_font(24)
        font_sm = _get_font(18)
        draw.rectangle([10, 10, 430, 130], fill=(255, 255, 255, 200))
        draw.text((20, 15), slide_name, fill=(0, 0, 0), font=font)
        draw.rectangle([20, 50, 40, 65], fill=(0, 255, 0, 200), outline=(0, 255, 0))
        draw.text((50, 47), f"Tissue: {len(tissue_tiles)} tiles", fill=(0, 100, 0), font=font_sm)
        draw.rectangle([20, 75, 40, 90], fill=(255, 0, 0, 80), outline=(255, 0, 0))
        draw.text((50, 72), f"Background: {len(bg_tiles)} tiles", fill=(150, 0, 0), font=font_sm)
        draw.text((20, 100), f"Threshold: {threshold:.1f} (K-means)", fill=(80, 80, 80), font=font_sm)

        map_path = output_dir / f"tissue_map_{slide_name}.png"
        img.convert('RGB').save(map_path, quality=95)
        logger.info(f"  Saved: {map_path}")
        all_outputs.append(map_path)

        del thumb
        gc.collect()

        # ── Raw vs Normalized tiles (all on one image per slide) ─────
        # Score tissue tiles by variance, pick top N
        scored = []
        for t in tissue_tiles:
            tx, ty = t['x'], t['y']
            tile_img = image_array[ty:ty + tile_size, tx:tx + tile_size]
            if tile_img.ndim == 3:
                gray = tile_img.mean(axis=2)
            else:
                gray = tile_img.astype(np.float32)
            scored.append((float(np.var(gray)), tx, ty))
        scored.sort(reverse=True)

        selected = scored[:tiles_per_slide]
        row_images = []
        gap = 10
        label_h = 60

        for idx, (var_score, tx, ty) in enumerate(selected):
            logger.info(f"  Tile {idx+1}: pos=({tx},{ty}), var={var_score:.0f}")

            raw_tile = _ensure_rgb(image_array[ty:ty + tile_size, tx:tx + tile_size].copy())
            norm_tile = apply_reinhard_normalization(raw_tile.copy(), norm_params)

            logger.info(f"    Raw  RGB median: R={np.median(raw_tile[:,:,0]):.1f} G={np.median(raw_tile[:,:,1]):.1f} B={np.median(raw_tile[:,:,2]):.1f}")
            logger.info(f"    Norm RGB median: R={np.median(norm_tile[:,:,0]):.1f} G={np.median(norm_tile[:,:,1]):.1f} B={np.median(norm_tile[:,:,2]):.1f}")

            # Side-by-side row: RAW | NORMALIZED
            th, tw = raw_tile.shape[:2]
            row = np.ones((th + label_h, tw * 2 + gap, 3), dtype=np.uint8) * 240
            row[label_h:, :tw] = raw_tile
            row[label_h:, tw + gap:] = norm_tile

            row_img = Image.fromarray(row)
            row_draw = ImageDraw.Draw(row_img)
            row_draw.text((20, 10), f"RAW  pos=({tx},{ty})  var={var_score:.0f}", fill=(200, 0, 0), font=_get_font(32))
            row_draw.text((tw + gap + 20, 10), "REINHARD NORMALIZED", fill=(0, 130, 0), font=_get_font(32))

            row_images.append(row_img)
            del raw_tile, norm_tile

        # Stack all rows vertically with a header
        if row_images:
            row_w = row_images[0].width
            header_h = 60
            row_gap = 6
            total_h = header_h + sum(r.height for r in row_images) + row_gap * (len(row_images) - 1)

            comp_img = Image.new('RGB', (row_w, total_h), (240, 240, 240))
            comp_draw = ImageDraw.Draw(comp_img)
            comp_draw.text((20, 12), f"{slide_name} — RAW vs REINHARD ({len(selected)} tiles)", fill=(0, 0, 0), font=_get_font(36))

            y_pos = header_h
            for row_img in row_images:
                comp_img.paste(row_img, (0, y_pos))
                y_pos += row_img.height + row_gap

            # Downscale to keep file size reasonable
            comp_img = comp_img.resize((comp_img.width // 2, comp_img.height // 2), Image.LANCZOS)
            comp_path = output_dir / f"comparison_{slide_name}.png"
            comp_img.save(comp_path, quality=95)
            logger.info(f"  Saved: {comp_path}")
            all_outputs.append(comp_path)

        del row_images
        gc.collect()

        # Free slide
        loader.close()
        del loader, image_array
        gc.collect()

    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATION COMPLETE — {len(all_outputs)} images saved to: {output_dir}/")
    for p in all_outputs:
        logger.info(f"  {p.name}")
    logger.info(f"{'='*60}")
    logger.info(f"\nInspect these images before launching segmentation!")


if __name__ == "__main__":
    main()
