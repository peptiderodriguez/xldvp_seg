#!/usr/bin/env python3
"""
Compute global normalization parameters from all 16 slides.
Saves parameters to JSON for use in parallel segmentation jobs.

Pipeline:
  1. Per slide: Otsu threshold → sample 1M tissue pixels → compute per-slide LAB stats
  2. Compute initial global LAB stats from all slides combined
  3. Outlier rejection: reject slides where any of |slide_median - global_median| > 1*MAD
  4. Recompute global stats from surviving slides only
  5. Save JSON with global target + per-slide source stats + rejected list
"""

import argparse
import gc
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from segmentation.io.czi_loader import get_loader
from segmentation.utils.logging import setup_logging, get_logger
from segmentation.preprocessing.stain_normalization import (
    apply_reinhard_normalization,
)
from segmentation.detection.tissue import (
    compute_otsu_threshold,
    has_tissue,
)

setup_logging()
logger = get_logger(__name__)


def _rgb_samples_to_lab_stats(samples_rgb):
    """Convert RGB samples (N, 3) uint8 to LAB and compute median+MAD per channel.

    Returns:
        dict with L_median, L_mad, a_median, a_mad, b_median, b_mad
    """
    samples_img = np.clip(samples_rgb, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    lab = cv2.cvtColor(samples_img, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0  # L: [0,255] -> [0,100]
    lab[:, 1] -= 128.0                       # a: [0,255] -> [-128,127]
    lab[:, 2] -= 128.0                       # b: [0,255] -> [-128,127]

    L_med = float(np.median(lab[:, 0]))
    a_med = float(np.median(lab[:, 1]))
    b_med = float(np.median(lab[:, 2]))

    return {
        'L_median': L_med,
        'L_mad': float(np.median(np.abs(lab[:, 0] - L_med))),
        'a_median': a_med,
        'a_mad': float(np.median(np.abs(lab[:, 1] - a_med))),
        'b_median': b_med,
        'b_mad': float(np.median(np.abs(lab[:, 2] - b_med))),
    }


def sample_pixels_from_slide(czi_path, channel=0, n_samples=1000000):
    """Sample tissue pixels from a slide using Otsu-only tissue detection.

    For each slide:
      1. Load RGB → grayscale → compute Otsu threshold
      2. Rejection-sample n_samples tissue pixels: (gray > 0) & (gray < otsu)
      3. Compute per-slide LAB stats from samples

    Returns:
        tuple: (samples_rgb, otsu_threshold, per_slide_lab_stats) or (None, None, None)
    """
    logger.info(f"Sampling from {czi_path.name}...")

    try:
        loader = get_loader(str(czi_path), load_to_ram=True, channel=channel)
        channel_data = loader.get_channel_data(channel)

        if channel_data is None:
            logger.warning(f"  No data loaded for {czi_path.name}")
            return None, None, None

        # Ensure RGB
        if len(channel_data.shape) == 2:
            channel_data = np.stack([channel_data] * 3, axis=-1)

        h, w, c = channel_data.shape
        logger.info(f"  Shape: {channel_data.shape}")

        # Compute grayscale
        full_gray = cv2.cvtColor(channel_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Compute Otsu threshold (subsampled, excludes black padding)
        otsu_val = compute_otsu_threshold(full_gray)
        logger.info(f"  Otsu threshold: {otsu_val:.1f}")

        # Count tissue pixels for logging
        # Use subsampled estimate to avoid full boolean array allocation
        n_check = min(5_000_000, h * w)
        check_rows = np.random.randint(0, h, size=n_check)
        check_cols = np.random.randint(0, w, size=n_check)
        check_g = full_gray[check_rows, check_cols]
        tissue_frac = float(np.mean((check_g > 0) & (check_g < otsu_val)))
        logger.info(f"  Estimated tissue fraction: {tissue_frac*100:.1f}%")

        if tissue_frac < 0.001:
            logger.warning(f"  <0.1% tissue pixels in {czi_path.name}, skipping")
            loader.close()
            from segmentation.io.czi_loader import clear_cache
            clear_cache()
            del loader, channel_data, full_gray
            gc.collect()
            return None, None, None

        # Rejection-sample n_samples tissue pixels
        logger.info(f"  Rejection-sampling {n_samples:,} tissue pixels...")
        collected = []
        total_collected = 0
        max_attempts = int(n_samples / max(tissue_frac, 0.01) * 2)
        batch_size = min(n_samples * 3, max_attempts)

        attempt = 0
        while total_collected < n_samples and attempt < max_attempts:
            remaining = n_samples - total_collected
            batch = min(batch_size, int(remaining / max(tissue_frac, 0.01) * 2))
            batch = max(batch, remaining)  # at least try remaining

            rows = np.random.randint(0, h, size=batch)
            cols = np.random.randint(0, w, size=batch)
            g = full_gray[rows, cols]
            mask = (g > 0) & (g < otsu_val)

            if mask.any():
                tissue_rows = rows[mask]
                tissue_cols = cols[mask]
                rgb_pixels = channel_data[tissue_rows, tissue_cols]  # (K, 3)
                collected.append(rgb_pixels)
                total_collected += len(rgb_pixels)
            attempt += batch

        if total_collected == 0:
            logger.warning(f"  No tissue pixels found after {max_attempts} attempts in {czi_path.name}, skipping")
            loader.close()
            from segmentation.io.czi_loader import clear_cache
            clear_cache()
            del loader, channel_data, full_gray
            gc.collect()
            return None, None, None

        samples = np.concatenate(collected)[:n_samples]
        del collected, full_gray
        logger.info(f"  Sampled {len(samples):,} tissue pixels, median RGB intensity: {np.median(samples):.1f}")

        # Compute per-slide LAB stats
        per_slide_stats = _rgb_samples_to_lab_stats(samples)
        per_slide_stats['otsu_threshold'] = float(otsu_val)
        logger.info(f"  Per-slide LAB: L={per_slide_stats['L_median']:.2f}±{per_slide_stats['L_mad']:.2f}, "
                     f"a={per_slide_stats['a_median']:.2f}±{per_slide_stats['a_mad']:.2f}, "
                     f"b={per_slide_stats['b_median']:.2f}±{per_slide_stats['b_mad']:.2f}")

        # Close and clear to prevent memory accumulation across slides
        loader.close()
        from segmentation.io.czi_loader import clear_cache
        clear_cache()
        del loader, channel_data
        gc.collect()

        return samples, otsu_val, per_slide_stats

    except Exception as e:
        logger.error(f"  Failed to sample from {czi_path.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Compute Reinhard normalization parameters")
    parser.add_argument('--n-slides', type=int, default=None,
                        help='Number of slides to use (default: all). Selects evenly spaced slides.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: reinhard_params_16slides_MEDIAN_NEW.json)')
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("COMPUTING GLOBAL NORMALIZATION PARAMETERS (Otsu-only)")
    logger.info("="*70)

    czi_dir = Path("/viper/ptmp2/edrod/2025_11_18")
    slides = sorted(czi_dir.glob("2025_11_18_*.czi"))

    logger.info(f"Found {len(slides)} slides total")

    if args.n_slides is not None and args.n_slides < len(slides):
        # Select evenly spaced slides for representative sampling
        indices = np.linspace(0, len(slides) - 1, args.n_slides, dtype=int)
        slides = [slides[i] for i in indices]
        logger.info(f"Selected {len(slides)} slides: {[s.stem for s in slides]}")

    # ── Phase 1: Sample from all slides ──────────────────────────────
    np.random.seed(42)  # Reproducible sampling
    all_samples = {}     # slide_name -> samples (N, 3) RGB
    tissue_thresholds = {}  # slide_name -> per-slide stats dict

    for czi_path in slides:
        samples, otsu_val, per_slide_stats = sample_pixels_from_slide(czi_path, channel=0, n_samples=1000000)
        if samples is not None:
            slide_name = czi_path.stem
            all_samples[slide_name] = samples
            tissue_thresholds[slide_name] = per_slide_stats
        gc.collect()

    if len(all_samples) == 0:
        logger.error("No samples collected!")
        return

    # ── Phase 2: Compute initial global LAB stats ────────────────────
    logger.info("")
    logger.info("="*70)
    logger.info("Computing initial global LAB stats from ALL slides...")

    combined = np.vstack(list(all_samples.values()))
    logger.info(f"Total samples: {len(combined):,} from {len(all_samples)} slides")

    initial_global_stats = _rgb_samples_to_lab_stats(combined)
    logger.info(f"  Initial global: L={initial_global_stats['L_median']:.2f}±{initial_global_stats['L_mad']:.2f}, "
                 f"a={initial_global_stats['a_median']:.2f}±{initial_global_stats['a_mad']:.2f}, "
                 f"b={initial_global_stats['b_median']:.2f}±{initial_global_stats['b_mad']:.2f}")

    # ── Phase 3: Outlier rejection ───────────────────────────────────
    # Reject if ANY of L, a, b deviates more than 1*MAD from the global median
    logger.info("")
    logger.info("="*70)
    logger.info("Outlier rejection: |slide_median - global_median| > 1*MAD on ANY of L, a, b")

    channels = [
        ('L', initial_global_stats['L_median'], initial_global_stats['L_mad']),
        ('a', initial_global_stats['a_median'], initial_global_stats['a_mad']),
        ('b', initial_global_stats['b_median'], initial_global_stats['b_mad']),
    ]
    for ch_name, ch_med, ch_mad in channels:
        logger.info(f"  Global {ch_name}: median={ch_med:.2f}, MAD={ch_mad:.2f}, range=[{ch_med - ch_mad:.2f}, {ch_med + ch_mad:.2f}]")

    rejected_slides = []
    surviving_slides = []

    for slide_name, stats in tissue_thresholds.items():
        reject_reasons = []
        for ch_name, ch_med, ch_mad in channels:
            slide_val = stats[f'{ch_name}_median']
            deviation = abs(slide_val - ch_med)
            if deviation > ch_mad:
                reject_reasons.append(f"{ch_name}={slide_val:.2f} (dev={deviation:.2f} > MAD={ch_mad:.2f})")

        if reject_reasons:
            rejected_slides.append(slide_name)
            logger.info(f"  REJECTED: {slide_name} — {'; '.join(reject_reasons)}")
        else:
            surviving_slides.append(slide_name)
            logger.info(f"  OK: {slide_name} — L={stats['L_median']:.2f}, a={stats['a_median']:.2f}, b={stats['b_median']:.2f}")

    logger.info(f"\n  Rejected: {len(rejected_slides)} / {len(tissue_thresholds)} slides")

    # ── Phase 4: Recompute global stats from survivors ───────────────
    if rejected_slides:
        logger.info("")
        logger.info("Recomputing global stats from surviving slides only...")
        surviving_samples = [all_samples[sn] for sn in surviving_slides]
        combined = np.vstack(surviving_samples)
        del surviving_samples
    # else: combined already has all samples

    logger.info(f"Final samples: {len(combined):,} from {len(surviving_slides)} slides")

    final_stats = _rgb_samples_to_lab_stats(combined)

    logger.info("="*70)
    logger.info("FINAL GLOBAL REINHARD TARGET PARAMETERS:")
    logger.info(f"  L: median={final_stats['L_median']:.2f}, MAD={final_stats['L_mad']:.2f}")
    logger.info(f"  a: median={final_stats['a_median']:.2f}, MAD={final_stats['a_mad']:.2f}")
    logger.info(f"  b: median={final_stats['b_median']:.2f}, MAD={final_stats['b_mad']:.2f}")
    logger.info("="*70)

    # ── Save JSON ────────────────────────────────────────────────────
    params = {
        'L_median': final_stats['L_median'],
        'L_mad': final_stats['L_mad'],
        'a_median': final_stats['a_median'],
        'a_mad': final_stats['a_mad'],
        'b_median': final_stats['b_median'],
        'b_mad': final_stats['b_mad'],
        'n_slides': len(surviving_slides),
        'n_total_pixels': len(combined),
        'method': 'reinhard_median',
        'slides': [s.name for s in slides],
        'samples_per_slide': 1000000,
        'sampling_method': 'otsu_rejection_sampling',
        'tissue_thresholds': tissue_thresholds,
        'rejected_slides': rejected_slides,
        'rejection_criterion': 'any of L/a/b median outside 1*MAD of global',
    }

    output_file = Path(args.output) if args.output else Path("/viper/ptmp2/edrod/xldvp_seg_fresh/reinhard_params_16slides_MEDIAN_NEW.json")
    with open(output_file, 'w') as f:
        json.dump(params, f)

    logger.info(f"\nSaved to: {output_file}")

    # ── Visual validation ─────────────────────────────────────────────
    logger.info("")
    logger.info("="*70)
    logger.info("VISUAL VALIDATION: tissue maps + raw vs normalized")
    logger.info("="*70)

    # Free the sampling data before loading slides for validation
    del all_samples, combined
    gc.collect()

    # Only validate surviving slides (skip rejected ones)
    surviving_slides = [s for s in slides if s.stem not in rejected_slides]
    logger.info(f"Validating {len(surviving_slides)} surviving slides (skipping {len(rejected_slides)} rejected)")

    generate_visual_validation(surviving_slides, params, output_file.parent / "verification_tiles",
                               tissue_thresholds=tissue_thresholds)


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


def generate_visual_validation(slide_paths, norm_params, output_dir, tissue_thresholds,
                               tile_size=3000, block_size=512, tiles_per_slide=3):
    """
    For each slide, produce:
      1. tissue_map_{slide}.png  — downsampled whole-slide with green/red tile grid
      2. comparison_{slide}.png  — side-by-side RAW vs REINHARD NORMALIZED

    Uses the EXACT same has_tissue() and apply_reinhard_normalization() code paths
    as step 2 segmentation, ensuring validation matches production behavior.
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

        # Use pre-computed per-slide thresholds
        slide_thresh = tissue_thresholds.get(slide_name)
        if slide_thresh is None:
            logger.warning(f"  No precomputed thresholds for {slide_name}, skipping")
            loader.close()
            del loader, image_array
            gc.collect()
            continue

        otsu_thresh = slide_thresh['otsu_threshold']
        logger.info(f"  Using step 1 Otsu threshold: {otsu_thresh:.1f}")

        # Filter tiles using has_tissue(modality='brightfield') — SAME as step 2
        # variance_threshold=0 (unused for brightfield), intensity_threshold=otsu
        tissue_tiles = []
        bg_tiles = []
        for t in tiles:
            tile_img = image_array[t['y']:t['y']+t['h'], t['x']:t['x']+t['w']]
            has_t, _ = has_tissue(tile_img, 0, block_size=block_size,
                                  intensity_threshold=otsu_thresh, modality='brightfield')
            if has_t:
                tissue_tiles.append(t)
            else:
                bg_tiles.append(t)
        logger.info(f"  Tissue tiles: {len(tissue_tiles)} / {len(tiles)} ({100*len(tissue_tiles)/max(len(tiles),1):.1f}%)")

        # Create downsampled overview (~2000px long side)
        scale = max(1, max(w, h) // 2000)
        thumb_w, thumb_h = w // scale, h // scale
        logger.info(f"  Creating overview at 1/{scale}x ({thumb_w}x{thumb_h})...")

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
        draw.rectangle([10, 10, 530, 130], fill=(255, 255, 255, 200))
        draw.text((20, 15), slide_name, fill=(0, 0, 0), font=font)
        draw.rectangle([20, 50, 40, 65], fill=(0, 255, 0, 200), outline=(0, 255, 0))
        draw.text((50, 47), f"Tissue: {len(tissue_tiles)} tiles", fill=(0, 100, 0), font=font_sm)
        draw.rectangle([20, 75, 40, 90], fill=(255, 0, 0, 80), outline=(255, 0, 0))
        draw.text((50, 72), f"Background: {len(bg_tiles)} tiles", fill=(150, 0, 0), font=font_sm)
        draw.text((20, 100), f"otsu={otsu_thresh:.0f} (brightfield Otsu-only)", fill=(80, 80, 80), font=font_sm)

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
                gray = cv2.cvtColor(tile_img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            else:
                gray = tile_img.astype(np.float32)
            scored.append((float(np.var(gray)), tx, ty))
        scored.sort(reverse=True)

        selected = scored[:tiles_per_slide]
        row_images = []
        gap = 10
        label_h = 60

        # Build per-slide LAB stats dict for normalization (same keys as step 2 reads)
        slide_lab_stats = {
            'L_median': slide_thresh['L_median'],
            'L_mad': slide_thresh['L_mad'],
            'a_median': slide_thresh['a_median'],
            'a_mad': slide_thresh['a_mad'],
            'b_median': slide_thresh['b_median'],
            'b_mad': slide_thresh['b_mad'],
        }

        for idx, (var_score, tx, ty) in enumerate(selected):
            logger.info(f"  Tile {idx+1}: pos=({tx},{ty}), var={var_score:.0f}")

            raw_tile = _ensure_rgb(image_array[ty:ty + tile_size, tx:tx + tile_size].copy())

            # Use EXACT same apply_reinhard_normalization as step 2
            norm_tile = apply_reinhard_normalization(
                raw_tile.copy(), norm_params,
                otsu_threshold=otsu_thresh,
                slide_lab_stats=slide_lab_stats,
            )

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
        from segmentation.io.czi_loader import clear_cache
        clear_cache()
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
