#!/usr/bin/env python3
"""Detect bright regions from a CZI channel and prepare for LMD export.

Percentile-thresholds a single channel, applies morphological cleanup,
splits large regions into equal-area pieces, extracts full features
(morphology + per-channel intensity + SAM2 embeddings), and outputs a
pipeline-compatible detection JSON.

Use for any bright-region detection where the full segmentation pipeline
is overkill: NfL nerve fibers, BTX NMJ regions, autofluorescent deposits,
bright marker+ tissue patches, etc.

Usage:
    # Basic: threshold NfL channel, split into 250 um² pieces
    python scripts/detect_regions_for_lmd.py \
        --czi-path slide.czi \
        --channel 2 \
        --output-dir output/ \
        --percentile 98 \
        --target-area-um2 250

    # With SAM2 embeddings (needs GPU)
    python scripts/detect_regions_for_lmd.py \
        --czi-path slide.czi \
        --channel-spec "detect=NfL" \
        --output-dir output/ \
        --percentile 98 \
        --target-area-um2 250 \
        --sam2

    # Custom morphology: larger closing, more dilation
    python scripts/detect_regions_for_lmd.py \
        --czi-path slide.czi \
        --channel 2 \
        --output-dir output/ \
        --percentile 95 \
        --close-radius 7 \
        --dilate-radius 5 \
        --min-area-um2 500
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt, label as ndlabel
from skimage.measure import (
    block_reduce, regionprops, find_contours, approximate_polygon,
)
from skimage.morphology import remove_small_objects, closing, dilation, erosion, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.draw import polygon as skpolygon

from segmentation.io.czi_loader import CZILoader
from segmentation.utils.json_utils import atomic_json_dump
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------

def load_channel_reduced(loader, channel_idx, block_size=8):
    """Load a CZI channel and downsample via block_reduce.

    Returns:
        reduced: 2D float32 array at reduced resolution
        full_shape: (full_h, full_w) at native resolution
        scale: actual scale factor (1/block_size)
    """
    logger.info(f"  Loading channel {channel_idx} to RAM...")
    loader.load_channel(channel_idx)
    full = loader.get_channel_data(channel_idx)
    full_shape = full.shape
    logger.info(f"  Full res: {full.shape}, dtype: {full.dtype}")

    logger.info(f"  Downsampling {block_size}x via block_reduce (mean)...")
    reduced = block_reduce(full, (block_size, block_size), np.mean).astype(np.float32)
    del full
    logger.info(f"  Reduced: {reduced.shape}")

    return reduced, full_shape, 1.0 / block_size


# ---------------------------------------------------------------------------
# Thresholding + morphological cleanup
# ---------------------------------------------------------------------------

def threshold_and_clean(channel, percentile, close_radius, dilate_radius,
                        min_area_um2, reduced_pixel_size):
    """Percentile-threshold a channel and apply morphological cleanup.

    Args:
        channel: 2D float32 array (reduced resolution)
        percentile: threshold percentile (e.g., 98 = top 2%)
        close_radius: disk radius for morphological closing
        dilate_radius: disk radius for dilate→erode rounding
        min_area_um2: minimum region area in µm²
        reduced_pixel_size: µm per pixel at reduced resolution

    Returns:
        labeled: labeled connected components
        n_regions: number of regions
    """
    # Gaussian smooth to avoid blocky edges
    logger.info("  Gaussian smooth (sigma=2)...")
    smoothed = gaussian_filter(channel, sigma=2)

    # Percentile threshold on nonzero pixels
    valid = smoothed[smoothed > 0]
    if len(valid) == 0:
        logger.warning("No valid pixels in channel")
        return np.zeros_like(channel, dtype=np.int32), 0

    thresh = float(np.percentile(valid, percentile))
    p95 = float(np.percentile(valid, 95))
    p99 = float(np.percentile(valid, 99))
    logger.info(f"  p95={p95:.0f}, p{percentile}={thresh:.0f}, p99={p99:.0f}")
    mask = smoothed > thresh

    # Morphological cleanup: close → fill → dilate → erode
    if close_radius > 0:
        mask = closing(mask, disk(close_radius))
    mask = binary_fill_holes(mask)
    if dilate_radius > 0:
        mask = dilation(mask, disk(dilate_radius))
        mask = erosion(mask, disk(dilate_radius))

    # Remove small objects
    min_px = int(min_area_um2 / (reduced_pixel_size ** 2))
    mask = remove_small_objects(mask, min_size=max(min_px, 10))

    labeled, n_regions = ndlabel(mask)
    logger.info(f"  {n_regions} regions above p{percentile} (min {min_area_um2} µm²)")

    return labeled, n_regions


# ---------------------------------------------------------------------------
# Recursive splitting
# ---------------------------------------------------------------------------

def split_region(region_mask, target_px, depth=0, max_depth=10):
    """Recursively split a binary mask into pieces close to target_px area.

    Uses watershed with distance-transform seeds. Falls back to grid slicing
    along the longest axis when watershed can't find enough seeds (thin/elongated
    regions).

    Returns: list of binary masks, each close to target_px area.
    """
    area = region_mask.sum()
    if area < target_px * 0.9:
        return []  # too small
    if area <= target_px * 1.1:
        return [region_mask]  # just right
    if depth > max_depth:
        return [region_mask]  # safety limit

    n = max(2, int(area / target_px))

    # Try watershed
    dist = distance_transform_edt(region_mask)
    min_dist = max(2, int(np.sqrt(area / n) / 2))
    pts = peak_local_max(dist, min_distance=min_dist, num_peaks=n)
    if len(pts) < 2:
        pts = peak_local_max(dist, min_distance=1, num_peaks=n)

    if len(pts) >= 2:
        markers = np.zeros_like(region_mask, dtype=np.int32)
        for j, (r, c) in enumerate(pts):
            markers[r, c] = j + 1
        ws = watershed(-dist, markers, mask=region_mask)
        result = []
        for lbl in range(1, ws.max() + 1):
            result.extend(split_region(ws == lbl, target_px, depth + 1, max_depth))
        return result

    # Fallback: grid split along longest axis
    rows, cols = np.where(region_mask)
    if len(rows) == 0:
        return []
    r_range = rows.max() - rows.min()
    c_range = cols.max() - cols.min()
    result = []

    if c_range >= r_range:
        sorted_vals = np.sort(np.unique(cols))
        chunk = max(1, len(sorted_vals) // n)
        for ci in range(0, len(sorted_vals), chunk):
            val_set = set(sorted_vals[ci:ci + chunk])
            sub = region_mask.copy()
            for c in range(region_mask.shape[1]):
                if c not in val_set:
                    sub[:, c] = False
            result.extend(split_region(sub, target_px, depth + 1, max_depth))
    else:
        sorted_vals = np.sort(np.unique(rows))
        chunk = max(1, len(sorted_vals) // n)
        for ri in range(0, len(sorted_vals), chunk):
            val_set = set(sorted_vals[ri:ri + chunk])
            sub = region_mask.copy()
            for r in range(region_mask.shape[0]):
                if r not in val_set:
                    sub[r, :] = False
            result.extend(split_region(sub, target_px, depth + 1, max_depth))

    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_morph_features(piece_mask, reduced_pixel_size):
    """Extract morphological features from a binary mask."""
    labeled = piece_mask.astype(np.int32)
    rp = regionprops(labeled)
    if not rp:
        return {}
    p = rp[0]
    perimeter = max(p.perimeter, 1)
    return {
        'area': int(p.area / (1.0 ** 2)),  # will be rescaled later
        'area_um2': float(p.area * reduced_pixel_size ** 2),
        'perimeter': float(perimeter),
        'circularity': float(4 * math.pi * p.area / (perimeter ** 2)),
        'solidity': float(p.solidity),
        'eccentricity': float(p.eccentricity),
        'aspect_ratio': float(p.major_axis_length / max(p.minor_axis_length, 1)),
        'extent': float(p.extent),
        'equiv_diameter': float(p.equivalent_diameter_area * reduced_pixel_size),
        'elongation': float(p.major_axis_length / max(p.minor_axis_length, 1)),
    }


def extract_channel_features(piece_mask, channel_arrays):
    """Extract per-channel intensity statistics within a mask."""
    feats = {}
    for ch_idx, ch_data in channel_arrays.items():
        pixels = ch_data[piece_mask]
        if len(pixels) == 0:
            continue
        mean = float(np.mean(pixels))
        feats[f'ch{ch_idx}_mean'] = mean
        feats[f'ch{ch_idx}_std'] = float(np.std(pixels))
        feats[f'ch{ch_idx}_median'] = float(np.median(pixels))
        feats[f'ch{ch_idx}_max'] = float(np.max(pixels))
        feats[f'ch{ch_idx}_min'] = float(np.min(pixels))
        p5, p25, p75, p95 = np.percentile(pixels, [5, 25, 75, 95])
        feats[f'ch{ch_idx}_p5'] = float(p5)
        feats[f'ch{ch_idx}_p25'] = float(p25)
        feats[f'ch{ch_idx}_p75'] = float(p75)
        feats[f'ch{ch_idx}_p95'] = float(p95)
        feats[f'ch{ch_idx}_iqr'] = float(p75 - p25)
        feats[f'ch{ch_idx}_cv'] = float(np.std(pixels) / max(mean, 1e-6))
    return feats


def extract_sam2_embeddings(detections, czi_path, loader, device='cuda'):
    """Extract SAM2 256-dim embeddings for each detection.

    Crops a 512x512 region around each detection center from the CZI,
    builds a 2-channel RGB (detect channel + nuclear channel), and
    runs SAM2 encoder to get a pooled 256-dim embedding.
    """
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from segmentation.utils.detection_utils import _percentile_normalize_single

    repo = Path(__file__).resolve().parent.parent
    checkpoint = str(repo / 'checkpoints' / 'sam2.1_hiera_large.pt')
    config = 'configs/sam2.1/sam2.1_hiera_l.yaml'

    if not Path(checkpoint).exists():
        logger.warning(f"SAM2 checkpoint not found at {checkpoint}, skipping embeddings")
        return

    logger.info(f"  Loading SAM2 model on {device}...")
    sam2_model = build_sam2(config, checkpoint, device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)

    # Need full-res channels for SAM2 crops
    mx, my = loader.mosaic_origin
    n_channels = loader.get_num_channels()

    # Use first 2 loaded channels for RGB (detect channel = R, first other = G)
    loaded = list(loader.loaded_channels())
    if len(loaded) < 1:
        logger.warning("No channels loaded for SAM2, skipping")
        return
    ch_r = loaded[0] if loaded else 0
    ch_g = loaded[1] if len(loaded) > 1 else loaded[0]
    data_r = loader.get_channel_data(ch_r)
    data_g = loader.get_channel_data(ch_g)

    crop_size = 512
    half = crop_size // 2
    n_done = 0

    for det in detections:
        gc = det.get('global_center')
        if gc is None:
            continue
        cx, cy = int(gc[0]), int(gc[1])

        y1 = max(0, cy - half)
        y2 = min(data_r.shape[0], cy + half)
        x1 = max(0, cx - half)
        x2 = min(data_r.shape[1], cx + half)
        if y2 - y1 < 64 or x2 - x1 < 64:
            continue

        crop_r = _percentile_normalize_single(data_r[y1:y2, x1:x2])
        crop_g = _percentile_normalize_single(data_g[y1:y2, x1:x2])
        rgb = np.stack([crop_r, crop_g, np.zeros_like(crop_r)], axis=-1)

        with torch.inference_mode():
            predictor.set_image(rgb)
            img_embed = predictor.get_image_embedding()
            embed = img_embed.mean(dim=(2, 3)).squeeze().cpu().numpy()

        for si in range(256):
            det['features'][f'sam2_{si}'] = float(embed[si])

        n_done += 1
        if n_done % 200 == 0:
            logger.info(f"    SAM2: {n_done}/{len(detections)}...")

    logger.info(f"  SAM2 embeddings: {n_done}/{len(detections)} pieces")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def detect_regions(args):
    """Main detection + featurization pipeline."""
    setup_logging(level="INFO")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve channel
    loader = CZILoader(args.czi_path)
    pixel_size = loader.get_pixel_size()
    full_w, full_h = loader.mosaic_size
    mosaic_x, mosaic_y = loader.mosaic_origin

    if args.channel is not None:
        detect_ch = args.channel
    elif args.channel_spec:
        spec_val = args.channel_spec.split('=', 1)[-1].strip()
        try:
            detect_ch = int(spec_val)
        except ValueError:
            from segmentation.io.czi_loader import parse_markers_from_filename
            markers = parse_markers_from_filename(Path(args.czi_path).name)
            detect_ch = None
            for i, m in enumerate(markers):
                if spec_val.lower() in m.get('name', '').lower():
                    detect_ch = i
                    break
            if detect_ch is None:
                logger.error(f"Could not resolve channel spec '{args.channel_spec}'")
                sys.exit(1)
        logger.info(f"Resolved channel spec → channel {detect_ch}")
    else:
        logger.error("Provide --channel or --channel-spec")
        sys.exit(1)

    logger.info(f"CZI: {Path(args.czi_path).name}")
    logger.info(f"  Mosaic: {full_w}x{full_h}, pixel_size={pixel_size:.4f} µm/px")
    logger.info(f"  Detect channel: {detect_ch}")

    block = args.block_size
    scale = 1.0 / block
    reduced_ps = pixel_size / scale

    # Step 1: Load + downsample detect channel
    detect_reduced, full_shape, scale = load_channel_reduced(loader, detect_ch, block)

    # Step 2: Threshold + morphological cleanup
    labeled, n_regions = threshold_and_clean(
        detect_reduced, args.percentile, args.close_radius,
        args.dilate_radius, args.min_area_um2, reduced_ps)

    if n_regions == 0:
        logger.warning("No regions found. Try lowering --percentile or --min-area-um2.")
        return

    # Step 3: Split into pieces + extract contours
    target_px = int(args.target_area_um2 / (reduced_ps ** 2))
    props = regionprops(labeled)
    slide_name = Path(args.czi_path).stem

    logger.info(f"  Splitting {len(props)} regions into ~{args.target_area_um2} µm² pieces...")

    detections = []
    for prop in props:
        area_um2 = prop.area * (reduced_ps ** 2)
        if area_um2 < args.min_area_um2:
            continue

        minr, minc, maxr, maxc = prop.bbox
        pad = 2
        minr_p = max(0, minr - pad)
        minc_p = max(0, minc - pad)
        maxr_p = min(labeled.shape[0], maxr + pad)
        maxc_p = min(labeled.shape[1], maxc + pad)
        region_mask = labeled[minr_p:maxr_p, minc_p:maxc_p] == prop.label

        pieces = split_region(region_mask, target_px)

        for piece_mask in pieces:
            piece_area_px = piece_mask.sum()
            piece_area_um2 = piece_area_px * (reduced_ps ** 2)
            if piece_area_um2 < args.target_area_um2 * 0.9:
                continue

            # Smooth contour
            piece_smooth = gaussian_filter(piece_mask.astype(np.float64), sigma=1.5)
            contours = find_contours(piece_smooth, 0.5)
            if not contours:
                continue
            contour = approximate_polygon(max(contours, key=len), tolerance=1.0)

            # Global full-res coordinates
            contour_global = []
            for r, c in contour:
                gx = float((c + minc_p) / scale + mosaic_x)
                gy = float((r + minr_p) / scale + mosaic_y)
                contour_global.append([gx, gy])

            prows, pcols = np.where(piece_mask)
            cx = float((np.mean(pcols) + minc_p) / scale + mosaic_x)
            cy = float((np.mean(prows) + minr_p) / scale + mosaic_y)

            # Create full-image mask for feature extraction
            full_mask = np.zeros(detect_reduced.shape[:2], dtype=bool)
            # Map piece_mask back to full reduced image coords
            pr, pc = np.where(piece_mask)
            full_mask[pr + minr_p, pc + minc_p] = True

            uid = f"region_{int(cx)}_{int(cy)}"
            feats = extract_morph_features(full_mask, reduced_ps)
            feats['area'] = int(piece_area_px / (scale ** 2))
            feats['pixel_size_um'] = pixel_size
            feats['detection_method'] = f'percentile_{args.percentile}'

            detections.append({
                'id': uid,
                'uid': uid,
                'center': [cx - mosaic_x, cy - mosaic_y],
                'global_center': [cx, cy],
                'global_center_um': [cx * pixel_size, cy * pixel_size],
                'slide_name': slide_name,
                'pixel_size_um': pixel_size,
                'outer_contour_global': contour_global,
                'contour_dilated_px': contour_global,
                'features': feats,
                'rf_prediction': None,
                '_full_mask_coords': (pr + minr_p, pc + minc_p),  # temp, removed before save
            })

    logger.info(f"  {len(detections)} pieces from {len(props)} regions")

    # Step 4: Per-channel intensity features
    logger.info("  Loading all channels for intensity features...")
    channel_arrays = {}
    for ch_idx in range(loader.get_num_channels()):
        if ch_idx == detect_ch:
            channel_arrays[ch_idx] = detect_reduced
        else:
            logger.info(f"    Loading ch{ch_idx}...")
            ch_reduced, _, _ = load_channel_reduced(loader, ch_idx, block)
            channel_arrays[ch_idx] = ch_reduced

    logger.info(f"    Extracting intensity features ({len(channel_arrays)} channels)...")
    for det in detections:
        coords = det.pop('_full_mask_coords')
        full_mask = np.zeros(detect_reduced.shape[:2], dtype=bool)
        full_mask[coords[0], coords[1]] = True
        ch_feats = extract_channel_features(full_mask, channel_arrays)
        det['features'].update(ch_feats)

    # Step 5: SAM2 embeddings (optional, needs GPU)
    if args.sam2:
        from segmentation.utils.device import get_default_device
        device = get_default_device()
        if str(device) == 'cpu':
            logger.warning("No GPU available, skipping SAM2 embeddings")
        else:
            extract_sam2_embeddings(detections, args.czi_path, loader, device)

    # Summary
    if detections:
        n_feats = len(detections[0]['features'])
        areas = [d['features']['area_um2'] for d in detections]
        logger.info(f"  {n_feats} features per piece")
        logger.info(f"  Area: {min(areas):.0f}-{max(areas):.0f} µm², median {np.median(areas):.0f}")
        ys = [d['global_center'][1] for d in detections]
        logger.info(f"  Y range: {min(ys):.0f}-{max(ys):.0f} (CZI height: {full_h})")

    # Save
    output_path = output_dir / 'region_pieces.json'
    atomic_json_dump(detections, output_path)
    logger.info(f"Saved {len(detections)} pieces to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect bright regions from CZI and prepare for LMD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input
    parser.add_argument('--czi-path', required=True, help='CZI file path')
    parser.add_argument('--channel', type=int, default=None,
                        help='Channel index for detection')
    parser.add_argument('--channel-spec', default=None,
                        help='Channel spec (e.g., "detect=NfL" or "detect=750")')
    parser.add_argument('--output-dir', required=True, help='Output directory')

    # Thresholding
    parser.add_argument('--percentile', type=float, default=98,
                        help='Intensity percentile threshold (default: 98 = top 2%%)')
    parser.add_argument('--block-size', type=int, default=8,
                        help='Downsampling block size (default: 8)')

    # Morphology
    parser.add_argument('--close-radius', type=int, default=5,
                        help='Disk radius for morphological closing (default: 5)')
    parser.add_argument('--dilate-radius', type=int, default=3,
                        help='Disk radius for dilate/erode rounding (default: 3)')

    # Splitting
    parser.add_argument('--target-area-um2', type=float, default=250,
                        help='Target piece area in µm² (default: 250)')
    parser.add_argument('--min-area-um2', type=float, default=1000,
                        help='Minimum region area in µm² before splitting (default: 1000)')

    # Features
    parser.add_argument('--sam2', action='store_true',
                        help='Extract SAM2 256-dim embeddings (needs GPU)')
    parser.add_argument('--scene', type=int, default=0,
                        help='CZI scene index (default: 0)')

    args = parser.parse_args()
    detect_regions(args)


if __name__ == '__main__':
    main()
