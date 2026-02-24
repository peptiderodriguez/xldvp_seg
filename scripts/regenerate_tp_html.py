#!/usr/bin/env python3
"""
DEPRECATED: Use scripts/regenerate_html.py instead (consolidated all-cell-type version).

    python scripts/regenerate_html.py --output-dir DIR --czi-path CZI --cell-type tissue_pattern

---
Regenerate HTML viewer from existing tissue_pattern (or any cell type) detections.

Reads per-tile masks (HDF5) and features (JSON) from the tiles/ directory,
composes multi-channel RGB crops from CZI data loaded to RAM, and generates
the full HTML annotation interface using export_samples_to_html().

This avoids re-running detection — only crop rendering + HTML export.

Usage:
    python scripts/regenerate_tp_html.py \
        --output-dir /path/to/run_output \
        --czi-path /path/to/slide.czi \
        --display-channels 1,2,0 \
        --max-samples 15000
"""

import warnings
warnings.warn(
    "regenerate_tp_html.py is deprecated. Use scripts/regenerate_html.py instead:\n"
    "  python scripts/regenerate_html.py --output-dir DIR --czi-path CZI --cell-type tissue_pattern",
    DeprecationWarning, stacklevel=1,
)

import argparse
import json
import gc
import sys
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict

# HDF5 LZ4 support
try:
    import hdf5plugin
except ImportError:
    pass

from segmentation.io.czi_loader import get_loader
from segmentation.io.html_export import (
    export_samples_to_html,
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)
from segmentation.utils.logging import get_logger, setup_logging
from PIL import Image

logger = get_logger(__name__)


def compose_tile_rgb(channel_arrays, tile_x, tile_y, tile_size, display_channels,
                     x_start, y_start, mosaic_h, mosaic_w):
    """Extract a tile region and compose RGB from display channels."""
    # Convert to array coordinates (subtract mosaic origin)
    ay = tile_y - y_start
    ax = tile_x - x_start

    # Clamp to array bounds
    ay_end = min(ay + tile_size, mosaic_h)
    ax_end = min(ax + tile_size, mosaic_w)
    ay = max(0, ay)
    ax = max(0, ax)

    if ay_end <= ay or ax_end <= ax:
        return None

    h = ay_end - ay
    w = ax_end - ax
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i, ch_idx in enumerate(display_channels[:3]):
        if ch_idx < len(channel_arrays) and channel_arrays[ch_idx] is not None:
            ch_data = channel_arrays[ch_idx][ay:ay_end, ax:ax_end]
            # Percentile normalize to uint8
            valid = ch_data > 0
            if valid.any():
                p1 = np.percentile(ch_data[valid], 1)
                p99 = np.percentile(ch_data[valid], 99.5)
                if p99 > p1:
                    norm = np.clip((ch_data.astype(np.float32) - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
                    norm[~valid] = 0
                    rgb[:, :, i] = norm
    return rgb


def create_sample(tile_rgb, masks, feat, pixel_size_um, slide_name, cell_type):
    """Create an HTML sample dict from a detection. Mask-bounded crop."""
    mask_label = feat.get('tile_mask_label', feat.get('mask_label', 0))
    if mask_label == 0:
        # Try parsing from id
        try:
            mask_label = int(feat['id'].split('_')[-1])
        except (KeyError, ValueError, IndexError):
            return None

    mask = masks == mask_label
    if mask.sum() == 0:
        return None

    # Get centroid from features (local tile coords)
    center = feat.get('center', None)
    if center is None:
        # Fall back to computing from mask
        ys, xs = np.where(mask)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
    else:
        cx, cy = center[0], center[1]

    # Mask bounding box for dynamic crop size
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_size = max(mask_h, mask_w)

    # Crop = 2x mask size, clamped to [224, 800]
    crop_size = max(224, min(800, int(mask_size * 2)))
    half = crop_size // 2

    # Crop bounds
    y1_ideal = int(cy) - half
    y2_ideal = int(cy) + half
    x1_ideal = int(cx) - half
    x2_ideal = int(cx) + half

    y1 = max(0, y1_ideal)
    y2 = min(tile_rgb.shape[0], y2_ideal)
    x1 = max(0, x1_ideal)
    x2 = min(tile_rgb.shape[1], x2_ideal)

    if y2 <= y1 or x2 <= x1:
        return None

    crop = tile_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Pad if clamped at edges
    pad_top = max(0, y1 - y1_ideal)
    pad_bottom = max(0, y2_ideal - y2)
    pad_left = max(0, x1 - x1_ideal)
    pad_right = max(0, x2_ideal - x2)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                       mode='constant', constant_values=0)
        crop_mask = np.pad(crop_mask, ((pad_top, pad_bottom), (pad_left, pad_right)),
                           mode='constant', constant_values=False)

    # Draw contour on normalized crop
    crop_with_contour = draw_mask_contour(crop, crop_mask, color=(0, 255, 0), thickness=2)

    pil_img = Image.fromarray(crop_with_contour)
    img_b64, mime = image_to_base64(pil_img, format='PNG')

    # Build UID from global center
    tile_origin = feat.get('tile_origin', [0, 0])
    global_cx = tile_origin[0] + cx
    global_cy = tile_origin[1] + cy
    uid = feat.get('uid', f"{slide_name}_{cell_type}_{int(round(global_cx))}_{int(round(global_cy))}")

    features = feat.get('features', {})
    area_um2 = features.get('area', 0) * (pixel_size_um ** 2)

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'rf_prediction' in feat and feat['rf_prediction'] is not None:
        stats['rf_prediction'] = feat['rf_prediction']

    return {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime,
        'stats': stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate HTML from existing detections (no re-detection)')
    parser.add_argument('--output-dir', required=True,
                        help='Path to existing run output directory (contains tiles/ and *_detections.json)')
    parser.add_argument('--czi-path', required=True,
                        help='Path to CZI file')
    parser.add_argument('--display-channels', default='1,2,0',
                        help='Comma-separated channel indices for R,G,B display (default: 1,2,0)')
    parser.add_argument('--channels', default=None,
                        help='Comma-separated channel indices to load (default: auto from display-channels)')
    parser.add_argument('--cell-type', default='tissue_pattern',
                        help='Cell type (default: tissue_pattern)')
    parser.add_argument('--max-samples', type=int, default=15000,
                        help='Max HTML samples to generate (default: 15000)')
    parser.add_argument('--samples-per-page', type=int, default=300,
                        help='Samples per HTML page (default: 300)')
    parser.add_argument('--scene', type=int, default=0,
                        help='CZI scene index (default: 0)')
    parser.add_argument('--tile-size', type=int, default=4000,
                        help='Tile size used during detection (default: 4000)')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in µm (auto-detected from CZI if not specified)')

    args = parser.parse_args()
    setup_logging(level="INFO")

    output_dir = Path(args.output_dir)
    tiles_dir = output_dir / "tiles"
    html_dir = output_dir / "html"
    czi_path = Path(args.czi_path)
    slide_name = czi_path.stem
    cell_type = args.cell_type
    display_channels = [int(x) for x in args.display_channels.split(',')]

    # Determine which channels to load
    if args.channels:
        channels_to_load = [int(x) for x in args.channels.split(',')]
    else:
        channels_to_load = sorted(set(display_channels))

    # Load detections JSON to get the deduped detection set
    det_files = list(output_dir.glob(f"{cell_type}_detections.json"))
    if not det_files:
        det_files = list(output_dir.glob("*_detections.json"))
    if not det_files:
        logger.error(f"No detections JSON found in {output_dir}")
        sys.exit(1)

    det_file = det_files[0]
    logger.info(f"Loading detections from {det_file}...")
    with open(det_file) as f:
        all_detections = json.load(f)
    logger.info(f"Loaded {len(all_detections):,} detections")

    # Build set of deduped UIDs
    deduped_uids = {d.get('uid', '') for d in all_detections}

    # Sample if needed
    if args.max_samples > 0 and len(all_detections) > args.max_samples:
        indices = np.random.default_rng(42).choice(
            len(all_detections), args.max_samples, replace=False)
        sampled = [all_detections[i] for i in indices]
        logger.info(f"Sampled {len(sampled):,} / {len(all_detections):,} detections for HTML")
    else:
        sampled = all_detections

    # Group sampled detections by tile
    tile_groups = defaultdict(list)
    for det in sampled:
        to = det.get('tile_origin')
        if to is None:
            continue
        tile_key = f"tile_{to[0]}_{to[1]}"
        tile_groups[tile_key].append(det)

    logger.info(f"{len(sampled):,} detections across {len(tile_groups)} tiles")

    # Load CZI channels to RAM
    logger.info(f"Loading CZI channels {channels_to_load} to RAM...")
    channel_arrays = {}
    loader = get_loader(czi_path, load_to_ram=True, channel=channels_to_load[0],
                        scene=args.scene)
    x_start = loader.x_start
    y_start = loader.y_start
    mosaic_w = loader.width
    mosaic_h = loader.height
    pixel_size_um = args.pixel_size or loader.pixel_size_um

    channel_arrays[channels_to_load[0]] = loader.channel_data
    logger.info(f"  Channel {channels_to_load[0]} loaded: {loader.channel_data.nbytes / 1e9:.1f} GB")

    for ch in channels_to_load[1:]:
        ch_loader = get_loader(czi_path, load_to_ram=True, channel=ch, scene=args.scene)
        channel_arrays[ch] = ch_loader.channel_data
        logger.info(f"  Channel {ch} loaded: {ch_loader.channel_data.nbytes / 1e9:.1f} GB")

    # Build ordered array for compose_tile_rgb
    max_ch = max(channels_to_load) + 1
    ch_array_list = [None] * max_ch
    for ch_idx, arr in channel_arrays.items():
        ch_array_list[ch_idx] = arr

    # Channel legend from CZI metadata
    channel_legend = None
    try:
        meta = loader.czi_reader
        # Try to get channel names from metadata
        import xml.etree.ElementTree as ET
        meta_xml = meta.meta
        root = ET.fromstring(meta_xml)
        seen = set()
        ch_info = []
        for ch_el in root.iter('Channel'):
            name = ch_el.get('Name', '')
            if name and name not in seen:
                seen.add(name)
                em = ch_el.find('.//EmissionWavelength')
                em_nm = float(em.text) if em is not None else None
                ch_info.append({'name': name, 'em': em_nm})
        if len(ch_info) >= max(display_channels) + 1:
            colors = ['red', 'green', 'blue']
            channel_legend = {}
            for i, ch_idx in enumerate(display_channels[:3]):
                ci = ch_info[ch_idx]
                label = ci['name']
                if ci['em']:
                    label += f" ({ci['em']:.0f}nm)"
                channel_legend[colors[i]] = label
            logger.info(f"Channel legend: {channel_legend}")
    except Exception as e:
        logger.warning(f"Could not extract channel legend: {e}")

    # Process tiles and generate samples
    logger.info(f"Generating HTML crops for {len(sampled):,} detections...")
    all_samples = []
    tiles_processed = 0

    from tqdm import tqdm
    for tile_key, tile_dets in tqdm(tile_groups.items(), desc="Tiles"):
        tile_dir = tiles_dir / tile_key
        mask_file = tile_dir / f"{cell_type}_masks.h5"

        if not mask_file.exists():
            logger.warning(f"No mask file for {tile_key}, skipping {len(tile_dets)} detections")
            continue

        # Load masks
        with h5py.File(mask_file, 'r') as hf:
            if 'masks' in hf:
                masks = hf['masks'][:]
            elif 'labels' in hf:
                masks = hf['labels'][:]
            else:
                logger.warning(f"No masks dataset in {mask_file}")
                continue
            if masks.ndim == 3 and masks.shape[0] == 1:
                masks = masks[0]

        # Parse tile origin from first detection
        tile_origin = tile_dets[0].get('tile_origin', [0, 0])
        tile_x, tile_y = tile_origin[0], tile_origin[1]

        # Compose tile RGB from display channels
        tile_rgb = compose_tile_rgb(
            ch_array_list, tile_x, tile_y, args.tile_size,
            display_channels, x_start, y_start, mosaic_h, mosaic_w)

        if tile_rgb is None:
            continue

        # Generate crop for each detection
        for det in tile_dets:
            sample = create_sample(
                tile_rgb, masks, det, pixel_size_um, slide_name, cell_type)
            if sample:
                all_samples.append(sample)

        tiles_processed += 1
        del masks, tile_rgb
        if tiles_processed % 100 == 0:
            gc.collect()

    logger.info(f"Generated {len(all_samples):,} HTML samples from {tiles_processed} tiles")

    # Sort by area (ascending) — same as annotation run default
    all_samples.sort(key=lambda x: x['stats'].get('area_um2', 0))

    # Export HTML
    experiment_name = f"{slide_name}_regen"
    export_samples_to_html(
        all_samples,
        html_dir,
        cell_type,
        samples_per_page=args.samples_per_page,
        title=f"{cell_type.upper()} Annotation Review (regenerated)",
        page_prefix=f'{cell_type}_page',
        experiment_name=experiment_name,
        file_name=f"{slide_name}.czi",
        pixel_size_um=pixel_size_um,
        tiles_processed=tiles_processed,
        tiles_total=len(tile_groups),
        channel_legend=channel_legend,
    )

    logger.info(f"HTML exported to {html_dir}")
    logger.info(f"  {len(all_samples):,} samples, {(len(all_samples) + args.samples_per_page - 1) // args.samples_per_page} pages")


if __name__ == '__main__':
    main()
