#!/usr/bin/env python3
"""
Regenerate NMJ HTML sorted by classifier score (descending).

Loads the detections JSON, sorts by score, loads tile data from CZI,
generates crops, and exports HTML pages.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from segmentation.io.czi_loader import get_loader
from segmentation.io.html_export import (
    export_samples_to_html,
    draw_mask_contour,
    image_to_base64,
    percentile_normalize,
)
from segmentation.utils.logging import get_logger, setup_logging

setup_logging(level="INFO", console=True)
logger = get_logger(__name__)


def deduplicate_by_mask_overlap(detections, tiles_dir, min_overlap_fraction=0.3):
    """Remove duplicate detections by checking actual mask overlap.

    For each pair of detections, checks if their masks overlap in global coordinates.
    When masks overlap by more than min_overlap_fraction, keeps the larger one.

    Args:
        detections: List of detection dicts with 'id', 'tile_origin', 'mask_label', 'features'
        tiles_dir: Path to tiles directory containing mask h5 files
        min_overlap_fraction: Minimum overlap (as fraction of smaller mask) to consider duplicates
    """
    from pathlib import Path

    if not detections:
        return []

    tiles_dir = Path(tiles_dir)

    # Load all masks and compute global bounding boxes
    det_info = []  # List of (det, global_bbox, global_mask_coords)

    # Cache loaded mask files
    mask_cache = {}

    for det in detections:
        tile_origin = tuple(det.get('tile_origin', [0, 0]))
        tile_x, tile_y = tile_origin
        tile_id = f"tile_{tile_x}_{tile_y}"
        mask_label = det.get('mask_label')

        if mask_label is None:
            # Try to extract from ID
            det_id = det['id']
            try:
                mask_label = int(det_id.split('_')[-1])
            except:
                det_info.append((det, None, None))
                continue

        # Load masks if not cached
        if tile_id not in mask_cache:
            masks_file = tiles_dir / tile_id / "nmj_masks.h5"
            if masks_file.exists():
                with h5py.File(masks_file, 'r') as f:
                    mask_cache[tile_id] = f['masks'][:]
            else:
                mask_cache[tile_id] = None

        masks = mask_cache.get(tile_id)
        if masks is None:
            det_info.append((det, None, None))
            continue

        # Get mask pixels in local coords
        local_ys, local_xs = np.where(masks == mask_label)
        if len(local_ys) == 0:
            det_info.append((det, None, None))
            continue

        # Convert to global coords
        global_xs = local_xs + tile_x
        global_ys = local_ys + tile_y

        # Compute bounding box (x_min, y_min, x_max, y_max)
        bbox = (global_xs.min(), global_ys.min(), global_xs.max(), global_ys.max())

        # Store global mask coords as set for fast overlap checking
        global_coords = set(zip(global_xs, global_ys))

        det_info.append((det, bbox, global_coords))

    # Sort by area descending (keep larger ones)
    det_info.sort(key=lambda x: x[0].get('features', {}).get('area', 0), reverse=True)

    # Greedy deduplication: keep detection if it doesn't significantly overlap with any kept detection
    kept = []
    kept_info = []  # (bbox, global_coords) for kept detections

    for det, bbox, coords in det_info:
        if bbox is None or coords is None:
            # Can't check overlap, keep it
            kept.append(det)
            continue

        # Check overlap with all kept detections
        is_duplicate = False
        for kept_bbox, kept_coords in kept_info:
            # Quick bbox overlap check first
            if (bbox[0] > kept_bbox[2] or bbox[2] < kept_bbox[0] or
                bbox[1] > kept_bbox[3] or bbox[3] < kept_bbox[1]):
                # Bboxes don't overlap
                continue

            # Check actual pixel overlap
            overlap = len(coords & kept_coords)
            smaller_size = min(len(coords), len(kept_coords))

            if smaller_size > 0 and overlap / smaller_size >= min_overlap_fraction:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)
            kept_info.append((bbox, coords))

    return kept


def create_sample_from_detection_with_tile(
    tile_rgb, masks, feat, pixel_size_um, slide_name, cell_type='nmj'
):
    """Create an HTML sample from a detection with tile data."""
    det_id = feat['id']

    # Use mask_label if available
    if 'mask_label' in feat:
        det_num = feat['mask_label']
    else:
        det_num = int(det_id.split('_')[-1])

    mask = masks == det_num
    if mask.sum() == 0:
        return None

    # Get centroid (local to tile)
    cy, cx = feat['center'][1], feat['center'][0]

    # Calculate crop size based on mask bounding box
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return None
    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_size = max(mask_h, mask_w)

    # Make crop 2x the mask size, with min 224, max 800
    crop_size = max(224, min(800, int(mask_size * 2)))
    half = crop_size // 2

    # Calculate crop bounds
    y1_ideal = int(cy) - half
    y2_ideal = int(cy) + half
    x1_ideal = int(cx) - half
    x2_ideal = int(cx) + half

    # Clamp to tile bounds
    y1 = max(0, y1_ideal)
    y2 = min(tile_rgb.shape[0], y2_ideal)
    x1 = max(0, x1_ideal)
    x2 = min(tile_rgb.shape[1], x2_ideal)

    if y2 <= y1 or x2 <= x1:
        return None

    crop = tile_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Pad to center if needed
    pad_top = max(0, y1 - y1_ideal)
    pad_bottom = max(0, y2_ideal - y2)
    pad_left = max(0, x1 - x1_ideal)
    pad_right = max(0, x2_ideal - x2)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode='constant', constant_values=0)
        crop_mask = np.pad(crop_mask, ((pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='constant', constant_values=False)

    # Normalize and draw contour
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5)
    crop_with_contour = draw_mask_contour(crop_norm, crop_mask, color=(0, 255, 0), thickness=2)

    pil_img = Image.fromarray(crop_with_contour)
    img_b64, mime = image_to_base64(pil_img, format='PNG')

    # Create unique ID
    tile_origin = feat.get('tile_origin', [0, 0])
    local_cx, local_cy = feat['center'][0], feat['center'][1]
    global_cx = tile_origin[0] + local_cx
    global_cy = tile_origin[1] + local_cy
    uid = f"{slide_name}_{cell_type}_{int(round(global_cx))}_{int(round(global_cy))}"

    # Get stats
    features = feat['features']
    area_um2 = features.get('area', 0) * (pixel_size_um ** 2)

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }

    # Add classifier score
    if 'score' in feat:
        stats['score'] = feat['score']

    return {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime,
        'stats': stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Regenerate NMJ HTML sorted by score')
    parser.add_argument('--output-dir', required=True, help='Output directory with detections')
    parser.add_argument('--czi-path', required=True, help='Path to CZI file')
    parser.add_argument('--samples-per-page', type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    czi_path = Path(args.czi_path)
    slide_name = czi_path.stem

    # Find the slide subdirectory
    slide_dir = output_dir / slide_name
    if not slide_dir.exists():
        # Try without subdirectory
        slide_dir = output_dir

    detections_file = slide_dir / 'nmj_detections.json'
    tiles_dir = slide_dir / 'tiles'
    html_dir = slide_dir / 'html'

    logger.info(f"Loading detections from {detections_file}")
    with open(detections_file) as f:
        all_detections = json.load(f)

    logger.info(f"Loaded {len(all_detections)} detections")

    # Deduplicate by checking actual mask overlap (keep larger mask)
    original_count = len(all_detections)
    logger.info("Checking mask overlaps for deduplication...")
    all_detections = deduplicate_by_mask_overlap(all_detections, tiles_dir, min_overlap_fraction=0.1)
    if len(all_detections) < original_count:
        logger.info(f"Deduplicated: {original_count} -> {len(all_detections)} ({original_count - len(all_detections)} duplicates removed)")

    # Sort by score descending
    all_detections.sort(key=lambda x: x.get('score', 0), reverse=True)
    logger.info("Sorted detections by classifier score (descending)")

    # Load CZI channels - load each channel explicitly
    logger.info(f"Loading CZI file: {czi_path}")

    # Load all 3 channels - use get_channel_data() to get the specific channel
    all_channel_data = {}
    pixel_size_um = None

    # Load all channels into a single loader
    loader = get_loader(czi_path, load_to_ram=True, channels=[0, 1, 2], quiet=False)
    pixel_size_um = loader.get_pixel_size()

    for ch in [0, 1, 2]:
        all_channel_data[ch] = loader.get_channel_data(ch)
        logger.info(f"  Channel {ch} shape: {all_channel_data[ch].shape}, dtype: {all_channel_data[ch].dtype}")

    # Group detections by tile
    detections_by_tile = {}
    for det in all_detections:
        tile_origin = tuple(det.get('tile_origin', [0, 0]))
        if tile_origin not in detections_by_tile:
            detections_by_tile[tile_origin] = []
        detections_by_tile[tile_origin].append(det)

    logger.info(f"Detections span {len(detections_by_tile)} tiles")

    # Process each tile and create samples
    all_samples = []
    tile_size = 3000  # Default tile size

    first_tile = True
    for tile_origin, tile_dets in tqdm(detections_by_tile.items(), desc="Processing tiles"):
        tile_x, tile_y = tile_origin

        # Load tile RGB - stack channels 0,1,2 as R,G,B
        tile_rgb = np.stack([
            all_channel_data[i][tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
            for i in range(3)
        ], axis=-1)

        # Debug first tile
        if first_tile:
            logger.info(f"First tile RGB shape: {tile_rgb.shape}, dtype: {tile_rgb.dtype}")
            for ch in range(3):
                ch_data = tile_rgb[:, :, ch]
                logger.info(f"  Channel {ch}: min={ch_data.min()}, max={ch_data.max()}, mean={ch_data.mean():.1f}")
            first_tile = False

        # Load masks
        tile_id = f"tile_{tile_x}_{tile_y}"
        masks_file = tiles_dir / tile_id / "nmj_masks.h5"

        if not masks_file.exists():
            logger.warning(f"Masks file not found: {masks_file}")
            continue

        with h5py.File(masks_file, 'r') as f:
            masks = f['masks'][:]

        # Create samples for each detection in this tile
        for det in tile_dets:
            sample = create_sample_from_detection_with_tile(
                tile_rgb, masks, det, pixel_size_um, slide_name, 'nmj'
            )
            if sample:
                all_samples.append(sample)

    logger.info(f"Created {len(all_samples)} samples")

    # Re-sort samples by score (in case grouping messed up order)
    all_samples.sort(key=lambda x: x['stats'].get('score', 0), reverse=True)

    # Export to HTML
    logger.info(f"Exporting to HTML ({len(all_samples)} samples)...")

    # Parse channel legend from filename
    channel_legend = None
    if 'nuc' in slide_name.lower() or '488' in slide_name:
        channel_legend = {'red': 'Nuclear (488)', 'green': 'BTX (647)', 'blue': 'NFL (750)'}

    export_samples_to_html(
        all_samples,
        html_dir,
        'nmj',
        samples_per_page=args.samples_per_page,
        title="NMJ Annotation Review",
        page_prefix='nmj_page',
        file_name=f"{slide_name}.czi",
        pixel_size_um=pixel_size_um,
        channel_legend=channel_legend,
    )

    logger.info(f"HTML exported to {html_dir}")


if __name__ == '__main__':
    main()
