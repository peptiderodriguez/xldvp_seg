#!/usr/bin/env python3
"""
Generate combined HTML viewer for all 16 slides with MK and HSPC detections.

Uses the HTMLPageGenerator from the package for consistent styling and functionality.
"""

import argparse
import json
import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Import package modules
from segmentation.io.czi_loader import get_loader
from segmentation.io.html_generator import HTMLPageGenerator
from segmentation.io.html_export import (
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)
from segmentation.utils.logging import get_logger, setup_logging

# Import hdf5plugin for LZ4 decompression
try:
    import hdf5plugin
except ImportError:
    pass

logger = get_logger(__name__)


def load_channel_to_ram(czi_path, channel, strip_height=5000):
    """Load a single channel from CZI into RAM."""
    loader = get_loader(czi_path, load_to_ram=True, channel=channel, strip_height=strip_height)
    channel_data = loader.channel_data
    return channel_data, (loader.x_start, loader.y_start, loader.width, loader.height)


def collect_all_detections(output_dir, slide_names, cell_type):
    """
    Collect all detections from multiple slide directories.

    Returns:
        List of detections with slide_name added to each detection
    """
    output_dir = Path(output_dir)
    all_detections = []

    logger.info(f"Collecting {cell_type} detections from {len(slide_names)} slides...")

    for slide_name in slide_names:
        slide_dir = output_dir / slide_name / cell_type / "tiles"

        if not slide_dir.exists():
            logger.warning(f"Skipping {slide_name} - {cell_type} directory not found")
            continue

        slide_count = 0
        for tile_dir in slide_dir.iterdir():
            if not tile_dir.is_dir():
                continue

            feat_file = tile_dir / "features.json"
            if not feat_file.exists():
                continue

            with open(feat_file) as f:
                tile_dets = json.load(f)

            for det in tile_dets:
                det['slide_name'] = slide_name
                det['tile_key'] = tile_dir.name
                slide_count += 1

            all_detections.extend(tile_dets)

        logger.info(f"  {slide_name}: {slide_count} {cell_type}s")

    logger.info(f"Total {cell_type} detections: {len(all_detections)}")
    return all_detections


def generate_samples(
    detections,
    output_dir,
    czi_dir,
    cell_type,
    channel=0,  # 0 for RGB, 1 for hematoxylin
    crop_size=300,
    display_size=250,
    contour_thickness=4,
    contour_color=(50, 255, 50),
    sort_by='area',
    sort_order='desc',
):
    """
    Generate HTML samples from detections across multiple slides.
    """
    output_dir = Path(output_dir)
    czi_dir = Path(czi_dir)

    # Sort detections
    reverse = (sort_order == 'desc')

    def get_sort_key(det):
        if sort_by in det.get('features', {}):
            return det['features'][sort_by]
        elif sort_by in det:
            return det[sort_by]
        return 0

    detections_sorted = sorted(detections, key=get_sort_key, reverse=reverse)
    logger.info(f"Sorted {len(detections_sorted)} detections by {sort_by} ({sort_order})")

    # Group by slide for efficient processing
    detections_by_slide = defaultdict(list)
    for det in detections_sorted:
        detections_by_slide[det['slide_name']].append(det)

    samples = []
    pixel_size = 0.1725  # Default pixel size in µm

    # Process each slide
    for slide_name, slide_dets in tqdm(detections_by_slide.items(), desc="Processing slides"):
        # Find CZI file
        czi_path = czi_dir / f"{slide_name}.czi"
        if not czi_path.exists():
            logger.warning(f"CZI not found: {czi_path}")
            continue

        # Load channel to RAM
        logger.info(f"Loading {slide_name} channel {channel} to RAM...")
        channel_data, (x_start, y_start, width, height) = load_channel_to_ram(
            str(czi_path), channel
        )

        # Process detections for this slide
        for det in tqdm(slide_dets, desc=f"  {slide_name}", leave=False):
            tile_key = det['tile_key']
            tile_dir = output_dir / slide_name / cell_type / "tiles" / tile_key

            # Load masks
            mask_file = tile_dir / f"{cell_type}_masks.h5"
            if not mask_file.exists():
                mask_file = tile_dir / "segmentation.h5"
            if not mask_file.exists():
                continue

            with h5py.File(mask_file, 'r') as f:
                masks = f['labels'][:]
                if masks.ndim == 3 and masks.shape[0] == 1:
                    masks = masks[0]

            # Get tile origin from window.csv
            window_file = tile_dir / "window.csv"
            if window_file.exists():
                import re
                with open(window_file) as f:
                    window_str = f.read()
                matches = re.findall(r'slice\((\d+),\s*(\d+)', window_str)
                if len(matches) >= 2:
                    tile_y = int(matches[0][0])
                    tile_x = int(matches[1][0])
                else:
                    tile_x, tile_y = 0, 0
            else:
                tile_x, tile_y = 0, 0

            # Get detection ID
            det_id = int(det['id'].split('_')[-1])

            # Get mask
            mask = (masks == det_id)
            if not mask.any():
                continue

            # Find mask centroid
            ys, xs = np.where(mask)
            cy, cx = np.mean(ys), np.mean(xs)

            # Global position
            global_cy = tile_y + cy
            global_cx = tile_x + cx

            # Extract crop centered on mask
            half = crop_size // 2
            y1 = max(0, int(global_cy - half))
            y2 = min(height, int(global_cy + half))
            x1 = max(0, int(global_cx - half))
            x2 = min(width, int(global_cx + half))

            if y2 <= y1 or x2 <= x1:
                continue

            crop = channel_data[y1:y2, x1:x2].copy()

            # Create crop mask
            crop_h, crop_w = y2 - y1, x2 - x1
            crop_mask = np.zeros((crop_h, crop_w), dtype=bool)

            # Map mask pixels to crop coords
            global_ys = ys + tile_y
            global_xs = xs + tile_x
            crop_ys = global_ys - y1
            crop_xs = global_xs - x1

            valid = (crop_ys >= 0) & (crop_ys < crop_h) & (crop_xs >= 0) & (crop_xs < crop_w)
            crop_mask[crop_ys[valid], crop_xs[valid]] = True

            # Normalize and draw contour
            crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5)
            crop_rgb = draw_mask_contour(
                crop_norm, crop_mask,
                color=contour_color,
                thickness=contour_thickness
            )

            # Resize
            crop_resized = cv2.resize(crop_rgb, (display_size, display_size))

            # Convert to base64
            image_b64, _ = image_to_base64(crop_resized)

            # Get features
            features = det['features']
            area = features.get('area', 0)
            area_um2 = area * (pixel_size ** 2)

            # Create sample dict for HTMLPageGenerator
            # The generator expects: uid, image_b64, stats dict
            sample = {
                'uid': det['uid'],
                'image_b64': image_b64,
                'stats': {
                    'slide': slide_name,
                    'area_um2': round(area_um2, 1),
                    'area_px': area,
                    'elongation': features.get('elongation', 0),
                }
            }

            samples.append(sample)

    logger.info(f"Generated {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate combined HTML for all 16 slides')
    parser.add_argument('--output-dir', required=True,
                       help='Path to segmentation output directory')
    parser.add_argument('--czi-dir', required=True,
                       help='Path to directory containing CZI files')
    parser.add_argument('--cell-type', choices=['mk', 'hspc'], required=True,
                       help='Cell type to process')
    parser.add_argument('--channel', type=int, default=0,
                       help='Channel to visualize (0=RGB, 1=hematoxylin, default: 0)')
    parser.add_argument('--sort-by', default='area',
                       help='Feature to sort by (default: area)')
    parser.add_argument('--sort-order', choices=['asc', 'desc'], default='desc',
                       help='Sort order (default: desc)')
    parser.add_argument('--crop-size', type=int, default=300,
                       help='Crop size in pixels (default: 300)')
    parser.add_argument('--display-size', type=int, default=250,
                       help='Display size in HTML (default: 250)')
    parser.add_argument('--samples-per-page', type=int, default=300,
                       help='Samples per page (default: 300)')
    parser.add_argument('--experiment-name', default='mi300a_10pct_all16',
                       help='Experiment name for localStorage (default: mi300a_10pct_all16)')

    args = parser.parse_args()

    setup_logging()

    # All 16 slide names
    slide_names = [
        '2025_11_18_FGC1', '2025_11_18_FGC2', '2025_11_18_FGC3', '2025_11_18_FGC4',
        '2025_11_18_FHU1', '2025_11_18_FHU2', '2025_11_18_FHU3', '2025_11_18_FHU4',
        '2025_11_18_MGC1', '2025_11_18_MGC2', '2025_11_18_MGC3', '2025_11_18_MGC4',
        '2025_11_18_MHU1', '2025_11_18_MHU2', '2025_11_18_MHU3', '2025_11_18_MHU4',
    ]

    # Collect detections from all slides
    detections = collect_all_detections(
        args.output_dir,
        slide_names,
        args.cell_type
    )

    if not detections:
        logger.error("No detections found!")
        return

    # Generate samples with image crops
    samples = generate_samples(
        detections,
        args.output_dir,
        args.czi_dir,
        args.cell_type,
        channel=args.channel,
        crop_size=args.crop_size,
        display_size=args.display_size,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
    )

    # Create HTML generator
    generator = HTMLPageGenerator(
        cell_type=args.cell_type,
        experiment_name=args.experiment_name,
        storage_strategy='experiment',
        samples_per_page=args.samples_per_page,
        title=f"{args.cell_type.upper()} - All 16 Slides (10% sampling)"
    )

    # Register custom formatters
    generator.register_formatter('slide', lambda v: f"<span style='color:#888'>{v}</span>")

    # Export to HTML
    html_dir = Path(args.output_dir) / "html_combined" / args.cell_type
    html_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating HTML pages in {html_dir}...")
    generator.export_to_html(samples, html_dir)

    logger.info(f"\n✓ Done! View at: {html_dir / 'index.html'}")


if __name__ == '__main__':
    main()
