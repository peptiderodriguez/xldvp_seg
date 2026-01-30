#!/usr/bin/env python3
"""
Generate combined HTML viewer from existing features.json files (with embedded crops).

The crops are already saved as base64 in features.json with cleaned masks overlaid.
"""

import argparse
import json
import base64
import io
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

from segmentation.io.html_generator import HTMLPageGenerator
from segmentation.io.html_export import image_to_base64, draw_mask_contour
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def decode_base64_image(b64_string):
    """Decode base64 string to numpy array."""
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def draw_contour_on_crop(crop_b64, mask_b64, color=(50, 255, 50), thickness=3):
    """Draw mask contour on crop image and return new base64."""
    # Decode images
    crop = decode_base64_image(crop_b64)
    mask = decode_base64_image(mask_b64)

    # Convert mask to boolean
    if mask.ndim == 3:
        mask = mask[:, :, 0]  # Take first channel if RGB
    mask_bool = mask > 127  # Threshold to boolean

    # Draw contour on crop
    crop_with_contour = draw_mask_contour(crop, mask_bool, color=color, thickness=thickness)

    # Encode back to base64
    b64, _ = image_to_base64(crop_with_contour)
    return b64


def collect_samples_from_features(output_dir, slide_names, cell_type, sort_by='area', sort_order='desc'):
    """
    Collect samples from features.json files across multiple slides.

    Each features.json already contains:
    - crop_b64: Base64-encoded image crop with mask overlay
    - mask_b64: Base64-encoded mask visualization
    - features: All computed features
    - uid, center, etc.
    """
    output_dir = Path(output_dir)
    samples = []

    logger.info(f"Collecting {cell_type} samples from {len(slide_names)} slides...")

    for slide_name in tqdm(slide_names, desc="Loading slides"):
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
                detections = json.load(f)

            for det in detections:
                # Skip if no crop_b64 (shouldn't happen but check anyway)
                if not det.get('crop_b64'):
                    continue

                features = det['features']
                area = features.get('area', 0)

                # Draw contour on crop if mask_b64 is available
                if det.get('mask_b64'):
                    try:
                        image_with_contour = draw_contour_on_crop(
                            det['crop_b64'],
                            det['mask_b64'],
                            color=(50, 255, 50),
                            thickness=3
                        )
                    except Exception as e:
                        logger.warning(f"Failed to draw contour for {det['uid']}: {e}")
                        image_with_contour = det['crop_b64']
                else:
                    image_with_contour = det['crop_b64']

                # Generate UID if not available (fallback to ID + slide)
                uid = det.get('uid') or f"{slide_name}_{det.get('id', 'unknown')}"

                # Create sample for HTMLPageGenerator
                sample = {
                    'uid': uid,
                    'image': image_with_contour,  # Crop with contour drawn
                    'stats': {
                        'area_um2': round(area * (0.1725 ** 2), 1),
                        'area_px': area,
                    }
                }

                samples.append(sample)
                slide_count += 1

        logger.info(f"  {slide_name}: {slide_count} samples")

    logger.info(f"Total samples collected: {len(samples)}")

    # Sort samples
    reverse = (sort_order == 'desc')

    def get_sort_key(sample):
        if sort_by == 'area':
            return sample['stats'].get('area_px', 0)
        elif sort_by == 'area_um2':
            return sample['stats'].get('area_um2', 0)
        return sample['stats'].get(sort_by, 0)

    samples_sorted = sorted(samples, key=get_sort_key, reverse=reverse)
    logger.info(f"Sorted by {sort_by} ({sort_order})")

    return samples_sorted


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined HTML from features.json files'
    )
    parser.add_argument('--output-dir', required=True,
                       help='Path to segmentation output directory')
    parser.add_argument('--cell-type', choices=['mk', 'hspc'], required=True,
                       help='Cell type to process')
    parser.add_argument('--sort-by', default='area',
                       help='Feature to sort by (default: area)')
    parser.add_argument('--sort-order', choices=['asc', 'desc'], default='desc',
                       help='Sort order (default: desc)')
    parser.add_argument('--samples-per-page', type=int, default=300,
                       help='Samples per page (default: 300)')
    parser.add_argument('--experiment-name', default='mi300a_10pct_all16',
                       help='Experiment name for localStorage')

    args = parser.parse_args()

    setup_logging()

    # All 16 slide names
    slide_names = [
        '2025_11_18_FGC1', '2025_11_18_FGC2', '2025_11_18_FGC3', '2025_11_18_FGC4',
        '2025_11_18_FHU1', '2025_11_18_FHU2', '2025_11_18_FHU3', '2025_11_18_FHU4',
        '2025_11_18_MGC1', '2025_11_18_MGC2', '2025_11_18_MGC3', '2025_11_18_MGC4',
        '2025_11_18_MHU1', '2025_11_18_MHU2', '2025_11_18_MHU3', '2025_11_18_MHU4',
    ]

    # Collect samples from features.json files
    samples = collect_samples_from_features(
        args.output_dir,
        slide_names,
        args.cell_type,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
    )

    if not samples:
        logger.error("No samples found!")
        return

    # Create HTML generator with package template
    generator = HTMLPageGenerator(
        cell_type=args.cell_type,
        experiment_name=args.experiment_name,
        storage_strategy='experiment',
        samples_per_page=args.samples_per_page,
        title=f"{args.cell_type.upper()} - All 16 Slides (10% sampling, cleaned masks)"
    )

    # Export to HTML
    html_dir = Path(args.output_dir) / "html_combined" / args.cell_type
    html_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating HTML pages in {html_dir}...")
    generator.export_to_html(samples, html_dir)

    logger.info(f"\n✓ Done! View at: {html_dir / 'index.html'}")


if __name__ == '__main__':
    main()
