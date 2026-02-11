#!/usr/bin/env python3
"""
Step 3: Generate combined HTML annotation pages from saved crops.

Reads crop_b64 + metadata from features.json files across all slides
(produced by step 2 parallel jobs) and builds combined MK/HSPC HTML
pages using the existing html_generator functions.

No CZI reloading required -- all data comes from saved features.json.

Usage:
    python generate_combined_html.py \
        --output-dir /viper/ptmp2/edrod/unified_2026-02-10_10pct_2gpu \
        --html-output-dir /viper/ptmp2/edrod/docs_2026-02-10_10pct_2gpu \
        --experiment-name 2026-02-10_10pct_2gpu \
        --samples-per-page 300
"""

import argparse
import base64
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from segmentation.io.html_export import draw_mask_contour
from segmentation.io.html_generator import (
    generate_mk_hspc_pages,
    create_mk_hspc_index,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('generate_combined_html')

PIXEL_SIZE_UM = 0.1725  # Default pixel size for area conversions


def _draw_contour_on_crop(crop_b64, mask_b64):
    """Decode crop + mask, draw green contour, re-encode to base64 JPEG."""
    # Decode crop
    crop_bytes = base64.b64decode(crop_b64)
    crop_img = np.array(Image.open(BytesIO(crop_bytes)).convert('RGB'))

    # Decode mask
    mask_bytes = base64.b64decode(mask_b64)
    mask_img = np.array(Image.open(BytesIO(mask_bytes)).convert('L'))
    mask_bool = mask_img > 127

    # Draw contour
    crop_with_contour = draw_mask_contour(crop_img, mask_bool, color=(0, 255, 0), thickness=2)

    # Re-encode to base64 JPEG
    pil_out = Image.fromarray(crop_with_contour)
    buf = BytesIO()
    pil_out.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_samples_from_disk(output_dir, cell_type):
    """
    Read crop_b64 + mask_b64 from features.json across all slides,
    draw mask contours on crops, and return samples for HTML generation.

    Args:
        output_dir: Path to segmentation output directory (contains slide subdirs)
        cell_type: 'mk' or 'hspc'

    Returns:
        List of sample dicts compatible with generate_mk_hspc_pages()
    """
    samples = []
    output_path = Path(output_dir)

    for slide_dir in sorted(output_path.iterdir()):
        if not slide_dir.is_dir():
            continue

        tiles_dir = slide_dir / cell_type / "tiles"
        if not tiles_dir.exists():
            continue

        slide_samples = 0
        for tile_dir in sorted(tiles_dir.iterdir()):
            if not tile_dir.is_dir():
                continue

            feat_file = tile_dir / "features.json"
            if not feat_file.exists():
                continue

            with open(feat_file) as f:
                feats = json.load(f)

            for feat in feats:
                if 'crop_b64' not in feat:
                    continue

                area_px = feat.get('area', feat['features'].get('area', 0))
                area_um2 = feat.get('area_um2', area_px * PIXEL_SIZE_UM ** 2)

                # Draw mask contour on the crop (crop_b64 is raw, mask_b64 has the mask)
                img_b64 = feat['crop_b64']
                if 'mask_b64' in feat:
                    try:
                        img_b64 = _draw_contour_on_crop(feat['crop_b64'], feat['mask_b64'])
                    except Exception as e:
                        logger.debug(f"Failed to draw contour for {feat['id']}: {e}")

                samples.append({
                    'tile_id': tile_dir.name,
                    'det_id': feat['id'],
                    'global_id': feat.get('global_id'),
                    'area_px': area_px,
                    'area_um2': area_um2,
                    'image': img_b64,
                    'features': feat['features'],
                    'solidity': feat['features'].get('solidity', 0),
                    'circularity': feat['features'].get('circularity', 0),
                    'global_x': int(feat['center'][0]),
                    'global_y': int(feat['center'][1]),
                    'slide': slide_dir.name,
                })
                slide_samples += 1

        if slide_samples > 0:
            logger.info(f"  {slide_dir.name}: {slide_samples} {cell_type}s loaded")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined HTML annotation pages from saved crops (Step 3)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Segmentation output directory (contains slide subdirs with features.json)')
    parser.add_argument('--html-output-dir', type=str, required=True,
                        help='Directory to write HTML files')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name for localStorage isolation and annotation filenames')
    parser.add_argument('--samples-per-page', type=int, default=300,
                        help='Number of cell samples per HTML page (default: 300)')
    parser.add_argument('--mk-min-area-um', type=float, default=200,
                        help='Minimum MK area in um^2 (default: 200)')
    parser.add_argument('--mk-max-area-um', type=float, default=2000,
                        help='Maximum MK area in um^2 (default: 2000)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    html_output_dir = Path(args.html_output_dir)
    html_output_dir.mkdir(parents=True, exist_ok=True)

    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        sys.exit(1)

    logger.info(f"{'='*70}")
    logger.info("STEP 3: Generate Combined HTML from Saved Crops")
    logger.info(f"{'='*70}")
    logger.info(f"  Output dir:       {output_dir}")
    logger.info(f"  HTML output dir:  {html_output_dir}")
    logger.info(f"  Experiment name:  {args.experiment_name}")
    logger.info(f"  Samples/page:     {args.samples_per_page}")
    logger.info(f"  MK area range:    {args.mk_min_area_um}-{args.mk_max_area_um} um^2")

    t0 = time.time()

    # Load MK samples
    logger.info(f"\nLoading MK samples...")
    all_mk_samples = load_samples_from_disk(output_dir, 'mk')
    logger.info(f"  Total MK samples loaded: {len(all_mk_samples)}")

    # Load HSPC samples
    logger.info(f"\nLoading HSPC samples...")
    all_hspc_samples = load_samples_from_disk(output_dir, 'hspc')
    logger.info(f"  Total HSPC samples loaded: {len(all_hspc_samples)}")

    if len(all_mk_samples) == 0 and len(all_hspc_samples) == 0:
        logger.error("No samples found! Check that step 2 completed and output-dir is correct.")
        logger.error(f"  Looked in: {output_dir}/*/{{mk,hspc}}/tiles/*/features.json")
        sys.exit(1)

    # Filter MK by size
    um_to_px_factor = PIXEL_SIZE_UM ** 2
    mk_min_px = int(args.mk_min_area_um / um_to_px_factor)
    mk_max_px = int(args.mk_max_area_um / um_to_px_factor)

    mk_before = len(all_mk_samples)
    all_mk_samples = [s for s in all_mk_samples if mk_min_px <= s.get('area_px', 0) <= mk_max_px]
    logger.info(f"\n  MK size filter: {mk_before} -> {len(all_mk_samples)} "
                f"({args.mk_min_area_um}-{args.mk_max_area_um} um^2)")

    # Sort: MK by area descending, HSPC by area descending
    all_mk_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)
    all_hspc_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)

    # Build slides summary
    slide_names = sorted(set(s['slide'] for s in all_mk_samples + all_hspc_samples))
    num_slides = len(slide_names)
    if num_slides > 0:
        short_names = [name.split('_')[-1] for name in slide_names]
        if len(short_names) > 6:
            preview = ', '.join(short_names[:4]) + ', ...'
        else:
            preview = ', '.join(short_names)
        slides_summary = f"{num_slides} slides ({preview})"
    else:
        slides_summary = None

    logger.info(f"  Slides: {slides_summary}")

    # Generate pages
    logger.info(f"\nGenerating HTML pages...")
    generate_mk_hspc_pages(
        all_mk_samples, "mk", html_output_dir, args.samples_per_page,
        slides_summary=slides_summary, experiment_name=args.experiment_name
    )
    generate_mk_hspc_pages(
        all_hspc_samples, "hspc", html_output_dir, args.samples_per_page,
        slides_summary=slides_summary, experiment_name=args.experiment_name
    )

    # Create index
    mk_pages = (len(all_mk_samples) + args.samples_per_page - 1) // args.samples_per_page if all_mk_samples else 0
    hspc_pages = (len(all_hspc_samples) + args.samples_per_page - 1) // args.samples_per_page if all_hspc_samples else 0

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    create_mk_hspc_index(
        html_output_dir,
        len(all_mk_samples), len(all_hspc_samples),
        mk_pages, hspc_pages,
        slides_summary=slides_summary,
        timestamp=timestamp,
        experiment_name=args.experiment_name
    )

    elapsed = time.time() - t0
    logger.info(f"\n{'='*70}")
    logger.info(f"HTML GENERATION COMPLETE ({elapsed:.1f}s)")
    logger.info(f"{'='*70}")
    logger.info(f"  MKs:   {len(all_mk_samples)} ({mk_pages} pages)")
    logger.info(f"  HSPCs: {len(all_hspc_samples)} ({hspc_pages} pages)")
    logger.info(f"  Output: {html_output_dir}")
    if args.experiment_name:
        logger.info(f"  localStorage key: mk_{args.experiment_name}_annotations")
        logger.info(f"  Download file:    annotations_{args.experiment_name}.json")


if __name__ == "__main__":
    main()
