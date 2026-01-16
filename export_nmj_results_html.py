#!/usr/bin/env python3
"""
Export NMJ inference results to HTML for visualization.
Shows classified NMJs with confidence scores and morphological features.

Uses the shared HTML export module for consistent styling and functionality.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import h5py
import hdf5plugin  # Required for reading compressed HDF5 masks

# Use shared segmentation utilities
from segmentation.io.html_export import (
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
    export_samples_to_html,  # Shared HTML export function
)
from segmentation.utils.logging import get_logger, setup_logging
from segmentation.io.czi_loader import get_loader, CZILoader

logger = get_logger(__name__)


def extract_crop_with_mask(loader, tile_x, tile_y, centroid, nmj_id,
                           seg_dir, tile_size=3000, crop_size=300):
    """Extract crop from CZI and overlay mask contour.

    Args:
        loader: CZILoader instance with channel data loaded
        tile_x: Tile X origin in mosaic coordinates
        tile_y: Tile Y origin in mosaic coordinates
        centroid: [x, y] centroid position within tile
        nmj_id: NMJ ID string for mask lookup
        seg_dir: Path to segmentation directory containing masks
        tile_size: Size of tiles
        crop_size: Size of output crop

    Returns:
        PIL Image with crop and mask contour overlay, or None if failed
    """
    # Read tile using shared loader
    tile_data = loader.get_tile(tile_x, tile_y, tile_size)

    if tile_data is None or tile_data.size == 0:
        return None

    # Extract crop centered on centroid (stored as [x, y])
    cx, cy = int(centroid[0]), int(centroid[1])
    half = crop_size // 2

    h, w = tile_data.shape

    # Validate centroid is within tile bounds (Issue #2)
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        logger.warning(f"NMJ centroid ({cx},{cy}) outside tile bounds ({w}x{h})")
        return None

    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)

    # Validate crop bounds before extracting
    if y2 <= y1 or x2 <= x1:
        return None

    crop = tile_data[y1:y2, x1:x2].copy()

    # Normalize
    crop = percentile_normalize(crop)

    # Convert to RGB
    crop_rgb = np.stack([crop] * 3, axis=-1)

    # Load mask and extract corresponding region
    mask_crop = None
    tile_dir = seg_dir / f"tile_{tile_x}_{tile_y}"
    mask_file = tile_dir / "nmj_masks.h5"

    if mask_file.exists():
        try:
            # Extract label number from nmj_id (e.g., "nmj_1" -> 1 or "225000_75000_nmj_1" -> 1)
            label_num = int(nmj_id.split('_')[-1])

            with h5py.File(mask_file, 'r') as f:
                # Masks stored as single labeled image - use slicing for efficiency
                if 'masks' in f:
                    # Read only the crop region (HDF5 handles this efficiently)
                    mask_region = f['masks'][y1:y2, x1:x2]
                    mask_crop = (mask_region == label_num).astype(np.uint8)

                    # Issue #5, #15: Don't load full HDF5 - just log warning if mask not in crop
                    if mask_crop.sum() == 0:
                        logger.debug(f"{nmj_id}: mask {label_num} not in crop region y={y1}:{y2}, x={x1}:{x2}")
        except Exception as e:
            logger.debug(f"Error loading mask for {nmj_id}: {e}")

    # Resize crop
    if crop_rgb.shape[0] > 0 and crop_rgb.shape[1] > 0:
        pil_img = Image.fromarray(crop_rgb)
        pil_img = pil_img.resize((crop_size, crop_size), Image.LANCZOS)

        # Resize and overlay mask if available
        if mask_crop is not None and mask_crop.size > 0:
            # Resize mask to match crop size
            mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((crop_size, crop_size), Image.NEAREST)
            mask_resized = np.array(mask_pil) > 127

            # Draw contour on image (convert to/from numpy for shared function)
            img_array = np.array(pil_img)
            img_with_contour = draw_mask_contour(img_array, mask_resized,
                                                  color=(144, 238, 144),
                                                  thickness=2, dotted=True)
            pil_img = Image.fromarray(img_with_contour)

        return pil_img

    return None


## Removed local generate_html_page and generate_index_html functions ##
# Now using shared export_samples_to_html from segmentation.io.html_export


def main():
    parser = argparse.ArgumentParser(description='Export NMJ results to HTML')
    parser.add_argument('--results-json', type=str, required=True,
                        help='Path to nmj_detections.json')
    parser.add_argument('--czi-path', type=str, required=True,
                        help='Path to CZI file')
    parser.add_argument('--segmentation-dir', type=str, default=None,
                        help='Directory with segmentation tiles (for mask overlay)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for HTML files')
    parser.add_argument('--channel', type=int, default=1,
                        help='Channel index')
    parser.add_argument('--per-page', type=int, default=300,
                        help='NMJs per page (default: 300 to match shared module)')
    parser.add_argument('--min-area', type=float, default=27,
                        help='Minimum area in um^2 to display (default: 27)')
    parser.add_argument('--min-confidence', type=float, default=0.75,
                        help='Minimum confidence to display (default: 0.75)')
    parser.add_argument('--load-to-ram', action='store_true', default=True,
                        help='Load entire channel into RAM for faster crop extraction (default: True)')
    parser.add_argument('--no-load-to-ram', dest='load_to_ram', action='store_false',
                        help='Disable RAM loading (use less memory but slower)')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")

    # Load results
    logger.info("Loading results...")
    with open(args.results_json) as f:
        results = json.load(f)

    # Handle both formats: list directly or dict with 'nmjs' key
    if isinstance(results, list):
        nmjs = results
        slide_name = nmjs[0].get('slide_name', 'unknown') if nmjs else 'unknown'
        # Normalize field names from run_segmentation.py format
        for nmj in nmjs:
            # Extract area_um2 from features if not at top level
            if 'area_um2' not in nmj and 'features' in nmj:
                feat = nmj['features']
                pixel_size = 0.1725  # Default for this slide
                if 'area' in feat:
                    nmj['area_um2'] = feat['area'] * (pixel_size ** 2)
                else:
                    nmj['area_um2'] = 0
            # Extract confidence from features if not at top level
            if 'confidence' not in nmj and 'features' in nmj:
                feat = nmj['features']
                nmj['confidence'] = feat.get('prob_nmj', feat.get('confidence', 1.0))
            # Map center to local_centroid if needed
            if 'local_centroid' not in nmj and 'center' in nmj:
                nmj['local_centroid'] = nmj['center']
            # Extract tile coordinates from tile_origin if not present
            if 'tile_x' not in nmj and 'tile_origin' in nmj:
                nmj['tile_x'] = nmj['tile_origin'][0]
                nmj['tile_y'] = nmj['tile_origin'][1]
            # Extract display fields from features
            if 'features' in nmj:
                feat = nmj['features']
                if 'elongation' not in nmj:
                    nmj['elongation'] = feat.get('elongation', 0.0)
                if 'skeleton_length' not in nmj:
                    nmj['skeleton_length'] = feat.get('skeleton_length', 0)
        # Build results dict with required fields
        results = {
            'slide_name': slide_name,
            'nmjs': nmjs,
            'total_nmjs': len(nmjs),
            'total_candidates': len(nmjs),  # Same as total when loading from list
            'pixel_size_um': 0.1725  # Default for this slide
        }
    else:
        nmjs = results['nmjs']
        slide_name = results.get('slide_name', 'unknown')

    logger.info(f"Total NMJs: {len(nmjs)}")

    if not nmjs:
        logger.warning("No NMJs to export!")
        return

    # Filter by area and confidence (but don't delete from results)
    original_count = len(nmjs)
    if args.min_area > 0 or args.min_confidence > 0:
        nmjs = [n for n in nmjs if n['area_um2'] >= args.min_area and n['confidence'] >= args.min_confidence]
        logger.info(f"After filtering (area >= {args.min_area} um^2, conf >= {args.min_confidence}): {len(nmjs)} ({original_count - len(nmjs)} filtered out)")

    # Sort by area in descending order (largest first)
    nmjs = sorted(nmjs, key=lambda x: x['area_um2'], reverse=True)
    logger.info(f"Sorted by area descending - largest: {nmjs[0]['area_um2']:.1f} um^2, smallest: {nmjs[-1]['area_um2']:.1f} um^2")

    # Setup output
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.results_json).parent / "html"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find segmentation directory for mask overlay
    if args.segmentation_dir:
        seg_dir = Path(args.segmentation_dir)
    else:
        # Auto-detect from results path
        seg_dir = Path(args.results_json).parent.parent / "tiles"
    logger.info(f"Segmentation directory: {seg_dir}")

    # Load CZI using shared loader (RAM-first for better performance)
    logger.info("Loading CZI...")
    loader = CZILoader(args.czi_path, load_to_ram=args.load_to_ram, channel=args.channel)
    logger.info(f"  Mosaic dimensions: {loader.width} x {loader.height}")
    logger.info(f"  RAM loading: {args.load_to_ram}")

    # Extract crops and convert to format expected by shared HTML export
    logger.info("Extracting crops with mask overlay...")
    html_samples = []
    failed_crops = 0

    for nmj in tqdm(nmjs, desc="Extracting"):
        try:
            crop = extract_crop_with_mask(
                loader,
                nmj['tile_x'],
                nmj['tile_y'],
                nmj['local_centroid'],
                nmj['id'],
                seg_dir
            )
            if crop:
                # image_to_base64 returns (base64_string, mime_type)
                img_b64, mime = image_to_base64(np.array(crop))

                # Build unique ID for this NMJ
                unique_id = f"{nmj['tile_x']}_{nmj['tile_y']}_{nmj['id']}"

                # Build sample dict in format expected by shared HTML export
                html_samples.append({
                    'uid': unique_id,
                    'image': img_b64,
                    'mime_type': mime,
                    'stats': {
                        'area_um2': nmj['area_um2'],
                        'confidence': nmj['confidence'],
                        'elongation': nmj.get('elongation', 0.0),
                    }
                })
            else:
                failed_crops += 1
        except Exception as e:
            logger.error(f"Extracting crop for NMJ {nmj['id']}: {e}")
            failed_crops += 1

    if failed_crops > 0:
        logger.warning(f"{failed_crops} crops failed to extract")

    logger.info(f"NMJs with valid crops: {len(html_samples)}")

    if not html_samples:
        logger.error("No valid crops to export!")
        return

    # Build filter subtitle
    filter_parts = []
    if args.min_area > 0:
        filter_parts.append(f"area >= {args.min_area} um^2")
    if args.min_confidence > 0:
        filter_parts.append(f"confidence >= {args.min_confidence:.0%}")
    subtitle = f"Filtered: {', '.join(filter_parts)}" if filter_parts else None

    # Use shared HTML export function with consistent naming
    logger.info("Generating HTML pages using shared module...")
    n_samples, n_pages = export_samples_to_html(
        samples=html_samples,
        output_dir=str(output_dir),
        cell_type='nmj',
        samples_per_page=args.per_page,
        title=f'NMJ Results - {slide_name}',
        subtitle=subtitle,
        page_prefix='nmj_page',  # Consistent naming with run_segmentation.py
        experiment_name=slide_name,
    )

    logger.info("HTML export complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Exported {n_samples} samples to {n_pages} pages")


if __name__ == '__main__':
    main()
