#!/usr/bin/env python3
"""
Regenerate vessel HTML from saved crops without re-reading CZI.

Usage:
    python regenerate_html.py --input-dir /path/to/vessel_output [--thickness 10] [--inner-dotted]
"""

import sys

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from segmentation.io.html_export import (
    export_samples_to_html,
    draw_mask_contour,
    image_to_base64,
)


def regenerate_from_crops(
    detections_json: Path,
    crops_dir: Path,
    output_dir: Path,
    thickness: int = 10,
    inner_dotted: bool = False,
    outer_dotted: bool = False,
    pixel_size_um: float = None,
    channel_legend: dict = None,
):
    """Regenerate HTML from saved crops with configurable contour style.

    Args:
        channel_legend: Optional dict mapping colors to channel names,
            e.g., {"red": "SMA (647)", "green": "CD31 (555)", "blue": "Nuclear (488)"}.
            If None, auto-detected from detection JSON metadata or slide name.
    """

    # Load detections
    with open(detections_json) as f:
        data = json.load(f)

    vessels = data.get('detections', data.get('vessels', []))
    slide_name = data.get('slide_name', 'unknown')

    # Read pixel_size from detection metadata; CLI arg takes priority
    if pixel_size_um is None:
        pixel_size_um = data.get('pixel_size_um')
        if pixel_size_um is None:
            # Try nested metadata
            pixel_size_um = data.get('metadata', {}).get('pixel_size_um')
        if pixel_size_um is None:
            print("WARNING: pixel_size_um not found in detection data and not provided via --pixel-size. "
                  "Defaulting to 0.1725 um/px.")
            pixel_size_um = 0.1725

    # Auto-detect channel legend from metadata if not provided via CLI
    if channel_legend is None:
        channel_legend = data.get('channel_legend')
    if channel_legend is None:
        # Try to parse from slide_name using the standard parser
        try:
            from run_segmentation import parse_channel_legend_from_filename
            channel_legend = parse_channel_legend_from_filename(slide_name)
        except (ImportError, Exception):
            pass

    print(f"Loaded {len(vessels)} vessels from {detections_json}")
    print(f"Crops directory: {crops_dir}")
    print(f"Contour style: thickness={thickness}, inner_dotted={inner_dotted}")

    samples = []
    missing_crops = 0

    for vessel in tqdm(vessels, desc="Processing crops"):
        uid = vessel['uid']
        crop_path = crops_dir / f"{uid}.jpg"

        if not crop_path.exists():
            missing_crops += 1
            continue

        # Load raw crop
        crop_bgr = cv2.imread(str(crop_path))
        if crop_bgr is None:
            missing_crops += 1
            continue
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # Get shifted contours (may not exist in older JSONs)
        outer_shifted = vessel.get('outer_contour_shifted')
        inner_shifted = vessel.get('inner_contour_shifted')

        # If no shifted contours, just use the crop as-is (no contour overlay)
        if outer_shifted is None:
            img_b64, mime_type = image_to_base64(crop_rgb, format='JPEG', quality=85)
            sample = {
                'uid': uid,
                'image': img_b64,
                'mime_type': mime_type,
                'stats': vessel.get('stats', {}),
                'features': vessel.get('features', {}),
            }
            samples.append(sample)
            continue

        outer_shifted = np.array(outer_shifted, dtype=np.int32)

        # Create mask for outer contour
        h, w = crop_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [outer_shifted], 0, 255, -1)

        # Draw outer contour
        crop_with_contour = draw_mask_contour(
            crop_rgb,
            mask > 0,
            color=(255, 255, 255),
            thickness=thickness,
            dotted=outer_dotted
        )

        # Draw inner contour if exists
        if inner_shifted is not None:
            inner_shifted = np.array(inner_shifted, dtype=np.int32)
            inner_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inner_mask, [inner_shifted], 0, 255, -1)
            crop_with_contour = draw_mask_contour(
                crop_with_contour,
                inner_mask > 0,
                color=(255, 255, 255),
                thickness=thickness,
                dotted=inner_dotted
            )

        # Convert to base64
        img_b64, mime_type = image_to_base64(crop_with_contour, format='JPEG', quality=85)

        # Build sample dict (copy stats from original)
        sample = {
            'uid': uid,
            'image': img_b64,
            'mime_type': mime_type,
            'stats': vessel.get('stats', {}),
            'features': vessel.get('features', {}),
            'outer_contour': vessel.get('outer_contour'),
            'inner_contour': vessel.get('inner_contour'),
        }
        samples.append(sample)

    if missing_crops > 0:
        print(f"Warning: {missing_crops} crops missing or invalid")

    print(f"Generating HTML for {len(samples)} vessels...")

    # Export HTML
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    export_samples_to_html(
        samples=samples,
        output_dir=html_dir,
        cell_type='vessel',
        samples_per_page=200,
        experiment_name=f"{slide_name} - Lumen-First Detection",
        channel_legend=channel_legend,
    )

    print(f"HTML exported to: {html_dir}")


def main():
    parser = argparse.ArgumentParser(description='Regenerate vessel HTML from saved crops')
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory with vessel_detections.json and crops/')
    parser.add_argument('--thickness', type=int, default=10,
                        help='Contour line thickness (default: 10)')
    parser.add_argument('--inner-dotted', action='store_true',
                        help='Use dotted line for inner contour')
    parser.add_argument('--outer-dotted', action='store_true',
                        help='Use dotted line for outer contour')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in micrometers (overrides value from detection JSON)')
    parser.add_argument('--channel-legend', type=str, default=None,
                        help='Channel legend as JSON string, e.g., '
                             '\'{"red": "SMA (647)", "green": "CD31 (555)", "blue": "Nuclear (488)"}\'. '
                             'If not provided, auto-detected from detection JSON metadata or slide name.')

    args = parser.parse_args()

    # Parse channel legend from CLI arg if provided
    cli_channel_legend = None
    if args.channel_legend:
        try:
            cli_channel_legend = json.loads(args.channel_legend)
        except json.JSONDecodeError:
            print(f"Error: --channel-legend must be valid JSON. Got: {args.channel_legend}")
            sys.exit(1)

    detections_json = args.input_dir / 'vessel_detections.json'
    crops_dir = args.input_dir / 'crops'

    if not detections_json.exists():
        print(f"Error: {detections_json} not found")
        sys.exit(1)

    if not crops_dir.exists():
        print(f"Error: {crops_dir} not found")
        print("Run the main detection script first to save crops.")
        sys.exit(1)

    regenerate_from_crops(
        detections_json=detections_json,
        crops_dir=crops_dir,
        output_dir=args.input_dir,
        thickness=args.thickness,
        inner_dotted=args.inner_dotted,
        outer_dotted=args.outer_dotted,
        pixel_size_um=args.pixel_size,
        channel_legend=cli_channel_legend,
    )


if __name__ == '__main__':
    main()
