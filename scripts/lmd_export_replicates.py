#!/usr/bin/env python3
"""
Generate LMD XML from pre-assigned replicate JSON + pre-extracted contours.

This script is for the replicate-based LMD workflow where well assignments
are already made (e.g. via select_mks_for_lmd.py). It takes:
  - Replicate JSON (lmd_replicates_full.json) with per-cell well assignments
  - Pre-extracted contour JSON (mk_contours_overlay.json) with contour_yx
  - Crosses from napari_place_crosses.py

Coordinate conventions:
  - Contours are in native CZI pixel space [y, x] -> converted to [x, y] um
  - Crosses are in flipped display space (tissue-down) from napari
  - XML output: X flipped (tissue-down) + Y flipped (LMD convention)

Usage:
    python lmd_export_replicates.py \
        --sampling-results lmd_replicates_full.json \
        --contours-json mk_contours_overlay.json \
        --crosses-dir ./crosses \
        --output-dir ./xml
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    from segmentation.utils.logging import get_logger
    log = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

try:
    from segmentation.utils.json_utils import atomic_json_dump
except ImportError:
    def atomic_json_dump(data, path, **kwargs):
        with open(path, 'w') as f:
            json.dump(data, f)


DEFAULT_PIXEL_SIZE_UM = 0.1725


def find_slide_crosses(crosses_dir, slide_name):
    """Find most recent crosses JSON for a slide. Returns Path or None."""
    if crosses_dir is None:
        return None
    crosses_dir = Path(crosses_dir)
    if not crosses_dir.exists():
        return None

    candidates = sorted(
        crosses_dir.glob(f"{slide_name}_crosses*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def get_pixel_size(crosses_data, slide_data):
    """Get pixel size from crosses, replicate metadata, or default."""
    if crosses_data and crosses_data.get('pixel_size_um'):
        return crosses_data['pixel_size_um']

    if isinstance(slide_data, dict):
        meta_px = slide_data.get('pixel_size_um')
        if meta_px:
            return meta_px

    warnings.warn(
        f"No pixel_size_um in crosses or replicate data, "
        f"using default {DEFAULT_PIXEL_SIZE_UM}",
        stacklevel=2,
    )
    return DEFAULT_PIXEL_SIZE_UM


def export_slide_xml(shapes_by_well, slide_name, output_path,
                     pixel_size_um, image_width_px, image_height_px,
                     crosses_data=None):
    """Export shapes to Leica LMD XML via py-lmd.

    Args:
        shapes_by_well: dict {well: [(uid, contour_um_xy), ...]}
            contour_um_xy is Nx2 array in [x, y] um, native CZI space.
        slide_name: for logging.
        output_path: where to save the XML.
        pixel_size_um: for coordinate transforms.
        image_width_px, image_height_px: CZI bounding box dimensions.
        crosses_data: dict from crosses JSON (napari), or None.
    """
    from lmd.lib import Collection
    from lmd.tools import makeCross

    img_w_um = image_width_px * pixel_size_um
    img_h_um = image_height_px * pixel_size_um

    if crosses_data:
        # Cross positions: X already flipped by napari (tissue-down),
        # flip Y for LMD convention
        calibration_points = np.array([
            [c['x_um'], img_h_um - c['y_um']]
            for c in crosses_data['crosses']
        ])
        log.info(f"  Calibration: user-placed crosses")
        for c in crosses_data['crosses']:
            log.info(f"    Cross {c.get('id', '?')}: napari ({c['x_um']:.0f}, "
                     f"{c['y_um']:.0f}) um -> XML ({c['x_um']:.0f}, "
                     f"{img_h_um - c['y_um']:.0f}) um")
    else:
        # Fallback: corners of slide extent
        calibration_points = np.array([
            [0, 0],
            [img_w_um, 0],
            [0, img_h_um],
        ])
        log.warning(f"  Calibration: NO CROSSES - using dummy corner points")

    collection = Collection(calibration_points=calibration_points)

    # Add calibration crosses
    for cx, cy in calibration_points:
        cross_col = makeCross(
            center=np.array([cx, cy]),
            arms=[100, 100, 100, 100], width=10, dist=5,
        )
        collection.join(cross_col)

    # Add shapes: flip to LMD coordinate system
    # Contours are in native CZI [x, y] um
    # LMD needs: flip X (tissue-down) + flip Y (LMD convention)
    n_shapes = 0
    for well, cell_list in shapes_by_well.items():
        for uid, contour_um in cell_list:
            polygon = contour_um.copy()
            polygon[:, 0] = img_w_um - polygon[:, 0]   # flip X (tissue-down)
            polygon[:, 1] = img_h_um - polygon[:, 1]   # flip Y (LMD convention)

            # Close polygon
            if not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0]])

            collection.new_shape(polygon, well=well, name=uid)
            n_shapes += 1

    collection.save(str(output_path))
    log.info(f"  Exported {n_shapes} shapes to {output_path}")
    return n_shapes


def main():
    parser = argparse.ArgumentParser(
        description='Generate LMD XML from pre-assigned replicate data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python lmd_export_replicates.py \\\n'
            '      --sampling-results lmd_replicates_full.json \\\n'
            '      --contours-json mk_contours_overlay.json \\\n'
            '      --crosses-dir ./crosses \\\n'
            '      --output-dir ./xml'
        ),
    )
    parser.add_argument('--sampling-results', required=True,
                        help='Path to replicate JSON (lmd_replicates_full.json)')
    parser.add_argument('--contours-json', required=True,
                        help='Pre-extracted contours JSON (mk_contours_overlay.json)')
    parser.add_argument('--crosses-dir', default=None,
                        help='Dir with {slide}_crosses.json from napari')
    parser.add_argument('--output-dir', required=True,
                        help='Where to save XML files')
    parser.add_argument('--slides', nargs='+', default=None,
                        help='Process only these slides (default: all)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.sampling_results) as f:
        all_results = json.load(f)

    with open(args.contours_json) as f:
        contours_by_slide = json.load(f)

    log.info(f"Replicate data: {len(all_results)} slides")
    log.info(f"Contour data: {len(contours_by_slide)} slides")
    if args.crosses_dir:
        log.info(f"Crosses dir: {args.crosses_dir}")
    else:
        log.warning("No --crosses-dir - XMLs will have dummy calibration points")

    slides = args.slides or list(all_results.keys())

    for slide_name in slides:
        if slide_name not in all_results:
            log.warning(f"Slide '{slide_name}' not in replicate JSON - skipping")
            continue

        slide_data = all_results[slide_name]
        log.info(f"\n{'='*60}")
        log.info(f"Slide: {slide_name}")
        log.info(f"{'='*60}")

        # Find crosses
        crosses_data = None
        crosses_path = find_slide_crosses(args.crosses_dir, slide_name)
        if crosses_path:
            with open(crosses_path) as f:
                crosses_data = json.load(f)
            log.info(f"  Crosses: {crosses_path.name} "
                     f"(saved {crosses_data.get('timestamp', '?')})")
        elif args.crosses_dir:
            log.warning(f"  Crosses: NONE FOUND for {slide_name}")
        else:
            log.info(f"  Crosses: none (no --crosses-dir)")

        # Get pixel size
        pixel_size_um = get_pixel_size(crosses_data, slide_data)
        log.info(f"  Pixel size: {pixel_size_um:.4f} um/px")

        # Get image dimensions from crosses (most reliable)
        if crosses_data:
            image_width_px = crosses_data['image_width_px']
            image_height_px = crosses_data['image_height_px']
        else:
            # Estimate from contour extents
            image_width_px = 0
            image_height_px = 0

        # Build shapes from contours
        slide_contours = contours_by_slide.get(slide_name, [])
        if not slide_contours:
            log.warning(f"  No contours in JSON for {slide_name} - skipping")
            continue

        # Index contours by UID for fast lookup
        contour_by_uid = {}
        for entry in slide_contours:
            contour_by_uid[entry['uid']] = entry

        shapes_by_well = {}
        n_ok = 0
        n_miss = 0
        max_x = max_y = 0

        for rep in slide_data.get('replicates', []):
            well = rep['well']
            shapes_by_well.setdefault(well, [])

            for cell in rep.get('cells', []):
                uid = cell['uid']
                entry = contour_by_uid.get(uid)
                if entry is None:
                    n_miss += 1
                    continue

                # contour_yx is [y, x] in global pixels -> [x, y] in um
                pts_yx = np.array(entry['contour_yx'])
                pts_xy_um = pts_yx[:, ::-1] * pixel_size_um
                shapes_by_well[well].append((uid, pts_xy_um))
                n_ok += 1

                max_x = max(max_x, pts_yx[:, 1].max())
                max_y = max(max_y, pts_yx[:, 0].max())

        log.info(f"  Contours matched: {n_ok}, missing: {n_miss}")

        # Use estimated extents if no crosses
        if not crosses_data:
            image_width_px = int(max_x + 1000)
            image_height_px = int(max_y + 1000)
            log.info(f"  Estimated image extent: {image_width_px} x {image_height_px} px")

        # Export XML
        xml_path = output_dir / f"{slide_name}_lmd.xml"
        n_shapes = export_slide_xml(
            shapes_by_well, slide_name, xml_path,
            pixel_size_um, image_width_px, image_height_px,
            crosses_data,
        )

        # Summary JSON
        summary = {
            'slide': slide_name,
            'pixel_size_um': pixel_size_um,
            'image_width_px': image_width_px,
            'image_height_px': image_height_px,
            'has_crosses': crosses_data is not None,
            'crosses_file': crosses_path.name if crosses_path else None,
            'n_shapes': n_shapes,
            'n_missing_contours': n_miss,
            'wells': {},
        }
        for well, cells in shapes_by_well.items():
            summary['wells'][well] = {
                'n_shapes': len(cells),
                'uids': [uid for uid, _ in cells],
            }
        summary_path = output_dir / f"{slide_name}_summary.json"
        atomic_json_dump(summary, str(summary_path))

    log.info("\nDone!")


if __name__ == '__main__':
    main()
