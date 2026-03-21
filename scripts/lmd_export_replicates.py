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
  - Crosses are in display space (after flip_h + rotate_cw_90) from napari
  - Export applies the same display transforms to contours so they match
    the crosses, then Y-flips for LMD convention
  - display_transform metadata in crosses JSON drives the coordinate mapping

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
    from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
except ImportError:
    def atomic_json_dump(data, path, **kwargs):
        with open(path, 'w') as f:
            json.dump(data, f)
    def fast_json_load(path):
        with open(path) as f:
            return json.load(f)


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


try:
    from segmentation.lmd.contour_processing import transform_native_to_display as _transform_native_to_display
except ImportError:
    def _transform_native_to_display(pts_xy_um, orig_w_um, orig_h_um, flip_h, rot90):
        """Fallback when segmentation package is not on PYTHONPATH."""
        pts = pts_xy_um.copy()
        if flip_h:
            pts[:, 0] = orig_w_um - pts[:, 0]
        if rot90:
            x_new = orig_h_um - pts[:, 1]
            y_new = pts[:, 0].copy()
            pts[:, 0] = x_new
            pts[:, 1] = y_new
        return pts


def _build_serpentine_index():
    """Build well→index mapping for 384-well serpentine order."""
    from segmentation.lmd.well_plate import generate_plate_wells
    wells = generate_plate_wells(308)
    return {w: i for i, w in enumerate(wells)}


_SERPENTINE_INDEX = None


def _well_sort_key(well):
    """Sort key for 384-well serpentine order (B2→B3→C3→C2)."""
    global _SERPENTINE_INDEX
    if _SERPENTINE_INDEX is None:
        _SERPENTINE_INDEX = _build_serpentine_index()
    return _SERPENTINE_INDEX.get(well, 999)


def export_slide_xml(shapes_by_well, slide_name, output_path,
                     pixel_size_um, image_width_px, image_height_px,
                     crosses_data=None, y_offset_um=0.0):
    """Export shapes to Leica LMD XML via py-lmd.

    Args:
        shapes_by_well: dict {well: [(uid, contour_um_xy), ...]}
            contour_um_xy is Nx2 array in [x, y] um, native CZI space.
        slide_name: for logging.
        output_path: where to save the XML.
        pixel_size_um: for coordinate transforms.
        image_width_px, image_height_px: CZI bounding box dimensions (original).
        crosses_data: dict from crosses JSON (napari), or None.
        y_offset_um: calibration correction added to contour Y in LMD space
            (compensates for systematic laser-to-calibration offset; crosses
            define the reference frame and are NOT shifted).
    """
    from lmd.lib import Collection

    orig_w_um = image_width_px * pixel_size_um
    orig_h_um = image_height_px * pixel_size_um

    # Read display transforms from crosses metadata
    flip_h = False
    rot90 = False
    if crosses_data:
        dt = crosses_data.get('display_transform', {})
        flip_h = dt.get('flip_horizontal', False)
        rot90 = dt.get('rotate_cw_90', False)

    # After rotation, display dimensions swap
    if rot90:
        display_h_um = orig_w_um   # original width becomes display height
    else:
        display_h_um = orig_h_um

    if crosses_data:
        # Crosses are already in display space; Y-flip for LMD convention
        calibration_points = np.array([
            [c['x_um'], display_h_um - c['y_um']]
            for c in crosses_data['crosses'][:3]
        ])
        log.info(f"  Calibration: user-placed crosses")
        log.info(f"  Display transforms: flip_h={flip_h}, rot90={rot90}")
        log.info(f"  Display height for LMD Y-flip: {display_h_um:.0f} um")
        for c in crosses_data['crosses']:
            log.info(f"    Cross {c.get('id', '?')}: napari ({c['x_um']:.0f}, "
                     f"{c['y_um']:.0f}) um -> XML ({c['x_um']:.0f}, "
                     f"{display_h_um - c['y_um']:.0f}) um")
    else:
        # Fallback: corners of slide extent (no transforms)
        calibration_points = np.array([
            [0, 0],
            [orig_w_um, 0],
            [0, orig_h_um],
        ])
        log.warning(f"  Calibration: NO CROSSES - using dummy corner points")

    collection = Collection(calibration_points=calibration_points)
    # Note: calibration points in the XML header are sufficient for LMD.
    # Visual cross shapes are NOT added — they have no well assignment and
    # confuse the LMD software (shows as extra shapes with 100/1000 um² area).

    # Bright colors for wells (RGBDef = R + G*256 + B*65536)
    WELL_COLORS = [
        (255,   0,   0),  # red
        (  0, 255,   0),  # green
        (  0, 100, 255),  # blue
        (255, 255,   0),  # yellow
        (255,   0, 255),  # magenta
        (  0, 255, 255),  # cyan
        (255, 128,   0),  # orange
        (128,   0, 255),  # purple
        (  0, 255, 128),  # spring green
        (255,   0, 128),  # rose
        (128, 255,   0),  # lime
        (  0, 128, 255),  # sky blue
    ]

    # Transform contours from native CZI space to display space (matching
    # crosses), then Y-flip for LMD convention.
    # Wells are sorted in serpentine order so the LMD processes them
    # in the same sequence as the plate layout.
    n_shapes = 0
    sorted_wells = sorted(shapes_by_well.keys(), key=_well_sort_key)
    for wi, well in enumerate(sorted_wells):
        r, g, b = WELL_COLORS[wi % len(WELL_COLORS)]
        cell_list = shapes_by_well[well]
        for uid, contour_um in cell_list:
            polygon = _transform_native_to_display(
                contour_um, orig_w_um, orig_h_um, flip_h, rot90)
            polygon[:, 1] = display_h_um - polygon[:, 1]   # Y-flip for LMD
            if y_offset_um != 0:
                polygon[:, 1] += y_offset_um               # calibration correction

            # Close polygon
            if not np.allclose(polygon[0], polygon[-1]):
                polygon = np.vstack([polygon, polygon[0]])

            collection.new_shape(polygon, well=well, name=uid)
            n_shapes += 1

    collection.save(str(output_path))

    # Patch RGBDef into saved XML — insert right after CapID element
    # (compatible with all py-lmd versions)
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(output_path))
    root = tree.getroot()
    shape_idx = 0
    for wi, well in enumerate(sorted_wells):
        r, g, b = WELL_COLORS[wi % len(WELL_COLORS)]
        rgb_def = r + g * 256 + b * 65536
        for _ in shapes_by_well[well]:
            shape_idx += 1
            shape_el = root.find(f'Shape_{shape_idx}')
            if shape_el is not None:
                rgb_el = ET.Element('RGBDef')
                rgb_el.text = str(rgb_def)
                # Insert after CapID (index 1) so LMD reads it before points
                cap_idx = None
                for ci, child in enumerate(shape_el):
                    if child.tag == 'CapID':
                        cap_idx = ci
                        break
                insert_pos = (cap_idx + 1) if cap_idx is not None else 1
                shape_el.insert(insert_pos, rgb_el)
    if shape_idx > 0:
        test_el = root.find('Shape_1')
        if test_el is None or test_el.find('RGBDef') is None:
            log.warning("  RGBDef patching may have failed — verify py-lmd XML format")
    tree.write(str(output_path), encoding='utf-8', xml_declaration=True)
    log.info(f"  Exported {n_shapes} shapes in {len(sorted_wells)} wells "
             f"(serpentine order) to {output_path}")
    if y_offset_um != 0:
        log.info(f"  Y offset correction: {y_offset_um:+.2f} um")
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
    parser.add_argument('--y-offset-um', type=float, default=0.0,
                        help='Y calibration correction in um (default: 0, compensates laser-to-calibration offset)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = fast_json_load(str(args.sampling_results))

    contours_by_slide = fast_json_load(str(args.contours_json))

    log.info(f"Replicate data: {len(all_results)} slides")
    log.info(f"Contour data: {len(contours_by_slide)} slides")
    if args.crosses_dir:
        log.info(f"Crosses dir: {args.crosses_dir}")
    else:
        log.warning("No --crosses-dir provided - all slides will be skipped")

    slides = args.slides or list(all_results.keys())

    for slide_name in slides:
        if slide_name not in all_results:
            log.warning(f"Slide '{slide_name}' not in replicate JSON - skipping")
            continue

        slide_data = all_results[slide_name]

        # Find crosses — skip slides without 3 reference crosses
        crosses_data = None
        crosses_path = find_slide_crosses(args.crosses_dir, slide_name)
        if crosses_path:
            crosses_data = fast_json_load(str(crosses_path))
            n_crosses = len(crosses_data.get('crosses', []))
            if n_crosses < 3:
                log.warning(f"Slide '{slide_name}': only {n_crosses} crosses "
                            f"(need 3) - skipping")
                continue
        else:
            log.info(f"Slide '{slide_name}': no crosses file - skipping")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Slide: {slide_name}")
        log.info(f"{'='*60}")
        log.info(f"  Crosses: {crosses_path.name} "
                 f"(saved {crosses_data.get('timestamp', '?')})")

        # Get pixel size
        pixel_size_um = get_pixel_size(crosses_data, slide_data)
        log.info(f"  Pixel size: {pixel_size_um:.4f} um/px")

        # Image dimensions from crosses (always available — slides without
        # crosses are skipped above)
        image_width_px = crosses_data['image_width_px']
        image_height_px = crosses_data['image_height_px']

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

        # Export XML
        xml_path = output_dir / f"{slide_name}_lmd.xml"
        n_shapes = export_slide_xml(
            shapes_by_well, slide_name, xml_path,
            pixel_size_um, image_width_px, image_height_px,
            crosses_data, y_offset_um=args.y_offset_um,
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
