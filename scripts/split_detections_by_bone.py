#!/usr/bin/env python
"""Split detections by bone region (femur/humerus).

Takes detection JSON and bone region annotations, assigns each detection
to a bone based on centroid containment, and outputs separate files.

Usage:
    python scripts/split_detections_by_bone.py \
        --detections /path/to/mk_contours_overlay.json \
        --regions /path/to/bone_regions.json \
        --output-dir /path/to/output

Output:
    - mk_contours_femur.json
    - mk_contours_humerus.json
    - bone_assignment_summary.json
"""
import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


def compute_centroid_from_contour(contour_yx):
    """Compute centroid from contour points.

    Args:
        contour_yx: List of [y, x] points (note: y,x order!)

    Returns:
        (x, y) centroid in standard x,y order
    """
    points = np.array(contour_yx)
    # contour_yx is [y, x] so we need to flip
    y_coords = points[:, 0]
    x_coords = points[:, 1]
    return float(np.mean(x_coords)), float(np.mean(y_coords))


def transform_rotated_to_original(x, y, orig_width, orig_height):
    """Transform coordinates from rotated space back to original CZI space.

    Inverse of: flip-horizontal then rotate-90-CW.
    Rotated image has dimensions (orig_height, orig_width).

    Args:
        x, y: Coordinates in rotated space
        orig_width: Original CZI width
        orig_height: Original CZI height

    Returns:
        (x_orig, y_orig): Coordinates in original CZI space
    """
    # Inverse of rotate-90-CW: rotate-90-CCW
    # (x, y) -> (y, rotated_width - x) where rotated_width = orig_height
    x_after_unrotate = y
    y_after_unrotate = orig_height - x

    # Inverse of flip-horizontal: flip-horizontal again
    # (x, y) -> (orig_width - x, y)
    x_orig = orig_width - x_after_unrotate
    y_orig = y_after_unrotate

    return x_orig, y_orig


def create_prepared_polygon(vertices):
    """Create a prepared polygon for fast repeated containment tests.

    Args:
        vertices: List of [x, y] vertices

    Returns:
        Prepared polygon or None if invalid
    """
    try:
        poly = Polygon(vertices)
        if not poly.is_valid:
            # Try to fix self-intersections
            poly = poly.buffer(0)
        if poly.is_valid and not poly.is_empty:
            return prep(poly)
    except Exception as e:
        print(f"  Warning: Invalid polygon: {e}")
    return None


def point_in_prepared_polygon(x, y, prepared_poly):
    """Test if point (x, y) is inside prepared polygon.

    Args:
        x, y: Point coordinates
        prepared_poly: Shapely prepared polygon

    Returns:
        bool: True if point is inside polygon
    """
    if prepared_poly is None:
        return False
    try:
        return prepared_poly.contains(Point(x, y))
    except Exception:
        return False


def load_detections(path):
    """Load detections from JSON.

    Handles the format: {slide_name: [detection, ...], ...}
    """
    with open(path) as f:
        return json.load(f)


def load_bone_regions(path):
    """Load bone region annotations from JSON.

    Expected format:
    {
        "slides": {
            "slide_name": {
                "femur": {"vertices_px": [[x,y], ...]},
                "humerus": {"vertices_px": [[x,y], ...]}
            }
        }
    }
    """
    with open(path) as f:
        data = json.load(f)

    # Handle both formats
    if 'slides' in data:
        return data['slides']
    return data


def assign_detections_to_bones(detections_by_slide, regions_by_slide, include_unknown=False,
                                transform_from_rotated=False):
    """Assign each detection to a bone region.

    Args:
        detections_by_slide: {slide_name: [detection, ...]}
        regions_by_slide: {slide_name: {femur: {vertices_px: [...]}, humerus: {...}}}
        include_unknown: If True, include detections outside both regions in output
        transform_from_rotated: If True, transform detection coords from rotated space

    Returns:
        femur_detections: {slide_name: [detection, ...]}
        humerus_detections: {slide_name: [detection, ...]}
        unknown_detections: {slide_name: [detection, ...]} (if include_unknown)
        summary: assignment statistics
    """
    femur_detections = defaultdict(list)
    humerus_detections = defaultdict(list)
    unknown_detections = defaultdict(list)

    summary = {
        'by_slide': {},
        'totals': {
            'femur': 0,
            'humerus': 0,
            'unknown': 0,
            'total': 0
        }
    }

    for slide_name, detections in detections_by_slide.items():
        slide_regions = regions_by_slide.get(slide_name, {})

        # Create prepared polygons ONCE per slide for fast repeated tests
        femur_poly = None
        humerus_poly = None

        if 'femur' in slide_regions and slide_regions['femur'].get('vertices_px'):
            femur_poly = create_prepared_polygon(slide_regions['femur']['vertices_px'])
        if 'humerus' in slide_regions and slide_regions['humerus'].get('vertices_px'):
            humerus_poly = create_prepared_polygon(slide_regions['humerus']['vertices_px'])

        slide_summary = {'femur': 0, 'humerus': 0, 'unknown': 0, 'total': len(detections)}

        # Get original dimensions for coordinate transform
        orig_width = slide_regions.get('full_width')
        orig_height = slide_regions.get('full_height')

        for det in detections:
            # Compute centroid from contour_yx
            if 'contour_yx' in det:
                cx, cy = compute_centroid_from_contour(det['contour_yx'])
            elif 'centroid' in det:
                cx, cy = det['centroid']
            elif 'global_center' in det:
                cx, cy = det['global_center']
            else:
                # Try to parse from UID
                parts = det.get('uid', '').split('_')
                if len(parts) >= 2:
                    try:
                        cx, cy = float(parts[-2]), float(parts[-1])
                    except ValueError:
                        print(f"  Warning: Cannot determine centroid for {det.get('uid')}")
                        slide_summary['unknown'] += 1
                        continue
                else:
                    slide_summary['unknown'] += 1
                    continue

            # Transform from rotated space to original CZI space if needed
            if transform_from_rotated and orig_width and orig_height:
                cx, cy = transform_rotated_to_original(cx, cy, orig_width, orig_height)

            # Test containment using prepared polygons (fast!)
            bone_region = None

            if femur_poly and point_in_prepared_polygon(cx, cy, femur_poly):
                bone_region = 'femur'
            elif humerus_poly and point_in_prepared_polygon(cx, cy, humerus_poly):
                bone_region = 'humerus'

            # Deep copy to avoid shared references
            det_copy = copy.deepcopy(det)
            det_copy['bone_region'] = bone_region or 'unknown'
            det_copy['centroid_xy'] = [cx, cy]

            if bone_region == 'femur':
                femur_detections[slide_name].append(det_copy)
                slide_summary['femur'] += 1
            elif bone_region == 'humerus':
                humerus_detections[slide_name].append(det_copy)
                slide_summary['humerus'] += 1
            else:
                slide_summary['unknown'] += 1
                if include_unknown:
                    unknown_detections[slide_name].append(det_copy)

        summary['by_slide'][slide_name] = slide_summary
        summary['totals']['femur'] += slide_summary['femur']
        summary['totals']['humerus'] += slide_summary['humerus']
        summary['totals']['unknown'] += slide_summary['unknown']
        summary['totals']['total'] += slide_summary['total']

        print(f"  {slide_name}: {slide_summary['femur']} femur, "
              f"{slide_summary['humerus']} humerus, {slide_summary['unknown']} unknown")

    result = (dict(femur_detections), dict(humerus_detections), summary)
    if include_unknown:
        result = (dict(femur_detections), dict(humerus_detections), dict(unknown_detections), summary)
    return result


def main():
    parser = argparse.ArgumentParser(description='Split detections by bone region')
    parser.add_argument('--detections', '-d', type=Path, required=True,
                        help='Path to detection JSON (e.g., mk_contours_overlay.json)')
    parser.add_argument('--regions', '-r', type=Path, required=True,
                        help='Path to bone regions JSON (from annotation tool)')
    parser.add_argument('--output-dir', '-o', type=Path, required=True,
                        help='Output directory for split files')
    parser.add_argument('--prefix', type=str, default='mk_contours',
                        help='Output file prefix (default: mk_contours)')
    parser.add_argument('--include-unknown', action='store_true',
                        help='Also output detections outside both regions')
    parser.add_argument('--transform-from-rotated', action='store_true',
                        help='Transform detection coords from flip-H + rotate-90-CW space')

    args = parser.parse_args()

    # Validate inputs
    if not args.detections.exists():
        print(f"Error: Detection file not found: {args.detections}")
        sys.exit(1)
    if not args.regions.exists():
        print(f"Error: Regions file not found: {args.regions}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading detections from {args.detections}...")
    detections = load_detections(args.detections)
    print(f"  Found {sum(len(v) for v in detections.values())} detections across {len(detections)} slides")

    print(f"Loading bone regions from {args.regions}...")
    regions = load_bone_regions(args.regions)
    print(f"  Found regions for {len(regions)} slides")

    # Check for missing regions
    missing_regions = set(detections.keys()) - set(regions.keys())
    if missing_regions:
        print(f"  Warning: No regions for slides: {missing_regions}")

    # Assign detections to bones
    print("\nAssigning detections to bone regions...")
    if args.transform_from_rotated:
        print("  (transforming detection coords from rotated space)")
    result = assign_detections_to_bones(detections, regions,
                                        include_unknown=args.include_unknown,
                                        transform_from_rotated=args.transform_from_rotated)

    if args.include_unknown:
        femur_dets, humerus_dets, unknown_dets, summary = result
    else:
        femur_dets, humerus_dets, summary = result

    # Write output files
    femur_path = args.output_dir / f'{args.prefix}_femur.json'
    humerus_path = args.output_dir / f'{args.prefix}_humerus.json'
    summary_path = args.output_dir / 'bone_assignment_summary.json'

    print(f"\nWriting {femur_path}...")
    with open(femur_path, 'w') as f:
        json.dump(femur_dets, f, indent=2)

    print(f"Writing {humerus_path}...")
    with open(humerus_path, 'w') as f:
        json.dump(humerus_dets, f, indent=2)

    output_files = {
        'femur': str(femur_path),
        'humerus': str(humerus_path)
    }

    if args.include_unknown:
        unknown_path = args.output_dir / f'{args.prefix}_unknown.json'
        print(f"Writing {unknown_path}...")
        with open(unknown_path, 'w') as f:
            json.dump(unknown_dets, f, indent=2)
        output_files['unknown'] = str(unknown_path)

    # Add metadata to summary
    summary['metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'detections_file': str(args.detections),
        'regions_file': str(args.regions),
        'output_files': output_files
    }

    print(f"Writing {summary_path}...")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total detections:   {summary['totals']['total']}")
    print(f"  Femur:            {summary['totals']['femur']}")
    print(f"  Humerus:          {summary['totals']['humerus']}")
    print(f"  Unknown/Outside:  {summary['totals']['unknown']}")
    print(f"\nOutput files:")
    print(f"  {femur_path}")
    print(f"  {humerus_path}")
    if args.include_unknown:
        print(f"  {unknown_path}")
    print(f"  {summary_path}")


if __name__ == '__main__':
    main()
