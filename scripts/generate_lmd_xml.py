#!/usr/bin/env python3
"""
Generate Leica LMD XML from export data.

Creates shapes.xml file compatible with Leica LMD7 software.
Includes reference crosses for calibration.
"""

import json
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def _extract_crosses_list(reference_crosses):
    """Extract list of crosses from either a list or a dict with 'crosses' key."""
    if isinstance(reference_crosses, dict):
        crosses_list = reference_crosses.get('crosses', [])
        if not isinstance(crosses_list, list):
            raise ValueError(
                f"reference_crosses['crosses'] must be a list, got {type(crosses_list).__name__}"
            )
        return crosses_list
    elif isinstance(reference_crosses, list):
        return reference_crosses
    else:
        raise ValueError(
            f"reference_crosses must be a list or dict with 'crosses' key, "
            f"got {type(reference_crosses).__name__}"
        )


def create_lmd_xml(
    export_data: Dict,
    reference_crosses: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> str:
    """
    Create Leica LMD XML from export data.

    Args:
        export_data: Full export with singles and clusters
        reference_crosses: List of cross positions [{'x': um, 'y': um}, ...] or
                          dict with 'crosses' key and optional 'image_width_px'/'image_height_px'
        output_path: Where to save XML
        image_width: Image width in pixels (overrides other sources)
        image_height: Image height in pixels (overrides other sources)

    Returns:
        XML string
    """
    # Resolve image dimensions from multiple sources:
    # 1. Explicit arguments (highest priority)
    # 2. reference_crosses dict (from napari_place_crosses.py)
    # 3. export_data metadata
    if image_width is None and isinstance(reference_crosses, dict):
        image_width = reference_crosses.get('image_width_px')
    if image_width is None:
        image_width = export_data.get('metadata', {}).get('image_width_px')
    if image_width is None:
        raise ValueError(
            "Image width not available. Provide --image-width CLI arg, or ensure "
            "reference_crosses.json or export metadata contains 'image_width_px'."
        )

    if image_height is None and isinstance(reference_crosses, dict):
        image_height = reference_crosses.get('image_height_px')
    if image_height is None:
        image_height = export_data.get('metadata', {}).get('image_height_px')
    if image_height is None:
        raise ValueError(
            "Image height not available. Provide --image-height CLI arg, or ensure "
            "reference_crosses.json or export metadata contains 'image_height_px'."
        )

    # Create root element
    root = ET.Element("ImageData")
    root.set("version", "1.0")

    # Add metadata
    meta = ET.SubElement(root, "GlobalImageInfo")
    ET.SubElement(meta, "ImageWidth").text = str(int(image_width))
    ET.SubElement(meta, "ImageHeight").text = str(int(image_height))
    ET.SubElement(meta, "PixelSizeX").text = str(export_data['metadata']['pixel_size_um'])
    ET.SubElement(meta, "PixelSizeY").text = str(export_data['metadata']['pixel_size_um'])

    # ---- Format compatibility layer ----
    # The unified pipeline (run_lmd_export.py) produces export_data['shapes']
    # with 'type' fields, while the legacy pipeline (generate_full_lmd_export.py)
    # produces export_data['singles'] and export_data['clusters'] directly.
    if 'shapes' in export_data and 'singles' not in export_data:
        # Unified format: extract singles/clusters from shapes list by type
        singles = [s for s in export_data['shapes'] if s.get('type') == 'single']
        clusters_raw = [s for s in export_data['shapes'] if s.get('type') == 'cluster']
        # Convert cluster shapes to the expected format with 'detections' as 'nmjs'
        clusters = []
        for c in clusters_raw:
            clusters.append({
                'well': c.get('well', ''),
                'well_index': c.get('well_index', 0),
                'cluster_id': c.get('cluster_id', 0),
                'n_detections': len(c.get('detections', c.get('nmjs', []))),
                'detections': c.get('detections', c.get('nmjs', [])),
            })
        export_data = dict(export_data)  # Don't mutate original
        export_data['singles'] = singles
        export_data['clusters'] = clusters
    else:
        # Legacy format: singles/clusters already present, use as-is
        if 'singles' not in export_data:
            export_data['singles'] = []
        if 'clusters' not in export_data:
            export_data['clusters'] = []

    # Create shapes container
    shapes = ET.SubElement(root, "ShapeList")

    shape_id = 1

    # Add reference crosses first (if provided)
    if reference_crosses:
        crosses_list = _extract_crosses_list(reference_crosses)

        for cross in crosses_list:
            shape = ET.SubElement(shapes, "Shape")
            shape.set("id", str(shape_id))
            shape.set("type", "ReferenceCross")
            shape.set("name", f"Cross_{cross.get('id', shape_id)}")

            # Position in micrometers (handle both 'x'/'y' and 'x_um'/'y_um' formats)
            x_um = cross.get('x_um', cross.get('x', 0))
            y_um = cross.get('y_um', cross.get('y', 0))

            pos = ET.SubElement(shape, "Position")
            ET.SubElement(pos, "X").text = f"{x_um:.2f}"
            ET.SubElement(pos, "Y").text = f"{y_um:.2f}"

            shape_id += 1

    # Add singles
    for single in export_data['singles']:
        shape = ET.SubElement(shapes, "Shape")
        shape.set("id", str(shape_id))
        shape.set("type", "Polygon")
        shape.set("name", f"{single['well']}_{single['uid']}")
        shape.set("well", single['well'])
        shape.set("category", "single")

        # Add contour points
        contour = ET.SubElement(shape, "Contour")
        for point in single['contour_um']:
            pt = ET.SubElement(contour, "Point")
            ET.SubElement(pt, "X").text = f"{point[0]:.2f}"
            ET.SubElement(pt, "Y").text = f"{point[1]:.2f}"

        # Add metadata
        info = ET.SubElement(shape, "Info")
        ET.SubElement(info, "Area").text = f"{single.get('area_um2', 0):.2f}"
        ET.SubElement(info, "WellIndex").text = str(single.get('well_index', shape_id))

        shape_id += 1

    # Add clusters
    for cluster in export_data['clusters']:
        # Create a group for the cluster
        group = ET.SubElement(shapes, "ShapeGroup")
        group.set("id", str(shape_id))
        group.set("name", f"{cluster['well']}_cluster_{cluster['cluster_id']}")
        group.set("well", cluster['well'])
        group.set("category", "cluster")
        n_items = cluster.get('n_detections', cluster.get('n_nmjs', 0))
        group.set("n_detections", str(n_items))

        shape_id += 1

        # Add each detection in the cluster (supports both 'detections' and legacy 'nmjs' key)
        cluster_items = cluster.get('detections', cluster.get('nmjs', []))
        for nmj in cluster_items:
            shape = ET.SubElement(group, "Shape")
            shape.set("id", str(shape_id))
            shape.set("type", "Polygon")
            shape.set("name", nmj['uid'])

            contour = ET.SubElement(shape, "Contour")
            for point in nmj['contour_um']:
                pt = ET.SubElement(contour, "Point")
                ET.SubElement(pt, "X").text = f"{point[0]:.2f}"
                ET.SubElement(pt, "Y").text = f"{point[1]:.2f}"

            info = ET.SubElement(shape, "Info")
            ET.SubElement(info, "Area").text = f"{nmj['area_um2']:.2f}"

            shape_id += 1

    # Pretty print
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(pretty_xml)
        print(f"Saved XML: {output_path}")

    return pretty_xml


def main(output_dir: Path, image_width: Optional[int] = None,
         image_height: Optional[int] = None):
    print("=" * 70)
    print("GENERATE LMD XML")
    print("=" * 70)

    # Load export data
    export_path = output_dir / "lmd_export_full.json"
    print(f"\nLoading: {export_path}")
    with open(export_path) as f:
        export_data = json.load(f)

    n_singles = export_data.get('summary', {}).get('n_singles', 0)
    n_clusters = export_data.get('summary', {}).get('n_clusters', 0)
    n_in_clusters = export_data.get('summary', {}).get('n_detections_in_clusters',
                        export_data.get('summary', {}).get('n_nmjs_in_clusters', 0))
    print(f"  Singles: {n_singles}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Total detections: {n_singles + n_in_clusters}")

    # Load reference crosses if available
    crosses_path = output_dir / "reference_crosses.json"
    reference_crosses = None
    if crosses_path.exists():
        print(f"\nLoading reference crosses: {crosses_path}")
        with open(crosses_path) as f:
            reference_crosses = json.load(f)
        crosses_list = _extract_crosses_list(reference_crosses)
        print(f"  Found {len(crosses_list)} crosses")
    else:
        print(f"\nNo reference crosses found at {crosses_path}")
        print("  You'll need to place these in Napari before LMD calibration")

    # Generate XML
    print("\nGenerating XML...")
    output_path = output_dir / "shapes.xml"
    create_lmd_xml(export_data, reference_crosses, output_path,
                   image_width=image_width, image_height=image_height)

    # Summary
    print("\n" + "=" * 70)
    print("XML GENERATION COMPLETE")
    print("=" * 70)

    # Count shapes
    n_shapes = export_data.get('summary', {}).get('n_singles', 0)
    for cluster in export_data.get('clusters', []):
        n_shapes += cluster.get('n_detections', cluster.get('n_nmjs', 0))

    print(f"  Output: {output_path}")
    print(f"  Total shapes: {n_shapes}")
    if reference_crosses:
        crosses_list = _extract_crosses_list(reference_crosses)
        print(f"  Reference crosses: {len(crosses_list)}")
    print(f"\nNext steps:")
    print(f"  1. Place reference crosses in Napari (if not done)")
    print(f"  2. Transfer shapes.xml to LMD computer")
    print(f"  3. Load in Leica LMD software")
    print(f"  4. Calibrate using reference crosses")
    print(f"  5. Start collection!")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Leica LMD XML from export data')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory containing lmd_export_full.json (and optionally reference_crosses.json)')
    parser.add_argument('--image-width', type=int, default=None,
                        help='Full mosaic width in pixels (auto-read from reference_crosses.json or export metadata if available)')
    parser.add_argument('--image-height', type=int, default=None,
                        help='Full mosaic height in pixels (auto-read from reference_crosses.json or export metadata if available)')
    args = parser.parse_args()
    main(args.output_dir, image_width=args.image_width,
         image_height=args.image_height)
