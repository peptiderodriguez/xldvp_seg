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


def create_lmd_xml(
    export_data: Dict,
    reference_crosses: Optional[List[Dict]] = None,
    output_path: Optional[Path] = None
) -> str:
    """
    Create Leica LMD XML from export data.

    Args:
        export_data: Full export with singles and clusters
        reference_crosses: List of cross positions [{'x': um, 'y': um}, ...]
        output_path: Where to save XML

    Returns:
        XML string
    """
    # Create root element
    root = ET.Element("ImageData")
    root.set("version", "1.0")

    # Add metadata
    meta = ET.SubElement(root, "GlobalImageInfo")
    ET.SubElement(meta, "ImageWidth").text = "254976"  # Full mosaic width in pixels
    ET.SubElement(meta, "ImageHeight").text = "100503"
    ET.SubElement(meta, "PixelSizeX").text = str(export_data['metadata']['pixel_size_um'])
    ET.SubElement(meta, "PixelSizeY").text = str(export_data['metadata']['pixel_size_um'])

    # Create shapes container
    shapes = ET.SubElement(root, "ShapeList")

    shape_id = 1

    # Add reference crosses first (if provided)
    if reference_crosses:
        # Handle both formats: list of crosses or dict with 'crosses' key
        crosses_list = reference_crosses.get('crosses', reference_crosses) if isinstance(reference_crosses, dict) else reference_crosses

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
        ET.SubElement(info, "Area").text = f"{single['area_um2']:.2f}"
        ET.SubElement(info, "WellIndex").text = str(single['well_index'])

        shape_id += 1

    # Add clusters
    for cluster in export_data['clusters']:
        # Create a group for the cluster
        group = ET.SubElement(shapes, "ShapeGroup")
        group.set("id", str(shape_id))
        group.set("name", f"{cluster['well']}_cluster_{cluster['cluster_id']}")
        group.set("well", cluster['well'])
        group.set("category", "cluster")
        group.set("n_nmjs", str(cluster['n_nmjs']))

        shape_id += 1

        # Add each NMJ in the cluster
        for nmj in cluster['nmjs']:
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


def main(output_dir: Path):
    print("=" * 70)
    print("GENERATE LMD XML")
    print("=" * 70)

    # Load export data
    export_path = output_dir / "lmd_export_full.json"
    print(f"\nLoading: {export_path}")
    with open(export_path) as f:
        export_data = json.load(f)

    print(f"  Singles: {export_data['summary']['n_singles']}")
    print(f"  Clusters: {export_data['summary']['n_clusters']}")
    print(f"  Total NMJs: {export_data['summary']['n_singles'] + export_data['summary']['n_nmjs_in_clusters']}")

    # Load reference crosses if available
    crosses_path = output_dir / "reference_crosses.json"
    reference_crosses = None
    if crosses_path.exists():
        print(f"\nLoading reference crosses: {crosses_path}")
        with open(crosses_path) as f:
            reference_crosses = json.load(f)
        crosses_list = reference_crosses.get('crosses', reference_crosses) if isinstance(reference_crosses, dict) else reference_crosses
        print(f"  Found {len(crosses_list)} crosses")
    else:
        print(f"\nNo reference crosses found at {crosses_path}")
        print("  You'll need to place these in Napari before LMD calibration")

    # Generate XML
    print("\nGenerating XML...")
    output_path = output_dir / "shapes.xml"
    create_lmd_xml(export_data, reference_crosses, output_path)

    # Summary
    print("\n" + "=" * 70)
    print("XML GENERATION COMPLETE")
    print("=" * 70)

    # Count shapes
    n_shapes = export_data['summary']['n_singles']
    for cluster in export_data['clusters']:
        n_shapes += cluster['n_nmjs']

    print(f"  Output: {output_path}")
    print(f"  Total shapes: {n_shapes}")
    if reference_crosses:
        print(f"  Reference crosses: {len(reference_crosses)}")
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
    args = parser.parse_args()
    main(args.output_dir)
