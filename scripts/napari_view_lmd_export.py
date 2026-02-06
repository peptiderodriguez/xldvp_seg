#!/usr/bin/env python3
"""
View LMD export in Napari with NMJ and control overlays.

Opens the OME-Zarr pyramid and overlays:
- Singles (green polygons)
- Single controls (cyan polygons)
- Clusters (red polygons)
- Cluster controls (orange polygons)

Usage:
    python napari_view_export.py
    python napari_view_export.py --zarr /path/to/pyramid.zarr
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import napari
except ImportError:
    print("ERROR: napari not installed. Install with: pip install napari[all]")
    exit(1)

try:
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
except ImportError:
    print("ERROR: ome-zarr not installed. Install with: pip install ome-zarr")
    exit(1)

# Default paths (set via CLI args)
DEFAULT_ZARR = None
DEFAULT_EXPORT = None

PIXEL_SIZE_UM = 0.1725


def load_export_shapes(export_path: Path):
    """Load shapes from LMD export JSON."""
    with open(export_path) as f:
        data = json.load(f)

    pixel_size = data.get("metadata", {}).get("pixel_size_um", PIXEL_SIZE_UM)

    singles = []
    single_controls = []
    clusters = []
    cluster_controls = []

    for shape in data.get("shapes", []):
        shape_type = shape.get("type", "")
        well = shape.get("well", "")

        # Get contour(s)
        if shape_type == "cluster":
            # Clusters have multiple contours
            contours_um = shape.get("contours_um", [])
            for contour_um in contours_um:
                if contour_um and len(contour_um) >= 3:
                    # Convert µm to pixels, and to [y, x] for Napari
                    contour_px = np.array([[pt[1] / pixel_size, pt[0] / pixel_size] for pt in contour_um])
                    clusters.append((contour_px, well))
        else:
            contour_um = shape.get("contour_um")
            if contour_um and len(contour_um) >= 3:
                # Convert µm to pixels, and to [y, x] for Napari
                contour_px = np.array([[pt[1] / pixel_size, pt[0] / pixel_size] for pt in contour_um])

                if shape_type == "single":
                    singles.append((contour_px, well))
                elif shape_type == "single_control":
                    single_controls.append((contour_px, well))

        # Handle cluster_control with multiple contours (same as clusters)
        if shape_type == "cluster_control":
            contours_um = shape.get("contours_um", [])
            if contours_um:
                for contour_um in contours_um:
                    if contour_um and len(contour_um) >= 3:
                        contour_px = np.array([[pt[1] / pixel_size, pt[0] / pixel_size] for pt in contour_um])
                        cluster_controls.append((contour_px, well))
            else:
                # Fallback to single contour_um if contours_um not present
                contour_um = shape.get("contour_um")
                if contour_um and len(contour_um) >= 3:
                    contour_px = np.array([[pt[1] / pixel_size, pt[0] / pixel_size] for pt in contour_um])
                    cluster_controls.append((contour_px, well))

    return {
        "singles": singles,
        "single_controls": single_controls,
        "clusters": clusters,
        "cluster_controls": cluster_controls,
    }


def main():
    parser = argparse.ArgumentParser(description="View LMD export in Napari")
    parser.add_argument("--zarr", type=str, required=True,
                        help="Path to OME-Zarr pyramid")
    parser.add_argument("--export", type=str, required=True,
                        help="Path to LMD export JSON (lmd_export_with_controls.json)")
    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    export_path = Path(args.export)

    if not zarr_path.exists():
        print(f"ERROR: Zarr not found: {zarr_path}")
        return

    if not export_path.exists():
        print(f"ERROR: Export not found: {export_path}")
        return

    # Load OME-Zarr
    print(f"Loading OME-Zarr: {zarr_path}")
    store = parse_url(zarr_path, mode="r")
    reader = Reader(store)
    nodes = list(reader())

    if not nodes:
        print("ERROR: No data found in zarr")
        return

    image_data = nodes[0].data
    print(f"  Pyramid levels: {len(image_data)}")
    print(f"  Full resolution shape: {image_data[0].shape}")

    # Load export shapes
    print(f"Loading export: {export_path}")
    shapes = load_export_shapes(export_path)
    print(f"  Singles: {len(shapes['singles'])}")
    print(f"  Single controls: {len(shapes['single_controls'])}")
    print(f"  Clusters: {len(shapes['clusters'])}")
    print(f"  Cluster controls: {len(shapes['cluster_controls'])}")

    # Create Napari viewer
    viewer = napari.Viewer(title="LMD Export Viewer")

    # Add image
    viewer.add_image(
        image_data,
        name=zarr_path.stem,
        multiscale=True,
    )

    # Add shapes layers
    # Singles - green
    if shapes["singles"]:
        single_polygons = [s[0] for s in shapes["singles"]]
        single_labels = [s[1] for s in shapes["singles"]]
        viewer.add_shapes(
            single_polygons,
            shape_type="polygon",
            name="Singles (NMJ)",
            edge_color="lime",
            face_color=[0, 1, 0, 0.2],
            edge_width=2,
            text={"string": single_labels, "size": 10, "color": "lime"},
        )

    # Single controls - cyan
    if shapes["single_controls"]:
        ctrl_polygons = [s[0] for s in shapes["single_controls"]]
        ctrl_labels = [s[1] for s in shapes["single_controls"]]
        viewer.add_shapes(
            ctrl_polygons,
            shape_type="polygon",
            name="Single Controls",
            edge_color="cyan",
            face_color=[0, 1, 1, 0.2],
            edge_width=2,
            text={"string": ctrl_labels, "size": 10, "color": "cyan"},
        )

    # Clusters - red
    if shapes["clusters"]:
        cluster_polygons = [s[0] for s in shapes["clusters"]]
        cluster_labels = [s[1] for s in shapes["clusters"]]
        viewer.add_shapes(
            cluster_polygons,
            shape_type="polygon",
            name="Clusters (NMJ)",
            edge_color="red",
            face_color=[1, 0, 0, 0.2],
            edge_width=2,
            text={"string": cluster_labels, "size": 10, "color": "red"},
        )

    # Cluster controls - orange
    if shapes["cluster_controls"]:
        cctrl_polygons = [s[0] for s in shapes["cluster_controls"]]
        cctrl_labels = [s[1] for s in shapes["cluster_controls"]]
        viewer.add_shapes(
            cctrl_polygons,
            shape_type="polygon",
            name="Cluster Controls",
            edge_color="orange",
            face_color=[1, 0.5, 0, 0.2],
            edge_width=2,
            text={"string": cctrl_labels, "size": 10, "color": "orange"},
        )

    print("\n" + "="*60)
    print("NAPARI VIEWER READY")
    print("  - Green: Singles (NMJs)")
    print("  - Cyan: Single controls")
    print("  - Red: Clusters (NMJs)")
    print("  - Orange: Cluster controls")
    print("="*60)

    napari.run()


if __name__ == "__main__":
    main()
