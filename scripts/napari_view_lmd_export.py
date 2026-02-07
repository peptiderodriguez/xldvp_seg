#!/usr/bin/env python3
"""
View LMD export in Napari with detection and control overlays.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Opens the OME-Zarr pyramid and overlays:
- Singles (green polygons)
- Single controls (cyan polygons)
- Clusters (red polygons)
- Cluster controls (orange polygons)

Usage:
    python napari_view_lmd_export.py --zarr /path/to/pyramid.zarr --export export.json
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
    """Load shapes from unified LMD export JSON.

    Reads the 'shapes' list with 'type' field:
    - single: one contour_um
    - single_control: one contour_um
    - cluster: multiple contours_um
    - cluster_control: multiple contours_um

    Returns dict with lists of (contour_px, tooltip) tuples per category.
    """
    with open(export_path) as f:
        data = json.load(f)

    pixel_size = data.get("metadata", {}).get("pixel_size_um", PIXEL_SIZE_UM)

    singles = []
    single_controls = []
    clusters = []
    cluster_controls = []

    def um_to_napari(contour_um):
        """Convert um contour to Napari [y, x] pixel coords."""
        return np.array([[pt[1] / pixel_size, pt[0] / pixel_size] for pt in contour_um])

    for shape in data.get("shapes", []):
        shape_type = shape.get("type", "")
        well = shape.get("well", "")
        uid = shape.get("uid", "")
        tooltip = f"{well} | {uid}"

        if shape_type in ("cluster", "cluster_control"):
            contours_um = shape.get("contours_um", [])
            target = clusters if shape_type == "cluster" else cluster_controls
            n_members = len(contours_um)
            for i, contour_um in enumerate(contours_um):
                if contour_um and len(contour_um) >= 3:
                    member_tip = f"{well} | {uid} [{i+1}/{n_members}]"
                    target.append((um_to_napari(contour_um), member_tip))
        elif shape_type in ("single", "single_control"):
            contour_um = shape.get("contour_um")
            if contour_um and len(contour_um) >= 3:
                contour_px = um_to_napari(contour_um)
                if shape_type == "single":
                    singles.append((contour_px, tooltip))
                else:
                    single_controls.append((contour_px, tooltip))

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

    n_s = len(shapes['singles'])
    n_sc = len(shapes['single_controls'])
    n_c = len(shapes['clusters'])
    n_cc = len(shapes['cluster_controls'])
    print(f"  Singles:          {n_s} contours")
    print(f"  Single controls:  {n_sc} contours")
    print(f"  Clusters:         {n_c} contours (cluster members)")
    print(f"  Cluster controls: {n_cc} contours")
    print(f"  Total shapes:     {n_s + n_sc + n_c + n_cc}")

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
            name="Singles",
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
            name="Clusters",
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
    print("  - Green: Singles")
    print("  - Cyan: Single controls")
    print("  - Red: Clusters")
    print("  - Orange: Cluster controls")
    print("="*60)

    napari.run()


if __name__ == "__main__":
    main()
