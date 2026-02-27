#!/usr/bin/env python3
"""
Napari-based interactive reference cross placement for LMD export.

This script provides an interactive Napari viewer for placing reference crosses
on OME-Zarr pyramid images. These crosses are used for calibrating the Leica
LMD (Laser Microdissection) system.

Features:
- Opens OME-Zarr pyramids with lazy loading
- Interactive cross placement with visual feedback
- Keyboard shortcuts for save, undo, clear, quit
- Optional overlay of existing detections for context
- Exports JSON compatible with run_lmd_export.py

Usage:
    python napari_place_crosses.py /path/to/image.zarr --output crosses.json

    # With detection overlay:
    python napari_place_crosses.py /path/to/image.zarr \\
        --output crosses.json \\
        --detections /path/to/detections.json

Keyboard Shortcuts:
    S - Save crosses to JSON
    U - Undo last cross
    C - Clear all crosses
    Q - Quit and save (if >= 3 crosses placed)

Author: xldvp_seg pipeline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import napari
    from napari.utils.notifications import show_info, show_warning, show_error
except ImportError:
    print("ERROR: napari not installed. Install with:")
    print("  pip install napari[all]")
    sys.exit(1)

try:
    import zarr
except ImportError:
    print("ERROR: zarr not installed. Install with:")
    print("  pip install zarr")
    sys.exit(1)

try:
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
except ImportError:
    print("ERROR: ome-zarr not installed. Install with:")
    print("  pip install ome-zarr")
    sys.exit(1)


def load_ome_zarr(zarr_path: str):
    """
    Load OME-Zarr pyramid for Napari viewing.

    Returns:
        tuple: (data, pixel_size_um, image_shape)
            - data: list of dask arrays (pyramid levels)
            - pixel_size_um: pixel size in micrometers
            - image_shape: (height, width) of full resolution image
    """
    zarr_path = Path(zarr_path)

    # Parse the OME-Zarr
    store = parse_url(zarr_path, mode="r")
    if store is None:
        raise ValueError(f"Could not parse OME-Zarr at: {zarr_path}")

    reader = Reader(store)
    nodes = list(reader())

    if not nodes:
        raise ValueError(f"No data found in OME-Zarr: {zarr_path}")

    # Get image data (pyramid levels)
    node = nodes[0]
    data = node.data  # List of dask arrays for each pyramid level

    # Extract pixel size from zarr metadata
    pixel_size_um = 0.1725  # Default fallback

    try:
        root = zarr.open(zarr_path, mode='r')
        if 'multiscales' in root.attrs:
            multiscales = root.attrs['multiscales']
            if multiscales and len(multiscales) > 0:
                ms = multiscales[0]
                if 'datasets' in ms and len(ms['datasets']) > 0:
                    ds = ms['datasets'][0]
                    if 'coordinateTransformations' in ds:
                        transforms = ds['coordinateTransformations']
                        for t in transforms:
                            if t.get('type') == 'scale':
                                scale = t.get('scale', [])
                                # Usually [c, y, x] or [z, y, x] or [y, x]
                                if len(scale) >= 2:
                                    pixel_size_um = scale[-1]  # x scale
                                    break
    except Exception as e:
        print(f"Warning: Could not read pixel size from metadata: {e}")

    # Get image dimensions from highest resolution level
    # Use zarr metadata for reliable axis order
    full_res = data[0]
    image_shape = None

    try:
        if 'multiscales' in root.attrs:
            ms = root.attrs['multiscales'][0]
            if 'axes' in ms:
                axes = [a['name'] for a in ms['axes']]
                # Find y and x indices
                y_idx = axes.index('y') if 'y' in axes else None
                x_idx = axes.index('x') if 'x' in axes else None
                if y_idx is not None and x_idx is not None:
                    image_shape = (full_res.shape[y_idx], full_res.shape[x_idx])
    except Exception:
        pass

    # Fallback: assume last two dims are Y, X
    if image_shape is None:
        image_shape = full_res.shape[-2:]

    return data, pixel_size_um, image_shape


def load_detections(detections_path: str) -> list:
    """
    Load detection coordinates from JSON file.

    Returns:
        list: List of [y, x] coordinates (Napari format)
    """
    with open(detections_path, 'r') as f:
        data = json.load(f)

    # Handle both list and dict with 'detections' key
    if isinstance(data, dict):
        if 'detections' in data:
            detections = data['detections']
        else:
            detections = list(data.values()) if data else []
    else:
        detections = data

    coords = []
    for det in detections:
        # Try different coordinate formats
        if 'global_center' in det:
            x, y = det['global_center']
            coords.append([y, x])  # Napari uses [y, x]
        elif 'center' in det:
            x, y = det['center']
            coords.append([y, x])
        elif 'x_px' in det and 'y_px' in det:
            coords.append([det['y_px'], det['x_px']])

    return coords


class CrossPlacer:
    """
    Interactive cross placement manager for Napari.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        pixel_size_um: float,
        image_width_px: int,
        image_height_px: int,
        output_path: str
    ):
        self.viewer = viewer
        self.pixel_size_um = pixel_size_um
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        self.output_path = output_path
        self.crosses = []

        # Create the points layer for crosses
        self.points_layer = viewer.add_points(
            np.empty((0, 2)),
            name='Reference Crosses',
            symbol='cross',
            size=50,
            border_color='red',
            face_color='red',
            border_width=0.1,
            out_of_slice_display=True,
        )

        # Enable adding mode
        self.points_layer.mode = 'add'

        # Connect to data change events
        self.points_layer.events.data.connect(self._on_points_changed)

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Update title
        self._update_title()

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts to viewer."""

        @self.viewer.bind_key('s', overwrite=True)
        def save_crosses(viewer):
            try:
                print("S key pressed - attempting save...")
                self.save_crosses()
            except Exception as e:
                print(f"ERROR saving: {e}")
                import traceback
                traceback.print_exc()

        @self.viewer.bind_key('u', overwrite=True)
        def undo_last(viewer):
            self.undo_last()

        @self.viewer.bind_key('c', overwrite=True)
        def clear_all(viewer):
            self.clear_all()

        @self.viewer.bind_key('q', overwrite=True)
        def quit_and_save(viewer):
            self.quit_and_save()

    def _on_points_changed(self, event):
        """Handle points data changes."""
        self._update_title()

    def _update_title(self):
        """Update window title with cross count and status."""
        n_crosses = len(self.points_layer.data)
        status = "READY" if n_crosses >= 3 else f"Need {3 - n_crosses} more"
        self.viewer.title = f"LMD Cross Placement - {n_crosses} crosses ({status}) | S=Save U=Undo C=Clear Q=Quit"

    def save_crosses(self) -> bool:
        """Save crosses to JSON file."""
        points = self.points_layer.data

        if len(points) < 3:
            show_warning(f"Need at least 3 crosses! Currently have {len(points)}.")
            return False

        crosses = []
        for i, pt in enumerate(points):
            # Napari uses [y, x] format
            y_px, x_px = pt[0], pt[1]
            crosses.append({
                'id': i + 1,
                'x_px': float(x_px),
                'y_px': float(y_px),
                'x_um': float(x_px * self.pixel_size_um),
                'y_um': float(y_px * self.pixel_size_um)
            })

        data = {
            'image_width_px': self.image_width_px,
            'image_height_px': self.image_height_px,
            'pixel_size_um': self.pixel_size_um,
            'crosses': crosses
        }

        with open(self.output_path, 'w') as f:
            json.dump(data, f)

        show_info(f"Saved {len(crosses)} crosses to: {self.output_path}")
        print(f"Saved {len(crosses)} reference crosses to: {self.output_path}")
        return True

    def undo_last(self):
        """Remove the last placed cross."""
        if len(self.points_layer.data) > 0:
            self.points_layer.data = self.points_layer.data[:-1]
            show_info("Removed last cross")
        else:
            show_warning("No crosses to remove")

    def clear_all(self):
        """Remove all crosses."""
        if len(self.points_layer.data) > 0:
            n = len(self.points_layer.data)
            self.points_layer.data = np.empty((0, 2))
            show_info(f"Cleared {n} crosses")
        else:
            show_warning("No crosses to clear")

    def quit_and_save(self):
        """Save crosses and close viewer."""
        if len(self.points_layer.data) >= 3:
            if self.save_crosses():
                self.viewer.close()
        else:
            show_warning(
                f"Cannot quit: Need at least 3 crosses (have {len(self.points_layer.data)}). "
                "Use Ctrl+Q to force quit without saving."
            )


def main():
    parser = argparse.ArgumentParser(
        description='Interactive reference cross placement in Napari for LMD export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Keyboard Shortcuts:
  S - Save crosses to JSON
  U - Undo last cross
  C - Clear all crosses
  Q - Quit and save (requires >= 3 crosses)

Examples:
  # Basic usage
  python napari_place_crosses.py /path/to/image.zarr

  # With custom output path
  python napari_place_crosses.py /path/to/image.zarr --output my_crosses.json

  # With detection overlay for context
  python napari_place_crosses.py /path/to/image.zarr \\
      --detections /path/to/nmj_detections.json

  # With explicit pixel size
  python napari_place_crosses.py /path/to/image.zarr --pixel-size 0.22
'''
    )

    parser.add_argument(
        'zarr_path',
        type=str,
        help='Path to OME-Zarr pyramid file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='reference_crosses.json',
        help='Output path for crosses JSON (default: reference_crosses.json)'
    )

    parser.add_argument(
        '--detections', '-d',
        type=str,
        default=None,
        help='Optional: Path to detections JSON to show as overlay (green dots)'
    )

    parser.add_argument(
        '--pixel-size',
        type=float,
        default=None,
        help='Pixel size in micrometers (auto-detected from Zarr metadata if not set)'
    )

    parser.add_argument(
        '--load-existing',
        type=str,
        default=None,
        help='Load existing crosses JSON to continue editing'
    )

    args = parser.parse_args()

    # Validate input path
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        print(f"ERROR: Zarr path does not exist: {zarr_path}")
        sys.exit(1)

    print(f"Loading OME-Zarr from: {zarr_path}")

    # Load the image
    try:
        data, pixel_size_um, image_shape = load_ome_zarr(str(zarr_path))
    except Exception as e:
        print(f"ERROR loading OME-Zarr: {e}")
        sys.exit(1)

    # Override pixel size if provided
    if args.pixel_size is not None:
        pixel_size_um = args.pixel_size

    image_height_px, image_width_px = image_shape

    print(f"  Image size: {image_width_px} x {image_height_px} px")
    print(f"  Pixel size: {pixel_size_um} um/px")
    print(f"  Pyramid levels: {len(data)}")

    # Create viewer
    viewer = napari.Viewer(title="LMD Cross Placement - Loading...")

    # Add image layers (pyramid)
    viewer.add_image(
        data,
        name=zarr_path.stem,
        multiscale=True,
    )

    # Add detections overlay if provided
    if args.detections:
        det_path = Path(args.detections)
        if det_path.exists():
            print(f"Loading detections from: {det_path}")
            try:
                det_coords = load_detections(str(det_path))
                if det_coords:
                    viewer.add_points(
                        np.array(det_coords),
                        name='Detections',
                        symbol='disc',
                        size=20,
                        border_color='lime',
                        face_color='lime',
                        opacity=0.6,
                    )
                    print(f"  Added {len(det_coords)} detection points")
                else:
                    print("  Warning: No detection coordinates found")
            except Exception as e:
                print(f"  Warning: Could not load detections: {e}")
        else:
            print(f"Warning: Detections file not found: {det_path}")

    # Create cross placer
    cross_placer = CrossPlacer(
        viewer=viewer,
        pixel_size_um=pixel_size_um,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        output_path=args.output
    )

    # Load existing crosses if provided
    if args.load_existing:
        existing_path = Path(args.load_existing)
        if existing_path.exists():
            print(f"Loading existing crosses from: {existing_path}")
            try:
                with open(existing_path, 'r') as f:
                    existing_data = json.load(f)
                if 'crosses' in existing_data:
                    existing_points = []
                    for c in existing_data['crosses']:
                        # Convert to [y, x] for Napari
                        existing_points.append([c['y_px'], c['x_px']])
                    if existing_points:
                        cross_placer.points_layer.data = np.array(existing_points)
                        print(f"  Loaded {len(existing_points)} existing crosses")
            except Exception as e:
                print(f"  Warning: Could not load existing crosses: {e}")

    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("  1. Click on the image to place reference crosses")
    print("  2. Crosses should be at identifiable landmarks")
    print("  3. Place at least 3 crosses (4 recommended)")
    print("  4. Press S to save, Q to quit and save")
    print("")
    print("KEYBOARD SHORTCUTS:")
    print("  S - Save crosses to JSON")
    print("  U - Undo last cross")
    print("  C - Clear all crosses")
    print("  Q - Quit and save")
    print("="*60 + "\n")

    # Run Napari
    napari.run()


if __name__ == '__main__':
    main()
