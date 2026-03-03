#!/usr/bin/env python3
"""
Napari-based interactive reference cross placement for LMD export.

Supports both CZI (direct read_mosaic) and OME-Zarr pyramid inputs.
Places exactly 3 RGB-coded reference crosses with zoom-adaptive filled rectangles.

Features:
- CZI mode: aicspylibczi read_mosaic with scale_factor for fast reduced-res loading
- OME-Zarr mode: lazy dask pyramid loading (best for very large slides)
- Display transforms: --flip-horizontal, --rotate-cw-90 for LMD7 orientation
- RGB keybind crosses: R=red, G=green, B=blue with auto-advance
- Auto-save when all 3 placed, auto-load existing crosses on startup
- --fresh flag to ignore existing crosses
- Contour overlay from detection JSON
- Batch mode: --czi-dir + --slides for multiple slides

Keyboard Shortcuts:
    R / G / B - Select cross color (auto-advances to next unplaced)
    Space / P - Place cross at cursor position
    S         - Save crosses to JSON
    U         - Undo last cross
    C         - Clear all crosses
    Q         - Save + quit

Usage:
    # CZI (fast, reduced resolution)
    python napari_place_crosses.py -i /path/to/slide.czi --channel 0

    # OME-Zarr (lazy pyramids)
    python napari_place_crosses.py -i /path/to/image.zarr

    # With LMD7 display transforms
    python napari_place_crosses.py -i slide.czi --flip-horizontal --rotate-cw-90

    # Batch mode
    python napari_place_crosses.py --czi-dir /data/slides --slides A B C --output-dir crosses/

Author: xldvp_seg pipeline
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import napari
    from napari.utils.notifications import show_info, show_warning
except ImportError:
    print("ERROR: napari not installed. Install with: pip install napari[all]")
    sys.exit(1)


# Cross colors: RGB order
CROSS_COLORS = [
    ('red', [1, 0, 0, 1]),
    ('green', [0, 1, 0, 1]),
    ('blue', [0, 0.4, 1, 1]),
]
CROSS_NAMES = ['Red', 'Green', 'Blue']
THICKNESS_FRAC = 0.012  # Cross arm thickness as fraction of view extent


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_czi_image(czi_path, channel=0, scale_factor=8):
    """Load a single channel from CZI at reduced resolution via read_mosaic.

    Args:
        czi_path: Path to CZI file
        channel: Channel index to display
        scale_factor: Downsampling factor (8 = 1/8 resolution)

    Returns:
        (image_2d, pixel_size_um, full_res_shape) where full_res_shape is (H, W)
    """
    from aicspylibczi import CziFile

    czi = CziFile(str(czi_path))

    # Get pixel size
    pixel_size_um = None
    try:
        scaling = czi.meta.find('.//Scaling/Items/Distance[@Id="X"]/Value')
        if scaling is not None:
            pixel_size_um = float(scaling.text) * 1e6
    except Exception:
        pass
    if pixel_size_um is None:
        print("  WARNING: Could not read pixel size from CZI metadata. Use --pixel-size.")

    # Get full-res bounding box
    try:
        bbox = czi.get_mosaic_scene_bounding_box(index=0)
    except TypeError:
        bbox = czi.get_mosaic_bounding_box()

    full_h, full_w = bbox.h, bbox.w

    print(f"  Full resolution: {full_w} x {full_h}")
    print(f"  Loading at 1/{scale_factor}x ({full_w // scale_factor} x {full_h // scale_factor})...")

    # Read at reduced resolution
    data = czi.read_mosaic(
        region=(bbox.x, bbox.y, bbox.w, bbox.h),
        scale_factor=1.0 / scale_factor,
        C=channel,
    )
    img = np.squeeze(data)

    # Handle multi-dimensional output
    if img.ndim == 3:
        img = img[0] if img.shape[0] == 1 else img[..., 0]

    print(f"  Loaded shape: {img.shape}, dtype={img.dtype}")

    return img, pixel_size_um, (full_h, full_w)


def load_ome_zarr_image(zarr_path):
    """Load OME-Zarr pyramid for Napari viewing.

    Returns:
        (data_list, pixel_size_um, full_res_shape) where data_list is list of
        dask arrays (pyramid levels) and full_res_shape is (H, W)
    """
    import zarr as zarr_lib
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    store = parse_url(Path(zarr_path), mode="r")
    if store is None:
        raise ValueError(f"Could not parse OME-Zarr: {zarr_path}")

    reader = Reader(store)
    nodes = list(reader())
    if not nodes:
        raise ValueError(f"No data in OME-Zarr: {zarr_path}")

    data = nodes[0].data  # list of dask arrays (pyramid)

    # Read pixel size from metadata
    pixel_size_um = None
    try:
        root = zarr_lib.open(zarr_path, mode='r')
        if 'multiscales' in root.attrs:
            ms = root.attrs['multiscales'][0]
            for ds in ms.get('datasets', []):
                for t in ds.get('coordinateTransformations', []):
                    if t.get('type') == 'scale':
                        scale = t.get('scale', [])
                        if len(scale) >= 2:
                            pixel_size_um = scale[-1]
                            break
                if pixel_size_um is not None:
                    break
    except Exception:
        pass

    # Get full-res shape
    full_res = data[0]
    full_res_shape = full_res.shape[-2:]  # (H, W)

    return data, pixel_size_um, full_res_shape


# ---------------------------------------------------------------------------
# Display transforms
# ---------------------------------------------------------------------------

def apply_display_transforms(img, flip_horizontal=False, rotate_cw_90=False):
    """Apply display transforms to image array.

    Args:
        img: 2D numpy array (H, W) or list of dask arrays (pyramid)
        flip_horizontal: Mirror left-right
        rotate_cw_90: 90° clockwise rotation

    Returns:
        Transformed image (same type as input)
    """
    if isinstance(img, list):
        # Pyramid: transform each level
        result = []
        for level in img:
            arr = level
            if flip_horizontal:
                arr = arr[..., ::-1]
            if rotate_cw_90:
                # For dask arrays with shape (..., Y, X), rotate last 2 dims
                # np.rot90 k=-1 means CW 90°
                import dask.array as da
                if hasattr(arr, 'dask'):
                    arr = da.rot90(arr, k=-1, axes=(-2, -1))
                else:
                    arr = np.rot90(arr, k=-1, axes=(-2, -1))
            result.append(arr)
        return result
    else:
        # Single array
        if flip_horizontal:
            img = np.fliplr(img)
        if rotate_cw_90:
            img = np.rot90(img, k=-1)
        return img


def display_to_slide_coords(display_y, display_x, image_shape,
                            flip_horizontal=False, rotate_cw_90=False,
                            scale_factor=1):
    """Convert display (Napari) coordinates back to full-resolution slide pixels.

    Applies inverse transforms in reverse order, then scales up.

    Args:
        display_y, display_x: Coordinates in Napari display space
        image_shape: (H, W) of the DISPLAYED image (after transforms)
        flip_horizontal: Whether display was flipped
        rotate_cw_90: Whether display was rotated CW 90°
        scale_factor: Downsampling factor used for CZI loading

    Returns:
        (slide_x, slide_y) in full-resolution pixel coordinates [x, y] format
    """
    y, x = display_y, display_x
    disp_h, disp_w = image_shape

    # Undo transforms in reverse order
    if rotate_cw_90:
        # Inverse of CW 90° = CCW 90°
        # CW 90° maps (y, x) -> (x, H-1-y) in the original frame
        # So inverse: (y, x) in rotated -> (disp_w - 1 - x, y) in pre-rotation
        new_y = disp_w - 1 - x
        new_x = y
        y, x = new_y, new_x
        # After undoing rotation, the image shape is (disp_w, disp_h)
        disp_h, disp_w = disp_w, disp_h

    if flip_horizontal:
        x = disp_w - 1 - x

    # Scale up to full resolution
    slide_x = x * scale_factor
    slide_y = y * scale_factor

    return float(slide_x), float(slide_y)


def slide_to_display_coords(slide_x, slide_y, image_shape,
                            flip_horizontal=False, rotate_cw_90=False,
                            scale_factor=1):
    """Convert full-resolution slide pixel coordinates to display (Napari) coords.

    Args:
        slide_x, slide_y: Full-resolution slide pixel coordinates [x, y]
        image_shape: (H, W) of the image BEFORE transforms (at reduced resolution)
        flip_horizontal: Whether display is flipped
        rotate_cw_90: Whether display is rotated CW 90°
        scale_factor: Downsampling factor used for CZI loading

    Returns:
        (display_y, display_x) in Napari coordinate space [row, col]
    """
    # Scale down
    x = slide_x / scale_factor
    y = slide_y / scale_factor
    pre_h, pre_w = image_shape

    # Apply transforms in order
    if flip_horizontal:
        x = pre_w - 1 - x

    if rotate_cw_90:
        # CW 90°: (y, x) -> (x, pre_h - 1 - y)
        new_y = x
        new_x = pre_h - 1 - y
        y, x = new_y, new_x

    return float(y), float(x)


# ---------------------------------------------------------------------------
# Cross placer
# ---------------------------------------------------------------------------

class RGBCrossPlacer:
    """Interactive RGB cross placement manager for Napari.

    Places exactly 3 crosses (Red, Green, Blue) with zoom-adaptive
    filled rectangle rendering.
    """

    def __init__(self, viewer, pixel_size_um, image_shape_display,
                 image_shape_pre_transform, full_res_shape, output_path,
                 flip_horizontal=False, rotate_cw_90=False, scale_factor=1,
                 is_zarr=False):
        self.viewer = viewer
        self.pixel_size_um = pixel_size_um
        self.full_res_shape = full_res_shape  # (H, W) at full resolution
        self.image_shape_display = image_shape_display  # (H, W) after transforms
        self.image_shape_pre_transform = image_shape_pre_transform
        self.output_path = output_path
        self.flip_horizontal = flip_horizontal
        self.rotate_cw_90 = rotate_cw_90
        self.scale_factor = scale_factor if not is_zarr else 1
        self.is_zarr = is_zarr

        # 3 crosses: None = not placed, (display_y, display_x) = placed
        self.crosses = [None, None, None]
        self.active_idx = 0  # Which cross to place next (0=R, 1=G, 2=B)

        # Shapes layer for cross rendering
        self.shapes_layer = viewer.add_shapes(
            name='Reference Crosses',
            edge_width=0,
            face_color='red',
        )

        self._bind_shortcuts()
        self._update_title()

        # Re-render on zoom
        viewer.camera.events.zoom.connect(self._on_zoom)

    def _bind_shortcuts(self):
        @self.viewer.bind_key('r', overwrite=True)
        def select_red(viewer):
            self.active_idx = 0
            self._update_title()
            show_info("Active: RED cross")

        @self.viewer.bind_key('g', overwrite=True)
        def select_green(viewer):
            self.active_idx = 1
            self._update_title()
            show_info("Active: GREEN cross")

        @self.viewer.bind_key('b', overwrite=True)
        def select_blue(viewer):
            self.active_idx = 2
            self._update_title()
            show_info("Active: BLUE cross")

        @self.viewer.bind_key('Space', overwrite=True)
        def place_cross(viewer):
            self._place_at_cursor()

        @self.viewer.bind_key('p', overwrite=True)
        def place_cross_p(viewer):
            self._place_at_cursor()

        @self.viewer.bind_key('s', overwrite=True)
        def save(viewer):
            self.save_crosses()

        @self.viewer.bind_key('u', overwrite=True)
        def undo(viewer):
            self.undo_last()

        @self.viewer.bind_key('c', overwrite=True)
        def clear(viewer):
            self.clear_all()

        @self.viewer.bind_key('q', overwrite=True)
        def quit_save(viewer):
            if self._all_placed():
                self.save_crosses()
                viewer.close()
            else:
                show_warning(f"Need all 3 crosses placed. Missing: {self._missing_str()}")

    def _place_at_cursor(self):
        """Place the active cross at the current cursor position."""
        # Get cursor position from the viewer
        cursor_pos = self.viewer.cursor.position
        if cursor_pos is None or len(cursor_pos) < 2:
            show_warning("Move cursor over the image first")
            return

        display_y, display_x = cursor_pos[-2], cursor_pos[-1]
        self.crosses[self.active_idx] = (display_y, display_x)

        self._render_crosses()

        color_name = CROSS_NAMES[self.active_idx]
        show_info(f"Placed {color_name} cross at ({display_x:.0f}, {display_y:.0f})")

        # Auto-advance to next unplaced
        for i in range(3):
            next_idx = (self.active_idx + 1 + i) % 3
            if self.crosses[next_idx] is None:
                self.active_idx = next_idx
                break

        self._update_title()

        # Auto-save when all 3 placed
        if self._all_placed():
            self.save_crosses()
            show_info("All 3 crosses placed! Auto-saved. Press Q to quit.")

    def _all_placed(self):
        return all(c is not None for c in self.crosses)

    def _missing_str(self):
        missing = [CROSS_NAMES[i] for i in range(3) if self.crosses[i] is None]
        return ', '.join(missing)

    def _update_title(self):
        placed = sum(1 for c in self.crosses if c is not None)
        active = CROSS_NAMES[self.active_idx]
        status_parts = []
        for i, name in enumerate(CROSS_NAMES):
            mark = "+" if self.crosses[i] is not None else "-"
            status_parts.append(f"{name[0]}:{mark}")
        status = ' '.join(status_parts)
        self.viewer.title = (
            f"LMD Crosses [{status}] | Active: {active} | "
            f"R/G/B=select Space=place S=save U=undo Q=quit"
        )

    def _on_zoom(self, event=None):
        self._render_crosses()

    def _render_crosses(self):
        """Re-render all placed crosses as zoom-adaptive filled rectangles."""
        shapes = []
        colors = []

        # Calculate cross size based on current view extent
        camera = self.viewer.camera
        # View extent in data coordinates
        extent = max(self.image_shape_display) / max(camera.zoom, 0.01)
        thickness = extent * THICKNESS_FRAC
        arm_length = thickness * 6

        for i, pos in enumerate(self.crosses):
            if pos is None:
                continue

            cy, cx = pos
            color = CROSS_COLORS[i][1]

            # Horizontal bar
            h_rect = np.array([
                [cy - thickness / 2, cx - arm_length],
                [cy - thickness / 2, cx + arm_length],
                [cy + thickness / 2, cx + arm_length],
                [cy + thickness / 2, cx - arm_length],
            ])
            shapes.append(h_rect)
            colors.append(color)

            # Vertical bar
            v_rect = np.array([
                [cy - arm_length, cx - thickness / 2],
                [cy - arm_length, cx + thickness / 2],
                [cy + arm_length, cx + thickness / 2],
                [cy + arm_length, cx - thickness / 2],
            ])
            shapes.append(v_rect)
            colors.append(color)

        if shapes:
            self.shapes_layer.data = shapes
            self.shapes_layer.face_color = colors
            self.shapes_layer.edge_width = 0
        else:
            self.shapes_layer.data = []

    def save_crosses(self) -> bool:
        """Save crosses to JSON with full-resolution slide coordinates."""
        if not self._all_placed():
            show_warning(f"Need all 3 crosses. Missing: {self._missing_str()}")
            return False

        full_h, full_w = self.full_res_shape
        crosses_list = []

        for i, (dy, dx) in enumerate(self.crosses):
            slide_x, slide_y = display_to_slide_coords(
                dy, dx, self.image_shape_display,
                flip_horizontal=self.flip_horizontal,
                rotate_cw_90=self.rotate_cw_90,
                scale_factor=self.scale_factor,
            )
            crosses_list.append({
                'id': i + 1,
                'color': CROSS_NAMES[i].lower(),
                'x_px': slide_x,
                'y_px': slide_y,
                'x_um': slide_x * self.pixel_size_um,
                'y_um': slide_y * self.pixel_size_um,
            })

        data = {
            'image_width_px': full_w,
            'image_height_px': full_h,
            'pixel_size_um': self.pixel_size_um,
            'display_transform': {
                'flip_horizontal': self.flip_horizontal,
                'rotate_cw_90': self.rotate_cw_90,
            },
            'crosses': crosses_list,
        }

        # Atomic write: temp file + rename to prevent corruption on crash
        import tempfile, os
        dir_name = os.path.dirname(self.output_path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.json.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f)
            os.replace(tmp_path, self.output_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        print(f"Saved 3 reference crosses to: {self.output_path}")
        show_info(f"Saved to: {self.output_path}")
        return True

    def undo_last(self):
        """Remove the most recently placed cross."""
        # Find last placed
        for i in reversed(range(3)):
            if self.crosses[i] is not None:
                name = CROSS_NAMES[i]
                self.crosses[i] = None
                self.active_idx = i
                self._render_crosses()
                self._update_title()
                show_info(f"Removed {name} cross")
                return
        show_warning("No crosses to remove")

    def clear_all(self):
        """Remove all crosses."""
        if any(c is not None for c in self.crosses):
            self.crosses = [None, None, None]
            self.active_idx = 0
            self._render_crosses()
            self._update_title()
            show_info("Cleared all crosses")
        else:
            show_warning("No crosses to clear")

    def load_existing(self, crosses_data):
        """Load existing crosses from JSON data."""
        crosses = crosses_data.get('crosses', [])
        transform = crosses_data.get('display_transform', {})

        for c in crosses[:3]:
            idx = c.get('id', 1) - 1
            if idx < 0 or idx >= 3:
                continue

            slide_x = c['x_px']
            slide_y = c['y_px']

            dy, dx = slide_to_display_coords(
                slide_x, slide_y, self.image_shape_pre_transform,
                flip_horizontal=self.flip_horizontal,
                rotate_cw_90=self.rotate_cw_90,
                scale_factor=self.scale_factor,
            )
            self.crosses[idx] = (dy, dx)

        # Advance to first unplaced
        for i in range(3):
            if self.crosses[i] is None:
                self.active_idx = i
                break

        self._render_crosses()
        self._update_title()
        placed = sum(1 for c in self.crosses if c is not None)
        print(f"  Loaded {placed} existing crosses")


# ---------------------------------------------------------------------------
# Contour overlay
# ---------------------------------------------------------------------------

def load_contour_overlay(viewer, contours_path, image_shape_pre_transform,
                         flip_horizontal=False, rotate_cw_90=False,
                         scale_factor=1, slide_filter=None):
    """Load detection contours as a Napari Shapes overlay.

    Supports two formats:
      - Pipeline format: list of dicts with 'outer_contour_global' in [x, y] px
      - BM/overlay format: dict of {slide: [{contour_yx: [[y,x], ...]}]}

    Args:
        slide_filter: Optional slide name to select from per-slide dict format.
    """
    with open(contours_path) as f:
        data = json.load(f)

    # Detect format: per-slide dict of lists (BM overlay format)
    # vs pipeline detection format (list or dict with 'detections' key)
    detections = []
    is_per_slide = False
    if isinstance(data, dict):
        # Check if values are lists of dicts with contour_yx (BM format)
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, list) and first_val and isinstance(first_val[0], dict) \
                and 'contour_yx' in first_val[0]:
            is_per_slide = True
            if slide_filter:
                # Match by exact name or substring
                for key, entries in data.items():
                    if slide_filter in key or key in slide_filter:
                        detections = entries
                        print(f"  Contours: matched slide '{key}' ({len(entries)} contours)")
                        break
                if not detections:
                    print(f"  Warning: slide '{slide_filter}' not found in contours file")
            else:
                # Flatten all slides
                for entries in data.values():
                    detections.extend(entries)
        else:
            detections = data.get('detections', data.get('shapes', list(data.values())))
            if isinstance(detections, dict):
                detections = list(detections.values())
    else:
        detections = data

    shapes = []
    for det in detections:
        # BM overlay format: contour_yx in [y, x] pixel coords
        contour_yx = det.get('contour_yx')
        if contour_yx is not None and len(contour_yx) >= 3:
            pts_yx = np.array(contour_yx, dtype=np.float64)
            # Already [row, col] — just apply scale
            if scale_factor != 1:
                pts_yx = pts_yx / scale_factor
            shapes.append(pts_yx)
            continue

        # Pipeline format: outer_contour_global in [x, y] pixel coords
        contour_px = det.get('outer_contour_global')
        if contour_px is not None and len(contour_px) >= 3:
            pts = np.array(contour_px, dtype=np.float64)
        elif det.get('contour_um') is not None or det.get('contour_dilated_um') is not None:
            # um-valued contours can't be displayed without pixel_size conversion
            continue
        else:
            continue

        # Convert [x, y] to display coords [row, col]
        display_pts = []
        for pt in pts:
            dy, dx = slide_to_display_coords(
                pt[0], pt[1], image_shape_pre_transform,
                flip_horizontal=flip_horizontal,
                rotate_cw_90=rotate_cw_90,
                scale_factor=scale_factor,
            )
            display_pts.append([dy, dx])

        shapes.append(np.array(display_pts))

    if shapes:
        viewer.add_shapes(
            shapes,
            shape_type='polygon',
            edge_color='lime',
            edge_width=1,
            face_color=[0, 0, 0, 0],
            name='Contours',
            opacity=0.6,
        )
        print(f"  Overlay: {len(shapes)} contours")
    else:
        print("  Warning: no valid contours found for overlay")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_slide(args, input_path=None, output_path=None):
    """Run cross placement for a single slide."""
    input_path = input_path or args.input
    output_path = output_path or args.output

    if input_path is None:
        print("ERROR: --input / -i is required")
        sys.exit(1)

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        sys.exit(1)

    output_path = Path(output_path or 'reference_crosses.json')

    # Auto-detect format
    is_zarr = input_path.suffix == '.zarr' or (input_path / '.zattrs').exists()
    is_czi = input_path.suffix.lower() == '.czi'

    if not is_zarr and not is_czi:
        print(f"ERROR: Unsupported format: {input_path.suffix}. Use .czi or .zarr")
        sys.exit(1)

    flip_h = getattr(args, 'flip_horizontal', False)
    rotate = getattr(args, 'rotate_cw_90', False)
    scale = getattr(args, 'scale_factor', 8)
    channel = getattr(args, 'channel', 0)
    fresh = getattr(args, 'fresh', False)

    # Load image
    if is_czi:
        print(f"Loading CZI: {input_path}")
        img, pixel_size_um, full_res_shape = load_czi_image(
            input_path, channel=channel, scale_factor=scale
        )
        pre_transform_shape = img.shape[:2]  # (H, W) before transforms
        img = apply_display_transforms(img, flip_h, rotate)
        display_shape = img.shape[:2]
        multiscale = False
    else:
        print(f"Loading OME-Zarr: {input_path}")
        data, pixel_size_um, full_res_shape = load_ome_zarr_image(str(input_path))
        pre_transform_shape = full_res_shape
        data = apply_display_transforms(data, flip_h, rotate)
        if rotate:
            display_shape = (full_res_shape[1], full_res_shape[0])
        else:
            display_shape = full_res_shape
        img = data
        multiscale = True
        scale = 1  # zarr handles its own resolution

    # Override pixel size from CLI
    if getattr(args, 'pixel_size', None):
        pixel_size_um = args.pixel_size

    if pixel_size_um is None:
        print("ERROR: Could not determine pixel size. Use --pixel-size.")
        sys.exit(1)

    print(f"  Pixel size: {pixel_size_um:.4f} um/px")
    if flip_h or rotate:
        transforms = []
        if flip_h:
            transforms.append("flip-H")
        if rotate:
            transforms.append("rot-CW-90")
        print(f"  Display transforms: {', '.join(transforms)}")

    # Create viewer
    viewer = napari.Viewer(title="LMD Cross Placement - Loading...")

    if multiscale:
        viewer.add_image(img, name=input_path.stem, multiscale=True)
    else:
        viewer.add_image(img, name=input_path.stem)

    # Create cross placer
    placer = RGBCrossPlacer(
        viewer=viewer,
        pixel_size_um=pixel_size_um,
        image_shape_display=display_shape,
        image_shape_pre_transform=pre_transform_shape,
        full_res_shape=full_res_shape,
        output_path=str(output_path),
        flip_horizontal=flip_h,
        rotate_cw_90=rotate,
        scale_factor=scale,
        is_zarr=is_zarr,
    )

    # Auto-load existing crosses (unless --fresh)
    if not fresh and output_path.exists():
        print(f"  Auto-loading existing crosses from: {output_path}")
        try:
            with open(output_path) as f:
                existing = json.load(f)
            placer.load_existing(existing)
        except Exception as e:
            print(f"  Warning: Could not load existing crosses: {e}")

    # Contour overlay
    contours_path = getattr(args, 'contours', None)
    if contours_path:
        contours_path = Path(contours_path)
        if contours_path.exists():
            print(f"Loading contour overlay: {contours_path}")
            try:
                load_contour_overlay(
                    viewer, str(contours_path), pre_transform_shape,
                    flip_horizontal=flip_h, rotate_cw_90=rotate,
                    scale_factor=scale,
                    slide_filter=getattr(args, 'slide', None),
                )
            except Exception as e:
                print(f"  Warning: Could not load contours: {e}")

    print("\n" + "=" * 60)
    print("CROSS PLACEMENT (3 crosses: Red, Green, Blue)")
    print("=" * 60)
    print("  R/G/B  = select cross color")
    print("  Space  = place cross at cursor")
    print("  S      = save    U = undo    C = clear    Q = save+quit")
    print("=" * 60 + "\n")

    napari.run()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive RGB cross placement for LMD export (CZI or OME-Zarr)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Keyboard:  R/G/B=select  Space=place  S=save  U=undo  C=clear  Q=save+quit

Examples:
  # CZI with LMD7 transforms
  python napari_place_crosses.py -i slide.czi --flip-horizontal --rotate-cw-90

  # OME-Zarr
  python napari_place_crosses.py -i slide.zarr -o crosses.json

  # Batch mode
  python napari_place_crosses.py --czi-dir /data --slides A B --output-dir crosses/
''',
    )

    # Single-slide input
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='CZI or OME-Zarr path (auto-detect from extension)')
    # Backward compat: positional arg
    parser.add_argument('zarr_path', nargs='?', type=str, default=None,
                        help='[Deprecated] OME-Zarr path (use -i instead)')

    parser.add_argument('--output', '-o', type=str, default='reference_crosses.json',
                        help='Output crosses JSON path (default: reference_crosses.json)')

    # CZI options
    parser.add_argument('--channel', type=int, default=0,
                        help='CZI channel index for display (default: 0)')
    parser.add_argument('--scale-factor', type=int, default=8,
                        help='CZI downsampling factor (default: 8 = 1/8 resolution)')

    # Display transforms
    parser.add_argument('--flip-horizontal', action='store_true',
                        help='Mirror image horizontally (tissue-down LMD7 view)')
    parser.add_argument('--rotate-cw-90', action='store_true',
                        help='Rotate image 90° clockwise (LMD7 orientation)')

    # Overlays
    parser.add_argument('--contours', type=str, default=None,
                        help='Path to detection/contour JSON for overlay')
    parser.add_argument('--slide', type=str, default=None,
                        help='Slide name filter for per-slide contour files')

    # Pixel size override
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in um (auto-detected if not set)')

    # Auto-load control
    parser.add_argument('--fresh', action='store_true',
                        help="Don't auto-load existing crosses (start from scratch)")

    # Batch mode
    parser.add_argument('--czi-dir', type=str, default=None,
                        help='Batch mode: directory containing CZI files')
    parser.add_argument('--slides', nargs='+', default=None,
                        help='Batch mode: slide name prefixes to process')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Batch mode: output directory for per-slide crosses')

    # Backward compat
    parser.add_argument('--detections', '-d', type=str, default=None,
                        help='[Deprecated] Use --contours instead')
    parser.add_argument('--load-existing', type=str, default=None,
                        help='[Deprecated] Auto-load is now default; use --fresh to disable')

    args = parser.parse_args()

    # Handle backward compat: positional zarr_path -> --input
    if args.input is None and args.zarr_path:
        args.input = args.zarr_path

    # Handle deprecated --detections -> --contours
    if args.contours is None and args.detections:
        args.contours = args.detections

    # Batch mode
    if args.czi_dir and args.slides:
        czi_dir = Path(args.czi_dir)
        output_dir = Path(args.output_dir or 'crosses')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect slides to process
        slides_to_process = []
        for slide_name in args.slides:
            matches = list(czi_dir.glob(f"{slide_name}*.czi"))
            if not matches:
                print(f"WARNING: No CZI found for slide '{slide_name}' in {czi_dir}")
                continue

            czi_path = matches[0]
            out_path = output_dir / f"{czi_path.stem}_crosses.json"

            # Skip if already done (unless --fresh)
            if out_path.exists() and not args.fresh:
                print(f"Skipping {czi_path.stem} (crosses exist: {out_path})")
                continue

            slides_to_process.append((czi_path, out_path))

        # Process slides sequentially — each gets its own viewer
        # napari.run() starts the Qt event loop; it returns when the viewer
        # is closed. Creating a new Viewer + run() works across slides as
        # long as we don't call QApplication.quit().
        for i, (czi_path, out_path) in enumerate(slides_to_process):
            print(f"\n{'='*60}")
            print(f"Slide {i+1}/{len(slides_to_process)}: {czi_path.stem}")
            print(f"{'='*60}")

            run_single_slide(args, input_path=str(czi_path), output_path=str(out_path))
        return

    # Single-slide mode
    run_single_slide(args)


if __name__ == '__main__':
    main()
