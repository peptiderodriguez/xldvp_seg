#!/usr/bin/env python3
"""
Place 3 reference crosses on a CZI or OME-Zarr slide for LMD calibration.

Loads 2 resolution levels as numpy arrays and displays as napari multiscale.
scale=(base, base) makes world coords = full-res reference pixels, so cursor
position IS the saved coordinate — no coordinate conversion needed.

Keys:
  1/R = select cross 1 (red)
  2/G = select cross 2 (green)
  3/B = select cross 3 (cyan)
  Space/P = place at cursor
  S = save  |  U = undo  |  C = clear  |  Q = save+quit

All layers are locked (non-draggable). Rotate CW 90 is ON by default for LMD7.

Files needed (copy from server):
  - CZI slide files
  - lmd_replicates_full.json  (from select_mks_for_lmd.py — replicate/well assignments)
  - mk_contours_overlay.json  (from extract contours step — cell outlines for overlay)

Usage:
  # Single slide (LMD7 orientation: flip + rotate is default)
  python napari_place_crosses.py -i slide.czi --flip-horizontal

  # Batch with replicate overlay
  python napari_place_crosses.py --czi-dir /data --slides A B \
      --sampling-results lmd_replicates_full.json \
      --contours mk_contours_overlay.json \
      --flip-horizontal --pyramid-levels 2 8 --output-dir crosses/ --fresh

  # Without rotation (e.g. non-LMD viewing)
  python napari_place_crosses.py -i slide.czi --no-rotate-cw-90

  # OME-Zarr
  python napari_place_crosses.py -i slide.ome.zarr
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import napari
    from napari.utils.notifications import show_info, show_warning
except ImportError:
    sys.exit("pip install 'napari[all]'")

CROSS_COLORS = ['red', 'lime', 'cyan']
CROSS_LABELS = ['Cross 1 (RED)', 'Cross 2 (GREEN)', 'Cross 3 (CYAN)']
SCREEN_PX = 80        # cross arm = 80 screen pixels
THICKNESS_FRAC = 0.02  # bar thickness = 2% of arm length

REP_COLORS = [
    (0.12, 0.56, 1.0, 0.9),   # dodgerblue
    (1.0, 0.39, 0.28, 0.9),   # tomato
    (0.20, 0.80, 0.20, 0.9),  # limegreen
    (1.0, 0.84, 0.0, 0.9),    # gold
    (0.58, 0.44, 0.86, 0.9),  # mediumpurple
]


# ---------------------------------------------------------------------------
# Image loading — always returns (pyramid, full_w, full_h, pixel_size_um, ...)
# ---------------------------------------------------------------------------

def load_czi_pyramid(czi_path, channel=0, scale_factors=(2, 8),
                     flip_horizontal=False, rotate_cw_90=False):
    """Load CZI at 2+ scales as uint8 RGB numpy arrays.

    Transforms (fliplr, rot90) are baked into the data.
    With scale=(base, base), world coords = full-res CZI pixels (post-transform).

    Returns:
        pyramid: list of (H, W, 3) uint8 arrays
        full_w, full_h: CZI bounding box dimensions (pre-transform)
        pixel_size_um: from CZI metadata or None
    """
    from aicspylibczi import CziFile

    czi = CziFile(str(czi_path))
    try:
        bbox = czi.get_mosaic_scene_bounding_box(index=0)
    except TypeError:
        bbox = czi.get_mosaic_bounding_box()

    fw, fh = bbox.w, bbox.h
    x0, y0 = bbox.x, bbox.y

    pixel_size_um = None
    try:
        s = czi.meta.find('.//Scaling/Items/Distance[@Id="X"]/Value')
        if s is not None:
            pixel_size_um = float(s.text) * 1e6
    except Exception:
        pass

    print(f"  Full: {fw:,} x {fh:,} px, pixel_size={pixel_size_um}")

    pyramid = []
    for sf in scale_factors:
        print(f"  Loading 1/{sf}...", end=" ", flush=True)
        img = np.squeeze(czi.read_mosaic(
            region=(x0, y0, fw, fh), scale_factor=1.0 / sf, C=channel))

        # Ensure RGB uint8
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] != 3:
            img = np.moveaxis(img, 0, -1)

        if img.dtype != np.uint8:
            mx = img.max()
            if mx > 0:
                p99 = np.percentile(img[img > 0], 99.5) if np.any(img > 0) else mx
                img = np.clip(img.astype(np.float32) / p99 * 255, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        if flip_horizontal:
            img = np.ascontiguousarray(np.fliplr(img))
        if rotate_cw_90:
            img = np.ascontiguousarray(np.rot90(img, k=-1))

        pyramid.append(img)
        print(f"{img.shape} ({img.nbytes // 1_000_000} MB)")

    return pyramid, fw, fh, pixel_size_um


def load_zarr_pyramid(zarr_path, level_indices=(1, 3),
                      flip_horizontal=False, rotate_cw_90=False):
    """Load OME-Zarr pyramid levels as uint8 RGB numpy arrays.

    Transforms (fliplr, rot90) are applied if requested (on top of any baked-in).
    With scale=(2^first_level, 2^first_level), world coords = zarr level 0 pixels.

    Returns:
        pyramid, full_w, full_h, pixel_size_um, zarr_meta, base_scale
    """
    import zarr as zarr_lib

    root = zarr_lib.open(str(zarr_path), mode='r')

    avail = sorted([k for k in root.keys() if k.isdigit()], key=int)
    if not avail:
        raise ValueError(f"No pyramid levels in {zarr_path}")

    # Read metadata
    pixel_size_um = None
    zarr_meta = None
    if 'multiscales' in root.attrs:
        ms = root.attrs['multiscales'][0]
        zarr_meta = ms.get('metadata')
        for ds in ms.get('datasets', []):
            if ds.get('path') == '0':
                for t in ds.get('coordinateTransformations', []):
                    if t.get('type') == 'scale':
                        sc = t.get('scale', [])
                        if len(sc) >= 2:
                            pixel_size_um = sc[-1]
                break

    # Level 0 shape for reference
    level0_shape = root['0'].shape[-2:]
    full_h, full_w = level0_shape

    # Pick levels to load (fallback to available)
    to_load = []
    for li in level_indices:
        key = str(li)
        if key in root:
            to_load.append(li)
        elif avail:
            closest = min([int(k) for k in avail], key=lambda k: abs(k - li))
            if closest not in to_load:
                to_load.append(closest)

    if not to_load:
        to_load = [int(avail[0])]
    # Always include a coarser level for overview
    if len(to_load) < 2 and len(avail) > 1:
        coarsest = int(avail[-1])
        if coarsest not in to_load:
            to_load.append(coarsest)
    to_load = sorted(to_load)

    pyramid = []
    for li in to_load:
        print(f"  Loading level {li}...", end=" ", flush=True)
        arr = np.array(root[str(li)])

        # Channel handling
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.moveaxis(arr, 0, -1)  # (3,H,W) → (H,W,3)

        # To uint8 RGB
        if arr.ndim == 2:
            if arr.dtype != np.uint8:
                mx = arr.max()
                if mx > 0:
                    p99 = np.percentile(arr[arr > 0], 99.5) if np.any(arr > 0) else mx
                    arr = np.clip(arr.astype(np.float32) / p99 * 255, 0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 3 and arr.dtype != np.uint8:
            mx = arr.max()
            if mx > 0:
                arr = np.clip(arr.astype(np.float32) / mx * 255, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        if flip_horizontal:
            arr = np.ascontiguousarray(np.fliplr(arr))
        if rotate_cw_90:
            arr = np.ascontiguousarray(np.rot90(arr, k=-1))

        pyramid.append(arr)
        print(f"{arr.shape} ({arr.nbytes // 1_000_000} MB)")

    base_scale = 2 ** to_load[0]

    return pyramid, full_w, full_h, pixel_size_um, zarr_meta, base_scale


# ---------------------------------------------------------------------------
# Contour overlay
# ---------------------------------------------------------------------------

def transform_contour_coords(pts_yx, orig_h, orig_w, flip_h, rot90):
    """Transform contour [y, x] from native pixel coords to display world coords.

    Applies the same transforms that were applied to the image data.
    Must be called BEFORE dividing by world_scale.
    """
    y = pts_yx[:, 0].astype(np.float64)
    x = pts_yx[:, 1].astype(np.float64)

    if flip_h:
        x = orig_w - x
    if rot90:
        y_new = x.copy()
        x_new = orig_h - y
        y, x = y_new, x_new

    return np.column_stack([y, x])


GROUP_COLORS = [
    ('lime',       [0.0, 1.0, 0.0, 1.0]),
    ('hotpink',    [1.0, 0.2, 0.6, 1.0]),
    ('cyan',       [0.0, 1.0, 1.0, 1.0]),
    ('orange',     [1.0, 0.6, 0.0, 1.0]),
    ('yellow',     [1.0, 1.0, 0.0, 1.0]),
    ('magenta',    [1.0, 0.0, 1.0, 1.0]),
    ('dodgerblue', [0.1, 0.5, 1.0, 1.0]),
    ('white',      [1.0, 1.0, 1.0, 1.0]),
]


def _get_group_value(det, field):
    """Extract a grouping value from a detection dict.

    Checks top-level keys first, then features dict.
    """
    if field in det:
        return det[field]
    feat = det.get('features', {})
    if field in feat:
        return feat[field]
    return None


def _polygon_to_dashes(pts_yx, dash_len=30, gap_len=15):
    """Convert a closed polygon into dash segments (list of short paths).

    Works in world coords, so dash_len/gap_len are in world pixels.
    """
    # Close the polygon
    closed = np.vstack([pts_yx, pts_yx[0:1]])
    # Cumulative arc length
    diffs = np.diff(closed, axis=0)
    seg_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < dash_len:
        return [pts_yx]  # too small to dash, return as-is

    dashes = []
    cycle = dash_len + gap_len
    pos = 0.0
    while pos < total:
        d_start = pos
        d_end = min(pos + dash_len, total)
        # Interpolate points along the perimeter
        seg_pts = []
        for t in np.linspace(d_start, d_end, max(2, int((d_end - d_start) / 5) + 2)):
            idx = np.searchsorted(cum, t, side='right') - 1
            idx = min(idx, len(closed) - 2)
            frac = (t - cum[idx]) / seg_lens[idx] if seg_lens[idx] > 0 else 0
            pt = closed[idx] + frac * (closed[idx + 1] - closed[idx])
            seg_pts.append(pt)
        if len(seg_pts) >= 2:
            dashes.append(np.array(seg_pts))
        pos += cycle
    return dashes


def _parse_contour(det, orig_h, orig_w, contour_flip, contour_rot90, world_scale):
    """Extract and transform a contour from a detection dict. Returns pts_yx or None."""
    contour_yx = det.get('contour_yx')
    if contour_yx is not None and len(contour_yx) >= 3:
        pts = np.array(contour_yx, dtype=np.float64)
        pts = transform_contour_coords(pts, orig_h, orig_w, contour_flip, contour_rot90)
        return pts / world_scale

    contour_px = det.get('outer_contour_global')
    if contour_px is not None and len(contour_px) >= 3:
        pts_xy = np.array(contour_px, dtype=np.float64)
        pts_yx = pts_xy[:, ::-1]
        pts_yx = transform_contour_coords(pts_yx, orig_h, orig_w, contour_flip, contour_rot90)
        return pts_yx / world_scale

    return None


def load_contour_overlay(viewer, contours_path, slide_filter,
                         orig_h, orig_w, flip_h, rot90, world_scale,
                         zarr_meta=None, color_by=None):
    """Load contour polygons and display as napari Shapes.

    Args:
        orig_h, orig_w: Native image dimensions (CZI H, W) for coordinate transforms.
        flip_h, rot90: What transforms were applied to image (explicit + baked-in).
        world_scale: Divide contour native coords by this to get world coords.
            CZI mode: 1.0 (world = CZI native). Zarr mode: e.g. 2.0 (zarr = CZI/2).
        zarr_meta: Zarr metadata dict, used to detect baked-in transforms.
        color_by: Field name to group contours by (e.g. 'group', 'classification',
            'score_class'). Each unique value gets a distinct color layer.
    """
    with open(contours_path) as f:
        data = json.load(f)

    # Detect format
    detections = []
    if isinstance(data, dict):
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, list) and first_val and isinstance(first_val[0], dict) \
                and 'contour_yx' in first_val[0]:
            if slide_filter:
                for key, entries in data.items():
                    if slide_filter in key or key in slide_filter:
                        detections = entries
                        print(f"  Contours: matched '{key}' ({len(entries)})")
                        break
                if not detections:
                    print(f"  Warning: slide '{slide_filter}' not in contours file")
            else:
                for entries in data.values():
                    detections.extend(entries)
        else:
            detections = data.get('detections', data.get('shapes', []))
    else:
        detections = data

    if not detections:
        print("  Warning: no contours found")
        return

    # Detect baked-in transforms from zarr metadata
    contour_flip = flip_h
    contour_rot90 = rot90
    if zarr_meta is not None:
        desc = zarr_meta.get('description', '').lower()
        if 'fliplr' in desc:
            contour_flip = True
        if 'rot90' in desc:
            contour_rot90 = True

    if color_by:
        # Group contours by field value, keep per-contour labels
        groups = {}
        label_centroids = []
        label_texts = []
        label_colors = []
        for det in detections:
            pts = _parse_contour(det, orig_h, orig_w, contour_flip, contour_rot90, world_scale)
            if pts is None:
                continue
            val = _get_group_value(det, color_by)
            if val is None:
                val = 'unknown'
            key = str(val)
            groups.setdefault(key, []).append(pts)
            # Centroid for label
            centroid = pts.mean(axis=0)
            label_centroids.append(centroid)
            label_texts.append(key)

        if not groups:
            print("  Warning: no valid contours for overlay")
            return

        color_map = {}
        for i, (group_name, polys) in enumerate(sorted(groups.items())):
            color_name, color_rgba = GROUP_COLORS[i % len(GROUP_COLORS)]
            color_map[group_name] = color_rgba
            dashes = []
            for poly in polys:
                dashes.extend(_polygon_to_dashes(poly))
            lyr = viewer.add_shapes(
                dashes, shape_type='path',
                edge_color=color_rgba, edge_width=3,
                face_color=[0, 0, 0, 0],
                name=f'{group_name} ({len(polys)})', opacity=0.8,
            )
            lyr.mouse_pan = False
            lyr.mouse_zoom = False
            print(f"  {color_name:12s} {group_name}: {len(polys)} contours ({len(dashes)} dashes)")

        # Add text labels at contour centroids
        if label_centroids:
            pts_arr = np.array(label_centroids)
            per_point_colors = [color_map.get(t, [1, 1, 1, 1]) for t in label_texts]
            text_props = {
                'string': label_texts,
                'color': per_point_colors,
                'size': 10,
                'anchor': 'center',
            }
            lyr = viewer.add_points(
                pts_arr, size=0.01, face_color=[0, 0, 0, 0],
                text=text_props, name='Labels',
            )
            lyr.mouse_pan = False
            lyr.mouse_zoom = False
    else:
        # Single layer, all lime
        polys = []
        for det in detections:
            pts = _parse_contour(det, orig_h, orig_w, contour_flip, contour_rot90, world_scale)
            if pts is not None:
                polys.append(pts)

        if polys:
            dashes = []
            for poly in polys:
                dashes.extend(_polygon_to_dashes(poly))
            lyr = viewer.add_shapes(
                dashes, shape_type='path',
                edge_color='lime', edge_width=3,
                face_color=[0, 0, 0, 0],
                name='Contours', opacity=0.7,
            )
            lyr.mouse_pan = False
            lyr.mouse_zoom = False
            print(f"  Overlay: {len(polys)} contours ({len(dashes)} dashes)")
        else:
            print("  Warning: no valid contours for overlay")


# ---------------------------------------------------------------------------
# Sampling results overlay (replicate dots/contours)
# ---------------------------------------------------------------------------

def load_sampling_overlay(viewer, path, slide_name, image_width_px,
                          pixel_size_um, flip_h, rot90, world_scale,
                          orig_h=None, orig_w=None):
    """Overlay colored sized dots per replicate/well on the slide.

    Loads replicate JSON, filters by slide name, adds one add_points layer
    per replicate with per-cell sized discs.

    Args:
        path: Path to replicate JSON (lmd_replicates_full.json).
        slide_name: Key to look up in the JSON.
        image_width_px: CZI bounding box width for flip transforms.
        pixel_size_um: For converting area_um2 to pixel diameter.
        flip_h, rot90: Display transforms applied to image.
        world_scale: Divisor to convert native px to world coords.
        orig_h, orig_w: Native image dimensions for rot90 transform.
    """
    with open(path) as f:
        data = json.load(f)

    if slide_name not in data:
        # Try substring match
        matched = [k for k in data if slide_name in k or k in slide_name]
        if len(matched) == 1:
            slide_name = matched[0]
            print(f"  Sampling: matched slide key '{slide_name}'")
        elif matched:
            print(f"  Warning: ambiguous slide match: {matched}")
            return
        else:
            print(f"  Warning: slide '{slide_name}' not in sampling results")
            return

    slide_data = data[slide_name]
    replicates = slide_data.get('replicates', [])
    if not replicates:
        print(f"  Warning: no replicates for '{slide_name}'")
        return

    for i, rep in enumerate(replicates):
        cells = rep.get('cells', [])
        if not cells:
            continue

        pts = []
        sizes = []
        for c in cells:
            cx, cy = c['center_x'], c['center_y']

            # Apply same transforms as image
            if flip_h:
                cx = image_width_px - cx
            if rot90:
                # CW 90: (x, y) -> (y, orig_h - x) in napari [row, col]
                new_row = cx
                new_col = (orig_h or image_width_px) - cy
                pts.append((new_row / world_scale, new_col / world_scale))
            else:
                pts.append((cy / world_scale, cx / world_scale))

            # Diameter in full-res pixels from area_um2
            area_um2 = c.get('area_um2', 100)
            diam_px = 2 * (area_um2 / 3.14159) ** 0.5 / pixel_size_um
            sizes.append(diam_px / world_scale)

        color = REP_COLORS[i % len(REP_COLORS)]
        rep_num = rep.get('replicate', i + 1)
        well = rep.get('well', '?')

        lyr = viewer.add_points(
            np.array(pts), name=f"Rep{rep_num} ({well})",
            symbol='disc', size=np.array(sizes),
            face_color=np.array([color]),
            edge_color=np.array([color]), opacity=0.4,
        )
        lyr.mouse_pan = False
        lyr.mouse_zoom = False

    print(f"  Sampling overlay: {len(replicates)} replicates")


# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------

def run_single_slide(args, input_path=None, output_path=None):
    """Load image, place crosses, save."""
    input_path = Path(input_path or args.input)
    if not input_path.exists():
        sys.exit(f"Not found: {input_path}")

    slide_name = input_path.stem.replace('_rotated', '')
    output_path = Path(output_path or args.output or f"{slide_name}_crosses.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_zarr = input_path.suffix == '.zarr' or (input_path / '.zattrs').exists()
    is_czi = input_path.suffix.lower() == '.czi'
    if not is_zarr and not is_czi:
        sys.exit(f"Unsupported format: {input_path.suffix}")

    flip_h = getattr(args, 'flip_horizontal', False)
    rotate = getattr(args, 'rotate_cw_90', False)
    fresh = getattr(args, 'fresh', False)

    print(f"\n{'='*60}\n{slide_name}\n{'='*60}")

    zarr_meta = None

    if is_czi:
        # Use --pyramid-levels if specified, otherwise compute from --scale-factor
        pyramid_levels = getattr(args, 'pyramid_levels', None)
        if pyramid_levels:
            scale_factors = tuple(sorted(set(pyramid_levels)))
        else:
            sf = getattr(args, 'scale_factor', 2)
            scale_factors = tuple(sorted(set([sf, sf * 4])))
        pyramid, fw, fh, pxum = load_czi_pyramid(
            input_path, channel=getattr(args, 'channel', 0),
            scale_factors=scale_factors,
            flip_horizontal=flip_h, rotate_cw_90=rotate,
        )
        base_scale = scale_factors[0]
        # For contour overlay: CZI native dimensions, world_scale = 1
        contour_orig_h, contour_orig_w = fh, fw
        contour_world_scale = 1.0
    else:
        # Parse --level to pick which zarr levels to load
        level_str = getattr(args, 'level', '1/2') or '1/2'
        fraction_to_level = {'1/2': 1, '1/4': 2, '1/8': 3, '1/16': 4, '1': 0, '1/1': 0}
        if level_str in fraction_to_level:
            base_level = fraction_to_level[level_str]
        elif level_str.isdigit():
            base_level = int(level_str)
        else:
            sys.exit(f"Invalid --level '{level_str}'. Use: 1/2, 1/4, 1/8, or integer")

        # Load base level + a coarser one for overview
        coarse = min(base_level + 2, 4)
        level_indices = (base_level, coarse)

        pyramid, fw, fh, pxum, zarr_meta, base_scale = load_zarr_pyramid(
            input_path, level_indices=level_indices,
            flip_horizontal=flip_h, rotate_cw_90=rotate,
        )
        # For contour overlay: detect original CZI dims from zarr metadata
        contour_orig_h, contour_orig_w = fh, fw
        if zarr_meta:
            oh = zarr_meta.get('original_height')
            ow = zarr_meta.get('original_width')
            if oh and ow:
                contour_orig_h, contour_orig_w = oh, ow
        # Auto-detect contour scale (CZI native vs zarr level 0)
        contour_world_scale = 1.0
        if contour_orig_h > fh * 1.3 or contour_orig_w > fw * 1.3:
            ratio = max(contour_orig_h / fh, contour_orig_w / fw)
            contour_world_scale = round(ratio)
            print(f"  Contour scale: {contour_world_scale}x (CZI native → zarr level 0)")

    # Override pixel size from CLI
    if getattr(args, 'pixel_size', None):
        pxum = args.pixel_size
    if pxum is None:
        sys.exit("Could not determine pixel size. Use --pixel-size.")

    transforms = []
    if flip_h:
        transforms.append("flip-H")
    if rotate:
        transforms.append("rot-CW-90")
    print(f"  Pixel size: {pxum:.4f} um/px")
    if transforms:
        print(f"  Transforms: {', '.join(transforms)}")

    # ── Display ──────────────────────────────────────────────────
    viewer = napari.Viewer(title=slide_name)

    # scale=(base, base) → world coords = full-res reference pixels
    viewer.add_image(
        pyramid, name=slide_name, multiscale=True,
        contrast_limits=[0, 255], scale=(base_scale, base_scale),
    )
    print(f"  Display: {len(pyramid)} levels, base_scale={base_scale}")

    # ── Contour overlay ─────────────────────────────────────────
    contours_path = getattr(args, 'contours', None)
    if contours_path and Path(contours_path).exists():
        print(f"  Loading contours: {contours_path}")
        try:
            load_contour_overlay(
                viewer, str(contours_path),
                slide_filter=getattr(args, 'slide', None) or slide_name,
                orig_h=contour_orig_h, orig_w=contour_orig_w,
                flip_h=flip_h, rot90=rotate,
                world_scale=contour_world_scale,
                zarr_meta=zarr_meta,
                color_by=getattr(args, 'color_by', None),
            )
        except Exception as e:
            print(f"  Warning: contour load failed: {e}")

    # ── Sampling results overlay ────────────────────────────────
    sampling_path = getattr(args, 'sampling_results', None)
    if sampling_path and Path(sampling_path).exists():
        print(f"  Loading sampling results: {sampling_path}")
        try:
            slide_filter = getattr(args, 'slide', None) or slide_name
            # Sampling points (center_x/y) are in CZI native pixels — same
            # coordinate space as contours, so use same transform parameters.
            load_sampling_overlay(
                viewer, str(sampling_path),
                slide_name=slide_filter,
                image_width_px=contour_orig_w,
                pixel_size_um=pxum,
                flip_h=flip_h, rot90=rotate,
                world_scale=contour_world_scale,
                orig_h=contour_orig_h, orig_w=contour_orig_w,
            )
        except Exception as e:
            print(f"  Warning: sampling overlay failed: {e}")

    # ── Cross placement ─────────────────────────────────────────
    positions = [None, None, None]
    active = [0]

    # 3 separate shape layers (one per cross color, like reference)
    cross_layers = []
    for i in range(3):
        dummy = [np.array([[-1e8, -1e8], [-1e8, -1e8+1],
                           [-1e8+1, -1e8+1], [-1e8+1, -1e8]])]
        lyr = viewer.add_shapes(
            dummy, shape_type='polygon', name=CROSS_LABELS[i],
            face_color=CROSS_COLORS[i], edge_color=CROSS_COLORS[i],
            edge_width=0, opacity=1.0,
        )
        lyr.mouse_pan = False
        lyr.mouse_zoom = False
        cross_layers.append(lyr)

    # Lock ALL layers so nothing can be accidentally dragged/transformed
    for lyr in viewer.layers:
        lyr.editable = False

    # CRITICAL: keep image layer active so shapes layers don't eat keypresses
    img_layer = viewer.layers[slide_name]
    viewer.layers.selection.active = img_layer

    def _keep_image_active(event=None):
        if viewer.layers.selection.active in cross_layers:
            viewer.layers.selection.active = img_layer
    viewer.layers.selection.events.active.connect(_keep_image_active)

    def get_arm():
        zoom = viewer.camera.zoom
        return SCREEN_PX / zoom if zoom > 0 else 3000

    def draw_cross(i, y, x):
        a = get_arm()
        t = a * THICKNESS_FRAC
        h_bar = np.array([[y-t, x-a], [y-t, x+a], [y+t, x+a], [y+t, x-a]])
        v_bar = np.array([[y-a, x-t], [y-a, x+t], [y+a, x+t], [y+a, x-t]])
        cross_layers[i].data = [h_bar, v_bar]

    def redraw_all(event=None):
        for i in range(3):
            if positions[i] is not None:
                draw_cross(i, *positions[i])

    viewer.camera.events.zoom.connect(redraw_all)

    def update_title():
        tags = []
        for i in range(3):
            if positions[i] is not None:
                tags.append(f"{i+1}:OK")
            elif i == active[0]:
                tags.append(f"{i+1}:>>")
            else:
                tags.append(f"{i+1}:..")
        n = sum(p is not None for p in positions)
        extra = " S=save Q=quit" if n == 3 else ""
        viewer.title = f"[{' | '.join(tags)}]{extra}  Space=place 1/2/3=select"

    def save_crosses():
        n = sum(p is not None for p in positions)
        if n < 3:
            show_warning(f"Need 3 crosses, have {n}")
            return False
        crosses = []
        for i in range(3):
            y, x = positions[i]
            crosses.append({
                'id': i + 1,
                'color': ['red', 'green', 'blue'][i],
                'x_px': float(x),
                'y_px': float(y),
                'x_um': float(x * pxum),
                'y_um': float(y * pxum),
            })
        data = {
            'image_width_px': fw,
            'image_height_px': fh,
            'pixel_size_um': pxum,
            'display_transform': {
                'flip_horizontal': flip_h,
                'rotate_cw_90': rotate,
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H%M%S'),
            'crosses': crosses,
        }
        import tempfile, os
        d = str(output_path.parent) if output_path.parent.exists() else '.'
        fd, tmp = tempfile.mkstemp(dir=d, suffix='.json.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, str(output_path))
        except Exception:
            os.unlink(tmp)
            raise
        show_info(f"Saved to {output_path}")
        print(f"  Saved: {output_path}")
        return True

    def place():
        pos = viewer.cursor.position
        if pos is None or len(pos) < 2:
            show_warning("Move cursor over image first")
            return
        y, x = float(pos[-2]), float(pos[-1])
        i = active[0]
        positions[i] = (y, x)
        draw_cross(i, y, x)
        show_info(f"Placed {CROSS_LABELS[i]}")
        print(f"  Placed {CROSS_LABELS[i]} at ({x:.0f}, {y:.0f})")
        # Auto-advance
        for j in range(3):
            if positions[j] is None:
                active[0] = j
                break
        update_title()
        # Auto-save when all 3
        if all(p is not None for p in positions):
            save_crosses()

    # ── Keybindings ─────────────────────────────────────────────
    def _select_cross(idx):
        active[0] = idx
        show_info(f"Selected {CROSS_LABELS[idx]}")
        update_title()

    @viewer.bind_key('r', overwrite=True)
    def _(v): _select_cross(0)

    @viewer.bind_key('g', overwrite=True)
    def _(v): _select_cross(1)

    @viewer.bind_key('b', overwrite=True)
    def _(v): _select_cross(2)

    # Number keys (override napari's layer-switch)
    @viewer.bind_key('1', overwrite=True)
    def _(v): _select_cross(0)

    @viewer.bind_key('2', overwrite=True)
    def _(v): _select_cross(1)

    @viewer.bind_key('3', overwrite=True)
    def _(v): _select_cross(2)

    @viewer.bind_key('Space', overwrite=True)
    def _(v):
        place()

    @viewer.bind_key('p', overwrite=True)
    def _(v):
        place()

    @viewer.bind_key('s', overwrite=True)
    def _(v):
        save_crosses()

    @viewer.bind_key('u', overwrite=True)
    def _(v):
        # Undo: remove last placed
        for i in reversed(range(3)):
            if positions[i] is not None:
                positions[i] = None
                cross_layers[i].data = [np.array([[-1e8, -1e8], [-1e8, -1e8+1],
                                                   [-1e8+1, -1e8+1], [-1e8+1, -1e8]])]
                active[0] = i
                show_info(f"Removed {CROSS_LABELS[i]}")
                update_title()
                return
        show_warning("Nothing to undo")

    @viewer.bind_key('c', overwrite=True)
    def _(v):
        for i in range(3):
            positions[i] = None
            cross_layers[i].data = [np.array([[-1e8, -1e8], [-1e8, -1e8+1],
                                               [-1e8+1, -1e8+1], [-1e8+1, -1e8]])]
        active[0] = 0
        show_info("Cleared all")
        update_title()

    @viewer.bind_key('q', overwrite=True)
    def _(v):
        n = sum(p is not None for p in positions)
        if n == 3:
            save_crosses()
        elif n > 0:
            show_warning(f"Only {n}/3 crosses placed — not saved")
        v.close()

    # ── Load existing crosses ───────────────────────────────────
    if not fresh and output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            for c in existing.get('crosses', []):
                i = c.get('id', 1) - 1
                if 0 <= i < 3 and 'x_px' in c and 'y_px' in c:
                    positions[i] = (c['y_px'], c['x_px'])
                    draw_cross(i, c['y_px'], c['x_px'])
            placed = sum(p is not None for p in positions)
            if placed:
                print(f"  Loaded {placed} existing crosses from {output_path}")
                # Advance to first empty
                for j in range(3):
                    if positions[j] is None:
                        active[0] = j
                        break
        except Exception as e:
            print(f"  Warning: could not load existing crosses: {e}")

    update_title()
    print("\n  1/R=red  2/G=green  3/B=cyan  |  Space=place  S=save  U=undo  C=clear  Q=quit\n")
    napari.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Place 3 LMD reference crosses on CZI or OME-Zarr',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Keys: R/G/B=select  Space=place  S=save  U=undo  C=clear  Q=quit',
    )

    parser.add_argument('--input', '-i', type=str, default=None,
                        help='CZI or OME-Zarr path')
    parser.add_argument('zarr_path', nargs='?', type=str, default=None,
                        help='[Deprecated] Use -i instead')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output crosses JSON')

    # CZI options
    parser.add_argument('--channel', type=int, default=0,
                        help='CZI channel (default: 0)')
    parser.add_argument('--scale-factor', type=int, default=2,
                        help='CZI base downsampling (default: 2 = 1/2 res). '
                             'Second level is auto 4x coarser.')
    parser.add_argument('--pyramid-levels', type=int, nargs='+', default=None,
                        help='CZI downsampling factors (e.g. --pyramid-levels 2 8). '
                             'Overrides --scale-factor.')

    # Zarr options
    parser.add_argument('--level', type=str, default='1/2',
                        help='Zarr pyramid level: "1/2", "1/4", "1/8", or integer '
                             '(default: 1/2)')

    # Display transforms
    parser.add_argument('--flip-horizontal', action='store_true',
                        help='Mirror horizontally (tissue-down for LMD7)')
    parser.add_argument('--rotate-cw-90', action='store_true', default=True,
                        help='Rotate 90 degrees clockwise (default: on for LMD7)')
    parser.add_argument('--no-rotate-cw-90', dest='rotate_cw_90',
                        action='store_false',
                        help='Disable 90-degree clockwise rotation')

    # Overlays
    parser.add_argument('--contours', type=str, default=None,
                        help='Contour JSON for overlay')
    parser.add_argument('--slide', type=str, default=None,
                        help='Slide name filter for per-slide contour files')
    parser.add_argument('--color-by', type=str, default=None,
                        help='Color contours by field (e.g. group, classification, '
                             'score_class, tdTomato_class)')
    parser.add_argument('--sampling-results', type=str, default=None,
                        help='Replicate JSON (lmd_replicates_full.json) — overlay '
                             'colored dots per replicate/well')

    # Other
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size um (auto-detected if not set)')
    parser.add_argument('--fresh', action='store_true',
                        help="Don't auto-load existing crosses")

    # Batch mode
    parser.add_argument('--czi-dir', type=str, default=None,
                        help='Batch: directory of CZIs')
    parser.add_argument('--slides', nargs='+', default=None,
                        help='Batch: slide name prefixes')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Batch: output directory')

    # Backward compat
    parser.add_argument('--detections', '-d', type=str, default=None,
                        help='[Deprecated] Use --contours')
    parser.add_argument('--load-existing', type=str, default=None,
                        help='[Deprecated] Auto-load is default; use --fresh')

    args = parser.parse_args()

    if args.input is None and args.zarr_path:
        args.input = args.zarr_path
    if args.contours is None and args.detections:
        args.contours = args.detections

    # Guidance: check overlay files exist
    transforms = []
    if args.flip_horizontal:
        transforms.append('flip-H')
    if args.rotate_cw_90:
        transforms.append('rot-CW-90')
    print(f"\nDisplay transforms: {', '.join(transforms) if transforms else 'none'}")

    missing = []
    if args.sampling_results and not Path(args.sampling_results).exists():
        missing.append(('--sampling-results', args.sampling_results,
                        'replicate/well assignments (from select_mks_for_lmd.py)'))
    if args.contours and not Path(args.contours).exists():
        missing.append(('--contours', args.contours,
                        'cell outlines for overlay'))
    if missing:
        print("\nMissing overlay files — copy from server first:")
        for flag, path, desc in missing:
            print(f"  {flag} {path}")
            print(f"    ({desc})")
        sys.exit(1)

    # Batch mode
    if args.czi_dir and args.slides:
        czi_dir = Path(args.czi_dir)
        output_dir = Path(args.output_dir or 'crosses')
        output_dir.mkdir(parents=True, exist_ok=True)

        slides_to_process = []
        for slide_name in args.slides:
            matches = list(czi_dir.glob(f"{slide_name}*.czi"))
            if not matches:
                print(f"WARNING: No CZI for '{slide_name}' in {czi_dir}")
                continue
            czi_path = matches[0]
            out_path = output_dir / f"{czi_path.stem}_crosses.json"
            if out_path.exists() and not args.fresh:
                print(f"Skipping {czi_path.stem} (exists: {out_path})")
                continue
            slides_to_process.append((czi_path, out_path))

        for i, (czi_path, out_path) in enumerate(slides_to_process):
            print(f"\n{'='*60}")
            print(f"Slide {i+1}/{len(slides_to_process)}: {czi_path.stem}")
            run_single_slide(args, input_path=str(czi_path), output_path=str(out_path))
        return

    # Single slide
    if args.input is None:
        parser.error("--input / -i is required (or use --czi-dir + --slides for batch)")
    run_single_slide(args)


if __name__ == '__main__':
    main()
