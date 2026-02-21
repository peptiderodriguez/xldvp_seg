#!/usr/bin/env python3
"""Generate an HTML islet overview from a completed islet detection run.

Iterates all tiles in a run directory, loads masks and features, runs DBSCAN
spatial clustering on endocrine cell coordinates, applies Otsu filtering to
separate true islets from spurious clusters, and generates an HTML page with
one card per true islet showing:
  - Solid neon pink boundary (union of cell masks + 15px elliptical dilation)
  - Thin dashed colored cell contours by marker type
  - Composition bar per islet
  - Sorted by cell count descending

Marker colors: red=alpha(Gcg), green=beta(Ins), blue=delta(Sst),
               orange=multi, gray=none

Usage:
  python scripts/generate_islet_overview.py \\
      --run-dir /path/to/islet_run_output \\
      --czi-path /path/to/slide.czi \\
      --eps-um 50 --min-samples 5
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from sklearn.cluster import DBSCAN

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.io.czi_loader import get_loader
from run_segmentation import classify_islet_marker, compute_islet_marker_thresholds

try:
    import hdf5plugin  # noqa: F401 — registers LZ4 codec for h5py
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

PADDING = 40
PINK = (255, 0, 255)
MARKER_COLORS = {
    'alpha': (255, 50, 50),
    'beta': (50, 255, 50),
    'delta': (50, 50, 255),
    'multi': (255, 170, 0),
    'none': (128, 128, 128),
}


def pct_norm(img):
    """Percentile-normalize a 3-channel image, preserving zero (padding) pixels."""
    out = np.zeros_like(img, dtype=np.uint8)
    valid = np.any(img > 0, axis=-1)
    for c in range(3):
        ch = img[:, :, c].astype(float)
        vals = ch[valid]
        if len(vals) == 0:
            continue
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99.5)
        if hi > lo:
            out[:, :, c] = np.clip(255 * (ch - lo) / (hi - lo), 0, 255).astype(np.uint8)
    return out


def draw_dashed_contours(img, contours, color, thickness=1, dash_len=6, gap_len=4):
    """Draw dashed contour lines on *img* in-place."""
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 2:
            continue
        all_pts = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dist = np.linalg.norm(p2 - p1)
            if dist < 1:
                continue
            n_steps = max(int(dist), 1)
            for t in np.linspace(0, 1, n_steps, endpoint=False):
                all_pts.append((p1 + t * (p2 - p1)).astype(int))
        cycle = dash_len + gap_len
        for i, pt in enumerate(all_pts):
            if (i % cycle) < dash_len:
                cv2.circle(img, tuple(pt), 0, color, thickness)


def render_islet_card(islet_id, cells, masks, tile_vis, tile_x, tile_y,
                      tile_h, tile_w, pixel_size, signal_per_cell):
    """Render a single islet card and return HTML string, or None if empty."""
    cell_info = []
    mask_labels = []
    for d in cells:
        gc = d.get('global_center', d.get('center', [0, 0]))
        cx_rel = gc[0] - tile_x
        cy_rel = gc[1] - tile_y
        ml = d.get('mask_label')
        cell_info.append((cx_rel, cy_rel, ml, d.get('marker_class', 'none')))
        if ml is not None and ml > 0:
            mask_labels.append(ml)

    if not mask_labels:
        return None

    # Union mask + dilate for boundary
    union_mask = np.zeros((tile_h, tile_w), dtype=np.uint8)
    for ml in mask_labels:
        union_mask |= (masks == ml).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(union_mask, kernel)

    ys, xs = np.where(dilated > 0)
    if len(xs) == 0:
        return None

    x_min = max(0, int(xs.min()) - PADDING)
    x_max = min(tile_w, int(xs.max()) + PADDING)
    y_min = max(0, int(ys.min()) - PADDING)
    y_max = min(tile_h, int(ys.max()) + PADDING)

    crop = tile_vis[y_min:y_max, x_min:x_max].copy()

    # Dashed cell contours
    for _cx_rel, _cy_rel, ml, mc in cell_info:
        color = MARKER_COLORS.get(mc, (128, 128, 128))
        if ml is not None and ml > 0:
            mask_crop = (masks[y_min:y_max, x_min:x_max] == ml).astype(np.uint8)
            if mask_crop.any():
                contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                draw_dashed_contours(crop, contours, color, thickness=1)

    # Solid pink islet boundary
    boundary_crop = dilated[y_min:y_max, x_min:x_max]
    contours, _ = cv2.findContours(boundary_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, PINK, 2, cv2.LINE_AA)

    _, buf = cv2.imencode('.png', cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf).decode()

    n = len(cells)
    na = sum(1 for d in cells if d.get('marker_class') == 'alpha')
    nb = sum(1 for d in cells if d.get('marker_class') == 'beta')
    nd = sum(1 for d in cells if d.get('marker_class') == 'delta')
    nm = sum(1 for d in cells if d.get('marker_class') == 'multi')
    area_um2 = float(union_mask.sum()) * pixel_size * pixel_size

    total = max(n, 1)
    bar_parts = []
    if nb > 0:
        bar_parts.append(f'<div style="width:{100*nb/total:.0f}%;background:#33cc33" title="beta {nb}"></div>')
    if na > 0:
        bar_parts.append(f'<div style="width:{100*na/total:.0f}%;background:#ff3333" title="alpha {na}"></div>')
    if nd > 0:
        bar_parts.append(f'<div style="width:{100*nd/total:.0f}%;background:#3333ff" title="delta {nd}"></div>')
    if nm > 0:
        bar_parts.append(f'<div style="width:{100*nm/total:.0f}%;background:#ffaa00" title="multi {nm}"></div>')
    bar = (
        f'<div style="display:flex;height:8px;width:100%;border-radius:4px;'
        f'overflow:hidden;margin:4px 0">{"".join(bar_parts)}</div>'
    )

    spc = signal_per_cell
    return f'''
    <div style="display:inline-block;margin:10px;background:#111;border:2px solid #333;
         border-radius:8px;padding:8px;vertical-align:top;max-width:{crop.shape[1]+20}px">
        <img src="data:image/png;base64,{b64}" style="display:block;border-radius:4px">
        <div style="color:white;font-family:monospace;font-size:12px;margin-top:6px">
            <b style="color:#ff00ff">Islet {islet_id}</b> | {n} cells | {area_um2:.0f} um2 | sig/cell={spc:.2f}
            {bar}
            <span style="color:#ff3333">a:{na}</span>
            <span style="color:#33cc33">b:{nb}</span>
            <span style="color:#3333ff">d:{nd}</span>
            <span style="color:#ffaa00">m:{nm}</span>
        </div>
    </div>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate HTML islet overview from completed run directory'
    )
    parser.add_argument('--run-dir', required=True,
                        help='Path to existing run output directory')
    parser.add_argument('--czi-path', required=True,
                        help='Path to CZI file')
    parser.add_argument('--eps-um', type=float, default=50.0,
                        help='DBSCAN epsilon in micrometers (default: 50)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN min_samples (default: 5)')
    parser.add_argument('--marker-only', action='store_true',
                        help='Only show marker-positive cells (exclude none/gray)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    run_dir = Path(args.run_dir)
    czi_path = Path(args.czi_path)

    # ---------------------------------------------------------------
    # 1. Load global detections
    # ---------------------------------------------------------------
    det_path = run_dir / 'islet_detections.json'
    if not det_path.exists():
        logger.error(f"Detections not found: {det_path}")
        sys.exit(1)

    print(f"Loading detections from {det_path}...")
    with open(det_path) as f:
        all_dets = json.load(f)
    print(f"  {len(all_dets)} total detections")

    if len(all_dets) == 0:
        logger.warning("No detections found — nothing to do")
        sys.exit(0)

    # ---------------------------------------------------------------
    # 2. Compute marker thresholds and classify
    # ---------------------------------------------------------------
    marker_thresholds = compute_islet_marker_thresholds(all_dets)
    if marker_thresholds is None:
        print(f"WARNING: Only {len(all_dets)} detections — too few for marker thresholds. "
              "All cells will be shown as 'none' (gray).")
        if args.marker_only:
            print("ERROR: --marker-only requires marker thresholds (need >= 10 detections).")
            sys.exit(1)
    marker_counts = {}
    for det in all_dets:
        mc, _ = classify_islet_marker(det.get('features', {}), marker_thresholds)
        det['marker_class'] = mc
        marker_counts[mc] = marker_counts.get(mc, 0) + 1
    print(f"Marker classification: {marker_counts}")

    # ---------------------------------------------------------------
    # 3. Load CZI display channels
    # ---------------------------------------------------------------
    print(f"Loading CZI channels (2=Gcg, 3=Ins, 5=Sst)...")
    loader = get_loader(czi_path, load_to_ram=True, channel=1)
    pixel_size = loader.get_pixel_size()
    x_start = loader.x_start
    y_start = loader.y_start

    ch_data = {}
    for ch in [2, 3, 5]:
        print(f"  Loading channel {ch}...")
        loader.load_channel(ch)
        ch_data[ch] = loader._channel_data[ch]

    # ---------------------------------------------------------------
    # 4. DBSCAN spatial clustering on endocrine cell coordinates
    # ---------------------------------------------------------------
    # Build coordinate array for DBSCAN
    coords = []
    coord_det_indices = []  # maps coord row -> index in all_dets
    for i, det in enumerate(all_dets):
        mc = det.get('marker_class', 'none')
        if args.marker_only and mc == 'none':
            continue
        gc = det.get('global_center', det.get('center'))
        if gc is None:
            continue
        coords.append([gc[0] * pixel_size, gc[1] * pixel_size])  # convert to um
        coord_det_indices.append(i)

    if len(coords) == 0:
        logger.warning("No cells with coordinates found — nothing to cluster")
        sys.exit(0)

    coords_um = np.array(coords)
    print(f"Running DBSCAN on {len(coords_um)} cells (eps={args.eps_um} um, min_samples={args.min_samples})...")

    dbscan = DBSCAN(eps=args.eps_um, min_samples=args.min_samples)
    cluster_labels = dbscan.fit_predict(coords_um)

    # Assign islet_id to detections (-1 = noise/unclustered)
    for row_idx, det_idx in enumerate(coord_det_indices):
        all_dets[det_idx]['islet_id'] = int(cluster_labels[row_idx])

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"  {n_clusters} clusters, {n_noise} noise points")

    if n_clusters == 0:
        logger.warning("DBSCAN found no clusters — try decreasing --eps-um or --min-samples")
        sys.exit(0)

    # ---------------------------------------------------------------
    # 5. Compute normalized signal/cell per islet and Otsu filter
    # ---------------------------------------------------------------
    gcg_all = np.array([d.get('features', {}).get('ch2_mean', 0) for d in all_dets])
    ins_all = np.array([d.get('features', {}).get('ch3_mean', 0) for d in all_dets])
    sst_all = np.array([d.get('features', {}).get('ch5_mean', 0) for d in all_dets])

    def norm_arr(arr):
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99.5)
        return np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)

    total_n = norm_arr(gcg_all) + norm_arr(ins_all) + norm_arr(sst_all)

    # Group by islet_id (exclude noise = -1)
    islet_groups = {}  # islet_id -> [det_index, ...]
    for i, det in enumerate(all_dets):
        iid = det.get('islet_id')
        if iid is None or iid < 0:
            continue
        islet_groups.setdefault(iid, []).append(i)

    islet_sig_per_cell = {}
    for iid, idx_list in islet_groups.items():
        islet_sig_per_cell[iid] = np.mean([total_n[i] for i in idx_list])

    # Otsu filter — guard against edge cases
    spc_arr = np.array(list(islet_sig_per_cell.values()))
    if len(spc_arr) < 5:
        logger.warning(f"Only {len(spc_arr)} islets — too few for reliable Otsu threshold, keeping all")
        otsu_spc = 0
    else:
        try:
            from skimage.filters import threshold_otsu
            otsu_spc = threshold_otsu(spc_arr)
        except (ValueError, IndexError):
            # Empty array, all values identical, or single islet
            logger.warning("Cannot compute Otsu threshold — keeping all islets")
            otsu_spc = 0  # keep all

    print(f"Otsu signal/cell threshold: {otsu_spc:.3f}")

    true_islets = {iid for iid, spc in islet_sig_per_cell.items() if spc >= otsu_spc}
    rejected = {iid for iid in islet_sig_per_cell if iid not in true_islets}
    n_cells_true = sum(len(islet_groups[i]) for i in true_islets)
    n_cells_rejected = sum(len(islet_groups[i]) for i in rejected)
    print(f"True islets: {len(true_islets)} ({n_cells_true} cells)")
    print(f"Rejected: {len(rejected)} ({n_cells_rejected} cells)")

    if len(true_islets) == 0:
        logger.warning("No islets passed Otsu filter — try lowering --min-samples or check detections")
        sys.exit(0)

    # ---------------------------------------------------------------
    # 6. Determine which tiles each islet spans
    # ---------------------------------------------------------------
    tiles_dir = run_dir / 'tiles'
    if not tiles_dir.exists():
        logger.error(f"Tiles directory not found: {tiles_dir}")
        sys.exit(1)

    # Discover available tiles
    tile_info = {}  # (tile_x, tile_y) -> tile_dir path
    for td in sorted(tiles_dir.iterdir()):
        if not td.is_dir() or not td.name.startswith('tile_'):
            continue
        parts = td.name.split('_')
        if len(parts) < 3:
            continue
        try:
            tx, ty = int(parts[1]), int(parts[2])
            tile_info[(tx, ty)] = td
        except ValueError:
            continue

    print(f"Found {len(tile_info)} tile directories")

    if len(tile_info) == 0:
        logger.error("No tile directories found")
        sys.exit(1)

    # Map each detection to its tile by checking global_center
    # We need to know tile dimensions to do this; load first tile's masks for shape
    first_tile_dir = next(iter(tile_info.values()))
    mask_path = first_tile_dir / 'islet_masks.h5'
    with h5py.File(mask_path, 'r') as f:
        first_masks = f['masks'][:]
    tile_h, tile_w = first_masks.shape[:2]
    del first_masks

    # For each true islet, pick the tile where the islet bounding box is most
    # interior (furthest from tile edges), so the islet is never cut off.
    islet_tile_map = {}  # islet_id -> (tile_x, tile_y)
    for iid in true_islets:
        # Collect all cell coordinates in this islet
        cell_coords = []
        for det_idx in islet_groups[iid]:
            det = all_dets[det_idx]
            gc = det.get('global_center', det.get('center', [0, 0]))
            cell_coords.append(gc)
        if not cell_coords:
            continue
        cell_coords = np.array(cell_coords)
        # Islet bounding box in global coordinates
        islet_xmin, islet_ymin = cell_coords.min(axis=0)
        islet_xmax, islet_ymax = cell_coords.max(axis=0)

        # Score each candidate tile: min distance from islet bbox to tile edge
        best_tile = None
        best_margin = -1
        for (tx, ty) in tile_info:
            # Check tile fully contains the islet bbox
            margin_left = islet_xmin - tx
            margin_right = (tx + tile_w) - islet_xmax
            margin_top = islet_ymin - ty
            margin_bottom = (ty + tile_h) - islet_ymax
            min_margin = min(margin_left, margin_right, margin_top, margin_bottom)
            if min_margin > best_margin:
                best_margin = min_margin
                best_tile = (tx, ty)
        if best_tile is not None:
            islet_tile_map[iid] = best_tile
            if best_margin < 0:
                logger.warning(f"Islet {iid}: no tile fully contains it "
                               f"(best margin={best_margin:.0f}px). "
                               f"May be clipped at tile edge.")

    # ---------------------------------------------------------------
    # 7. Load masks + build visuals per tile (only tiles with true islets)
    # ---------------------------------------------------------------
    needed_tiles = set(islet_tile_map.values())
    tile_masks_cache = {}   # (tx, ty) -> masks array
    tile_vis_cache = {}     # (tx, ty) -> percentile-normalized RGB

    for (tx, ty) in sorted(needed_tiles):
        td = tile_info.get((tx, ty))
        if td is None:
            logger.warning(f"Tile ({tx}, {ty}) not found in tiles directory — skipping")
            continue

        # Load masks
        mask_path = td / 'islet_masks.h5'
        if not mask_path.exists():
            logger.warning(f"No masks file in {td} — skipping")
            continue
        with h5py.File(mask_path, 'r') as f:
            masks = f['masks'][:]
        tile_masks_cache[(tx, ty)] = masks

        # Build RGB from display channels
        th, tw = masks.shape[:2]
        rel_tx = tx - x_start
        rel_ty = ty - y_start
        tile_rgb = np.stack([
            ch_data[2][rel_ty:rel_ty+th, rel_tx:rel_tx+tw],
            ch_data[3][rel_ty:rel_ty+th, rel_tx:rel_tx+tw],
            ch_data[5][rel_ty:rel_ty+th, rel_tx:rel_tx+tw],
        ], axis=-1)
        tile_vis_cache[(tx, ty)] = pct_norm(tile_rgb)

        print(f"  Loaded tile ({tx}, {ty}): masks {masks.shape}, {masks.max()} labels")

    # ---------------------------------------------------------------
    # 8. Render HTML cards for each true islet
    # ---------------------------------------------------------------
    cards_html = ''
    rendered_count = 0
    skipped_count = 0

    for iid in sorted(true_islets, key=lambda x: len(islet_groups[x]), reverse=True):
        tile_key = islet_tile_map.get(iid)
        if tile_key is None:
            skipped_count += 1
            continue
        tx, ty = tile_key

        if tile_key not in tile_masks_cache or tile_key not in tile_vis_cache:
            skipped_count += 1
            continue

        masks = tile_masks_cache[tile_key]
        tile_vis = tile_vis_cache[tile_key]
        th, tw = masks.shape[:2]

        # Filter to cells within this tile — cross-tile cells have mask_labels
        # from their own tile's label array, which would collide with this tile's labels
        all_islet_cells = [all_dets[i] for i in islet_groups[iid]]
        cells = []
        for det in all_islet_cells:
            gc = det.get('global_center', det.get('center', [0, 0]))
            if tx <= gc[0] < tx + tw and ty <= gc[1] < ty + th:
                cells.append(det)
        if not cells:
            skipped_count += 1
            continue
        spc = islet_sig_per_cell[iid]

        card = render_islet_card(
            iid, cells, masks, tile_vis, tx, ty, th, tw, pixel_size, spc
        )
        if card is not None:
            cards_html += card
            rendered_count += 1
        else:
            skipped_count += 1

    print(f"\nRendered {rendered_count} islet cards ({skipped_count} skipped)")

    # ---------------------------------------------------------------
    # 9. Assemble final HTML
    # ---------------------------------------------------------------
    slide_name = czi_path.stem
    n_true = len(true_islets)

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Islet Overview</title>
<style>body {{ background: #000; margin: 20px; }}
h1 {{ color: white; font-family: sans-serif; }}
.sub {{ color: #aaa; font-family: monospace; font-size: 14px; margin-bottom: 20px; }}</style>
</head><body>
<h1>Islet Overview &mdash; {slide_name}</h1>
<div class="sub">{n_true} true islets ({n_cells_true} cells), {len(rejected)} rejected
(Otsu sig/cell &ge; {otsu_spc:.3f}) |
DBSCAN eps={args.eps_um} um, min_samples={args.min_samples} |
<span style="color:#ff00ff">pink = islet boundary</span> |
<span style="color:#ff3333">red=alpha</span>
<span style="color:#33cc33">green=beta</span>
<span style="color:#3333ff">blue=delta</span>
<span style="color:#ffaa00">orange=multi</span> (dashed)</div>
{cards_html}
</body></html>'''

    html_dir = run_dir / 'html'
    html_dir.mkdir(parents=True, exist_ok=True)
    out_path = html_dir / 'islet_overview.html'
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Saved: {out_path} ({len(html)/1024/1024:.1f} MB)")


if __name__ == '__main__':
    main()
