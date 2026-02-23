#!/usr/bin/env python3
"""
Generate per-cluster HTML galleries and spatial overlay maps.

Takes clustered detections (from cluster_by_features.py) and generates:
  1. Spatial cluster map overlaid on tissue zones
  2. Per-cluster HTML pages with ~N sampled cell crops (spatially stratified)
  3. Index page linking to all cluster galleries

Usage:
  python scripts/generate_cluster_gallery.py \
      --detections analysis/clustering_shape/detections_clustered.json \
      --czi-path /path/to/slide.czi \
      --tiles-dir /path/to/output/tiles \
      --output-dir analysis/cluster_gallery \
      --display-channels 1,0 \
      --n-per-cluster 100

  # With zone overlay:
  python scripts/generate_cluster_gallery.py \
      --detections ... --czi-path ... --tiles-dir ... \
      --zones analysis/zones/detections_zoned.json \
      --output-dir analysis/cluster_gallery
"""

import argparse
import base64
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Attempt imports
try:
    import h5py
except ImportError:
    h5py = None

try:
    import hdf5plugin  # noqa: F401 — needed for LZ4 compressed HDF5
except ImportError:
    pass


def spatial_stratified_sample(cells, n, grid_bins=10):
    """Sample up to n cells spatially stratified across a grid.

    Divides tissue extent into grid_bins x grid_bins cells and samples
    proportionally from each occupied bin, ensuring spatial diversity.
    """
    if len(cells) <= n:
        return cells

    xs = np.array([c['global_center'][0] for c in cells])
    ys = np.array([c['global_center'][1] for c in cells])

    x_edges = np.linspace(xs.min(), xs.max() + 1, grid_bins + 1)
    y_edges = np.linspace(ys.min(), ys.max() + 1, grid_bins + 1)

    x_bins = np.digitize(xs, x_edges) - 1
    y_bins = np.digitize(ys, y_edges) - 1

    # Group by bin
    bins = defaultdict(list)
    for i, (xb, yb) in enumerate(zip(x_bins, y_bins)):
        bins[(xb, yb)].append(i)

    # Sample proportionally from each bin
    rng = np.random.RandomState(42)
    n_bins = len(bins)
    per_bin = max(1, n // n_bins)
    sampled = []
    for indices in bins.values():
        k = min(per_bin, len(indices))
        chosen = rng.choice(indices, size=k, replace=False)
        sampled.extend(chosen.tolist())

    # If we have too few, add more from random bins
    if len(sampled) < n:
        all_indices = set(range(len(cells)))
        remaining = list(all_indices - set(sampled))
        extra = min(n - len(sampled), len(remaining))
        sampled.extend(rng.choice(remaining, size=extra, replace=False).tolist())

    # If too many, trim
    if len(sampled) > n:
        sampled = rng.choice(sampled, size=n, replace=False).tolist()

    return [cells[i] for i in sorted(sampled)]


def render_crop(tile_data, center_local, crop_size=128):
    """Extract a crop from multi-channel tile data around a local center.

    Args:
        tile_data: dict {channel_idx: 2D array} or 3D array (H, W, C)
        center_local: (x, y) in tile-local coordinates
        crop_size: size of square crop

    Returns:
        RGB uint8 array (crop_size, crop_size, 3)
    """
    cx, cy = int(center_local[0]), int(center_local[1])
    half = crop_size // 2

    if isinstance(tile_data, dict):
        # Get tile dimensions from first channel
        ch_keys = sorted(tile_data.keys())
        h, w = tile_data[ch_keys[0]].shape
    else:
        h, w = tile_data.shape[:2]

    # Clamp to tile bounds
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    if isinstance(tile_data, dict):
        ch_keys = sorted(tile_data.keys())
        n_ch = len(ch_keys)
        crop_raw = np.zeros((y1 - y0, x1 - x0, n_ch), dtype=np.float32)
        for i, ch in enumerate(ch_keys):
            crop_raw[:, :, i] = tile_data[ch][y0:y1, x0:x1].astype(np.float32)
    else:
        crop_raw = tile_data[y0:y1, x0:x1].astype(np.float32)
        if crop_raw.ndim == 2:
            crop_raw = crop_raw[:, :, np.newaxis]

    # Percentile normalize each channel (non-zero only)
    crop_norm = np.zeros_like(crop_raw)
    for c in range(crop_raw.shape[2]):
        ch = crop_raw[:, :, c]
        valid = ch[ch > 0]
        if len(valid) > 0:
            lo = np.percentile(valid, 1)
            hi = np.percentile(valid, 99.5)
            if hi > lo:
                crop_norm[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)

    # Map to RGB (up to 3 channels)
    h_crop, w_crop = crop_norm.shape[:2]
    rgb = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)
    for i in range(min(3, crop_norm.shape[2])):
        rgb[:, :, i] = (crop_norm[:, :, i] * 255).astype(np.uint8)

    # Pad to crop_size if needed
    if rgb.shape[0] < crop_size or rgb.shape[1] < crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        padded[:rgb.shape[0], :rgb.shape[1]] = rgb
        rgb = padded

    return rgb


def crop_to_base64(rgb_array):
    """Convert RGB array to base64 PNG string."""
    from PIL import Image
    img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def load_tile_channels(czi_path, tile_x, tile_y, tile_size, display_channels,
                       loader=None):
    """Load tile channels from CZI (disk read, no RAM required).

    Returns: dict {channel_idx: 2D uint16 array}
    """
    if loader is None:
        from segmentation.io.czi_loader import CZILoader
        loader = CZILoader(czi_path)

    result = {}
    for ch in display_channels:
        tile = loader.get_tile(tile_x, tile_y, tile_size, channel=ch)
        if tile is not None:
            result[ch] = tile
    return result


def load_tile_from_masks_dir(tiles_dir, tile_x, tile_y):
    """Load mask labels from tile HDF5 file.

    Returns: 2D int array of mask labels, or None
    """
    if h5py is None:
        return None
    tile_dir = Path(tiles_dir) / f"tile_{tile_x}_{tile_y}"
    # Find mask file
    for pattern in ['*_masks.h5', '*_masks.hdf5']:
        matches = list(tile_dir.glob(pattern))
        if matches:
            try:
                import hdf5plugin  # noqa
            except ImportError:
                pass
            with h5py.File(matches[0], 'r') as f:
                return f['masks'][:]
    return None


def generate_spatial_map(detections, output_dir):
    """Generate spatial scatter plot of cells colored by cluster."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.stdout.flush()

    # Extract coords and labels
    xs, ys, labels = [], [], []
    for det in detections:
        gc = det.get('global_center')
        cl = det.get('cluster_label', 'other')
        if gc and cl:
            xs.append(gc[0])
            ys.append(gc[1])
            labels.append(cl)

    xs = np.array(xs)
    ys = np.array(ys)
    labels = np.array(labels)

    unique_labels = sorted(set(labels))

    # Color map
    fixed = {'noise': 'lightgray', 'unclassified': 'lightgray'}
    tab = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    color_map = {}
    ci = 0
    for lbl in unique_labels:
        if lbl in fixed:
            color_map[lbl] = fixed[lbl]
        else:
            color_map[lbl] = tab[ci % len(tab)]
            ci += 1

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot clusters (skip noise first, then overlay)
    for lbl in unique_labels:
        if lbl == 'noise':
            continue
        mask = labels == lbl
        n = mask.sum()
        ax.scatter(xs[mask], ys[mask], c=[color_map[lbl]],
                  s=3, alpha=0.5, label=f'{lbl} ({n})', zorder=1)

    # Noise last (background)
    noise_mask = labels == 'noise'
    if noise_mask.any():
        ax.scatter(xs[noise_mask], ys[noise_mask], c='lightgray',
                  s=1, alpha=0.2, label=f'noise ({noise_mask.sum()})', zorder=0)

    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_title(f'Shape Clusters on Tissue ({len(xs)} cells, '
                 f'{len(unique_labels)} clusters)')
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # Legend — only show clusters with >1% of cells
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles[:20], lbls[:20], loc='upper left',
             fontsize=7, markerscale=3, ncol=2)

    fig.tight_layout()
    out_path = output_dir / 'cluster_spatial_map.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)

    return out_path


def _esc(s):
    """HTML-escape a string."""
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def generate_cluster_html(cluster_label, cells, crops_b64, output_dir,
                          marker_channels=None):
    """Generate HTML gallery page for one cluster.

    Args:
        cluster_label: cluster name
        cells: list of detection dicts (sampled)
        crops_b64: list of base64 PNG strings (parallel to cells)
        output_dir: where to write HTML
        marker_channels: dict {name: ch_idx} for displaying marker values
    """
    html_path = output_dir / f'cluster_{cluster_label}.html'

    # Compute summary stats
    areas = [c.get('features', {}).get('area_um2', 0) for c in cells]
    circs = [c.get('features', {}).get('circularity', 0) for c in cells]

    lines = [
        '<!DOCTYPE html><html><head>',
        f'<title>Cluster: {_esc(cluster_label)}</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 20px; }',
        '.grid { display: flex; flex-wrap: wrap; gap: 8px; }',
        '.cell { text-align: center; font-size: 10px; }',
        '.cell img { width: 128px; height: 128px; image-rendering: pixelated; }',
        '.stats { margin: 10px 0; padding: 10px; background: #222; border-radius: 5px; }',
        'a { color: #6af; }',
        '</style></head><body>',
        f'<h1>Cluster: {_esc(cluster_label)} ({len(cells)} sampled cells)</h1>',
        '<p><a href="index.html">Back to index</a></p>',
        '<div class="stats">',
        f'  Area: {np.mean(areas):.1f} um2 (std {np.std(areas):.1f}) | ',
        f'  Circularity: {np.mean(circs):.2f} (std {np.std(circs):.2f})',
    ]

    if marker_channels:
        for mname, midx in sorted(marker_channels.items(), key=lambda x: x[1]):
            vals = [c.get('features', {}).get(f'ch{midx}_mean', 0) for c in cells]
            lines.append(f'  | {_esc(mname)}: {np.mean(vals):.0f}')

    lines.append('</div>')
    lines.append('<div class="grid">')

    for cell, b64 in zip(cells, crops_b64):
        uid = _esc(cell.get('uid', ''))
        gc = cell.get('global_center', [0, 0])
        area = cell.get('features', {}).get('area_um2', 0)
        lines.append(f'<div class="cell">')
        lines.append(f'  <img src="data:image/png;base64,{b64}" '
                     f'title="{uid}">')
        lines.append(f'  <br>{area:.0f}um2 ({gc[0]:.0f},{gc[1]:.0f})')
        lines.append(f'</div>')

    lines.append('</div></body></html>')

    with open(html_path, 'w') as f:
        f.write('\n'.join(lines))

    return html_path


def generate_index_html(cluster_summaries, output_dir, spatial_map_path=None):
    """Generate index HTML linking to all cluster galleries."""
    lines = [
        '<!DOCTYPE html><html><head>',
        '<title>Cluster Gallery Index</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 20px; }',
        'table { border-collapse: collapse; margin: 20px 0; }',
        'th, td { padding: 6px 12px; border: 1px solid #444; text-align: right; }',
        'th { background: #333; }',
        'a { color: #6af; }',
        'img { max-width: 100%; margin: 20px 0; }',
        '</style></head><body>',
        '<h1>Cluster Gallery</h1>',
    ]

    if spatial_map_path:
        rel = spatial_map_path.name
        lines.append(f'<img src="{rel}" alt="Spatial cluster map">')

    lines.append('<table>')
    lines.append('<tr><th>Cluster</th><th>N cells</th><th>N sampled</th>'
                 '<th>Area (um2)</th><th>Circularity</th><th>Gallery</th></tr>')

    for s in sorted(cluster_summaries, key=lambda x: -x['n_total']):
        lbl = _esc(s['label'])
        lines.append(
            f'<tr><td>{lbl}</td><td>{s["n_total"]}</td>'
            f'<td>{s["n_sampled"]}</td>'
            f'<td>{s["area_mean"]:.1f}</td>'
            f'<td>{s["circ_mean"]:.2f}</td>'
            f'<td><a href="cluster_{lbl}.html">View</a></td></tr>'
        )

    lines.append('</table></body></html>')

    index_path = output_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-cluster HTML galleries with spatial sampling')
    parser.add_argument('--detections', required=True,
                        help='Path to clustered detections JSON '
                        '(from cluster_by_features.py)')
    parser.add_argument('--czi-path', required=True,
                        help='Path to CZI slide file')
    parser.add_argument('--tiles-dir', default=None,
                        help='Path to tiles directory (for mask overlay)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for HTML galleries')
    parser.add_argument('--zones', default=None,
                        help='Path to zoned detections JSON '
                        '(for zone overlay on spatial map)')
    parser.add_argument('--display-channels', type=str, default=None,
                        help='Channels to display as R,G,B: e.g. "1,0,2". '
                        'Default: first 3 available channels')
    parser.add_argument('--marker-channels', type=str, default=None,
                        help='Marker channels for stats: "name:idx,..." '
                        'e.g. "msln:2,pm:1"')
    parser.add_argument('--n-per-cluster', type=int, default=100,
                        help='Max cells to sample per cluster (default: 100)')
    parser.add_argument('--crop-size', type=int, default=128,
                        help='Crop size in pixels (default: 128)')
    parser.add_argument('--tile-size', type=int, default=4000,
                        help='Tile size used during detection (default: 4000)')
    parser.add_argument('--min-cluster-cells', type=int, default=20,
                        help='Skip clusters with fewer cells (default: 20)')
    parser.add_argument('--cluster-field', type=str, default='cluster_label',
                        help='Field to group by (default: cluster_label, '
                        'also: subcluster_label)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse display channels
    if args.display_channels:
        display_channels = [int(c) for c in args.display_channels.split(',')]
    else:
        display_channels = None  # auto-detect

    # Parse marker channels
    marker_channels = None
    if args.marker_channels:
        marker_channels = {}
        for pair in args.marker_channels.split(','):
            name, idx = pair.split(':')
            marker_channels[name.strip()] = int(idx.strip())

    # Load detections — strip features to reduce memory
    # (we only need coords, cluster labels, and a few feature keys for stats)
    print(f"Loading detections from {args.detections}...", flush=True)
    with open(args.detections) as f:
        detections = json.load(f)
    print(f"  {len(detections)} detections", flush=True)

    # Slim down detections: keep only what we need
    _KEEP_FEAT_KEYS = {
        'area_um2', 'circularity', 'eccentricity', 'solidity', 'aspect_ratio',
    }
    # Also keep marker channel means
    if marker_channels:
        for midx in marker_channels.values():
            _KEEP_FEAT_KEYS.add(f'ch{midx}_mean')
    for det in detections:
        feats = det.get('features', {})
        if feats:
            det['features'] = {k: v for k, v in feats.items() if k in _KEEP_FEAT_KEYS}
    import gc as _gc
    _gc.collect()
    print(f"  Slimmed features to {len(_KEEP_FEAT_KEYS)} keys", flush=True)

    # Auto-detect display channels if needed
    from segmentation.io.czi_loader import CZILoader
    loader = CZILoader(args.czi_path)
    if display_channels is None:
        n_channels = loader.num_channels
        display_channels = list(range(min(3, n_channels)))
        print(f"  Auto-detected display channels: {display_channels}", flush=True)

    # Group by cluster
    cluster_field = args.cluster_field
    clusters = defaultdict(list)
    for det in detections:
        label = det.get(cluster_field)
        if label:
            clusters[label].append(det)

    print(f"  {len(clusters)} unique clusters (field: {cluster_field})", flush=True)

    # Generate spatial map
    print("\nGenerating spatial cluster map...", flush=True)
    spatial_path = generate_spatial_map(detections, output_dir)

    # Sample cells and generate crops
    print(f"\nGenerating galleries ({args.n_per_cluster} cells/cluster, "
          f"crop {args.crop_size}px)...", flush=True)

    # Batch by tile: figure out which tiles we need
    sampled_by_cluster = {}
    tile_cells = defaultdict(list)  # (tx, ty) -> [(cluster, cell), ...]

    for label in sorted(clusters.keys()):
        cells = clusters[label]
        if len(cells) < args.min_cluster_cells:
            continue
        sampled = spatial_stratified_sample(cells, args.n_per_cluster)
        sampled_by_cluster[label] = sampled

        for cell in sampled:
            to = cell.get('tile_origin', [0, 0])
            tx, ty = int(to[0]), int(to[1])
            tile_cells[(tx, ty)].append((label, cell))

    print(f"  {len(sampled_by_cluster)} clusters to render, "
          f"{len(tile_cells)} tiles to read", flush=True)

    # Read tiles and extract crops
    crops = defaultdict(list)  # cluster_label -> [base64, ...]
    crop_cells = defaultdict(list)  # cluster_label -> [cell, ...]

    n_tiles = len(tile_cells)
    for ti, ((tx, ty), cells_in_tile) in enumerate(sorted(tile_cells.items())):
        if ti % 10 == 0:
            print(f"  Reading tile {ti+1}/{n_tiles}: ({tx}, {ty}) "
                  f"({len(cells_in_tile)} crops)", flush=True)

        # Load tile channels from CZI
        tile_data = {}
        for ch in display_channels:
            arr = loader.get_tile(tx, ty, args.tile_size, channel=ch)
            if arr is not None:
                tile_data[ch] = arr

        if not tile_data:
            print(f"  WARNING: No data for tile ({tx}, {ty})")
            continue

        # Extract crops for each cell in this tile
        for label, cell in cells_in_tile:
            center_local = cell.get('center', [0, 0])
            rgb = render_crop(tile_data, center_local, args.crop_size)
            b64 = crop_to_base64(rgb)
            crops[label].append(b64)
            crop_cells[label].append(cell)

    # Generate per-cluster HTML
    print(f"\nWriting HTML galleries...")
    cluster_summaries = []

    for label in sorted(sampled_by_cluster.keys()):
        cells = crop_cells.get(label, [])
        b64s = crops.get(label, [])

        if not cells:
            continue

        html_path = generate_cluster_html(
            label, cells, b64s, output_dir, marker_channels)

        areas = [c.get('features', {}).get('area_um2', 0) for c in cells]
        circs = [c.get('features', {}).get('circularity', 0) for c in cells]

        cluster_summaries.append({
            'label': label,
            'n_total': len(clusters[label]),
            'n_sampled': len(cells),
            'area_mean': float(np.mean(areas)) if areas else 0,
            'circ_mean': float(np.mean(circs)) if circs else 0,
        })
        print(f"  {label}: {len(cells)} crops -> {html_path.name}")

    # Generate index page
    generate_index_html(cluster_summaries, output_dir, spatial_path)

    print(f"\nDone! {len(cluster_summaries)} cluster galleries in {output_dir}")


if __name__ == '__main__':
    main()
