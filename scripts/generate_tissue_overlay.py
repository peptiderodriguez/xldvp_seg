#!/usr/bin/env python
"""Generate tissue overview image with cluster assignments overlaid.

Reads the CZI at low resolution, overlays colored dots for each cell
colored by cluster assignment, and outputs a high-res PNG + interactive HTML.

Usage:
    python generate_tissue_overlay.py \
        --czi-path /path/to/slide.czi \
        --spatial-csv /path/to/spatial.csv \
        --display-channels 3,1,2 \
        --cluster-field cluster_label \
        --output-dir /path/to/output
"""
import argparse
import base64
import gc
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_czi_thumbnail(czi_path, display_channels, scale_factor=0.02):
    """Read CZI mosaic at low resolution, composite 3 channels to RGB."""
    from aicspylibczi import CziFile

    czi = CziFile(czi_path)

    # Get pixel size
    pixel_size = None
    try:
        scaling = czi.get_scaling()
        if scaling and len(scaling) >= 2:
            pixel_size = scaling[0] * 1e6  # m -> um
    except Exception:
        pass

    channels = []
    for ch in display_channels:
        print(f"  Reading channel {ch} at scale {scale_factor}...", flush=True)
        img = czi.read_mosaic(C=ch, scale_factor=scale_factor)
        # Shape: (1, 1, 1, H, W) or (H, W) depending on version
        img = np.squeeze(img)
        channels.append(img)
        print(f"    Shape: {img.shape}, dtype: {img.dtype}", flush=True)

    # Percentile normalize each channel to uint8
    rgb = np.zeros((*channels[0].shape, 3), dtype=np.uint8)
    for i, ch_data in enumerate(channels):
        valid = ch_data[ch_data > 0]
        if len(valid) == 0:
            continue
        p_low, p_high = np.percentile(valid, [1, 99.5])
        if p_high <= p_low:
            p_high = p_low + 1
        normalized = np.clip((ch_data.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)
        rgb[:, :, i] = (normalized * 255).astype(np.uint8)
        # Keep zeros as zero (background)
        rgb[ch_data == 0, i] = 0

    del channels
    gc.collect()
    return rgb, pixel_size


def generate_cluster_colors(labels, cluster_field):
    """Generate a color map for cluster labels."""
    unique = sorted(set(labels))

    # Named marker clusters get fixed colors
    marker_colors = {
        'pm': '#e6194b',       # red
        'msln': '#3cb44b',     # green
        'noise': '#808080',    # gray
        'other': '#4363d8',    # blue
    }

    colors = {}
    # Use tab20 + Set3 for many clusters
    import matplotlib.pyplot as plt
    if len(unique) <= 20:
        cmap = plt.cm.tab20
    else:
        # Generate enough colors by cycling through multiple colormaps
        cmap_list = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c, plt.cm.Set1, plt.cm.Set3]
        all_colors = []
        for cm in cmap_list:
            for i in range(cm.N):
                all_colors.append(cm(i))
        cmap = None

    for i, label in enumerate(unique):
        label_str = str(label)
        if label_str in marker_colors:
            colors[label] = marker_colors[label_str]
        elif cmap is not None:
            rgba = cmap(i % cmap.N)
            colors[label] = '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        else:
            rgba = all_colors[i % len(all_colors)]
            colors[label] = '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    # Noise cluster (-1) always gray
    if -1 in colors:
        colors[-1] = '#808080'
    if '-1' in colors:
        colors['-1'] = '#808080'

    return colors


def generate_overlay_png(rgb, df, cluster_field, color_map, scale_factor,
                         output_path, dot_size=2, alpha=0.7, dpi=200):
    """Generate matplotlib figure with tissue + cluster overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    h, w = rgb.shape[:2]
    fig_w = w / dpi
    fig_h = h / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w * 1.3, fig_h), dpi=dpi)

    # Show tissue
    ax.imshow(rgb, origin='upper')

    # Plot each cluster
    unique_labels = sorted(df[cluster_field].unique(), key=lambda x: (str(x) == '-1', str(x)))
    for label in unique_labels:
        mask = df[cluster_field] == label
        sub = df[mask]
        x_px = sub['x'].values * scale_factor
        y_px = sub['y'].values * scale_factor
        color = color_map.get(label, '#ffffff')
        ax.scatter(x_px, y_px, s=dot_size, c=color, alpha=alpha,
                   edgecolors='none', label=f"{label} ({len(sub)})", rasterized=True)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    # Legend outside image
    n_labels = len(unique_labels)
    ncol = max(1, min(4, n_labels // 10 + 1))
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get(l, '#fff'),
                       markersize=6, label=f"{l} ({(df[cluster_field]==l).sum()})")
               for l in unique_labels]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=5, ncol=ncol, frameon=True, facecolor='#1a1a2e',
              edgecolor='#333', labelcolor='white', markerscale=1.0)

    fig.patch.set_facecolor('#1a1a2e')
    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='#1a1a2e', pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved PNG: {output_path} ({dpi} dpi)", flush=True)


def generate_interactive_html(rgb, df, cluster_field, color_map, scale_factor,
                              output_path, pixel_size=None):
    """Generate interactive HTML with pan/zoom tissue overlay."""
    from PIL import Image

    # Encode tissue image as base64
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    del buf
    gc.collect()

    # Build cell data as compact JSON arrays
    unique_labels = sorted(df[cluster_field].unique(), key=lambda x: (str(x) == '-1', str(x)))

    # Build per-cluster data
    cluster_data_js = []
    for label in unique_labels:
        mask = df[cluster_field] == label
        sub = df[mask]
        x_arr = (sub['x'].values * scale_factor).astype(np.float32)
        y_arr = (sub['y'].values * scale_factor).astype(np.float32)
        color = color_map.get(label, '#ffffff')
        label_str = str(label).replace("'", "\\'").replace('"', '\\"')
        cluster_data_js.append(
            f'{{label:"{label_str}",color:"{color}",n:{len(sub)},'
            f'x:new Float32Array([{",".join(f"{v:.1f}" for v in x_arr)}]),'
            f'y:new Float32Array([{",".join(f"{v:.1f}" for v in y_arr)}])}}'
        )

    clusters_js = ',\n'.join(cluster_data_js)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tissue Cluster Overlay</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }}
  #container {{ position: relative; width: 100vw; height: 100vh; overflow: hidden; cursor: grab; }}
  #container.dragging {{ cursor: grabbing; }}
  canvas {{ position: absolute; top: 0; left: 0; image-rendering: pixelated; }}
  #legend {{
    position: fixed; right: 10px; top: 10px; background: rgba(26,26,46,0.92);
    border: 1px solid #444; border-radius: 8px; padding: 10px; max-height: 90vh;
    overflow-y: auto; z-index: 100; font-size: 12px; min-width: 160px;
  }}
  #legend h3 {{ margin-bottom: 8px; font-size: 13px; color: #aaa; }}
  .leg-item {{
    display: flex; align-items: center; gap: 6px; padding: 2px 4px;
    cursor: pointer; border-radius: 3px; user-select: none;
  }}
  .leg-item:hover {{ background: rgba(255,255,255,0.08); }}
  .leg-item.hidden {{ opacity: 0.3; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .leg-label {{ white-space: nowrap; }}
  #controls {{
    position: fixed; left: 10px; bottom: 10px; background: rgba(26,26,46,0.9);
    border: 1px solid #444; border-radius: 8px; padding: 8px 12px; z-index: 100; font-size: 12px;
  }}
  #tooltip {{
    position: fixed; display: none; background: rgba(0,0,0,0.85); color: #fff;
    padding: 4px 8px; border-radius: 4px; font-size: 11px; pointer-events: none; z-index: 200;
  }}
  .size-slider {{ width: 100px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="container">
  <canvas id="tissue"></canvas>
  <canvas id="dots"></canvas>
</div>
<div id="legend">
  <h3>Clusters ({cluster_field})</h3>
  <div id="leg-items"></div>
  <div style="margin-top:8px;border-top:1px solid #333;padding-top:6px;">
    <span style="color:#888;font-size:11px;">Click to toggle</span>
  </div>
</div>
<div id="controls">
  Zoom: <span id="zoom-val">1.0</span>x &nbsp;|&nbsp;
  Dot size: <input type="range" class="size-slider" id="dot-size" min="1" max="8" value="3" step="0.5">
  <span id="dot-val">3</span>px &nbsp;|&nbsp;
  Opacity: <input type="range" class="size-slider" id="opacity" min="0.1" max="1" value="0.8" step="0.05">
  <span id="op-val">0.8</span> &nbsp;|&nbsp;
  Cells: <span id="cell-count">0</span>
</div>
<div id="tooltip"></div>
<script>
const IMG_W = {w}, IMG_H = {h};
const clusters = [{clusters_js}];

// State
let zoom = 1, panX = 0, panY = 0;
let dragging = false, dragStartX, dragStartY, panStartX, panStartY;
let hidden = new Set();
let dotSize = 3, dotAlpha = 0.8;

// Load tissue image
const tissueCanvas = document.getElementById('tissue');
const dotsCanvas = document.getElementById('dots');
const container = document.getElementById('container');
const tissueCtx = tissueCanvas.getContext('2d');
const dotsCtx = dotsCanvas.getContext('2d');

const img = new Image();
img.onload = () => {{
  resize();
  render();
}};
img.src = 'data:image/png;base64,{img_b64}';

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const cw = window.innerWidth, ch = window.innerHeight;
  for (const c of [tissueCanvas, dotsCanvas]) {{
    c.width = cw * dpr;
    c.height = ch * dpr;
    c.style.width = cw + 'px';
    c.style.height = ch + 'px';
    c.getContext('2d').scale(dpr, dpr);
  }}
}}

function render() {{
  const cw = window.innerWidth, ch = window.innerHeight;
  // Clear
  tissueCtx.fillStyle = '#0d0d1a';
  tissueCtx.fillRect(0, 0, cw, ch);
  dotsCtx.clearRect(0, 0, cw, ch);

  // Draw tissue
  tissueCtx.save();
  tissueCtx.translate(panX, panY);
  tissueCtx.scale(zoom, zoom);
  tissueCtx.imageSmoothingEnabled = zoom < 2;
  tissueCtx.drawImage(img, 0, 0, IMG_W, IMG_H);
  tissueCtx.restore();

  // Draw dots
  dotsCtx.save();
  dotsCtx.translate(panX, panY);
  dotsCtx.scale(zoom, zoom);
  dotsCtx.globalAlpha = dotAlpha;
  const r = dotSize / zoom;  // constant screen-space size
  let total = 0;
  for (const cl of clusters) {{
    if (hidden.has(cl.label)) continue;
    dotsCtx.fillStyle = cl.color;
    for (let i = 0; i < cl.n; i++) {{
      dotsCtx.fillRect(cl.x[i] - r/2, cl.y[i] - r/2, r, r);
    }}
    total += cl.n;
  }}
  dotsCtx.restore();
  document.getElementById('cell-count').textContent = total.toLocaleString();
  document.getElementById('zoom-val').textContent = zoom.toFixed(1);
}}

// Pan & zoom
container.addEventListener('mousedown', e => {{
  dragging = true;
  container.classList.add('dragging');
  dragStartX = e.clientX; dragStartY = e.clientY;
  panStartX = panX; panStartY = panY;
}});
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  panX = panStartX + (e.clientX - dragStartX);
  panY = panStartY + (e.clientY - dragStartY);
  render();
}});
window.addEventListener('mouseup', () => {{
  dragging = false;
  container.classList.remove('dragging');
}});
container.addEventListener('wheel', e => {{
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  const mx = e.clientX, my = e.clientY;
  // Zoom toward cursor
  panX = mx - factor * (mx - panX);
  panY = my - factor * (my - panY);
  zoom *= factor;
  zoom = Math.max(0.1, Math.min(50, zoom));
  render();
}}, {{passive: false}});

// Legend
const legDiv = document.getElementById('leg-items');
for (const cl of clusters) {{
  const item = document.createElement('div');
  item.className = 'leg-item';
  item.innerHTML = `<span class="leg-dot" style="background:${{cl.color}}"></span>` +
    `<span class="leg-label">${{cl.label}} (${{cl.n.toLocaleString()}})</span>`;
  item.onclick = () => {{
    if (hidden.has(cl.label)) hidden.delete(cl.label); else hidden.add(cl.label);
    item.classList.toggle('hidden');
    render();
  }};
  legDiv.appendChild(item);
}}

// Controls
document.getElementById('dot-size').oninput = e => {{
  dotSize = parseFloat(e.target.value);
  document.getElementById('dot-val').textContent = dotSize;
  render();
}};
document.getElementById('opacity').oninput = e => {{
  dotAlpha = parseFloat(e.target.value);
  document.getElementById('op-val').textContent = dotAlpha.toFixed(2);
  render();
}};
window.addEventListener('resize', () => {{ resize(); render(); }});

// Fit to window on load
setTimeout(() => {{
  const cw = window.innerWidth, ch = window.innerHeight;
  zoom = Math.min((cw - 200) / IMG_W, ch / IMG_H) * 0.95;
  panX = (cw - IMG_W * zoom) / 2;
  panY = (ch - IMG_H * zoom) / 2;
  render();
}}, 100);
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Saved interactive HTML: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Generate tissue overlay with cluster colors')
    parser.add_argument('--czi-path', required=True, help='Path to CZI file')
    parser.add_argument('--spatial-csv', required=True, help='spatial.csv with cluster assignments')
    parser.add_argument('--display-channels', default='1,0',
                        help='Channel indices for R,G,B (e.g., "3,1,2" or "1,0" for 2-channel)')
    parser.add_argument('--cluster-field', default='cluster_label',
                        help='Column name for cluster coloring (cluster_label, cluster_id, etc.)')
    parser.add_argument('--scale-factor', type=float, default=0.02,
                        help='CZI read scale factor (0.02 = 2%%)')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--dot-size', type=float, default=2, help='Dot size in pixels (PNG)')
    parser.add_argument('--alpha', type=float, default=0.7, help='Dot opacity (PNG)')
    parser.add_argument('--dpi', type=int, default=200, help='PNG DPI')
    parser.add_argument('--no-html', action='store_true', help='Skip interactive HTML generation')
    parser.add_argument('--no-png', action='store_true', help='Skip static PNG generation')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display_channels = [int(c) for c in args.display_channels.split(',')]
    # Pad to 3 channels for RGB
    while len(display_channels) < 3:
        display_channels.append(-1)  # -1 = black channel

    # 1. Read CZI thumbnail
    print(f"Reading CZI thumbnail at {args.scale_factor*100:.0f}% scale...", flush=True)
    valid_channels = [ch for ch in display_channels if ch >= 0]
    rgb, pixel_size = read_czi_thumbnail(args.czi_path, valid_channels, args.scale_factor)

    # If fewer than 3 channels, remap
    if len(valid_channels) < 3:
        full_rgb = np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
        for i, ch in enumerate(display_channels):
            if ch >= 0:
                src_idx = valid_channels.index(ch)
                full_rgb[:, :, i] = rgb[:, :, src_idx]
        rgb = full_rgb

    print(f"  Thumbnail: {rgb.shape[1]}x{rgb.shape[0]} px", flush=True)
    if pixel_size:
        print(f"  Pixel size: {pixel_size:.4f} um/px", flush=True)

    # 2. Load spatial data
    print(f"Loading spatial data from {args.spatial_csv}...", flush=True)
    df = pd.read_csv(args.spatial_csv)
    print(f"  {len(df)} cells, columns: {list(df.columns)}", flush=True)

    if args.cluster_field not in df.columns:
        print(f"ERROR: cluster_field '{args.cluster_field}' not in columns: {list(df.columns)}")
        sys.exit(1)

    # 3. Generate color map
    color_map = generate_cluster_colors(df[args.cluster_field].values, args.cluster_field)
    n_clusters = len(color_map)
    print(f"  {n_clusters} clusters", flush=True)

    # 4. Generate PNG
    if not args.no_png:
        print("Generating overlay PNG...", flush=True)
        png_path = output_dir / f"tissue_overlay_{args.cluster_field}.png"
        generate_overlay_png(rgb, df, args.cluster_field, color_map, args.scale_factor,
                            png_path, dot_size=args.dot_size, alpha=args.alpha, dpi=args.dpi)

    # 5. Generate interactive HTML
    if not args.no_html:
        print("Generating interactive HTML...", flush=True)
        html_path = output_dir / f"tissue_overlay_{args.cluster_field}.html"
        generate_interactive_html(rgb, df, args.cluster_field, color_map, args.scale_factor,
                                 html_path, pixel_size=pixel_size)

    print("Done!", flush=True)


if __name__ == '__main__':
    main()
