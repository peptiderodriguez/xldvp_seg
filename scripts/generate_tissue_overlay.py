#!/usr/bin/env python3
"""Generate tissue overview with fluorescence image, cell overlay, ROI drawing, and LMD export.

Reads the CZI at low resolution, overlays colored dots for each cell
colored by marker profile or cluster assignment. Supports:
- Per-channel fluorescence toggle (R/G/B)
- Interactive ROI drawing (circle, rect, polygon) with cell stats
- ROI JSON export with cell UIDs for downstream LMD export
- Input from detections JSON or spatial CSV

Usage:
    # From detections JSON (preferred — includes UIDs and marker profiles)
    python generate_tissue_overlay.py \\
        --czi-path /path/to/slide.czi \\
        --detections /path/to/cell_detections_classified.json \\
        --display-channels 2,0,1 \\
        --group-field marker_profile \\
        --output-dir /path/to/output

    # From spatial CSV (legacy)
    python generate_tissue_overlay.py \\
        --czi-path /path/to/slide.czi \\
        --spatial-csv /path/to/spatial.csv \\
        --display-channels 3,1,2 \\
        --group-field cluster_label \\
        --output-dir /path/to/output
"""

import argparse
import base64
import gc
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from segmentation.utils.json_utils import fast_json_load


def read_czi_thumbnail(czi_path, display_channels, scale_factor=0.02, scene=0):
    """Read CZI mosaic at low resolution, return per-channel uint8 arrays.

    Args:
        czi_path: Path to CZI file
        display_channels: List of channel indices to read
        scale_factor: Downsampling factor (e.g. 0.02 = 2%)
        scene: CZI scene index (0-based, default 0)

    Returns:
        channel_arrays: list of uint8 arrays (one per channel, percentile-normalized)
        pixel_size: pixel size in um, or None
    """
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

    bbox = czi.get_mosaic_scene_bounding_box(index=scene)
    region = (bbox.x, bbox.y, bbox.w, bbox.h)
    print(f"  Scene {scene}: {bbox.w}x{bbox.h} px at ({bbox.x}, {bbox.y})", flush=True)

    channel_arrays = []
    for ch in display_channels:
        print(f"  Reading channel {ch} at scale {scale_factor}...", flush=True)
        img = czi.read_mosaic(C=ch, region=region, scale_factor=scale_factor)
        img = np.squeeze(img)
        print(f"    Shape: {img.shape}, dtype: {img.dtype}", flush=True)

        # Percentile normalize to uint8
        valid = img[img > 0]
        if len(valid) == 0:
            normalized = np.zeros(img.shape, dtype=np.uint8)
        else:
            p_low, p_high = np.percentile(valid, [1, 99.5])
            if p_high <= p_low:
                p_high = p_low + 1
            normalized = np.clip((img.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)
            normalized = (normalized * 255).astype(np.uint8)
            normalized[img == 0] = 0

        channel_arrays.append(normalized)

    return channel_arrays, pixel_size


def load_detections_json(detections_path, group_field="marker_profile", marker_filter=None):
    """Load detections JSON and extract cell data for visualization.

    Returns:
        df: DataFrame with columns [x, y, uid, group]
    """
    print(f"Loading detections from {detections_path}...", flush=True)
    detections = fast_json_load(detections_path)
    print(f"  {len(detections)} detections loaded", flush=True)

    if marker_filter:
        from segmentation.utils.detection_utils import apply_marker_filter

        detections = apply_marker_filter(detections, marker_filter)
        print(f"  {len(detections)} after marker filter: {marker_filter}", flush=True)

    rows = []
    for det in detections:
        feat = det.get("features", {})
        # Global center coords (slide-level pixels)
        gc = det.get("global_center")
        if gc is None:
            gc = feat.get("global_center")
        if gc is None:
            continue

        uid = det.get("uid", "")
        group = feat.get(group_field, "unknown")
        if group is None:
            group = "unknown"
        rows.append({"x": gc[0], "y": gc[1], "uid": uid, "group": str(group)})

    df = pd.DataFrame(rows)
    print(f"  {len(df)} cells with coordinates", flush=True)

    # Show group distribution
    if len(df) > 0:
        for grp, cnt in df["group"].value_counts().items():
            print(f"    {grp}: {cnt} ({100*cnt/len(df):.1f}%)", flush=True)

    return df


def generate_cluster_colors(labels, cluster_field):
    """Generate a color map for cluster labels."""
    unique = sorted(set(labels))

    # Fixed colors for common marker profiles
    marker_colors = {
        "NeuN+/tdTomato+": "#f1c40f",  # yellow
        "NeuN+/tdTomato-": "#e74c3c",  # red
        "NeuN-/tdTomato+": "#2ecc71",  # green
        "NeuN-/tdTomato-": "#3498db",  # blue
        "pm": "#e6194b",
        "msln": "#3cb44b",
        "noise": "#808080",
        "other": "#4363d8",
    }

    colors = {}
    import matplotlib.pyplot as plt

    if len(unique) <= 20:
        cmap = plt.cm.tab20
    else:
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
            colors[label] = (
                f"#{int(rgba[0] * 255):02x}{int(rgba[1] * 255):02x}{int(rgba[2] * 255):02x}"
            )
        else:
            rgba = all_colors[i % len(all_colors)]
            colors[label] = (
                f"#{int(rgba[0] * 255):02x}{int(rgba[1] * 255):02x}{int(rgba[2] * 255):02x}"
            )

    if -1 in colors:
        colors[-1] = "#808080"
    if "-1" in colors:
        colors["-1"] = "#808080"

    return colors


def generate_overlay_png(
    channel_arrays,
    display_channels,
    df,
    cluster_field,
    color_map,
    scale_factor,
    output_path,
    dot_size=2,
    alpha=0.7,
    dpi=200,
):
    """Generate matplotlib figure with tissue + cluster overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Composite to RGB
    h, w = channel_arrays[0].shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, ch_data in enumerate(channel_arrays):
        if i < 3:
            rgb[:, :, i] = ch_data

    fig_w = w / dpi
    fig_h = h / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w * 1.3, fig_h), dpi=dpi)
    ax.imshow(rgb, origin="upper")

    unique_labels = sorted(df[cluster_field].unique(), key=lambda x: (str(x) == "-1", str(x)))
    for label in unique_labels:
        mask = df[cluster_field] == label
        sub = df[mask]
        x_px = sub["x"].values * scale_factor
        y_px = sub["y"].values * scale_factor
        color = color_map.get(label, "#ffffff")
        ax.scatter(
            x_px,
            y_px,
            s=dot_size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            label=f"{label} ({len(sub)})",
            rasterized=True,
        )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    n_labels = len(unique_labels)
    ncol = max(1, min(4, n_labels // 10 + 1))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(l, "#fff"),
            markersize=6,
            label=f"{l} ({(df[cluster_field]==l).sum()})",
        )
        for l in unique_labels
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=5,
        ncol=ncol,
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#333",
        labelcolor="white",
        markerscale=1.0,
    )

    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="#1a1a2e", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved PNG: {output_path} ({dpi} dpi)", flush=True)


def _encode_channel_b64(ch_array):
    """Encode a single-channel uint8 array as PNG base64."""
    from PIL import Image

    img = Image.fromarray(ch_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    del buf
    return b64


def generate_interactive_html(
    channel_arrays,
    display_channels,
    channel_names,
    df,
    group_field,
    color_map,
    scale_factor,
    output_path,
    pixel_size=None,
    has_uids=False,
):
    """Generate interactive HTML with fluorescence channel toggles, ROI drawing, and cell overlay.

    Args:
        channel_arrays: list of uint8 arrays, one per display channel
        display_channels: list of CZI channel indices
        channel_names: list of human-readable channel names (e.g. ['tdTomato', 'nuc', 'NeuN'])
        df: DataFrame with x, y, group columns (and optionally uid)
        group_field: name of the group column
        color_map: dict mapping group label -> hex color
        scale_factor: CZI scale factor used
        output_path: path to write HTML file
        pixel_size: um/px, or None
        has_uids: whether df has uid column
    """
    h, w = channel_arrays[0].shape

    # Encode each channel as separate base64 PNG for JS toggle
    ch_b64 = []
    for ch_arr in channel_arrays:
        ch_b64.append(_encode_channel_b64(ch_arr))
    while len(ch_b64) < 3:
        ch_b64.append(_encode_channel_b64(np.zeros((h, w), dtype=np.uint8)))

    # Channel display colors (R, G, B)
    ch_colors = ["#ff4444", "#44ff44", "#4488ff"]
    # Pad channel_names
    while len(channel_names) < 3:
        channel_names.append("(off)")

    # Build per-group cell data with UIDs
    unique_labels = sorted(df[group_field].unique(), key=lambda x: (str(x) == "-1", str(x)))
    cluster_data_js = []
    for label in unique_labels:
        mask = df[group_field] == label
        sub = df[mask]
        x_arr = (sub["x"].values * scale_factor).astype(np.float32)
        y_arr = (sub["y"].values * scale_factor).astype(np.float32)
        color = color_map.get(label, "#ffffff")
        label_str = str(label).replace("'", "\\'").replace('"', '\\"')

        uid_js = "null"
        if has_uids and "uid" in sub.columns:
            uids = sub["uid"].values.tolist()
            uid_js = json.dumps(uids)

        cluster_data_js.append(
            f'{{"label":"{label_str}","color":"{color}","n":{len(sub)},'
            f'"x":new Float32Array([{",".join(f"{v:.1f}" for v in x_arr)}]),'
            f'"y":new Float32Array([{",".join(f"{v:.1f}" for v in y_arr)}]),'
            f'"uids":{uid_js}}}'
        )

    clusters_js = ",\n".join(cluster_data_js)
    ch_names_js = json.dumps(channel_names[:3])
    ch_colors_js = json.dumps(ch_colors[:3])
    scale_inv = 1.0 / scale_factor if scale_factor > 0 else 1.0

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Tissue ROI Viewer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d0d1a; color: #eee; font-family: system-ui, sans-serif; overflow: hidden; }}
  #container {{ position: relative; width: 100vw; height: 100vh; overflow: hidden; cursor: grab; }}
  #container.dragging {{ cursor: grabbing; }}
  #container.drawing {{ cursor: crosshair; }}
  canvas {{ position: absolute; top: 0; left: 0; image-rendering: pixelated; }}
  .sidebar {{
    position: fixed; right: 0; top: 0; width: 240px; height: 100vh;
    background: rgba(20,20,40,0.95); border-left: 1px solid #444;
    overflow-y: auto; z-index: 100; font-size: 12px; padding: 10px;
  }}
  .sidebar h3 {{ font-size: 13px; color: #aaa; margin: 8px 0 6px; border-bottom: 1px solid #333; padding-bottom: 4px; }}
  .sidebar h3:first-child {{ margin-top: 0; }}
  .ch-toggle {{ display: flex; align-items: center; gap: 6px; padding: 3px 4px; cursor: pointer; border-radius: 3px; user-select: none; }}
  .ch-toggle:hover {{ background: rgba(255,255,255,0.06); }}
  .ch-toggle.off {{ opacity: 0.3; }}
  .ch-toggle input {{ accent-color: var(--ch-color); }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; padding: 2px 4px; cursor: pointer; border-radius: 3px; user-select: none; }}
  .leg-item:hover {{ background: rgba(255,255,255,0.08); }}
  .leg-item.hidden {{ opacity: 0.3; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .leg-label {{ white-space: nowrap; font-size: 11px; }}
  .btn {{ background: #2a2a4a; border: 1px solid #555; color: #ccc; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; }}
  .btn:hover {{ background: #3a3a5a; }}
  .btn.active {{ background: #4466aa; border-color: #6688cc; color: #fff; }}
  .btn-row {{ display: flex; gap: 4px; flex-wrap: wrap; margin: 4px 0; }}
  .roi-item {{ display: flex; align-items: center; gap: 4px; padding: 3px 0; font-size: 11px; border-bottom: 1px solid #222; }}
  .roi-name {{ flex: 1; min-width: 0; color: #ccc; cursor: text; padding: 1px 3px; border-radius: 2px; }}
  .roi-name:hover {{ background: rgba(255,255,255,0.05); }}
  .roi-stats {{ color: #888; font-size: 10px; white-space: nowrap; }}
  .roi-del {{ color: #a44; cursor: pointer; font-weight: bold; padding: 0 4px; }}
  .roi-del:hover {{ color: #f66; }}
  .ctrl-row {{ display: flex; align-items: center; gap: 6px; margin: 4px 0; }}
  .ctrl-row label {{ font-size: 11px; color: #999; min-width: 50px; }}
  .size-slider {{ width: 80px; }}
  #controls {{
    position: fixed; left: 10px; bottom: 10px; background: rgba(26,26,46,0.9);
    border: 1px solid #444; border-radius: 8px; padding: 8px 12px; z-index: 100; font-size: 12px;
  }}
  #roi-detail {{
    position: fixed; left: 10px; top: 10px; background: rgba(20,20,40,0.95);
    border: 1px solid #444; border-radius: 8px; padding: 10px; z-index: 100; font-size: 11px;
    max-height: 50vh; overflow-y: auto; display: none; min-width: 200px;
  }}
  #roi-detail h4 {{ margin: 0 0 6px; color: #aaa; }}
  #roi-detail table {{ border-collapse: collapse; width: 100%; }}
  #roi-detail td, #roi-detail th {{ padding: 2px 6px; text-align: left; border-bottom: 1px solid #222; }}
  #roi-detail th {{ color: #888; font-weight: normal; }}
</style>
</head>
<body>
<div id="container">
  <canvas id="tissue"></canvas>
  <canvas id="dots"></canvas>
  <canvas id="draw"></canvas>
</div>

<div class="sidebar">
  <h3>Channels</h3>
  <div id="ch-toggles"></div>

  <h3>Groups ({group_field})</h3>
  <div id="leg-items"></div>
  <div style="margin-top:4px;font-size:10px;color:#666;">Click to toggle</div>

  <h3>Drawing</h3>
  <div class="btn-row">
    <button class="btn active" id="mode-pan" data-mode="pan">Pan</button>
    <button class="btn" id="mode-circle" data-mode="circle">Circle</button>
    <button class="btn" id="mode-rect" data-mode="rect">Rect</button>
    <button class="btn" id="mode-poly" data-mode="polygon">Poly</button>
  </div>
  <div style="font-size:10px;color:#555;margin:4px 0;">
    Circle/Rect: click+drag &middot; Poly: click vertices, dbl-click to close
  </div>
  <div id="roi-list"></div>
  <div class="btn-row" style="margin-top:6px;">
    <button class="btn" id="btn-download-roi">Download ROIs</button>
    <button class="btn" id="btn-clear-roi">Clear All</button>
  </div>
  <div class="ctrl-row">
    <input type="checkbox" id="roi-filter">
    <span style="font-size:11px;">Show only ROI cells</span>
  </div>
  <div id="roi-summary" style="font-size:10px;color:#777;margin-top:4px;"></div>

  <h3>Display</h3>
  <div class="ctrl-row">
    <label>Dot size</label>
    <input type="range" class="size-slider" id="dot-size" min="0.5" max="8" value="2" step="0.5">
    <span id="dot-val">2</span>
  </div>
  <div class="ctrl-row">
    <label>Opacity</label>
    <input type="range" class="size-slider" id="opacity" min="0" max="1" value="0.8" step="0.05">
    <span id="op-val">0.8</span>
  </div>
</div>

<div id="controls">
  Zoom: <span id="zoom-val">1.0</span>x &nbsp;|&nbsp;
  Cells: <span id="cell-count">0</span>
</div>

<div id="roi-detail"></div>

<script>
const IMG_W = {w}, IMG_H = {h};
const SCALE_INV = {scale_inv:.2f};
const PIXEL_SIZE = {pixel_size if pixel_size else 'null'};
const CH_NAMES = {ch_names_js};
const CH_COLORS = {ch_colors_js};
const HAS_UIDS = {'true' if has_uids else 'false'};
const CH_INDICES = {json.dumps(display_channels[:3])};
const clusters = [{clusters_js}];

// State
let zoom = 1, panX = 0, panY = 0;
let dragging = false, dragStartX, dragStartY, panStartX, panStartY;
let hidden = new Set();
let chEnabled = [true, true, true];
let channelsDirty = true;
let dotSize = 2, dotAlpha = 0.8;
let drawMode = 'pan';

// ROI state
let rois = [];
let roiCounter = 0;
let drawStart = null;
let polyVerts = [];
let roiFilterActive = false;

// Canvases
const tissueCanvas = document.getElementById('tissue');
const dotsCanvas = document.getElementById('dots');
const drawCanvas = document.getElementById('draw');
const container = document.getElementById('container');
const tissueCtx = tissueCanvas.getContext('2d');
const dotsCtx = dotsCanvas.getContext('2d');
const drawCtx = drawCanvas.getContext('2d');

// Load channel images
const chImages = [null, null, null];
let chLoaded = 0;
const CH_B64 = [{','.join(f'"{b}"' for b in ch_b64)}];

function loadChannels() {{
  for (let i = 0; i < 3; i++) {{
    chImages[i] = new Image();
    chImages[i].onload = () => {{
      chLoaded++;
      if (chLoaded === 3) {{ fitAndRender(); }}
    }};
    chImages[i].src = 'data:image/png;base64,' + CH_B64[i];
  }}
}}

// Offscreen canvas for compositing channels
const compCanvas = document.createElement('canvas');
compCanvas.width = IMG_W; compCanvas.height = IMG_H;
const compCtx = compCanvas.getContext('2d', {{ willReadFrequently: true }});

function compositeChannels() {{
  if (!channelsDirty) return;
  channelsDirty = false;
  compCtx.clearRect(0, 0, IMG_W, IMG_H);
  // Draw each enabled channel, extract pixels, composite additively
  const result = new Uint8ClampedArray(IMG_W * IMG_H * 4);
  for (let c = 0; c < 3; c++) {{
    if (!chEnabled[c] || !chImages[c]) continue;
    compCtx.clearRect(0, 0, IMG_W, IMG_H);
    compCtx.drawImage(chImages[c], 0, 0);
    const data = compCtx.getImageData(0, 0, IMG_W, IMG_H).data;
    for (let p = 0; p < IMG_W * IMG_H; p++) {{
      const v = data[p * 4]; // grayscale — R channel of the PNG
      result[p * 4 + c] = Math.min(255, result[p * 4 + c] + v);
      result[p * 4 + 3] = 255;
    }}
  }}
  const imgData = new ImageData(result, IMG_W, IMG_H);
  compCtx.putImageData(imgData, 0, 0);
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const cw = window.innerWidth - 240, ch = window.innerHeight;
  for (const c of [tissueCanvas, dotsCanvas, drawCanvas]) {{
    c.width = cw * dpr;
    c.height = ch * dpr;
    c.style.width = cw + 'px';
    c.style.height = ch + 'px';
    c.getContext('2d').scale(dpr, dpr);
  }}
}}

function render() {{
  const cw = window.innerWidth - 240, ch = window.innerHeight;
  tissueCtx.fillStyle = '#0d0d1a';
  tissueCtx.fillRect(0, 0, cw, ch);
  dotsCtx.clearRect(0, 0, cw, ch);
  drawCtx.clearRect(0, 0, cw, ch);

  // Tissue
  compositeChannels();
  tissueCtx.save();
  tissueCtx.translate(panX, panY);
  tissueCtx.scale(zoom, zoom);
  tissueCtx.imageSmoothingEnabled = zoom < 2;
  tissueCtx.drawImage(compCanvas, 0, 0, IMG_W, IMG_H);
  tissueCtx.restore();

  // Dots
  dotsCtx.save();
  dotsCtx.translate(panX, panY);
  dotsCtx.scale(zoom, zoom);
  dotsCtx.globalAlpha = dotAlpha;
  const r = dotSize / zoom;
  let total = 0;
  for (const cl of clusters) {{
    if (hidden.has(cl.label)) continue;
    dotsCtx.fillStyle = cl.color;
    for (let i = 0; i < cl.n; i++) {{
      const px = cl.x[i], py = cl.y[i];
      if (roiFilterActive && rois.length > 0 && !cellInAnyROI(px, py)) continue;
      dotsCtx.fillRect(px - r/2, py - r/2, r, r);
      total++;
    }}
  }}
  dotsCtx.restore();
  document.getElementById('cell-count').textContent = total.toLocaleString();
  document.getElementById('zoom-val').textContent = zoom.toFixed(1);

  // ROI shapes
  drawCtx.save();
  drawCtx.translate(panX, panY);
  drawCtx.scale(zoom, zoom);
  for (const roi of rois) {{
    drawROIShape(drawCtx, roi, 2/zoom);
  }}
  // In-progress polygon
  if (drawMode === 'polygon' && polyVerts.length > 0) {{
    drawCtx.strokeStyle = '#ffff00';
    drawCtx.lineWidth = 2/zoom;
    drawCtx.setLineDash([6/zoom, 4/zoom]);
    drawCtx.beginPath();
    drawCtx.moveTo(polyVerts[0][0], polyVerts[0][1]);
    for (let i = 1; i < polyVerts.length; i++) drawCtx.lineTo(polyVerts[i][0], polyVerts[i][1]);
    drawCtx.stroke();
    drawCtx.setLineDash([]);
    for (const v of polyVerts) {{
      drawCtx.fillStyle = '#ffff00';
      drawCtx.fillRect(v[0]-3/zoom, v[1]-3/zoom, 6/zoom, 6/zoom);
    }}
  }}
  drawCtx.restore();
}}

function drawROIShape(ctx, roi, lw) {{
  ctx.strokeStyle = '#ffcc00';
  ctx.lineWidth = lw;
  ctx.setLineDash([]);
  if (roi.type === 'circle') {{
    ctx.beginPath();
    ctx.arc(roi.data.cx, roi.data.cy, roi.data.r, 0, Math.PI*2);
    ctx.stroke();
  }} else if (roi.type === 'rect') {{
    const x = Math.min(roi.data.x1, roi.data.x2), y = Math.min(roi.data.y1, roi.data.y2);
    const w = Math.abs(roi.data.x2 - roi.data.x1), h = Math.abs(roi.data.y2 - roi.data.y1);
    ctx.strokeRect(x, y, w, h);
  }} else if (roi.type === 'polygon') {{
    ctx.beginPath();
    ctx.moveTo(roi.data.verts[0][0], roi.data.verts[0][1]);
    for (let i = 1; i < roi.data.verts.length; i++) ctx.lineTo(roi.data.verts[i][0], roi.data.verts[i][1]);
    ctx.closePath();
    ctx.stroke();
  }}
  // Label
  ctx.fillStyle = '#ffcc00';
  ctx.font = (11/zoom) + 'px system-ui';
  let lx, ly;
  if (roi.type === 'circle') {{ lx = roi.data.cx; ly = roi.data.cy - roi.data.r - 4/zoom; }}
  else if (roi.type === 'rect') {{ lx = Math.min(roi.data.x1,roi.data.x2); ly = Math.min(roi.data.y1,roi.data.y2) - 4/zoom; }}
  else if (roi.type === 'polygon') {{ lx = roi.data.verts[0][0]; ly = roi.data.verts[0][1] - 4/zoom; }}
  if (lx !== undefined) ctx.fillText(roi.name, lx, ly);
}}

// ROI geometry
function pointInCircle(px, py, cx, cy, r) {{ return (px-cx)**2 + (py-cy)**2 <= r*r; }}
function pointInRect(px, py, x1, y1, x2, y2) {{
  const minx = Math.min(x1,x2), maxx = Math.max(x1,x2);
  const miny = Math.min(y1,y2), maxy = Math.max(y1,y2);
  return px >= minx && px <= maxx && py >= miny && py <= maxy;
}}
function pointInPoly(px, py, verts) {{
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {{
    const xi = verts[i][0], yi = verts[i][1], xj = verts[j][0], yj = verts[j][1];
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi)
      inside = !inside;
  }}
  return inside;
}}
function pointInROI(px, py, roi) {{
  if (roi.type === 'circle') return pointInCircle(px, py, roi.data.cx, roi.data.cy, roi.data.r);
  if (roi.type === 'rect') return pointInRect(px, py, roi.data.x1, roi.data.y1, roi.data.x2, roi.data.y2);
  if (roi.type === 'polygon') return pointInPoly(px, py, roi.data.verts);
  return false;
}}
function cellInAnyROI(px, py) {{
  for (const roi of rois) if (pointInROI(px, py, roi)) return true;
  return false;
}}

// Screen to image coords
function screenToImg(sx, sy) {{ return [(sx - panX) / zoom, (sy - panY) / zoom]; }}

// Mouse events on draw canvas
let mouseDown = false;
drawCanvas.addEventListener('mousedown', e => {{
  if (e.button !== 0) return;
  const [ix, iy] = screenToImg(e.clientX, e.clientY);
  if (drawMode === 'pan') {{
    dragging = true;
    container.classList.add('dragging');
    dragStartX = e.clientX; dragStartY = e.clientY;
    panStartX = panX; panStartY = panY;
  }} else if (drawMode === 'polygon') {{
    polyVerts.push([ix, iy]);
    render();
  }} else {{
    mouseDown = true;
    drawStart = {{ x: ix, y: iy }};
  }}
}});

drawCanvas.addEventListener('mousemove', e => {{
  if (drawMode === 'pan' && dragging) {{
    panX = panStartX + (e.clientX - dragStartX);
    panY = panStartY + (e.clientY - dragStartY);
    render();
  }}
}});

drawCanvas.addEventListener('mouseup', e => {{
  if (drawMode === 'pan') {{
    dragging = false;
    container.classList.remove('dragging');
    return;
  }}
  if (!mouseDown || drawMode === 'polygon') return;
  mouseDown = false;
  const [ix, iy] = screenToImg(e.clientX, e.clientY);
  if (drawMode === 'circle') {{
    const r = Math.sqrt((ix-drawStart.x)**2 + (iy-drawStart.y)**2);
    if (r > 2) addROI('circle', {{ cx: drawStart.x, cy: drawStart.y, r }});
  }} else if (drawMode === 'rect') {{
    const w = Math.abs(ix - drawStart.x), h = Math.abs(iy - drawStart.y);
    if (w > 2 && h > 2) addROI('rect', {{ x1: drawStart.x, y1: drawStart.y, x2: ix, y2: iy }});
  }}
  drawStart = null;
}});

drawCanvas.addEventListener('dblclick', e => {{
  if (drawMode === 'polygon' && polyVerts.length >= 3) {{
    addROI('polygon', {{ verts: polyVerts.slice() }});
    polyVerts = [];
  }}
}});

drawCanvas.addEventListener('wheel', e => {{
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  panX = e.clientX - factor * (e.clientX - panX);
  panY = e.clientY - factor * (e.clientY - panY);
  zoom *= factor;
  zoom = Math.max(0.1, Math.min(50, zoom));
  render();
}}, {{passive: false}});

// ROI management
function addROI(type, data) {{
  roiCounter++;
  rois.push({{ id: 'ROI_' + roiCounter, type, data, name: 'ROI_' + roiCounter }});
  updateROIList();
  render();
}}

function deleteROI(id) {{
  rois = rois.filter(r => r.id !== id);
  updateROIList();
  render();
}}

function updateROIList() {{
  const div = document.getElementById('roi-list');
  div.innerHTML = '';
  for (const roi of rois) {{
    const item = document.createElement('div');
    item.className = 'roi-item';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'roi-name';
    nameSpan.contentEditable = true;
    nameSpan.textContent = roi.name;
    nameSpan.onblur = () => {{ roi.name = nameSpan.textContent.trim() || roi.id; }};
    nameSpan.onkeydown = (e) => {{ if (e.key === 'Enter') {{ e.preventDefault(); nameSpan.blur(); }} }};

    const statsSpan = document.createElement('span');
    statsSpan.className = 'roi-stats';
    // Count cells
    let cnt = 0;
    for (const cl of clusters) {{
      if (hidden.has(cl.label)) continue;
      for (let i = 0; i < cl.n; i++) {{
        if (pointInROI(cl.x[i], cl.y[i], roi)) cnt++;
      }}
    }}
    statsSpan.textContent = cnt.toLocaleString();

    const viewBtn = document.createElement('span');
    viewBtn.textContent = '\\u25B6';
    viewBtn.style.cssText = 'cursor:pointer;padding:0 3px;color:#6a6;font-size:10px;';
    viewBtn.title = 'Show breakdown';
    viewBtn.onclick = () => showROIDetail(roi);

    const delBtn = document.createElement('span');
    delBtn.className = 'roi-del';
    delBtn.textContent = '\\u00d7';
    delBtn.onclick = () => deleteROI(roi.id);

    item.appendChild(nameSpan);
    item.appendChild(statsSpan);
    item.appendChild(viewBtn);
    item.appendChild(delBtn);
    div.appendChild(item);
  }}
  document.getElementById('roi-summary').textContent = rois.length ? rois.length + ' ROI(s)' : '';
}}

function showROIDetail(roi) {{
  const det = document.getElementById('roi-detail');
  let html = '<h4>' + roi.name + ' breakdown</h4><table><tr><th>Group</th><th>Count</th><th>%</th></tr>';
  let total = 0;
  const counts = {{}};
  for (const cl of clusters) {{
    let c = 0;
    for (let i = 0; i < cl.n; i++) {{
      if (pointInROI(cl.x[i], cl.y[i], roi)) c++;
    }}
    counts[cl.label] = c;
    total += c;
  }}
  for (const cl of clusters) {{
    const c = counts[cl.label];
    if (c === 0) continue;
    const pct = total > 0 ? (100*c/total).toFixed(1) : '0';
    html += '<tr><td><span style="color:' + cl.color + '">\\u25CF</span> ' + cl.label + '</td><td>' + c.toLocaleString() + '</td><td>' + pct + '%</td></tr>';
  }}
  html += '<tr style="border-top:1px solid #444;font-weight:bold"><td>Total</td><td>' + total.toLocaleString() + '</td><td>100%</td></tr></table>';
  html += '<div style="margin-top:6px;font-size:10px;color:#666;">Click elsewhere to close</div>';
  det.innerHTML = html;
  det.style.display = 'block';
}}

function downloadROIs() {{
  const out = {{ rois: [], metadata: {{ group_field: '{group_field}', scale_inv: SCALE_INV, pixel_size: PIXEL_SIZE }} }};
  for (const roi of rois) {{
    const entry = {{ id: roi.id, type: roi.type, name: roi.name }};
    // Geometry in CZI pixel coordinates (multiply back by scale_inv)
    if (roi.type === 'circle') {{
      entry.center_px = [roi.data.cx * SCALE_INV, roi.data.cy * SCALE_INV];
      entry.radius_px = roi.data.r * SCALE_INV;
      if (PIXEL_SIZE) {{
        entry.center_um = [roi.data.cx * SCALE_INV * PIXEL_SIZE, roi.data.cy * SCALE_INV * PIXEL_SIZE];
        entry.radius_um = roi.data.r * SCALE_INV * PIXEL_SIZE;
      }}
    }} else if (roi.type === 'rect') {{
      entry.min_px = [Math.min(roi.data.x1,roi.data.x2)*SCALE_INV, Math.min(roi.data.y1,roi.data.y2)*SCALE_INV];
      entry.max_px = [Math.max(roi.data.x1,roi.data.x2)*SCALE_INV, Math.max(roi.data.y1,roi.data.y2)*SCALE_INV];
    }} else if (roi.type === 'polygon') {{
      entry.vertices_px = roi.data.verts.map(v => [v[0]*SCALE_INV, v[1]*SCALE_INV]);
    }}
    // Collect cell UIDs inside this ROI
    if (HAS_UIDS) {{
      const uids = [];
      for (const cl of clusters) {{
        if (!cl.uids) continue;
        for (let i = 0; i < cl.n; i++) {{
          if (pointInROI(cl.x[i], cl.y[i], roi)) uids.push(cl.uids[i]);
        }}
      }}
      entry.cell_uids = uids;
      entry.cell_count = uids.length;
    }}
    // Group breakdown
    const breakdown = {{}};
    for (const cl of clusters) {{
      let c = 0;
      for (let i = 0; i < cl.n; i++) {{
        if (pointInROI(cl.x[i], cl.y[i], roi)) c++;
      }}
      if (c > 0) breakdown[cl.label] = c;
    }}
    entry.group_counts = breakdown;
    out.rois.push(entry);
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'tissue_rois.json';
  a.click();
  URL.revokeObjectURL(a.href);
}}

// Init channel toggles
function initChannelToggles() {{
  const div = document.getElementById('ch-toggles');
  for (let i = 0; i < 3; i++) {{
    const row = document.createElement('div');
    row.className = 'ch-toggle';
    row.style.setProperty('--ch-color', CH_COLORS[i]);
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.onchange = () => {{ chEnabled[i] = cb.checked; channelsDirty = true; row.classList.toggle('off', !cb.checked); render(); }};
    const lbl = document.createElement('span');
    lbl.style.color = CH_COLORS[i];
    lbl.textContent = CH_NAMES[i] + ' (ch' + CH_INDICES[i] + ')';
    row.appendChild(cb);
    row.appendChild(lbl);
    div.appendChild(row);
  }}
}}

// Init legend
function initLegend() {{
  const div = document.getElementById('leg-items');
  for (const cl of clusters) {{
    const item = document.createElement('div');
    item.className = 'leg-item';
    item.innerHTML = '<span class="leg-dot" style="background:' + cl.color + '"></span>' +
      '<span class="leg-label">' + cl.label + ' (' + cl.n.toLocaleString() + ')</span>';
    item.onclick = () => {{
      if (hidden.has(cl.label)) hidden.delete(cl.label); else hidden.add(cl.label);
      item.classList.toggle('hidden');
      render();
    }};
    div.appendChild(item);
  }}
}}

// Mode buttons
document.querySelectorAll('[data-mode]').forEach(btn => {{
  btn.onclick = () => {{
    drawMode = btn.dataset.mode;
    polyVerts = [];
    document.querySelectorAll('[data-mode]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    container.classList.toggle('drawing', drawMode !== 'pan');
    render();
  }};
}});

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
document.getElementById('roi-filter').onchange = e => {{
  roiFilterActive = e.target.checked;
  render();
}};
document.getElementById('btn-download-roi').onclick = downloadROIs;
document.getElementById('btn-clear-roi').onclick = () => {{
  rois = []; roiCounter = 0; updateROIList(); render();
  document.getElementById('roi-detail').style.display = 'none';
}};

// Close detail panel on click outside
document.addEventListener('click', e => {{
  const det = document.getElementById('roi-detail');
  if (det.style.display === 'block' && !det.contains(e.target)) det.style.display = 'none';
}});

window.addEventListener('resize', () => {{ resize(); render(); }});

// Keyboard: Escape cancels polygon
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    polyVerts = [];
    render();
  }}
}});

function fitAndRender() {{
  resize();
  const cw = window.innerWidth - 240, ch = window.innerHeight;
  zoom = Math.min(cw / IMG_W, ch / IMG_H) * 0.95;
  panX = (cw - IMG_W * zoom) / 2;
  panY = (ch - IMG_H * zoom) / 2;
  render();
}}

// Boot
initChannelToggles();
initLegend();
loadChannels();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  Saved interactive HTML: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tissue overlay with fluorescence channels, cell overlay, and ROI drawing"
    )
    parser.add_argument("--czi-path", required=True, help="Path to CZI file")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--detections", help="Detections JSON (preferred — includes UIDs and features)"
    )
    input_group.add_argument("--spatial-csv", help="Spatial CSV (legacy)")
    parser.add_argument(
        "--display-channels",
        default="1,0",
        help='Channel indices for R,G,B (e.g., "2,0,1" for tdTom=R, nuc=G, NeuN=B)',
    )
    parser.add_argument(
        "--channel-names",
        default=None,
        help='Comma-separated channel names (e.g., "tdTomato,nuc488,NeuN"). Auto-detected from CZI if omitted.',
    )
    parser.add_argument(
        "--group-field",
        default="marker_profile",
        help="Field to color cells by (default: marker_profile). Use cluster_label for spatial CSV.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.02,
        help="CZI read scale factor (0.02 = 2%%)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dot-size", type=float, default=2, help="Dot size in pixels (PNG)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Dot opacity (PNG)")
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI")
    parser.add_argument("--no-html", action="store_true", help="Skip interactive HTML generation")
    parser.add_argument("--no-png", action="store_true", help="Skip static PNG generation")
    parser.add_argument("--scene", type=int, default=0, help="CZI scene index (0-based)")
    parser.add_argument(
        "--marker-filter",
        default=None,
        help='Filter detections by marker class (e.g., "MSLN_class==positive")',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display_channels = [int(c) for c in args.display_channels.split(",")]
    # Pad to 3 channels for RGB
    while len(display_channels) < 3:
        display_channels.append(-1)

    # Channel names
    if args.channel_names:
        channel_names = args.channel_names.split(",")
    else:
        # Try to get from CZI filename
        czi_name = Path(args.czi_path).stem
        try:
            from segmentation.io.czi_loader import parse_markers_from_filename

            markers = parse_markers_from_filename(czi_name)
            # Map display channel index to marker name
            channel_names = []
            for ch_idx in display_channels[:3]:
                if ch_idx < 0:
                    channel_names.append("(off)")
                else:
                    found = False
                    # Best-effort: use positional match (filename order ≠ CZI order,
                    # but it's the best we can do without CZI metadata for a display label)
                    if ch_idx < len(markers):
                        channel_names.append(markers[ch_idx].get("name", f"ch{ch_idx}"))
                        found = True
                    if not found:
                        channel_names.append(f"ch{ch_idx}")
        except Exception:
            channel_names = [f"ch{c}" if c >= 0 else "(off)" for c in display_channels[:3]]

    # 1. Read CZI channels
    print(f"Reading CZI channels at {args.scale_factor*100:.0f}% scale...", flush=True)
    valid_channels = [ch for ch in display_channels if ch >= 0]
    raw_channels, pixel_size = read_czi_thumbnail(
        args.czi_path, valid_channels, args.scale_factor, scene=args.scene
    )

    # Map to 3-channel array (R, G, B positions)
    channel_arrays = []
    h, w = raw_channels[0].shape
    for i, ch_idx in enumerate(display_channels[:3]):
        if ch_idx >= 0 and ch_idx in valid_channels:
            src_idx = valid_channels.index(ch_idx)
            channel_arrays.append(raw_channels[src_idx])
        else:
            channel_arrays.append(np.zeros((h, w), dtype=np.uint8))

    del raw_channels
    gc.collect()

    print(f"  Thumbnail: {w}x{h} px", flush=True)
    if pixel_size:
        print(f"  Pixel size: {pixel_size:.4f} um/px", flush=True)

    # 2. Load cell data
    has_uids = False
    if args.detections:
        df = load_detections_json(
            args.detections, args.group_field, marker_filter=args.marker_filter
        )
        has_uids = "uid" in df.columns and df["uid"].notna().any()
        cluster_field = "group"
    else:
        print(f"Loading spatial data from {args.spatial_csv}...", flush=True)
        df = pd.read_csv(args.spatial_csv)
        cluster_field = args.group_field
        if cluster_field not in df.columns:
            print(f"ERROR: field '{cluster_field}' not in columns: {list(df.columns)}")
            sys.exit(1)
        # Rename for consistency
        df["group"] = df[cluster_field].astype(str)
        cluster_field = "group"
        print(f"  {len(df)} cells", flush=True)

    # 3. Generate color map
    color_map = generate_cluster_colors(df[cluster_field].values, cluster_field)
    print(f"  {len(color_map)} groups", flush=True)

    # 4. Generate PNG
    if not args.no_png:
        print("Generating overlay PNG...", flush=True)
        png_path = output_dir / f"tissue_overlay_{args.group_field}.png"
        generate_overlay_png(
            channel_arrays,
            display_channels,
            df,
            cluster_field,
            color_map,
            args.scale_factor,
            png_path,
            dot_size=args.dot_size,
            alpha=args.alpha,
            dpi=args.dpi,
        )

    # 5. Generate interactive HTML
    if not args.no_html:
        print("Generating interactive HTML...", flush=True)
        html_path = output_dir / f"tissue_overlay_{args.group_field}.html"
        generate_interactive_html(
            channel_arrays,
            display_channels[:3],
            channel_names[:3],
            df,
            cluster_field,
            color_map,
            args.scale_factor,
            html_path,
            pixel_size=pixel_size,
            has_uids=has_uids,
        )

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
