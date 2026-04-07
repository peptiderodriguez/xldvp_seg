#!/usr/bin/env python3
"""Generate multi-slide spatial viewer HTML from classified detections.

Loads classified detections from multiple slides and renders a self-contained
HTML with one canvas panel per slide in a responsive grid, cells colored by
marker class. Supports interactive ROI drawing (circle, rectangle, freeform
polygon) with JSON export, focus view (double-click to zoom in on one slide),
and per-panel independent pan/zoom.

Data is embedded as base64-encoded Float32Array + Uint8Array for compact
binary transfer.  Canvas 2D rendering handles 50k+ cells per slide.

Usage:
    # Auto-discover from pipeline output directory
    python scripts/generate_multi_slide_spatial_viewer.py \\
        --input-dir /path/to/output/ \\
        --detection-glob "cell_detections_classified.json" \\
        --group-field tdTomato_class \\
        --title "Senescence tdTomato Spatial Overview" \\
        --output spatial_viewer.html

    # Explicit list of detection files
    python scripts/generate_multi_slide_spatial_viewer.py \\
        --detections slide1/cell_detections_classified.json \\
                     slide2/cell_detections_classified.json \\
        --group-field tdTomato_class \\
        --output spatial_viewer.html
"""

import argparse
import html as html_mod
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from xldvp_seg.visualization.colors import assign_group_colors
from xldvp_seg.visualization.data_loading import (
    apply_top_n_filtering,
    discover_slides,
    load_slide_data,
)
from xldvp_seg.visualization.encoding import (
    build_contour_js_data,
    safe_json,
)
from xldvp_seg.visualization.fluorescence import (
    encode_channel_b64,
    parse_scene_index,
    read_czi_thumbnail_channels,
)
from xldvp_seg.visualization.graph_patterns import compute_graph_patterns
from xldvp_seg.visualization.html_builder import (
    build_group_index,
    collect_auto_eps,
    compact_region_data,
    serialize_slide_positions,
)

# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(
    slides_data,
    output_path,
    color_map,
    title,
    group_field,
    default_min_cells=10,
    min_hull_cells=24,
    has_regions=False,
    has_multiscale=False,
    scale_keys=None,
    fluor_data=None,
    contour_data=None,
    ch_names=None,
):
    """Generate self-contained scrollable HTML with focus view, ROI, and DBSCAN clustering.

    Data is embedded as base64-encoded TypedArrays for compact transfer.
    Each slide stores a single Float32Array of interleaved [x0,y0,x1,y1,...]
    positions and a Uint8Array of group indices.

    Args:
        slides_data: List of (slide_name, data_dict) tuples.
        output_path: Output HTML file path.
        color_map: Dict of group_label -> hex color.
        title: Page title.
        group_field: Group field name (for metadata export).
        default_min_cells: Default DBSCAN min_samples for clustering.
        min_hull_cells: Min cells in cluster to draw convex hull.
        fluor_data: Optional dict {slide_name: {'channels': [b64_png,...],
            'names': [...], 'width': w, 'height': h, 'scale': s,
            'mosaic_x': mx, 'mosaic_y': my, 'pixel_size': ps}}.
            Images are greyscale PNGs, composited additively in RGB order.
        contour_data: Optional dict {slide_name: list of contour dicts}
            from build_contour_js_data(). Coordinates are in um.
        ch_names: Optional list of 3 channel names for toggle button labels
            (e.g. ['PM', 'nuc', 'SMA']). Defaults to ['Ch0','Ch1','Ch2'].
    """
    title_escaped = html_mod.escape(title)

    # Build group label -> index mapping (consistent across all slides)
    group_labels, group_to_idx, color_map = build_group_index(
        slides_data, color_map, max_groups=255
    )

    # Serialize each slide as base64 binary data
    slides_meta, slides_b64_positions, slides_b64_groups, slide_names_ordered = (
        serialize_slide_positions(slides_data, group_to_idx)
    )

    # Build per-slide per-group auto_eps for DBSCAN clustering
    slides_auto_eps = collect_auto_eps(slides_data, group_labels)

    # Serialize region data per-slide (compact format for embedding)
    slides_region_data = compact_region_data(slides_data)

    # Build legend info
    legend_items = []
    total_counts = {}
    for _, data in slides_data:
        for g in data["groups"]:
            total_counts[g["label"]] = total_counts.get(g["label"], 0) + g["n"]

    for lbl in group_labels:
        legend_items.append(
            {
                "label": lbl,
                "color": color_map[lbl],
                "count": total_counts.get(lbl, 0),
            }
        )

    n_slides = len(slides_data)
    is_single = n_slides == 1
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Resolve channel names (default Ch0/Ch1/Ch2)
    has_fluor = bool(fluor_data)
    has_contours = bool(contour_data)
    if ch_names is None:
        ch_names = ["Ch0", "Ch1", "Ch2"]
    ch_names = (list(ch_names) + ["Ch0", "Ch1", "Ch2"])[:3]

    # --- Build conditional sidebar sections ---
    # Build regions sidebar (conditional on --graph-patterns)
    regions_sidebar_html = ""
    if has_regions:
        scale_slider_html = ""
        if has_multiscale and scale_keys:
            mid = len(scale_keys) // 2
            scale_slider_html = (
                '      <div class="ctrl-row">\n'
                "        <label>Scale</label>\n"
                f'        <input type="range" id="region-scale" min="0" max="{len(scale_keys)-1}" value="{mid}" step="1">\n'
                f'        <span class="val" id="region-scale-val">{scale_keys[mid]} &micro;m</span>\n'
                "      </div>\n"
            )
        regions_sidebar_html = (
            "    <!-- Regions (graph patterns) -->\n"
            '    <div class="sidebar-section">\n'
            "      <h3>Regions</h3>\n"
            '      <div class="ctrl-row">\n'
            '        <label style="min-width:auto"><input type="checkbox" id="show-regions" checked> Show</label>\n'
            '        <label style="min-width:auto"><input type="checkbox" id="show-region-labels" checked> Labels</label>\n'
            '        <label style="min-width:auto"><input type="checkbox" id="show-region-bnd" checked> Borders</label>\n'
            "      </div>\n"
            '      <div class="ctrl-row">\n'
            "        <label>Opacity</label>\n"
            '        <input type="range" id="region-opacity" min="0" max="0.8" value="0.25" step="0.05">\n'
            '        <span class="val" id="region-op-val">0.25</span>\n'
            "      </div>\n" + scale_slider_html + "    </div>\n"
        )

    # Build fluorescence/contour sidebar (conditional on --czi-path/--czi-dir/--contours)
    fluor_sidebar_html = ""
    if has_fluor or has_contours:
        ch0_name = html_mod.escape(ch_names[0])
        ch1_name = html_mod.escape(ch_names[1])
        ch2_name = html_mod.escape(ch_names[2])
        fluor_sidebar_html = "    <!-- Fluorescence & Contours -->\n"
        fluor_sidebar_html += '    <div class="sidebar-section">\n'
        fluor_sidebar_html += "      <h3>Fluorescence</h3>\n"
        if has_fluor:
            fluor_sidebar_html += (
                '      <div class="ctrl-row">\n'
                '        <label style="min-width:auto"><input type="checkbox" id="show-fluor" checked> Show</label>\n'
                "      </div>\n"
                '      <div class="ctrl-row">\n'
                "        <label>Opacity</label>\n"
                '        <input type="range" id="fluor-opacity" min="0" max="1" value="0.8" step="0.05">\n'
                '        <span class="val" id="fluor-op-val">0.80</span>\n'
                "      </div>\n"
                '      <div class="btn-row" style="margin:4px 0;">\n'
                f'        <button class="btn active" id="btn-ch0" style="border-left:3px solid #ff4444">{ch0_name}</button>\n'
                f'        <button class="btn active" id="btn-ch1" style="border-left:3px solid #44ff44">{ch1_name}</button>\n'
                f'        <button class="btn active" id="btn-ch2" style="border-left:3px solid #4488ff">{ch2_name}</button>\n'
                "      </div>\n"
            )
        if has_contours:
            fluor_sidebar_html += (
                '      <div class="ctrl-row">\n'
                '        <label style="min-width:auto"><input type="checkbox" id="show-contours" checked> Contours</label>\n'
                "      </div>\n"
            )
        fluor_sidebar_html += (
            '      <div class="ctrl-row">\n'
            '        <label style="min-width:auto"><input type="checkbox" id="show-dots" checked> Dots</label>\n'
            "      </div>\n"
        )
        fluor_sidebar_html += "    </div>\n"

    # --- Build the HTML ---
    html_parts = []
    html_parts.append(
        f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title_escaped}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #eee; font-family: system-ui, -apple-system, sans-serif; overflow: hidden; }}
  #app {{ display: flex; width: 100vw; height: 100vh; }}

  /* Sidebar */
  #sidebar {{
    width: 280px; min-width: 240px; background: rgba(26,26,46,0.97);
    border-right: 1px solid #333; overflow-y: auto; padding: 12px;
    display: flex; flex-direction: column; gap: 12px; z-index: 20;
  }}
  #sidebar h2 {{ font-size: 14px; color: #ddd; margin-bottom: 2px; }}
  #sidebar h3 {{ font-size: 12px; color: #999; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .sidebar-section {{ border-top: 1px solid #333; padding-top: 10px; }}

  /* Legend */
  .leg-item {{
    display: flex; align-items: center; gap: 6px; padding: 3px 6px;
    cursor: pointer; border-radius: 4px; user-select: none; font-size: 12px;
  }}
  .leg-item:hover {{ background: rgba(255,255,255,0.06); }}
  .leg-item.hidden {{ opacity: 0.25; text-decoration: line-through; }}
  .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  .leg-label {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .leg-count {{ color: #777; font-size: 10px; flex-shrink: 0; }}

  /* Buttons */
  .btn {{
    background: #2a2a4a; border: 1px solid #555; color: #ccc; padding: 4px 10px;
    border-radius: 4px; cursor: pointer; font-size: 11px; transition: background 0.15s;
  }}
  .btn:hover {{ background: #3a3a5a; }}
  .btn.active {{ background: #3a5a3a; border-color: #6a6; color: #fff; }}
  .btn-row {{ display: flex; gap: 4px; flex-wrap: wrap; }}
  .mode-btn {{ min-width: 50px; text-align: center; }}

  /* Controls */
  .ctrl-row {{ display: flex; align-items: center; gap: 6px; font-size: 11px; margin-bottom: 5px; }}
  .ctrl-row label {{ min-width: 56px; color: #aaa; }}
  .ctrl-row input[type=range] {{ flex: 1; min-width: 60px; }}
  .ctrl-row .val {{ color: #ccc; min-width: 28px; text-align: right; }}

  /* Slide select */
  select {{
    background: #1a1a2e; color: #ccc; border: 1px solid #555; padding: 4px 8px;
    border-radius: 4px; font-size: 11px; width: 100%;
  }}

  /* Main area */
  #main-area {{ flex: 1; position: relative; overflow: hidden; }}

  /* Grid view */
  #grid {{
    width: 100%; height: 100%;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    grid-auto-rows: 400px;
    gap: 3px; padding: 3px;
    overflow-y: auto;
  }}
  #grid.single-slide {{
    grid-template-columns: 1fr;
    grid-auto-rows: 100%;
    gap: 0; padding: 0;
  }}

  /* Focus view (hidden by default) */
  #focus-view {{
    display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    z-index: 10;
  }}
  #focus-view.active {{ display: block; }}
  #focus-back {{
    position: absolute; top: 8px; left: 8px; z-index: 15;
    background: rgba(30,30,50,0.9); border: 1px solid #555; color: #ccc;
    padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px;
  }}
  #focus-back:hover {{ background: rgba(50,50,80,0.95); }}
  #focus-label {{
    position: absolute; top: 8px; left: 50%; transform: translateX(-50%); z-index: 15;
    font-size: 13px; color: #ccc; background: rgba(26,26,46,0.85);
    padding: 4px 12px; border-radius: 4px; pointer-events: none;
  }}

  /* Panel styling */
  .panel {{
    position: relative; overflow: hidden; background: #111122;
    border: 1px solid #333; border-radius: 4px; cursor: grab;
  }}
  .panel.dragging {{ cursor: grabbing; }}
  .panel canvas {{ position: absolute; top: 0; left: 0; }}
  .panel .draw-overlay {{ z-index: 5; pointer-events: none; }}
  .panel.draw-mode .draw-overlay {{ pointer-events: auto; cursor: crosshair; }}
  .panel-label {{
    position: absolute; top: 4px; left: 6px; z-index: 10;
    font-size: 11px; color: #ccc; background: rgba(17,17,34,0.85);
    padding: 2px 8px; border-radius: 3px; pointer-events: none;
  }}
  .panel-count {{
    position: absolute; bottom: 4px; left: 6px; z-index: 10;
    font-size: 10px; color: #777; pointer-events: none;
  }}
  .panel-measure {{
    position: absolute; bottom: 4px; right: 6px; z-index: 10;
    font-size: 10px; color: #0f8; pointer-events: none; display: none;
  }}

  /* ROI list */
  #roi-list {{ max-height: 180px; overflow-y: auto; }}
  .roi-item {{
    display: flex; align-items: center; gap: 4px; padding: 3px 4px;
    font-size: 11px; border-radius: 3px;
  }}
  .roi-item:hover {{ background: rgba(255,255,255,0.05); }}
  .roi-item .roi-name {{
    flex: 1; min-width: 0; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; cursor: text; color: #ddd; padding: 1px 3px;
    border-radius: 2px;
  }}
  .roi-item .roi-name:focus {{ outline: 1px solid #555; background: #1a1a2e; }}
  .roi-item .roi-category {{ font-size:9px; color:#8a8; cursor:text; padding:1px 3px; border-radius:2px; max-width:55px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; border:1px dashed #444; margin:0 3px; }}
  .roi-item .roi-category:empty::before {{ content:'cat'; color:#555; }}
  .roi-item .roi-category:focus {{ outline:1px solid #555; background:#1a1a2e; }}
  .roi-item .roi-stats {{ color: #888; font-size: 10px; white-space: nowrap; }}
  .roi-del {{ cursor: pointer; color: #a55; font-size: 14px; line-height: 1; }}
  .roi-del:hover {{ color: #f66; }}

  /* Help text */
  .help-text {{ font-size: 10px; color: #555; line-height: 1.4; }}
</style>
</head>
<body>
<div id="app">
  <button id="floating-reset-zoom" style="position:fixed;bottom:20px;right:20px;z-index:1000;padding:8px 16px;background:#333;color:#fff;border:1px solid #555;border-radius:6px;cursor:pointer;font-size:13px;opacity:0.85;" onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.85'">Reset Zoom</button>
  <div id="sidebar">
    <div>
      <h2>{title_escaped}</h2>
      <div style="font-size:10px;color:#666;margin-bottom:6px;">
        {n_slides} slide{'s' if n_slides != 1 else ''} &middot;
        {sum(d['n_cells'] for _, d in slides_data):,} cells
      </div>
    </div>

    <!-- Legend -->
    <div>
      <h3>Legend</h3>
      <div id="leg-items"></div>
      <div class="btn-row" style="margin-top:6px;">
        <button class="btn" id="btn-show-all">All on</button>
        <button class="btn" id="btn-hide-all">All off</button>
      </div>
    </div>

    <!-- Slide navigation -->
    <div class="sidebar-section">
      <h3>Slides</h3>
      <select id="slide-select">
        <option value="">Jump to slide...</option>
      </select>
    </div>

    <!-- Display controls -->
    <div class="sidebar-section">
      <h3>Display</h3>
      <div class="ctrl-row">
        <label>Dot size</label>
        <input type="range" id="dot-size" min="1" max="8" value="3" step="0.5">
        <span class="val" id="dot-val">3</span>
      </div>
      <div class="ctrl-row">
        <label>Opacity</label>
        <input type="range" id="opacity" min="0.1" max="1" value="0.7" step="0.05">
        <span class="val" id="op-val">0.70</span>
      </div>
      <div class="ctrl-row">
        <button class="btn" id="btn-reset-zoom">Reset Zoom</button>
      </div>
    </div>

    <!-- KDE Density -->
    <div class="sidebar-section">
      <h3>KDE Density</h3>
      <div class="ctrl-row">
        <label style="min-width:auto"><input type="checkbox" id="show-kde"> Show</label>
      </div>
      <div class="ctrl-row">
        <label>Bandwidth</label>
        <input type="range" id="kde-bw" min="0" max="9" value="3" step="1">
        <span class="val" id="kde-bw-val">300 &micro;m</span>
      </div>
      <div class="ctrl-row">
        <label>Levels</label>
        <input type="range" id="kde-levels" min="1" max="6" value="3" step="1">
        <span class="val" id="kde-levels-val">3</span>
      </div>
      <div class="ctrl-row">
        <label>Opacity</label>
        <input type="range" id="kde-opacity" min="0.1" max="1.0" value="0.5" step="0.05">
        <span class="val" id="kde-op-val">0.50</span>
      </div>
      <div class="ctrl-row">
        <label style="min-width:auto"><input type="checkbox" id="kde-fill"> Fill</label>
        <label style="min-width:auto"><input type="checkbox" id="kde-lines"> Lines</label>
      </div>
    </div>

{regions_sidebar_html}
{fluor_sidebar_html}
    <!-- Clustering -->
    <div class="sidebar-section">
      <h3>Clustering</h3>
      <div class="ctrl-row">
        <label>Eps scale</label>
        <input type="range" id="eps-slider" min="0.25" max="3.0" value="1.0" step="0.05">
        <span class="val" id="eps-val">1.00</span><span>x</span>
      </div>
      <div class="ctrl-row">
        <label>Min cells</label>
        <input type="range" id="min-cells" min="3" max="50" value="{default_min_cells}" step="1">
        <span class="val" id="min-cells-val">{default_min_cells}</span>
      </div>
      <div class="ctrl-row">
        <label style="min-width:auto"><input type="checkbox" id="show-hulls"> Hulls</label>
        <label style="min-width:auto"><input type="checkbox" id="show-labels"> Labels</label>
      </div>
      <div id="cluster-status" style="font-size:10px;color:#777;"></div>
    </div>

    <!-- ROI Drawing -->
    <div class="sidebar-section">
      <h3>ROI Drawing</h3>
      <div class="btn-row">
        <button class="btn mode-btn active" id="mode-pan" data-mode="pan">Pan</button>
        <button class="btn mode-btn" id="mode-circle" data-mode="circle">Circle</button>
        <button class="btn mode-btn" id="mode-rect" data-mode="rect">Rect</button>
        <button class="btn mode-btn" id="mode-poly" data-mode="polygon">Poly</button>
        <button class="btn mode-btn" id="mode-path" data-mode="path">Path</button>
      </div>
      <div style="font-size:10px;color:#555;margin:4px 0;">
        Circle/Rect: click+drag &middot; Poly: click, dbl-click close<br>
        Path: click waypoints, dbl-click to finish (open)
      </div>
      <div id="roi-list"></div>
      <div class="btn-row" style="margin-top:6px;">
        <button class="btn" id="btn-download-roi">Download ROIs JSON</button>
      </div>
      <div class="ctrl-row">
        <input type="checkbox" id="roi-filter">
        <span style="font-size:11px;">Filter by ROIs</span>
      </div>
      <div id="roi-stats" style="font-size:10px;color:#777;"></div>
      <div class="ctrl-row" id="corridor-row" style="display:none;">
        <label>Corridor</label>
        <input type="range" id="corridor-slider" min="25" max="500" value="100" step="25">
        <span class="val" id="corridor-val">100</span><span>&micro;m</span>
      </div>
    </div>

    <!-- Help -->
    <div class="sidebar-section help-text">
      Scroll to zoom &middot; Drag to pan<br>
      {'Double-click panel for focus view' if not is_single else 'Single-slide mode'}<br>
      Click legend items to toggle groups
    </div>
  </div>

  <div id="main-area">
    <div id="grid" {'class="single-slide"' if is_single else ''}></div>
    <div id="focus-view">
      <button id="focus-back">Back to grid</button>
      <div id="focus-label"></div>
    </div>
  </div>
</div>

<script>
// ===================================================================
// Decode base64 binary data into TypedArrays
// ===================================================================
function b64toF32(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}
function b64toU8(b64) {{
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}}

// ===================================================================
// Slide data (binary-encoded)
// ===================================================================
const SLIDE_META = {safe_json(slides_meta)};
const GROUP_LABELS = {safe_json(group_labels)};
const GROUP_COLORS = {safe_json([color_map[lbl] for lbl in group_labels])};
const N_GROUPS = GROUP_LABELS.length;
const IS_SINGLE = {'true' if is_single else 'false'};
const GENERATED = {safe_json(timestamp)};
const GROUP_FIELD = {safe_json(group_field)};
const TITLE = {safe_json(title)};
const AUTO_EPS = {safe_json(slides_auto_eps)};
const MIN_HULL = {min_hull_cells};
const REGION_DATA = {safe_json(slides_region_data)};
const HAS_REGIONS = {'true' if has_regions else 'false'};
const HAS_MULTISCALE = {'true' if has_multiscale else 'false'};
const SCALE_KEYS = {safe_json(scale_keys or [])};
"""
    )

    # Emit base64 data arrays
    html_parts.append("const SLIDE_POS_B64 = [\n")
    for i, b64 in enumerate(slides_b64_positions):
        comma = "," if i < len(slides_b64_positions) - 1 else ""
        html_parts.append(f'  "{b64}"{comma}\n')
    html_parts.append("];\n")

    html_parts.append("const SLIDE_GRP_B64 = [\n")
    for i, b64 in enumerate(slides_b64_groups):
        comma = "," if i < len(slides_b64_groups) - 1 else ""
        html_parts.append(f'  "{b64}"{comma}\n')
    html_parts.append("];\n")

    # Fluorescence channel images: one entry per slide, null if no data for that slide
    html_parts.append("// Fluorescence channel data (grayscale PNG base64, one entry per slide)\n")
    html_parts.append("const FLUOR_META = [\n")
    for i, name in enumerate(slide_names_ordered):
        comma = "," if i < len(slide_names_ordered) - 1 else ""
        fd = (fluor_data or {}).get(name)
        if fd is None:
            html_parts.append(f"  null{comma}\n")
        else:
            entry = {
                "w": fd["width"],
                "h": fd["height"],
                "scale": fd["scale"],
                "mx": fd.get("mosaic_x", 0),
                "my": fd.get("mosaic_y", 0),
                "pixel_size": fd.get("pixel_size", 0.22),
                "names": fd.get("names", ["Ch0", "Ch1", "Ch2"]),
            }
            html_parts.append(f"  {safe_json(entry)}{comma}\n")
    html_parts.append("];\n")

    # Emit channel image base64 data as a flat array (3 images * n_slides)
    # Layout: FLUOR_CH_B64[slideIdx * 3 + channelIdx] = b64string or ''
    html_parts.append("const FLUOR_CH_B64 = [\n")
    for i, name in enumerate(slide_names_ordered):
        fd = (fluor_data or {}).get(name)
        for ci in range(3):
            is_last = (i == len(slide_names_ordered) - 1) and ci == 2
            comma = "" if is_last else ","
            if fd is None or ci >= len(fd["channels"]) or not fd["channels"][ci]:
                html_parts.append(f'  ""{comma}\n')
            else:
                html_parts.append(f'  "{fd["channels"][ci]}"{comma}\n')
    html_parts.append("];\n")

    # Contour data: one entry per slide
    html_parts.append("// Detection contours in um coordinates\n")
    html_parts.append("const CONTOUR_DATA = [\n")
    for i, name in enumerate(slide_names_ordered):
        comma = "," if i < len(slide_names_ordered) - 1 else ""
        cd = (contour_data or {}).get(name)
        if not cd:
            html_parts.append(f"  []{comma}\n")
        else:
            # Emit as JSON; pts arrays are plain lists (will become JS arrays)
            html_parts.append(f"  {safe_json(cd)}{comma}\n")
    html_parts.append("];\n")

    html_parts.append(
        f"""
const HAS_FLUOR = {'true' if has_fluor else 'false'};
const HAS_CONTOURS = {'true' if has_contours else 'false'};
const CH_NAMES = {safe_json(ch_names)};
"""
    )

    html_parts.append(
        """
// Decode binary data into per-slide arrays
const SLIDES = SLIDE_META.map((meta, i) => {
  const pos = b64toF32(SLIDE_POS_B64[i]);
  const grp = b64toU8(SLIDE_GRP_B64[i]);
  const rd = REGION_DATA[i] || {};
  return {
    name: meta.name,
    n: meta.n,
    xr: meta.xr,
    yr: meta.yr,
    pos: pos,  // interleaved [x0,y0,x1,y1,...] Float32Array
    grp: grp,  // group index per cell Uint8Array
    regions: rd.regions || [],
    regionScales: rd.regionScales || null,
  };
});

// Free the base64 strings to reduce memory
SLIDE_POS_B64.length = 0;
SLIDE_GRP_B64.length = 0;

// Build fluorescence image objects (decoded lazily on first render)
const fluorImages = SLIDES.map((_, si) => {
  const meta = FLUOR_META[si];
  if (!meta) return null;
  const imgs = [null, null, null];
  let loadedCount = 0;
  const result = { meta, imgs, ready: false, _canvas: null, _dirty: true };
  for (let ci = 0; ci < 3; ci++) {
    const b64 = FLUOR_CH_B64[si * 3 + ci];
    if (!b64) { loadedCount++; if (loadedCount === 3) result.ready = true; continue; }
    const img = new Image();
    img.onload = () => {
      imgs[ci] = img;
      result._dirty = true;
      loadedCount++;
      if (loadedCount === 3) {
        result.ready = true;
        // Re-render all panels once images are ready
        scheduleRenderAll();
      }
    };
    img.src = 'data:image/png;base64,' + b64;
  }
  return result;
});

// Free large base64 channel strings
FLUOR_CH_B64.length = 0;

// ===================================================================
// State
// ===================================================================
const hidden = new Set();
let dotSize = 3, dotAlpha = 0.7;
let showHulls = false, showLabels = false;
let drawMode = 'pan';  // pan | circle | rect | polygon

// KDE state
const KDE_RADII = [50, 100, 200, 300, 400, 500, 600, 700, 800, 1000];
let showKDE = false, kdeBWIdx = 3, kdeLevels = 3, kdeAlpha = 0.5, kdeFill = false, kdeLines = false;
const kdeCache = new Map();  // slideIdx -> {bwIdx, levels, hiddenKey, data}
let kdeDebounceTimer = null;

// Region state
let showRegions = HAS_REGIONS, showRegionLabels = HAS_REGIONS, showRegionBnd = HAS_REGIONS;
let regionAlpha = 0.25;

// Fluorescence + contour state
let showFluor = HAS_FLUOR, fluorAlpha = 0.8;
let chEnabled = [true, true, true];
let showContours = HAS_CONTOURS;
let showDots = true;

// Channel tint colors: R, G, B for additive compositing
const CH_TINTS = [[255,0,0], [0,255,0], [0,100,255]];

// Clustering state
const clusterData = new Array(SLIDES.length).fill(null);  // per-slide cluster results

// ROI storage
const rois = [];
let roiCounter = 0;
let roiFilterActive = false;

// Polygon in-progress
let polySlideIdx = -1;
let polyVerts = [];

// Drag/draw in-progress
let drawStart = null;
let drawCurrent = null;
let corridorWidth = 100;

// Panel state
const panels = [];
let activePanel = null;
let focusedIdx = -1;  // -1 = grid view, >= 0 = focused panel index

// RAF batching
let rafId = 0;
const rafDirty = new Set();

function scheduleRender(p) {
  rafDirty.add(p);
  if (!rafId) {
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    });
  }
}

function scheduleRenderAll() {
  panels.forEach(p => rafDirty.add(p));
  if (!rafId) {
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      for (const dp of rafDirty) renderPanel(dp);
      rafDirty.clear();
    });
  }
}

// ===================================================================
// ROI geometry tests
// ===================================================================
function pointInCircle(px, py, cx, cy, r) {
  const dx = px - cx, dy = py - cy;
  return dx * dx + dy * dy <= r * r;
}

function pointInRect(px, py, x1, y1, x2, y2) {
  const minX = Math.min(x1, x2), maxX = Math.max(x1, x2);
  const minY = Math.min(y1, y2), maxY = Math.max(y1, y2);
  return px >= minX && px <= maxX && py >= minY && py <= maxY;
}

function pointInPolygon(px, py, verts) {
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
    const xi = verts[i][0], yi = verts[i][1];
    const xj = verts[j][0], yj = verts[j][1];
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function pointNearPath(px, py, waypoints, halfWidth) {
  const hw2 = halfWidth * halfWidth;
  for (let i = 0; i < waypoints.length - 1; i++) {
    const [ax, ay] = waypoints[i];
    const [bx, by] = waypoints[i + 1];
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) continue;
    let t = ((px - ax) * dx + (py - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const projX = ax + t * dx, projY = ay + t * dy;
    const distSq = (px - projX) * (px - projX) + (py - projY) * (py - projY);
    if (distSq <= hw2) return true;
  }
  return false;
}

function pointInROI(px, py, roi) {
  if (roi.type === 'circle') {
    return pointInCircle(px, py, roi.data.cx, roi.data.cy, roi.data.r);
  } else if (roi.type === 'rect') {
    return pointInRect(px, py, roi.data.x1, roi.data.y1, roi.data.x2, roi.data.y2);
  } else if (roi.type === 'polygon') {
    return pointInPolygon(px, py, roi.data.verts);
  } else if (roi.type === 'path') {
    return pointNearPath(px, py, roi.data.waypoints, (roi.data.corridorWidth || corridorWidth) / 2);
  }
  return false;
}

function cellPassesROIFilter(px, py, slideIdx) {
  if (!roiFilterActive || rois.length === 0) return true;
  for (const roi of rois) {
    if (roi.slideIdx === slideIdx && pointInROI(px, py, roi)) return true;
  }
  return false;
}

// ===================================================================
// Coordinate transforms
// ===================================================================
function screenToData(p, sx, sy) {
  return [(sx - p.panX) / p.zoom, (sy - p.panY) / p.zoom];
}

function dataToScreen(p, dx, dy) {
  return [dx * p.zoom + p.panX, dy * p.zoom + p.panY];
}

// ===================================================================
// DBSCAN with grid spatial index
// ===================================================================
function dbscan(x, y, n, eps, minPts) {
  const labels = new Int32Array(n).fill(-1);
  if (n === 0 || eps <= 0) return labels;

  const grid = new Map();
  for (let i = 0; i < n; i++) {
    const gx = Math.floor(x[i] / eps);
    const gy = Math.floor(y[i] / eps);
    const key = gx + ',' + gy;
    let cell = grid.get(key);
    if (!cell) { cell = []; grid.set(key, cell); }
    cell.push(i);
  }

  const eps2 = eps * eps;
  function getNeighbors(idx) {
    const px = x[idx], py = y[idx];
    const gx = Math.floor(px / eps);
    const gy = Math.floor(py / eps);
    const result = [];
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const cell = grid.get((gx + dx) + ',' + (gy + dy));
        if (!cell) continue;
        for (let k = 0; k < cell.length; k++) {
          const j = cell[k];
          const ddx = x[j] - px, ddy = y[j] - py;
          if (ddx * ddx + ddy * ddy <= eps2) result.push(j);
        }
      }
    }
    return result;
  }

  let clusterId = 0;
  const visited = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = 1;
    const nbrs = getNeighbors(i);
    if (nbrs.length < minPts) continue;

    labels[i] = clusterId;
    const queue = [];
    for (let k = 0; k < nbrs.length; k++) {
      if (nbrs[k] !== i) queue.push(nbrs[k]);
    }
    let qi = 0;
    while (qi < queue.length) {
      const j = queue[qi++];
      if (!visited[j]) {
        visited[j] = 1;
        const jnbrs = getNeighbors(j);
        if (jnbrs.length >= minPts) {
          for (let k = 0; k < jnbrs.length; k++) {
            if (!visited[jnbrs[k]]) queue.push(jnbrs[k]);
          }
        }
      }
      if (labels[j] === -1) labels[j] = clusterId;
    }
    clusterId++;
  }
  return labels;
}

// ===================================================================
// Convex hull (Andrew's monotone chain)
// ===================================================================
function convexHull(points) {
  const n = points.length;
  if (n < 3) return points.slice();
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  const pts = [sorted[0]];
  for (let i = 1; i < n; i++) {
    if (sorted[i][0] !== sorted[i-1][0] || sorted[i][1] !== sorted[i-1][1])
      pts.push(sorted[i]);
  }
  if (pts.length < 3) return pts;

  function cross(O, A, B) {
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
  }
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

// ===================================================================
// Per-group position extraction (lazy, for DBSCAN)
// ===================================================================
function getGroupPositions(slideIdx) {
  const slide = SLIDES[slideIdx];
  if (slide._groupPos) return slide._groupPos;
  const gp = new Array(N_GROUPS).fill(null).map(() => ({xi:[], yi:[]}));
  for (let i = 0; i < slide.n; i++) {
    const gi = slide.grp[i];
    gp[gi].xi.push(slide.pos[i*2]);
    gp[gi].yi.push(slide.pos[i*2+1]);
  }
  slide._groupPos = gp.map(g => ({
    x: new Float32Array(g.xi),
    y: new Float32Array(g.yi),
    n: g.xi.length,
  }));
  return slide._groupPos;
}

// ===================================================================
// Hex color to RGB
// ===================================================================
function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [parseInt(h.substring(0,2),16), parseInt(h.substring(2,4),16), parseInt(h.substring(4,6),16)];
}

// ===================================================================
// Fluorescence rendering
// ===================================================================

function drawFluorescence(ctx, slideIdx, panZoom) {
  const fd = fluorImages[slideIdx];
  if (!fd || !fd.ready) return;

  const meta = fd.meta;
  const iw = meta.w, ih = meta.h;
  // Scale factor maps thumbnail pixels -> full-resolution pixels.
  // The viewer coordinate space is in um, so we also multiply by pixel_size.
  // thumbnail_pixel = full_res_pixel * scale
  // um = full_res_pixel * pixel_size
  // => full_res_pixel = thumbnail_pixel / scale
  // => um = (thumbnail_pixel / scale) * pixel_size
  // => thumbnail_pixel = um / pixel_size * scale
  // Draw position in um space: mosaic origin in full-res pixels -> um
  const mx_um = meta.mx * meta.pixel_size;
  const my_um = meta.my * meta.pixel_size;
  const scale_inv = 1.0 / meta.scale;  // thumbnail pixel -> full-res pixel
  const draw_w = iw * scale_inv * meta.pixel_size;  // um
  const draw_h = ih * scale_inv * meta.pixel_size;  // um

  // Rebuild composite offscreen canvas only when needed
  if (fd._dirty || !fd._canvas) {
    if (!fd._canvas) {
      fd._canvas = document.createElement('canvas');
      fd._canvas.width = iw;
      fd._canvas.height = ih;
    }
    const fctx = fd._canvas.getContext('2d', { willReadFrequently: true });
    // Additive channel compositing via pixel-level blend
    const result = new Uint8ClampedArray(iw * ih * 4);
    for (let ci = 0; ci < 3; ci++) {
      if (!chEnabled[ci] || !fd.imgs[ci]) continue;
      // Draw grayscale channel to temp canvas, read pixels
      const tmp = document.createElement('canvas');
      tmp.width = iw; tmp.height = ih;
      const tctx = tmp.getContext('2d');
      tctx.drawImage(fd.imgs[ci], 0, 0);
      const px = tctx.getImageData(0, 0, iw, ih).data;
      const [tr, tg, tb] = CH_TINTS[ci];
      for (let i = 0; i < iw * ih; i++) {
        const v = px[i * 4] / 255;
        result[i * 4]     = Math.min(255, result[i * 4]     + tr * v);
        result[i * 4 + 1] = Math.min(255, result[i * 4 + 1] + tg * v);
        result[i * 4 + 2] = Math.min(255, result[i * 4 + 2] + tb * v);
        result[i * 4 + 3] = 255;
      }
    }
    fctx.putImageData(new ImageData(result, iw, ih), 0, 0);
    fd._dirty = false;
  }

  ctx.globalAlpha = fluorAlpha;
  ctx.drawImage(fd._canvas, mx_um, my_um, draw_w, draw_h);
  ctx.globalAlpha = 1;
}

// ===================================================================
// Detection contour rendering
// ===================================================================

function drawContours(ctx, p, panZoom) {
  const contours = CONTOUR_DATA[p.idx];
  if (!contours || contours.length === 0) return;

  // Compute visible data bounds (in um) for viewport culling
  // Panel transform: screen = data * zoom + pan => data = (screen - pan) / zoom
  const vx1 = (0 - p.panX) / p.zoom;
  const vy1 = (0 - p.panY) / p.zoom;
  const vx2 = (p.cw - p.panX) / p.zoom;
  const vy2 = (p.ch - p.panY) / p.zoom;

  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 1.5 / panZoom;
  ctx.setLineDash([6 / panZoom, 4 / panZoom]);
  ctx.globalAlpha = 0.85;

  for (let ci = 0; ci < contours.length; ci++) {
    const c = contours[ci];
    // Viewport culling via pre-computed bounding box
    if (c.bx2 < vx1 || c.bx1 > vx2 || c.by2 < vy1 || c.by1 > vy2) continue;

    const pts = c.pts;
    if (!pts || pts.length < 6) continue;  // at least 3 points (6 floats)

    const path = new Path2D();
    path.moveTo(pts[0], pts[1]);
    for (let j = 2; j < pts.length; j += 2) {
      path.lineTo(pts[j], pts[j + 1]);
    }
    path.closePath();
    ctx.stroke(path);
  }

  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}

// ===================================================================
// KDE: Histogram2D + Gaussian blur + Marching Squares
// ===================================================================

function computeHistogram2D(x, y, w, nx, ny, xr, yr) {
  const grid = new Float32Array(ny * nx);
  const sx = nx / (xr[1] - xr[0]);
  const sy = ny / (yr[1] - yr[0]);
  const n = x.length;
  for (let i = 0; i < n; i++) {
    const gx = Math.min(Math.floor((x[i] - xr[0]) * sx), nx - 1);
    const gy = Math.min(Math.floor((y[i] - yr[0]) * sy), ny - 1);
    if (gx >= 0 && gy >= 0) {
      grid[gy * nx + gx] += (w ? w[i] : 1);
    }
  }
  return grid;
}

function gaussianBlur1D(src, nx, ny, sigma, horizontal) {
  const radius = Math.ceil(sigma * 3);
  const kernel = new Float32Array(2 * radius + 1);
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    kernel[i + radius] = Math.exp(-0.5 * (i / sigma) * (i / sigma));
    ksum += kernel[i + radius];
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  const dst = new Float32Array(ny * nx);

  if (horizontal) {
    for (let row = 0; row < ny; row++) {
      for (let col = 0; col < nx; col++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const c = Math.min(Math.max(col + k, 0), nx - 1);
          sum += src[row * nx + c] * kernel[k + radius];
        }
        dst[row * nx + col] = sum;
      }
    }
  } else {
    for (let col = 0; col < nx; col++) {
      for (let row = 0; row < ny; row++) {
        let sum = 0;
        for (let k = -radius; k <= radius; k++) {
          const r = Math.min(Math.max(row + k, 0), ny - 1);
          sum += src[r * nx + col] * kernel[k + radius];
        }
        dst[row * nx + col] = sum;
      }
    }
  }
  return dst;
}

function gaussianBlur(grid, nx, ny, sigma) {
  if (sigma < 0.5) return grid;
  const tmp = gaussianBlur1D(grid, nx, ny, sigma, true);
  return gaussianBlur1D(tmp, nx, ny, sigma, false);
}

function marchingSquares(grid, nx, ny, threshold, xr, yr) {
  const stepX = (xr[1] - xr[0]) / nx;
  const stepY = (yr[1] - yr[0]) / ny;

  function lerp(v1, v2) {
    const d = v2 - v1;
    return Math.abs(d) < 1e-10 ? 0.5 : (threshold - v1) / d;
  }

  const segments = [];
  for (let row = 0; row < ny - 1; row++) {
    for (let col = 0; col < nx - 1; col++) {
      const tl = grid[row * nx + col] >= threshold ? 1 : 0;
      const tr = grid[row * nx + col + 1] >= threshold ? 1 : 0;
      const br = grid[(row + 1) * nx + col + 1] >= threshold ? 1 : 0;
      const bl = grid[(row + 1) * nx + col] >= threshold ? 1 : 0;
      let code = (tl << 3) | (tr << 2) | (br << 1) | bl;

      if (code === 0 || code === 15) continue;

      const x0 = xr[0] + col * stepX;
      const y0 = yr[0] + row * stepY;
      const vTL = grid[row * nx + col];
      const vTR = grid[row * nx + col + 1];
      const vBR = grid[(row + 1) * nx + col + 1];
      const vBL = grid[(row + 1) * nx + col];

      const top = [x0 + lerp(vTL, vTR) * stepX, y0];
      const right = [x0 + stepX, y0 + lerp(vTR, vBR) * stepY];
      const bottom = [x0 + lerp(vBL, vBR) * stepX, y0 + stepY];
      const left = [x0, y0 + lerp(vTL, vBL) * stepY];

      if (code === 5 || code === 10) {
        const center = (vTL + vTR + vBR + vBL) / 4;
        if (center >= threshold) {
          if (code === 5) code = 17;
          else code = 18;
        }
      }

      let segs;
      switch (code) {
        case 1:  segs = [[left, bottom]]; break;
        case 2:  segs = [[bottom, right]]; break;
        case 3:  segs = [[left, right]]; break;
        case 4:  segs = [[right, top]]; break;
        case 5:  segs = [[left, top], [bottom, right]]; break;
        case 17: segs = [[left, bottom], [top, right]]; break;
        case 6:  segs = [[bottom, top]]; break;
        case 7:  segs = [[left, top]]; break;
        case 8:  segs = [[top, left]]; break;
        case 9:  segs = [[top, bottom]]; break;
        case 10: segs = [[top, right], [left, bottom]]; break;
        case 18: segs = [[top, left], [bottom, right]]; break;
        case 11: segs = [[top, right]]; break;
        case 12: segs = [[right, left]]; break;
        case 13: segs = [[right, bottom]]; break;
        case 14: segs = [[bottom, left]]; break;
        default: segs = null;
      }

      if (segs) {
        for (const seg of segs) segments.push(seg);
      }
    }
  }

  if (segments.length === 0) return [];

  const eps = stepX * 0.01;
  const eps2 = eps * eps;

  function ptKey(p) {
    return Math.round(p[0] / eps) + ',' + Math.round(p[1] / eps);
  }

  const endHash = new Map();
  for (let i = 0; i < segments.length; i++) {
    for (let e = 0; e < 2; e++) {
      const k = ptKey(segments[i][e]);
      if (!endHash.has(k)) endHash.set(k, []);
      endHash.get(k).push({ si: i, ei: e });
    }
  }

  function dist2(a, b) {
    const dx = a[0] - b[0], dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }

  const polys = [];
  const used = new Uint8Array(segments.length);

  for (let start = 0; start < segments.length; start++) {
    if (used[start]) continue;
    used[start] = 1;
    const poly = [segments[start][0], segments[start][1]];

    let found = true;
    while (found) {
      found = false;
      const tail = poly[poly.length - 1];
      const rx = Math.round(tail[0] / eps);
      const ry = Math.round(tail[1] / eps);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const bucket = endHash.get((rx + dx) + ',' + (ry + dy));
          if (!bucket) continue;
          for (const entry of bucket) {
            if (used[entry.si]) continue;
            const seg = segments[entry.si];
            if (dist2(tail, seg[entry.ei]) < eps2) {
              poly.push(seg[1 - entry.ei]);
              used[entry.si] = 1;
              found = true;
              break;
            }
            if (dist2(tail, seg[1 - entry.ei]) < eps2) {
              poly.push(seg[entry.ei]);
              used[entry.si] = 1;
              found = true;
              break;
            }
          }
          if (found) break;
        }
        if (found) break;
      }
    }
    if (poly.length >= 3) polys.push(poly);
  }

  return polys;
}

function computeKDE(slideIdx, bandwidthUm, nLevels) {
  const slide = SLIDES[slideIdx];
  const xr = slide.xr, yr = slide.yr;
  const dataW = xr[1] - xr[0], dataH = yr[1] - yr[0];
  if (dataW <= 0 || dataH <= 0) return null;

  const nxGrid = 200, nyGrid = 200;
  const pixelSize = Math.max(dataW, dataH) / Math.max(nxGrid, nyGrid);
  const sigma = bandwidthUm / pixelSize;

  let nx, ny;
  if (dataW > dataH) {
    nx = nxGrid;
    ny = Math.max(1, Math.round(nxGrid * dataH / dataW));
  } else {
    ny = nyGrid;
    nx = Math.max(1, Math.round(nyGrid * dataW / dataH));
  }

  const gpos = getGroupPositions(slideIdx);
  const result = [];

  for (let gi = 0; gi < N_GROUPS; gi++) {
    if (hidden.has(GROUP_LABELS[gi])) continue;
    const gp = gpos[gi];
    if (gp.n < 5) continue;

    const hist = computeHistogram2D(gp.x, gp.y, null, nx, ny, xr, yr);
    const blurred = gaussianBlur(hist, nx, ny, sigma);

    let maxD = 0;
    for (let i = 0; i < blurred.length; i++) {
      if (blurred[i] > maxD) maxD = blurred[i];
    }
    if (maxD <= 0) continue;

    const contours = [];
    for (let li = 1; li <= nLevels; li++) {
      const frac = li / (nLevels + 1);
      const threshold = maxD * frac;
      const polys = marchingSquares(blurred, nx, ny, threshold, xr, yr);
      contours.push({ level: frac, polys });
    }

    result.push({ gi, color: GROUP_COLORS[gi], contours });
  }

  return result;
}

function getKDE(slideIdx) {
  const hiddenKey = Array.from(hidden).sort().join('|');
  const cached = kdeCache.get(slideIdx);
  if (cached && cached.bwIdx === kdeBWIdx && cached.levels === kdeLevels && cached.hiddenKey === hiddenKey) {
    return cached.data;
  }
  const bw = KDE_RADII[kdeBWIdx];
  const data = computeKDE(slideIdx, bw, kdeLevels);
  kdeCache.set(slideIdx, { bwIdx: kdeBWIdx, levels: kdeLevels, hiddenKey, data });
  return data;
}

function drawKDEContours(ctx, kdeData, panZoom, opacity, fill, lines) {
  if (!kdeData) return;

  for (const entry of kdeData) {
    const color = entry.color;
    const [r, g, b] = hexToRgb(color);

    for (let li = entry.contours.length - 1; li >= 0; li--) {
      const { level, polys } = entry.contours[li];

      for (const poly of polys) {
        if (poly.length < 3) continue;

        const path = new Path2D();
        path.moveTo(poly[0][0], poly[0][1]);
        for (let i = 1; i < poly.length; i++) {
          path.lineTo(poly[i][0], poly[i][1]);
        }
        path.closePath();

        if (fill) {
          ctx.globalAlpha = opacity * (1 - level) * 0.6;
          ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',1)';
          ctx.fill(path);
        }

        if (lines) {
          ctx.globalAlpha = opacity;
          ctx.strokeStyle = color;
          ctx.lineWidth = (1.5 - level * 0.5) / panZoom;
          ctx.stroke(path);
        }
      }
    }
  }
}

// ===================================================================
// Draw precomputed regions
// ===================================================================
function drawRegions(ctx, regions, panZoom, opacity, showLbl, showBnd) {
  if (!regions || regions.length === 0) return;

  for (const reg of regions) {
    if (reg.bnd.length < 3) continue;
    if (reg.type && hidden.has(reg.type)) continue;

    const path = new Path2D();
    path.moveTo(reg.bnd[0][0], reg.bnd[0][1]);
    for (let i = 1; i < reg.bnd.length; i++) {
      path.lineTo(reg.bnd[i][0], reg.bnd[i][1]);
    }
    path.closePath();

    ctx.globalAlpha = opacity;
    ctx.fillStyle = reg.color;
    ctx.fill(path);

    if (showBnd) {
      ctx.globalAlpha = Math.min(opacity * 3, 0.9);
      ctx.strokeStyle = reg.color;
      ctx.lineWidth = 2.5 / panZoom;
      ctx.stroke(path);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.0 / panZoom;
      ctx.stroke(path);
    }

    if (showLbl) {
      let cx = 0, cy = 0;
      for (const pt of reg.bnd) { cx += pt[0]; cy += pt[1]; }
      cx /= reg.bnd.length;
      cy /= reg.bnd.length;

      const fontSize = 11 / panZoom;
      ctx.font = 'bold ' + fontSize + 'px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.globalAlpha = 0.9;

      const line1 = reg.label;
      const line2 = reg.n + ' cells (' + (reg.dfrac * 100).toFixed(0) + '%)';
      const lh = fontSize * 1.3;

      ctx.fillStyle = '#000';
      ctx.fillText(line1, cx + 0.8/panZoom, cy - lh/2 + 0.8/panZoom);
      ctx.fillText(line2, cx + 0.8/panZoom, cy + lh/2 + 0.8/panZoom);
      ctx.fillStyle = '#fff';
      ctx.fillText(line1, cx, cy - lh/2);
      ctx.fillText(line2, cx, cy + lh/2);
    }
  }
}

// ===================================================================
// Re-cluster all groups in all slides
// ===================================================================
function reclusterAll() {
  const mult = parseFloat(document.getElementById('eps-slider').value);
  const minCells = parseInt(document.getElementById('min-cells').value);
  let totalClusters = 0, totalHulls = 0;
  let epsMin = Infinity, epsMax = 0;
  const t0 = performance.now();

  for (let si = 0; si < SLIDES.length; si++) {
    const gpos = getGroupPositions(si);
    const slideClusters = [];

    for (let gi = 0; gi < N_GROUPS; gi++) {
      if (hidden.has(GROUP_LABELS[gi])) { slideClusters.push([]); continue; }
      const gp = gpos[gi];
      if (gp.n === 0) { slideClusters.push([]); continue; }

      const eps = AUTO_EPS[si][gi] * mult;
      if (eps < epsMin) epsMin = eps;
      if (eps > epsMax) epsMax = eps;
      const labels = dbscan(gp.x, gp.y, gp.n, eps, minCells);

      const clusterMap = new Map();
      for (let i = 0; i < gp.n; i++) {
        const cl = labels[i];
        if (cl === -1) continue;
        let arr = clusterMap.get(cl);
        if (!arr) { arr = []; clusterMap.set(cl, arr); }
        arr.push(i);
      }

      const groupClusters = [];
      let num = 0;
      for (const [clId, indices] of clusterMap) {
        num++;
        totalClusters++;
        const pts = [];
        let sx = 0, sy = 0;
        for (const idx of indices) {
          const px = gp.x[idx], py = gp.y[idx];
          pts.push([px, py]);
          sx += px; sy += py;
        }
        const cx = sx / indices.length;
        const cy = sy / indices.length;

        let hull = [];
        if (indices.length >= MIN_HULL) {
          hull = convexHull(pts);
          if (hull.length >= 3) totalHulls++;
          else hull = [];
        }

        groupClusters.push({
          label: GROUP_LABELS[gi] + ' #' + num,
          n: indices.length,
          hull: hull,
          cx: cx,
          cy: cy,
        });
      }
      slideClusters.push(groupClusters);
    }
    clusterData[si] = slideClusters;
  }

  const dt = (performance.now() - t0).toFixed(0);
  const epsRange = epsMin === Infinity ? '' :
    ' | eps ' + Math.round(epsMin) + '-' + Math.round(epsMax) + ' um';
  document.getElementById('cluster-status').textContent =
    totalClusters + ' clusters (' + totalHulls + ' hulls) ' + dt + 'ms' + epsRange;
}

// ===================================================================
// IntersectionObserver for lazy rendering
// ===================================================================
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const idx = parseInt(entry.target.dataset.idx);
    const p = panels[idx];
    if (entry.isIntersecting) {
      p.visible = true;
      scheduleRender(p);
    } else {
      p.visible = false;
    }
  }
}, { root: document.getElementById('grid'), threshold: 0.01 });

// ===================================================================
// Panel initialization
// ===================================================================
function initPanels() {
  const grid = document.getElementById('grid');
  const select = document.getElementById('slide-select');

  SLIDES.forEach((slide, idx) => {
    const div = document.createElement('div');
    div.className = 'panel';
    div.dataset.idx = idx;

    const labelEl = document.createElement('div');
    labelEl.className = 'panel-label';
    labelEl.textContent = slide.name;

    const countEl = document.createElement('div');
    countEl.className = 'panel-count';

    const measureEl = document.createElement('div');
    measureEl.className = 'panel-measure';

    const canvas = document.createElement('canvas');
    const drawCanvas = document.createElement('canvas');
    drawCanvas.className = 'draw-overlay';

    div.appendChild(labelEl);
    div.appendChild(countEl);
    div.appendChild(measureEl);
    div.appendChild(canvas);
    div.appendChild(drawCanvas);
    grid.appendChild(div);

    const ctx = canvas.getContext('2d');
    const dctx = drawCanvas.getContext('2d');

    const state = {
      div, canvas, ctx, drawCanvas, dctx, countEl, measureEl, slide, idx,
      zoom: 1, panX: 0, panY: 0,
      dragStartX: 0, dragStartY: 0, panStartX: 0, panStartY: 0,
      visible: false, cw: 0, ch: 0,
    };
    panels.push(state);
    observer.observe(div);

    // Double-click to focus (grid -> focus view)
    if (!IS_SINGLE) {
      div.addEventListener('dblclick', e => {
        if (drawMode !== 'pan') return;
        enterFocusView(idx);
        e.preventDefault();
      });
    }

    // Pan on data canvas
    canvas.addEventListener('mousedown', e => {
      if (drawMode !== 'pan') return;
      activePanel = state;
      div.classList.add('dragging');
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      state.panStartX = state.panX;
      state.panStartY = state.panY;
      e.preventDefault();
    });

    // Drawing events on overlay canvas
    drawCanvas.addEventListener('mousedown', e => {
      if (drawMode === 'pan') return;
      const rect = div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(state, sx, sy);

      if (drawMode === 'polygon' || drawMode === 'path') {
        if (polySlideIdx !== idx) {
          polySlideIdx = idx;
          polyVerts = [];
        }
        polyVerts.push([dx, dy]);
        renderDrawOverlay(state);
      } else {
        drawStart = { x: dx, y: dy, panel: state };
        drawCurrent = { x: dx, y: dy };
      }
      e.preventDefault();
    });

    // mousemove/mouseup for draw are handled at window level (below)
    // to avoid losing events when mouse leaves the canvas during drag

    drawCanvas.addEventListener('dblclick', e => {
      if (drawMode === 'polygon' && polySlideIdx === idx && polyVerts.length >= 3) {
        addROI(idx, 'polygon', { verts: polyVerts.slice() });
      } else if (drawMode === 'path' && polySlideIdx === idx && polyVerts.length >= 2) {
        addROI(idx, 'path', { waypoints: polyVerts.slice(), corridorWidth: corridorWidth });
      } else {
        return;
      }
      polyVerts = [];
      polySlideIdx = -1;
      renderDrawOverlay(state);
      e.preventDefault();
      e.stopPropagation();
    });

    // Wheel zoom on both canvases
    function handleWheel(e) {
      e.preventDefault();
      const rect = div.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      state.panX = mx - factor * (mx - state.panX);
      state.panY = my - factor * (my - state.panY);
      state.zoom *= factor;
      state.zoom = Math.max(0.001, Math.min(500, state.zoom));
      scheduleRender(state);
      renderDrawOverlay(state);
    }
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    drawCanvas.addEventListener('wheel', handleWheel, { passive: false });

    // Slide dropdown
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = slide.name + ' (' + slide.n.toLocaleString() + ')';
    select.appendChild(opt);
  });

  // Global mouse handlers for pan drag
  window.addEventListener('mousemove', e => {
    // Pan drag
    if (activePanel) {
      activePanel.panX = activePanel.panStartX + (e.clientX - activePanel.dragStartX);
      activePanel.panY = activePanel.panStartY + (e.clientY - activePanel.dragStartY);
      scheduleRender(activePanel);
      return;
    }
    // Draw drag (circle/rect) — handled at window level so mouse leaving canvas doesn't break it
    if (drawStart && drawMode !== 'pan') {
      const p = drawStart.panel;
      const rect = p.div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(p, sx, sy);
      drawCurrent = { x: dx, y: dy };
      if (drawMode === 'circle') {
        const ddx = dx - drawStart.x, ddy = dy - drawStart.y;
        const r = Math.sqrt(ddx * ddx + ddy * ddy);
        p.measureEl.style.display = 'block';
        p.measureEl.textContent = 'r = ' + r.toFixed(0) + ' \\u00b5m';
      } else if (drawMode === 'rect') {
        const w = Math.abs(dx - drawStart.x);
        const h = Math.abs(dy - drawStart.y);
        p.measureEl.style.display = 'block';
        p.measureEl.textContent = w.toFixed(0) + ' \\u00d7 ' + h.toFixed(0) + ' \\u00b5m';
      }
      renderDrawOverlay(p);
    }
  });
  window.addEventListener('mouseup', e => {
    // Pan drag end
    if (activePanel) {
      activePanel.div.classList.remove('dragging');
      activePanel = null;
      return;
    }
    // Draw drag end (circle/rect)
    if (drawStart && drawMode !== 'pan' && drawMode !== 'polygon' && drawMode !== 'path') {
      const p = drawStart.panel;
      const rect = p.div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(p, sx, sy);
      if (drawMode === 'circle') {
        const cdx = dx - drawStart.x, cdy = dy - drawStart.y;
        const r = Math.sqrt(cdx * cdx + cdy * cdy);
        if (r > 1) addROI(p.idx, 'circle', { cx: drawStart.x, cy: drawStart.y, r });
      } else if (drawMode === 'rect') {
        const w = Math.abs(dx - drawStart.x), h = Math.abs(dy - drawStart.y);
        if (w > 1 && h > 1) addROI(p.idx, 'rect', { x1: drawStart.x, y1: drawStart.y, x2: dx, y2: dy });
      }
      drawStart = null;
      drawCurrent = null;
      p.measureEl.style.display = 'none';
      renderDrawOverlay(p);
    }
  });
}

// ===================================================================
// Focus view
// ===================================================================
function enterFocusView(idx) {
  if (IS_SINGLE) return;
  focusedIdx = idx;
  const focusView = document.getElementById('focus-view');
  const focusLabel = document.getElementById('focus-label');
  const grid = document.getElementById('grid');
  const p = panels[idx];

  // Move panel div into focus view
  focusView.appendChild(p.div);
  p.div.style.position = 'absolute';
  p.div.style.top = '0';
  p.div.style.left = '0';
  p.div.style.width = '100%';
  p.div.style.height = '100%';
  p.div.style.borderRadius = '0';

  focusLabel.textContent = p.slide.name + ' \\u2014 ' + p.slide.n.toLocaleString() + ' cells';
  focusView.classList.add('active');
  grid.style.display = 'none';

  // Resize and re-render
  setTimeout(() => {
    resizePanel(p);
    fitPanel(p);
    p.visible = true;
    scheduleRender(p);
    renderDrawOverlay(p);
  }, 50);
}

function exitFocusView() {
  if (focusedIdx < 0) return;
  const p = panels[focusedIdx];
  const focusView = document.getElementById('focus-view');
  const grid = document.getElementById('grid');

  // Move panel back to grid
  p.div.style.position = '';
  p.div.style.top = '';
  p.div.style.left = '';
  p.div.style.width = '';
  p.div.style.height = '';
  p.div.style.borderRadius = '';

  // Re-insert at correct position in grid
  const nextIdx = focusedIdx + 1;
  if (nextIdx < panels.length) {
    grid.insertBefore(p.div, panels[nextIdx].div);
  } else {
    grid.appendChild(p.div);
  }

  focusView.classList.remove('active');
  grid.style.display = '';
  focusedIdx = -1;

  // Resize all and re-render
  setTimeout(() => {
    resizePanels();
    panels.forEach(fitPanel);
    scheduleRenderAll();
    panels.forEach(pp => renderDrawOverlay(pp));
  }, 50);
}

// Escape key exits focus view
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && focusedIdx >= 0) {
    exitFocusView();
  }
});

document.getElementById('focus-back').addEventListener('click', exitFocusView);

// ===================================================================
// Resize / fit
// ===================================================================
function resizePanel(p) {
  const dpr = window.devicePixelRatio || 1;
  const rect = p.div.getBoundingClientRect();
  const w = Math.floor(rect.width);
  const h = Math.floor(rect.height);
  if (w <= 0 || h <= 0) return;
  p.cw = w;
  p.ch = h;
  p.canvas.width = w * dpr;
  p.canvas.height = h * dpr;
  p.canvas.style.width = w + 'px';
  p.canvas.style.height = h + 'px';
  p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  p.drawCanvas.width = w * dpr;
  p.drawCanvas.height = h * dpr;
  p.drawCanvas.style.width = w + 'px';
  p.drawCanvas.style.height = h + 'px';
  p.dctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function resizePanels() {
  panels.forEach(resizePanel);
}

function fitPanel(p) {
  const cw = p.cw || 400;
  const ch = p.ch || 400;
  const s = p.slide;
  const dataW = s.xr[1] - s.xr[0];
  const dataH = s.yr[1] - s.yr[0];
  if (dataW <= 0 || dataH <= 0) {
    p.zoom = 1;
    p.panX = cw / 2;
    p.panY = ch / 2;
    return;
  }
  const pad = 0.05;
  p.zoom = Math.min(cw / (dataW * (1 + 2 * pad)), ch / (dataH * (1 + 2 * pad)));
  p.panX = (cw - dataW * p.zoom) / 2 - s.xr[0] * p.zoom;
  p.panY = (ch - dataH * p.zoom) / 2 - s.yr[0] * p.zoom;
}

// ===================================================================
// Render data panel
// ===================================================================
function renderPanel(p) {
  if (!p.visible && focusedIdx !== p.idx) return;
  const cw = p.cw || 400;
  const ch = p.ch || 400;
  const ctx = p.ctx;
  ctx.save();
  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#111122';
  ctx.fillRect(0, 0, cw, ch);
  ctx.translate(p.panX, p.panY);
  ctx.scale(p.zoom, p.zoom);

  // Layer 0: Fluorescence background
  if (showFluor && HAS_FLUOR) {
    drawFluorescence(ctx, p.idx, p.zoom);
  }

  // Layer 1: Regions (lowest)
  if (showRegions && p.slide.regions && p.slide.regions.length > 0) {
    drawRegions(ctx, p.slide.regions, p.zoom, regionAlpha, showRegionLabels, showRegionBnd);
  }

  // Layer 2: KDE contours
  if (showKDE) {
    const kdeData = getKDE(p.idx);
    drawKDEContours(ctx, kdeData, p.zoom, kdeAlpha, kdeFill, kdeLines);
  }

  // Layer 2.5: Detection contours
  if (showContours && HAS_CONTOURS) {
    drawContours(ctx, p, p.zoom);
  }

  // Layer 3: Cell dots
  const r = dotSize / p.zoom;
  const halfR = r / 2;
  const slide = p.slide;
  const pos = slide.pos;
  const grp = slide.grp;
  const n = slide.n;
  let total = 0;

  const useROIFilter = roiFilterActive && rois.length > 0;

  if (showDots) {
    for (let gi = 0; gi < N_GROUPS; gi++) {
      if (hidden.has(GROUP_LABELS[gi])) continue;
      ctx.globalAlpha = dotAlpha;
      ctx.fillStyle = GROUP_COLORS[gi];

      for (let i = 0; i < n; i++) {
        if (grp[i] !== gi) continue;
        const x = pos[i * 2];
        const y = pos[i * 2 + 1];
        if (useROIFilter && !cellPassesROIFilter(x, y, p.idx)) continue;
        ctx.fillRect(x - halfR, y - halfR, r, r);
        total++;
      }
    }
  } else {
    // Count visible cells even when dots are hidden
    const useFilter = roiFilterActive && rois.length > 0;
    for (let i = 0; i < n; i++) {
      const gi = grp[i];
      if (hidden.has(GROUP_LABELS[gi])) continue;
      if (useFilter && !cellPassesROIFilter(pos[i*2], pos[i*2+1], p.idx)) continue;
      total++;
    }
  }

  // Layer 4: Cluster hulls (top)
  const sc = clusterData[p.idx];
  if (sc) {
    for (let gi = 0; gi < N_GROUPS; gi++) {
      if (hidden.has(GROUP_LABELS[gi])) continue;
      const groupClusters = sc[gi];
      if (!groupClusters) continue;

      for (const cl of groupClusters) {
        if (showHulls && cl.hull && cl.hull.length >= 3) {
          ctx.globalAlpha = 1;
          const path = new Path2D();
          path.moveTo(cl.hull[0][0], cl.hull[0][1]);
          for (let i = 1; i < cl.hull.length; i++) {
            path.lineTo(cl.hull[i][0], cl.hull[i][1]);
          }
          path.closePath();

          ctx.setLineDash([6/p.zoom, 4/p.zoom]);
          ctx.strokeStyle = '#000';
          ctx.lineWidth = 2.5 / p.zoom;
          ctx.stroke(path);
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1.2 / p.zoom;
          ctx.stroke(path);
          ctx.setLineDash([]);
        }

        if (showLabels && cl.hull && cl.hull.length >= 3) {
          ctx.globalAlpha = 1;
          const fontSize = 11 / p.zoom;
          ctx.font = fontSize + 'px system-ui';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const line1 = cl.n + ' cells';
          ctx.fillStyle = '#000';
          ctx.fillText(line1, cl.cx + 0.5/p.zoom, cl.cy + 0.5/p.zoom);
          ctx.fillStyle = '#fff';
          ctx.fillText(line1, cl.cx, cl.cy);
        }
      }
    }
  }

  ctx.restore();
  p.countEl.textContent = total.toLocaleString() + ' cells';
}

// ===================================================================
// Render draw overlay
// ===================================================================
function renderDrawOverlay(p) {
  const cw = p.cw || 400;
  const ch = p.ch || 400;
  const dctx = p.dctx;
  dctx.clearRect(0, 0, cw, ch);
  dctx.save();
  dctx.translate(p.panX, p.panY);
  dctx.scale(p.zoom, p.zoom);

  const lw = 1.5 / p.zoom;

  // Draw existing ROIs for this slide
  for (const roi of rois) {
    if (roi.slideIdx !== p.idx) continue;
    dctx.strokeStyle = '#ffcc00';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.8;
    dctx.setLineDash([4 / p.zoom, 3 / p.zoom]);

    if (roi.type === 'circle') {
      dctx.beginPath();
      dctx.arc(roi.data.cx, roi.data.cy, roi.data.r, 0, Math.PI * 2);
      dctx.stroke();
    } else if (roi.type === 'rect') {
      const x = Math.min(roi.data.x1, roi.data.x2);
      const y = Math.min(roi.data.y1, roi.data.y2);
      const w = Math.abs(roi.data.x2 - roi.data.x1);
      const h = Math.abs(roi.data.y2 - roi.data.y1);
      dctx.strokeRect(x, y, w, h);
    } else if (roi.type === 'polygon') {
      dctx.beginPath();
      dctx.moveTo(roi.data.verts[0][0], roi.data.verts[0][1]);
      for (let i = 1; i < roi.data.verts.length; i++) {
        dctx.lineTo(roi.data.verts[i][0], roi.data.verts[i][1]);
      }
      dctx.closePath();
      dctx.stroke();
    } else if (roi.type === 'path') {
      // Corridor fill
      dctx.save();
      dctx.globalAlpha = 0.12;
      dctx.strokeStyle = '#ffcc00';
      dctx.lineWidth = roi.data.corridorWidth || corridorWidth;
      dctx.lineCap = 'round';
      dctx.lineJoin = 'round';
      dctx.setLineDash([]);
      dctx.beginPath();
      dctx.moveTo(roi.data.waypoints[0][0], roi.data.waypoints[0][1]);
      for (let i = 1; i < roi.data.waypoints.length; i++) {
        dctx.lineTo(roi.data.waypoints[i][0], roi.data.waypoints[i][1]);
      }
      dctx.stroke();
      dctx.restore();
      // Centerline
      dctx.beginPath();
      dctx.moveTo(roi.data.waypoints[0][0], roi.data.waypoints[0][1]);
      for (let i = 1; i < roi.data.waypoints.length; i++) {
        dctx.lineTo(roi.data.waypoints[i][0], roi.data.waypoints[i][1]);
      }
      dctx.stroke();
      // Endpoint markers: green=start(CV), red=end(PV)
      const er = 4 / p.zoom;
      dctx.globalAlpha = 1;
      dctx.setLineDash([]);
      dctx.fillStyle = '#00ff00';
      dctx.beginPath();
      dctx.arc(roi.data.waypoints[0][0], roi.data.waypoints[0][1], er, 0, Math.PI * 2);
      dctx.fill();
      dctx.fillStyle = '#ff4444';
      const last = roi.data.waypoints[roi.data.waypoints.length - 1];
      dctx.beginPath();
      dctx.arc(last[0], last[1], er, 0, Math.PI * 2);
      dctx.fill();
    }
    dctx.setLineDash([]);

    // ROI label
    const fontSize = 10 / p.zoom;
    dctx.font = fontSize + 'px system-ui';
    dctx.fillStyle = '#ffcc00';
    dctx.globalAlpha = 0.9;
    dctx.textAlign = 'left';
    dctx.textBaseline = 'top';
    let labelX, labelY;
    if (roi.type === 'circle') {
      labelX = roi.data.cx - roi.data.r;
      labelY = roi.data.cy - roi.data.r - fontSize * 1.3;
    } else if (roi.type === 'rect') {
      labelX = Math.min(roi.data.x1, roi.data.x2);
      labelY = Math.min(roi.data.y1, roi.data.y2) - fontSize * 1.3;
    } else if (roi.type === 'polygon') {
      labelX = roi.data.verts[0][0];
      labelY = roi.data.verts[0][1] - fontSize * 1.3;
    } else if (roi.type === 'path') {
      labelX = roi.data.waypoints[0][0];
      labelY = roi.data.waypoints[0][1] - fontSize * 1.3;
    }
    dctx.fillText(roi.name, labelX, labelY);
  }

  // Draw in-progress shape
  if (drawStart && drawCurrent && drawStart.panel === p) {
    dctx.strokeStyle = '#00ff88';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.7;
    dctx.setLineDash([3 / p.zoom, 2 / p.zoom]);

    if (drawMode === 'circle') {
      const dx = drawCurrent.x - drawStart.x;
      const dy = drawCurrent.y - drawStart.y;
      const r = Math.sqrt(dx * dx + dy * dy);
      dctx.beginPath();
      dctx.arc(drawStart.x, drawStart.y, r, 0, Math.PI * 2);
      dctx.stroke();
      // Radius text
      const fontSize = 10 / p.zoom;
      dctx.font = fontSize + 'px system-ui';
      dctx.fillStyle = '#00ff88';
      dctx.textAlign = 'center';
      dctx.fillText('r=' + r.toFixed(0) + ' \\u00b5m', drawStart.x, drawStart.y - r - fontSize);
    } else if (drawMode === 'rect') {
      const x = Math.min(drawStart.x, drawCurrent.x);
      const y = Math.min(drawStart.y, drawCurrent.y);
      const w = Math.abs(drawCurrent.x - drawStart.x);
      const h = Math.abs(drawCurrent.y - drawStart.y);
      dctx.strokeRect(x, y, w, h);
      // Dimensions text
      const fontSize = 10 / p.zoom;
      dctx.font = fontSize + 'px system-ui';
      dctx.fillStyle = '#00ff88';
      dctx.textAlign = 'center';
      dctx.fillText(w.toFixed(0) + ' \\u00d7 ' + h.toFixed(0) + ' \\u00b5m', x + w / 2, y - fontSize);
    }
    dctx.setLineDash([]);
  }

  // Draw in-progress polygon/path
  if ((drawMode === 'polygon' || drawMode === 'path') && polySlideIdx === p.idx && polyVerts.length > 0) {
    dctx.strokeStyle = '#00ff88';
    dctx.lineWidth = lw;
    dctx.globalAlpha = 0.7;
    dctx.beginPath();
    dctx.moveTo(polyVerts[0][0], polyVerts[0][1]);
    for (let i = 1; i < polyVerts.length; i++) {
      dctx.lineTo(polyVerts[i][0], polyVerts[i][1]);
    }
    dctx.stroke();
    // Draw vertices
    const vr = 3 / p.zoom;
    dctx.fillStyle = '#00ff88';
    for (const v of polyVerts) {
      dctx.beginPath();
      dctx.arc(v[0], v[1], vr, 0, Math.PI * 2);
      dctx.fill();
    }
    // Vertex count
    const fontSize = 10 / p.zoom;
    dctx.font = fontSize + 'px system-ui';
    dctx.textAlign = 'left';
    dctx.fillText(polyVerts.length + ' pts', polyVerts[polyVerts.length - 1][0] + 5 / p.zoom, polyVerts[polyVerts.length - 1][1]);
  }

  dctx.restore();
}

// ===================================================================
// ROI management
// ===================================================================
function addROI(slideIdx, type, data) {
  roiCounter++;
  const roi = {
    id: 'ROI_' + roiCounter,
    slideIdx,
    type,
    data,
    name: 'ROI_' + roiCounter,
    category: '',
  };
  rois.push(roi);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
  updateCorridorVisibility();
}

function deleteROI(id) {
  const idx = rois.findIndex(r => r.id === id);
  if (idx >= 0) rois.splice(idx, 1);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
  updateCorridorVisibility();
}

function updateROIList() {
  const div = document.getElementById('roi-list');
  div.innerHTML = '';
  for (const roi of rois) {
    const item = document.createElement('div');
    item.className = 'roi-item';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'roi-name';
    nameSpan.contentEditable = true;
    nameSpan.textContent = roi.name;
    nameSpan.title = SLIDES[roi.slideIdx].name + ' | ' + roi.type;
    nameSpan.onblur = () => { roi.name = nameSpan.textContent.trim() || roi.id; };
    nameSpan.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); nameSpan.blur(); } };

    const catSpan = document.createElement('span');
    catSpan.className = 'roi-category';
    catSpan.contentEditable = true;
    catSpan.textContent = roi.category || '';
    catSpan.title = 'Category: e.g. central_vein, portal_vein, liver';
    catSpan.onblur = () => { roi.category = catSpan.textContent.trim(); };
    catSpan.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); catSpan.blur(); } };

    const statsSpan = document.createElement('span');
    statsSpan.className = 'roi-stats';
    statsSpan.dataset.roiId = roi.id;

    const delBtn = document.createElement('span');
    delBtn.className = 'roi-del';
    delBtn.textContent = '\\u00d7';
    delBtn.onclick = () => deleteROI(roi.id);

    item.appendChild(nameSpan);
    item.appendChild(catSpan);
    item.appendChild(statsSpan);
    item.appendChild(delBtn);
    div.appendChild(item);
  }
}

function updateROIStats() {
  // Count cells inside each ROI
  for (const roi of rois) {
    let count = 0;
    const slide = SLIDES[roi.slideIdx];
    const pos = slide.pos;
    const grp = slide.grp;
    for (let i = 0; i < slide.n; i++) {
      if (hidden.has(GROUP_LABELS[grp[i]])) continue;
      if (pointInROI(pos[i * 2], pos[i * 2 + 1], roi)) count++;
    }
    const el = document.querySelector('[data-roi-id="' + roi.id + '"]');
    if (el) el.textContent = count.toLocaleString();
  }

  const statsDiv = document.getElementById('roi-stats');
  if (rois.length === 0) {
    statsDiv.textContent = '';
  } else {
    statsDiv.textContent = rois.length + ' ROI(s) drawn';
  }
}

function updateCorridorVisibility() {
  const hasPath = rois.some(r => r.type === 'path');
  document.getElementById('corridor-row').style.display = hasPath ? 'flex' : 'none';
}

function downloadROIs() {
  const out = {
    rois: [],
    metadata: {
      generated: GENERATED,
      title: TITLE,
      group_field: GROUP_FIELD,
    },
  };
  for (const roi of rois) {
    const slideName = SLIDES[roi.slideIdx].name;
    const entry = { id: roi.id, slide: slideName, type: roi.type, name: roi.name };
    entry.category = roi.category || '';
    if (roi.type === 'circle') {
      entry.center_um = [roi.data.cx, roi.data.cy];
      entry.radius_um = roi.data.r;
    } else if (roi.type === 'rect') {
      entry.min_um = [Math.min(roi.data.x1, roi.data.x2), Math.min(roi.data.y1, roi.data.y2)];
      entry.max_um = [Math.max(roi.data.x1, roi.data.x2), Math.max(roi.data.y1, roi.data.y2)];
    } else if (roi.type === 'polygon') {
      entry.vertices_um = roi.data.verts;
    } else if (roi.type === 'path') {
      entry.waypoints_um = roi.data.waypoints;
      entry.corridor_um = roi.data.corridorWidth || corridorWidth;
    }
    out.rois.push(entry);
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'rois.json';
  a.click();
  URL.revokeObjectURL(url);
}

// ===================================================================
// Legend
// ===================================================================
function initLegend() {
  const legDiv = document.getElementById('leg-items');
  // Compute total counts per group
  const totals = new Array(N_GROUPS).fill(0);
  for (const slide of SLIDES) {
    for (let i = 0; i < slide.n; i++) {
      totals[slide.grp[i]]++;
    }
  }

  for (let gi = 0; gi < N_GROUPS; gi++) {
    const item = document.createElement('div');
    item.className = 'leg-item';
    item.dataset.gi = gi;

    const dot = document.createElement('span');
    dot.className = 'leg-dot';
    dot.style.background = GROUP_COLORS[gi];

    const label = document.createElement('span');
    label.className = 'leg-label';
    label.title = GROUP_LABELS[gi];
    label.textContent = GROUP_LABELS[gi];

    const count = document.createElement('span');
    count.className = 'leg-count';
    count.textContent = totals[gi].toLocaleString();

    item.appendChild(dot);
    item.appendChild(label);
    item.appendChild(count);

    item.onclick = () => {
      const lbl = GROUP_LABELS[gi];
      if (hidden.has(lbl)) {
        hidden.delete(lbl);
        item.classList.remove('hidden');
      } else {
        hidden.add(lbl);
        item.classList.add('hidden');
      }
      kdeCache.clear();
      reclusterAll();
      scheduleRenderAll();
    };
    legDiv.appendChild(item);
  }
}

// ===================================================================
// Controls
// ===================================================================
function initControls() {
  // Dot size
  document.getElementById('dot-size').oninput = e => {
    dotSize = parseFloat(e.target.value);
    document.getElementById('dot-val').textContent = dotSize;
    scheduleRenderAll();
  };

  // Opacity
  document.getElementById('opacity').oninput = e => {
    dotAlpha = parseFloat(e.target.value);
    document.getElementById('op-val').textContent = dotAlpha.toFixed(2);
    scheduleRenderAll();
  };

  // Show all / hide all
  document.getElementById('btn-show-all').onclick = () => {
    hidden.clear();
    document.querySelectorAll('.leg-item').forEach(el => el.classList.remove('hidden'));
    kdeCache.clear();
    reclusterAll();
    scheduleRenderAll();
  };
  document.getElementById('btn-hide-all').onclick = () => {
    GROUP_LABELS.forEach(l => hidden.add(l));
    document.querySelectorAll('.leg-item').forEach(el => el.classList.add('hidden'));
    kdeCache.clear();
    reclusterAll();
    scheduleRenderAll();
  };

  // Reset zoom (sidebar button + floating button)
  const resetZoomFn = () => {
    resizePanels();
    panels.forEach(fitPanel);
    scheduleRenderAll();
    panels.forEach(p => renderDrawOverlay(p));
  };
  document.getElementById('btn-reset-zoom').onclick = resetZoomFn;
  document.getElementById('floating-reset-zoom').onclick = resetZoomFn;

  // Slide jump
  document.getElementById('slide-select').onchange = e => {
    const idx = parseInt(e.target.value);
    if (isNaN(idx) || !panels[idx]) return;
    if (focusedIdx >= 0) {
      // In focus view: switch to this slide
      exitFocusView();
      setTimeout(() => enterFocusView(idx), 100);
    } else {
      panels[idx].div.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // Draw mode buttons
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.onclick = () => {
      drawMode = btn.dataset.mode;
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      panels.forEach(p => {
        if (drawMode === 'pan') {
          p.div.classList.remove('draw-mode');
        } else {
          p.div.classList.add('draw-mode');
        }
      });
      // Clear in-progress drawing
      drawStart = null;
      drawCurrent = null;
      if (drawMode !== 'polygon' && drawMode !== 'path') {
        polyVerts = [];
        polySlideIdx = -1;
      }
      panels.forEach(p => {
        p.measureEl.style.display = 'none';
        renderDrawOverlay(p);
      });
    };
  });

  // ROI controls
  document.getElementById('btn-download-roi').onclick = downloadROIs;
  document.getElementById('roi-filter').onchange = e => {
    roiFilterActive = e.target.checked;
    scheduleRenderAll();
  };
  document.getElementById('corridor-slider').oninput = e => {
    corridorWidth = parseFloat(e.target.value);
    document.getElementById('corridor-val').textContent = corridorWidth;
    rois.forEach(r => { if (r.type === 'path') r.data.corridorWidth = corridorWidth; });
    panels.forEach(p => renderDrawOverlay(p));
    updateROIStats();
    if (roiFilterActive) scheduleRenderAll();
  };

  // Clustering controls
  const epsSlider = document.getElementById('eps-slider');
  const minCellsSlider = document.getElementById('min-cells');
  epsSlider.oninput = e => {
    document.getElementById('eps-val').textContent = parseFloat(e.target.value).toFixed(2);
  };
  epsSlider.onchange = () => { reclusterAll(); scheduleRenderAll(); };
  minCellsSlider.oninput = e => {
    document.getElementById('min-cells-val').textContent = e.target.value;
  };
  minCellsSlider.onchange = () => { reclusterAll(); scheduleRenderAll(); };
  document.getElementById('show-hulls').onchange = e => {
    showHulls = e.target.checked;
    scheduleRenderAll();
  };
  document.getElementById('show-labels').onchange = e => {
    showLabels = e.target.checked;
    scheduleRenderAll();
  };

  // KDE controls
  const showKDECb = document.getElementById('show-kde');
  if (showKDECb) {
    showKDECb.onchange = e => { showKDE = e.target.checked; scheduleRenderAll(); };

    function kdeDebounced() {
      if (kdeDebounceTimer) clearTimeout(kdeDebounceTimer);
      kdeDebounceTimer = setTimeout(() => { kdeCache.clear(); scheduleRenderAll(); }, 200);
    }

    document.getElementById('kde-bw').oninput = e => {
      kdeBWIdx = parseInt(e.target.value);
      document.getElementById('kde-bw-val').textContent = KDE_RADII[kdeBWIdx] + ' \\u00b5m';
      kdeDebounced();
    };
    document.getElementById('kde-levels').oninput = e => {
      kdeLevels = parseInt(e.target.value);
      document.getElementById('kde-levels-val').textContent = kdeLevels;
      kdeDebounced();
    };
    document.getElementById('kde-opacity').oninput = e => {
      kdeAlpha = parseFloat(e.target.value);
      document.getElementById('kde-op-val').textContent = kdeAlpha.toFixed(2);
      scheduleRenderAll();
    };
    document.getElementById('kde-fill').onchange = e => { kdeFill = e.target.checked; scheduleRenderAll(); };
    document.getElementById('kde-lines').onchange = e => { kdeLines = e.target.checked; scheduleRenderAll(); };
  }

  // Region controls
  const showRegCb = document.getElementById('show-regions');
  if (showRegCb) {
    showRegCb.onchange = e => { showRegions = e.target.checked; scheduleRenderAll(); };
    document.getElementById('show-region-labels').onchange = e => { showRegionLabels = e.target.checked; scheduleRenderAll(); };
    document.getElementById('show-region-bnd').onchange = e => { showRegionBnd = e.target.checked; scheduleRenderAll(); };
    document.getElementById('region-opacity').oninput = e => {
      regionAlpha = parseFloat(e.target.value);
      document.getElementById('region-op-val').textContent = regionAlpha.toFixed(2);
      scheduleRenderAll();
    };

    // Scale slider (multi-scale regions)
    const scaleSlider = document.getElementById('region-scale');
    if (scaleSlider) {
      scaleSlider.oninput = e => {
        const idx = parseInt(e.target.value);
        const key = String(SCALE_KEYS[idx]);
        document.getElementById('region-scale-val').textContent = key + ' \\u00b5m';
        for (const slide of SLIDES) {
          if (slide.regionScales && slide.regionScales[key]) {
            slide.regions = slide.regionScales[key];
          }
        }
        scheduleRenderAll();
      };
    }
  }

  // Fluorescence controls
  const showFluorCb = document.getElementById('show-fluor');
  if (showFluorCb) {
    showFluorCb.onchange = e => { showFluor = e.target.checked; scheduleRenderAll(); };
  }
  const fluorOpSlider = document.getElementById('fluor-opacity');
  if (fluorOpSlider) {
    fluorOpSlider.oninput = e => {
      fluorAlpha = parseFloat(e.target.value);
      document.getElementById('fluor-op-val').textContent = fluorAlpha.toFixed(2);
      scheduleRenderAll();
    };
  }
  for (let ci = 0; ci < 3; ci++) {
    const btn = document.getElementById('btn-ch' + ci);
    if (btn) {
      btn.onclick = () => {
        chEnabled[ci] = !chEnabled[ci];
        btn.classList.toggle('active', chEnabled[ci]);
        // Invalidate composited canvas for all slides
        fluorImages.forEach(fd => { if (fd) fd._dirty = true; });
        scheduleRenderAll();
      };
    }
  }
  const showContoursCb = document.getElementById('show-contours');
  if (showContoursCb) {
    showContoursCb.onchange = e => { showContours = e.target.checked; scheduleRenderAll(); };
  }
  const showDotsCb = document.getElementById('show-dots');
  if (showDotsCb) {
    showDotsCb.onchange = e => { showDots = e.target.checked; scheduleRenderAll(); };
  }
}

// ===================================================================
// Init
// ===================================================================
initPanels();
initLegend();
initControls();

function fullInit() {
  resizePanels();
  panels.forEach(fitPanel);
  reclusterAll();
  scheduleRenderAll();
}

// Single-slide: go straight to focus-like rendering (full panel)
if (IS_SINGLE && panels.length === 1) {
  panels[0].visible = true;
}

setTimeout(fullInit, 80);

window.addEventListener('resize', () => {
  if (focusedIdx >= 0) {
    resizePanel(panels[focusedIdx]);
  } else {
    resizePanels();
  }
  scheduleRenderAll();
  panels.forEach(p => renderDrawOverlay(p));
});
"""
    )

    html_parts.append("</script>\n</body>\n</html>")

    html_content = "".join(html_parts)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-slide spatial viewer HTML from classified detections"
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing per-slide subdirectories "
        "(or a single slide dir with classified JSON)",
    )
    parser.add_argument("--detections", nargs="+", help="Explicit list of detection JSON files")
    parser.add_argument(
        "--detection-glob",
        default="cell_detections_classified.json",
        help="Glob pattern for detection files within slide subdirs "
        "(default: cell_detections_classified.json)",
    )
    parser.add_argument(
        "--group-field",
        required=True,
        help="Field in features dict to color cells by "
        "(e.g. tdTomato_class, MSLN_class, marker_profile)",
    )
    parser.add_argument(
        "--group-label-prefix",
        default=None,
        help='Prefix for legend labels (e.g. "nuclei" turns "2" into "nuclei: 2")',
    )
    parser.add_argument("--title", default="Multi-Slide Spatial Overview", help="HTML page title")
    parser.add_argument(
        "--output", default=None, help="Output HTML path (default: {input-dir}/spatial_viewer.html)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help='Keep top N groups by cell count, lump rest into "other"',
    )
    parser.add_argument(
        "--exclude-groups", default=None, help="Comma-separated group labels to exclude entirely"
    )
    parser.add_argument(
        "--default-min-cells", type=int, default=10, help="Default DBSCAN min_samples (default: 10)"
    )
    parser.add_argument(
        "--min-hull-cells",
        type=int,
        default=24,
        help="Min cells in cluster to draw convex hull (default: 24)",
    )
    parser.add_argument(
        "--no-graph-patterns",
        action="store_true",
        help="Disable graph-based spatial pattern regions (enabled by default)",
    )
    parser.add_argument(
        "--connect-radius",
        type=float,
        nargs="+",
        default=[50, 100, 200, 300, 400, 500, 600, 700, 800, 1000],
        help="Connection radii in um for graph patterns (default: 10 scales)",
    )
    parser.add_argument(
        "--min-region-cells",
        type=int,
        default=8,
        help="Min cells per connected component for regions (default: 8)",
    )
    # Fluorescence background
    parser.add_argument(
        "--czi-path",
        help="CZI file for fluorescence background (single slide or matched " "to all slides)",
    )
    parser.add_argument("--czi-dir", help="Directory of CZI files matched to slides by stem name")
    parser.add_argument(
        "--display-channels",
        default=None,
        help='Channel indices for R,G,B display (e.g. "1,2,0"). '
        "Default: first 3 channels (0,1,2).",
    )
    parser.add_argument(
        "--channel-names",
        default=None,
        help='Override channel names for R,G,B legend (e.g. "SMA,CD31,nuc"). '
        "Default: auto-detected from CZI filename.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.0625,
        help="CZI downsample factor for background image (default: 1/16)",
    )
    parser.add_argument(
        "--scene",
        type=int,
        default=None,
        help="CZI scene index for single-scene viewing (auto-creates scene_N directory structure)",
    )
    # Detection contours (on by default)
    parser.add_argument(
        "--no-contours",
        action="store_true",
        help="Disable detection contour embedding (enabled by default when "
        "detections have outer_contour_global)",
    )
    parser.add_argument(
        "--contour-score-threshold",
        type=float,
        default=None,
        help="Only embed contours for detections with score >= threshold",
    )
    parser.add_argument(
        "--max-contours",
        type=int,
        default=100_000,
        help="Maximum contours to embed per slide (default: 100000)",
    )
    parser.add_argument(
        "--marker-filter",
        default=None,
        help='Filter detections by marker class (e.g., "MSLN_class==positive")',
    )
    args = parser.parse_args()

    if not args.input_dir and not args.detections:
        parser.error("Provide either --input-dir or --detections")

    # Determine output path
    if args.output is None:
        if args.input_dir:
            args.output = str(Path(args.input_dir) / "spatial_viewer.html")
        else:
            args.output = "spatial_viewer.html"

    # --scene shortcut: wrap a single detection file as scene_N for correct
    # CZI scene loading (no manual symlink setup needed)
    if args.scene is not None and args.detections:
        scene_name = f"scene_{args.scene}"
        slide_files = [(scene_name, Path(args.detections[0]))]
        print(f"Single-scene mode: {scene_name} from {args.detections[0]}")
    elif args.input_dir:
        slide_files = discover_slides(args.input_dir, args.detection_glob)
        if not slide_files:
            print(
                f"Error: no detection files matching '{args.detection_glob}' "
                f"found in {args.input_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Found {len(slide_files)} slides in {args.input_dir}")
    else:
        slide_files = []
        for p in args.detections:
            path = Path(p)
            name = path.parent.name if path.parent.name else path.stem
            slide_files.append((name, path))

    # Load data
    want_contours = not args.no_contours
    slides_data = []
    for name, path in slide_files:
        print(f"  Loading {name}...", end="", flush=True)
        data = load_slide_data(
            path,
            args.group_field,
            include_contours=want_contours,
            score_threshold=args.contour_score_threshold,
            marker_filter=args.marker_filter,
        )
        if data is None:
            print(" skipped (no data)")
            continue
        groups_str = ", ".join(f"{g['label']}:{g['n']}" for g in data["groups"])
        slides_data.append((name, data))
        n_contours = len(data.get("contours_raw", []))
        extra = f", {n_contours} contours" if want_contours else ""
        print(f" {data['n_cells']} cells [{groups_str}]{extra}")

    if not slides_data:
        print("Error: no valid slide data loaded", file=sys.stderr)
        sys.exit(1)

    # Apply group label prefix (e.g. "nuclei" turns "2" into "nuclei: 2")
    if args.group_label_prefix:
        pfx = args.group_label_prefix
        for _, data in slides_data:
            for g in data["groups"]:
                g["label"] = f"{pfx}: {g['label']}"

    # Apply top-N filtering and exclusions
    exclude_groups = set()
    if args.exclude_groups:
        exclude_groups = {s.strip() for s in args.exclude_groups.split(",")}
    if args.top_n or exclude_groups:
        apply_top_n_filtering(slides_data, args.top_n, exclude_groups)

    # Assign colors
    color_map = assign_group_colors(slides_data)
    print(f"\nGroups: {', '.join(f'{k} ({v})' for k, v in sorted(color_map.items()))}")

    # Compute graph-pattern regions if requested
    has_regions = False
    has_multiscale = False
    scale_keys = None
    if not args.no_graph_patterns:
        radii = sorted(args.connect_radius)
        scale_keys = [str(int(r)) for r in radii]
        has_multiscale = len(radii) > 1
        mid_idx = len(radii) // 2

        for name, data in slides_data:
            # Build position/type arrays from groups
            pos_list = []
            type_list = []
            type_labels = []
            type_colors = []
            for gi, g in enumerate(data["groups"]):
                type_labels.append(g["label"])
                type_colors.append(g["color"])
                pos_list.append(np.column_stack([g["x"], g["y"]]))
                type_list.append(np.full(g["n"], gi, dtype=np.int32))

            positions = np.vstack(pos_list)
            types_arr = np.concatenate(type_list)

            print(f"  Computing graph patterns for {name}...")
            if has_multiscale:
                scales = {}
                tree_cache = {}  # reuse KDTrees across radii
                for r in radii:
                    scales[str(int(r))] = compute_graph_patterns(
                        positions,
                        types_arr,
                        type_labels,
                        type_colors,
                        connect_radius_um=r,
                        min_cluster_cells=args.min_region_cells,
                        boundary_dilate_um=r * 0.4,
                        _cached_trees=tree_cache,
                    )
                data["region_scales"] = scales
                data["regions"] = scales[str(int(radii[mid_idx]))]
            else:
                data["regions"] = compute_graph_patterns(
                    positions,
                    types_arr,
                    type_labels,
                    type_colors,
                    connect_radius_um=radii[0],
                    min_cluster_cells=args.min_region_cells,
                    boundary_dilate_um=radii[0] * 0.4,
                )

        has_regions = any(data.get("regions") for _, data in slides_data)

    # Build contour data per slide
    contour_data = None
    if want_contours:
        contour_data = {}
        for name, data in slides_data:
            raw = data.pop("contours_raw", [])
            if raw:
                cd = build_contour_js_data(raw, max_contours=args.max_contours)
                contour_data[name] = cd
                print(f"  Contours for {name}: {len(raw)} raw -> {len(cd)} embedded")

    # Load fluorescence backgrounds from CZI files
    fluor_data = None
    ch_names = None
    if args.czi_path or args.czi_dir:
        display_channels = [0, 1, 2]
        if args.display_channels:
            display_channels = [int(x.strip()) for x in args.display_channels.split(",")]
        display_channels = display_channels[:3]

        # Build CZI path map: slide_name -> Path (or '*' for single CZI)
        czi_map = {}
        if args.czi_path:
            czi_map["*"] = Path(args.czi_path)
        elif args.czi_dir:
            for czi_file in sorted(Path(args.czi_dir).glob("*.czi")):
                czi_map[czi_file.stem] = czi_file

        fluor_data = {}
        ch_names_collected = None
        for name, _ in slides_data:
            # Find matching CZI: exact stem match, then wildcard, then fuzzy
            czi_path = czi_map.get(name) or czi_map.get("*")
            if czi_path is None:
                for stem, path in czi_map.items():
                    if stem != "*" and (name in stem or stem in name):
                        czi_path = path
                        break
            if czi_path is None:
                print(f"  No CZI found for slide '{name}', skipping fluorescence")
                continue

            # For multi-scene CZIs: derive scene index from panel name
            # (e.g. "scene_3" -> scene=3). Single-scene slides return 0.
            _scene_idx = parse_scene_index(name)

            print(
                f"  Loading fluorescence for '{name}' from {czi_path.name}"
                + (f" (scene {_scene_idx})" if _scene_idx else "")
                + "..."
            )
            # Check for cached thumbnail (avoids slow CZI re-reads during iteration)
            _cache_dir = Path(args.output).parent if args.output else Path(".")
            _ch_key = "_".join(str(c) for c in display_channels)
            _scale_key = f"{args.scale_factor:.4f}"
            _cache_stem = czi_path.stem + (f"_scene{_scene_idx}" if _scene_idx else "")
            _cache_file = (
                _cache_dir / f".thumbnail_cache_{_cache_stem}_ch{_ch_key}_s{_scale_key}.npz"
            )
            ch_arrays = None
            pixel_size = None
            mx = my = 0
            if _cache_file.exists():
                try:
                    _cached = np.load(str(_cache_file))
                    ch_arrays = [
                        _cached[f"ch{i}"] if _cached[f"ch{i}"].size > 0 else None
                        for i in range(len(display_channels))
                    ]
                    _ps = str(_cached["pixel_size"])
                    pixel_size = float(_ps) if _ps != "None" else None
                    mx = int(_cached["mosaic_x"])
                    my = int(_cached["mosaic_y"])
                    print(f"    Using cached thumbnail: {_cache_file.name}", flush=True)
                except Exception:
                    ch_arrays = None  # fall through to CZI read

            if ch_arrays is None:
                try:
                    ch_arrays, pixel_size, mx, my = read_czi_thumbnail_channels(
                        czi_path,
                        display_channels,
                        scale_factor=args.scale_factor,
                        scene=_scene_idx,
                    )
                    # Cache for next time
                    try:
                        _save = {
                            f"ch{i}": (arr if arr is not None else np.empty(0, dtype=np.uint8))
                            for i, arr in enumerate(ch_arrays)
                        }
                        _save["pixel_size"] = str(pixel_size) if pixel_size is not None else "None"
                        _save["mosaic_x"] = mx
                        _save["mosaic_y"] = my
                        np.savez_compressed(str(_cache_file), **_save)
                        print(f"    Cached thumbnail: {_cache_file.name}", flush=True)
                    except Exception:
                        pass  # caching is best-effort
                except Exception as exc:
                    print(f"  WARNING: failed to load CZI for '{name}': {exc}", file=sys.stderr)
                    continue

            if pixel_size is None:
                # Try to derive from detection features (area vs area_um2)
                import math as _math

                for _det in (data.get("_raw_detections") or [])[:100]:
                    _f = _det.get("features", {})
                    if _f.get("area") and _f.get("area_um2") and _f["area"] > 0:
                        pixel_size = _math.sqrt(_f["area_um2"] / _f["area"])
                        break
                if pixel_size is None:
                    raise ValueError(
                        f"Could not determine pixel size for '{name}': "
                        f"no area/area_um2 features found in detections. "
                        f"Ensure detections have both 'area' and 'area_um2' features."
                    )

            # Determine channel names by matching filename markers to CZI channels
            # via wavelength. Filename marker order ≠ CZI channel order, so we
            # resolve each CZI channel's wavelength to the filename marker name.
            from xldvp_seg.io.czi_loader import parse_markers_from_filename

            markers = parse_markers_from_filename(czi_path.name)
            # Build wavelength → marker name lookup from filename
            wl_to_name = {}
            for m in markers:
                if m.get("wavelength") and m["wavelength"] not in wl_to_name:
                    wl_to_name[m["wavelength"]] = m["name"]

            # Get CZI channel wavelengths from metadata
            try:
                import re

                import aicspylibczi

                czi_file = aicspylibczi.CziFile(str(czi_path))
                czi_root = czi_file.meta
                czi_channels = []
                seen = set()
                for ch_el in czi_root.iter("Channel"):
                    ch_name = ch_el.get("Name", "")
                    if ch_name in seen:
                        continue  # skip duplicates from multi-scene metadata
                    seen.add(ch_name)
                    # Extract excitation wavelength: try metadata field first,
                    # then parse from channel name (e.g., 'AF488' → 488)
                    ex_wl = None
                    for ex in ch_el.iter("ExcitationWavelength"):
                        try:
                            ex_wl = int(round(float(ex.text)))
                        except (TypeError, ValueError):
                            pass
                    if ex_wl is None and ch_name:
                        wl_match = re.search(r"(\d{3})", ch_name)
                        if wl_match:
                            ex_wl = int(wl_match.group(1))
                    czi_channels.append({"name": ch_name, "wavelength": ex_wl})
            except Exception as e:
                print(f"  Warning: could not parse CZI channel metadata: {e}")
                czi_channels = []

            this_ch_names = []
            for ch_idx in display_channels:
                ch_label = f"Ch{ch_idx}"
                if ch_idx < len(czi_channels):
                    wl = czi_channels[ch_idx].get("wavelength")
                    if wl and wl in wl_to_name:
                        ch_label = wl_to_name[wl]
                    elif wl:
                        ch_label = f"{wl}nm"
                this_ch_names.append(ch_label)
            if ch_names_collected is None:
                ch_names_collected = this_ch_names

            # Encode channels as base64 PNGs
            ch_b64 = []
            for ch_arr in ch_arrays:
                if ch_arr is None:
                    ch_b64.append("")
                else:
                    ch_b64.append(encode_channel_b64(ch_arr))
            while len(ch_b64) < 3:
                ch_b64.append("")

            h, w = ch_arrays[0].shape if ch_arrays[0] is not None else (0, 0)
            fluor_data[name] = {
                "channels": ch_b64,
                "names": this_ch_names,
                "width": w,
                "height": h,
                "scale": args.scale_factor,
                "mosaic_x": mx,
                "mosaic_y": my,
                "pixel_size": pixel_size,
            }
            print(
                f"    Encoded {sum(1 for b in ch_b64 if b)} channels " f"({w}x{h} px thumbnail)",
                flush=True,
            )

        ch_names = ch_names_collected
        # CLI override for channel names (when auto-detection from filename is wrong)
        if args.channel_names:
            ch_names = [n.strip() for n in args.channel_names.split(",")][:3]
        if not fluor_data:
            fluor_data = None
            print("  No fluorescence data loaded")

    # Generate HTML
    total_cells = sum(d["n_cells"] for _, d in slides_data)
    print(f"\nGenerating HTML for {len(slides_data)} slides, " f"{total_cells:,} total cells...")
    generate_html(
        slides_data,
        args.output,
        color_map,
        title=args.title,
        group_field=args.group_field,
        default_min_cells=args.default_min_cells,
        min_hull_cells=args.min_hull_cells,
        has_regions=has_regions,
        has_multiscale=has_multiscale,
        scale_keys=scale_keys,
        fluor_data=fluor_data,
        contour_data=contour_data,
        ch_names=ch_names,
    )


if __name__ == "__main__":
    main()
