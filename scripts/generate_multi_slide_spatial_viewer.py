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
from xldvp_seg.visualization.js_loader import load_js

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

    # Load all JS components from xldvp_seg/visualization/js/
    js_code = load_js(
        # Utility functions (no dependencies)
        "base64_decode",
        "coordinate",
        # Canvas/panel management (depends on: renderPanel via RAF callback)
        "canvas_setup",
        # Pan/zoom event handlers (depends on: screenToData, scheduleRender, renderDrawOverlay, addROI)
        "pan_zoom",
        # ROI geometry (depends on: roiFilterActive, rois, corridorWidth)
        "roi_geometry",
        # ROI overlay drawing (depends on: rois, drawMode, drawStart, drawCurrent, polyVerts, polySlideIdx)
        "roi_drawing",
        # ROI CRUD (depends on: SLIDES, GROUP_LABELS, hidden, rois, roiCounter, roiFilterActive,
        #   GENERATED, TITLE, GROUP_FIELD, corridorWidth, panels, renderDrawOverlay, scheduleRenderAll)
        "roi_management",
        # Clustering algorithms (no dependencies on viewer state)
        "dbscan",
        "convex_hull",
        # KDE density (depends on: SLIDES, GROUP_LABELS, GROUP_COLORS, N_GROUPS, hidden, kdeCache)
        "kde",
        # Fluorescence compositing (depends on: fluorImages, chEnabled, fluorAlpha, CH_TINTS)
        "fluorescence",
        # Detection contour rendering (depends on: CONTOUR_DATA)
        "contour_draw",
        # Focus view (depends on: IS_SINGLE, panels, focusedIdx, resizePanel, fitPanel, etc.)
        "focus_view",
        # Region drawing (depends on: hidden)
        "regions",
        # Main render function (depends on: all drawing functions + state)
        "render_panel",
        # Legend + clustering + all sidebar controls (depends on: all state + functions)
        "controls",
        # Data decode, state declarations, panel init, boot sequence (depends on: everything above)
        "init",
    )
    html_parts.append(js_code)
    html_parts.append("\n</script>\n</body>\n</html>\n")

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
