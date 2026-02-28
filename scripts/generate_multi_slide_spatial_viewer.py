#!/usr/bin/env python
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
import base64
import html as html_mod
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Binary positive/negative
BINARY_COLORS = {'positive': '#ff4444', 'negative': '#4488ff'}

# 4-group palette (multi-marker profiles)
QUAD_COLORS = ['#ff4444', '#4488ff', '#44cc44', '#ff8844']

# 20-color maximally-distinct palette for N groups
AUTO_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabebe', '#469990',
    '#e6beff', '#9a6324', '#ffe119', '#aaffc3', '#800000',
    '#ffd8b1', '#000075', '#a9a9a9', '#808000', '#ff69b4',
]


def hsl_palette(n):
    """Generate n maximally-separated HSL colors as hex strings."""
    colors = []
    for i in range(n):
        h = (i * 360 / n) % 360
        s = 70 + (i % 3) * 10  # 70-90% saturation
        l = 55 + (i % 2) * 10  # 55-65% lightness
        colors.append(_hsl_to_hex(h, s, l))
    return colors


def _hsl_to_hex(h, s, l):
    """Convert HSL (h=0-360, s=0-100, l=0-100) to hex color string."""
    s /= 100
    l /= 100
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    ri = int((r + m) * 255)
    gi = int((g + m) * 255)
    bi = int((b + m) * 255)
    return f'#{ri:02x}{gi:02x}{bi:02x}'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def extract_position_um(det):
    """Extract (x, y) position in microns from a detection dict.

    Tries global_center_um first, then falls back to global_x / global_y
    (pixel coords multiplied by pixel_size_um).

    Returns (x, y) tuple or None if position unavailable.
    """
    # Primary: global_center_um in features
    pos = det.get('features', {}).get('global_center_um')
    if pos is None:
        pos = det.get('global_center_um')
    if pos is not None and len(pos) == 2:
        x, y = float(pos[0]), float(pos[1])
        if np.isfinite(x) and np.isfinite(y):
            return (x, y)

    # Fallback: pixel coordinates * pixel_size
    gx = det.get('global_x')
    gy = det.get('global_y')
    if gx is not None and gy is not None:
        pixel_size = det.get('features', {}).get('pixel_size_um')
        if pixel_size is None or not isinstance(pixel_size, (int, float)):
            return None  # never hardcode pixel_size — CZI metadata is ground truth
        x = float(gx) * float(pixel_size)
        y = float(gy) * float(pixel_size)
        if np.isfinite(x) and np.isfinite(y):
            return (x, y)

    return None


def extract_group(det, group_field):
    """Extract group label from a detection dict.

    Checks top-level dict first, then features sub-dict, falls back to
    'unknown' if the field is missing everywhere.
    """
    val = det.get(group_field)
    if val is None:
        val = det.get('features', {}).get(group_field)
    if val is None:
        return 'unknown'
    return str(val)


def load_slide_data(path, group_field):
    """Load a classified detection JSON and extract positions + groups.

    Args:
        path: Path to classified detection JSON.
        group_field: Field name to group by.

    Returns:
        Dict with slide data, or None if no valid data.
    """
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping", file=sys.stderr)
        return None

    with open(path, encoding='utf-8') as f:
        detections = json.load(f)

    if not isinstance(detections, list):
        print(f"  WARNING: {path} is not a JSON list, skipping", file=sys.stderr)
        return None

    group_cells = {}  # group_label -> list of (x, y)

    for det in detections:
        pos = extract_position_um(det)
        if pos is None:
            continue
        group = extract_group(det, group_field)
        group_cells.setdefault(group, []).append(pos)

    if not group_cells:
        return None

    groups_out = []
    for label, cells in sorted(group_cells.items()):
        arr = np.array(cells, dtype=np.float32)
        groups_out.append({
            'label': label,
            'n': len(cells),
            'x': arr[:, 0],
            'y': arr[:, 1],
        })

    all_x = np.concatenate([g['x'] for g in groups_out])
    all_y = np.concatenate([g['y'] for g in groups_out])

    return {
        'groups': groups_out,
        'n_cells': sum(g['n'] for g in groups_out),
        'x_range': [float(all_x.min()), float(all_x.max())],
        'y_range': [float(all_y.min()), float(all_y.max())],
    }


def discover_slides(input_dir, detection_glob):
    """Discover per-slide detection files in subdirectories.

    Also checks for a detection file directly in input_dir (single-slide
    case).

    Returns list of (slide_name, detection_path) tuples.
    """
    input_dir = Path(input_dir)
    results = []
    seen_paths = set()

    # Check for file directly in input_dir
    direct = list(input_dir.glob(detection_glob))
    if direct:
        rp = direct[0].resolve()
        if rp not in seen_paths:
            seen_paths.add(rp)
            results.append((input_dir.name, direct[0]))

    # Check subdirectories
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        matches = list(subdir.glob(detection_glob))
        if matches:
            rp = matches[0].resolve()
            if rp not in seen_paths:
                seen_paths.add(rp)
                results.append((subdir.name, matches[0]))

    return results


def assign_group_colors(slides_data):
    """Assign colors to groups across all slides.

    - 2 groups with positive/negative: red/blue
    - 2 arbitrary groups: red/blue
    - 4 groups: red/blue/green/orange
    - N groups (N <= 20): auto palette
    - N groups (N > 20): HSL-generated palette
    """
    all_groups = set()
    for _, data in slides_data:
        for g in data['groups']:
            all_groups.add(g['label'])

    n = len(all_groups)
    sorted_groups = sorted(all_groups)

    if all_groups == {'positive', 'negative'}:
        color_map = dict(BINARY_COLORS)
    elif n <= 2:
        palette = ['#ff4444', '#4488ff']
        color_map = {lbl: palette[i] for i, lbl in enumerate(sorted_groups)}
    elif n <= 4:
        color_map = {lbl: QUAD_COLORS[i] for i, lbl in enumerate(sorted_groups)}
    elif n <= 20:
        color_map = {lbl: AUTO_COLORS[i] for i, lbl in enumerate(sorted_groups)}
    else:
        palette = hsl_palette(n)
        color_map = {lbl: palette[i] for i, lbl in enumerate(sorted_groups)}

    # Apply colors to group dicts
    for _, data in slides_data:
        for g in data['groups']:
            g['color'] = color_map[g['label']]

    return color_map


# ---------------------------------------------------------------------------
# Binary data encoding
# ---------------------------------------------------------------------------

def encode_float32_base64(arr):
    """Encode a numpy float32 array as base64 string (little-endian)."""
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii')


def encode_uint8_base64(arr):
    """Encode a numpy uint8 array as base64 string."""
    return base64.b64encode(arr.astype(np.uint8).tobytes()).decode('ascii')


def safe_json(obj):
    """JSON-encode an object safe for embedding in <script> blocks.

    Escapes '</' sequences to prevent premature </script> termination (XSS).
    """
    return json.dumps(obj).replace('</', '<\\/')


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(slides_data, output_path, color_map, title, group_field):
    """Generate self-contained scrollable HTML with focus view and ROI drawing.

    Data is embedded as base64-encoded TypedArrays for compact transfer.
    Each slide stores a single Float32Array of interleaved [x0,y0,x1,y1,...]
    positions and a Uint8Array of group indices.

    Args:
        slides_data: List of (slide_name, data_dict) tuples.
        output_path: Output HTML file path.
        color_map: Dict of group_label -> hex color.
        title: Page title.
        group_field: Group field name (for metadata export).
    """
    title_escaped = html_mod.escape(title)

    # Build group label -> index mapping (consistent across all slides)
    group_labels = sorted(color_map.keys())
    if len(group_labels) > 255:
        print(f"WARNING: {len(group_labels)} groups exceeds Uint8 limit (255). "
              f"Keeping top 254 groups, collapsing rest into 'other'.", file=sys.stderr)
        # Keep the 254 most common groups, collapse the rest
        all_counts = {}
        for _, data in slides_data:
            for g in data['groups']:
                all_counts[g['label']] = all_counts.get(g['label'], 0) + g['n']
        top_labels = sorted(all_counts, key=all_counts.get, reverse=True)[:254]
        group_labels = sorted(top_labels) + ['other']
        # Re-map collapsed groups in color_map
        other_color = '#808080'
        color_map = {lbl: color_map.get(lbl, other_color) for lbl in group_labels}
    group_to_idx = {lbl: i for i, lbl in enumerate(group_labels)}

    # Serialize each slide as base64 binary data
    slides_meta = []
    slides_b64_positions = []
    slides_b64_groups = []

    for name, data in slides_data:
        # Interleave all positions into one flat array: [x0,y0,x1,y1,...]
        all_x = []
        all_y = []
        all_gi = []
        for g in data['groups']:
            gi = group_to_idx.get(g['label'], group_to_idx.get('other', 0))
            all_x.append(g['x'])
            all_y.append(g['y'])
            all_gi.append(np.full(g['n'], gi, dtype=np.uint8))

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        all_gi = np.concatenate(all_gi)
        n = len(all_x)

        # Interleave x,y into flat float32 array
        positions = np.empty(n * 2, dtype=np.float32)
        positions[0::2] = all_x
        positions[1::2] = all_y

        slides_b64_positions.append(encode_float32_base64(positions))
        slides_b64_groups.append(encode_uint8_base64(all_gi))

        slides_meta.append({
            'name': name,
            'n': int(n),
            'xr': [float(data['x_range'][0]), float(data['x_range'][1])],
            'yr': [float(data['y_range'][0]), float(data['y_range'][1])],
        })

    # Build legend info
    legend_items = []
    total_counts = {}
    for _, data in slides_data:
        for g in data['groups']:
            total_counts[g['label']] = total_counts.get(g['label'], 0) + g['n']

    for lbl in group_labels:
        legend_items.append({
            'label': lbl,
            'color': color_map[lbl],
            'count': total_counts.get(lbl, 0),
        })

    n_slides = len(slides_data)
    is_single = n_slides == 1
    timestamp = datetime.now().isoformat(timespec='seconds')

    # --- Build the HTML ---
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
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
  .roi-item .roi-stats {{ color: #888; font-size: 10px; white-space: nowrap; }}
  .roi-del {{ cursor: pointer; color: #a55; font-size: 14px; line-height: 1; }}
  .roi-del:hover {{ color: #f66; }}

  /* Help text */
  .help-text {{ font-size: 10px; color: #555; line-height: 1.4; }}
</style>
</head>
<body>
<div id="app">
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

    <!-- ROI Drawing -->
    <div class="sidebar-section">
      <h3>ROI Drawing</h3>
      <div class="btn-row">
        <button class="btn mode-btn active" id="mode-pan" data-mode="pan">Pan</button>
        <button class="btn mode-btn" id="mode-circle" data-mode="circle">Circle</button>
        <button class="btn mode-btn" id="mode-rect" data-mode="rect">Rect</button>
        <button class="btn mode-btn" id="mode-poly" data-mode="polygon">Poly</button>
      </div>
      <div style="font-size:10px;color:#555;margin:4px 0;">
        Circle: click+drag &middot; Rect: click+drag<br>
        Polygon: click vertices, dbl-click to close
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
""")

    # Emit base64 data arrays
    html_parts.append("const SLIDE_POS_B64 = [\n")
    for i, b64 in enumerate(slides_b64_positions):
        comma = ',' if i < len(slides_b64_positions) - 1 else ''
        html_parts.append(f'  "{b64}"{comma}\n')
    html_parts.append("];\n")

    html_parts.append("const SLIDE_GRP_B64 = [\n")
    for i, b64 in enumerate(slides_b64_groups):
        comma = ',' if i < len(slides_b64_groups) - 1 else ''
        html_parts.append(f'  "{b64}"{comma}\n')
    html_parts.append("];\n")

    html_parts.append("""
// Decode binary data into per-slide arrays
const SLIDES = SLIDE_META.map((meta, i) => {
  const pos = b64toF32(SLIDE_POS_B64[i]);
  const grp = b64toU8(SLIDE_GRP_B64[i]);
  return {
    name: meta.name,
    n: meta.n,
    xr: meta.xr,
    yr: meta.yr,
    pos: pos,  // interleaved [x0,y0,x1,y1,...] Float32Array
    grp: grp,  // group index per cell Uint8Array
  };
});

// Free the base64 strings to reduce memory
SLIDE_POS_B64.length = 0;
SLIDE_GRP_B64.length = 0;

// ===================================================================
// State
// ===================================================================
const hidden = new Set();
let dotSize = 3, dotAlpha = 0.7;
let drawMode = 'pan';  // pan | circle | rect | polygon

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

function pointInROI(px, py, roi) {
  if (roi.type === 'circle') {
    return pointInCircle(px, py, roi.data.cx, roi.data.cy, roi.data.r);
  } else if (roi.type === 'rect') {
    return pointInRect(px, py, roi.data.x1, roi.data.y1, roi.data.x2, roi.data.y2);
  } else if (roi.type === 'polygon') {
    return pointInPolygon(px, py, roi.data.verts);
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

      if (drawMode === 'polygon') {
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

    drawCanvas.addEventListener('mousemove', e => {
      if (drawMode === 'pan') return;
      if (drawStart && drawStart.panel === state) {
        const rect = div.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const [dx, dy] = screenToData(state, sx, sy);
        drawCurrent = { x: dx, y: dy };

        // Show measurement while dragging
        if (drawMode === 'circle') {
          const ddx = dx - drawStart.x, ddy = dy - drawStart.y;
          const r = Math.sqrt(ddx * ddx + ddy * ddy);
          measureEl.style.display = 'block';
          measureEl.textContent = 'r = ' + r.toFixed(0) + ' \\u00b5m';
        } else if (drawMode === 'rect') {
          const w = Math.abs(dx - drawStart.x);
          const h = Math.abs(dy - drawStart.y);
          measureEl.style.display = 'block';
          measureEl.textContent = w.toFixed(0) + ' \\u00d7 ' + h.toFixed(0) + ' \\u00b5m';
        }
        renderDrawOverlay(state);
      }
    });

    drawCanvas.addEventListener('mouseup', e => {
      if (drawMode === 'pan' || drawMode === 'polygon') return;
      if (!drawStart || drawStart.panel !== state) return;
      const rect = div.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData(state, sx, sy);

      if (drawMode === 'circle') {
        const cdx = dx - drawStart.x, cdy = dy - drawStart.y;
        const r = Math.sqrt(cdx * cdx + cdy * cdy);
        if (r > 1) {
          addROI(idx, 'circle', { cx: drawStart.x, cy: drawStart.y, r });
        }
      } else if (drawMode === 'rect') {
        const w = Math.abs(dx - drawStart.x), h = Math.abs(dy - drawStart.y);
        if (w > 1 && h > 1) {
          addROI(idx, 'rect', { x1: drawStart.x, y1: drawStart.y, x2: dx, y2: dy });
        }
      }
      drawStart = null;
      drawCurrent = null;
      measureEl.style.display = 'none';
      renderDrawOverlay(state);
    });

    drawCanvas.addEventListener('dblclick', e => {
      if (drawMode !== 'polygon') return;
      if (polySlideIdx === idx && polyVerts.length >= 3) {
        addROI(idx, 'polygon', { verts: polyVerts.slice() });
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
    if (!activePanel) return;
    activePanel.panX = activePanel.panStartX + (e.clientX - activePanel.dragStartX);
    activePanel.panY = activePanel.panStartY + (e.clientY - activePanel.dragStartY);
    scheduleRender(activePanel);
  });
  window.addEventListener('mouseup', () => {
    if (activePanel) {
      activePanel.div.classList.remove('dragging');
      activePanel = null;
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

  const r = dotSize / p.zoom;
  const halfR = r / 2;
  const slide = p.slide;
  const pos = slide.pos;
  const grp = slide.grp;
  const n = slide.n;
  let total = 0;

  // For performance with >50k cells, batch by group
  const useROIFilter = roiFilterActive && rois.length > 0;

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
    } else {
      labelX = roi.data.verts[0][0];
      labelY = roi.data.verts[0][1] - fontSize * 1.3;
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

  // Draw in-progress polygon
  if (drawMode === 'polygon' && polySlideIdx === p.idx && polyVerts.length > 0) {
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
  };
  rois.push(roi);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
}

function deleteROI(id) {
  const idx = rois.findIndex(r => r.id === id);
  if (idx >= 0) rois.splice(idx, 1);
  updateROIList();
  updateROIStats();
  panels.forEach(p => renderDrawOverlay(p));
  if (roiFilterActive) scheduleRenderAll();
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

    const statsSpan = document.createElement('span');
    statsSpan.className = 'roi-stats';
    statsSpan.dataset.roiId = roi.id;

    const delBtn = document.createElement('span');
    delBtn.className = 'roi-del';
    delBtn.textContent = '\\u00d7';
    delBtn.onclick = () => deleteROI(roi.id);

    item.appendChild(nameSpan);
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
    if (roi.type === 'circle') {
      entry.center_um = [roi.data.cx, roi.data.cy];
      entry.radius_um = roi.data.r;
    } else if (roi.type === 'rect') {
      entry.min_um = [Math.min(roi.data.x1, roi.data.x2), Math.min(roi.data.y1, roi.data.y2)];
      entry.max_um = [Math.max(roi.data.x1, roi.data.x2), Math.max(roi.data.y1, roi.data.y2)];
    } else if (roi.type === 'polygon') {
      entry.vertices_um = roi.data.verts;
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
    scheduleRenderAll();
  };
  document.getElementById('btn-hide-all').onclick = () => {
    GROUP_LABELS.forEach(l => hidden.add(l));
    document.querySelectorAll('.leg-item').forEach(el => el.classList.add('hidden'));
    scheduleRenderAll();
  };

  // Reset zoom
  document.getElementById('btn-reset-zoom').onclick = () => {
    resizePanels();
    panels.forEach(fitPanel);
    scheduleRenderAll();
    panels.forEach(p => renderDrawOverlay(p));
  };

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
      if (drawMode !== 'polygon') {
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
""")

    html_parts.append("</script>\n</body>\n</html>")

    html_content = ''.join(html_parts)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({file_size_mb:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-slide spatial viewer HTML from classified detections')
    parser.add_argument('--input-dir',
                        help='Directory containing per-slide subdirectories '
                             '(or a single slide dir with classified JSON)')
    parser.add_argument('--detections', nargs='+',
                        help='Explicit list of detection JSON files')
    parser.add_argument('--detection-glob', default='cell_detections_classified.json',
                        help='Glob pattern for detection files within slide subdirs '
                             '(default: cell_detections_classified.json)')
    parser.add_argument('--group-field', required=True,
                        help='Field in features dict to color cells by '
                             '(e.g. tdTomato_class, MSLN_class, marker_profile)')
    parser.add_argument('--title', default='Multi-Slide Spatial Overview',
                        help='HTML page title')
    parser.add_argument('--output', default=None,
                        help='Output HTML path (default: {input-dir}/spatial_viewer.html)')
    args = parser.parse_args()

    if not args.input_dir and not args.detections:
        parser.error('Provide either --input-dir or --detections')

    # Determine output path
    if args.output is None:
        if args.input_dir:
            args.output = str(Path(args.input_dir) / 'spatial_viewer.html')
        else:
            args.output = 'spatial_viewer.html'

    # Discover or use explicit files
    if args.input_dir:
        slide_files = discover_slides(args.input_dir, args.detection_glob)
        if not slide_files:
            print(f"Error: no detection files matching '{args.detection_glob}' "
                  f"found in {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(slide_files)} slides in {args.input_dir}")
    else:
        slide_files = []
        for p in args.detections:
            path = Path(p)
            name = path.parent.name if path.parent.name else path.stem
            slide_files.append((name, path))

    # Load data
    slides_data = []
    for name, path in slide_files:
        print(f"  Loading {name}...", end='', flush=True)
        data = load_slide_data(path, args.group_field)
        if data is None:
            print(" skipped (no data)")
            continue
        groups_str = ', '.join(f"{g['label']}:{g['n']}" for g in data['groups'])
        slides_data.append((name, data))
        print(f" {data['n_cells']} cells [{groups_str}]")

    if not slides_data:
        print("Error: no valid slide data loaded", file=sys.stderr)
        sys.exit(1)

    # Assign colors
    color_map = assign_group_colors(slides_data)
    print(f"\nGroups: {', '.join(f'{k} ({v})' for k, v in sorted(color_map.items()))}")

    # Generate HTML
    total_cells = sum(d['n_cells'] for _, d in slides_data)
    print(f"\nGenerating HTML for {len(slides_data)} slides, "
          f"{total_cells:,} total cells...")
    generate_html(slides_data, args.output, color_map,
                  title=args.title, group_field=args.group_field)


if __name__ == '__main__':
    main()
