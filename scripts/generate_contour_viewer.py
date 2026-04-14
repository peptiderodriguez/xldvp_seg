#!/usr/bin/env python
"""Generate a self-contained HTML viewer for polygon contour overlays on CZI fluorescence.

Displays contours from one or more JSON files (vessel lumens, cell detections, etc.)
overlaid on a fluorescence CZI background.  Contours are grouped by a configurable
field and color-coded with an interactive legend.

Usage::

    python scripts/generate_contour_viewer.py \
        --contours vessel_lumens.json \
        --group-field vessel_type \
        --czi-path slide.czi \
        --display-channels 1,3,0 \
        --channel-names "SMA,CD31,nuc" \
        --title "Vessel Lumen Detection" \
        --output vessel_viewer.html

Reuses components from ``xldvp_seg.visualization`` (fluorescence loading, JS modules,
encoding, color palettes) so this script stays compact (~500 lines).
"""

from __future__ import annotations

import argparse
import html as html_mod
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from xldvp_seg.utils.detection_utils import get_contour_px, get_contour_um
from xldvp_seg.utils.json_utils import fast_json_load
from xldvp_seg.utils.logging import get_logger
from xldvp_seg.visualization.colors import AUTO_COLORS, hsl_palette
from xldvp_seg.visualization.data_loading import extract_group
from xldvp_seg.visualization.encoding import safe_json
from xldvp_seg.visualization.fluorescence import (
    encode_channel_b64,
    read_czi_thumbnail_channels,
)
from xldvp_seg.visualization.js_loader import load_js

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

_CONTOUR_FIELDS = [
    "contour_um",
    "contour_dilated_um",
    "contour_global_um",
    "contour_px",
    "contour_dilated_px",
    "outer_contour_global",
]

_METADATA_EXCLUDE = frozenset(_CONTOUR_FIELDS) | {"features", "mask_rle", "sam2_embedding"}


def _extract_contour(det: dict, pixel_size_um: float | None):
    """Return contour as Nx2 float array in microns, or None.

    Tries um-coordinate fields first (identity scaling), then pixel fields
    (scaled by *pixel_size_um* or per-detection ``pixel_size_um``).
    """
    # 1) Try um-coordinate helpers
    pts = get_contour_um(det)
    if pts is not None and len(pts) >= 3:
        return np.asarray(pts, dtype=np.float64)[:, :2]

    # 2) Try contour_global_um (vessel lumens)
    pts = det.get("contour_global_um")
    if pts is not None and len(pts) >= 3:
        return np.asarray(pts, dtype=np.float64)[:, :2]

    # Resolve pixel size: CLI arg > per-detection field
    ps = pixel_size_um
    if ps is None:
        ps = det.get("pixel_size_um") or det.get("features", {}).get("pixel_size_um")
        if ps is not None:
            ps = float(ps)

    # 3) Pixel-coordinate helpers (need pixel_size)
    pts = get_contour_px(det)
    if pts is not None and len(pts) >= 3 and ps:
        return np.asarray(pts, dtype=np.float64)[:, :2] * ps

    # 4) outer_contour_global (pixel coords)
    pts = det.get("outer_contour_global")
    if pts is not None and len(pts) >= 3 and ps:
        return np.asarray(pts, dtype=np.float64)[:, :2] * ps

    return None


def _build_metadata(det: dict, exclude_keys: set | None = None) -> dict:
    """Build a compact metadata dict for click-to-inspect display.

    Includes scalar top-level fields and selected feature keys.  Excludes
    large list/array fields (contours, masks) to keep HTML size small.
    """
    if exclude_keys is None:
        exclude_keys = _METADATA_EXCLUDE
    meta = {}
    for k, v in det.items():
        if k in exclude_keys:
            continue
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 3:
            meta[k] = v
    # Pull a few useful feature values
    feats = det.get("features", {})
    for fk in (
        "area_um2",
        "pixel_size_um",
        "global_center_um",
        "marker_profile",
        "vessel_type",
        "morphology",
        "equiv_diameter_um",
        "perimeter_um",
        "circularity",
        "elongation",
    ):
        if fk in feats:
            val = feats[fk]
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                continue
            meta[f"feat.{fk}"] = val
    return meta


# ---------------------------------------------------------------------------
# Load and process contours from JSON
# ---------------------------------------------------------------------------


def load_contours(
    json_paths: list[Path],
    group_field: str,
    pixel_size_um: float | None,
    score_threshold: float | None,
    max_contours: int,
    max_area_um2: float | None = None,
) -> tuple[list[dict], list[str]]:
    """Load contours from JSON files, extract groups, build JS-ready dicts.

    Returns:
        (contour_records, group_labels) where each record is
        ``{pts: [...], bx1, by1, bx2, by2, gi: int, meta: {...}}``.
    """
    raw_dets = []
    for jp in json_paths:
        logger.info("Loading %s ...", jp)
        data = fast_json_load(jp)
        if isinstance(data, dict):
            # Some JSON files wrap detections under a key
            for key in ("detections", "vessels", "lumens", "cells", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # Single-object file — wrap it
                data = [data]
        if not isinstance(data, list):
            logger.warning("Skipping %s — not a list or recognized dict format", jp)
            continue
        raw_dets.extend(data)
    logger.info("Loaded %d raw records from %d file(s)", len(raw_dets), len(json_paths))

    # Optional score filter
    if score_threshold is not None:
        before = len(raw_dets)
        raw_dets = [d for d in raw_dets if d.get("score", 1.0) >= score_threshold]
        logger.info("Score filter (>= %.2f): %d -> %d", score_threshold, before, len(raw_dets))

    # Area filter (remove giant false positives)
    if max_area_um2 is not None:
        before = len(raw_dets)
        raw_dets = [d for d in raw_dets if d.get("area_um2", 0) <= max_area_um2]
        if len(raw_dets) < before:
            logger.info("Area filter (<= %.0f um^2): %d -> %d", max_area_um2, before, len(raw_dets))

    # Subsample if needed
    if len(raw_dets) > max_contours:
        step = max(1, len(raw_dets) // max_contours)
        raw_dets = raw_dets[::step]
        logger.info("Subsampled to %d contours (max_contours=%d)", len(raw_dets), max_contours)

    # Extract contours + groups
    group_set: set[str] = set()
    records: list[tuple[np.ndarray, str, dict]] = []
    skipped = 0

    for i, det in enumerate(raw_dets):
        raw_dets[i] = None  # free memory as we go
        pts = _extract_contour(det, pixel_size_um)
        if pts is None or len(pts) < 3:
            skipped += 1
            continue
        grp = extract_group(det, group_field)
        meta = _build_metadata(det)
        group_set.add(grp)
        records.append((pts, grp, meta))
    del raw_dets

    if skipped:
        logger.info("Skipped %d records with no valid contour", skipped)

    # Stable sort order for groups
    group_labels = sorted(group_set)
    group_to_idx = {lbl: i for i, lbl in enumerate(group_labels)}

    # Build JS-ready dicts
    contour_data = []
    for pts, grp, meta in records:
        flat = pts.ravel().astype(np.float32).tolist()
        bx1 = float(pts[:, 0].min())
        bx2 = float(pts[:, 0].max())
        by1 = float(pts[:, 1].min())
        by2 = float(pts[:, 1].max())
        # Pass through uid if present in metadata; fallback to centroid-based
        uid = meta.pop("uid", None)
        if uid is None:
            cx = round((bx1 + bx2) / 2, 2)
            cy = round((by1 + by2) / 2, 2)
            uid = f"lumen_{cx}_{cy}"
        contour_data.append(
            {
                "pts": flat,
                "bx1": round(bx1, 1),
                "by1": round(by1, 1),
                "bx2": round(bx2, 1),
                "by2": round(by2, 1),
                "gi": group_to_idx[grp],
                "uid": uid,
                "meta": meta,
            }
        )

    logger.info(
        "Prepared %d contours in %d groups: %s",
        len(contour_data),
        len(group_labels),
        ", ".join(group_labels),
    )
    return contour_data, group_labels


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------


def _assign_colors(group_labels: list[str]) -> list[str]:
    """Assign maximally-distinct hex colors to groups."""
    n = len(group_labels)
    if n <= 20:
        return AUTO_COLORS[:n]
    return hsl_palette(n)


# ---------------------------------------------------------------------------
# Fluorescence data
# ---------------------------------------------------------------------------


def _load_fluorescence(
    czi_path: Path,
    display_channels: list[int],
    channel_names: list[str] | None,
    scale_factor: float,
    scene: int,
) -> dict | None:
    """Load CZI fluorescence thumbnails, return dict for HTML embedding."""
    try:
        ch_arrays, pixel_size_um, mosaic_x, mosaic_y = read_czi_thumbnail_channels(
            czi_path, display_channels, scale_factor=scale_factor, scene=scene
        )
    except Exception as exc:
        logger.warning("Failed to read CZI fluorescence: %s", exc)
        return None

    if pixel_size_um is None:
        raise ValueError(
            "Could not determine pixel size from CZI metadata. " "Pass --pixel-size-um explicitly."
        )

    # Encode each channel as PNG base64
    ch_b64 = []
    height, width = 0, 0
    for arr in ch_arrays:
        if arr is not None:
            ch_b64.append(encode_channel_b64(arr))
            height, width = arr.shape[:2]
        else:
            ch_b64.append("")

    # Auto-detect channel names from filename if not provided
    if channel_names is None:
        channel_names = _auto_channel_names(czi_path, display_channels)

    return {
        "channels": ch_b64,
        "width": width,
        "height": height,
        "scale": scale_factor,
        "mosaic_x": mosaic_x,
        "mosaic_y": mosaic_y,
        "pixel_size": pixel_size_um,
        "names": channel_names,
    }


def _auto_channel_names(czi_path: Path, display_channels: list[int]) -> list[str]:
    """Try to resolve channel names from CZI filename markers."""
    try:
        from xldvp_seg.io.czi_loader import CZILoader, parse_markers_from_filename

        markers = parse_markers_from_filename(czi_path.name)
        if not markers:
            return [f"Ch{c}" for c in display_channels]

        # Build wavelength -> marker name lookup
        wl_to_name = {}
        for m in markers:
            wl = m.get("wavelength")
            if wl and wl not in wl_to_name:
                wl_to_name[wl] = m["name"]

        # Get CZI channel wavelengths
        loader = CZILoader(str(czi_path))
        ch_info = loader.get_channel_info()
        del loader  # close file handle

        names = []
        for c in display_channels:
            if c < len(ch_info):
                em_wl = ch_info[c].get("emission_wavelength_nm")
                name = wl_to_name.get(em_wl, f"Ch{c}") if em_wl else f"Ch{c}"
                names.append(name)
            else:
                names.append(f"Ch{c}")
        return names
    except Exception:
        return [f"Ch{c}" for c in display_channels]


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(
    contour_data: list[dict],
    group_labels: list[str],
    group_colors: list[str],
    fluor_data: dict | None,
    title: str,
    group_field: str,
    output_path: Path,
):
    """Write self-contained HTML viewer to *output_path*."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    title_escaped = html_mod.escape(title)

    # Compute data extent from contour bounding boxes (for fit-to-view)
    if contour_data:
        xmin = min(c["bx1"] for c in contour_data)
        xmax = max(c["bx2"] for c in contour_data)
        ymin = min(c["by1"] for c in contour_data)
        ymax = max(c["by2"] for c in contour_data)
    else:
        xmin, xmax, ymin, ymax = 0, 1000, 0, 1000

    # Shared JS from package
    shared_js = load_js("coordinate")

    # Group counts for legend
    group_counts = [0] * len(group_labels)
    for c in contour_data:
        group_counts[c["gi"]] += 1

    parts = []
    parts.append(
        f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_escaped}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #111; color: #eee; font-family: -apple-system, BlinkMacSystemFont,
  'Segoe UI', Roboto, sans-serif; display: flex; height: 100vh; overflow: hidden; }}

#sidebar {{
  width: 280px; min-width: 280px; background: #1a1a1a; border-right: 1px solid #333;
  display: flex; flex-direction: column; overflow: hidden;
}}
#sidebar h2 {{ padding: 12px 14px 8px; font-size: 14px; color: #aaa;
  border-bottom: 1px solid #333; }}
#legend {{ flex: 1; overflow-y: auto; padding: 6px 10px; }}
.leg-item {{
  display: flex; align-items: center; gap: 8px; padding: 4px 6px;
  cursor: pointer; border-radius: 4px; font-size: 13px;
  transition: opacity 0.15s;
}}
.leg-item:hover {{ background: #2a2a2a; }}
.leg-item.hidden {{ opacity: 0.3; }}
.leg-dot {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }}
.leg-label {{ flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.leg-count {{ color: #888; font-size: 11px; }}

#controls {{
  padding: 10px 12px; border-top: 1px solid #333;
  display: flex; flex-direction: column; gap: 8px;
}}
#controls label {{ font-size: 12px; color: #aaa; }}
#controls input[type=range] {{ width: 100%; }}
.ch-btns {{ display: flex; gap: 4px; }}
.ch-btn {{
  padding: 3px 8px; border-radius: 4px; border: 1px solid #555;
  background: #333; color: #eee; font-size: 12px; cursor: pointer;
}}
.ch-btn.active {{ border-color: #8af; background: #234; }}

#meta-panel {{
  max-height: 200px; overflow-y: auto; padding: 8px 12px;
  border-top: 1px solid #333; font-size: 12px; display: none;
}}
#meta-panel h3 {{ font-size: 12px; color: #aaa; margin-bottom: 4px; }}
#meta-panel table {{ width: 100%; }}
#meta-panel td {{ padding: 1px 4px; vertical-align: top; }}
#meta-panel td:first-child {{ color: #888; white-space: nowrap; }}
#meta-panel td:last-child {{ word-break: break-all; }}

#anno-buttons {{ display: flex; gap: 4px; margin-top: 6px; flex-wrap: wrap; }}
.anno-btn {{
  padding: 3px 8px; border-radius: 4px; border: 1px solid #555;
  background: #333; color: #eee; font-size: 11px; cursor: pointer;
  transition: background 0.15s;
}}
.anno-btn:hover {{ background: #444; }}
.anno-btn.active-true {{ background: #264d26; border-color: #4a4; }}
.anno-btn.active-false {{ background: #4d2626; border-color: #a44; }}
.anno-btn.active-type {{ background: #263d4d; border-color: #48a; }}

#anno-toolbar {{
  padding: 6px 12px; border-top: 1px solid #333; font-size: 11px;
  display: flex; flex-direction: column; gap: 4px;
}}
#anno-toolbar .anno-row {{ display: flex; align-items: center; gap: 6px; }}
#anno-toolbar .anno-count {{ color: #aaa; }}
.anno-mode-btn {{
  padding: 2px 6px; border-radius: 3px; border: 1px solid #555;
  background: #333; color: #eee; font-size: 10px; cursor: pointer;
}}
.anno-mode-btn.active {{ border-color: #8af; background: #234; }}
.anno-export-btn {{
  padding: 2px 8px; border-radius: 3px; border: 1px solid #555;
  background: #2a4a2a; color: #cfc; font-size: 10px; cursor: pointer;
}}
.anno-import-btn {{
  padding: 2px 8px; border-radius: 3px; border: 1px solid #555;
  background: #2a2a4a; color: #ccf; font-size: 10px; cursor: pointer;
}}

#canvas-wrap {{
  flex: 1; position: relative; overflow: hidden; background: #000;
}}
canvas {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  image-rendering: pixelated;
}}
#info-bar {{
  position: absolute; bottom: 8px; left: 8px;
  background: rgba(0,0,0,0.7); padding: 4px 10px; border-radius: 4px;
  font-size: 11px; color: #aaa; pointer-events: none;
}}
#floating-fit {{
  position: absolute; top: 10px; right: 10px;
  background: rgba(0,0,0,0.7); padding: 5px 12px; border-radius: 4px;
  border: 1px solid #555; color: #ccc; font-size: 12px; cursor: pointer;
}}
</style>
</head>
<body>

<div id="sidebar">
  <h2>{title_escaped} &mdash; {len(contour_data):,} contours</h2>
  <div id="legend"></div>
  <div id="meta-panel"><h3>Click a contour to inspect</h3>
    <table id="meta-table"></table>
    <div id="anno-buttons"></div>
  </div>
  <div id="anno-toolbar">
    <div class="anno-row">
      <span class="anno-count" id="anno-count">0 annotated</span>
      <button class="anno-mode-btn active" id="mode-binary">Binary</button>
      <button class="anno-mode-btn" id="mode-type">Type</button>
    </div>
    <div class="anno-row">
      <button class="anno-export-btn" id="anno-export">Export</button>
      <button class="anno-import-btn" id="anno-import">Import</button>
      <input type="file" id="anno-file" accept=".json" style="display:none">
    </div>
  </div>
  <div id="controls">
    <label>Contour opacity
      <input type="range" id="opacity-slider" min="0" max="100" value="70">
    </label>
    <label>Stroke width
      <input type="range" id="stroke-slider" min="1" max="10" value="3">
    </label>
    <label>Fill opacity
      <input type="range" id="fill-slider" min="0" max="100" value="15">
    </label>
    <div class="ch-btns">
      <button class="ch-btn" id="btn-fit">Fit</button>
      <button class="ch-btn" id="btn-reset">Reset zoom</button>
    </div>"""
    )

    # Fluorescence channel toggle buttons
    if fluor_data:
        ch_names = fluor_data.get("names", ["Ch0", "Ch1", "Ch2"])
        tint_labels = ["R", "G", "B"]
        parts.append('    <div class="ch-btns">')
        for ci in range(min(3, len(fluor_data["channels"]))):
            name = ch_names[ci] if ci < len(ch_names) else f"Ch{ci}"
            parts.append(
                f'      <button class="ch-btn active" data-ch="{ci}">'
                f"{tint_labels[ci]}: {html_mod.escape(name)}</button>"
            )
        parts.append("    </div>")
        parts.append(
            "    <label>Background brightness"
            '\n      <input type="range" id="fluor-slider" min="0" max="100" value="80">'
            "\n    </label>"
        )

    parts.append(
        f"""\
  </div>
</div>

<div id="canvas-wrap">
  <canvas id="canvas"></canvas>
  <button id="floating-fit">Fit view</button>
  <div id="info-bar">Scroll to zoom, drag to pan, click contour to inspect.
    Generated {timestamp}</div>
</div>

<script>
// === Shared JS components ===
{shared_js}

// === Data ===
const GROUP_LABELS = {safe_json(group_labels)};
const GROUP_COLORS = {safe_json(group_colors)};
const GROUP_COUNTS = {safe_json(group_counts)};
const N_GROUPS = GROUP_LABELS.length;
const TITLE = {safe_json(title)};
const GROUP_FIELD = {safe_json(group_field)};
const DATA_EXTENT = {safe_json([xmin, ymin, xmax, ymax])};
"""
    )

    # Emit contour data as JSON (pts arrays stay as plain arrays for simplicity;
    # browser converts to typed arrays on init)
    parts.append(f"const CONTOUR_DATA = {safe_json(contour_data)};\n")

    # Fluorescence data
    if fluor_data:
        meta = {
            "w": fluor_data["width"],
            "h": fluor_data["height"],
            "scale": fluor_data["scale"],
            "mx": fluor_data.get("mosaic_x", 0),
            "my": fluor_data.get("mosaic_y", 0),
            "pixel_size": fluor_data.get("pixel_size", 0.22),
        }
        parts.append(f"const FLUOR_META = {safe_json(meta)};\n")
        parts.append("const FLUOR_CH_B64 = [\n")
        for ci, b64 in enumerate(fluor_data["channels"]):
            comma = "," if ci < len(fluor_data["channels"]) - 1 else ""
            parts.append(f'  "{b64}"{comma}\n')
        parts.append("];\n")
    else:
        parts.append("const FLUOR_META = null;\n")
        parts.append("const FLUOR_CH_B64 = [];\n")

    # Viewer JS
    parts.append(
        """\
const CH_TINTS = [[255,80,80],[80,255,80],[80,80,255]];
let chEnabled = [true, true, true];
let fluorAlpha = 0.8;
let fluorCanvas = null;  // offscreen composite
let fluorDirty = true;
let fluorImages = [null, null, null];  // decoded Image objects per channel

// Panel state
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const wrap = document.getElementById('canvas-wrap');
let cw = 0, ch = 0, zoom = 1, panX = 0, panY = 0;
let contourAlpha = 0.7, strokeWidth = 3, fillAlpha = 0.15;
const hidden = new Set();  // hidden group labels

// ----- Resize + fit -----
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect();
  cw = Math.floor(rect.width);
  ch = Math.floor(rect.height);
  canvas.width = cw * dpr;
  canvas.height = ch * dpr;
  canvas.style.width = cw + 'px';
  canvas.style.height = ch + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function fitView() {
  const [xmin, ymin, xmax, ymax] = DATA_EXTENT;
  const dataW = xmax - xmin;
  const dataH = ymax - ymin;
  if (dataW <= 0 || dataH <= 0) { zoom = 1; panX = cw/2; panY = ch/2; return; }
  const pad = 0.05;
  zoom = Math.min(cw / (dataW * (1 + 2*pad)), ch / (dataH * (1 + 2*pad)));
  panX = (cw - dataW * zoom) / 2 - xmin * zoom;
  panY = (ch - dataH * zoom) / 2 - ymin * zoom;
}

// ----- Fluorescence compositing -----
function rebuildFluorComposite() {
  if (!FLUOR_META) return;
  const iw = FLUOR_META.w, ih = FLUOR_META.h;
  if (!fluorCanvas) {
    fluorCanvas = document.createElement('canvas');
    fluorCanvas.width = iw;
    fluorCanvas.height = ih;
  }
  const fctx = fluorCanvas.getContext('2d', { willReadFrequently: true });
  const result = new Uint8ClampedArray(iw * ih * 4);
  for (let ci = 0; ci < 3; ci++) {
    if (!chEnabled[ci] || !fluorImages[ci]) continue;
    const tmp = document.createElement('canvas');
    tmp.width = iw; tmp.height = ih;
    const tctx = tmp.getContext('2d');
    tctx.drawImage(fluorImages[ci], 0, 0);
    const px = tctx.getImageData(0, 0, iw, ih).data;
    const [tr, tg, tb] = CH_TINTS[ci];
    for (let i = 0; i < iw * ih; i++) {
      const v = px[i*4] / 255;
      result[i*4]   = Math.min(255, result[i*4]   + tr * v);
      result[i*4+1] = Math.min(255, result[i*4+1] + tg * v);
      result[i*4+2] = Math.min(255, result[i*4+2] + tb * v);
      result[i*4+3] = 255;
    }
  }
  fctx.putImageData(new ImageData(result, iw, ih), 0, 0);
  fluorDirty = false;
}

function drawFluorescence() {
  if (!FLUOR_META || !fluorCanvas) return;
  const m = FLUOR_META;
  const mx_um = m.mx * m.pixel_size;
  const my_um = m.my * m.pixel_size;
  const scale_inv = 1.0 / m.scale;
  const dw = m.w * scale_inv * m.pixel_size;
  const dh = m.h * scale_inv * m.pixel_size;
  ctx.globalAlpha = fluorAlpha;
  ctx.drawImage(fluorCanvas, mx_um, my_um, dw, dh);
  ctx.globalAlpha = 1;
}

// ----- Contour rendering -----
function render() {
  ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);

  // Fluorescence background
  if (fluorDirty) rebuildFluorComposite();
  drawFluorescence();

  // Viewport culling bounds (data coords)
  const vx1 = -panX / zoom;
  const vy1 = -panY / zoom;
  const vx2 = (cw - panX) / zoom;
  const vy2 = (ch - panY) / zoom;

  const lw = strokeWidth / zoom;

  for (let i = 0; i < CONTOUR_DATA.length; i++) {
    const c = CONTOUR_DATA[i];
    // Skip hidden groups
    if (hidden.has(GROUP_LABELS[c.gi])) continue;
    // Viewport culling
    if (c.bx2 < vx1 || c.bx1 > vx2 || c.by2 < vy1 || c.by1 > vy2) continue;

    const pts = c.pts;
    if (!pts || pts.length < 6) continue;

    const color = GROUP_COLORS[c.gi];
    const path = new Path2D();
    path.moveTo(pts[0], pts[1]);
    for (let j = 2; j < pts.length; j += 2) {
      path.lineTo(pts[j], pts[j+1]);
    }
    path.closePath();

    // Fill
    if (fillAlpha > 0.005) {
      ctx.globalAlpha = fillAlpha;
      ctx.fillStyle = color;
      ctx.fill(path);
    }

    // Stroke
    ctx.globalAlpha = contourAlpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.stroke(path);
  }

  ctx.globalAlpha = 1;
  ctx.restore();
}

// ----- Point-in-polygon hit test -----
function pointInPolygon(px, py, pts) {
  let inside = false;
  const n = pts.length;
  for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
    const xi = pts[i], yi = pts[i+1];
    const xj = pts[j], yj = pts[j+1];
    if (((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function findContourAt(dx, dy) {
  // Search backwards so topmost-drawn contour is found first
  for (let i = CONTOUR_DATA.length - 1; i >= 0; i--) {
    const c = CONTOUR_DATA[i];
    if (hidden.has(GROUP_LABELS[c.gi])) continue;
    if (dx < c.bx1 || dx > c.bx2 || dy < c.by1 || dy > c.by2) continue;
    if (pointInPolygon(dx, dy, c.pts)) return c;
  }
  return null;
}

// ----- Metadata panel -----
const metaPanel = document.getElementById('meta-panel');
const metaTable = document.getElementById('meta-table');

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showMeta(c) {
  metaPanel.style.display = 'block';
  const grp = GROUP_LABELS[c.gi];
  let html = '<tr><td>group</td><td>' + escHtml(grp) + '</td></tr>';
  if (c.meta) {
    for (const [k, v] of Object.entries(c.meta)) {
      let vs = (typeof v === 'number') ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v);
      if (vs.length > 60) vs = vs.slice(0, 57) + '...';
      html += '<tr><td>' + escHtml(k) + '</td><td>' + escHtml(vs) + '</td></tr>';
    }
  }
  metaTable.innerHTML = html;
}

function hideMeta() {
  metaPanel.style.display = 'none';
  metaTable.innerHTML = '';
}

// ----- RAF batching -----
let rafPending = false;
function scheduleRender() {
  if (!rafPending) { rafPending = true; requestAnimationFrame(() => { rafPending = false; render(); }); }
}

// ----- Pan / zoom handlers -----
let dragging = false, dragStartX = 0, dragStartY = 0, panStartX = 0, panStartY = 0;

canvas.addEventListener('wheel', function(e) {
  e.preventDefault();
  const rect = wrap.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1/1.15;
  panX = mx - factor * (mx - panX);
  panY = my - factor * (my - panY);
  zoom *= factor;
  zoom = Math.max(0.001, Math.min(500, zoom));
  scheduleRender();
}, { passive: false });

canvas.addEventListener('mousedown', function(e) {
  if (e.button === 0) {
    dragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    panStartX = panX;
    panStartY = panY;
    canvas.style.cursor = 'grabbing';
    e.preventDefault();
  }
});

window.addEventListener('mousemove', function(e) {
  if (dragging) {
    panX = panStartX + (e.clientX - dragStartX);
    panY = panStartY + (e.clientY - dragStartY);
    scheduleRender();
  }
});

window.addEventListener('mouseup', function(e) {
  if (dragging) {
    // If drag distance is tiny, treat as click for hit-testing
    const dist = Math.abs(e.clientX - dragStartX) + Math.abs(e.clientY - dragStartY);
    if (dist < 4) {
      const rect = wrap.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = screenToData({ panX, panY, zoom }, sx, sy);
      const hit = findContourAt(dx, dy);
      if (hit) showMeta(hit);
      else hideMeta();
    }
    dragging = false;
    canvas.style.cursor = 'default';
  }
});

// ----- Legend -----
(function buildLegend() {
  const legend = document.getElementById('legend');
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
    count.textContent = GROUP_COUNTS[gi].toLocaleString();

    item.appendChild(dot);
    item.appendChild(label);
    item.appendChild(count);

    item.addEventListener('click', function() {
      const lbl = GROUP_LABELS[gi];
      if (hidden.has(lbl)) {
        hidden.delete(lbl);
        item.classList.remove('hidden');
      } else {
        hidden.add(lbl);
        item.classList.add('hidden');
      }
      render();
    });

    legend.appendChild(item);
  }
})();

// ----- Controls -----
document.getElementById('opacity-slider').addEventListener('input', function() {
  contourAlpha = this.value / 100;
  render();
});
document.getElementById('stroke-slider').addEventListener('input', function() {
  strokeWidth = parseFloat(this.value);
  render();
});
document.getElementById('fill-slider').addEventListener('input', function() {
  fillAlpha = this.value / 100;
  render();
});

// Channel toggle buttons
document.querySelectorAll('.ch-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    const ci = parseInt(this.dataset.ch);
    chEnabled[ci] = !chEnabled[ci];
    this.classList.toggle('active');
    fluorDirty = true;
    render();
  });
});

// Fluorescence brightness slider
const fluorSlider = document.getElementById('fluor-slider');
if (fluorSlider) {
  fluorSlider.addEventListener('input', function() {
    fluorAlpha = this.value / 100;
    render();
  });
}

// ----- Load fluorescence images -----
function loadFluorImages() {
  if (!FLUOR_META) return;
  let loaded = 0;
  const total = FLUOR_CH_B64.filter(b => b.length > 0).length;
  for (let ci = 0; ci < FLUOR_CH_B64.length; ci++) {
    if (!FLUOR_CH_B64[ci]) continue;
    const img = new Image();
    img.onload = function() {
      fluorImages[ci] = img;
      loaded++;
      if (loaded >= total) {
        fluorDirty = true;
        render();
      }
    };
    img.src = 'data:image/png;base64,' + FLUOR_CH_B64[ci];
  }
}

// ----- Init -----
function init() {
  resize();
  fitView();
  loadFluorImages();
  render();
}

window.addEventListener('resize', function() { resize(); render(); });

// ===== ANNOTATION SYSTEM =====
const annotations = {};  // {uid: label}
let selectedContour = null;
let annoMode = 'binary';  // 'binary' or 'type'
const ANNO_COLORS = {
  'true': '#44aa44', 'false_positive': '#aa4444',
  'artery': '#cc4444', 'arteriole': '#cc8844', 'vein': '#4444cc',
  'capillary': '#cc44cc', 'lymphatic': '#44cccc', 'collecting_lymphatic': '#44cc88'
};
const STORAGE_KEY = 'contour_anno_' + TITLE.replace(/[^a-zA-Z0-9]/g, '_');

// Build UID lookup for O(1) access
const uidToIdx = {};
for (let i = 0; i < CONTOUR_DATA.length; i++) {
  if (CONTOUR_DATA[i].uid) uidToIdx[CONTOUR_DATA[i].uid] = i;
}

function updateAnnoCount() {
  const n = Object.keys(annotations).length;
  document.getElementById('anno-count').textContent = n + ' / ' + CONTOUR_DATA.length + ' annotated';
}

function setAnnotation(uid, label) {
  if (!uid) return;
  if (label === null || label === undefined) {
    delete annotations[uid];
  } else {
    annotations[uid] = label;
  }
  saveAnnotations();
  updateAnnoCount();
  scheduleRender();
  updateAnnoButtons();
}

function saveAnnotations() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
  } catch(e) {
    console.warn('localStorage save failed:', e);
    const el = document.getElementById('anno-count');
    el.style.color = '#f44';
    el.title = 'localStorage full - export your annotations!';
  }
}

function loadAnnotations() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      for (const [k, v] of Object.entries(parsed)) {
        if (uidToIdx[k] !== undefined) annotations[k] = v;
      }
      updateAnnoCount();
    }
  } catch(e) {}
}

function updateAnnoButtons() {
  const container = document.getElementById('anno-buttons');
  if (!selectedContour) { container.innerHTML = ''; return; }
  const uid = selectedContour.uid;
  const current = annotations[uid] || null;

  let html = '';
  if (annoMode === 'binary') {
    html += btnHtml('True', 'true', current === 'true', 'active-true', 'Y');
    html += btnHtml('False', 'false_positive', current === 'false_positive', 'active-false', 'N');
    html += btnHtml('Clear', null, false, '', 'U');
  } else {
    const types = ['artery','arteriole','vein','capillary','lymphatic','collecting_lymphatic','false_positive'];
    const keys  = ['A','R','V','C','L','G','N'];
    for (let i = 0; i < types.length; i++) {
      const isActive = current === types[i];
      html += btnHtml(types[i].replace('_',' '), types[i], isActive,
        types[i] === 'false_positive' ? 'active-false' : 'active-type', keys[i]);
    }
    html += btnHtml('Clear', null, false, '', 'U');
  }
  container.innerHTML = html;
}

function btnHtml(label, value, isActive, activeClass, key) {
  const cls = 'anno-btn' + (isActive ? ' ' + activeClass : '');
  const vAttr = value === null ? 'data-clear="1"' : 'data-value="' + value + '"';
  return '<button class="' + cls + '" ' + vAttr + ' title="' + key + '">' +
         label + ' <small style="opacity:0.5">' + key + '</small></button>';
}

document.getElementById('anno-buttons').addEventListener('click', function(e) {
  const btn = e.target.closest('.anno-btn');
  if (!btn || !selectedContour) return;
  const value = btn.dataset.clear ? null : btn.dataset.value;
  setAnnotation(selectedContour.uid, value);
});

// Mode toggle
document.getElementById('mode-binary').addEventListener('click', function() {
  annoMode = 'binary';
  this.classList.add('active');
  document.getElementById('mode-type').classList.remove('active');
  updateAnnoButtons();
});
document.getElementById('mode-type').addEventListener('click', function() {
  annoMode = 'type';
  this.classList.add('active');
  document.getElementById('mode-binary').classList.remove('active');
  updateAnnoButtons();
});

// Keyboard shortcuts (only when a contour is selected)
window.addEventListener('keydown', function(e) {
  if (!selectedContour) return;
  // Don't capture if user is typing in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const key = e.key.toUpperCase();
  if (annoMode === 'binary') {
    if (key === 'Y') setAnnotation(selectedContour.uid, 'true');
    else if (key === 'N') setAnnotation(selectedContour.uid, 'false_positive');
    else if (key === 'U') setAnnotation(selectedContour.uid, null);
    else return;
  } else {
    const typeMap = {A:'artery',R:'arteriole',V:'vein',C:'capillary',
                     L:'lymphatic',G:'collecting_lymphatic',N:'false_positive'};
    if (key === 'U') setAnnotation(selectedContour.uid, null);
    else if (typeMap[key]) setAnnotation(selectedContour.uid, typeMap[key]);
    else return;
  }
  e.preventDefault();
});

// Export
document.getElementById('anno-export').addEventListener('click', function() {
  const positive = [], negative = [];
  const labels = {};
  for (const [uid, label] of Object.entries(annotations)) {
    labels[uid] = label;
    if (label === 'false_positive') negative.push(uid);
    else positive.push(uid);
  }
  const data = {
    mode: annoMode,
    exported_at: new Date().toISOString(),
    source: TITLE,
    n_annotated: Object.keys(annotations).length,
    n_total: CONTOUR_DATA.length,
    annotations: labels,
    labels: labels,
    positive: positive,
    negative: negative
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'lumen_annotations.json';
  a.click();
  URL.revokeObjectURL(url);
});

// Import
document.getElementById('anno-import').addEventListener('click', function() {
  document.getElementById('anno-file').click();
});
document.getElementById('anno-file').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(ev) {
    try {
      const data = JSON.parse(ev.target.result);
      // Support both {labels: {uid: label}} and {annotations: {uid: label}} formats
      const src = data.labels || data.annotations || {};
      let imported = 0;
      for (const [uid, label] of Object.entries(src)) {
        if (uidToIdx[uid] !== undefined) {
          annotations[uid] = String(label);
          imported++;
        }
      }
      saveAnnotations();
      updateAnnoCount();
      updateAnnoButtons();
      scheduleRender();
      alert('Imported ' + imported + ' annotations');
    } catch(err) { alert('Failed to parse JSON: ' + err.message); }
  };
  reader.readAsText(file);
  e.target.value = '';
});

// Patch showMeta to track selectedContour
const _origShowMeta = showMeta;
showMeta = function(c) {
  selectedContour = c;
  _origShowMeta(c);
  updateAnnoButtons();
};
const _origHideMeta = hideMeta;
hideMeta = function() {
  selectedContour = null;
  _origHideMeta();
  document.getElementById('anno-buttons').innerHTML = '';
};

// Patch render to draw annotation highlights
const _origRender = render;
render = function() {
  _origRender();
  // Draw annotation overlays
  const nAnno = Object.keys(annotations).length;
  if (nAnno === 0 && !selectedContour) return;
  ctx.save();
  ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
  ctx.translate(panX, panY);
  ctx.scale(zoom, zoom);
  const vx1 = -panX / zoom, vy1 = -panY / zoom;
  const vx2 = (cw - panX) / zoom, vy2 = (ch - panY) / zoom;
  const lw = Math.max(2, strokeWidth * 1.5) / zoom;

  // Only iterate annotated contours + selection (not all contours)
  const toRender = new Set(Object.keys(annotations));
  if (selectedContour) toRender.add(selectedContour.uid);
  for (const uid of toRender) {
    const idx = uidToIdx[uid];
    if (idx === undefined) continue;
    const c = CONTOUR_DATA[idx];
    if (hidden.has(GROUP_LABELS[c.gi])) continue;
    if (c.bx2 < vx1 || c.bx1 > vx2 || c.by2 < vy1 || c.by1 > vy2) continue;
    const pts = c.pts;
    if (!pts || pts.length < 6) continue;
    const path = new Path2D();
    path.moveTo(pts[0], pts[1]);
    for (let j = 2; j < pts.length; j += 2) path.lineTo(pts[j], pts[j+1]);
    path.closePath();
    const label = annotations[uid];
    if (label) {
      ctx.globalAlpha = 0.25;
      ctx.fillStyle = ANNO_COLORS[label] || '#888';
      ctx.fill(path);
    }
    if (selectedContour && uid === selectedContour.uid) {
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = lw;
      ctx.stroke(path);
    }
  }
  ctx.globalAlpha = 1;
  ctx.restore();
};

// Load saved annotations on startup
loadAnnotations();
// ===== END ANNOTATION SYSTEM =====

// ===== Zoom buttons =====
document.getElementById('btn-fit').addEventListener('click', function() { fitView(); scheduleRender(); });
document.getElementById('btn-reset').addEventListener('click', function() { zoom = 1; panX = 0; panY = 0; scheduleRender(); });
document.getElementById('floating-fit').addEventListener('click', function() { fitView(); scheduleRender(); });

init();
</script>
</body>
</html>
"""
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    logger.info("Wrote %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an interactive contour overlay viewer on CZI fluorescence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--contours",
        nargs="+",
        required=True,
        type=Path,
        help="One or more JSON files containing contour/detection dicts.",
    )
    p.add_argument(
        "--group-field",
        default="group",
        help="Field to group contours by (default: group). "
        "Checked in top-level dict, then features sub-dict.",
    )
    p.add_argument(
        "--czi-path",
        type=Path,
        default=None,
        help="CZI file for fluorescence background overlay.",
    )
    p.add_argument(
        "--display-channels",
        default="0,1,2",
        help="Comma-separated CZI channel indices for R,G,B (default: 0,1,2).",
    )
    p.add_argument(
        "--channel-names",
        default=None,
        help="Comma-separated channel names (default: auto-detect from CZI filename).",
    )
    p.add_argument(
        "--scale-factor",
        type=float,
        default=0.0625,
        help="CZI downsample factor (default: 0.0625 = 1/16).",
    )
    p.add_argument(
        "--scene",
        type=int,
        default=0,
        help="CZI scene index (default: 0).",
    )
    p.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help="Pixel size in um for contours in pixel coordinates. "
        "Auto-detected from CZI if --czi-path is provided.",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Filter contours by score >= threshold.",
    )
    p.add_argument(
        "--max-contours",
        type=int,
        default=50_000,
        help="Maximum number of contours (default: 50000).",
    )
    p.add_argument(
        "--max-area-um2",
        type=float,
        default=None,
        help="Filter out contours with area > this (um^2). Useful for removing background FPs.",
    )
    p.add_argument(
        "--title",
        default="Contour Viewer",
        help="HTML page title.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("contour_viewer.html"),
        help="Output HTML file path.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    """CLI entry point for contour viewer generation."""
    args = parse_args(argv)

    # Parse display channels
    display_channels = [int(c.strip()) for c in args.display_channels.split(",")]
    channel_names = None
    if args.channel_names:
        channel_names = [n.strip() for n in args.channel_names.split(",")]

    # Determine pixel_size_um for pixel-coordinate contours
    pixel_size_um = args.pixel_size_um

    # Load fluorescence if CZI provided
    fluor_data = None
    if args.czi_path:
        if not args.czi_path.exists():
            logger.error("CZI file not found: %s", args.czi_path)
            sys.exit(1)
        fluor_data = _load_fluorescence(
            args.czi_path,
            display_channels,
            channel_names,
            args.scale_factor,
            args.scene,
        )
        # Use pixel size from CZI if not explicitly provided
        if pixel_size_um is None and fluor_data:
            pixel_size_um = fluor_data.get("pixel_size")
            if pixel_size_um:
                logger.info("Using pixel_size_um=%.4f from CZI metadata", pixel_size_um)

    # Load contours
    for jp in args.contours:
        if not jp.exists():
            logger.error("Contour file not found: %s", jp)
            sys.exit(1)

    contour_data, group_labels = load_contours(
        args.contours,
        args.group_field,
        pixel_size_um,
        args.score_threshold,
        args.max_contours,
        max_area_um2=args.max_area_um2,
    )

    if not contour_data:
        logger.warning("No valid contours found. Generating empty viewer.")

    # Assign colors
    group_colors = _assign_colors(group_labels)

    # Generate HTML
    generate_html(
        contour_data,
        group_labels,
        group_colors,
        fluor_data,
        args.title,
        args.group_field,
        args.output,
    )

    logger.info("Contour viewer: %s", args.output)
    logger.info("  %s contours in %d groups", f"{len(contour_data):,}", len(group_labels))
    if fluor_data:
        logger.info("  Fluorescence: %dx%d px", fluor_data["width"], fluor_data["height"])


if __name__ == "__main__":
    main()
