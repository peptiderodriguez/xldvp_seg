"""Interactive HTML region viewer generation.

Generates self-contained HTML viewers for region label maps overlaid on
fluorescence channel thumbnails. Two modes:

- **Single-layer** with per-region cell/nuclear stats (sidebar with
  distribution bars, detail panel, click-to-inspect).
- **Multi-layer comparison** for toggling between segmentation runs
  (e.g., point density series, different channel combos).

Both use shared JS (``region_viewer.js``) for pan/zoom, PNG download,
thickness control, and background channel switching.

Usage::

    from xldvp_seg.visualization.region_viewer import (
        extract_region_contours,
        generate_region_viewer,
        generate_multi_layer_viewer,
    )
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import find_objects

from xldvp_seg.io.html_utils import _esc, _js_esc
from xldvp_seg.utils.logging import get_logger
from xldvp_seg.visualization.encoding import safe_json
from xldvp_seg.visualization.js_loader import load_js

logger = get_logger(__name__)

# Nuclear count bar chart colors (n=0 dark, n=1 green, n=2 blue, ...)
_NUC_COLORS = ["#333", "#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

# Package-style CSS (matches html_styles.py: #0a0a0a bg, monospace, sticky header)
_BASE_CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#0a0a0a;color:#ddd;display:flex;height:100vh}
#sidebar{width:400px;overflow-y:auto;background:#111;padding:0;border-right:1px solid #333;flex-shrink:0}
.hdr{background:#111;padding:12px 16px;border-bottom:1px solid #333;position:sticky;top:0;z-index:10}
.hdr h1{font-size:1.1em;font-weight:normal;margin-bottom:4px}
.hdr .sub{font-size:11px;color:#888}
.sec{padding:8px 12px;border-bottom:1px solid #222}
.sec b{font-size:11px;color:#888}
.sec label{display:block;font-size:11px;margin:2px 0;cursor:pointer;color:#aaa}
.sec label:hover{color:#fff}
.bt{padding:8px 12px;display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid #222}
.bt button{padding:3px 10px;font-size:11px;cursor:pointer;background:#1a1a1a;color:#aaa;border:1px solid #333;border-radius:3px}
.bt button:hover{background:#222;color:#fff}
.bt button.on{background:#1a2a1a;color:#8c8;border-color:#484}
#tb{position:absolute;top:10px;right:10px;z-index:10;display:flex;gap:6px;align-items:center}
#tb button{padding:5px 12px;font-size:11px;cursor:pointer;background:rgba(17,17,17,0.85);color:#aaa;border:1px solid #444;border-radius:3px;backdrop-filter:blur(4px)}
#tb button:hover{background:rgba(34,34,34,0.9);color:#fff}
#tb label{display:flex;align-items:center;gap:4px;color:#aaa;font-size:11px;background:rgba(17,17,17,0.85);padding:3px 8px;border-radius:3px;border:1px solid #444}
#cw{flex:1;overflow:hidden;position:relative;background:#0a0a0a}
canvas{position:absolute;top:0;left:0}
"""


def extract_region_contours(label_map: np.ndarray) -> dict[int, list]:
    """Extract the largest external contour for each region in a label map.

    Uses ``scipy.ndimage.find_objects`` for efficient per-region bbox lookup,
    then ``cv2.findContours`` on the bbox crop.

    Args:
        label_map: 2D int32 label map (0 = background).

    Returns:
        Dict mapping region_id → list of ``[x, y]`` contour points.
    """
    bboxes = find_objects(label_map)
    contours: dict[int, list] = {}
    for lid in range(1, len(bboxes) + 1):
        bbox = bboxes[lid - 1]
        if bbox is None:
            continue
        r_slice, c_slice = bbox
        crop = (label_map[r_slice, c_slice] == lid).astype(np.uint8)
        cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        largest = max(cnts, key=cv2.contourArea)
        pts = largest.reshape(-1, 2).astype(float)
        pts[:, 0] += c_slice.start
        pts[:, 1] += r_slice.start
        if len(pts) >= 3:
            contours[lid] = pts.tolist()
    return contours


def generate_region_viewer(
    label_map: np.ndarray,
    fluor_thumbnails: list[tuple[str, str]],
    output_path: str | Path,
    *,
    region_stats: dict[int, dict] | None = None,
    min_cells: int = 0,
    title: str = "",
    contour_thickness: float = 1.2,
    highlight_ids: set[int] | None = None,
) -> None:
    """Generate a single-layer interactive region viewer.

    Args:
        label_map: 2D int32 label map.
        fluor_thumbnails: List of ``(name, base64_str)`` for background channels.
            First entry is used as default background.
        output_path: Path to write the HTML file.
        region_stats: Optional per-region stats from
            :func:`~xldvp_seg.analysis.region_segmentation.compute_region_nuc_stats`.
            If provided, sidebar shows cell count + nuclear distribution.
        min_cells: Minimum nucleated cell count to include a region (default: 0).
        title: Viewer title (default: auto-generated).
        contour_thickness: Default contour line width (default: 1.2).
        highlight_ids: Optional set of region IDs to render with a bold outline +
            colored fill (e.g., statistical outliers). Non-highlighted regions get
            a thin outline only. ``None`` = all regions drawn with the default
            stroke style.
    """
    img_h, img_w = label_map.shape
    contours = extract_region_contours(label_map)

    # Filter by min_cells if stats provided
    if region_stats and min_cells > 0:
        contours = {
            rid: pts
            for rid, pts in contours.items()
            if region_stats.get(rid, {}).get("count", 0) >= min_cells
        }

    # Sort by cell count (descending) if stats, else by region ID
    if region_stats:
        sorted_rids = sorted(
            contours.keys(), key=lambda k: -region_stats.get(k, {}).get("count", 0)
        )
    else:
        sorted_rids = sorted(contours.keys())

    n_reg = len(sorted_rids)
    total_cells = (
        sum(region_stats.get(rid, {}).get("count", 0) for rid in sorted_rids) if region_stats else 0
    )

    # Build JS region data
    hl_set = set(highlight_ids) if highlight_ids else set()
    js_regions = []
    for i, rid in enumerate(sorted_rids):
        hue = int(360 * i / max(n_reg, 1))
        entry = {
            "id": int(rid),
            "pts": contours[rid],
            "hue": hue,
            "hl": int(rid) in hl_set,
        }
        if region_stats and rid in region_stats:
            s = region_stats[rid]
            entry["cells"] = int(s.get("count", 0))
            entry["mean"] = float(s.get("mean_nuc", 0))
            entry["med"] = int(s.get("median_nuc", 1))
            entry["nd"] = {str(k): int(v) for k, v in s.get("nuc_dist", {}).items()}
        js_regions.append(entry)

    if not title:
        title = f"Region Viewer — {n_reg} regions"
        if total_cells:
            title += f", {total_cells:,} nucleated cells"

    has_stats = region_stats is not None and any("cells" in r for r in js_regions)

    html = _build_single_layer_html(
        js_regions, fluor_thumbnails, img_w, img_h, title, has_stats, contour_thickness
    )

    Path(output_path).write_text(html)
    logger.info("Wrote %s (%d regions, %.1f MB)", output_path, n_reg, len(html) / 1e6)


def generate_multi_layer_viewer(
    label_maps: list[tuple[str, np.ndarray]],
    fluor_thumbnails: list[tuple[str, str]],
    output_path: str | Path,
    *,
    title: str = "",
    contour_thickness: float = 1.2,
) -> None:
    """Generate a multi-layer comparison viewer.

    Args:
        label_maps: List of ``(name, label_map_array)`` tuples.
        fluor_thumbnails: List of ``(name, base64_str)`` for backgrounds.
        output_path: Path to write the HTML file.
        title: Viewer title.
        contour_thickness: Default contour line width.
    """
    if not label_maps:
        logger.warning("No label maps provided")
        return

    img_h, img_w = label_maps[0][1].shape

    js_layers = []
    for li, (name, lbl) in enumerate(label_maps):
        regions = extract_region_contours(lbl)
        base_hue = (li * 33) % 360
        js_regions = []
        for j, (rid, pts) in enumerate(regions.items()):
            hue = (base_hue + j * 7) % 360
            js_regions.append({"id": int(rid), "pts": pts, "hue": hue})
        is_filled = "filled" in name.lower()
        js_layers.append(
            {
                "name": name,
                "n": len(regions),
                "regions": js_regions,
                "filled": is_filled,
            }
        )

    if not title:
        title = f"Multi-Layer Viewer — {len(js_layers)} layers"

    html = _build_multi_layer_html(
        js_layers, fluor_thumbnails, img_w, img_h, title, contour_thickness
    )

    Path(output_path).write_text(html)
    logger.info("Wrote %s (%d layers, %.1f MB)", output_path, len(js_layers), len(html) / 1e6)


# ---------------------------------------------------------------------------
# Internal HTML builders
# ---------------------------------------------------------------------------


def _toolbar_html(contour_thickness: float) -> str:
    return f"""\
<div id="tb">
<button onclick="fitView()">Fit View</button>
<button onclick="dlPng(true)">PNG + Masks</button>
<button onclick="dlPng(false)">PNG Only</button>
<label>Width <input type=range id=lw min=0.3 max=5 step=0.1 value={contour_thickness}
 oninput="_lwV=parseFloat(this.value);draw()" style="width:80px"><span id=lwl>{contour_thickness}</span></label>
</div>"""


def _bg_radios_html(fluor_thumbnails: list[tuple[str, str]]) -> str:
    html = '<div class="sec"><b>Background</b>\n'
    for i, (name, _) in enumerate(fluor_thumbnails):
        checked = " checked" if i == 0 else ""
        esc_name = _esc(name)
        html += f'<label><input type=radio name=bg value="{esc_name}"{checked} onchange="setBg(this.value)"> {esc_name}</label>\n'
    html += "</div>"
    return html


def _bg_js(fluor_thumbnails: list[tuple[str, str]]) -> str:
    lines = ["const BGS = {};"]
    for name, b64 in fluor_thumbnails:
        lines.append(f"BGS['{_js_esc(name)}'] = 'data:image/jpeg;base64,{b64}';")
    return "\n".join(lines)


def _build_single_layer_html(
    js_regions, fluor_thumbnails, img_w, img_h, title, has_stats, contour_thickness
):
    n_reg = len(js_regions)

    # Region sidebar CSS
    region_css = """\
.r{margin:1px 4px;padding:5px 8px;border:1px solid transparent;border-radius:3px;cursor:pointer;font-size:12px}
.r:hover{background:#1a1a1a}.r.sel{border-color:#fff;background:#1a1a1a}
.r label{display:flex;align-items:flex-start;gap:6px;cursor:pointer}
.r .info{flex:1}.r .nm{font-weight:600}
.r .st{color:#777;font-size:10px;margin-top:1px}
.r .bar{display:flex;height:8px;border-radius:2px;overflow:hidden;margin-top:2px;background:#222}
.r .bar div{height:100%}
#detail{padding:10px;background:#0f0f0f;border-top:1px solid #333;min-height:80px;position:sticky;bottom:0}
#detail h3{font-size:12px;margin-bottom:4px}
#detail table{font-size:11px;border-collapse:collapse;width:100%}
#detail td{padding:2px 6px;border-bottom:1px solid #1a1a1a}
"""

    region_js = load_js("region_viewer")

    esc_title = _esc(title)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{esc_title}</title>
<style>{_BASE_CSS}{region_css}</style></head><body>
<div id="sidebar">
<div class="hdr"><h1>{esc_title}</h1>
<div class="sub">{n_reg} regions</div></div>
<div class="bt"><button onclick="allOn()">All On</button><button onclick="allOff()">All Off</button>
<button id="sb" onclick="toggleSolo()">Solo</button></div>
{_bg_radios_html(fluor_thumbnails)}
<div id="rl"></div>
<div id="detail"><h3>Click a region to inspect</h3></div>
</div>
<div id="cw">{_toolbar_html(contour_thickness)}<canvas id="cv"></canvas></div>
<script>
const R = {safe_json(js_regions)};
const W = {img_w}, H = {img_h};
const NC = {safe_json(_NUC_COLORS)};
{_bg_js(fluor_thumbnails)}

{region_js}

let V = new Set(R.map((_, i) => i));
let si = null, solo = false;

function drawRegions(ctx, zm, lw) {{
    // Two-pass: thin/normal first, highlighted on top so bold outlines aren't covered
    for (let pass = 0; pass < 2; pass++) {{
        for (let i = 0; i < R.length; i++) {{
            if (!V.has(i)) continue;
            const r = R[i], p = r.pts;
            if (!p || p.length < 3) continue;
            const hl = !!r.hl;
            if (pass === 0 && hl) continue;
            if (pass === 1 && !hl) continue;
            ctx.beginPath();
            ctx.moveTo(p[0][0], p[0][1]);
            for (let j = 1; j < p.length; j++) ctx.lineTo(p[j][0], p[j][1]);
            ctx.closePath();
            const sel = i === si;
            if (hl) {{ ctx.fillStyle = 'hsla(' + r.hue + ',70%,55%,0.3)'; ctx.fill(); }}
            else if (sel) {{ ctx.fillStyle = 'hsla(' + r.hue + ',70%,55%,0.2)'; ctx.fill(); }}
            ctx.strokeStyle = sel ? '#fff' : 'hsl(' + r.hue + ',70%,55%)';
            const w = hl ? lw * 2.5 : lw;
            ctx.lineWidth = (sel ? w * 1.5 : w) / zm;
            ctx.stroke();
        }}
    }}
}}

_canvas.addEventListener('click', e => {{
    const [mx, my] = canvasToImage(e);
    let f = null;
    for (let i = 0; i < R.length; i++) {{
        if (!V.has(i)) continue;
        if (pointInPoly(mx, my, R[i].pts)) {{ f = i; break; }}
    }}
    selectRegion(f);
}});

function selectRegion(idx) {{
    si = idx;
    document.querySelectorAll('.r').forEach((el, i) => el.classList.toggle('sel', i === idx));
    showDetail(idx);
    draw();
}}

function showDetail(idx) {{
    const el = document.getElementById('detail');
    if (idx === null) {{ el.innerHTML = '<h3>Click a region to inspect</h3>'; return; }}
    const r = R[idx];
    if (!r.nd) {{ el.innerHTML = '<h3 style="color:hsl(' + r.hue + ',70%,55%)">Region ' + r.id + '</h3>'; return; }}
    const nd = r.nd, total = Object.values(nd).reduce((a, b) => a + b, 0);
    let rows = '';
    for (const [k, v] of Object.entries(nd)) {{
        rows += '<tr><td>n=' + k + '</td><td>' + v.toLocaleString() + '</td><td>' + (100 * v / total).toFixed(1) + '%</td></tr>';
    }}
    el.innerHTML = '<h3 style="color:hsl(' + r.hue + ',70%,55%)">Region ' + r.id + '</h3>' +
        '<p style="font-size:11px;margin:3px 0">Nucleated: <b>' + (r.cells || 0).toLocaleString() +
        '</b> | Mean: <b>' + (r.mean || 0).toFixed(2) + '</b> | Median: <b>' + (r.med || 0) + '</b></p>' +
        '<table>' + rows + '</table>';
}}

function allOn() {{ V = new Set(R.map((_, i) => i)); syncCB(); draw(); }}
function allOff() {{ V = new Set(); syncCB(); draw(); }}
function syncCB() {{ document.querySelectorAll('.r input').forEach((c, i) => c.checked = V.has(i)); }}
function toggleSolo() {{
    solo = !solo;
    const b = document.getElementById('sb');
    b.className = solo ? 'on' : '';
    b.textContent = solo ? 'Solo ON' : 'Solo';
}}
function tog(i, on) {{
    if (solo) {{ V = new Set([i]); syncCB(); }}
    else {{ if (on) V.add(i); else V.delete(i); }}
    draw();
}}

// Build sidebar
const rl = document.getElementById('rl');
R.forEach((r, i) => {{
    const d = document.createElement('div');
    d.className = 'r';
    let inner = '<label><input type=checkbox checked onchange="tog(' + i + ',this.checked)">' +
        '<span style="color:hsl(' + r.hue + ',70%,55%)">\\u25A0</span>' +
        '<div class=info><div class=nm>#' + r.id + '</div>';
"""
    if has_stats:
        html += """
    if (r.cells !== undefined && r.nd) {{
        const nd = r.nd, total = Object.values(nd).reduce((a, b) => a + b, 0);
        let barHtml = '';
        for (const [k, v] of Object.entries(nd)) {{
            const pct = (100 * v / total).toFixed(1);
            const col = NC[Math.min(+k, NC.length - 1)];
            barHtml += '<div style="width:' + pct + '%;background:' + col + '" title="n=' + k + ': ' + pct + '%"></div>';
        }}
        inner += '<div class=st>' + r.cells.toLocaleString() + ' cells | mean ' + r.mean.toFixed(2) + '</div>';
        inner += '<div class=bar>' + barHtml + '</div>';
    }}
"""
    html += """
    inner += '</div></label>';
    d.innerHTML = inner;
    d.addEventListener('click', () => selectRegion(i));
    rl.appendChild(d);
}});
</script></body></html>"""
    return html


def _build_multi_layer_html(js_layers, fluor_thumbnails, img_w, img_h, title, contour_thickness):
    n_layers = len(js_layers)
    layer_css = """\
.layer{margin:3px 0;padding:4px 8px;border:1px solid #333;border-radius:4px;cursor:pointer;font-size:12px}
.layer:hover{background:#1a1a1a}
.layer.filled{border-left:3px solid #4a9}
.layer label{display:flex;align-items:center;gap:6px;cursor:pointer}
.layer .n{color:#888;font-size:11px;margin-left:auto}
.layer .tag{font-size:9px;padding:1px 4px;border-radius:2px;margin-left:4px}
.tag.f{background:#2a5a4a;color:#8fc}
.tag.c{background:#3a3a5a;color:#aaf}
"""

    region_js = load_js("region_viewer")

    esc_title = _esc(title)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{esc_title}</title>
<style>{_BASE_CSS}{layer_css}</style></head><body>
<div id="sidebar">
<div class="hdr"><h1>{esc_title}</h1>
<div class="sub">{n_layers} segmentation layers</div></div>
<div class="bt">
<button onclick="allOn()">All On</button><button onclick="allOff()">All Off</button>
<button onclick="showFilled()">Filled</button><button onclick="showClean()">Clean</button>
<button id="sb" onclick="toggleSolo()">Solo</button></div>
{_bg_radios_html(fluor_thumbnails)}
<div id="layers"></div>
</div>
<div id="cw">{_toolbar_html(contour_thickness)}<canvas id="cv"></canvas></div>
<script>
const LAYERS = {safe_json(js_layers)};
const W = {img_w}, H = {img_h};
{_bg_js(fluor_thumbnails)}

{region_js}

let V = new Set();
LAYERS.forEach((l, i) => {{ if (l.filled) V.add(i); }});
let solo = false;

function drawRegions(ctx, zm, lw) {{
    for (let li = 0; li < LAYERS.length; li++) {{
        if (!V.has(li)) continue;
        for (const r of LAYERS[li].regions) {{
            const p = r.pts;
            if (!p || p.length < 3) continue;
            ctx.beginPath();
            ctx.moveTo(p[0][0], p[0][1]);
            for (let j = 1; j < p.length; j++) ctx.lineTo(p[j][0], p[j][1]);
            ctx.closePath();
            ctx.strokeStyle = 'hsl(' + r.hue + ',70%,55%)';
            ctx.lineWidth = lw / zm;
            ctx.stroke();
        }}
    }}
}}

function allOn() {{ V = new Set(LAYERS.map((_, i) => i)); syncCB(); draw(); }}
function allOff() {{ V = new Set(); syncCB(); draw(); }}
function showFilled() {{ V = new Set(); LAYERS.forEach((l, i) => {{ if (l.filled) V.add(i); }}); syncCB(); draw(); }}
function showClean() {{ V = new Set(); LAYERS.forEach((l, i) => {{ if (!l.filled) V.add(i); }}); syncCB(); draw(); }}
function syncCB() {{ document.querySelectorAll('.layer input').forEach((c, i) => c.checked = V.has(i)); }}
function toggleSolo() {{
    solo = !solo;
    const b = document.getElementById('sb');
    b.className = solo ? 'on' : '';
    b.textContent = solo ? 'Solo ON' : 'Solo';
}}
function tog(i, on) {{
    if (solo) {{ V = new Set([i]); syncCB(); }}
    else {{ if (on) V.add(i); else V.delete(i); }}
    draw();
}}

const ld = document.getElementById('layers');
// Phase 2.4 fix: build DOM with textContent for l.name so a hostile layer
// name (from file stems / label-map files) can't inject HTML.
LAYERS.forEach((l, i) => {{
    const d = document.createElement('div');
    d.className = 'layer' + (l.filled ? ' filled' : '');
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = V.has(i);
    cb.addEventListener('change', (e) => tog(i, e.target.checked));
    label.appendChild(cb);
    const nameSpan = document.createElement('span');
    nameSpan.textContent = l.name;
    label.appendChild(nameSpan);
    const tagSpan = document.createElement('span');
    tagSpan.className = 'tag ' + (l.filled ? 'f' : 'c');
    tagSpan.textContent = l.filled ? 'filled' : 'clean';
    label.appendChild(tagSpan);
    const nSpan = document.createElement('span');
    nSpan.className = 'n';
    nSpan.textContent = l.n;
    label.appendChild(nSpan);
    d.appendChild(label);
    ld.appendChild(d);
}});
</script></body></html>"""
    return html
