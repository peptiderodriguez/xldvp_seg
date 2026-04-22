"""Generic HTML builder for the linked 3D-UMAP + slide-thumbnail viewer.

Two ad-hoc n45 scripts (manifold_linked_viewer.py, lmd_viewer.py) both built
the same 2-panel layout:

  - Left: plotly 3D UMAP of cells, colored by group.
  - Right: whole-slide fluorescence thumbnail with cells drawn as dots.
  - Click a point on the UMAP -> that group's cells light up on the slide.

This module exposes a single public function, :func:`build_linked_viewer_html`,
that emits a fully self-contained HTML string for the linked view. Both
ad-hoc callers collapse to thin CLI wrappers that only prepare the input
arrays and positions.

No new runtime dependencies beyond what xldvp_seg already uses: numpy, PIL,
and plotly (via CDN for client-side rendering).
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from xldvp_seg.utils.logging import get_logger
from xldvp_seg.visualization.colors import shuffled_hsv_palette
from xldvp_seg.visualization.encoding import safe_json

logger = get_logger(__name__)


_DEFAULT_HOVER = "group %{customdata}<extra></extra>"


def _thumbnail_data_uri(thumbnail_rgb: np.ndarray, quality: int = 80) -> tuple[str, int, int]:
    """Encode an (H, W, 3) uint8 array as a JPEG data URI. Returns (b64, W, H)."""
    if thumbnail_rgb.ndim != 3 or thumbnail_rgb.shape[2] != 3:
        raise ValueError(f"thumbnail_rgb must have shape (H, W, 3); got {thumbnail_rgb.shape}")
    arr = np.ascontiguousarray(thumbnail_rgb, dtype=np.uint8)
    img_h, img_w = arr.shape[:2]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, int(img_w), int(img_h)


def _build_plotly_figure_html(
    umap_coords_3d: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    hover_template: str,
) -> str:
    """Build the plotly 3D scatter <div> (CDN-loaded plotly.js).

    Uses orthographic projection + cube aspect mode + turntable drag mode for
    a less "dramatic" perspective than plotly's default.
    """
    # Local import so that module import doesn't pull plotly when unused.
    import plotly.graph_objects as go

    point_colors = [
        f"rgb({int(palette[g, 0])},{int(palette[g, 1])},{int(palette[g, 2])})"
        for g in labels.tolist()
    ]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=umap_coords_3d[:, 0].tolist(),
                y=umap_coords_3d[:, 1].tolist(),
                z=umap_coords_3d[:, 2].tolist(),
                mode="markers",
                marker=dict(size=2, color=point_colors, opacity=0.85),
                customdata=labels.tolist(),
                hovertemplate=hover_template,
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="#0a0a0a",
        font=dict(color="#ddd"),
        scene=dict(
            xaxis=dict(backgroundcolor="#0a0a0a", gridcolor="#333", color="#aaa", title="UMAP 1"),
            yaxis=dict(backgroundcolor="#0a0a0a", gridcolor="#333", color="#aaa", title="UMAP 2"),
            zaxis=dict(backgroundcolor="#0a0a0a", gridcolor="#333", color="#aaa", title="UMAP 3"),
            bgcolor="#0a0a0a",
            aspectmode="cube",
            dragmode="turntable",
            camera=dict(projection=dict(type="orthographic")),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="umap-plot")


_CSS_TEMPLATE = """
body{{background:#0a0a0a;color:#e0e0e0;margin:0;font-family:system-ui,sans-serif;overflow:hidden}}
#root{{display:flex;height:100vh;width:100vw}}
#left{{flex:1;min-width:45%;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}}
#right{{flex:1;min-width:45%;display:flex;align-items:center;justify-content:center;overflow:hidden;border-left:1px solid #222;position:relative}}
#umap-plot{{width:100% !important;height:100% !important}}
#slide-wrap{{position:relative;display:block;max-width:100%;max-height:100%;background:#000}}
#slide-img{{display:block;width:100%;height:100%;object-fit:contain;opacity:{slide_opacity}}}
#slide-canvas{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}}
#info{{position:absolute;top:10px;right:10px;background:rgba(10,10,10,0.92);padding:8px 12px;border:1px solid #333;font-size:11px;z-index:10;max-width:240px}}
#info b{{color:#fff}}
#reset{{margin-top:6px;cursor:pointer;color:#8ff;font-size:10px}}
"""


_JS_TEMPLATE = """
const CELLS = {cells_json};
const COLORS = {colors_json};
const IMG_W = {img_w};
const IMG_H = {img_h};
const SELECTED_ONLY = {selected_only_js};

const canvas = document.getElementById('slide-canvas');
const ctx = canvas.getContext('2d');
let selectedGroup = null;

function resizeCanvas() {{
  const right = document.getElementById('right');
  const wrap = document.getElementById('slide-wrap');
  const aspect = IMG_W / IMG_H;
  let w = Math.max(10, right.clientWidth - 10);
  let h = w / aspect;
  const maxH = Math.max(10, right.clientHeight - 10);
  if (h > maxH) {{ h = maxH; w = h * aspect; }}
  wrap.style.width = w + 'px';
  wrap.style.height = h + 'px';
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  drawCells();
  const plot = document.getElementById('umap-plot');
  if (plot && window.Plotly) {{ Plotly.Plots.resize(plot); }}
}}

function drawCells() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const N = CELLS.x_rel.length;
  const highlighting = selectedGroup !== null;

  if (!SELECTED_ONLY) {{
    // Always-draw mode: dim all cells, emphasize selection if any.
    ctx.globalAlpha = highlighting ? 0.18 : 0.55;
    for (let i = 0; i < N; i++) {{
      const g = CELLS.group[i];
      if (highlighting && g === selectedGroup) continue;
      const c = COLORS[g];
      ctx.fillStyle = `rgba(${{c[0]}},${{c[1]}},${{c[2]}},1)`;
      const x = CELLS.x_rel[i] * canvas.width;
      const y = CELLS.y_rel[i] * canvas.height;
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }}
  }}

  if (highlighting) {{
    ctx.globalAlpha = 1.0;
    const c = COLORS[selectedGroup];
    ctx.fillStyle = `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < N; i++) {{
      if (CELLS.group[i] !== selectedGroup) continue;
      const x = CELLS.x_rel[i] * canvas.width;
      const y = CELLS.y_rel[i] * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }}
  }}
  ctx.globalAlpha = 1.0;
}}

function resetSelection() {{
  selectedGroup = null;
  const info = document.getElementById('sel-info');
  if (info) info.textContent = '';
  drawCells();
}}

function onPlotlyClick(evt) {{
  if (!evt || !evt.points || evt.points.length === 0) return;
  const pt = evt.points[0];
  selectedGroup = pt.customdata;
  const n = CELLS.group.filter(g => g === selectedGroup).length;
  const c = COLORS[selectedGroup];
  const info = document.getElementById('sel-info');
  if (info) {{
    info.innerHTML =
      '<b style="color:rgb(' + c.join(',') + ')">Group ' + selectedGroup + '</b>: '
      + n + ' cells';
  }}
  drawCells();
}}

function armHandlers() {{
  const plot = document.getElementById('umap-plot');
  if (!plot || !plot.on) {{ setTimeout(armHandlers, 200); return; }}
  plot.on('plotly_click', onPlotlyClick);
}}

window.addEventListener('resize', resizeCanvas);
window.addEventListener('load', () => {{ resizeCanvas(); armHandlers(); }});
"""


def build_linked_viewer_html(
    umap_coords_3d: np.ndarray,
    positions_um: np.ndarray,
    labels: np.ndarray,
    thumbnail_rgb: np.ndarray,
    slide_extent_um: tuple[float, float],
    *,
    slide_opacity: float = 1.0,
    selected_only_on_slide: bool = True,
    group_colors: np.ndarray | None = None,
    title: str = "Manifold viewer",
    hover_template: str | None = None,
    info_extra_html: str = "",
) -> str:
    """Build a self-contained linked 3D-UMAP + slide-viewer HTML string.

    Layout: flex row. Left pane = plotly 3D UMAP; right pane = slide thumbnail
    with cells as a canvas overlay. Clicking a point in the UMAP highlights
    that group's cells on the slide.

    Args:
        umap_coords_3d: (N, 3) 3D UMAP embedding. One row per cell.
        positions_um: (N, 2) slide x,y coordinates in micrometers.
        labels: (N,) manifold group id for each cell (0-based).
        thumbnail_rgb: (H, W, 3) uint8 RGB image of the slide.
        slide_extent_um: (width_um, height_um) -- slide extent in native
            micrometers. Used to convert positions_um -> (0..1) canvas
            fractions.
        slide_opacity: CSS opacity for the slide image (0..1, default 1.0).
        selected_only_on_slide: if True, cells only appear on the slide when
            a group is selected in the UMAP. If False, all cells are drawn
            dimmed and the selected group is highlighted on top.
        group_colors: optional (K, 3) uint8 RGB palette. If None,
            shuffled_hsv_palette(K, seed=0) is used.
        title: HTML <title> and info-panel header.
        hover_template: plotly hovertemplate string (None -> default).
        info_extra_html: extra HTML injected into the info panel for
            caller-specific summaries (replicate counts, etc.).

    Returns:
        Self-contained HTML string (no external deps except plotly CDN).
    """
    # --- Input validation ---------------------------------------------------
    umap_coords_3d = np.asarray(umap_coords_3d, dtype=np.float32)
    positions_um = np.asarray(positions_um, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if umap_coords_3d.ndim != 2 or umap_coords_3d.shape[1] != 3:
        raise ValueError(f"umap_coords_3d must have shape (N, 3); got {umap_coords_3d.shape}")
    if positions_um.ndim != 2 or positions_um.shape[1] != 2:
        raise ValueError(f"positions_um must have shape (N, 2); got {positions_um.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D; got shape {labels.shape}")

    n = umap_coords_3d.shape[0]
    if positions_um.shape[0] != n or labels.shape[0] != n:
        raise ValueError(
            "umap_coords_3d, positions_um, and labels must share N; "
            f"got {umap_coords_3d.shape[0]}, {positions_um.shape[0]}, {labels.shape[0]}"
        )

    if n == 0:
        raise ValueError("build_linked_viewer_html requires at least one cell")

    if labels.min() < 0:
        raise ValueError(f"labels must be non-negative; min={int(labels.min())}")

    width_um, height_um = float(slide_extent_um[0]), float(slide_extent_um[1])
    if width_um <= 0 or height_um <= 0:
        raise ValueError(f"slide_extent_um must be positive; got ({width_um}, {height_um})")

    slide_opacity = float(np.clip(slide_opacity, 0.0, 1.0))

    # --- Color palette ------------------------------------------------------
    k = int(labels.max()) + 1
    if group_colors is None:
        palette = shuffled_hsv_palette(k, seed=0)
    else:
        palette = np.asarray(group_colors, dtype=np.uint8)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError(f"group_colors must have shape (K, 3); got {palette.shape}")
        if palette.shape[0] < k:
            raise ValueError(
                f"group_colors has only {palette.shape[0]} rows but labels " f"reference {k} groups"
            )

    # --- Thumbnail encoding -------------------------------------------------
    thumb_b64, img_w, img_h = _thumbnail_data_uri(thumbnail_rgb)

    # --- Coordinate normalization (um -> 0..1) ------------------------------
    x_rel = np.clip(positions_um[:, 0] / width_um, 0.0, 1.0).astype(np.float32)
    y_rel = np.clip(positions_um[:, 1] / height_um, 0.0, 1.0).astype(np.float32)

    # --- Plotly figure ------------------------------------------------------
    hover = hover_template if hover_template is not None else _DEFAULT_HOVER
    plotly_div = _build_plotly_figure_html(umap_coords_3d, labels, palette, hover)

    # --- JS data payload (safe_json escapes </ and <!--) --------------------
    cells_json = safe_json(
        {
            "x_rel": x_rel.tolist(),
            "y_rel": y_rel.tolist(),
            "group": labels.astype(np.int32).tolist(),
        }
    )
    colors_json = safe_json(
        [[int(palette[i, 0]), int(palette[i, 1]), int(palette[i, 2])] for i in range(k)]
    )

    css = _CSS_TEMPLATE.format(slide_opacity=slide_opacity)
    js = _JS_TEMPLATE.format(
        cells_json=cells_json,
        colors_json=colors_json,
        img_w=img_w,
        img_h=img_h,
        selected_only_js="true" if selected_only_on_slide else "false",
    )

    # Escape title for the <title> tag (no user-controlled quotes expected,
    # but be defensive). We only need HTML-escape here since no quotes go
    # into attributes.
    title_safe = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title_safe}</title>
<style>{css}</style>
</head><body>
<div id="root">
  <div id="left">{plotly_div}</div>
  <div id="right">
    <div id="slide-wrap">
      <img id="slide-img" src="data:image/jpeg;base64,{thumb_b64}">
      <canvas id="slide-canvas"></canvas>
    </div>
    <div id="info">
      <b>{title_safe}</b><br>
      {n:,} cells &middot; {k:,} groups<br>
      Click a point in UMAP to highlight its group on the slide.<br>
      {info_extra_html}
      <span id="sel-info"></span>
      <div id="reset" onclick="resetSelection()">[reset]</div>
    </div>
  </div>
</div>
<script>{js}</script>
</body></html>"""

    logger.debug(
        "Built linked viewer HTML: N=%d cells, K=%d groups, thumb=%dx%d, bytes=%d",
        n,
        k,
        img_w,
        img_h,
        len(html),
    )
    return html
