"""Core HTML export -- page generators for annotation interfaces.

Delegates image utilities to ``html_utils.py``, CSS to ``html_styles.py``,
and JavaScript to ``html_scripts.py``.  Contains:

**Page generators:** generate_annotation_page, generate_index_page,
    generate_dual_index_page, export_samples_to_html
**Vessel pages:** generate_vessel_annotation_page, generate_vessel_index_page,
    export_vessel_samples_to_html

MK/HSPC functions (load_samples_from_ram, create_mk_hspc_index, etc.) are
backward-compatible shims delegating to ``html_generator.py``.

See also ``html_generator.py`` for the class-based HTMLPageGenerator API.
"""

import json
from pathlib import Path

from xldvp_seg.io.html_scripts import (
    generate_preload_annotations_js,
    get_js,
    get_vessel_js,
)
from xldvp_seg.io.html_styles import get_css, get_vessel_css
from xldvp_seg.io.html_utils import (
    HDF5_COMPRESSION_KWARGS,
    HDF5_COMPRESSION_NAME,
    _esc,
    compose_tile_rgb,
    create_hdf5_dataset,
    draw_mask_contour,
    get_largest_connected_component,
    image_to_base64,
    percentile_normalize,
)
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Re-export everything for backward compatibility -- external code imports
# these names from xldvp_seg.io.html_export and must keep working.
__all__ = [
    # html_utils re-exports
    "_esc",
    "HDF5_COMPRESSION_KWARGS",
    "HDF5_COMPRESSION_NAME",
    "create_hdf5_dataset",
    "get_largest_connected_component",
    "percentile_normalize",
    "draw_mask_contour",
    "image_to_base64",
    "compose_tile_rgb",
    # html_styles re-exports
    "get_css",
    "get_vessel_css",
    # html_scripts re-exports
    "get_js",
    "get_vessel_js",
    "generate_preload_annotations_js",
    # Page generators (defined here)
    "generate_annotation_page",
    "generate_index_page",
    "generate_dual_index_page",
    "export_samples_to_html",
    "generate_vessel_annotation_page",
    "generate_vessel_index_page",
    "export_vessel_samples_to_html",
    # MK/HSPC shims
    "load_samples_from_ram",
    "create_mk_hspc_index",
    "generate_mk_hspc_page_html",
    "generate_mk_hspc_pages",
    "export_mk_hspc_html_from_ram",
]


def generate_annotation_page(
    samples,
    cell_type,
    page_num,
    total_pages,
    title=None,
    page_prefix="page",
    experiment_name=None,
    channel_legend=None,
    subtitle=None,
    include_preload_script=False,
):
    """
    Generate an HTML annotation page.

    Args:
        samples: List of sample dicts with keys:
            - uid: Unique identifier
            - image: Base64 encoded image string
            - stats: Dict of stats to display (e.g., {'area_um2': 150.5, 'confidence': 0.95})
        cell_type: Type identifier (e.g., 'nmj', 'mk', 'hspc')
        page_num: Current page number
        total_pages: Total number of pages
        title: Optional title override
        page_prefix: Prefix for page filenames
        experiment_name: Optional experiment name for localStorage isolation
        channel_legend: Optional dict mapping colors to channel names,
            e.g., {'red': 'nuc488', 'green': 'Bgtx647', 'blue': 'NfL750'}
        subtitle: Optional subtitle (e.g., filename) shown below title
        include_preload_script: If True, include script tag for preload_annotations.js

    Returns:
        HTML string
    """
    if title is None:
        title = cell_type.upper()
    title = _esc(title)

    # Build navigation
    nav_html = '<div class="nav-buttons">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{page_prefix}_{page_num-1}.html" class="nav-btn">Prev</a>'
    nav_html += f'<span class="page-info">Page {page_num} / {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{page_prefix}_{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += "</div>"

    # Build channel legend HTML if provided
    channel_legend_html = ""
    ch_r_label = channel_legend.get("red", "Ch-R") if channel_legend else "Ch-R"
    ch_g_label = channel_legend.get("green", "Ch-G") if channel_legend else "Ch-G"
    ch_b_label = channel_legend.get("blue", "Ch-B") if channel_legend else "Ch-B"
    # Shorten labels for buttons
    ch_r_label = ch_r_label.split("(")[0].strip()[:10] if ch_r_label else "Ch-R"
    ch_g_label = ch_g_label.split("(")[0].strip()[:10] if ch_g_label else "Ch-G"
    ch_b_label = ch_b_label.split("(")[0].strip()[:10] if ch_b_label else "Ch-B"

    if channel_legend:
        channel_legend_html = (
            '<div class="channel-legend"><span class="stats-label">Channels:</span>'
        )
        if "red" in channel_legend:
            channel_legend_html += f'<span class="ch-red">R={_esc(channel_legend["red"])}</span>'
        if "green" in channel_legend:
            channel_legend_html += (
                f'<span class="ch-green">G={_esc(channel_legend["green"])}</span>'
            )
        if "blue" in channel_legend:
            channel_legend_html += f'<span class="ch-blue">B={_esc(channel_legend["blue"])}</span>'
        channel_legend_html += "</div>"

    # Build subtitle HTML if provided
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="header-subtitle">{_esc(subtitle)}</div>'

    # Build cards
    cards_html = ""
    for sample in samples:
        uid = _esc(sample["uid"])
        img_b64 = sample["image"]
        mime = sample.get("mime_type", "jpeg")
        if mime not in ("jpeg", "png"):
            mime = "jpeg"  # Safe default
        stats = sample.get("stats", {})

        # Format stats line
        stats_parts = []
        if "area_um2" in stats:
            stats_parts.append(f"{stats['area_um2']:.1f} &micro;m&sup2;")
        if "area_px" in stats:
            stats_parts.append(f"{stats['area_px']:.0f} px")
        if "rf_prediction" in stats:
            stats_parts.append(f"RF: {stats['rf_prediction']:.2f}")
        elif "score" in stats:
            stats_parts.append(f"score: {stats['score']:.2f}")
        if "sma_ratio" in stats:
            stats_parts.append(f"sma: {stats['sma_ratio']:.2f}")
        if "diameter_um" in stats:
            stats_parts.append(f"&empty; {stats['diameter_um']:.0f} &micro;m")
        if "scale" in stats:
            stats_parts.append(f"{_esc(stats['scale'])}")
        if "solidity" in stats:
            stats_parts.append(f"sol: {stats['solidity']:.2f}")
        # Only show confidence if it's numeric and not 1.0 (i.e., after classifier training)
        if "confidence" in stats:
            conf = stats["confidence"]
            if isinstance(conf, (int, float)) and conf < 0.999:
                stats_parts.append(f"{conf*100:.0f}%")
            elif isinstance(conf, str):
                stats_parts.append(f"{_esc(conf)}")
        if "marker_class" in stats:
            mc = _esc(str(stats["marker_class"]))
            # Use stored marker_color (from classify_islet_marker contour color) if available
            mc_color = stats.get("marker_color", "#888")
            if mc in ("multi",):
                mc_color = "#ffaa00"
            elif mc in ("none",):
                mc_color = "#888"
            stats_parts.append(f'<span style="color:{mc_color};font-weight:bold">{mc}</span>')
        if "islet_id" in stats and stats["islet_id"] is not None:
            stats_parts.append(f'I{_esc(str(stats["islet_id"]))}')

        stats_str = " | ".join(stats_parts) if stats_parts else ""

        img_clean_b64 = sample.get("image_clean", "")
        img_contour_only_b64 = sample.get("image_contour_only", "")
        # Base layer: clean image (gets channel SVG filters)
        # Contour layer: green on black with mix-blend-mode:lighten (always visible)
        base_src = img_clean_b64 or img_b64
        contour_src = img_contour_only_b64 or ""
        contour_img = (
            (
                f'<img class="img-contour" src="data:image/{mime};base64,{contour_src}" '
                f'style="position:absolute;top:0;left:0;width:100%;height:100%;'
                f'mix-blend-mode:lighten;pointer-events:none;object-fit:contain" alt="">'
            )
            if contour_src
            else ""
        )
        cards_html += f"""
        <div class="card" id="{uid}" data-label="-1">
            <div class="card-img-container" style="position:relative">
                <img class="img-base" src="data:image/{mime};base64,{base_src}" alt="{uid}">
                {contour_img}
            </div>
            <div class="card-info">
                <div class="card-meta">
                    <div class="card-id">{uid}</div>
                    <div class="card-stats">{stats_str}</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{uid}', 1)">Y</button>
                    <button class="btn btn-unsure" onclick="setLabel('{uid}', 2)">?</button>
                    <button class="btn btn-no" onclick="setLabel('{uid}', 0)">N</button>
                </div>
            </div>
        </div>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title} - Page {page_num}/{total_pages}</title>
    <style>{get_css()}</style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <div>
                <h1>{title} - Page {page_num}/{total_pages}</h1>
                {subtitle_html}
            </div>
            {nav_html}
        </div>
        <div class="stats-row">
            <div class="stats-group">
                <span class="stats-label">Page:</span>
                <div class="stat positive">Yes: <span id="localYes">0</span></div>
                <div class="stat negative">No: <span id="localNo">0</span></div>
            </div>
            <div class="stats-group">
                <span class="stats-label">Total:</span>
                <div class="stat positive">Yes: <span id="globalYes">0</span></div>
                <div class="stat negative">No: <span id="globalNo">0</span></div>
            </div>
            <button class="btn btn-export" onclick="exportAnnotations()">Export</button>
            <button class="btn" onclick="importAnnotations()">Import</button>
            <button class="btn" id="toggleContourBtn" onclick="toggleContours()" style="background:#2a5a2a;min-width:90px">Contours</button>
            <button class="btn" id="toggleChRBtn" onclick="toggleChannel('r')" style="background:#8b2222;min-width:70px">{ch_r_label}</button>
            <button class="btn" id="toggleChGBtn" onclick="toggleChannel('g')" style="background:#228b22;min-width:70px">{ch_g_label}</button>
            <button class="btn" id="toggleChBBtn" onclick="toggleChannel('b')" style="background:#22228b;min-width:70px">{ch_b_label}</button>
            <button class="btn" onclick="clearPage()">Clear Page</button>
            <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
            {channel_legend_html}
        </div>
    </div>

    <!-- SVG filters for channel toggling (all 7 combinations of 3 channels off) -->
    <svg style="display:none">
        <filter id="no-r"><feColorMatrix type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1 0"/></filter>
        <filter id="no-g"><feColorMatrix type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0"/></filter>
        <filter id="no-b"><feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
        <filter id="no-rg"><feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0"/></filter>
        <filter id="no-rb"><feColorMatrix type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
        <filter id="no-gb"><feColorMatrix type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
        <filter id="no-rgb"><feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
    </svg>

    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>

    <div class="keyboard-hint">
        Keyboard: Y=Yes, N=No, U=Unsure, Arrow keys=Navigate
    </div>

    <div class="footer">
        {nav_html}
    </div>

    {'<script src="preload_annotations.js"></script>' if include_preload_script else ''}
    <script>{get_js(cell_type, total_pages, experiment_name, page_num)}</script>
</body>
</html>"""

    return html


def generate_index_page(
    cell_type,
    total_samples,
    total_pages,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix="page",
    experiment_name=None,
    file_name=None,
    pixel_size_um=None,
    tiles_processed=None,
    tiles_total=None,
    tissue_tiles=None,
    timestamp=None,
):
    """
    Generate the index/landing page.

    Args:
        cell_type: Type identifier
        total_samples: Total number of samples
        total_pages: Total number of pages
        title: Page title
        subtitle: Optional subtitle
        extra_stats: Dict of additional stats to display
        page_prefix: Prefix for page filenames
        experiment_name: Optional experiment name for localStorage isolation
        file_name: Source file name
        pixel_size_um: Pixel size in micrometers
        tiles_processed: Number of tiles processed (sampled)
        tiles_total: Total number of tiles
        tissue_tiles: Number of tissue-containing tiles
        timestamp: Segmentation timestamp string

    Returns:
        HTML string
    """
    if title is None:
        title = f"{cell_type.upper()} Annotation Review"
    title = _esc(title)

    # Build info lines
    info_lines = []
    info_lines.append(f"Detection type: {_esc(cell_type.upper())}")
    if file_name:
        info_lines.append(f"File: {_esc(file_name)}")
    if pixel_size_um:
        info_lines.append(f"Pixel size: {pixel_size_um:.4f} &micro;m/px")
    if tiles_processed is not None:
        # Calculate percentage based on tissue tiles if available, else total tiles
        denominator = tissue_tiles if tissue_tiles else tiles_total
        if denominator:
            pct = 100.0 * tiles_processed / denominator
            label = "Tissue tiles processed" if tissue_tiles else "Tiles processed"
            info_lines.append(f"{label}: {tiles_processed:,} / {denominator:,} ({pct:.1f}%)")
    info_lines.append(f"Total detections: {total_samples:,}")
    info_lines.append(f"Pages: {total_pages}")
    if timestamp:
        info_lines.append(f"Segmentation: {_esc(timestamp)}")

    info_html = "<br>".join(info_lines)

    extra_stats_html = ""
    if extra_stats:
        for label, value in extra_stats.items():
            extra_stats_html += f"""
            <div class="stat">
                <span>{_esc(label)}</span>
                <span class="number">{_esc(value)}</span>
            </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; text-align: center; }}

        .header {{
            background: #111;
            padding: 30px;
            border: 1px solid #333;
            margin-bottom: 20px;
        }}

        h1 {{
            font-size: 1.5em;
            font-weight: normal;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: #888;
            margin-bottom: 20px;
        }}

        .info-block {{
            color: #aaa;
            line-height: 1.8;
            margin: 20px 0;
            font-size: 1.1em;
        }}

        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 25px 0;
            flex-wrap: wrap;
        }}

        .stat {{
            padding: 15px 30px;
            background: #1a1a1a;
            border: 1px solid #333;
        }}

        .stat .number {{
            display: block;
            font-size: 2em;
            margin-top: 10px;
            color: #4a4;
        }}

        .section {{
            margin: 30px 0;
            text-align: center;
        }}

        .btn {{
            padding: 15px 40px;
            background: #1a1a1a;
            border: 1px solid #4a4;
            color: #4a4;
            cursor: pointer;
            font-family: monospace;
            font-size: 1.1em;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
        }}

        .btn:hover {{
            background: #0f130f;
        }}

        .btn-export {{
            border-color: #44a;
            color: #44a;
        }}

        .btn-export:hover {{
            background: #0f0f13;
        }}

        .btn-danger {{
            border-color: #a44;
            color: #a44;
        }}

        .btn-danger:hover {{
            background: #311;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="info-block">
            {info_html}
        </div>
        {f'<div class="stats">{extra_stats_html}</div>' if extra_stats_html else ''}
    </div>

    <div class="section">
        <a href="{page_prefix}_1.html" class="btn">Start Review</a>
    </div>

    <div class="section">
        <button class="btn btn-export" onclick="exportAnnotations()">Export Annotations</button>
        <button class="btn" onclick="importAnnotations()">Import Annotations</button>
        <button class="btn" id="toggleContourBtn" onclick="toggleContours()" style="background:#2a5a2a">Contours: ON</button>
        <button class="btn" id="togglePMBtn" onclick="toggleChannel('red')" style="background:#8b2222">PM: ON</button>
        <button class="btn" id="toggleNucBtn" onclick="toggleChannel('green')" style="background:#228b22">Nuc: ON</button>
        <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
    </div>

    <!-- SVG filters for channel toggling -->
    <svg style="display:none">
        <filter id="no-r"><feColorMatrix type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1 0"/></filter>
        <filter id="no-g"><feColorMatrix type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0"/></filter>
        <filter id="no-b"><feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
        <filter id="no-rg"><feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0"/></filter>
        <filter id="no-rb"><feColorMatrix type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
        <filter id="no-gb"><feColorMatrix type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
        <filter id="no-rgb"><feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"/></filter>
    </svg>

    <script>
        let contoursVisible = true;
        function toggleContours() {{
            contoursVisible = !contoursVisible;
            const btn = document.getElementById('toggleContourBtn');
            document.querySelectorAll('.img-contour').forEach(img => {{
                img.style.display = contoursVisible ? '' : 'none';
            }});
            if (btn) {{
                btn.style.background = contoursVisible ? '#2a5a2a' : '#555';
                btn.style.opacity = contoursVisible ? '1' : '0.5';
            }}
        }}
        const chState = {{ r: true, g: true, b: true }};
        const chColors = {{ r: '#8b2222', g: '#228b22', b: '#22228b' }};
        const chBtnIds = {{ r: 'toggleChRBtn', g: 'toggleChGBtn', b: 'toggleChBBtn' }};
        function toggleChannel(ch) {{
            chState[ch] = !chState[ch];
            const btn = document.getElementById(chBtnIds[ch]);
            if (btn) {{
                btn.style.background = chState[ch] ? chColors[ch] : '#555';
                btn.style.opacity = chState[ch] ? '1' : '0.5';
            }}
            let off = '';
            if (!chState.r) off += 'r';
            if (!chState.g) off += 'g';
            if (!chState.b) off += 'b';
            const filterVal = off ? 'url(#no-' + off + ')' : 'none';
            document.querySelectorAll('.img-base').forEach(img => {{
                img.style.filter = filterVal;
            }});
        }}
        const CELL_TYPE = '{_esc(cell_type)}';
        const EXPERIMENT_NAME = '{_esc(experiment_name or "")}';
        const STORAGE_KEY = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations' : CELL_TYPE + '_annotations';
        const PAGE_KEY_PREFIX = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_labels_page' : CELL_TYPE + '_labels_page';

        function exportAnnotations() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            const labels = stored ? JSON.parse(stored) : {{}};

            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations.json' : CELL_TYPE + '_annotations.json';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function importAnnotations() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = (e) => {{
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = (ev) => {{
                    try {{
                        const data = JSON.parse(ev.target.result);
                        let imported = {{}};
                        if (data.annotations) {{
                            for (const [uid, val] of Object.entries(data.annotations)) {{
                                if (val === 'yes' || val === 1) imported[uid] = 1;
                                else if (val === 'no' || val === 0) imported[uid] = 0;
                                else if (val === 'unsure' || val === 2) imported[uid] = 2;
                            }}
                        }} else {{
                            (data.positive || []).forEach(uid => imported[uid] = 1);
                            (data.negative || []).forEach(uid => imported[uid] = 0);
                            (data.unsure || []).forEach(uid => imported[uid] = 2);
                        }}
                        let existing = {{}};
                        try {{
                            const gs = localStorage.getItem(STORAGE_KEY);
                            if (gs) existing = JSON.parse(gs);
                        }} catch(ex) {{}}
                        const merged = {{...imported, ...existing}};
                        localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
                        const n = Object.keys(imported).length;
                        alert('Imported ' + n + ' annotations (' + Object.keys(merged).length + ' total after merge)');
                    }} catch(err) {{
                        alert('Error importing: ' + err.message);
                    }}
                }};
                reader.readAsText(file);
            }};
            input.click();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            localStorage.setItem(STORAGE_KEY, JSON.stringify({{}}));
            for (let i = 0; i < localStorage.length; i++) {{
                const key = localStorage.key(i);
                if (key && key.startsWith(PAGE_KEY_PREFIX)) {{
                    localStorage.setItem(key, JSON.stringify({{}}));
                }}
            }}
            alert('All annotations cleared. Refresh any open pages to see the change.');
        }}
    </script>
</body>
</html>"""

    return html


def generate_dual_index_page(
    cell_types: dict,
    title: str = None,
    subtitle: str = None,
    experiment_name: str = None,
    file_name: str = None,
    pixel_size_um: float = None,
    tiles_processed: int = None,
    tiles_total: int = None,
    tissue_tiles: int = None,
    timestamp: str = None,
):
    """
    Generate an index page for multiple cell types (e.g., MK + HSPC batch runs).

    Args:
        cell_types: Dict mapping cell type to info dict, e.g.,
            {
                'mk': {'total_samples': 1234, 'total_pages': 5, 'page_prefix': 'mk_page'},
                'hspc': {'total_samples': 567, 'total_pages': 2, 'page_prefix': 'hspc_page'},
            }
        title: Page title (default: "Multi-Cell Annotation Review")
        subtitle: Optional subtitle (e.g., "16 slides (FGC1, FGC2, ...)")
        experiment_name: Optional experiment name for localStorage
        file_name: Source file/slide name(s)
        pixel_size_um: Pixel size in micrometers
        tiles_processed: Number of tiles processed
        tiles_total: Total number of tiles
        tissue_tiles: Number of tissue tiles
        timestamp: Segmentation timestamp

    Returns:
        HTML string
    """
    if title is None:
        types_str = " + ".join(ct.upper() for ct in cell_types.keys())
        title = f"{types_str} Annotation Review"
    title = _esc(title)

    # Build info lines
    info_lines = []
    if file_name:
        info_lines.append(f"Source: {_esc(file_name)}")
    if pixel_size_um:
        info_lines.append(f"Pixel size: {pixel_size_um:.4f} &micro;m/px")
    if tiles_processed is not None:
        denominator = tissue_tiles if tissue_tiles else tiles_total
        if denominator:
            pct = 100.0 * tiles_processed / denominator
            label = "Tissue tiles processed" if tissue_tiles else "Tiles processed"
            info_lines.append(f"{label}: {tiles_processed:,} / {denominator:,} ({pct:.1f}%)")
    if timestamp:
        info_lines.append(f"Segmentation: {_esc(timestamp)}")

    info_html = "<br>".join(info_lines) if info_lines else ""
    subtitle_html = f'<div class="subtitle">{_esc(subtitle)}</div>' if subtitle else ""

    # Build cell type sections
    sections_html = ""
    for ct, info in cell_types.items():
        ct_safe = _esc(ct)
        total_samples = info.get("total_samples", 0)
        total_pages = info.get("total_pages", 0)
        page_prefix = info.get("page_prefix", f"{ct}_page")

        sections_html += f"""
        <div class="cell-type-section">
            <h2>{ct_safe.upper()}</h2>
            <div class="stats">
                <div class="stat">
                    <span>Detections</span>
                    <span class="number">{total_samples:,}</span>
                </div>
                <div class="stat">
                    <span>Pages</span>
                    <span class="number">{total_pages}</span>
                </div>
            </div>
            <a href="{page_prefix}_1.html" class="btn">Review {ct_safe.upper()}</a>
        </div>
        """

    # Build export buttons for each cell type
    export_buttons = ""
    for ct in cell_types.keys():
        ct_safe = _esc(ct)
        export_buttons += f"""
        <button class="btn btn-export" onclick="exportAnnotations('{ct_safe}')">Export {ct_safe.upper()}</button>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; }}

        .header {{
            background: #111;
            padding: 30px;
            border: 1px solid #333;
            margin-bottom: 20px;
            text-align: center;
        }}

        h1 {{ font-size: 1.5em; font-weight: normal; margin-bottom: 10px; }}
        h2 {{ font-size: 1.2em; font-weight: normal; color: #4a4; margin-bottom: 15px; }}

        .subtitle {{ color: #888; margin-bottom: 20px; }}
        .info-block {{ color: #aaa; line-height: 1.8; margin: 20px 0; font-size: 1.1em; }}

        .cell-types-container {{
            display: flex;
            justify-content: center;
            gap: 40px;
            flex-wrap: wrap;
            margin: 30px 0;
        }}

        .cell-type-section {{
            background: #111;
            border: 1px solid #333;
            padding: 25px 40px;
            text-align: center;
            min-width: 280px;
        }}

        .stats {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }}

        .stat {{
            padding: 10px 20px;
            background: #1a1a1a;
            border: 1px solid #333;
            text-align: center;
        }}

        .stat .number {{
            display: block;
            font-size: 1.8em;
            margin-top: 8px;
            color: #4a4;
        }}

        .btn {{
            padding: 12px 30px;
            background: #1a1a1a;
            border: 1px solid #4a4;
            color: #4a4;
            cursor: pointer;
            font-family: monospace;
            font-size: 1em;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
        }}

        .btn:hover {{ background: #0f130f; }}
        .btn-export {{ border-color: #44a; color: #44a; }}
        .btn-export:hover {{ background: #0f0f13; }}
        .btn-danger {{ border-color: #a44; color: #a44; }}
        .btn-danger:hover {{ background: #311; }}

        .actions {{ margin: 30px 0; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        {subtitle_html}
        {f'<div class="info-block">{info_html}</div>' if info_html else ''}
    </div>

    <div class="cell-types-container">
        {sections_html}
    </div>

    <div class="actions">
        {export_buttons}
        <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
    </div>

    <script>
        const EXPERIMENT_NAME = '{_esc(experiment_name or "")}';

        function getStorageKey(cellType) {{
            return EXPERIMENT_NAME ? cellType + '_' + EXPERIMENT_NAME + '_annotations' : cellType + '_annotations';
        }}

        function getPageKeyPrefix(cellType) {{
            return EXPERIMENT_NAME ? cellType + '_' + EXPERIMENT_NAME + '_labels_page' : cellType + '_labels_page';
        }}

        function exportAnnotations(cellType) {{
            const key = getStorageKey(cellType);
            const stored = localStorage.getItem(key);
            const labels = stored ? JSON.parse(stored) : {{}};

            const data = {{
                cell_type: cellType,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = EXPERIMENT_NAME ? cellType + '_' + EXPERIMENT_NAME + '_annotations.json' : cellType + '_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations for ALL cell types? This cannot be undone.')) return;
            const cellTypes = {json.dumps(list(cell_types.keys()))};
            cellTypes.forEach(ct => {{
                localStorage.setItem(getStorageKey(ct), JSON.stringify({{}}));
                // Also clear page-specific keys for this experiment
                const prefix = getPageKeyPrefix(ct);
                for (let i = 0; i < localStorage.length; i++) {{
                    const key = localStorage.key(i);
                    if (key && key.startsWith(prefix)) {{
                        localStorage.setItem(key, JSON.stringify({{}}));
                    }}
                }}
            }});
            alert('All annotations cleared. Refresh any open pages.');
        }}
    </script>
</body>
</html>"""

    return html


def export_samples_to_html(
    samples,
    output_dir,
    cell_type,
    samples_per_page=300,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix="page",
    experiment_name=None,
    file_name=None,
    pixel_size_um=None,
    tiles_processed=None,
    tiles_total=None,
    tissue_tiles=None,
    channel_legend=None,
    timestamp=None,
    prior_annotations=None,
):
    """
    Export samples to paginated HTML files.

    Args:
        samples: List of sample dicts (see generate_annotation_page for format)
        output_dir: Output directory path
        cell_type: Type identifier
        samples_per_page: Number of samples per page
        title: Optional title for index page
        subtitle: Optional subtitle
        extra_stats: Dict of extra stats for index page
        page_prefix: Prefix for page filenames
        experiment_name: Optional experiment name for localStorage isolation
        file_name: Source file name for index page
        pixel_size_um: Pixel size in micrometers
        tiles_processed: Number of tiles processed
        tiles_total: Total number of tiles
        channel_legend: Optional dict mapping colors to channel names,
            e.g., {'red': 'nuc488', 'green': 'Bgtx647', 'blue': 'NfL750'}
        timestamp: Segmentation timestamp string
        prior_annotations: Optional path to prior annotations JSON file.
            If provided, annotations will be pre-loaded into localStorage
            when the HTML pages load. This is useful for round-2 annotation
            after classifier training, so round-1 annotations are visible.

    Returns:
        Tuple of (total_samples, total_pages)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        logger.info(f"No {cell_type} samples to export")
        return 0, 0

    # Generate preload_annotations.js if prior annotations provided
    has_preload = False
    if prior_annotations:
        preload_js = generate_preload_annotations_js(prior_annotations, cell_type, experiment_name)
        if preload_js:
            preload_path = output_dir / "preload_annotations.js"
            with open(preload_path, "w") as f:
                f.write(preload_js)
            has_preload = True
            # Count annotations for logging
            with open(prior_annotations) as f:
                ann_data = json.load(f)
            n_pos = len(ann_data.get("positive", []))
            n_neg = len(ann_data.get("negative", []))
            if "annotations" in ann_data:
                n_pos = sum(1 for v in ann_data["annotations"].values() if v == "yes")
                n_neg = sum(1 for v in ann_data["annotations"].values() if v == "no")
            logger.info(
                f"  Pre-loading {n_pos + n_neg} prior annotations ({n_pos} yes, {n_neg} no)"
            )

    # Paginate
    pages = [samples[i : i + samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    logger.info(f"Generating {total_pages} {cell_type} HTML pages...")

    # Generate pages
    for page_num, page_samples in enumerate(pages, 1):
        html = generate_annotation_page(
            page_samples,
            cell_type,
            page_num,
            total_pages,
            title=title,
            page_prefix=page_prefix,
            experiment_name=experiment_name,
            channel_legend=channel_legend,
            subtitle=subtitle or file_name,  # Use subtitle or fallback to file_name
            include_preload_script=has_preload,
        )

        page_path = output_dir / f"{page_prefix}_{page_num}.html"
        with open(page_path, "w") as f:
            f.write(html)

        file_size = page_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")

    # Generate index
    index_html = generate_index_page(
        cell_type,
        len(samples),
        total_pages,
        title=title,
        subtitle=subtitle,
        extra_stats=extra_stats,
        page_prefix=page_prefix,
        experiment_name=experiment_name,
        file_name=file_name,
        pixel_size_um=pixel_size_um,
        tiles_processed=tiles_processed,
        tiles_total=tiles_total,
        tissue_tiles=tissue_tiles,
        timestamp=timestamp,
    )

    index_path = output_dir / "index.html"
    with open(index_path, "w") as f:
        f.write(index_html)

    logger.info(f"Export complete: {output_dir}")

    return len(samples), total_pages


# =============================================================================
# MK/HSPC BATCH HTML EXPORT (RAM-based)
# =============================================================================
# Backward-compatible shims -- authoritative implementations live in html_generator.py.
# These shims accept the legacy ``logger=None`` parameter for API compatibility
# but delegate to the html_generator versions (which use module-level logging).
# Imports are lazy (inside each function) to avoid circular imports, since
# html_generator.py imports draw_mask_contour / get_largest_connected_component /
# percentile_normalize from this module.


def load_samples_from_ram(*args, logger=None, **kwargs):
    """Backward-compatible shim -- delegates to html_generator."""
    from xldvp_seg.io.html_generator import load_samples_from_ram as _impl

    return _impl(*args, **kwargs)


def create_mk_hspc_index(*args, **kwargs):
    """Backward-compatible shim -- delegates to html_generator."""
    from xldvp_seg.io.html_generator import create_mk_hspc_index as _impl

    return _impl(*args, **kwargs)


def generate_mk_hspc_page_html(*args, **kwargs):
    """Backward-compatible shim -- delegates to html_generator."""
    from xldvp_seg.io.html_generator import generate_mk_hspc_page_html as _impl

    return _impl(*args, **kwargs)


def generate_mk_hspc_pages(*args, logger=None, **kwargs):
    """Backward-compatible shim -- delegates to html_generator."""
    from xldvp_seg.io.html_generator import generate_mk_hspc_pages as _impl

    return _impl(*args, **kwargs)


def export_mk_hspc_html_from_ram(*args, logger=None, **kwargs):
    """Backward-compatible shim -- delegates to html_generator."""
    from xldvp_seg.io.html_generator import export_mk_hspc_html_from_ram as _impl

    return _impl(*args, **kwargs)


# =============================================================================
# VESSEL ANNOTATION HTML EXPORT (RF Training Support)
# =============================================================================
# Enhanced annotation interface specifically for vessel cross-sections with:
# - Batch annotation (select multiple, bulk annotate)
# - Feature filtering (diameter range, confidence, etc.)
# - RF-ready export (CSV with features, JSON for scikit-learn)
# - Live annotation statistics


def generate_vessel_annotation_page(
    samples,
    cell_type,
    page_num,
    total_pages,
    title=None,
    page_prefix="page",
    experiment_name=None,
    subtitle=None,
):
    """
    Generate an HTML annotation page optimized for vessel annotation with RF training support.

    Args:
        samples: List of sample dicts with keys:
            - uid: Unique identifier
            - image: Base64 encoded image string
            - features: Dict of all extracted features for filtering/export
        cell_type: Type identifier (e.g., 'vessel')
        page_num: Current page number
        total_pages: Total number of pages
        title: Optional title override
        page_prefix: Prefix for page filenames
        experiment_name: Optional experiment name for localStorage isolation
        subtitle: Optional subtitle shown below title

    Returns:
        HTML string
    """
    import json

    if title is None:
        title = cell_type.upper()
    title = _esc(title)

    # Build navigation
    nav_html = '<div class="nav-buttons">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{page_prefix}_{page_num-1}.html" class="nav-btn">Prev</a>'
    nav_html += f'<span class="page-info">Page {page_num} / {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{page_prefix}_{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += "</div>"

    # Build subtitle HTML if provided
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="header-subtitle">{_esc(subtitle)}</div>'

    # Collect all features for this page (for filtering/export)
    all_features = {}
    for sample in samples:
        uid_raw = sample["uid"]
        feat = sample.get("features", {})
        # Filter to numeric features only
        all_features[uid_raw] = {
            k: v for k, v in feat.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

    all_features_json = json.dumps(all_features)

    # Build cards
    cards_html = ""
    for idx, sample in enumerate(samples):
        uid = _esc(sample["uid"])
        img_b64 = sample["image"]
        img_raw_b64 = sample.get("image_raw")  # Raw image without contours
        mime = sample.get("mime_type", "jpeg")
        if mime not in ("jpeg", "png"):
            mime = "jpeg"  # Safe default
        feat = sample.get("features", {})

        # Format stats line with vessel-specific features
        stats_parts = []
        if "outer_diameter_um" in feat:
            stats_parts.append(f"D={feat['outer_diameter_um']:.1f}&micro;m")
        if "wall_thickness_mean_um" in feat:
            stats_parts.append(f"wall={feat['wall_thickness_mean_um']:.1f}&micro;m")
        if "circularity" in feat:
            stats_parts.append(f"circ={feat['circularity']:.2f}")
        if "confidence" in feat:
            conf = feat["confidence"]
            if isinstance(conf, str):
                stats_parts.append(_esc(conf))
            else:
                stats_parts.append(f"{conf*100:.0f}%")

        stats_str = " | ".join(stats_parts) if stats_parts else ""

        # Additional feature line
        feat_parts = []
        if "aspect_ratio" in feat:
            feat_parts.append(f"AR={feat['aspect_ratio']:.2f}")
        if "ring_completeness" in feat:
            feat_parts.append(f"ring={feat['ring_completeness']:.2f}")
        if "lumen_area_um2" in feat:
            feat_parts.append(f"lumen={feat['lumen_area_um2']:.0f}&micro;m&sup2;")
        feat_str = " | ".join(feat_parts) if feat_parts else ""

        # Build image container - side-by-side if raw image available
        if img_raw_b64:
            img_container = f"""
            <div class="card-img-container card-img-sidebyside" onclick="selectCard({idx})">
                <div class="img-half">
                    <img src="data:image/{mime};base64,{img_raw_b64}" alt="{uid} raw">
                    <div class="img-label">Raw</div>
                </div>
                <div class="img-half">
                    <img src="data:image/{mime};base64,{img_b64}" alt="{uid} contours">
                    <div class="img-label">Contours</div>
                </div>
            </div>"""
        else:
            img_container = f"""
            <div class="card-img-container" onclick="selectCard({idx})">
                <img src="data:image/{mime};base64,{img_b64}" alt="{uid}">
            </div>"""

        cards_html += f"""
        <div class="card" id="{uid}" data-label="-1"
             data-diameter="{feat.get('outer_diameter_um', 0)}"
             data-confidence="{feat.get('confidence', 'unknown')}">
            <input type="checkbox" class="card-checkbox" onclick="event.stopPropagation(); toggleBatchSelect('{uid}')">
            {img_container}
            <div class="card-info">
                <div class="card-meta">
                    <div class="card-id">{uid}</div>
                    <div class="card-stats">{stats_str}</div>
                    <div class="card-features">{feat_str}</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{uid}', 1)">Y</button>
                    <button class="btn btn-unsure" onclick="setLabel('{uid}', 2)">?</button>
                    <button class="btn btn-no" onclick="setLabel('{uid}', 0)">N</button>
                </div>
            </div>
        </div>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title} - Page {page_num}/{total_pages}</title>
    <style>{get_vessel_css()}</style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <div>
                <h1>{title} - Page {page_num}/{total_pages}</h1>
                {subtitle_html}
            </div>
            {nav_html}
        </div>
        <div class="stats-row">
            <div class="stats-group">
                <span class="stats-label">Page:</span>
                <div class="stat positive">Yes: <span id="localYes">0</span></div>
                <div class="stat negative">No: <span id="localNo">0</span></div>
                <div class="stat remaining">Remaining: <span id="localRemaining">0</span></div>
                <div class="stat">Total: <span id="localTotal">0</span></div>
            </div>
            <div class="stats-group">
                <span class="stats-label">Global:</span>
                <div class="stat positive">Yes: <span id="globalYes">0</span></div>
                <div class="stat negative">No: <span id="globalNo">0</span></div>
                <div class="stat">Total: <span id="globalTotal">0</span></div>
            </div>
        </div>
    </div>

    <!-- Filter Panel -->
    <div class="filter-panel">
        <div class="filter-group">
            <label>Diameter (&micro;m):</label>
            <input type="number" id="filterMinDiam" placeholder="Min" step="1">
            <span>-</span>
            <input type="number" id="filterMaxDiam" placeholder="Max" step="1">
        </div>
        <div class="filter-group">
            <label>Confidence:</label>
            <select id="filterConfidence">
                <option value="all">All</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
            </select>
        </div>
        <div class="filter-group">
            <label><input type="checkbox" id="filterShowAnnotated" checked> Annotated</label>
        </div>
        <div class="filter-group">
            <label><input type="checkbox" id="filterShowUnannotated" checked> Unannotated</label>
        </div>
        <button class="filter-btn" onclick="applyFilters()">Apply Filters</button>
        <button class="filter-btn" onclick="resetFilters()">Reset</button>
    </div>

    <!-- Batch Selection Toolbar -->
    <div class="batch-toolbar hidden" id="batchToolbar">
        <span class="batch-count" id="batchCount">0 selected</span>
        <button class="batch-btn batch-btn-yes" onclick="batchLabel(1)">Label All Yes</button>
        <button class="batch-btn batch-btn-no" onclick="batchLabel(0)">Label All No</button>
        <button class="batch-btn" onclick="clearBatchSelection()">Clear Selection</button>
        <button class="batch-btn" onclick="selectAllVisible()">Select All Visible</button>
        <button class="batch-btn" onclick="selectUnannotated()">Select Unannotated</button>
    </div>

    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>

    <div class="keyboard-hint">
        Keyboard: Y=Yes, N=No, U=Unsure, Arrow keys=Navigate, Space=Toggle batch select
    </div>

    <!-- Export Panel -->
    <div class="export-panel">
        <button class="btn btn-export" onclick="previewExport()">Preview Export</button>
        <button class="btn btn-export" onclick="exportAnnotationsJSON()">Export JSON</button>
        <button class="btn btn-export" onclick="exportForRF()">Export CSV (RF)</button>
        <button class="btn btn-export" onclick="exportRFJSON()">Export sklearn JSON</button>
        <button class="btn" onclick="clearPage()">Clear Page</button>
        <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
    </div>

    <div class="footer">
        {nav_html}
    </div>

    <!-- Export Preview Modal -->
    <div class="modal" id="exportModal" onclick="if(event.target===this)closeModal()">
        <div class="modal-content">
            <div class="modal-header">
                <h3>Export Preview</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <pre id="exportPreview"></pre>
            <div style="margin-top: 15px; text-align: center;">
                <button class="btn btn-export" onclick="exportAnnotationsJSON()">Download JSON</button>
                <button class="btn btn-export" onclick="exportForRF()">Download CSV</button>
                <button class="btn btn-export" onclick="exportRFJSON()">Download sklearn JSON</button>
            </div>
        </div>
    </div>

    <script>{get_vessel_js(cell_type, total_pages, experiment_name, all_features_json, page_num)}</script>
</body>
</html>"""

    return html


def generate_vessel_index_page(
    cell_type,
    total_samples,
    total_pages,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix="page",
    experiment_name=None,
    file_name=None,
    pixel_size_um=None,
    timestamp=None,
    feature_summary=None,
):
    """
    Generate the index/landing page for vessel annotation.

    Args:
        cell_type: Type identifier
        total_samples: Total number of samples
        total_pages: Total number of pages
        title: Page title
        subtitle: Optional subtitle
        extra_stats: Dict of additional stats to display
        page_prefix: Prefix for page filenames
        experiment_name: Optional experiment name for localStorage isolation
        file_name: Source file name
        pixel_size_um: Pixel size in micrometers
        timestamp: Segmentation timestamp string
        feature_summary: Dict with feature statistics for display

    Returns:
        HTML string
    """
    if title is None:
        title = f"{cell_type.upper()} Annotation Review"
    title = _esc(title)

    # Build info lines
    info_lines = []
    info_lines.append(f"Detection type: {_esc(cell_type.upper())}")
    if file_name:
        info_lines.append(f"File: {_esc(file_name)}")
    if pixel_size_um:
        info_lines.append(f"Pixel size: {pixel_size_um:.4f} &micro;m/px")
    info_lines.append(f"Total detections: {total_samples:,}")
    info_lines.append(f"Pages: {total_pages}")
    if timestamp:
        info_lines.append(f"Segmentation: {_esc(timestamp)}")

    info_html = "<br>".join(info_lines)

    # Feature summary section
    feature_html = ""
    if feature_summary:
        feature_html = '<div class="feature-summary"><h3>Feature Summary</h3><table>'
        for key, stats in feature_summary.items():
            feature_html += f"""
            <tr>
                <td>{_esc(key)}</td>
                <td>min: {stats.get("min", 0):.2f}</td>
                <td>max: {stats.get("max", 0):.2f}</td>
                <td>mean: {stats.get("mean", 0):.2f}</td>
            </tr>"""
        feature_html += "</table></div>"

    extra_stats_html = ""
    if extra_stats:
        for label, value in extra_stats.items():
            extra_stats_html += f"""
            <div class="stat">
                <span>{_esc(label)}</span>
                <span class="number">{_esc(value)}</span>
            </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; }}

        .header {{
            background: #111;
            padding: 30px;
            border: 1px solid #333;
            margin-bottom: 20px;
            text-align: center;
        }}

        h1 {{
            font-size: 1.5em;
            font-weight: normal;
            margin-bottom: 10px;
        }}

        h3 {{
            font-size: 1.1em;
            font-weight: normal;
            margin: 20px 0 10px 0;
            color: #888;
        }}

        .subtitle {{
            color: #888;
            margin-bottom: 20px;
        }}

        .info-block {{
            color: #aaa;
            line-height: 1.8;
            margin: 20px 0;
            font-size: 1.1em;
        }}

        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 25px 0;
            flex-wrap: wrap;
        }}

        .stat {{
            padding: 15px 30px;
            background: #1a1a1a;
            border: 1px solid #333;
            text-align: center;
        }}

        .stat .number {{
            display: block;
            font-size: 2em;
            margin-top: 10px;
            color: #4a4;
        }}

        .feature-summary {{
            margin: 20px auto;
            max-width: 600px;
            text-align: left;
        }}

        .feature-summary table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .feature-summary td {{
            padding: 8px;
            border-bottom: 1px solid #333;
            font-size: 0.9em;
        }}

        .section {{
            margin: 30px 0;
            text-align: center;
        }}

        .btn {{
            padding: 15px 40px;
            background: #1a1a1a;
            border: 1px solid #4a4;
            color: #4a4;
            cursor: pointer;
            font-family: monospace;
            font-size: 1.1em;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
        }}

        .btn:hover {{
            background: #0f130f;
        }}

        .btn-export {{
            border-color: #44a;
            color: #44a;
        }}

        .btn-export:hover {{
            background: #0f0f13;
        }}

        .btn-danger {{
            border-color: #a44;
            color: #a44;
        }}

        .btn-danger:hover {{
            background: #311;
        }}

        .annotation-stats {{
            margin: 20px 0;
            padding: 20px;
            background: #111;
            border: 1px solid #333;
        }}

        .annotation-stats h3 {{
            margin-bottom: 15px;
        }}

        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #1a1a1a;
            border: 1px solid #333;
            margin: 10px 0;
        }}

        .progress-fill {{
            height: 100%;
            background: #4a4;
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="info-block">
            {info_html}
        </div>
        {f'<div class="stats">{extra_stats_html}</div>' if extra_stats_html else ''}
        {feature_html}
    </div>

    <div class="annotation-stats" id="annotationStats">
        <h3>Annotation Progress</h3>
        <div class="stats">
            <div class="stat" style="border-left: 3px solid #4a4;">
                <span>Positive</span>
                <span class="number" id="posCount">0</span>
            </div>
            <div class="stat" style="border-left: 3px solid #a44;">
                <span>Negative</span>
                <span class="number" id="negCount">0</span>
            </div>
            <div class="stat" style="border-left: 3px solid #aa4;">
                <span>Remaining</span>
                <span class="number" id="remainingCount">{total_samples}</span>
            </div>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
        </div>
        <div style="text-align: center; color: #888; margin-top: 10px;">
            <span id="progressPct">0%</span> complete
        </div>
    </div>

    <div class="section">
        <a href="{page_prefix}_1.html" class="btn">Start Review</a>
    </div>

    <div class="section">
        <button class="btn btn-export" onclick="exportAnnotationsJSON()">Export JSON</button>
        <button class="btn btn-export" onclick="exportForRF()">Export CSV (RF Training)</button>
        <button class="btn btn-export" onclick="exportRFJSON()">Export sklearn JSON</button>
        <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
    </div>

    <script>
        const CELL_TYPE = '{_esc(cell_type)}';
        const EXPERIMENT_NAME = '{_esc(experiment_name or "")}';
        const TOTAL_SAMPLES = {total_samples};
        const STORAGE_KEY = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations' : CELL_TYPE + '_annotations';
        const PAGE_KEY_PREFIX = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_labels_page' : CELL_TYPE + '_labels_page';

        function updateProgress() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            const labels = stored ? JSON.parse(stored) : {{}};

            let pos = 0, neg = 0;
            for (const label of Object.values(labels)) {{
                if (label === 1) pos++;
                else if (label === 0) neg++;
            }}

            const total = pos + neg;
            const remaining = TOTAL_SAMPLES - total;
            const pct = TOTAL_SAMPLES > 0 ? (total / TOTAL_SAMPLES * 100).toFixed(1) : 0;

            document.getElementById('posCount').textContent = pos;
            document.getElementById('negCount').textContent = neg;
            document.getElementById('remainingCount').textContent = remaining;
            document.getElementById('progressFill').style.width = pct + '%';
            document.getElementById('progressPct').textContent = pct + '%';
        }}

        function exportAnnotationsJSON() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            const labels = stored ? JSON.parse(stored) : {{}};

            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations.json' : CELL_TYPE + '_annotations.json';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportForRF() {{
            alert('CSV export with features is only available from annotation pages where feature data is loaded.');
        }}

        function exportRFJSON() {{
            alert('sklearn JSON export with features is only available from annotation pages where feature data is loaded.');
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            localStorage.setItem(STORAGE_KEY, JSON.stringify({{}}));
            // Also clear page-specific keys for this experiment
            for (let i = 0; i < localStorage.length; i++) {{
                const key = localStorage.key(i);
                if (key && key.startsWith(PAGE_KEY_PREFIX)) {{
                    localStorage.setItem(key, JSON.stringify({{}}));
                }}
            }}
            updateProgress();
            alert('All annotations cleared. Refresh any open pages to see the change.');
        }}

        // Initialize
        updateProgress();
        // Auto-refresh progress every 5 seconds
        setInterval(updateProgress, 5000);
    </script>
</body>
</html>"""

    return html


def export_vessel_samples_to_html(
    samples,
    output_dir,
    cell_type="vessel",
    samples_per_page=200,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix="page",
    experiment_name=None,
    file_name=None,
    pixel_size_um=None,
    timestamp=None,
):
    """
    Export vessel samples to paginated HTML files with RF training support.

    Args:
        samples: List of sample dicts with:
            - uid: Unique identifier
            - image: Base64 encoded image string
            - features: Dict of all extracted features
        output_dir: Output directory path
        cell_type: Type identifier (default 'vessel')
        samples_per_page: Number of samples per page (default 200 for faster loading)
        title: Optional title for index page
        subtitle: Optional subtitle
        extra_stats: Dict of extra stats for index page
        page_prefix: Prefix for page filenames
        experiment_name: Optional experiment name for localStorage isolation
        file_name: Source file name for index page
        pixel_size_um: Pixel size in micrometers
        timestamp: Segmentation timestamp string

    Returns:
        Tuple of (total_samples, total_pages)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        logger.info(f"No {cell_type} samples to export")
        return 0, 0

    # Calculate feature summary for index page
    feature_summary = {}
    key_features = ["outer_diameter_um", "wall_thickness_mean_um", "circularity", "aspect_ratio"]
    for feat_key in key_features:
        values = [s["features"].get(feat_key) for s in samples if feat_key in s.get("features", {})]
        if values:
            values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if values:
                feature_summary[feat_key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }

    # Paginate
    pages = [samples[i : i + samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    logger.info(f"Generating {total_pages} {cell_type} HTML pages...")

    # Generate pages
    for page_num, page_samples in enumerate(pages, 1):
        html = generate_vessel_annotation_page(
            page_samples,
            cell_type,
            page_num,
            total_pages,
            title=title,
            page_prefix=page_prefix,
            experiment_name=experiment_name,
            subtitle=subtitle or file_name,
        )

        page_path = output_dir / f"{page_prefix}_{page_num}.html"
        with open(page_path, "w") as f:
            f.write(html)

        file_size = page_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")

    # Generate index
    index_html = generate_vessel_index_page(
        cell_type,
        len(samples),
        total_pages,
        title=title,
        subtitle=subtitle,
        extra_stats=extra_stats,
        page_prefix=page_prefix,
        experiment_name=experiment_name,
        file_name=file_name,
        pixel_size_um=pixel_size_um,
        timestamp=timestamp,
        feature_summary=feature_summary,
    )

    index_path = output_dir / "index.html"
    with open(index_path, "w") as f:
        f.write(index_html)

    logger.info(f"Export complete: {output_dir}")

    return len(samples), total_pages
