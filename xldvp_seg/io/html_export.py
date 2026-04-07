"""Core HTML export — page generators, CSS/JS templates, image utilities.

Used by the detection pipeline for per-tile HTML annotation pages. Contains:

**Image utilities:** percentile_normalize, draw_mask_contour, image_to_base64,
    get_largest_connected_component, compose_tile_rgb, create_hdf5_dataset
**CSS/JS generators:** get_css, get_js, get_vessel_css, get_vessel_js
**Page generators:** generate_annotation_page, generate_index_page,
    generate_dual_index_page, export_samples_to_html
**Vessel pages:** generate_vessel_annotation_page, generate_vessel_index_page,
    export_vessel_samples_to_html

MK/HSPC functions (load_samples_from_ram, create_mk_hspc_index, etc.) are
backward-compatible shims delegating to ``html_generator.py``.

See also ``html_generator.py`` for the class-based HTMLPageGenerator API.
"""

import base64
import html as html_mod
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def _esc(value) -> str:
    """Escape a value for safe insertion into HTML/JS strings.

    Prevents XSS by escaping <, >, &, ", and ' characters.
    """
    return html_mod.escape(str(value), quote=True)


# Try to use LZ4 compression (faster than gzip), fallback to gzip
try:
    import hdf5plugin

    # LZ4 is ~3-5x faster than gzip with similar compression ratio for image masks
    HDF5_COMPRESSION_KWARGS = hdf5plugin.LZ4(nbytes=0)  # Returns dict-like for **unpacking
    HDF5_COMPRESSION_NAME = "LZ4"
except ImportError:
    HDF5_COMPRESSION_KWARGS = {"compression": "gzip"}
    HDF5_COMPRESSION_NAME = "gzip"


def create_hdf5_dataset(f, name, data):
    """Create HDF5 dataset with best available compression (LZ4 or gzip)."""
    if isinstance(HDF5_COMPRESSION_KWARGS, dict):
        f.create_dataset(name, data=data, **HDF5_COMPRESSION_KWARGS)
    else:
        # hdf5plugin filter object — pass as compression kwarg
        f.create_dataset(name, data=data, **dict(HDF5_COMPRESSION_KWARGS))


def generate_preload_annotations_js(
    annotations_path: str, cell_type: str, experiment_name: str = None
) -> str:
    """
    Generate JavaScript that pre-loads prior annotations into localStorage.

    This is used when regenerating HTML after classifier training, so that
    the user's round-1 annotations are visible alongside the classifier's
    new predictions.

    Args:
        annotations_path: Path to annotations JSON file (exported from HTML viewer)
        cell_type: Cell type identifier (e.g., 'nmj', 'mk')
        experiment_name: Optional experiment name for localStorage key isolation.
            Must match the experiment_name used in the page JS.

    Returns:
        JavaScript code string to write to preload_annotations.js
    """
    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        return None

    with open(annotations_path) as f:
        data = json.load(f)

    # Convert from export format {positive: [...], negative: [...]}
    # to localStorage format {uid: 1, uid: 0}
    ls_format = {}
    for uid in data.get("positive", []):
        ls_format[uid] = 1
    for uid in data.get("negative", []):
        ls_format[uid] = 0
    for uid in data.get("unsure", []):
        ls_format[uid] = 2

    # Also handle the alternative format {annotations: {uid: "yes", uid: "no"}}
    if "annotations" in data:
        for uid, label in data["annotations"].items():
            if label == "yes":
                ls_format[uid] = 1
            elif label == "no":
                ls_format[uid] = 0
            elif label == "unsure":
                ls_format[uid] = 2

    if not ls_format:
        return None

    if experiment_name:
        global_key = _esc(f"{cell_type}_{experiment_name}_annotations")
        page_key_prefix = _esc(f"{cell_type}_{experiment_name}_labels_page")
    else:
        global_key = _esc(f"{cell_type}_annotations")
        page_key_prefix = _esc(f"{cell_type}_labels_page")

    # Escape </ sequences to prevent </script> injection in inline JS
    safe_json = json.dumps(ls_format).replace("</", r"<\/")
    js_content = f"""// Pre-loaded annotations from {_esc(annotations_path.name)}
// Generated automatically during HTML export
// These are EXISTING annotations - new annotations take precedence

const PRELOADED_ANNOTATIONS = {safe_json};

// Merge: preloaded as base, existing localStorage on top (so new annotations aren't overwritten)
// Write to BOTH the global key and any existing page-specific keys
(function() {{
    try {{
        // Merge into global key
        let existingGlobal = {{}};
        const savedGlobal = localStorage.getItem('{global_key}');
        if (savedGlobal) existingGlobal = JSON.parse(savedGlobal);
        const mergedGlobal = {{...PRELOADED_ANNOTATIONS, ...existingGlobal}};
        localStorage.setItem('{global_key}', JSON.stringify(mergedGlobal));

        // Also merge into any existing page-specific keys for this cell type
        for (let i = 0; i < localStorage.length; i++) {{
            const key = localStorage.key(i);
            if (key && key.startsWith('{page_key_prefix}')) {{
                let existingPage = {{}};
                try {{ existingPage = JSON.parse(localStorage.getItem(key)); }} catch(e2) {{}}
                const mergedPage = {{...PRELOADED_ANNOTATIONS, ...existingPage}};
                localStorage.setItem(key, JSON.stringify(mergedPage));
            }}
        }}

        const preloadedCount = Object.keys(PRELOADED_ANNOTATIONS).length;
        const existingCount = Object.keys(existingGlobal).length;
        const newFromPreload = Object.keys(mergedGlobal).length - existingCount;

        if (newFromPreload > 0) {{
            console.log('Loaded ' + newFromPreload + ' annotations from preload file (' + preloadedCount + ' total in file)');
        }}
    }} catch(e) {{ console.error('Failed to load annotations:', e); }}
}})();
"""
    return js_content


def get_largest_connected_component(mask):
    """Extract only the largest connected component from a binary mask."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    # Find largest component
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def percentile_normalize(image, p_low=1, p_high=99.5, global_percentiles=None):
    """
    Normalize image using percentiles.

    Percentiles are computed on non-zero pixels only (CZI padding is exactly 0).
    Zero pixels stay black after normalization.

    Args:
        image: 2D or 3D numpy array
        p_low: Lower percentile for normalization (default 1)
        p_high: Upper percentile for normalization (default 99.5)
        global_percentiles: Optional dict {channel_index: (low_val, high_val)}.
            When provided and the channel index exists in the dict, use these
            precomputed percentile values instead of computing from the crop.
            Only applies to multi-channel (3D) images.

    Returns:
        uint8 normalized image
    """
    if image.ndim == 2:
        # Percentile on non-zero pixels only (exclude CZI padding)
        nonzero = image[image > 0]
        if len(nonzero) == 0:
            return np.zeros_like(image, dtype=np.uint8)
        low_val = np.percentile(nonzero, p_low)
        high_val = np.percentile(nonzero, p_high)
        if high_val > low_val:
            normalized = (image.astype(np.float32) - low_val) / (high_val - low_val) * 255
            result = np.clip(normalized, 0, 255).astype(np.uint8)
            result[image == 0] = 0  # Keep padding black
            return result
        if image.dtype == np.uint16:
            return (image / 256).astype(np.uint8)
        from xldvp_seg.utils.detection_utils import safe_to_uint8

        return safe_to_uint8(image)
    else:
        # Multi-channel: valid pixel = any channel > 0
        h, w, c = image.shape
        valid_mask = np.max(image, axis=2) > 0
        result = np.zeros_like(image, dtype=np.uint8)
        for ch in range(c):
            ch_data = image[:, :, ch]
            valid_pixels = ch_data[valid_mask]
            if len(valid_pixels) == 0:
                continue
            if global_percentiles is not None and ch in global_percentiles:
                low_val, high_val = global_percentiles[ch]
            else:
                low_val = np.percentile(valid_pixels, p_low)
                high_val = np.percentile(valid_pixels, p_high)
            if high_val > low_val:
                normalized = (ch_data.astype(np.float32) - low_val) / (high_val - low_val) * 255
                result[:, :, ch] = np.clip(normalized, 0, 255).astype(np.uint8)
            else:
                if image.dtype == np.uint16:
                    result[:, :, ch] = (ch_data / 256).astype(np.uint8)
                else:
                    from xldvp_seg.utils.detection_utils import safe_to_uint8

                    result[:, :, ch] = safe_to_uint8(ch_data)
        # Keep padding pixels black
        result[~valid_mask] = 0
        return result


def draw_mask_contour(
    img_array, mask, color=(0, 255, 0), thickness=2, dotted=False, use_cv2=True, bw_dashed=False
):
    """
    Draw mask contour on image.

    Args:
        img_array: RGB image array (or grayscale, will be converted)
        mask: Binary mask
        color: RGB tuple for contour color
        thickness: Contour thickness in pixels
        dotted: Whether to use dotted line
        use_cv2: Use OpenCV for faster, smoother contours (default True)
        bw_dashed: Draw alternating black/white dashed contour (overrides color)

    Returns:
        Image with contour drawn (always RGB)
    """
    import cv2

    # Ensure RGB
    if img_array.ndim == 2:
        img_out = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_out = img_array.copy()

    if bw_dashed:
        # Thin green dashed contour line
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        dash_len, gap_len = 8, 5
        line_thickness = thickness
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            cycle = dash_len + gap_len
            n_pts = len(pts)
            i = 0
            while i < n_pts:
                j = min(i + dash_len, n_pts - 1)
                if j > i:
                    cv2.line(img_out, tuple(pts[i]), tuple(pts[j]), (0, 255, 0), line_thickness)
                i += cycle
        return img_out

    if use_cv2 and not dotted:
        # Use cv2.drawContours for smooth, thick lines
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Convert RGB to BGR for cv2, then back
        cv2.drawContours(img_out, contours, -1, color, thickness)
        return img_out

    # Fallback to dilation method
    dilated = ndimage.binary_dilation(mask, iterations=thickness)
    contour = dilated & ~mask
    ys, xs = np.where(contour)

    if len(ys) == 0:
        return img_out

    if dotted:
        # Subsample every 3rd pixel for dotted effect
        dot_mask = np.zeros(len(ys), dtype=bool)
        dot_mask[::3] = True
        ys_draw, xs_draw = ys[dot_mask], xs[dot_mask]
    else:
        ys_draw, xs_draw = ys, xs
    valid = (
        (ys_draw >= 0)
        & (ys_draw < img_out.shape[0])
        & (xs_draw >= 0)
        & (xs_draw < img_out.shape[1])
    )
    img_out[ys_draw[valid], xs_draw[valid]] = color

    return img_out


def image_to_base64(img_array, format="JPEG", quality=85):
    """
    Convert numpy array or PIL image to base64 string.

    Args:
        img_array: numpy array or PIL Image
        format: Image format ('JPEG' or 'PNG')
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded string
    """
    if isinstance(img_array, np.ndarray):
        pil_img = Image.fromarray(img_array)
    else:
        pil_img = img_array

    # Use JPEG for opaque images (smaller, faster). PNG only for transparency.
    has_alpha = pil_img.mode in ("RGBA", "LA", "PA")
    buffer = BytesIO()
    if has_alpha:
        pil_img.save(buffer, format="PNG", optimize=True)
        mime_type = "png"
    else:
        out_format = format.upper() if format else "JPEG"
        if pil_img.mode != "RGB" and out_format == "JPEG":
            pil_img = pil_img.convert("RGB")
        pil_img.save(buffer, format=out_format, quality=quality)
        mime_type = out_format.lower()

    return base64.b64encode(buffer.getvalue()).decode("utf-8"), mime_type


def get_css():
    """Get the unified CSS styles."""
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: monospace; background: #0a0a0a; color: #ddd; }

        .header {
            background: #111;
            padding: 12px 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .header h1 {
            font-size: 1.2em;
            font-weight: normal;
        }

        .header-subtitle {
            font-size: 0.85em;
            color: #888;
            margin-top: 2px;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .nav-btn {
            padding: 8px 15px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ddd;
            text-decoration: none;
            cursor: pointer;
            font-family: monospace;
        }

        .nav-btn:hover {
            background: #222;
        }

        .page-info {
            padding: 8px 15px;
            color: #888;
        }

        .stats-row {
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            flex-wrap: wrap;
            align-items: center;
        }

        .stats-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .stats-label {
            color: #888;
            font-size: 0.9em;
        }

        .stat {
            padding: 4px 10px;
            background: #1a1a1a;
            border: 1px solid #333;
        }

        .stat.positive {
            border-left: 3px solid #4a4;
        }

        .stat.negative {
            border-left: 3px solid #a44;
        }

        .channel-legend {
            margin-left: auto;
            padding: 4px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
        }

        .channel-legend span {
            margin: 0 6px;
            font-weight: bold;
        }

        .ch-red { color: #ff6666; }
        .ch-green { color: #66ff66; }
        .ch-blue { color: #6666ff; }

        .content {
            padding: 15px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }

        .card {
            background: #111;
            border: 2px solid #333;
            overflow: hidden;
            transition: border-color 0.2s;
        }

        .card.selected {
            box-shadow: 0 0 0 3px #fff;
        }

        .card.labeled-yes {
            border-color: #4a4 !important;
            background: #0f130f !important;
        }

        .card.labeled-no {
            border-color: #a44 !important;
            background: #130f0f !important;
        }

        .card.labeled-unsure {
            border-color: #aa4 !important;
            background: #13130f !important;
        }

        .card-img-container {
            width: 100%;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }

        .card img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .card-info {
            padding: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #333;
            gap: 10px;
        }

        .card-meta {
            flex: 1;
            min-width: 0;
        }

        .card-id {
            font-size: 0.75em;
            color: #888;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-stats {
            font-size: 0.8em;
            color: #aaa;
            margin-top: 3px;
        }

        .buttons {
            display: flex;
            gap: 5px;
            flex-shrink: 0;
        }

        .btn {
            padding: 6px 12px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #ddd;
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }

        .btn:hover {
            background: #222;
        }

        .btn-yes {
            border-color: #4a4;
            color: #4a4;
        }

        .btn-no {
            border-color: #a44;
            color: #a44;
        }

        .btn-unsure {
            border-color: #aa4;
            color: #aa4;
        }

        .btn-export {
            border-color: #44a;
            color: #44a;
        }

        .btn-danger {
            border-color: #a44;
            color: #a44;
        }

        .btn-danger:hover {
            background: #311;
        }

        .keyboard-hint {
            text-align: center;
            padding: 15px;
            color: #555;
            font-size: 0.85em;
            border-top: 1px solid #222;
        }

        .footer {
            background: #111;
            padding: 15px;
            border-top: 1px solid #333;
            display: flex;
            justify-content: center;
            gap: 10px;
        }
    """


def get_js(cell_type, total_pages, experiment_name=None, page_num=1):
    """
    Get the unified JavaScript for annotation handling.

    Uses BOTH a page-specific localStorage key and a global key so that
    annotations are visible from both single-page and cross-page views.

    Args:
        cell_type: Type identifier (e.g., 'nmj', 'mk', 'hspc')
        total_pages: Total number of pages
        experiment_name: Optional experiment name for localStorage key isolation
                        If provided, global key is '{cell_type}_{experiment_name}_annotations'
                        Otherwise, global key is '{cell_type}_annotations'
        page_num: Current page number (1-indexed) for page-specific key

    Returns:
        JavaScript code string
    """
    # Build storage keys: both global and page-specific
    if experiment_name:
        global_key = _esc(f"{cell_type}_{experiment_name}_annotations")
        page_key = _esc(f"{cell_type}_{experiment_name}_labels_page{page_num}")
    else:
        global_key = _esc(f"{cell_type}_annotations")
        page_key = _esc(f"{cell_type}_labels_page{page_num}")
    cell_type_safe = _esc(cell_type)
    experiment_name_safe = _esc(experiment_name or "")
    if experiment_name:
        page_key_prefix = _esc(f"{cell_type}_{experiment_name}_labels_page")
    else:
        page_key_prefix = _esc(f"{cell_type}_labels_page")

    return f"""
        const CELL_TYPE = '{cell_type_safe}';
        const EXPERIMENT_NAME = '{experiment_name_safe}';
        const TOTAL_PAGES = {total_pages};
        const GLOBAL_STORAGE_KEY = '{global_key}';
        const PAGE_STORAGE_KEY = '{page_key}';
        const PAGE_KEY_PREFIX = '{page_key_prefix}';

        let labels = {{}};
        let selectedIdx = -1;
        const cards = document.querySelectorAll('.card');

        // Save to BOTH page-specific and global localStorage keys
        function saveLabels() {{
            localStorage.setItem(PAGE_STORAGE_KEY, JSON.stringify(labels));
            // Merge into global store instead of overwriting
            let globalLabels = {{}};
            try {{ globalLabels = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(e) {{}}
            Object.assign(globalLabels, labels);
            // Remove toggled-off annotations: delete page UIDs not in labels
            cards.forEach(card => {{ if (!(card.id in labels)) delete globalLabels[card.id]; }});
            localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(globalLabels));
        }}

        // Load from localStorage: page-specific first, then global fallback
        function loadAnnotations() {{
            try {{
                let saved = localStorage.getItem(PAGE_STORAGE_KEY);
                if (!saved) {{
                    // Fallback: load only UIDs present on this page from global
                    const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                    if (globalSaved) {{
                        const globalLabels = JSON.parse(globalSaved);
                        const pageUids = new Set(Array.from(cards).map(c => c.id));
                        for (const [uid, label] of Object.entries(globalLabels)) {{
                            if (pageUids.has(uid)) labels[uid] = label;
                        }}
                    }}
                }} else {{
                    labels = JSON.parse(saved);
                }}
            }} catch(e) {{ console.error(e); }}

            // Apply to cards
            cards.forEach((card, i) => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    applyLabelToCard(card, labels[uid]);
                }}
            }});

            updateStats();
        }}

        function applyLabelToCard(card, label) {{
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            card.dataset.label = label;
            if (label === 1) card.classList.add('labeled-yes');
            else if (label === 0) card.classList.add('labeled-no');
            else if (label === 2) card.classList.add('labeled-unsure');
        }}

        function setLabel(uid, label, autoAdvance = false) {{
            // Toggle off if same label
            if (labels[uid] === label) {{
                delete labels[uid];
                const card = document.getElementById(uid);
                if (card) {{
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                    card.dataset.label = -1;
                }}
            }} else {{
                labels[uid] = label;
                const card = document.getElementById(uid);
                if (card) applyLabelToCard(card, label);
            }}

            saveLabels();
            updateStats();

            if (autoAdvance && selectedIdx >= 0 && selectedIdx < cards.length - 1) {{
                selectCard(selectedIdx + 1);
            }}
        }}

        function updateStats() {{
            let localYes = 0, localNo = 0, localUnsure = 0;
            let globalYes = 0, globalNo = 0;

            // Count current page
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] === 1) localYes++;
                else if (labels[uid] === 0) localNo++;
                else if (labels[uid] === 2) localUnsure++;
            }});

            // Count global from the global key (contains all annotations)
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) {{
                    const globalLabels = JSON.parse(globalSaved);
                    for (const v of Object.values(globalLabels)) {{
                        if (v === 1) globalYes++;
                        else if (v === 0) globalNo++;
                    }}
                }}
            }} catch(e) {{ console.error(e); }}

            const localYesEl = document.getElementById('localYes');
            const localNoEl = document.getElementById('localNo');
            const globalYesEl = document.getElementById('globalYes');
            const globalNoEl = document.getElementById('globalNo');

            if (localYesEl) localYesEl.textContent = localYes;
            if (localNoEl) localNoEl.textContent = localNo;
            if (globalYesEl) globalYesEl.textContent = globalYes;
            if (globalNoEl) globalNoEl.textContent = globalNo;
        }}

        function selectCard(idx) {{
            cards.forEach(c => c.classList.remove('selected'));
            if (idx >= 0 && idx < cards.length) {{
                selectedIdx = idx;
                cards[idx].classList.add('selected');
                cards[idx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        function exportAnnotations() {{
            // Export from global key to get all annotations across pages
            let allLabels = {{}};
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) allLabels = JSON.parse(globalSaved);
            }} catch(e) {{ console.error(e); }}

            const data = {{
                cell_type: CELL_TYPE,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: [],
                unsure: []
            }};

            for (const [uid, label] of Object.entries(allLabels)) {{
                if (label === 1) data.positive.push(uid);
                else if (label === 0) data.negative.push(uid);
                else if (label === 2) data.unsure.push(uid);
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations.json' : CELL_TYPE + '_annotations.json';
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
                            const gs = localStorage.getItem(GLOBAL_STORAGE_KEY);
                            if (gs) existing = JSON.parse(gs);
                        }} catch(ex) {{}}
                        const merged = {{...imported, ...existing}};
                        localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(merged));
                        for (const [uid, val] of Object.entries(imported)) {{
                            if (labels[uid] === undefined) labels[uid] = val;
                        }}
                        saveLabels();
                        cards.forEach(card => {{
                            const uid = card.id;
                            if (labels[uid] !== undefined) {{
                                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                                if (labels[uid] === 1) card.classList.add('labeled-yes');
                                else if (labels[uid] === 0) card.classList.add('labeled-no');
                                else if (labels[uid] === 2) card.classList.add('labeled-unsure');
                                card.dataset.label = labels[uid];
                            }}
                        }});
                        updateStats();
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

        function clearPage() {{
            if (!confirm('Clear annotations on this page?')) return;
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    delete labels[uid];
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                    card.dataset.label = -1;
                }}
            }});
            saveLabels();
            updateStats();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            labels = {{}};
            saveLabels();
            for (let i = 1; i <= TOTAL_PAGES; i++) {{
                localStorage.setItem(PAGE_KEY_PREFIX + i, JSON.stringify({{}}));
            }}
            cards.forEach(card => {{
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                card.dataset.label = -1;
            }});
            updateStats();
            alert('All annotations cleared.');
        }}

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
            // Build filter ID from off-channels
            let off = '';
            if (!chState.r) off += 'r';
            if (!chState.g) off += 'g';
            if (!chState.b) off += 'b';
            const filterVal = off ? 'url(#no-' + off + ')' : 'none';
            document.querySelectorAll('.img-base').forEach(img => {{
                img.style.filter = filterVal;
            }});
        }}

        document.addEventListener('keydown', (e) => {{
            // Navigation
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                e.preventDefault();
                selectCard(Math.min(selectedIdx + 1, cards.length - 1));
            }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                e.preventDefault();
                selectCard(Math.max(selectedIdx - 1, 0));
            }}
            // Labeling
            else if (selectedIdx >= 0) {{
                const uid = cards[selectedIdx].id;
                if (e.key.toLowerCase() === 'y') setLabel(uid, 1, true);
                else if (e.key.toLowerCase() === 'n') setLabel(uid, 0, true);
                else if (e.key.toLowerCase() === 'u') setLabel(uid, 2, true);
            }}
        }});

        // Initialize
        loadAnnotations();
    """


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
# TILE RGB COMPOSITION (shared by pipeline resume + regenerate_html.py)
# =============================================================================


def compose_tile_rgb(
    channel_arrays,
    tile_x,
    tile_y,
    tile_size,
    display_channels,
    x_start,
    y_start,
    mosaic_h,
    mosaic_w,
):
    """Extract a tile region and compose RGB from display channels.

    Args:
        channel_arrays: List of 2D arrays indexed by channel number (None for missing).
        tile_x, tile_y: Tile origin in mosaic (global) coordinates.
        tile_size: Tile dimension in pixels.
        display_channels: List of channel indices for [R, G, B].
        x_start, y_start: Mosaic origin offset.
        mosaic_h, mosaic_w: Mosaic array dimensions.

    Returns:
        (h, w, 3) uint8 array with per-channel percentile normalization,
        or None if tile is entirely outside bounds.
    """
    import numpy as np

    # Convert to array coordinates (subtract mosaic origin)
    ay = tile_y - y_start
    ax = tile_x - x_start

    # Clamp to array bounds
    ay_end = min(ay + tile_size, mosaic_h)
    ax_end = min(ax + tile_size, mosaic_w)
    ay = max(0, ay)
    ax = max(0, ax)

    if ay_end <= ay or ax_end <= ax:
        return None

    h = ay_end - ay
    w = ax_end - ax
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i, ch_idx in enumerate(display_channels[:3]):
        if ch_idx < len(channel_arrays) and channel_arrays[ch_idx] is not None:
            ch_data = channel_arrays[ch_idx][ay:ay_end, ax:ax_end]
            # Percentile normalize to uint8 (non-zero pixels only)
            valid = ch_data > 0
            if valid.any():
                p1 = np.percentile(ch_data[valid], 1)
                p99 = np.percentile(ch_data[valid], 99.5)
                if p99 > p1:
                    norm = np.clip(
                        (ch_data.astype(np.float32) - p1) / (p99 - p1) * 255,
                        0,
                        255,
                    ).astype(np.uint8)
                    norm[~valid] = 0
                    rgb[:, :, i] = norm
    return rgb


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


def get_vessel_css():
    """Get enhanced CSS styles for vessel annotation interface."""
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: monospace; background: #0a0a0a; color: #ddd; }

        .header {
            background: #111;
            padding: 12px 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            border-bottom: 1px solid #333;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .header h1 {
            font-size: 1.2em;
            font-weight: normal;
        }

        .header-subtitle {
            font-size: 0.85em;
            color: #888;
            margin-top: 2px;
        }

        .nav-buttons {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .nav-btn {
            padding: 8px 15px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ddd;
            text-decoration: none;
            cursor: pointer;
            font-family: monospace;
        }

        .nav-btn:hover {
            background: #222;
        }

        .page-info {
            padding: 8px 15px;
            color: #888;
        }

        /* Filter panel */
        .filter-panel {
            background: #0d0d0d;
            padding: 12px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }

        .filter-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .filter-group label {
            color: #888;
            font-size: 0.85em;
        }

        .filter-group input[type="number"],
        .filter-group select {
            padding: 5px 8px;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ddd;
            font-family: monospace;
            width: 80px;
        }

        .filter-group input[type="checkbox"] {
            width: 16px;
            height: 16px;
        }

        .filter-btn {
            padding: 6px 12px;
            background: #1a1a1a;
            border: 1px solid #44a;
            color: #44a;
            cursor: pointer;
            font-family: monospace;
        }

        .filter-btn:hover {
            background: #0f0f13;
        }

        /* Stats row */
        .stats-row {
            display: flex;
            gap: 20px;
            font-size: 0.85em;
            flex-wrap: wrap;
            align-items: center;
            padding: 8px 0;
        }

        .stats-group {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .stats-label {
            color: #888;
            font-size: 0.9em;
        }

        .stat {
            padding: 4px 10px;
            background: #1a1a1a;
            border: 1px solid #333;
        }

        .stat.positive {
            border-left: 3px solid #4a4;
        }

        .stat.negative {
            border-left: 3px solid #a44;
        }

        .stat.remaining {
            border-left: 3px solid #aa4;
        }

        /* Batch selection toolbar */
        .batch-toolbar {
            background: #111;
            padding: 10px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .batch-toolbar.hidden {
            display: none;
        }

        .batch-btn {
            padding: 6px 12px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #ddd;
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }

        .batch-btn:hover {
            background: #222;
        }

        .batch-btn-yes {
            border-color: #4a4;
            color: #4a4;
        }

        .batch-btn-no {
            border-color: #a44;
            color: #a44;
        }

        .batch-count {
            color: #888;
        }

        /* Content area */
        .content {
            padding: 15px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }

        /* Card styles */
        .card {
            background: #111;
            border: 2px solid #333;
            overflow: hidden;
            transition: border-color 0.2s, transform 0.1s;
            position: relative;
        }

        .card.selected {
            box-shadow: 0 0 0 3px #fff;
        }

        .card.batch-selected {
            box-shadow: 0 0 0 3px #44a;
        }

        .card.labeled-yes {
            border-color: #4a4 !important;
            background: #0f130f !important;
        }

        .card.labeled-no {
            border-color: #a44 !important;
            background: #130f0f !important;
        }

        .card.labeled-unsure {
            border-color: #aa4 !important;
            background: #13130f !important;
        }

        .card.hidden {
            display: none;
        }

        .card-checkbox {
            position: absolute;
            top: 8px;
            left: 8px;
            width: 20px;
            height: 20px;
            z-index: 10;
        }

        .card-img-container {
            width: 100%;
            height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
            cursor: pointer;
        }

        .card img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Side-by-side image layout for raw + contours */
        .card-img-sidebyside {
            display: flex;
            flex-direction: row;
            gap: 2px;
        }

        .img-half {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
        }

        .img-half img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .img-label {
            position: absolute;
            bottom: 4px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.7);
            color: #aaa;
            font-size: 0.65em;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .card-info {
            padding: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #333;
            gap: 10px;
        }

        .card-meta {
            flex: 1;
            min-width: 0;
        }

        .card-id {
            font-size: 0.75em;
            color: #888;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .card-stats {
            font-size: 0.8em;
            color: #aaa;
            margin-top: 3px;
        }

        .card-features {
            font-size: 0.7em;
            color: #666;
            margin-top: 2px;
        }

        .buttons {
            display: flex;
            gap: 5px;
            flex-shrink: 0;
        }

        .btn {
            padding: 6px 12px;
            border: 1px solid #333;
            background: #1a1a1a;
            color: #ddd;
            cursor: pointer;
            font-family: monospace;
            font-size: 0.85em;
        }

        .btn:hover {
            background: #222;
        }

        .btn-yes {
            border-color: #4a4;
            color: #4a4;
        }

        .btn-no {
            border-color: #a44;
            color: #a44;
        }

        .btn-unsure {
            border-color: #aa4;
            color: #aa4;
        }

        .btn-export {
            border-color: #44a;
            color: #44a;
        }

        .btn-danger {
            border-color: #a44;
            color: #a44;
        }

        .btn-danger:hover {
            background: #311;
        }

        /* Export panel */
        .export-panel {
            background: #111;
            padding: 15px 20px;
            border-top: 1px solid #333;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            justify-content: center;
        }

        .keyboard-hint {
            text-align: center;
            padding: 15px;
            color: #555;
            font-size: 0.85em;
            border-top: 1px solid #222;
        }

        .footer {
            background: #111;
            padding: 15px;
            border-top: 1px solid #333;
            display: flex;
            justify-content: center;
            gap: 10px;
        }

        /* Modal for export preview */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background: #111;
            border: 1px solid #333;
            padding: 20px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .modal-close {
            background: none;
            border: none;
            color: #888;
            font-size: 1.5em;
            cursor: pointer;
        }

        .modal-close:hover {
            color: #fff;
        }

        pre {
            background: #0a0a0a;
            padding: 15px;
            overflow: auto;
            max-height: 400px;
            font-size: 0.85em;
        }
    """


def get_vessel_js(cell_type, total_pages, experiment_name=None, all_features_json="{}", page_num=1):
    """
    Get enhanced JavaScript for vessel annotation with RF training support.

    Args:
        cell_type: Type identifier (e.g., 'vessel')
        total_pages: Total number of pages
        experiment_name: Optional experiment name for localStorage key isolation
        all_features_json: JSON string of all vessel features for filtering/export

    Returns:
        JavaScript code string
    """
    # Build storage keys: both global and page-specific
    if experiment_name:
        global_key = _esc(f"{cell_type}_{experiment_name}_annotations")
        page_key = _esc(f"{cell_type}_{experiment_name}_labels_page{page_num}")
    else:
        global_key = _esc(f"{cell_type}_annotations")
        page_key = _esc(f"{cell_type}_labels_page{page_num}")
    cell_type_safe = _esc(cell_type)
    experiment_name_safe = _esc(experiment_name or "")
    if experiment_name:
        page_key_prefix = _esc(f"{cell_type}_{experiment_name}_labels_page")
    else:
        page_key_prefix = _esc(f"{cell_type}_labels_page")

    return f"""
        const CELL_TYPE = '{cell_type_safe}';
        const EXPERIMENT_NAME = '{experiment_name_safe}';
        const TOTAL_PAGES = {total_pages};
        const GLOBAL_STORAGE_KEY = '{global_key}';
        const PAGE_STORAGE_KEY = '{page_key}';
        const PAGE_KEY_PREFIX = '{page_key_prefix}';
        const ALL_FEATURES = {all_features_json};

        let labels = {{}};
        let selectedIdx = -1;
        let batchSelected = new Set();
        const cards = document.querySelectorAll('.card');

        // Save to BOTH page-specific and global localStorage keys
        function saveLabels() {{
            localStorage.setItem(PAGE_STORAGE_KEY, JSON.stringify(labels));
            // Merge into global store instead of overwriting
            let globalLabels = {{}};
            try {{ globalLabels = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(e) {{}}
            Object.assign(globalLabels, labels);
            // Remove toggled-off annotations: delete page UIDs not in labels
            cards.forEach(card => {{ if (!(card.id in labels)) delete globalLabels[card.id]; }});
            localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(globalLabels));
        }}

        // Load from localStorage: page-specific first, then global fallback
        function loadAnnotations() {{
            try {{
                let saved = localStorage.getItem(PAGE_STORAGE_KEY);
                if (!saved) {{
                    // Fallback: load only UIDs present on this page from global
                    const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                    if (globalSaved) {{
                        const globalLabels = JSON.parse(globalSaved);
                        const pageUids = new Set(Array.from(cards).map(c => c.id));
                        for (const [uid, label] of Object.entries(globalLabels)) {{
                            if (pageUids.has(uid)) labels[uid] = label;
                        }}
                    }}
                }} else {{
                    labels = JSON.parse(saved);
                }}
            }} catch(e) {{ console.error(e); }}

            // Apply to cards
            cards.forEach((card, i) => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    applyLabelToCard(card, labels[uid]);
                }}
            }});

            updateStats();
        }}

        function applyLabelToCard(card, label) {{
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            card.dataset.label = label;
            if (label === 1) card.classList.add('labeled-yes');
            else if (label === 0) card.classList.add('labeled-no');
            else if (label === 2) card.classList.add('labeled-unsure');
        }}

        function setLabel(uid, label, autoAdvance = false) {{
            // Toggle off if same label
            if (labels[uid] === label) {{
                delete labels[uid];
                const card = document.getElementById(uid);
                if (card) {{
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                    card.dataset.label = -1;
                }}
            }} else {{
                labels[uid] = label;
                const card = document.getElementById(uid);
                if (card) applyLabelToCard(card, label);
            }}

            saveLabels();
            updateStats();

            if (autoAdvance && selectedIdx >= 0) {{
                // Find next visible card
                const visibleCards = Array.from(cards).filter(c => !c.classList.contains('hidden'));
                const currentVisibleIdx = visibleCards.findIndex(c => c === cards[selectedIdx]);
                if (currentVisibleIdx >= 0 && currentVisibleIdx < visibleCards.length - 1) {{
                    const nextCard = visibleCards[currentVisibleIdx + 1];
                    const nextIdx = Array.from(cards).indexOf(nextCard);
                    selectCard(nextIdx);
                }}
            }}
        }}

        function updateStats() {{
            let localYes = 0, localNo = 0, localUnsure = 0, localTotal = 0;
            let globalYes = 0, globalNo = 0, globalTotal = 0;

            // Count current page (visible only)
            cards.forEach(card => {{
                if (!card.classList.contains('hidden')) {{
                    localTotal++;
                    const uid = card.id;
                    if (labels[uid] === 1) localYes++;
                    else if (labels[uid] === 0) localNo++;
                    else if (labels[uid] === 2) localUnsure++;
                }}
            }});

            // Count global from the global key
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) {{
                    const globalLabels = JSON.parse(globalSaved);
                    for (const v of Object.values(globalLabels)) {{
                        if (v === 1) globalYes++;
                        else if (v === 0) globalNo++;
                        globalTotal++;
                    }}
                }}
            }} catch(e) {{ console.error(e); }}

            const localRemaining = localTotal - localYes - localNo - localUnsure;

            document.getElementById('localYes').textContent = localYes;
            document.getElementById('localNo').textContent = localNo;
            document.getElementById('localRemaining').textContent = localRemaining;
            document.getElementById('localTotal').textContent = localTotal;
            document.getElementById('globalYes').textContent = globalYes;
            document.getElementById('globalNo').textContent = globalNo;
            document.getElementById('globalTotal').textContent = globalTotal;
        }}

        function selectCard(idx) {{
            cards.forEach(c => c.classList.remove('selected'));
            if (idx >= 0 && idx < cards.length) {{
                selectedIdx = idx;
                cards[idx].classList.add('selected');
                cards[idx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        // Batch selection functions
        function toggleBatchSelect(uid) {{
            const card = document.getElementById(uid);
            if (batchSelected.has(uid)) {{
                batchSelected.delete(uid);
                card.classList.remove('batch-selected');
            }} else {{
                batchSelected.add(uid);
                card.classList.add('batch-selected');
            }}
            updateBatchToolbar();
        }}

        function updateBatchToolbar() {{
            const toolbar = document.getElementById('batchToolbar');
            const countEl = document.getElementById('batchCount');
            if (batchSelected.size > 0) {{
                toolbar.classList.remove('hidden');
                countEl.textContent = batchSelected.size + ' selected';
            }} else {{
                toolbar.classList.add('hidden');
            }}
        }}

        function batchLabel(label) {{
            batchSelected.forEach(uid => {{
                labels[uid] = label;
                const card = document.getElementById(uid);
                if (card) applyLabelToCard(card, label);
            }});
            saveLabels();
            clearBatchSelection();
            updateStats();
        }}

        function clearBatchSelection() {{
            batchSelected.forEach(uid => {{
                const card = document.getElementById(uid);
                if (card) card.classList.remove('batch-selected');
            }});
            batchSelected.clear();
            updateBatchToolbar();
        }}

        function selectAllVisible() {{
            cards.forEach(card => {{
                if (!card.classList.contains('hidden')) {{
                    batchSelected.add(card.id);
                    card.classList.add('batch-selected');
                }}
            }});
            updateBatchToolbar();
        }}

        function selectUnannotated() {{
            cards.forEach(card => {{
                if (!card.classList.contains('hidden') && labels[card.id] === undefined) {{
                    batchSelected.add(card.id);
                    card.classList.add('batch-selected');
                }}
            }});
            updateBatchToolbar();
        }}

        // Filtering functions
        function applyFilters() {{
            const minDiam = parseFloat(document.getElementById('filterMinDiam').value) || 0;
            const maxDiam = parseFloat(document.getElementById('filterMaxDiam').value) || 9999;
            const confidence = document.getElementById('filterConfidence').value;
            const showAnnotated = document.getElementById('filterShowAnnotated').checked;
            const showUnannotated = document.getElementById('filterShowUnannotated').checked;

            let visibleCount = 0;
            cards.forEach(card => {{
                const uid = card.id;
                const feat = ALL_FEATURES[uid] || {{}};
                const diam = feat.outer_diameter_um || 0;
                const conf = feat.confidence || 'unknown';
                const isAnnotated = labels[uid] !== undefined;

                let visible = true;

                // Diameter filter
                if (diam < minDiam || diam > maxDiam) visible = false;

                // Confidence filter
                if (confidence !== 'all' && conf !== confidence) visible = false;

                // Annotation status filter
                if (isAnnotated && !showAnnotated) visible = false;
                if (!isAnnotated && !showUnannotated) visible = false;

                if (visible) {{
                    card.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    card.classList.add('hidden');
                }}
            }});

            updateStats();
            clearBatchSelection();
        }}

        function resetFilters() {{
            document.getElementById('filterMinDiam').value = '';
            document.getElementById('filterMaxDiam').value = '';
            document.getElementById('filterConfidence').value = 'all';
            document.getElementById('filterShowAnnotated').checked = true;
            document.getElementById('filterShowUnannotated').checked = true;

            cards.forEach(card => card.classList.remove('hidden'));
            updateStats();
        }}

        // Export functions
        function exportAnnotationsJSON() {{
            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                total_annotations: Object.keys(labels).length,
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
            // Export annotations with features for Random Forest training
            const rows = [];
            const featureKeys = Object.keys(Object.values(ALL_FEATURES)[0] || {{}}).filter(k =>
                typeof (Object.values(ALL_FEATURES)[0] || {{}})[k] === 'number'
            );

            // Header
            const header = ['uid', 'annotation'].concat(featureKeys);
            rows.push(header.join(','));

            // Data rows
            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1 || label === 0) {{  // Only yes/no, skip unsure
                    const feat = ALL_FEATURES[uid] || {{}};
                    const row = [uid, label === 1 ? 'yes' : 'no'];
                    featureKeys.forEach(k => {{
                        const val = feat[k];
                        row.push(typeof val === 'number' ? val : '');
                    }});
                    rows.push(row.join(','));
                }}
            }}

            const csv = rows.join('\\n');
            const blob = new Blob([csv], {{type: 'text/csv'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_rf_training.csv' : CELL_TYPE + '_rf_training.csv';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function exportRFJSON() {{
            // Export in scikit-learn compatible format
            const featureKeys = Object.keys(Object.values(ALL_FEATURES)[0] || {{}}).filter(k =>
                typeof (Object.values(ALL_FEATURES)[0] || {{}})[k] === 'number'
            );

            const X = [];
            const y = [];
            const uids = [];

            for (const [uid, label] of Object.entries(labels)) {{
                if (label === 1 || label === 0) {{
                    const feat = ALL_FEATURES[uid] || {{}};
                    const row = featureKeys.map(k => feat[k] || 0);
                    X.push(row);
                    y.push(label);
                    uids.push(uid);
                }}
            }}

            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                feature_names: featureKeys,
                X: X,
                y: y,
                uids: uids,
                n_positive: y.filter(v => v === 1).length,
                n_negative: y.filter(v => v === 0).length
            }};

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const filename = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_rf_sklearn.json' : CELL_TYPE + '_rf_sklearn.json';
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }}

        function previewExport() {{
            const modal = document.getElementById('exportModal');
            const preview = document.getElementById('exportPreview');

            const featureKeys = Object.keys(Object.values(ALL_FEATURES)[0] || {{}}).filter(k =>
                typeof (Object.values(ALL_FEATURES)[0] || {{}})[k] === 'number'
            );

            let posCount = 0, negCount = 0;
            for (const label of Object.values(labels)) {{
                if (label === 1) posCount++;
                else if (label === 0) negCount++;
            }}

            const summary = `Export Summary
==============
Cell Type: ${{CELL_TYPE}}
Experiment: ${{EXPERIMENT_NAME || 'N/A'}}

Annotations:
  - Positive (yes): ${{posCount}}
  - Negative (no): ${{negCount}}
  - Total for training: ${{posCount + negCount}}

Features: ${{featureKeys.length}} numeric features
${{featureKeys.slice(0, 20).join('\\n')}}
${{featureKeys.length > 20 ? '... and ' + (featureKeys.length - 20) + ' more' : ''}}

Export Formats:
  1. JSON (simple) - positive/negative/unsure lists
  2. CSV (RF) - uid, annotation, all features
  3. JSON (sklearn) - X matrix, y vector, feature names`;

            preview.textContent = summary;
            modal.classList.add('show');
        }}

        function closeModal() {{
            document.getElementById('exportModal').classList.remove('show');
        }}

        function clearPage() {{
            if (!confirm('Clear annotations on this page?')) return;
            cards.forEach(card => {{
                const uid = card.id;
                if (labels[uid] !== undefined) {{
                    delete labels[uid];
                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                    card.dataset.label = -1;
                }}
            }});
            saveLabels();
            updateStats();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            labels = {{}};
            saveLabels();
            // Also clear all other page-specific keys for this experiment
            for (let i = 1; i <= TOTAL_PAGES; i++) {{
                localStorage.setItem(PAGE_KEY_PREFIX + i, JSON.stringify({{}}));
            }}
            cards.forEach(card => {{
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                card.dataset.label = -1;
            }});
            updateStats();
            alert('All annotations cleared.');
        }}

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            // Navigation
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                e.preventDefault();
                const visibleCards = Array.from(cards).filter(c => !c.classList.contains('hidden'));
                const currentVisibleIdx = visibleCards.findIndex(c => c === cards[selectedIdx]);
                if (currentVisibleIdx < visibleCards.length - 1) {{
                    const nextCard = visibleCards[currentVisibleIdx + 1];
                    selectCard(Array.from(cards).indexOf(nextCard));
                }}
            }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                e.preventDefault();
                const visibleCards = Array.from(cards).filter(c => !c.classList.contains('hidden'));
                const currentVisibleIdx = visibleCards.findIndex(c => c === cards[selectedIdx]);
                if (currentVisibleIdx > 0) {{
                    const prevCard = visibleCards[currentVisibleIdx - 1];
                    selectCard(Array.from(cards).indexOf(prevCard));
                }}
            }}
            // Labeling
            else if (selectedIdx >= 0) {{
                const uid = cards[selectedIdx].id;
                if (e.key.toLowerCase() === 'y') setLabel(uid, 1, true);
                else if (e.key.toLowerCase() === 'n') setLabel(uid, 0, true);
                else if (e.key.toLowerCase() === 'u') setLabel(uid, 2, true);
                else if (e.key === ' ') {{
                    e.preventDefault();
                    toggleBatchSelect(uid);
                }}
            }}
            // Escape to close modal or clear selection
            if (e.key === 'Escape') {{
                closeModal();
                clearBatchSelection();
            }}
        }});

        // Initialize
        loadAnnotations();
    """


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
