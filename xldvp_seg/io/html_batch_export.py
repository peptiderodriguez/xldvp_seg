"""MK/HSPC batch HTML export functions.

Batch HTML annotation page generation for megakaryocyte (MK) and
hematopoietic stem/progenitor cell (HSPC) review workflows. Extracted
from ``html_generator.py`` to reduce file size.

Functions:
    load_samples_from_ram: Load cell samples from segmentation output using
        in-memory slide image (avoids repeated disk/network I/O).
    create_mk_hspc_index: Generate the main index.html for MK + HSPC review.
    generate_mk_hspc_page_html: Generate HTML for a single annotation page.
    generate_mk_hspc_pages: Generate all annotation pages for one cell type.
    export_mk_hspc_html_from_ram: Main entry point for MK/HSPC batch export.

See also ``html_generator.py`` for the class-based ``HTMLPageGenerator`` and
``html_export.py`` for core utilities and NMJ/vessel page generators.
"""

import base64
import json
import re
import warnings
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from xldvp_seg.io.html_utils import _esc, _js_esc
from xldvp_seg.utils.logging import get_logger

_logger = get_logger(__name__)


def load_samples_from_ram(tiles_dir, slide_image, pixel_size_um, cell_type="mk", max_samples=None):
    """
    Load cell samples from segmentation output, using in-memory slide image.

    This function reads segmentation results from disk and extracts image crops
    from a slide image already loaded into RAM, avoiding repeated disk/network I/O.

    Args:
        tiles_dir: Path to tiles directory (e.g., output/mk/tiles)
        slide_image: numpy array of full slide image (already in RAM)
        pixel_size_um: Pixel size in microns
        cell_type: 'mk' or 'hspc' - affects mask selection
        max_samples: Maximum samples to load (None for all)

    Returns:
        List of sample dicts with image data and metadata
    """
    from xldvp_seg.io.html_utils import (
        draw_mask_contour,
        get_largest_connected_component,
        percentile_normalize,
    )

    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        return []

    samples = []
    tile_dirs = sorted([d for d in tiles_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name))

    for tile_dir in tile_dirs:
        features_file = tile_dir / "features.json"
        seg_file = tile_dir / "segmentation.h5"
        window_file = tile_dir / "window.csv"

        if not all(f.exists() for f in [features_file, seg_file, window_file]):
            continue

        # Load tile window coordinates
        with open(window_file) as f:
            window_str = f.read().strip()
        try:
            matches = re.findall(r"slice\((\d+),\s*(\d+)", window_str)
            if len(matches) >= 2:
                tile_y1, _tile_y2 = int(matches[0][0]), int(matches[0][1])
                tile_x1, _tile_x2 = int(matches[1][0]), int(matches[1][1])
            else:
                continue
        except Exception as e:
            _logger.debug(f"Failed to parse tile coordinates from {seg_file.parent.name}: {e}")
            continue

        # Load features
        with open(features_file) as f:
            tile_features = json.load(f)

        # Load segmentation masks
        with h5py.File(seg_file, "r") as f:
            masks = f["labels"][0]  # Shape: (H, W)

        # For HSPCs, sort by solidity (higher = more confident/solid shape)
        if cell_type == "hspc":
            tile_features = sorted(
                tile_features, key=lambda x: x["features"].get("solidity", 0), reverse=True
            )

        # Extract each cell
        for feat_dict in tile_features:
            det_id = feat_dict["id"]
            features = feat_dict["features"]
            area_px = features.get("area", 0)
            area_um2 = area_px * (pixel_size_um**2)

            try:
                cell_idx = int(det_id.split("_")[1]) + 1
            except Exception as e:
                _logger.debug(f"Failed to parse cell index from {det_id}: {e}")
                continue

            cell_mask = masks == cell_idx
            if not cell_mask.any():
                cell_mask = masks == int(det_id.split("_")[1])
                if not cell_mask.any():
                    continue

            # For MKs, extract only the largest connected component
            if cell_type == "mk":
                cell_mask = get_largest_connected_component(cell_mask)
                if not cell_mask.any():
                    continue

            ys, xs = np.where(cell_mask)
            if len(ys) == 0:
                continue

            # Calculate mask centroid for centering
            centroid_y = int(np.mean(ys))
            centroid_x = int(np.mean(xs))

            # Calculate mask bounding box
            y1_local, y2_local = ys.min(), ys.max()
            x1_local, x2_local = xs.min(), xs.max()
            mask_h = y2_local - y1_local
            mask_w = x2_local - x1_local

            # Create a centered crop around the mask centroid
            # Crop size: 2x mask size, clamped to [300, 800] px
            crop_size = max(300, min(800, int(max(mask_h, mask_w) * 2)))
            half_size = crop_size // 2

            # Crop bounds centered on centroid
            crop_y1 = max(0, centroid_y - half_size)
            crop_y2 = min(masks.shape[0], centroid_y + half_size)
            crop_x1 = max(0, centroid_x - half_size)
            crop_x2 = min(masks.shape[1], centroid_x + half_size)

            # Read from in-memory slide image (instead of CZI)
            global_y1 = tile_y1 + crop_y1
            global_y2 = tile_y1 + crop_y2
            global_x1 = tile_x1 + crop_x1
            global_x2 = tile_x1 + crop_x2

            # Bounds check
            global_y2 = min(global_y2, slide_image.shape[0])
            global_x2 = min(global_x2, slide_image.shape[1])

            try:
                crop = slide_image[global_y1:global_y2, global_x1:global_x2]
            except Exception as e:
                _logger.debug(f"Failed to extract crop at ({global_x1}, {global_y1}): {e}")
                continue

            if crop is None or crop.size == 0:
                continue

            # Convert to RGB if needed
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)
            elif crop.shape[2] == 4:
                crop = crop[:, :, :3]

            # Normalize using same percentile normalization as main pipeline
            crop = percentile_normalize(crop)

            # Extract the mask for this crop region
            local_mask = cell_mask[crop_y1:crop_y2, crop_x1:crop_x2]

            # Resize crop and mask to 300x300
            pil_img = Image.fromarray(crop)
            pil_img = pil_img.resize((300, 300), Image.LANCZOS)
            crop_resized = np.array(pil_img)

            # Resize mask to match
            if local_mask.shape[0] > 0 and local_mask.shape[1] > 0:
                mask_pil = Image.fromarray(local_mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((300, 300), Image.NEAREST)
                mask_resized = np.array(mask_pil) > 127

                # Draw solid bright green contour on the image (6px thick)
                crop_with_contour = draw_mask_contour(
                    crop_resized, mask_resized, color=(0, 255, 0), dotted=False
                )
            else:
                crop_with_contour = crop_resized

            # Convert to base64 (JPEG for smaller file sizes)
            pil_img_final = Image.fromarray(crop_with_contour)
            buffer = BytesIO()
            pil_img_final.save(buffer, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Use global center from features.json if available, otherwise compute
            if "center" in feat_dict:
                # center is already in global coordinates
                global_centroid_x = int(feat_dict["center"][0])
                global_centroid_y = int(feat_dict["center"][1])
            else:
                # Backwards compatibility: compute from tile origin + local centroid
                global_centroid_x = tile_x1 + centroid_x
                global_centroid_y = tile_y1 + centroid_y

            # Get global_id if available
            global_id = feat_dict.get("global_id", None)

            samples.append(
                {
                    "tile_id": tile_dir.name,
                    "det_id": det_id,
                    "global_id": global_id,
                    "area_px": area_px,
                    "area_um2": area_um2,
                    "image": img_b64,
                    "features": features,
                    "solidity": features.get("solidity", 0),
                    "circularity": features.get("circularity", 0),
                    "global_x": global_centroid_x,
                    "global_y": global_centroid_y,
                }
            )

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


def create_mk_hspc_index(
    output_dir,
    total_mks,
    total_hspcs,
    mk_pages,
    hspc_pages,
    slides_summary=None,
    timestamp=None,
    experiment_name=None,
):
    """
    Create the main index.html page for MK + HSPC batch review.

    Args:
        output_dir: Directory to write index.html
        total_mks: Total number of MK samples
        total_hspcs: Total number of HSPC samples
        mk_pages: Number of MK pages
        hspc_pages: Number of HSPC pages
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)"
        timestamp: Segmentation timestamp string
        experiment_name: Optional experiment identifier for localStorage isolation and download filenames
    """
    subtitle_html = (
        f'<p style="color: #888; margin-bottom: 10px;">{slides_summary}</p>'
        if slides_summary
        else ""
    )
    timestamp_html = (
        f'<p style="color: #666; font-size: 0.9em;">Segmentation: {timestamp}</p>'
        if timestamp
        else ""
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>MK + HSPC Cell Review</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; }}
        .header {{ background: #111; padding: 20px; border: 1px solid #333; margin-bottom: 20px; text-align: center; }}
        h1 {{ font-size: 1.5em; font-weight: normal; margin-bottom: 15px; }}
        .stats {{ display: flex; justify-content: center; gap: 30px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; }}
        .stat .number {{ display: block; font-size: 2em; margin-top: 10px; }}
        .section {{ margin: 40px 0; }}
        .section h2 {{ font-size: 1.3em; margin-bottom: 15px; padding: 10px; background: #111; border: 1px solid #333; border-left: 3px solid #555; }}
        .controls {{ text-align: center; margin: 30px 0; }}
        .btn {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; color: #ddd; cursor: pointer; font-family: monospace; font-size: 1.1em; margin: 10px; text-decoration: none; display: inline-block; }}
        .btn:hover {{ background: #222; }}
        .btn-primary {{ border-color: #4a4; color: #4a4; }}
        .btn-export {{ border-color: #44a; color: #44a; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MK + HSPC Cell Review</h1>
        {subtitle_html}
        {timestamp_html}
        <p style="color: #888;">Annotation Interface</p>
        <div class="stats">
            <div class="stat"><span>Total MKs</span><span class="number">{total_mks:,}</span></div>
            <div class="stat"><span>Total HSPCs</span><span class="number">{total_hspcs:,}</span></div>
            <div class="stat"><span>MK Pages</span><span class="number">{mk_pages}</span></div>
            <div class="stat"><span>HSPC Pages</span><span class="number">{hspc_pages}</span></div>
        </div>
    </div>
    <div class="section">
        <h2>Megakaryocytes (MKs)</h2>
        <div class="controls">
            <a href="mk_page1.html" class="btn btn-primary">Review MKs</a>
        </div>
    </div>
    <div class="section">
        <h2>HSPCs</h2>
        <div class="controls">
            <a href="hspc_page1.html" class="btn btn-primary">Review HSPCs</a>
        </div>
    </div>
    <div class="section">
        <h2>Manage Annotations</h2>
        <div class="controls">
            <button class="btn" onclick="importAnnotations()" style="border-color: #4a8; color: #4a8;">Import Annotations JSON</button>
            <button class="btn btn-export" onclick="exportAnnotations()">Download Annotations JSON</button>
            <button class="btn" onclick="clearAllMK()" style="border-color: #a44; color: #a44;">Clear All MK</button>
            <button class="btn" onclick="clearAllHSPC()" style="border-color: #a44; color: #a44;">Clear All HSPC</button>
            <button class="btn" onclick="clearEverything()" style="border-color: #f44; color: #f44;">Clear Everything</button>
        </div>
    </div>
    <script>
        const EXPERIMENT_NAME = '{_js_esc(experiment_name or "")}';
        const MK_PAGES = {mk_pages};
        const HSPC_PAGES = {hspc_pages};

        function getGlobalKey(ct) {{
            return EXPERIMENT_NAME ? ct + '_' + EXPERIMENT_NAME + '_annotations' : ct + '_annotations';
        }}
        function getPageKey(ct, page) {{
            return EXPERIMENT_NAME ? ct + '_' + EXPERIMENT_NAME + '_labels_page' + page : ct + '_labels_page' + page;
        }}

        function clearCellType(ct, totalPages) {{
            localStorage.removeItem(getGlobalKey(ct));
            for (let i = 1; i <= totalPages; i++) {{
                localStorage.removeItem(getPageKey(ct, i));
            }}
        }}
        function clearAllMK() {{
            if (!confirm('Clear ALL MK annotations across all ' + MK_PAGES + ' pages?')) return;
            clearCellType('mk', MK_PAGES);
            alert('All MK annotations cleared.');
        }}
        function clearAllHSPC() {{
            if (!confirm('Clear ALL HSPC annotations across all ' + HSPC_PAGES + ' pages?')) return;
            clearCellType('hspc', HSPC_PAGES);
            alert('All HSPC annotations cleared.');
        }}
        function clearEverything() {{
            if (!confirm('Clear ALL MK and HSPC annotations? This cannot be undone.')) return;
            clearCellType('mk', MK_PAGES);
            clearCellType('hspc', HSPC_PAGES);
            alert('All annotations cleared.');
        }}

        function importAnnotations() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = function(e) {{
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(ev) {{
                    try {{
                        const data = JSON.parse(ev.target.result);
                        let totalImported = 0;
                        ['mk', 'hspc'].forEach(ct => {{
                            let imported = {{}};
                            const section = data[ct];
                            if (section) {{
                                (section.positive || []).forEach(uid => imported[uid] = 1);
                                (section.negative || []).forEach(uid => imported[uid] = 0);
                                (section.unsure || []).forEach(uid => imported[uid] = 2);
                            }} else if (data.cell_type === ct) {{
                                (data.positive || []).forEach(uid => imported[uid] = 1);
                                (data.negative || []).forEach(uid => imported[uid] = 0);
                                (data.unsure || []).forEach(uid => imported[uid] = 2);
                            }}
                            if (Object.keys(imported).length > 0) {{
                                const key = getGlobalKey(ct);
                                let existing = {{}};
                                try {{ existing = JSON.parse(localStorage.getItem(key)) || {{}}; }} catch(ex) {{}}
                                Object.assign(existing, imported);
                                localStorage.setItem(key, JSON.stringify(existing));
                                totalImported += Object.keys(imported).length;
                            }}
                            // Clear page-specific keys so reload picks up from global
                            const pages = ct === 'mk' ? MK_PAGES : HSPC_PAGES;
                            for (let i = 1; i <= pages; i++) {{
                                localStorage.removeItem(getPageKey(ct, i));
                            }}
                        }});
                        if (totalImported === 0) {{
                            alert('No annotations found in file.');
                        }} else {{
                            alert('Imported ' + totalImported + ' annotations. Reloading...');
                            location.reload();
                        }}
                    }} catch(err) {{
                        alert('Failed to parse JSON: ' + err.message);
                    }}
                }};
                reader.readAsText(file);
            }};
            input.click();
        }}

        function exportAnnotations() {{
            const allLabels = {{}};
            ['mk', 'hspc'].forEach(ct => {{
                const result = {{ positive: [], negative: [], unsure: [] }};
                const globalStored = localStorage.getItem(getGlobalKey(ct));
                if (globalStored) {{
                    try {{
                        const labels = JSON.parse(globalStored);
                        for (const [uid, label] of Object.entries(labels)) {{
                            if (label === 1) result.positive.push(uid);
                            else if (label === 0) result.negative.push(uid);
                            else if (label === 2) result.unsure.push(uid);
                        }}
                    }} catch(e) {{ console.error(e); }}
                }} else {{
                    const totalPages = ct === 'mk' ? MK_PAGES : HSPC_PAGES;
                    for (let i = 1; i <= totalPages; i++) {{
                        try {{
                            const labels = JSON.parse(localStorage.getItem(getPageKey(ct, i)));
                            if (!labels) continue;
                            for (const [uid, label] of Object.entries(labels)) {{
                                if (label === 1) result.positive.push(uid);
                                else if (label === 0) result.negative.push(uid);
                                else if (label === 2) result.unsure.push(uid);
                            }}
                        }} catch(e) {{ console.error(e); }}
                    }}
                }}
                allLabels[ct] = result;
            }});
            if (EXPERIMENT_NAME) {{
                allLabels['experiment_name'] = EXPERIMENT_NAME;
            }}
            const blob = new Blob([JSON.stringify(allLabels, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = EXPERIMENT_NAME ? 'annotations_' + EXPERIMENT_NAME + '.json' : 'all_labels_combined.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>"""
    with open(Path(output_dir) / "index.html", "w") as f:
        f.write(html)


def generate_mk_hspc_page_html(
    samples, cell_type, page_num, total_pages, slides_summary=None, experiment_name=None
):
    """
    Generate HTML for a single MK or HSPC annotation page.

    Args:
        samples: List of sample dicts with image data
        cell_type: 'mk' or 'hspc'
        page_num: Current page number (1-indexed)
        total_pages: Total number of pages for this cell type
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)" for subtitle
        experiment_name: Optional experiment identifier for localStorage isolation and download filenames

    Returns:
        HTML string for the page
    """
    cell_type_display = _esc("Megakaryocytes (MKs)" if cell_type == "mk" else "HSPCs")
    cell_type_safe = _esc(cell_type)

    # Build subtitle HTML
    subtitle_html = ""
    if slides_summary:
        subtitle_html = f'<div class="header-subtitle">{_esc(slides_summary)}</div>'

    nav_html = '<div class="page-nav">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{cell_type_safe}_page{page_num-1}.html" class="nav-btn">Previous</a>'
    nav_html += f'<span class="page-info">Page {page_num} of {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{cell_type_safe}_page{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += "</div>"

    cards_html = ""
    for sample in samples:
        slide = sample.get("slide", "unknown").replace(".", "-")
        global_x = sample.get("global_x", 0)
        global_y = sample.get("global_y", 0)
        # Always use spatial UID format for consistency across all cell types
        # Format: {slide}_{celltype}_{round(x)}_{round(y)}
        uid = _esc(f"{slide}_{cell_type}_{int(round(global_x))}_{int(round(global_y))}")
        # Extract short slide name (e.g., "FGC1" from "2025_11_18_FGC1")
        short_slide = slide.split("_")[-1] if "_" in slide else slide
        display_id = _esc(
            f"{short_slide}_{cell_type}_{int(round(global_x))}_{int(round(global_y))}"
        )
        # Keep legacy global_id in data attribute for backwards compatibility
        legacy_global_id = sample.get("global_id")
        area_um2 = sample.get("area_um2", 0)
        area_px = sample.get("area_px", 0)
        mk_score = sample.get("mk_score")
        score_html = (
            f'<div class="card-score">CLF: {mk_score:.2f}</div>' if mk_score is not None else ""
        )
        img_b64 = sample["image"]
        # Include legacy_global_id as data attribute for migration support
        legacy_attr = (
            f' data-legacy-id="{_esc(legacy_global_id)}"' if legacy_global_id is not None else ""
        )
        cards_html += f"""
        <div class="card" id="{uid}" data-label="-1"{legacy_attr}>
            <div class="card-img-container">
                <img src="data:image/jpeg;base64,{img_b64}" alt="{display_id}">
            </div>
            <div class="card-info">
                <div>
                    <div class="card-id">{display_id}</div>
                    <div class="card-area">{area_um2:.1f} um2 | {area_px:.0f} px2</div>
                        {score_html}
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{uid}', 1)">Y</button>
                    <button class="btn btn-unsure" onclick="setLabel('{uid}', 2)">?</button>
                    <button class="btn btn-no" onclick="setLabel('{uid}', 0)">N</button>
                </div>
            </div>
        </div>
"""

    prev_page = page_num - 1
    next_page = page_num + 1

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{cell_type_display} - Page {page_num}/{total_pages}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; }}
        .header {{ background: #111; padding: 12px 20px; display: flex; flex-direction: column; gap: 8px; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ font-size: 1.2em; font-weight: normal; }}
        .header-subtitle {{ font-size: 0.85em; color: #888; margin-top: 2px; }}
        .stats-row {{ display: flex; gap: 20px; font-size: 0.85em; flex-wrap: wrap; }}
        .stats-group {{ display: flex; gap: 8px; align-items: center; }}
        .stats-label {{ color: #888; font-size: 0.9em; }}
        .stat {{ padding: 4px 10px; background: #1a1a1a; border: 1px solid #333; }}
        .stat.positive {{ border-left: 3px solid #4a4; }}
        .stat.negative {{ border-left: 3px solid #a44; }}
        .stat.global {{ background: #0f1a0f; }}
        .page-nav {{ text-align: center; padding: 15px; background: #111; border-bottom: 1px solid #333; }}
        .nav-btn {{ display: inline-block; padding: 8px 16px; margin: 0 10px; background: #1a1a1a; color: #ddd; text-decoration: none; border: 1px solid #333; }}
        .nav-btn:hover {{ background: #222; }}
        .page-info {{ margin: 0 20px; }}
        .content {{ padding: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }}
        .card {{ background: #111; border: 1px solid #333; display: flex; flex-direction: column; }}
        .card-img-container {{ width: 100%; height: 280px; display: flex; align-items: center; justify-content: center; background: #0a0a0a; overflow: hidden; }}
        .card img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        .card-info {{ padding: 8px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #333; }}
        .card-id {{ font-size: 0.75em; color: #888; }}
        .card-area {{ font-size: 0.8em; }}
        .card-score {{ font-size: 0.8em; color: #888; }}
        .buttons {{ display: flex; gap: 4px; }}
        .btn {{ padding: 6px 12px; border: 1px solid #333; background: #1a1a1a; color: #ddd; cursor: pointer; font-family: monospace; }}
        .btn:hover {{ background: #222; }}
        .card.labeled-yes {{ border: 3px solid #0f0 !important; background: #131813 !important; }}
        .card.labeled-no {{ border: 3px solid #f00 !important; background: #181111 !important; }}
        .card.labeled-unsure {{ border: 3px solid #fa0 !important; background: #181611 !important; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <h1>{cell_type_display} - Page {page_num}/{total_pages}</h1>
            {subtitle_html}
        </div>
        <div class="stats-row">
            <div class="stats-group">
                <span class="stats-label">This Page:</span>
                <div class="stat">Total: <span id="sample-count">{len(samples)}</span></div>
                <div class="stat positive">Yes: <span id="positive-count">0</span></div>
                <div class="stat negative">No: <span id="negative-count">0</span></div>
            </div>
            <div class="stats-group">
                <span class="stats-label">Global ({total_pages} pages):</span>
                <div class="stat global positive">Yes: <span id="global-positive">0</span></div>
                <div class="stat global negative">No: <span id="global-negative">0</span></div>
            </div>
            <button class="btn" style="border-color: #4a8; color: #4a8;" onclick="importAnnotations()">Import</button>
            <button class="btn" onclick="exportAnnotations()">Export</button>
            <button class="btn" onclick="clearPage()">Clear Page</button>
            <button class="btn" onclick="clearAll()">Clear All</button>
        </div>
    </div>
    {nav_html}
    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>
    {nav_html}
    <script>
        const EXPERIMENT_NAME = '{_js_esc(experiment_name or "")}';
        const PAGE_STORAGE_KEY = EXPERIMENT_NAME ? '{cell_type_safe}_' + EXPERIMENT_NAME + '_labels_page{page_num}' : '{cell_type_safe}_labels_page{page_num}';
        const GLOBAL_STORAGE_KEY = EXPERIMENT_NAME ? '{cell_type_safe}_' + EXPERIMENT_NAME + '_annotations' : '{cell_type_safe}_annotations';
        const CELL_TYPE = '{cell_type_safe}';
        const TOTAL_PAGES = {total_pages};

        function loadAnnotations() {{
            const allCards = document.querySelectorAll('.card');
            try {{
                let stored = localStorage.getItem(PAGE_STORAGE_KEY);
                if (!stored) {{
                    // Fallback: load only UIDs present on this page from global
                    const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                    if (globalSaved) {{
                        const globalLabels = JSON.parse(globalSaved);
                        const pageUids = new Set(Array.from(allCards).map(c => c.id));
                        for (const [uid, label] of Object.entries(globalLabels)) {{
                            if (pageUids.has(uid)) {{
                                const card = document.getElementById(uid);
                                if (card && label !== -1) {{
                                    card.dataset.label = label;
                                    card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                                    if (label === 1) card.classList.add('labeled-yes');
                                    else if (label === 2) card.classList.add('labeled-unsure');
                                    else if (label === 0) card.classList.add('labeled-no');
                                }}
                            }}
                        }}
                    }}
                }} else {{
                    const labels = JSON.parse(stored);
                    for (const [uid, label] of Object.entries(labels)) {{
                        const card = document.getElementById(uid);
                        if (card && label !== -1) {{
                            card.dataset.label = label;
                            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                            if (label === 1) card.classList.add('labeled-yes');
                            else if (label === 2) card.classList.add('labeled-unsure');
                            else if (label === 0) card.classList.add('labeled-no');
                        }}
                    }}
                }}
                updateStats();
            }} catch(e) {{ console.error(e); }}
        }}

        function setLabel(uid, label, autoAdvance = false) {{
            const card = document.getElementById(uid);
            if (!card) return;
            // Toggle off if same label
            const currentLabel = parseInt(card.dataset.label);
            if (currentLabel === label) {{
                card.dataset.label = -1;
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            }} else {{
                card.dataset.label = label;
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                if (label === 1) card.classList.add('labeled-yes');
                else if (label === 2) card.classList.add('labeled-unsure');
                else if (label === 0) card.classList.add('labeled-no');
            }}
            saveAnnotations();
            updateStats();
        }}

        function exportAnnotations() {{
            let globalLabels = {{}};
            try {{ globalLabels = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(e) {{}}
            const result = {{ positive: [], negative: [], unsure: [] }};
            for (const [uid, label] of Object.entries(globalLabels)) {{
                if (label === 1) result.positive.push(uid);
                else if (label === 0) result.negative.push(uid);
                else if (label === 2) result.unsure.push(uid);
            }}
            const data = {{ cell_type: CELL_TYPE, experiment_name: EXPERIMENT_NAME, ...result }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = EXPERIMENT_NAME ? CELL_TYPE + '_annotations_' + EXPERIMENT_NAME + '.json' : CELL_TYPE + '_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        function clearPage() {{
            if (!confirm('Clear all annotations on this page?')) return;
            document.querySelectorAll('.card').forEach(card => {{
                card.dataset.label = -1;
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            }});
            saveAnnotations();
            updateStats();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL ' + CELL_TYPE.toUpperCase() + ' annotations across ALL pages?')) return;
            localStorage.removeItem(GLOBAL_STORAGE_KEY);
            for (let i = 1; i <= TOTAL_PAGES; i++) {{
                const pageKey = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_labels_page' + i : CELL_TYPE + '_labels_page' + i;
                localStorage.removeItem(pageKey);
            }}
            document.querySelectorAll('.card').forEach(card => {{
                card.dataset.label = -1;
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            }});
            updateStats();
        }}

        function saveAnnotations() {{
            const labels = {{}};
            document.querySelectorAll('.card').forEach(card => {{
                const label = parseInt(card.dataset.label);
                if (label !== -1) labels[card.id] = label;
            }});
            localStorage.setItem(PAGE_STORAGE_KEY, JSON.stringify(labels));
            // Merge into global store: add/update labeled, remove unlabeled
            let globalLabels = {{}};
            try {{ globalLabels = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(e) {{}}
            Object.assign(globalLabels, labels);
            // Delete UIDs from global that were un-labeled on this page
            document.querySelectorAll('.card').forEach(card => {{
                if (!(card.id in labels)) delete globalLabels[card.id];
            }});
            localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(globalLabels));
        }}

        function updateStats() {{
            // Local stats (this page)
            let pos = 0, neg = 0;
            document.querySelectorAll('.card').forEach(card => {{
                const label = parseInt(card.dataset.label);
                if (label === 1) pos++;
                else if (label === 0) neg++;
            }});
            document.getElementById('positive-count').textContent = pos;
            document.getElementById('negative-count').textContent = neg;

            // Global stats from global key
            let globalPos = 0, globalNeg = 0;
            try {{
                const globalSaved = localStorage.getItem(GLOBAL_STORAGE_KEY);
                if (globalSaved) {{
                    const globalLabels = JSON.parse(globalSaved);
                    for (const v of Object.values(globalLabels)) {{
                        if (v === 1) globalPos++;
                        else if (v === 0) globalNeg++;
                    }}
                }}
            }} catch(e) {{}}
            document.getElementById('global-positive').textContent = globalPos;
            document.getElementById('global-negative').textContent = globalNeg;
        }}

        function importAnnotations() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = function(e) {{
                const file = e.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(ev) {{
                    try {{
                        const data = JSON.parse(ev.target.result);
                        let imported = {{}};
                        if (data.positive || data.negative || data.unsure) {{
                            (data.positive || []).forEach(uid => imported[uid] = 1);
                            (data.negative || []).forEach(uid => imported[uid] = 0);
                            (data.unsure || []).forEach(uid => imported[uid] = 2);
                        }} else if (data[CELL_TYPE]) {{
                            const section = data[CELL_TYPE];
                            (section.positive || []).forEach(uid => imported[uid] = 1);
                            (section.negative || []).forEach(uid => imported[uid] = 0);
                            (section.unsure || []).forEach(uid => imported[uid] = 2);
                        }} else {{
                            for (const [k, v] of Object.entries(data)) {{
                                if (v === 0 || v === 1 || v === 2) imported[k] = v;
                            }}
                        }}
                        if (Object.keys(imported).length === 0) {{
                            alert('No annotations found in file.');
                            return;
                        }}
                        let existing = {{}};
                        try {{ existing = JSON.parse(localStorage.getItem(GLOBAL_STORAGE_KEY)) || {{}}; }} catch(ex) {{}}
                        Object.assign(existing, imported);
                        localStorage.setItem(GLOBAL_STORAGE_KEY, JSON.stringify(existing));
                        // Clear page-specific keys so reload picks up from global
                        for (let i = 1; i <= TOTAL_PAGES; i++) {{
                            const pageKey = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_labels_page' + i : CELL_TYPE + '_labels_page' + i;
                            localStorage.removeItem(pageKey);
                        }}
                        alert('Imported ' + Object.keys(imported).length + ' annotations. Reloading...');
                        location.reload();
                    }} catch(err) {{
                        alert('Failed to parse JSON: ' + err.message);
                    }}
                }};
                reader.readAsText(file);
            }};
            input.click();
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && {page_num} > 1)
                window.location.href = '{cell_type_safe}_page{prev_page}.html';
            else if (e.key === 'ArrowRight' && {page_num} < {total_pages})
                window.location.href = '{cell_type_safe}_page{next_page}.html';
        }});

        loadAnnotations();
    </script>
</body>
</html>"""
    return html


def generate_mk_hspc_pages(
    samples, cell_type, output_dir, samples_per_page, slides_summary=None, experiment_name=None
):
    """
    Generate separate annotation pages for MK or HSPC samples.

    Args:
        samples: List of sample dicts with image data
        cell_type: 'mk' or 'hspc'
        output_dir: Directory to write HTML files
        samples_per_page: Number of samples per page
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)" for subtitle
        experiment_name: Optional experiment identifier for localStorage isolation
    """
    if not samples:
        _logger.info(f"  No {cell_type.upper()} samples to export")
        return

    pages = [samples[i : i + samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    _logger.info(f"  Generating {total_pages} {cell_type.upper()} pages...")

    for page_num in range(1, total_pages + 1):
        page_samples = pages[page_num - 1]
        html = generate_mk_hspc_page_html(
            page_samples,
            cell_type,
            page_num,
            total_pages,
            slides_summary=slides_summary,
            experiment_name=experiment_name,
        )

        html_path = Path(output_dir) / f"{cell_type}_page{page_num}.html"
        with open(html_path, "w") as f:
            f.write(html)


def export_mk_hspc_html_from_ram(
    slide_data,
    output_base,
    html_output_dir,
    samples_per_page=300,
    mk_min_area_um=200,
    mk_max_area_um=2000,
    timestamp=None,
    experiment_name=None,
):
    """
    Export MK + HSPC HTML annotation pages using slide images already in RAM.

    This is the main entry point for MK/HSPC batch HTML export. It loads sample
    data from the segmentation output directory and generates HTML pages for
    annotation review.

    Args:
        slide_data: dict of {slide_name: {'image': np.array, 'czi_path': path, ...}}
        output_base: Path to segmentation output directory
        html_output_dir: Path to write HTML files
        samples_per_page: Number of samples per HTML page
        mk_min_area_um: Min MK area filter in um^2
        mk_max_area_um: Max MK area filter in um^2
        timestamp: Segmentation timestamp string
    """
    _logger.info(f"\n{'='*70}")
    _logger.info("EXPORTING HTML (using images in RAM)")
    _logger.info(f"{'='*70}")

    html_output_dir = Path(html_output_dir)
    html_output_dir.mkdir(parents=True, exist_ok=True)

    all_mk_samples = []
    all_hspc_samples = []

    _LEGACY_PIXEL_SIZE_UM = 0.1725  # DEPRECATED: Only used as last-resort fallback

    for slide_name, data in slide_data.items():
        slide_dir = output_base / slide_name
        if not slide_dir.exists():
            continue

        _logger.info(f"  Loading {slide_name}...")

        # Get pixel size from summary
        summary_file = slide_dir / "summary.json"
        pixel_size_um = _LEGACY_PIXEL_SIZE_UM
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                ps = summary.get("pixel_size_um")
                if ps:
                    pixel_size_um = ps[0] if isinstance(ps, list) else ps
        if pixel_size_um == _LEGACY_PIXEL_SIZE_UM:
            warnings.warn(
                f"Using legacy fallback pixel size {_LEGACY_PIXEL_SIZE_UM} um/px. "
                "Provide pixel_size_um in summary.json for accurate results.",
                stacklevel=2,
            )

        slide_image = data["image"]

        # Load MK samples (uses largest connected component)
        mk_samples = load_samples_from_ram(
            slide_dir / "mk" / "tiles", slide_image, pixel_size_um, cell_type="mk"
        )

        # Load HSPC samples (sorted by solidity/confidence)
        hspc_samples = load_samples_from_ram(
            slide_dir / "hspc" / "tiles", slide_image, pixel_size_um, cell_type="hspc"
        )

        # Add slide name to each sample
        for s in mk_samples:
            s["slide"] = slide_name
        for s in hspc_samples:
            s["slide"] = slide_name

        all_mk_samples.extend(mk_samples)
        all_hspc_samples.extend(hspc_samples)

        _logger.info(f"    {len(mk_samples)} MKs, {len(hspc_samples)} HSPCs")

    # Filter MK by size
    um_to_px_factor = _LEGACY_PIXEL_SIZE_UM**2
    mk_min_px = int(mk_min_area_um / um_to_px_factor)
    mk_max_px = int(mk_max_area_um / um_to_px_factor)

    mk_before = len(all_mk_samples)
    all_mk_samples = [s for s in all_mk_samples if mk_min_px <= s.get("area_px", 0) <= mk_max_px]
    _logger.info(f"  MK size filter: {mk_before} -> {len(all_mk_samples)}")

    # Sort by area
    all_mk_samples.sort(key=lambda x: x.get("area_um2", 0), reverse=True)
    all_hspc_samples.sort(key=lambda x: x.get("area_um2", 0), reverse=True)

    # Build slides summary for subtitle (e.g., "16 slides (FGC1, FGC2, ...)")
    slide_names = sorted(slide_data.keys())
    num_slides = len(slide_names)
    if num_slides > 0:
        # Extract short identifiers (e.g., "FGC1" from "2025_11_18_FGC1")
        short_names = []
        for name in slide_names:
            parts = name.split("_")
            # Take the last part that looks like a group identifier
            short = parts[-1] if parts else name
            short_names.append(short)
        # Show first few names with ellipsis if many
        if len(short_names) > 6:
            preview = ", ".join(short_names[:4]) + ", ..."
        else:
            preview = ", ".join(short_names)
        slides_summary = f"{num_slides} slides ({preview})"
    else:
        slides_summary = None

    # Generate pages
    generate_mk_hspc_pages(
        all_mk_samples,
        "mk",
        html_output_dir,
        samples_per_page,
        slides_summary=slides_summary,
        experiment_name=experiment_name,
    )
    generate_mk_hspc_pages(
        all_hspc_samples,
        "hspc",
        html_output_dir,
        samples_per_page,
        slides_summary=slides_summary,
        experiment_name=experiment_name,
    )

    # Create index
    mk_pages = (
        (len(all_mk_samples) + samples_per_page - 1) // samples_per_page if all_mk_samples else 0
    )
    hspc_pages = (
        (len(all_hspc_samples) + samples_per_page - 1) // samples_per_page
        if all_hspc_samples
        else 0
    )
    create_mk_hspc_index(
        html_output_dir,
        len(all_mk_samples),
        len(all_hspc_samples),
        mk_pages,
        hspc_pages,
        slides_summary=slides_summary,
        timestamp=timestamp,
        experiment_name=experiment_name,
    )

    _logger.info(f"\n  HTML export complete: {html_output_dir}")
    _logger.info(f"  Total: {len(all_mk_samples)} MKs, {len(all_hspc_samples)} HSPCs")


# Backward compatibility aliases (original function names)
create_export_index = create_mk_hspc_index
generate_export_page_html = generate_mk_hspc_page_html
generate_export_pages = generate_mk_hspc_pages
export_html_from_ram = export_mk_hspc_html_from_ram
