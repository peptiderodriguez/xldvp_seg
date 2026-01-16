#!/usr/bin/env python3
"""
Export ALL MKs and ALL HSPCs across all slides as SEPARATE page sets.
- mk_page1.html, mk_page2.html, ...
- hspc_page1.html, hspc_page2.html, ...

Self-contained version with all required functions.
"""

import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import h5py
import hdf5plugin  # Required for LZ4 compressed h5 files
import re
from scipy import ndimage

# Use segmentation utilities
from segmentation.io.html_export import percentile_normalize, draw_mask_contour, image_to_base64
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Default contour thickness for MK/HSPC (6px for visibility)
DEFAULT_CONTOUR_THICKNESS = 6


def get_largest_connected_component(mask):
    """Extract only the largest connected component from a binary mask."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def load_samples(tiles_dir, reader, x_start, y_start, cell_type, pixel_size_um, max_samples=None):
    """
    Load cell samples from segmentation output.

    Args:
        tiles_dir: Path to tiles directory (e.g., output/mk/tiles)
        reader: CZI reader object
        x_start, y_start: CZI ROI offset
        cell_type: 'mk' or 'hspc'
        pixel_size_um: Pixel size in microns
        max_samples: Maximum samples to load (None for all)

    Returns:
        List of sample dicts with image data and metadata
    """
    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        return []

    samples = []
    tile_dirs = sorted([d for d in tiles_dir.iterdir() if d.is_dir()],
                       key=lambda x: int(x.name))

    for tile_dir in tile_dirs:
        features_file = tile_dir / "features.json"
        seg_file = tile_dir / "segmentation.h5"
        window_file = tile_dir / "window.csv"

        if not all(f.exists() for f in [features_file, seg_file, window_file]):
            continue

        # Load tile window coordinates
        with open(window_file, 'r') as f:
            window_str = f.read().strip()
        try:
            matches = re.findall(r'slice\((\d+),\s*(\d+)', window_str)
            if len(matches) >= 2:
                tile_y1, tile_y2 = int(matches[0][0]), int(matches[0][1])
                tile_x1, tile_x2 = int(matches[1][0]), int(matches[1][1])
            else:
                continue
        except Exception as e:
            logger.debug(f"Failed to parse tile coordinates from {seg_file.parent.name}: {e}")
            continue

        # Load features
        with open(features_file, 'r') as f:
            tile_features = json.load(f)

        # Load segmentation masks
        with h5py.File(seg_file, 'r') as f:
            masks = f['labels'][0]  # Shape: (H, W)

        # For HSPCs, sort by solidity (higher = more confident/solid shape)
        if cell_type == 'hspc':
            tile_features = sorted(tile_features,
                                   key=lambda x: x['features'].get('solidity', 0),
                                   reverse=True)

        # Extract each cell
        for feat_dict in tile_features:
            det_id = feat_dict['id']
            features = feat_dict['features']
            area_px = features.get('area', 0)
            area_um2 = area_px * (pixel_size_um ** 2)

            # Get cell ID from det_id (format: det_N)
            try:
                cell_idx = int(det_id.split('_')[1]) + 1  # masks are 1-indexed
            except Exception as e:
                logger.debug(f"Failed to parse cell index from {det_id}: {e}")
                continue

            # Find cell bounding box in mask
            cell_mask = masks == cell_idx
            if not cell_mask.any():
                cell_mask = masks == int(det_id.split('_')[1])
                if not cell_mask.any():
                    continue

            # For MKs, extract only the largest connected component
            if cell_type == 'mk':
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
            crop_size = max(300, max(mask_h, mask_w) + 100)
            half_size = crop_size // 2

            # Crop bounds centered on centroid
            crop_y1 = max(0, centroid_y - half_size)
            crop_y2 = min(masks.shape[0], centroid_y + half_size)
            crop_x1 = max(0, centroid_x - half_size)
            crop_x2 = min(masks.shape[1], centroid_x + half_size)

            # Read from CZI
            roi_x = x_start + tile_x1 + crop_x1
            roi_y = y_start + tile_y1 + crop_y1
            roi_w = crop_x2 - crop_x1
            roi_h = crop_y2 - crop_y1

            try:
                crop = reader.read(roi=(roi_x, roi_y, roi_w, roi_h), plane={'C': 0})
            except Exception as e:
                continue

            if crop is None or crop.size == 0:
                continue

            # Convert to RGB if needed
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)
            elif crop.shape[2] == 4:
                crop = crop[:, :, :3]

            # Normalize using same percentile normalization as main pipeline
            crop = percentile_normalize(crop, p_low=5, p_high=95)

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
                crop_with_contour = draw_mask_contour(crop_resized, mask_resized,
                                                       color=(0, 255, 0),
                                                       thickness=DEFAULT_CONTOUR_THICKNESS,
                                                       dotted=False)
            else:
                crop_with_contour = crop_resized

            # Convert to base64 (JPEG for smaller file sizes)
            img_b64, _ = image_to_base64(crop_with_contour, format='JPEG', quality=85)

            samples.append({
                'tile_id': tile_dir.name,
                'det_id': det_id,
                'area_px': area_px,
                'area_um2': area_um2,
                'image': img_b64,
                'features': features,
                'solidity': features.get('solidity', 0),
                'circularity': features.get('circularity', 0)
            })

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


def create_index(output_dir, total_mks, total_hspcs, mk_pages, hspc_pages):
    """Create the main index.html page."""

    html = f'''<!DOCTYPE html>
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
        <h2>Export Annotations</h2>
        <div class="controls">
            <button class="btn btn-export" onclick="exportAnnotations()">Download Annotations JSON</button>
        </div>
    </div>

    <script>
        function exportAnnotations() {{
            const allLabels = {{}};
            const mkLabels = {{ positive: [], negative: [] }};
            const hspcLabels = {{ positive: [], negative: [] }};

            // Collect all localStorage keys
            for (let i = 0; i < localStorage.length; i++) {{
                const key = localStorage.key(i);
                if (key.startsWith('mk_labels_page') || key.startsWith('hspc_labels_page')) {{
                    try {{
                        const labels = JSON.parse(localStorage.getItem(key));
                        const cellType = key.startsWith('mk_') ? mkLabels : hspcLabels;
                        for (const [uid, label] of Object.entries(labels)) {{
                            if (label === 1) cellType.positive.push(uid);
                            else if (label === 0) cellType.negative.push(uid);
                        }}
                    }} catch(e) {{ console.error(e); }}
                }}
            }}

            allLabels.mk = mkLabels;
            allLabels.hspc = hspcLabels;

            // Download
            const blob = new Blob([JSON.stringify(allLabels, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'all_labels_combined.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''

    with open(Path(output_dir) / 'index.html', 'w') as f:
        f.write(html)


def generate_cell_type_pages(samples, cell_type, output_dir, samples_per_page):
    """Generate separate pages for a single cell type."""

    if not samples:
        logger.warning(f"\nNo {cell_type.upper()} samples to export")
        return

    pages = [samples[i:i+samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    logger.info(f"\nGenerating {total_pages} {cell_type.upper()} pages...")

    for page_num in range(1, total_pages + 1):
        page_samples = pages[page_num - 1]
        html = generate_single_type_page_html(page_samples, cell_type, page_num, total_pages, output_dir)

        html_path = Path(output_dir) / f"{cell_type}_page{page_num}.html"
        with open(html_path, 'w') as f:
            f.write(html)

        file_size = html_path.stat().st_size / (1024*1024)
        logger.info(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")


def generate_single_type_page_html(samples, cell_type, page_num, total_pages, output_dir):
    """Generate HTML for a single cell type page."""

    cell_type_display = "Megakaryocytes (MKs)" if cell_type == "mk" else "HSPCs"
    prev_page = page_num - 1
    next_page = page_num + 1

    # Navigation
    nav_html = '<div class="page-nav">'
    nav_html += f'<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{cell_type}_page{page_num-1}.html" class="nav-btn">Previous</a>'
    nav_html += f'<span class="page-info">Page {page_num} of {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{cell_type}_page{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += '</div>'

    # Generate cards
    cards_html = ""
    for sample in samples:
        slide = sample.get('slide', 'unknown').replace('.', '-')
        tile_id = str(sample.get('tile_id', '0'))
        det_id = sample.get('det_id', 'unknown')
        uid = f"{slide}_{tile_id}_{det_id}"

        area_um2 = sample.get('area_um2', 0)
        area_px = sample.get('area_px', 0)
        img_b64 = sample['image']

        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1">
            <div class="card-img-container">
                <img src="data:image/jpeg;base64,{img_b64}" alt="{det_id}">
            </div>
            <div class="card-info">
                <div>
                    <div class="card-id">{slide} | {tile_id} | {det_id}</div>
                    <div class="card-area">{area_um2:.1f} um2 | {area_px:.0f} px2</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{cell_type}', '{uid}', 1)">Yes</button>
                    <button class="btn btn-unsure" onclick="setLabel('{cell_type}', '{uid}', 2)">?</button>
                    <button class="btn btn-no" onclick="setLabel('{cell_type}', '{uid}', 0)">No</button>
                </div>
            </div>
        </div>
'''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{cell_type_display} - Page {page_num}/{total_pages}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; }}
        .header {{ background: #111; padding: 12px 20px; display: flex; flex-direction: column; gap: 8px; border-bottom: 1px solid #333; position: sticky; top: 0; z-index: 100; }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ font-size: 1.2em; font-weight: normal; }}
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
        </div>
    </div>
    {nav_html}
    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>
    {nav_html}
    <script>
        const STORAGE_KEY = '{cell_type}_labels_page{page_num}';
        const CELL_TYPE = '{cell_type}';
        const TOTAL_PAGES = {total_pages};

        function loadAnnotations() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            if (!stored) return;
            try {{
                const labels = JSON.parse(stored);
                for (const [uid, label] of Object.entries(labels)) {{
                    const card = document.getElementById(uid);
                    if (card && label !== -1) {{
                        card.dataset.label = label;
                        card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                        if (label == 1) card.classList.add('labeled-yes');
                        else if (label == 2) card.classList.add('labeled-unsure');
                        else card.classList.add('labeled-no');
                    }}
                }}
                updateStats();
            }} catch(e) {{ console.error(e); }}
        }}

        function setLabel(cellType, uid, label) {{
            const card = document.getElementById(uid);
            if (!card) return;
            card.dataset.label = label;
            card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
            if (label == 1) card.classList.add('labeled-yes');
            else if (label == 2) card.classList.add('labeled-unsure');
            else card.classList.add('labeled-no');
            saveAnnotations();
            updateStats();
        }}

        function saveAnnotations() {{
            const labels = {{}};
            document.querySelectorAll('.card').forEach(card => {{
                const label = parseInt(card.dataset.label);
                if (label !== -1) labels[card.id] = label;
            }});
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
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

            // Global stats (all pages)
            let globalPos = 0, globalNeg = 0;
            for (let i = 1; i <= TOTAL_PAGES; i++) {{
                const key = CELL_TYPE + '_labels_page' + i;
                const stored = localStorage.getItem(key);
                if (stored) {{
                    try {{
                        const labels = JSON.parse(stored);
                        for (const label of Object.values(labels)) {{
                            if (label === 1) globalPos++;
                            else if (label === 0) globalNeg++;
                        }}
                    }} catch(e) {{}}
                }}
            }}
            document.getElementById('global-positive').textContent = globalPos;
            document.getElementById('global-negative').textContent = globalNeg;
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft' && {page_num} > 1)
                window.location.href = '{cell_type}_page{prev_page}.html';
            else if (e.key === 'ArrowRight' && {page_num} < {total_pages})
                window.location.href = '{cell_type}_page{next_page}.html';
        }});

        // Ensure DOM is ready before initializing
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', loadAnnotations);
        }} else {{
            loadAnnotations();
        }}
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description='Export separate MK and HSPC pages for all slides')
    parser.add_argument('--base-dir', type=str, required=True,
                       help='Base directory containing slide folders')
    parser.add_argument('--czi-base', type=str, required=True,
                       help='Base directory containing CZI files')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--samples-per-page', type=int, default=300)
    parser.add_argument('--mk-min-area-um', type=float, default=200,
                       help='Minimum MK area in um2 (must match segmentation filter)')
    parser.add_argument('--mk-max-area-um', type=float, default=2000,
                       help='Maximum MK area in um2 (must match segmentation filter)')

    args = parser.parse_args()

    from pylibCZIrw import czi as pyczi

    base_dir = Path(args.base_dir)
    czi_base = Path(args.czi_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all slide directories (flexible pattern matching)
    slide_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

    logger.info(f"\n{'='*60}")
    logger.info(f"Exporting SEPARATE MK and HSPC Pages")
    logger.info(f"{'='*60}")
    logger.info(f"Found {len(slide_dirs)} slides")
    logger.info(f"Samples per page: {args.samples_per_page}")
    logger.info(f"{'='*60}\n")

    all_mk_samples = []
    all_hspc_samples = []

    # Load samples from all slides
    for slide_dir in slide_dirs:
        slide_name = slide_dir.name

        # Try multiple CZI path patterns
        czi_candidates = [
            czi_base / f"{slide_name}.czi",
            czi_base / slide_name / f"{slide_name}.czi",
        ]
        czi_path = None
        for c in czi_candidates:
            if c.exists():
                czi_path = c
                break

        if not czi_path:
            logger.warning(f"Skipping {slide_name}: CZI not found")
            continue

        logger.info(f"Loading {slide_name}...")

        # Load pixel size
        summary_file = slide_dir / "summary.json"
        pixel_size_um = 0.1725
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                ps = summary.get('pixel_size_um')
                if ps:
                    pixel_size_um = ps[0] if isinstance(ps, list) else ps

        # Open CZI
        reader = pyczi.CziReader(str(czi_path))
        scenes = reader.scenes_bounding_rectangle
        if scenes:
            rect = scenes[0]
            x_start, y_start = rect.x, rect.y
        else:
            bbox = reader.total_bounding_box
            x_start, y_start = bbox['X'][0], bbox['Y'][0]

        # Load MK samples
        mk_samples = load_samples(
            slide_dir / "mk" / "tiles",
            reader, x_start, y_start,
            "mk", pixel_size_um, max_samples=None
        )

        # Load HSPC samples
        hspc_samples = load_samples(
            slide_dir / "hspc" / "tiles",
            reader, x_start, y_start,
            "hspc", pixel_size_um, max_samples=None
        )

        reader.close()

        # Add slide name to each sample
        for s in mk_samples:
            s['slide'] = slide_name
        for s in hspc_samples:
            s['slide'] = slide_name

        all_mk_samples.extend(mk_samples)
        all_hspc_samples.extend(hspc_samples)

        logger.info(f"  Loaded {len(mk_samples)} MKs, {len(hspc_samples)} HSPCs")

    logger.info(f"\n{'='*60}")
    logger.info(f"Total: {len(all_mk_samples)} MKs, {len(all_hspc_samples)} HSPCs")
    logger.info(f"{'='*60}\n")

    # Convert um2 to px2 for filtering
    PIXEL_SIZE_UM = 0.1725
    um_to_px_factor = PIXEL_SIZE_UM ** 2
    mk_min_px = int(args.mk_min_area_um / um_to_px_factor)
    mk_max_px = int(args.mk_max_area_um / um_to_px_factor)

    # Filter MK cells by size
    logger.info(f"Filtering MK cells by size ({args.mk_min_area_um}-{args.mk_max_area_um} um2)...")
    mk_before = len(all_mk_samples)
    all_mk_samples = [s for s in all_mk_samples if mk_min_px <= s.get('area_px', 0) <= mk_max_px]
    mk_after = len(all_mk_samples)
    logger.info(f"  MK cells: {mk_before} -> {mk_after} (removed {mk_before - mk_after})")

    # Sort by area (largest to smallest)
    all_mk_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)
    all_hspc_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)

    # Generate pages
    generate_cell_type_pages(all_mk_samples, "mk", output_dir, args.samples_per_page)
    generate_cell_type_pages(all_hspc_samples, "hspc", output_dir, args.samples_per_page)

    # Create index
    mk_pages = (len(all_mk_samples) + args.samples_per_page - 1) // args.samples_per_page if all_mk_samples else 0
    hspc_pages = (len(all_hspc_samples) + args.samples_per_page - 1) // args.samples_per_page if all_hspc_samples else 0
    create_index(output_dir, len(all_mk_samples), len(all_hspc_samples), mk_pages, hspc_pages)

    logger.info(f"\n{'='*60}")
    logger.info(f"Export complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Open: {output_dir / 'index.html'}")


if __name__ == '__main__':
    main()
