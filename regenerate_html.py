#!/usr/bin/env python3
"""
Regenerate HTML viewer for detection results with sorting options.

This script regenerates the HTML annotation interface from existing detections,
allowing you to:
- Sort by area (size), confidence, or other features
- Adjust crop size and mask visualization
- Re-center crops on mask centroids

Usage:
    python regenerate_html.py --output-dir /path/to/output --sort-by area --sort-order desc
"""

import argparse
import json
import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
from segmentation.io.czi_loader import get_loader

# Import hdf5plugin for LZ4 decompression
try:
    import hdf5plugin
except ImportError:
    pass

from segmentation.io.html_export import (
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def load_channel_to_ram(czi_path, channel, strip_height=5000):
    """
    Load a single channel from CZI into RAM using CZILoader.

    Uses the shared get_loader() to leverage the global cache.
    """
    loader = get_loader(czi_path, load_to_ram=True, channel=channel, strip_height=strip_height)
    channel_data = loader.channel_data
    return channel_data, (loader.x_start, loader.y_start, loader.width, loader.height)


def regenerate_html(
    output_dir,
    czi_path=None,
    channel=1,
    sort_by='area',
    sort_order='desc',
    crop_size=300,
    display_size=250,
    contour_thickness=4,
    contour_color=(50, 255, 50),
    samples_per_page=300,
    experiment_name=None,
    cell_type=None,
):
    """
    Regenerate HTML viewer with sorting and improved visualization.

    Args:
        output_dir: Path to segmentation output directory
        czi_path: Path to CZI file (auto-detected if None)
        channel: Channel to load for visualization
        sort_by: Feature to sort by ('area', 'elongation', 'confidence', etc.)
        sort_order: 'asc' or 'desc'
        crop_size: Size of crop around each detection
        display_size: Size of displayed image in HTML
        contour_thickness: Thickness of mask contour
        experiment_name: Experiment name for localStorage isolation
        contour_color: RGB color for contour
        samples_per_page: Number of samples per HTML page
        cell_type: Cell type ('mk' or 'hspc') for batch output structure
    """
    output_dir = Path(output_dir)
    html_dir = output_dir / "html"

    # Handle batch output structure (output_dir/{cell_type}/tiles/)
    if cell_type:
        tiles_dir = output_dir / cell_type / "tiles"
    else:
        tiles_dir = output_dir / "tiles"

    # Try to find detections file first
    det_files = list(output_dir.glob("*_detections.json"))

    if det_files:
        # Load from detections file
        det_file = det_files[0]
        if not cell_type:
            cell_type = det_file.stem.replace('_detections', '').split('_')[-1]
        logger.info(f"Loading detections from {det_file}")
        with open(det_file) as f:
            detections = json.load(f)
    elif tiles_dir.exists():
        # Load from individual features.json files (batch output structure)
        logger.info(f"Loading detections from tiles in {tiles_dir}")
        detections = []
        for tile_dir in tiles_dir.iterdir():
            if not tile_dir.is_dir():
                continue
            feat_file = tile_dir / "features.json"
            if feat_file.exists():
                with open(feat_file) as f:
                    tile_dets = json.load(f)
                for det in tile_dets:
                    det['tile_key'] = tile_dir.name
                detections.extend(tile_dets)
        if not cell_type:
            cell_type = 'detection'
    else:
        raise FileNotFoundError(f"No detections file or tiles directory found in {output_dir}")

    if not detections:
        logger.warning("No detections to process")
        return

    logger.info(f"Loaded {len(detections)} detections")

    # Sort detections
    reverse = (sort_order == 'desc')

    def get_sort_key(det):
        if sort_by in det.get('features', {}):
            return det['features'][sort_by]
        elif sort_by in det:
            return det[sort_by]
        return 0

    detections_sorted = sorted(detections, key=get_sort_key, reverse=reverse)
    logger.info(f"Sorted by {sort_by} ({sort_order})")

    # Auto-detect CZI path from summary
    if czi_path is None:
        summary_file = output_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            # Try to find CZI in common locations
            slide_name = summary.get('slide_name', '')
            for search_dir in [Path("/home/dude/images"), Path("/mnt/x")]:
                for czi in search_dir.rglob(f"*{slide_name}*.czi"):
                    czi_path = str(czi)
                    break
                if czi_path:
                    break

    if not czi_path:
        raise ValueError("Could not find CZI file. Please specify --czi-path")

    # Load channel into RAM
    channel_data, (x_start, y_start, width, height) = load_channel_to_ram(czi_path, channel)
    pixel_size = 0.1725  # Default, could read from summary

    # Group detections by tile
    tile_detections = {}
    for det in detections_sorted:
        # Handle both old format (tile_origin) and new format (tile_key from batch output)
        if 'tile_origin' in det:
            tile_x, tile_y = det['tile_origin']
            tile_key = f"tile_{tile_x}_{tile_y}"
        elif 'tile_key' in det:
            tile_key = det['tile_key']
        else:
            # Skip detections without tile info
            continue
        if tile_key not in tile_detections:
            tile_detections[tile_key] = []
        tile_detections[tile_key].append(det)

    logger.info(f"Processing {len(tile_detections)} tiles...")

    # Generate samples
    samples = []

    for tile_key, tile_dets in tqdm(tile_detections.items(), desc="Processing"):
        tile_path = tiles_dir / tile_key

        # Try different mask file naming conventions
        mask_file = tile_path / f"{cell_type}_masks.h5"
        if not mask_file.exists():
            mask_file = tile_path / "segmentation.h5"

        if not mask_file.exists():
            continue

        # Load masks
        with h5py.File(mask_file, 'r') as f:
            # Support both old format ('labels') and new format ('masks')
            if 'masks' in f:
                masks = f['masks'][:]
            elif 'labels' in f:
                masks = f['labels'][:]
            else:
                available = list(f.keys())
                raise KeyError(f"No 'masks' or 'labels' dataset found. Available: {available}")
            # Handle 3D masks with shape (1, H, W) - squeeze to 2D
            if masks.ndim == 3 and masks.shape[0] == 1:
                masks = masks[0]

        # Parse tile origin from window.csv if available, otherwise from tile_key
        window_file = tile_path / "window.csv"
        if window_file.exists():
            import re
            with open(window_file) as f:
                window_str = f.read()
            matches = re.findall(r'slice\((\d+),\s*(\d+)', window_str)
            if len(matches) >= 2:
                tile_y = int(matches[0][0])
                tile_x = int(matches[1][0])
            else:
                tile_x, tile_y = 0, 0
        else:
            # Try to parse from tile_key
            parts = tile_key.split('_')
            if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
                tile_x, tile_y = int(parts[-2]), int(parts[-1])
            elif len(parts) >= 2 and parts[1].isdigit():
                tile_x, tile_y = int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
            else:
                tile_x, tile_y = 0, 0

        for det in tile_dets:
            det_id = int(det['id'].split('_')[-1])

            # Quality filtering for vessels
            if cell_type == 'vessel':
                features = det.get('features', {})
                # Ring completeness: require at least 30% (handle None values)
                ring_completeness = features.get('ring_completeness')
                if ring_completeness is None or ring_completeness < 0.30:
                    continue
                # Circularity: require at least 0.15 (handle None values)
                circularity = features.get('circularity')
                if circularity is None or circularity < 0.15:
                    continue
                # Wall thickness: require at least 1.5 µm (handle None values)
                wall_thickness = features.get('wall_thickness_mean_um')
                if wall_thickness is None or wall_thickness < 1.5:
                    continue

            # Get mask
            mask = (masks == det_id)
            if not mask.any():
                continue

            # Find mask centroid for centering
            ys, xs = np.where(mask)
            cy, cx = np.mean(ys), np.mean(xs)

            # Calculate dynamic crop size based on mask bounding box
            # Crop should be 100% larger than mask (mask fills ~50% of crop)
            mask_h = ys.max() - ys.min()
            mask_w = xs.max() - xs.min()
            mask_size = max(mask_h, mask_w)
            dynamic_crop_size = max(crop_size, min(800, int(mask_size * 2)))

            # Global position
            global_cy = tile_y + cy
            global_cx = tile_x + cx

            # Extract crop centered on mask
            half = dynamic_crop_size // 2
            y1 = max(0, int(global_cy - half))
            y2 = min(height, int(global_cy + half))
            x1 = max(0, int(global_cx - half))
            x2 = min(width, int(global_cx + half))

            # Validate crop bounds before extracting
            if y2 <= y1 or x2 <= x1:
                logger.warning(f"Invalid crop bounds for detection {det_id}, skipping")
                continue

            crop = channel_data[y1:y2, x1:x2].copy()

            # Create mask for crop region
            crop_h, crop_w = y2 - y1, x2 - x1
            crop_mask = np.zeros((crop_h, crop_w), dtype=bool)

            # Map mask pixels from tile coords to crop coords (vectorized)
            # Mask pixel at (my, mx) in tile -> global (tile_y + my, tile_x + mx)
            # Crop position is global position minus crop origin (y1, x1)
            global_ys = ys + tile_y
            global_xs = xs + tile_x
            crop_ys = global_ys - y1
            crop_xs = global_xs - x1

            # Vectorized bounds check
            valid = (crop_ys >= 0) & (crop_ys < crop_h) & (crop_xs >= 0) & (crop_xs < crop_w)
            crop_mask[crop_ys[valid], crop_xs[valid]] = True

            # Normalize and draw contour
            crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5)
            crop_rgb = draw_mask_contour(
                crop_norm, crop_mask,
                color=contour_color,
                thickness=contour_thickness
            )

            # Resize
            crop_resized = cv2.resize(crop_rgb, (display_size, display_size))

            # Get features for display
            features = det['features']
            area = features.get('area', 0)
            area_um2 = area * (pixel_size ** 2)
            elongation = features.get('elongation', 0)

            # Vessel-specific features
            diameter_um = features.get('outer_diameter_um', 0)
            wall_thickness = features.get('wall_thickness_mean_um', 0)
            ring_completeness = features.get('ring_completeness', 0)
            circularity = features.get('circularity', 0)

            # image_to_base64 returns (base64_string, mime_type)
            image_b64, _ = image_to_base64(crop_resized)

            samples.append({
                'uid': det['uid'],
                'image_b64': image_b64,
                'area': area,
                'area_um2': round(area_um2, 1),
                'elongation': elongation,
                'diameter_um': round(diameter_um, 1),
                'wall_thickness': round(wall_thickness, 1),
                'ring_completeness': round(ring_completeness, 2),
                'circularity': round(circularity, 2),
            })

    logger.info(f"Generated {len(samples)} samples")

    # Generate HTML pages
    n_pages = (len(samples) + samples_per_page - 1) // samples_per_page

    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>{cell_type} Detection - Page {page}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 10px; }}
        .header {{ background: #111; padding: 10px 15px; position: sticky; top: 0; z-index: 100; border: 1px solid #333; margin-bottom: 10px; }}
        .nav-row {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
        .nav-links {{ display: flex; gap: 8px; align-items: center; }}
        .nav-links a {{ color: #4a4; text-decoration: none; padding: 6px 12px; border: 1px solid #333; }}
        .nav-links a:hover {{ background: #1a1a1a; }}
        .page-info {{ color: #888; }}
        .stats-row {{ display: flex; gap: 15px; margin-top: 10px; flex-wrap: wrap; align-items: center; }}
        .stat-group {{ display: flex; gap: 8px; align-items: center; }}
        .stat-label {{ color: #666; font-size: 0.85em; }}
        .stat {{ padding: 4px 10px; background: #1a1a1a; border: 1px solid #333; font-size: 0.9em; }}
        .stat.yes {{ border-left: 3px solid #4a4; color: #4a4; }}
        .stat.no {{ border-left: 3px solid #a44; color: #a44; }}
        .action-btn {{ padding: 5px 12px; background: #1a1a1a; border: 1px solid #444; color: #888; cursor: pointer; font-family: monospace; font-size: 0.85em; }}
        .action-btn:hover {{ background: #222; color: #aaa; }}
        .action-btn.danger {{ border-color: #a44; color: #a44; }}
        .action-btn.danger:hover {{ background: #2a1a1a; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 10px; }}
        .card {{ background: #111; border: 2px solid #333; padding: 10px; text-align: center; transition: border-color 0.2s; }}
        .card.labeled-yes {{ border-color: #4a4; background: #0f130f; }}
        .card.labeled-no {{ border-color: #a44; background: #130f0f; }}
        .card img {{ width: {display_size}px; height: {display_size}px; border: 1px solid #333; }}
        .card .info {{ font-size: 0.8em; color: #888; margin: 5px 0; }}
        .card .area {{ color: #4a4; font-weight: bold; }}
        .buttons {{ margin-top: 8px; }}
        .buttons button {{ padding: 8px 20px; margin: 0 5px; cursor: pointer; font-family: monospace; border: none; }}
        .btn-yes {{ background: #1a3a1a; color: #4a4; }}
        .btn-yes:hover {{ background: #2a4a2a; }}
        .btn-yes.selected {{ background: #4a4; color: #000; }}
        .btn-no {{ background: #3a1a1a; color: #a44; }}
        .btn-no:hover {{ background: #4a2a2a; }}
        .btn-no.selected {{ background: #a44; color: #000; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="nav-row">
            <div class="nav-links">
                {prev_link}
                <a href="index.html">Index</a>
                {next_link}
            </div>
            <span class="page-info">Page {page}/{total_pages} | Sorted by {sort_by} ({sort_order})</span>
        </div>
        <div class="stats-row">
            <div class="stat-group">
                <span class="stat-label">Page:</span>
                <span class="stat yes">Yes: <span id="pageYes">0</span></span>
                <span class="stat no">No: <span id="pageNo">0</span></span>
            </div>
            <div class="stat-group">
                <span class="stat-label">Total:</span>
                <span class="stat yes">Yes: <span id="totalYes">0</span></span>
                <span class="stat no">No: <span id="totalNo">0</span></span>
            </div>
            <button class="action-btn" onclick="clearPage()">Clear Page</button>
            <button class="action-btn danger" onclick="clearAll()">Clear All</button>
            <button class="action-btn" onclick="exportAnnotations()">Export JSON</button>
        </div>
    </div>
    <div class="grid">
        {cards}
    </div>
    <script>
        const CELL_TYPE = '{cell_type}';
        const EXPERIMENT_NAME = '{experiment_name}';
        const STORAGE_KEY = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations' : CELL_TYPE + '_annotations';
        let labels = {{}};

        function loadLabels() {{
            try {{
                labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
            }} catch(e) {{ labels = {{}}; }}
        }}

        function saveLabels() {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
        }}

        function setLabel(uid, val) {{
            // Toggle off if clicking same button
            if (labels[uid] === val) {{
                delete labels[uid];
            }} else {{
                labels[uid] = val;
            }}
            saveLabels();
            updateCard(uid);
            updateStats();
        }}

        function updateCard(uid) {{
            const card = document.querySelector(`[data-uid="${{uid}}"]`);
            if (!card) return;
            card.classList.remove('labeled-yes', 'labeled-no');
            card.querySelectorAll('button').forEach(b => b.classList.remove('selected'));
            const val = labels[uid];
            if (val === 1) {{
                card.classList.add('labeled-yes');
                card.querySelector('.btn-yes')?.classList.add('selected');
            }} else if (val === 0) {{
                card.classList.add('labeled-no');
                card.querySelector('.btn-no')?.classList.add('selected');
            }}
        }}

        function updateStats() {{
            let pageYes = 0, pageNo = 0, totalYes = 0, totalNo = 0;
            // Page stats
            document.querySelectorAll('.card').forEach(card => {{
                const uid = card.dataset.uid;
                if (labels[uid] === 1) pageYes++;
                else if (labels[uid] === 0) pageNo++;
            }});
            // Total stats
            for (const v of Object.values(labels)) {{
                if (v === 1) totalYes++;
                else if (v === 0) totalNo++;
            }}
            document.getElementById('pageYes').textContent = pageYes;
            document.getElementById('pageNo').textContent = pageNo;
            document.getElementById('totalYes').textContent = totalYes;
            document.getElementById('totalNo').textContent = totalNo;
        }}

        function clearPage() {{
            if (!confirm('Clear all annotations on this page?')) return;
            document.querySelectorAll('.card').forEach(card => {{
                const uid = card.dataset.uid;
                delete labels[uid];
                updateCard(uid);
            }});
            saveLabels();
            updateStats();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across all pages? This cannot be undone.')) return;
            labels = {{}};
            saveLabels();
            document.querySelectorAll('.card').forEach(card => updateCard(card.dataset.uid));
            updateStats();
        }}

        function exportAnnotations() {{
            const data = {{
                cell_type: '{cell_type}',
                exported_at: new Date().toISOString(),
                positive: [],
                negative: []
            }};
            for (const [uid, val] of Object.entries(labels)) {{
                if (val === 1) data.positive.push(uid);
                else if (val === 0) data.negative.push(uid);
            }}
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{cell_type}_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
        }}

        // Initialize
        loadLabels();
        document.querySelectorAll('.card').forEach(c => updateCard(c.dataset.uid));
        updateStats();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight') {{
                const next = document.querySelector('.nav-links a[href*="page_{next_page}"]');
                if (next) window.location = next.href;
            }} else if (e.key === 'ArrowLeft') {{
                const prev = document.querySelector('.nav-links a[href*="page_{prev_page}"]');
                if (prev) window.location = prev.href;
            }}
        }});
    </script>
</body>
</html>'''

    card_template = '''<div class="card" data-uid="{uid}">
    <img src="data:image/jpeg;base64,{image_b64}" alt="{uid}">
    <div class="info"><span class="area">⌀{diameter_um}&micro;m</span> | wall:{wall_thickness}&micro;m | ring:{ring_completeness}</div>
    <div class="buttons">
        <button class="btn-yes" onclick="setLabel('{uid}', 1)">Yes</button>
        <button class="btn-no" onclick="setLabel('{uid}', 0)">No</button>
    </div>
</div>'''

    html_dir.mkdir(exist_ok=True)

    for page_num in range(1, n_pages + 1):
        start = (page_num - 1) * samples_per_page
        end = min(start + samples_per_page, len(samples))
        page_samples = samples[start:end]

        cards = '\n'.join(card_template.format(**s) for s in page_samples)

        prev_link = f'<a href="{cell_type}_page_{page_num-1}.html">← Prev</a>' if page_num > 1 else '<span></span>'
        next_link = f'<a href="{cell_type}_page_{page_num+1}.html">Next →</a>' if page_num < n_pages else '<span></span>'

        html = html_template.format(
            cell_type=cell_type,
            page=page_num,
            total_pages=n_pages,
            prev_link=prev_link,
            next_link=next_link,
            cards=cards,
            sort_by=sort_by,
            sort_order=sort_order,
            display_size=display_size,
            next_page=page_num + 1,
            prev_page=page_num - 1,
            experiment_name=experiment_name or "",
        )

        page_file = html_dir / f"{cell_type}_page_{page_num}.html"
        with open(page_file, 'w') as f:
            f.write(html)

        page_size_mb = len(html) / 1e6
        logger.info(f"  Page {page_num}: {len(page_samples)} samples ({page_size_mb:.1f} MB)")

    # Update index - handle None experiment_name for f-string
    experiment_name = experiment_name or ""
    index_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{cell_type.upper()} Detection Results</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; }}
        .header {{ background: #111; padding: 30px; border: 1px solid #333; margin-bottom: 20px; text-align: center; }}
        h1 {{ font-size: 1.5em; font-weight: normal; margin-bottom: 10px; }}
        .stats {{ display: flex; justify-content: center; gap: 30px; margin: 25px 0; flex-wrap: wrap; }}
        .stat {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; text-align: center; }}
        .stat .number {{ display: block; font-size: 2em; margin-top: 10px; color: #4a4; }}
        .stat.annotation {{ border-left: 3px solid #4a4; }}
        .stat.annotation .number {{ color: #6c6; }}
        .section {{ margin: 30px 0; text-align: center; }}
        .btn {{ padding: 15px 40px; background: #1a1a1a; border: 1px solid #4a4; color: #4a4; cursor: pointer; font-family: monospace; font-size: 1.1em; text-decoration: none; display: inline-block; margin: 10px; }}
        .btn:hover {{ background: #0f130f; }}
        .btn.secondary {{ border-color: #666; color: #888; }}
        .btn.secondary:hover {{ background: #151515; }}
        .btn.danger {{ border-color: #a44; color: #a44; }}
        .btn.danger:hover {{ background: #1a0f0f; }}
        .info {{ color: #888; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{cell_type.upper()} Detection Results</h1>
        <div class="stats">
            <div class="stat"><span>Total Samples</span><span class="number">{len(samples):,}</span></div>
            <div class="stat"><span>Pages</span><span class="number">{n_pages}</span></div>
            <div class="stat annotation"><span>Annotated</span><span class="number" id="annotatedCount">0</span></div>
        </div>
        <p class="info">Sorted by {sort_by} ({sort_order})</p>
    </div>
    <div class="section">
        <a href="{cell_type}_page_1.html" class="btn">Start Review</a>
    </div>
    <div class="section">
        <button class="btn secondary" onclick="exportAnnotations()">Export Annotations</button>
        <button class="btn danger" onclick="clearAll()">Clear All Annotations</button>
    </div>
    <script>
        const CELL_TYPE = '{cell_type}';
        const EXPERIMENT_NAME = '{experiment_name}';
        const STORAGE_KEY = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations' : CELL_TYPE + '_annotations';

        function updateCount() {{
            try {{
                const labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
                document.getElementById('annotatedCount').textContent = Object.keys(labels).length;
            }} catch(e) {{}}
        }}

        function exportAnnotations() {{
            const labels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
            const data = {{
                cell_type: CELL_TYPE,
                experiment_name: EXPERIMENT_NAME || undefined,
                exported_at: new Date().toISOString(),
                positive: [],
                negative: []
            }};
            for (const [uid, val] of Object.entries(labels)) {{
                if (val === 1) data.positive.push(uid);
                else if (val === 0) data.negative.push(uid);
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

        function clearAll() {{
            if (!confirm('Clear ALL annotations? This cannot be undone.')) return;
            localStorage.removeItem(STORAGE_KEY);
            updateCount();
        }}

        updateCount();
    </script>
</body>
</html>'''

    with open(html_dir / "index.html", 'w') as f:
        f.write(index_html)

    logger.info(f"\nDone! View at: {html_dir / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description='Regenerate HTML viewer with sorting')
    parser.add_argument('--output-dir', required=True, help='Path to segmentation output directory')
    parser.add_argument('--czi-path', help='Path to CZI file (auto-detected if not specified)')
    parser.add_argument('--channel', type=int, default=1, help='Channel to visualize (default: 1)')
    parser.add_argument('--sort-by', default='area', help='Feature to sort by (default: area)')
    parser.add_argument('--sort-order', choices=['asc', 'desc'], default='desc', help='Sort order (default: desc)')
    parser.add_argument('--crop-size', type=int, default=300, help='Crop size in pixels (default: 300)')
    parser.add_argument('--display-size', type=int, default=250, help='Display size in HTML (default: 250)')
    parser.add_argument('--contour-thickness', type=int, default=4, help='Contour thickness (default: 4)')
    parser.add_argument('--samples-per-page', type=int, default=300, help='Samples per page (default: 300)')
    parser.add_argument('--experiment-name', help='Experiment name for localStorage isolation (required)')
    parser.add_argument('--cell-type', choices=['mk', 'hspc'], help='Cell type for batch output structure')

    args = parser.parse_args()

    regenerate_html(
        output_dir=args.output_dir,
        czi_path=args.czi_path,
        channel=args.channel,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        crop_size=args.crop_size,
        display_size=args.display_size,
        contour_thickness=args.contour_thickness,
        samples_per_page=args.samples_per_page,
        experiment_name=args.experiment_name,
        cell_type=args.cell_type,
    )


if __name__ == '__main__':
    main()
