#!/usr/bin/env python3
"""
Regenerate HTML for batch MK/HSPC output.
Handles the mk/ and hspc/ subdirectory structure from batch runs.
"""
import argparse
import json
import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from segmentation.io.czi_loader import CZILoader
from segmentation.io.html_export import (
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)
from segmentation.utils.logging import get_logger, setup_logging

setup_logging(level="INFO", console=True)
logger = get_logger(__name__)

try:
    import hdf5plugin
except ImportError:
    pass


def load_all_detections(output_dir, cell_type):
    """Load all detections from a cell type's tiles directory."""
    tiles_dir = output_dir / cell_type / "tiles"
    if not tiles_dir.exists():
        return []

    detections = []
    for tile_dir in tiles_dir.iterdir():
        if not tile_dir.is_dir():
            continue
        feat_file = tile_dir / "features.json"
        if feat_file.exists():
            with open(feat_file) as f:
                tile_dets = json.load(f)
            for det in tile_dets:
                det['tile_dir'] = str(tile_dir)
            detections.extend(tile_dets)

    return detections


def generate_html_page(samples, page_num, total_pages, cell_type, output_dir, experiment_name):
    """Generate a single HTML page."""

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{cell_type.upper()} Annotations - Page {page_num}/{total_pages}</title>
    <style>
        body {{ background: #1a1a2e; color: #eee; font-family: Arial; margin: 0; padding: 20px; }}
        .header {{ position: fixed; top: 0; left: 0; right: 0; background: #16213e; padding: 10px 20px; z-index: 1000; display: flex; justify-content: space-between; align-items: center; }}
        .stats {{ font-size: 14px; }}
        .grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 80px; justify-content: center; }}
        .card {{ background: #16213e; border-radius: 8px; padding: 10px; width: 270px; cursor: pointer; }}
        .card img {{ width: 250px; height: 250px; object-fit: contain; border-radius: 4px; }}
        .card.yes {{ border: 3px solid #4ade80; }}
        .card.no {{ border: 3px solid #f87171; }}
        .info {{ font-size: 11px; color: #aaa; margin-top: 5px; }}
        .buttons {{ display: flex; gap: 5px; margin-top: 8px; }}
        .btn {{ flex: 1; padding: 8px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }}
        .btn-yes {{ background: #4ade80; color: black; }}
        .btn-no {{ background: #f87171; color: black; }}
        .nav {{ margin-top: 20px; text-align: center; }}
        .nav a {{ color: #60a5fa; margin: 0 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <div><strong>{cell_type.upper()}</strong> - Page {page_num}/{total_pages} ({len(samples)} samples)</div>
        <div class="stats">
            <span id="yes-count">0</span> Yes | <span id="no-count">0</span> No | <span id="remaining">0</span> Remaining
        </div>
        <button onclick="exportAnnotations()">Export JSON</button>
    </div>
    <div class="grid">
'''

    for sample in samples:
        uid = sample.get('uid', sample.get('id', 'unknown'))
        area = sample.get('features', {}).get('area', 0)
        area_um = area * (0.1725 ** 2)  # Convert to µm²

        html += f'''
        <div class="card" id="card-{uid}" data-uid="{uid}">
            <img src="data:image/png;base64,{sample['image_b64']}" alt="{uid}">
            <div class="info">{uid}<br>Area: {area_um:.0f} µm²</div>
            <div class="buttons">
                <button class="btn btn-yes" onclick="annotate('{uid}', 'yes')">Yes (Y)</button>
                <button class="btn btn-no" onclick="annotate('{uid}', 'no')">No (N)</button>
            </div>
        </div>
'''

    html += f'''
    </div>
    <div class="nav">
'''
    if page_num > 1:
        html += f'        <a href="{cell_type}_page_{page_num-1}.html">← Previous</a>\n'
    if page_num < total_pages:
        html += f'        <a href="{cell_type}_page_{page_num+1}.html">Next →</a>\n'

    html += f'''
    </div>
    <script>
        const STORAGE_KEY = '{experiment_name}_{cell_type}_annotations';

        function getAnnotations() {{
            return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
        }}

        function saveAnnotations(anns) {{
            localStorage.setItem(STORAGE_KEY, JSON.stringify(anns));
        }}

        function annotate(uid, value) {{
            const anns = getAnnotations();
            anns[uid] = value;
            saveAnnotations(anns);
            updateCard(uid, value);
            updateStats();
        }}

        function updateCard(uid, value) {{
            const card = document.getElementById('card-' + uid);
            if (card) {{
                card.classList.remove('yes', 'no');
                if (value) card.classList.add(value);
            }}
        }}

        function updateStats() {{
            const anns = getAnnotations();
            const uids = Array.from(document.querySelectorAll('.card')).map(c => c.dataset.uid);
            let yes = 0, no = 0;
            uids.forEach(uid => {{
                if (anns[uid] === 'yes') yes++;
                else if (anns[uid] === 'no') no++;
            }});
            document.getElementById('yes-count').textContent = yes;
            document.getElementById('no-count').textContent = no;
            document.getElementById('remaining').textContent = uids.length - yes - no;
        }}

        function exportAnnotations() {{
            const anns = getAnnotations();
            const blob = new Blob([JSON.stringify(anns, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = '{cell_type}_annotations.json';
            a.click();
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            const anns = getAnnotations();
            Object.entries(anns).forEach(([uid, value]) => updateCard(uid, value));
            updateStats();
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            const focused = document.querySelector('.card:hover');
            if (!focused) return;
            const uid = focused.dataset.uid;
            if (e.key === 'y' || e.key === 'Y') annotate(uid, 'yes');
            if (e.key === 'n' || e.key === 'N') annotate(uid, 'no');
        }});
    </script>
</body>
</html>
'''
    return html


def generate_index(html_dir, cell_types, experiment_name, total_counts):
    """Generate index.html."""
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{experiment_name} - Detection Results</title>
    <style>
        body {{ background: #1a1a2e; color: #eee; font-family: Arial; padding: 40px; }}
        h1 {{ color: #60a5fa; }}
        .section {{ margin: 30px 0; }}
        a {{ color: #60a5fa; font-size: 18px; }}
        .count {{ color: #aaa; }}
    </style>
</head>
<body>
    <h1>{experiment_name}</h1>
'''
    for ct in cell_types:
        pages = list(html_dir.glob(f"{ct}_page_*.html"))
        html += f'''
    <div class="section">
        <h2>{ct.upper()} Detections <span class="count">({total_counts.get(ct, 0)} total)</span></h2>
        <a href="{ct}_page_1.html">View {ct.upper()} annotations →</a>
    </div>
'''
    html += '</body></html>'
    return html


def regenerate_batch_html(
    output_base,
    czi_base,
    slides,
    channel=0,
    crop_size=300,
    display_size=250,
    samples_per_page=300,
):
    """Regenerate HTML for batch output."""

    output_base = Path(output_base)
    czi_base = Path(czi_base)

    # Collect all detections across slides
    all_mk = []
    all_hspc = []

    for slide in slides:
        slide_dir = output_base / slide
        if not slide_dir.exists():
            logger.warning(f"Slide dir not found: {slide_dir}")
            continue

        mk_dets = load_all_detections(slide_dir, 'mk')
        hspc_dets = load_all_detections(slide_dir, 'hspc')

        for det in mk_dets:
            det['slide'] = slide
        for det in hspc_dets:
            det['slide'] = slide

        all_mk.extend(mk_dets)
        all_hspc.extend(hspc_dets)

        logger.info(f"{slide}: {len(mk_dets)} MKs, {len(hspc_dets)} HSPCs")

    logger.info(f"Total: {len(all_mk)} MKs, {len(all_hspc)} HSPCs")

    # Sort by area (ascending - smallest first)
    all_mk.sort(key=lambda x: x.get('features', {}).get('area', 0))
    all_hspc.sort(key=lambda x: x.get('features', {}).get('area', 0))

    # Create HTML directory
    html_dir = output_base / "html"
    html_dir.mkdir(exist_ok=True)

    # Load CZI files and generate crops
    logger.info("Loading CZI files for crop generation...")

    slide_loaders = {}
    for slide in slides:
        czi_path = czi_base / f"{slide}.czi"
        if czi_path.exists():
            logger.info(f"Loading {slide}...")
            loader = CZILoader(str(czi_path), load_to_ram=True, channel=channel)
            slide_loaders[slide] = loader

    def generate_crop(det, cell_type):
        """Generate crop image for a detection."""
        slide = det['slide']
        if slide not in slide_loaders:
            return None

        loader = slide_loaders[slide]
        center = det.get('center', [0, 0])
        cx, cy = int(center[0]), int(center[1])

        # Convert global coordinates to local array coordinates
        local_cx = cx - loader.x_start
        local_cy = cy - loader.y_start

        # Check if detection center is within the loaded region
        if local_cx < 0 or local_cx >= loader.width or local_cy < 0 or local_cy >= loader.height:
            return None

        # Get crop bounds in local coordinates
        half = crop_size // 2
        x1 = max(0, local_cx - half)
        y1 = max(0, local_cy - half)
        x2 = min(loader.width, local_cx + half)
        y2 = min(loader.height, local_cy + half)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract crop
        crop = loader.channel_data[y1:y2, x1:x2].copy()

        # Load mask
        tile_dir = Path(det['tile_dir'])
        seg_file = tile_dir / "segmentation.h5"
        if seg_file.exists():
            with h5py.File(seg_file, 'r') as f:
                masks = f['labels'][:]
                # Handle 3D masks with shape (1, H, W)
                if masks.ndim == 3 and masks.shape[0] == 1:
                    masks = masks[0]

            # Get mask for this detection
            det_id = det.get('id', 1)
            if isinstance(det_id, str) and det_id.startswith('det_'):
                det_id = int(det_id.replace('det_', ''))

            mask = (masks == det_id).astype(np.uint8)

            # Get tile window to convert global coords to tile coords
            window_file = tile_dir / "window.csv"
            if window_file.exists():
                import re
                with open(window_file) as f:
                    window_str = f.read()
                matches = re.findall(r'slice\((\d+),\s*(\d+)', window_str)
                if len(matches) >= 2:
                    tile_y1, tile_y2 = int(matches[0][0]), int(matches[0][1])
                    tile_x1, tile_x2 = int(matches[1][0]), int(matches[1][1])

                    # Crop mask to same region as image
                    mask_x1 = max(0, x1 + loader.x_start - tile_x1)
                    mask_y1 = max(0, y1 + loader.y_start - tile_y1)
                    mask_x2 = mask_x1 + (x2 - x1)
                    mask_y2 = mask_y1 + (y2 - y1)

                    if mask_y2 <= mask.shape[0] and mask_x2 <= mask.shape[1]:
                        mask_crop = mask[mask_y1:mask_y2, mask_x1:mask_x2]
                    else:
                        mask_crop = None
                else:
                    mask_crop = None
            else:
                mask_crop = None
        else:
            mask_crop = None

        # Normalize crop
        if len(crop.shape) == 2:
            crop = np.stack([crop]*3, axis=-1)
        crop = percentile_normalize(crop)

        # Draw mask contour
        if mask_crop is not None and mask_crop.shape[:2] == crop.shape[:2]:
            crop = draw_mask_contour(crop, mask_crop, color=(255, 255, 255), thickness=3)

        # Resize
        crop = cv2.resize(crop, (display_size, display_size))

        return image_to_base64(crop)

    # Generate crops and pages for MK
    logger.info("Generating MK HTML pages...")
    mk_samples = []
    for det in tqdm(all_mk, desc="MK crops"):
        b64 = generate_crop(det, 'mk')
        if b64:
            det['image_b64'] = b64
            mk_samples.append(det)

    # Generate MK pages
    num_mk_pages = (len(mk_samples) + samples_per_page - 1) // samples_per_page
    for i in range(num_mk_pages):
        start = i * samples_per_page
        end = start + samples_per_page
        page_samples = mk_samples[start:end]
        html = generate_html_page(page_samples, i+1, num_mk_pages, 'mk', output_base, output_base.name)
        with open(html_dir / f"mk_page_{i+1}.html", 'w') as f:
            f.write(html)
    logger.info(f"Generated {num_mk_pages} MK pages")

    # Generate crops and pages for HSPC
    logger.info("Generating HSPC HTML pages...")
    hspc_samples = []
    for det in tqdm(all_hspc, desc="HSPC crops"):
        b64 = generate_crop(det, 'hspc')
        if b64:
            det['image_b64'] = b64
            hspc_samples.append(det)

    # Generate HSPC pages
    num_hspc_pages = (len(hspc_samples) + samples_per_page - 1) // samples_per_page
    for i in range(num_hspc_pages):
        start = i * samples_per_page
        end = start + samples_per_page
        page_samples = hspc_samples[start:end]
        html = generate_html_page(page_samples, i+1, num_hspc_pages, 'hspc', output_base, output_base.name)
        with open(html_dir / f"hspc_page_{i+1}.html", 'w') as f:
            f.write(html)
    logger.info(f"Generated {num_hspc_pages} HSPC pages")

    # Generate index
    index_html = generate_index(html_dir, ['mk', 'hspc'], output_base.name,
                                {'mk': len(mk_samples), 'hspc': len(hspc_samples)})
    with open(html_dir / "index.html", 'w') as f:
        f.write(index_html)

    logger.info(f"HTML export complete: {html_dir}")

    # Cleanup
    for loader in slide_loaders.values():
        loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Regenerate HTML for batch MK/HSPC output')
    parser.add_argument('--output-base', required=True, help='Base output directory')
    parser.add_argument('--czi-base', required=True, help='Base CZI directory')
    parser.add_argument('--slides', nargs='+', help='Slide names (auto-detected if not specified)')
    parser.add_argument('--channel', type=int, default=0, help='Channel to visualize')
    parser.add_argument('--crop-size', type=int, default=300, help='Crop size in pixels')
    parser.add_argument('--samples-per-page', type=int, default=300, help='Samples per page')

    args = parser.parse_args()

    # Auto-detect slides if not specified
    if args.slides:
        slides = args.slides
    else:
        output_base = Path(args.output_base)
        slides = sorted([d.name for d in output_base.iterdir()
                        if d.is_dir() and d.name.startswith('2025_11_18')])

    regenerate_batch_html(
        args.output_base,
        args.czi_base,
        slides,
        channel=args.channel,
        crop_size=args.crop_size,
        samples_per_page=args.samples_per_page,
    )
