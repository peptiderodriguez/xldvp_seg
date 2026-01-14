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
from aicspylibczi import CziFile

# Import hdf5plugin for LZ4 decompression
try:
    import hdf5plugin
except ImportError:
    pass

from shared.html_export import (
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)


def load_channel_to_ram(czi_path, channel, strip_height=5000):
    """Load a single channel from CZI into RAM."""
    czi = CziFile(czi_path)

    # Get mosaic info
    bbox = czi.get_mosaic_scene_bounding_box()
    x_start, y_start = bbox.x, bbox.y
    width, height = bbox.w, bbox.h

    print(f"Loading channel {channel} ({width:,} x {height:,} px)...")

    n_strips = (height + strip_height - 1) // strip_height
    channel_data = np.empty((height, width), dtype=np.uint16)

    for i in tqdm(range(n_strips), desc="Loading"):
        y_off = i * strip_height
        h = min(strip_height, height - y_off)
        strip = czi.read_mosaic(
            region=(x_start, y_start + y_off, width, h),
            scale_factor=1,
            C=channel
        )
        channel_data[y_off:y_off+h, :] = np.squeeze(strip)

    return channel_data, (x_start, y_start, width, height)


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
        contour_color: RGB color for contour
        samples_per_page: Number of samples per HTML page
    """
    output_dir = Path(output_dir)
    html_dir = output_dir / "html"
    tiles_dir = output_dir / "tiles"

    # Find detections file
    det_files = list(output_dir.glob("*_detections.json"))
    if not det_files:
        raise FileNotFoundError(f"No detections file found in {output_dir}")
    det_file = det_files[0]
    cell_type = det_file.stem.replace('_detections', '').split('_')[-1]

    print(f"Loading detections from {det_file}")
    with open(det_file) as f:
        detections = json.load(f)

    if not detections:
        print("No detections to process")
        return

    print(f"Loaded {len(detections)} detections")

    # Sort detections
    reverse = (sort_order == 'desc')

    def get_sort_key(det):
        if sort_by in det.get('features', {}):
            return det['features'][sort_by]
        elif sort_by in det:
            return det[sort_by]
        return 0

    detections_sorted = sorted(detections, key=get_sort_key, reverse=reverse)
    print(f"Sorted by {sort_by} ({sort_order})")

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
        tile_x, tile_y = det['tile_origin']
        tile_key = f"tile_{tile_x}_{tile_y}"
        if tile_key not in tile_detections:
            tile_detections[tile_key] = []
        tile_detections[tile_key].append(det)

    print(f"Processing {len(tile_detections)} tiles...")

    # Generate samples
    samples = []

    for tile_key, tile_dets in tqdm(tile_detections.items(), desc="Processing"):
        tile_path = tiles_dir / tile_key
        mask_file = tile_path / f"{cell_type}_masks.h5"

        if not mask_file.exists():
            continue

        # Load masks
        with h5py.File(mask_file, 'r') as f:
            masks = f['masks'][:]

        # Parse tile origin
        parts = tile_key.split('_')
        tile_x, tile_y = int(parts[1]), int(parts[2])

        for det in tile_dets:
            det_id = int(det['id'].split('_')[-1])

            # Get mask
            mask = (masks == det_id)
            if not mask.any():
                continue

            # Find mask centroid for centering
            ys, xs = np.where(mask)
            cy, cx = np.mean(ys), np.mean(xs)

            # Global position
            global_cy = tile_y + cy
            global_cx = tile_x + cx

            # Extract crop centered on mask
            half = crop_size // 2
            y1 = max(0, int(global_cy - half))
            y2 = min(height, int(global_cy + half))
            x1 = max(0, int(global_cx - half))
            x2 = min(width, int(global_cx + half))

            crop = channel_data[y1:y2, x1:x2].copy()

            # Create mask for crop region
            crop_h, crop_w = y2 - y1, x2 - x1
            crop_mask = np.zeros((crop_h, crop_w), dtype=bool)

            # Calculate where mask falls in crop
            mask_offset_y = int(global_cy - half - tile_y)
            mask_offset_x = int(global_cx - half - tile_x)

            for my, mx in zip(ys, xs):
                cy_in_crop = my - mask_offset_y
                cx_in_crop = mx - mask_offset_x
                if 0 <= cy_in_crop < crop_h and 0 <= cx_in_crop < crop_w:
                    crop_mask[cy_in_crop, cx_in_crop] = True

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
            area = det['features'].get('area', 0)
            area_um2 = area * (pixel_size ** 2)
            elongation = det['features'].get('elongation', 0)

            samples.append({
                'uid': det['uid'],
                'image_b64': image_to_base64(crop_resized),
                'area': area,
                'area_um2': round(area_um2, 1),
                'elongation': elongation,
            })

    print(f"Generated {len(samples)} samples")

    # Generate HTML pages
    n_pages = (len(samples) + samples_per_page - 1) // samples_per_page

    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>{cell_type} Detection - Page {page}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 10px; }}
        .nav {{ background: #111; padding: 10px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 100; border: 1px solid #333; }}
        .nav a {{ color: #4a4; text-decoration: none; padding: 5px 15px; border: 1px solid #333; }}
        .nav a:hover {{ background: #1a1a1a; }}
        .stats {{ color: #888; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 10px; }}
        .card {{ background: #111; border: 1px solid #333; padding: 10px; text-align: center; }}
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
    <div class="nav">
        <div>
            {prev_link}
            <a href="index.html">Index</a>
            {next_link}
        </div>
        <div class="stats">Page {page}/{total_pages} | Sorted by {sort_by} ({sort_order})</div>
    </div>
    <div class="grid">
        {cards}
    </div>
    <script>
        const STORAGE_KEY = '{cell_type}_annotations';
        function getLabels() {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }}
        function setLabel(uid, val) {{
            const labels = getLabels();
            labels[uid] = val;
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
            updateButtons(uid);
        }}
        function updateButtons(uid) {{
            const labels = getLabels();
            document.querySelectorAll(`[data-uid="${{uid}}"] button`).forEach(b => b.classList.remove('selected'));
            const val = labels[uid];
            if (val === 1) document.querySelector(`[data-uid="${{uid}}"] .btn-yes`)?.classList.add('selected');
            if (val === 0) document.querySelector(`[data-uid="${{uid}}"] .btn-no`)?.classList.add('selected');
        }}
        document.querySelectorAll('.card').forEach(c => updateButtons(c.dataset.uid));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight') {{
                const next = document.querySelector('.nav a[href*="page_{next_page}"]');
                if (next) window.location = next.href;
            }} else if (e.key === 'ArrowLeft') {{
                const prev = document.querySelector('.nav a[href*="page_{prev_page}"]');
                if (prev) window.location = prev.href;
            }}
        }});
    </script>
</body>
</html>'''

    card_template = '''<div class="card" data-uid="{uid}">
    <img src="data:image/jpeg;base64,{image_b64}" alt="{uid}">
    <div class="info"><span class="area">{area_um2} µm²</span> | elong: {elongation:.2f}</div>
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
        )

        page_file = html_dir / f"{cell_type}_page_{page_num}.html"
        with open(page_file, 'w') as f:
            f.write(html)

        page_size_mb = len(html) / 1e6
        print(f"  Page {page_num}: {len(page_samples)} samples ({page_size_mb:.1f} MB)")

    # Update index
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
        .section {{ margin: 30px 0; text-align: center; }}
        .btn {{ padding: 15px 40px; background: #1a1a1a; border: 1px solid #4a4; color: #4a4; cursor: pointer; font-family: monospace; font-size: 1.1em; text-decoration: none; display: inline-block; margin: 10px; }}
        .btn:hover {{ background: #0f130f; }}
        .info {{ color: #888; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{cell_type.upper()} Detection Results</h1>
        <div class="stats">
            <div class="stat"><span>Total</span><span class="number">{len(samples):,}</span></div>
            <div class="stat"><span>Pages</span><span class="number">{n_pages}</span></div>
        </div>
        <p class="info">Sorted by {sort_by} ({sort_order})</p>
    </div>
    <div class="section">
        <a href="{cell_type}_page_1.html" class="btn">Start Review</a>
    </div>
</body>
</html>'''

    with open(html_dir / "index.html", 'w') as f:
        f.write(index_html)

    print(f"\nDone! View at: {html_dir / 'index.html'}")


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
    )


if __name__ == '__main__':
    main()
