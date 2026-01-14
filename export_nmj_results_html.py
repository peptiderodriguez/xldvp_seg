#!/usr/bin/env python3
"""
Export NMJ inference results to HTML for visualization.
Shows classified NMJs with confidence scores and morphological features.
"""

import argparse
import json
import base64
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm
from aicspylibczi import CziFile
import h5py
import hdf5plugin  # Required for reading compressed HDF5 masks
from skimage import measure


def percentile_normalize(img, p_low=5, p_high=95):
    """Normalize image using percentiles."""
    img = img.astype(np.float32)
    p_lo = np.percentile(img, p_low)
    p_hi = np.percentile(img, p_high)
    if p_hi - p_lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - p_lo) / (p_hi - p_lo)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def draw_mask_contour(img, mask, color=(144, 238, 144), dotted=True):
    """Draw mask contour on image with dotted line."""
    # Find contours
    contours = measure.find_contours(mask, 0.5)

    draw = ImageDraw.Draw(img)

    for contour in contours:
        # Convert to list of (x, y) tuples
        points = [(int(p[1]), int(p[0])) for p in contour]

        if dotted:
            # Draw dotted line
            for i in range(0, len(points) - 1, 2):
                if i + 1 < len(points):
                    draw.line([points[i], points[i + 1]], fill=color, width=2)
        else:
            # Draw solid line
            if len(points) > 1:
                draw.line(points, fill=color, width=2)

    return img


def extract_crop_with_mask(reader, tile_x, tile_y, centroid, nmj_id, channel,
                           seg_dir, tile_size=3000, crop_size=300):
    """Extract crop from CZI and overlay mask contour."""
    # Read tile
    tile_data = reader.read_mosaic(
        region=(tile_x, tile_y, tile_size, tile_size),
        scale_factor=1,
        C=channel
    )

    if tile_data is None or tile_data.size == 0:
        return None

    tile_data = np.squeeze(tile_data)
    if tile_data.ndim != 2:
        return None

    # Extract crop centered on centroid
    # Note: centroid stored as [row, col] but actual mask uses [y, x] = [row, col]
    # The features file stores centroid as [col, row] = [x, y], so swap
    cx, cy = int(centroid[0]), int(centroid[1])
    half = crop_size // 2

    h, w = tile_data.shape
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)

    crop = tile_data[y1:y2, x1:x2].copy()

    # Normalize
    crop = percentile_normalize(crop)

    # Convert to RGB
    crop_rgb = np.stack([crop] * 3, axis=-1)

    # Load mask and extract corresponding region
    mask_crop = None
    tile_dir = seg_dir / f"tile_{tile_x}_{tile_y}"
    mask_file = tile_dir / "nmj_masks.h5"

    if mask_file.exists():
        try:
            # Extract label number from nmj_id (e.g., "nmj_1" -> 1 or "225000_75000_nmj_1" -> 1)
            label_num = int(nmj_id.split('_')[-1])

            with h5py.File(mask_file, 'r') as f:
                # Masks stored as single labeled image
                if 'masks' in f:
                    full_masks = f['masks'][:]
                    # Extract crop region and create binary mask for this label
                    mask_region = full_masks[y1:y2, x1:x2]
                    mask_crop = (mask_region == label_num).astype(np.uint8)

                    # Debug: check if mask was found
                    if mask_crop.sum() == 0:
                        # Try finding where the label actually is in full image
                        label_coords = np.where(full_masks == label_num)
                        if len(label_coords[0]) > 0:
                            actual_y = label_coords[0].mean()
                            actual_x = label_coords[1].mean()
                            print(f"  DEBUG {nmj_id}: mask at y={actual_y:.0f}, x={actual_x:.0f}, but crop y={y1}:{y2}, x={x1}:{x2}")
        except Exception as e:
            print(f"  Error loading mask for {nmj_id}: {e}")

    # Resize crop
    if crop_rgb.shape[0] > 0 and crop_rgb.shape[1] > 0:
        pil_img = Image.fromarray(crop_rgb)
        pil_img = pil_img.resize((crop_size, crop_size), Image.LANCZOS)

        # Resize and overlay mask if available
        if mask_crop is not None and mask_crop.size > 0:
            # Resize mask to match crop size
            mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((crop_size, crop_size), Image.NEAREST)
            mask_resized = np.array(mask_pil) > 127

            # Draw contour on image
            pil_img = draw_mask_contour(pil_img, mask_resized,
                                        color=(144, 238, 144), dotted=True)

        return pil_img

    return None


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def generate_html_page(nmjs, page_num, total_pages, slide_name):
    """Generate HTML page for NMJ results with annotation support."""

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>NMJ Annotation - {slide_name} - Page {page_num}/{total_pages}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 15px;
            background: #fff;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 5px 0;
            font-size: 1.5em;
            color: #333;
        }}
        .header p {{
            margin: 0;
            color: #666;
        }}
        .stats-bar {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            padding: 12px 20px;
            background: #fff;
            border: 1px solid #ddd;
            margin-bottom: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
            padding: 8px 15px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: 600;
            color: #2e7d32;
        }}
        .stat-value.negative {{
            color: #c62828;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.85em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            padding: 10px;
        }}
        .card {{
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}
        .card.annotated-yes {{
            border: 2px solid #2e7d32;
        }}
        .card.annotated-no {{
            border: 2px solid #c62828;
        }}
        .card img {{
            width: 100%;
            height: 280px;
            object-fit: contain;
            background: #000;
        }}
        .card-info {{
            padding: 12px;
        }}
        .confidence {{
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .confidence.high {{
            color: #2e7d32;
        }}
        .confidence.medium {{
            color: #f9a825;
        }}
        .confidence.low {{
            color: #c62828;
        }}
        .features {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
            font-size: 0.8em;
            color: #666;
        }}
        .feature {{
            padding: 3px 0;
        }}
        .annotation-buttons {{
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }}
        .annotation-buttons button {{
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9em;
            cursor: pointer;
            background: #fff;
        }}
        .btn-yes {{
            color: #2e7d32;
            border-color: #2e7d32;
        }}
        .btn-yes:hover, .btn-yes.active {{
            background: #2e7d32;
            color: #fff;
        }}
        .btn-no {{
            color: #c62828;
            border-color: #c62828;
        }}
        .btn-no:hover, .btn-no.active {{
            background: #c62828;
            color: #fff;
        }}
        .annotation-status {{
            position: absolute;
            top: 8px;
            right: 8px;
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: 600;
            font-size: 0.8em;
        }}
        .annotation-status.yes {{
            background: #2e7d32;
            color: #fff;
        }}
        .annotation-status.no {{
            background: #c62828;
            color: #fff;
        }}
        .pagination {{
            display: flex;
            justify-content: center;
            gap: 5px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .pagination a {{
            padding: 8px 12px;
            background: #fff;
            color: #333;
            text-decoration: none;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        .pagination a:hover {{
            background: #f0f0f0;
        }}
        .pagination a.current {{
            background: #333;
            color: #fff;
            border-color: #333;
        }}
        .nav-buttons {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 15px 0;
        }}
        .nav-buttons a {{
            padding: 10px 25px;
            background: #fff;
            color: #333;
            text-decoration: none;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .nav-buttons a:hover {{
            background: #f0f0f0;
        }}
        .export-btn {{
            padding: 8px 20px;
            background: #333;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        .export-btn:hover {{
            background: #555;
        }}
        .clear-btn {{
            padding: 8px 15px;
            background: #fff;
            color: #666;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }}
        .clear-btn:hover {{
            background: #f0f0f0;
            color: #c62828;
            border-color: #c62828;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NMJ Annotation - {slide_name}</h1>
        <p>Page {page_num} of {total_pages}</p>
    </div>

    <div class="stats-bar">
        <div class="stat">
            <div class="stat-value">{len(nmjs)}</div>
            <div class="stat-label">This page</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="page-yes">0</div>
            <div class="stat-label">Yes (page)</div>
        </div>
        <div class="stat">
            <div class="stat-value negative" id="page-no">0</div>
            <div class="stat-label">No (page)</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="global-yes">0</div>
            <div class="stat-label">Yes (all)</div>
        </div>
        <div class="stat">
            <div class="stat-value negative" id="global-no">0</div>
            <div class="stat-label">No (all)</div>
        </div>
        <button class="export-btn" onclick="exportAnnotations()">Export</button>
        <button class="clear-btn" onclick="clearPage()">Clear Page</button>
        <button class="clear-btn" onclick="clearAll()">Clear All</button>
    </div>

    <div class="nav-buttons">
'''

    if page_num > 1:
        html += f'        <a href="nmj_results_page_{page_num-1}.html">&larr; Previous</a>\n'
    if page_num < total_pages:
        html += f'        <a href="nmj_results_page_{page_num+1}.html">Next &rarr;</a>\n'

    html += '''    </div>

    <div class="grid">
'''

    for i, nmj in enumerate(nmjs):
        conf = nmj['confidence']
        conf_class = 'high' if conf >= 0.8 else ('medium' if conf >= 0.6 else 'low')
        # Create unique ID for this NMJ across all pages
        unique_id = f"{nmj['tile_x']}_{nmj['tile_y']}_{nmj['id']}"

        html += f'''        <div class="card" data-id="{unique_id}" data-index="{i}">
            <div class="annotation-status" style="display:none;"></div>
            <img src="data:image/png;base64,{nmj['image_b64']}" alt="NMJ">
            <div class="card-info">
                <div class="confidence {conf_class}">Confidence: {conf*100:.1f}%</div>
                <div class="features">
                    <div class="feature">Area: {nmj['area_um2']:.1f} &mu;m&sup2;</div>
                    <div class="feature">Elongation: {nmj['elongation']:.2f}</div>
                    <div class="feature">Skeleton: {nmj['skeleton_length']} px</div>
                    <div class="feature">ID: {nmj['id']}</div>
                </div>
                <div class="annotation-buttons">
                    <button class="btn-yes" onclick="annotate('{unique_id}', 'yes', {i})">Yes</button>
                    <button class="btn-no" onclick="annotate('{unique_id}', 'no', {i})">No</button>
                </div>
            </div>
        </div>
'''

    html += '''    </div>

    <div class="pagination">
'''

    # Show pagination links
    for p in range(1, total_pages + 1):
        if p == page_num:
            html += f'        <a href="nmj_results_page_{p}.html" class="current">{p}</a>\n'
        else:
            html += f'        <a href="nmj_results_page_{p}.html">{p}</a>\n'

    html += '''    </div>

    <script>
        const STORAGE_KEY = 'nmj_annotations';
        const cards = document.querySelectorAll('.card');

        function getAnnotations() {
            const stored = localStorage.getItem(STORAGE_KEY);
            return stored ? JSON.parse(stored) : {};
        }

        function saveAnnotations(annotations) {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
        }

        function annotate(id, value, cardIndex) {
            const annotations = getAnnotations();
            // Toggle off if clicking same value
            if (annotations[id] === value) {
                delete annotations[id];
                saveAnnotations(annotations);
                clearCardUI(cardIndex);
            } else {
                annotations[id] = value;
                saveAnnotations(annotations);
                updateCardUI(cardIndex, value);
            }
            updateStats();
        }

        function clearCardUI(index) {
            const card = cards[index];
            card.classList.remove('annotated-yes', 'annotated-no');
            const status = card.querySelector('.annotation-status');
            status.style.display = 'none';
            const yesBtn = card.querySelector('.btn-yes');
            const noBtn = card.querySelector('.btn-no');
            yesBtn.classList.remove('active');
            noBtn.classList.remove('active');
        }

        function updateCardUI(index, value) {
            const card = cards[index];
            card.classList.remove('annotated-yes', 'annotated-no');
            card.classList.add('annotated-' + value);

            const status = card.querySelector('.annotation-status');
            status.style.display = 'block';
            status.className = 'annotation-status ' + value;
            status.textContent = value.toUpperCase();

            const yesBtn = card.querySelector('.btn-yes');
            const noBtn = card.querySelector('.btn-no');
            yesBtn.classList.toggle('active', value === 'yes');
            noBtn.classList.toggle('active', value === 'no');
        }

        function updateStats() {
            const annotations = getAnnotations();
            let globalYes = 0, globalNo = 0, pageYes = 0, pageNo = 0;

            for (const [id, val] of Object.entries(annotations)) {
                if (val === 'yes') globalYes++;
                else if (val === 'no') globalNo++;
            }

            cards.forEach(card => {
                const id = card.dataset.id;
                if (annotations[id] === 'yes') pageYes++;
                else if (annotations[id] === 'no') pageNo++;
            });

            document.getElementById('global-yes').textContent = globalYes;
            document.getElementById('global-no').textContent = globalNo;
            document.getElementById('page-yes').textContent = pageYes;
            document.getElementById('page-no').textContent = pageNo;
        }

        function exportAnnotations() {
            const annotations = getAnnotations();
            const data = {
                exported_at: new Date().toISOString(),
                total_yes: Object.values(annotations).filter(v => v === 'yes').length,
                total_no: Object.values(annotations).filter(v => v === 'no').length,
                annotations: annotations
            };

            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'nmj_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
        }

        function clearPage() {
            if (!confirm('Clear all annotations on this page?')) return;
            const annotations = getAnnotations();
            cards.forEach((card, i) => {
                const id = card.dataset.id;
                if (annotations[id]) {
                    delete annotations[id];
                    clearCardUI(i);
                }
            });
            saveAnnotations(annotations);
            updateStats();
        }

        function clearAll() {
            if (!confirm('Clear ALL annotations across all pages?')) return;
            localStorage.removeItem(STORAGE_KEY);
            cards.forEach((card, i) => clearCardUI(i));
            updateStats();
        }

        // Initialize
        (function() {
            const annotations = getAnnotations();
            cards.forEach((card, i) => {
                const id = card.dataset.id;
                if (annotations[id]) {
                    updateCardUI(i, annotations[id]);
                }
            });
            updateStats();
        })();
    </script>
</body>
</html>
'''

    return html


def generate_index_html(displayed_nmjs, total_nmjs, total_candidates, slide_name, total_pages, pixel_size, min_area=0, min_confidence=0):
    """Generate index page with summary."""

    # Build filter info text
    filter_parts = []
    if min_area > 0:
        filter_parts.append(f"area &ge; {min_area} &mu;m&sup2;")
    if min_confidence > 0:
        filter_parts.append(f"confidence &ge; {min_confidence:.0%}")
    filter_text = " and ".join(filter_parts) if filter_parts else ""

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>NMJ Results - {slide_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 40px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00ff88;
            margin-bottom: 40px;
        }}
        .summary {{
            background: #16213e;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        .summary h2 {{
            margin-top: 0;
            color: #fff;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-box {{
            background: #0f3460;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #00ff88;
        }}
        .stat-box .label {{
            color: #aaa;
            margin-top: 10px;
        }}
        .start-btn {{
            display: block;
            text-align: center;
            padding: 20px 40px;
            background: #00ff88;
            color: #000;
            text-decoration: none;
            border-radius: 10px;
            font-size: 1.3em;
            font-weight: bold;
            margin: 40px auto;
            max-width: 300px;
        }}
        .start-btn:hover {{
            background: #00cc6a;
        }}
        .info {{
            color: #888;
            text-align: center;
            margin-top: 20px;
        }}
        .filter-note {{
            color: #f9a825;
            text-align: center;
            margin-top: 15px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NMJ Classification Results</h1>

        <div class="summary">
            <h2>{slide_name}</h2>

            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{displayed_nmjs}</div>
                    <div class="label">NMJs Displayed</div>
                </div>
                <div class="stat-box">
                    <div class="value">{total_nmjs}</div>
                    <div class="label">Total Classified</div>
                </div>
                <div class="stat-box">
                    <div class="value">{total_candidates}</div>
                    <div class="label">Initial Candidates</div>
                </div>
                <div class="stat-box">
                    <div class="value">{total_pages}</div>
                    <div class="label">Pages</div>
                </div>
            </div>

            <p class="info">Pixel size: {pixel_size:.4f} um/px</p>
            {f'<p class="filter-note">Filtered: {filter_text} ({total_nmjs - displayed_nmjs} hidden)</p>' if filter_text else ''}
        </div>

        <a href="nmj_results_page_1.html" class="start-btn">View Results &rarr;</a>
    </div>
</body>
</html>
'''

    return html


def main():
    parser = argparse.ArgumentParser(description='Export NMJ results to HTML')
    parser.add_argument('--results-json', type=str, required=True,
                        help='Path to nmj_detections.json')
    parser.add_argument('--czi-path', type=str, required=True,
                        help='Path to CZI file')
    parser.add_argument('--segmentation-dir', type=str, default=None,
                        help='Directory with segmentation tiles (for mask overlay)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for HTML files')
    parser.add_argument('--channel', type=int, default=1,
                        help='Channel index')
    parser.add_argument('--per-page', type=int, default=50,
                        help='NMJs per page')
    parser.add_argument('--min-area', type=float, default=0,
                        help='Minimum area in um^2 to display')
    parser.add_argument('--min-confidence', type=float, default=0,
                        help='Minimum confidence to display')
    args = parser.parse_args()

    # Load results
    print("Loading results...")
    with open(args.results_json) as f:
        results = json.load(f)

    nmjs = results['nmjs']
    print(f"Total NMJs: {len(nmjs)}")

    if not nmjs:
        print("No NMJs to export!")
        return

    # Filter by area and confidence (but don't delete from results)
    original_count = len(nmjs)
    if args.min_area > 0 or args.min_confidence > 0:
        nmjs = [n for n in nmjs if n['area_um2'] >= args.min_area and n['confidence'] >= args.min_confidence]
        print(f"After filtering (area >= {args.min_area} um², conf >= {args.min_confidence}): {len(nmjs)} ({original_count - len(nmjs)} filtered out)")

    # Sort by area (largest first), then by confidence (highest first)
    nmjs = sorted(nmjs, key=lambda x: (x['area_um2'], x['confidence']), reverse=True)

    # Setup output
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.results_json).parent / "html"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find segmentation directory for mask overlay
    if args.segmentation_dir:
        seg_dir = Path(args.segmentation_dir)
    else:
        # Auto-detect from results path
        seg_dir = Path(args.results_json).parent.parent / "tiles"
    print(f"Segmentation directory: {seg_dir}")

    # Load CZI
    print("Loading CZI...")
    reader = CziFile(str(args.czi_path))

    # Extract crops and add base64 images
    print("Extracting crops with mask overlay...")
    for nmj in tqdm(nmjs, desc="Extracting"):
        crop = extract_crop_with_mask(
            reader,
            nmj['tile_x'],
            nmj['tile_y'],
            nmj['local_centroid'],
            nmj['id'],
            args.channel,
            seg_dir
        )
        if crop:
            nmj['image_b64'] = image_to_base64(crop)
        else:
            nmj['image_b64'] = ''

    # Filter out any without images
    nmjs = [n for n in nmjs if n['image_b64']]
    print(f"NMJs with valid crops: {len(nmjs)}")

    # Generate pages
    per_page = args.per_page
    total_pages = (len(nmjs) + per_page - 1) // per_page

    print(f"Generating {total_pages} HTML pages...")
    for page_num in range(1, total_pages + 1):
        start_idx = (page_num - 1) * per_page
        end_idx = start_idx + per_page
        page_nmjs = nmjs[start_idx:end_idx]

        html = generate_html_page(page_nmjs, page_num, total_pages, results['slide_name'])

        output_file = output_dir / f"nmj_results_page_{page_num}.html"
        with open(output_file, 'w') as f:
            f.write(html)

    # Generate index
    index_html = generate_index_html(
        len(nmjs),  # displayed count (after filtering)
        results['total_nmjs'],  # total classified
        results['total_candidates'],
        results['slide_name'],
        total_pages,
        results['pixel_size_um'],
        min_area=args.min_area,
        min_confidence=args.min_confidence
    )

    index_file = output_dir / "index.html"
    with open(index_file, 'w') as f:
        f.write(index_html)

    print(f"\nHTML export complete!")
    print(f"Output directory: {output_dir}")
    print(f"Open {index_file} to view results")


if __name__ == '__main__':
    main()
