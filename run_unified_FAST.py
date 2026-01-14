"""
Unified segmentation: MKs (SAM2) + HSPCs (Cellpose-SAM) in one pass.

Processes each tile once and outputs:
- MK masks (SAM2 automatic mask generation, size-filtered by mk-min/max-area)
- HSPC masks (Cellpose-SAM detection + SAM2 refinement, size-invariant)
- All features: 22 custom + 256 SAM2 + 2048 ResNet = 2326 per cell

Usage:
    python run_unified_segmentation.py \
        --czi-path /path/to/slide.czi \
        --output-dir /path/to/output \
        --mk-min-area 1000
"""

# Set cellpose model path FIRST, before any imports (cellpose caches this at import time)
import os
from pathlib import Path

# Auto-detect checkpoint directory (local or cluster)
_script_dir = Path(__file__).parent.resolve()
_checkpoint_candidates = [
    _script_dir / "checkpoints",  # Local: same dir as script
    Path.home() / ".cache" / "cellpose",  # User cache
    Path("/ptmp/edrod/MKsegmentation/checkpoints"),  # Cluster path
]
for _cp in _checkpoint_candidates:
    if _cp.exists():
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(_cp)
        break
else:
    # Default to local checkpoints dir (will be created)
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(_script_dir / "checkpoints")

import gc
import numpy as np
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
import torch.multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import psutil
import base64
from io import BytesIO
import re

# Try to use LZ4 compression (faster than gzip), fallback to gzip
try:
    import hdf5plugin
    # LZ4 is ~3-5x faster than gzip with similar compression ratio for image masks
    HDF5_COMPRESSION_KWARGS = hdf5plugin.LZ4(nbytes=0)  # Returns dict-like for **unpacking
    HDF5_COMPRESSION_NAME = "LZ4"
except ImportError:
    HDF5_COMPRESSION_KWARGS = {'compression': 'gzip'}
    HDF5_COMPRESSION_NAME = "gzip"


def create_hdf5_dataset(f, name, data):
    """Create HDF5 dataset with best available compression (LZ4 or gzip)."""
    if isinstance(HDF5_COMPRESSION_KWARGS, dict):
        f.create_dataset(name, data=data, **HDF5_COMPRESSION_KWARGS)
    else:
        # hdf5plugin filter object
        f.create_dataset(name, data=data, **HDF5_COMPRESSION_KWARGS)


# =============================================================================
# HTML EXPORT FUNCTIONS (integrated to use slide data in RAM)
# =============================================================================

def get_largest_connected_component(mask):
    """Extract only the largest connected component from a binary mask."""
    from scipy import ndimage
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    # Find largest component
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def draw_mask_contour(img_array, mask, color=(144, 238, 144), dotted=True):
    """
    Draw mask contour on image as dotted light green line.

    Args:
        img_array: numpy array (H, W, 3) uint8
        mask: binary mask (H, W)
        color: RGB tuple for contour color (default: light green)
        dotted: if True, draw dotted line

    Returns:
        img_array with contour drawn
    """
    from scipy import ndimage

    # Find contour by detecting edges of the mask
    # Dilate then subtract original to get the outline (6px thick)
    dilated = ndimage.binary_dilation(mask, iterations=6)
    contour = dilated & ~mask

    # Get contour pixel coordinates
    ys, xs = np.where(contour)

    if len(ys) == 0:
        return img_array

    img_out = img_array.copy()

    if dotted:
        # Draw every other pixel for dotted effect
        for i, (y, x) in enumerate(zip(ys, xs)):
            if i % 2 == 0:  # Dotted pattern
                if 0 <= y < img_out.shape[0] and 0 <= x < img_out.shape[1]:
                    img_out[y, x] = color
    else:
        for y, x in zip(ys, xs):
            if 0 <= y < img_out.shape[0] and 0 <= x < img_out.shape[1]:
                img_out[y, x] = color

    return img_out


def load_samples_from_ram(tiles_dir, slide_image, pixel_size_um, cell_type='mk', max_samples=None):
    """
    Load cell samples from segmentation output, using in-memory slide image.

    Args:
        tiles_dir: Path to tiles directory (e.g., output/mk/tiles)
        slide_image: numpy array of full slide image (already in RAM)
        pixel_size_um: Pixel size in microns
        cell_type: 'mk' or 'hspc' - affects mask selection
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
        except:
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

            try:
                cell_idx = int(det_id.split('_')[1]) + 1
            except:
                continue

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
            # Crop size should be at least mask size + padding, minimum 300px
            crop_size = max(300, max(mask_h, mask_w) + 100)
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
            except Exception:
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
            original_size = pil_img.size
            pil_img = pil_img.resize((300, 300), Image.LANCZOS)
            crop_resized = np.array(pil_img)

            # Resize mask to match
            if local_mask.shape[0] > 0 and local_mask.shape[1] > 0:
                mask_pil = Image.fromarray(local_mask.astype(np.uint8) * 255)
                mask_pil = mask_pil.resize((300, 300), Image.NEAREST)
                mask_resized = np.array(mask_pil) > 127

                # Draw solid bright green contour on the image (6px thick)
                crop_with_contour = draw_mask_contour(crop_resized, mask_resized,
                                                       color=(0, 255, 0), dotted=False)
            else:
                crop_with_contour = crop_resized

            # Convert to base64 (JPEG for smaller file sizes)
            pil_img_final = Image.fromarray(crop_with_contour)
            buffer = BytesIO()
            pil_img_final.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

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


def create_export_index(output_dir, total_mks, total_hspcs, mk_pages, hspc_pages):
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


def generate_export_page_html(samples, cell_type, page_num, total_pages):
    """Generate HTML for a single cell type page."""
    cell_type_display = "Megakaryocytes (MKs)" if cell_type == "mk" else "HSPCs"

    nav_html = '<div class="page-nav">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{cell_type}_page{page_num-1}.html" class="nav-btn">Previous</a>'
    nav_html += f'<span class="page-info">Page {page_num} of {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{cell_type}_page{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += '</div>'

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

    prev_page = page_num - 1
    next_page = page_num + 1

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

        loadAnnotations();
    </script>
</body>
</html>'''
    return html


def generate_export_pages(samples, cell_type, output_dir, samples_per_page):
    """Generate separate pages for a single cell type."""
    if not samples:
        print(f"  No {cell_type.upper()} samples to export")
        return

    pages = [samples[i:i+samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    print(f"  Generating {total_pages} {cell_type.upper()} pages...")

    for page_num in range(1, total_pages + 1):
        page_samples = pages[page_num - 1]
        html = generate_export_page_html(page_samples, cell_type, page_num, total_pages)

        html_path = Path(output_dir) / f"{cell_type}_page{page_num}.html"
        with open(html_path, 'w') as f:
            f.write(html)


def export_html_from_ram(slide_data, output_base, html_output_dir, samples_per_page=300,
                         mk_min_area_um=200, mk_max_area_um=2000):
    """
    Export HTML pages using slide images already in RAM.

    Args:
        slide_data: dict of {slide_name: {'image': np.array, 'czi_path': path, ...}}
        output_base: Path to segmentation output directory
        html_output_dir: Path to write HTML files
        samples_per_page: Number of samples per HTML page
        mk_min_area_um: Min MK area filter
        mk_max_area_um: Max MK area filter
    """
    print(f"\n{'='*70}")
    print("EXPORTING HTML (using images in RAM)")
    print(f"{'='*70}")

    html_output_dir = Path(html_output_dir)
    html_output_dir.mkdir(parents=True, exist_ok=True)

    all_mk_samples = []
    all_hspc_samples = []

    PIXEL_SIZE_UM = 0.1725  # Default pixel size

    for slide_name, data in slide_data.items():
        slide_dir = output_base / slide_name
        if not slide_dir.exists():
            continue

        print(f"  Loading {slide_name}...", flush=True)

        # Get pixel size from summary
        summary_file = slide_dir / "summary.json"
        pixel_size_um = PIXEL_SIZE_UM
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                ps = summary.get('pixel_size_um')
                if ps:
                    pixel_size_um = ps[0] if isinstance(ps, list) else ps

        slide_image = data['image']

        # Load MK samples (uses largest connected component)
        mk_samples = load_samples_from_ram(
            slide_dir / "mk" / "tiles",
            slide_image, pixel_size_um,
            cell_type='mk'
        )

        # Load HSPC samples (sorted by solidity/confidence)
        hspc_samples = load_samples_from_ram(
            slide_dir / "hspc" / "tiles",
            slide_image, pixel_size_um,
            cell_type='hspc'
        )

        # Add slide name to each sample
        for s in mk_samples:
            s['slide'] = slide_name
        for s in hspc_samples:
            s['slide'] = slide_name

        all_mk_samples.extend(mk_samples)
        all_hspc_samples.extend(hspc_samples)

        print(f"    {len(mk_samples)} MKs, {len(hspc_samples)} HSPCs")

    # Filter MK by size
    um_to_px_factor = PIXEL_SIZE_UM ** 2
    mk_min_px = int(mk_min_area_um / um_to_px_factor)
    mk_max_px = int(mk_max_area_um / um_to_px_factor)

    mk_before = len(all_mk_samples)
    all_mk_samples = [s for s in all_mk_samples if mk_min_px <= s.get('area_px', 0) <= mk_max_px]
    print(f"  MK size filter: {mk_before} -> {len(all_mk_samples)}")

    # Sort by area
    all_mk_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)
    all_hspc_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)

    # Generate pages
    generate_export_pages(all_mk_samples, "mk", html_output_dir, samples_per_page)
    generate_export_pages(all_hspc_samples, "hspc", html_output_dir, samples_per_page)

    # Create index
    mk_pages = (len(all_mk_samples) + samples_per_page - 1) // samples_per_page if all_mk_samples else 0
    hspc_pages = (len(all_hspc_samples) + samples_per_page - 1) // samples_per_page if all_hspc_samples else 0
    create_export_index(html_output_dir, len(all_mk_samples), len(all_hspc_samples), mk_pages, hspc_pages)

    print(f"\n  HTML export complete: {html_output_dir}")
    print(f"  Total: {len(all_mk_samples)} MKs, {len(all_hspc_samples)} HSPCs")


# Global variable for worker process
segmenter = None
shared_image = None

def init_worker(mk_classifier_path, hspc_classifier_path, gpu_queue, mm_path, mm_shape, mm_dtype):
    """Initialize the segmenter and attach to shared memory map once per worker process."""
    global segmenter, shared_image
    
    # Assign GPU
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # Get a GPU ID from the queue
            gpu_id = gpu_queue.get(timeout=5)
            device = f"cuda:{gpu_id}"
        except:
            # Fallback: simple modulo or default
            n_gpus = torch.cuda.device_count()
            if n_gpus > 0:
                gpu_id = mp.current_process().pid % n_gpus
                device = f"cuda:{gpu_id}"
            else:
                device = "cuda"
    
    print(f"Worker {mp.current_process().pid} initialized on {device}", flush=True)
    
    # Initialize Segmenter
    segmenter = UnifiedSegmenter(
        mk_classifier_path=mk_classifier_path,
        hspc_classifier_path=hspc_classifier_path,
        device=device
    )

    # Attach to Memory Map (Read-Only)
    try:
        shared_image = np.memmap(mm_path, dtype=mm_dtype, mode='r', shape=mm_shape)
        print(f"Worker {mp.current_process().pid} attached to memmap: {mm_path}", flush=True)
    except Exception as e:
        print(f"Worker {mp.current_process().pid} FAILED to attach to memmap: {e}", flush=True)
        shared_image = None

def process_tile_worker(args):
    """
    Worker function for processing a single tile in a separate process.
    """
    # Unpack all arguments
    tile, _, _, _, output_dir, \
    mk_min_area, mk_max_area, variance_threshold, \
    calibration_block_size = args

    # Use global variables
    global segmenter, shared_image
    
    tid = tile['id']
    
    # Extract tile from Shared Memory
    if shared_image is None:
        return {'tid': tid, 'status': 'error', 'error': "Shared memory not available in worker"}

    try:
        # Slice directly from shared memory (no disk I/O)
        # Coordinate system of shared_image matches the tiling grid (0,0 is start of image ROI)
        img = shared_image[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]
        
        # Check if valid
        if img.size == 0:
             return {'tid': tid, 'status': 'error', 'error': f"Empty crop extracted for tile {tid}"}
             
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Memory slice error: {e}"}

    # Convert to RGB
    if img.ndim == 2:
        img_rgb = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 4:
        img_rgb = img[:, :, :3]
    else:
        img_rgb = img

    if img_rgb.max() == 0:
        return {'tid': tid, 'status': 'empty'}

    img_rgb = percentile_normalize(img_rgb, p_low=5, p_high=95)
    has_tissue_content, _ = has_tissue(img_rgb, variance_threshold, block_size=calibration_block_size)
    if not has_tissue_content:
        return {'tid': tid, 'status': 'no_tissue'}

    try:
        mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
            img_rgb, mk_min_area, mk_max_area
        )
        return {
            'tid': tid, 'status': 'success',
            'mk_masks': mk_masks, 'hspc_masks': hspc_masks,
            'mk_feats': mk_feats, 'hspc_feats': hspc_feats,
            'tile': tile
        }
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Processing error: {e}"}


_ROCM_PATCH_APPLIED = False

def _apply_rocm_patch_if_needed():
    """Apply ROCm INT_MAX fix lazily - call this before using SAM2."""
    global _ROCM_PATCH_APPLIED
    if _ROCM_PATCH_APPLIED:
        return
    _ROCM_PATCH_APPLIED = True

    try:
        import torch
        from typing import List, Dict, Any
        import sam2.utils.amg as amg

        def mask_to_rle_pytorch_rocm_safe(tensor: torch.Tensor) -> List[Dict[str, Any]]:
            """
            Encodes masks to an uncompressed RLE, with ROCm INT_MAX workaround.
            Moves tensor to CPU before nonzero() to avoid INT_MAX issues.
            """
            # Put in fortran order and flatten h,w
            b, h, w = tensor.shape
            tensor = tensor.permute(0, 2, 1).flatten(1)

            # Compute change indices
            diff = tensor[:, 1:] ^ tensor[:, :-1]

            # ROCm FIX: Move to CPU before nonzero() to avoid INT_MAX error
            diff_cpu = diff.cpu()
            change_indices = diff_cpu.nonzero()

            # Encode run length
            out = []
            for i in range(b):
                cur_idxs = change_indices[change_indices[:, 0] == i, 1]
                cur_idxs = torch.cat(
                    [
                        torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                        cur_idxs + 1,
                        torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
                    ]
                )
                btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
                counts = [] if tensor[i, 0] == 0 else [0]
                counts.extend(btw_idxs.detach().cpu().tolist())
                out.append({"size": [h, w], "counts": counts})
            return out

        # Apply the patch
        amg.mask_to_rle_pytorch = mask_to_rle_pytorch_rocm_safe
        print("[ROCm FIX] Patched sam2.utils.amg.mask_to_rle_pytorch for INT_MAX workaround")
    except ImportError as e:
        print(f"[ROCm FIX] Could not apply patch: {e}")

def get_pixel_size_from_czi(czi_path):
    """Extract pixel size in microns from CZI metadata."""
    from pylibCZIrw import czi as pyczi

    reader = pyczi.CziReader(str(czi_path))
    meta = reader.metadata
    reader.close()

    try:
        scaling = meta['ImageDocument']['Metadata']['Scaling']['Items']['Distance']
        px_x = px_y = None
        for item in scaling:
            if item.get('@Id') == 'X':
                px_x = float(item['Value']) * 1e6
            elif item.get('@Id') == 'Y':
                px_y = float(item['Value']) * 1e6
        if px_x and px_y:
            return px_x, px_y
    except (KeyError, TypeError):
        pass
    raise ValueError(f"Could not extract pixel size from {czi_path}")


def percentile_normalize(image, p_low=5, p_high=95):
    """
    Normalize image intensity using percentile scaling.
    Maps p_low to p_high percentile range to 0-255.
    Reduces slide-to-slide variation from staining differences.
    """
    if image.ndim == 2:
        low_val = np.percentile(image, p_low)
        high_val = np.percentile(image, p_high)
        if high_val > low_val:
            normalized = (image.astype(np.float32) - low_val) / (high_val - low_val) * 255
            return np.clip(normalized, 0, 255).astype(np.uint8)
        return image.astype(np.uint8)
    elif image.ndim == 3:
        # Normalize each channel independently
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            result[:, :, c] = percentile_normalize(image[:, :, c], p_low, p_high)
        return result
    return image


def calculate_block_variances(gray_image, block_size=512):
    """Calculate variance for each block in the image."""
    variances = []
    height, width = gray_image.shape

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = gray_image[y:y+block_size, x:x+block_size]
            if block.size < (block_size * block_size) / 4:
                continue
            variances.append(np.var(block))

    return variances


def calibrate_tissue_threshold(tiles, reader=None, x_start=0, y_start=0, calibration_samples=50, block_size=512, image_array=None):
    """
    Auto-detect variance threshold using K-means clustering.
    Supports reading from CZI reader OR memory-mapped/shared array.
    """
    import cv2
    from sklearn.cluster import KMeans

    print(f"Calibrating tissue threshold (K-means 3-cluster on {calibration_samples} random tiles)...")

    # Sample 50 tiles
    n_calibration = calibration_samples
    np.random.seed(42)
    calibration_tiles = list(np.random.choice(tiles, min(n_calibration, len(tiles)), replace=False))

    all_variances = []
    empty_count = 0

    for tile in calibration_tiles:
        # Extract tile from Array (Preferred) or CZI
        if image_array is not None:
            # image_array is (H, W, C) or (H, W) matching the tiles grid
            img = image_array[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]
        elif reader is not None:
            roi = (x_start + tile['x'], y_start + tile['y'], tile['w'], tile['h'])
            try:
                img = reader.read(roi=roi, plane={'C': 0})
            except:
                continue
        else:
            continue

        # Empty tiles contribute low variance (important for calibration)
        if img.max() == 0:
            n_blocks = (tile['w'] // block_size) * (tile['h'] // block_size)
            all_variances.extend([0.0] * max(1, n_blocks))
            empty_count += 1
            continue

        # Percentile normalize (5-95%) to standardize variance across slides
        if img.ndim == 3:
            img_norm = percentile_normalize(img, p_low=5, p_high=95)
            gray = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
        else:
            gray = percentile_normalize(img, p_low=5, p_high=95)

        block_vars = calculate_block_variances(gray, block_size)
        if block_vars:
            all_variances.extend(block_vars)

    if len(all_variances) < 20:
        print("  WARNING: Not enough samples, using default threshold 15.0")
        return 15.0

    variances = np.array(all_variances)

    # K-means with 3 clusters: background (low var), tissue (medium var), artifacts (high var)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(variances.reshape(-1, 1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.flatten()

    # Find the bottom cluster (lowest center = background)
    sorted_indices = np.argsort(centers)
    bottom_cluster_idx = sorted_indices[0]

    # Threshold = max variance in the bottom cluster (to exclude all background blocks)
    bottom_cluster_variances = variances[labels == bottom_cluster_idx]
    threshold = float(np.max(bottom_cluster_variances)) if len(bottom_cluster_variances) > 0 else 15.0

    print(f"  K-means centers: {sorted(centers)[0]:.1f} (bg), {sorted(centers)[1]:.1f} (tissue), {sorted(centers)[2]:.1f} (outliers)")
    print(f"  Threshold (bg cluster max): {threshold:.1f}")
    print(f"  Sampled {len(calibration_tiles)} tiles ({empty_count} empty), {len(variances)} blocks")

    return threshold


def has_tissue(tile_image, variance_threshold, min_tissue_fraction=0.15, block_size=512):
    """
    Check if a tile contains tissue using block-based variance.
    """
    import cv2

    # Handle all-black tiles (empty CZI regions)
    if tile_image.max() == 0:
        return False, 0.0

    # Convert to grayscale (already normalized)
    if tile_image.ndim == 3:
        gray = cv2.cvtColor(tile_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = tile_image.astype(np.uint8)

    variances = calculate_block_variances(gray, block_size)

    if len(variances) == 0:
        return False, 0.0

    tissue_blocks = sum(1 for v in variances if v >= variance_threshold)
    tissue_fraction = tissue_blocks / len(variances)

    return tissue_fraction >= min_tissue_fraction, tissue_fraction


def extract_morphological_features(mask, image):
    """Extract 22 morphological/intensity features from a mask."""
    from skimage import measure

    if not mask.any():
        return {}

    area = int(mask.sum())
    props = measure.regionprops(mask.astype(int))[0]

    perimeter = props.perimeter if hasattr(props, 'perimeter') else 0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    solidity = props.solidity if hasattr(props, 'solidity') else 0
    aspect_ratio = props.major_axis_length / props.minor_axis_length if props.minor_axis_length > 0 else 1
    extent = props.extent if hasattr(props, 'extent') else 0
    equiv_diameter = props.equivalent_diameter if hasattr(props, 'equivalent_diameter') else 0

    # Intensity features
    if image.ndim == 3:
        masked_pixels = image[mask]
        red_mean, red_std = float(np.mean(masked_pixels[:, 0])), float(np.std(masked_pixels[:, 0]))
        green_mean, green_std = float(np.mean(masked_pixels[:, 1])), float(np.std(masked_pixels[:, 1]))
        blue_mean, blue_std = float(np.mean(masked_pixels[:, 2])), float(np.std(masked_pixels[:, 2]))
        gray = np.mean(masked_pixels, axis=1)
    else:
        gray = image[mask]
        red_mean = green_mean = blue_mean = float(np.mean(gray))
        red_std = green_std = blue_std = float(np.std(gray))

    gray_mean, gray_std = float(np.mean(gray)), float(np.std(gray))

    # HSV features
    if image.ndim == 3:
        import colorsys
        hsv = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in masked_pixels])
        hue_mean = float(np.mean(hsv[:, 0]) * 180)
        sat_mean = float(np.mean(hsv[:, 1]) * 255)
        val_mean = float(np.mean(hsv[:, 2]) * 255)
    else:
        hue_mean = sat_mean = 0.0
        val_mean = gray_mean

    # Texture features
    relative_brightness = gray_mean - np.mean(image) if image.size > 0 else 0
    intensity_variance = float(np.var(gray))
    dark_fraction = float(np.mean(gray < 100))
    nuclear_complexity = gray_std

    return {
        'area': area,
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'solidity': float(solidity),
        'aspect_ratio': float(aspect_ratio),
        'extent': float(extent),
        'equiv_diameter': float(equiv_diameter),
        'red_mean': red_mean, 'red_std': red_std,
        'green_mean': green_mean, 'green_std': green_std,
        'blue_mean': blue_mean, 'blue_std': blue_std,
        'gray_mean': gray_mean, 'gray_std': gray_std,
        'hue_mean': hue_mean, 'saturation_mean': sat_mean, 'value_mean': val_mean,
        'relative_brightness': float(relative_brightness),
        'intensity_variance': intensity_variance,
        'dark_region_fraction': dark_fraction,
        'nuclear_complexity': nuclear_complexity,
    }


class UnifiedSegmenter:
    """Unified segmenter for MKs and HSPCs."""

    def __init__(
        self,
        sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        mk_classifier_path=None,
        hspc_classifier_path=None,
        device=None
    ):
        _apply_rocm_patch_if_needed()
        from cellpose.models import CellposeModel
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        # Convert device to torch.device if it's a string
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Find SAM2 checkpoint (auto-detect local or cluster)
        script_dir = Path(__file__).parent.resolve()
        sam2_candidates = [
            script_dir / sam2_checkpoint,  # Local: same dir as script
            script_dir / "checkpoints" / Path(sam2_checkpoint).name,  # Local checkpoints subdir
            Path("/ptmp/edrod/MKsegmentation") / sam2_checkpoint,  # Cluster path
        ]
        checkpoint_path = None
        for cp in sam2_candidates:
            if cp.exists():
                checkpoint_path = cp
                break
        if checkpoint_path is None:
            # Default to local path (will fail with helpful error if missing)
            checkpoint_path = script_dir / "checkpoints" / Path(sam2_checkpoint).name

        print(f"Loading SAM2 from {checkpoint_path}...")
        sam2_model = build_sam2(sam2_config, str(checkpoint_path), device=self.device)

        # SAM2 for auto mask generation (MKs)
        self.sam2_auto = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=24, # Changed from 32 to 24
            pred_iou_thresh=0.5,  # Stricter filtering for speed
            stability_score_thresh=0.4,
            min_mask_region_area=500,
            crop_n_layers=1 # Added crop_n_layers
        )

        # SAM2 predictor for point prompts (HSPCs)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Cellpose-SAM for HSPC detection (v4+ with SAM backbone)
        print(f"Loading Cellpose-SAM model on {self.device}...")
        self.cellpose = CellposeModel(pretrained_model='cpsam', gpu=True, device=self.device)

        # ResNet for deep features
        print("Loading ResNet-50...")
        resnet = tv_models.resnet50(weights='DEFAULT')
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.resnet.eval().to(self.device)
        self.resnet_transform = tv_transforms.Compose([
            tv_transforms.Resize(224),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load classifiers if provided
        self.mk_classifier = None
        self.mk_feature_names = None
        self.hspc_classifier = None
        self.hspc_feature_names = None

        if mk_classifier_path:
            print(f"Loading MK classifier: {mk_classifier_path}")
            import joblib
            clf_data = joblib.load(mk_classifier_path)
            self.mk_classifier = clf_data['classifier']
            self.mk_feature_names = clf_data['feature_names']
            print(f"  Features: {len(self.mk_feature_names)}, Trained on {clf_data.get('n_samples', '?')} samples")

        if hspc_classifier_path:
            print(f"Loading HSPC classifier: {hspc_classifier_path}")
            import joblib
            clf_data = joblib.load(hspc_classifier_path)
            self.hspc_classifier = clf_data['classifier']
            self.hspc_feature_names = clf_data['feature_names']
            print(f"  Features: {len(self.hspc_feature_names)}, Trained on {clf_data.get('n_samples', '?')} samples")

        print("Models loaded.")

    def apply_classifier(self, features_dict, cell_type):
        """Apply classifier to features and return (is_positive, confidence).

        Args:
            features_dict: Dict of feature_name -> value
            cell_type: 'mk' or 'hspc'

        Returns:
            (bool, float): (is_positive, confidence_score)
        """
        if cell_type == 'mk':
            clf = self.mk_classifier
            feature_names = self.mk_feature_names
        else:
            clf = self.hspc_classifier
            feature_names = self.hspc_feature_names

        if clf is None:
            return True, 1.0  # No classifier = accept all

        # Build feature vector in correct order
        X = np.array([[features_dict.get(name, 0.0) for name in feature_names]])

        # Predict
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        confidence = proba[1] if pred == 1 else proba[0]

        return bool(pred == 1), float(confidence)

    def extract_resnet_features(self, crop):
        """Extract 2048D ResNet features from crop."""
        if crop.ndim == 2:
            crop = np.stack([crop]*3, axis=-1)
        pil_img = Image.fromarray(crop.astype(np.uint8))
        tensor = self.resnet_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.resnet(tensor).cpu().numpy().flatten()
        return features

    def extract_sam2_embedding(self, cy, cx):
        """Extract 256D SAM2 embedding at location."""
        try:
            emb_y = int(cy / 16)
            emb_x = int(cx / 16)
            shape = self.sam2_predictor._features.shape
            emb_y = min(max(0, emb_y), shape[2] - 1)
            emb_x = min(max(0, emb_x), shape[3] - 1)
            return self.sam2_predictor._features[0, :, emb_y, emb_x].cpu().numpy()
        except:
            return np.zeros(256)

    def process_tile(
        self,
        image_rgb,
        mk_min_area=1000,
        mk_max_area=100000
    ):
        """Process a single tile for both MKs and HSPCs.

        Note: mk_min_area/mk_max_area only filter MKs, not HSPCs.
        Cellpose-SAM handles HSPC detection without size parameters.

        Returns:
            mk_masks: Label array for MKs
            hspc_masks: Label array for HSPCs
            mk_features: List of feature dicts
            hspc_features: List of feature dicts
        """
        from scipy import ndimage

        # ============================================
        # MK Detection: SAM2 automatic mask generation
        # ============================================
        # Process MK detection FIRST (auto generator manages its own predictor)
        # This avoids holding embeddings in TWO predictors simultaneously
        sam2_results = self.sam2_auto.generate(image_rgb)

        # Filter by size and sort by area (largest first)
        valid_results = []
        for result in sam2_results:
            area = result['segmentation'].sum()
            if mk_min_area <= area <= mk_max_area:
                result['area'] = area
                valid_results.append(result)
        valid_results.sort(key=lambda x: x['area'], reverse=True)

        # Delete original sam2_results to free memory (can be 3GB+ of masks)
        del sam2_results
        gc.collect()

        mk_masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        mk_features = []
        mk_id = 1

        for result in valid_results:
            mask = result['segmentation']
            # Ensure boolean type for indexing (critical for NVIDIA CUDA compatibility)
            if mask.dtype != bool:
                mask = (mask > 0.5).astype(bool)

            # Check overlap with existing masks - skip if >50% overlaps (larger already added)
            if mk_masks.max() > 0:
                overlap = ((mask > 0) & (mk_masks > 0)).sum()
                if overlap > 0.5 * mask.sum():
                    continue

            # Add to label array
            mk_masks[mask] = mk_id

            # Get centroid
            cy, cx = ndimage.center_of_mass(mask)

            # Extract all features
            morph = extract_morphological_features(mask, image_rgb)

            # SAM2 embeddings
            sam2_emb = self.extract_sam2_embedding(cy, cx)
            for i, v in enumerate(sam2_emb):
                morph[f'sam2_emb_{i}'] = float(v)

            # ResNet features from masked crop
            ys, xs = np.where(mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop = image_rgb[y1:y2+1, x1:x2+1].copy()
            crop_mask = mask[y1:y2+1, x1:x2+1]
            crop[~crop_mask] = 0

            resnet_feats = self.extract_resnet_features(crop)
            for i, v in enumerate(resnet_feats):
                morph[f'resnet_{i}'] = float(v)

            mk_features.append({
                'id': f'det_{mk_id}',
                'center': [float(cx), float(cy)],
                'sam2_iou': float(result.get('predicted_iou', 0)),
                'sam2_stability': float(result.get('stability_score', 0)),
                'features': morph
            })

            mk_id += 1

        # Delete valid_results to free memory (large mask arrays)
        del valid_results
        gc.collect()
        torch.cuda.empty_cache()  # Clear GPU cache after MK processing

        # ============================================
        # HSPC Detection: Cellpose-SAM + SAM2 refinement
        # ============================================
        # Now set image for SAM2 predictor (after auto generator is done)
        # This ensures we only hold ONE predictor's embeddings at a time
        self.sam2_predictor.set_image(image_rgb)

        # Cellpose with grayscale mode, let cpsam auto-detect size
        cellpose_masks, _, _ = self.cellpose.eval(image_rgb, channels=[0,0])

        # Get Cellpose centroids and limit to top 500 by mask area
        cellpose_ids = np.unique(cellpose_masks)
        cellpose_ids = cellpose_ids[cellpose_ids > 0]

        # Limit Cellpose candidates to avoid processing thousands per tile
        MAX_CELLPOSE_CANDIDATES = 500
        if len(cellpose_ids) > MAX_CELLPOSE_CANDIDATES:
            # Sort by mask area (larger first) and take top N
            areas = [(cp_id, (cellpose_masks == cp_id).sum()) for cp_id in cellpose_ids]
            areas.sort(key=lambda x: x[1], reverse=True)
            cellpose_ids = np.array([a[0] for a in areas[:MAX_CELLPOSE_CANDIDATES]])

        # Collect all HSPC candidates with SAM2 refinement
        hspc_candidates = []
        for cp_id in cellpose_ids:
            cp_mask = cellpose_masks == cp_id
            cy, cx = ndimage.center_of_mass(cp_mask)

            # Use centroid as SAM2 prompt
            point_coords = np.array([[cx, cy]])
            point_labels = np.array([1])

            masks_pred, scores, _ = self.sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )

            # Take best mask
            best_idx = np.argmax(scores)
            sam2_mask = masks_pred[best_idx]
            # Ensure boolean type for indexing (critical for NVIDIA CUDA compatibility)
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)
            sam2_score = float(scores[best_idx])

            if sam2_mask.sum() < 10:
                continue

            # Keep all candidates - overlap filtering happens after sorting by confidence
            hspc_candidates.append({
                'mask': sam2_mask,
                'score': sam2_score,
                'center': (cx, cy),
                'cp_id': cp_id
            })

        # Sort by SAM2 score (most confident first) and handle overlaps
        hspc_candidates.sort(key=lambda x: x['score'], reverse=True)

        hspc_masks = np.zeros(image_rgb.shape[:2], dtype=np.uint32)
        hspc_features = []
        hspc_id = 1

        for cand in hspc_candidates:
            sam2_mask = cand['mask']
            # Ensure boolean type for indexing (critical for NVIDIA CUDA compatibility)
            if sam2_mask.dtype != bool:
                sam2_mask = (sam2_mask > 0.5).astype(bool)
            sam2_score = cand['score']
            cx, cy = cand['center']

            # Check overlap with existing HSPC masks - skip if >50% overlaps
            if hspc_masks.max() > 0:
                overlap = ((sam2_mask > 0) & (hspc_masks > 0)).sum()
                if overlap > 0.5 * sam2_mask.sum():
                    continue

            # Add to label array
            hspc_masks[sam2_mask] = hspc_id

            # Extract features
            morph = extract_morphological_features(sam2_mask, image_rgb)

            # SAM2 embeddings
            sam2_emb = self.extract_sam2_embedding(cy, cx)
            for i, v in enumerate(sam2_emb):
                morph[f'sam2_emb_{i}'] = float(v)

            # ResNet features
            ys, xs = np.where(sam2_mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            crop = image_rgb[y1:y2+1, x1:x2+1].copy()
            crop_mask = sam2_mask[y1:y2+1, x1:x2+1]
            crop[~crop_mask] = 0

            resnet_feats = self.extract_resnet_features(crop)
            for i, v in enumerate(resnet_feats):
                morph[f'resnet_{i}'] = float(v)

            hspc_features.append({
                'id': f'det_{hspc_id}',
                'center': [float(cx), float(cy)],
                'cellpose_id': int(cand['cp_id']),
                'sam2_score': sam2_score,
                'features': morph
            })

            hspc_id += 1

        # Delete large temporary arrays to free memory
        del hspc_candidates
        del cellpose_masks
        gc.collect()

        # Clear SAM2 cached features after processing this tile
        self.sam2_predictor.reset_predictor()
        torch.cuda.empty_cache()

        return mk_masks, hspc_masks, mk_features, hspc_features


def run_unified_segmentation(
    czi_path,
    output_dir,
    mk_min_area=1000,
    mk_max_area=100000,
    tile_size=4096,
    overlap=512,
    sample_fraction=1.0,
    calibration_block_size=512,
    calibration_samples=50,
    num_workers=4,
    mk_classifier_path=None,
    hspc_classifier_path=None
):
    """Run unified MK + HSPC segmentation with multiprocessing."""
    from pylibCZIrw import czi as pyczi

    # Set start method to 'spawn' for GPU safety
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)

    czi_path = Path(czi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("UNIFIED SEGMENTATION: MK + HSPC (MULTIPROCESSING)")
    print(f"{'='*70}")
    print(f"CZI: {czi_path}")

    # Open CZI
    reader = pyczi.CziReader(str(czi_path))
    scenes = reader.scenes_bounding_rectangle
    if scenes:
        rect = scenes[0]
        x_start, y_start = rect.x, rect.y
        full_width, full_height = rect.w, rect.h
    else:
        bbox = reader.total_bounding_box
        x_start, y_start = bbox['X'][0], bbox['Y'][0]
        full_width = bbox['X'][1] - bbox['X'][0]
        full_height = bbox['Y'][1] - bbox['Y'][0]

    print(f"Image: {full_width} x {full_height}")

    # Load full image into Memory Map
    print("Loading image into Memory Map...", flush=True)
    # Note: read() returns (H, W, C) or (H, W)
    full_img = reader.read(plane={"C": 0, "T": 0, "Z": 0}, roi=(x_start, y_start, full_width, full_height))

    # Create temporary directory for memmap
    # OPTIMIZATION: Use /dev/shm (RAM-backed tmpfs) if available and has space
    # This eliminates disk I/O entirely - tiles are read directly from RAM
    import shutil
    import os

    img_size_gb = full_img.nbytes / (1024**3)
    use_ramdisk = False

    # Check if /dev/shm exists and has enough space (need 2x for safety margin)
    shm_path = Path("/dev/shm")
    if shm_path.exists():
        try:
            shm_stat = os.statvfs("/dev/shm")
            shm_free_gb = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024**3)
            if shm_free_gb > img_size_gb * 1.5:  # 50% safety margin
                use_ramdisk = True
                print(f"  Using RAM-backed storage (/dev/shm): {shm_free_gb:.1f} GB free")
        except:
            pass

    if use_ramdisk:
        temp_mm_dir = shm_path / f"mkseg_{os.getpid()}"
    else:
        temp_mm_dir = output_dir / "temp_mm"
        print(f"  Using disk-backed storage (fallback)")

    temp_mm_dir.mkdir(parents=True, exist_ok=True)
    mm_path = temp_mm_dir / "image.dat"

    # Create writable memmap
    shm_arr = np.memmap(mm_path, dtype=full_img.dtype, mode='w+', shape=full_img.shape)

    # Copy image to memmap
    np.copyto(shm_arr, full_img)
    shm_arr.flush()

    storage_type = "RAM" if use_ramdisk else "disk"
    print(f"Image loaded to {storage_type}: {mm_path} ({img_size_gb:.2f} GB)")
    
    # Capture shape/dtype for workers
    mm_shape = full_img.shape
    mm_dtype = full_img.dtype
    
    # Free local memory
    del full_img
    del shm_arr # Close handle in main process
    
    # Create tiles
    n_tx = int(np.ceil(full_width / (tile_size - overlap)))
    n_ty = int(np.ceil(full_height / (tile_size - overlap)))
    tiles = []
    for ty in range(n_ty):
        for tx in range(n_tx):
            tiles.append({
                'id': len(tiles),
                'x': tx * (tile_size - overlap),
                'y': ty * (tile_size - overlap),
                'w': min(tile_size, full_width - tx * (tile_size - overlap)),
                'h': min(tile_size, full_height - ty * (tile_size - overlap))
            })

    print(f"Total tiles: {len(tiles)}")

    # Calibrate tissue threshold using MEMMAP array
    # Re-open memmap in read mode for calibration
    calib_arr = np.memmap(mm_path, dtype=mm_dtype, mode='r', shape=mm_shape)
    
    variance_threshold = calibrate_tissue_threshold(
        tiles, reader=None, x_start=x_start, y_start=y_start, 
        calibration_samples=calibration_samples, 
        block_size=calibration_block_size,
        image_array=calib_arr
    )
    
    del calib_arr # Close calibration handle

    reader.close() # Close reader in main process

    if sample_fraction < 1.0:
        n = max(1, int(len(tiles) * sample_fraction))
        np.random.seed(42)
        tiles = list(np.random.choice(tiles, n, replace=False))
        print(f"Sampling {len(tiles)} tiles for processing")

    # Prepare arguments for worker processes (NO reader, NO large objects)
    worker_args = []
    for tile in tiles:
        worker_args.append((
            tile, czi_path, x_start, y_start, output_dir,
            mk_min_area, mk_max_area, variance_threshold,
            calibration_block_size
        ))
    
    # Process tiles in parallel
    mk_count = 0  # Just count, don't accumulate features (already saved to disk)
    hspc_count = 0
    mk_gid = 1
    hspc_gid = 1

    # Setup GPU distribution
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            # Fill queue with GPU IDs for workers to consume
            for i in range(num_workers):
                gpu_queue.put(i % n_gpus)
    
    # Pass Memmap path to initializer
    # Note: passing path as string is safe for pickling
    init_args = (mk_classifier_path, hspc_classifier_path, gpu_queue, str(mm_path), mm_shape, mm_dtype)
    
    try:
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            with tqdm(total=len(tiles), desc="Processing tiles") as pbar:
                for result in pool.imap_unordered(process_tile_worker, worker_args):
                    pbar.update(1)
                    if result['status'] == 'success':
                        tile = result['tile']
                        tid = result['tid']
                        
                        # Process and save MK results
                        if result['mk_feats']:
                            mk_dir = output_dir / "mk" / "tiles"
                            mk_tile_dir = mk_dir / str(tid)
                            mk_tile_dir.mkdir(parents=True, exist_ok=True)
                            
                            with open(mk_tile_dir / "window.csv", 'w') as f:
                                f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")
                            
                            new_mk = np.zeros_like(result['mk_masks'])
                            mk_tile_cells = []
                            for feat in result['mk_feats']:
                                old_id = int(feat['id'].split('_')[1])
                                new_mk[result['mk_masks'] == old_id] = mk_gid
                                feat['id'] = f'det_{mk_gid - 1}'
                                feat['global_id'] = mk_gid
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                mk_tile_cells.append(mk_gid)
                                mk_count += 1  # Just count, features already saved to disk
                                mk_gid += 1
                            
                            with open(mk_tile_dir / "classes.csv", 'w') as f:
                                for c in mk_tile_cells: f.write(f"{c}\n")
                            with h5py.File(mk_tile_dir / "segmentation.h5", 'w') as f:
                                create_hdf5_dataset(f, 'labels', new_mk[np.newaxis])
                            with open(mk_tile_dir / "features.json", 'w') as f:
                                json.dump([{'id': m['id'], 'features': m['features']} for m in result['mk_feats']], f)

                            # Explicit cleanup to prevent memory accumulation
                            del new_mk, mk_tile_cells

                        # Process and save HSPC results
                        if result['hspc_feats']:
                            hspc_dir = output_dir / "hspc" / "tiles"
                            hspc_tile_dir = hspc_dir / str(tid)
                            hspc_tile_dir.mkdir(parents=True, exist_ok=True)
                            
                            with open(hspc_tile_dir / "window.csv", 'w') as f:
                                f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")
                            
                            new_hspc = np.zeros_like(result['hspc_masks'])
                            hspc_tile_cells = []
                            for feat in result['hspc_feats']:
                                old_id = int(feat['id'].split('_')[1])
                                new_hspc[result['hspc_masks'] == old_id] = hspc_gid
                                feat['id'] = f'det_{hspc_gid - 1}'
                                feat['global_id'] = hspc_gid
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                hspc_tile_cells.append(hspc_gid)
                                hspc_count += 1  # Just count, features already saved to disk
                                hspc_gid += 1
                                
                            with open(hspc_tile_dir / "classes.csv", 'w') as f:
                                for c in hspc_tile_cells: f.write(f"{c}\n")
                            with h5py.File(hspc_tile_dir / "segmentation.h5", 'w') as f:
                                create_hdf5_dataset(f, 'labels', new_hspc[np.newaxis])
                            with open(hspc_tile_dir / "features.json", 'w') as f:
                                json.dump([{'id': h['id'], 'features': h['features']} for h in result['hspc_feats']], f)

                            # Explicit cleanup to prevent memory accumulation
                            del new_hspc, hspc_tile_cells

                    elif result['status'] == 'error':
                        print(f"  Tile {result['tid']} error: {result['error']}")

                    # Explicit memory cleanup after processing each result
                    # Delete large mask arrays from result before del result
                    if 'mk_masks' in result:
                        del result['mk_masks']
                    if 'hspc_masks' in result:
                        del result['hspc_masks']
                    if 'mk_feats' in result:
                        del result['mk_feats']
                    if 'hspc_feats' in result:
                        del result['hspc_feats']
                    del result

                    # Run GC every tile to prevent accumulation
                    gc.collect()
    finally:
        # Always clean up temp memmap directory
        if 'temp_mm_dir' in locals() and temp_mm_dir.exists():
            try:
                shutil.rmtree(temp_mm_dir)
                print(f"Cleaned up temp directory: {temp_mm_dir}")
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_mm_dir}: {e}")

    # Save summaries
    # Get pixel size from reader if possible (re-open temporarily)
    try:
        pixel_size_um = get_pixel_size_from_czi(czi_path)
    except:
        pixel_size_um = None

    summary = {
        'czi_path': str(czi_path),
        'pixel_size_um': pixel_size_um,
        'mk_count': mk_count,
        'hspc_count': hspc_count,
        'feature_count': '22 morphological + 256 SAM2 + 2048 ResNet = 2326'
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"MKs detected: {mk_count}")
    print(f"HSPCs detected: {hspc_count}")
    print(f"Features per cell: 2326 (22 + 256 + 2048)")
    print(f"Output: {output_dir}")
    print(f"  MK tiles: {output_dir}/mk/tiles/")
    print(f"  HSPC tiles: {output_dir}/hspc/tiles/")


def run_multi_slide_segmentation(
    czi_paths,
    output_base,
    mk_min_area=1000,
    mk_max_area=100000,
    tile_size=4096,
    overlap=512,
    sample_fraction=1.0,
    calibration_block_size=512,
    calibration_samples=50,
    num_workers=4,
    mk_classifier_path=None,
    hspc_classifier_path=None,
    html_output_dir=None,
    samples_per_page=300,
    mk_min_area_um=200,
    mk_max_area_um=2000
):
    """
    Process multiple slides with UNIFIED SAMPLING.

    - Loads ALL slides into RAM
    - Identifies tissue-containing tiles across ALL slides
    - Samples from the combined pool (truly representative)
    - Processes sampled tiles with models loaded ONCE
    """
    from pylibCZIrw import czi as pyczi
    import shutil
    import cv2

    # Set start method to 'spawn' for GPU safety
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"UNIFIED SAMPLING SEGMENTATION: {len(czi_paths)} slides")
    print(f"{'='*70}")
    print(f"Step 1: Load ALL slides into RAM")
    print(f"Step 2: Identify tissue tiles across all slides")
    print(f"Step 3: Sample {sample_fraction*100:.0f}% from combined pool")
    print(f"Step 4: Process with models loaded ONCE")
    print(f"{'='*70}")

    # =========================================================================
    # PHASE 1: Load all slides into RAM
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: LOADING ALL SLIDES INTO RAM")
    print(f"{'='*70}")

    slide_data = {}  # slide_name -> {'image': np.array, 'shape': tuple, 'czi_path': Path}
    total_size_gb = 0

    for slide_idx, czi_path in enumerate(czi_paths):
        czi_path = Path(czi_path)
        slide_name = czi_path.stem

        print(f"\n[{slide_idx+1}/{len(czi_paths)}] Loading {slide_name}...", flush=True)

        try:
            reader = pyczi.CziReader(str(czi_path))
            scenes = reader.scenes_bounding_rectangle
            if scenes:
                rect = scenes[0]
                x_start, y_start = rect.x, rect.y
                full_width, full_height = rect.w, rect.h
            else:
                bbox = reader.total_bounding_box
                x_start, y_start = bbox['X'][0], bbox['Y'][0]
                full_width = bbox['X'][1] - bbox['X'][0]
                full_height = bbox['Y'][1] - bbox['Y'][0]

            print(f"  Reading {full_width} x {full_height}...", flush=True)
            img = reader.read(plane={"C": 0, "T": 0, "Z": 0},
                              roi=(x_start, y_start, full_width, full_height))
            reader.close()
            del reader  # Explicit cleanup

            size_gb = img.nbytes / (1024**3)
            total_size_gb += size_gb

            slide_data[slide_name] = {
                'image': img,
                'shape': (full_width, full_height),
                'czi_path': czi_path
            }

            print(f"  Loaded: {full_width} x {full_height} ({size_gb:.2f} GB)", flush=True)

            # Force cleanup between loads to prevent memory fragmentation
            gc.collect()

        except Exception as e:
            print(f"  ERROR loading {slide_name}: {e}", flush=True)
            continue

    print(f"\nTotal RAM used: {total_size_gb:.2f} GB")

    # Check available RAM
    import psutil
    mem = psutil.virtual_memory()
    print(f"System RAM: {mem.total/(1024**3):.1f} GB total, {mem.available/(1024**3):.1f} GB available")

    # =========================================================================
    # PHASE 2: Create tiles and identify tissue
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: IDENTIFYING TISSUE TILES")
    print(f"{'='*70}")

    all_tiles = []  # List of (slide_name, tile_dict, has_tissue)

    # First pass: create all tiles
    for slide_name, data in slide_data.items():
        img = data['image']
        full_width, full_height = data['shape']

        n_tx = int(np.ceil(full_width / (tile_size - overlap)))
        n_ty = int(np.ceil(full_height / (tile_size - overlap)))

        slide_tiles = []
        for ty in range(n_ty):
            for tx in range(n_tx):
                tile = {
                    'id': len(slide_tiles),
                    'x': tx * (tile_size - overlap),
                    'y': ty * (tile_size - overlap),
                    'w': min(tile_size, full_width - tx * (tile_size - overlap)),
                    'h': min(tile_size, full_height - ty * (tile_size - overlap))
                }
                slide_tiles.append(tile)

        print(f"  {slide_name}: {len(slide_tiles)} tiles")

        for tile in slide_tiles:
            all_tiles.append((slide_name, tile))

    print(f"\nTotal tiles across all slides: {len(all_tiles)}")

    # Calibrate tissue threshold using samples from all slides (PARALLEL)
    n_calib_samples = min(calibration_samples * len(czi_paths), len(all_tiles))
    print(f"\nCalibrating tissue threshold from {n_calib_samples} samples...", flush=True)

    np.random.seed(42)
    calib_indices = np.random.choice(len(all_tiles), n_calib_samples, replace=False)
    calib_tiles = [all_tiles[idx] for idx in calib_indices]

    # Use parallel calibration (same as tissue checking)
    calib_threads = max(1, int(os.cpu_count() * 0.8))

    def calc_tile_variances(args):
        """Calculate block variances for a single tile. Thread-safe."""
        slide_name, tile = args
        img = slide_data[slide_name]['image']
        tile_img = img[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]

        if tile_img.max() == 0:
            return [0.0]

        if tile_img.ndim == 3:
            tile_norm = percentile_normalize(tile_img, p_low=5, p_high=95)
            gray = cv2.cvtColor(tile_norm, cv2.COLOR_RGB2GRAY)
        else:
            gray = percentile_normalize(tile_img, p_low=5, p_high=95)

        block_vars = calculate_block_variances(gray, calibration_block_size)
        return block_vars if block_vars else []

    all_variances = []
    with ThreadPoolExecutor(max_workers=calib_threads) as executor:
        futures = {executor.submit(calc_tile_variances, tile): tile for tile in calib_tiles}
        for future in tqdm(as_completed(futures), total=len(calib_tiles), desc="Calibrating"):
            result = future.result()
            all_variances.extend(result)

    # K-means clustering for threshold
    from sklearn.cluster import KMeans
    variances = np.array(all_variances)
    print(f"  Running K-means on {len(variances)} variance samples...", flush=True)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(variances.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    sorted_indices = np.argsort(centers)
    bottom_cluster_idx = sorted_indices[0]
    bottom_cluster_variances = variances[labels == bottom_cluster_idx]
    variance_threshold = float(np.max(bottom_cluster_variances)) if len(bottom_cluster_variances) > 0 else 15.0

    print(f"  K-means centers: {sorted(centers)[0]:.1f} (bg), {sorted(centers)[1]:.1f} (tissue), {sorted(centers)[2]:.1f} (outliers)")
    print(f"  Threshold: {variance_threshold:.1f}")

    # Filter to tissue-containing tiles (PARALLEL)
    print(f"\nFiltering to tissue-containing tiles...")

    # Use 80% of CPU cores for tissue checking (NumPy/cv2 release GIL)
    tissue_check_threads = max(1, int(os.cpu_count() * 0.8))
    print(f"  Using {tissue_check_threads} threads for parallel tissue checking")

    def check_tile_tissue(args):
        """Check if a single tile contains tissue. Thread-safe."""
        slide_name, tile = args
        img = slide_data[slide_name]['image']
        tile_img = img[tile['y']:tile['y']+tile['h'], tile['x']:tile['x']+tile['w']]

        if tile_img.max() == 0:
            return None

        tile_norm = percentile_normalize(tile_img, p_low=5, p_high=95)
        has_tissue_flag, _ = has_tissue(tile_norm, variance_threshold, block_size=calibration_block_size)

        if has_tissue_flag:
            return (slide_name, tile)
        return None

    tissue_tiles = []
    with ThreadPoolExecutor(max_workers=tissue_check_threads) as executor:
        futures = {executor.submit(check_tile_tissue, tile_args): tile_args for tile_args in all_tiles}
        for future in tqdm(as_completed(futures), total=len(all_tiles), desc="Checking tissue"):
            result = future.result()
            if result is not None:
                tissue_tiles.append(result)

    print(f"\nTissue tiles: {len(tissue_tiles)} / {len(all_tiles)} ({100*len(tissue_tiles)/len(all_tiles):.1f}%)")

    # =========================================================================
    # PHASE 3: Sample from combined pool
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: SAMPLING FROM COMBINED POOL")
    print(f"{'='*70}")

    if sample_fraction < 1.0:
        n_sample = max(1, int(len(tissue_tiles) * sample_fraction))
        np.random.seed(42)
        sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
        sampled_tiles = [tissue_tiles[i] for i in sample_indices]
    else:
        sampled_tiles = tissue_tiles

    # Count per slide
    slide_counts = {}
    for slide_name, tile in sampled_tiles:
        slide_counts[slide_name] = slide_counts.get(slide_name, 0) + 1

    print(f"Sampled {len(sampled_tiles)} tiles ({sample_fraction*100:.0f}% of tissue tiles)")
    print(f"\nPer-slide distribution:")
    for slide_name in sorted(slide_counts.keys()):
        print(f"  {slide_name}: {slide_counts[slide_name]} tiles")

    # =========================================================================
    # PHASE 4: Process sampled tiles
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 4: PROCESSING SAMPLED TILES")
    print(f"{'='*70}")

    # Setup worker pool
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            for i in range(num_workers):
                gpu_queue.put(i % n_gpus)

    # Dummy memmap for init
    temp_init_dir = output_base / "temp_init"
    temp_init_dir.mkdir(parents=True, exist_ok=True)
    dummy_mm_path = temp_init_dir / "dummy.dat"
    dummy_arr = np.memmap(dummy_mm_path, dtype=np.uint8, mode='w+', shape=(100, 100, 3))
    dummy_arr.flush()
    del dummy_arr

    init_args = (mk_classifier_path, hspc_classifier_path, gpu_queue,
                 str(dummy_mm_path), (100, 100, 3), np.uint8)

    # Track results per slide
    slide_results = {name: {'mk_count': 0, 'hspc_count': 0, 'mk_gid': 1, 'hspc_gid': 1}
                     for name in slide_data.keys()}

    try:
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            print(f"Worker pool ready with {num_workers} workers")

            def tile_generator():
                for slide_name, tile in sampled_tiles:
                    img = slide_data[slide_name]['image']
                    tile_img = img[tile['y']:tile['y']+tile['h'],
                                   tile['x']:tile['x']+tile['w']].copy()
                    output_dir = output_base / slide_name
                    yield (tile, tile_img, output_dir, mk_min_area, mk_max_area,
                           variance_threshold, calibration_block_size, slide_name)

            with tqdm(total=len(sampled_tiles), desc="Processing tiles") as pbar:
                for result in pool.imap_unordered(process_tile_worker_with_data_and_slide, tile_generator()):
                    pbar.update(1)
                    if result['status'] == 'success':
                        slide_name = result['slide_name']
                        tile = result['tile']
                        tid = result['tid']
                        output_dir = output_base / slide_name
                        output_dir.mkdir(parents=True, exist_ok=True)

                        sr = slide_results[slide_name]

                        if result['mk_feats']:
                            mk_dir = output_dir / "mk" / "tiles"
                            mk_tile_dir = mk_dir / str(tid)
                            mk_tile_dir.mkdir(parents=True, exist_ok=True)

                            with open(mk_tile_dir / "window.csv", 'w') as f:
                                f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

                            new_mk = np.zeros_like(result['mk_masks'])
                            mk_tile_cells = []
                            for feat in result['mk_feats']:
                                old_id = int(feat['id'].split('_')[1])
                                new_mk[result['mk_masks'] == old_id] = sr['mk_gid']
                                feat['id'] = f'det_{sr["mk_gid"] - 1}'
                                feat['global_id'] = sr['mk_gid']
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                mk_tile_cells.append(sr['mk_gid'])
                                sr['mk_count'] += 1
                                sr['mk_gid'] += 1

                            with open(mk_tile_dir / "classes.csv", 'w') as f:
                                for c in mk_tile_cells: f.write(f"{c}\n")
                            with h5py.File(mk_tile_dir / "segmentation.h5", 'w') as f:
                                create_hdf5_dataset(f, 'labels', new_mk[np.newaxis])
                            with open(mk_tile_dir / "features.json", 'w') as f:
                                json.dump([{'id': m['id'], 'features': m['features']} for m in result['mk_feats']], f)

                            del new_mk, mk_tile_cells

                        if result['hspc_feats']:
                            hspc_dir = output_dir / "hspc" / "tiles"
                            hspc_tile_dir = hspc_dir / str(tid)
                            hspc_tile_dir.mkdir(parents=True, exist_ok=True)

                            with open(hspc_tile_dir / "window.csv", 'w') as f:
                                f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

                            new_hspc = np.zeros_like(result['hspc_masks'])
                            hspc_tile_cells = []
                            for feat in result['hspc_feats']:
                                old_id = int(feat['id'].split('_')[1])
                                new_hspc[result['hspc_masks'] == old_id] = sr['hspc_gid']
                                feat['id'] = f'det_{sr["hspc_gid"] - 1}'
                                feat['global_id'] = sr['hspc_gid']
                                feat['center'][0] += tile['x']
                                feat['center'][1] += tile['y']
                                hspc_tile_cells.append(sr['hspc_gid'])
                                sr['hspc_count'] += 1
                                sr['hspc_gid'] += 1

                            with open(hspc_tile_dir / "classes.csv", 'w') as f:
                                for c in hspc_tile_cells: f.write(f"{c}\n")
                            with h5py.File(hspc_tile_dir / "segmentation.h5", 'w') as f:
                                create_hdf5_dataset(f, 'labels', new_hspc[np.newaxis])
                            with open(hspc_tile_dir / "features.json", 'w') as f:
                                json.dump([{'id': h['id'], 'features': h['features']} for h in result['hspc_feats']], f)

                            del new_hspc, hspc_tile_cells

                    elif result['status'] == 'error':
                        print(f"  Tile error: {result['error']}")

                    # Cleanup
                    for key in ['mk_masks', 'hspc_masks', 'mk_feats', 'hspc_feats']:
                        if key in result:
                            del result[key]
                    del result
                    gc.collect()

    finally:
        # Cleanup
        if temp_init_dir.exists():
            try:
                shutil.rmtree(temp_init_dir)
            except:
                pass

    # Save summaries
    total_mk = 0
    total_hspc = 0

    for slide_name, data in slide_data.items():
        sr = slide_results[slide_name]
        output_dir = output_base / slide_name

        if sr['mk_count'] > 0 or sr['hspc_count'] > 0:
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                pixel_size_um = get_pixel_size_from_czi(data['czi_path'])
            except:
                pixel_size_um = None

            summary = {
                'czi_path': str(data['czi_path']),
                'pixel_size_um': pixel_size_um,
                'mk_count': sr['mk_count'],
                'hspc_count': sr['hspc_count'],
                'feature_count': '22 morphological + 256 SAM2 + 2048 ResNet = 2326'
            }
            with open(output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

        total_mk += sr['mk_count']
        total_hspc += sr['hspc_count']
        print(f"  {slide_name}: {sr['mk_count']} MKs, {sr['hspc_count']} HSPCs")

    # Export HTML while slide data is still in RAM
    if html_output_dir:
        export_html_from_ram(
            slide_data=slide_data,
            output_base=output_base,
            html_output_dir=html_output_dir,
            samples_per_page=samples_per_page,
            mk_min_area_um=mk_min_area_um,
            mk_max_area_um=mk_max_area_um
        )

    # Clear slide data from RAM
    del slide_data
    gc.collect()

    print(f"\n{'='*70}")
    print("ALL SLIDES COMPLETE")
    print(f"{'='*70}")
    print(f"Total MKs: {total_mk}")
    print(f"Total HSPCs: {total_hspc}")
    print(f"Output: {output_base}")


def process_tile_worker_with_data_and_slide(args):
    """
    Worker function for unified sampling mode - includes slide_name in result.
    """
    tile, img_data, output_dir, mk_min_area, mk_max_area, variance_threshold, calibration_block_size, slide_name = args

    global segmenter
    tid = tile['id']

    if segmenter is None:
        return {'tid': tid, 'status': 'error', 'error': "Segmenter not initialized", 'slide_name': slide_name}

    # Convert to RGB
    if img_data.ndim == 2:
        img_rgb = np.stack([img_data]*3, axis=-1)
    elif img_data.shape[2] == 4:
        img_rgb = img_data[:, :, :3]
    else:
        img_rgb = img_data

    if img_rgb.max() == 0:
        return {'tid': tid, 'status': 'empty', 'slide_name': slide_name}

    img_rgb = percentile_normalize(img_rgb, p_low=5, p_high=95)

    try:
        mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
            img_rgb, mk_min_area, mk_max_area
        )
        return {
            'tid': tid, 'status': 'success',
            'mk_masks': mk_masks, 'hspc_masks': hspc_masks,
            'mk_feats': mk_feats, 'hspc_feats': hspc_feats,
            'tile': tile, 'slide_name': slide_name
        }
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Processing error: {e}", 'slide_name': slide_name}


def process_tile_gpu_only(args):
    """
    GPU-only worker: receives pre-normalized tile, does only GPU work.
    CPU pre/post processing happens in main process threads.
    """
    tile, img_rgb, mk_min_area, mk_max_area, slide_name = args

    global segmenter
    tid = tile['id']

    if segmenter is None:
        return {'tid': tid, 'status': 'error', 'error': "Segmenter not initialized", 'slide_name': slide_name}

    if img_rgb.max() == 0:
        return {'tid': tid, 'status': 'empty', 'slide_name': slide_name}

    try:
        mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
            img_rgb, mk_min_area, mk_max_area
        )
        return {
            'tid': tid, 'status': 'success',
            'mk_masks': mk_masks, 'hspc_masks': hspc_masks,
            'mk_feats': mk_feats, 'hspc_feats': hspc_feats,
            'tile': tile, 'slide_name': slide_name
        }
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Processing error: {e}", 'slide_name': slide_name}


def preprocess_tile_cpu(slide_data, slide_name, tile, use_pinned_memory=True):
    """
    CPU pre-processing: extract tile from RAM, normalize.
    Runs in ThreadPoolExecutor in main process.

    If use_pinned_memory=True and CUDA available, allocates output in pinned memory
    for faster CPU→GPU transfer (DMA, doesn't block CPU).
    """
    img = slide_data[slide_name]['image']
    tile_img = img[tile['y']:tile['y']+tile['h'],
                   tile['x']:tile['x']+tile['w']].copy()

    # Convert to RGB
    if tile_img.ndim == 2:
        img_rgb = np.stack([tile_img]*3, axis=-1)
    elif tile_img.shape[2] == 4:
        img_rgb = tile_img[:, :, :3]
    else:
        img_rgb = tile_img

    # Normalize
    if img_rgb.max() > 0:
        img_rgb = percentile_normalize(img_rgb, p_low=5, p_high=95)

    # Use pinned memory for faster GPU transfer (double-buffering optimization)
    if use_pinned_memory and torch.cuda.is_available():
        try:
            # Convert to pinned numpy array via torch
            # This enables DMA transfer to GPU without blocking CPU
            img_tensor = torch.from_numpy(img_rgb.copy()).pin_memory()
            img_rgb = img_tensor.numpy()
        except Exception:
            pass  # Fall back to regular memory if pinning fails

    return (tile, img_rgb, slide_name)


def save_tile_results(result, output_base, slide_results, slide_results_lock):
    """
    CPU post-processing: save masks and features to disk.
    Runs in ThreadPoolExecutor in main process.

    Processes ONE tile at a time. Thread-safe via slide_results_lock.
    """
    if result['status'] != 'success':
        return result

    slide_name = result['slide_name']
    tile = result['tile']
    tid = result['tid']
    output_dir = output_base / slide_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Thread-safe access to per-slide counters
    with slide_results_lock:
        sr = slide_results[slide_name]

        # Reserve global IDs for this tile's cells (atomic operation)
        mk_start_gid = sr['mk_gid']
        mk_count = len(result.get('mk_feats', []))
        sr['mk_gid'] += mk_count
        sr['mk_count'] += mk_count

        hspc_start_gid = sr['hspc_gid']
        hspc_count = len(result.get('hspc_feats', []))
        sr['hspc_gid'] += hspc_count
        sr['hspc_count'] += hspc_count

    # Now do the actual I/O outside the lock (allows parallelism)
    if result['mk_feats']:
        mk_dir = output_dir / "mk" / "tiles"
        mk_tile_dir = mk_dir / str(tid)
        mk_tile_dir.mkdir(parents=True, exist_ok=True)

        with open(mk_tile_dir / "window.csv", 'w') as f:
            f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

        new_mk = np.zeros_like(result['mk_masks'])
        mk_tile_cells = []
        current_gid = mk_start_gid
        for feat in result['mk_feats']:
            old_id = int(feat['id'].split('_')[1])
            new_mk[result['mk_masks'] == old_id] = current_gid
            feat['id'] = f'det_{current_gid - 1}'
            feat['global_id'] = current_gid
            feat['center'][0] += tile['x']
            feat['center'][1] += tile['y']
            mk_tile_cells.append(current_gid)
            current_gid += 1

        with open(mk_tile_dir / "classes.csv", 'w') as f:
            for c in mk_tile_cells:
                f.write(f"{c}\n")
        with h5py.File(mk_tile_dir / "segmentation.h5", 'w') as f:
            create_hdf5_dataset(f, 'labels', new_mk[np.newaxis])
        with open(mk_tile_dir / "features.json", 'w') as f:
            json.dump([{'id': m['id'], 'features': m['features']} for m in result['mk_feats']], f)

    if result['hspc_feats']:
        hspc_dir = output_dir / "hspc" / "tiles"
        hspc_tile_dir = hspc_dir / str(tid)
        hspc_tile_dir.mkdir(parents=True, exist_ok=True)

        with open(hspc_tile_dir / "window.csv", 'w') as f:
            f.write(f"(slice({tile['y']}, {tile['y']+tile['h']}, None), slice({tile['x']}, {tile['x']+tile['w']}, None))")

        new_hspc = np.zeros_like(result['hspc_masks'])
        hspc_tile_cells = []
        current_gid = hspc_start_gid
        for feat in result['hspc_feats']:
            old_id = int(feat['id'].split('_')[1])
            new_hspc[result['hspc_masks'] == old_id] = current_gid
            feat['id'] = f'det_{current_gid - 1}'
            feat['global_id'] = current_gid
            feat['center'][0] += tile['x']
            feat['center'][1] += tile['y']
            hspc_tile_cells.append(current_gid)
            current_gid += 1

        with open(hspc_tile_dir / "classes.csv", 'w') as f:
            for c in hspc_tile_cells:
                f.write(f"{c}\n")
        with h5py.File(hspc_tile_dir / "segmentation.h5", 'w') as f:
            create_hdf5_dataset(f, 'labels', new_hspc[np.newaxis])
        with open(hspc_tile_dir / "features.json", 'w') as f:
            json.dump([{'id': h['id'], 'features': h['features']} for h in result['hspc_feats']], f)

    return {'slide_name': slide_name, 'mk_count': mk_count, 'hspc_count': hspc_count}


def run_pipelined_segmentation(
    czi_paths,
    output_base,
    slide_data,  # Pre-loaded slides
    sampled_tiles,  # Pre-sampled (slide_name, tile) tuples
    variance_threshold,
    mk_min_area=1000,
    mk_max_area=100000,
    num_workers=1,
    mk_classifier_path=None,
    hspc_classifier_path=None,
    preprocess_threads=None,  # CPU threads for pre-processing (default: 80% of cores)
    save_threads=None,  # CPU threads for saving (default: shares the 80% pool)
):
    """
    PIPELINED segmentation: parallel CPU pre/post processing + serial GPU.

    Architecture:
      CPU ThreadPool (pre)  -->  Queue  -->  GPU Worker  -->  Queue  -->  CPU ThreadPool (save)
           N threads                          1 process                       M threads

    Thread allocation (default 80% of CPU cores):
      - Pre-process: 60% of allocated threads (extracting tiles, normalizing)
      - Post-process: 40% of allocated threads (saving HDF5, features)

    This keeps the GPU constantly fed while CPU handles I/O in parallel.
    """
    # Calculate CPU thread allocation: use 80% of available cores
    total_cores = os.cpu_count() or 8
    cpu_pool_size = int(total_cores * 0.8)  # 80% of cores

    if preprocess_threads is None:
        preprocess_threads = max(4, int(cpu_pool_size * 0.6))  # 60% for pre-processing
    if save_threads is None:
        save_threads = max(2, int(cpu_pool_size * 0.4))  # 40% for post-processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import queue
    import threading
    import shutil

    output_base = Path(output_base)

    print(f"\n{'='*70}")
    print("PIPELINED PROCESSING")
    print(f"{'='*70}")
    print(f"Pre-process threads: {preprocess_threads}")
    print(f"GPU workers: {num_workers}")
    print(f"Save threads: {save_threads}")
    print(f"Tiles to process: {len(sampled_tiles)}")
    print(f"HDF5 compression: {HDF5_COMPRESSION_NAME}")
    print(f"Pinned memory: {'enabled' if torch.cuda.is_available() else 'disabled (no CUDA)'}")
    print(f"{'='*70}")

    # Setup GPU worker pool
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            for i in range(num_workers):
                gpu_queue.put(i % n_gpus)

    temp_init_dir = output_base / "temp_init"
    temp_init_dir.mkdir(parents=True, exist_ok=True)
    dummy_mm_path = temp_init_dir / "dummy.dat"
    dummy_arr = np.memmap(dummy_mm_path, dtype=np.uint8, mode='w+', shape=(100, 100, 3))
    dummy_arr.flush()
    del dummy_arr

    init_args = (mk_classifier_path, hspc_classifier_path, gpu_queue,
                 str(dummy_mm_path), (100, 100, 3), np.uint8)

    # Track results per slide
    slide_results = {Path(p).stem: {'mk_count': 0, 'hspc_count': 0, 'mk_gid': 1, 'hspc_gid': 1}
                     for p in czi_paths}
    slide_results_lock = threading.Lock()

    # Queues for pipelining
    preprocess_queue = queue.Queue(maxsize=preprocess_threads * 2)  # Buffer pre-processed tiles
    save_queue = queue.Queue()  # Results waiting to be saved

    # Stats
    stats = {'preprocessed': 0, 'processed': 0, 'saved': 0}
    stats_lock = threading.Lock()

    def preprocess_worker():
        """Thread that pre-processes tiles and feeds the queue."""
        with ThreadPoolExecutor(max_workers=preprocess_threads) as executor:
            futures = []
            for slide_name, tile in sampled_tiles:
                future = executor.submit(preprocess_tile_cpu, slide_data, slide_name, tile)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    preprocess_queue.put(result)
                    with stats_lock:
                        stats['preprocessed'] += 1
                except Exception as e:
                    print(f"Preprocess error: {e}")

            # Signal end
            preprocess_queue.put(None)

    def save_worker():
        """Thread that saves results using a thread pool for parallel I/O."""
        with ThreadPoolExecutor(max_workers=save_threads) as save_executor:
            pending_saves = []
            while True:
                item = save_queue.get()
                if item is None:
                    break
                # Submit save job to thread pool (pass lock for thread-safe counter access)
                future = save_executor.submit(save_tile_results, item, output_base, slide_results, slide_results_lock)
                pending_saves.append(future)
                save_queue.task_done()

                # Check completed saves
                done = [f for f in pending_saves if f.done()]
                for f in done:
                    try:
                        f.result()
                        with stats_lock:
                            stats['saved'] += 1
                    except Exception as e:
                        print(f"Save error: {e}")
                    pending_saves.remove(f)

            # Wait for remaining saves
            for future in as_completed(pending_saves):
                try:
                    future.result()
                    with stats_lock:
                        stats['saved'] += 1
                except Exception as e:
                    print(f"Save error: {e}")

    try:
        # Start save worker thread
        save_thread = threading.Thread(target=save_worker, daemon=True)
        save_thread.start()

        # Start preprocess thread
        preprocess_thread = threading.Thread(target=preprocess_worker, daemon=True)
        preprocess_thread.start()

        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
            print(f"GPU worker pool ready")

            def gpu_input_generator():
                """Generate GPU inputs from preprocess queue."""
                while True:
                    item = preprocess_queue.get()
                    if item is None:
                        break
                    tile, img_rgb, slide_name = item
                    yield (tile, img_rgb, mk_min_area, mk_max_area, slide_name)

            with tqdm(total=len(sampled_tiles), desc="Processing") as pbar:
                for result in pool.imap_unordered(process_tile_gpu_only, gpu_input_generator()):
                    with stats_lock:
                        stats['processed'] += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'pre': stats['preprocessed'],
                        'gpu': stats['processed'],
                        'save': stats['saved']
                    })

                    if result['status'] == 'success':
                        save_queue.put(result)
                    elif result['status'] == 'error':
                        print(f"  GPU error: {result['error']}")

                    # Cleanup large arrays
                    for key in ['mk_masks', 'hspc_masks', 'mk_feats', 'hspc_feats']:
                        if key in result:
                            del result[key]
                    gc.collect()

        # Wait for saves to complete
        save_queue.put(None)  # Signal save worker to stop
        save_thread.join(timeout=60)

    finally:
        if temp_init_dir.exists():
            try:
                shutil.rmtree(temp_init_dir)
            except:
                pass

    # Summary
    total_mk = sum(sr['mk_count'] for sr in slide_results.values())
    total_hspc = sum(sr['hspc_count'] for sr in slide_results.values())

    print(f"\n{'='*70}")
    print("PIPELINED PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total MKs: {total_mk}")
    print(f"Total HSPCs: {total_hspc}")

    return slide_results


def process_tile_worker_with_data(args):
    """
    Worker function that receives tile image data directly (for multi-slide mode).
    Models are already loaded in the worker via init_worker.
    """
    tile, img_data, output_dir, mk_min_area, mk_max_area, variance_threshold, calibration_block_size = args

    global segmenter
    tid = tile['id']

    if segmenter is None:
        return {'tid': tid, 'status': 'error', 'error': "Segmenter not initialized"}

    # Convert to RGB
    if img_data.ndim == 2:
        img_rgb = np.stack([img_data]*3, axis=-1)
    elif img_data.shape[2] == 4:
        img_rgb = img_data[:, :, :3]
    else:
        img_rgb = img_data

    if img_rgb.max() == 0:
        return {'tid': tid, 'status': 'empty'}

    img_rgb = percentile_normalize(img_rgb, p_low=5, p_high=95)
    has_tissue_content, _ = has_tissue(img_rgb, variance_threshold, block_size=calibration_block_size)
    if not has_tissue_content:
        return {'tid': tid, 'status': 'no_tissue'}

    try:
        mk_masks, hspc_masks, mk_feats, hspc_feats = segmenter.process_tile(
            img_rgb, mk_min_area, mk_max_area
        )
        return {
            'tid': tid, 'status': 'success',
            'mk_masks': mk_masks, 'hspc_masks': hspc_masks,
            'mk_feats': mk_feats, 'hspc_feats': hspc_feats,
            'tile': tile
        }
    except Exception as e:
        return {'tid': tid, 'status': 'error', 'error': f"Processing error: {e}"}


def main():
    parser = argparse.ArgumentParser(description='Unified MK + HSPC segmentation')
    parser.add_argument('--czi-path', help='Single CZI file path')
    parser.add_argument('--czi-paths', nargs='+', help='Multiple CZI file paths (models loaded once)')
    parser.add_argument('--output-dir', required=True, help='Output directory (subdirs created per slide if multiple)')
    parser.add_argument('--mk-min-area-um', type=float, default=200,
                        help='Minimum MK area in µm² (only applies to MKs)')
    parser.add_argument('--mk-max-area-um', type=float, default=2000,
                        help='Maximum MK area in µm² (only applies to MKs)')
    parser.add_argument('--tile-size', type=int, default=4096)
    parser.add_argument('--overlap', type=int, default=512)
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--calibration-block-size', type=int, default=512,
                        help='Block size for variance calculation in tissue calibration and detection.')
    parser.add_argument('--calibration-samples', type=int, default=50,
                        help='Number of tiles to sample for tissue calibration.')
    parser.add_argument('--mk-classifier', type=str, help='Path to trained MK classifier (.pkl)')
    parser.add_argument('--hspc-classifier', type=str, help='Path to trained HSPC classifier (.pkl)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers for tile processing.')
    parser.add_argument('--html-output-dir', type=str, default=None,
                        help='Directory for HTML export (default: output-dir/../docs)')
    parser.add_argument('--samples-per-page', type=int, default=300,
                        help='Number of cell samples per HTML page')

    args = parser.parse_args()

    # Build list of CZI paths
    czi_paths = []
    if args.czi_paths:
        czi_paths = [Path(p) for p in args.czi_paths]
    elif args.czi_path:
        czi_paths = [Path(args.czi_path)]
    else:
        parser.error("Must provide either --czi-path or --czi-paths")

    # Validate all paths exist
    for p in czi_paths:
        if not p.exists():
            parser.error(f"CZI file not found: {p}")

    # Convert µm² to px² using pixel size (0.1725 µm/px)
    PIXEL_SIZE_UM = 0.1725
    um_to_px_factor = PIXEL_SIZE_UM ** 2  # 0.02975625
    mk_min_area_px = int(args.mk_min_area_um / um_to_px_factor)
    mk_max_area_px = int(args.mk_max_area_um / um_to_px_factor)

    print(f"MK area filter: {args.mk_min_area_um}-{args.mk_max_area_um} µm² = {mk_min_area_px}-{mk_max_area_px} px²")

    # Set HTML output directory (default: output_dir/../docs)
    html_output_dir = args.html_output_dir
    if html_output_dir is None:
        html_output_dir = Path(args.output_dir).parent / "docs"
    html_output_dir = Path(html_output_dir)

    # Process slides
    if len(czi_paths) == 1:
        # Single slide - use output_dir directly
        run_unified_segmentation(
            czi_paths[0], args.output_dir,
            mk_min_area_px, mk_max_area_px,
            args.tile_size, args.overlap, args.sample_fraction,
            calibration_block_size=args.calibration_block_size,
            calibration_samples=args.calibration_samples,
            num_workers=args.num_workers,
            mk_classifier_path=args.mk_classifier,
            hspc_classifier_path=args.hspc_classifier
        )
    else:
        # Multiple slides - load models once, process all slides
        run_multi_slide_segmentation(
            czi_paths, args.output_dir,
            mk_min_area_px, mk_max_area_px,
            args.tile_size, args.overlap, args.sample_fraction,
            calibration_block_size=args.calibration_block_size,
            calibration_samples=args.calibration_samples,
            num_workers=args.num_workers,
            mk_classifier_path=args.mk_classifier,
            hspc_classifier_path=args.hspc_classifier,
            html_output_dir=str(html_output_dir),
            samples_per_page=args.samples_per_page,
            mk_min_area_um=args.mk_min_area_um,
            mk_max_area_um=args.mk_max_area_um
        )


if __name__ == "__main__":
    main()
