"""
NMJ (Neuromuscular Junction) Segmentation Pipeline

Detects NMJs in muscle tissue CZI images using:
1. Intensity thresholding (bright regions)
2. Elongation filtering (skeleton length / sqrt(area))
3. HTML annotation export for training data collection

Usage:
    python run_nmj_segmentation.py \
        --czi-path /path/to/muscle.czi \
        --output-dir /path/to/output \
        --sample-fraction 0.10
"""

import os
from pathlib import Path

import gc
import numpy as np
import h5py
import json
import argparse
from tqdm import tqdm
from scipy import ndimage
from skimage.morphology import skeletonize, remove_small_objects, binary_opening, binary_closing, disk
from skimage.measure import label, regionprops
from PIL import Image

from segmentation.utils.logging import get_logger, setup_logging, log_parameters
from segmentation.io.czi_loader import get_loader, CZILoader
from segmentation.io.html_export import (
    create_hdf5_dataset,  # Import shared HDF5 utilities (Issue #7)
    HDF5_COMPRESSION_KWARGS,
    HDF5_COMPRESSION_NAME,
)

logger = get_logger(__name__)


# =============================================================================
# NMJ DETECTION (Simple: threshold + elongation)
# =============================================================================

def detect_nmjs(image, intensity_percentile=99, min_area=150, min_skeleton_length=30, min_elongation=1.5):
    """
    Detect NMJs using intensity threshold + elongation filter.

    Args:
        image: RGB or grayscale image
        intensity_percentile: Percentile for bright region threshold (default 99)
        min_area: Minimum NMJ area in pixels
        min_skeleton_length: Minimum skeleton length (elongation measure)
        min_elongation: Minimum elongation ratio (skeleton_length / sqrt(area))

    Returns:
        nmj_masks: Label array with NMJ IDs
        nmj_features: List of feature dicts
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = np.mean(image[:, :, :3], axis=2)
    else:
        gray = image.astype(float)

    # Threshold bright regions
    threshold = np.percentile(gray, intensity_percentile)
    bright_mask = gray > threshold

    # Morphological cleanup
    bright_mask = binary_opening(bright_mask, disk(1))
    bright_mask = binary_closing(bright_mask, disk(2))
    bright_mask = remove_small_objects(bright_mask, min_size=min_area)

    # Label connected components
    labeled = label(bright_mask)
    props = regionprops(labeled, intensity_image=gray)

    # Filter by elongation
    nmj_masks = np.zeros(image.shape[:2], dtype=np.uint32)
    nmj_features = []
    nmj_id = 1

    for prop in props:
        if prop.area < min_area:
            continue

        region_mask = labeled == prop.label
        skeleton = skeletonize(region_mask)
        skeleton_length = skeleton.sum()
        elongation = skeleton_length / np.sqrt(prop.area) if prop.area > 0 else 0

        if skeleton_length >= min_skeleton_length and elongation >= min_elongation:
            nmj_masks[region_mask] = nmj_id

            nmj_features.append({
                'id': f'nmj_{nmj_id}',
                'area': int(prop.area),
                'skeleton_length': int(skeleton_length),
                'elongation': float(elongation),
                'eccentricity': float(prop.eccentricity),
                'mean_intensity': float(prop.mean_intensity),
                'centroid': [float(prop.centroid[1]), float(prop.centroid[0])],  # x, y
                'bbox': list(prop.bbox),
                'perimeter': float(prop.perimeter),
                'solidity': float(prop.solidity),
            })

            nmj_id += 1

    return nmj_masks, nmj_features


# =============================================================================
# HTML EXPORT (using shared utilities)
# =============================================================================

from segmentation.io.html_export import percentile_normalize, draw_mask_contour, image_to_base64

# NMJ uses wider percentile range for normalization
NMJ_PERCENTILE_LOW = 1
NMJ_PERCENTILE_HIGH = 99.5


def create_nmj_index_html(output_dir, total_nmjs, total_pages):
    """Create the main index.html page."""
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>NMJ Annotation Review</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 20px; }}
        .header {{ background: #111; padding: 20px; border: 1px solid #333; margin-bottom: 20px; text-align: center; }}
        h1 {{ font-size: 1.5em; font-weight: normal; margin-bottom: 15px; }}
        .stats {{ display: flex; justify-content: center; gap: 30px; margin: 20px 0; flex-wrap: wrap; }}
        .stat {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; }}
        .stat .number {{ display: block; font-size: 2em; margin-top: 10px; }}
        .section {{ margin: 40px 0; }}
        .controls {{ text-align: center; margin: 30px 0; }}
        .btn {{ padding: 15px 30px; background: #1a1a1a; border: 1px solid #333; color: #ddd; cursor: pointer; font-family: monospace; font-size: 1.1em; margin: 10px; text-decoration: none; display: inline-block; }}
        .btn:hover {{ background: #222; }}
        .btn-primary {{ border-color: #4a4; color: #4a4; }}
        .btn-export {{ border-color: #44a; color: #44a; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NMJ Annotation Review</h1>
        <p style="color: #888;">Neuromuscular Junction Detection</p>
        <div class="stats">
            <div class="stat"><span>Total NMJ Candidates</span><span class="number">{total_nmjs:,}</span></div>
            <div class="stat"><span>Pages</span><span class="number">{total_pages}</span></div>
        </div>
    </div>
    <div class="section">
        <div class="controls">
            <a href="nmj_page1.html" class="btn btn-primary">Start Review</a>
        </div>
    </div>
    <div class="section">
        <h2 style="text-align: center; margin-bottom: 15px;">Export Annotations</h2>
        <div class="controls">
            <button class="btn btn-export" onclick="exportAnnotations()">Download Annotations JSON</button>
        </div>
    </div>
    <script>
        function exportAnnotations() {{
            const labels = {{ positive: [], negative: [], unsure: [] }};
            for (let i = 0; i < localStorage.length; i++) {{
                const key = localStorage.key(i);
                if (key.startsWith('nmj_labels_page')) {{
                    try {{
                        const pageLabels = JSON.parse(localStorage.getItem(key));
                        for (const [uid, label] of Object.entries(pageLabels)) {{
                            if (label === 1) labels.positive.push(uid);
                            else if (label === 0) labels.negative.push(uid);
                            else if (label === 2) labels.unsure.push(uid);
                        }}
                    }} catch(e) {{ console.error(e); }}
                }}
            }}
            const blob = new Blob([JSON.stringify(labels, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'nmj_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''
    with open(Path(output_dir) / 'index.html', 'w') as f:
        f.write(html)


def generate_nmj_page_html(samples, page_num, total_pages):
    """Generate HTML for a single annotation page."""
    nav_html = '<div class="page-nav">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="nmj_page{page_num-1}.html" class="nav-btn">Prev</a>'
    nav_html += f'<span class="page-info">Page {page_num} of {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="nmj_page{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += '</div>'

    cards_html = ""
    for sample in samples:
        slide = sample.get('slide', 'unknown').replace('.', '-')
        tile_id = str(sample.get('tile_id', '0'))
        det_id = sample.get('det_id', 'unknown')
        uid = f"{slide}_{tile_id}_{det_id}"
        area_px = sample.get('area_px', 0)
        area_um2 = sample.get('area_um2', 0)
        elongation = sample.get('elongation', 0)
        img_b64 = sample['image']

        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1">
            <div class="card-img-container">
                <img src="data:image/png;base64,{img_b64}" alt="{det_id}">
            </div>
            <div class="card-info">
                <div>
                    <div class="card-id">{slide} | {tile_id} | {det_id}</div>
                    <div class="card-area">{area_um2:.1f} &micro;m&sup2; | {area_px:.0f} px | elong: {elongation:.2f}</div>
                </div>
                <div class="buttons">
                    <button class="btn btn-yes" onclick="setLabel('{uid}', 1)">Yes (Y)</button>
                    <button class="btn btn-unsure" onclick="setLabel('{uid}', 2)">? (U)</button>
                    <button class="btn btn-no" onclick="setLabel('{uid}', 0)">No (N)</button>
                </div>
            </div>
        </div>
'''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>NMJ Review - Page {page_num}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: monospace; background: #0a0a0a; color: #ddd; padding: 10px; }}
        .header {{ background: #111; padding: 15px; border: 1px solid #333; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
        h1 {{ font-size: 1.2em; font-weight: normal; }}
        .page-nav {{ display: flex; gap: 10px; align-items: center; }}
        .nav-btn {{ padding: 8px 15px; background: #1a1a1a; border: 1px solid #333; color: #ddd; text-decoration: none; }}
        .nav-btn:hover {{ background: #222; }}
        .page-info {{ padding: 8px 15px; color: #888; }}
        .stats {{ display: flex; gap: 20px; color: #888; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }}
        .card {{ background: #111; border: 1px solid #333; overflow: hidden; }}
        .card[data-label="1"] {{ border-color: #4a4; }}
        .card[data-label="0"] {{ border-color: #a44; }}
        .card[data-label="2"] {{ border-color: #aa4; }}
        .card.selected {{ box-shadow: 0 0 0 3px #fff; }}
        .card-img-container {{ display: flex; justify-content: center; align-items: center; background: #000; padding: 5px; }}
        .card-img-container img {{ max-width: 300px; max-height: 300px; }}
        .card-info {{ padding: 10px; display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
        .card-id {{ font-size: 0.8em; color: #888; }}
        .card-area {{ font-size: 0.75em; color: #666; margin-top: 3px; }}
        .buttons {{ display: flex; gap: 5px; }}
        .btn {{ padding: 5px 10px; background: #1a1a1a; border: 1px solid #333; color: #ddd; cursor: pointer; font-family: monospace; }}
        .btn:hover {{ background: #222; }}
        .btn-yes {{ border-color: #4a4; color: #4a4; }}
        .btn-no {{ border-color: #a44; color: #a44; }}
        .btn-unsure {{ border-color: #aa4; color: #aa4; }}
        .keyboard-hint {{ text-align: center; padding: 10px; color: #555; font-size: 0.85em; margin-top: 15px; }}
        .footer {{ background: #111; padding: 15px; border: 1px solid #333; margin-top: 15px; display: flex; justify-content: center; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NMJ Review</h1>
        {nav_html}
        <div class="stats">
            <span>Page: <span id="localYes">0</span> Yes / <span id="localNo">0</span> No</span>
            <span>Total: <span id="globalYes">0</span> Yes / <span id="globalNo">0</span> No</span>
        </div>
    </div>
    <div class="grid">
        {cards_html}
    </div>
    <div class="keyboard-hint">
        Keyboard: Y=Yes, N=No, U=Unsure, Arrow keys=Navigate
    </div>
    <div class="footer">
        {nav_html}
    </div>
    <script>
        const PAGE_NUM = {page_num};
        const STORAGE_KEY = 'nmj_labels_page' + PAGE_NUM;
        let labels = {{}};
        let selectedIdx = -1;
        const cards = document.querySelectorAll('.card');

        try {{
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) labels = JSON.parse(saved);
        }} catch(e) {{}}

        for (const [uid, label] of Object.entries(labels)) {{
            const card = document.getElementById(uid);
            if (card) card.dataset.label = label;
        }}

        function setLabel(uid, label, autoAdvance = false) {{
            labels[uid] = label;
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
            const card = document.getElementById(uid);
            if (card) card.dataset.label = label;
            updateStats();
            // Auto-advance only on keyboard (not button clicks)
            if (autoAdvance && selectedIdx >= 0 && selectedIdx < cards.length - 1) {{
                selectCard(selectedIdx + 1);
            }}
        }}

        function updateStats() {{
            let localYes = 0, localNo = 0;
            for (const v of Object.values(labels)) {{
                if (v === 1) localYes++;
                else if (v === 0) localNo++;
            }}
            document.getElementById('localYes').textContent = localYes;
            document.getElementById('localNo').textContent = localNo;

            let globalYes = 0, globalNo = 0;
            for (let i = 0; i < localStorage.length; i++) {{
                const key = localStorage.key(i);
                if (key.startsWith('nmj_labels_page')) {{
                    try {{
                        const pageLabels = JSON.parse(localStorage.getItem(key));
                        for (const v of Object.values(pageLabels)) {{
                            if (v === 1) globalYes++;
                            else if (v === 0) globalNo++;
                        }}
                    }} catch(e) {{}}
                }}
            }}
            document.getElementById('globalYes').textContent = globalYes;
            document.getElementById('globalNo').textContent = globalNo;
        }}

        function selectCard(idx) {{
            cards.forEach(c => c.classList.remove('selected'));
            if (idx >= 0 && idx < cards.length) {{
                selectedIdx = idx;
                cards[idx].classList.add('selected');
                cards[idx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
                selectCard(Math.min(selectedIdx + 1, cards.length - 1));
            }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
                selectCard(Math.max(selectedIdx - 1, 0));
            }} else if (selectedIdx >= 0) {{
                const uid = cards[selectedIdx].id;
                if (e.key.toLowerCase() === 'y') setLabel(uid, 1, true);
                else if (e.key.toLowerCase() === 'n') setLabel(uid, 0, true);
                else if (e.key.toLowerCase() === 'u') setLabel(uid, 2, true);
            }}
        }});

        updateStats();
    </script>
</body>
</html>'''
    return html


# =============================================================================
# CZI PROCESSING
# =============================================================================

def process_czi_for_nmj(czi_path, output_dir,
                        tile_size=3000,
                        sample_fraction=0.10,
                        samples_per_page=300,
                        intensity_percentile=99,
                        min_area=150,
                        min_skeleton_length=30,
                        min_elongation=1.5,
                        channel=1,
                        load_to_ram=True):
    """
    Process a CZI file for NMJ detection.

    Args:
        czi_path: Path to CZI file
        output_dir: Output directory for results
        tile_size: Size of tiles to process
        sample_fraction: Fraction of tiles to sample
        samples_per_page: Number of samples per HTML page
        intensity_percentile: Percentile for intensity threshold
        min_area: Minimum NMJ area in pixels
        min_skeleton_length: Minimum skeleton length
        min_elongation: Minimum elongation ratio
        channel: Channel index to use
        load_to_ram: If True, load entire channel into RAM (faster for network mounts)
    """
    czi_path = Path(czi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_name = czi_path.stem

    # Setup logging
    setup_logging(level="INFO")

    logger.info("=" * 60)
    logger.info(f"NMJ SEGMENTATION: {slide_name}")
    logger.info("=" * 60)

    # Load CZI using shared loader (RAM-first for better performance)
    logger.info("Loading CZI file...")
    loader = CZILoader(czi_path, load_to_ram=load_to_ram, channel=channel)

    # Get mosaic info from loader
    x_start, y_start = loader.mosaic_origin
    width, height = loader.mosaic_size
    pixel_size_um = loader.get_pixel_size()

    logger.info(f"  Mosaic dimensions: {width} x {height}")
    logger.info(f"  Mosaic origin: ({x_start}, {y_start})")
    logger.info(f"  Pixel size: {pixel_size_um:.4f} um/px")
    logger.info(f"  Detection params: p{intensity_percentile}, min_area={min_area}, min_skel={min_skeleton_length}, min_elong={min_elongation}")
    logger.info(f"  Channel: {channel}")
    logger.info(f"  RAM loading: {load_to_ram}")

    # Generate tile grid over actual data extent
    tiles = []
    for y in range(y_start, y_start + height, tile_size):
        for x in range(x_start, x_start + width, tile_size):
            tiles.append((x, y))

    logger.info(f"  Total tiles: {len(tiles)}")

    # Sample tiles
    n_sample = max(1, int(len(tiles) * sample_fraction))
    sampled_indices = np.random.choice(len(tiles), n_sample, replace=False)
    sampled_tiles = [tiles[i] for i in sampled_indices]

    logger.info(f"  Sampled tiles: {len(sampled_tiles)} ({sample_fraction*100:.1f}%)")

    # Process tiles
    logger.info("Processing tiles...")
    all_samples = []
    tiles_dir = output_dir / slide_name / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    for tile_x, tile_y in tqdm(sampled_tiles, desc="Tiles"):
        # Read tile using shared loader
        try:
            tile_data = loader.get_tile(tile_x, tile_y, tile_size)
            if tile_data is None or tile_data.size == 0:
                continue

            # Skip empty tiles (no data)
            if tile_data.max() == 0:
                continue
        except Exception as e:
            logger.warning(f"Failed to read tile ({tile_x}, {tile_y}): {e}")
            continue

        # Convert to RGB
        if tile_data.ndim == 2:
            tile_rgb = np.stack([tile_data] * 3, axis=-1)
        else:
            tile_rgb = tile_data[:, :, :3] if tile_data.shape[2] >= 3 else np.stack([tile_data[:,:,0]] * 3, axis=-1)

        # Detect NMJs
        nmj_masks, nmj_features = detect_nmjs(
            tile_rgb,
            intensity_percentile=intensity_percentile,
            min_area=min_area,
            min_skeleton_length=min_skeleton_length,
            min_elongation=min_elongation
        )

        if len(nmj_features) == 0:
            continue

        # Save masks
        tile_id = f"tile_{tile_x}_{tile_y}"
        tile_out = tiles_dir / tile_id
        tile_out.mkdir(exist_ok=True)

        with h5py.File(tile_out / "nmj_masks.h5", 'w') as f:
            create_hdf5_dataset(f, 'masks', nmj_masks.astype(np.uint16))

        with open(tile_out / "nmj_features.json", 'w') as f:
            json.dump(nmj_features, f)

        # Create samples for HTML export (NMJ uses wider percentile range)
        tile_rgb_norm = percentile_normalize(tile_rgb, p_low=NMJ_PERCENTILE_LOW, p_high=NMJ_PERCENTILE_HIGH)

        for feat in nmj_features:
            nmj_id = int(feat['id'].split('_')[1])
            mask = nmj_masks == nmj_id

            if mask.sum() == 0:
                continue

            # Center crop on mask centroid
            ys, xs = np.where(mask)
            cy, cx = int(np.mean(ys)), int(np.mean(xs))

            # Calculate crop size based on mask extent with 7.5x zoom out
            mask_h = ys.max() - ys.min()
            mask_w = xs.max() - xs.min()
            mask_extent = max(mask_h, mask_w)
            crop_size = int(mask_extent * 7.5) if mask_extent > 0 else 500  # 7.5x zoom out for context

            half = crop_size // 2
            y1 = max(0, cy - half)
            y2 = min(tile_rgb.shape[0], cy + half)
            x1 = max(0, cx - half)
            x2 = min(tile_rgb.shape[1], cx + half)

            # Validate crop bounds before extracting
            if y2 <= y1 or x2 <= x1:
                logger.warning(f"Invalid crop bounds: y1={y1}, y2={y2}, x1={x1}, x2={x2}, skipping {feat['id']}")
                continue

            crop = tile_rgb_norm[y1:y2, x1:x2].copy()
            crop_mask = mask[y1:y2, x1:x2]

            # Draw contour
            crop_with_contour = draw_mask_contour(crop, crop_mask, color=(0, 255, 0), thickness=2)

            # Resize to 300x300
            pil_img = Image.fromarray(crop_with_contour)
            pil_img = pil_img.resize((300, 300), Image.LANCZOS)

            # Convert to base64 (image_to_base64 returns tuple)
            img_b64, _ = image_to_base64(np.array(pil_img))

            # Calculate area in µm²
            area_um2 = feat['area'] * (pixel_size_um ** 2)

            all_samples.append({
                'slide': slide_name,
                'tile_id': tile_id,
                'det_id': feat['id'],
                'area_px': feat['area'],
                'area_um2': area_um2,
                'elongation': feat['elongation'],
                'image': img_b64
            })

        del tile_data, tile_rgb, nmj_masks
        gc.collect()

    # Generate HTML export
    logger.info(f"Generating HTML export ({len(all_samples)} samples)...")

    html_dir = output_dir / "html"
    html_dir.mkdir(exist_ok=True)

    if all_samples:
        # Sort by area (smallest first) for easier review
        all_samples.sort(key=lambda x: x['area_px'], reverse=False)

        pages = [all_samples[i:i+samples_per_page] for i in range(0, len(all_samples), samples_per_page)]
        total_pages = len(pages)

        create_nmj_index_html(html_dir, len(all_samples), total_pages)

        for page_num, page_samples in enumerate(pages, 1):
            html = generate_nmj_page_html(page_samples, page_num, total_pages)
            with open(html_dir / f"nmj_page{page_num}.html", 'w') as f:
                f.write(html)
            logger.debug(f"  Page {page_num}: {len(page_samples)} samples")

    logger.info("Done!")
    logger.info(f"  HTML output: {html_dir}")
    logger.info(f"  Total NMJ candidates: {len(all_samples)}")

    return all_samples


def main():
    parser = argparse.ArgumentParser(description='NMJ Segmentation Pipeline')
    parser.add_argument('--czi-path', type=str, required=True, help='Path to CZI file')
    parser.add_argument('--output-dir', type=str, default='/home/dude/nmj_output', help='Output directory')
    parser.add_argument('--tile-size', type=int, default=3000, help='Tile size in pixels')
    parser.add_argument('--sample-fraction', type=float, default=0.10, help='Fraction of tiles to process')
    parser.add_argument('--samples-per-page', type=int, default=300, help='Samples per HTML page')
    parser.add_argument('--intensity-percentile', type=float, default=99, help='Intensity percentile threshold')
    parser.add_argument('--min-area', type=int, default=150, help='Min NMJ area in pixels')
    parser.add_argument('--min-skeleton-length', type=int, default=30, help='Min skeleton length')
    parser.add_argument('--min-elongation', type=float, default=1.5, help='Min elongation ratio')
    parser.add_argument('--channel', type=int, default=1, help='Channel index to use (default 1 for 647)')
    parser.add_argument('--load-to-ram', action='store_true', default=True,
                        help='Load entire channel into RAM (default: True, faster for network mounts)')
    parser.add_argument('--no-load-to-ram', dest='load_to_ram', action='store_false',
                        help='Disable RAM loading (use less memory but slower)')

    args = parser.parse_args()

    process_czi_for_nmj(
        czi_path=args.czi_path,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        sample_fraction=args.sample_fraction,
        samples_per_page=args.samples_per_page,
        intensity_percentile=args.intensity_percentile,
        min_area=args.min_area,
        min_skeleton_length=args.min_skeleton_length,
        min_elongation=args.min_elongation,
        channel=args.channel,
        load_to_ram=args.load_to_ram
    )


if __name__ == '__main__':
    main()
