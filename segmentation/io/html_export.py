"""
Unified HTML export module for cell annotation interfaces.

Provides a consistent dark-themed annotation interface for all cell types
(MK, HSPC, NMJ, etc.) with:
- Keyboard navigation (Y/N keys for labeling, arrows for navigation)
- Local + global annotation statistics
- Single global localStorage key per cell type
- JSON export functionality
"""

import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage


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


def get_largest_connected_component(mask):
    """Extract only the largest connected component from a binary mask."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    # Find largest component
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1
    return labeled == largest_label


def percentile_normalize(image, p_low=1, p_high=99.5):
    """
    Normalize image using percentiles.

    Args:
        image: 2D or 3D numpy array
        p_low: Lower percentile for normalization (default 1)
        p_high: Upper percentile for normalization (default 99.5)

    Returns:
        uint8 normalized image
    """
    if image.ndim == 2:
        low_val = np.percentile(image, p_low)
        high_val = np.percentile(image, p_high)
        if high_val > low_val:
            normalized = (image.astype(np.float32) - low_val) / (high_val - low_val) * 255
            return np.clip(normalized, 0, 255).astype(np.uint8)
        return image.astype(np.uint8)
    else:
        # Vectorized multi-channel normalization
        h, w, c = image.shape
        result = np.zeros_like(image, dtype=np.uint8)
        # Compute percentiles for all channels at once
        flat = image.reshape(-1, c)
        low_vals = np.percentile(flat, p_low, axis=0)
        high_vals = np.percentile(flat, p_high, axis=0)
        for ch in range(c):
            if high_vals[ch] > low_vals[ch]:
                normalized = (image[:, :, ch].astype(np.float32) - low_vals[ch]) / (high_vals[ch] - low_vals[ch]) * 255
                result[:, :, ch] = np.clip(normalized, 0, 255).astype(np.uint8)
            else:
                result[:, :, ch] = image[:, :, ch].astype(np.uint8)
        return result


def draw_mask_contour(img_array, mask, color=(0, 255, 0), thickness=2, dotted=False, use_cv2=True):
    """
    Draw mask contour on image.

    Args:
        img_array: RGB image array (or grayscale, will be converted)
        mask: Binary mask
        color: RGB tuple for contour color
        thickness: Contour thickness in pixels
        dotted: Whether to use dotted line
        use_cv2: Use OpenCV for faster, smoother contours (default True)

    Returns:
        Image with contour drawn (always RGB)
    """
    import cv2

    # Ensure RGB
    if img_array.ndim == 2:
        img_out = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_out = img_array.copy()

    if use_cv2 and not dotted:
        # Use cv2.drawContours for smooth, thick lines
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
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
        for i, (y, x) in enumerate(zip(ys, xs)):
            if i % 3 == 0:  # Every 3rd pixel for dotted effect
                if 0 <= y < img_out.shape[0] and 0 <= x < img_out.shape[1]:
                    img_out[y, x] = color
    else:
        for y, x in zip(ys, xs):
            if 0 <= y < img_out.shape[0] and 0 <= x < img_out.shape[1]:
                img_out[y, x] = color

    return img_out


def image_to_base64(img_array, format='JPEG', quality=85):
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

    buffer = BytesIO()
    if format.upper() == 'JPEG':
        pil_img.save(buffer, format='JPEG', quality=quality)
        mime_type = 'jpeg'
    else:
        pil_img.save(buffer, format='PNG', optimize=True)
        mime_type = 'png'

    return base64.b64encode(buffer.getvalue()).decode('utf-8'), mime_type


def get_css():
    """Get the unified CSS styles."""
    return '''
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
            max-width: 100%;
            max-height: 100%;
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
    '''


def get_js(cell_type, total_pages, experiment_name=None):
    """
    Get the unified JavaScript for annotation handling.

    Args:
        cell_type: Type identifier (e.g., 'nmj', 'mk', 'hspc')
        total_pages: Total number of pages
        experiment_name: Optional experiment name for localStorage key isolation
                        If provided, key is '{cell_type}_{experiment_name}_annotations'
                        Otherwise, key is '{cell_type}_annotations'

    Returns:
        JavaScript code string
    """
    # Build storage key with optional experiment name
    if experiment_name:
        storage_key = f"{cell_type}_{experiment_name}_annotations"
    else:
        storage_key = f"{cell_type}_annotations"

    return f'''
        const CELL_TYPE = '{cell_type}';
        const EXPERIMENT_NAME = '{experiment_name or ""}';
        const TOTAL_PAGES = {total_pages};
        const STORAGE_KEY = '{storage_key}';

        let labels = {{}};
        let selectedIdx = -1;
        const cards = document.querySelectorAll('.card');

        // Load from localStorage
        function loadAnnotations() {{
            try {{
                const saved = localStorage.getItem(STORAGE_KEY);
                if (saved) labels = JSON.parse(saved);
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

            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
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

            // Count global
            for (const v of Object.values(labels)) {{
                if (v === 1) globalYes++;
                else if (v === 0) globalNo++;
            }}

            document.getElementById('localYes').textContent = localYes;
            document.getElementById('localNo').textContent = localNo;
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

        function exportAnnotations() {{
            const data = {{
                cell_type: CELL_TYPE,
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
            a.download = CELL_TYPE + '_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
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
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
            updateStats();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            labels = {{}};
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
            cards.forEach(card => {{
                card.classList.remove('labeled-yes', 'labeled-no', 'labeled-unsure');
                card.dataset.label = -1;
            }});
            updateStats();
            alert('All annotations cleared.');
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
    '''


def generate_annotation_page(
    samples,
    cell_type,
    page_num,
    total_pages,
    title=None,
    page_prefix='page',
    experiment_name=None,
    channel_legend=None,
    subtitle=None,
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

    Returns:
        HTML string
    """
    if title is None:
        title = cell_type.upper()

    # Build navigation
    nav_html = '<div class="nav-buttons">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{page_prefix}_{page_num-1}.html" class="nav-btn">Prev</a>'
    nav_html += f'<span class="page-info">Page {page_num} / {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{page_prefix}_{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += '</div>'

    # Build channel legend HTML if provided
    channel_legend_html = ''
    if channel_legend:
        channel_legend_html = '<div class="channel-legend"><span class="stats-label">Channels:</span>'
        if 'red' in channel_legend:
            channel_legend_html += f'<span class="ch-red">R={channel_legend["red"]}</span>'
        if 'green' in channel_legend:
            channel_legend_html += f'<span class="ch-green">G={channel_legend["green"]}</span>'
        if 'blue' in channel_legend:
            channel_legend_html += f'<span class="ch-blue">B={channel_legend["blue"]}</span>'
        channel_legend_html += '</div>'

    # Build subtitle HTML if provided
    subtitle_html = ''
    if subtitle:
        subtitle_html = f'<div class="header-subtitle">{subtitle}</div>'

    # Build cards
    cards_html = ''
    for sample in samples:
        uid = sample['uid']
        img_b64 = sample['image']
        mime = sample.get('mime_type', 'jpeg')
        stats = sample.get('stats', {})

        # Format stats line
        stats_parts = []
        if 'area_um2' in stats:
            stats_parts.append(f"{stats['area_um2']:.1f} &micro;m&sup2;")
        if 'area_px' in stats:
            stats_parts.append(f"{stats['area_px']:.0f} px")
        if 'solidity' in stats:
            stats_parts.append(f"sol: {stats['solidity']:.2f}")
        # Only show confidence if it's not 1.0 (i.e., after classifier training)
        if 'confidence' in stats and stats['confidence'] < 0.999:
            stats_parts.append(f"{stats['confidence']*100:.0f}%")

        stats_str = ' | '.join(stats_parts) if stats_parts else ''

        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1">
            <div class="card-img-container">
                <img src="data:image/{mime};base64,{img_b64}" alt="{uid}">
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
'''

    html = f'''<!DOCTYPE html>
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
            <button class="btn" onclick="clearPage()">Clear Page</button>
            <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
            {channel_legend_html}
        </div>
    </div>

    <div class="content">
        <div class="grid">{cards_html}</div>
    </div>

    <div class="keyboard-hint">
        Keyboard: Y=Yes, N=No, U=Unsure, Arrow keys=Navigate
    </div>

    <div class="footer">
        {nav_html}
    </div>

    <script>{get_js(cell_type, total_pages, experiment_name)}</script>
</body>
</html>'''

    return html


def generate_index_page(
    cell_type,
    total_samples,
    total_pages,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix='page',
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

    # Build info lines
    info_lines = []
    info_lines.append(f"Detection type: {cell_type.upper()}")
    if file_name:
        info_lines.append(f"File: {file_name}")
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
        info_lines.append(f"Segmentation: {timestamp}")

    info_html = '<br>'.join(info_lines)

    extra_stats_html = ''
    if extra_stats:
        for label, value in extra_stats.items():
            extra_stats_html += f'''
            <div class="stat">
                <span>{label}</span>
                <span class="number">{value}</span>
            </div>'''

    html = f'''<!DOCTYPE html>
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
        <button class="btn btn-danger" onclick="clearAll()">Clear All</button>
    </div>

    <script>
        const CELL_TYPE = '{cell_type}';
        const EXPERIMENT_NAME = '{experiment_name or ""}';
        const STORAGE_KEY = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations' : CELL_TYPE + '_annotations';

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

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            localStorage.setItem(STORAGE_KEY, JSON.stringify({{}}));
            alert('All annotations cleared. Refresh any open pages to see the change.');
        }}
    </script>
</body>
</html>'''

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

    # Build info lines
    info_lines = []
    if file_name:
        info_lines.append(f"Source: {file_name}")
    if pixel_size_um:
        info_lines.append(f"Pixel size: {pixel_size_um:.4f} &micro;m/px")
    if tiles_processed is not None:
        denominator = tissue_tiles if tissue_tiles else tiles_total
        if denominator:
            pct = 100.0 * tiles_processed / denominator
            label = "Tissue tiles processed" if tissue_tiles else "Tiles processed"
            info_lines.append(f"{label}: {tiles_processed:,} / {denominator:,} ({pct:.1f}%)")
    if timestamp:
        info_lines.append(f"Segmentation: {timestamp}")

    info_html = '<br>'.join(info_lines) if info_lines else ''
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ''

    # Build cell type sections
    sections_html = ''
    for ct, info in cell_types.items():
        total_samples = info.get('total_samples', 0)
        total_pages = info.get('total_pages', 0)
        page_prefix = info.get('page_prefix', f'{ct}_page')

        sections_html += f'''
        <div class="cell-type-section">
            <h2>{ct.upper()}</h2>
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
            <a href="{page_prefix}_1.html" class="btn">Review {ct.upper()}</a>
        </div>
        '''

    # Build export buttons for each cell type
    export_buttons = ''
    for ct in cell_types.keys():
        export_buttons += f'''
        <button class="btn btn-export" onclick="exportAnnotations('{ct}')">Export {ct.upper()}</button>
        '''

    html = f'''<!DOCTYPE html>
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
        const EXPERIMENT_NAME = '{experiment_name or ""}';

        function getStorageKey(cellType) {{
            return EXPERIMENT_NAME ? cellType + '_' + EXPERIMENT_NAME + '_annotations' : cellType + '_annotations';
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
            const cellTypes = {list(cell_types.keys())};
            cellTypes.forEach(ct => {{
                localStorage.setItem(getStorageKey(ct), JSON.stringify({{}}));
            }});
            alert('All annotations cleared. Refresh any open pages.');
        }}
    </script>
</body>
</html>'''

    return html


def export_samples_to_html(
    samples,
    output_dir,
    cell_type,
    samples_per_page=300,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix='page',
    experiment_name=None,
    file_name=None,
    pixel_size_um=None,
    tiles_processed=None,
    tiles_total=None,
    tissue_tiles=None,
    channel_legend=None,
    timestamp=None,
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

    Returns:
        Tuple of (total_samples, total_pages)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        print(f"No {cell_type} samples to export")
        return 0, 0

    # Paginate
    pages = [samples[i:i+samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    print(f"Generating {total_pages} {cell_type} HTML pages...")

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
        )

        page_path = output_dir / f"{page_prefix}_{page_num}.html"
        with open(page_path, 'w') as f:
            f.write(html)

        file_size = page_path.stat().st_size / (1024*1024)
        print(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")

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

    index_path = output_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write(index_html)

    print(f"Export complete: {output_dir}")

    return len(samples), total_pages


# =============================================================================
# MK/HSPC BATCH HTML EXPORT (RAM-based)
# =============================================================================
# These functions support batch processing of MK and HSPC cell types,
# loading samples from slide images already in RAM for efficiency.


def load_samples_from_ram(tiles_dir, slide_image, pixel_size_um, cell_type='mk', max_samples=None, logger=None):
    """
    Load cell samples from segmentation output, using in-memory slide image.

    Args:
        tiles_dir: Path to tiles directory (e.g., output/mk/tiles)
        slide_image: numpy array of full slide image (already in RAM)
        pixel_size_um: Pixel size in microns
        cell_type: 'mk' or 'hspc' - affects mask selection
        max_samples: Maximum samples to load (None for all)
        logger: Optional logger instance for debug messages

    Returns:
        List of sample dicts with image data and metadata
    """
    import re
    import json
    import h5py

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
            if logger:
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

            try:
                cell_idx = int(det_id.split('_')[1]) + 1
            except Exception as e:
                if logger:
                    logger.debug(f"Failed to parse cell index from {det_id}: {e}")
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
            except Exception as e:
                if logger:
                    logger.debug(f"Failed to extract crop at ({global_x1}, {global_y1}): {e}")
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
                crop_with_contour = draw_mask_contour(crop_resized, mask_resized,
                                                       color=(0, 255, 0), dotted=False)
            else:
                crop_with_contour = crop_resized

            # Convert to base64 (JPEG for smaller file sizes)
            pil_img_final = Image.fromarray(crop_with_contour)
            buffer = BytesIO()
            pil_img_final.save(buffer, format='JPEG', quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Use global center from features.json if available, otherwise compute
            if 'center' in feat_dict:
                # center is already in global coordinates
                global_centroid_x = int(feat_dict['center'][0])
                global_centroid_y = int(feat_dict['center'][1])
            else:
                # Backwards compatibility: compute from tile origin + local centroid
                global_centroid_x = tile_x1 + centroid_x
                global_centroid_y = tile_y1 + centroid_y

            # Get global_id if available
            global_id = feat_dict.get('global_id', None)

            samples.append({
                'tile_id': tile_dir.name,
                'det_id': det_id,
                'global_id': global_id,
                'area_px': area_px,
                'area_um2': area_um2,
                'image': img_b64,
                'features': features,
                'solidity': features.get('solidity', 0),
                'circularity': features.get('circularity', 0),
                'global_x': global_centroid_x,
                'global_y': global_centroid_y
            })

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


def create_mk_hspc_index(output_dir, total_mks, total_hspcs, mk_pages, hspc_pages, slides_summary=None):
    """Create the main index.html page for MK+HSPC batch review."""
    subtitle_html = f'<p style="color: #888; margin-bottom: 10px;">{slides_summary}</p>' if slides_summary else ''
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
        {subtitle_html}
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


def generate_mk_hspc_page_html(samples, cell_type, page_num, total_pages, slides_summary=None):
    """Generate HTML for a single MK or HSPC cell type page.

    Args:
        samples: List of sample dicts with image data
        cell_type: 'mk' or 'hspc'
        page_num: Current page number
        total_pages: Total number of pages
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)" for subtitle
    """
    cell_type_display = "Megakaryocytes (MKs)" if cell_type == "mk" else "HSPCs"

    # Build subtitle HTML
    subtitle_html = ''
    if slides_summary:
        subtitle_html = f'<div class="header-subtitle">{slides_summary}</div>'

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
        global_x = sample.get('global_x', 0)
        global_y = sample.get('global_y', 0)
        # Always use spatial UID format for consistency across all cell types
        # Format: {slide}_{celltype}_{round(x)}_{round(y)}
        uid = f"{slide}_{cell_type}_{int(round(global_x))}_{int(round(global_y))}"
        display_id = f"{cell_type}_{int(round(global_x))}_{int(round(global_y))}"
        # Keep legacy global_id in data attribute for backwards compatibility
        legacy_global_id = sample.get('global_id')
        area_um2 = sample.get('area_um2', 0)
        area_px = sample.get('area_px', 0)
        img_b64 = sample['image']
        # Include legacy_global_id as data attribute for migration support
        legacy_attr = f' data-legacy-id="{legacy_global_id}"' if legacy_global_id is not None else ''
        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1"{legacy_attr}>
            <div class="card-img-container">
                <img src="data:image/jpeg;base64,{img_b64}" alt="{display_id}">
            </div>
            <div class="card-info">
                <div>
                    <div class="card-id">{display_id}</div>
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


def generate_mk_hspc_pages(samples, cell_type, output_dir, samples_per_page, slides_summary=None, logger=None):
    """Generate separate pages for a single cell type (MK or HSPC).

    Args:
        samples: List of sample dicts with image data
        cell_type: 'mk' or 'hspc'
        output_dir: Directory to write HTML files
        samples_per_page: Number of samples per page
        slides_summary: Optional string like "16 slides (FGC1, FGC2, ...)" for subtitle
        logger: Optional logger instance
    """
    if not samples:
        if logger:
            logger.info(f"  No {cell_type.upper()} samples to export")
        return

    pages = [samples[i:i+samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    if logger:
        logger.info(f"  Generating {total_pages} {cell_type.upper()} pages...")

    for page_num in range(1, total_pages + 1):
        page_samples = pages[page_num - 1]
        html = generate_mk_hspc_page_html(page_samples, cell_type, page_num, total_pages, slides_summary=slides_summary)

        html_path = Path(output_dir) / f"{cell_type}_page{page_num}.html"
        with open(html_path, 'w') as f:
            f.write(html)


def export_mk_hspc_html_from_ram(slide_data, output_base, html_output_dir, samples_per_page=300,
                                  mk_min_area_um=200, mk_max_area_um=2000, logger=None):
    """
    Export HTML pages using slide images already in RAM.

    Args:
        slide_data: dict of {slide_name: {'image': np.array, 'czi_path': path, ...}}
        output_base: Path to segmentation output directory
        html_output_dir: Path to write HTML files
        samples_per_page: Number of samples per HTML page
        mk_min_area_um: Min MK area filter
        mk_max_area_um: Max MK area filter
        logger: Optional logger instance
    """
    import json

    if logger:
        logger.info(f"\n{'='*70}")
        logger.info("EXPORTING HTML (using images in RAM)")
        logger.info(f"{'='*70}")

    html_output_dir = Path(html_output_dir)
    html_output_dir.mkdir(parents=True, exist_ok=True)

    all_mk_samples = []
    all_hspc_samples = []

    PIXEL_SIZE_UM = 0.1725  # Default pixel size

    for slide_name, data in slide_data.items():
        slide_dir = output_base / slide_name
        if not slide_dir.exists():
            continue

        if logger:
            logger.info(f"  Loading {slide_name}...")

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
            cell_type='mk',
            logger=logger
        )

        # Load HSPC samples (sorted by solidity/confidence)
        hspc_samples = load_samples_from_ram(
            slide_dir / "hspc" / "tiles",
            slide_image, pixel_size_um,
            cell_type='hspc',
            logger=logger
        )

        # Add slide name to each sample
        for s in mk_samples:
            s['slide'] = slide_name
        for s in hspc_samples:
            s['slide'] = slide_name

        all_mk_samples.extend(mk_samples)
        all_hspc_samples.extend(hspc_samples)

        if logger:
            logger.info(f"    {len(mk_samples)} MKs, {len(hspc_samples)} HSPCs")

    # Filter MK by size
    um_to_px_factor = PIXEL_SIZE_UM ** 2
    mk_min_px = int(mk_min_area_um / um_to_px_factor)
    mk_max_px = int(mk_max_area_um / um_to_px_factor)

    mk_before = len(all_mk_samples)
    all_mk_samples = [s for s in all_mk_samples if mk_min_px <= s.get('area_px', 0) <= mk_max_px]
    if logger:
        logger.info(f"  MK size filter: {mk_before} -> {len(all_mk_samples)}")

    # Sort by area
    all_mk_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)
    all_hspc_samples.sort(key=lambda x: x.get('area_um2', 0), reverse=True)

    # Build slides summary for subtitle (e.g., "16 slides (FGC1, FGC2, ...)")
    slide_names = sorted(slide_data.keys())
    num_slides = len(slide_names)
    if num_slides > 0:
        # Extract short identifiers (e.g., "FGC1" from "2025_11_18_FGC1")
        short_names = []
        for name in slide_names:
            parts = name.split('_')
            # Take the last part that looks like a group identifier
            short = parts[-1] if parts else name
            short_names.append(short)
        # Show first few names with ellipsis if many
        if len(short_names) > 6:
            preview = ', '.join(short_names[:4]) + ', ...'
        else:
            preview = ', '.join(short_names)
        slides_summary = f"{num_slides} slides ({preview})"
    else:
        slides_summary = None

    # Generate pages
    generate_mk_hspc_pages(all_mk_samples, "mk", html_output_dir, samples_per_page, slides_summary=slides_summary, logger=logger)
    generate_mk_hspc_pages(all_hspc_samples, "hspc", html_output_dir, samples_per_page, slides_summary=slides_summary, logger=logger)

    # Create index
    mk_pages = (len(all_mk_samples) + samples_per_page - 1) // samples_per_page if all_mk_samples else 0
    hspc_pages = (len(all_hspc_samples) + samples_per_page - 1) // samples_per_page if all_hspc_samples else 0
    create_mk_hspc_index(html_output_dir, len(all_mk_samples), len(all_hspc_samples), mk_pages, hspc_pages, slides_summary=slides_summary)

    if logger:
        logger.info(f"\n  HTML export complete: {html_output_dir}")
        logger.info(f"  Total: {len(all_mk_samples)} MKs, {len(all_hspc_samples)} HSPCs")


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
    return '''
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
            max-width: 100%;
            max-height: 100%;
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
    '''


def get_vessel_js(cell_type, total_pages, experiment_name=None, all_features_json='{}'):
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
    # Build storage key with optional experiment name
    if experiment_name:
        storage_key = f"{cell_type}_{experiment_name}_annotations"
    else:
        storage_key = f"{cell_type}_annotations"

    return f'''
        const CELL_TYPE = '{cell_type}';
        const EXPERIMENT_NAME = '{experiment_name or ""}';
        const TOTAL_PAGES = {total_pages};
        const STORAGE_KEY = '{storage_key}';
        const ALL_FEATURES = {all_features_json};

        let labels = {{}};
        let selectedIdx = -1;
        let batchSelected = new Set();
        const cards = document.querySelectorAll('.card');

        // Load from localStorage
        function loadAnnotations() {{
            try {{
                const saved = localStorage.getItem(STORAGE_KEY);
                if (saved) labels = JSON.parse(saved);
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

            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
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

            // Count global
            for (const v of Object.values(labels)) {{
                if (v === 1) globalYes++;
                else if (v === 0) globalNo++;
                globalTotal++;
            }}

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
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
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
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
            updateStats();
        }}

        function clearAll() {{
            if (!confirm('Clear ALL annotations across ALL pages? This cannot be undone.')) return;
            labels = {{}};
            localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
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
    '''


def generate_vessel_annotation_page(
    samples,
    cell_type,
    page_num,
    total_pages,
    title=None,
    page_prefix='page',
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

    # Build navigation
    nav_html = '<div class="nav-buttons">'
    nav_html += '<a href="index.html" class="nav-btn">Home</a>'
    if page_num > 1:
        nav_html += f'<a href="{page_prefix}_{page_num-1}.html" class="nav-btn">Prev</a>'
    nav_html += f'<span class="page-info">Page {page_num} / {total_pages}</span>'
    if page_num < total_pages:
        nav_html += f'<a href="{page_prefix}_{page_num+1}.html" class="nav-btn">Next</a>'
    nav_html += '</div>'

    # Build subtitle HTML if provided
    subtitle_html = ''
    if subtitle:
        subtitle_html = f'<div class="header-subtitle">{subtitle}</div>'

    # Collect all features for this page (for filtering/export)
    all_features = {}
    for sample in samples:
        uid = sample['uid']
        feat = sample.get('features', {})
        # Filter to numeric features only
        all_features[uid] = {k: v for k, v in feat.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)}

    all_features_json = json.dumps(all_features)

    # Build cards
    cards_html = ''
    for idx, sample in enumerate(samples):
        uid = sample['uid']
        img_b64 = sample['image']
        mime = sample.get('mime_type', 'jpeg')
        feat = sample.get('features', {})

        # Format stats line with vessel-specific features
        stats_parts = []
        if 'outer_diameter_um' in feat:
            stats_parts.append(f"D={feat['outer_diameter_um']:.1f}&micro;m")
        if 'wall_thickness_mean_um' in feat:
            stats_parts.append(f"wall={feat['wall_thickness_mean_um']:.1f}&micro;m")
        if 'circularity' in feat:
            stats_parts.append(f"circ={feat['circularity']:.2f}")
        if 'confidence' in feat:
            conf = feat['confidence']
            if isinstance(conf, str):
                stats_parts.append(conf)
            else:
                stats_parts.append(f"{conf*100:.0f}%")

        stats_str = ' | '.join(stats_parts) if stats_parts else ''

        # Additional feature line
        feat_parts = []
        if 'aspect_ratio' in feat:
            feat_parts.append(f"AR={feat['aspect_ratio']:.2f}")
        if 'ring_completeness' in feat:
            feat_parts.append(f"ring={feat['ring_completeness']:.2f}")
        if 'lumen_area_um2' in feat:
            feat_parts.append(f"lumen={feat['lumen_area_um2']:.0f}&micro;m&sup2;")
        feat_str = ' | '.join(feat_parts) if feat_parts else ''

        cards_html += f'''
        <div class="card" id="{uid}" data-label="-1"
             data-diameter="{feat.get('outer_diameter_um', 0)}"
             data-confidence="{feat.get('confidence', 'unknown')}">
            <input type="checkbox" class="card-checkbox" onclick="event.stopPropagation(); toggleBatchSelect('{uid}')">
            <div class="card-img-container" onclick="selectCard({idx})">
                <img src="data:image/{mime};base64,{img_b64}" alt="{uid}">
            </div>
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
'''

    html = f'''<!DOCTYPE html>
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

    <script>{get_vessel_js(cell_type, total_pages, experiment_name, all_features_json)}</script>
</body>
</html>'''

    return html


def generate_vessel_index_page(
    cell_type,
    total_samples,
    total_pages,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix='page',
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

    # Build info lines
    info_lines = []
    info_lines.append(f"Detection type: {cell_type.upper()}")
    if file_name:
        info_lines.append(f"File: {file_name}")
    if pixel_size_um:
        info_lines.append(f"Pixel size: {pixel_size_um:.4f} &micro;m/px")
    info_lines.append(f"Total detections: {total_samples:,}")
    info_lines.append(f"Pages: {total_pages}")
    if timestamp:
        info_lines.append(f"Segmentation: {timestamp}")

    info_html = '<br>'.join(info_lines)

    # Feature summary section
    feature_html = ''
    if feature_summary:
        feature_html = '<div class="feature-summary"><h3>Feature Summary</h3><table>'
        for key, stats in feature_summary.items():
            feature_html += f'''
            <tr>
                <td>{key}</td>
                <td>min: {stats.get("min", 0):.2f}</td>
                <td>max: {stats.get("max", 0):.2f}</td>
                <td>mean: {stats.get("mean", 0):.2f}</td>
            </tr>'''
        feature_html += '</table></div>'

    extra_stats_html = ''
    if extra_stats:
        for label, value in extra_stats.items():
            extra_stats_html += f'''
            <div class="stat">
                <span>{label}</span>
                <span class="number">{value}</span>
            </div>'''

    html = f'''<!DOCTYPE html>
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
        const CELL_TYPE = '{cell_type}';
        const EXPERIMENT_NAME = '{experiment_name or ""}';
        const TOTAL_SAMPLES = {total_samples};
        const STORAGE_KEY = EXPERIMENT_NAME ? CELL_TYPE + '_' + EXPERIMENT_NAME + '_annotations' : CELL_TYPE + '_annotations';

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
            updateProgress();
            alert('All annotations cleared. Refresh any open pages to see the change.');
        }}

        // Initialize
        updateProgress();
        // Auto-refresh progress every 5 seconds
        setInterval(updateProgress, 5000);
    </script>
</body>
</html>'''

    return html


def export_vessel_samples_to_html(
    samples,
    output_dir,
    cell_type='vessel',
    samples_per_page=200,
    title=None,
    subtitle=None,
    extra_stats=None,
    page_prefix='page',
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
        print(f"No {cell_type} samples to export")
        return 0, 0

    # Calculate feature summary for index page
    feature_summary = {}
    key_features = ['outer_diameter_um', 'wall_thickness_mean_um', 'circularity', 'aspect_ratio']
    for feat_key in key_features:
        values = [s['features'].get(feat_key) for s in samples if feat_key in s.get('features', {})]
        if values:
            values = [v for v in values if v is not None and isinstance(v, (int, float))]
            if values:
                feature_summary[feat_key] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                }

    # Paginate
    pages = [samples[i:i+samples_per_page] for i in range(0, len(samples), samples_per_page)]
    total_pages = len(pages)

    print(f"Generating {total_pages} {cell_type} HTML pages...")

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
        with open(page_path, 'w') as f:
            f.write(html)

        file_size = page_path.stat().st_size / (1024*1024)
        print(f"  Page {page_num}: {len(page_samples)} samples ({file_size:.1f} MB)")

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

    index_path = output_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write(index_html)

    print(f"Export complete: {output_dir}")

    return len(samples), total_pages
