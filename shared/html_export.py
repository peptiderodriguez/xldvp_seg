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


def percentile_normalize(image, p_low=5, p_high=95):
    """
    Normalize image using percentiles.

    Args:
        image: 2D or 3D numpy array
        p_low: Lower percentile for normalization
        p_high: Upper percentile for normalization

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
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(image.shape[2]):
            result[:, :, c] = percentile_normalize(image[:, :, c], p_low, p_high)
        return result


def draw_mask_contour(img_array, mask, color=(0, 255, 0), thickness=2, dotted=False):
    """
    Draw mask contour on image.

    Args:
        img_array: RGB image array
        mask: Binary mask
        color: RGB tuple for contour color
        thickness: Contour thickness in pixels
        dotted: Whether to use dotted line

    Returns:
        Image with contour drawn
    """
    dilated = ndimage.binary_dilation(mask, iterations=thickness)
    contour = dilated & ~mask
    ys, xs = np.where(contour)

    if len(ys) == 0:
        return img_array

    img_out = img_array.copy()

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


def get_js(cell_type, total_pages):
    """
    Get the unified JavaScript for annotation handling.

    Args:
        cell_type: Type identifier (e.g., 'nmj', 'mk', 'hspc')
        total_pages: Total number of pages

    Returns:
        JavaScript code string
    """
    return f'''
        const CELL_TYPE = '{cell_type}';
        const TOTAL_PAGES = {total_pages};
        const STORAGE_KEY = CELL_TYPE + '_annotations';

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
        if 'confidence' in stats:
            stats_parts.append(f"{stats['confidence']*100:.0f}%")
        if 'elongation' in stats:
            stats_parts.append(f"elong: {stats['elongation']:.2f}")

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
    <title>{title} - Page {page_num}/{total_pages}</title>
    <style>{get_css()}</style>
</head>
<body>
    <div class="header">
        <div class="header-top">
            <h1>{title} - Page {page_num}/{total_pages}</h1>
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

    <script>{get_js(cell_type, total_pages)}</script>
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

    Returns:
        HTML string
    """
    if title is None:
        title = f"{cell_type.upper()} Annotation Review"
    if subtitle is None:
        subtitle = "Cell Detection Pipeline"

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
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p class="subtitle">{subtitle}</p>
        <div class="stats">
            <div class="stat">
                <span>Total Samples</span>
                <span class="number">{total_samples:,}</span>
            </div>
            <div class="stat">
                <span>Pages</span>
                <span class="number">{total_pages}</span>
            </div>
            {extra_stats_html}
        </div>
    </div>

    <div class="section">
        <a href="{page_prefix}_1.html" class="btn">Start Review</a>
    </div>

    <div class="section">
        <button class="btn btn-export" onclick="exportAnnotations()">Export Annotations</button>
    </div>

    <script>
        const STORAGE_KEY = '{cell_type}_annotations';

        function exportAnnotations() {{
            const stored = localStorage.getItem(STORAGE_KEY);
            const labels = stored ? JSON.parse(stored) : {{}};

            const data = {{
                cell_type: '{cell_type}',
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
            a.download = '{cell_type}_annotations.json';
            a.click();
            URL.revokeObjectURL(url);
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
    )

    index_path = output_dir / 'index.html'
    with open(index_path, 'w') as f:
        f.write(index_html)

    print(f"Export complete: {output_dir}")

    return len(samples), total_pages
