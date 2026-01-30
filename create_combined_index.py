#!/usr/bin/env python3
"""
Create a combined index page for MK and HSPC HTML viewers using package template style.
"""

import json
from pathlib import Path


# Dark theme color palette (matching HTMLPageGenerator)
COLORS = {
    'bg_primary': '#0a0a0a',
    'bg_secondary': '#111',
    'bg_tertiary': '#1a1a1a',
    'border': '#333',
    'text_primary': '#ddd',
    'text_secondary': '#888',
    'text_tertiary': '#555',
    'positive': '#4a4',
    'negative': '#a44',
    'unsure': '#aa4',
    'export': '#44a',
}


def create_combined_index(output_dir, experiment_name="mi300a_10pct_all16_fixed"):
    """Create a combined index page linking MK and HSPC viewers using package template style."""
    output_dir = Path(output_dir)
    html_combined = output_dir / "html_combined"

    # Count detections from each viewer
    mk_dir = html_combined / "mk"
    hspc_dir = html_combined / "hspc"

    # Count samples by looking at directory structure
    mk_count = 0
    hspc_count = 0

    # Count MK samples
    for slide_dir in (output_dir).glob("2025_11_18_*/mk/tiles"):
        for tile_dir in slide_dir.iterdir():
            if tile_dir.is_dir():
                feat_file = tile_dir / "features.json"
                if feat_file.exists():
                    with open(feat_file) as f:
                        mk_count += len(json.load(f))

    # Count HSPC samples
    for slide_dir in (output_dir).glob("2025_11_18_*/hspc/tiles"):
        for tile_dir in slide_dir.iterdir():
            if tile_dir.is_dir():
                feat_file = tile_dir / "features.json"
                if feat_file.exists():
                    with open(feat_file) as f:
                        hspc_count += len(json.load(f))

    # Count pages
    mk_pages = len(list(mk_dir.glob("page_*.html"))) if mk_dir.exists() else 0
    hspc_pages = len(list(hspc_dir.glob("page_*.html"))) if hspc_dir.exists() else 0

    c = COLORS
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MK + HSPC Combined Viewer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: monospace;
            background: {c['bg_primary']};
            color: {c['text_primary']};
            padding: 20px;
        }}

        .header {{
            background: {c['bg_secondary']};
            padding: 30px;
            border: 1px solid {c['border']};
            margin-bottom: 20px;
            text-align: center;
        }}

        h1 {{
            font-size: 1.5em;
            font-weight: normal;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: {c['text_secondary']};
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
            background: {c['bg_tertiary']};
            border: 1px solid {c['border']};
            text-align: center;
        }}

        .stat .number {{
            display: block;
            font-size: 2em;
            margin-top: 10px;
            color: {c['positive']};
        }}

        .section {{
            margin: 30px 0;
            text-align: center;
        }}

        .viewer-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 800px;
            margin: 30px auto;
        }}

        .viewer-card {{
            background: {c['bg_secondary']};
            border: 1px solid {c['border']};
            padding: 20px;
            text-align: center;
        }}

        .viewer-card h2 {{
            font-size: 1.2em;
            font-weight: normal;
            margin-bottom: 15px;
        }}

        .viewer-card .viewer-stats {{
            margin: 15px 0;
            color: {c['text_secondary']};
        }}

        .btn {{
            padding: 15px 40px;
            background: {c['bg_tertiary']};
            border: 1px solid {c['positive']};
            color: {c['positive']};
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
            border-color: {c['export']};
            color: {c['export']};
        }}

        .btn-export:hover {{
            background: #0f0f13;
        }}

        .btn-danger {{
            border-color: {c['negative']};
            color: {c['negative']};
        }}

        .btn-danger:hover {{
            background: #130f0f;
        }}

        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid {c['border']};
            color: {c['text_tertiary']};
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MK + HSPC Combined Viewer</h1>
        <p class="subtitle">All 16 Slides • 10% Sampling • Cleaned Masks • AMD MI300A</p>
        <div class="stats">
            <div class="stat">
                <span>Total Detections</span>
                <span class="number">{mk_count + hspc_count:,}</span>
            </div>
            <div class="stat">
                <span>Megakaryocytes</span>
                <span class="number">{mk_count:,}</span>
            </div>
            <div class="stat">
                <span>HSPCs</span>
                <span class="number">{hspc_count:,}</span>
            </div>
        </div>
    </div>

    <div class="viewer-grid">
        <div class="viewer-card">
            <h2>Megakaryocytes</h2>
            <div class="viewer-stats">
                {mk_count:,} cells • {mk_pages} pages<br>
                SAM2 detection on RGB
            </div>
            <a href="mk/page_1.html" class="btn">View MK Viewer</a>
        </div>

        <div class="viewer-card">
            <h2>HSPCs</h2>
            <div class="viewer-stats">
                {hspc_count:,} cells • {hspc_pages} pages<br>
                Cellpose + SAM2 on H channel
            </div>
            <a href="hspc/page_1.html" class="btn">View HSPC Viewer</a>
        </div>
    </div>

    <div class="section">
        <button class="btn btn-export" onclick="exportAllAnnotations()">Export All Annotations</button>
        <button class="btn btn-danger" onclick="clearAllAnnotations()">Clear All Annotations</button>
    </div>

    <div class="footer">
        xldvp_seg pipeline • PyTorch 2.5.1+rocm6.2 • AMD MI300A (ROCm 6.3)
    </div>

    <script>
        const EXPERIMENT_NAME = '{experiment_name}';
        const MK_STORAGE_KEY = 'mk_' + EXPERIMENT_NAME + '_annotations';
        const HSPC_STORAGE_KEY = 'hspc_' + EXPERIMENT_NAME + '_annotations';

        function clearAllAnnotations() {{
            if (!confirm('Clear ALL annotations for both MK and HSPC? This cannot be undone.')) return;

            localStorage.removeItem(MK_STORAGE_KEY);
            localStorage.removeItem(HSPC_STORAGE_KEY);

            alert('✓ All annotations cleared for both MK and HSPC viewers');
        }}

        function exportAllAnnotations() {{
            const mkLabels = JSON.parse(localStorage.getItem(MK_STORAGE_KEY) || '{{}}');
            const hspcLabels = JSON.parse(localStorage.getItem(HSPC_STORAGE_KEY) || '{{}}');

            const data = {{
                experiment_name: EXPERIMENT_NAME,
                exported_at: new Date().toISOString(),
                mk: {{
                    positive: [],
                    negative: [],
                    unsure: []
                }},
                hspc: {{
                    positive: [],
                    negative: [],
                    unsure: []
                }}
            }};

            // Process MK annotations
            for (const [uid, val] of Object.entries(mkLabels)) {{
                if (val === 1) data.mk.positive.push(uid);
                else if (val === 0) data.mk.negative.push(uid);
                else if (val === 2) data.mk.unsure.push(uid);
            }}

            // Process HSPC annotations
            for (const [uid, val] of Object.entries(hspcLabels)) {{
                if (val === 1) data.hspc.positive.push(uid);
                else if (val === 0) data.hspc.negative.push(uid);
                else if (val === 2) data.hspc.unsure.push(uid);
            }}

            const totalAnnotations =
                data.mk.positive.length + data.mk.negative.length + data.mk.unsure.length +
                data.hspc.positive.length + data.hspc.negative.length + data.hspc.unsure.length;

            if (totalAnnotations === 0) {{
                alert('No annotations to export');
                return;
            }}

            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = EXPERIMENT_NAME + '_all_annotations.json';
            a.click();
            URL.revokeObjectURL(url);

            alert(`✓ Exported ${{totalAnnotations}} annotations\\n` +
                  `MK: ${{data.mk.positive.length}} yes, ${{data.mk.negative.length}} no, ${{data.mk.unsure.length}} unsure\\n` +
                  `HSPC: ${{data.hspc.positive.length}} yes, ${{data.hspc.negative.length}} no, ${{data.hspc.unsure.length}} unsure`);
        }}
    </script>
</body>
</html>'''

    # Write combined index
    index_path = html_combined / "index.html"
    with open(index_path, 'w') as f:
        f.write(html)

    print(f"✓ Created combined index at: {index_path}")
    return index_path


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/viper/ptmp2/edrod/unified_10pct_mi300a"
    create_combined_index(output_dir)
