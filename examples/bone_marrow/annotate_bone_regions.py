#!/usr/bin/env python
"""Generate HTML tool for annotating bone regions (femur/humerus) on slides.

Creates a self-contained HTML file where users can draw polygon regions
around each bone on each slide. Regions are saved to localStorage and
can be exported as JSON.

Usage:
    python scripts/annotate_bone_regions.py \
        --czi-dir /path/to/czi/files \
        --output /path/to/bone_annotation.html

    # Or with OME-Zarr files:
    python scripts/annotate_bone_regions.py \
        --zarr-dir /path/to/zarr/files \
        --output /path/to/bone_annotation.html
"""
import argparse
import base64
import gc
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def read_czi_thumbnail(czi_path, scale_factor=0.05, display_channel=0):
    """Read CZI mosaic at low resolution.

    Args:
        czi_path: Path to CZI file
        scale_factor: Downsampling factor (e.g. 0.05 = 5%)
        display_channel: Channel to display (default 0, or -1 for all/RGB)

    Returns:
        rgb: uint8 RGB array (H, W, 3)
        full_width: Full resolution width
        full_height: Full resolution height
        pixel_size_um: Pixel size in micrometers
    """
    from aicspylibczi import CziFile

    czi = CziFile(str(czi_path))

    # Get pixel size
    pixel_size_um = None
    try:
        scaling = czi.get_scaling()
        if scaling and len(scaling) >= 2:
            pixel_size_um = scaling[0] * 1e6  # m -> um
    except Exception:
        pass

    # Get scene bounding box
    bbox = czi.get_mosaic_scene_bounding_box(index=0)
    region = (bbox.x, bbox.y, bbox.w, bbox.h)
    full_width, full_height = bbox.w, bbox.h

    print(f"  Reading {czi_path.name}: {bbox.w}x{bbox.h} px at scale {scale_factor}...", flush=True)

    # Check if this is a brightfield (RGB) image
    dims = czi.dims
    is_rgb = 'A' in dims  # 'A' dimension indicates RGB/BGR

    if is_rgb:
        # Brightfield RGB - need to specify C even if there's only one "channel"
        # The 'A' dimension gives us the RGB values
        img = czi.read_mosaic(C=0, region=region, scale_factor=scale_factor)
        img = np.squeeze(img)
        print(f"    Brightfield shape: {img.shape}, dtype: {img.dtype}", flush=True)

        # Handle various shapes
        if img.ndim == 3:
            if img.shape[-1] == 3:
                # Already (H, W, 3)
                rgb = img
            elif img.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                rgb = np.transpose(img, (1, 2, 0))
            else:
                # Take first channel and make grayscale
                rgb = np.stack([img[0]] * 3, axis=-1)
        else:
            # 2D grayscale
            rgb = np.stack([img] * 3, axis=-1)

        # Normalize to uint8 if needed
        if rgb.dtype != np.uint8:
            if rgb.max() > 255:
                rgb = (rgb / rgb.max() * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
    else:
        # Fluorescence - read single channel
        img = czi.read_mosaic(C=display_channel, region=region, scale_factor=scale_factor)
        img = np.squeeze(img)
        print(f"    Fluorescence shape: {img.shape}, dtype: {img.dtype}", flush=True)

        # Percentile normalize to uint8 grayscale -> RGB
        valid = img[img > 0]
        if len(valid) > 0:
            p_low, p_high = np.percentile(valid, [1, 99.5])
            if p_high <= p_low:
                p_high = p_low + 1
            normalized = np.clip((img.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)
            gray = (normalized * 255).astype(np.uint8)
            gray[img == 0] = 0
        else:
            gray = np.zeros(img.shape, dtype=np.uint8)

        # Convert grayscale to RGB
        rgb = np.stack([gray, gray, gray], axis=-1)

    del img
    gc.collect()

    return rgb, full_width, full_height, pixel_size_um


def read_zarr_thumbnail(zarr_path, level=2):
    """Read OME-Zarr at a pyramid level.

    Args:
        zarr_path: Path to OME-Zarr directory
        level: Pyramid level to read (higher = more downsampled)

    Returns:
        rgb: uint8 RGB array (H, W, 3)
        full_width: Full resolution width
        full_height: Full resolution height
        pixel_size_um: Pixel size in micrometers
    """
    import zarr

    root = zarr.open(str(zarr_path), mode='r')

    # Get metadata
    pixel_size_um = None
    full_width, full_height = None, None

    if '.zattrs' in root.attrs or 'multiscales' in root.attrs:
        multiscales = root.attrs.get('multiscales', [{}])[0]
        datasets = multiscales.get('datasets', [])
        if datasets:
            # Full resolution from level 0
            level0 = root[datasets[0]['path']]
            if len(level0.shape) >= 2:
                full_height, full_width = level0.shape[-2], level0.shape[-1]
            elif len(level0.shape) >= 3:
                full_height, full_width = level0.shape[-2], level0.shape[-1]

        # Pixel size from coordinate transforms
        transforms = multiscales.get('coordinateTransformations', [])
        for t in transforms:
            if t.get('type') == 'scale':
                scale = t.get('scale', [])
                if len(scale) >= 2:
                    pixel_size_um = scale[-1]  # Assume um

    # Read requested level
    available_levels = [k for k in root.keys() if k.isdigit()]
    if str(level) in available_levels:
        data = np.array(root[str(level)])
    elif available_levels:
        # Use highest available level
        data = np.array(root[max(available_levels, key=int)])
    else:
        raise ValueError(f"No pyramid levels found in {zarr_path}")

    print(f"  Reading {zarr_path.name}: level {level}, shape {data.shape}", flush=True)

    # Handle different shapes: (C, H, W), (H, W), (H, W, C)
    if data.ndim == 3 and data.shape[0] <= 4:
        # (C, H, W) -> use first channel
        data = data[0]
    elif data.ndim == 3 and data.shape[-1] <= 4:
        # (H, W, C) -> use first channel
        data = data[:, :, 0]

    # Normalize to uint8
    valid = data[data > 0]
    if len(valid) > 0:
        p_low, p_high = np.percentile(valid, [1, 99.5])
        if p_high <= p_low:
            p_high = p_low + 1
        normalized = np.clip((data.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)
        gray = (normalized * 255).astype(np.uint8)
        gray[data == 0] = 0
    else:
        gray = np.zeros(data.shape, dtype=np.uint8)

    # Convert to RGB
    rgb = np.stack([gray, gray, gray], axis=-1)

    return rgb, full_width, full_height, pixel_size_um


def image_to_base64(img_array, format='JPEG', quality=85):
    """Convert numpy array to base64 string."""
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format=format, quality=quality)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def generate_html(slides_data, output_path):
    """Generate the bone annotation HTML file.

    Args:
        slides_data: List of dicts with keys:
            - name: slide name
            - image_b64: base64-encoded thumbnail
            - thumb_width: thumbnail width
            - thumb_height: thumbnail height
            - full_width: full resolution width
            - full_height: full resolution height
            - scale_factor: thumbnail scale factor
            - pixel_size_um: pixel size in micrometers
        output_path: Path to write HTML file
    """
    # Build slides JSON for embedding
    slides_json = json.dumps([{
        'name': s['name'],
        'thumbWidth': s['thumb_width'],
        'thumbHeight': s['thumb_height'],
        'fullWidth': s['full_width'],
        'fullHeight': s['full_height'],
        'scaleFactor': s['scale_factor'],
        'pixelSizeUm': s['pixel_size_um'],
    } for s in slides_data])

    # Build image data section
    image_tags = []
    for s in slides_data:
        image_tags.append(
            f'<img id="img-{s["name"]}" src="data:image/jpeg;base64,{s["image_b64"]}" style="display:none;">'
        )
    images_html = '\n'.join(image_tags)

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Bone Region Annotation</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e;
    color: #eee;
    min-height: 100vh;
}}
.header {{
    background: #16213e;
    padding: 15px 20px;
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
}}
.header h1 {{ font-size: 1.4em; font-weight: 500; }}
.btn {{
    background: #0f3460;
    color: #fff;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
}}
.btn:hover {{ background: #1a4a7a; }}
.btn.active {{ background: #e94560; }}
.btn-export {{ background: #2ecc71; }}
.btn-export:hover {{ background: #27ae60; }}
.btn-clear {{ background: #e74c3c; }}
.btn-clear:hover {{ background: #c0392b; }}
.status {{ margin-left: auto; font-size: 13px; color: #888; }}
.slides-container {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 20px;
    padding: 20px;
}}
.slide-card {{
    background: #16213e;
    border-radius: 8px;
    overflow: hidden;
}}
.slide-header {{
    padding: 10px 15px;
    background: #0f3460;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.slide-title {{ font-weight: 500; flex: 1; }}
.region-btns {{ display: flex; gap: 5px; }}
.region-btn {{
    padding: 4px 10px;
    border-radius: 3px;
    font-size: 12px;
    cursor: pointer;
    border: 2px solid transparent;
}}
.region-btn.femur {{ background: rgba(52, 152, 219, 0.3); color: #3498db; border-color: #3498db; }}
.region-btn.humerus {{ background: rgba(231, 76, 60, 0.3); color: #e74c3c; border-color: #e74c3c; }}
.region-btn.active {{ background: rgba(255, 255, 255, 0.2); }}
.region-btn.done {{ opacity: 0.5; }}
.canvas-container {{
    position: relative;
    cursor: crosshair;
}}
canvas {{
    display: block;
    width: 100%;
    height: auto;
}}
.instructions {{
    padding: 10px 15px;
    font-size: 12px;
    color: #888;
    border-top: 1px solid #0f3460;
}}
.legend {{
    display: flex;
    gap: 15px;
    padding: 10px 20px;
    background: #16213e;
    font-size: 13px;
}}
.legend-item {{ display: flex; align-items: center; gap: 5px; }}
.legend-color {{
    width: 16px;
    height: 16px;
    border-radius: 3px;
}}
.legend-color.femur {{ background: rgba(52, 152, 219, 0.7); border: 2px solid #3498db; }}
.legend-color.humerus {{ background: rgba(231, 76, 60, 0.7); border: 2px solid #e74c3c; }}
</style>
</head>
<body>

<div class="header">
    <h1>Bone Region Annotation</h1>
    <button class="btn btn-export" onclick="exportRegions()">Export JSON</button>
    <button class="btn" onclick="loadFromFile()">Import JSON</button>
    <input type="file" id="import-file" accept=".json" style="display:none" onchange="handleImport(event)">
    <div class="status" id="status">Click a region button, then click to draw polygon vertices. Double-click to close.</div>
</div>

<div class="legend">
    <div class="legend-item"><div class="legend-color femur"></div> Femur</div>
    <div class="legend-item"><div class="legend-color humerus"></div> Humerus</div>
    <span style="margin-left: auto; color: #888;">Annotated: <span id="progress">0</span> / {len(slides_data) * 2} regions</span>
</div>

<div class="slides-container" id="slides-container"></div>

<!-- Hidden images -->
<div style="display:none;">
{images_html}
</div>

<script>
const SLIDES = {slides_json};
const STORAGE_KEY = 'bone_regions_v1';

// State
let regions = {{}};  // {{slideName: {{femur: [[x,y],...], humerus: [[x,y],...] }}}}
let activeSlide = null;
let activeRegion = null;  // 'femur' or 'humerus'
let currentVerts = [];

// Initialize - wait for all images to load before rendering
document.addEventListener('DOMContentLoaded', () => {{
    loadFromStorage();

    // Wait for all hidden images to load
    const images = document.querySelectorAll('img[id^="img-"]');
    let loadedCount = 0;
    const totalImages = images.length;

    function checkAllLoaded() {{
        loadedCount++;
        document.getElementById('status').textContent = `Loading images: ${{loadedCount}}/${{totalImages}}...`;
        if (loadedCount >= totalImages) {{
            renderSlides();
            updateProgress();
            document.getElementById('status').textContent = 'Click a region button, then click to draw polygon vertices. Double-click to close.';
        }}
    }}

    images.forEach(img => {{
        if (img.complete && img.naturalWidth > 0) {{
            checkAllLoaded();
        }} else {{
            img.onload = checkAllLoaded;
            img.onerror = () => {{
                console.error('Failed to load image:', img.id);
                checkAllLoaded();
            }};
        }}
    }});

    // Fallback if no images
    if (totalImages === 0) {{
        renderSlides();
        updateProgress();
    }}
}});

function loadFromStorage() {{
    try {{
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {{
            regions = JSON.parse(saved);
        }}
    }} catch (e) {{
        console.error('Failed to load from storage:', e);
    }}
}}

function saveToStorage() {{
    try {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify(regions));
    }} catch (e) {{
        console.error('Failed to save to storage:', e);
    }}
}}

function renderSlides() {{
    const container = document.getElementById('slides-container');
    container.innerHTML = '';

    SLIDES.forEach((slide, idx) => {{
        const card = document.createElement('div');
        card.className = 'slide-card';
        card.innerHTML = `
            <div class="slide-header">
                <span class="slide-title">${{slide.name}}</span>
                <div class="region-btns">
                    <button class="region-btn femur" data-slide="${{slide.name}}" data-region="femur"
                        onclick="setActiveRegion('${{slide.name}}', 'femur', this)">Femur</button>
                    <button class="region-btn humerus" data-slide="${{slide.name}}" data-region="humerus"
                        onclick="setActiveRegion('${{slide.name}}', 'humerus', this)">Humerus</button>
                    <button class="btn btn-clear" style="font-size:11px;padding:4px 8px;"
                        onclick="clearSlideRegions('${{slide.name}}')">Clear</button>
                </div>
            </div>
            <div class="canvas-container">
                <canvas id="canvas-${{slide.name}}" data-slide="${{slide.name}}"></canvas>
            </div>
            <div class="instructions">
                Select region type → click vertices → double-click to close polygon
            </div>
        `;
        container.appendChild(card);

        // Set up canvas
        const canvas = card.querySelector('canvas');
        const img = document.getElementById('img-' + slide.name);
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        // Draw initial state
        drawCanvas(slide.name);

        // Event listeners
        canvas.addEventListener('click', (e) => handleClick(e, slide.name));
        canvas.addEventListener('dblclick', (e) => handleDoubleClick(e, slide.name));
        canvas.addEventListener('mousemove', (e) => handleMouseMove(e, slide.name));
    }});

    updateButtonStates();
}}

function setActiveRegion(slideName, regionType, btn) {{
    // Clear previous selection
    document.querySelectorAll('.region-btn').forEach(b => b.classList.remove('active'));

    // Set new active
    activeSlide = slideName;
    activeRegion = regionType;
    currentVerts = [];
    btn.classList.add('active');

    document.getElementById('status').textContent =
        `Drawing ${{regionType}} for ${{slideName}}. Click to add vertices, double-click to close.`;

    // Redraw all canvases to show selection
    SLIDES.forEach(s => drawCanvas(s.name));
}}

function handleClick(e, slideName) {{
    if (activeSlide !== slideName || !activeRegion) return;

    const canvas = e.target;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    currentVerts.push([x, y]);
    drawCanvas(slideName);
}}

function handleDoubleClick(e, slideName) {{
    if (activeSlide !== slideName || !activeRegion) return;
    if (currentVerts.length < 3) {{
        alert('Need at least 3 vertices to form a polygon');
        return;
    }}

    // Save region
    if (!regions[slideName]) regions[slideName] = {{}};
    regions[slideName][activeRegion] = [...currentVerts];

    // Reset state
    currentVerts = [];
    activeSlide = null;
    const prevRegion = activeRegion;
    activeRegion = null;

    document.querySelectorAll('.region-btn').forEach(b => b.classList.remove('active'));

    saveToStorage();
    updateProgress();
    updateButtonStates();
    drawCanvas(slideName);

    document.getElementById('status').textContent =
        `Saved ${{prevRegion}} region for ${{slideName}}. Select another region to continue.`;
}}

function handleMouseMove(e, slideName) {{
    if (activeSlide !== slideName || !activeRegion || currentVerts.length === 0) return;
    drawCanvas(slideName, e);
}}

function drawCanvas(slideName, mouseEvent = null) {{
    const canvas = document.getElementById('canvas-' + slideName);
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('img-' + slideName);

    // Draw image
    ctx.drawImage(img, 0, 0);

    // Draw saved regions
    const slideRegions = regions[slideName] || {{}};

    if (slideRegions.femur) {{
        drawPolygon(ctx, slideRegions.femur, 'rgba(52, 152, 219, 0.3)', '#3498db', true);
    }}
    if (slideRegions.humerus) {{
        drawPolygon(ctx, slideRegions.humerus, 'rgba(231, 76, 60, 0.3)', '#e74c3c', true);
    }}

    // Draw current in-progress polygon
    if (activeSlide === slideName && activeRegion && currentVerts.length > 0) {{
        const color = activeRegion === 'femur' ? '#3498db' : '#e74c3c';
        const fill = activeRegion === 'femur' ? 'rgba(52, 152, 219, 0.2)' : 'rgba(231, 76, 60, 0.2)';

        // Draw vertices and lines
        ctx.beginPath();
        ctx.moveTo(currentVerts[0][0], currentVerts[0][1]);
        for (let i = 1; i < currentVerts.length; i++) {{
            ctx.lineTo(currentVerts[i][0], currentVerts[i][1]);
        }}

        // Draw line to mouse position
        if (mouseEvent) {{
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const mx = (mouseEvent.clientX - rect.left) * scaleX;
            const my = (mouseEvent.clientY - rect.top) * scaleY;
            ctx.lineTo(mx, my);
        }}

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw vertices as circles
        currentVerts.forEach(([x, y]) => {{
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }});
    }}
}}

function drawPolygon(ctx, verts, fillColor, strokeColor, closed) {{
    if (verts.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(verts[0][0], verts[0][1]);
    for (let i = 1; i < verts.length; i++) {{
        ctx.lineTo(verts[i][0], verts[i][1]);
    }}
    if (closed) ctx.closePath();

    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 3;
    ctx.stroke();

    // Label
    const cx = verts.reduce((s, v) => s + v[0], 0) / verts.length;
    const cy = verts.reduce((s, v) => s + v[1], 0) / verts.length;
    ctx.font = 'bold 16px sans-serif';
    ctx.fillStyle = strokeColor;
    ctx.textAlign = 'center';
    ctx.fillText(fillColor.includes('52, 152, 219') ? 'FEMUR' : 'HUMERUS', cx, cy);
}}

function updateButtonStates() {{
    document.querySelectorAll('.region-btn').forEach(btn => {{
        const slide = btn.dataset.slide;
        const region = btn.dataset.region;
        const isDone = regions[slide]?.[region]?.length > 0;
        btn.classList.toggle('done', isDone);
    }});
}}

function updateProgress() {{
    let count = 0;
    for (const slide of SLIDES) {{
        if (regions[slide.name]?.femur?.length > 0) count++;
        if (regions[slide.name]?.humerus?.length > 0) count++;
    }}
    document.getElementById('progress').textContent = count;
}}

function clearSlideRegions(slideName) {{
    if (!confirm(`Clear all regions for ${{slideName}}?`)) return;
    delete regions[slideName];
    currentVerts = [];
    if (activeSlide === slideName) {{
        activeSlide = null;
        activeRegion = null;
    }}
    saveToStorage();
    updateProgress();
    updateButtonStates();
    drawCanvas(slideName);
}}

function exportRegions() {{
    // Scale vertices back to full resolution
    const exportData = {{
        slides: {{}},
        metadata: {{
            generated_at: new Date().toISOString(),
            version: '1.0'
        }}
    }};

    for (const slide of SLIDES) {{
        const slideRegions = regions[slide.name];
        if (!slideRegions) continue;

        const scaleFactor = slide.scaleFactor;
        exportData.slides[slide.name] = {{
            full_width: slide.fullWidth,
            full_height: slide.fullHeight,
            pixel_size_um: slide.pixelSizeUm,
            scale_factor: scaleFactor
        }};

        if (slideRegions.femur) {{
            exportData.slides[slide.name].femur = {{
                vertices_px: slideRegions.femur.map(([x, y]) => [
                    Math.round(x / scaleFactor),
                    Math.round(y / scaleFactor)
                ])
            }};
        }}
        if (slideRegions.humerus) {{
            exportData.slides[slide.name].humerus = {{
                vertices_px: slideRegions.humerus.map(([x, y]) => [
                    Math.round(x / scaleFactor),
                    Math.round(y / scaleFactor)
                ])
            }};
        }}
    }}

    // Download as file
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bone_regions.json';
    a.click();
    URL.revokeObjectURL(url);

    document.getElementById('status').textContent = 'Exported bone_regions.json';
}}

function loadFromFile() {{
    document.getElementById('import-file').click();
}}

function handleImport(event) {{
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {{
        try {{
            const data = JSON.parse(e.target.result);

            // Convert back to thumbnail coordinates
            for (const slide of SLIDES) {{
                const slideData = data.slides?.[slide.name];
                if (!slideData) continue;

                if (!regions[slide.name]) regions[slide.name] = {{}};

                const scaleFactor = slide.scaleFactor;

                if (slideData.femur?.vertices_px) {{
                    regions[slide.name].femur = slideData.femur.vertices_px.map(([x, y]) => [
                        x * scaleFactor,
                        y * scaleFactor
                    ]);
                }}
                if (slideData.humerus?.vertices_px) {{
                    regions[slide.name].humerus = slideData.humerus.vertices_px.map(([x, y]) => [
                        x * scaleFactor,
                        y * scaleFactor
                    ]);
                }}
            }}

            saveToStorage();
            updateProgress();
            updateButtonStates();
            SLIDES.forEach(s => drawCanvas(s.name));

            document.getElementById('status').textContent = 'Imported regions from file';
        }} catch (err) {{
            alert('Failed to parse JSON: ' + err.message);
        }}
    }};
    reader.readAsText(file);
    event.target.value = '';  // Reset for re-import
}}
</script>

</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nWrote {output_path}")
    print(f"Open in browser to annotate {len(slides_data)} slides")


def main():
    parser = argparse.ArgumentParser(description='Generate bone region annotation HTML')
    parser.add_argument('--czi-dir', type=Path, help='Directory containing CZI files')
    parser.add_argument('--zarr-dir', type=Path, help='Directory containing OME-Zarr files')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output HTML file path')
    parser.add_argument('--scale-factor', type=float, default=0.05,
                        help='Thumbnail scale factor for CZI (default: 0.05)')
    parser.add_argument('--channel', type=int, default=0,
                        help='Channel to display (default: 0)')
    parser.add_argument('--zarr-level', type=int, default=2,
                        help='Pyramid level for OME-Zarr (default: 2)')

    args = parser.parse_args()

    if not args.czi_dir and not args.zarr_dir:
        parser.error('Must specify --czi-dir or --zarr-dir')

    slides_data = []

    # Process CZI files
    if args.czi_dir:
        czi_files = sorted(args.czi_dir.glob('*.czi'))
        print(f"Found {len(czi_files)} CZI files in {args.czi_dir}")

        for czi_path in czi_files:
            try:
                rgb, full_w, full_h, px_size = read_czi_thumbnail(
                    czi_path,
                    scale_factor=args.scale_factor,
                    display_channel=args.channel
                )

                slides_data.append({
                    'name': czi_path.stem,
                    'image_b64': image_to_base64(rgb),
                    'thumb_width': rgb.shape[1],
                    'thumb_height': rgb.shape[0],
                    'full_width': full_w,
                    'full_height': full_h,
                    'scale_factor': args.scale_factor,
                    'pixel_size_um': px_size,
                })

                del rgb
                gc.collect()

            except Exception as e:
                print(f"  Error processing {czi_path.name}: {e}")

    # Process OME-Zarr files
    if args.zarr_dir:
        zarr_dirs = sorted([d for d in args.zarr_dir.iterdir()
                           if d.is_dir() and d.suffix in ('.zarr', '.ome.zarr')
                           or (d / '.zarray').exists() or (d / '.zgroup').exists()])

        # Also look for .ome.zarr directories
        zarr_dirs.extend(sorted(args.zarr_dir.glob('*.ome.zarr')))
        zarr_dirs = sorted(set(zarr_dirs))

        print(f"Found {len(zarr_dirs)} OME-Zarr directories in {args.zarr_dir}")

        for zarr_path in zarr_dirs:
            try:
                rgb, full_w, full_h, px_size = read_zarr_thumbnail(
                    zarr_path,
                    level=args.zarr_level
                )

                # Calculate effective scale factor
                if full_w and full_h:
                    scale_factor = rgb.shape[1] / full_w
                else:
                    scale_factor = 1.0 / (2 ** args.zarr_level)

                slides_data.append({
                    'name': zarr_path.stem.replace('.ome', '').replace('_rotated', ''),
                    'image_b64': image_to_base64(rgb),
                    'thumb_width': rgb.shape[1],
                    'thumb_height': rgb.shape[0],
                    'full_width': full_w or rgb.shape[1],
                    'full_height': full_h or rgb.shape[0],
                    'scale_factor': scale_factor,
                    'pixel_size_um': px_size,
                })

                del rgb
                gc.collect()

            except Exception as e:
                print(f"  Error processing {zarr_path.name}: {e}")

    if not slides_data:
        print("No slides found to process!")
        sys.exit(1)

    print(f"\nProcessed {len(slides_data)} slides")

    # Generate HTML
    generate_html(slides_data, args.output)


if __name__ == '__main__':
    main()
