#!/usr/bin/env python
"""Compare tissue detection outlines vs manual bone region annotations.

Generates an HTML visualization showing both overlaid on slide thumbnails.
"""

import argparse
import base64
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from skimage.measure import find_contours


def process_slide(czi_path, bone_regions, scale_factor=0.03):
    """Process a single slide and return visualization data."""
    from aicspylibczi import CziFile

    czi = CziFile(str(czi_path))
    slide_name = czi_path.stem

    # Get pixel size
    pixel_size_um = 0.22
    try:
        scaling = czi.get_scaling()
        if scaling and len(scaling) >= 2:
            pixel_size_um = scaling[0] * 1e6
    except Exception:
        pass

    # Read image
    bbox = czi.get_mosaic_scene_bounding_box(index=0)
    region = (bbox.x, bbox.y, bbox.w, bbox.h)
    full_width, full_height = bbox.w, bbox.h

    dims = czi.dims
    is_rgb = "A" in dims

    if is_rgb:
        img = czi.read_mosaic(C=0, region=region, scale_factor=scale_factor)
        img = np.squeeze(img)
        if img.ndim == 3:
            if img.shape[-1] == 3:
                rgb = img
                gray = np.mean(img, axis=-1)
            elif img.shape[0] == 3:
                rgb = np.transpose(img, (1, 2, 0))
                gray = np.mean(img, axis=0)
            else:
                gray = img[0]
                rgb = np.stack([gray, gray, gray], axis=-1)
        else:
            gray = img
            rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        img = czi.read_mosaic(C=0, region=region, scale_factor=scale_factor)
        gray = np.squeeze(img).astype(np.float32)
        # Normalize for display
        p1, p99 = np.percentile(gray[gray > 0], [1, 99])
        gray_disp = np.clip((gray - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([gray_disp, gray_disp, gray_disp], axis=-1)

    # Ensure uint8
    if rgb.dtype != np.uint8:
        if rgb.max() > 255:
            rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    gray = gray.astype(np.float32)

    # Otsu thresholding for tissue detection
    if is_rgb:
        gray_inv = 255 - gray if gray.max() > 1 else 1 - gray
        threshold = threshold_otsu(gray_inv[gray_inv > 0])
        tissue_mask = gray_inv > threshold
    else:
        valid = gray[gray > 0]
        if len(valid) > 0:
            threshold = threshold_otsu(valid)
            tissue_mask = gray > threshold
        else:
            tissue_mask = np.zeros_like(gray, dtype=bool)

    # Find tissue contours
    tissue_contours = find_contours(tissue_mask.astype(float), 0.5)

    # Create visualization
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    # Draw tissue contours (green, thin)
    for contour in tissue_contours:
        if len(contour) > 10:  # Skip tiny contours
            points = [(int(c[1]), int(c[0])) for c in contour]
            if len(points) > 2:
                draw.line(points + [points[0]], fill=(0, 255, 0), width=1)

    # Draw bone regions (thick, colored)
    if slide_name in bone_regions:
        for bone, color in [("femur", (0, 100, 255)), ("humerus", (255, 100, 0))]:
            if bone in bone_regions[slide_name] and "vertices_px" in bone_regions[slide_name][bone]:
                verts = bone_regions[slide_name][bone]["vertices_px"]
                # Scale to thumbnail coordinates
                scaled_verts = [(int(v[0] * scale_factor), int(v[1] * scale_factor)) for v in verts]
                if len(scaled_verts) > 2:
                    draw.line(scaled_verts + [scaled_verts[0]], fill=color, width=3)
                    # Label
                    cx = sum(v[0] for v in scaled_verts) // len(scaled_verts)
                    cy = sum(v[1] for v in scaled_verts) // len(scaled_verts)
                    draw.text((cx - 20, cy - 10), bone.upper(), fill=color)

    # Convert to base64
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "name": slide_name,
        "image_b64": b64,
        "width": pil_img.width,
        "height": pil_img.height,
        "full_width": full_width,
        "full_height": full_height,
        "threshold": float(threshold),
    }


def generate_html(slides_data, output_path):
    """Generate comparison HTML."""

    slides_html = []
    for s in slides_data:
        slides_html.append(
            f"""
        <div class="slide-card">
            <div class="slide-header">{s['name']}</div>
            <img src="data:image/jpeg;base64,{s['image_b64']}"
                 style="width:100%; height:auto;">
            <div class="slide-footer">
                Otsu threshold: {s['threshold']:.1f} |
                {s['full_width']}x{s['full_height']} px
            </div>
        </div>
        """
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Tissue Detection vs Bone Annotations</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    background: #1a1a2e;
    color: #eee;
    margin: 0;
    padding: 20px;
}}
h1 {{ text-align: center; margin-bottom: 10px; }}
.legend {{
    text-align: center;
    margin-bottom: 20px;
    font-size: 14px;
}}
.legend span {{
    margin: 0 15px;
    padding: 4px 12px;
    border-radius: 4px;
}}
.legend .tissue {{ background: rgba(0,255,0,0.3); border: 2px solid #0f0; }}
.legend .femur {{ background: rgba(0,100,255,0.3); border: 2px solid #0064ff; }}
.legend .humerus {{ background: rgba(255,100,0,0.3); border: 2px solid #ff6400; }}
.container {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
    gap: 20px;
}}
.slide-card {{
    background: #16213e;
    border-radius: 8px;
    overflow: hidden;
}}
.slide-header {{
    padding: 10px 15px;
    background: #0f3460;
    font-weight: 500;
}}
.slide-footer {{
    padding: 8px 15px;
    font-size: 12px;
    color: #888;
    border-top: 1px solid #0f3460;
}}
</style>
</head>
<body>
<h1>Tissue Detection vs Bone Annotations</h1>
<div class="legend">
    <span class="tissue">Green = Otsu tissue detection</span>
    <span class="femur">Blue = Femur annotation</span>
    <span class="humerus">Orange = Humerus annotation</span>
</div>
<div class="container">
{''.join(slides_html)}
</div>
<p style="text-align:center; color:#666; margin-top:20px;">
    Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--czi-dir", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--scale-factor", type=float, default=0.03)
    args = parser.parse_args()

    # Load bone regions
    with open(args.regions) as f:
        regions_data = json.load(f)
    bone_regions = regions_data.get("slides", regions_data)

    # Process slides
    czi_files = sorted(args.czi_dir.glob("*.czi"))
    print(f"Processing {len(czi_files)} slides...")

    slides_data = []
    for czi_path in czi_files:
        print(f"  {czi_path.name}...", flush=True)
        try:
            data = process_slide(czi_path, bone_regions, args.scale_factor)
            slides_data.append(data)
        except Exception as e:
            print(f"    Error: {e}")

    generate_html(slides_data, args.output)


if __name__ == "__main__":
    main()
