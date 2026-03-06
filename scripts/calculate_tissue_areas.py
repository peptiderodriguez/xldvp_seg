#!/usr/bin/env python
"""Calculate tissue areas from CZI files using the pipeline's variance-based tissue detection.

Uses the same K-means variance-based approach as the segmentation pipeline
(segmentation.detection.tissue), ensuring consistency between tissue detection
used for segmentation and for density normalization.

Optionally intersects with bone region polygons for per-bone density.

Usage:
    # Whole-slide tissue area
    python scripts/calculate_tissue_areas.py \
        --czi-dir /path/to/czis \
        --output /path/to/tissue_areas.json

    # Per-bone tissue area with density
    python scripts/calculate_tissue_areas.py \
        --czi-dir /path/to/czis \
        --regions /path/to/bone_regions.json \
        --detections /path/to/all_mks_with_bone.json \
        --output /path/to/tissue_areas_by_bone.json
"""
import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.draw import polygon as draw_polygon
from skimage.morphology import closing, dilation, disk, erosion, remove_small_objects

# Use the pipeline's own tissue detection
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segmentation.detection.tissue import (
    _normalize_to_uint8,
    calculate_block_variances,
    compute_pixel_level_tissue_mask,
    compute_variance_threshold,
)


def read_czi_and_detect_tissue(czi_path, scale_factor=0.05):
    """Read CZI at reduced resolution and detect tissue using pipeline's variance method.

    Uses the same approach as the segmentation pipeline:
    1. Normalize to uint8 (percentile-based, same as pipeline)
    2. Calculate block variances
    3. K-means 3-cluster → variance threshold (max of background cluster)
    4. Pixel-level tissue mask via local variance (uniform_filter)

    Returns:
        gray_u8: uint8 grayscale image (normalized)
        rgb: uint8 RGB for visualization
        tissue_mask: bool tissue mask (pipeline-consistent)
        full_width, full_height: full-res dimensions
        pixel_size_um: pixel size
        variance_threshold: calibrated variance threshold
    """
    from aicspylibczi import CziFile

    czi = CziFile(str(czi_path))

    pixel_size_um = 0.22
    try:
        scaling = czi.get_scaling()
        if scaling and len(scaling) >= 2:
            pixel_size_um = scaling[0] * 1e6
    except Exception:
        pass

    bbox = czi.get_mosaic_scene_bounding_box(index=0)
    region = (bbox.x, bbox.y, bbox.w, bbox.h)
    full_width, full_height = bbox.w, bbox.h

    print(f"  Reading {czi_path.name} ({full_width}x{full_height}) at {scale_factor:.0%}...",
          end=" ", flush=True)

    img = czi.read_mosaic(C=0, region=region, scale_factor=scale_factor)
    img = np.squeeze(img)

    # Get raw grayscale (for intensity-based thresholding)
    if img.ndim == 3:
        gray_raw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[-1] == 3 else img[0]
    else:
        gray_raw = img
    gray_raw = gray_raw.astype(np.float32)

    # Normalize to uint8 — same function the pipeline uses (for variance-based detection)
    if img.dtype == np.uint8:
        gray_u8 = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_u8, ok = _normalize_to_uint8(gray_raw.astype(img.dtype))
        if not ok:
            print("WARNING: normalization failed")
            h, w = gray_raw.shape[:2]
            return (gray_u8, gray_raw, np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=bool), full_width, full_height, pixel_size_um, 0)

    # Build RGB for visualization
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

    # Scale block_size proportionally (pipeline uses 512 at full res)
    block_size = max(16, int(512 * scale_factor))

    # Step 1: Calculate block variances (same as pipeline's calibrate_tissue_threshold)
    variances, means = calculate_block_variances(gray_u8, block_size=block_size)

    if len(variances) < 10:
        print("WARNING: too few blocks for calibration")
        variance_threshold = 15.0
    else:
        # Step 2: K-means 3-cluster → threshold (same as pipeline)
        variance_threshold = compute_variance_threshold(np.array(variances))

    # Step 3: Pixel-level tissue mask via local variance (same as pipeline)
    tissue_mask = compute_pixel_level_tissue_mask(gray_u8, variance_threshold, block_size=7)

    # NOTE: morphological cleanup + intensity filtering done per-bone in
    # calculate_tissue_area_by_bone() so each bone gets its own Otsu threshold

    tissue_frac = np.sum(tissue_mask) / tissue_mask.size
    print(f"var_threshold={variance_threshold:.1f}, tissue={tissue_frac:.0%}")

    return gray_u8, gray_raw, rgb, tissue_mask, full_width, full_height, pixel_size_um, variance_threshold


def rasterize_polygon_to_mask(vertices_px, scale_factor, mask_shape):
    """Rasterize polygon vertices to a boolean mask at thumbnail resolution."""
    verts = np.array(vertices_px)
    cols = (verts[:, 0] * scale_factor).astype(int)
    rows = (verts[:, 1] * scale_factor).astype(int)
    rows = np.clip(rows, 0, mask_shape[0] - 1)
    cols = np.clip(cols, 0, mask_shape[1] - 1)

    rr, cc = draw_polygon(rows, cols, shape=mask_shape)
    mask = np.zeros(mask_shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def calculate_tissue_area_whole_slide(czi_path, scale_factor=0.05):
    """Calculate whole-slide tissue area (backwards-compatible)."""
    gray_u8, gray_raw, rgb, tissue_mask, full_width, full_height, pixel_size_um, var_thresh = \
        read_czi_and_detect_tissue(czi_path, scale_factor)

    tissue_pixels = int(np.sum(tissue_mask))
    original_pixels_per_reduced = (1 / scale_factor) ** 2
    pixel_area_um2 = pixel_size_um ** 2
    tissue_area_mm2 = tissue_pixels * original_pixels_per_reduced * pixel_area_um2 / 1e6
    tissue_fraction = tissue_pixels / tissue_mask.size

    return {
        'slide': czi_path.stem,
        'tissue_area_mm2': round(tissue_area_mm2, 3),
        'tissue_fraction': round(tissue_fraction, 4),
        'variance_threshold': round(var_thresh, 2),
        'full_width_px': full_width,
        'full_height_px': full_height,
        'pixel_size_um': pixel_size_um,
        'scale_factor_used': scale_factor,
    }


def calculate_tissue_area_by_bone(czi_path, slide_regions, scale_factor=0.05):
    """Calculate tissue area per bone region using pipeline's variance-based detection."""
    gray_u8, gray_raw, rgb, tissue_mask, full_width, full_height, pixel_size_um, var_thresh = \
        read_czi_and_detect_tissue(czi_path, scale_factor)

    original_pixels_per_reduced = (1 / scale_factor) ** 2
    pixel_area_um2 = pixel_size_um ** 2
    mask_shape = tissue_mask.shape

    total_tissue_pixels = int(np.sum(tissue_mask))
    total_tissue_area_mm2 = total_tissue_pixels * original_pixels_per_reduced * pixel_area_um2 / 1e6

    bone_results = {}
    bone_masks = {}       # polygon outlines
    bone_tissue_masks = {}  # tissue within each bone (per-bone thresholded)

    for bone_name in ['femur', 'humerus']:
        bone_data = slide_regions.get(bone_name, {})
        vertices = bone_data.get('vertices_px')
        if not vertices or len(vertices) < 3:
            print(f"    {bone_name}: no polygon")
            bone_results[bone_name] = {
                'polygon_area_mm2': 0, 'tissue_area_mm2': 0, 'tissue_fraction': 0,
            }
            continue

        bone_mask = rasterize_polygon_to_mask(vertices, scale_factor, mask_shape)

        # Per-bone Otsu on RAW intensity within the bone polygon
        # Brightfield: tissue is DARK (absorbs light), background is BRIGHT
        bone_pixels_raw = gray_raw[bone_mask]
        bone_nonzero_raw = bone_pixels_raw[bone_pixels_raw > 0]
        if len(bone_nonzero_raw) > 100:
            from skimage.filters import threshold_otsu
            bone_otsu = threshold_otsu(bone_nonzero_raw)
            intensity_ceiling = bone_otsu
            bone_tissue = bone_mask & (gray_raw > 0) & (gray_raw < intensity_ceiling)
        else:
            bone_tissue = bone_mask & (gray_raw > 0)
            intensity_ceiling = 0

        # Light cleanup: just remove small noise specks, no dilation/filling
        if np.any(bone_tissue):
            bone_tissue = remove_small_objects(bone_tissue, min_size=50)

        bone_masks[bone_name] = bone_mask
        bone_tissue_masks[bone_name] = bone_tissue
        bone_tissue_pixels = int(np.sum(bone_tissue))
        bone_polygon_pixels = int(np.sum(bone_mask))

        bone_tissue_area_mm2 = bone_tissue_pixels * original_pixels_per_reduced * pixel_area_um2 / 1e6
        bone_polygon_area_mm2 = bone_polygon_pixels * original_pixels_per_reduced * pixel_area_um2 / 1e6
        bone_tissue_fraction = bone_tissue_pixels / bone_polygon_pixels if bone_polygon_pixels > 0 else 0

        bone_results[bone_name] = {
            'polygon_area_mm2': round(bone_polygon_area_mm2, 3),
            'tissue_area_mm2': round(bone_tissue_area_mm2, 3),
            'tissue_fraction': round(bone_tissue_fraction, 4),
        }

        print(f"    {bone_name}: {bone_tissue_area_mm2:.1f} mm² tissue "
              f"({bone_tissue_fraction:.0%} of {bone_polygon_area_mm2:.1f} mm² polygon)"
              f" [otsu_ceiling={intensity_ceiling:.0f}]")

    result = {
        'slide': czi_path.stem,
        'full_width_px': full_width,
        'full_height_px': full_height,
        'pixel_size_um': pixel_size_um,
        'variance_threshold': round(var_thresh, 2),
        'whole_slide_tissue_area_mm2': round(total_tissue_area_mm2, 3),
        'bones': bone_results,
    }

    return result, rgb, gray_raw, tissue_mask, bone_masks, bone_tissue_masks


def load_detections_by_bone(path):
    """Load detections and count per slide x bone."""
    with open(path) as f:
        data = json.load(f)

    counts = defaultdict(lambda: defaultdict(int))
    for det in data:
        slide = det.get('slide', '')
        bone = det.get('bone', 'unknown')
        counts[slide][bone] += 1
    return dict(counts)


def generate_visualization_html(slides_vis, output_path):
    """Generate HTML with tissue mask overlaid on slide thumbnails, tinted by bone region."""
    from PIL import Image, ImageDraw

    cards_html = []
    for sv in slides_vis:
        rgb = sv['rgb']
        bone_tissue_masks = sv['bone_tissue_masks']
        result = sv['result']
        det_counts = sv.get('det_counts', {})

        # Subtle tint on tissue pixels (now correct: dark pixels = tissue in brightfield)
        overlay = rgb.copy().astype(np.float32)
        tint_alpha = 0.35
        for bone_name, color in [('femur', (50, 120, 255)), ('humerus', (255, 150, 30))]:
            if bone_name in bone_tissue_masks:
                m = bone_tissue_masks[bone_name]
                overlay[m, 0] = overlay[m, 0] * (1 - tint_alpha) + color[0] * tint_alpha
                overlay[m, 1] = overlay[m, 1] * (1 - tint_alpha) + color[1] * tint_alpha
                overlay[m, 2] = overlay[m, 2] * (1 - tint_alpha) + color[2] * tint_alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Draw bone polygon outlines
        pil_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_img)

        scale = sv.get('scale_factor', 0.03)
        label_colors = {'femur': (100, 180, 255), 'humerus': (255, 180, 80)}
        for bone_name in ['femur', 'humerus']:
            bone_data = sv.get('regions', {}).get(bone_name, {})
            verts = bone_data.get('vertices_px')
            if verts and len(verts) >= 3:
                # Thin white polygon outline — just a boundary, not a fill
                scaled = [(int(v[0] * scale), int(v[1] * scale)) for v in verts]
                draw.line(scaled + [scaled[0]], fill=(255, 255, 255), width=2)
                # Label in bone color
                cx = sum(v[0] for v in scaled) // len(scaled)
                cy = sum(v[1] for v in scaled) // len(scaled)
                draw.text((cx - 25, cy - 10), bone_name.upper(), fill=label_colors[bone_name])


        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        slide_name = result['slide']
        bones = result['bones']
        stats_rows = []
        for bone_name in ['femur', 'humerus']:
            b = bones.get(bone_name, {})
            n_cells = det_counts.get(bone_name, 0)
            tissue_mm2 = b.get('tissue_area_mm2', 0)
            density = n_cells / tissue_mm2 if tissue_mm2 > 0 else 0
            poly_mm2 = b.get('polygon_area_mm2', 0)
            frac = b.get('tissue_fraction', 0)
            stats_rows.append(
                f'<tr class="{bone_name}">'
                f'<td>{bone_name.title()}</td>'
                f'<td>{tissue_mm2:.1f}</td>'
                f'<td>{poly_mm2:.1f}</td>'
                f'<td>{frac:.0%}</td>'
                f'<td>{n_cells}</td>'
                f'<td>{density:.2f}</td>'
                f'</tr>'
            )

        cards_html.append(f'''
        <div class="slide-card">
            <div class="slide-header">{slide_name}
                <span class="var-thresh">var_thresh: {result.get("variance_threshold", "?")}</span>
            </div>
            <img src="data:image/jpeg;base64,{b64}" style="width:100%;height:auto;">
            <table class="stats-table">
                <tr><th>Bone</th><th>Tissue mm²</th><th>Polygon mm²</th><th>Fill %</th><th>Cells</th><th>Density</th></tr>
                {''.join(stats_rows)}
            </table>
        </div>
        ''')

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Tissue Area by Bone Region (Pipeline Method)</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    background: #1a1a2e;
    color: #eee;
    margin: 0;
    padding: 20px;
}}
h1 {{ text-align: center; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }}
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
.legend .femur {{ background: rgba(80,160,255,0.3); border: 2px solid #50a0ff; }}
.legend .humerus {{ background: rgba(255,160,50,0.3); border: 2px solid #ffa032; }}
.container {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
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
    font-size: 15px;
    display: flex;
    justify-content: space-between;
}}
.var-thresh {{
    font-size: 12px;
    color: #888;
    font-weight: 400;
}}
.stats-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.stats-table th {{
    padding: 6px 10px;
    text-align: left;
    background: #0f3460;
    font-weight: 500;
    color: #aaa;
}}
.stats-table td {{
    padding: 5px 10px;
    border-top: 1px solid #0f3460;
}}
.stats-table tr.femur td {{ color: #6cb4ff; }}
.stats-table tr.humerus td {{ color: #ffb050; }}
.stats-table td:nth-child(n+2) {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<h1>Tissue Area by Bone Region</h1>
<p class="subtitle">Pipeline variance-based tissue detection (K-means calibrated) intersected with bone polygons &middot; density = cells / tissue mm²</p>
<div class="legend">
    <span class="femur">Femur (blue tint = tissue)</span>
    <span class="humerus">Humerus (orange tint = tissue)</span>
</div>
<div class="container">
{''.join(cards_html)}
</div>
<p style="text-align:center; color:#555; margin-top:20px; font-size:12px;">
    Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</p>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\nVisualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate tissue areas using pipeline variance-based detection')
    parser.add_argument('--czi-dir', type=Path, required=True,
                        help='Directory containing CZI files')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output JSON file')
    parser.add_argument('--scale-factor', type=float, default=0.05,
                        help='Downsampling factor (default: 0.05)')
    parser.add_argument('--slides', nargs='+',
                        help='Only process specific slides (by stem name)')
    parser.add_argument('--regions', type=Path,
                        help='Bone regions JSON (from annotate_bone_regions.py)')
    parser.add_argument('--detections', type=Path,
                        help='Detections JSON with bone field (for density calculation)')
    parser.add_argument('--html', type=Path,
                        help='Output HTML visualization (auto-set to <output>.html if --regions used)')

    args = parser.parse_args()

    czi_files = sorted(args.czi_dir.glob('*.czi'))
    print(f"Found {len(czi_files)} CZI files")

    if args.slides:
        czi_files = [f for f in czi_files if f.stem in args.slides]
        print(f"Filtering to {len(czi_files)} specified slides")

    # Load bone regions if provided
    bone_regions = None
    if args.regions:
        with open(args.regions) as f:
            regions_data = json.load(f)
        bone_regions = regions_data.get('slides', regions_data)
        print(f"Loaded bone regions for {len(bone_regions)} slides")

    # Load detection counts if provided
    det_counts = None
    if args.detections:
        det_counts = load_detections_by_bone(args.detections)
        print(f"Loaded detection counts for {len(det_counts)} slides")

    # Auto-set HTML output
    html_path = args.html
    if html_path is None and bone_regions:
        html_path = args.output.with_suffix('.html')

    # Process slides
    results = []
    slides_vis = []

    for czi_path in czi_files:
        slide_name = czi_path.stem
        try:
            if bone_regions and slide_name in bone_regions:
                slide_regions = bone_regions[slide_name]
                result, rgb, gray_raw, tissue_mask, bone_masks, bone_tissue_masks = \
                    calculate_tissue_area_by_bone(
                        czi_path, slide_regions, scale_factor=args.scale_factor,
                    )

                if det_counts and slide_name in det_counts:
                    slide_det = det_counts[slide_name]
                    for bone_name in ['femur', 'humerus']:
                        n = slide_det.get(bone_name, 0)
                        tissue_mm2 = result['bones'][bone_name]['tissue_area_mm2']
                        result['bones'][bone_name]['n_cells'] = n
                        result['bones'][bone_name]['density'] = round(
                            n / tissue_mm2, 4) if tissue_mm2 > 0 else 0

                results.append(result)

                if html_path:
                    slides_vis.append({
                        'rgb': rgb,  # normalized display
                        'gray_raw': gray_raw,  # raw intensity for tissue viz
                        'bone_masks': bone_masks,
                        'bone_tissue_masks': bone_tissue_masks,
                        'result': result,
                        'det_counts': det_counts.get(slide_name, {}) if det_counts else {},
                        'regions': slide_regions,
                        'scale_factor': args.scale_factor,
                    })
            else:
                result = calculate_tissue_area_whole_slide(czi_path, scale_factor=args.scale_factor)
                results.append(result)

        except Exception as e:
            print(f"  Error processing {slide_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if bone_regions:
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'method': 'pipeline_variance_kmeans',
            'czi_dir': str(args.czi_dir),
            'regions_file': str(args.regions),
            'detections_file': str(args.detections) if args.detections else None,
            'scale_factor': args.scale_factor,
            'n_slides': len(results),
            'results': results,
        }
    else:
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'method': 'pipeline_variance_kmeans',
            'czi_dir': str(args.czi_dir),
            'scale_factor': args.scale_factor,
            'n_slides': len(results),
            'slides': {r['slide']: r for r in results}
        }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Generate HTML visualization
    if html_path and slides_vis:
        generate_visualization_html(slides_vis, html_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if bone_regions:
        print(f"{'Slide':<25} {'Femur mm²':>10} {'Humerus mm²':>12} "
              f"{'F cells':>8} {'H cells':>8} {'F dens':>8} {'H dens':>8}")
        print("-" * 80)
        for r in results:
            fem = r['bones'].get('femur', {})
            hum = r['bones'].get('humerus', {})
            print(f"{r['slide']:<25} "
                  f"{fem.get('tissue_area_mm2', 0):>10.1f} "
                  f"{hum.get('tissue_area_mm2', 0):>12.1f} "
                  f"{fem.get('n_cells', 0):>8} "
                  f"{hum.get('n_cells', 0):>8} "
                  f"{fem.get('density', 0):>8.2f} "
                  f"{hum.get('density', 0):>8.2f}")
    else:
        total_area = sum(r['tissue_area_mm2'] for r in results)
        print(f"Total tissue area: {total_area:.1f} mm²")
        if results:
            print(f"Mean per slide: {total_area/len(results):.1f} mm²")


if __name__ == '__main__':
    main()
