#!/usr/bin/env python3
"""Detect nerve fibers (NfL, etc.) from CZI and split into LMD-sized pieces.

Uses a ridge/tubeness filter (Meijering) to find filamentous structures,
then thresholds, extracts connected components, and splits large regions
into equal-area pieces for LMD export.

Works at reduced resolution for speed, scales contours back to full res.

Usage:
    python scripts/detect_fibers_for_lmd.py \
        --czi-path /path/to/slide.czi \
        --channel-spec "detect=NfL" \
        --output-dir /path/to/output

    # With custom scale and fiber width range
    python scripts/detect_fibers_for_lmd.py \
        --czi-path /path/to/slide.czi \
        --channel 2 \
        --scale-factor 0.25 \
        --fiber-width-um 2,10 \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import label as ndlabel
from skimage.filters import meijering
from skimage.measure import regionprops, find_contours
from skimage.morphology import remove_small_objects, binary_closing, disk

from segmentation.io.czi_loader import CZILoader
from segmentation.utils.json_utils import atomic_json_dump
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def load_channel_reduced(czi_path, channel_idx, scale_factor, scene=0):
    """Load a single CZI channel at reduced resolution.

    Returns:
        channel_array: 2D uint16 array (height x width)
        pixel_size_um: full-resolution pixel size in um
        mosaic_x, mosaic_y: mosaic origin in full-res pixels
        full_h, full_w: full-resolution dimensions
    """
    from aicspylibczi import CziFile

    loader = CZILoader(str(czi_path))
    pixel_size_um = loader.get_pixel_size()

    czi = CziFile(str(czi_path))
    try:
        bbox = czi.get_mosaic_scene_bounding_box(index=scene)
    except Exception:
        bbox = czi.get_mosaic_bounding_box()

    region = (bbox.x, bbox.y, bbox.w, bbox.h)
    logger.info(f"CZI scene {scene}: {bbox.w}x{bbox.h} px, pixel_size={pixel_size_um:.4f} um/px")
    logger.info(f"Reading channel {channel_idx} at {scale_factor:.0%} scale...")

    img = czi.read_mosaic(C=channel_idx, region=region, scale_factor=scale_factor)
    img = np.squeeze(img)
    logger.info(f"  Shape: {img.shape}, dtype: {img.dtype}")

    return img, pixel_size_um, bbox.x, bbox.y, bbox.h, bbox.w


def detect_fibers(channel, pixel_size_um, scale_factor, fiber_width_um=(2, 10),
                  min_area_um2=50):
    """Detect filamentous structures using Meijering ridge filter.

    Args:
        channel: 2D array (reduced resolution)
        pixel_size_um: full-res pixel size
        scale_factor: reduction factor used
        fiber_width_um: (min, max) fiber width in um
        min_area_um2: minimum region area in um^2

    Returns:
        labeled: labeled array of fiber regions
        n_regions: number of regions found
    """
    # Convert fiber widths from um to reduced-resolution pixels
    reduced_pixel_size = pixel_size_um / scale_factor
    min_sigma = fiber_width_um[0] / reduced_pixel_size / 2  # sigma ~ half-width
    max_sigma = fiber_width_um[1] / reduced_pixel_size / 2
    sigmas = np.linspace(max(0.5, min_sigma), max(1.0, max_sigma), 8)

    logger.info(f"  Fiber width: {fiber_width_um[0]}-{fiber_width_um[1]} um "
                f"= {min_sigma:.1f}-{max_sigma:.1f} px (reduced res)")
    logger.info(f"  Meijering sigmas: {[f'{s:.1f}' for s in sigmas]}")

    # Percentile normalize to float [0, 1] (exclude zeros = CZI padding)
    valid = channel[channel > 0]
    if len(valid) == 0:
        logger.warning("  No valid pixels in channel")
        return np.zeros_like(channel, dtype=np.int32), 0

    p1 = float(np.percentile(valid, 1))
    p99 = float(np.percentile(valid, 99.5))
    norm = np.clip((channel.astype(np.float32) - p1) / max(p99 - p1, 1), 0, 1)
    # Keep CZI padding as 0
    norm[channel == 0] = 0

    logger.info("  Running Meijering ridge filter...")
    ridge = meijering(norm, sigmas=sigmas, black_ridges=False)

    # Threshold: Otsu on nonzero ridge values
    ridge_valid = ridge[ridge > 0]
    if len(ridge_valid) == 0:
        logger.warning("  No ridge response")
        return np.zeros_like(channel, dtype=np.int32), 0

    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(ridge_valid)
    logger.info(f"  Ridge Otsu threshold: {thresh:.4f}")

    binary = ridge > thresh

    # Morphological cleanup: close small gaps in fibers
    close_radius = max(1, int(fiber_width_um[0] / reduced_pixel_size))
    binary = binary_closing(binary, disk(close_radius))

    # Remove small objects
    min_area_px = int(min_area_um2 / (reduced_pixel_size ** 2))
    min_area_px = max(min_area_px, 10)
    binary = remove_small_objects(binary, min_size=min_area_px)

    labeled, n_regions = ndlabel(binary)
    logger.info(f"  Found {n_regions} fiber regions (min_area={min_area_um2} um²)")

    return labeled, n_regions


def regions_to_detections(labeled, n_regions, pixel_size_um, scale_factor,
                          mosaic_x, mosaic_y, slide_name,
                          target_area_um2=200, min_piece_area_um2=50):
    """Convert labeled regions to detection dicts with contours in global coordinates.

    Large regions are split into equal-area pieces via watershed.

    Args:
        labeled: labeled array at reduced resolution
        n_regions: number of labeled regions
        pixel_size_um: full-res pixel size
        scale_factor: reduction factor
        mosaic_x, mosaic_y: mosaic origin in full-res px
        slide_name: slide name for UIDs
        target_area_um2: target piece area for splitting
        min_piece_area_um2: minimum piece area

    Returns:
        list of detection dicts compatible with LMD export
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed

    reduced_pixel_size = pixel_size_um / scale_factor
    scale_inv = 1.0 / scale_factor  # reduced px → full-res px

    props = regionprops(labeled)
    detections = []
    det_id = 0

    for prop in props:
        area_um2 = prop.area * (reduced_pixel_size ** 2)
        if area_um2 < min_piece_area_um2:
            continue

        # Extract region mask
        minr, minc, maxr, maxc = prop.bbox
        region_mask = labeled[minr:maxr, minc:maxc] == prop.label

        # Split large regions
        n_pieces = max(1, int(area_um2 / target_area_um2))
        if n_pieces > 1:
            # Watershed split
            dist = distance_transform_edt(region_mask)
            from skimage.feature import peak_local_max
            min_dist = max(3, int(np.sqrt(prop.area / n_pieces) / 2))
            coords = peak_local_max(dist, min_distance=min_dist, num_peaks=n_pieces)
            if len(coords) < 2:
                coords = peak_local_max(dist, min_distance=2, num_peaks=n_pieces)
            markers = np.zeros_like(region_mask, dtype=np.int32)
            for i, (r, c) in enumerate(coords):
                markers[r, c] = i + 1
            if markers.max() > 0:
                pieces = watershed(-dist, markers, mask=region_mask)
            else:
                pieces = region_mask.astype(np.int32)
        else:
            pieces = region_mask.astype(np.int32)

        # Extract each piece
        for piece_label in range(1, pieces.max() + 1) if pieces.max() > 0 else [1]:
            if pieces.max() > 0:
                piece_mask = pieces == piece_label
            else:
                piece_mask = region_mask

            piece_area_px = piece_mask.sum()
            piece_area_um2 = piece_area_px * (reduced_pixel_size ** 2)
            if piece_area_um2 < min_piece_area_um2:
                continue

            # Find contour at reduced resolution
            contours = find_contours(piece_mask.astype(np.float64), 0.5)
            if not contours:
                continue
            # Take the longest contour
            contour = max(contours, key=len)

            # Convert to global full-res coordinates: (row, col) → (x_global, y_global)
            # contour is in (row, col) format within the bbox
            contour_global = []
            for r, c in contour:
                # Reduced-res local → reduced-res global → full-res global
                global_col = (c + minc) * scale_inv + mosaic_x
                global_row = (r + minr) * scale_inv + mosaic_y
                contour_global.append([float(global_col), float(global_row)])

            # Centroid in global full-res pixels
            piece_rows, piece_cols = np.where(piece_mask)
            cx_global = float((np.mean(piece_cols) + minc) * scale_inv + mosaic_x)
            cy_global = float((np.mean(piece_rows) + minr) * scale_inv + mosaic_y)

            det_id += 1
            uid = f"{slide_name}_nfl_{int(cx_global)}_{int(cy_global)}"

            det = {
                'id': uid,
                'uid': uid,
                'center': [cx_global - mosaic_x, cy_global - mosaic_y],  # tile-local (legacy)
                'global_center': [cx_global, cy_global],
                'global_center_um': [cx_global * pixel_size_um, cy_global * pixel_size_um],
                'slide_name': slide_name,
                'pixel_size_um': pixel_size_um,
                'outer_contour_global': contour_global,
                'contour_dilated_px': contour_global,  # alias for compatibility
                'features': {
                    'area': int(piece_area_px * (scale_inv ** 2)),  # full-res px
                    'area_um2': float(piece_area_um2),
                    'pixel_size_um': pixel_size_um,
                    'n_pieces': n_pieces,
                    'detection_method': 'meijering_ridge',
                },
                'rf_prediction': None,  # no classifier
            }
            detections.append(det)

    logger.info(f"  Generated {len(detections)} fiber pieces "
                f"({det_id} from {len(props)} regions)")
    return detections


def main():
    parser = argparse.ArgumentParser(
        description='Detect nerve fibers from CZI and split into LMD pieces')
    parser.add_argument('--czi-path', required=True, help='CZI file path')
    parser.add_argument('--channel', type=int, default=None,
                        help='Channel index for fiber detection')
    parser.add_argument('--channel-spec', default=None,
                        help='Channel spec (e.g., "detect=NfL" or "detect=750")')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--scale-factor', type=float, default=0.25,
                        help='CZI downsample factor (default: 1/4)')
    parser.add_argument('--fiber-width-um', default='1,30',
                        help='Min,max fiber width in um (default: 1,30 — covers thin axons to thick bundles)')
    parser.add_argument('--target-area-um2', type=float, default=200,
                        help='Target piece area in um² (default: 200)')
    parser.add_argument('--min-area-um2', type=float, default=50,
                        help='Minimum region area in um² (default: 50)')
    parser.add_argument('--scene', type=int, default=0, help='CZI scene index')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve channel
    if args.channel is not None:
        channel_idx = args.channel
    elif args.channel_spec:
        loader = CZILoader(args.czi_path)
        from segmentation.io.czi_loader import resolve_channel_indices
        mapping = resolve_channel_indices(args.channel_spec, loader)
        channel_idx = mapping.get('detect', list(mapping.values())[0])
        logger.info(f"Resolved channel spec '{args.channel_spec}' → channel {channel_idx}")
    else:
        parser.error("Provide --channel or --channel-spec")

    # Parse fiber width
    fw = [float(x) for x in args.fiber_width_um.split(',')]
    fiber_width_um = (fw[0], fw[1]) if len(fw) >= 2 else (fw[0], fw[0] * 5)

    # Load channel
    channel, pixel_size_um, mx, my, full_h, full_w = load_channel_reduced(
        args.czi_path, channel_idx, args.scale_factor, scene=args.scene)

    slide_name = Path(args.czi_path).stem

    # Detect fibers
    labeled, n_regions = detect_fibers(
        channel, pixel_size_um, args.scale_factor,
        fiber_width_um=fiber_width_um,
        min_area_um2=args.min_area_um2)

    if n_regions == 0:
        logger.warning("No fiber regions detected. Try adjusting --fiber-width-um or --min-area-um2.")
        return

    # Convert to detections
    detections = regions_to_detections(
        labeled, n_regions, pixel_size_um, args.scale_factor,
        mx, my, slide_name,
        target_area_um2=args.target_area_um2,
        min_piece_area_um2=args.min_area_um2)

    # Save
    output_path = output_dir / 'nfl_fiber_pieces.json'
    atomic_json_dump(detections, output_path)
    logger.info(f"Saved {len(detections)} fiber pieces to {output_path}")

    # Summary
    areas = [d['features']['area_um2'] for d in detections]
    logger.info(f"  Area range: {min(areas):.0f} - {max(areas):.0f} um²")
    logger.info(f"  Median area: {np.median(areas):.0f} um²")
    xs = [d['global_center'][0] for d in detections]
    ys = [d['global_center'][1] for d in detections]
    logger.info(f"  X range: {min(xs):.0f} - {max(xs):.0f} (CZI width: {full_w})")
    logger.info(f"  Y range: {min(ys):.0f} - {max(ys):.0f} (CZI height: {full_h})")


if __name__ == '__main__':
    main()
