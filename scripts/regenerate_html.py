#!/usr/bin/env python3
"""
Consolidated HTML regeneration from existing detections.

Replaces all cell-type-specific regenerate scripts with a single generic version
that handles NMJ, MK, vessel, islet, tissue_pattern, and any future cell types.

Reads per-tile masks (HDF5) and the deduped detections JSON, loads CZI display
channels, composes RGB crops, and generates the full HTML annotation interface.

Usage:
    # Tissue pattern
    python scripts/regenerate_html.py \\
        --output-dir /path/to/run_output \\
        --czi-path /path/to/slide.czi \\
        --display-channels 1,2,0 \\
        --max-samples 15000

    # Islet with marker coloring
    python scripts/regenerate_html.py \\
        --output-dir /path/to/run_output \\
        --czi-path /path/to/slide.czi \\
        --cell-type islet \\
        --display-channels 2,3,5 \\
        --islet-marker-channels gcg:2,ins:3,sst:5

    # Vessel with quality filter
    python scripts/regenerate_html.py \\
        --output-dir /path/to/run_output \\
        --czi-path /path/to/slide.czi \\
        --cell-type vessel \\
        --vessel-quality-filter

    # NMJ sorted by RF score
    python scripts/regenerate_html.py \\
        --output-dir /path/to/run_output \\
        --czi-path /path/to/slide.czi \\
        --cell-type nmj \\
        --sort-by rf_prediction --sort-order desc
"""

import argparse
import json
import gc
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py

# HDF5 LZ4 support
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import cv2

from segmentation.io.czi_loader import get_loader, get_czi_metadata
from segmentation.io.html_export import (
    export_samples_to_html,
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
)
from segmentation.utils.logging import get_logger, setup_logging
from segmentation.utils.islet_utils import classify_islet_marker, compute_islet_marker_thresholds
from PIL import Image

logger = get_logger(__name__)


def create_sample(tile_rgb, masks, feat, pixel_size_um, slide_name, cell_type,
                  tile_percentiles=None, marker_thresholds=None, marker_map=None,
                  contour_thickness=2):
    """Create an HTML sample dict from a detection with mask-bounded crop."""
    mask_label = feat.get('tile_mask_label', feat.get('mask_label', 0))
    if mask_label == 0:
        try:
            mask_label = int(feat['id'].split('_')[-1])
        except (KeyError, ValueError, IndexError):
            return None

    mask = masks == mask_label
    if mask.sum() == 0:
        return None

    # Get centroid from features (local tile coords)
    center = feat.get('center', None)
    if center is None:
        ys, xs = np.where(mask)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
    else:
        cx, cy = center[0], center[1]

    # Mask bounding box for dynamic crop size
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_size = max(mask_h, mask_w)

    # Crop = 2x mask size, clamped to [224, 800]
    crop_size = max(224, min(800, int(mask_size * 2)))
    half = crop_size // 2

    y1_ideal = int(cy) - half
    y2_ideal = int(cy) + half
    x1_ideal = int(cx) - half
    x2_ideal = int(cx) + half

    y1 = max(0, y1_ideal)
    y2 = min(tile_rgb.shape[0], y2_ideal)
    x1 = max(0, x1_ideal)
    x2 = min(tile_rgb.shape[1], x2_ideal)

    if y2 <= y1 or x2 <= x1:
        return None

    crop = tile_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Pad if clamped at edges
    pad_top = max(0, y1 - y1_ideal)
    pad_bottom = max(0, y2_ideal - y2)
    pad_left = max(0, x1 - x1_ideal)
    pad_right = max(0, x2_ideal - x2)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                      mode='constant', constant_values=0)
        crop_mask = np.pad(crop_mask, ((pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='constant', constant_values=False)

    # Determine contour color
    features = feat.get('features', {})
    contour_color = (0, 255, 0)  # default green
    marker_class = None
    if cell_type == 'islet' and marker_thresholds is not None:
        marker_class, contour_color = classify_islet_marker(
            features, marker_thresholds, marker_map=marker_map)

    # Normalize and draw contour
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5, global_percentiles=tile_percentiles)
    _bw = (cell_type == 'islet')
    crop_with_contour = draw_mask_contour(
        crop_norm, crop_mask, color=contour_color, thickness=contour_thickness, bw_dashed=_bw)

    pil_img = Image.fromarray(crop_with_contour)
    img_b64, mime = image_to_base64(pil_img, format='PNG')

    # Build UID from global center
    tile_origin = feat.get('tile_origin', [0, 0])
    global_cx = tile_origin[0] + cx
    global_cy = tile_origin[1] + cy
    uid = feat.get('uid', f"{slide_name}_{cell_type}_{int(round(global_cx))}_{int(round(global_cy))}")

    area_um2 = features.get('area_um2', features.get('area', 0) * (pixel_size_um ** 2))

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }

    if marker_class is not None:
        stats['marker_class'] = marker_class
        stats['marker_color'] = f'#{contour_color[0]:02x}{contour_color[1]:02x}{contour_color[2]:02x}'
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'rf_prediction' in feat and feat['rf_prediction'] is not None:
        stats['rf_prediction'] = feat['rf_prediction']
    if 'detection_method' in features:
        dm = features['detection_method']
        stats['detection_method'] = ', '.join(dm) if isinstance(dm, list) else dm
    # Vessel-specific
    for vk in ('ring_completeness', 'circularity', 'wall_thickness_mean_um',
               'outer_diameter_um', 'vessel_type', 'has_sma_ring',
               'cd31_score', 'sma_thickness_mean_um'):
        if vk in features:
            stats[vk] = features[vk]

    return {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime,
        'stats': stats,
    }


def create_sample_from_contours(det, channel_arrays, display_channels, x_start, y_start,
                                mosaic_h, mosaic_w, pixel_size_um, slide_name, cell_type,
                                contour_thickness=2, max_crop_px=800, min_crop_px=300):
    """Create an HTML sample by cropping from CZI around a detection's center.

    Generic alternative to create_sample() — works for any cell type where detections
    have global coordinates and optionally contour data, but no per-tile HDF5 masks.
    Used for multiscale vessel, or any resumed run where masks were not saved.

    Draws contour outlines (inner/outer) if present in the detection dict.
    """
    features = det.get('features', {})

    # Get center in full-res CZI-global pixel coords
    # NOTE: 'center' has full-res coords; 'global_center' may be downscaled (multiscale)
    center = det.get('center')
    if center is None or len(center) < 2:
        return None
    cx, cy = float(center[0]), float(center[1])

    # Determine crop size — use contour bounding box when available
    # so the entire detection (including large vessels) is always visible
    contour_keys = ['inner_contour_global', 'outer_contour_global', 'sma_contour_global']
    all_contour_pts = []
    for ck in contour_keys:
        cd = det.get(ck)
        if cd is not None:
            pts = np.array(cd, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                all_contour_pts.append(pts)

    if all_contour_pts:
        combined = np.concatenate(all_contour_pts, axis=0)
        # Center on contour centroid (det['center'] can be wrong for multiscale)
        cx = float(np.mean(combined[:, 0]))
        cy = float(np.mean(combined[:, 1]))
        bbox_w = np.ptp(combined[:, 0])
        bbox_h = np.ptp(combined[:, 1])
        # 2x padding around the contour bounding box to ensure full visibility
        desired_crop = int(max(bbox_w, bbox_h) * 2.0)
    else:
        diameter_um = features.get('outer_diameter_um', 0)
        if diameter_um > 0:
            desired_crop = int((diameter_um / pixel_size_um) * 2)
        else:
            area_px = features.get('area', 0)
            if area_px > 0:
                desired_crop = int(2 * np.sqrt(area_px / np.pi) * 2)
            else:
                desired_crop = min_crop_px

    desired_crop = max(min_crop_px, desired_crop)
    # If the desired crop is larger than display max, we'll extract full-size and downscale
    scale_factor = 1.0
    extract_size = desired_crop
    if desired_crop > max_crop_px:
        scale_factor = max_crop_px / desired_crop
        # Keep extract_size as the full desired size for CZI extraction
    half = extract_size // 2

    # Convert to array-relative coords
    rel_cx = cx - x_start
    rel_cy = cy - y_start

    y1 = max(0, int(rel_cy) - half)
    x1 = max(0, int(rel_cx) - half)
    y2 = min(mosaic_h, int(rel_cy) + half)
    x2 = min(mosaic_w, int(rel_cx) + half)

    if y2 <= y1 or x2 <= x1:
        return None

    # Compose RGB crop from channel arrays
    rgb_channels = []
    for ch_idx in display_channels[:3]:
        if ch_idx in channel_arrays:
            rgb_channels.append(channel_arrays[ch_idx][y1:y2, x1:x2])
        else:
            _dtype = next(iter(channel_arrays.values())).dtype
            rgb_channels.append(np.zeros((y2 - y1, x2 - x1), dtype=_dtype))
    crop = np.stack(rgb_channels, axis=-1)

    if crop.size == 0:
        return None

    # Percentile normalize
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5)

    # Downscale if needed (show entire vessel at lower resolution)
    if scale_factor < 1.0:
        target_h = max(1, int(crop_norm.shape[0] * scale_factor))
        target_w = max(1, int(crop_norm.shape[1] * scale_factor))
        crop_norm = cv2.resize(crop_norm, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Draw contours if available (generic: look for any *_contour_global keys)
    contour_colors = {
        'inner_contour_global': (0, 255, 255),    # cyan — lumen
        'outer_contour_global': (0, 255, 0),       # green — outer boundary
        'sma_contour_global': (255, 0, 255),        # magenta — SMA ring
    }
    for contour_key, color in contour_colors.items():
        contour_data = det.get(contour_key)
        if contour_data is None:
            continue
        contour_pts = np.array(contour_data, dtype=np.float64).copy()
        if contour_pts.ndim != 2 or contour_pts.shape[0] < 3:
            continue
        # Shift global contour coords to crop-local: subtract (x1+x_start, y1+y_start)
        contour_pts[:, 0] -= (x1 + x_start)
        contour_pts[:, 1] -= (y1 + y_start)
        # Apply downscale factor
        if scale_factor < 1.0:
            contour_pts *= scale_factor
        contour_int = contour_pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(crop_norm, [contour_int], -1, color, contour_thickness)

    uid = det.get('uid', f"{slide_name}_{cell_type}_{int(round(cx))}_{int(round(cy))}")
    pil_img = Image.fromarray(crop_norm)
    img_b64, mime = image_to_base64(pil_img, format='PNG')

    area_um2 = features.get('area_um2', features.get('area', 0) * (pixel_size_um ** 2))
    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'rf_prediction' in det and det['rf_prediction'] is not None:
        stats['rf_prediction'] = det['rf_prediction']
    if 'detection_method' in features:
        dm = features['detection_method']
        stats['detection_method'] = ', '.join(dm) if isinstance(dm, list) else dm
    if 'scale_detected' in det:
        stats['scale_detected'] = det['scale_detected']
    # Vessel-specific stats
    for vk in ('ring_completeness', 'circularity', 'wall_thickness_mean_um',
               'outer_diameter_um', 'vessel_type', 'has_sma_ring',
               'cd31_score', 'sma_thickness_mean_um'):
        if vk in features:
            stats[vk] = features[vk]

    return {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime,
        'stats': stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate HTML from existing detections (all cell types)')

    # Required
    parser.add_argument('--output-dir', required=True,
                        help='Path to existing run output directory (contains tiles/ and *_detections.json)')
    parser.add_argument('--czi-path', required=True,
                        help='Path to CZI file')

    # General
    parser.add_argument('--detections', default=None,
                        help='Path to detections JSON file (overrides auto-detection from output-dir)')
    parser.add_argument('--cell-type', default=None,
                        help='Cell type (auto-detected from detections filename if omitted)')
    parser.add_argument('--display-channels', default=None,
                        help='Comma-separated channel indices for R,G,B display (default: auto)')
    parser.add_argument('--channels', default=None,
                        help='Comma-separated channel indices to load (default: auto from display-channels)')
    parser.add_argument('--scene', type=int, default=0,
                        help='CZI scene index (default: 0)')
    parser.add_argument('--tile-size', type=int, default=None,
                        help='Tile size used during detection (auto-detected from masks if omitted)')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in um (auto-detected from CZI if not specified)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Max HTML samples to generate (0=all, default: 0)')
    parser.add_argument('--samples-per-page', type=int, default=300,
                        help='Samples per HTML page (default: 300)')
    parser.add_argument('--sort-by', default='area',
                        help='Sort key: area, rf_prediction, elongation, confidence (default: area)')
    parser.add_argument('--sort-order', default='asc',
                        help='Sort order: asc or desc (default: asc)')
    parser.add_argument('--contour-thickness', type=int, default=2,
                        help='Contour line thickness (default: 2)')
    parser.add_argument('--score-threshold', type=float, default=0.0,
                        help='Min rf_prediction to show (default: 0.0 = all)')
    parser.add_argument('--prior-annotations', default=None,
                        help='Path to prior annotations JSON (pre-loaded into HTML localStorage)')
    parser.add_argument('--html-dir', default=None,
                        help='Custom HTML output directory (default: <output-dir>/html)')

    # Vessel-specific
    parser.add_argument('--vessel-quality-filter', action='store_true', default=False,
                        help='Apply vessel quality filter (ring>=0.30, circ>=0.15, wall>=1.5um)')

    # Islet-specific
    parser.add_argument('--islet-marker-channels', type=str, default='gcg:2,ins:3,sst:5',
                        help='Marker-to-channel mapping for islet classification (default: gcg:2,ins:3,sst:5)')
    parser.add_argument('--gmm-p-cutoff', type=float, default=0.75,
                        help='GMM posterior probability cutoff for marker classification (default: 0.75)')
    parser.add_argument('--ratio-min', type=float, default=1.5,
                        help='Dominant marker must be >= ratio_min * runner-up (default: 1.5)')
    parser.add_argument('--marker-top-pct', type=float, default=5,
                        help='For percentile-method channels, top N%% as positive (default: 5)')
    parser.add_argument('--marker-pct-channels', type=str, default='sst',
                        help='Comma-separated marker names using percentile thresholding (default: sst)')

    args = parser.parse_args()
    setup_logging(level="INFO")

    output_dir = Path(args.output_dir)
    tiles_dir = output_dir / "tiles"
    html_dir = Path(args.html_dir) if args.html_dir else output_dir / "html"
    czi_path = Path(args.czi_path)
    slide_name = czi_path.stem

    # Validate prior-annotations path early (before expensive CZI load)
    if args.prior_annotations and not Path(args.prior_annotations).exists():
        logger.error(f"Prior annotations file not found: {args.prior_annotations}")
        sys.exit(1)

    # Auto-detect cell type from detections filename
    cell_type = args.cell_type
    if cell_type is None:
        det_files = list(output_dir.glob("*_detections.json"))
        if det_files:
            # Parse cell type from filename: {cell_type}_detections.json
            cell_type = det_files[0].stem.replace('_detections', '')
            logger.info(f"Auto-detected cell type: {cell_type}")
        else:
            logger.error("No detections JSON found and no --cell-type specified")
            sys.exit(1)

    # Load pipeline config if available
    config_file = output_dir / 'pipeline_config.json'
    pipeline_config = {}
    if config_file.exists():
        with open(config_file) as f:
            pipeline_config = json.load(f)
        logger.info(f"Loaded pipeline config from {config_file}")

    # Resolve display channels
    if args.display_channels:
        display_channels = [int(x.strip()) for x in args.display_channels.split(',')]
    elif 'display_channels' in pipeline_config:
        display_channels = pipeline_config['display_channels']
        logger.info(f"Using display channels from pipeline config: {display_channels}")
    elif cell_type == 'islet':
        display_channels = [2, 3, 5]
    elif cell_type == 'tissue_pattern':
        display_channels = [1, 2, 0]
    else:
        display_channels = [0, 1, 2]

    # Resolve tile size
    tile_size = args.tile_size or pipeline_config.get('tile_size', 4000)

    # Determine channels to load
    if args.channels:
        channels_to_load = [int(x.strip()) for x in args.channels.split(',')]
    else:
        channels_to_load = sorted(set(display_channels))

    # Load detections JSON — explicit path or auto-detect
    if args.detections:
        det_file = Path(args.detections)
        if not det_file.exists():
            logger.error(f"Specified detections file not found: {det_file}")
            sys.exit(1)
    else:
        det_file = output_dir / f"{cell_type}_detections.json"
        multiscale_file = output_dir / f"{cell_type}_detections_multiscale.json"
        if not det_file.exists():
            if multiscale_file.exists():
                det_file = multiscale_file
            else:
                det_files = list(output_dir.glob("*_detections*.json"))
                det_files = [f for f in det_files if 'checkpoint' not in f.name]
                if det_files:
                    det_file = det_files[0]
                else:
                    logger.error(f"No detections JSON found in {output_dir}")
                    sys.exit(1)

    logger.info(f"Loading detections from {det_file}...")
    with open(det_file) as f:
        all_detections = json.load(f)
    logger.info(f"Loaded {len(all_detections):,} detections")

    # Apply vessel quality filter
    if args.vessel_quality_filter and cell_type == 'vessel':
        pre_filter = len(all_detections)
        filtered = []
        for det in all_detections:
            feat = det.get('features', {})
            if feat.get('ring_completeness', 1.0) < 0.30:
                continue
            if feat.get('circularity', 1.0) < 0.15:
                continue
            wt = feat.get('wall_thickness_mean_um')
            if wt is not None and wt < 1.5:
                continue
            filtered.append(det)
        all_detections = filtered
        logger.info(f"Vessel quality filter: {pre_filter} → {len(all_detections)}")

    # Score threshold filter (rf_prediction)
    if args.score_threshold > 0:
        pre_filter = len(all_detections)
        all_detections = [d for d in all_detections
                          if (d.get('rf_prediction') or 0) >= args.score_threshold]
        logger.info(f"Score filter (>= {args.score_threshold}): {pre_filter:,} -> {len(all_detections):,}")

    # Sample if needed
    if args.max_samples > 0 and len(all_detections) > args.max_samples:
        indices = np.random.default_rng(42).choice(
            len(all_detections), args.max_samples, replace=False)
        sampled = [all_detections[i] for i in indices]
        logger.info(f"Sampled {len(sampled):,} / {len(all_detections):,} detections for HTML")
    else:
        sampled = all_detections

    # Decide crop mode: tile+mask (HDF5) vs contour-based (direct CZI crop)
    # Contour mode is used when tile masks aren't available — e.g. multiscale vessel,
    # or any run where masks weren't saved. Works for any cell type.
    has_tile_masks = tiles_dir.exists() and any(tiles_dir.glob(f"tile_*/{cell_type}_masks.h5"))
    use_contour_mode = not has_tile_masks
    tile_groups = {}
    if use_contour_mode:
        logger.info("No per-tile HDF5 masks found — using contour-based cropping from CZI")
    else:
        # Group sampled detections by tile for tile+mask mode
        tile_groups = defaultdict(list)
        for det in sampled:
            to = det.get('tile_origin')
            if to is None:
                continue
            tile_key = f"tile_{to[0]}_{to[1]}"
            tile_groups[tile_key].append(det)
        logger.info(f"{len(sampled):,} detections across {len(tile_groups)} tiles")

    # Load CZI channels to RAM
    logger.info(f"Loading CZI channels {channels_to_load} to RAM...")
    loader = get_loader(czi_path, load_to_ram=True, channel=channels_to_load[0],
                        scene=args.scene)
    x_start = loader.x_start
    y_start = loader.y_start
    mosaic_w = loader.width
    mosaic_h = loader.height
    pixel_size_um = args.pixel_size or loader.get_pixel_size()

    channel_arrays = {}
    channel_arrays[channels_to_load[0]] = loader.channel_data
    logger.info(f"  Channel {channels_to_load[0]} loaded: {loader.channel_data.nbytes / 1e9:.1f} GB")

    for ch in channels_to_load[1:]:
        ch_loader = get_loader(czi_path, load_to_ram=True, channel=ch, scene=args.scene)
        channel_arrays[ch] = ch_loader.channel_data
        logger.info(f"  Channel {ch} loaded: {ch_loader.channel_data.nbytes / 1e9:.1f} GB")

    # Islet marker thresholds
    marker_thresholds = None
    marker_map = None
    if cell_type == 'islet':
        marker_map = {}
        for pair in args.islet_marker_channels.split(','):
            name, ch = pair.strip().split(':')
            marker_map[name.strip()] = int(ch.strip())

        _pct_channels = set(s.strip() for s in args.marker_pct_channels.split(',')) if args.marker_pct_channels else set()
        marker_thresholds = compute_islet_marker_thresholds(
            all_detections,
            marker_map=marker_map,
            marker_top_pct=args.marker_top_pct,
            pct_channels=_pct_channels,
            gmm_p_cutoff=args.gmm_p_cutoff,
            ratio_min=args.ratio_min,
        ) if all_detections else None

        if marker_thresholds:
            counts = {}
            for det in all_detections:
                mc, _ = classify_islet_marker(
                    det.get('features', {}), marker_thresholds, marker_map=marker_map)
                det['marker_class'] = mc
                counts[mc] = counts.get(mc, 0) + 1
            logger.info(f"Islet marker classification: {counts}")

    # Channel legend from CZI metadata
    channel_legend = None
    try:
        meta = get_czi_metadata(czi_path, scene=args.scene)

        def _ch_label(idx):
            for ch in meta['channels']:
                if ch['index'] == idx:
                    em = f" ({ch['emission_nm']:.0f}nm)" if ch.get('emission_nm') else ''
                    return f"{ch['name']}{em}"
            return f'Ch{idx}'

        channel_legend = {
            'red': _ch_label(display_channels[0]) if len(display_channels) > 0 else 'none',
            'green': _ch_label(display_channels[1]) if len(display_channels) > 1 else 'none',
            'blue': _ch_label(display_channels[2]) if len(display_channels) > 2 else 'none',
        }
        logger.info(f"Channel legend: {channel_legend}")
    except Exception as e:
        logger.warning(f"Could not extract channel legend: {e}")

    # Generate samples — two modes
    logger.info(f"Generating HTML crops for {len(sampled):,} detections...")
    all_samples = []
    tiles_processed = 0

    from tqdm import tqdm

    if use_contour_mode:
        # ---- Contour mode: crop directly from CZI around each detection ----
        # Parallelize with ThreadPool (I/O + numpy, releases GIL)
        from concurrent.futures import ThreadPoolExecutor
        import os
        n_workers = min(int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)), 32)
        logger.info(f"Using {n_workers} threads for contour crop generation")

        def _make_sample(det):
            return create_sample_from_contours(
                det, channel_arrays, display_channels,
                x_start, y_start, mosaic_h, mosaic_w,
                pixel_size_um, slide_name, cell_type,
                contour_thickness=args.contour_thickness,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(tqdm(pool.map(_make_sample, sampled),
                                total=len(sampled), desc="Detections"))
        all_samples = [s for s in results if s is not None]
        tiles_processed = len(sampled)  # each detection is its own "tile"

    else:
        # ---- Tile+mask mode: group by tile, load HDF5 masks ----
        for tile_key, tile_dets in tqdm(tile_groups.items(), desc="Tiles"):
            tile_dir = tiles_dir / tile_key
            mask_file = tile_dir / f"{cell_type}_masks.h5"

            if not mask_file.exists():
                logger.warning(f"No mask file for {tile_key}, skipping {len(tile_dets)} detections")
                continue

            # Load masks
            with h5py.File(mask_file, 'r') as hf:
                if 'masks' in hf:
                    masks = hf['masks'][:]
                elif 'labels' in hf:
                    masks = hf['labels'][:]
                else:
                    logger.warning(f"No masks dataset in {mask_file}")
                    continue
                if masks.ndim == 3 and masks.shape[0] == 1:
                    masks = masks[0]

            # Parse tile origin from first detection
            tile_origin = tile_dets[0].get('tile_origin', [0, 0])
            tile_x, tile_y = tile_origin[0], tile_origin[1]

            # Extract tile RGB directly from channel arrays
            rel_tx = tile_x - x_start
            rel_ty = tile_y - y_start
            tile_h, tile_w = masks.shape[:2]
            rgb_channels = []
            for ch_idx in display_channels[:3]:
                if ch_idx in channel_arrays:
                    rgb_channels.append(
                        channel_arrays[ch_idx][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w])
                else:
                    _dtype = next(iter(channel_arrays.values())).dtype
                    rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_dtype))
            if not rgb_channels:
                continue
            tile_rgb = np.stack(rgb_channels, axis=-1)

            # Generate crop for each detection in this tile
            for det in tile_dets:
                sample = create_sample(
                    tile_rgb, masks, det, pixel_size_um, slide_name, cell_type,
                    marker_thresholds=marker_thresholds,
                    marker_map=marker_map,
                    contour_thickness=args.contour_thickness,
                )
                if sample:
                    all_samples.append(sample)

            tiles_processed += 1
            del masks, tile_rgb
            if tiles_processed % 100 == 0:
                gc.collect()

    logger.info(f"Generated {len(all_samples):,} HTML samples from {tiles_processed} tiles")

    # Sort
    sort_key = args.sort_by
    reverse = (args.sort_order == 'desc')
    if sort_key == 'rf_prediction':
        all_samples.sort(key=lambda x: x['stats'].get('rf_prediction') or 0, reverse=reverse)
    elif sort_key == 'area':
        all_samples.sort(key=lambda x: x['stats'].get('area_um2', 0), reverse=reverse)
    elif sort_key == 'elongation':
        all_samples.sort(key=lambda x: x['stats'].get('elongation', 0), reverse=reverse)
    elif sort_key == 'confidence':
        all_samples.sort(key=lambda x: x['stats'].get('confidence', 0), reverse=reverse)
    else:
        all_samples.sort(key=lambda x: x['stats'].get(sort_key, 0), reverse=reverse)

    # Export HTML
    experiment_name = f"{slide_name}_regen"

    export_samples_to_html(
        all_samples,
        html_dir,
        cell_type,
        samples_per_page=args.samples_per_page,
        title=f"{cell_type.upper()} Annotation Review (regenerated)",
        page_prefix=f'{cell_type}_page',
        experiment_name=experiment_name,
        file_name=f"{slide_name}.czi",
        pixel_size_um=pixel_size_um,
        tiles_processed=tiles_processed,
        tiles_total=len(tile_groups) if tile_groups else len(sampled),
        channel_legend=channel_legend,
        prior_annotations=args.prior_annotations,
    )

    logger.info(f"HTML exported to {html_dir}")
    logger.info(f"  {len(all_samples):,} samples, "
                f"{(len(all_samples) + args.samples_per_page - 1) // args.samples_per_page} pages")


if __name__ == '__main__':
    main()
