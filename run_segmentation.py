#!/usr/bin/env python3
"""
Unified Cell Segmentation Pipeline

A general-purpose pipeline for detecting and classifying cells in CZI microscopy images.
Supports multiple cell types with shared infrastructure and full feature extraction.

Cell Types:
    - nmj: Neuromuscular junctions (intensity + elongation filter)
    - mk: Megakaryocytes (SAM2 automatic mask generation)
    - cell: Hematopoietic stem/progenitor cells (Cellpose-SAM + SAM2 refinement)
    - vessel: Blood vessel cross-sections (ring detection via contour hierarchy)
    - islet: Pancreatic islet cells (Cellpose membrane+nuclear + SAM2)

Features extracted per cell: up to 6478 total (with --extract-deep-features and --all-channels)
    - ~78 morphological features (22 base + NMJ-specific + multi-channel stats)
    - 256 SAM2 embedding features (always)
    - 4096 ResNet-50 features (2x2048 masked+context, opt-in via --extract-deep-features)
    - 2048 DINOv2-L features (2x1024 masked+context, opt-in via --extract-deep-features)
    + Cell-type specific features (elongation for NMJ, wall thickness for vessel, etc.)

Outputs:
    - {cell_type}_detections.json: All detections with universal IDs and global coordinates
    - {cell_type}_coordinates.csv: Quick export with center coordinates in pixels and µm
    - {cell_type}_masks.h5: Per-tile mask arrays
    - html/: Interactive HTML viewer for annotation

After processing, automatically starts HTTP server and Cloudflare tunnel for remote viewing.
Server runs in background by default - script exits but server keeps running.

Server options:
    --serve-background  Start server in background (default) - script exits, server persists
    --serve             Start server in foreground - wait for Ctrl+C to stop
    --no-serve          Don't start server
    --port PORT         HTTP server port (default: 8081)
    --stop-server       Stop any running background server and exit
    --server-status     Show status of running server (including public URL)

Usage:
    # NMJ detection
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj --channel 1

    # Vessel detection (SMA staining)
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --min-vessel-diameter 10 --max-vessel-diameter 500

    # Vessel with CD31 validation
    python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 \\
        --cd31-channel 1
"""

import os
import gc
import re
import sys
import json
import argparse
import subprocess
import signal
import atexit
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py
import torch
# Note: torchvision imports (tv_models, tv_transforms) are loaded lazily inside
# CellDetector strategy classes to avoid loading them when not needed.
# Import segmentation modules
from segmentation.detection.tissue import (
    calibrate_tissue_threshold,
    filter_tissue_tiles,
    has_tissue,
)
from segmentation.io.html_export import (
    export_samples_to_html,
    percentile_normalize,
    draw_mask_contour,
    image_to_base64,
    create_hdf5_dataset,  # Import shared HDF5 utilities
    HDF5_COMPRESSION_KWARGS,
    HDF5_COMPRESSION_NAME,
)
from segmentation.utils.logging import get_logger, setup_logging, log_parameters
from segmentation.io.czi_loader import get_loader, CZILoader, get_czi_metadata, print_czi_metadata
from segmentation.utils.islet_utils import classify_islet_marker, compute_islet_marker_thresholds

# Import new CellDetector and strategies
from segmentation.detection.cell_detector import CellDetector
from segmentation.detection.strategies.mk import MKStrategy
from segmentation.detection.strategies.nmj import NMJStrategy
from segmentation.detection.strategies.vessel import VesselStrategy
from segmentation.detection.strategies.cell import CellStrategy
from segmentation.detection.strategies.mesothelium import MesotheliumStrategy
from segmentation.detection.strategies.islet import IsletStrategy
from segmentation.detection.strategies.tissue_pattern import TissuePatternStrategy

# Import vessel classifier for ML-based classification
from segmentation.classification.vessel_classifier import VesselClassifier, classify_vessel
from segmentation.classification.vessel_type_classifier import VesselTypeClassifier

logger = get_logger(__name__)


from segmentation.utils.json_utils import NumpyEncoder


def detect_resume_stage(slide_output_dir, cell_type):
    """Detect which pipeline stages have completed in an existing run directory.

    Returns dict with:
        has_tiles: bool - tile directories with masks/features exist
        tile_count: int - number of tile directories found
        has_detections: bool - deduped detections JSON exists
        detection_count: int - number of detections in JSON (0 if no file)
        has_html: bool - html/index.html exists
    """
    slide_output_dir = Path(slide_output_dir)
    tiles_dir = slide_output_dir / "tiles"
    det_file = slide_output_dir / f"{cell_type}_detections.json"
    html_index = slide_output_dir / "html" / "index.html"

    # Count tile directories with valid outputs
    tile_count = 0
    if tiles_dir.exists():
        for td in tiles_dir.iterdir():
            if td.is_dir() and td.name.startswith("tile_"):
                mask_file = td / f"{cell_type}_masks.h5"
                feat_file = td / f"{cell_type}_features.json"
                if mask_file.exists() and feat_file.exists():
                    tile_count += 1

    # Check detections JSON
    detection_count = 0
    has_detections = det_file.exists()
    if has_detections:
        try:
            with open(det_file) as f:
                dets = json.load(f)
            detection_count = len(dets)
        except (json.JSONDecodeError, IOError):
            has_detections = False

    return {
        'has_tiles': tile_count > 0,
        'tile_count': tile_count,
        'has_detections': has_detections,
        'detection_count': detection_count,
        'has_html': html_index.exists(),
    }


def reload_detections_from_tiles(tiles_dir, cell_type):
    """Reload all detections from per-tile feature JSON files.

    Iterates tile_*/ directories, loads {cell_type}_features.json from each,
    and returns the merged list (same format as all_detections in pipeline).
    """
    tiles_dir = Path(tiles_dir)
    all_detections = []

    tile_dirs = sorted(
        [d for d in tiles_dir.iterdir() if d.is_dir() and d.name.startswith("tile_")],
        key=lambda d: d.name,
    )

    for td in tile_dirs:
        feat_file = td / f"{cell_type}_features.json"
        if feat_file.exists():
            try:
                with open(feat_file) as f:
                    features_list = json.load(f)
                all_detections.extend(features_list)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {feat_file}: {e}")

    return all_detections


def _resume_generate_html_samples(args, all_detections, tiles_dir,
                                  all_channel_data, loader, pixel_size_um,
                                  slide_name, x_start, y_start):
    """Generate HTML samples from saved tile masks + CZI data (resume path).

    Groups detections by tile_origin, loads masks from HDF5, composes tile RGB
    from CZI channels, and creates HTML crops — same output as the normal path.
    """
    from collections import defaultdict

    cell_type = args.cell_type

    # Determine display channels
    if cell_type == 'islet':
        display_chs = getattr(args, 'islet_display_chs', [2, 3, 5])
    elif cell_type == 'tissue_pattern':
        display_chs = getattr(args, 'tp_display_channels_list', [0, 3, 1])
    else:
        display_chs = sorted(all_channel_data.keys())[:3]

    # Group detections by tile
    tile_groups = defaultdict(list)
    for det in all_detections:
        to = det.get('tile_origin')
        if to is None:
            continue
        tile_key = f"tile_{to[0]}_{to[1]}"
        tile_groups[tile_key].append(det)

    # Islet marker thresholds (population-level, needed before generating crops)
    marker_thresholds = None
    _islet_mm = None
    if cell_type == 'islet':
        _islet_mm = getattr(args, 'islet_marker_map', None)
        _marker_top_pct = getattr(args, 'marker_top_pct', 5)
        _pct_chs_str = getattr(args, 'marker_pct_channels', 'sst')
        _pct_channels = set(s.strip() for s in _pct_chs_str.split(',')) if _pct_chs_str else set()
        _gmm_p = getattr(args, 'gmm_p_cutoff', 0.75)
        _ratio_min = getattr(args, 'ratio_min', 1.5)
        marker_thresholds = compute_islet_marker_thresholds(
            all_detections, marker_map=_islet_mm,
            marker_top_pct=_marker_top_pct,
            pct_channels=_pct_channels,
            gmm_p_cutoff=_gmm_p,
            ratio_min=_ratio_min) if all_detections else None
        if marker_thresholds:
            counts = {}
            for det in all_detections:
                mc, _ = classify_islet_marker(
                    det.get('features', {}), marker_thresholds, marker_map=_islet_mm)
                det['marker_class'] = mc
                counts[mc] = counts.get(mc, 0) + 1
            logger.info(f"Islet marker classification (resume): {counts}")

    # Sample if max_html_samples set
    _max_html = getattr(args, 'max_html_samples', 0)

    # Process tiles
    all_samples = []
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    from tqdm import tqdm as tqdm_progress
    for tile_key, tile_dets in tqdm_progress(tile_groups.items(), desc="Resume HTML"):
        # Check max_html cap BEFORE expensive I/O
        if _max_html > 0 and len(all_samples) >= _max_html:
            break

        tile_dir = tiles_dir / tile_key
        mask_file = tile_dir / f"{cell_type}_masks.h5"

        if not mask_file.exists():
            continue
        with h5py.File(mask_file, 'r') as hf:
            if 'masks' in hf:
                masks = hf['masks'][:]
            elif 'labels' in hf:
                masks = hf['labels'][:]
            else:
                continue
            if masks.ndim == 3 and masks.shape[0] == 1:
                masks = masks[0]

        # Get tile origin
        tile_origin = tile_dets[0].get('tile_origin', [0, 0])
        tile_x, tile_y = tile_origin[0], tile_origin[1]

        # Compose tile RGB — match normal path: direct channel extraction, /256 for uint16
        rel_tx = tile_x - x_start
        rel_ty = tile_y - y_start
        tile_h, tile_w = masks.shape[:2]
        ch_keys = sorted(all_channel_data.keys())
        n_channels = len(ch_keys)

        if cell_type in ('islet', 'tissue_pattern'):
            # Use configured display channels (keep uint16 for low-signal fluorescence)
            rgb_channels = []
            for ch_idx in display_chs[:3]:
                if ch_idx in all_channel_data:
                    rgb_channels.append(
                        all_channel_data[ch_idx][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w])
                else:
                    if not all_channel_data:
                        raise ValueError("No channel data loaded — check --channel and CZI file")
                    _dtype = next(iter(all_channel_data.values())).dtype
                    rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_dtype))
            tile_rgb = np.stack(rgb_channels, axis=-1)
        elif n_channels >= 3:
            # Standard 3-channel: first 3 loaded channels → R,G,B
            tile_rgb = np.stack([
                all_channel_data[ch_keys[i]][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w]
                for i in range(3)
            ], axis=-1)
        else:
            # Single channel: duplicate to RGB
            tile_rgb = np.stack([
                all_channel_data[ch_keys[0]][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w]
            ] * 3, axis=-1)

        if tile_rgb.size == 0:
            continue

        # Keep uint16 for percentile_normalize — it handles float32 conversion
        # internally with full 16-bit precision. The old /256 uint8 conversion
        # caused visible banding/blur after flat-field correction.

        tile_pct = _compute_tile_percentiles(tile_rgb) if getattr(args, 'html_normalization', 'crop') == 'tile' else None

        # Generate crops for each detection in this tile
        _vp = {
            'min_ring_completeness': getattr(args, 'min_ring_completeness', 0.3),
            'min_circularity': getattr(args, 'min_circularity', 0.15),
            'min_wall_thickness_um': getattr(args, 'min_wall_thickness', 1.5),
        } if cell_type == 'vessel' else None
        html_samples = filter_and_create_html_samples(
            tile_dets, tile_x, tile_y, tile_rgb, masks,
            pixel_size_um, slide_name, cell_type,
            args.html_score_threshold,
            tile_percentiles=tile_pct,
            marker_thresholds=marker_thresholds,
            marker_map=_islet_mm,
            candidate_mode=getattr(args, 'candidate_mode', False),
            vessel_params=_vp,
        )
        all_samples.extend(html_samples)

        del masks, tile_rgb
        gc.collect()

    return all_samples


def _build_channel_legend(cell_type, args, czi_path, slide_name=None):
    """Build channel legend dict from CZI metadata for HTML export.

    Args:
        cell_type: Detection cell type string
        args: Parsed arguments (for display channel config)
        czi_path: Path to CZI file (for metadata extraction)
        slide_name: Slide name (fallback for NMJ filename parsing)

    Returns:
        Dict with 'red', 'green', 'blue' keys, or None on failure.
    """
    try:
        _czi_meta = get_czi_metadata(czi_path, scene=getattr(args, 'scene', 0))

        def _channel_label(ch_idx):
            for ch in _czi_meta['channels']:
                if ch['index'] == ch_idx:
                    name = ch['name']
                    em = f" ({ch['emission_nm']:.0f}nm)" if ch.get('emission_nm') else ''
                    return f'{name}{em}'
            return f'Ch{ch_idx}'

        if cell_type == 'islet':
            _islet_disp = getattr(args, 'islet_display_chs', [2, 3, 5])
            return {
                'red': _channel_label(_islet_disp[0]) if len(_islet_disp) > 0 else 'none',
                'green': _channel_label(_islet_disp[1]) if len(_islet_disp) > 1 else 'none',
                'blue': _channel_label(_islet_disp[2]) if len(_islet_disp) > 2 else 'none',
            }
        elif cell_type == 'tissue_pattern':
            tp_disp = getattr(args, 'tp_display_channels_list', [0, 3, 1])
            return {
                'red': _channel_label(tp_disp[0]) if len(tp_disp) > 0 else 'none',
                'green': _channel_label(tp_disp[1]) if len(tp_disp) > 1 else 'none',
                'blue': _channel_label(tp_disp[2]) if len(tp_disp) > 2 else 'none',
            }
        elif cell_type == 'nmj' and getattr(args, 'all_channels', False):
            try:
                return {
                    'red': _channel_label(0),
                    'green': _channel_label(1),
                    'blue': _channel_label(2),
                }
            except Exception:
                return parse_channel_legend_from_filename(slide_name) if slide_name else None
        else:
            return {
                'red': _channel_label(0),
                'green': _channel_label(1),
                'blue': _channel_label(2),
            }
    except Exception:
        return None


def _finish_pipeline(args, all_detections, all_samples, slide_output_dir, tiles_dir,
                     pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
                     skip_html=False, all_tiles=None, tissue_tiles=None, sampled_tiles=None,
                     resumed=False, params=None, classifier_loaded=False,
                     is_multiscale=False, detector=None):
    """Shared post-processing: HTML export, CSV, summary, server (used by both normal and resume paths).

    Args:
        resumed: Whether this is a resumed pipeline run (affects title/summary).
        params: Detection parameters dict (for summary, normal path only).
        classifier_loaded: Whether a classifier was loaded (for sort order, normal path).
        is_multiscale: Whether multiscale mode was used (for checkpoint cleanup).
        detector: CellDetector instance to cleanup (normal path only).
    """
    cell_type = args.cell_type
    czi_path = Path(args.czi_path)

    # ---- Save detections JSON + CSV FIRST (before HTML) ----
    # This ensures dedup results are persisted even if HTML generation crashes/hangs.
    # On resume, the pipeline will find the detections JSON and skip detection+dedup.
    for det in all_detections:
        det['tile_mask_label'] = det.get('mask_label', 0)
        _gc = det.get('global_center', det.get('center', [0, 0]))
        det['global_id'] = f"{int(round(_gc[0]))}_{int(round(_gc[1]))}"

    detections_file = slide_output_dir / f'{cell_type}_detections.json'
    from segmentation.utils.timestamps import timestamped_path, update_symlink
    ts_detections = timestamped_path(detections_file)
    with open(ts_detections, 'w') as f:
        json.dump(all_detections, f, cls=NumpyEncoder)
    update_symlink(detections_file, ts_detections)
    logger.info(f"Saved {len(all_detections)} detections to {ts_detections}")

    csv_file = slide_output_dir / f'{cell_type}_coordinates.csv'
    ts_csv = timestamped_path(csv_file)
    with open(ts_csv, 'w') as f:
        if cell_type == 'vessel':
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,outer_diameter_um,wall_thickness_um,confidence\n')
            for det in all_detections:
                g_center = det.get('global_center')
                g_center_um = det.get('global_center_um')
                if g_center is None or g_center_um is None:
                    continue
                if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                    continue
                if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                    continue
                feat = det.get('features', {})
                f.write(f"{det['uid']},{g_center[0]:.1f},{g_center[1]:.1f},"
                        f"{g_center_um[0]:.2f},{g_center_um[1]:.2f},"
                        f"{feat.get('outer_diameter_um', 0):.2f},{feat.get('wall_thickness_mean_um', 0):.2f},"
                        f"{feat.get('confidence', 'unknown')}\n")
        else:
            f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,area_um2\n')
            for det in all_detections:
                g_center = det.get('global_center')
                g_center_um = det.get('global_center_um')
                if g_center is None or g_center_um is None:
                    continue
                if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                    continue
                if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                    continue
                feat = det.get('features', {})
                area_um2 = feat.get('area', 0) * (pixel_size_um ** 2)
                f.write(f"{det['uid']},{g_center[0]:.1f},{g_center[1]:.1f},"
                        f"{g_center_um[0]:.2f},{g_center_um[1]:.2f},{area_um2:.2f}\n")
    update_symlink(csv_file, ts_csv)
    logger.info(f"Saved coordinates to {ts_csv}")

    # Sort samples: classifier runs → RF score descending; else → area ascending
    _has_classifier = classifier_loaded or (
        (cell_type == 'nmj' and getattr(args, 'nmj_classifier', None)) or
        (cell_type == 'islet' and getattr(args, 'islet_classifier', None)) or
        (cell_type == 'tissue_pattern' and getattr(args, 'tp_classifier', None))
    )
    if _has_classifier:
        all_samples.sort(key=lambda x: x['stats'].get('rf_prediction') or 0, reverse=True)
    else:
        all_samples.sort(key=lambda x: x['stats'].get('area_um2', 0))

    # Export to HTML (unless skipped)
    if not skip_html and all_samples:
        if cell_type in ('nmj', 'islet', 'tissue_pattern') and len(all_detections) > len(all_samples):
            logger.info(f"Total detections (all scores): {len(all_detections)}, "
                         f"shown in HTML (rf_prediction >= {args.html_score_threshold}): {len(all_samples)}")
        logger.info(f"Exporting to HTML ({len(all_samples)} samples)...")
        html_dir = slide_output_dir / "html"

        channel_legend = _build_channel_legend(cell_type, args, czi_path, slide_name=slide_name)

        prior_ann = getattr(args, 'prior_annotations', None)
        experiment_name = f"{slide_name}_{run_timestamp}_{pct}pct"
        _title_suffix = " (resumed)" if resumed else ""
        export_samples_to_html(
            all_samples,
            html_dir,
            cell_type,
            samples_per_page=args.samples_per_page,
            title=f"{cell_type.upper()} Annotation Review{_title_suffix}",
            page_prefix=f'{cell_type}_page',
            experiment_name=experiment_name,
            file_name=f"{czi_path.name}" if resumed else f"{slide_name}.czi",
            pixel_size_um=pixel_size_um,
            tiles_processed=len(sampled_tiles) if sampled_tiles else 0,
            tiles_total=len(all_tiles) if all_tiles else 0,
            channel_legend=channel_legend,
            prior_annotations=prior_ann,
        )
    elif skip_html:
        logger.info("HTML export skipped (already exists)")

    # Clean up multiscale checkpoints after successful completion
    if is_multiscale:
        checkpoint_dir = slide_output_dir / "checkpoints"
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info("Multiscale checkpoints cleaned up after successful completion")

    # Save summary
    summary = {
        'slide_name': slide_name,
        'cell_type': cell_type,
        'pixel_size_um': pixel_size_um,
        'mosaic_width': mosaic_info['width'],
        'mosaic_height': mosaic_info['height'],
        'total_tiles': len(all_tiles) if all_tiles else 0,
        'tissue_tiles': len(tissue_tiles) if tissue_tiles else 0,
        'sampled_tiles': len(sampled_tiles) if sampled_tiles else 0,
        'total_detections': len(all_detections),
        'html_displayed': len(all_samples),
        'resumed': resumed,
        'params': params if params else {},
        'detections_file': str(detections_file),
        'coordinates_file': str(csv_file),
    }
    with open(slide_output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, cls=NumpyEncoder)

    # Cleanup detector resources
    if detector is not None:
        detector.cleanup()

    _status_label = "COMPLETE (resumed)" if resumed else "COMPLETE"
    logger.info("=" * 60)
    logger.info(_status_label)
    logger.info("=" * 60)
    logger.info(f"Total detections: {len(all_detections)}")
    logger.info(f"Displayed in HTML: {len(all_samples)} (score >= {args.html_score_threshold})")
    logger.info(f"Output: {slide_output_dir}")
    html_dir = slide_output_dir / "html"
    if html_dir.exists():
        logger.info(f"HTML viewer: {html_dir / 'index.html'}")

    # Start HTTP server
    no_serve = getattr(args, 'no_serve', False)
    serve_foreground = getattr(args, 'serve', False)
    serve_background = getattr(args, 'serve_background', True)
    port = getattr(args, 'port', 8081)

    if no_serve:
        logger.info("Server disabled (--no-serve)")
    elif html_dir.exists() and not skip_html:
        if serve_foreground:
            http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(
                html_dir, port, background=False,
                slide_name=slide_name, cell_type=cell_type)
            if http_proc is not None:
                wait_for_server_shutdown(http_proc, tunnel_proc)
        elif serve_background:
            start_server_and_tunnel(
                html_dir, port, background=True,
                slide_name=slide_name, cell_type=cell_type)
            print("")
            show_server_status()


# Global list to track spawned processes for cleanup (foreground mode only)
_spawned_processes = []

# PID file directory for background servers (one file per port)
SERVER_PID_DIR = Path.home() / '.segmentation_servers'
# Legacy single PID file (for backwards compatibility)
SERVER_PID_FILE = Path.home() / '.segmentation_server.pid'


def parse_channel_legend_from_filename(filename: str) -> dict:
    """
    Parse channel information from filename to create legend.

    Looks for patterns like:
    - nuc488, nuc405 -> nuclear stain (keeps original like 'nuc488')
    - Bgtx647, BTX647 -> bungarotoxin (keeps original)
    - NfL750, NFL750 -> neurofilament (keeps original)
    - DAPI -> nuclear
    - SMA -> smooth muscle actin
    - _647_ -> standalone wavelength

    Args:
        filename: Slide filename (e.g., '20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch')

    Returns:
        Dict with 'red', 'green', 'blue' keys mapping to channel names,
        or None if no channels detected.
    """
    channels = []

    # Specific channel patterns - use original text from filename
    # Order: patterns that include wavelength first, then standalone wavelengths
    patterns = [
        # Patterns with wavelength embedded (capture the whole thing)
        r'nuc\d{3}',           # nuc488, nuc405
        r'bgtx\d{3}',          # Bgtx647
        r'btx\d{3}',           # BTX647
        r'nfl?\d{3}',          # NfL750, NFL750
        r'sma\d*',             # SMA, SMA488
        r'cd\d+',              # CD31, CD34
        # Named stains without wavelength
        r'dapi',
        r'bungarotoxin',
        r'neurofilament',
        # Standalone wavelengths (must be bounded by _ or - or start/end)
        r'(?:^|[_-])(\d{3})(?:[_-]|$)',  # _647_, -488-
    ]

    # Find all channel mentions with their positions
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, filename, re.IGNORECASE):
            # For grouped patterns, use group(1) if it exists
            if match.lastindex:
                text = match.group(1)
                pos = match.start(1)
            else:
                text = match.group(0)
                pos = match.start()
            found.append((pos, text))

    # Sort by position in filename and deduplicate
    found.sort(key=lambda x: x[0])
    seen = set()
    for pos, name in found:
        name_lower = name.lower()
        if name_lower not in seen:
            channels.append(name)
            seen.add(name_lower)

    if len(channels) >= 3:
        return {
            'red': channels[0],
            'green': channels[1],
            'blue': channels[2]
        }
    elif len(channels) == 2:
        return {
            'red': channels[0],
            'green': channels[1]
        }
    elif len(channels) == 1:
        return {
            'green': channels[0]  # Single channel shown as green
        }

    return None


def _get_pid_file(port: int) -> Path:
    """Get PID file path for a specific port."""
    SERVER_PID_DIR.mkdir(exist_ok=True)
    return SERVER_PID_DIR / f'server_{port}.json'


def _get_all_servers() -> list:
    """Get list of all server info dicts."""
    servers = []

    # Check legacy single PID file first
    if SERVER_PID_FILE.exists():
        try:
            data = json.loads(SERVER_PID_FILE.read_text())
            data['_pid_file'] = SERVER_PID_FILE
            servers.append(data)
        except Exception:
            pass

    # Check new per-port PID files
    if SERVER_PID_DIR.exists():
        for pid_file in SERVER_PID_DIR.glob('server_*.json'):
            try:
                data = json.loads(pid_file.read_text())
                data['_pid_file'] = pid_file
                # Skip if already covered by legacy file (same port)
                if not any(s.get('port') == data.get('port') for s in servers):
                    servers.append(data)
            except Exception:
                pass

    return servers


def _cleanup_processes():
    """Cleanup spawned HTTP server and tunnel processes on exit."""
    for proc in _spawned_processes:
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            pass


atexit.register(_cleanup_processes)


def stop_background_server():
    """Stop all running background servers using _get_all_servers()."""
    servers = _get_all_servers()

    if not servers:
        logger.info("No background server running (no PID files found)")
        return False

    stopped = False
    for data in servers:
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        pid_file = data.get('_pid_file')

        for name, pid in [('HTTP server', http_pid), ('Cloudflare tunnel', tunnel_pid)]:
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Stopped {name} (PID {pid})")
                    stopped = True
                except ProcessLookupError:
                    logger.info(f"{name} (PID {pid}) was not running")
                except PermissionError:
                    logger.error(f"Permission denied stopping {name} (PID {pid})")

        # Clean up PID file
        if pid_file and pid_file.exists():
            try:
                pid_file.unlink()
            except OSError:
                pass

    return stopped


def show_server_status():
    """Show status of all running background servers."""
    servers = _get_all_servers()

    if not servers:
        print("No background servers running")
        return False

    # Filter to only running servers
    running_servers = []
    for data in servers:
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        http_running = http_pid and _pid_exists(http_pid)
        tunnel_running = tunnel_pid and _pid_exists(tunnel_pid)

        if http_running or tunnel_running:
            data['_http_running'] = http_running
            data['_tunnel_running'] = tunnel_running
            running_servers.append(data)

    if not running_servers:
        print("No background servers running")
        return False

    print("=" * 70)
    print(f"ACTIVE SEGMENTATION SERVERS ({len(running_servers)} running)")
    print("=" * 70)

    for i, data in enumerate(running_servers):
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        url = data.get('url')
        port = data.get('port', 8081)
        slide_name = data.get('slide_name', 'unknown')
        cell_type = data.get('cell_type', 'unknown')
        http_running = data.get('_http_running', False)
        tunnel_running = data.get('_tunnel_running', False)

        # Build human-readable name
        if slide_name and slide_name != 'unknown' and cell_type and cell_type != 'unknown':
            serving_name = f"{slide_name} ({cell_type.upper()})"
        elif slide_name and slide_name != 'unknown':
            serving_name = slide_name
        else:
            serving_name = f"Server on port {port}"

        if i > 0:
            print("-" * 70)

        print(f"\n[{i+1}] {serving_name}")
        print(f"    Slide:      {slide_name}")
        print(f"    Cell Type:  {cell_type}")
        print(f"    Port:       {port}")
        print(f"    Status:     HTTP={'OK' if http_running else 'DOWN'}, Tunnel={'OK' if tunnel_running else 'DOWN'}")
        if url and tunnel_running:
            print(f"    PUBLIC:     {url}")
        print(f"    LOCAL:      http://localhost:{port}")

    print("\n" + "=" * 70)
    print(f"To stop all: python run_segmentation.py --stop-server")
    print("=" * 70)
    return True


def _pid_exists(pid):
    """Check if a process with given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def start_server_and_tunnel(html_dir: Path, port: int = 8081, background: bool = False,
                            slide_name: str = None, cell_type: str = None) -> tuple:
    """
    Start HTTP server and Cloudflare tunnel for viewing results.

    Args:
        html_dir: Path to the HTML directory to serve
        port: Port for HTTP server (default 8081)
        background: If True, detach processes so they survive script exit
        slide_name: Name of the slide being served (for status display)
        cell_type: Type of cells being detected (for status display)

    Returns:
        Tuple of (http_process, tunnel_process, tunnel_url)
    """
    global _spawned_processes

    html_dir = Path(html_dir)
    if not html_dir.exists():
        logger.error(f"HTML directory does not exist: {html_dir}")
        return None, None, None

    # Check for existing server - reuse tunnel if possible
    existing_tunnel_url = None
    existing_tunnel_pid = None
    if background and SERVER_PID_FILE.exists():
        try:
            data = json.loads(SERVER_PID_FILE.read_text())
            old_http_pid = data.get('http_pid')
            old_tunnel_pid = data.get('tunnel_pid')
            old_port = data.get('port', 8081)
            existing_tunnel_url = data.get('url')

            # Check if tunnel is still running
            tunnel_running = old_tunnel_pid and _pid_exists(old_tunnel_pid)

            if tunnel_running and old_port == port:
                # Tunnel is running on same port - keep it, just restart HTTP server
                logger.info(f"Reusing existing Cloudflare tunnel: {existing_tunnel_url}")
                existing_tunnel_pid = old_tunnel_pid

                # Stop only the HTTP server (not the tunnel)
                if old_http_pid and _pid_exists(old_http_pid):
                    try:
                        os.kill(old_http_pid, signal.SIGTERM)
                        logger.info(f"Stopped old HTTP server (PID {old_http_pid})")
                        time.sleep(0.5)  # Give it time to release the port
                    except Exception:
                        pass
            else:
                # Tunnel not running or different port - start fresh
                logger.info("Starting new server and tunnel...")
                stop_background_server()
                existing_tunnel_url = None
        except Exception:
            stop_background_server()

    # Common args for background mode (detach from parent)
    bg_kwargs = {}
    if background:
        bg_kwargs = {
            'start_new_session': True,  # Detach from parent process group
            'stdin': subprocess.DEVNULL,
        }

    # Start HTTP server
    logger.info(f"Starting HTTP server on port {port}...")
    http_proc = subprocess.Popen(
        ['python', '-m', 'http.server', str(port)],
        cwd=str(html_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **bg_kwargs,
    )
    if not background:
        _spawned_processes.append(http_proc)
    time.sleep(1)  # Give server time to start

    if http_proc.poll() is not None:
        logger.error("HTTP server failed to start")
        return None, None, None

    logger.info(f"HTTP server running: http://localhost:{port}")

    # Start Cloudflare tunnel (or reuse existing one)
    tunnel_proc = None
    tunnel_url = existing_tunnel_url

    if existing_tunnel_pid:
        # Reusing existing tunnel - create a dummy process reference
        logger.info(f"Tunnel already running (PID {existing_tunnel_pid})")
        tunnel_proc = None  # We don't have the process object, just the PID

        # Update PID file with new HTTP server but keep tunnel info
        SERVER_PID_FILE.write_text(json.dumps({
            'http_pid': http_proc.pid,
            'tunnel_pid': existing_tunnel_pid,
            'port': port,
            'html_dir': str(html_dir),
            'url': existing_tunnel_url,
            'slide_name': slide_name,
            'cell_type': cell_type,
        }))
    else:
        # Need to start a new tunnel
        cloudflared_path = os.path.expanduser('~/cloudflared')
        if not os.path.exists(cloudflared_path):
            logger.warning("Cloudflare tunnel not found at ~/cloudflared")
            logger.info("Install with: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared && chmod +x ~/cloudflared")
            if background:
                # Save HTTP server PID even without tunnel
                SERVER_PID_FILE.write_text(json.dumps({
                    'http_pid': http_proc.pid,
                    'tunnel_pid': None,
                    'port': port,
                    'html_dir': str(html_dir),
                    'url': None,
                    'slide_name': slide_name,
                    'cell_type': cell_type,
                }))
            return http_proc, None, None

        logger.info("Starting Cloudflare tunnel...")

        # For background mode, we need to capture output to get URL but still detach
        if background:
            # Create a log file for tunnel output
            tunnel_log = html_dir / '.tunnel.log'
            tunnel_log_file = open(tunnel_log, 'w')
            try:
                tunnel_proc = subprocess.Popen(
                    [cloudflared_path, 'tunnel', '--url', f'http://localhost:{port}'],
                    stdout=tunnel_log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
                # Wait briefly and parse log for URL
                time.sleep(5)
                tunnel_log_file.flush()
            finally:
                tunnel_log_file.close()
            tunnel_url = None
            try:
                log_content = tunnel_log.read_text()
                match = re.search(r'(https://[^\s]+\.trycloudflare\.com)', log_content)
                if match:
                    tunnel_url = match.group(1)
            except Exception:
                pass
        else:
            tunnel_proc = subprocess.Popen(
                [cloudflared_path, 'tunnel', '--url', f'http://localhost:{port}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _spawned_processes.append(tunnel_proc)

            # Wait for tunnel URL (usually appears within 5-10 seconds)
            tunnel_url = None
            start_time = time.time()
            while time.time() - start_time < 30:
                line = tunnel_proc.stdout.readline()
                if not line:
                    if tunnel_proc.poll() is not None:
                        logger.error("Cloudflare tunnel exited unexpectedly")
                        break
                    continue
                # Look for the tunnel URL in the output
                if 'trycloudflare.com' in line:
                    match = re.search(r'(https://[^\s]+\.trycloudflare\.com)', line)
                    if match:
                        tunnel_url = match.group(1)
                        break

        # Save PID file for background mode
        if background:
            SERVER_PID_FILE.write_text(json.dumps({
                'http_pid': http_proc.pid,
                'tunnel_pid': tunnel_proc.pid if tunnel_proc else None,
                'port': port,
                'html_dir': str(html_dir),
                'url': tunnel_url,
                'slide_name': slide_name,
                'cell_type': cell_type,
            }))

    if tunnel_url:
        logger.info("=" * 60)
        logger.info("REMOTE ACCESS AVAILABLE")
        logger.info("=" * 60)
        logger.info(f"Public URL: {tunnel_url}")
        logger.info(f"Local URL:  http://localhost:{port}")
        if background:
            logger.info("")
            logger.info("Server running in BACKGROUND")
            logger.info(f"To stop: python run_segmentation.py --stop-server")
            logger.info(f"PID file: {SERVER_PID_FILE}")
        else:
            logger.info("")
            logger.info("Press Ctrl+C to stop server and tunnel")
        logger.info("=" * 60)
    else:
        logger.warning("Could not get tunnel URL (tunnel may still be starting)")
        logger.info(f"Local URL: http://localhost:{port}")
        if background:
            logger.info(f"Check tunnel log: {html_dir / '.tunnel.log'}")

    return http_proc, tunnel_proc, tunnel_url


def wait_for_server_shutdown(http_proc, tunnel_proc):
    """Wait for user to press Ctrl+C, then cleanup."""
    if http_proc is None:
        return

    try:
        logger.info("Server running. Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if http_proc.poll() is not None:
                logger.warning("HTTP server stopped unexpectedly")
                break
            if tunnel_proc and tunnel_proc.poll() is not None:
                logger.warning("Tunnel stopped unexpectedly")
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        _cleanup_processes()
        logger.info("Server stopped")


# =============================================================================
# STRATEGY HELPER FUNCTIONS
# =============================================================================

def create_strategy_for_cell_type(cell_type, params, pixel_size_um):
    """
    Create the appropriate detection strategy for a cell type.

    Args:
        cell_type: One of 'nmj', 'mk', 'cell', 'vessel'
        params: Cell-type specific parameters dict
        pixel_size_um: Pixel size in microns

    Returns:
        DetectionStrategy instance

    Raises:
        ValueError: If cell_type is not supported by the new strategy pattern
    """
    from segmentation.processing.strategy_factory import create_strategy
    return create_strategy(
        cell_type=cell_type,
        strategy_params=params,
        extract_deep_features=params.get('extract_deep_features', False),
        extract_sam2_embeddings=params.get('extract_sam2_embeddings', True),
        pixel_size_um=pixel_size_um,
    )


# Import from canonical location (segmentation.processing.tile_processing)
from segmentation.processing.tile_processing import detections_to_features_list, process_single_tile


def apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier):
    """Apply vessel binary + 6-class classifiers to detection features in-place."""
    if vessel_classifier is not None:
        for feat in features_list:
            try:
                vessel_type, confidence = vessel_classifier.predict(feat['features'])
                feat['features']['vessel_type'] = vessel_type
                feat['features']['vessel_type_confidence'] = float(confidence)
                feat['features']['classification_method'] = 'ml'
            except Exception as e:
                vessel_type, confidence = VesselClassifier.rule_based_classify(feat['features'])
                feat['features']['vessel_type'] = vessel_type
                feat['features']['vessel_type_confidence'] = float(confidence)
                feat['features']['classification_method'] = 'rule_based_fallback'
                logger.debug(f"ML classification failed, using rule-based: {e}")

    if vessel_type_classifier is not None:
        for feat in features_list:
            try:
                vessel_type, confidence = vessel_type_classifier.predict(feat['features'])
                probs = vessel_type_classifier.predict_proba(feat['features'])
                feat['features']['vessel_type_6class'] = vessel_type
                feat['features']['vessel_type_6class_confidence'] = float(confidence)
                feat['features']['vessel_type_6class_probabilities'] = {
                    k: float(v) for k, v in probs.items()
                } if probs else {}
                feat['features']['classification_method_6class'] = 'ml_vessel_type_classifier'
            except Exception as e:
                try:
                    vessel_type, confidence = vessel_type_classifier.rule_based_classify(feat['features'])
                    feat['features']['vessel_type_6class'] = vessel_type
                    feat['features']['vessel_type_6class_confidence'] = float(confidence)
                    feat['features']['classification_method_6class'] = 'rule_based_fallback'
                except Exception as e2:
                    logger.debug(f"VesselTypeClassifier failed: {e}, {e2}")


def _compute_tile_percentiles(tile_rgb, p_low=1, p_high=99.5):
    """Compute per-channel percentiles from an entire tile for uniform HTML normalization.

    Returns dict {channel_idx: (low_val, high_val)} suitable for percentile_normalize().
    """
    valid_mask = np.max(tile_rgb, axis=2) > 0
    percentiles = {}
    for ch in range(tile_rgb.shape[2]):
        valid = tile_rgb[:, :, ch][valid_mask]
        if len(valid) > 0:
            percentiles[ch] = (float(np.percentile(valid, p_low)), float(np.percentile(valid, p_high)))
    return percentiles if percentiles else None


# classify_islet_marker() moved to segmentation/utils/islet_utils.py (imported at top)


def calibrate_islet_marker_gmm(pilot_tiles, loader, all_channel_data, slide_shm_arr,
                               ch_to_slot, marker_channels, membrane_channel=1,
                               nuclear_channel=4, tile_size=4000, pixel_size_um=None,
                               nuclei_only=False, mosaic_origin=(0, 0)):
    """Run Cellpose on pilot tiles to calibrate global GMM pre-filter thresholds.

    Similar to calibrate_tissue_threshold(): samples a few tiles to estimate
    slide-level parameters before the main multi-GPU processing loop.

    Note: This is a PRE-FILTER for skipping expensive feature extraction on
    background cells. It uses med+3MAD fallback for low-separation channels
    (intentionally more permissive than the final GMM P=0.75 classification
    in compute_islet_marker_thresholds). No area filtering — matches segment().

    Args:
        pilot_tiles: list of tile dicts with 'x', 'y' keys (~5% of tissue tiles)
        loader: CZI loader (for get_tile fallback)
        all_channel_data: dict of channel_idx → 2D array (or None if using shm)
        slide_shm_arr: shared memory array [n_slots, H, W] (or None)
        ch_to_slot: dict mapping CZI channel → shm slot index
        marker_channels: list of CZI channel indices for markers (e.g. [2, 3, 5])
        membrane_channel: CZI channel for Cellpose cytoplasm input
        nuclear_channel: CZI channel for Cellpose nuclear input
        tile_size: tile dimensions in pixels
        pixel_size_um: pixel size in micrometers

    Returns:
        dict mapping channel_idx → raw GMM threshold (P(signal) = 0.5),
        or empty dict if calibration fails.
    """
    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required — must come from CZI metadata")
    from sklearn.mixture import GaussianMixture

    if not pilot_tiles:
        return {}

    logger.info(f"Calibrating islet marker thresholds on {len(pilot_tiles)} pilot tiles...")

    # Load Cellpose model (lightweight — single GPU, no SAM2 needed)
    try:
        from cellpose.models import CellposeModel
        cellpose_model = CellposeModel(gpu=True, pretrained_model='cpsam')
    except Exception as e:
        logger.warning(f"Failed to load Cellpose for calibration: {e}")
        return {}

    # Helper to read a channel tile from shm or all_channel_data
    def _read_channel(ch_idx, x, y, w, h):
        if slide_shm_arr is not None and ch_to_slot and ch_idx in ch_to_slot:
            slot = ch_to_slot[ch_idx]
            return slide_shm_arr[y:y+h, x:x+w, slot].copy()
        elif all_channel_data and ch_idx in all_channel_data:
            return all_channel_data[ch_idx][y:y+h, x:x+w].copy()
        return None

    # Helper: percentile normalize uint16 → uint8
    def _pct_norm(img):
        if img is None or img.size == 0:
            return np.zeros_like(img, dtype=np.uint8) if img is not None else None
        valid = img[img > 0]
        if len(valid) == 0:
            return np.zeros_like(img, dtype=np.uint8)
        lo, hi = np.percentile(valid, 1), np.percentile(valid, 99.5)
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
        return np.clip(255 * (img.astype(np.float32) - lo) / (hi - lo), 0, 255).astype(np.uint8)

    # Collect marker means from all pilot tiles
    all_marker_means = {ch: [] for ch in marker_channels}
    total_cells = 0

    ox, oy = mosaic_origin
    for ti, tile in enumerate(pilot_tiles):
        tx, ty = tile['x'] - ox, tile['y'] - oy
        logger.info(f"  Pilot tile {ti+1}/{len(pilot_tiles)} at ({tile['x']}, {tile['y']})...")

        # Read channels for Cellpose
        nuclear = _read_channel(nuclear_channel, tx, ty, tile_size, tile_size)
        if nuclear is None:
            logger.warning(f"  Skipping tile ({tx},{ty}): missing nuclear channel")
            continue
        nuc_u8 = _pct_norm(nuclear)

        if nuclei_only:
            cellpose_input = nuc_u8
            cellpose_channels = [0, 0]
        else:
            membrane = _read_channel(membrane_channel, tx, ty, tile_size, tile_size)
            if membrane is None:
                logger.warning(f"  Skipping tile ({tx},{ty}): missing membrane channel")
                continue
            mem_u8 = _pct_norm(membrane)
            cellpose_input = np.stack([mem_u8, nuc_u8], axis=-1)
            cellpose_channels = [1, 2]

        # Run Cellpose
        try:
            masks, _, _ = cellpose_model.eval(
                cellpose_input, channels=cellpose_channels,
                diameter=None, flow_threshold=0.4,
            )
        except Exception as e:
            logger.warning(f"  Cellpose failed on tile ({tx},{ty}): {e}")
            continue

        n_cells = masks.max()
        if n_cells == 0:
            continue

        # Read marker channels
        marker_data = {}
        for ch_idx in marker_channels:
            ch_tile = _read_channel(ch_idx, tx, ty, tile_size, tile_size)
            if ch_tile is not None:
                marker_data[ch_idx] = ch_tile

        # Extract per-cell marker means (same as Phase 0 in detect())
        from scipy.ndimage import find_objects
        slices = find_objects(masks)
        tile_cells = 0
        for label_idx, sl in enumerate(slices):
            if sl is None:
                continue
            label = label_idx + 1
            cell_mask = masks[sl] == label
            area_px = cell_mask.sum()
            if area_px < 10:  # noise rejection only, matches segment()
                continue
            for ch_idx, ch_data in marker_data.items():
                vals = ch_data[sl][cell_mask]
                nonzero = vals[vals > 0]
                mean_val = float(np.mean(nonzero)) if len(nonzero) > 0 else 0.0
                all_marker_means[ch_idx].append(mean_val)
            tile_cells += 1

        total_cells += tile_cells
        logger.info(f"    {n_cells} Cellpose masks → {tile_cells} cells")

    logger.info(f"  Pilot calibration: {total_cells} cells from {len(pilot_tiles)} tiles")

    if total_cells < 100:
        logger.warning(f"  Only {total_cells} pilot cells — too few for reliable GMM. "
                       "Falling back to per-tile GMM.")
        return {}

    # Fit 2-component GMM per marker channel
    gmm_thresholds = {}
    for ch_idx in marker_channels:
        arr = np.array(all_marker_means[ch_idx])
        arr_pos = arr[arr > 0]
        if len(arr_pos) < 50:
            logger.warning(f"  ch{ch_idx}: only {len(arr_pos)} nonzero values — skipping")
            continue

        log_vals = np.log1p(arr_pos).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        gmm.fit(log_vals)

        signal_idx = int(np.argmax(gmm.means_.flatten()))
        bg_mean = float(np.exp(gmm.means_.flatten()[1 - signal_idx]))
        sig_mean = float(np.exp(gmm.means_.flatten()[signal_idx]))

        # Check separation quality
        gmm_means = gmm.means_.flatten()
        gmm_stds = np.sqrt(gmm.covariances_.flatten())
        separation = abs(gmm_means[signal_idx] - gmm_means[1 - signal_idx]) / max(gmm_stds[1 - signal_idx], gmm_stds[signal_idx])

        if separation >= 1.0:
            # Good separation: use GMM P=0.75 crossover (matches final classification)
            lo_raw, hi_raw = 0.0, float(arr.max())
            for _ in range(50):
                mid = (lo_raw + hi_raw) / 2
                p = gmm.predict_proba(np.log1p(np.array([[mid]])))
                if p[0, signal_idx] > 0.75:
                    hi_raw = mid
                else:
                    lo_raw = mid
            threshold = (lo_raw + hi_raw) / 2
            method = 'GMM(P≥0.75)'
        else:
            # Poor separation: use top-5% percentile (matches compute_islet_marker_thresholds)
            threshold = float(np.percentile(arr_pos, 95))
            method = 'top-5%'

        gmm_thresholds[ch_idx] = threshold
        n_pos = int(np.sum(arr > threshold))
        logger.info(f"  ch{ch_idx}: {method} bg={bg_mean:.0f}, sig={sig_mean:.0f}, sep={separation:.2f}σ, "
                    f"threshold={threshold:.0f} (raw), "
                    f"{n_pos}/{len(arr)} positive ({100*n_pos/len(arr):.1f}%)")

    if gmm_thresholds:
        logger.info(f"Islet marker calibration complete: "
                    + ", ".join(f"ch{k}={v:.0f}" for k, v in sorted(gmm_thresholds.items())))

    # Clean up Cellpose model
    del cellpose_model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return gmm_thresholds


# compute_islet_marker_thresholds() moved to segmentation/utils/islet_utils.py (imported at top)


def filter_and_create_html_samples(
    features_list, tile_x, tile_y, tile_rgb, masks, pixel_size_um,
    slide_name, cell_type, html_score_threshold, min_area_um2=25.0,
    tile_percentiles=None, marker_thresholds=None, marker_map=None,
    candidate_mode=False, vessel_params=None,
):
    """Filter detections by quality and create HTML samples.

    Returns list of (sample, count) tuples for accepted detections.
    """
    samples = []
    for feat in features_list:
        features_dict = feat.get('features', {})

        rf_score = feat.get('rf_prediction', feat.get('score', 1.0))
        if rf_score is None:
            rf_score = 1.0  # No classifier loaded; show all candidates
        if rf_score < html_score_threshold:
            continue

        area_um2 = features_dict.get('area', 0) * (pixel_size_um ** 2)
        if area_um2 < min_area_um2:
            continue

        if cell_type == 'vessel' and not candidate_mode:
            vp = vessel_params or {}
            min_ring = vp.get('min_ring_completeness', 0.3)
            min_circ = vp.get('min_circularity', 0.15)
            min_wt = vp.get('min_wall_thickness_um', 1.5)
            if features_dict.get('ring_completeness', 1.0) < min_ring:
                continue
            if features_dict.get('circularity', 1.0) < min_circ:
                continue
            wt = features_dict.get('wall_thickness_mean_um')
            if wt is not None and wt < min_wt:
                continue

        sample = create_sample_from_detection(
            tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name,
            cell_type=cell_type, tile_percentiles=tile_percentiles,
            marker_thresholds=marker_thresholds, marker_map=marker_map,
        )
        if sample:
            samples.append(sample)
    return samples


# =============================================================================
# CZI LOADING
# =============================================================================
# get_czi_metadata() and print_czi_metadata() moved to segmentation/io/czi_loader.py (imported at top)


def generate_tile_grid(mosaic_info, tile_size, overlap_fraction=0.0):
    """Generate tile coordinates covering the mosaic.

    Args:
        mosaic_info: Dict with x, y, width, height
        tile_size: Size of each tile in pixels
        overlap_fraction: Fraction of tile to overlap (0.0 = no overlap, 0.1 = 10% overlap)

    Returns:
        List of tile coordinate dicts with 'x' and 'y' keys
    """
    tiles = []
    x_start = mosaic_info['x']
    y_start = mosaic_info['y']
    width = mosaic_info['width']
    height = mosaic_info['height']

    # Calculate stride (step size) based on overlap
    stride = int(tile_size * (1 - overlap_fraction))
    stride = max(stride, 1)  # Ensure at least 1 pixel stride

    for y in range(y_start, y_start + height, stride):
        for x in range(x_start, x_start + width, stride):
            tiles.append({'x': x, 'y': y})

    return tiles


# =============================================================================
# SAMPLE CREATION FOR HTML
# =============================================================================

# =============================================================================
# SAMPLE CREATION FOR HTML
# =============================================================================

def create_sample_from_detection(tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um, slide_name, cell_type='nmj', crop_size=None, tile_percentiles=None, marker_thresholds=None, marker_map=None):
    """Create an HTML sample from a detection.

    Crop size is calculated dynamically to be 100% larger than the mask,
    ensuring the full mask is visible with context around it.
    Minimum crop size is 224px, maximum is 800px.
    """
    det_id = feat['id']
    # Use mask_label if available, otherwise parse from id (legacy fallback)
    if 'mask_label' in feat:
        det_num = feat['mask_label']
    else:
        det_num = int(det_id.split('_')[-1])
    mask = masks == det_num

    if mask.sum() == 0:
        return None

    # Get centroid
    cy, cx = feat['center'][1], feat['center'][0]

    # Calculate crop size based on mask bounding box
    # Crop should be 100% larger than mask (mask fills ~50% of crop)
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        return None
    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_size = max(mask_h, mask_w)

    # Make crop 2x the mask size (100% larger), with min 224, max 800
    if crop_size is None:
        crop_size = max(224, min(800, int(mask_size * 2)))

    half = crop_size // 2

    # Calculate ideal crop bounds (may extend beyond tile)
    y1_ideal = int(cy) - half
    y2_ideal = int(cy) + half
    x1_ideal = int(cx) - half
    x2_ideal = int(cx) + half

    # Clamp to tile bounds
    y1 = max(0, y1_ideal)
    y2 = min(tile_rgb.shape[0], y2_ideal)
    x1 = max(0, x1_ideal)
    x2 = min(tile_rgb.shape[1], x2_ideal)

    # Validate crop bounds before extracting
    if y2 <= y1 or x2 <= x1:
        logger.warning(f"Invalid crop bounds: y1={y1}, y2={y2}, x1={x1}, x2={x2}, skipping detection {det_id}")
        return None

    crop = tile_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Pad to center the mask if crop was clamped at edges
    # Use max(0, ...) to ensure non-negative padding values
    pad_top = max(0, y1 - y1_ideal)
    pad_bottom = max(0, y2_ideal - y2)
    pad_left = max(0, x1 - x1_ideal)
    pad_right = max(0, x2_ideal - x2)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        # Pad crop with zeros (black)
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        crop_mask = np.pad(crop_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=False)

    # Determine contour color
    features = feat['features']
    contour_color = (0, 255, 0)  # default green
    marker_class = None
    if cell_type == 'islet' and marker_thresholds is not None:
        marker_class, contour_color = classify_islet_marker(features, marker_thresholds, marker_map=marker_map)

    # Normalize and draw contour
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5, global_percentiles=tile_percentiles)
    _bw = (cell_type == 'islet')
    crop_with_contour = draw_mask_contour(crop_norm, crop_mask, color=contour_color, thickness=2, bw_dashed=_bw)

    # Keep at 224x224 (same as classifier input) - already correct size from crop
    img_b64, mime = image_to_base64(crop_with_contour, format='JPEG')

    # Create unique ID using global coordinates (consistent with detection JSON)
    # Global center = tile origin + local center
    local_cx, local_cy = feat['center'][0], feat['center'][1]
    global_cx = tile_x + local_cx
    global_cy = tile_y + local_cy
    uid = f"{slide_name}_{cell_type}_{int(round(global_cx))}_{int(round(global_cy))}"

    # Get stats from features
    area_um2 = features.get('area', 0) * (pixel_size_um ** 2)

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }

    # Add marker classification for islet
    if marker_class is not None:
        stats['marker_class'] = marker_class
        stats['marker_color'] = f'#{contour_color[0]:02x}{contour_color[1]:02x}{contour_color[2]:02x}'

    # Add vessel detection method provenance
    if 'detection_method' in features:
        dm = features['detection_method']
        stats['detection_method'] = ', '.join(dm) if isinstance(dm, list) else dm

    # Add cell-type specific stats
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'sam2_score' in features:
        stats['confidence'] = features['sam2_score']

    # Add classifier score if available (from multi-GPU pipeline)
    rf_pred = feat.get('rf_prediction')
    if rf_pred is not None:
        stats['rf_prediction'] = rf_pred
    else:
        score = feat.get('score')
        if score is not None:
            stats['rf_prediction'] = score

    return {
        'uid': uid,
        'image': img_b64,
        'mime_type': mime,
        'stats': stats,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(args):
    """Main pipeline execution."""
    # Setup logging
    setup_logging(level="DEBUG" if getattr(args, 'verbose', False) else "INFO")

    from datetime import datetime
    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    slide_name = czi_path.stem
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set random seed early — ensures all multi-node shards get identical
    # tissue calibration and tile sampling (same tile list on every node)
    np.random.seed(getattr(args, 'random_seed', 42))

    logger.info("=" * 60)
    logger.info("UNIFIED SEGMENTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Slide: {slide_name}")
    logger.info(f"Run: {run_timestamp}")
    logger.info(f"Cell type: {args.cell_type}")
    logger.info(f"Channel: {args.channel}")
    if args.scene != 0:
        logger.info(f"Scene: {args.scene}")
    if getattr(args, 'multi_marker', False):
        logger.info("Multi-marker mode: ENABLED (auto-enabled --all-channels and --parallel-detection)")
    if getattr(args, 'tile_shard', None):
        shard_idx, shard_total = args.tile_shard
        logger.info(f"Tile shard: {shard_idx}/{shard_total} (detection-only)")
    elif getattr(args, 'detection_only', False):
        logger.info("Detection-only mode (skipping dedup/HTML/CSV)")
    logger.info(f"Random seed: {getattr(args, 'random_seed', 42)}")
    logger.info("=" * 60)

    # ---- Resume early-exit: skip CZI loading if ALL stages are done ----
    if getattr(args, 'resume', None):
        resume_dir = Path(args.resume)
        resume_info = detect_resume_stage(resume_dir, args.cell_type)
        _has_det = resume_info['has_detections'] and not getattr(args, 'force_detect', False)
        _has_html = resume_info['has_html'] and not getattr(args, 'force_html', False)
        _has_tiles = resume_info['has_tiles'] and not getattr(args, 'force_detect', False)

        logger.info(f"Resume state: tiles={resume_info['tile_count']}, "
                     f"detections={'yes' if resume_info['has_detections'] else 'no'}"
                     f" ({resume_info['detection_count']}), "
                     f"html={'yes' if resume_info['has_html'] else 'no'}")

        if _has_det and _has_html:
            # Everything done — just regenerate CSV/summary without loading CZI
            logger.info("All stages complete — regenerating CSV and summary only (no CZI load needed)")
            slide_output_dir = resume_dir
            det_file = slide_output_dir / f"{args.cell_type}_detections.json"
            with open(det_file) as f:
                all_detections = json.load(f)

            # Load pipeline config for metadata
            config_file = slide_output_dir / 'pipeline_config.json'
            if config_file.exists():
                with open(config_file) as f:
                    cfg = json.load(f)
                pixel_size_um = cfg.get('pixel_size_um')
                if pixel_size_um is None:
                    raise ValueError(
                        f"pipeline_config.json is missing 'pixel_size_um'. "
                        f"Re-run detection or add pixel_size_um to {config_file}"
                    )
                mosaic_info = {'width': cfg.get('width', 0), 'height': cfg.get('height', 0),
                               'x': cfg.get('x_start', 0), 'y': cfg.get('y_start', 0)}
            else:
                # Minimal load: just metadata from CZI
                loader = get_loader(czi_path, load_to_ram=False, channel=args.channel, scene=args.scene)
                pixel_size_um = loader.get_pixel_size()
                mosaic_info = {'width': loader.mosaic_size[0], 'height': loader.mosaic_size[1],
                               'x': loader.mosaic_origin[0], 'y': loader.mosaic_origin[1]}

            pct = int(args.sample_fraction * 100)
            _finish_pipeline(
                args, all_detections, [], slide_output_dir, slide_output_dir / "tiles",
                pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
                skip_html=True, resumed=True,
            )
            return

    # RAM-first architecture: Load CZI channel into RAM ONCE at pipeline start
    # This eliminates repeated network I/O for files on network mounts
    # Default is RAM loading for single slides (best performance on network mounts)
    if not args.load_to_ram:
        logger.warning("--no-ram is deprecated and ignored. All data is loaded to RAM.")
        args.load_to_ram = True
    use_ram = True

    logger.info("Loading CZI file with get_loader() (RAM-first architecture)...")
    loader = get_loader(
        czi_path,
        load_to_ram=use_ram,
        channel=args.channel,
        quiet=False,
        scene=args.scene
    )

    # Get mosaic bounds from loader properties
    x_start, y_start = loader.mosaic_origin
    width, height = loader.mosaic_size

    # Build mosaic_info dict for compatibility with existing functions
    mosaic_info = {
        'x': x_start,
        'y': y_start,
        'width': width,
        'height': height,
    }
    pixel_size_um = loader.get_pixel_size()

    logger.info(f"  Mosaic: {mosaic_info['width']} x {mosaic_info['height']} px")
    logger.info(f"  Origin: ({mosaic_info['x']}, {mosaic_info['y']})")
    logger.info(f"  Pixel size: {pixel_size_um:.4f} um/px")
    if use_ram:
        logger.info(f"  Channel {args.channel} loaded to RAM")

    # Always log channel metadata so the log is self-documenting
    try:
        _czi_meta = get_czi_metadata(czi_path, scene=args.scene)
        logger.info(f"  CZI channels ({_czi_meta['n_channels']}):")
        for _ch in _czi_meta['channels']:
            _ex = f"{_ch['excitation_nm']:.0f}" if _ch['excitation_nm'] else "?"
            _em = f"{_ch['emission_nm']:.0f}" if _ch['emission_nm'] else "?"
            _label = _ch['fluorophore'] if _ch['fluorophore'] != 'N/A' else _ch['name']
            logger.info(f"    [{_ch['index']}] {_ch['name']:<20s}  Ex {_ex} → Em {_em} nm  ({_label})")
    except Exception as _e:
        logger.warning(f"  Could not read channel metadata: {_e}")

    # Load additional channels if --all-channels specified (for NMJ specificity checking)
    all_channel_data = {args.channel: loader.channel_data}  # Primary channel
    if getattr(args, 'all_channels', False) and use_ram:
        # Determine which channels to load
        if getattr(args, 'channels', None):
            ch_list = [int(x.strip()) for x in args.channels.split(',')]
            logger.info(f"Loading specified channels {ch_list} for multi-channel analysis...")
        else:
            # Load all channels from CZI
            try:
                dims = loader.reader.get_dims_shape()[0]
                n_channels = dims.get('C', (0, 3))[1]  # Default to 3 channels
            except Exception:
                n_channels = 3  # Fallback
            ch_list = list(range(n_channels))
            logger.info(f"Loading all {len(ch_list)} channels for multi-channel analysis...")

        for ch in ch_list:
            if ch != args.channel:
                logger.info(f"  Loading channel {ch}...")
                ch_loader = get_loader(czi_path, load_to_ram=True, channel=ch, quiet=True, scene=args.scene)
                all_channel_data[ch] = ch_loader.get_channel_data(ch)
        logger.info(f"  Loaded channels: {sorted(all_channel_data.keys())}")

    # Also load CD31 channel to RAM if specified (for vessel validation)
    # Note: get_loader() with the same path returns the cached loader and just adds the new channel
    if args.cell_type == 'vessel' and args.cd31_channel is not None:
        logger.info(f"Loading CD31 channel {args.cd31_channel} to RAM...")
        # Use get_loader to add channel to the same cached loader instance
        loader = get_loader(
            czi_path,
            load_to_ram=use_ram,
            channel=args.cd31_channel,
            quiet=False,
            scene=args.scene
        )

    # Apply photobleaching correction if requested (slide-wide, before tiling)
    if getattr(args, 'photobleaching_correction', False) and use_ram:
        from segmentation.preprocessing.illumination import normalize_rows_columns, estimate_band_severity

        logger.info("Applying slide-wide photobleaching correction...")

        for ch, ch_data in all_channel_data.items():
            original_dtype = ch_data.dtype

            # Report severity before
            severity_before = estimate_band_severity(ch_data)
            logger.info(f"  Channel {ch} before: row_cv={severity_before['row_cv']:.1f}%, "
                       f"col_cv={severity_before['col_cv']:.1f}% ({severity_before['severity']})")

            # Apply row/column normalization to fix banding
            # Note: uses float32 internally, may need ~2x memory temporarily
            corrected = normalize_rows_columns(ch_data)

            # Convert back to original dtype (in-place clip to avoid extra copy)
            if original_dtype == np.uint16:
                np.clip(corrected, 0, 65535, out=corrected)
                corrected = corrected.astype(np.uint16)
            elif original_dtype == np.uint8:
                np.clip(corrected, 0, 255, out=corrected)
                corrected = corrected.astype(np.uint8)
            else:
                corrected = corrected.astype(original_dtype)

            # Update all_channel_data AND loader's internal cache to free stale data.
            # Without this, the loader holds the old pre-correction array
            # (~45 GB per channel) even after all_channel_data is updated.
            all_channel_data[ch] = corrected
            if hasattr(loader, 'set_channel_data'):
                loader.set_channel_data(ch, corrected)

            # Report severity after
            severity_after = estimate_band_severity(corrected)
            logger.info(f"  Channel {ch} after:  row_cv={severity_after['row_cv']:.1f}%, "
                       f"col_cv={severity_after['col_cv']:.1f}% ({severity_after['severity']})")

            # Drop the local reference so GC can free any intermediate float64 arrays
            del corrected

            # Force garbage collection after each channel to free float64 intermediate
            gc.collect()

        logger.info("Photobleaching correction complete.")

    # Apply flat-field illumination correction (smooth out regional intensity gradients)
    if getattr(args, 'normalize_features', True) and use_ram:
        from segmentation.preprocessing.flat_field import estimate_illumination_profile

        logger.info(f"\n{'='*70}")
        logger.info("FLAT-FIELD ILLUMINATION CORRECTION")
        logger.info(f"{'='*70}")
        logger.info("Estimating slide-level illumination profile...")

        illumination_profile = estimate_illumination_profile(all_channel_data)

        for ch in all_channel_data:
            illumination_profile.correct_channel_inplace(all_channel_data[ch], ch)
            # Sync corrected data back to loader for all channels
            if hasattr(loader, 'set_channel_data'):
                loader.set_channel_data(ch, all_channel_data[ch])
            elif ch == args.channel:
                loader.channel_data = all_channel_data[ch]

        gc.collect()
        logger.info("Flat-field correction complete.")

    # Apply Reinhard normalization if params file provided (whole-slide, before tiling)
    if getattr(args, 'norm_params_file', None) and use_ram:
        from segmentation.preprocessing.stain_normalization import apply_reinhard_normalization_MEDIAN

        logger.info(f"\n{'='*70}")
        logger.info("CROSS-SLIDE REINHARD NORMALIZATION (median/MAD)")
        logger.info(f"{'='*70}")
        logger.info(f"Loading params from: {args.norm_params_file}")

        with open(args.norm_params_file, 'r') as f:
            norm_params = json.load(f)

        # Validate required keys
        required_keys = {'L_median', 'L_mad', 'a_median', 'a_mad', 'b_median', 'b_mad'}
        missing_keys = required_keys - set(norm_params.keys())
        if missing_keys:
            raise ValueError(
                f"Normalization params file missing required keys: {missing_keys}. "
                f"Required: {required_keys}"
            )

        logger.info(f"  Target: L_median={norm_params['L_median']:.2f}, L_mad={norm_params['L_mad']:.2f}")
        logger.info(f"  Target: a_median={norm_params['a_median']:.2f}, a_mad={norm_params['a_mad']:.2f}")
        logger.info(f"  Target: b_median={norm_params['b_median']:.2f}, b_mad={norm_params['b_mad']:.2f}")
        if 'n_slides' in norm_params:
            logger.info(f"  Computed from {norm_params['n_slides']} slides, {norm_params.get('n_total_pixels', '?')} pixels")

        # Build RGB image for normalization
        primary_data = loader.channel_data
        if primary_data.ndim == 3 and primary_data.shape[2] >= 3:
            # Already RGB (or more channels) — use first 3 (view, not copy)
            rgb_for_norm = primary_data[:, :, :3]
            # Convert to uint8 if needed (Reinhard expects uint8)
            if rgb_for_norm.dtype == np.uint16:
                logger.info(f"  Converting uint16 → uint8 for normalization ({rgb_for_norm.nbytes / 1e9:.1f} GB)")
                rgb_for_norm = (rgb_for_norm >> 8).astype(np.uint8)
            elif rgb_for_norm.dtype != np.uint8:
                from segmentation.utils.detection_utils import safe_to_uint8
                rgb_for_norm = safe_to_uint8(rgb_for_norm)
        elif primary_data.ndim == 2:
            # Single channel: convert to uint8 FIRST, then stack 3x.
            # This avoids creating a 3-channel uint16 copy (3x memory waste).
            single_u8 = primary_data
            if single_u8.dtype == np.uint16:
                logger.info(f"  Converting single-channel uint16 → uint8 before stacking ({single_u8.nbytes / 1e9:.1f} GB)")
                single_u8 = (single_u8 >> 8).astype(np.uint8)
            elif single_u8.dtype != np.uint8:
                from segmentation.utils.detection_utils import safe_to_uint8
                single_u8 = safe_to_uint8(single_u8)
            rgb_for_norm = np.stack([single_u8] * 3, axis=-1)
            del single_u8
        else:
            raise ValueError(f"Unexpected channel data shape for normalization: {primary_data.shape}")

        logger.info(f"  RGB shape: {rgb_for_norm.shape}, dtype: {rgb_for_norm.dtype} ({rgb_for_norm.nbytes / 1e9:.1f} GB)")
        logger.info(f"  Applying Reinhard normalization (this normalizes tissue blocks, preserves background)...")

        normalized_rgb = apply_reinhard_normalization_MEDIAN(rgb_for_norm, norm_params)
        del rgb_for_norm
        gc.collect()

        # Update channel data with normalized values
        if primary_data.ndim == 3 and primary_data.shape[2] >= 3:
            # Only split normalized RGB back to individual channels if all_channel_data
            # was built from a single 3-channel loader (channels 0,1,2 from RGB CZI)
            loader.channel_data = normalized_rgb
            all_channel_data[args.channel] = normalized_rgb
            # Only update individual channels if they came from the RGB decomposition
            ch_keys = sorted(all_channel_data.keys())
            if len(ch_keys) >= 3 and ch_keys[:3] == [0, 1, 2]:
                for i in range(3):
                    all_channel_data[i] = normalized_rgb[:, :, i]
        else:
            # Single channel — take first channel from normalized RGB
            normalized_single = normalized_rgb[:, :, 0].copy()
            loader.channel_data = normalized_single
            all_channel_data[args.channel] = normalized_single
            del normalized_single

        del normalized_rgb
        gc.collect()

        logger.info("  Reinhard normalization complete.")

    # Generate tile grid (using global coordinates)
    overlap = getattr(args, 'tile_overlap', 0.0)
    logger.info(f"Generating tile grid (size={args.tile_size}, overlap={overlap*100:.0f}%)...")
    all_tiles = generate_tile_grid(mosaic_info, args.tile_size, overlap_fraction=overlap)
    logger.info(f"  Total tiles: {len(all_tiles)}")

    # Determine tissue detection channel BEFORE calibration
    # For islet/tissue_pattern: use nuclear channel (universal cell marker)
    tissue_channel = args.channel
    if args.cell_type == 'islet':
        tissue_channel = getattr(args, 'nuclear_channel', 4)
        logger.info(f"Islet: using DAPI (ch{tissue_channel}) for tissue detection")
    elif args.cell_type == 'tissue_pattern':
        tissue_channel = getattr(args, 'tp_nuclear_channel', 4)
        logger.info(f"Tissue pattern: using nuclear (ch{tissue_channel}) for tissue detection")

    # Calibrate tissue threshold on the SAME channel used for filtering
    manual_threshold = getattr(args, 'variance_threshold', None)
    if manual_threshold is not None:
        variance_threshold = manual_threshold
        logger.info(f"Using manual variance threshold: {variance_threshold:.1f} (skipping K-means calibration)")
    else:
        logger.info("Calibrating tissue threshold...")
        variance_threshold = calibrate_tissue_threshold(
            all_tiles,
            calibration_samples=min(50, len(all_tiles)),
            channel=tissue_channel,
            tile_size=args.tile_size,
            loader=loader,  # Loader handles mosaic origin offset correctly
        )
    logger.info("Filtering to tissue-containing tiles...")
    tissue_tiles = filter_tissue_tiles(
        all_tiles,
        variance_threshold,
        channel=tissue_channel,
        tile_size=args.tile_size,
        loader=loader,  # Loader handles mosaic origin offset correctly
    )

    if len(tissue_tiles) == 0:
        logger.error("No tissue-containing tiles found!")
        return

    # Sample from tissue tiles
    n_sample = max(1, int(len(tissue_tiles) * args.sample_fraction))
    sample_indices = np.random.choice(len(tissue_tiles), n_sample, replace=False)
    sampled_tiles = [tissue_tiles[i] for i in sample_indices]

    logger.info(f"Sampled {len(sampled_tiles)} tiles ({args.sample_fraction*100:.0f}% of {len(tissue_tiles)} tissue tiles)")

    # Sort sampled tiles deterministically (by position) before sharding
    # so all nodes agree on the same ordering regardless of np.random.choice order
    sampled_tiles.sort(key=lambda t: (t['y'], t['x']))  # sort by (y, x)

    # Setup output directories (timestamped to avoid overwriting previous runs)
    pct = int(args.sample_fraction * 100)
    if getattr(args, 'resume', None):
        slide_output_dir = Path(args.resume)
        logger.info(f"Resuming into existing output directory: {slide_output_dir}")
    elif getattr(args, 'resume_from', None):
        slide_output_dir = Path(args.resume_from)
        logger.info(f"Resuming into existing output directory: {slide_output_dir}")
    else:
        slide_output_dir = output_dir / f'{slide_name}_{run_timestamp}_{pct}pct'
    tiles_dir = slide_output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Multi-node tile list sharing: write/read sampled_tiles.json so all shards
    # process the exact same tile list even if tissue calibration diverges slightly.
    # Tiles are dicts with 'x' and 'y' keys.
    if getattr(args, 'tile_shard', None):
        tile_list_file = slide_output_dir / 'sampled_tiles.json'
        if tile_list_file.exists():
            # Another shard already wrote the tile list — use it
            with open(tile_list_file) as f:
                sampled_tiles = json.load(f)
            logger.info(f"Loaded shared tile list from {tile_list_file} ({len(sampled_tiles)} tiles)")
        else:
            # First shard to arrive — write the tile list (atomic via rename)
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(dir=slide_output_dir, suffix='.json')
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump(sampled_tiles, f)
                os.rename(tmp_path, tile_list_file)
                logger.info(f"Wrote shared tile list to {tile_list_file} ({len(sampled_tiles)} tiles)")
            except OSError:
                # Another shard beat us in a race — read theirs
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                if tile_list_file.exists():
                    with open(tile_list_file) as f:
                        sampled_tiles = json.load(f)
                    logger.info(f"Race: loaded shared tile list from {tile_list_file} ({len(sampled_tiles)} tiles)")

        # Round-robin shard assignment
        # NOTE: Per-tile resume not implemented for shard mode.
        # If a shard crashes, it re-processes all tiles in its shard.
        # For large slides, consider using smaller shard counts.
        shard_idx, shard_total = args.tile_shard
        total_before = len(sampled_tiles)
        sampled_tiles = [t for i, t in enumerate(sampled_tiles) if i % shard_total == shard_idx]
        logger.info(f"Tile shard {shard_idx}/{shard_total}: processing {len(sampled_tiles)}/{total_before} tiles")

        # Write shard manifest for auditability
        manifest = {
            'shard_idx': shard_idx, 'shard_total': shard_total,
            'tiles': sampled_tiles,
            'total_sampled': total_before,
            'random_seed': getattr(args, 'random_seed', 42),
        }
        manifest_file = slide_output_dir / f'shard_{shard_idx}_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)

    # ---- Resume: detect completed stages and set skip flags ----
    skip_detection = False
    skip_dedup = False
    skip_html = False

    if getattr(args, 'resume', None) and not getattr(args, 'detection_only', False):
        resume_info = detect_resume_stage(slide_output_dir, args.cell_type)
        logger.info(f"Resume state: tiles={resume_info['tile_count']}, "
                     f"detections={'yes' if resume_info['has_detections'] else 'no'}"
                     f" ({resume_info['detection_count']}), "
                     f"html={'yes' if resume_info['has_html'] else 'no'}")

        # --merge-shards: always reload from tiles (shard detections are pre-dedup),
        # always run dedup, always generate HTML. Checkpointed at each stage.
        if getattr(args, 'merge_shards', False):
            # Validate shard completeness
            shard_manifests = list(slide_output_dir.glob('shard_*_manifest.json'))
            if shard_manifests:
                with open(shard_manifests[0]) as f:
                    m0 = json.load(f)
                expected_shards = m0.get('shard_total', len(shard_manifests))
                if len(shard_manifests) < expected_shards:
                    logger.warning(f"Only {len(shard_manifests)}/{expected_shards} shard manifests found — "
                                   f"some shards may not have completed. Merge will use available data only.")

            merged_det_file = slide_output_dir / f'{args.cell_type}_detections_merged.json'
            deduped_det_file = slide_output_dir / f'{args.cell_type}_detections.json'

            # Checkpoint 1: merged detections (all shards concatenated)
            if merged_det_file.exists() and not getattr(args, 'force_detect', False):
                with open(merged_det_file) as f:
                    all_detections = json.load(f)
                logger.info(f"Checkpoint: loaded {len(all_detections)} merged detections from {merged_det_file.name}")
            elif resume_info['has_tiles']:
                all_detections = reload_detections_from_tiles(tiles_dir, args.cell_type)
                logger.info(f"Merged {len(all_detections)} detections from {resume_info['tile_count']} tile dirs")
                # Save checkpoint
                with open(merged_det_file, 'w') as f:
                    json.dump(all_detections, f, cls=NumpyEncoder)
                logger.info(f"Checkpoint saved: {merged_det_file.name}")
            else:
                logger.error("No tile dirs found — nothing to merge")
                return

            # Checkpoint 2: deduped detections
            if deduped_det_file.exists() and not getattr(args, 'force_detect', False):
                with open(deduped_det_file) as f:
                    all_detections = json.load(f)
                logger.info(f"Checkpoint: loaded {len(all_detections)} deduped detections from {deduped_det_file.name}")
                skip_dedup = True
            skip_detection = True
            skip_html = False  # Always regenerate HTML for merge
            args.force_html = True
        elif resume_info['has_detections'] and not getattr(args, 'force_detect', False):
            skip_detection = True
            skip_dedup = True
            det_file = slide_output_dir / f"{args.cell_type}_detections.json"
            with open(det_file) as f:
                all_detections = json.load(f)
            logger.info(f"Loaded {len(all_detections)} detections from {det_file} (skipping detection + dedup)")
        elif resume_info['has_tiles'] and not getattr(args, 'force_detect', False):
            skip_detection = True
            all_detections = reload_detections_from_tiles(tiles_dir, args.cell_type)
            logger.info(f"Reloaded {len(all_detections)} detections from {resume_info['tile_count']} tile dirs (skipping detection, will dedup)")

        if resume_info['has_html'] and not getattr(args, 'force_html', False):
            skip_html = True
            logger.info("HTML exists — skipping HTML generation (use --force-html to regenerate)")

    # Save pipeline config for resume/regeneration
    pipeline_config = {
        'czi_path': str(czi_path),
        'cell_type': args.cell_type,
        'tile_size': args.tile_size,
        'pixel_size_um': pixel_size_um,
        'scene': args.scene,
        'width': width,
        'height': height,
        'x_start': x_start,
        'y_start': y_start,
        'sample_fraction': args.sample_fraction,
        'tile_overlap': getattr(args, 'tile_overlap', 0.0),
        'channel': args.channel,
    }
    # Add display channel config
    if args.cell_type == 'islet':
        pipeline_config['display_channels'] = getattr(args, 'islet_display_chs', [2, 3, 5])
        pipeline_config['marker_channels'] = getattr(args, 'islet_marker_channels', 'gcg:2,ins:3,sst:5')
    elif args.cell_type == 'tissue_pattern':
        pipeline_config['display_channels'] = getattr(args, 'tp_display_channels_list', [0, 3, 1])
    config_file = slide_output_dir / 'pipeline_config.json'
    if not config_file.exists() or not getattr(args, 'resume', None):
        with open(config_file, 'w') as f:
            json.dump(pipeline_config, f)

    # ---- Resume fast-path: skip detection, go straight to dedup/HTML/CSV ----
    if skip_detection:
        is_multiscale = args.cell_type == 'vessel' and getattr(args, 'multi_scale', False)

        # Run dedup if reloaded from tiles (not from deduped detections JSON)
        if not skip_dedup and getattr(args, 'tile_overlap', 0) > 0 and len(all_detections) > 0 and not is_multiscale:
            from segmentation.processing.deduplication import deduplicate_by_mask_overlap
            pre_dedup = len(all_detections)
            mask_fn = f'{args.cell_type}_masks.h5'
            dedup_sort = 'confidence' if getattr(args, 'dedup_by_confidence', False) else 'area'
            all_detections = deduplicate_by_mask_overlap(
                all_detections, tiles_dir, min_overlap_fraction=0.1,
                mask_filename=mask_fn, sort_by=dedup_sort,
            )
            logger.info(f"Dedup (resume): {pre_dedup} → {len(all_detections)}")

        # Regenerate HTML if needed (requires CZI + tile masks)
        all_samples = []
        if not skip_html:
            # Ensure all channels are loaded for HTML generation
            if args.all_channels or (args.cell_type == 'cell' and getattr(args, 'cellpose_input_channels', None)):
                try:
                    dims = loader.reader.get_dims_shape()[0]
                    _n_ch = dims.get('C', (0, 3))[1]
                except Exception:
                    _n_ch = 3
                for ch in range(_n_ch):
                    if ch not in all_channel_data:
                        logger.info(f"  Loading channel {ch} for HTML generation (resume)...")
                        loader.load_channel(ch)
                        all_channel_data[ch] = loader.get_channel_data(ch)

            logger.info(f"Regenerating HTML for {len(all_detections)} detections from saved tiles...")
            all_samples = _resume_generate_html_samples(
                args, all_detections, tiles_dir,
                all_channel_data, loader, pixel_size_um, slide_name,
                x_start, y_start,
            )
            logger.info(f"Generated {len(all_samples)} HTML samples from resume path")

        # Run the same post-processing as the normal path (HTML export, CSV, summary)
        _finish_pipeline(
            args, all_detections, all_samples, slide_output_dir, tiles_dir,
            pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
            skip_html=skip_html, resumed=True,
            all_tiles=all_tiles, tissue_tiles=tissue_tiles, sampled_tiles=sampled_tiles,
        )
        return

    # ---- Normal path: full detection pipeline ----
    # Initialize detector
    # Use CellDetector + strategy pattern for all cell types
    logger.info("Initializing detector...")

    # CellDetector with strategy pattern
    # Note: mesothelium strategy doesn't need SAM2 (uses ridge detection)
    detector = CellDetector(device="cuda")

    # Load NMJ classifier if provided (supports CNN .pth or RF .pkl)
    classifier_loaded = False
    if args.cell_type == 'nmj' and getattr(args, 'nmj_classifier', None):
        from segmentation.detection.strategies.nmj import load_classifier

        logger.info(f"Loading NMJ classifier from {args.nmj_classifier}...")
        classifier_data = load_classifier(args.nmj_classifier, device=detector.device)

        if classifier_data['type'] == 'cnn':
            # CNN classifier - use transform pipeline
            from torchvision import transforms
            classifier_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            detector.models['classifier'] = classifier_data['model']
            detector.models['classifier_type'] = 'cnn'
            detector.models['transform'] = classifier_transform
            detector.models['device'] = classifier_data['device']
            logger.info("CNN classifier loaded successfully")
            classifier_loaded = True
        else:
            # RF classifier - use features directly
            # New format uses 'pipeline', legacy uses 'model'
            if 'pipeline' in classifier_data:
                detector.models['classifier'] = classifier_data['pipeline']
                detector.models['scaler'] = None  # Pipeline handles scaling internally
            else:
                detector.models['classifier'] = classifier_data['model']
                detector.models['scaler'] = classifier_data.get('scaler')
            detector.models['classifier_type'] = 'rf'
            detector.models['feature_names'] = classifier_data['feature_names']
            logger.info(f"RF classifier loaded successfully ({len(classifier_data['feature_names'])} features)")
            classifier_loaded = True
    # Load islet classifier if provided (generic RF loading)
    elif args.cell_type == 'islet' and getattr(args, 'islet_classifier', None):
        from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
        logger.info(f"Loading islet RF classifier from {args.islet_classifier}...")
        classifier_data = load_nmj_rf_classifier(args.islet_classifier)
        # load_nmj_rf_classifier always returns 'pipeline' key (wraps legacy format)
        detector.models['classifier'] = classifier_data['pipeline']
        detector.models['scaler'] = None
        detector.models['classifier_type'] = 'rf'
        detector.models['feature_names'] = classifier_data['feature_names']
        logger.info(f"Islet RF classifier loaded ({len(classifier_data['feature_names'])} features)")
        classifier_loaded = True
    # Load tissue_pattern classifier if provided (generic RF loading)
    elif args.cell_type == 'tissue_pattern' and getattr(args, 'tp_classifier', None):
        from segmentation.detection.strategies.nmj import load_nmj_rf_classifier
        logger.info(f"Loading tissue_pattern RF classifier from {args.tp_classifier}...")
        classifier_data = load_nmj_rf_classifier(args.tp_classifier)
        detector.models['classifier'] = classifier_data['pipeline']
        detector.models['scaler'] = None
        detector.models['classifier_type'] = 'rf'
        detector.models['feature_names'] = classifier_data['feature_names']
        logger.info(f"Tissue pattern RF classifier loaded ({len(classifier_data['feature_names'])} features)")
        classifier_loaded = True

    # Auto-detect annotation run: no classifier → show ALL candidates in HTML
    # Must happen BEFORE tile processing so filter_and_create_html_samples uses threshold=0.0
    if args.cell_type in ('nmj', 'islet', 'tissue_pattern') and not classifier_loaded and args.html_score_threshold > 0:
        logger.info(f"No classifier loaded — annotation run detected. "
                     f"Overriding --html-score-threshold from {args.html_score_threshold} to 0.0 "
                     f"(will show ALL candidates for annotation)")
        args.html_score_threshold = 0.0

    # Detection parameters
    if args.cell_type == 'nmj':
        params = {
            'intensity_percentile': args.intensity_percentile,
            'min_area': args.min_area,
            'min_skeleton_length': args.min_skeleton_length,
            'max_solidity': args.max_solidity,
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
        }
    elif args.cell_type == 'mk':
        params = {
            'mk_min_area': args.mk_min_area,
            'mk_max_area': args.mk_max_area,
        }
    elif args.cell_type == 'cell':
        params = {
            'min_area_um': args.min_cell_area,
            'max_area_um': args.max_cell_area,
        }
        if args.cellpose_input_channels:
            try:
                parts = args.cellpose_input_channels.split(',')
                params['cellpose_input_channels'] = [int(parts[0]), int(parts[1])]
            except (ValueError, IndexError):
                raise ValueError(f"--cellpose-input-channels must be two integers like '1,0', got '{args.cellpose_input_channels}'")
    elif args.cell_type == 'vessel':
        params = {
            'min_vessel_diameter_um': args.min_vessel_diameter,
            'max_vessel_diameter_um': args.max_vessel_diameter,
            'min_wall_thickness_um': args.min_wall_thickness,
            'max_aspect_ratio': args.max_aspect_ratio,
            'min_circularity': args.min_circularity,
            'min_ring_completeness': args.min_ring_completeness,
            'pixel_size_um': pixel_size_um,
            'classify_vessel_types': args.classify_vessel_types,
            'use_ml_classification': args.use_ml_classification,
            'vessel_classifier_path': args.vessel_classifier_path,
            'candidate_mode': args.candidate_mode,
            'lumen_first': getattr(args, 'lumen_first', False),
            'ring_only': getattr(args, 'ring_only', False),
            'parallel_detection': getattr(args, 'parallel_detection', False),
            'parallel_workers': getattr(args, 'parallel_workers', 3),
            'multi_marker': getattr(args, 'multi_marker', False),
            'smooth_contours': not getattr(args, 'no_smooth_contours', False),
            'smooth_contours_factor': getattr(args, 'smooth_contours_factor', 3.0),
        }
    elif args.cell_type == 'mesothelium':
        params = {
            'target_chunk_area_um2': args.target_chunk_area,
            'min_ribbon_width_um': args.min_ribbon_width,
            'max_ribbon_width_um': args.max_ribbon_width,
            'min_fragment_area_um2': args.min_fragment_area,
            'pixel_size_um': pixel_size_um,
        }
    elif args.cell_type == 'islet':
        nuclei_only = getattr(args, 'nuclei_only', False)
        params = {
            'membrane_channel': None if nuclei_only else getattr(args, 'membrane_channel', 1),
            'nuclear_channel': getattr(args, 'nuclear_channel', 4),
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
            'marker_signal_factor': getattr(args, 'marker_signal_factor', 2.0),
        }
    elif args.cell_type == 'tissue_pattern':
        params = {
            'detection_channels': [int(x) for x in args.tp_detection_channels.split(',')],
            'nuclear_channel': getattr(args, 'tp_nuclear_channel', 4),
            'min_area_um': getattr(args, 'tp_min_area', 20.0),
            'max_area_um': getattr(args, 'tp_max_area', 300.0),
            'extract_deep_features': getattr(args, 'extract_deep_features', False),
        }
    else:
        raise ValueError(f"Unknown cell type: {args.cell_type}")

    logger.info(f"Detection params: {params}")

    # Create strategy for detector
    strategy = create_strategy_for_cell_type(args.cell_type, params, pixel_size_um)
    logger.info(f"Using {strategy.name} strategy: {strategy.get_config()}")

    # Load vessel classifier if ML classification requested
    vessel_classifier = None
    if args.cell_type == 'vessel' and args.use_ml_classification:
        classifier_path = args.vessel_classifier_path
        if classifier_path and Path(classifier_path).exists():
            try:
                vessel_classifier = VesselClassifier.load(classifier_path)
                logger.info(f"Loaded vessel classifier from: {classifier_path}")
                _cv_acc = vessel_classifier.metrics.get('cv_accuracy_mean', 'N/A')
                if isinstance(_cv_acc, (int, float)):
                    logger.info(f"  CV accuracy: {_cv_acc:.4f}")
                else:
                    logger.info(f"  CV accuracy: {_cv_acc}")
            except Exception as e:
                logger.warning(f"Failed to load vessel classifier: {e}")
                logger.warning("Falling back to rule-based classification")
        else:
            logger.warning("--use-ml-classification specified but no model path provided or file not found")
            logger.warning("Falling back to rule-based classification")
            if args.classify_vessel_types:
                logger.info("Using rule-based diameter thresholds for vessel classification")

    # Load VesselTypeClassifier if path provided (for multi-marker 6-type classification)
    vessel_type_classifier = None
    if args.cell_type == 'vessel' and getattr(args, 'vessel_type_classifier', None):
        classifier_path = args.vessel_type_classifier
        if Path(classifier_path).exists():
            try:
                vessel_type_classifier = VesselTypeClassifier.load(classifier_path)
                logger.info(f"Loaded VesselTypeClassifier from: {classifier_path}")
                if vessel_type_classifier.metrics:
                    accuracy = vessel_type_classifier.metrics.get('cv_accuracy_mean', 'N/A')
                    if isinstance(accuracy, float):
                        logger.info(f"  CV accuracy: {accuracy:.4f}")
                    else:
                        logger.info(f"  CV accuracy: {accuracy}")
            except Exception as e:
                logger.warning(f"Failed to load VesselTypeClassifier: {e}")
                vessel_type_classifier = None
        else:
            logger.warning(f"VesselTypeClassifier path does not exist: {classifier_path}")

    # Process tiles
    logger.info("Processing tiles...")
    all_samples = []
    all_detections = []  # Universal list with global coordinates
    deferred_html_tiles = []  # For islet: defer HTML until marker thresholds computed
    is_multiscale = args.cell_type == 'vessel' and getattr(args, 'multi_scale', False)

    # ---- Shared memory creation (used by BOTH regular and multiscale paths) ----
    num_gpus = getattr(args, 'num_gpus', 1)

    if len(all_channel_data) < 2:
        logger.warning("Pipeline works best with --all-channels for multi-channel features")

    from segmentation.processing.multigpu_shm import SharedSlideManager
    from segmentation.processing.multigpu_worker import MultiGPUTileProcessor

    shm_manager = SharedSlideManager()

    try:
        # Load ALL channels to shared memory (RGB display uses first 3, feature extraction needs all)
        ch_keys = sorted(all_channel_data.keys())
        n_channels = len(ch_keys)
        h, w = all_channel_data[ch_keys[0]].shape
        logger.info(f"Creating shared memory for {n_channels} channels ({h}x{w})...")
        logger.info(f"  Channel mapping: {ch_keys}")

        slide_shm_arr = shm_manager.create_slide_buffer(
            slide_name, (h, w, n_channels), all_channel_data[ch_keys[0]].dtype
        )
        for i, ch_key in enumerate(ch_keys):
            slide_shm_arr[:, :, i] = all_channel_data[ch_key]
            logger.info(f"  Loaded channel {ch_key} to shared memory slot {i}")

        ch_to_slot = {ch_key: i for i, ch_key in enumerate(ch_keys)}

        # --- Islet marker calibration (pilot phase, before freeing channel data) ---
        # Like tissue calibration: run Cellpose on ~5% of tiles to estimate
        # global GMM marker thresholds, then use for all tiles.
        islet_gmm_thresholds = {}
        if args.cell_type == 'islet':
            marker_chs_str = getattr(args, 'islet_marker_channels', 'gcg:2,ins:3,sst:5')
            try:
                marker_chs_list = [int(pair.split(':')[1]) for pair in marker_chs_str.split(',')]
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Invalid --islet-marker-channels format: '{marker_chs_str}'. "
                    f"Expected 'name:ch_idx,...' e.g. 'gcg:2,ins:3,sst:5'. Error: {e}"
                )
            n_pilot = max(1, int(len(sampled_tiles) * 0.05))
            pilot_indices = np.random.choice(len(sampled_tiles), n_pilot, replace=False)
            pilot_tiles = [sampled_tiles[i] for i in pilot_indices]
            nuclei_only = getattr(args, 'nuclei_only', False)
            nuc_ch = getattr(args, 'nuclear_channel', 4)
            mem_ch = nuc_ch if nuclei_only else getattr(args, 'membrane_channel', 1)
            islet_gmm_thresholds = calibrate_islet_marker_gmm(
                pilot_tiles=pilot_tiles,
                loader=loader,
                all_channel_data=all_channel_data,
                slide_shm_arr=None,  # Use all_channel_data directly (still in RAM)
                ch_to_slot=None,
                marker_channels=marker_chs_list,
                membrane_channel=mem_ch,
                nuclear_channel=nuc_ch,
                tile_size=args.tile_size,
                pixel_size_um=pixel_size_um,
                nuclei_only=nuclei_only,
                mosaic_origin=(x_start, y_start),
            )

        # Free original channel data — everything is now in shared memory
        mem_freed_gb = sum(arr.nbytes for arr in all_channel_data.values()) / (1024**3)
        del all_channel_data
        # Clear ALL loader channel data (not just primary) to free memory
        if hasattr(loader, 'clear_all_channels'):
            loader.clear_all_channels()
        else:
            loader.channel_data = None
        gc.collect()
        logger.info(f"Freed all_channel_data ({mem_freed_gb:.1f} GB) — using shared memory for all reads")

        # Build strategy parameters from the already-constructed params dict
        strategy_params = dict(params)
        if islet_gmm_thresholds:
            strategy_params['gmm_prefilter_thresholds'] = islet_gmm_thresholds

        # Get classifier path
        classifier_path = None
        if args.cell_type == 'nmj':
            classifier_path = getattr(args, 'nmj_classifier', None)
            if classifier_path:
                logger.info(f"Using specified NMJ classifier: {classifier_path}")
            else:
                logger.info("No --nmj-classifier specified — will return all candidates (annotation run)")
        elif args.cell_type == 'islet':
            classifier_path = getattr(args, 'islet_classifier', None)
            if classifier_path:
                logger.info(f"Using specified islet classifier: {classifier_path}")
            else:
                logger.info("No --islet-classifier specified — will return all candidates (annotation run)")
        elif args.cell_type == 'tissue_pattern':
            classifier_path = getattr(args, 'tp_classifier', None)
            if classifier_path:
                logger.info(f"Using specified tissue_pattern classifier: {classifier_path}")
            else:
                logger.info("No --tp-classifier specified — will return all candidates (annotation run)")

        extract_deep = getattr(args, 'extract_deep_features', False)

        # Vessel-specific params for multi-GPU
        mgpu_cd31_channel = getattr(args, 'cd31_channel', None) if args.cell_type == 'vessel' else None
        mgpu_channel_names = None
        if args.cell_type == 'vessel' and getattr(args, 'channel_names', None):
            names = args.channel_names.split(',')
            mgpu_channel_names = {ch_keys[i]: name.strip()
                                  for i, name in enumerate(names)
                                  if i < len(ch_keys)}

        # Add mosaic origin to slide_info so workers can convert global→relative coords
        mgpu_slide_info = shm_manager.get_slide_info()
        mgpu_slide_info[slide_name]['mosaic_origin'] = (x_start, y_start)

        # ---- Multi-scale vessel detection mode ----
        if is_multiscale:
            logger.info("=" * 60)
            logger.info(f"MULTI-SCALE VESSEL DETECTION — {num_gpus} GPU(s)")
            logger.info("=" * 60)

            from segmentation.utils.multiscale import (
                get_scale_params, generate_tile_grid_at_scale,
                convert_detection_to_full_res, merge_detections_across_scales,
            )
            from tqdm import tqdm as tqdm_progress

            scales = [int(s.strip()) for s in args.scales.split(',')]
            iou_threshold = getattr(args, 'multiscale_iou_threshold', 0.3)
            tile_size = args.tile_size

            logger.info(f"Scales: {scales} (coarse to fine)")
            logger.info(f"IoU threshold for deduplication: {iou_threshold}")

            # One MultiGPUTileProcessor for all scales (workers stay alive, models stay loaded)
            with MultiGPUTileProcessor(
                num_gpus=num_gpus,
                slide_info=mgpu_slide_info,
                cell_type='vessel',
                strategy_params=strategy_params,
                pixel_size_um=pixel_size_um,
                classifier_path=classifier_path,
                extract_deep_features=extract_deep,
                extract_sam2_embeddings=True,
                detection_channel=tissue_channel,
                cd31_channel=mgpu_cd31_channel,
                channel_names=mgpu_channel_names,
                variance_threshold=variance_threshold,
                channel_keys=ch_keys,
            ) as processor:

                all_scale_detections = []  # Accumulate full-res detections across scales
                total_tiles_submitted = 0

                # Resume from checkpoints if available
                completed_scales = set()
                if getattr(args, 'resume_from', None):
                    checkpoint_dir = Path(args.resume_from) / "checkpoints"
                    if checkpoint_dir.exists():
                        # Sort by modification time (not lexicographic — scale_8x > scale_16x lex)
                        checkpoint_files = sorted(
                            checkpoint_dir.glob("scale_*x.json"),
                            key=lambda p: p.stat().st_mtime,
                        )
                        if checkpoint_files:
                            latest = checkpoint_files[-1]
                            with open(latest) as f:
                                all_scale_detections = json.load(f)
                            # Restore numpy arrays for contours (json.load produces lists)
                            for det in all_scale_detections:
                                for key in ('outer', 'inner', 'outer_contour', 'inner_contour'):
                                    if key in det and det[key] is not None:
                                        det[key] = np.array(det[key], dtype=np.int32)
                            for cf in checkpoint_files:
                                # Parse scale from filename like "scale_32x.json"
                                try:
                                    s = int(cf.stem.split('_')[1].rstrip('x'))
                                    completed_scales.add(s)
                                except (IndexError, ValueError):
                                    logger.warning(f"Skipping unrecognized checkpoint file: {cf.name}")
                            logger.info(
                                f"Resumed from {latest}: {len(all_scale_detections)} detections, "
                                f"completed scales: {sorted(completed_scales)}"
                            )

                for scale in scales:
                    if scale in completed_scales:
                        logger.info(f"Scale 1/{scale}x: skipping (checkpointed)")
                        continue

                    scale_params = get_scale_params(scale)
                    scale_tiles = generate_tile_grid_at_scale(
                        mosaic_info['width'], mosaic_info['height'],
                        tile_size, scale, overlap=0,
                    )

                    # Sample tiles if requested
                    if args.sample_fraction < 1.0:
                        n_sample = max(1, int(len(scale_tiles) * args.sample_fraction))
                        indices = np.random.choice(len(scale_tiles), n_sample, replace=False)
                        scale_tiles = [scale_tiles[i] for i in indices]

                    logger.info(
                        f"Scale 1/{scale}x: {len(scale_tiles)} tiles, "
                        f"pixel_size={pixel_size_um * scale:.3f} µm, "
                        f"target: {scale_params.get('description', '')}"
                    )

                    # Submit tiles for this scale with scale metadata
                    for tx_s, ty_s in scale_tiles:
                        # Tile coords in full-res space (for shm extraction):
                        # worker subtracts mosaic_origin, then strides by sf
                        tile_with_dims = {
                            'x': x_start + tx_s * scale,
                            'y': y_start + ty_s * scale,
                            'w': tile_size * scale,
                            'h': tile_size * scale,
                            'scale_factor': scale,
                            'scale_params': scale_params,
                            'tile_x_scaled': tx_s,
                            'tile_y_scaled': ty_s,
                        }
                        processor.submit_tile(slide_name, tile_with_dims)

                    # Collect results for this scale
                    pbar = tqdm_progress(total=len(scale_tiles), desc=f"Scale 1/{scale}x")
                    scale_det_count = 0
                    results_collected = 0

                    while results_collected < len(scale_tiles):
                        result = processor.collect_result(timeout=3600)
                        if result is None:
                            logger.warning(f"Timeout at scale 1/{scale}x")
                            break

                        results_collected += 1
                        pbar.update(1)

                        if result['status'] == 'success':
                            try:
                                tile_dict = result['tile']
                                features_list = result['features_list']
                                sf = result.get('scale_factor', scale)
                                tx_s = tile_dict.get('tile_x_scaled', 0)
                                ty_s = tile_dict.get('tile_y_scaled', 0)

                                # Apply vessel classifiers
                                apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier)

                                # Convert each detection from downscaled-local to full-res global
                                for feat in features_list:
                                    # FIX 1: Promote contour keys — detections_to_features_list
                                    # outputs 'outer_contour'/'inner_contour' but
                                    # convert_detection_to_full_res expects 'outer'/'inner'
                                    if 'outer_contour' in feat and 'outer' not in feat:
                                        feat['outer'] = feat.pop('outer_contour')
                                    if 'inner_contour' in feat and 'inner' not in feat:
                                        feat['inner'] = feat.pop('inner_contour')
                                    if feat.get('features', {}).get('detection_type') == 'arc':
                                        feat['is_arc'] = True

                                    det_fullres = convert_detection_to_full_res(
                                        feat, sf, tx_s, ty_s,
                                        smooth=True,
                                        smooth_base_factor=getattr(args, 'smooth_contours_factor', 3.0),
                                    )

                                    # FIX 2: Add mosaic origin — convert_detection_to_full_res
                                    # produces mosaic-relative coords (0-indexed into shm).
                                    # Add (x_start, y_start) for CZI-global coords matching
                                    # the regular pipeline.
                                    for key in ('center', 'centroid'):
                                        if key in det_fullres:
                                            det_fullres[key][0] += x_start
                                            det_fullres[key][1] += y_start
                                    feats_d = det_fullres.get('features', {})
                                    if isinstance(feats_d, dict):
                                        # features['center'] not scaled by convert_detection_to_full_res
                                        if 'center' in feats_d and feats_d['center'] is not None:
                                            fc = feats_d['center']
                                            feats_d['center'] = [
                                                (fc[0] + tx_s) * sf + x_start,
                                                (fc[1] + ty_s) * sf + y_start,
                                            ]
                                        # outer_center/inner_center already scaled, add mosaic origin
                                        for ck in ('outer_center', 'inner_center'):
                                            if ck in feats_d and feats_d[ck] is not None:
                                                feats_d[ck][0] += x_start
                                                feats_d[ck][1] += y_start
                                    mosaic_offset = np.array([x_start, y_start], dtype=np.int32)
                                    if 'outer' in det_fullres and det_fullres['outer'] is not None:
                                        det_fullres['outer'] = det_fullres['outer'] + mosaic_offset
                                    if 'inner' in det_fullres and det_fullres['inner'] is not None:
                                        det_fullres['inner'] = det_fullres['inner'] + mosaic_offset

                                    # FIX 3: Rebuild outer_contour/outer_contour_global from
                                    # scaled+offset contours (worker created these in downscaled
                                    # local space with tile_x=0, tile_y=0)
                                    for ckey in ('outer', 'inner'):
                                        if ckey in det_fullres and det_fullres[ckey] is not None:
                                            det_fullres[f'{ckey}_contour'] = det_fullres[ckey]
                                            det_fullres[f'{ckey}_contour_global'] = [
                                                [int(pt[0][0]), int(pt[0][1])]
                                                for pt in det_fullres[ckey]
                                            ]

                                    det_fullres['scale_detected'] = sf
                                    all_scale_detections.append(det_fullres)
                                    scale_det_count += 1

                            except Exception as e:
                                import traceback
                                _tid = result.get('tid', '?')
                                logger.error(f"Error post-processing multiscale tile {_tid}: {e}")
                                logger.error(f"Traceback:\n{traceback.format_exc()}")

                        elif result['status'] == 'error':
                            logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

                    pbar.close()
                    total_tiles_submitted += results_collected
                    logger.info(f"Scale 1/{scale}x: {scale_det_count} detections")

                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()

                    # Save checkpoint after each scale
                    checkpoint_dir = slide_output_dir / "checkpoints"
                    checkpoint_dir.mkdir(exist_ok=True)
                    checkpoint_file = checkpoint_dir / f"scale_{scale}x.json"
                    with open(checkpoint_file, 'w') as f:
                        json.dump(all_scale_detections, f, cls=NumpyEncoder)
                    logger.info(f"Checkpoint saved: {checkpoint_file} ({len(all_scale_detections)} detections)")

            # Merge across scales (contour-based IoU dedup)
            logger.info(f"Merging {len(all_scale_detections)} detections across scales...")
            merged_detections = merge_detections_across_scales(
                all_scale_detections, iou_threshold=iou_threshold,
                tile_size=args.tile_size,
            )
            logger.info(f"After merge: {len(merged_detections)} vessels")

            # Regenerate UIDs from full-res global coords and build all_detections
            for det in merged_detections:
                features_dict = det.get('features', {})
                center = features_dict.get('center', det.get('center', [0, 0]))
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    cx, cy = int(center[0]), int(center[1])
                else:
                    cx, cy = 0, 0

                uid = f"{slide_name}_vessel_{cx}_{cy}"
                det['uid'] = uid
                det['slide'] = slide_name
                det['center'] = [cx, cy]
                det['center_um'] = [cx * pixel_size_um, cy * pixel_size_um]
                det['global_center'] = [cx, cy]
                det['global_center_um'] = [cx * pixel_size_um, cy * pixel_size_um]
                all_detections.append(det)

            # Generate HTML crops from shared memory with percentile normalization
            logger.info(f"Generating HTML crops for {len(all_detections)} multiscale detections...")
            from segmentation.io.html_export import image_to_base64

            for det in all_detections:
                features_dict = det.get('features', {})
                cx, cy = det['center']

                diameter_um = features_dict.get('outer_diameter_um', 50)
                diameter_px = int(diameter_um / pixel_size_um)
                crop_size = max(300, min(800, int(diameter_px * 2)))
                half = crop_size // 2

                # Crop from shared memory (0-indexed, subtract mosaic origin)
                rel_cy = cy - y_start
                rel_cx = cx - x_start
                y1 = max(0, rel_cy - half)
                x1 = max(0, rel_cx - half)
                y2 = min(h, rel_cy + half)
                x2 = min(w, rel_cx + half)

                if n_channels >= 3:
                    crop_rgb = np.stack([
                        slide_shm_arr[y1:y2, x1:x2, i] for i in range(3)
                    ], axis=-1)
                else:
                    crop_rgb = np.stack([slide_shm_arr[y1:y2, x1:x2, 0]] * 3, axis=-1)

                if crop_rgb.size == 0:
                    continue

                # Percentile normalize (not /256) for proper dynamic range
                crop_rgb = percentile_normalize(crop_rgb, p_low=1, p_high=99.5)

                b64_str, _ = image_to_base64(crop_rgb)
                sample = {
                    'uid': det['uid'],
                    'image': b64_str,
                    'stats': features_dict,
                }
                all_samples.append(sample)

            logger.info(f"Multi-scale mode: {len(all_detections)} detections, {len(all_samples)} HTML samples "
                        f"from {total_tiles_submitted} tiles on {num_gpus} GPUs")

        # ---- Regular (non-multiscale) tile processing ----
        else:
            logger.info("=" * 60)
            logger.info(f"{args.cell_type.upper()} DETECTION — {num_gpus} GPU(s)")
            logger.info("=" * 60)

            with MultiGPUTileProcessor(
                num_gpus=num_gpus,
                slide_info=mgpu_slide_info,
                cell_type=args.cell_type,
                strategy_params=strategy_params,
                pixel_size_um=pixel_size_um,
                classifier_path=classifier_path,
                extract_deep_features=extract_deep,
                extract_sam2_embeddings=True,
                detection_channel=tissue_channel,
                cd31_channel=mgpu_cd31_channel,
                channel_names=mgpu_channel_names,
                variance_threshold=variance_threshold,
                channel_keys=ch_keys,
                islet_display_channels=getattr(args, 'islet_display_chs', None),
            ) as processor:

                # Submit all tiles (add tile dimensions for worker)
                logger.info(f"Submitting {len(sampled_tiles)} tiles to {num_gpus} GPUs...")
                tile_size = args.tile_size
                for tile in sampled_tiles:
                    # Worker expects 'x', 'y', 'w', 'h' keys
                    tile_with_dims = {
                        'x': tile['x'],
                        'y': tile['y'],
                        'w': tile_size,
                        'h': tile_size,
                    }
                    processor.submit_tile(slide_name, tile_with_dims)

                # Collect results with progress bar
                from tqdm import tqdm as tqdm_progress
                pbar = tqdm_progress(total=len(sampled_tiles), desc="Processing tiles")

                results_collected = 0
                while results_collected < len(sampled_tiles):
                    result = processor.collect_result(timeout=14400)  # 4h timeout per tile (islet: ~7K cells @ 33/min = ~3.5h)
                    if result is None:
                        logger.warning("Timeout waiting for result")
                        break

                    results_collected += 1
                    pbar.update(1)

                    if result['status'] == 'success':
                        try:
                            tile = result['tile']
                            tile_x, tile_y = tile['x'], tile['y']
                            masks = result['masks']
                            features_list = result['features_list']

                            # Skip tiles with no detections (masks=None from worker)
                            if masks is None or len(features_list) == 0:
                                continue

                            # Apply vessel classifier post-processing BEFORE saving
                            if args.cell_type == 'vessel':
                                apply_vessel_classifiers(features_list, vessel_classifier, vessel_type_classifier)

                            # Save tile outputs
                            tile_id = f"tile_{tile_x}_{tile_y}"
                            tile_out = tiles_dir / tile_id
                            tile_out.mkdir(exist_ok=True)

                            # Save masks
                            with h5py.File(tile_out / f"{args.cell_type}_masks.h5", 'w') as f:
                                create_hdf5_dataset(f, 'masks', masks)

                            # Save features (includes vessel classification if applicable)
                            with open(tile_out / f"{args.cell_type}_features.json", 'w') as f:
                                json.dump(features_list, f, cls=NumpyEncoder)

                            # Add detections to global list
                            for feat in features_list:
                                all_detections.append(feat)

                            # Create samples for HTML
                            # Convert global tile coords to 0-based array indices
                            rel_tx = tile_x - x_start
                            rel_ty = tile_y - y_start
                            # Use masks.shape to handle edge tiles (smaller than tile_size at boundaries)
                            tile_h, tile_w = masks.shape[:2]
                            # Read HTML crops from shared memory (all_channel_data freed after shm creation)
                            if args.cell_type == 'islet' and hasattr(args, 'islet_display_chs'):
                                # Islet: display channels from --islet-display-channels (R, G, B)
                                _islet_disp = args.islet_display_chs
                                _shm_dtype = slide_shm_arr.dtype
                                rgb_channels = []
                                for _ci in range(3):
                                    if _ci < len(_islet_disp) and _islet_disp[_ci] in ch_to_slot:
                                        rgb_channels.append(
                                            slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, ch_to_slot[_islet_disp[_ci]]]
                                        )
                                    else:
                                        rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_shm_dtype))
                                tile_rgb_html = np.stack(rgb_channels, axis=-1)
                            elif args.cell_type == 'tissue_pattern':
                                # Tissue pattern: configurable R/G/B display channels
                                # Handles any number of display channels (1, 2, or 3+), always produces (h, w, 3)
                                tp_disp = getattr(args, 'tp_display_channels_list', [0, 3, 1])
                                _shm_dtype = slide_shm_arr.dtype
                                rgb_channels = []
                                for _ci in range(3):
                                    if _ci < len(tp_disp) and tp_disp[_ci] in ch_to_slot:
                                        rgb_channels.append(
                                            slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, ch_to_slot[tp_disp[_ci]]]
                                        )
                                    else:
                                        rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_shm_dtype))
                                tile_rgb_html = np.stack(rgb_channels, axis=-1)
                            elif n_channels >= 3:
                                tile_rgb_html = np.stack([
                                    slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, i]
                                    for i in range(3)
                                ], axis=-1)
                            else:
                                tile_rgb_html = np.stack([
                                    slide_shm_arr[rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w, 0]
                                ] * 3, axis=-1)
                            # Keep uint16 for percentile_normalize — it handles float32 conversion
                            # internally with full 16-bit precision. The old /256 uint8 conversion
                            # caused visible banding/blur after flat-field correction.

                            tile_pct = _compute_tile_percentiles(tile_rgb_html) if getattr(args, 'html_normalization', 'crop') == 'tile' else None

                            if args.cell_type == 'islet':
                                # Flush tile data to disk — keep only lightweight metadata in memory
                                # to avoid OOM from accumulating masks+tile_rgb across all tiles
                                np.save(tile_out / 'tile_rgb_html.npy', tile_rgb_html)
                                if tile_pct is not None:
                                    with open(tile_out / 'tile_pct.json', 'w') as f_pct:
                                        json.dump(tile_pct, f_pct)
                                deferred_html_tiles.append({
                                    'tile_dir': str(tile_out),
                                    'tile_x': tile_x, 'tile_y': tile_y,
                                    'tile_pct': tile_pct,
                                })
                                del masks, tile_rgb_html, features_list
                                result['masks'] = None  # Release array ref from result dict
                                result['features_list'] = None
                                gc.collect()
                            else:
                                _max_html = getattr(args, 'max_html_samples', 0)
                                if _max_html > 0 and len(all_samples) >= _max_html:
                                    pass  # Skip HTML crop generation — cap reached
                                else:
                                    html_samples = filter_and_create_html_samples(
                                        features_list, tile_x, tile_y, tile_rgb_html, masks,
                                        pixel_size_um, slide_name, args.cell_type,
                                        args.html_score_threshold,
                                        tile_percentiles=tile_pct,
                                        candidate_mode=args.candidate_mode,
                                        vessel_params=params if args.cell_type == 'vessel' else None,
                                    )
                                    all_samples.extend(html_samples)

                        except Exception as e:
                            import traceback
                            logger.error(f"Error post-processing tile ({tile_x}, {tile_y}): {e}")
                            logger.error(f"Traceback:\n{traceback.format_exc()}")

                    elif result['status'] in ('empty', 'no_tissue'):
                        pass  # Normal - no tissue in tile
                    elif result['status'] == 'error':
                        logger.warning(f"Tile {result['tid']} error: {result.get('error', 'unknown')}")

                pbar.close()

                # Deferred HTML generation for islet (needs population-level marker thresholds)
                if deferred_html_tiles and args.cell_type == 'islet':
                    _islet_mm = getattr(args, 'islet_marker_map', None)
                    _marker_top_pct = getattr(args, 'marker_top_pct', 5)
                    _pct_chs_str = getattr(args, 'marker_pct_channels', 'sst')
                    _pct_channels = set(s.strip() for s in _pct_chs_str.split(',')) if _pct_chs_str else set()
                    _gmm_p = getattr(args, 'gmm_p_cutoff', 0.75)
                    _ratio_min = getattr(args, 'ratio_min', 1.5)
                    marker_thresholds = compute_islet_marker_thresholds(
                        all_detections, marker_map=_islet_mm,
                        marker_top_pct=_marker_top_pct,
                        pct_channels=_pct_channels,
                        gmm_p_cutoff=_gmm_p,
                        ratio_min=_ratio_min) if all_detections else None
                    # Add marker_class to each detection for JSON export
                    if marker_thresholds:
                        counts = {}
                        for det in all_detections:
                            mc, _ = classify_islet_marker(
                                det.get('features', {}), marker_thresholds, marker_map=_islet_mm)
                            det['marker_class'] = mc
                            counts[mc] = counts.get(mc, 0) + 1
                        logger.info(f"Islet marker classification: {counts}")
                    # Build UID→marker_class lookup for injecting into reloaded features
                    uid_to_marker = {d.get('uid', ''): d.get('marker_class') for d in all_detections}
                    try:
                        for dt in deferred_html_tiles:
                            # Reload tile data from disk (one at a time to control memory)
                            _td = Path(dt['tile_dir'])
                            _tile_rgb = np.load(_td / 'tile_rgb_html.npy')
                            with h5py.File(_td / f'{args.cell_type}_masks.h5', 'r') as _hf:
                                _tile_masks = _hf['masks'][:]
                            with open(_td / f'{args.cell_type}_features.json', 'r') as _ff:
                                _tile_feats = json.load(_ff)
                            # Inject marker_class into reloaded features
                            for _feat in _tile_feats:
                                _mc = uid_to_marker.get(_feat.get('uid', ''))
                                if _mc:
                                    _feat['marker_class'] = _mc
                            html_samples = filter_and_create_html_samples(
                                _tile_feats, dt['tile_x'], dt['tile_y'],
                                _tile_rgb, _tile_masks,
                                pixel_size_um, slide_name, args.cell_type,
                                args.html_score_threshold,
                                tile_percentiles=dt['tile_pct'],
                                marker_thresholds=marker_thresholds,
                                marker_map=_islet_mm,
                                candidate_mode=args.candidate_mode,
                                vessel_params=params if args.cell_type == 'vessel' else None,
                            )
                            all_samples.extend(html_samples)
                            # Clean up deferred temp files
                            _npy_path = _td / 'tile_rgb_html.npy'
                            if _npy_path.exists():
                                _npy_path.unlink()
                            _pct_path = _td / 'tile_pct.json'
                            if _pct_path.exists():
                                _pct_path.unlink()
                            del _tile_rgb, _tile_masks, _tile_feats
                            gc.collect()
                    finally:
                        # Ensure ALL deferred .npy files are cleaned up even on error
                        for dt in deferred_html_tiles:
                            _td = Path(dt['tile_dir'])
                            for _tmp in ('tile_rgb_html.npy', 'tile_pct.json'):
                                _p = _td / _tmp
                                if _p.exists():
                                    _p.unlink()
                        deferred_html_tiles = []
                    gc.collect()

            logger.info(f"Processing complete: {len(all_detections)} {args.cell_type} detections from {results_collected} tiles")

    finally:
        # Cleanup shared memory
        shm_manager.cleanup()

    logger.info(f"Total detections (pre-dedup): {len(all_detections)}")

    # Detection-only mode: skip dedup, HTML, CSV — just save per-tile results and exit
    if getattr(args, 'detection_only', False):
        logger.info(f"Detection-only mode: {len(all_detections)} detections saved to tile dirs. Exiting.")
        if getattr(args, 'tile_shard', None):
            shard_idx, shard_total = args.tile_shard
            logger.info(f"Shard {shard_idx}/{shard_total} complete.")
        return

    # Deduplication: tile overlap causes same detection in adjacent tiles
    # Uses actual mask pixel overlap (loads HDF5 mask files) for accurate dedup
    # Skip for multiscale — already deduped by contour IoU in merge_detections_across_scales()
    if not is_multiscale and getattr(args, 'tile_overlap', 0) > 0 and len(all_detections) > 0:
        from segmentation.processing.deduplication import deduplicate_by_mask_overlap
        pre_dedup = len(all_detections)
        mask_fn = f'{args.cell_type}_masks.h5'
        dedup_sort = 'confidence' if getattr(args, 'dedup_by_confidence', False) else 'area'
        all_detections = deduplicate_by_mask_overlap(
            all_detections, tiles_dir, min_overlap_fraction=0.1,
            mask_filename=mask_fn, sort_by=dedup_sort,
        )

        # Filter HTML samples to match deduped detections and remove duplicate UIDs
        deduped_uids = {det.get('uid', det.get('id', '')) for det in all_detections}
        seen_uids = set()
        unique_samples = []
        for s in all_samples:
            uid = s.get('uid', '')
            if uid in deduped_uids and uid not in seen_uids:
                seen_uids.add(uid)
                unique_samples.append(s)
        logger.info(f"Dedup: {len(all_samples)} HTML samples -> {len(unique_samples)} (removed {len(all_samples) - len(unique_samples)} duplicate UIDs)")
        all_samples = unique_samples

    # (annotation threshold override already applied before tile loop)

    # ---- Shared post-processing: CSV, JSON, HTML, summary, server ----
    _finish_pipeline(
        args, all_detections, all_samples, slide_output_dir, tiles_dir,
        pixel_size_um, slide_name, mosaic_info, run_timestamp, pct,
        all_tiles=all_tiles, tissue_tiles=tissue_tiles, sampled_tiles=sampled_tiles,
        resumed=False, params=params, classifier_loaded=classifier_loaded,
        is_multiscale=is_multiscale, detector=detector,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Unified Cell Segmentation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required (unless using utility commands like --stop-server or --server-status)
    parser.add_argument('--czi-path', type=str, required=False, help='Path to CZI file')
    parser.add_argument('--cell-type', type=str, default=None,
                        choices=['nmj', 'mk', 'cell', 'vessel', 'mesothelium', 'islet', 'tissue_pattern'],
                        help='Cell type to detect (not required if --show-metadata)')

    # CZI scene selection (multi-scene slides, e.g. brain with 2 tissue sections)
    parser.add_argument('--scene', type=int, default=0,
                        help='CZI scene index (0-based, default 0). '
                             'Multi-scene slides store separate tissue sections as scenes.')

    # Metadata inspection
    parser.add_argument('--show-metadata', action='store_true',
                        help='Show CZI channel/dimension info and exit (no processing)')

    # Performance options - RAM loading is the default for single slides (best for network mounts)
    parser.add_argument('--load-to-ram', action='store_true', default=True,
                        help='Load entire channel into RAM first (default: True for best performance on network mounts)')
    parser.add_argument('--no-ram', dest='load_to_ram', action='store_false',
                        help='[DEPRECATED - ignored] RAM loading is always used. This flag is kept for backward compatibility only.')

    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./output)')

    # Tile processing
    parser.add_argument('--tile-size', type=int, default=3000, help='Tile size in pixels')
    parser.add_argument('--tile-overlap', type=float, default=0.10, help='Tile overlap fraction (0.0-0.5, default: 0.10 = 10%% overlap)')
    parser.add_argument('--sample-fraction', type=float, default=1.0, help='Fraction of tissue tiles to process (default: 100%%)')
    parser.add_argument('--channel', type=int, default=None,
                        help='Primary channel index for detection (default: 1 for NMJ, 0 for MK/vessel/cell)')
    parser.add_argument('--all-channels', action='store_true',
                        help='Load all channels for multi-channel analysis (NMJ specificity checking)')
    parser.add_argument('--channel-names', type=str, default=None,
                        help='Comma-separated channel names for feature naming (e.g., "nuclear,sma,pm,cd31" or "nuclear,sma,pm,lyve1")')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    parser.add_argument('--photobleaching-correction', action='store_true',
                        help='Apply slide-wide photobleaching correction (fixes horizontal/vertical banding)')
    parser.add_argument('--norm-params-file', type=str, default=None,
                        help='Path to pre-computed Reinhard normalization params JSON (from compute_normalization_params.py). '
                             'Applies whole-slide Lab-space normalization before tile processing.')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='Apply flat-field illumination correction (default: ON)')
    parser.add_argument('--no-normalize-features', dest='normalize_features', action='store_false',
                        help='Disable flat-field correction (use raw intensities)')
    parser.add_argument('--html-normalization', choices=['tile', 'crop'], default='tile',
                        help='HTML crop normalization scope: tile=shared percentiles per tile, crop=per-crop (default: tile)')

    # NMJ parameters
    parser.add_argument('--intensity-percentile', type=float, default=98)
    parser.add_argument('--min-area', type=int, default=150)
    parser.add_argument('--min-skeleton-length', type=int, default=30)
    parser.add_argument('--max-solidity', type=float, default=0.85,
                        help='Maximum solidity for NMJ detection (branched structures have low solidity)')
    parser.add_argument('--nmj-classifier', type=str, default=None,
                        help='Path to trained NMJ classifier (.pth file)')
    parser.add_argument('--html-score-threshold', type=float, default=0.5,
                        help='Minimum rf_prediction score to show in HTML (default 0.5). '
                             'All detections still saved to JSON regardless. '
                             'Auto-set to 0.0 when no classifier is loaded (annotation run). '
                             'Use --html-score-threshold 0.0 to show ALL candidates explicitly.')
    parser.add_argument('--prior-annotations', type=str, default=None,
                        help='Path to prior annotations JSON file (from round-1 annotation). '
                             'Pre-loads annotations into HTML localStorage so round-1 labels '
                             'are visible during round-2 review after classifier training.')

    # MK parameters (area in um²)
    parser.add_argument('--mk-min-area', type=float, default=200.0,
                        help='Minimum MK area in um² (default 200)')
    parser.add_argument('--mk-max-area', type=float, default=2000.0,
                        help='Maximum MK area in um² (default 2000)')

    # Cell strategy parameters
    parser.add_argument('--cellpose-input-channels', type=str, default=None,
                        help='Two CZI channel indices for 2-channel Cellpose: CYTO,NUC (e.g., 1,0). '
                             'Cyto = plasma membrane/cytoplasmic marker, Nuc = nuclear stain.')
    parser.add_argument('--min-cell-area', type=float, default=50.0,
                        help='Minimum cell area in um² for --cell-type cell (default 50)')
    parser.add_argument('--max-cell-area', type=float, default=200.0,
                        help='Maximum cell area in um² for --cell-type cell (default 200)')

    # Vessel parameters
    parser.add_argument('--min-vessel-diameter', type=float, default=10,
                        help='Minimum vessel outer diameter in µm')
    parser.add_argument('--max-vessel-diameter', type=float, default=1000,
                        help='Maximum vessel outer diameter in µm')
    parser.add_argument('--min-wall-thickness', type=float, default=2,
                        help='Minimum vessel wall thickness in µm')
    parser.add_argument('--max-aspect-ratio', type=float, default=4.0,
                        help='Maximum aspect ratio (exclude longitudinal sections)')
    parser.add_argument('--min-circularity', type=float, default=0.3,
                        help='Minimum circularity for vessel detection')
    parser.add_argument('--min-ring-completeness', type=float, default=0.5,
                        help='Minimum ring completeness (fraction of SMA+ perimeter)')
    parser.add_argument('--cd31-channel', type=int, default=None,
                        help='CD31 channel index for vessel validation (optional)')
    parser.add_argument('--classify-vessel-types', action='store_true',
                        help='Auto-classify vessels by size (capillary/arteriole/artery) using rule-based method')
    parser.add_argument('--use-ml-classification', action='store_true',
                        help='Use ML-based vessel classification (requires trained model)')
    parser.add_argument('--vessel-classifier-path', type=str, default=None,
                        help='Path to trained vessel classifier (.joblib). If not provided with '
                             '--use-ml-classification, falls back to rule-based classification.')
    parser.add_argument('--candidate-mode', action='store_true',
                        help='Enable candidate generation mode for vessel detection. '
                             'Relaxes all thresholds to catch more potential vessels (higher recall). '
                             'Includes detection_confidence score (0-1) for each candidate. '
                             'Use for generating training data for manual annotation + RF classifier.')
    parser.add_argument('--lumen-first', action='store_true',
                        help='[DEPRECATED] Lumen-first detection now runs automatically as a '
                             'supplementary pass alongside ring detection. This flag is a no-op. '
                             'Use --ring-only to disable the supplementary lumen-first pass.')
    parser.add_argument('--ring-only', action='store_true',
                        help='Disable the supplementary lumen-first detection pass. '
                             'Only use Canny edge + contour hierarchy ring detection. '
                             'Useful if you know there are no great vessels in the tissue.')
    parser.add_argument('--parallel-detection', action='store_true',
                        help='Enable parallel multi-marker vessel detection. '
                             'Runs SMA, CD31, and LYVE1 detection in parallel using CPU threads. '
                             'Requires --channel-names to specify marker channels. '
                             'Example: --channel-names "nuclear,sma,cd31,lyve1" --parallel-detection')
    parser.add_argument('--parallel-workers', type=int, default=3,
                        help='Number of parallel workers for multi-marker detection (default: 3). '
                             'One worker per marker type (SMA, CD31, LYVE1).')
    parser.add_argument('--multi-marker', action='store_true',
                        help='Enable full multi-marker vessel detection pipeline. '
                             'Automatically enables --all-channels and --parallel-detection. '
                             'Detects SMA+ rings, CD31+ capillaries, and LYVE1+ lymphatics. '
                             'Merges overlapping candidates from different markers. '
                             'Extracts multi-channel features for downstream classification. '
                             'Example: --multi-marker --channel-names "nuclear,sma,cd31,lyve1"')
    parser.add_argument('--no-smooth-contours', action='store_true',
                        help='Disable B-spline contour smoothing (on by default). '
                             'Smoothing removes stair-step artifacts from coarse-scale detection.')
    parser.add_argument('--smooth-contours-factor', type=float, default=3.0,
                        help='Spline smoothing factor for vessel contours (default: 3.0). '
                             'Higher = smoother. 0 = interpolating spline (passes through all points).')
    parser.add_argument('--vessel-type-classifier', type=str, default=None,
                        help='Path to trained VesselTypeClassifier model (.joblib) for 6-type '
                             'vessel classification (artery/arteriole/vein/capillary/lymphatic/'
                             'collecting_lymphatic). Used with --multi-marker for automated '
                             'vessel type prediction based on marker profiles and morphology.')

    # Multi-scale vessel detection
    parser.add_argument('--multi-scale', action='store_true',
                        help='Enable multi-scale vessel detection. Detects at multiple resolutions '
                             '(1/8x, 1/4x, 1x) to capture all vessel sizes and avoid cross-tile '
                             'fragmentation. Large vessels are detected at coarse scale (1/8x) '
                             'where they fit within a single tile. Requires --cell-type vessel.')
    parser.add_argument('--scales', type=str, default='32,16,8,4,2',
                        help='Comma-separated scale factors for multi-scale detection (default: "32,16,8,4,2"). '
                             'Numbers represent downsampling factors: 32=1/32x (large arteries), '
                             '16=1/16x, 8=1/8x, 4=1/4x (medium), 2=1/2x (small vessels). '
                             'Detection runs coarse-to-fine with IoU deduplication.')
    parser.add_argument('--multiscale-iou-threshold', type=float, default=0.3,
                        help='IoU threshold for deduplicating vessels detected at multiple scales '
                             '(default: 0.3). If a vessel is detected at both coarse and fine scales '
                             'with IoU > threshold, the detection with larger contour area is kept.')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume multiscale run from checkpoints in a previous run directory. '
                             'Skips already-completed scales and reuses the output directory.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume pipeline from an existing run directory. '
                             'Auto-detects completed stages (tiles, detections, HTML) and skips them. '
                             'Requires --czi-path (for HTML crop rendering unless everything is done).')
    parser.add_argument('--force-html', action='store_true', default=False,
                        help='Force HTML regeneration even if html/ exists (use with --resume)')
    parser.add_argument('--force-detect', action='store_true', default=False,
                        help='Force re-detection even if tiles/ has data (use with --resume)')

    # Mesothelium parameters (for LMD chunking)
    parser.add_argument('--target-chunk-area', type=float, default=1500,
                        help='Target area for mesothelium chunks in µm²')
    parser.add_argument('--min-ribbon-width', type=float, default=5,
                        help='Minimum expected ribbon width in µm')
    parser.add_argument('--max-ribbon-width', type=float, default=30,
                        help='Maximum expected ribbon width in µm')
    parser.add_argument('--min-fragment-area', type=float, default=1500,
                        help='Skip mesothelium fragments smaller than this (µm²)')
    parser.add_argument('--add-fiducials', action='store_true', default=True,
                        help='Add calibration cross markers to LMD export')
    parser.add_argument('--no-fiducials', dest='add_fiducials', action='store_false',
                        help='Do not add calibration markers')

    # Islet parameters
    parser.add_argument('--membrane-channel', type=int, default=1,
                        help='Membrane marker channel index for islet Cellpose input (default: 1, AF633)')
    parser.add_argument('--nuclear-channel', type=int, default=4,
                        help='Nuclear marker channel index for islet Cellpose input (default: 4, DAPI)')
    parser.add_argument('--islet-classifier', type=str, default=None,
                        help='Path to trained islet RF classifier (.pkl)')
    parser.add_argument('--islet-display-channels', type=str, default='2,3,5',
                        help='Comma-separated R,G,B channel indices for islet HTML display (default: 2,3,5). '
                             'Channels are mapped to R/G/B in order.')
    parser.add_argument('--islet-marker-channels', type=str, default='gcg:2,ins:3,sst:5',
                        help='Marker-to-channel mapping for islet classification, as name:index pairs. '
                             'Format: "gcg:2,ins:3,sst:5". Names are used in logs and legends.')
    parser.add_argument('--nuclei-only', action='store_true', default=False,
                        help='Nuclei-only mode for islet: use DAPI grayscale for Cellpose '
                             '(channels=[0,0]) instead of membrane+nuclear. SAM2 still runs.')
    parser.add_argument('--marker-signal-factor', type=float, default=2.0,
                        help='Pre-filter divisor for GMM threshold. Cells need marker '
                             'signal > auto_threshold/N to get full features + SAM2. '
                             'Higher = more permissive. 0 = disable. (default 2.0)')
    parser.add_argument('--marker-top-pct', type=float, default=5,
                        help='For percentile-method channels (see --marker-pct-channels), '
                             'classify the top N%% of cells as marker-positive. '
                             '(default 5 = 95th percentile)')
    parser.add_argument('--marker-pct-channels', type=str, default='sst',
                        help='Comma-separated marker names that use percentile-based '
                             'thresholding instead of GMM (default: sst)')
    parser.add_argument('--gmm-p-cutoff', type=float, default=0.75,
                        help='GMM posterior probability cutoff for marker classification. '
                             'Higher = stricter (fewer false positives). (default 0.75)')
    parser.add_argument('--ratio-min', type=float, default=1.5,
                        help='Dominant marker must be >= ratio_min * runner-up for '
                             'single-marker classification. Below → "multi". (default 1.5)')
    parser.add_argument('--dedup-by-confidence', action='store_true', default=False,
                        help='Sort by confidence (score) instead of area during deduplication. '
                             'Default: sort by area (largest mask wins overlap).')

    # Tissue pattern parameters
    parser.add_argument('--tp-detection-channels', type=str, default='0,3',
                        help='Comma-separated channel indices to sum for tissue_pattern detection (default: 0,3 = Slc17a7+Gad1)')
    parser.add_argument('--tp-nuclear-channel', type=int, default=4,
                        help='Nuclear channel for tissue detection (default: 4, Hoechst)')
    parser.add_argument('--tp-display-channels', type=str, default='0,3,1',
                        help='Comma-separated R,G,B channel indices for HTML display (default: 0,3,1 = Slc17a7/Gad1/Htr2a)')
    parser.add_argument('--tp-classifier', type=str, default=None,
                        help='Path to trained tissue_pattern RF classifier (.pkl)')
    parser.add_argument('--tp-min-area', type=float, default=20.0,
                        help='Minimum cell area in um² for tissue_pattern (default 20)')
    parser.add_argument('--tp-max-area', type=float, default=300.0,
                        help='Maximum cell area in um² for tissue_pattern (default 300)')

    # Tissue detection
    parser.add_argument('--variance-threshold', type=float, default=None,
                        help='Manual variance threshold for tissue detection, overriding K-means calibration. '
                             'Use when auto-calibration is too strict (e.g. out-of-focus scenes). '
                             'Check logs for calibrated values to guide manual setting.')

    # Channel selection
    parser.add_argument('--channels', type=str, default=None,
                        help='Comma-separated list of CZI channel indices to load (e.g. "8,9,10,11"). '
                             'If not specified, all channels are loaded when --all-channels is active. '
                             'Use with multi-channel CZIs that have EDF/processing layers to avoid loading unnecessary data.')

    # Feature extraction options
    parser.add_argument('--extract-deep-features', action='store_true',
                        help='Extract ResNet and DINOv2 features (opt-in, default morph+SAM2 only)')
    parser.add_argument('--skip-deep-features', action='store_true',
                        help='Deprecated: deep features are off by default now. Use --extract-deep-features to enable.')

    # GPU processing (always uses multi-GPU infrastructure, even with 1 GPU)
    parser.add_argument('--multi-gpu', action='store_true', default=True,
                        help='[DEPRECATED - always enabled] Multi-GPU processing is now the only code path. '
                             'Use --num-gpus to control how many GPUs are used (default: auto-detect).')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: auto-detect via torch.cuda.device_count(), '
                             'minimum 1). The pipeline always uses the multi-GPU infrastructure, '
                             'even with --num-gpus 1.')

    # HTML export
    parser.add_argument('--samples-per-page', type=int, default=300)
    parser.add_argument('--max-html-samples', type=int, default=0,
                        help='Maximum HTML samples to keep in memory (0=unlimited). '
                             'For full runs with 500K+ cells, set to e.g. 5000 to avoid OOM from base64 crop accumulation.')

    # Server options
    parser.add_argument('--serve', action='store_true', default=False,
                        help='Start HTTP server and wait for Ctrl+C (foreground mode)')
    parser.add_argument('--serve-background', action='store_true', default=True,
                        help='Start HTTP server in background and exit (default: True)')
    parser.add_argument('--no-serve', action='store_true',
                        help='Do not start server after processing')

    # Multi-node sharding
    parser.add_argument('--tile-shard', type=str, default=None,
                        help='Tile shard specification as INDEX/TOTAL (e.g. "0/4" = shard 0 of 4). '
                             'Round-robin assignment: tile i goes to shard i%%TOTAL. '
                             'Implies --detection-only (skips dedup/HTML/CSV).')
    parser.add_argument('--detection-only', action='store_true',
                        help='Skip dedup, HTML generation, and CSV export after tile processing. '
                             'Useful for multi-node runs where a separate merge step handles post-processing.')
    parser.add_argument('--merge-shards', action='store_true', default=False,
                        help='Merge multi-node shard outputs: load all tile detections, dedup, '
                             'generate HTML+CSV. Auto-enabled when --resume finds shard manifests. '
                             'Uses checkpoints so crashes can be resumed.')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for np.random before tissue calibration and tile sampling. '
                             'Ensures all nodes get the same tile list (default: 42).')
    parser.add_argument('--port', type=int, default=8081,
                        help='Port for HTTP server (default: 8081)')
    parser.add_argument('--stop-server', action='store_true',
                        help='Stop any running background server and exit')
    parser.add_argument('--server-status', action='store_true',
                        help='Show status of running server (including public URL) and exit')

    args = parser.parse_args()

    # Handle --stop-server (exit early)
    if args.stop_server:
        setup_logging()
        stop_background_server()
        return

    # Handle --server-status (exit early)
    if args.server_status:
        show_server_status()
        return

    # Handle --show-metadata (exit early)
    if args.show_metadata:
        print_czi_metadata(args.czi_path, scene=args.scene)
        return

    # Require --czi-path for actual pipeline runs
    if args.czi_path is None:
        parser.error("--czi-path is required (unless using --stop-server, --server-status, or --show-metadata)")

    # Require --cell-type if not showing metadata
    if args.cell_type is None:
        parser.error("--cell-type is required unless using --show-metadata")

    # Cell-type-dependent defaults for output-dir and channel
    if args.output_dir is None:
        args.output_dir = str(Path.cwd() / 'output')
    if args.channel is None:
        if args.cell_type == 'nmj':
            args.channel = 1
        elif args.cell_type == 'islet':
            if getattr(args, 'nuclei_only', False):
                args.channel = getattr(args, 'nuclear_channel', 4)
            else:
                args.channel = getattr(args, 'membrane_channel', 1)
        elif args.cell_type == 'tissue_pattern':
            # Primary channel = first detection channel (for tissue loading)
            if not getattr(args, 'tp_detection_channels', None):
                parser.error("--tp-detection-channels is required for tissue_pattern cell type")
            try:
                args.channel = int(args.tp_detection_channels.split(',')[0])
            except (ValueError, IndexError):
                parser.error(f"--tp-detection-channels: first entry must be integer, got '{args.tp_detection_channels}'")
        elif args.cell_type == 'cell' and args.cellpose_input_channels:
            try:
                args.channel = int(args.cellpose_input_channels.split(',')[0])
            except (ValueError, IndexError):
                parser.error(f"--cellpose-input-channels: first entry must be integer, got '{args.cellpose_input_channels}'")
            # 2-channel Cellpose needs both channels loaded into shared memory
            args.all_channels = True
        else:
            args.channel = 0

    # Handle --cell-type islet: auto-enable all-channels, dedup by area (largest wins)
    if args.cell_type == 'islet':
        args.all_channels = True
        # Parse --islet-display-channels into list of ints
        args.islet_display_chs = [int(x.strip()) for x in args.islet_display_channels.split(',')]
        # Parse --islet-marker-channels into dict: {name: channel_index}
        args.islet_marker_map = {}
        for pair in args.islet_marker_channels.split(','):
            pair = pair.strip()
            if ':' not in pair:
                parser.error(f"--islet-marker-channels: each entry must be NAME:CHANNEL, got '{pair}'")
            name, ch = pair.split(':', 1)
            try:
                args.islet_marker_map[name.strip()] = int(ch.strip())
            except ValueError:
                parser.error(f"--islet-marker-channels: channel must be integer, got '{ch.strip()}' in '{pair}'")

    # Handle --cell-type tissue_pattern: auto-enable all-channels, parse display channels
    if args.cell_type == 'tissue_pattern':
        args.all_channels = True
        args.tp_display_channels_list = [int(x) for x in args.tp_display_channels.split(',')]

    # Handle --multi-marker: automatically enable dependent flags
    if getattr(args, 'multi_marker', False):
        if args.cell_type != 'vessel':
            parser.error("--multi-marker is only valid with --cell-type vessel")
        # Auto-enable all-channels and parallel-detection
        args.all_channels = True
        args.parallel_detection = True
        # Note: logger not available yet, will log in run_pipeline()

    # Auto-detect number of GPUs if not specified
    if args.num_gpus is None:
        try:
            args.num_gpus = max(1, torch.cuda.device_count())
        except Exception:
            args.num_gpus = 1

    # --multi-gpu is always True now (kept for backward compatibility)
    args.multi_gpu = True

    # Handle --tile-shard: parse "INDEX/TOTAL" into tuple, implies --detection-only
    if args.tile_shard is not None:
        try:
            parts = args.tile_shard.split('/')
            shard_idx, shard_total = int(parts[0]), int(parts[1])
            if shard_idx < 0 or shard_idx >= shard_total or shard_total < 1:
                parser.error(f"--tile-shard: INDEX must be 0..TOTAL-1, got {shard_idx}/{shard_total}")
            args.tile_shard = (shard_idx, shard_total)
        except (ValueError, IndexError):
            parser.error(f"--tile-shard must be INDEX/TOTAL (e.g. '0/4'), got '{args.tile_shard}'")
        args.detection_only = True  # sharding implies detection-only
        args.no_serve = True  # no server for detection shards
        if not args.resume and not args.resume_from:
            print("WARNING: --tile-shard without --resume: each node will create its own directory. "
                  "Use --resume <shared-dir> so all shards write to the same location.", flush=True)

    # Handle --resume: also set resume_from for multiscale backward compat
    if args.resume:
        if not Path(args.resume).exists():
            parser.error(f"--resume directory does not exist: {args.resume}")
        # Set resume_from so multiscale checkpoint logic also picks it up
        if args.resume_from is None:
            args.resume_from = args.resume
        # Auto-detect shard manifests → enable merge-shards
        shard_manifests = list(Path(args.resume).glob('shard_*_manifest.json'))
        if shard_manifests and not args.merge_shards:
            print(f"Auto-detected {len(shard_manifests)} shard manifests — enabling --merge-shards", flush=True)
            args.merge_shards = True

    # --merge-shards requires --resume
    if args.merge_shards and not args.resume:
        parser.error("--merge-shards requires --resume <shared-output-dir>")

    run_pipeline(args)


if __name__ == '__main__':
    main()
