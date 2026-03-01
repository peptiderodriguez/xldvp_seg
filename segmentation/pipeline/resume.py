"""Resume/checkpoint detection and tile reload for the segmentation pipeline.

Functions for detecting which pipeline stages have completed in an existing
run directory, reloading detections from per-tile feature files, and
regenerating HTML samples from saved tile masks and CZI data.
"""

import gc
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py

from segmentation.utils.logging import get_logger
from segmentation.utils.islet_utils import classify_islet_marker, compute_islet_marker_thresholds
from segmentation.pipeline.samples import _compute_tile_percentiles, filter_and_create_html_samples

logger = get_logger(__name__)


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
    from CZI channels, and creates HTML crops -- same output as the normal path.
    """
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

        # Compose tile RGB -- match normal path: direct channel extraction, /256 for uint16
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
                        raise ValueError("No channel data loaded -- check --channel and CZI file")
                    _dtype = next(iter(all_channel_data.values())).dtype
                    rgb_channels.append(np.zeros((tile_h, tile_w), dtype=_dtype))
            tile_rgb = np.stack(rgb_channels, axis=-1)
        elif n_channels >= 3:
            # Standard 3-channel: first 3 loaded channels -> R,G,B
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

        # Keep uint16 for percentile_normalize -- it handles float32 conversion
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
