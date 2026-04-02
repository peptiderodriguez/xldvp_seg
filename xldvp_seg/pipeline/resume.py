"""Resume/checkpoint detection and tile reload for the segmentation pipeline.

Functions for detecting which pipeline stages have completed in an existing
run directory, reloading detections from per-tile feature files, and
regenerating HTML samples from saved tile masks and CZI data.
"""

import gc
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from segmentation.pipeline.samples import _compute_tile_percentiles, filter_and_create_html_samples
from segmentation.utils.islet_utils import classify_islet_marker, compute_islet_marker_thresholds
from segmentation.utils.json_utils import fast_json_load
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def compose_tile_rgb(
    display_channels, region, cell_type, slide_shm_arr=None, ch_to_slot=None, all_channel_data=None
):
    rel_ty, rel_tx, tile_h, tile_w = region
    if slide_shm_arr is not None:
        n_slots = slide_shm_arr.shape[2]
        if cell_type in ("islet", "tissue_pattern") and display_channels:
            dtype = slide_shm_arr.dtype
            rgb_channels = []
            for i in range(3):
                if i < len(display_channels) and display_channels[i] in ch_to_slot:
                    rgb_channels.append(
                        slide_shm_arr[
                            rel_ty : rel_ty + tile_h,
                            rel_tx : rel_tx + tile_w,
                            ch_to_slot[display_channels[i]],
                        ]
                    )
                else:
                    rgb_channels.append(np.zeros((tile_h, tile_w), dtype=dtype))
            return np.stack(rgb_channels, axis=-1)
        elif n_slots >= 3:
            return np.stack(
                [
                    slide_shm_arr[rel_ty : rel_ty + tile_h, rel_tx : rel_tx + tile_w, i]
                    for i in range(3)
                ],
                axis=-1,
            )
        else:
            return np.stack(
                [slide_shm_arr[rel_ty : rel_ty + tile_h, rel_tx : rel_tx + tile_w, 0]] * 3, axis=-1
            )
    else:
        ch_keys = sorted(all_channel_data.keys())
        n_channels = len(ch_keys)
        if cell_type in ("islet", "tissue_pattern") and display_channels:
            rgb_channels = []
            for ch_idx in display_channels[:3]:
                if ch_idx in all_channel_data:
                    rgb_channels.append(
                        all_channel_data[ch_idx][rel_ty : rel_ty + tile_h, rel_tx : rel_tx + tile_w]
                    )
                else:
                    if not all_channel_data:
                        raise ValueError("No channel data loaded -- check --channel and CZI file")
                    dtype = next(iter(all_channel_data.values())).dtype
                    rgb_channels.append(np.zeros((tile_h, tile_w), dtype=dtype))
            return np.stack(rgb_channels, axis=-1)
        elif n_channels >= 3:
            return np.stack(
                [
                    all_channel_data[ch_keys[i]][rel_ty : rel_ty + tile_h, rel_tx : rel_tx + tile_w]
                    for i in range(3)
                ],
                axis=-1,
            )
        else:
            return np.stack(
                [all_channel_data[ch_keys[0]][rel_ty : rel_ty + tile_h, rel_tx : rel_tx + tile_w]]
                * 3,
                axis=-1,
            )


def compute_and_apply_islet_markers(args, all_detections, label=""):
    marker_map = getattr(args, "islet_marker_map", None)
    top_pct = getattr(args, "marker_top_pct", 5)
    pct_chs_str = getattr(args, "marker_pct_channels", "sst")
    pct_channels = set(s.strip() for s in pct_chs_str.split(",")) if pct_chs_str else set()
    gmm_p = getattr(args, "gmm_p_cutoff", 0.75)
    ratio_min = getattr(args, "ratio_min", 1.5)
    thresholds = (
        compute_islet_marker_thresholds(
            all_detections,
            marker_map=marker_map,
            marker_top_pct=top_pct,
            pct_channels=pct_channels,
            gmm_p_cutoff=gmm_p,
            ratio_min=ratio_min,
        )
        if all_detections
        else None
    )
    if thresholds:
        counts = {}
        for det in all_detections:
            mc, _ = classify_islet_marker(
                det.get("features", {}), thresholds, marker_map=marker_map
            )
            det["marker_class"] = mc
            counts[mc] = counts.get(mc, 0) + 1
        logger.info(f"Islet marker classification{label}: {counts}")
    return thresholds, marker_map


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
            dets = fast_json_load(det_file)
            detection_count = len(dets)
        except (OSError, json.JSONDecodeError):
            has_detections = False

    return {
        "has_tiles": tile_count > 0,
        "tile_count": tile_count,
        "has_detections": has_detections,
        "detection_count": detection_count,
        "has_html": html_index.exists(),
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
                features_list = fast_json_load(feat_file)
                all_detections.extend(features_list)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load {feat_file}: {e}")

    # Warn about pre-existing classifier scores
    if all_detections:
        from segmentation.utils.classifier_registry import extract_classifier_info

        scored_count, prov_count, sample_info = extract_classifier_info(all_detections)
        if scored_count > 0:
            if prov_count > 0 and sample_info:
                logger.warning(
                    "Loaded %d detections with pre-existing RF scores "
                    "(classifier: %s, F1=%s, scored %s)",
                    scored_count,
                    sample_info.get("classifier_name", "unknown"),
                    sample_info.get("cv_f1", "?"),
                    sample_info.get("scored_at", "?"),
                )
            else:
                logger.warning(
                    "Loaded %d detections with RF scores but NO provenance metadata. "
                    "These scores are from an unknown classifier -- they may be stale "
                    "or from a different run. Consider re-scoring with apply_classifier.py.",
                    scored_count,
                )

    return all_detections


def _resume_generate_html_samples(
    args,
    all_detections,
    tiles_dir,
    all_channel_data,
    loader,
    pixel_size_um,
    slide_name,
    x_start,
    y_start,
):
    """Generate HTML samples from saved tile masks + CZI data (resume path).

    Groups detections by tile_origin, loads masks from HDF5, composes tile RGB
    from CZI channels, and creates HTML crops -- same output as the normal path.
    """
    cell_type = args.cell_type

    # Determine display channels
    if cell_type == "islet":
        display_chs = getattr(args, "islet_display_chs", [2, 3, 5])
    elif cell_type == "tissue_pattern":
        display_chs = getattr(args, "tp_display_channels_list", [0, 3, 1])
    else:
        display_chs = sorted(all_channel_data.keys())[:3]

    # Group detections by tile
    tile_groups = defaultdict(list)
    for det in all_detections:
        to = det.get("tile_origin")
        if to is None:
            continue
        tile_key = f"tile_{to[0]}_{to[1]}"
        tile_groups[tile_key].append(det)

    # Islet marker thresholds (population-level, needed before generating crops)
    marker_thresholds = None
    _islet_mm = None
    if cell_type == "islet":
        marker_thresholds, _islet_mm = compute_and_apply_islet_markers(
            args, all_detections, label=" (resume)"
        )

    # Sample if max_html_samples set
    _max_html = getattr(args, "max_html_samples", 20000)
    _use_cache = not getattr(args, "force_html", False) and not getattr(args, "force_detect", False)

    # Process tiles (sorted for deterministic ordering when capped)
    all_samples = []
    n_cached = 0
    n_cached_samples = 0
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    from tqdm import tqdm as tqdm_progress

    for tile_key, tile_dets in tqdm_progress(sorted(tile_groups.items()), desc="Resume HTML"):
        # Check max_html cap BEFORE expensive I/O
        if _max_html > 0 and len(all_samples) >= _max_html:
            logger.info(
                f"HTML sample cap reached ({_max_html}). "
                f"Use --max-html-samples 0 for unlimited."
            )
            break

        tile_dir = tiles_dir / tile_key

        # Fast path: load cached HTML samples from previous run
        if _use_cache:
            cache_path = tile_dir / f"{cell_type}_html_samples.json"
            if cache_path.exists():
                try:
                    cached = fast_json_load(cache_path)
                    all_samples.extend(cached)
                    n_cached += 1
                    n_cached_samples += len(cached)
                    continue  # skip expensive re-rendering
                except Exception as e:
                    logger.debug(f"Failed to load HTML cache {cache_path}: {e}, re-rendering")

        mask_file = tile_dir / f"{cell_type}_masks.h5"

        if not mask_file.exists():
            continue
        with h5py.File(mask_file, "r") as hf:
            if "masks" in hf:
                masks = hf["masks"][:]
            elif "labels" in hf:
                masks = hf["labels"][:]
            else:
                continue
            if masks.ndim == 3 and masks.shape[0] == 1:
                masks = masks[0]

        # Get tile origin
        tile_origin = tile_dets[0].get("tile_origin", [0, 0])
        tile_x, tile_y = tile_origin[0], tile_origin[1]

        # Compose tile RGB -- match normal path: direct channel extraction, /256 for uint16
        rel_tx = tile_x - x_start
        rel_ty = tile_y - y_start
        if rel_tx < 0 or rel_ty < 0:
            logger.warning(
                f"Tile origin ({tile_x}, {tile_y}) is outside mosaic bounds "
                f"({x_start}, {y_start}), skipping HTML for this tile"
            )
            continue
        tile_h, tile_w = masks.shape[:2]

        tile_rgb = compose_tile_rgb(
            display_chs,
            (rel_ty, rel_tx, tile_h, tile_w),
            cell_type,
            all_channel_data=all_channel_data,
        )

        if tile_rgb.size == 0:
            continue

        # Keep uint16 for percentile_normalize -- it handles float32 conversion
        # internally with full 16-bit precision. The old /256 uint8 conversion
        # caused visible banding/blur after flat-field correction.

        tile_pct = (
            _compute_tile_percentiles(tile_rgb)
            if getattr(args, "html_normalization", "crop") == "tile"
            else None
        )

        # Generate crops for each detection in this tile
        _vp = (
            {
                "min_ring_completeness": getattr(args, "min_ring_completeness", 0.3),
                "min_circularity": getattr(args, "min_circularity", 0.15),
                "min_wall_thickness_um": getattr(args, "min_wall_thickness", 1.5),
            }
            if cell_type == "vessel"
            else None
        )
        html_samples = filter_and_create_html_samples(
            tile_dets,
            tile_x,
            tile_y,
            tile_rgb,
            masks,
            pixel_size_um,
            slide_name,
            cell_type,
            args.html_score_threshold,
            tile_percentiles=tile_pct,
            marker_thresholds=marker_thresholds,
            marker_map=_islet_mm,
            candidate_mode=getattr(args, "candidate_mode", False),
            vessel_params=_vp,
        )
        all_samples.extend(html_samples)

        del masks, tile_rgb
        gc.collect()

    if n_cached > 0:
        logger.info(
            f"Loaded {n_cached_samples} cached HTML samples from {n_cached}/{len(tile_groups)} tiles, "
            f"{len(all_samples) - n_cached_samples} freshly rendered"
        )

    return all_samples
