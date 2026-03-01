"""HTML sample creation, tile grid generation, and islet marker calibration.

Functions for creating HTML annotation crops from detections, generating tile
coordinate grids, computing tile-level percentiles, and calibrating islet marker
thresholds via GMM on pilot tiles.
"""

import gc

import numpy as np

from segmentation.utils.logging import get_logger
from segmentation.utils.islet_utils import classify_islet_marker
from segmentation.io.html_export import percentile_normalize, draw_mask_contour, image_to_base64

logger = get_logger(__name__)


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
    in compute_islet_marker_thresholds). No area filtering -- matches segment().

    Args:
        pilot_tiles: list of tile dicts with 'x', 'y' keys (~5% of tissue tiles)
        loader: CZI loader (for get_tile fallback)
        all_channel_data: dict of channel_idx -> 2D array (or None if using shm)
        slide_shm_arr: shared memory array [n_slots, H, W] (or None)
        ch_to_slot: dict mapping CZI channel -> shm slot index
        marker_channels: list of CZI channel indices for markers (e.g. [2, 3, 5])
        membrane_channel: CZI channel for Cellpose cytoplasm input
        nuclear_channel: CZI channel for Cellpose nuclear input
        tile_size: tile dimensions in pixels
        pixel_size_um: pixel size in micrometers
        nuclei_only: if True, use nuclear channel only for Cellpose
        mosaic_origin: (x, y) mosaic origin offset

    Returns:
        dict mapping channel_idx -> raw GMM threshold (P(signal) = 0.5),
        or empty dict if calibration fails.
    """
    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required -- must come from CZI metadata")
    from sklearn.mixture import GaussianMixture

    if not pilot_tiles:
        return {}

    logger.info(f"Calibrating islet marker thresholds on {len(pilot_tiles)} pilot tiles...")

    # Load Cellpose model (lightweight -- single GPU, no SAM2 needed)
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

    # Helper: percentile normalize uint16 -> uint8
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
        logger.info(f"    {n_cells} Cellpose masks -> {tile_cells} cells")

    logger.info(f"  Pilot calibration: {total_cells} cells from {len(pilot_tiles)} tiles")

    if total_cells < 100:
        logger.warning(f"  Only {total_cells} pilot cells -- too few for reliable GMM. "
                       "Falling back to per-tile GMM.")
        return {}

    # Fit 2-component GMM per marker channel
    gmm_thresholds = {}
    for ch_idx in marker_channels:
        arr = np.array(all_marker_means[ch_idx])
        arr_pos = arr[arr > 0]
        if len(arr_pos) < 50:
            logger.warning(f"  ch{ch_idx}: only {len(arr_pos)} nonzero values -- skipping")
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
            method = 'GMM(P>=0.75)'
        else:
            # Poor separation: use top-5% percentile (matches compute_islet_marker_thresholds)
            threshold = float(np.percentile(arr_pos, 95))
            method = 'top-5%'

        gmm_thresholds[ch_idx] = threshold
        n_pos = int(np.sum(arr > threshold))
        logger.info(f"  ch{ch_idx}: {method} bg={bg_mean:.0f}, sig={sig_mean:.0f}, sep={separation:.2f}sigma, "
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


def filter_and_create_html_samples(
    features_list, tile_x, tile_y, tile_rgb, masks, pixel_size_um,
    slide_name, cell_type, html_score_threshold, min_area_um2=25.0,
    tile_percentiles=None, marker_thresholds=None, marker_map=None,
    candidate_mode=False, vessel_params=None,
):
    """Filter detections by quality and create HTML samples.

    Returns list of sample dicts for accepted detections.
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


def create_sample_from_detection(tile_x, tile_y, tile_rgb, masks, feat, pixel_size_um,
                                 slide_name, cell_type='nmj', crop_size=None,
                                 tile_percentiles=None, marker_thresholds=None,
                                 marker_map=None, contour_thickness=2,
                                 image_format='JPEG'):
    """Create an HTML sample from a detection.

    Crop size is calculated dynamically to be 100% larger than the mask,
    ensuring the full mask is visible with context around it.
    Minimum crop size is 224px, maximum is 800px.

    Works both during live pipeline runs (feat has 'id', 'features', 'center')
    and post-hoc from saved JSON (feat may use 'tile_mask_label', 'uid', etc.).

    Args:
        tile_x, tile_y: Tile origin in global pixel coordinates.
        tile_rgb: HxWx3 tile image array.
        masks: HxW label array (integer mask labels).
        feat: Detection dict with 'features', 'center', etc.
        pixel_size_um: Pixel size in micrometers.
        slide_name: Slide name for UID construction.
        cell_type: Cell type string.
        crop_size: Fixed crop size (None = auto from mask).
        tile_percentiles: Pre-computed tile percentiles for normalization.
        marker_thresholds: Islet marker thresholds dict.
        marker_map: Islet marker-to-channel mapping.
        contour_thickness: Contour line thickness in pixels.
        image_format: 'JPEG' or 'PNG' for the encoded image.
    """
    # Get mask label: try tile_mask_label (saved JSON), mask_label, or parse from id
    det_num = feat.get('tile_mask_label', feat.get('mask_label', 0))
    if det_num == 0 and 'id' in feat:
        try:
            det_num = int(str(feat['id']).split('_')[-1])
        except (ValueError, IndexError):
            pass
    if det_num == 0:
        return None

    mask = masks == det_num
    if mask.sum() == 0:
        return None

    # Get centroid (with fallback to mask centroid)
    center = feat.get('center')
    if center is not None and len(center) >= 2:
        cx, cy = float(center[0]), float(center[1])
    else:
        ys, xs = np.where(mask)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))

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
        return None

    crop = tile_rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Pad to center the mask if crop was clamped at edges
    pad_top = max(0, y1 - y1_ideal)
    pad_bottom = max(0, y2_ideal - y2)
    pad_left = max(0, x1 - x1_ideal)
    pad_right = max(0, x2_ideal - x2)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        crop_mask = np.pad(crop_mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=False)

    # Determine contour color
    features = feat.get('features', {})
    contour_color = (0, 255, 0)  # default green
    marker_class = None
    if cell_type == 'islet' and marker_thresholds is not None:
        marker_class, contour_color = classify_islet_marker(features, marker_thresholds, marker_map=marker_map)

    # Normalize and draw contour
    crop_norm = percentile_normalize(crop, p_low=1, p_high=99.5, global_percentiles=tile_percentiles)
    _bw = (cell_type == 'islet')
    crop_with_contour = draw_mask_contour(crop_norm, crop_mask, color=contour_color, thickness=contour_thickness, bw_dashed=_bw)

    img_b64, mime = image_to_base64(crop_with_contour, format=image_format)

    # Use existing UID if available, otherwise construct from global coords
    if 'uid' in feat:
        uid = feat['uid']
    else:
        local_cx, local_cy = cx, cy
        global_cx = tile_x + local_cx
        global_cy = tile_y + local_cy
        uid = f"{slide_name}_{cell_type}_{int(round(global_cx))}_{int(round(global_cy))}"

    # Build stats
    area_um2 = features.get('area_um2', features.get('area', 0) * (pixel_size_um ** 2))

    stats = {
        'area_um2': area_um2,
        'area_px': features.get('area', 0),
    }

    # Marker classification for islet
    if marker_class is not None:
        stats['marker_class'] = marker_class
        stats['marker_color'] = f'#{contour_color[0]:02x}{contour_color[1]:02x}{contour_color[2]:02x}'

    # Detection method provenance
    if 'detection_method' in features:
        dm = features['detection_method']
        stats['detection_method'] = ', '.join(dm) if isinstance(dm, list) else dm

    # Morphological stats
    if 'elongation' in features:
        stats['elongation'] = features['elongation']
    if 'sam2_iou' in features:
        stats['confidence'] = features['sam2_iou']
    if 'sam2_score' in features:
        stats['confidence'] = features['sam2_score']

    # Classifier score
    rf_pred = feat.get('rf_prediction')
    if rf_pred is not None:
        stats['rf_prediction'] = rf_pred
    else:
        score = feat.get('score')
        if score is not None:
            stats['rf_prediction'] = score

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
