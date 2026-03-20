#!/usr/bin/env python3
"""Detect bright regions from a CZI channel and prepare for LMD export.

Percentile-thresholds a single channel, applies morphological cleanup,
splits large regions into equal-area pieces, extracts full features
(morphology + per-channel intensity + SAM2 embeddings), and outputs a
pipeline-compatible detection JSON.

Use for any bright-region detection where the full segmentation pipeline
is overkill: NfL nerve fibers, BTX NMJ regions, autofluorescent deposits,
bright marker-positive tissue patches, etc.

Output format is compatible with run_lmd_export.py (contour_dilated_px,
outer_contour_global, global_center, features) and classify_markers.py
(ch{N}_mean keys).

Usage:
    # Basic: threshold NfL channel, split into 250 um^2 pieces
    python scripts/detect_regions_for_lmd.py \\
        --czi-path slide.czi \\
        --channel 2 \\
        --output-dir output/ \\
        --percentile 98 \\
        --target-area-um2 250

    # Channel by name (auto-resolves via CZI metadata):
    python scripts/detect_regions_for_lmd.py \\
        --czi-path slide.czi \\
        --channel-spec "detect=NfL" \\
        --output-dir output/ \\
        --percentile 98 \\
        --target-area-um2 250

    # With SAM2 embeddings (needs GPU):
    python scripts/detect_regions_for_lmd.py \\
        --czi-path slide.czi \\
        --channel-spec "detect=NfL" \\
        --output-dir output/ \\
        --sam2

    # Custom morphology: larger closing, more dilation
    python scripts/detect_regions_for_lmd.py \\
        --czi-path slide.czi \\
        --channel 2 \\
        --output-dir output/ \\
        --percentile 95 \\
        --close-radius 7 \\
        --dilate-radius 5 \\
        --min-area-um2 500

    # Restrict intensity features to specific channels (skip bad stains):
    python scripts/detect_regions_for_lmd.py \\
        --czi-path slide.czi \\
        --channel 2 \\
        --output-dir output/ \\
        --channels "0,1,2"
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes,
    distance_transform_edt,
    label as ndlabel,
)
from skimage.measure import (
    block_reduce,
    regionprops,
    find_contours,
    approximate_polygon,
)
from skimage.morphology import remove_small_objects, closing, dilation, erosion, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from segmentation.io.czi_loader import CZILoader, get_czi_metadata
from segmentation.utils.json_utils import atomic_json_dump
from segmentation.utils.logging import get_logger, setup_logging
from segmentation.analysis.nuclear_count import _percentile_normalize_to_uint8

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Channel loading
# ---------------------------------------------------------------------------


def load_channel_reduced(loader, channel_idx, block_size=8):
    """Load a CZI channel and downsample via block_reduce (mean pooling).

    The loader retains the full-resolution data in RAM (lazy-loaded).
    Call ``loader._channel_data.pop(channel_idx)`` after this function
    if you no longer need the full-res array for that channel.

    Args:
        loader: CZILoader instance.
        channel_idx: Integer channel index in the CZI.
        block_size: Downsampling factor (default: 8). Each output pixel
            is the mean of a block_size x block_size native pixel block.

    Returns:
        reduced: 2D float32 array at reduced resolution.
        full_shape: (full_h, full_w) at native resolution.
        scale: Scale factor from reduced to native (= 1 / block_size).
    """
    logger.info(f"  Loading channel {channel_idx} to RAM...")
    loader.load_channel(channel_idx)
    full = loader.get_channel_data(channel_idx)
    if full is None:
        raise RuntimeError(
            f"Channel {channel_idx} returned None from loader. "
            f"Loaded channels: {loader.loaded_channels}"
        )
    full_shape = full.shape
    logger.info(f"  Full res: {full.shape}, dtype: {full.dtype}")

    logger.info(f"  Downsampling {block_size}x via block_reduce (mean)...")
    reduced = block_reduce(full, (block_size, block_size), np.mean).astype(np.float32)
    logger.info(f"  Reduced: {reduced.shape}")

    return reduced, full_shape, 1.0 / block_size


# ---------------------------------------------------------------------------
# Thresholding + morphological cleanup
# ---------------------------------------------------------------------------


def threshold_and_clean(
    channel,
    percentile,
    close_radius,
    dilate_radius,
    min_area_um2,
    reduced_pixel_size,
    smooth_sigma=2.0,
):
    """Percentile-threshold a channel and apply morphological cleanup.

    Pipeline: Gaussian smooth -> percentile threshold on nonzero pixels ->
    morphological close -> fill holes -> dilate/erode (rounding) ->
    remove small objects -> label connected components.

    Args:
        channel: 2D float32 array (reduced resolution).
        percentile: Threshold percentile (e.g., 98 = keep top 2%).
        close_radius: Disk radius for morphological closing (0 to skip).
        dilate_radius: Disk radius for dilate/erode rounding (0 to skip).
        min_area_um2: Minimum region area in um^2 after cleanup.
        reduced_pixel_size: um per pixel at reduced resolution.
        smooth_sigma: Gaussian sigma for pre-threshold smoothing (default: 2.0).

    Returns:
        labeled: 2D int32 array of labeled connected components.
        n_regions: Number of regions found.
    """
    logger.info(f"  Gaussian smooth (sigma={smooth_sigma})...")
    smoothed = gaussian_filter(channel, sigma=smooth_sigma)

    # Percentile threshold on nonzero pixels (zeros are CZI padding)
    valid = smoothed[smoothed > 0]
    if len(valid) == 0:
        logger.warning("No nonzero pixels in channel -- nothing to detect")
        return np.zeros_like(channel, dtype=np.int32), 0

    thresh = float(np.percentile(valid, percentile))
    p95 = float(np.percentile(valid, 95))
    p99 = float(np.percentile(valid, 99))
    logger.info(f"  p95={p95:.0f}, p{percentile:.0f}={thresh:.0f}, p99={p99:.0f}")
    mask = smoothed > thresh

    # Morphological cleanup: close -> fill -> dilate -> erode
    if close_radius > 0:
        mask = closing(mask, disk(close_radius))
    mask = binary_fill_holes(mask)
    if dilate_radius > 0:
        mask = dilation(mask, disk(dilate_radius))
        mask = erosion(mask, disk(dilate_radius))

    # Remove small objects
    min_px = int(min_area_um2 / (reduced_pixel_size**2))
    mask = remove_small_objects(mask, min_size=max(min_px, 10))

    labeled, n_regions = ndlabel(mask)
    logger.info(
        f"  {n_regions} regions above p{percentile:.0f} (min {min_area_um2} um^2)"
    )

    return labeled, n_regions


# ---------------------------------------------------------------------------
# Recursive splitting
# ---------------------------------------------------------------------------


def split_region(region_mask, target_px, depth=0, max_depth=10):
    """Recursively split a binary mask into pieces close to target_px area.

    Strategy:
      1. If area is within +/-10% of target, return as-is.
      2. Try watershed with distance-transform seeds.
      3. Fall back to vectorized grid slicing along the longest axis
         when watershed cannot find enough seeds (thin/elongated regions).

    Args:
        region_mask: 2D boolean array (may be a sub-region crop).
        target_px: Target area in pixels per piece.
        depth: Current recursion depth (internal).
        max_depth: Maximum recursion depth before returning mask as-is.

    Returns:
        List of 2D boolean masks, each close to target_px area.
        Empty masks and those below 0.9 * target_px are discarded.
    """
    area = int(region_mask.sum())
    if area < target_px * 0.9:
        return []  # too small to be a useful piece
    if area <= target_px * 1.1:
        return [region_mask]  # within tolerance
    if depth >= max_depth:
        return [region_mask]  # safety limit reached

    n = max(2, area // target_px)

    # --- Strategy 1: watershed on distance-transform peaks ---
    dist = distance_transform_edt(region_mask)
    min_dist = max(2, int(np.sqrt(area / n) / 2))
    pts = peak_local_max(dist, min_distance=min_dist, num_peaks=n)
    if len(pts) < 2:
        # Relax min_distance for thin regions
        pts = peak_local_max(dist, min_distance=1, num_peaks=n)

    if len(pts) >= 2:
        markers = np.zeros_like(region_mask, dtype=np.int32)
        for j, (r, c) in enumerate(pts):
            markers[r, c] = j + 1
        ws = watershed(-dist, markers, mask=region_mask)
        result = []
        for lbl in range(1, ws.max() + 1):
            sub = ws == lbl
            if sub.any():
                result.extend(split_region(sub, target_px, depth + 1, max_depth))
        return result

    # --- Strategy 2: vectorized grid split along longest axis ---
    rows, cols = np.where(region_mask)
    if len(rows) == 0:
        return []

    r_range = rows.max() - rows.min()
    c_range = cols.max() - cols.min()
    result = []

    if c_range >= r_range:
        # Split along columns
        col_min, col_max = int(cols.min()), int(cols.max())
        boundaries = np.linspace(col_min, col_max + 1, n + 1, dtype=int)
        for i in range(n):
            sub = region_mask.copy()
            # Zero out columns outside [boundaries[i], boundaries[i+1])
            if boundaries[i] > 0:
                sub[:, :boundaries[i]] = False
            if boundaries[i + 1] < sub.shape[1]:
                sub[:, boundaries[i + 1]:] = False
            if sub.any():
                result.extend(split_region(sub, target_px, depth + 1, max_depth))
    else:
        # Split along rows
        row_min, row_max = int(rows.min()), int(rows.max())
        boundaries = np.linspace(row_min, row_max + 1, n + 1, dtype=int)
        for i in range(n):
            sub = region_mask.copy()
            if boundaries[i] > 0:
                sub[:boundaries[i], :] = False
            if boundaries[i + 1] < sub.shape[0]:
                sub[boundaries[i + 1]:, :] = False
            if sub.any():
                result.extend(split_region(sub, target_px, depth + 1, max_depth))

    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_morph_features(piece_mask, reduced_pixel_size):
    """Extract morphological features from a binary mask at reduced resolution.

    Args:
        piece_mask: 2D boolean array in reduced-resolution image coordinates.
        reduced_pixel_size: um per pixel at reduced resolution.

    Returns:
        Dict of morphological features. Returns empty dict if regionprops
        finds no labeled region (should not happen for valid masks).
    """
    labeled = piece_mask.astype(np.int32)
    rp = regionprops(labeled)
    if not rp:
        return {}
    p = rp[0]
    perimeter = max(p.perimeter, 1.0)
    minor = max(p.minor_axis_length, 1.0)
    return {
        "area_um2": float(p.area * reduced_pixel_size**2),
        "perimeter": float(perimeter * reduced_pixel_size),
        "circularity": float(4 * math.pi * p.area / (perimeter**2)),
        "solidity": float(p.solidity),
        "eccentricity": float(p.eccentricity),
        "aspect_ratio": float(p.major_axis_length / minor),
        "extent": float(p.extent),
        "equiv_diameter_um": float(p.equivalent_diameter_area * reduced_pixel_size),
    }


def extract_channel_features(piece_mask, channel_arrays):
    """Extract per-channel intensity statistics for pixels within a mask.

    Feature keys follow the pipeline convention: ch{N}_mean, ch{N}_std, etc.

    Args:
        piece_mask: 2D boolean array (same shape as channel arrays).
        channel_arrays: Dict mapping channel index (int) to 2D float32 array.

    Returns:
        Dict of per-channel intensity features (12 features per channel).
    """
    feats = {}
    for ch_idx, ch_data in sorted(channel_arrays.items()):
        pixels = ch_data[piece_mask]
        if len(pixels) == 0:
            continue
        mean_val = float(np.mean(pixels))
        std_val = float(np.std(pixels))
        feats[f"ch{ch_idx}_mean"] = mean_val
        feats[f"ch{ch_idx}_std"] = std_val
        feats[f"ch{ch_idx}_median"] = float(np.median(pixels))
        feats[f"ch{ch_idx}_max"] = float(np.max(pixels))
        feats[f"ch{ch_idx}_min"] = float(np.min(pixels))
        p5, p25, p75, p95 = np.percentile(pixels, [5, 25, 75, 95])
        feats[f"ch{ch_idx}_p5"] = float(p5)
        feats[f"ch{ch_idx}_p25"] = float(p25)
        feats[f"ch{ch_idx}_p75"] = float(p75)
        feats[f"ch{ch_idx}_p95"] = float(p95)
        feats[f"ch{ch_idx}_iqr"] = float(p75 - p25)
        feats[f"ch{ch_idx}_cv"] = float(std_val / max(mean_val, 1e-6))
        feats[f"ch{ch_idx}_snr"] = float(mean_val / max(std_val, 1e-6))
    return feats


def extract_sam2_embeddings(detections, loader, detect_ch, device="cuda"):
    """Extract SAM2 256-dim embeddings for each detection.

    Crops a 512x512 region around each detection center from full-resolution
    channel data, builds a 2-channel pseudo-RGB (detect channel = R,
    first other loaded channel = G, blue = 0), and runs the SAM2 encoder.

    The embedding is the spatial mean of the 256-channel feature map
    (global average pooling), matching the pipeline convention.

    Args:
        detections: List of detection dicts (modified in place: sam2_0..sam2_255
            keys added to each detection's features).
        loader: CZILoader with channels already loaded.
        detect_ch: Channel index used for detection (placed in R channel).
        device: Torch device string (e.g., 'cuda', 'cuda:0').
    """
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from segmentation.utils.device import empty_cache

    repo = Path(__file__).resolve().parent.parent
    checkpoint = str(repo / "checkpoints" / "sam2.1_hiera_large.pt")
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    if not Path(checkpoint).exists():
        logger.warning(
            f"SAM2 checkpoint not found at {checkpoint}, skipping embeddings"
        )
        return

    logger.info(f"  Loading SAM2 model on {device}...")
    sam2_model = build_sam2(config, checkpoint, device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)

    mx, my = loader.mosaic_origin

    # Select channels for pseudo-RGB: detect=R, first other=G
    loaded = loader.loaded_channels
    if not loaded:
        logger.warning("No channels loaded for SAM2, skipping")
        return
    # Prefer detect_ch for R; pick a different channel for G if available
    ch_r = detect_ch if detect_ch in loaded else loaded[0]
    ch_g = next((ch for ch in loaded if ch != ch_r), ch_r)
    data_r = loader.get_channel_data(ch_r)
    data_g = loader.get_channel_data(ch_g)

    if data_r is None or data_g is None:
        logger.warning("Channel data unavailable for SAM2, skipping")
        return

    crop_size = 512
    half = crop_size // 2
    n_done = 0

    for det in detections:
        gc = det.get("global_center")
        if gc is None:
            continue
        # global_center includes mosaic origin; subtract to get array indices
        cx_arr = int(gc[0]) - mx
        cy_arr = int(gc[1]) - my

        y1 = max(0, cy_arr - half)
        y2 = min(data_r.shape[0], cy_arr + half)
        x1 = max(0, cx_arr - half)
        x2 = min(data_r.shape[1], cx_arr + half)
        if y2 - y1 < 64 or x2 - x1 < 64:
            continue

        crop_r = _percentile_normalize_to_uint8(data_r[y1:y2, x1:x2])
        crop_g = _percentile_normalize_to_uint8(data_g[y1:y2, x1:x2])
        rgb = np.stack([crop_r, crop_g, np.zeros_like(crop_r)], axis=-1)

        with torch.inference_mode():
            predictor.set_image(rgb)
            # SAM2 stores spatial feature map in predictor._features["image_embed"]
            # Shape: (1, 256, H_emb, W_emb). Global average pool to get 256-dim.
            img_embed = predictor._features["image_embed"]
            embed = img_embed.mean(dim=(2, 3)).squeeze().cpu().numpy()

        # Reset predictor state to free per-image GPU memory
        predictor.reset_predictor()

        for si in range(256):
            det["features"][f"sam2_{si}"] = float(embed[si])

        n_done += 1
        if n_done % 100 == 0:
            logger.info(f"    SAM2: {n_done}/{len(detections)}...")

    logger.info(f"  SAM2 embeddings: {n_done}/{len(detections)} pieces")

    # Free GPU memory
    del predictor, sam2_model
    empty_cache()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def detect_regions(args):
    """Main detection + featurization pipeline.

    Steps:
      1. Load and downsample the detect channel.
      2. Percentile-threshold + morphological cleanup.
      3. Split large regions into target-area pieces.
      4. Extract morphological features per piece.
      5. Load other channels, extract per-channel intensity features.
      6. (Optional) Extract SAM2 embeddings from full-res crops.
      7. Save pipeline-compatible detection JSON.
    """
    setup_logging(level="INFO")

    # --- Validate inputs ---
    czi_path = Path(args.czi_path)
    if not czi_path.exists():
        logger.error(f"CZI file not found: {czi_path}")
        sys.exit(1)

    if not (0 < args.percentile < 100):
        logger.error(f"--percentile must be in (0, 100), got {args.percentile}")
        sys.exit(1)

    if args.target_area_um2 <= 0:
        logger.error(f"--target-area-um2 must be positive, got {args.target_area_um2}")
        sys.exit(1)

    if args.block_size < 1:
        logger.error(f"--block-size must be >= 1, got {args.block_size}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize loader and resolve channel ---
    loader = CZILoader(str(czi_path), scene=args.scene)
    pixel_size = loader.get_pixel_size()
    full_w, full_h = loader.mosaic_size
    mosaic_x, mosaic_y = loader.mosaic_origin
    czi_meta = get_czi_metadata(str(czi_path), scene=args.scene)
    n_channels = czi_meta["n_channels"]

    if args.channel is not None:
        detect_ch = args.channel
        if detect_ch < 0 or detect_ch >= n_channels:
            logger.error(
                f"--channel {detect_ch} out of range. "
                f"CZI has {n_channels} channels (0-{n_channels - 1})."
            )
            sys.exit(1)
    elif args.channel_spec:
        from segmentation.io.czi_loader import (
            resolve_channel_indices,
            ChannelResolutionError,
        )

        # Parse "detect=NfL" or "detect=750" or just "NfL"
        spec_val = args.channel_spec.split("=", 1)[-1].strip()
        try:
            resolved = resolve_channel_indices(
                czi_meta, [spec_val], filename=czi_path.name
            )
            detect_ch = list(resolved.values())[0]
        except ChannelResolutionError as e:
            logger.error(
                f"Could not resolve channel spec '{args.channel_spec}': {e}"
            )
            sys.exit(1)
        logger.info(f"Resolved channel spec '{args.channel_spec}' -> channel {detect_ch}")
    else:
        logger.error("Provide --channel (index) or --channel-spec (name/wavelength)")
        sys.exit(1)

    # Determine which channels to extract intensity features from
    if args.channels:
        try:
            feature_channels = [int(c.strip()) for c in args.channels.split(",")]
        except ValueError:
            logger.error(f"--channels must be comma-separated integers, got: {args.channels}")
            sys.exit(1)
        for ch in feature_channels:
            if ch < 0 or ch >= n_channels:
                logger.error(f"Channel {ch} out of range (CZI has 0-{n_channels - 1})")
                sys.exit(1)
        # Always include the detect channel
        if detect_ch not in feature_channels:
            feature_channels.append(detect_ch)
        feature_channels = sorted(set(feature_channels))
    else:
        feature_channels = list(range(n_channels))

    logger.info(f"CZI: {czi_path.name}")
    logger.info(f"  Mosaic: {full_w}x{full_h}, pixel_size={pixel_size:.4f} um/px")
    logger.info(f"  Detect channel: {detect_ch}, n_channels: {n_channels}")
    logger.info(f"  Intensity feature channels: {feature_channels}")

    block = args.block_size
    slide_name = czi_path.stem

    # --- Step 1: Load + downsample detect channel ---
    detect_reduced, full_shape, scale = load_channel_reduced(loader, detect_ch, block)
    reduced_ps = pixel_size * block  # um per reduced pixel

    # --- Step 2: Threshold + morphological cleanup ---
    labeled, n_regions = threshold_and_clean(
        detect_reduced,
        args.percentile,
        args.close_radius,
        args.dilate_radius,
        args.min_area_um2,
        reduced_ps,
        smooth_sigma=args.smooth_sigma,
    )

    if n_regions == 0:
        logger.warning("No regions found. Try lowering --percentile or --min-area-um2.")
        # Write empty JSON so downstream tools do not fail on missing file
        output_path = output_dir / f"{slide_name}_region_pieces.json"
        atomic_json_dump([], output_path)
        logger.info(f"Wrote empty detection file: {output_path}")
        return

    # --- Step 3: Split into pieces + extract contours + morph features ---
    target_px = max(1, int(args.target_area_um2 / (reduced_ps**2)))
    props = regionprops(labeled)
    logger.info(
        f"  Splitting {len(props)} regions into ~{args.target_area_um2} um^2 pieces "
        f"(target_px={target_px} at {block}x reduction)..."
    )

    detections = []
    # Store reduced-resolution mask coordinates for each piece.
    # These are consumed in Step 4 (channel features) and then discarded.
    piece_mask_coords = []

    for prop in props:
        area_um2 = prop.area * (reduced_ps**2)
        if area_um2 < args.min_area_um2:
            continue

        minr, minc, maxr, maxc = prop.bbox
        pad = 2
        minr_p = max(0, minr - pad)
        minc_p = max(0, minc - pad)
        maxr_p = min(labeled.shape[0], maxr + pad)
        maxc_p = min(labeled.shape[1], maxc + pad)
        region_mask = labeled[minr_p:maxr_p, minc_p:maxc_p] == prop.label

        pieces = split_region(region_mask, target_px)

        for piece_mask in pieces:
            piece_area_px = int(piece_mask.sum())
            piece_area_um2 = piece_area_px * (reduced_ps**2)
            if piece_area_um2 < args.target_area_um2 * 0.9:
                continue

            # Smooth contour for LMD export
            piece_smooth = gaussian_filter(piece_mask.astype(np.float64), sigma=1.5)
            contours = find_contours(piece_smooth, 0.5)
            if not contours:
                continue
            contour = approximate_polygon(max(contours, key=len), tolerance=1.0)

            # Convert contour from reduced-crop coords to global native-pixel coords
            contour_global = []
            for r, c in contour:
                gx = float((c + minc_p) * block + mosaic_x)
                gy = float((r + minr_p) * block + mosaic_y)
                contour_global.append([gx, gy])

            # Centroid in global native-pixel coords
            prows, pcols = np.where(piece_mask)
            cx = float((np.mean(pcols) + minc_p) * block + mosaic_x)
            cy = float((np.mean(prows) + minr_p) * block + mosaic_y)

            # Reduced-image coordinates for intensity feature extraction
            pr_global = prows + minr_p
            pc_global = pcols + minc_p

            # Area in native pixels
            area_native_px = piece_area_px * (block**2)

            uid = f"{slide_name}_region_{int(cx)}_{int(cy)}"

            # Morph features from reduced-resolution mask
            full_mask = np.zeros(detect_reduced.shape[:2], dtype=bool)
            full_mask[pr_global, pc_global] = True
            feats = extract_morph_features(full_mask, reduced_ps)
            feats["area"] = area_native_px
            feats["pixel_size_um"] = pixel_size
            feats["detection_method"] = f"percentile_{args.percentile}"
            feats["block_size"] = block

            det = {
                "id": uid,
                "uid": uid,
                "center": [cx - mosaic_x, cy - mosaic_y],
                "global_center": [cx, cy],
                "global_center_um": [cx * pixel_size, cy * pixel_size],
                "slide_name": slide_name,
                "pixel_size_um": pixel_size,
                "outer_contour_global": contour_global,
                "contour_dilated_px": contour_global,
                "features": feats,
                "rf_prediction": None,
            }
            detections.append(det)
            piece_mask_coords.append((pr_global, pc_global))

    logger.info(f"  {len(detections)} pieces from {len(props)} regions")

    if not detections:
        logger.warning("No pieces survived splitting. Check --target-area-um2 and --min-area-um2.")
        output_path = output_dir / f"{slide_name}_region_pieces.json"
        atomic_json_dump([], output_path)
        logger.info(f"Wrote empty detection file: {output_path}")
        return

    # --- Step 4: Per-channel intensity features ---
    logger.info(
        f"  Loading {len(feature_channels)} channels for intensity features..."
    )
    channel_arrays = {}
    for ch_idx in feature_channels:
        if ch_idx == detect_ch:
            # Detect channel is already loaded and downsampled
            channel_arrays[ch_idx] = detect_reduced
        else:
            logger.info(f"    Loading ch{ch_idx}...")
            ch_reduced, _, _ = load_channel_reduced(loader, ch_idx, block)
            channel_arrays[ch_idx] = ch_reduced
            # Free full-res data from loader to reduce memory pressure.
            # SAM2 (Step 5) reloads the channels it needs at full res.
            loader._channel_data.pop(ch_idx, None)

    logger.info(
        f"    Extracting intensity features ({len(channel_arrays)} channels)..."
    )
    for det, (pr, pc) in zip(detections, piece_mask_coords):
        full_mask = np.zeros(detect_reduced.shape[:2], dtype=bool)
        full_mask[pr, pc] = True
        ch_feats = extract_channel_features(full_mask, channel_arrays)
        det["features"].update(ch_feats)

    # Free downsampled arrays (keep detect_reduced reference for shape only)
    del channel_arrays
    del piece_mask_coords

    # --- Step 5: SAM2 embeddings (optional, needs GPU) ---
    if args.sam2:
        from segmentation.utils.device import get_default_device

        device = get_default_device()
        if str(device) == "cpu":
            logger.warning("No GPU available, skipping SAM2 embeddings")
        else:
            # SAM2 needs full-res channel data (loader still has detect_ch loaded)
            extract_sam2_embeddings(detections, loader, detect_ch, device)

    # --- Summary ---
    if detections:
        n_feats = len(detections[0]["features"])
        areas = [d["features"]["area_um2"] for d in detections]
        logger.info(f"  {n_feats} features per piece")
        logger.info(
            f"  Area: {min(areas):.0f}-{max(areas):.0f} um^2, "
            f"median {np.median(areas):.0f}"
        )
        ys = [d["global_center"][1] for d in detections]
        logger.info(f"  Y range: {min(ys):.0f}-{max(ys):.0f} (CZI height: {full_h})")

    # --- Save ---
    output_path = output_dir / f"{slide_name}_region_pieces.json"
    atomic_json_dump(detections, output_path)
    logger.info(f"Saved {len(detections)} pieces to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect bright regions from a CZI channel via percentile thresholding, "
            "split into equal-area pieces, extract features (morph + intensity + SAM2), "
            "and output pipeline-compatible detection JSON for LMD export."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input ---
    inp = parser.add_argument_group("Input")
    inp.add_argument(
        "--czi-path",
        required=True,
        help="Path to CZI file",
    )
    inp.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Channel index for detection (0-based). Use czi_info.py to check.",
    )
    inp.add_argument(
        "--channel-spec",
        default=None,
        help=(
            'Channel spec for detection, e.g., "detect=NfL", "detect=750", '
            'or just "NfL". Resolves via CZI metadata (name or wavelength).'
        ),
    )
    inp.add_argument(
        "--output-dir",
        required=True,
        help="Output directory (created if needed)",
    )
    inp.add_argument(
        "--scene",
        type=int,
        default=0,
        help="CZI scene index (default: 0)",
    )

    # --- Thresholding ---
    thresh = parser.add_argument_group("Thresholding")
    thresh.add_argument(
        "--percentile",
        type=float,
        default=98,
        help=(
            "Intensity percentile threshold on nonzero pixels "
            "(default: 98 = keep top 2%%). Lower = more regions."
        ),
    )
    thresh.add_argument(
        "--block-size",
        type=int,
        default=8,
        help=(
            "Downsampling factor for thresholding/splitting "
            "(default: 8). Increase for very large CZIs to save RAM."
        ),
    )
    thresh.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for pre-threshold smoothing (default: 2.0)",
    )

    # --- Morphology ---
    morph = parser.add_argument_group("Morphological cleanup")
    morph.add_argument(
        "--close-radius",
        type=int,
        default=5,
        help="Disk radius for morphological closing, 0 to skip (default: 5)",
    )
    morph.add_argument(
        "--dilate-radius",
        type=int,
        default=3,
        help="Disk radius for dilate/erode rounding, 0 to skip (default: 3)",
    )

    # --- Splitting ---
    split = parser.add_argument_group("Region splitting")
    split.add_argument(
        "--target-area-um2",
        type=float,
        default=250,
        help=(
            "Target piece area in um^2 (default: 250). "
            "Regions larger than this are split; smaller are kept or discarded."
        ),
    )
    split.add_argument(
        "--min-area-um2",
        type=float,
        default=1000,
        help=(
            "Minimum region area in um^2 before splitting (default: 1000). "
            "Regions smaller than this are discarded entirely."
        ),
    )

    # --- Features ---
    feat = parser.add_argument_group("Feature extraction")
    feat.add_argument(
        "--sam2",
        action="store_true",
        help="Extract SAM2 256-dim embeddings per piece (needs GPU)",
    )
    feat.add_argument(
        "--channels",
        default=None,
        help=(
            'Comma-separated channel indices for intensity features, '
            'e.g., "0,1,2". Default: all channels. Use to skip bad stains.'
        ),
    )

    args = parser.parse_args()
    detect_regions(args)


if __name__ == "__main__":
    main()
