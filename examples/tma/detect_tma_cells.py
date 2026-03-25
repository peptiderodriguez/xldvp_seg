#!/usr/bin/env python3
"""Detect TMA (tissue microarray) cores and segment cells within each core.

Finds circular TMA cores via morphological thresholding on a low-resolution
overview, assigns grid labels (A1, A2, ..., B1, B2, ...), then runs
Cellpose+SAM2 cell detection within each core. Each detection gets a core_id
for per-core downstream analysis.

Pipeline:
  1. Load CZI channels via strip-based reader
  2. Downsample detection channel, threshold to find tissue regions
  3. Filter by circularity and diameter range to keep only TMA cores
  4. Assign grid labels (row-major, A1..B2..) based on spatial position
  5. Init Cellpose + SAM2 per GPU (threaded)
  6. For each core: crop channels, detect cells, enrich with global coords
  7. Save detections JSON with core_id on each detection

Usage:
    python examples/tma/detect_tma_cells.py \\
        --czi-path /path/to/tma_slide.czi \\
        --detection-channel 0 \\
        --nuclear-channel 1 \\
        --min-core-diameter-um 400 \\
        --max-core-diameter-um 3000 \\
        --num-gpus 4 \\
        --output-dir /path/to/output
"""

import argparse
import gc
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import label as sklabel
from skimage.measure import regionprops

from segmentation.io.czi_loader import CZILoader, get_czi_metadata
from segmentation.utils.json_utils import atomic_json_dump
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DOWNSAMPLE = 16  # fixed downsample for core finding (fast, cores are large)
MIN_CIRCULARITY = 0.4  # permissive — cores can be oval, chipped, or irregular


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--czi-path", required=True, help="Path to TMA CZI file")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument(
        "--detection-channel",
        type=int,
        default=0,
        help="Channel index for core detection (brightest tissue channel, default: 0)",
    )
    p.add_argument(
        "--nuclear-channel",
        type=int,
        default=1,
        help="Nuclear channel for Cellpose (default: 1)",
    )
    p.add_argument(
        "--cyto-channel",
        type=int,
        default=None,
        help="Cytoplasm/membrane channel for Cellpose 2-channel mode (optional)",
    )
    p.add_argument(
        "--all-channels",
        type=str,
        default=None,
        help="Comma-separated channel indices for multi-channel features (e.g., '0,1,2,3')",
    )
    p.add_argument(
        "--min-core-diameter-um",
        type=float,
        default=400.0,
        help="Minimum TMA core diameter in um (default: 400). Standard punches: 600/1000/1500/2000 um.",
    )
    p.add_argument(
        "--max-core-diameter-um",
        type=float,
        default=3000.0,
        help="Maximum TMA core diameter in um (default: 3000). Covers all standard punch sizes.",
    )
    p.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs (default: 4)")
    p.add_argument(
        "--roi-padding-px",
        type=int,
        default=50,
        help="Pixel padding around core bboxes (default: 50)",
    )
    p.add_argument("--scene", type=int, default=0, help="CZI scene index (default: 0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TMA core finding
# ---------------------------------------------------------------------------


def find_tma_cores(channel_data, pixel_size_um, min_diam_um, max_diam_um):
    """Find circular TMA cores in a single channel via morphological thresholding.

    Args:
        channel_data: 2D uint16 array (full resolution)
        pixel_size_um: um per pixel at full resolution
        min_diam_um: minimum core diameter in um
        max_diam_um: maximum core diameter in um

    Returns:
        List of dicts with region_id, centroid_y, centroid_x, diameter_um,
        circularity, bbox (all in downsampled coordinates), plus full-res bbox.
    """
    from skimage.measure import block_reduce

    ds = DOWNSAMPLE
    ds_pixel = pixel_size_um * ds

    # Downsample for speed
    reduced = block_reduce(channel_data, block_size=(ds, ds), func=np.mean).astype(np.float32)

    # Otsu threshold on the downsampled image
    from skimage.filters import threshold_otsu

    thresh = threshold_otsu(reduced[reduced > 0]) if np.any(reduced > 0) else 0
    binary = reduced > thresh

    # Morphological cleanup: close small gaps, remove small objects
    kernel_px = max(3, int(round(50 / ds_pixel)))  # ~50 um closing kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    binary_u8 = binary.astype(np.uint8) * 255
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    # Label connected components
    labeled = sklabel(binary_u8 > 0)
    props = regionprops(labeled)

    # Convert diameter bounds to downsampled pixel area bounds
    min_area_px = np.pi * (min_diam_um / (2 * ds_pixel)) ** 2
    max_area_px = np.pi * (max_diam_um / (2 * ds_pixel)) ** 2

    cores = []
    for prop in props:
        # Area filter
        if prop.area < min_area_px * 0.5 or prop.area > max_area_px * 2.0:
            continue

        # Circularity filter: 4*pi*area / perimeter^2
        perimeter = prop.perimeter
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * prop.area / (perimeter**2)
        if circularity < MIN_CIRCULARITY:
            continue

        # Equivalent diameter in um
        equiv_diam_um = prop.equivalent_diameter_area * ds_pixel
        if equiv_diam_um < min_diam_um or equiv_diam_um > max_diam_um:
            continue

        cores.append(
            {
                "region_id": prop.label,
                "centroid_y_ds": prop.centroid[0],
                "centroid_x_ds": prop.centroid[1],
                "diameter_um": equiv_diam_um,
                "circularity": round(circularity, 3),
                "area_ds": prop.area,
                # Full-res bounding box (before padding)
                "bbox_y0": int(prop.bbox[0] * ds),
                "bbox_x0": int(prop.bbox[1] * ds),
                "bbox_y1": int(prop.bbox[2] * ds),
                "bbox_x1": int(prop.bbox[3] * ds),
            }
        )

    logger.info(
        f"Found {len(cores)} circular regions "
        f"(circularity >= {MIN_CIRCULARITY}, "
        f"diameter {min_diam_um}-{max_diam_um} um)"
    )
    return cores


def number_cores_grid(cores):
    """Assign grid labels (A1, A2, ..., B1, B2, ...) based on spatial position.

    Clusters cores into rows by y-coordinate, then sorts left-to-right within
    each row. Labels use letters for rows (A, B, C, ...) and numbers for
    columns (1, 2, 3, ...).
    """
    if not cores:
        return cores

    from scipy.cluster.hierarchy import fcluster, linkage

    # Cluster by y-coordinate to find rows
    y_coords = np.array([[c["centroid_y_ds"]] for c in cores])

    if len(cores) == 1:
        cores[0]["grid_label"] = "A1"
        cores[0]["grid_row"] = 0
        cores[0]["grid_col"] = 0
        return cores

    Z = linkage(y_coords, method="complete")
    # Use a distance threshold based on median core diameter
    median_diam_ds = np.median([c["area_ds"] ** 0.5 for c in cores])
    row_labels = fcluster(Z, t=median_diam_ds * 1.5, criterion="distance")

    # Sort rows by mean y-coordinate
    row_means = {}
    for c, rl in zip(cores, row_labels):
        row_means.setdefault(rl, []).append(c["centroid_y_ds"])
    row_order = sorted(row_means.keys(), key=lambda r: np.mean(row_means[r]))
    row_map = {old: new for new, old in enumerate(row_order)}

    # Assign grid labels
    for c, rl in zip(cores, row_labels):
        c["grid_row"] = row_map[rl]

    # Sort within each row by x-coordinate
    for row_idx in range(len(row_order)):
        row_cores = [c for c in cores if c["grid_row"] == row_idx]
        row_cores.sort(key=lambda c: c["centroid_x_ds"])
        for col_idx, c in enumerate(row_cores):
            c["grid_col"] = col_idx
            row_letter = chr(ord("A") + row_idx)
            c["grid_label"] = f"{row_letter}{col_idx + 1}"

    return cores


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------


def extract_core_rois(cores, x_start, y_start, full_width, full_height, padding_px=50):
    """Convert core bboxes to padded ROI dicts with global coordinates.

    Returns list of dicts with array-relative and global mosaic coordinates.
    """
    rois = []
    for core in cores:
        y0 = max(0, core["bbox_y0"] - padding_px)
        y1 = min(full_height, core["bbox_y1"] + padding_px)
        x0 = max(0, core["bbox_x0"] - padding_px)
        x1 = min(full_width, core["bbox_x1"] + padding_px)

        rois.append(
            {
                "region_id": core["region_id"],
                "grid_label": core["grid_label"],
                "grid_row": core["grid_row"],
                "grid_col": core["grid_col"],
                "diameter_um": core["diameter_um"],
                "circularity": core["circularity"],
                # Array-relative coords (for slicing channel data)
                "ay0": y0,
                "ax0": x0,
                "height": y1 - y0,
                "width": x1 - x0,
                # Global mosaic coords (for coordinate enrichment)
                "gx0": x0 + x_start,
                "gy0": y0 + y_start,
            }
        )

    return rois


# ---------------------------------------------------------------------------
# Per-ROI detection
# ---------------------------------------------------------------------------


def _percentile_normalize(arr, lo=1, hi=99.5):
    """Normalize uint16 array to uint8 using percentile clipping."""
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    vmin = np.percentile(nonzero, lo)
    vmax = np.percentile(nonzero, hi)
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    return (norm * 255).astype(np.uint8)


def detect_in_roi(roi, ch_data, nuc_ch, cyto_ch, strategy, models, pixel_size, slide_name):
    """Run cell detection on a single ROI crop.

    Returns list of enriched detection dicts with global coordinates and core_id.
    """
    ay0 = roi["ay0"]
    ax0 = roi["ax0"]
    h = roi["height"]
    w = roi["width"]

    # Slice channels for this ROI (numpy views, no copy)
    extra_channels = {}
    for ch_idx, full_arr in ch_data.items():
        extra_channels[ch_idx] = full_arr[ay0 : ay0 + h, ax0 : ax0 + w]

    # Build RGB display tile (nuc = blue, cyto = green if available)
    blue = _percentile_normalize(extra_channels.get(nuc_ch, np.zeros((h, w), dtype=np.uint16)))
    if cyto_ch is not None and cyto_ch in extra_channels:
        green = _percentile_normalize(extra_channels[cyto_ch])
    else:
        green = np.zeros((h, w), dtype=np.uint8)
    red = np.zeros((h, w), dtype=np.uint8)
    tile_rgb = np.stack([red, green, blue], axis=-1)

    # Detect cells
    label_array, detections = strategy.detect(
        tile=tile_rgb,
        models=models,
        pixel_size_um=pixel_size,
        extract_features=True,
        extra_channels=extra_channels,
    )

    if not detections:
        return []

    # Extract contours from label array
    contours_by_label = {}
    for det_idx in range(len(detections)):
        det_label = det_idx + 1
        binary = (label_array == det_label).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            contours_by_label[det_label] = cnt.squeeze().tolist()

    # Build detection dicts with global coordinates
    gx0 = roi["gx0"]
    gy0 = roi["gy0"]
    det_dicts = []
    for det_idx, det in enumerate(detections):
        feat = dict(det.features) if det.features else {}
        cx, cy = det.centroid
        feat["center"] = [cx, cy]
        feat["global_center"] = [float(gx0 + cx), float(gy0 + cy)]
        feat["global_center_um"] = [
            float((gx0 + cx) * pixel_size),
            float((gy0 + cy) * pixel_size),
        ]
        feat["tile_origin"] = [gx0, gy0]
        feat["slide_name"] = slide_name
        uid = f"{slide_name}_tma_{int(round(gx0 + cx))}_{int(round(gy0 + cy))}"
        feat["uid"] = uid
        feat["core_id"] = roi["grid_label"]
        feat["core_region_id"] = roi["region_id"]
        feat["score"] = det.score

        # Contour in global coords
        det_label = det_idx + 1
        local_contour = contours_by_label.get(det_label)
        global_contour = None
        if local_contour:
            if isinstance(local_contour[0], list):
                global_contour = [[pt[0] + gx0, pt[1] + gy0] for pt in local_contour]
            else:
                global_contour = [[local_contour[0] + gx0, local_contour[1] + gy0]]

        det_dicts.append(
            {
                "uid": uid,
                "global_center": feat["global_center"],
                "global_center_um": feat["global_center_um"],
                "tile_origin": feat["tile_origin"],
                "slide_name": slide_name,
                "center": feat["center"],
                "core_id": roi["grid_label"],
                "core_region_id": roi["region_id"],
                "score": det.score,
                "features": feat,
                "contour": global_contour,
            }
        )

    del label_array, detections, extra_channels, tile_rgb, contours_by_label
    return det_dicts


# ---------------------------------------------------------------------------
# Multi-GPU workers
# ---------------------------------------------------------------------------


def _init_gpu(gpu_id):
    """Initialize CellDetector on a specific GPU."""
    from segmentation.utils.device import set_device_for_worker

    device = set_device_for_worker(gpu_id)

    from segmentation.detection.cell_detector import CellDetector

    logger.info(f"[GPU-{gpu_id}] Loading models on {device}...")
    detector = CellDetector(device=str(device))
    _ = detector.models["cellpose"]
    _ = detector.models["sam2_predictor"]
    logger.info(f"[GPU-{gpu_id}] Models ready")
    return detector


def _gpu_worker(gpu_id, detector, assigned_rois, ch_data, nuc_ch, cyto_ch, pixel_size, slide_name):
    """Worker thread: process ROIs on a pre-initialized GPU."""
    from segmentation.detection.strategies.cell import CellStrategy
    from segmentation.utils.device import set_device_for_worker

    set_device_for_worker(gpu_id)

    strategy = CellStrategy(
        nuclear_channel=nuc_ch,
        membrane_channel=cyto_ch,
        extract_sam2_embeddings=True,
    )

    logger.info(f"[GPU-{gpu_id}] Processing {len(assigned_rois)} cores...")

    all_dets = []
    for i, roi in enumerate(assigned_rois):
        t_roi = time.time()
        roi_dets = detect_in_roi(
            roi, ch_data, nuc_ch, cyto_ch, strategy, detector.models, pixel_size, slide_name
        )
        all_dets.extend(roi_dets)
        dt = time.time() - t_roi
        logger.info(
            f"[GPU-{gpu_id}] Core {roi['grid_label']} "
            f"({roi['width']}x{roi['height']}): "
            f"{len(roi_dets)} cells ({dt:.1f}s)"
        )

    logger.info(f"[GPU-{gpu_id}] Done: {len(all_dets)} cells total")
    return all_dets


def process_cores_multigpu(rois, ch_data, nuc_ch, cyto_ch, pixel_size, slide_name, num_gpus):
    """Detect cells in all TMA cores using multiple GPUs in parallel."""
    # Init models on each GPU
    logger.info(f"Initializing Cellpose + SAM2 on {num_gpus} GPUs...")
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        detectors = list(pool.map(_init_gpu, range(num_gpus)))

    # Split ROIs round-robin across GPUs
    rois_per_gpu = [[] for _ in range(num_gpus)]
    for i, roi in enumerate(rois):
        rois_per_gpu[i % num_gpus].append(roi)

    logger.info(
        f"Distributing {len(rois)} cores across {num_gpus} GPUs: "
        + ", ".join(f"GPU-{g}={len(rois_per_gpu[g])}" for g in range(num_gpus))
    )

    all_dets = []
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = []
        for gpu_id in range(num_gpus):
            if not rois_per_gpu[gpu_id]:
                continue
            future = pool.submit(
                _gpu_worker,
                gpu_id,
                detectors[gpu_id],
                rois_per_gpu[gpu_id],
                ch_data,
                nuc_ch,
                cyto_ch,
                pixel_size,
                slide_name,
            )
            futures.append(future)

        for future in futures:
            all_dets.extend(future.result())

    # Cleanup GPU memory
    for det in detectors:
        det.cleanup()
    from segmentation.utils.device import empty_cache

    empty_cache()

    return all_dets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    setup_logging()
    t0 = time.time()

    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_name = czi_path.stem
    nuc_ch = args.nuclear_channel
    cyto_ch = args.cyto_channel
    det_ch = args.detection_channel

    # Determine which channels to load
    channels_to_load = sorted(set(ch for ch in [det_ch, nuc_ch, cyto_ch] if ch is not None))
    if args.all_channels:
        extra = [int(x.strip()) for x in args.all_channels.split(",")]
        channels_to_load = sorted(set(channels_to_load + extra))

    logger.info(f"=== TMA Core Detection + Cell Segmentation ({args.num_gpus} GPUs) ===")
    logger.info(f"CZI: {czi_path}")
    logger.info(f"Detection channel: {det_ch}, Nuclear: {nuc_ch}, Cyto: {cyto_ch}")
    logger.info(f"Core diameter range: {args.min_core_diameter_um}-{args.max_core_diameter_um} um")

    # ------------------------------------------------------------------
    # 1. Load CZI metadata + channels
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading CZI channels...")
    meta = get_czi_metadata(str(czi_path), scene=args.scene)
    pixel_size = meta["pixel_size_um"]
    logger.info(f"Pixel size: {pixel_size:.4f} um/px, channels: {meta['n_channels']}")

    loader = CZILoader(
        str(czi_path),
        load_to_ram=True,
        channels=channels_to_load,
        scene=args.scene,
    )
    full_width, full_height = loader.mosaic_size
    x_start, y_start = loader.mosaic_origin
    ch_data = loader._channel_data
    logger.info(
        f"Loaded {len(ch_data)} channels, slide {full_width}x{full_height}, "
        f"origin ({x_start}, {y_start})"
    )

    # ------------------------------------------------------------------
    # 2. Find TMA cores
    # ------------------------------------------------------------------
    logger.info("Step 2: Finding TMA cores...")
    if det_ch not in ch_data:
        logger.error(f"Detection channel {det_ch} not loaded. Available: {list(ch_data.keys())}")
        return

    cores = find_tma_cores(
        ch_data[det_ch],
        pixel_size,
        args.min_core_diameter_um,
        args.max_core_diameter_um,
    )

    if not cores:
        logger.error("No TMA cores found. Try adjusting --min/max-core-diameter-um.")
        return

    # ------------------------------------------------------------------
    # 3. Assign grid labels
    # ------------------------------------------------------------------
    logger.info("Step 3: Assigning grid labels...")
    cores = number_cores_grid(cores)

    for core in sorted(cores, key=lambda c: (c["grid_row"], c["grid_col"])):
        logger.info(
            f"  {core['grid_label']}: "
            f"diameter={core['diameter_um']:.0f} um, "
            f"circularity={core['circularity']:.2f}"
        )

    # ------------------------------------------------------------------
    # 4. Extract ROI bounding boxes
    # ------------------------------------------------------------------
    logger.info("Step 4: Extracting core ROIs...")
    rois = extract_core_rois(
        cores, x_start, y_start, full_width, full_height, padding_px=args.roi_padding_px
    )
    logger.info(f"{len(rois)} core ROIs extracted")

    # Save core metadata
    atomic_json_dump(cores, output_dir / "tma_cores.json")

    gc.collect()

    # ------------------------------------------------------------------
    # 5. Detect cells in each core
    # ------------------------------------------------------------------
    logger.info("Step 5: Detecting cells in TMA cores...")
    all_dets = process_cores_multigpu(
        rois, ch_data, nuc_ch, cyto_ch, pixel_size, slide_name, args.num_gpus
    )

    # ------------------------------------------------------------------
    # 6. Save detections
    # ------------------------------------------------------------------
    logger.info("Step 6: Saving detections...")
    out_path = output_dir / "tma_cell_detections.json"
    atomic_json_dump(all_dets, out_path)
    logger.info(f"Saved {len(all_dets)} detections to {out_path}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    core_counts = Counter(d["core_id"] for d in all_dets)
    logger.info("=== Summary ===")
    logger.info(f"TMA cores found: {len(cores)}")
    logger.info(f"Total cells: {len(all_dets)}")
    for label in sorted(core_counts.keys()):
        logger.info(f"  Core {label}: {core_counts[label]} cells")

    dt_total = time.time() - t0
    logger.info(f"Done in {dt_total:.0f}s ({dt_total / 60:.1f} min)")


if __name__ == "__main__":
    main()
