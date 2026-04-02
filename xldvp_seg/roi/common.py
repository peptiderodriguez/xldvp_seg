"""Shared ROI utilities: bbox extraction, spatial numbering, tile/detection filtering,
and multi-GPU per-ROI detection.

All coordinates follow the project convention: [x, y] = [horizontal, vertical].
Array indexing uses [row, col] = [y, x].
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy import ndimage

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# Type alias used throughout this module
ROI = dict[str, Any]


# ---------------------------------------------------------------------------
# Bounding-box extraction
# ---------------------------------------------------------------------------


def extract_region_bboxes(
    region_labels: np.ndarray,
    downsample: int,
    x_start: int,
    y_start: int,
    full_width: int,
    full_height: int,
    padding_px: int = 50,
) -> list[ROI]:
    """Extract full-resolution bounding boxes from a downsampled region label array.

    Uses :func:`scipy.ndimage.find_objects` for tight per-region slices, then
    up-scales to full resolution, adds padding, and clips to slide boundaries.

    Args:
        region_labels: 2-D labeled array at downsampled resolution (from
            ``scipy.ndimage.label``).  Label 0 is background; 1..N are regions.
        downsample: Factor by which *region_labels* was downsampled relative to
            the full-resolution image.
        x_start: Mosaic-origin x (horizontal) in global slide coordinates.
        y_start: Mosaic-origin y (vertical) in global slide coordinates.
        full_width: Full-resolution slide width (pixels).
        full_height: Full-resolution slide height (pixels).
        padding_px: Pixel padding to add around each bounding box (full-res).

    Returns:
        List of ROI dicts, each containing:
            roi_id   (int) – region label (1-indexed)
            ay0, ax0 (int) – array-relative top-left corner (for slicing)
            height, width (int) – bounding-box dimensions (full-res)
            gx0, gy0 (int) – global mosaic coordinates of top-left corner
            area_px  (int) – approximate area in full-res pixels
    """
    slices = ndimage.find_objects(region_labels)
    rois: list[ROI] = []

    for region_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        n_px = int(np.sum(region_labels[sl] == region_id))
        if n_px == 0:
            continue

        # Convert downsampled coords to full-res array coords
        y0 = sl[0].start * downsample
        y1 = sl[0].stop * downsample
        x0 = sl[1].start * downsample
        x1 = sl[1].stop * downsample

        # Add padding, clip to slide boundaries
        y0 = max(0, y0 - padding_px)
        y1 = min(full_height, y1 + padding_px)
        x0 = max(0, x0 - padding_px)
        x1 = min(full_width, x1 + padding_px)

        rois.append(
            {
                "roi_id": region_id,
                # Array-relative coords (for slicing channel data)
                "ay0": y0,
                "ax0": x0,
                "height": y1 - y0,
                "width": x1 - x0,
                # Global mosaic coords (for coordinate enrichment)
                "gx0": x0 + x_start,
                "gy0": y0 + y_start,
                # Approximate area in full-res pixels
                "area_px": n_px * downsample * downsample,
            }
        )

    logger.info(
        "Extracted %d ROI bboxes from %d labeled regions (ds=%d, padding=%d)",
        len(rois),
        int(region_labels.max()) if region_labels.size > 0 else 0,
        downsample,
        padding_px,
    )
    return rois


# ---------------------------------------------------------------------------
# Spatial numbering
# ---------------------------------------------------------------------------


def number_rois_spatial(
    rois: list[ROI],
    pixel_size_um: float = 1.0,
    row_tolerance_um: float = 500.0,
    grid_mode: bool = False,
) -> list[ROI]:
    """Sort ROIs by spatial position and assign sequential or grid-based IDs.

    Numbering proceeds top-to-bottom, left-to-right.  ROIs whose y-centers
    are within *row_tolerance_um* of each other are considered part of the
    same row.

    Args:
        rois: List of ROI dicts (must contain ``gy0``, ``gx0``, ``height``,
            ``width``).  Modified in-place **and** returned.
        pixel_size_um: Pixel size in micrometres (used with *row_tolerance_um*).
        row_tolerance_um: Maximum y-center separation (in um) for two ROIs to
            be assigned to the same row.
        grid_mode: If *True*, assign ``grid_label`` strings like ``"A1"``,
            ``"A2"``, ``"B1"`` instead of plain sequential integers.

    Returns:
        The same *rois* list, each dict now having ``roi_id`` (int, 1-indexed)
        and, when *grid_mode* is True, ``grid_label`` (str).
    """
    if not rois:
        return rois

    row_tolerance_px = row_tolerance_um / pixel_size_um

    # Compute y-center for each ROI
    centers = [(r["gy0"] + r["height"] / 2.0, r["gx0"] + r["width"] / 2.0, r) for r in rois]
    # Sort by y-center first
    centers.sort(key=lambda c: c[0])

    # Group into rows
    rows: list[list[ROI]] = []
    current_row: list[tuple[float, float, ROI]] = [centers[0]]

    for entry in centers[1:]:
        row_mean_y = sum(e[0] for e in current_row) / len(current_row)
        if abs(entry[0] - row_mean_y) <= row_tolerance_px:
            current_row.append(entry)
        else:
            rows.append([e[2] for e in sorted(current_row, key=lambda e: e[1])])
            current_row = [entry]
    # Flush last row
    rows.append([e[2] for e in sorted(current_row, key=lambda e: e[1])])

    # Assign IDs
    seq_id = 1
    for row_idx, row in enumerate(rows):
        row_letter = chr(ord("A") + row_idx) if row_idx < 26 else f"R{row_idx}"
        for col_idx, roi in enumerate(row):
            roi["roi_id"] = seq_id
            if grid_mode:
                roi["grid_label"] = f"{row_letter}{col_idx + 1}"
            seq_id += 1

    logger.info(
        "Numbered %d ROIs into %d rows (tolerance=%.0f um, grid=%s)",
        len(rois),
        len(rows),
        row_tolerance_um,
        grid_mode,
    )
    return rois


# ---------------------------------------------------------------------------
# Tile filtering
# ---------------------------------------------------------------------------


def filter_tiles_by_rois(
    tiles: list[dict[str, int]],
    rois: list[ROI],
    tile_size: int,
) -> list[dict[str, int]]:
    """Keep only tiles that overlap at least one ROI bounding box.

    Args:
        tiles: List of tile dicts, each with ``"x"`` and ``"y"`` keys giving
            the top-left corner in global coordinates.
        rois: List of ROI dicts (must contain ``gx0``, ``gy0``, ``width``,
            ``height``).
        tile_size: Side length of each square tile (pixels).

    Returns:
        Subset of *tiles* that overlap at least one ROI.
    """
    if not rois:
        return []

    kept: list[dict[str, int]] = []
    for tile in tiles:
        tx0 = tile["x"]
        ty0 = tile["y"]
        tx1 = tx0 + tile_size
        ty1 = ty0 + tile_size

        for roi in rois:
            rx0 = roi["gx0"]
            ry0 = roi["gy0"]
            rx1 = rx0 + roi["width"]
            ry1 = ry0 + roi["height"]

            # Axis-aligned bounding-box overlap test
            if tx0 < rx1 and tx1 > rx0 and ty0 < ry1 and ty1 > ry0:
                kept.append(tile)
                break  # tile overlaps at least one ROI — keep it

    logger.info("Tile filter: %d / %d tiles overlap ROIs", len(kept), len(tiles))
    return kept


# ---------------------------------------------------------------------------
# Detection filtering
# ---------------------------------------------------------------------------


def filter_detections_by_roi_mask(
    detections: list[dict],
    region_labels: np.ndarray,
    downsample: int,
    x_start: int,
    y_start: int,
) -> list[dict]:
    """Filter detections to those whose centroid falls inside a labeled region.

    Each kept detection receives an ``roi_id`` field set to the label value at
    its centroid position.

    Args:
        detections: List of detection dicts (must have ``global_center``
            ``[x, y]``).
        region_labels: 2-D labeled array at downsampled resolution.
        downsample: Downsample factor for *region_labels*.
        x_start: Mosaic-origin x in global coordinates.
        y_start: Mosaic-origin y in global coordinates.

    Returns:
        Filtered list of detection dicts (new list; dicts are mutated in-place
        to add ``roi_id``).
    """
    ds_h, ds_w = region_labels.shape
    kept: list[dict] = []

    for det in detections:
        gx, gy = det["global_center"]
        # Global pixel → downsampled label array indices
        lx = int(round((gx - x_start) / downsample))
        ly = int(round((gy - y_start) / downsample))

        if 0 <= ly < ds_h and 0 <= lx < ds_w:
            label_val = int(region_labels[ly, lx])
            if label_val > 0:
                det["roi_id"] = label_val
                kept.append(det)

    logger.info(
        "Detection filter: %d / %d detections inside labeled regions",
        len(kept),
        len(detections),
    )
    return kept


# ---------------------------------------------------------------------------
# Multi-GPU per-ROI detection
# ---------------------------------------------------------------------------


def _detect_single_roi(
    roi: ROI,
    channel_data: dict[int, np.ndarray],
    pixel_size: float,
    cell_type: str,
    strategy: Any,
    models: dict,
) -> list[dict]:
    """Detect cells in a single ROI crop using a pre-initialised strategy and models.

    Slices channel data to the ROI bounding box, runs detection, and converts
    local coordinates to global.
    """
    import cv2

    ay0, ax0, h, w = roi["ay0"], roi["ax0"], roi["height"], roi["width"]
    gx0, gy0 = roi["gx0"], roi["gy0"]
    roi_id = roi["roi_id"]

    # Slice channels (numpy views — no copy)
    extra_channels: dict[int, np.ndarray] = {}
    for ch_idx, full_arr in channel_data.items():
        extra_channels[ch_idx] = full_arr[ay0 : ay0 + h, ax0 : ax0 + w]

    # Build a simple RGB from the first three available channels (or repeat)
    ch_indices = sorted(extra_channels.keys())
    rgb_planes: list[np.ndarray] = []
    for i in range(3):
        idx = ch_indices[i] if i < len(ch_indices) else ch_indices[0]
        plane = extra_channels[idx].astype(np.float32)
        nonzero = plane[plane > 0]
        if nonzero.size > 0:
            p1, p99 = np.percentile(nonzero, [1, 99])
            if p99 > p1:
                plane = np.clip((plane - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
            else:
                plane = np.zeros((h, w), dtype=np.uint8)
        else:
            plane = np.zeros((h, w), dtype=np.uint8)
        rgb_planes.append(plane)
    tile_rgb = np.stack(rgb_planes, axis=-1)

    # Run detection
    label_array, dets = strategy.detect(
        tile=tile_rgb,
        models=models,
        pixel_size_um=pixel_size,
        extract_features=True,
        extra_channels=extra_channels,
    )

    if not dets:
        return []

    # Extract contours from label array (ROI-local coords)
    contours_by_label: dict[int, list] = {}
    for det_idx in range(len(dets)):
        det_label = det_idx + 1
        binary = (label_array == det_label).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            contours_by_label[det_label] = cnt.squeeze().tolist()

    # Convert Detection objects to enriched dicts with global coordinates
    det_dicts: list[dict] = []
    for det_idx, det in enumerate(dets):
        feat = dict(det.features) if det.features else {}
        cx, cy = det.centroid
        feat["center"] = [cx, cy]
        feat["global_center"] = [float(gx0 + cx), float(gy0 + cy)]
        feat["global_center_um"] = [
            float((gx0 + cx) * pixel_size),
            float((gy0 + cy) * pixel_size),
        ]
        feat["tile_origin"] = [gx0, gy0]
        feat["roi_id"] = roi_id

        uid = f"roi{roi_id}_{cell_type}_{int(round(gx0 + cx))}_{int(round(gy0 + cy))}"
        feat["uid"] = uid

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
                "center": feat["center"],
                "roi_id": roi_id,
                "score": det.score,
                "features": feat,
                "contour": global_contour,
            }
        )

    return det_dicts


def detect_in_rois(
    rois: list[ROI],
    channel_data: dict[int, np.ndarray],
    pixel_size: float,
    cell_type: str = "cell",
    num_gpus: int = 1,
    channel_map: dict[str, int] | None = None,
    extract_sam2: bool = True,
    extract_deep_features: bool = False,
) -> list[dict]:
    """Run multi-GPU per-ROI cell detection.

    ROIs are distributed round-robin across *num_gpus* GPU workers using
    :class:`~concurrent.futures.ThreadPoolExecutor`.  Each thread initialises
    its own :class:`CellDetector` and detection strategy, processes its
    assigned ROI crops, and converts coordinates to global.

    Args:
        rois: List of ROI dicts (from :func:`extract_region_bboxes`).
        channel_data: ``{channel_index: np.ndarray}`` full-slide 2-D arrays
            (read-only, shared across threads).
        pixel_size: Micrometres per pixel.
        cell_type: Detection strategy name (e.g. ``"cell"``, ``"nmj"``).
        num_gpus: Number of GPUs to use.
        channel_map: Optional ``{role: channel_index}`` mapping (e.g.
            ``{"cyto": 1, "nuc": 4}``).
        extract_sam2: Whether to extract SAM2 embeddings.
        extract_deep_features: Whether to extract ResNet/DINOv2 features.

    Returns:
        Flat list of detection dicts from all ROIs, each containing
        ``roi_id``, ``global_center``, ``features``, ``contour``, etc.
    """
    if not rois:
        logger.warning("detect_in_rois called with 0 ROIs — returning empty list")
        return []

    num_gpus = max(1, min(num_gpus, len(rois)))
    rois_per_gpu: list[list[ROI]] = [[] for _ in range(num_gpus)]
    for i, roi in enumerate(rois):
        rois_per_gpu[i % num_gpus].append(roi)

    logger.info(
        "Distributing %d ROIs across %d GPUs: %s",
        len(rois),
        num_gpus,
        ", ".join(f"GPU-{g}={len(rois_per_gpu[g])}" for g in range(num_gpus)),
    )

    def _worker(gpu_id: int, assigned: list[ROI]) -> list[dict]:
        from xldvp_seg.detection.cell_detector import CellDetector
        from xldvp_seg.processing.strategy_factory import create_strategy
        from xldvp_seg.utils.device import set_device_for_worker

        device = set_device_for_worker(gpu_id)

        # Init CellDetector ONCE per worker (avoid model reload per ROI)
        detector = CellDetector(device=str(device))

        strategy_params: dict[str, Any] = {}
        if channel_map:
            strategy_params.update(channel_map)
        strategy = create_strategy(
            cell_type,
            strategy_params=strategy_params,
            extract_sam2_embeddings=extract_sam2,
            extract_deep_features=extract_deep_features,
        )

        all_dets: list[dict] = []
        try:
            for roi in assigned:
                t0 = time.time()
                roi_dets = _detect_single_roi(
                    roi,
                    channel_data,
                    pixel_size,
                    cell_type,
                    strategy,
                    detector.models,
                )
                all_dets.extend(roi_dets)
                dt = time.time() - t0
                logger.info(
                    "[GPU-%d] ROI %d (%dx%d): %d cells (%.1fs)",
                    gpu_id,
                    roi["roi_id"],
                    roi["width"],
                    roi["height"],
                    len(roi_dets),
                    dt,
                )
        finally:
            detector.cleanup()
        return all_dets

    all_detections: list[dict] = []
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = []
        for gpu_id in range(num_gpus):
            if rois_per_gpu[gpu_id]:
                futures.append(pool.submit(_worker, gpu_id, rois_per_gpu[gpu_id]))
        try:
            for future in futures:
                all_detections.extend(future.result())
        except Exception:
            from xldvp_seg.utils.device import empty_cache

            empty_cache()
            raise

    logger.info("detect_in_rois: %d total detections from %d ROIs", len(all_detections), len(rois))
    return all_detections
