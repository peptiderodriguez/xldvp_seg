#!/usr/bin/env python3
"""Post-hoc mask refinement for an existing detection run.

Applies ``xldvp_seg.utils.mask_cleanup.refine_mask_intensity`` to every
per-tile HDF5 mask, recomputes ``contour_px`` / ``contour_um`` /
``area`` / ``area_um2`` / ``solidity`` / ``circularity`` from the refined mask,
and writes a new detections JSON with the updated geometry.

Works for any cell type that has per-tile ``{cell_type}_masks.h5`` files
under ``<run_dir>/tiles/tile_*/``. Generic — no MK-specific assumptions.

The refinement is non-uniform and adaptive:
  - Computes a 90th-percentile intensity threshold from the *interior* of
    each mask.
  - Iteratively peels boundary pixels that are brighter than that threshold
    (i.e. bleed into bright/empty space).
  - Size guard: reverts if refinement removes >50% of original area.
  - Unchanged for cells where no boundary is brighter than interior.

Usage:
    python scripts/refine_detection_masks.py \\
        --run-dir /path/to/<slide>/<timestamp>_100pct \\
        --czi-path /path/to/slide.czi \\
        --cell-type mk \\
        [--workers 16] \\
        [--output /path/to/<cell_type>_detections_refined.json]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401 — LZ4 support for segmentation.h5
except ImportError:
    pass

import cv2

from xldvp_seg.io.czi_loader import get_loader
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger, setup_logging
from xldvp_seg.utils.mask_cleanup import recompute_mask_features, refine_mask_intensity

logger = get_logger(__name__)


def _find_tile_mask_files(run_dir: Path, cell_type: str) -> list[Path]:
    """Find every {cell_type}_masks.h5 under run_dir/tiles/tile_*/."""
    tiles_dir = run_dir / "tiles"
    return sorted(tiles_dir.glob(f"tile_*/{cell_type}_masks.h5"))


def _parse_tile_origin(mask_file: Path) -> tuple[int, int]:
    """Tile directory name encodes origin as ``tile_{x}_{y}``."""
    parts = mask_file.parent.name.split("_")
    return int(parts[1]), int(parts[2])


def _load_tile_rgb(
    channel_array: np.ndarray,
    x_start: int,
    y_start: int,
    tile_x: int,
    tile_y: int,
    tile_h: int,
    tile_w: int,
):
    """Slice a tile from the in-RAM channel array. Works for 2D grayscale uint16
    and 3D RGB uint8 transparently (numpy slicing preserves trailing axes).
    """
    rel_x = tile_x - x_start
    rel_y = tile_y - y_start
    return channel_array[rel_y : rel_y + tile_h, rel_x : rel_x + tile_w]


def _refine_one_tile(task: dict) -> dict:
    """Process one tile's masks. Returns per-detection updates keyed by UID.

    ThreadPoolExecutor worker — shares the pre-loaded channel_array via closure.
    """
    mask_file = Path(task["mask_file"])
    channel_array = task["channel_array"]
    x_start = task["x_start"]
    y_start = task["y_start"]
    pixel_size_um = task["pixel_size_um"]
    tile_x, tile_y = task["tile_origin"]
    det_uids = task["det_uids"]  # {mask_label: uid} for this tile's detections

    # Load mask label array
    with h5py.File(mask_file, "r") as hf:
        if "masks" in hf:
            masks = hf["masks"][:]
        elif "labels" in hf:
            masks = hf["labels"][:]
        else:
            return {"updates": {}, "tile": f"{tile_x}_{tile_y}", "skipped": "no masks dataset"}
        if masks.ndim == 3 and masks.shape[0] == 1:
            masks = masks[0]

    tile_h, tile_w = masks.shape[:2]
    tile = _load_tile_rgb(channel_array, x_start, y_start, tile_x, tile_y, tile_h, tile_w)
    if tile is None or tile.size == 0:
        return {"updates": {}, "tile": f"{tile_x}_{tile_y}", "skipped": "empty tile"}

    updates: dict[str, dict] = {}
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    for label in unique_labels:
        uid = det_uids.get(int(label))
        if uid is None:
            continue

        mask = masks == label
        orig_area = int(mask.sum())
        if orig_area == 0:
            continue

        # Apply intensity-based refinement with user-configurable thresholds.
        # Size guard returns the ORIGINAL mask when refinement would drop
        # below min_area_fraction. An empty refined mask here would be
        # unexpected — warn and skip.
        refined = refine_mask_intensity(
            mask,
            tile,
            opening_radius=task["opening_radius"],
            bright_percentile=task["bright_percentile"],
            peel_iterations=task["peel_iterations"],
            min_area_fraction=task["min_area_fraction"],
        )
        if not refined.any():
            logger.warning(
                f"Refinement emptied mask for {uid} in tile ({tile_x},{tile_y}); "
                "keeping original"
            )
            continue

        # Recompute shape features on the refined mask
        feats = recompute_mask_features(refined, pixel_size_um=pixel_size_um)

        # Extract contour in tile-local coords, shift to global (stage) coords
        contour_local = _contour_from_mask(refined)
        if contour_local is None:
            continue
        # contour_local is [X, Y] tile-local; shift to global stage coords
        contour_global = contour_local.astype(np.float64).copy()
        contour_global[:, 0] += tile_x
        contour_global[:, 1] += tile_y

        # Global centroid
        cx_local, cy_local = feats["centroid_xy"]
        global_cx = cx_local + tile_x
        global_cy = cy_local + tile_y

        updates[uid] = {
            "contour_px": contour_global.astype(np.int32).tolist(),
            "area": feats["area_px"],
            "area_um2": feats["area_um2"],
            "solidity": feats["solidity"],
            "circularity": feats["circularity"],
            "refined_area_fraction": feats["area_px"] / orig_area if orig_area > 0 else 1.0,
            "global_center": [float(global_cx), float(global_cy)],
        }

    # Note: we do NOT rewrite the HDF5 mask file. The refined contour/features
    # are returned as updates to the detection JSON only. Downstream tools that
    # read masks from HDF5 (e.g. regenerate_html.py's tile+mask mode) will still
    # see the unrefined mask — pass the refined detections JSON directly via
    # --detections to use the refined contour_px instead.
    return {"updates": updates, "tile": f"{tile_x}_{tile_y}", "n_refined": len(updates)}


def _contour_from_mask(binary_mask: np.ndarray):
    """Largest external contour from a binary mask, [X, Y] tile-local coords."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, help="Detection output dir (contains tiles/)")
    parser.add_argument("--czi-path", required=True, help="CZI file for intensity reference")
    parser.add_argument(
        "--cell-type", required=True, help="Cell type (for *_detections.json / *_masks.h5)"
    )
    parser.add_argument(
        "--channel", type=int, default=0, help="CZI channel for intensity (default 0)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON (default <run_dir>/<cell_type>_detections_refined.json)",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Parallel tile workers (default 16)"
    )
    # Refinement tuning — exposed to cope with varied staining intensity
    parser.add_argument(
        "--bright-percentile",
        type=float,
        default=70.0,
        help="Percentile of mask interior used as the peel threshold. Boundary "
        "pixels brighter than this percentile of the interior are removed. "
        "Lower = more aggressive. Default 70 (was 90 in mask_cleanup default — "
        "too lenient for masks with preexisting white-bleed in interior).",
    )
    parser.add_argument(
        "--peel-iterations",
        type=int,
        default=10,
        help="Number of boundary-peeling iterations. Default 10 (was 3 in "
        "mask_cleanup default — stops too early on wide bleed).",
    )
    parser.add_argument(
        "--min-area-fraction",
        type=float,
        default=0.3,
        help="Size guard: if refinement reduces mask below this fraction of "
        "original area, revert to original. Default 0.3 (was 0.5 — allow up "
        "to 70%% removal for heavily-bleeding masks).",
    )
    parser.add_argument(
        "--opening-radius",
        type=int,
        default=3,
        help="Morphological opening radius in pixels before peeling. 0 disables.",
    )
    args = parser.parse_args()
    setup_logging(level="INFO")

    run_dir = Path(args.run_dir)
    det_path = run_dir / f"{args.cell_type}_detections.json"
    if not det_path.exists():
        logger.error(f"Detections JSON not found: {det_path}")
        sys.exit(1)

    out_path = (
        Path(args.output) if args.output else run_dir / f"{args.cell_type}_detections_refined.json"
    )

    logger.info(f"Loading detections from {det_path}")
    detections = fast_json_load(str(det_path))
    logger.info(f"  {len(detections):,} detections")

    # Load CZI channel into RAM once (aicspylibczi isn't safe for concurrent
    # access, so we load upfront and share the numpy array via closure to
    # ThreadPool workers — threads release the GIL during cv2/numpy ops).
    logger.info(f"Loading CZI channel {args.channel} to RAM...")
    loader = get_loader(args.czi_path, load_to_ram=True, channel=args.channel)
    channel_array = loader.get_channel_data(args.channel)
    pixel_size_um = loader.get_pixel_size()
    x_start, y_start = loader.x_start, loader.y_start
    logger.info(
        f"  Pixel size: {pixel_size_um} um/px, "
        f"array shape={channel_array.shape} dtype={channel_array.dtype} "
        f"({channel_array.nbytes / 1e9:.1f} GB)"
    )

    # Group detections by tile_origin for batch per-tile processing
    by_tile: dict[tuple[int, int], dict[int, str]] = {}
    for det in detections:
        to = det.get("tile_origin")
        label = det.get("mask_label") or det.get("tile_mask_label")
        uid = det.get("uid")
        if to is None or label is None or uid is None:
            continue
        tile_key = (int(to[0]), int(to[1]))
        by_tile.setdefault(tile_key, {})[int(label)] = uid
    logger.info(f"  {len(by_tile)} tiles with mask labels")

    # Build task list — one per tile. channel_array is shared by reference across
    # threads (no copy, no pickling).
    mask_files = _find_tile_mask_files(run_dir, args.cell_type)
    logger.info(f"  {len(mask_files)} mask files on disk")

    tasks = []
    for mf in mask_files:
        tx, ty = _parse_tile_origin(mf)
        det_uids = by_tile.get((tx, ty))
        if not det_uids:
            continue
        tasks.append(
            {
                "mask_file": str(mf),
                "channel_array": channel_array,
                "x_start": x_start,
                "y_start": y_start,
                "pixel_size_um": pixel_size_um,
                "tile_origin": (tx, ty),
                "det_uids": det_uids,
                "bright_percentile": args.bright_percentile,
                "peel_iterations": args.peel_iterations,
                "min_area_fraction": args.min_area_fraction,
                "opening_radius": args.opening_radius,
            }
        )
    logger.info(f"Processing {len(tasks)} tiles with {args.workers} threads...")

    # Run refinement (ThreadPool — cv2 + numpy release the GIL)
    all_updates: dict[str, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_refine_one_tile, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            res = future.result()
            done += 1
            all_updates.update(res.get("updates", {}))
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                logger.info(
                    f"  [{done}/{len(tasks)}] refined={len(all_updates):,} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s"
                )
    elapsed = time.time() - t0
    logger.info(
        f"Refined {len(all_updates):,} masks in {elapsed:.1f}s ({len(tasks)/elapsed:.1f} tiles/s)"
    )

    # Merge updates into detections
    area_deltas = []
    for det in detections:
        uid = det.get("uid")
        upd = all_updates.get(uid)
        if upd is None:
            continue
        orig_area = det.get("area", 0)
        det.update(upd)
        # Also update um-scale contour if present
        if "contour_um" in det:
            det["contour_um"] = (
                np.array(det["contour_px"], dtype=np.float64) * pixel_size_um
            ).tolist()
        if orig_area > 0 and upd.get("area"):
            area_deltas.append(upd["area"] / orig_area)

    if area_deltas:
        area_deltas = np.array(area_deltas)
        logger.info(
            f"Area change: mean={area_deltas.mean():.3f}x, median={np.median(area_deltas):.3f}x, "
            f"min={area_deltas.min():.3f}x, max={area_deltas.max():.3f}x"
        )

    logger.info(f"Writing refined detections to {out_path}")
    atomic_json_dump(detections, out_path)
    logger.info(f"Done. {len(all_updates):,} / {len(detections):,} detections refined.")


if __name__ == "__main__":
    main()
