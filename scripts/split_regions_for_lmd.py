#!/usr/bin/env python3
"""split_regions_for_lmd.py — Post-process pipeline detections to split large
regions into equal-area sub-pieces for LMD export.

This is a POST-PROCESSOR, not a standalone detector. It expects detections from
a prior pipeline run (e.g., --cell-type nmj on the NfL channel). The pipeline
handles tiling, SHM, multi-GPU, dedup, contours — this script just splits
detections that exceed a target area.

Workflow:
  1. Load detection JSON from a pipeline run
  2. For each detection above target_area_um2:
     a. Load its HDF5 mask from the tile directory
     b. Watershed-split into N = floor(area / target) equal-area pieces
     c. Extract contour per piece in global coordinates
  3. Detections below target_area are kept as-is
  4. Write output detection JSON (compatible with run_lmd_export.py)

Usage:
  # Step 1: Run detection with the pipeline (NMJ strategy works for any
  # bright-region thresholding — NfL, BTX, etc.)
  python run_segmentation.py --cell-type nmj \\
      --channel-spec "detect=NfL" --output-dir /path/to/output

  # Step 2: Post-process to split large detections into 200 um^2 pieces
  python scripts/split_regions_for_lmd.py \\
      --detections /path/to/output/nmj_detections.json \\
      --tiles-dir /path/to/output/tiles \\
      --target-area-um2 200 \\
      --output /path/to/output/nfl_pieces.json
"""

import os
import sys
import argparse

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

try:
    import hdf5plugin  # noqa: F401 — register LZ4 codec before h5py
except ImportError:
    pass
import h5py

from segmentation.utils.json_utils import fast_json_load, atomic_json_dump
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Region splitting (same watershed logic as before)
# ---------------------------------------------------------------------------

def split_region(mask: np.ndarray, n_pieces: int, seed: int = 42) -> list:
    """Split a binary mask into n_pieces roughly equal-area sub-regions.

    Uses k-means seeded watershed on the distance transform.
    """
    if n_pieces <= 1:
        return [mask]

    dist = distance_transform_edt(mask)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [mask]

    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    rng = np.random.default_rng(seed)
    n_pieces = min(n_pieces, len(coords))
    if n_pieces <= 1:
        return [mask]

    init_idx = rng.choice(len(coords), n_pieces, replace=False)
    centroids = coords[init_idx].copy()

    for _ in range(50):
        tree = cKDTree(centroids)
        _, labels_km = tree.query(coords)
        new_centroids = np.array([
            coords[labels_km == i].mean(axis=0) if (labels_km == i).any() else centroids[i]
            for i in range(n_pieces)
        ])
        if np.allclose(centroids, new_centroids, atol=0.5):
            break
        centroids = new_centroids

    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (cx, cy) in enumerate(centroids):
        ry, rx = int(round(cy)), int(round(cx))
        ry = max(0, min(mask.shape[0] - 1, ry))
        rx = max(0, min(mask.shape[1] - 1, rx))
        if not mask[ry, rx]:
            sq = (xs - rx) ** 2 + (ys - ry) ** 2
            nearest = sq.argmin()
            ry, rx = int(ys[nearest]), int(xs[nearest])
        markers[ry, rx] = i + 1

    ws_labels = watershed(-dist, markers, mask=mask)
    return [ws_labels == i for i in range(1, n_pieces + 1) if (ws_labels == i).any()]


def extract_contour_global(piece: np.ndarray, x_offset: int, y_offset: int) -> list:
    """Extract largest external contour in global [x, y] coordinates."""
    contours, _ = cv2.findContours(
        piece.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    pts_global = pts.copy().astype(int)
    pts_global[:, 0] += x_offset
    pts_global[:, 1] += y_offset
    return pts_global.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process pipeline detections: split large regions into "
            "equal-area sub-pieces for LMD export."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument("--detections", required=True, type=Path,
                        help="Detection JSON from pipeline run (e.g., nmj_detections.json)")
    parser.add_argument("--tiles-dir", required=True, type=Path,
                        help="Tiles directory with HDF5 masks")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path")
    parser.add_argument("--target-area-um2", type=float, default=None,
                        help="Target area per piece in um^2. If not set, computed from --target-area-percentile.")
    parser.add_argument("--target-area-percentile", type=float, default=75.0,
                        help="Use this percentile of detection area as both the target piece size "
                             "and minimum cutoff (default: 75 = p75). Detections below this are "
                             "discarded, detections above are split into pieces of this size. "
                             "Ignored if --target-area-um2 is set explicitly.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for watershed seeding")
    parser.add_argument("--cell-type", default="nmj",
                        help="Cell type label in mask filenames (default: nmj)")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load detections ---
    logger.info(f"Loading detections: {args.detections}")
    detections = fast_json_load(str(args.detections))
    if isinstance(detections, dict):
        detections = detections.get("detections", [])
    logger.info(f"  {len(detections):,} detections loaded")

    # --- Get pixel size from first detection ---
    pixel_size_um = None
    for det in detections[:10]:
        ps = det.get("pixel_size_um") or det.get("features", {}).get("pixel_size_um")
        if ps:
            pixel_size_um = float(ps)
            break
    if pixel_size_um is None:
        # Fallback: compute from global_center vs global_center_um
        for det in detections[:10]:
            gc = det.get("global_center")
            gc_um = det.get("global_center_um")
            if gc and gc_um and gc[0] > 0:
                pixel_size_um = gc_um[0] / gc[0]
                break
    if pixel_size_um is None:
        from segmentation.utils.config import _LEGACY_PIXEL_SIZE_UM
        pixel_size_um = _LEGACY_PIXEL_SIZE_UM
        logger.warning(f"Could not detect pixel size, using legacy fallback {pixel_size_um}")
    logger.info(f"  Pixel size: {pixel_size_um:.4f} um/px")

    px2 = pixel_size_um ** 2

    # Determine target area (= piece size AND minimum cutoff)
    if args.target_area_um2 is not None:
        target_area_um2 = args.target_area_um2
    else:
        # Compute from percentile of detection area distribution
        all_areas = [
            det.get("features", {}).get("area_um2", det.get("features", {}).get("area", 0) * px2)
            for det in detections
        ]
        all_areas = [a for a in all_areas if a > 0]
        if all_areas:
            target_area_um2 = float(np.percentile(all_areas, args.target_area_percentile))
            logger.info(f"  Area p{args.target_area_percentile:.0f} = {target_area_um2:.1f} um2 "
                        f"(from {len(all_areas):,} detections)")
        else:
            target_area_um2 = 200.0
            logger.warning("No valid areas found, using default 200 um2")

    # p75 is both the target piece size AND the minimum cutoff:
    # detections < p75 are discarded, detections > p75 are split into p75-sized pieces
    min_area_um2 = target_area_um2
    target_area_px = max(1, int(target_area_um2 / px2))
    min_area_px = target_area_px
    logger.info(f"  Target piece size: {target_area_um2:.1f} um2 ({target_area_px} px)")
    logger.info(f"  Min area cutoff: {min_area_um2:.1f} um2 (same as target — discard smaller)")

    # --- Group detections by tile ---
    by_tile = {}  # (tile_x, tile_y) -> [det_indices]
    for i, det in enumerate(detections):
        origin = det.get("tile_origin")
        if origin:
            key = (int(origin[0]), int(origin[1]))
            by_tile.setdefault(key, []).append(i)

    logger.info(f"  {len(by_tile)} tiles with detections")

    # --- Process ---
    output_detections = []
    n_kept = 0
    n_split = 0
    n_pieces_total = 0
    n_too_small = 0

    for (tile_x, tile_y), det_indices in sorted(by_tile.items()):
        # Find HDF5 mask file for this tile
        tile_dir = args.tiles_dir / f"tile_{tile_x}_{tile_y}"
        mask_files = list(tile_dir.glob(f"{args.cell_type}_masks.h5")) + \
                     list(tile_dir.glob(f"{args.cell_type}_masks.hdf5"))
        if not mask_files:
            # Keep detections as-is if no mask file
            for i in det_indices:
                output_detections.append(detections[i])
                n_kept += 1
            continue

        with h5py.File(str(mask_files[0]), "r") as hf:
            if "masks" in hf:
                masks = hf["masks"][:]
            elif "labels" in hf:
                masks = hf["labels"][:]
            else:
                logger.warning(f"No masks/labels dataset in {mask_files[0]}")
                for i in det_indices:
                    output_detections.append(detections[i])
                    n_kept += 1
                continue

        for i in det_indices:
            det = detections[i]
            features = det.get("features", {})
            area_um2 = features.get("area_um2", features.get("area", 0) * px2)

            # Too small — discard (below target)
            if area_um2 < min_area_um2 * 0.90:
                n_too_small += 1
                continue

            # Within ±10% of target — keep as-is (already the right size)
            if abs(area_um2 - target_area_um2) <= target_area_um2 * 0.10:
                output_detections.append(det)
                n_kept += 1
                continue

            # Between 90% and 150% of target but not within ±10% — too big to keep, too small to split
            if area_um2 < target_area_um2 * 1.5:
                n_too_small += 1
                continue

            # Large detection — split
            mask_label = det.get("tile_mask_label", det.get("mask_label"))
            if mask_label is None or int(mask_label) == 0:
                output_detections.append(det)
                n_kept += 1
                continue

            cell_mask = (masks == int(mask_label))
            if not cell_mask.any():
                output_detections.append(det)
                n_kept += 1
                continue

            area_px = int(cell_mask.sum())
            n_pieces = max(1, area_px // target_area_px)

            if n_pieces <= 1:
                output_detections.append(det)
                n_kept += 1
                continue

            # Watershed split
            pieces = split_region(cell_mask, n_pieces, seed=args.seed + i)
            n_split += 1

            # Extract slide stem from detection (full slide name, not just first UID token)
            slide_stem = det.get("slide_name", "")
            if not slide_stem:
                uid = det.get("uid", "")
                # UIDs are {slide}_{celltype}_{x}_{y} — strip last 3 tokens
                parts = uid.rsplit("_", 3)
                slide_stem = parts[0] if len(parts) >= 4 else uid
            cell_type = args.cell_type

            for piece_id, piece in enumerate(pieces):
                if not piece.any():
                    continue

                contour = extract_contour_global(piece, x_offset=tile_x, y_offset=tile_y)
                if not contour:
                    continue

                piece_ys, piece_xs = np.where(piece)
                cx = float(piece_xs.mean()) + tile_x
                cy = float(piece_ys.mean()) + tile_y
                p_area_px = int(piece.sum())
                p_area_um2 = p_area_px * px2

                # Discard pieces not within ±10% of target size
                if abs(p_area_um2 - target_area_um2) > target_area_um2 * 0.10:
                    continue

                uid = f"{slide_stem}_{cell_type}_{int(round(cx))}_{int(round(cy))}"

                new_det = {
                    "uid": uid,
                    "global_center": [int(round(cx)), int(round(cy))],
                    "global_center_um": [round(cx * pixel_size_um, 3),
                                         round(cy * pixel_size_um, 3)],
                    "tile_origin": [tile_x, tile_y],
                    "features": {
                        "area": p_area_px,
                        "area_um2": round(p_area_um2, 3),
                        "parent_uid": det.get("uid"),
                        "parent_area_um2": round(area_um2, 3),
                        "piece_id": piece_id,
                        "n_pieces": len(pieces),
                        "pixel_size_um": pixel_size_um,
                    },
                    "outer_contour_global": contour,
                }
                output_detections.append(new_det)
                n_pieces_total += 1

    # --- Summary ---
    logger.info(f"Results: {n_kept:,} kept, {n_split:,} split into "
                f"{n_pieces_total:,} pieces, {n_too_small:,} too small")
    logger.info(f"Total output detections: {len(output_detections):,}")

    # --- Write ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(output_detections, str(args.output))
    logger.info(f"Wrote: {args.output}")

    if output_detections:
        areas = [d["features"]["area_um2"] for d in output_detections]
        logger.info(f"Area: min={min(areas):.1f} max={max(areas):.1f} "
                    f"mean={sum(areas)/len(areas):.1f} um2")


if __name__ == "__main__":
    sys.exit(main() or 0)
