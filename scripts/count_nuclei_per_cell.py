#!/usr/bin/env python3
"""Count nuclei per cell for existing detection runs.

Loads cell detection JSON + per-tile HDF5 masks, runs Cellpose (cpsam) in
single-channel mode on the nuclear stain, and enriches each detection with
nuclear count and morphology features.

Usage:
    PYTHONPATH=$REPO python scripts/count_nuclei_per_cell.py \\
        --detections /path/to/cell_detections.json \\
        --czi-path /path/to/slide.czi \\
        --tiles-dir /path/to/tiles \\
        --nuclear-channel 3 \\
        --output /path/to/cell_detections_with_nuclei.json

    # Or with channel-spec:
    PYTHONPATH=$REPO python scripts/count_nuclei_per_cell.py \\
        --detections /path/to/cell_detections.json \\
        --czi-path /path/to/slide.czi \\
        --tiles-dir /path/to/tiles \\
        --channel-spec "nuc=Hoechst" \\
        --output /path/to/cell_detections_with_nuclei.json
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import h5py

try:
    import hdf5plugin  # noqa: F401 — register LZ4 codec before h5py reads
except ImportError:
    pass

from segmentation.analysis.nuclear_count import count_nuclei_for_tile
from segmentation.io.czi_loader import CZILoader, get_czi_metadata, resolve_channel_indices
from segmentation.utils.json_utils import fast_json_load, atomic_json_dump
from segmentation.utils.logging import get_logger
from segmentation.utils.device import get_default_device, device_supports_gpu

logger = get_logger(__name__)


def load_cellpose_cpsam(device=None):
    """Load the cpsam Cellpose model for nuclear segmentation."""
    from cellpose.models import CellposeModel

    if device is None:
        device = get_default_device()
    logger.info(f"Loading Cellpose cpsam model on {device}...")
    model = CellposeModel(
        pretrained_model='cpsam',
        gpu=device_supports_gpu(device),
        device=device,
    )
    return model


def find_tile_dirs(tiles_dir: Path) -> list:
    """Find all tile directories containing HDF5 masks.

    Returns list of (tile_x, tile_y, tile_dir_path) sorted by position.
    """
    tiles = []
    for d in sorted(tiles_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("tile_"):
            continue
        # Parse tile_X_Y
        parts = d.name.split("_")
        if len(parts) >= 3:
            try:
                tx, ty = int(parts[1]), int(parts[2])
                # Check for HDF5 mask file
                h5_files = list(d.glob("*_masks.h5")) + list(d.glob("*_masks.hdf5"))
                if h5_files:
                    tiles.append((tx, ty, d, h5_files[0]))
            except ValueError:
                continue
    return tiles


def build_uid_to_tile_label_map(detections: list, tiles_dir: Path) -> dict:
    """Build mapping from detection UID to (tile_x, tile_y, mask_label).

    Reads per-tile feature JSONs to find the tile_mask_label for each detection.
    Falls back to parsing the UID for tile coordinates.
    """
    uid_map = {}

    for det in detections:
        uid = det.get("uid", "")
        # Try to get tile info from detection
        tile_origin = det.get("tile_origin")
        mask_label = det.get("tile_mask_label", det.get("mask_label"))

        if tile_origin and mask_label:
            uid_map[uid] = {
                "tile_x": int(tile_origin[0]),
                "tile_y": int(tile_origin[1]),
                "mask_label": int(mask_label),
            }
        else:
            # Parse from UID: {slide}_{celltype}_{global_x}_{global_y}
            # We'll match to tiles by checking which tile contains the global center
            center = det.get("global_center")
            if center:
                uid_map[uid] = {
                    "global_x": int(center[0]),
                    "global_y": int(center[1]),
                    "mask_label": int(mask_label) if mask_label else None,
                }

    return uid_map


def main():
    parser = argparse.ArgumentParser(
        description="Count nuclei per cell using Cellpose on the nuclear channel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--detections", required=True, type=Path,
                        help="Cell detection JSON file")
    parser.add_argument("--czi-path", required=True, type=Path,
                        help="CZI file path")
    parser.add_argument("--tiles-dir", required=True, type=Path,
                        help="Tiles directory with HDF5 masks")
    parser.add_argument("--nuclear-channel", type=int, default=None,
                        help="Nuclear channel index")
    parser.add_argument("--channel-spec", default=None,
                        help='Channel spec for nuclear channel, e.g., "nuc=Hoechst"')
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSON path (enriched detections)")
    parser.add_argument("--min-nuclear-area", type=int, default=50,
                        help="Minimum nuclear area in pixels (default: 50)")
    parser.add_argument("--gpu-device", type=int, default=0,
                        help="GPU device index (default: 0)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile size in pixels (auto-detected from masks if omitted)")

    args = parser.parse_args()

    # --- Resolve nuclear channel ---
    nuc_ch = args.nuclear_channel
    if nuc_ch is None and args.channel_spec is not None:
        spec_str = args.channel_spec
        if "=" in spec_str:
            _, spec_str = spec_str.split("=", 1)
        spec_str = spec_str.strip()
        meta = get_czi_metadata(str(args.czi_path))
        resolved = resolve_channel_indices(
            czi_metadata=meta,
            marker_specs=[spec_str],
            filename=args.czi_path.name,
        )
        nuc_ch = resolved[spec_str]
        logger.info(f"Resolved nuclear channel '{spec_str}' → ch{nuc_ch}")

    if nuc_ch is None:
        parser.error("One of --nuclear-channel or --channel-spec is required")

    # --- Load detections ---
    logger.info(f"Loading detections: {args.detections}")
    detections = fast_json_load(str(args.detections))
    if isinstance(detections, dict):
        detections = detections.get("detections", [])
    logger.info(f"  {len(detections):,} detections loaded")

    # --- Build UID → detection index map ---
    uid_to_idx = {det.get("uid", ""): i for i, det in enumerate(detections)}

    # --- Find tiles ---
    tile_dirs = find_tile_dirs(args.tiles_dir)
    logger.info(f"Found {len(tile_dirs)} tiles with masks")

    if not tile_dirs:
        logger.error("No tile directories with masks found")
        sys.exit(1)

    # --- Load CZI nuclear channel ---
    logger.info(f"Loading nuclear channel {nuc_ch} from CZI...")
    loader = CZILoader(str(args.czi_path), load_to_ram=True, channel=nuc_ch)
    pixel_size_um = loader.get_pixel_size()
    nuc_data = loader.get_channel_data(nuc_ch)
    if nuc_data is None:
        logger.error(f"Failed to load channel {nuc_ch}")
        sys.exit(1)
    logger.info(f"  Nuclear channel: {nuc_data.shape}, pixel_size={pixel_size_um:.4f} µm/px")

    # --- Load Cellpose ---
    device_str = f"cuda:{args.gpu_device}" if args.gpu_device >= 0 else "cpu"
    import torch
    if not torch.cuda.is_available():
        device_str = "cpu"
    cellpose_model = load_cellpose_cpsam(device=device_str)

    # --- Process tiles ---
    logger.info("Processing tiles...")
    t_start = time.time()

    # Build a map: for each detection, find which tile it belongs to
    # by matching tile_origin or global_center to tile coordinates
    det_tile_map = {}  # uid -> (tile_x, tile_y)
    for det in detections:
        tile_origin = det.get("tile_origin")
        if tile_origin:
            det_tile_map[det["uid"]] = (int(tile_origin[0]), int(tile_origin[1]))

    # Group detections by tile
    tile_to_dets = {}  # (tile_x, tile_y) -> [det_indices]
    for uid, (tx, ty) in det_tile_map.items():
        key = (tx, ty)
        if key not in tile_to_dets:
            tile_to_dets[key] = []
        if uid in uid_to_idx:
            tile_to_dets[key].append(uid_to_idx[uid])

    total_nuclei = 0
    tiles_processed = 0
    cells_enriched = 0

    for tile_x, tile_y, tile_dir, mask_path in tile_dirs:
        tile_key = (tile_x, tile_y)
        det_indices = tile_to_dets.get(tile_key, [])
        if not det_indices:
            continue  # no detections in this tile

        # Load cell masks
        with h5py.File(str(mask_path), "r") as hf:
            cell_masks = hf["masks"][:]
        tile_h, tile_w = cell_masks.shape[:2]

        # Extract nuclear channel tile region
        # tile_x, tile_y are global pixel coordinates of tile origin
        # CZI data is in (H, W) = (rows, cols) format
        nuc_tile = nuc_data[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]

        if nuc_tile.shape != cell_masks.shape:
            # Edge tile may be smaller
            h = min(nuc_tile.shape[0], cell_masks.shape[0])
            w = min(nuc_tile.shape[1], cell_masks.shape[1])
            nuc_tile = nuc_tile[:h, :w]
            cell_masks = cell_masks[:h, :w]

        # Run nuclear counting
        results, n_nuc = count_nuclei_for_tile(
            cell_masks, nuc_tile, cellpose_model,
            pixel_size_um, args.min_nuclear_area,
            tile_x, tile_y,
        )
        total_nuclei += n_nuc

        # Enrich detections with nuclear features
        for det_idx in det_indices:
            det = detections[det_idx]
            mask_label = det.get("tile_mask_label", det.get("mask_label"))
            if mask_label is None:
                continue
            mask_label = int(mask_label)

            if mask_label in results:
                nuc_feats = results[mask_label]
                features = det.get("features", {})
                for key, val in nuc_feats.items():
                    features[key] = val
                det["features"] = features
                cells_enriched += 1

        tiles_processed += 1
        if tiles_processed % 50 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"  {tiles_processed}/{len(tile_dirs)} tiles, "
                f"{cells_enriched:,} cells enriched, "
                f"{total_nuclei:,} nuclei, "
                f"{elapsed:.0f}s elapsed"
            )

    elapsed = time.time() - t_start
    logger.info(
        f"Done: {tiles_processed} tiles, {cells_enriched:,} cells enriched, "
        f"{total_nuclei:,} total nuclei in {elapsed:.0f}s"
    )

    # --- Summary statistics ---
    counts = []
    for det in detections:
        n = det.get("features", {}).get("n_nuclei")
        if n is not None:
            counts.append(n)

    if counts:
        counter = Counter(counts)
        logger.info("Nuclear count distribution:")
        for k in sorted(counter.keys()):
            pct = 100 * counter[k] / len(counts)
            logger.info(f"  {k} nuclei: {counter[k]:,} cells ({pct:.1f}%)")

    # --- Write output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(detections, str(args.output))
    logger.info(f"Wrote enriched detections: {args.output}")


if __name__ == "__main__":
    main()
