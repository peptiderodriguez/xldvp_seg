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
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from skimage.draw import polygon as skimage_polygon

try:
    import hdf5plugin  # noqa: F401 — register LZ4 codec before h5py reads
except ImportError:
    pass

from segmentation.analysis.nuclear_count import count_nuclei_for_tile
from segmentation.io.czi_loader import CZILoader, get_czi_metadata, resolve_channel_indices
from segmentation.utils.device import device_supports_gpu, get_default_device
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger, setup_logging

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
    parser.add_argument("--extract-deep-features", action="store_true",
                        help="Extract ResNet + DINOv2 features per nucleus (slower, higher dimensional)")
    parser.add_argument("--no-sam2", action="store_true",
                        help="Skip SAM2 embedding extraction (not recommended — SAM2 is default)")
    parser.add_argument("--gpu-device", type=int, default=0,
                        help="GPU device index (default: 0)")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile size in pixels (auto-detected from masks if omitted)")

    args = parser.parse_args()
    setup_logging(level="INFO")

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
    x_start, y_start = loader.mosaic_origin
    nuc_data = loader.get_channel_data(nuc_ch)
    if nuc_data is None:
        logger.error(f"Failed to load channel {nuc_ch}")
        sys.exit(1)
    logger.info(f"  Nuclear channel: {nuc_data.shape}, pixel_size={pixel_size_um:.4f} µm/px")

    # --- Load models ---
    import torch
    device_str = f"cuda:{args.gpu_device}" if args.gpu_device >= 0 and torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    cellpose_model = load_cellpose_cpsam(device=device)

    # SAM2 (default ON — same as cell detection)
    sam2_predictor = None
    manager = None
    if not args.no_sam2:
        from segmentation.models.manager import ModelManager
        logger.info("Loading SAM2 for nuclear embeddings...")
        manager = ModelManager(device=device)
        sam2_predictor = manager.sam2_predictor  # lazy-loads via property
        logger.info("  SAM2 ready")

    # Deep features (optional — ResNet + DINOv2)
    resnet_model, resnet_transform = None, None
    dinov2_model, dinov2_transform = None, None
    if args.extract_deep_features:
        if manager is None:
            from segmentation.models.manager import ModelManager
            manager = ModelManager(device=device)
        logger.info("Loading ResNet + DINOv2 for deep nuclear features...")
        resnet_model, resnet_transform = manager.get_resnet()
        dinov2_model, dinov2_transform = manager.get_dinov2()
        logger.info("  ResNet + DINOv2 ready")

    # --- Process tiles ---
    logger.info("Processing tiles...")
    t_start = time.time()

    # Build a map: for each detection, find which tile it belongs to
    # by matching tile_origin or global_center to tile coordinates
    det_tile_map = {}  # uid -> (tile_x, tile_y)
    n_from_tile_origin = 0
    n_from_global_center = 0
    for det in detections:
        tile_origin = det.get("tile_origin")
        if tile_origin:
            det_tile_map[det["uid"]] = (int(tile_origin[0]), int(tile_origin[1]))
            n_from_tile_origin += 1

    # Fallback: for detections without tile_origin, compute tile from global_center.
    # This handles detections from detect_regions_for_lmd.py and similar scripts.
    dets_without_tile = [det for det in detections if det["uid"] not in det_tile_map]
    if dets_without_tile:
        # Build set of known tile origins from tile directories
        tile_origins_set = {(tx, ty) for tx, ty, _, _ in tile_dirs}

        # Infer tile size from the tile grid spacing (or use --tile-size if provided)
        tile_size = args.tile_size
        if tile_size is None:
            # Find minimum positive gap between sorted unique tile X and Y coordinates
            tile_xs = sorted({tx for tx, _, _, _ in tile_dirs})
            tile_ys = sorted({ty for _, ty, _, _ in tile_dirs})
            x_gaps = [tile_xs[i + 1] - tile_xs[i] for i in range(len(tile_xs) - 1) if tile_xs[i + 1] > tile_xs[i]]
            y_gaps = [tile_ys[i + 1] - tile_ys[i] for i in range(len(tile_ys) - 1) if tile_ys[i + 1] > tile_ys[i]]
            if x_gaps or y_gaps:
                tile_size = min(x_gaps + y_gaps)
                logger.info(f"  Inferred tile_size={tile_size} from tile grid spacing")
            else:
                # Single tile — read its mask to get the size
                _, _, _, first_mask = tile_dirs[0]
                with h5py.File(str(first_mask), "r") as hf:
                    th, tw = hf["masks"].shape[:2]
                tile_size = max(th, tw)
                logger.info(f"  Single tile — inferred tile_size={tile_size} from mask shape")

        for det in dets_without_tile:
            gc = det.get("global_center")
            if gc is None:
                continue
            cx, cy = float(gc[0]), float(gc[1])
            # Compute expected tile origin by snapping to grid
            candidate_tx = int(cx // tile_size) * tile_size
            candidate_ty = int(cy // tile_size) * tile_size
            candidate = (candidate_tx, candidate_ty)
            if candidate in tile_origins_set:
                det_tile_map[det["uid"]] = candidate
                n_from_global_center += 1
            else:
                # Grid snap didn't match — find closest tile whose bounding box
                # could contain this point (within tile_size of origin)
                best = None
                best_dist = float("inf")
                for otx, oty in tile_origins_set:
                    if otx <= cx < otx + tile_size * 1.1 and oty <= cy < oty + tile_size * 1.1:
                        dist = (cx - otx) ** 2 + (cy - oty) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best = (otx, oty)
                if best is not None:
                    det_tile_map[det["uid"]] = best
                    n_from_global_center += 1

        if n_from_global_center > 0:
            logger.info(
                f"  Tile mapping: {n_from_tile_origin} from tile_origin, "
                f"{n_from_global_center} from global_center fallback"
            )
        unmapped = len(detections) - len(det_tile_map)
        if unmapped > 0:
            logger.warning(f"  {unmapped} detections could not be mapped to any tile")

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
    tiles_with_dets = sum(1 for tx, ty, _, _ in tile_dirs if (tx, ty) in tile_to_dets)
    logger.info(f"  {tiles_with_dets} tiles have detections (of {len(tile_dirs)} total)")

    for tile_x, tile_y, _tile_dir, mask_path in tile_dirs:
        tile_key = (tile_x, tile_y)
        det_indices = tile_to_dets.get(tile_key, [])
        if not det_indices:
            continue  # no detections in this tile

        # Load cell masks
        with h5py.File(str(mask_path), "r") as hf:
            cell_masks = hf["masks"][:]
        tile_h, tile_w = cell_masks.shape[:2]

        # For detections without mask_label (e.g., from detect_regions_for_lmd.py),
        # rasterize their outer_contour_global into the cell_masks array with
        # synthetic labels so count_nuclei_for_tile can process them.
        synthetic_label_map = {}  # det_idx -> synthetic mask label
        next_label = int(cell_masks.max()) + 1 if cell_masks.size > 0 else 1
        # Check if any contour-based detections need synthetic labels
        has_contour_dets = any(
            detections[di].get("tile_mask_label") is None
            and detections[di].get("mask_label") is None
            and (detections[di].get("outer_contour_global") is not None
                 or detections[di].get("contour_dilated_px") is not None)
            for di in det_indices
        )
        # Ensure dtype can hold new synthetic labels
        if has_contour_dets and np.issubdtype(cell_masks.dtype, np.integer):
            max_needed = next_label + len(det_indices)
            if max_needed > np.iinfo(cell_masks.dtype).max:
                cell_masks = cell_masks.astype(np.int32)
        for det_idx in det_indices:
            det = detections[det_idx]
            if det.get("tile_mask_label") is not None or det.get("mask_label") is not None:
                continue  # has a real mask label — skip
            contour = det.get("outer_contour_global") or det.get("contour_dilated_px")
            if contour is None:
                continue
            # Convert global contour to tile-local coordinates
            try:
                pts = np.array(contour, dtype=np.float64)
                if pts.ndim != 2 or pts.shape[1] != 2:
                    continue
                # Contour points are [x, y] in global pixel coords
                local_x = pts[:, 0] - tile_x
                local_y = pts[:, 1] - tile_y
                # Rasterize polygon into tile mask (row=y, col=x)
                rr, cc = skimage_polygon(local_y, local_x, shape=(tile_h, tile_w))
                if len(rr) == 0:
                    continue
                cell_masks[rr, cc] = next_label
                synthetic_label_map[det_idx] = next_label
                next_label += 1
            except Exception as e:
                logger.debug(f"Failed to rasterize contour for {det.get('uid')}: {e}")
                continue

        # Extract nuclear channel tile region
        # tile_x, tile_y are global pixel coordinates of tile origin
        # CZI data is in (H, W) = (rows, cols) format
        nuc_tile = nuc_data[tile_y - y_start:tile_y - y_start + tile_h, tile_x - x_start:tile_x - x_start + tile_w]

        if nuc_tile.shape != cell_masks.shape[:2]:
            logger.warning(
                f"Tile ({tile_x}, {tile_y}): shape mismatch nuc={nuc_tile.shape} "
                f"vs masks={cell_masks.shape}, cropping to common region"
            )
            h = min(nuc_tile.shape[0], cell_masks.shape[0])
            w = min(nuc_tile.shape[1], cell_masks.shape[1])
            nuc_tile = nuc_tile[:h, :w]
            cell_masks = cell_masks[:h, :w]

        # Set SAM2 image for this tile — nuclear channel only (not the cyto+nuc
        # composite used in cell detection; this gives nuclear-specific embeddings)
        if sam2_predictor is not None:
            from segmentation.analysis.nuclear_count import _percentile_normalize_to_uint8
            nuc_uint8 = _percentile_normalize_to_uint8(nuc_tile)
            nuc_rgb = np.stack([nuc_uint8] * 3, axis=-1)
            sam2_predictor.set_image(nuc_rgb)

        # Run nuclear counting with feature extraction
        results, n_nuc = count_nuclei_for_tile(
            cell_masks, nuc_tile, cellpose_model,
            pixel_size_um, args.min_nuclear_area,
            tile_x, tile_y,
            sam2_predictor=sam2_predictor,
            resnet_model=resnet_model,
            resnet_transform=resnet_transform,
            device=device,
            dinov2_model=dinov2_model,
            dinov2_transform=dinov2_transform,
        )
        total_nuclei += n_nuc

        # Free SAM2 cached features for this tile
        if sam2_predictor is not None:
            try:
                sam2_predictor.reset_predictor()
            except Exception:
                pass

        # Enrich detections with nuclear features
        for det_idx in det_indices:
            det = detections[det_idx]
            # Use synthetic label for contour-based detections, or real mask_label
            mask_label = synthetic_label_map.get(det_idx)
            if mask_label is None:
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
                f"  {tiles_processed}/{tiles_with_dets} tiles, "
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

    # --- Cleanup GPU resources ---
    if manager is not None:
        manager.cleanup()
    del cellpose_model
    from segmentation.utils.device import empty_cache
    empty_cache()


if __name__ == "__main__":
    main()
