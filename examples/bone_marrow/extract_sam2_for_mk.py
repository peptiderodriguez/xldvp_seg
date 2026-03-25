#!/usr/bin/env python3
"""Extract mask-aware SAM2 embeddings for MK detections from CZI files.

For each cell, computes a **masked average pool** over the SAM2 image encoder's
spatial feature map, using only the pixels within the cell's contour. This gives
a 256D embedding that represents the cell's actual appearance, excluding background.

Works from the combined detection JSON (center_x/center_y/contour_yx) directly.
Groups detections into virtual tiles for efficient CZI I/O and SAM2 encoding.

Multi-GPU: spawns one worker per GPU. SLURM array distributes slides across nodes.

Usage:
    # SLURM array: each node processes 4 slides across 4 GPUs
    python scripts/extract_sam2_for_mk.py extract \
        --detections all_mks_with_rejected3_full.json \
        --czi-dir /path/to/czi/files \
        --output-dir sam2_embeddings/ \
        --node-index 0 --num-nodes 4

    # All slides locally
    python scripts/extract_sam2_for_mk.py extract \
        --detections all_mks_with_rejected3_full.json \
        --czi-dir /path/to/czi/files \
        --output-dir sam2_embeddings/

    # Merge into detection JSON
    python scripts/extract_sam2_for_mk.py merge \
        --embeddings-dir sam2_embeddings/ \
        --target all_mks_with_rejected3_full.json \
        --output all_mks_with_rejected3_full_sam2.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SAM2_DIM = 256
TILE_SIZE = 2048
TILE_OVERLAP = 256  # Overlap to ensure cells near tile edges are fully covered


def _parse_uid_coords(uid):
    """Parse (x, y) coordinates from UID format: {slide}_mk_{x}_{y}."""
    parts = uid.rsplit("_", 2)
    if len(parts) >= 3:
        try:
            return float(parts[-2]), float(parts[-1])
        except ValueError:
            pass
    return None, None


def _parse_uid_slide(uid):
    """Parse slide name from UID format: {slide}_mk_{x}_{y}."""
    parts = uid.rsplit("_mk_", 1)
    return parts[0] if len(parts) == 2 else ""


def load_detections_by_slide(detections_path, training_data_path=None):
    """Load detection JSON(s), group by slide.

    Args:
        detections_path: Combined detection JSON with center_x/center_y/contour_yx
        training_data_path: Optional training data JSON. Training samples get
            coordinates parsed from UIDs and use centroid-based extraction.
    """
    print(f"Loading detections from {detections_path}...")
    with open(detections_path) as f:
        dets = json.load(f)
    print(f"  {len(dets)} total detections")

    by_slide = defaultdict(list)
    uid_set = set()

    for det in dets:
        slide = det.get("slide", "")
        if not slide:
            slide = _parse_uid_slide(det.get("uid", ""))
        by_slide[slide].append(det)
        uid_set.add(det.get("uid", ""))

    # Load training data samples (for SAM2 extraction of training negatives/positives)
    if training_data_path:
        print(f"\nLoading training samples from {training_data_path}...")
        with open(training_data_path) as f:
            td = json.load(f)

        training_samples = td.get("training_samples", [])
        n_added = 0
        n_skip = 0
        for ts in training_samples:
            uid = ts.get("uid", "")
            if uid in uid_set:
                n_skip += 1
                continue  # Already in combined JSON

            # Parse coordinates from UID
            cx, cy = _parse_uid_coords(uid)
            if cx is None:
                continue

            slide = _parse_uid_slide(uid)
            if not slide:
                continue

            # Create a minimal detection dict (centroid-based, no contour)
            by_slide[slide].append(
                {
                    "uid": uid,
                    "center_x": cx,
                    "center_y": cy,
                    "slide": slide,
                    # No contour_yx → extraction will use centroid fallback
                }
            )
            uid_set.add(uid)
            n_added += 1

        print(f"  Training: {n_added} added, {n_skip} already in combined JSON")

    slides = sorted(by_slide.keys())
    print(f"  {len(slides)} slides:")
    for s in slides:
        n_contour = sum(1 for d in by_slide[s] if d.get("contour_yx"))
        n_centroid = len(by_slide[s]) - n_contour
        print(f"    {s}: {len(by_slide[s])} cells ({n_contour} masked, {n_centroid} centroid)")
    return by_slide, slides


def create_virtual_tiles(
    detections, tile_size=TILE_SIZE, overlap=TILE_OVERLAP, use_czi_coords=False
):
    """Group detections into virtual tiles.

    Each detection is assigned to exactly one tile. Tiles are sized to ensure
    the full contour of each cell fits within its assigned tile.

    Args:
        use_czi_coords: If True, use _czi_cx/_czi_cy/_czi_contour_yx
            (CZI absolute coordinates). Otherwise use center_x/center_y/contour_yx.

    Returns list of (x0, y0, x1, y1, [det_indices]) tuples.
    """
    if not detections:
        return []

    cx_key = "_czi_cx" if use_czi_coords else "center_x"
    cy_key = "_czi_cy" if use_czi_coords else "center_y"
    contour_key = "_czi_contour_yx" if use_czi_coords else "contour_yx"

    # Get bounding box for each detection from contour
    bboxes = []
    for det in detections:
        contour = det.get(contour_key, [])
        if contour:
            ys = [p[0] for p in contour]
            xs = [p[1] for p in contour]
            bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        else:
            cx = float(det.get(cx_key, 0))
            cy = float(det.get(cy_key, 0))
            bboxes.append((cx - 50, cy - 50, cx + 50, cy + 50))

    # Global bounding box with padding
    all_x0 = min(b[0] for b in bboxes) - 256
    all_y0 = min(b[1] for b in bboxes) - 256
    all_x1 = max(b[2] for b in bboxes) + 256
    all_y1 = max(b[3] for b in bboxes) + 256

    step = tile_size - overlap
    tiles = []

    x = all_x0
    while x < all_x1:
        y = all_y0
        while y < all_y1:
            tx0, ty0 = int(x), int(y)
            tx1, ty1 = tx0 + tile_size, ty0 + tile_size

            # Find detections whose centroid falls in this tile
            indices = []
            for i, det in enumerate(detections):
                cx = float(det.get(cx_key, 0))
                cy = float(det.get(cy_key, 0))
                if tx0 <= cx < tx1 and ty0 <= cy < ty1:
                    indices.append(i)

            if indices:
                tiles.append((tx0, ty0, tx1, ty1, indices))

            y += step
        x += step

    # Deduplicate: assign each detection to first tile containing its centroid
    assigned = set()
    deduped = []
    for tx0, ty0, tx1, ty1, indices in tiles:
        new_indices = [i for i in indices if i not in assigned]
        if new_indices:
            assigned.update(new_indices)
            deduped.append((tx0, ty0, tx1, ty1, new_indices))

    return deduped


def rasterize_contour_to_embedding(contour_yx, tile_x0, tile_y0, tile_size, emb_h, emb_w):
    """Rasterize a contour into a binary mask at embedding resolution.

    Args:
        contour_yx: List of [y, x] points in global coordinates
        tile_x0, tile_y0: Tile origin in global coordinates
        tile_size: Tile size in pixels
        emb_h, emb_w: Embedding spatial dimensions (typically 64x64)

    Returns:
        Binary mask of shape (emb_h, emb_w), True inside contour
    """
    import cv2

    if not contour_yx:
        return np.zeros((emb_h, emb_w), dtype=bool)

    # Convert global contour to tile-local coordinates
    local_points = []
    for y, x in contour_yx:
        lx = (x - tile_x0) / tile_size * emb_w
        ly = (y - tile_y0) / tile_size * emb_h
        local_points.append([lx, ly])

    # Rasterize with OpenCV fillPoly
    mask = np.zeros((emb_h, emb_w), dtype=np.uint8)
    pts = np.array(local_points, dtype=np.float32).reshape(-1, 1, 2)
    pts_int = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts_int.reshape(-1, 2)], 1)

    return mask.astype(bool)


def masked_average_pool(feature_map, mask):
    """Average pool a feature map within a binary mask.

    Args:
        feature_map: (C, H, W) tensor or array
        mask: (H, W) boolean array

    Returns:
        (C,) averaged feature vector, or zeros if mask is empty
    """
    if not np.any(mask):
        return np.zeros(feature_map.shape[0], dtype=np.float32)

    # feature_map: (C, H, W), mask: (H, W)
    masked = feature_map[:, mask]  # (C, N_pixels)
    return masked.mean(axis=1).astype(np.float32)


def _find_sam2_checkpoint():
    """Find SAM2 checkpoint."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "checkpoints" / "sam2.1_hiera_large.pt",
        repo_root / "checkpoints" / "sam2.1_hiera_l.pt",
        Path.home() / "sam2_checkpoints" / "sam2.1_hiera_large.pt",
    ]
    for cp in candidates:
        if cp.exists():
            return cp
    print("ERROR: SAM2 checkpoint not found. Searched:")
    for cp in candidates:
        print(f"  {cp}")
    sys.exit(1)


def _worker_extract_slide(args_tuple):
    """Worker function for ProcessPoolExecutor. Runs on one GPU."""
    slide_name, detections, czi_dir, gpu_id, output_dir, checkpoint = args_tuple

    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if already done (require non-empty and matching count)
    out_file = Path(output_dir) / f"sam2_{slide_name}.json"
    if out_file.exists() and out_file.stat().st_size > 10:
        try:
            with open(out_file) as f:
                existing = json.load(f)
            if len(existing) >= len(detections):
                print(f"[GPU {gpu_id}] {slide_name}: already extracted ({len(existing)}), skipping")
                return slide_name, len(existing), 0.0
        except (json.JSONDecodeError, ValueError):
            pass  # Corrupt file, re-extract

    t0 = time.time()
    print(f"[GPU {gpu_id}] {slide_name}: {len(detections)} cells, loading SAM2...")

    from aicspylibczi import CziFile
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Find CZI
    czi_dir = Path(czi_dir)
    czi_file = czi_dir / f"{slide_name}.czi"
    if not czi_file.exists():
        candidates = list(czi_dir.glob(f"*{slide_name}*.czi"))
        czi_file = candidates[0] if candidates else None
    if not czi_file or not czi_file.exists():
        print(f"[GPU {gpu_id}] WARNING: No CZI for {slide_name}")
        return slide_name, 0, 0.0

    # Load SAM2
    sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(checkpoint),
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2_model)
    czi = CziFile(str(czi_file))

    # Get mosaic bounding box — detection coords are 0-indexed (pipeline subtracts origin),
    # but read_mosaic expects CZI absolute coords. Add (bbox.x, bbox.y) offset.
    bbox = czi.get_mosaic_bounding_box()
    x_offset, y_offset = bbox.x, bbox.y
    print(f"[GPU {gpu_id}] {slide_name}: CZI bbox x={bbox.x} y={bbox.y} w={bbox.w} h={bbox.h}")

    # Offset all detection coordinates to CZI absolute space
    # Some rescued slide cells lack center_x/center_y — parse from UID
    for det in detections:
        cx = det.get("center_x")
        cy = det.get("center_y")
        if cx is None or cy is None:
            cx, cy = _parse_uid_coords(det.get("uid", ""))
            if cx is not None:
                det["center_x"] = cx
                det["center_y"] = cy

        if "center_x" in det:
            det["_czi_cx"] = float(det["center_x"]) + x_offset
            det["_czi_cy"] = float(det["center_y"]) + y_offset
        if "contour_yx" in det and det["contour_yx"]:
            det["_czi_contour_yx"] = [[y + y_offset, x + x_offset] for y, x in det["contour_yx"]]

    # Create virtual tiles in CZI absolute coordinates
    tiles = create_virtual_tiles(detections, tile_size=TILE_SIZE, use_czi_coords=True)
    print(f"[GPU {gpu_id}] {slide_name}: {len(tiles)} tiles for {len(detections)} cells")

    embeddings = {}

    for ti, (tx0, ty0, tx1, ty1, indices) in enumerate(tiles):
        tile_w, tile_h = tx1 - tx0, ty1 - ty0

        # Load CZI region (channel 0)
        try:
            tile_data = czi.read_mosaic(
                region=(tx0, ty0, tile_w, tile_h),
                C=0,
                scale_factor=1.0,
            )
            tile_img = np.squeeze(tile_data)

            # Percentile normalize to uint8
            if tile_img.dtype == np.uint16:
                nz = tile_img[tile_img > 0]
                if nz.size > 0:
                    p2, p98 = np.percentile(nz, [2, 98])
                    if p98 > p2:
                        tile_img = np.clip(
                            (tile_img.astype(np.float32) - p2) / (p98 - p2) * 255,
                            0,
                            255,
                        ).astype(np.uint8)
                    else:
                        tile_img = np.zeros_like(tile_img, dtype=np.uint8)
                else:
                    tile_img = np.zeros_like(tile_img, dtype=np.uint8)
            elif tile_img.dtype != np.uint8:
                tile_img = np.clip(tile_img, 0, 255).astype(np.uint8)

            if tile_img.ndim == 2:
                tile_img = np.stack([tile_img] * 3, axis=-1)

        except Exception as e:
            print(f"[GPU {gpu_id}] WARNING: tile ({tx0},{ty0}): {e}")
            continue

        # Encode tile with SAM2
        predictor.set_image(tile_img)
        emb_shape = predictor._features["image_embed"].shape
        emb_h, emb_w = emb_shape[2], emb_shape[3]

        # Get the full embedding feature map as numpy: (256, H, W)
        emb_map = predictor._features["image_embed"][0].cpu().numpy()  # (C, H, W)

        # Extract masked-average-pooled embedding for each cell
        # Use CZI-offset coordinates (_czi_*) since tiles are in CZI absolute space
        for idx in indices:
            det = detections[idx]
            uid = det.get("uid", "")
            contour = det.get("_czi_contour_yx", [])

            if contour:
                # Rasterize contour at embedding resolution
                mask = rasterize_contour_to_embedding(contour, tx0, ty0, TILE_SIZE, emb_h, emb_w)
                emb = masked_average_pool(emb_map, mask)
            else:
                # Fallback: centroid point sampling
                cx = float(det.get("_czi_cx", 0))
                cy = float(det.get("_czi_cy", 0))
                local_x, local_y = cx - tx0, cy - ty0
                emb_x = min(max(int(local_x * emb_w / tile_w), 0), emb_w - 1)
                emb_y = min(max(int(local_y * emb_h / tile_h), 0), emb_h - 1)
                emb = emb_map[:, emb_y, emb_x].astype(np.float32)

            embeddings[uid] = [float(v) for v in emb]

        predictor.reset_predictor()

        if (ti + 1) % 10 == 0 or ti == 0:
            elapsed = time.time() - t0
            print(f"[GPU {gpu_id}] {slide_name}: {ti+1}/{len(tiles)} tiles ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[GPU {gpu_id}] {slide_name}: {len(embeddings)} embeddings in {elapsed:.1f}s")

    # Save
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(embeddings, f)

    del predictor, sam2_model
    torch.cuda.empty_cache()

    return slide_name, len(embeddings), elapsed


def merge_embeddings(embeddings_dir, target_json, output_json):
    """Merge per-slide SAM2 embeddings into the combined detection JSON."""
    embeddings_dir = Path(embeddings_dir)

    all_embeddings = {}
    files = sorted(embeddings_dir.glob("sam2_*.json"))
    for f in files:
        print(f"  Loading {f.name}...")
        with open(f) as fh:
            embs = json.load(fh)
        all_embeddings.update(embs)
    print(f"  Total: {len(all_embeddings)} embeddings from {len(files)} files")

    print(f"Loading target: {target_json}...")
    with open(target_json) as f:
        detections = json.load(f)
    print(f"  {len(detections)} detections")

    n_updated = 0
    n_missing = 0

    for det in detections:
        uid = det.get("uid", "")
        feats = det.get("features_morph_color", det.get("features", {}))

        if uid in all_embeddings:
            emb = all_embeddings[uid]
            for i, v in enumerate(emb):
                feats[f"sam2_{i}"] = v
            n_updated += 1
        else:
            n_missing += 1

    print(f"  Updated: {n_updated}")
    print(f"  Missing: {n_missing}")

    print(f"Writing to {output_json}...")
    with open(output_json, "w") as f:
        json.dump(detections, f)
    print(f"  Done ({len(detections)} detections)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Extract command
    ext = sub.add_parser("extract", help="Extract mask-aware SAM2 embeddings from CZI")
    ext.add_argument(
        "--detections", type=Path, required=True, help="Combined detection JSON with contour_yx"
    )
    ext.add_argument("--czi-dir", type=Path, required=True, help="Directory containing CZI files")
    ext.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for per-slide embedding JSONs",
    )
    ext.add_argument(
        "--node-index",
        type=int,
        default=None,
        help="This node's index (0-based, from SLURM_ARRAY_TASK_ID)",
    )
    ext.add_argument("--num-nodes", type=int, default=None, help="Total number of nodes")
    ext.add_argument(
        "--num-gpus", type=int, default=None, help="GPUs per node (default: auto-detect)"
    )
    ext.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help="Training data JSON — adds training samples for SAM2 extraction "
        "(centroid-based, coords parsed from UIDs)",
    )

    # Merge command
    mrg = sub.add_parser("merge", help="Merge per-slide embeddings into detection JSON")
    mrg.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory with per-slide sam2_*.json files",
    )
    mrg.add_argument("--target", type=Path, required=True, help="Detection JSON to update")
    mrg.add_argument(
        "--output", type=Path, required=True, help="Output path for updated detection JSON"
    )

    args = parser.parse_args()

    if args.command == "extract":
        import torch

        num_gpus = args.num_gpus or torch.cuda.device_count()
        if num_gpus == 0:
            print("WARNING: No GPUs detected, running on CPU")
            num_gpus = 1
        print(f"Node GPUs: {num_gpus}")

        by_slide, slides = load_detections_by_slide(
            args.detections,
            training_data_path=args.training_data,
        )

        # Determine which slides this node handles
        if args.node_index is not None and args.num_nodes is not None:
            my_slides = [
                slides[i] for i in range(len(slides)) if i % args.num_nodes == args.node_index
            ]
        else:
            my_slides = slides

        print(f"\nThis node: {len(my_slides)} slides: {my_slides}")
        print(f"Using {num_gpus} GPUs in parallel")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = _find_sam2_checkpoint()

        # Assign slides to GPUs round-robin
        gpu_assignments = []
        for i, slide_name in enumerate(my_slides):
            gpu_id = i % num_gpus
            gpu_assignments.append(
                (
                    slide_name,
                    by_slide[slide_name],
                    str(args.czi_dir),
                    gpu_id,
                    str(args.output_dir),
                    str(checkpoint),
                )
            )

        t0 = time.time()
        import multiprocessing as mp

        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
            results = list(executor.map(_worker_extract_slide, gpu_assignments))

        total_time = time.time() - t0
        total_embs = sum(r[1] for r in results)
        print(f"\n{'='*50}")
        print(f"Done: {total_embs} embeddings from {len(my_slides)} slides in {total_time:.1f}s")
        for slide_name, n_embs, elapsed in results:
            status = "OK" if n_embs > 0 else "FAILED"
            print(f"  {slide_name}: {n_embs} ({elapsed:.1f}s) [{status}]")

    elif args.command == "merge":
        merge_embeddings(args.embeddings_dir, args.target, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
