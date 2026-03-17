#!/usr/bin/env python3
"""split_regions_for_lmd.py — Segment bright regions from a CZI channel, split large
regions into equal-area sub-pieces via watershed, and output a detection-compatible
JSON for downstream LMD export.

Workflow:
  1. Load a single CZI channel (full slide) into RAM
  2. Threshold bright regions at a configurable percentile (default: 98th)
  3. Morphological cleanup (opening + closing)
  4. Connected component labeling + area filter
  5. Large regions: split into N = floor(area / target_area) equal-area pieces
     using k-means seeded watershed on the distance transform
  6. Extract per-piece contour and centroid
  7. Write detection-compatible JSON

Output JSON fields per detection:
  uid                    — "<stem>_<celltype>_<cx>_<cy>"
  global_center          — [x_px, y_px]  (slide coordinates)
  global_center_um       — [x_um, y_um]
  features               — area, area_um2, region_id, piece_id,
                           parent_region_area_um2, n_pieces_in_region
  outer_contour_global   — [[x, y], ...]  (slide coordinates)

Usage:
  PYTHONPATH=$REPO python scripts/split_regions_for_lmd.py \\
      --czi-path /path/to/slide.czi \\
      --channel 2 \\
      --output-dir /path/to/output

  # Or using channel-spec:
  PYTHONPATH=$REPO python scripts/split_regions_for_lmd.py \\
      --czi-path /path/to/slide.czi \\
      --channel-spec "detect=NfL" \\
      --output-dir /path/to/output
"""

import os
import sys
import argparse

# Allow running as standalone script from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from segmentation.io.czi_loader import CZILoader, get_czi_metadata, resolve_channel_indices
from segmentation.utils.json_utils import atomic_json_dump
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Region splitting
# ---------------------------------------------------------------------------

def split_region(mask: np.ndarray, n_pieces: int, seed: int = 42) -> list:
    """Split a binary mask into n_pieces roughly equal-area sub-regions.

    Uses k-means seeded watershed on the distance transform.  Each piece is
    returned as a boolean mask of the same shape as the input.

    Args:
        mask:     2-D boolean array
        n_pieces: Number of pieces to produce.  If <= 1, returns [mask].
        seed:     RNG seed for reproducible k-means initialisation.

    Returns:
        List of boolean masks, one per piece.  Empty watershed labels are
        dropped so the list may be shorter than n_pieces if the mask is
        small.
    """
    if n_pieces <= 1:
        return [mask]

    dist = distance_transform_edt(mask)

    # K-means on mask pixel coordinates to find seed locations
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [mask]

    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    rng = np.random.default_rng(seed)

    # Clamp n_pieces to available pixels (degenerate tiny regions)
    n_pieces = min(n_pieces, len(coords))
    if n_pieces <= 1:
        return [mask]

    # Initialise centroids by random sampling
    init_idx = rng.choice(len(coords), n_pieces, replace=False)
    centroids = coords[init_idx].copy()

    for _ in range(50):  # k-means iterations
        # Assign each pixel to nearest centroid using cKDTree.
        # O(N log K) time/memory instead of O(NK) from broadcast.
        tree = cKDTree(centroids)
        _, labels_km = tree.query(coords)

        new_centroids = np.array([
            coords[labels_km == i].mean(axis=0) if (labels_km == i).any() else centroids[i]
            for i in range(n_pieces)
        ])
        if np.allclose(centroids, new_centroids, atol=0.5):
            break
        centroids = new_centroids

    # Place integer markers at centroid positions
    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, (cx, cy) in enumerate(centroids):
        ry = int(round(cy))
        rx = int(round(cx))
        ry = max(0, min(mask.shape[0] - 1, ry))
        rx = max(0, min(mask.shape[1] - 1, rx))
        # If the centroid landed outside the mask push to nearest mask pixel
        if not mask[ry, rx]:
            # Find nearest foreground pixel
            fg_ys, fg_xs = ys, xs
            sq = (fg_xs - rx) ** 2 + (fg_ys - ry) ** 2
            nearest = sq.argmin()
            ry, rx = int(fg_ys[nearest]), int(fg_xs[nearest])
        markers[ry, rx] = i + 1

    ws_labels = watershed(-dist, markers, mask=mask)

    pieces = []
    for i in range(1, n_pieces + 1):
        piece = ws_labels == i
        if piece.any():
            pieces.append(piece.astype(bool))
    return pieces


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contour_global(piece: np.ndarray, x_offset: int, y_offset: int) -> list:
    """Extract the largest external contour of a binary mask and return as
    global [[x, y], ...] coordinates.

    Args:
        piece:    2-D boolean/uint8 mask in local (tile) coordinates.
        x_offset: X offset of the tile in slide (global) pixels.
        y_offset: Y offset of the tile in slide (global) pixels.

    Returns:
        List of [x, y] pairs in global coordinates, or empty list.
    """
    contours, _ = cv2.findContours(
        piece.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze()
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    # Convert to global coordinates
    pts_global = pts.copy().astype(int)
    pts_global[:, 0] += x_offset
    pts_global[:, 1] += y_offset
    return pts_global.tolist()


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_channel(
    channel_data: np.ndarray,
    pixel_size_um: float,
    target_area_um2: float,
    min_area_um2: float,
    threshold_pct: float,
    morph_kernel: int,
    seed: int,
    cell_type: str,
    slide_stem: str,
) -> list:
    """Threshold a single channel, split large regions, and return detections.

    Args:
        channel_data:    Full-slide 2-D uint16 array (H, W).
        pixel_size_um:   µm per pixel.
        target_area_um2: Target area per split piece in µm².
        min_area_um2:    Minimum region area in µm² to keep.
        threshold_pct:   Intensity percentile threshold (e.g. 98).
        morph_kernel:    Morphological disk radius (pixels).
        seed:            RNG seed.
        cell_type:       Label used in UIDs (e.g. "nfl").
        slide_stem:      CZI filename stem, used in UIDs.

    Returns:
        List of detection dicts.
    """
    px = pixel_size_um
    px2 = px * px  # µm² per pixel

    min_area_px = max(1, int(min_area_um2 / px2))
    target_area_px = max(1, int(target_area_um2 / px2))

    logger.info(
        f"Processing {channel_data.shape[1]:,}x{channel_data.shape[0]:,} px image, "
        f"pixel_size={px:.4f} um/px"
    )
    logger.info(
        f"min_area={min_area_um2} um2 = {min_area_px} px | "
        f"target_area={target_area_um2} um2 = {target_area_px} px"
    )

    # --- Threshold (memory-efficient: subsample for percentile) ---
    # Full-slide nonzero extraction would create a ~40 GB temporary array.
    # Instead, subsample up to 10M random nonzero pixels for percentile estimation.
    flat = channel_data.ravel()
    nonzero_indices = np.flatnonzero(flat)
    if nonzero_indices.size == 0:
        logger.warning("Channel is entirely zero — no detections produced.")
        return []

    rng_pct = np.random.default_rng(seed)
    sample_size = min(nonzero_indices.size, 10_000_000)
    if sample_size < nonzero_indices.size:
        sample_idx = rng_pct.choice(nonzero_indices, sample_size, replace=False)
    else:
        sample_idx = nonzero_indices
    threshold = np.percentile(flat[sample_idx], threshold_pct)
    del flat, nonzero_indices, sample_idx  # free temporaries
    logger.info(f"Intensity threshold (p{threshold_pct:.0f}): {threshold:.1f}")

    bright_mask = channel_data > threshold
    logger.info(f"Bright pixels: {bright_mask.sum():,} ({bright_mask.mean()*100:.2f}%)")

    # Free channel_data before morphology — morphological operations create
    # temporary copies the same size as bright_mask (~23 GB for a full-slide image).
    # Releasing channel_data (~47 GB) keeps peak memory under control.
    del channel_data

    # --- Morphological cleanup ---
    se_open = disk(morph_kernel)
    se_close = disk(morph_kernel * 2)
    bright_mask = binary_opening(bright_mask, se_open)
    bright_mask = binary_closing(bright_mask, se_close)
    bright_mask = remove_small_objects(bright_mask, min_size=min_area_px)
    logger.info(f"After morphology: {bright_mask.sum():,} foreground pixels")

    # --- Connected components ---
    labeled = label(bright_mask)
    props = regionprops(labeled)
    logger.info(f"Connected components: {len(props)}")

    # Filter by area
    props = [p for p in props if p.area >= min_area_px]
    logger.info(f"Regions >= {min_area_um2} um2: {len(props)}")

    detections = []
    region_id = 0

    for prop in props:
        region_mask = labeled == prop.label
        region_area_px = prop.area
        region_area_um2 = region_area_px * px2

        # Number of pieces this region should be split into
        n_pieces = max(1, int(region_area_px / target_area_px))

        logger.debug(
            f"Region {region_id}: area={region_area_um2:.1f} um2, "
            f"n_pieces={n_pieces}"
        )

        # Bounding box for local processing (offset into global coords)
        min_row, min_col, max_row, max_col = prop.bbox
        local_mask = region_mask[min_row:max_row, min_col:max_col]

        pieces = split_region(local_mask, n_pieces, seed=seed)

        for piece_id, piece in enumerate(pieces):
            if not piece.any():
                continue

            # Contour in global coords
            contour_global = extract_contour_global(piece, x_offset=min_col, y_offset=min_row)
            if not contour_global:
                continue

            # Centroid in global coords
            piece_ys, piece_xs = np.where(piece)
            cx_local = float(piece_xs.mean())
            cy_local = float(piece_ys.mean())
            cx_global = cx_local + min_col
            cy_global = cy_local + min_row

            piece_area_px = int(piece.sum())
            piece_area_um2 = piece_area_px * px2

            cx_um = cx_global * px
            cy_um = cy_global * px

            uid = f"{slide_stem}_{cell_type}_{int(round(cx_global))}_{int(round(cy_global))}"

            det = {
                "uid": uid,
                "global_center": [int(round(cx_global)), int(round(cy_global))],
                "global_center_um": [round(cx_um, 3), round(cy_um, 3)],
                "features": {
                    "area": piece_area_px,
                    "area_um2": round(piece_area_um2, 3),
                    "region_id": region_id,
                    "piece_id": piece_id,
                    "parent_region_area_um2": round(region_area_um2, 3),
                    "n_pieces_in_region": len(pieces),
                },
                "outer_contour_global": contour_global,
            }
            detections.append(det)

        region_id += 1

    logger.info(f"Total detections produced: {len(detections)}")
    return detections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Segment bright regions from a CZI channel, split large regions into "
            "equal-area sub-pieces, and output a detection-compatible JSON for LMD export."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--czi-path", required=True,
        help="Path to the CZI file.",
    )
    parser.add_argument(
        "--channel", type=int, default=None,
        help=(
            "Integer channel index for thresholding. "
            "Mutually exclusive with --channel-spec."
        ),
    )
    parser.add_argument(
        "--channel-spec", default=None,
        help=(
            'Channel spec string for automatic resolution, e.g. "detect=NfL" or '
            '"detect=647". Mutually exclusive with --channel.'
        ),
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write the output JSON file.",
    )
    parser.add_argument(
        "--target-area-um2", type=float, default=200.0,
        help="Target area per split piece in µm².",
    )
    parser.add_argument(
        "--min-area-um2", type=float, default=200.0,
        help="Minimum region area in µm² to include.",
    )
    parser.add_argument(
        "--threshold-pct", type=float, default=98.0,
        help="Intensity percentile threshold for bright-region detection.",
    )
    parser.add_argument(
        "--morph-kernel", type=int, default=2,
        help="Morphological disk radius in pixels for opening/closing.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for k-means watershed seeding.",
    )
    parser.add_argument(
        "--cell-type", default="region",
        help="Label for the detected structures, used in UIDs and output filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)

    if not czi_path.exists():
        logger.error(f"CZI file not found: {czi_path}")
        sys.exit(1)

    # --- Resolve channel index ---
    if args.channel is not None and args.channel_spec is not None:
        logger.error("Specify either --channel or --channel-spec, not both.")
        sys.exit(1)

    if args.channel is None and args.channel_spec is None:
        logger.error("One of --channel or --channel-spec is required.")
        sys.exit(1)

    channel_index = args.channel  # may still be None if using spec

    if args.channel_spec is not None:
        # Parse "role=spec" format (e.g. "detect=NfL") or bare spec
        spec_str = args.channel_spec
        if "=" in spec_str:
            _, spec_str = spec_str.split("=", 1)
        spec_str = spec_str.strip()

        logger.info(f"Resolving channel spec: '{spec_str}'")
        meta = get_czi_metadata(str(czi_path))
        resolved = resolve_channel_indices(
            czi_metadata=meta,
            marker_specs=[spec_str],
            filename=czi_path.name,
        )
        channel_index = resolved[spec_str]
        logger.info(f"Resolved '{spec_str}' -> channel {channel_index}")

    logger.info(f"CZI: {czi_path.name}")
    logger.info(f"Channel: {channel_index}")
    logger.info(f"Target area: {args.target_area_um2} um2")
    logger.info(f"Min area: {args.min_area_um2} um2")
    logger.info(f"Threshold percentile: {args.threshold_pct}")

    # --- Load channel ---
    assert channel_index is not None, "channel_index must be resolved before loading"
    logger.info("Loading channel into RAM (this may take several minutes for large slides)...")
    loader = CZILoader(str(czi_path), load_to_ram=True, channel=channel_index)
    pixel_size_um = loader.get_pixel_size()
    channel_data = loader.get_channel_data(channel_index)

    if channel_data is None:
        logger.error(f"Channel {channel_index} could not be loaded.")
        sys.exit(1)

    logger.info(
        f"Channel loaded: shape={channel_data.shape}, dtype={channel_data.dtype}, "
        f"pixel_size={pixel_size_um:.4f} um/px, "
        f"memory={channel_data.nbytes / (1024**3):.2f} GB"
    )

    # --- Process ---
    slide_stem = czi_path.stem
    detections = process_channel(
        channel_data=channel_data,
        pixel_size_um=pixel_size_um,
        target_area_um2=args.target_area_um2,
        min_area_um2=args.min_area_um2,
        threshold_pct=args.threshold_pct,
        morph_kernel=args.morph_kernel,
        seed=args.seed,
        cell_type=args.cell_type,
        slide_stem=slide_stem,
    )

    # --- Write output ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{slide_stem}_{args.cell_type}_detections.json"
    atomic_json_dump(detections, str(out_path))
    logger.info(f"Wrote {len(detections)} detections to {out_path}")

    # Summary statistics
    if detections:
        areas = [d["features"]["area_um2"] for d in detections]
        logger.info(
            f"Area summary: min={min(areas):.1f} max={max(areas):.1f} "
            f"mean={sum(areas)/len(areas):.1f} um2"
        )
        n_regions = max(d["features"]["region_id"] for d in detections) + 1
        logger.info(f"Parent regions processed: {n_regions}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
