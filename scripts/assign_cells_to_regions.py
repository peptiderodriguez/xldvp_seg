#!/usr/bin/env python
"""Assign detected cells to regions defined by a 2D label map in downsampled space.

The label map is a 2D integer array where each pixel value is a region ID
(0 = background). The map lives in some downsampled coordinate space (e.g.
fluorescence thumbnail, block-face-warped photo). Each cell's CZI-space
centroid is scaled down to the label-map resolution, and the region ID at
that pixel is attached as ``organ_id``.

Output: a new detections JSON identical to the input, with ``organ_id``
added to every detection. An index JSON (``{output}.regions.json``) summarizes
cell counts per region.

Usage:
    python scripts/assign_cells_to_regions.py \\
        --detections cell_detections.json \\
        --label-map labels_best77.npy \\
        --czi-path slide.czi \\
        --output cell_detections_with_organs.json
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.utils.detection_utils import extract_positions_um  # noqa: E402
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _get_czi_mosaic_shape(czi_path: str) -> tuple[int, int]:
    """Return (height, width) of the CZI mosaic in pixels."""
    from xldvp_seg.io.czi_loader import CZILoader

    loader = CZILoader(czi_path)
    return int(loader.height), int(loader.width)


def assign_regions(
    detections: list[dict],
    label_map: np.ndarray,
    mosaic_h: int,
    mosaic_w: int,
    pixel_size_um: float | None = None,
) -> list[dict]:
    """Attach ``organ_id`` to each detection via label-map lookup.

    Args:
        detections: list of detection dicts.
        label_map: 2D int array (H, W) in downsampled space.
        mosaic_h, mosaic_w: full-res CZI dimensions (pixels).
        pixel_size_um: CZI pixel size. Inferred from detections if None.
    """
    positions_um, pixel_size_um, valid_idx = extract_positions_um(
        detections, pixel_size_um=pixel_size_um, return_indices=True
    )
    if pixel_size_um is None:
        raise ValueError("pixel_size_um could not be inferred; pass --pixel-size")

    lh, lw = label_map.shape
    scale_x = mosaic_w / lw
    scale_y = mosaic_h / lh
    logger.info(
        "Label map %dx%d, CZI %dx%d, scale_x=%.2f scale_y=%.2f",
        lh,
        lw,
        mosaic_h,
        mosaic_w,
        scale_x,
        scale_y,
    )

    # positions are (x, y) in microns. Convert to CZI px, then to label-map px.
    xs_czi = positions_um[:, 0] / pixel_size_um
    ys_czi = positions_um[:, 1] / pixel_size_um
    xs_lbl = np.clip((xs_czi / scale_x).astype(int), 0, lw - 1)
    ys_lbl = np.clip((ys_czi / scale_y).astype(int), 0, lh - 1)
    region_ids = label_map[ys_lbl, xs_lbl].astype(int)

    # Write organ_id onto the detections (including ones with no valid position)
    id_by_idx = dict(zip(valid_idx, region_ids.tolist()))
    for i, det in enumerate(detections):
        det["organ_id"] = id_by_idx.get(i, 0)

    return detections


def _summarize(detections: list[dict]) -> dict:
    counts = Counter(det["organ_id"] for det in detections)
    total = sum(counts.values())
    return {
        "n_cells_total": total,
        "n_cells_assigned": total - counts.get(0, 0),
        "n_regions_with_cells": sum(1 for k, v in counts.items() if k != 0 and v > 0),
        "cells_per_region": {str(k): v for k, v in sorted(counts.items())},
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections", required=True, help="Input cell_detections.json")
    parser.add_argument("--label-map", required=True, help="2D int label map .npy (H, W)")
    parser.add_argument("--czi-path", help="CZI file (used to read mosaic shape)")
    parser.add_argument(
        "--mosaic-shape",
        help="Fallback: 'H,W' of CZI mosaic in pixels if --czi-path not available",
    )
    parser.add_argument(
        "--pixel-size", type=float, help="CZI pixel size in um (inferred if omitted)"
    )
    parser.add_argument("--output", required=True, help="Output JSON with organ_id added")
    args = parser.parse_args()

    logger.info("Loading detections: %s", args.detections)
    detections = fast_json_load(args.detections)
    logger.info("  %d detections loaded", len(detections))

    logger.info("Loading label map: %s", args.label_map)
    label_map = np.load(args.label_map)
    if label_map.ndim != 2:
        raise ValueError(f"Label map must be 2D, got shape {label_map.shape}")
    logger.info(
        "  shape=%s, regions=%d (max id %d)",
        label_map.shape,
        len(np.unique(label_map)),
        label_map.max(),
    )

    if args.czi_path:
        mosaic_h, mosaic_w = _get_czi_mosaic_shape(args.czi_path)
    elif args.mosaic_shape:
        mosaic_h, mosaic_w = (int(x) for x in args.mosaic_shape.split(","))
    else:
        raise ValueError("Must provide --czi-path or --mosaic-shape")
    logger.info("CZI mosaic: %d x %d (H x W)", mosaic_h, mosaic_w)

    detections = assign_regions(
        detections, label_map, mosaic_h, mosaic_w, pixel_size_um=args.pixel_size
    )

    summary = _summarize(detections)
    logger.info(
        "Assigned: %d / %d cells into %d regions",
        summary["n_cells_assigned"],
        summary["n_cells_total"],
        summary["n_regions_with_cells"],
    )

    atomic_json_dump(detections, args.output)
    logger.info("Wrote %s", args.output)

    summary_path = args.output + ".regions.json"
    atomic_json_dump(summary, summary_path)
    logger.info("Wrote summary %s", summary_path)


if __name__ == "__main__":
    main()
