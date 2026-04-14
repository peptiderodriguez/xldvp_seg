#!/usr/bin/env python
"""Extract morphological and per-channel intensity features for threshold-detected lumens.

Reads vessel_lumens_threshold.json + OME-Zarr, derives binary masks from contours,
and enriches each lumen with a ``features`` dict suitable for classifier training.

Usage:
    PYTHONPATH=$REPO $XLDVP_PYTHON scripts/extract_lumen_features.py \
        --lumens vessel_lumens_threshold.json \
        --zarr-path slide.ome.zarr \
        --output vessel_lumens_features.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger, setup_logging
from xldvp_seg.utils.zarr_io import read_all_channels_crop, resolve_zarr_level

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mask from contour rasterization
# ---------------------------------------------------------------------------


def rasterize_contour(
    contour_um: list[list[float]],
    refined_scale: int,
    base_pixel_size: float,
) -> tuple[np.ndarray, int, int, int, int] | None:
    """Rasterize a contour_global_um back to a binary mask.

    Returns (mask, bbox_y, bbox_x, bbox_h, bbox_w) in pixel coords at refined_scale,
    or None if the contour is too small.
    """
    pts = np.asarray(contour_um, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None

    pixel_size = base_pixel_size * refined_scale
    pts_px = pts / pixel_size  # (N, 2) in (x, y) pixel coords

    # Bounding box in pixel coords (clamp to >= 0)
    x_min = max(0, int(np.floor(pts_px[:, 0].min())))
    y_min = max(0, int(np.floor(pts_px[:, 1].min())))
    x_max = int(np.ceil(pts_px[:, 0].max()))
    y_max = int(np.ceil(pts_px[:, 1].max()))
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    if w < 2 or h < 2:
        return None

    # Rasterize contour relative to bbox origin
    local_pts = (pts_px - [x_min, y_min]).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [local_pts], 1)

    return mask.astype(bool), y_min, x_min, h, w


# ---------------------------------------------------------------------------
# Morphological features (mask-only, no image needed)
# ---------------------------------------------------------------------------


def extract_morph_features(mask: np.ndarray, pixel_size_um: float) -> dict[str, float]:
    """Extract shape features from a binary mask.

    Args:
        mask: 2D boolean array.
        pixel_size_um: Pixel size in microns at the mask scale.

    Returns:
        Dict of morphological features.
    """
    area_per_px = pixel_size_um**2
    mask_u8 = mask.astype(np.uint8)

    # Contour for perimeter and shape descriptors
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {}
    contour = max(contours, key=cv2.contourArea)

    area_px = int(mask.sum())
    area_um2 = area_px * area_per_px
    perimeter_px = cv2.arcLength(contour, True)
    perimeter_um = perimeter_px * pixel_size_um

    # Moments
    moments = cv2.moments(mask_u8)

    # Circularity: 4*pi*area / perimeter^2
    circularity = (4 * np.pi * area_px) / max(perimeter_px**2, 1e-10)
    circularity = min(circularity, 1.0)  # clamp

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / max(hull_area, 1e-10)
    hull_perim = cv2.arcLength(hull, True)
    hull_ratio = hull_perim / max(perimeter_px, 1e-10)  # perimeter-based, <=1 for convex

    # Fit ellipse for eccentricity
    eccentricity = 0.0
    aspect_ratio = 1.0
    major_axis_um = 0.0
    minor_axis_um = 0.0
    if len(contour) >= 5:
        try:
            (_, (ma, MA), _) = cv2.fitEllipse(contour)
            if MA > 0:
                e_ratio = ma / MA  # minor/major
                eccentricity = np.sqrt(1 - e_ratio**2) if e_ratio < 1 else 0.0
                aspect_ratio = MA / max(ma, 1e-10)
                major_axis_um = MA * pixel_size_um
                minor_axis_um = ma * pixel_size_um
        except cv2.error:
            pass

    # Bounding rect for extent
    x, y, bw, bh = cv2.boundingRect(contour)
    extent = area_px / max(bw * bh, 1)

    equiv_diameter_um = np.sqrt(4 * area_um2 / np.pi)
    compactness = (perimeter_um**2) / max(area_um2, 1e-10)

    # Hu moments (7 values, log-transformed)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = np.array([-np.sign(h) * np.log10(max(abs(h), 1e-30)) for h in hu])

    features = {
        "morph_area_um2": round(area_um2, 1),
        "morph_perimeter_um": round(perimeter_um, 1),
        "morph_circularity": round(circularity, 4),
        "morph_solidity": round(solidity, 4),
        "morph_eccentricity": round(eccentricity, 4),
        "morph_aspect_ratio": round(aspect_ratio, 4),
        "morph_extent": round(extent, 4),
        "morph_equiv_diameter_um": round(equiv_diameter_um, 2),
        "morph_compactness": round(compactness, 2),
        "morph_hull_ratio": round(hull_ratio, 4),
        "morph_major_axis_um": round(major_axis_um, 2),
        "morph_minor_axis_um": round(minor_axis_um, 2),
    }
    for i, val in enumerate(hu_log):
        features[f"morph_hu{i}"] = round(float(val), 4)

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract morphological and per-channel features for threshold-detected lumens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--lumens",
        required=True,
        type=Path,
        help="Path to vessel_lumens_threshold.json.",
    )
    p.add_argument(
        "--zarr-path",
        required=True,
        type=Path,
        help="Path to OME-Zarr file.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <input_dir>/vessel_lumens_features.json).",
    )
    p.add_argument(
        "--no-channel-stats",
        action="store_true",
        help="Skip per-channel intensity features.",
    )
    p.add_argument(
        "--max-area-um2",
        type=float,
        default=1_000_000.0,
        help="Skip lumens with area > this (default: 1e6 um^2 = 1 mm^2).",
    )
    p.add_argument(
        "--max-lumens",
        type=int,
        default=None,
        help="Process at most N lumens (for testing).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    setup_logging()
    args = parse_args(argv)
    t0 = time.time()

    logger.info("=" * 70)
    logger.info("LUMEN FEATURE EXTRACTION")
    logger.info("=" * 70)
    logger.info("  Lumens: %s", args.lumens)
    logger.info("  Zarr: %s", args.zarr_path)

    # Load lumens
    lumens = fast_json_load(str(args.lumens))
    if isinstance(lumens, dict):
        for key in ("lumens", "detections", "data"):
            if key in lumens and isinstance(lumens[key], list):
                lumens = lumens[key]
                break
    logger.info("  Loaded %d lumens", len(lumens))

    if args.max_lumens and len(lumens) > args.max_lumens:
        lumens = lumens[: args.max_lumens]
        logger.info("  Truncated to %d lumens (--max-lumens)", len(lumens))

    # Open zarr
    import zarr

    zarr_root = zarr.open(str(args.zarr_path), mode="r")

    # Get base pixel size from zarr
    base_pixel_size = None
    multiscales = zarr_root.attrs.get("multiscales", [])
    if multiscales:
        datasets = multiscales[0].get("datasets", [])
        if datasets:
            transforms = datasets[0].get("coordinateTransformations", [])
            for t in transforms:
                if t.get("type") == "scale":
                    base_pixel_size = float(t["scale"][-1])
                    break
    if base_pixel_size is None:
        base_pixel_size = zarr_root.attrs.get("pixel_size_um")
    if base_pixel_size is None:
        logger.error("Cannot determine pixel size from zarr metadata.")
        sys.exit(1)
    logger.info("  Base pixel size: %.4f um", base_pixel_size)

    # Resolve zarr levels for each scale we might need
    scales_needed = set()
    for lumen in lumens:
        scales_needed.add(lumen.get("refined_scale", lumen.get("discovery_scale", 4)))
    logger.info("  Scales needed: %s", sorted(scales_needed))

    level_cache: dict[int, tuple[Any, int]] = {}
    for scale in scales_needed:
        level_cache[scale] = resolve_zarr_level(zarr_root, scale)

    # Channel stats mixin (standalone instantiation)
    extract_channel_stats = not args.no_channel_stats
    channel_mixin = None
    if extract_channel_stats:
        from xldvp_seg.detection.strategies.mixins import MultiChannelFeatureMixin

        channel_mixin = MultiChannelFeatureMixin()

    # Process lumens
    n_success = 0
    n_failed = 0
    n_skipped_large = 0
    log_interval = max(1, len(lumens) // 20)

    for idx, lumen in enumerate(lumens):
        # Generate UID if missing
        if "uid" not in lumen:
            contour = lumen.get("contour_global_um")
            if contour:
                pts = np.asarray(contour)
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                lumen["uid"] = f"lumen_{cx:.2f}_{cy:.2f}"
                lumen["centroid_x_um"] = round(cx, 2)
                lumen["centroid_y_um"] = round(cy, 2)
            else:
                lumen["uid"] = f"lumen_{idx}"

        # Skip lumens that are too large (background false positives)
        area = lumen.get("area_um2", 0)
        if args.max_area_um2 and area > args.max_area_um2:
            n_skipped_large += 1
            lumen["features"] = {}
            continue

        # Rasterize contour → mask
        contour = lumen.get("contour_global_um")
        scale = lumen.get("refined_scale", lumen.get("discovery_scale", 4))
        pixel_size = base_pixel_size * scale

        if not contour:
            n_failed += 1
            lumen["features"] = {}
            continue

        result = rasterize_contour(contour, scale, base_pixel_size)
        if result is None:
            n_failed += 1
            lumen["features"] = {}
            continue

        mask, bbox_y, bbox_x, bbox_h, bbox_w = result

        # Phase 1: morph features
        features = extract_morph_features(mask, pixel_size)

        # Phase 2: per-channel stats
        if channel_mixin and extract_channel_stats:
            level_data, extra_ds = level_cache[scale]
            channels = read_all_channels_crop(level_data, extra_ds, bbox_y, bbox_x, bbox_h, bbox_w)
            # _include_zeros=True: zarr data is post-processing, zeros are real signal
            # (not CZI padding). Lumens ARE expected to have low/zero intensity.
            channels_dict = {f"ch{ci}": ch_data for ci, ch_data in enumerate(channels)}
            ch_features = channel_mixin.extract_multichannel_features(
                mask, channels_dict, _include_zeros=True
            )
            features.update(ch_features)

        lumen["features"] = features
        n_success += 1

        if (idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 0.1)
            remaining = (len(lumens) - idx - 1) / max(rate, 0.01)
            logger.info(
                "  %d/%d lumens (%.1f/s, ~%.0fs remaining, %d failed, %d skipped large)",
                idx + 1,
                len(lumens),
                rate,
                remaining,
                n_failed,
                n_skipped_large,
            )

    if n_skipped_large:
        logger.info("Skipped %d lumens with area > %.0f um^2", n_skipped_large, args.max_area_um2)

    logger.info(
        "Feature extraction complete: %d success, %d failed out of %d",
        n_success,
        n_failed,
        len(lumens),
    )

    # Write output
    out_path = args.output or (args.lumens.parent / "vessel_lumens_features.json")
    atomic_json_dump(lumens, str(out_path))
    logger.info("Wrote %d lumens to %s", len(lumens), out_path)

    elapsed = time.time() - t0
    logger.info("Total time: %.1fs (%.1f min)", elapsed, elapsed / 60.0)


if __name__ == "__main__":
    main()
