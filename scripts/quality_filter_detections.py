#!/usr/bin/env python3
"""Apply morphological quality filters to detections without annotation/classifier.

For clean slides where Cellpose detection quality is high, this replaces
the annotation -> RF classifier step with simple heuristic filters:
  - Area range (default: 50-2000 um^2)
  - Solidity minimum (default: 0.85)
  - Optional: eccentricity max, aspect_ratio max
  - Optional: per-channel mean intensity minimum

Sets rf_prediction=1.0 for passing detections, 0.0 for rejected.
Stores a quality_filter_info dict on each detection for provenance,
analogous to classifier_info from apply_classifier.py.

Output is compatible with all downstream tools (apply_classifier, regenerate_html,
run_lmd_export, select_mks_for_lmd, paper_figure_sampling, etc.).

Usage:
    # Default filters (50-2000 um^2, solidity > 0.85)
    python scripts/quality_filter_detections.py \\
        --detections cell_detections.json \\
        --output cell_detections_filtered.json

    # Custom range with eccentricity filter
    python scripts/quality_filter_detections.py \\
        --detections cell_detections.json \\
        --output cell_detections_filtered.json \\
        --min-area-um2 100 --max-area-um2 1500 \\
        --min-solidity 0.90 --max-eccentricity 0.95

    # Filter by channel intensity (e.g., keep only cells with high ch1 signal)
    python scripts/quality_filter_detections.py \\
        --detections cell_detections.json \\
        --output cell_detections_filtered.json \\
        --min-channel-mean 1:500

    # Multiple channel thresholds (ch0 > 200 AND ch2 > 300)
    python scripts/quality_filter_detections.py \\
        --detections cell_detections.json \\
        --output cell_detections_filtered.json \\
        --min-channel-mean 0:200 --min-channel-mean 2:300
"""

import argparse
import math
import sys
from pathlib import Path

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger, setup_logging
from xldvp_seg.utils.timestamps import timestamped_path, update_symlink

logger = get_logger(__name__)


def parse_channel_threshold(value):
    """Parse 'CHANNEL:THRESHOLD' string into (channel_index, threshold).

    Args:
        value: String like '1:500' meaning ch1_mean >= 500.

    Returns:
        Tuple of (int, float).

    Raises:
        argparse.ArgumentTypeError: If format is invalid.
    """
    try:
        parts = value.split(":", 1)
        if len(parts) != 2:
            raise ValueError("expected CHANNEL:THRESHOLD")
        ch = int(parts[0])
        thresh = float(parts[1])
        if ch < 0:
            raise ValueError("channel index must be >= 0")
        return (ch, thresh)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid channel threshold '{value}': {e}. "
            f"Expected format CHANNEL:THRESHOLD (e.g., 1:500)"
        ) from None


def auto_detect_pixel_size(detections, max_scan=100):
    """Auto-detect pixel size from detection features.

    Checks up to max_scan detections for:
      1. Explicit pixel_size_um field (set by tile_processing.py)
      2. Derivation from area_um2 / area ratio

    Args:
        detections: List of detection dicts.
        max_scan: Maximum number of detections to scan.

    Returns:
        Detected pixel size in um, or None if not found.
    """
    for det in detections[:max_scan]:
        feat = det.get("features", {})

        # Method 1: explicit pixel_size_um (canonical, set by enrich_detection_features)
        ps = feat.get("pixel_size_um")
        if ps is not None:
            ps = float(ps)
            if 0.01 < ps < 10.0:
                return ps
            logger.warning(f"pixel_size_um={ps} outside plausible range [0.01, 10.0] um, skipping")

        # Method 2: derive from area vs area_um2
        area_px = feat.get("area")
        area_um2 = feat.get("area_um2")
        if area_px and area_um2 and area_px > 0 and area_um2 > 0:
            derived = math.sqrt(area_um2 / area_px)
            if 0.01 < derived < 10.0:
                return derived
            logger.warning(f"Derived pixel_size={derived:.4f} um outside plausible range, skipping")

    return None


def validate_args(args):
    """Validate CLI argument ranges.

    Raises:
        SystemExit: If any argument is out of valid range.
    """
    errors = []
    if args.min_area_um2 < 0:
        errors.append(f"--min-area-um2 must be >= 0, got {args.min_area_um2}")
    if args.max_area_um2 <= 0:
        errors.append(f"--max-area-um2 must be > 0, got {args.max_area_um2}")
    if args.min_area_um2 >= args.max_area_um2:
        errors.append(
            f"--min-area-um2 ({args.min_area_um2}) must be < "
            f"--max-area-um2 ({args.max_area_um2})"
        )
    if not 0.0 <= args.min_solidity <= 1.0:
        errors.append(f"--min-solidity must be in [0, 1], got {args.min_solidity}")
    if args.max_eccentricity is not None and not 0.0 <= args.max_eccentricity <= 1.0:
        errors.append(f"--max-eccentricity must be in [0, 1], got {args.max_eccentricity}")
    if args.max_aspect_ratio is not None and args.max_aspect_ratio <= 0:
        errors.append(f"--max-aspect-ratio must be > 0, got {args.max_aspect_ratio}")
    if args.pixel_size is not None and not 0.01 < args.pixel_size < 10.0:
        errors.append(f"--pixel-size must be in (0.01, 10.0) um, got {args.pixel_size}")
    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Quality-filter detections using morphological heuristics. "
        "Sets rf_prediction=1.0 for passing detections, 0.0 for rejected. "
        "Output is compatible with all downstream tools (LMD export, sampling, HTML viewer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else None,
    )
    parser.add_argument(
        "--detections",
        required=True,
        help="Input detections JSON file (from pipeline or apply_classifier)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output filtered detections JSON (timestamped copy + symlink)",
    )

    # --- Morphological filters ---
    morph_group = parser.add_argument_group("morphological filters")
    morph_group.add_argument(
        "--min-area-um2",
        type=float,
        default=50,
        help="Minimum cell area in um^2 (default: 50)",
    )
    morph_group.add_argument(
        "--max-area-um2",
        type=float,
        default=2000,
        help="Maximum cell area in um^2 (default: 2000)",
    )
    morph_group.add_argument(
        "--min-solidity",
        type=float,
        default=0.85,
        help="Minimum solidity [0-1] (default: 0.85). "
        "Solidity = area / convex_hull_area. Low solidity = irregular shape.",
    )
    morph_group.add_argument(
        "--max-eccentricity",
        type=float,
        default=None,
        help="Maximum eccentricity [0-1] (default: no filter). "
        "0 = circle, 1 = line. Typical cell filter: 0.95.",
    )
    morph_group.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=None,
        help="Maximum major/minor axis ratio (default: no filter). "
        "1 = circle. Typical cell filter: 3.0.",
    )

    # --- Channel intensity filters ---
    channel_group = parser.add_argument_group("channel intensity filters")
    channel_group.add_argument(
        "--min-channel-mean",
        type=parse_channel_threshold,
        action="append",
        default=None,
        metavar="CH:THRESH",
        help="Minimum mean intensity for a channel. Format: CHANNEL:THRESHOLD "
        "(e.g., 1:500 means ch1_mean >= 500). Can be repeated for multiple channels.",
    )

    # --- Pixel size ---
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in um/pixel. Auto-detected from features['pixel_size_um'] "
        "if omitted. Falls back to 0.1725 um (legacy default) with a warning.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    validate_args(args)

    # --- Check input file exists ---
    det_path = Path(args.detections)
    if not det_path.exists():
        logger.error(f"Detections file not found: {det_path}")
        sys.exit(1)

    # --- Load ---
    logger.info(f"Loading {args.detections}...")
    detections = fast_json_load(args.detections)
    logger.info(f"  {len(detections):,} detections loaded")

    if not detections:
        logger.warning("No detections in input file. Writing empty output.")
        out_path = Path(args.output)
        ts_out = timestamped_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump([], ts_out)
        update_symlink(out_path, ts_out)
        logger.info(f"Saved: {ts_out}")
        return

    # --- Auto-detect pixel size ---
    pixel_size = args.pixel_size
    if pixel_size is None:
        pixel_size = auto_detect_pixel_size(detections)
    if pixel_size is None:
        from xldvp_seg.utils.config import _LEGACY_PIXEL_SIZE_UM

        pixel_size = _LEGACY_PIXEL_SIZE_UM
        logger.warning(
            f"Could not auto-detect pixel size from features. "
            f"Using legacy default {pixel_size} um/pixel. "
            f"Pass --pixel-size explicitly if this is wrong."
        )
    logger.info(f"  Pixel size: {pixel_size:.4f} um/pixel")

    # --- Check for missing features ---
    n_missing_area = 0
    n_missing_solidity = 0
    n_missing_eccentricity = 0
    n_missing_aspect_ratio = 0
    check_count = min(len(detections), 200)
    for det in detections[:check_count]:
        feat = det.get("features", {})
        if feat.get("area_um2") is None and feat.get("area") is None:
            n_missing_area += 1
        if "solidity" not in feat:
            n_missing_solidity += 1
        if args.max_eccentricity is not None and "eccentricity" not in feat:
            n_missing_eccentricity += 1
        if args.max_aspect_ratio is not None and "aspect_ratio" not in feat:
            n_missing_aspect_ratio += 1

    if n_missing_area > check_count * 0.1:
        logger.warning(
            f"  {n_missing_area}/{check_count} sampled detections missing area features. "
            f"These will be treated as area=0 um^2 (will fail min-area filter)."
        )
    if n_missing_solidity > check_count * 0.1:
        logger.warning(
            f"  {n_missing_solidity}/{check_count} sampled detections missing 'solidity'. "
            f"Missing values default to 1.0 (always passes solidity filter)."
        )
    if n_missing_eccentricity > check_count * 0.1:
        logger.warning(
            f"  {n_missing_eccentricity}/{check_count} sampled detections missing 'eccentricity'. "
            f"Missing values default to 0.0 (always passes eccentricity filter)."
        )
    if n_missing_aspect_ratio > check_count * 0.1:
        logger.warning(
            f"  {n_missing_aspect_ratio}/{check_count} sampled detections missing 'aspect_ratio'. "
            f"Missing values default to 1.0 (always passes aspect_ratio filter)."
        )

    # --- Build filter criteria dict for logging and provenance ---
    filter_criteria = {
        "min_area_um2": args.min_area_um2,
        "max_area_um2": args.max_area_um2,
        "min_solidity": args.min_solidity,
        "pixel_size_um": pixel_size,
    }
    if args.max_eccentricity is not None:
        filter_criteria["max_eccentricity"] = args.max_eccentricity
    if args.max_aspect_ratio is not None:
        filter_criteria["max_aspect_ratio"] = args.max_aspect_ratio
    if args.min_channel_mean:
        filter_criteria["min_channel_mean"] = {
            f"ch{ch}": thresh for ch, thresh in args.min_channel_mean
        }

    logger.info("  Filter criteria:")
    logger.info(f"    Area: {args.min_area_um2} - {args.max_area_um2} um^2")
    logger.info(f"    Solidity: >= {args.min_solidity}")
    if args.max_eccentricity is not None:
        logger.info(f"    Eccentricity: <= {args.max_eccentricity}")
    if args.max_aspect_ratio is not None:
        logger.info(f"    Aspect ratio: <= {args.max_aspect_ratio}")
    if args.min_channel_mean:
        for ch, thresh in args.min_channel_mean:
            logger.info(f"    ch{ch}_mean: >= {thresh}")

    # --- Apply filters ---
    n_pass = 0
    n_fail = 0
    # Per-filter rejection counters for diagnostics
    reject_area = 0
    reject_solidity = 0
    reject_eccentricity = 0
    reject_aspect_ratio = 0
    reject_channel = 0

    for det in detections:
        feat = det.get("features", {})

        # Area filter
        area_um2 = feat.get("area_um2")
        if area_um2 is None:
            area_px = feat.get("area", 0)
            if area_px is None:
                area_px = 0
            area_um2 = area_px * pixel_size**2

        # Solidity filter
        solidity = feat.get("solidity", 1.0)
        if solidity is None:
            solidity = 1.0

        # Apply filters (track which filter(s) rejected each detection)
        passed = True
        reasons = []

        if area_um2 < args.min_area_um2 or area_um2 > args.max_area_um2:
            passed = False
            reasons.append("area")
            reject_area += 1

        if solidity < args.min_solidity:
            passed = False
            reasons.append("solidity")
            reject_solidity += 1

        if args.max_eccentricity is not None:
            ecc = feat.get("eccentricity", 0)
            if ecc is None:
                ecc = 0
            if ecc > args.max_eccentricity:
                passed = False
                reasons.append("eccentricity")
                reject_eccentricity += 1

        if args.max_aspect_ratio is not None:
            ar = feat.get("aspect_ratio", 1)
            if ar is None:
                ar = 1
            if ar > args.max_aspect_ratio:
                passed = False
                reasons.append("aspect_ratio")
                reject_aspect_ratio += 1

        if args.min_channel_mean:
            for ch, thresh in args.min_channel_mean:
                ch_mean = feat.get(f"ch{ch}_mean")
                if ch_mean is None or ch_mean < thresh:
                    passed = False
                    reasons.append(f"ch{ch}_mean")
                    reject_channel += 1
                    break  # count once per detection, not per channel

        det["rf_prediction"] = 1.0 if passed else 0.0
        det["quality_filter"] = "pass" if passed else "fail"
        if not passed:
            det["quality_filter_reasons"] = reasons

        if passed:
            n_pass += 1
        else:
            n_fail += 1

    # --- Summary ---
    total = len(detections)
    logger.info("  Results:")
    logger.info(f"    Pass: {n_pass:,} ({100 * n_pass / total:.1f}%)")
    logger.info(f"    Fail: {n_fail:,} ({100 * n_fail / total:.1f}%)")
    logger.info("  Per-filter rejection counts (detections can fail multiple):")
    logger.info(f"    Area:         {reject_area:,}")
    logger.info(f"    Solidity:     {reject_solidity:,}")
    if args.max_eccentricity is not None:
        logger.info(f"    Eccentricity: {reject_eccentricity:,}")
    if args.max_aspect_ratio is not None:
        logger.info(f"    Aspect ratio: {reject_aspect_ratio:,}")
    if args.min_channel_mean:
        logger.info(f"    Channel mean: {reject_channel:,}")

    if n_pass == 0:
        logger.warning("All detections were rejected. Consider relaxing filter thresholds.")
    if n_fail == 0:
        logger.info(
            "All detections passed. Filters may be too lenient, "
            "or detection quality is already high."
        )

    # --- Add provenance info ---
    quality_filter_info = {
        "method": "quality_filter",
        "criteria": filter_criteria,
        "total": total,
        "passed": n_pass,
        "rejected": n_fail,
    }
    for det in detections:
        det["quality_filter_info"] = quality_filter_info

    # --- Save with timestamp + symlink ---
    out_path = Path(args.output)
    ts_out = timestamped_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(detections, ts_out)
    update_symlink(out_path, ts_out)
    logger.info(f"Saved: {ts_out}")
    logger.info(f"Symlink: {out_path} -> {ts_out.name}")


if __name__ == "__main__":
    main()
