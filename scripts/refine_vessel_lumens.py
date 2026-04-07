#!/usr/bin/env python3
"""Post-process SAM2 lumen candidates: dedup, darkness filter, cell proximity.

Operates on pre-computed shard JSON files — no GPU, no images needed — enabling
fast iterative tuning of lumen detection thresholds.

Three sequential steps:
  1. Dedup: IoU + containment + cross-scale merge (finest-first)
  2. Darkness verification: contrast_ratio + optional interior_median cap
  3. Cell proximity metadata (optional, never filters)

Usage::

    python scripts/refine_vessel_lumens.py \\
        --input vessel_lumens_shard_*.json \\
        --min-contrast-ratio 2.0 \\
        --output-dir refined/ \\
        --czi-path slide.czi \\
        --display-channels 1,3,0 --channel-names "SMA,CD31,nuc"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Deduplication
# ---------------------------------------------------------------------------


def _build_polygon(pts: list[list[float]]):
    """Build a Shapely Polygon from contour points (full resolution, no simplification).

    Args:
        pts: List of [x, y] coordinate pairs.

    Returns:
        Valid Shapely Polygon or None.
    """
    from shapely.geometry import Polygon

    if pts is None or len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_valid or poly.is_empty:
            return None
        return poly
    except Exception as exc:
        logger.debug("Polygon construction failed: %s", exc)
        return None


def dedup_lumens(
    lumens: list[dict],
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.8,
) -> tuple[list[dict], list[dict]]:
    """IoU + containment dedup, keeping largest by area (tiebreak by stability).

    For overlapping pairs (IoU >= threshold), the LARGEST lumen by ``area_px``
    is kept; ties are broken by ``sam2_stability``.  Additionally, if the
    intersection covers >= ``containment_threshold`` of the smaller polygon's
    area, the smaller lumen is suppressed regardless of IoU.

    Args:
        lumens: Input lumen dicts with ``contour_global_um``, ``area_px``,
            ``sam2_stability``.
        iou_threshold: IoU threshold for duplicate suppression.
        containment_threshold: Fraction of smaller polygon area that must be
            covered by intersection for containment suppression.

    Returns:
        Tuple of (kept, rejected) lumen lists.
    """
    from shapely.strtree import STRtree

    if len(lumens) <= 1:
        return lumens, []

    polys: list = []
    valid_indices: list[int] = []
    for i, lumen in enumerate(lumens):
        poly = _build_polygon(lumen.get("contour_global_um"))
        if poly is not None:
            polys.append(poly)
            valid_indices.append(i)

    if len(polys) <= 1:
        kept = [lumens[i] for i in valid_indices]
        rejected_idx = set(range(len(lumens))) - set(valid_indices)
        rejected = [lumens[i] for i in rejected_idx]
        for r in rejected:
            r["rejection_reason"] = "invalid_polygon"
        return kept, rejected

    tree = STRtree(polys)
    suppressed: set[int] = set()

    for qi in range(len(polys)):
        if qi in suppressed:
            continue
        poly_q = polys[qi]

        try:
            candidates = tree.query(poly_q)
        except Exception as exc:
            logger.debug("STRtree query failed: %s", exc)
            continue

        for ci_raw in candidates:
            ci = int(ci_raw)
            if ci <= qi or ci in suppressed:
                continue

            poly_c = polys[ci]
            try:
                inter_area = poly_q.intersection(poly_c).area
                union_area = poly_q.union(poly_c).area
            except Exception:
                continue

            if union_area <= 0:
                continue

            iou = inter_area / union_area

            # Containment check: suppress smaller if intersection covers most of it
            area_q = poly_q.area
            area_c = poly_c.area
            smaller_area = min(area_q, area_c)
            containment = inter_area / smaller_area if smaller_area > 0 else 0.0

            should_suppress = iou >= iou_threshold or containment >= containment_threshold
            if not should_suppress:
                continue

            # Keep largest by physical size (equiv_diameter_um), tiebreak by sam2_stability.
            # NOTE: area_px is in detection-scale pixels (scale-dependent), so a scale-64
            # lumen has tiny area_px despite being physically huge. Use um-based size.
            size_q = lumens[valid_indices[qi]].get("equiv_diameter_um", 0)
            size_c = lumens[valid_indices[ci]].get("equiv_diameter_um", 0)
            if size_q > size_c:
                suppressed.add(ci)
            elif size_c > size_q:
                suppressed.add(qi)
                break  # qi suppressed — stop checking
            else:
                # Equal size — tiebreak by stability
                stab_q = lumens[valid_indices[qi]].get("sam2_stability", 0)
                stab_c = lumens[valid_indices[ci]].get("sam2_stability", 0)
                if stab_q >= stab_c:
                    suppressed.add(ci)
                else:
                    suppressed.add(qi)
                    break

    kept = [lumens[valid_indices[i]] for i in range(len(polys)) if i not in suppressed]
    rejected = [lumens[valid_indices[i]] for i in suppressed]
    for r in rejected:
        r["rejection_reason"] = "iou_or_containment_dedup"

    # Also collect lumens with invalid polygons as rejected
    invalid_idx = set(range(len(lumens))) - set(valid_indices)
    for i in invalid_idx:
        lumens[i]["rejection_reason"] = "invalid_polygon"
        rejected.append(lumens[i])

    logger.info(
        "IoU+containment dedup: %d -> %d kept, %d rejected "
        "(iou_thresh=%.2f, contain_thresh=%.2f)",
        len(lumens),
        len(kept),
        len(rejected),
        iou_threshold,
        containment_threshold,
    )
    return kept, rejected


def merge_across_scales(
    lumens: list[dict],
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.8,
    coarsest_first: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Cross-scale merge: suppress lumens that overlap already-accepted ones.

    Default is coarsest-first (scale 64 → 16 → 8 → 4 → 2) so large vessels
    get clean coarse-scale boundaries. Fine-scale lumens fill gaps where
    coarse detection missed. Set ``coarsest_first=False`` for finest-first.

    Uses BOTH IoU and containment checks — a tiny lumen fully inside a large
    one has low IoU but high containment, so both criteria are needed.

    Args:
        lumens: Input lumen dicts with ``scale`` and ``contour_global_um``.
        iou_threshold: IoU threshold for cross-scale overlap.
        containment_threshold: Fraction of candidate's area covered by
            intersection with an accepted lumen for suppression.
        coarsest_first: If True, process coarsest scales first.

    Returns:
        Tuple of (accepted, rejected) lumen lists.
    """
    from shapely.strtree import STRtree

    if len(lumens) <= 1:
        return lumens, []

    # Group by scale
    by_scale: dict[int, list[dict]] = {}
    for lumen in lumens:
        s = lumen.get("scale", 1)
        by_scale.setdefault(s, []).append(lumen)

    accepted: list[dict] = []
    accepted_polys: list = []
    rejected: list[dict] = []

    scale_order = sorted(by_scale.keys(), reverse=coarsest_first)
    order_label = "coarsest-first" if coarsest_first else "finest-first"
    logger.info("Cross-scale merge order: %s %s", order_label, list(scale_order))

    for scale in scale_order:
        scale_lumens = by_scale[scale]
        n_added = 0

        # Rebuild STRtree from all previously accepted lumens.
        # NOTE: Same-scale lumens accepted in this loop are NOT in the tree,
        # so intra-scale overlaps are not caught here — they should already
        # be resolved by dedup_lumens() in step 1a.
        accepted_tree = STRtree(accepted_polys) if accepted_polys else None

        for lumen in scale_lumens:
            poly = _build_polygon(lumen.get("contour_global_um"))
            if poly is None:
                lumen["rejection_reason"] = "invalid_polygon_cross_scale"
                rejected.append(lumen)
                continue

            overlaps = False
            if accepted_tree is not None:
                try:
                    candidates = accepted_tree.query(poly)
                except Exception as exc:
                    logger.debug("STRtree query failed: %s", exc)
                    candidates = []

                poly_area = poly.area
                for ci_raw in candidates:
                    ci = int(ci_raw)
                    ap = accepted_polys[ci]
                    try:
                        inter = poly.intersection(ap).area
                    except Exception as exc:
                        logger.debug("Polygon operation failed: %s", exc)
                        continue

                    # Check IoU
                    union = poly_area + ap.area - inter
                    if union > 0 and (inter / union) >= iou_threshold:
                        overlaps = True
                        break

                    # Check containment: is this candidate mostly inside
                    # an already-accepted lumen?
                    if poly_area > 0 and (inter / poly_area) >= containment_threshold:
                        overlaps = True
                        break

            if not overlaps:
                accepted.append(lumen)
                accepted_polys.append(poly)
                n_added += 1
            else:
                lumen["rejection_reason"] = "cross_scale_overlap"
                rejected.append(lumen)

        logger.info("  Scale %dx: accepted %d / %d lumens", scale, n_added, len(scale_lumens))

    logger.info("Cross-scale merge: %d -> %d lumens", len(lumens), len(accepted))
    return accepted, rejected


# ---------------------------------------------------------------------------
# Step 2: Darkness verification
# ---------------------------------------------------------------------------


def _assign_darkness_tier(interior_median: float | None) -> str:
    """Classify a lumen into a darkness tier based on interior median.

    Returns:
        One of "very_dark", "dark", "moderate", "light", or "unknown".
    """
    if interior_median is None:
        return "unknown"
    if interior_median < 5:
        return "very_dark"
    if interior_median < 15:
        return "dark"
    if interior_median < 30:
        return "moderate"
    return "light"


def filter_by_darkness(
    lumens: list[dict],
    min_contrast_ratio: float = 2.0,
    max_interior_median: float | None = None,
) -> tuple[list[dict], list[dict]]:
    """Filter lumens by contrast ratio and optional interior median cap.

    Adds ``darkness_tier`` field to all lumens (kept and rejected).

    Args:
        lumens: Input lumen dicts with ``contrast_ratio``, ``interior_median``.
        min_contrast_ratio: Minimum boundary/interior contrast ratio.
        max_interior_median: Optional absolute cap on interior brightness.
            ``None`` disables this filter.

    Returns:
        Tuple of (kept, rejected) lumen lists.
    """
    kept: list[dict] = []
    rejected: list[dict] = []

    for lumen in lumens:
        interior_med = lumen.get("interior_median")
        lumen["darkness_tier"] = _assign_darkness_tier(interior_med)

        contrast = lumen.get("contrast_ratio", 0.0)
        if contrast is None:
            contrast = 0.0

        if contrast < min_contrast_ratio:
            lumen["rejection_reason"] = (
                f"low_contrast_ratio ({contrast:.2f} < {min_contrast_ratio:.2f})"
            )
            rejected.append(lumen)
            continue

        if max_interior_median is not None and interior_med is not None:
            if interior_med > max_interior_median:
                lumen["rejection_reason"] = (
                    f"high_interior_median ({interior_med:.1f} > {max_interior_median:.1f})"
                )
                rejected.append(lumen)
                continue

        kept.append(lumen)

    # Log darkness tier distribution for kept lumens
    tier_counts: dict[str, int] = {}
    for lumen in kept:
        tier = lumen.get("darkness_tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    logger.info(
        "Darkness filter: %d -> %d kept, %d rejected " "(min_contrast=%.2f, max_interior=%s)",
        len(lumens),
        len(kept),
        len(rejected),
        min_contrast_ratio,
        f"{max_interior_median:.1f}" if max_interior_median is not None else "None",
    )
    for tier in ("very_dark", "dark", "moderate", "light", "unknown"):
        count = tier_counts.get(tier, 0)
        if count > 0:
            logger.info("  %-12s %5d (%5.1f%%)", tier, count, 100.0 * count / max(len(kept), 1))

    return kept, rejected


# ---------------------------------------------------------------------------
# Step 3: Cell proximity metadata (optional, never filters)
# ---------------------------------------------------------------------------


def add_cell_proximity_metadata(
    lumens: list[dict],
    detections: list[dict],
    marker_names: list[str] | None = None,
) -> None:
    """Annotate lumens with nearest-cell proximity metrics (in-place).

    For each lumen, computes:
      - ``nearest_marker_cell_distance_um``: distance from lumen centroid to
        nearest marker+ cell
      - ``n_marker_cells_within_diameter``: count within 1x equiv_diameter
      - ``n_marker_cells_within_2x_diameter``: count within 2x equiv_diameter

    If ``marker_names`` is provided, only cells positive for ANY listed marker
    are considered.  Otherwise all cells are used.

    Args:
        lumens: Lumen dicts (modified in-place).
        detections: Cell detection dicts.
        marker_names: Optional list of marker names (e.g. ``["SMA", "CD31"]``).
    """
    from scipy.spatial import cKDTree

    if not lumens or not detections:
        logger.info("Cell proximity: skipped (no lumens or no detections)")
        return

    # Extract marker+ cell positions
    marker_cells: list[dict] = []
    if marker_names:
        for det in detections:
            feats = det.get("features", {})
            for mname in marker_names:
                cls_key = f"{mname}_class"
                cls_val = det.get(cls_key) or feats.get(cls_key)
                if cls_val == "positive":
                    marker_cells.append(det)
                    break  # counted once even if positive for multiple markers
    else:
        marker_cells = detections

    if not marker_cells:
        logger.warning(
            "Cell proximity: 0 marker+ cells found (markers=%s). Skipping.",
            marker_names,
        )
        return

    # Build positions array for marker+ cells
    from xldvp_seg.utils.detection_utils import extract_positions_um

    positions, _ = extract_positions_um(marker_cells)
    if len(positions) == 0:
        logger.warning("Cell proximity: could not resolve any marker+ cell positions. Skipping.")
        return

    cell_tree = cKDTree(positions)
    logger.info(
        "Cell proximity: %d marker+ cells, %d lumens (markers=%s)",
        len(positions),
        len(lumens),
        marker_names,
    )

    # Compute lumen centroids from contour_global_um
    for lumen in lumens:
        pts = lumen.get("contour_global_um")
        if pts is None or len(pts) < 3:
            continue

        arr = np.asarray(pts, dtype=np.float64)
        centroid_um = arr.mean(axis=0)  # [x, y]
        equiv_d = lumen.get("equiv_diameter_um", 0.0) or 0.0

        # Nearest marker+ cell distance
        dist, _ = cell_tree.query(centroid_um, k=1)
        lumen["nearest_marker_cell_distance_um"] = float(dist)

        # Count within 1x and 2x equivalent diameter
        if equiv_d > 0:
            n_1x = len(cell_tree.query_ball_point(centroid_um, r=equiv_d))
            n_2x = len(cell_tree.query_ball_point(centroid_um, r=2.0 * equiv_d))
        else:
            n_1x = 0
            n_2x = 0
        lumen["n_marker_cells_within_diameter"] = n_1x
        lumen["n_marker_cells_within_2x_diameter"] = n_2x


# ---------------------------------------------------------------------------
# Statistics logging
# ---------------------------------------------------------------------------


def _log_summary(
    n_input: int,
    n_after_dedup: int,
    n_after_darkness: int,
    kept: list[dict],
    rejected: list[dict],
) -> None:
    """Log a concise pipeline summary."""
    logger.info("=" * 60)
    logger.info("Refinement summary")
    logger.info("=" * 60)
    logger.info("  Input lumens:          %6d", n_input)
    logger.info("  After dedup+merge:     %6d", n_after_dedup)
    logger.info("  After darkness filter: %6d", n_after_darkness)
    logger.info("  Total rejected:        %6d", len(rejected))
    logger.info("=" * 60)

    # Rejection reason breakdown
    reason_counts: dict[str, int] = {}
    for r in rejected:
        reason = r.get("rejection_reason", "unknown")
        # Normalize parametric reasons to just the prefix
        prefix = reason.split(" (")[0]
        reason_counts[prefix] = reason_counts.get(prefix, 0) + 1

    if reason_counts:
        logger.info("Rejection breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            logger.info("  %-35s %5d", reason, count)

    # Scale distribution for kept lumens
    scale_counts: dict[int, int] = {}
    for lumen in kept:
        s = lumen.get("scale", 0)
        scale_counts[s] = scale_counts.get(s, 0) + 1

    if scale_counts:
        logger.info("Scale distribution (kept):")
        for scale in sorted(scale_counts.keys()):
            count = scale_counts[scale]
            logger.info("  scale %3d: %5d", scale, count)


# ---------------------------------------------------------------------------
# Viewer generation
# ---------------------------------------------------------------------------


def _generate_viewer(
    output_json: Path,
    output_html: Path,
    group_field: str,
    czi_path: Path | None,
    display_channels: str,
    channel_names: str | None,
    scale_factor: float,
    scene: int,
    max_contours: int,
    title: str = "Refined Lumens",
) -> None:
    """Launch generate_contour_viewer.py as a subprocess."""
    # Locate script relative to this file
    script_dir = Path(__file__).resolve().parent
    viewer_script = script_dir / "generate_contour_viewer.py"

    if not viewer_script.exists():
        logger.warning("Contour viewer script not found at %s — skipping viewer.", viewer_script)
        return

    cmd = [
        sys.executable,
        str(viewer_script),
        "--contours",
        str(output_json),
        "--group-field",
        group_field,
        "--max-contours",
        str(max_contours),
        "--title",
        title,
        "--output",
        str(output_html),
    ]

    if czi_path is not None:
        cmd.extend(["--czi-path", str(czi_path)])
        cmd.extend(["--display-channels", display_channels])
        cmd.extend(["--scale-factor", str(scale_factor)])
        cmd.extend(["--scene", str(scene)])
        if channel_names:
            cmd.extend(["--channel-names", channel_names])

    logger.info("Generating viewer: %s", output_html.name)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Viewer generation failed (exit %d):\n%s", result.returncode, result.stderr)
    else:
        if result.stderr:
            # Log viewer output at debug level
            for line in result.stderr.strip().split("\n"):
                logger.debug("  viewer: %s", line)
        logger.info("Viewer written: %s", output_html)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Refine SAM2 lumen candidates: dedup, darkness filter, cell proximity. "
            "Operates on pre-computed shard JSON files (no GPU, no images)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input / Output ---
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="One or more shard JSON files (e.g. vessel_lumens_shard_*.json).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for refined JSON and viewer.",
    )
    p.add_argument(
        "--output-prefix",
        default="refined_lumens",
        help="Filename prefix for output files (default: refined_lumens).",
    )

    # --- Dedup ---
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for dedup (default: 0.3).",
    )
    p.add_argument(
        "--containment-threshold",
        type=float,
        default=0.8,
        help="Containment fraction for small-inside-large suppression (default: 0.8).",
    )

    # --- Darkness ---
    p.add_argument(
        "--min-contrast-ratio",
        type=float,
        default=2.0,
        help="Minimum boundary/interior contrast ratio (default: 2.0).",
    )
    p.add_argument(
        "--max-interior-median",
        type=float,
        default=None,
        help="Optional absolute cap on interior brightness (default: disabled).",
    )

    # --- Cell proximity (optional) ---
    p.add_argument(
        "--detections",
        type=Path,
        default=None,
        help="Cell detections JSON for proximity metadata (optional).",
    )
    p.add_argument(
        "--marker-names",
        default=None,
        help="Comma-separated marker names for proximity (requires --detections).",
    )

    # --- Viewer ---
    p.add_argument(
        "--group-field",
        default="darkness_tier",
        help="Field for viewer group coloring (default: darkness_tier).",
    )
    p.add_argument(
        "--czi-path",
        type=Path,
        default=None,
        help="CZI file for fluorescence background in viewer.",
    )
    p.add_argument(
        "--display-channels",
        default="1,3,0",
        help="Comma-separated CZI channel indices for R,G,B (default: 1,3,0).",
    )
    p.add_argument(
        "--channel-names",
        default=None,
        help="Comma-separated channel names for viewer legend.",
    )
    p.add_argument(
        "--scale-factor",
        type=float,
        default=0.0625,
        help="CZI downsample factor for viewer (default: 0.0625 = 1/16).",
    )
    p.add_argument(
        "--scene",
        type=int,
        default=0,
        help="CZI scene index for multi-scene files (default: 0).",
    )
    p.add_argument(
        "--skip-viewer",
        action="store_true",
        help="Skip HTML viewer generation.",
    )
    p.add_argument(
        "--max-contours",
        type=int,
        default=50_000,
        help="Maximum contours in viewer (default: 50000).",
    )

    # --- Misc ---
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report statistics without writing output files.",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for lumen refinement."""
    args = parse_args(argv)
    setup_logging(level="INFO")
    t0 = time.time()

    # Validate inputs
    for p in args.input:
        if not p.exists():
            logger.error("Input file not found: %s", p)
            sys.exit(1)

    if args.detections and not args.detections.exists():
        logger.error("Detections file not found: %s", args.detections)
        sys.exit(1)

    if args.marker_names and not args.detections:
        logger.error("--marker-names requires --detections")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load shards
    # -----------------------------------------------------------------------
    logger.info("Loading %d shard file(s)...", len(args.input))
    all_lumens: list[dict] = []
    for shard_path in sorted(args.input):
        shard_data = fast_json_load(shard_path)
        logger.info("  %s: %d candidates", shard_path.name, len(shard_data))
        all_lumens.extend(shard_data)

    n_input = len(all_lumens)
    logger.info("Total input candidates: %d", n_input)

    if n_input == 0:
        logger.warning("No candidates found in input shards.")
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / f"{args.output_prefix}.json"
            atomic_json_dump([], str(out_path))
            logger.info("Wrote empty output: %s", out_path)
        return

    # -----------------------------------------------------------------------
    # Step 1: Dedup (IoU + containment)
    # -----------------------------------------------------------------------
    logger.info("Step 1a: IoU + containment dedup...")
    deduped, rejected_dedup = dedup_lumens(
        all_lumens,
        iou_threshold=args.iou_threshold,
        containment_threshold=args.containment_threshold,
    )

    # Step 1b: Cross-scale merge (finest first)
    logger.info("Step 1b: Cross-scale merge...")
    merged, rejected_merge = merge_across_scales(
        deduped,
        iou_threshold=args.iou_threshold,
        containment_threshold=args.containment_threshold,
        coarsest_first=True,
    )
    n_after_dedup = len(merged)
    all_rejected = rejected_dedup + rejected_merge

    # -----------------------------------------------------------------------
    # Step 2: Darkness verification
    # -----------------------------------------------------------------------
    logger.info("Step 2: Darkness verification...")
    kept, rejected_dark = filter_by_darkness(
        merged,
        min_contrast_ratio=args.min_contrast_ratio,
        max_interior_median=args.max_interior_median,
    )
    n_after_darkness = len(kept)
    all_rejected.extend(rejected_dark)

    # -----------------------------------------------------------------------
    # Step 3: Cell proximity metadata (optional, never filters)
    # -----------------------------------------------------------------------
    if args.detections:
        logger.info("Step 3: Cell proximity metadata...")
        cell_dets = fast_json_load(args.detections)
        logger.info("  Loaded %d cell detections", len(cell_dets))
        marker_names = (
            [m.strip() for m in args.marker_names.split(",")] if args.marker_names else None
        )
        add_cell_proximity_metadata(kept, cell_dets, marker_names)
    else:
        logger.info("Step 3: Cell proximity skipped (no --detections)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _log_summary(n_input, n_after_dedup, n_after_darkness, kept, all_rejected)

    elapsed = time.time() - t0
    logger.info("Total time: %.1fs", elapsed)

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    if args.dry_run:
        logger.info("Dry run — no files written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_kept = args.output_dir / f"{args.output_prefix}.json"
    out_rejected = args.output_dir / f"{args.output_prefix}_rejected.json"

    atomic_json_dump(kept, str(out_kept))
    logger.info("Wrote %d refined lumens: %s", len(kept), out_kept)

    atomic_json_dump(all_rejected, str(out_rejected))
    logger.info("Wrote %d rejected lumens: %s", len(all_rejected), out_rejected)

    # -----------------------------------------------------------------------
    # Viewer
    # -----------------------------------------------------------------------
    if not args.skip_viewer and kept:
        out_html = args.output_dir / f"{args.output_prefix}_viewer.html"
        _generate_viewer(
            output_json=out_kept,
            output_html=out_html,
            group_field=args.group_field,
            czi_path=args.czi_path,
            display_channels=args.display_channels,
            channel_names=args.channel_names,
            scale_factor=args.scale_factor,
            scene=args.scene,
            max_contours=args.max_contours,
        )

    logger.info("Done. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
