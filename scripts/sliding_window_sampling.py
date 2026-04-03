#!/usr/bin/env python3
"""Sliding window cell sampling along ROI centerlines for LMD.

Places circular windows along the morphological skeleton (centerline) of an
arbitrary polygon ROI, then samples cells into each window such that:
  - Each cell is assigned to exactly one window (no reuse — cells are LMD-cut)
  - Total cell area per window matches a target (N × median cell area ± tolerance)
  - Cells are spatially balanced within each window (farthest-point sampling)

Supports grid search over (radius, overlap) to find zero-rejection configurations.

Workflow:
    1. Draw a polygon ROI in the spatial viewer, export as JSON
    2. Run --grid-search to find (radius, overlap) combos with zero rejections
    3. Pick a combo and run sampling
    4. Visualize the result, iterate if needed

Recommended workflow for new ROIs:
    # Step 1: find what works
    python scripts/sliding_window_sampling.py \\
        --detections dets.json --roi rois.json --czi-path slide.czi \\
        --grid-search --target-multiplier 20

    # Step 2: run with best combo
    python scripts/sliding_window_sampling.py \\
        --detections dets.json --roi rois.json --czi-path slide.czi \\
        --from-grid zero_rejection_combos.json --grid-index 0 \\
        --output samples.json --output-viz viz.png

    # Or run directly with known-good parameters
    python scripts/sliding_window_sampling.py \\
        --detections dets.json --roi rois.json --czi-path slide.czi \\
        --radius 70 --overlap 0.4 --target-multiplier 20

Reference settings (e14 WT coronal brain, Y-shaped ROI, ~660 cells, ~6700 cells/mm²):
    20x target (20 × median area per window):
        r=70um, overlap=40% -> 20 windows, 0 rejected, 54% coverage
        r=85um, overlap=60% -> 23 windows, 0 rejected, 62% coverage
    30x target (30 × median area per window):
        r=90um, overlap=40% -> 15 windows, 0 rejected, 60% coverage
        r=100um, overlap=50% -> 16 windows, 0 rejected, 65% coverage
    Key: narrow/curved ROIs need larger windows for zero rejections.
    Always use --czi-path for pixel size (don't let it infer from dilated areas).

Notes:
    - The ROI polygon is from the spatial viewer's ROI export (vertices_um)
    - Centerline is computed via morphological skeletonization of the polygon
    - For Y-shaped or branching ROIs, the skeleton naturally splits into branches
    - Window overlap is between consecutive windows along the centerline, not 2D
    - "Zero rejection" means every placed window position can fill its area target
    - Grid search sweeps radius 40-100um and overlap 1/3 to 2/3 by default
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from xldvp_seg.analysis.sliding_window_sampling import (
    compute_skeleton_paths,
    grid_search,
    place_windows_along_paths,
    run_sampling,
)
from xldvp_seg.utils.json_utils import atomic_json_dump
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_detections(detections_path, pixel_size_um=None):
    """Load detection positions and areas from JSON.

    Returns:
        (positions, areas) — Nx2 array and length-N array for ALL detections.
    """
    from xldvp_seg.utils.json_utils import fast_json_load

    dets = fast_json_load(str(detections_path))
    if isinstance(dets, dict):
        dets = dets.get("detections", [])

    xs, ys, areas = [], [], []
    for d in dets:
        gc_um = d.get("global_center_um")
        if gc_um is None:
            gc_um = d.get("features", {}).get("global_center_um")
        if gc_um is not None:
            xs.append(gc_um[0])
            ys.append(gc_um[1])
        elif "global_center" in d and pixel_size_um is not None:
            gc = d["global_center"]
            xs.append(gc[0] * pixel_size_um)
            ys.append(gc[1] * pixel_size_um)
        else:
            continue

        f = d.get("features", {})
        a = f.get("area_um2", f.get("area"))
        areas.append(a if a is not None else 0)

    if not xs:
        logger.error("No cell positions could be extracted from detections")
        sys.exit(1)
    positions = np.column_stack([xs, ys])
    areas = np.array(areas, dtype=float)

    nonzero = areas[areas > 0]
    if len(nonzero) > 0 and (areas == 0).any():
        n_missing = (areas == 0).sum()
        areas[areas == 0] = np.median(nonzero)
        logger.warning(
            f"{n_missing} detections missing area, using median ({np.median(nonzero):.1f} um²)"
        )

    return positions, areas


def load_rois(roi_path, roi_id=None):
    """Load ROI polygons from JSON.

    Args:
        roi_path: Path to ROI JSON (from spatial viewer export).
        roi_id: Specific ROI id/name, or None for all ROIs.

    Returns:
        List of (roi_dict, verts, poly) tuples.
    """
    from xldvp_seg.utils.json_utils import fast_json_load

    rois = fast_json_load(str(roi_path))
    roi_list = rois.get("rois", []) if isinstance(rois, dict) else rois
    if not isinstance(roi_list, list) or not roi_list:
        logger.error("No ROIs found in %s", roi_path)
        sys.exit(1)

    if roi_id is not None:
        matched = [r for r in roi_list if r.get("id") == roi_id or r.get("name") == roi_id]
        if not matched:
            logger.error(
                f"ROI '{roi_id}' not found. Available: {[r.get('id', r.get('name')) for r in roi_list]}"
            )
            sys.exit(1)
        roi_list = matched

    result = []
    for roi in roi_list:
        verts = np.array(roi["vertices_um"])
        poly = Polygon(verts)
        result.append((roi, verts, poly))

    return result


def get_roi_cells(positions, areas, poly):
    """Filter cells to those inside a polygon ROI.

    Returns:
        (roi_indices, roi_positions, roi_areas)
    """
    inside_mask = contains_xy(poly, positions[:, 0], positions[:, 1])
    roi_indices = np.where(inside_mask)[0]
    return roi_indices, positions[roi_indices], areas[roi_indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sliding window cell sampling along ROI centerlines for LMD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--detections", required=True, type=Path, help="Detection JSON file")
    parser.add_argument(
        "--roi", required=True, type=Path, help="ROI JSON (from spatial viewer export)"
    )
    parser.add_argument(
        "--roi-id",
        default=None,
        help="ROI id/name to process (default: all ROIs in the file)",
    )
    parser.add_argument(
        "--radius", type=float, default=65, help="Window radius in um (default: 65)"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5, help="Overlap fraction 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "--target-multiplier",
        type=float,
        default=20,
        help="Target area = N × median cell area (default: 20)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.10, help="Area tolerance ± fraction (default: 0.10)"
    )
    parser.add_argument(
        "--czi-path",
        type=Path,
        default=None,
        help="CZI file path — pixel size is read from CZI metadata (recommended)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in um (alternative to --czi-path, from czi_info.py output)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output sampling JSON")
    parser.add_argument("--output-viz", type=Path, default=None, help="Output visualization PNG")
    parser.add_argument(
        "--exclude-cells",
        type=Path,
        default=None,
        help="Prior sampling JSON — cells from this run are excluded (for incremental ROI sessions)",
    )

    # Grid search mode
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Search for zero-rejection (radius, overlap) combos",
    )
    parser.add_argument("--output-grid", type=Path, default=None, help="Output grid search JSON")
    parser.add_argument(
        "--radii",
        default=None,
        help="Comma-separated radii for grid search (default: 40,45,...,100)",
    )
    parser.add_argument(
        "--overlaps",
        default=None,
        help="Comma-separated overlap fractions for grid search (default: 0.33,0.4,0.5,0.6,0.67)",
    )

    # Run from grid search result
    parser.add_argument(
        "--from-grid", type=Path, default=None, help="Load params from grid search JSON"
    )
    parser.add_argument(
        "--grid-index",
        type=int,
        default=0,
        help="Index into grid search results (default: 0 = best)",
    )

    args = parser.parse_args()
    setup_logging(level="INFO")

    # Resolve pixel size from CZI metadata (authoritative) or CLI arg
    pixel_size_um = args.pixel_size
    if args.czi_path:
        from xldvp_seg.io.czi_loader import get_czi_metadata

        meta = get_czi_metadata(str(args.czi_path))
        pixel_size_um = meta["pixel_size_um"]
        logger.info(f"Pixel size from CZI: {pixel_size_um:.4f} um/px")
    if pixel_size_um is None:
        logger.warning(
            "No --czi-path or --pixel-size provided. Pixel size will be inferred from "
            "detection features — this may be inaccurate if contours were dilated. "
            "Always prefer --czi-path."
        )

    # Load detections (once for all ROIs)
    logger.info(f"Loading detections: {args.detections}")
    positions, areas = load_detections(args.detections, pixel_size_um=pixel_size_um)

    # Load ROIs — all if no --roi-id, else just the specified one
    logger.info(f"Loading ROIs: {args.roi}")
    roi_entries = load_rois(args.roi, roi_id=args.roi_id)
    logger.info(f"  {len(roi_entries)} ROI(s) to process")

    # --- Load params from grid search ---
    radius = args.radius
    overlap = args.overlap
    if args.from_grid:
        from xldvp_seg.utils.json_utils import fast_json_load

        grid_data = fast_json_load(str(args.from_grid))
        combos = grid_data.get("zero_rejection_combos", grid_data.get("all_combos", []))
        if not combos:
            logger.error("No combos found in grid search file")
            sys.exit(1)
        if args.grid_index >= len(combos):
            logger.error(f"grid-index {args.grid_index} out of range (0-{len(combos)-1})")
            sys.exit(1)
        chosen = combos[args.grid_index]
        radius = chosen["radius_um"]
        overlap = chosen["overlap_pct"] / 100
        logger.info(
            f"From grid search [{args.grid_index}]: radius={radius}um, overlap={overlap*100:.0f}%"
        )

    # --- Load excluded cells from prior session ---
    used_global = set()
    if args.exclude_cells:
        from xldvp_seg.utils.json_utils import fast_json_load

        prior = fast_json_load(str(args.exclude_cells))
        prior_rois = prior.get("rois", [prior]) if isinstance(prior, dict) else [prior]
        for pr in prior_rois:
            for w in pr.get("windows", []):
                used_global.update(w.get("cell_indices", []))
        logger.info(
            f"Excluding {len(used_global)} cells from prior session: {args.exclude_cells.name}"
        )

    # Capture prior exclusion count before processing
    n_excluded_prior = len(used_global)

    # Compute step once (depends only on radius/overlap, constant across ROIs)
    if not args.grid_search:
        step = 2 * radius * (1 - overlap)

    # Parse grid search params once (invariant across ROIs)
    _grid_radii = [int(x) for x in args.radii.split(",")] if args.radii else None
    _grid_overlaps = [float(x) for x in args.overlaps.split(",")] if args.overlaps else None

    # --- Process each ROI (shared used_global prevents double-counting) ---
    all_roi_results = []
    all_roi_paths = []  # skeleton paths for visualization

    for roi_idx, (roi_dict, verts, poly) in enumerate(roi_entries):
        roi_name = roi_dict.get("name", roi_dict.get("id", f"ROI_{roi_idx}"))
        logger.info(f"\n{'='*60}")
        logger.info(f"ROI {roi_idx + 1}/{len(roi_entries)}: {roi_name}")

        roi_indices, roi_positions, roi_areas = get_roi_cells(positions, areas, poly)

        # Exclude cells already claimed by previous ROIs
        available_mask = np.array([i not in used_global for i in roi_indices])
        roi_indices_avail = roi_indices[available_mask]
        roi_positions_avail = roi_positions[available_mask]
        roi_areas_avail = roi_areas[available_mask]

        n_cells = len(roi_indices)
        n_available = len(roi_indices_avail)
        if n_cells != n_available:
            logger.info(
                f"  {n_cells} cells in ROI, {n_available} available ({n_cells - n_available} claimed by prior ROIs)"
            )
        else:
            logger.info(f"  {n_cells} cells in ROI")

        if n_available == 0:
            logger.warning(f"  No available cells in {roi_name}, skipping")
            continue

        median_area = float(np.median(roi_areas_avail))
        logger.info(f"  Median cell area: {median_area:.1f} um²")

        # Compute skeleton
        paths = compute_skeleton_paths(poly)
        logger.info(f"  {len(paths)} skeleton paths")
        all_roi_paths.extend(paths)

        # Target area
        target_area = args.target_multiplier * median_area
        area_lo = target_area * (1 - args.tolerance)
        area_hi = target_area * (1 + args.tolerance)
        logger.info(
            f"  Target: {args.target_multiplier:.0f} × {median_area:.1f} = {target_area:.0f} um² "
            f"(±{args.tolerance*100:.0f}%)"
        )

        # --- Grid search mode ---
        if args.grid_search:
            logger.info(f"  Grid search for {roi_name}...")
            results = grid_search(
                roi_positions_avail,
                roi_areas_avail,
                roi_indices_avail,
                paths,
                area_lo,
                area_hi,
                n_available,
                radii=_grid_radii,
                overlaps=_grid_overlaps,
            )
            zero_rej = sorted(
                [r for r in results if r["n_rejected"] == 0],
                key=lambda r: -r["coverage_pct"],
            )
            logger.info(f"  All combos: {len(results)}, zero-rejection: {len(zero_rej)}")
            if zero_rej:
                logger.info(
                    f"  {'Radius':>7} {'Overlap':>8} {'Step':>6} {'Windows':>8} {'Coverage':>9}"
                )
                for r in zero_rej[:10]:
                    logger.info(
                        f"  {r['radius_um']:>5}um {r['overlap_pct']:>6}% {r['step_um']:>4}um "
                        f"{r['n_filled']:>8} {r['coverage_pct']:>8.1f}%"
                    )

            all_roi_results.append(
                {
                    "roi": roi_dict,
                    "roi_name": roi_name,
                    "n_cells": n_available,
                    "zero_rejection_combos": zero_rej,
                    "all_combos": results,
                }
            )
            continue  # grid search doesn't sample — move to next ROI

        # --- Run sampling ---
        centers = place_windows_along_paths(paths, radius, step)
        logger.info(f"  {len(centers)} window positions, step={step:.0f}um")

        if len(centers) == 0:
            logger.warning(f"  No valid window positions for {roi_name}")
            continue

        windows, n_rejected = run_sampling(
            roi_positions_avail,
            roi_areas_avail,
            roi_indices_avail,
            centers,
            radius,
            area_lo,
            area_hi,
        )

        # Track globally claimed cells
        roi_cells = [c for w in windows for c in w["cell_indices"]]
        if len(roi_cells) != len(set(roi_cells)):
            raise RuntimeError(f"BUG: duplicate cells in {roi_name}")
        used_global.update(roi_cells)
        coverage = len(roi_cells) / n_available * 100 if n_available > 0 else 0

        logger.info(f"  Windows: {len(windows)}, Rejected: {n_rejected}")
        if windows:
            logger.info(
                f"  Cells/window: {min(w['n_cells'] for w in windows)}-{max(w['n_cells'] for w in windows)}"
            )
        logger.info(f"  Sampled: {len(roi_cells)}/{n_available} ({coverage:.1f}%)")

        all_roi_results.append(
            {
                "roi": roi_dict,
                "roi_name": roi_name,
                "n_cells_in_roi": n_cells,
                "n_available": n_available,
                "n_windows": len(windows),
                "n_rejected": n_rejected,
                "n_sampled": len(roi_cells),
                "coverage_pct": round(coverage, 1),
                "windows": windows,
            }
        )

    # --- Grid search: save and return ---
    if args.grid_search:
        out_path = args.output_grid or Path(args.roi).parent / "zero_rejection_combos.json"
        output = {
            "description": "Zero-rejection combos per ROI",
            "roi_file": str(args.roi),
            "detections_file": str(args.detections),
            "target_multiplier": args.target_multiplier,
            "tolerance": args.tolerance,
            "rois": all_roi_results,
        }
        atomic_json_dump(output, out_path)
        logger.info(f"\nSaved: {out_path}")
        return

    # --- Summary ---
    total_windows = sum(r.get("n_windows", 0) for r in all_roi_results)
    total_sampled = sum(r.get("n_sampled", 0) for r in all_roi_results)

    logger.info(f"\n{'='*60}")
    logger.info(
        f"TOTAL: {len(all_roi_results)} ROIs, {total_windows} windows, "
        f"{total_sampled} unique cells sampled"
    )
    if n_excluded_prior > 0:
        logger.info(f"  ({n_excluded_prior} cells excluded from prior session)")
    logger.info("VERIFIED: no cell sampled more than once across all ROIs")

    # Save
    from datetime import datetime

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or Path(args.detections).parent / f"sliding_window_samples_{_ts}.json"
    output = {
        "created": _ts,
        "detections_file": str(args.detections),
        "roi_file": str(args.roi),
        "exclude_cells_file": str(args.exclude_cells) if args.exclude_cells else None,
        "n_excluded_from_prior": n_excluded_prior,
        "window_size_um": round(2 * radius, 1),
        "radius_um": radius,
        "step_um": round(step, 1),
        "overlap_frac": round(overlap, 3),
        "overlap_pct": round(overlap * 100),
        "window_shape": "circular",
        "centerline": "morphological_skeleton",
        "sampling": "spatially_balanced_farthest_point",
        "target_multiplier": args.target_multiplier,
        "tolerance": args.tolerance,
        "n_rois": len(all_roi_results),
        "n_total_windows": total_windows,
        "n_total_unique_cells": total_sampled,
        "rois": all_roi_results,
    }
    atomic_json_dump(output, out_path)
    logger.info(f"Saved: {out_path}")

    # Visualize — combine all ROIs into one plot
    all_verts = [verts for _, verts, _ in roi_entries]
    all_windows = [w for r in all_roi_results for w in r.get("windows", [])]
    all_roi_indices = (
        np.unique(
            np.concatenate([get_roi_cells(positions, areas, poly)[0] for _, _, poly in roi_entries])
        )
        if roi_entries
        else np.array([])
    )

    if args.output_viz is not None:
        viz_path = args.output_viz
    else:
        viz_path = (
            Path(args.detections).parent
            / f"sliding_window_{len(all_roi_results)}rois_{total_windows}win.png"
        )

    # Multi-ROI visualization
    import colorsys

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    golden = (1 + 5**0.5) / 2
    n_win = len(all_windows)
    colors = [colorsys.hsv_to_rgb((i / golden) % 1.0, 0.85, 0.95) for i in range(n_win)]

    fig, ax = plt.subplots(1, 1, figsize=(18, 7))
    ax.set_title(
        f"{len(all_roi_results)} ROIs, {n_win} windows, "
        f"{total_sampled} unique cells | Each cell exactly once",
        fontsize=11,
    )

    for v in all_verts:
        roi_patch = MplPolygon(
            v, closed=True, fill=False, edgecolor="white", linewidth=2, linestyle="--"
        )
        ax.add_patch(roi_patch)
    for path in all_roi_paths:
        ax.plot(path[:, 0], path[:, 1], "w-", linewidth=0.8, alpha=0.3, zorder=1)

    unsampled = [i for i in all_roi_indices if i not in used_global]
    if unsampled:
        ax.scatter(
            positions[unsampled, 0],
            positions[unsampled, 1],
            s=8,
            c="gray",
            alpha=0.5,
            label=f"Unsampled ({len(unsampled)})",
            zorder=1,
            edgecolors="white",
            linewidths=0.5,
        )

    for i, w in enumerate(all_windows):
        c = colors[i]
        circle = plt.Circle(
            (w["center_x"], w["center_y"]),
            radius,
            fill=False,
            edgecolor=c,
            linewidth=1.5,
            alpha=0.7,
            zorder=2,
        )
        ax.add_patch(circle)
        cell_pos = positions[w["cell_indices"]]
        ax.scatter(
            cell_pos[:, 0],
            cell_pos[:, 1],
            s=14,
            c=[c],
            alpha=0.95,
            zorder=3,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.text(
            w["center_x"],
            w["center_y"],
            f"{i}\n{w['n_cells']}c",
            fontsize=4,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=c, alpha=0.7, edgecolor="none"),
        )

    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(viz_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info("Saved visualization: %s", viz_path)


if __name__ == "__main__":
    main()
