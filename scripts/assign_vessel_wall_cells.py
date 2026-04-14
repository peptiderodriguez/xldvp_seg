#!/usr/bin/env python
"""Assign marker+ cells to vessel lumens for LMD replicate sampling.

For each vessel lumen, finds the marker-positive cells in a wall ring around
the lumen (inner radius = equiv_r * ring_min_factor, outer = equiv_r + ring_pad_um).
Stores per-marker counts, ordered cell UIDs (sorted by marker intensity, brightest
first — for LMD selection), and replicate counts (n_marker // replicate_size).

Output: same vessels JSON with added fields per marker:
    n_{marker}_wall           — total marker+ cells in ring
    {marker}_wall_cell_uids   — ordered list of cell UIDs (brightest first)
    n_{marker}_replicates     — floor(n_{marker}_wall / replicate_size)
And aggregate fields:
    n_replicates_total        — sum of per-marker replicates
    n_marker_wall             — union of all marker+ cells in ring (for back-compat)

Usage:
    python scripts/assign_vessel_wall_cells.py \\
        --vessels vessel_lumens_scored.json \\
        --cells cell_detections_snr2_markers.json \\
        --markers "SMA,LYVE1" \\
        --output vessel_lumens_with_cells.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _cell_intensity(cell: dict, marker: str) -> float:
    """Return the brightness of a cell for the given marker.

    Prefers marker-specific corrected/raw intensity fields, falls back to 0.
    """
    feats = cell.get("features", {})
    for key in (f"{marker}_value", f"{marker}_raw"):
        v = feats.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _build_marker_index(
    cells: list[dict], markers: list[str]
) -> dict[str, tuple[np.ndarray, list[str], list[float]]]:
    """Build per-marker (positions, uids, intensities) arrays.

    Returns dict mapping marker name -> (positions Nx2, uids, intensities).
    """
    out = {}
    for m in markers:
        key = f"{m}_class"
        pos, uids, inten = [], [], []
        for c in cells:
            feats = c.get("features", {})
            if feats.get(key) != "positive":
                continue
            p = c.get("global_center_um")
            if not p:
                continue
            pos.append(p)
            uids.append(c.get("uid") or c.get("id") or f"cell_{len(uids)}")
            inten.append(_cell_intensity(c, m))
        out[m] = (
            np.asarray(pos) if pos else np.empty((0, 2)),
            uids,
            inten,
        )
        logger.info("  Marker %s+: %d cells", m, len(uids))
    return out


def assign_wall_cells(
    vessels: list[dict],
    cells: list[dict],
    markers: list[str],
    ring_min_factor: float = 0.5,
    ring_pad_um: float = 20.0,
    replicate_size: int = 8,
) -> list[dict]:
    """Add per-marker wall cell counts + UIDs + replicate counts to each vessel.

    Args:
        vessels: list of vessel lumen dicts (must have centroid_x_um, centroid_y_um,
                 equiv_diameter_um). Falls back to bbox center * pixel_size if
                 centroids absent.
        cells: list of cell detection dicts (with features[{marker}_class]).
        markers: list of marker names to assign (e.g. ["SMA", "LYVE1"]).
        ring_min_factor: inner ring radius = equiv_r * ring_min_factor (default 0.5
                         excludes cells inside the lumen).
        ring_pad_um: outer ring radius = equiv_r + ring_pad_um (default 20um).
        replicate_size: cells per replicate (default 8).

    Returns:
        Augmented vessels list (mutates in place).
    """
    marker_idx = _build_marker_index(cells, markers)
    trees = {
        m: (cKDTree(p), uids, inten) for m, (p, uids, inten) in marker_idx.items() if len(p) > 0
    }

    n_assigned = 0
    for v in vessels:
        # Centroid in um
        cx = v.get("centroid_x_um")
        cy = v.get("centroid_y_um")
        if cx is None or cy is None:
            # Derive from bbox if refined_scale present
            rs = v.get("refined_scale")
            if rs is None:
                logger.warning("Vessel missing centroid + refined_scale, skipping")
                continue
            # Pixel size per scale (base_px = 0.1725 is CZI default; override via cli if needed)
            # NOTE: base pixel size must be known — we infer from area ratio if possible
            area_um2 = v.get("area_um2", 0)
            area_px = v.get("area_px", 1)
            if area_px > 0 and area_um2 > 0:
                ps = np.sqrt(area_um2 / area_px)
                cx = (v["bbox_x"] + v.get("bbox_w", 0) / 2) * ps
                cy = (v["bbox_y"] + v.get("bbox_h", 0) / 2) * ps
            else:
                continue

        equiv_r = v.get("equiv_diameter_um", 0) / 2
        if equiv_r <= 0:
            equiv_r = np.sqrt(v.get("area_um2", 0) / np.pi)

        r_inner = equiv_r * ring_min_factor
        r_outer = equiv_r + ring_pad_um

        all_wall_uids: set[str] = set()
        total_reps = 0
        # Initialize all markers to zero (even if no cells of that marker exist)
        for m in markers:
            v.setdefault(f"n_{m}_wall", 0)
            v.setdefault(f"{m}_wall_cell_uids", [])
            v.setdefault(f"n_{m}_replicates", 0)
        for m, (tree, uids, inten) in trees.items():
            idx_outer = tree.query_ball_point([cx, cy], r_outer)
            if not idx_outer:
                v[f"n_{m}_wall"] = 0
                v[f"{m}_wall_cell_uids"] = []
                v[f"n_{m}_replicates"] = 0
                continue
            # Filter to ring: distance > r_inner
            pts = tree.data[idx_outer]
            dists = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
            keep = [i for i, d in zip(idx_outer, dists) if d > r_inner]
            if not keep:
                v[f"n_{m}_wall"] = 0
                v[f"{m}_wall_cell_uids"] = []
                v[f"n_{m}_replicates"] = 0
                continue
            # Sort by intensity descending (brightest first for LMD)
            keep_sorted = sorted(keep, key=lambda i: -inten[i])
            kept_uids = [uids[i] for i in keep_sorted]
            v[f"n_{m}_wall"] = len(kept_uids)
            v[f"{m}_wall_cell_uids"] = kept_uids
            v[f"n_{m}_replicates"] = len(kept_uids) // replicate_size
            all_wall_uids.update(kept_uids)
            total_reps += v[f"n_{m}_replicates"]

        v["n_marker_wall"] = len(all_wall_uids)  # union count for back-compat
        v["n_replicates_total"] = total_reps
        if total_reps > 0:
            n_assigned += 1

    logger.info("Vessels with ≥1 replicate: %d / %d", n_assigned, len(vessels))
    return vessels


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--vessels", type=Path, required=True, help="Input vessels JSON")
    p.add_argument(
        "--cells",
        type=Path,
        required=True,
        help="Cell detections JSON (with marker classifications)",
    )
    p.add_argument(
        "--markers",
        type=str,
        required=True,
        help="Comma-separated marker names (e.g. 'SMA,CD31' or 'SMA,LYVE1'). "
        "For each, cells with features['{M}_class']=='positive' are used.",
    )
    p.add_argument("--output", type=Path, required=True, help="Output augmented vessels JSON")
    p.add_argument("--replicate-size", type=int, default=8, help="Cells per replicate (default: 8)")
    p.add_argument(
        "--ring-min-factor",
        type=float,
        default=0.5,
        help="Inner ring radius = equiv_r * factor (default: 0.5, excludes lumen interior)",
    )
    p.add_argument(
        "--ring-pad-um",
        type=float,
        default=20.0,
        help="Outer ring radius = equiv_r + pad_um (default: 20.0)",
    )
    p.add_argument(
        "--filter-min-replicates",
        type=int,
        default=0,
        help="Drop vessels with fewer than N total replicates (default: 0 = keep all)",
    )
    args = p.parse_args()

    markers = [m.strip() for m in args.markers.split(",") if m.strip()]
    if not markers:
        raise SystemExit("Must specify at least one marker via --markers")

    logger.info("Loading vessels: %s", args.vessels)
    vessels = fast_json_load(str(args.vessels))
    logger.info("  %d vessels", len(vessels))

    logger.info("Loading cells: %s", args.cells)
    cells = fast_json_load(str(args.cells))
    logger.info("  %d cells", len(cells))

    logger.info("Markers: %s, replicate_size=%d", markers, args.replicate_size)
    vessels = assign_wall_cells(
        vessels,
        cells,
        markers,
        ring_min_factor=args.ring_min_factor,
        ring_pad_um=args.ring_pad_um,
        replicate_size=args.replicate_size,
    )

    if args.filter_min_replicates > 0:
        before = len(vessels)
        vessels = [
            v for v in vessels if v.get("n_replicates_total", 0) >= args.filter_min_replicates
        ]
        logger.info(
            "Filter ≥%d replicates: %d → %d", args.filter_min_replicates, before, len(vessels)
        )

    # Summary per marker
    for m in markers:
        n_any = sum(1 for v in vessels if v.get(f"n_{m}_wall", 0) > 0)
        n_ge8 = sum(1 for v in vessels if v.get(f"n_{m}_wall", 0) >= args.replicate_size)
        total_reps = sum(v.get(f"n_{m}_replicates", 0) for v in vessels)
        total_cells = sum(v.get(f"n_{m}_wall", 0) for v in vessels)
        logger.info(
            "  %s: %d vessels with ≥1 cell, %d with ≥%d cells, %d replicates, %d cells",
            m,
            n_any,
            n_ge8,
            args.replicate_size,
            total_reps,
            total_cells,
        )

    total = sum(v.get("n_replicates_total", 0) for v in vessels)
    logger.info("Total replicates across all markers: %d", total)

    logger.info("Writing: %s", args.output)
    atomic_json_dump(vessels, str(args.output))


if __name__ == "__main__":
    main()
