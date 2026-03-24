#!/usr/bin/env python3
"""Unified replicate sampling script for paper figures.

Loads scored/classified detections, applies filters (RF score, marker class,
cluster label), then builds area-matched or spatially-clustered replicates and
assigns them to 384-well plate positions.

Two modes:
  Mode 1 (default):  Area-matched replicates via select_cells_for_lmd().
  Mode 2 (--spatial-cluster):  Spatially-clustered replicates via two_stage_clustering().

Output JSON is compatible with lmd_export_replicates.py.

Examples
--------
# Single-slide, filter DCN+ cells, 20 cells × 100 µm² per replicate:
python scripts/paper_figure_sampling.py \\
    --detections /path/to/detections.json \\
    --output-dir /path/to/output \\
    --marker-filter "DCN_class==positive" \\
    --cells-per-rep 20 \\
    --area-per-cell 100

# Multi-slide grouped by slide+tissue, stratified by morphological cluster:
python scripts/paper_figure_sampling.py \\
    --detections /path/to/detections.json \\
    --output-dir /path/to/output \\
    --group-key "slide,tissue" \\
    --cluster-key "morph_cluster" \\
    --score-threshold 0.7

# Spatially-clustered mode:
python scripts/paper_figure_sampling.py \\
    --detections /path/to/detections.json \\
    --output-dir /path/to/output \\
    --spatial-cluster \\
    --cells-per-rep 10 \\
    --area-per-cell 100 \\
    --spatial-radius 500
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from segmentation.lmd.selection import select_cells_for_lmd
from segmentation.lmd.clustering import two_stage_clustering
from segmentation.lmd.well_plate import (
    generate_multiplate_wells,
    insert_empty_wells,
    WELLS_PER_PLATE,
)
from segmentation.utils.json_utils import fast_json_load, atomic_json_dump
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_marker_filter(expr: str):
    """Parse a marker filter expression like 'DCN_class==positive'.

    Returns a callable (detection -> bool) that is True when the detection
    PASSES the filter (i.e. should be INCLUDED).

    Supported operators: ==, !=
    The field is looked up at the top level of the detection dict first, then
    inside features.
    """
    for op in ("!=", "=="):
        if op in expr:
            key, _, value = expr.partition(op)
            key = key.strip()
            value = value.strip()

            def _filter(d, k=key, v=value, o=op):
                raw = d.get(k)
                if raw is None:
                    raw = d.get("features", {}).get(k)
                if raw is None:
                    return False
                raw_str = str(raw)
                return (raw_str == v) if o == "==" else (raw_str != v)

            return _filter
    raise ValueError(
        f"Cannot parse marker filter {expr!r}. "
        "Expected format: 'KEY==VALUE' or 'KEY!=VALUE'."
    )


def _get_field(det: dict, key: str):
    """Look up a field at detection top-level or inside features."""
    val = det.get(key)
    if val is None:
        val = det.get("features", {}).get(key)
    return val


def _get_score(det: dict, score_key: str) -> float:
    v = _get_field(det, score_key)
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _get_area_um2(det: dict) -> float:
    """Return cell area in µm², checking top-level and features."""
    v = det.get("area_um2")
    if v is not None:
        return float(v)
    v = det.get("features", {}).get("area_um2")
    if v is not None:
        return float(v)
    return 0.0


def _auto_pixel_size(detections: list) -> float:
    """Try to infer pixel size from detection features.

    First checks pixel_size_um directly. Falls back to deriving from
    area (px) and area_um2 if both are present: pixel_size = sqrt(area_um2 / area).
    """
    for det in detections[:200]:
        ps = _get_field(det, "pixel_size_um")
        if ps is not None:
            try:
                return float(ps)
            except (TypeError, ValueError):
                pass
    # Fallback: derive from area vs area_um2
    for det in detections[:200]:
        feat = det.get("features", {})
        area_px = feat.get("area")
        area_um2 = feat.get("area_um2")
        if area_px and area_um2 and area_px > 0 and area_um2 > 0:
            import math
            return math.sqrt(area_um2 / area_px)
    return None


def _get_global_center(det: dict):
    """Return (x, y) global center, trying multiple field names."""
    for key in ("global_center", "center"):
        c = det.get(key)
        if c is not None and len(c) >= 2:
            return float(c[0]), float(c[1])
    return None


# ---------------------------------------------------------------------------
# Mode 1: area-matched replicates
# ---------------------------------------------------------------------------

def run_area_matched(
    detections: list,
    *,
    group_keys: list,
    score_key: str,
    score_threshold: float,
    target_area: float,
    max_replicates: int,
    cluster_key: str | None,
    community_key: str | None,
    seed: int,
) -> tuple[list, dict]:
    """Build area-matched replicates using select_cells_for_lmd().

    Returns (replicate_dicts, raw_summary).
    """

    def _group_key_fn(d):
        parts = [str(_get_field(d, k) or "") for k in group_keys]
        return tuple(parts) if len(parts) > 1 else parts[0]

    def _score_fn(d):
        return _get_score(d, score_key)

    # If stratifying by cluster, we subdivide before calling select_cells_for_lmd
    # so that replicates are built within each (group, cluster) partition.
    if cluster_key:
        # Extend the group key to include the cluster label
        def _group_key_fn_with_cluster(d):
            base = _group_key_fn(d)
            cl = str(_get_field(d, cluster_key) or "unassigned")
            if isinstance(base, tuple):
                return base + (cl,)
            return (base, cl)
        effective_group_fn = _group_key_fn_with_cluster
    else:
        effective_group_fn = _group_key_fn

    groups_result, summary = select_cells_for_lmd(
        detections=detections,
        group_key_fn=effective_group_fn,
        score_fn=_score_fn,
        score_threshold=score_threshold,
        target_area=target_area,
        max_replicates=max_replicates,
        seed=seed,
        area_fn=_get_area_um2,
    )

    # Flatten into replicate dicts, computing centroid and community_ids
    uid_to_det = {d["uid"]: d for d in detections if "uid" in d}
    rep_list = []
    rep_id = 0

    for gkey, gdata in sorted(groups_result.items(), key=lambda x: str(x[0])):
        # Parse group and cluster from the composite key
        if cluster_key:
            if isinstance(gkey, tuple) and len(gkey) > len(group_keys):
                group_val = gkey[: len(group_keys)]
                cluster_val = gkey[len(group_keys)] if len(gkey) > len(group_keys) else None
            else:
                group_val = gkey
                cluster_val = None
        else:
            group_val = gkey
            cluster_val = None

        group_str = "/".join(group_val) if isinstance(group_val, tuple) else str(group_val)

        for rep in gdata["replicates"]:
            uids = rep["uids"]
            members = [uid_to_det[u] for u in uids if u in uid_to_det]

            # Centroid in µm: average of global_center × pixel_size
            centroid_um = None
            if members:
                centers = []
                for m in members:
                    c = _get_global_center(m)
                    if c is not None:
                        ps = _get_field(m, "pixel_size_um") or 1.0
                        centers.append((float(c[0]) * float(ps),
                                        float(c[1]) * float(ps)))
                if centers:
                    cx = float(np.mean([c[0] for c in centers]))
                    cy = float(np.mean([c[1] for c in centers]))
                    centroid_um = [round(cx, 2), round(cy, 2)]

            # Community IDs
            community_ids = None
            if community_key:
                cids = list({
                    str(_get_field(m, community_key))
                    for m in members
                    if _get_field(m, community_key) is not None
                })
                community_ids = sorted(cids) if cids else None

            rep_list.append({
                "replicate_id": rep_id,
                "group": group_str,
                "cluster": cluster_val,
                "well": None,   # filled later
                "plate": None,
                "uids": uids,
                "n_cells": len(uids),
                "total_area_um2": round(rep["total_area_um2"], 2),
                "centroid_um": centroid_um,
                "community_ids": community_ids,
            })
            rep_id += 1

    return rep_list, summary


# ---------------------------------------------------------------------------
# Mode 2: spatially-clustered replicates
# ---------------------------------------------------------------------------

def run_spatial_clustered(
    detections: list,
    *,
    group_keys: list,
    score_key: str,
    score_threshold: float,
    cells_per_rep: int,
    area_per_cell: float,
    spatial_radius: float,
    pixel_size: float,
    community_key: str | None,
    seed: int,
) -> tuple[list, dict]:
    """Build spatially-clustered replicates using two_stage_clustering().

    Runs clustering per group defined by group_keys, then maps each spatial
    cluster to a replicate entry.

    Returns (replicate_dicts, raw_summary).
    """
    target_area = cells_per_rep * area_per_cell
    area_min = target_area * 0.75
    area_max = target_area * 1.25
    dist_round1 = spatial_radius
    dist_round2 = spatial_radius * 2.0

    def _group_key_fn(d):
        parts = [str(_get_field(d, k) or "") for k in group_keys]
        return tuple(parts) if len(parts) > 1 else parts[0]

    # Pre-filter by score, build groups
    grouped: dict = {}
    for d in detections:
        if _get_score(d, score_key) < score_threshold:
            continue
        if "uid" not in d:
            continue
        key = _group_key_fn(d)
        grouped.setdefault(key, []).append(d)

    uid_to_det = {d["uid"]: d for d in detections if "uid" in d}
    rep_list = []
    rep_id = 0
    total_clustered = 0
    total_outliers = 0

    for gkey in sorted(grouped.keys(), key=str):
        group_dets = grouped[gkey]
        group_str = "/".join(gkey) if isinstance(gkey, tuple) else str(gkey)

        result = two_stage_clustering(
            group_dets,
            pixel_size=pixel_size,
            area_min=area_min,
            area_max=area_max,
            dist_round1=dist_round1,
            dist_round2=dist_round2,
            min_score=score_threshold,
        )

        total_clustered += result["summary"]["n_detections_in_clusters"]
        total_outliers += result["summary"]["n_singles"]

        for cl in result["main_clusters"]:
            idxs = cl["detection_indices"]
            members = [group_dets[i] for i in idxs if i < len(group_dets)]
            uids = [m["uid"] for m in members]

            centroid_um = None
            if pixel_size and pixel_size > 0:
                centroid_um = [
                    round(cl["cx"] * pixel_size, 2),
                    round(cl["cy"] * pixel_size, 2),
                ]

            community_ids = None
            if community_key:
                cids = list({
                    str(_get_field(m, community_key))
                    for m in members
                    if _get_field(m, community_key) is not None
                })
                community_ids = sorted(cids) if cids else None

            rep_list.append({
                "replicate_id": rep_id,
                "group": group_str,
                "cluster": None,
                "well": None,
                "plate": None,
                "uids": uids,
                "n_cells": len(uids),
                "total_area_um2": round(cl["total_area_um2"], 2),
                "centroid_um": centroid_um,
                "community_ids": community_ids,
            })
            rep_id += 1

    total_filtered = sum(len(v) for v in grouped.values())
    total_selected = sum(r["n_cells"] for r in rep_list)

    raw_summary = {
        "total_cells_filtered": total_filtered,
        "total_replicates": len(rep_list),
        "total_cells_selected": total_selected,
        "total_outliers_unclustered": total_outliers,
        "groups_with_max_reps": 0,   # not applicable in spatial mode
        "groups_with_fewer_reps": 0,
        "groups_with_no_reps": 0,
    }

    return rep_list, raw_summary


# ---------------------------------------------------------------------------
# Well assignment
# ---------------------------------------------------------------------------

def assign_wells(rep_list: list, *, empty_pct: float, seed: int) -> tuple[list, list]:
    """Shuffle replicates, assign well addresses, insert blank wells.

    Returns (rep_list_with_wells, blanks_list).
    """
    if not rep_list:
        return rep_list, []

    # Shuffle replicates to avoid systematic plate-position bias in mass spec
    rng = np.random.default_rng(seed)
    rng.shuffle(rep_list)

    n_reps = len(rep_list)
    n_blanks = max(1, math.ceil(n_reps * empty_pct / 100))
    total_wells = n_reps + n_blanks

    n_plates = math.ceil(total_wells / WELLS_PER_PLATE)
    if n_plates > 1:
        logger.info(
            f"  {total_wells} total wells ({n_reps} samples + {n_blanks} blanks) "
            f"— using {n_plates} plate(s)"
        )

    plate_wells = generate_multiplate_wells(total_wells)
    _, blank_positions = insert_empty_wells(
        plate_wells, n_reps, empty_pct=empty_pct, seed=seed
    )

    blanks_list = []
    n_non_blank = len(plate_wells) - len(blank_positions)
    assert n_non_blank == len(rep_list), (
        f"Well count mismatch: {len(plate_wells)} total - {len(blank_positions)} blanks "
        f"= {n_non_blank} non-blank, but {len(rep_list)} replicates"
    )

    rep_iter = iter(rep_list)
    for pos, (plate_num, well) in enumerate(plate_wells):
        if pos in blank_positions:
            blanks_list.append(well)
        else:
            rep = next(rep_iter)
            rep["well"] = well
            rep["plate"] = plate_num

    # Verify all replicates got assigned
    unassigned = [r for r in rep_list if r["plate"] is None]
    if unassigned:
        logger.error(f"{len(unassigned)} replicates not assigned to wells")

    return rep_list, blanks_list


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(rep_list: list, blanks_list: list, *, group_keys: list):
    """Print a per-group replicate count table plus final totals."""
    if not rep_list:
        print("No replicates generated.")
        return

    # Group stats
    from collections import defaultdict
    group_stats: dict = defaultdict(lambda: {"reps": 0, "cells": 0, "area": 0.0})
    for r in rep_list:
        key = (r["group"], r["cluster"])
        group_stats[key]["reps"] += 1
        group_stats[key]["cells"] += r["n_cells"]
        group_stats[key]["area"] += r["total_area_um2"]

    col_group = max(len("Group"), max(len(str(k[0])) for k in group_stats))
    col_cluster = max(len("Cluster"), max(len(str(k[1] or "-")) for k in group_stats))
    col_group = min(col_group, 40)
    col_cluster = min(col_cluster, 20)

    header = (
        f"{'Group':<{col_group}}  {'Cluster':<{col_cluster}}  "
        f"{'Reps':>6}  {'Cells':>7}  {'Area µm²':>10}"
    )
    print()
    print(header)
    print("-" * len(header))

    for key in sorted(group_stats.keys(), key=lambda k: (str(k[0]), str(k[1]))):
        g, cl = key
        s = group_stats[key]
        cl_str = str(cl) if cl is not None else "-"
        print(
            f"{str(g):<{col_group}}  {cl_str:<{col_cluster}}  "
            f"{s['reps']:>6}  {s['cells']:>7}  {s['area']:>10.0f}"
        )

    print("-" * len(header))
    total_reps = len(rep_list)
    total_cells = sum(r["n_cells"] for r in rep_list)
    total_area = sum(r["total_area_um2"] for r in rep_list)
    print(
        f"{'TOTAL':<{col_group}}  {'':<{col_cluster}}  "
        f"{total_reps:>6}  {total_cells:>7}  {total_area:>10.0f}"
    )
    print(f"\n  Blank (QC) wells: {len(blanks_list)}")
    n_plates = max((r["plate"] or 1) for r in rep_list) if rep_list else 1
    total_wells = total_reps + len(blanks_list)
    print(f"  Total wells: {total_wells} across {n_plates} plate(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument(
        "--detections", required=True, type=Path,
        help="Path to detections JSON (scored + classified)",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory (created if missing)",
    )

    # Replicate geometry
    parser.add_argument(
        "--cells-per-rep", type=int, default=20,
        help="Target number of cells per replicate (default: 20)",
    )
    parser.add_argument(
        "--area-per-cell", type=float, default=100.0,
        help="Target area per cell in µm² (default: 100)",
    )

    # Filtering
    parser.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Minimum RF score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--score-key", default="rf_prediction",
        help="Key for RF score in detection dict (default: rf_prediction)",
    )
    parser.add_argument(
        "--marker-filter", default=None,
        help=(
            "Filter by marker class expression, e.g. 'DCN_class==positive' "
            "or 'MSLN_class!=negative'. Multiple filters: "
            "'DCN_class==positive,MSLN_class==positive' (AND logic)."
        ),
    )

    # Grouping / stratification
    parser.add_argument(
        "--group-key", default="slide",
        help=(
            "Comma-separated detection fields to group by "
            "(default: 'slide'). E.g. 'slide,bone'."
        ),
    )
    parser.add_argument(
        "--cluster-key", default=None,
        help=(
            "Field for morphological cluster label (e.g. 'morph_cluster'). "
            "If set, replicates are built within each (group, cluster) partition."
        ),
    )
    parser.add_argument(
        "--community-key", default=None,
        help=(
            "Optional field tracking vessel/spatial community IDs "
            "(e.g. 'vessel_community_id'). Preserved in output as community_ids list."
        ),
    )

    # Mode
    parser.add_argument(
        "--spatial-cluster", action="store_true",
        help=(
            "Use spatially-clustered mode (two_stage_clustering) instead of "
            "area-matched random mode."
        ),
    )
    parser.add_argument(
        "--spatial-radius", type=float, default=500.0,
        help="Spatial clustering radius in µm for Mode 2 (default: 500)",
    )

    # Limits
    parser.add_argument(
        "--max-replicates", type=int, default=999,
        help=(
            "Maximum replicates per group in Mode 1 (default: 999 = unlimited). "
            "Not used in spatial-cluster mode."
        ),
    )

    # Well plate
    parser.add_argument(
        "--empty-pct", type=float, default=10.0,
        help="QC empty well percentage (default: 10)",
    )

    # Pixel size
    parser.add_argument(
        "--pixel-size", type=float, default=None,
        help=(
            "Pixel size in µm (default: auto-detect from features). "
            "Required for spatial-cluster mode if not in features."
        ),
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load detections
    # ------------------------------------------------------------------
    logger.info(f"Loading detections from {args.detections}")
    detections = fast_json_load(str(args.detections))
    if not isinstance(detections, list):
        # Some outputs wrap in a dict
        detections = detections.get("detections", [])
    logger.info(f"  Loaded {len(detections):,} detections")

    # ------------------------------------------------------------------
    # Pixel size
    # ------------------------------------------------------------------
    pixel_size = args.pixel_size
    if pixel_size is None:
        pixel_size = _auto_pixel_size(detections)
    if pixel_size is None:
        if args.spatial_cluster:
            parser.error(
                "Could not auto-detect pixel size from features. "
                "Please supply --pixel-size."
            )
        else:
            pixel_size = 1.0  # area_um2 already in µm², pixel_size only used for centroids
            logger.warning(
                "Could not auto-detect pixel size; defaulting to 1.0 µm/px. "
                "Centroid coordinates may be in pixels rather than µm."
            )

    logger.info(f"  Pixel size: {pixel_size} µm/px")

    # ------------------------------------------------------------------
    # Parse filters
    # ------------------------------------------------------------------
    group_keys = [k.strip() for k in args.group_key.split(",") if k.strip()]
    if not group_keys:
        group_keys = ["slide"]

    marker_filters = []
    if args.marker_filter:
        for expr in args.marker_filter.split(","):
            expr = expr.strip()
            if expr:
                marker_filters.append(_parse_marker_filter(expr))

    # ------------------------------------------------------------------
    # Apply marker filters up front (score filter done inside selection)
    # ------------------------------------------------------------------
    if marker_filters:
        before = len(detections)
        detections = [d for d in detections if all(f(d) for f in marker_filters)]
        logger.info(
            f"  Marker filter '{args.marker_filter}': "
            f"{before:,} → {len(detections):,} detections"
        )

    if not detections:
        logger.warning("No detections remain after filtering. Nothing to output.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Run sampling
    # ------------------------------------------------------------------
    target_area = args.cells_per_rep * args.area_per_cell

    if args.spatial_cluster:
        logger.info(
            f"Mode: spatially-clustered  "
            f"(target {args.cells_per_rep} cells × {args.area_per_cell} µm² = "
            f"{target_area:.0f} µm², radius={args.spatial_radius} µm)"
        )
        rep_list, raw_summary = run_spatial_clustered(
            detections,
            group_keys=group_keys,
            score_key=args.score_key,
            score_threshold=args.score_threshold,
            cells_per_rep=args.cells_per_rep,
            area_per_cell=args.area_per_cell,
            spatial_radius=args.spatial_radius,
            pixel_size=pixel_size,
            community_key=args.community_key,
            seed=args.seed,
        )
    else:
        logger.info(
            f"Mode: area-matched  "
            f"(target {args.cells_per_rep} cells × {args.area_per_cell} µm² = "
            f"{target_area:.0f} µm², max_reps={args.max_replicates})"
        )
        rep_list, raw_summary = run_area_matched(
            detections,
            group_keys=group_keys,
            score_key=args.score_key,
            score_threshold=args.score_threshold,
            target_area=target_area,
            max_replicates=args.max_replicates,
            cluster_key=args.cluster_key,
            community_key=args.community_key,
            seed=args.seed,
        )

    logger.info(
        f"  Built {len(rep_list)} replicates "
        f"({raw_summary['total_cells_selected']:,} cells selected from "
        f"{raw_summary['total_cells_filtered']:,} filtered)"
    )

    # ------------------------------------------------------------------
    # Assign wells
    # ------------------------------------------------------------------
    rep_list, blanks_list = assign_wells(
        rep_list, empty_pct=args.empty_pct, seed=args.seed
    )

    n_plates = max((r["plate"] or 1) for r in rep_list) if rep_list else 1
    total_wells = len(rep_list) + len(blanks_list)

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    output = {
        "parameters": {
            "cells_per_rep": args.cells_per_rep,
            "area_per_cell_um2": args.area_per_cell,
            "target_area_um2": target_area,
            "score_threshold": args.score_threshold,
            "score_key": args.score_key,
            "marker_filter": args.marker_filter,
            "group_key": args.group_key,
            "cluster_key": args.cluster_key,
            "community_key": args.community_key,
            "spatial_cluster_mode": args.spatial_cluster,
            "spatial_radius_um": args.spatial_radius if args.spatial_cluster else None,
            "max_replicates": args.max_replicates if not args.spatial_cluster else None,
            "empty_pct": args.empty_pct,
            "pixel_size_um": pixel_size,
            "seed": args.seed,
        },
        "summary": {
            "total_input": len(detections),
            "total_filtered": raw_summary["total_cells_filtered"],
            "total_selected": raw_summary["total_cells_selected"],
            "total_replicates": len(rep_list),
            "total_wells": total_wells,
            "blank_wells": len(blanks_list),
            "total_plates": n_plates,
        },
        "replicates": rep_list,
        "blanks": blanks_list,
    }

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "sampling_results.json"
    atomic_json_dump(output, str(out_path))
    logger.info(f"Wrote sampling results to {out_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print_summary_table(rep_list, blanks_list, group_keys=group_keys)
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
