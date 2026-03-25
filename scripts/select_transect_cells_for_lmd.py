#!/usr/bin/env python3
"""Select cells along hepatic zonation transect paths for LMD export.

Bridges the output of zonation_transect.py with run_lmd_export.py.

Workflow:
1. Load zonation_transect.csv (cell UIDs + path_id + fractional_pos)
2. Load full detection JSON (to get contours and features)
3. Filter cells: only those present in the transect CSV
4. Apply optional filters: --score-threshold, --path-filter, --frac-range, --max-cells
5. Assign wells (384-well serpentine, with empty QC wells)
6. Write a filtered detection JSON compatible with run_lmd_export.py
7. Write a summary JSON and a per-cell CSV (uid, path_id, frac_pos, well, plate)

Usage:
    python scripts/select_transect_cells_for_lmd.py \\
        --transect-csv zonation_liver/zonation_transect.csv \\
        --detections cell_detections_postdedup.json \\
        --output-dir lmd_transect/
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

from segmentation.lmd.well_plate import generate_multiplate_wells, insert_empty_wells
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_transect_csv(csv_path):
    """Load zonation_transect.csv into a dict keyed by cell_uid.

    Returns
    -------
    dict[str, dict]
        {uid: {'path_id': str, 'path_name': str, 'fractional_pos': float,
               <marker>: float, ...}}
    """
    records = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "cell_uid" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV missing required column 'cell_uid'. " f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            uid = row["cell_uid"]
            entry = {
                "path_id": row["path_id"],
                "path_name": row.get("path_name", row["path_id"]),
                "fractional_pos": float(row["fractional_pos"]),
            }
            # Copy any additional marker columns (e.g. DCN, GluI, Pck1)
            for key, val in row.items():
                if key not in ("cell_uid", "path_id", "path_name", "fractional_pos"):
                    try:
                        entry[key] = float(val)
                    except (ValueError, TypeError):
                        entry[key] = val
            records[uid] = entry
    return records


def load_zonation_scores(scores_csv_path):
    """Optionally load zonation_scores.csv for global zonation scores.

    Returns
    -------
    dict[str, float]  — {uid: zonation_score} or empty dict if file absent
    """
    if scores_csv_path is None or not Path(scores_csv_path).exists():
        return {}
    scores = {}
    with open(scores_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("cell_uid", "")
            try:
                scores[uid] = float(row.get("zonation_score", 0.0))
            except (ValueError, TypeError):
                pass
    return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_score(det, score_key):
    """Get RF score from a detection dict.

    Checks top-level and features dict for the specified key.
    Returns 1.0 (pass-through) if no score key is present at all,
    so unscored detections are always included.
    """
    val = det.get(score_key)
    if val is None:
        val = det.get("features", {}).get(score_key)
    if val is None:
        return 1.0  # no score key present — pass through
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _stratified_sample(uids, transect_records, n_target, seed=42):
    """Sample n_target UIDs proportionally from each path.

    Each path gets floor(n_target * path_fraction) cells, with remainder
    allocated to the largest paths first.
    """
    rng = random.Random(seed)

    # Group by path
    path_to_uids = {}
    for uid in uids:
        pid = transect_records[uid]["path_id"]
        path_to_uids.setdefault(pid, []).append(uid)

    n_paths = len(path_to_uids)
    n_total = len(uids)

    # Compute per-path allocations
    allocations = {}
    for pid, puids in path_to_uids.items():
        frac = len(puids) / n_total
        allocations[pid] = int(frac * n_target)

    # Distribute remainder to largest paths
    allocated = sum(allocations.values())
    remainder = n_target - allocated
    if remainder > 0:
        sorted_by_size = sorted(
            path_to_uids.keys(), key=lambda p: len(path_to_uids[p]), reverse=True
        )
        for i in range(remainder):
            allocations[sorted_by_size[i % n_paths]] += 1

    # Sample from each path
    selected = []
    for pid, puids in path_to_uids.items():
        k = min(allocations[pid], len(puids))
        if k > 0:
            selected.extend(rng.sample(puids, k))

    if len(selected) < n_target:
        logger.warning(
            f"Stratified sample yielded {len(selected)} cells "
            f"(requested {n_target}) due to per-path capacity limits"
        )

    return selected


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def parse_frac_range(frac_range_str):
    """Parse '0.2,0.8' -> (0.2, 0.8)."""
    if frac_range_str is None:
        return (0.0, 1.0)
    parts = [p.strip() for p in frac_range_str.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--frac-range must be 'low,high', got: {frac_range_str!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"--frac-range values must satisfy 0 <= low < high <= 1, got {lo},{hi}")
    return (lo, hi)


def parse_path_filter(path_filter_str):
    """Parse 'auto_cv0_pv7,auto_cv1_pv2' -> set of path ids."""
    if path_filter_str is None:
        return None
    return {p.strip() for p in path_filter_str.split(",") if p.strip()}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Select transect cells for LMD export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--transect-csv",
        required=True,
        help="Path to zonation_transect.csv from zonation_transect.py",
    )
    parser.add_argument(
        "--detections",
        required=True,
        help="Path to full detections JSON (to extract contours and features)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for filtered JSON, summary JSON, and CSV",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Min RF score for cell quality (default: 0.5)",
    )
    parser.add_argument(
        "--score-key",
        default="rf_prediction",
        help="Feature key for RF score (default: rf_prediction)",
    )
    parser.add_argument(
        "--path-filter", default=None, help="Only include specific paths (comma-separated path_ids)"
    )
    parser.add_argument(
        "--top-paths",
        type=int,
        default=None,
        help="Use only the top N paths ranked by gradient quality × cell count. "
        "Gradient = Spearman correlation of markers vs fractional position.",
    )
    parser.add_argument(
        "--gradient-markers",
        default=None,
        help="Comma-separated marker pair for gradient scoring (default: auto-detect "
        "from CSV columns, excluding cell_uid/path_id/fractional_pos)",
    )
    parser.add_argument(
        "--frac-range",
        default=None,
        help='Fractional position range, e.g. "0.0,1.0" (default: all)',
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Max cells to select (randomly sampled, default: unlimited)",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=None,
        help="Take every Nth cell along each path (e.g., 3 = every 3rd cell). "
        "Applied after sorting by fractional position within each path.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling (default: 42)"
    )
    parser.add_argument(
        "--empty-pct",
        type=float,
        default=10.0,
        help="Percentage of QC empty wells to insert (default: 10)",
    )
    parser.add_argument(
        "--no-empty-wells", action="store_true", help="Disable QC empty well insertion"
    )
    parser.add_argument(
        "--zonation-scores-csv",
        default=None,
        help="Path to zonation_scores.csv (optional, to attach global scores)",
    )

    args = parser.parse_args()
    setup_logging()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load transect CSV
    # -----------------------------------------------------------------------
    transect_path = Path(args.transect_csv)
    logger.info(f"Loading transect CSV: {transect_path}")
    transect_records = load_transect_csv(transect_path)
    n_csv_total = len(transect_records)
    logger.info(f"Transect CSV: {n_csv_total:,} cells across all paths")

    # Count per-path before any filtering
    path_counts_raw = {}
    for rec in transect_records.values():
        pid = rec["path_id"]
        path_counts_raw[pid] = path_counts_raw.get(pid, 0) + 1
    logger.info(f"  Paths in CSV: {len(path_counts_raw)}")

    # Optional zonation scores
    global_scores = load_zonation_scores(args.zonation_scores_csv)
    if global_scores:
        logger.info(f"Loaded global zonation scores for {len(global_scores):,} cells")

    # -----------------------------------------------------------------------
    # Step 1b: Rank paths by gradient quality and select top N (if requested)
    # -----------------------------------------------------------------------
    if args.top_paths is not None:
        from scipy.stats import spearmanr

        # Group cells by path
        by_path = {}
        for uid, rec in transect_records.items():
            by_path.setdefault(rec["path_id"], []).append(rec)

        # Detect marker columns for gradient scoring
        skip_cols = {"path_id", "path_name", "fractional_pos"}
        sample_rec = next(iter(transect_records.values()))
        if args.gradient_markers:
            marker_cols = [m.strip() for m in args.gradient_markers.split(",")]
        else:
            marker_cols = [
                k
                for k in sample_rec
                if k not in skip_cols and isinstance(sample_rec[k], (int, float))
            ]

        # Score each path: gradient strength × cell count
        path_scores = []
        for pid, cells in by_path.items():
            n = len(cells)
            if n < 10:
                continue
            fpos = [c["fractional_pos"] for c in cells]
            rhos = []
            for mk in marker_cols:
                vals = [c.get(mk, 0) for c in cells]
                if all(isinstance(v, (int, float)) for v in vals):
                    rho, _ = spearmanr(fpos, vals)
                    rhos.append(abs(rho))
            gradient = float(sum(rhos) / len(rhos)) if rhos else 0.0
            path_scores.append((pid, n, gradient, gradient * n))

        path_scores.sort(key=lambda x: x[3], reverse=True)
        top_pids = {s[0] for s in path_scores[: args.top_paths]}

        before = len(transect_records)
        transect_records = {
            uid: rec for uid, rec in transect_records.items() if rec["path_id"] in top_pids
        }
        logger.info(
            f"Top {args.top_paths} paths (by gradient × size): "
            f"{before:,} -> {len(transect_records):,} cells"
        )
        for pid, n, grad, score in path_scores[: args.top_paths]:
            logger.info(f"    {pid}: {n} cells, gradient={grad:.3f}")

    # -----------------------------------------------------------------------
    # Step 2: Apply pre-filters to the transect records
    #         (path filter + fractional range) before touching detections JSON
    # -----------------------------------------------------------------------
    path_filter_set = parse_path_filter(args.path_filter)
    frac_lo, frac_hi = parse_frac_range(args.frac_range)

    if path_filter_set is not None:
        before = len(transect_records)
        transect_records = {
            uid: rec for uid, rec in transect_records.items() if rec["path_id"] in path_filter_set
        }
        logger.info(
            f"Path filter ({', '.join(sorted(path_filter_set))}): "
            f"{before:,} -> {len(transect_records):,} cells"
        )

    if frac_lo > 0.0 or frac_hi < 1.0:
        before = len(transect_records)
        transect_records = {
            uid: rec
            for uid, rec in transect_records.items()
            if frac_lo <= rec["fractional_pos"] <= frac_hi
        }
        logger.info(
            f"Frac range [{frac_lo:.3f}, {frac_hi:.3f}]: "
            f"{before:,} -> {len(transect_records):,} cells"
        )

    if not transect_records:
        logger.error("No cells remain after pre-filters — aborting")
        sys.exit(1)

    # UIDs to look up in the detections JSON
    target_uids = set(transect_records.keys())

    # -----------------------------------------------------------------------
    # Step 3: Load detections JSON — single pass, build UID lookup
    # -----------------------------------------------------------------------
    det_path = Path(args.detections)
    if not det_path.exists():
        logger.error(f"Detections file not found: {det_path}")
        sys.exit(1)
    logger.info(f"Loading detections: {det_path} " f"({det_path.stat().st_size / 1e9:.1f} GB)")
    raw = fast_json_load(str(det_path))
    if isinstance(raw, dict):
        detections = raw.get("detections", raw.get("cells", []))
    elif isinstance(raw, list):
        detections = raw
    else:
        logger.error(f"Unexpected detections format: {type(raw)}")
        sys.exit(1)
    logger.info(f"Loaded {len(detections):,} detections")

    # Single-pass lookup: only keep detections whose uid is in target_uids
    uid_to_det = {}
    for det in detections:
        uid = det.get("uid", det.get("id", ""))
        if uid in target_uids:
            uid_to_det[uid] = det
    logger.info(
        f"Matched {len(uid_to_det):,} / {len(target_uids):,} transect UIDs " f"in detections JSON"
    )

    missing = target_uids - uid_to_det.keys()
    if missing:
        logger.warning(
            f"  {len(missing):,} transect UIDs not found in detections JSON "
            f"(detections may have been filtered/deduplicated differently)"
        )

    # -----------------------------------------------------------------------
    # Step 4: Score filter on matched detections
    # -----------------------------------------------------------------------
    if args.score_threshold > 0.0:
        before = len(uid_to_det)
        score_key = args.score_key
        uid_to_det = {
            uid: det
            for uid, det in uid_to_det.items()
            if _get_score(det, score_key) >= args.score_threshold
        }
        logger.info(
            f"Score filter (>= {args.score_threshold} on '{score_key}'): "
            f"{before:,} -> {len(uid_to_det):,} cells"
        )

    if not uid_to_det:
        logger.error("No cells remain after score filter — aborting")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 5: Optional max-cells subsampling (stratified by path)
    # -----------------------------------------------------------------------
    selected_uids = list(uid_to_det.keys())

    if args.max_cells is not None and len(selected_uids) > args.max_cells:
        logger.info(
            f"Subsampling {len(selected_uids):,} -> {args.max_cells} cells "
            f"(stratified by path, seed={args.seed})"
        )
        selected_uids = _stratified_sample(
            selected_uids, transect_records, args.max_cells, seed=args.seed
        )
        logger.info(f"After subsampling: {len(selected_uids):,} cells")

    # Sort by path_id then fractional_pos for reproducible well ordering
    selected_uids.sort(
        key=lambda u: (
            transect_records[u]["path_id"],
            transect_records[u]["fractional_pos"],
        )
    )

    # Take every Nth cell per path (preserves spatial distribution along gradient)
    if args.every_nth is not None and args.every_nth > 1:
        before = len(selected_uids)
        by_path = {}
        for u in selected_uids:
            pid = transect_records[u]["path_id"]
            by_path.setdefault(pid, []).append(u)
        selected_uids = []
        for pid in sorted(by_path):
            selected_uids.extend(by_path[pid][:: args.every_nth])
        logger.info(
            f"Every-{args.every_nth} subsampling: {before:,} -> {len(selected_uids):,} cells"
        )

    n_selected = len(selected_uids)
    logger.info(f"Final selection: {n_selected:,} cells")

    # -----------------------------------------------------------------------
    # Step 6: Assign wells
    #
    # insert_empty_wells() marks positions within an existing list as empty.
    # The caller must pre-allocate n_samples + n_empty total wells so every
    # real cell still gets a unique well slot.
    # -----------------------------------------------------------------------
    empty_positions = set()
    if not args.no_empty_wells and n_selected > 0:
        n_empty = max(1, math.ceil(n_selected * args.empty_pct / 100))
        n_total_wells = n_selected + n_empty
    else:
        n_total_wells = n_selected

    plate_well_pairs = generate_multiplate_wells(n_total_wells)

    if not args.no_empty_wells and n_selected > 0:
        plate_well_pairs, empty_positions = insert_empty_wells(
            plate_well_pairs,
            n_selected,
            empty_pct=args.empty_pct,
            seed=args.seed,
        )
        logger.info(
            f"Inserted {len(empty_positions)} QC empty wells "
            f"({args.empty_pct:.0f}% of {n_selected})"
        )

    n_plates = plate_well_pairs[-1][0] if plate_well_pairs else 1
    logger.info(
        f"Well assignment: {n_selected} cells + {len(empty_positions)} empty "
        f"across {n_plates} plate(s)"
    )

    # Map real-cell positions (skip empty positions) to UIDs
    uid_well_map = {}  # uid -> (plate, well)
    cell_iter = iter(selected_uids)
    for pos_idx, (plate_num, well_addr) in enumerate(plate_well_pairs):
        if pos_idx in empty_positions:
            continue
        try:
            uid = next(cell_iter)
        except StopIteration:
            break
        uid_well_map[uid] = (plate_num, well_addr)

    # -----------------------------------------------------------------------
    # Step 7: Build output detections list
    #         Each detection gets zonation metadata injected into features
    # -----------------------------------------------------------------------
    output_detections = []
    for uid in selected_uids:
        det = uid_to_det[uid]
        trans = transect_records[uid]
        plate_num, well_addr = uid_well_map[uid]

        # Deep-copy detection and inject zonation metadata
        out_det = dict(det)
        out_features = dict(det.get("features", {}))

        out_features["path_id"] = trans["path_id"]
        out_features["path_name"] = trans["path_name"]
        out_features["fractional_pos"] = trans["fractional_pos"]
        out_features["lmd_plate"] = plate_num
        out_features["lmd_well"] = well_addr

        # Attach global zonation score if available
        if uid in global_scores:
            out_features["zonation_score"] = global_scores[uid]

        # Attach marker values from transect CSV
        for key, val in trans.items():
            if key not in ("path_id", "path_name", "fractional_pos"):
                if f"transect_{key}" not in out_features:
                    out_features[f"transect_{key}"] = val

        out_det["features"] = out_features
        output_detections.append(out_det)

    # -----------------------------------------------------------------------
    # Step 8: Write outputs
    # -----------------------------------------------------------------------

    # 8a. Filtered detections JSON (compatible with run_lmd_export.py)
    filtered_json_path = out_dir / "transect_cells_for_lmd.json"
    atomic_json_dump(output_detections, filtered_json_path)
    logger.info(
        f"Wrote filtered detections: {filtered_json_path} " f"({len(output_detections):,} cells)"
    )

    # 8b. Per-cell CSV (cell_uid, path_id, fractional_pos, plate, well)
    csv_path = out_dir / "transect_cells_wells.csv"
    # Build marker column names from first transect record
    sample_trans = next(iter(transect_records.values()))
    marker_cols = [k for k in sample_trans if k not in ("path_id", "path_name", "fractional_pos")]

    csv_fieldnames = [
        "cell_uid",
        "path_id",
        "path_name",
        "fractional_pos",
        "plate",
        "well",
    ] + marker_cols
    if global_scores:
        csv_fieldnames.insert(csv_fieldnames.index("plate"), "zonation_score")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for uid in selected_uids:
            trans = transect_records[uid]
            plate_num, well_addr = uid_well_map[uid]
            row = {
                "cell_uid": uid,
                "path_id": trans["path_id"],
                "path_name": trans["path_name"],
                "fractional_pos": f"{trans['fractional_pos']:.6f}",
                "plate": plate_num,
                "well": well_addr,
            }
            for m in marker_cols:
                row[m] = trans.get(m, "")
            if global_scores:
                row["zonation_score"] = f"{global_scores.get(uid, float('nan')):.4f}"
            writer.writerow(row)
    logger.info(f"Wrote well assignments CSV: {csv_path}")

    # 8c. Summary JSON
    path_counts_selected = {}
    for uid in selected_uids:
        pid = transect_records[uid]["path_id"]
        path_counts_selected[pid] = path_counts_selected.get(pid, 0) + 1

    summary = {
        "n_cells_in_transect_csv": n_csv_total,
        "n_cells_matched_in_detections": len(uid_to_det),
        "n_cells_selected": n_selected,
        "n_plates": n_plates,
        "n_empty_wells": len(empty_positions),
        "score_threshold": args.score_threshold,
        "score_key": args.score_key,
        "path_filter": args.path_filter,
        "frac_range": [frac_lo, frac_hi],
        "max_cells": args.max_cells,
        "seed": args.seed,
        "empty_pct": args.empty_pct if not args.no_empty_wells else 0.0,
        "per_path_counts": path_counts_selected,
        "outputs": {
            "filtered_detections_json": str(filtered_json_path),
            "wells_csv": str(csv_path),
        },
    }
    summary_path = out_dir / "transect_lmd_summary.json"
    atomic_json_dump(summary, summary_path)
    logger.info(f"Wrote summary: {summary_path}")

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    logger.info("=== Selection Summary ===")
    logger.info(
        f"  Transect CSV cells:      {n_csv_total:,} (total), "
        f"{len(transect_records):,} (after pre-filters)"
    )
    logger.info(f"  Matched in detections:   {len(uid_to_det):,}")
    logger.info(f"  Selected for LMD:        {n_selected:,}")
    logger.info(f"  Plates needed:           {n_plates}")
    logger.info(f"  QC empty wells:          {len(empty_positions)}")
    logger.info("  Per-path breakdown:")
    for pid in sorted(path_counts_selected):
        logger.info(f"    {pid}: {path_counts_selected[pid]:,} cells")
    logger.info(f"  Outputs in: {out_dir}")
    logger.info("  Run LMD export with:")
    logger.info("    python run_lmd_export.py \\")
    logger.info(f"      --detections {filtered_json_path} \\")
    logger.info("      --crosses <crosses.json> \\")
    logger.info(f"      --output-dir {out_dir}/xml --export")


if __name__ == "__main__":
    main()
