#!/usr/bin/env python
"""Winner-take-all aggregation of manifold-group annotations.

Reads the annotation JSON exported from the card-grid HTML (``positive`` /
``negative`` / ``unsure`` UID lists), joins against the
``manifold_exemplars.json`` to recover per-cell ``manifold_group_id``, and
classifies each Voronoi group as kept / dropped via a positive-fraction
threshold (default 60%). ``unsure`` annotations are excluded from the
denominator so a group needing only one more positive annotation isn't
penalised for an unsure pick.

Also filters ``manifold_replicates.json`` to the kept groups only, writing
``lmd_selected_positive.json`` + ``.csv`` — ready for LMD plate layout.

Typical usage::

    scripts/aggregate_group_annotations.py \\
        --annotations path/to/cell_annotations.json \\
        --manifold-dir path/to/manifold_annot_<ts>/ \\
        --threshold 0.60 \\
        --min-annotated 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_annotations(path: Path) -> dict[str, int]:
    """Load annotation JSON and flatten to ``{uid: 1|0|2}`` (pos/neg/unsure)."""
    data = fast_json_load(str(path))
    labels: dict[str, int] = {}
    for uid in data.get("positive", []) or []:
        labels[str(uid)] = 1
    for uid in data.get("negative", []) or []:
        labels[str(uid)] = 0
    for uid in data.get("unsure", []) or []:
        labels[str(uid)] = 2
    logger.info(
        "Loaded %d annotations from %s (pos=%d, neg=%d, unsure=%d)",
        len(labels),
        path.name,
        sum(1 for v in labels.values() if v == 1),
        sum(1 for v in labels.values() if v == 0),
        sum(1 for v in labels.values() if v == 2),
    )
    return labels


def _load_uid_to_group(exemplars_path: Path) -> dict[str, int]:
    """Map ``uid -> manifold_group_id`` from ``manifold_exemplars.json``."""
    dets = fast_json_load(str(exemplars_path))
    out: dict[str, int] = {}
    for det in dets:
        uid = det.get("uid")
        gid = det.get("manifold_group_id", det.get("_exemplar_group_id"))
        if uid is None or gid is None:
            continue
        out[str(uid)] = int(gid)
    logger.info("Built uid->group map for %d exemplars", len(out))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(
    labels: dict[str, int],
    uid_to_group: dict[str, int],
    *,
    threshold: float,
    min_annotated: int,
) -> dict[int, dict[str, float | int]]:
    """Per-group stats + kept/dropped decision.

    A group is "kept" iff ``n_annotated >= min_annotated`` AND
    ``positive / (positive + negative) >= threshold``. ``unsure`` votes count
    toward ``n_annotated`` but not toward the keep-decision fraction.

    Args:
        labels: per-uid label (1=pos, 0=neg, 2=unsure).
        uid_to_group: per-uid manifold_group_id.
        threshold: positive fraction required to keep a group.
        min_annotated: minimum cards annotated per group before the
            threshold decision applies (groups with fewer are auto-dropped).

    Returns:
        ``{group_id: {"n_pos", "n_neg", "n_unsure", "n_annotated",
            "positive_fraction", "kept"}}``.
    """
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"n_pos": 0, "n_neg": 0, "n_unsure": 0})
    unassigned = 0
    for uid, lbl in labels.items():
        gid = uid_to_group.get(uid)
        if gid is None:
            unassigned += 1
            continue
        if lbl == 1:
            counts[gid]["n_pos"] += 1
        elif lbl == 0:
            counts[gid]["n_neg"] += 1
        else:
            counts[gid]["n_unsure"] += 1
    if unassigned:
        logger.warning(
            "%d annotated uids had no manifold_group_id in exemplars — skipped.",
            unassigned,
        )

    out: dict[int, dict[str, float | int]] = {}
    for gid, c in counts.items():
        n_ann = c["n_pos"] + c["n_neg"] + c["n_unsure"]
        denom = c["n_pos"] + c["n_neg"]
        pos_frac = (c["n_pos"] / denom) if denom > 0 else 0.0
        kept = bool(n_ann >= min_annotated and pos_frac >= threshold)
        out[gid] = {
            "n_pos": c["n_pos"],
            "n_neg": c["n_neg"],
            "n_unsure": c["n_unsure"],
            "n_annotated": n_ann,
            "positive_fraction": round(pos_frac, 4),
            "kept": kept,
        }
    return out


# ---------------------------------------------------------------------------
# Replicate filtering
# ---------------------------------------------------------------------------


def _filter_replicates(
    replicates_path: Path,
    kept_group_ids: set[int],
) -> list[dict]:
    """Drop replicates whose ``manifold_group_id`` didn't make the cut."""
    reps = fast_json_load(str(replicates_path))
    keep = [r for r in reps if int(r.get("manifold_group_id", -1)) in kept_group_ids]
    logger.info(
        "Replicate filter: kept %d/%d replicates from %d positive groups",
        len(keep),
        len(reps),
        len(kept_group_ids),
    )
    return keep


def _write_replicates_csv(replicates: list[dict], path: Path) -> None:
    """Write the per-replicate summary CSV (matches manifold_sample.py schema)."""
    fieldnames = [
        "replicate_id",
        "manifold_group_id",
        "organ_id",
        "within_pair_replicate_idx",
        "n_cells",
        "total_area_um2",
        "mean_anchor_distance",
        "mean_xy_x_um",
        "mean_xy_y_um",
        "xy_spread_um",
        "partial",
    ]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in replicates:
            mean_xy = r.get("mean_xy_um") or [0.0, 0.0]
            w.writerow(
                {
                    "replicate_id": r.get("replicate_id"),
                    "manifold_group_id": r.get("manifold_group_id"),
                    "organ_id": r.get("organ_id"),
                    "within_pair_replicate_idx": r.get("within_pair_replicate_idx"),
                    "n_cells": r.get("n_cells"),
                    "total_area_um2": r.get("total_area_um2"),
                    "mean_anchor_distance": r.get("mean_anchor_distance"),
                    "mean_xy_x_um": mean_xy[0],
                    "mean_xy_y_um": mean_xy[1],
                    "xy_spread_um": r.get("xy_spread_um"),
                    "partial": r.get("partial"),
                }
            )
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Annotation JSON exported from the card-grid HTML "
        "(cell_annotations.json, positive/negative/unsure arrays).",
    )
    p.add_argument(
        "--manifold-dir",
        type=Path,
        required=True,
        help="Output dir from xlseg manifold-sample "
        "(contains manifold_exemplars.json + manifold_replicates.json).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Positive-fraction threshold to keep a group. Default: 0.60.",
    )
    p.add_argument(
        "--min-annotated",
        type=int,
        default=3,
        help="Skip groups with fewer than this many annotations (pos+neg+unsure). "
        "Default: 3 (out of 5 cards).",
    )
    p.add_argument(
        "--exemplars-json",
        type=Path,
        default=None,
        help="Override path to manifold_exemplars.json "
        "(default: <manifold-dir>/manifold_exemplars.json).",
    )
    p.add_argument(
        "--replicates-json",
        type=Path,
        default=None,
        help="Override path to manifold_replicates.json "
        "(default: <manifold-dir>/manifold_replicates.json).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    exemplars_json = args.exemplars_json or args.manifold_dir / "manifold_exemplars.json"
    replicates_json = args.replicates_json or args.manifold_dir / "manifold_replicates.json"

    labels = _load_annotations(args.annotations)
    uid_to_group = _load_uid_to_group(exemplars_json)
    stats = aggregate(
        labels,
        uid_to_group,
        threshold=args.threshold,
        min_annotated=args.min_annotated,
    )

    kept_ids = {gid for gid, s in stats.items() if s["kept"]}
    n_total_groups = len(stats)
    logger.info(
        "Winner-take-all: %d/%d groups kept (threshold=%.2f, min_annotated=%d)",
        len(kept_ids),
        n_total_groups,
        args.threshold,
        args.min_annotated,
    )

    # kept_groups.json — full per-group stats + kept flag.
    out_groups = args.manifold_dir / "kept_groups.json"
    payload = {
        "threshold": args.threshold,
        "min_annotated": args.min_annotated,
        "n_annotated_total": sum(1 for v in labels.values()),
        "n_groups_total": n_total_groups,
        "n_groups_kept": len(kept_ids),
        "groups": [{"manifold_group_id": gid, **s} for gid, s in sorted(stats.items())],
    }
    atomic_json_dump(payload, str(out_groups))
    logger.info("Wrote %s", out_groups)

    # Filter manifold_replicates.json -> lmd_selected_positive.{json,csv}.
    if replicates_json.exists():
        filtered = _filter_replicates(replicates_json, kept_ids)
        out_pos_json = args.manifold_dir / "lmd_selected_positive.json"
        out_pos_csv = args.manifold_dir / "lmd_selected_positive.csv"
        atomic_json_dump(filtered, str(out_pos_json))
        _write_replicates_csv(filtered, out_pos_csv)
        logger.info("Wrote %s (%d replicates) + %s", out_pos_json, len(filtered), out_pos_csv)
    else:
        logger.warning(
            "Replicates JSON not found at %s — skipping filtered-replicates step.",
            replicates_json,
        )

    logger.info("=== aggregate_group_annotations complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
