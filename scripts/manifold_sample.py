#!/usr/bin/env python
"""CLI entry for manifold-spanning LMD replicate pool building.

Runs :func:`xldvp_seg.analysis.manifold_sampling.discover_manifold_replicates`
over a detection JSON: reuses the rare-cell PCA embedding -> FPS anchors +
Voronoi (Level 1 ``manifold_group_id``) -> outlier flag -> Ward-linkage
spatial clustering per ``(manifold_group, organ)`` pair into
``target_area_um2``-sized replicates -> optional :func:`select_lmd_replicates`
cap + rank for plate allocation.

Typical usage (n45, GPU-accelerated on p.hpcl93)::

    xlseg manifold-sample \\
        --detections cell_detections_with_organs.json \\
        --output-dir manifold_pool/ \\
        --k-anchors 1000 \\
        --target-area-um2 2500 \\
        --outlier-method global_pct --outlier-threshold 98 \\
        --cap-per-group 5 --priority anchor_dist \\
        --use-gpu

Outputs (all atomically written):
    - ``manifold_replicates.json``            -- every emitted :class:`Replicate`
    - ``lmd_selected_replicates.json`` + ``.csv``  -- capped selection
    - ``manifold_state_<hash>.npz``           -- cache (written by orchestrator)
    - ``manifold_sample_stats.json``          -- summary statistics
    - ``exemplar_detections.json``            -- kept cells with ``manifold_group_id``
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

from xldvp_seg.analysis.manifold_sampling import (  # noqa: E402
    ManifoldSamplingConfig,
    discover_manifold_replicates,
    sample_group_exemplars,
    select_lmd_replicates,
)
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def _write_replicates_csv(replicates: list, path: Path) -> None:
    """Write replicate summary CSV (atomic tmp + os.replace).

    One row per :class:`Replicate`; lists (cell_uids/cell_indices) are dropped
    for CSV compactness -- the JSON sibling retains them.
    """
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
            w.writerow(
                {
                    "replicate_id": r.replicate_id,
                    "manifold_group_id": r.manifold_group_id,
                    "organ_id": r.organ_id,
                    "within_pair_replicate_idx": r.within_pair_replicate_idx,
                    "n_cells": r.n_cells,
                    "total_area_um2": r.total_area_um2,
                    "mean_anchor_distance": r.mean_anchor_distance,
                    "mean_xy_x_um": r.mean_xy_um[0],
                    "mean_xy_y_um": r.mean_xy_um[1],
                    "xy_spread_um": r.xy_spread_um,
                    "partial": r.partial,
                }
            )
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_group_weights(raw: str) -> str | dict[str, float]:
    """Accept ``equal`` / ``raw`` sentinels or a JSON dict of per-group multipliers."""
    if raw in ("equal", "raw"):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(
            f"--feature-group-weights must be 'equal', 'raw', or JSON dict; got {raw!r} ({e})"
        ) from e
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--feature-group-weights JSON must decode to a dict; got {type(parsed).__name__}"
        )
    return {str(k): float(v) for k, v in parsed.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Input detection JSON (or run-dir with cell_detections.json).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for manifold_replicates.json + lmd selection + cache.",
    )

    # ManifoldSamplingConfig
    p.add_argument("--k-anchors", type=int, default=1000, help="FPS anchor count. Default: 1000.")
    p.add_argument(
        "--target-area-um2",
        type=float,
        default=2500.0,
        help="Target cumulative cell area per Level-2 replicate. Default: 2500 um^2.",
    )
    p.add_argument(
        "--target-n-cells",
        type=int,
        default=None,
        help="Optional per-replicate cell-count target (max-of with area).",
    )
    p.add_argument(
        "--outlier-method",
        choices=["global_pct", "per_group_mad"],
        default="global_pct",
        help="Outlier flagging method. Default: global_pct.",
    )
    p.add_argument(
        "--outlier-threshold",
        type=float,
        default=98.0,
        help="Percentile (global_pct) or MAD multiplier (per_group_mad). Default: 98.",
    )
    p.add_argument(
        "--cap-per-group",
        type=int,
        default=5,
        help="Max replicates per manifold group in the LMD selection. Default: 5.",
    )
    p.add_argument(
        "--priority",
        choices=["anchor_dist", "spatial_tight", "composite"],
        default="anchor_dist",
        help="Replicate-ranking metric for the LMD selection. Default: anchor_dist.",
    )
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="Emit replicates even when total area < target (tagged partial=True).",
    )
    p.add_argument(
        "--min-spread-replicate-radii",
        type=float,
        default=4.0,
        help="Force n_rep=1 when xy extent < this many target-area radii. "
        "Default: 4.0 = two replicate diameters (so a 2-split only emits "
        "when the bounding box fits two non-overlapping target-area disks).",
    )
    p.add_argument(
        "--ward-chunk-size",
        type=int,
        default=2000,
        help="Ward linkage O(n^2) chunking threshold. Default: 2000 cells.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    p.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cupy for FPS+Voronoi and cuML for PCA if available. Default: on.",
    )
    p.add_argument(
        "--plate-budget",
        type=int,
        default=None,
        help="If set, trim LMD selection to N replicates total (after cap-per-group).",
    )

    # Feature-pipeline passthrough — thread through ManifoldSamplingConfig
    # to the orchestrator's RareCellConfig.
    p.add_argument(
        "--feature-groups",
        type=str,
        default="shape,color,sam2",
        help="Comma-separated feature groups (shape,color,sam2,channel,deep). "
        "Default: shape,color,sam2.",
    )
    p.add_argument(
        "--feature-group-weights",
        type=_parse_group_weights,
        default="equal",
        help="Per-group feature weighting: 'equal' (default, 1/sqrt(dim) so "
        "SAM2's 256 dims don't drown morphology), 'raw', or a JSON dict "
        '(e.g. \'{"shape":1.0,"sam2":0.5}\').',
    )
    p.add_argument(
        "--max-pcs",
        type=int,
        default=30,
        help="PCA component cap. Default: 30.",
    )
    p.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="PCA cumulative-variance target (capped by --max-pcs). Default: 0.95.",
    )
    p.add_argument(
        "--exclude-channels",
        type=str,
        default="",
        help="Comma-separated channel ids excluded from the feature matrix.",
    )

    # Pre-filter passthrough.
    p.add_argument("--filter-nc-min", type=float, default=0.02)
    p.add_argument("--filter-nc-max", type=float, default=0.95)
    p.add_argument("--filter-min-overlap", type=float, default=0.8)
    p.add_argument("--filter-area-min-um2", type=float, default=20.0)
    p.add_argument("--filter-area-max-um2", type=float, default=5000.0)
    # Mask-quality filters (off by default — 0.0 disables).
    p.add_argument(
        "--min-solidity",
        type=float,
        default=0.0,
        help="Drop cells with solidity < this (area / convex-hull area). "
        "Filters broken/fused masks. 0.0 disables. Try 0.75 to trim outliers.",
    )
    p.add_argument(
        "--min-max-channel-snr",
        type=float,
        default=0.0,
        help="Drop cells where max(ch_N_snr across channels) < this. "
        "Filters masks over empty tissue. 0.0 disables. Try 1.5.",
    )

    # Organ handling (Level 2 grouping).
    p.add_argument(
        "--organ-field",
        type=str,
        default="organ_id",
        help="Per-cell field holding the organ id. Default: organ_id.",
    )
    p.add_argument(
        "--organ-drop-value",
        type=int,
        default=0,
        help="Cells whose organ id equals this are excluded from Level 2. Default: 0.",
    )
    p.add_argument(
        "--organ-required",
        action="store_true",
        help="If set, abort when every cell has organ_id == organ_drop_value. "
        "Default (off): single-tier fallback (one replicate per manifold group).",
    )

    # Output toggles
    p.add_argument(
        "--no-select-lmd",
        action="store_true",
        help="Skip the final cap-and-rank LMD selection step (emit all replicates only).",
    )
    p.add_argument(
        "--no-exemplar-json",
        action="store_true",
        help="Skip writing exemplar_detections.json (kept cells with manifold_group_id).",
    )
    p.add_argument(
        "--exemplars-per-group",
        type=int,
        default=0,
        help="If > 0, also emit manifold_exemplars.json with up to N cells per "
        "Voronoi group (ranked by distance-to-anchor). Deduped by construction "
        "— each cell appears in exactly one group. 0 = skip. Default: 0.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_groups = tuple(s.strip() for s in args.feature_groups.split(",") if s.strip())
    exclude_channels = tuple(int(s.strip()) for s in args.exclude_channels.split(",") if s.strip())

    cfg = ManifoldSamplingConfig(
        k_anchors=args.k_anchors,
        target_area_um2=args.target_area_um2,
        target_n_cells=args.target_n_cells,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold,
        cap_per_group=args.cap_per_group,
        priority=args.priority,
        include_partial=args.include_partial,
        min_spread_replicate_radii=args.min_spread_replicate_radii,
        ward_chunk_size=args.ward_chunk_size,
        seed=args.seed,
        use_gpu=args.use_gpu,
        cache_dir=args.output_dir,
        # Embedding passthrough.
        feature_groups=feature_groups,
        feature_group_weights=args.feature_group_weights,
        max_pcs=args.max_pcs,
        pca_variance=args.pca_variance,
        exclude_channels=exclude_channels,
        # Pre-filter passthrough.
        nuc_filter_nc_min=args.filter_nc_min,
        nuc_filter_nc_max=args.filter_nc_max,
        nuc_filter_min_overlap=args.filter_min_overlap,
        area_filter_min_um2=args.filter_area_min_um2,
        area_filter_max_um2=args.filter_area_max_um2,
        min_solidity=args.min_solidity,
        min_max_channel_snr=args.min_max_channel_snr,
        # Organ handling.
        organ_field=args.organ_field,
        organ_drop_value=args.organ_drop_value,
        organ_required=args.organ_required,
    )

    logger.info("Loading detections: %s", args.detections)
    detections = fast_json_load(str(args.detections))
    logger.info("  %d detections loaded", len(detections))

    result = discover_manifold_replicates(detections, cfg)

    replicates = result["replicates"]
    stats = result["stats"]

    # All-replicates JSON
    out_reps = args.output_dir / "manifold_replicates.json"
    atomic_json_dump([dataclasses.asdict(r) for r in replicates], str(out_reps))
    logger.info("Wrote %d replicates: %s", len(replicates), out_reps)

    # LMD-selected subset (cap + rank)
    if not args.no_select_lmd:
        selected = select_lmd_replicates(
            replicates,
            cap_per_group=cfg.cap_per_group,
            priority=cfg.priority,
            plate_budget=args.plate_budget,
        )
        sel_json = args.output_dir / "lmd_selected_replicates.json"
        sel_csv = args.output_dir / "lmd_selected_replicates.csv"
        atomic_json_dump([dataclasses.asdict(r) for r in selected], str(sel_json))
        _write_replicates_csv(selected, sel_csv)
        logger.info(
            "Wrote LMD selection: %d replicates -> %s + %s",
            len(selected),
            sel_json.name,
            sel_csv.name,
        )
        stats["n_selected"] = len(selected)

    # Exemplar kept-detections JSON (tagged with manifold_group_id)
    if not args.no_exemplar_json:
        out_ex = args.output_dir / "exemplar_detections.json"
        atomic_json_dump(result["kept_detections"], str(out_ex))
        logger.info("Wrote kept detections: %s", out_ex)

    # Per-group card-grid exemplars — deduped by Voronoi assignment.
    if args.exemplars_per_group > 0:
        exemplars = sample_group_exemplars(
            result["kept_detections"],
            result["labels"],
            result["d_to_anchor"],
            per_group=args.exemplars_per_group,
            outlier_mask=result["outlier_mask"],
        )
        out_card = args.output_dir / "manifold_exemplars.json"
        atomic_json_dump(exemplars, str(out_card))
        logger.info(
            "Wrote card-grid exemplars: %d picks -> %s",
            len(exemplars),
            out_card,
        )
        stats["n_card_exemplars"] = len(exemplars)

    # Summary stats
    out_stats = args.output_dir / "manifold_sample_stats.json"
    atomic_json_dump(stats, str(out_stats))
    logger.info("Wrote stats: %s", out_stats)

    logger.info(
        "=== manifold-sample complete === "
        "kept %d cells | %d anchors | %d replicates | %d outliers (%.1f%%) | "
        "PCA %d dims (%.1f%% var)",
        stats["n_kept_cells"],
        stats["n_anchors"],
        stats["n_replicates"],
        stats["n_outliers"],
        100 * stats["outlier_fraction"],
        stats["pca_n_components"],
        100 * stats["pca_variance"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
