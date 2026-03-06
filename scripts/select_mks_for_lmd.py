#!/usr/bin/env python3
"""Select MK detections for LMD proteomics replicates.

Thin wrapper around segmentation.lmd.selection with MK-specific config:
bone grouping, mk_score field, sex/treatment parsing, FGC3 exclusion.

Usage:
    python3 scripts/select_mks_for_lmd.py --score-threshold 0.80
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.lmd.selection import select_cells_for_lmd

DATASET_DIR = Path("/Volumes/pool-mann-edwin/bm_lmd_feb2026/mk_clf084_dataset")
BONE_JSON = DATASET_DIR / "all_mks_clf075_with_bone.json"
EXCLUDE_SLIDES = set()  # FGC3 excluded from ANOVA but collected for proteomics


def parse_slide_metadata(slide: str) -> dict:
    """Extract sex and treatment from slide name like 2025_11_18_FGC1."""
    tag = slide.split("_")[-1]
    sex = tag[0]
    treatment = re.match(r"[FM]([A-Z]+)", tag).group(1)[:2]
    return {"sex": sex, "treatment": treatment}


def serpentine_wells_384():
    """Generate 384-well positions across quadrants with alternating serpentine.

    Rules (fixed for all LMD experiments):
    - No outer wells ever (row A, row P, col 1, col 24)
    - Quadrant order: B2 → B3 → C3 → C2
      (named by first usable well; B3 = even rows/odd cols,
       C3 = odd rows/odd cols, C2 = odd rows/even cols)
    - First quadrant (B2): top-left → bottom-right serpentine
    - Second quadrant (B3): bottom-right → top-left serpentine
    - Alternates direction each quadrant
    - 7 rows × 11 cols = 77 usable wells per quadrant, 308 total
    """
    # (row_parity, col_parity): 0=odd(A-row/1-col), 1=even(B-row/2-col)
    # B2(even,even) → B3(even,odd) → C3(odd,odd) → C2(odd,even)
    quadrant_order = [(1, 1), (1, 0), (0, 0), (0, 1)]

    for q_idx, (row_par, col_par) in enumerate(quadrant_order):
        all_rows = [string.ascii_uppercase[i] for i in range(row_par, 16, 2)]
        all_cols = list(range(col_par + 1, 25, 2))  # +1: cols are 1-based

        rows = [r for r in all_rows if r not in ("A", "P")]
        cols = [c for c in all_cols if c not in (1, 24)]

        # Odd quadrants: reverse rows (bottom-right → top-left)
        if q_idx % 2 == 1:
            rows = list(reversed(rows))

        for i, row in enumerate(rows):
            if q_idx % 2 == 0:
                c = cols if i % 2 == 0 else list(reversed(cols))
            else:
                c = list(reversed(cols)) if i % 2 == 0 else cols
            for col in c:
                yield f"{row}{col}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-threshold", type=float, default=0.80)
    parser.add_argument("--target-area", type=float, default=10000.0,
                        help="Target total area per replicate in um^2")
    parser.add_argument("--max-replicates", type=int, default=4)
    parser.add_argument("--min-replicate-fraction", type=float, default=0.5,
                        help="Minimum fraction of target area to keep a partial replicate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bone-json", type=Path, default=BONE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR / "lmd_selections")
    args = parser.parse_args()

    # Load detections with bone assignments
    with open(args.bone_json) as f:
        all_cells = json.load(f)
    print(f"Loaded {len(all_cells)} detections from {args.bone_json.name}")

    # MK-specific callables
    groups_result, summary = select_cells_for_lmd(
        detections=all_cells,
        group_key_fn=lambda d: (d["slide"], d["bone"]),
        score_fn=lambda d: d["mk_score"],
        score_threshold=args.score_threshold,
        target_area=args.target_area,
        max_replicates=args.max_replicates,
        min_replicate_fraction=args.min_replicate_fraction,
        seed=args.seed,
        exclude_fn=lambda d: (d.get("bone") is None
                              or d["slide"] in EXCLUDE_SLIDES),
    )

    print(f"After score>={args.score_threshold} + exclusions + dedup: "
          f"{summary['total_cells_filtered']} cells")

    # Build per-group entries with MK-specific metadata
    group_entries = []
    for (slide, bone), gdata in groups_result.items():
        meta = parse_slide_metadata(slide)
        rep_entries = []
        for i, rep in enumerate(gdata["replicates"], 1):
            rep_id = f"{slide.split('_')[-1]}_{bone}_rep{i}"
            rep_entries.append({
                "replicate_id": rep_id,
                "n_cells": len(rep["uids"]),
                "total_area_um2": rep["total_area_um2"],
                "uids": rep["uids"],
            })

        group_entries.append({
            "slide": slide,
            "bone": bone,
            "sex": meta["sex"],
            "treatment": meta["treatment"],
            "replicates": rep_entries,
            "available_cells": gdata["available_cells"],
            "unused_cells": gdata["unused_cells"],
        })

    # Randomize order: slides shuffled, bones within each slide shuffled,
    # replicates within each bone shuffled — but all groups from same slide
    # stay contiguous (one slide loaded at a time on the LMD).
    rng = np.random.default_rng(args.seed)

    # Group entries by slide
    by_slide = {}
    for g in group_entries:
        by_slide.setdefault(g["slide"], []).append(g)

    # Shuffle slide order, then shuffle bones within each slide,
    # then shuffle replicates within each bone
    slide_keys = list(by_slide.keys())
    rng.shuffle(slide_keys)

    results = []
    for slide in slide_keys:
        slide_groups = by_slide[slide]
        rng.shuffle(slide_groups)
        for g in slide_groups:
            rng.shuffle(g["replicates"])
            results.append(g)

    # Assign wells — serpentine across quadrants (B2→B1→A2→A1, 308 max)
    # Insert 10% blank wells randomly distributed for QC
    total_reps = sum(len(g["replicates"]) for g in results)
    n_blanks = int(np.ceil(total_reps * 0.10))
    total_wells = total_reps + n_blanks
    if total_wells > 308:
        print(f"WARNING: {total_wells} wells needed ({total_reps} samples + "
              f"{n_blanks} blanks) exceeds 308-well capacity")

    wells = list(zip(range(total_wells), serpentine_wells_384()))
    blank_positions = set(rng.choice(total_wells, size=n_blanks, replace=False))

    blanks_list = []
    rep_iter = iter([(g, rep) for g in results for rep in g["replicates"]])
    for pos, well in wells:
        if pos in blank_positions:
            blanks_list.append(well)
        else:
            _, rep = next(rep_iter)
            rep["well"] = well

    output = {
        "params": {
            "score_threshold": args.score_threshold,
            "target_area_um2": args.target_area,
            "cells_equivalent": int(args.target_area / 500),
            "max_replicates": args.max_replicates,
            "min_replicate_fraction": args.min_replicate_fraction,
            "seed": args.seed,
        },
        "groups": results,
        "blanks": blanks_list,
        "summary": {
            "total_replicates": summary["total_replicates"],
            "total_cells_selected": summary["total_cells_selected"],
            "total_wells": total_wells,
            "blank_wells": n_blanks,
            "groups_with_max_reps": summary["groups_with_max_reps"],
            "groups_with_fewer_reps": summary["groups_with_fewer_reps"],
            "groups_with_no_reps": summary["groups_with_no_reps"],
        },
    }

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "mk_lmd_selections.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote selections to {out_path}")

    # Build flat list of all wells in serpentine (collection) order
    well_order = {w: i for i, (_, w) in enumerate(wells)}
    all_wells = []
    for g in results:
        for rep in g["replicates"]:
            all_wells.append({"well": rep["well"], "type": "sample",
                              "group": g, "rep": rep})
    for well in blanks_list:
        all_wells.append({"well": well, "type": "blank"})

    all_wells.sort(key=lambda w: well_order[w["well"]])

    # Print summary table in plate order
    print(f"\n{'Well':<6} {'Slide':<22} {'Bone':<9} {'Sex':>3} {'Tx':>3} {'RepID':<20} {'Cells':>5} {'Area':>8}")
    print("-" * 85)
    current_slide = None
    for entry in all_wells:
        if entry["type"] == "blank":
            print(f"{entry['well']:<6} {'--- BLANK ---'}")
        else:
            g, rep = entry["group"], entry["rep"]
            if g["slide"] != current_slide:
                if current_slide is not None:
                    print()
                current_slide = g["slide"]
            print(f"{rep['well']:<6} {g['slide']:<22} {g['bone']:<9} {g['sex']:>3} {g['treatment']:>3} "
                  f"{rep['replicate_id']:<20} {rep['n_cells']:>5} {rep['total_area_um2']:>8.0f}")

    print(f"\nTotal: {total_reps} samples + {n_blanks} blanks = {total_wells} wells")
    print(f"  {summary['groups_with_max_reps']} groups with {args.max_replicates} reps, "
          f"{summary['groups_with_fewer_reps']} with fewer, "
          f"{summary['groups_with_no_reps']} with none")


if __name__ == "__main__":
    main()
