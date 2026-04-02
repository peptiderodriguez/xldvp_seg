"""Generic cell selection for LMD proteomics replicates.

Groups cells by a user-defined key, then greedily fills area-matched
replicates (e.g. target ~10,000 um^2 each). Each replicate stays within
a single group — never mixed.

Works with any cell type — caller provides callables for grouping, scoring,
and area extraction.

Usage:
    from xldvp_seg.lmd.selection import select_cells_for_lmd

    groups, summary = select_cells_for_lmd(
        detections=cells,
        group_key_fn=lambda d: (d["slide"], d["bone"]),
        score_fn=lambda d: d["mk_score"],
        score_threshold=0.80,
    )
"""

from collections.abc import Callable

import numpy as np


def build_replicates(cells, target_area, max_reps, min_fraction, rng, area_fn=None):
    """Greedily assign shuffled cells into area-matched replicates.

    Adds cells randomly. For the last cell in each replicate (the one that
    would cross the target), picks the best-fitting cell from the remaining
    pool to minimize overshoot.

    Args:
        cells: list of detection dicts
        target_area: target total area per replicate
        max_reps: maximum number of replicates to build
        min_fraction: minimum fraction of target_area to keep a partial replicate
        rng: numpy random generator
        area_fn: callable returning area from a detection (default: d["area_um2"])

    Returns:
        list of {"uids": [...], "total_area_um2": float}
    """
    if area_fn is None:
        area_fn = lambda d: d["area_um2"]

    cell_areas = np.array([area_fn(c) for c in cells])
    indices = list(range(len(cells)))
    rng.shuffle(indices)

    available = set(indices)
    replicates = []

    while len(replicates) < max_reps and available:
        current_uids = []
        current_area = 0.0

        # Greedy random fill
        for idx in [i for i in indices if i in available]:
            candidate_total = current_area + cell_areas[idx]

            if candidate_total >= target_area:
                # This cell would cross the target — instead, find the
                # cell that lands closest to target (over or under)
                gap = target_area - current_area
                remaining = list(available)
                best_idx = min(remaining, key=lambda i: abs(cell_areas[i] - gap))
                available.discard(best_idx)
                current_uids.append(cells[best_idx]["uid"])
                current_area += cell_areas[best_idx]
                # Accept if within ±5% of target
                if current_area >= target_area * 0.95:
                    replicates.append({"uids": current_uids, "total_area_um2": current_area})
                break

            available.discard(idx)
            current_uids.append(cells[idx]["uid"])
            current_area += cell_areas[idx]
        else:
            # Exhausted available cells without reaching target
            if current_uids and current_area >= target_area * min_fraction:
                replicates.append({"uids": current_uids, "total_area_um2": current_area})
            break

    return replicates


def select_cells_for_lmd(
    detections: list,
    group_key_fn: Callable,
    score_fn: Callable,
    score_threshold: float = 0.80,
    target_area: float = 10000.0,
    max_replicates: int = 4,
    min_replicate_fraction: float = 0.5,
    seed: int = 42,
    area_fn: Callable | None = None,
    exclude_fn: Callable | None = None,
):
    """Select cells for LMD proteomics, grouped into area-matched replicates.

    Args:
        detections: list of detection dicts (any cell type)
        group_key_fn: callable, returns grouping key (tuple or str) from a detection
        score_fn: callable, returns numeric score from a detection
        score_threshold: minimum score to include a detection
        target_area: target total area per replicate in um^2
        max_replicates: max replicates per group
        min_replicate_fraction: min fraction of target_area to keep partial replicate
        seed: random seed for reproducibility
        area_fn: callable returning area from a detection (default: d["area_um2"])
        exclude_fn: optional callable, return True to exclude a detection

    Returns:
        (groups_dict, summary_dict) where:
        - groups_dict: {group_key: {"cells": [...], "replicates": [...]}}
        - summary_dict: {"total_replicates", "total_cells_selected", ...}
    """
    if area_fn is None:
        area_fn = lambda d: d["area_um2"]

    # Filter by score and exclusions, deduplicate by UID
    seen_uids = set()
    cells = []
    for d in detections:
        if score_fn(d) < score_threshold:
            continue
        if exclude_fn and exclude_fn(d):
            continue
        if d["uid"] in seen_uids:
            continue
        cells.append(d)
        seen_uids.add(d["uid"])

    # Group by key
    grouped = {}
    for c in cells:
        key = group_key_fn(c)
        grouped.setdefault(key, []).append(c)

    rng = np.random.default_rng(seed)
    groups_result = {}
    all_selected_uids = set()

    for key in sorted(grouped.keys()):
        group_cells = grouped[key]
        reps = build_replicates(
            group_cells, target_area, max_replicates, min_replicate_fraction, rng, area_fn
        )

        for rep in reps:
            rep["total_area_um2"] = round(rep["total_area_um2"], 1)
            all_selected_uids.update(rep["uids"])

        n_selected = sum(len(r["uids"]) for r in reps)
        groups_result[key] = {
            "cells": group_cells,
            "replicates": reps,
            "available_cells": len(group_cells),
            "unused_cells": len(group_cells) - n_selected,
        }

    # Verify no uid duplication across replicates
    uid_list = []
    for g in groups_result.values():
        for r in g["replicates"]:
            uid_list.extend(r["uids"])
    assert len(uid_list) == len(set(uid_list)), "Duplicate UIDs across replicates!"

    total_reps = sum(len(g["replicates"]) for g in groups_result.values())
    groups_with_max = sum(
        1 for g in groups_result.values() if len(g["replicates"]) == max_replicates
    )
    groups_with_fewer = sum(
        1 for g in groups_result.values() if 0 < len(g["replicates"]) < max_replicates
    )
    groups_with_none = sum(1 for g in groups_result.values() if len(g["replicates"]) == 0)

    summary = {
        "total_cells_filtered": len(cells),
        "total_replicates": total_reps,
        "total_cells_selected": len(all_selected_uids),
        "groups_with_max_reps": groups_with_max,
        "groups_with_fewer_reps": groups_with_fewer,
        "groups_with_no_reps": groups_with_none,
    }

    return groups_result, summary
