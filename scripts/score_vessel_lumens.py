#!/usr/bin/env python
"""Train an RF classifier on annotated vessel lumens, score all lumens, and filter.

Replaces the inline Python in score_and_view*.sh with a proper CLI. Optionally
runs per-marker wall-cell assignment (via assign_vessel_wall_cells.assign_wall_cells)
before filtering, so the whole score→assign→filter pipeline is one command.

Outputs:
    vessel_lumens_scored.json   — all lumens with rf_score + rf_prediction
    vessel_lumens_final.json    — filtered set (RF + marker + annotation overrides)
    vessel_lumen_rf.joblib      — trained model + feature names + CV metrics

Usage:
    python scripts/score_vessel_lumens.py \\
        --lumens vessel_lumens_threshold.json \\
        --annotations vessel_lumen_annotations.json \\
        --output-dir scored/ \\
        --cells cell_detections_snr2_markers.json \\
        --markers "SMA,LYVE1" \\
        --rf-threshold 0.75 --min-marker-cells 8
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from xldvp_seg.utils.detection_utils import extract_feature_matrix  # noqa: E402
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load  # noqa: E402
from xldvp_seg.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

# Top-level numeric fields to promote into features dict for training/scoring.
_BASIC_FIELDS = [
    "area_um2",
    "equiv_diameter_um",
    "perimeter_um",
    "contrast_ratio",
    "interior_median",
    "boundary_median",
]
# NOTE: n_marker_wall is excluded from features — it correlates directly with
# the downstream filter criterion (marker count threshold), which would inflate
# CV F1 (label leakage). The RF should classify on morphology + intensity only.

# Top-level keys to exclude from auto-detection (non-features).
_EXCLUDE_PREFIXES = ("bbox_", "contour_", "uid", "discovery_scale", "refined_scale")


def _promote_basic_fields(lumens: list[dict]) -> None:
    """Copy top-level numeric fields into each lumen's features dict."""
    for l in lumens:
        feats = l.setdefault("features", {})
        for key in _BASIC_FIELDS:
            val = l.get(key)
            if isinstance(val, (int, float)) and key not in feats:
                feats[key] = val


def _collect_feature_names(lumens: list[dict], uids: set[str]) -> list[str]:
    """Auto-detect numeric feature keys from annotated lumens."""
    keys: set[str] = set()
    for l in lumens:
        if l.get("uid") not in uids:
            continue
        for k, v in l.get("features", {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not any(k.startswith(p) for p in _EXCLUDE_PREFIXES):
                    keys.add(k)
    return sorted(keys)


def _filter_final(
    lumens: list[dict],
    rf_threshold: float,
    markers: list[str],
    min_marker_cells: int,
    pos_uids: set[str],
    neg_uids: set[str],
) -> list[dict]:
    """Apply RF threshold + marker count + annotation overrides."""
    # Sample widely to avoid false negatives if first N lumens lack the field
    has_per_marker = markers and any(l.get(f"n_{markers[0]}_wall") is not None for l in lumens)

    validated_uids: set[str] = set()
    for l in lumens:
        uid = l.get("uid")
        if not uid:
            continue
        score = l.get("rf_score", 0)
        if score < rf_threshold:
            continue
        if has_per_marker:
            if any(l.get(f"n_{m}_wall", 0) >= min_marker_cells for m in markers):
                validated_uids.add(uid)
        else:
            if l.get("n_marker_wall", 0) >= min_marker_cells:
                validated_uids.add(uid)

    # Annotation overrides: union positives, subtract negatives
    final_uids = (validated_uids | pos_uids) - neg_uids

    uid_to_l = {l.get("uid"): l for l in lumens if l.get("uid")}
    return [uid_to_l[u] for u in final_uids if u in uid_to_l]


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--lumens", type=Path, required=True, help="Vessel lumens JSON (from detection)")
    p.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Annotations JSON with 'positive' and 'negative' UID lists",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--rf-threshold",
        type=float,
        default=0.75,
        help="RF score threshold for final filtering (default: 0.75)",
    )
    p.add_argument(
        "--cells",
        type=Path,
        default=None,
        help="Cell detections JSON (with marker classifications). "
        "If provided with --markers, runs per-marker wall-cell assignment before filtering.",
    )
    p.add_argument(
        "--markers",
        type=str,
        default=None,
        help="Comma-separated marker names for per-marker filtering (e.g. 'SMA,LYVE1'). "
        "Requires --cells.",
    )
    p.add_argument(
        "--min-marker-cells",
        type=int,
        default=8,
        help="Minimum marker+ cells per marker to keep vessel (default: 8)",
    )
    p.add_argument("--n-estimators", type=int, default=300, help="RF trees (default: 300)")
    p.add_argument("--max-depth", type=int, default=15, help="RF max depth (default: 15)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()

    markers = [m.strip() for m in args.markers.split(",") if m.strip()] if args.markers else []

    # --- Load data ---
    logger.info("Loading lumens: %s", args.lumens)
    lumens = fast_json_load(str(args.lumens))
    if not lumens:
        raise SystemExit(f"ERROR: No lumens loaded from {args.lumens}")
    logger.info("  %d lumens", len(lumens))

    logger.info("Loading annotations: %s", args.annotations)
    with open(args.annotations) as f:
        annots = json.load(f)
    pos_uids = set(annots.get("positive", []))
    neg_uids = set(annots.get("negative", []))
    logger.info("  %d positive, %d negative", len(pos_uids), len(neg_uids))

    # --- Promote basic fields + auto-detect features ---
    _promote_basic_fields(lumens)
    all_uids = pos_uids | neg_uids
    feature_names = _collect_feature_names(lumens, all_uids)
    logger.info("Auto-detected %d features", len(feature_names))

    # --- Build training set ---
    uid_to_idx = {l.get("uid"): i for i, l in enumerate(lumens) if l.get("uid")}
    train_indices = []
    train_labels = []
    for uid in all_uids:
        idx = uid_to_idx.get(uid)
        if idx is not None and lumens[idx].get("features"):
            train_indices.append(idx)
            train_labels.append(1 if uid in pos_uids else 0)

    if not train_indices:
        raise SystemExit("ERROR: No annotations matched lumens by UID. Check annotation file.")

    train_lumens = [lumens[i] for i in train_indices]
    y = np.array(train_labels)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    logger.info("Training set: %d samples (%d pos, %d neg)", len(y), n_pos, n_neg)

    if n_pos == 0 or n_neg == 0:
        raise SystemExit(
            f"ERROR: Need both positive and negative annotations for training. "
            f"Got {n_pos} positive, {n_neg} negative."
        )

    X_train, valid_idx = extract_feature_matrix(train_lumens, feature_names)
    # extract_feature_matrix already fills missing→0; nan_to_num not needed
    y_train = y[valid_idx] if len(valid_idx) < len(y) else y

    # --- Train + CV ---
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1")
    cv_f1 = float(scores.mean())
    cv_std = float(scores.std())
    logger.info("CV F1: %.3f +/- %.3f", cv_f1, cv_std)

    rf.fit(X_train, y_train)
    logger.info("Top 10 features:")
    for name, imp in sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])[:10]:
        logger.info("  %s: %.3f", name, imp)

    # --- Score all lumens ---
    X_all, all_valid = extract_feature_matrix(lumens, feature_names)
    probs = rf.predict_proba(X_all)[:, 1]
    for row_idx, lumen_idx in enumerate(all_valid):
        lumens[lumen_idx]["rf_score"] = round(float(probs[row_idx]), 4)
        lumens[lumen_idx]["rf_prediction"] = "positive" if probs[row_idx] >= 0.5 else "negative"
    # Lumens without features get rf_score=0
    for i, l in enumerate(lumens):
        if "rf_score" not in l:
            l["rf_score"] = 0.0
            l["rf_prediction"] = "negative"

    n_rf_pos = sum(1 for l in lumens if l["rf_prediction"] == "positive")
    logger.info("RF positive (>=0.5): %d / %d", n_rf_pos, len(lumens))

    # --- Optional: per-marker wall-cell assignment ---
    if args.cells and markers:
        import importlib.util

        _spec = importlib.util.spec_from_file_location(
            "assign_vessel_wall_cells",
            str(REPO / "scripts" / "assign_vessel_wall_cells.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        assign_wall_cells = _mod.assign_wall_cells

        logger.info("Running wall-cell assignment: markers=%s", markers)
        cells = fast_json_load(str(args.cells))
        assign_wall_cells(lumens, cells, markers)
        del cells

    # --- Save scored lumens ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / "vessel_lumens_scored.json"
    atomic_json_dump(lumens, str(scored_path))
    logger.info("Saved: %s (%d lumens)", scored_path, len(lumens))

    # --- Final filtering ---
    final = _filter_final(
        lumens, args.rf_threshold, markers, args.min_marker_cells, pos_uids, neg_uids
    )
    final_path = args.output_dir / "vessel_lumens_final.json"
    atomic_json_dump(final, str(final_path))
    logger.info(
        "Final: %d vessels (RF>=%.2f + >=%d/marker + annotation overrides)",
        len(final),
        args.rf_threshold,
        args.min_marker_cells,
    )

    # --- Summary ---
    if markers:
        for m in markers:
            n_ge = sum(1 for l in final if l.get(f"n_{m}_wall", 0) >= args.min_marker_cells)
            n_reps = sum(l.get(f"n_{m}_replicates", 0) for l in final)
            logger.info(
                "  %s: %d vessels with >=%d cells, %d replicates",
                m,
                n_ge,
                args.min_marker_cells,
                n_reps,
            )
    total_reps = sum(l.get("n_replicates_total", 0) for l in final)
    logger.info("Total replicates: %d", total_reps)

    # --- Save model ---
    model_path = args.output_dir / "vessel_lumen_rf.joblib"
    joblib.dump(
        {
            "model": rf,
            "feature_names": feature_names,
            "cv_f1_mean": cv_f1,
            "cv_f1_std": cv_std,
            "n_positive": n_pos,
            "n_negative": n_neg,
        },
        str(model_path),
    )
    logger.info("Saved model: %s", model_path)


if __name__ == "__main__":
    main()
