#!/usr/bin/env python3
"""RBC vascularization analysis for MK hindlimb unloading project.

Trains an RF classifier on annotated RBC clusters from unfiltered SAM detections,
scores all detections, and performs spatial analysis of MK-to-RBC relationships
across the 2×2×2 experimental design (sex × treatment × bone).

Workflow:
  1. --train   : Match annotations to features, train RF, save model + metrics
  2. --score   : Score all 230K detections (one slide at a time), save results
  3. --analyze : Spatial analysis — MK-RBC distances, densities, co-localization
  4. --report  : Generate summary figures and ANOVA table

Usage:
    PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/rbc_vascularization_analysis.py \
        --train --score --analyze \
        --annotations rbc_annotations.json \
        --unfiltered-dir per_slide_unfiltered/ \
        --mk-detections all_mks_curated_deduped.json \
        --tissue-areas tissue_areas_by_bone.json \
        --output-dir rbc_analysis/
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from segmentation.utils.logging import get_logger, setup_logging
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load

log = get_logger(__name__)

# Morph features available in unfiltered detections
MORPH_FEATURES = [
    "area", "aspect_ratio", "blue_mean", "blue_std", "circularity",
    "dark_fraction", "dark_region_fraction", "eccentricity", "elongation",
    "equiv_diameter", "extent", "gray_mean", "gray_std", "green_mean",
    "green_std", "hue_mean", "intensity_variance", "nuclear_complexity",
    "perimeter", "red_mean", "red_std", "relative_brightness",
    "saturation_mean", "solidity", "value_mean",
]

# Extra top-level features useful for classification
EXTRA_FEATURES = ["mk_score", "sam2_iou", "sam2_stability", "area_um2"]


def parse_slide_metadata(slide_name):
    """Extract sex, treatment, replicate from slide name.

    Format: 2025_11_18_{Sex}{Trt}{Rep} e.g. 2025_11_18_FGC1
    """
    tag = slide_name.split("_")[-1]  # e.g. FGC1
    sex = tag[0]         # F or M
    trt = tag[1:3]       # GC or HU
    rep = tag[3:]         # 1-4
    return {"sex": sex, "treatment": trt, "replicate": rep}


def extract_feature_vector(det):
    """Extract feature vector from a detection dict.

    Returns numpy array of shape (n_features,) or None if features missing.
    """
    feats = det.get("features", {})
    if not feats:
        return None

    vec = []
    for fname in MORPH_FEATURES:
        v = feats.get(fname)
        vec.append(float(v) if v is not None else 0.0)

    for fname in EXTRA_FEATURES:
        v = det.get(fname)
        vec.append(float(v) if v is not None else 0.0)

    return np.array(vec, dtype=np.float32)


def get_feature_names():
    return MORPH_FEATURES + EXTRA_FEATURES


# ---------------------------------------------------------------------------
# Step 1: Train
# ---------------------------------------------------------------------------

def train_classifier(annotations_path, unfiltered_dir, output_dir):
    """Train RF classifier from annotations + unfiltered detection features."""
    log.info("=== TRAIN ===")

    # Load annotations
    with open(annotations_path) as f:
        annot = json.load(f)

    rbc_uids = set(annot.get("rbc", []))
    other_uids = set(annot.get("other", []))
    mk_uids = set(annot.get("mk", []))
    all_annotated = rbc_uids | other_uids | mk_uids

    log.info(f"Annotations: {len(rbc_uids)} RBC, {len(other_uids)} other, "
             f"{len(mk_uids)} MK = {len(all_annotated)} total")

    # Find annotated detections across unfiltered files
    X_list = []
    y_list = []
    uid_list = []
    found = set()

    unfiltered_dir = Path(unfiltered_dir)
    for jf in sorted(unfiltered_dir.glob("*_full_unfiltered.json")):
        slide = jf.stem.replace("_full_unfiltered", "")
        log.info(f"  Scanning {slide}...")

        detections = fast_json_load(jf)

        for det in detections:
            uid = det["uid"]
            if uid not in all_annotated or uid in found:
                continue

            vec = extract_feature_vector(det)
            if vec is None:
                continue

            if uid in rbc_uids:
                label = 1
            else:
                label = 0  # other + mk both = "not RBC"

            X_list.append(vec)
            y_list.append(label)
            uid_list.append(uid)
            found.add(uid)

        del detections

        if len(found) >= len(all_annotated):
            log.info(f"  All {len(all_annotated)} annotations found")
            break

    missing = all_annotated - found
    if missing:
        log.warning(f"  {len(missing)} annotated UIDs not found in unfiltered data")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    feature_names = get_feature_names()

    log.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    log.info(f"  Class 1 (RBC): {np.sum(y == 1)}, Class 0 (other): {np.sum(y == 0)}")

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    cv_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[val_idx])
        y_prob = rf.predict_proba(X[val_idx])[:, 1]

        f1 = f1_score(y[val_idx], y_pred)
        try:
            auc = roc_auc_score(y[val_idx], y_prob)
        except ValueError:
            auc = 0.0
        cv_scores.append(f1)
        cv_aucs.append(auc)
        log.info(f"  Fold {fold + 1}: F1={f1:.3f}, AUC={auc:.3f}")

    log.info(f"  CV F1: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")
    log.info(f"  CV AUC: {np.mean(cv_aucs):.3f} +/- {np.std(cv_aucs):.3f}")

    # Train final model on all data
    rf_final = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_final.fit(X, y)

    # Feature importances
    importances = rf_final.feature_importances_
    imp_order = np.argsort(importances)[::-1]
    log.info("  Top 10 features:")
    for i in range(min(10, len(imp_order))):
        idx = imp_order[i]
        log.info(f"    {feature_names[idx]}: {importances[idx]:.4f}")

    # Save model
    model_path = output_dir / "rbc_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(rf_final, f)
    log.info(f"  Model saved: {model_path}")

    # Save metrics
    metrics = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_rbc": int(np.sum(y == 1)),
        "n_other": int(np.sum(y == 0)),
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
        "cv_auc_mean": float(np.mean(cv_aucs)),
        "cv_auc_std": float(np.std(cv_aucs)),
        "feature_names": feature_names,
        "feature_importances": {
            feature_names[i]: float(importances[i])
            for i in imp_order[:15]
        },
    }
    atomic_json_dump(metrics, str(output_dir / "rbc_classifier_metrics.json"))

    return rf_final


# ---------------------------------------------------------------------------
# Step 2: Score
# ---------------------------------------------------------------------------

def score_detections(model_path, unfiltered_dir, output_dir, score_threshold=0.5):
    """Score all detections one slide at a time. Save scored results."""
    log.info("=== SCORE ===")

    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    unfiltered_dir = Path(unfiltered_dir)
    all_slide_stats = {}

    for jf in sorted(unfiltered_dir.glob("*_full_unfiltered.json")):
        slide = jf.stem.replace("_full_unfiltered", "")
        log.info(f"  Scoring {slide}...")

        detections = fast_json_load(jf)

        # Extract features for all detections
        X_list = []
        valid_indices = []
        for i, det in enumerate(detections):
            vec = extract_feature_vector(det)
            if vec is not None:
                X_list.append(vec)
                valid_indices.append(i)

        if not X_list:
            log.warning(f"    No valid features in {slide}")
            del detections
            continue

        X = np.array(X_list, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        probs = rf.predict_proba(X)[:, 1]

        # Build scored results — only keep detections above threshold
        # Save lightweight version (no crop/mask/contour)
        scored = []
        n_above = 0
        for j, idx in enumerate(valid_indices):
            det = detections[idx]
            rbc_score = float(probs[j])
            if rbc_score >= score_threshold:
                n_above += 1
                scored.append({
                    "uid": det["uid"],
                    "slide": slide,
                    "center_x": det["center_x"],
                    "center_y": det["center_y"],
                    "area_um2": det.get("area_um2", 0),
                    "mk_score": det.get("mk_score", 0),
                    "rbc_score": rbc_score,
                })

        all_slide_stats[slide] = {
            "total": len(detections),
            "scored": len(valid_indices),
            "rbc_above_threshold": n_above,
        }

        log.info(f"    {len(detections)} total, {n_above} RBC (>={score_threshold})")

        # Save per-slide scored results
        slide_path = output_dir / f"{slide}_rbc_scored.json"
        atomic_json_dump(scored, str(slide_path))

        del detections, X_list, X, probs, scored

    # Save summary
    atomic_json_dump(all_slide_stats, str(output_dir / "rbc_scoring_summary.json"))
    log.info(f"  Scoring complete for {len(all_slide_stats)} slides")

    return all_slide_stats


# ---------------------------------------------------------------------------
# Step 3: Spatial analysis
# ---------------------------------------------------------------------------

def load_bone_regions(bone_regions_path):
    """Load bone region polygons. Returns dict: (slide, bone) -> Path object."""
    from matplotlib.path import Path as MplPath

    with open(bone_regions_path) as f:
        data = json.load(f)

    regions = {}
    slides_data = data.get("slides", data)
    for slide_name, slide_info in slides_data.items():
        if not isinstance(slide_info, dict):
            continue
        for bone in ["femur", "humerus"]:
            bone_data = slide_info.get(bone)
            if not bone_data:
                continue
            vertices = bone_data.get("vertices_px", bone_data)
            if isinstance(vertices, list) and len(vertices) >= 3:
                regions[(slide_name, bone)] = MplPath(vertices)

    return regions


def spatial_analysis(output_dir, mk_detections_path, tissue_areas_path=None,
                     bone_regions_path=None):
    """Compute MK-to-RBC spatial metrics per slide/bone."""
    log.info("=== SPATIAL ANALYSIS ===")

    # Load MK detections (curated, deduped)
    mk_dets = fast_json_load(mk_detections_path)
    log.info(f"  Loaded {len(mk_dets)} curated MKs")

    # Load bone regions for RBC assignment
    bone_regions = {}
    if bone_regions_path and Path(bone_regions_path).exists():
        bone_regions = load_bone_regions(bone_regions_path)
        log.info(f"  Loaded {len(bone_regions)} bone region polygons")

    # Load tissue areas: (slide, bone) -> tissue_area_mm2
    tissue_areas = {}
    pixel_size_um = 0.22  # default
    if tissue_areas_path and Path(tissue_areas_path).exists():
        with open(tissue_areas_path) as f:
            ta_data = json.load(f)
        for entry in ta_data.get("results", []):
            slide = entry["slide"]
            if entry.get("pixel_size_um"):
                pixel_size_um = entry["pixel_size_um"]
            for bone_name, bone_info in entry.get("bones", {}).items():
                ta = bone_info.get("tissue_area_mm2", 0)
                tissue_areas[(slide, bone_name)] = ta
        log.info(f"  Loaded tissue areas for {len(tissue_areas)} regions, "
                 f"pixel_size={pixel_size_um} um/px")

    # Index MKs by slide
    mks_by_slide = {}
    for mk in mk_dets:
        s = mk.get("slide", "")
        mks_by_slide.setdefault(s, []).append(mk)

    # Process each slide's RBC scored results
    output_dir = Path(output_dir)
    results = []

    for rbc_file in sorted(output_dir.glob("*_rbc_scored.json")):
        slide = rbc_file.stem.replace("_rbc_scored", "")
        meta = parse_slide_metadata(slide)

        with open(rbc_file) as f:
            rbc_dets = json.load(f)

        if not rbc_dets:
            log.info(f"  {slide}: 0 RBCs — skipping")
            continue

        # Assign each RBC to a bone using vectorized polygon containment
        rbcs_by_bone = {"femur": [], "humerus": [], "unknown": []}
        all_centroids = np.array(
            [[r["center_x"], r["center_y"]] for r in rbc_dets], dtype=np.float64
        )
        # bone_label[i] = first bone that contains rbc_dets[i], or "unknown"
        bone_labels = ["unknown"] * len(rbc_dets)
        for bone in ["femur", "humerus"]:
            key = (slide, bone)
            if key not in bone_regions:
                continue
            inside = bone_regions[key].contains_points(all_centroids)
            for i, flag in enumerate(inside):
                if flag and bone_labels[i] == "unknown":
                    bone_labels[i] = bone
        for rbc, bone in zip(rbc_dets, bone_labels):
            rbcs_by_bone[bone].append(rbc)

        n_assigned = len(rbcs_by_bone["femur"]) + len(rbcs_by_bone["humerus"])
        log.info(f"  {slide}: {len(rbc_dets)} RBCs total, "
                 f"{n_assigned} assigned to bones "
                 f"(F={len(rbcs_by_bone['femur'])}, "
                 f"H={len(rbcs_by_bone['humerus'])})")

        # Get MKs for this slide
        slide_mks = mks_by_slide.get(slide, [])
        if not slide_mks:
            log.warning(f"  {slide}: RBCs but 0 MKs — skipping")
            continue

        # Separate MKs by bone
        mks_by_bone = {}
        for mk in slide_mks:
            bone = mk.get("bone", "unknown")
            mks_by_bone.setdefault(bone, []).append(mk)

        for bone in ["femur", "humerus"]:
            bone_mks = mks_by_bone.get(bone, [])
            bone_rbcs = rbcs_by_bone.get(bone, [])

            if not bone_mks or not bone_rbcs:
                log.info(f"    {bone}: {len(bone_mks)} MKs, "
                         f"{len(bone_rbcs)} RBCs — skipping")
                continue

            mk_xy = np.array([[m["center_x"], m["center_y"]]
                              for m in bone_mks])
            rbc_xy = np.array([[r["center_x"], r["center_y"]]
                               for r in bone_rbcs])
            rbc_tree = cKDTree(rbc_xy)

            # MK-to-nearest-RBC distance (in pixels -> um)
            dists_px, _ = rbc_tree.query(mk_xy, k=1)
            dists_um = dists_px * pixel_size_um

            # RBC count in MK neighborhood (200um radius)
            radius_px = 200.0 / pixel_size_um
            rbc_near_mk = rbc_tree.query_ball_point(mk_xy, r=radius_px)
            rbc_counts_near_mk = [len(x) for x in rbc_near_mk]

            # Co-localization: fraction of MKs with RBC within threshold
            coloc_50 = float(np.mean(dists_um <= 50))
            coloc_100 = float(np.mean(dists_um <= 100))
            coloc_200 = float(np.mean(dists_um <= 200))

            # Tissue area for density normalization
            tissue_area_mm2 = tissue_areas.get((slide, bone), 0)

            row = {
                "slide": slide,
                "bone": bone,
                **meta,
                "n_mk": len(bone_mks),
                "n_rbc": len(bone_rbcs),
                "tissue_area_mm2": tissue_area_mm2,
                "rbc_density_per_mm2": (
                    len(bone_rbcs) / tissue_area_mm2
                    if tissue_area_mm2 > 0 else 0
                ),
                "mk_density_per_mm2": (
                    len(bone_mks) / tissue_area_mm2
                    if tissue_area_mm2 > 0 else 0
                ),
                # MK-to-RBC distance stats (um)
                "mk_rbc_dist_mean_um": float(np.mean(dists_um)),
                "mk_rbc_dist_median_um": float(np.median(dists_um)),
                "mk_rbc_dist_std_um": float(np.std(dists_um)),
                "mk_rbc_dist_p25_um": float(np.percentile(dists_um, 25)),
                "mk_rbc_dist_p75_um": float(np.percentile(dists_um, 75)),
                # RBC count in MK neighborhood (200um)
                "rbc_near_mk_200um_mean": float(np.mean(rbc_counts_near_mk)),
                "rbc_near_mk_200um_median": float(np.median(rbc_counts_near_mk)),
                # Co-localization
                "mk_rbc_coloc_50um": coloc_50,
                "mk_rbc_coloc_100um": coloc_100,
                "mk_rbc_coloc_200um": coloc_200,
                # MK area (reference)
                "mk_area_mean_um2": float(np.mean(
                    [m["area_um2"] for m in bone_mks])),
            }
            results.append(row)

            log.info(
                f"    {bone}: {len(bone_mks)} MKs, {len(bone_rbcs)} RBCs, "
                f"dist={np.median(dists_um):.0f}um (med), "
                f"coloc_100={coloc_100:.2f}"
            )

        del rbc_dets

    # Save results
    atomic_json_dump(results, str(output_dir / "rbc_spatial_results.json"))

    # Also save as CSV for easy analysis
    if results:
        import csv
        csv_path = output_dir / "rbc_spatial_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        log.info(f"  CSV saved: {csv_path}")

    log.info(f"  Spatial analysis complete: {len(results)} slide/bone combinations")
    return results


# ---------------------------------------------------------------------------
# Step 4: ANOVA
# ---------------------------------------------------------------------------

def run_anova(output_dir):
    """Run 2×2×2 ANOVA (sex × treatment × bone) on spatial metrics."""
    log.info("=== ANOVA ===")

    results_path = output_dir / "rbc_spatial_results.json"
    if not results_path.exists():
        log.error("No spatial results found — run --analyze first")
        return

    with open(results_path) as f:
        results = json.load(f)

    try:
        import pandas as pd
        from scipy import stats
    except ImportError:
        log.error("pandas/scipy required for ANOVA")
        return

    df = pd.DataFrame(results)

    if len(df) == 0:
        log.error("No results to analyze")
        return

    log.info(f"  {len(df)} observations (slide × bone)")
    log.info(f"  Groups: {df.groupby(['sex', 'treatment', 'bone']).size().to_dict()}")

    # Metrics to test
    metrics = [
        "mk_rbc_dist_mean_um",
        "mk_rbc_dist_median_um",
        "rbc_near_mk_200um_mean",
        "rbc_density_per_mm2",
        "n_rbc",
    ]

    anova_results = []

    for metric in metrics:
        if metric not in df.columns:
            log.warning(f"  Metric '{metric}' not found in columns: {sorted(df.columns.tolist())}")
            continue
        if df[metric].isna().all():
            log.warning(f"  Metric '{metric}' is all NaN — skipping")
            continue

        log.info(f"\n  --- {metric} ---")

        # Group means
        for (sex, trt, bone), grp in df.groupby(["sex", "treatment", "bone"]):
            vals = grp[metric].values
            log.info(f"    {sex}{trt} {bone}: "
                     f"mean={np.mean(vals):.1f}, n={len(vals)}")

        # 3-way ANOVA via statsmodels if available
        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            formula = f"{metric} ~ C(sex) * C(treatment) * C(bone)"
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            for effect, row in anova_table.iterrows():
                if effect == "Residual":
                    continue
                p = row.get("PR(>F)", 1.0)
                f_val = row.get("F", 0.0)
                ss = row.get("sum_sq", 0.0)
                total_ss = anova_table["sum_sq"].sum()
                eta2 = ss / total_ss if total_ss > 0 else 0

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                log.info(f"    {effect}: F={f_val:.2f}, p={p:.4f}, "
                         f"η²={eta2:.3f} {sig}")

                anova_results.append({
                    "metric": metric,
                    "effect": str(effect),
                    "F": float(f_val) if not np.isnan(f_val) else 0,
                    "p": float(p) if not np.isnan(p) else 1,
                    "eta_squared": float(eta2),
                    "significant": p < 0.05,
                })

        except ImportError:
            log.warning("  statsmodels not available — skipping full ANOVA")
            # Fallback: simple group comparisons
            for factor in ["sex", "treatment", "bone"]:
                groups = [grp[metric].values for _, grp in df.groupby(factor)]
                if len(groups) == 2:
                    t, p = stats.ttest_ind(groups[0], groups[1])
                    log.info(f"    {factor}: t={t:.2f}, p={p:.4f}")

    # Save ANOVA table
    if anova_results:
        atomic_json_dump(anova_results, str(output_dir / "rbc_anova_results.json"))

        try:
            import csv
            csv_path = output_dir / "rbc_anova_results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=anova_results[0].keys())
                writer.writeheader()
                writer.writerows(anova_results)
            log.info(f"  ANOVA table saved: {csv_path}")
        except Exception:
            pass

    log.info("  ANOVA complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RBC vascularization analysis for MK HU project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", action="store_true",
                        help="Train RF classifier from annotations")
    parser.add_argument("--score", action="store_true",
                        help="Score all detections with trained classifier")
    parser.add_argument("--analyze", action="store_true",
                        help="Run spatial analysis (MK-RBC distances)")
    parser.add_argument("--report", action="store_true",
                        help="Run ANOVA on spatial results")

    parser.add_argument("--annotations", type=str,
                        help="Path to rbc_annotations.json")
    parser.add_argument("--unfiltered-dir", type=str,
                        help="Dir with *_full_unfiltered.json")
    parser.add_argument("--mk-detections", type=str,
                        help="Path to curated MK detections JSON")
    parser.add_argument("--tissue-areas", type=str, default=None,
                        help="Path to tissue_areas_by_bone.json")
    parser.add_argument("--bone-regions", type=str, default=None,
                        help="Path to bone_regions.json for RBC bone assignment")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for all results")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="RBC score threshold (default: 0.5)")

    args = parser.parse_args()
    setup_logging(level="INFO")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not any([args.train, args.score, args.analyze, args.report]):
        log.error("Specify at least one of: --train, --score, --analyze, --report")
        sys.exit(1)

    if args.train:
        if not args.annotations or not args.unfiltered_dir:
            log.error("--train requires --annotations and --unfiltered-dir")
            sys.exit(1)
        train_classifier(
            Path(args.annotations),
            Path(args.unfiltered_dir),
            output_dir,
        )

    if args.score:
        model_path = output_dir / "rbc_classifier.pkl"
        if not model_path.exists():
            log.error(f"No model at {model_path} — run --train first")
            sys.exit(1)
        if not args.unfiltered_dir:
            log.error("--score requires --unfiltered-dir")
            sys.exit(1)
        score_detections(
            model_path,
            Path(args.unfiltered_dir),
            output_dir,
            score_threshold=args.score_threshold,
        )

    if args.analyze:
        if not args.mk_detections:
            log.error("--analyze requires --mk-detections")
            sys.exit(1)
        spatial_analysis(
            output_dir,
            Path(args.mk_detections),
            tissue_areas_path=args.tissue_areas,
            bone_regions_path=args.bone_regions,
        )

    if args.report:
        run_anova(output_dir)


if __name__ == "__main__":
    main()
