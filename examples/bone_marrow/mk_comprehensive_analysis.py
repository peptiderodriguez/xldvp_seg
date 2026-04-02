#!/usr/bin/env python3
"""Comprehensive MK analysis across all dimensions, combinations, and interactions.

Tests every feature across:
- 2-way: Sex, Treatment, Bone (Mann-Whitney U)
- Stratified 2-way: e.g. Treatment|femur, Sex|GC, etc.
- 3-way stratified: e.g. Treatment|F_femur, Sex|HU_humerus
- Multi-group (Kruskal-Wallis): Sex×Treatment (4 groups), Sex×Bone (4), Treatment×Bone (4), Sex×Treatment×Bone (8)
- Interactions: does treatment effect differ by sex? by bone? Does sex×treatment interaction differ by bone?
- Replicate effects: Kruskal across replicates 1-4
- Humerus-corrected ratios: femur/humerus ratio per slide, tested across groups

All at SLIDE level (median per slide×bone) to avoid pseudoreplication.

Outputs:
- mk_comprehensive_stats.csv (all pairwise results)
- mk_multigroup_stats.csv (Kruskal-Wallis + interaction results)
- Figures
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

from xldvp_seg.utils.json_utils import fast_json_load

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata

# ── Config ──────────────────────────────────────────────────────────────
DETECTIONS_FULL = Path("/path/to/data/bm_lmd_feb2026/mk_clf084_dataset/all_mks_clf075_light.json")
DETECTIONS_BONE = Path(
    "/path/to/data/bm_lmd_feb2026/mk_clf084_dataset/all_mks_clf075_with_bone.json"
)
TISSUE_AREAS = Path("/path/to/data/bm_lmd_feb2026/mk_clf084_dataset/tissue_areas_by_bone.json")
OUTPUT_DIR = Path("/path/to/data/bm_lmd_feb2026/mk_clf084_dataset")

# Features to skip (embeddings are high-dim, not interpretable individually)
SKIP_PREFIXES = ("sam2_", "resnet_", "dinov2_")


# ── Parse slide name ────────────────────────────────────────────────────
def parse_slide(name):
    """Parse '2025_11_18_FGC1' → {sex, treatment, replicate}."""
    short = name.replace("2025_11_18_", "")
    sex = "F" if short[0] == "F" else "M"
    treatment = "GC" if "GC" in short else "HU"
    replicate = short[-1]
    return {"sex": sex, "treatment": treatment, "replicate": replicate, "short": short}


def cohens_d(g1, g2):
    """Cohen's d effect size."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * np.std(g1, ddof=1) ** 2 + (n2 - 1) * np.std(g2, ddof=1) ** 2) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def aligned_rank_transform_interaction(df, feat, factor_a, factor_b):
    """Aligned rank transform for nonparametric interaction test.

    Strips main effects, ranks residuals, then runs Kruskal on aligned ranks.
    Returns p-value for the interaction term.
    """
    vals = df[[feat, factor_a, factor_b]].dropna()
    if len(vals) < 6:
        return np.nan

    y = vals[feat].values.astype(float)
    a = vals[factor_a].values
    b = vals[factor_b].values

    # Compute cell means and marginal means
    grand_mean = np.mean(y)
    a_levels = np.unique(a)
    b_levels = np.unique(b)

    a_means = {lv: np.mean(y[a == lv]) for lv in a_levels}
    b_means = {lv: np.mean(y[b == lv]) for lv in b_levels}

    # Aligned residuals: strip main effects, keep interaction
    aligned = np.zeros_like(y, dtype=float)
    for i in range(len(y)):
        aligned[i] = y[i] - a_means[a[i]] - b_means[b[i]] + grand_mean

    # Rank the aligned residuals
    ranks = rankdata(aligned)

    # Kruskal-Wallis on ranks grouped by A×B cells
    groups = {}
    for i in range(len(ranks)):
        key = (a[i], b[i])
        groups.setdefault(key, []).append(ranks[i])

    group_list = [np.array(v) for v in groups.values() if len(v) > 0]
    if len(group_list) < 2:
        return np.nan

    try:
        _, p = kruskal(*group_list)
        return p
    except Exception:
        return np.nan


def main():
    # ── Load data ───────────────────────────────────────────────────────
    print("Loading data...")
    full_data = fast_json_load(DETECTIONS_FULL)
    bone_data = fast_json_load(DETECTIONS_BONE)
    tissue_data = fast_json_load(TISSUE_AREAS)

    # Build UID → bone mapping
    uid_to_bone = {}
    for d in bone_data:
        uid_to_bone[d["uid"]] = d["bone"]

    # Build slide → bone → tissue info
    tissue_info = {}
    for r in tissue_data["results"]:
        slide = r["slide"]
        tissue_info[slide] = r["bones"]

    # ── Build per-cell DataFrame ────────────────────────────────────────
    print("Building per-cell DataFrame...")
    rows = []
    for det in full_data:
        uid = det["uid"]
        slide = det["slide"]
        bone = uid_to_bone.get(uid, "")
        if not bone or bone == "unknown":
            continue

        meta = parse_slide(slide)
        feats = det.get("features", {})

        row = {
            "uid": uid,
            "slide": slide,
            "short": meta["short"],
            "sex": meta["sex"],
            "treatment": meta["treatment"],
            "replicate": meta["replicate"],
            "bone": bone,
            "area_um2": det.get("area_um2", feats.get("area", np.nan)),
            "mk_score": det.get("mk_score", np.nan),
        }

        # Add interpretable features (skip embedding dims)
        for k, v in feats.items():
            if any(k.startswith(p) for p in SKIP_PREFIXES):
                continue
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                continue

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} cells, {len(df.columns)} columns")
    print(f"  Slides: {sorted(df['slide'].unique())}")
    print(f"  Bones: {dict(df['bone'].value_counts())}")

    # Add tissue-normalized density per slide×bone
    density_rows = []
    for slide in df["slide"].unique():
        for bone in ["femur", "humerus"]:
            mask = (df["slide"] == slide) & (df["bone"] == bone)
            n_cells = mask.sum()
            t_info = tissue_info.get(slide, {}).get(bone, {})
            tissue_area = t_info.get("tissue_area_mm2", np.nan)
            density = n_cells / tissue_area if tissue_area and tissue_area > 0 else np.nan
            tissue_frac = t_info.get("tissue_fraction", np.nan)
            density_rows.append(
                {
                    "slide": slide,
                    "bone": bone,
                    "n_cells": n_cells,
                    "tissue_area_mm2": tissue_area,
                    "density_per_mm2": density,
                    "tissue_fraction": tissue_frac,
                }
            )
    density_df = pd.DataFrame(density_rows)

    # ── Identify feature columns ────────────────────────────────────────
    meta_cols = {"uid", "slide", "short", "sex", "treatment", "replicate", "bone", "mk_score"}
    feature_cols = [
        c for c in df.columns if c not in meta_cols and df[c].dtype in ("float64", "int64")
    ]
    print(f"  {len(feature_cols)} features to test: {feature_cols}")

    # ── Aggregate to slide×bone level ───────────────────────────────────
    print("\nAggregating to slide×bone level...")
    agg_df = (
        df.groupby(["slide", "short", "sex", "treatment", "replicate", "bone"])[feature_cols]
        .median()
        .reset_index()
    )

    # Merge density info
    agg_df = agg_df.merge(
        density_df[
            ["slide", "bone", "n_cells", "tissue_area_mm2", "density_per_mm2", "tissue_fraction"]
        ],
        on=["slide", "bone"],
        how="left",
    )
    # Add density and tissue_fraction to feature list
    extra_features = ["n_cells", "tissue_area_mm2", "density_per_mm2", "tissue_fraction"]
    all_features = feature_cols + extra_features

    print(f"  {len(agg_df)} slide×bone observations")
    print(f"  {len(all_features)} total features (incl. density, tissue)")

    # ── Exclude slides ────────────────────────────────────────────────────
    exclude_slides = {"2025_11_18_FGC3"}
    n_before = len(agg_df)
    agg_df = agg_df[~agg_df["slide"].isin(exclude_slides)].reset_index(drop=True)
    print(f"\nExcluded slides: {exclude_slides} ({n_before} → {len(agg_df)} observations)")

    # Also exclude from per-cell df (for density recalc)
    density_df = density_df[~density_df["slide"].isin(exclude_slides)].reset_index(drop=True)

    # ── IQR outlier filtering (per feature) ─────────────────────────────
    # Preserve specific values that are biologically real, not artifacts
    keep_rules = {
        # tissue_fraction outliers fixed with otsu*0.8 — no longer need protection
        ("n_cells", "FHU2", "femur"),
        ("n_cells", "FHU4", "femur"),
    }

    print("\nIQR outlier filtering (1.5×IQR per feature)...")
    outlier_counts = defaultdict(int)
    for feat in all_features:
        vals = agg_df[feat].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        feat_outliers = (agg_df[feat] < lo) | (agg_df[feat] > hi)
        feat_outliers = feat_outliers.fillna(False)

        # Check keep rules — don't remove protected values
        for idx in agg_df[feat_outliers].index:
            row = agg_df.loc[idx]
            if (feat, row["short"], row["bone"]) in keep_rules:
                feat_outliers.at[idx] = False

        n_out = feat_outliers.sum()
        if n_out > 0:
            outlier_counts[feat] = n_out
            for idx in agg_df[feat_outliers].index:
                row = agg_df.loc[idx]
                print(
                    f"    {row['short']:8s} {row['bone']:10s} {feat:25s} = {vals.get(idx, '?'):.4g}  (outside [{lo:.4g}, {hi:.4g}])"
                )
            agg_df.loc[feat_outliers, feat] = np.nan

    n_total_outliers = sum(outlier_counts.values())
    print(f"  Removed {n_total_outliers} outlier values across {len(outlier_counts)} features")
    if outlier_counts:
        for feat, n in sorted(outlier_counts.items(), key=lambda x: -x[1]):
            print(f"    {feat:25s}: {n} outliers")

    # Also filter ratio_df the same way after it's built (done below)

    # Add composite grouping columns
    agg_df["sex_trt"] = agg_df["sex"] + "_" + agg_df["treatment"]
    agg_df["sex_bone"] = agg_df["sex"] + "_" + agg_df["bone"]
    agg_df["trt_bone"] = agg_df["treatment"] + "_" + agg_df["bone"]
    agg_df["sex_trt_bone"] = agg_df["sex"] + "_" + agg_df["treatment"] + "_" + agg_df["bone"]

    # ── Define all PAIRWISE comparisons ─────────────────────────────────
    comparisons = []

    # --- 1. Single-dimension comparisons (whole dataset) ---
    comparisons.append(("Sex", "sex", {"F": "F", "M": "M"}, None))
    comparisons.append(("Treatment", "treatment", {"GC": "GC", "HU": "HU"}, None))
    comparisons.append(("Bone", "bone", {"femur": "femur", "humerus": "humerus"}, None))

    # --- 2. Within-bone comparisons ---
    for bone in ["femur", "humerus"]:
        comparisons.append((f"Sex|{bone}", "sex", {"F": "F", "M": "M"}, {"bone": bone}))
        comparisons.append(
            (f"Treatment|{bone}", "treatment", {"GC": "GC", "HU": "HU"}, {"bone": bone})
        )

    # --- 3. Within-sex comparisons ---
    for sex in ["F", "M"]:
        comparisons.append(
            (f"Treatment|{sex}", "treatment", {"GC": "GC", "HU": "HU"}, {"sex": sex})
        )
        comparisons.append(
            (f"Bone|{sex}", "bone", {"femur": "femur", "humerus": "humerus"}, {"sex": sex})
        )

    # --- 4. Within-treatment comparisons ---
    for trt in ["GC", "HU"]:
        comparisons.append((f"Sex|{trt}", "sex", {"F": "F", "M": "M"}, {"treatment": trt}))
        comparisons.append(
            (f"Bone|{trt}", "bone", {"femur": "femur", "humerus": "humerus"}, {"treatment": trt})
        )

    # --- 5. Three-way: within sex×bone ---
    for sex in ["F", "M"]:
        for bone in ["femur", "humerus"]:
            comparisons.append(
                (
                    f"Treatment|{sex}_{bone}",
                    "treatment",
                    {"GC": "GC", "HU": "HU"},
                    {"sex": sex, "bone": bone},
                )
            )

    # --- 6. Three-way: within sex×treatment ---
    for sex in ["F", "M"]:
        for trt in ["GC", "HU"]:
            comparisons.append(
                (
                    f"Bone|{sex}_{trt}",
                    "bone",
                    {"femur": "femur", "humerus": "humerus"},
                    {"sex": sex, "treatment": trt},
                )
            )

    # --- 7. Three-way: within treatment×bone ---
    for trt in ["GC", "HU"]:
        for bone in ["femur", "humerus"]:
            comparisons.append(
                (f"Sex|{trt}_{bone}", "sex", {"F": "F", "M": "M"}, {"treatment": trt, "bone": bone})
            )

    # --- 8. All pairwise among the 4 Sex×Treatment groups ---
    sex_trt_groups = ["F_GC", "F_HU", "M_GC", "M_HU"]
    for g1, g2 in combinations(sex_trt_groups, 2):
        comparisons.append((f"{g1}_vs_{g2}", "sex_trt", {g1: g1, g2: g2}, None))

    # --- 9. All pairwise among the 8 Sex×Treatment×Bone groups ---
    stb_groups = [
        f"{s}_{t}_{b}" for s in ["F", "M"] for t in ["GC", "HU"] for b in ["femur", "humerus"]
    ]
    for g1, g2 in combinations(stb_groups, 2):
        comparisons.append((f"{g1}_vs_{g2}", "sex_trt_bone", {g1: g1, g2: g2}, None))

    # --- 10. Humerus-corrected ratio analysis ---
    ratio_rows = []
    for slide in agg_df["slide"].unique():
        fem = agg_df[(agg_df["slide"] == slide) & (agg_df["bone"] == "femur")]
        hum = agg_df[(agg_df["slide"] == slide) & (agg_df["bone"] == "humerus")]
        if len(fem) == 0 or len(hum) == 0:
            continue
        fem_row = fem.iloc[0]
        hum_row = hum.iloc[0]
        ratio_row = {
            "slide": slide,
            "short": fem_row["short"],
            "sex": fem_row["sex"],
            "treatment": fem_row["treatment"],
            "replicate": fem_row["replicate"],
            "sex_trt": fem_row["sex_trt"],
        }
        for feat in all_features:
            f_val = fem_row.get(feat, np.nan)
            h_val = hum_row.get(feat, np.nan)
            if pd.notna(f_val) and pd.notna(h_val) and h_val != 0:
                ratio_row[f"ratio_{feat}"] = f_val / h_val
            else:
                ratio_row[f"ratio_{feat}"] = np.nan
        ratio_rows.append(ratio_row)
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df = ratio_df[~ratio_df["slide"].isin(exclude_slides)].reset_index(drop=True)
    ratio_features = [c for c in ratio_df.columns if c.startswith("ratio_")]

    # IQR filter ratio features
    print("\nIQR outlier filtering on ratio features...")
    ratio_outlier_counts = 0
    for feat in ratio_features:
        vals = ratio_df[feat].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        feat_outliers = (ratio_df[feat] < lo) | (ratio_df[feat] > hi)
        feat_outliers = feat_outliers.fillna(False)
        n_out = feat_outliers.sum()
        if n_out > 0:
            ratio_outlier_counts += n_out
            ratio_df.loc[feat_outliers, feat] = np.nan
    print(f"  Removed {ratio_outlier_counts} ratio outlier values")

    # Ratio comparisons
    comparisons.append(("Ratio_Sex", "sex", {"F": "F", "M": "M"}, None))
    comparisons.append(("Ratio_Treatment", "treatment", {"GC": "GC", "HU": "HU"}, None))
    for sex in ["F", "M"]:
        comparisons.append(
            (f"Ratio_Treatment|{sex}", "treatment", {"GC": "GC", "HU": "HU"}, {"sex": sex})
        )
    # Ratio across 4 groups pairwise
    for g1, g2 in combinations(sex_trt_groups, 2):
        comparisons.append((f"Ratio_{g1}_vs_{g2}", "sex_trt", {g1: g1, g2: g2}, None))

    print(f"\n{len(comparisons)} pairwise comparison groups")

    # ── Run all pairwise tests ──────────────────────────────────────────
    print("\nRunning pairwise tests...")
    results = []

    for comp_name, group_col, group_vals, filters in comparisons:
        is_ratio = comp_name.startswith("Ratio_")

        if is_ratio:
            working_df = ratio_df.copy()
            test_features = ratio_features
        else:
            working_df = agg_df.copy()
            test_features = all_features

        # Apply filters
        if filters:
            for fk, fv in filters.items():
                working_df = working_df[working_df[fk] == fv]

        g1_label, g2_label = list(group_vals.keys())
        g1_df = working_df[working_df[group_col] == group_vals[g1_label]]
        g2_df = working_df[working_df[group_col] == group_vals[g2_label]]

        n1, n2 = len(g1_df), len(g2_df)
        if n1 < 2 or n2 < 2:
            continue

        for feat in test_features:
            v1 = g1_df[feat].dropna().values
            v2 = g2_df[feat].dropna().values

            if len(v1) < 2 or len(v2) < 2:
                continue

            try:
                stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                d = cohens_d(v1, v2)
            except Exception:
                continue

            feat_display = feat.replace("ratio_", "") if is_ratio else feat
            results.append(
                {
                    "comparison": comp_name,
                    "feature": feat_display,
                    "test_type": "pairwise",
                    "is_ratio": is_ratio,
                    "group_var": group_col,
                    "g1_label": g1_label,
                    "g2_label": g2_label,
                    "g1_median": float(np.median(v1)),
                    "g2_median": float(np.median(v2)),
                    "g1_n": len(v1),
                    "g2_n": len(v2),
                    "U_statistic": stat,
                    "p_value": p,
                    "cohens_d": d,
                    "abs_cohens_d": abs(d),
                    "direction": (
                        f"{g1_label}>{g2_label}"
                        if np.median(v1) > np.median(v2)
                        else f"{g2_label}>{g1_label}"
                    ),
                }
            )

    # ── Multi-group tests (Kruskal-Wallis) ──────────────────────────────
    print("Running multi-group Kruskal-Wallis tests...")
    multigroup_results = []

    # Define multi-group comparisons
    kw_tests = [
        # (name, group_col, filter)
        ("KW_SexTrt(4grp)", "sex_trt", None),
        ("KW_SexBone(4grp)", "sex_bone", None),
        ("KW_TrtBone(4grp)", "trt_bone", None),
        ("KW_SexTrtBone(8grp)", "sex_trt_bone", None),
        ("KW_Replicate", "replicate", None),
    ]
    # Within-bone multi-group
    for bone in ["femur", "humerus"]:
        kw_tests.append((f"KW_SexTrt|{bone}", "sex_trt", {"bone": bone}))
        kw_tests.append((f"KW_Replicate|{bone}", "replicate", {"bone": bone}))
    # Within-sex multi-group
    for sex in ["F", "M"]:
        kw_tests.append((f"KW_TrtBone|{sex}", "trt_bone", {"sex": sex}))
        kw_tests.append((f"KW_Replicate|{sex}", "replicate", {"sex": sex}))
    # Within-treatment multi-group
    for trt in ["GC", "HU"]:
        kw_tests.append((f"KW_SexBone|{trt}", "sex_bone", {"treatment": trt}))
        kw_tests.append((f"KW_Replicate|{trt}", "replicate", {"treatment": trt}))

    for kw_name, group_col, filters in kw_tests:
        working_df = agg_df.copy()
        if filters:
            for fk, fv in filters.items():
                working_df = working_df[working_df[fk] == fv]

        groups = working_df[group_col].unique()
        if len(groups) < 2:
            continue

        for feat in all_features:
            group_values = []
            group_labels = []
            for g in sorted(groups):
                vals = working_df[working_df[group_col] == g][feat].dropna().values
                if len(vals) >= 1:
                    group_values.append(vals)
                    group_labels.append(g)

            if len(group_values) < 2 or any(len(v) < 1 for v in group_values):
                continue

            try:
                stat, p = kruskal(*group_values)
            except Exception:
                continue

            # Compute eta-squared (effect size for Kruskal-Wallis)
            n_total = sum(len(v) for v in group_values)
            eta_sq = (
                (stat - len(group_values) + 1) / (n_total - len(group_values))
                if n_total > len(group_values)
                else 0
            )

            multigroup_results.append(
                {
                    "comparison": kw_name,
                    "feature": feat,
                    "test_type": "kruskal_wallis",
                    "n_groups": len(group_values),
                    "groups": ", ".join(str(g) for g in group_labels),
                    "group_sizes": ", ".join(str(len(v)) for v in group_values),
                    "group_medians": ", ".join(f"{np.median(v):.4g}" for v in group_values),
                    "H_statistic": stat,
                    "p_value": p,
                    "eta_squared": eta_sq,
                }
            )

    # ── Interaction tests (Aligned Rank Transform) ──────────────────────
    print("Running interaction tests (Aligned Rank Transform)...")
    interaction_tests = [
        ("Interaction_Sex*Treatment", "sex", "treatment"),
        ("Interaction_Sex*Bone", "sex", "bone"),
        ("Interaction_Treatment*Bone", "treatment", "bone"),
    ]
    # Within-level interactions
    for bone in ["femur", "humerus"]:
        interaction_tests.append((f"Interaction_Sex*Trt|{bone}", "sex", "treatment"))
    for sex in ["F", "M"]:
        interaction_tests.append((f"Interaction_Trt*Bone|{sex}", "treatment", "bone"))
    for trt in ["GC", "HU"]:
        interaction_tests.append((f"Interaction_Sex*Bone|{trt}", "sex", "bone"))

    for int_name, factor_a, factor_b in interaction_tests:
        # Parse optional filter from name
        working_df = agg_df.copy()
        if "|" in int_name:
            filter_part = int_name.split("|")[1]
            # Determine which column the filter applies to
            for col in ["bone", "sex", "treatment"]:
                if filter_part in working_df[col].values:
                    working_df = working_df[working_df[col] == filter_part]
                    break

        for feat in all_features:
            p_int = aligned_rank_transform_interaction(working_df, feat, factor_a, factor_b)
            if np.isnan(p_int):
                continue

            multigroup_results.append(
                {
                    "comparison": int_name,
                    "feature": feat,
                    "test_type": "interaction_ART",
                    "n_groups": len(working_df[factor_a].unique())
                    * len(working_df[factor_b].unique()),
                    "groups": f"{factor_a} x {factor_b}",
                    "group_sizes": str(len(working_df)),
                    "group_medians": "",
                    "H_statistic": np.nan,
                    "p_value": p_int,
                    "eta_squared": np.nan,
                }
            )

    # Also do interaction tests on ratio data
    for int_name_base, factor_a, factor_b in [("Interaction_Sex*Trt_RATIO", "sex", "treatment")]:
        for feat in ratio_features:
            p_int = aligned_rank_transform_interaction(ratio_df, feat, factor_a, factor_b)
            if np.isnan(p_int):
                continue
            multigroup_results.append(
                {
                    "comparison": int_name_base,
                    "feature": feat.replace("ratio_", ""),
                    "test_type": "interaction_ART_ratio",
                    "n_groups": 4,
                    "groups": f"{factor_a} x {factor_b} (ratio)",
                    "group_sizes": str(len(ratio_df)),
                    "group_medians": "",
                    "H_statistic": np.nan,
                    "p_value": p_int,
                    "eta_squared": np.nan,
                }
            )

    mg_df = pd.DataFrame(multigroup_results)
    mg_df = mg_df.sort_values("p_value")

    # BH correction for multi-group tests
    if len(mg_df) > 0:
        n_mg = len(mg_df)
        mg_df["rank"] = range(1, n_mg + 1)
        mg_df["p_bh"] = mg_df["p_value"] * n_mg / mg_df["rank"]
        mg_df["p_bh"] = mg_df["p_bh"].clip(upper=1.0)
        mg_df["p_bh"] = mg_df["p_bh"][::-1].cummin()[::-1]
        mg_df["significant_raw"] = mg_df["p_value"] < 0.05
        mg_df["significant_bh"] = mg_df["p_bh"] < 0.05

    # ── BH correction for pairwise tests ────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value")

    n_tests = len(results_df)
    results_df["rank"] = range(1, n_tests + 1)
    results_df["p_bh"] = results_df["p_value"] * n_tests / results_df["rank"]
    results_df["p_bh"] = results_df["p_bh"].clip(upper=1.0)
    results_df["p_bh"] = results_df["p_bh"][::-1].cummin()[::-1]

    results_df["significant_raw"] = results_df["p_value"] < 0.05
    results_df["significant_bh"] = results_df["p_bh"] < 0.05

    # ── Save CSVs ───────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "mk_comprehensive_stats.csv"
    save_cols = [
        "comparison",
        "feature",
        "test_type",
        "is_ratio",
        "group_var",
        "g1_label",
        "g2_label",
        "g1_median",
        "g2_median",
        "g1_n",
        "g2_n",
        "U_statistic",
        "p_value",
        "p_bh",
        "cohens_d",
        "abs_cohens_d",
        "direction",
        "significant_raw",
        "significant_bh",
    ]
    results_df[save_cols].to_csv(csv_path, index=False)
    print(f"\nSaved {len(results_df)} pairwise test results to {csv_path}")

    mg_csv = OUTPUT_DIR / "mk_multigroup_stats.csv"
    mg_df.to_csv(mg_csv, index=False)
    print(f"Saved {len(mg_df)} multi-group/interaction test results to {mg_csv}")

    # ── Print summary ───────────────────────────────────────────────────
    sig_raw = results_df[results_df["significant_raw"]]
    sig_bh = results_df[results_df["significant_bh"]]
    mg_sig = mg_df[mg_df["significant_raw"]] if len(mg_df) > 0 else pd.DataFrame()
    mg_sig_bh = mg_df[mg_df["significant_bh"]] if len(mg_df) > 0 else pd.DataFrame()

    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print("\nPAIRWISE TESTS:")
    print(f"  Total tests: {n_tests}")
    print(f"  Significant (p<0.05 uncorrected): {len(sig_raw)}")
    print(f"  Significant (BH-corrected q<0.05): {len(sig_bh)}")
    print("\nMULTI-GROUP / INTERACTION TESTS:")
    print(f"  Total tests: {len(mg_df)}")
    print(f"  Significant (p<0.05 uncorrected): {len(mg_sig)}")
    print(f"  Significant (BH-corrected q<0.05): {len(mg_sig_bh)}")

    # ── Top pairwise results ────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("TOP 30 PAIRWISE RESULTS (sorted by p-value)")
    print(f"{'─'*80}")
    for i, row in results_df.head(30).iterrows():
        star = (
            "***"
            if row["p_bh"] < 0.001
            else (
                "**"
                if row["p_bh"] < 0.01
                else "*" if row["p_bh"] < 0.05 else "." if row["p_value"] < 0.05 else ""
            )
        )
        ratio_tag = " [RATIO]" if row["is_ratio"] else ""
        print(
            f"  p={row['p_value']:.4f} (q={row['p_bh']:.4f}) d={row['cohens_d']:+.2f} {star:4s} "
            f"{row['comparison']:30s} {row['feature']:25s} "
            f"{row['direction']:20s}{ratio_tag}"
        )

    # ── Top multi-group results ─────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("TOP 30 MULTI-GROUP / INTERACTION RESULTS")
    print(f"{'─'*80}")
    for i, row in mg_df.head(30).iterrows():
        star = (
            "***"
            if row["p_bh"] < 0.001
            else (
                "**"
                if row["p_bh"] < 0.01
                else "*" if row["p_bh"] < 0.05 else "." if row["p_value"] < 0.05 else ""
            )
        )
        print(
            f"  p={row['p_value']:.4f} (q={row['p_bh']:.4f}) {star:4s} "
            f"{row['comparison']:35s} {row['feature']:25s} "
            f"[{row['test_type']}]"
        )

    # ── Significant multi-group by comparison ───────────────────────────
    if len(mg_sig) > 0:
        print(f"\n{'─'*80}")
        print("SIGNIFICANT MULTI-GROUP / INTERACTION (p<0.05)")
        print(f"{'─'*80}")
        for comp in sorted(mg_sig["comparison"].unique()):
            comp_res = mg_sig[mg_sig["comparison"] == comp].sort_values("p_value")
            print(f"\n  {comp} ({len(comp_res)} significant):")
            for _, row in comp_res.iterrows():
                bh_tag = " [BH]" if row["significant_bh"] else ""
                if row["test_type"] == "kruskal_wallis":
                    print(
                        f"    p={row['p_value']:.4f} H={row['H_statistic']:.1f} eta²={row['eta_squared']:.3f} "
                        f"{row['feature']:25s} medians=[{row['group_medians']}]{bh_tag}"
                    )
                else:
                    print(
                        f"    p={row['p_value']:.4f} {row['feature']:25s} ({row['groups']}){bh_tag}"
                    )

    # ── Significant pairwise by comparison group ────────────────────────
    print(f"\n{'─'*80}")
    print("SIGNIFICANT PAIRWISE (uncorrected p<0.05) BY COMPARISON GROUP")
    print(f"{'─'*80}")
    for comp in sorted(sig_raw["comparison"].unique()):
        comp_results = sig_raw[sig_raw["comparison"] == comp].sort_values("p_value")
        print(f"\n  {comp} ({len(comp_results)} significant features):")
        for _, row in comp_results.iterrows():
            bh_tag = " [BH]" if row["significant_bh"] else ""
            ratio_tag = " [RATIO]" if row["is_ratio"] else ""
            print(
                f"    p={row['p_value']:.4f} d={row['cohens_d']:+.2f} {row['feature']:25s} {row['direction']}{bh_tag}{ratio_tag}"
            )

    # ── Generate summary figures ────────────────────────────────────────
    print("\nGenerating figures...")

    # Figure 1: Volcano-style plot — pairwise + multi-group
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "Comprehensive MK Analysis: All Tests × All Features", fontsize=16, fontweight="bold"
    )

    # Panel 1: Pairwise volcano
    direct = results_df[~results_df["is_ratio"]]
    ax = axes[0, 0]
    ax.scatter(direct["cohens_d"], -np.log10(direct["p_value"]), alpha=0.15, s=10, c="gray")
    sig_mask = direct["p_value"] < 0.05
    ax.scatter(
        direct.loc[sig_mask, "cohens_d"],
        -np.log10(direct.loc[sig_mask, "p_value"]),
        alpha=0.6,
        s=20,
        c="red",
        label=f"p<0.05 (n={sig_mask.sum()})",
    )
    bh_mask = direct["p_bh"] < 0.05
    ax.scatter(
        direct.loc[bh_mask, "cohens_d"],
        -np.log10(direct.loc[bh_mask, "p_value"]),
        alpha=0.9,
        s=50,
        c="darkred",
        marker="*",
        label=f"BH q<0.05 (n={bh_mask.sum()})",
    )
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"Pairwise Tests (n={len(direct)})")
    ax.legend(fontsize=9)

    # Panel 2: Multi-group / interaction -log10(p) strip
    ax = axes[0, 1]
    if len(mg_df) > 0:
        kw_rows = mg_df[mg_df["test_type"] == "kruskal_wallis"]
        int_rows = mg_df[mg_df["test_type"].str.startswith("interaction")]

        if len(kw_rows) > 0:
            ax.scatter(
                np.random.uniform(-0.3, 0.3, len(kw_rows)),
                -np.log10(kw_rows["p_value"]),
                alpha=0.3,
                s=15,
                c="steelblue",
                label=f"Kruskal-Wallis (n={len(kw_rows)})",
            )
            kw_sig = kw_rows[kw_rows["p_value"] < 0.05]
            ax.scatter(
                np.random.uniform(-0.3, 0.3, len(kw_sig)),
                -np.log10(kw_sig["p_value"]),
                alpha=0.8,
                s=30,
                c="blue",
                label=f"KW p<0.05 (n={len(kw_sig)})",
            )

        if len(int_rows) > 0:
            ax.scatter(
                np.random.uniform(0.7, 1.3, len(int_rows)),
                -np.log10(int_rows["p_value"]),
                alpha=0.3,
                s=15,
                c="orange",
                label=f"Interaction ART (n={len(int_rows)})",
            )
            int_sig = int_rows[int_rows["p_value"] < 0.05]
            ax.scatter(
                np.random.uniform(0.7, 1.3, len(int_sig)),
                -np.log10(int_sig["p_value"]),
                alpha=0.8,
                s=30,
                c="darkorange",
                label=f"Int p<0.05 (n={len(int_sig)})",
            )

        ax.axhline(-np.log10(0.05), color="red", linestyle="--", alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Kruskal-Wallis", "Interaction (ART)"])
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"Multi-group & Interaction Tests (n={len(mg_df)})")
    ax.legend(fontsize=8)

    # Panel 3: Heatmap — all significant (pairwise + multigroup) by feature
    ax = axes[1, 0]
    # Combine significant results
    all_sig_features = list(sig_raw["feature"].value_counts().head(15).index)
    if len(mg_sig) > 0:
        mg_feat_counts = mg_sig["feature"].value_counts()
        for f in mg_feat_counts.head(5).index:
            if f not in all_sig_features:
                all_sig_features.append(f)
    all_sig_features = all_sig_features[:15]

    # Get all comparison names with significant results
    all_sig_comps = list(sig_raw["comparison"].value_counts().head(10).index)
    if len(mg_sig) > 0:
        for c in mg_sig["comparison"].value_counts().head(5).index:
            if c not in all_sig_comps:
                all_sig_comps.append(c)
    all_sig_comps = all_sig_comps[:15]

    if all_sig_features and all_sig_comps:
        heat_data = pd.DataFrame(np.nan, index=all_sig_comps, columns=all_sig_features)
        for _, row in sig_raw.iterrows():
            if row["feature"] in all_sig_features and row["comparison"] in all_sig_comps:
                heat_data.loc[row["comparison"], row["feature"]] = -np.log10(row["p_value"])
        if len(mg_sig) > 0:
            for _, row in mg_sig.iterrows():
                if row["feature"] in all_sig_features and row["comparison"] in all_sig_comps:
                    heat_data.loc[row["comparison"], row["feature"]] = -np.log10(row["p_value"])

        im = ax.imshow(
            heat_data.values.astype(float), aspect="auto", cmap="YlOrRd", interpolation="nearest"
        )
        ax.set_xticks(range(len(all_sig_features)))
        ax.set_xticklabels(all_sig_features, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(all_sig_comps)))
        ax.set_yticklabels(all_sig_comps, fontsize=7)
        ax.set_title("Significance Heatmap (pairwise + multi-group)")
        plt.colorbar(im, ax=ax, shrink=0.8, label="-log10(p)")

    # Panel 4: Summary counts by test type
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = []
    summary_text.append(f"TOTAL TESTS: {n_tests + len(mg_df)}")
    summary_text.append("")
    summary_text.append(f"PAIRWISE (Mann-Whitney U): {n_tests}")
    summary_text.append(f"  Significant p<0.05: {len(sig_raw)}")
    summary_text.append(f"  Significant BH q<0.05: {len(sig_bh)}")
    summary_text.append("")
    if len(mg_df) > 0:
        for tt in mg_df["test_type"].unique():
            tt_df = mg_df[mg_df["test_type"] == tt]
            tt_sig = tt_df[tt_df["p_value"] < 0.05]
            tt_bh = tt_df[tt_df["p_bh"] < 0.05]
            summary_text.append(f"{tt.upper()}: {len(tt_df)}")
            summary_text.append(f"  Significant p<0.05: {len(tt_sig)}")
            summary_text.append(f"  Significant BH q<0.05: {len(tt_bh)}")
            summary_text.append("")
    summary_text.append("")
    summary_text.append("TOP SIGNIFICANT INTERACTIONS:")
    if len(mg_sig) > 0:
        for _, row in mg_sig[mg_sig["test_type"].str.contains("interaction")].head(5).iterrows():
            summary_text.append(f"  {row['comparison']}: {row['feature']} p={row['p_value']:.4f}")

    ax.text(
        0.05,
        0.95,
        "\n".join(summary_text),
        transform=ax.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=9,
    )
    ax.set_title("Summary", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "mk_comprehensive_volcano.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_path}")

    # Figure 2: Comparison-centric — how many features significant per comparison
    all_comp_counts = sig_raw.groupby("comparison").size()
    if len(mg_sig) > 0:
        mg_comp_counts = mg_sig.groupby("comparison").size()
        all_comp_counts = pd.concat([all_comp_counts, mg_comp_counts])
    all_comp_counts = all_comp_counts.sort_values(ascending=False).head(30)

    if len(all_comp_counts) > 0:
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = []
        for comp in all_comp_counts.index:
            if "Interaction" in comp:
                colors.append("#ff8800")
            elif "KW_" in comp:
                colors.append("#4488cc")
            elif "Ratio" in comp:
                colors.append("#8844aa")
            elif "_vs_" in comp:
                colors.append("#44aa44")
            elif "|" not in comp and "KW" not in comp:
                colors.append("#cc4444")
            elif comp.count("_") >= 1 and "|" in comp:
                colors.append("#66bb66")
            else:
                colors.append("#cc8844")

        ax.barh(range(len(all_comp_counts)), all_comp_counts.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(all_comp_counts)))
        ax.set_yticklabels(all_comp_counts.index, fontsize=8)
        ax.set_xlabel("Number of significant features (p<0.05)")
        ax.set_title(
            "Significant Features per Comparison (all test types)", fontsize=14, fontweight="bold"
        )
        ax.invert_yaxis()

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#cc4444", alpha=0.8, label="Main effects (pairwise)"),
            Patch(facecolor="#cc8844", alpha=0.8, label="Stratified pairwise"),
            Patch(facecolor="#44aa44", alpha=0.8, label="Group-vs-group pairwise"),
            Patch(facecolor="#66bb66", alpha=0.8, label="3-way stratified"),
            Patch(facecolor="#4488cc", alpha=0.8, label="Kruskal-Wallis"),
            Patch(facecolor="#ff8800", alpha=0.8, label="Interaction (ART)"),
            Patch(facecolor="#8844aa", alpha=0.8, label="Humerus-corrected ratio"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.tight_layout()
        fig_path2 = OUTPUT_DIR / "mk_comprehensive_comparisons.png"
        plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fig_path2}")

    # ── Print final summary ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("BIOLOGICAL HIGHLIGHTS")
    print(f"{'='*80}")

    # Interactions
    if len(mg_sig) > 0:
        int_sig = mg_sig[mg_sig["test_type"].str.contains("interaction")]
        if len(int_sig) > 0:
            print(f"\n  INTERACTIONS ({len(int_sig)} significant):")
            for _, row in int_sig.head(15).iterrows():
                bh_tag = " [BH]" if row["significant_bh"] else ""
                print(
                    f"    {row['comparison']:35s} {row['feature']:25s} p={row['p_value']:.4f}{bh_tag}"
                )

        kw_sig = mg_sig[mg_sig["test_type"] == "kruskal_wallis"]
        if len(kw_sig) > 0:
            print(f"\n  MULTI-GROUP KRUSKAL-WALLIS ({len(kw_sig)} significant):")
            for _, row in kw_sig.head(15).iterrows():
                bh_tag = " [BH]" if row["significant_bh"] else ""
                print(
                    f"    {row['comparison']:35s} {row['feature']:25s} p={row['p_value']:.4f} H={row['H_statistic']:.1f}{bh_tag}"
                )

    # Pairwise highlights by theme
    themes = {
        "Sex effect": sig_raw[(sig_raw["group_var"] == "sex") & (~sig_raw["is_ratio"])],
        "Treatment effect": sig_raw[(sig_raw["group_var"] == "treatment") & (~sig_raw["is_ratio"])],
        "Bone effect": sig_raw[(sig_raw["group_var"] == "bone") & (~sig_raw["is_ratio"])],
        "Humerus-corrected": sig_raw[sig_raw["is_ratio"]],
        "4-group pairwise": sig_raw[
            sig_raw["comparison"].str.contains("_vs_") & ~sig_raw["is_ratio"]
        ],
    }

    for theme, theme_df in themes.items():
        if len(theme_df) > 0:
            print(f"\n  {theme} ({len(theme_df)} significant):")
            for _, row in theme_df.head(10).iterrows():
                bh_tag = " [BH]" if row["significant_bh"] else ""
                print(
                    f"    {row['comparison']:35s} {row['feature']:25s} p={row['p_value']:.4f} d={row['cohens_d']:+.2f} {row['direction']}{bh_tag}"
                )
        else:
            print(f"\n  {theme}: none significant")

    print("\n\nDone.")
    print(f"  Pairwise CSV:    {csv_path}")
    print(f"  Multi-group CSV: {mg_csv}")
    print(f"  Figures:         {OUTPUT_DIR}/mk_comprehensive_*.png")


if __name__ == "__main__":
    main()
