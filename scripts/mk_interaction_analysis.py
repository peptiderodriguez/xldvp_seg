#!/usr/bin/env python3
"""ART Interaction Effects Analysis for MK morphological features.

Uses proper Aligned Rank Transform (ART, Wobbrock et al. 2011) with
statsmodels Type II ANOVA to decompose main effects and interactions
in a 3-factor design: Sex x Treatment x Bone.

For each effect of interest:
1. Fit full OLS on raw data to decompose variance
2. Align: strip all effects EXCEPT the one being tested
3. Rank the aligned residuals
4. Run ANOVA on ranks, extract F/p for the target effect

Also runs a simple rank ANOVA (rank then fit) for comparison.

Outputs:
- mk_clf075_anova_table.csv  -- full ANOVA decomposition per feature
- mk_clf075_interactions.png -- interaction plots for near-significant interactions
- mk_clf075_anova_summary.png -- heatmap of p-values and effect sizes
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ── Config ──────────────────────────────────────────────────────────────
DETECTIONS_FULL = Path('/Volumes/pool-mann-edwin/bm_lmd_feb2026/mk_clf084_dataset/all_mks_clf075_light.json')
DETECTIONS_BONE = Path('/Volumes/pool-mann-edwin/bm_lmd_feb2026/mk_clf084_dataset/all_mks_clf075_with_bone.json')
TISSUE_AREAS = Path('/Volumes/pool-mann-edwin/bm_lmd_feb2026/mk_clf084_dataset/tissue_areas_by_bone.json')
OUTPUT_DIR = Path('/Volumes/pool-mann-edwin/bm_lmd_feb2026/mk_clf084_dataset')

SKIP_PREFIXES = ('sam2_', 'resnet_', 'dinov2_')

# Morphological features to analyze (whitelist — no color/intensity features)
MORPH_FEATURES = [
    'area', 'area_um2', 'perimeter', 'circularity', 'solidity',
    'aspect_ratio', 'eccentricity', 'elongation', 'equiv_diameter', 'extent',
]
DENSITY_FEATURES = ['n_cells', 'density_per_mm2']

EXCLUDE_SLIDES = {'2025_11_18_FGC3'}

# Effect name <-> statsmodels ANOVA table label mapping
SM_EFFECT_MAP = {
    'sex': 'C(sex)',
    'treatment': 'C(treatment)',
    'bone': 'C(bone)',
    'sex:treatment': 'C(sex):C(treatment)',
    'sex:bone': 'C(sex):C(bone)',
    'treatment:bone': 'C(treatment):C(bone)',
    'sex:treatment:bone': 'C(sex):C(treatment):C(bone)',
}

ALL_EFFECTS = list(SM_EFFECT_MAP.keys())
INTERACTION_EFFECTS = [e for e in ALL_EFFECTS if ':' in e]

LINE_COLORS = {
    'F': '#e74c3c', 'M': '#3498db',
    'GC': '#e67e22', 'HU': '#2ecc71',
    'femur': '#9b59b6', 'humerus': '#1abc9c',
}


# ── Data loading (same logic as mk_comprehensive_analysis.py) ──────────
def parse_slide(name):
    """Parse '2025_11_18_FGC1' -> {sex, treatment, replicate}."""
    short = name.replace('2025_11_18_', '')
    sex = 'F' if short[0] == 'F' else 'M'
    treatment = 'GC' if 'GC' in short else 'HU'
    replicate = short[-1]
    return {'sex': sex, 'treatment': treatment, 'replicate': replicate, 'short': short}


def load_and_prepare_data(score_threshold=0.75):
    """Load detections, aggregate to slide x bone medians, IQR filter."""
    print(f"Loading data (score threshold >= {score_threshold})...")
    with open(DETECTIONS_FULL) as f:
        full_data = json.load(f)
    with open(DETECTIONS_BONE) as f:
        bone_data = json.load(f)
    with open(TISSUE_AREAS) as f:
        tissue_data = json.load(f)

    uid_to_bone = {d['uid']: d['bone'] for d in bone_data}
    tissue_info = {r['slide']: r['bones'] for r in tissue_data['results']}

    # Build per-cell DataFrame (with score filtering)
    n_total, n_filtered = 0, 0
    rows = []
    for det in full_data:
        n_total += 1
        score = det.get('mk_score', 0)
        if score < score_threshold:
            n_filtered += 1
            continue
        uid = det['uid']
        slide = det['slide']
        bone = uid_to_bone.get(uid, '')
        if not bone or bone == 'unknown':
            continue
        meta = parse_slide(slide)
        feats = det.get('features', det.get('features_morph_color', {}))
        row = {
            'uid': uid, 'slide': slide, 'short': meta['short'],
            'sex': meta['sex'], 'treatment': meta['treatment'],
            'replicate': meta['replicate'], 'bone': bone,
            'area_um2': det.get('area_um2', feats.get('area', np.nan)),
        }
        for k, v in feats.items():
            if any(k.startswith(p) for p in SKIP_PREFIXES):
                continue
            try:
                row[k] = float(v)
            except (ValueError, TypeError):
                continue
        rows.append(row)
    if n_filtered:
        print(f"  Score filter: {n_total} total, {n_filtered} below {score_threshold}, {len(rows)} kept")

    df = pd.DataFrame(rows)
    print(f"  {len(df)} cells loaded")

    # Identify feature columns for aggregation
    meta_cols = {'uid', 'slide', 'short', 'sex', 'treatment', 'replicate', 'bone', 'mk_score'}
    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in ('float64', 'int64')]

    # Aggregate to slide x bone medians
    agg_df = df.groupby(
        ['slide', 'short', 'sex', 'treatment', 'replicate', 'bone']
    )[feature_cols].median().reset_index()

    # Add density metrics
    density_rows = []
    for slide in df['slide'].unique():
        for bone in ['femur', 'humerus']:
            mask = (df['slide'] == slide) & (df['bone'] == bone)
            n_cells = mask.sum()
            t_info = tissue_info.get(slide, {}).get(bone, {})
            tissue_area = t_info.get('tissue_area_mm2', np.nan)
            density = n_cells / tissue_area if tissue_area and tissue_area > 0 else np.nan
            density_rows.append({
                'slide': slide, 'bone': bone,
                'n_cells': n_cells, 'density_per_mm2': density,
            })
    agg_df = agg_df.merge(pd.DataFrame(density_rows), on=['slide', 'bone'], how='left')

    # Exclude slides
    agg_df = agg_df[~agg_df['slide'].isin(EXCLUDE_SLIDES)].reset_index(drop=True)
    print(f"  {len(agg_df)} slide x bone observations (after excluding {EXCLUDE_SLIDES})")

    # Select morph + density features (whitelist)
    available = set(agg_df.columns)
    features = [f for f in MORPH_FEATURES + DENSITY_FEATURES if f in available]
    print(f"  Features for analysis: {features}")

    # IQR outlier filtering
    keep_rules = {('n_cells', 'FHU2', 'femur'), ('n_cells', 'FHU4', 'femur')}
    n_removed = 0
    for feat in features:
        vals = agg_df[feat].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((agg_df[feat] < lo) | (agg_df[feat] > hi)).fillna(False)
        for idx in agg_df[outliers].index:
            row = agg_df.loc[idx]
            if (feat, row['short'], row['bone']) in keep_rules:
                outliers.at[idx] = False
        n_out = outliers.sum()
        if n_out > 0:
            n_removed += n_out
            print(f"    IQR: {feat} -- removed {n_out} outlier(s)")
            agg_df.loc[outliers, feat] = np.nan

    if n_removed:
        print(f"  Total outlier values removed: {n_removed}")

    return agg_df, features


# ── ART ANOVA implementation ────────────────────────────────────────────
def _get_effect_col_indices(model, effect_name):
    """Map an effect name to its column indices in the design matrix."""
    effect_factors = set(effect_name.split(':'))
    indices = []
    for i, pname in enumerate(model.model.exog_names):
        if pname == 'Intercept':
            continue
        parts = pname.split(':')
        param_factors = set()
        for p in parts:
            for fname in ('sex', 'treatment', 'bone'):
                if fname in p:
                    param_factors.add(fname)
        if param_factors == effect_factors:
            indices.append(i)
    return indices


def art_anova_feature(df, feat):
    """Run proper ART ANOVA for one feature (Wobbrock et al. 2011).

    For each effect E in the full 3-factor model:
      1. Fit full OLS -> get residuals and per-effect contributions
      2. Aligned_E = residuals + contribution_of_E  (strips everything except E)
      3. Rank the aligned values
      4. Fit full ANOVA on ranks, read F/p for effect E (Type II SS)

    Returns dict of {effect_name: {F, p, df_effect, df_resid, partial_eta2, n}}.
    """
    working = df[['sex', 'treatment', 'bone', feat]].dropna().copy()
    if len(working) < 10:
        return None

    formula = f'Q("{feat}") ~ C(sex) * C(treatment) * C(bone)'
    try:
        full_model = smf.ols(formula, data=working).fit()
    except Exception as e:
        print(f"    WARNING: full OLS failed for {feat}: {e}")
        return None

    X = full_model.model.exog
    params = full_model.params
    residuals = full_model.resid.values

    results = {}
    for effect_name in ALL_EFFECTS:
        col_indices = _get_effect_col_indices(full_model, effect_name)
        if not col_indices:
            continue

        # Effect contribution for each observation
        effect_contrib = sum(X[:, ci] * params.iloc[ci] for ci in col_indices)

        # Align: keep only this effect + error
        aligned = residuals + effect_contrib
        ranked = rankdata(aligned)

        temp = working.copy()
        temp['_rank'] = ranked

        try:
            rank_model = smf.ols('_rank ~ C(sex) * C(treatment) * C(bone)', data=temp).fit()
            anova_tab = anova_lm(rank_model, typ=2)
        except Exception:
            continue

        sm_label = SM_EFFECT_MAP[effect_name]
        if sm_label not in anova_tab.index:
            continue

        row = anova_tab.loc[sm_label]
        ss_eff = row['sum_sq']
        ss_res = anova_tab.loc['Residual', 'sum_sq']
        peta2 = ss_eff / (ss_eff + ss_res) if (ss_eff + ss_res) > 0 else 0.0

        results[effect_name] = {
            'F': row['F'],
            'p': row['PR(>F)'],
            'df_effect': int(row['df']),
            'df_resid': int(anova_tab.loc['Residual', 'df']),
            'ss_effect': ss_eff,
            'ss_resid': ss_res,
            'partial_eta2': peta2,
            'n': len(working),
        }

    return results


def simple_rank_anova_feature(df, feat):
    """Simple rank ANOVA: rank raw values, fit OLS, Type II.

    Not proper ART (no alignment step) but useful for comparison.
    """
    working = df[['sex', 'treatment', 'bone', feat]].dropna().copy()
    if len(working) < 10:
        return None

    working['_rank'] = rankdata(working[feat].values)
    try:
        model = smf.ols('_rank ~ C(sex) * C(treatment) * C(bone)', data=working).fit()
        anova_tab = anova_lm(model, typ=2)
    except Exception:
        return None

    results = {}
    for effect_name, sm_label in SM_EFFECT_MAP.items():
        if sm_label not in anova_tab.index:
            continue
        row = anova_tab.loc[sm_label]
        ss_eff = row['sum_sq']
        ss_res = anova_tab.loc['Residual', 'sum_sq']
        peta2 = ss_eff / (ss_eff + ss_res) if (ss_eff + ss_res) > 0 else 0.0
        results[effect_name] = {
            'F': row['F'],
            'p': row['PR(>F)'],
            'df_effect': int(row['df']),
            'df_resid': int(anova_tab.loc['Residual', 'df']),
            'partial_eta2': peta2,
            'n': len(working),
        }
    return results


# ── BH correction ───────────────────────────────────────────────────────
def bh_correct(results_df, p_col='art_p', out_col='art_p_bh'):
    """Benjamini-Hochberg correction on a p-value column."""
    valid = results_df[p_col].dropna()
    if len(valid) == 0:
        results_df[out_col] = 1.0
        return results_df

    sorted_idx = valid.sort_values().index
    n = len(sorted_idx)
    for rank, idx in enumerate(sorted_idx, 1):
        results_df.loc[idx, out_col] = min(1.0, results_df.loc[idx, p_col] * n / rank)

    # Enforce monotonicity (step-up)
    bh_vals = results_df.loc[sorted_idx, out_col].values.copy()
    for i in range(len(bh_vals) - 2, -1, -1):
        bh_vals[i] = min(bh_vals[i], bh_vals[i + 1])
    results_df.loc[sorted_idx, out_col] = bh_vals

    results_df[out_col] = results_df[out_col].fillna(1.0)
    return results_df


# ── Interaction plots ───────────────────────────────────────────────────
def _cell_stats(df, feat, factor_x, factor_line, factor_facet=None):
    """Compute cell means +/- SE for interaction plots."""
    working = df[['sex', 'treatment', 'bone', feat]].dropna()
    x_levels = sorted(working[factor_x].unique())
    line_levels = sorted(working[factor_line].unique())
    facet_levels = sorted(working[factor_facet].unique()) if factor_facet else [None]

    stats = {}
    for facet in facet_levels:
        stats[facet] = {}
        for line in line_levels:
            means, ses, ns = [], [], []
            for x in x_levels:
                mask = (working[factor_x] == x) & (working[factor_line] == line)
                if facet is not None:
                    mask &= (working[factor_facet] == facet)
                vals = working.loc[mask, feat].values
                n = len(vals)
                means.append(np.mean(vals) if n > 0 else np.nan)
                ses.append(np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0)
                ns.append(n)
            stats[facet][line] = {'means': means, 'ses': ses, 'ns': ns}
    return x_levels, line_levels, facet_levels, stats


def _draw_interaction(ax, df, feat, factor_x, factor_line, factor_facet=None,
                      p_value=None, title_label=None):
    """Draw one interaction plot on the given axes."""
    x_levels, line_levels, facet_levels, stats = _cell_stats(
        df, feat, factor_x, factor_line, factor_facet)
    x_pos = np.arange(len(x_levels))

    for line_lev in line_levels:
        color = LINE_COLORS.get(line_lev, '#333333')
        if factor_facet:
            for fi, facet in enumerate(facet_levels):
                m = stats[facet][line_lev]['means']
                s = stats[facet][line_lev]['ses']
                ls = '-' if fi == 0 else '--'
                ax.errorbar(x_pos + (fi - 0.5) * 0.08, m, yerr=s,
                            marker='o', markersize=5, color=color,
                            linestyle=ls, alpha=0.7, capsize=3, linewidth=1.5,
                            label=f'{line_lev} ({facet})')
        else:
            m = stats[None][line_lev]['means']
            s = stats[None][line_lev]['ses']
            ax.errorbar(x_pos, m, yerr=s, marker='o', markersize=7,
                        color=color, linestyle='-', alpha=0.85,
                        capsize=4, linewidth=2, label=line_lev)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_levels)
    ax.set_xlabel(factor_x.capitalize())

    # Background color by significance
    if p_value is not None:
        if p_value < 0.05:
            ax.set_facecolor('#fff8dc')
        elif p_value < 0.10:
            ax.set_facecolor('#fffff0')

    # Title with p-value and stars
    title = title_label or f'{factor_line} x {factor_x}'
    if p_value is not None:
        stars = ('***' if p_value < 0.001 else '**' if p_value < 0.01
                 else '*' if p_value < 0.05 else '\u2020' if p_value < 0.10 else '')
        title += f'\np={p_value:.3f} {stars}'
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='best')


def generate_interaction_plots(df, features, anova_results, output_path):
    """Generate interaction plot grid for features with near-significant interactions."""
    # Find features with any interaction p < 0.10
    plot_features = []
    for feat in features:
        if feat not in anova_results:
            continue
        art = anova_results[feat]['art']
        if any(art.get(e, {}).get('p', 1.0) < 0.10 for e in INTERACTION_EFFECTS):
            plot_features.append(feat)

    # Fallback: show top 6 by minimum interaction p
    if not plot_features:
        scored = []
        for feat in features:
            if feat not in anova_results:
                continue
            art = anova_results[feat]['art']
            min_p = min((art[e]['p'] for e in INTERACTION_EFFECTS if e in art), default=1.0)
            scored.append((feat, min_p))
        scored.sort(key=lambda x: x[1])
        plot_features = [f for f, _ in scored[:6]]
        print(f"  No interactions at p<0.10; showing top {len(plot_features)} by min interaction p")

    n_feats = len(plot_features)
    if n_feats == 0:
        print("  No features to plot")
        return

    # Layout: rows = features, 4 columns
    #   Col 0: Sex x Treatment  (x=Treatment, lines=Sex)
    #   Col 1: Sex x Bone       (x=Bone, lines=Sex)
    #   Col 2: Treatment x Bone (x=Bone, lines=Treatment)
    #   Col 3: Sex x Treatment  faceted by Bone
    interaction_configs = [
        ('sex:treatment', 'treatment', 'sex', None, 'Sex x Treatment'),
        ('sex:bone', 'bone', 'sex', None, 'Sex x Bone'),
        ('treatment:bone', 'bone', 'treatment', None, 'Treatment x Bone'),
        ('sex:treatment', 'treatment', 'sex', 'bone', 'Sex x Trt | Bone'),
    ]
    n_cols = len(interaction_configs)

    fig, axes = plt.subplots(n_feats, n_cols, figsize=(16, 3.5 * n_feats),
                             squeeze=False)
    fig.suptitle('ART Interaction Analysis: MK Morphological Features\n'
                 'Yellow = p<0.05, Light yellow = p<0.10, White = n.s.',
                 fontsize=13, fontweight='bold', y=1.02)

    for ri, feat in enumerate(plot_features):
        art = anova_results[feat]['art']
        for ci, (effect_key, fx, fl, ff, label) in enumerate(interaction_configs):
            ax = axes[ri, ci]
            p_val = art.get(effect_key, {}).get('p', 1.0)
            _draw_interaction(ax, df, feat, fx, fl, ff,
                              p_value=p_val, title_label=f'{feat}\n{label}')
            if ci == 0:
                ax.set_ylabel(feat, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


# ── ANOVA summary heatmap ───────────────────────────────────────────────
def generate_anova_summary_figure(results_df, features, output_path):
    """Heatmap of -log10(p) and partial eta-squared (effects x features)."""
    feat_order = [f for f in features if f in results_df['feature'].values]
    if not feat_order:
        print("  No features for summary figure")
        return

    pivot_p = results_df.pivot(index='feature', columns='effect', values='art_p')
    pivot_eta = results_df.pivot(index='feature', columns='effect', values='art_partial_eta2')

    # Reorder rows and columns
    pivot_p = pivot_p.reindex(index=feat_order, columns=ALL_EFFECTS)
    pivot_eta = pivot_eta.reindex(index=feat_order, columns=ALL_EFFECTS)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, max(5, len(feat_order) * 0.6)),
        gridspec_kw={'width_ratios': [3, 2]})

    # Panel 1: -log10(p) heatmap
    log_p = -np.log10(np.nan_to_num(pivot_p.values.astype(float), nan=1.0))
    im1 = ax1.imshow(log_p, aspect='auto', cmap='YlOrRd', interpolation='nearest',
                     vmin=0, vmax=max(3, np.nanmax(log_p) * 1.1))

    # Significance stars
    for i in range(len(feat_order)):
        for j in range(len(ALL_EFFECTS)):
            p = pivot_p.iloc[i, j]
            if pd.isna(p):
                continue
            txt = ('***' if p < 0.001 else '**' if p < 0.01
                   else '*' if p < 0.05 else '\u2020' if p < 0.10 else '')
            if txt:
                ax1.text(j, i, txt, ha='center', va='center', fontsize=8,
                         fontweight='bold' if p < 0.05 else 'normal')

    ax1.set_xticks(range(len(ALL_EFFECTS)))
    ax1.set_xticklabels(ALL_EFFECTS, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(feat_order)))
    ax1.set_yticklabels(feat_order, fontsize=9)
    ax1.set_title('ART ANOVA: -log10(p-value)', fontsize=12, fontweight='bold')
    ax1.axvline(2.5, color='white', linewidth=2)  # separate main / interaction
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='-log10(p)')

    # Panel 2: partial eta-squared heatmap
    eta_vals = np.nan_to_num(pivot_eta.values.astype(float), nan=0)
    im2 = ax2.imshow(eta_vals, aspect='auto', cmap='Blues', interpolation='nearest',
                     vmin=0, vmax=max(0.5, np.nanmax(eta_vals) * 1.1))

    for i in range(len(feat_order)):
        for j in range(len(ALL_EFFECTS)):
            val = pivot_eta.iloc[i, j]
            if pd.notna(val) and val > 0.06:
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7)

    ax2.set_xticks(range(len(ALL_EFFECTS)))
    ax2.set_xticklabels(ALL_EFFECTS, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(feat_order)))
    ax2.set_yticklabels(feat_order, fontsize=9)
    ax2.set_title('Partial \u03b7\u00b2 (effect size)', fontsize=12, fontweight='bold')
    ax2.axvline(2.5, color='white', linewidth=2)
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Partial \u03b7\u00b2')

    plt.suptitle(f'MK -- ART ANOVA Decomposition (Sex x Treatment x Bone)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ART Interaction Effects Analysis')
    parser.add_argument('--score-threshold', type=float, default=0.75,
                        help='Minimum mk_score to include (default: 0.75)')
    args = parser.parse_args()

    score_thresh = args.score_threshold
    # Tag for output filenames (e.g. 0.80 -> "080")
    tag = f"clf{score_thresh:.2f}".replace('0.', '0')

    agg_df, features = load_and_prepare_data(score_threshold=score_thresh)

    # Print design summary
    print(f"\n  Design cells (Sex x Treatment x Bone):")
    for s in sorted(agg_df['sex'].unique()):
        for t in sorted(agg_df['treatment'].unique()):
            for b in sorted(agg_df['bone'].unique()):
                n = ((agg_df['sex'] == s) & (agg_df['treatment'] == t) & (agg_df['bone'] == b)).sum()
                print(f"    {s}_{t}_{b}: n={n}")

    # Run ART ANOVA for each feature
    print(f"\nRunning ART ANOVA (proper alignment, Wobbrock et al. 2011)...")
    anova_results = {}
    for feat in features:
        art = art_anova_feature(agg_df, feat)
        simple = simple_rank_anova_feature(agg_df, feat)
        if art is None and simple is None:
            print(f"  SKIP {feat}: insufficient data")
            continue
        anova_results[feat] = {'art': art or {}, 'simple': simple or {}}

    # Build flat results table
    rows = []
    for feat, res in anova_results.items():
        for effect in ALL_EFFECTS:
            a = res['art'].get(effect, {})
            s = res['simple'].get(effect, {})
            rows.append({
                'feature': feat,
                'effect': effect,
                'is_interaction': ':' in effect,
                'art_F': a.get('F', np.nan),
                'art_p': a.get('p', np.nan),
                'art_partial_eta2': a.get('partial_eta2', np.nan),
                'art_df_effect': a.get('df_effect', np.nan),
                'art_df_resid': a.get('df_resid', np.nan),
                'art_ss_effect': a.get('ss_effect', np.nan),
                'art_ss_resid': a.get('ss_resid', np.nan),
                'art_n': a.get('n', np.nan),
                'simple_F': s.get('F', np.nan),
                'simple_p': s.get('p', np.nan),
                'simple_partial_eta2': s.get('partial_eta2', np.nan),
            })

    results_df = pd.DataFrame(rows)

    # BH correction
    results_df = bh_correct(results_df, 'art_p', 'art_p_bh')
    results_df['art_sig_raw'] = results_df['art_p'] < 0.05
    results_df['art_sig_bh'] = results_df['art_p_bh'] < 0.05
    results_df = results_df.sort_values('art_p')

    # Save CSV
    csv_path = OUTPUT_DIR / f'mk_{tag}_anova_table.csv'
    results_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nSaved ANOVA table ({len(results_df)} rows): {csv_path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("ART ANOVA RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Features tested: {len(anova_results)}")
    print(f"Total effect tests: {len(results_df)}")
    sig_raw = results_df['art_sig_raw'].sum()
    sig_bh = results_df['art_sig_bh'].sum()
    print(f"Significant (p<0.05 uncorrected): {sig_raw}")
    print(f"Significant (BH q<0.05): {sig_bh}")

    # Main effects
    print(f"\n{'_' * 80}")
    print("MAIN EFFECTS (ART, sorted by p)")
    print(f"{'_' * 80}")
    mains = results_df[~results_df['is_interaction']].sort_values('art_p')
    for _, r in mains.head(20).iterrows():
        stars = ('***' if r['art_p'] < 0.001 else '**' if r['art_p'] < 0.01
                 else '*' if r['art_p'] < 0.05 else '\u2020' if r['art_p'] < 0.10 else '')
        print(f"  F({r['art_df_effect']:.0f},{r['art_df_resid']:.0f})={r['art_F']:6.2f}  "
              f"p={r['art_p']:.4f} (q={r['art_p_bh']:.4f})  "
              f"\u03b7\u00b2p={r['art_partial_eta2']:.3f}  {stars:4s}  "
              f"{r['effect']:20s} {r['feature']}")

    # Interactions
    print(f"\n{'_' * 80}")
    print("INTERACTIONS (ART, sorted by p)")
    print(f"{'_' * 80}")
    ints = results_df[results_df['is_interaction']].sort_values('art_p')
    for _, r in ints.head(25).iterrows():
        stars = ('***' if r['art_p'] < 0.001 else '**' if r['art_p'] < 0.01
                 else '*' if r['art_p'] < 0.05 else '\u2020' if r['art_p'] < 0.10 else '')
        print(f"  F({r['art_df_effect']:.0f},{r['art_df_resid']:.0f})={r['art_F']:6.2f}  "
              f"p={r['art_p']:.4f} (q={r['art_p_bh']:.4f})  "
              f"\u03b7\u00b2p={r['art_partial_eta2']:.3f}  {stars:4s}  "
              f"{r['effect']:28s} {r['feature']}")

    # ART vs simple rank ANOVA comparison
    print(f"\n{'_' * 80}")
    print("ART vs SIMPLE RANK ANOVA COMPARISON (interaction terms)")
    print(f"{'_' * 80}")
    for _, r in ints.head(15).iterrows():
        print(f"  {r['effect']:28s} {r['feature']:18s}  "
              f"ART p={r['art_p']:.4f}  Simple p={r['simple_p']:.4f}  "
              f"\u0394={abs(r['art_p'] - r['simple_p']):.4f}")

    # Generate figures
    print(f"\n{'_' * 80}")
    print("Generating figures...")

    fig_interactions = OUTPUT_DIR / f'mk_{tag}_interactions.png'
    generate_interaction_plots(agg_df, features, anova_results, fig_interactions)

    fig_summary = OUTPUT_DIR / f'mk_{tag}_anova_summary.png'
    generate_anova_summary_figure(results_df, features, fig_summary)

    print(f"\nDone.")
    print(f"  ANOVA table:       {csv_path}")
    print(f"  Interaction plots: {fig_interactions}")
    print(f"  Summary heatmap:   {fig_summary}")


if __name__ == '__main__':
    main()
