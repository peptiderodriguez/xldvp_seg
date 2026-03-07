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
- mk_clf{tag}_anova_table.csv  -- full ANOVA decomposition per feature
- mk_clf{tag}_interactions.png -- interaction plots for near-significant interactions
- mk_clf{tag}_anova_summary.png -- heatmap of p-values and effect sizes
- mk_clf{tag}_dashboard.png -- 3x3 biological summary dashboard
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


# ── Dashboard helpers ──────────────────────────────────────────────────
def _compute_hu_effects(agg_df, features):
    """Compute HU-GC delta for each feature, per sex, per bone (and collapsed).

    Returns: effects[feat][sex][bone] with bone in {femur, humerus, collapsed}.
    Each entry has gc_mean, hu_mean, delta, pct_change.
    """
    effects = {}
    for feat in features:
        effects[feat] = {}
        for sex in ['F', 'M']:
            effects[feat][sex] = {}
            for bone in ['femur', 'humerus', 'collapsed']:
                if bone == 'collapsed':
                    slide_means = agg_df[agg_df['sex'] == sex].groupby(
                        ['slide', 'treatment'])[feat].mean().reset_index()
                    gc_vals = slide_means[slide_means['treatment'] == 'GC'][feat].dropna().values
                    hu_vals = slide_means[slide_means['treatment'] == 'HU'][feat].dropna().values
                else:
                    mask = (agg_df['sex'] == sex) & (agg_df['bone'] == bone)
                    gc_vals = agg_df[mask & (agg_df['treatment'] == 'GC')][feat].dropna().values
                    hu_vals = agg_df[mask & (agg_df['treatment'] == 'HU')][feat].dropna().values

                gc_mean = np.mean(gc_vals) if len(gc_vals) > 0 else np.nan
                hu_mean = np.mean(hu_vals) if len(hu_vals) > 0 else np.nan
                delta = hu_mean - gc_mean if not (np.isnan(gc_mean) or np.isnan(hu_mean)) else np.nan
                pct = (delta / abs(gc_mean) * 100) if (not np.isnan(delta) and abs(gc_mean) > 1e-9) else np.nan

                effects[feat][sex][bone] = {
                    'gc_mean': gc_mean, 'hu_mean': hu_mean,
                    'delta': delta, 'pct_change': pct,
                }
    return effects


# ── Summary dashboard ──────────────────────────────────────────────────
def generate_summary_dashboard(agg_df, features, anova_results, results_df, output_path):
    """Generate 3x3 biological summary dashboard.

    Row 1: Sex x Treatment interaction (collapsed across bones) — line plots
    Row 2: Bone attenuation (HU effect by bone) — grouped delta bars
    Row 3: Attenuation ratios, bone concordance scatter, ANOVA heatmap
    """
    KEY_FEATS = ['area_um2', 'elongation', 'density_per_mm2']
    FEAT_LABELS = {
        'area_um2': 'Size (area, \u00b5m\u00b2)',
        'elongation': 'Shape (elongation)',
        'density_per_mm2': 'Density (per mm\u00b2)',
    }
    available = [f for f in KEY_FEATS if f in agg_df.columns]
    if len(available) < 3:
        print(f"  WARNING: only {len(available)}/3 key features available, skipping dashboard")
        return

    all_feats = list(set(features) | set(KEY_FEATS))
    hu_effects = _compute_hu_effects(agg_df, [f for f in all_feats if f in agg_df.columns])

    # Slide-level means collapsed across bones
    slide_collapsed = agg_df.groupby(
        ['slide', 'short', 'sex', 'treatment', 'replicate']
    )[KEY_FEATS].mean().reset_index()

    fig = plt.figure(figsize=(17, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.35,
                          left=0.07, right=0.95, top=0.92, bottom=0.06)
    fig.suptitle('MK Response to Hindlimb Unloading: Sex-Dimorphic Niche Modulation',
                 fontsize=14, fontweight='bold')
    panels = 'ABCDEFGHI'
    rng = np.random.default_rng(42)

    # ── ROW 1: Sex x Treatment interaction (collapsed across bones) ────
    for ci, feat in enumerate(KEY_FEATS):
        ax = fig.add_subplot(gs[0, ci])

        for sex in ['F', 'M']:
            means, ses = [], []
            for trt in ['GC', 'HU']:
                vals = slide_collapsed[
                    (slide_collapsed['sex'] == sex) &
                    (slide_collapsed['treatment'] == trt)
                ][feat].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                ses.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            ax.errorbar([0, 1], means, yerr=ses, marker='o', markersize=8,
                        color=LINE_COLORS[sex], linewidth=2.5, capsize=5,
                        label=sex, zorder=3)

        # Individual slide dots
        for sex in ['F', 'M']:
            for ti, trt in enumerate(['GC', 'HU']):
                vals = slide_collapsed[
                    (slide_collapsed['sex'] == sex) &
                    (slide_collapsed['treatment'] == trt)
                ][feat].dropna().values
                if len(vals) > 0:
                    jitter = rng.uniform(-0.06, 0.06, len(vals))
                    ax.scatter(np.full(len(vals), ti) + jitter, vals,
                               color=LINE_COLORS[sex], alpha=0.5, s=25,
                               edgecolors='black', linewidth=0.5, zorder=2)

        # % change annotations
        for si, sex in enumerate(['F', 'M']):
            eff = hu_effects[feat][sex]['collapsed']
            pct = eff['pct_change']
            if np.isnan(pct):
                continue
            sign = '+' if pct > 0 else ''
            y_pos = eff['hu_mean']
            x_off = 0.22 if si == 0 else -0.30
            ax.annotate(f'{sign}{pct:.0f}%', xy=(1, y_pos),
                        xytext=(1 + x_off, y_pos),
                        fontsize=10, fontweight='bold', color=LINE_COLORS[sex],
                        arrowprops=dict(arrowstyle='->', color=LINE_COLORS[sex],
                                        lw=1.2, shrinkA=3, shrinkB=3),
                        ha='center', va='center')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['GC', 'HU'], fontsize=10)
        ax.set_title(f'{panels[ci]}: {FEAT_LABELS[feat]}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.set_xlabel('Treatment', fontsize=9)

    # ── ROW 2: Bone attenuation (delta bars by bone) ──────────────────
    for ci, feat in enumerate(KEY_FEATS):
        ax = fig.add_subplot(gs[1, ci])

        x = np.arange(2)
        width = 0.35
        fem_d, hum_d = [], []
        for sex in ['F', 'M']:
            fem_d.append(hu_effects[feat][sex]['femur']['delta'])
            hum_d.append(hu_effects[feat][sex]['humerus']['delta'])

        ax.bar(x - width / 2, fem_d, width, label='Femur',
               color=LINE_COLORS['femur'], alpha=0.8, edgecolor='white')
        ax.bar(x + width / 2, hum_d, width, label='Humerus',
               color=LINE_COLORS['humerus'], alpha=0.8, edgecolor='white')
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

        # Overlay individual slide-level deltas
        for si, sex in enumerate(['F', 'M']):
            for bi, bone in enumerate(['femur', 'humerus']):
                bone_color = LINE_COLORS[bone]
                x_center = si + (bi - 0.5) * width
                # Get per-slide values for HU and GC
                mask_sex_bone = (agg_df['sex'] == sex) & (agg_df['bone'] == bone)
                gc_vals = agg_df[mask_sex_bone & (agg_df['treatment'] == 'GC')][feat].dropna().values
                hu_vals = agg_df[mask_sex_bone & (agg_df['treatment'] == 'HU')][feat].dropna().values
                gc_mean = np.mean(gc_vals) if len(gc_vals) > 0 else np.nan
                if np.isnan(gc_mean):
                    continue
                # Each HU slide's delta from the GC mean
                slide_deltas = hu_vals - gc_mean
                if len(slide_deltas) > 0:
                    jitter = rng.uniform(-0.04, 0.04, len(slide_deltas))
                    ax.scatter(np.full(len(slide_deltas), x_center) + jitter,
                               slide_deltas, color=bone_color, alpha=0.6, s=22,
                               edgecolors='black', linewidth=0.5, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(['Female', 'Male'], fontsize=10)
        ax.set_title(f'{panels[ci + 3]}: {FEAT_LABELS[feat]} \u2014 by bone',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.set_ylabel('\u0394 (HU \u2212 GC)', fontsize=9)

        # Attenuation ratio annotation
        for si, sex in enumerate(['F', 'M']):
            afd = abs(hu_effects[feat][sex]['femur']['delta'])
            ahd = abs(hu_effects[feat][sex]['humerus']['delta'])
            if afd < 1e-9 or ahd < 1e-9:
                continue
            ratio = afd / ahd
            ratio_str = f'{ratio:.1f}:1' if ratio >= 1 else f'1:{1 / ratio:.1f}'
            y_vals = [fem_d[si], hum_d[si]]
            y_top = max(y_vals) if max(y_vals) > 0 else min(y_vals)
            offset = abs(max(y_vals) - min(y_vals)) * 0.25 + abs(y_top) * 0.08
            y_pos = y_top + offset if y_top >= 0 else y_top - offset
            ax.text(si, y_pos, f'F:H {ratio_str}', ha='center', fontsize=8,
                    style='italic', color='#555555')

    # ── ROW 3 ─────────────────────────────────────────────────────────

    # Panel G: Attenuation ratio summary (horizontal bars, all split by sex)
    ax_g = fig.add_subplot(gs[2, 0])
    ratio_items = []
    for feat, label in [('area_um2', 'Size'), ('elongation', 'Shape'),
                         ('density_per_mm2', 'Density')]:
        for sex, slabel in [('M', 'M'), ('F', 'F')]:
            afd = abs(hu_effects[feat][sex]['femur']['delta'])
            ahd = abs(hu_effects[feat][sex]['humerus']['delta'])
            if afd > 1e-9 and ahd > 1e-9:
                ratio_items.append({'label': f'{label} ({slabel})',
                                    'ratio': afd / ahd,
                                    'color': LINE_COLORS[sex],
                                    'feat': feat, 'sex': sex})

    if ratio_items:
        y_r = np.arange(len(ratio_items))
        ax_g.barh(y_r, [r['ratio'] for r in ratio_items],
                  color=[r['color'] for r in ratio_items], alpha=0.7, height=0.55)
        # Per-slide ratio dots
        for i, r in enumerate(ratio_items):
            feat, sex = r['feat'], r['sex']
            mask_sex = agg_df['sex'] == sex
            gc_fem = agg_df[mask_sex & (agg_df['treatment'] == 'GC') &
                            (agg_df['bone'] == 'femur')][feat].dropna().values
            gc_hum = agg_df[mask_sex & (agg_df['treatment'] == 'GC') &
                            (agg_df['bone'] == 'humerus')][feat].dropna().values
            gc_fem_mean = np.mean(gc_fem) if len(gc_fem) > 0 else np.nan
            gc_hum_mean = np.mean(gc_hum) if len(gc_hum) > 0 else np.nan
            if np.isnan(gc_fem_mean) or np.isnan(gc_hum_mean):
                continue
            # For each HU slide, compute per-slide ratio
            hu_slides = agg_df[mask_sex & (agg_df['treatment'] == 'HU')]['slide'].unique()
            slide_ratios = []
            for sl in hu_slides:
                fv = agg_df[(agg_df['slide'] == sl) & (agg_df['bone'] == 'femur')][feat].dropna().values
                hv = agg_df[(agg_df['slide'] == sl) & (agg_df['bone'] == 'humerus')][feat].dropna().values
                if len(fv) == 0 or len(hv) == 0:
                    continue
                fd = abs(fv[0] - gc_fem_mean)
                hd = abs(hv[0] - gc_hum_mean)
                if hd > 1e-9:
                    slide_ratios.append(fd / hd)
            if slide_ratios:
                jitter = rng.uniform(-0.12, 0.12, len(slide_ratios))
                ax_g.scatter(slide_ratios, np.full(len(slide_ratios), i) + jitter,
                             color=r['color'], alpha=0.6, s=20,
                             edgecolors='black', linewidth=0.5, zorder=4)
        ax_g.set_yticks(y_r)
        ax_g.set_yticklabels([r['label'] for r in ratio_items], fontsize=9)
        ax_g.axvline(1.0, color='gray', linewidth=1, linestyle='--', alpha=0.6)
        for i, r in enumerate(ratio_items):
            v = r['ratio']
            txt = f'{v:.1f}:1' if v >= 1 else f'1:{1 / v:.1f}'
            ax_g.text(v + 0.12, i, txt, va='center', fontsize=9, fontweight='bold')
        ax_g.set_xlabel('Femur : Humerus  |effect| ratio', fontsize=9)
    ax_g.set_title(f'{panels[6]}: Bone attenuation ratios', fontsize=11, fontweight='bold')

    # Panel H: Within-sex bone concordance (% change scatter)
    ax_h = fig.add_subplot(gs[2, 1])
    morph_feats = [f for f in features if f in hu_effects and f not in DENSITY_FEATURES]

    for sex in ['F', 'M']:
        fem_pcts, hum_pcts = [], []
        for feat in morph_feats:
            fp = hu_effects[feat][sex]['femur']['pct_change']
            hp = hu_effects[feat][sex]['humerus']['pct_change']
            if not (np.isnan(fp) or np.isnan(hp)):
                fem_pcts.append(fp)
                hum_pcts.append(hp)

        if len(fem_pcts) < 3:
            continue
        ax_h.scatter(fem_pcts, hum_pcts, color=LINE_COLORS[sex], s=55, alpha=0.75,
                     label=sex, zorder=3, edgecolors='black', linewidth=0.5)
        r_val = np.corrcoef(fem_pcts, hum_pcts)[0, 1]
        # Regression line
        fa, ha_arr = np.array(fem_pcts), np.array(hum_pcts)
        slope, intercept = np.polyfit(fa, ha_arr, 1)
        x_line = np.linspace(fa.min(), fa.max(), 50)
        ax_h.plot(x_line, slope * x_line + intercept, color=LINE_COLORS[sex],
                  linestyle='--', alpha=0.5, linewidth=1.5)
        y_anchor = 0.95 if sex == 'F' else 0.85
        ax_h.text(0.05, y_anchor, f'{sex}: r={r_val:.2f}',
                  transform=ax_h.transAxes, fontsize=9, color=LINE_COLORS[sex],
                  fontweight='bold', va='top')

    # Identity line
    all_pcts = []
    for sex in ['F', 'M']:
        for feat in morph_feats:
            if feat in hu_effects:
                for bone in ['femur', 'humerus']:
                    p = hu_effects[feat][sex][bone]['pct_change']
                    if not np.isnan(p):
                        all_pcts.append(p)
    ax_h.plot([-20, 40], [-20, 40], 'k--', alpha=0.25, linewidth=0.8)
    ax_h.set_xlim(-20, 40)
    ax_h.set_ylim(-20, 40)
    ax_h.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax_h.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax_h.set_xlabel('Femur % change (HU vs GC)', fontsize=9)
    ax_h.set_ylabel('Humerus % change (HU vs GC)', fontsize=9)
    ax_h.set_title(f'{panels[7]}: Bone concordance (morphology)',
                   fontsize=11, fontweight='bold')
    ax_h.legend(fontsize=8)

    # Panel I: ANOVA effect size heatmap (compact)
    ax_i = fig.add_subplot(gs[2, 2])
    eta_mat = np.full((len(KEY_FEATS), len(ALL_EFFECTS)), np.nan)
    p_mat = np.full_like(eta_mat, np.nan)
    for fi, feat in enumerate(KEY_FEATS):
        for ei, eff in enumerate(ALL_EFFECTS):
            mask = (results_df['feature'] == feat) & (results_df['effect'] == eff)
            rows = results_df[mask]
            if len(rows) > 0:
                eta_mat[fi, ei] = rows.iloc[0]['art_partial_eta2']
                p_mat[fi, ei] = rows.iloc[0]['art_p']

    im = ax_i.imshow(eta_mat, aspect='auto', cmap='Blues', interpolation='nearest',
                     vmin=0, vmax=max(0.5, np.nanmax(eta_mat) * 1.1))
    for fi in range(len(KEY_FEATS)):
        for ei in range(len(ALL_EFFECTS)):
            eta = eta_mat[fi, ei]
            p = p_mat[fi, ei]
            if np.isnan(eta):
                continue
            stars = ('***' if p < 0.001 else '**' if p < 0.01
                     else '*' if p < 0.05 else '\u2020' if p < 0.10 else '')
            txt = f'{eta:.2f}\n{stars}' if stars else f'{eta:.2f}'
            ax_i.text(ei, fi, txt, ha='center', va='center', fontsize=7,
                      fontweight='bold' if p < 0.05 else 'normal',
                      color='white' if eta > 0.4 else 'black')

    # Divider between main effects and interactions
    ax_i.axvline(2.5, color='white', linewidth=2)
    ax_i.set_xticks(range(len(ALL_EFFECTS)))
    eff_labels = [e.replace(':', '\n\u00d7\n') for e in ALL_EFFECTS]
    ax_i.set_xticklabels(eff_labels, fontsize=7)
    ax_i.set_yticks(range(len(KEY_FEATS)))
    ax_i.set_yticklabels([FEAT_LABELS[f] for f in KEY_FEATS], fontsize=9)
    ax_i.set_title(f'{panels[8]}: Partial \u03b7\u00b2 (ART ANOVA)',
                   fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_i, shrink=0.7, label='Partial \u03b7\u00b2')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved dashboard: {output_path}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ART Interaction Effects Analysis')
    parser.add_argument('--score-threshold', type=float, default=0.75,
                        help='Minimum mk_score to include (default: 0.75)')
    parser.add_argument('--detections-full', type=Path, default=None,
                        help='Override full detections JSON (with features)')
    parser.add_argument('--detections-bone', type=Path, default=None,
                        help='Override bone-assigned detections JSON')
    parser.add_argument('--no-exclude', action='store_true',
                        help='Include all slides (no exclusions)')
    args = parser.parse_args()

    # Override globals if CLI args provided
    global DETECTIONS_FULL, DETECTIONS_BONE, EXCLUDE_SLIDES
    if args.detections_full:
        DETECTIONS_FULL = args.detections_full
    if args.detections_bone:
        DETECTIONS_BONE = args.detections_bone
    if args.no_exclude:
        EXCLUDE_SLIDES = set()

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

    fig_dashboard = OUTPUT_DIR / f'mk_{tag}_dashboard.png'
    generate_summary_dashboard(agg_df, features, anova_results, results_df, fig_dashboard)

    print(f"\nDone.")
    print(f"  ANOVA table:       {csv_path}")
    print(f"  Interaction plots: {fig_interactions}")
    print(f"  Summary heatmap:   {fig_summary}")
    print(f"  Dashboard:         {fig_dashboard}")


if __name__ == '__main__':
    main()
