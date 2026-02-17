#!/usr/bin/env python3
"""Early/Late MK Maturation Analysis — Publication-quality plot series.

Splits 1958 MKs into Early (bottom 33%) and Late (top 33%) based on
slide-corrected spectral pseudotime, dropping the middle third.

Outputs (saved to OUTPUT_DIR):
  1. early_late_proportions.png  — Stacked bar chart (3 panels: 4-group, sex, condition)
  2. representative_gallery.png  — Per-group early vs late cell images
  3. morph_comparison.png        — Box plots for morphological features
  4. tsne_early_late.png         — t-SNE colored by stage and group
  5. area_by_group_stage.png     — Area distributions per group/stage
  6. pseudotime_distribution.png — Pseudotime density by group
  7. early_late_stats.json       — All statistics
"""

import base64
import json
import os
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from scipy.stats import chi2_contingency, fisher_exact, gaussian_kde, mannwhitneyu

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = '/viper/ptmp2/edrod/maturation_analysis_v2'
OUTPUT_DIR = os.path.join(DATA_DIR, 'early_late_analysis')

# ─── Constants ────────────────────────────────────────────────────────────────
GROUP_ORDER = ['FGC', 'FHU', 'MGC', 'MHU']
GROUP_COLORS = {'FGC': '#d62728', 'FHU': '#9467bd', 'MGC': '#1f77b4', 'MHU': '#2ca02c'}
STAGE_COLORS = {'Early': '#4e79a7', 'Mid': '#999999', 'Late': '#e15759'}
SEX_MAP = {'F': ['FGC', 'FHU'], 'M': ['MGC', 'MHU']}
COND_MAP = {'GC': ['FGC', 'MGC'], 'HU': ['FHU', 'MHU']}


def load_data():
    """Load all required data files."""
    print('Loading data...')
    data = {}
    with np.load(os.path.join(DATA_DIR, 'maturation_data.npz'), allow_pickle=True) as mat:
        data['groups'] = np.array(mat['groups'])
        data['slides'] = np.array(mat['slides'])
        data['area_um2'] = np.array(mat['area_um2'])
    with np.load(os.path.join(DATA_DIR, 'nuclear_deep_features.npz'), allow_pickle=True) as nuc:
        data['morph_features'] = np.array(nuc['morph_features'])
        data['morph_feature_names'] = list(nuc['morph_feature_names'])
    with np.load(os.path.join(DATA_DIR, 'pseudotime_slide_corrected.npz')) as corr:
        data['pseudotime'] = np.array(corr['pseudotime_corrected'])
    with np.load(os.path.join(DATA_DIR, 'maturation_clusters.npz'), allow_pickle=True) as clust:
        data['X_tsne'] = np.array(clust['X_tsne'])

    print(f'  N={len(data["pseudotime"])} MKs loaded')
    return data


def load_crops():
    """Load crop and mask images (lazy — only when needed)."""
    print('Loading crops JSON...')
    with open(os.path.join(DATA_DIR, 'maturation_data_crops.json'), 'r') as f:
        crops = json.load(f)
    print(f'  {len(crops["crop_b64"])} crops loaded')
    return crops


def compute_splits(pseudotime, pct_lo=20, pct_hi=80):
    """Compute early/mid/late splits at given percentiles."""
    t_lo = np.percentile(pseudotime, pct_lo)
    t_hi = np.percentile(pseudotime, pct_hi)
    is_early = pseudotime < t_lo
    is_late = pseudotime > t_hi
    is_mid = ~is_early & ~is_late
    return is_early, is_mid, is_late, t_lo, t_hi


def sig_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def fmt_pval(p):
    """Format p-value for display, handling underflow."""
    if p == 0.0:
        return 'p<1e-300'
    elif p < 0.001:
        return f'p={p:.2e}'
    else:
        return f'p={p:.3f}'


def decode_crop(b64_str, mask_b64_str=None):
    """Decode a base64 image crop. Optionally apply mask (white background)."""
    img = np.array(Image.open(BytesIO(base64.b64decode(b64_str))))
    if mask_b64_str is not None:
        mask = np.array(Image.open(BytesIO(base64.b64decode(mask_b64_str))))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Resize mask to match crop if needed
        if mask.shape[:2] != img.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST))
        # Apply mask: background = white
        if img.ndim == 3:
            img[mask == 0] = [255, 255, 255]
        else:
            img[mask == 0] = 255
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Early/Late Proportions (stacked bar chart, 3 panels)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_proportions(data, is_early, is_late, stats):
    """Stacked bar chart: 4-group, sex, condition."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    groups = data['groups']

    def _stacked_bar(ax, labels, table_arr, title_str):
        fracs = table_arr / table_arr.sum(axis=1, keepdims=True)
        x = np.arange(len(labels))
        w = 0.6 if len(labels) > 2 else 0.5
        ax.bar(x, fracs[:, 0], label='Early', color=STAGE_COLORS['Early'], width=w)
        ax.bar(x, fracs[:, 1], bottom=fracs[:, 0], label='Late',
               color=STAGE_COLORS['Late'], width=w)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Proportion')
        ax.set_title(title_str, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.18)
        for i in range(len(labels)):
            n = int(table_arr[i].sum())
            # Early % inside bar
            early_pct = fracs[i, 0] * 100
            late_pct = fracs[i, 1] * 100
            ax.text(i, fracs[i, 0] / 2, f'{early_pct:.0f}%', ha='center',
                    va='center', fontsize=10, fontweight='bold', color='white')
            # Late % inside bar
            ax.text(i, fracs[i, 0] + fracs[i, 1] / 2, f'{late_pct:.0f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white')
            ax.text(i, 1.03, f'n={n}', ha='center', fontsize=8, color='#444')

    # Panel A: 4 groups
    table_4g = []
    for g in GROUP_ORDER:
        gm = groups == g
        table_4g.append([(gm & is_early).sum(), (gm & is_late).sum()])
    table_4g = np.array(table_4g)
    chi2_4g, p_4g, dof_4g, _ = chi2_contingency(table_4g)
    _stacked_bar(axes[0], GROUP_ORDER, table_4g,
                 f'4-group (χ²={chi2_4g:.1f}, p={p_4g:.1e})')

    # Panel B: Sex
    sex_table = []
    sex_labels = []
    for sex, grps in [('Female', SEX_MAP['F']), ('Male', SEX_MAP['M'])]:
        sm = np.isin(groups, grps)
        sex_table.append([(sm & is_early).sum(), (sm & is_late).sum()])
        sex_labels.append(sex)
    sex_table = np.array(sex_table)
    chi2_sex, p_sex, _, _ = chi2_contingency(sex_table)
    OR_sex, p_fisher_sex = fisher_exact(sex_table)
    _stacked_bar(axes[1], sex_labels, sex_table,
                 f'Sex (OR={OR_sex:.2f}, p={p_fisher_sex:.2e})')

    # Panel C: Condition
    cond_table = []
    cond_labels = []
    for cond, grps in [('GC', COND_MAP['GC']), ('HU', COND_MAP['HU'])]:
        cm = np.isin(groups, grps)
        cond_table.append([(cm & is_early).sum(), (cm & is_late).sum()])
        cond_labels.append(cond)
    cond_table = np.array(cond_table)
    chi2_cond, p_cond, _, _ = chi2_contingency(cond_table)
    OR_cond, p_fisher_cond = fisher_exact(cond_table)
    _stacked_bar(axes[2], cond_labels, cond_table,
                 f'Condition (OR={OR_cond:.2f}, p={p_fisher_cond:.2e})')

    # Within-sex condition tests (interaction: sex × condition)
    within_sex_cond = {}
    for sex_label, gc_grp, hu_grp in [('Female', 'FGC', 'FHU'), ('Male', 'MGC', 'MHU')]:
        gc_m = groups == gc_grp
        hu_m = groups == hu_grp
        tbl = np.array([
            [(gc_m & is_early).sum(), (gc_m & is_late).sum()],
            [(hu_m & is_early).sum(), (hu_m & is_late).sum()],
        ])
        OR_ws, p_ws = fisher_exact(tbl)
        within_sex_cond[sex_label] = {
            'gc_group': gc_grp, 'hu_group': hu_grp,
            'table': tbl.tolist(),
            'OR': float(OR_ws), 'p_fisher': float(p_ws),
        }
        print(f'    Within {sex_label}: {gc_grp} vs {hu_grp} — '
              f'OR={OR_ws:.2f}, p={p_ws:.3f}')

    # Annotate 4-group panel with within-sex brackets
    for sex_label, (i_gc, i_hu) in [('F', (0, 1)), ('M', (2, 3))]:
        info = within_sex_cond['Female' if sex_label == 'F' else 'Male']
        p_ws = info['p_fisher']
        stars = sig_stars(p_ws)
        y_bracket = 1.07
        axes[0].plot([i_gc, i_gc, i_hu, i_hu],
                     [y_bracket, y_bracket + 0.02, y_bracket + 0.02, y_bracket],
                     color='black', linewidth=0.8, clip_on=False)
        axes[0].text((i_gc + i_hu) / 2, y_bracket + 0.025,
                     f'{stars}' if stars != 'ns' else 'ns',
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     clip_on=False)

    plt.suptitle('Early vs Late MK Proportions (slide-corrected pseudotime)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'early_late_proportions.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')

    # Store stats
    stats['four_group'] = {
        'chi2': float(chi2_4g), 'p': float(p_4g), 'dof': int(dof_4g),
        'per_group': {}
    }
    for i, g in enumerate(GROUP_ORDER):
        n_e, n_l = int(table_4g[i, 0]), int(table_4g[i, 1])
        stats['four_group']['per_group'][g] = {
            'N_early': n_e, 'N_late': n_l,
            'pct_early': round(n_e / (n_e + n_l) * 100, 1),
            'pct_late': round(n_l / (n_e + n_l) * 100, 1),
        }
    stats['sex'] = {
        'chi2': float(chi2_sex), 'p': float(p_sex),
        'OR': float(OR_sex), 'p_fisher': float(p_fisher_sex),
        'Female': {'N_early': int(sex_table[0, 0]), 'N_late': int(sex_table[0, 1])},
        'Male': {'N_early': int(sex_table[1, 0]), 'N_late': int(sex_table[1, 1])},
    }
    stats['condition'] = {
        'chi2': float(chi2_cond), 'p': float(p_cond),
        'OR': float(OR_cond), 'p_fisher': float(p_fisher_cond),
        'GC': {'N_early': int(cond_table[0, 0]), 'N_late': int(cond_table[0, 1])},
        'HU': {'N_early': int(cond_table[1, 0]), 'N_late': int(cond_table[1, 1])},
    }
    stats['within_sex_condition'] = {
        sex: {'OR': v['OR'], 'p_fisher': v['p_fisher'],
              f"{v['gc_group']}_early": v['table'][0][0],
              f"{v['gc_group']}_late": v['table'][0][1],
              f"{v['hu_group']}_early": v['table'][1][0],
              f"{v['hu_group']}_late": v['table'][1][1]}
        for sex, v in within_sex_cond.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Representative Gallery
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gallery(data, is_early, is_late, crops):
    """8 rows × 6 cols: per-group early/late representative images."""
    groups = data['groups']
    slides = data['slides']
    areas = data['area_um2']
    ncols = 6
    nrows = len(GROUP_ORDER) * 2  # early + late per group
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))

    for gi, g in enumerate(GROUP_ORDER):
        gm = groups == g
        for si, (stage, stage_mask) in enumerate([('Early', is_early), ('Late', is_late)]):
            row = gi * 2 + si
            sel = np.where(gm & stage_mask)[0]
            # Sort by area so gallery shows range from small to large
            sel = sel[np.argsort(areas[sel])]
            # Pick ncols evenly spaced indices across the size range
            if len(sel) >= ncols:
                pick_idx = np.linspace(0, len(sel) - 1, ncols, dtype=int)
                picked = sel[pick_idx]
            else:
                picked = sel

            for ci in range(ncols):
                ax = axes[row, ci]
                ax.set_xticks([])
                ax.set_yticks([])
                if ci < len(picked):
                    idx = picked[ci]
                    img = decode_crop(crops['crop_b64'][idx], crops['mask_b64'][idx])
                    ax.imshow(img)
                    # Title: area + slide short name
                    slide_short = slides[idx].split('_')[-1]  # e.g. "FGC1"
                    ax.set_title(f'{areas[idx]:.0f} µm²\n{slide_short}',
                                 fontsize=7, pad=2)
                else:
                    ax.axis('off')

                # Left border strip for group color
                if ci == 0:
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.spines['left'].set_visible(True)
                    ax.spines['left'].set_color(GROUP_COLORS[g])
                    ax.spines['left'].set_linewidth(5)
                else:
                    for spine in ax.spines.values():
                        spine.set_visible(False)

            # Row label
            axes[row, 0].set_ylabel(f'{g} {stage}', fontsize=9, fontweight='bold',
                                    rotation=0, labelpad=55, va='center')

    plt.suptitle('Representative MK Gallery — Early vs Late per Group',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'representative_gallery.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Morphological Feature Comparison (box plots)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_morph_comparison(data, is_early, is_late, stats):
    """2×2 box plots for area_um2, nc_ratio, circularity, solidity."""
    morph = data['morph_features']
    morph_names = data['morph_feature_names']
    areas = data['area_um2']

    # Features to plot: area_um2 (from maturation_data), then nc_ratio, circularity, solidity from morph
    features_to_plot = []
    labels = []

    # area_um2
    features_to_plot.append(areas)
    labels.append('Area (µm²)')

    # nc_ratio, circularity, solidity from morph_features
    for name in ['nc_ratio', 'circularity', 'solidity']:
        if name in morph_names:
            idx = morph_names.index(name)
            features_to_plot.append(morph[:, idx])
            nice = name.replace('_', ' ').title()
            labels.append(nice)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    stats['morph_comparison'] = {}

    for i, (feat, label) in enumerate(zip(features_to_plot, labels)):
        ax = axes[i]
        early_vals = feat[is_early]
        late_vals = feat[is_late]
        U, p = mannwhitneyu(early_vals, late_vals, alternative='two-sided')

        bp = ax.boxplot(
            [early_vals, late_vals],
            tick_labels=['Early', 'Late'],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color='black', linewidth=1.5),
        )
        bp['boxes'][0].set_facecolor(STAGE_COLORS['Early'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(STAGE_COLORS['Late'])
        bp['boxes'][1].set_alpha(0.7)

        stars = sig_stars(p)
        ax.set_title(f'{label}\n{stars} ({fmt_pval(p)})', fontsize=11)
        ax.set_ylabel(label)

        stats['morph_comparison'][label] = {
            'early_median': float(np.median(early_vals)),
            'late_median': float(np.median(late_vals)),
            'early_mean': float(np.mean(early_vals)),
            'late_mean': float(np.mean(late_vals)),
            'U': float(U),
            'p': float(p),
        }

    plt.suptitle('Morphological Features: Early vs Late MKs',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'morph_comparison.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: t-SNE colored by stage and group
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tsne(data, is_early, is_mid, is_late):
    """Side-by-side t-SNE: left = Early/Mid/Late, right = Group colors."""
    X_tsne = data['X_tsne']
    groups = data['groups']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by stage
    # Plot mid first (background), then early/late on top
    for stage, mask, color, zorder in [
        ('Mid', is_mid, STAGE_COLORS['Mid'], 1),
        ('Early', is_early, STAGE_COLORS['Early'], 2),
        ('Late', is_late, STAGE_COLORS['Late'], 2),
    ]:
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color,
                    s=8, alpha=0.5, label=f'{stage} (n={mask.sum()})',
                    zorder=zorder, edgecolors='none')
    ax1.set_title('t-SNE by Maturation Stage', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, markerscale=2)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')

    # Right: colored by group
    for g in GROUP_ORDER:
        gm = groups == g
        ax2.scatter(X_tsne[gm, 0], X_tsne[gm, 1], c=GROUP_COLORS[g],
                    s=8, alpha=0.5, label=f'{g} (n={gm.sum()})',
                    edgecolors='none')
    ax2.set_title('t-SNE by Group', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, markerscale=2)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')

    plt.suptitle('t-SNE Embedding of Nuclear Deep Features',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'tsne_early_late.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: Area by Group and Stage (violin/box)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_area_by_group(data, is_early, is_late, stats):
    """Grouped box plots: area within each group, early vs late side by side."""
    groups = data['groups']
    areas = data['area_um2']

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    box_data = []
    colors_list = []
    xtick_positions = []
    xtick_labels = []

    stats['area_by_group'] = {}
    offset = 0
    for gi, g in enumerate(GROUP_ORDER):
        gm = groups == g
        early_vals = areas[gm & is_early]
        late_vals = areas[gm & is_late]

        pos_e = offset
        pos_l = offset + 0.4
        positions.extend([pos_e, pos_l])
        box_data.extend([early_vals, late_vals])
        colors_list.extend([STAGE_COLORS['Early'], STAGE_COLORS['Late']])
        xtick_positions.append(offset + 0.2)
        xtick_labels.append(g)

        U, p = mannwhitneyu(early_vals, late_vals, alternative='two-sided')
        stars = sig_stars(p)

        # Draw significance bracket
        ymax = max(np.percentile(early_vals, 95), np.percentile(late_vals, 95))
        bracket_y = ymax * 1.05
        ax.plot([pos_e, pos_e, pos_l, pos_l],
                [bracket_y, bracket_y * 1.02, bracket_y * 1.02, bracket_y],
                color='black', linewidth=0.8)
        ax.text((pos_e + pos_l) / 2, bracket_y * 1.03, stars,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        stats['area_by_group'][g] = {
            'early_median': float(np.median(early_vals)),
            'late_median': float(np.median(late_vals)),
            'early_n': int(len(early_vals)),
            'late_n': int(len(late_vals)),
            'U': float(U), 'p': float(p),
        }

        offset += 1.2

    bp = ax.boxplot(box_data, positions=positions, widths=0.35,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_ylabel('Area (µm²)', fontsize=11)
    ax.set_title('Cell Area by Group and Maturation Stage', fontsize=13,
                 fontweight='bold')

    # Custom legend
    legend_elements = [
        Patch(facecolor=STAGE_COLORS['Early'], alpha=0.7, label='Early'),
        Patch(facecolor=STAGE_COLORS['Late'], alpha=0.7, label='Late'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'area_by_group_stage.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6: Pseudotime Distribution by Group
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pseudotime_dist(data, t33, t67):
    """Pseudotime density histograms per group with tertile shading."""
    pt = data['pseudotime']
    groups = data['groups']

    fig, ax = plt.subplots(figsize=(10, 5))

    x_grid = np.linspace(pt.min() - 0.01, pt.max() + 0.01, 300)

    for g in GROUP_ORDER:
        gm = groups == g
        kde = gaussian_kde(pt[gm], bw_method='scott')
        density = kde(x_grid)
        ax.plot(x_grid, density, linewidth=2, color=GROUP_COLORS[g],
                label=f'{g} (n={gm.sum()})')
        ax.fill_between(x_grid, density, alpha=0.08, color=GROUP_COLORS[g])

    # Shaded regions
    ax.axvspan(pt.min(), t33, alpha=0.08, color=STAGE_COLORS['Early'], zorder=0)
    ax.axvspan(t67, pt.max(), alpha=0.08, color=STAGE_COLORS['Late'], zorder=0)

    # Vertical lines at tertiles
    ax.axvline(t33, color='gray', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(t67, color='gray', linestyle='--', linewidth=1, alpha=0.8)

    # Freeze ylim after all drawing, then add text labels
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
    ax.text(t33, ylim[1] * 0.95, '  33rd %ile', fontsize=8, color='gray',
            va='top')
    ax.text(t67, ylim[1] * 0.95, '  67th %ile', fontsize=8, color='gray',
            va='top')

    # Region labels
    mid_early = (pt.min() + t33) / 2
    mid_late = (t67 + pt.max()) / 2
    ax.text(mid_early, ylim[1] * 0.85, 'EARLY', ha='center', fontsize=10,
            color=STAGE_COLORS['Early'], fontweight='bold', alpha=0.6)
    ax.text(mid_late, ylim[1] * 0.85, 'LATE', ha='center', fontsize=10,
            color=STAGE_COLORS['Late'], fontweight='bold', alpha=0.6)

    ax.set_xlabel('Pseudotime (slide-corrected)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Pseudotime Distribution by Group', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'pseudotime_distribution.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Output directory: {OUTPUT_DIR}\n')

    # Load data
    data = load_data()
    pt = data['pseudotime']
    is_early, is_mid, is_late, t33, t67 = compute_splits(pt)

    N = len(pt)
    print(f'\nSplit: N={N}, Early={is_early.sum()}, Mid={is_mid.sum()}, '
          f'Late={is_late.sum()}\n')

    stats = {
        'N_total': int(N),
        'N_early': int(is_early.sum()),
        'N_mid': int(is_mid.sum()),
        'N_late': int(is_late.sum()),
        'tertile_33': float(t33),
        'tertile_67': float(t67),
    }

    # Plot 1: Proportions
    print('Plot 1: Early/Late proportions...')
    plot_proportions(data, is_early, is_late, stats)

    # Plot 2: Representative gallery (needs crops)
    print('Plot 2: Representative gallery...')
    crops = load_crops()
    plot_gallery(data, is_early, is_late, crops)
    del crops  # free memory

    # Plot 3: Morphological comparison
    print('Plot 3: Morphological comparison...')
    plot_morph_comparison(data, is_early, is_late, stats)

    # Plot 4: t-SNE
    print('Plot 4: t-SNE early/late...')
    plot_tsne(data, is_early, is_mid, is_late)

    # Plot 5: Area by group and stage
    print('Plot 5: Area by group/stage...')
    plot_area_by_group(data, is_early, is_late, stats)

    # Plot 6: Pseudotime distribution
    print('Plot 6: Pseudotime distribution...')
    plot_pseudotime_dist(data, t33, t67)

    # Save stats JSON
    stats_path = os.path.join(OUTPUT_DIR, 'early_late_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\n  Saved {stats_path}')

    print(f'\nDone! All outputs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
