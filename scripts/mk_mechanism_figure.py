#!/usr/bin/env python3
"""Generate MK mechanosensing figure with exaggerated but data-faithful cells.

Cell dimensions computed from real measurements. Differences between GC and HU
are amplified 3× for visual clarity (noted on figure). GC outline overlaid on
HU cell for direct comparison. Density bars to scale.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np
from pathlib import Path

OUTPUT = Path('/Volumes/pool-mann-edwin/bm_lmd_feb2026/mk_clf084_dataset/mk_mechanism_figure.png')

# ── Actual data (median of slide medians, score >= 0.80) ────────────
DATA = {
    ('F', 'GC', 'femur'):    {'area': 430, 'elong': 0.196, 'density': 2.11},
    ('F', 'HU', 'femur'):    {'area': 586, 'elong': 0.224, 'density': 3.59},
    ('F', 'GC', 'humerus'):  {'area': 454, 'elong': 0.184, 'density': 2.27},
    ('F', 'HU', 'humerus'):  {'area': 549, 'elong': 0.225, 'density': 1.93},
    ('M', 'GC', 'femur'):    {'area': 556, 'elong': 0.238, 'density': 2.33},
    ('M', 'HU', 'femur'):    {'area': 579, 'elong': 0.177, 'density': 1.10},
    ('M', 'GC', 'humerus'):  {'area': 550, 'elong': 0.257, 'density': 2.43},
    ('M', 'HU', 'humerus'):  {'area': 592, 'elong': 0.178, 'density': 1.71},
}

EXAGGERATION = 3.0  # Amplify GC→HU differences by this factor

# ── Colors ──────────────────────────────────────────────────────────
COL_F = '#D64550'
COL_M = '#4477AA'
COL_F_FILL = '#F0A0A8'
COL_M_FILL = '#8BB8DE'
COL_NUC = '#2D1B4E'
COL_GC = '#888888'
COL_LOCAL = '#8B6914'
COL_SYSTEMIC = '#6A5ACD'
COL_BONE_F = '#E8D5B7'
COL_BONE_H = '#D5C4A1'


def cell_axes_um(area_um2, elongation):
    """Semi-major and semi-minor axes (µm) from area and elongation."""
    minor_frac = 1.0 - elongation
    a = np.sqrt(area_um2 / (np.pi * minor_frac))
    b = a * minor_frac
    return a, b


def exaggerate(gc_val, hu_val, factor):
    """Amplify the HU-GC difference by factor, centered on GC."""
    delta = hu_val - gc_val
    return gc_val + delta * factor


def draw_cell(ax, cx, cy, a_um, b_um, um_scale, fill, edge, lw=2.5,
              alpha=0.8, linestyle='-', zorder=3):
    """Draw one MK cell ellipse, return (w, h) in figure coords."""
    w = a_um * 2 * um_scale
    h = b_um * 2 * um_scale
    ell = Ellipse((cx, cy), w, h,
                  facecolor=fill, edgecolor=edge,
                  linewidth=lw, alpha=alpha, linestyle=linestyle, zorder=zorder)
    ax.add_patch(ell)
    return w, h


def main():
    fig = plt.figure(figsize=(16, 18))
    rng = np.random.default_rng(42)

    # Compute exaggerated HU values for visualization
    EXAG_DATA = {}
    for key, d in DATA.items():
        EXAG_DATA[key] = dict(d)
    for sex in ['F', 'M']:
        for bone in ['femur', 'humerus']:
            gc = DATA[(sex, 'GC', bone)]
            hu = DATA[(sex, 'HU', bone)]
            EXAG_DATA[(sex, 'HU', bone)] = {
                'area': exaggerate(gc['area'], hu['area'], EXAGGERATION),
                'elong': exaggerate(gc['elong'], hu['elong'], EXAGGERATION),
                'density': hu['density'],  # density bars stay true
            }

    # Scale: largest exaggerated cell determines figure scale
    max_a = 0
    for key, d in EXAG_DATA.items():
        elong = max(0.01, min(0.99, d['elong']))  # clamp exaggerated elongation
        a, _ = cell_axes_um(max(d['area'], 100), elong)
        max_a = max(max_a, a)
    UM_SCALE = 0.07 / (max_a * 2)

    # ═══════════════════════════════════════════════════════════════════
    # TOP: Cell comparison panels
    # ═══════════════════════════════════════════════════════════════════
    ax = fig.add_axes([0.02, 0.34, 0.96, 0.63])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.99, 'MK Morphology & Density Under Hindlimb Unloading',
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.96,
            f'Shape differences amplified {EXAGGERATION:.0f}× for visibility. '
            f'Density bars and annotations show true values.',
            ha='center', va='top', fontsize=10, color='#666')

    bones = ['femur', 'humerus']
    sexes = ['F', 'M']
    bone_block_cx = [0.27, 0.73]
    row_cy = [0.66, 0.28]

    bone_labels = {
        'femur': 'FEMUR  (hindlimb — unloaded)',
        'humerus': 'HUMERUS  (forelimb — overloaded)',
    }
    for ci, bone in enumerate(bones):
        ax.text(bone_block_cx[ci], 0.925, bone_labels[bone],
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='#5A4A2A',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=COL_BONE_F if bone == 'femur' else COL_BONE_H,
                          edgecolor='#8B7355', linewidth=1.5))

    for ri, sex in enumerate(sexes):
        color = COL_F if sex == 'F' else COL_M
        ax.text(0.03, row_cy[ri], 'FEMALE' if sex == 'F' else 'MALE',
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=color, rotation=90)

    max_density = max(d['density'] for d in DATA.values())

    for ri, sex in enumerate(sexes):
        for ci, bone in enumerate(bones):
            bcx = bone_block_cx[ci]
            cy = row_cy[ri]
            sex_fill = COL_F_FILL if sex == 'F' else COL_M_FILL
            sex_col = COL_F if sex == 'F' else COL_M

            gc_real = DATA[(sex, 'GC', bone)]
            hu_real = DATA[(sex, 'HU', bone)]
            hu_exag = EXAG_DATA[(sex, 'HU', bone)]

            # Panel background
            panel_w, panel_h = 0.38, 0.28
            bg = FancyBboxPatch(
                (bcx - panel_w/2, cy - panel_h/2), panel_w, panel_h,
                boxstyle='round,pad=0.008', facecolor='#FAFAFA',
                edgecolor='#DDD', linewidth=1, zorder=0)
            ax.add_patch(bg)

            # ── Center: HU cell (exaggerated, solid colored) ─────────
            cell_cx = bcx - 0.02
            hu_elong_clamped = max(0.01, min(0.95, hu_exag['elong']))
            hu_a, hu_b = cell_axes_um(max(hu_exag['area'], 50), hu_elong_clamped)
            hu_w, hu_h = draw_cell(ax, cell_cx, cy, hu_a, hu_b, UM_SCALE,
                                   sex_fill, sex_col, lw=2.5, alpha=0.7, zorder=3)

            # ── GC outline overlaid (dashed gray) ────────────────────
            gc_a, gc_b = cell_axes_um(gc_real['area'], gc_real['elong'])
            draw_cell(ax, cell_cx, cy, gc_a, gc_b, UM_SCALE,
                      'none', COL_GC, lw=2, alpha=0.9, linestyle='--', zorder=4)

            # Dimension annotations on cell
            hu_w_real = cell_axes_um(hu_real['area'], hu_real['elong'])[0] * 2
            gc_w_real = gc_a * 2
            ax.text(cell_cx, cy + max(hu_h, gc_b * 2 * UM_SCALE)/2 + 0.015,
                    f'{gc_w_real:.0f} → {hu_w_real:.0f} µm',
                    ha='center', va='bottom', fontsize=7.5, color='#555',
                    fontstyle='italic')

            # ── Density bars (right side) ────────────────────────────
            bar_x = bcx + 0.145
            bar_max_h = 0.20
            bar_w = 0.022

            gc_bar_h = (gc_real['density'] / max_density) * bar_max_h
            gc_bar = FancyBboxPatch(
                (bar_x - bar_w - 0.004, cy - bar_max_h/2),
                bar_w, gc_bar_h,
                boxstyle='round,pad=0.002', facecolor=COL_GC,
                edgecolor='#888', linewidth=1, alpha=0.5)
            ax.add_patch(gc_bar)
            ax.text(bar_x - bar_w/2 - 0.004, cy - bar_max_h/2 - 0.018,
                    f'{gc_real["density"]:.1f}', ha='center', va='top',
                    fontsize=7.5, color=COL_GC, fontweight='bold')

            hu_bar_h = (hu_real['density'] / max_density) * bar_max_h
            hu_bar = FancyBboxPatch(
                (bar_x + 0.004, cy - bar_max_h/2),
                bar_w, hu_bar_h,
                boxstyle='round,pad=0.002', facecolor=sex_col,
                edgecolor=sex_col, linewidth=1, alpha=0.5)
            ax.add_patch(hu_bar)
            ax.text(bar_x + bar_w/2 + 0.004, cy - bar_max_h/2 - 0.018,
                    f'{hu_real["density"]:.1f}', ha='center', va='top',
                    fontsize=7.5, color=sex_col, fontweight='bold')

            ax.text(bar_x + 0.01, cy + bar_max_h/2 + 0.008, 'MK/mm²',
                    ha='center', va='bottom', fontsize=7, color='#888')

            # ── Delta annotations (top of panel) ─────────────────────
            area_pct = (hu_real['area'] - gc_real['area']) / gc_real['area'] * 100
            elong_pct = (hu_real['elong'] - gc_real['elong']) / gc_real['elong'] * 100
            dens_pct = (hu_real['density'] - gc_real['density']) / gc_real['density'] * 100

            y_top = cy + panel_h/2 - 0.01
            ax.text(bcx - 0.02, y_top,
                    f'area {area_pct:+.0f}%   elong {elong_pct:+.0f}%',
                    ha='center', va='top', fontsize=9.5,
                    fontfamily='monospace', fontweight='bold', color=COL_SYSTEMIC,
                    bbox=dict(boxstyle='round,pad=0.06', facecolor='#F0EDFF',
                              edgecolor=COL_SYSTEMIC, linewidth=1, alpha=0.9))

            ax.text(bcx - 0.02, y_top - 0.038,
                    f'density {dens_pct:+.0f}%',
                    ha='center', va='top', fontsize=9.5,
                    fontfamily='monospace', fontweight='bold', color=COL_LOCAL,
                    bbox=dict(boxstyle='round,pad=0.06', facecolor='#FFF8E8',
                              edgecolor=COL_LOCAL, linewidth=1, alpha=0.9))

    # Legend
    ax.add_patch(Ellipse((0.09, 0.065), 0.018, 0.014,
                         facecolor='none', edgecolor=COL_GC,
                         linewidth=2, linestyle='--'))
    ax.text(0.115, 0.065, 'GC baseline (true scale)', va='center',
            fontsize=9, color=COL_GC)
    ax.add_patch(Ellipse((0.09, 0.035), 0.018, 0.014,
                         facecolor=COL_F_FILL, edgecolor=COL_F,
                         linewidth=2, alpha=0.7))
    ax.text(0.115, 0.035, f'HU (differences ×{EXAGGERATION:.0f})', va='center',
            fontsize=9, color='#333')

    ax.text(0.42, 0.065, '■', fontsize=10, color=COL_SYSTEMIC, va='center')
    ax.text(0.44, 0.065, 'Systemic (parallel in both bones)',
            fontsize=9, color=COL_SYSTEMIC, va='center')
    ax.text(0.42, 0.035, '■', fontsize=10, color=COL_LOCAL, va='center')
    ax.text(0.44, 0.035, 'Local mechanical (bone-specific)',
            fontsize=9, color=COL_LOCAL, va='center')

    # Scale bar (true µm, using GC cell scale)
    scale_um = 15
    sb_len = scale_um * UM_SCALE
    # Show scale bar is for GC outline (true scale)
    sb_x = 0.85
    sb_y = 0.05
    ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color='black', linewidth=3)
    ax.plot([sb_x, sb_x], [sb_y - 0.006, sb_y + 0.006], color='black', linewidth=2)
    ax.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - 0.006, sb_y + 0.006],
            color='black', linewidth=2)
    ax.text(sb_x + sb_len/2, sb_y - 0.015, f'{scale_um} µm (GC scale)',
            ha='center', va='top', fontsize=8, fontweight='bold')

    # ═══════════════════════════════════════════════════════════════════
    # BOTTOM: Mechanism diagram
    # ═══════════════════════════════════════════════════════════════════
    ax_m = fig.add_axes([0.03, 0.01, 0.94, 0.30])
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)
    ax_m.axis('off')

    ax_m.text(0.5, 0.97, 'Two Dissociable Mechanosensing Pathways',
              ha='center', va='top', fontsize=14, fontweight='bold')

    ax_m.plot([0.50, 0.50], [0.00, 0.90], color='#CCC', linewidth=1.5,
              linestyle=':', zorder=0)

    # ── LEFT: Local mechanical ───────────────────────────────────────
    lx = 0.25
    ax_m.text(lx, 0.83, 'LOCAL MECHANICAL',
              ha='center', fontsize=13, fontweight='bold', color=COL_LOCAL,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E8',
                        edgecolor=COL_LOCAL, linewidth=2))
    ax_m.text(lx, 0.72, 'Controls: MK DENSITY',
              ha='center', fontsize=11, fontweight='bold', color=COL_LOCAL)

    evidence_local = (
        "Density diverges between bones:\n"
        "  ♀  femur +70%   humerus −15%\n"
        "  ♂  femur −53%   humerus −30%\n\n"
        "Magnitude tracks perturbation:\n"
        "  complete unload >> partial overload\n\n"
        "Treatment × Bone   p = 0.09"
    )
    ax_m.text(lx, 0.40, evidence_local,
              ha='center', va='center', fontsize=9.5,
              fontfamily='monospace', color='#333', linespacing=1.3,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFAE6',
                        edgecolor=COL_LOCAL, linewidth=1.5))

    for bx, blabel, acol, alabel, ay1, ay2 in [
        (lx - 0.10, 'Femur', '#CC0000', '−load', 0.05, -0.01),
        (lx + 0.10, 'Humerus', '#006600', '+load', 0.13, 0.19),
    ]:
        box = FancyBboxPatch((bx - 0.06, 0.06), 0.12, 0.06,
                              boxstyle='round,pad=0.01',
                              facecolor=COL_BONE_F if 'Fem' in blabel else COL_BONE_H,
                              edgecolor=COL_LOCAL, linewidth=1.5)
        ax_m.add_patch(box)
        ax_m.text(bx, 0.09, blabel, ha='center', va='center',
                  fontsize=8, fontweight='bold', color=COL_LOCAL)
        ax_m.annotate('', xy=(bx, ay1), xytext=(bx, ay2),
                      arrowprops=dict(arrowstyle='->', color=acol,
                                      linewidth=2.5, mutation_scale=15))
        ty = ay2 - 0.025 if ay2 < ay1 else ay2 + 0.015
        ax_m.text(bx, ty, alabel, ha='center', fontsize=8,
                  fontweight='bold', color=acol)

    # ── RIGHT: Systemic / humoral ────────────────────────────────────
    rx = 0.75
    ax_m.text(rx, 0.83, 'SYSTEMIC / HUMORAL',
              ha='center', fontsize=13, fontweight='bold', color=COL_SYSTEMIC,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0EDFF',
                        edgecolor=COL_SYSTEMIC, linewidth=2))
    ax_m.text(rx, 0.72, 'Controls: MK MORPHOLOGY',
              ha='center', fontsize=11, fontweight='bold', color=COL_SYSTEMIC)

    evidence_sys = (
        "Size & shape shift SAME direction\n"
        "in BOTH bones within each sex:\n"
        "  ♀  area ↑36%/↑21%  elong ↑14%/↑23%\n"
        "  ♂  area  ↑4%/ ↑8%  elong ↓25%/↓31%\n\n"
        "Bones under opposite perturbations\n"
        "→ signal must be circulating\n\n"
        "Sex × Treatment   p < 0.02"
    )
    ax_m.text(rx, 0.40, evidence_sys,
              ha='center', va='center', fontsize=9.5,
              fontfamily='monospace', color='#333', linespacing=1.3,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F2FF',
                        edgecolor=COL_SYSTEMIC, linewidth=1.5))

    for bx, blabel in [(rx - 0.10, 'Femur'), (rx + 0.10, 'Humerus')]:
        box = FancyBboxPatch((bx - 0.06, 0.06), 0.12, 0.06,
                              boxstyle='round,pad=0.01',
                              facecolor=COL_BONE_F if 'Fem' in blabel else COL_BONE_H,
                              edgecolor=COL_SYSTEMIC, linewidth=1.5)
        ax_m.add_patch(box)
        ax_m.text(bx, 0.09, blabel, ha='center', va='center',
                  fontsize=8, fontweight='bold', color=COL_SYSTEMIC)

    ax_m.text(rx, 0.09, '?', ha='center', va='center',
              fontsize=16, fontweight='bold', color=COL_SYSTEMIC,
              bbox=dict(boxstyle='circle,pad=0.12', facecolor='#E8E0FF',
                        edgecolor=COL_SYSTEMIC, linewidth=2))
    for sign in [-1, 1]:
        ax_m.annotate('', xy=(rx + sign * 0.04, 0.09),
                      xytext=(rx + sign * 0.02, 0.09),
                      arrowprops=dict(arrowstyle='->', color=COL_SYSTEMIC,
                                      linewidth=2, mutation_scale=12))
    ax_m.text(rx, -0.01,
              'Circulating factor from unloaded limb?  Sex-hormone modulated receptor?',
              ha='center', fontsize=8.5, fontstyle='italic', color=COL_SYSTEMIC)

    plt.savefig(OUTPUT, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
