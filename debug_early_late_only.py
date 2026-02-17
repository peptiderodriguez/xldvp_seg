#!/usr/bin/env python3
"""Early vs Late only (drop middle third) across groups, slide-corrected."""

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

corr = np.load('/viper/ptmp2/edrod/maturation_analysis_v2/pseudotime_slide_corrected.npz')
mat = np.load('/viper/ptmp2/edrod/maturation_analysis_v2/maturation_data.npz', allow_pickle=True)

pt = corr['pseudotime_corrected']
groups = mat['groups']
slides = mat['slides']
areas = mat['area_um2']

N = len(pt)
group_order = ['FGC', 'FHU', 'MGC', 'MHU']

t33 = np.percentile(pt, 33.3)
t67 = np.percentile(pt, 66.7)

is_early = pt < t33
is_late = pt > t67
use = is_early | is_late  # drop middle

print(f"Total: {N}, Early: {is_early.sum()}, Late: {is_late.sum()}, Dropped mid: {N - use.sum()}")

# --- 4-group ---
print(f"\n{'Group':>5} {'N_used':>7} {'Early':>6} {'Late':>6} {'%Early':>7} {'%Late':>7}")
table = []
for g in group_order:
    gm = groups == g
    n_early = (gm & is_early).sum()
    n_late = (gm & is_late).sum()
    n_used = n_early + n_late
    table.append([n_early, n_late])
    print(f"{g:>5} {n_used:>7} {n_early:>6} {n_late:>6} {n_early/n_used*100:>6.1f}% {n_late/n_used*100:>6.1f}%")

chi2, p_chi2, dof, _ = chi2_contingency(table)
print(f"\n4-group Chi-squared: χ²={chi2:.2f}, dof={dof}, p={p_chi2:.2e}")

# --- Sex ---
print(f"\n{'Sex':>6} {'N_used':>7} {'Early':>6} {'Late':>6} {'%Early':>7}")
sex_table = []
for sex, sex_groups in [('F', ['FGC', 'FHU']), ('M', ['MGC', 'MHU'])]:
    sm = np.isin(groups, sex_groups)
    n_early = (sm & is_early).sum()
    n_late = (sm & is_late).sum()
    sex_table.append([n_early, n_late])
    n_used = n_early + n_late
    print(f"{sex:>6} {n_used:>7} {n_early:>6} {n_late:>6} {n_early/n_used*100:>6.1f}%")
chi2_sex, p_sex, _, _ = chi2_contingency(sex_table)
OR_sex, p_fisher_sex = fisher_exact(sex_table)
print(f"Sex χ²={chi2_sex:.2f}, p={p_sex:.2e} | Fisher OR={OR_sex:.2f}, p={p_fisher_sex:.2e}")

# --- Condition ---
print(f"\n{'Cond':>6} {'N_used':>7} {'Early':>6} {'Late':>6} {'%Early':>7}")
cond_table = []
for cond, cond_groups in [('GC', ['FGC', 'MGC']), ('HU', ['FHU', 'MHU'])]:
    cm = np.isin(groups, cond_groups)
    n_early = (cm & is_early).sum()
    n_late = (cm & is_late).sum()
    cond_table.append([n_early, n_late])
    n_used = n_early + n_late
    print(f"{cond:>6} {n_used:>7} {n_early:>6} {n_late:>6} {n_early/n_used*100:>6.1f}%")
chi2_cond, p_cond, _, _ = chi2_contingency(cond_table)
OR_cond, p_fisher_cond = fisher_exact(cond_table)
print(f"Condition χ²={chi2_cond:.2f}, p={p_cond:.2e} | Fisher OR={OR_cond:.2f}, p={p_fisher_cond:.2e}")

# --- Bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel 1: 4 groups
table_arr = np.array(table)
fracs = table_arr / table_arr.sum(axis=1, keepdims=True)
x = np.arange(len(group_order))
bars1 = axes[0].bar(x, fracs[:, 0], label='Early', color='#4e79a7', width=0.6)
bars2 = axes[0].bar(x, fracs[:, 1], bottom=fracs[:, 0], label='Late', color='#e15759', width=0.6)
axes[0].set_xticks(x)
axes[0].set_xticklabels(group_order)
axes[0].set_ylabel('Proportion')
axes[0].set_title(f'4-group (χ²={chi2:.1f}, p={p_chi2:.2e})')
axes[0].legend()
for i in range(len(group_order)):
    n = table_arr[i].sum()
    axes[0].text(i, 0.5, f'{fracs[i,0]*100:.0f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    axes[0].text(i, 1.02, f'n={n}', ha='center', fontsize=8)

# Panel 2: sex
sex_arr = np.array(sex_table)
sex_fracs = sex_arr / sex_arr.sum(axis=1, keepdims=True)
x2 = np.arange(2)
axes[1].bar(x2, sex_fracs[:, 0], label='Early', color='#4e79a7', width=0.5)
axes[1].bar(x2, sex_fracs[:, 1], bottom=sex_fracs[:, 0], label='Late', color='#e15759', width=0.5)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(['Female', 'Male'])
axes[1].set_title(f'Sex (χ²={chi2_sex:.1f}, p={p_sex:.2e}\nOR={OR_sex:.2f})')
axes[1].legend()
for i in range(2):
    n = sex_arr[i].sum()
    axes[1].text(i, 0.5, f'{sex_fracs[i,0]*100:.0f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    axes[1].text(i, 1.02, f'n={n}', ha='center', fontsize=8)

# Panel 3: condition
cond_arr = np.array(cond_table)
cond_fracs = cond_arr / cond_arr.sum(axis=1, keepdims=True)
x3 = np.arange(2)
axes[2].bar(x3, cond_fracs[:, 0], label='Early', color='#4e79a7', width=0.5)
axes[2].bar(x3, cond_fracs[:, 1], bottom=cond_fracs[:, 0], label='Late', color='#e15759', width=0.5)
axes[2].set_xticks(x3)
axes[2].set_xticklabels(['GC', 'HU'])
axes[2].set_title(f'Condition (χ²={chi2_cond:.1f}, p={p_cond:.2e}\nOR={OR_cond:.2f})')
axes[2].legend()
for i in range(2):
    n = cond_arr[i].sum()
    axes[2].text(i, 0.5, f'{cond_fracs[i,0]*100:.0f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    axes[2].text(i, 1.02, f'n={n}', ha='center', fontsize=8)

for ax in axes:
    ax.set_ylim(0, 1.1)

plt.suptitle('Early vs Late MKs (bottom/top third, middle dropped)\nSlide-corrected pseudotime', fontsize=13)
plt.tight_layout()

out_path = '/viper/ptmp2/edrod/maturation_analysis_v2/early_late_only.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved to {out_path}")
