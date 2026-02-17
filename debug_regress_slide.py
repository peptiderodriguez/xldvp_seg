#!/usr/bin/env python3
"""Regress out slide effects from nuclear deep features, re-run spectral pseudotime."""

import numpy as np
from scipy.stats import kruskal, spearmanr
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

mat = np.load('/viper/ptmp2/edrod/maturation_analysis_v2/maturation_data.npz', allow_pickle=True)
nuc = np.load('/viper/ptmp2/edrod/maturation_analysis_v2/nuclear_deep_features.npz', allow_pickle=True)
pt_old = np.load('/viper/ptmp2/edrod/maturation_analysis_v2/results/pseudotime.npz')

groups = mat['groups']
slides = mat['slides']

print('Available keys in nuclear features:', list(nuc.keys()))
# Get PCA features
pca_features = nuc['X_pca_valid']  # N x 500
print(f'PCA features shape: {pca_features.shape}')
N = len(pca_features)

# --- Method 1: Subtract slide means ---
print('\n=== Method 1: Subtract slide means ===')
corrected = pca_features.copy()
for s in np.unique(slides):
    mask = slides == s
    slide_mean = corrected[mask].mean(axis=0)
    corrected[mask] -= slide_mean

print(f'Corrected features shape: {corrected.shape}')

# --- Spectral pseudotime on corrected features ---
def spectral_pseudotime(features, k=15):
    """Compute spectral pseudotime via graph Laplacian."""
    print(f'  Building {k}-NN graph on {features.shape}...')
    A = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
    A = A + A.T  # symmetrize
    A[A > 1] = 1

    print('  Computing graph Laplacian...')
    L = laplacian(A, normed=True)

    print('  Finding eigenvectors...')
    eigenvalues, eigenvectors = eigsh(L, k=3, which='SM')
    print(f'  Eigenvalues: {eigenvalues}')

    # First non-trivial eigenvector (skip the constant one)
    pseudo = eigenvectors[:, 1]
    # Center
    pseudo = pseudo - pseudo.mean()
    return pseudo

pt_corrected = spectral_pseudotime(corrected)

# Orient pseudotime consistently (correlate with old pseudotime direction)
pt_old_spec = pt_old['pseudotime_spec']
rho, _ = spearmanr(pt_corrected, pt_old_spec)
if rho < 0:
    pt_corrected = -pt_corrected
    print('  Flipped pseudotime to match original direction')

# --- Compare confounding ---
def compute_stats(pt_arr, slide_arr, group_arr, label=''):
    gm = pt_arr.mean()
    sst = ((pt_arr - gm)**2).sum()
    ss_sl = sum(
        len(pt_arr[slide_arr == sl]) * (pt_arr[slide_arr == sl].mean() - gm)**2
        for sl in np.unique(slide_arr)
    )
    ss_gr = sum(
        len(pt_arr[group_arr == g]) * (pt_arr[group_arr == g].mean() - gm)**2
        for g in np.unique(group_arr)
    )
    e2s = ss_sl / sst
    e2g = ss_gr / sst

    ga = [pt_arr[group_arr == g] for g in np.unique(group_arr)]
    _, pg = kruskal(*ga)
    sa = [pt_arr[slide_arr == s] for s in np.unique(slide_arr)]
    _, ps = kruskal(*sa)

    print(f'{label:>25} eta2_slide={e2s:.4f} ({e2s*100:.1f}%)  eta2_group={e2g:.4f} ({e2g*100:.1f}%)  '
          f'ratio={e2s/e2g:.2f}  slide_p={ps:.2e}  group_p={pg:.2e}')
    return e2s, e2g

print(f'\n{"":>25} {"eta2_slide":>20} {"eta2_group":>20} {"ratio":>7} {"slide_p":>12} {"group_p":>12}')
compute_stats(pt_old_spec, slides, groups, label='Original')
compute_stats(pt_corrected, slides, groups, label='Slide-corrected')

# Per-slide breakdown
print(f'\nPer-slide pseudotime (corrected):')
print(f'{"Slide":>20} {"N":>5} {"Grp":>4} {"median_pt":>12} {"mean_pt":>12}')
for s in sorted(np.unique(slides)):
    mask = slides == s
    pts = pt_corrected[mask]
    grp = groups[mask][0]
    print(f'{s:>20} {mask.sum():>5} {grp:>4} {np.median(pts):>12.5f} {np.mean(pts):>12.5f}')

# Within-group slide effects
print(f'\nWithin-group slide effects (corrected):')
for g in sorted(np.unique(groups)):
    g_mask = groups == g
    g_slides = np.unique(slides[g_mask])
    if len(g_slides) > 1:
        within = [pt_corrected[(slides == s) & g_mask] for s in g_slides]
        Hw, pw = kruskal(*within)
        print(f'  {g}: KW H={Hw:.1f}, p={pw:.2e}')

# Save corrected pseudotime
out_path = '/viper/ptmp2/edrod/maturation_analysis_v2/pseudotime_slide_corrected.npz'
np.savez(out_path,
         pseudotime_corrected=pt_corrected,
         pseudotime_original=pt_old_spec,
         corrected_features=corrected)
print(f'\nSaved to {out_path}')
