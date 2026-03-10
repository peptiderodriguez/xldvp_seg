#!/usr/bin/env python3
"""Multi-scale vessel structure detection and classification.

Takes classified cell detections (from classify_markers.py) and:
1. Filters to marker-positive cells (SMA+, CD31+, double+)
2. Builds spatial graphs at multiple radii → connected components
3. Classifies each component morphology (ring, arc, linear, cluster)
4. Infers vessel type from marker composition + morphology + SNR
5. Computes hierarchical nesting across scales
6. Outputs enriched JSON + summary CSV
7. Optionally generates interactive spatial viewer HTML
8. Optionally runs squidpy spatial statistics

Usage:
    python scripts/vessel_community_analysis.py \
        --detections cell_detections_classified.json \
        --output-dir vessel_analysis/
"""

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

# Add repo to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Morphology classification (same thresholds as compute_graph_patterns())
# ---------------------------------------------------------------------------

def classify_morphology(pts):
    """Classify spatial pattern of a point set.

    Returns (pattern, metrics) where pattern is one of:
        'ring', 'arc', 'linear', 'cluster'

    Metrics dict contains: elongation, circularity, hollowness, has_curvature.
    Thresholds match generate_multi_slide_spatial_viewer.py:compute_graph_patterns().
    """
    nc = len(pts)
    if nc < 3:
        return 'cluster', {'elongation': 1.0, 'circularity': 0.0,
                           'hollowness': 0.0, 'has_curvature': False}

    # PCA elongation
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T) if nc > 2 else np.eye(2)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    lam1 = max(eigvals[0], 1e-10)
    lam2 = max(eigvals[1], 1e-10)
    elongation = np.sqrt(lam1 / lam2)

    # Circularity and hollowness
    cx, cy = pts.mean(axis=0)
    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    mean_r = radii.mean()
    circularity = (1.0 - radii.std() / mean_r) if mean_r > 1e-6 else 0.0
    hollowness = np.median(radii) / max(radii.max(), 1e-6)

    # Curvature check (2nd-order polynomial in PCA space)
    has_curvature = False
    if nc > 5 and elongation > 2.5:
        eigvecs = np.linalg.eigh(cov)[1]
        pc1 = eigvecs[:, -1]
        pc2 = eigvecs[:, -2]
        proj1 = centered @ pc1
        proj2 = centered @ pc2
        coeffs = np.polyfit(proj1, proj2, 2)
        pred = np.polyval(coeffs, proj1)
        ss_res = ((proj2 - pred) ** 2).sum()
        ss_tot = ((proj2 - proj2.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        if r2 > 0.3 and abs(coeffs[0]) > 1e-6:
            has_curvature = True

    # Decision tree
    if elongation > 4 and not has_curvature:
        pattern = 'linear'
    elif elongation > 3 and has_curvature:
        pattern = 'arc'
    elif circularity > 0.65 and hollowness > 0.55 and elongation < 3:
        pattern = 'ring'
    else:
        pattern = 'cluster'

    metrics = {
        'elongation': round(float(elongation), 3),
        'circularity': round(float(circularity), 3),
        'hollowness': round(float(hollowness), 3),
        'has_curvature': has_curvature,
    }
    return pattern, metrics


# ---------------------------------------------------------------------------
# Multi-scale connected components
# ---------------------------------------------------------------------------

def find_vessel_structures(positions, radii_um, min_cells=3):
    """Find connected components of positive cells at multiple spatial radii.

    Args:
        positions: (N, 2) array of cell coordinates in microns
        radii_um: list of radii to try
        min_cells: minimum cells per component

    Returns:
        dict mapping radius → list of component dicts, each with:
            'indices': array of cell indices in the component
            'positions': (n, 2) array of positions
    """
    tree = cKDTree(positions)
    results = {}

    for radius in radii_um:
        pairs = tree.query_pairs(r=radius)
        if not pairs:
            results[radius] = []
            continue

        rows, cols = zip(*pairs)
        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        n = len(positions)
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        adj = adj + adj.T

        n_comp, labels = connected_components(adj, directed=False)

        components = []
        for ci in range(n_comp):
            mask = labels == ci
            nc = int(mask.sum())
            if nc < min_cells:
                continue
            idx = np.where(mask)[0]
            components.append({
                'indices': idx,
                'positions': positions[idx],
            })

        # Sort by size descending
        components.sort(key=lambda c: len(c['indices']), reverse=True)
        results[radius] = components
        logger.info(f"  Radius {radius} um: {len(components)} structures "
                    f"({sum(len(c['indices']) for c in components)} cells)")

    return results


# ---------------------------------------------------------------------------
# Vessel type inference (SNR-informed)
# ---------------------------------------------------------------------------

def infer_vessel_type(component_indices, all_detections, morphology,
                      sma_snr_key='SMA_snr', cd31_snr_key='CD31_snr',
                      sma_class_key='SMA_class', cd31_class_key='CD31_class'):
    """Infer vessel type from marker composition + morphology + SNR.

    Returns (vessel_type, composition_dict).
    """
    n = len(component_indices)
    n_sma = 0
    n_cd31 = 0
    n_double = 0
    sma_snrs = []
    cd31_snrs = []

    for idx in component_indices:
        feat = all_detections[idx].get('features', {})
        is_sma = feat.get(sma_class_key) == 'positive'
        is_cd31 = feat.get(cd31_class_key) == 'positive'
        if is_sma and is_cd31:
            n_double += 1
        elif is_sma:
            n_sma += 1
        elif is_cd31:
            n_cd31 += 1

        sma_snr = feat.get(sma_snr_key, 0.0)
        cd31_snr = feat.get(cd31_snr_key, 0.0)
        sma_snrs.append(sma_snr)
        cd31_snrs.append(cd31_snr)

    sma_total = n_sma + n_double
    cd31_total = n_cd31 + n_double
    sma_frac = sma_total / n
    cd31_frac = cd31_total / n
    double_frac = n_double / n
    mean_sma_snr = float(np.mean(sma_snrs)) if sma_snrs else 0.0
    mean_cd31_snr = float(np.mean(cd31_snrs)) if cd31_snrs else 0.0

    # Classification rules (SNR-informed)
    if morphology == 'ring' and sma_frac > 0.5:
        vessel_type = 'artery_like'
    elif morphology in ('cluster', 'arc') and sma_frac > 0.3:
        # Distinguish confident vs low-confidence vein by SNR
        vessel_type = 'vein_like'
    elif n < 10 and cd31_frac > 0.5:
        vessel_type = 'capillary_like'
    elif morphology == 'linear' and cd31_frac > 0.5:
        vessel_type = 'endothelial_network'
    else:
        vessel_type = 'unclassified'

    composition = {
        'n_cells': n,
        'n_sma_only': n_sma,
        'n_cd31_only': n_cd31,
        'n_double_pos': n_double,
        'n_negative': n - n_sma - n_cd31 - n_double,
        'sma_frac': round(sma_frac, 3),
        'cd31_frac': round(cd31_frac, 3),
        'double_frac': round(double_frac, 3),
        'mean_sma_snr': round(mean_sma_snr, 4),
        'mean_cd31_snr': round(mean_cd31_snr, 4),
    }
    return vessel_type, composition


# ---------------------------------------------------------------------------
# Hierarchical nesting
# ---------------------------------------------------------------------------

def compute_hierarchy(multi_scale_results, radii_um):
    """Map fine-scale structures to coarse-scale parents.

    For each component at radius r[i], find which component at radius r[i+1]
    contains the majority of its cells (by index overlap).
    """
    hierarchy = {}  # (radius, comp_idx) → (parent_radius, parent_comp_idx)

    sorted_radii = sorted(radii_um)
    for i in range(len(sorted_radii) - 1):
        fine_r = sorted_radii[i]
        coarse_r = sorted_radii[i + 1]
        fine_comps = multi_scale_results.get(fine_r, [])
        coarse_comps = multi_scale_results.get(coarse_r, [])

        if not fine_comps or not coarse_comps:
            continue

        # Build index → coarse component mapping
        coarse_lookup = {}
        for ci, comp in enumerate(coarse_comps):
            for idx in comp['indices']:
                coarse_lookup[idx] = ci

        for fi, fcomp in enumerate(fine_comps):
            # Find which coarse component contains most cells
            parent_votes = Counter()
            for idx in fcomp['indices']:
                parent_ci = coarse_lookup.get(idx, -1)
                if parent_ci >= 0:
                    parent_votes[parent_ci] += 1

            if parent_votes:
                best_parent = parent_votes.most_common(1)[0][0]
                hierarchy[(fine_r, fi)] = (coarse_r, best_parent)

    return hierarchy


# ---------------------------------------------------------------------------
# ConvexHull area
# ---------------------------------------------------------------------------

def safe_hull_area(pts):
    """Compute convex hull area, return 0 for degenerate cases."""
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # 2D: volume = area
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_vessel_structures(detections, positive_indices, positions,
                              radii_um, min_cells, marker_names,
                              snr_keys, class_keys):
    """Run full multi-scale vessel structure analysis.

    Returns:
        structures: list of structure dicts (one per component per radius)
        hierarchy: nesting map
    """
    logger.info(f"Analyzing {len(positive_indices)} positive cells at "
                f"radii: {radii_um} um")

    # Build positions array for positive cells only
    pos_positions = positions[positive_indices]

    # Find structures at each radius
    multi_scale = find_vessel_structures(pos_positions, radii_um, min_cells)

    # Compute hierarchy
    hierarchy = compute_hierarchy(multi_scale, radii_um)

    # Classify each structure
    structures = []
    struct_id = 0

    for radius in sorted(radii_um):
        components = multi_scale.get(radius, [])
        for ci, comp in enumerate(components):
            # Map local indices back to global detection indices
            global_indices = positive_indices[comp['indices']]
            pts = comp['positions']

            # Morphology
            morphology, morph_metrics = classify_morphology(pts)

            # Vessel type (SNR-informed)
            vessel_type, composition = infer_vessel_type(
                global_indices, detections, morphology,
                sma_snr_key=snr_keys.get('SMA', 'SMA_snr'),
                cd31_snr_key=snr_keys.get('CD31', 'CD31_snr'),
                sma_class_key=class_keys.get('SMA', 'SMA_class'),
                cd31_class_key=class_keys.get('CD31', 'CD31_class'),
            )

            # Hull area
            hull_area = safe_hull_area(pts)

            # Centroid
            centroid = pts.mean(axis=0)

            # Parent in hierarchy
            parent_key = hierarchy.get((radius, ci))
            parent_info = None
            if parent_key:
                parent_info = {
                    'radius_um': parent_key[0],
                    'component_idx': parent_key[1],
                }

            structure = {
                'id': struct_id,
                'radius_um': radius,
                'component_idx': ci,
                'n_cells': len(global_indices),
                'morphology': morphology,
                'vessel_type': vessel_type,
                'centroid_x_um': round(float(centroid[0]), 2),
                'centroid_y_um': round(float(centroid[1]), 2),
                'hull_area_um2': round(hull_area, 1),
                'morph_metrics': morph_metrics,
                'composition': composition,
                'parent': parent_info,
                'cell_indices': global_indices.tolist(),
            }
            structures.append(structure)
            struct_id += 1

    return structures, hierarchy


def enrich_detections(detections, structures, best_radius):
    """Add vessel structure fields to detection features.

    Uses the specified radius as the primary assignment.
    Each cell gets: vessel_community_id, vessel_type, vessel_morphology, vessel_scale_um.
    """
    # Build cell → structure mapping at the best radius
    cell_to_struct = {}
    for s in structures:
        if s['radius_um'] != best_radius:
            continue
        for idx in s['cell_indices']:
            cell_to_struct[idx] = s

    n_assigned = 0
    for idx, det in enumerate(detections):
        feat = det.setdefault('features', {})
        s = cell_to_struct.get(idx)
        if s:
            feat['vessel_community_id'] = s['id']
            feat['vessel_type'] = s['vessel_type']
            feat['vessel_morphology'] = s['morphology']
            feat['vessel_scale_um'] = s['radius_um']
            n_assigned += 1
        else:
            feat['vessel_community_id'] = -1
            feat['vessel_type'] = 'none'
            feat['vessel_morphology'] = 'none'
            feat['vessel_scale_um'] = 0

    logger.info(f"Enriched {n_assigned} detections with vessel structure info "
                f"(radius={best_radius} um)")
    return detections


def write_summary_csv(structures, output_path):
    """Write one-row-per-structure summary CSV."""
    if not structures:
        logger.warning("No structures to write")
        return

    fieldnames = [
        'id', 'radius_um', 'n_cells', 'morphology', 'vessel_type',
        'sma_frac', 'cd31_frac', 'double_frac',
        'mean_sma_snr', 'mean_cd31_snr',
        'hull_area_um2', 'centroid_x_um', 'centroid_y_um',
        'elongation', 'circularity', 'hollowness',
        'parent_radius', 'parent_idx',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in structures:
            row = {
                'id': s['id'],
                'radius_um': s['radius_um'],
                'n_cells': s['n_cells'],
                'morphology': s['morphology'],
                'vessel_type': s['vessel_type'],
                'sma_frac': s['composition']['sma_frac'],
                'cd31_frac': s['composition']['cd31_frac'],
                'double_frac': s['composition']['double_frac'],
                'mean_sma_snr': s['composition']['mean_sma_snr'],
                'mean_cd31_snr': s['composition']['mean_cd31_snr'],
                'hull_area_um2': s['hull_area_um2'],
                'centroid_x_um': s['centroid_x_um'],
                'centroid_y_um': s['centroid_y_um'],
                'elongation': s['morph_metrics']['elongation'],
                'circularity': s['morph_metrics']['circularity'],
                'hollowness': s['morph_metrics']['hollowness'],
                'parent_radius': s['parent']['radius_um'] if s['parent'] else '',
                'parent_idx': s['parent']['component_idx'] if s['parent'] else '',
            }
            writer.writerow(row)

    logger.info(f"Wrote {len(structures)} structures to {output_path}")


def run_squidpy_analysis(detections, positive_indices, positions, output_dir,
                         marker_names, class_keys, snr_keys):
    """Run squidpy spatial statistics on positive cells."""
    try:
        import anndata
        import squidpy as sq
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("squidpy/anndata not installed — skipping spatial stats")
        return

    squidpy_dir = Path(output_dir) / 'squidpy'
    squidpy_dir.mkdir(parents=True, exist_ok=True)

    # Build AnnData from positive cells
    pos_pos = positions[positive_indices]
    n_pos = len(positive_indices)

    # Cell type labels and SNR features
    cell_types = []
    sma_snr_vals = []
    cd31_snr_vals = []
    for idx in positive_indices:
        feat = detections[idx].get('features', {})
        is_sma = feat.get(class_keys.get('SMA', 'SMA_class')) == 'positive'
        is_cd31 = feat.get(class_keys.get('CD31', 'CD31_class')) == 'positive'
        if is_sma and is_cd31:
            cell_types.append('SMA+CD31+')
        elif is_sma:
            cell_types.append('SMA+')
        elif is_cd31:
            cell_types.append('CD31+')
        else:
            cell_types.append('other')
        sma_snr_vals.append(feat.get(snr_keys.get('SMA', 'SMA_snr'), 0.0))
        cd31_snr_vals.append(feat.get(snr_keys.get('CD31', 'CD31_snr'), 0.0))

    import pandas as pd
    obs = pd.DataFrame({
        'cell_type': pd.Categorical(cell_types),
        'SMA_snr': sma_snr_vals,
        'CD31_snr': cd31_snr_vals,
    })

    # X = SNR values as feature matrix
    X = np.column_stack([sma_snr_vals, cd31_snr_vals]).astype(np.float32)
    adata = anndata.AnnData(X=X, obs=obs)
    adata.obsm['spatial'] = pos_pos.astype(np.float64)
    adata.var_names = ['SMA_snr', 'CD31_snr']

    logger.info(f"Built AnnData: {adata.shape[0]} cells, {adata.shape[1]} features")

    # Spatial neighbors
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=15)

    # Neighborhood enrichment
    try:
        sq.gr.nhood_enrichment(adata, cluster_key='cell_type')
        sq.pl.nhood_enrichment(adata, cluster_key='cell_type')
        plt.savefig(squidpy_dir / 'nhood_enrichment.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved nhood_enrichment.png")
    except Exception as e:
        logger.warning(f"  nhood_enrichment failed: {e}")

    # Co-occurrence
    try:
        sq.gr.co_occurrence(adata, cluster_key='cell_type')
        sq.pl.co_occurrence(adata, cluster_key='cell_type')
        plt.savefig(squidpy_dir / 'co_occurrence.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved co_occurrence.png")
    except Exception as e:
        logger.warning(f"  co_occurrence failed: {e}")

    # Ripley's L
    try:
        sq.gr.ripley(adata, cluster_key='cell_type', mode='L')
        sq.pl.ripley(adata, cluster_key='cell_type', mode='L')
        plt.savefig(squidpy_dir / 'ripley_L.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  Saved ripley_L.png")
    except Exception as e:
        logger.warning(f"  ripley_L failed: {e}")

    logger.info(f"Squidpy analysis complete → {squidpy_dir}")


# ---------------------------------------------------------------------------
# Leiden clustering on full feature space
# ---------------------------------------------------------------------------

def run_leiden_clustering(detections, positions, output_dir, resolution=0.5):
    """Leiden clustering on morph + channel SNR + SAM2 features.

    Produces: UMAP/t-SNE plots, spatial scatter, cluster composition bar chart,
    h5ad file, and adds leiden_cluster to detection features.
    """
    try:
        import scanpy as sc
        import anndata
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("scanpy/anndata not installed — skipping Leiden clustering")
        return detections

    logger.info("Building feature matrix for Leiden clustering...")

    # Select numeric features: morph + channel SNR/raw/std + SAM2
    sample_feat = detections[0].get('features', {})
    feature_keys = []
    for k in sorted(sample_feat.keys()):
        if k.startswith('sam2_'):
            feature_keys.append(k)
        elif k.startswith('ch') and k.split('_')[-1] in (
                'mean_raw', 'snr', 'std', 'max', 'median'):
            feature_keys.append(k)
        elif (not k.startswith('ch') and not k.startswith('sam2')
              and not k.startswith('SMA') and not k.startswith('CD31')
              and not k.startswith('marker') and not k.startswith('vessel')):
            v = sample_feat[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                feature_keys.append(k)

    logger.info(f"  Using {len(feature_keys)} features")

    X = np.zeros((len(detections), len(feature_keys)), dtype=np.float32)
    marker_profiles = []
    for i, det in enumerate(detections):
        feat = det.get('features', {})
        for j, k in enumerate(feature_keys):
            v = feat.get(k, 0.0)
            X[i, j] = float(v) if isinstance(v, (int, float)) else 0.0
        marker_profiles.append(feat.get('marker_profile', 'unknown'))

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    obs = pd.DataFrame({
        'marker_profile': pd.Categorical(marker_profiles),
    })
    adata = anndata.AnnData(X=X, obs=obs)
    adata.var_names = feature_keys
    adata.obsm['spatial'] = positions.astype(np.float64)

    logger.info(f"  AnnData: {adata.shape}")

    # Scanpy workflow
    logger.info("  Scaling...")
    sc.pp.scale(adata, max_value=10)

    logger.info("  PCA...")
    sc.tl.pca(adata, n_comps=50)

    import os
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))
    logger.info(f"  Using {n_jobs} threads for neighbors/UMAP")

    logger.info("  Neighbors...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

    logger.info(f"  Leiden (resolution={resolution})...")
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    n_clusters = len(adata.obs['leiden'].unique())
    logger.info(f"  Found {n_clusters} clusters")

    logger.info(f"  UMAP (n_jobs={n_jobs})...")
    # umap-learn reads NUMBA_NUM_THREADS for parallelization
    os.environ['NUMBA_NUM_THREADS'] = str(n_jobs)
    import umap as _umap
    # Call umap-learn directly with n_jobs instead of going through scanpy
    reducer = _umap.UMAP(n_components=2, random_state=0, n_jobs=n_jobs)
    adata.obsm['X_umap'] = reducer.fit_transform(adata.obsm['X_pca'][:, :50])

    # Plots
    leiden_dir = Path(output_dir) / 'leiden'
    leiden_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sc.pl.umap(adata, color='leiden', ax=axes[0], show=False,
               title=f'UMAP — Leiden (res={resolution})')
    sc.pl.umap(adata, color='marker_profile', ax=axes[1], show=False,
               title='UMAP — Marker Profile')
    plt.tight_layout()
    plt.savefig(leiden_dir / 'umap_leiden.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved umap_leiden.png")

    # Combined: marker-colored points with Leiden cluster contour outlines
    from scipy.ndimage import gaussian_filter
    umap_coords = adata.obsm['X_umap']
    marker_color_map = {
        'SMA+/CD31-': '#e41a1c',   # red
        'SMA-/CD31+': '#377eb8',   # blue
        'SMA+/CD31+': '#984ea3',   # purple
        'SMA-/CD31-': '#cccccc',   # light gray
    }
    leiden_cats = adata.obs['leiden'].cat.categories
    leiden_cmap = plt.cm.tab20(np.linspace(0, 1, min(len(leiden_cats), 20)))

    fig, ax = plt.subplots(figsize=(14, 10))
    # Plot points colored by marker profile
    for profile, color in marker_color_map.items():
        mask = adata.obs['marker_profile'] == profile
        if mask.any():
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       s=0.5, c=color, alpha=0.4, label=profile, rasterized=True)

    # Overlay Leiden cluster contours
    for ci, cat in enumerate(leiden_cats):
        mask = (adata.obs['leiden'] == cat).values
        if mask.sum() < 10:
            continue
        pts = umap_coords[mask]
        # KDE on a grid for this cluster
        xmin, xmax = pts[:, 0].min() - 1, pts[:, 0].max() + 1
        ymin, ymax = pts[:, 1].min() - 1, pts[:, 1].max() + 1
        bins = 80
        H, xedges, yedges = np.histogram2d(
            pts[:, 0], pts[:, 1], bins=bins,
            range=[[xmin, xmax], [ymin, ymax]])
        H = gaussian_filter(H.T, sigma=2.0)
        # Draw a single contour at ~20% of max density
        level = H.max() * 0.2
        if level > 0:
            ax.contour(
                np.linspace(xmin, xmax, bins),
                np.linspace(ymin, ymax, bins),
                H, levels=[level],
                colors=[leiden_cmap[ci % 20]], linewidths=1.5, alpha=0.8)
            # Label at centroid
            cx, cy = pts.mean(axis=0)
            ax.text(cx, cy, str(cat), fontsize=7, fontweight='bold',
                    color=leiden_cmap[ci % 20], ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.7))

    ax.set_title(f'UMAP — Marker Profile + Leiden Contours (res={resolution})')
    ax.legend(markerscale=8, fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(leiden_dir / 'umap_combined.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved umap_combined.png")

    # Spatial scatter
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for ci, cat in enumerate(leiden_cats):
        mask = adata.obs['leiden'] == cat
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   s=0.3, c=[leiden_cmap[ci % 20]], alpha=0.5,
                   label=cat, rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(f'Spatial — Leiden (res={resolution})')
    ax.invert_yaxis()
    if len(leiden_cats) <= 20:
        ax.legend(markerscale=10, fontsize=6, loc='upper right')
    plt.savefig(leiden_dir / 'spatial_leiden.png', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved spatial_leiden.png")

    # Cluster composition
    ct = pd.crosstab(adata.obs['leiden'], adata.obs['marker_profile'],
                     normalize='index')
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title(f'Cluster Composition — Leiden (res={resolution})')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Fraction')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(leiden_dir / 'cluster_composition.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved cluster_composition.png")

    # Save h5ad
    adata.write(leiden_dir / 'vessel_adata.h5ad')
    logger.info(f"  Saved vessel_adata.h5ad")

    # Add cluster labels to detections
    leiden_labels = adata.obs['leiden'].values
    for i, det in enumerate(detections):
        det.setdefault('features', {})['leiden_cluster'] = str(leiden_labels[i])

    # Summary
    logger.info(f"\n  Leiden cluster summary:")
    for cl in sorted(adata.obs['leiden'].unique(), key=int):
        mask = adata.obs['leiden'] == cl
        n = mask.sum()
        profiles = adata.obs.loc[mask, 'marker_profile'].value_counts()
        top = profiles.head(2)
        desc = ', '.join(f'{p}: {c}' for p, c in top.items())
        logger.info(f"    Cluster {cl}: {n:,} cells — {desc}")

    logger.info(f"  Leiden clustering complete → {leiden_dir}")
    return detections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-scale vessel structure detection and classification')

    parser.add_argument('--detections', required=True,
                        help='Path to classified detections JSON '
                             '(from classify_markers.py)')
    parser.add_argument('--marker-field', default='marker_profile',
                        help='Feature field with combined marker label '
                             '(default: marker_profile)')
    parser.add_argument('--positive-values', default='SMA+/CD31-,SMA-/CD31+,SMA+/CD31+',
                        help='Comma-separated marker_profile values to include '
                             '(default: SMA+/CD31-,SMA-/CD31+,SMA+/CD31+)')
    parser.add_argument('--marker-names', default='SMA,CD31',
                        help='Comma-separated marker names (default: SMA,CD31)')
    parser.add_argument('--snr-channels', default=None,
                        help='Comma-separated channel indices for SNR keys '
                             '(e.g. "1,3" → ch1_snr, ch3_snr). '
                             'Auto-detected if not specified.')
    parser.add_argument('--radii', default='25,50,100,200',
                        help='Comma-separated radii in um (default: 25,50,100,200)')
    parser.add_argument('--best-radius', type=float, default=50,
                        help='Radius to use for primary cell assignment (default: 50)')
    parser.add_argument('--min-cells', type=int, default=3,
                        help='Minimum cells per structure (default: 3)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: same as detections)')
    parser.add_argument('--generate-viewer', action='store_true',
                        help='Generate interactive spatial viewer HTML')
    parser.add_argument('--run-squidpy', action='store_true',
                        help='Run squidpy spatial statistics')
    parser.add_argument('--run-leiden', action='store_true',
                        help='Run Leiden clustering on full feature space '
                             '(morph + channel SNR + SAM2)')
    parser.add_argument('--leiden-resolution', type=float, default=0.5,
                        help='Leiden resolution (default: 0.5)')
    parser.add_argument('--viewer-group-field', default='vessel_type',
                        help='Field to color by in viewer (default: vessel_type)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    det_path = Path(args.detections)
    if not det_path.exists():
        logger.error(f"Detections not found: {det_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else det_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    radii_um = [float(r) for r in args.radii.split(',')]
    positive_values = [v.strip() for v in args.positive_values.split(',')]
    marker_names = [m.strip() for m in args.marker_names.split(',')]

    # Load detections
    logger.info(f"Loading detections from {det_path}...")
    detections = fast_json_load(det_path)
    logger.info(f"Loaded {len(detections):,} detections")

    # Build key mappings — auto-detect SNR keys
    class_keys = {m: f'{m}_class' for m in marker_names}

    # Auto-detect SNR keys: prefer {marker}_snr, fall back to ch{N}_snr
    sample_feat = detections[0].get('features', {}) if detections else {}
    if args.snr_channels:
        ch_indices = [int(c) for c in args.snr_channels.split(',')]
        snr_keys = {m: f'ch{ch}_snr' for m, ch in zip(marker_names, ch_indices)}
    else:
        snr_keys = {}
        for m in marker_names:
            if f'{m}_snr' in sample_feat:
                snr_keys[m] = f'{m}_snr'
            else:
                # No marker-named SNR — user must specify --snr-channels
                snr_keys[m] = f'{m}_snr'

        # Verify keys exist — fail loudly if missing
        missing = [m for m in marker_names if snr_keys[m] not in sample_feat]
        if missing:
            available = [k for k in sample_feat if k.endswith('_snr')]
            logger.error(
                f"SNR keys not found for {missing}. "
                f"Available SNR keys: {available}. "
                f"Use --snr-channels to specify channel indices "
                f"(e.g. --snr-channels 1,3 for ch1_snr,ch3_snr).")
            sys.exit(1)

    logger.info(f"SNR keys: {snr_keys}")
    logger.info(f"Class keys: {class_keys}")

    # Extract positions (global_center_um)
    positions = np.array([
        d.get('global_center_um', d.get('global_center', [0, 0]))
        for d in detections
    ], dtype=np.float64)

    # Filter to positive cells
    marker_field = args.marker_field
    positive_mask = np.array([
        d.get('features', {}).get(marker_field, '') in positive_values
        for d in detections
    ])
    positive_indices = np.where(positive_mask)[0]
    logger.info(f"Positive cells: {len(positive_indices):,} / {len(detections):,} "
                f"({100 * len(positive_indices) / max(len(detections), 1):.1f}%)")

    if len(positive_indices) < args.min_cells:
        logger.error(f"Too few positive cells ({len(positive_indices)}) for analysis")
        sys.exit(1)

    # Run analysis
    logger.info("=" * 60)
    logger.info("Multi-scale vessel structure analysis")
    logger.info("=" * 60)

    structures, hierarchy = analyze_vessel_structures(
        detections, positive_indices, positions,
        radii_um, args.min_cells, marker_names, snr_keys, class_keys
    )

    # Summary
    logger.info(f"\nFound {len(structures)} structures across all scales:")
    for radius in sorted(radii_um):
        r_structs = [s for s in structures if s['radius_um'] == radius]
        if r_structs:
            types = Counter(s['vessel_type'] for s in r_structs)
            morphs = Counter(s['morphology'] for s in r_structs)
            logger.info(f"  Radius {radius} um: {len(r_structs)} structures")
            logger.info(f"    Types: {dict(types)}")
            logger.info(f"    Morphologies: {dict(morphs)}")

    # Write summary CSV
    csv_path = output_dir / 'vessel_structures.csv'
    write_summary_csv(structures, csv_path)

    # Enrich detections with vessel assignments
    best_radius = args.best_radius
    if best_radius not in radii_um:
        best_radius = radii_um[len(radii_um) // 2]  # middle radius
        logger.info(f"Best radius {args.best_radius} not in radii; using {best_radius}")

    detections = enrich_detections(detections, structures, best_radius)

    # Leiden clustering (before saving so labels are included)
    if args.run_leiden:
        logger.info("\nRunning Leiden clustering on full feature space...")
        detections = run_leiden_clustering(
            detections, positions, output_dir, args.leiden_resolution
        )

    # Save enriched detections
    enriched_path = output_dir / 'cell_detections_vessel_analysis.json'
    logger.info(f"Saving enriched detections to {enriched_path}...")
    atomic_json_dump(detections, enriched_path)
    logger.info(f"  Wrote {len(detections):,} detections")

    # Squidpy analysis
    if args.run_squidpy:
        logger.info("\nRunning squidpy spatial statistics...")
        run_squidpy_analysis(
            detections, positive_indices, positions, output_dir,
            marker_names, class_keys, snr_keys
        )

    # Generate spatial viewer
    if args.generate_viewer:
        logger.info("\nGenerating interactive spatial viewer...")
        # Create slim JSON (coords + group fields only) to avoid OOM
        slim_path = output_dir / 'cell_detections_slim.json'
        keep_keys = {'vessel_type', 'vessel_morphology', 'vessel_community_id',
                     'vessel_scale_um', 'marker_profile', 'leiden_cluster'}
        keep_keys.update(f'{m}_class' for m in marker_names)
        slim = []
        for d in detections:
            slim.append({
                'global_center_um': d.get('global_center_um',
                                          d.get('global_center', [0, 0])),
                'features': {k: v for k, v in d.get('features', {}).items()
                             if k in keep_keys},
            })
        atomic_json_dump(slim, slim_path)
        logger.info(f"  Wrote slim JSON ({len(slim):,} detections) → {slim_path}")

        import subprocess
        viewer_script = REPO / 'scripts' / 'generate_multi_slide_spatial_viewer.py'
        viewer_html = output_dir / 'vessel_viewer.html'
        cmd = [
            sys.executable, str(viewer_script),
            '--detections', str(slim_path),
            '--group-field', args.viewer_group_field,
            '--output', str(viewer_html),
            '--title', 'Vessel Community Analysis',
        ]
        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"  Viewer generated → {viewer_html}")
        else:
            logger.warning(f"  Viewer generation failed:\n{result.stderr}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Vessel community analysis complete")
    logger.info("=" * 60)
    logger.info(f"  Structures found: {len(structures)}")
    logger.info(f"  Enriched detections: {enriched_path}")
    logger.info(f"  Summary CSV: {csv_path}")
    if args.run_squidpy:
        logger.info(f"  Squidpy results: {output_dir / 'squidpy/'}")


if __name__ == '__main__':
    main()
