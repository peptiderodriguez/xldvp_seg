"""Per-region feature exploration: PCA + UMAP + clustering with 4 methods.

Shared helpers used by :mod:`scripts.region_pca_viewer` and
:mod:`scripts.combined_region_viewer` for per-organ-region feature analysis.

Pipeline (per :func:`process_region`):

1. Drop constant features (zero-variance → NaN after StandardScaler)
2. StandardScaler → PCA with enough PCs to hit ``var_cutoff`` (default 90%)
3. Subsample to ``max_points_plot`` (for UMAP / viewer rendering speed)
4. UMAP 2D for visualization (on PCA-reduced subsample)
5. Clustering (all on PCA subsample — principled; UMAP distorts distances):

   - K-means with elbow method (:func:`find_optimal_k_elbow`)
   - Leiden on PCA-kNN graph (:func:`cluster_leiden`)
   - HDBSCAN on PCA space (:func:`cluster_hdbscan`)
   - HDBSCAN on UMAP coords (visual lobes — less principled but matches viz)

6. Hopkins statistic (:func:`hopkins_statistic`) on PCA space — clustering
   tendency (0.5 random, >0.75 strong structure)

The returned dict is directly JSON-serializable for embedding in HTML viewers.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from xldvp_seg.analysis.cluster_features import _extract_feature_matrix
from xldvp_seg.utils.logging import get_logger
from xldvp_seg.utils.seeding import resolve_seed

logger = get_logger(__name__)


def hopkins_statistic(X, n_samples=None, rng=None):
    """Hopkins statistic — clustering tendency of a point cloud.

    H ~ 0.5 means uniform/random. H > 0.7 suggests meaningful clusters;
    H > 0.9 suggests very strong clustering structure.

    Args:
        X: ``(n, d)`` feature matrix (ideally scaled).
        n_samples: Number of test points (default: ``min(200, max(10, n // 2))``).
        rng: numpy random Generator (default: seed 42).

    Returns:
        Hopkins statistic in ``[0, 1]``.
    """
    if rng is None:

        rng = np.random.default_rng(resolve_seed(None, caller="region_clustering"))
    n, d = X.shape
    if n_samples is None:
        n_samples = min(200, max(10, n // 2))
    n_samples = min(n_samples, n - 1)

    tree = BallTree(X)

    idx = rng.choice(n, size=n_samples, replace=False)
    X_sample = X[idx]
    nn_dist, _ = tree.query(X_sample, k=2)
    w = nn_dist[:, 1]  # second neighbor = nearest non-self

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    U = rng.uniform(mins, maxs, size=(n_samples, d))
    nn_dist_u, _ = tree.query(U, k=1)
    u = nn_dist_u[:, 0]

    u_sum, w_sum = u.sum(), w.sum()
    if u_sum + w_sum == 0:
        return 0.5
    return u_sum / (u_sum + w_sum)


def find_optimal_k_elbow(X, max_k=8, rng=None):
    """Pick k for k-means via the elbow method (kneedle-like).

    Fits k-means for k=2..max_k, picks k at the maximum perpendicular distance
    from the straight line joining the two endpoints of the inertia curve.
    Silhouette is still computed per k for display but does NOT drive selection.

    Args:
        X: ``(n, d)`` feature matrix.
        max_k: Maximum k to consider (default 8).
        rng: numpy random Generator (default: seed 42).

    Returns:
        ``(best_k, best_silhouette, best_labels, calinski_harabasz,
           silhouette_per_k, inertia_per_k)``.
    """
    if rng is None:

        rng = np.random.default_rng(resolve_seed(None, caller="region_clustering"))
    n = X.shape[0]
    max_k = min(max_k, n - 1)
    if max_k < 2:
        # Degenerate input: return single-entry sentinels instead of empty dicts
        # so downstream consumers (plotting, CSV export) don't crash on KeyError.
        return 1, 0.0, np.zeros(n, dtype=int), 0.0, {1: 0.0}, {1: 0.0}

    sil_per_k: dict[int, float] = {}
    inertia_per_k: dict[int, float] = {}
    labels_per_k: dict[int, np.ndarray] = {}

    # Subsample for k-means fit speed; always predict on full X
    fit_X = X
    if n > 10000:
        idx = rng.choice(n, size=10000, replace=False)
        fit_X = X[idx]

    for k in range(2, max_k + 1):

        km = KMeans(
            n_clusters=k,
            n_init=5,
            random_state=resolve_seed(None, caller="find_optimal_k_elbow"),
            max_iter=100,
        )
        if n > 10000:
            km.fit(fit_X)
            labels = km.predict(X)
            centers = km.cluster_centers_
            diffs = X - centers[labels]
            inertia = float(np.sum(diffs * diffs))
        else:
            labels = km.fit_predict(X)
            inertia = float(km.inertia_)

        labels_per_k[k] = labels
        inertia_per_k[k] = round(inertia, 2)

        # Silhouette for display only (subsampled for speed)
        if n > 5000:
            sub_idx = rng.choice(n, size=5000, replace=False)
            sub_labels = labels[sub_idx]
            if len(np.unique(sub_labels)) < 2:
                sil = -1.0
            else:
                sil = silhouette_score(X[sub_idx], sub_labels)
        else:
            sil = silhouette_score(X, labels)
        sil_per_k[k] = round(sil, 3)

    # Elbow: normalize (k, inertia) to [0,1]; pick k at max perpendicular distance
    # from the line joining (k_min, inertia_min) to (k_max, inertia_max).
    ks = np.array(sorted(inertia_per_k.keys()), dtype=float)
    ins = np.array([inertia_per_k[int(k)] for k in ks], dtype=float)
    if len(ks) == 1:
        best_k = int(ks[0])
    else:
        k_n = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
        if ins.max() - ins.min() > 1e-12:
            i_n = (ins - ins.min()) / (ins.max() - ins.min())
        else:
            i_n = np.zeros_like(ins)
        # Line from (0,1) to (1,0) → |y - (1 - x)| is the perpendicular distance
        distances = np.abs(i_n - (1.0 - k_n))
        best_k = int(ks[int(np.argmax(distances))])

    best_labels = labels_per_k[best_k]
    best_sil = sil_per_k[best_k]
    best_ch = round(calinski_harabasz_score(X, best_labels), 1)

    return best_k, best_sil, best_labels, best_ch, sil_per_k, inertia_per_k


def cluster_leiden(X, *, n_neighbors=15, resolution=1.0, seed=42):
    """Leiden community detection on a k-NN graph built in Euclidean space.

    Follows the scverse/scanpy convention: build kNN graph (default k=15) on
    the input space, then find communities via Leiden with RB-configuration
    partition quality.

    Args:
        X: ``(n, d)`` feature matrix.
        n_neighbors: k for the kNN graph (default 15).
        resolution: Higher → more, smaller communities (default 1.0).
        seed: RNG seed for Leiden (default 42).

    Returns:
        Integer cluster labels ``(n,)`` in ``{0, .., K-1}``.
    """
    import igraph as ig
    import leidenalg

    n = X.shape[0]
    if n < 2:
        # NearestNeighbors + leiden both require n >= 2; return a single cluster.
        return np.zeros(n, dtype=np.int32)
    n_neighbors = min(n_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X)
    _, idxs = nn.kneighbors(X)
    src = np.repeat(np.arange(n), n_neighbors)
    dst = idxs[:, 1:].reshape(-1)
    edges = np.column_stack([src, dst]).tolist()
    g = ig.Graph(n=n, edges=edges, directed=False)
    g = g.simplify(multiple=True, loops=True)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.array(partition.membership, dtype=np.int32)


def cluster_hdbscan(X, *, min_cluster_size=50):
    """HDBSCAN density-based clustering.

    ``min_cluster_size`` is internally capped at ``n // 5`` to avoid
    pathological settings on small inputs (lower bound 5).

    Args:
        X: ``(n, d)`` feature matrix.
        min_cluster_size: Minimum members for a cluster to be kept (default 50).
            Noise points are labeled -1.

    Returns:
        Integer cluster labels ``(n,)``. Value ``-1`` = noise.
    """
    import hdbscan

    n = X.shape[0]
    mcs = max(5, min(min_cluster_size, n // 5))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=None, core_dist_n_jobs=1)
    return clusterer.fit_predict(X).astype(np.int32)


def process_region(
    detections,
    feature_names,
    max_k=8,
    max_points_plot=5000,
    rng=None,
    *,
    seed: int | None = None,
    var_cutoff=0.90,
    max_pcs=50,
    umap_neighbors=15,
    umap_min_dist=0.1,
    leiden_resolution=1.0,
    leiden_knn=15,
    hdbscan_min_size=50,
):
    """Full per-region PCA → UMAP → 4-way clustering pipeline.

    **Clustering runs on the full set of cells** (kmeans, Leiden, HDBSCAN-PCA).
    UMAP runs on a subsample for speed and display (UMAP scales poorly past
    ~30K points, and the HTML viewer can't render more than that anyway).
    HDBSCAN-UMAP runs on the UMAP subsample (that's the only place UMAP coords
    exist — document this in the viewer).

    Returns a JSON-serializable dict suitable for embedding in HTML viewers,
    or ``None`` if the region has too few valid cells (<50 after filtering)
    or too few non-constant features (<2).

    See module docstring for the full pipeline description.
    """
    from umap import UMAP

    if rng is None:

        rng = np.random.default_rng(resolve_seed(None, caller="region_clustering"))

    X, _, _ = _extract_feature_matrix(detections, feature_names)
    if X is None or X.shape[0] < 50:
        return None

    # Drop zero-variance features (would produce NaN after scaling)
    variances = np.var(X, axis=0)
    nonconstant = variances > 1e-12
    active_names = [feature_names[i] for i in range(len(feature_names)) if nonconstant[i]]
    X = X[:, nonconstant]
    if X.shape[1] < 2:
        return None

    X_scaled = StandardScaler().fit_transform(X)

    # PCA — enough components to hit var_cutoff
    full_max = min(X_scaled.shape[0] - 1, X_scaled.shape[1])

    _rc_rs = resolve_seed(seed, caller="region_clustering.process_region")
    # If caller didn't supply rng, build one from the resolved seed so
    # subsampling is reproducible too.
    if rng is None:
        rng = np.random.default_rng(_rc_rs)
    pca_full = PCA(n_components=full_max, random_state=_rc_rs)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_cutoff = int(np.searchsorted(cumvar, var_cutoff) + 1)
    n_pcs = max(2, min(n_for_cutoff, max_pcs, full_max))

    X_pca = pca_full.transform(X_scaled)[:, :n_pcs]
    n_total = X_pca.shape[0]

    # --- Clustering on FULL PCA space ---
    hopkins = hopkins_statistic(X_pca, rng=rng)
    best_k, sil, labels_kmeans, ch, sil_per_k, inertia_per_k = find_optimal_k_elbow(
        X_pca, max_k=max_k, rng=rng
    )
    try:
        labels_leiden = cluster_leiden(X_pca, n_neighbors=leiden_knn, resolution=leiden_resolution)
    except Exception as e:
        logger.warning("Leiden failed: %s", e)
        labels_leiden = np.zeros(n_total, dtype=np.int32)
    try:
        labels_hdb_pca = cluster_hdbscan(X_pca, min_cluster_size=hdbscan_min_size)
    except Exception as e:
        logger.warning("HDBSCAN (PCA) failed: %s", e)
        labels_hdb_pca = np.zeros(n_total, dtype=np.int32)

    # --- Subsample for UMAP + viewer display ---
    if n_total > max_points_plot:
        sub_idx = rng.choice(n_total, size=max_points_plot, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n_total)
    X_pca_sub = X_pca[sub_idx]

    # UMAP on subsample (full-set UMAP is too slow per-region; 5K is ample for viz)
    n_neighbors = min(umap_neighbors, X_pca_sub.shape[0] - 1)
    umap = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        random_state=_rc_rs,
        n_jobs=1,
    )
    X_umap = umap.fit_transform(X_pca_sub)

    # HDBSCAN on UMAP coords — only subsample has UMAP
    try:
        labels_hdb_umap_sub = cluster_hdbscan(X_umap, min_cluster_size=hdbscan_min_size)
    except Exception as e:
        logger.warning("HDBSCAN (UMAP) failed: %s", e)
        labels_hdb_umap_sub = np.zeros(len(sub_idx), dtype=np.int32)
    # Extend HDBSCAN-UMAP labels to full-set length so indexing is uniform.
    # Non-subsampled cells get label -1 (noise) since we have no UMAP coord for them.
    labels_hdb_umap = np.full(n_total, -1, dtype=np.int32)
    labels_hdb_umap[sub_idx] = labels_hdb_umap_sub

    def _n_clusters(lbl):
        return int(len(np.unique(lbl[lbl >= 0])))

    def _n_noise(lbl):
        return int(np.sum(lbl < 0))

    # Variance explained by first 3 PCs (padded)
    var_first3 = [round(float(v), 4) for v in pca_full.explained_variance_ratio_[:3]]
    while len(var_first3) < 3:
        var_first3.append(0.0)

    # Top loadings for first 3 PCs
    top_loadings = {}
    for pc_idx in range(min(3, pca_full.components_.shape[0])):
        loadings = pca_full.components_[pc_idx]
        top_idx = np.argsort(np.abs(loadings))[::-1][:5]
        top_loadings[f"PC{pc_idx + 1}"] = [
            {"feature": active_names[i], "loading": round(float(loadings[i]), 3)} for i in top_idx
        ]

    # Cluster counts / noise counts reflect FULL-SET labels (all cells in region).
    # Display label arrays (labels_*) are subsample-sliced to match the scatter.
    return {
        "n_cells": n_total,
        "n_features": len(active_names),
        "n_pcs_used": int(n_pcs),
        "n_plotted": len(sub_idx),
        "pc1": X_pca_sub[:, 0].tolist(),
        "pc2": X_pca_sub[:, 1].tolist() if X_pca_sub.shape[1] >= 2 else [0.0] * len(sub_idx),
        "umap_x": X_umap[:, 0].tolist(),
        "umap_y": X_umap[:, 1].tolist(),
        # Subsample-sliced labels for the scatter display:
        "labels": labels_kmeans[sub_idx].tolist(),  # backward-compat (kmeans default)
        "labels_kmeans": labels_kmeans[sub_idx].tolist(),
        "labels_leiden": labels_leiden[sub_idx].tolist(),
        "labels_hdbscan_pca": labels_hdb_pca[sub_idx].tolist(),
        "labels_hdbscan_umap": labels_hdb_umap_sub.tolist(),  # already subsample
        # Counts computed on FULL-SET labels (true cluster count for the region):
        "n_clusters": {
            "kmeans": int(best_k),
            "leiden": _n_clusters(labels_leiden),
            "hdbscan_pca": _n_clusters(labels_hdb_pca),
            "hdbscan_umap": _n_clusters(labels_hdb_umap_sub),  # HDBSCAN-UMAP full=subsample
        },
        "n_noise": {
            "kmeans": 0,
            "leiden": _n_noise(labels_leiden),
            "hdbscan_pca": _n_noise(labels_hdb_pca),
            "hdbscan_umap": _n_noise(labels_hdb_umap_sub),
        },
        "hopkins": round(float(hopkins), 3),
        "best_k": int(best_k),
        "silhouette": float(sil),
        "calinski_harabasz": float(ch),
        "silhouette_per_k": sil_per_k,
        "inertia_per_k": inertia_per_k,
        "var_explained": var_first3,
        "top_loadings": top_loadings,
    }
