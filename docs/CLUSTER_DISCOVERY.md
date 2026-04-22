# Morphological Cluster Discovery

Discover morphologically distinct cell populations across the full detection
set via HDBSCAN clustering + stability checks + Moran's I spatial cohesion +
Ward taxonomy. Covers the whole population — both common types (large
coherent clusters) and smaller surprising ones — and flags each by density
stability, spatial cohesion, and per-cluster size. Pairs with the
`global_cluster_spatial_viewer.py --rare-mode` interactive viewer for review.

## When to use it

- You have ≥10,000 cells with morphology + SAM2 features extracted.
- You want to **partition the whole population** by morphology and review
  each cluster's size, stability, and spatial distribution.
- You're looking for **populations**, not single-cell outliers. For
  single-cell outlier detection use a different method (Isolation Forest,
  per-cell anomaly scoring) — HDBSCAN at ``min_cluster_size=1000`` cannot find
  per-cell anomalies.

## Pipeline

```
detections → pre-filter → feature matrix (shape + color + SAM2)
           → log1p on area-like features → RobustScaler → per-group weighting
           → PCA (≤30 dims, ~95% variance)
           → HDBSCAN (primary run + stability runs at ± min_cluster_size)
           → Moran's I on k-NN graph (spatial cohesion)
           → Ward linkage on cluster centroids (taxonomy dendrogram)
           → exemplar cards (card-grid HTML for review)
```

Three-value sentinel on output ``rare_pop_id``:

- ``0, 1, 2, ...`` — density-based cluster member
- ``-1`` — HDBSCAN noise (passed pre-filter, not dense enough)
- ``-2`` — pre-filter drop (``rare_pop_filter_reason`` gives why:
  ``n_nuclei`` / ``nc_ratio`` / ``overlap`` / ``area`` / ``missing_features``)

## Feature-group weighting

The default ``feature_group_weights="equal"`` scales each feature group's
columns by ``1/sqrt(group_dim)`` so every group contributes roughly 1 unit of
squared Euclidean distance per cell pair, regardless of how many dims the
group has.

Without this, SAM2's 256 dims would drown shape+color's ~78 dims by **~3×**
purely by column count, making HDBSCAN clusters reflect texture variance
rather than morphology variance.

Set ``"raw"`` to disable (SAM2-dominated) or pass a dict like
``{"shape": 1.0, "sam2": 0.5}`` for explicit per-group multipliers on top of
the equal baseline.

## CLI

```bash
xlseg discover-rare-cells \
    --detections cell_detections_with_organs.json \
    --output-dir rare_cells_out/ \
    --feature-groups shape,color,sam2 \
    --feature-group-weights equal \
    --min-cluster-size 1000 \
    --stability-sizes 500,1000,2000 \
    --max-pcs 30 \
    --pca-variance 0.95 \
    --czi-path slide.czi \
    --display-channels 2,4 \
    --n-exemplars-per-cluster 100
```

## Outputs

| File | Contents |
|------|----------|
| ``detections_with_rare_labels.json`` | Input detections + ``rare_pop_id``, ``hdbscan_prob``, ``rare_pop_filter_reason`` fields |
| ``cluster_summary.csv`` | Per-cluster: size, persistence, Moran's I, stable, top_regions, top_morph_features, noise_pct |
| ``linkage.npy`` | SciPy Ward linkage matrix on cluster centroids |
| ``cluster_ids.npy`` | Cluster IDs ordered to match linkage rows |
| ``dendrogram.png`` | Ward-linkage plot of cluster centroids |
| ``exemplars_annotation/`` | Card-grid HTML of sampled cells per cluster (via ``regenerate_html.py``) |
| ``X_pca_<hash>.npz`` | Cached PCA matrix for fast re-runs |
| ``W_knn_k<k>_<pos_hash>.npz`` | Cached k-NN adjacency for fast re-runs |

Cache keys include feature groups, pre-filter thresholds, ``max_pcs``,
``pca_variance``, seed, excluded channels, and weighting mode — any change
forces recomputation.

## Interactive review

```bash
python scripts/global_cluster_spatial_viewer.py \
    --rare-mode \
    --detections rare_cells_out/detections_with_rare_labels.json \
    --rare-cluster-summary rare_cells_out/cluster_summary.csv \
    --rare-linkage-matrix rare_cells_out/linkage.npy \
    --rare-cluster-ids rare_cells_out/cluster_ids.npy \
    --output rare_cells_viewer.html
```

Taxonomy-only viewer: clickable Ward dendrogram + per-cluster sidebar (size,
persistence, Moran's I, stable flag, top regions, top morph features). Click
a leaf → cluster detail panel.

## Design choices

- **HDBSCAN not k-means** — requires no K; "rare" clusters shouldn't have to
  compete with the main mass on inertia.
- **PCA cap at 30 dims** — HDBSCAN density estimation degrades in very high
  dimensions (curse-of-dimensionality uniformity).
- **Reciprocal-best-match Jaccard stability** — a cluster is ``stable`` iff
  it matches the same alt-cluster across ≥N alternate-``min_cluster_size``
  runs via reciprocal argmax. Prevents two primary clusters from both claiming
  the same merged alt cluster (silent stability inflation).
- **Moran's I with NaN on degenerate** — clusters where z ≡ 0 (all-in / all-
  out of a tissue region) return NaN, not 0 — distinguishes "undefined" from
  "no spatial structure."
- **Atomic cache writes** — tmp file + ``os.replace``; crash-safe.
- **SAM2 captures visual similarity, not cell-type identity** — cells that
  look alike in PM+Hoechst staining cluster together even if they're
  biologically distinct. For cell-type discovery by marker expression, use
  the marker-classification pipeline (``xlseg markers``) instead or add the
  ``channel`` feature group.

## See also

- Core module: ``xldvp_seg.analysis.rare_cell_discovery``
- Viewer extension: ``scripts/global_cluster_spatial_viewer.py --rare-mode``
- Exemplar HTML template: ``scripts/regenerate_html.py``
- [LMD Export Guide](LMD_EXPORT_GUIDE.md) for downstream dissection
