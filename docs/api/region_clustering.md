# Region Clustering

Per-region feature exploration: PCA → UMAP → four-way clustering
(K-means via elbow, Leiden on PCA-kNN, HDBSCAN on PCA, HDBSCAN on UMAP) plus
the Hopkins statistic for clustering-tendency assessment.

Used by `scripts/region_pca_viewer.py` and `scripts/combined_region_viewer.py`
to generate the interactive HTML viewers. All helpers are pure (numpy in,
numpy out) so they're straightforward to reuse from notebooks or other
pipelines.

## When to use each method

| Method | Finds | Notes |
|--------|-------|-------|
| K-means (elbow) | Convex, roughly equal-sized clusters | Fast; misses non-linear structure |
| Leiden on PCA-kNN | Communities in neighborhood graph | Scverse-standard; captures non-linear structure, matches UMAP layout |
| HDBSCAN on PCA | Density-based clusters + noise | Returns `-1` for noise; handles outliers |
| HDBSCAN on UMAP | Lobes visible in 2D projection | Good visual match but UMAP distorts distances — less principled |

Hopkins statistic is computed on the PCA space:

- H ≈ 0.5: uniform/random (no clustering)
- H > 0.75: strong clustering tendency
- H > 0.9: very strong structure

::: xldvp_seg.analysis.region_clustering
    options:
      show_root_heading: false
      members_order: source
