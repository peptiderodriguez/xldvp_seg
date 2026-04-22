# Manifold Sampling for LMD Replicate Pools

Partition the full cell population into **K spatially-tight, morphologically-
coherent replicate pools** targeted at a fixed tissue-area budget (default
2500 µm² ≈ 25 cells). Designed for laser-microdissection experiments that
need many pools spanning cell-type diversity plus spatial coverage.

Pairs with `global_cluster_spatial_viewer.py --rare-mode` for review and with
`xlseg export-lmd` for Leica XML generation.

## When to use it

- You need LMD pools, not cluster labels. This isn't a discovery pipeline —
  it's a sampler.
- You have ≥10,000 cells with morphology + SAM2 features.
- You want coverage across cell-type diversity **and** tissue location,
  budgeted to ~1000 pools (one LMD plate × 10 days throughput).

## Pipeline

```
detections
  → pre-filter (drop artifacts)
  → feature matrix (shape + color + SAM2)
  → log1p → RobustScaler → group-weighting → PCA (≤30 dims)
  → Level 1: FPS in PCA → K anchors maximally spread on manifold
           → Voronoi assign (each cell → nearest anchor → manifold_group_id)
           → outlier flag (top 2% by anchor-distance, or per-group MAD)
  → Level 2: for each (manifold_group, organ_id):
           → chunked Ward on xy → spatially-tight subgroups of ~25 cells
           → each subgroup = one Replicate
  → LMD selection: cap N per group, rank by morph tightness (or spatial /
    composite), optional plate-budget cut.
```

Output: `list[Replicate]` with full schema (`replicate_id`,
`manifold_group_id`, `organ_id`, `within_pair_replicate_idx`, `cell_uids`,
`cell_indices`, `n_cells`, `total_area_um2`, `mean_anchor_distance`,
`mean_xy_um`, `xy_spread_um`, `partial`).

## CLI

```bash
xlseg manifold-sample \
    --detections /path/to/cell_detections_with_organs.json \
    --output-dir /path/to/output \
    --k-anchors 1000 \
    --target-area-um2 2500 \
    --cap-per-group 5 \
    --priority composite
```

Typical wall time for 500K cells on 1× L40S + 32 CPUs:

| Step | Time |
|------|------|
| JSON load + pre-filter + feature matrix | 5-6 min |
| PCA (cuML GPU) | ~30 s |
| FPS K=1000 (GPU) | ~2 s |
| Voronoi + outlier flag | ~2 s |
| Level 2 chunked Ward across (group × organ) pairs | ~1 min |
| LMD selection + CSV/JSON writes | ~5 s |

## YAML config

See [`examples/configs/manifold_sample.yaml`](../examples/configs/manifold_sample.yaml).

## Outputs

| File | Contents |
|------|----------|
| `manifold_replicates.json` | list of `Replicate` dicts |
| `lmd_selected_replicates.json` / `.csv` | the capped subset for LMD |
| `manifold_state_<hash>.npz` | cache: `picked_idx`, `labels`, `d_to_anchor`, `outlier_mask`, `pca_cache_key` (cross-link to sibling `X_pca_<hash>.npz`) |
| `manifold_sample_stats.json` | per-group/per-organ counts |

Cache invalidates on any change to: `feature_groups`, pre-filter thresholds
(`nuc_filter_*`, `area_filter_*`), `max_pcs`, `pca_variance`, `seed`,
`exclude_channels`, `feature_group_weights`, `k_anchors`, `outlier_method`,
`outlier_threshold`, `organ_field`, `organ_drop_value`.

## Viewer

```bash
python scripts/manifold_viewer.py \
    --state-npz /path/to/output/manifold_state_<hash>.npz \
    --detections /path/to/cell_detections.json \
    --czi-path /path/to/slide.czi \
    --output /path/to/viewer.html
```

Produces a linked 3D-UMAP + slide thumbnail HTML. Click a group in UMAP →
its cells light up on the slide.

## Design choices

### Ward over K-means for (group, organ) spatial splitting
Ward is deterministic and diameter-minimizing — matches the "tight physical
pool" intent. K-means minimizes within-cluster variance, less suitable.
Beyond `ward_chunk_size` (default 2000), a kd-tree bin + stitched-Ward
keeps memory bounded. See `_chunked_ward_cluster` in `manifold_sampling.py`.

### FPS over K-means++ for anchor selection
FPS gives a 2-approximation to the k-center problem — maximally spreads
anchors across the manifold. K-means++ concentrates at density modes (bad
for rare-region coverage).

### 1/√(group_dim) group weighting
Without this, SAM2's 256 dims would dominate morphology's ~78 dims by 3×
purely by column count. Controlled by `feature_group_weights="equal"`
(default) vs `"raw"` (SAM2-dominated, not recommended).

### Two outlier methods
- `global_pct` (default, back-compat): top (100 − threshold)% by
  anchor-distance across all cells.
- `per_group_mad`: within each Voronoi cell, flag points
  `d > median + k·MAD`. Respects per-group geometry; useful when one
  cluster has a tight core and another has a heavy tail.

### Cross-device FPS seed
First anchor is picked via `np.random.default_rng(seed).integers(0, N)` —
deterministic across CPU/GPU runs. (cupy's own RNG diverges from numpy for
the same seed.)

### Spatial-spread guard
If a `(group, organ)` pair's cells fit inside ~1 replicate-radius, the
Level-2 splitter forces `n_rep=1`. Prevents Ward from finding noise
substructure in spatially coincident cells.

## See also

- [`docs/CLUSTER_DISCOVERY.md`](CLUSTER_DISCOVERY.md) — morphological cluster
  discovery via HDBSCAN + stability checks. Different problem: cluster
  discovery surfaces density-peaky subpopulations; manifold sampling
  partitions the whole population into tissue-area-budget LMD pools.
- [`docs/LMD_EXPORT_GUIDE.md`](LMD_EXPORT_GUIDE.md) — downstream XML
  generation for the Leica instrument.

## API reference

- `xldvp_seg.analysis.manifold_sampling.discover_manifold_replicates`
- `xldvp_seg.analysis.manifold_sampling.ManifoldSamplingConfig`
- `xldvp_seg.analysis.manifold_sampling.Replicate`
- `xldvp_seg.analysis.manifold_sampling.select_lmd_replicates`
- `xldvp_seg.analysis.manifold_sampling.fps_anchors`
- `xldvp_seg.analysis.manifold_sampling.voronoi_assign`
- `xldvp_seg.analysis.manifold_sampling.flag_outliers`
