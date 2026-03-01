You are guiding the user through SpatialData export and scverse ecosystem analysis for the xldvp_seg pipeline.

---

## Prerequisites Check

1. **Detections must exist.** Look for `*_detections.json` or `*_detections_classified.json` in the output directory. If not found, redirect to `/analyze` first.

2. **Check if SpatialData was already auto-generated.** Look for `*_spatialdata.zarr` in the output directory. If it exists, ask if the user wants to re-generate (e.g., with squidpy analyses) or use the existing one.

3. **Check deps are installed:**
```bash
$MKSEG_PYTHON -c "import spatialdata, anndata, squidpy, geopandas; print('All scverse deps OK')"
```
If missing: `pip install spatialdata anndata scanpy squidpy geopandas`

---

## Step 1: Identify Input Data

Ask: *"Which detections file should I convert? And what cell type is it?"*

Look for detection files:
```bash
find <output_dir> -name "*_detections*.json" -type f 2>/dev/null | sort -t/ -k$(echo <output_dir> | tr '/' '\n' | wc -l) | head -10
```

Auto-detect cell type from filename (e.g., `cell_detections.json` -> cell, `vessel_detections.json` -> vessel).

Also check for:
- **Tiles directory** (`<run_dir>/tiles/`) — needed for polygon shape extraction from HDF5 masks
- **Classified detections** (`*_classified.json`) — preferred over raw detections (has marker classes)
- **Score threshold** — ask if the user wants to filter by `rf_prediction`

---

## Step 2: Configure Export

Ask the user about these options:

| Option | Default | What it does |
|--------|---------|-------------|
| **Shape extraction** | On | Extract polygon contours from HDF5 masks (circle fallback if unavailable) |
| **OME-Zarr image** | Auto-detect | Link slide image as lazy dask layer (no RAM) |
| **Squidpy analyses** | Off | Run spatial neighbors, Moran's I, neighborhood enrichment, co-occurrence, Ripley's |
| **Squidpy cluster key** | — | Which obs column to analyze (e.g., `tdTomato_class`, `GFP_class`, `cell_type`) |
| **Score threshold** | None | Filter detections by rf_prediction >= threshold |

For squidpy analyses, **a cluster key is required** — this should be a categorical column like a marker class. Ask:
*"Do you have marker classifications? Which column should squidpy analyze for co-location patterns?"*

Available columns can be discovered from the detections:
```bash
$MKSEG_PYTHON -c "
import json
d = json.load(open('<detections.json>'))
feats = d[0].get('features', {}) if d else {}
class_cols = [k for k in feats if k.endswith('_class')]
print('Classification columns:', class_cols or 'none found')
print('Top-level keys:', [k for k in d[0] if k.endswith('_class')]) if d else None
"
```

---

## Step 3: Run Conversion

**Basic conversion (table + shapes, no squidpy):**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/convert_to_spatialdata.py \
    --detections <detections.json> \
    --output <output>/<celltype>_spatialdata.zarr \
    --cell-type <celltype> \
    --tiles-dir <run_dir>/tiles \
    --overwrite
```

**With squidpy analyses:**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/convert_to_spatialdata.py \
    --detections <detections.json> \
    --output <output>/<celltype>_spatialdata.zarr \
    --cell-type <celltype> \
    --tiles-dir <run_dir>/tiles \
    --run-squidpy \
    --squidpy-cluster-key <marker_class_column> \
    --overwrite
```

**With OME-Zarr image link:**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/convert_to_spatialdata.py \
    --detections <detections.json> \
    --output <output>/<celltype>_spatialdata.zarr \
    --cell-type <celltype> \
    --tiles-dir <run_dir>/tiles \
    --zarr-image <slide>.ome.zarr \
    --run-squidpy \
    --squidpy-cluster-key <marker_class_column> \
    --overwrite
```

**Vessel pipeline** (uses JSON contour fields, no HDF5 masks needed):
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/convert_to_spatialdata.py \
    --detections <vessel_detections.json> \
    --output <output>/vessel_spatialdata.zarr \
    --cell-type vessel \
    --overwrite
```
Vessel produces 3 shape layers: `vessel_outer` (CD31 boundary), `vessel_lumen` (inner), `vessel_sma` (SMA ring).

---

## Step 4: Verify Output

Check the zarr store:
```bash
$MKSEG_PYTHON -c "
import spatialdata as sd
sdata = sd.read_zarr('<output>.zarr')
print(sdata)
print()
adata = sdata['table']
print(f'Table: {adata.n_obs} observations x {adata.n_vars} features')
print(f'obsm layers: {list(adata.obsm.keys())}')
print(f'obs columns: {list(adata.obs.columns)}')
if 'spatial' in adata.obsm:
    coords = adata.obsm['spatial']
    print(f'Spatial range: x=[{coords[:,0].min():.0f}, {coords[:,0].max():.0f}], y=[{coords[:,1].min():.0f}, {coords[:,1].max():.0f}] um')
"
```

If squidpy was run, also check:
```bash
ls -la <output>_squidpy/
# Should contain: morans_i.csv, nhood_enrichment.png, co_occurrence.png
```

Show the user: *"Your SpatialData is ready. Here's what's inside:"*
- **Table**: AnnData with morphological features in X, spatial coordinates in obsm, embeddings (SAM2/ResNet/DINOv2) in obsm
- **Shapes**: Polygon contours for each detection (or circles as fallback)
- **Images**: Linked OME-Zarr (if provided) — lazy/dask, never loaded to RAM

---

## Step 5: Show Usage Examples

Provide copy-paste Python snippets:

**Load and explore:**
```python
import spatialdata as sd
sdata = sd.read_zarr("<output>.zarr")
adata = sdata["table"]

# Quick look
print(adata)
print(adata.obs.head())
```

**Spatial scatter plot:**
```python
import squidpy as sq
sq.pl.spatial_scatter(adata, color="rf_prediction", size=3)
# or: color="tdTomato_class", color="area", etc.
```

**Custom spatial analysis:**
```python
import squidpy as sq

# Build spatial graph (if not already done with --run-squidpy)
sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=15)

# Neighborhood enrichment
sq.gr.nhood_enrichment(adata, cluster_key="tdTomato_class")
sq.pl.nhood_enrichment(adata, cluster_key="tdTomato_class")

# Moran's I (which features are spatially clustered?)
sq.gr.spatial_autocorr(adata, mode='moran')
print(adata.uns['moranI'].sort_values('I', ascending=False).head(10))
```

**Scanpy integration:**
```python
import scanpy as sc

# PCA + UMAP on morphological features
sc.pp.pca(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["area", "rf_prediction", "tdTomato_class"])

# Leiden clustering
sc.tl.leiden(adata)
sc.pl.umap(adata, color="leiden")
```

**Use embeddings directly:**
```python
# SAM2 embeddings for custom analysis
sam2 = adata.obsm["X_sam2"]  # (n_cells, 256)

# Or use as PCA input
import anndata as ad
adata_sam2 = ad.AnnData(X=sam2, obs=adata.obs)
sc.pp.neighbors(adata_sam2)
sc.tl.umap(adata_sam2)
```

---

## Rules

- Use `$MKSEG_PYTHON` as interpreter, `PYTHONPATH=$REPO`.
- All file paths should be absolute.
- SpatialData export is lightweight (CPU-only, ~30 seconds for 50k detections). No need for GPU or SLURM for the conversion itself.
- Squidpy analyses add ~1-2 minutes for typical datasets.
- The zarr store is self-contained — it can be copied/shared without the original CZI.
- For vessel cell type, shape extraction uses JSON contour fields (no tiles directory needed).
- If the user hasn't run detection yet, redirect to `/analyze`.

$ARGUMENTS
