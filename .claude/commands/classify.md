You are guiding the user through annotation, classifier training, and feature exploration for the xldvp_seg pipeline.

---

## Step 1: Check Annotation Status

Ask: *"Do you have annotations already, or do you need to create them first?"*

**If no annotations yet:**
1. Find the detection HTML in the output directory (`<output>/html/index.html`)
2. Serve it: `$MKSEG_PYTHON $REPO/serve_html.py <output_dir>`
3. Explain the annotation interface:
   - Click green checkmark = real detection (positive)
   - Click red X = false positive (negative)
   - Progress bar shows how many you've annotated
   - Click "Export" to save annotations as JSON
4. For beginners: *"Aim for ~200+ annotations minimum, balanced between positive and negative. The more annotations, the better the classifier."*

---

## Step 2: Feature Selection

Explain the available feature subsets and help the user choose:

| `--feature-set` value | Dimensions | What it captures | When to use |
|----------------------|-----------|-----------------|-------------|
| `morph` | ~78 | Shape, size, intensity, texture | **Default** — nearly as good as all (F1=0.900 on NMJ benchmark) |
| `morph_sam2` | ~334 | + learned visual embeddings from SAM2 | When morphology alone isn't enough |
| `channel_stats` | ~78 + 15/ch | + per-channel intensity distributions | When marker expression matters (needs `--all-channels`) |
| `all` | ~6,478 | Everything: morph + SAM2 + ResNet + DINOv2 | Maximum accuracy (F1=0.909). Needs `--extract-deep-features` during detection |

**Ask:** *"Which features were extracted during detection?"*
- SAM2 embeddings: always extracted (256D)
- Per-channel stats: only if `--all-channels` was used
- ResNet + DINOv2: only if `--extract-deep-features` was used

**Offer comparison:** *"Want to run a systematic comparison of feature subsets? This takes ~1 minute and shows which combination works best for your data."*

If yes:
```bash
$MKSEG_PYTHON $REPO/scripts/compare_feature_sets.py \
    --detections <detections.json> \
    --annotations <annotations.json> \
    --output-dir <output>/feature_comparison
```
This runs 5-fold CV on each feature combination and outputs a ranked table of F1/precision/recall.

---

## Step 3: Train + Apply

**Train the RF classifier:**
```bash
$MKSEG_PYTHON $REPO/train_classifier.py \
    --detections <detections.json> \
    --annotations <annotations.json> \
    --output-dir <output> \
    --feature-set <chosen_set>
```

Show the output: cross-validation scores, top 20 feature importances, saved model path.

**Apply to all detections:**
```bash
$MKSEG_PYTHON $REPO/scripts/apply_classifier.py \
    --detections <detections.json> \
    --classifier <output>/rf_classifier.pkl \
    --output <output>/<celltype>_detections_scored.json
```

**Regenerate HTML with threshold:**
```bash
$MKSEG_PYTHON $REPO/scripts/regenerate_html.py \
    --detections <output>/<celltype>_detections_scored.json \
    --czi-path <czi_path> \
    --output-dir <output> \
    --score-threshold 0.5
```

---

## Step 4: Feature Exploration (optional)

Ask: *"Want to explore the feature space with dimensionality reduction?"*

**UMAP + HDBSCAN clustering:**
```bash
$MKSEG_PYTHON $REPO/scripts/cluster_by_features.py \
    --detections <detections.json> \
    --output-dir <output>/clustering \
    --feature-groups "morph,sam2"
```

Available feature groups: `morph`, `shape`, `color`, `sam2`, `channel`, `deep`

Outputs:
- `umap_plot.png` — 2D projection colored by cluster
- `cluster_summary.csv` — per-cluster statistics
- `spatial.h5ad` — AnnData format for scanpy
- `detections_clustered.json` — detections with cluster assignments

For beginners: *"UMAP is a way to project high-dimensional features into 2D so you can see if cells form natural groups. HDBSCAN automatically finds those groups (clusters). This helps you discover cell subtypes you might not have known about."*

---

## Step 5: SpatialData Export (optional)

Ask: *"Want to export the classified detections to SpatialData for scverse ecosystem analysis (squidpy, scanpy)?"*

This is especially useful after classification because the marker classes (e.g., `tdTomato_class`) enable neighborhood enrichment and co-occurrence analyses.

```bash
$MKSEG_PYTHON $REPO/scripts/convert_to_spatialdata.py \
    --detections <scored_or_classified_detections.json> \
    --output <output>/<celltype>_spatialdata.zarr \
    --tiles-dir <output>/tiles \
    --run-squidpy \
    --squidpy-cluster-key <marker_class_column> \
    --overwrite
```

Outputs:
- `*_spatialdata.zarr/` — SpatialData zarr store with AnnData table, polygon shapes, embeddings
- `*_spatialdata_squidpy/morans_i.csv` — spatially autocorrelated features
- `*_spatialdata_squidpy/nhood_enrichment.png` — which cell types co-locate
- `*_spatialdata_squidpy/co_occurrence.png` — type co-occurrence at multiple distances

For beginners: *"SpatialData is the standard format for the scverse spatial analysis ecosystem. It lets you use squidpy for spatial statistics and scanpy for single-cell-style analysis on your detections."*

---

## Rules

- Use `$MKSEG_PYTHON` as the Python interpreter and set `PYTHONPATH=$REPO`.
- All file paths should be absolute.
- If the user hasn't run detection yet, redirect them to `/analyze`.
- If detection was run without `--extract-deep-features`, don't offer deep feature sets — explain they'd need to re-run detection to get those.

$ARGUMENTS
