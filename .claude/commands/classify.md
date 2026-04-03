You are guiding the user through annotation, classifier training, and feature exploration for the xldvp_seg pipeline.

---

## Step 1: Check Annotation Status

Ask: *"Do you have annotations already, or do you need to create them first?"*

**If no annotations yet:**
1. Find the detection HTML in the output directory (`<output>/html/index.html`)
2. Serve it: `$XLDVP_PYTHON $REPO/serve_html.py <output_dir>`
3. Explain the annotation interface:
   - Click green checkmark = real detection (positive)
   - Click red X = false positive (negative)
   - Progress bar shows how many you've annotated
   - Click "Export" to save annotations as JSON
4. For beginners: *"Aim for ~200+ annotations minimum, balanced between positive and negative. The more annotations, the better the classifier."*

---

## Step 1b: Background Correction

**Background correction is automatic.** The pipeline performs pixel-level background correction during detection (post-dedup phase). All `ch{N}_*` features (mean, std, percentiles, etc.) are extracted from corrected pixels. Each cell has:
- `ch{N}_background` — local background estimate (median of k=30 nearest neighbors)
- `ch{N}_snr` — signal-to-noise ratio
- `ch{N}_mean_raw`, `ch{N}_std_raw`, etc. — uncorrected originals

**Double correction is impossible.** `classify_markers.py` auto-detects pipeline-corrected data (via `ch{N}_background` keys) and disables ALL its own bg subtraction — both `--correct-all-channels` and per-marker `bg_subtract` (including the `otsu` method's default). No user action needed.

**For older detections (pre-Mar 2026) only:** Use `--correct-all-channels`:
```bash
$XLDVP_PYTHON $REPO/scripts/classify_markers.py \
    --detections <detections.json> \
    --marker-channel 1,2 --marker-name NeuN,tdTomato \
    --correct-all-channels
```

---

## Step 2: Feature Selection

*Why this matters:* The RF classifier only sees the features that were extracted at detection time. Choosing a larger feature set isn't always better — 78 morphological features reach F1=0.900, while the full 6,478-feature set reaches F1=0.909 on the NMJ benchmark. The extra complexity rarely justifies itself unless morph alone is clearly failing.

| `--feature-set` value | Dimensions | What it captures | When to use |
|----------------------|-----------|-----------------|-------------|
| `morph` | ~78 | Shape, size, intensity, texture | **Start here** — fast to train (~10s), nearly as accurate as everything else |
| `morph_sam2` | ~334 | + SAM2 visual embeddings (global shape context) | When cells look similar in size/shape but differ visually |
| `channel_stats` | ~15/ch | Per-channel intensity distributions | When marker expression (bright vs dim) is the key discriminator — needs `--all-channels` |
| `all` | ~6,478 | Everything: morph + SAM2 + ResNet + DINOv2 | When nothing else works; needs `--extract-deep-features` during detection |

**Check what was extracted:**
- SAM2 (256D): always present
- Per-channel stats (~15/ch): only if `--all-channels` was used during detection
- ResNet (4096D) + DINOv2 (2048D): only if `--extract-deep-features` was used — if not, `all` falls back to morph+SAM2

**Run a 1-minute comparison first** — this tells you definitively whether SAM2 or channel features help for *your specific data* before committing to a feature set:
```bash
$XLDVP_PYTHON $REPO/scripts/compare_feature_sets.py \
    --detections <detections.json> \
    --annotations <annotations.json> \
    --output-dir <output>/feature_comparison
```
Outputs a ranked table of F1/precision/recall per subset. If `morph` and `morph_sam2` are within 0.01 F1 of each other, use `morph` — simpler is more robust on new slides.

---

## Step 3: Train + Apply

**Train the RF classifier:**
```bash
$XLDVP_PYTHON $REPO/train_classifier.py \
    --detections <detections.json> \
    --annotations <annotations.json> \
    --output-dir <output> \
    --feature-set <chosen_set>
```

Show the output: cross-validation scores, top 20 feature importances, saved model path.

**Apply to all detections:**
```bash
$XLDVP_PYTHON $REPO/scripts/apply_classifier.py \
    --detections <detections.json> \
    --classifier <output>/rf_classifier.pkl \
    --output <output>/<celltype>_detections_scored.json
```

**Regenerate HTML with threshold:**
```bash
$XLDVP_PYTHON $REPO/scripts/regenerate_html.py \
    --detections <output>/<celltype>_detections_scored.json \
    --czi-path <czi_path> \
    --output-dir <output> \
    --score-threshold 0.5
```
*Why 0.5?* The RF outputs a probability (0=definitely false positive, 1=definitely real). 0.5 is the natural decision boundary. Increase to 0.7–0.8 if you want higher precision (fewer false positives, but miss some real cells). Decrease to 0.3 if recall matters more (catch everything, accept more noise). Check the score distribution first: `$XLDVP_PYTHON -c "import json; d=json.load(open('<scored.json>')); scores=[x.get('rf_prediction',0) for x in d]; print(f'mean={sum(scores)/len(scores):.2f}, >0.5: {sum(1 for s in scores if s>0.5)}/{len(scores)}')`

---

## Step 4: Feature Exploration (optional)

Ask: *"Want to explore the feature space with dimensionality reduction?"*

**UMAP + HDBSCAN clustering** (core: `xldvp_seg.analysis.cluster_features`):
```bash
# Via CLI subcommand
xlseg cluster --detections <detections.json> \
    --output-dir <output>/clustering \
    --feature-groups "morph,sam2"

# Or directly
$XLDVP_PYTHON $REPO/scripts/cluster_by_features.py \
    --detections <detections.json> \
    --output-dir <output>/clustering \
    --feature-groups "morph,sam2"
```

Available feature groups: `morph`, `shape`, `color`, `sam2`, `channel`, `deep`

Outputs:
- `umap_plot.png` — 2D projection colored by cluster
- `cluster_summary.csv` — per-cluster statistics
- `detections_clustered.json` — detections with cluster assignments
- SpatialData zarr also auto-exported from detection — see Step 5

For beginners: *"UMAP is a way to project high-dimensional features into 2D so you can see if cells form natural groups. HDBSCAN automatically finds those groups (clusters). This helps you discover cell subtypes you might not have known about."*

---

## Step 5: SpatialData Export (optional)

Ask: *"Want to export the classified detections to SpatialData for scverse ecosystem analysis (squidpy, scanpy)?"*

This is especially useful after classification because the marker classes (e.g., `tdTomato_class`) enable neighborhood enrichment and co-occurrence analyses.

```bash
$XLDVP_PYTHON $REPO/scripts/convert_to_spatialdata.py \
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

## Adaptive Guidance

After each step, review the results and give targeted feedback:

**After annotation check (Step 1):**
- < 100 annotations: *"That's a small training set. Aim for 200+ with balanced pos/neg for a reliable classifier. More annotations almost always help."*
- Highly imbalanced (>80% one class): *"Your annotations are pretty skewed. The RF handles imbalance well, but try to annotate more of the minority class if possible."*

**After feature comparison (Step 2):**
- If `morph` and `morph_sam2` are within 0.01 F1: recommend `morph` — simpler, faster, more robust across slides.
- If `morph_sam2` is >0.02 better: recommend it — SAM2 embeddings capture visual context that shape stats miss.
- If all sets underperform (F1 < 0.80): suggest more annotations, check for annotation noise, or consider `--extract-deep-features` for the next detection run.

**After training (Step 3):**
- F1 > 0.90: *"Excellent classifier. Proceed to scoring."*
- F1 0.85-0.90: *"Solid. This should work well for filtering."*
- F1 0.80-0.85: *"Decent but there's room to improve. Consider annotating another 100-200 detections, especially borderline cases."*
- F1 < 0.80: Diagnose — is it too few annotations, class imbalance, or genuinely hard cases? Suggest concrete next steps.
- Check precision vs recall: if precision >> recall, suggest lowering score threshold (0.3 instead of 0.5). If recall >> precision, suggest a second annotation round on false positives.

**After scoring (Step 3):**
- Check score distribution. If bimodal (peaks near 0 and 1): good separation. If most scores cluster around 0.5: the classifier is uncertain — more annotations or better features needed.
- Report: *"X detections scored > 0.5 out of Y total (Z%)."*

**General:**
- If the user wants to try deep features and detection was run without `--extract-deep-features`: explain they'd need to re-run detection (checkpointed, so it resumes from dedup). Don't gatekeep — it's a reasonable thing to try.
- If exploring features with UMAP reveals unexpected clusters: *"Interesting — there might be morphological subtypes here. Want to annotate by cluster to see if there are biologically meaningful groups?"*

## Rules

- Use `$XLDVP_PYTHON` as the Python interpreter and set `PYTHONPATH=$REPO`.
- All file paths should be absolute.
- If the user hasn't run detection yet, redirect them to `/analyze`.
- If detection was run without `--extract-deep-features`, don't block — explain re-detection is needed but it resumes from checkpoints.

$ARGUMENTS
