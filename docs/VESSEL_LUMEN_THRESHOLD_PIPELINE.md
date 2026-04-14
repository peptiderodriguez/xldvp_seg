# Vessel Lumen Threshold Detection Pipeline

Threshold-based vessel lumen detection on OME-Zarr pyramids for whole-mount tissue sections. Uses Gaussian local-mean thresholding + seeded watershed at multiple scales, marker-cell validation, RF classification, and per-marker wall-cell assignment for LMD replicate sampling.

## When to use this

| Approach | Best for | Script |
|----------|----------|--------|
| **Threshold lumens** (this guide) | Whole-mount cross-sections with OME-Zarr. Reproducible, no GPU needed. | `detect_vessel_lumens_threshold.py` |
| SAM2 lumens | Flexible shapes, oblique cuts. Needs GPU. | `segment_vessel_lumens.py` |
| Graph topology | Longitudinal sections, strips, collapsed vessels. | `detect_vessel_structures.py` |

## Pipeline overview

```
detect_vessel_lumens_threshold.py  →  vessel_lumens_threshold.json
                                            ↓
generate_lumen_annotation.py       →  annotation HTML (Y/N cards)
                                            ↓  (user annotates)
score_vessel_lumens.py             →  vessel_lumens_scored.json
  (optional --cells for wall assignment)     vessel_lumens_final.json
                                            ↓
assign_vessel_wall_cells.py        →  vessel_lumens_with_cells.json
  (standalone, or integrated via --cells)    (per-marker UIDs + replicates)
```

## Step 1: Detection

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON scripts/detect_vessel_lumens_threshold.py \
    --zarr-path slide.ome.zarr \
    --scales 4,8,16,64 \
    --discovery-scales 64,16,8,4 \
    --block-size-um 2400 \
    --threshold-fraction 0.5 \
    --fill-expansion 1.5 \
    --min-area-um2 50 \
    --marker-cells-json cell_detections_snr2_markers.json \
    --marker-classes "SMA,LYVE1" \
    --min-marker-cells 6 \
    --output-dir lumens_out/ \
    --save-debug \
    --czi-path slide.czi \
    --display-channels 1,2,0 \
    --channel-names "SMA,LYVE1,nuc" \
    --skip-viewer
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scales` | required | Pyramid scales to use (comma-separated, e.g. `4,8,16,64`) |
| `--discovery-scales` | all scales | Which scales discover new lumens (vs refine-only) |
| `--block-size-um` | 2400 | Gaussian sigma for local mean threshold (um). Tuned for whole-mount cross-sections. Reduce to 100-500 for tight tissue or small vessel panels. |
| `--threshold-fraction` | 0.5 | Pixel must be < this fraction of local mean to be "dark" |
| `--fill-expansion` | 1.5 | Growth threshold = seed_threshold * fill_expansion (watershed) |
| `--marker-classes` | `SMA,CD31` | Marker names for pre-filter (looks for `{NAME}_class == positive`) |
| `--min-marker-cells` | 1 | Minimum marker+ cells in ring to keep for refinement |

**SLURM:** CPU-only, `p.hpcl8`, 370G RAM, 24h for whole-mount slides.

**Output:** `vessel_lumens_threshold.json` — all lumens with inline features (morph + per-channel stats), contour coordinates, `n_marker_wall`.

## Step 2: Annotation

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON scripts/generate_lumen_annotation.py \
    --lumens lumens_out/vessel_lumens_threshold.json \
    --zarr-path slide.ome.zarr \
    --output-dir lumens_out/annotation_v1/ \
    --display-channels 1,2,0 \
    --channel-names "SMA,LYVE1,nuc" \
    --pixel-size-um 0.1725 \
    --per-page 50 \
    --title "Vessel Lumen Annotation"
```

After RF scoring (step 3), regenerate sorted by score:

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON scripts/generate_lumen_annotation.py \
    --lumens lumens_out/vessel_lumens_scored.json \
    --zarr-path slide.ome.zarr \
    --output-dir lumens_out/annotation_v2_scored/ \
    --sort-by rf_score --sort-descending \
    ...same channel args...
```

Open `index.html` in browser. Label Y/N per lumen. Export annotations JSON.

## Step 3: RF scoring + filtering

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON scripts/score_vessel_lumens.py \
    --lumens lumens_out/vessel_lumens_threshold.json \
    --annotations lumens_out/annotation_v1/annotations.json \
    --output-dir lumens_out/scored/ \
    --cells cell_detections_snr2_markers.json \
    --markers "SMA,LYVE1" \
    --rf-threshold 0.75 \
    --min-marker-cells 8
```

**What it does:**
1. Trains RF on your annotations (5-fold CV F1 reported)
2. Scores all lumens (`rf_score`, `rf_prediction`)
3. If `--cells` + `--markers` provided: runs per-marker wall-cell assignment
4. Filters: `rf_score >= threshold` AND `>=N cells of any marker` + annotated-positive rescue - annotated-negative exclusion

**Outputs:**
- `vessel_lumens_scored.json` — all lumens with scores
- `vessel_lumens_final.json` — filtered validated set
- `vessel_lumen_rf.joblib` — trained model

## Step 4: Wall-cell assignment (standalone)

If you didn't pass `--cells` to the scoring script, or want to re-run with different parameters:

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON scripts/assign_vessel_wall_cells.py \
    --vessels lumens_out/scored/vessel_lumens_final.json \
    --cells cell_detections_snr2_markers.json \
    --markers "SMA,LYVE1" \
    --replicate-size 8 \
    --output lumens_out/vessel_lumens_with_cells.json
```

**Per-marker fields added:** `n_{MARKER}_wall`, `{MARKER}_wall_cell_uids` (brightest-first), `n_{MARKER}_replicates`, `n_replicates_total`.

## Algorithm details

### Discovery (per scale, coarse to fine)
1. Percentile-normalize each channel, sum to single signal
2. Gaussian blur (5um) for noise
3. Local Gaussian-mean threshold with seeded watershed:
   - Seed: `signal < local_mean * threshold_fraction`
   - Growth: `signal < local_mean * (threshold_fraction * fill_expansion)`
   - Watershed fills seeds, stops at bright walls
4. Morphological opening (disk, 5um)
5. Label + size filter + exclusion bitmap from previous scales

### Refinement
Fast median threshold: `signal < crop_median * fraction OR signal < global_p10_floor`. Single multi-channel zarr read per lumen.

### Marker pre-filter
KD-tree of marker+ cells. Lumens with fewer than `--min-marker-cells` in ring `[equiv_r*0.5, equiv_r+30um]` are dropped before refinement. This is the primary speedup (94K to ~1K candidates on a typical slide).

## Repeating on a new slide

1. `czi_info.py slide.czi` — confirm channel indices
2. Detect cells: `xlseg detect --marker-snr-channels "MARKER1:N,MARKER2:M"`
3. Copy and adapt the SLURM template (paths, `--marker-classes`, `--display-channels`)
4. Run detection (step 1)
5. Generate annotation pages (step 2), annotate ~300-500 lumens
6. Score + filter (step 3)
7. Review final viewer, iterate annotations if needed
