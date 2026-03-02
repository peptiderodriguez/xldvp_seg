You are guiding the user through laser microdissection (LMD) export for the xldvp_seg pipeline.

---

## Prerequisites Check

1. **Detections must exist.** Look for `*_detections.json` or `*_detections_postdedup.json` in the output directory. If not found, tell the user to run `/analyze` first.

2. **Contour processing.** Contours are dilated (+0.5µm) and RDP-simplified during the detection pipeline's post-dedup phase — already done for any run with `--no-contour-processing` NOT set (the default). `run_lmd_export.py` uses these pre-computed contours and does NOT re-process. If the run used `--no-contour-processing`, contours are extracted fresh during export.

3. **Classifier scores** (optional but recommended). Check if `rf_prediction` field exists in detections. If not, ask if they want to filter by classifier score (requires `/classify` first). Without it, all detections are exported.

---

## Step 1: Reference Cross Placement

Ask: *"Do you have reference crosses placed already (a crosses.json file)?"*

**If no crosses yet:**

Option A — Interactive Napari (requires display, e.g., local workstation):
```bash
# Convert CZI to OME-Zarr first (if not done)
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/czi_to_ome_zarr.py <czi_path> <output>.zarr

# Place crosses interactively
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py <output>.zarr --output <crosses.json>
```

Option B — Manual (SSH/headless): Create crosses.json by hand:
```json
{"crosses": [[x1, y1], [x2, y2], [x3, y3]]}
```
Coordinates are in mosaic pixel space (same coordinate system as detection centroids). At least 3 crosses needed for affine registration to LMD stage coordinates.

---

## Step 2: Configure Export

Ask the user about:
- **Well plate format**: 384-well (default, max 308 wells, serpentine B2→B3→C3→C2) or 96-well (`--plate-format 96`)
- **Score threshold**: minimum `rf_prediction` to include — default 0.5 if classifier was run, or include all
- **Generate controls?** (recommended — negative control regions, 100µm offset, 8 directions, cluster controls preserve arrangement)
- **Dilation**: additional contour dilation in µm beyond what was done in post-dedup (default 0.5µm; set 0 if post-dedup dilation is sufficient)
- **Clustering**: two-stage greedy clustering is always run (Round 1=500µm, Round 2=1000µm, target 375–425µm²). Unclustered = singles.

---

## Step 3: Run LMD Export

**Standard export (with RF score filter and controls):**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <detections.json> \
    --crosses <crosses.json> \
    --output-dir <output>/lmd \
    --generate-controls \
    --score-threshold 0.5 \
    --export
```

**96-well plate:**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <detections.json> \
    --crosses <crosses.json> \
    --output-dir <output>/lmd \
    --generate-controls \
    --plate-format 96 \
    --score-threshold 0.5 \
    --export
```

**Export all detections (no score filter):**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <detections.json> \
    --crosses <crosses.json> \
    --output-dir <output>/lmd \
    --generate-controls \
    --export
```

The pipeline will:
1. Load detections (optionally filter by `rf_prediction >= score-threshold`)
2. Use pre-computed contours from post-dedup processing (or extract fresh if unavailable)
3. Two-stage clustering: Round 1=500µm, Round 2=1000µm, target 375–425µm²
4. Early capacity check — warns if detection count would exceed 308 wells (384-well) before expensive processing
5. Assign to wells (384-well serpentine: B2→B3→C3→C2)
6. Generate controls (100µm offset, 8 directions; cluster controls preserve spatial arrangement)
7. Write XML for Leica LMD instrument
8. Generate Napari visualization script

---

## Step 4: Validate Output

```bash
ls -la <output>/lmd/
```

Check:
- XML file exists (e.g., `lmd_export.xml`)
- Total well count ≤ 308 for 384-well (≤ 88 for 96-well)
- Show summary: N detections → N wells (N singles + N clusters + N controls)
- Show the path to XML for transfer to LMD instrument computer

Offer to view the export overlaid on the slide in Napari:
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_view_lmd_export.py \
    --zarr <slide>.zarr --export <output>/lmd
```

---

## Multiple Cell Types / Slides

Each cell type requires its own export (separate detections.json, separate output-dir, separate XML). If running multiple:
```bash
# Cell type 1
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <output>/nmj_detections.json --crosses <crosses.json> \
    --output-dir <output>/lmd_nmj --generate-controls --export

# Cell type 2
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <output>/cell_detections.json --crosses <crosses.json> \
    --output-dir <output>/lmd_cell --generate-controls --export
```

---

## Rules

- Use `$MKSEG_PYTHON` as interpreter, `PYTHONPATH=$REPO`.
- All coordinates in this pipeline are [x, y] (horizontal, vertical).
- LMD export is CPU-only and fast (~seconds for <10k detections). No GPU or SLURM needed.
- For large exports (>5k detections), still run on login node is fine — it's not memory-intensive.
- Max 308 wells per 384-well plate — the pipeline warns early if this limit would be exceeded.
- The crosses.json coordinate space must match the CZI mosaic pixel space (same as detection centroids).

$ARGUMENTS
