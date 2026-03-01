You are guiding the user through laser microdissection (LMD) export for the xldvp_seg pipeline.

---

## Prerequisites Check

1. **Detections must exist.** Look for `*_detections.json` in the output directory. If not found, tell the user to run `/analyze` first.

2. **Classifier scores** (optional but recommended). Check if `rf_prediction` field exists in detections. If not, ask if they want to filter by classifier score (requires `/classify` first).

---

## Step 1: Reference Cross Placement

Ask: *"Do you have reference crosses placed already (a crosses.json file)?"*

**If no crosses yet:**

Option A — If the user has a Napari-capable display:
```bash
# Convert CZI to OME-Zarr first (if not done)
$MKSEG_PYTHON $REPO/scripts/czi_to_ome_zarr.py <czi_path> <output>.zarr

# Place crosses interactively
$MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py <output>.zarr --output <crosses.json>
```

Option B — If no display (e.g., SSH to cluster): Explain that crosses.json can be created manually:
```json
{"crosses": [[x1, y1], [x2, y2], [x3, y3]]}
```
Coordinates are in mosaic pixel space. At least 3 crosses needed for affine registration.

---

## Step 2: Configure Export

Ask about:
- **Well plate format**: 384-well (default, max 308 wells) or 96-well
- **Generate controls?** (recommended — negative control regions offset from each detection)
- **Score threshold**: minimum `rf_prediction` to include (default 0.5, or include all if no classifier)
- **Dilation**: contour dilation in µm (default 0.5 µm beyond cell boundary)

---

## Step 3: Run LMD Export

```bash
$MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <detections.json> \
    --crosses <crosses.json> \
    --output-dir <output>/lmd \
    --generate-controls \
    --export
```

The pipeline will:
1. Load detections and H5 masks
2. Extract contours, dilate +0.5µm, simplify with RDP (epsilon=5px)
3. Two-stage clustering: Round 1 = 500µm, Round 2 = 1000µm, target 375-425µm²
4. Assign to wells (384-well serpentine: B2→B3→C3→C2, max 308 wells)
5. Generate controls (100µm offset, 8 directions)
6. Write XML for Leica LMD instrument
7. Generate Napari visualization script

---

## Step 4: Validate Output

Check:
- XML file exists and is valid
- Total well count (warn if >308 for 384-well)
- Show summary: N detections → N wells (N singles + N clusters)
- Show path to XML for transfer to LMD instrument

Offer to view the export in Napari:
```bash
$MKSEG_PYTHON $REPO/scripts/napari_view_lmd_export.py <output>/lmd
```

---

## Rules

- Use `$MKSEG_PYTHON` as interpreter, `PYTHONPATH=$REPO`.
- All coordinates in this pipeline are [x, y] (horizontal, vertical).
- Never run heavy processing on the SLURM login node — use `sbatch` for the export if the slide is large.
- The LMD export agent (`.claude/agents/lmd-export.md`) has additional details for complex cases like quadrant selection and custom well ordering.

$ARGUMENTS
