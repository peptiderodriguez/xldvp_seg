You are guiding the user through laser microdissection (LMD) export for the xldvp_seg pipeline.

The export pipeline is fully automated — 384-well plate, serpentine well ordering, nearest-neighbor path optimization, clustering, and controls are all built-in defaults. Just confirm a few inputs and run.

---

## Default Behavior (always automatic)

- **Plate**: 384-well, serpentine order (B2->B3->C3->C2), skipping edge wells — 308 usable wells
- **Well ordering**: nearest-neighbor path optimization minimizes stage travel on slide
- **Clustering**: two-stage greedy (Round 1=500um, Round 2=1000um, target 375-425um2). Unclustered = singles (1 NMJ/well). Clusters = grouped cells (all in one well)
- **Controls**: negative control regions at 100um offset (8 directions), cluster controls preserve spatial arrangement
- **Contours**: pre-dilated (+0.5um) + RDP-simplified during detection post-dedup. Export uses them as-is
- **Capacity check**: pipeline warns early if detection count would exceed 308 wells before expensive processing
- **OME-Zarr**: auto-generated at end of pipeline (no separate conversion step needed)

---

## Step 1: Check Inputs

**Check detections exist:**
```bash
ls <output_dir>/*_detections*.json 2>/dev/null
```
If missing -> redirect to `/analyze` first.

**Check if RF classifier was run** (look for `rf_prediction` in detections):
```bash
$MKSEG_PYTHON -c "import json; d=json.load(open('<detections.json>')); print('rf_prediction' in (d[0].get('features') or d[0]))"
```
If True -> default `--min-score 0.5` filters to high-confidence detections.
If False -> export all detections (no filter).

**Check for reference crosses:**
```bash
ls <output_dir>/crosses*.json <output_dir>/reference_crosses.json 2>/dev/null
```

---

## Step 2: Place Reference Crosses (if needed)

Reference crosses are physical marks on the slide visible both in the microscope image and on the LMD stage. They register slide coordinates to stage coordinates.

**Option A — CZI-native Napari** (recommended, no conversion needed):
```bash
# Place 3 RGB-coded crosses directly on CZI
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i <czi_path> --channel 0 -o <crosses.json>

# With LMD7 display orientation (tissue-down + rotated)
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i <czi_path> --flip-horizontal --rotate-cw-90 -o <crosses.json>

# Start fresh (ignore existing crosses)
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i <czi_path> --fresh -o <crosses.json>
```

**Option B — OME-Zarr Napari** (for very large slides):
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i <output>.ome.zarr -o <crosses.json>
```

The OME-Zarr is auto-generated at the end of the pipeline run. Or convert manually:
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/czi_to_ome_zarr.py <czi_path> <output>.zarr
```

**Option C — Manual JSON** (SSH/headless):
```json
{"crosses": [{"id": 1, "x_px": 1000, "y_px": 2000}, {"id": 2, "x_px": 5000, "y_px": 2000}, {"id": 3, "x_px": 3000, "y_px": 8000}], "image_width_px": 10000, "image_height_px": 10000, "pixel_size_um": 0.22}
```
Coordinates in mosaic pixel space (same as detection centroids). 3 crosses required.

**Batch mode** (multiple slides):
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py \
    --czi-dir /data/slides --slides SlideA SlideB SlideC \
    --output-dir crosses/ --flip-horizontal --rotate-cw-90
```
Produces `{slide}_crosses.json` per slide. Skips slides with existing crosses unless `--fresh`.

---

## Step 3: Run Export

```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <detections.json> \
    --crosses <crosses.json> \
    --output-dir <output>/lmd \
    --generate-controls \
    --min-score 0.5 \
    --export
```

Drop `--min-score 0.5` if no RF classifier was run.

**Optional: erosion at export time** (shrink contours so laser cuts inside the target):
```bash
# Erode by 5% of sqrt(area)
--erode-pct 0.05

# Erode by absolute distance (0.2 um)
--erosion-um 0.2
```

**Multi-slide batch export:**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --input-dir /path/to/runs \
    --crosses-dir /path/to/crosses \
    --output-dir /path/to/lmd_batch \
    --generate-controls --min-score 0.5 --export
```
Discovers `*_detections.json` + matching `*_crosses.json`, exports per-slide, writes `batch_summary.json`.

**What the pipeline does automatically:**
1. Filters detections by score (if `--min-score` set)
2. Early capacity check — warns if >308 detections before any processing
3. Uses pre-computed contours from post-dedup phase
4. Applies export-time erosion (if `--erosion-um` or `--erode-pct`)
5. Two-stage clustering (500um -> 1000um)
6. Serpentine well assignment (B2->B3->C3->C2, 384-well, 308 max)
7. Nearest-neighbor ordering on slide to minimize stage travel
8. Generates spatial controls (100um offset, 8 directions)
9. Writes Leica LMD XML

---

## Step 4: Validate and Hand Off

```bash
ls -la <output>/lmd/
```

Show:
- XML file path (transfer this to LMD computer)
- Summary: N detections -> N wells (singles + clusters + controls)
- Warn if well count approaches 308

**Optional: verify in Napari:**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_view_lmd_export.py \
    --zarr <slide>.ome.zarr --export <output>/lmd
```

---

## Multiple Cell Types

Each cell type gets its own export:
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <output>/nmj_detections.json --crosses <crosses.json> \
    --output-dir <output>/lmd_nmj --generate-controls --min-score 0.5 --export

PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/run_lmd_export.py \
    --detections <output>/cell_detections.json --crosses <crosses.json> \
    --output-dir <output>/lmd_cell --generate-controls --min-score 0.5 --export
```

---

## Adaptive Guidance

After each step, review results and give targeted feedback:

**After input check (Step 1):**
- If RF scores exist: *"Classifier was run. Default --min-score 0.5 keeps high-confidence detections. Adjust lower (0.3) if you want more cells per well, or higher (0.7) for precision."*
- If no RF scores: *"No classifier was applied — all detections will be exported. If you have too many false positives, consider running /classify first to filter."*
- If detection count > 250: *"You have N detections — getting close to the 308-well capacity limit. With controls, each detection uses ~2 wells. Consider a score threshold or clustering to stay within plate capacity."*
- If detection count > 500: *"That's N detections — this will exceed the 308-well plate. You'll need a score threshold or clustering to reduce to ~150 detections (each needs a control well)."*

**After cross placement (Step 2):**
- If user places crosses very close together: *"Those crosses are clustered in one area. Spreading them across the tissue gives better calibration accuracy across the whole slide."*
- If using --flip-horizontal --rotate-cw-90: *"Good — LMD7 display transforms applied. The coordinate inversion is handled automatically at export. Make sure the tissue orientation in Napari matches what you'll see on the LMD stage."*

**After export (Step 3):**
- Report well utilization: *"Used N of 308 wells (X% capacity). N singles, N clustered, N controls."*
- If wells > 280: *"Running tight on plate capacity. If you need to re-export later with more detections, consider increasing --min-score to free up wells."*
- If clustering produced very uneven clusters (some >200 cells, some <10): *"Cluster sizes are quite uneven. This is normal for spatially irregular tissue. The small clusters may be isolated cells that didn't merge with neighbors."*
- Always confirm: *"Transfer the XML to the LMD computer and verify cross alignment before cutting."*

**Erosion guidance:**
- If user doesn't specify erosion: *"No erosion applied — contours include the 0.5 um dilation from post-dedup. The laser will cut at the dilated boundary. Add --erosion-um 0.2 or --erode-pct 0.05 if you want to cut inside the cell boundary."*
- If user sets both erosion flags: *"Both --erosion-um and --erode-pct are set. They'll be applied sequentially (absolute first, then percentage). Usually one or the other is enough."*

---

## Rules

- 384-well + serpentine + controls + clustering are always on by default — don't ask about them unless the user specifically wants to change something
- Use `$MKSEG_PYTHON` as interpreter, `PYTHONPATH=$REPO`
- All coordinates are [x, y] (horizontal, vertical)
- LMD export is CPU-only and fast (~seconds). No GPU or SLURM needed
- The XML must be transferred to the LMD instrument computer — confirm the transfer path
- CZI-native cross placement is the recommended default (no OME-Zarr conversion needed)
- OME-Zarr is auto-generated at end of pipeline runs (for viewing/verification)
- Give helpful guidance about plate capacity, erosion tradeoffs, and cross placement quality — but don't block the export

$ARGUMENTS
