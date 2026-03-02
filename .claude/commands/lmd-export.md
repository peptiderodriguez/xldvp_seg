You are guiding the user through laser microdissection (LMD) export for the xldvp_seg pipeline.

The export pipeline is fully automated — 384-well plate, serpentine well ordering, nearest-neighbor path optimization, clustering, and controls are all built-in defaults. Just confirm a few inputs and run.

---

## Default Behavior (always automatic)

- **Plate**: 384-well, serpentine order (B2→B3→C3→C2), skipping edge wells — 308 usable wells
- **Well ordering**: nearest-neighbor path optimization minimizes stage travel on slide
- **Clustering**: two-stage greedy (Round 1=500µm, Round 2=1000µm, target 375–425µm²). Unclustered = singles (1 NMJ/well). Clusters = grouped cells (all in one well)
- **Controls**: negative control regions at 100µm offset (8 directions), cluster controls preserve spatial arrangement
- **Contours**: pre-dilated (+0.5µm) + RDP-simplified during detection post-dedup. Export uses them as-is
- **Capacity check**: pipeline warns early if detection count would exceed 308 wells before expensive processing

---

## Step 1: Check Inputs

**Check detections exist:**
```bash
ls <output_dir>/*_detections*.json 2>/dev/null
```
If missing → redirect to `/analyze` first.

**Check if RF classifier was run** (look for `rf_prediction` in detections):
```bash
$MKSEG_PYTHON -c "import json; d=json.load(open('<detections.json>')); print('rf_prediction' in (d[0].get('features') or d[0]))"
```
If True → default `--min-score 0.5` filters to high-confidence detections.
If False → export all detections (no filter).

**Check for reference crosses:**
```bash
ls <output_dir>/crosses*.json <output_dir>/reference_crosses.json 2>/dev/null
```

---

## Step 2: Place Reference Crosses (if needed)

Reference crosses are physical marks on the slide visible both in the microscope image and on the LMD stage. They register slide coordinates to stage coordinates.

**Option A — Interactive Napari** (requires display):
```bash
# Convert CZI to OME-Zarr (lazy pyramids, no RAM issue)
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/czi_to_ome_zarr.py <czi_path> <output>.zarr

# Place 3–4 crosses at tissue corners/landmarks
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_place_crosses.py <output>.zarr --output <crosses.json>
```

**Option B — Manual JSON** (SSH/headless):
```json
{"crosses": [[x1, y1], [x2, y2], [x3, y3]]}
```
Coordinates in mosaic pixel space (same as detection centroids). 3 crosses minimum.

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

**What the pipeline does automatically:**
1. Filters detections by score (if `--min-score` set)
2. Early capacity check — warns if >308 detections before any processing
3. Uses pre-computed contours from post-dedup phase
4. Two-stage clustering (500µm → 1000µm)
5. Serpentine well assignment (B2→B3→C3→C2, 384-well, 308 max)
6. Nearest-neighbor ordering on slide to minimize stage travel
7. Generates spatial controls (100µm offset, 8 directions)
8. Writes Leica LMD XML

---

## Step 4: Validate and Hand Off

```bash
ls -la <output>/lmd/
```

Show:
- XML file path (transfer this to LMD computer)
- Summary: N detections → N wells (singles + clusters + controls)
- Warn if well count approaches 308

**Optional: verify in Napari:**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/napari_view_lmd_export.py \
    --zarr <slide>.zarr --export <output>/lmd
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

## Rules

- 384-well + serpentine + controls + clustering are always on by default — don't ask about them unless the user specifically wants to change something
- Use `$MKSEG_PYTHON` as interpreter, `PYTHONPATH=$REPO`
- All coordinates are [x, y] (horizontal, vertical)
- LMD export is CPU-only and fast (~seconds). No GPU or SLURM needed
- The XML must be transferred to the LMD instrument computer — confirm the transfer path

$ARGUMENTS
