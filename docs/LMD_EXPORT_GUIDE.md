# LMD Export Guide

Export scored cell detections to Leica Laser Microdissection (LMD) XML format for automated tissue collection.

## Overview

After running segmentation and scoring detections with a trained classifier, this workflow exports high-confidence detections to Leica LMD XML. The pipeline supports:

- **Napari cross placement** -- interactive calibration point placement on CZI slides
- **Score-based filtering** -- export only detections above a classifier confidence threshold
- **Spatial clustering** -- group nearby detections into wells
- **384-well plate layout** -- serpentine well assignment (max 308 wells/plate, multi-plate overflow)
- **Contour erosion** -- shrink contours so the laser cuts inside the cell boundary
- **Batch export** -- process multiple slides in one command
- **Replicate-based export** -- area-normalized replicates for DVP proteomics

## Prerequisites

1. **py-lmd library**:
   ```bash
   pip install py-lmd
   ```
2. **Scored detections JSON** from `apply_classifier.py` (contains `rf_prediction` scores)
3. **CZI slide** for cross placement in Napari

## Workflow

### Step 1: Place Reference Crosses in Napari

Three calibration crosses align the microscope stage coordinates with your image. Place them at landmarks visible under the LMD microscope.

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i /path/to/slide.czi --flip-horizontal \
    -o /path/to/output/reference_crosses.json
```

**Display transforms:** `--rotate-cw-90` is ON by default (LMD7 tissue-down orientation). Use `--no-rotate-cw-90` to disable. `--flip-horizontal` must be specified explicitly.

**Keybinds:**

| Key | Action |
|-----|--------|
| R / 1 | Select cross 1 (red) |
| G / 2 | Select cross 2 (green) |
| B / 3 | Select cross 3 (cyan) |
| Space / P | Place cross at cursor |
| U | Undo last placement |
| S | Save crosses |
| Q | Save and quit |

**Tips:** Choose landmarks visible in both Napari and the LMD microscope. Spread crosses across the tissue. Do not place crosses on cells you intend to collect. Exactly 3 crosses are used for calibration.

Optional contour overlay: add `--contours /path/to/detections.json --color-by well` to verify detections while placing crosses.

#### Crosses JSON and display_transform

The saved JSON stores image dimensions, pixel size (from CZI metadata), cross coordinates in display space, and a `display_transform` block recording which transforms were active during placement (`flip_horizontal`, `rotate_cw_90`). The export script reads `display_transform` and applies the same transforms to detection contours so they match the cross coordinate space. No manual coordinate conversion is needed.

### Step 2: Export to LMD XML

#### Single-slide export

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_lmd_export.py \
    --detections /path/to/scored_detections.json \
    --crosses /path/to/reference_crosses.json \
    --output-dir /path/to/output/lmd \
    --min-score 0.5 \
    --generate-controls \
    --export
```

Key flags:
- `--min-score 0.5` -- only export detections with `rf_prediction >= 0.5`
- `--generate-controls` -- create paired spatial control wells (offset from each target)
- `--export` -- write the LMD XML file (without this, only summary is printed)

#### Batch export (multiple slides)

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_lmd_export.py \
    --input-dir /path/to/runs \
    --crosses-dir /path/to/crosses \
    --output-dir /path/to/output/lmd \
    --min-score 0.5 \
    --generate-controls \
    --export
```

The batch mode discovers `*_detections.json` files in `--input-dir` and matches each to its `*_crosses.json` in `--crosses-dir` by slide name prefix. Slides without exactly 3 crosses are skipped with a warning.

### Contour Processing

Contour simplification, dilation, and erosion are all applied at **LMD export time** (not during detection). The pipeline stores original mask contours as `contour_px` / `contour_um`; the export script processes them for LMD hardware:

Both simplification and dilation use **adaptive** tolerances by default (5% each). These are independently adjustable:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-area-change-pct` | **5.0** | Adaptive RDP: max symmetric-difference deviation (%). Set 0 for fixed `--rdp-epsilon`. |
| `--max-dilation-area-pct` | **5.0** | Adaptive dilation: max area increase (%) for laser buffer. Set 0 for fixed `--dilation-um`. |
| `--dilation-um` | 0.5 | Fixed dilation distance (only used when `--max-dilation-area-pct 0`) |
| `--rdp-epsilon` | 5.0 | Fixed RDP epsilon in pixels (only used when `--max-area-change-pct 0`) |
| `--erosion-um` | 0.0 | Shrink contours by absolute distance in um |
| `--erode-pct` | 0.0 | Shrink contours by percentage of sqrt(area) (e.g., 0.05 = 5%) |

Processing order: simplify (adaptive RDP) → dilate (adaptive buffer) → erode (optional). Erosion is applied last. Use `--erosion-um 0.2` or `--erode-pct 0.05` to ensure the laser cuts inside the cell boundary.

**Why adaptive?** Fixed dilation (0.5 um) adds ~17% area to a typical 100 um² cell — large enough to distort area measurements. Adaptive dilation caps the area increase per cell, giving small cells a smaller buffer and large cells a larger one.

### Well Assignment

384-well plate, 4 quadrants, serpentine ordering: B2 -> B3 -> C3 -> C2. Max 308 wells per plate. Multi-plate overflow is automatic when detections exceed plate capacity.

When `--generate-controls` is enabled, empty QC wells (~10%) are inserted evenly across the plate. Each target well gets a paired control well containing tissue from the same neighborhood but offset by `--control-offset-um` (default: 100 um).

### Spatial Clustering (optional)

To group nearby detections into shared wells:

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/cluster_detections.py \
    --detections /path/to/scored_detections.json \
    --min-score 0.5 \
    --output /path/to/clusters.json

PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_lmd_export.py \
    --detections /path/to/scored_detections.json \
    --crosses /path/to/reference_crosses.json \
    --clusters /path/to/clusters.json \
    --output-dir /path/to/output/lmd \
    --generate-controls --export
```

Note: when `--clusters` is provided, the cluster file's own score filtering applies. Do not also pass `--min-score` (it would double-filter).

### Zone Filtering (optional)

If detections have zone assignments (from `assign_tissue_zones.py`):

```bash
--zone-filter "1,3,5"    # include only these zone IDs
--zone-exclude "0"        # exclude these zone IDs
```

## Replicate-Based Export (DVP Proteomics)

For proteomics workflows requiring area-normalized biological replicates across slides:

**Step 1 -- Select cells and assign replicates:**
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/examples/bone_marrow/select_mks_for_lmd.py \
    --score-threshold 0.80 \
    --target-area 10000 \
    --max-replicates 4
```

This produces `lmd_replicates_full.json` with per-cell well assignments and replicate grouping.

**Step 2 -- Export XML from pre-assigned wells:**
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/lmd_export_replicates.py \
    --sampling-results lmd_replicates_full.json \
    --contours-json mk_contours_overlay.json \
    --crosses-dir ./crosses \
    --output-dir ./xml
```

The replicate export reads `display_transform` from each slide's crosses file and applies matching coordinate transforms to contours. Pixel size is read from crosses metadata or replicate data (never hardcoded).

## Coordinate System

The pipeline handles coordinate transforms automatically:

1. **Napari placement** -- crosses are placed in display space (after flip/rotate transforms)
2. **Crosses JSON** -- stores `display_transform` metadata (`flip_horizontal`, `rotate_cw_90`)
3. **Export** -- applies `_transform_native_to_display()` to contours, matching them to the cross coordinate space
4. **LMD Y-flip** -- final Y-axis inversion for stage coordinates (disable with `--no-flip-y`)

After CW90 rotation, the display height equals the original CZI width. The export script handles this swap automatically.

## Output Files

| File | Description |
|------|-------------|
| `shapes.xml` | Leica LMD XML with calibration points and all contours |
| `shapes.csv` | Coordinates with well/cluster assignments |
| `shapes_cluster_summary.json` | Cluster details (when `--clusters` is used) |

## Complete Example

```bash
# 1. Detect
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type cell --channel-spec "cyto=PM,nuc=488" \
    --all-channels --output-dir /path/to/output

# 2. Annotate in HTML viewer, then train + score
$XLDVP_PYTHON $REPO/serve_html.py /path/to/output
# ... annotate ~200 cells, export annotations ...
$XLDVP_PYTHON $REPO/train_classifier.py \
    --detections /path/to/output/cell_detections.json \
    --annotations annotations.json --output-dir /path/to/output
$XLDVP_PYTHON $REPO/scripts/apply_classifier.py \
    --detections /path/to/output/cell_detections.json \
    --classifier /path/to/output/rf_classifier.pkl \
    --output /path/to/output/cell_detections_scored.json

# 3. Place 3 reference crosses in Napari
$XLDVP_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i /path/to/slide.czi --flip-horizontal \
    -o /path/to/output/lmd/reference_crosses.json

# 4. Export to LMD XML
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_lmd_export.py \
    --detections /path/to/output/cell_detections_scored.json \
    --crosses /path/to/output/lmd/reference_crosses.json \
    --output-dir /path/to/output/lmd \
    --min-score 0.5 --generate-controls --export

# 5. Load shapes.xml into Leica LMD software
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Need at least 3 reference crosses" | Place exactly 3 crosses in Napari before saving |
| Detections not in correct locations | Verify crosses were placed with the same `--flip-horizontal` / `--rotate-cw-90` that the export expects (check `display_transform` in crosses JSON) |
| Too many detections for one plate | Multi-plate overflow is automatic (308 wells/plate). Increase `--min-score` to reduce count |
| Contours too large / laser burns neighbors | Add `--erosion-um 0.2` or `--erode-pct 0.05` to shrink contours |
| Pixel size mismatch | Pixel size is auto-detected from CZI metadata in the crosses file. Override with `--pixel-size` only if needed |
| py-lmd import error | `pip install py-lmd` or `pip install git+https://github.com/MannLabs/py-lmd.git` |
| Batch skips a slide | Ensure the slide has a matching `*_crosses.json` in `--crosses-dir` with exactly 3 crosses |
