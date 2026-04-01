# NMJ Detection Pipeline Guide

Detect neuromuscular junctions (NMJs) in CZI muscle tissue slides using 98th-percentile thresholding + morphology + watershed, with SAM2 embeddings for classification features.

## Prerequisites

```bash
# Environment
export REPO=/path/to/xldvp_seg
export XLDVP_PYTHON=/path/to/miniforge3/envs/xldvp_seg/bin/python

# Install dependencies (auto-detects CUDA)
./install.sh
```

SAM2.1 Large checkpoint (`checkpoints/sam2.1_hiera_large.pt`) is required for embedding extraction. Download from Meta if not included.

## Step 1: Inspect CZI Channels

**Always run `czi_info.py` first.** CZI channel order is NOT wavelength-sorted and cannot be inferred from the filename.

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/czi_info.py /path/to/slide.czi
```

Example output for a 3-channel NMJ slide:
```
  [0] AF488    Ex 493 -> Em 517 nm  Alexa Fluor 488   <- nuc488
  [1] AF647    Ex 653 -> Em 668 nm  Alexa Fluor 647   <- BTX647
  [2] AF750    Ex 752 -> Em 779 nm  Alexa Fluor 750   <- NFL750
```

Use `--channel-spec` to resolve channels by marker name or wavelength -- never hardcode indices:
```bash
--channel-spec "detect=BTX"     # marker name from filename
--channel-spec "detect=647"     # by wavelength
```

## Step 2: Run Detection

### Local launch

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --output-dir /path/to/output \
    --cell-type nmj \
    --channel-spec "detect=BTX" \
    --all-channels \
    --num-gpus 2 \
    --html-sample-fraction 0.10
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `--cell-type nmj` | NMJ detection strategy (98th percentile + morphology + watershed) |
| `--channel-spec "detect=BTX"` | BTX channel for detection (resolved from CZI metadata) |
| `--all-channels` | Extract per-channel intensity features (~15/channel). Always use for multi-channel slides. |
| `--num-gpus N` | Number of GPUs for parallel tile processing |
| `--html-sample-fraction 0.10` | Subsample HTML viewer to 10% of detections (browser-friendly for large slides) |
| `--photobleaching-correction` | Correct intensity decay in sequential tile scans (note: with `-ing`) |
| `--extract-deep-features` | Add ResNet+DINOv2 embeddings (6,144 dims). Off by default; try if morph F1 < 0.85. |

Flat-field correction is ON by default. Disable with `--no-normalize-features`.

Detection always processes 100% of tiles. Do not change `--sample-fraction`.

### SLURM launch (YAML config)

Check cluster availability first: `$XLDVP_PYTHON $REPO/scripts/system_info.py`

Create a YAML config (e.g., `configs/nmj_experiment.yaml`):
```yaml
name: nmj_experiment
czi_path: /path/to/slide.czi
# czi_dir: /path/to/slides       # OR directory for multi-slide batch
output_dir: /path/to/output
cell_type: nmj
channel_map:
  detect: BTX
all_channels: true
html_sample_fraction: 0.10
slurm:
  partition: <from system_info.py>
  cpus: 192                    # ~75% of node
  mem_gb: 556
  gpus: "l40s:4"
  time: "3-00:00:00"
  slides_per_job: 1
  num_jobs: 1
```

Launch:
```bash
scripts/run_pipeline.sh configs/nmj_experiment.yaml
```

### Post-dedup processing (automatic)

After detection + deduplication, three post-processing phases run automatically:

1. **Contour extraction** -- extracts original mask contour and stores as `contour_px` / `contour_um`
2. **KD-tree background estimation** -- local background from 30 nearest neighbors
3. **Background-corrected intensity features** -- re-extracts per-channel stats from the **original mask** with bg subtraction

Features are always computed from the original segmentation mask. Contour simplification (adaptive RDP) and dilation are applied at LMD export time only (`--max-area-change-pct 5.0`).

Override defaults:
```bash
--bg-neighbors 30           # KD-tree neighbors for bg estimation (default: 30)
--no-contour-processing     # Skip contour extraction
--no-background-correction  # Skip phases 2+3
```

## Step 3: Resume a Crashed/Cancelled Run

Point `--resume` at the exact timestamped run directory containing `tiles/`:

```bash
# Local
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel-spec "detect=BTX" \
    --all-channels \
    --resume /path/to/output/slide_name/nmj_20260317_120000

# SLURM: add to YAML
resume_dir: /path/to/output/slide_name/nmj_20260317_120000
```

The pipeline auto-detects the most advanced checkpoint and skips completed stages.

## Step 4: Annotate Detections

Serve the HTML viewer and annotate NMJs:

```bash
$XLDVP_PYTHON $REPO/serve_html.py /path/to/output
```

This opens a Cloudflare tunnel to the HTML viewer. Click green checkmark for real NMJs, red X for false positives. Annotate 200+ detections for good classifier performance. Export annotations via the Export button.

## Step 5: Train Classifier

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/train_classifier.py \
    --detections /path/to/output/nmj_detections.json \
    --annotations /path/to/annotations.json \
    --output-dir /path/to/output/classifier \
    --feature-set morph
```

Feature sets: `morph` (78D, fast -- often sufficient), `morph_sam2` (334D), `channel_stats` (per-channel intensities), `all` (6,478D). The trainer runs 5-fold cross-validation and reports precision/recall/F1.

## Step 6: Score All Detections

Apply the trained classifier to score every detection (CPU, seconds):

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/apply_classifier.py \
    --detections /path/to/output/nmj_detections.json \
    --classifier /path/to/output/classifier/rf_classifier.pkl \
    --output /path/to/output/nmj_detections_scored.json
```

Regenerate the HTML viewer filtered by score:

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/regenerate_html.py \
    --detections /path/to/output/nmj_detections_scored.json \
    --czi-path /path/to/slide.czi \
    --output-dir /path/to/output \
    --score-threshold 0.5
```

## Step 7: Marker Classification (Optional)

For multi-channel NMJ slides, classify each detection as positive/negative per fluorescent marker. Background correction from detection is auto-detected and reused.

```bash
# By wavelength (preferred -- auto-resolves via CZI metadata):
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/classify_markers.py \
    --detections /path/to/output/nmj_detections_scored.json \
    --marker-wavelength 647,750 \
    --marker-name BTX,NFL \
    --czi-path /path/to/slide.czi
```

Methods: `otsu` (default), `otsu_half` (more permissive for dim markers), `gmm` (2-component Gaussian).

Output adds `{marker}_class`, `{marker}_value`, `{marker}_threshold`, and `marker_profile` fields to each detection.

## Step 8: LMD Export (Optional)

Export scored NMJs for laser microdissection.

**Place 3 reference crosses** in Napari:
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i /path/to/slide.czi --flip-horizontal -o /path/to/crosses.json
```

**Export XML:**
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_lmd_export.py \
    --detections /path/to/output/nmj_detections_scored.json \
    --crosses /path/to/crosses.json \
    --output-dir /path/to/output/lmd \
    --min-score 0.5 --generate-controls --export
```

Max 308 wells per 384-well plate; multi-plate overflow is automatic. QC empty wells (10%) inserted evenly.

## Output Structure

```
output_dir/
  slide_name/
    nmj_YYYYMMDD_HHMMSS/           # timestamped run directory
      tiles/
        tile_X_Y/
          nmj_masks.h5              # segmentation masks (HDF5)
          nmj_detections.json       # per-tile detections
          nmj_html_samples.json     # cached HTML crops
      nmj_detections.json           # final deduplicated detections
      nmj_detections_postdedup.json # post-dedup checkpoint
      nmj_coordinates.csv           # coordinate export
      nmj_spatialdata.zarr/         # SpatialData (auto-generated)
      html/
        index.html                  # annotation viewer
        nmj_page_*.html
      run.log
```

## Expected Runtime

| Setup | Slide (263k x 89k px, ~1800 tiles) |
|-------|------|
| 4x L40S (SLURM) | 1-3 hours |
| 2x GPU (local) | 3-6 hours |
| 1x GPU | 6-12 hours |

Runtime depends on NMJ density, GPU count, and whether deep features are enabled.

## Memory Requirements

| Resource | Typical |
|----------|---------|
| RAM | 60-80 GB (direct-to-SHM, scales with channel count) |
| GPU VRAM | 2-3 GB per GPU (SAM2 inference) |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| OOM | Reduce `--num-gpus` or reduce `--tile-size` |
| HDF5 LZ4 error | `import hdf5plugin` before `h5py` |
| Wrong channel detected on | Re-run `czi_info.py` and verify `--channel-spec` |
| Slow on network mount | `--load-to-ram` is default; check mount with `ls /path/` |
| Need to resume | `--resume /path/to/exact/timestamped/run_dir` (must contain `tiles/`) |
