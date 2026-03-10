# xldvp_seg - Image Analysis Pipelines

## Session Behaviors

These behaviors apply throughout every Claude Code session on this project:

**Context management:**
- When context reaches ~15% remaining, proactively: (1) update your auto-memory files with anything learned this session (patterns, bugs found, architectural decisions), (2) update any code documentation that changed, (3) commit uncommitted work with a descriptive message. Tell the user you're doing this so they're not surprised.
- When starting a continued session (context was compacted), read your memory files first to rebuild context before diving into work.

**Code hygiene:**
- After completing any significant code change (new feature, bug fix, refactor), review what you wrote before moving on. Catch your own mistakes.
- After modifying pipeline code, check if CLAUDE.md, the relevant docs/*.md, or slash commands need updating. Keep documentation in sync with code — don't let them drift.
- When you fix a bug, check if the same pattern exists elsewhere in the codebase. Fix all instances, not just the one the user pointed out.

**Communication:**
- When running long operations (SLURM jobs, large file reads, multi-agent reviews), give the user a brief status update rather than going silent.
- When you encounter something unexpected (a file that doesn't match docs, a function that behaves differently than expected), flag it to the user — don't silently work around it.
- After completing a multi-step task, give a concise summary: what changed, how many files, any notable findings.
- **Always use the AskUserQuestion tool** when you need to ask questions — never list questions inline in text responses.
- **Always enter plan mode first** for implementation tasks — design the approach and get approval before writing code.

**Pipeline-specific:**
- Always run `czi_info.py` before writing any channel configuration. No exceptions.
- Never hardcode pixel sizes, channel indices, or file paths that should come from CZI metadata.
- When writing SLURM configs, check partition busyness first (`system_info.py`).
- Prefer `--channel-spec "detect=MARKER"` over raw `--channel N` in all examples and configs.

## Quick Start (Claude Code)

Type `/analyze` to begin. Claude will detect your system, inspect your data,
and guide you through the full pipeline — detection through LMD export.

| Command | What it does |
|---------|-------------|
| `/analyze` | Full pipeline: detect, annotate, classify, spatial analysis, LMD export |
| `/status` | Check running SLURM jobs, tail logs, monitor progress |
| `/czi-info` | Inspect CZI metadata — channels, dimensions, pixel size |
| `/preview-preprocessing` | Preview flat-field / photobleach correction on any channel |
| `/classify` | Train RF classifier from annotations, compare feature sets, explore features |
| `/lmd-export` | Export detections for laser microdissection (contours, wells, XML) |
| `/vessel-analysis` | Multi-scale vessel structure detection + spatial viewer |
| `/view-results` | Launch HTML result viewer with Cloudflare tunnel |

## Quick Start

**Pipelines available:**
1. **MK/HSPC** - Bone marrow cell segmentation (Megakaryocytes + Stem Cells)
2. **NMJ** - Neuromuscular junction detection in muscle tissue
3. **Cell** - Generic 2-channel Cellpose segmentation (e.g. NeuN+nuc, senescence)
4. **Vessel** - Blood vessel morphometry (SMA+ ring detection)
5. **Mesothelium** - Mesothelial ribbon detection for laser microdissection
6. **Islet** - Pancreatic islet cell detection (nuclear + membrane channels)
7. **Tissue Pattern** - Whole-mount tissue detection (brain FISH, coronal sections)

### Documentation
- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Full user guide
- **[NMJ_PIPELINE_GUIDE.md](docs/NMJ_PIPELINE_GUIDE.md)** - NMJ detection with classifier
- **[NMJ_LMD_EXPORT_WORKFLOW.md](docs/NMJ_LMD_EXPORT_WORKFLOW.md)** - Full LMD export workflow
- **[LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md)** - Basic LMD export
- **[COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md)** - Coordinate conventions

### Key Locations
| What | Where |
|------|-------|
| This repo | `$REPO` (auto-detected, or set `REPO=/path/to/xldvp_seg`) |
| Output dirs | `<output_dir>/<slide_name>/<timestamp>/` (timestamped per run) |
| Conda env | `mkseg` |

**Activate environment:**
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg
```

---

## Common Commands

### Unified Segmentation
```bash
# NMJ detection — resolve BTX channel automatically from filename
python run_segmentation.py \
    --czi-path /path/to/nuc488_BTX647_NFL750.czi \
    --cell-type nmj \
    --channel-spec "detect=BTX"

# Generic cell with 2-channel Cellpose — resolve by marker name
python run_segmentation.py \
    --czi-path /path/to/nuc488_NeuN647_tdTom555.czi \
    --cell-type cell \
    --channel-spec "cyto=NeuN,nuc=nuc"

# Vessel detection (raw index still works)
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 0 \
    --candidate-mode

# MK detection
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type mk \
    --channel 0
```

### Performance Options
```bash
--load-to-ram       # [Default] Direct-to-SHM loading (channels loaded from CZI directly into shared memory)
--num-gpus 4        # Number of GPUs (always multi-GPU, even with 1)
--num-gpus 1        # Single GPU — safer memory usage for large slides
--html-sample-fraction 0.10  # Subsample HTML to 10% of detections (saves RAM on large slides)
--max-html-samples 20000     # Hard OOM cap during per-tile accumulation (default: 20000)
```

### Post-Dedup Processing Options
```bash
--no-contour-processing     # Skip contour dilation + RDP (default ON)
--dilation-um 0.5           # Contour dilation in micrometers (default: 0.5)
--rdp-epsilon 5.0           # RDP simplification epsilon in pixels (default: 5)
--no-background-correction  # Skip local background subtraction (default ON)
--bg-neighbors 30           # KD-tree neighbors for background (default: 30)
```

### OME-Zarr Generation
```bash
--no-zarr                   # Skip OME-Zarr generation at end of pipeline (default ON)
--force-zarr                # Overwrite existing OME-Zarr (default: skip if exists)
--zarr-levels 5             # Number of pyramid levels (default: 5)
```

OME-Zarr is auto-generated at the end of every pipeline run from SHM data (fast, no CZI re-read). Used for cross placement and Napari visualization.

Post-dedup phases 1 and 3 are parallelized with ThreadPoolExecutor (auto-detects CPU count).

### Multi-Node Sharding
```bash
# Split detection across 4 nodes (each processes 1/4 of tiles)
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 0/4 --resume /shared/output/dir  # node 0
    --tile-shard 1/4 --resume /shared/output/dir  # node 1
    # ... etc

# Merge all shards (auto-detects shard manifests, checkpointed)
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --resume /shared/output/dir --merge-shards
# Checkpoints: merged_detections.json → detections.json → HTML
```

### View Results
```bash
python -m http.server 8080 --directory /home/dude/mk_output/project/html
~/cloudflared tunnel --url http://localhost:8080
```

---

## NMJ Pipeline (Detect Once, Classify Later)

1. Load CZI channels directly into shared memory (no RAM intermediate, `--load-to-ram` default)
2. Preprocessing on SHM views (flat-field, photobleach correction)
3. Tile with 10% overlap (`--tile-overlap 0.10`)
4. Detect 100% of tiles (or multi-node with `--tile-shard`)
5. Segment: 98th percentile intensity threshold + morphology + watershed
6. Initial feature extraction: morph + SAM2 (always), ResNet + DINOv2 (opt-in `--extract-deep-features`)
7. HTML crops cached to disk per-tile (fast resume — `{tile_dir}/{celltype}_html_samples.json`)
8. Deduplicate overlapping masks (>10% pixel overlap) — or `--merge-shards` for multi-node
9. **Post-dedup processing** (default ON, parallelized with ThreadPoolExecutor):
   - Phase 1: Dilate contours +0.5 um (`--dilation-um`), RDP simplify (`--rdp-epsilon 5`), extract quick means
   - Phase 2: Local background estimation (KD-tree, k=30 global neighbors, `--bg-neighbors`)
   - Phase 3: Subtract per-cell background from pixels, then extract intensity features on corrected data (morph features preserved from detection)
   - Disable with `--no-contour-processing` / `--no-background-correction`
10. Generate annotation HTML (subsample via `--html-sample-fraction 0.10` or `scripts/regenerate_html.py --max-samples 1500`)
11. Train RF classifier with annotations (`train_classifier.py`, balanced classes)
12. Score ALL detections: `scripts/apply_classifier.py` (CPU, seconds — no re-detection)
13. Generate filtered HTML: `scripts/regenerate_html.py --score-threshold 0.5`
14. Two-stage clustering: Round 1 = 500 um, Round 2 = 1000 um, target 375-425 um²
15. Unclustered = singles
16. Controls: 100 um offset, 8 directions, cluster controls preserve arrangement
17. Napari visualization (4 colors: singles/controls/clusters/cluster-controls)
18. 384-well plate serpentine B2 → B3 → C3 → C2 (max 308 wells)
19. OME-Zarr pyramid for Napari viewing
20. XML export with reference crosses

### Pipeline Checkpoints (resume with `--resume`)

| Stage | Checkpoint file | What's saved |
|-------|----------------|--------------|
| Detection | Per-tile dirs (`tile_X_Y/`) | Masks (HDF5) + per-tile detections (JSON) + HTML cache |
| Merge shards | `{celltype}_detections_merged.json` | All shard detections concatenated |
| Dedup | `{celltype}_detections.json` | Deduplicated detections (merge-shards only) |
| Post-dedup | `{celltype}_detections_postdedup.json` | Contours + features + bg correction |
| Finalize | `{celltype}_detections.json` + HTML/CSV | Final output |

On `--resume`, the pipeline loads the most advanced checkpoint and skips completed steps.
Contour processing and background correction are checked independently — if contours were
processed but bg correction wasn't, only bg correction runs on resume.

### Channel Mapping

**CRITICAL: CZI channel order ≠ filename order, and is NOT simply sorted by wavelength.**
Channel indices are determined by acquisition/detector assignment in the CZI file.
**Always run `czi_info.py` first — it is the only authoritative source.**

**Step 1 — Run czi_info.py (mandatory before any config):**
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/czi_info.py /path/to/slide.czi
```
Output example:
```
  [0] AF488    Ex 493 → Em 517 nm  Alexa Fluor 488   ← nuc488
  [1] AF647    Ex 653 → Em 668 nm  Alexa Fluor 647   ← SMA647
  [2] AF750    Ex 752 → Em 779 nm  Alexa Fluor 750   ← PM750
  [3] AF555    Ex 553 → Em 568 nm  Alexa Fluor 555   ← CD31_555
```
Note: [1]=647nm comes before [3]=555nm — this is NOT wavelength-sorted.
Never guess the order. Never sort by wavelength manually. Always use this output.

**Step 2 — Match indices to markers using filename + fluorophore name:**
Cross-reference the `Alexa Fluor NNN` name with the marker wavelengths in the filename
(e.g. `SMA647` → Alexa647 → whichever index shows `Em 668nm`).

**Step 3 — Build your channel assignments:**
- `cellpose_input_channels: [cyto_idx, nuc_idx]` — use PM/membrane channel as cyto
- `markers: [{channel: idx, name: MARKER}]` — one entry per marker to classify
- `load_channels: "0,1,2"` — exclude failed stains (e.g. bad PDGFRa)
- Document the verified mapping as a comment in the YAML (see `configs/` examples)

**Automated resolution via `--channel-spec`** (preferred):
```bash
# Specify channels by marker name or wavelength — resolved at startup
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --channel-spec "detect=BTX"            # name from filename

python run_segmentation.py --czi-path slide.czi --cell-type cell \
    --channel-spec "cyto=PM,nuc=488"       # mix of name and wavelength
```

Resolution order: integer index → wavelength (±10nm) → marker name (via filename parsing) → CZI metadata name (exact match, then substring match for ≥3-char specs, e.g., "Hoechst" matches "Hoechst 33258").

**For `classify_markers.py`**: Use `--marker-wavelength 647,555 --czi-path slide.czi` instead of `--marker-channel 1,2`. No `--correct-all-channels` needed — pipeline does bg correction automatically, and `classify_markers.py` auto-detects this to prevent double correction.

**For YAML configs** (`run_pipeline.sh`): Add a `channel_map:` section:
```yaml
channel_map:
  detect: SMA       # resolved to CZI channel index at runtime
  cyto: PM
  nuc: 488
```

**Manual fallback**: Raw indices (`--channel 1`, `--marker-channel 1,3`) still work.

**Post-dedup processing YAML keys:**
```yaml
background_correction: true    # default ON — local KD-tree bg subtraction
contour_processing: true       # default ON — dilate + RDP contours
dilation_um: 0.5               # contour dilation in micrometers
rdp_epsilon: 5                 # RDP simplification epsilon in pixels
bg_neighbors: 30               # KD-tree neighbor count
html_sample_fraction: 0.10     # subsample HTML to 10% of detections (saves RAM)
```

**SLURM recommended defaults:**
```yaml
slurm:
  slides_per_job: 1    # 1 slide per SLURM task (better cluster scheduling)
  num_jobs: 24          # one job per slide
```

**Resume with `run_pipeline.sh`**: add `resume_dir: /path/to/run_dir` to YAML. Without it, re-running always starts fresh (auto-discovery removed to prevent accidentally resuming old test/sample runs).
```yaml
resume_dir: /fs/pool/pool-mann-edwin/my_output/slide_20260302_121345_100pct
```

**NMJ example (3-channel):**
- ch0: Nuclear (488nm)
- ch1: BTX (647nm) — NMJ marker (detection channel)
- ch2: NFL (750nm)

### Feature Hierarchy
| Feature Set | Dimensions | When |
|-------------|-----------|------|
| Morphological | ~78 | Always |
| Per-channel stats | ~50 | When `--all-channels` + 2+ channels |
| SAM2 embeddings | 256 | Always (default) |
| ResNet50 (masked + context) | 4,096 | `--extract-deep-features` |
| DINOv2-L (masked + context) | 2,048 | `--extract-deep-features` |

**Feature comparison (3-channel, 844 annotated detections):**

| Feature Set | n | F1 |
|-------------|---|-----|
| **all_features** | 6478 | **0.909** |
| morph+dinov2_combined | 2126 | 0.901 |
| morphological | 78 | 0.900 |
| dinov2_context | 1024 | 0.843 |
| resnet_context | 2048 | 0.836 |
| sam2 | 256 | 0.728 |

Use `train_classifier.py --feature-set` to compare subsets and pick the best for your final RF classifier. Morph-only (78 features) is nearly as good as all 6478 combined.

---

## Coordinate System

**All coordinates are [x, y] (horizontal, vertical).**

UID format: `{slide}_{celltype}_{x}_{y}`

**Mosaic origin:** CZI tiles use global coordinates. RAM arrays are 0-indexed.
- `loader.get_tile()` handles the offset correctly
- Direct `all_channel_data` indexing must subtract `x_start, y_start`

---

## Entry Points

| Script | Use Case |
|--------|----------|
| `run_segmentation.py` | Unified entry point: detect, dedup, HTML, CSV (recommended) |
| `run_lmd_export.py` | Export to Leica LMD format (any cell type) |
| `train_classifier.py` | Train RF classifier from annotated detections |
| `scripts/apply_classifier.py` | Score existing detections with trained classifier (no re-detection) |
| `scripts/classify_markers.py` | Marker classification (Otsu/GMM). Auto-detects pipeline bg correction, no double-correction. |
| `scripts/regenerate_html.py` | Regenerate HTML viewer from saved detections (all cell types) |
| `scripts/czi_to_ome_zarr.py` | Convert CZI to OME-Zarr with pyramids |
| `scripts/napari_place_crosses.py` | Interactive cross placement (CZI + OME-Zarr, RGB crosses, flip/rotate, batch) |
| `scripts/cluster_detections.py` | Biological clustering for LMD well assignment |
| `scripts/napari_view_lmd_export.py` | View LMD export overlaid on slide |
| `scripts/convert_to_spatialdata.py` | Convert detections to SpatialData zarr (scverse ecosystem) |
| `scripts/generate_multi_slide_spatial_viewer.py` | Unified spatial viewer: KDE contours, graph-pattern regions, DBSCAN + hulls, ROI, focus view |
| `scripts/view_slide.py` | One-command visualization: classify + spatial cluster + interactive viewer |
| `scripts/vessel_community_analysis.py` | Multi-scale vessel structure detection (connected components + morphology + SNR) |
| `scripts/spatial_cell_analysis.py` | Spatial network analysis (connected components, graph metrics) |
| `scripts/preview_preprocessing.py` | Preview flat-field / photobleach correction at reduced resolution |
| `scripts/run_pipeline.sh` | YAML config-driven multi-slide SLURM launcher |
| `scripts/select_mks_for_lmd.py` | MK-specific LMD replicate selection + multi-plate well assignment |
| `scripts/system_info.py` | Environment detection + resource recommendation for SLURM |

---

## Module Reference

### Detection Strategies (all support MultiChannelFeatureMixin)
| Strategy | File |
|----------|------|
| NMJ | `segmentation/detection/strategies/nmj.py` |
| MK | `segmentation/detection/strategies/mk.py` |
| Cell | `segmentation/detection/strategies/cell.py` |
| Vessel | `segmentation/detection/strategies/vessel.py` |
| Islet | `segmentation/detection/strategies/islet.py` |
| Mesothelium | `segmentation/detection/strategies/mesothelium.py` |
| Tissue Pattern | `segmentation/detection/strategies/tissue_pattern.py` |

### Multi-GPU Processing (always used, even with --num-gpus 1)
| Module | Purpose |
|--------|---------|
| `segmentation/processing/multigpu_worker.py` | Generic GPU worker (all cell types) |
| `segmentation/processing/multigpu_shm.py` | Shared memory manager (SIGTERM cleanup) |
| `segmentation/processing/tile_processing.py` | Shared `process_single_tile()` |

### Post-Dedup Processing
| Module | Purpose |
|--------|---------|
| `segmentation/pipeline/post_detection.py` | Contour dilation + feature re-extraction + bg correction |
| `segmentation/pipeline/background.py` | KD-tree local background correction (shared w/ classify_markers) |

### LMD Export
| Module | Purpose |
|--------|---------|
| `run_lmd_export.py` | Unified pipeline: contours, controls, wells, XML (single + batch mode) |
| `segmentation/lmd/clustering.py` | Two-stage greedy clustering |
| `segmentation/lmd/contour_processing.py` | Dilation + RDP + erosion (absolute + percent) |
| `segmentation/lmd/selection.py` | Generic cell selection for area-normalized LMD replicates |
| `segmentation/lmd/well_plate.py` | Multi-plate 384-well serpentine well generation + QC empty wells |

### OME-Zarr Export
| Module | Purpose |
|--------|---------|
| `segmentation/io/ome_zarr_export.py` | SHM-to-zarr with pyramid generation (auto at end of pipeline) |

### Normalization
| Module | Purpose |
|--------|---------|
| `compute_normalization_params.py` | Phase 1: compute cross-slide LAB stats |
| `segmentation/preprocessing/stain_normalization.py` | Phase 2: apply Reinhard per-slide |

Tissue detection in normalization uses calibrated threshold / 10 for permissive detection.

---

## Vessel Pipeline

### Detection Method
1. Adaptive + Otsu thresholding for SMA+ regions
2. Contour hierarchy analysis (find rings)
3. Ellipse fitting, wall thickness at 36 angles
4. **3-contour system**: lumen (cyan), CD31 outer (green), SMA ring (magenta)
5. Adaptive per-pixel dilation on CD31 and SMA channels (irregular contours following signal)
6. Full-resolution contour refinement (reads CZI ROI at native res per vessel)
7. Full-resolution crop rendering for HTML

### 3-Contour Ring Detection
- **Lumen** (cyan): Inner boundary from SAM2/threshold segmentation
- **CD31** (green): Endothelial outer boundary via adaptive dilation on CD31 channel
- **SMA** (magenta): Smooth muscle ring via adaptive dilation on SMA channel, expanding from lumen (not from CD31 boundary, since CD31 and SMA intermingle)

### CLI Options (Vessel-specific)
```bash
--candidate-mode               # Relaxed thresholds for training data generation
--ring-only                    # Disable supplementary lumen-first pass
--no-smooth-contours           # Disable B-spline contour smoothing (on by default)
--smooth-contours-factor 3.0   # Spline smoothing factor (default: 3.0)
--multi-scale                  # Multi-scale detection (coarse to fine)
```

### Multi-Marker (6-type classification)
artery, arteriole, vein, capillary, lymphatic, collecting_lymphatic

---

## Hardware (SLURM Cluster)
- **p.hpcl8:** 55 nodes, 24 CPUs, 380G RAM, 2x RTX 5000 each (interactive dev, CPU jobs)
- **p.hpcl93:** 19 nodes, 256 CPUs, 760G RAM, 4x L40S each (heavy GPU batch jobs, requires `--gres=gpu:`)
- Time limit: 42 days on both partitions

## MPS Support (Apple Silicon)

The pipeline supports Apple Silicon GPUs via PyTorch's MPS backend. Device selection is automatic — the pipeline detects `cuda`, `mps`, or `cpu` and configures accordingly.

**Central utility:** `segmentation/utils/device.py` — use `get_default_device()`, `set_device_for_worker()`, `empty_cache()`, `device_supports_gpu()`.

**Key behaviors on MPS:**
- `--num-gpus` defaults to 1 (Apple Silicon has one GPU)
- Multi-worker subprocess architecture still works (1 spawned worker)
- All 4 ML models confirmed working: SAM2, Cellpose-SAM, ResNet50, DINOv2
- Memory validation reports unified memory (GPU shares system RAM)
- `empty_cache()` calls `torch.mps.synchronize()` before clearing

**Do NOT:**
- Hardcode `device="cuda"` — use `device=None` or `get_default_device()`
- Use `torch.cuda.is_available()` for Cellpose `gpu=` flag — use `device_supports_gpu()`
- Use bare `torch.cuda.empty_cache()` — use `empty_cache()` from `segmentation.utils.device`

## Troubleshooting

### OOM: reduce `--num-gpus`, reduce tile size
### CUDA Boolean: `mask = mask.astype(bool)` for SAM2
### SAM2 _orig_hw: `img_h, img_w = sam2_predictor._orig_hw[0]` (list of tuple)
### HDF5 LZ4: `import hdf5plugin` before `h5py`
### Network Mounts: Socket timeout 60s automatic. Check with `ls /mnt/x/`

---

## OME-Zarr / LMD Export Workflow

OME-Zarr is auto-generated at the end of every pipeline run (from SHM, fast). No separate conversion step needed. Use `--no-zarr` to skip.

```bash
# 1. Place reference crosses (CZI-native recommended, no conversion needed)
python scripts/napari_place_crosses.py \
    -i slide.czi --channel 0 -o crosses.json

# With LMD7 display transforms (tissue-down + rotated)
python scripts/napari_place_crosses.py \
    -i slide.czi --flip-horizontal --rotate-cw-90 -o crosses.json

# Or use OME-Zarr (auto-generated by pipeline, for very large slides)
python scripts/napari_place_crosses.py \
    -i slide.ome.zarr -o crosses.json

# 2. Export to LMD (with controls + erosion)
python run_lmd_export.py \
    --detections detections.json \
    --crosses crosses.json \
    --output-dir lmd_export \
    --generate-controls \
    --min-score 0.5 \
    --export

# Optional: erosion at export time (shrink contours so laser cuts inside)
    --erosion-um 0.2      # Absolute distance (um)
    --erode-pct 0.05      # Percent of sqrt(area)

# Batch export (multiple slides)
python run_lmd_export.py \
    --input-dir /path/to/runs \
    --crosses-dir /path/to/crosses \
    --output-dir lmd_batch \
    --generate-controls --min-score 0.5 --export
```

Max 308 wells per plate. Early capacity check warns before expensive processing. For >308 wells, use `generate_multiplate_wells()` from `segmentation.lmd.well_plate` for automatic overflow to additional plates. Empty QC wells (default 10% of samples) inserted evenly via `insert_empty_wells()`. For proteomics replicates, use `segmentation.lmd.selection.select_cells_for_lmd()` to build area-normalized replicates, then `scripts/select_mks_for_lmd.py` (MK-specific wrapper) for full plate assignment.

---

## SpatialData Integration (scverse ecosystem)

Pipeline detections are automatically exported to SpatialData format (zarr) for use with squidpy, scanpy, and the broader scverse ecosystem.

### Automatic Export
Every pipeline run exports `{celltype}_spatialdata.zarr` alongside the JSON/CSV/HTML outputs. Requires spatialdata/anndata/geopandas (installed by default).

### Standalone Converter
```bash
# Convert any existing detection JSON
python scripts/convert_to_spatialdata.py \
    --detections /path/to/cell_detections.json \
    --output /path/to/output.zarr \
    --tiles-dir /path/to/tiles/ \
    --run-squidpy --squidpy-cluster-key tdTomato_class
```

### YAML Config (run_pipeline.sh)
```yaml
spatialdata:
  enabled: true
  extract_shapes: true
  run_squidpy: true
  squidpy_cluster_key: tdTomato_class
```

### Load in Python
```python
import spatialdata as sd
sdata = sd.read_zarr("output.zarr")
adata = sdata["table"]  # AnnData with spatial coords, features, embeddings

import squidpy as sq
sq.gr.spatial_neighbors(adata)
sq.pl.spatial_scatter(adata, color="tdTomato_class")
```

---

## Installation

```bash
conda create -n mkseg python=3.11 -y && conda activate mkseg
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
./install.sh  # Auto-detects CUDA
```
