# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# xldvp_seg — Image Analysis & DVP Pipelines

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

---

## Development Commands

```bash
# Environment
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")"; pwd)"  # or set manually
export XLDVP_PYTHON="${XLDVP_PYTHON:-$(which python)}"          # xldvp_seg conda env python

# Run all tests
PYTHONPATH=$REPO $XLDVP_PYTHON -m pytest tests/ -v --tb=short

# Run single test file
PYTHONPATH=$REPO $XLDVP_PYTHON -m pytest tests/test_coordinates.py -v

# Run single test class
PYTHONPATH=$REPO $XLDVP_PYTHON -m pytest tests/test_coordinates.py::TestCoordinateConversion -v

# Lint
PYTHONPATH=$REPO $XLDVP_PYTHON -m ruff check .

# Format check
PYTHONPATH=$REPO $XLDVP_PYTHON -m black --check .

# Inspect CZI metadata (mandatory before any channel config)
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/czi_info.py /path/to/slide.czi
```

**Style:** Black (line-length 100), Ruff (E/F/W/I/N/UP/B/C4, E501 ignored). Python 3.10+.

### Tests

Tests are in `tests/` using pytest. Fixtures in `conftest.py`: `sample_tile` (512×512 RGB), `sample_mask` (boolean circle), `mock_loader` (mocked CZI loader), `temp_output_dir`, `sample_tile_uint16`, `mock_regionprop`.

| Test file | What it validates |
|-----------|------------------|
| `test_coordinates.py` | [x,y] coordinate conversion, UID parsing, spatial validation |
| `test_detection_base.py` | `Detection` dataclass, mask area calculations |
| `test_feature_extraction.py` | Feature dimensions: morph=78, SAM2=256, ResNet=4096, DINOv2=2048, total=6478 |
| `test_utils.py` | Config module, `get_feature_dimensions()` |
| `test_module_imports.py` | HTML export functions, MK/HSPC module imports |
| `test_mk_hspc_imports.py` | Feature dimension constants match config |
| `test_nmj_imports.py` | NMJ strategy class methods, classifier loaders |

Tests use `sys.path.insert(0, ...)` before importing `segmentation.*` to handle CWD reset.

---

## Quick Start

Type `/analyze` inside Claude Code to begin — it detects your system, inspects your data, and walks you through the full pipeline.

| Command | What it does |
|---------|-------------|
| `/analyze` | Full pipeline: detect → annotate → classify → spatial analysis → LMD export |
| `/status` | Check running SLURM jobs, tail logs, monitor progress |
| `/czi-info` | Inspect CZI metadata — channels, dimensions, pixel size |
| `/preview-preprocessing` | Preview flat-field / photobleach correction on any channel |
| `/classify` | Train RF classifier from annotations, compare feature sets, explore features |
| `/lmd-export` | Export detections for laser microdissection (contours, wells, XML) |
| `/vessel-analysis` | Multi-scale vessel structure detection + spatial viewer |
| `/view-results` | Launch HTML result viewer with Cloudflare tunnel |
| `/spatialdata` | Export to SpatialData zarr + squidpy spatial analysis |

All commands are in `.claude/commands/`. Documentation: `docs/GETTING_STARTED.md`, `docs/NMJ_PIPELINE_GUIDE.md`, `docs/LMD_EXPORT_GUIDE.md`, `docs/COORDINATE_SYSTEM.md`, `docs/VESSEL_COMMUNITY_ANALYSIS.md`.

**Pipelines available:**

| Type | Detection Method | Use Case |
|------|-----------------|----------|
| **Cell** | Cellpose 2-channel (cyto+nuc) + SAM2 embeddings | Generic cell detection (e.g. NeuN+nuc, senescence) |
| **NMJ** | 98th percentile threshold + morphology + watershed | Neuromuscular junction detection in muscle |
| **MK** | SAM2 auto-mask + size filter | Bone marrow megakaryocytes + stem cells |
| **Vessel** | SMA+ ring detection, 3-contour hierarchy, adaptive dilation | Blood vessel morphometry |
| **Islet** | Cellpose membrane+nuclear + marker classification | Pancreatic islet cells |
| **Mesothelium** | Ridge detection for ribbon structures | Mesothelial ribbon for LMD |
| **Tissue Pattern** | Cellpose + spatial frequency analysis | Whole-mount tissue (brain FISH, coronal) |

---

## Pipeline Workflow: CZI → Detection → Classification → LMD

This is the DVP (Deep Visual Proteomics) workflow — spatial proteomics where LMD-cut cells go into mass spec analysis.

### Overview (for beginners)

1. **Inspect** — Look at your slide's channels (which stains are in which position). Seconds.
2. **Detect** — AI models (SAM2 + custom segmentation) scan every tile to find all cells. Each gets a contour outline and features (shape, size, brightness, AI embeddings). 1–3 hours on GPU.
3. **Review** — Open a web viewer, click yes/no on ~200+ cell crops to teach the system what's real vs false positive.
4. **Classify** — A random forest trains on your annotations and scores every detection in seconds — no re-detection needed.
5. **Markers** — Classify each cell as positive/negative for each fluorescent marker (e.g., NeuN+ neurons vs NeuN- glia). Uses automatic intensity thresholding.
6. **Explore** — UMAP/clustering can reveal cell subtypes from the feature space. Spatial network analysis shows neighborhood patterns.
7. **Export for LMD** — Package selected cells into XML for the laser microdissection machine with 384-well plate assignment + controls.
8. **DVP** — LMD cuts cells → mass spec spatial proteomics.

### Phase 1: Data Inspection

**CRITICAL: CZI channel order ≠ filename order and is NOT wavelength-sorted.** Channel indices are determined by acquisition/detector assignment. Always run `czi_info.py` first — it is the only authoritative source.

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/czi_info.py /path/to/slide.czi
```
Output example:
```
  [0] AF488    Ex 493 → Em 517 nm  Alexa Fluor 488   ← nuc488
  [1] AF647    Ex 653 → Em 668 nm  Alexa Fluor 647   ← SMA647
  [2] AF750    Ex 752 → Em 779 nm  Alexa Fluor 750   ← PM750
  [3] AF555    Ex 553 → Em 568 nm  Alexa Fluor 555   ← CD31_555
```
Note: [1]=647nm before [3]=555nm. Never sort by wavelength. Always confirm this table with the user before proceeding.

**Use `--channel-spec`** to resolve channels by name or wavelength automatically:
```bash
--channel-spec "detect=BTX"          # marker name from filename
--channel-spec "cyto=PM,nuc=488"     # mix of name and wavelength
--channel-spec "detect=647"          # direct wavelength
```
Resolution order: integer index → wavelength (±10nm) → marker name (filename parsing) → CZI metadata name (substring match for ≥3-char specs, e.g., "Hoechst" matches "Hoechst 33258"). Raw indices (`--channel 1`) still work as fallback.

For YAML configs (`run_pipeline.sh`), use `channel_map:` section:
```yaml
channel_map:
  detect: SMA       # resolved to CZI channel index at runtime
  cyto: PM
  nuc: 488
```

### Phase 2: Detection

**Check cluster first:** `$XLDVP_PYTHON $REPO/scripts/system_info.py` shows partition availability and recommends resources.

**Key parameters:**
- `--cell-type {nmj,mk,vessel,mesothelium,islet,tissue_pattern,cell}`
- `--all-channels` — enables per-channel intensity features (~15/channel). Always use for multi-channel slides.
- `--channels "0,1,2"` — skip failed stains to save RAM
- `--extract-deep-features` — adds ResNet+DINOv2 (6,144 dims). Off by default; morph-only (78 features) is often competitive with all 6,478. Worth trying if morph F1 < 0.85 after annotation.
- `--photobleaching-correction` (with `-ing`) — for sequential tile scans with intensity decay
- Flat-field ON by default (`--no-normalize-features` to disable; no `--flat-field-correction` flag)
- `--html-sample-fraction 0.10` — subsample HTML viewer to 10% of detections (browser-friendly)
- `--sample-fraction` is ALWAYS 1.0 — detect 100%, never suggest partial detection

**SLURM launch (YAML config):**
```yaml
name: my_experiment
czi_path: /path/to/slide.czi     # single slide
# czi_dir: /path/to/slides       # OR directory for multi-slide
output_dir: /path/to/output
cell_type: cell
channel_map:
  cyto: PM
  nuc: 488
all_channels: true
html_sample_fraction: 0.10
slurm:
  partition: p.hpcl93          # from system_info.py
  cpus: 192                    # ~75% of node (leaves headroom for other users)
  mem_gb: 556
  gpus: "l40s:4"
  time: "3-00:00:00"
  slides_per_job: 1             # 1 slide/job = parallel SLURM array tasks
  num_jobs: 1
```
```bash
scripts/run_pipeline.sh configs/my_experiment.yaml
```

**Local launch:**
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type cell \
    --channel-spec "cyto=PM,nuc=488" \
    --all-channels --num-gpus 2 \
    --output-dir /path/to/output
```

**Resume crashed/cancelled runs:**
- SLURM: add `resume_dir: /path/to/exact/timestamped/run_dir` to YAML, re-run `run_pipeline.sh`. Without it, always starts fresh.
- Local: `--resume /path/to/exact/timestamped/run_dir` (must contain `tiles/` directly — NOT the slide-level dir)
- Pipeline auto-detects the most advanced checkpoint and skips completed stages.

**Pipeline checkpoints:**

| Stage | Checkpoint file | What's saved |
|-------|----------------|--------------|
| Detection | Per-tile dirs (`tile_X_Y/`) | Masks (HDF5) + detections (JSON) + HTML cache |
| Merge shards | `{celltype}_detections_merged.json` | All shard detections concatenated |
| Dedup | `{celltype}_detections.json` | Deduplicated detections |
| Post-dedup | `{celltype}_detections_postdedup.json` | Contours + features + bg correction |
| Finalize | `{celltype}_detections.json` + HTML/CSV | Final output |

Contour processing and background correction are checked independently on resume.

### Phase 3: Annotation & Classification

**Step 1 — Serve results:** `$XLDVP_PYTHON $REPO/serve_html.py <output_dir>` — opens HTML viewer via Cloudflare tunnel. Click green ✓ for real detections, red ✗ for false positives. Export annotations via the Export button.

**Step 2 — Train classifier:**
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/train_classifier.py \
    --detections <detections.json> \
    --annotations <annotations.json> \
    --output-dir <output> \
    --feature-set morph   # or morph_sam2, channel_stats, all
```
Feature sets: `morph` (78D, fast), `morph_sam2` (334D), `channel_stats` (per-channel intensities), `all` (6,478D). Use `train_classifier.py --feature-set` to compare — performance varies by cell type and staining.

**Step 3 — Score all detections + regenerate filtered HTML:**
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/apply_classifier.py \
    --detections <detections.json> \
    --classifier <rf_classifier.pkl> \
    --output <scored_detections.json>

PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/regenerate_html.py \
    --detections <scored_detections.json> \
    --czi-path <path> --output-dir <output> \
    --score-threshold 0.5
```

### Phase 4: Marker Classification

For multi-channel slides — classify each cell as positive/negative per fluorescent marker.

**Background correction is automatic.** The pipeline does pixel-level bg correction during detection. `classify_markers.py` auto-detects this (via `ch{N}_background` keys) and skips its own correction. No `--correct-all-channels` needed.

```bash
# By wavelength (preferred — auto-resolves via CZI metadata):
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/classify_markers.py \
    --detections <detections.json> \
    --marker-wavelength 647,555 \
    --marker-name NeuN,tdTomato \
    --czi-path <czi_path>

# By channel index:
    --marker-channel 1,2 --marker-name NeuN,tdTomato
```

**Methods:** `otsu` (default — auto-threshold maximizing inter-class variance), `otsu_half` (more permissive for dim markers), `gmm` (2-component Gaussian for overlapping distributions).

**Output fields:** `{marker}_class` (positive/negative), `{marker}_value`, `{marker}_threshold`, `marker_profile` (e.g., `NeuN+/tdTomato-`). Pipeline also stores `ch{N}_background`, `ch{N}_snr`, `ch{N}_mean_raw` in features.

### Phase 5: Spatial Analysis & Exploration

See **Available Analyses** section below for the full catalog.

### Phase 6: LMD Export

**Step 1 — Place 3 reference crosses** in Napari:
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i <czi_path> --flip-horizontal -o <crosses.json>
```
Keybinds: R/G/B = cross color, Space = place, S = save, U = undo, Q = save+quit. `--rotate-cw-90` is ON by default (LMD7 orientation). Crosses JSON stores `display_transform`; export scripts apply matching transforms to contours.

**Step 2 — Export XML:**
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_lmd_export.py \
    --detections <detections.json> \
    --crosses <crosses.json> \
    --output-dir <output>/lmd \
    --generate-controls --min-score 0.5 --export
    # Optional: --erosion-um 0.2  or  --erode-pct 0.05 (shrink contours for laser)
```
Batch: `--input-dir <runs> --crosses-dir <crosses>`. Max 308 wells/plate; multi-plate overflow is automatic. Empty QC wells (10%) inserted evenly. Slides without exactly 3 crosses are skipped.

**Replicate-based export** (DVP proteomics, area-normalized replicates):
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/select_mks_for_lmd.py \
    --score-threshold 0.80 --target-area 10000 --max-replicates 4

PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/lmd_export_replicates.py \
    --sampling-results lmd_replicates_full.json \
    --contours-json contours.json \
    --crosses-dir ./crosses --output-dir ./xml
```

---

## Available Analyses

Beyond the core detect → classify → LMD workflow, the pipeline supports:

| Analysis | Script | What it does |
|----------|--------|-------------|
| **RF classifier training** | `train_classifier.py` | Train random forest from annotations, 5-fold CV, feature set comparison |
| **Batch scoring** | `scripts/apply_classifier.py` | Score all detections with trained classifier (CPU, seconds) |
| **Marker classification** | `scripts/classify_markers.py` | Otsu/GMM pos/neg per channel, auto bg correction, SNR |
| **Feature exploration** | `scripts/cluster_by_features.py` | UMAP/t-SNE + Leiden/HDBSCAN, interactive plotly, --trajectory (diffusion map, pseudotime, PAGA, force-directed layout) |
| **Spatial network** | `scripts/spatial_cell_analysis.py` | Delaunay graphs, connected components, community detection, neighborhoods |
| **Interactive spatial viewer** | `scripts/generate_multi_slide_spatial_viewer.py` | KDE density contours, graph-pattern regions (linear/arc/ring/cluster), DBSCAN + convex hulls, ROI drawing + stats |
| **Vessel community analysis** | `scripts/vessel_community_analysis.py` | Multi-scale vessel structure detection (connected components + morphology + SNR) |
| **SpatialData / scverse** | `scripts/convert_to_spatialdata.py` | Export to zarr for squidpy (spatial stats), scanpy (dim reduction), anndata |
| **One-command viz** | `scripts/view_slide.py` | Classify → spatial cluster → interactive viewer → serve (all in one) |
| **Preprocessing preview** | `scripts/preview_preprocessing.py` | Before/after flat-field, photobleach correction at 1/8 resolution |
| **Nuclear counting** | `scripts/count_nuclei_per_cell.py` | Count nuclei per cell (Cellpose 2nd pass on nuclear channel), per-nucleus morph+SAM2 features |
| **Quality filter** | `scripts/quality_filter_detections.py` | Heuristic area+solidity+channel filter as RF alternative for clean slides |
| **Region detection** | `scripts/detect_regions_for_lmd.py` | Percentile-threshold any channel → morph cleanup → split → full features (morph+channel+SAM2) |
| **Region splitting** | `scripts/split_regions_for_lmd.py` | Post-process pipeline detections → watershed split large regions |
| **Replicate sampling** | `scripts/paper_figure_sampling.py` | Area-matched or spatially-clustered replicate building, 384-well assignment |
| **Transect selection** | `scripts/select_transect_cells_for_lmd.py` | Select cells along zonation transect paths for LMD export |
| **Distance bins** | `scripts/assign_distance_bins.py` | Concentric rings around vascular landmarks, distance features, spatial model comparison |
| **LMD clustering** | `scripts/cluster_detections.py` | Two-stage biological clustering for well assignment |

**SpatialData** is auto-exported at end of every pipeline run (`{celltype}_spatialdata.zarr`). Load with `spatialdata.read_zarr()`. Run squidpy spatial stats (neighborhood enrichment, co-occurrence, Moran's I) via `--run-squidpy --squidpy-cluster-key <marker_class>`.

**OME-Zarr** is auto-generated at end of every pipeline run from SHM (fast, no CZI re-read). Used for Napari viewing and cross placement. Flags: `--no-zarr` to skip, `--force-zarr` to overwrite, `--zarr-levels 5`.

---

## Architecture

### Data Flow

```
CZI file → czi_loader.py (channel resolution, tiling)
         → Direct-to-SHM loading (no RAM intermediate)
         → Preprocessing (flat-field, photobleach) on SHM views
         → Multi-GPU tile processing (multigpu_worker.py)
           → Strategy.detect_in_tile() per tile
           → Feature extraction (morph + SAM2 + optional ResNet/DINOv2)
           → Per-tile HTML cache + HDF5 masks + JSON detections
         → Deduplication (>10% pixel overlap)
         → Post-dedup pipeline (post_detection.py):
           Phase 1: contour dilation + RDP + quick means (ThreadPool)
           Phase 2: KD-tree background estimation (single-thread)
           Phase 3: bg-corrected intensity features (ThreadPool)
           Phase 4: nuclear counting (optional, --count-nuclei, single-thread GPU)
         → Finalize: JSON + CSV + HTML + OME-Zarr + SpatialData
```

### Pipeline Package (`segmentation/pipeline/`)

`run_segmentation.py` is a ~1400-line orchestrator importing from 9 pipeline modules:

| Module | Purpose |
|--------|---------|
| `cli.py` | Argparse + postprocess_args() + channel-spec resolution |
| `preprocessing.py` | Photobleach, flat-field, Reinhard normalization |
| `detection_setup.py` | `build_detection_params()` — strategy config |
| `samples.py` | HTML sample creation, tile grid, islet GMM calibration |
| `resume.py` | Checkpoint detection, tile reload, `compose_tile_rgb()` |
| `post_detection.py` | 3-phase post-dedup (contour, bg, intensity) — ThreadPool parallelized |
| `finalize.py` | Channel legend, CSV/JSON/HTML export, summary |
| `server.py` | HTTP server + Cloudflare tunnel |
| `background.py` | KD-tree local background correction (shared with classify_markers.py) |

**Dependency DAG** (no cycles): `resume → samples`, `finalize → server`, `post_detection → background → standalone`, all others standalone.

### Detection Strategies (`segmentation/detection/strategies/`)

All inherit from `base.py`, use `MultiChannelFeatureMixin` (from `mixins.py`), implement `detect_in_tile()`. Selection via `strategy_factory.py`.

### Multi-GPU Processing (`segmentation/processing/`)

**Always multi-GPU** (even `--num-gpus 1`). No separate single-GPU path.
- `multigpu_worker.py` — generic worker for ALL cell types, config dict (not positional args)
- `multigpu_shm.py` — shared memory with SIGTERM cleanup handler
- `tile_processing.py` — shared `process_single_tile()`
- Multi-node: `--tile-shard INDEX/TOTAL` round-robin, `--merge-shards` on resume

---

## Critical Code Patterns

### Device Handling

Never hardcode `device="cuda"`. Use from `segmentation.utils.device`:
- `get_default_device()` — detects cuda/mps/cpu automatically
- `device_supports_gpu()` — for Cellpose `gpu=` flag (not `torch.cuda.is_available()`)
- `empty_cache()` — handles cuda (clear cache), mps (synchronize + clear), and cpu (no-op)

### SHM Lifetime

`shm_manager.cleanup()` MUST be deferred until after post-dedup AND `_finish_pipeline()` — both read pixel data from SHM. Detection try block uses `except Exception: cleanup(); raise` for crash safety.

### Zero-Pixel Handling

- Uncorrected data → exclude zeros (CZI padding bias)
- Bg-corrected data → include zeros (real signal)
- Controlled via `_include_zeros` param on `MultiChannelFeatureMixin.extract_multichannel_features()`

### JSON I/O

Always use `atomic_json_dump()` from `segmentation.utils.json_utils` — temp file + `os.replace()`, auto-sanitizes NaN/Inf, uses orjson when available. Use `fast_json_load()` for large detection JSON. No `indent=2` anywhere.

### safe_to_uint8

Canonical in `segmentation/utils/detection_utils.py`. For uint16: simple `arr/256` — very dim for low-signal channels. Use `_percentile_normalize_single()` for proper normalization. Float >1.0 clipped to [0,255].

### Centroids

Background correction KD-tree MUST use `global_center` (slide-level), NOT `features["centroid"]` (tile-local). Canonical: `_extract_centroids()` in `background.py`. KD-tree is built once and cached across channels via `tree_and_indices` parameter (4x speedup on 4-channel slides).

### Shared Utilities (`segmentation/utils/detection_utils.py`)

- `extract_positions_um(detections, pixel_size_um=None)` — canonical position extraction with 3-level fallback: `global_center_um` → `global_center * pixel_size` → `global_x/y * pixel_size`. Auto-infers pixel_size from `area/area_um2`. Use instead of writing inline position extraction.
- `load_rf_classifier(model_path)` — generic RF classifier loader (replaces NMJ-specific `load_nmj_rf_classifier`). Handles Pipeline and dict formats, tries multiple sidecar feature-name files.
- `transform_native_to_display()` in `segmentation/lmd/contour_processing.py` — canonical LMD coordinate transform (flip_h, rot90). Single source of truth for both `run_lmd_export.py` and `lmd_export_replicates.py`.

### Logging

`get_logger(__name__)` from `segmentation.utils.logging` everywhere. No bare `logging.getLogger`.

### Coordinate System

All coordinates are [x, y] (horizontal, vertical). UID format: `{slide}_{celltype}_{x}_{y}`. CZI tiles use global coordinates; RAM arrays are 0-indexed. `loader.get_tile()` handles offsets; direct `all_channel_data` indexing must subtract `x_start, y_start`. See `docs/COORDINATE_SYSTEM.md`.

---

## CLI Reference

### Flag Gotchas

- `--no-normalize-features` disables flat-field (no `--flat-field-correction` flag exists)
- `--photobleaching-correction` (with `-ing`)
- `--sequential` does NOT exist — use `--num-gpus 1`
- `--nuclear-channel` / `--membrane-channel` are islet-only (validated only for `cell_type=='islet'`)
- `--sample-fraction` is ALWAYS 1.0 — detect 100%, use `--html-sample-fraction` to subsample HTML only

### Post-Dedup Processing (default ON)

```bash
--no-contour-processing     # Skip contour dilation + RDP
--dilation-um 0.5           # Contour dilation in micrometers (default: 0.5)
--rdp-epsilon 5.0           # RDP simplification epsilon (default: 5)
--no-background-correction  # Skip local background subtraction
--bg-neighbors 30           # KD-tree neighbors (default: 30)
```

YAML equivalents: `contour_processing`, `background_correction`, `dilation_um`, `rdp_epsilon`, `bg_neighbors`.

### Performance Options

```bash
--load-to-ram               # [Default] Direct-to-SHM loading
--num-gpus 4                # Number of GPUs (always multi-GPU architecture)
--html-sample-fraction 0.10 # Subsample HTML to 10% of detections
--max-html-samples 20000    # Hard OOM cap during per-tile accumulation
```

### Vessel-Specific Flags

```bash
--candidate-mode               # Relaxed thresholds for training data
--ring-only                    # Disable supplementary lumen-first pass
--no-smooth-contours           # Disable B-spline smoothing
--smooth-contours-factor 3.0   # Spline smoothing factor (default: 3.0)
--multi-scale                  # Multi-scale detection (coarse to fine)
```

Vessel 3-contour system: lumen (cyan, inner boundary), CD31 (green, endothelial outer), SMA (magenta, smooth muscle ring expanding from lumen). 6-type classification: artery, arteriole, vein, capillary, lymphatic, collecting_lymphatic.

### Multi-Node Sharding

```bash
# Split detection across 4 nodes
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 0/4 --resume /shared/output/dir  # node 0
    --tile-shard 1/4 --resume /shared/output/dir  # node 1

# Merge all shards
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --resume /shared/output/dir --merge-shards
```

---

## Hardware (SLURM Cluster)

- **p.hpcl93:** 19 nodes, 256 CPUs, 760G RAM, 4x L40S each — heavy GPU jobs (requires `--gres=gpu:`)
- **p.hpcl8:** 55 nodes, 24 CPUs, 380G RAM, 2x RTX 5000 each — interactive dev, CPU jobs
- Time limit: 42 days on both partitions
- Run `$XLDVP_PYTHON $REPO/scripts/system_info.py` to check live availability and get recommendations

**Resource policy:** On shared SLURM clusters, request ~75% of node CPUs/RAM to leave headroom for other users. `system_info.py` applies this automatically. Request all GPUs (GPU scheduling is typically exclusive per device). Use `--exclusive` only when you genuinely need the full node.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| OOM | Reduce `--num-gpus`, reduce tile size |
| CUDA boolean error | `mask = mask.astype(bool)` for SAM2 |
| SAM2 `_orig_hw` | `img_h, img_w = sam2_predictor._orig_hw[0]` (list of tuple) |
| HDF5 LZ4 error | `import hdf5plugin` before `h5py` |
| Network mount timeout | Socket timeout 60s automatic. Check with `ls /mnt/x/` |

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `run_segmentation.py` | Unified detection pipeline (all cell types) |
| `run_lmd_export.py` | LMD XML export (single + batch) |
| `train_classifier.py` | Train RF classifier from annotations |
| `scripts/apply_classifier.py` | Score detections with trained classifier |
| `scripts/classify_markers.py` | Marker pos/neg classification (Otsu/GMM) |
| `scripts/regenerate_html.py` | Regenerate HTML viewer from saved detections |
| `scripts/czi_info.py` | CZI channel metadata (run first!) |
| `scripts/run_pipeline.sh` | YAML config-driven SLURM launcher |
| `scripts/napari_place_crosses.py` | Interactive cross placement for LMD |
| `scripts/generate_multi_slide_spatial_viewer.py` | Unified spatial viewer with ROI + stats |
| `scripts/convert_to_spatialdata.py` | Export to SpatialData zarr (scverse) |
| `scripts/view_slide.py` | One-command: classify + spatial + viewer + serve |
| `scripts/vessel_community_analysis.py` | Multi-scale vessel structure detection |
| `scripts/spatial_cell_analysis.py` | Spatial network analysis |
| `scripts/cluster_by_features.py` | UMAP/t-SNE + Leiden/HDBSCAN, interactive plotly, --trajectory (diffmap, pseudotime, PAGA, force-directed) |
| `scripts/compare_feature_sets.py` | Compare RF feature subsets via stratified CV |
| `scripts/count_nuclei_per_cell.py` | Count nuclei per cell (Cellpose 2nd pass + per-nucleus features) |
| `scripts/detect_regions_for_lmd.py` | Percentile-threshold channel → split → full features (morph+channel+SAM2) |
| `scripts/quality_filter_detections.py` | Heuristic area+solidity+channel filter (RF alternative) |
| `scripts/split_regions_for_lmd.py` | Post-process pipeline detections → watershed split large regions |
| `scripts/paper_figure_sampling.py` | Replicate sampling (area-matched or spatial) with 384-well assignment |
| `scripts/select_transect_cells_for_lmd.py` | Select zonation transect cells for LMD |
| `scripts/cluster_detections.py` | Biological clustering for LMD wells |
| `scripts/generate_tissue_overlay.py` | Fluorescence image + cell overlay + ROI viewer |
| `scripts/assign_tissue_zones.py` | Spatially-constrained marker-based zone discovery |
| `scripts/zonation_transect.py` | Pericentral → periportal gradient analysis |
| `scripts/calculate_tissue_areas.py` | Variance-based tissue area measurement |
| `scripts/annotate_bone_regions.py` | Interactive HTML bone region annotation |
| `scripts/maturation_analysis.py` | MK maturation staging (nuclear deep features) |
| `scripts/mk_comprehensive_analysis.py` | Multi-dimensional MK feature analysis |
| `scripts/analyze_islets.py` | Spatial analysis of pancreatic islets |
| `scripts/select_mks_for_lmd.py` | MK replicate selection + multi-plate wells |
| `scripts/lmd_export_replicates.py` | Replicate-based LMD XML export |
| `scripts/system_info.py` | Environment detection + SLURM recommendations |
| `scripts/preview_preprocessing.py` | Correction preview at reduced resolution |
| `serve_html.py` | HTTP server + Cloudflare tunnel |
