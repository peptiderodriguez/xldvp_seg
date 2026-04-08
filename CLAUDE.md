# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# xldvp_seg — Image Analysis & DVP Pipelines

## Session Behaviors

These behaviors apply throughout every Claude Code session on this project:

**Context management:**
- When context reaches ~15% remaining, proactively: (1) update your auto-memory files with anything learned this session (patterns, bugs found, architectural decisions), (2) update any code documentation that changed, (3) commit uncommitted work with a descriptive message. Tell the user you're doing this so they're not surprised.
- When starting a continued session (context was compacted), read your memory files first to rebuild context before diving into work.

**Code hygiene:**
- **ALWAYS run `make format` before committing.** No exceptions. Black formatting differences between Python versions have caused repeated CI failures.
- After completing any significant code change (new feature, bug fix, refactor), review what you wrote before moving on: check for bugs/errors, missing imports, wrong dict keys, computational inefficiencies, code duplications, missing tests, and poor/stale documentation. Catch your own mistakes — don't rely on review agents to find basic issues like `sys.exit` without `import sys`.
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

**SLURM job submission (MANDATORY — no exceptions):**
- Use `run_pipeline.sh` + YAML configs as the standard workflow. Verify the generated sbatch ONCE for a new template, then reuse. Do NOT rewrite sbatch from scratch each time — that leads to errors.
- Before EVERY `sbatch` submission: verify `--dependency` job IDs are correct and not cancelled/stale, verify `--num-gpus` matches SLURM allocation, verify python path, verify input file paths exist.
- After EVERY submission: check first output lines within 30 seconds to confirm correct startup (GPU worker count, input file loaded).
- When cancelling and resubmitting: cancel ALL downstream jobs, update ALL dependency IDs, verify ALL sbatch files before resubmitting. Never submit before verifying.
- Never run heavy compute on login nodes — always SLURM.

---

## Development Commands

```bash
# Install (editable, registers xlseg CLI + xldvp_seg package)
pip install -e .

# Environment (for scripts that are not yet xlseg subcommands)
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")"; pwd)"  # or set manually
export XLDVP_PYTHON="${XLDVP_PYTHON:-$(which python)}"          # xldvp_seg conda env python

# Run all tests
$XLDVP_PYTHON -m pytest tests/ -v --tb=short

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

**Style:** Black (line-length 100), Ruff (E/F/W/I/N/UP/B/C4, E501 ignored, F841 per-file for vessel.py only). **Python 3.10 | 3.11** (both CI-tested).

### Tests

Tests in `tests/` using pytest. Run `make test` for current counts and coverage. Fixtures in `conftest.py`. Tests rely on `pip install -e .` (or `PYTHONPATH=$REPO`) for `xldvp_seg.*` imports.

**Development workflow:**
```bash
make install-dev   # Install with dev deps (pytest, ruff, black, pytest-cov)
make test          # Run all tests with coverage
make lint          # Check formatting (ruff + black)
make format        # Auto-fix formatting
```

---

## Quick Start

After `pip install -e .`, the `xlseg` CLI is available:

```bash
xlseg info /path/to/slide.czi           # Inspect CZI metadata
xlseg detect --czi-path slide.czi ...   # Run detection pipeline
xlseg classify --detections ...          # Train RF classifier
xlseg markers --detections ...           # Marker pos/neg classification
xlseg score --detections ... --classifier ... # Score detections
xlseg cluster --detections ...           # Feature clustering (UMAP/t-SNE + Leiden)
xlseg qc /path/to/output                 # Quick quality check (no HTML needed)
xlseg export-lmd --detections ...        # LMD export
xlseg serve /path/to/html               # Serve HTML viewer
xlseg system                             # Show system info
xlseg strategies                         # List detection strategies
xlseg models                             # List model checkpoints
xlseg download-models --brightfield      # Download gated HF models
```

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
| `/new-experiment` | Fast-track: inspect CZI, generate YAML config, launch pipeline |

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
| **InstanSeg** | InstanSeg 3.8M-param alternative to Cellpose | `--segmenter instanseg` with `--cell-type cell` |

---

## Pipeline Workflow: CZI → Detection → Classification → LMD

This is the DVP (Deep Visual Proteomics) workflow — spatial proteomics where LMD-cut cells go into mass spec analysis.

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
- `--photobleaching-correction` (with `-ing`) — **EXPERIMENTAL**, results unreliable — for sequential tile scans with intensity decay
- Flat-field ON by default (`--no-normalize-features` to disable; no `--flat-field-correction` flag)
- `--html-sample-fraction 0.10` — subsample HTML viewer to 10% of detections (browser-friendly)
- `--sample-fraction` is ALWAYS 1.0 — detect 100%, never suggest partial detection
- `--marker-snr-channels "SMA:1,CD31:3"` — classify markers during detection using pre-computed SNR >= 1.5 at zero extra cost (format: `"NAME:CHANNEL_INDEX,..."`)

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
  # Multi-scene CZIs: add scenes + scene_parallel
  # scenes: "0-9"           # process scenes 0 through 9
  # scene_parallel: true    # true = array job (one task/scene), false = sequential loop
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
| Post-dedup | `{celltype}_detections_postdedup.json` | Original-mask contours + features + bg correction |
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

**Methods:** `snr` (default — median-based SNR >= 1.5, robust to membrane stains with median=0 inside cells), `otsu` (auto-threshold maximizing inter-class variance), `otsu_half` (more permissive for dim markers), `gmm` (2-component Gaussian with BIC model selection — returns all-negative for unimodal data). Optional `--normalize-channel` normalizes per-channel intensities before thresholding, but is NOT recommended as default because PM membrane stains have median=0 inside cells.

**Shortcut — `--marker-snr-channels`:** Instead of running `classify_markers.py` as a separate step, pass `--marker-snr-channels "SMA:1,CD31:3"` to `xlseg detect` (or `run_segmentation.py`). This classifies markers during detection using the pre-computed SNR >= 1.5 threshold at zero extra cost. Format: `"NAME:CHANNEL_INDEX,..."`. The same output fields are produced.

**Output fields:** `{marker}_class` (positive/negative), `{marker}_value`, `{marker}_raw`, `{marker}_background`, `{marker}_snr`, `marker_profile` (e.g., `NeuN+/tdTomato-`). Pipeline also stores `ch{N}_background`, `ch{N}_snr`, `ch{N}_median_raw` in features.

### Phase 5: Spatial Analysis & Exploration

See **Available Analyses** section below for the full catalog.

**Vessel analysis — lumen-first approach** (recommended for vasculature/lymphatics):
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/segment_vessel_lumens.py \
    --zarr-path <output>/slide.ome.zarr \
    --detections <output>/top5pct/cell_detections_top5pct.json \
    --marker-names SMA,CD31 \
    --rgb-channels 1,3,0 \
    --scales 2,4,8,16 \
    --output-dir <output>/vessel_lumens/
```
Reads OME-Zarr at 4 scales, runs SAM2 auto-mask to find dark lumens, validates via marker+ cell proximity, classifies vessel types. No circularity requirement — oblique cuts and irregular lumens are valid. Outputs: `vessel_lumens.json`, `cell_detections_vessels.json`, `vessel_summary.csv`.

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
    # Contour processing (all at export time, adaptive by default):
    # --max-area-change-pct 10.0   # adaptive RDP simplification (default 10%)
    # --max-dilation-area-pct 10.0 # adaptive dilation / laser buffer (default 10%)
    # --erosion-um 0.2            # optional: shrink contours for laser
```
Batch: `--input-dir <runs> --crosses-dir <crosses>`. Max 308 wells/plate; multi-plate overflow is automatic. Replicate-based export via `lmd_export_replicates.py` (see `/analyze` Phase 5).

---

## Available Analyses

Use `/analyze` for the full interactive catalog. Key scripts beyond detect → classify → LMD:

- **Spatial**: `spatial_cell_analysis.py` (Delaunay networks; core: `xldvp_seg.analysis.spatial_network`), `cluster_by_features.py` (UMAP/t-SNE + Leiden; core: `xldvp_seg.analysis.cluster_features`), `generate_multi_slide_spatial_viewer.py` (interactive HTML viewer with fluorescence background + contours + ROI), `generate_contour_viewer.py` (contour overlays on CZI fluorescence with pan/zoom, group toggling, click-to-inspect metadata; core: `xldvp_seg.visualization`)
- **Vessel**: `segment_vessel_lumens.py` (SAM2 lumen-first: OME-Zarr multi-scale → dark-lumen detection → marker-cell validation → vessel typing), `detect_vessel_structures.py` (graph topology: ring/arc/strip from marker+ cells), `vessel_community_analysis.py` (multi-scale + SNR). Shared characterization: `xldvp_seg.analysis.vessel_characterization`
- **Curvilinear**: `detect_curvilinear_patterns.py` (KD-tree graph → linearity → strip/ribbon detection; core: `xldvp_seg.analysis.pattern_detection`)
- **Markers**: `classify_markers.py` (median SNR / Otsu / GMM; core: `xldvp_seg.analysis.marker_classification`)
- **LMD sampling**: `sliding_window_sampling.py` (core: `xldvp_seg.analysis.sliding_window_sampling`), `paper_figure_sampling.py`, `select_transect_cells_for_lmd.py`, `cluster_detections.py`
- **Nuclear counting**: integrated via `--count-nuclei` (default ON), or standalone `count_nuclei_per_cell.py` for existing runs
- **SpatialData**: auto-exported as `{celltype}_spatialdata.zarr` every run. `--run-squidpy` for spatial stats.
- **OME-Zarr**: auto-generated from SHM. `--no-zarr` to skip, `--force-zarr` to overwrite.

---

## Architecture

### Data Flow

```
CZI file → czi_loader.py (channel resolution, tiling)
         → shm_setup.py: Direct-to-SHM loading (no RAM intermediate)
         → Preprocessing (flat-field, photobleach) on SHM views
         → detection_loop.py: Multi-GPU tile processing (multigpu_worker.py)
           → Strategy.detect_in_tile() per tile
           → Feature extraction (morph + SAM2 + optional ResNet/DINOv2)
           → Per-tile HTML cache + HDF5 masks + JSON detections
         → detection_loop.py: Deduplication (>10% pixel overlap)
         → Post-dedup pipeline (post_detection.py):
           Phase 1: contour extraction + per-cell median intensities (ThreadPool)
           Phase 2: KD-tree local background estimation (single-thread)
           Phase 3: bg-corrected intensity features from original mask (ThreadPool)
           Phase 4: nuclear counting (optional, --count-nuclei, single-thread GPU)
         → Finalize: JSON + CSV + HTML + OME-Zarr + SpatialData
```

### Pipeline Package (`xldvp_seg/pipeline/`)

`run_segmentation.py` is a ~1,030-line orchestrator importing from 11 pipeline modules:

| Module | Purpose |
|--------|---------|
| `cli.py` | Argparse + postprocess_args() + channel-spec resolution |
| `preprocessing.py` | Photobleach, flat-field, Reinhard normalization |
| `detection_setup.py` | `build_detection_params()` — strategy config |
| `shm_setup.py` | Shared memory allocation + CZI channel loading |
| `detection_loop.py` | Tile dispatch, multi-GPU detection, shard merge, dedup |
| `samples.py` | HTML sample creation, tile grid, islet GMM calibration |
| `resume.py` | Checkpoint detection, tile reload, `compose_tile_rgb()` |
| `post_detection.py` | 3-phase post-dedup + optional Phase 4 nuclear counting (original-mask contour extraction, bg, intensity, nuclei) — ThreadPool parallelized |
| `finalize.py` | Channel legend, CSV/JSON/HTML export, summary |
| `server.py` | HTTP server + Cloudflare tunnel |
| `background.py` | KD-tree local background correction (shared with classify_markers.py) |

**Dependency DAG** (no cycles): `detection_loop → shm_setup`, `resume → samples`, `finalize → server`, `post_detection → background → standalone`, all others standalone.

### Detection Strategies (`xldvp_seg/detection/strategies/`)

All inherit from `base.py`, use `MultiChannelFeatureMixin` (from `mixins.py`), implement `detect_in_tile()`. Strategies self-register via `@register_strategy` decorator in `xldvp_seg/detection/registry.py`. Selection via `strategy_factory.py` (registry lookup + per-strategy kwargs builder).

### Vessel Classification (`xldvp_seg/classification/`)

Four RF classifiers sharing `BaseVesselClassifier` (`base.py`): `VesselDetectorRF` (binary), `VesselClassifier` (3-class), `ArteryVeinClassifier`, `VesselTypeClassifier` (6-class). Base class provides shared save/load (joblib), feature importance, feature extraction with template hooks (`_coerce_value`, `_compute_derived_features`). Predict/train/evaluate stay in subclasses (divergent signatures).

### Custom Exceptions (`xldvp_seg/exceptions.py`)

Domain-specific exception hierarchy with dual-inheritance from builtins for backward compatibility: `XldvpSegError` (base), `ConfigError(ValueError)`, `DataLoadError(IOError)`, `DetectionError(RuntimeError)`, `ClassificationError(RuntimeError)`, `ExportError(RuntimeError)`, `ChannelResolutionError(ValueError)`. Existing `except ValueError` blocks still catch new types.

### LMD Export (`xldvp_seg/lmd/`)

`export.py` contains 18 pure-logic functions promoted from `run_lmd_export.py`: detection loading/filtering, contour extraction from HDF5, spatial control generation, well assignment (serpentine ordering), XML export via py-lmd. `contour_processing.py` handles coordinate transforms. `well_plate.py` handles plate geometry. `run_lmd_export.py` retains CLI orchestration only.

### Model Registry (`xldvp_seg/models/registry.py`)

Metadata catalog for all models (feature extractors + segmenters). Tracks name, feature_dim, modality (fluorescence/brightfield/both), license, HuggingFace URL, and gated status. Does NOT handle loading — that stays in `ModelManager`. Use `list_models(modality="brightfield")` to filter. Brightfield FMs (UNI2, Virchow2, CONCH, Phikon-v2) are gated on HuggingFace — download via `xlseg download-models --brightfield`.

### Multi-GPU Processing (`xldvp_seg/processing/`)

**Always multi-GPU** (even `--num-gpus 1`). No separate single-GPU path.
- `multigpu_worker.py` — generic worker for ALL cell types, config dict (not positional args)
- `multigpu_shm.py` — shared memory with SIGTERM cleanup handler
- `tile_processing.py` — shared `process_single_tile()`
- Multi-node: `--tile-shard INDEX/TOTAL` round-robin, `--merge-shards` on resume

### ROI-Restricted Detection (`xldvp_seg.roi`)

Reusable utilities for detection within specific regions: `roi.common` (bbox, filtering, multi-GPU), `roi.marker_threshold` (Otsu on marker signal), `roi.circular_objects` (TMA cores, islets), `roi.from_file` (polygon/mask). Examples: `examples/islet/`, `examples/tma/`, `examples/bone_marrow/`.

### Analysis Modules (`xldvp_seg/analysis/`)

Post-detection analysis functions promoted from scripts/ into the package for clean programmatic access:

| Module | Purpose |
|--------|---------|
| `marker_classification.py` | Marker pos/neg classification (SNR, Otsu, GMM) — used by `xlseg markers` and `tl.markers()` |
| `cluster_features.py` | Feature selection, matrix building, normalization, cluster labeling — `ClusteringConfig` dataclass replaces SimpleNamespace — used by `xlseg cluster` |
| `spatial_network.py` | Delaunay networks, Louvain communities, RF/morph UMAP — used by `tl.spatial()` |
| `pattern_detection.py` | Strip/cluster spatial pattern classification (curvilinear detection) |
| `sliding_window_sampling.py` | Skeleton-based spatial sampling along ROI centerlines for LMD |
| `aggregation.py` | Slide-level and cohort-level feature aggregation |
| `nuclear_count.py` | Cellpose-based nuclear segmentation within cells |
| `omic_linker.py` | Morphology-to-proteomics bridge (DVP linking). Supports dvp-io for direct search engine report parsing (AlphaDIA, DIANN, MaxQuant, Spectronaut, etc.) |
| `vessel_characterization.py` | Shared vessel analysis: marker composition, spatial layering, vessel typing, lumen/wall morphometry |

### Visualization Package (`xldvp_seg/visualization/`)

Reusable HTML visualization components extracted from the monolithic spatial viewer. Used by `generate_multi_slide_spatial_viewer.py` and `generate_contour_viewer.py`.

| Module | Purpose |
|--------|---------|
| `fluorescence.py` | CZI thumbnail loading + base64 encoding (`read_czi_thumbnail_channels`, `encode_channel_b64`, `parse_scene_index`) |
| `colors.py` | Color palettes + group assignment (`BINARY_COLORS`, `QUAD_COLORS`, `AUTO_COLORS`, `hsl_palette`, `assign_group_colors`) |
| `encoding.py` | Binary data encoding for HTML (`encode_float32_base64`, `encode_uint8_base64`, `safe_json`, `build_contour_js_data`) |
| `data_loading.py` | Detection JSON streaming + position/group extraction (`compute_auto_eps`, `extract_position_um`, `extract_group`, `load_slide_data`, `discover_slides`, `apply_top_n_filtering`) |
| `graph_patterns.py` | Spatial graph pattern detection (`compute_graph_patterns`) |
| `html_builder.py` | Group index construction, position serialization, auto-eps collection, region compaction (`build_group_index`, `serialize_slide_positions`, `collect_auto_eps`, `compact_region_data`) |
| `js_loader.py` | Composable JS component loading (`load_js`) |
| `js/` | 17 reusable Canvas 2D components (pan/zoom, contour rendering, viewport culling, metadata panel, controls, init, regions, render_panel) |

### HTML Export (`xldvp_seg/io/`)

Split across focused modules with backward-compatible re-exports in `html_export.py`:

| Module | Purpose |
|--------|---------|
| `html_export.py` | Page generators: `generate_annotation_page`, `generate_index_page`, `generate_dual_index_page`, `export_samples_to_html`, vessel variants. Re-exports all names from submodules for backward compatibility. |
| `html_styles.py` | CSS generators: `get_css()`, `get_vessel_css()` |
| `html_scripts.py` | JS generators: `get_js()`, `get_vessel_js()`, `generate_preload_annotations_js()` |
| `html_utils.py` | Image utilities: `percentile_normalize`, `draw_mask_contour`, `image_to_base64`, `compose_tile_rgb`, `_esc`, HDF5 compression constants |
| `html_generator.py` | Class-based `HTMLPageGenerator` API for per-tile HTML generation |
| `html_batch_export.py` | MK/HSPC batch export: `load_samples_from_ram`, `export_mk_hspc_html_from_ram`, index/page generators |

### Training Utilities (`xldvp_seg/training/`)

| Module | Purpose |
|--------|---------|
| `feature_loader.py` | Load features + annotations, feature set filtering — used by `tl.train()` |

---

## Critical Code Patterns

### Random Seeds & Reproducibility

Pipeline sets numpy, stdlib, and torch seeds at startup (`--random-seed`, default 42). `torch.cuda.manual_seed_all()` is called when CUDA is available. **GPU non-determinism:** cuDNN and CUDA atomics mean results may differ slightly across runs even with fixed seeds. Full determinism requires `torch.use_deterministic_algorithms(True)` which incurs ~10-20% overhead and is not enabled by default. This is a known limitation.

### Device Handling

Never hardcode `device="cuda"`. Use from `xldvp_seg.utils.device`:
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

Always use `atomic_json_dump()` from `xldvp_seg.utils.json_utils` — temp file + `os.replace()`, auto-sanitizes NaN/Inf, uses orjson when available. Use `fast_json_load()` for large detection JSON. No `indent=2` anywhere.

### safe_to_uint8

Canonical in `xldvp_seg/utils/detection_utils.py`. For uint16: simple `arr/256` — very dim for low-signal channels. Use `_percentile_normalize_single()` for proper normalization. Float >1.0 clipped to [0,255].

### Centroids

Background correction KD-tree MUST use `global_center` (slide-level), NOT `features["centroid"]` (tile-local). Canonical: `_extract_centroids()` in `background.py`. KD-tree is built once and cached across channels via `tree_and_indices` parameter (4x speedup on 4-channel slides). Background estimate is median-based: per-cell background = median of neighbor median intensities. SNR = median_raw / median_of_neighbor_medians. Known limitation: KD-tree neighbors may span tissue boundaries (vessel walls, tumor margins) — verify marker classification visually at boundaries.

### Shared Utilities (`xldvp_seg/utils/detection_utils.py`)

- `extract_positions_um(detections, pixel_size_um=None, return_indices=False)` — canonical position extraction with 3-level fallback: `global_center_um` → `global_center * pixel_size` → `global_x/y * pixel_size`. Auto-infers pixel_size from `area/area_um2`. Pass `return_indices=True` to get a 3-tuple `(positions, pixel_size, valid_indices)`. Use instead of writing inline position extraction.
- `load_rf_classifier(model_path)` — generic RF classifier loader (replaces NMJ-specific `load_nmj_rf_classifier`). Handles Pipeline and dict formats, tries multiple sidecar feature-name files.
- `transform_native_to_display()` in `xldvp_seg/lmd/contour_processing.py` — canonical LMD coordinate transform (flip_h, rot90). Single source of truth for both `run_lmd_export.py` and `lmd_export_replicates.py`.

### Graph Topology (`xldvp_seg/utils/graph_topology.py`)

Shared graph topology analysis for spatial structure detection. Used by both `detect_curvilinear_patterns.py` (mesothelium strips) and `detect_vessel_structures.py` (vessel rings/arcs/strips).

- **Graph construction**: `build_radius_graph_sparse()` uses scipy.sparse for CC (scalable to 100K+ cells), `build_component_subgraph()` for per-component NetworkX analysis.
- **Strip metrics**: `double_bfs_diameter()`, `component_linearity()`, `component_width()` (half-width from centerline), `path_length_um()`.
- **Ring/arc metrics (graph)**: `ring_score()` (angular connectivity with AR > 3 guard), `arc_fraction()` (max contiguous arc, same AR guard).
- **Ring/arc metrics (geometric/PCA)**: `elongation()`, `circularity()` (clamped [0,1]), `hollowness()`, `has_curvature()` (min 10 points).
- **Geometry**: `safe_hull_area()`, `bounding_box_aspect_ratio()`.
- **Convenience**: `compute_all_metrics()` returns all metrics as a dict for a single component.

### Logging

`get_logger(__name__)` from `xldvp_seg.utils.logging` everywhere. No bare `logging.getLogger`.

### Coordinate System

All coordinates are [x, y] (horizontal, vertical). UID format: `{slide}_{celltype}_{x}_{y}`. CZI tiles use global coordinates; RAM arrays are 0-indexed. `loader.get_tile()` handles offsets; direct `all_channel_data` indexing must subtract `x_start, y_start`. See `docs/COORDINATE_SYSTEM.md`.

---

## CLI Reference

### Flag Gotchas

- `--no-normalize-features` disables flat-field (no `--flat-field-correction` flag exists)
- `--photobleaching-correction` (with `-ing`) — **EXPERIMENTAL**, results unreliable
- `--sequential` does NOT exist — use `--num-gpus 1`
- `--nuclear-channel` / `--membrane-channel` are islet-only (validated only for `cell_type=='islet'`)
- `--tissue-channels "2,3,5"` — marker channels that identify tissue regions worth segmenting. Tiles with low signal in these channels are skipped. Also controls morphological feature extraction and SAM2 embeddings. NOT for Cellpose segmentation (use `--channel-spec`). Required for islet; optional for other cell types. Legacy: `--islet-display-channels` / `--display-channels` / `--rgb-channels` still accepted.
- `--sample-fraction` is ALWAYS 1.0 — detect 100%, use `--html-sample-fraction` to subsample HTML only
- `--segmenter {cellpose,instanseg}` — alternative cell segmenter (default: cellpose). Requires `pip install -e .[instanseg]`. Only applies to `--cell-type cell`.
- `--marker-snr-channels "SMA:1,CD31:3"` — auto-classify markers during detection using pre-computed SNR >= 1.5. Replaces the separate `classify_markers.py` step. Format: `"NAME:CHANNEL_INDEX,..."`. Requires `--all-channels`.

### Deduplication

```bash
--dedup-method mask_overlap  # [Default] Pixel-exact mask overlap (>10% of smaller mask)
--dedup-method iou_nms       # Contour-based IoU NMS with Shapely STRtree (faster, less memory)
--iou-threshold 0.2          # IoU threshold for iou_nms method (default: 0.2)
```

YAML equivalents: `dedup_method`, `iou_threshold`. Note: IoU and overlap-fraction are different metrics — IoU may miss size-mismatched overlaps. Benchmark both before switching default.

### Post-Dedup Processing (default ON)

```bash
--no-contour-processing     # Skip contour extraction from HDF5 masks
--no-background-correction  # Skip local background subtraction
--bg-neighbors 30           # KD-tree neighbors (default: 30)
# --dilation-um, --rdp-epsilon: deprecated — dilation and RDP are now
# applied at LMD export time only (see run_lmd_export.py --max-area-change-pct)
```

YAML equivalents: `contour_processing`, `background_correction`, `bg_neighbors`.

Features are always extracted from the **original** segmentation mask (not a dilated or simplified version). Contour simplification and dilation are applied at LMD export time only, both with adaptive tolerances:
- `--max-area-change-pct 10.0` — adaptive RDP: max 10% symmetric-difference deviation
- `--max-dilation-area-pct 10.0` — adaptive dilation: max 10% area increase for laser buffer
Set either to 0 to fall back to fixed values (`--rdp-epsilon`, `--dilation-um`).

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

Vessel 3-contour system: lumen (cyan, inner boundary), CD31 (green, endothelial outer), SMA (magenta, smooth muscle ring expanding from lumen). 7-type classification: artery, arteriole, vein, venule, capillary, lymphatic, collecting_lymphatic.

### Multi-Node Sharding

```bash
# Split detection across 4 nodes
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 0/4 --resume /shared/output/dir  # node 0
    --tile-shard 1/4 --resume /shared/output/dir  # node 1

# Merge all shards (MUST include --resume — without it, argparse errors out silently)
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --resume /shared/output/dir --merge-shards
```

**CRITICAL**: `--merge-shards` REQUIRES `--resume <shared-output-dir>`. Without `--resume`, the command prints argparse help and exits with error code 0 — which means SLURM reports "COMPLETED" even though nothing was merged. Always verify the merge log shows actual detection counts, not just argparse output.

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

**Core:** `run_segmentation.py` (detection), `run_lmd_export.py` (LMD XML), `train_classifier.py` (RF), `serve_html.py` (viewer). SLURM: `scripts/run_pipeline.sh` (YAML-driven launcher). **CLI:** `xlseg` with 13 subcommands (info, detect, classify, cluster, markers, score, qc, export-lmd, serve, system, models, strategies, download-models).

**Scripts (`scripts/`):** 31 reusable tools — `ls scripts/` for full list. Key: `czi_info.py`, `classify_markers.py`, `apply_classifier.py`, `regenerate_html.py`, `generate_multi_slide_spatial_viewer.py`, `generate_contour_viewer.py`, `detect_vessel_structures.py`, `segment_vessel_lumens.py`, `count_nuclei_per_cell.py`, `sliding_window_sampling.py`. Core logic of 6 promoted scripts now lives in `xldvp_seg/analysis/` for programmatic access (scripts delegate to package modules).

**Examples (`examples/`):** Project-specific scripts by experiment — `bone_marrow/`, `mesothelium/`, `islet/`, `tma/`, `liver/`, `nmj/`, `vessel/`, `tissue_pattern/`, `configs/` (YAML templates).
