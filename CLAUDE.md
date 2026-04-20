# CLAUDE.md

Guidance for Claude Code sessions on the `xldvp_seg` project — Spatial cell segmentation and DVP pipelines for CZI microscopy.

## Session Behaviors (load-bearing)

**Context management:**
- At ~15% context remaining: update auto-memory, sync docs for changes made this session, commit uncommitted work. Tell the user you're doing this.
- When continuing a session (context compacted): read memory files first.

**Code hygiene:**
- **ALWAYS run `make format` before committing.** Black version skew has caused repeated CI failures.
- After significant code changes, self-review before moving on: bugs, missing imports, wrong dict keys, inefficiencies, duplications, missing tests, stale docs. Don't rely on review agents for basics like `sys.exit` without `import sys`.
- Keep CLAUDE.md, `docs/*.md`, and slash commands in sync with code.
- When you fix a bug, check if the same pattern exists elsewhere and fix all instances.

**Communication:**
- Give brief status updates during long operations (SLURM, reviews, large reads).
- Flag unexpected findings to the user — don't silently work around them.
- **Always use AskUserQuestion** when asking — never list questions inline in text.
- **Always enter plan mode first** for implementation tasks.

**Pipeline-specific:**
- Always run `czi_info.py` before writing any channel configuration.
- Never hardcode pixel sizes, channel indices, or file paths that should come from CZI metadata.
- Prefer `--channel-spec "detect=MARKER"` over raw `--channel N`.

**SLURM (MANDATORY):**
- Use `run_pipeline.sh` + YAML configs. Verify a new sbatch template ONCE, then reuse. Don't rewrite from scratch each time.
- Before EVERY submission: verify `--dependency` job IDs aren't stale, verify `--num-gpus` matches allocation, verify python path + input paths.
- After EVERY submission: check first output lines within 30s to confirm startup.
- When cancelling/resubmitting: cancel ALL downstream jobs, update ALL dependency IDs, verify ALL sbatch files before resubmitting.
- Never run heavy compute on login nodes.

---

## Development Commands

```bash
./install.sh                            # Install pipeline + deps from lock file (Linux/Mac)
./install.sh --dev                      # Adds pytest/ruff/black/mkdocs
./install.sh --with-claude-code         # Also installs Claude Code CLI (opt-in)
make test                               # All tests + coverage
make lint                               # ruff + black check
make format                             # Auto-fix formatting
```

**Never** run `pip install -e .` standalone — combining scverse transitive deps with our constraints hits `ResolutionTooDeep`. `install.sh` installs from the pre-solved `requirements-lock.txt` to avoid that path.

**Style:** Black (line-length 100), Ruff (E/F/W/I/N/UP/B/C4, E501 ignored). Python 3.11 (CI-tested). Tests in `tests/` using pytest; rely on `pip install -e .` for `xldvp_seg.*` imports.

---

## Quick Start

After `pip install -e .`, the `xlseg` CLI is available:

```bash
xlseg info slide.czi              # Inspect CZI metadata (ALWAYS first)
xlseg detect --czi-path ... --cell-type cell --channel-spec "cyto=PM,nuc=488" --all-channels
xlseg classify / markers / score / cluster / qc / export-lmd / serve / system
xlseg strategies / models / download-models --brightfield
```

Type `/analyze` inside Claude Code for guided pipeline. Other slash commands: `/status`, `/czi-info`, `/classify`, `/lmd-export`, `/vessel-analysis`, `/view-results`, `/spatialdata`, `/new-experiment`, `/preview-preprocessing`. All in `.claude/commands/`.

**Docs:** `docs/GETTING_STARTED.md`, `NMJ_PIPELINE_GUIDE.md`, `LMD_EXPORT_GUIDE.md`, `COORDINATE_SYSTEM.md`, `VESSEL_COMMUNITY_ANALYSIS.md`, `VESSEL_LUMEN_THRESHOLD_PIPELINE.md`, `BLOCKFACE_REGISTRATION.md`.

**Cell types** (`--cell-type`): `cell` (Cellpose + SAM2), `nmj`, `mk`, `vessel`, `islet`, `mesothelium`, `tissue_pattern`. Alt segmenter: `--segmenter instanseg`.

---

## Pipeline Workflow

### Phase 1 — Inspection
**CRITICAL: CZI channel order is NOT wavelength-sorted.** Indices follow acquisition/detector assignment. Always run `czi_info.py` first; confirm the table with the user. Example: `[1]=647nm before [3]=555nm` is normal.

Use `--channel-spec` for resolution by name/wavelength:
```
--channel-spec "cyto=PM,nuc=488"   # name + wavelength
--channel-spec "detect=647"         # direct wavelength
```
Resolution order: integer → wavelength (±10nm) → marker name (filename) → CZI metadata name. For YAML: `channel_map:` block.

### Phase 2 — Detection

**Check cluster first:** `scripts/system_info.py` for partition availability + resource recommendations.

Key flags:
- `--all-channels` — per-channel intensity features. Always use for multi-channel.
- `--extract-deep-features` — ResNet+DINOv2 (+6144 dims). Morph-only (78) is often competitive; try if morph F1 < 0.85.
- `--marker-snr-channels "SMA:1,CD31:3"` — classify markers during detection at zero cost (SNR≥1.5). Replaces separate `classify_markers.py` step.
- `--html-sample-fraction 0.10` — browser-friendly HTML viewer.
- `--sample-fraction` is ALWAYS 1.0.

YAML launch: `scripts/run_pipeline.sh configs/my_experiment.yaml` (see `examples/configs/`).

Local: `xlseg detect --czi-path ... --cell-type cell --channel-spec "..." --all-channels --num-gpus N`

**Resume:** SLURM → add `resume_dir:` to YAML. Local → `--resume /path/to/run_dir`. Pipeline auto-detects the latest checkpoint.

**Checkpoints:** per-tile dirs → merge shards → dedup → post-dedup (contours + features + bg) → finalize (JSON/CSV/HTML).

### Phase 3 — Annotation & Classification
Serve HTML viewer → annotate yes/no → train RF → score. Feature sets: `morph` (78D), `morph_sam2` (334D), `channel_stats`, `all` (6,478D).

### Phase 4 — Marker Classification

Background correction is automatic (pipeline does pixel-level bg during detection; `classify_markers.py` detects `ch{N}_background` keys and skips its own correction).

```bash
xlseg markers --detections ... --marker-wavelength 647,555 --marker-name NeuN,tdTomato --czi-path ...
# Or by index: --marker-channel 1,2
```

**Methods:** `snr` (default, median-based ≥1.5, robust to membrane stains), `otsu`, `otsu_half` (dim markers), `gmm` (2-component with BIC, delta≥6). `include_zeros` auto-enabled when bg correction is active. Don't use `--normalize-channel` — membrane stains have median=0 inside cells.

**Shortcut:** pass `--marker-snr-channels "SMA:1,CD31:3"` to `xlseg detect` instead of a separate markers step.

**Output fields:** `{marker}_class/value/raw/background/snr`, `marker_profile` (e.g. `NeuN+/tdTomato-`), plus per-channel `ch{N}_{background,snr,median_raw}`.

### Phase 5 — Spatial Analysis
See **Available Analyses** below.

### Phase 6 — LMD Export
Place 3 reference crosses in Napari → `run_lmd_export.py`. Adaptive RDP (10% shape tolerance) + dilation (10% area). Max 308 wells/plate, multi-plate overflow. See `docs/LMD_EXPORT_GUIDE.md`.

---

## Available Analyses

Use `/analyze` for the interactive catalog. Scripts beyond detect → classify → LMD:

- **Region segmentation**: `segment_regions.py` (SAM2 on fluorescence thumbnails), `assign_cells_to_regions.py`, `generate_region_viewer.py` (interactive HTML, per-region nuclear stats). Core: `xldvp_seg.analysis.region_segmentation`.
- **Per-region feature exploration**: `region_pca_viewer.py` (PCA→UMAP per region with 4 clusterings: kmeans-elbow / Leiden / HDBSCAN-PCA / HDBSCAN-UMAP; color toggle in HTML), `combined_region_viewer.py` (spatial map + UMAP side-by-side — click region on map to jump UMAP), `region_multinuc_plot.py` (per-region multinucleation histogram + Tukey + GMM outlier detection).
- **Global cluster + spatial divergence**: `global_cluster_spatial_viewer.py` — inverse of per-region analysis. Clusters ALL nucleated cells globally; per-cluster spatial-distribution metrics (`k_90`, `focal_multimodal`, entropy, `n_major_regions`) rank clusters from "organ-specific" through "focal multi-modal" to "ubiquitous". 4-method toggle (Leiden/kmeans/HDBSCAN-PCA/HDBSCAN-UMAP all on full set). Core: `xldvp_seg.analysis.region_clustering`.
- **Transcript export**: `export_transcript.py` (Claude Code session JSONL → markdown/HTML; `--mode curate` for keep/skip review, `--mode present` for PNG export).
- **Spatial**: `spatial_cell_analysis.py` (Delaunay), `cluster_by_features.py` (UMAP + Leiden), `generate_multi_slide_spatial_viewer.py`, `generate_contour_viewer.py`. Cores in `xldvp_seg.analysis.*`.
- **Vessel (4 tools)**: `detect_vessel_lumens_threshold.py` (threshold + watershed on OME-Zarr, CPU, recommended for whole-mount — see `docs/VESSEL_LUMEN_THRESHOLD_PIPELINE.md`), `score_vessel_lumens.py` (RF), `generate_lumen_annotation.py` (card-grid HTML), `assign_vessel_wall_cells.py` (per-marker wall cells + LMD replicates). Also: `segment_vessel_lumens.py` (SAM2 lumen-first), `detect_vessel_structures.py` (graph topology), `vessel_community_analysis.py`. Shared: `xldvp_seg.analysis.vessel_characterization`.
- **Curvilinear patterns**: `detect_curvilinear_patterns.py` (strips/ribbons via graph linearity).
- **Markers**: `classify_markers.py` (SNR/Otsu/GMM).
- **LMD sampling**: `sliding_window_sampling.py`, `paper_figure_sampling.py`, `select_transect_cells_for_lmd.py`, `cluster_detections.py`.
- **Nuclear counting**: `--count-nuclei` (default ON) or standalone `count_nuclei_per_cell.py`.
- **SpatialData**: auto-exported as `{celltype}_spatialdata.zarr`. `--run-squidpy` for stats.
- **OME-Zarr**: auto-generated from SHM. `--no-zarr` / `--force-zarr`.
- **Block-face registration**: `docs/BLOCKFACE_REGISTRATION.md`.

---

## Architecture

**Data flow:** CZI → channel resolution → direct-to-SHM load → preprocessing (flat-field, photobleach) → multi-GPU tile detection → per-tile HDF5+JSON+HTML → shard merge → dedup (>10% overlap) → post-dedup 3-phase (contours → bg → bg-corrected features; Phases 1/3 ProcessPool due to h5py phil lock) → optional Phase 4 nuclear counting (multi-GPU) → finalize (CSV/HTML/OME-Zarr/SpatialData).

**Package layout** (`xldvp_seg/`):
- `pipeline/` — 11 modules orchestrated by `run_segmentation.py`: cli, preprocessing, detection_{setup,loop}, shm_setup, samples, resume, post_detection, multigpu_phase4, finalize, server. DAG: detection_loop→shm_setup, post_detection→`analysis.background` (background correction moved to analysis/ in Apr 2026).
- `processing/` — `multiprocess_tiles.TileProcessor` (CPU ProcessPool for per-tile HDF5 work, reusable), `multigpu_worker`, `multigpu_shm`, `shm_attach`, `tile_processing`. Always multi-GPU (even `--num-gpus 1`).
- `detection/strategies/` — all inherit `base.py`, use `MultiChannelFeatureMixin`, implement `detect_in_tile()`. Self-register via `@register_strategy`. Selection via `strategy_factory.py`.
- `classification/` — 4 RF classifiers sharing `BaseVesselClassifier`: `VesselDetectorRF` (binary), `VesselClassifier` (3-class), `ArteryVeinClassifier`, `VesselTypeClassifier` (6-class).
- `analysis/` — 12 modules promoted from scripts for programmatic use: `marker_classification`, `cluster_features` (ClusteringConfig dataclass), `spatial_network`, `pattern_detection`, `sliding_window_sampling`, `aggregation`, `nuclear_count`, `omic_linker` (supports dvp-io), `vessel_characterization`, `region_segmentation`, `region_clustering` (per-region PCA/UMAP + 4 clusterings: kmeans-elbow / Leiden / HDBSCAN-PCA / HDBSCAN-UMAP), `background` (bg subtraction + per-cell bg-corrected feature recomputation; moved from `pipeline/` — was always pure analysis).
- `roi/` — ROI-restricted detection: `common`, `marker_threshold`, `circular_objects` (TMA/islets), `from_file`. See `examples/{islet,tma,bone_marrow}/`.
- `models/registry.py` — model metadata catalog (feature dim, modality, license, HF URL, gated status). Loading stays in `ModelManager`. Brightfield FMs (UNI2, Virchow2, CONCH, Phikon-v2) gated — `xlseg download-models --brightfield`.
- `lmd/` — 18 pure functions in `export.py` (loading, contour extraction, spatial controls, well assignment, XML via py-lmd); `contour_processing`, `well_plate`.
- `visualization/` — 9 modules + 18 JS components (`fluorescence`, `colors`, `encoding`, `data_loading`, `js_loader`, `region_viewer`, etc.).
- `io/` — 6 HTML modules: `html_export` (pages), `html_styles`, `html_scripts`, `html_utils`, `html_generator`, `html_batch_export`.
- `training/feature_loader.py` — load features + annotations, feature set filtering.
- `exceptions.py` — dual-inheritance hierarchy: `XldvpSegError`, `ConfigError(ValueError)`, `DataLoadError(IOError)`, `DetectionError(RuntimeError)`, `ClassificationError(RuntimeError)`, `ExportError(RuntimeError)`, `ChannelResolutionError(ValueError)`. Existing `except ValueError` still catches new types.

---

## Critical Code Patterns

**Random seeds:** `--random-seed` (default 42) sets numpy/stdlib/torch seeds. cuDNN non-determinism means slight variation across runs; full determinism needs `torch.use_deterministic_algorithms(True)` (~10-20% overhead, not default).

**Device handling:** Use `xldvp_seg.utils.device`: `get_default_device()`, `device_supports_gpu()` (for Cellpose `gpu=`, not `torch.cuda.is_available()`), `empty_cache()` (handles cuda/mps/cpu). Never hardcode `device="cuda"`.

**SHM lifetime:** `shm_manager.cleanup()` MUST defer until after post-dedup AND `_finish_pipeline()` — both read pixel data. Detection uses `except Exception: cleanup(); raise` for crash safety.

**Zero-pixel handling:** Uncorrected data → exclude zeros (CZI padding creates artificial zeros that bias stats low). Bg-corrected data → include zeros (corrected zeros represent real below-background signal). Controlled by `_include_zeros=True/False` on `MultiChannelFeatureMixin.extract_multichannel_features()`. `classify_markers.py` auto-enables this when it detects bg correction was run.

**Channel ratios:** `ch{N}_{M}_ratio` and `channel_specificity` use **median** intensities for robustness to outlier pixels. Consistent across initial detection (`_compute_channel_ratios` in `mixins.py`) and bg recomputation (`correct_all_channels` in `background.py`) — keep in sync if you touch one.

**JSON I/O:** Always `atomic_json_dump()` from `xldvp_seg.utils.json_utils` (temp + `os.replace`, sanitizes NaN/Inf, uses orjson). `fast_json_load()` for large files. No `indent=2`.

**Centroids (TRAP):** Background KD-tree MUST use `global_center` (slide-level coords), NOT `features["centroid"]` (tile-local coords). Using tile-local centroids silently produces wrong neighbor lookups across tile boundaries. Canonical extractor: `_extract_centroids()` in `background.py`. Tree cached across channels via `tree_and_indices` (4x speedup on 4-channel slides). Bg = median of neighbor medians; SNR = median_raw / median_of_neighbor_medians. Known limit: KD-tree neighbors may span tissue boundaries (vessel walls, tumor margins) — verify marker classification visually at edges.

**Shared utilities** (`xldvp_seg/utils/detection_utils.py`):
- `extract_positions_um()` — canonical position extraction with 3-level fallback (`global_center_um` → `global_center * px` → `global_x/y * px`). Auto-infers pixel_size from `area/area_um2`.
- `load_rf_classifier()` — generic RF loader (Pipeline + dict formats, sidecar feature-name files).
- `extract_feature_matrix()` — pre-allocates numpy, fills missing=0, returns `(X[valid], valid_indices)`.
- `transform_native_to_display()` in `xldvp_seg/lmd/contour_processing.py` — canonical LMD coord transform (flip_h, rot90).

**Graph topology** (`xldvp_seg/utils/graph_topology.py`): shared by `detect_curvilinear_patterns.py` and `detect_vessel_structures.py`. Strip metrics (`double_bfs_diameter`, `component_linearity`, `component_width`), ring/arc graph metrics (`ring_score`, `arc_fraction` — both with AR>3 guard), geometric/PCA metrics (`elongation`, `circularity`, `hollowness`), `compute_all_metrics()` dict aggregator.

**Logging:** `get_logger(__name__)` from `xldvp_seg.utils.logging`. No bare `logging.getLogger`.

**Coordinates:** All `[x, y]` (horizontal, vertical). UID: `{slide}_{celltype}_{x}_{y}`. CZI tiles use global coords; RAM arrays are 0-indexed (subtract `x_start, y_start` when indexing `all_channel_data` directly). `loader.get_tile()` handles offsets. See `docs/COORDINATE_SYSTEM.md`.

**safe_to_uint8** (`xldvp_seg/utils/detection_utils.py`): naive uint16 conversion is `arr/256`, which produces very dim output for low-signal channels — use `_percentile_normalize_single()` for proper dynamic-range normalization instead. Float inputs >1.0 are clipped to [0,255].

---

## CLI Reference

### Flag gotchas
- `--no-normalize-features` disables flat-field (no `--flat-field-correction` flag exists)
- `--photobleaching-correction` (note spelling) — **EXPERIMENTAL**, results unreliable
- `--sequential` does NOT exist — use `--num-gpus 1`
- `--nuclear-channel` / `--membrane-channel` are islet-only
- `--tissue-channels "2,3,5"` — marker channels identifying tissue regions worth segmenting. Also controls morph/SAM2 feature extraction. NOT for Cellpose (use `--channel-spec`). Legacy aliases still accepted: `--islet-display-channels`, `--display-channels`, `--rgb-channels`.
- `--sample-fraction` is ALWAYS 1.0 — use `--html-sample-fraction` to subsample HTML only
- `--segmenter {cellpose,instanseg}` — InstanSeg requires `pip install -e ".[instanseg]"`. Only applies to `--cell-type cell`.
- `--marker-snr-channels "SMA:1,CD31:3"` — auto-classify during detection (SNR≥1.5). Requires `--all-channels`.

### Deduplication
- `--dedup-method mask_overlap` (default, pixel-exact >10% of smaller mask) or `iou_nms` (Shapely STRtree, faster, `--iou-threshold 0.2`). YAML: `dedup_method`, `iou_threshold`.
- IoU and overlap-fraction are different metrics — IoU may miss size-mismatched overlaps. Benchmark before switching default.

### Post-dedup (default ON)
- `--no-contour-processing`, `--no-background-correction`, `--bg-neighbors 30`. YAML equivalents in snake_case.
- Features are ALWAYS from the original mask. Simplification + dilation are applied at LMD export only, adaptively:
  - `--max-area-change-pct 10.0` — adaptive RDP (10% symmetric-difference)
  - `--max-dilation-area-pct 10.0` — adaptive dilation (10% area for laser buffer)
  - Set to 0 to fall back to fixed `--rdp-epsilon`, `--dilation-um`.

### Performance
```
--load-to-ram                   # [Default] Direct-to-SHM
--num-gpus 4                    # Always multi-GPU architecture
--html-sample-fraction 0.10     # Browser-friendly HTML
--max-html-samples 20000        # Hard OOM cap during per-tile accumulation
```

### Vessel strategy flags
`--candidate-mode` (relaxed thresholds for training), `--ring-only` (no lumen-first supplement), `--no-smooth-contours`, `--smooth-contours-factor 3.0`, `--multi-scale`.

3-contour system: lumen (cyan), CD31 outer (green), SMA ring (magenta). 7 vessel types: artery, arteriole, vein, venule, capillary, lymphatic, collecting_lymphatic.

### Multi-node sharding
```bash
# N nodes, each with same --resume, different --tile-shard INDEX/TOTAL
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 0/4 --resume /shared/output/dir   # node 0
    --tile-shard 1/4 --resume /shared/output/dir   # node 1
# Merge (after all shards complete):
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --resume /shared/output/dir --merge-shards
```
**CRITICAL SLURM FOOTGUN:** `--merge-shards` REQUIRES `--resume <shared-output-dir>`. Without `--resume`, argparse prints help and exits with code 0 — which means SLURM reports the job as **"COMPLETED"** even though nothing was merged. Always verify the merge log shows actual detection counts, not just argparse output.

---

## Hardware (SLURM)

- **p.hpcl93:** 19 nodes, 256 CPUs, 760G, 4× L40S — heavy GPU (`--gres=gpu:`)
- **p.hpcl8:** 55 nodes, 24 CPUs, 380G, 2× RTX 5000 — interactive dev, CPU jobs
- Time limit: 42 days both partitions
- `scripts/system_info.py` — live availability + recommendations

**Resource policy:** request ~75% of node CPUs/RAM for shared cluster etiquette (`system_info.py` applies automatically). Request all GPUs (scheduling is exclusive per device). `--exclusive` only when truly needed.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| OOM | Reduce `--num-gpus`, reduce tile size |
| CUDA boolean error | `mask = mask.astype(bool)` for SAM2 |
| SAM2 `_orig_hw` | `img_h, img_w = sam2_predictor._orig_hw[0]` (list of tuple) |
| HDF5 LZ4 error | `import hdf5plugin` before `h5py` |
| Network mount timeout | Socket timeout 60s automatic. Check with `ls /mnt/x/` |

---

## Python API (`xldvp_seg.api`)

Scanpy-style wrappers operating on `SlideAnalysis` objects.

| Module | Function | Purpose |
|--------|----------|---------|
| `tl` | `markers`, `score`, `train`, `cluster`, `spatial` | Post-detection tools (score fills missing features with 0.0) |
| `pp` | `inspect`, `detect` | CZI metadata + detection CLI generation |
| `io` | `read_proteomics`, `to_spatialdata` | Proteomics (csv or dvp-io) + scverse zarr |
| `pl` | `umap` | Visualization (delegates to clustering) |

**OmicLinker** (`xldvp_seg.analysis.omic_linker`): morphology → proteomics bridge. `link()` aggregates per-cell to well (median scalars, mean embeddings, `pool_std_*` for within-well variability). `correlate(fdr_correct=True)` (BH). `differential_features()` returns Cohen's d (capped ±10) + `n_a/n_b` + FDR p-values. `rank_proteins(sort_by="p_adjusted")`.

---

## Entry Points

**Core scripts:** `run_segmentation.py` (detection), `run_lmd_export.py` (LMD XML), `train_classifier.py` (RF), `serve_html.py` (viewer). **SLURM launcher:** `scripts/run_pipeline.sh` (YAML-driven). **CLI:** `xlseg` with 13 subcommands.

**Scripts (`scripts/`):** 42 reusable tools — see `ls scripts/` or README for full table. Core logic for 10 modules lives in `xldvp_seg/analysis/` (scripts delegate).

**Examples (`examples/`):** project-specific analyses by experiment (`bone_marrow/`, `mesothelium/`, `islet/`, `tma/`, `liver/`, `nmj/`, `vessel/`, `tissue_pattern/`, `configs/`).
