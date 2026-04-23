# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — preprocessing cache generalization + shared primitives (Apr 23 2026, part 2)

- **Tissue-filter cache** (`<cache_dir>/tissue_filter.json`). The calibration + per-tile tissue scan (~3-4 min per shard) is now cached alongside the flat-field profile. Benefits brightfield equally since tissue filtering runs for both modalities. Cache key: CZI identity (path/mtime/size) + scene + tile_size + tile_overlap + tissue_channel + modality + manual_threshold + n_all_tiles + algorithm_version. Atomic write via ``atomic_json_dump``. POSIX ``O_CREAT|O_EXCL`` advisory lock (``tissue_filter.computing``) serializes the first compute across concurrent shards so only one of N recomputes on cold start; the others poll then load. 3h stale-lock recovery matches flat-field. New module ``xldvp_seg.preprocessing.tissue_filter_cache``.
- **``--flat-field-cache-dir`` now controls the full preprocessing cache.** The flag name is retained but it's the shared slide-level cache dir for both ``flat_field_profile.npz`` and ``tissue_filter.json``. Default remains ``slide_output_dir`` (co-located with the run).
- **Scene joins the cache key for both caches.** Multi-scene CZIs can have coincidentally-identical ``slide_shape``/``n_all_tiles`` across scenes (e.g. multi-well plates where each well is a scene); without ``scene`` in the key a cache from scene 0 could false-hit for scene 1. Pre-scene caches (written before the key existed) are treated as scene=0 for back-compat — in-flight jobs with already-populated caches hit on merge/resume without redundant recompute. Validator log lines now report the raw stored value (or ``(missing)``) instead of the back-compat default so debugging is unambiguous.
- **Shared preprocessing-cache primitives** in ``xldvp_seg.utils.cache_utils``: ``czi_identity()`` (the CZI path/mtime/size triplet), ``get_scene()`` (scene resolver with None-guard for hand-built namespaces), ``meta_mismatch()`` (key-by-key freshness check with raw-value reporting and scene back-compat), and the ``try_acquire_compute_lock`` / ``release_compute_lock`` / ``wait_for_compute`` advisory-lock lifecycle. Both the flat-field cache and the tissue cache use these helpers — ~60 lines of duplicated logic collapsed to imports.
- **Internal kwarg rename** for accuracy: ``apply_slide_preprocessing(slide_output_dir=...)`` → ``apply_slide_preprocessing(cache_dir=...)``. Same for ``_apply_flat_field_correction``. The value is the shared cache dir, not always the slide's output dir. Public CLI flag unchanged.
- **Tissue cache robustness**: ``load()`` now catches ``TypeError``/``ValueError`` from ``float(threshold)`` / ``list(tiles)`` coercion so a hand-edited JSON with valid structure but wrong value types falls through to recompute instead of crashing the pipeline. Flat-field ``load`` already caught ``zipfile.BadZipFile``/``EOFError`` for corrupt npz files.
- **Tests**: ``test_cache_utils.py`` (+26 tests) covering czi_identity, get_scene matrix, meta_mismatch with scene back-compat asymmetry, lock lifecycle, and a multiprocess race (spawn 2 children, assert exactly 1 acquires the lock). ``test_tissue_filter_cache.py`` (+6 tests): end-to-end compute→save→hit cycle, ``--flat-field-cache-dir`` routing for tissue cache, pre-scene back-compat hit + scene=1 miss, wrong-value-type payload recovery. ``test_flat_field_cache.py`` (+2 tests): same pre-scene back-compat coverage. Full suite: 1337 passing (+21 this round).

### Added — flat-field cache + Cellpose BF16 version gate (Apr 23 2026)

- **Flat-field illumination profile cache** (`<slide_output_dir>/flat_field_profile.npz`). The ~1-2h slide-wide scan that ran once per fluorescence detection job is now serialized to a metadata-keyed `.npz` and reused across `--tile-shard` workers and `--resume` runs. Cache key covers CZI `(path, mtime, size)` + channel list + slide shape + `photobleaching_correction` + `ALGORITHM_VERSION`; any mismatch recomputes. Write is atomic (`.partial.npz` → `os.replace`) and `np.load(..., allow_pickle=False)` blocks RCE via tampered caches. Corrupt / truncated caches are caught (`ValueError`, `OSError`, `KeyError`, `zipfile.BadZipFile`, `EOFError`) and treated as misses. POSIX `O_CREAT|O_EXCL` advisory lock (`flat_field_profile.computing`) serializes the first compute across concurrent shards; waiters poll every 30s and retry-acquire after a stale lock is removed so exactly one node recomputes. Lock release is inside a `try/finally` so SLURM kills during compute never leak the lock. `IlluminationProfile.save()` / `.load()` added to the public module. `apply_slide_preprocessing(slide_output_dir=...)` kwarg + `setup_shared_memory` now resolves the output dir before preprocessing runs instead of after. Cuts 4-shard n45 re-detection wall time from ~4h (preprocessing repeated per shard) to ~2.5h (one compute + three cache loads). New `tests/test_flat_field_cache.py` (39 tests).
- **Cellpose BFloat16 version gate** (`xldvp_seg.utils.device.cellpose_supports_bfloat16`). Cellpose 4.x defaults `use_bfloat16=True`; on torch <2.3 the SAM backbone's relative-position encoding hits `upsample_linear1d` which had no BFloat16 kernel until 2.3. Symptom was catastrophic and silent: every tile raised `"upsample_linear1d_out_frame" not implemented for 'BFloat16'`, the pipeline saved 0 detections from 3954 tiles, and exited with code 0 so SLURM reported COMPLETED. Helper parses `torch.__version__` and returns `False` on <2.3 or unparseable strings (conservative). All 5 `CellposeModel` instantiation sites pass `use_bfloat16=cellpose_supports_bfloat16()`: `detection/cell_detector.py`, `models/manager.py`, `processing/multigpu_worker.py`, `pipeline/multigpu_phase4.py`, `pipeline/samples.py`. `cell_detector.py` logs the fp32 fallback at startup so the ~20% SAM slowdown isn't a silent surprise. Auto-re-enables once env is bumped to torch ≥2.3.
- **Cell detection max-area default raised 200 → 2000 µm²** (`xldvp_seg.detection.strategies.cell.CellStrategy.max_area_um` + `xldvp_seg.pipeline.cli` `--max-cell-area`). The 200 µm² default silently dropped polyploid hepatocytes (tetraploid ~400, octoploid ~800) and multinucleated giant cells, which surfaced as truncated manifold-sampling groups on the n45 whole-mouse section. New default covers the biological range; `pipeline/detection_setup.py` now logs the effective min/max area at startup with biology references so silent caps cannot recur.

### Added — RGB brightfield + tile overlap + post-hoc mask refinement (Apr 21–22 2026)

- **RGB brightfield CZI support** (H&E-style, single packed-RGB channel). `xldvp_seg.pipeline.shm_setup.setup_shared_memory` probes channels via `loader.is_channel_rgb()`; when any channel is RGB it allocates an `(H, W, 3)` uint8 SHM buffer (instead of the uint16 `(H, W, n_ch)` default) and passes `modality='brightfield'` to `filter_tissue_tiles` so the H&E Otsu path is used for tissue detection. `apply_slide_preprocessing` now early-returns on RGB (photobleach/flat-field/Reinhard are fluorescence-specific and crash on uint8 RGB). Errors out on mixed RGB+grayscale or multi-channel RGB — single RGB channel only. Existing CZI loader (`is_channel_rgb`, `load_to_shared_memory`), MK strategy, and `multigpu_worker.py` already handled RGB tiles end-to-end; the gap was only in SHM allocation.
- **`tile_size` + `tile_overlap` YAML keys** (`scripts/run_pipeline.sh`). Default `tile_overlap=0.10` (66 µm at 0.22 µm/px) is smaller than MK diameter (100–150 µm), so cells at tile edges get bisected and dedup keeps both partial detections. Setting `tile_overlap: 0.25` (~165 µm) eliminates clipping — bisected cells are now fully contained in at least one tile's overlap zone. Observed: **16-slide MK re-run recovered 151,917 detections at 25% overlap vs 137,782 at 10% (~10% more)**. Cross-tile merging proper is still vessel-only (`detect_vessel_structures.py`); porting to MK/cell is future work.
- **`scripts/refine_detection_masks.py`** — generic post-hoc intensity-based mask refinement. Loads per-tile `{cell_type}_masks.h5` + CZI, applies `xldvp_seg.utils.mask_cleanup.refine_mask_intensity` per detection (morphological opening + iterative brightness-based boundary peeling using 90th-percentile of mask interior as threshold, largest-CC, size guard reverts at <50%), recomputes `contour_px`/`contour_um`/`area`/`area_um2`/`solidity`/`circularity` from the refined mask, writes `{cell_type}_detections_refined.json`. Generic — works on any `--cell-type`/`--run-dir`/`--czi-path`, parallelizable with `--workers`.
- **Multi-slide HTML regen**: `scripts/regenerate_html.py --czi-dir <dir>` groups detections by slide, processes each CZI in its own process (capped at `XLDVP_MAX_SLIDE_WORKERS`, default 8, to avoid OOM), merges samples into one paginated HTML. Supports alternate detection JSON formats (`center_x/center_y` → `global_center` with stage offset, `contour_yx` [Y,X] → `contour_px` [X,Y] with stage offset, `mk_score` → `rf_prediction`, top-level `area_um2` → `features.area_um2`) without mutating caller dicts.
- **RGB channel handling in `create_sample_from_contours`**: when channel data is already 3D (brightfield RGB packed as (H, W, 3) uint8), use it directly instead of `np.stack([...], axis=-1)` which would produce a 4D array that `percentile_normalize` can't consume.
- **Scaled contour thickness in HTML crops**: contour line width now scales with rendered crop size (`max(base, rendered_size // 60)`) so large cells get proportionally thicker outlines. Fixes the "thin contour on big downscaled cells" visual illusion. Baseline default bumped `2 → 6`. Applies to both `regenerate_html.py` and `xldvp_seg/pipeline/samples.py`.

### Changed

- **Module layering cleanup**: `xldvp_seg.pipeline.background` → `xldvp_seg.analysis.background` (background subtraction and per-cell bg-corrected feature recomputation are analysis, not pipeline-stage orchestration). All 6 importers updated; external callers must rename their imports.
- **`visualization.encoding.safe_json` is now strict-JSON**: `NaN`/`Infinity` are replaced with `null` before serialization (`allow_nan=False`). stdlib default emits non-standard NaN/Infinity tokens that break `JSON.parse` in browsers.
- **Python 3.11 required** (dropped 3.10). Several scverse deps (anndata ≥0.12, spatialdata ≥0.7, squidpy ≥1.8) require 3.11. CI matrix pinned to 3.11 only. `pyproject.toml`, `install.sh`, README, docs all updated.
- **Dependency pins hardened** in `pyproject.toml` to prevent `ResolutionTooDeep` on fresh installs: `numpy>=1.26`, `pandas<3`, `anndata>=0.11`, `spatialdata>=0.7`. `requirements-lock.txt` always used by `install.sh` — skips pip's resolver entirely. `install.sh` hardened against silent failures (HTTP errors, missing lock pins, `pip install torch` no-ops, SAM2 `ResolutionTooDeep`).
- **Cross-platform install**: `install.sh` explicit paths for Linux + Mac; Windows has a 3-step manual recipe in README (SAM2 checkpoint download included). Apple Silicon MPS autodetected for Cellpose (3-10× faster than CPU).

### Added

- **Reviewer-audit safety fixes** (Apr 20 2026): degenerate-input guards across `region_clustering` + n45 viewers (`find_optimal_k_elbow` returns `{1: 0.0}` sentinel dicts on `n<2` instead of empty dicts that crashed downstream plotting; `cluster_leiden` early-returns `np.zeros(n)` when `n<2`; `region_multinuc_plot.py` skips GMM when fewer than 2 regions; `global_cluster_spatial_viewer.py` + `region_pca_viewer.py` + `combined_region_viewer.py` use `math.isfinite()` before `int()` coercion on `n_nuclei`). `install.sh` hardened: SAM2 checkpoint size check runs unconditionally (catches half-copied files), `stat` failure is graceful, activate-hook wiring skips the conda `base` env and refuses `CONDA_PREFIX` paths with shell metacharacters. New `tests/test_cluster_spatial_stats.py` (8 tests) + 5 new degenerate-input tests in `tests/test_region_clustering.py`. Total: **1169 passing tests across 56 files**.
- **Global cluster + spatial divergence viewer**: `scripts/global_cluster_spatial_viewer.py` — inverse of region-first analysis. Clusters ALL nucleated cells globally (Leiden on full PCA-kNN + k-means elbow + HDBSCAN on PCA + HDBSCAN on full UMAP). Spatial divergence metrics per cluster: entropy, `n_major_regions`, `top3_frac`, `k_90`, `focal_multimodal` (explicitly rewards 2-5 distinct regions — "same feature profile, different anatomy"). Interactive HTML viewer with 4-method coloring toggle, min-cells-per-region threshold, click-to-select on UMAP or spatial map. 520K-cell run ≈ 90-150 min on 64 CPUs.
- **Per-region PCA/UMAP + clustering**: `xldvp_seg.analysis.region_clustering` exposes `hopkins_statistic`, `find_optimal_k_elbow`, `cluster_leiden`, `cluster_hdbscan`, and `process_region`. Driven by `scripts/region_pca_viewer.py` (single-HTML viewer with 4-way color toggle: kmeans/leiden/hdbscan-PCA/hdbscan-UMAP) and `scripts/combined_region_viewer.py` (2-pane viewer: spatial region map clickable → per-region UMAP). Tested at `tests/test_region_clustering.py` (12 tests).
- **Per-region multinucleation analysis**: `scripts/region_multinuc_plot.py` generates a 3-panel PNG (histogram+KDE, ranked %multi, stacked n=1/2/3/4+ composition) plus Tukey fences + GMM(k=2 via BIC) outlier detection with a ranked CSV.
- **Consolidated transcript tool**: `scripts/export_transcript.py` merges prior two scripts (`export_session_transcript.py`, `transcript_add_png_export.py`) into a single tool with `--mode curate` (keep/skip review UI) and `--mode present` (per-card + combined PNG export). Works on any Claude Code session JSONL.
- **Environment classification** in `scripts/system_info.py`: `environment` field now returns `slurm` / `workstation` / `laptop` (heuristic: ≥16 cores OR ≥64 GB RAM OR has GPU = workstation). `/analyze` uses this to tailor guidance: SLURM on a cluster, direct `xlseg detect` on a workstation, expectations set upfront on a laptop. The 75% resource cap already applied everywhere; this just makes the UX path obvious.
- **Block-face registration workflow**: register gross tissue photos (phone, dissection scope) to fluorescence CZI via VALIS 2-pass nonrigid registration. Recursive SAM2 auto-segments anatomical regions with area-scaled point density. Per-region UMAP and organ-specific LMD export. See `docs/BLOCKFACE_REGISTRATION.md`.
- `_js_esc()` function in `html_utils.py` for safe JavaScript string escaping (separate from `_esc()` for HTML context). All JS `const` assignments across `html_generator.py`, `html_batch_export.py`, `html_export.py`, and `html_scripts.py` now use `_js_esc()`.
- `classify_gmm()` now performs BIC-based model selection (1 vs 2 components) — returns all-negative for unimodal data instead of forcing a 2-component fit. Configurable `posterior_threshold` parameter.
- `classify_otsu()` and `classify_otsu_half()` gain `include_zeros` parameter for background-corrected data where zeros represent genuine signal.
- `correlate(fdr_correct=True)` applies Benjamini-Hochberg FDR correction to p-value matrix (default on).
- `rank_proteins(sort_by="p_adjusted")` parameter for significance-ordered ranking.
- `link()` adds `pool_std_{feature}` columns for within-well variability assessment.
- `ClusteringConfig.pca_variance` field makes PCA variance retention configurable (default 0.95).
- `nuc_channels` parameter in `count_nuclei_in_cells()` and `count_nuclei_for_tile()` (default `[0, 0]`).
- `adata.uns["pipeline"]` provenance in `build_anndata()` (package, version, cell_type).
- 82 new tests: `test_visualization.py` (38), `test_lmd_export.py` (33), `test_api.py` integration tests (2), plus 9 fixes to existing tests. Total: 1047 tests across 46 files.
- `pytest.importorskip("dvpio")` guard on dvp-io tests for graceful CI degradation.
- `xldvp_seg/visualization/` package -- reusable HTML visualization components extracted from the monolithic spatial viewer. 8 Python modules (`fluorescence.py`, `colors.py`, `encoding.py`, `data_loading.py`, `graph_patterns.py`, `html_builder.py`, `js_loader.py`, `__init__.py`) + 17 JS component files in `js/` subdirectory.
- `scripts/generate_contour_viewer.py` -- generates self-contained HTML for contour overlays on CZI fluorescence. Supports grouping by configurable field (vessel_type, scale, etc.), R/G/B channel toggle, pan/zoom with RAF batching, viewport culling for 50K+ contours, and click-to-inspect metadata panel.
- `--marker-snr-channels` flag for built-in SNR marker classification during detection (zero extra cost).
- `--tissue-channels` flag (replaces `--islet-display-channels`) -- generic flag for selecting marker channels that identify tissue regions worth segmenting. Required for islet; optional for other cell types.
- `xlseg qc` subcommand for quick quality checks without HTML viewer (now 13 subcommands total).
- Lazy torch imports for faster package loading (`__getattr__` pattern in `utils/` and `detection/` `__init__.py`).
- `xldvp_seg/exceptions.py` -- custom exception hierarchy (`XldvpSegError`, `ConfigError`, `DataLoadError`, `DetectionError`, `ClassificationError`, `ExportError`, `ChannelResolutionError`) with dual-inheritance from builtins for backward compatibility.
- `xldvp_seg/classification/base.py` -- `BaseVesselClassifier` abstract base class (shared save/load/feature_importance for 4 vessel classifiers).
- `xldvp_seg/lmd/export.py` -- 18 pure-logic LMD export functions promoted from `run_lmd_export.py` (script reduced from 1,904 to 1,086 lines).
- `ClusteringConfig` dataclass in `cluster_features.py` replaces `SimpleNamespace` (28 typed fields, IDE autocomplete).
- `pl.umap()` API implemented (delegates to `run_clustering(methods='umap')`).
- **dvp-io integration**: `OmicLinker.load_proteomics_report()` parses search engine reports (AlphaDIA, DIANN, MaxQuant, Spectronaut, etc.) directly into the morphology→proteomics linking workflow. dvp-io is now a core dependency.
- `OmicLinker.available_engines()` lists supported proteomics search engines.
- `xseg.io.read_proteomics()` public API function for CSV or dvp-io report loading.

### Changed

**Nuclear counting:**
- Nuclear-to-cell assignment uses mask overlap instead of centroid lookup. Each nucleus is assigned to the cell with the most overlapping pixels, correctly handling peripheral nuclei in multinucleated cells (megakaryocytes, osteoclasts).
- Per-nucleus `overlap_area_um2` (area within assigned cell) for correct N:C ratio; `area_um2` remains full nuclear area for ploidy. `nuclear_area_fraction` clamped ≤ 1.0.

**Feature extraction:**
- Inter-channel ratios (`ch{N}_{M}_ratio`, `channel_specificity`) now use median intensities instead of means — robust to outlier pixels (debris, membrane hotspots). Feature key names unchanged; trained classifiers should be retrained.

**Security & deserialization:**
- `torch.load()` uses `weights_only=True` in NMJ strategy and cell detector.
- `joblib.load()` in `BaseVesselClassifier.load()` validates dict type and raises `ClassificationError` on model type mismatch. `load_rf_classifier()` raises `DataLoadError` on unexpected type.
- `safe_json` escapes both `</` and `<!--` sequences. `_js_esc()` function for JS string contexts across all HTML generator files.
- `generate_cross_placement_html()` validates numeric inputs are finite before embedding in HTML.

**Marker classification:**
- `classify_gmm()` uses BIC model selection (delta ≥ 6 for 2-component preference), returns all-negative when separation < 0.5 AND minor component weight < 0.1. New `include_zeros` and `posterior_threshold` parameters.
- `classify_single_marker()` passes `include_zeros=True` to Otsu/Otsu-half/GMM when `bg_subtract` or `global_background` is active.

**Pipeline & background correction:**
- Phase 1 quick medians filter zero-valued pixels (CZI padding). `ch{N}_snr` always written (0.0 when bg is zero).
- Phase 3 cleanup in `try/finally`. Contour global coordinates use float64 (was float32).
- `_extract_centroids()` uses NaN + slide-median imputation instead of `[0, 0]` fallback.
- `correct_all_channels` scans first 100 detections for feature keys (was 10).
- Multi-node sharding: shard 0 writes `sampled_tiles.json`; others wait and read.
- OME-Zarr export re-raises `MemoryError` and `OSError`.

**Reproducibility:**
- `np.random.default_rng(seed)` migration across `shm_setup.py`, `detection_loop.py`, `stain_normalization.py`, `tissue.py`, `processing/pipeline.py`, `_sample.py`.
- `multigpu_worker.py` SAM2 cache uses `$SLURM_TMPDIR` / `tempfile.gettempdir()` instead of hardcoded `/tmp`.

**OmicLinker & statistics:**
- `correlate(fdr_correct=True)` applies BH FDR correction to p-value matrix. `rank_proteins(sort_by="p_adjusted")` for significance-ordered ranking.
- `link()` adds `pool_std_{feature}` columns (`.fillna(0.0)` for single-cell wells).
- `differential_features()` returns `n_a`, `n_b` sample sizes. Cohen's d capped at ±10 with zero-variance guard.
- `tl.score()` defaults missing features to 0.0 (scores all detections, logs warning with feature names).

**Clustering & spatial analysis:**
- `ClusteringConfig.threshold` default 0.5 → 0.0. New `pca_variance` field (default 0.95).
- `discover_channels_from_features()` scans first 10 detections instead of breaking after first.
- `pattern_detection.py` betweenness refinement uses rank-based demotion (handles tied zero values).
- `_ISLET_MARKER_DEFAULTS` emits `DeprecationWarning` with v3.0 removal timeline.

**SpatialData & export:**
- SpatialData shape polygons use micron coordinates (matching `obsm["spatial"]`).
- `adata.uns["pipeline"]` provenance in `build_anndata()`. `_discover_features()` sorts prefixes by length descending.
- `aggregation.py`: `*_count` columns moved to obs; `adata.layers["missing"]` stores NaN positions before zero-fill.
- `build_contour_js_data` uses random subsampling for spatially unbiased contour selection.

**Model & infrastructure:**
- `models/manager.py`: torch/torchvision imports moved inside methods (faster CLI startup). `ModelRegistry.reset()` for test isolation.
- `_LazyModelDict._load_model` marks key as loaded only after successful assignment.
- `vessel_features.py`: `pixel_size_um` default None with warning+fallback (was hardcoded 0.22).
- `count_nuclei_for_tile()` promotes dinov2 kwargs to explicit parameters. Centroid-only assignment documented.
- `process_contour()` uses `cv2.contourArea` instead of Shapely Polygon for initial area.

**Module restructuring:**
- Removed 3 API stubs (`pl.spatial`, `tl.nuclei`, `io.export_lmd`).
- `scripts/generate_multi_slide_spatial_viewer.py`: inline JS → `load_js()` (3,115 → 1,115 lines).
- `html_export.py` split into `html_utils.py`, `html_styles.py`, `html_scripts.py` (3,696 → 1,790 lines).
- `html_batch_export.py` extracted from `html_generator.py` (2,247 → 1,236 lines).
- Exception migration: 82 bare `RuntimeError`/`ValueError` → domain-specific exceptions.
- HTML module consolidation: 5 duplicate functions replaced with shims (-749 lines).
- `py.typed` in `pyproject.toml` package-data. Removed `numpy<2.0` pin.
- F841 ruff: 41 dead-code violations fixed. Spatial viewer JS performance optimizations.

### Fixed

- Cohen's d epsilon inflation: `pooled_std + 1e-10` replaced with proper zero-variance guard (`cohens_d = 0.0` when `pooled_var < 1e-20`).
- `rank_proteins()` crash on empty result set (no proteins with >=5 non-NA observations).
- `rank_proteins()` missing `morph_feature` validation (now raises `ConfigError`).
- Post-detection pipeline logs warning when Phase 1 or Phase 3 failure rate exceeds 5%.
- Background correction tissue-boundary limitation documented in `background.py` docstring.
- `base.py` load() docstring updated: `ValueError` -> `ClassificationError`.
- `safe_to_uint8` docstring corrected: `arr // 256` -> `arr / 256` to match code.
- Stale test count updated: 974 → 1047 → **1169** tests across **56** files.
- 20+ bug fixes including: SHM leak on resume, marker_profile dict level, non-numeric columns crash, islet hardcoded channel fallback, single-detection sampling, worker BrokenPipeError handling, SHM cleanup PID guard, tile list race condition, temp dir leak in `tl.markers()`.
- 16+ documentation findings (nesting, stale counts, field names).
- mmap chunk boundary escape bug (overlap chunks by 1 byte + try/finally).
- `sys.exit()` in library code replaced with proper exceptions.
- AnnData nuclear fields placement (obs, not X).
- Coordinate mutation safety in position extraction utilities.

## [2.0.0] - 2026-03-25

### Added

- `xlseg` unified CLI with 13 subcommands (info, detect, classify, markers, score, cluster, qc, export-lmd, serve, system, models, strategies, download-models)
- Scanpy-style Python API (`xldvp_seg.api.{pp,tl,pl,io}`)
- `SlideAnalysis` central state object for notebook workflows
- Model registry with modality tracking (fluorescence/brightfield/both)
- Strategy registry with `@register_strategy` decorator (8 strategies)
- IoU NMS deduplication (`--dedup-method iou_nms`) via Shapely STRtree
- InstanSeg alternative segmenter (`--segmenter instanseg`)
- Brightfield foundation models (UNI2, Virchow2, CONCH, Phikon-v2)
- SpatialData/scverse integration with auto-export to zarr
- OME-Zarr auto-generation from shared memory
- Multi-node sharding (`--tile-shard INDEX/TOTAL`, `--merge-shards`)
- Direct-to-SHM CZI loading (eliminates ~9 GB peak memory)
- KD-tree local background correction with per-channel caching
- Median-based SNR marker classification (default `--method snr`)
- Segmentation metrics module (IoU, Dice, Panoptic Quality, Hungarian matching)
- OmicLinker for bridging morphology features to proteomics
- Slide-level and cohort aggregation utilities
- Feature-gated spatial smoothing for UMAP/t-SNE
- Sample dataset generator for testing (500 synthetic detections)
- Channel resolution by name/wavelength (`--channel-spec`)
- `run_pipeline.sh` YAML config-driven SLURM launcher with downstream job chains
- Pre-commit hooks (ruff + black), Makefile, CI with coverage
- ROI-restricted detection module (`xldvp_seg/roi/`) with marker threshold, circular object, and polygon/mask ROI discovery
- 974 tests passing across 43 test files

### Changed

- All cell types use unified multi-GPU pipeline (no separate single-GPU path)
- Atomic JSON writes via `atomic_json_dump()` with orjson acceleration
- ThreadPool-parallelized post-dedup processing (contour, background, intensity)
- Standardized logging with `get_logger(__name__)` throughout

### Removed

- `shared/` backward-compatibility module
- Legacy `xldvp-seg`, `xldvp-seg-nmj`, `xldvp-seg-serve` entry points
- Project-specific scripts moved to `examples/`

## [1.0.0] - 2026-01-15

### Added

- Initial unified pipeline with NMJ, MK, vessel, cell, islet, mesothelium, tissue_pattern strategies
- Multi-GPU tile processing with shared memory
- Cellpose + SAM2 segmentation
- LMD export with 384-well plate assignment
- HTML annotation viewer
- RF classifier training from annotations
