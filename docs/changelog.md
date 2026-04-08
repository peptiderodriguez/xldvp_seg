# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- `classify_gmm()` gains `include_zeros` parameter, wired from `classify_single_marker` when bg correction is active (regression fix from round 3).
- SpatialData shape polygons now use micron coordinates (matching `obsm["spatial"]`) instead of pixel coordinates (fixes spatial analysis coordinate mismatch).
- `aggregation.py`: `*_count` columns moved from AnnData X to obs; `adata.layers["missing"]` stores NaN positions before zero-fill.
- `vessel_features.py`: pixel_size_um default changed from 0.22 to None with warning+fallback.
- `models/manager.py`: torch/torchvision imports moved inside methods (faster CLI startup).
- `stain_normalization.py`, `tissue.py`, `processing/pipeline.py`: np.random global state migrated to `default_rng(seed)`.
- `multigpu_worker.py`: SAM2 cache uses `$SLURM_TMPDIR` / `tempfile.gettempdir()` instead of hardcoded `/tmp`.
- `pattern_detection.py`: betweenness refinement uses rank-based demotion (handles tied zero values).
- Nuclear counting centroid-only assignment documented as known limitation.
- `generate_cross_placement_html()` validates `pixel_size_um`, `image_width_px`, `image_height_px` are finite before embedding in HTML.
- `ModelRegistry.reset()` classmethod for test isolation (clears and re-registers defaults).
- `process_contour()` uses `cv2.contourArea` for initial area instead of constructing a Shapely Polygon.
- `feature_extraction.py` tile_global_mean fallback uses `logger.debug` instead of `warnings.warn`.
- `_discover_features()` in `spatialdata_export.py` sorts `EMBEDDING_PREFIXES` by length descending (robust prefix matching).
- `count_nuclei_for_tile()` promotes `dinov2_model` and `dinov2_transform` from `**kwargs` to explicit parameters.
- `classify_single_marker()` now passes `include_zeros=True` to Otsu/Otsu-half when `bg_subtract` or `global_background` is active, ensuring background-corrected zeros are included in threshold computation.
- `classify_gmm()` BIC comparison requires delta ≥ 6 (Raftery's "strong evidence") for 2-component preference, plus hard guard: returns all-negative when separation < 0.5 AND minor component weight < 0.1.
- `tl.score()` now defaults missing feature values to 0.0 instead of skipping detections, matching `extract_feature_matrix` behavior. All detections are now scored.
- `load_rf_classifier()` raises `DataLoadError` on unexpected model type (was `logger.warning` + fallthrough).
- `differential_features()` now returns `n_a` and `n_b` sample size columns. Cohen's d capped at ±10.
- `pool_std` columns use `fillna(0.0)` for single-cell wells (std is undefined with ddof=1 for n=1).
- Phase 1 quick medians now filter zero-valued pixels (CZI padding) before computing median.
- `ch{N}_snr` is now always written (0.0 when background is zero), consistent with `correct_all_channels`.
- Phase 3 cleanup (`_bg_quick_medians`, `_postdedup_idx`) wrapped in `try/finally` for exception safety.
- Contour global coordinates use float64 (was float32) to preserve sub-pixel precision for large slides.
- `build_contour_js_data` uses random subsampling (was fixed-step) for spatially unbiased contour selection.
- `correct_all_channels` scans first 100 detections for feature keys (was 10) to handle edge-tile variability.
- `_LazyModelDict._load_model` marks key as loaded only after successful model assignment (prevents stale `_loaded` on load failure).
- `_ISLET_MARKER_DEFAULTS` deprecation warning now includes removal timeline (v3.0).
- `safe_json` escapes both `</` and `<!--` sequences consistently across `encoding.py` and `html_scripts.py`.
- `torch.load()` now uses `weights_only=True` in NMJ strategy and cell detector (security hardening).
- `joblib.load()` in `BaseVesselClassifier.load()` validates dict type and raises `ClassificationError` on model type mismatch (was `logger.warning`).
- `ClusteringConfig.threshold` default changed from 0.5 to 0.0 (matches CLI + API defaults).
- `discover_channels_from_features()` scans first 10 detections instead of breaking after the first with features.
- `_extract_centroids()` uses NaN + slide-median imputation instead of `[0, 0]` fallback for missing `global_center`.
- Multi-node tile sharding: only shard 0 writes `sampled_tiles.json`; other shards wait and read (eliminates race condition).
- OME-Zarr export re-raises `MemoryError` and `OSError` instead of silently catching all exceptions.
- Pipeline np.random calls in `shm_setup.py` and `detection_loop.py` migrated from global state to `np.random.default_rng(seed)`.
- `_sample.py` migrated from `np.random.RandomState` to `np.random.default_rng`.
- `py.typed` added to `pyproject.toml` package-data for PEP 561 compliance.
- `_ISLET_MARKER_DEFAULTS` emits `DeprecationWarning` at usage site.
- Removed 3 API stubs (`pl.spatial`, `tl.nuclei`, `io.export_lmd`) that raised `NotImplementedError`. These operations require CLI/SLURM and were never going to be simple function calls.
- `scripts/generate_multi_slide_spatial_viewer.py` refactored: inline JS replaced with `load_js()` from 17 component files (3,115 to 1,115 lines, -64%). Same external behavior.
- Extracted `html_utils.py` (image/HDF5 utilities), `html_styles.py` (CSS generators), and `html_scripts.py` (JS generators) from `html_export.py` (3,696 to 1,790 lines). Backward-compatible re-exports maintained.
- Extracted `html_batch_export.py` (MK/HSPC batch functions) from `html_generator.py` (2,247 to 1,236 lines). Backward-compatible re-exports in `html_generator.py`.
- Spatial viewer JS performance: viewport culling for cell dots, temp canvas reuse in fluorescence compositing, debounced window resize handler.
- `get_largest_connected_component` in `html_utils.py` replaced with import from canonical `mask_cleanup.py`.
- SVG channel filter block deduplicated in `html_export.py` (extracted to `_SVG_CHANNEL_FILTERS` constant).
- Exception migration: 82 bare `RuntimeError`/`ValueError` sites across 39 files replaced with domain-specific exceptions from `xldvp_seg.exceptions` (~35 genuine `ValueError` sites retained intentionally).
- HTML module consolidation: 5 MK/HSPC duplicate functions replaced with backward-compatible shims (749 lines removed from `html_export.py`).
- F841 ruff: 41 dead-code violations fixed, global suppress replaced with per-file ignore for `vessel.py` only.
- Removed `numpy<2.0` upper bound pin from `pyproject.toml` (now `numpy>=1.24`).

### Fixed

- Cohen's d epsilon inflation: `pooled_std + 1e-10` replaced with proper zero-variance guard (`cohens_d = 0.0` when `pooled_var < 1e-20`).
- `rank_proteins()` crash on empty result set (no proteins with >=5 non-NA observations).
- `rank_proteins()` missing `morph_feature` validation (now raises `ConfigError`).
- Post-detection pipeline logs warning when Phase 1 or Phase 3 failure rate exceeds 5%.
- Background correction tissue-boundary limitation documented in `background.py` docstring.
- `base.py` load() docstring updated: `ValueError` -> `ClassificationError`.
- `safe_to_uint8` docstring corrected: `arr // 256` -> `arr / 256` to match code.
- Stale test count updated: 974 -> 1047 (77 -> 46 files).
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
