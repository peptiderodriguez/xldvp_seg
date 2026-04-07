# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `xldvp_seg/visualization/` package -- reusable HTML visualization components extracted from the monolithic spatial viewer. 8 Python modules (`fluorescence.py`, `colors.py`, `encoding.py`, `data_loading.py`, `graph_patterns.py`, `html_builder.py`, `js_loader.py`, `__init__.py`) + 13 JS component files in `js/` subdirectory.
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
- 934 tests across 40 test files (up from 781 across 32 at 2.0.0 release).

### Changed

- `scripts/generate_multi_slide_spatial_viewer.py` refactored to import from `xldvp_seg.visualization` instead of defining utilities inline (4062 to 3115 lines, ~30% smaller). Same external behavior.

### Fixed

- 16+ documentation findings (nesting, stale counts, field names).
- mmap chunk boundary escape bug (overlap chunks by 1 byte + try/finally).
- `sys.exit()` in library code replaced with proper exceptions.
- AnnData nuclear fields placement (obs, not X).
- Coordinate mutation safety in position extraction utilities.
- Various bug fixes: SHM leak on resume, marker_profile dict level, non-numeric columns crash.

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
- 781 automated tests across 32 test files

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
