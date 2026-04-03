# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-25

### Added
- `xlseg` unified CLI with 12 subcommands (info, detect, classify, markers, score, cluster, export-lmd, serve, system, models, strategies, download-models)
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
- TMA core detection example (`examples/tma/`) with grid labeling and per-core cell segmentation
- `/new-experiment` slash command for fast-track YAML config generation
- 516 automated tests across 21 test files

### Changed
- All cell types use unified multi-GPU pipeline (no separate single-GPU path)
- Atomic JSON writes via `atomic_json_dump()` with orjson acceleration
- ThreadPool-parallelized post-dedup processing (contour, background, intensity)
- Standardized logging with `get_logger(__name__)` throughout
- HTML crop caching during detection (resume from cache in <30 sec)

### Removed
- `shared/` backward-compatibility module
- Legacy `xldvp-seg`, `xldvp-seg-nmj`, `xldvp-seg-serve` entry points
- Project-specific scripts moved to `examples/` (bone_marrow, mesothelium, islet, liver, nmj, vessel, etc.)
- Legacy SLURM scripts moved to `examples/slurm/`

## [1.0.0] - 2026-01-15

### Added
- Initial unified pipeline with NMJ, MK, vessel, cell, islet, mesothelium, tissue_pattern strategies
- Multi-GPU tile processing with shared memory
- Cellpose + SAM2 segmentation
- LMD export with 384-well plate assignment
- HTML annotation viewer
- RF classifier training from annotations
