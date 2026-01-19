# Entry Point Unification Design Document

**Author:** Claude Code
**Date:** 2026-01-19
**Status:** Design (Not Implemented)

---

## Executive Summary

The `xldvp_seg_repo` project has evolved organically with 5 separate entry point scripts, each optimized for different use cases but sharing significant code. This document proposes unifying them into a single `run.py` CLI with subcommands while preserving the specialized functionality of each.

---

## 1. Current State Analysis

### 1.1 Entry Point Scripts

| Script | Primary Purpose | Lines of Code | Cell Types | Batch Support |
|--------|----------------|---------------|------------|---------------|
| `run_segmentation.py` | Unified single-slide pipeline | ~3,245 | nmj, mk, cell, vessel, mesothelium | No (single slide) |
| `run_unified_FAST.py` | MK/HSPC batch processing | ~3,520 | mk, hspc only | Yes (multi-slide) |
| `run_nmj_segmentation.py` | Legacy NMJ detection | ~628 | nmj only | No |
| `run_nmj_inference.py` | NMJ classification | ~327 | nmj only | No |
| `run_lmd_export.py` | LMD export workflow | ~922 | Any (post-processing) | No |

**Total: ~8,642 lines across entry points** (excluding segmentation/ library code)

### 1.2 Detailed Script Analysis

#### `run_segmentation.py` (Unified Single-Slide Pipeline)
**Purpose:** General-purpose detection for single slides with tissue detection and HTTP server.

**Key Features:**
- Supports 5 cell types via strategy pattern (`StrategyRegistry`)
- Automatic tissue detection before sampling
- Built-in HTTP server + Cloudflare tunnel for remote viewing
- Full feature extraction (2,326 features per detection)
- Multi-channel support (`--all-channels`)

**CLI Arguments (36 total):**
```
--czi-path, --cell-type, --output-dir, --tile-size, --sample-fraction,
--channel, --all-channels, --load-to-ram, --no-ram, --show-metadata,
--intensity-percentile, --min-area, --min-skeleton-length, --max-solidity,
--nmj-classifier, --mk-min-area, --mk-max-area,
--min-vessel-diameter, --max-vessel-diameter, --min-wall-thickness,
--max-aspect-ratio, --min-circularity, --min-ring-completeness,
--cd31-channel, --classify-vessel-types,
--target-chunk-area, --min-ribbon-width, --max-ribbon-width,
--min-fragment-area, --add-fiducials, --no-fiducials,
--extract-full-features, --samples-per-page,
--serve, --serve-background, --no-serve, --port, --stop-server, --server-status
```

**Dependencies:**
- `segmentation.detection.strategies.*` (strategy pattern)
- `segmentation.detection.cell_detector.CellDetector`
- `segmentation.detection.tissue` (tissue filtering)
- `segmentation.io.html_export` (HTML generation)
- `segmentation.io.czi_loader` (CZI loading)

#### `run_unified_FAST.py` (MK/HSPC Batch Pipeline)
**Purpose:** High-performance batch processing of multiple slides with model sharing.

**Key Features:**
- Multi-slide batch support (`--czi-paths` with glob expansion)
- Models loaded ONCE, shared across all slides
- Memory validation and auto-adjustment of workers
- 4-phase pipeline: Load -> Tissue ID -> Sample -> ML Processing
- Integrated HTML export while slides are in RAM
- Parallel tile processing with shared memory
- Sequential mode for stability (`--sequential`)

**CLI Arguments (16 total):**
```
--czi-path, --czi-paths, --output-dir, --mk-min-area-um, --mk-max-area-um,
--tile-size, --overlap, --sample-fraction, --calibration-block-size,
--calibration-samples, --mk-classifier, --hspc-classifier,
--cellpose-channels, --num-workers, --sequential,
--html-output-dir, --samples-per-page
```

**Unique Capabilities Not in `run_segmentation.py`:**
- Multi-slide batch processing
- Shared memory tile extraction
- Worker count auto-adjustment based on RAM
- Overlap parameter for tile stitching
- Cellpose channel configuration

#### `run_nmj_segmentation.py` (Legacy NMJ Detection)
**Purpose:** Original NMJ detection using intensity + solidity filtering.

**Key Features:**
- Intensity thresholding with skeleton-based elongation filter
- HDF5 mask storage per tile
- Simple HTML annotation interface
- Deprecated in favor of `run_segmentation.py --cell-type nmj`

**CLI Arguments (13 total):**
```
--czi-path, --output-dir, --tile-size, --sample-fraction, --samples-per-page,
--intensity-percentile, --min-area, --min-skeleton-length, --max-solidity,
--channel, --load-to-ram, --no-load-to-ram
```

**Status:** Legacy script with deprecation notice. Functionality fully covered by `run_segmentation.py`.

#### `run_nmj_inference.py` (NMJ Classification)
**Purpose:** Classify pre-detected NMJ candidates using trained ResNet18 model.

**Key Features:**
- Uses output from `run_nmj_segmentation.py` as input
- ResNet18 classifier with batch inference
- Confidence thresholding
- Aggregates results across tiles

**CLI Arguments (9 total):**
```
--czi-path, --segmentation-dir, --model-path, --output-dir,
--channel, --tile-size, --confidence-threshold,
--load-to-ram, --no-load-to-ram
```

**Unique Capabilities:**
- Two-stage workflow (detection -> classification)
- Uses pre-computed segmentation results

#### `run_lmd_export.py` (LMD Export)
**Purpose:** Export detections to Leica LMD XML format for laser microdissection.

**Key Features:**
- Two-stage workflow: Generate cross HTML -> Export with crosses
- Spatial clustering for well assignment (greedy/kmeans/dbscan)
- 96-well and 384-well plate support
- Interactive HTML for placing reference crosses
- Coordinate transformation to stage coordinates

**CLI Arguments (17 total):**
```
--detections, --annotations, --crosses, --output-dir, --output-name,
--generate-cross-html, --export, --pixel-size, --image-width, --image-height,
--cluster-size, --plate-format, --clustering-method, --no-flip-y
```

**Unique Capabilities:**
- LMD XML generation (uses `py-lmd` library)
- Spatial clustering for multi-well assignment
- Reference cross placement workflow

### 1.3 Overlapping Functionality

| Functionality | run_segmentation | run_unified_FAST | run_nmj_seg | run_nmj_inf | run_lmd |
|---------------|------------------|------------------|-------------|-------------|---------|
| CZI Loading | Yes | Yes | Yes | Yes | No |
| Tile Processing | Yes | Yes | Yes | Yes | No |
| Tissue Detection | Yes | Yes | No | No | No |
| SAM2 | Yes | Yes | No | No | No |
| Cellpose | Yes | Yes | No | No | No |
| ResNet Features | Yes | Yes | No | Yes | No |
| HTML Export | Yes | Yes | Yes | No | No |
| HTTP Server | Yes | No | No | No | No |
| Batch (multi-slide) | No | Yes | No | No | No |
| LMD Export | No | No | No | No | Yes |
| Classifier Inference | Yes (NMJ) | Yes (MK/HSPC) | No | Yes | No |

### 1.4 Shared Library Infrastructure

The codebase already has significant shared infrastructure in `segmentation/`:

```
segmentation/
  detection/
    registry.py          # StrategyRegistry for cell type strategies
    cell_detector.py     # CellDetector class
    tissue.py            # Tissue detection
    strategies/
      base.py            # DetectionStrategy ABC
      nmj.py, mk.py, cell.py, vessel.py, mesothelium.py
  models/
    manager.py           # ModelManager singleton for SAM2/Cellpose/ResNet
  processing/
    memory.py            # Memory validation and worker count
    mk_hspc_utils.py     # Tile processing utilities
    batch.py             # BatchProcessor (partial implementation)
    pipeline.py          # Pipeline utilities
  io/
    czi_loader.py        # CZILoader class
    html_export.py       # HTML generation utilities
    html_generator.py    # Additional HTML utilities
  utils/
    config.py            # Configuration constants
    feature_extraction.py # Feature extraction
    logging.py           # Logging setup
    schemas.py           # JSON schema validation
  cli.py                 # Partial CLI implementation (not in use)
```

---

## 2. Proposed Unified Structure

### 2.1 Single Entry Point: `run.py`

```bash
python run.py <command> <subcommand> [options]
```

### 2.2 Command Hierarchy

```
run.py
  |
  +-- segment                    # Cell detection
  |     +-- nmj                  # NMJ detection
  |     +-- mk                   # Megakaryocyte detection
  |     +-- hspc                 # HSPC detection (alias for 'cell')
  |     +-- vessel               # Vessel morphometry
  |     +-- mesothelium          # Mesothelium for LMD
  |
  +-- batch                      # Multi-slide batch processing
  |     +-- mk-hspc              # MK+HSPC batch (current run_unified_FAST.py)
  |     +-- segment              # Generic batch for any cell type
  |
  +-- classify                   # Post-detection classification
  |     +-- nmj                  # NMJ classification from detections
  |
  +-- export                     # Export to various formats
  |     +-- lmd                  # Leica LMD XML export
  |     +-- csv                  # Coordinate CSV export
  |     +-- geojson              # GeoJSON for visualization
  |
  +-- serve                      # Viewing server management
  |     +-- start                # Start HTTP server
  |     +-- stop                 # Stop running server
  |     +-- status               # Show server status
  |
  +-- info                       # Information utilities
        +-- metadata             # Show CZI metadata
        +-- strategies           # List available strategies
```

### 2.3 Example Usage

```bash
# Single-slide NMJ detection (replaces run_segmentation.py --cell-type nmj)
python run.py segment nmj --czi-path /path/to/slide.czi --output-dir /path/to/output

# Single-slide vessel detection with CD31 validation
python run.py segment vessel --czi-path /path/to/slide.czi \
    --min-vessel-diameter 10 --cd31-channel 1

# Batch MK/HSPC processing (replaces run_unified_FAST.py)
python run.py batch mk-hspc --czi-paths /path/to/*.czi \
    --output-dir /path/to/output --sample-fraction 0.15 --sequential

# NMJ classification from pre-computed detections (replaces run_nmj_inference.py)
python run.py classify nmj --detections /path/to/nmj_detections.json \
    --model /path/to/classifier.pth --confidence 0.75

# LMD export (replaces run_lmd_export.py)
python run.py export lmd --detections /path/to/detections.json \
    --annotations /path/to/annotations.json --generate-cross-html

# Server management (replaces --serve/--stop-server flags)
python run.py serve start --html-dir /path/to/html --port 8081
python run.py serve status
python run.py serve stop --port 8081

# Show CZI metadata (replaces --show-metadata)
python run.py info metadata --czi-path /path/to/slide.czi
```

### 2.4 Shared Options (Global)

These options would be available across all subcommands:

```
--verbose, -v          # Increase logging verbosity
--quiet, -q            # Suppress non-error output
--config FILE          # Load settings from YAML config file
--dry-run              # Show what would be done without executing
```

### 2.5 Shared Options (Segmentation)

Common to `segment` and `batch` commands:

```
# Input/Output
--czi-path PATH        # Single CZI file (segment only)
--czi-paths PATHS      # Multiple CZI files (batch only)
--output-dir PATH      # Output directory

# Tile Processing
--tile-size INT        # Tile size in pixels (default: 3000)
--sample-fraction FLOAT # Fraction of tiles to process (default: 0.20)
--overlap INT          # Tile overlap for stitching (default: 512)

# Performance
--load-to-ram          # Load image to RAM first (default: True)
--num-workers INT      # Number of parallel workers (default: 4)
--sequential           # Process tiles sequentially

# Feature Extraction
--channel INT          # Primary channel index
--all-channels         # Load all channels for multi-channel analysis

# Output
--samples-per-page INT # Samples per HTML page (default: 300)
--no-html              # Skip HTML generation
--serve / --no-serve   # Start/skip HTTP server after processing
```

---

## 3. Implementation Plan

### 3.1 Phase 1: Create Unified CLI Framework (Effort: 2-3 days)

**Goal:** Create `run.py` with command parsing infrastructure that delegates to existing scripts.

**Tasks:**
1. Create `run.py` with argparse subparsers
2. Define command hierarchy (segment, batch, classify, export, serve, info)
3. Initially delegate to existing scripts via subprocess or function calls
4. Add global options (--verbose, --config, --dry-run)

**Files to Create:**
- `run.py` (main entry point)
- `segmentation/cli/commands/__init__.py`
- `segmentation/cli/commands/segment.py`
- `segmentation/cli/commands/batch.py`
- `segmentation/cli/commands/export.py`
- `segmentation/cli/commands/serve.py`

**Risk:** LOW - This is additive; existing scripts continue to work.

### 3.2 Phase 2: Unify Single-Slide Processing (Effort: 3-5 days)

**Goal:** Refactor `run_segmentation.py` into modular components callable from unified CLI.

**Tasks:**
1. Extract pipeline orchestration into `segmentation/processing/single_slide.py`
2. Move server management to `segmentation/server/manager.py`
3. Consolidate cell-type specific parameters into strategy configs
4. Create `segment` command handlers that call the refactored code

**Files to Modify:**
- `run_segmentation.py` -> Extract to `segmentation/processing/single_slide.py`
- `segmentation/cli.py` -> Replace with `segmentation/cli/commands/segment.py`

**Files to Create:**
- `segmentation/server/manager.py` (server lifecycle management)
- `segmentation/config/strategies.yaml` (default parameters per cell type)

**Risk:** MEDIUM - Core pipeline changes could introduce bugs.

### 3.3 Phase 3: Unify Batch Processing (Effort: 3-5 days)

**Goal:** Refactor `run_unified_FAST.py` into modular components.

**Tasks:**
1. Extract 4-phase pipeline into `segmentation/processing/batch_pipeline.py`
2. Generalize batch processing to support any cell type (not just MK/HSPC)
3. Consolidate memory management utilities (already in `segmentation/processing/memory.py`)
4. Create `batch` command handlers

**Files to Modify:**
- `run_unified_FAST.py` -> Extract to `segmentation/processing/batch_pipeline.py`

**Key Challenge:** The MK/HSPC pipeline has specialized logic (dual cell type detection, HSPC confidence sorting) that would need to be generalized.

**Risk:** MEDIUM-HIGH - Batch processing is complex with multiprocessing, shared memory.

### 3.4 Phase 4: Integrate Legacy Scripts (Effort: 2-3 days)

**Goal:** Integrate `run_nmj_segmentation.py`, `run_nmj_inference.py`, `run_lmd_export.py`.

**Tasks:**
1. Mark `run_nmj_segmentation.py` as deprecated, redirect to `run.py segment nmj`
2. Extract inference logic from `run_nmj_inference.py` into `segmentation/classification/nmj.py`
3. Extract LMD export into `segmentation/export/lmd.py`
4. Create `classify` and `export` command handlers

**Files to Create:**
- `segmentation/classification/__init__.py`
- `segmentation/classification/nmj.py`
- `segmentation/export/__init__.py`
- `segmentation/export/lmd.py`

**Risk:** LOW - These are relatively isolated scripts.

### 3.5 Phase 5: Deprecation and Cleanup (Effort: 1-2 days)

**Goal:** Update documentation, add deprecation warnings, create migration guide.

**Tasks:**
1. Add deprecation warnings to old entry points
2. Update `CLAUDE.md` with new usage examples
3. Update `docs/GETTING_STARTED.md`
4. Create `docs/MIGRATION_GUIDE.md` for users of old scripts

**Risk:** LOW - Documentation only.

---

## 4. Migration Path

### 4.1 For Users

| Old Command | New Command |
|-------------|-------------|
| `python run_segmentation.py --czi-path X --cell-type nmj` | `python run.py segment nmj --czi-path X` |
| `python run_segmentation.py --czi-path X --cell-type vessel --cd31-channel 1` | `python run.py segment vessel --czi-path X --cd31-channel 1` |
| `python run_segmentation.py --stop-server` | `python run.py serve stop` |
| `python run_segmentation.py --show-metadata --czi-path X` | `python run.py info metadata --czi-path X` |
| `python run_unified_FAST.py --czi-paths *.czi --output-dir Y` | `python run.py batch mk-hspc --czi-paths *.czi --output-dir Y` |
| `python run_nmj_inference.py --czi-path X --model-path M` | `python run.py classify nmj --czi-path X --model M` |
| `python run_lmd_export.py --detections D --generate-cross-html` | `python run.py export lmd --detections D --generate-cross-html` |

### 4.2 Deprecation Strategy

1. **Phase 1-3:** Old scripts continue to work, emit deprecation warning
2. **Phase 4:** Old scripts show warning at startup with migration instructions
3. **Future Release:** Old scripts print error and exit, pointing to new CLI
4. **Final Release:** Old scripts removed

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Regression in batch processing | Medium | High | Extensive testing on existing 16-slide dataset |
| Memory leaks in refactored code | Medium | High | Use `ModelManager` context manager consistently |
| Multiprocessing issues | Medium | Medium | Keep `--sequential` as fallback |
| Configuration conflicts | Low | Medium | Clear precedence: CLI > config file > defaults |
| Breaking user workflows | Medium | Medium | Maintain backwards compatibility in Phase 1-3 |

### 5.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope creep | High | Medium | Strict adherence to phase boundaries |
| Incomplete migration | Medium | Low | Old scripts remain functional |
| Documentation lag | Medium | Low | Update docs with each phase |

### 5.3 Effort Estimates

| Phase | Optimistic | Expected | Pessimistic |
|-------|------------|----------|-------------|
| Phase 1: CLI Framework | 1 day | 2-3 days | 5 days |
| Phase 2: Single-Slide | 2 days | 3-5 days | 8 days |
| Phase 3: Batch | 2 days | 3-5 days | 8 days |
| Phase 4: Legacy Scripts | 1 day | 2-3 days | 4 days |
| Phase 5: Documentation | 0.5 days | 1-2 days | 3 days |
| **Total** | **6.5 days** | **11-18 days** | **28 days** |

---

## 6. Testing Strategy

### 6.1 Unit Tests

Create tests for each extracted module:
- `tests/test_single_slide.py`
- `tests/test_batch_pipeline.py`
- `tests/test_classification.py`
- `tests/test_lmd_export.py`

### 6.2 Integration Tests

Test end-to-end workflows:
- Single slide NMJ detection
- Single slide vessel detection with CD31
- Batch MK/HSPC on 2-3 small test slides
- LMD export with clustering

### 6.3 Regression Tests

Compare outputs before/after refactoring:
- Same input -> Same detections (within numerical tolerance)
- Same input -> Same HTML output
- Same input -> Same LMD XML

---

## 7. Configuration File Format (Proposed)

For advanced users, support a YAML configuration file:

```yaml
# config.yaml
global:
  log_level: INFO
  output_dir: /home/dude/output

segment:
  tile_size: 3000
  sample_fraction: 0.20
  load_to_ram: true

  nmj:
    channel: 1
    intensity_percentile: 99
    min_area: 150
    max_solidity: 0.85

  vessel:
    min_vessel_diameter: 10
    max_vessel_diameter: 500
    cd31_channel: 1

batch:
  num_workers: 4
  sequential: false

  mk-hspc:
    mk_min_area_um: 200
    mk_max_area_um: 2000

serve:
  port: 8081
  background: true
```

Usage:
```bash
python run.py --config config.yaml segment nmj --czi-path /path/to/slide.czi
```

---

## 8. Appendix: Code Structure After Unification

```
xldvp_seg_repo/
  run.py                          # Single unified entry point (NEW)

  # Legacy entry points (deprecated but functional)
  run_segmentation.py             # -> Delegates to run.py segment
  run_unified_FAST.py             # -> Delegates to run.py batch mk-hspc
  run_nmj_segmentation.py         # -> Deprecated, use run.py segment nmj
  run_nmj_inference.py            # -> Delegates to run.py classify nmj
  run_lmd_export.py               # -> Delegates to run.py export lmd

  segmentation/
    cli/
      __init__.py
      main.py                     # CLI parsing (NEW)
      commands/
        __init__.py
        segment.py                # segment command handlers (NEW)
        batch.py                  # batch command handlers (NEW)
        classify.py               # classify command handlers (NEW)
        export.py                 # export command handlers (NEW)
        serve.py                  # serve command handlers (NEW)
        info.py                   # info command handlers (NEW)

    processing/
      single_slide.py             # Single-slide pipeline (EXTRACT from run_segmentation.py)
      batch_pipeline.py           # Batch pipeline (EXTRACT from run_unified_FAST.py)
      memory.py                   # Already exists
      mk_hspc_utils.py            # Already exists

    classification/
      __init__.py                 # NEW
      nmj.py                      # NMJ classifier (EXTRACT from run_nmj_inference.py)

    export/
      __init__.py                 # NEW
      lmd.py                      # LMD export (EXTRACT from run_lmd_export.py)
      csv.py                      # CSV coordinate export (NEW)

    server/
      __init__.py                 # NEW
      manager.py                  # HTTP server + tunnel (EXTRACT from run_segmentation.py)

    config/
      __init__.py                 # NEW
      strategies.yaml             # Default parameters per cell type (NEW)
      loader.py                   # Config file loading (NEW)

    # Existing modules (unchanged)
    detection/
    models/
    io/
    utils/

  docs/
    GETTING_STARTED.md
    LMD_EXPORT_GUIDE.md
    ENTRY_POINT_UNIFICATION.md    # This document
    MIGRATION_GUIDE.md            # NEW (Phase 5)
```

---

## 9. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-19 | Use subcommands (segment/batch/export) not flags | Clearer semantics, easier to extend |
| 2026-01-19 | Keep old scripts as deprecated wrappers | Backwards compatibility for existing workflows |
| 2026-01-19 | Phase 1 uses delegation, not immediate refactor | Reduces risk, allows incremental validation |
| 2026-01-19 | Support YAML config files | Power users need reproducible configurations |
| 2026-01-19 | Separate `batch mk-hspc` from generic `batch segment` | MK/HSPC has specialized dual-type logic |

---

## 10. Open Questions

1. **Should `run.py` be installable as a CLI command?**
   - Option A: `pip install -e .` with `entry_points` -> `segmentation segment nmj ...`
   - Option B: Keep as `python run.py ...`

2. **Should configuration files support inheritance/profiles?**
   - E.g., `base.yaml` + `production.yaml` that overrides specific values

3. **Should batch processing be generalized to all cell types?**
   - Currently only MK/HSPC; vessel/NMJ might benefit from batch mode

4. **Should we add a `train` command for classifier training?**
   - Currently training scripts (`train_nmj_classifier.py`, etc.) are separate

---

## 11. Conclusion

Unifying the 5 entry points into a single CLI will:
- Reduce code duplication (~30-40% of entry point code is shared patterns)
- Improve discoverability (`--help` shows all capabilities)
- Enable consistent behavior (logging, error handling, config)
- Simplify documentation and user training

The phased approach allows incremental validation and maintains backwards compatibility throughout the migration.

**Recommended Next Step:** Implement Phase 1 (CLI Framework) to validate the proposed structure before committing to deeper refactoring.
