# xldvp_seg - Session History

This file contains detailed session notes, bug fixes, and development history.
For project overview and usage, see [CLAUDE.md](../CLAUDE.md).

---

## Current State (as of Feb 4, 2026)

### Full Slide Run In Progress

Running 100% segmentation (SAMPLE_FRACTION=1.0) followed by RF classifier filtering at **0.80 probability threshold**.

**Expected output:** ~870 vessels (extrapolated from 10% sample)

### BEST RESULTS TO DATE - CD31 Endothelial Lining Filter

**Key breakthrough**: Requiring CD31+ endothelial lining at the lumen edge filters out tissue tears while keeping real vessels.

**10% Sample Run Results (Jan 31):**
| Scale | Vessels |
|-------|---------|
| 1/32  | 12      |
| 1/16  | 28      |
| 1/8   | 177     |
| 1/4   | 376     |
| 1/2   | 1 (776 filtered as uncorroborated) |
| **Final (merged)** | **594** |

**What made this work:**
1. **CD31 edge coverage filter** - Checks for endothelial lining at lumen boundary (≥40% coverage required)
2. **1/2 scale corroboration** - Finest scale detections only kept if overlapping with coarser scales
3. **Cross-scale merging** - Keeps finest segmentation with ≥90% area coverage

**Output:** `/home/dude/vessel_output/sam2_multiscale/`

### RF Classifier Training (Feb 1)

Trained Random Forest on 330 annotations (95 positive, 235 negative):

| Metric | Value |
|--------|-------|
| 5-Fold CV Accuracy | **89.1%** |
| Precision (vessel) | 87% |
| Recall (vessel) | 73% |
| F1 (vessel) | 79% |

**Top features by importance:**
1. `nuclear_ratio` (17.8%) - vessels have nuclei in walls
2. `sma_wall` (16.9%) - SMA intensity in vessel wall
3. `sma_ratio` (11.2%) - overall SMA enrichment
4. `cd31_ratio` (7.7%) - CD31 enrichment
5. `cd31_edge_coverage` (6.4%) - endothelial lining filter

**Model saved:** `/home/dude/vessel_output/sam2_multiscale/vessel_rf_classifier.joblib`

### Current Pipeline Features
- **CD31 endothelial lining filter**: Requires 40% of lumen perimeter to have CD31+ lining (filters tissue tears)
- **Side-by-side HTML crops**: Shows raw image + contoured image for each vessel
- **Cross-scale merging**: Keeps finest segmentation with 90% coverage threshold
- **1/2 scale detection**: Finest scale, requires corroboration from coarser scales
- **Union-Find clustering**: Groups all overlapping vessels before selecting best per cluster
- **RF classifier**: Post-hoc filtering with 0.80 probability threshold (89.1% CV accuracy)

---

## Feb 4, 2026 - Slide-Wide Photobleaching Correction (DISABLED)

**Problem**: Horizontal/vertical banding artifacts visible across tile boundaries. Per-tile correction normalized each tile independently, so banding that spanned tiles remained visible.

**Attempted Solution**: Apply photobleaching correction to the full mosaic (at 1/2 scale, ~45GB in RAM) *before* extracting tiles.

**Implementation:**
- New method `DownsampledChannelCache.apply_photobleaching_correction()`
- Uses `normalize_rows_columns()` to fix banding + `morphological_background_subtraction()` (201px kernel) for gradients

**ISSUE: TOO SLOW** - Morphological background subtraction with 201px kernel on 117k x 52k image took 16+ hours and never completed. The operation is O(n²) with kernel size.

**Resolution**: Disabled slide-wide photobleaching correction. The successful 10% run that produced good results did NOT use this correction anyway. Code remains in place but commented out.

**Location:** `scripts/sam2_multiscale_vessels.py` lines ~540-584 (method exists but call is commented out at line ~2150)

**Future options if banding becomes an issue:**
1. Use smaller kernel (51px instead of 201px)
2. Only apply row/column normalization (skip morphological step)
3. Process in strips/chunks instead of full mosaic
4. Apply correction at coarser scale (1/8 or 1/16) then upsample

---

## Feb 1, 2026 - RF Classifier Training

Trained vessel vs non-vessel classifier on manual annotations.

**Training data:** 330 annotations (95 positive vessels, 235 negative/tears)

**Features used (15 total):**
- Intensity ratios: `cd31_ratio`, `nuclear_ratio`, `pm_ratio`, `sma_ratio`
- CD31 edge coverage: `cd31_edge_coverage`
- SMA metrics: `sma_inside`, `sma_wall`
- Size metrics: `inner_area_px`, `outer_area_px`, `wall_area_px`, `area`
- Diameter metrics: `inner_diameter_um`, `outer_diameter_um`, `wall_thickness_um`
- Scale: `scale_factor`

**Results:** 89.1% accuracy, 87% precision, 73% recall on vessels

---

## Jan 30-31, 2026 - CD31 Endothelial Lining Filter

**Problem**: Detection was capturing tissue tears (dark gaps) in addition to real vessels. Previous runs had hundreds of false positives.

**Solution**: Check for CD31+ endothelial lining at the lumen edge. Real vessels have endothelium lining the lumen; tears don't.

**New function `compute_cd31_edge_coverage()`:**
- Creates thin ring (3px) immediately adjacent to lumen boundary
- Samples CD31 intensity at ~36 points around perimeter
- Counts fraction with CD31 > 1.5x background
- Requires ≥40% coverage (configurable via `CD31_EDGE_COVERAGE_THRESHOLD`)

**New constant:**
```python
CD31_EDGE_COVERAGE_THRESHOLD = 0.40  # 40% of perimeter must have CD31+ lining
```

**Behavior:**
| Lumen Type | CD31 Edge Coverage | Result |
|------------|-------------------|--------|
| Real vessel | 40-100% | Kept |
| Tissue tear | 0-30% | Filtered |
| Partial section | 30-40% | Borderline |

**Impact**: Thousands of tissue tears filtered out, leaving ~594 high-quality vessel detections.

---

## Jan 30, 2026 - Side-by-Side HTML Crops

**Feature**: HTML viewer now shows raw image alongside contoured image for each vessel.

**Changes:**

1. **`save_vessel_crop()`** now saves two files:
   - `{uid}_raw.jpg` - Raw image without contours
   - `{uid}.jpg` - Image with green (outer) and cyan (inner) contours

2. **`generate_html()`** passes both images to HTML template

3. **HTML template** displays side-by-side with "Raw" and "Contours" labels

**CSS classes added:**
- `.card-img-sidebyside` - Flex container for side-by-side layout
- `.img-half` - Half-width container for each image
- `.img-label` - Overlay label ("Raw" / "Contours")

---

## Jan 30, 2026 - Cross-Scale Vessel Merging: Keep Finest Segmentation

**Problem**: Previous merge strategy kept coarsest-scale detections, losing fine vessel wall detail.

**Solution**: Reverse merge logic to prefer finest segmentation that captures the full vessel.

**Scale Configuration:**
| Scale | Scale Factor | Tile Size | Full-Res Coverage |
|-------|--------------|-----------|-------------------|
| 1/64  | 64           | 1000 px   | 64,000 px         |
| 1/32  | 32           | 1200 px   | 38,400 px         |
| 1/16  | 16           | 1400 px   | 22,400 px         |
| 1/8   | 8            | 1700 px   | 13,600 px         |
| 1/4   | 4            | 2000 px   | 8,000 px          |
| 1/2   | 2            | 2500 px   | 5,000 px          |

**Merge Behavior:**
| Scenario | Result |
|----------|--------|
| Fine-scale covers ≥90% of coarse | Keep fine (better detail) |
| Fine-scale incomplete (<90%) | Fallback to coarser |
| 1/2 scale with coarser overlap | Keep if coverage ≥90% |
| 1/2 scale alone (no overlap) | Filtered out |

---

## Previous State (Jan 28, 2026)
- **RAM-first pipeline**: Load ALL slides to RAM in Phase 1 for fast tissue detection
- **RGB CZI support**: Auto-detects RGB vs grayscale CZI files, allocates correct buffer shape
- **Mask cleanup for LMD export**: Optional hole filling and small object removal (`--cleanup-masks`)
- **HSPC nuclear-only mode**: H&E deconvolution to detect HSPCs on hematoxylin channel (`--hspc-nuclear-only`)
- **No normalization for H&E tissue detection**: Raw pixel values for variance-based tissue detection
- **Percentile normalization for segmentation**: 5th-95th percentile norm before SAM2/Cellpose
- **Multi-GPU support**: 4 GPUs process tiles in parallel via shared memory
- **Note**: HTML export skipped in multi-GPU shared memory mode - run `regenerate_html_fast.py` after

---

## Jan 27-28, 2026 - HSPC Nuclear-Only Mode

**Problem**: Cellpose HSPC detection on H&E images picks up many non-HSPC cells (cytoplasmic structures, RBCs, etc.) because it processes full RGB.

**Solution**: H&E color deconvolution to extract hematoxylin (nuclear) channel only for HSPC detection.

**New Function: `extract_hematoxylin_channel()`** (run_unified_FAST.py:96-127)
- Uses `skimage.color.rgb2hed` for H&E color deconvolution
- Extracts hematoxylin channel (index 0) which highlights nuclei
- Normalizes to 0-255 and returns as uint8 grayscale
- Cellpose then detects on this nuclear-only channel

**New CLI Flag:**
```bash
python run_unified_FAST.py --czi-paths /path/*.czi --output-dir /out \
    --cleanup-masks --hspc-nuclear-only
```

**Modified Files:**
| File | Changes |
|------|---------|
| `run_unified_FAST.py` | Added `extract_hematoxylin_channel()`, `--hspc-nuclear-only` arg, updated `process_tile()` |
| `segmentation/processing/multigpu_shm.py` | Pass `hspc_nuclear_only` config to workers |
| `slurm/test_nuclear_only.sbatch` | L40S sbatch script for nuclear-only testing |
| `slurm/test_nuclear_only_rtx5000.sbatch` | RTX 5000 sbatch script (smaller tile size) |

**Test Runs:**

| Date | Partition | Tile Size | Result |
|------|-----------|-----------|--------|
| Jan 27 | RTX 5000 (p.hpcl8) | 3000 | CUDA OOM (16GB VRAM insufficient) |
| Jan 27 | RTX 5000 (p.hpcl8) | 2000 | RAM OOM (380GB insufficient for 4 slides + shared memory) |
| Jan 28 | L40S (p.hpcl93) | 3000 | Queued - estimated ~2 days wait (fair-share penalty) |

**Earlier Cleanup-Masks Test (1% sampling):**
- Jobs: 2046450, 2046451, 2046452
- Results: 1,982 MKs + 5,124 HSPCs across 16 slides
- HTML: `/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/test_cleanup_output/combined/html/`

**Pending Nuclear-Only Test (10% sampling):**
- Jobs: 2049316-2049319 on L40S partition
- Input: 16 bone marrow slides (4 per node)
- Expected: ~10x more detections than 1% run

---

## Jan 27, 2026 - Mask Cleanup for LMD Export

**Problem**: SAM2-generated masks have internal holes and noise artifacts that cause issues with laser microdissection (LMD) cutting paths.

**Solution**: New mask cleanup module with configurable parameters

**New Module: `segmentation/utils/mask_cleanup.py`**
- `cleanup_mask()` - Main function for mask post-processing
- `fill_holes()` - Fill internal holes below size threshold
- `remove_small_objects()` - Remove disconnected noise regions
- `get_largest_component()` - Keep only largest connected component

**New CLI Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--cleanup-masks` | False | Enable mask cleanup for LMD export |
| `--no-fill-holes` | False | Disable hole filling (keep internal holes) |
| `--max-hole-fraction` | 0.1 | Max hole size as fraction of mask area |

**Modified Files:**
| File | Changes |
|------|---------|
| `run_unified_FAST.py` | Added `--cleanup-masks`, `--no-fill-holes`, `--max-hole-fraction` args; integrated cleanup into mask generation |
| `segmentation/processing/multigpu_shm.py` | Added mask cleanup to shared memory worker functions |
| `slurm/test_cleanup_masks.sbatch` | Updated to include Phase 5 HTML generation |

**Usage:**
```bash
# Enable mask cleanup with default settings
python run_unified_FAST.py --czi-paths /path/*.czi --output-dir /out --cleanup-masks

# Keep holes, only remove small objects
python run_unified_FAST.py --czi-paths /path/*.czi --output-dir /out --cleanup-masks --no-fill-holes

# Custom hole threshold (5% of mask area)
python run_unified_FAST.py --czi-paths /path/*.czi --output-dir /out --cleanup-masks --max-hole-fraction 0.05
```

**Note on Multi-GPU Mode:**
HTML export is skipped when using shared memory mode (`--multi-gpu`). After processing completes, run:
```bash
python regenerate_html_fast.py --input-dir /path/to/output --output-dir /path/to/output
```

---

## Jan 26, 2026 - RAM-First Pipeline + RGB CZI Support

**Problem**: Previous streaming mode was slow (~42 min for tissue detection over network)

**Solution**: RAM-first architecture
- Phase 1: Load ALL slides to RAM (~80 GB per 4-slide batch)
- Phase 2: Tissue detection from RAM (fast - ~2 min vs 42 min)
- Phase 3: Sample 10% of tissue tiles
- Phase 4: Copy RAM → shared memory (fast memcpy), then GPU processing

**RGB CZI Support**
- Added `is_channel_rgb()` method to CZILoader - probes 100x100 region to detect RGB
- Phase 1 now detects RGB and allocates correct buffer shape `(H, W, 3)` with `uint8` dtype
- Phase 4 uses `add_slide()` to copy numpy array directly to shared memory

**Bug Fixes**
| Bug | Fix |
|-----|-----|
| `CZI channel 0 contains RGB data but shm_buffer is 2D` | Detect RGB before buffer allocation |
| `KeyError: 'global_id'` in feature JSON export | Use `.get()` for optional fields |

**Production Run Results (16 H&E bone marrow slides)**
| Batch | Slides | MKs | HSPCs | Runtime |
|-------|--------|-----|-------|---------|
| FGC1-4 | 4 | 4,591 | 11,176 | 1:02:03 |
| FHU1-4 | 4 | 6,480 | 11,857 | 1:09:10 |
| MGC1-4 | 4 | 5,491 | 11,755 | ~1:16 |
| MHU1-4 | 4 | 4,227 | 16,157 | ~1:36 |
| **TOTAL** | **16** | **20,789** | **50,945** | |

**Commits**
- `c8121e3` - RAM-first pipeline + RGB CZI support
- `41e2ce0` - Add SAM2 multi-scale vessel detection (separate feature)

---

## Previous State (Jan 24, 2026)
- **RAM + Shared Memory mode**: Best of both worlds - fast Phase 2 (RAM) + zero-copy Phase 4 (shared memory)
- **No normalization for H&E**: Disabled percentile normalization for better HSPC detection
- **Crop + Mask separation**: Saves raw crops and masks separately, outline drawn at HTML generation
- **Bug fixes (Jan 24)**: `slide_name` undefined fix, `shared_calibrate_tissue_threshold` added, OpenBLAS thread limit fix
- **Multi-GPU support**: 4 GPUs process tiles in parallel via shared memory
- **Installable package**: `./install.sh` handles PyTorch+CUDA, SAM2, all dependencies
- **Slurm support**: Ready-to-use sbatch scripts in `slurm/` directory
- **Code refactoring**: Phase 1-7 complete (model manager, memory utils, tile workers, strategy registry)

---

## Jan 24, 2026 - RAM + Shared Memory Pipeline & H&E Fixes

**Optimized Memory Architecture**
Previous streaming mode was slow (reads tiles from network CZI files). New hybrid approach:
- Phase 1: Load ALL slides into RAM (~350GB for 16 slides)
- Phase 2: Tissue detection from RAM (fast - no I/O)
- Phase 3: Sample 10% of tissue tiles
- Phase 4: Move RAM -> shared memory (one slide at a time, ~25 sec total)
- Phase 4: Workers read zero-copy from shared memory (no pickle serialization)

**H&E Image Fixes**
- Disabled `percentile_normalize()` in all 4 worker functions - raw pixel values work better for H&E
- Disabled normalization in tissue detection calibration
- HSPC detection now uses raw H&E staining without boosting background

**Crop Architecture Changes**
- `generate_detection_crop()` now returns `{'crop': b64, 'mask': b64}` (separate images)
- Mask outline drawn at HTML generation time, not baked into saved crop
- All worker functions updated to save `crop_b64` and `mask_b64` separately

**Bug Fixes**
| Bug | Fix |
|-----|-----|
| `slide_name` undefined in single-slide mode | Added `slide_name = czi_path.stem` at function start |
| Missing `shared_calibrate_tissue_threshold` | Added K-means calibration function for single-slide mode |
| OpenBLAS thread limit crash | Set `OPENBLAS_NUM_THREADS=32` in sbatch (was 128) |
| Standard mode copying tiles | Now uses SharedSlideManager for zero-copy access |

**Current Run (Job 2037820)**
- 16 slides, 4 L40s GPUs, 700GB RAM
- Output: `/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/mk_output/2026_01_24_BM_16slides_4gpu_RAM_SHM`

---

## Jan 20, 2026 - Streaming Multi-GPU Pipeline & Bug Fixes

**Memory-Efficient Streaming Mode**
Previous OOM crash (640GB peak for 16 slides) fixed with streaming approach:
- Phase 1: Open CZI readers only (~0 GB RAM)
- Phase 2: Tissue detection reads tiles on-demand from CZI
- Phase 3: Sample tiles from combined pool
- Phase 4: Load ONE slide at a time directly into shared memory
- Peak memory now ~320 GB (vs 640 GB before)

**Critical Bug Fixes:**
| Bug | Fix | Commit |
|-----|-----|--------|
| Coordinate mismatch in streaming | Convert 0-based to global coords when calling `get_tile()` | `3f7c177` |
| Tile ID collision across slides | Use `{slide_name}_{idx}` format for globally unique IDs | `37b0c88` |
| CZI reader FD leak on error | Move loader cleanup into `finally` block | `37b0c88` |

**Data Location (Cluster):**
CZI files at: `/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/bonemarrow/2025_11_18/`

---

## Jan 19, 2026 - Stability Fixes for Long Batch Runs
Previous runs crashed the machine (system restarts due to OOM). Fixed:

**Memory Validation (run_unified_FAST.py:75-181)**
- `validate_system_resources()` - Checks RAM/GPU before starting, aborts if <8GB available
- `get_safe_worker_count()` - Auto-reduces workers based on available memory
- Startup shows: RAM available, GPU memory, recommended worker count

**Sequential Mode Fixes**
- Added `--sequential` flag to single-slide mode (was missing)
- Wrapped sequential processing in try/finally for proper cleanup
- GPU cache cleared on errors (`torch.cuda.empty_cache()` in exception handler)
- More aggressive GC: every 10 tiles instead of every 50

**Network Mount Stability**
- Added `socket.setdefaulttimeout(60)` - prevents indefinite hangs on /mnt/x/
- Added `HDF5_USE_FILE_LOCKING=FALSE` - prevents file descriptor exhaustion
- CZI loader close() now properly releases reader reference

**Default Changes**
- Default tile size: 4096 -> 3000 (safer memory usage)
- Default workers: 4 (with auto-adjustment based on RAM)

**Safe Run Command**
```bash
python run_unified_FAST.py \
    --czi-paths /mnt/x/01_Users/EdRo_axioscan/bonemarrow/2025_11_18/*.czi \
    --output-dir /home/dude/mk_output/2025_01_19_BM_16slides_15pct \
    --sample-fraction 0.15 \
    --sequential \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000
```

---

## Jan 19, 2026 - Code Refactoring (Phase 1)
Consolidated duplicated code into shared modules:

**New Module: `segmentation/models/manager.py`**
- `ModelManager` class for centralized model loading (SAM2, Cellpose, ResNet)
- Lazy loading - models only load on first use
- `get_model_manager(device)` - singleton pattern per device
- `find_checkpoint(model_name)` - unified checkpoint discovery
- Context manager support for automatic cleanup

**New Module: `segmentation/processing/memory.py`**
- Extracted from run_unified_FAST.py
- `validate_system_resources(num_workers, tile_size)` - checks RAM/GPU
- `get_safe_worker_count(requested, tile_size)` - auto-adjusts workers
- `get_memory_usage()` / `log_memory_status()` - monitoring helpers

**Updated: `segmentation/utils/config.py`**
- Added `BATCH_SIZES` dict (resnet: 32, gc_interval: 10)
- Added `MEMORY_THRESHOLDS` dict (min_ram: 8GB, min_gpu: 6GB)
- Helper functions: `get_batch_size()`, `get_memory_threshold()`

**Updated: `segmentation/io/html_export.py`**
- Added `generate_dual_index_page()` for MK+HSPC batch runs
- Supports multiple cell types with per-type sections and export buttons

**Usage:**
```python
from segmentation.models import get_model_manager
from segmentation.processing.memory import validate_system_resources

# Validate before starting
result = validate_system_resources(num_workers=4, tile_size=3000)
if result['should_abort']:
    sys.exit(1)

# Use model manager
with get_model_manager("cuda") as manager:
    sam2_pred, sam2_auto = manager.get_sam2()
    # ... do work
# Automatic cleanup on exit
```

---

## Jan 19, 2026 - Code Refactoring (Phase 2 - Implementation)
Completed tile worker unification, strategy registry, and batch feature extraction.

**New Module: `segmentation/processing/mk_hspc_utils.py`**
- `ensure_rgb_array()` - Convert grayscale/RGBA to RGB
- `check_tile_validity()` - Empty tile detection
- `prepare_tile_for_detection()` - Percentile normalization
- `build_mk_hspc_result()` - Standardized result dict builder
- `extract_tile_from_shared_memory()` - Safe memory extraction

**Tile Worker Unification (COMPLETED):**
All 3 worker functions now use shared utilities from mk_hspc_utils.py:
- `process_tile_worker()` (line 789) - shared memory mode
- `process_tile_worker_with_data_and_slide()` (line 2356) - direct data with slide name
- `process_tile_worker_with_data()` (line 2716) - direct data mode

Code reduction: ~50% (18 LOC RGB -> 1 LOC, 6 LOC check -> 2 LOC, 21 LOC result -> 1 LOC call)

**New Module: `segmentation/detection/registry.py` (COMPLETED)**
- `StrategyRegistry` class with class methods (not instantiated)
- `register(cell_type, strategy_class)` - register new strategies
- `create(cell_type, **kwargs)` - instantiate strategies by name
- `list_strategies()` - returns `['nmj', 'mk', 'cell', 'vessel', 'mesothelium']`
- `get_strategy_class(cell_type)` - get class without instantiating

**Updated: `segmentation/detection/strategies/base.py` (COMPLETED)**
Added `_extract_full_features_batch()` method to DetectionStrategy base class:
- Extracts 22 morphological + 256 SAM2 + 2048 ResNet features per mask
- Batch processing for ResNet (configurable batch_size, default 32)
- Memory-efficient: sets SAM2 image once, resets after batch
- Strategies can call `self._extract_full_features_batch(masks, tile, models)` instead of duplicating ~200 LOC

---

## Jan 19, 2026 - Code Refactoring (Phase 3a - Split process_tile)
Split `UnifiedSegmenter.process_tile()` into separate MK and HSPC methods for better maintainability.

**Changes to `run_unified_FAST.py`:**
- `process_tile()` now delegates to `_process_tile_mk()` and `_process_tile_hspc()`
- Each method handles its own detection logic and feature extraction
- Shared preprocessing remains in the parent method
- Easier to modify MK vs HSPC detection independently

---

## Jan 19, 2026 - Code Refactoring (Phase 3b - Extract run_multi_slide phases)
Extracted 4 phases from `run_multi_slide_segmentation()` into separate functions.

**New Functions in `run_unified_FAST.py`:**
| Function | Purpose | Returns |
|----------|---------|---------|
| `_phase1_load_slides(czi_paths, tile_size, overlap, channel)` | Load all CZI slides into RAM | `(slide_data, slide_loaders)` |
| `_phase2_identify_tissue_tiles(slide_data, ...)` | Create tile grid, calibrate threshold, filter to tissue tiles | `(tissue_tiles, variance_threshold)` |
| `_phase3_sample_tiles(tissue_tiles, sample_fraction)` | Sample from combined tissue tile pool | `sampled_tiles` |
| `_phase4_process_tiles(sampled_tiles, ...)` | ML processing with multiprocessing pool | `(total_mk, total_hspc)` |

**Benefits:**
- Each phase is now testable independently
- Clear separation of concerns (I/O, preprocessing, sampling, ML)
- Main function reduced from ~574 lines to ~50 lines of orchestration
- Docstrings explain what each phase does and its parameters

---

## Jan 19, 2026 - Code Refactoring (Phase 4 - HTML Export Consolidation)
Consolidated HTML export functionality into a dedicated module.

**New Module: `segmentation/export/html_export.py`:**
- Extracted common HTML generation logic
- `HTMLExporter` class with configurable templates
- Shared CSS/JS generation functions
- Both MK/HSPC and NMJ export use the same base components

---

## Jan 19, 2026 - Code Refactoring (Phase 5 - Config Schema Validation)
Added TypedDict schemas and validation to `segmentation/utils/config.py`.

**New TypedDict Schemas:**
- `BatchSizeConfig` - ResNet batch size, GC interval
- `MemoryConfig` - Min RAM, min GPU thresholds
- `PixelSizeConfig` - Default pixel sizes
- `TileSizeConfig` - Tile dimensions, overlap

**New Functions:**
- `validate_config(config: dict) -> tuple[bool, list[str]]` - Validates config against schema
- `get_config_summary() -> dict` - Returns human-readable config overview

---

## Jan 19, 2026 - Code Refactoring (Phase 6 - Multi-Channel Feature Mixin)
Created a mixin class for channel-agnostic feature extraction.

**New Module: `segmentation/detection/strategies/mixins.py`:**
- `MultiChannelFeatureMixin` class with methods:
  - `extract_channel_stats(image, channel_idx)` - Per-channel intensity statistics
  - `extract_multichannel_features(image, mask)` - Combined multi-channel features (~56 features for 3 channels)
  - `extract_channel_intensity_simple(image, mask, channel)` - Quick single-channel extraction
- Used by NMJ strategy for 3-channel feature extraction
- Enables adding new channels without code duplication

---

## Jan 19, 2026 - Code Refactoring (Phase 7 - Entry Point Design Doc)
Created design document for unifying entry point scripts.

**New Document: `docs/ENTRY_POINT_UNIFICATION.md`:**
- 641-line design document outlining unified entry point architecture
- Proposes single `run.py` with subcommands: `segment`, `batch`, `classify`, `export`, `serve`, `info`
- Implementation plan with 5 phases
- Backward compatibility strategy for existing scripts

---

## Jan 19, 2026 - Type Hints Added
Added Python type hints to key modules for better IDE support and documentation.

**Modules with Type Hints:**
| Module | Functions Annotated |
|--------|---------------------|
| `segmentation/utils/config.py` | 12 public functions |
| `segmentation/processing/memory.py` | `validate_system_resources`, `get_safe_worker_count`, `get_memory_usage`, `log_memory_status` |
| `segmentation/processing/mk_hspc_utils.py` | `ensure_rgb_array`, `check_tile_validity`, `prepare_tile_for_detection`, `build_mk_hspc_result`, `extract_tile_from_shared_memory` |

---

## Jan 19, 2026 - Magic Numbers Extracted to Config
Extracted hardcoded magic numbers from `run_unified_FAST.py` into named constants.

**New Constants in `segmentation/utils/config.py`:**
```python
# Feature Dimensions
MORPHOLOGICAL_FEATURES_COUNT = 22      # Custom morphological + intensity features
SAM2_EMBEDDING_DIMENSION = 256         # SAM2 256D embedding vectors
RESNET_EMBEDDING_DIMENSION = 2048      # ResNet50 2048D feature vectors
TOTAL_FEATURES_PER_CELL = 2326         # Total: 22 + 256 + 2048

# Pixel Sizes
DEFAULT_PIXEL_SIZE_UM = 0.1725         # Default pixel size in micrometers

# Batch Processing
RESNET_INFERENCE_BATCH_SIZE = 16       # Default batch size for ResNet

# Processing Parameters
CPU_UTILIZATION_FRACTION = 0.8         # Use 80% of available CPU cores
```

**New Helper Functions:**
- `get_feature_dimensions()` - Returns dict with all feature dimension constants
- `get_cpu_worker_count(total_cores)` - Calculates safe worker count based on available cores

---

## Jan 19, 2026 - requirements.txt Created
Created comprehensive `requirements.txt` organized by category.

**Categories:**
- Core numerical (numpy, scipy, scikit-learn)
- Image processing (opencv-python, scikit-image, pillow)
- Deep learning (torch, torchvision)
- Segmentation models (cellpose, segment-anything-2)
- CZI/microscopy (aicspylibczi)
- Data storage (h5py, zarr)
- Utilities (tqdm, psutil, colorlog)

---

## Refactoring Summary Table

| Phase | Description | Status | Files |
|-------|-------------|--------|-------|
| Phase 1 | Model manager + memory module | Complete | `segmentation/models/manager.py`, `segmentation/processing/memory.py` |
| Phase 2 | Tile worker unification + StrategyRegistry | Complete | `segmentation/processing/mk_hspc_utils.py`, `segmentation/detection/registry.py` |
| Phase 3a | Split process_tile() -> MK/HSPC methods | Complete | `run_unified_FAST.py` |
| Phase 3b | Extract phases from run_multi_slide_segmentation() | Complete | `run_unified_FAST.py` |
| Phase 4 | HTML export consolidation | Complete | `segmentation/export/html_export.py` |
| Phase 5 | Config schema validation | Complete | `segmentation/utils/config.py` |
| Phase 6 | Multi-channel feature mixin | Complete | `segmentation/detection/strategies/mixins.py` |
| Phase 7 | Entry point unification design doc | Complete | `docs/ENTRY_POINT_UNIFICATION.md` |
| - | Type hints on key modules | Complete | Multiple modules |
| - | Magic numbers -> named constants | Complete | `segmentation/utils/config.py` |
| - | requirements.txt | Complete | `requirements.txt` |

---

## Jan 20, 2026 - Vessel Pipeline Major Enhancements

### New Modules Created

#### 1. Reporting Module (`segmentation/reporting/`)
Full PDF/HTML report generation with interactive Plotly visualizations.

```
segmentation/reporting/
├── __init__.py
├── stats.py             # VesselStatistics, quantiles, type breakdown
├── plots.py             # 10 Plotly visualizations (dark theme)
└── vessel_report.py     # VesselReport, BatchVesselReport classes
```

**Usage:**
```python
from segmentation.reporting import VesselReport, BatchVesselReport

# Single slide report
report = VesselReport.from_json("vessel_detections.json")
report.generate_html("report.html")

# Batch comparison (multiple slides)
batch = BatchVesselReport.from_directory("output/", pattern="**/vessel_detections.json")
batch.generate_html("batch_report.html")
```

#### 2. Classification Module (`segmentation/classification/`)
Two-stage ML classification pipeline.

```
segmentation/classification/
├── __init__.py
├── vessel_classifier.py      # Vessel TYPE classifier (capillary/arteriole/artery)
├── vessel_detector_rf.py     # Stage 2: Is this a vessel? (yes/no)
├── artery_vein_classifier.py # Stage 3: Artery vs vein
└── feature_selection.py      # Feature importance, RFECV, learning curves
```

**Two-stage pipeline:**
1. **Stage 1:** Candidate detection (permissive mode) - catches all potential vessels
2. **Stage 2:** VesselDetectorRF - filters false positives (vessel vs non-vessel)
3. **Stage 3:** ArteryVeinClassifier - classifies confirmed vessels

#### 3. Vessel-Specific Features (`segmentation/utils/vessel_features.py`)
28 biologically meaningful features replacing generic morphological features.

| Category | Features |
|----------|----------|
| Ring/Wall (6) | `ring_completeness`, `wall_uniformity`, `wall_thickness_cv`, `wall_asymmetry`, `lumen_wall_ratio`, `wall_fraction` |
| Shape (5) | `circularity`, `ellipticity`, `convexity`, `roughness`, `compactness` |
| Size (4) | `outer_diameter_um`, `inner_diameter_um`, `diameter_ratio`, `hydraulic_diameter` |
| Intensity (6) | `wall_intensity_mean/std`, `lumen_intensity_mean`, `wall_lumen_contrast`, `edge_gradient_mean/std` |
| Context (2) | `background_intensity`, `wall_background_contrast` |
| Derived (5) | `wall_thickness_range`, `wall_eccentricity`, `lumen_circularity`, `center_offset`, `wall_coverage` |

#### 4. Candidate Detection Mode
Permissive mode with relaxed thresholds for high recall.

```bash
python run_segmentation.py --czi-path slide.czi --cell-type vessel --candidate-mode
```

| Parameter | Standard | Candidate Mode |
|-----------|----------|----------------|
| `min_circularity` | 0.3 | 0.1 |
| `min_ring_completeness` | 0.5 | 0.2 |
| `max_aspect_ratio` | 4.0 | 6.0 |
| `min_diameter_um` | 10 | 5 |
| `max_diameter_um` | 1000 | 2000 |

---

### Vessel Pipeline Status (Jan 20, 2026)

| Component | Status | Files |
|-----------|--------|-------|
| Reporting module | Complete | `segmentation/reporting/` |
| Coordinate unification | Complete | `docs/COORDINATE_SYSTEM.md`, `segmentation/processing/coordinates.py` |
| Vessel type classifier | Complete | `segmentation/classification/vessel_classifier.py` |
| HTML annotation workflow | Complete | `segmentation/io/html_export.py`, `scripts/prepare_rf_training_data.py` |
| Vessel/non-vessel classifier | Complete | `segmentation/classification/vessel_detector_rf.py` |
| Artery/vein classifier | Complete | `segmentation/classification/artery_vein_classifier.py` |
| Vessel-specific features | Complete | `segmentation/utils/vessel_features.py` (32 features including log-transformed + size class) |
| Permissive candidate mode | Complete | `--candidate-mode` flag in `run_segmentation.py` |
| Full pipeline script | Complete | `scripts/run_full_pipeline.py` (3-stage pipeline orchestration) |
| Multi-channel feature extraction | Complete | `--all-channels` now works for vessels |
| CD31 tubular detection | Complete | Capillary detection via intensity thresholding + tubularity filter |
| LYVE1 lymphatic detection | Complete | Lymphatic detection with SMA check for collecting lymphatics |
| Multi-marker candidate merging | Complete | Union-Find IoU-based deduplication in `_merge_candidates()` |
| VesselTypeClassifier (6 types) | Complete | `segmentation/classification/vessel_type_classifier.py` |
| `--multi-marker` CLI mode | Complete | Auto-enables `--all-channels`, `--parallel-detection`, and candidate merging |

---

### Code Review Findings (Jan 20, 2026)

**Critical bugs - FIXED (commit 6ff6b0c):**
| Issue | Location | Status |
|-------|----------|--------|
| Uninitialized variables | `vessel.py:2995-2996` | Already initialized (false positive) |
| Empty array crash | `vessel.py:838` | Fixed - added empty array check |
| Hardcoded tile coords | `vessel.py:1031` | Fixed - added tile_x/tile_y params |

**Medium priority - FIXED:**
| Issue | Location | Status |
|-------|----------|--------|
| channel_names wrong indices | `run_segmentation.py:2942` | Fixed - uses actual channel indices |
| O(n^2) contourArea | `vessel.py:2198-2205` | Already cached (false positive) |

---

### Multi-Marker Parallel Detection Issue (Jan 20, 2026)

**Issue:** `--multi-marker` mode crashes during tile processing with no error message captured.

**Symptoms:**
- All 4 channels load successfully (~200GB total)
- Tissue detection completes (2030/2975 tiles = 68.2%)
- 203 tiles sampled for processing
- Process terminates with no tiles output

**Workaround:** Use basic single-channel mode while investigating:
```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 2 \
    --candidate-mode \
    --sample-fraction 0.10 \
    --load-to-ram \
    --output-dir /path/to/output
```

---

### Multi-Scale Vessel Detection (Jan 21, 2026)

**Problem Solved:** Large vessels (>100um) spanning multiple tiles were being fragmented.

**Architecture:**
```
Scale 1/8x: ~1.4 um/px -> Detects: Large arteries >100 um
Scale 1/4x: ~0.7 um/px -> Detects: Medium vessels 30-150 um
Scale 1x: 0.17 um/px   -> Detects: Capillaries 3-50 um
Merge: IoU-based deduplication (threshold 0.3)
```

**CLI Usage:**
```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 2 \
    --multi-scale \
    --scales "8,4,1" \
    --multiscale-iou-threshold 0.3
```

---

### Jan 24, 2026 - Lumen-First Multi-Scale Detection with Full Scale Range

**Major Update:** Extended multi-scale detection to include very coarse scales (1/64x, 1/32x) for detecting aorta and major arteries.

**New Configuration:**
- **Tile size**: 20,000x20,000 pixels (3.45mm coverage, fits mouse aorta 0.8-1.5mm)
- **Scales**: [64, 32, 16, 8, 4, 2, 1] (7 scales from coarsest to finest)
- **Per-channel photobleaching correction**: Applied to all RGB channels before display

**Scale-Specific Parameters:**
| Scale | Pixel Size | Min Lumen Area | Max Lumen Area | Target Vessels |
|-------|------------|----------------|----------------|----------------|
| 1/64x | ~11 um | 500,000 um^2 | 100,000,000 um^2 | Aorta (>1mm) |
| 1/32x | ~5.5 um | 100,000 um^2 | 25,000,000 um^2 | Major arteries (500-5000 um) |
| 1/16x | ~2.8 um | 20,000 um^2 | 8,000,000 um^2 | Large arteries (200-3000 um) |
| 1/8x | ~1.4 um | 5,000 um^2 | 1,000,000 um^2 | Medium-large (100-1000 um) |
| 1/4x | ~0.7 um | 1,500 um^2 | 100,000 um^2 | Medium (50-300 um) |
| 1/2x | ~0.35 um | 200 um^2 | 25,000 um^2 | Small-medium (20-150 um) |
| 1x | 0.17 um | 75 um^2 | 6,000 um^2 | Capillaries (5-75 um) |

**New Scripts:**
| Script | Purpose |
|--------|---------|
| `scripts/run_lumen_first_10pct.py` | Full multi-scale lumen-first detection with HTML export |
| `scripts/regenerate_html.py` | Fast HTML regeneration from saved crops |

**Key Features:**
1. **Lumen-first detection**: Find dark lumens, check for bright SMA+ walls around them
2. **Per-channel photobleaching correction**: Fixes banding artifacts in RGB display
3. **Crop caching**: Raw crops saved to disk for instant HTML style changes
4. **IoU deduplication**: 0.3 threshold, prefer detections from finer scales

**Usage:**
```bash
# Run lumen-first detection (100% of tiles)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg
python scripts/run_lumen_first_10pct.py

# Regenerate HTML with different contour style
python scripts/regenerate_html.py --thickness 10 --inner-dotted
```

**Output:** `/home/dude/vessel_output/lumen_first_test/`
