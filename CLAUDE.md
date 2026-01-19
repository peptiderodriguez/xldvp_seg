# xldvp_seg - Session Notes

## Quick Start for New AI

**Read this file first!** It contains context for multiple image analysis pipelines:

1. **MK/HSPC Pipeline** - Bone marrow cell segmentation (Megakaryocytes + Stem Cells)
2. **NMJ Pipeline** - Neuromuscular junction detection in muscle tissue
3. **Vessel Pipeline** - Blood vessel morphometry (SMA+ ring detection)
4. **Mesothelium Pipeline** - Mesothelial ribbon detection for laser microdissection

### Documentation
- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Full user guide for all pipelines
- **[LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md)** - Laser microdissection export workflow

### Performance: RAM Loading
For large files on network mounts, use `--load-to-ram` to load all channels into RAM once:
```bash
python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj --load-to-ram
```
This eliminates repeated network I/O - the image is read once, then all tiles are extracted from RAM.

### Key Locations
| What | Where |
|------|-------|
| **This repo** | `/home/dude/code/xldvp_seg_repo/` |
| **MK/HSPC output** | `/home/dude/mk_output/` |
| **NMJ output** | `/home/dude/nmj_output/` |
| **Vessel output** | `/home/dude/vessel_output/` |
| **Conda env** | `mkseg` (activate: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg`) |

### Project Structure Convention
**All new projects MUST follow this naming structure:**
```
/home/dude/{celltype}_output/{project_name}/
├── html/                        # Annotation viewer
│   ├── index.html
│   └── {celltype}_page_*.html
├── {celltype}_detections.json   # All detections with UIDs
├── {celltype}_coordinates.csv   # Quick coordinate export
└── {slide_name}/                # Per-slide data (if multi-slide)
    └── tiles/{tile_id}/
        ├── segmentation.h5
        ├── features.json
        └── window.csv
```

**Examples:**
- NMJ single slide: `/home/dude/nmj_output/20251107_Fig5_fresh/`
- MK 16-slide batch: `/home/dude/mk_output/2025_11_18_BM_16slides/`
- Vessel project: `/home/dude/vessel_output/2025_01_SMA_study/`

**Naming rules:**
- `{celltype}` = `nmj`, `mk`, `hspc`, `vessel`, `mesothelium`
- `{project_name}` = descriptive name with date (e.g., `20251107_Fig5_fresh`, `2025_11_18_BM_16slides`)

### Common Tasks
- **Run unified segmentation:** `python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj`
- **Run NMJ inference:** See "NMJ Analysis Pipeline" section below
- **View results:** Start HTTP server + Cloudflare tunnel (port 8080 for MK, 8081 for NMJ)
- **Retrain classifier:** Use `train_nmj_classifier.py` with merged annotations

### Current State (as of Jan 19, 2026)
- **Stability fixes**: Added memory validation, sequential mode fixes, network timeout handling
- **Code refactoring**: Phase 1 + Phase 2 complete (model manager, memory utils, tile workers, strategy registry)
- **Multi-channel feature extraction**: Now extracts ~2,400 features from all 3 channels (nuclear, BTX, NFL)
- **BTX-only thresholding**: Mask detection uses only BTX channel (ch1/647nm), not averaged channels
- **HTML improvements**: True RGB display (3-channel combined), white mask outlines, channel legend on floating bar, cards sorted by area ascending, stats with area µm²/px and solidity
- NMJ classifier: RF model trained on multi-channel features for NMJ vs autofluorescence
- Working slide: `20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi` (3-channel)

### Jan 19, 2026 - Stability Fixes for Long Batch Runs
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
- Default tile size: 4096 → 3000 (safer memory usage)
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

### Jan 19, 2026 - Code Refactoring (Phase 1)
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

### Jan 19, 2026 - Code Refactoring (Phase 2 - Implementation)
Completed tile worker unification, strategy registry, and batch feature extraction.

**New Module: `segmentation/processing/mk_hspc_utils.py`**
- `ensure_rgb_array()` - Convert grayscale/RGBA to RGB
- `check_tile_validity()` - Empty tile detection
- `prepare_tile_for_detection()` - Percentile normalization
- `build_mk_hspc_result()` - Standardized result dict builder
- `extract_tile_from_shared_memory()` - Safe memory extraction

**Tile Worker Unification (COMPLETED):**
All 3 worker functions now use shared utilities from mk_hspc_utils.py:
- `process_tile_worker()` (line 789) - shared memory mode ✓
- `process_tile_worker_with_data_and_slide()` (line 2356) - direct data with slide name ✓
- `process_tile_worker_with_data()` (line 2716) - direct data mode ✓

Code reduction: ~50% (18 LOC RGB → 1 LOC, 6 LOC check → 2 LOC, 21 LOC result → 1 LOC call)

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

### Jan 19, 2026 - Code Refactoring (Phase 3a - Split process_tile)
Split `UnifiedSegmenter.process_tile()` into separate MK and HSPC methods for better maintainability.

**Changes to `run_unified_FAST.py`:**
- `process_tile()` now delegates to `_process_tile_mk()` and `_process_tile_hspc()`
- Each method handles its own detection logic and feature extraction
- Shared preprocessing remains in the parent method
- Easier to modify MK vs HSPC detection independently

### Jan 19, 2026 - Code Refactoring (Phase 3b - Extract run_multi_slide phases)
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

### Jan 19, 2026 - Code Refactoring (Phase 4 - HTML Export Consolidation)
Consolidated HTML export functionality into a dedicated module.

**New Module: `segmentation/export/html_export.py`:**
- Extracted common HTML generation logic
- `HTMLExporter` class with configurable templates
- Shared CSS/JS generation functions
- Both MK/HSPC and NMJ export use the same base components

### Jan 19, 2026 - Code Refactoring (Phase 5 - Config Schema Validation)
Added TypedDict schemas and validation to `segmentation/utils/config.py`.

**New TypedDict Schemas:**
- `BatchSizeConfig` - ResNet batch size, GC interval
- `MemoryConfig` - Min RAM, min GPU thresholds
- `PixelSizeConfig` - Default pixel sizes
- `TileSizeConfig` - Tile dimensions, overlap

**New Functions:**
- `validate_config(config: dict) -> tuple[bool, list[str]]` - Validates config against schema
- `get_config_summary() -> dict` - Returns human-readable config overview

### Jan 19, 2026 - Code Refactoring (Phase 6 - Multi-Channel Feature Mixin)
Created a mixin class for channel-agnostic feature extraction.

**New Module: `segmentation/detection/strategies/mixins.py`:**
- `MultiChannelFeatureMixin` class with methods:
  - `extract_channel_stats(image, channel_idx)` - Per-channel intensity statistics
  - `extract_multichannel_features(image, mask)` - Combined multi-channel features (~56 features for 3 channels)
  - `extract_channel_intensity_simple(image, mask, channel)` - Quick single-channel extraction
- Used by NMJ strategy for 3-channel feature extraction
- Enables adding new channels without code duplication

### Jan 19, 2026 - Code Refactoring (Phase 7 - Entry Point Design Doc)
Created design document for unifying entry point scripts.

**New Document: `docs/ENTRY_POINT_UNIFICATION.md`:**
- 641-line design document outlining unified entry point architecture
- Proposes single `run.py` with subcommands: `segment`, `batch`, `classify`, `export`, `serve`, `info`
- Implementation plan with 5 phases
- Backward compatibility strategy for existing scripts

### Jan 19, 2026 - Type Hints Added
Added Python type hints to key modules for better IDE support and documentation.

**Modules with Type Hints:**
| Module | Functions Annotated |
|--------|---------------------|
| `segmentation/utils/config.py` | 12 public functions |
| `segmentation/processing/memory.py` | `validate_system_resources`, `get_safe_worker_count`, `get_memory_usage`, `log_memory_status` |
| `segmentation/processing/mk_hspc_utils.py` | `ensure_rgb_array`, `check_tile_validity`, `prepare_tile_for_detection`, `build_mk_hspc_result`, `extract_tile_from_shared_memory` |

**Type Annotations Used:**
```python
from typing import Dict, List, Optional, Tuple, Union, Any
```

### Jan 19, 2026 - Magic Numbers Extracted to Config
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

### Jan 19, 2026 - requirements.txt Created
Created comprehensive `requirements.txt` organized by category.

**Categories:**
- Core numerical (numpy, scipy, scikit-learn)
- Image processing (opencv-python, scikit-image, pillow)
- Deep learning (torch, torchvision)
- Segmentation models (cellpose, segment-anything-2)
- CZI/microscopy (aicspylibczi)
- Data storage (h5py, zarr)
- Utilities (tqdm, psutil, colorlog)

**Includes PyTorch CUDA installation instructions in comments.**

---

## Refactoring Summary Table

| Phase | Description | Status | Files |
|-------|-------------|--------|-------|
| Phase 1 | Model manager + memory module | ✅ Complete | `segmentation/models/manager.py`, `segmentation/processing/memory.py` |
| Phase 2 | Tile worker unification + StrategyRegistry | ✅ Complete | `segmentation/processing/mk_hspc_utils.py`, `segmentation/detection/registry.py` |
| Phase 3a | Split process_tile() → MK/HSPC methods | ✅ Complete | `run_unified_FAST.py` |
| Phase 3b | Extract phases from run_multi_slide_segmentation() | ✅ Complete | `run_unified_FAST.py` |
| Phase 4 | HTML export consolidation | ✅ Complete | `segmentation/export/html_export.py` |
| Phase 5 | Config schema validation | ✅ Complete | `segmentation/utils/config.py` |
| Phase 6 | Multi-channel feature mixin | ✅ Complete | `segmentation/detection/strategies/mixins.py` |
| Phase 7 | Entry point unification design doc | ✅ Complete | `docs/ENTRY_POINT_UNIFICATION.md` |
| - | Type hints on key modules | ✅ Complete | Multiple modules |
| - | Magic numbers → named constants | ✅ Complete | `segmentation/utils/config.py` |
| - | requirements.txt | ✅ Complete | `requirements.txt` |

---

## Unified Pipeline Architecture

### Shared Modules (`shared/`)
| Module | Purpose |
|--------|---------|
| `tissue_detection.py` | K-means based tissue detection, variance thresholding |
| `html_export.py` | Unified dark-theme HTML annotation interface |
| `config.py` | Centralized configuration management and defaults |
| `coordinates.py` | Coordinate conversion helpers (prevents x/y swap bugs) |
| `czi_loader.py` | Unified CZI loading with optional RAM caching |

### Coordinate System
**All coordinates are stored as [x, y] (horizontal, vertical).**

- `tile_origin`: [x, y] position of tile top-left corner in mosaic
- `local_centroid`: [x, y] position of cell center within tile
- `global_centroid` / `global_center`: [x, y] position in full mosaic

**UID Formats:**
- **MK/HSPC (run_unified_FAST.py):** `{slide}_{celltype}_{global_id}` (e.g., `2025_11_18_FGC1_mk_123`)
- **NMJ/Vessel (run_segmentation.py):** `{slide}_{celltype}_{round(x)}_{round(y)}` (e.g., `slide_nmj_45678_12345`)

Note: NumPy arrays use [row, col] indexing internally, but all stored/exported coordinates are [x, y].

### Entry Points
| Script | Use Case |
|--------|----------|
| `run_segmentation.py` | Unified entry point with tissue detection |
| `run_lmd_export.py` | Export to Leica LMD format with clustering |
| `run_unified_FAST.py` | Original MK/HSPC pipeline |
| `run_nmj_segmentation.py` | Original NMJ segmentation |
| `run_nmj_inference.py` | NMJ classification with trained model |

### Unified Segmentation Usage
```bash
# NMJ detection with tissue filtering
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel 1 \
    --sample-fraction 0.10

# MK detection
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type mk \
    --channel 0

# Vessel detection (SMA staining)
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 0 \
    --min-vessel-diameter 10 \
    --max-vessel-diameter 500

# Key improvements over original scripts:
# 1. Automatic tissue detection before sampling (samples % of TISSUE tiles, not all tiles)
# 2. Unified HTML export with consistent dark theme
# 3. Single global localStorage key for annotations
# 4. Universal IDs with global coordinates for all detections
# 5. CSV export with coordinates in pixels and µm
```

### Output Files
Each run produces:
- `{cell_type}_detections.json` - All detections with universal IDs and global coordinates
- `{cell_type}_coordinates.csv` - Quick export with center coordinates
- `tiles/tile_X_Y/{cell_type}_masks.h5` - Per-tile mask arrays
- `html/index.html` - Interactive annotation viewer

---

## Project Overview
Bone Marrow Megakaryocyte (MK) and Hematopoietic Stem/Progenitor Cell (HSPC) segmentation pipeline for whole-slide CZI images from Axioscan microscope.

## Hardware Configuration
- **CPU:** 48 cores
- **RAM:** 432 GB
- **GPU:** NVIDIA RTX 4090 (24 GB VRAM)
- **Storage:** Network mount at `/mnt/x/01_Users/EdRo_axioscan/bonemarrow/`

## Current Dataset
- **Location:** `/mnt/x/01_Users/EdRo_axioscan/bonemarrow/2025_11_18/`
- **Slides:** 16 CZI files (~20-23 GB each, ~350 GB total)
- **Naming:** `2025_11_18_{group}{number}.czi`
  - Groups: FGC, FHU, MGC, MHU (1-4 each)

## Pipeline Capabilities

### 1. Unified Batch Segmentation (`run_unified_FAST.py`)
- **Multi-slide batch processing:** Load models ONCE, process all slides
- **Unified sampling:** Sample tiles across ALL slides for balanced representation
- **Memory-efficient:** Loads slides into RAM, processes tiles as generators
- **4-Phase pipeline:**
  1. Load all slides into RAM
  2. Identify tissue tiles (grid-based, tissue detection)
  3. Sample X% from combined pool
  4. ML processing with SAM2 + Cellpose + ResNet

### 2. Cell Detection Models
- **SAM2 (Segment Anything Model 2):** Initial mask proposals
  - Checkpoint: `checkpoints/sam2.1_hiera_large.pt`
  - Config: `configs/sam2.1/sam2.1_hiera_l.yaml`
- **Cellpose:** HSPC detection (nuclei-based)
  - Model: `cyto3`
- **ResNet50:** MK classification (MK vs non-MK)
  - Checkpoint: `checkpoints/best_model.pth`
  - Classes: megakaryocyte, non-megakaryocyte

### 3. Filtering & Post-processing
- **MK area filter:** Configurable min/max in µm² (default 200-2000 µm²)
- **Pixel conversion:** Automatic based on CZI metadata (typically 0.22 µm/px)
- **Largest connected component:** For MKs, keeps only largest contiguous region
- **HSPC confidence sorting:** By mask solidity

### 4. HTML Export (Integrated)
- **Automatic export:** Happens while slides are still in RAM
- **Separate pages:** MK and HSPC annotation pages
- **Features:**
  - Slides summary subtitle (e.g., "16 slides (FGC1, FGC2, ...)")
  - Unique card IDs using global_id (e.g., `mk_123`, `hspc_456`)
  - Dotted light green mask contours
  - Centered crops on mask centroid
  - Percentile normalization (5th-95th)
  - Local stats (current page yes/no counts)
  - Global stats (across all pages via localStorage)
  - Keyboard navigation (Y/N/Arrow keys)
  - Pagination (configurable samples per page)

### 5. Output Structure
```
/home/dude/xldvp_seg_output/
├── {slide_name}/
│   ├── mk/tiles/{tile_id}/
│   │   ├── segmentation.h5      # Mask labels
│   │   ├── features.json        # Per-cell features (see format below)
│   │   ├── window.csv           # Tile coordinates in mosaic
│   │   └── classes.csv          # Global IDs for this tile
│   ├── hspc/tiles/{tile_id}/
│   │   └── (same structure)
│   └── summary.json

/home/dude/code/xldvp_seg_repo/docs/
├── index.html
├── mk_page_1.html ... mk_page_N.html
└── hspc_page_1.html ... hspc_page_N.html
```

**features.json format (per tile):**
```json
[
  {
    "id": "det_122",
    "global_id": 123,
    "center": [45678, 12345],  // Global [x, y] coordinates
    "features": {"area": 1500, "solidity": 0.85, ...}
  }
]
```

### 6. External Access
- **Cloudflare Tunnel:** Installed at `~/cloudflared`
  - No bandwidth limits (unlike ngrok)
  - Usage: `~/cloudflared tunnel --url http://localhost:8080`
  - Provides `*.trycloudflare.com` URL

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_local.sh` | Main entry point, batch/single mode |
| `run_unified_FAST.py` | Core segmentation + integrated export |
| `export_separate_mk_hspc.py` | Standalone export (reads from CZI) |
| `convert_annotations_to_training.py` | Convert HTML annotations to training data |

## Command Line Usage

### Batch Mode (16 slides)
```bash
./run_local.sh  # MODE="batch" in script
```

### Single Slide
```bash
# Edit run_local.sh: MODE="single", SLIDE="2025_11_18_FGC1"
./run_local.sh
```

### Key Parameters
- `--tile-size`: Tile dimensions (default 3000x3000)
- `--sample-fraction`: Fraction of tiles to process (default 0.10 = 10%)
- `--mk-min-area-um` / `--mk-max-area-um`: MK size filter in µm²
- `--html-output-dir`: Where to write HTML export
- `--samples-per-page`: Samples per HTML page (default 300)

## Processing Times (Estimates)
- **Phase 1 (Loading):** ~30-60 min for 16 slides
- **Phase 2-3 (Tile ID + Sampling):** ~20 min
- **Phase 4 (ML Processing):** ~18-20 hours for 10% sample
- **HTML Export:** ~10 min

## Previous Run Results
- **Total MKs detected:** 39,362
- **Total HSPCs detected:** 12,486
- **Processing time:** ~20 hours

## Troubleshooting

### System Restarts / OOM Crashes
If the system restarts during batch runs:
1. Use `--sequential` flag (processes one tile at a time)
2. Reduce `--num-workers` to 2 or 1
3. Reduce `--tile-size` from 4096 to 3000
4. Memory validation now runs at startup and will warn/abort if insufficient

### CUDA/Boolean Type Error
SAM2 masks need explicit boolean conversion:
```python
mask = mask.astype(bool)  # Fix for NVIDIA CUDA compatibility
```

### HDF5 Plugin Path / File Descriptor Errors
If HDF5 errors or "too many open files" occur:
```bash
export HDF5_PLUGIN_PATH=""
export HDF5_USE_FILE_LOCKING=FALSE  # Added automatically in run_unified_FAST.py
```

### Network Mount Hangs
If processing hangs on /mnt/x/ network mount:
- Socket timeout now set to 60s automatically
- If still hanging, check network connectivity: `ls /mnt/x/`

### Memory Management
- Explicit cleanup after each tile: `del masks, features; gc.collect()`
- `torch.cuda.empty_cache()` after MK processing and on errors
- Generator pattern for tile data to avoid memory spikes
- GC runs every 10 tiles in sequential mode (was 50)

### Monitoring a Long Run
```bash
# Watch log
tail -f /home/dude/mk_output/*/run.log

# Check GPU
nvidia-smi -l 1

# Check RAM
watch -n 5 free -h

# Check process
ps aux | grep python | grep -v grep
```

## Dependencies
- Python 3.10+ (mkseg conda environment)
- PyTorch with CUDA
- SAM2, Cellpose, scikit-learn
- aicspylibczi (CZI reading)
- scipy, numpy, h5py

---

## NMJ (Neuromuscular Junction) Analysis Pipeline

### Overview
Secondary pipeline for NMJ detection and classification in muscle tissue CZI images.

### Detection Method
NMJ detection uses a multi-stage approach:
1. **Intensity thresholding** - BTX channel only (ch1/647nm) for mask formation
2. **Morphological cleanup** - Opening/closing to remove noise
3. **Solidity filtering** - NMJs have branched shapes with low solidity (max_solidity=0.85)
4. **Watershed expansion** - Masks are expanded to capture full BTX signal using watershed
5. **Smoothing** - Binary smoothing before and after expansion for clean boundaries
6. **Optional classifier** - Random Forest trained on multi-channel features

### Multi-Channel Feature Extraction
With `--all-channels` flag, extracts ~2,400 features per detection:
- **Per-channel stats (45):** mean, std, max, min, median, percentiles, variance, skewness, kurtosis, IQR, dynamic_range, CV for each of ch0/ch1/ch2
- **Inter-channel ratios (8):** btx_nuclear_ratio, btx_nfl_ratio, nuclear_nfl_ratio, channel_specificity, etc.
- **ResNet-50 embeddings (2,048):** From true 3-channel RGB image
- **SAM2 embeddings (256):** From true 3-channel RGB image
- **Morphological features (~25):** area, perimeter, solidity, skeleton_length, etc.

Channel mapping for 3-channel slides:
- **R (ch0):** Nuclear stain (488nm)
- **G (ch1):** Bungarotoxin/BTX (647nm) - NMJ marker
- **B (ch2):** Neurofilament/NFL (750nm)

```bash
# Multi-channel NMJ detection
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel 1 \
    --all-channels \
    --load-to-ram \
    --sample-fraction 0.10
```

### Current Dataset
- **Slide:** `20251109_PMCA1_647_nuc488-EDFvar-stitch.czi`
- **Location:** `/home/dude/nmj_test_output/`
- **Mosaic:** 254,976 × 100,503 px
- **Pixel size:** 0.1725 µm/px

### NMJ Classifier
Two classifier options are available for distinguishing NMJs from autofluorescence:

#### Option 1: ResNet18 on Images (`train_nmj_classifier.py`)
- **Architecture:** ResNet18 (pretrained, fine-tuned)
- **Checkpoint:** `/home/dude/nmj_output/nmj_classifier.pth`
- **Validation accuracy:** 96.64%
- **Training data:** 544 positive, 642 negative annotations

#### Option 2: Random Forest on Extracted Features (`train_nmj_classifier_features.py`)
- **Architecture:** Random Forest classifier on 2,382 multi-channel features
- **Checkpoint:** `nmj_classifier_rf.pkl`
- **Accuracy:** 91% trained on 796 annotations
- **Top discriminative features:**
  - `ch0_cv` (coefficient of variation in nuclear channel)
  - `ch1` intensity stats (BTX channel - primary NMJ marker)
  - Morphological features: area, perimeter

**Training the feature-based classifier:**
```bash
python train_nmj_classifier_features.py \
    --detections /path/to/nmj_detections.json \
    --annotations /path/to/annotations.json \
    --output-dir /path/to/output
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `run_nmj_segmentation.py` | Initial segmentation to find NMJ candidates |
| `run_nmj_inference.py` | Classify candidates using trained model |
| `export_nmj_results_html.py` | Generate HTML viewer with annotation support |
| `train_nmj_classifier.py` | Train/retrain classifier from annotations |
| `merge_nmj_annotations.py` | Merge old + new annotation files |

### Inference Pipeline
```bash
# 1. Run inference with trained classifier
python run_nmj_inference.py \
    --czi-path /path/to/slide.czi \
    --model-path /home/dude/nmj_output/nmj_classifier.pth

# 2. Export to HTML with filters
python export_nmj_results_html.py \
    --results-json /path/to/nmj_detections.json \
    --czi-path /path/to/slide.czi \
    --min-area 27 --min-confidence 0.75
```

### HTML Viewer Features
- **RGB display**: True 3-channel visualization (all channels combined) instead of single channel
- **White mask outlines**: Solid white contours (thickness 5) for visibility on RGB backgrounds
- **Channel legend on floating bar**: R=nuc488, G=Bgtx647, B=NfL750
- **Cards sorted by area**: Ascending order (smallest NMJs first)
- **Stats display**: area (um² and px), solidity; confidence hidden when 100%
- Annotation buttons (Yes/No) with localStorage persistence
- Export annotations to JSON
- Filtering by area and confidence
- **Short mask IDs**: Displays `nmj_x_y` instead of full slide name

**Note:** The `export_nmj_results_html.py` script loads all 3 channels from the CZI file to generate true RGB visualization crops, providing better context for annotation than single-channel display.

### Spatial Grouping
NMJs can be grouped into sets for analysis:

**Output files:** `/home/dude/nmj_output/20251109_PMCA1_647_nuc488-EDFvar-stitch/inference/`
- `nmj_detections.json` - All classified NMJs with coordinates
- `nmj_sets_spatial_tight.json` - Spatially grouped sets (~1500 µm² each)
- `nmj_sets_coordinates.csv` - CSV with global X,Y coordinates
- `nmj_map_spatial_tight.png` - Visual map of sets

**Current grouping (tight distribution 1200-1800 µm²):**
- **10 main sets:** 179 NMJs, mean 1511 µm², std dev 168 µm²
- **12 outlier sets:** 108 NMJs (small isolated clusters)
- **Total:** 287 NMJs, 24,375 µm² (after filtering >1000 µm² singles)

### Filtering Criteria
- Area ≥ 27 µm² (minimum)
- Area < 1000 µm² (exclude large singles)
- Solidity ≤ 0.85 (branched structures have low solidity)
- Confidence ≥ 75%
- Spatial proximity < 2mm for grouping

### Annotation Format
Two formats supported:
```json
// Old format
{"positive": ["id1", "id2"], "negative": ["id3", "id4"]}

// New format (from HTML export)
{"annotations": {"id1": "yes", "id2": "no"}}
```

### Cloudflare Tunnel for Review
```bash
# Port 8081 for NMJ viewer
~/cloudflared tunnel --url http://localhost:8081
# Serves from: /home/dude/nmj_output/.../inference_v2/html/
```

---

## Vessel Morphometry Pipeline

### Overview
Detection and measurement of blood vessel cross-sections in SMA-stained whole mouse coronal sections.

### Detection Method
Uses contour hierarchy analysis to find ring structures:
1. Adaptive + Otsu thresholding to segment SMA+ regions
2. `cv2.findContours` with `RETR_CCOMP` to get parent-child hierarchy
3. Identify outer contours (adventitial side) that have inner contours (lumen)
4. Fit ellipses to both outer and inner contours
5. Calculate wall thickness at 36 angles around the vessel
6. Optional CD31 validation (endothelial marker at lumen boundary)

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-vessel-diameter` | 10 µm | Minimum outer diameter |
| `--max-vessel-diameter` | 1000 µm | Maximum outer diameter |
| `--min-wall-thickness` | 2 µm | Minimum wall thickness |
| `--max-aspect-ratio` | 4.0 | Exclude longitudinal sections |
| `--min-circularity` | 0.3 | Minimum circularity (0-1) |
| `--min-ring-completeness` | 0.5 | Minimum SMA+ fraction of perimeter |
| `--cd31-channel` | None | CD31 channel for validation |

### Vessel-Specific Features
In addition to standard 2326 features, vessels get:
- `outer_diameter_um`, `inner_diameter_um`
- `wall_thickness_mean_um`, `wall_thickness_std_um`, `wall_thickness_min_um`, `wall_thickness_max_um`
- `lumen_area_um2`, `wall_area_um2`
- `orientation_deg`, `aspect_ratio`
- `ring_completeness`
- `cd31_validated`, `cd31_score`
- `confidence` (high/medium/low)

### Usage
```bash
# Basic vessel detection
python run_segmentation.py \
    --czi-path /path/to/sma_stained.czi \
    --cell-type vessel \
    --channel 0 \
    --sample-fraction 0.10

# With CD31 validation
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 0 \
    --cd31-channel 1 \
    --min-vessel-diameter 15 \
    --max-vessel-diameter 300
```

### Output
- `vessel_detections.json` - All vessels with contours in global coordinates
- `vessel_coordinates.csv` - Quick export: uid, center (px/µm), diameter, wall thickness, confidence
- `html/` - Interactive viewer for annotation
