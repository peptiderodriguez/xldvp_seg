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

### Installation (Server/Cluster)
```bash
# 1. Create conda environment
conda create -n mkseg python=3.11 -y
conda activate mkseg

# 2. Clone and install
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
./install.sh  # Auto-detects CUDA, installs PyTorch + SAM2 + all deps

# Options:
./install.sh --cuda 11.8    # Force specific CUDA version
./install.sh --rocm         # For AMD GPUs
./install.sh --cpu          # CPU only
```

### Multi-GPU Processing (Slurm Clusters)
True multi-GPU tile processing: 4 GPUs, each processes 1 tile at a time from the same slide.

```bash
# Create input file list
ls /path/to/slides/*.czi > input_files.txt

# Submit Slurm job (4x L40s GPUs)
sbatch slurm/run_multigpu.sbatch input_files.txt /path/to/output
```

**How it works:**
1. Phase 1: Load all slides into RAM (single process)
2. Phase 2-3: Identify tissue tiles, sample from pool
3. Phase 4: Spawn 4 GPU workers, each loads models ONCE, processes tiles from queue
4. Phase 5: HTML export

**CLI flags:**
- `--multi-gpu` - Enable multi-GPU mode
- `--num-gpus 4` - Number of GPUs (default: 4)

**Module:** `segmentation/processing/multigpu.py`
- `MultiGPUTileProcessor` class with queue-based architecture
- Workers pinned to GPUs via `CUDA_VISIBLE_DEVICES`

### Current State (as of Jan 20, 2026)
- **Multi-GPU streaming mode**: Memory-efficient pipeline that streams CZI directly to shared memory
- **Bug fixes (Jan 20)**: Coordinate system fix, tile ID collision fix, CZI reader cleanup
- **Multi-GPU support**: 4 GPUs process tiles in parallel (`--multi-gpu` flag)
- **Installable package**: `./install.sh` handles PyTorch+CUDA, SAM2, all dependencies
- **Slurm support**: Ready-to-use sbatch scripts in `slurm/` directory
- **Stability fixes**: Memory validation, sequential mode fixes, network timeout handling
- **Code refactoring**: Phase 1-7 complete (model manager, memory utils, tile workers, strategy registry)
- **Multi-channel feature extraction**: Now extracts ~2,400 features from all 3 channels (nuclear, BTX, NFL)
- **BTX-only thresholding**: Mask detection uses only BTX channel (ch1/647nm), not averaged channels
- **HTML improvements**: True RGB display (3-channel combined), white mask outlines, channel legend on floating bar, cards sorted by area ascending, stats with area µm²/px and solidity
- NMJ classifier: RF model trained on multi-channel features for NMJ vs autofluorescence
- Working slide: `20251107_Fig5_nuc488_Bgtx647_NfL750-1-EDFvar-stitch.czi` (3-channel)

### Jan 20, 2026 - Streaming Multi-GPU Pipeline & Bug Fixes

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

**Canonical UID Format (All Cell Types):**
All pipelines now use the **spatial UID format** for consistency:
```
{slide}_{celltype}_{round(x)}_{round(y)}
```
Examples:
- `2025_11_18_FGC1_mk_12346_67890`
- `slide_01_hspc_5000_3000`
- `muscle_sample_nmj_1234_5678`
- `tissue_vessel_9876_5432`

**Legacy Support:** The old numeric `global_id` is preserved in `features.json` for backwards compatibility, but the spatial UID is now the primary identifier.

See `docs/COORDINATE_SYSTEM.md` for full specification.

**Coordinate Utilities:** Use `segmentation.processing.coordinates` for:
- `generate_uid()` - Generate spatial UIDs
- `parse_uid()` - Parse UID into components
- `migrate_uid_format()` - Convert legacy UIDs
- `validate_xy_coordinates()` - Validate coordinate bounds
- `format_coordinates_for_export()` - Format coordinates for JSON/CSV

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
  - **Auto-embed prior annotations** (see below)

#### Auto-Embed Prior Annotations (Round-2 Workflow)
When regenerating HTML after classifier training, you can pre-load the round-1 annotations so they're visible alongside the classifier's predictions. This enables continuing annotation from where you left off.

**Usage in `export_samples_to_html`:**
```python
export_samples_to_html(
    samples,
    html_dir,
    'nmj',
    prior_annotations="/path/to/nmj_annotations.json",  # NEW: auto-embed prior annotations
    ...
)
```

**How it works:**
1. Reads the prior annotations JSON file (exported from round-1 HTML viewer)
2. Generates `preload_annotations.js` with annotations in localStorage format
3. Injects `<script src="preload_annotations.js">` into each HTML page
4. On page load, annotations are merged into localStorage (existing > preloaded)

**Annotation JSON formats supported:**
```json
// Format 1: Export format
{"positive": ["uid1", "uid2"], "negative": ["uid3"]}

// Format 2: Alternative format
{"annotations": {"uid1": "yes", "uid2": "no"}}
```

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

# Multi-marker parallel detection (SMA + CD31 + LYVE1)
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 1 \
    --all-channels \
    --channel-names "nuclear,sma,cd31,lyve1" \
    --parallel-detection \
    --parallel-workers 3
```

### Parallel Multi-Marker Detection
For slides with multiple vessel markers (SMA, CD31, LYVE1), parallel detection runs all marker analyses simultaneously using CPU threads:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--parallel-detection` | False | Enable parallel multi-marker detection |
| `--parallel-workers` | 3 | Number of parallel workers (one per marker) |
| `--channel-names` | None | Comma-separated channel names (e.g., "nuclear,sma,cd31,lyve1") |

**How it works:**
- Uses `ThreadPoolExecutor` for CPU parallelism (detection is CPU-bound OpenCV operations)
- GPU operations (SAM2, ResNet) remain sequential
- 3x speedup for multi-marker slides (48 CPU cores available)
- Detects: SMA+ ring vessels, CD31+ capillaries (tubular), LYVE1+ lymphatics

**Marker-specific detection:**
| Marker | Detection Method | Vessel Types |
|--------|-----------------|--------------|
| SMA | Contour hierarchy (ring detection) | Arteries, arterioles, large veins |
| CD31 | Connected components (tubular) | Capillaries (SMA-negative) |
| LYVE1 | Connected components (intensity) | Lymphatic vessels |

### Multi-Marker Convenience Mode (`--multi-marker`)
The `--multi-marker` flag is a convenience option that auto-enables all multi-marker features:

```bash
# Full multi-marker vessel detection with 6-type classification
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 1 \
    --multi-marker \
    --channel-names "nuclear,sma,cd31,lyve1" \
    --candidate-mode \
    --sample-fraction 0.10
```

**What `--multi-marker` does:**
1. Auto-enables `--all-channels` (loads all channels to RAM)
2. Auto-enables `--parallel-detection` (runs SMA/CD31/LYVE1 detection in parallel)
3. Merges overlapping candidates from different markers using IoU-based deduplication
4. Extracts multi-channel intensity features (per-channel wall/lumen stats + cross-channel ratios)

**6-Type Vessel Classification:**
| Vessel Type | Marker Profile | Typical Size |
|-------------|----------------|--------------|
| `artery` | SMA+, CD31- | >100µm diameter, thick wall |
| `arteriole` | SMA+, CD31- | 10-100µm diameter |
| `vein` | SMA+/weak, CD31+ | Large, thin wall |
| `capillary` | SMA-, CD31+ | 3-10µm diameter, tubular |
| `lymphatic` | SMA-, LYVE1+ | Irregular shape |
| `collecting_lymphatic` | SMA+, LYVE1+ | Has smooth muscle wall |

Use `--vessel-type-classifier /path/to/model.joblib` to apply trained ML classifier for 6-type classification.

### Output
- `vessel_detections.json` - All vessels with contours in global coordinates
- `vessel_coordinates.csv` - Quick export: uid, center (px/µm), diameter, wall thickness, confidence
- `html/` - Interactive viewer for annotation

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

**Visualizations:** Diameter/wall thickness histograms, scatter plots, vessel type pie charts, quality metrics, batch comparison violin plots.

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

**Training scripts:**
```bash
# Train vessel detector (binary: vessel vs non-vessel)
python scripts/train_vessel_detector.py \
    --annotations annotations.json \
    --detections vessel_detections.json \
    --output-dir ./classifier_output

# Train with stratified sampling by size (prevents size bias)
python scripts/train_vessel_detector.py \
    --annotations annotations.json \
    --detections vessel_detections.json \
    --output-dir ./classifier_output \
    --stratify-by-size

# Run full pipeline
python scripts/run_full_pipeline.py \
    --input candidates.json \
    --vessel-detector vessel_detector.joblib \
    --artery-vein artery_vein.joblib \
    --output final_results.json
```

**Stratified Sampling by Vessel Size:**
To prevent the classifier from learning size as a proxy for vessel vs non-vessel, use `--stratify-by-size`:
- Bins vessels into 3 size classes: small (<50 um), medium (50-200 um), large (>200 um)
- Balances training data across size classes
- Uses compound stratification (label + size) for cross-validation
- Logs size distribution analysis before and after balancing

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

**Usage:**
```python
from segmentation.utils import VESSEL_FEATURE_NAMES, extract_vessel_features
features = extract_vessel_features(outer_contour, inner_contour, image, pixel_size_um)
```

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

**New features:**
- **Open vessel detection:** Detects arcs/curves (not just closed rings)
- **`detection_confidence`:** 0-1 score based on ring completeness, circularity, wall uniformity, aspect ratio
- **Arc-specific fields:** `arc_length_um`, `avg_curvature`, `straightness`

#### 5. Enhanced HTML Annotation for RF Training
Updated `segmentation/io/html_export.py` with RF-ready exports.

**New features:**
- **Batch annotation:** Select multiple → bulk Yes/No
- **Filtering:** By diameter range, confidence level, annotated/unannotated
- **Statistics display:** Yes/No counts, progress bar, remaining count
- **Export formats:**
  - CSV (RF-ready): `uid, annotation, feature1, feature2, ...`
  - sklearn JSON: `{X: [], y: [], feature_names: []}`

**RF training data preparation:**
```bash
# Basic preparation
python scripts/prepare_rf_training_data.py \
    --annotations vessel_annotations.json \
    --detections vessel_detections.json \
    --output-dir ./rf_training_data

# With stratified sampling by vessel size (recommended)
python scripts/prepare_rf_training_data.py \
    --annotations vessel_annotations.json \
    --detections vessel_detections.json \
    --output-dir ./rf_training_data \
    --stratify-by-size

# With custom samples per size class
python scripts/prepare_rf_training_data.py \
    --annotations vessel_annotations.json \
    --detections vessel_detections.json \
    --output-dir ./rf_training_data \
    --stratify-by-size \
    --samples-per-size-class 50
```

**Size Classes for Stratification:**
| Class | Diameter Range | Typical Vessel Type |
|-------|----------------|---------------------|
| capillary | 0-10 um | Capillaries |
| arteriole | 10-50 um | Arterioles, venules |
| small_artery | 50-150 um | Small arteries/veins |
| artery | >150 um | Large arteries/veins |

### Vessel Pipeline Workflow (Complete)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VESSEL ANALYSIS PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STAGE 1: Candidate Detection (--candidate-mode)                   │
│  ┌─────────────────────────────────────────┐                       │
│  │ • Relaxed thresholds (high recall)      │                       │
│  │ • Detects rings + arcs                   │                       │
│  │ • Extracts 28 vessel-specific features   │                       │
│  │ • + 256 SAM2 + 2048 ResNet features      │                       │
│  └─────────────────────────────────────────┘                       │
│                         ↓                                           │
│  STAGE 2: HTML Annotation                                          │
│  ┌─────────────────────────────────────────┐                       │
│  │ • Review candidates in browser           │                       │
│  │ • Mark YES (vessel) or NO (not vessel)   │                       │
│  │ • Batch operations, filtering            │                       │
│  │ • Export to CSV/sklearn JSON             │                       │
│  └─────────────────────────────────────────┘                       │
│                         ↓                                           │
│  STAGE 3: Train Vessel Detector RF                                 │
│  ┌─────────────────────────────────────────┐                       │
│  │ • Binary classifier: vessel vs non-vessel│                       │
│  │ • Uses your annotations as ground truth  │                       │
│  │ • Stratified by size to avoid bias       │                       │
│  └─────────────────────────────────────────┘                       │
│                         ↓                                           │
│  STAGE 4: Run on 100% of tiles                                     │
│  ┌─────────────────────────────────────────┐                       │
│  │ • Candidate detection on all tiles       │                       │
│  │ • Apply trained RF to filter             │                       │
│  │ • Output: confirmed vessels only         │                       │
│  └─────────────────────────────────────────┘                       │
│                         ↓                                           │
│  STAGE 5: Artery vs Vein Classification                            │
│  ┌─────────────────────────────────────────┐                       │
│  │ • Only for confirmed vessels             │                       │
│  │ • Features: wall thickness, diameter     │                       │
│  │ • RF or rule-based classification        │                       │
│  └─────────────────────────────────────────┘                       │
│                         ↓                                           │
│  STAGE 6: Full Report                                              │
│  ┌─────────────────────────────────────────┐                       │
│  │ • Vessel counts by type                  │                       │
│  │ • Diameter/thickness distributions       │                       │
│  │ • Plotly interactive visualizations      │                       │
│  └─────────────────────────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Status (Jan 20, 2026)

| Component | Status | Files |
|-----------|--------|-------|
| Reporting module | ✅ Complete | `segmentation/reporting/` |
| Coordinate unification | ✅ Complete | `docs/COORDINATE_SYSTEM.md`, `segmentation/processing/coordinates.py` |
| Vessel type classifier | ✅ Complete | `segmentation/classification/vessel_classifier.py` |
| HTML annotation workflow | ✅ Complete | `segmentation/io/html_export.py`, `scripts/prepare_rf_training_data.py` |
| Vessel/non-vessel classifier | ✅ Complete | `segmentation/classification/vessel_detector_rf.py` |
| Artery/vein classifier | ✅ Complete | `segmentation/classification/artery_vein_classifier.py` |
| Vessel-specific features | ✅ Complete | `segmentation/utils/vessel_features.py` (32 features including log-transformed + size class) |
| Permissive candidate mode | ✅ Complete | `--candidate-mode` flag in `run_segmentation.py` |
| Full pipeline script | ✅ Complete | `scripts/run_full_pipeline.py` (3-stage pipeline orchestration) |
| Partial vessel detection | 🔄 In progress | Building blocks exist but **no orchestration**. See notes below. |
| Full feature extraction | ✅ Complete | Vessel strategy extracts 22 morph + 256 SAM2 + 2048 ResNet + 32 vessel-specific features. |
| Size bias fixes | ✅ Complete | Log-transformed features, size_class categorical, size-adaptive sampling all implemented in `vessel_features.py`. |
| Code review | ✅ Complete | Module imports, feature extraction, pipeline scripts verified Jan 20, 2026. |
| Multi-channel feature extraction | ✅ Complete | `--all-channels` now works for vessels. `segmentation/utils/vessel_features.py` |
| CD31 tubular detection | ✅ Complete | Capillary detection via intensity thresholding + tubularity filter |
| LYVE1 lymphatic detection | ✅ Complete | Lymphatic detection with SMA check for collecting lymphatics |
| Multi-marker candidate merging | ✅ Complete | Union-Find IoU-based deduplication in `_merge_candidates()` |
| VesselTypeClassifier (6 types) | ✅ Complete | `segmentation/classification/vessel_type_classifier.py` |
| `--multi-marker` CLI mode | ✅ Complete | Auto-enables `--all-channels`, `--parallel-detection`, and candidate merging |

### Code Review Findings (Jan 20, 2026)

**Critical bugs - FIXED (commit 6ff6b0c):**
| Issue | Location | Status |
|-------|----------|--------|
| Uninitialized variables | `vessel.py:2995-2996` | ✅ Already initialized (false positive) |
| Empty array crash | `vessel.py:838` | ✅ Fixed - added empty array check |
| Hardcoded tile coords | `vessel.py:1031` | ✅ Fixed - added tile_x/tile_y params |

**Medium priority - FIXED:**
| Issue | Location | Status |
|-------|----------|--------|
| channel_names wrong indices | `run_segmentation.py:2942` | ✅ Fixed - uses actual channel indices |
| O(n²) contourArea | `vessel.py:2198-2205` | ✅ Already cached (false positive) |

**Remaining (low priority):**
- `vessel.py:3116` - Fixed 72-point sampling in candidate mode vs adaptive in regular mode
- `run_segmentation.py:632` - `merge_iou_threshold` not exposed to CLI

### Multi-Marker Parallel Detection Issue (Jan 20, 2026)

**Issue:** `--multi-marker` mode crashes during tile processing with no error message captured.

**Symptoms:**
- All 4 channels load successfully (~200GB total)
- Tissue detection completes (2030/2975 tiles = 68.2%)
- 203 tiles sampled for processing
- Process starts at high CPU (300%+) indicating multi-threaded detection
- After ~5 minutes of tile processing, process terminates with no tiles output
- No OOM messages in dmesg, no Python traceback captured

**Suspected cause:** Issue in `_detect_all_markers_parallel()` or one of:
- `_detect_cd31_tubular()` - CD31 capillary detection
- `_detect_lyve1_structures()` - LYVE1 lymphatic detection
- Thread synchronization issue in ThreadPoolExecutor

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

**To investigate:**
1. Add try/except around each detection method in `_detect_all_markers_parallel()`
2. Run with verbose logging to capture which tile/marker fails
3. Test each detection method individually (SMA-only, CD31-only, LYVE1-only)

**Notes on incomplete items:**

**Partial vessel detection (cross-tile merging) - 🔄 In Progress:**

*What exists:*
- `PartialVessel` dataclass for storing partial vessel data
- `CrossTileMergeConfig` for configuring merge parameters
- `_detect_boundary_partial_vessels()` - detects vessels touching tile edges
- `_merge_partial_vessels()` - merges two partial vessels into one
- `merge_across_tiles()` - attempts to match and merge across adjacent tiles
- `get_partial_vessels()` / `clear_partial_vessels()` - accessor methods
- `_partial_vessels` dictionary stores detected partials keyed by tile coordinates

*What is missing:*
- **No orchestration code** - `merge_across_tiles()` is defined but **never called** anywhere in the codebase
- The tile processing loop in `run_segmentation.py` does not call the merge function after processing adjacent tiles
- No post-processing step exists to iterate through tile pairs and merge vessels

*To complete:*
1. Modify `run_segmentation.py` to track processed tiles and call `merge_across_tiles()` after adjacent tiles are both done
2. OR add a separate post-processing script that loads partial vessel data and performs merging
3. Need to handle the merged vessels in the final output (deduplicate, update UIDs, etc.)

### Multi-Channel Vessel Feature Extraction (Jan 20, 2026)

The `--all-channels` flag now works for vessels, enabling per-channel intensity features and cross-channel ratios.

**Channel Mapping (from CZI metadata):**
| Channel | Fluorophore | Stain | Purpose |
|---------|-------------|-------|---------|
| ch0 | AF488 | Nuclear | Reference |
| ch1 | AF647 | **SMA** | Detection channel |
| ch2 | AF750 | PM750 | Plasma membrane |
| ch3 | AF555 | CD31 | Endothelial marker |

**New Features (per vessel):**
| Category | Count | Examples |
|----------|-------|----------|
| Per-channel wall intensity | 16 | `ch0_wall_mean`, `sma_wall_std`, `cd31_wall_cv` |
| Per-channel lumen intensity | 16 | `ch0_lumen_mean`, `sma_lumen_median` |
| Cross-channel ratios | 6 | `sma_cd31_wall_ratio`, `cd31_lumen_wall_ratio` |

**Total features:** ~2378 (22 morph + 32 vessel + 38 multichannel + 256 SAM2 + 2048 ResNet)

**Usage:**
```bash
# CD31 slide
python run_segmentation.py \
    --czi-path /path/to/4channel_slide.czi \
    --cell-type vessel \
    --channel 1 \
    --all-channels \
    --channel-names "nuclear,sma,pm,cd31" \
    --candidate-mode \
    --sample-fraction 0.10 \
    --sequential \
    --load-to-ram \
    --output-dir /home/dude/vessel_output/project_name

# LYVE1 slide (lymphatics)
python run_segmentation.py \
    --czi-path /path/to/lyve1_slide.czi \
    --cell-type vessel \
    --channel 1 \
    --all-channels \
    --channel-names "nuclear,sma,pm,lyve1" \
    --candidate-mode \
    --sample-fraction 0.10 \
    --sequential \
    --load-to-ram \
    --output-dir /home/dude/vessel_output/lyve1_project
```

**Files modified:**
- `segmentation/utils/vessel_features.py` - Added `extract_multichannel_intensity_features()`, `compute_channel_ratios()`, `extract_all_vessel_features_multichannel()`
- `run_segmentation.py` - Extended `--all-channels` to support vessels
- `segmentation/detection/strategies/vessel.py` - Added `extra_channels` parameter to `detect()`

### Multi-Marker All-Vessel Detection (Planned)

**Goal:** Detect ALL vessel types (arteries, veins, capillaries, lymphatics) using multiple markers.

**Biological marker profiles:**
| Vessel Type | SMA | CD31 | LYVE1 | Current Status |
|-------------|-----|------|-------|----------------|
| Artery/Arteriole | +++ | + | - | ✅ Detected (SMA ring) |
| Vein | +/- | + | - | ⚠️ Partial (weak SMA) |
| Capillary | - | + | - | ❌ Missed (no SMA) |
| Lymphatic | - | - | + | ❌ Missed (no SMA) |
| Collecting lymphatic | + | - | + | ⚠️ Partial (rare) |

**Planned approach: Single permissive detection + classify**
```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Per-Channel Detection (parallel)                  │
│    SMA → rings (arteries/veins)                            │
│    CD31 → tubular structures (capillaries)                 │
│    LYVE1 → irregular structures (lymphatics)               │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Merge overlapping candidates (IoU > 0.5)         │
├─────────────────────────────────────────────────────────────┤
│  Stage 3: Multi-channel feature extraction                  │
│    + marker profile features (sma_score, cd31_score, etc.) │
├─────────────────────────────────────────────────────────────┤
│  Stage 4: Classification                                    │
│    A) VesselDetectorRF: Is this a vessel? (yes/no)         │
│    B) VesselTypeClassifier: artery/vein/capillary/lymph    │
└─────────────────────────────────────────────────────────────┘
```

**New CLI flags (planned):**
- `--multi-marker` - Enable multi-marker detection
- `--detect-capillaries` - Enable CD31-based capillary detection
- `--detect-lymphatics` - Enable LYVE1-based lymphatic detection
- `--lyve1-channel` - LYVE1 channel index

**Implementation status:**
- [x] CD31 tubular detection - `_detect_cd31_tubular()` in vessel.py
- [x] LYVE1 detection (lymphatics + collecting) - `_detect_lyve1_structures()` in vessel.py
- [x] Candidate merging (Union-Find) - `_merge_candidates()` in vessel.py
- [x] VesselTypeClassifier (6 types) - `segmentation/classification/vessel_type_classifier.py`
- [x] Wire up `--multi-marker` CLI flag

**Vessel type classification (marker combinations):**
| SMA | CD31 | LYVE1 | Type |
|-----|------|-------|------|
| +++ | + | - | artery |
| ++ | + | - | arteriole |
| +/- | + | - | vein |
| - | + | - | capillary |
| - | - | + | lymphatic |
| + | - | + | collecting_lymphatic |

**New methods in vessel.py:**
- `_detect_cd31_tubular()` - CD31+ tubular structures (capillaries, 3-20µm)
- `_detect_lyve1_structures()` - LYVE1+ structures (lymphatics, checks SMA for collecting)
- `_merge_candidates()` - Deduplicates overlapping detections via IoU
- `_compute_iou()` - IoU between contours

### Multi-Scale Vessel Detection (Jan 21, 2026)

**Problem Solved:** Large vessels (>100µm) spanning multiple tiles were being fragmented. Multi-scale detection runs at coarse resolution first (1/8x) where large vessels fit within a single tile, then progressively finer scales (1/4x, 1x) for smaller vessels.

**Architecture:**
```
┌────────────────────────────────────────────────────────────────────┐
│                    MULTI-SCALE DETECTION PIPELINE                   │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Scale 1/8x: ~1.4 µm/px                                           │
│  ├─ Tile covers ~5500 µm (4000px × 1.38µm)                        │
│  ├─ Detects: Large arteries >100 µm                               │
│  └─ Fast: 1/64th of full-res pixels                               │
│                              ↓                                     │
│  Scale 1/4x: ~0.7 µm/px                                           │
│  ├─ Tile covers ~2760 µm                                          │
│  ├─ Detects: Medium vessels 30-150 µm                             │
│  └─ Medium: 1/16th of full-res pixels                             │
│                              ↓                                     │
│  Scale 1x: 0.17 µm/px                                             │
│  ├─ Tile covers ~680 µm                                           │
│  ├─ Detects: Capillaries 3-50 µm                                  │
│  └─ Slow but necessary for small vessels                          │
│                              ↓                                     │
│  Merge: IoU-based deduplication (threshold 0.3)                   │
│  └─ Vessels detected at multiple scales → keep finest             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**CLI Usage:**
```bash
# Enable multi-scale detection
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 2 \
    --multi-scale \
    --scales "8,4,1" \
    --multiscale-iou-threshold 0.3 \
    --sample-fraction 0.10 \
    --load-to-ram \
    --output-dir /path/to/output
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--multi-scale` | False | Enable multi-scale detection |
| `--scales` | "8,4,1" | Comma-separated scale factors (coarse to fine) |
| `--multiscale-iou-threshold` | 0.3 | IoU threshold for deduplication |

**Scale-Specific Parameters (automatic):**
| Scale | Pixel Size | Min Diameter | Max Diameter | Target Vessels |
|-------|------------|--------------|--------------|----------------|
| 1/8x | ~1.4 µm | 100 µm | 2000 µm | Large arteries |
| 1/4x | ~0.7 µm | 30 µm | 200 µm | Medium vessels |
| 1x | 0.17 µm | 3 µm | 75 µm | Capillaries |

**Implementation Files:**
| File | Purpose |
|------|---------|
| `segmentation/utils/multiscale.py` | Scale utilities, IoU computation, merge logic |
| `segmentation/io/czi_loader.py` | `get_tile(scale_factor=N)` for multi-res reading |
| `segmentation/detection/strategies/vessel.py` | `detect_multiscale()` method |
| `run_segmentation.py` | CLI arguments and orchestration |

**Key Functions:**
- `multiscale.get_scale_params(scale)` - Returns detection params for scale
- `multiscale.compute_iou_contours()` - Mask-based IoU between contours
- `multiscale.merge_detections_across_scales()` - Deduplicate, prefer finer scale
- `CZILoader.get_tile(..., scale_factor=N)` - Extract at reduced resolution

### Recommended Tile Size
**Default tile size: 4000x4000 pixels** (increased from 3000x3000 for better vessel detection at boundaries)

### Jan 24, 2026 - Lumen-First Multi-Scale Detection with Full Scale Range

**Major Update:** Extended multi-scale detection to include very coarse scales (1/64x, 1/32x) for detecting aorta and major arteries.

**New Configuration:**
- **Tile size**: 20,000×20,000 pixels (3.45mm coverage, fits mouse aorta 0.8-1.5mm)
- **Scales**: [64, 32, 16, 8, 4, 2, 1] (7 scales from coarsest to finest)
- **Per-channel photobleaching correction**: Applied to all RGB channels before display

**Scale-Specific Parameters:**
| Scale | Pixel Size | Min Lumen Area | Max Lumen Area | Target Vessels |
|-------|------------|----------------|----------------|----------------|
| 1/64x | ~11 µm | 500,000 µm² | 100,000,000 µm² | Aorta (>1mm) |
| 1/32x | ~5.5 µm | 100,000 µm² | 25,000,000 µm² | Major arteries (500-5000 µm) |
| 1/16x | ~2.8 µm | 20,000 µm² | 8,000,000 µm² | Large arteries (200-3000 µm) |
| 1/8x | ~1.4 µm | 5,000 µm² | 1,000,000 µm² | Medium-large (100-1000 µm) |
| 1/4x | ~0.7 µm | 1,500 µm² | 100,000 µm² | Medium (50-300 µm) |
| 1/2x | ~0.35 µm | 200 µm² | 25,000 µm² | Small-medium (20-150 µm) |
| 1x | 0.17 µm | 75 µm² | 6,000 µm² | Capillaries (5-75 µm) |

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
- `vessel_detections.json` - All vessels with contours
- `crops/` - Raw JPEG crops for each vessel
- `html/` - Interactive annotation viewer
