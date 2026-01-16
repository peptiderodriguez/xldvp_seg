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
| **MK/HSPC output** | `/home/dude/xldvp_seg_output/` |
| **NMJ output** | `/home/dude/nmj_output/` |
| **NMJ test data** | `/home/dude/nmj_test_output/` |
| **Conda env** | `mkseg` (activate: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg`) |

### Common Tasks
- **Run unified segmentation:** `python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj`
- **Run NMJ inference:** See "NMJ Analysis Pipeline" section below
- **View results:** Start HTTP server + Cloudflare tunnel (port 8080 for MK, 8081 for NMJ)
- **Retrain classifier:** Use `train_nmj_classifier.py` with merged annotations

### Current State (as of Jan 2026)
- NMJ classifier trained: 95.83% accuracy (on 20251107_Fig5 slide, 115 pos / 123 neg samples)
- Full-slide inference in progress with trained classifier (threshold 0.75)
- Spatial grouping targets ~1500 µm² per set with 1200-1800 µm² tolerance
- Output maps: `nmj_map_spatial_tight.png`

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
- UIDs include coordinates: `{slide}_{celltype}_{round(x)}_{round(y)}`

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
│   ├── tile_{x}_{y}/
│   │   ├── mk_masks.h5
│   │   ├── hspc_masks.h5
│   │   └── features.h5
│   └── summary.json

/home/dude/code/xldvp_seg_repo/docs/
├── index.html
├── mk_page_1.html ... mk_page_N.html
└── hspc_page_1.html ... hspc_page_N.html
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

### CUDA/Boolean Type Error
SAM2 masks need explicit boolean conversion:
```python
mask = mask.astype(bool)  # Fix for NVIDIA CUDA compatibility
```

### HDF5 Plugin Path
If HDF5 errors occur:
```bash
export HDF5_PLUGIN_PATH=""
```

### Memory Management
- Explicit cleanup after each tile: `del masks, features; gc.collect()`
- `torch.cuda.empty_cache()` after MK processing
- Generator pattern for tile data to avoid memory spikes

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
1. **Intensity thresholding** - Bright fluorescent regions (default 99th percentile)
2. **Morphological cleanup** - Opening/closing to remove noise
3. **Solidity filtering** - NMJs have branched shapes with low solidity (max_solidity=0.85)
4. **Watershed expansion** - Masks are expanded to capture full BTX signal using watershed
5. **Smoothing** - Binary smoothing before and after expansion for clean boundaries
6. **Optional classifier** - ResNet18 trained to distinguish NMJ vs non-NMJ

### Current Dataset
- **Slide:** `20251109_PMCA1_647_nuc488-EDFvar-stitch.czi`
- **Location:** `/home/dude/nmj_test_output/`
- **Mosaic:** 254,976 × 100,503 px
- **Pixel size:** 0.1725 µm/px

### NMJ Classifier
- **Architecture:** ResNet18 (pretrained, fine-tuned)
- **Checkpoint:** `/home/dude/nmj_output/nmj_classifier.pth`
- **Validation accuracy:** 96.64%
- **Training data:** 544 positive, 642 negative annotations

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
- Annotation buttons (Yes/No) with localStorage persistence
- Export annotations to JSON
- Filtering by area and confidence
- Mask contour overlay (dotted green)
- Stats bar showing annotation progress

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
