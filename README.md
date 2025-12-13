# BM_MK_seg - Bone Marrow MK/HSPC Segmentation

Automated segmentation pipeline for Megakaryocytes (MK) and Hematopoietic Stem/Progenitor Cells (HSPC) in bone marrow microscopy images.

## Overview

This pipeline uses multiple deep learning models to segment and extract features from large CZI microscopy images:
- **SAM2** (Segment Anything Model 2) for MK detection
- **Cellpose** for HSPC detection
- **ResNet-50** for feature extraction

## Features

- Memory-mapped image loading for efficient processing of large files
- Multiprocessing with GPU worker distribution
- Tissue detection to skip background regions
- Comprehensive feature extraction (2326 features per cell)

## Hardware Requirements

### Minimum
- **GPU**: 24GB VRAM (tested on RTX 3090)
- **RAM**: 64GB system memory
- **Storage**: ~50GB for models and temp files

### Recommended for 3000×3000 tiles with 2-3 workers
- **GPU**: 24GB VRAM
- **RAM**: 128GB+
- **CPUs**: 8+ cores

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/BM_MK_seg.git
cd BM_MK_seg
```

### 2. Create Python Environment
```bash
# Using conda (recommended)
conda create -n mkseg python=3.10
conda activate mkseg

# Or using venv
python -m venv mkseg_env
source mkseg_env/bin/activate
```

### 3. Install Dependencies

**For NVIDIA GPUs (RTX 3090, etc.):**
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**For AMD GPUs (Viper cluster with MI250X):**
```bash
module load rocm/6.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

### 4. Download Model Checkpoints

Models will auto-download on first use, but you can pre-download:

```bash
mkdir checkpoints
cd checkpoints

# SAM2 checkpoint (~857MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Cellpose and ResNet auto-download on first use
```

## Usage

### Basic Usage (Single Slide)

```bash
python run_unified_FAST.py \
    --czi-path /path/to/slide.czi \
    --output-dir ./output/slide_name \
    --tile-size 3000 \
    --num-workers 2 \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100 \
    --sample-fraction 0.10
```

### Parameters

- `--czi-path`: Path to input CZI file
- `--output-dir`: Output directory for results
- `--tile-size`: Tile size in pixels (default: 3000, max: 4096)
- `--num-workers`: Number of parallel workers (2-4 recommended for RTX 3090)
- `--mk-min-area-um`: Minimum MK area in µm² (default: 100)
- `--mk-max-area-um`: Maximum MK area in µm² (default: 2100)
- `--sample-fraction`: Fraction of tiles to process (0.1 = 10%)

### Memory Optimization

**For RTX 3090 (24GB VRAM):**
- Use `--tile-size 3000` and `--num-workers 2`
- Estimated memory: ~12GB VRAM, ~31GB system RAM

**For larger GPUs (>32GB VRAM):**
- Can use `--tile-size 4096` and `--num-workers 3-4`

**If you encounter OOM errors:**
1. Reduce `--num-workers` to 1
2. Reduce `--tile-size` to 2048
3. Check that no other processes are using GPU

## Output Structure

```
output_dir/
├── mk/
│   └── tiles/
│       ├── 0/
│       │   ├── features.json       # MK features for this tile
│       │   ├── segmentation.h5     # MK segmentation masks
│       │   ├── classes.csv         # Cell IDs
│       │   └── window.csv          # Tile coordinates
│       └── 1/
│           └── ...
└── hspc/
    └── tiles/
        ├── 0/
        │   ├── features.json       # HSPC features for this tile
        │   ├── segmentation.h5     # HSPC segmentation masks
        │   ├── classes.csv         # Cell IDs
        │   └── window.csv          # Tile coordinates
        └── 1/
            └── ...
```

## Known Issues & Solutions

### ROCm-Specific (AMD GPUs on Viper cluster)
The code includes a patch for ROCm INT_MAX issues in SAM2. This is automatically applied and can be ignored on NVIDIA GPUs.

### Memory Leaks Fixed
- ✅ SAM2 mask arrays deleted after filtering
- ✅ Sequential predictor processing (no double-caching)
- ✅ Explicit garbage collection every 5 tiles
- ✅ Cellpose intermediate results cleared

### Performance Tips
1. **Tissue calibration**: Automatically skips 30-50% of background tiles
2. **Memory-mapped files**: Shared image data across workers (no copies)
3. **GPU distribution**: Workers evenly distributed across available GPUs

## Development

### Code Structure

**Main Components:**
- `UnifiedSegmenter`: Main segmentation class
  - SAM2 automatic mask generation for MKs
  - Cellpose + SAM2 refinement for HSPCs
  - Feature extraction (morphology + embeddings)

- `process_tile_worker`: Multiprocessing worker function
  - Processes individual tiles
  - Shares memory-mapped image data

- `run_unified_segmentation`: Main entry point
  - Coordinates multiprocessing pool
  - Manages GPU distribution
  - Saves results to disk

### Memory Management Strategy

1. **Large arrays deleted immediately after use**
2. **Sequential model processing** (avoid simultaneous caching)
3. **Explicit predictor cache clearing** (`reset_predictor()`)
4. **Periodic garbage collection**

## Citation

If you use this code, please cite:
- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [Cellpose](https://github.com/MouseLand/cellpose)

## License

[Add your license here]

## Contact

[Add contact information]
