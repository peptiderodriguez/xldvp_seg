# xldvp_seg - Unified Cell Segmentation Pipeline

Automated segmentation and classification pipeline for multiple cell types in microscopy images:
- **MK** (Megakaryocytes) and **HSPC** in bone marrow
- **NMJ** (Neuromuscular Junctions) in muscle tissue
- **Vessels** (SMA+ blood vessel cross-sections)
- **Mesothelium** ribbons for laser microdissection

## Overview

This pipeline combines deep learning segmentation with Random Forest classification to identify and analyze bone marrow cells:
- **SAM2** (Segment Anything Model 2) for MK detection
- **Cellpose** for HSPC detection
- **ResNet-50** for feature extraction
- **Random Forest** classifiers for quality filtering

## Complete Workflow

This repository supports the full pipeline from raw images to validated cell counts:

1. **Sample & Annotate** - Segment 10% of tiles, export to HTML for manual annotation
2. **Train Classifiers** - Build MK and HSPC Random Forest models from annotations
3. **Validate** - Check classifier quality against thresholds (75% accuracy, 70% recall/precision)
4. **Deploy** - Apply validated classifiers to full 100% dataset

📖 **See [WORKFLOW.md](WORKFLOW.md) for detailed step-by-step instructions**

📋 **See [SESSION_NOTES.md](SESSION_NOTES.md) for development history and troubleshooting**

## Hardware Requirements

### Lab Machine (Your Setup)
- **CPU**: 24 cores
- **GPU**: 1× RTX 3090 (24GB VRAM)
- **RAM**: 512GB
- **Recommended config**: `--num-workers 4-6` with `--tile-size 4096`

### Minimum Requirements
- **GPU**: 24GB VRAM (RTX 3090 or better)
- **RAM**: 64GB system memory
- **Storage**: ~50GB for models and temp files

### Cluster (Viper/Raven)
- **GPUs**: 2× AMD MI250X (40GB VRAM each)
- **RAM**: 128-192GB
- **Recommended**: `--num-workers 10-16` with `--tile-size 4096`

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
```

### 2. Create Python Environment
```bash
# Using conda (recommended)
conda create -n mkseg python=3.10
conda activate mkseg
```

### 3. Install Dependencies

**For NVIDIA GPUs (RTX 3090, etc.):**
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**For AMD GPUs (Viper cluster):**
```bash
module load rocm/6.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
```

### 4. Download Model Checkpoints

```bash
mkdir -p checkpoints
cd checkpoints

# SAM2 checkpoint (~857MB)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Cellpose and ResNet models auto-download on first use.

## Quick Start

### 1. Run Segmentation on 10% Sample

```bash
python run_unified_FAST.py \
    --czi-path /path/to/slide.czi \
    --output-dir ./output_10pct \
    --tile-size 4096 \
    --num-workers 4 \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100 \
    --sample-fraction 0.10
```

### 2. Export to HTML for Annotation

```bash
python export_separate_mk_hspc.py \
    --base-dir ./output_10pct \
    --output-dir ./html_annotation \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100
```

Upload `html_annotation/` to GitHub Pages for annotation at:
`https://YOUR_USERNAME.github.io/mk_hspc_review/`

### 3. Convert Annotations to Training Data

After annotating and downloading `all_labels_combined.json`:

```bash
python convert_annotations_to_training.py \
    --annotations all_labels_combined.json \
    --base-dir ./output_10pct \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100 \
    --output training_data.json
```

### 4. Train Random Forest Classifiers

```bash
python train_separate_classifiers.py \
    --training-data training_data.json \
    --output-mk mk_classifier.pkl \
    --output-hspc hspc_classifier.pkl \
    --morph-only
```

### 5. Validate Classifiers

```bash
python validate_classifier.py \
    --mk-classifier mk_classifier.pkl \
    --hspc-classifier hspc_classifier.pkl \
    --min-accuracy 0.75 \
    --min-recall 0.70 \
    --min-precision 0.70
```

If validation passes, proceed to step 6. If not, collect more annotations and retrain.

### 6. Apply to Full Dataset

```bash
python run_unified_FAST.py \
    --czi-path /path/to/slide.czi \
    --output-dir ./output_100pct \
    --tile-size 4096 \
    --num-workers 4 \
    --mk-classifier mk_classifier.pkl \
    --hspc-classifier hspc_classifier.pkl \
    --mk-min-area-um 100 \
    --mk-max-area-um 2100 \
    --sample-fraction 1.0
```

## Key Parameters

### Segmentation
- `--czi-path`: Input CZI microscopy file
- `--output-dir`: Output directory for results
- `--tile-size`: Tile size in pixels (3000-4096, default: 4096)
- `--num-workers`: Parallel workers (adjust based on RAM/GPU)
- `--mk-min-area-um`: Minimum MK area in µm² (default: 100)
- `--mk-max-area-um`: Maximum MK area in µm² (default: 2100)
- `--sample-fraction`: Fraction of tiles (0.1 = 10%, 1.0 = 100%)
- `--mk-classifier`: Optional MK Random Forest classifier path
- `--hspc-classifier`: Optional HSPC Random Forest classifier path

### Memory Tuning (Lab Machine)

**For RTX 3090 with 512GB RAM:**
- `--tile-size 4096` (maximum resolution)
- `--num-workers 4-6` (optimal for 24GB GPU)
- Each worker: ~4-6GB GPU, ~10-15GB RAM
- Total usage: ~24GB GPU, ~60-90GB RAM

**If you encounter OOM:**
1. Reduce `--num-workers` from 6 → 4 → 2
2. Reduce `--tile-size` from 4096 → 3000
3. Check no other processes using GPU (`nvidia-smi`)

## Output Structure

```
output_dir/
├── mk/
│   └── tiles/
│       ├── 0/
│       │   ├── features.json       # MK features (2326 per cell)
│       │   ├── segmentation.h5     # Segmentation masks
│       │   ├── classes.csv         # Cell IDs
│       │   └── window.csv          # Tile coordinates
│       └── 1/
│           └── ...
└── hspc/
    └── tiles/
        └── ...
```

## Scripts Reference

### Core Scripts
- `run_unified_FAST.py` - Main segmentation pipeline
- `train_separate_classifiers.py` - Train MK/HSPC Random Forest models
- `validate_classifier.py` - Validate classifier performance
- `convert_annotations_to_training.py` - Convert HTML annotations to training format
- `export_separate_mk_hspc.py` - Export segmentation results to HTML

### Batch Scripts (for processing multiple slides)
- `run_unified_10pct.sh` - SLURM batch job for 10% sampling
- `run_all_slides_local.sh` - Local batch processing script
- `export_10pct_to_html.sh` - Export 10% results to HTML
- `export_all_html.sh` - Export all slides to HTML

### Documentation
- `WORKFLOW.md` - Complete step-by-step pipeline guide
- `SESSION_NOTES.md` - Development history, bugs fixed, troubleshooting

## Pixel Size Conversion

All area thresholds use **µm² (micrometers squared)** for biological interpretability:

- **Pixel size**: 0.1725 µm/px
- **Conversion factor**: 0.02975625 (= 0.1725²)
- **100-2100 µm²** = 3360-70573 px²

The scripts automatically handle conversion internally.

## Known Issues & Fixes

### ✅ Fixed Issues
- Export script filter mismatch (hardcoded px² vs µm² parameters)
- Device type bug in Cellpose initialization
- Memory leaks in SAM2/Cellpose processing
- OOM errors with >16 workers on 128GB RAM systems

### Active Optimizations
- Sequential predictor processing (no double-caching)
- Explicit garbage collection every 5 tiles
- Memory-mapped CZI files (shared across workers)
- Tissue detection to skip background (~30-50% speedup)

## Performance Tips

**Lab Machine (24 CPU, 512GB RAM, RTX 3090):**
- Use 4-6 workers for optimal GPU utilization
- Enable tile sampling (`--sample-fraction 0.10`) for initial annotation
- Full slide (100%) processing: ~2-4 hours per slide

**Cluster (2× MI250X, 192GB RAM):**
- Use 10-16 workers across 2 GPUs
- Each worker uses ~4GB GPU, ~7.5GB RAM
- Full slide (100%) processing: ~1-2 hours per slide

## Troubleshooting

### GPU Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `--num-workers` or `--tile-size`

### System RAM Exhausted
```
Killed (OOM)
```
**Solution**:
- Reduce `--num-workers` (each uses ~10-15GB RAM)
- On lab machine: 4-6 workers safe for 512GB RAM

### Low Classifier Accuracy (<75%)
**Solution**: Collect more training annotations (aim for 10% sampling = ~850 MK cells)

## Citation

If you use this code, please cite:
- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [Cellpose](https://github.com/MouseLand/cellpose)

## License

MIT License - See LICENSE file for details

## Contact

GitHub: [@peptiderodriguez](https://github.com/peptiderodriguez)
