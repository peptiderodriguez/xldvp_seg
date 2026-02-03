# NMJ Detection Pipeline Guide

Run the full NMJ detection pipeline with the pre-trained morph+SAM2 classifier.

## Prerequisites

1. **Clone the repo:**
   ```bash
   git clone https://github.com/peptiderodriguez/xldvp_seg.git
   cd xldvp_seg
   ```

2. **Install dependencies:**
   ```bash
   ./install.sh  # Auto-detects CUDA
   ```

3. **Activate environment:**
   ```bash
   source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg
   ```

4. **Download SAM2 checkpoint** (if not included):
   ```bash
   # SAM2.1 Large model (~900MB)
   wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   ```

## Quick Start

```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --output-dir /path/to/output \
    --cell-type nmj \
    --channel 1 \
    --intensity-percentile 97 \
    --sample-fraction 1.0 \
    --all-channels \
    --load-to-ram \
    --extract-full-features \
    --skip-deep-features \
    --tile-overlap 0.1 \
    --nmj-classifier checkpoints/nmj_classifier_morph_sam2.joblib \
    --no-serve
```

## Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--czi-path` | `/path/to/slide.czi` | Input CZI file |
| `--output-dir` | `/path/to/output` | Output directory |
| `--cell-type` | `nmj` | Detection type |
| `--channel` | `1` | BTX channel (NMJ marker) |
| `--intensity-percentile` | `97` | Threshold percentile for detection |
| `--sample-fraction` | `1.0` | Process 100% of tiles |
| `--all-channels` | flag | Load all 3 channels for multi-channel features |
| `--load-to-ram` | flag | Load channels to RAM (faster for network mounts) |
| `--extract-full-features` | flag | Extract all features (morph + SAM2) |
| `--skip-deep-features` | flag | Skip ResNet/DINOv2 (use morph+SAM2 only) |
| `--tile-overlap` | `0.1` | 10% tile overlap to catch boundary NMJs |
| `--nmj-classifier` | `checkpoints/...` | Pre-trained classifier |
| `--no-serve` | flag | Don't start HTTP server after completion |

## Channel Mapping

For 3-channel NMJ slides:
- **Channel 0**: Nuclear (488nm)
- **Channel 1**: BTX (647nm) - NMJ marker, used for detection
- **Channel 2**: NFL (750nm) - Neurofilament

## Classifier Details

The included classifier (`nmj_classifier_morph_sam2.joblib`) uses:
- **78 morphological features**: Per-channel intensity stats, inter-channel ratios
- **256 SAM2 embedding features**: Semantic image features
- **Total: 334 features**

Performance:
- Precision: **0.952** (optimized for minimal false positives)
- Recall: 0.840
- F1: 0.891

## Output Structure

```
output_dir/
├── run.log                              # Pipeline log
├── slide_name/
│   └── tiles/
│       └── tile_X_Y/
│           ├── nmj_features.json        # Detections + features
│           └── nmj_masks.h5             # Segmentation masks
├── nmj_detections.json                  # All detections merged
├── nmj_coordinates.csv                  # Quick coordinate export
└── html/                                # Annotation viewer
    ├── index.html
    └── nmj_page_*.html
```

## Expected Runtime

| Slide Size | Tiles | Approx. Time |
|------------|-------|--------------|
| 263k x 89k px | ~1800 | ~60-80 hours |

Processing is ~120-150 sec/tile due to:
- SAM2 embedding computation per tile
- Multi-channel feature extraction
- Morphological filtering

## Memory Requirements

- **RAM**: ~140 GB (3 channels × 44 GB each)
- **GPU VRAM**: ~2-3 GB (SAM2 inference)

## Post-Processing Notes

With `--tile-overlap 0.1`, NMJs at tile boundaries may be detected twice.
Deduplication should be applied in post-processing before LMD export.

## Troubleshooting

### OOM Errors
```bash
# Use sequential processing (slower but uses less memory)
--sequential
```

### Slow Network Mount
```bash
# Always use --load-to-ram for network-mounted CZI files
--load-to-ram
```

### Check Progress
```bash
tail -f output_dir/run.log
```

### Monitor GPU
```bash
nvidia-smi -l 1
```
