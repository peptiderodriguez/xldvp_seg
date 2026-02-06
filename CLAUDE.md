# xldvp_seg - Image Analysis Pipelines

## Quick Start

**Pipelines available:**
1. **MK/HSPC** - Bone marrow cell segmentation (Megakaryocytes + Stem Cells)
2. **NMJ** - Neuromuscular junction detection in muscle tissue
3. **Vessel** - Blood vessel morphometry (SMA+ ring detection)
4. **Mesothelium** - Mesothelial ribbon detection for laser microdissection

### Documentation
- **[GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Full user guide
- **[NMJ_PIPELINE_GUIDE.md](docs/NMJ_PIPELINE_GUIDE.md)** - NMJ detection with classifier
- **[NMJ_LMD_EXPORT_WORKFLOW.md](docs/NMJ_LMD_EXPORT_WORKFLOW.md)** - Full LMD export workflow
- **[LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md)** - Basic LMD export
- **[COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md)** - Coordinate conventions

### Key Locations
| What | Where |
|------|-------|
| This repo | `/home/dude/code/xldvp_seg_repo/` |
| MK/HSPC output | `/home/dude/mk_output/` |
| NMJ output | `/home/dude/nmj_output/` |
| Vessel output | `/home/dude/vessel_output/` |
| Conda env | `mkseg` |

**Activate environment:**
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg
```

### Output Structure
```
/home/dude/{celltype}_output/{project_name}/
├── html/                        # Annotation viewer
│   ├── index.html
│   └── {celltype}_page_*.html
├── {celltype}_detections.json   # All detections with UIDs
├── {celltype}_coordinates.csv   # Quick coordinate export
└── tiles/{tile_id}/
    ├── segmentation.h5
    ├── features.json
    └── window.csv
```

---

## Common Commands

### Unified Segmentation
```bash
# NMJ detection
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

# Vessel detection
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 0 \
    --candidate-mode
```

### Performance Options
```bash
--load-to-ram       # Load all channels into RAM (faster for network mounts)
--sequential        # Process one tile at a time (safer memory usage)
--multi-gpu         # Enable multi-GPU mode (cluster)
--num-gpus 4        # Number of GPUs
```

### View Results
```bash
# Start HTTP server
python -m http.server 8080 --directory /home/dude/mk_output/project/html

# Cloudflare tunnel (no bandwidth limits)
~/cloudflared tunnel --url http://localhost:8080
```

---

## Hardware
- **CPU:** 48 cores
- **RAM:** 432 GB
- **GPU:** NVIDIA RTX 4090 (24 GB VRAM)
- **Storage:** Network mount at `/mnt/x/`

---

## Entry Points

| Script | Use Case |
|--------|----------|
| `run_segmentation.py` | Unified entry point (recommended) |
| `run_lmd_export.py` | Export to Leica LMD format |
| `run_unified_FAST.py` | Legacy MK/HSPC batch pipeline |
| `run_nmj_inference.py` | NMJ classification with trained model |
| `scripts/czi_to_ome_zarr.py` | Convert CZI to OME-Zarr with pyramids |
| `scripts/napari_place_crosses.py` | Interactive reference cross placement |
| `scripts/contour_processing.py` | Contour dilation and RDP simplification |
| `scripts/napari_view_lmd_export.py` | View LMD export overlaid on slide |

---

## Coordinate System

**All coordinates are [x, y] (horizontal, vertical).**

UID format: `{slide}_{celltype}_{x}_{y}`
- Example: `2025_11_18_FGC1_mk_12346_67890`

Utilities in `segmentation.processing.coordinates`:
- `generate_uid()`, `parse_uid()`, `validate_xy_coordinates()`

---

## MK/HSPC Pipeline

### Models
- **SAM2:** Mask proposals (`checkpoints/sam2.1_hiera_large.pt`)
- **Cellpose:** HSPC detection (cyto3 model)
- **ResNet50:** MK classification (`checkpoints/best_model.pth`)

### Parameters
```bash
--tile-size 3000              # Tile dimensions
--sample-fraction 0.10        # Fraction of tiles to process
--mk-min-area-um 200          # MK size filter (µm²)
--mk-max-area-um 2000
--samples-per-page 300        # HTML pagination
```

---

## NMJ Pipeline

### Detection Method
1. Intensity thresholding (BTX channel only)
2. Morphological cleanup
3. Solidity filtering (max 0.85 for branched shapes)
4. Watershed expansion
5. Optional RF classifier

### Multi-Channel Features (with `--all-channels`)

All feature extractors use true 3-channel RGB input:
- **Morphological (22)**: Per-channel intensity stats (red/green/blue mean/std), HSV, shape
- **SAM2 embeddings (256)**: Extracted from 3-channel tile
- **ResNet50 (4,096)**: Masked + context, both from 3-channel crops
- **DINOv2-L (2,048)**: Masked + context, both from 3-channel crops

**Total: ~6,400 deep features per detection**

### Channel Mapping (3-channel slides)
- R (ch0): Nuclear (488nm)
- G (ch1): BTX (647nm) - NMJ marker (used for mask detection)
- B (ch2): NFL (750nm)

Note: Mask detection uses BTX channel only (intensity thresholding), but all feature extraction uses the full 3-channel RGB.

### Classifiers
| Type | File | Accuracy |
|------|------|----------|
| ResNet18 | `nmj_classifier.pth` | 96.6% |
| Random Forest | `nmj_classifier_rf.pkl` | 91% |

```bash
# Train classifier
python train_nmj_classifier_features.py \
    --detections nmj_detections.json \
    --annotations annotations.json \
    --output-dir /path/to/output
```

### Feature Extraction Options

**ResNet50 features (2048D):**
- `resnet_0` to `resnet_2047`: Masked features (background zeroed out)
- `resnet_ctx_0` to `resnet_ctx_2047`: Context features (full tissue crop)

Use `--all-channels` flag to extract features from true 3-channel RGB instead of single-channel BTX stacked 3x.

**Feature comparison results (3-channel, 844 annotated detections):**

| Feature Set | n | Accuracy | Precision | Recall | F1 |
|-------------|---|----------|-----------|--------|-----|
| **all_features** | 6478 | **0.937** | 0.925 | **0.897** | **0.909** |
| morph+dinov2_combined | 2126 | 0.931 | 0.922 | 0.883 | 0.901 |
| morphological | 78 | 0.931 | 0.937 | 0.867 | 0.900 |
| dinov2_context | 1024 | 0.891 | 0.864 | 0.827 | 0.843 |
| resnet_context | 2048 | 0.883 | 0.834 | 0.840 | 0.836 |
| resnet_masked | 2048 | 0.838 | 0.799 | 0.727 | 0.761 |
| sam2 | 256 | 0.834 | 0.868 | 0.630 | 0.728 |
| dinov2_masked | 1024 | 0.815 | 0.796 | 0.650 | 0.714 |

**Key findings:**
- Context features outperform masked features for both ResNet and DINOv2
- Morphological features alone achieve F1=0.900 (very discriminative)
- Best efficiency: morph+dinov2_combined (F1=0.901 with only 2126 features)
- All features combined gives best overall performance (F1=0.909)

**DINOv2 models:**

| Model | Params | Feature Dim |
|-------|--------|-------------|
| dinov2_vits14 | 21M | 384 |
| dinov2_vitb14 | 86M | 768 |
| dinov2_vitl14 | 300M | 1024 ← **used in pipeline** |

**Full feature set per detection (with --all-channels):**
| Feature | Dimension | Description |
|---------|-----------|-------------|
| `resnet_0-2047` | 2048 | ResNet50 masked (bg zeroed) |
| `resnet_ctx_0-2047` | 2048 | ResNet50 context (full tissue) |
| `dinov2_0-1023` | 1024 | DINOv2-L masked |
| `dinov2_ctx_0-1023` | 1024 | DINOv2-L context |
| `sam2_emb_0-255` | 256 | SAM2 spatial embeddings |
| morphological | ~78 | Area, solidity, channel stats, etc. |

**Total: ~6,478 features per detection**

**DINOv2 vs ResNet (v1 single-channel results):**
| Method | Accuracy | Recall+ | F1+ |
|--------|----------|---------|-----|
| ResNet50 masked | 0.717 | 0.153 | 0.253 |
| DINOv2-S context | 0.674 | 0.208 | 0.286 |
| DINOv2-S combined | 0.700 | 0.208 | **0.303** |

Key finding: DINOv2 context > DINOv2 masked (self-supervised models need full context)

---

## Vessel Pipeline

### Detection Method
1. Adaptive + Otsu thresholding for SMA+ regions
2. Contour hierarchy analysis (find rings with inner/outer contours)
3. Ellipse fitting for measurements
4. Wall thickness calculation at 36 angles

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-vessel-diameter` | 10 µm | Minimum outer diameter |
| `--max-vessel-diameter` | 1000 µm | Maximum outer diameter |
| `--min-wall-thickness` | 2 µm | Minimum wall thickness |
| `--max-aspect-ratio` | 4.0 | Exclude longitudinal sections |
| `--min-circularity` | 0.3 | Minimum circularity |
| `--candidate-mode` | False | Relaxed thresholds for high recall |

### Multi-Marker Detection
```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 1 \
    --multi-marker \
    --channel-names "nuclear,sma,cd31,lyve1"
```

**6-Type Classification:**
| Type | Markers |
|------|---------|
| artery | SMA+, CD31- |
| arteriole | SMA+, CD31- (smaller) |
| vein | SMA+/weak, CD31+ |
| capillary | SMA-, CD31+ |
| lymphatic | SMA-, LYVE1+ |
| collecting_lymphatic | SMA+, LYVE1+ |

### Vessel-Specific Features (32)
Ring/wall metrics, shape descriptors, size measurements, intensity features, log-transformed variants.

### Training Workflow
```bash
# 1. Run candidate detection
python run_segmentation.py --cell-type vessel --candidate-mode ...

# 2. Annotate in HTML viewer, export annotations

# 3. Train detector
python scripts/train_vessel_detector.py \
    --annotations annotations.json \
    --detections vessel_detections.json \
    --stratify-by-size

# 4. Run full pipeline
python scripts/run_full_pipeline.py \
    --input candidates.json \
    --vessel-detector vessel_detector.joblib
```

---

## Module Reference

### Core Modules
| Module | Purpose |
|--------|---------|
| `segmentation/models/manager.py` | Model loading (SAM2, Cellpose, ResNet) |
| `segmentation/processing/memory.py` | Memory validation, worker count |
| `segmentation/detection/registry.py` | Strategy registry for cell types |
| `segmentation/io/html_export.py` | HTML annotation viewer generation |
| `segmentation/utils/config.py` | Configuration constants |

### Detection Strategies
| Strategy | File |
|----------|------|
| NMJ | `segmentation/detection/strategies/nmj.py` |
| MK | `segmentation/detection/strategies/mk.py` |
| Vessel | `segmentation/detection/strategies/vessel.py` |

### Classification
| Classifier | File |
|------------|------|
| Vessel detector (binary) | `segmentation/classification/vessel_detector_rf.py` |
| Vessel type (6-class) | `segmentation/classification/vessel_type_classifier.py` |
| Artery/vein | `segmentation/classification/artery_vein_classifier.py` |

### Reporting
```python
from segmentation.reporting import VesselReport, BatchVesselReport

report = VesselReport.from_json("vessel_detections.json")
report.generate_html("report.html")
```

---

## Troubleshooting

### OOM / System Crashes
1. Use `--sequential` flag
2. Reduce `--num-workers` to 2 or 1
3. Reduce `--tile-size` to 3000

### CUDA Boolean Error
```python
mask = mask.astype(bool)  # Fix for SAM2 masks
```

### SAM2 Embedding Extraction
The SAM2 predictor's `_orig_hw` attribute returns a list containing a tuple, not just a tuple:
```python
# Wrong (causes "not enough values to unpack" error):
img_h, img_w = sam2_predictor._orig_hw

# Correct:
img_h, img_w = sam2_predictor._orig_hw[0]
```

### Feature Extraction Best Practices
When extracting deep features (ResNet, DINOv2) for detections:
- **Use actual detection masks** for cropping, not area-based approximations
- **Use actual bounding boxes** from mask coordinates (`ys.min(), ys.max()`)
- **Never use fixed crop sizes** - each detection should have a unique crop
- The pipeline in `nmj.py` does this correctly; avoid writing standalone extraction scripts

### HDF5 Errors
```bash
export HDF5_PLUGIN_PATH=""
export HDF5_USE_FILE_LOCKING=FALSE
```

### Network Mount Hangs
Socket timeout is set to 60s automatically. Check connectivity with `ls /mnt/x/`.

### Monitoring
```bash
tail -f /home/dude/mk_output/*/run.log  # Watch log
nvidia-smi -l 1                          # GPU usage
watch -n 5 free -h                       # RAM usage
```

---

## Installation

```bash
# Create environment
conda create -n mkseg python=3.11 -y
conda activate mkseg

# Install
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
./install.sh  # Auto-detects CUDA

# Options:
./install.sh --cuda 11.8    # Force CUDA version
./install.sh --rocm         # AMD GPUs
./install.sh --cpu          # CPU only
```

---

## Known Issues

### Multi-Marker Mode Crash
`--multi-marker` may crash silently during tile processing. Workaround: use single-channel mode.

### Partial Vessel Detection
Cross-tile vessel merging code exists but is not wired up. `merge_across_tiles()` is never called.

---

## OME-Zarr / Napari Workflow

For large slides, convert CZI to OME-Zarr with pyramids for fast Napari viewing.

### Convert CZI to OME-Zarr
```bash
# Dry run (validate CZI structure)
python scripts/czi_to_ome_zarr.py input.czi output.zarr --dry-run

# Convert with conservative settings
python scripts/czi_to_ome_zarr.py input.czi output.zarr \
    --channel-names "nuc488" "Bgtx647" "NfL750" \
    --overwrite

# Fast mode (use more resources)
python scripts/czi_to_ome_zarr.py input.czi output.zarr \
    --strip-height 20000 \
    --workers 8 \
    --chunk-size 2048 \
    --overwrite
```

### Place Reference Crosses in Napari
```bash
python scripts/napari_place_crosses.py /path/to/image.zarr \
    --output reference_crosses.json \
    --detections nmj_detections.json  # Optional: show detections as overlay
```

**Keyboard shortcuts:**
- `S` - Save crosses to JSON
- `U` - Undo last cross
- `C` - Clear all crosses
- `Q` - Quit and save (min 3 crosses required)

### Full LMD Export Workflow
```bash
# 1. Convert CZI to OME-Zarr
python scripts/czi_to_ome_zarr.py slide.czi slide.zarr

# 2. Place reference crosses in Napari
python scripts/napari_place_crosses.py slide.zarr --output crosses.json

# 3. Export to LMD
python run_lmd_export.py \
    --detections detections.json \
    --crosses crosses.json \
    --output-dir lmd_export \
    --export
```

### Spatial Controls for LMD Export

Generate paired control regions for each NMJ target (offset 150 µm, same shape):

**Key functions in `run_lmd_export.py`:**
- `generate_spatial_control()` - Shifts NMJ contour by offset, tries 8 directions to avoid collisions
- `generate_wells_serpentine_4_quadrants()` - 384-well serpentine order across 4 quadrants (B2, C2, B3, C3)
- `check_polygon_overlap()` - Shapely-based collision detection

**Well assignment pattern:**
- Alternating: NMJ → Control → NMJ → Control
- Singles processed first, then clusters
- Within each group: nearest-neighbor ordering to minimize slide movement
- Cluster transition starts from nearest cluster to last single's control

**Output files:**
- `lmd_export_with_controls.json` - All shapes with well assignments
- `shapes_with_controls.xml` - LMD XML for Leica microscope

**Napari visualization:**
```bash
# View export overlaid on OME-Zarr pyramid
python scripts/napari_view_lmd_export.py --zarr slide.zarr --export lmd_export_with_controls.json
```
- Green: Singles (NMJs)
- Cyan: Single controls
- Red: Clusters (NMJs)
- Orange: Cluster controls
