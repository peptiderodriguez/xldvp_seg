# xldvp_seg - Project Reference

## Quick Start

**Pipelines:**
1. **MK/HSPC** - Bone marrow cell segmentation (Megakaryocytes + Stem Cells)
2. **NMJ** - Neuromuscular junction detection in muscle tissue
3. **Vessel** - Blood vessel morphometry (SMA+ ring detection)
4. **Mesothelium** - Mesothelial ribbon detection for laser microdissection

**Documentation:** See `docs/GETTING_STARTED.md` and `docs/LMD_EXPORT_GUIDE.md`

### Key Locations
| What | Where |
|------|-------|
| This repo | `/home/dude/code/xldvp_seg_repo/` |
| MK/HSPC output | `/home/dude/mk_output/` |
| NMJ output | `/home/dude/nmj_output/` |
| Vessel output | `/home/dude/vessel_output/` |
| Conda env | `mkseg` |

**Activate:** `source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg`

### Output Structure
```
/home/dude/{celltype}_output/{project_name}/
├── html/                        # Annotation viewer
├── {celltype}_detections.json   # All detections with UIDs
├── {celltype}_coordinates.csv   # Quick coordinate export
└── tiles/{tile_id}/
    ├── segmentation.h5
    ├── features.json
    └── window.csv
```

---

## Common Commands

```bash
# NMJ detection
python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj --channel 1 --sample-fraction 0.10

# MK detection
python run_segmentation.py --czi-path /path/to/slide.czi --cell-type mk --channel 0

# Vessel detection (basic)
python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 0 --sample-fraction 0.10

# Vessel with multi-scale (for large vessels)
python run_segmentation.py --czi-path /path/to/slide.czi --cell-type vessel --channel 2 \
    --multi-scale --scales "8,4,1" --load-to-ram

# Multi-channel with all features
python run_segmentation.py --czi-path /path/to/slide.czi --cell-type nmj \
    --channel 1 --all-channels --load-to-ram
```

**Key flags:**
- `--load-to-ram` - Load CZI to RAM (faster for network mounts)
- `--sequential` - Process one tile at a time (safer memory)
- `--candidate-mode` - Relaxed thresholds for high recall (vessels)
- `--multi-scale` - Multi-resolution detection (vessels)
- `--all-channels` - Extract features from all channels

---

## Hardware & Processing

- **CPU:** 48 cores | **RAM:** 432 GB | **GPU:** RTX 4090 (24 GB)
- **Default tile size:** 4000x4000 pixels
- **Default workers:** 4 (auto-adjusts based on RAM)

**Multi-GPU (Slurm):**
```bash
ls /path/to/slides/*.czi > input_files.txt
sbatch slurm/run_multigpu.sbatch input_files.txt /path/to/output
```

---

## Coordinate System

**All coordinates stored as [x, y] (horizontal, vertical).**

**UID format:** `{slide}_{celltype}_{round(x)}_{round(y)}`
- Example: `2025_11_18_FGC1_mk_12346_67890`

Utilities in `segmentation.processing.coordinates`: `generate_uid()`, `parse_uid()`, `validate_xy_coordinates()`

---

## Pipeline Details

### MK/HSPC Pipeline
- **Models:** SAM2 (mask proposals) + Cellpose (HSPC nuclei) + ResNet50 (MK classification)
- **Checkpoints:** `checkpoints/sam2.1_hiera_large.pt`, `checkpoints/best_model.pth`
- **MK filter:** 200-2000 µm² (configurable via `--mk-min-area-um`, `--mk-max-area-um`)

### NMJ Pipeline
- **Detection:** BTX channel thresholding → morphological cleanup → solidity filter (≤0.85) → watershed expansion
- **Classifiers:** ResNet18 (96.6% accuracy) or Random Forest on 2,382 features
- **Training:** `python train_nmj_classifier.py --detections nmj_detections.json --annotations annotations.json`

### Vessel Pipeline

**SAM2-based detection (recommended):** `scripts/sam2_multiscale_vessels.py`
- Loads channels at 1/2 scale into RAM (~45 GB for 4 channels)
- Processes at 1/8 scale with 4000px tiles (covers 32000px at full res)
- SAM2 generates mask proposals → filter to lumens → watershed expand to walls
- No diameter filters during detection - filter in post-processing
- Output: `vessel_detections_multiscale.json`, `crops/`, `index.html`

```bash
# SAM2 vessel detection
python scripts/sam2_multiscale_vessels.py
# Edit CZI_PATH and OUTPUT_DIR in script before running
```

**Legacy contour-based detection:** `run_segmentation.py --cell-type vessel`
- Contour hierarchy analysis for ring structures (SMA+ wall around dark lumen)
- Classification: 3-stage pipeline (candidate → vessel detector RF → artery/vein classifier)
- 6 vessel types: artery, arteriole, vein, capillary, lymphatic, collecting_lymphatic

**Vessel parameters (legacy):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-vessel-diameter` | 10 µm | Minimum outer diameter |
| `--max-vessel-diameter` | 1000 µm | Maximum outer diameter |
| `--min-circularity` | 0.3 | Minimum circularity |
| `--min-ring-completeness` | 0.5 | Minimum SMA+ perimeter fraction |

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `segmentation/models/manager.py` | Centralized model loading (SAM2, Cellpose, ResNet) |
| `segmentation/processing/memory.py` | Memory validation, safe worker counts |
| `segmentation/detection/registry.py` | Strategy registry for cell types |
| `segmentation/utils/vessel_features.py` | 32 vessel-specific features |
| `segmentation/classification/` | Vessel detector RF, artery/vein classifier |
| `segmentation/reporting/` | PDF/HTML reports with Plotly visualizations |
| `segmentation/io/html_export.py` | HTML annotation viewer generation |

---

## Troubleshooting

### OOM / System Crashes
1. Use `--sequential` flag
2. Reduce `--num-workers` to 2 or 1
3. Reduce `--tile-size` to 3000
4. Memory validation runs at startup and warns if insufficient

### Network Mount Hangs
- Socket timeout set to 60s automatically
- Check connectivity: `ls /mnt/x/`

### HDF5 Errors
```bash
export HDF5_PLUGIN_PATH=""
export HDF5_USE_FILE_LOCKING=FALSE
```

### CUDA Boolean Type Error
```python
mask = mask.astype(bool)  # Fix for SAM2 masks
```

### Monitoring
```bash
tail -f /home/dude/mk_output/*/run.log  # Watch log
nvidia-smi -l 1                          # GPU
watch -n 5 free -h                       # RAM
```

---

## Known Issues

**Multi-marker mode (`--multi-marker`)** may crash silently during tile processing. Workaround: use single-channel mode.

**Partial vessel detection** (cross-tile merging): Building blocks exist but orchestration not wired up. `merge_across_tiles()` defined but never called.

---

## External Access

**Cloudflare Tunnel:** `~/cloudflared tunnel --url http://localhost:8080`
- Port 8080 for MK, 8081 for NMJ
