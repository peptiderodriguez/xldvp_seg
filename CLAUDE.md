# xldvp_seg - Image Analysis Pipelines

## Quick Start (Claude Code)

Type `/analyze` to begin. Claude will detect your system, inspect your data,
and guide you through the full pipeline — detection through LMD export.

| Command | What it does |
|---------|-------------|
| `/analyze` | Full pipeline: detect, annotate, classify, spatial analysis, LMD export |
| `/status` | Check running SLURM jobs, tail logs, monitor progress |
| `/czi-info` | Inspect CZI metadata — channels, dimensions, pixel size |
| `/preview-preprocessing` | Preview flat-field / photobleach correction on any channel |
| `/classify` | Train RF classifier from annotations, compare feature sets, explore features |
| `/lmd-export` | Export detections for laser microdissection (contours, wells, XML) |
| `/view-results` | Launch HTML result viewer with Cloudflare tunnel |

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

---

## Common Commands

### Unified Segmentation
```bash
# NMJ detection (10% annotation run — shows ALL candidates)
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel 1 \
    --sample-fraction 0.10

# NMJ with classifier (100% run — shows rf_prediction >= 0.5)
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel 1 \
    --nmj-classifier path/to/rf_classifier.pkl \
    --prior-annotations path/to/round1_annotations.json

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
--load-to-ram       # Load all channels into RAM (default, faster for network mounts)
--sequential        # Process one tile at a time (safer memory usage)
--num-gpus 4        # Number of GPUs (always multi-GPU, even with 1)
```

### Multi-Node Sharding
```bash
# Split detection across 4 nodes (each processes 1/4 of tiles)
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 0/4 --resume /shared/output/dir  # node 0
    --tile-shard 1/4 --resume /shared/output/dir  # node 1
    # ... etc

# Merge all shards (auto-detects shard manifests, checkpointed)
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --resume /shared/output/dir --merge-shards
# Checkpoints: merged_detections.json → detections.json → HTML
```

### View Results
```bash
python -m http.server 8080 --directory /home/dude/mk_output/project/html
~/cloudflared tunnel --url http://localhost:8080
```

---

## NMJ Pipeline (Detect Once, Classify Later)

1. Import full CZI into RAM (`--load-to-ram`, default)
2. Tile with 10% overlap (`--tile-overlap 0.10`)
3. Detect 100% of tiles (or multi-node with `--tile-shard`)
4. Segment: 98th percentile intensity threshold + morphology + watershed
5. Extract features: morph + SAM2 (always), ResNet + DINOv2 (opt-in `--extract-deep-features`)
6. Deduplicate overlapping masks (>10% pixel overlap) — or `--merge-shards` for multi-node
7. Generate annotation HTML (subsample 1500 via `scripts/regenerate_html.py --max-samples 1500`)
8. Train RF classifier with annotations (`train_classifier.py`, balanced classes)
9. Score ALL detections: `scripts/apply_classifier.py` (CPU, seconds — no re-detection)
10. Generate filtered HTML: `scripts/regenerate_html.py --score-threshold 0.5`
12. Dilate +0.5 um, RDP simplify (epsilon=5 px)
13. Two-stage clustering: Round 1 = 500 um, Round 2 = 1000 um, target 375-425 um²
14. Unclustered = singles
15. Controls: 100 um offset, 8 directions, cluster controls preserve arrangement
16. Napari visualization (4 colors: singles/controls/clusters/cluster-controls)
17. 384-well plate serpentine B2 → B3 → C3 → C2 (max 308 wells)
18. OME-Zarr pyramid for Napari viewing
19. XML export with reference crosses

### Channel Mapping (3-channel slides)
- R (ch0): Nuclear (488nm)
- G (ch1): BTX (647nm) — NMJ marker (detection channel)
- B (ch2): NFL (750nm)

### Feature Hierarchy
| Feature Set | Dimensions | When |
|-------------|-----------|------|
| Morphological | ~78 | Always |
| Per-channel stats | ~50 | When `--all-channels` + 2+ channels |
| SAM2 embeddings | 256 | Always (default) |
| ResNet50 (masked + context) | 4,096 | `--extract-deep-features` |
| DINOv2-L (masked + context) | 2,048 | `--extract-deep-features` |

**Feature comparison (3-channel, 844 annotated detections):**

| Feature Set | n | F1 |
|-------------|---|-----|
| **all_features** | 6478 | **0.909** |
| morph+dinov2_combined | 2126 | 0.901 |
| morphological | 78 | 0.900 |
| dinov2_context | 1024 | 0.843 |
| resnet_context | 2048 | 0.836 |
| sam2 | 256 | 0.728 |

Use `train_classifier.py --feature-set` to compare subsets and pick the best for your final RF classifier. Morph-only (78 features) is nearly as good as all 6478 combined.

---

## Coordinate System

**All coordinates are [x, y] (horizontal, vertical).**

UID format: `{slide}_{celltype}_{x}_{y}`

**Mosaic origin:** CZI tiles use global coordinates. RAM arrays are 0-indexed.
- `loader.get_tile()` handles the offset correctly
- Direct `all_channel_data` indexing must subtract `x_start, y_start`

---

## Entry Points

| Script | Use Case |
|--------|----------|
| `run_segmentation.py` | Unified entry point: detect, dedup, HTML, CSV (recommended) |
| `run_lmd_export.py` | Export to Leica LMD format (any cell type) |
| `train_classifier.py` | Train RF classifier from annotated detections |
| `scripts/apply_classifier.py` | Score existing detections with trained classifier (no re-detection) |
| `scripts/regenerate_html.py` | Regenerate HTML viewer from saved detections (all cell types) |
| `scripts/czi_to_ome_zarr.py` | Convert CZI to OME-Zarr with pyramids |
| `scripts/napari_place_crosses.py` | Interactive reference cross placement |
| `scripts/cluster_detections.py` | Biological clustering for LMD well assignment |
| `scripts/napari_view_lmd_export.py` | View LMD export overlaid on slide |

---

## Module Reference

### Detection Strategies (all support MultiChannelFeatureMixin)
| Strategy | File |
|----------|------|
| NMJ | `segmentation/detection/strategies/nmj.py` |
| MK | `segmentation/detection/strategies/mk.py` |
| Cell | `segmentation/detection/strategies/cell.py` |
| Vessel | `segmentation/detection/strategies/vessel.py` |

### Multi-GPU Processing (always used, even with --num-gpus 1)
| Module | Purpose |
|--------|---------|
| `segmentation/processing/multigpu_worker.py` | Generic GPU worker (all cell types) |
| `segmentation/processing/multigpu_shm.py` | Shared memory manager (SIGTERM cleanup) |
| `segmentation/processing/tile_processing.py` | Shared `process_single_tile()` |

### LMD Export
| Module | Purpose |
|--------|---------|
| `run_lmd_export.py` | Unified pipeline: contours, controls, wells, XML |
| `segmentation/lmd/clustering.py` | Two-stage greedy clustering |
| `segmentation/lmd/contour_processing.py` | Dilation + RDP simplification |

### Normalization
| Module | Purpose |
|--------|---------|
| `compute_normalization_params.py` | Phase 1: compute cross-slide LAB stats |
| `segmentation/preprocessing/stain_normalization.py` | Phase 2: apply Reinhard per-slide |

Tissue detection in normalization uses calibrated threshold / 10 for permissive detection.

---

## Vessel Pipeline

### Detection Method
1. Adaptive + Otsu thresholding for SMA+ regions
2. Contour hierarchy analysis (find rings)
3. Ellipse fitting, wall thickness at 36 angles
4. **3-contour system**: lumen (cyan), CD31 outer (green), SMA ring (magenta)
5. Adaptive per-pixel dilation on CD31 and SMA channels (irregular contours following signal)
6. Full-resolution contour refinement (reads CZI ROI at native res per vessel)
7. Full-resolution crop rendering for HTML

### 3-Contour Ring Detection
- **Lumen** (cyan): Inner boundary from SAM2/threshold segmentation
- **CD31** (green): Endothelial outer boundary via adaptive dilation on CD31 channel
- **SMA** (magenta): Smooth muscle ring via adaptive dilation on SMA channel, expanding from lumen (not from CD31 boundary, since CD31 and SMA intermingle)
- `has_sma_ring`: True when SMA expansion > 5% larger than lumen. Veins/capillaries lack SMA, so SMA contour collapses to lumen boundary.

### CLI Options (Vessel-specific)
```bash
--no-refine             # Skip full-res refinement (use tile-scale contours)
--spline                # Enable spline smoothing (off by default)
--spline-smoothing 3.0  # Spline smoothing factor
--dilation-mode adaptive|uniform  # Dilation mode for CD31+SMA (default: adaptive)
--min-sma-intensity 30  # Min SMA signal to attempt ring detection
```

### Multi-Marker (6-type classification)
artery, arteriole, vein, capillary, lymphatic, collecting_lymphatic

---

## Hardware (SLURM Cluster)
- **p.hpcl8:** 55 nodes, 24 CPUs, 380G RAM, 2x RTX 5000 each (interactive dev, CPU jobs)
- **p.hpcl93:** 19 nodes, 256 CPUs, 760G RAM, 4x L40S each (heavy GPU batch jobs, requires `--gres=gpu:`)
- Time limit: 42 days on both partitions

## Troubleshooting

### OOM: reduce `--num-gpus`, reduce tile size
### CUDA Boolean: `mask = mask.astype(bool)` for SAM2
### SAM2 _orig_hw: `img_h, img_w = sam2_predictor._orig_hw[0]` (list of tuple)
### HDF5 LZ4: `import hdf5plugin` before `h5py`
### Network Mounts: Socket timeout 60s automatic. Check with `ls /mnt/x/`

---

## OME-Zarr / LMD Export Workflow

```bash
# 1. Convert CZI to OME-Zarr
python scripts/czi_to_ome_zarr.py slide.czi slide.zarr

# 2. Place reference crosses in Napari
python scripts/napari_place_crosses.py slide.zarr --output crosses.json

# 3. Export to LMD (with controls)
python run_lmd_export.py \
    --detections detections.json \
    --crosses crosses.json \
    --output-dir lmd_export \
    --generate-controls \
    --export
```

Max 308 wells per plate. Early capacity check warns before expensive processing.

---

## Installation

```bash
conda create -n mkseg python=3.11 -y && conda activate mkseg
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
./install.sh  # Auto-detects CUDA
```
