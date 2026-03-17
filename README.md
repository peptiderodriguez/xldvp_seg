# xldvp_seg — Image Analysis & DVP Pipeline

Automated cell detection, annotation, classification, spatial analysis, and LMD export for whole-slide CZI microscopy images. Designed for Deep Visual Proteomics (DVP) workflows where laser-microdissected cells go into mass spec analysis.

Runs on SLURM GPU clusters or local workstations (CUDA, Apple Silicon MPS, or CPU). Claude Code provides an interactive AI assistant that guides you through the entire pipeline.

## Supported Cell Types

| Type | Detection Method | Use Case |
|------|-----------------|----------|
| **Cell** | Cellpose 2-channel (cyto+nuc) + SAM2 embeddings | Generic cell detection (e.g. NeuN+nuc, senescence) |
| **NMJ** | 98th percentile threshold + morphology + watershed | Neuromuscular junction detection |
| **MK** | SAM2 auto-mask + size filter | Megakaryocyte detection |
| **Vessel** | SMA+ ring detection + 3-contour hierarchy | Blood vessel morphometry |
| **Islet** | Cellpose membrane+nuclear + marker classification | Pancreatic islet cells |
| **Tissue Pattern** | Cellpose + spatial frequency analysis | Brain FISH cell typing |
| **Mesothelium** | Ridge detection | Mesothelial ribbon for LMD |

---

## Prerequisites

- **Conda** or **Miniforge** ([install miniforge](https://github.com/conda-forge/miniforge#miniforge3))
- **GPU** (recommended): NVIDIA with CUDA 11.8+, or Apple Silicon. CPU-only mode available.
- **Node.js 18+** (only for Claude Code): `conda install -c conda-forge nodejs` or [nodejs.org](https://nodejs.org/)

---

## Installation

### Step 1: Clone and create environment

```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
conda create -n mkseg python=3.11 -y && conda activate mkseg
```

### Step 2: Install everything

```bash
./install.sh  # Auto-detects CUDA, installs all dependencies + downloads models
```

This single command installs:
- **PyTorch** with CUDA support (auto-detected)
- **SAM2** (Segment Anything Model 2) from Facebook Research
- **SAM2 checkpoint** (~890 MB, downloaded to `checkpoints/`)
- **Cellpose**, scikit-learn, scipy, anndata, squidpy, spatialdata, geopandas, etc.
- **orjson** + **napari** (optional but recommended)
- **cloudflared** (for remote HTML viewing on SLURM clusters)

**Options:**
```bash
./install.sh --cuda 12.4    # Specify CUDA version (11.8, 12.1, 12.4)
./install.sh --cpu           # CPU-only (no GPU)
./install.sh --rocm          # AMD GPU (ROCm)
./install.sh --dev           # Include dev tools (pytest, black, ruff)
```

### Step 3: Verify

```bash
conda activate mkseg
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cellpose; print(f'Cellpose {cellpose.__version__}')"
python -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"
```

### Step 4 (recommended): Install Claude Code

Claude Code is an AI CLI that knows this entire codebase. It detects your system, inspects your CZI data, builds YAML configs, launches SLURM jobs, diagnoses failures, and guides you through annotation, classification, and export interactively.

```bash
# Requires Node.js 18+ (conda install -c conda-forge nodejs, or nodejs.org)
npm install -g @anthropic-ai/claude-code

# Start in the repo directory
cd xldvp_seg
claude
```

On first launch, Claude reads `CLAUDE.md` and `.claude/commands/` to understand the pipeline. No additional setup needed.

---

## Getting Started

### With Claude Code (recommended)

Inside Claude Code, type:

```
/analyze
```

Claude will:
1. Detect your system (local GPU, SLURM cluster, available partitions)
2. Ask for your CZI file(s) and inspect channel metadata
3. Build the channel map, confirm with you, and configure detection
4. Write a YAML config and launch the pipeline (SLURM or local)
5. Monitor progress, diagnose failures, guide annotation/classification/export

Other commands:

| Command | What it does |
|---------|-------------|
| `/analyze` | Full pipeline: detect → annotate → classify → spatial → LMD export |
| `/status` | Check SLURM jobs, tail logs, monitor progress |
| `/czi-info` | Inspect CZI metadata — channels, dimensions, pixel size |
| `/classify` | Train RF classifier from annotations, compare feature sets |
| `/lmd-export` | Export detections for laser microdissection |
| `/vessel-analysis` | Multi-scale vessel structure detection |
| `/view-results` | Launch HTML result viewer with tunnel |
| `/spatialdata` | Export to SpatialData zarr + squidpy spatial analysis |
| `/preview-preprocessing` | Preview flat-field/photobleach correction |

### Without Claude Code

```bash
conda activate mkseg

# Step 0: ALWAYS inspect CZI channels first (order is NOT wavelength-sorted)
python scripts/czi_info.py /path/to/slide.czi

# Step 1: Detect cells (use --channel-spec to resolve by marker name)
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type cell \
    --channel-spec "cyto=NeuN,nuc=488" \
    --all-channels \
    --num-gpus 4 \
    --output-dir /path/to/output

# Step 2: View results
python serve_html.py /path/to/output
```

### SLURM batch pipeline

Write a YAML config (see `configs/` for examples), then:

```bash
scripts/run_pipeline.sh configs/my_experiment.yaml
```

This generates an sbatch script, submits to SLURM, and chains detection → marker classification → spatial analysis → viewer generation. One slide per SLURM array task for parallel throughput.

---

## DVP Workflow: Detect Once, Classify Later

The core workflow for Deep Visual Proteomics — from slide to mass spec:

1. **Inspect** — `scripts/czi_info.py` reads channel metadata (seconds)
2. **Detect** — `run_segmentation.py` finds all cells with AI segmentation. Each gets a contour + 6,478 features (morphological, intensity, SAM2/ResNet/DINOv2 embeddings). Checkpointed per-tile. (1–3 hours on GPU)
3. **Post-process** — Automatic contour dilation + pixel-level background correction (KD-tree, k=30 neighbors)
4. **Annotate** — Open HTML viewer, click yes/no on ~200+ cell crops. Export annotations.
5. **Train** — `train_classifier.py` trains RF classifier in seconds. Use `scripts/compare_feature_sets.py` to find the best feature combination for your data.
6. **Score** — `scripts/apply_classifier.py` scores every detection (CPU, seconds — no re-detection)
7. **Filter** — `scripts/regenerate_html.py --score-threshold 0.5` shows only confident detections
8. **Markers** — `scripts/classify_markers.py` classifies pos/neg per fluorescent channel (Otsu/GMM)
9. **Explore** — Spatial analysis, UMAP clustering, tissue zonation (see Available Analyses below)
10. **Export** — `run_lmd_export.py` generates XML for the Leica LMD with 384-well plate assignment

---

## Available Analyses

Beyond the core detection → LMD workflow, the pipeline provides:

| Analysis | Script | What it does |
|----------|--------|-------------|
| **Feature comparison** | `scripts/compare_feature_sets.py` | Compare morph/SAM2/deep feature subsets via 5-fold CV |
| **Marker classification** | `scripts/classify_markers.py` | Otsu/GMM pos/neg per marker, auto bg correction |
| **Feature exploration** | `scripts/cluster_by_features.py` | UMAP + HDBSCAN clustering — discover cell subtypes |
| **Spatial network** | `scripts/spatial_cell_analysis.py` | Delaunay graphs, community detection, neighborhoods |
| **Interactive spatial viewer** | `scripts/generate_multi_slide_spatial_viewer.py` | KDE contours, graph-pattern regions, DBSCAN + hulls, ROI drawing |
| **Tissue overlay viewer** | `scripts/generate_tissue_overlay.py` | Fluorescence image + cell overlay + ROI + LMD export |
| **Tissue zone assignment** | `scripts/assign_tissue_zones.py` | Spatially-constrained marker-based zone discovery |
| **Zonation transects** | `scripts/zonation_transect.py` | Pericentral → periportal gradient analysis |
| **Tissue area measurement** | `scripts/calculate_tissue_areas.py` | Variance-based tissue detection from CZI |
| **Bone region annotation** | `scripts/annotate_bone_regions.py` | Interactive HTML tool for bone region labeling |
| **Vessel community analysis** | `scripts/vessel_community_analysis.py` | Multi-scale vessel structures (morphology + SNR) |
| **MK maturation staging** | `scripts/maturation_analysis.py` | Nuclear deep features for maturation states |
| **MK comprehensive** | `scripts/mk_comprehensive_analysis.py` | Multi-dimensional MK feature analysis |
| **Islet spatial analysis** | `scripts/analyze_islets.py` | Spatial analysis of pancreatic islets |
| **SpatialData export** | `scripts/convert_to_spatialdata.py` | Export to scverse zarr (squidpy, scanpy, anndata) |
| **One-command viz** | `scripts/view_slide.py` | Classify → spatial → viewer → serve (all in one) |
| **Preprocessing preview** | `scripts/preview_preprocessing.py` | Before/after flat-field, photobleach at 1/8 resolution |

SpatialData zarr is auto-exported at the end of every detection run. Load with `spatialdata.read_zarr()` for squidpy spatial statistics, scanpy dimensionality reduction, and anndata analysis.

---

## Features

- **Multi-GPU always-on**: Even `--num-gpus 1` uses the multi-GPU code path
- **Multi-node sharding**: `--tile-shard INDEX/TOTAL` for splitting across SLURM array tasks
- **Automatic channel resolution**: `--channel-spec "detect=BTX"` resolves marker names from CZI metadata
- **Up to 6,478 features per detection**: Morphological (78) + per-channel stats + SAM2 (256) + ResNet (4,096) + DINOv2 (2,048)
- **Pixel-level background correction**: KD-tree local background estimation, automatic during detection
- **Checkpoint/resume**: Per-tile checkpoints, dedup checkpoint, post-dedup checkpoint. `--resume` skips completed work
- **Direct-to-SHM loading**: CZI channels loaded directly into shared memory (~9 GB savings for 3-channel slides)
- **SpatialData integration**: Auto-exports to scverse ecosystem (squidpy, scanpy)
- **Interactive viewers**: HTML spatial viewer with KDE contours, ROI drawing, tissue overlay
- **LMD export**: Contour dilation, clustering, 384-well serpentine layout, multi-plate overflow, XML for Leica

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/czi_info.py` | **Run first** — CZI channels, dimensions, pixel size |
| `run_segmentation.py` | Unified detection pipeline (all cell types) |
| `train_classifier.py` | Train RF classifier from annotations |
| `scripts/apply_classifier.py` | Score detections with trained classifier |
| `scripts/classify_markers.py` | Marker pos/neg classification (Otsu/GMM) |
| `scripts/regenerate_html.py` | Regenerate HTML viewer from saved detections |
| `scripts/run_pipeline.sh` | YAML config-driven SLURM batch launcher |
| `run_lmd_export.py` | Export to Leica LMD format |
| `scripts/system_info.py` | Detect environment + recommend SLURM resources |
| `scripts/view_slide.py` | One-command: classify → spatial → viewer → serve |
| `serve_html.py` | HTTP server + Cloudflare tunnel |

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)** — Technical reference: architecture, code patterns, CLI flags, all entry points
- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** — Detailed user guide with examples
- **[docs/NMJ_PIPELINE_GUIDE.md](docs/NMJ_PIPELINE_GUIDE.md)** — NMJ detection + classifier workflow
- **[docs/NMJ_LMD_EXPORT_WORKFLOW.md](docs/NMJ_LMD_EXPORT_WORKFLOW.md)** — Full NMJ → LMD export
- **[docs/LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md)** — LMD export reference
- **[docs/COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md)** — Coordinate conventions
- **[docs/VESSEL_COMMUNITY_ANALYSIS.md](docs/VESSEL_COMMUNITY_ANALYSIS.md)** — Vessel community analysis

## Best Practices

- **Always run `czi_info.py` before any channel config** — CZI channel order follows acquisition/detector assignment, NOT wavelength order. This is the only authoritative source.
- **Use `--channel-spec`** instead of raw indices — resolves marker names against CZI metadata automatically
- **Use `--all-channels`** for multi-channel slides — enables per-channel feature extraction
- **Always detect 100%** (default) — annotate from the HTML subsample (`--html-sample-fraction 0.10`), apply classifier post-hoc, never re-detect
- **Check `scripts/system_info.py`** before SLURM launches — recommends partition, GPU count, memory
- **Resume crashed runs** — add `resume_dir: /path/to/run_dir` to YAML config, then re-run `scripts/run_pipeline.sh`

## Citation

- [SAM2](https://github.com/facebookresearch/segment-anything-2) — Segment Anything Model 2
- [Cellpose](https://github.com/MouseLand/cellpose) — Cell segmentation

## License

MIT License — See LICENSE file for details
