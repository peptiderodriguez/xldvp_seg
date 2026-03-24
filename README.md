# xldvp_seg — Image Analysis & DVP Pipeline

Automated cell detection, annotation, classification, spatial analysis, and LMD (laser microdissection) export for whole-slide CZI microscopy images. Built for the DVP (Deep Visual Proteomics) workflow: find cells on a slide, classify them by type and marker expression, then export selected cells for laser cutting and mass spec analysis.

Runs on SLURM GPU clusters or local workstations (NVIDIA CUDA, Apple Silicon MPS, or CPU-only). Works with or without [Claude Code](https://claude.ai/claude-code) — an optional AI assistant that can guide you through the entire pipeline interactively.

**Key terms:** CZI = Zeiss microscopy image format. Channels = different fluorescent stains in your image (e.g., nuclear, membrane, antibody markers). SAM2/Cellpose = AI models for cell segmentation. SLURM = cluster job scheduler.

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
| **InstanSeg** | InstanSeg 3.8M-param lightweight segmenter | Alternative to Cellpose (`--segmenter instanseg`) |

---

## Prerequisites

- **Conda** or **Miniforge** ([install miniforge](https://github.com/conda-forge/miniforge#miniforge3))
- **GPU** (recommended): NVIDIA with CUDA 11.8+ and 8+ GB VRAM (16+ GB for large slides). Apple Silicon also supported. CPU-only mode available but slow.
- **Node.js 18+** (optional — only needed if you want Claude Code): `conda install -c conda-forge nodejs`

---

## Installation

### Step 1: Clone and create environment

```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
conda create -n xldvp_seg python=3.11 -y && conda activate xldvp_seg
```

### Step 2: Install the package

```bash
pip install -e .          # Core install (fluorescence pipeline)
pip install -e .[dev]     # + dev tools (pytest, black, ruff)
pip install -e .[instanseg]   # + InstanSeg alternative segmenter
pip install -e .[brightfield] # + brightfield foundation models (timm, transformers)
```

### Step 2b: Install PyTorch + SAM2 (~10-15 min)

```bash
./install.sh  # Auto-detects CUDA, installs PyTorch + SAM2 checkpoint
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
conda activate xldvp_seg
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

The pipeline works the same whether you use Claude Code or run commands directly. Claude Code just automates the configuration and guides you through each step.

### Option A: With Claude Code

```
claude          # start Claude Code in the repo directory
/analyze        # type this to begin — Claude walks you through everything
```

Other Claude Code commands: `/status` (monitor jobs), `/czi-info` (inspect channels), `/classify` (train classifier), `/lmd-export` (LMD XML), `/view-results` (HTML viewer), `/spatialdata` (scverse export).

### Option B: Command line (`xlseg` CLI)

```bash
conda activate xldvp_seg

# Inspect channels (ALWAYS first — order is NOT wavelength-sorted)
xlseg info /path/to/slide.czi

# Detect cells
xlseg detect --czi-path /path/to/slide.czi \
    --cell-type cell \
    --channel-spec "cyto=PM,nuc=488" \
    --all-channels --num-gpus 4 \
    --output-dir /path/to/output

# View results
xlseg serve /path/to/output

# Other commands
xlseg classify --detections det.json --annotations ann.json
xlseg markers --detections det.json --marker-wavelength 647 --marker-name NeuN
xlseg score --detections det.json --classifier rf.pkl
xlseg export-lmd --detections det.json --crosses crosses.json
xlseg models              # list registered models
xlseg strategies          # list detection strategies
xlseg download-models --brightfield  # download gated HF models
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
8. **Markers** — `scripts/classify_markers.py` classifies pos/neg per fluorescent channel (median SNR ≥ 1.5 default, Otsu/GMM alternatives)
9. **Explore** — Spatial analysis, UMAP clustering, tissue zonation (see Available Analyses below)
10. **Export** — `run_lmd_export.py` generates XML for the Leica LMD with 384-well plate assignment

---

## Available Analyses

Beyond the core detection → LMD workflow, the pipeline provides:

| Analysis | Script | What it does |
|----------|--------|-------------|
| **Feature comparison** | `scripts/compare_feature_sets.py` | Compare morph/SAM2/deep feature subsets via 5-fold CV |
| **Marker classification** | `scripts/classify_markers.py` | Median SNR (default ≥1.5) / Otsu / GMM pos/neg per marker |
| **Feature exploration** | `scripts/cluster_by_features.py` | UMAP/t-SNE + Leiden/HDBSCAN, spatial smoothing, trajectory |
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

- **`xlseg` CLI**: Unified command-line interface with 11 subcommands (info, detect, classify, markers, score, export-lmd, serve, system, models, strategies, download-models)
- **Python API**: Scanpy-style `segmentation.api` module (`pp`, `tl`, `pl`, `io`) + `SlideAnalysis` central state object for notebook workflows
- **8 detection strategies**: Cell (Cellpose), NMJ, MK, Vessel, Islet, Tissue Pattern, Mesothelium, InstanSeg — all self-registered via `@register_strategy` decorator
- **Model registry**: Metadata catalog tracking 9+ models with modality (fluorescence/brightfield/both), license, and HuggingFace URLs. Brightfield FMs: UNI2, Virchow2, CONCH, Phikon-v2.
- **Multi-GPU always-on**: Even `--num-gpus 1` uses the multi-GPU code path
- **Multi-node sharding**: `--tile-shard INDEX/TOTAL` for splitting across SLURM array tasks
- **Automatic channel resolution**: `--channel-spec "detect=BTX"` resolves marker names from CZI metadata
- **Up to 6,478 features per detection**: Morphological (78) + per-channel stats + SAM2 (256) + ResNet (4,096) + DINOv2 (2,048)
- **Median-based SNR marker classification**: Default method for pos/neg per channel (SNR ≥ 1.5), with Otsu and GMM alternatives
- **Feature-gated spatial smoothing**: `--spatial-smooth` weights neighbors by both proximity AND feature similarity
- **IoU NMS deduplication**: Contour-based alternative to mask-overlap dedup (`--dedup-method iou_nms`)
- **Segmentation metrics**: IoU, Dice, Panoptic Quality, Hungarian matching for benchmarking
- **Pixel-level background correction**: KD-tree local background estimation, automatic during detection
- **Checkpoint/resume**: Per-tile checkpoints, dedup, post-dedup. `--resume` skips completed work
- **Direct-to-SHM loading**: CZI channels loaded directly into shared memory (~9 GB savings)
- **SpatialData integration**: Auto-exports to scverse ecosystem (squidpy, scanpy)
- **Cohort analysis**: Slide-level feature aggregation for multi-slide experiments
- **Multi-omic linking**: Bridge morphological features to mass-spec proteomics (OmicLinker)
- **Interactive viewers**: HTML spatial viewer with KDE contours, ROI drawing, tissue overlay
- **LMD export**: Contour dilation, clustering, 384-well serpentine layout, multi-plate overflow, XML for Leica
- **GitHub Actions CI**: Automated testing on Python 3.10 + 3.12

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

### Python API (for notebooks)

```python
from segmentation.core import SlideAnalysis
from segmentation.api import tl

slide = SlideAnalysis.load("output/my_slide/run_20260324/")
tl.markers(slide, marker_channels=[1, 2], marker_names=["NeuN", "tdTomato"])
tl.score(slide, classifier="classifiers/rf_morph.pkl")
tl.cluster(slide, methods="both", output_dir="clustering/")
slide.save("scored_detections.json")
adata = slide.to_anndata()  # export to AnnData for scanpy
```

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
