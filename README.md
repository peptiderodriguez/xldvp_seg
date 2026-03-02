# xldvp_seg - Image Analysis & Segmentation Pipeline

Automated detection, annotation, classification, and LMD export for multiple cell/structure types in CZI whole-slide images. Designed for SLURM GPU clusters with Claude Code as the interactive interface.

## Supported Cell Types

| Type | Method | Use Case |
|------|--------|----------|
| **Cell** | Cellpose + SAM2 embeddings | Generic cell detection (2-channel: cyto + nuc) |
| **NMJ** | Intensity threshold + morphology + watershed | Neuromuscular junction detection |
| **MK** | SAM2 auto-mask + size filter | Megakaryocyte detection |
| **Vessel** | SMA+ ring detection + 3-contour hierarchy | Blood vessel morphometry |
| **Islet** | Cellpose membrane+nuclear + marker classification | Pancreatic islet cells |
| **Tissue Pattern** | Cellpose + spatial frequency analysis | Brain FISH cell typing |
| **Mesothelium** | Ridge detection | Mesothelial ribbon for LMD |

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
conda create -n mkseg python=3.11 -y && conda activate mkseg
./install.sh  # Auto-detects CUDA version, installs PyTorch + SAM2 + Cellpose
```

### 2. Verify

```bash
conda activate mkseg
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cellpose; print(f'Cellpose {cellpose.__version__}')"
```

### 3. Install Claude Code (recommended)

Claude Code is an AI-powered CLI that guides you through the entire pipeline interactively. It knows the codebase, detects your system, inspects your data, and builds the right commands.

```bash
# Install Claude Code (requires Node.js 18+)
npm install -g @anthropic-ai/claude-code

# Start Claude Code in the repo directory
cd xldvp_seg
claude
```

On first launch, Claude Code reads the project's `CLAUDE.md` and `.claude/commands/` directory to understand the pipeline. No additional configuration needed.

### 4. Start the pipeline

Inside Claude Code, type:

```
/analyze
```

Claude will:
1. Detect your system (local GPU, SLURM cluster, available partitions)
2. Ask you to point to your CZI file(s)
3. Inspect channel metadata and recommend detection settings
4. Build a YAML config and launch the pipeline
5. Monitor progress, diagnose failures, and guide you through annotation/classification/export

Other useful commands:

| Command | What it does |
|---------|-------------|
| `/analyze` | Full pipeline: detect, annotate, classify, spatial analysis, LMD export |
| `/status` | Check running SLURM jobs, tail logs, monitor progress |
| `/czi-info` | Inspect CZI metadata (channels, dimensions, pixel size) |
| `/classify` | Train RF classifier from annotations, compare feature sets |
| `/lmd-export` | Export detections for laser microdissection |
| `/view-results` | Launch HTML result viewer with Cloudflare tunnel |

---

## Manual Quick Start (without Claude Code)

```bash
conda activate mkseg

# Run detection
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type cell \
    --channel-spec "cyto=NeuN,nuc=488" \
    --all-channels \
    --num-gpus 4 \
    --output-dir /path/to/output

# View results
python serve_html.py /path/to/output
```

### SLURM batch pipeline

Write a YAML config (see `configs/` for examples), then:

```bash
scripts/run_pipeline.sh configs/my_experiment.yaml
```

This generates an sbatch script with the right flags, submits to SLURM, and chains detection -> marker classification -> spatial analysis -> viewer generation.

---

## Workflow: Detect Once, Classify Later

1. **Detect** 100% of tiles (multi-GPU, multi-node with `--tile-shard`)
2. **Post-process** contour dilation + background correction (automatic)
3. **Annotate** subsample in HTML viewer (green = real, red = false positive)
4. **Train** RF classifier: `python train_classifier.py`
5. **Score** all detections: `python scripts/apply_classifier.py` (CPU, seconds)
6. **Review** filtered HTML: `python scripts/regenerate_html.py --score-threshold 0.5`
7. **Export** to LMD: `python run_lmd_export.py`

---

## Features

- **Multi-GPU always-on**: Even `--num-gpus 1` uses the multi-GPU code path
- **Multi-node sharding**: `--tile-shard INDEX/TOTAL` for splitting across SLURM array tasks
- **Automatic channel resolution**: `--channel-spec "detect=BTX"` resolves marker names from CZI metadata
- **6,478 features per detection**: Morphological + per-channel stats + SAM2 + ResNet + DINOv2
- **Pixel-level background correction**: KD-tree local background estimation, automatic during detection
- **Checkpoint/resume**: Per-tile checkpoints, dedup checkpoint, post-dedup checkpoint. `--resume` skips completed work
- **SpatialData integration**: Auto-exports to scverse ecosystem (squidpy, scanpy)
- **Interactive spatial viewer**: KDE density contours, multi-scale graph-pattern region detection, DBSCAN clustering with convex hulls, ROI drawing
- **Direct-to-SHM loading**: CZI channels loaded directly into shared memory (no RAM intermediate), ~9 GB savings for 3-channel slides
- **LMD export**: Contour dilation, clustering, 384-well serpentine layout, XML for Leica instruments

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_segmentation.py` | Unified detection pipeline (all cell types) |
| `train_classifier.py` | Train RF classifier from annotations |
| `scripts/apply_classifier.py` | Score detections with trained classifier |
| `scripts/classify_markers.py` | Post-detection marker classification (Otsu/GMM) |
| `scripts/regenerate_html.py` | Regenerate HTML viewer from saved detections |
| `scripts/run_pipeline.sh` | YAML config-driven SLURM batch launcher |
| `run_lmd_export.py` | Export to Leica LMD format |
| `scripts/convert_to_spatialdata.py` | Convert detections to SpatialData zarr |
| `scripts/generate_multi_slide_spatial_viewer.py` | Interactive spatial viewer: KDE contours, graph-pattern regions, DBSCAN clustering |
| `scripts/view_slide.py` | One-command visualization: classify + spatial analysis + viewer + serve |
| `serve_html.py` | HTTP server + Cloudflare tunnel for remote viewing |

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete technical reference (CLI args, architecture, modules)
- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Detailed user guide with examples
- **[docs/NMJ_PIPELINE_GUIDE.md](docs/NMJ_PIPELINE_GUIDE.md)** - NMJ-specific guide
- **[docs/LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md)** - LMD export workflow
- **[docs/COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md)** - Coordinate conventions

## Best Practices

- **Always use `--all-channels`** for multi-channel slides — enables per-channel feature extraction and cross-channel ratios
- **Start with 10% sample** (`--sample-fraction 0.10`) for annotation, then run 100% for full detection
- **Use `--channel-spec`** instead of raw channel indices — automatically resolves marker names against CZI metadata
- **Check `scripts/system_info.py`** before launching — it detects your system and recommends partition, GPU count, and memory settings
- **Always verify channel order from CZI metadata** before writing configs — CZI sorts channels by emission wavelength, which may differ from filename order. Run the pre-flight check in CLAUDE.md or use `/czi-info` to confirm each C index maps to the right marker before setting `cellpose_input_channels`, `marker-channel`, or YAML `channels:`
- **For SLURM restarts**: add `resume_dir: /path/to/run_dir` to your YAML config, then re-run `scripts/run_pipeline.sh` — per-tile checkpoints are used automatically when `--resume` is passed. Without `resume_dir:` in the YAML, re-running starts a fresh full-detection run

## Citation

- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [Cellpose](https://github.com/MouseLand/cellpose)

## License

MIT License - See LICENSE file for details
