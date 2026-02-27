# xldvp_seg - Image Analysis & Segmentation Pipeline

Automated detection, annotation, classification, and LMD export for multiple cell/structure types in CZI whole-slide images.

## Supported Cell Types

| Type | Method | Use Case |
|------|--------|----------|
| **NMJ** | Intensity threshold + morphology + watershed | Neuromuscular junction detection |
| **MK** | SAM2 auto-mask + size filter | Megakaryocyte detection |
| **Vessel** | SMA+ ring detection + contour hierarchy | Blood vessel morphometry |
| **Islet** | Cellpose membrane+nuclear + marker classification | Pancreatic islet cells |
| **Tissue Pattern** | Cellpose + spatial frequency analysis | Brain FISH cell typing |
| **Mesothelium** | Ridge detection | Mesothelial ribbon for LMD |

## Quick Start

```bash
# Install
conda create -n mkseg python=3.11 -y && conda activate mkseg
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
./install.sh  # Auto-detects CUDA

# Run detection (10% annotation run)
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel 1 \
    --sample-fraction 0.10

# View results
python -m http.server 8080 --directory output/html
```

## Workflow: Detect Once, Classify Later

1. **Detect** 100% of tiles (or multi-node with `--tile-shard`)
2. **Annotate** subsample in HTML viewer
3. **Train** RF classifier: `python train_classifier.py`
4. **Score** all detections: `python scripts/apply_classifier.py` (CPU, seconds)
5. **Review** filtered HTML: `python scripts/regenerate_html.py --score-threshold 0.5`
6. **Export** to LMD: `python run_lmd_export.py`

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete technical reference (CLI args, architecture, modules)
- **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** - User guide with examples
- **[docs/NMJ_PIPELINE_GUIDE.md](docs/NMJ_PIPELINE_GUIDE.md)** - NMJ-specific guide
- **[docs/LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md)** - LMD export workflow
- **[docs/COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md)** - Coordinate conventions

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_segmentation.py` | Unified detection pipeline (all cell types) |
| `train_classifier.py` | Train RF classifier from annotations |
| `scripts/apply_classifier.py` | Score detections with trained classifier |
| `scripts/regenerate_html.py` | Regenerate HTML viewer from saved detections |
| `run_lmd_export.py` | Export to Leica LMD format |

## Multi-GPU / Multi-Node

```bash
# Multi-GPU (single node)
python run_segmentation.py --num-gpus 4 ...

# Multi-node sharding (SLURM array jobs)
python run_segmentation.py --tile-shard 0/4 ...  # shard 0 of 4
python run_segmentation.py --tile-shard 1/4 ...  # shard 1 of 4

# Merge shards after all complete
python run_segmentation.py --resume /path/to/output --merge-shards ...
```

## Citation

- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [Cellpose](https://github.com/MouseLand/cellpose)

## License

MIT License - See LICENSE file for details
