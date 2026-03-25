# xldvp_seg

**Spatial cell segmentation and Deep Visual Proteomics pipeline for CZI microscopy**

[![CI](https://github.com/peptiderodriguez/xldvp_seg/actions/workflows/test.yml/badge.svg)](https://github.com/peptiderodriguez/xldvp_seg/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: 460](https://img.shields.io/badge/tests-460%20passed-brightgreen.svg)]()

Detect cells in whole-slide CZI images, classify them by type and marker expression, analyze spatial organization, and export selected cells for laser microdissection and mass spectrometry. End-to-end DVP (Deep Visual Proteomics) from slide to spatial proteomics.

```
CZI slide → AI detection → annotation → classification → spatial analysis → LMD export → mass spec
```

## Why this pipeline?

- **8 detection strategies** covering NMJ, megakaryocytes, vessels, islet cells, mesothelium, tissue patterns, and generic cells (Cellpose or InstanSeg)
- **6,478 features per cell** — morphological (78) + per-channel intensity + SAM2 (256) + ResNet (4,096) + DINOv2 (2,048)
- **Multi-GPU, multi-node** — scales from laptop to SLURM cluster with per-tile checkpointing and crash resume
- **Detect once, classify later** — train RF classifier on annotations, score all detections in seconds without re-running detection
- **Full spatial analysis** — UMAP/t-SNE, Leiden clustering, Delaunay networks, tissue zonation, SpatialData/scverse integration
- **LMD-ready** — 384-well plate assignment, contour dilation, serpentine well ordering, XML export for Leica

---

## Quick Start

```bash
# Install
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
conda create -n xldvp_seg python=3.10 -y && conda activate xldvp_seg
pip install -e .
./install.sh                    # PyTorch + SAM2 (auto-detects CUDA)

# Run
xlseg info slide.czi            # Inspect channels (ALWAYS first)
xlseg detect --czi-path slide.czi --cell-type cell \
    --channel-spec "cyto=PM,nuc=488" --all-channels --num-gpus 4 \
    --output-dir output/
xlseg serve output/             # View results in browser
```

Or use [Claude Code](https://claude.ai/claude-code) for interactive guidance:

```
cd xldvp_seg && claude
/analyze                        # Claude walks you through everything
```

---

## Supported Cell Types

| Type | Method | Use Case |
|------|--------|----------|
| **Cell** | Cellpose 2-channel (cyto+nuc) + SAM2 | Generic cell detection |
| **NMJ** | Percentile threshold + morphology + watershed | Neuromuscular junctions |
| **MK** | SAM2 auto-mask + size filter | Bone marrow megakaryocytes |
| **Vessel** | SMA+ ring detection, 3-contour hierarchy | Blood vessel morphometry |
| **Islet** | Cellpose membrane+nuclear + markers | Pancreatic islet cells |
| **Tissue Pattern** | Cellpose + spatial frequency analysis | Brain FISH, coronal sections |
| **Mesothelium** | Ridge detection for ribbon structures | Mesothelial ribbon for LMD |
| **InstanSeg** | 3.8M-param lightweight alternative | `--segmenter instanseg` |

---

## Architecture

```
segmentation/              # Main package (pip install -e .)
├── api/                   # Scanpy-style API (pp, tl, pl, io)
├── core/                  # SlideAnalysis central state object
├── detection/
│   ├── strategies/        # 8 strategies, self-registered via @register_strategy
│   └── registry.py        # Strategy registry with decorator pattern
├── models/                # Model registry (SAM2, ResNet, DINOv2, brightfield FMs)
├── pipeline/              # 9 modules: cli, preprocessing, post_detection, background, ...
├── processing/            # Multi-GPU workers, deduplication, strategy factory
├── lmd/                   # Well plates, contour processing, clustering
├── analysis/              # OmicLinker, aggregation, nuclear counting
├── metrics/               # IoU, Dice, Panoptic Quality, Hungarian matching
├── datasets/              # Synthetic test data generator
├── io/                    # CZI loader, HTML export, OME-Zarr
├── preprocessing/         # Flat-field, photobleach, stain normalization
├── reporting/             # Stats, plots, vessel reports
└── utils/                 # JSON I/O, device handling, logging, config

scripts/                   # 25 reusable CLI tools
examples/                  # Project-specific analyses by experiment
├── bone_marrow/           # MK, RBC vascularization, bone regions
├── mesothelium/           # MSLN detection + annotation
├── islet/                 # Pancreatic islet analysis
├── tma/                   # TMA core detection + per-core cell segmentation
├── liver/                 # Hepatic zonation, DCN+, transects
├── nmj/                   # NMJ SLURM scripts
├── vessel/                # Vessel classifier training
├── tissue_pattern/        # Brain FISH analysis
├── senescence/            # Senescence cell configs
├── configs/               # YAML pipeline templates
└── legacy/                # Deprecated scripts (archived)

tests/                     # 460 tests across 18 files
```

---

## DVP Workflow

The pipeline follows a **detect-once, classify-later** design:

| Step | What happens | Time |
|------|-------------|------|
| 1. **Inspect** | Read CZI channel metadata | seconds |
| 2. **Detect** | AI segmentation (Cellpose/InstanSeg + SAM2). Checkpointed per-tile. | 1-3 hours |
| 3. **Post-process** | Contour dilation + KD-tree background correction (automatic) | minutes |
| 4. **Annotate** | Click yes/no on cell crops in HTML viewer. Export JSON. | 10-30 min |
| 5. **Train** | RF classifier from annotations (morph, SAM2, or all 6,478 features) | seconds |
| 6. **Score** | Apply classifier to all detections (no re-detection) | seconds |
| 7. **Markers** | Classify pos/neg per fluorescent channel (SNR ≥ 1.5 default) | seconds |
| 8. **Explore** | UMAP, Leiden clustering, spatial networks, tissue zonation | minutes |
| 9. **Export** | LMD XML with 384-well plates, or SpatialData zarr for scverse | seconds |

---

## Python API

```python
from segmentation.core import SlideAnalysis
from segmentation.api import tl

# Load pipeline output
slide = SlideAnalysis.load("output/my_slide/run_20260324/")
print(f"{slide.n_detections} detections, {slide.cell_type}")

# Classify markers + score with trained RF
tl.markers(slide, marker_channels=[1, 2], marker_names=["NeuN", "tdTomato"])
tl.score(slide, classifier="classifiers/rf_morph.pkl")

# Filter and export
neurons = slide.filter(score_threshold=0.5).filter(marker="NeuN", positive=True)
adata = neurons.to_anndata()  # AnnData for scanpy/scverse
```

---

## SLURM Batch Pipeline

Write a YAML config and launch:

```yaml
# examples/configs/my_experiment.yaml
name: my_experiment
czi_path: /data/slide.czi
output_dir: /output/my_experiment
cell_type: cell
channel_map:
  cyto: PM
  nuc: 488
all_channels: true
html_sample_fraction: 0.10
slurm:
  partition: gpu
  gpus: "l40s:4"
  cpus: 128
  mem_gb: 500
  time: "3-00:00:00"
```

```bash
scripts/run_pipeline.sh examples/configs/my_experiment.yaml
```

Chains detection → marker classification → nuclei counting → HTML viewer generation as separate SLURM jobs with correct dependencies.

---

## Available Analyses

| Analysis | Script | Description |
|----------|--------|-------------|
| Feature comparison | `scripts/compare_feature_sets.py` | 5-fold CV across morph/SAM2/deep feature subsets |
| Marker classification | `scripts/classify_markers.py` | Median SNR / Otsu / GMM per channel |
| UMAP + clustering | `scripts/cluster_by_features.py` | Leiden/HDBSCAN, trajectory, spatial smoothing |
| Spatial networks | `scripts/spatial_cell_analysis.py` | Delaunay graphs, community detection |
| Interactive viewer | `scripts/generate_multi_slide_spatial_viewer.py` | KDE contours, ROI drawing, graph patterns |
| Vessel structures | `scripts/vessel_community_analysis.py` | Multi-scale morphology + SNR |
| SpatialData export | `scripts/convert_to_spatialdata.py` | scverse zarr (squidpy, scanpy) |
| Nuclear counting | `scripts/count_nuclei_per_cell.py` | Cellpose 2nd pass, per-nucleus features |
| Quality filter | `scripts/quality_filter_detections.py` | Heuristic filter (no annotation needed) |
| One-command viz | `scripts/view_slide.py` | Classify → cluster → viewer → serve |
| ROI detection | `examples/islet/`, `examples/tma/` | Find islet regions, TMA cores, or other ROIs → detect cells within ROIs only |

See `examples/` for experiment-specific analyses (bone marrow, liver zonation, islets, TMA, vessels, NMJ, mesothelium).

---

## Development

```bash
pip install -e ".[dev]"     # Install with dev tools
make test                   # 460 tests with coverage
make lint                   # ruff + black check
make format                 # Auto-fix formatting
pre-commit install          # Hook for pre-commit checks
```

**Style:** Black (line-length 100) + Ruff. Python 3.10 (pinned). See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Technical reference — architecture, CLI flags, code patterns |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, style guide, PR process |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Detailed user guide with examples |
| [docs/COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md) | Coordinate conventions ([x,y] everywhere) |
| [docs/LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md) | LMD export reference |
| [docs/NMJ_PIPELINE_GUIDE.md](docs/NMJ_PIPELINE_GUIDE.md) | NMJ detection workflow |
| [docs/VESSEL_COMMUNITY_ANALYSIS.md](docs/VESSEL_COMMUNITY_ANALYSIS.md) | Vessel community analysis |

---

## Key Design Decisions

- **Channel order is NOT wavelength-sorted.** Always run `xlseg info` first. The only authoritative source is CZI metadata.
- **Detect 100%, subsample HTML.** `--html-sample-fraction 0.10` keeps the viewer fast; detection is always full coverage.
- **Atomic JSON writes.** `atomic_json_dump()` uses temp file + `os.replace()` to prevent corruption on crash.
- **KD-tree background correction.** Per-cell local background from k=30 nearest neighbors, cached across channels.
- **Direct-to-SHM loading.** CZI channels loaded directly into shared memory, eliminating ~9 GB peak memory.
- **Strategy pattern.** Detection strategies self-register via `@register_strategy` decorator — add a new cell type in one file.

---

## Citation

If you use this pipeline, please cite:

- [SAM2](https://github.com/facebookresearch/segment-anything-2) — Segment Anything Model 2 (Meta AI)
- [Cellpose](https://github.com/MouseLand/cellpose) — Cell segmentation (Stringer et al.)

## License

[MIT License](LICENSE)
