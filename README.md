# xldvp_seg

**Spatial cell segmentation and Deep Visual Proteomics pipeline for CZI microscopy**

[![CI](https://github.com/peptiderodriguez/xldvp_seg/actions/workflows/test.yml/badge.svg)](https://github.com/peptiderodriguez/xldvp_seg/actions)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/peptiderodriguez/xldvp_seg/branch/main/graph/badge.svg)](https://codecov.io/gh/peptiderodriguez/xldvp_seg)

Detect cells in whole-slide CZI images, classify them by type and marker expression, analyze spatial organization, and export selected cells for laser microdissection and mass spectrometry. End-to-end DVP (Deep Visual Proteomics) from slide to spatial proteomics.

```
CZI slide → AI detection → annotation → classification → spatial analysis → LMD export → mass spec
```

---

## Quick Start

```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
conda create -n xldvp_seg python=3.10 -y && conda activate xldvp_seg
pip install -e .
./install.sh                    # auto-detects CUDA, installs PyTorch + SAM2 + deps
```

### Two install modes

| Mode | Command | Best for |
|------|---------|----------|
| **Reproducible** (default) | `./install.sh` | Exact pinned versions from `requirements-lock.txt`. Two people running this a month apart get identical environments. Recommended for most users. |
| **Latest** | `./install.sh --latest` | Auto-detects your CUDA version, installs latest compatible PyTorch + SAM2 + all deps. Use when setting up on new hardware with a different CUDA version than the lock file. |

Additional flags: `--cuda 11.8|12.1|12.4` (implies --latest), `--cpu` (CPU-only, implies --latest), `--rocm` (AMD GPUs, implies --latest), `--dev` (add pytest/ruff/black).

**Optional extras:** `pip install -e ".[brightfield]"` for brightfield foundation models. `pip install -e ".[instanseg]"` for InstanSeg segmenter.

```bash
# Run
xlseg info slide.czi            # Inspect channels (ALWAYS first)
xlseg detect --czi-path slide.czi --cell-type cell \
    --channel-spec "cyto=PM,nuc=488" --all-channels --num-gpus 4 \
    --output-dir output/
xlseg serve output/             # View results in browser
```

### Try the API without data

```python
from xldvp_seg.datasets import sample
from xldvp_seg.core import SlideAnalysis

slide = SlideAnalysis.from_detections(sample()["detections"])
print(f"{slide.n_detections} synthetic detections")
adata = slide.to_anndata()  # AnnData ready for scanpy
```

---

## Claude Code Integration

Use [Claude Code](https://claude.ai/claude-code) for interactive, AI-guided analysis:

```
cd xldvp_seg && claude
/analyze                        # Claude walks you through everything
/new-experiment                 # Fast-track: CZI inspect → YAML config → launch
```

The package ships with a custom Claude Code configuration (`.claude/commands/`, `.claude/agents/`, `CLAUDE.md`) that turns Claude into a pipeline-aware assistant:

- **Adaptive experience level** — type `/analyze` and Claude asks if you're new or experienced. Beginners get step-by-step explanations with jargon defined inline; experienced users get concise commands with confirmation prompts. Switch anytime by saying "beginner mode" or "advanced mode."
- **Guided workflows** — Claude follows the standard DVP pipeline (inspect → detect → annotate → classify → markers → spatial → LMD) but you're free to skip steps, jump ahead, or mix and match. It will warn you if you're missing a dependency (e.g., trying marker classification without `--all-channels`).
- **Interactive decisions** — at key points (cell type selection, channel mapping, marker thresholds, which analyses to run), Claude uses a structured question tool so you can click options rather than type. It never assumes — it asks.
- **SLURM-aware** — Claude checks cluster availability, writes YAML configs, generates sbatch scripts, monitors jobs, and diagnoses failures from log files.

### Security

The package includes project-level security settings (`.claude/settings.json`) that apply to all users:

```json
{
  "permissions": {
    "deny": [
      "Read(~/.ssh/**)", "Read(~/.aws/**)", "Read(**/.env)",
      "Bash(rm:*)", "Bash(sudo:*)", "Bash(su:*)",
      "Bash(curl:*)", "Bash(wget:*)", "Bash(ssh:*)"
    ]
  }
}
```

These rules are **project-scoped** — they only apply when Claude Code is launched from within this repo directory. They do not affect your global Claude Code settings or other projects. All tool calls still require your approval via the standard permission prompt — Claude proposes, you approve.

**Data directory access:** By default Claude can only access the project directory. When you provide CZI or output paths outside the repo, Claude will offer to add them:

```bash
claude config set additionalDirectories '/path/to/your/data'
```

You approve the command, and the directory is added to your local settings (`.claude/settings.local.json`, gitignored — never shared).

---

## Why This Pipeline?

- **8 detection strategies** — NMJ, megakaryocytes, vessels, islet cells, mesothelium, tissue patterns, and generic cells (Cellpose or InstanSeg)
- **Up to 6,478 features per cell** — morphological (78) + per-channel intensity (~15/ch) + [SAM2](https://github.com/facebookresearch/segment-anything-2) (256) + ResNet-50 (4,096) + [DINOv2](https://github.com/facebookresearch/dinov2) (2,048). Brightfield pathology FMs also supported: [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1), [UNI2](https://huggingface.co/MahmoodLab/UNI2-h), [Virchow2](https://huggingface.co/paige-ai/Virchow2), [CONCH](https://huggingface.co/MahmoodLab/CONCH), [Phikon-v2](https://huggingface.co/owkin/phikon-v2)
- **Multi-GPU, multi-node** — scales from laptop to SLURM cluster with per-tile checkpointing and crash resume
- **Detect once, classify later** — train RF classifier on annotations, score all detections in seconds without re-running detection
- **Integrated nuclear counting** — Cellpose segments nuclei within each cell during detection (no extra I/O), adds N:C ratio and per-nucleus features
- **ROI support (before or after detection)** — define ROIs before detection to skip 95%+ of non-ROI tissue (islets, TMA cores, bone regions), or draw ROIs after detection in the spatial viewer to select cells for analysis, LMD export, or sliding window sampling along tissue structures
- **Full spatial analysis** — UMAP/t-SNE, Leiden clustering, Delaunay networks, tissue zonation, SpatialData/scverse integration
- **LMD-ready** — adaptive contour simplification (10% shape tolerance) + adaptive dilation (10% area tolerance), 384-well plate assignment with serpentine collection through B2→B3→C3→C2 quadrants, nearest-neighbor path optimization on the slide, spatial negative controls, multi-plate overflow, XML export for Leica LMD7

---

## DVP Workflow

The pipeline follows a **detect-once, classify-later** design. All features are extracted from the **original segmentation mask** — contour simplification and dilation are applied only at LMD export time.

| Step | What happens | Time |
|------|-------------|------|
| 1. **Inspect** | Read CZI channel metadata (`xlseg info`) | seconds |
| 2. **Detect** | AI segmentation (Cellpose/InstanSeg + SAM2). Checkpointed per-tile. | 1-3 hours |
| 3. **Post-process** | Contour extraction + background correction + nuclear counting (automatic) | minutes |
| 4. **Annotate** | Click yes/no on cell crops in HTML viewer. Export JSON. | 10-30 min |
| 5. **Train** | RF classifier from annotations (morph, SAM2, or all features) | seconds |
| 6. **Score** | Apply classifier to all detections (no re-detection) | seconds |
| 7. **Markers** | Classify pos/neg per fluorescent channel (SNR ≥ 1.5 default) | seconds |
| 8. **Explore** | UMAP, Leiden clustering, spatial networks, tissue zonation | minutes |
| 9. **Export** | LMD XML with adaptive contours + 384-well plates, or SpatialData zarr | seconds |

---

## Supported Cell Types

| Type | Method | Use Case |
|------|--------|----------|
| **Cell** | Cellpose 2-channel (cyto+nuc) + SAM2 | Generic cell detection |
| **NMJ** | Percentile threshold + morphology + watershed | Neuromuscular junctions |
| **MK** | SAM2 auto-mask + size filter | Bone marrow megakaryocytes |
| **Vessel** | SMA+ ring detection, 3-contour hierarchy | Blood vessel morphometry (7 vessel types) |
| **Islet** | Cellpose membrane+nuclear + markers | Pancreatic islet cells |
| **Tissue Pattern** | Cellpose + spatial frequency analysis | Brain FISH, coronal sections |
| **Mesothelium** | Ridge detection for ribbon structures | Mesothelial ribbon for LMD |
| **InstanSeg** | 3.8M-param lightweight alternative | `--cell-type cell --segmenter instanseg` |

---

## Data Export & Downstream Analysis

### AnnData Layout

`to_anndata()` produces a scanpy-ready object with full pipeline provenance:

| Slot | Content |
|------|---------|
| **`X`** | Morphological + per-channel intensity features (float32). `area_um2`, `n_nuclei`, `nuclear_area_fraction` are in obs, not X. |
| **`obs`** | Per-cell metadata: `uid`, `slide_name`, `cell_type`, `pixel_size_um`, `area_um2`, `rf_prediction`, `marker_profile`, `*_class`, `n_nuclei`, `nuclear_area_fraction`, `nuclear_solidity`, `nuclear_eccentricity`, `largest_nucleus_um2` |
| **`var`** | Feature metadata with `feature_group` column (`morph` / `channel` / `ratio` / `nuclear`) — filter with `adata[:, adata.var["feature_group"] == "morph"]` |
| **`obsm["spatial"]`** | (N, 2) cell positions in micrometers |
| **`obsm["X_sam2"]`** | SAM2 embeddings (256D) |
| **`obsm["X_resnet"]`**, **`obsm["X_resnet_ctx"]`** | ResNet-50 masked + context features (2×2048D, if `--extract-deep-features`) |
| **`obsm["X_dinov2"]`**, **`obsm["X_dinov2_ctx"]`** | DINOv2 masked + context features (2×1024D, if `--extract-deep-features`) |
| **`uns["pipeline"]`** | Provenance: package version, slide name, cell type, pixel size, detection count, channel map |

Multi-scene slides: each scene produces a separate `SlideAnalysis`; concatenate with `anndata.concat(adatas, label="scene")`.

### Single-Cell Analysis (scanpy + squidpy)

Each detection is a row in the AnnData — use standard scanpy workflows:

```python
import scanpy as sc
import squidpy as sq
from xldvp_seg.core import SlideAnalysis

slide = SlideAnalysis.load("output/my_slide/run_20260324/")
adata = slide.to_anndata()

# Scale features (z-score — NOT normalize_total/log1p, those are for RNA-seq)
sc.pp.scale(adata)

# Dimensionality reduction + clustering
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.3)
sc.pl.umap(adata, color=["leiden", "marker_profile", "area_um2"])

# Spatial analysis
sq.gr.spatial_neighbors(adata, coord_type="generic")
sq.gr.nhood_enrichment(adata, cluster_key="marker_profile")
sq.gr.spatial_autocorr(adata, mode="moran")  # spatially variable features
```

### Pooled-Cell Proteomics (after LMD + mass spec)

DVP typically **pools multiple cells per well** for sufficient protein yield. After LMD cutting and mass spectrometry, `OmicLinker` bridges per-cell morphology to well-level proteomics:

```python
from xldvp_seg.analysis.omic_linker import OmicLinker

linker = OmicLinker.from_slide(slide)
linker.load_proteomics("proteomics.csv")    # wells × proteins (pooled measurement)
linker.load_well_mapping("lmd_export/")     # cell → well assignment
linked = linker.link()                       # DataFrame: aggregated features + proteomics per well
```

**Feature aggregation per well:**

| Feature type | Aggregation | Rationale |
|-------------|-------------|-----------|
| Morphology (area, solidity, ...) | **median** | Robust to outlier cells |
| Channel intensity (ch0_snr, ...) | **median** | Robust to outlier pixels |
| Embeddings (sam2_, resnet_, ...) | **mean** | Preserves centroid in representation space |
| Spatial position | **centroid** | Pool center-of-mass on tissue (`pool_x_um`, `pool_y_um`) |

Each well also gets: `pool_total_area_um2` (summed cell area — correlates with protein yield), `pool_n_cells` (cell count), `pool_x_um`/`pool_y_um` (spatial centroid), `pool_spread_um` (spatial spread), and `pool_std_{feature}` (within-well standard deviation for assessing pool heterogeneity).

```python
# Differential analysis between marker populations
diff = linker.differential_features("marker_profile", "NeuN+/tdTomato-", "NeuN-/tdTomato+")
corr = linker.correlate(method="spearman")   # FDR-corrected well-level correlations
corr, pvals = linker.correlate(return_pvalues=True)  # with Benjamini-Hochberg adjusted p-values
```

For rare large cells (e.g., MKs), single-cell-per-well is sometimes feasible — in that case the aggregation is a no-op (median of 1 = the value itself).

---

## Available Analyses

| Analysis | Tool | Description |
|----------|------|-------------|
| Feature comparison | `scripts/compare_feature_sets.py` | 5-fold CV across morph/SAM2/deep feature subsets |
| Marker classification | `scripts/classify_markers.py` | Median SNR / Otsu / GMM (with BIC model selection) per channel (core: `xldvp_seg.analysis.marker_classification`) |
| UMAP + clustering | `xlseg cluster` / `scripts/cluster_by_features.py` | Leiden/HDBSCAN, trajectory, spatial smoothing (core: `xldvp_seg.analysis.cluster_features`) |
| Spatial networks | `scripts/spatial_cell_analysis.py` | Delaunay graphs, community detection (core: `xldvp_seg.analysis.spatial_network`) |
| Interactive viewer | `scripts/generate_multi_slide_spatial_viewer.py` | Fluorescence overlay + cell contours + ROI drawing |
| Contour viewer | `scripts/generate_contour_viewer.py` | Contour overlays on CZI fluorescence with pan/zoom, group toggling, click-to-inspect |
| Sliding window | `scripts/sliding_window_sampling.py` | Area-matched rolling window along ROI centerlines for LMD (core: `xldvp_seg.analysis.sliding_window_sampling`) |
| Curvilinear patterns | `scripts/detect_curvilinear_patterns.py` | Strip/ribbon detection via graph diameter linearity (core: `xldvp_seg.analysis.pattern_detection`) |
| Vessel structures | `scripts/detect_vessel_structures.py` | Ring/arc/strip classification from marker+ cells |
| Vessel communities | `scripts/vessel_community_analysis.py` | Multi-scale morphology + SNR |
| SpatialData export | `scripts/convert_to_spatialdata.py` | scverse zarr (squidpy, scanpy) |
| Nuclear counting | `--count-nuclei` (default ON) | Integrated in detection; standalone: `scripts/count_nuclei_per_cell.py` |
| Quality filter | `scripts/quality_filter_detections.py` | Heuristic filter (no annotation needed) |
| One-command viz | `scripts/view_slide.py` | Classify → cluster → viewer → serve |
| ROI detection | `examples/islet/`, `examples/tma/` | Pre-detection: find ROIs → detect within ROIs only. Post-detection: draw ROIs in viewer → filter/export selected cells |

See `examples/` for experiment-specific analyses (bone marrow, liver zonation, islets, TMA, vessels, NMJ, mesothelium).

---

## Python API

`tl.*` is the primary programmatic API (markers, scoring, clustering, spatial analysis). `pl.umap()`, `io.to_spatialdata()`, and `io.read_proteomics()` are also implemented. Detection and LMD export run via CLI (`xlseg detect`, `xlseg export-lmd`).

```python
from xldvp_seg.core import SlideAnalysis
from xldvp_seg.api import tl

# Load pipeline output
slide = SlideAnalysis.load("output/my_slide/run_20260324/")
print(f"{slide.n_detections} detections, {slide.cell_type}")

# Classify markers + score with trained RF
tl.markers(slide, marker_channels=[1, 2], marker_names=["NeuN", "tdTomato"])
tl.score(slide, classifier="classifiers/rf_morph.pkl")

# Cluster by features (UMAP + Leiden)
tl.cluster(slide, feature_groups="morph,channel", output_dir="results/clustering/")

# Spatial network analysis (Delaunay + Louvain communities)
tl.spatial(slide, output_dir="results/spatial/", marker_filter="NeuN_class==positive")

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

Chains detection → marker classification → nuclei counting → HTML viewer as separate SLURM jobs with correct dependencies. Unified work-item model: slides × scenes cross product.

---

## Architecture

```
xldvp_seg/              # Main package (pip install -e .)
├── api/                   # Scanpy-style API (tl primary, pl.umap + io.to_spatialdata implemented)
├── classification/        # Vessel type classifiers, feature selection
├── cli/                   # xlseg CLI entry point (13 subcommands)
├── core/                  # SlideAnalysis central state object + detection schema
├── detection/strategies/  # 8 strategies, self-registered via @register_strategy
├── io/                    # CZI loader, HTML export (6 modules), OME-Zarr, SpatialData export
├── lmd/                   # Well plates, contour processing (adaptive RDP + dilation)
├── analysis/              # 9 modules: marker classification, clustering, spatial networks, patterns, sampling, OmicLinker, aggregation, nuclear counting, vessel characterization
├── visualization/         # Reusable HTML visualization: fluorescence, colors, encoding, data loading, HTML builder, graph patterns, 17 JS components
├── training/              # Classifier training: feature loading, annotation matching
├── models/                # Model registry (SAM2, ResNet, DINOv2, brightfield FMs)
├── pipeline/              # 11 modules: shm_setup, detection_loop, preprocessing, post_detection, ...
├── processing/            # Multi-GPU workers, deduplication, strategy factory
├── roi/                   # ROI support: pre-detection (restrict to regions) or post-detection (spatial filtering)
└── utils/                 # JSON I/O, device handling, logging, config

scripts/                   # 31 reusable CLI tools
examples/                  # Project-specific analyses by experiment
tests/                     # pytest suite (1048 tests — run `make test`)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Channel order is NOT wavelength-sorted** | Always run `xlseg info` first. CZI metadata is the only authoritative source. |
| **Features from original mask** | Contour simplification + dilation applied at LMD export only — features always reflect the true segmentation boundary. |
| **Detect 100%, subsample HTML** | `--html-sample-fraction 0.10` keeps the viewer fast; detection is always full coverage. |
| **Adaptive contour processing** | Binary search for largest RDP epsilon / dilation within 10% tolerance. Cell-size-aware, not one-size-fits-all. |
| **Atomic JSON writes** | `atomic_json_dump()` uses temp file + `os.replace()` to prevent corruption on crash. |
| **KD-tree background correction** | Per-cell local background from k=30 nearest neighbors, cached across channels. |
| **Direct-to-SHM loading** | CZI channels loaded directly into shared memory — no intermediate RAM copy. |
| **Strategy pattern** | Detection strategies self-register via `@register_strategy` — add a new cell type in one file. |
| **Nuclear counting integrated** | `--count-nuclei` (default ON) runs during post-dedup using SHM data — zero extra I/O. |
| **Pickle security** | Classifier files (`.pkl`) use Python pickle serialization via joblib. The pipeline validates structure (dict type check) and model type after loading, and `torch.load` uses `weights_only=True`. Only load classifiers from trusted sources — joblib cannot prevent arbitrary code execution from malicious pickle files. |

## Development

```bash
pip install -e ".[dev]"     # Install with dev tools
make test                   # Run all tests with coverage
make lint                   # ruff + black check
make format                 # Auto-fix formatting
```

**Style:** Black (line-length 100) + Ruff. Python 3.10 | 3.11 (both CI-tested).

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Technical reference — architecture, CLI flags, code patterns |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Detailed user guide with examples |
| [docs/COORDINATE_SYSTEM.md](docs/COORDINATE_SYSTEM.md) | Coordinate conventions ([x,y] everywhere) |
| [docs/LMD_EXPORT_GUIDE.md](docs/LMD_EXPORT_GUIDE.md) | LMD export reference |
| [docs/NMJ_PIPELINE_GUIDE.md](docs/NMJ_PIPELINE_GUIDE.md) | NMJ detection workflow |
| [docs/VESSEL_COMMUNITY_ANALYSIS.md](docs/VESSEL_COMMUNITY_ANALYSIS.md) | Vessel structure analysis |

---

## Citation

If you use this pipeline, please cite:

- [SAM2](https://github.com/facebookresearch/segment-anything-2) — Segment Anything Model 2 (Meta AI)
- [Cellpose](https://github.com/MouseLand/cellpose) — Cell segmentation (Stringer et al.)

## License

[MIT License](LICENSE)
