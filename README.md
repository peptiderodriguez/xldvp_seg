# xldvp_seg

**Spatial cell segmentation and Deep Visual Proteomics pipeline for CZI microscopy**

[![CI](https://github.com/peptiderodriguez/xldvp_seg/actions/workflows/test.yml/badge.svg)](https://github.com/peptiderodriguez/xldvp_seg/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/peptiderodriguez/xldvp_seg/branch/main/graph/badge.svg)](https://codecov.io/gh/peptiderodriguez/xldvp_seg)

Detect cells in whole-slide CZI images, classify them by type and marker expression, analyze spatial organization, and export selected cells for laser microdissection and mass spectrometry. End-to-end DVP (Deep Visual Proteomics) from slide to spatial proteomics.

```
CZI slide → AI detection → annotation → classification → spatial analysis → LMD export → mass spec
```

---

## Quick Start

**Every install path installs three things: the package, PyTorch, and SAM2 (model + checkpoint).** On Linux/Mac `./install.sh` does all three. On Windows you do them manually (3 commands).

### Linux (cluster / workstation / laptop)

```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
conda create -n xldvp_seg python=3.11 -y && conda activate xldvp_seg
./install.sh                    # installs package + PyTorch (auto-detects CUDA) + SAM2 + checkpoint
```

### macOS (Apple Silicon or Intel)

```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
conda create -n xldvp_seg python=3.11 -y && conda activate xldvp_seg
./install.sh --cpu              # Mac: no CUDA — PyTorch's MPS backend kicks in automatically for Cellpose
```
Apple Silicon MPS is autodetected at runtime via `xldvp_seg.utils.device` — Cellpose segmentation runs 3–10× faster on MPS than on CPU. No manual flag needed.

### Windows (no `install.sh` — manual, one-time)

```powershell
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
conda create -n xldvp_seg python=3.11 -y
conda activate xldvp_seg
# 1. PyTorch (pick CUDA or CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # if no GPU
# 2. Package
pip install -e .
# 3. SAM2 + checkpoint
pip install "git+https://github.com/facebookresearch/sam2.git"
mkdir checkpoints
# Download the checkpoint manually (PowerShell):
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -OutFile "checkpoints\sam2.1_hiera_large.pt"
```

### Verify (any platform)

```bash
xlseg --version                 # confirms package installed
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
xlseg info slide.czi            # test CZI reading (ALWAYS run before detection)
```

### `install.sh` mode flags (Linux/Mac)

| Mode | Command | Best for |
|------|---------|----------|
| **Reproducible** (default) | `./install.sh` | Exact pinned versions from `requirements-lock.txt` — two people running this a month apart get identical environments |
| **Latest** | `./install.sh --latest` | Auto-detects CUDA version, installs latest compatible PyTorch + SAM2 |
| **CPU-only** | `./install.sh --cpu` | No GPU (Mac Intel, some laptops); MPS still works on Mac Apple Silicon |
| **AMD GPU** | `./install.sh --rocm` | ROCm instead of CUDA |
| **Dev tools** | `./install.sh --dev` | Adds pytest/ruff/black |
| **With Claude Code** | `./install.sh --with-claude-code` | Also installs the Claude Code CLI (opt-in) so `/analyze` is available right after install |

Specify CUDA version with `--cuda 11.8|12.1|12.4`.

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

**The recommended way to use this package** is through [Claude Code](https://claude.ai/claude-code) — the repo ships with slash commands, custom agents, and a pipeline-aware `CLAUDE.md` that guide you end-to-end.

### Install Claude Code (one-time)

```bash
# macOS / Linux — recommended
curl -fsSL https://claude.ai/install.sh | bash

# Windows (PowerShell) — recommended
irm https://claude.ai/install.ps1 | iex

# Alternative — any platform with Node 18+
npm install -g @anthropic-ai/claude-code
```

Then sign in once: `claude` (first run prompts for browser auth against your Anthropic / Claude account).

### Use it with xldvp_seg

```bash
cd xldvp_seg && claude       # opens Claude Code inside the repo; /analyze etc. become available
/analyze                     # Claude walks you through install, detection, analysis — one question at a time
/new-experiment              # Fast-track: CZI inspect → YAML config → launch
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
| **MK** | SAM2 auto-mask + size filter; **RGB brightfield CZIs supported**; set `tile_overlap: 0.25` in YAML for large cells | Bone marrow megakaryocytes |
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
| Vessel lumen threshold | `scripts/detect_vessel_lumens_threshold.py` | Gaussian local threshold + watershed on OME-Zarr (CPU, no GPU). See `docs/VESSEL_LUMEN_THRESHOLD_PIPELINE.md` |
| Vessel lumen scoring | `scripts/score_vessel_lumens.py` | RF training, scoring, filtering with annotation overrides |
| Lumen annotation | `scripts/generate_lumen_annotation.py` | Card-grid annotation HTML from zarr crops |
| Vessel wall cells | `scripts/assign_vessel_wall_cells.py` | Per-marker KD-tree wall cell assignment + LMD replicates |
| Vessel structures | `scripts/detect_vessel_structures.py` | Ring/arc/strip classification from marker+ cells |
| Vessel communities | `scripts/vessel_community_analysis.py` | Multi-scale morphology + SNR |
| SpatialData export | `scripts/convert_to_spatialdata.py` | scverse zarr (squidpy, scanpy) |
| Nuclear counting | `--count-nuclei` (default ON) | Integrated in detection; standalone: `scripts/count_nuclei_per_cell.py` |
| Quality filter | `scripts/quality_filter_detections.py` | Heuristic filter (no annotation needed) |
| Post-hoc mask refinement | `scripts/refine_detection_masks.py` | Adaptive per-cell intensity-based boundary peeling (removes bleed into white/empty space) — recomputes contours + shape features without re-running detection. Generic across cell types. |
| One-command viz | `scripts/view_slide.py` | Classify → cluster → viewer → serve |
| ROI detection | `examples/islet/`, `examples/tma/` | Pre-detection: find ROIs → detect within ROIs only. Post-detection: draw ROIs in viewer → filter/export selected cells |
| Block-face registration | VALIS + SAM2 | Register gross tissue photo to fluorescence CZI, auto-segment organs with recursive SAM2, assign detections to anatomical regions for organ-specific LMD. See `docs/BLOCKFACE_REGISTRATION.md` |
| Region segmentation | `scripts/segment_regions.py` / `scripts/assign_cells_to_regions.py` / `scripts/generate_region_viewer.py` | SAM2 on fluorescence thumbnails → per-cell organ assignment → interactive viewer (core: `xldvp_seg.analysis.region_segmentation`) |
| Per-region PCA/UMAP | `scripts/region_pca_viewer.py` | PCA → UMAP with 4 clusterings (kmeans-elbow, Leiden on PCA-kNN, HDBSCAN on PCA, HDBSCAN on UMAP); interactive HTML with color toggle (core: `xldvp_seg.analysis.region_clustering`) |
| Combined region + UMAP viewer | `scripts/combined_region_viewer.py` | 2-pane HTML: whole-slide region map (click to select) + UMAP/clustering side-by-side |
| Global cluster + spatial divergence | `scripts/global_cluster_spatial_viewer.py` | Inverse: cluster ALL cells globally, rank by spatial-divergence metrics (`focal_multimodal`, `k_90`) to find "same feature profile, different anatomy" cell populations |
| Morphological cluster discovery | `xlseg discover-rare-cells` / `scripts/discover_rare_cell_types.py` | HDBSCAN in PCA space with reciprocal-best-match Jaccard stability + vectorized Moran's I + Ward taxonomy. Per-group 1/√(dim) weighting so SAM2 doesn't drown morphology; `-2` sentinel distinguishes pre-filter drops from HDBSCAN noise. Pairs with `global_cluster_spatial_viewer.py --rare-mode` for clickable-dendrogram review. See [docs/CLUSTER_DISCOVERY.md](docs/CLUSTER_DISCOVERY.md) |
| Manifold sampling (LMD pools) | `xlseg manifold-sample` / `scripts/manifold_sample.py` | FPS + Voronoi partition the whole population into K morphologically-coherent "manifold groups", then Ward on xy within each `(group, organ)` pair emits spatially-tight replicate pools at a fixed tissue-area budget (default 2500 µm² ≈ 25 cells). Same embedding as rare-cell discovery; output is LMD-ready pools, not cluster labels. Pairs with `global_cluster_spatial_viewer.py --rare-mode` for review and `xlseg export-lmd` for Leica XML. See [docs/MANIFOLD_SAMPLING.md](docs/MANIFOLD_SAMPLING.md) |
| Per-region multinucleation | `scripts/region_multinuc_plot.py` | Histogram + KDE + Tukey fences + GMM(k=2 via BIC) outlier detection |
| Transcript export | `scripts/export_transcript.py` | Claude Code session JSONL → markdown/HTML (curate + present modes with PNG export) |

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
├── cli/                   # xlseg CLI entry point (15 subcommands)
├── core/                  # SlideAnalysis central state object + detection schema
├── detection/strategies/  # 8 strategies, self-registered via @register_strategy
├── io/                    # CZI loader, HTML export (6 modules), OME-Zarr, SpatialData export
├── lmd/                   # Well plates, contour processing (adaptive RDP + dilation)
├── analysis/              # 14 modules: marker classification, clustering (whole-slide + per-region), spatial networks, patterns, sampling, OmicLinker, aggregation, nuclear counting, vessel characterization, region segmentation, background correction, morphological cluster discovery, manifold sampling (LMD replicate pools)
├── visualization/         # Reusable HTML visualization: fluorescence, colors, encoding, data loading, HTML builder, graph patterns, 17 JS components
├── training/              # Classifier training: feature loading, annotation matching
├── models/                # Model registry (SAM2, ResNet, DINOv2, brightfield FMs)
├── pipeline/              # 11 modules: shm_setup, detection_loop, preprocessing, post_detection, ... (bg correction now lives in analysis/)
├── processing/            # Multi-GPU workers, deduplication, strategy factory
├── roi/                   # ROI support: pre-detection (restrict to regions) or post-detection (spatial filtering)
└── utils/                 # JSON I/O, device handling, logging, config

scripts/                   # 44 reusable CLI tools
examples/                  # Project-specific analyses by experiment
tests/                     # pytest suite (1206 tests across 59 files — run `make test`)
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

**Style:** Black (line-length 100) + Ruff. Python 3.11 (CI-tested).

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
| [docs/CLUSTER_DISCOVERY.md](docs/CLUSTER_DISCOVERY.md) | Morphological cluster discovery (HDBSCAN + Ward taxonomy) |
| [docs/MANIFOLD_SAMPLING.md](docs/MANIFOLD_SAMPLING.md) | Manifold sampling for LMD replicate pools (FPS + Voronoi + Ward) |

---

## Citation

If you use this pipeline, please cite:

```bibtex
@software{rodriguez2026xldvp,
  title = {xldvp_seg: Spatial Cell Segmentation and Deep Visual Proteomics Pipeline},
  author = {Rodriguez, Edwin},
  year = {2026},
  url = {https://github.com/peptiderodriguez/xldvp_seg},
  version = {2.0.0}
}
```

Also cite the underlying methods:
- [SAM2](https://github.com/facebookresearch/segment-anything-2) — Segment Anything Model 2 (Meta AI)
- [Cellpose](https://github.com/MouseLand/cellpose) — Cell segmentation (Stringer et al.)

## License

[MIT License](LICENSE)
