# Getting Started with xldvp_seg

A modular pipeline for detecting, classifying, and exporting cells and structures
from whole-slide CZI microscopy images. Runs on single or multiple GPU nodes,
extracts thousands of features per detection, and exports to Leica LMD format
for laser microdissection.

---

## Table of Contents

1. [Installation](#installation)
2. [Pipeline Overview](#pipeline-overview)
3. [Step 1 -- Import CZI](#step-1----import-czi)
4. [Step 2 -- Tissue Detection and Sampling](#step-2----tissue-detection-and-sampling)
5. [Step 3 -- Segmentation](#step-3----segmentation)
6. [Step 4 -- Mask Post-Processing](#step-4----mask-post-processing)
7. [Step 5 -- Feature Extraction](#step-5----feature-extraction)
8. [Step 6 -- Deduplication](#step-6----deduplication)
9. [Step 7 -- Annotation and Classification](#step-7----annotation-and-classification)
10. [Step 8 -- Spatial Analysis](#step-8----spatial-analysis)
11. [Step 9 -- LMD Export](#step-9----lmd-export)
12. [Chain Launcher (SLURM)](#chain-launcher-slurm)
13. [Interesting Side Analyses](#interesting-side-analyses)
14. [Entry Points Reference](#entry-points-reference)
15. [Output Structure](#output-structure)
16. [Coordinate System and Conventions](#coordinate-system-and-conventions)
17. [Hardware and SLURM Cluster](#hardware-and-slurm-cluster)
18. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
conda create -n mkseg python=3.11 -y && conda activate mkseg
git clone https://github.com/peptiderodriguez/xldvp_seg.git && cd xldvp_seg
./install.sh   # Auto-detects CUDA version and installs PyTorch + SAM2 + Cellpose
```

Verify the environment:

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Pipeline Overview

```
CZI Image (20-180 GB)
    |
    v
[Import CZI] ---- Load whole slide into RAM (all channels, per-channel arrays)
    |                Optional: photobleaching correction, Reinhard normalization,
    |                          flat-field illumination correction
    v
[Tissue Detection] -- Otsu-calibrated threshold on sampled tiles
    |                  Filter tiles to tissue-containing regions
    v
[Tile Sampling] ---- Process 10-100% of tissue tiles (--sample-fraction)
    |
    v
[Segmentation] ----- Strategy-specific detection (one of 7 cell types)
    |                  Multi-GPU processing (always, even with --num-gpus 1)
    |                  Optional multi-node sharding (--tile-shard INDEX/TOTAL)
    v
[Mask Post-Processing] -- Largest component, hole fill, erosion/dilation, RDP
    |
    v
[Feature Extraction] --- Up to 6,478 features per detection
    |                     Morph + channel stats + SAM2 + ResNet + DINOv2
    v
[Deduplication] -------- Mask-overlap dedup (>10% pixel overlap of smaller mask)
    |                     or --merge-shards for multi-node runs
    v
[HTML Annotation] ------ Interactive dark-themed viewer (Y/N keys, JSON export)
    |
    v
[Train Classifier] ----- Random Forest on raw features (scale-invariant)
    |                     Multiple rounds of annotation supported
    v
[Apply Classifier] ----- Score all detections (CPU-only, seconds)
    |
    v
[Spatial Analysis] ----- UMAP, HDBSCAN, Delaunay networks, community detection
    |
    v
[LMD Export] ----------- Contours -> clusters -> controls -> 384-well plate -> XML
```

---

## Step 1 -- Import CZI

The pipeline reads Zeiss CZI files via `aicspylibczi`. By default, the entire
slide is loaded into RAM (`--load-to-ram`, which is ON by default) for fast
tile access, especially important on network mounts.

**Key metadata is always read from the CZI file itself:**
- `pixel_size_um` -- physical pixel size in microns (never hardcoded)
- Channel names and fluorophore wavelengths
- Mosaic bounding box (global coordinate origin)
- Scene count (for multi-scene slides, select with `--scene N`)

```bash
# Inspect a slide without processing
python run_segmentation.py --czi-path /path/to/slide.czi --show-metadata
```

### Image Preprocessing (all optional)

Three corrections can be applied before segmentation:

| Flag | What it does |
|------|-------------|
| `--photobleaching-correction` | Row/column mean normalization to fix horizontal and vertical banding from stitched acquisitions |
| `--norm-params-file params.json` | Reinhard LAB-space normalization for cross-slide intensity harmonization (pre-computed with `compute_normalization_params.py`) |
| `--normalize-features` (default ON) | Flat-field illumination correction per-channel (morphological background subtraction) |
| `--no-normalize-features` | Disable flat-field correction; use raw intensities |

```bash
# Compute cross-slide normalization params (run once across all slides)
python compute_normalization_params.py \
    --czi-dir /path/to/all_slides/ \
    --output /path/to/norm_params.json

# Apply normalization during detection
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel 1 \
    --norm-params-file /path/to/norm_params.json \
    --photobleaching-correction
```

### Multi-Channel Loading

```bash
--all-channels          # Load all CZI channels for multi-channel feature extraction
--channel N             # Primary detection channel index (derived from CZI metadata)
--channel-names "nuclear,sma,pm,cd31"  # Human-readable names for feature prefixes
```

---

## Step 2 -- Tissue Detection and Sampling

Before segmentation, the pipeline identifies which tiles contain tissue and
optionally subsamples them:

1. **Calibration** -- Sample ~50 random tiles, compute Otsu threshold on
   intensity histograms
2. **Filtering** -- Score all tiles against the calibrated threshold, keep tiles
   with tissue above the detection threshold
3. **Sampling** -- Randomly select `--sample-fraction` of tissue tiles for
   processing (default 10%)

```bash
--sample-fraction 0.10     # Process 10% of tissue tiles (annotation runs)
--sample-fraction 1.0      # Process 100% of tissue tiles (production runs)
--skip-tissue-detection    # Process ALL tiles regardless of tissue content
--calibration-samples 50   # Number of tiles for threshold calibration
--random-seed 42           # Deterministic tile sampling
```

Tiles are generated with configurable overlap:

```bash
--tile-size 3000           # Tile dimensions in pixels (default: 3000)
--tile-overlap 0.10        # Overlap fraction between adjacent tiles (default: 10%)
```

---

## Step 3 -- Segmentation

Seven cell types are supported, each with a dedicated detection strategy:

| Cell Type | Strategy | Detection Method |
|-----------|----------|-----------------|
| `nmj` | NMJStrategy | Intensity threshold + solidity filter + watershed expansion |
| `mk` | MKStrategy | SAM2 automatic mask generation + size filter |
| `cell` | CellStrategy | Cellpose nuclei detection + SAM2 refinement |
| `vessel` | VesselStrategy | Canny edge + contour hierarchy ring detection + 3-contour system |
| `mesothelium` | MesotheliumStrategy | Ridge detection + skeleton chunking for LMD |
| `islet` | IsletStrategy | Cellpose (membrane+nuclear) + SAM2 + GMM/percentile marker classification |
| `tissue_pattern` | TissuePatternStrategy | Multi-channel summed Cellpose (no SAM2 refinement) |

### NMJ Detection

NMJs are bright, branched structures in BTX-stained muscle tissue. The pipeline
uses intensity thresholding, morphological filtering (low solidity = branched shape),
and watershed expansion.

```bash
# 10% annotation run (shows ALL candidates for labeling)
python run_segmentation.py \
    --czi-path /path/to/muscle.czi \
    --cell-type nmj \
    --channel 1 \
    --sample-fraction 0.10

# 100% production run with trained RF classifier
python run_segmentation.py \
    --czi-path /path/to/muscle.czi \
    --cell-type nmj \
    --channel 1 \
    --sample-fraction 1.0 \
    --all-channels \
    --nmj-classifier /path/to/rf_classifier.pkl \
    --prior-annotations /path/to/round1_annotations.json \
    --num-gpus 4
```

NMJ-specific parameters:

```bash
--intensity-percentile 98   # Bright region threshold percentile
--min-area 150               # Minimum area in pixels
--min-skeleton-length 30     # Minimum skeleton length in pixels
--max-solidity 0.85          # Maximum solidity (branched = low solidity)
--nmj-classifier path.pkl    # Trained RF classifier for filtering HTML display
--html-score-threshold 0.5   # Min rf_prediction to show in HTML (auto 0.0 without classifier)
--prior-annotations ann.json # Pre-load round-1 annotations into round-2 HTML
```

### MK (Megakaryocyte) Detection

Large polyploid cells detected by SAM2 automatic mask generation with size filtering.

```bash
python run_segmentation.py \
    --czi-path /path/to/bonemarrow.czi \
    --cell-type mk \
    --channel 0 \
    --mk-min-area 200 \
    --mk-max-area 2000 \
    --sample-fraction 0.10
```

### Cell (HSPC) Detection

Generic Cellpose + SAM2 pipeline for hematopoietic stem/progenitor cells.

```bash
python run_segmentation.py \
    --czi-path /path/to/bonemarrow.czi \
    --cell-type cell \
    --channel 0 \
    --sample-fraction 0.10
```

### Vessel Detection

Detects blood vessel cross-sections via SMA+ ring structures with a 3-contour system:

- **Lumen** (cyan): Inner boundary from SAM2/threshold segmentation
- **CD31** (green): Endothelial outer boundary via adaptive dilation on CD31 channel
- **SMA** (magenta): Smooth muscle ring via adaptive dilation on SMA channel, expanding from lumen
- `has_sma_ring`: True when SMA expansion > 5% larger than lumen (veins/capillaries lack SMA)

```bash
# Basic vessel detection
python run_segmentation.py \
    --czi-path /path/to/sma_stained.czi \
    --cell-type vessel \
    --channel 0 \
    --candidate-mode

# With CD31 endothelial validation and type classification
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 0 \
    --cd31-channel 1 \
    --classify-vessel-types \
    --all-channels
```

Vessel-specific parameters:

```bash
--min-vessel-diameter 10       # Min outer diameter in um
--max-vessel-diameter 1000     # Max outer diameter in um
--min-wall-thickness 2         # Min wall thickness in um
--max-aspect-ratio 4.0         # Exclude longitudinal sections
--min-circularity 0.3          # Circularity filter
--min-ring-completeness 0.5    # Fraction of SMA+ perimeter
--cd31-channel 1               # CD31 channel for validation
--candidate-mode               # Relaxed thresholds for annotation (higher recall)
--ring-only                    # Disable supplementary lumen-first pass
--classify-vessel-types        # Rule-based type classification (6 types)
--multi-marker                 # Parallel multi-marker detection (SMA, CD31, LYVE1)
--multi-scale                  # Multi-scale detection for large vessels
--scales "32,16,8,4,2"         # Downsampling factors for multi-scale
```

Vessel type classification (6 types): artery, arteriole, vein, capillary, lymphatic, collecting_lymphatic.

### Mesothelium Detection

Ridge detection for thin mesothelial ribbons, chunked to target area for LMD collection.

```bash
python run_segmentation.py \
    --czi-path /path/to/mesothelin.czi \
    --cell-type mesothelium \
    --channel 0 \
    --target-chunk-area 1500 \
    --min-ribbon-width 5 \
    --max-ribbon-width 30
```

### Islet Cell Detection

Pancreatic islet cells via Cellpose (membrane + nuclear channels) with SAM2 refinement.
Multi-channel feature extraction from all 6 channels, with GMM or percentile-based marker
classification (alpha/beta/delta by Gcg/Ins/Sst expression).

```bash
python run_segmentation.py \
    --czi-path /path/to/pancreas.czi \
    --cell-type islet \
    --membrane-channel 1 \
    --nuclear-channel 4 \
    --all-channels \
    --islet-display-channels 2,3,5 \
    --islet-marker-channels "gcg:2,ins:3,sst:5" \
    --gmm-p-cutoff 0.75 \
    --marker-top-pct 5 \
    --sample-fraction 1.0
```

Islet-specific parameters:

```bash
--membrane-channel 1          # AF633 membrane marker
--nuclear-channel 4           # DAPI
--islet-display-channels 2,3,5  # Channels for RGB HTML display
--islet-marker-channels "gcg:2,ins:3,sst:5"  # Marker name:channel pairs
--nuclei-only                 # DAPI-only mode (no membrane channel)
--marker-signal-factor 2.0    # Fold above background for positive call
--marker-top-pct 5            # Top N% percentile for positive call
--gmm-p-cutoff 0.75           # GMM posterior probability cutoff
--ratio-min 1.5               # Minimum channel ratio for marker assignment
--dedup-by-confidence         # Prefer higher-confidence detections during dedup
```

### Tissue Pattern / Brain FISH Detection

Multi-channel cell detection using summed detection channels fed into Cellpose.
No SAM2 refinement (dense tissue causes whole-region expansion). SAM2 embeddings
are still extracted per-cell for downstream features.

```bash
python run_segmentation.py \
    --czi-path /path/to/brain_fish.czi \
    --cell-type tissue_pattern \
    --tp-detection-channels 0,3 \
    --tp-nuclear-channel 4 \
    --tp-display-channels 0,3,1 \
    --all-channels \
    --sample-fraction 1.0
```

### Multi-GPU Processing

All detection runs through the multi-GPU worker (`multigpu_worker.py`), even
with `--num-gpus 1`. Tiles are dispatched round-robin to GPU workers via shared
memory (`multigpu_shm.py`).

```bash
--num-gpus 4         # Use 4 GPUs on this node
--num-gpus 1         # Single GPU — lower memory usage, slower
```

### Multi-Node Sharding

For very large slides, split detection across multiple SLURM nodes:

```bash
# Submit 4 parallel shard jobs (each processes 1/4 of tiles)
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 0/4 --resume /shared/output/dir   # node 0
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 1/4 --resume /shared/output/dir   # node 1
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 2/4 --resume /shared/output/dir   # node 2
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --tile-shard 3/4 --resume /shared/output/dir   # node 3

# After all shards complete: merge + dedup + HTML (single CPU node)
python run_segmentation.py --czi-path slide.czi --cell-type nmj \
    --resume /shared/output/dir --merge-shards
```

Sharding uses round-robin tile assignment. The merge step is checkpointed:
`merged_detections.json` -> `detections.json` (deduped) -> HTML.

---

## Step 4 -- Mask Post-Processing

Masks go through several cleanup steps before feature extraction:

| Operation | What it does | Module |
|-----------|-------------|--------|
| Largest component | Remove disconnected fragments from each mask | `segmentation/utils/mask_cleanup.py` |
| Hole filling | Fill internal holes below 50% of mask area | `segmentation/utils/mask_cleanup.py` |
| Binary smoothing | Opening/closing to clean jagged edges | Strategy-specific |
| Watershed expansion | Expand masks to capture full signal (NMJ) | `segmentation/detection/strategies/nmj.py` |
| SAM2 refinement | Point-prompt refinement of Cellpose masks | `segmentation/detection/strategies/cell.py` |

For LMD export, additional post-processing happens in `segmentation/lmd/contour_processing.py`:

| Operation | What it does |
|-----------|-------------|
| Dilation | Expand contour by +0.5 um (laser cuts outside the target) |
| RDP simplification | Ramer-Douglas-Peucker with epsilon=5 pixels (reduce point count for LMD hardware) |
| Polygon validation | Fix self-intersections via Shapely `make_valid()` |

---

## Step 5 -- Feature Extraction

Every detection has features extracted across multiple tiers:

| Feature Set | Dimensions | When Extracted |
|-------------|-----------|----------------|
| Morphological | ~78 | Always |
| Per-channel stats | ~50 per channel | When `--all-channels` with 2+ channels |
| SAM2 embeddings | 256 | Always (default) |
| ResNet50 (masked + context) | 4,096 | `--extract-deep-features` |
| DINOv2-L (masked + context) | 2,048 | `--extract-deep-features` |
| Cell-type-specific | varies | Always (vessel wall thickness, NMJ skeleton, etc.) |

**Total: up to 6,478 features** with `--extract-deep-features` and `--all-channels` on a 3-channel slide.

### Morphological Features (~78)

Area, perimeter, eccentricity, solidity, extent, circularity, aspect ratio, compactness,
equivalent diameter, major/minor axis length, orientation, Feret diameter, skeleton length,
Hu moments, Haralick texture, Gabor filter responses, LBP (local binary pattern),
gradient features, curvature, roughness, concavity, branching, and endpoints.

### Per-Channel Intensity Stats (~50 per channel)

For each channel: mean, std, min, max, median, range, percentiles (p10/p25/p75/p90),
skewness, kurtosis, entropy. Plus inter-channel ratios (e.g., `ch0_ch2_ratio`).
Feature names are prefixed with `ch0_`, `ch1_`, etc.

### SAM2 Embeddings (256D)

Image encoder features from SAM2 (always extracted). Stored as `sam2_0` through `sam2_255`.

### Deep Features (optional, 6,144D total)

ResNet-50 (pre-trained ImageNet) extracts 2,048D features from both a masked crop
and a context crop (4,096D total as `resnet_masked_*` and `resnet_context_*`).
DINOv2 ViT-L/14 extracts 1,024D from each (2,048D total as `dinov2_masked_*` and
`dinov2_context_*`).

### Channel-Corrected Features

Signal-to-noise features computed in the cell neighborhood for intensity normalization.

### Feature Set Comparison (from 844 annotated NMJ detections)

| Feature Set | n_features | F1 |
|-------------|------------|-----|
| **all_features** | 6,478 | **0.909** |
| morph + dinov2_combined | 2,126 | 0.901 |
| morphological | 78 | 0.900 |
| dinov2_context | 1,024 | 0.843 |
| resnet_context | 2,048 | 0.836 |
| sam2 | 256 | 0.728 |

Morphological features alone (78 features) achieve nearly the same F1 as all 6,478
combined. Use `train_classifier.py --feature-set` to compare subsets for your data.

---

## Step 6 -- Deduplication

Tile overlap (default 10%) creates duplicate detections near tile borders. The
deduplication module (`segmentation/processing/deduplication.py`) resolves this:

1. Load mask pixels in global coordinates for each detection
2. For each pair of nearby detections, compute pixel-level overlap
3. When overlap exceeds 10% of the smaller mask, keep the larger detection (default)
   or the higher-confidence detection (`--dedup-by-confidence`)

For multi-node runs, `--merge-shards` first combines per-node shard manifests into
a single merged detection list, then runs the standard dedup pass.

---

## Step 7 -- Annotation and Classification

### Interactive HTML Viewer

The pipeline generates a dark-themed HTML annotation interface with:
- Keyboard navigation: `Y` = Yes (positive), `N` = No (negative), arrow keys to navigate
- Dual localStorage persistence (page-specific + global keys)
- Per-page and global annotation statistics
- JSON export of all annotations
- Auto-JPEG for opaque images (5-10x smaller than PNG)

```bash
# View results (pipeline auto-starts server after detection)
python serve_html.py /path/to/output/html --port 8081

# Or manually
python -m http.server 8081 --directory /path/to/output/html
~/cloudflared tunnel --url http://localhost:8081
```

### Training an RF Classifier

After annotating detections in the HTML viewer, export the annotations JSON and train
a Random Forest classifier. RF is scale-invariant, so no StandardScaler is needed.

```bash
python train_classifier.py \
    --detections /path/to/nmj_detections.json \
    --annotations /path/to/annotations.json \
    --output-dir /path/to/classifier_output \
    --n-estimators 200 \
    --feature-set all
```

Feature set options: `morph` (78), `morph_sam2` (334), `channel_stats`, `all` (default).

### Apply Classifier (Detect Once, Classify Later)

Score all detections with a trained classifier without re-running detection.
This is CPU-only and takes seconds.

```bash
python scripts/apply_classifier.py \
    --detections /path/to/nmj_detections.json \
    --classifier /path/to/rf_classifier.pkl \
    --output /path/to/nmj_detections_scored.json
```

### Regenerate HTML

Regenerate the HTML viewer from existing detections (useful after scoring, filtering,
or changing display options). Works with any cell type.

```bash
# NMJ sorted by RF score, filtered to score >= 0.5
python scripts/regenerate_html.py \
    --output-dir /path/to/run_output \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --sort-by rf_prediction --sort-order desc \
    --score-threshold 0.5 \
    --max-samples 1500

# Islet with marker coloring
python scripts/regenerate_html.py \
    --output-dir /path/to/run_output \
    --czi-path /path/to/slide.czi \
    --cell-type islet \
    --display-channels 2,3,5 \
    --islet-marker-channels "gcg:2,ins:3,sst:5"

# Vessel with quality filter
python scripts/regenerate_html.py \
    --output-dir /path/to/run_output \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --vessel-quality-filter
```

### Multiple Rounds of Annotation

The pipeline supports iterative refinement:

1. Run 10% annotation run (all candidates, no classifier)
2. Annotate in HTML, export `annotations.json`
3. Train RF classifier
4. Apply classifier to 100% run
5. Regenerate HTML with `--score-threshold 0.5` and `--prior-annotations round1.json`
6. Review round-2, export more annotations
7. Re-train with combined annotations

---

## Step 8 -- Spatial Analysis

The `scripts/spatial_cell_analysis.py` script provides three independent analysis
modes that can be combined:

### RF Leaf-Node Embedding (`--rf-embedding`)

Uses the trained RF classifier's leaf node assignments to build a co-occurrence
matrix, then projects it via UMAP (hamming distance). Reveals which cells the
classifier considers similar.

```bash
python scripts/spatial_cell_analysis.py \
    --detections detections.json \
    --output-dir analysis/ \
    --classifier rf_classifier.pkl \
    --rf-embedding
```

Outputs: `rf_embedding_umap.png`, `rf_feature_importance.png`

### Morphological Feature UMAP (`--morph-umap`)

UMAP of interpretable morphological features (area, circularity, solidity, etc.)
with small-multiple grid coloring by each feature. Reveals morphological subtypes.

```bash
python scripts/spatial_cell_analysis.py \
    --detections detections.json \
    --output-dir analysis/ \
    --morph-umap
```

Outputs: `morph_umap_grid.png`, `morph_umap_by_score.png` (if RF scores present)

### Spatial Network Analysis (`--spatial-network`)

Builds a Delaunay-triangulation cell adjacency graph, prunes by distance, then
finds connected components and Louvain communities.

```bash
python scripts/spatial_cell_analysis.py \
    --detections detections.json \
    --output-dir analysis/ \
    --spatial-network \
    --marker-filter "ch0_mean>100"
```

Outputs: `spatial_network_summary.csv`, `spatial_components.png`, community metrics

### UMAP + HDBSCAN Clustering (`scripts/cluster_by_features.py`)

Unsupervised clustering using UMAP dimensionality reduction and HDBSCAN density
clustering. Auto-labels clusters by dominant marker channel expression (z-score).

```bash
python scripts/cluster_by_features.py \
    --detections detections.json \
    --output-dir clustering/ \
    --marker-channels "gcg:2,ins:3,sst:5" \
    --feature-groups "morph,sam2,channel"
```

Feature group options: `morph`, `shape`, `color`, `sam2`, `channel`, `deep`.

Outputs: `detections_clustered.json`, `cluster_summary.csv`, `umap_plot.png`,
`marker_violin.png`, `spatial.h5ad` (AnnData for scanpy), `spatial.csv`.

### Islet Spatial Analysis (`scripts/analyze_islets.py`)

Specialized islet-level analysis: finds islet regions from summed marker channels,
assigns cells to islets, computes per-islet morphometry, composition (Shannon
entropy, dominant type), spatial features (nearest-neighbor, radial distribution,
mantle-core index, mixing index), and atypical flags.

```bash
python scripts/analyze_islets.py \
    --run-dir /path/to/islet_output \
    --czi-path /path/to/slide.czi \
    --buffer-um 25 --min-cells 5
```

Outputs: `islet_summary.csv`, `islet_detections.json`, `html/islet_analysis.html`

### MK Maturation Analysis (`scripts/maturation_analysis.py`)

4-phase MK maturation staging using nuclear deep features:
1. Load + filter + dedup
2. Nuclear segmentation + SAM2/ResNet deep features + PCA
3. Cluster on nuclear PCA features
4. Validate, pseudotime, group comparison plots

---

## Step 9 -- LMD Export

Export detection contours to Leica LMD XML format for laser microdissection.

### Full Workflow

```bash
# 1. Convert CZI to OME-Zarr for Napari viewing
python scripts/czi_to_ome_zarr.py slide.czi slide.zarr

# 2. Place reference crosses in Napari (3+ calibration points)
python scripts/napari_place_crosses.py slide.zarr --output crosses.json

# 3. Cluster detections for well assignment
python scripts/cluster_detections.py \
    --detections nmj_detections.json \
    --pixel-size 0.1725 \
    --area-min 375 --area-max 425 \
    --dist-round1 500 --dist-round2 1000 \
    --min-score 0.5 \
    --output nmj_clusters.json

# 4. Export to LMD XML
python run_lmd_export.py \
    --detections nmj_detections_scored.json \
    --cell-type nmj \
    --crosses crosses.json \
    --clusters nmj_clusters.json \
    --tiles-dir /path/to/tiles \
    --output-dir lmd_export \
    --export --generate-controls

# 5. View export overlaid on slide
python scripts/napari_view_lmd_export.py \
    --zarr slide.zarr \
    --lmd-dir lmd_export
```

### LMD Pipeline Details

| Step | Description |
|------|-------------|
| Contour extraction | Extract outer contour from H5 masks per detection |
| Dilation | Expand contour by +0.5 um so the laser cuts outside the cell |
| RDP simplification | Reduce vertex count (epsilon=5 px) for LMD hardware limits |
| Two-stage clustering | Round 1: 500 um radius, Round 2: 1000 um radius. Target 375-425 um^2 per cluster |
| Singles | Unclustered detections become individual wells |
| Controls | 100 um offset in 8 compass directions. Cluster controls preserve spatial arrangement |
| Well assignment | 384-well plate, serpentine ordering: B2 -> B3 -> C3 -> C2 (max 308 wells) |
| Capacity check | Early warning if detections exceed plate capacity (before expensive processing) |
| XML export | Leica LMD XML format with reference crosses for slide alignment |

Napari visualization uses 4 colors: singles / controls / clusters / cluster-controls.

---

## Chain Launcher (SLURM)

The `slurm/launch_pipeline.sh` script orchestrates the full pipeline with SLURM
dependency chaining (`--dependency=afterok`). Each step waits for the previous one
to succeed.

### Available Steps

| Step | Script | Resources |
|------|--------|-----------|
| `detect` | `run_segmentation.py` | GPU (multi-node sharding) |
| `merge` | `run_segmentation.py --merge-shards` | CPU (multi-node only) |
| `html` | `scripts/regenerate_html.py` | CPU |
| `score` | `scripts/apply_classifier.py` | CPU |
| `analysis` | `scripts/spatial_cell_analysis.py` | CPU |
| `lmd` | `run_lmd_export.py` | CPU |

### Examples

```bash
# Full NMJ pipeline: detect on 4 nodes -> merge -> annotation HTML
bash slurm/launch_pipeline.sh \
    --czi /path/to/slide.czi \
    --cell-type nmj --channel 1 \
    --nodes 4 --sample-fraction 1.0 \
    --steps detect,merge,html

# Post-annotation: score + analysis + LMD (reuses existing output)
bash slurm/launch_pipeline.sh \
    --output-dir /path/to/existing/run \
    --czi /path/to/slide.czi \
    --cell-type nmj \
    --classifier /path/to/rf_classifier.pkl \
    --annotations /path/to/annotations.json \
    --crosses /path/to/crosses.json \
    --steps score,analysis,lmd

# Single-node vessel detection + HTML
bash slurm/launch_pipeline.sh \
    --czi /path/to/slide.czi \
    --cell-type vessel --channel 0 \
    --nodes 1 --gpus 4 --partition p.hpcl93 \
    --steps detect,html

# Dry run (print commands without submitting)
bash slurm/launch_pipeline.sh \
    --czi /path/to/slide.czi \
    --cell-type nmj --channel 1 \
    --nodes 4 --steps detect,merge,html \
    --dry-run
```

### Chain Launcher Options

```bash
# Detection
--czi PATH                   # CZI slide path (required for detect)
--cell-type TYPE             # nmj|mk|cell|vessel|islet|tissue_pattern|mesothelium
--channel N                  # Detection channel index
--nodes N                    # Number of SLURM nodes for detect (default: 1)
--gpus N                     # GPUs per node (default: 4)
--sample-fraction F          # Tile sampling fraction (default: 1.0)
--partition-gpu NAME         # GPU partition (default: p.hpcl93)
--partition-cpu NAME         # CPU partition (default: p.hpcl8)
--extra-seg-args "ARGS"      # Extra arguments for run_segmentation.py

# Output
--output-dir PATH            # Reuse existing output directory
--output-base PATH           # Base directory for new timestamped output

# Classification
--classifier PATH            # RF classifier .pkl (required for score step)
--annotations PATH           # Annotations JSON
--score-threshold F          # RF score filter (default: 0.5)

# HTML
--max-samples N              # Max HTML samples (default: 1500)
--display-channels "1,2,0"   # Channel indices for RGB display

# Analysis
--analysis-modes LIST        # Comma-separated: rf-embedding,morph-umap,spatial-network

# LMD
--crosses PATH               # Reference crosses JSON (required for lmd step)
--clusters PATH              # Biological clusters JSON

# General
--seed N                     # Random seed (default: 42)
--dry-run                    # Print commands without submitting
```

---

## Interesting Side Analyses

The pipeline's rich feature extraction and spatial capabilities enable many
downstream analyses beyond basic cell detection:

### 1. Vessel Identification via SMA+ Cell Networks

Use `--spatial-network` mode with `--marker-filter "ch0_mean>100"` on SMA+ cells.
Delaunay-connected components of high-SMA cells reveal vessel-like spatial
aggregations, even without explicit ring detection.

### 2. Cell Type Co-Localization

Delaunay-based proximity analysis between different marker-positive populations.
Run spatial network on all cells, then examine edge connections between
populations to quantify co-localization and spatial mixing.

### 3. Expression Profiling

Per-channel intensity distributions from the multi-channel feature extraction.
Cross-marker correlations between any pair of channels using the `ch*_mean`,
`ch*_median`, and `ch*_percentile_*` features stored per-detection.

### 4. Spatial Clustering (Morphological Subtypes)

HDBSCAN on UMAP embeddings (`scripts/cluster_by_features.py`) to find
morphological subtypes within a population. Auto-labels clusters by dominant
marker channel using z-scored mean expression.

### 5. Tissue Zone Analysis

Expression groups from HDBSCAN clustering mapped back to spatial coordinates.
Identify spatially coherent zones with distinct expression profiles using the
`spatial.csv` output.

### 6. Marker-Positive Cell Selection

Two methods for identifying marker-positive cells:
- **Percentile**: Top N% by channel intensity (`--marker-top-pct 5`)
- **GMM**: Gaussian Mixture Model posterior probability (`--gmm-p-cutoff 0.75`)

Both methods work per-channel and are implemented in the islet strategy but
applicable to any multi-channel detection.

### 7. Cross-Slide Normalization

Reinhard LAB-space normalization for consistent intensity across slides.
Two-phase: `compute_normalization_params.py` computes global target statistics
from all slides (with outlier rejection), then `--norm-params-file` applies the
normalization during detection.

### 8. Feature Importance Analysis

RF feature importance + RFECV for identifying which features are most
discriminative. `train_classifier.py --feature-set` compares subsets (morph,
morph_sam2, channel_stats, all). The spatial analysis script saves
`rf_feature_importance.png` with top-20 Gini importances.

### 9. Community Detection on Spatial Cell Graphs

Louvain communities on Delaunay cell adjacency graphs reveal tissue
organizational units. Connected components identify spatially isolated groups.
Per-component metrics: cell count, mean degree, convex hull area, density,
graph diameter.

### 10. Contour-Based Vessel Morphometry

Wall thickness at 36 angles, lumen area, wall area, outer/inner diameter,
aspect ratio, orientation, ring completeness. The 3-contour system (lumen/CD31/SMA)
quantifies endothelial and smooth muscle contributions separately.

---

## Entry Points Reference

| Script | Purpose |
|--------|---------|
| `run_segmentation.py` | Unified entry point: detect, dedup, HTML, CSV (recommended) |
| `run_lmd_export.py` | Export to Leica LMD format (any cell type) |
| `train_classifier.py` | Train RF classifier from annotated detections |
| `scripts/apply_classifier.py` | Score existing detections with trained classifier (no re-detection) |
| `scripts/regenerate_html.py` | Regenerate HTML viewer from saved detections (all cell types) |
| `scripts/czi_to_ome_zarr.py` | Convert CZI to OME-Zarr with pyramids |
| `scripts/napari_place_crosses.py` | Interactive reference cross placement |
| `scripts/cluster_detections.py` | Biological clustering for LMD well assignment |
| `scripts/cluster_by_features.py` | UMAP + HDBSCAN clustering with auto-labeling |
| `scripts/spatial_cell_analysis.py` | RF embedding, morph UMAP, spatial network analysis |
| `scripts/analyze_islets.py` | Islet-level spatial analysis with composition metrics |
| `scripts/maturation_analysis.py` | MK maturation staging via nuclear deep features |
| `scripts/napari_view_lmd_export.py` | View LMD export overlaid on slide |
| `serve_html.py` | HTTP server + Cloudflare tunnel for remote HTML viewing |
| `compute_normalization_params.py` | Pre-compute Reinhard LAB stats across slides |
| `slurm/launch_pipeline.sh` | SLURM dependency chain launcher |

---

## Output Structure

Each run creates a timestamped output directory:

```
output_dir/                            # e.g., nmj_output/slide_20260227_143000_100pct/
|-- {cell_type}_detections.json        # All detections with UIDs, coordinates, features
|-- {cell_type}_coordinates.csv        # Quick export: center, area, features
|-- tiles/
|   |-- tile_{x}_{y}/
|   |   |-- {cell_type}_masks.h5       # Per-tile label arrays (LZ4 compressed)
|   |   |-- {cell_type}_features.json  # Per-tile detection features
|   |-- ...
|-- html/
|   |-- index.html                     # Main viewer
|   |-- page_*.html                    # Paginated detection pages
|-- shard_*_manifest.json              # Per-shard tile lists (multi-node only)
|-- merged_detections.json             # Pre-dedup merged shards (multi-node only)
```

### Detection JSON Format

Each detection is a dictionary:

```json
{
  "uid": "slidename_nmj_45678_12345",
  "id": "nmj_3",
  "tile_origin": [3000, 6000],
  "center": [150, 200],
  "global_center": [3150, 6200],
  "global_center_um": [543.4, 1069.5],
  "area_px": 4523,
  "area_um2": 219.4,
  "mask_label": 3,
  "tile_mask_label": 3,
  "pixel_size_um": 0.1725,
  "rf_prediction": 0.87,
  "features": {
    "pixel_size_um": 0.1725,
    "area": 4523,
    "area_um2": 219.4,
    "solidity": 0.89,
    "eccentricity": 0.34,
    "circularity": 0.72,
    "ch0_mean": 145.3,
    "ch1_mean": 2034.7,
    "sam2_0": 0.0234,
    "...": "..."
  }
}
```

**Important**: Every detection carries `pixel_size_um` in its features dict,
ensuring downstream tools (LMD export, clustering, analysis) always have the
correct physical scale without relying on hardcoded values.

---

## Coordinate System and Conventions

**All coordinates are [x, y] (horizontal, vertical).**

- **UID format**: `{slide}_{celltype}_{global_x}_{global_y}` -- coordinates
  ensure uniqueness across tiles
- **Mosaic origin**: CZI tiles use global coordinates from the microscope stage.
  RAM arrays are 0-indexed from the mosaic bounding box origin.
  - `loader.get_tile()` handles the offset correctly
  - Direct `all_channel_data[ch]` indexing must subtract `(x_start, y_start)`
- **pixel_size_um**: Always read from CZI metadata. Every detection stores this
  value in its features dict for downstream tools.
- **Output directories**: Timestamped by default
  (`{slide}_{YYYYMMDD_HHMMSS}_{pct}pct`)
- **Output files**: Key outputs (detections JSON, CSV, LMD XML) are also
  timestamped (e.g., `nmj_detections_20260227_143052.json`) with a symlink
  from the canonical name (e.g., `nmj_detections.json` -> latest). This
  preserves history while keeping downstream tools working via the symlink.

---

## Hardware and SLURM Cluster

### Partitions

| Partition | Nodes | CPUs/node | RAM/node | GPUs/node | Use |
|-----------|-------|-----------|----------|-----------|-----|
| `p.hpcl8` | 55 | 24 | 380 GB | 2x RTX 5000 | Interactive dev, CPU-only jobs |
| `p.hpcl93` | 19 | 256 | 760 GB | 4x L40S | Heavy GPU batch jobs (requires `--gres=gpu:`) |

Time limit: 42 days on both partitions.

### Interactive Session

```bash
srun --partition=p.hpcl8 --exclusive --cpus-per-task=24 --mem=350G \
    --gres=gpu:rtx5000:2 --time=24:00:00 --job-name=dev --pty bash
```

### Resource Guidelines

| Slide Size | RAM | GPUs | Time (single node) |
|------------|-----|------|--------------------|
| 20-25 GB CZI | 64 GB | 1-2 RTX 5000 | 2-6 hours |
| 50-100 GB CZI | 200 GB | 2-4 L40S | 4-12 hours |
| 100-180 GB CZI | 500 GB + multi-node | 4 L40S x N nodes | 6-24 hours |

---

## Troubleshooting

### CUDA Out of Memory

Reduce GPU count or tile size:

```bash
python run_segmentation.py ... --num-gpus 1 --tile-size 2000
```

### HDF5 LZ4 Plugin Errors

The pipeline uses LZ4 compression for mask HDF5 files. If you see HDF5 filter
errors, make sure `hdf5plugin` is importable:

```bash
python -c "import hdf5plugin; print('OK')"
```

### Network Mount Timeouts

Socket timeout is automatic (60s). Check connectivity:

```bash
ls /mnt/your_mount/
```

Use `--load-to-ram` (default ON) to load CZI channels into RAM once and avoid
repeated network I/O.

### SAM2 Issues

- **Boolean mask**: SAM2 requires `mask.astype(bool)`, not integer masks
- **`_orig_hw`**: Access via `sam2_predictor._orig_hw[0]` (list of tuple, not tuple)

### Empty Results

- Verify channel index matches fluorescence target (`--show-metadata`)
- Check that `--sample-fraction` is not too low for sparse structures
- Ensure tissue detection threshold is appropriate (try `--skip-tissue-detection` to test)

### Slow Processing

- Use `--load-to-ram` for network mounts (default)
- Increase `--num-gpus` for more parallelism
- Use multi-node sharding for very large slides

### Port Conventions

| Port | Use |
|------|-----|
| 8080 | MK/HSPC viewer |
| 8081 | NMJ viewer (default for `run_segmentation.py`) |
| 8082 | Vessel viewer |
