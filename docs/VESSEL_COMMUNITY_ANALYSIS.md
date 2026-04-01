# Vessel Analysis

Two complementary tools for vessel analysis from classified cell detections.

## Tool 1: Graph Topology Vessel Detection (Recommended)

**Script:** `scripts/detect_vessel_structures.py`

Identifies individual vessel structures from marker-positive cells (SMA+, CD31+, LYVE1+) using graph topology (ring_score, arc_fraction, linearity) and geometric/PCA metrics (circularity, hollowness, elongation). Classifies morphology (ring, arc, strip, cluster), computes vessel morphometry (diameter, lumen, wall extent), analyzes spatial marker layering (Mann-Whitney U), and assigns vessel types (artery, vein, lymphatic, capillary).

```bash
python scripts/detect_vessel_structures.py \
    --detections cell_detections_classified.json \
    --marker-filter "SMA_class==positive" \
    --marker-filter "CD31_class==positive" \
    --marker-logic or \
    --radius 50 --min-cells 5 \
    --output-dir vessel_structures/ --output-prefix vessel
```

Key parameters:
- `--marker-filter`: repeat for each vessel marker (OR logic by default)
- `--radius`: connection distance in µm (30-75 typical)
- `--min-cells`: minimum cells per vessel structure (5-15)
- `--linearity-threshold`: strip detection cutoff (default 3.0)
- `--ring-threshold`: ring score cutoff (default 0.5)

Outputs: `cell_detections_vessel_tagged.json`, `cell_detections_vessel_only.json`, `vessel_structures.json`, `vessel_structures.csv`.

Uses the shared `segmentation.utils.graph_topology` module (also used by `detect_curvilinear_patterns.py` for mesothelium strip detection).

## Tool 2: Vessel Community Analysis (Multi-Scale)

**Script:** `scripts/vessel_community_analysis.py`

## Overview

This workflow takes cell detections that have been classified by marker expression (SMA, CD31) and groups marker-positive cells into vessel structures at multiple spatial scales. Each structure is classified by morphology (ring, arc, linear, cluster) and vessel type (artery-like, vein-like, capillary-like, endothelial network).

## Prerequisites

1. **Cell detections** from the pipeline (`cell_detections.json`)
2. **Marker classification** via `classify_markers.py` → `cell_detections_classified.json`
   - Requires SMA and CD31 marker labels
   - Uses SNR (signal-to-noise ratio) values for classification

## Quick Start

```bash
# 1. Classify markers (if not already done)
python scripts/classify_markers.py \
    --detections cell_detections.json \
    --marker-wavelength 647,555 \
    --czi-path slide.czi \
    --marker-name SMA,CD31 \
    --method otsu

# 2. Run vessel community analysis
python scripts/vessel_community_analysis.py \
    --detections cell_detections_classified.json \
    --output-dir vessel_analysis/ \
    --generate-viewer --run-squidpy --run-leiden
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--detections` | (required) | Path to classified detections JSON |
| `--marker-field` | `marker_profile` | Feature field with combined marker label |
| `--positive-values` | `SMA+/CD31-,SMA-/CD31+,SMA+/CD31+` | Which profiles count as positive |
| `--marker-names` | `SMA,CD31` | Marker names (maps to `{name}_snr`, `{name}_class`) |
| `--radii` | `25,50,100,200` | Spatial radii in microns |
| `--best-radius` | `50` | Primary radius for cell assignment |
| `--min-cells` | `3` | Minimum cells per structure |
| `--output-dir` | (same as input) | Output directory |
| `--generate-viewer` | off | Generate interactive HTML viewer |
| `--run-squidpy` | off | Run squidpy spatial statistics |
| `--run-leiden` | off | Leiden clustering on full feature space (morph + SNR + SAM2) |
| `--leiden-resolution` | `0.5` | Leiden resolution parameter (higher = more clusters) |
| `--snr-channels` | auto-detect | Channel indices for SNR keys (e.g. `1,3` → `ch1_snr`, `ch3_snr`) |
| `--viewer-group-field` | `vessel_type` | Field to color by in viewer |

## How It Works

### 1. Multi-Scale Connected Components

For each radius r in [25, 50, 100, 200] um:
- Build KD-tree from positive cell coordinates
- Find all cell pairs within distance r
- Build sparse adjacency matrix
- Extract connected components (scipy.sparse.csgraph)
- Filter components with ≥ min_cells

Smaller radii capture fine structures (capillaries), larger radii merge nearby cells into larger structures (arteries, veins).

### 2. Morphology Classification

Each component is classified using the same thresholds as the spatial viewer:

| Pattern | Criteria |
|---------|----------|
| **linear** | PCA elongation > 4, no curvature |
| **arc** | PCA elongation > 3, has curvature (R² > 0.3) |
| **ring** | circularity > 0.65, hollowness > 0.55, elongation < 3 |
| **cluster** | default (all others) |

### 3. Vessel Type Inference (SNR-Informed)

Uses marker composition + morphology + mean SNR:

| Vessel Type | Criteria |
|-------------|----------|
| **artery_like** | Ring morphology, SMA fraction > 50% |
| **vein_like** | Cluster/arc, SMA fraction > 30%, high mean SMA SNR |
| **capillary_like** | Small (<10 cells), CD31 fraction > 50% |
| **endothelial_network** | Linear, CD31 fraction > 50% |
| **unclassified** | None of the above |

### 4. Hierarchical Nesting

Fine-scale structures are mapped to their coarse-scale parents by cell index overlap. This reveals how capillary-scale structures nest within larger vascular zones.

## Output Files

| File | Contents |
|------|----------|
| `vessel_structures.csv` | One row per structure: morphology, vessel type, composition, SNR, hull area, centroid |
| `cell_detections_vessel_analysis.json` | All detections enriched with `vessel_community_id`, `vessel_type`, `vessel_morphology`, `vessel_scale_um` |
| `cell_detections_slim.json` | Lightweight JSON for viewer (coords + group fields, ~72MB vs 4GB) |
| `leiden/umap_leiden.png` | UMAP colored by Leiden cluster + marker profile |
| `leiden/spatial_leiden.png` | Spatial scatter colored by Leiden cluster |
| `leiden/cluster_composition.png` | Bar chart: marker composition per Leiden cluster |
| `leiden/vessel_adata.h5ad` | AnnData with features, embeddings, Leiden labels (for scanpy) |
| `squidpy/nhood_enrichment.png` | Do SMA+ cells preferentially neighbor CD31+ cells? |
| `squidpy/co_occurrence.png` | Distance-dependent co-occurrence of marker types |
| `squidpy/ripley_L.png` | Spatial clustering (L > 0) or repulsion (L < 0) per cell type |

## SNR vs Background-Corrected Intensity

This workflow uses **SNR (signal-to-noise ratio)** throughout instead of background-corrected intensity:

- `ch*_mean` (bg-corrected): clamps to 0 when background > signal — most cells read 0.0
- `ch*_snr = raw / background`: ratio, no clamping, scale-invariant
- SNR > 1 means signal is above background; SNR = 0.5 means signal is half the background level

The `classify_markers.py` script stores `{marker}_snr` for each cell, and `vessel_community_analysis.py` uses mean SNR per structure to distinguish artery-like (high SMA SNR) from vein-like (moderate SMA SNR) structures.

## Squidpy Integration

With `--run-squidpy`, the script builds an AnnData object from positive cells and runs:

- **Neighborhood enrichment**: Tests whether cell types co-localize more than expected by chance
- **Co-occurrence**: Distance-dependent co-occurrence patterns
- **Ripley's L**: Tests for spatial clustering (L > 0) or repulsion (L < 0)

The feature matrix uses SNR values (SMA_snr, CD31_snr) rather than raw intensities.

## Leiden Clustering

With `--run-leiden`, the script clusters **all** cells (not just positive) on the full feature space:

- **Features**: morphological (area, perimeter, eccentricity, etc.) + channel SNR/raw/std/max/median + SAM2 embeddings = ~295 features
- **Workflow**: scale → PCA (50 components) → KNN neighbors (k=15) → Leiden community detection
- **Resolution**: `--leiden-resolution 0.5` (default). Higher = more fine-grained clusters.
- **Output**: UMAP plot, spatial scatter, cluster composition bar chart, h5ad file

This reveals data-driven cell subtypes beyond binary marker classification — e.g., distinguishing pericytes from smooth muscle, or different endothelial phenotypes. The `leiden_cluster` label is added to each detection's features and can be used as `--viewer-group-field leiden_cluster` in the spatial viewer.

**Memory note**: Leiden on 306K cells requires ~50-100GB RAM. Run as a SLURM job on p.hpcl8 (370G nodes), not on the login node.
