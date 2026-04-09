# Output Formats

## Directory Structure

A typical pipeline run produces the following output:

```
output_dir/
  slide_name/
    YYYYMMDD_HHMMSS/               # timestamped run directory
      tiles/                        # per-tile intermediate data
        tile_0_0/
          masks.h5                  # HDF5 segmentation masks (LZ4 compressed)
          detections.json           # per-tile detections
          html_cache/               # pre-rendered HTML crops
      cell_detections_merged.json   # all shard detections concatenated
      cell_detections.json          # deduplicated final detections
      cell_detections_postdedup.json  # with contours + bg correction
      cell_detections.csv           # flat CSV export
      cell_detections.html          # interactive HTML viewer
      slide.ome.zarr/               # OME-Zarr pyramid
      cell_spatialdata.zarr/        # SpatialData zarr store
      pipeline_config.json          # pipeline configuration used
      summary.json                  # run metadata + statistics
```

## Detection JSON Schema

Each detection is a dictionary with **two levels of nesting**:

- **Top-level keys**: identity and geometry (`uid`, `cell_type`, `global_center`, `global_center_um`, `tile_origin`, `mask_label`, `rf_prediction`, `contour_px`, `contour_um`)
- **`features` sub-dict**: all computed features (`area`, `circularity`, `ch0_mean`, `SMA_class`, `marker_profile`, etc.)

When writing custom analysis code, access features via `det["features"]["area"]`, not `det["area"]`.

### Required Fields (top-level, always present)

| Field | Type | Description |
|-------|------|-------------|
| `uid` | string | Unique ID: `{slide}_{celltype}_{x}_{y}` |
| `cell_type` | string | Detection strategy name |
| `global_center` | [float, float] | Centroid in pixel coordinates [x, y] |
| `global_center_um` | [float, float] | Centroid in micrometers [x, y] |
| `tile_origin` | [int, int] | Tile origin in global pixels |
| `mask_label` | int | Label index in the tile's HDF5 mask |

### Morphological Features (in `features` sub-dict)

| Field | Type | Description |
|-------|------|-------------|
| `area` | float | Area in pixels |
| `area_um2` | float | Area in square micrometers |
| `perimeter` | float | Perimeter in pixels |
| `circularity` | float | 4 * pi * area / perimeter^2 |
| `solidity` | float | Area / convex hull area |
| `aspect_ratio` | float | Major axis / minor axis |
| `eccentricity` | float | Eccentricity of fitted ellipse |

### Channel Features (in `features` sub-dict, with `--all-channels`)

For each channel N:

| Field | Type | Description |
|-------|------|-------------|
| `ch{N}_mean` | float | Mean intensity in mask |
| `ch{N}_median` | float | Median intensity in mask |
| `ch{N}_std` | float | Standard deviation |
| `ch{N}_median_raw` | float | Median before background correction |
| `ch{N}_background` | float | Local background estimate |
| `ch{N}_snr` | float | Signal-to-noise ratio (median_raw / background) |

### Classification Fields (after scoring)

| Field | Location | Type | Description |
|-------|----------|------|-------------|
| `rf_prediction` | top-level | float | Random forest confidence (0.0 to 1.0) |

### Marker Fields (in `features` sub-dict, after marker classification)

| Field | Type | Description |
|-------|------|-------------|
| `{marker}_class` | string | "positive" or "negative" |
| `{marker}_value` | float | Intensity value used for classification (background-subtracted when bg correction is enabled; raw otherwise) |
| `{marker}_raw` | float | Raw (uncorrected) intensity value (only when bg correction is enabled) |
| `{marker}_background` | float | Per-cell background estimate (only when bg correction is enabled) |
| `{marker}_snr` | float | Signal-to-noise ratio: raw / background (only when bg correction is enabled) |
| `marker_profile` | string | Combined profile, e.g., "SMA+/CD31-" |

The threshold used for each marker is stored in the summary dict returned by `classify_single_marker()`, not per-detection. Classification methods: `snr` (default, median-based SNR >= 1.5), `otsu` (automatic threshold), `otsu_half` (permissive), `gmm` (2-component Gaussian mixture with BIC model selection — automatically returns all-negative when data is unimodal).

### Nuclear Counting Fields (in `features` sub-dict, with `--count-nuclei`)

| Field | Type | Description |
|-------|------|-------------|
| `n_nuclei` | int | Number of nuclei assigned to this cell (overlap-based) |
| `nuclear_area_um2` | float | Total nuclear overlap area within cell (µm²) |
| `nuclear_area_fraction` | float | Overlap area / cell area (N:C ratio, clamped ≤ 1.0) |
| `largest_nucleus_um2` | float | Full area of largest nucleus (for ploidy estimation) |
| `nuclear_solidity` | float | Mean solidity of nuclear objects |
| `nuclear_eccentricity` | float | Mean eccentricity of nuclear objects |

Per-nucleus details are in the top-level `nuclei` list. Each entry has `area_um2` (full nuclear area for ploidy), `overlap_area_um2` (area within assigned cell), `solidity`, `eccentricity`, `perimeter_um`, `mean_intensity`, and `centroid_local`.

### Differential Analysis Fields (from `OmicLinker.differential_features()`)

| Field | Type | Description |
|-------|------|-------------|
| `feature` | string | Feature name |
| `statistic` | float | Test statistic (Mann-Whitney U or t) |
| `p_value` | float | Raw p-value |
| `p_adjusted` | float | Benjamini-Hochberg FDR-adjusted p-value |
| `effect_size` | float | Cohen's d (capped at ±10 to prevent inflation from near-zero variance) |
| `mean_diff` | float | Mean difference (group A - group B) |
| `mean_a` | float | Mean in group A |
| `mean_b` | float | Mean in group B |
| `n_a` | int | Sample size in group A |
| `n_b` | int | Sample size in group B |

### Contour Fields (top-level, after post-dedup)

| Field | Type | Description |
|-------|------|-------------|
| `contour_px` | list | Contour points in pixels [[x,y], ...] |
| `contour_um` | list | Contour points in micrometers |

## CSV Export

The CSV file (`cell_detections.csv`) is a flat table with one row per detection.
All features and metadata fields are columns. Nested fields (contours, lists)
are excluded.

## AnnData Layout

When using `slide.to_anndata()`:

| Slot | Content |
|------|---------|
| `X` | Feature matrix (morphological + channel features) |
| `obs` | Per-cell metadata: `uid`, `cell_type`, `rf_prediction`, marker classes, cluster labels |
| `obsm["spatial"]` | Spatial coordinates as Nx2 array (x, y in micrometers) |
| `var` | Feature metadata with `feature_group` column (morph/channel/sam2/deep) |
| `uns` | Provenance: pipeline version, parameters, timestamp, pixel_size_um |

## OME-Zarr

The `slide.ome.zarr/` directory contains a multi-resolution pyramid of the
full slide image in OME-Zarr format. Generated automatically from shared
memory during pipeline execution. Use `--no-zarr` to skip or `--force-zarr`
to overwrite.

## SpatialData

The `cell_spatialdata.zarr/` store contains:

- **images**: OME-Zarr reference to the slide pyramid
- **shapes**: GeoDataFrame of detection contours (polygons)
- **table**: AnnData with features linked to shapes

Compatible with squidpy spatial analysis (`--run-squidpy`).

## Contour Viewer HTML

`scripts/generate_contour_viewer.py` produces a self-contained HTML file for
visualizing contour overlays on CZI fluorescence. The viewer embeds:

- CZI fluorescence thumbnail as base64-encoded channel images
- Contour coordinate data in binary-encoded Float32 arrays
- Per-contour metadata (group assignment, morphometry, vessel type, etc.)
- Composable Canvas 2D JS components from `xldvp_seg/visualization/js/`

Features: pan/zoom with RAF batching, viewport culling for 50K+ contours,
per-group color-coded contour toggling, R/G/B fluorescence channel toggle,
click-to-inspect metadata panel. Output is a single `.html` file with no
external dependencies.

```bash
python scripts/generate_contour_viewer.py \
    --contours vessel_lumens.json \
    --group-field vessel_type \
    --czi-path slide.czi \
    --display-channels 1,3,0 \
    --channel-names "SMA,CD31,nuc" \
    --title "Vessel Lumen Detection" \
    --output vessel_viewer.html
```
