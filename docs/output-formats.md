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
      detection_summary.json        # run metadata + statistics
```

## Detection JSON Schema

Each detection is a dictionary with these fields:

### Required Fields (always present)

| Field | Type | Description |
|-------|------|-------------|
| `uid` | string | Unique ID: `{slide}_{celltype}_{x}_{y}` |
| `cell_type` | string | Detection strategy name |
| `global_center` | [float, float] | Centroid in pixel coordinates [x, y] |
| `global_center_um` | [float, float] | Centroid in micrometers [x, y] |
| `tile_origin` | [int, int] | Tile origin in global pixels |
| `mask_label` | int | Label index in the tile's HDF5 mask |

### Morphological Features

| Field | Type | Description |
|-------|------|-------------|
| `area` | float | Area in pixels |
| `area_um2` | float | Area in square micrometers |
| `perimeter` | float | Perimeter in pixels |
| `circularity` | float | 4 * pi * area / perimeter^2 |
| `solidity` | float | Area / convex hull area |
| `aspect_ratio` | float | Major axis / minor axis |
| `eccentricity` | float | Eccentricity of fitted ellipse |

### Channel Features (with `--all-channels`)

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

| Field | Type | Description |
|-------|------|-------------|
| `rf_prediction` | float | Random forest confidence (0.0 to 1.0) |

### Marker Fields (after marker classification)

| Field | Type | Description |
|-------|------|-------------|
| `{marker}_class` | string | "positive" or "negative" |
| `{marker}_value` | float | Raw intensity value used |
| `{marker}_threshold` | float | Threshold applied |
| `marker_profile` | string | Combined profile, e.g., "SMA+/CD31-" |

### Contour Fields (after post-dedup)

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
