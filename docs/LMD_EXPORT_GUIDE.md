# LMD Export Guide

Export annotated cell detections to Leica Laser Microdissection (LMD) format for automated tissue collection.

## Overview

After running segmentation and annotating candidates (yes/no), this tool exports the positive detections to Leica LMD XML format. The workflow supports:

- **Reference cross placement** - Interactive HTML tool for setting calibration points
- **Spatial clustering** - Group nearby detections into wells (e.g., 100 cells per well)
- **384/96-well plate support** - Automatic well assignment (A1, B1, C1, ...)

## Prerequisites

1. **py-lmd library** installed:
   ```bash
   pip install py-lmd
   # or from source:
   pip install git+https://github.com/MannLabs/py-lmd.git
   ```

2. **Detection JSON** from segmentation pipeline (`*_detections.json`)

3. **Annotations JSON** from HTML annotation interface (optional but recommended)

## Workflow

### Step 1: Generate Cross Placement HTML

Reference crosses are calibration points that align the microscope stage coordinates with your image. Place them at identifiable landmarks visible under the LMD microscope.

```bash
python run_lmd_export.py \
    --detections /path/to/detections.json \
    --annotations /path/to/annotations.json \
    --output-dir /path/to/output \
    --generate-cross-html
```

This creates `place_crosses.html`. Open it in a browser:

1. Green dots show your detection locations
2. Click to place reference crosses (minimum 3 required)
3. Place crosses at tissue corners or distinctive features
4. Click "Save Crosses" to download `reference_crosses.json`

**Tips for cross placement:**
- Choose landmarks visible in both the overview image and under the LMD microscope
- Spread crosses across the tissue area for better calibration
- Avoid placing crosses directly on cells you want to collect

### Step 2: Export to LMD Format

With crosses placed, export to Leica XML:

```bash
python run_lmd_export.py \
    --detections /path/to/detections.json \
    --annotations /path/to/annotations.json \
    --crosses reference_crosses.json \
    --output-dir /path/to/output \
    --export
```

## Spatial Clustering

For collecting many cells, group them into wells automatically. Each cluster becomes one well on the collection plate.

```bash
python run_lmd_export.py \
    --detections detections.json \
    --annotations annotations.json \
    --crosses reference_crosses.json \
    --output-dir output/lmd \
    --export \
    --cluster-size 100 \
    --plate-format 384
```

### Clustering Options

| Option | Description |
|--------|-------------|
| `--cluster-size N` | Target detections per well (e.g., 100) |
| `--plate-format {384,96}` | Well plate format |
| `--clustering-method {greedy,kmeans,dbscan}` | Algorithm for grouping |

### Clustering Methods

- **greedy** (default) - Builds clusters by adding nearest neighbors iteratively. Produces compact, spatially coherent clusters.

- **kmeans** - Standard K-means clustering. Good for evenly distributed detections.

- **dbscan** - Density-based clustering. Good when detections form natural groups with gaps between them.

### Well Assignment

Clusters are sorted spatially (top-to-bottom, left-to-right) and assigned wells in column-major order:

```
384-well plate: A1, B1, C1, ... P1, A2, B2, ... P24
96-well plate:  A1, B1, C1, ... H1, A2, B2, ... H12
```

## Output Files

| File | Description |
|------|-------------|
| `shapes.xml` | Leica LMD XML with all shapes and calibration |
| `shapes.csv` | Coordinates with well/cluster assignments |
| `shapes_cluster_summary.json` | Detailed cluster info (when clustering enabled) |

### CSV Format

```csv
name,type,x_um,y_um,well,cluster
RefCross_1,calibration,1234.56,7890.12,CAL,
RefCross_2,calibration,2345.67,8901.23,CAL,
det_001,shape,3456.78,9012.34,A1,0
det_002,shape,3567.89,9123.45,A1,0
det_003,shape,5678.90,1234.56,B1,1
```

### Cluster Summary JSON

```json
{
  "total_detections": 500,
  "total_clusters": 5,
  "cluster_size_target": 100,
  "clustering_method": "greedy",
  "plate_format": "384",
  "clusters": [
    {
      "cluster_id": 0,
      "well": "A1",
      "n_detections": 102,
      "centroid_px": [12345.6, 7890.1],
      "centroid_um": [2680.5, 1714.2],
      "detection_uids": ["det_001", "det_002", ...]
    }
  ]
}
```

## Complete Example

```bash
# 1. Run segmentation
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type mesothelium \
    --output-dir /path/to/output

# 2. Annotate in HTML viewer (open output/html/index.html)
#    Mark cells as Yes/No, export annotations

# 3. Generate cross placement HTML
python run_lmd_export.py \
    --detections /path/to/output/mesothelium_detections.json \
    --annotations /path/to/output/annotations.json \
    --output-dir /path/to/output/lmd \
    --generate-cross-html

# 4. Open place_crosses.html, place 3+ crosses, save JSON

# 5. Export with clustering (100 cells per well)
python run_lmd_export.py \
    --detections /path/to/output/mesothelium_detections.json \
    --annotations /path/to/output/annotations.json \
    --crosses /path/to/output/lmd/reference_crosses.json \
    --output-dir /path/to/output/lmd \
    --export \
    --cluster-size 100 \
    --plate-format 384

# 6. Load shapes.xml into Leica LMD software
```

## Coordinate Systems

The tool handles coordinate transformations automatically:

- **Image coordinates** - Origin at top-left, Y increases downward
- **Stage coordinates** - Origin at top-left, Y increases upward (default flip)

Use `--no-flip-y` if your LMD system uses image-style coordinates.

## Troubleshooting

### "Need at least 3 reference crosses"
Place at least 3 crosses in the HTML tool before saving.

### Detections not appearing in correct locations
- Verify `--pixel-size` matches your image (auto-detected from detections if available)
- Check if `--no-flip-y` is needed for your LMD system
- Ensure crosses are placed at the same landmarks visible in LMD software

### Clustering produces uneven cluster sizes
- Try different `--clustering-method` options
- DBSCAN may produce more natural groupings for clustered data
- Greedy method tends to produce more uniform sizes

### Import errors for py-lmd
```bash
pip install py-lmd
# or
pip install git+https://github.com/MannLabs/py-lmd.git
```
