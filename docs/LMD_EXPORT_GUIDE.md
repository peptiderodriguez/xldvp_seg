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

### Step 1: Place Reference Crosses in Napari

Reference crosses are calibration points that align the microscope stage coordinates with your image. Place them at identifiable landmarks visible under the LMD microscope.

```bash
# CZI-native (recommended — no OME-Zarr conversion needed)
python scripts/napari_place_crosses.py \
    -i /path/to/slide.czi --channel 0 \
    -o /path/to/output/reference_crosses.json

# With LMD7 display transforms (tissue-down + rotated)
python scripts/napari_place_crosses.py \
    -i /path/to/slide.czi --channel 0 \
    --flip-horizontal --rotate-cw-90 \
    -o /path/to/output/reference_crosses.json

# With contour overlay
python scripts/napari_place_crosses.py \
    -i /path/to/slide.czi --channel 0 \
    --contours /path/to/detections.json --color-by well \
    -o /path/to/output/reference_crosses.json
```

Keybinds: R/G/B select cross color, Space places at cursor, S saves, Q saves+quits. Minimum 3 crosses required.

**Tips for cross placement:**
- Choose landmarks visible in both Napari and under the LMD microscope
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

Clustering is done as a separate step before LMD export using `scripts/cluster_detections.py`:

```bash
# Step 1: Cluster detections biologically (area-based grouping)
python scripts/cluster_detections.py \
    --detections detections.json \
    --pixel-size 0.1725 \
    --area-min 375 --area-max 425 \
    --dist-round1 500 --dist-round2 1000 \
    --min-score 0.5 \
    --output clusters.json

# Step 2: Export with clustering and controls
python run_lmd_export.py \
    --detections detections.json \
    --crosses reference_crosses.json \
    --clusters clusters.json \
    --tiles-dir /path/to/tiles \
    --output-dir output/lmd \
    --export --generate-controls
```

### Contour Processing at Export

| Option | Default | Description |
|--------|---------|-------------|
| `--dilation-um` | 0.5 | Expand contours so laser cuts outside the cell |
| `--rdp-epsilon` | 5 | RDP simplification (reduce vertex count for LMD hardware) |
| `--erosion-um` | 0 | Shrink contours by absolute distance (um) |
| `--erode-pct` | 0 | Shrink contours by % of sqrt(area) |

### Well Assignment

384-well plate, 4 quadrants, serpentine ordering: B2 -> B3 -> C3 -> C2 (max 308 wells).
Singles are assigned before clusters. Every sample gets a paired control well.

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

# 3. Place reference crosses in Napari (3+ at tissue landmarks)
python scripts/napari_place_crosses.py \
    -i /path/to/mesothelin.czi --channel 0 \
    -o /path/to/output/lmd/reference_crosses.json

# 4. Cluster detections
python scripts/cluster_detections.py \
    --detections /path/to/output/mesothelium_detections.json \
    --pixel-size 0.1725 \
    --output /path/to/output/lmd/clusters.json

# 5. Export to LMD XML with controls
python run_lmd_export.py \
    --detections /path/to/output/mesothelium_detections.json \
    --crosses /path/to/output/lmd/reference_crosses.json \
    --clusters /path/to/output/lmd/clusters.json \
    --tiles-dir /path/to/output/slide_name/tiles \
    --output-dir /path/to/output/lmd \
    --export --generate-controls

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
- Adjust `--area-min` and `--area-max` targets in `cluster_detections.py`
- Try different `--dist-round1` and `--dist-round2` distance thresholds
- Use `--min-score` to pre-filter low-confidence detections

### Import errors for py-lmd
```bash
pip install py-lmd
# or
pip install git+https://github.com/MannLabs/py-lmd.git
```
