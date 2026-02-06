# NMJ LMD Export Workflow

Complete workflow for exporting NMJ detections to Leica Laser Microdissection (LMD) format, including contour extraction, post-processing, spatial controls, and well plate assignment.

## Overview

This workflow takes classifier-filtered NMJ detections and prepares them for laser microdissection collection on 384-well plates. Key steps:

1. **Extract contours** from tile-level mask HDF5 files
2. **Post-process contours** - dilate (+0.5µm buffer) and simplify (RDP)
3. **Cluster nearby NMJs** for grouped collection
4. **Generate spatial controls** - offset 150µm for negative controls
5. **Assign wells** - serpentine ordering across 4 quadrants
6. **Create OME-Zarr pyramid** for Napari visualization
7. **Place reference crosses** for LMD calibration
8. **Export Leica XML** - final format for LMD microscope

## Prerequisites

### Installation

```bash
# Clone repo and install
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
./install.sh  # Auto-detects CUDA

# Activate environment
source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg

# Additional packages for LMD export
pip install napari[all] ome-zarr shapely hdf5plugin
```

### SAM2 Checkpoint

Download the SAM2.1 Large model (~900MB):
```bash
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 64 GB | 128+ GB |
| GPU VRAM | 8 GB | 24 GB |
| Storage | 500 GB | 1+ TB (for zarr pyramids) |

For very large slides (250k x 100k pixels), loading all channels to RAM requires ~140GB.

**Input files required:**
- `nmj_detections.json` - Detection results from `run_segmentation.py`
- `nmj_masks.h5` - Per-tile mask files in `tiles/tile_X_Y/` directories
- CZI slide file (for pyramid generation and cross placement)

## Workflow Steps

### Step 1: Run NMJ Detection with Classifier

**Channel mapping for 3-channel NMJ slides:**
| Channel | Wavelength | Marker | Role |
|---------|------------|--------|------|
| 0 (R) | 488nm | Nuclear | Context |
| 1 (G) | 647nm | BTX | NMJ marker (detection channel) |
| 2 (B) | 750nm | NFL | Neurofilament context |

**Classifier details:**
- File: `checkpoints/nmj_classifier_morph_sam2.joblib`
- Features: 334 (78 morphological + 256 SAM2 embeddings)
- Performance: Precision **0.952**, Recall 0.840, F1 0.891
- Training data: `training_data/nmj_annotations_20250202_morph_sam2_training.json` (844 annotations)

```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --output-dir /path/to/output \
    --cell-type nmj \
    --channel 1 \
    --intensity-percentile 97 \
    --all-channels \
    --load-to-ram \
    --extract-full-features \
    --skip-deep-features \
    --tile-overlap 0.1 \
    --nmj-classifier checkpoints/nmj_classifier_morph_sam2.joblib
```

**Key parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--channel 1` | BTX channel | Used for intensity-based detection |
| `--intensity-percentile 97` | Threshold | Pixels above 97th percentile |
| `--all-channels` | flag | Load all 3 channels for feature extraction |
| `--load-to-ram` | flag | Faster for network-mounted slides |
| `--tile-overlap 0.1` | 10% | Catches NMJs at tile boundaries |
| `--skip-deep-features` | flag | Use morph+SAM2 only (faster) |

**Output:**
```
output/
├── slide_name/tiles/tile_X_Y/
│   ├── nmj_masks.h5         # Label masks (uint16)
│   └── nmj_features.json    # Per-tile detections
├── nmj_detections.json      # Merged detections with RF predictions
└── html/                    # Annotation viewer
```

**Expected runtime:** ~120-150 sec/tile due to SAM2 embedding extraction. A 250k x 100k pixel slide with ~1800 tiles takes 60-80 hours at 100% sampling.

**Note on tile overlap:** With `--tile-overlap 0.1`, NMJs at tile boundaries may be detected in multiple tiles. Deduplication is applied automatically based on centroid distance.

### Step 1b: Review Detections (Optional)

Before proceeding to LMD export, review detections in the HTML viewer:

```bash
# Start HTTP server
python -m http.server 8080 --directory /path/to/output/html

# Or use Cloudflare tunnel for remote access
~/cloudflared tunnel --url http://localhost:8080
```

Open `index.html` in browser. Detections are paginated (300 per page). Use keyboard navigation (arrow keys) to review. The classifier has 95.2% precision, so ~5% may be false positives.

### Step 2: Extract Contours from Masks

The detection JSON only stores centroids. Contours must be extracted from the per-tile HDF5 mask files.

```python
#!/usr/bin/env python3
"""extract_contours.py - Extract contours from mask files."""

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import json
import numpy as np
import hdf5plugin  # Must import BEFORE h5py
import h5py
import cv2
from pathlib import Path

def extract_contour_from_mask(mask, label):
    """Extract outer contour for a specific label."""
    binary = (mask == label).astype(np.uint8)
    if binary.sum() == 0:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2)  # Shape (N, 2) as [x, y]

def local_to_global(contour, tile_origin):
    """Convert tile coords to global mosaic coords."""
    contour_global = contour.copy().astype(float)
    contour_global[:, 0] += tile_origin[0]  # Add tile_x
    contour_global[:, 1] += tile_origin[1]  # Add tile_y
    return contour_global

# Main extraction loop
tiles_dir = Path("/path/to/output/slide_name/tiles")
detections = json.load(open("nmj_detections.json"))

for det in detections:
    if det.get('rf_prediction') != 1:
        continue

    tile_name = det['tile_name']
    mask_path = tiles_dir / tile_name / "nmj_masks.h5"

    with h5py.File(mask_path, 'r') as f:
        masks = f['masks'][:]

    # Label is extracted from detection ID (e.g., "det_5" -> 5)
    label = int(det['id'].split('_')[-1])
    contour = extract_contour_from_mask(masks, label)

    if contour is not None:
        contour_global = local_to_global(contour, det['tile_origin'])
        det['outer_contour_global'] = contour_global.tolist()
```

### Step 3: Post-Process Contours

Contours need processing for LMD:
1. **Dilation** - Add 0.5µm buffer so laser cuts outside the NMJ
2. **RDP simplification** - Reduce point count for LMD hardware limits

```python
"""contour_processing.py - Dilate and simplify contours."""

import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.validation import make_valid

PIXEL_SIZE_UM = 0.1725
DILATION_UM = 0.5
RDP_EPSILON = 5  # pixels

def process_contour(contour_px, pixel_size_um=PIXEL_SIZE_UM,
                    dilation_um=DILATION_UM, rdp_epsilon=RDP_EPSILON):
    """
    Process contour: validate, dilate, simplify.

    Args:
        contour_px: Contour in pixels, list of [x, y] pairs

    Returns:
        Processed contour in micrometers as numpy array
    """
    if len(contour_px) < 3:
        return None

    contour_px = np.array(contour_px)
    contour_um = contour_px * pixel_size_um

    # Create Shapely polygon and dilate
    poly = Polygon(contour_um)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area < 0.1:
        return None

    poly_dilated = poly.buffer(dilation_um)

    if poly_dilated.geom_type == 'MultiPolygon':
        poly_dilated = max(poly_dilated.geoms, key=lambda p: p.area)

    # Get coordinates and simplify with RDP
    coords = np.array(poly_dilated.exterior.coords)[:-1]
    coords_px = coords / pixel_size_um

    # RDP simplification in OpenCV
    coords_cv = coords_px.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(coords_cv, rdp_epsilon, closed=True)
    simplified_um = simplified.reshape(-1, 2) * pixel_size_um

    return simplified_um
```

**Typical results:**
- Points reduction: ~70-80% (e.g., 200 → 40 points)
- Area increase: ~5-10% (from dilation buffer)

### Step 4: Cluster NMJs and Assign Wells

Group nearby NMJs into clusters for efficient collection. Singles (outliers) and clusters are handled separately.

```bash
# Using run_lmd_export.py with clustering
python run_lmd_export.py \
    --detections nmj_detections_with_contours.json \
    --output-dir lmd_export \
    --cluster-size 10 \
    --clustering-method greedy \
    --plate-format 384
```

**Clustering options:**
| Option | Description |
|--------|-------------|
| `--cluster-size N` | Target detections per well (e.g., 10) |
| `--clustering-method` | `greedy` (default), `kmeans`, or `dbscan` |
| `--plate-format` | `384` or `96` |

**Well assignment pattern:**
- **Singles first**, then **clusters**
- Within each group: nearest-neighbor ordering to minimize slide movement
- Serpentine order across 4 quadrants (B2, C2, B3, C3)
- Alternating: NMJ → Control → NMJ → Control

### Step 5: Generate Spatial Controls

For each NMJ target, create a paired control region offset 150µm away. The control has the identical shape (same contour shifted).

```python
"""Spatial control generation (in run_lmd_export.py)."""

CONTROL_DIRECTIONS = {
    'E':  (1, 0),   'NE': (1, -1),  'N':  (0, -1),  'NW': (-1, -1),
    'W':  (-1, 0),  'SW': (-1, 1),  'S':  (0, 1),   'SE': (1, 1)
}

def generate_spatial_control(detection, all_detections, offset_um=150.0,
                             pixel_size=0.1725, image_bounds=None):
    """
    Generate control by shifting NMJ contour.
    Tries 8 directions, returns first non-overlapping option.
    """
    contour = detection.get('outer_contour_global')
    if not contour:
        return None

    offset_px = offset_um / pixel_size
    contour = np.array(contour)

    # Try each direction
    for direction_name, (dx, dy) in CONTROL_DIRECTIONS.items():
        offset_vec = np.array([dx, dy]) * offset_px
        shifted = contour + offset_vec

        # Check bounds
        if image_bounds:
            if (shifted[:, 0].min() < image_bounds[0] or
                shifted[:, 0].max() > image_bounds[2] or
                shifted[:, 1].min() < image_bounds[1] or
                shifted[:, 1].max() > image_bounds[3]):
                continue

        # Check collision with all detections
        if not check_polygon_overlap(shifted, all_detections):
            return {
                'uid': detection['uid'] + '_ctrl',
                'control_of': detection['uid'],
                'offset_direction': direction_name,
                'offset_um': offset_um,
                'outer_contour_global': shifted.tolist(),
                'global_center': [
                    detection['global_center'][0] + offset_vec[0],
                    detection['global_center'][1] + offset_vec[1]
                ],
                'is_control': True
            }

    return None  # All directions overlap
```

### Step 6: Create OME-Zarr Pyramid

For smooth Napari viewing of large slides, convert CZI to OME-Zarr with multi-resolution pyramids.

```bash
python scripts/czi_to_ome_zarr.py \
    /path/to/slide.czi \
    /path/to/output.zarr \
    --channel-names "nuc488" "Bgtx647" "NfL750" \
    --overwrite
```

**Or use RAM-based generation for faster processing:**

```python
"""generate_zarr_pyramid.py - RAM-based pyramid generation."""

from aicsimageio import AICSImage
import zarr
from numcodecs import Blosc
from skimage.transform import downscale_local_mean

# Load full CZI into RAM
img = AICSImage(czi_path)
data = img.data  # ~176GB for large slide

# Generate pyramid levels
levels = [1, 2, 4, 8, 16, 32]  # Scale factors
for level in levels:
    if level == 1:
        level_data = data
    else:
        level_data = downscale_local_mean(data, (1, 1, 1, level, level))

    # Write to zarr with OME-NGFF metadata
    root.create_dataset(str(i), data=level_data, ...)
```

### Step 7: Place Reference Crosses in Napari

Reference crosses are calibration points that align LMD stage coordinates with image coordinates. Place at identifiable tissue landmarks.

```bash
python scripts/napari_place_crosses.py \
    /path/to/pyramid.zarr \
    --output reference_crosses.json \
    --detections nmj_detections.json
```

**Keyboard shortcuts:**
- `S` - Save crosses to JSON
- Click to place cross (minimum 3 required)
- Scroll to zoom, drag to pan

**Output format:**
```json
{
  "image_width_px": 254976,
  "image_height_px": 100503,
  "pixel_size_um": 0.1725,
  "crosses": [
    {"id": 1, "x_px": 12345, "y_px": 67890, "x_um": 2129.5, "y_um": 11711.0},
    {"id": 2, "x_px": 200000, "y_px": 50000, ...},
    {"id": 3, "x_px": 100000, "y_px": 80000, ...}
  ]
}
```

### Step 8: Export to Leica LMD XML

Final export generates XML for the Leica LMD microscope.

```bash
python run_lmd_export.py \
    --detections nmj_detections_with_contours.json \
    --crosses reference_crosses.json \
    --output-dir lmd_export \
    --export \
    --generate-controls \
    --control-offset-um 150
```

**Output files:**
| File | Description |
|------|-------------|
| `lmd_export_with_controls.json` | All shapes with well assignments |
| `shapes_with_controls.xml` | Leica LMD XML format |
| `well_assignment_384.json` | Well plate mapping |

### Step 9: Verify in Napari

Visualize the final export overlaid on the slide to verify correctness.

```bash
python scripts/napari_view_lmd_export.py \
    --zarr /path/to/pyramid.zarr \
    --export lmd_export_with_controls.json
```

**Color coding:**
- **Green**: Singles (NMJs)
- **Cyan**: Single controls
- **Red**: Clusters (NMJs)
- **Orange**: Cluster controls

## File Locations

### Input Data
| File | Location |
|------|----------|
| Classifier | `checkpoints/nmj_classifier_morph_sam2.joblib` |
| Training annotations | `training_data/nmj_annotations_20250202_morph_sam2_training.json` |

### Output Structure
```
lmd_export/
├── singles_with_contours.json     # Extracted + processed contours
├── well_assignment_384.json       # Well plate mapping
├── reference_crosses.json         # Calibration points
├── lmd_export_with_controls.json  # Final export data
├── shapes_with_controls.xml       # Leica LMD format
└── collection_map.png             # Visual overview
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PIXEL_SIZE_UM` | 0.1725 | Micrometers per pixel (for this slide) |
| `DILATION_UM` | 0.5 | Buffer around NMJ contour for laser cutting |
| `RDP_EPSILON` | 5 | RDP simplification threshold (pixels) |
| `CONTROL_OFFSET_UM` | 150 | Control region offset distance (µm) |
| `--intensity-percentile` | 97 | Detection threshold percentile |
| `--tile-overlap` | 0.1 | Tile overlap fraction for boundary NMJs |

## Well Plate Layout

**384-well plate with 4 quadrants:**

```
     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
   ┌──────────────────────────────────────────────────────────────────────┐
 A │  (outer row - not used)                                              │
 B │     B2────────────────────►     B3────────────────────►              │
 C │     ◄────────────────────C2     ◄────────────────────C3              │
 D │     D2────────────────────►     D3────────────────────►              │
 E │     ◄────────────────────E2     ...                                  │
 F │     ...                                                              │
   │                                                                      │
 O │                                                                      │
 P │  (outer row - not used)                                              │
   └──────────────────────────────────────────────────────────────────────┘
```

**Serpentine pattern:**
- Quadrant B2: B2 → B4 → B6 → ... → B22 → D22 → D20 → ... → D2 → F2 → ...
- Then C2, B3, C3 quadrants in sequence

**Alternating NMJ/Control:**
- Well 1: NMJ #1
- Well 2: Control for NMJ #1
- Well 3: NMJ #2
- Well 4: Control for NMJ #2
- ...

## Troubleshooting

### HDF5 Plugin Errors
```python
# Must set BEFORE importing h5py
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import hdf5plugin  # BEFORE h5py
import h5py
```

### Contour Extraction Fails
- Check that `det['id']` matches mask label (e.g., `"det_5"` → label 5)
- Verify tile masks exist in `tiles/tile_X_Y/nmj_masks.h5`

### Controls All Overlap
- Increase `CONTROL_OFFSET_UM` (e.g., 200µm instead of 150µm)
- Check image bounds are set correctly

### Napari Pyramid Loading Slow
- Use OME-Zarr format instead of loading CZI directly
- Pre-compute pyramid with `czi_to_ome_zarr.py`

### XML Import Issues in LMD Software
- Verify Y-axis flip is correct (`flip_y=True` by default)
- Check reference crosses are at identifiable landmarks
- Confirm pixel size matches LMD microscope settings

## Related Documentation

- [NMJ Pipeline Guide](NMJ_PIPELINE_GUIDE.md) - Detection and classification
- [LMD Export Guide](LMD_EXPORT_GUIDE.md) - Basic LMD export
- [Coordinate System](COORDINATE_SYSTEM.md) - UID and coordinate conventions
