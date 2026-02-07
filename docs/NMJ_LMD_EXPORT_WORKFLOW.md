# NMJ LMD Export Workflow

Complete workflow for exporting NMJ detections to Leica Laser Microdissection (LMD) format, including contour extraction, post-processing, biological clustering, spatial controls, and well plate assignment.

## Overview

This workflow takes classifier-filtered NMJ detections and prepares them for laser microdissection collection on 384-well plates. Key steps:

1. **Run NMJ detection** with classifier
2. **Cluster NMJs** biologically (area-based grouping, 375-425 um2 target)
3. **Extract contours** from tile-level mask HDF5 files
4. **Post-process contours** - dilate (+0.5um buffer) and simplify (RDP)
5. **Generate spatial controls** - offset 100um for negative controls (every sample gets a control)
6. **Assign wells** - serpentine ordering across 4 quadrants (B2->B3->C3->C2)
7. **Create OME-Zarr pyramid** for Napari visualization
8. **Place reference crosses** for LMD calibration
9. **Export Leica XML** - final format for LMD microscope
10. **Verify in Napari** - 4-color overlay check

## Prerequisites

### Installation

```bash
git clone https://github.com/peptiderodriguez/xldvp_seg.git
cd xldvp_seg
./install.sh  # Auto-detects CUDA

source ~/miniforge3/etc/profile.d/conda.sh && conda activate mkseg

# Additional packages for LMD export
pip install napari[all] ome-zarr shapely hdf5plugin py-lmd
```

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 64 GB | 128+ GB |
| GPU VRAM | 8 GB | 24 GB |
| Storage | 500 GB | 1+ TB (for zarr pyramids) |

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

```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --output-dir /path/to/output \
    --cell-type nmj \
    --channel 1 \
    --intensity-percentile 98 \
    --all-channels \
    --load-to-ram \
    --extract-full-features \
    --skip-deep-features \
    --tile-overlap 0.1 \
    --nmj-classifier checkpoints/nmj_classifier_morph_sam2.joblib
```

**Output:**
```
output/
├── slide_name/tiles/tile_X_Y/
│   ├── nmj_masks.h5         # Label masks (uint16)
│   └── nmj_features.json    # Per-tile detections
├── nmj_detections.json      # Merged detections with RF predictions
└── html/                    # Annotation viewer
```

### Step 2: Review Detections (Optional)

```bash
python -m http.server 8080 --directory /path/to/output/html
```

### Step 3: Cluster Detections Biologically

Group nearby detections into clusters targeting 375-425 um2 total area per cluster. Two-stage spatial clustering:
- Round 1: 500 um distance threshold (tight groups)
- Round 2: 1000 um threshold (remaining detections)

```bash
python scripts/cluster_detections.py \
    --detections nmj_detections.json \
    --pixel-size 0.1725 \
    --area-min 375 --area-max 425 \
    --dist-round1 500 --dist-round2 1000 \
    --min-score 0.5 \
    --output nmj_clusters.json
```

**Note:** The clustering algorithm is cell-type-agnostic. The same script works for any detection type. The underlying module is `segmentation.lmd.clustering`.

**Output:** `nmj_clusters.json` with `main_clusters` and `outliers` (singles).

### Step 4: Create OME-Zarr Pyramid

```bash
python scripts/czi_to_ome_zarr.py \
    /path/to/slide.czi \
    /path/to/output.zarr \
    --channel-names "nuc488" "Bgtx647" "NfL750" \
    --overwrite
```

### Step 5: Place Reference Crosses in Napari

```bash
python scripts/napari_place_crosses.py \
    /path/to/pyramid.zarr \
    --output reference_crosses.json \
    --detections nmj_detections.json
```

Minimum 3 crosses at identifiable tissue landmarks.

### Step 6: Run Unified LMD Export

Single command handles contour extraction, processing, controls, and well assignment:

```bash
python run_lmd_export.py \
    --detections nmj_detections.json \
    --crosses reference_crosses.json \
    --clusters nmj_clusters.json \
    --tiles-dir /path/to/output/slide_name/tiles \
    --output-dir lmd_export \
    --export \
    --generate-controls \
    --control-offset-um 100 \
    --min-score 0.5
```

**Key flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--clusters` | - | Clusters JSON from step 3 |
| `--tiles-dir` | - | Path to tiles/ with H5 masks (auto-extracts contours) |
| `--generate-controls` | off | Generate control for every sample |
| `--control-offset-um` | 100 | Control offset distance in um |
| `--dilation-um` | 0.5 | Contour dilation buffer in um |
| `--rdp-epsilon` | 5 | RDP simplification epsilon (pixels) |
| `--min-score` | - | Filter detections by rf_prediction score |

**Output files:**
| File | Description |
|------|-------------|
| `shapes_with_controls.json` | Unified export with all shapes + wells |
| `shapes_summary.csv` | Per-well CSV summary |
| `shapes.xml` | Leica LMD XML format |

### Step 7: Verify in Napari

```bash
python scripts/napari_view_lmd_export.py \
    --zarr /path/to/pyramid.zarr \
    --export lmd_export/shapes_with_controls.json
```

**Color coding:**
- **Green**: Singles (NMJs)
- **Cyan**: Single controls
- **Red**: Clusters (NMJs)
- **Orange**: Cluster controls

## Well Plate Layout

**384-well plate, 4 quadrants, serpentine order: B2 -> B3 -> C3 -> C2**

```
     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
   +------------------------------------------------------------------------+
 A | (outer row - not used)                                                  |
 B |    B2---->B4---->...---->B22    B3---->B5---->...---->B23               |
 C |    <----C20<----...<----C2      <----C21<----...<----C3                |
 D |    D2---->D4---->...---->D22    D3---->D5---->...---->D23               |
   |    ...                          ...                                     |
 N |    N2---->N4---->...---->N22    N3---->N5---->...---->N23               |
 O |    <----O20<----...<----O2      <----O21<----...<----O3                |
 P | (outer row - not used)                                                  |
   +------------------------------------------------------------------------+
```

**Well assignment with alternating NMJ/Control:**
```
Well 1 (B2):  Single NMJ #1
Well 2 (B4):  Control for NMJ #1
Well 3 (B6):  Single NMJ #2
Well 4 (B8):  Control for NMJ #2
...
Well N:       Cluster #1 (all member contours in one well)
Well N+1:     Control for Cluster #1 (all shifted contours)
...
```

Every sample always has a paired control. Singles before clusters.

## Unified Export JSON Format

```json
{
  "metadata": {
    "plate_format": "384",
    "quadrant_order": ["B2", "B3", "C3", "C2"],
    "pixel_size_um": 0.1725,
    "dilation_um": 0.5,
    "rdp_epsilon_px": 5,
    "control_offset_um": 100
  },
  "summary": {
    "n_singles": 74, "n_single_controls": 74,
    "n_clusters": 58, "n_cluster_controls": 58,
    "n_nmjs_in_clusters": 593,
    "total_wells_used": 264
  },
  "shapes": [
    {"type": "single", "well": "B2", "uid": "...", "contour_um": [[x,y],...]},
    {"type": "single_control", "well": "B4", "control_of": "...", "contour_um": [[x,y],...]},
    {"type": "cluster", "well": "D6", "cluster_id": 3, "n_nmjs": 4, "contours_um": [[[x,y],...], ...]},
    {"type": "cluster_control", "well": "D8", "control_of_cluster": 3, "contours_um": [[[x,y],...], ...]}
  ],
  "well_order": ["B2", "B4", "B6", ...]
}
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PIXEL_SIZE_UM` | 0.1725 | Micrometers per pixel |
| `DILATION_UM` | 0.5 | Buffer around NMJ contour for laser cutting |
| `RDP_EPSILON` | 5 | RDP simplification threshold (pixels) |
| `CONTROL_OFFSET_UM` | 100 | Control region offset distance (um) |
| Cluster area target | 375-425 | Total area per cluster (um2) |
| Round 1 distance | 500 | First clustering pass distance (um) |
| Round 2 distance | 1000 | Second clustering pass distance (um) |

## Troubleshooting

### HDF5 Plugin Errors
```python
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import hdf5plugin  # BEFORE h5py
import h5py
```

### Contour Extraction Fails
- Use `det['mask_label']` for mask label lookup (centroid-based, with 3x3 neighborhood fallback)
- Verify tile masks exist in `tiles/tile_X_Y/nmj_masks.h5`

### Controls Use Fallback Offset
- If `_fallback` appears in offset_direction, all 8 directions overlapped at normal offset
- The pipeline automatically increases offset (1.5x per retry, 3 attempts)
- This ensures every sample always gets a control

### XML Import Issues in LMD Software
- Verify Y-axis flip is correct (`flip_y=True` by default)
- Check reference crosses are at identifiable landmarks
- Confirm pixel size matches LMD microscope settings

## Related Documentation

- [NMJ Pipeline Guide](NMJ_PIPELINE_GUIDE.md) - Detection and classification
- [LMD Export Guide](LMD_EXPORT_GUIDE.md) - Basic LMD export
- [Coordinate System](COORDINATE_SYSTEM.md) - UID and coordinate conventions

## Legacy Scripts

The following scripts have been superseded by the unified pipeline and moved to `scripts/legacy/`:
- `scripts/legacy/generate_full_lmd_export.py` -> use `run_lmd_export.py`
- `scripts/legacy/extract_singles_contours.py` -> use `run_lmd_export.py --tiles-dir`
