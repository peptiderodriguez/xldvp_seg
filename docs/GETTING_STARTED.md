# Getting Started with xldvp_seg

A unified pipeline for cell detection and segmentation in whole-slide CZI microscopy images.

## Supported Cell Types

| Cell Type | Description | Primary Use |
|-----------|-------------|-------------|
| `mk` | Megakaryocytes | Bone marrow analysis |
| `hspc` | Hematopoietic stem/progenitor cells | Bone marrow analysis |
| `nmj` | Neuromuscular junctions | Muscle tissue analysis |
| `vessel` | Blood vessels (SMA+ rings) | Vascular morphometry |
| `mesothelium` | Mesothelial ribbons | Laser microdissection |

## Quick Start

### 1. Environment Setup

```bash
# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mkseg

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. Run Segmentation

```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --output-dir /path/to/output \
    --sample-fraction 0.10
```

### 3. Review Results

```bash
# Start local server
python -m http.server 8080 --directory /path/to/output/html

# Or use Cloudflare tunnel for remote access
~/cloudflared tunnel --url http://localhost:8080
```

Open the provided URL to view and annotate detections.

---

## Pipeline Architecture

```
CZI Image
    |
    v
[Tissue Detection] --> Identifies tissue-containing tiles
    |
    v
[Tile Sampling] --> Samples X% of tissue tiles
    |
    v
[Cell Detection] --> SAM2 / Cellpose / Custom detectors
    |
    v
[Classification] --> ResNet / Random Forest classifiers
    |
    v
[HTML Export] --> Interactive annotation viewer
    |
    v
[Training] --> Retrain classifier with annotations
    |
    v
[LMD Export] --> Leica laser microdissection format
```

---

## Detailed Usage

### Megakaryocyte Detection (MK)

Detects large polyploid cells in bone marrow using SAM2 + ResNet classification.

```bash
python run_segmentation.py \
    --czi-path /path/to/bonemarrow.czi \
    --cell-type mk \
    --channel 0 \
    --mk-min-area-um 200 \
    --mk-max-area-um 2000 \
    --sample-fraction 0.10
```

**Key parameters:**
- `--mk-min-area-um` / `--mk-max-area-um`: Size filter in square microns
- `--channel`: Fluorescence channel (usually 0 for brightfield)

### HSPC Detection

Detects small hematopoietic stem cells using Cellpose nuclei segmentation.

```bash
python run_segmentation.py \
    --czi-path /path/to/bonemarrow.czi \
    --cell-type hspc \
    --channel 0 \
    --sample-fraction 0.10
```

### NMJ Detection

Detects neuromuscular junctions in muscle tissue.

```bash
python run_segmentation.py \
    --czi-path /path/to/muscle.czi \
    --cell-type nmj \
    --channel 1 \
    --sample-fraction 0.10
```

**Inference with trained classifier:**
```bash
python run_nmj_inference.py \
    --czi-path /path/to/muscle.czi \
    --model-path /path/to/nmj_classifier.pth
```

### Vessel Detection

Detects blood vessel cross-sections via SMA+ ring structures.

```bash
python run_segmentation.py \
    --czi-path /path/to/sma_stained.czi \
    --cell-type vessel \
    --channel 0 \
    --min-vessel-diameter 10 \
    --max-vessel-diameter 500
```

**With CD31 endothelial validation:**
```bash
python run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type vessel \
    --channel 0 \
    --cd31-channel 1
```

**Output features:**
- Outer/inner diameter
- Wall thickness (mean, std, min, max)
- Lumen area, wall area
- Aspect ratio, orientation
- Ring completeness score

### Mesothelium Detection

Detects mesothelial ribbons and chunks them for laser microdissection.

```bash
python run_segmentation.py \
    --czi-path /path/to/mesothelin.czi \
    --cell-type mesothelium \
    --channel 0 \
    --target-chunk-area 1500 \
    --min-ribbon-width 5 \
    --max-ribbon-width 50
```

**Key parameters:**
- `--target-chunk-area`: Target chunk size in square microns (default: 1500)
- `--min-ribbon-width` / `--max-ribbon-width`: Filter by ribbon thickness

---

## Output Structure

Each run produces:

```
output_dir/
├── {cell_type}_detections.json    # All detections with UIDs and coordinates
├── {cell_type}_coordinates.csv    # Quick export (center, area, features)
├── tiles/
│   └── tile_X_Y/
│       └── {cell_type}_masks.h5   # Per-tile mask arrays
└── html/
    ├── index.html                 # Main viewer
    └── page_*.html                # Paginated detection pages
```

### Detection JSON Format

```json
{
  "uid": "slide_tile0_det001",
  "tile_origin": [3000, 6000],
  "center": [150, 200],
  "global_center": [3150, 6200],
  "global_center_um": [693.0, 1364.0],
  "area_px": 4523,
  "area_um2": 219.4,
  "features": {
    "pixel_size_um": 0.22,
    "solidity": 0.89,
    "eccentricity": 0.34
  }
}
```

---

## Annotation Workflow

### 1. Open HTML Viewer

```bash
python -m http.server 8080 --directory /path/to/output/html
```

### 2. Annotate Detections

- Click **Yes** (green) or **No** (red) for each detection
- Use keyboard: `Y` = Yes, `N` = No, Arrow keys = navigate
- Progress saved to browser localStorage

### 3. Export Annotations

Click "Export" button to download `annotations.json`:

```json
{
  "annotations": {
    "slide_tile0_det001": "yes",
    "slide_tile0_det002": "no"
  }
}
```

### 4. Train/Retrain Classifier

```bash
python train_nmj_classifier.py \
    --detections /path/to/detections.json \
    --annotations /path/to/annotations.json \
    --output-model /path/to/classifier.pth
```

---

## LMD Export

Export annotated detections to Leica Laser Microdissection format.

### Step 1: Place Reference Crosses

```bash
python run_lmd_export.py \
    --detections detections.json \
    --annotations annotations.json \
    --output-dir output/lmd \
    --generate-cross-html
```

Open `place_crosses.html`, click to place 3+ calibration crosses, save JSON.

### Step 2: Export with Clustering

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

This groups ~100 detections per well for collection into a 384-well plate.

See [LMD_EXPORT_GUIDE.md](LMD_EXPORT_GUIDE.md) for detailed documentation.

---

## Key Parameters Reference

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--czi-path` | required | Path to CZI file |
| `--cell-type` | required | Detection target (mk, hspc, nmj, vessel, mesothelium) |
| `--output-dir` | required | Output directory |
| `--channel` | 0 | Fluorescence channel index |
| `--tile-size` | 3000 | Tile dimensions in pixels |
| `--sample-fraction` | 0.10 | Fraction of tissue tiles to process |

### Performance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--load-to-ram` | false | Load ALL channels into RAM first (faster for network mounts) |
| `--show-metadata` | false | Show CZI metadata and exit (no processing) |

**RAM Loading:** For large files on network mounts, `--load-to-ram` loads all channels into memory once, then extracts tiles from RAM. This eliminates repeated network I/O and is significantly faster despite the upfront load time.

```bash
# Example: 176GB file on network mount
python run_segmentation.py --czi-path /mnt/x/slide.czi --cell-type nmj --load-to-ram
# Loads ~50GB per channel, then processes tiles instantly from RAM
```

### Tissue Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--skip-tissue-detection` | false | Process all tiles (skip tissue filtering) |
| `--calibration-samples` | 50 | Tiles to sample for threshold calibration |

### HTML Export

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--samples-per-page` | 300 | Detections per HTML page |
| `--crop-size` | 200 | Crop size around each detection |

---

## Remote Access

### Cloudflare Tunnel (Recommended)

No bandwidth limits, free:

```bash
# Install (one-time)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared
chmod +x ~/cloudflared

# Start tunnel
~/cloudflared tunnel --url http://localhost:8080
```

Provides a `*.trycloudflare.com` URL accessible from any device.

### Port Conventions

| Port | Use |
|------|-----|
| 8080 | MK/HSPC viewer |
| 8081 | NMJ viewer |
| 8082 | Vessel viewer |

---

## Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 32+ cores |
| RAM | 64 GB | 256+ GB |
| GPU | RTX 3080 | RTX 4090 |
| Storage | SSD | NVMe SSD |

Whole-slide CZI files are typically 20-25 GB each. Processing uses:
- ~30 GB RAM per slide loaded
- GPU for SAM2/Cellpose inference
- Multi-threaded CPU for tissue detection

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or tile size:
```bash
python run_segmentation.py ... --tile-size 2000
```

### HDF5 Plugin Errors

```bash
export HDF5_PLUGIN_PATH=""
```

### Slow Tissue Detection

Increase parallel workers:
```bash
# Default uses 80% of CPUs
# Tissue detection is I/O bound, more workers may help
```

### Empty Results

- Check channel index matches fluorescence target
- Verify tile sampling isn't too aggressive
- Ensure tissue detection threshold is appropriate

---

## File Locations

| What | Where |
|------|-------|
| This repo | `/home/dude/code/xldvp_seg_repo/` |
| MK/HSPC output | `/home/dude/xldvp_seg_output/` |
| NMJ output | `/home/dude/nmj_output/` |
| Conda env | `mkseg` |
| SAM2 checkpoint | `checkpoints/sam2.1_hiera_large.pt` |
| MK classifier | `checkpoints/best_model.pth` |
