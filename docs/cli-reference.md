# CLI Reference

The `xlseg` command provides 13 subcommands covering the full pipeline from
CZI inspection through detection, classification, and export.

```bash
xlseg --help
```

## Subcommands

### `xlseg info`

Inspect CZI metadata -- channels, dimensions, pixel size, scene count.

```bash
xlseg info /path/to/slide.czi
```

**Always run this before writing any channel configuration.** Channel order
in CZI files is NOT wavelength-sorted.

---

### `xlseg detect`

Run the cell detection pipeline. Supports all 8 detection strategies with
multi-GPU processing.

```bash
xlseg detect \
    --czi-path /path/to/slide.czi \
    --cell-type cell \
    --channel-spec "cyto=PM,nuc=488" \
    --all-channels \
    --num-gpus 4 \
    --output-dir /path/to/output
```

Key flags:

| Flag | Description |
|------|-------------|
| `--cell-type` | Strategy: `cell`, `nmj`, `mk`, `vessel`, `mesothelium`, `islet`, `tissue_pattern` |
| `--channel-spec` | Resolve channels by name or wavelength (e.g., `"cyto=PM,nuc=488"`) |
| `--all-channels` | Extract per-channel intensity features (~15 per channel) |
| `--num-gpus N` | Number of GPUs (always multi-GPU architecture) |
| `--extract-deep-features` | Add ResNet + DINOv2 features (6,144 dims, off by default) |
| `--html-sample-fraction` | Subsample HTML viewer (e.g., 0.10 for 10%) |
| `--segmenter` | `cellpose` (default) or `instanseg` |
| `--dedup-method` | `mask_overlap` (default) or `iou_nms` |
| `--resume` | Resume from a previous run directory |
| `--marker-snr-channels` | Auto-classify markers during detection using pre-computed SNR >= 1.5 (format: `"SMA:1,CD31:3"`) |
| `--tissue-channels` | Marker channels identifying tissue regions to segment (e.g., `"2,3,5"`). Required for islet. Not for Cellpose segmentation (use `--channel-spec`). Legacy: `--islet-display-channels` |
| `--no-contour-processing` | Skip contour extraction from HDF5 masks |
| `--no-background-correction` | Skip local background subtraction |

---

### `xlseg classify`

Train a random forest classifier from annotated detections.

```bash
xlseg classify \
    --detections /path/to/cell_detections.json \
    --annotations /path/to/annotations.json \
    --output-dir /path/to/classifiers \
    --feature-set morph
```

| Flag | Description |
|------|-------------|
| `--feature-set` | `morph` (78D), `morph_sam2` (334D), `channel_stats`, `all` (6,478D) |

---

### `xlseg score`

Apply a trained classifier to score all detections.

```bash
xlseg score \
    --detections /path/to/cell_detections.json \
    --classifier /path/to/rf_classifier.pkl \
    --output /path/to/cell_detections_scored.json
```

---

### `xlseg markers`

Classify each cell as marker-positive or marker-negative per fluorescent channel.

```bash
xlseg markers \
    --detections /path/to/cell_detections.json \
    --marker-wavelength 647,555 \
    --marker-name SMA,CD31 \
    --czi-path /path/to/slide.czi
```

| Flag | Description |
|------|-------------|
| `--marker-wavelength` | Wavelengths to resolve (auto-matches CZI channels) |
| `--marker-channel` | Direct channel indices (alternative to wavelength) |
| `--marker-name` | Human-readable names for each marker |
| `--method` | `snr` (default), `otsu`, `otsu_half`, `gmm` (BIC model selection; returns all-negative for unimodal data) |

---

### `xlseg cluster`

Feature clustering with UMAP/t-SNE dimensionality reduction and
Leiden/HDBSCAN community detection.

```bash
xlseg cluster \
    --detections /path/to/cell_detections.json \
    --feature-groups "morph,channel" \
    --output-dir /path/to/clusters
```

| Flag | Description |
|------|-------------|
| `--feature-groups` | Comma-separated: `morph`, `shape`, `color`, `sam2`, `channel`, `deep` |
| `--clustering` | `leiden` (default) or `hdbscan` |

---

### `xlseg qc`

Quick quality check on pipeline output. Prints detection count, feature summary,
area distribution, classifier scores, marker profiles, per-channel SNR, and
nuclear counting summary -- no HTML viewer needed.

```bash
xlseg qc /path/to/output
```

---

### `xlseg export-lmd`

Export scored detections to Leica LMD XML format for laser microdissection.

```bash
xlseg export-lmd \
    --detections /path/to/cell_detections_scored.json \
    --crosses /path/to/crosses.json \
    --output-dir /path/to/lmd \
    --min-score 0.5 \
    --export
```

| Flag | Description |
|------|-------------|
| `--crosses` | Reference cross positions from Napari |
| `--min-score` | Minimum classifier score for export |
| `--max-area-change-pct` | Adaptive RDP simplification tolerance (default 10%) |
| `--max-dilation-area-pct` | Adaptive dilation for laser buffer (default 10%) |
| `--generate-controls` | Add control wells |

---

### `xlseg serve`

Launch the HTML detection viewer with Cloudflare tunnel.

```bash
xlseg serve /path/to/output
```

---

### `xlseg system`

Show system information: CPU, RAM, GPU, SLURM partition availability,
and resource recommendations.

```bash
xlseg system
```

---

### `xlseg models`

List all registered model checkpoints with feature dimensions,
modality, and HuggingFace availability.

```bash
xlseg models
```

---

### `xlseg strategies`

List all registered detection strategies.

```bash
xlseg strategies
```

---

### `xlseg download-models`

Download gated model checkpoints from HuggingFace (requires HF token).

```bash
xlseg download-models --brightfield   # UNI2, Virchow2, CONCH, Phikon-v2
xlseg download-models --all           # All registered models
xlseg download-models --model uni2    # Specific model
```

## Standalone Scripts

### `generate_contour_viewer.py`

Generate a self-contained HTML viewer for contour overlays on CZI fluorescence.
Loads contours from JSON files (vessel lumens, cell detections), groups by a
configurable field, and renders with pan/zoom, viewport culling for 50K+ contours,
R/G/B channel toggle, and click-to-inspect metadata panel.

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

| Flag | Description |
|------|-------------|
| `--contours` | Path to JSON file with contour data (vessel lumens, cell detections) |
| `--group-field` | Field to group contours by (e.g., `vessel_type`, `scale`, `marker_profile`) |
| `--czi-path` | CZI file for fluorescence background |
| `--display-channels` | Comma-separated channel indices for R,G,B display |
| `--channel-names` | Comma-separated human-readable channel names |
| `--title` | Viewer title |
| `--output` | Output HTML file path |

The viewer uses the `xldvp_seg.visualization` package for CZI thumbnail loading,
color palette assignment, binary data encoding, and composable JS components.

---

## Flag Gotchas

- `--no-normalize-features` disables flat-field (no `--flat-field-correction` flag exists)
- `--photobleaching-correction` (with `-ing`) is experimental
- `--sequential` does not exist -- use `--num-gpus 1`
- `--sample-fraction` is always 1.0 -- use `--html-sample-fraction` to subsample the viewer
- `--nuclear-channel` / `--membrane-channel` are islet-only
