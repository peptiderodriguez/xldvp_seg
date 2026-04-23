# xldvp_seg

**Spatial cell segmentation and Deep Visual Proteomics pipeline.**

xldvp_seg detects cells in fluorescence microscopy images (CZI format), extracts
up to 6,478 features per cell, classifies markers, and exports contours for laser
microdissection. It supports 8 detection strategies, multi-GPU processing, and
integrates with the scverse ecosystem (scanpy, AnnData, SpatialData).

## Recommended: use with [Claude Code](https://claude.ai/claude-code)

The repo ships slash commands (`/analyze`, `/new-experiment`, `/czi-info`, ...) and
a pipeline-aware `CLAUDE.md` that let Claude Code walk you through the whole
workflow — install, detection, classification, LMD export, MS queue — as a
conversation.

```bash
# Install Claude Code once (macOS/Linux)
curl -fsSL https://claude.ai/install.sh | bash
# Windows: irm https://claude.ai/install.ps1 | iex
# or: npm install -g @anthropic-ai/claude-code

# Use it inside the repo
cd xldvp_seg && claude
/analyze   # installs the package if needed, then guides detection end-to-end
```

See [Getting Started](GETTING_STARTED.md) for details.

## Quick Example

### CLI

```bash
# 1. Inspect CZI channels (always do this first)
xlseg info /path/to/slide.czi

# 2. Run detection (Cellpose + SAM2, multi-GPU)
xlseg detect \
    --czi-path /path/to/slide.czi \
    --cell-type cell \
    --channel-spec "cyto=PM,nuc=488" \
    --all-channels \
    --num-gpus 4 \
    --output-dir /path/to/output

# 3. View and annotate detections in the browser
xlseg serve /path/to/output

# 4. Train classifier from annotations
xlseg classify \
    --detections /path/to/output/cell_detections.json \
    --annotations /path/to/annotations.json \
    --output-dir /path/to/classifiers

# 5. Score all detections with the trained classifier
xlseg score \
    --detections /path/to/output/cell_detections.json \
    --classifier /path/to/classifiers/rf_classifier.pkl \
    --output /path/to/output/cell_detections_scored.json

# 6. Classify markers (SNR >= 1.5)
xlseg markers \
    --detections /path/to/output/cell_detections_scored.json \
    --marker-wavelength 647,555 \
    --marker-name SMA,CD31 \
    --czi-path /path/to/slide.czi

# 7. Feature clustering (UMAP + Leiden)
xlseg cluster \
    --detections /path/to/output/cell_detections_scored.json \
    --feature-groups "morph,channel" \
    --output-dir /path/to/output/clusters

# 8. Quick quality check (detection count, area stats, marker profiles)
xlseg qc /path/to/output

# 9. LMD XML + MS queue CSV (after placing reference crosses in Napari)
xlseg export-lmd --detections ... --crosses ... --output-dir .../lmd --export
xlseg ms-queue  --samples .../lmd/replicates.csv --config ms_queue.yaml \
                --output-dir .../ms_queues --combined
```

!!! tip
    Use `--marker-snr-channels "SMA:1,CD31:3"` on `xlseg detect` to classify
    markers automatically during detection -- no separate `xlseg markers` step needed.

### Python API

```python
from xldvp_seg.core import SlideAnalysis
from xldvp_seg.api import tl

# Load pipeline output
slide = SlideAnalysis.load("/path/to/output/slide_name/run_timestamp/")
print(f"{slide.n_detections} detections, {slide.cell_type}")

# Classify markers
tl.markers(slide, marker_channels=[1, 3], marker_names=["SMA", "CD31"])

# Score with trained classifier
tl.score(slide, classifier="classifiers/rf_morph.pkl")

# Cluster by features
tl.cluster(slide, feature_groups="morph,channel", output_dir="results/clusters/")

# Export to AnnData for scanpy/squidpy workflows
adata = slide.to_anndata()
```

## Detection Strategies

| Type | Detection Method | Use Case |
|------|-----------------|----------|
| **Cell** | Cellpose 2-channel (cyto+nuc) + SAM2 embeddings | Generic cell detection |
| **NMJ** | 98th percentile threshold + morphology + watershed | Neuromuscular junctions |
| **MK** | SAM2 auto-mask + size filter | Bone marrow megakaryocytes |
| **Vessel** | SMA+ ring detection, 3-contour hierarchy | Blood vessel morphometry |
| **Islet** | Cellpose membrane+nuclear + marker classification | Pancreatic islet cells |
| **Mesothelium** | Ridge detection for ribbon structures | Mesothelial ribbon for LMD |
| **Tissue Pattern** | Cellpose + spatial frequency analysis | Whole-mount tissue (brain FISH) |
| **InstanSeg** | InstanSeg 3.8M-param alternative to Cellpose | `--cell-type cell --segmenter instanseg` |

`tl.*` is the primary programmatic API (markers, scoring, clustering, spatial analysis). `pl.umap()`, `io.to_spatialdata()`, and `io.read_proteomics()` are also implemented. Detection and LMD export run via CLI (`xlseg detect`, `xlseg export-lmd`).

## Next Steps

- [Getting Started](GETTING_STARTED.md) -- installation, all cell types, SLURM setup
- [Tutorial](tutorial.md) -- step-by-step cell detection and marker classification
- [API Reference](api/tl.md) -- Python API documentation
- [CLI Reference](cli-reference.md) -- all `xlseg` subcommands
