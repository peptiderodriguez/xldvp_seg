# Tutorial: Cell Detection and Marker Classification

This tutorial walks through a typical cell detection + marker classification
workflow on a multi-channel fluorescence CZI slide. By the end you will have
scored, marker-classified detections ready for downstream analysis or LMD export.

## Prerequisites

- xldvp_seg installed (`pip install -e .`)
- A CZI file with at least a cytoplasmic and nuclear channel
- GPU access (SLURM or local)

## Step 1: Inspect Channels

**Always inspect the CZI before writing any configuration.** Channel order in CZI
files is determined by acquisition settings and is NOT wavelength-sorted.

=== "CLI"

    ```bash
    xlseg info /path/to/slide.czi
    ```

    Example output:

    ```
      [0] AF488    Ex 493 -> Em 517 nm  Alexa Fluor 488   <- nuc488
      [1] AF647    Ex 653 -> Em 668 nm  Alexa Fluor 647   <- SMA647
      [2] AF750    Ex 752 -> Em 779 nm  Alexa Fluor 750   <- PM750
      [3] AF555    Ex 553 -> Em 568 nm  Alexa Fluor 555   <- CD31_555
    ```

    Note: channel [1]=647nm comes before [3]=555nm. Never assume wavelength order.

=== "Python"

    ```python
    from xldvp_seg.io.czi_loader import get_czi_metadata

    meta = get_czi_metadata("/path/to/slide.czi")
    for ch in meta["channels"]:
        print(f"  [{ch['index']}] {ch.get('name', '')}  Ex {ch.get('excitation_nm')} nm")
    ```

## Step 2: Run Detection

Use `--channel-spec` to resolve channels by marker name or wavelength,
avoiding fragile hard-coded indices.

=== "CLI"

    ```bash
    xlseg detect \
        --czi-path /path/to/slide.czi \
        --cell-type cell \
        --channel-spec "cyto=PM,nuc=488" \
        --all-channels \
        --num-gpus 4 \
        --output-dir /path/to/output
    ```

=== "SLURM (YAML config)"

    ```yaml
    # configs/my_experiment.yaml
    name: my_experiment
    czi_path: /path/to/slide.czi
    output_dir: /path/to/output
    cell_type: cell
    channel_map:
      cyto: PM
      nuc: 488
    all_channels: true
    html_sample_fraction: 0.10
    slurm:
      partition: p.hpcl93
      cpus: 192
      mem_gb: 556
      gpus: "l40s:4"
      time: "3-00:00:00"
    ```

    ```bash
    scripts/run_pipeline.sh configs/my_experiment.yaml
    ```

Key parameters:

- `--all-channels` enables per-channel intensity features (~15 per channel)
- `--html-sample-fraction 0.10` keeps the HTML viewer browser-friendly
- `--num-gpus` always uses the multi-GPU architecture (even with 1 GPU)
- `--sample-fraction` is always 1.0 -- detect 100% of cells

## Step 3: View and Annotate

Launch the HTML viewer to inspect detections and annotate true/false positives.

```bash
xlseg serve /path/to/output
```

This opens an interactive viewer via Cloudflare tunnel:

- Click the green checkmark for real detections
- Click the red X for false positives
- Use the Export button to save annotations as JSON

## Step 4: Train Classifier

Train a random forest classifier from your annotations.

=== "CLI"

    ```bash
    xlseg classify \
        --detections /path/to/output/cell_detections.json \
        --annotations /path/to/annotations.json \
        --output-dir /path/to/classifiers \
        --feature-set morph
    ```

=== "Python"

    ```python
    slide = SlideAnalysis.load("/path/to/output/...")
    tl.train(slide, annotations="annotations.json",
             feature_set="morph", output_dir="classifiers/")
    ```

Feature sets:

| Set | Dimensions | Description |
|-----|-----------|-------------|
| `morph` | 78 | Shape + color features (fast, often competitive) |
| `morph_sam2` | 334 | Morph + SAM2 embeddings |
| `channel_stats` | per-channel | Per-channel intensity statistics |
| `all` | 6,478 | Everything including ResNet + DINOv2 |

!!! tip
    Start with `morph`. Only add deep features (`--extract-deep-features` at
    detection time) if morph F1 is below 0.85.

## Step 5: Score Detections

Apply the trained classifier to all detections.

=== "CLI"

    ```bash
    xlseg score \
        --detections /path/to/output/cell_detections.json \
        --classifier /path/to/classifiers/rf_classifier.pkl \
        --output /path/to/output/cell_detections_scored.json
    ```

=== "Python"

    ```python
    tl.score(slide, classifier="classifiers/rf_classifier.pkl")
    ```

## Step 6: Classify Markers

Classify each cell as positive or negative for each fluorescent marker.
The default `snr` method uses median-based signal-to-noise ratio with a
threshold of 1.5.

=== "CLI"

    ```bash
    xlseg markers \
        --detections /path/to/output/cell_detections_scored.json \
        --marker-wavelength 647,555 \
        --marker-name SMA,CD31 \
        --czi-path /path/to/slide.czi
    ```

=== "Python"

    ```python
    tl.markers(slide, marker_channels=[1, 3], marker_names=["SMA", "CD31"])
    ```

This adds to each detection:

- `{marker}_class` -- "positive" or "negative"
- `{marker}_value` -- the raw intensity value
- `{marker}_threshold` -- the threshold used
- `marker_profile` -- e.g., "SMA+/CD31-"

Available methods: `snr` (default), `otsu`, `otsu_half`, `gmm`.

!!! note
    Background correction is automatic. The pipeline computes pixel-level
    background during detection, and marker classification auto-detects this
    via `ch{N}_background` keys.

!!! tip
    Alternatively, use `--marker-snr-channels "SMA:1,CD31:3"` on `xlseg detect`
    to classify markers automatically during detection -- no separate step needed.
    This uses the pre-computed SNR >= 1.5 threshold at zero extra cost.

## Step 7: Explore with Clustering

Group cells by morphological and intensity features using UMAP/t-SNE
dimensionality reduction and Leiden community detection.

=== "CLI"

    ```bash
    xlseg cluster \
        --detections /path/to/output/cell_detections_scored.json \
        --feature-groups "morph,channel" \
        --output-dir /path/to/output/clusters
    ```

=== "Python"

    ```python
    tl.cluster(slide, feature_groups="morph,channel",
               output_dir="results/clusters/")
    ```

## Step 8: Export to AnnData

Convert detections to AnnData for integration with scanpy and squidpy.

```python
from xldvp_seg.core import SlideAnalysis

slide = SlideAnalysis.load("/path/to/output/...")
adata = slide.to_anndata()

# Now use scanpy/squidpy
import scanpy as sc
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color="marker_profile")
```

The AnnData object contains:

- `X` -- feature matrix (morphological + channel features)
- `obs` -- per-cell metadata (cell_type, marker classifications, scores, cluster labels)
- `obsm["spatial"]` -- spatial coordinates (x, y in micrometers)
- `var` -- feature metadata with `feature_group` column
- `uns` -- provenance metadata (pipeline version, parameters, timestamps)

!!! tip
    After any step, run `xlseg qc /path/to/output` for a quick quality summary
    (detection count, area distribution, classifier scores, marker profiles,
    per-channel SNR) without needing the HTML viewer.

## Next Steps

- [LMD Export](LMD_EXPORT_GUIDE.md) -- export scored detections for laser microdissection
- [Output Formats](output-formats.md) -- detailed schema for all output files
- [API Reference](api/tl.md) -- full Python API documentation
