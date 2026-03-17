You are guiding the user through **vessel community analysis** for the xldvp_seg pipeline. This workflow takes classified cell detections and performs multi-scale spatial clustering to identify vessel structures.

---

## Workflow Steps

### Step 1: Check Prerequisites

1. **Find the detection run directory** — look for `cell_detections.json` or `cell_detections_classified.json`
2. **Check if marker classification is done** — look for `cell_detections_classified.json` with `SMA_class` and `CD31_class` fields
3. If not classified yet, run `classify_markers.py` first:
   ```bash
   PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/classify_markers.py \
       --detections $RUN_DIR/cell_detections.json \
       --marker-wavelength 647,555 \
       --czi-path $CZI \
       --marker-name SMA,CD31 \
       --method otsu
   ```
   **IMPORTANT**: Always run `czi_info.py` first to verify channel-to-wavelength mapping.

### Step 2: Run Vessel Community Analysis

```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/vessel_community_analysis.py \
    --detections $RUN_DIR/cell_detections_classified.json \
    --radii 25,50,100,200 \
    --min-cells 3 \
    --best-radius 50 \
    --output-dir $RUN_DIR/vessel_analysis/ \
    --generate-viewer \
    --run-squidpy \
    --run-leiden \
    --snr-channels 1,3
```

**Key options:**
- `--radii`: Spatial radii for connected component clustering (um). Smaller = capillaries, larger = arteries.
- `--best-radius`: Which radius to use for primary cell-to-structure assignment (default: 50 um)
- `--min-cells`: Minimum cells per structure (default: 3)
- `--snr-channels`: Map marker SNR to channel indices (e.g. `1,3` → `ch1_snr` for SMA, `ch3_snr` for CD31)
- `--generate-viewer`: Create interactive HTML viewer with zoom/pan
- `--run-squidpy`: Run neighborhood enrichment, co-occurrence, Ripley's L
- `--run-leiden`: Leiden clustering on full feature space (morph + SNR + SAM2)
- `--leiden-resolution`: Leiden resolution (default: 0.5, higher = more clusters)

### Step 3: Review Results

Check the outputs:
- **`vessel_structures.csv`**: One row per structure — morphology, vessel type, marker composition, SNR values
- **`cell_detections_vessel_analysis.json`**: Enriched detections with `vessel_type`, `vessel_morphology`, `vessel_community_id`, `leiden_cluster`
- **`leiden/`**: UMAP, spatial scatter, cluster composition plots, h5ad file
- **`squidpy/`**: Neighborhood enrichment, co-occurrence, Ripley's L plots

**Interpretation guide:**
- **Morphology**: ring (circular), arc (curved), linear (elongated), cluster (compact)
- **Vessel types**: artery_like (SMA+ ring), vein_like (SMA+ cluster/arc), capillary_like (small CD31+), endothelial_network (CD31+ linear)
- **SNR values**: Higher mean_sma_snr = stronger smooth muscle expression, higher mean_cd31_snr = stronger endothelial expression

### Step 4: Visualize

If `--generate-viewer` was used, serve the HTML:
```bash
$MKSEG_PYTHON $REPO/serve_html.py --directory $RUN_DIR/vessel_analysis/
```

The viewer supports zoom/pan, cell coloring by vessel_type or vessel_morphology.

---

## Notes

- SNR (signal-to-noise ratio) is used throughout for marker classification and vessel type inference
- The script uses the same morphology classification as `generate_multi_slide_spatial_viewer.py` (PCA elongation, circularity, hollowness)
- **Memory**: 306K cells needs ~50-100GB RAM for Leiden/viewer. Submit as SLURM job on p.hpcl8 with `--mem=370G --cpus-per-task=24`
- `--snr-channels` is required when pipeline bg correction stored SNR as `ch{N}_snr` instead of `{marker}_snr`
- The viewer uses a slim JSON (coords + group fields) to avoid OOM on the 4GB full JSON
- For more details: `docs/VESSEL_COMMUNITY_ANALYSIS.md`
