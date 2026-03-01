You are previewing preprocessing effects on CZI microscopy data for the xldvp_seg pipeline.

---

## What to do

**Step 1 — Get inputs.** If not provided in arguments, ask:
- CZI file path
- Channel: can be an index (e.g., 1), wavelength (e.g., 647), marker name (e.g., SMA), or "all". If the user gives a name/wavelength, resolve it using `parse_markers_from_filename()` + `resolve_channel_indices()` from `segmentation.io.czi_loader`.
- Which preprocessing: flat-field, photobleach, row/column normalization, or all

**Step 2 — Detect system** (silently):
```bash
$MKSEG_PYTHON $REPO/scripts/system_info.py --json
```

**Step 3 — Run the preview.**

This is lightweight and OK to run on a SLURM login node (uses 1/8 resolution by default):

```bash
$MKSEG_PYTHON $REPO/scripts/preview_preprocessing.py \
    --czi-path <path> \
    --channel <N> \
    --preprocessing <flat_field|photobleach|rowcol|all> \
    --output-dir <output_dir>/preview/ \
    --scale-factor 8
```

For a more detailed comparison with illumination profiles and row/column mean plots:
```bash
$MKSEG_PYTHON $REPO/scripts/visualize_corrections.py \
    --czi-path <path> \
    --channel <N> \
    --output-dir <output_dir>/corrections/
```

**Step 4 — Show results.** List the generated PNG files and their paths. If on a cluster with Cloudflare tunnel available, offer to serve the output directory for browser viewing.

**Step 5 — Interpret.** Help the user decide:
- **Flat-field**: Corrects uneven illumination across the slide. Look for: dark corners/edges becoming uniform. Always recommended for tiled mosaic data.
- **Photobleach**: Corrects intensity decay from sequential tile scanning. Look for: gradient across scan direction being removed. Useful if intensity drops noticeably from first to last tile.
- **Row/column normalization**: Corrects banding artifacts from sensor readout. Look for: horizontal/vertical stripes being removed. Only needed if visible banding present.

Recommend which corrections to enable in the full pipeline run based on the preview results.

$ARGUMENTS
