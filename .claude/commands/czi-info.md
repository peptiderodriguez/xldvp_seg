You are inspecting CZI microscopy files for the xldvp_seg pipeline.

## What to do

**Step 1 — Get the CZI path.** If not provided in the arguments, ask the user for the path to their CZI file (or directory of CZIs).

**Step 2 — Run czi_info.** For each CZI file:
```bash
PYTHONPATH=$REPO $MKSEG_PYTHON $REPO/scripts/czi_info.py <path>
```
Where `$REPO` is the repo root and `$MKSEG_PYTHON` is the mkseg Python binary.

**Step 3 — Interpret the results.** For each channel, suggest what it likely is:
- Wavelengths around 405-488nm with names like "DAPI", "Hoechst", "nuc" → Nuclear stain
- 488nm with "GFP", "Alexa488" → Green fluorescent marker
- 555nm with "tdTomato", "Cy3" → Red fluorescent marker
- 647nm with "BTX", "bungarotoxin" → NMJ detection channel (BTX labels acetylcholine receptors)
- 647nm with "Alexa647" → Far-red marker
- 750nm with "NFL", "neurofilament" → Nerve fiber marker
- Channels with "SMA" or smooth muscle → Vessel SMA channel
- Channels with "CD31" or endothelial → Vessel CD31 channel

**Step 3b — Auto-parse filename markers.** Run:
```bash
$MKSEG_PYTHON -c "from segmentation.io.czi_loader import parse_markers_from_filename; import json; print(json.dumps(parse_markers_from_filename('<filename>'), indent=2))"
```
Show the parsed marker→wavelength mappings alongside the CZI metadata to build the complete channel table. Ask the user to confirm the mapping.

**Step 4 — Recommend a pipeline.** Based on channels, suggest `--channel-spec` (preferred) or raw indices:
- Has BTX channel → "This looks like an NMJ slide. Use `--cell-type nmj --channel-spec 'detect=BTX'`" (or `--channel <BTX_index>`)
- Has SMA + CD31 → "This looks like a vessel slide. Use `--cell-type vessel --channel-spec 'detect=SMA'`"
- Has nuclear + marker → "Generic cell detection with Cellpose. Use `--cell-type cell --channel-spec 'cyto=<marker>,nuc=<nuclear>'`" (or `--cellpose-input-channels <cyto>,<nuc>`)
- Brightfield only → "MK/bone marrow detection. Use `--cell-type mk --channel 0`"

`--channel-spec` resolves marker names and wavelengths to CZI channel indices automatically at startup.

**Step 5 — Show mosaic info.** Display dimensions and estimate:
- Total pixels (width x height x channels)
- Approximate memory needed (pixels x 2 bytes for uint16)
- Estimated processing time (rough: ~1 min/1000 tiles at 4 GPUs)

$ARGUMENTS
