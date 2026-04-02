You are inspecting CZI microscopy files for the xldvp_seg pipeline.

## What to do

**Step 1 — Get the CZI path.** If not provided in the arguments, ask the user for the path to their CZI file (or directory of CZIs).

**Step 2 — Run czi_info.** For each CZI file:
```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/czi_info.py <path>
```
Where `$REPO` is the repo root and `$XLDVP_PYTHON` is the xldvp_seg Python binary.

**Step 3 — Build the confirmed channel table.** The `czi_info.py` output is the authoritative channel order — CZI indices are NOT wavelength-sorted and cannot be inferred from the filename alone.

Also parse the filename to match antibody names to fluorophores:
```bash
$XLDVP_PYTHON -c "from xldvp_seg.io.czi_loader import parse_markers_from_filename; import json; print(json.dumps(parse_markers_from_filename('<filename>'), indent=2))"
```

Combine both outputs into a table and **show it to the user for confirmation before doing anything else**:
```
Index  Ex→Em      Fluorophore        Marker (from filename)   Role
[0]    493→517nm  Alexa Fluor 488    nuc488                   nuclear (nuc input)
[1]    653→668nm  Alexa Fluor 647    SMA647                   marker classification
[2]    752→779nm  Alexa Fluor 750    PM750                    cytoplasm (cyto input)
[3]    553→568nm  Alexa Fluor 555    CD31_555                 marker classification
```

Use the Em wavelength column to match fluorophores to markers (e.g. Em=668nm = AF647 = whatever 647nm marker is in the filename). Do not assume — verify each row.

**Step 3b — Ask about channel exclusions.** *"Do any channels have failed stains or should be excluded? (e.g., a PDGFRa channel where the antibody didn't work)"* If yes, note the index — this feeds into `load_channels: "0,1,2"` (YAML) or `--channels "0,1,2"` (CLI).

**Step 4 — Recommend a pipeline.** Based on the confirmed channel table, suggest `--channel-spec` (preferred) or explicit indices:
- Has BTX channel → "This looks like an NMJ slide. Use `--cell-type nmj --channel-spec 'detect=BTX'`" (or `--channel <BTX_index>`)
- Has SMA + CD31 → "This looks like a vessel slide. Use `--cell-type vessel --channel-spec 'detect=SMA'`"
- Has nuclear + marker → "Generic cell detection with Cellpose. Use `--cell-type cell --channel-spec 'cyto=<marker>,nuc=<nuclear>'`" (or `--cellpose-input-channels <cyto>,<nuc>`)
- Brightfield only → "MK/bone marrow detection. Use `--cell-type mk --channel 0`"

`--channel-spec` resolves marker names and wavelengths to CZI channel indices automatically at startup.

**Step 5 — Show mosaic info.** Display dimensions and estimate:
- Total pixels (width x height x channels)
- Approximate memory needed (pixels x 2 bytes for uint16)
- Estimated processing time (rough: ~1 min/1000 tiles at 4 GPUs)

---

## Adaptive Guidance

**After showing channel table (Step 3):**
- If a channel has very low emission wavelength separation from another (<20nm difference): *"Channels [X] and [Y] have close emission wavelengths — check for spectral bleedthrough if results look noisy."*
- If filename markers don't match fluorophore assignments cleanly: *"Some markers in the filename couldn't be confidently matched to CZI channels. Please confirm the mapping above before proceeding."*
- Always emphasize: *"CZI channel order is set by the acquisition software, NOT by wavelength. The table above is the ground truth."*

**After channel exclusion check (Step 3b):**
- If user reports a failed stain: *"Got it — excluding channel [X]. This saves memory and prevents the failed stain from confusing marker classification. Use --channels 'N,N,N' to load only the good channels."*
- If all channels look good: *"All channels look usable. The pipeline will load all of them by default."*

**After pipeline recommendation (Step 4):**
- If the slide matches multiple pipelines (e.g., has both BTX and cell markers): *"This slide could work with either NMJ or generic cell detection, depending on what you're looking for. NMJ detection is tuned for neuromuscular junction morphology; cell detection uses Cellpose for general segmentation."*
- If the slide has only 1 channel: *"Single-channel slide — MK detection or NMJ (if it's a BTX stain) are your options. Cell detection needs at least 2 channels (cyto + nuc)."*
- If many channels (>4): *"Rich multi-channel slide — --all-channels will extract per-channel intensity features for all loaded channels, which helps the classifier distinguish marker combinations."*

**After memory estimate (Step 5):**
- If estimated memory > 200 GB: *"This is a large slide. Direct-to-SHM loading keeps peak memory manageable, but consider --num-gpus 1 if RAM is tight."*
- If < 50 GB: *"Moderate-sized slide — should run comfortably on a single node with default settings."*

$ARGUMENTS
