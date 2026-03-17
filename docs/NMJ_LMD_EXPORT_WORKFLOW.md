# NMJ LMD Export Workflow

This document has been consolidated into two guides:

- **[NMJ Pipeline Guide](NMJ_PIPELINE_GUIDE.md)** — Full workflow: CZI inspection → detection → annotation → classification → marker classification → LMD export
- **[LMD Export Guide](LMD_EXPORT_GUIDE.md)** — Detailed LMD export reference: cross placement, contour processing, well plate layout, batch export, replicate workflows, coordinate transforms

## Quick Reference

The end-to-end NMJ → LMD workflow:

```bash
# 1. Inspect channels
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/czi_info.py /path/to/slide.czi

# 2. Detect (always 100%, checkpointed per-tile)
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_segmentation.py \
    --czi-path /path/to/slide.czi \
    --cell-type nmj \
    --channel-spec "detect=BTX" \
    --all-channels --num-gpus 4 \
    --output-dir /path/to/output

# 3. Annotate in HTML viewer, train classifier, score detections
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/train_classifier.py \
    --detections <detections.json> --annotations <annotations.json> \
    --output-dir <output> --feature-set morph

PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/apply_classifier.py \
    --detections <detections.json> --classifier <rf_classifier.pkl> \
    --output <scored_detections.json>

# 4. Place crosses in Napari
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/napari_place_crosses.py \
    -i /path/to/slide.czi --flip-horizontal -o crosses.json

# 5. Export to LMD
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/run_lmd_export.py \
    --detections <scored_detections.json> \
    --crosses crosses.json \
    --output-dir lmd_export \
    --generate-controls --min-score 0.5 --export
```

## Well Plate Layout

384-well plate, 4 quadrants, serpentine order (B2 → B3 → C3 → C2):

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

Max 308 wells per plate. Multi-plate overflow is automatic. Empty QC wells (10%) inserted evenly.
