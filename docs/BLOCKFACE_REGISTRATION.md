# Block-Face Registration & Organ Segmentation

Register a gross tissue photo (phone, dissection scope) to fluorescence microscopy, segment anatomical regions, and assign detections to organs for organ-specific LMD export.

## Overview

```
Phone photo of block face
    ↓ rotate + flip (manual, one-time)
    ↓ SAM2 auto-mask (zero thresholds, recursive, area-scaled points)
    → 400+ anatomical regions on the photo
    ↓ VALIS nonrigid registration (2-pass: nonrigid → rigid refinement)
    → transform mapping photo ↔ CZI coordinates
    ↓ warp region polygons to CZI full-res space
    ↓ assign detections to regions
    → organ-labeled detections → LMD export per organ
```

## Prerequisites

```bash
pip install valis-wsi pillow-heif
```

**VALIS requires `libvips`** — install via conda on HPC:
```bash
mamba install -c conda-forge libvips
```

**Known patches needed** (VALIS 1.2.0 + skimage 0.26 + PyTorch 2.x):
- `skimage/transform/_geometric.py:2228`: change `if scale not in (None, 1)` to `if scale is not None and not np.array_equal(scale, 1)`
- `valis/feature_detectors.py`, `valis/feature_matcher.py`, `valis/non_rigid_registrars.py`: change all `.detach().numpy()` to `.detach().cpu().numpy()`

## Step 1: Prepare the Photo

Convert from phone format (HEIC) and determine the correct orientation by comparing with a CZI thumbnail:

```python
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

# Convert HEIC
photo = Image.open("block_face.heic").convert("RGB")

# Determine orientation: compare all 4 rotations against CZI thumbnail
# Block face is MIRRORED relative to mounted section
# Typical: rotate 90° CCW + flip horizontal
photo = photo.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
photo.save("blockface_oriented.png")
```

**Trim background** — mask out the white paper/surface so SAM2 focuses on tissue:

```python
from skimage.color import rgb2hsv
from scipy import ndimage

hsv = rgb2hsv(np.array(photo))
mask = hsv[:,:,1] > threshold_otsu(hsv[:,:,1]) * 0.5
mask = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=3))
photo_trimmed = photo.copy()
photo_trimmed[~mask] = 0
```

## Step 2: SAM2 Organ Segmentation

### Initial segmentation (zero thresholds, smallest-first)

```python
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

gen = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    pred_iou_thresh=0.0,       # accept everything
    stability_score_thresh=0.0,
    min_mask_region_area=300,
)
masks = gen.generate(photo)
```

Build a non-overlapping label map (smallest masks claim pixels first):

```python
masks = sorted(masks, key=lambda m: m["area"])  # smallest first
label_map = np.zeros(photo.shape[:2], dtype=np.int32)
for i, m in enumerate(masks):
    seg = gaussian_filter(m["segmentation"].astype(float), sigma=5) > 0.5
    label_map[(label_map == 0) & seg] = i + 1
```

### Recursive refinement (area-scaled points)

For each region, crop it and re-run SAM2 with points scaled to the region's area:

```python
pts = max(4, min(128, int(math.sqrt(area_px / 10))))
```

This gives large regions (liver, muscle) up to 128 points to find sub-structures, while small regions (individual GI loops) get just 4-7 points. The recursive pass typically doubles the region count (e.g., 219 → 452).

### Key parameters

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `points_per_side` | Grid density of prompt points | 64 for initial, area-scaled for recursive |
| `pred_iou_thresh` | Minimum predicted IoU to keep a mask | 0.0 (accept all) |
| `stability_score_thresh` | Minimum stability to keep a mask | 0.0 (accept all) |
| `min_mask_region_area` | Minimum mask size in pixels | 300 |
| Gaussian sigma on masks | Smooths jagged boundaries | 5 |

## Step 3: VALIS Registration (2-pass)

The CZI fluorescence image is the **ground truth** coordinate system. The photo warps onto it.

**Pass 1 — Nonrigid**: handles tissue deformation from sectioning/mounting.

```python
from valis import registration

reg1 = registration.Valis(
    src_dir="inputs/",           # blockface.png + fluorescence_thumbnail.jpg
    dst_dir="pass1_output/",
    reference_img_f="fluorescence_thumbnail.jpg",  # CZI is ground
    align_to_reference=True,
)
reg1.register()
reg1.warp_and_save_slides("pass1_output/", crop="overlap")
```

**Pass 2 — Rigid refinement**: takes the warped output from pass 1 and fine-tunes alignment.

```python
# Use warped blockface from pass 1 as input
reg2 = registration.Valis(
    src_dir="pass2_inputs/",     # warped_blockface.png + fluorescence_thumbnail.jpg
    dst_dir="pass2_output/",
    reference_img_f="fluorescence_thumbnail.jpg",
    align_to_reference=True,
)
reg2.register()
```

VALIS uses deep feature matching (DISK + LightGlue) — handles cross-modal matching (brightfield ↔ fluorescence) automatically.

## Step 4: Transform Organ Polygons to CZI Space

Region polygons from Step 2 are in photo pixel coordinates. To map them to full-resolution CZI coordinates:

1. Apply Pass 1 transform (nonrigid warp)
2. Apply Pass 2 transform (rigid refinement)
3. Scale from thumbnail coordinates to full-res CZI coordinates:
   `full_res_xy = thumbnail_xy × (czi_width / thumbnail_width)`

## Step 5: Assign Detections to Organs

Use existing `xldvp_seg` ROI tools:

```python
from xldvp_seg.roi.common import filter_detections_by_roi_mask

# Rasterize transformed organ polygons at downsampled resolution (256x)
# to avoid 113 GB full-res allocation
organ_detections = filter_detections_by_roi_mask(
    detections, organ_label_map, downsample=256, x_start=0, y_start=0
)
# Each detection now has 'roi_id' field → organ assignment
```

## Step 6: Organ-Specific LMD Export

Standard pipeline per organ:

```bash
# Filter detections by organ, then export
xlseg export-lmd --detections liver_detections.json \
    --crosses crosses.json --output-dir lmd_liver/ --export
```

## CZI Thumbnail Generation

```python
from xldvp_seg.visualization.fluorescence import read_czi_thumbnail_channels

channels, pixel_size, mx, my = read_czi_thumbnail_channels(
    czi_path, display_channels=[4], scale_factor=1/256
)
```

## Step 7: Per-Region Analysis & LMD Sampling

With detections assigned to anatomical regions, each region becomes its own analysis unit:

1. **Filter regions with cells** — discard empty or near-empty regions (e.g., GI lumen contents)
2. **Per-region UMAP** — for each region with enough cells, run feature clustering:
   ```python
   tl.cluster(region_slide, feature_groups="morph,channel", output_dir=f"clustering/{region_name}/")
   ```
3. **Explore morphological diversity** — each region's UMAP reveals cell populations within that organ (e.g., periportal vs pericentral hepatocytes, alveolar types I vs II in lung)
4. **Sample from each space** — select representative cells from each cluster within each region for organ-specific LMD:
   ```bash
   scripts/sliding_window_sampling.py --detections liver_detections.json \
       --czi-path slide.czi --output-dir lmd_liver/
   ```

This gives you: **organ-resolved, morphologically-stratified cell sampling** — every LMD plate contains cells from defined anatomical regions with known morphological diversity.

## Notes

- The rotation/flip between block face and mounted section depends on how the tissue was sectioned and picked up. Determine once per experiment by visual comparison.
- VALIS registration quality depends on tissue contrast. Hoechst/DAPI channel works well as the fluorescence reference.
- For very large slides (>200K pixels), generate the CZI thumbnail at 1/256 scale (~1000px wide) for registration.
- SAM2's zero-threshold mode finds ~200-400 regions on a typical whole-mouse section — the recursive pass doubles this.
- All intermediate results (label maps, overlays, transforms) should be saved for reproducibility.
