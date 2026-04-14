# Block-Face Registration & Organ Segmentation

## Recommended: Fluorescence-Native Segmentation

For slides where the fluorescence channels provide sufficient tissue contrast (Hoechst, PM, marker channels), **SAM2 can segment regions directly on fluorescence thumbnails** without any cross-modal registration. This is simpler, faster, and avoids all registration artifacts.

```bash
# 1. Segment regions (point density series, parallel SLURM)
python scripts/segment_regions.py \
    --czi-path slide.czi --display-channels "4,2" \
    --points-series 32,64,128,256,512 \
    --fill --fill-interstitial \
    --slurm --partition p.hpcl93 --mem 556G --time 1-00:00:00 \
    --viewer-after --output-dir regions/

# 2. Pick the best density, assign cells
python scripts/assign_cells_to_regions.py \
    --detections cell_detections.json \
    --label-map regions/labels_4_2_pts64_filled.npy \
    --czi-path slide.czi \
    --output cell_detections_with_regions.json

# 3. Generate viewer with nuclear stats
python scripts/generate_region_viewer.py \
    --label-maps regions/labels_4_2_pts64_filled.npy \
    --czi-path slide.czi --display-channels "4" \
    --nuc-stats regions/region_nuc_stats.json \
    --min-cells 1000 --output viewer.html
```

The block-face registration workflow below is only needed when fluorescence channels lack organ-level contrast (rare — Hoechst alone usually suffices).

---

## Block-Face Registration (when fluorescence contrast is insufficient)

Register a gross tissue photo (phone, dissection scope) to a fluorescence CZI, segment anatomical regions in the fluorescence coordinate frame, and assign detections to organs for organ-specific LMD export.

This is the **protocol as it actually works** — earlier attempts (warping label images through VALIS, warping contour points, direct ANTs/MONAI without pre-alignment) did not survive contact with real data. See "Why this order" at the bottom for what was tried and why it was abandoned.

## Inputs

| File | What | Example |
|---|---|---|
| `slide.czi` | Fluorescence mosaic (the ground truth coordinate system) | 254,898 × 111,090 px, 0.1722 µm/px |
| `photo.heic` / `.jpg` | Gross photo of the block face | phone photo, ~900×400 px |
| Nuclear channel | Used to build the fluorescence thumbnail reference | ch4 (Hoechst) |

## Overview

```
CZI                            Photo
 ↓ thumbnail (1/256 scale)       ↓ HEIC → PNG, rotate + flip
 ↓                               ↓ trim paper background
 ↓ nuclear-channel 2D            ↓
 └────────────┬──────────────────┘
              ↓ VALIS 2-pass registration (nonrigid + rigid)
              ↓ (saved as valis_warped_photo.npy — pre-compute on login
              ↓  node where pyvips works; run ANTs on compute node)
 WARPED PHOTO at fluor-thumbnail resolution
              ↓
              ↓ SAM2 auto-mask (pts=64, zero thresholds)
              ↓ smallest-first label map, Gaussian σ=5 smoothing
              ↓ (optional recursive pass for finer regions)
              ↓
 COARSE ORGAN LABEL MAP (~76 regions, in warped-photo space)
              ↓
              ↓ ANTs SyN edges refinement
              ↓ (Canny + blur + dilate edges on both images,
              ↓  one transform applied to photo AND labels)
              ↓
 FINAL LABEL MAP (aligned to CZI fluorescence, at thumbnail resolution)
              ↓
              ↓ CZI detection pipeline (run_segmentation.py)
              ↓   cell_detections.json (~650K cells in CZI px)
              ↓
              ↓ scripts/assign_cells_to_regions.py
              ↓   scales CZI px → label-map px, attaches organ_id
              ↓
 cell_detections_with_organs.json
              ↓
              ↓ per-organ UMAP / LMD export
```

## Prerequisites

```bash
# Core
pip install valis-wsi pillow-heif antspyx

# conda is needed for libvips (VALIS dependency)
mamba install -c conda-forge libvips
```

**VALIS patches** (v1.2.0 with skimage ≥0.26 + PyTorch 2.x on CUDA):

- `skimage/transform/_geometric.py:2228`:
  `if scale not in (None, 1)` → `if scale is not None and not np.array_equal(scale, 1)`
- `valis/feature_detectors.py`, `valis/feature_matcher.py`, `valis/non_rigid_registrars.py`:
  all `.detach().numpy()` → `.detach().cpu().numpy()`

**libvips / libtiff conflict**: installing ANTsPy into the same environment can break pyvips. Work around by running VALIS on the login node (where the lib versions still match) and saving the warped photo as a plain numpy array; run SAM2 + ANTs on a compute node using only that numpy file. No pyvips needed downstream.

## Step 1 — Prepare the photo

```python
import pillow_heif
from PIL import Image
import numpy as np
pillow_heif.register_heif_opener()

photo = Image.open("block_face.heic").convert("RGB")
# Orientation: block face is mirrored relative to the mounted section.
# Determine once per experiment by visual comparison.
photo = photo.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
photo.save("blockface_oriented.png")
```

Trim the paper/surface background so VALIS/SAM2 focus on tissue:

```python
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy import ndimage

arr = np.array(photo)
hsv = rgb2hsv(arr)
mask = hsv[:, :, 1] > threshold_otsu(hsv[:, :, 1]) * 0.5
mask = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=3))
arr[~mask] = 0
Image.fromarray(arr).save("blockface_trimmed.png")
```

## Step 2 — Build the fluorescence thumbnail

```python
from xldvp_seg.visualization.fluorescence import read_czi_thumbnail_channels

channels, pixel_size, mx, my = read_czi_thumbnail_channels(
    czi_path, display_channels=[4], scale_factor=1/256,  # Hoechst
)
# Save a single-channel 8-bit thumbnail for VALIS
Image.fromarray(channels[0]).save("fluor_thumbnail.jpg")
```

## Step 3 — VALIS 2-pass registration (run on login node)

The CZI fluorescence is the **ground truth**. Photo warps onto it.

```python
from valis import registration

# Pass 1: nonrigid — handles tissue deformation from sectioning
reg1 = registration.Valis(
    src_dir="pass1_in/",     # contains blockface_trimmed.png + fluor_thumbnail.jpg
    dst_dir="pass1_out/",
    reference_img_f="fluor_thumbnail.jpg",
    align_to_reference=True,
)
reg1.register()
reg1.warp_and_save_slides("pass1_out/", crop="overlap")  # may segfault on cleanup; outputs are already saved

bf1 = [s for s in reg1.slide_dict.values() if "blockface" in s.name.lower()][0]
warped_photo_1 = bf1.warp_img(np.array(Image.open("blockface_trimmed.png")),
                               interp_method="bilinear")

# Pass 2: rigid refinement on the warped output
Image.fromarray(warped_photo_1).save("pass2_in/warped_blockface.png")
reg2 = registration.Valis(
    src_dir="pass2_in/",
    dst_dir="pass2_out/",
    reference_img_f="fluor_thumbnail.jpg",
    align_to_reference=True,
)
reg2.register()
bf2 = [s for s in reg2.slide_dict.values() if "blockface" in s.name.lower()][0]
valis_warped_photo = bf2.warp_img(warped_photo_1, interp_method="bilinear")

np.save("valis_warped_photo.npy", valis_warped_photo)  # (H, W, 3) at thumbnail resolution
```

VALIS matches features with DISK + LightGlue — cross-modal BF↔FL works out of the box. Registration is non-deterministic across runs; save the warped photo as soon as you have one you like.

## Step 4 — SAM2 organ segmentation (run on compute node)

Run on the **VALIS-warped** photo, not the original. This is the key insight — masks are born in the correct coordinate frame, so no separate label warping is needed.

```python
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from scipy.ndimage import gaussian_filter

wp = np.load("valis_warped_photo.npy")

gen = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    pred_iou_thresh=0.1,          # accept almost everything
    stability_score_thresh=0.1,
    min_mask_region_area=500,
)
masks = gen.generate(wp)

# Smallest-first non-overlapping label map, smoothed
masks = sorted(masks, key=lambda m: m["area"])
label_map = np.zeros(wp.shape[:2], dtype=np.int32)
for i, m in enumerate(masks):
    seg = gaussian_filter(m["segmentation"].astype(float), sigma=5) > 0.5
    label_map[(label_map == 0) & seg] = i + 1

np.save("organ_label_map_coarse.npy", label_map)
```

For sub-organ granularity, a **recursive pass** (crop each region, re-run SAM2 with point count scaled to `max(4, min(128, int(sqrt(area_px / 10))))`) typically doubles the region count. For large whole-mouse sections the ~76-region coarse pass is usually enough.

## Step 5 — ANTs SyN edges refinement

SAM2 masks are close but not pixel-perfect. Align edge maps of the warped photo and fluorescence using ANTs' diffeomorphic SyN transform, then apply the same transform to the label map.

```python
import ants, cv2

def edges(img_gray, blur=3, dilate=2):
    e = cv2.Canny(img_gray, 50, 150)
    e = cv2.GaussianBlur(e, (blur*2+1, blur*2+1), 0)
    return cv2.dilate(e, np.ones((dilate, dilate), np.uint8))

photo_gray = cv2.cvtColor(wp, cv2.COLOR_RGB2GRAY)
fluor_gray = np.array(Image.open("fluor_thumbnail.jpg"))

ep = edges(photo_gray)
ef = edges(fluor_gray)

result = ants.registration(
    fixed=ants.from_numpy(ef.astype(np.float32)),
    moving=ants.from_numpy(ep.astype(np.float32)),
    type_of_transform="SyNRA",
    syn_metric="meansquares",
)

# Apply the same transform to the label map (nearest-neighbor)
labels_ants = ants.from_numpy(label_map.astype(np.float32))
labels_aligned = ants.apply_transforms(
    fixed=ants.from_numpy(ef.astype(np.float32)),
    moving=labels_ants,
    transformlist=result["fwdtransforms"],
    interpolator="nearestNeighbor",
).numpy().astype(np.int32)

np.save("labels_final.npy", labels_aligned)
```

## Step 6 — Run CZI cell detection

Independent of registration. Standard pipeline:

```yaml
# configs/<experiment>.yaml
name: my_experiment
czi_path: /path/to/slide.czi
output_dir: /path/to/detection
cell_type: cell
channel_map:
  cyto: 750       # membrane marker
  nuc: Hoechst
all_channels: true
slurm:
  partition: p.hpcl93
  cpus: 128
  mem_gb: 556
  gpus: "l40s:4"
  time: "3-00:00:00"
```

```bash
scripts/run_pipeline.sh configs/my_experiment.yaml
```

Produces `<run_dir>/cell_detections.json`.

## Step 7 — Assign cells to organ regions

```bash
PYTHONPATH=$REPO $XLDVP_PYTHON $REPO/scripts/assign_cells_to_regions.py \
    --detections <run_dir>/cell_detections.json \
    --label-map labels_final.npy \
    --czi-path /path/to/slide.czi \
    --output <run_dir>/cell_detections_with_organs.json
```

The script is cell-type-agnostic: it reads CZI-space centroids via `extract_positions_um`, derives the scale from `czi_mosaic_shape / label_map.shape`, and attaches `organ_id` (0 = background) to every detection. A companion `<output>.regions.json` summarizes cells per region.

## Step 8 — Per-organ downstream (future)

Each region with enough cells can be its own analysis unit:

- **UMAP / clustering** — `xlseg cluster --detections liver_detections.json ...`
- **LMD export** — standard `xlseg export-lmd` per organ
- **Sliding-window sampling** — `scripts/sliding_window_sampling.py` for spatially-stratified organ-resolved plates

## Why this order

Earlier approaches that were tried and failed:

| Approach | Why it failed |
|---|---|
| SAM2 on original photo → warp labels through VALIS | Nearest-neighbor warping of a dense label image produces artifacts at region boundaries; labels drift |
| Warp contour points through VALIS transform | `warp_xy` and `warp_img` live in different internal coordinate spaces; assembling a consistent full-res result was brittle |
| Direct ANTs / MONAI without VALIS pre-alignment | Photo (portrait) gets squashed to fluorescence aspect ratio; no nonrigid coarse alignment means SyN can't recover |
| DISK+LightGlue directly between photo and fluor | Zero cross-modal matches — modalities too different without the VALIS feature normalization |
| Optical flow | Assumes brightness constancy — fails on BF↔FL |
| SimpleITK BSpline mutual information | Dense grids produce wavy artifacts; sparse grids don't recover bowel deformation |

The winning order — VALIS first for coarse nonrigid, SAM2 on the warped photo so masks are born correct, ANTs SyN edges for fine refinement — is the combination that actually works end-to-end.

## Outputs (for record)

| File | Shape / Contents |
|---|---|
| `valis_warped_photo.npy` | `(H, W, 3)` uint8 at fluorescence-thumbnail resolution |
| `organ_label_map_coarse.npy` | `(H, W)` int32 with region IDs, background=0 |
| `labels_final.npy` | post-ANTs label map, aligned to fluorescence edges |
| `cell_detections.json` | from `run_segmentation.py` |
| `cell_detections_with_organs.json` | adds `organ_id` to every detection |
| `cell_detections_with_organs.json.regions.json` | cells-per-region summary |

## Notes

- All intermediate arrays should be saved as `.npy` and not re-derived. VALIS is non-deterministic and ANTs runs for minutes; you don't want to recompute.
- The coarse SAM2 pass (≈76 regions on a whole-mouse section) is often the sweet spot — enough granularity for organ-resolved LMD, few enough that empty regions don't dominate.
- Cells on a region boundary get snapped to whichever pixel their centroid lands in. For boundary-aware assignment (distance-to-boundary scoring), extend `assign_cells_to_regions.py`.
- Phase 3 of the detection pipeline (intensity feature re-extraction) is currently GIL-bound and slow on >500K-cell slides. Known issue; fix pending.
