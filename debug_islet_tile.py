#!/usr/bin/env python3
"""Quick diagnostic: run islet detection on one tile and trace where detections are lost."""
import sys
import gc
import numpy as np
import torch
sys.path.insert(0, '/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg')

from segmentation.io.czi_loader import get_loader
from segmentation.detection.strategies.islet import IsletStrategy, _percentile_normalize_channel
from segmentation.utils.feature_extraction import extract_morphological_features

CZI = '/fs/pool/pool-mann-edwin/marvin_test/2025_09_03_30610012_BS-100.czi'
SAM2_CKPT = '/fs/gpfs41/lv12/fileset02/pool/pool-mann-edwin/code_bin/xldvp_seg/checkpoints/sam2.1_hiera_large.pt'
SAM2_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
TILE_SIZE = 3000
PIXEL_SIZE = 0.325

# --- Load channels ---
print("Loading CZI...")
loader = get_loader(CZI, channel=1, load_to_ram=True)
all_ch = {1: loader.channel_data}
for ch in [0, 2, 3, 4, 5]:
    loader.load_channel(ch)
    all_ch[ch] = loader.get_channel_data(ch)

for ch in sorted(all_ch.keys()):
    nz = all_ch[ch][all_ch[ch] > 0]
    print(f"  ch{ch}: nz_mean={float(nz.mean()) if len(nz) > 0 else 0:.1f}")

# --- Pick a tile ---
tile_x, tile_y = 2700, 2700
tile_h = min(TILE_SIZE, all_ch[1].shape[0] - tile_y)
tile_w = min(TILE_SIZE, all_ch[1].shape[1] - tile_x)
print(f"\nTile ({tile_x}, {tile_y}), size {tile_w}x{tile_h}")

extra_channels = {}
for ch in range(6):
    extra_channels[ch] = all_ch[ch][tile_y:tile_y+tile_h, tile_x:tile_x+tile_w]

# Build tile_rgb (same as multi-GPU worker: first 3 channels, /256)
tile_raw_3ch = np.stack([extra_channels[0], extra_channels[1], extra_channels[2]], axis=-1)
tile_rgb = (tile_raw_3ch / 256).astype(np.uint8)
print(f"tile_rgb: shape={tile_rgb.shape}, max={tile_rgb.max()}")

# --- Load models ---
print("\nLoading Cellpose...")
from cellpose import models as cellpose_models
cellpose_model = cellpose_models.CellposeModel(gpu=True, pretrained_model='cpsam')

print("Loading SAM2...")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device='cuda')
sam2_predictor = SAM2ImagePredictor(sam2_model)

# --- Run IsletStrategy.segment() step by step ---
print("\n=== SEGMENT STEP BY STEP ===")
strategy = IsletStrategy(
    membrane_channel=1, nuclear_channel=4,
    min_area_um=30, max_area_um=500, max_candidates=1000,
)

# Step 1: Cellpose
membrane_u8 = _percentile_normalize_channel(extra_channels[1])
nuclear_u8 = _percentile_normalize_channel(extra_channels[4])
cellpose_input = np.stack([membrane_u8, nuclear_u8], axis=-1)
cellpose_masks, _, _ = cellpose_model.eval(cellpose_input, channels=[1, 2])
cellpose_ids = np.unique(cellpose_masks)
cellpose_ids = cellpose_ids[cellpose_ids > 0]
print(f"Cellpose: {len(cellpose_ids)} cells")

# Limit candidates
if len(cellpose_ids) > 1000:
    areas = [(cp_id, (cellpose_masks == cp_id).sum()) for cp_id in cellpose_ids]
    areas.sort(key=lambda x: x[1], reverse=True)
    cellpose_ids = np.array([a[0] for a in areas[:1000]])
    print(f"  Limited to {len(cellpose_ids)} largest")

# Step 2: SAM2 refinement
from scipy import ndimage
sam2_rgb = np.stack([membrane_u8, membrane_u8, membrane_u8], axis=-1)
sam2_predictor.set_image(sam2_rgb)

candidates = []
for cp_id in cellpose_ids[:20]:  # just test first 20
    cp_mask = cellpose_masks == cp_id
    cy, cx = ndimage.center_of_mass(cp_mask)
    point_coords = np.array([[cx, cy]])
    point_labels = np.array([1])
    masks_pred, scores, _ = sam2_predictor.predict(
        point_coords=point_coords, point_labels=point_labels, multimask_output=True,
    )
    best_idx = np.argmax(scores)
    sam2_mask = masks_pred[best_idx]
    if sam2_mask.dtype != bool:
        sam2_mask = (sam2_mask > 0.5).astype(bool)
    area_px = sam2_mask.sum()
    area_um = area_px * PIXEL_SIZE**2
    candidates.append({
        'mask': sam2_mask, 'score': float(scores[best_idx]),
        'area_px': area_px, 'area_um': area_um,
    })

sam2_predictor.reset_predictor()
torch.cuda.empty_cache()

print(f"\nSAM2 refinement (first 20):")
for i, c in enumerate(candidates):
    status = "PASS" if 30 <= c['area_um'] <= 500 else "FAIL"
    print(f"  {i}: area={c['area_px']}px={c['area_um']:.1f}um2, score={c['score']:.3f} [{status}]")

pass_filter = sum(1 for c in candidates if 30 <= c['area_um'] <= 500)
print(f"\n  {pass_filter}/{len(candidates)} pass area filter")

# Step 3: Check extract_morphological_features
print("\n=== FEATURE EXTRACTION ===")
for i, c in enumerate(candidates[:5]):
    feat = extract_morphological_features(c['mask'], tile_rgb)
    if feat:
        print(f"  mask {i}: area={feat.get('area', 0)} features={len(feat)} keys")
    else:
        print(f"  mask {i}: extract_morphological_features returned EMPTY")

# Step 4: Full detect() pipeline
print("\n=== FULL DETECT ===")
sam2_model2 = build_sam2(SAM2_CFG, SAM2_CKPT, device='cuda')
sam2_pred2 = SAM2ImagePredictor(sam2_model2)

models_dict = {
    'cellpose': cellpose_model,
    'sam2_predictor': sam2_pred2,
    'device': torch.device('cuda'),
}

label_array, detections = strategy.detect(
    tile_rgb, models_dict, PIXEL_SIZE,
    extra_channels=extra_channels,
)

print(f"detect() returned: {len(detections)} detections")
if detections:
    for d in detections[:10]:
        a = d.features.get('area', 0)
        print(f"  {d.id}: score={d.score:.3f}, area={a}px={a*PIXEL_SIZE**2:.1f}um2")
else:
    print("  Zero detections — checking segment() output...")
    sam2_model3 = build_sam2(SAM2_CFG, SAM2_CKPT, device='cuda')
    sam2_pred3 = SAM2ImagePredictor(sam2_model3)
    models3 = {'cellpose': cellpose_model, 'sam2_predictor': sam2_pred3}
    raw_masks = strategy.segment(tile_rgb, models3, extra_channels=extra_channels)
    print(f"  segment() returned {len(raw_masks)} masks")
    if raw_masks:
        areas_um = [m.sum() * PIXEL_SIZE**2 for m in raw_masks]
        print(f"  Area range: {min(areas_um):.1f}-{max(areas_um):.1f} um2")
        n_pass = sum(1 for a in areas_um if 30 <= a <= 500)
        print(f"  {n_pass}/{len(raw_masks)} pass area filter")

# Step 5: Full process_single_tile
print("\n=== PROCESS_SINGLE_TILE ===")
from segmentation.processing.tile_processing import process_single_tile
sam2_model4 = build_sam2(SAM2_CFG, SAM2_CKPT, device='cuda')
sam2_pred4 = SAM2ImagePredictor(sam2_model4)
models4 = {
    'cellpose': cellpose_model,
    'sam2_predictor': sam2_pred4,
    'device': torch.device('cuda'),
}

result = process_single_tile(
    tile_rgb=tile_rgb,
    extra_channel_tiles=extra_channels,
    strategy=strategy,
    models=models4,
    pixel_size_um=PIXEL_SIZE,
    cell_type='islet',
    slide_name='test_slide',
    tile_x=tile_x,
    tile_y=tile_y,
)

if result is None:
    print("  process_single_tile returned None")
else:
    masks_out, features_list = result
    print(f"  Returned {len(features_list)} features, masks shape={masks_out.shape}")

print("\nDone.")
