#!/usr/bin/env python3
"""MK Maturation Staging v2 — Nuclear Deep Features.

4-phase analysis:
  phase1_load    — Load features, filter, dedup, save crops + metadata (no PCA)
  phase2_nuclear — Otsu nuclear seg + SAM2/ResNet deep features on nucleus (GPU) + PCA
  phase3_cluster — Cluster on nuclear PCA features
  phase4_validate — Validate clusters, pseudotime, group comparison plots

Usage:
  python maturation_analysis.py phase1_load --input-dir ... --output ...
  python maturation_analysis.py phase2_nuclear --input ... --output ... [--num-gpus 2]
  python maturation_analysis.py phase3_cluster --input ... --output ...
  python maturation_analysis.py phase4_validate --data ... --nuclear ... --clusters ... --output-dir ...
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, spectral_embedding
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PIXEL_SIZE_UM = 0.1725

# Group mapping
def slide_to_group(slide_name):
    """Extract group (FGC/FHU/MGC/MHU) from slide name like '2025_11_18_FGC1'."""
    parts = slide_name.split('_')
    for p in parts:
        for grp in ['FGC', 'FHU', 'MGC', 'MHU']:
            if p.startswith(grp):
                return grp
    return 'UNKNOWN'

GROUP_COLORS = {
    'FGC': '#e74c3c',  # red (female)
    'FHU': '#e67e22',  # orange (female)
    'MGC': '#3498db',  # blue (male)
    'MHU': '#2ecc71',  # green (male)
}
GROUP_ORDER = ['FGC', 'FHU', 'MGC', 'MHU']


# =============================================================================
# Phase 1: Load, filter, dedup
# =============================================================================

def deduplicate_samples(samples, distance_threshold_px=50):
    """Remove duplicate detections from tile overlap. Keeps larger area."""
    if not samples:
        return samples

    by_slide = defaultdict(list)
    for s in samples:
        by_slide[s['slide']].append(s)

    kept = []
    total_removed = 0

    for slide, slide_samples in by_slide.items():
        slide_samples.sort(key=lambda s: s.get('area_px', 0), reverse=True)

        grid = {}
        cell_size = distance_threshold_px
        slide_kept = []

        for s in slide_samples:
            x, y = s['global_x'], s['global_y']
            gx, gy = int(x // cell_size), int(y // cell_size)

            is_dup = False
            for dx in (-1, 0, 1):
                if is_dup:
                    break
                for dy in (-1, 0, 1):
                    for existing in grid.get((gx + dx, gy + dy), []):
                        dist = ((x - existing['global_x'])**2 + (y - existing['global_y'])**2) ** 0.5
                        if dist < distance_threshold_px:
                            is_dup = True
                            break
                    if is_dup:
                        break

            if not is_dup:
                slide_kept.append(s)
                grid.setdefault((gx, gy), []).append(s)

        total_removed += len(slide_samples) - len(slide_kept)
        kept.extend(slide_kept)

    logger.info(f"  Deduplication: {len(samples)} -> {len(kept)} ({total_removed} duplicates removed)")
    return kept


def load_all_mk_features(input_dir, min_area_um=200, max_area_um=2000, min_clf_score=0.80):
    """Load all MK features from segmentation output directory."""
    input_dir = Path(input_dir)
    samples = []

    for slide_dir in sorted(input_dir.iterdir()):
        if not slide_dir.is_dir():
            continue
        tiles_dir = slide_dir / 'mk' / 'tiles'
        if not tiles_dir.exists():
            continue

        slide_count = 0
        for tile_dir in sorted(tiles_dir.iterdir()):
            if not tile_dir.is_dir():
                continue
            feat_file = tile_dir / 'features.json'
            if not feat_file.exists():
                continue

            with open(feat_file) as f:
                feats = json.load(f)

            for feat in feats:
                if 'crop_b64' not in feat or 'features' not in feat:
                    continue

                area_px = feat.get('area', feat['features'].get('area', 0))
                area_um2 = feat.get('area_um2', area_px * PIXEL_SIZE_UM ** 2)
                mk_score = feat.get('mk_score')

                # Size filter
                if area_um2 < min_area_um or area_um2 > max_area_um:
                    continue

                # Classifier score filter
                if min_clf_score is not None and (mk_score is None or mk_score < min_clf_score):
                    continue

                samples.append({
                    'slide': slide_dir.name,
                    'group': slide_to_group(slide_dir.name),
                    'tile_id': tile_dir.name,
                    'det_id': feat.get('id', ''),
                    'area_px': area_px,
                    'area_um2': area_um2,
                    'mk_score': mk_score,
                    'global_x': round(feat['center'][0]),
                    'global_y': round(feat['center'][1]),
                    'crop_b64': feat['crop_b64'],
                    'mask_b64': feat.get('mask_b64'),
                })
                slide_count += 1

        if slide_count > 0:
            logger.info(f"  {slide_dir.name}: {slide_count} MKs loaded")

    return samples


def run_phase1(args):
    """Phase 1: Load features, filter, dedup, save crops + metadata (no PCA)."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading and preprocessing MK features")
    logger.info("=" * 60)

    t0 = time.time()

    # Load
    logger.info(f"Loading from {args.input_dir}...")
    samples = load_all_mk_features(
        args.input_dir,
        min_area_um=args.min_area_um,
        max_area_um=args.max_area_um,
        min_clf_score=args.min_clf_score,
    )
    logger.info(f"  Loaded {len(samples)} MKs (after size + clf filter)")

    # Dedup
    samples = deduplicate_samples(samples, distance_threshold_px=50)
    logger.info(f"  After dedup: {len(samples)} MKs")

    # Per-group counts
    group_counts = defaultdict(int)
    for s in samples:
        group_counts[s['group']] += 1
    for g in GROUP_ORDER:
        logger.info(f"    {g}: {group_counts.get(g, 0)}")

    N = len(samples)

    # Metadata arrays
    slides = np.array([s['slide'] for s in samples])
    groups = np.array([s['group'] for s in samples])
    area_um2 = np.array([s['area_um2'] for s in samples], dtype=np.float32)
    mk_scores = np.array([s['mk_score'] for s in samples], dtype=np.float32)
    global_x = np.array([s['global_x'] for s in samples], dtype=np.float32)
    global_y = np.array([s['global_y'] for s in samples], dtype=np.float32)

    # Save crop_b64 and mask_b64 separately (large — needed for phase 2 GPU processing)
    crop_b64_list = [s['crop_b64'] for s in samples]
    mask_b64_list = [s.get('mask_b64', '') for s in samples]

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata as npz (no PCA — that happens in Phase 2 on nuclear features)
    np.savez_compressed(
        str(output),
        slides=slides,
        groups=groups,
        area_um2=area_um2,
        mk_scores=mk_scores,
        global_x=global_x,
        global_y=global_y,
    )

    # Save crop/mask b64 as separate JSON (too large for npz string arrays)
    crops_file = output.parent / (output.stem + '_crops.json')
    with open(crops_file, 'w') as f:
        json.dump({'crop_b64': crop_b64_list, 'mask_b64': mask_b64_list}, f)

    dt = time.time() - t0
    logger.info(f"Phase 1 complete in {dt:.1f}s. Saved to {output} + {crops_file}")
    logger.info(f"  {N} MKs across {len(set(slides))} slides")


# =============================================================================
# Phase 2: Nuclear deep feature extraction (GPU)
# =============================================================================

# Nuclear morphological feature names (extracted from nucleus mask)
NUC_MORPH_NAMES = ['nc_ratio', 'circularity', 'solidity', 'lobe_count',
                   'intensity_mean', 'area_fraction']

_ROCM_PATCH_APPLIED = False

def _apply_rocm_patch_if_needed():
    """Apply ROCm INT_MAX fix lazily — call before using SAM2."""
    global _ROCM_PATCH_APPLIED
    if _ROCM_PATCH_APPLIED:
        return
    _ROCM_PATCH_APPLIED = True

    try:
        import torch
        from typing import List, Dict, Any
        import sam2.utils.amg as amg

        def mask_to_rle_pytorch_rocm_safe(tensor: torch.Tensor) -> List[Dict[str, Any]]:
            b, h, w = tensor.shape
            tensor = tensor.permute(0, 2, 1).flatten(1)
            diff = tensor[:, 1:] ^ tensor[:, :-1]
            diff_cpu = diff.cpu()
            change_indices = diff_cpu.nonzero()
            out = []
            for i in range(b):
                cur_idxs = change_indices[change_indices[:, 0] == i, 1]
                cur_idxs = torch.cat([
                    torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                    cur_idxs + 1,
                    torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
                ])
                btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
                counts = [] if tensor[i, 0] == 0 else [0]
                counts.extend(btw_idxs.detach().cpu().tolist())
                out.append({"size": [h, w], "counts": counts})
            return out

        amg.mask_to_rle_pytorch = mask_to_rle_pytorch_rocm_safe
        logger.info("[ROCm FIX] Patched sam2.utils.amg.mask_to_rle_pytorch")
    except ImportError as e:
        logger.info(f"[ROCm FIX] Could not apply patch: {e}")


def _gpu_worker_nuclear_features(gpu_id, start_idx, end_idx, crops_file_path,
                                  output_path, sam2_checkpoint, sam2_config):
    """GPU worker: extract nuclear deep features from a chunk of MK crops.

    Each worker:
    1. Sets CUDA_VISIBLE_DEVICES to isolate GPU
    2. Loads SAM2 + ResNet models
    3. For each crop: Otsu nuclear seg → SAM2 + ResNet features on nucleus
    4. Saves partial results to output_path
    """
    # MUST set CUDA_VISIBLE_DEVICES before importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    from PIL import Image
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import regionprops, label as sk_label
    from skimage.morphology import remove_small_objects

    # Configure logging for subprocess
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sub_logger = logging.getLogger(f'gpu_worker_{gpu_id}')

    device = torch.device('cuda:0')
    sub_logger.info(f"GPU {gpu_id}: processing indices {start_idx}-{end_idx} "
                    f"({end_idx - start_idx} crops)")

    # Apply ROCm patch before SAM2
    _apply_rocm_patch_if_needed()

    # Load models (SAM2 + ResNet only — nuclear seg is Otsu-based)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2

    sub_logger.info(f"GPU {gpu_id}: Loading SAM2 from {sam2_checkpoint}...")
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    sub_logger.info(f"GPU {gpu_id}: Loading ResNet-50...")
    resnet = tv_models.resnet50(weights='DEFAULT')
    resnet_model = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet_model.eval().to(device)
    resnet_transform = tv_transforms.Compose([
        tv_transforms.Resize(224),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Morphological kernels for nuclear segmentation cleanup
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # 10px radius
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))    # bridge chromatin gaps
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))     # snap thin tails

    sub_logger.info(f"GPU {gpu_id}: Models loaded. Processing crops...")

    # Load crops (read full file, slice to our chunk)
    with open(crops_file_path) as f:
        crops_data = json.load(f)
    crop_b64_chunk = crops_data['crop_b64'][start_idx:end_idx]
    mask_b64_chunk = crops_data['mask_b64'][start_idx:end_idx]
    del crops_data

    chunk_size = end_idx - start_idx
    resnet_features = np.zeros((chunk_size, 2048), dtype=np.float32)
    sam2_features = np.zeros((chunk_size, 256), dtype=np.float32)
    morph_features = np.full((chunk_size, 6), np.nan, dtype=np.float32)
    valid = np.zeros(chunk_size, dtype=bool)

    for local_i in range(chunk_size):
        try:
            # Decode crop image
            crop_bytes = base64.b64decode(crop_b64_chunk[local_i])
            crop_img = np.array(Image.open(BytesIO(crop_bytes)).convert('RGB'))

            # Decode cell mask
            if mask_b64_chunk[local_i]:
                mask_bytes = base64.b64decode(mask_b64_chunk[local_i])
                mask_img = np.array(Image.open(BytesIO(mask_bytes)).convert('L'))
                cell_mask = mask_img > 127
            else:
                cell_mask = np.ones(crop_img.shape[:2], dtype=bool)

            if cell_mask.sum() < 10:
                continue

            # --- Nuclear segmentation via inv. blue + Otsu×1.05 ---
            gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            inv_blue = 255 - crop_img[:, :, 2]  # brown proxy: high where blue is low

            # Erode cell mask to exclude peripheral dark cells
            eroded_mask = cv2.erode(cell_mask.astype(np.uint8), erode_kernel).astype(bool)
            if eroded_mask.sum() < 10:
                eroded_mask = cell_mask  # fallback for tiny cells

            # Otsu×1.05 on inverted blue channel within eroded mask
            cell_pixels = inv_blue[eroded_mask]
            otsu_thresh, _ = cv2.threshold(cell_pixels, 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            relaxed_thresh = min(otsu_thresh * 1.05, 255)
            nucleus_mask = (inv_blue > relaxed_thresh) & eroded_mask

            # Cleanup: close gaps, fill holes, remove small fragments
            nucleus_mask = cv2.morphologyEx(
                nucleus_mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
            ).astype(bool)
            nucleus_mask = binary_fill_holes(nucleus_mask)
            nucleus_mask = remove_small_objects(nucleus_mask, min_size=50)

            # Morphological opening to snap thin tails
            nucleus_mask = cv2.morphologyEx(
                nucleus_mask.astype(np.uint8), cv2.MORPH_OPEN, open_kernel
            ).astype(bool)

            # Keep all substantial components (>= 25% of largest)
            labeled_mask = sk_label(nucleus_mask)
            if labeled_mask.max() == 0:
                continue
            regions_mask = regionprops(labeled_mask)
            largest_area = max(r.area for r in regions_mask)
            min_component = largest_area * 0.25
            nucleus_mask = np.zeros_like(nucleus_mask)
            for r in regions_mask:
                if r.area >= min_component:
                    nucleus_mask |= (labeled_mask == r.label)

            nuc_area = nucleus_mask.sum()
            if nuc_area < 10:
                continue

            # --- Morphological nuclear features (6D) ---
            # (compute before deep features — needed for SAM2 centroid)
            cell_area = cell_mask.sum()
            nc_ratio = float(nuc_area) / float(cell_area)

            labeled_nuc = sk_label(nucleus_mask)
            regions = regionprops(labeled_nuc)
            lobe_count = len(regions)

            largest = max(regions, key=lambda r: r.area)
            perim = largest.perimeter
            circularity = 4 * np.pi * largest.area / (perim ** 2) if perim > 0 else 0.0
            solidity = largest.solidity

            intensity_mean = float(gray[nucleus_mask].mean())
            area_fraction = nc_ratio

            # Create nucleus-only image (everything outside nucleus = white)
            nucleus_img = crop_img.copy()
            nucleus_img[~nucleus_mask] = 255

            # Tight crop around nucleus bounding box (with padding)
            ys, xs = np.where(nucleus_mask)
            pad = 5
            y0 = max(0, ys.min() - pad)
            y1 = min(crop_img.shape[0], ys.max() + pad + 1)
            x0 = max(0, xs.min() - pad)
            x1 = min(crop_img.shape[1], xs.max() + pad + 1)
            nucleus_crop = nucleus_img[y0:y1, x0:x1]

            if nucleus_crop.shape[0] < 10 or nucleus_crop.shape[1] < 10:
                continue

            # --- ResNet features (2048D) ---
            pil_img = Image.fromarray(nucleus_crop)
            tensor = resnet_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                resnet_feat = resnet_model(tensor).cpu().numpy().flatten()
            resnet_features[local_i] = resnet_feat

            # --- SAM2 features (256D) ---
            sam2_predictor.set_image(nucleus_crop)
            # Use largest component centroid (not mean of all components,
            # which could land in the gap between lobes)
            largest_nuc = labeled_nuc == largest.label
            nuc_ys, nuc_xs = np.where(largest_nuc)
            cy_crop = nuc_ys.mean() - y0
            cx_crop = nuc_xs.mean() - x0
            sam2_feat = np.zeros(256, dtype=np.float32)
            if hasattr(sam2_predictor, '_features'):
                feat_map = sam2_predictor._features
                if isinstance(feat_map, dict) and "image_embed" in feat_map:
                    feat_map = feat_map["image_embed"]
                emb_h, emb_w = feat_map.shape[2], feat_map.shape[3]
                if hasattr(sam2_predictor, '_orig_hw'):
                    orig_hw = sam2_predictor._orig_hw
                    # SAM2 stores as [(H, W)] list — unwrap if needed
                    if isinstance(orig_hw, (list, tuple)) and len(orig_hw) == 1:
                        img_h, img_w = orig_hw[0]
                    else:
                        img_h, img_w = orig_hw
                else:
                    img_h, img_w = nucleus_crop.shape[:2]
                ey = min(max(0, int(cy_crop / img_h * emb_h)), emb_h - 1)
                ex = min(max(0, int(cx_crop / img_w * emb_w)), emb_w - 1)
                sam2_feat = feat_map[0, :, ey, ex].cpu().numpy()
            sam2_features[local_i] = sam2_feat

            morph_features[local_i] = [nc_ratio, circularity, solidity,
                                        lobe_count, intensity_mean, area_fraction]
            valid[local_i] = True

        except Exception as e:
            sub_logger.warning(f"GPU {gpu_id}: MK {start_idx + local_i} error: {e}")

        if (local_i + 1) % 200 == 0:
            n_valid = valid[:local_i + 1].sum()
            sub_logger.info(f"GPU {gpu_id}: {local_i + 1}/{chunk_size} "
                            f"({(local_i+1)/chunk_size*100:.0f}%, {n_valid} valid)")
            # Periodic GPU memory cleanup (SAM2 set_image accumulates tensors)
            torch.cuda.empty_cache()

    n_valid = valid.sum()
    sub_logger.info(f"GPU {gpu_id}: Done. {n_valid}/{chunk_size} valid "
                    f"({n_valid/chunk_size*100:.1f}%)")

    # Save partial results
    np.savez_compressed(
        output_path,
        resnet=resnet_features,
        sam2=sam2_features,
        morph=morph_features,
        valid=valid,
        chunk_start=np.array([start_idx]),
        chunk_end=np.array([end_idx]),
    )
    sub_logger.info(f"GPU {gpu_id}: Saved partial results to {output_path}")


def run_phase2(args):
    """Phase 2: Nuclear deep feature extraction on GPU + PCA."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Nuclear Deep Feature Extraction (GPU)")
    logger.info("=" * 60)

    t0 = time.time()

    # Load phase 1 metadata
    data = np.load(args.input, allow_pickle=True)
    N = len(data['groups'])
    logger.info(f"  {N} MKs to process")

    # Crops file path
    crops_file = str(Path(args.input).parent / (Path(args.input).stem + '_crops.json'))
    if not Path(crops_file).exists():
        raise FileNotFoundError(f"Crops file not found: {crops_file}")

    # Find SAM2 checkpoint
    sam2_checkpoint = args.sam2_checkpoint
    sam2_config = args.sam2_config
    if not Path(sam2_checkpoint).exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")

    num_gpus = args.num_gpus
    logger.info(f"  Using {num_gpus} GPUs, SAM2: {sam2_checkpoint}")

    # Spawn GPU workers
    import torch.multiprocessing as torch_mp
    ctx = torch_mp.get_context('spawn')

    chunk_size = (N + num_gpus - 1) // num_gpus
    partial_files = []
    processes = []

    for gpu_id in range(num_gpus):
        start = gpu_id * chunk_size
        end = min(start + chunk_size, N)
        if start >= N:
            break

        partial_path = str(Path(args.output).parent / f'_partial_gpu{gpu_id}.npz')
        partial_files.append(partial_path)

        p = ctx.Process(
            target=_gpu_worker_nuclear_features,
            args=(gpu_id, start, end, crops_file, partial_path,
                  sam2_checkpoint, sam2_config),
        )
        p.start()
        processes.append(p)
        logger.info(f"  Spawned GPU worker {gpu_id}: indices {start}-{end}")

    # Wait for all workers
    failed_workers = []
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            logger.error(f"  Worker {i} (PID {p.pid}) exited with code {p.exitcode}")
            failed_workers.append(i)

    if failed_workers:
        logger.warning(f"  {len(failed_workers)} worker(s) failed: {failed_workers}")

    logger.info("  Merging results from successful workers...")

    # Merge partial results
    resnet_all = np.zeros((N, 2048), dtype=np.float32)
    sam2_all = np.zeros((N, 256), dtype=np.float32)
    morph_all = np.full((N, 6), np.nan, dtype=np.float32)
    valid_all = np.zeros(N, dtype=bool)

    for pf in partial_files:
        if not Path(pf).exists():
            logger.warning(f"  Partial file missing (worker crashed?): {pf}")
            continue
        partial = np.load(pf)
        s = int(partial['chunk_start'][0])
        e = int(partial['chunk_end'][0])
        resnet_all[s:e] = partial['resnet']
        sam2_all[s:e] = partial['sam2']
        morph_all[s:e] = partial['morph']
        valid_all[s:e] = partial['valid']
        os.remove(pf)

    n_valid = valid_all.sum()
    logger.info(f"  Valid nuclear segmentations: {n_valid}/{N} ({n_valid/N*100:.1f}%)")

    # PCA on valid nuclear deep features (2048 ResNet + 256 SAM2 = 2304D)
    valid_idx = np.where(valid_all)[0]
    if len(valid_idx) < 2:
        raise RuntimeError(f"Only {len(valid_idx)} valid nuclear segmentations out of {N}. "
                           "Check nuclear segmentation and crop quality. Cannot proceed with PCA.")

    X_deep_valid = np.concatenate([resnet_all[valid_idx], sam2_all[valid_idx]], axis=1)

    logger.info(f"  Z-score normalizing {X_deep_valid.shape[0]} x {X_deep_valid.shape[1]} features...")
    scaler = StandardScaler()
    X_deep_scaled = scaler.fit_transform(X_deep_valid)

    n_components = min(500, len(valid_idx) - 1, 2304)
    logger.info(f"  Running PCA ({n_components} components)...")
    pca = PCA(n_components=n_components)
    X_pca_valid = pca.fit_transform(X_deep_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    total_var = cumvar[-1]
    n95_raw = int(np.searchsorted(cumvar, 0.95)) + 1
    if n95_raw <= pca.n_components_:
        n95 = n95_raw
        logger.info(f"  PCA: {pca.n_components_} components, 95% variance at {n95} PCs "
                    f"({total_var*100:.1f}% total captured)")
    else:
        n95 = pca.n_components_  # use all available
        logger.info(f"  PCA: {pca.n_components_} components capture {total_var*100:.1f}% variance "
                    f"(95% not reached — would need ~{n95_raw} PCs)")
    logger.info(f"  Top-10 cumulative variance: {cumvar[:10].round(3)}")

    # Morph feature stats
    logger.info("  Nuclear morphological feature stats (valid MKs):")
    for j, fname in enumerate(NUC_MORPH_NAMES):
        vals = morph_all[valid_idx, j]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            logger.info(f"    {fname}: median={np.median(vals):.3f}, "
                        f"mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(output),
        resnet_features=resnet_all,
        sam2_features=sam2_all,
        morph_features=morph_all,
        morph_feature_names=np.array(NUC_MORPH_NAMES),
        valid_mask=valid_all,
        valid_indices=valid_idx,
        X_pca_valid=X_pca_valid,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        n95_pcs=np.array([n95]),
    )

    dt = time.time() - t0
    logger.info(f"Phase 2 complete in {dt:.1f}s. Saved to {output}")
    logger.info(f"  {n_valid} valid MKs, using {n95} PCA components ({total_var*100:.1f}% variance)")


# =============================================================================
# Phase 3: Clustering on nuclear deep features
# =============================================================================

def run_phase3(args):
    """Phase 3: Unsupervised clustering on PCA-reduced nuclear deep features."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Clustering on nuclear deep features")
    logger.info("=" * 60)

    t0 = time.time()

    # Load phase 1 metadata + phase 2 nuclear features
    data = np.load(args.data, allow_pickle=True)
    nuclear = np.load(args.input, allow_pickle=True)

    area_um2 = data['area_um2']
    groups = data['groups']

    X_pca_valid = nuclear['X_pca_valid']
    valid_idx = nuclear['valid_indices']
    n95 = int(nuclear['n95_pcs'][0])
    N_valid = len(valid_idx)

    # Use 95% variance PCs
    X_pca95 = X_pca_valid[:, :n95]
    logger.info(f"  Using {n95} PCA components ({N_valid} valid MKs)")

    # Area for valid MKs only (for cluster ordering)
    area_valid = area_um2[valid_idx]

    # Parse k range
    k_values = [int(k) for k in args.k_range.split(',')]
    logger.info(f"  K values: {k_values}")

    # ---- KMeans sweep ----
    logger.info("\n--- KMeans clustering ---")
    kmeans_results = {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(X_pca95)
        sil = silhouette_score(X_pca95, labels) if k > 1 else 0
        ch = calinski_harabasz_score(X_pca95, labels) if k > 1 else 0
        inertia = km.inertia_
        cluster_areas = [float(area_valid[labels == c].mean()) for c in range(k)]
        kmeans_results[k] = {
            'labels': labels, 'silhouette': sil,
            'calinski_harabasz': ch, 'inertia': inertia,
            'cluster_areas': cluster_areas,
        }
        logger.info(f"  k={k}: silhouette={sil:.4f}, CH={ch:.1f}, inertia={inertia:.0f}")
        logger.info(f"         cluster mean areas: {[f'{a:.0f}' for a in cluster_areas]}")

    # ---- Spectral clustering ----
    logger.info("\n--- Spectral clustering ---")
    spectral_results = {}
    for knn_k in [15, 30]:
        logger.info(f"  Building kNN graph (k={knn_k})...")
        connectivity = kneighbors_graph(X_pca95, n_neighbors=knn_k,
                                         mode='connectivity', include_self=False)
        affinity = (connectivity + connectivity.T) / 2
        affinity_dense = affinity.toarray()
        for k in k_values:
            try:
                sc = SpectralClustering(
                    n_clusters=k, affinity='precomputed', random_state=42, n_init=10,
                )
                labels = sc.fit_predict(affinity_dense)
                sil = silhouette_score(X_pca95, labels) if k > 1 else 0
                ch = calinski_harabasz_score(X_pca95, labels) if k > 1 else 0
                spectral_results[(knn_k, k)] = {
                    'labels': labels, 'silhouette': sil, 'calinski_harabasz': ch,
                }
                logger.info(f"  kNN={knn_k}, k={k}: silhouette={sil:.4f}, CH={ch:.1f}")
            except Exception as e:
                logger.warning(f"  kNN={knn_k}, k={k}: FAILED - {e}")

    # ---- Agglomerative clustering ----
    logger.info("\n--- Agglomerative clustering (Ward) ---")
    agglo_results = {}
    for k in k_values:
        ac = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = ac.fit_predict(X_pca95)
        sil = silhouette_score(X_pca95, labels) if k > 1 else 0
        ch = calinski_harabasz_score(X_pca95, labels) if k > 1 else 0
        agglo_results[k] = {
            'labels': labels, 'silhouette': sil, 'calinski_harabasz': ch,
        }
        logger.info(f"  k={k}: silhouette={sil:.4f}, CH={ch:.1f}")

    # ---- Select best ----
    logger.info("\n--- Summary: best silhouette by k ---")
    best_overall = {'sil': -1}
    for k in k_values:
        candidates = []
        if k in kmeans_results:
            candidates.append(('KMeans', kmeans_results[k]['silhouette'],
                               kmeans_results[k]['labels']))
        for knn_k in [15, 30]:
            key = (knn_k, k)
            if key in spectral_results:
                candidates.append((f'Spectral_kNN{knn_k}',
                                   spectral_results[key]['silhouette'],
                                   spectral_results[key]['labels']))
        if k in agglo_results:
            candidates.append(('Agglomerative', agglo_results[k]['silhouette'],
                               agglo_results[k]['labels']))
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates:
            best_name, best_sil, best_labels = candidates[0]
            logger.info(f"  k={k}: best={best_name} (sil={best_sil:.4f})")
            if best_sil > best_overall['sil']:
                best_overall = {'k': k, 'method': best_name, 'sil': best_sil,
                                'labels': best_labels}

    logger.info(f"\n  BEST: k={best_overall['k']}, method={best_overall['method']}, "
                f"silhouette={best_overall['sil']:.4f}")

    best_labels = best_overall['labels']
    best_k = best_overall['k']

    # Order clusters by mean area
    cluster_mean_area = np.array([float(area_valid[best_labels == c].mean())
                                  for c in range(best_k)])
    cluster_order = np.argsort(cluster_mean_area)
    remap = np.zeros(best_k, dtype=int)
    for new_label, old_label in enumerate(cluster_order):
        remap[old_label] = new_label
    best_labels_ordered = remap[best_labels]

    # ---- t-SNE embedding ----
    logger.info("\nRunning t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_pca95)
    logger.info(f"  t-SNE done. KL divergence: {tsne.kl_divergence_:.4f}")

    # ---- Spectral embedding for pseudotime ----
    logger.info("Computing spectral embedding (kNN=15) for pseudotime...")
    knn_graph = kneighbors_graph(X_pca95, n_neighbors=15,
                                  mode='connectivity', include_self=False)
    knn_graph = (knn_graph + knn_graph.T) / 2
    spec_embed = spectral_embedding(
        knn_graph, n_components=5, random_state=42, drop_first=True,
    )
    logger.info(f"  Spectral embedding shape: {spec_embed.shape}")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    km_silhouettes = np.array([(k, kmeans_results[k]['silhouette']) for k in k_values])

    np.savez_compressed(
        str(output),
        best_labels=best_labels_ordered,
        best_k=np.array([best_k]),
        best_method=np.array([best_overall['method']]),
        best_silhouette=np.array([best_overall['sil']]),
        cluster_order=cluster_order,
        X_tsne=X_tsne,
        spec_embed=spec_embed,
        km_silhouettes=km_silhouettes,
        valid_indices=valid_idx,
        **{f'kmeans_labels_k{k}': kmeans_results[k]['labels'] for k in k_values},
        **{f'kmeans_sil_k{k}': np.array([kmeans_results[k]['silhouette']]) for k in k_values},
    )

    dt = time.time() - t0
    logger.info(f"\nPhase 3 complete in {dt:.1f}s. Saved to {output}")
    logger.info(f"  Best: k={best_k}, method={best_overall['method']}, "
                f"silhouette={best_overall['sil']:.4f}")


# =============================================================================
# Phase 4: Validation, pseudotime, group comparison
# =============================================================================

def run_phase4(args):
    """Phase 4: Validate clusters, pseudotime inference, group comparison."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Validation, Pseudotime & Group Comparison")
    logger.info("=" * 60)

    t0 = time.time()

    # Load all data
    data = np.load(args.data, allow_pickle=True)
    nuclear = np.load(args.nuclear, allow_pickle=True)
    clusters = np.load(args.clusters, allow_pickle=True)

    # Phase 1 metadata (all MKs)
    groups_all = data['groups']
    slides = data['slides']
    area_um2_all = data['area_um2']
    mk_scores_all = data['mk_scores']
    N_all = len(groups_all)

    # Phase 2 nuclear features (valid_indices maps into the full array)
    valid_idx = nuclear['valid_indices']
    nuc_features = nuclear['morph_features'][valid_idx]  # only valid MKs
    nuc_feature_names = list(nuclear['morph_feature_names'])
    X_pca = nuclear['X_pca_valid']
    n95 = int(nuclear['n95_pcs'][0])

    # Phase 3 clusters (indexed on valid MKs only)
    best_labels = clusters['best_labels']
    best_k = int(clusters['best_k'][0])
    best_method = str(clusters['best_method'][0])
    X_tsne = clusters['X_tsne']
    spec_embed = clusters['spec_embed']
    km_silhouettes = clusters['km_silhouettes']

    # Subset metadata to valid MKs (matching cluster/t-SNE indices)
    groups = groups_all[valid_idx]
    area_um2 = area_um2_all[valid_idx]
    mk_scores = mk_scores_all[valid_idx]
    N = len(valid_idx)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  {N} valid MKs (of {N_all} total), {best_k} clusters ({best_method}), "
                f"{nuc_features.shape[1]} nuclear morph features")

    # ---- 4a: Validate clusters with nuclear features ----
    logger.info("\n--- 4a: Cluster validation with nuclear features ---")

    # Per-cluster nuclear feature stats
    cluster_stats = {}
    for c in range(best_k):
        mask = best_labels == c
        n_in_cluster = mask.sum()
        cstats = {'n': int(n_in_cluster), 'mean_area_um2': float(area_um2[mask].mean())}
        for j, fname in enumerate(nuc_feature_names):
            vals = nuc_features[mask, j]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                cstats[f'{fname}_mean'] = float(np.mean(valid))
                cstats[f'{fname}_std'] = float(np.std(valid))
                cstats[f'{fname}_median'] = float(np.median(valid))
            else:
                cstats[f'{fname}_mean'] = np.nan
                cstats[f'{fname}_std'] = np.nan
                cstats[f'{fname}_median'] = np.nan
        cluster_stats[c] = cstats
        logger.info(f"  Cluster {c}: n={n_in_cluster}, mean_area={cstats['mean_area_um2']:.0f} um2")

    # Monotonicity tests (Spearman correlation of cluster index vs feature median)
    logger.info("\n  Spearman correlations (cluster_index vs nuclear feature):")
    monotonicity = {}
    for fname in nuc_feature_names:
        cluster_medians = [cluster_stats[c].get(f'{fname}_median', np.nan) for c in range(best_k)]
        if any(np.isnan(v) for v in cluster_medians):
            logger.info(f"    {fname}: NaN in some clusters, skipping")
            continue
        rho, pval = stats.spearmanr(range(best_k), cluster_medians)
        monotonicity[fname] = {'rho': float(rho), 'pval': float(pval)}
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        logger.info(f"    {fname}: rho={rho:.3f}, p={pval:.4f} {sig}")

    # Save cluster stats as JSON
    with open(outdir / 'cluster_stats.json', 'w') as f:
        json.dump({
            'best_k': best_k,
            'best_method': best_method,
            'cluster_stats': {str(k): v for k, v in cluster_stats.items()},
            'monotonicity': monotonicity,
        }, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else float(x) if isinstance(x, (np.floating, np.integer)) else x)

    # ---- Heatmap: cluster x nuclear feature means ----
    logger.info("\n  Generating cluster-nuclear feature heatmap...")
    _plot_cluster_nuclear_heatmap(cluster_stats, nuc_feature_names, best_k, outdir)

    # ---- 4b: Pseudotime inference ----
    logger.info("\n--- 4b: Pseudotime inference ---")

    # Spectral embedding component 1 as pseudotime
    pseudotime_spec = spec_embed[:, 0]

    # Orient pseudotime: correlate with nuclear circularity
    # Circularity should DECREASE with maturation, so pseudotime should correlate negatively
    circ_idx = nuc_feature_names.index('circularity')
    valid_mask = ~np.isnan(nuc_features[:, circ_idx])
    if valid_mask.sum() > 10:
        rho, _ = stats.spearmanr(pseudotime_spec[valid_mask], nuc_features[valid_mask, circ_idx])
        if rho > 0:
            # Flip: we want circularity to decrease along pseudotime
            pseudotime_spec = -pseudotime_spec
            logger.info(f"  Flipped pseudotime (circularity was positively correlated, rho={rho:.3f})")
        else:
            logger.info(f"  Pseudotime orientation OK (circularity rho={rho:.3f})")

    # PCA component 1 as alternative pseudotime
    pseudotime_pca = X_pca[:, 0]
    rho_pca, _ = stats.spearmanr(pseudotime_pca[valid_mask], nuc_features[valid_mask, circ_idx])
    if rho_pca > 0:
        pseudotime_pca = -pseudotime_pca

    # Validate pseudotime
    logger.info("\n  Pseudotime validation (spectral embedding):")
    for fname in nuc_feature_names:
        fidx = nuc_feature_names.index(fname)
        vm = ~np.isnan(nuc_features[:, fidx])
        if vm.sum() > 10:
            rho, pval = stats.spearmanr(pseudotime_spec[vm], nuc_features[vm, fidx])
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            logger.info(f"    {fname}: rho={rho:.3f}, p={pval:.2e} {sig}")

    logger.info("\n  Pseudotime validation (PCA component 1):")
    for fname in nuc_feature_names:
        fidx = nuc_feature_names.index(fname)
        vm = ~np.isnan(nuc_features[:, fidx])
        if vm.sum() > 10:
            rho, pval = stats.spearmanr(pseudotime_pca[vm], nuc_features[vm, fidx])
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            logger.info(f"    {fname}: rho={rho:.3f}, p={pval:.2e} {sig}")

    # Agreement between spectral and PCA pseudotime
    rho_agree, _ = stats.spearmanr(pseudotime_spec, pseudotime_pca)
    logger.info(f"\n  Spectral vs PCA pseudotime agreement: rho={rho_agree:.3f}")

    # ---- 4c: Group comparison ----
    logger.info("\n--- 4c: Group comparison ---")

    # Cluster composition per group
    logger.info("\n  Cluster composition per group:")
    contingency = np.zeros((len(GROUP_ORDER), best_k), dtype=int)
    for i, g in enumerate(GROUP_ORDER):
        gmask = groups == g
        for c in range(best_k):
            contingency[i, c] = ((best_labels == c) & gmask).sum()
        row_sum = contingency[i].sum()
        if row_sum > 0:
            props = contingency[i] / row_sum * 100
            logger.info(f"    {g}: {dict(zip(range(best_k), contingency[i]))} "
                        f"({', '.join(f'{p:.1f}%' for p in props)})")
        else:
            logger.warning(f"    {g}: no valid MKs in this group")

    # Chi-squared test (filter out groups with 0 valid MKs)
    nonzero_rows = contingency.sum(axis=1) > 0
    if nonzero_rows.sum() >= 2:
        contingency_nz = contingency[nonzero_rows]
        chi2, chi2_pval, dof, expected = stats.chi2_contingency(contingency_nz)
        logger.info(f"\n  Chi-squared test: chi2={chi2:.2f}, p={chi2_pval:.2e}, dof={dof}")
    else:
        chi2, chi2_pval, dof = 0.0, 1.0, 0
        logger.warning("\n  Chi-squared test: fewer than 2 non-empty groups, skipping")

    # Pseudotime distributions per group
    logger.info("\n  Pseudotime distributions per group (spectral):")
    for g in GROUP_ORDER:
        gmask = groups == g
        vals = pseudotime_spec[gmask]
        logger.info(f"    {g}: n={len(vals)}, median={np.median(vals):.4f}, "
                    f"mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

    # Kruskal-Wallis (filter out empty groups)
    group_pseudotimes = [pseudotime_spec[groups == g] for g in GROUP_ORDER]
    nonempty_gpt = [v for v in group_pseudotimes if len(v) > 0]
    if len(nonempty_gpt) >= 2:
        kw_stat, kw_pval = stats.kruskal(*nonempty_gpt)
        logger.info(f"\n  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_pval:.2e}")
    else:
        kw_stat, kw_pval = 0.0, 1.0
        logger.warning("\n  Kruskal-Wallis: fewer than 2 non-empty groups, skipping")

    # Pairwise Mann-Whitney U with Bonferroni
    n_comparisons = len(GROUP_ORDER) * (len(GROUP_ORDER) - 1) // 2
    logger.info(f"\n  Pairwise Mann-Whitney U (Bonferroni correction, {n_comparisons} comparisons):")
    pairwise_results = {}
    for i in range(len(GROUP_ORDER)):
        for j in range(i + 1, len(GROUP_ORDER)):
            g1, g2 = GROUP_ORDER[i], GROUP_ORDER[j]
            v1 = pseudotime_spec[groups == g1]
            v2 = pseudotime_spec[groups == g2]
            if len(v1) == 0 or len(v2) == 0:
                logger.warning(f"    {g1} vs {g2}: skipped (empty group)")
                continue
            u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            p_adj = min(p_val * n_comparisons, 1.0)
            sig = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*' if p_adj < 0.05 else 'ns'
            pairwise_results[f'{g1}_vs_{g2}'] = {'U': float(u_stat), 'p_raw': float(p_val), 'p_adj': float(p_adj)}
            logger.info(f"    {g1} vs {g2}: U={u_stat:.0f}, p_adj={p_adj:.2e} {sig}")

    # Sex and condition comparisons
    logger.info("\n  Sex comparison (F vs M):")
    female_mask = (groups == 'FGC') | (groups == 'FHU')
    male_mask = (groups == 'MGC') | (groups == 'MHU')
    if female_mask.sum() > 0 and male_mask.sum() > 0:
        u_sex, p_sex = stats.mannwhitneyu(pseudotime_spec[female_mask], pseudotime_spec[male_mask])
        logger.info(f"    F vs M: U={u_sex:.0f}, p={p_sex:.2e}")
    else:
        u_sex, p_sex = 0.0, 1.0
        logger.warning("    F vs M: skipped (empty group)")

    logger.info("\n  Condition comparison (GC vs HU):")
    gc_mask = (groups == 'FGC') | (groups == 'MGC')
    hu_mask = (groups == 'FHU') | (groups == 'MHU')
    if gc_mask.sum() > 0 and hu_mask.sum() > 0:
        u_cond, p_cond = stats.mannwhitneyu(pseudotime_spec[gc_mask], pseudotime_spec[hu_mask])
        logger.info(f"    GC vs HU: U={u_cond:.0f}, p={p_cond:.2e}")
    else:
        u_cond, p_cond = 0.0, 1.0
        logger.warning("    GC vs HU: skipped (empty group)")

    # Save stats
    with open(outdir / 'group_comparison_stats.json', 'w') as f:
        json.dump({
            'chi_squared': {'chi2': float(chi2), 'p': float(chi2_pval), 'dof': int(dof)},
            'kruskal_wallis': {'H': float(kw_stat), 'p': float(kw_pval)},
            'pairwise_mannwhitney': pairwise_results,
            'sex_comparison': {'U': float(u_sex), 'p': float(p_sex)},
            'condition_comparison': {'U': float(u_cond), 'p': float(p_cond)},
            'contingency_table': contingency.tolist(),
        }, f, indent=2)

    # ---- Generate all plots ----
    logger.info("\n--- Generating plots ---")

    # 1. Silhouette score elbow plot
    _plot_silhouette_elbow(km_silhouettes, outdir)

    # 2. t-SNE colored by cluster
    _plot_tsne_colored(X_tsne, best_labels, best_k, 'Cluster', outdir / 'tsne_by_cluster',
                       cmap='tab10', discrete=True)

    # 3. t-SNE colored by group
    _plot_tsne_by_group(X_tsne, groups, outdir / 'tsne_by_group')

    # 4. t-SNE colored by area
    _plot_tsne_colored(X_tsne, area_um2, best_k, 'Area (um2)', outdir / 'tsne_by_area',
                       cmap='viridis', discrete=False)

    # 5. t-SNE colored by mk_score
    _plot_tsne_colored(X_tsne, mk_scores, best_k, 'CLF Score', outdir / 'tsne_by_score',
                       cmap='plasma', discrete=False)

    # 6. t-SNE colored by pseudotime
    _plot_tsne_colored(X_tsne, pseudotime_spec, best_k, 'Pseudotime (spectral)',
                       outdir / 'tsne_by_pseudotime', cmap='coolwarm', discrete=False)

    # 7. Cluster composition stacked bar
    _plot_cluster_composition(contingency, GROUP_ORDER, best_k, outdir)

    # 8. Pseudotime violin plots by group
    _plot_pseudotime_violins(pseudotime_spec, groups, pairwise_results, outdir)

    # 9. Pseudotime step histograms by group
    _plot_pseudotime_histograms(pseudotime_spec, groups, outdir)

    # 10. Representative gallery per cluster
    logger.info("  Generating representative gallery...")
    crops_file = Path(args.data).parent / (Path(args.data).stem + '_crops.json')
    if crops_file.exists():
        with open(crops_file) as f:
            crops_data = json.load(f)
        # Subset crops to valid MKs (matching cluster indices)
        valid_crops = [crops_data['crop_b64'][i] for i in valid_idx]
        _plot_representative_gallery(valid_crops, best_labels, best_k,
                                     area_um2, pseudotime_spec, outdir)
    else:
        logger.warning(f"  Crops file not found: {crops_file}, skipping gallery")

    # 11. t-SNE cluster + group side by side
    _plot_tsne_cluster_group_sidebyside(X_tsne, best_labels, best_k, groups, outdir)

    # Save pseudotime values for future use
    np.savez_compressed(
        str(outdir / 'pseudotime.npz'),
        pseudotime_spec=pseudotime_spec,
        pseudotime_pca=pseudotime_pca,
        pseudotime_agreement_rho=np.array([rho_agree]),
    )

    dt = time.time() - t0
    logger.info(f"\nPhase 4 complete in {dt:.1f}s. All outputs in {outdir}/")


# =============================================================================
# Plotting helpers
# =============================================================================

def _save_fig(fig, path_stem):
    """Save figure as PNG and PDF."""
    fig.savefig(f'{path_stem}.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{path_stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved {path_stem}.png/.pdf")


def _plot_silhouette_elbow(km_silhouettes, outdir):
    """Silhouette score vs k for KMeans."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = km_silhouettes[:, 0].astype(int)
    sils = km_silhouettes[:, 1]
    ax.plot(ks, sils, 'o-', color='#2c3e50', linewidth=2, markersize=8)
    ax.set_xlabel('Number of clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette score', fontsize=12)
    ax.set_title('KMeans Silhouette Score vs k', fontsize=14)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    best_idx = np.argmax(sils)
    ax.annotate(f'Best: k={ks[best_idx]}', xy=(ks[best_idx], sils[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#e74c3c')
    _save_fig(fig, str(outdir / 'silhouette_elbow'))


def _plot_tsne_colored(X_tsne, values, k, label, path_stem, cmap='tab10', discrete=True):
    """Generic t-SNE scatter colored by values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    if discrete:
        for c in range(k):
            mask = values == c
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.5,
                       label=f'Cluster {c} (n={mask.sum()})')
        ax.legend(markerscale=3, fontsize=10)
    else:
        sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=values, s=5, alpha=0.5, cmap=cmap)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f't-SNE colored by {label}', fontsize=14)
    ax.set_aspect('equal')
    _save_fig(fig, str(path_stem))


def _plot_tsne_by_group(X_tsne, groups, path_stem):
    """t-SNE colored by experimental group."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for g in GROUP_ORDER:
        mask = groups == g
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.4,
                   color=GROUP_COLORS[g], label=f'{g} (n={mask.sum()})')
    ax.legend(markerscale=3, fontsize=10)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE colored by Group', fontsize=14)
    ax.set_aspect('equal')
    _save_fig(fig, str(path_stem))


def _plot_cluster_nuclear_heatmap(cluster_stats, nuc_feature_names, best_k, outdir):
    """Heatmap of cluster x nuclear feature medians."""
    # Short names for display
    short_names = {
        'nc_ratio': 'N:C Ratio',
        'circularity': 'Circularity',
        'solidity': 'Solidity',
        'lobe_count': 'Lobe Count',
        'intensity_mean': 'Intensity',
        'area_fraction': 'Area Fraction',
    }

    data_matrix = np.zeros((best_k, len(nuc_feature_names)))
    for c in range(best_k):
        for j, fname in enumerate(nuc_feature_names):
            data_matrix[c, j] = cluster_stats[c].get(f'{fname}_median', np.nan)

    # Z-score normalize columns for visualization
    data_z = np.copy(data_matrix)
    for j in range(data_z.shape[1]):
        col = data_z[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1 and np.std(valid) > 0:
            data_z[:, j] = (col - np.mean(valid)) / np.std(valid)

    fig, ax = plt.subplots(figsize=(14, max(4, best_k * 0.8 + 2)))
    im = ax.imshow(data_z, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.8)

    # Labels
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f'Cluster {c}\n(n={cluster_stats[c]["n"]})' for c in range(best_k)])
    ax.set_xticks(range(len(nuc_feature_names)))
    ax.set_xticklabels([short_names.get(f, f) for f in nuc_feature_names], rotation=45, ha='right')

    # Annotate with raw values
    for c in range(best_k):
        for j in range(len(nuc_feature_names)):
            val = data_matrix[c, j]
            if not np.isnan(val):
                ax.text(j, c, f'{val:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(data_z[c, j]) > 1.5 else 'black')

    ax.set_title('Nuclear Features by Cluster (ordered by mean cell area)', fontsize=13)
    plt.tight_layout()
    _save_fig(fig, str(outdir / 'cluster_nuclear_heatmap'))


def _plot_cluster_composition(contingency, group_order, best_k, outdir):
    """Stacked bar chart of cluster proportions per group."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, best_k))

    # Absolute counts
    x = np.arange(len(group_order))
    bottom = np.zeros(len(group_order))
    for c in range(best_k):
        vals = contingency[:, c]
        ax1.bar(x, vals, bottom=bottom, color=colors[c], label=f'Cluster {c}', edgecolor='white', linewidth=0.5)
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_order, fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Cluster composition (absolute)', fontsize=13)
    ax1.legend(fontsize=9)

    # Proportions (guard against zero-sum rows)
    row_sums = contingency.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    props = contingency / row_sums
    bottom = np.zeros(len(group_order))
    for c in range(best_k):
        vals = props[:, c]
        ax2.bar(x, vals, bottom=bottom, color=colors[c], label=f'Cluster {c}', edgecolor='white', linewidth=0.5)
        # Label with percentage
        for i, v in enumerate(vals):
            if v > 0.05:
                ax2.text(i, bottom[i] + v / 2, f'{v*100:.0f}%', ha='center', va='center', fontsize=9)
        bottom += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_order, fontsize=12)
    ax2.set_ylabel('Proportion', fontsize=12)
    ax2.set_title('Cluster composition (proportional)', fontsize=13)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    _save_fig(fig, str(outdir / 'cluster_composition'))


def _plot_pseudotime_violins(pseudotime, groups, pairwise_results, outdir):
    """Violin plots of pseudotime by group with significance brackets."""
    fig, ax = plt.subplots(figsize=(10, 7))

    group_data = []
    plot_groups = []
    for g in GROUP_ORDER:
        vals = pseudotime[groups == g]
        if len(vals) > 0:
            group_data.append(vals)
            plot_groups.append(g)

    if len(group_data) < 2:
        logger.warning("  Fewer than 2 non-empty groups, skipping violin plot")
        plt.close(fig)
        return

    parts = ax.violinplot(group_data, positions=range(len(plot_groups)), showmedians=True,
                          showextrema=False)

    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(GROUP_COLORS[plot_groups[i]])
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')

    # Scale widths by count (modify vertices in-place)
    max_n = max(len(d) for d in group_data)
    for i, pc in enumerate(parts['bodies']):
        path = pc.get_paths()[0]
        center = i
        scale = len(group_data[i]) / max_n
        path.vertices[:, 0] = center + (path.vertices[:, 0] - center) * scale

    ax.set_xticks(range(len(plot_groups)))
    ax.set_xticklabels([f'{g}\n(n={len(d)})' for g, d in zip(plot_groups, group_data)], fontsize=11)
    ax.set_ylabel('Pseudotime (spectral embedding)', fontsize=12)
    ax.set_title('MK Maturation Pseudotime by Group', fontsize=14)

    # Add significance brackets
    y_max = max(v.max() for v in group_data)
    y_range = y_max - min(v.min() for v in group_data)
    bracket_y = y_max + 0.05 * y_range
    bracket_step = 0.06 * y_range

    # Map group names to violin positions for bracket placement
    group_pos = {g: i for i, g in enumerate(plot_groups)}
    sig_pairs = []
    for i in range(len(plot_groups)):
        for j in range(i + 1, len(plot_groups)):
            key = f'{plot_groups[i]}_vs_{plot_groups[j]}'
            if key in pairwise_results:
                p_adj = pairwise_results[key]['p_adj']
                if p_adj < 0.05:
                    sig = '***' if p_adj < 0.001 else '**' if p_adj < 0.01 else '*'
                    sig_pairs.append((i, j, sig, p_adj))

    for idx, (i, j, sig, p_adj) in enumerate(sig_pairs):
        y = bracket_y + idx * bracket_step
        ax.plot([i, i, j, j], [y - 0.01 * y_range, y, y, y - 0.01 * y_range],
                color='black', linewidth=1)
        ax.text((i + j) / 2, y + 0.005 * y_range, sig, ha='center', va='bottom', fontsize=11)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save_fig(fig, str(outdir / 'pseudotime_violin'))


def _plot_pseudotime_histograms(pseudotime, groups, outdir):
    """Step histograms of pseudotime by group (sex-coded colors)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    bins = np.linspace(pseudotime.min(), pseudotime.max(), 50)

    # Top: raw counts
    for g in GROUP_ORDER:
        vals = pseudotime[groups == g]
        ax1.hist(vals, bins=bins, histtype='step', linewidth=2,
                 color=GROUP_COLORS[g], label=f'{g} (n={len(vals)})')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('MK Pseudotime Distribution by Group', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom: density normalized
    for g in GROUP_ORDER:
        vals = pseudotime[groups == g]
        ax2.hist(vals, bins=bins, histtype='step', linewidth=2, density=True,
                 color=GROUP_COLORS[g], label=f'{g}')
    ax2.set_xlabel('Pseudotime (spectral embedding)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Normalized', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, str(outdir / 'pseudotime_histogram'))


def _plot_representative_gallery(crop_b64_list, labels, best_k, area_um2, pseudotime, outdir, n_per_cluster=8):
    """Grid of representative MK crops per cluster, ordered by pseudotime."""
    fig, axes = plt.subplots(best_k, n_per_cluster, figsize=(2.5 * n_per_cluster, 2.5 * best_k))
    if best_k == 1:
        axes = axes.reshape(1, -1)

    for c in range(best_k):
        cmask = labels == c
        c_indices = np.where(cmask)[0]
        # Sort by pseudotime within cluster
        c_pt = pseudotime[c_indices]
        sort_order = np.argsort(c_pt)
        c_indices_sorted = c_indices[sort_order]

        # Pick evenly spaced samples across the pseudotime range
        if len(c_indices_sorted) >= n_per_cluster:
            pick_idx = np.linspace(0, len(c_indices_sorted) - 1, n_per_cluster, dtype=int)
            chosen = c_indices_sorted[pick_idx]
        else:
            chosen = c_indices_sorted

        for j in range(n_per_cluster):
            ax = axes[c, j]
            if j < len(chosen):
                idx = chosen[j]
                try:
                    crop_bytes = base64.b64decode(crop_b64_list[idx])
                    crop_img = np.array(Image.open(BytesIO(crop_bytes)).convert('RGB'))
                    ax.imshow(crop_img)
                    ax.set_title(f'{area_um2[idx]:.0f}um2', fontsize=8)
                except Exception:
                    ax.text(0.5, 0.5, 'err', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        # Row label
        axes[c, 0].set_ylabel(f'C{c}\n(n={cmask.sum()})', fontsize=11, rotation=0,
                               labelpad=50, va='center')

    plt.suptitle('Representative MKs per Cluster (ordered by pseudotime, left=early, right=late)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, str(outdir / 'representative_gallery'))


def _plot_tsne_cluster_group_sidebyside(X_tsne, labels, best_k, groups, outdir):
    """Side-by-side t-SNE: left=clusters, right=groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Left: clusters
    for c in range(best_k):
        mask = labels == c
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.5,
                    label=f'Cluster {c} (n={mask.sum()})')
    ax1.legend(markerscale=3, fontsize=9)
    ax1.set_title('Clusters', fontsize=13)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_aspect('equal')

    # Right: groups
    for g in GROUP_ORDER:
        mask = groups == g
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.4,
                    color=GROUP_COLORS[g], label=f'{g} (n={mask.sum()})')
    ax2.legend(markerscale=3, fontsize=9)
    ax2.set_title('Groups', fontsize=13)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_aspect('equal')

    plt.suptitle('t-SNE: Clusters vs Groups', fontsize=14)
    plt.tight_layout()
    _save_fig(fig, str(outdir / 'tsne_cluster_vs_group'))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MK Maturation Staging Pipeline')
    subparsers = parser.add_subparsers(dest='phase', help='Pipeline phase to run')

    # Phase 1: Load, filter, dedup
    p1 = subparsers.add_parser('phase1_load', help='Load features, filter, dedup, save crops')
    p1.add_argument('--input-dir', required=True, help='Segmentation output directory')
    p1.add_argument('--min-area-um', type=float, default=200, help='Min area in um2')
    p1.add_argument('--max-area-um', type=float, default=2000, help='Max area in um2')
    p1.add_argument('--min-clf-score', type=float, default=0.80, help='Min classifier score')
    p1.add_argument('--output', required=True, help='Output .npz file')

    # Phase 2: Nuclear deep feature extraction (GPU)
    p2 = subparsers.add_parser('phase2_nuclear', help='Otsu nuclear seg + SAM2/ResNet deep features (GPU)')
    p2.add_argument('--input', required=True, help='Phase 1 output .npz')
    p2.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs')
    p2.add_argument('--sam2-checkpoint', default='/ptmp/edrod/MKsegmentation/checkpoints/sam2.1_hiera_large.pt',
                    help='SAM2 checkpoint path')
    p2.add_argument('--sam2-config', default='configs/sam2.1/sam2.1_hiera_l.yaml',
                    help='SAM2 config name')
    p2.add_argument('--output', required=True, help='Output .npz file')

    # Phase 3: Clustering on nuclear features
    p3 = subparsers.add_parser('phase3_cluster', help='Cluster on nuclear deep features')
    p3.add_argument('--input', required=True, help='Phase 2 nuclear features .npz')
    p3.add_argument('--data', required=True, help='Phase 1 metadata .npz')
    p3.add_argument('--k-range', default='3,4,5,6,7,8', help='Comma-separated k values')
    p3.add_argument('--output', required=True, help='Output .npz file')

    # Phase 4: Validation, pseudotime, group comparison
    p4 = subparsers.add_parser('phase4_validate', help='Validation, pseudotime, group comparison')
    p4.add_argument('--data', required=True, help='Phase 1 .npz')
    p4.add_argument('--nuclear', required=True, help='Phase 2 nuclear features .npz')
    p4.add_argument('--clusters', required=True, help='Phase 3 clusters .npz')
    p4.add_argument('--output-dir', required=True, help='Output directory for plots + stats')

    args = parser.parse_args()

    if args.phase == 'phase1_load':
        run_phase1(args)
    elif args.phase == 'phase2_nuclear':
        run_phase2(args)
    elif args.phase == 'phase3_cluster':
        run_phase3(args)
    elif args.phase == 'phase4_validate':
        run_phase4(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
