#!/usr/bin/env python3
"""ROI-based islet segmentation: nuclei-only vs PM+nuclei comparison.

Finds islet regions via tissue-level marker signal, then runs Cellpose+SAM2
only on those ROI crops — two modes in one run. Avoids processing the ~95%
non-islet area.

Multi-GPU: each GPU gets its own Cellpose + SAM2 models. ROIs are split
across GPUs (round-robin) and processed in parallel via threads. The
channel data is shared read-only — no pickling, no shared memory needed.

Pipeline:
  1. Load channels via strip-based CZI reader
  2. find_islet_regions() → downsampled label array
  3. Extract region bounding boxes via ndimage.find_objects()
  4. Init Cellpose + SAM2 per GPU (threaded)
  5. For each mode: split ROIs across GPUs, detect in parallel
  6. Population-level marker classification (p95)
  7. Save detections JSON + comparison stats

Usage:
  python scripts/segment_islet_regions.py \\
      --czi-path /path/to/BS-100.czi \\
      --marker-channels gcg:2,ins:3,sst:5 \\
      --membrane-channel 1 --nuclear-channel 4 \\
      --display-channels 2,3,5 \\
      --num-gpus 2 \\
      --output-dir /path/to/output
"""

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import hdf5plugin  # noqa: F401 — registers LZ4 codec for h5py
except ImportError:
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--czi-path', required=True, help='Path to 6-channel islet CZI')
    p.add_argument('--output-dir', required=True, help='Output directory')
    p.add_argument('--marker-channels', default='gcg:2,ins:3,sst:5',
                   help='name:ch_idx pairs (default: gcg:2,ins:3,sst:5)')
    p.add_argument('--membrane-channel', type=int, default=1,
                   help='Membrane channel for PM+nuclei mode (default: 1)')
    p.add_argument('--nuclear-channel', type=int, default=4,
                   help='Nuclear channel (default: 4, DAPI)')
    p.add_argument('--display-channels', default='2,3,5',
                   help='R,G,B channel indices for RGB display (default: 2,3,5)')
    p.add_argument('--num-gpus', type=int, default=2,
                   help='Number of GPUs (default: 2)')
    p.add_argument('--otsu-multiplier', type=float, default=2.0,
                   help='Otsu multiplier for islet finding (default: 2.0)')
    p.add_argument('--buffer-um', type=float, default=25.0,
                   help='Dilation buffer around islet regions in um (default: 25)')
    p.add_argument('--roi-padding-px', type=int, default=50,
                   help='Pixel padding around region bboxes (default: 50)')
    p.add_argument('--min-cells', type=int, default=5,
                   help='Min cells per islet for classification (default: 5)')
    p.add_argument('--marker-percentile', type=float, default=95,
                   help='Percentile threshold for marker classification (default: 95)')
    # --scene not yet plumbed through to load_czi_direct (hardcodes scene 0).
    # BS-100 is single-scene so this is unused. Add support if multi-scene needed.
    p.add_argument('--scene', type=int, default=0, help='CZI scene index (default: 0, not yet supported)')
    p.add_argument('--mode', choices=['both', 'nuclei', 'pm'], default='both',
                   help='Which modes to run (default: both)')
    return p.parse_args()


def parse_marker_channels(s):
    """Parse 'gcg:2,ins:3,sst:5' → OrderedDict {name: ch_idx}."""
    from collections import OrderedDict
    result = OrderedDict()
    for pair in s.split(','):
        name, ch = pair.strip().split(':')
        result[name.strip()] = int(ch.strip())
    return result


# ---------------------------------------------------------------------------
# ROI extraction
# ---------------------------------------------------------------------------

def extract_region_bboxes(region_labels, downsample, x_start, y_start,
                          full_width, full_height, padding_px=50):
    """Extract full-res bounding boxes from downsampled region labels.

    Returns list of dicts with region_id, array coords, and mosaic coords.
    """
    slices = ndimage.find_objects(region_labels)
    rois = []
    for region_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        n_px = int(np.sum(region_labels[sl] == region_id))
        if n_px == 0:
            continue

        # Convert downsampled coords to full-res array coords
        y0 = sl[0].start * downsample
        y1 = sl[0].stop * downsample
        x0 = sl[1].start * downsample
        x1 = sl[1].stop * downsample

        # Add padding
        y0 = max(0, y0 - padding_px)
        y1 = min(full_height, y1 + padding_px)
        x0 = max(0, x0 - padding_px)
        x1 = min(full_width, x1 + padding_px)

        rois.append({
            'region_id': region_id,
            # Array-relative coords (for slicing ch_data)
            'ay0': y0, 'ax0': x0,
            'height': y1 - y0, 'width': x1 - x0,
            # Global mosaic coords (for coordinate enrichment)
            'gx0': x0 + x_start,
            'gy0': y0 + y_start,
            'n_px_downsampled': n_px,
        })

    return rois


# ---------------------------------------------------------------------------
# Per-ROI detection
# ---------------------------------------------------------------------------

def detect_roi(roi, ch_data, display_chs, strategy, models, pixel_size,
               marker_map):
    """Run detection on a single ROI crop.

    Args:
        roi: dict with ay0, ax0, height, width, gx0, gy0, region_id
        ch_data: {ch_idx: full-slide np.ndarray} (read-only views)
        display_chs: list of 3 channel indices for RGB tile
        strategy: IsletStrategy instance (owned by this GPU thread)
        models: CellDetector.models dict (owned by this GPU thread)
        pixel_size: um/px
        marker_map: {name: ch_idx}

    Returns:
        list of enriched detection dicts (with global coords + medians)
    """
    from segmentation.detection.strategies.islet import _percentile_normalize_channel

    ay0 = roi['ay0']
    ax0 = roi['ax0']
    h = roi['height']
    w = roi['width']

    # Slice all channels for this ROI (numpy views — no copy)
    extra_channels = {}
    for ch_idx, full_arr in ch_data.items():
        extra_channels[ch_idx] = full_arr[ay0:ay0 + h, ax0:ax0 + w]

    # Build RGB display tile from display channels
    rgb_planes = []
    for ch_idx in display_chs[:3]:
        if ch_idx in extra_channels:
            rgb_planes.append(_percentile_normalize_channel(extra_channels[ch_idx]))
        else:
            rgb_planes.append(np.zeros((h, w), dtype=np.uint8))
    tile_rgb = np.stack(rgb_planes, axis=-1)  # HxWx3 uint8

    # Run detection (Cellpose + SAM2 on this thread's GPU)
    label_array, detections = strategy.detect(
        tile=tile_rgb,
        models=models,
        pixel_size_um=pixel_size,
        extract_features=True,
        extra_channels=extra_channels,
    )

    if not detections:
        return []

    # Extract marker medians inline from label_array + channel views
    label_slices = ndimage.find_objects(label_array)
    for det_idx, det in enumerate(detections):
        det_label = det_idx + 1  # label_array is 1-indexed
        if det_label > len(label_slices):
            continue
        sl = label_slices[det_label - 1]
        if sl is None:
            continue
        cell_mask = label_array[sl] == det_label
        for name, ch_idx in marker_map.items():
            if ch_idx not in extra_channels:
                continue
            vals = extra_channels[ch_idx][sl][cell_mask]
            vals = vals[vals > 0]
            median_val = float(np.median(vals)) if len(vals) > 0 else 0.0
            det.features[f'ch{ch_idx}_median'] = median_val

    # Extract contours from label_array (in ROI-local coords)
    import cv2
    contours_by_label = {}
    for det_idx in range(len(detections)):
        det_label = det_idx + 1
        binary = (label_array == det_label).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # Take the largest contour
            cnt = max(cnts, key=cv2.contourArea)
            contours_by_label[det_label] = cnt.squeeze().tolist()  # [[x,y], ...]

    # Convert Detection objects → feature dicts with global coordinates
    gx0 = roi['gx0']
    gy0 = roi['gy0']
    slide_name = roi.get('slide_name', 'slide')
    det_dicts = []
    for det_idx, det in enumerate(detections):
        feat = dict(det.features) if det.features else {}
        cx, cy = det.centroid
        feat['center'] = [cx, cy]
        feat['global_center'] = [float(gx0 + cx), float(gy0 + cy)]
        feat['global_center_um'] = [
            float((gx0 + cx) * pixel_size),
            float((gy0 + cy) * pixel_size),
        ]
        feat['tile_origin'] = [gx0, gy0]
        feat['slide_name'] = slide_name
        uid = f"{slide_name}_islet_{int(round(gx0 + cx))}_{int(round(gy0 + cy))}"
        feat['uid'] = uid
        feat['islet_id'] = roi['region_id']
        feat['score'] = det.score

        # Contour in global coords
        det_label = det_idx + 1
        local_contour = contours_by_label.get(det_label)
        global_contour = None
        if local_contour:
            if isinstance(local_contour[0], list):
                global_contour = [[pt[0] + gx0, pt[1] + gy0] for pt in local_contour]
            else:
                # Single-point contour (degenerate)
                global_contour = [[local_contour[0] + gx0, local_contour[1] + gy0]]

        det_dicts.append({
            'uid': feat['uid'],
            'global_center': feat['global_center'],
            'global_center_um': feat['global_center_um'],
            'tile_origin': feat['tile_origin'],
            'slide_name': feat['slide_name'],
            'center': feat['center'],
            'islet_id': roi['region_id'],
            'score': det.score,
            'features': feat,
            'contour': global_contour,
        })

    del label_array, detections, extra_channels, tile_rgb, contours_by_label
    return det_dicts


# ---------------------------------------------------------------------------
# Multi-GPU worker thread
# ---------------------------------------------------------------------------

def _init_gpu(gpu_id):
    """Initialize CellDetector on a specific GPU. Called once per GPU."""
    import torch
    torch.cuda.set_device(gpu_id)

    from segmentation.detection.cell_detector import CellDetector

    device = f"cuda:{gpu_id}"
    print(f"  [GPU-{gpu_id}] Loading models on {device}...", flush=True)
    detector = CellDetector(device=device)
    # Trigger lazy load of Cellpose + SAM2
    _ = detector.models['cellpose']
    _ = detector.models['sam2_predictor']
    print(f"  [GPU-{gpu_id}] Models ready", flush=True)
    return detector


def _gpu_worker(gpu_id, detector, assigned_rois, ch_data, display_chs,
                membrane_ch, nuclear_ch, pixel_size, marker_map):
    """Worker thread: processes ROIs on a pre-initialized GPU.

    Models are loaded once in _init_gpu() and reused across modes.
    ch_data is shared read-only across threads.

    Returns:
        list of detection dicts from all assigned ROIs
    """
    import torch
    torch.cuda.set_device(gpu_id)

    from segmentation.detection.strategies.islet import IsletStrategy

    strategy = IsletStrategy(
        membrane_channel=membrane_ch,
        nuclear_channel=nuclear_ch,
        extract_sam2_embeddings=True,
        marker_signal_factor=0,  # disable GMM pre-filter (all cells are in islets)
    )

    print(f"  [GPU-{gpu_id}] Processing {len(assigned_rois)} ROIs "
          f"(membrane={'None' if membrane_ch is None else membrane_ch})...",
          flush=True)

    all_dets = []
    for i, roi in enumerate(assigned_rois):
        t_roi = time.time()
        roi_dets = detect_roi(
            roi, ch_data, display_chs, strategy, detector.models,
            pixel_size, marker_map,
        )
        all_dets.extend(roi_dets)
        dt = time.time() - t_roi
        print(f"  [GPU-{gpu_id}] Region {roi['region_id']} "
              f"({roi['width']}x{roi['height']}): "
              f"{len(roi_dets)} cells ({dt:.1f}s)", flush=True)

    print(f"  [GPU-{gpu_id}] Done: {len(all_dets)} cells total", flush=True)
    return all_dets


def init_gpus(num_gpus):
    """Initialize CellDetector on each GPU in parallel. Returns list of detectors."""
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        detectors = list(pool.map(_init_gpu, range(num_gpus)))
    return detectors


def process_mode_multigpu(mode_name, membrane_ch, nuclear_ch, rois, ch_data,
                          display_chs, pixel_size, marker_map, num_gpus,
                          detectors):
    """Process all ROIs for one mode using multiple GPUs in parallel.

    Splits ROIs round-robin across GPUs, launches one thread per GPU.
    Detectors are pre-initialized and reused across modes.
    """
    # Split ROIs across GPUs (round-robin for load balance)
    rois_per_gpu = [[] for _ in range(num_gpus)]
    for i, roi in enumerate(rois):
        rois_per_gpu[i % num_gpus].append(roi)

    print(f"  Distributing {len(rois)} ROIs across {num_gpus} GPUs: "
          + ", ".join(f"GPU-{g}={len(rois_per_gpu[g])}" for g in range(num_gpus)),
          flush=True)

    all_dets = []
    with ThreadPoolExecutor(max_workers=num_gpus) as pool:
        futures = []
        for gpu_id in range(num_gpus):
            if not rois_per_gpu[gpu_id]:
                continue
            future = pool.submit(
                _gpu_worker, gpu_id, detectors[gpu_id],
                rois_per_gpu[gpu_id], ch_data,
                display_chs, membrane_ch, nuclear_ch, pixel_size, marker_map,
            )
            futures.append(future)

        for future in futures:
            all_dets.extend(future.result())

    return all_dets


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_detections(detections, marker_map, percentile=95):
    """Population-level marker classification using percentile thresholds.

    Same logic as analyze_islets.classify_by_percentile() but operates on
    the nested dict format from detect_roi().
    """
    ch_vals = {}
    for name, ch_idx in marker_map.items():
        key = f'ch{ch_idx}_median'
        vals = [d.get('features', {}).get(key) for d in detections]
        ch_vals[name] = [v for v in vals if v is not None and v > 0]

    thresholds = {}
    for name, vals in ch_vals.items():
        thresholds[name] = float(np.percentile(vals, percentile)) if vals else float('inf')

    print(f"  Percentile thresholds (p{percentile}):")
    for name, t in thresholds.items():
        ch_idx = marker_map[name]
        n_above = sum(1 for v in ch_vals[name] if v >= t)
        print(f"    {name} (ch{ch_idx}): {t:.1f}  ({n_above} cells above)")

    counts = Counter()
    for det in detections:
        feats = det.get('features', {})
        above = {}
        for name, ch_idx in marker_map.items():
            val = feats.get(f'ch{ch_idx}_median', 0)
            if val >= thresholds[name]:
                above[name] = val

        if not above:
            det['marker_class'] = 'none'
        elif len(above) == 1:
            det['marker_class'] = next(iter(above))
        else:
            sorted_markers = sorted(above.items(), key=lambda x: x[1], reverse=True)
            dominant_val = sorted_markers[0][1]
            runner_up_val = sorted_markers[1][1]
            if runner_up_val > 0 and dominant_val < 1.5 * runner_up_val:
                det['marker_class'] = 'multi'
            else:
                det['marker_class'] = sorted_markers[0][0]

        counts[det['marker_class']] += 1

    print(f"  Classification: {dict(counts)}")
    return thresholds


# ---------------------------------------------------------------------------
# Comparison stats
# ---------------------------------------------------------------------------

def compute_comparison(dets_nuc, dets_pm, marker_map, pixel_size):
    """Compute per-islet and overall comparison between nuclei-only and PM+nuclei."""
    stats = {'per_islet': [], 'overall': {}}

    def group_by_islet(dets):
        groups = {}
        for d in dets:
            iid = d.get('islet_id', -1)
            if iid > 0:
                groups.setdefault(iid, []).append(d)
        return groups

    nuc_groups = group_by_islet(dets_nuc)
    pm_groups = group_by_islet(dets_pm)
    all_islet_ids = sorted(set(nuc_groups.keys()) | set(pm_groups.keys()))

    for iid in all_islet_ids:
        nuc_cells = nuc_groups.get(iid, [])
        pm_cells = pm_groups.get(iid, [])

        row = {'islet_id': iid}
        row['n_nuclei'] = len(nuc_cells)
        row['n_pm'] = len(pm_cells)
        row['count_ratio'] = round(len(pm_cells) / len(nuc_cells), 3) if len(nuc_cells) > 0 else 0

        def median_area(cells):
            areas = [c.get('features', {}).get('area', 0) for c in cells]
            areas = [a * pixel_size ** 2 for a in areas if a > 0]
            return float(np.median(areas)) if areas else 0.0

        row['median_area_nuc_um2'] = round(median_area(nuc_cells), 1)
        row['median_area_pm_um2'] = round(median_area(pm_cells), 1)
        if row['median_area_nuc_um2'] > 0:
            row['area_ratio'] = round(row['median_area_pm_um2'] / row['median_area_nuc_um2'], 3)
        else:
            row['area_ratio'] = 0

        for name in marker_map.keys():
            nuc_frac = sum(1 for c in nuc_cells if c.get('marker_class') == name) / len(nuc_cells) if nuc_cells else 0
            pm_frac = sum(1 for c in pm_cells if c.get('marker_class') == name) / len(pm_cells) if pm_cells else 0
            row[f'frac_{name}_nuc'] = round(nuc_frac, 3)
            row[f'frac_{name}_pm'] = round(pm_frac, 3)

        stats['per_islet'].append(row)

    # Overall
    stats['overall']['total_nuclei'] = len(dets_nuc)
    stats['overall']['total_pm'] = len(dets_pm)
    stats['overall']['n_islets_nuclei'] = len(nuc_groups)
    stats['overall']['n_islets_pm'] = len(pm_groups)

    for mode_name, dets in [('nuclei', dets_nuc), ('pm', dets_pm)]:
        counts = Counter(d.get('marker_class', 'none') for d in dets)
        for name in list(marker_map.keys()) + ['multi', 'none']:
            stats['overall'][f'{mode_name}_{name}'] = counts.get(name, 0)

    return stats


# ---------------------------------------------------------------------------
# JSON sanitization
# ---------------------------------------------------------------------------

def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()

    czi_path = Path(args.czi_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_map = parse_marker_channels(args.marker_channels)
    display_chs = [int(x) for x in args.display_channels.split(',')]
    slide_name = czi_path.stem
    num_gpus = args.num_gpus

    print(f"=== ROI-based Islet Segmentation ({num_gpus} GPUs) ===")
    print(f"  CZI: {czi_path}")
    print(f"  Markers: {marker_map}")
    print(f"  Display: {display_chs}")
    print(f"  Mode: {args.mode}")
    print()

    # ------------------------------------------------------------------
    # 1. Load channels into RAM
    # ------------------------------------------------------------------
    print("Step 1: Loading CZI channels...")
    all_ch_indices = sorted(set(
        list(marker_map.values()) +
        display_chs +
        [args.membrane_channel, args.nuclear_channel]
    ))

    sys.path.insert(0, str(REPO / 'scripts'))
    from analyze_islets import load_czi_direct, find_islet_regions

    pixel_size, x_start, y_start, ch_data = load_czi_direct(
        czi_path, all_ch_indices
    )
    any_ch = next(iter(ch_data.values()))
    full_height, full_width = any_ch.shape
    print(f"  Loaded {len(ch_data)} channels, slide {full_width}x{full_height}, "
          f"pixel_size={pixel_size:.4f} um")
    print(f"  Mosaic origin: ({x_start}, {y_start})")
    print()

    # ------------------------------------------------------------------
    # 2. Find islet regions
    # ------------------------------------------------------------------
    print("Step 2: Finding islet regions...")
    marker_ch_data = {ch: ch_data[ch] for ch in marker_map.values() if ch in ch_data}

    region_labels, downsample, signal = find_islet_regions(
        marker_ch_data, marker_map, pixel_size,
        otsu_multiplier=args.otsu_multiplier,
        buffer_um=args.buffer_um,
    )
    n_regions = int(region_labels.max())
    print(f"  Found {n_regions} islet regions")
    print()

    # ------------------------------------------------------------------
    # 3. Extract region bounding boxes
    # ------------------------------------------------------------------
    print("Step 3: Extracting region bounding boxes...")
    rois = extract_region_bboxes(
        region_labels, downsample, x_start, y_start,
        full_width, full_height, padding_px=args.roi_padding_px,
    )
    for roi in rois:
        roi['slide_name'] = slide_name

    print(f"  {len(rois)} ROIs extracted:")
    for roi in rois:
        print(f"    Region {roi['region_id']}: {roi['width']}x{roi['height']} px "
              f"at ({roi['gx0']}, {roi['gy0']})")

    # Save region bboxes
    with open(output_dir / 'region_bboxes.json', 'w') as f:
        json.dump(sanitize_for_json([dict(r) for r in rois]), f)
    print()

    # Free region finding intermediates (keep region_labels for post-detection filtering)
    del signal, marker_ch_data
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Init models on each GPU (once, reused across modes)
    # ------------------------------------------------------------------
    print(f"Step 4: Initializing Cellpose + SAM2 on {num_gpus} GPUs...")
    detectors = init_gpus(num_gpus)
    print()

    # ------------------------------------------------------------------
    # 5. Process each mode (multi-GPU parallel)
    # ------------------------------------------------------------------
    modes_to_run = []
    if args.mode in ('both', 'nuclei'):
        modes_to_run.append(('nuclei', None))
    if args.mode in ('both', 'pm'):
        modes_to_run.append(('pm', args.membrane_channel))

    results = {}  # mode_name -> list of det dicts

    for mode_name, membrane_ch in modes_to_run:
        t_mode = time.time()
        print(f"Step 5: Processing ROIs — {mode_name} mode "
              f"(membrane={'None' if membrane_ch is None else membrane_ch}, "
              f"{num_gpus} GPUs)...")

        mode_dets = process_mode_multigpu(
            mode_name, membrane_ch, args.nuclear_channel,
            rois, ch_data, display_chs, pixel_size, marker_map, num_gpus,
            detectors,
        )

        # Filter: keep only cells whose centroid falls inside the islet mask
        pre_filter = len(mode_dets)
        filtered = []
        ds_h, ds_w = region_labels.shape
        for d in mode_dets:
            gx, gy = d['global_center']
            # Convert global pixel coords to downsampled label coords
            lx = int(round((gx - x_start) / downsample))
            ly = int(round((gy - y_start) / downsample))
            if 0 <= ly < ds_h and 0 <= lx < ds_w and region_labels[ly, lx] > 0:
                filtered.append(d)
        mode_dets = filtered

        dt_mode = time.time() - t_mode
        print(f"  [{mode_name}] Total: {len(mode_dets)} cells "
              f"({pre_filter} detected, {pre_filter - len(mode_dets)} outside islets) "
              f"across {len(rois)} ROIs ({dt_mode:.0f}s)")
        results[mode_name] = mode_dets
        print()

    del region_labels
    gc.collect()

    # ------------------------------------------------------------------
    # 5. Population-level marker classification
    # ------------------------------------------------------------------
    print("Step 6: Marker classification...")
    thresholds = {}
    for mode_name, dets in results.items():
        print(f"  --- {mode_name} ---")
        if len(dets) >= args.min_cells:
            thresholds[mode_name] = classify_detections(
                dets, marker_map, percentile=args.marker_percentile
            )
        else:
            print(f"  Too few cells ({len(dets)}) for classification")
            for d in dets:
                d['marker_class'] = 'none'
    print()

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    # Cleanup GPU memory
    import torch
    for det in detectors:
        det.cleanup()
    torch.cuda.empty_cache()

    print("Step 7: Saving outputs...")
    for mode_name, dets in results.items():
        out_path = output_dir / f'detections_{mode_name}.json'
        clean = sanitize_for_json(dets)
        with open(out_path, 'w') as f:
            json.dump(clean, f)
        print(f"  {out_path} ({len(dets)} detections)")

    # Comparison stats (if both modes ran)
    if 'nuclei' in results and 'pm' in results:
        print("\n  --- Comparison ---")
        comp = compute_comparison(
            results['nuclei'], results['pm'], marker_map, pixel_size
        )
        comp_path = output_dir / 'comparison_stats.json'
        with open(comp_path, 'w') as f:
            json.dump(sanitize_for_json(comp), f)
        print(f"  Saved: {comp_path}")

        # Print summary table
        print(f"\n  {'Islet':>6} {'Nuc':>5} {'PM':>5} {'Ratio':>6} "
              f"{'AreaNuc':>8} {'AreaPM':>8} {'AratR':>6}")
        for row in comp['per_islet']:
            print(f"  {row['islet_id']:>6} {row['n_nuclei']:>5} {row['n_pm']:>5} "
                  f"{row['count_ratio']:>6.2f} "
                  f"{row['median_area_nuc_um2']:>8.1f} {row['median_area_pm_um2']:>8.1f} "
                  f"{row['area_ratio']:>6.2f}")

        ov = comp['overall']
        if ov['total_nuclei'] > 0:
            print(f"\n  Overall: nuclei={ov['total_nuclei']}, pm={ov['total_pm']}, "
                  f"ratio={ov['total_pm']/ov['total_nuclei']:.2f}")

    dt_total = time.time() - t0
    print(f"\nDone in {dt_total:.0f}s ({dt_total/60:.1f} min)")


if __name__ == '__main__':
    main()
