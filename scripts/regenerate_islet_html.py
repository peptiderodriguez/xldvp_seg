#!/usr/bin/env python3
"""Regenerate islet HTML with marker-colored contours from existing detections.

Reads saved features + masks, loads only the 3 display channels from CZI,
computes population-level marker thresholds, and regenerates HTML with
colored contours: red=alpha(Gcg), green=beta(Ins), blue=delta(Sst), gray=none.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.io.czi_loader import get_loader
from run_segmentation import (
    classify_islet_marker,
    compute_islet_marker_thresholds,
    filter_and_create_html_samples,
    _compute_tile_percentiles,
)
from segmentation.io.html_export import export_samples_to_html


def main():
    parser = argparse.ArgumentParser(description="Regenerate islet HTML with marker coloring")
    parser.add_argument("--run-dir", required=True, help="Path to existing run output directory")
    parser.add_argument("--czi-path", required=True, help="Path to CZI file")
    parser.add_argument("--marker-only", action="store_true",
                        help="Only show cells expressing a marker (exclude gray/none)")
    parser.add_argument("--threshold-gcg", type=float, default=None,
                        help="Override Gcg (alpha) normalized threshold (default: Otsu auto)")
    parser.add_argument("--threshold-ins", type=float, default=None,
                        help="Override Ins (beta) normalized threshold (default: Otsu auto)")
    parser.add_argument("--threshold-sst", type=float, default=None,
                        help="Override Sst (delta) normalized threshold (default: Otsu auto)")
    parser.add_argument("--ratio-min", type=float, default=1.5,
                        help="Dominant marker must be >= ratio_min * runner-up (default: 1.5). "
                             "Cells below this ratio are classified as 'multi'.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    czi_path = Path(args.czi_path)

    # Load existing detections
    det_path = run_dir / "islet_detections.json"
    print(f"Loading detections from {det_path}")
    with open(det_path) as f:
        all_detections = json.load(f)
    print(f"  {len(all_detections)} detections")

    # Load CZI — only the 3 display channels + detection channel for metadata
    print(f"Loading CZI display channels (2=Gcg, 3=Ins, 5=Sst)...")
    loader = get_loader(czi_path, load_to_ram=True, channel=1)
    pixel_size_um = loader.get_pixel_size()
    x_start = loader.x_start
    y_start = loader.y_start
    slide_name = czi_path.stem

    # Load display channels
    all_channel_data = {}
    for ch in [2, 3, 5]:
        print(f"  Loading channel {ch}...")
        loader.load_channel(ch)
        all_channel_data[ch] = loader._channel_data[ch]

    # Compute marker thresholds from population (Otsu auto per channel, with optional overrides)
    overrides = {}
    if args.threshold_gcg is not None:
        overrides['ch2'] = args.threshold_gcg
    if args.threshold_ins is not None:
        overrides['ch3'] = args.threshold_ins
    if args.threshold_sst is not None:
        overrides['ch5'] = args.threshold_sst
    marker_thresholds = compute_islet_marker_thresholds(
        all_detections, vis_threshold_overrides=overrides or None,
        ratio_min=args.ratio_min,
    )

    # Classify all detections
    counts = {}
    for det in all_detections:
        mc, _ = classify_islet_marker(det.get('features', {}), marker_thresholds)
        det['marker_class'] = mc
        counts[mc] = counts.get(mc, 0) + 1
    print(f"Marker classification: {counts}")

    # Save updated detections with marker_class
    # Use recursive sanitizer to handle NaN/inf (NumpyEncoder passes them through)
    import math
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (float,)):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        if hasattr(obj, 'item'):  # numpy scalar
            v = obj.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        if hasattr(obj, 'tolist'):  # numpy array
            return _sanitize(obj.tolist())
        return obj
    with open(det_path, 'w') as f:
        json.dump(_sanitize(all_detections), f)
    print(f"Updated {det_path} with marker_class")

    # Process each tile
    tiles_dir = run_dir / "tiles"
    all_samples = []

    for tile_dir in sorted(tiles_dir.iterdir()):
        if not tile_dir.is_dir() or not tile_dir.name.startswith('tile_'):
            continue

        # Parse tile coordinates from directory name
        try:
            parts = tile_dir.name.split('_')
            tile_x = int(parts[1])
            tile_y = int(parts[2])
        except (IndexError, ValueError):
            print(f"Skipping unrecognized directory: {tile_dir.name}")
            continue
        print(f"\nProcessing {tile_dir.name} (tile_x={tile_x}, tile_y={tile_y})")

        # Load features
        feat_path = tile_dir / "islet_features.json"
        if not feat_path.exists():
            print(f"  No features file, skipping")
            continue
        with open(feat_path) as f:
            features_list = json.load(f)
        if not features_list:
            print(f"  No features, skipping")
            continue

        # Filter to marker-positive only if requested
        if args.marker_only:
            features_list = [
                f for f in features_list
                if classify_islet_marker(f.get('features', {}), marker_thresholds)[0] != 'none'
            ]
        print(f"  {len(features_list)} detections")

        # Load masks
        mask_path = tile_dir / "islet_masks.h5"
        if not mask_path.exists():
            print(f"  No mask file, skipping")
            continue
        with h5py.File(mask_path, 'r') as f:
            masks = f['masks'][:]

        # Build RGB tile from display channels
        rel_tx = tile_x - x_start
        rel_ty = tile_y - y_start
        tile_h, tile_w = masks.shape[:2]

        tile_rgb = np.stack([
            all_channel_data[2][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
            all_channel_data[3][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
            all_channel_data[5][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w],
        ], axis=-1)

        tile_pct = _compute_tile_percentiles(tile_rgb)

        html_samples = filter_and_create_html_samples(
            features_list, tile_x, tile_y, tile_rgb, masks,
            pixel_size_um, slide_name, 'islet',
            html_score_threshold=0.0,
            tile_percentiles=tile_pct,
            marker_thresholds=marker_thresholds,
        )
        all_samples.extend(html_samples)
        print(f"  {len(html_samples)} HTML samples")

    print(f"\nTotal HTML samples: {len(all_samples)}")

    # Generate HTML
    html_dir = run_dir / "html"
    channel_legend = {
        'red': 'Gcg (alpha)',
        'green': 'Ins (beta)',
        'blue': 'Sst (delta)',
    }

    export_samples_to_html(
        all_samples,
        str(html_dir),
        title=f"{slide_name} — Islet Cells (marker-colored)",
        cell_type='islet',
        channel_legend=channel_legend,
    )
    print(f"\nHTML exported to {html_dir}")
    print(f"Marker legend: red contour=alpha, green=beta, blue=delta, gray=none")


if __name__ == '__main__':
    main()
