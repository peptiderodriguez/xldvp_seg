#!/usr/bin/env python3
"""Regenerate islet HTML with marker-colored contours from existing detections.

Reads saved features + masks, loads only the display channels from CZI,
computes population-level marker thresholds, and regenerates HTML with
colored contours by marker type.
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
    get_czi_metadata,
)
from segmentation.io.html_export import export_samples_to_html


def parse_marker_channels(s):
    """Parse 'gcg:2,ins:3,sst:5' into dict."""
    result = {}
    for pair in s.split(','):
        name, ch = pair.strip().split(':')
        result[name.strip()] = int(ch.strip())
    return result


def main():
    parser = argparse.ArgumentParser(description="Regenerate islet HTML with marker coloring")
    parser.add_argument("--run-dir", required=True, help="Path to existing run output directory")
    parser.add_argument("--czi-path", required=True, help="Path to CZI file")
    parser.add_argument("--marker-only", action="store_true",
                        help="Only show cells expressing a marker (exclude gray/none)")
    parser.add_argument("--display-channels", type=str, default="2,3,5",
                        help="Comma-separated R,G,B channel indices for display (default: 2,3,5)")
    parser.add_argument("--marker-channels", type=str, default="gcg:2,ins:3,sst:5",
                        help="Marker-to-channel mapping for classification (default: gcg:2,ins:3,sst:5)")
    parser.add_argument("--threshold-gcg", type=float, default=None,
                        help="Override first marker normalized threshold (default: Otsu auto)")
    parser.add_argument("--threshold-ins", type=float, default=None,
                        help="Override second marker normalized threshold (default: Otsu auto)")
    parser.add_argument("--threshold-sst", type=float, default=None,
                        help="Override third marker normalized threshold (default: Otsu auto)")
    parser.add_argument("--ratio-min", type=float, default=1.5,
                        help="Dominant marker must be >= ratio_min * runner-up (default: 1.5). "
                             "Cells below this ratio are classified as 'multi'.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    czi_path = Path(args.czi_path)

    # Parse channel config
    display_chs = [int(x.strip()) for x in args.display_channels.split(',')]
    marker_map = parse_marker_channels(args.marker_channels)
    marker_names = list(marker_map.keys())

    # Load existing detections
    det_path = run_dir / "islet_detections.json"
    print(f"Loading detections from {det_path}")
    with open(det_path) as f:
        all_detections = json.load(f)
    print(f"  {len(all_detections)} detections")

    # Load CZI — only the display channels
    print(f"Loading CZI display channels {display_chs}...")
    loader = get_loader(czi_path, load_to_ram=True, channel=display_chs[0])
    pixel_size_um = loader.get_pixel_size()
    x_start = loader.x_start
    y_start = loader.y_start
    slide_name = czi_path.stem

    # Load display channels
    all_channel_data = {}
    for ch in display_chs:
        print(f"  Loading channel {ch}...")
        loader.load_channel(ch)
        all_channel_data[ch] = loader._channel_data[ch]

    # Build threshold overrides keyed by ch{N}
    overrides = {}
    threshold_args = [args.threshold_gcg, args.threshold_ins, args.threshold_sst]
    for i, name in enumerate(marker_names):
        if i < len(threshold_args) and threshold_args[i] is not None:
            overrides[f'ch{marker_map[name]}'] = threshold_args[i]

    marker_thresholds = compute_islet_marker_thresholds(
        all_detections, vis_threshold_overrides=overrides or None,
        ratio_min=args.ratio_min, marker_map=marker_map,
    )

    # Classify all detections
    counts = {}
    for det in all_detections:
        mc, _ = classify_islet_marker(det.get('features', {}), marker_thresholds, marker_map=marker_map)
        det['marker_class'] = mc
        counts[mc] = counts.get(mc, 0) + 1
    print(f"Marker classification: {counts}")

    # Save updated detections with marker_class
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
                if classify_islet_marker(f.get('features', {}), marker_thresholds, marker_map=marker_map)[0] != 'none'
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

        rgb_channels = []
        for ch in display_chs[:3]:
            if ch in all_channel_data:
                rgb_channels.append(all_channel_data[ch][rel_ty:rel_ty+tile_h, rel_tx:rel_tx+tile_w])
            else:
                rgb_channels.append(np.zeros((tile_h, tile_w), dtype=np.uint16))
        while len(rgb_channels) < 3:
            rgb_channels.append(np.zeros((tile_h, tile_w), dtype=np.uint16))
        tile_rgb = np.stack(rgb_channels, axis=-1)

        tile_pct = _compute_tile_percentiles(tile_rgb)

        html_samples = filter_and_create_html_samples(
            features_list, tile_x, tile_y, tile_rgb, masks,
            pixel_size_um, slide_name, 'islet',
            html_score_threshold=0.0,
            tile_percentiles=tile_pct,
            marker_thresholds=marker_thresholds,
            marker_map=marker_map,
        )
        all_samples.extend(html_samples)
        print(f"  {len(html_samples)} HTML samples")

    print(f"\nTotal HTML samples: {len(all_samples)}")

    # Generate HTML — derive legend from CZI metadata
    html_dir = run_dir / "html"
    try:
        meta = get_czi_metadata(czi_path)
        def _ch_label(idx):
            for ch in meta['channels']:
                if ch['index'] == idx:
                    em = f" ({ch['emission_nm']:.0f}nm)" if ch.get('emission_nm') else ''
                    return f"{ch['name']}{em}"
            return f'Ch{idx}'
    except Exception:
        def _ch_label(idx):
            return f'Ch{idx}'

    channel_legend = {
        'red': _ch_label(display_chs[0]) if len(display_chs) > 0 else 'none',
        'green': _ch_label(display_chs[1]) if len(display_chs) > 1 else 'none',
        'blue': _ch_label(display_chs[2]) if len(display_chs) > 2 else 'none',
    }

    export_samples_to_html(
        all_samples,
        str(html_dir),
        title=f"{slide_name} — Islet Cells (marker-colored)",
        cell_type='islet',
        channel_legend=channel_legend,
    )
    print(f"\nHTML exported to {html_dir}")
    marker_colors = ['red', 'green', 'blue']
    legend_parts = [f"{c} contour={n}" for c, n in zip(marker_colors, marker_names)]
    print(f"Marker legend: {', '.join(legend_parts)}, gray=none")


if __name__ == '__main__':
    main()
