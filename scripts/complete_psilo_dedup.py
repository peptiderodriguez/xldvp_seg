#!/usr/bin/env python3
"""Whole Mouse PM: reassemble detections from per-tile JSONs,
run spatial-grid dedup, save final JSON + CSV.

The original full run (job 2265425) completed all 1403 tiles but got stuck
in the O(n*kept) dedup loop at 190K/443K. This script uses the new spatial
grid dedup to finish in minutes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


DEFAULT_RUN_DIR = ("/fs/pool/pool-mann-edwin/psilo_output/tp_full/"
                   "20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct")
DEFAULT_CELL_TYPE = "tissue_pattern"
DEFAULT_PIXEL_SIZE = 0.1725  # um/px


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reassemble per-tile detections and run spatial-grid dedup."
    )
    parser.add_argument('--run-dir', type=str, default=DEFAULT_RUN_DIR,
                        help='Run output directory containing tiles/')
    parser.add_argument('--tiles-dir', type=str, default=None,
                        help='Tiles directory (default: <run-dir>/tiles)')
    parser.add_argument('--cell-type', type=str, default=DEFAULT_CELL_TYPE,
                        help='Cell type string for filenames (default: tissue_pattern)')
    parser.add_argument('--pixel-size', type=float, default=DEFAULT_PIXEL_SIZE,
                        help='Pixel size in um/px (default: 0.1725)')
    return parser.parse_args()


def main():
    args = parse_args()
    RUN_DIR = Path(args.run_dir)
    TILES_DIR = Path(args.tiles_dir) if args.tiles_dir else RUN_DIR / "tiles"
    CELL_TYPE = args.cell_type
    PIXEL_SIZE = args.pixel_size

    # 1. Reassemble all detections from per-tile feature JSONs
    print(f"Reassembling detections from {TILES_DIR}...")
    all_detections = []
    tile_dirs = sorted(TILES_DIR.iterdir())
    n_tiles = 0
    for td in tile_dirs:
        feat_file = td / f"{CELL_TYPE}_features.json"
        if not feat_file.exists():
            continue
        with open(feat_file) as f:
            features_list = json.load(f)
        all_detections.extend(features_list)
        n_tiles += 1

    print(f"  {n_tiles} tiles, {len(all_detections)} detections (pre-dedup)")

    # 2. Run spatial grid dedup
    from segmentation.processing.deduplication import deduplicate_by_mask_overlap
    pre_dedup = len(all_detections)
    all_detections = deduplicate_by_mask_overlap(
        all_detections, TILES_DIR, min_overlap_fraction=0.1,
        mask_filename=f'{CELL_TYPE}_masks.h5', sort_by='area',
    )
    print(f"  Dedup: {pre_dedup} -> {len(all_detections)} "
          f"({pre_dedup - len(all_detections)} removed)")

    # 3. Add tile_mask_label + global_id (same as run_segmentation.py lines 3129-3132)
    for det in all_detections:
        det['tile_mask_label'] = det.get('mask_label', 0)
        gc = det.get('global_center', det.get('center', [0, 0]))
        det['global_id'] = f"{int(round(gc[0]))}_{int(round(gc[1]))}"

    # 4. Save final JSON
    det_file = RUN_DIR / f"{CELL_TYPE}_detections.json"
    with open(det_file, 'w') as f:
        json.dump(all_detections, f, cls=NumpyEncoder)
    print(f"  Saved {len(all_detections)} detections to {det_file}")

    # 5. Save CSV
    csv_file = RUN_DIR / f"{CELL_TYPE}_coordinates.csv"
    with open(csv_file, 'w') as f:
        f.write('uid,global_x_px,global_y_px,global_x_um,global_y_um,area_um2\n')
        for det in all_detections:
            g_center = det.get('global_center')
            g_center_um = det.get('global_center_um')
            if g_center is None or g_center_um is None:
                continue
            if len(g_center) < 2 or g_center[0] is None or g_center[1] is None:
                continue
            if len(g_center_um) < 2 or g_center_um[0] is None or g_center_um[1] is None:
                continue
            feat = det.get('features', {})
            area_um2 = feat.get('area', 0) * (PIXEL_SIZE ** 2)
            f.write(f"{det.get('uid','')},{g_center[0]:.1f},{g_center[1]:.1f},"
                    f"{g_center_um[0]:.2f},{g_center_um[1]:.2f},{area_um2:.2f}\n")
    print(f"  Saved coordinates to {csv_file}")

    print("\nDone! HTML was not regenerated (requires in-memory tile crops).")
    print(f"To regenerate HTML, re-run the full pipeline or use a crop-from-disk script.")


if __name__ == '__main__':
    main()
