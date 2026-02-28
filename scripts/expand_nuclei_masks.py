#!/usr/bin/env python3
"""Expand nuclei-only masks to approximate PM+nuclei cell body size.

After running islet segmentation in both nuclei-only and PM+nuclei modes,
this script:
1. Compares area distributions between the two runs
2. Computes the dilation radius needed to match nuclei→PM+nuclei sizes
3. Dilates nuclei-only masks in HDF5 files (label-aware, no overlap)
4. Updates detection JSON with new areas

The dilated masks capture cytoplasmic signal that nuclear-only segmentation
misses, giving better marker quantification (Gcg, Ins, Sst are cytoplasmic).

Usage:
    python scripts/expand_nuclei_masks.py \
        --nuclei-dir /path/to/BS100_nuclei_only/run_dir \
        --pm-dir /path/to/BS100_pm_nuclei/run_dir \
        [--dilation-px N]  # override auto-computed dilation
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


def load_detections(run_dir):
    """Load islet detections JSON from a run directory."""
    det_path = Path(run_dir) / 'islet_detections.json'
    if not det_path.exists():
        raise FileNotFoundError(f"No detections at {det_path}")
    with open(det_path) as f:
        return json.load(f)


def get_area_distribution(detections):
    """Extract area (pixels) from all detections."""
    areas = []
    for det in detections:
        feat = det.get('features', {})
        area = feat.get('area', 0)
        if area > 0:
            areas.append(area)
    return np.array(areas)


def compute_dilation_radius(nuclei_areas, pm_areas, pixel_size_um=None):
    """Compute dilation radius in pixels from area distributions.

    Models cells as circles: r = sqrt(area / pi).
    Dilation = r_pm - r_nuc (median of each distribution).

    Returns:
        dilation_px: integer dilation radius in pixels
        stats: dict with comparison statistics
    """
    if pixel_size_um is None:
        raise ValueError("pixel_size_um is required — get from CZI metadata")
    nuc_median = np.median(nuclei_areas)
    pm_median = np.median(pm_areas)

    r_nuc = np.sqrt(nuc_median / np.pi)
    r_pm = np.sqrt(pm_median / np.pi)
    dilation_px = max(1, int(round(r_pm - r_nuc)))

    area_ratio = pm_median / nuc_median if nuc_median > 0 else 1.0

    stats = {
        'nuclei_count': len(nuclei_areas),
        'pm_count': len(pm_areas),
        'nuclei_median_area_px': float(nuc_median),
        'pm_median_area_px': float(pm_median),
        'nuclei_median_area_um2': float(nuc_median * pixel_size_um ** 2),
        'pm_median_area_um2': float(pm_median * pixel_size_um ** 2),
        'area_ratio': float(area_ratio),
        'nuclei_median_radius_px': float(r_nuc),
        'pm_median_radius_px': float(r_pm),
        'dilation_px': dilation_px,
        'dilation_um': float(dilation_px * pixel_size_um),
        'nuclei_p25_area': float(np.percentile(nuclei_areas, 25)),
        'nuclei_p75_area': float(np.percentile(nuclei_areas, 75)),
        'pm_p25_area': float(np.percentile(pm_areas, 25)),
        'pm_p75_area': float(np.percentile(pm_areas, 75)),
    }

    return dilation_px, stats


def dilate_label_array(label_array, dilation_px):
    """Dilate each label in a label array by dilation_px pixels.

    Expands each cell outward into unclaimed (0) pixels. When two cells
    compete for the same pixel, the nearer cell (by distance transform) wins.

    Uses a distance-based approach:
    1. Compute distance transform from each label's boundary
    2. For unclaimed pixels within dilation_px of any label, assign to nearest

    Args:
        label_array: 2D uint32 label array (0=background, >0=cell labels)
        dilation_px: number of pixels to dilate

    Returns:
        Dilated label array (same shape, dtype)
    """
    if dilation_px <= 0 or label_array.max() == 0:
        return label_array.copy()

    # Distance from each pixel to its nearest labeled pixel
    background = label_array == 0
    if not background.any():
        return label_array.copy()  # no room to expand

    # Distance transform: distance from each background pixel to nearest foreground
    dist, nearest_idx = ndimage.distance_transform_edt(
        background, return_distances=True, return_indices=True
    )

    # Expand: assign background pixels within dilation_px to nearest label
    result = label_array.copy()
    expand_mask = background & (dist <= dilation_px)
    if expand_mask.any():
        # nearest_idx[0] = row indices, nearest_idx[1] = col indices of nearest foreground
        result[expand_mask] = label_array[
            nearest_idx[0][expand_mask],
            nearest_idx[1][expand_mask]
        ]

    return result


def expand_tile_masks(mask_path, dilation_px):
    """Dilate masks in an HDF5 file and return updated areas per label.

    Args:
        mask_path: Path to islet_masks.h5
        dilation_px: pixels to dilate

    Returns:
        dict mapping label -> new_area_px, or None if file doesn't exist
    """
    import h5py
    try:
        import hdf5plugin  # noqa: F401 — needed for LZ4 decompression
    except ImportError:
        pass

    if not mask_path.exists():
        return None

    with h5py.File(mask_path, 'r') as f:
        labels = f['masks'][:]

    if labels.max() == 0:
        return {}

    expanded = dilate_label_array(labels, dilation_px)

    # Save back
    with h5py.File(mask_path, 'w') as f:
        f.create_dataset('masks', data=expanded, compression='gzip', compression_opts=4)

    # Compute new areas
    unique_labels = np.unique(expanded)
    unique_labels = unique_labels[unique_labels > 0]
    new_areas = {}
    for lbl in unique_labels:
        new_areas[int(lbl)] = int(np.sum(expanded == lbl))

    return new_areas


def main():
    parser = argparse.ArgumentParser(
        description='Expand nuclei-only masks to match PM+nuclei cell body size'
    )
    parser.add_argument('--nuclei-dir', required=True,
                        help='Run directory for nuclei-only segmentation')
    parser.add_argument('--pm-dir', required=True,
                        help='Run directory for PM+nuclei segmentation')
    parser.add_argument('--dilation-px', type=int, default=None,
                        help='Override auto-computed dilation radius (pixels)')
    parser.add_argument('--pixel-size-um', type=float, required=True,
                        help='Pixel size in micrometers (from CZI metadata)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only compute stats, do not modify mask files')
    args = parser.parse_args()

    nuclei_dir = Path(args.nuclei_dir)
    pm_dir = Path(args.pm_dir)

    # --- Load detections ---
    print(f"Loading nuclei-only detections from {nuclei_dir}...")
    nuc_dets = load_detections(nuclei_dir)
    print(f"  {len(nuc_dets)} detections")

    print(f"Loading PM+nuclei detections from {pm_dir}...")
    pm_dets = load_detections(pm_dir)
    print(f"  {len(pm_dets)} detections")

    # --- Compare area distributions ---
    nuc_areas = get_area_distribution(nuc_dets)
    pm_areas = get_area_distribution(pm_dets)

    if len(nuc_areas) == 0 or len(pm_areas) == 0:
        print("ERROR: One or both runs have no valid areas")
        sys.exit(1)

    dilation_px, stats = compute_dilation_radius(nuc_areas, pm_areas, args.pixel_size_um)

    if args.dilation_px is not None:
        dilation_px = args.dilation_px
        stats['dilation_px_override'] = dilation_px

    print("\n=== Area Distribution Comparison ===")
    print(f"Nuclei-only:  {stats['nuclei_count']:,} cells, "
          f"median area = {stats['nuclei_median_area_px']:.0f} px "
          f"({stats['nuclei_median_area_um2']:.1f} um²)")
    print(f"  IQR: {stats['nuclei_p25_area']:.0f} - {stats['nuclei_p75_area']:.0f} px")
    print(f"PM+nuclei:    {stats['pm_count']:,} cells, "
          f"median area = {stats['pm_median_area_px']:.0f} px "
          f"({stats['pm_median_area_um2']:.1f} um²)")
    print(f"  IQR: {stats['pm_p25_area']:.0f} - {stats['pm_p75_area']:.0f} px")
    print(f"Area ratio (PM/nuc): {stats['area_ratio']:.2f}x")
    print(f"Radius: nuclei={stats['nuclei_median_radius_px']:.1f} px, "
          f"PM={stats['pm_median_radius_px']:.1f} px")
    print(f"Dilation: {dilation_px} px ({dilation_px * args.pixel_size_um:.2f} um)")

    # Save stats
    stats_path = nuclei_dir / 'expansion_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f"\nSaved comparison stats to {stats_path}")

    if args.dry_run:
        print("\n[DRY RUN] No mask files modified.")
        return

    # --- Dilate masks in nuclei-only HDF5 files ---
    tiles_dir = nuclei_dir / 'tiles'
    if not tiles_dir.exists():
        print(f"ERROR: No tiles directory at {tiles_dir}")
        sys.exit(1)

    tile_dirs = sorted(tiles_dir.glob('tile_*'))
    print(f"\n=== Expanding masks in {len(tile_dirs)} tiles (dilation={dilation_px} px) ===")

    total_labels_expanded = 0
    for i, td in enumerate(tile_dirs):
        mask_path = td / 'islet_masks.h5'
        if not mask_path.exists():
            continue

        new_areas = expand_tile_masks(mask_path, dilation_px)
        if new_areas:
            total_labels_expanded += len(new_areas)

        if (i + 1) % 5 == 0 or i == len(tile_dirs) - 1:
            print(f"  Processed {i+1}/{len(tile_dirs)} tiles "
                  f"({total_labels_expanded} labels expanded)")

    print(f"\nExpanded {total_labels_expanded} cell masks across {len(tile_dirs)} tiles")

    # --- Update detection JSON with new areas ---
    print("\nUpdating detection areas in JSON...")
    # Rebuild tile→label→area mapping from expanded masks
    tile_area_map = {}
    for td in tile_dirs:
        mask_path = td / 'islet_masks.h5'
        if not mask_path.exists():
            continue
        # Parse tile coords from dir name
        parts = td.name.split('_')  # tile_X_Y
        if len(parts) >= 3:
            tile_key = (int(parts[1]), int(parts[2]))
        else:
            continue

        import h5py
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        with h5py.File(mask_path, 'r') as f:
            labels = f['masks'][:]
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        areas = {}
        for lbl in unique_labels:
            areas[int(lbl)] = int(np.sum(labels == lbl))
        tile_area_map[tile_key] = areas

    n_updated = 0
    pixel_area_um2 = args.pixel_size_um ** 2
    for det in nuc_dets:
        to = det.get('tile_origin')
        ml = det.get('tile_mask_label', det.get('mask_label'))
        if to is None or ml is None:
            continue
        tile_key = (int(to[0]), int(to[1]))
        areas = tile_area_map.get(tile_key, {})
        new_area = areas.get(int(ml))
        if new_area is not None:
            feat = det.setdefault('features', {})
            feat['area_pre_expansion'] = feat.get('area', 0)
            feat['area'] = new_area
            feat['area_um2_pre_expansion'] = feat.get('area_um2', 0)
            feat['area_um2'] = new_area * pixel_area_um2
            n_updated += 1

    print(f"  Updated area for {n_updated}/{len(nuc_dets)} detections")

    # Save updated detections (backup original first)
    det_path = nuclei_dir / 'islet_detections.json'
    backup_path = nuclei_dir / 'islet_detections_pre_expansion.json'
    if det_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(det_path, backup_path)
        print(f"  Backed up original to {backup_path.name}")

    with open(det_path, 'w') as f:
        json.dump(nuc_dets, f)
    print(f"  Saved expanded detections to {det_path.name}")

    print(f"\n=== Done. Re-run analyze_islets.py on {nuclei_dir} to get updated marker classification ===")


if __name__ == '__main__':
    main()
