#!/usr/bin/env python3
"""
Generate crop-image annotation HTML for Msln+ cells, stratified by tier.

Loads bg-subtracted detections, samples up to 600 per tier, extracts 200x200
crops directly from the CZI (ch2=R, ch1=G, ch0=B) without loading full
channels into RAM, and generates paginated HTML using HTMLPageGenerator
with Yes/No annotation buttons.

Output: one combined HTML set with tier shown in stats, sorted by SNR (desc)
within each tier.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import h5py
import hdf5plugin  # noqa: F401 — enables LZ4 codec for h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from aicspylibczi import CziFile

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.io.html_generator import HTMLPageGenerator
from segmentation.io.html_export import (
    percentile_normalize, image_to_base64, draw_mask_contour,
)

# ---------------------------------------------------------------------------
# Default paths (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_DETECTIONS_PATH = (
    "/fs/pool/pool-mann-edwin/psilo_output/tp_full/"
    "20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct/"
    "detections_msln_density_normalized.json"
)
DEFAULT_CZI_PATH = (
    "/fs/pool/pool-mann-axioscan/01_Users/EdRo_axioscan/MPIP_psilo/"
    "20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1.czi"
)
DEFAULT_OUTPUT_DIR = "/fs/pool/pool-mann-edwin/brain_fish_output/msln_annotation_crops"
DEFAULT_TILES_DIR = (
    "/fs/pool/pool-mann-edwin/psilo_output/tp_full/"
    "20251114_Pdgfra546_Msln750_PM647_nuc488-EDFvar-1-stitch-1_20260223_094916_100pct/"
    "tiles"
)

# Display: Msln750 (ch2) as white, nuc488 (ch0) as blue, PM647 (ch1) hidden
CH_MSLN = 2   # Msln750 → white (all RGB)
CH_NUC = 0    # nuc488  → blue (B channel only)

TILES_DIR = None  # Set from CLI in main()

CROP_SIZE = 200          # pixels, centered on cell
SAMPLES_PER_TIER = None  # None = all cells (no sampling)
SAMPLES_PER_PAGE = 300
SEED = 42

# Cache open HDF5 files to avoid re-opening for cells in the same tile
_h5_cache = {}


def get_cell_mask_crop(tile_origin, center, mask_label, crop_size):
    """Load the cell mask from HDF5 and return a crop around the cell.

    Args:
        tile_origin: [tx, ty] pixel coords of tile top-left
        center: [cx, cy] local pixel coords within tile
        mask_label: integer label of this cell in the tile's mask image
        crop_size: size of the crop in pixels

    Returns:
        Binary mask (crop_size, crop_size) uint8, or None if mask unavailable
    """
    tx, ty = int(tile_origin[0]), int(tile_origin[1])
    tile_dir = TILES_DIR / f"tile_{tx}_{ty}"
    mask_path = tile_dir / "tissue_pattern_masks.h5"

    if not mask_path.exists():
        return None

    # Open (cached) HDF5 file
    key = str(mask_path)
    if key not in _h5_cache:
        _h5_cache[key] = h5py.File(mask_path, 'r')
    h5f = _h5_cache[key]

    if 'masks' not in h5f:
        return None

    label_img = h5f['masks']  # (H, W) uint32, lazy access
    h, w = label_img.shape

    cx, cy = int(round(center[0])), int(round(center[1]))
    half = crop_size // 2

    # Compute crop bounds within tile, clamped
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    if y1 <= y0 or x1 <= x0:
        return None

    # Read only the crop region from HDF5 (efficient with LZ4 chunks)
    label_crop = label_img[y0:y1, x0:x1]
    binary = (label_crop == mask_label).astype(np.uint8)

    # Pad to full crop_size if needed (edge cells)
    if binary.shape != (crop_size, crop_size):
        padded = np.zeros((crop_size, crop_size), dtype=np.uint8)
        # Offset within padded array
        py = (half - cy + y0)
        px = (half - cx + x0)
        padded[py:py + binary.shape[0], px:px + binary.shape[1]] = binary
        binary = padded

    return binary


def read_crop_from_czi(reader, global_cx, global_cy, crop_size):
    """Read Msln (white) + nuc (blue) crop from CZI.

    Msln750 (ch2) → white (all RGB channels equally).
    nuc488  (ch0) → blue  (B channel only).
    PM647   (ch1) → not shown.

    Returns:
        Percentile-normalized uint8 RGB array (crop_size, crop_size, 3), or None
    """
    half = crop_size // 2
    rx = int(global_cx) - half
    ry = int(global_cy) - half

    def _read_ch(ch_idx):
        try:
            tile = reader.read_mosaic(
                region=(rx, ry, crop_size, crop_size),
                scale_factor=1,
                C=ch_idx,
            )
            tile = np.squeeze(tile)
            return tile if tile.ndim == 2 else None
        except Exception:
            return None

    msln_raw = _read_ch(CH_MSLN)
    nuc_raw = _read_ch(CH_NUC)
    if msln_raw is None or nuc_raw is None:
        return None
    if msln_raw.size == 0 or nuc_raw.size == 0:
        return None

    # Percentile-normalize each channel independently (non-zero pixels)
    def _pnorm(ch):
        valid = ch[ch > 0]
        if valid.size == 0:
            return np.zeros_like(ch, dtype=np.uint8)
        lo = np.percentile(valid, 1)
        hi = np.percentile(valid, 99.5)
        if hi <= lo:
            return np.zeros_like(ch, dtype=np.uint8)
        out = (ch.astype(np.float32) - lo) / (hi - lo) * 255
        return np.clip(out, 0, 255).astype(np.uint8)

    msln = _pnorm(msln_raw)
    nuc = _pnorm(nuc_raw)

    # Compose: Msln=white (R=G=B=msln), nuc=blue (B+=nuc)
    r = msln.copy()
    g = msln.copy()
    b = np.clip(msln.astype(np.uint16) + nuc.astype(np.uint16), 0, 255).astype(np.uint8)

    crop_norm = np.stack([r, g, b], axis=-1)
    return crop_norm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate crop-image annotation HTML for Msln+ cells, stratified by tier."
    )
    parser.add_argument('--detections', type=str, default=DEFAULT_DETECTIONS_PATH,
                        help='Path to detections JSON')
    parser.add_argument('--czi-path', type=str, default=DEFAULT_CZI_PATH,
                        help='Path to CZI file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for HTML crops')
    parser.add_argument('--tiles-dir', type=str, default=DEFAULT_TILES_DIR,
                        help='Path to tiles directory (for mask HDF5 files)')
    return parser.parse_args()


def main():
    args = parse_args()
    DETECTIONS_PATH = Path(args.detections)
    CZI_PATH = Path(args.czi_path)
    OUTPUT_DIR = Path(args.output_dir)

    global TILES_DIR
    TILES_DIR = Path(args.tiles_dir)

    # 1. Load detections
    print(f"Loading detections from {DETECTIONS_PATH}...")
    with open(DETECTIONS_PATH) as f:
        all_dets = json.load(f)
    print(f"  Loaded {len(all_dets):,} detections")

    # 2. Group by tier
    tier_groups = defaultdict(list)
    for det in all_dets:
        tier = det.get('features', {}).get('msln_tier', 'unknown')
        tier_groups[tier].append(det)

    print("Tier breakdown:")
    for tier, dets in sorted(tier_groups.items()):
        print(f"  {tier}: {len(dets):,}")

    # 3. Sample (or take all) from each tier
    rng = np.random.default_rng(SEED)
    selected = []
    for tier in sorted(tier_groups.keys()):
        dets = tier_groups[tier]
        if SAMPLES_PER_TIER is not None and len(dets) > SAMPLES_PER_TIER:
            indices = rng.choice(len(dets), SAMPLES_PER_TIER, replace=False)
            sampled = [dets[i] for i in indices]
        else:
            sampled = dets
        print(f"  Selected {len(sampled)} from {tier}")
        selected.extend(sampled)

    # Sort by msln_density descending (highest density first)
    selected.sort(
        key=lambda d: -(d.get('features', {}).get('msln_density', 0) or 0)
    )
    print(f"Total selected: {len(selected):,}")

    # 4. Open CZI reader (on-demand, no RAM loading)
    print(f"\nOpening CZI: {CZI_PATH.name}")
    reader = CziFile(str(CZI_PATH))
    bbox = reader.get_mosaic_scene_bounding_box(index=0)
    print(f"  Mosaic bbox: x={bbox.x}, y={bbox.y}, w={bbox.w:,}, h={bbox.h:,}")

    # Get pixel size from metadata
    pixel_size_um = 0.22  # default
    try:
        metadata = reader.meta
        scaling = metadata.find('.//Scaling/Items/Distance[@Id="X"]/Value')
        if scaling is not None:
            pixel_size_um = float(scaling.text) * 1e6
    except Exception:
        pass
    print(f"  Pixel size: {pixel_size_um:.4f} um")

    # 5. Generate crops — read each crop directly from CZI (no RAM)
    print(f"\nGenerating {len(selected):,} crops ({CROP_SIZE}x{CROP_SIZE} px)...")
    print("  Reading each crop directly from CZI (low memory mode)...")
    samples = []
    skipped = 0
    for det in tqdm(selected, desc="Crops"):
        # Global pixel center = tile_origin + local center
        tile_origin = det.get('tile_origin', [0, 0])
        center = det.get('center', [0, 0])
        global_cx = tile_origin[0] + center[0]
        global_cy = tile_origin[1] + center[1]

        crop_norm = read_crop_from_czi(
            reader, global_cx, global_cy, CROP_SIZE
        )
        if crop_norm is None:
            skipped += 1
            continue

        # Overlay dashed B/W cell contour from mask
        mask_label = det.get('tile_mask_label') or det.get('mask_label')
        if mask_label is not None:
            cell_mask = get_cell_mask_crop(
                tile_origin, center, int(mask_label), CROP_SIZE
            )
            if cell_mask is not None and cell_mask.any():
                crop_norm = draw_mask_contour(
                    crop_norm, cell_mask, bw_dashed=True, thickness=1
                )

        # Encode as base64 PNG
        pil_img = Image.fromarray(crop_norm)
        img_b64, mime = image_to_base64(pil_img, format='PNG')

        features = det.get('features', {})
        uid = det.get('uid', det.get('id', 'unknown'))

        stats = {
            'tier': features.get('msln_tier', '?'),
            'density': features.get('msln_density', 0),
            'snr': features.get('msln_snr', 0),
            'bg_sub': features.get('msln_bg_sub', 0),
            'area_um2': features.get('area', 0) * (pixel_size_um ** 2),
        }

        samples.append({
            'uid': uid,
            'image': img_b64,
            'mime_type': mime,
            'stats': stats,
        })

    # Close cached HDF5 files
    for h5f in _h5_cache.values():
        h5f.close()
    _h5_cache.clear()

    print(f"Generated {len(samples):,} crop samples (skipped {skipped})")

    # 6. Generate HTML using HTMLPageGenerator
    print(f"\nExporting HTML to {OUTPUT_DIR}...")
    generator = HTMLPageGenerator(
        cell_type='msln',
        experiment_name='msln_annotation',
        storage_strategy='experiment',
        samples_per_page=SAMPLES_PER_PAGE,
        title='Msln+ Annotation',
    )

    # Register custom formatters for our stats
    generator.register_formatter('tier', lambda v: f"<b>{v}</b>")
    generator.register_formatter('density', lambda v: f"Density: {v:.1f}")
    generator.register_formatter('snr', lambda v: f"SNR: {v:.1f}x")
    generator.register_formatter('bg_sub', lambda v: f"BG-sub: {v:.0f}")
    generator.register_formatter('area_um2', lambda v: f"{v:.1f} &micro;m&sup2;")

    tier_counts = defaultdict(int)
    for s in samples:
        tier_counts[s['stats']['tier']] += 1
    extra_stats = {f"{tier}": f"{count}" for tier, count in sorted(tier_counts.items())}
    extra_stats['Total'] = str(len(samples))

    total_samples, total_pages = generator.export_to_html(
        samples,
        OUTPUT_DIR,
        page_prefix='msln_page',
        subtitle=f"Msln+ cells from WM PM ({CZI_PATH.stem})",
        extra_stats=extra_stats,
    )

    # Inject channel color legend into the stats-row bar on annotation pages
    channel_legend_items = (
        '<span style="color:#ffffff;font-weight:bold;font-size:12px">&#9608; Msln750</span>'
        '<span style="color:#6688ff;font-weight:bold;font-size:12px">&#9608; nuc488</span>'
    )
    channel_group = (
        f'<div class="stats-group" style="gap:8px">{channel_legend_items}</div>'
    )
    for html_file in OUTPUT_DIR.glob('*.html'):
        text = html_file.read_text()
        # Insert legend into the stats-row, before the Import button
        text = text.replace(
            '<button class="btn btn-import"',
            f'{channel_group}\n            <button class="btn btn-import"',
        )
        html_file.write_text(text)
    print("  Injected channel color legend into stats bar")

    print(f"\nDone! {total_samples:,} samples across {total_pages} pages")
    print(f"Open: {OUTPUT_DIR / 'index.html'}")


if __name__ == '__main__':
    main()
