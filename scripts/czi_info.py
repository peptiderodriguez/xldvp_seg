#!/usr/bin/env python3
"""
Print CZI file metadata: channels, wavelengths, dimensions, pixel size.

Always run this before launching a pipeline to confirm channel indices.

Usage:
    python scripts/czi_info.py /path/to/slide.czi
    python scripts/czi_info.py /path/to/slide.czi --json
"""
import sys
import json
import argparse
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.io.czi_loader import get_czi_metadata


def main():
    parser = argparse.ArgumentParser(description="Print CZI file metadata")
    parser.add_argument("czi_path", help="Path to CZI file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--scene", type=int, default=None,
                        help="Scene index to show (default: show all scenes)")
    args = parser.parse_args()

    czi_path = Path(args.czi_path)
    if not czi_path.exists():
        print(f"ERROR: File not found: {czi_path}", file=sys.stderr)
        sys.exit(1)

    # Get metadata for scene count (use scene=0 for channel info, it's shared)
    meta = get_czi_metadata(czi_path, scene=args.scene if args.scene is not None else 0)
    n_scenes = meta.get('n_scenes', 1)

    if args.json:
        # Include per-scene mosaic sizes in JSON output
        if n_scenes > 1 and args.scene is None:
            meta['scenes'] = []
            for s in range(n_scenes):
                s_meta = get_czi_metadata(czi_path, scene=s)
                meta['scenes'].append({
                    'index': s,
                    'mosaic_size': s_meta['mosaic_size'],
                })
        print(json.dumps(meta, default=str))
        return

    print(f"\n{'=' * 60}")
    print(f"CZI: {czi_path.name}")
    print(f"{'=' * 60}")

    print(f"Scenes:     {n_scenes}")

    # Show per-scene mosaic dimensions
    if args.scene is not None:
        # Single scene
        if meta['mosaic_size']:
            w, h = meta['mosaic_size']
            print(f"Mosaic:     {w:,} x {h:,} px  (scene {args.scene})")
    elif n_scenes > 1:
        for s in range(n_scenes):
            s_meta = get_czi_metadata(czi_path, scene=s)
            if s_meta['mosaic_size']:
                w, h = s_meta['mosaic_size']
                print(f"  Scene {s}: {w:,} x {h:,} px")
    else:
        if meta['mosaic_size']:
            w, h = meta['mosaic_size']
            print(f"Mosaic:     {w:,} x {h:,} px")

    print(f"Pixel size: {meta['pixel_size_um']:.4f} µm/px")
    print(f"Channels:   {meta['n_channels']}")
    print(f"{'-' * 60}")

    for ch in meta['channels']:
        ex = f"{ch['excitation_nm']:.0f}" if ch['excitation_nm'] else "?"
        em = f"{ch['emission_nm']:.0f}" if ch['emission_nm'] else "?"
        fluor = ch['fluorophore'] if ch['fluorophore'] != 'N/A' else ''
        dye = ch.get('dye', '')
        label = fluor or dye or ch['name']

        print(f"  [{ch['index']}] {ch['name']:<20s}  Ex {ex:>4s} → Em {em:>4s} nm  {label}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
