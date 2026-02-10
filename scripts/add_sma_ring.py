#!/usr/bin/env python3
"""
Add SMA ring (3rd contour) to existing vessel detections.

Post-processing script for runs that completed before SMA ring detection
was integrated into the main pipeline. Reads vessel_detections_multiscale.json,
runs full-resolution contour refinement (CD31 + SMA dilation + optional spline
smoothing), and regenerates crops/HTML/histograms.

All detection logic lives in sam2_multiscale_vessels.py — this script only
orchestrates the calls.

Usage:
    # Full refinement (default: adaptive SMA + spline smoothing)
    python scripts/add_sma_ring.py

    # Tune SMA sensitivity
    python scripts/add_sma_ring.py --min-sma-intensity 20

    # Skip spline, uniform SMA mode
    python scripts/add_sma_ring.py --no-spline --mode uniform

    # Fast iteration (skip crop regeneration)
    python scripts/add_sma_ring.py --skip-crops
"""

import json
import os
import sys
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.io.czi_loader import CZILoader
from scripts.sam2_multiscale_vessels import (
    DownsampledChannelCache, refine_vessel_contours_fullres,
    save_vessel_crop_fullres, generate_html, generate_histograms,
    CZI_PATH, OUTPUT_DIR, NUCLEAR, CD31, SMA, PM, BASE_SCALE,
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add SMA ring (3rd contour) to vessel detections")
    parser.add_argument('--input', type=str,
                        default=os.path.join(OUTPUT_DIR, 'vessel_detections_multiscale.json'),
                        help='Input JSON path (final or checkpoint)')
    parser.add_argument('--drop-ratio', type=float, default=0.9,
                        help='Drop ratio for CD31 and SMA dilation stopping (default: 0.9)')
    parser.add_argument('--min-sma-intensity', type=float, default=30,
                        help='Min SMA wall intensity to attempt dilation (default: 30)')
    parser.add_argument('--skip-crops', action='store_true',
                        help='Skip crop regeneration (faster, for tuning)')
    parser.add_argument('--dilation-mode', choices=['adaptive', 'uniform'], default='adaptive',
                        help='Dilation mode for CD31 and SMA: adaptive (irregular contours) '
                             'or uniform (default: adaptive)')
    parser.add_argument('--spline', action='store_true',
                        help='Enable spline smoothing (off by default)')
    parser.add_argument('--spline-smoothing', type=float, default=3.0,
                        help='Spline smoothing factor (default: 3.0). Higher = smoother.')
    args = parser.parse_args()

    # Load detections (supports both final and checkpoint format)
    print(f"Loading detections from {args.input}...")
    with open(args.input) as f:
        data = json.load(f)

    # Handle both formats: {'vessels': [...]} or {'results': {'vessels': [...]}}
    if 'vessels' in data:
        vessels = data['vessels']
    elif 'results' in data and 'vessels' in data['results']:
        vessels = data['results']['vessels']
    else:
        print("ERROR: Could not find vessels in JSON. Expected 'vessels' or 'results.vessels' key.")
        sys.exit(1)

    print(f"  {len(vessels)} vessels loaded")

    # Load CZI (needed for full-res ROI reads)
    print("Opening CZI...")
    loader = CZILoader(CZI_PATH)
    channel_cache = DownsampledChannelCache(loader, [NUCLEAR, CD31, SMA, PM], BASE_SCALE)

    # Run full-resolution refinement on every vessel
    # refine_vessel_contours_fullres() handles:
    #   - Reading full-res ROI from CZI
    #   - CD31 dilation (outer contour)
    #   - SMA dilation (3rd ring, adaptive or uniform)
    #   - Spline smoothing (optional)
    #   - Recomputing all measurements at native resolution
    print(f"\nRefining contours at full CZI resolution...")
    print(f"  Dilation mode: {args.dilation_mode} (CD31 + SMA)")
    print(f"  SMA min_wall_intensity: {args.min_sma_intensity}")
    print(f"  Spline: {'on' if args.spline else 'off'} (factor={args.spline_smoothing})")

    n_refined = 0
    n_failed = 0
    n_sma_positive = 0

    for v in tqdm(vessels, desc="Refining contours"):
        ok = refine_vessel_contours_fullres(
            v, channel_cache,
            spline=args.spline,
            spline_smoothing=args.spline_smoothing,
            cd31_drop_ratio=args.drop_ratio,
            cd31_mode=args.dilation_mode,
            sma_drop_ratio=args.drop_ratio,
            sma_min_wall_intensity=args.min_sma_intensity,
            sma_mode=args.dilation_mode,
        )
        if ok:
            n_refined += 1
            if v.get('has_sma_ring'):
                n_sma_positive += 1
        else:
            n_failed += 1

    print(f"\nDone! Refined {n_refined} vessels, {n_failed} failed")
    print(f"  SMA+ (has ring): {n_sma_positive} ({100*n_sma_positive/max(n_refined,1):.1f}%)")
    print(f"  SMA- (no ring):  {n_refined - n_sma_positive} ({100*(n_refined-n_sma_positive)/max(n_refined,1):.1f}%)")

    # Regenerate crops at full resolution with refined contours
    if not args.skip_crops:
        print("\nRegenerating crops at full resolution...")
        os.makedirs(os.path.join(OUTPUT_DIR, 'crops'), exist_ok=True)
        n_crops = 0
        n_crop_fail = 0
        for v in tqdm(vessels, desc="Full-res crops"):
            crop_path = save_vessel_crop_fullres(v, channel_cache)
            if crop_path:
                n_crops += 1
            else:
                n_crop_fail += 1
        print(f"  Generated {n_crops} full-res crops, {n_crop_fail} failed")

    # Save updated JSON
    print(f"\nSaving updated JSON to {args.input}...")
    with open(args.input, 'w') as f:
        json.dump(data, f, indent=2)

    # Regenerate HTML and histograms
    print("Regenerating HTML...")
    generate_html(vessels)
    print("Regenerating histograms...")
    generate_histograms(vessels, OUTPUT_DIR)

    channel_cache.release()
    loader.close()
    print("\nAll done!")


if __name__ == '__main__':
    main()
