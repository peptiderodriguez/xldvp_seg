#!/usr/bin/env python3
"""Apply morphological quality filters to detections without annotation/classifier.

For clean slides where Cellpose detection quality is high, this replaces
the annotation → RF classifier step with simple heuristic filters:
  - Area range (default: 50-2000 µm²)
  - Solidity minimum (default: 0.85)
  - Optional: eccentricity max, aspect_ratio max

Sets rf_prediction=1.0 for passing detections, 0.0 for rejected.
Output is compatible with all downstream tools (scoring, sampling, LMD).

Usage:
    # Default filters (50-2000 µm², solidity > 0.85)
    python scripts/quality_filter_detections.py \
        --detections cell_detections.json \
        --output cell_detections_filtered.json

    # Custom range
    python scripts/quality_filter_detections.py \
        --detections cell_detections.json \
        --output cell_detections_filtered.json \
        --min-area-um2 100 --max-area-um2 1500 --min-solidity 0.90

    # Relaxed (keep more, filter less)
    python scripts/quality_filter_detections.py \
        --detections cell_detections.json \
        --output cell_detections_filtered.json \
        --min-area-um2 30 --max-area-um2 5000 --min-solidity 0.75
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Quality-filter detections using morphological heuristics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--detections', required=True, help='Input detections JSON')
    parser.add_argument('--output', required=True, help='Output filtered detections JSON')
    parser.add_argument('--min-area-um2', type=float, default=50,
                        help='Minimum cell area in µm² (default: 50)')
    parser.add_argument('--max-area-um2', type=float, default=2000,
                        help='Maximum cell area in µm² (default: 2000)')
    parser.add_argument('--min-solidity', type=float, default=0.85,
                        help='Minimum solidity (default: 0.85)')
    parser.add_argument('--max-eccentricity', type=float, default=None,
                        help='Maximum eccentricity (default: no filter)')
    parser.add_argument('--max-aspect-ratio', type=float, default=None,
                        help='Maximum aspect ratio (default: no filter)')
    parser.add_argument('--pixel-size', type=float, default=None,
                        help='Pixel size in µm (auto-detected from features if omitted)')
    args = parser.parse_args()

    setup_logging(level="INFO")

    logger.info(f"Loading {args.detections}...")
    detections = fast_json_load(args.detections)
    logger.info(f"  {len(detections)} detections loaded")

    # Auto-detect pixel size
    pixel_size = args.pixel_size
    if pixel_size is None:
        for det in detections[:100]:
            feat = det.get('features', {})
            ps = feat.get('pixel_size_um') or det.get('pixel_size_um')
            if ps:
                pixel_size = float(ps)
                break
            # Derive from area vs area_um2
            if feat.get('area') and feat.get('area_um2') and feat['area'] > 0:
                import math
                pixel_size = math.sqrt(feat['area_um2'] / feat['area'])
                break
    if pixel_size is None:
        pixel_size = 0.1725
        logger.warning(f"Could not detect pixel size, using default {pixel_size}")

    n_pass = 0
    n_fail = 0
    for det in detections:
        feat = det.get('features', {})

        # Area filter
        area_um2 = feat.get('area_um2')
        if area_um2 is None:
            area_px = feat.get('area', 0)
            area_um2 = area_px * pixel_size ** 2

        # Solidity filter
        solidity = feat.get('solidity', 1.0)

        # Apply filters
        passed = True
        if area_um2 < args.min_area_um2 or area_um2 > args.max_area_um2:
            passed = False
        if solidity < args.min_solidity:
            passed = False
        if args.max_eccentricity is not None:
            if feat.get('eccentricity', 0) > args.max_eccentricity:
                passed = False
        if args.max_aspect_ratio is not None:
            if feat.get('aspect_ratio', 1) > args.max_aspect_ratio:
                passed = False

        det['rf_prediction'] = 1.0 if passed else 0.0
        det['quality_filter'] = 'pass' if passed else 'fail'
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    logger.info(f"  Pass: {n_pass} ({100*n_pass/len(detections):.1f}%)")
    logger.info(f"  Fail: {n_fail} ({100*n_fail/len(detections):.1f}%)")
    logger.info(f"  Filters: area {args.min_area_um2}-{args.max_area_um2} µm², "
                f"solidity >= {args.min_solidity}")

    atomic_json_dump(detections, args.output)
    logger.info(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
