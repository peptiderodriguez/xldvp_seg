#!/usr/bin/env python3
"""
Cluster detections for LMD well assignment.

Two-stage greedy spatial clustering with area constraint.
Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Usage:
    python scripts/cluster_detections.py \\
        --detections detections.json \\
        --pixel-size 0.1725 \\
        --area-min 375 --area-max 425 \\
        --dist-round1 500 --dist-round2 1000 \\
        --min-score 0.5 \\
        --output clusters.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add repo root to path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from segmentation.lmd.clustering import two_stage_clustering


def main():
    parser = argparse.ArgumentParser(
        description='Cluster detections for LMD well assignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
    python scripts/cluster_detections.py \\
        --detections nmj_detections.json \\
        --output nmj_clusters.json
''',
    )
    parser.add_argument('--detections', type=str, required=True,
                        help='Path to detections JSON')
    parser.add_argument('--pixel-size', type=float, required=True,
                        help='Pixel size in um/px (must match CZI metadata)')
    parser.add_argument('--area-min', type=float, default=375.0,
                        help='Cluster area lower bound in um2 (default: 375)')
    parser.add_argument('--area-max', type=float, default=425.0,
                        help='Cluster area upper bound in um2 (default: 425)')
    parser.add_argument('--dist-round1', type=float, default=500.0,
                        help='Distance threshold for round 1 in um (default: 500)')
    parser.add_argument('--dist-round2', type=float, default=1000.0,
                        help='Distance threshold for round 2 in um (default: 1000)')
    parser.add_argument('--min-score', type=float, default=0.5,
                        help='Minimum rf_prediction score (default: 0.5)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for clusters JSON')

    args = parser.parse_args()

    # Load detections
    print(f"Loading detections from: {args.detections}")
    with open(args.detections, 'r') as f:
        detections = json.load(f)
    print(f"  Total detections: {len(detections)}")

    # Run clustering
    print(f"\nClustering (area target: {args.area_min}-{args.area_max} um2)...")
    result = two_stage_clustering(
        detections,
        pixel_size=args.pixel_size,
        area_min=args.area_min,
        area_max=args.area_max,
        dist_round1=args.dist_round1,
        dist_round2=args.dist_round2,
        min_score=args.min_score,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f)

    # Summary
    s = result['summary']
    print(f"\nResults:")
    print(f"  Filtered detections (score >= {args.min_score}): {s['n_total_filtered']}")
    print(f"  Clusters: {s['n_clusters']}")
    print(f"  Detections in clusters: {s['n_detections_in_clusters']}")
    print(f"  Singles (outliers): {s['n_singles']}")

    if result['main_clusters']:
        areas = [c['total_area_um2'] for c in result['main_clusters']]
        sizes = [c['n'] for c in result['main_clusters']]
        print(f"\n  Cluster areas: {min(areas):.1f} - {max(areas):.1f} um2 "
              f"(mean {np.mean(areas):.1f}, median {np.median(areas):.1f})")
        print(f"  Cluster sizes: {min(sizes)} - {max(sizes)} detections "
              f"(mean {np.mean(sizes):.1f})")

    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
