#!/usr/bin/env python3
"""
Spatial cell analysis: RF embedding, morphological UMAP, and network analysis.

Three independent modes activated by flags. Can run one, two, or all three.

Modes:
  --rf-embedding     RF leaf-node co-occurrence UMAP (supervised embedding)
  --morph-umap       Morphological feature UMAP with small-multiple coloring
  --spatial-network  Delaunay-based cell adjacency graph + community detection

Usage:
    python scripts/spatial_cell_analysis.py \
        --detections detections.json \
        --output-dir analysis_output/ \
        --classifier rf_classifier.pkl \
        --rf-embedding --morph-umap --spatial-network

Dependencies: umap-learn, networkx, scipy, matplotlib, scikit-learn
"""

import argparse
import sys
from pathlib import Path

from xldvp_seg.analysis.spatial_network import (
    parse_marker_filter,  # noqa: F401 — re-exported for backward compat
    run_morph_umap,
    run_rf_embedding,
    run_spatial_network,
)
from xldvp_seg.utils.detection_utils import load_detections
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def save_enriched_detections(detections, output_path):
    """Save detections with enriched fields to JSON (no indent, timestamped)."""
    from xldvp_seg.utils.timestamps import save_with_timestamp

    save_with_timestamp(output_path, detections, fmt="json")
    logger.info(f"Saved enriched detections ({len(detections):,} entries)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Spatial cell analysis: RF embedding, morph UMAP, network analysis"
    )
    parser.add_argument("--detections", required=True, help="Path to detections JSON")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for plots, CSV, and enriched JSON"
    )
    parser.add_argument(
        "--classifier",
        default=None,
        help="Path to RF classifier (.pkl/.joblib) — required for --rf-embedding",
    )
    parser.add_argument(
        "--rf-embedding", action="store_true", help="Mode 1: RF leaf-node co-occurrence UMAP"
    )
    parser.add_argument(
        "--morph-umap",
        action="store_true",
        help="Mode 2: Morphological feature UMAP (small multiples)",
    )
    parser.add_argument(
        "--spatial-network",
        action="store_true",
        help="Mode 3: Delaunay-based spatial network analysis",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Filter detections by rf_prediction >= threshold",
    )
    parser.add_argument(
        "--marker-filter", default=None, help='Filter for spatial network, e.g. "ch0_mean>100"'
    )
    parser.add_argument(
        "--max-edge-distance",
        type=float,
        default=50.0,
        help="Max edge distance in um for network (default: 50)",
    )
    parser.add_argument(
        "--min-component-cells",
        type=int,
        default=3,
        help="Min cells per connected component (default: 3)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        required=True,
        help="Pixel size in um/px. Must match the CZI metadata of the source image.",
    )
    parser.add_argument(
        "--max-umap-samples",
        type=int,
        default=50000,
        help="Subsample limit for UMAP (default: 50000)",
    )
    parser.add_argument(
        "--umap-neighbors", type=int, default=30, help="UMAP n_neighbors (default: 30)"
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)"
    )
    args = parser.parse_args()
    setup_logging(level="INFO")

    if not (args.rf_embedding or args.morph_umap or args.spatial_network):
        parser.error("At least one mode required: --rf-embedding, --morph-umap, --spatial-network")

    if args.rf_embedding and not args.classifier:
        parser.error("--classifier is required when using --rf-embedding")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detections (shared across all modes)
    detections = load_detections(args.detections, args.score_threshold)

    if not detections:
        logger.error("No detections after filtering — nothing to analyze")
        sys.exit(1)

    # Run selected modes
    if args.rf_embedding:
        logger.info("=" * 60)
        logger.info("Mode 1: RF Embedding")
        logger.info("=" * 60)
        detections = run_rf_embedding(
            detections,
            args.classifier,
            output_dir,
            max_umap_samples=args.max_umap_samples,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

    if args.morph_umap:
        logger.info("=" * 60)
        logger.info("Mode 2: Morphological UMAP")
        logger.info("=" * 60)
        detections = run_morph_umap(
            detections,
            output_dir,
            max_umap_samples=args.max_umap_samples,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

    if args.spatial_network:
        logger.info("=" * 60)
        logger.info("Mode 3: Spatial Network Analysis")
        logger.info("=" * 60)
        detections = run_spatial_network(
            detections,
            output_dir,
            pixel_size=args.pixel_size,
            marker_filter=args.marker_filter,
            max_edge_distance=args.max_edge_distance,
            min_component_cells=args.min_component_cells,
        )

    # Save enriched detections
    enriched_path = output_dir / "detections_enriched.json"
    save_enriched_detections(detections, enriched_path)

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
