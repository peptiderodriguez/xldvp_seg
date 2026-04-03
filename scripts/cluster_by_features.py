#!/usr/bin/env python3
"""
Feature-based clustering of cell detections using UMAP and/or t-SNE + HDBSCAN.

Works with ANY cell type's channel features. Auto-discovers channel features
from detections (keys matching ch\\d+_*) and labels clusters by dominant marker.

For islet detections (backward compat): auto-detects ch2/ch3/ch5 and labels
alpha/beta/delta when --marker-channels is not specified.

Feature groups:
  - "morph":   all morphological features (= shape + color, backward compat)
  - "shape":   pure geometry (area, circularity, solidity, aspect_ratio, etc.)
  - "color":   intensity/color (gray_mean, hue_mean, relative_brightness, etc.)
  - "sam2":    SAM2 embedding features (sam2_0..sam2_255)
  - "channel": per-channel stats (ch0_mean, ch1_std, ch0_ch2_ratio, etc.)
  - "deep":    deep features (resnet_*, dinov2_*)

Auto-labels clusters by dominant marker:
  - For each cluster, z-score of mean expression per marker channel
  - Highest z-score marker name becomes the cluster label
  - Mixed/low -> "other"

Outputs:
  - detections_clustered.json   -- detections + cluster_id, cluster_label, umap_x, umap_y, tsne_x, tsne_y
  - cluster_summary.csv         -- per-cluster stats
  - umap_plot.png               -- UMAP colored by cluster (when --methods umap or both)
  - tsne_plot.png               -- t-SNE colored by cluster (when --methods tsne or both)
  - umap_tsne_plot.png          -- side-by-side UMAP + t-SNE (when --methods both)
  - marker_violin.png           -- marker intensity distributions per cluster
  - spatial.h5ad                -- AnnData format for scanpy
  - spatial.csv                 -- flat CSV with coords + clusters

Usage:
  # Islet (backward compat -- auto-detects ch2/ch3/ch5 as alpha/beta/delta)
  python scripts/cluster_by_features.py \\
      --detections /path/to/islet_detections.json \\
      --output-dir /path/to/clustering_output

  # Tissue pattern with 4 channels, exclude ch3 (bad stain)
  python scripts/cluster_by_features.py \\
      --detections /path/to/tissue_pattern_detections.json \\
      --output-dir /path/to/clustering_output \\
      --marker-channels "msln:2,pm:1" \\
      --exclude-channels "3" \\
      --feature-groups "morph,sam2,channel"

  # UMAP only (original behavior)
  python scripts/cluster_by_features.py \\
      --detections /path/to/detections.json \\
      --output-dir /path/to/output \\
      --methods umap

  # t-SNE only
  python scripts/cluster_by_features.py \\
      --detections /path/to/detections.json \\
      --output-dir /path/to/output \\
      --methods tsne --perplexity 50

Dependencies: umap-learn, hdbscan, anndata, matplotlib, pandas, scikit-learn
"""

import argparse
from pathlib import Path

from xldvp_seg.analysis.cluster_features import (
    discover_marker_channels,
    parse_exclude_channels,
    parse_marker_channels,
    run_clustering,
    run_subclustering,
)
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load, sanitize_for_json


def main():
    parser = argparse.ArgumentParser(
        description="Feature-based clustering of cell detections (any cell type)"
    )
    parser.add_argument(
        "--detections",
        default=None,
        help="Path to detections JSON file (required unless --subcluster-input)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for clustering results"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Minimum rf_prediction score (default: 0.5)"
    )
    parser.add_argument(
        "--marker-channels",
        type=str,
        default=None,
        help='Marker channels for cluster labeling: "name:ch_idx,..." '
        'e.g. "msln:2,pm:1" or "alpha:2,beta:3,delta:5". '
        "If not given, auto-detects from features (islet defaults if ch2/ch3/ch5 present)",
    )
    parser.add_argument(
        "--exclude-channels",
        type=str,
        default=None,
        help='Channel indices to exclude from feature matrix: "3" or "0,3,5"',
    )
    parser.add_argument(
        "--feature-groups",
        type=str,
        default="morph,sam2,channel",
        help="Comma-separated feature groups: morph (=shape+color), "
        'shape, color, sam2, channel, deep (default: "morph,sam2,channel")',
    )
    parser.add_argument(
        "--marker-only",
        action="store_true",
        help="Use only normalized marker channel _mean features "
        "(population p1-p99.5 percentile stretch)",
    )
    parser.add_argument(
        "--spatial-smooth",
        action="store_true",
        default=False,
        help="Apply feature-gated spatial smoothing before dim reduction. "
        "Weights neighbors by spatial proximity AND feature similarity "
        "(cosine in PCA space). Preserves tissue boundaries and rare cell types.",
    )
    parser.add_argument(
        "--smooth-k",
        type=int,
        default=15,
        help="Number of spatial neighbors for smoothing (default: 15)",
    )
    parser.add_argument(
        "--smooth-sim-threshold",
        type=float,
        default=0.5,
        help="Minimum cosine similarity to include a neighbor in smoothing "
        "(default: 0.5, range 0-1). Higher = more conservative.",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=30, help="UMAP n_neighbors (default: 30)"
    )
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist (default: 0.1)")
    parser.add_argument(
        "--min-cluster-size", type=int, default=50, help="HDBSCAN min_cluster_size (default: 50)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (default: None, uses min_cluster_size)",
    )
    parser.add_argument(
        "--subcluster",
        action="store_true",
        help="After main clustering, sub-cluster each parent cluster "
        "by appearance features (morph+sam2)",
    )
    parser.add_argument(
        "--subcluster-input",
        type=str,
        default=None,
        help="Path to pre-clustered detections JSON for standalone "
        "subclustering (skips main clustering)",
    )
    parser.add_argument(
        "--subcluster-features",
        type=str,
        default="shape,sam2",
        help='Feature groups for subclustering (default: "shape,sam2")',
    )
    parser.add_argument(
        "--subcluster-min-size",
        type=int,
        default=50,
        help="HDBSCAN min_cluster_size for sub-clusters (default: 50)",
    )
    parser.add_argument(
        "--gate-channel",
        type=int,
        default=None,
        help="Gate on this channel index before clustering "
        "(e.g., 2 for Msln). Keeps cells above --gate-percentile.",
    )
    parser.add_argument(
        "--gate-percentile",
        type=float,
        default=90,
        help="Percentile threshold for --gate-channel: keep cells above "
        "this percentile of chN_mean (default: 90 = top 10%%)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=["umap", "tsne", "both"],
        help="Dimensionality reduction method(s): umap, tsne, or both (default: both)",
    )
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity (default: 30)")
    parser.add_argument(
        "--tsne-n-iter", type=int, default=1000, help="t-SNE iterations (default: 1000)"
    )
    parser.add_argument(
        "--clustering",
        type=str,
        default="leiden",
        choices=["hdbscan", "leiden"],
        help="Clustering algorithm: hdbscan or leiden (default: leiden)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution parameter (default: 1.0, higher = more clusters)",
    )
    parser.add_argument(
        "--marker-rings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw colored rings around dots based on dominant marker expression "
        "(default: True). Disable with --no-marker-rings.",
    )
    parser.add_argument(
        "--trajectory",
        action="store_true",
        help="Run trajectory analysis: diffusion map, pseudotime, PAGA, " "force-directed layout",
    )
    parser.add_argument(
        "--root-cluster",
        default=None,
        help="Cluster label to use as pseudotime root " "(default: auto-detect largest)",
    )
    args = parser.parse_args()

    if args.subcluster_input:
        # Standalone subclustering on pre-clustered detections
        pass  # --detections not needed
    elif not args.detections:
        parser.error("--detections is required (unless using --subcluster-input)")

    if args.subcluster_input:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        marker_channels = parse_marker_channels(args.marker_channels)
        exclude_channels = parse_exclude_channels(args.exclude_channels)

        print(f"Loading pre-clustered detections from {args.subcluster_input}...")
        detections = fast_json_load(str(args.subcluster_input))
        print(f"  {len(detections)} detections")

        if marker_channels is None:
            marker_channels = discover_marker_channels(detections, exclude_channels) or {}
            if marker_channels:
                print(f"  Auto-detected marker channels: {marker_channels}")

        run_subclustering(
            detections,
            output_dir,
            marker_channels,
            exclude_channels,
            subcluster_features=args.subcluster_features,
            subcluster_min_size=args.subcluster_min_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )

        # Save enriched detections
        out_path = output_dir / "detections_subclustered.json"
        atomic_json_dump(sanitize_for_json(detections), str(out_path))
        print(f"\nSaved: {out_path}")
        return

    run_clustering(**vars(args))


if __name__ == "__main__":
    main()
