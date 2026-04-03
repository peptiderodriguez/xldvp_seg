#!/usr/bin/env python3
"""Detect curvilinear (strip/ribbon) spatial patterns in classified detections.

Builds a KD-tree radius graph on marker-positive cells, extracts connected
components, then classifies each component as **strip** or **blob** based on
graph diameter normalized by component size.

Strip components have high ``diameter / sqrt(n_nodes)`` — the graph path
through them is long relative to their size. This works for curved strips
because graph diameter follows the actual path through the component, unlike
PCA which assumes a linear axis.

Each detection is tagged with a ``{prefix}_pattern`` field in features:
  - ``strip``     — member of a component classified as curvilinear
  - ``cluster``   — member of a compact/blob component
  - ``noise``     — positive but in a tiny component (< min-component-size)
  - ``other``     — not positive for the marker

Example:
    python detect_curvilinear_patterns.py \\
        --detections classified.json \\
        --snr-channel 2 --snr-threshold 1.5 \\
        --radius 50 --min-component-size 10 \\
        --linearity-threshold 3.0 \\
        --output-prefix msln
"""

import argparse
from pathlib import Path

import numpy as np

from xldvp_seg.analysis.pattern_detection import (
    classify_components,
    refine_strip_cells,
    select_positive_cells,
)
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect curvilinear spatial patterns via graph component analysis"
    )
    parser.add_argument("--detections", required=True, help="Path to classified detection JSON")

    # Marker selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--snr-channel",
        type=int,
        help="Channel index for SNR-based filtering (uses ch{N}_snr from features)",
    )
    group.add_argument(
        "--marker-filter",
        help='Filter expression for positive cells (e.g., "MSLN_class==positive")',
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=1.5,
        help="SNR threshold for positive classification (default: 1.5)",
    )

    # Graph params
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Connection radius in µm for graph edges. Should be ~1.5-2x the "
        "typical cell-to-cell spacing. Larger values bridge gaps but may merge "
        "separate structures. 50-100 µm typical for mesothelial strips (default: 50)",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=10,
        help="Components smaller than this are classified as noise (default: 10)",
    )
    parser.add_argument(
        "--linearity-threshold",
        type=float,
        default=3.0,
        help="Min diameter/sqrt(n) ratio to classify as strip (default: 3.0)",
    )
    parser.add_argument(
        "--min-strip-cells",
        type=int,
        default=0,
        help="Min cells in a strip component (components passing linearity but below "
        "this size are classified as cluster instead). 0 = no filter (default: 0)",
    )
    parser.add_argument(
        "--min-strip-length",
        type=float,
        default=0,
        help="Min physical length in µm along the diameter path for a strip. "
        "0 = no filter (default: 0)",
    )
    parser.add_argument(
        "--max-strip-width",
        type=float,
        default=0,
        help="Max perpendicular width in µm. Components wider than this are "
        "classified as cluster even if they pass linearity. Mesothelial strips "
        "are typically 50-200µm wide. 0 = no filter (default: 0)",
    )

    # Cell-level refinement (trim hangers-on from strip components)
    parser.add_argument(
        "--refine-method",
        choices=["none", "betweenness", "degree_ratio", "k_core"],
        default="none",
        help="Per-cell refinement to trim absorbed non-strip cells. "
        "betweenness: keep top N%% by betweenness centrality. "
        "degree_ratio: keep cells where most neighbors are also strip cells. "
        "k_core: keep k-core backbone (removes leaf-like appendages). "
        "(default: none)",
    )
    parser.add_argument(
        "--refine-threshold",
        type=float,
        default=0,
        help="Threshold for refinement. "
        "betweenness: percentile cutoff (e.g., 30 = drop bottom 30%%). "
        "degree_ratio: min fraction of neighbors in same strip (e.g., 0.5). "
        "k_core: minimum degree k (e.g., 2). "
        "(default: 0 = method-specific default)",
    )

    # Output
    parser.add_argument(
        "--output-prefix",
        default="marker",
        help='Prefix for tagged field names (default: "marker" → "marker_pattern")',
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as input detections)",
    )

    return parser.parse_args()


def extract_aligned_positions(detections, positive_idx, extract_positions_um):
    """Extract positions ensuring exact index alignment with positive_idx."""
    positive_dets = [detections[i] for i in positive_idx]

    # Infer pixel_size from the batch
    _, pixel_size = extract_positions_um(positive_dets)
    px_str = f"{pixel_size:.4f}" if pixel_size else "N/A"

    # Resolve per-detection to maintain exact index alignment.
    # extract_positions_um silently skips unresolvable detections without
    # reporting which indices survived. Per-detection calls are the only way
    # to track the mapping between input indices and output positions.
    valid_positive_idx = []
    valid_positions = []
    for i, det in zip(positive_idx, positive_dets):
        pos_arr, _ = extract_positions_um([det], pixel_size_um=pixel_size)
        if len(pos_arr) == 1:
            valid_positive_idx.append(i)
            valid_positions.append(pos_arr[0])

    if len(valid_positive_idx) < len(positive_idx):
        logger.warning(
            "Dropped %d cells with unresolvable coordinates (%d → %d)",
            len(positive_idx) - len(valid_positive_idx),
            len(positive_idx),
            len(valid_positive_idx),
        )

    positions = (
        np.array(valid_positions, dtype=np.float64)
        if valid_positions
        else np.empty((0, 2), dtype=np.float64)
    )
    logger.info("  Positions extracted: %d cells, pixel_size=%s µm", len(positions), px_str)
    return valid_positive_idx, positions


def tag_detections(detections, positive_idx, labels, prefix):
    """Tag each detection with pattern classification."""
    field = f"{prefix}_pattern"

    for d in detections:
        d.setdefault("features", {})[field] = "other"

    for j, orig_idx in enumerate(positive_idx):
        detections[orig_idx]["features"][field] = labels[j]

    counts = {}
    for d in detections:
        c = d.get("features", {}).get(field, "other")
        counts[c] = counts.get(c, 0) + 1
    logger.info("Final tagged classes: %s", counts)

    return field


def main():
    args = parse_args()

    from xldvp_seg.utils.detection_utils import extract_positions_um
    from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load

    # Load
    logger.info("Loading %s...", args.detections)
    detections = fast_json_load(args.detections)
    logger.info("  %d detections loaded", len(detections))

    # Select positive cells
    positive_idx = select_positive_cells(
        detections,
        snr_channel=args.snr_channel,
        snr_threshold=args.snr_threshold,
        marker_filter=args.marker_filter,
    )
    if len(positive_idx) < 2:
        logger.error("Only %d positive cells — not enough for analysis", len(positive_idx))
        raise SystemExit(1)

    # Extract positions
    positive_idx, positions = extract_aligned_positions(
        detections, positive_idx, extract_positions_um
    )
    if len(positions) < 2:
        logger.error("Only %d resolved positions — not enough", len(positions))
        raise SystemExit(1)

    # Classify
    labels, comp_stats, G = classify_components(
        positions,
        args.radius,
        args.min_component_size,
        args.linearity_threshold,
        args.min_strip_cells,
        args.min_strip_length,
        args.max_strip_width,
    )

    # Per-cell refinement (trim hangers-on)
    if args.refine_method != "none":
        labels = refine_strip_cells(
            positions,
            labels,
            args.radius,
            args.refine_method,
            args.refine_threshold,
            G_all=G,
        )

    # Tag
    field = tag_detections(detections, positive_idx, labels, args.output_prefix)

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.detections).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    det_out = out_dir / f"cell_detections_{args.output_prefix}_strip_tagged.json"
    logger.info("Saving tagged detections to %s...", det_out)
    atomic_json_dump(detections, str(det_out))

    # Write strip-only JSON for fast viewer generation
    strip_dets = [d for d in detections if d.get("features", {}).get(field) == "strip"]
    strip_out = out_dir / f"cell_detections_{args.output_prefix}_strip_only.json"
    logger.info("Saving %d strip-only detections to %s...", len(strip_dets), strip_out)
    atomic_json_dump(strip_dets, str(strip_out))

    stats_out = out_dir / f"{args.output_prefix}_component_stats.json"
    atomic_json_dump(comp_stats, str(stats_out))
    logger.info("Saved component stats to %s", stats_out)
    logger.info("Done. Group field for viewer: %s", field)


if __name__ == "__main__":
    main()
