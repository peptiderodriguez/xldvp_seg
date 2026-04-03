#!/usr/bin/env python3
"""Detect vessel structures from marker-positive cells using graph topology.

Takes classified cell detections (from classify_markers.py) with SMA, CD31,
and/or LYVE1 marker classifications and identifies vessel structures:

1. Selects cells positive for ANY vessel marker (OR logic)
2. Builds spatial radius graph (scipy.sparse for CC, nx per-component)
3. Classifies each component morphology: ring, arc, strip, cluster
   - Graph topology: linearity (strips), ring_score (rings), arc_fraction (arcs)
   - Geometric/PCA: elongation, circularity, hollowness, curvature
4. Computes vessel morphometry: diameter, lumen, wall extent, cell layers
5. Analyzes marker composition + spatial layering (SMA outer vs CD31 inner)
6. Assigns vessel type: artery, arteriole, vein, venule, capillary, lymphatic, collecting_lymphatic
7. Tags detections and writes output JSON + CSV

Example:
    python scripts/detect_vessel_structures.py \\
        --detections cell_detections_classified.json \\
        --marker-filter "SMA_class==positive" \\
        --marker-filter "CD31_class==positive" \\
        --marker-logic or \\
        --radius 50 --min-cells 5 \\
        --output-dir vessel_structures/
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from xldvp_seg.analysis.vessel_characterization import (
    analyze_marker_composition,
    assign_vessel_type,
    detect_spatial_layering,
    tag_detections,
)
from xldvp_seg.utils.graph_topology import (
    build_radius_graph_sparse,
    compute_all_metrics,
)
from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Cell selection — multi-marker OR/AND
# ---------------------------------------------------------------------------


def select_multi_marker_cells(detections, marker_filters, logic="or"):
    """Select cells matching marker filter expressions.

    Args:
        detections: list of detection dicts.
        marker_filters: list of "key==value" filter expressions.
        logic: "or" (any filter matches) or "and" (all filters match).

    Returns:
        list of indices into detections.
    """
    if not marker_filters:
        return []

    # Parse each filter
    parsed = []
    for filt in marker_filters:
        if "==" not in filt:
            logger.error("Malformed filter: %r (expected 'key==value')", filt)
            raise SystemExit(1)
        key, value = filt.split("==", 1)
        key, value = key.strip(), value.strip()
        match_values = {value}
        if value.lower() in ("true", "false"):
            match_values.add(value.lower() == "true")
        parsed.append((key, match_values))

    positive = []
    for i, d in enumerate(detections):
        feat = d.get("features", {})
        matches = []
        for key, match_values in parsed:
            v1 = d.get(key)
            v2 = feat.get(key)
            matches.append(v1 in match_values or v2 in match_values)

        if logic == "or" and any(matches):
            positive.append(i)
        elif logic == "and" and all(matches):
            positive.append(i)

    logger.info(
        "Multi-marker selection (%s logic, %d filters): %d / %d cells positive",
        logic,
        len(marker_filters),
        len(positive),
        len(detections),
    )
    for filt in marker_filters:
        logger.info("  Filter: %s", filt)

    return positive


# ---------------------------------------------------------------------------
# Position extraction (reuse pattern from curvilinear)
# ---------------------------------------------------------------------------


def extract_aligned_positions(detections, positive_idx):
    """Extract um positions with exact index alignment."""
    from xldvp_seg.utils.detection_utils import extract_positions_um

    positive_dets = [detections[i] for i in positive_idx]
    _, pixel_size = extract_positions_um(positive_dets)

    valid_idx = []
    valid_pos = []
    for i, det in zip(positive_idx, positive_dets):
        pos_arr, _ = extract_positions_um([det], pixel_size_um=pixel_size)
        if len(pos_arr) == 1:
            valid_idx.append(i)
            valid_pos.append(pos_arr[0])

    if len(valid_idx) < len(positive_idx):
        logger.warning(
            "Dropped %d cells with unresolvable coordinates",
            len(positive_idx) - len(valid_idx),
        )

    positions = (
        np.array(valid_pos, dtype=np.float64) if valid_pos else np.empty((0, 2), dtype=np.float64)
    )
    logger.info("  Positions: %d cells resolved", len(positions))
    return valid_idx, positions


# ---------------------------------------------------------------------------
# Morphology classification
# ---------------------------------------------------------------------------


def classify_vessel_morphology(metrics, args):
    """Classify a component as ring, arc, strip, or cluster from its metrics.

    Uses both graph topology and geometric/PCA metrics with OR logic.
    Starting thresholds — expect iterative tuning.
    """
    lin = metrics["linearity"]
    elong = metrics["elongation"]
    rs = metrics["ring_score"]
    af = metrics["arc_fraction"]
    circ = metrics["circularity"]
    hollow = metrics["hollowness"]
    curv = metrics["has_curvature"]

    # Strip: high linearity (graph) OR high elongation without curvature (PCA)
    if lin > args.linearity_threshold or (elong > 4.0 and not curv):
        return "strip"

    # Ring: graph ring_score OR geometric circularity, plus hollowness
    if (rs > args.ring_threshold or circ > 0.65) and hollow > 0.5:
        return "ring"

    # Arc: partial ring (graph) or curved elongated (PCA)
    if af > args.arc_threshold and hollow > 0.4:
        return "arc"
    if curv and elong > 3.0:
        return "arc"

    return "cluster"


# ---------------------------------------------------------------------------
# Vessel morphometry
# ---------------------------------------------------------------------------


def compute_vessel_morphometry(
    positions, comp_local_indices, morphology, all_detections=None, comp_global_indices=None
):
    """Compute vessel measurements from cell positions.

    Args:
        positions: (N, 2) array of positive-cell positions.
        comp_local_indices: indices into positions array (local).
        morphology: "ring", "arc", "strip", or "cluster".
        all_detections: full detections list (for area_um2 lookup).
        comp_global_indices: indices into all_detections (global).
            Required when all_detections is provided.
    """
    pts = positions[comp_local_indices]
    n = len(pts)

    centroid = pts.mean(axis=0)
    dists = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))

    morph = {
        "centroid_x_um": round(float(centroid[0]), 2),
        "centroid_y_um": round(float(centroid[1]), 2),
        "cell_count": n,
    }

    # Lumen/wall metrics for rings and arcs (arcs are approximate but still useful for sizing)
    if morphology in ("ring", "arc") and n >= 5:
        outer_r = float(np.percentile(dists, 95))
        inner_r = float(np.percentile(dists, 5))
        wall_extent = outer_r - inner_r
        mean_r = float(np.mean(dists))

        morph.update(
            {
                "vessel_diameter_um": round(2 * outer_r, 1),
                "lumen_diameter_um": round(2 * inner_r, 1),
                "wall_extent_um": round(wall_extent, 1),
                "lumen_area_um2": round(np.pi * inner_r**2, 1),
                "wall_area_um2": round(np.pi * (outer_r**2 - inner_r**2), 1),
                "vessel_circumference_um": round(2 * np.pi * mean_r, 1),
                "wall_uniformity": round(max(0.0, float(1.0 - dists.std() / max(mean_r, 1e-6))), 3),
            }
        )

        # Wall cell layers — estimate from mean cell diameter
        if all_detections is not None and comp_global_indices is not None and wall_extent > 0:
            cell_areas = []
            for g_idx in comp_global_indices:
                area = all_detections[g_idx].get("features", {}).get("area_um2", 0)
                if area > 0:
                    cell_areas.append(area)
            if cell_areas:
                mean_cell_diam = float(2 * np.mean(np.sqrt(np.array(cell_areas) / np.pi)))
                morph["wall_cell_layers"] = round(wall_extent / max(mean_cell_diam, 1e-6), 1)

    # Size classification
    diam = morph.get("vessel_diameter_um", 0)
    if diam > 100:
        morph["size_class"] = "large"
    elif diam > 30:
        morph["size_class"] = "medium"
    elif diam > 0:
        morph["size_class"] = "small"
    else:
        morph["size_class"] = "unknown"

    return morph


# analyze_marker_composition, detect_spatial_layering, assign_vessel_type,
# and tag_detections are imported from xldvp_seg.analysis.vessel_characterization


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect vessel structures from marker-positive cells"
    )
    parser.add_argument("--detections", required=True, help="Classified detection JSON")
    parser.add_argument(
        "--marker-filter",
        action="append",
        default=[],
        help='Filter expression (e.g., "SMA_class==positive"). Repeat for multiple markers.',
    )
    parser.add_argument(
        "--marker-logic",
        choices=["or", "and"],
        default="or",
        help="How to combine marker filters (default: or)",
    )
    parser.add_argument(
        "--marker-names",
        default=None,
        help='Comma-separated marker names for composition (e.g., "SMA,CD31,LYVE1"). '
        "Auto-detected from filters if not specified.",
    )

    # Graph parameters
    parser.add_argument(
        "--radius", type=float, default=50.0, help="Connection radius µm (default: 50)"
    )
    parser.add_argument(
        "--min-cells", type=int, default=5, help="Min cells per component (default: 5)"
    )
    parser.add_argument(
        "--min-marker-cells",
        type=int,
        default=0,
        help="Min cell-equivalents of ANY marker (single-positive only). "
        "A vessel is kept if ANY marker has total single-positive area >= "
        "N * median_cell_area. Double-positive cells excluded (contaminate "
        "both proteomes). 0 = no filter (default: 0). Use 12 for DVP.",
    )

    # Classification thresholds
    parser.add_argument(
        "--linearity-threshold", type=float, default=3.0, help="Strip threshold (default: 3.0)"
    )
    parser.add_argument(
        "--ring-threshold", type=float, default=0.5, help="Ring score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--arc-threshold", type=float, default=0.3, help="Arc fraction threshold (default: 0.3)"
    )

    # Output
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--output-prefix", default="vessel", help="Output file prefix (default: vessel)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load detections
    logger.info("Loading %s...", args.detections)
    detections = fast_json_load(args.detections)
    logger.info("  %d detections loaded", len(detections))

    # Select marker-positive cells
    if not args.marker_filter:
        logger.error("No --marker-filter specified")
        raise SystemExit(1)

    positive_idx = select_multi_marker_cells(detections, args.marker_filter, args.marker_logic)
    if len(positive_idx) < args.min_cells:
        logger.error("Only %d positive cells — not enough", len(positive_idx))
        raise SystemExit(1)

    # Extract positions
    positive_idx, positions = extract_aligned_positions(detections, positive_idx)
    if len(positions) < args.min_cells:
        logger.error("Only %d resolved positions", len(positions))
        raise SystemExit(1)

    # Auto-detect marker names from filters
    if args.marker_names:
        marker_names = [m.strip() for m in args.marker_names.split(",")]
    else:
        marker_names = []
        for filt in args.marker_filter:
            key = filt.split("==")[0].strip()
            name = key.replace("_class", "")
            marker_names.append(name)
    class_keys = [f"{name}_class" for name in marker_names]
    class_keys_map = dict(zip(marker_names, class_keys))
    logger.info("Markers: %s", marker_names)

    # Pre-compute median cell area per marker (single-positive only)
    # for min-marker-cells area-equivalent filter.
    # Double-positive cells excluded — they can't be cleanly cut for either proteome.
    median_area_per_class = {}
    if args.min_marker_cells > 0:
        class_areas = {name: [] for name in marker_names}
        for idx in positive_idx:
            feat = detections[idx].get("features", {})
            pos = [n for n, k in zip(marker_names, class_keys) if feat.get(k) == "positive"]
            a = feat.get("area_um2", 0)
            if a > 0 and len(pos) == 1:
                class_areas[pos[0]].append(a)
        for name, areas in class_areas.items():
            if areas:
                median_area_per_class[name] = float(np.median(areas))
                logger.info(
                    "  %s: median single-positive cell area = %.1f µm² (%d cells)",
                    name,
                    median_area_per_class[name],
                    len(areas),
                )
            else:
                median_area_per_class[name] = 0

    # Build graph + find components
    logger.info("Building radius graph (r=%.0f µm)...", args.radius)
    n_comp, labels, pairs = build_radius_graph_sparse(positions, args.radius)
    logger.info("  %d connected components", n_comp)

    # Build full nx graph once for subgraph extraction (avoids O(n_comp * |pairs|))
    import networkx as nx

    G_full = nx.Graph()
    G_full.add_nodes_from(range(len(positions)))
    G_full.add_edges_from(pairs)

    # Analyze each component
    structures = []
    comp_assignments = {}  # global_idx → {vessel_id, vessel_type, morphology, size_class}
    struct_id = 0

    for comp_label in range(n_comp):
        comp_mask = labels == comp_label
        n_cells = int(comp_mask.sum())
        if n_cells < args.min_cells:
            continue

        comp_local = np.where(comp_mask)[0]
        comp_global = [positive_idx[i] for i in comp_local]
        comp_nodes = set(comp_local.tolist())

        # Marker composition (cheap — do before expensive graph metrics)
        composition = analyze_marker_composition(detections, comp_global, marker_names, class_keys)

        # Filter: require area equivalent of min_marker_cells of SINGLE-POSITIVE
        # cells for each marker. Double-positive cells are excluded from LMD
        # cuts (contaminate both proteomes), so they don't count toward viability.
        if args.min_marker_cells > 0:
            class_area = dict.fromkeys(marker_names, 0.0)
            for g_idx in comp_global:
                feat = detections[g_idx].get("features", {})
                pos = [n for n, k in zip(marker_names, class_keys) if feat.get(k) == "positive"]
                if len(pos) == 1:
                    class_area[pos[0]] += feat.get("area_um2", 0)
            # OR logic: ANY marker must meet the threshold (not all).
            # If no median available (no area_um2 in data), skip filter.
            any_above = False
            any_checked = False
            for name in marker_names:
                median_a = median_area_per_class.get(name, 0)
                if median_a <= 0:
                    continue
                any_checked = True
                if class_area[name] >= args.min_marker_cells * median_a:
                    any_above = True
                    break
            if any_checked and not any_above:
                continue

        # Subgraph via the pre-built full graph (O(k) not O(|pairs|))
        G = G_full.subgraph(comp_nodes)

        # Compute all metrics
        metrics = compute_all_metrics(positions, G, comp_nodes)

        # Classify morphology
        morphology = classify_vessel_morphology(metrics, args)

        # Morphometry — pass both local (for positions) and global (for detections) indices
        morphometry = compute_vessel_morphometry(
            positions, list(comp_local), morphology, detections, comp_global
        )

        # Spatial layering (for rings/arcs with sufficient cells)
        layering = {}
        if morphology in ("ring", "arc") and n_cells >= 10:
            layering = detect_spatial_layering(
                positions, detections, list(comp_local), comp_global, class_keys_map
            )

        # Vessel type
        vessel_type = assign_vessel_type(morphology, composition, layering, morphometry)

        # Build structure record
        structure = {
            "id": struct_id,
            "morphology": morphology,
            "vessel_type": vessel_type,
            **metrics,
            **morphometry,
            "composition": composition,
            "layering": layering if layering else None,
            "cell_indices": comp_global,
        }
        structures.append(structure)

        # Tag cells
        for global_i in comp_global:
            comp_assignments[global_i] = {
                "vessel_id": struct_id,
                "vessel_type": vessel_type,
                "morphology": morphology,
                "size_class": morphometry.get("size_class", "unknown"),
            }

        struct_id += 1

    logger.info("")
    logger.info("=" * 70)
    logger.info("VESSEL STRUCTURE DETECTION SUMMARY")
    logger.info("=" * 70)
    logger.info("  Total structures: %d", len(structures))
    morph_counts = {}
    type_counts = {}
    for s in structures:
        morph_counts[s["morphology"]] = morph_counts.get(s["morphology"], 0) + 1
        type_counts[s["vessel_type"]] = type_counts.get(s["vessel_type"], 0) + 1
    logger.info("  Morphology: %s", morph_counts)
    logger.info("  Vessel types: %s", type_counts)
    logger.info("  Total vessel cells: %d / %d positive", len(comp_assignments), len(positive_idx))

    # Tag all detections
    field = tag_detections(
        detections, comp_assignments, marker_names, class_keys, args.output_prefix
    )

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.detections).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tagged detections
    det_out = out_dir / f"cell_detections_{args.output_prefix}_tagged.json"
    logger.info("Saving tagged detections to %s...", det_out)
    atomic_json_dump(detections, str(det_out))

    # Vessel-only detections for fast viewer generation
    vessel_dets = [
        d for d in detections if d.get("features", {}).get(f"{args.output_prefix}_id", -1) >= 0
    ]
    vessel_out = out_dir / f"cell_detections_{args.output_prefix}_only.json"
    logger.info("Saving %d vessel-only detections to %s...", len(vessel_dets), vessel_out)
    atomic_json_dump(vessel_dets, str(vessel_out))

    # Structure summary JSON
    # Remove cell_indices for the summary (large lists), flatten composition
    summary_structures = []
    for s in structures:
        ss = {k: v for k, v in s.items() if k not in ("cell_indices", "composition", "layering")}
        ss["n_cells"] = len(s["cell_indices"])
        # Flatten composition into top-level keys for CSV (skip n_cells — already set above)
        if s.get("composition"):
            for ck, cv in s["composition"].items():
                if ck != "n_cells":
                    ss[ck] = cv
        summary_structures.append(ss)
    struct_out = out_dir / f"{args.output_prefix}_structures.json"
    atomic_json_dump(summary_structures, str(struct_out))
    logger.info("Saved %d structure summaries to %s", len(summary_structures), struct_out)

    # CSV summary
    csv_out = out_dir / f"{args.output_prefix}_structures.csv"
    if structures:
        # Dynamic fieldnames: core + marker-specific columns
        marker_cols = []
        for name in marker_names:
            marker_cols.extend(
                [
                    f"n_{name.lower()}",
                    f"{name.lower()}_frac",
                    f"{name.lower()}_total_frac",
                ]
            )
        marker_cols.extend(["n_double_pos", "double_pos_frac", "dominant_marker"])
        fieldnames = (
            [
                "id",
                "morphology",
                "vessel_type",
                "n_cells",
                "vessel_diameter_um",
                "lumen_diameter_um",
                "wall_extent_um",
                "size_class",
            ]
            + marker_cols
            + [
                "linearity",
                "ring_score",
                "arc_fraction",
                "elongation",
                "circularity",
                "hollowness",
                "hull_area_um2",
                "centroid_x_um",
                "centroid_y_um",
            ]
        )
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in summary_structures:
                writer.writerow(s)
        logger.info("Saved CSV summary to %s", csv_out)

    logger.info("Done. Vessel field: %s", field)


if __name__ == "__main__":
    main()
