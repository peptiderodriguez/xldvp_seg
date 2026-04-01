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

from segmentation.utils.graph_topology import (
    build_radius_graph_sparse,
    compute_all_metrics,
)
from segmentation.utils.json_utils import atomic_json_dump, fast_json_load
from segmentation.utils.logging import get_logger

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
    from segmentation.utils.detection_utils import extract_positions_um

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


# ---------------------------------------------------------------------------
# Marker composition + spatial layering
# ---------------------------------------------------------------------------


def analyze_marker_composition(detections, comp_global_indices, marker_names, class_keys):
    """Count marker-positive cells per marker in a component."""
    counts = dict.fromkeys(marker_names, 0)
    n = len(comp_global_indices)

    for idx in comp_global_indices:
        feat = detections[idx].get("features", {})
        for name, key in zip(marker_names, class_keys):
            if feat.get(key) == "positive":
                counts[name] += 1

    composition = {"n_cells": n}
    for name in marker_names:
        composition[f"n_{name.lower()}"] = counts[name]
        composition[f"{name.lower()}_frac"] = round(counts[name] / max(n, 1), 3)

    # Dominant marker
    if marker_names:
        dominant = max(marker_names, key=lambda m: counts[m])
        composition["dominant_marker"] = dominant if counts[dominant] > 0 else "none"

    return composition


def detect_spatial_layering(
    positions, detections, comp_local_indices, comp_global_indices, class_keys_map
):
    """Detect radial layering of markers (SMA outer vs CD31 inner).

    Uses Mann-Whitney U test for statistical rigor.
    """
    from scipy.stats import mannwhitneyu

    pts = positions[comp_local_indices]
    centroid = pts.mean(axis=0)
    dists = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))

    layering = {}

    # Get per-marker radial distributions — O(n) per marker via enumerate
    marker_radii = {}
    for marker_name, class_key in class_keys_map.items():
        radii = []
        for enum_i, global_i in enumerate(comp_global_indices):
            feat = detections[global_i].get("features", {})
            if feat.get(class_key) == "positive":
                radii.append(dists[enum_i])
        if radii:
            marker_radii[marker_name] = np.array(radii)

    # Pairwise layering tests — test BOTH directions per pair
    markers = list(marker_radii.keys())
    for i, m1 in enumerate(markers):
        for m2 in markers[i + 1 :]:
            r1 = marker_radii[m1]
            r2 = marker_radii[m2]
            if len(r1) >= 5 and len(r2) >= 5:
                # Test m1 outer vs m2
                _, pval_fwd = mannwhitneyu(r1, r2, alternative="greater")
                # Test m2 outer vs m1
                _, pval_rev = mannwhitneyu(r2, r1, alternative="greater")

                # Report the significant direction (or forward if neither)
                if pval_rev < pval_fwd:
                    # m2 is outer
                    key = f"{m2}_outer_vs_{m1}"
                    score = float(np.median(r2) - np.median(r1))
                    pval = pval_rev
                else:
                    # m1 is outer (or neither significant)
                    key = f"{m1}_outer_vs_{m2}"
                    score = float(np.median(r1) - np.median(r2))
                    pval = pval_fwd

                layering[key] = {
                    "score": round(score, 2),
                    "p_value": round(float(pval), 4),
                    "significant": pval < 0.05,
                    f"n_{m1.lower()}": len(r1),
                    f"n_{m2.lower()}": len(r2),
                }

    return layering


# ---------------------------------------------------------------------------
# Vessel type assignment
# ---------------------------------------------------------------------------


def assign_vessel_type(morphology, composition, layering, morphometry):
    """Assign vessel type from morphology + marker composition + wall morphometry.

    Artery vs vein: both have CD31 inner + SMA outer. Key distinction is wall
    thickness (wall_cell_layers > 1.5 or wall/diameter ratio > 0.3 → artery).

    Size subtyping: artery (>100µm) vs arteriole (≤100µm), vein (>50µm) vs
    venule (≤50µm).

    Lymphatics: LYVE1+ with SMA (≥15%) → collecting_lymphatic (has smooth
    muscle wall). LYVE1+ without SMA → initial lymphatic.

    Spatial layering (has_sma_outer) provides additional confidence for artery
    classification when both SMA and CD31 are present.
    """
    sma_frac = composition.get("sma_frac", 0)
    cd31_frac = composition.get("cd31_frac", 0)
    lyve1_frac = composition.get("lyve1_frac", 0)
    diameter = morphometry.get("vessel_diameter_um", 0) or 0

    # LYVE1+ vessels — distinguish collecting (has SMA) from initial (no SMA)
    # Morphology-aware: strips get specific longitudinal label below.
    if lyve1_frac > 0.3 and lyve1_frac > sma_frac and morphology != "strip":
        if sma_frac > 0.15:
            return "collecting_lymphatic"  # LYVE1+ with SMA smooth muscle wall
        return "lymphatic"  # initial lymphatic, LYVE1+ only

    # Check layering for SMA outer vs CD31 inner (artery signature)
    has_sma_outer = False
    if layering:
        for key, info in layering.items():
            if "SMA_outer" in key and info.get("significant"):
                has_sma_outer = True

    if morphology in ("ring", "arc"):
        # Both arteries and veins have CD31 inner + SMA outer.
        # Key distinction: wall thickness relative to vessel size.
        #   Artery: thick SMA wall, multiple smooth muscle layers
        #   Vein: thin SMA wall, larger/irregular lumen
        wall_extent = morphometry.get("wall_extent_um", 0) or 0
        wall_layers = morphometry.get("wall_cell_layers", 0) or 0
        wall_ratio = wall_extent / diameter if diameter > 0 else 0

        if sma_frac > 0.15 or cd31_frac > 0.15:
            # Has vascular markers — classify by wall morphometry
            has_morphometry = "wall_cell_layers" in morphometry
            is_thick_wall = wall_layers > 1.5 or wall_ratio > 0.3

            if is_thick_wall or (has_sma_outer and sma_frac > 0.3):
                # Thick wall OR confirmed SMA-outer layering → artery/arteriole
                if diameter > 100:
                    return "artery"
                elif diameter > 0:
                    return "arteriole"
                else:
                    return "artery"
            elif not has_morphometry and sma_frac > cd31_frac:
                # No morphometry available, SMA dominant → likely muscular vessel
                return "artery"
            elif cd31_frac > sma_frac:
                # CD31 dominant with thin wall = vein/venule
                if diameter > 50:
                    return "vein"
                elif diameter > 0:
                    return "venule"
                else:
                    return "vein"
            elif sma_frac > cd31_frac:
                # SMA dominant but thin wall → arteriole (still muscular)
                if diameter > 100:
                    return "artery"
                elif diameter > 0:
                    return "arteriole"
                else:
                    return "artery"
            else:
                return "unclassified"
        else:
            return "unclassified"
    elif morphology == "strip":
        if sma_frac > 0.3:
            return "artery_longitudinal"
        elif cd31_frac > 0.3:
            return "vein_longitudinal"
        elif lyve1_frac > 0.1:
            return "lymphatic_longitudinal"
        else:
            return "unclassified"
    elif morphology == "cluster":
        if cd31_frac > 0.5 and composition.get("n_cells", 0) < 15:
            return "capillary"
        return "unclassified"

    return "unclassified"


# ---------------------------------------------------------------------------
# Tag detections
# ---------------------------------------------------------------------------


def tag_detections(detections, positive_idx, comp_assignments, prefix="vessel"):
    """Tag each detection with vessel structure membership."""
    field_id = f"{prefix}_id"
    field_type = f"{prefix}_type"
    field_morph = f"{prefix}_morphology"
    field_size = f"{prefix}_size_class"

    # Initialize all detections
    for d in detections:
        feat = d.setdefault("features", {})
        feat[field_id] = -1
        feat[field_type] = "none"
        feat[field_morph] = "none"
        feat[field_size] = "none"

    # Apply assignments
    for global_idx, assignment in comp_assignments.items():
        feat = detections[global_idx].setdefault("features", {})
        feat[field_id] = assignment["vessel_id"]
        feat[field_type] = assignment["vessel_type"]
        feat[field_morph] = assignment["morphology"]
        feat[field_size] = assignment.get("size_class", "unknown")

    # Summary
    type_counts = {}
    for d in detections:
        vt = d.get("features", {}).get(field_type, "none")
        type_counts[vt] = type_counts.get(vt, 0) + 1
    logger.info("Vessel type distribution: %s", type_counts)

    return field_type


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

        # Marker composition
        composition = analyze_marker_composition(detections, comp_global, marker_names, class_keys)

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
    field = tag_detections(detections, positive_idx, comp_assignments, args.output_prefix)

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
            marker_cols.extend([f"n_{name.lower()}", f"{name.lower()}_frac"])
        marker_cols.append("dominant_marker")
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
