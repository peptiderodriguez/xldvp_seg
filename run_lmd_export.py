#!/usr/bin/env python3
"""
Unified LMD Export Tool - Export detections to Leica LMD format.

Works with any cell type (NMJ, MK, vessel, mesothelium, etc.).

Handles the complete pipeline:
1. Load detections + filter by score/annotations
2. Load biological clusters (from cluster_detections.py)
3. Extract contours from H5 masks (if needed)
4. Post-process contours (dilate + RDP simplify)
5. Order singles and clusters by nearest-neighbor path on slide
6. Generate spatial controls for ALL samples (singles and clusters)
7. Assign wells in serpentine order with alternating target/control
8. Export to Leica LMD XML via py-lmd

Usage:
    # Full pipeline with clusters and controls
    python run_lmd_export.py \\
        --detections detections.json \\
        --cell-type nmj \\
        --crosses reference_crosses.json \\
        --clusters clusters.json \\
        --tiles-dir /path/to/tiles \\
        --output-dir lmd_export \\
        --export --generate-controls

    # Generate cross placement HTML (step before export)
    python run_lmd_export.py \\
        --detections detections.json \\
        --output-dir lmd_export \\
        --generate-cross-html
"""

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
from pathlib import Path

import numpy as np

from xldvp_seg.utils.json_utils import fast_json_load
from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Import pure-logic functions from package module
# ---------------------------------------------------------------------------

from xldvp_seg.lmd.export import (
    _build_spatial_index,
    _make_polygon,
    assign_wells_with_controls,
    build_export_data,
    export_to_lmd_xml,
    extract_contours_for_detections,
    filter_detections,
    generate_cluster_control,
    generate_cross_placement_html,
    generate_spatial_control,
    get_detection_coordinates,
    load_annotations,
    load_clusters,
    load_detections,
    nearest_neighbor_order,
)
from xldvp_seg.lmd.well_plate import (
    WELLS_PER_PLATE,
    generate_plate_wells,
)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified LMD Export - detections to Leica LMD format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with clusters and controls
  python run_lmd_export.py \\
      --detections detections.json \\
      --cell-type nmj \\
      --crosses reference_crosses.json \\
      --clusters clusters.json \\
      --tiles-dir /path/to/tiles \\
      --output-dir lmd_export \\
      --export --generate-controls

  # Generate HTML for placing reference crosses
  python run_lmd_export.py \\
      --detections detections.json \\
      --output-dir lmd_export \\
      --generate-cross-html
""",
    )

    # Input files
    parser.add_argument(
        "--detections",
        type=str,
        required=False,
        default=None,
        help="Path to detections JSON file (required for single-slide mode)",
    )
    parser.add_argument(
        "--cell-type",
        type=str,
        default=None,
        help="Cell type (nmj, mk, vessel, mesothelium). "
        "Auto-derives mask filename if --mask-filename not set.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to annotations JSON (filters to positives only)",
    )
    parser.add_argument("--crosses", type=str, default=None, help="Path to reference crosses JSON")
    parser.add_argument(
        "--clusters",
        type=str,
        default=None,
        help="Path to clusters JSON from cluster_detections.py",
    )

    # Contour extraction
    parser.add_argument(
        "--tiles-dir", type=str, default=None, help="Path to tiles/ directory with H5 masks"
    )
    parser.add_argument(
        "--mask-filename",
        type=str,
        default=None,
        help="Mask filename within each tile dir (default: auto from --cell-type, or nmj_masks.h5)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for LMD files"
    )
    parser.add_argument(
        "--output-name", type=str, default="shapes", help="Base name for output files"
    )

    # Actions
    parser.add_argument(
        "--generate-cross-html",
        action="store_true",
        help="Generate HTML for placing reference crosses",
    )
    parser.add_argument(
        "--export", action="store_true", help="Export to LMD XML (requires --crosses)"
    )

    # Image metadata
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in um (auto-detect from detections if not set)",
    )
    parser.add_argument("--image-width", type=int, default=None, help="Image width in pixels")
    parser.add_argument("--image-height", type=int, default=None, help="Image height in pixels")

    # Filtering
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum rf_prediction score (filters detections)",
    )

    # Zone filtering
    parser.add_argument(
        "--zone-filter",
        type=str,
        default=None,
        help='Include only these zone IDs (comma-separated, e.g. "1,3,5")',
    )
    parser.add_argument(
        "--zone-exclude",
        type=str,
        default=None,
        help='Exclude these zone IDs (comma-separated, e.g. "0")',
    )

    # Controls
    parser.add_argument(
        "--generate-controls",
        action="store_true",
        help="Generate spatial control regions for every target",
    )
    parser.add_argument(
        "--control-offset-um",
        type=float,
        default=100.0,
        help="Offset distance for controls in um (default: 100)",
    )

    # Contour processing
    parser.add_argument(
        "--dilation-um", type=float, default=0.5, help="Contour dilation in um (default: 0.5)"
    )
    parser.add_argument(
        "--rdp-epsilon",
        type=float,
        default=5.0,
        help="RDP simplification epsilon in pixels (default: 5). "
        "Ignored when --max-area-change-pct is set.",
    )
    parser.add_argument(
        "--max-area-change-pct",
        type=float,
        default=10.0,
        help="Max shape deviation (%%) allowed by adaptive RDP simplification "
        "(symmetric difference / original area, default: 10.0). "
        "Set to 0 to disable adaptive RDP and use fixed --rdp-epsilon.",
    )
    parser.add_argument(
        "--max-dilation-area-pct",
        type=float,
        default=10.0,
        help="Max area increase (%%) allowed by adaptive dilation "
        "(default: 10.0). Overrides --dilation-um. "
        "Set to 0 to disable adaptive dilation and use fixed --dilation-um.",
    )
    parser.add_argument(
        "--erosion-um",
        type=float,
        default=0.0,
        help="Shrink contours by absolute distance in um (default: 0, no erosion). "
        "Applied after dilation+RDP.",
    )
    parser.add_argument(
        "--erode-pct",
        type=float,
        default=0.0,
        help="Shrink contours by percentage of sqrt(area) (default: 0, no erosion). "
        "E.g. 0.05 = 5%% erosion. Applied after dilation+RDP.",
    )

    # Multi-slide batch mode
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Batch mode: directory with per-slide *_detections.json files",
    )
    parser.add_argument(
        "--crosses-dir",
        type=str,
        default=None,
        help="Batch mode: directory with per-slide *_crosses.json files",
    )

    # Options
    parser.add_argument(
        "--no-flip-y", action="store_true", help="Do not flip Y axis for stage coordinates"
    )

    args = parser.parse_args()

    # Auto-derive mask filename from cell type
    if args.mask_filename is None:
        if args.cell_type:
            args.mask_filename = f"{args.cell_type}_masks.h5"
        else:
            args.mask_filename = "nmj_masks.h5"  # backward-compatible default

    # -----------------------------------------------------------------------
    # Batch mode dispatch
    # -----------------------------------------------------------------------
    if getattr(args, "input_dir", None):
        run_batch_export(args)
        return

    _run_single_slide(args)


def _run_single_slide(args):
    # -----------------------------------------------------------------------
    # Load detections (single-slide mode)
    # -----------------------------------------------------------------------
    if args.detections is None:
        logger.error("--detections is required (or use --input-dir for batch mode)")
        return

    logger.info("Loading detections from: %s", args.detections)
    all_detections = load_detections(args.detections)
    logger.info("Loaded %d detections", len(all_detections))

    # Keep original list for cluster index lookups (cluster_detections.py
    # indices reference the ORIGINAL unfiltered list, since it filters internally)
    detections = list(all_detections)

    # Filter by annotations
    if args.annotations:
        logger.info("Loading annotations from: %s", args.annotations)
        positive_uids = load_annotations(args.annotations)
        logger.info("Found %d positive annotations", len(positive_uids))
        detections = filter_detections(detections, positive_uids=positive_uids)
        logger.info("Filtered to %d positive detections", len(detections))

    # Filter by score (skip if --clusters provided, clustering already filtered)
    if args.min_score is not None and not args.clusters:
        before = len(detections)
        detections = filter_detections(detections, min_score=args.min_score)
        logger.info("Score filter (>= %s): %d -> %d", args.min_score, before, len(detections))
    elif args.min_score is not None and args.clusters:
        logger.info("Score filter skipped (clustering already filtered at >= %s)", args.min_score)

    # Filter by zone (from assign_tissue_zones.py)
    if args.zone_filter:
        zone_ids = {int(z) for z in args.zone_filter.split(",")}
        before = len(detections)
        detections = [d for d in detections if d.get("zone_id") in zone_ids]
        logger.info("Zone filter (include %s): %d -> %d", zone_ids, before, len(detections))
    if args.zone_exclude:
        exclude_ids = {int(z) for z in args.zone_exclude.split(",")}
        before = len(detections)
        detections = [d for d in detections if d.get("zone_id") not in exclude_ids]
        logger.info("Zone exclude (%s): %d -> %d", exclude_ids, before, len(detections))

    # Log classifier provenance
    from xldvp_seg.utils.classifier_registry import extract_classifier_info

    _scored, _prov, _clf_info = extract_classifier_info(detections)
    if _scored > 0:
        if _clf_info:
            _f1 = _clf_info.get("cv_f1")
            _f1_str = f" (F1={_f1:.3f})" if _f1 else ""
            _thresh_str = f", threshold={args.min_score}" if args.min_score else ""
            logger.info(
                "LMD export using %d detections scored by %s%s%s",
                len(detections),
                _clf_info.get("classifier_name", "unknown"),
                _f1_str,
                _thresh_str,
            )
        else:
            logger.warning(
                "Detections have %d RF scores with unknown provenance. Verify before cutting.",
                _scored,
            )

    if len(detections) == 0:
        logger.error("No detections to export!")
        return

    # Auto-detect metadata
    pixel_size = args.pixel_size
    if pixel_size is None:
        # Try to get pixel size from detections (top-level first, then features)
        for det in detections:
            if "pixel_size_um" in det:
                pixel_size = det["pixel_size_um"]
                break
            feat = det.get("features", {})
            if "pixel_size_um" in feat:
                pixel_size = feat["pixel_size_um"]
                break
        if pixel_size is None:
            logger.error(
                "pixel_size_um is required. Provide via --pixel-size or ensure "
                "detections JSON contains pixel_size_um."
            )
            return
        logger.info("Pixel size (from detections): %s um/px", pixel_size)
    else:
        logger.info("Pixel size (from CLI): %s um/px", pixel_size)

    image_width = args.image_width
    image_height = args.image_height
    if image_width is None or image_height is None:
        max_x = max_y = 0
        for det in all_detections:
            coords = get_detection_coordinates(det)
            if coords:
                max_x = max(max_x, coords[0])
                max_y = max(max_y, coords[1])
        if max_x == 0 or max_y == 0:
            logger.error(
                "Could not estimate image dimensions from detections. "
                "Provide --image-width and --image-height."
            )
            return
        if image_width is None:
            image_width = int(max_x * 1.1)
        if image_height is None:
            image_height = int(max_y * 1.1)
        logger.info("Image size (estimated): %d x %d px", image_width, image_height)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Generate cross placement HTML
    # -----------------------------------------------------------------------
    if args.generate_cross_html:
        generate_cross_placement_html(detections, output_dir, pixel_size, image_width, image_height)

    # -----------------------------------------------------------------------
    # Export pipeline
    # -----------------------------------------------------------------------
    if args.export:
        if not args.crosses:
            logger.error("--crosses required for export. First use --generate-cross-html")
            return

        logger.info("Loading crosses from: %s", args.crosses)
        crosses_data = fast_json_load(str(args.crosses))

        # Override metadata from CLI
        if args.pixel_size:
            crosses_data["pixel_size_um"] = args.pixel_size
        if args.image_width:
            crosses_data["image_width_px"] = args.image_width
        if args.image_height:
            crosses_data["image_height_px"] = args.image_height

        # Handle display_transform in crosses JSON (from napari_place_crosses.py)
        # Cross coordinates in JSON are already in slide pixel space (inverse
        # transform applied during placement). The display_transform field is
        # informational only — no further conversion needed here.
        _dt = crosses_data.get("display_transform")
        if _dt:
            _transforms = []
            if _dt.get("flip_horizontal"):
                _transforms.append("flip_horizontal")
            if _dt.get("rotate_cw_90"):
                _transforms.append("rotate_cw_90")
            if _transforms:
                logger.info("Crosses placed with display transforms: %s", ", ".join(_transforms))
                logger.info("(coordinates already in slide pixel space)")

        # -------------------------------------------------------------------
        # Step 1: Separate singles vs clustered detections
        # -------------------------------------------------------------------
        if args.clusters:
            logger.info("Loading clusters from: %s", args.clusters)
            cluster_data = load_clusters(args.clusters)

            # Cluster indices reference the ORIGINAL unfiltered detections list
            # (cluster_detections.py runs its own score filtering internally)
            outlier_indices = [
                o.get("detection_index", o.get("nmj_index")) for o in cluster_data["outliers"]
            ]

            single_dets = [all_detections[i] for i in outlier_indices if i < len(all_detections)]
            cluster_groups = []
            for c in cluster_data["main_clusters"]:
                member_indices = c.get("detection_indices", c.get("nmj_indices", []))
                members = [all_detections[i] for i in member_indices if i < len(all_detections)]
                if members:
                    cluster_groups.append(
                        {
                            "id": c["id"],
                            "members": members,
                            "cx": c.get("cx", 0),
                            "cy": c.get("cy", 0),
                        }
                    )

            logger.info("Singles: %d", len(single_dets))
            logger.info(
                "Clusters: %d (%d detections)",
                len(cluster_groups),
                sum(len(cg["members"]) for cg in cluster_groups),
            )
        else:
            # No clusters file: all detections are singles
            single_dets = detections
            cluster_groups = []
            logger.info("All %d detections treated as singles (no --clusters)", len(single_dets))

        # -------------------------------------------------------------------
        # Step 2: Extract contours from H5 masks if needed
        # -------------------------------------------------------------------
        all_dets_needing_contours = list(single_dets)
        for cg in cluster_groups:
            all_dets_needing_contours.extend(cg["members"])

        # Promote pipeline contours → contour_um (handles both old and new field names)
        _promoted = 0
        for d in all_dets_needing_contours:
            if d.get("contour_um") is None:
                _um = d.get("contour_dilated_um")
                if _um is not None:
                    d["contour_um"] = _um
                    _promoted += 1
        if _promoted:
            logger.info("Used %d pre-processed contours from pipeline", _promoted)

        _max_area_change = getattr(args, "max_area_change_pct", 10.0)
        _max_dilation_area = getattr(args, "max_dilation_area_pct", 10.0)

        need_extraction = any(
            d.get("contour_um") is None
            and d.get("outer_contour_global") is None
            and d.get("contour_px") is None
            and d.get("contour_dilated_px") is None
            for d in all_dets_needing_contours
        )

        if need_extraction and args.tiles_dir:
            logger.info("Extracting contours from H5 masks (%s)...", args.tiles_dir)
            contour_results = extract_contours_for_detections(
                all_dets_needing_contours,
                args.tiles_dir,
                pixel_size,
                mask_filename=args.mask_filename,
                dilation_um=args.dilation_um,
                rdp_epsilon=args.rdp_epsilon,
                max_area_change_pct=_max_area_change,
                max_dilation_area_pct=_max_dilation_area,
            )
            logger.info("Extracted %d contours", len(contour_results))

            # Attach contours to detections
            for det in all_dets_needing_contours:
                uid = det.get("uid", det.get("id", ""))
                if uid in contour_results:
                    det["contour_um"] = contour_results[uid]["contour_um"]
                    det["area_um2"] = contour_results[uid]["area_um2"]
        elif need_extraction and not args.tiles_dir:
            # Try to process existing pixel-coord contours (vessel or cell pipeline)
            from xldvp_seg.lmd.contour_processing import process_contour

            logger.info("Processing existing contours (adaptive RDP + dilation)...")
            processed_count = 0
            for det in all_dets_needing_contours:
                if det.get("contour_um") is not None:
                    continue
                contour_px = (
                    det.get("outer_contour_global")
                    or det.get("contour_px")
                    or det.get("contour_dilated_px")
                )
                if contour_px is None:
                    continue
                processed, stats = process_contour(
                    contour_px,
                    pixel_size_um=pixel_size,
                    dilation_um=args.dilation_um,
                    max_area_change_pct=_max_area_change,
                    max_dilation_area_pct=_max_dilation_area,
                    return_stats=True,
                )
                if processed is not None:
                    det["contour_um"] = processed.tolist()
                    det["area_um2"] = stats["area_after_um2"]
                    processed_count += 1
            logger.info("Processed %d contours", processed_count)

        # -------------------------------------------------------------------
        # Step 2b: Apply export-time erosion to existing contours (if requested)
        # -------------------------------------------------------------------
        _erosion_um = getattr(args, "erosion_um", 0.0)
        _erode_pct = getattr(args, "erode_pct", 0.0)
        if _erosion_um > 0 or _erode_pct > 0:
            from shapely.geometry import Polygon

            from xldvp_seg.lmd.contour_processing import erode_contour, erode_contour_percent

            erode_label = (
                f"{_erosion_um}um" if _erosion_um > 0 else f"{_erode_pct*100:.1f}% of sqrt(area)"
            )
            logger.info("Applying export-time erosion (%s) to all contours...", erode_label)
            _eroded_count = 0
            _collapsed_count = 0
            for det in all_dets_needing_contours:
                contour_um = det.get("contour_um")
                if contour_um is None or len(contour_um) < 3:
                    continue

                pts = np.array(contour_um, dtype=np.float64)
                if _erosion_um > 0:
                    result = erode_contour(pts, _erosion_um)
                else:
                    result = erode_contour_percent(pts, _erode_pct)

                if result is not None:
                    det["contour_um"] = result.tolist()
                    poly = Polygon(result)
                    det["area_um2"] = poly.area if poly.is_valid else 0
                    _eroded_count += 1
                else:
                    # Contour collapsed — remove it
                    det["contour_um"] = None
                    _collapsed_count += 1

            logger.info(
                "Eroded %d contours, %d collapsed (removed)", _eroded_count, _collapsed_count
            )

        # -------------------------------------------------------------------
        # Step 3: Order singles by nearest-neighbor path
        # -------------------------------------------------------------------
        logger.info("Ordering singles by nearest-neighbor path on slide...")
        singles_with_contours = []
        singles_positions = []
        for det in single_dets:
            if det.get("contour_um") is None:
                continue
            singles_with_contours.append(det)
            coords = get_detection_coordinates(det)
            if coords:
                singles_positions.append((coords[0], coords[1]))
            else:
                singles_positions.append((0, 0))

        if singles_positions:
            nn_order = nearest_neighbor_order(singles_positions)
            ordered_singles_dets = [singles_with_contours[i] for i in nn_order]
            last_pos = singles_positions[nn_order[-1]]
            total_dist = sum(
                np.linalg.norm(
                    np.array(singles_positions[nn_order[i]])
                    - np.array(singles_positions[nn_order[i + 1]])
                )
                for i in range(len(nn_order) - 1)
            )
            logger.info(
                "Ordered %d singles, path: %.1f mm",
                len(ordered_singles_dets),
                total_dist * pixel_size / 1000,
            )
        else:
            ordered_singles_dets = []
            last_pos = (0, 0)

        # -------------------------------------------------------------------
        # Step 4: Order clusters by nearest-neighbor (from last single)
        # -------------------------------------------------------------------
        logger.info("Ordering clusters by nearest-neighbor path...")
        clusters_with_contours = []
        cluster_centroids = []

        for cg in cluster_groups:
            # Check all members have contours
            member_contours = []
            member_uids = []
            for m in cg["members"]:
                contour = m.get("contour_um")
                if contour is not None:
                    member_contours.append(contour)
                    member_uids.append(m.get("uid", m.get("id", "")))

            if not member_contours:
                logger.warning("Cluster %s has no contours, skipping entirely", cg["id"])
                continue

            dropped = len(cg["members"]) - len(member_contours)
            if dropped > 0:
                logger.warning(
                    "Cluster %s lost %d/%d members (contour extraction failed)",
                    cg["id"],
                    dropped,
                    len(cg["members"]),
                )

            clusters_with_contours.append(
                {
                    "id": cg["id"],
                    "members": cg["members"],
                    "member_contours_um": member_contours,
                    "member_uids": member_uids,
                    "cx": cg["cx"],
                    "cy": cg["cy"],
                }
            )
            cluster_centroids.append((cg["cx"], cg["cy"]))

        if cluster_centroids:
            # Start from nearest cluster to last single position
            dists = [np.linalg.norm(np.array(cc) - np.array(last_pos)) for cc in cluster_centroids]
            start_cluster = int(np.argmin(dists))
            cluster_order = nearest_neighbor_order(cluster_centroids, start_idx=start_cluster)
            ordered_clusters_data = [clusters_with_contours[i] for i in cluster_order]
            logger.info("Ordered %d clusters", len(ordered_clusters_data))
        else:
            ordered_clusters_data = []

        # -------------------------------------------------------------------
        # Well capacity check (before expensive control generation)
        # -------------------------------------------------------------------
        n_items = len(ordered_singles_dets) + len(ordered_clusters_data)
        n_wells_needed = n_items * 2 if args.generate_controls else n_items
        if n_wells_needed > WELLS_PER_PLATE:
            logger.error("=" * 70)
            logger.error("WELL CAPACITY EXCEEDED")
            logger.error("=" * 70)
            logger.error("Singles:  %d", len(ordered_singles_dets))
            logger.error("Clusters: %d", len(ordered_clusters_data))
            logger.error("Controls: %s", "yes (x2)" if args.generate_controls else "no")
            logger.error("Wells needed: %d", n_wells_needed)
            logger.error("Wells available: %d (384-well plate, 4 quadrants)", WELLS_PER_PLATE)
            logger.error("Overflow: %d wells", n_wells_needed - WELLS_PER_PLATE)
            logger.error("Options:")
            logger.error("1. Increase --min-score to reduce detections")
            logger.error("2. Split detections across multiple plates")
            logger.error("3. Run without --generate-controls (halves well usage)")
            logger.error("=" * 70)
            return

        # -------------------------------------------------------------------
        # Step 5: Generate controls
        # -------------------------------------------------------------------
        if args.generate_controls:
            logger.info("Generating controls (offset: %s um)...", args.control_offset_um)
            logger.info("Every sample will have a control.")

            # Precompute Shapely polygons for all detection contours (avoids
            # recreating Polygon objects on every overlap check)
            precomputed_polygons = []
            for det in ordered_singles_dets:
                c = det.get("contour_um")
                if c and len(c) >= 3:
                    precomputed_polygons.append(_make_polygon(c))

            for cdata in ordered_clusters_data:
                for c in cdata["member_contours_um"]:
                    if c and len(c) >= 3:
                        precomputed_polygons.append(_make_polygon(c))

            logger.info("Precomputed %d collision polygons", len(precomputed_polygons))

            # Build spatial index for O(log N) overlap checks.
            # STRtree is static (built once from all detection polygons).
            # New controls are NOT added to the index — this is acceptable
            # since controls are offset in different directions and unlikely
            # to overlap each other.
            detection_tree = _build_spatial_index(precomputed_polygons)
            if detection_tree is not None:
                logger.info("Using STRtree spatial index for overlap checks")
            else:
                logger.info(
                    "STRtree unavailable, using linear overlap scan"
                    " (install shapely>=2.0 for speedup)"
                )

            # Generate single controls
            ordered_single_ctrls = []
            fallback_count = 0
            for det in ordered_singles_dets:
                contour_um = det.get("contour_um")
                shifted, direction, actual_offset = generate_spatial_control(
                    contour_um,
                    precomputed_polygons,
                    offset_um=args.control_offset_um,
                    spatial_tree=detection_tree,
                )
                if "fallback" in direction:
                    fallback_count += 1

                uid = det.get("uid", det.get("id", ""))
                ordered_single_ctrls.append(
                    {
                        "type": "single_control",
                        "uid": uid + "_ctrl",
                        "control_of": uid,
                        "contour_um": shifted,
                        "offset_direction": direction,
                        "offset_um": actual_offset,
                        "area_um2": det.get("area_um2", 0),
                    }
                )

            logger.info(
                "Single controls: %d (%d used fallback offset)",
                len(ordered_single_ctrls),
                fallback_count,
            )

            # Generate cluster controls
            ordered_cluster_ctrls = []
            cluster_fallback = 0
            for cdata in ordered_clusters_data:
                shifted_contours, direction, actual_offset = generate_cluster_control(
                    cdata["member_contours_um"],
                    precomputed_polygons,
                    offset_um=args.control_offset_um,
                    spatial_tree=detection_tree,
                )
                if "fallback" in direction:
                    cluster_fallback += 1

                ordered_cluster_ctrls.append(
                    {
                        "type": "cluster_control",
                        "uid": f"cluster_{cdata['id']}_ctrl",
                        "control_of_cluster": cdata["id"],
                        "contours_um": shifted_contours,
                        "offset_direction": direction,
                        "offset_um": actual_offset,
                    }
                )

            logger.info(
                "Cluster controls: %d (%d used fallback offset)",
                len(ordered_cluster_ctrls),
                cluster_fallback,
            )
        else:
            ordered_single_ctrls = []
            ordered_cluster_ctrls = []

        # -------------------------------------------------------------------
        # Step 6: Build shape dicts for export
        # -------------------------------------------------------------------
        # Convert singles to shape dicts
        ordered_singles = []
        for det in ordered_singles_dets:
            uid = det.get("uid", det.get("id", ""))
            ordered_singles.append(
                {
                    "type": "single",
                    "uid": uid,
                    "contour_um": det.get("contour_um"),
                    "area_um2": det.get("area_um2", 0),
                    "global_center": det.get("global_center"),
                }
            )

        # Convert clusters to shape dicts
        ordered_clusters = []
        for cdata in ordered_clusters_data:
            total_area = sum(
                m.get("area_um2", 0) for m in cdata["members"] if m.get("contour_um") is not None
            )
            ordered_clusters.append(
                {
                    "type": "cluster",
                    "uid": f"cluster_{cdata['id']}",
                    "cluster_id": cdata["id"],
                    "n_members": len(cdata["member_contours_um"]),
                    "contours_um": cdata["member_contours_um"],
                    "member_uids": cdata["member_uids"],
                    "total_area_um2": total_area,
                    "cx": cdata["cx"],
                    "cy": cdata["cy"],
                }
            )

        # -------------------------------------------------------------------
        # Step 7: Assign wells
        # -------------------------------------------------------------------
        try:
            if args.generate_controls:
                logger.info(
                    "Assigning wells (serpentine, B2->B3->C3->C2, alternating target/control)..."
                )
                assignments, well_order = assign_wells_with_controls(
                    ordered_singles,
                    ordered_single_ctrls,
                    ordered_clusters,
                    ordered_cluster_ctrls,
                )
            else:
                # No controls: just assign sequentially
                all_shapes = ordered_singles + ordered_clusters
                n_wells = len(all_shapes)
                well_order = generate_plate_wells(n_wells)
                for i, shape in enumerate(all_shapes):
                    shape["well"] = well_order[i] if i < len(well_order) else f"overflow_{i}"
                assignments = all_shapes
        except ValueError as e:
            # Well overflow — save partial results (shapes without well assignments)
            logger.error("%s", e)
            partial_shapes = (
                ordered_singles + ordered_single_ctrls + ordered_clusters + ordered_cluster_ctrls
            )
            partial_path = output_dir / f"{args.output_name}_NO_WELLS.json"
            partial_data = {
                "metadata": {
                    "cell_type": args.cell_type or "unknown",
                    "error": str(e),
                    "n_singles": len(ordered_singles),
                    "n_clusters": len(ordered_clusters),
                    "controls": args.generate_controls,
                },
                "shapes": partial_shapes,
            }
            from xldvp_seg.utils.json_utils import atomic_json_dump

            atomic_json_dump(partial_data, partial_path)
            logger.info("Partial results saved to: %s", partial_path)
            logger.info("(shapes without well assignments -- prune list and re-run)")
            return

        logger.info("Total wells used: %d", len(well_order))
        if well_order:
            logger.info("First well: %s, Last well: %s", well_order[0], well_order[-1])

        # -------------------------------------------------------------------
        # Step 8: Build and save export data
        # -------------------------------------------------------------------
        metadata = {
            "cell_type": args.cell_type or "unknown",
            "plate_format": "384",
            "quadrant_order": ["B2", "B3", "C3", "C2"],
            "pixel_size_um": pixel_size,
            "dilation_um": args.dilation_um,
            "rdp_epsilon_px": args.rdp_epsilon,
            "control_offset_um": args.control_offset_um,
        }

        export_data = build_export_data(assignments, well_order, metadata)

        # Save JSON (timestamped + symlink)
        from xldvp_seg.utils.timestamps import timestamped_path, update_symlink

        json_path = output_dir / f"{args.output_name}_with_controls.json"
        ts_json = timestamped_path(json_path)
        atomic_json_dump(export_data, ts_json)
        update_symlink(json_path, ts_json)
        logger.info("Saved export JSON: %s", ts_json)

        # Save CSV summary (timestamped + symlink)
        csv_path = output_dir / f"{args.output_name}_summary.csv"
        ts_csv = timestamped_path(csv_path)
        with open(ts_csv, "w") as f:
            f.write("well,type,uid,area_um2,n_contours,offset_direction\n")
            for shape in assignments:
                well = shape.get("well", "")
                stype = shape.get("type", "")
                uid = shape.get("uid", "")
                area = shape.get("area_um2", shape.get("total_area_um2", 0))
                n_contours = len(shape.get("contours_um", [])) or (
                    1 if shape.get("contour_um") else 0
                )
                direction = shape.get("offset_direction", "")
                f.write(f"{well},{stype},{uid},{area:.2f},{n_contours},{direction}\n")
        update_symlink(csv_path, ts_csv)
        logger.info("Saved CSV summary: %s", ts_csv)

        # Export LMD XML (timestamped + symlink)
        xml_path = output_dir / f"{args.output_name}.xml"
        ts_xml = timestamped_path(xml_path)
        try:
            export_to_lmd_xml(
                assignments,
                crosses_data,
                ts_xml,
                flip_y=not args.no_flip_y,
            )
            update_symlink(xml_path, ts_xml)
            logger.info("Saved LMD XML: %s", ts_xml)
        except ImportError:
            logger.warning("py-lmd not installed, skipping XML export.")
            logger.warning("Install with: pip install py-lmd")
        except Exception as e:
            logger.warning("XML export failed: %s", e)

        # Print summary
        s = export_data["summary"]
        logger.info("=" * 60)
        logger.info("EXPORT SUMMARY")
        logger.info("=" * 60)
        logger.info("Singles:          %d", s["n_singles"])
        logger.info("Single controls:  %d", s["n_single_controls"])
        logger.info("Clusters:         %d", s["n_clusters"])
        logger.info("Cluster controls: %d", s["n_cluster_controls"])
        logger.info("Detections in clusters: %d", s["n_detections_in_clusters"])
        logger.info("Total wells used: %d", s["total_wells_used"])
        logger.info("=" * 60)

    if not args.generate_cross_html and not args.export:
        logger.info("No action specified. Use --generate-cross-html or --export")


def run_batch_export(args):
    """Run LMD export for multiple slides discovered from --input-dir."""
    import copy
    import re

    input_dir = Path(args.input_dir)
    crosses_dir = Path(args.crosses_dir) if args.crosses_dir else input_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover detection files
    det_files = sorted(input_dir.glob("*_detections.json"))
    if not det_files:
        det_files = sorted(input_dir.glob("*_detections_postdedup.json"))
    if not det_files:
        logger.error("No *_detections.json files found in %s", input_dir)
        return

    logger.info("Batch mode: found %d detection files in %s", len(det_files), input_dir)

    batch_results = []
    for det_file in det_files:
        # Extract slide name from filename (e.g., "nmj_detections.json" -> "nmj")
        slide_prefix = re.sub(r"_(detections|postdedup)+", "", det_file.stem)

        # Find matching crosses file (prefer slide-specific, fall back to shared)
        crosses_file = None
        for pattern in [f"{slide_prefix}_crosses.json", f"{slide_prefix}*crosses*.json"]:
            matches = list(crosses_dir.glob(pattern))
            if matches:
                crosses_file = matches[0]
                break

        # Shared fallback — warn the user
        if crosses_file is None:
            for pattern in ["reference_crosses.json", "crosses.json"]:
                matches = list(crosses_dir.glob(pattern))
                if matches:
                    crosses_file = matches[0]
                    logger.info(
                        "NOTE: Using shared crosses file %s for %s",
                        crosses_file.name,
                        slide_prefix,
                    )
                    break

        if crosses_file is None and args.crosses:
            crosses_file = Path(args.crosses)

        if crosses_file is None or not crosses_file.exists():
            logger.warning("No crosses file for %s, skipping", slide_prefix)
            batch_results.append(
                {"slide": slide_prefix, "status": "skipped", "reason": "no crosses file"}
            )
            continue

        slide_output = output_dir / slide_prefix
        slide_output.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Slide: %s", slide_prefix)
        logger.info("Detections: %s", det_file)
        logger.info("Crosses: %s", crosses_file)
        logger.info("Output: %s", slide_output)
        logger.info("=" * 60)

        # Create per-slide args
        slide_args = copy.deepcopy(args)
        slide_args.detections = str(det_file)
        slide_args.crosses = str(crosses_file)
        slide_args.output_dir = str(slide_output)
        # Remove batch flags to avoid recursion
        slide_args.input_dir = None

        try:
            _run_single_slide(slide_args)
            batch_results.append({"slide": slide_prefix, "status": "success"})
        except Exception as e:
            logger.error("%s", e)
            batch_results.append({"slide": slide_prefix, "status": "failed", "reason": str(e)})

    # Write batch summary
    summary_path = output_dir / "batch_summary.json"
    from xldvp_seg.utils.json_utils import atomic_json_dump

    atomic_json_dump(
        {
            "n_slides": len(det_files),
            "n_success": sum(1 for r in batch_results if r["status"] == "success"),
            "n_failed": sum(1 for r in batch_results if r["status"] == "failed"),
            "n_skipped": sum(1 for r in batch_results if r["status"] == "skipped"),
            "results": batch_results,
        },
        summary_path,
    )

    logger.info("=" * 60)
    logger.info("BATCH SUMMARY: %s", summary_path)
    logger.info("=" * 60)
    for r in batch_results:
        status = r["status"].upper()
        reason = f" ({r.get('reason', '')})" if r.get("reason") else ""
        logger.info("%s: %s%s", r["slide"], status, reason)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
