#!/usr/bin/env python3
"""Assign cells to concentric distance bins around vascular landmarks.

For each cell, computes distance to nearest central vein (CV) and nearest
portal vein (PV), then bins into annular rings. Supports CV-centered rings,
PV-centered rings, or both.

Used to test classical lobule (CV-centered) and portal lobule (PV-centered)
spatial organization models via LMD sampling.

Usage:
    # CV-centered rings (classical lobule model)
    python scripts/assign_distance_bins.py \\
        --detections cell_detections_classified.json \\
        --landmarks zonation_landmarks.json \\
        --center cv \\
        --bin-edges 0,50,100,150,200 \\
        --output-dir rings_cv/

    # PV-centered rings
    python scripts/assign_distance_bins.py \\
        --detections cell_detections_classified.json \\
        --landmarks zonation_landmarks.json \\
        --center pv \\
        --bin-edges 0,50,100,150,200 \\
        --output-dir rings_pv/

    # Both + normalized ratio r = d_PV / (d_PV + d_CV) for model comparison
    python scripts/assign_distance_bins.py \\
        --detections cell_detections_classified.json \\
        --landmarks zonation_landmarks.json \\
        --center both \\
        --bin-edges 0,50,100,150,200 \\
        --output-dir rings_both/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
from scipy.spatial import cKDTree

from segmentation.utils.json_utils import fast_json_load, atomic_json_dump
from segmentation.utils.logging import get_logger

logger = get_logger(__name__)


def load_landmarks(landmarks_path):
    """Load CV and PV centroids from zonation_landmarks.json.

    Returns (cv_coords_um, pv_coords_um) as numpy arrays of shape (N, 2).
    """
    data = fast_json_load(str(landmarks_path))

    cv_coords = []
    pv_coords = []

    # Format from zonation_transect.py: {"cv_clusters": [...], "pv_clusters": [...]}
    for cluster in data.get("cv_clusters", []):
        cx = cluster.get("centroid_um", cluster.get("centroid", [0, 0]))
        cv_coords.append([float(cx[0]), float(cx[1])])

    for cluster in data.get("pv_clusters", []):
        cx = cluster.get("centroid_um", cluster.get("centroid", [0, 0]))
        pv_coords.append([float(cx[0]), float(cx[1])])

    return np.array(cv_coords), np.array(pv_coords)


def compute_distances(cell_coords_um, cv_coords_um, pv_coords_um):
    """Compute distance to nearest CV and PV for each cell.

    Returns (d_cv, d_pv, ratio, nearest_cv_idx, nearest_pv_idx) arrays.
    ratio = d_pv / (d_pv + d_cv), where 0 = at PV, 1 = at CV.
    """
    cv_tree = cKDTree(cv_coords_um)
    pv_tree = cKDTree(pv_coords_um)

    d_cv, nearest_cv = cv_tree.query(cell_coords_um)
    d_pv, nearest_pv = pv_tree.query(cell_coords_um)

    # Normalized ratio: 0 = at PV (periportal), 1 = at CV (pericentral)
    total = d_pv + d_cv
    ratio = np.where(total > 0, d_pv / total, 0.5)

    return d_cv, d_pv, ratio, nearest_cv, nearest_pv


def bin_distances(distances, bin_edges):
    """Assign each distance to a bin. Returns bin labels (str) and bin indices."""
    labels = []
    indices = []
    for d in distances:
        assigned = False
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= d < bin_edges[i + 1]:
                labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i+1])}um")
                indices.append(i)
                assigned = True
                break
        if not assigned:
            labels.append("outside")
            indices.append(-1)
    return labels, indices


def main():
    parser = argparse.ArgumentParser(
        description="Assign cells to concentric distance bins around vascular landmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--detections", required=True, type=Path,
                        help="Cell detections JSON (classified)")
    parser.add_argument("--landmarks", required=True, type=Path,
                        help="Zonation landmarks JSON (from zonation_transect.py)")
    parser.add_argument("--center", default="both", choices=["cv", "pv", "both"],
                        help="Center rings on CV, PV, or both (default: both)")
    parser.add_argument("--bin-edges", default="0,50,100,150,200",
                        help="Comma-separated bin edges in µm (default: 0,50,100,150,200)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Min RF score (default: 0.0 = all)")
    parser.add_argument("--score-key", default="rf_prediction")
    parser.add_argument("--every-nth", type=int, default=None,
                        help="Subsample every Nth cell per bin (for LMD well count control)")
    parser.add_argument("--max-per-landmark", type=int, default=None,
                        help="Max cells per bin per landmark. Ensures spatial distribution "
                             "across multiple lobules instead of clustering near one CV/PV.")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = [float(x) for x in args.bin_edges.split(",")]
    logger.info(f"Bin edges: {bin_edges} µm")

    # --- Load landmarks ---
    logger.info(f"Loading landmarks: {args.landmarks}")
    cv_coords, pv_coords = load_landmarks(args.landmarks)
    logger.info(f"  {len(cv_coords)} CVs, {len(pv_coords)} PVs")

    if len(cv_coords) == 0 or len(pv_coords) == 0:
        logger.error("Need both CV and PV landmarks")
        sys.exit(1)

    # --- Load detections ---
    logger.info(f"Loading detections: {args.detections}")
    detections = fast_json_load(str(args.detections))
    if isinstance(detections, dict):
        detections = detections.get("detections", [])
    logger.info(f"  {len(detections):,} detections")

    # --- Extract coordinates in µm ---
    cell_coords = []
    valid_indices = []
    for i, det in enumerate(detections):
        gc_um = det.get("global_center_um")
        if gc_um and len(gc_um) >= 2:
            # Score filter
            score = det.get(args.score_key) or det.get("features", {}).get(args.score_key)
            if score is not None and float(score) < args.score_threshold:
                continue
            cell_coords.append([float(gc_um[0]), float(gc_um[1])])
            valid_indices.append(i)

    cell_coords = np.array(cell_coords)
    logger.info(f"  {len(cell_coords):,} cells with coordinates (after score filter)")

    # --- Compute distances ---
    d_cv, d_pv, ratio, nearest_cv, nearest_pv = compute_distances(cell_coords, cv_coords, pv_coords)

    # --- Bin cells ---
    results = {}

    if args.center in ("cv", "both"):
        cv_labels, cv_bins = bin_distances(d_cv, bin_edges)
        results["cv"] = {"labels": cv_labels, "bins": cv_bins, "distances": d_cv,
                         "nearest_landmark": nearest_cv}

    if args.center in ("pv", "both"):
        pv_labels, pv_bins = bin_distances(d_pv, bin_edges)
        results["pv"] = {"labels": pv_labels, "bins": pv_bins, "distances": d_pv,
                         "nearest_landmark": nearest_pv}

    # --- Enrich detections and build per-bin outputs ---
    for center_type, data in results.items():
        labels = data["labels"]
        distances = data["distances"]
        nearest_lm = data["nearest_landmark"]

        # Group by (bin, landmark) for spatially distributed sampling
        bin_landmark = {}  # (label, landmark_idx) -> [det]
        for j, idx in enumerate(valid_indices):
            det = detections[idx]
            label = labels[j]
            if label == "outside":
                continue

            # Enrich detection
            features = det.get("features", {})
            features[f"d_{center_type}_um"] = round(float(distances[j]), 2)
            features["d_cv_um"] = round(float(d_cv[j]), 2)
            features["d_pv_um"] = round(float(d_pv[j]), 2)
            features["zonation_ratio"] = round(float(ratio[j]), 4)
            features[f"{center_type}_ring"] = label
            features[f"nearest_{center_type}_idx"] = int(nearest_lm[j])
            det["features"] = features

            lm_idx = int(nearest_lm[j])
            bin_landmark.setdefault((label, lm_idx), []).append(det)

        # Flatten with spatial distribution: max N cells per landmark per bin
        bins = {}
        for (label, lm_idx), dets in bin_landmark.items():
            if args.max_per_landmark and len(dets) > args.max_per_landmark:
                # Take evenly spaced subset from this landmark's cells
                step = max(1, len(dets) // args.max_per_landmark)
                dets = dets[::step][:args.max_per_landmark]
            bins.setdefault(label, []).extend(dets)

        # Additional global subsampling if requested
        if args.every_nth and args.every_nth > 1:
            for label in bins:
                bins[label] = bins[label][::args.every_nth]

        # Write per-bin and combined
        all_binned = []
        logger.info(f"\n{center_type.upper()}-centered rings:")
        for label in sorted(bins.keys()):
            dets = bins[label]
            all_binned.extend(dets)
            logger.info(f"  {label}: {len(dets):,} cells")

        out_path = args.output_dir / f"rings_{center_type}.json"
        atomic_json_dump(all_binned, str(out_path))
        logger.info(f"  Wrote: {out_path} ({len(all_binned):,} cells)")

    # --- Summary with model comparison features ---
    summary = {
        "n_cv_landmarks": len(cv_coords),
        "n_pv_landmarks": len(pv_coords),
        "n_cells_total": len(detections),
        "n_cells_with_coords": len(cell_coords),
        "bin_edges_um": bin_edges,
        "center_types": list(results.keys()),
        "per_bin_counts": {},
    }
    for ct, data in results.items():
        from collections import Counter
        counts = Counter(data["labels"])
        summary["per_bin_counts"][ct] = {k: v for k, v in sorted(counts.items())}

    # Distance stats
    summary["d_cv_stats"] = {
        "mean": round(float(d_cv.mean()), 1),
        "median": round(float(np.median(d_cv)), 1),
        "std": round(float(d_cv.std()), 1),
    }
    summary["d_pv_stats"] = {
        "mean": round(float(d_pv.mean()), 1),
        "median": round(float(np.median(d_pv)), 1),
        "std": round(float(d_pv.std()), 1),
    }
    summary["ratio_stats"] = {
        "mean": round(float(ratio.mean()), 3),
        "median": round(float(np.median(ratio)), 3),
        "std": round(float(ratio.std()), 3),
    }

    atomic_json_dump(summary, str(args.output_dir / "distance_bins_summary.json"))
    logger.info(f"\nSummary: {args.output_dir / 'distance_bins_summary.json'}")


if __name__ == "__main__":
    main()
