#!/usr/bin/env python3
"""Hepatic zonation transect analysis.

Loads ROIs exported from the spatial viewer (JSON) and classified detections,
finds cells along drawn paths (central vein -> portal vein), computes
fractional position along each path (0=CV, 1=PV), and generates gradient
plots of marker expression vs zonation position.

Usage:
    python examples/liver/zonation_transect.py \\
        --detections cell_detections_classified.json \\
        --rois rois.json \\
        --markers GluI,Pck1,DCN \\
        --output-dir zonation_output/
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from xldvp_seg.utils.json_utils import atomic_json_dump, fast_json_load
from xldvp_seg.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ===================================================================
# ROI loading
# ===================================================================


def load_rois(path):
    """Load ROIs JSON exported from the spatial viewer."""
    data = fast_json_load(str(path))
    if isinstance(data, dict) and "rois" in data:
        return data
    raise ValueError(f"Invalid ROI file: expected dict with 'rois' key, got {type(data)}")


def extract_paths(rois_data):
    """Extract path ROIs as structured dicts."""
    paths = []
    for roi in rois_data.get("rois", []):
        if roi.get("type") != "path":
            continue
        waypoints = roi.get("waypoints_um")
        if waypoints is None or len(waypoints) < 2:
            logger.warning(f"Skipping path {roi.get('id')}: <2 waypoints")
            continue
        paths.append(
            {
                "id": roi.get("id", ""),
                "name": roi.get("name", roi.get("id", "unnamed")),
                "waypoints": np.array(waypoints, dtype=np.float64),
                "corridor_um": roi.get("corridor_um", 100.0),
                "slide": roi.get("slide", ""),
                "category": roi.get("category", ""),
            }
        )
    return paths


def extract_liver_polygon(rois_data, category="liver"):
    """Find liver boundary polygon ROI by category.

    Checks both polygon (vertices_um) and closed path (waypoints_um) ROIs.
    If no ROI has the matching category, falls back to the first polygon/closed-path
    ROI found (useful when the user draws a boundary without setting category).
    """
    fallback = None
    for roi in rois_data.get("rois", []):
        # Try vertices_um (polygon ROI) or waypoints_um (path ROI used as boundary)
        verts = roi.get("vertices_um") or roi.get("waypoints_um")
        if not verts or len(verts) < 3:
            continue
        arr = np.array(verts, dtype=np.float64)
        # Deduplicate trailing repeated points (dblclick artifact)
        while len(arr) > 3 and np.allclose(arr[-1], arr[-2], atol=1e-6):
            arr = arr[:-1]
        if roi.get("category", "").lower() == category.lower():
            return arr
        if fallback is None:
            fallback = arr
    return fallback


# ===================================================================
# Position extraction
# ===================================================================


def extract_position_um(det):
    """Extract (x, y) position in microns from a detection dict.

    Tries global_center_um first, then global_x/global_y * pixel_size.
    """
    pos = det.get("features", {}).get("global_center_um")
    if pos is None:
        pos = det.get("global_center_um")
    if pos is not None and len(pos) == 2:
        x, y = float(pos[0]), float(pos[1])
        if np.isfinite(x) and np.isfinite(y):
            return (x, y)

    gx = det.get("global_x")
    gy = det.get("global_y")
    if gx is not None and gy is not None:
        pixel_size = det.get("features", {}).get("pixel_size_um")
        if pixel_size and isinstance(pixel_size, (int, float)):
            x = float(gx) * float(pixel_size)
            y = float(gy) * float(pixel_size)
            if np.isfinite(x) and np.isfinite(y):
                return (x, y)
    return None


# ===================================================================
# Core geometry: project cells onto path
# ===================================================================


def project_cells_onto_path(positions, waypoints, corridor_half_width):
    """Project cells onto a polyline path and compute fractional position.

    Parameters
    ----------
    positions : ndarray (N, 2) — cell positions in micrometers
    waypoints : ndarray (M, 2) — path waypoints in micrometers
    corridor_half_width : float — half-width of corridor in micrometers

    Returns
    -------
    in_corridor : ndarray (N,) bool — True if cell is within corridor
    frac_pos : ndarray (N,) float — fractional position along path [0, 1]
    """
    N = len(positions)
    M = len(waypoints)

    # Compute segment vectors and cumulative lengths
    segments = waypoints[1:] - waypoints[:-1]  # (M-1, 2)
    seg_lengths = np.linalg.norm(segments, axis=1)  # (M-1,)
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])  # (M,)
    total_length = cum_lengths[-1]

    if total_length < 1e-12:
        return np.zeros(N, dtype=bool), np.zeros(N, dtype=np.float64)

    hw2 = corridor_half_width**2

    best_dist_sq = np.full(N, np.inf, dtype=np.float64)
    best_frac = np.zeros(N, dtype=np.float64)

    for i in range(M - 1):
        a = waypoints[i]  # (2,)
        ab = segments[i]  # (2,)
        len2 = seg_lengths[i] ** 2
        if len2 < 1e-12:
            continue

        # Vectorized projection: t = dot(P-A, AB) / |AB|^2, clamped to [0,1]
        ap = positions - a  # (N, 2)
        t = np.sum(ap * ab, axis=1) / len2  # (N,)
        t = np.clip(t, 0, 1)

        # Projection point and distance
        proj = a + t[:, None] * ab  # (N, 2)
        diff = positions - proj
        dist_sq = np.sum(diff**2, axis=1)  # (N,)

        # Update best (closest segment)
        closer = dist_sq < best_dist_sq
        best_dist_sq[closer] = dist_sq[closer]
        best_frac[closer] = (cum_lengths[i] + t[closer] * seg_lengths[i]) / total_length

    in_corridor = best_dist_sq <= hw2
    return in_corridor, best_frac


# ===================================================================
# Auto-landmark detection from expression patterns
# ===================================================================


def find_marker_clusters(positions, positive_mask, min_cells=20, radius_um=100):
    """Find dense clusters of marker-positive cells as vein landmarks.

    Uses connected components at `radius_um` to find spatial clusters
    of positive cells. Returns centroids of clusters with >= min_cells.

    Parameters
    ----------
    positions : ndarray (N, 2) — all cell positions
    positive_mask : ndarray (N,) bool — which cells are marker-positive
    min_cells : int — minimum cluster size
    radius_um : float — connection radius for clustering

    Returns
    -------
    centroids : ndarray (K, 2) — cluster center positions
    labels : ndarray (N_pos,) — cluster label per positive cell (-1 = too small)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    pos_positions = positions[positive_mask]
    n_pos = len(pos_positions)
    if n_pos < min_cells:
        return np.empty((0, 2)), np.full(n_pos, -1, dtype=int)

    tree = cKDTree(pos_positions)
    pairs = tree.query_pairs(r=radius_um)

    if not pairs:
        return np.empty((0, 2)), np.full(n_pos, -1, dtype=int)

    rows, cols = zip(*pairs)
    rows, cols = np.array(rows), np.array(cols)
    data = np.ones(len(rows) * 2)
    row_idx = np.concatenate([rows, cols])
    col_idx = np.concatenate([cols, rows])
    adj = csr_matrix((data, (row_idx, col_idx)), shape=(n_pos, n_pos))

    n_comp, labels = connected_components(adj, directed=False)

    centroids = []
    final_labels = np.full(n_pos, -1, dtype=int)
    cluster_id = 0
    for c in range(n_comp):
        mask = labels == c
        if mask.sum() >= min_cells:
            centroids.append(pos_positions[mask].mean(axis=0))
            final_labels[mask] = cluster_id
            cluster_id += 1

    if not centroids:
        return np.empty((0, 2)), final_labels

    return np.array(centroids), final_labels


def compute_zonation_score(positions, cv_centroids, pv_centroids):
    """Compute zonation score for all cells based on proximity to landmarks.

    Score = d_cv / (d_cv + d_pv) where:
      d_cv = distance to nearest central vein cluster
      d_pv = distance to nearest portal vein cluster

    Returns 0 (pericentral) to 1 (periportal).
    """
    cv_tree = cKDTree(cv_centroids)
    pv_tree = cKDTree(pv_centroids)

    d_cv, _ = cv_tree.query(positions, k=1)
    d_pv, _ = pv_tree.query(positions, k=1)

    # Avoid division by zero for cells exactly on a landmark
    total = d_cv + d_pv
    total = np.maximum(total, 1e-12)
    score = d_cv / total

    return score, d_cv, d_pv


def plot_zonation_map(
    positions,
    scores,
    cv_centroids,
    pv_centroids,
    output_dir,
    title="Auto-detected zonation",
    paths=None,
    corridor_um=150,
):
    """Spatial scatter plot colored by zonation score with CV-PV paths."""
    fig, ax = plt.subplots(figsize=(14, 10))

    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=scores,
        cmap="RdYlBu_r",
        s=0.5,
        alpha=0.6,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="Zonation score (0=CV, 1=PV)", shrink=0.8)

    # Draw path corridors and centerlines
    if paths:
        for pi, p in enumerate(paths):
            wp = p["waypoints"]
            # Bright yellow-green centerline with white outline for visibility
            ax.plot(
                wp[:, 0],
                wp[:, 1],
                color="white",
                linewidth=2.0,
                alpha=0.9,
                solid_capstyle="round",
                zorder=6,
            )
            ax.plot(wp[:, 0], wp[:, 1], color="#00ff00", linewidth=1.0, alpha=0.8, zorder=7)

    # Mark landmarks
    if len(cv_centroids) > 0:
        ax.scatter(
            cv_centroids[:, 0],
            cv_centroids[:, 1],
            c="blue",
            s=100,
            marker="*",
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
            label=f"CV ({len(cv_centroids)})",
        )
    if len(pv_centroids) > 0:
        ax.scatter(
            pv_centroids[:, 0],
            pv_centroids[:, 1],
            c="red",
            s=100,
            marker="*",
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
            label=f"PV ({len(pv_centroids)})",
        )

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(f"{title} ({len(paths or [])} paths)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.invert_yaxis()

    out_path = Path(output_dir) / "zonation_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def plot_zone_map(positions, scores, cv_centroids, pv_centroids, output_dir, paths=None, n_zones=3):
    """Discrete zone map: cells colored by pericentral/mid-zone/periportal with
    labeled vein landmarks connected by routes. Axes clipped to data extent."""
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    # --- Discretize into zones ---
    zone_thresholds = np.linspace(0, 1, n_zones + 1)
    zone_labels = np.digitize(scores, zone_thresholds[1:-1])  # 0..n_zones-1

    zone_names = ["Pericentral (Zone 1)", "Mid-zone (Zone 2)", "Periportal (Zone 3)"]
    if n_zones > 3:
        zone_names = [f"Zone {i+1}" for i in range(n_zones)]
    zone_colors = ["#4a90d9", "#f5e663", "#d94040"]  # blue, gold, red
    if n_zones > 3:
        cmap_temp = plt.cm.get_cmap("RdYlBu_r", n_zones)
        zone_colors = [cmap_temp(i / (n_zones - 1)) for i in range(n_zones)]
    cmap = ListedColormap(zone_colors[:n_zones])

    fig, ax = plt.subplots(figsize=(14, 16))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Cells colored by discrete zone
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=zone_labels,
        cmap=cmap,
        s=1.0,
        alpha=0.6,
        vmin=-0.5,
        vmax=n_zones - 0.5,
        rasterized=True,
        zorder=1,
    )

    # --- Routes (thin, semi-transparent) ---
    if paths:
        for p in paths:
            wp = p["waypoints"]
            ax.plot(
                wp[:, 0],
                wp[:, 1],
                color="white",
                linewidth=0.6,
                alpha=0.25,
                solid_capstyle="round",
                zorder=4,
            )

    # --- Vein landmarks ---
    if len(cv_centroids) > 0:
        ax.scatter(
            cv_centroids[:, 0],
            cv_centroids[:, 1],
            c="#60b0ff",
            s=80,
            marker="o",
            edgecolors="white",
            linewidths=1.0,
            zorder=10,
        )
        for i, (x, y) in enumerate(cv_centroids):
            ax.annotate(
                "CV",
                xy=(x, y),
                fontsize=4.5,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=11,
            )
    if len(pv_centroids) > 0:
        ax.scatter(
            pv_centroids[:, 0],
            pv_centroids[:, 1],
            c="#ff6060",
            s=80,
            marker="s",
            edgecolors="white",
            linewidths=1.0,
            zorder=10,
        )
        for i, (x, y) in enumerate(pv_centroids):
            ax.annotate(
                "PV",
                xy=(x, y),
                fontsize=4.5,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=11,
            )

    # --- Clip axes to data extent ---
    margin = 300  # um
    xmin, xmax = positions[:, 0].min() - margin, positions[:, 0].max() + margin
    ymin, ymax = positions[:, 1].min() - margin, positions[:, 1].max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # --- Legend ---
    patches = [
        mpatches.Patch(facecolor=zone_colors[i], label=zone_names[i], edgecolor="gray")
        for i in range(min(n_zones, len(zone_names)))
    ]
    patches.append(mpatches.Patch(facecolor="#60b0ff", label=f"Central Vein ({len(cv_centroids)})"))
    patches.append(mpatches.Patch(facecolor="#ff6060", label=f"Portal Vein ({len(pv_centroids)})"))
    if paths:
        patches.append(
            plt.Line2D(
                [0], [0], color="white", linewidth=1.5, alpha=0.5, label=f"Routes ({len(paths)})"
            )
        )
    ax.legend(
        handles=patches,
        loc="upper right",
        fontsize=9,
        framealpha=0.8,
        facecolor="#2a2a4e",
        edgecolor="gray",
        labelcolor="white",
    )

    ax.set_xlabel("x (um)", color="white")
    ax.set_ylabel("y (um)", color="white")
    ax.set_title(
        f"Hepatic Zonation — {len(cv_centroids)} CV, {len(pv_centroids)} PV, "
        f"{len(paths or [])} routes",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("gray")

    out_path = Path(output_dir) / "zone_map.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ===================================================================
# Marker extraction
# ===================================================================


def discover_markers(detections):
    """Auto-discover marker names from detection features."""
    markers = []
    if not detections:
        return markers
    feat = detections[0].get("features", {})
    for key in feat:
        if key.endswith("_class"):
            name = key[: -len("_class")]
            # Only count as marker if it has a _value key (from classify_markers.py)
            if f"{name}_value" in feat:
                markers.append(name)
    return sorted(markers)


def extract_marker_values(detections, indices, markers):
    """Extract marker values for selected detections.

    Tries {marker}_value first (from classify_markers.py),
    then falls back to ch{N}_{feature}.
    """
    result = {}
    for marker in markers:
        values = []
        for idx in indices:
            feat = detections[idx].get("features", {})
            # Primary: classify_markers.py output
            val = feat.get(f"{marker}_value")
            if val is None:
                # Fallback: raw channel feature
                val = feat.get(f"{marker}_raw", 0.0)
            values.append(float(val) if val is not None else 0.0)
        result[marker] = np.array(values, dtype=np.float64)
    return result


# ===================================================================
# Plotting
# ===================================================================

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def plot_zonation_gradient(df_dict, markers, path_name, path_length, output_dir, n_bins=20):
    """Plot marker expression vs fractional position along a path.

    df_dict: {'frac_pos': array, 'marker1': array, ...}
    """
    frac = df_dict["frac_pos"]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(frac, bin_edges) - 1, 0, n_bins - 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for mi, marker in enumerate(markers):
        vals = df_dict[marker]
        # Normalize to [0, 1] for comparison across markers
        vmin, vmax = np.percentile(vals, [2, 98])
        if vmax - vmin < 1e-12:
            vmin, vmax = vals.min(), vals.max() + 1e-12
        vals_norm = np.clip((vals - vmin) / (vmax - vmin), 0, 1)

        means = np.zeros(n_bins)
        sems = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                means[b] = vals_norm[mask].mean()
                sems[b] = vals_norm[mask].std() / np.sqrt(mask.sum()) if mask.sum() > 1 else 0

        color = COLORS[mi % len(COLORS)]
        ax.plot(bin_centers, means, "-o", color=color, label=marker, markersize=4, linewidth=2)
        ax.fill_between(bin_centers, means - sems, means + sems, alpha=0.15, color=color)

    ax.set_xlabel("Fractional position (0 = Central Vein, 1 = Portal Vein)")
    ax.set_ylabel("Normalized intensity (2-98th percentile)")
    ax.set_title(f"{path_name} — zonation gradient ({path_length:.0f} um)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    out_path = Path(output_dir) / f"{path_name}_zonation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def plot_zonation_heatmap(all_data, markers, output_dir, n_bins=20):
    """Heatmap of marker expression across all paths aligned by fractional position."""
    if not all_data:
        return

    frac = np.concatenate([d["frac_pos"] for d in all_data])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(frac, bin_edges) - 1, 0, n_bins - 1)

    heatmap = np.zeros((len(markers), n_bins))
    for mi, marker in enumerate(markers):
        vals = np.concatenate([d[marker] for d in all_data])
        vmin, vmax = np.percentile(vals, [2, 98])
        if vmax - vmin < 1e-12:
            vmin, vmax = vals.min(), vals.max() + 1e-12
        vals_norm = np.clip((vals - vmin) / (vmax - vmin), 0, 1)
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                heatmap[mi, b] = vals_norm[mask].mean()

    fig, ax = plt.subplots(figsize=(12, max(3, len(markers) * 0.8)))
    im = ax.imshow(
        heatmap,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[0, 1, len(markers) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(markers)))
    ax.set_yticklabels(markers)
    ax.set_xlabel("Fractional position (0 = CV, 1 = PV)")
    ax.set_title("Zonation heatmap — all transects")
    plt.colorbar(im, ax=ax, label="Normalized intensity")

    out_path = Path(output_dir) / "zonation_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ===================================================================
# Main
# ===================================================================


def generate_auto_paths(cv_centroids, pv_centroids, max_path_length=2000, all_pairs=False):
    """Generate straight-line paths between CV-PV pairs.

    Parameters
    ----------
    cv_centroids : ndarray (K1, 2)
    pv_centroids : ndarray (K2, 2)
    max_path_length : float — skip paths longer than this (um)
    all_pairs : bool — if True, generate all CV-PV combinations within range
                       if False, greedy 1:1 nearest matching

    Returns
    -------
    paths : list of dicts with 'name', 'waypoints', etc.
    """
    if len(cv_centroids) == 0 or len(pv_centroids) == 0:
        return []

    paths = []

    if all_pairs:
        # All CV-PV combinations within max_path_length
        pv_tree = cKDTree(pv_centroids)
        for cv_idx in range(len(cv_centroids)):
            nearby = pv_tree.query_ball_point(cv_centroids[cv_idx], max_path_length)
            for pv_idx in nearby:
                d = float(np.linalg.norm(cv_centroids[cv_idx] - pv_centroids[pv_idx]))
                paths.append(
                    {
                        "id": f"auto_cv{cv_idx}_pv{pv_idx}",
                        "name": f"CV{cv_idx}_PV{pv_idx}",
                        "waypoints": np.array([cv_centroids[cv_idx], pv_centroids[pv_idx]]),
                        "corridor_um": 100.0,
                        "slide": "",
                        "category": "auto",
                        "cv_idx": int(cv_idx),
                        "pv_idx": int(pv_idx),
                        "length_um": d,
                    }
                )
    else:
        # Greedy 1:1 nearest matching
        pv_tree = cKDTree(pv_centroids)
        dists, pv_indices = pv_tree.query(cv_centroids, k=min(3, len(pv_centroids)))
        if dists.ndim == 1:
            dists = dists[:, None]
            pv_indices = pv_indices[:, None]

        order = np.argsort(dists[:, 0])
        used_pv = set()

        for cv_idx in order:
            for k in range(dists.shape[1]):
                pv_idx = int(pv_indices[cv_idx, k])
                d = float(dists[cv_idx, k])
                if pv_idx in used_pv or d > max_path_length:
                    continue
                used_pv.add(pv_idx)
                paths.append(
                    {
                        "id": f"auto_cv{cv_idx}_pv{pv_idx}",
                        "name": f"CV{cv_idx}_PV{pv_idx}",
                        "waypoints": np.array([cv_centroids[cv_idx], pv_centroids[pv_idx]]),
                        "corridor_um": 100.0,
                        "slide": "",
                        "category": "auto",
                        "cv_idx": int(cv_idx),
                        "pv_idx": int(pv_idx),
                        "length_um": d,
                    }
                )
                break

    logger.info(
        f"Generated {len(paths)} auto paths ({'all pairs' if all_pairs else 'nearest 1:1'}, "
        f"max {max_path_length} um)"
    )
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Hepatic zonation transect analysis",
        epilog="Auto mode: --auto --cv-marker GluI --pv-marker Pck1 (no --rois needed)",
    )
    parser.add_argument("--detections", required=True, help="Classified detections JSON")
    parser.add_argument("--rois", default=None, help="Exported ROIs JSON (manual path mode)")
    parser.add_argument("--markers", help="Comma-separated marker names (auto-discover if omitted)")
    parser.add_argument(
        "--corridor-um", type=float, default=100, help="Corridor width in um (default: 100)"
    )
    parser.add_argument("--n-bins", type=int, default=20, help="Number of bins for gradient plot")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--score-threshold", type=float, default=None, help="RF score filter")

    # Auto-landmark mode
    auto_group = parser.add_argument_group(
        "auto-landmark mode", "Detect vein landmarks from expression patterns"
    )
    auto_group.add_argument(
        "--auto", action="store_true", help="Auto-detect landmarks from marker expression"
    )
    auto_group.add_argument(
        "--cv-marker", default=None, help="Pericentral marker name (e.g., GluI)"
    )
    auto_group.add_argument("--pv-marker", default=None, help="Periportal marker name (e.g., Pck1)")
    auto_group.add_argument(
        "--cluster-radius",
        type=float,
        default=100,
        help="Radius for clustering positive cells (um, default: 100)",
    )
    auto_group.add_argument(
        "--min-cluster-cells", type=int, default=20, help="Min cells per vein cluster (default: 20)"
    )
    auto_group.add_argument(
        "--max-path-length",
        type=float,
        default=2000,
        help="Max CV-PV path length in um (default: 2000)",
    )
    auto_group.add_argument(
        "--all-pairs",
        action="store_true",
        help="Generate all CV-PV pairs within range (default: nearest 1:1)",
    )

    args = parser.parse_args()

    if not args.auto and not args.rois:
        parser.error("Either --rois (manual mode) or --auto (auto-landmark mode) is required")
    if args.auto and (not args.cv_marker or not args.pv_marker):
        parser.error("--auto requires --cv-marker and --pv-marker")

    setup_logging()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load detections
    det_path = Path(args.detections)
    logger.info(f"Loading detections from {det_path} ({det_path.stat().st_size / 1e9:.1f} GB)")
    det_data = fast_json_load(str(det_path))
    if isinstance(det_data, dict):
        detections = det_data.get("detections", det_data.get("cells", []))
    elif isinstance(det_data, list):
        detections = det_data
    else:
        detections = []
    logger.info(f"Loaded {len(detections):,} detections")

    # Score filter
    if args.score_threshold is not None:
        before = len(detections)
        detections = [
            d
            for d in detections
            if d.get("features", {}).get("rf_prediction", d.get("rf_prediction", 1.0))
            >= args.score_threshold
        ]
        logger.info(f"Score filter >= {args.score_threshold}: {before:,} -> {len(detections):,}")

    # Extract positions
    logger.info("Extracting positions...")
    positions = []
    valid_indices = []
    for i, det in enumerate(detections):
        pos = extract_position_um(det)
        if pos is not None:
            positions.append(pos)
            valid_indices.append(i)
    if not positions:
        logger.error("No valid positions found in detections")
        sys.exit(1)
    positions = np.array(positions, dtype=np.float64)
    logger.info(f"Valid positions: {len(positions):,} / {len(detections):,}")

    # Liver boundary filter (from ROIs if provided)
    liver_poly = None
    if args.rois:
        rois_data = load_rois(args.rois)
        liver_poly = extract_liver_polygon(rois_data)
    if liver_poly is not None:
        from matplotlib.path import Path as MplPath

        liver_path = MplPath(liver_poly)
        inside = liver_path.contains_points(positions)
        n_inside = inside.sum()
        logger.info(f"Liver filter: {n_inside:,} / {len(positions):,} cells inside boundary")
        positions = positions[inside]
        valid_indices = [valid_indices[i] for i in range(len(inside)) if inside[i]]

    # Discover markers
    if args.markers:
        markers = [m.strip() for m in args.markers.split(",")]
    else:
        markers = discover_markers(detections)
        if not markers:
            logger.error("No markers found. Specify --markers explicitly.")
            sys.exit(1)
    logger.info(f"Markers: {markers}")

    # --- Auto-landmark mode ---
    if args.auto:
        logger.info("=== Auto-landmark mode ===")

        # Build positive masks from valid_indices
        cv_positive = np.array(
            [
                detections[vi].get("features", {}).get(f"{args.cv_marker}_class") == "positive"
                for vi in valid_indices
            ],
            dtype=bool,
        )
        pv_positive = np.array(
            [
                detections[vi].get("features", {}).get(f"{args.pv_marker}_class") == "positive"
                for vi in valid_indices
            ],
            dtype=bool,
        )

        logger.info(f"{args.cv_marker}+ cells: {cv_positive.sum():,}")
        logger.info(f"{args.pv_marker}+ cells: {pv_positive.sum():,}")

        # Find clusters
        cv_centroids, cv_labels = find_marker_clusters(
            positions, cv_positive, args.min_cluster_cells, args.cluster_radius
        )
        pv_centroids, pv_labels = find_marker_clusters(
            positions, pv_positive, args.min_cluster_cells, args.cluster_radius
        )

        logger.info(f"CV clusters: {len(cv_centroids)}")
        logger.info(f"PV clusters: {len(pv_centroids)}")

        if len(cv_centroids) == 0 or len(pv_centroids) == 0:
            logger.error(
                "Need at least 1 CV and 1 PV cluster. "
                "Try lowering --min-cluster-cells or --cluster-radius."
            )
            sys.exit(1)

        # Compute global zonation score for every cell
        scores, d_cv, d_pv = compute_zonation_score(positions, cv_centroids, pv_centroids)
        logger.info(
            f"Zonation scores: mean={scores.mean():.3f}, " f"median={np.median(scores):.3f}"
        )

        # Save per-cell zonation scores to CSV
        logger.info("Saving per-cell zonation scores...")
        score_csv = out_dir / "zonation_scores.csv"
        with open(score_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["cell_uid", "zonation_score", "d_cv_um", "d_pv_um"]
            for m in markers:
                header.append(f"{m}_value")
            writer.writerow(header)
            for j, vi in enumerate(valid_indices):
                feat = detections[vi].get("features", {})
                row = [
                    detections[vi].get("uid", f"cell_{vi}"),
                    f"{scores[j]:.4f}",
                    f"{d_cv[j]:.1f}",
                    f"{d_pv[j]:.1f}",
                ]
                for m in markers:
                    row.append(f'{feat.get(f"{m}_value", 0.0):.4f}')
                writer.writerow(row)
        logger.info(f"Saved: {score_csv} ({len(valid_indices):,} cells)")

        # Global gradient plot (all cells, binned by zonation score)
        all_marker_vals = {}
        for m in markers:
            all_marker_vals[m] = np.array(
                [detections[vi].get("features", {}).get(f"{m}_value", 0.0) for vi in valid_indices],
                dtype=np.float64,
            )
        global_data = {"frac_pos": scores}
        global_data.update(all_marker_vals)
        plot_zonation_gradient(global_data, markers, "all_cells_auto", 0, out_dir, args.n_bins)

        # Generate auto paths between nearest CV-PV pairs
        paths = generate_auto_paths(
            cv_centroids, pv_centroids, args.max_path_length, all_pairs=args.all_pairs
        )

        # Spatial zonation map (with paths overlaid)
        plot_zonation_map(
            positions,
            scores,
            cv_centroids,
            pv_centroids,
            out_dir,
            title=f"Auto zonation: {args.cv_marker}(CV) vs {args.pv_marker}(PV)",
            paths=paths,
            corridor_um=args.corridor_um,
        )

        # Discrete zone map with labeled veins and routes
        plot_zone_map(positions, scores, cv_centroids, pv_centroids, out_dir, paths=paths)

        # Save landmarks JSON
        landmarks = {
            "cv_centroids": cv_centroids.tolist(),
            "pv_centroids": pv_centroids.tolist(),
            "cv_marker": args.cv_marker,
            "pv_marker": args.pv_marker,
            "n_cv_clusters": len(cv_centroids),
            "n_pv_clusters": len(pv_centroids),
            "auto_paths": [
                {
                    "name": p["name"],
                    "cv": p["waypoints"][0].tolist(),
                    "pv": p["waypoints"][1].tolist(),
                    "length_um": p["length_um"],
                }
                for p in paths
            ],
        }
        atomic_json_dump(landmarks, out_dir / "zonation_landmarks.json")
        logger.info(f"Saved landmarks: {out_dir / 'zonation_landmarks.json'}")

    # --- Manual path mode ---
    else:
        rois_data = load_rois(args.rois)
        paths = extract_paths(rois_data)
        if not paths:
            logger.error("No path ROIs found in ROI file")
            sys.exit(1)
        logger.info(f"Found {len(paths)} manual path(s)")
        scores = None  # no global zonation in manual mode

    # Process each path (shared between auto and manual modes)
    all_path_data = []
    all_rows = []
    summary = []

    for path_info in paths:
        corridor_um = (
            args.corridor_um if args.corridor_um is not None else path_info.get("corridor_um", 100)
        )
        corridor_half = corridor_um / 2.0
        path_length = float(np.sum(np.linalg.norm(np.diff(path_info["waypoints"], axis=0), axis=1)))

        logger.info(
            f"Path '{path_info['name']}': {len(path_info['waypoints'])} waypoints, "
            f"length={path_length:.0f} um, corridor={corridor_um:.0f} um"
        )

        in_corridor, frac_pos = project_cells_onto_path(
            positions, path_info["waypoints"], corridor_half
        )

        n_in = in_corridor.sum()
        logger.info(f"  {n_in:,} cells in corridor")

        if n_in == 0:
            logger.warning("  No cells in corridor — skipping")
            continue

        # Get indices of cells in corridor
        corridor_global_indices = [
            valid_indices[i] for i in range(len(in_corridor)) if in_corridor[i]
        ]
        corridor_frac = frac_pos[in_corridor]

        # Extract marker values
        marker_vals = extract_marker_values(detections, corridor_global_indices, markers)

        # Build per-path data dict
        path_data = {"frac_pos": corridor_frac}
        path_data.update(marker_vals)
        all_path_data.append(path_data)

        # Build CSV rows
        for j, gi in enumerate(corridor_global_indices):
            row = {
                "cell_uid": detections[gi].get("uid", f"cell_{gi}"),
                "path_id": path_info["id"],
                "path_name": path_info["name"],
                "fractional_pos": float(corridor_frac[j]),
            }
            for marker in markers:
                row[marker] = float(marker_vals[marker][j])
            all_rows.append(row)

        # Per-path gradient plot
        plot_zonation_gradient(
            path_data, markers, path_info["name"], path_length, out_dir, args.n_bins
        )

        # Summary stats
        path_summary = {
            "path_id": path_info["id"],
            "path_name": path_info["name"],
            "n_cells": int(n_in),
            "path_length_um": round(path_length, 1),
            "corridor_um": corridor_um,
            "n_waypoints": len(path_info["waypoints"]),
        }
        for marker in markers:
            vals = marker_vals[marker]
            rho, pval = stats.spearmanr(corridor_frac, vals)
            path_summary[f"{marker}_spearman_rho"] = (
                round(float(rho), 4) if np.isfinite(rho) else None
            )
            path_summary[f"{marker}_spearman_pval"] = (
                float(f"{pval:.2e}") if np.isfinite(pval) else None
            )
        summary.append(path_summary)

    # Aggregate heatmap
    if all_path_data:
        plot_zonation_heatmap(all_path_data, markers, out_dir, args.n_bins)

    # Save transect CSV
    if all_rows:
        csv_path = out_dir / "zonation_transect.csv"
        fieldnames = ["cell_uid", "path_id", "path_name", "fractional_pos"] + markers
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"Saved CSV: {csv_path} ({len(all_rows):,} rows)")

    # Save summary JSON
    summary_data = {"paths": summary, "markers": markers, "n_bins": args.n_bins}
    if args.auto:
        summary_data["mode"] = "auto"
        summary_data["cv_marker"] = args.cv_marker
        summary_data["pv_marker"] = args.pv_marker
    atomic_json_dump(summary_data, out_dir / "zonation_summary.json")
    logger.info(f"Saved summary: {out_dir / 'zonation_summary.json'}")

    # Print results
    logger.info("=== Zonation Transect Summary ===")
    if args.auto:
        logger.info(f"  Auto mode: {len(cv_centroids)} CV, {len(pv_centroids)} PV clusters")
        logger.info(f"  Global zonation: mean={scores.mean():.3f}, median={np.median(scores):.3f}")
    for s in summary:
        logger.info(f"  {s['path_name']}: {s['n_cells']:,} cells, {s['path_length_um']:.0f} um")
        for marker in markers:
            rho = s.get(f"{marker}_spearman_rho")
            if rho is not None:
                logger.info(f"    {marker}: Spearman rho = {rho:.3f}")


if __name__ == "__main__":
    main()
