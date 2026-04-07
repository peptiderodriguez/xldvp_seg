"""Vessel characterization — shared analysis for vessel detection approaches.

Used by both:
- scripts/detect_vessel_structures.py (cell-topology approach)
- scripts/segment_vessel_lumens.py (lumen-first approach)

Functions handle arbitrary lumen/vessel shapes (not necessarily circular/oval).
Oblique cuts, compressed vessels, and irregular lumens are all valid.
"""

from __future__ import annotations

import numpy as np

from xldvp_seg.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "analyze_marker_composition",
    "assign_vessel_type",
    "compute_lumen_morphometry",
    "compute_wall_morphometry",
    "detect_spatial_layering",
    "detect_spatial_layering_from_boundary",
    "tag_detections",
]


# ---------------------------------------------------------------------------
# Marker composition
# ---------------------------------------------------------------------------


def analyze_marker_composition(
    detections: list[dict],
    comp_global_indices: list[int],
    marker_names: list[str],
    class_keys: list[str],
) -> dict:
    """Count marker-positive cells per marker in a component.

    Tracks single-positive and double-positive cells separately.
    Double-positive cells (positive for 2+ markers) are counted in their
    own class, NOT in individual marker counts — avoids double-counting
    the same physical material.

    Args:
        detections: Full detections list.
        comp_global_indices: Indices into detections for this component.
        marker_names: Marker names (e.g., ["SMA", "CD31"]).
        class_keys: Corresponding feature keys (e.g., ["SMA_class", "CD31_class"]).

    Returns:
        Dict with n_cells, per-marker counts/fracs, double_pos stats, dominant_marker.
    """
    n = len(comp_global_indices)

    single_counts = dict.fromkeys(marker_names, 0)
    total_counts = dict.fromkeys(marker_names, 0)  # single + relevant double-positive
    double_pos_count = 0

    for idx in comp_global_indices:
        feat = detections[idx].get("features", {})
        pos_markers = [
            name for name, key in zip(marker_names, class_keys) if feat.get(key) == "positive"
        ]
        # Track per-marker totals (includes double-positive cells for THEIR markers)
        for m in pos_markers:
            total_counts[m] += 1
        if len(pos_markers) >= 2:
            double_pos_count += 1
        elif len(pos_markers) == 1:
            single_counts[pos_markers[0]] += 1

    composition = {"n_cells": n, "n_double_pos": double_pos_count}
    for name in marker_names:
        composition[f"n_{name.lower()}"] = single_counts[name]
        composition[f"{name.lower()}_frac"] = round(single_counts[name] / max(n, 1), 3)
        # Total fraction: single-positive + double-positive cells expressing THIS marker.
        # Used by assign_vessel_type for typing (avoids the bug where dp_frac was added
        # to all markers regardless of which markers the double-positive cells express).
        composition[f"{name.lower()}_total_frac"] = round(total_counts[name] / max(n, 1), 3)
    composition["double_pos_frac"] = round(double_pos_count / max(n, 1), 3)

    if marker_names:
        dominant = max(marker_names, key=lambda m: single_counts[m])
        composition["dominant_marker"] = dominant if single_counts[dominant] > 0 else "none"

    return composition


# ---------------------------------------------------------------------------
# Spatial layering
# ---------------------------------------------------------------------------


def detect_spatial_layering(
    positions: np.ndarray,
    detections: list[dict],
    comp_local_indices: list[int],
    comp_global_indices: list[int],
    class_keys_map: dict[str, str],
) -> dict:
    """Detect radial layering of markers (e.g., SMA outer vs CD31 inner).

    Uses Mann-Whitney U test for statistical rigor. Tests both directions
    for each marker pair and reports the significant direction.

    Args:
        positions: (N, 2) array of positive-cell positions in µm.
        detections: Full detections list.
        comp_local_indices: Indices into positions array.
        comp_global_indices: Indices into detections list.
        class_keys_map: {marker_name: feature_key} mapping.

    Returns:
        Dict of layering results keyed by "{outer}_outer_vs_{inner}".
    """
    from scipy.stats import mannwhitneyu

    pts = positions[comp_local_indices]
    centroid = pts.mean(axis=0)
    dists = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))

    layering = {}

    marker_radii = {}
    for marker_name, class_key in class_keys_map.items():
        radii = []
        for enum_i, global_i in enumerate(comp_global_indices):
            feat = detections[global_i].get("features", {})
            if feat.get(class_key) == "positive":
                radii.append(dists[enum_i])
        if radii:
            marker_radii[marker_name] = np.array(radii)

    markers = list(marker_radii.keys())
    for i, m1 in enumerate(markers):
        for m2 in markers[i + 1 :]:
            r1 = marker_radii[m1]
            r2 = marker_radii[m2]
            if len(r1) >= 5 and len(r2) >= 5:
                _, pval_fwd = mannwhitneyu(r1, r2, alternative="greater")
                _, pval_rev = mannwhitneyu(r2, r1, alternative="greater")

                if pval_rev < pval_fwd:
                    key = f"{m2}_outer_vs_{m1}"
                    score = float(np.median(r2) - np.median(r1))
                    pval = pval_rev
                else:
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
# Spatial layering from lumen boundary (for lumen-first approach)
# ---------------------------------------------------------------------------


def detect_spatial_layering_from_boundary(
    detections: list[dict],
    assigned_cell_indices: list[int],
    cell_distances_um: list[float],
    class_keys_map: dict[str, str],
) -> dict:
    """Detect radial layering using distance from lumen boundary instead of centroid.

    This variant is used by the lumen-first approach where we have actual
    lumen boundaries. Inner cells are closer to the lumen, outer cells are farther.

    Args:
        detections: Full detections list.
        assigned_cell_indices: Global indices of cells assigned to this vessel.
        cell_distances_um: Distance of each cell to the lumen boundary (µm).
        class_keys_map: {marker_name: feature_key} mapping.

    Returns:
        Dict of layering results keyed by "{outer}_outer_vs_{inner}".
    """
    from scipy.stats import mannwhitneyu

    dists = np.array(cell_distances_um)

    marker_dists: dict[str, np.ndarray] = {}
    for marker_name, class_key in class_keys_map.items():
        marker_d = []
        for i, global_i in enumerate(assigned_cell_indices):
            feat = detections[global_i].get("features", {})
            if feat.get(class_key) == "positive":
                marker_d.append(dists[i])
        if marker_d:
            marker_dists[marker_name] = np.array(marker_d)

    layering = {}
    markers = list(marker_dists.keys())
    for i, m1 in enumerate(markers):
        for m2 in markers[i + 1 :]:
            d1 = marker_dists[m1]
            d2 = marker_dists[m2]
            if len(d1) >= 5 and len(d2) >= 5:
                # "outer" = farther from lumen boundary
                _, pval_fwd = mannwhitneyu(d1, d2, alternative="greater")
                _, pval_rev = mannwhitneyu(d2, d1, alternative="greater")

                if pval_rev < pval_fwd:
                    key = f"{m2}_outer_vs_{m1}"
                    score = float(np.median(d2) - np.median(d1))
                    pval = pval_rev
                else:
                    key = f"{m1}_outer_vs_{m2}"
                    score = float(np.median(d1) - np.median(d2))
                    pval = pval_fwd

                layering[key] = {
                    "score": round(score, 2),
                    "p_value": round(float(pval), 4),
                    "significant": pval < 0.05,
                    f"n_{m1.lower()}": len(d1),
                    f"n_{m2.lower()}": len(d2),
                }

    return layering


# ---------------------------------------------------------------------------
# Vessel type assignment
# ---------------------------------------------------------------------------


def assign_vessel_type(
    morphology: str,
    composition: dict,
    layering: dict,
    morphometry: dict,
) -> str:
    """Assign vessel type from morphology + marker composition + wall morphometry.

    Decision tree:
      1. LYVE1+ dominant → lymphatic / collecting_lymphatic (has SMA)
      2. Ring/arc with thick wall (layers > 1.5 or wall/diameter > 0.3) → artery/arteriole
      3. Ring/arc with CD31-dominant thin wall → vein/venule
      4. Strip → longitudinal section (artery/vein/lymphatic by dominant marker)
      5. Cluster with CD31 > 50% and <15 cells → capillary

    Size subtyping: artery (>100µm) vs arteriole (≤100µm),
    vein (>50µm) vs venule (≤50µm).

    Args:
        morphology: "ring", "arc", "strip", or "cluster".
        composition: From analyze_marker_composition().
        layering: From detect_spatial_layering().
        morphometry: Dict with vessel_diameter_um, wall_cell_layers, etc.

    Returns:
        Vessel type string.
    """
    # Use per-marker total fracs (single + double-positive cells expressing THAT marker).
    # Falls back to single_frac if total_frac not available (old composition format).
    sma_total_frac = composition.get("sma_total_frac", composition.get("sma_frac", 0))
    cd31_total_frac = composition.get("cd31_total_frac", composition.get("cd31_frac", 0))
    lyve1_total_frac = composition.get("lyve1_total_frac", composition.get("lyve1_frac", 0))
    diameter = morphometry.get("vessel_diameter_um", 0) or 0

    # LYVE1+ vessels
    if lyve1_total_frac > 0.3 and lyve1_total_frac > sma_total_frac and morphology != "strip":
        if sma_total_frac > 0.15:
            return "collecting_lymphatic"
        return "lymphatic"

    # Check layering for SMA outer (artery signature)
    has_sma_outer = False
    if layering:
        for key, info in layering.items():
            if "SMA_outer" in key and info.get("significant"):
                has_sma_outer = True

    if morphology in ("ring", "arc"):
        wall_extent = morphometry.get("wall_extent_um", 0) or 0
        wall_layers = morphometry.get("wall_cell_layers", 0) or 0
        wall_ratio = wall_extent / diameter if diameter > 0 else 0

        if sma_total_frac > 0.15 or cd31_total_frac > 0.15:
            has_morphometry = "wall_cell_layers" in morphometry
            is_thick_wall = wall_layers > 1.5 or wall_ratio > 0.3

            if is_thick_wall or (has_sma_outer and sma_total_frac > 0.3):
                if diameter > 100:
                    return "artery"
                elif diameter > 0:
                    return "arteriole"
                else:
                    return "artery"
            elif not has_morphometry and sma_total_frac > cd31_total_frac:
                return "artery"
            elif cd31_total_frac > sma_total_frac:
                if diameter > 50:
                    return "vein"
                elif diameter > 0:
                    return "venule"
                else:
                    return "vein"
            elif sma_total_frac > cd31_total_frac:
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
        # Check LYVE1 first (same priority as non-strip branch above)
        if lyve1_total_frac > 0.3 and lyve1_total_frac > sma_total_frac:
            if sma_total_frac > 0.15:
                return "collecting_lymphatic_longitudinal"
            return "lymphatic_longitudinal"
        elif sma_total_frac > 0.3:
            return "artery_longitudinal"
        elif cd31_total_frac > 0.3:
            return "vein_longitudinal"
        elif lyve1_total_frac > 0.1:
            return "lymphatic_longitudinal"
        else:
            return "unclassified"
    elif morphology == "cluster":
        if cd31_total_frac > 0.5 and composition.get("n_cells", 0) < 15:
            return "capillary"
        return "unclassified"

    return "unclassified"


# ---------------------------------------------------------------------------
# Tag detections
# ---------------------------------------------------------------------------


def tag_detections(
    detections: list[dict],
    comp_assignments: dict[int, dict],
    marker_names: list[str],
    class_keys: list[str],
    prefix: str = "vessel",
) -> str:
    """Tag each detection with vessel structure membership + cell type within vessel.

    Args:
        detections: Full detections list (modified in-place).
        comp_assignments: {global_idx: {vessel_id, vessel_type, morphology, size_class}}.
        marker_names: Marker names for cell-type tagging.
        class_keys: Corresponding feature keys.
        prefix: Field name prefix (default: "vessel").

    Returns:
        The vessel_type field name (for downstream use).
    """
    field_id = f"{prefix}_id"
    field_type = f"{prefix}_type"
    field_morph = f"{prefix}_morphology"
    field_size = f"{prefix}_size_class"
    field_cell_type = f"{prefix}_cell_type"

    for d in detections:
        feat = d.setdefault("features", {})
        feat[field_id] = -1
        feat[field_type] = "none"
        feat[field_morph] = "none"
        feat[field_size] = "none"
        feat[field_cell_type] = "none"

    for global_idx, assignment in comp_assignments.items():
        feat = detections[global_idx].setdefault("features", {})
        feat[field_id] = assignment["vessel_id"]
        feat[field_type] = assignment["vessel_type"]
        feat[field_morph] = assignment["morphology"]
        feat[field_size] = assignment.get("size_class", "unknown")

        pos = [n for n, k in zip(marker_names, class_keys) if feat.get(k) == "positive"]
        if len(pos) >= 2:
            feat[field_cell_type] = "double_pos"
        elif len(pos) == 1:
            feat[field_cell_type] = f"{pos[0]}_only"
        else:
            feat[field_cell_type] = "negative"

    type_counts: dict[str, int] = {}
    for d in detections:
        vt = d.get("features", {}).get(field_type, "none")
        type_counts[vt] = type_counts.get(vt, 0) + 1
    logger.info("Vessel type distribution: %s", type_counts)

    return field_type


# ---------------------------------------------------------------------------
# Lumen morphometry (new — for lumen-first approach)
# ---------------------------------------------------------------------------


def compute_lumen_morphometry(contour_points: np.ndarray) -> dict:
    """Compute morphometry of a lumen contour.

    Handles arbitrary shapes — no circularity requirement.
    Oblique cuts, compressed vessels, and irregular lumens are all valid.

    Args:
        contour_points: (N, 2) array of contour points in µm, as (x, y).

    Returns:
        Dict with lumen_area_um2, lumen_perimeter_um, lumen_equiv_diameter_um,
        lumen_elongation, lumen_convexity, lumen_circularity.
    """
    import cv2

    pts = np.asarray(contour_points, dtype=np.float64)
    n = len(pts)

    if n < 3:
        return {
            "lumen_area_um2": 0.0,
            "lumen_perimeter_um": 0.0,
            "lumen_equiv_diameter_um": 0.0,
            "lumen_elongation": 1.0,
            "lumen_convexity": 1.0,
            "lumen_circularity": 0.0,
        }

    # Use shoelace formula for area (works in µm coordinates directly)
    # cv2 functions need int32 pixel coords, so use manual computation for µm
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - x[0] * y[-1]))

    # Perimeter: sum of segment lengths
    diffs = np.diff(pts, axis=0, append=pts[:1])
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    perimeter = float(np.sum(segment_lengths))

    equiv_diameter = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0

    # Elongation via cv2.fitEllipse (robust to non-uniform point density,
    # unlike PCA on boundary points which is biased by sampling density)
    if n >= 5:
        pts_cv = pts.astype(np.float32).reshape(-1, 1, 2)
        try:
            (_, _), (w, h), _ = cv2.fitEllipse(pts_cv)
            elongation = float(max(w, h) / max(min(w, h), 1e-6))
        except cv2.error:
            elongation = 1.0
    else:
        elongation = 1.0

    # Convexity: area / convex_hull_area
    pts_cv_hull = pts.astype(np.float32)
    hull = cv2.convexHull(pts_cv_hull)
    hull_area = float(cv2.contourArea(hull))
    convexity = area / hull_area if hull_area > 0 else 1.0
    convexity = min(convexity, 1.0)

    # Circularity: 4*pi*area / perimeter^2 (descriptive, NOT a filter)
    circularity = (4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    circularity = min(max(circularity, 0.0), 1.0)

    return {
        "lumen_area_um2": round(area, 1),
        "lumen_perimeter_um": round(perimeter, 1),
        "lumen_equiv_diameter_um": round(equiv_diameter, 1),
        "lumen_elongation": round(elongation, 2),
        "lumen_convexity": round(convexity, 3),
        "lumen_circularity": round(circularity, 3),
    }


# ---------------------------------------------------------------------------
# Wall morphometry (new — for lumen-first approach)
# ---------------------------------------------------------------------------


def compute_wall_morphometry(
    lumen_contour_um: np.ndarray,
    cell_positions_um: np.ndarray,
    cell_areas_um2: np.ndarray | None = None,
) -> dict:
    """Compute wall morphometry from cell distances to lumen boundary.

    For each assigned cell, computes minimum distance to the lumen contour.
    Wall thickness is derived from the distribution of these distances.

    Args:
        lumen_contour_um: (M, 2) array of lumen contour points in µm, as (x, y).
        cell_positions_um: (K, 2) array of assigned cell positions in µm.
        cell_areas_um2: Optional (K,) array of cell areas for cell-layer estimation.

    Returns:
        Dict with wall_thickness_um, wall_thickness_p5_um, wall_thickness_p95_um,
        wall_uniformity, wall_cell_layers (if areas provided), n_wall_cells.
    """
    contour = np.asarray(lumen_contour_um, dtype=np.float64)
    cells = np.asarray(cell_positions_um, dtype=np.float64)
    n_cells = len(cells)

    if n_cells == 0 or len(contour) < 3:
        return {
            "wall_thickness_um": 0.0,
            "wall_thickness_p5_um": 0.0,
            "wall_thickness_p95_um": 0.0,
            "wall_uniformity": 0.0,
            "n_wall_cells": 0,
        }

    # Compute minimum distance from each cell to any contour segment.
    # Vectorized: for each cell, find closest point on each contour segment,
    # then take the minimum across all segments.
    # Contour segments: from contour[i] to contour[(i+1) % M]
    seg_starts = contour
    seg_ends = np.roll(contour, -1, axis=0)
    seg_vecs = seg_ends - seg_starts
    seg_lens_sq = np.sum(seg_vecs**2, axis=1)

    distances = np.full(n_cells, np.inf)

    # Process in chunks to limit memory (n_cells × n_seg can be large)
    chunk_size = max(1, min(n_cells, 10000))
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        cell_chunk = cells[start:end]  # (chunk, 2)

        # Vector from segment start to each cell: (chunk, n_seg, 2)
        to_cell = cell_chunk[:, np.newaxis, :] - seg_starts[np.newaxis, :, :]

        # Project onto segment: t = dot(to_cell, seg_vec) / |seg_vec|^2
        # Clamp t to [0, 1] to stay on segment
        dots = np.sum(to_cell * seg_vecs[np.newaxis, :, :], axis=2)  # (chunk, n_seg)
        safe_lens_sq = np.maximum(seg_lens_sq[np.newaxis, :], 1e-12)
        t = np.clip(dots / safe_lens_sq, 0.0, 1.0)  # (chunk, n_seg)

        # Closest point on each segment
        closest = seg_starts[np.newaxis, :, :] + t[:, :, np.newaxis] * seg_vecs[np.newaxis, :, :]
        # Distance from cell to closest point
        dists_sq = np.sum((cell_chunk[:, np.newaxis, :] - closest) ** 2, axis=2)  # (chunk, n_seg)
        min_dists = np.sqrt(np.min(dists_sq, axis=1))  # (chunk,)
        distances[start:end] = min_dists

    wall_thickness = float(np.median(distances))
    p5 = float(np.percentile(distances, 5))
    p95 = float(np.percentile(distances, 95))
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    uniformity = max(0.0, 1.0 - std_dist / max(mean_dist, 1e-6))

    result = {
        "wall_thickness_um": round(wall_thickness, 1),
        "wall_thickness_p5_um": round(p5, 1),
        "wall_thickness_p95_um": round(p95, 1),
        "wall_uniformity": round(uniformity, 3),
        "n_wall_cells": n_cells,
    }

    if cell_areas_um2 is not None:
        areas = np.asarray(cell_areas_um2, dtype=np.float64)
        valid_areas = areas[areas > 0]
        if len(valid_areas) > 0:
            mean_cell_diam = float(2.0 * np.mean(np.sqrt(valid_areas / np.pi)))
            result["wall_cell_layers"] = round(wall_thickness / max(mean_cell_diam, 1e-6), 1)

    return result
