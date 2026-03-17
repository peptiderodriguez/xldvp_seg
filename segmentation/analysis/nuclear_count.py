"""Count nuclei per cell using Cellpose on the nuclear channel.

Runs Cellpose (cpsam model, single-channel mode) on the nuclear stain to segment
individual nuclei, then spatial-joins nuclear centroids to cell masks to count
how many nuclei fall inside each cell.

Features produced per cell:
    n_nuclei             — number of nuclear objects inside cell mask
    nuclear_area_um2     — total nuclear area in µm²
    nuclear_area_fraction — nuclear area / cell area (N:C ratio by area)
    largest_nucleus_um2  — area of the largest nucleus
    nuclear_solidity     — mean solidity of nuclear objects (1.0 = convex)
    nuclear_eccentricity — mean eccentricity of nuclear objects
    nuclei               — list of per-nucleus feature dicts:
        [{"area_um2", "solidity", "eccentricity", "perimeter_um",
          "major_axis_um", "minor_axis_um", "mean_intensity",
          "centroid_local": [x, y]}, ...]

Usage (as library):
    from segmentation.analysis.nuclear_count import count_nuclei_in_cells

    results = count_nuclei_in_cells(cell_masks, nuc_tile, cellpose_model, pixel_size)
    # results = {cell_label: {n_nuclei: 1, nuclei: [{area_um2: 45.2, ...}], ...}, ...}
"""

import numpy as np
from skimage.measure import regionprops

from segmentation.utils.logging import get_logger
from segmentation.utils.detection_utils import safe_to_uint8  # noqa: F401 — kept as fallback

logger = get_logger(__name__)


def _percentile_normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Percentile-normalize any numeric array to uint8 [0, 255].

    Uses 1st-99.5th percentile on nonzero pixels for proper dynamic range.
    Much better than safe_to_uint8() which does arr/256 for uint16 (very dim).
    """
    if arr.dtype == np.uint8:
        return arr
    nonzero = arr[arr > 0]
    if len(nonzero) == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    p_low, p_high = np.percentile(nonzero, [1, 99.5])
    if p_high <= p_low:
        return np.zeros_like(arr, dtype=np.uint8)
    result = np.clip((arr.astype(np.float32) - p_low) / (p_high - p_low) * 255, 0, 255)
    return result.astype(np.uint8)


def count_nuclei_in_cells(
    cell_masks: np.ndarray,
    nuclear_channel: np.ndarray,
    cellpose_model,
    pixel_size_um: float,
    min_nuclear_area_px: int = 50,
) -> dict:
    """Count nuclei per cell by running Cellpose on the nuclear channel.

    Args:
        cell_masks: 2D integer label array where each cell has a unique ID.
                    Background = 0.
        nuclear_channel: 2D uint16/uint8 array of the nuclear stain (same shape
                         as cell_masks). Will be normalized to uint8 for Cellpose.
        cellpose_model: Loaded CellposeModel (cpsam). Called with
                        model.eval(tile, channels=[0, 0]) for single-channel mode.
        pixel_size_um: Pixel size in micrometers.
        min_nuclear_area_px: Minimum nuclear object area in pixels to count.
                             Filters small debris/noise.

    Returns:
        Dict mapping cell label (int) to a feature dict:
        {cell_label: {n_nuclei, nuclear_area_um2, nuclear_area_fraction,
                      largest_nucleus_um2, nuclear_solidity, nuclear_eccentricity}}

        Cells with no nuclei detected get n_nuclei=0 and zero for all metrics.
    """
    if cell_masks.shape != nuclear_channel.shape:
        raise ValueError(
            f"Shape mismatch: cell_masks {cell_masks.shape} vs "
            f"nuclear_channel {nuclear_channel.shape}"
        )

    px2 = pixel_size_um ** 2

    # --- Step 1: Segment nuclei with Cellpose single-channel mode ---
    nuc_uint8 = _percentile_normalize_to_uint8(nuclear_channel)
    nuclear_masks, _, _ = cellpose_model.eval(nuc_uint8, channels=[0, 0])

    # --- Step 2: Compute nuclear properties (with intensity for mean_intensity) ---
    nuc_props = regionprops(nuclear_masks, intensity_image=nuclear_channel)

    # Filter by minimum area
    nuc_props = [p for p in nuc_props if p.area >= min_nuclear_area_px]

    # --- Step 3: Assign each nucleus to a cell via centroid lookup ---
    # For each nuclear centroid, check which cell label it falls inside
    cell_labels_in_image = set(np.unique(cell_masks)) - {0}

    # Initialize results for all cells
    results = {}
    for cl in cell_labels_in_image:
        results[cl] = {
            "n_nuclei": 0,
            "nuclear_area_um2": 0.0,
            "nuclear_area_fraction": 0.0,
            "largest_nucleus_um2": 0.0,
            "nuclear_solidity": 0.0,
            "nuclear_eccentricity": 0.0,
            "nuclei": [],  # per-nucleus feature dicts
        }

    for nuc_prop in nuc_props:
        # Centroid is (row, col) = (y, x)
        cy, cx = nuc_prop.centroid
        iy, ix = int(round(cy)), int(round(cx))

        # Bounds check
        if iy < 0 or iy >= cell_masks.shape[0] or ix < 0 or ix >= cell_masks.shape[1]:
            continue

        cell_label = cell_masks[iy, ix]
        if cell_label == 0:
            continue  # nucleus not inside any cell

        if cell_label not in results:
            continue

        # Per-nucleus features
        nuc_feat = {
            "area_um2": round(nuc_prop.area * px2, 3),
            "perimeter_um": round(nuc_prop.perimeter * pixel_size_um, 3),
            "solidity": round(float(nuc_prop.solidity), 4),
            "eccentricity": round(float(nuc_prop.eccentricity), 4),
            "major_axis_um": round(nuc_prop.major_axis_length * pixel_size_um, 3),
            "minor_axis_um": round(nuc_prop.minor_axis_length * pixel_size_um, 3),
            "mean_intensity": round(float(nuc_prop.mean_intensity), 2),
            "centroid_local": [round(float(cx), 1), round(float(cy), 1)],
        }

        results[cell_label]["nuclei"].append(nuc_feat)
        results[cell_label]["n_nuclei"] += 1

    # --- Step 4: Aggregate per-cell summary metrics ---
    cell_props = {p.label: p.area for p in regionprops(cell_masks)}

    for cl, r in results.items():
        nuclei = r["nuclei"]
        if nuclei:
            areas = [n["area_um2"] for n in nuclei]
            r["nuclear_area_um2"] = round(sum(areas), 3)
            r["largest_nucleus_um2"] = round(max(areas), 3)
            r["nuclear_solidity"] = round(
                float(np.mean([n["solidity"] for n in nuclei])), 4
            )
            r["nuclear_eccentricity"] = round(
                float(np.mean([n["eccentricity"] for n in nuclei])), 4
            )

            cell_area_um2 = cell_props.get(cl, 1) * px2
            if cell_area_um2 > 0:
                r["nuclear_area_fraction"] = round(
                    r["nuclear_area_um2"] / cell_area_um2, 4
                )

    return results


def count_nuclei_for_tile(
    tile_cell_masks: np.ndarray,
    tile_nuclear_channel: np.ndarray,
    cellpose_model,
    pixel_size_um: float,
    min_nuclear_area_px: int = 50,
    tile_x: int = 0,
    tile_y: int = 0,
) -> tuple:
    """Convenience wrapper that returns (nuclear_results, n_nuclei_segmented).

    Args:
        tile_cell_masks: 2D label array for this tile.
        tile_nuclear_channel: 2D nuclear stain array for this tile.
        cellpose_model: Loaded CellposeModel (cpsam).
        pixel_size_um: Pixel size in micrometers.
        min_nuclear_area_px: Min nuclear area to count.
        tile_x, tile_y: Tile origin (for logging only).

    Returns:
        (results_dict, n_total_nuclei) where results_dict maps
        cell_label -> feature dict.
    """
    results = count_nuclei_in_cells(
        tile_cell_masks, tile_nuclear_channel, cellpose_model,
        pixel_size_um, min_nuclear_area_px,
    )

    n_cells = len(results)
    n_with_nuclei = sum(1 for r in results.values() if r["n_nuclei"] > 0)
    n_multi = sum(1 for r in results.values() if r["n_nuclei"] > 1)
    n_total_nuc = sum(r["n_nuclei"] for r in results.values())

    logger.debug(
        f"Tile ({tile_x}, {tile_y}): {n_cells} cells, "
        f"{n_total_nuc} nuclei segmented, "
        f"{n_with_nuclei} cells with ≥1 nuc, {n_multi} with ≥2"
    )

    return results, n_total_nuc
